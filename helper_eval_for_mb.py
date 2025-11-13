import jax
import jax.experimental
import wandb
import jax.numpy as jnp
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from functools import partial
import os
import csv

def instance_norm_nhwc(x, eps=1e-5):
    # x: [B, H, W, C]
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)                     # [B,1,1,C]
    var = jnp.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)        # [B,1,1,C]
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return x_norm

def stream_mbvar_and_csv(
    FLAGS,
    train_state,
    shard_data,
    vae_decode,
    call_model,
    # ví dụ: batch_images.shape (latent nếu đang dùng VAE)
    batch_shape,
    step,                  # global step (để log/ghi CSV)
    # vd: os.path.join(FLAGS.save_dir or '.', 'mbvar_eval.csv')
    csv_path,
    T_list=(1, 4, 32, 128),   # các T quan tâm
    # mục tiêu ~số mẫu (sẽ làm tròn lên bội của GLOBAL_B)
    total_samples=1024,
    decode_to_pixel=False,  # True nếu muốn đo trên pixel-space
    labels_uncond=None,     # null-token labels (đã shard) cho sampling ổn định
    special_list_t = jnp.array([0.25, 0.5, 0.75], dtype=jnp.float32)
):
    """
    Stream ~total_samples qua từng T∈T_list, mỗi bước k=0..T:
      - Minibatch variance (σ² per-sample, spatial) -> mean/max/min/std (streaming)
      - BN-style channel sigma vector (σ_c) -> log ||σ||_2
    Ghi CSV: batchsize|iterations|d|t|mean_sigma2|max_sigma2|min_sigma2|std_sigma2|bn_sigma_l2
    Trả về: dict[T] = {'mean': [...], 'max': [...], 'min': [...], 'std': [...]} (k=0..T)
    """

    # --- GLOBAL_B ---
    _dummy_local = jnp.ones((batch_shape[0],), dtype=jnp.int32)
    _dummy_g = jax.experimental.multihost_utils.process_allgather(_dummy_local)
    GLOBAL_B = int(np.array(_dummy_g).reshape(-1).shape[0])

    # --- lọc T theo denoise_timesteps hiện tại ---
    T_list_final = []
    for T in T_list:
        if T == 128 and FLAGS.model.denoise_timesteps != 128:
            continue
        T_list_final.append(T)

    # --- header CSV nếu cần ---
    write_header = (jax.process_index() == 0) and (
        not os.path.exists(csv_path))
    if write_header:
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow(['batchsize', 'iterations', 'd', 't',
                        'mean_sigma2', 'max_sigma2', 'min_sigma2', 'std_sigma2',
                        'bn_sigma_l2'])

    # --- helpers gọn cho σ² per-sample & BN-channel sums ---
    def _accumulate_varb_stats(x, acc):
        a32 = x.astype(jnp.float32)
        red_axes = tuple(range(1, a32.ndim)) if a32.ndim >= 2 else ()
        mean_b = jnp.mean(a32, axis=red_axes)         # [B_local]
        mean2_b = jnp.mean(a32 * a32, axis=red_axes)   # [B_local]
        var_b = jnp.maximum(mean2_b - mean_b * mean_b, 0.0)
        var_b_g = jax.experimental.multihost_utils.process_allgather(var_b)
        vb = np.array(var_b_g).reshape(-1)             # [B_global]
        acc['sum'] += float(vb.sum())
        acc['sum2'] += float((vb*vb).sum())
        acc['min'] = float(vb.min()) if acc['count'] == 0 else float(
            min(acc['min'], vb.min()))
        acc['max'] = float(vb.max()) if acc['count'] == 0 else float(
            max(acc['max'], vb.max()))
        acc['count'] += int(vb.shape[0])

    def _accumulate_bn_channel_sums(x, acc):
        sum_c_local = jnp.sum(x, axis=(0, 1, 2))        # [C]
        sum2_c_local = jnp.sum(x * x, axis=(0, 1, 2))    # [C]
        cnt_local = jnp.array(
            [x.shape[0]*x.shape[1]*x.shape[2]], dtype=jnp.int64)
        s1_g = jax.experimental.multihost_utils.process_allgather(sum_c_local)
        s2_g = jax.experimental.multihost_utils.process_allgather(sum2_c_local)
        cn_g = jax.experimental.multihost_utils.process_allgather(cnt_local)
        s1 = np.array(s1_g).sum(axis=0)                # [C]
        s2 = np.array(s2_g).sum(axis=0)                # [C]
        cn = int(np.array(cn_g).sum())
        acc['sum_c'] += s1
        acc['sum2_c'] += s2
        acc['count_pix'] += cn

    # --- nhãn uncond mặc định nếu chưa cấp ---
    if labels_uncond is None:
        labels_uncond = jnp.ones(
            (batch_shape[0],), dtype=jnp.int32) * FLAGS.model['num_classes']
        labels_uncond = shard_data(labels_uncond)

    results = {}  # trả về cho vẽ plot

    # --- lặp qua các T ---
    for T in T_list_final:
        num_mb = int(np.ceil(total_samples / GLOBAL_B))
        used_samples = num_mb * GLOBAL_B
        C = batch_shape[-1]

        var_accs = [{'sum': 0.0, 'sum2': 0.0, 'min': 0.0,
                     'max': 0.0, 'count': 0} for _ in range(T+1)]
        bn_accs = [{'sum_c': np.zeros((C,), dtype=np.float64),
                    'sum2_c': np.zeros((C,), dtype=np.float64),
                    'count_pix': 0} for _ in range(T+1)]

        for mb in range(num_mb):
            key = jax.random.PRNGKey(1234)
            key = jax.random.fold_in(key, mb)
            key = jax.random.fold_in(key, jax.process_index())

            x_local = jax.random.normal(key, batch_shape)  # [B_local,H,W,C]
            x = shard_data(x_local)

            # k=0
            x_stat = (vae_decode(x) if (
                decode_to_pixel and FLAGS.model.use_stable_vae) else x)
            _accumulate_varb_stats(x_stat, var_accs[0])
            _accumulate_bn_channel_sums(x_stat, bn_accs[0])

            # k=1..T (Euler)
            delta_t = 1.0 / T
            B_local = batch_shape[0]
            for ti in range(T):
                t = ti / T
                t_vec_local = jnp.full((B_local,), t, dtype=jnp.float32)
                dt_base_local = jnp.ones_like(t_vec_local) * np.log2(T)
                if FLAGS.model.train_type == 'livereflow' and T < 128:
                    dt_base_local = jnp.zeros_like(t_vec_local)

                # ==== NEW: norm x nếu t thuộc special_list_t ====
                # dùng tolerance chút cho an toàn float
                is_special_t = jnp.any(jnp.abs(t - special_list_t) < 1e-6)
                # stream_mbvar_and_csv không jit, nên cast sang bool Python được:
                if bool(is_special_t):
                    # x hiện đang là sharded array [B_local,H,W,C] trên mỗi device
                    x = instance_norm_nhwc(x)
                # ==== END NEW ====

                # shard t, dt_base sau khi xử lý
                t_vec, dt_base = shard_data(t_vec_local, dt_base_local)

                v = call_model(train_state, x, t_vec, dt_base, labels_uncond)
                x = x + v * delta_t

                x_stat = (vae_decode(x) if (
                    decode_to_pixel and FLAGS.model.use_stable_vae) else x)
                _accumulate_varb_stats(x_stat, var_accs[ti+1])
                _accumulate_bn_channel_sums(x_stat, bn_accs[ti+1])

        # finalize + CSV (host 0) + build arrays trả về cho plot
        stats_mean, stats_max, stats_min, stats_std = [], [], [], []
        if jax.process_index() == 0:
            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                for k in range(T+1):
                    va = var_accs[k]
                    mean_sigma2 = va['sum'] / max(va['count'], 1)
                    var_sigma2 = max(
                        va['sum2']/max(va['count'], 1) - mean_sigma2**2, 0.0)
                    std_sigma2 = np.sqrt(var_sigma2)
                    max_sigma2 = va['max']
                    min_sigma2 = va['min']

                    ba = bn_accs[k]
                    if ba['count_pix'] > 0:
                        mu_c = ba['sum_c'] / ba['count_pix']
                        var_c = np.maximum(
                            ba['sum2_c']/ba['count_pix'] - mu_c*mu_c, 0.0)
                        sigma_c = np.sqrt(var_c, dtype=np.float64)
                        bn_sigma_l2 = float(np.linalg.norm(sigma_c, ord=2))
                    else:
                        bn_sigma_l2 = 0.0

                    # CSV
                    w.writerow([
                        int(used_samples),
                        int(step),
                        int(T),
                        int(k),
                        float(mean_sigma2),
                        float(max_sigma2),
                        float(min_sigma2),
                        float(std_sigma2),
                        float(bn_sigma_l2),
                    ])

                    # build arrays cho plot
                    stats_mean.append(float(mean_sigma2))
                    stats_max.append(float(max_sigma2))
                    stats_min.append(float(min_sigma2))
                    stats_std.append(float(std_sigma2))

            print(
                f"[MBVAR CSV] wrote T={T} ({T+1} rows) -> {csv_path} | batchsize={used_samples}")

        # luôn trả về dict (host khác có thể trống -> caller chỉ vẽ khi host 0)
        results[T] = {
            'mean': stats_mean,
            'max':  stats_max,
            'min':  stats_min,
            'std':  stats_std,
        }

    return results
##############################################################################


def _nice_xticks(T: int):
    """
    Tạo mốc tick trục X (0..T) dễ đọc: ~9 mốc lớn, và mốc nhỏ mỗi bước (nếu T không quá lớn).
    """
    if T <= 16:
        major = np.arange(0, T + 1, 1)
        minor = None  # đã đủ dày
    else:
        step_major = max(1, T // 8)  # ~8 khoảng
        major = np.arange(0, T + 1, step_major)
        minor = np.arange(0, T + 1, 1)
    return major, minor


######################################################################################
def _plot_mbvar(xs, stats_mean, stats_max, stats_min, stats_std, T, step):
    """
    Vẽ 4 đường: mean σ², max σ², min σ², std[σ²] theo xs (0..T).
    Các đại lượng này đều cùng đơn vị (σ²), nên đặt chung một trục Y là hợp lý.
    """

    # hàm này hỗ trợ cho aesthetics
    def _best_ylim(series_list):
        """
        series_list: danh sách các list số (vd. [stats_mean, stats_max, stats_var]).
        Trả về (ymin, ymax) có đệm nhỏ cho nhìn rõ.
        """
        ymin = min(min(s) for s in series_list)
        ymax = max(max(s) for s in series_list)
        if ymax <= ymin:
            ymax = ymin + 1e-6
        pad = 0.05 * (ymax - ymin)
        return ymin - pad, ymax + pad

    mean_s, max_s, min_s, std_s = stats_mean, stats_max, stats_min, stats_std
    y_label = "minibatch variance / std(σ²)"

    fig = plt.figure(figsize=(8, 5))
    plt.plot(xs, mean_s, marker='o', linewidth=1.6, label="mean σ²")
    plt.plot(xs, max_s,  marker='o', linewidth=1.6, label="max σ²")
    plt.plot(xs, min_s,  marker='o', linewidth=1.6, label="min σ²")
    plt.plot(xs, std_s,  marker='o', linewidth=1.6, label="std[σ²]")

    major, minor = _nice_xticks(T)
    ax = plt.gca()
    ax.set_xlim(0, T)
    ax.set_xticks(major)
    if minor is not None:
        ax.set_xticks(minor, minor=True)
    ax.set_xlabel("denoising step k (0..T)")

    ymin, ymax = _best_ylim([mean_s, max_s, min_s, std_s])
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15)

    ax.set_ylabel(y_label)
    ax.set_title(f"MB variance vs step | T={T} | step={step}")
    ax.legend()
    return fig
