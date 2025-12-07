###
# this is ablation of LFM.pdf, DSLab team meeting 05/12/2025
###
# keep shortcut model architecture but change the loss function 
# as normal as flow matching
import jax
import jax.numpy as jnp
import numpy as np


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    """
    Phiên bản Flow Matching ONLY cho train_type='shortcut':
      - Không dùng bootstrap / teacher / EMA để tạo target.
      - Vẫn trả về:
          x_t       : [B, C, H, W]  trạng thái tại timestep t
          v_t       : [B, C, H, W]  vector field tương ứng (flow-matching)
          t         : [B]           thời gian liên tục trong [0,1]
          dt_base   : [B] (int32)   index d (log2(1/d)) để feed vào shortcut model
          labels    : [B]           labels sau dropout
          info      : dict          info cho logging
      - d = 2^{-dt_base}, và v_t được suy ra từ cặp (x_t, x_{t+d}) theo đúng tinh thần phần 2.1.
    """
    # Không dùng train_state trong FM-only.
    del train_state

    # 0) RNG
    label_key, dt_key, t_key, noise_key = jax.random.split(key, 4)
    info = {}

    B = images.shape[0]
    T = FLAGS.model['denoise_timesteps']  # ví dụ 32 hoặc 128

    # 1) Class dropout (giữ nguyên như code gốc)
    dropout_prob = FLAGS.model.get('class_dropout_prob', 0.0)
    num_classes = FLAGS.model['num_classes']

    labels_dropout = jax.random.bernoulli(label_key, dropout_prob, (B,))
    labels_dropped = jnp.where(labels_dropout, num_classes, labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == num_classes)

    # 2) Sample dt_base (d thay đổi giống shortcut: d = 2^{-dt_base})
    #    0 -> d = 1.0, 1 -> 1/2, ..., log2(T) -> 1/T
    max_level = int(np.log2(T))  # ví dụ T=32 -> max_level=5
    dt_base = jax.random.randint(
        dt_key,
        (B,),
        minval=0,
        maxval=max_level + 1,  # inclusive 0..max_level
    ).astype(jnp.int32)

    # force_dt: override dt_base khi evaluation / loss-analysis
    if force_dt != -1:
        dt_base = jnp.ones_like(dt_base, dtype=jnp.int32) * jnp.int32(force_dt)

    # d = 2^{-dt_base}
    dt = 1.0 / (2.0 ** dt_base.astype(jnp.float32))          # [B]
    dt_sections = (2 ** dt_base).astype(jnp.int32)           # [B] = 1/d

    # 3) Sample t trên lưới {0, 1/2^k, ..., (2^k - 1)/2^k}, đảm bảo t + d <= 1
    u = jax.random.uniform(t_key, (B,))
    t_idx = jnp.floor(u * dt_sections).astype(jnp.float32)   # [B]
    t = t_idx / dt_sections.astype(jnp.float32)              # [B]

    # đảm bảo t + d <= 1 (phòng khi random gần 1)
    t = jnp.minimum(t, 1.0 - dt)

    # force_t: override t khi eval
    if force_t != -1:
        t = jnp.ones_like(t, dtype=jnp.float32) * float(force_t)

    t_full = t[:, None, None, None]      # [B,1,1,1]
    dt_full = dt[:, None, None, None]    # [B,1,1,1]

    # 4) Flow Matching analytic: tạo cặp (x_t, x_{t+d}) rồi rút ra v_t
    x_1 = images
    x_0 = jax.random.normal(noise_key, images.shape)
    eps = 1e-5

    # x_t = (1 - (1 - eps)*t) x0 + t x1
    x_t = (1.0 - (1.0 - eps) * t_full) * x_0 + t_full * x_1

    # x_{t+d} = (1 - (1 - eps)(t + d)) x0 + (t + d) x1
    t_plus_d = (t + dt)[:, None, None, None]
    x_t_plus_d = (1.0 - (1.0 - eps) * t_plus_d) * x_0 + t_plus_d * x_1

    # v_t = (x_{t+d} - x_t) / d  (đúng với FM, đồng thời “dựa” trên cặp x_t, x_{t+d})
    v_t = (x_t_plus_d - x_t) / dt_full

    # 5) Info cho logging (giữ field 'bootstrap_ratio' để không lỗi code khác)
    info['bootstrap_ratio'] = jnp.array(0.0)
    info['v_magnitude'] = jnp.sqrt(jnp.mean(jnp.square(v_t)))

    return x_t, v_t, t, dt_base, labels_dropped, info
