import jax
import jax.numpy as jnp
import numpy as np

# --- add these helpers from section (1) ---
# _cosine_alpha_bar(...)
# make_cosine_t_grid(...)
import jax
import jax.numpy as jnp

def _cosine_alpha_bar(u, s: float = 0.008):
    """
    alpha_bar(u) = cos^2(((u+s)/(1+s))*pi/2) / cos^2((s/(1+s))*pi/2)
    u in [0,1]. (Improved DDPM cosine schedule)
    """
    u = jnp.clip(u, 0.0, 1.0)
    ang0 = (s / (1.0 + s)) * (jnp.pi / 2.0)
    denom = jnp.cos(ang0) ** 2
    ang = ((u + s) / (1.0 + s)) * (jnp.pi / 2.0)
    return (jnp.cos(ang) ** 2) / denom

def make_cosine_t_grid(denoise_timesteps: int, s: float = 0.008, eps: float = 1e-3):
    """
    Build monotone t_grid[0..N] in [0,1], where:
      t_raw(u) = 1 - alpha_bar(u)
    eps pushes u away from 0/1, then we renormalize so t_grid[0]=0, t_grid[N]=1.
    """
    N = int(denoise_timesteps)
    u = jnp.linspace(0.0, 1.0, N + 1)
    u2 = eps + (1.0 - 2.0 * eps) * u
    t_raw = 1.0 - _cosine_alpha_bar(u2, s=s)   # increasing 0..1
    t = (t_raw - t_raw[0]) / (t_raw[-1] - t_raw[0] + 1e-12)
    return jnp.clip(t, 0.0, 1.0)


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    sg = jax.lax.stop_gradient

    N = int(FLAGS.model['denoise_timesteps'])
    cosine_s = float(FLAGS.model.get('cosine_s', 0.008))
    cosine_eps = float(FLAGS.model.get('cosine_eps', 1e-3))
    t_grid = make_cosine_t_grid(N, s=cosine_s, eps=cosine_eps)  # [N+1]

    # 1) =========== Sample dt (dt_base). ============
    bootstrap_batchsize = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    log2_sections = np.log2(N).astype(np.int32)

    if FLAGS.model['bootstrap_dt_bias'] == 0:
        dt_base = jnp.repeat(
            log2_sections - 1 - jnp.arange(log2_sections),
            bootstrap_batchsize // log2_sections
        )
        dt_base = jnp.concatenate([dt_base, jnp.zeros(bootstrap_batchsize - dt_base.shape[0],)])
        num_dt_cfg = bootstrap_batchsize // log2_sections
    else:
        dt_base = jnp.repeat(
            log2_sections - 1 - jnp.arange(log2_sections - 2),
            (bootstrap_batchsize // 2) // log2_sections
        )
        dt_base = jnp.concatenate([dt_base, jnp.ones(bootstrap_batchsize // 4), jnp.zeros(bootstrap_batchsize // 4)])
        dt_base = jnp.concatenate([dt_base, jnp.zeros(bootstrap_batchsize - dt_base.shape[0],)])
        num_dt_cfg = (bootstrap_batchsize // 2) // log2_sections

    # force_dt is interpreted as dt_base (int)
    dt_base = dt_base.astype(jnp.int32)
    force_dt_i32 = jnp.asarray(force_dt, dtype=jnp.int32)  # tracer-safe :contentReference[oaicite:1]{index=1}
    dt_base = jnp.where(force_dt_i32 != -1, force_dt_i32, dt_base)
    dt_base_bootstrap = dt_base + 1


    # 2) =========== Sample t from COSINE GRID. ============
    # dt_sections = 2^dt_base (how many chunks), step_idx = N / dt_sections (index jump)
    dt_sections = (2 ** dt_base).astype(jnp.int32)         # [1,2,4,8,16,32] if N=32
    step_idx = (N // dt_sections).astype(jnp.int32)        # [N, N/2, ..., 1?], here min is 2 in kvfran’s dt_base set
    half_step_idx = (step_idx // 2).astype(jnp.int32)      # for dt/2 bootstrap

    # choose m in [0, dt_sections-1], and align idx = m * step_idx so idx+step_idx <= N
    m = jax.random.randint(time_key, (bootstrap_batchsize,), minval=0, maxval=dt_sections)
    idx0 = (m * step_idx).astype(jnp.int32)
    idx1 = (idx0 + half_step_idx).astype(jnp.int32)
    idx2 = (idx0 + step_idx).astype(jnp.int32)

    t  = t_grid[idx0]          # [B]
    t_mid = t_grid[idx1]       # [B]
    t_next = t_grid[idx2]      # [B]

    # actual dt in t-space (variable, from schedule)
    dt1 = t_mid - t            # first half
    dt2 = t_next - t_mid       # second half
    dt_big = t_next - t        # for logging/debug

    force_t_vec = jnp.ones(bootstrap_batchsize, dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)

    t_full = t[:, None, None, None]

    # 3) =========== Generate Bootstrap Targets (with stopgrad) ============
    x_1 = images[:bootstrap_batchsize]
    x_0 = jax.random.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1

    bst_labels = labels[:bootstrap_batchsize]
    call_model_fn = train_state.call_model if FLAGS.model['bootstrap_ema'] == 0 else train_state.call_model_ema

    if not FLAGS.model['bootstrap_cfg']:
        v_b1 = sg(call_model_fn(x_t, t, dt_base_bootstrap, bst_labels, train=False))
        x_t2 = sg(x_t + dt1[:, None, None, None] * v_b1)
        x_t2 = sg(jnp.clip(x_t2, -4, 4))

        # IMPORTANT: second step time is t_mid (from grid), not t + dt/2
        v_b2 = sg(call_model_fn(x_t2, t_mid, dt_base_bootstrap, bst_labels, train=False))

        v_target = sg((v_b1 + v_b2) / 2.0)
    else:
        x_t_extra = jnp.concatenate([x_t, x_t[:num_dt_cfg]], axis=0)
        t_extra = jnp.concatenate([t, t[:num_dt_cfg]], axis=0)
        dt_base_extra = jnp.concatenate([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], axis=0)
        labels_extra = jnp.concatenate(
            [bst_labels, jnp.ones(num_dt_cfg, dtype=jnp.int32) * FLAGS.model['num_classes']],
            axis=0
        )

        v_b1_raw = sg(call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False))
        v_b_cond = v_b1_raw[:x_1.shape[0]]
        v_b_uncond = v_b1_raw[x_1.shape[0]:]
        v_cfg = v_b_uncond + FLAGS.model['cfg_scale'] * (v_b_cond[:num_dt_cfg] - v_b_uncond)
        v_b1 = jnp.concatenate([v_cfg, v_b_cond[num_dt_cfg:]], axis=0)

        x_t2 = sg(x_t + dt1[:, None, None, None] * v_b1)
        x_t2 = sg(jnp.clip(x_t2, -4, 4))

        x_t2_extra = jnp.concatenate([x_t2, x_t2[:num_dt_cfg]], axis=0)
        t2_extra = jnp.concatenate([t_mid, t_mid[:num_dt_cfg]], axis=0)

        v_b2_raw = sg(call_model_fn(x_t2_extra, t2_extra, dt_base_extra, labels_extra, train=False))
        v_b2_cond = v_b2_raw[:x_1.shape[0]]
        v_b2_uncond = v_b2_raw[x_1.shape[0]:]
        v_b2_cfg = v_b2_uncond + FLAGS.model['cfg_scale'] * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
        v_b2 = jnp.concatenate([v_b2_cfg, v_b2_cond[num_dt_cfg:]], axis=0)

        v_target = sg((v_b1 + v_b2) / 2.0)

    v_target = jnp.clip(v_target, -4, 4)

    bst_v = v_target
    bst_dt = dt_base
    bst_t = t
    bst_xt = x_t
    bst_l = bst_labels

    info['cosine_s'] = cosine_s
    info['cosine_eps'] = cosine_eps
    info['dt_big_mean'] = jnp.mean(dt_big)
    info['dt1_mean'] = jnp.mean(dt1)
    info['dt2_mean'] = jnp.mean(dt2)

    # 4) =========== Generate Flow-Matching Targets (t from cosine grid) ============
    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model['class_dropout_prob'], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model['num_classes'], labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == FLAGS.model['num_classes'])

    # sample index on cosine grid, then t = t_grid[idx]
    idx_fm = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=N).astype(jnp.int32)
    t = t_grid[idx_fm].astype(jnp.float32)

    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]

    x_0 = jax.random.normal(noise_key, images.shape)
    x_1 = images
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = np.log2(N).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    # 5) Merge Flow+Bootstrap
    bst_size = FLAGS.batch_size // FLAGS.model['bootstrap_every']
    bst_size_data = FLAGS.batch_size - bst_size

    x_t = jnp.concatenate([bst_xt, x_t[:bst_size_data]], axis=0)
    t = jnp.concatenate([bst_t, t[:bst_size_data]], axis=0)
    dt_base = jnp.concatenate([bst_dt, dt_base[:bst_size_data]], axis=0)
    v_t = jnp.concatenate([bst_v, v_t[:bst_size_data]], axis=0)
    labels_dropped = jnp.concatenate([bst_l, labels_dropped[:bst_size_data]], axis=0)

    info['bootstrap_ratio'] = jnp.mean(dt_base != dt_flow)
    info['v_magnitude_bootstrap'] = jnp.sqrt(jnp.mean(jnp.square(bst_v)))
    info['v_magnitude_b1'] = jnp.sqrt(jnp.mean(jnp.square(v_b1)))
    info['v_magnitude_b2'] = jnp.sqrt(jnp.mean(jnp.square(v_b2)))

    return x_t, v_t, t, dt_base, labels_dropped, info
