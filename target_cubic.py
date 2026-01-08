# cubic_target.py
import jax
import jax.numpy as jnp
import numpy as np

def make_cubic_t_grid(
    denoise_timesteps: int,
    *,
    eps: float = 1e-3,
    renorm: bool = True,
    dtype=jnp.float32,
):
    """
    Make a monotone time grid t_grid of length N+1 using Cubic Ease-Out:
        x = i/N
        x_tilde = eps + (1 - 2*eps) * x    (avoid endpoints)
        t_raw = 1 - (1 - x_tilde)^3        (Cubic: Very fast early, very slow late)
        t = renorm(t_raw) -> [0,1]

    Returns:
        t_grid: shape (N+1,), float32, increasing, in [0,1]
    """
    N = int(denoise_timesteps)
    x = jnp.linspace(jnp.array(0.0, dtype), jnp.array(1.0, dtype), N + 1, dtype=dtype)

    e = jnp.array(eps, dtype=dtype)
    # push away from 0 and 1
    x_tilde = e + (1.0 - 2.0 * e) * x
    x_tilde = jnp.clip(x_tilde, 0.0, 1.0)

    # --- THAY ĐỔI CHÍNH Ở ĐÂY: Hàm Bậc 3 (Cubic) ---
    # f(t) = 1 - (1-t)^3
    # Đạo hàm giảm nhanh hơn so với bậc 2, tập trung rất nhiều bước ở cuối
    t_raw = (1.0 - (1.0 - x_tilde) ** 3).astype(dtype)

    if renorm:
        t0 = t_raw[0]
        t1 = t_raw[-1]
        t = (t_raw - t0) / (t1 - t0 + 1e-12)
        return jnp.clip(t, 0.0, 1.0)

    return jnp.clip(t_raw, 0.0, 1.0)

def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    """
    Returns:
      x_t, v_t, t, dt_base, labels_dropped, info
    dt_base is the discrete "step size level" like kvfran code (int).
    """
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}
    sg = jax.lax.stop_gradient

    N = int(FLAGS.model["denoise_timesteps"])
    log2_sections = np.log2(N).astype(np.int32)

    # --- CUBIC t-grid setup ---
    # Sử dụng .get() để fallback về default nếu config chưa có key 'cubic_*'
    t_grid = make_cubic_t_grid(
        denoise_timesteps=N,
        eps=float(FLAGS.model.get("cubic_eps", 1e-3)), 
        renorm=bool(FLAGS.model.get("cubic_renorm", True)),
    )  # [N+1]

    # JIT-safe force scalars
    force_t_f32 = jnp.asarray(force_t, dtype=jnp.float32)
    force_dt_i32 = jnp.asarray(force_dt, dtype=jnp.int32)

    # 1) =========== Sample dt_base (bootstrap). ============
    bootstrap_batchsize = FLAGS.batch_size // FLAGS.model["bootstrap_every"]

    if FLAGS.model["bootstrap_dt_bias"] == 0:
        dt_base = jnp.repeat(
            log2_sections - 1 - jnp.arange(log2_sections, dtype=jnp.int32),
            bootstrap_batchsize // log2_sections,
        )
        dt_base = jnp.concatenate(
            [dt_base, jnp.zeros((bootstrap_batchsize - dt_base.shape[0],), dtype=jnp.int32)],
            axis=0,
        )
        num_dt_cfg = bootstrap_batchsize // log2_sections
    else:
        dt_base = jnp.repeat(
            log2_sections - 1 - jnp.arange(log2_sections - 2, dtype=jnp.int32),
            (bootstrap_batchsize // 2) // log2_sections,
        )
        dt_base = jnp.concatenate(
            [
                dt_base,
                jnp.ones((bootstrap_batchsize // 4,), dtype=jnp.int32),
                jnp.zeros((bootstrap_batchsize // 4,), dtype=jnp.int32),
            ],
            axis=0,
        )
        dt_base = jnp.concatenate(
            [dt_base, jnp.zeros((bootstrap_batchsize - dt_base.shape[0],), dtype=jnp.int32)],
            axis=0,
        )
        num_dt_cfg = (bootstrap_batchsize // 2) // log2_sections

    dt_base = dt_base.astype(jnp.int32)
    dt_base = jnp.where(force_dt_i32 != -1, force_dt_i32, dt_base)

    # bootstrap uses dt_base+1 (half-step “level”), same as kvfran
    dt_base_bootstrap = dt_base + 1

    # 2) =========== Sample t (bootstrap) from CUBIC grid, aligned to dt_base. ============
    dt_sections = jnp.power(2, dt_base)          # 1,2,4,8,...
    block = (N // dt_sections).astype(jnp.int32) # index jump on base grid
    half = (block // 2).astype(jnp.int32)

    # random aligned start index
    k = jax.random.randint(time_key, (bootstrap_batchsize,), minval=0, maxval=dt_sections).astype(jnp.int32)
    idx0_rand = k * block

    # if force_t is set (eval), align idx0 to nearest grid index then snap to multiple-of-block
    idx_near = jnp.argmin(jnp.abs(t_grid - force_t_f32)).astype(jnp.int32)  # scalar
    idx0_force = (idx_near // block) * block
    use_force_t = (force_t_f32 != -1.0)

    idx0 = jnp.where(use_force_t, idx0_force, idx0_rand)
    idx_mid = idx0 + half

    t_from_grid = t_grid[idx0].astype(jnp.float32)
    t_mid = t_grid[idx_mid].astype(jnp.float32)

    t = jnp.where(use_force_t, force_t_f32, t_from_grid)
    t_full = t[:, None, None, None]

    # actual "half-step" in t-space under CUBIC schedule
    dt_bootstrap = (t_mid - t).astype(jnp.float32)

    # 3) =========== Generate Bootstrap Targets (stopgrad). ============
    x_1 = images[:bootstrap_batchsize]
    x_0 = jax.random.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1

    bst_labels = labels[:bootstrap_batchsize]
    call_model_fn = train_state.call_model if FLAGS.model["bootstrap_ema"] == 0 else train_state.call_model_ema

    if not FLAGS.model["bootstrap_cfg"]:
        v_b1 = sg(call_model_fn(x_t, t, dt_base_bootstrap, bst_labels, train=False))

        x_t2 = sg(x_t + dt_bootstrap[:, None, None, None] * v_b1)
        x_t2 = sg(jnp.clip(x_t2, -4, 4))

        # IMPORTANT: second evaluation time is t_mid (grid), not t + dt_bootstrap
        v_b2 = sg(call_model_fn(x_t2, t_mid, dt_base_bootstrap, bst_labels, train=False))
        v_target = sg((v_b1 + v_b2) / 2.0)
    else:
        # ---- CFG path: keep original kvfran packing, just swap time for step2 to t_mid ----
        x_t_extra = jnp.concatenate([x_t, x_t[:num_dt_cfg]], axis=0)
        t_extra = jnp.concatenate([t, t[:num_dt_cfg]], axis=0)
        dt_base_extra = jnp.concatenate([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], axis=0)
        labels_extra = jnp.concatenate(
            [bst_labels, jnp.ones((num_dt_cfg,), dtype=jnp.int32) * FLAGS.model["num_classes"]],
            axis=0,
        )

        v_b1_raw = sg(call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False))
        v_b_cond = v_b1_raw[:x_1.shape[0]]
        v_b_uncond = v_b1_raw[x_1.shape[0]:]
        v_cfg = v_b_uncond + FLAGS.model["cfg_scale"] * (v_b_cond[:num_dt_cfg] - v_b_uncond)
        v_b1 = jnp.concatenate([v_cfg, v_b_cond[num_dt_cfg:]], axis=0)

        x_t2 = sg(x_t + dt_bootstrap[:, None, None, None] * v_b1)
        x_t2 = sg(jnp.clip(x_t2, -4, 4))

        x_t2_extra = jnp.concatenate([x_t2, x_t2[:num_dt_cfg]], axis=0)
        t2_extra = jnp.concatenate([t_mid, t_mid[:num_dt_cfg]], axis=0)  # grid midpoint time

        v_b2_raw = sg(call_model_fn(x_t2_extra, t2_extra, dt_base_extra, labels_extra, train=False))
        v_b2_cond = v_b2_raw[:x_1.shape[0]]
        v_b2_uncond = v_b2_raw[x_1.shape[0]:]
        v_b2_cfg = v_b2_uncond + FLAGS.model["cfg_scale"] * (v_b2_cond[:num_dt_cfg] - v_b2_uncond)
        v_b2 = jnp.concatenate([v_b2_cfg, v_b2_cond[num_dt_cfg:]], axis=0)

        v_target = sg((v_b1 + v_b2) / 2.0)

    v_target = jnp.clip(v_target, -4, 4)
    bst_v = v_target
    bst_dt = dt_base
    bst_t = t
    bst_xt = x_t
    bst_l = bst_labels

    # 4) =========== Generate Flow-Matching Targets (t from CUBIC grid). ============
    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model["class_dropout_prob"], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model["num_classes"], labels)
    info["dropped_ratio"] = jnp.mean(labels_dropped == FLAGS.model["num_classes"])

    # sample index then map via t_grid
    idx_fm = jax.random.randint(time_key, (images.shape[0],), minval=0, maxval=N).astype(jnp.int32)
    t_fm = t_grid[idx_fm].astype(jnp.float32)
    t_fm = jnp.where(use_force_t, force_t_f32, t_fm)
    t_full = t_fm[:, None, None, None]

    x_0 = jax.random.normal(noise_key, images.shape)
    x_1 = images
    x_t_fm = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t_fm = x_1 - (1 - 1e-5) * x_0

    dt_flow = np.log2(N).astype(jnp.int32)
    dt_base_fm = jnp.ones((images.shape[0],), dtype=jnp.int32) * dt_flow
    dt_base_fm = jnp.where(force_dt_i32 != -1, force_dt_i32, dt_base_fm)

    # 5) Merge Flow+Bootstrap
    bst_size = FLAGS.batch_size // FLAGS.model["bootstrap_every"]
    bst_size_data = FLAGS.batch_size - bst_size

    x_t_out = jnp.concatenate([bst_xt, x_t_fm[:bst_size_data]], axis=0)
    t_out = jnp.concatenate([bst_t, t_fm[:bst_size_data]], axis=0)
    dt_base_out = jnp.concatenate([bst_dt, dt_base_fm[:bst_size_data]], axis=0)
    v_out = jnp.concatenate([bst_v, v_t_fm[:bst_size_data]], axis=0)
    labels_out = jnp.concatenate([bst_l, labels_dropped[:bst_size_data]], axis=0)

    info["bootstrap_ratio"] = jnp.mean(dt_base_out != dt_flow)
    info["v_magnitude_bootstrap"] = jnp.sqrt(jnp.mean(jnp.square(bst_v)))

    return x_t_out, v_out, t_out, dt_base_out, labels_out, info