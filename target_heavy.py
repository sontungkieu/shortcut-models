import jax
import jax.numpy as jnp
import numpy as np

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def _prepare_grid(N, eps, dtype):
    """Helper: Create base linear grid with epsilon padding."""
    x = jnp.linspace(jnp.array(0.0, dtype), jnp.array(1.0, dtype), N + 1, dtype=dtype)
    e = jnp.array(eps, dtype=dtype)
    x_tilde = e + (1.0 - 2.0 * e) * x
    return jnp.clip(x_tilde, 0.0, 1.0)

def _renorm(t_raw):
    """Helper: Normalize t to exactly [0, 1]."""
    t0 = t_raw[0]
    t1 = t_raw[-1]
    t = (t_raw - t0) / (t1 - t0 + 1e-12)
    return jnp.clip(t, 0.0, 1.0)

# ==============================================================================
# 2. SPECIFIC SCHEDULE MAKERS (Core Math)
# ==============================================================================

def make_sin_t_grid(N, eps=1e-3, renorm=True, dtype=jnp.float32):
    x = _prepare_grid(N, eps, dtype)
    t_raw = jnp.sin((jnp.pi / 2.0) * x).astype(dtype)
    return _renorm(t_raw) if renorm else t_raw

def make_quad_t_grid(N, eps=1e-3, renorm=True, dtype=jnp.float32):
    x = _prepare_grid(N, eps, dtype)
    t_raw = (1.0 - (1.0 - x) ** 2).astype(dtype)
    return _renorm(t_raw) if renorm else t_raw

def make_cubic_t_grid(N, eps=1e-3, renorm=True, dtype=jnp.float32):
    x = _prepare_grid(N, eps, dtype)
    t_raw = (1.0 - (1.0 - x) ** 3).astype(dtype)
    return _renorm(t_raw) if renorm else t_raw

def make_quartic_t_grid(N, eps=1e-3, renorm=True, dtype=jnp.float32):
    x = _prepare_grid(N, eps, dtype)
    t_raw = (1.0 - (1.0 - x) ** 4).astype(dtype)
    return _renorm(t_raw) if renorm else t_raw

def make_quintic_t_grid(N, eps=1e-3, renorm=True, dtype=jnp.float32):
    x = _prepare_grid(N, eps, dtype)
    t_raw = (1.0 - (1.0 - x) ** 5).astype(dtype)
    return _renorm(t_raw) if renorm else t_raw

def make_sqrt_t_grid(N, eps=1e-3, renorm=True, dtype=jnp.float32):
    x = _prepare_grid(N, eps, dtype)
    t_raw = jnp.sqrt(1.0 - (1.0 - x) ** 2).astype(dtype)
    return _renorm(t_raw) if renorm else t_raw

def make_root4_t_grid(N, eps=1e-3, renorm=True, dtype=jnp.float32):
    x = _prepare_grid(N, eps, dtype)
    inner = 1.0 - (1.0 - x) ** 4
    t_raw = jnp.power(jnp.maximum(inner, 0.0), 0.25).astype(dtype)
    return _renorm(t_raw) if renorm else t_raw

# ==============================================================================
# 3. PUBLIC DISPATCHER (Gọi hàm này từ bên ngoài để lấy Grid)
# ==============================================================================

def get_scheduler_grid(FLAGS):
    """
    Public function to get the t_grid based on FLAGS configuration.
    Useful for initialization or logging outside the training loop.
    """
    N = int(FLAGS.model["denoise_timesteps"])
    train_type = FLAGS.model['train_type']
    
    # Common default params
    eps = float(FLAGS.model.get("schedule_eps", 1e-3))
    do_renorm = bool(FLAGS.model.get("schedule_renorm", True))

    if train_type == 'shortcut_sin':
        return make_sin_t_grid(N, eps=float(FLAGS.model.get("sin_eps", eps)), renorm=do_renorm)
    
    elif train_type == 'shortcut_quad':
        return make_quad_t_grid(N, eps=float(FLAGS.model.get("quad_eps", eps)), renorm=do_renorm)
        
    elif train_type == 'shortcut_cubic':
        return make_cubic_t_grid(N, eps=float(FLAGS.model.get("cubic_eps", eps)), renorm=do_renorm)
        
    elif train_type == 'shortcut_quartic':
        return make_quartic_t_grid(N, eps=float(FLAGS.model.get("quartic_eps", eps)), renorm=do_renorm)
        
    elif train_type == 'shortcut_quintic':
        return make_quintic_t_grid(N, eps=float(FLAGS.model.get("quintic_eps", eps)), renorm=do_renorm)
        
    elif train_type == 'shortcut_sqrt':
        return make_sqrt_t_grid(N, eps=float(FLAGS.model.get("sqrt_eps", eps)), renorm=do_renorm)
        
    elif train_type == 'shortcut_root4':
        return make_root4_t_grid(N, eps=float(FLAGS.model.get("root4_eps", eps)), renorm=do_renorm)
        
    else:
        # Fallback default
        return make_quad_t_grid(N, eps=eps, renorm=do_renorm)

# ==============================================================================
# 4. MAIN TARGET GENERATOR
# ==============================================================================

def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}
    sg = jax.lax.stop_gradient

    # --- A. GET GRID (Gọi qua hàm wrapper ở trên) ---
    t_grid = get_scheduler_grid(FLAGS)
    N = int(FLAGS.model["denoise_timesteps"])

    # --- B. PREPARE CONSTANTS ---
    log2_sections = np.log2(N).astype(np.int32)
    force_t_f32 = jnp.asarray(force_t, dtype=jnp.float32)
    force_dt_i32 = jnp.asarray(force_dt, dtype=jnp.int32)

    # --- C. BOOTSTRAP LOGIC ---
    bootstrap_batchsize = FLAGS.batch_size // FLAGS.model["bootstrap_every"]

    if FLAGS.model["bootstrap_dt_bias"] == 0:
        dt_base = jnp.repeat(
            log2_sections - 1 - jnp.arange(log2_sections, dtype=jnp.int32),
            bootstrap_batchsize // log2_sections,
        )
        dt_base = jnp.concatenate(
            [dt_base, jnp.zeros((bootstrap_batchsize - dt_base.shape[0],), dtype=jnp.int32)], axis=0
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
            ], axis=0
        )
        dt_base = jnp.concatenate(
            [dt_base, jnp.zeros((bootstrap_batchsize - dt_base.shape[0],), dtype=jnp.int32)], axis=0
        )
        num_dt_cfg = (bootstrap_batchsize // 2) // log2_sections

    dt_base = dt_base.astype(jnp.int32)
    dt_base = jnp.where(force_dt_i32 != -1, force_dt_i32, dt_base)
    dt_base_bootstrap = dt_base + 1

    # --- D. MAP INDICES TO TIME ---
    dt_sections = jnp.power(2, dt_base)
    block = (N // dt_sections).astype(jnp.int32)
    half = (block // 2).astype(jnp.int32)

    k = jax.random.randint(time_key, (bootstrap_batchsize,), minval=0, maxval=dt_sections).astype(jnp.int32)
    idx0_rand = k * block
    
    idx_near = jnp.argmin(jnp.abs(t_grid - force_t_f32)).astype(jnp.int32)
    idx0_force = (idx_near // block) * block
    use_force_t = (force_t_f32 != -1.0)

    idx0 = jnp.where(use_force_t, idx0_force, idx0_rand)
    idx_mid = idx0 + half

    t_from_grid = t_grid[idx0].astype(jnp.float32)
    t_mid = t_grid[idx_mid].astype(jnp.float32)

    t = jnp.where(use_force_t, force_t_f32, t_from_grid)
    t_full = t[:, None, None, None]
    dt_bootstrap = (t_mid - t).astype(jnp.float32)

    # --- E. GENERATE BOOTSTRAP TARGETS ---
    x_1 = images[:bootstrap_batchsize]
    x_0 = jax.random.normal(noise_key, x_1.shape)
    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1

    bst_labels = labels[:bootstrap_batchsize]
    call_model_fn = train_state.call_model if FLAGS.model["bootstrap_ema"] == 0 else train_state.call_model_ema

    if not FLAGS.model["bootstrap_cfg"]:
        v_b1 = sg(call_model_fn(x_t, t, dt_base_bootstrap, bst_labels, train=False))
        x_t2 = sg(x_t + dt_bootstrap[:, None, None, None] * v_b1)
        x_t2 = sg(jnp.clip(x_t2, -4, 4))
        v_b2 = sg(call_model_fn(x_t2, t_mid, dt_base_bootstrap, bst_labels, train=False))
        v_target = sg((v_b1 + v_b2) / 2.0)
    else:
        x_t_extra = jnp.concatenate([x_t, x_t[:num_dt_cfg]], axis=0)
        t_extra = jnp.concatenate([t, t[:num_dt_cfg]], axis=0)
        dt_base_extra = jnp.concatenate([dt_base_bootstrap, dt_base_bootstrap[:num_dt_cfg]], axis=0)
        labels_extra = jnp.concatenate(
            [bst_labels, jnp.ones((num_dt_cfg,), dtype=jnp.int32) * FLAGS.model["num_classes"]], axis=0
        )

        v_b1_raw = sg(call_model_fn(x_t_extra, t_extra, dt_base_extra, labels_extra, train=False))
        v_b_cond = v_b1_raw[:x_1.shape[0]]
        v_b_uncond = v_b1_raw[x_1.shape[0]:]
        v_cfg = v_b_uncond + FLAGS.model["cfg_scale"] * (v_b_cond[:num_dt_cfg] - v_b_uncond)
        v_b1 = jnp.concatenate([v_cfg, v_b_cond[num_dt_cfg:]], axis=0)

        x_t2 = sg(x_t + dt_bootstrap[:, None, None, None] * v_b1)
        x_t2 = sg(jnp.clip(x_t2, -4, 4))

        x_t2_extra = jnp.concatenate([x_t2, x_t2[:num_dt_cfg]], axis=0)
        t2_extra = jnp.concatenate([t_mid, t_mid[:num_dt_cfg]], axis=0)

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

    # --- F. FLOW MATCHING TARGETS ---
    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model["class_dropout_prob"], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model["num_classes"], labels)
    info["dropped_ratio"] = jnp.mean(labels_dropped == FLAGS.model["num_classes"])

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

    # --- G. MERGE AND RETURN ---
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