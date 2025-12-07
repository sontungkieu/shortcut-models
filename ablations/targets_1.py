import jax
import jax.numpy as jnp
import numpy as np


def get_targets(FLAGS, key, train_state, images, labels,
                force_t=-1, force_dt=-1):
    """
    Flow Matching ONLY + dt thay đổi kiểu shortcut.

    - KHÔNG bootstrap / teacher.
    - Model nhận (x_t, t, dt_base, labels) như shortcut.
    - Target v_t được xây từ cặp (x_t, x_{t+d}) đúng tinh thần 2.1.

    Trả về:
      x_t         : [B, C, H, W]
      v_t         : [B, C, H, W]
      t           : [B]
      dt_base     : [B] int32 (log2(1/d))
      labels_drop : [B]
      info        : dict
    """
    del train_state  # FM-only nên không dùng đến

    # --- 0) RNG split ---
    label_key, dt_key, t_key, noise_key = jax.random.split(key, 4)
    info = {}

    B = images.shape[0]
    T = FLAGS.model['denoise_timesteps']  # vd: 32 hoặc 128

    # --- 1) Class-dropout giống code gốc ---
    dropout_prob = FLAGS.model.get('class_dropout_prob', 0.0)
    num_classes = FLAGS.model['num_classes']

    labels_dropout = jax.random.bernoulli(label_key, dropout_prob, (B,))
    labels_dropped = jnp.where(labels_dropout, num_classes, labels)
    info['dropped_ratio'] = jnp.mean(labels_dropped == num_classes)

    # --- 2) Sample dt_base (d thay đổi) ---
    # dt_base ∈ {0, 1, ..., log2(T)}, d = 2^{-dt_base}
    max_level = int(np.log2(T))
    dt_base = jax.random.randint(
        dt_key,
        (B,),
        minval=0,
        maxval=max_level + 1,
    ).astype(jnp.int32)

    # ⚠️ KHÔNG dùng if Python trên force_dt nữa:
    #   dùng jnp.where để override khi force_dt != -1
    force_dt_val = jnp.asarray(force_dt, dtype=jnp.int32)
    dt_base = jnp.where(
        force_dt_val == -1,
        dt_base,
        jnp.ones_like(dt_base) * force_dt_val,
    )

    # d = 2^{-dt_base}
    dt = 1.0 / (2.0 ** dt_base.astype(jnp.float32))   # [B]
    dt_sections = (2 ** dt_base).astype(jnp.int32)    # [B] = 1/d

    # --- 3) Sample t trên lưới 1/d, đảm bảo t + d <= 1 ---
    u = jax.random.uniform(t_key, (B,))
    t_idx = jnp.floor(u * dt_sections).astype(jnp.float32)   # [B]
    t = t_idx / dt_sections.astype(jnp.float32)              # [B]

    t = jnp.minimum(t, 1.0 - dt)

    # override bằng force_t nếu cần, cũng dùng jnp.where
    force_t_val = jnp.asarray(force_t, dtype=jnp.float32)
    t = jnp.where(
        force_t_val == -1.0,
        t,
        jnp.ones_like(t, dtype=jnp.float32) * force_t_val,
    )

    t_full = t[:, None, None, None]      # [B,1,1,1]
    dt_full = dt[:, None, None, None]    # [B,1,1,1]

    # --- 4) Flow Matching analytic: (x_t, x_{t+d}), v_t ---
    x_1 = images
    x_0 = jax.random.normal(noise_key, images.shape)
    eps = 1e-5

    # x_t = (1 - (1-eps)*t) x0 + t x1
    x_t = (1.0 - (1.0 - eps) * t_full) * x_0 + t_full * x_1

    # x_{t+d} = (1 - (1-eps)(t+d)) x0 + (t+d) x1
    t_plus_d = (t + dt)[:, None, None, None]
    x_t_plus_d = (1.0 - (1.0 - eps) * t_plus_d) * x_0 + t_plus_d * x_1

    # v_t = (x_{t+d} - x_t) / d
    v_t = (x_t_plus_d - x_t) / dt_full

    # --- 5) Info logging ---
    info['bootstrap_ratio'] = jnp.array(0.0)
    info['v_magnitude'] = jnp.sqrt(jnp.mean(jnp.square(v_t)))

    return x_t, v_t, t, dt_base, labels_dropped, info
