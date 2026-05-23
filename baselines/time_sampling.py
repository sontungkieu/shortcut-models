import jax
import jax.numpy as jnp


def sample_flow_t(FLAGS, key, batch_size: int):
    mode = str(FLAGS.model.get("t_sampling", "discrete-dt")).lower().replace("_", "-")
    denoise_steps = int(FLAGS.model["denoise_timesteps"])

    if mode in ("discrete-dt", "uniform-discrete", "discrete-uniform"):
        t = jax.random.randint(key, (batch_size,), minval=0, maxval=denoise_steps).astype(jnp.float32)
        return t / denoise_steps

    if mode in ("uniform", "uniform-continuous", "continuous-uniform"):
        return jax.random.uniform(key, (batch_size,), dtype=jnp.float32)

    if mode in ("beta", "beta-continuous"):
        alpha = float(FLAGS.model.get("t_beta_alpha", 1.0))
        beta = float(FLAGS.model.get("t_beta_beta", 1.0))
        return jax.random.beta(key, alpha, beta, shape=(batch_size,), dtype=jnp.float32)

    if mode in ("beta-discrete", "discrete-beta"):
        alpha = float(FLAGS.model.get("t_beta_alpha", 1.0))
        beta = float(FLAGS.model.get("t_beta_beta", 1.0))
        t_cont = jax.random.beta(key, alpha, beta, shape=(batch_size,), dtype=jnp.float32)
        t_idx = jnp.floor(t_cont * denoise_steps)
        t_idx = jnp.clip(t_idx, 0, denoise_steps - 1)
        return t_idx.astype(jnp.float32) / denoise_steps

    raise ValueError(
        "Unknown model.t_sampling "
        f"{FLAGS.model.get('t_sampling')!r}; expected discrete-dt, uniform, beta, or beta-discrete."
    )
