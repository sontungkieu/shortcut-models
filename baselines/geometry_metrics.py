import jax.numpy as jnp


def flatten_samples(x):
    return jnp.reshape(x, (x.shape[0], -1))


def batch_cosine(a, b, eps: float = 1e-8):
    a_flat = flatten_samples(a)
    b_flat = flatten_samples(b)
    dot = jnp.sum(a_flat * b_flat, axis=-1)
    a_norm = jnp.sqrt(jnp.maximum(jnp.sum(a_flat * a_flat, axis=-1), eps))
    b_norm = jnp.sqrt(jnp.maximum(jnp.sum(b_flat * b_flat, axis=-1), eps))
    return dot / jnp.maximum(a_norm * b_norm, eps)


def cosine_summary(prefix: str, cosine):
    cosine = jnp.clip(cosine, -1.0, 1.0)
    return {
        f"{prefix}/cosine_mean": jnp.mean(cosine),
        f"{prefix}/cosine_std": jnp.std(cosine),
        f"{prefix}/cosine_min": jnp.min(cosine),
        f"{prefix}/cosine_max": jnp.max(cosine),
        f"{prefix}/angle_rad_mean": jnp.mean(jnp.arccos(cosine)),
    }


def pair_geometry_metrics(prefix: str, a, b):
    return cosine_summary(prefix, batch_cosine(a, b))
