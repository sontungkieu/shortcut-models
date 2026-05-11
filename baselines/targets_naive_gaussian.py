import jax
import jax.numpy as jnp
import numpy as np


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1):
    label_key, time_key, noise_key = jax.random.split(key, 3)
    info = {}

    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model["class_dropout_prob"], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model["num_classes"], labels)
    info["dropped_ratio"] = jnp.mean(labels_dropped == FLAGS.model["num_classes"])

    t = jax.random.randint(
        time_key,
        (images.shape[0],),
        minval=0,
        maxval=FLAGS.model["denoise_timesteps"],
    ).astype(jnp.float32)
    t /= FLAGS.model["denoise_timesteps"]
    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]

    if "latent" in FLAGS.dataset_name:
        x_0 = images[..., : images.shape[-1] // 2]
        x_1 = images[..., images.shape[-1] // 2 :]
    else:
        x_1 = images
        x_0 = jax.random.normal(noise_key, images.shape)

    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = np.log2(FLAGS.model["denoise_timesteps"]).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    info["x0_magnitude"] = jnp.sqrt(jnp.mean(jnp.square(x_0)))
    info["x1_magnitude"] = jnp.sqrt(jnp.mean(jnp.square(x_1)))
    info["v_magnitude_target"] = jnp.sqrt(jnp.mean(jnp.square(v_t)))
    info["x0_variance"] = jnp.mean(jnp.var(x_0, axis=0))
    info["x1_variance"] = jnp.mean(jnp.var(x_1, axis=0))
    info["v_variance_target"] = jnp.mean(jnp.var(v_t, axis=0))
    info["x0_second_moment"] = jnp.mean(jnp.square(x_0))
    info["x1_second_moment"] = jnp.mean(jnp.square(x_1))
    info["v_second_moment_target"] = jnp.mean(jnp.square(v_t))

    return x_t, v_t, t, dt_base, labels_dropped, info
