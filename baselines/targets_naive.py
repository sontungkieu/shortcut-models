import jax
import jax.numpy as jnp
import numpy as np

from baselines.geometry_metrics import pair_geometry_metrics
from gmm_utils import infer_component_params, mixture_mean_from_stats, sample_components


def get_targets(FLAGS, key, train_state, images, labels, force_t=-1, force_dt=-1, gmm_state=None):
    if gmm_state is None:
        raise ValueError("GMM-matched targets require gmm_state loaded from --model.gmm_stats_path")

    centered_source = FLAGS.model["train_type"] == "gmm-centered"
    center_scale = float(FLAGS.model["gmm_source_center_scale"]) if centered_source else None

    label_key, time_key, x0_key = jax.random.split(key, 3)
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
        x_1 = images[..., images.shape[-1] // 2 :]
    else:
        x_1 = images

    component_ids, q, _, _, _ = infer_component_params(gmm_state, x_1)
    x_0, gmm_mu, gmm_sigma = sample_components(
        x0_key,
        gmm_state,
        component_ids,
        x_1.shape[1:],
        center_scale=center_scale,
    )

    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = np.log2(FLAGS.model["denoise_timesteps"]).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    q_safe = jnp.maximum(q, 1e-8)
    q_entropy = -jnp.sum(q_safe * jnp.log(q_safe), axis=1)
    top1_prob = jnp.max(q, axis=1)
    hard_counts = jnp.bincount(component_ids, length=gmm_state["pi"].shape[0])
    info["gmm/assign_entropy"] = -jnp.sum((hard_counts / images.shape[0]) * jnp.log(jnp.maximum(hard_counts / images.shape[0], 1e-8)))
    info["gmm/assign_max_frac"] = jnp.max(hard_counts) / images.shape[0]
    info["gmm/num_unique_clusters"] = jnp.sum(hard_counts > 0)
    info["gmm/q_entropy_mean"] = jnp.mean(q_entropy)
    info["gmm/q_top1_prob_mean"] = jnp.mean(top1_prob)
    info["gmm/source_is_centered"] = jnp.asarray(float(centered_source), dtype=jnp.float32)
    if centered_source:
        mixture_mean = mixture_mean_from_stats(gmm_state, x_1.shape[1:])
        info["gmm/source_center_scale"] = jnp.asarray(center_scale, dtype=jnp.float32)
        info["gmm/fit_mixture_mean_magnitude"] = jnp.sqrt(jnp.mean(jnp.square(mixture_mean)))
        info["gmm/source_component_mean_magnitude"] = jnp.sqrt(jnp.mean(jnp.square(gmm_mu)))
    info["x0_magnitude"] = jnp.sqrt(jnp.mean(jnp.square(x_0)))
    info["x1_magnitude"] = jnp.sqrt(jnp.mean(jnp.square(x_1)))
    info["v_magnitude_target"] = jnp.sqrt(jnp.mean(jnp.square(v_t)))
    info["x0_variance"] = jnp.mean(jnp.var(x_0, axis=0))
    info["x1_variance"] = jnp.mean(jnp.var(x_1, axis=0))
    info["v_variance_target"] = jnp.mean(jnp.var(v_t, axis=0))
    info["x0_second_moment"] = jnp.mean(jnp.square(x_0))
    info["x1_second_moment"] = jnp.mean(jnp.square(x_1))
    info["v_second_moment_target"] = jnp.mean(jnp.square(v_t))
    info.update(pair_geometry_metrics("geometry/x0_x1", x_0, x_1))
    info.update(pair_geometry_metrics("geometry/v_x1", v_t, x_1))
    info.update(pair_geometry_metrics("geometry/v_x0", v_t, x_0))

    return x_t, v_t, t, dt_base, labels_dropped, info, gmm_mu, gmm_sigma
