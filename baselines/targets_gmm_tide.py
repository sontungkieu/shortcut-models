import jax
import jax.numpy as jnp
import numpy as np

from gmm_utils import (
    component_params_from_ids,
    flatten_latents,
    posterior_from_stats,
    sample_prior_components,
)


def _component_params_from_topk(gmm_state, component_ids, latent_shape, eps: float = 1e-6):
    batch_size, topk = component_ids.shape
    flat_ids = jnp.reshape(component_ids, (-1,))
    mu, sigma = component_params_from_ids(gmm_state, flat_ids, latent_shape, eps=eps)
    out_shape = (batch_size, topk) + tuple(latent_shape)
    return jnp.reshape(mu, out_shape), jnp.reshape(sigma, out_shape)


def _weight_view(weights, latent_shape):
    return jnp.reshape(weights, weights.shape + (1,) * len(tuple(latent_shape)))


def _apply_router(router_state, x):
    return router_state["model_def"].apply({"params": router_state["params"]}, x, train=False)


def make_tide_source(
    key,
    gmm_state,
    router_state,
    batch_size: int,
    latent_shape,
    topk: int,
    temperature: float = 1.0,
    eps: float = 1e-6,
):
    if gmm_state is None:
        raise ValueError("gmm-tide requires gmm_state loaded from --model.gmm_stats_path")
    if router_state is None:
        raise ValueError("gmm-tide requires router_state loaded from --model.gmm_router_path")

    base_key, sample_key = jax.random.split(key)
    x0_base, base_mu, base_sigma, base_ids = sample_prior_components(
        base_key,
        gmm_state,
        batch_size,
        latent_shape,
        eps=eps,
    )

    logits = _apply_router(router_state, x0_base)
    logits = logits / jnp.maximum(jnp.asarray(temperature, dtype=logits.dtype), eps)
    q_phi = jax.nn.softmax(logits, axis=-1)
    q_phi = jax.lax.stop_gradient(q_phi)

    topk = min(int(topk), int(q_phi.shape[-1]))
    top_probs, top_ids = jax.lax.top_k(q_phi, topk)
    top_weights = top_probs / jnp.maximum(jnp.sum(top_probs, axis=-1, keepdims=True), eps)

    top_mu, top_sigma = _component_params_from_topk(gmm_state, top_ids, latent_shape, eps=eps)
    noise = jax.random.normal(sample_key, top_mu.shape, dtype=top_mu.dtype)
    top_samples = top_mu + top_sigma * noise
    weights = _weight_view(top_weights, latent_shape)

    x0_tide = jnp.sum(weights * top_samples, axis=1)
    mu_tide = jnp.sum(weights * top_mu, axis=1)
    sigma_tide = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(weights * top_sigma), axis=1), eps))

    q_gmm_base, _, _ = posterior_from_stats(gmm_state, flatten_latents(x0_base), eps=eps)
    q_safe = jnp.maximum(q_phi, eps)
    q_gmm_safe = jnp.maximum(q_gmm_base, eps)
    top1_ids = top_ids[:, 0]
    counts = jnp.bincount(top1_ids, length=q_phi.shape[-1])
    usage = counts / jnp.maximum(batch_size, 1)
    usage_safe = jnp.maximum(usage, eps)

    info = {
        "router/topk": jnp.asarray(topk, dtype=jnp.float32),
        "router/topk_mass": jnp.mean(jnp.sum(top_probs, axis=-1)),
        "router/entropy": jnp.mean(-jnp.sum(q_safe * jnp.log(q_safe), axis=-1)),
        "router/top1_prob_mean": jnp.mean(jnp.max(q_phi, axis=-1)),
        "router/top1_agreement_to_gmm_base": jnp.mean(jnp.argmax(q_phi, axis=-1) == jnp.argmax(q_gmm_base, axis=-1)),
        "router/kl_to_gmm_base": jnp.mean(jnp.sum(q_gmm_safe * (jnp.log(q_gmm_safe) - jnp.log(q_safe)), axis=-1)),
        "router/assign_entropy": -jnp.sum(usage_safe * jnp.log(usage_safe)),
        "router/assign_max_frac": jnp.max(usage),
        "router/num_unique_clusters": jnp.sum(counts > 0),
        "tide/base_component_match_top1": jnp.mean(base_ids == top1_ids),
        "tide/x0_base_magnitude": jnp.sqrt(jnp.mean(jnp.square(x0_base))),
        "tide/x0_tide_magnitude": jnp.sqrt(jnp.mean(jnp.square(x0_tide))),
        "tide/mu_tide_magnitude": jnp.sqrt(jnp.mean(jnp.square(mu_tide))),
        "tide/sigma_tide_magnitude": jnp.sqrt(jnp.mean(jnp.square(sigma_tide))),
        "tide/base_mu_magnitude": jnp.sqrt(jnp.mean(jnp.square(base_mu))),
        "tide/base_sigma_magnitude": jnp.sqrt(jnp.mean(jnp.square(base_sigma))),
    }
    return x0_tide, mu_tide, sigma_tide, info


def get_targets(
    FLAGS,
    key,
    train_state,
    images,
    labels,
    force_t=-1,
    force_dt=-1,
    gmm_state=None,
    router_state=None,
):
    del train_state
    del force_dt
    label_key, time_key, source_key = jax.random.split(key, 3)
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

    x_0, gmm_mu, gmm_sigma, tide_info = make_tide_source(
        source_key,
        gmm_state,
        router_state,
        images.shape[0],
        x_1.shape[1:],
        topk=FLAGS.model["gmm_router_topk"],
        temperature=FLAGS.model["gmm_router_temperature"],
    )

    x_t = (1 - (1 - 1e-5) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = np.log2(FLAGS.model["denoise_timesteps"]).astype(jnp.int32)
    dt_base = jnp.ones(images.shape[0], dtype=jnp.int32) * dt_flow

    info.update(tide_info)
    info["x0_magnitude"] = jnp.sqrt(jnp.mean(jnp.square(x_0)))
    info["x1_magnitude"] = jnp.sqrt(jnp.mean(jnp.square(x_1)))
    info["v_magnitude_target"] = jnp.sqrt(jnp.mean(jnp.square(v_t)))
    info["x0_variance"] = jnp.mean(jnp.var(x_0, axis=0))
    info["x1_variance"] = jnp.mean(jnp.var(x_1, axis=0))
    info["v_variance_target"] = jnp.mean(jnp.var(v_t, axis=0))
    info["x0_second_moment"] = jnp.mean(jnp.square(x_0))
    info["x1_second_moment"] = jnp.mean(jnp.square(x_1))
    info["v_second_moment_target"] = jnp.mean(jnp.square(v_t))

    return x_t, v_t, t, dt_base, labels_dropped, info, gmm_mu, gmm_sigma
