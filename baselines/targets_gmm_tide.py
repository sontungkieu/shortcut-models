import jax
import jax.numpy as jnp
import numpy as np

from baselines.geometry_metrics import batch_cosine, cosine_summary, flatten_samples, pair_geometry_metrics
from baselines.time_sampling import sample_flow_t
from gmm_utils import (
    component_params_from_ids,
    flatten_latents,
    mixture_mean_from_stats,
    posterior_from_stats,
    sample_components,
    sample_prior_components,
)


def _component_params_from_topk(gmm_state, component_ids, latent_shape, eps: float = 1e-6):
    batch_size, topk = component_ids.shape
    flat_ids = jnp.reshape(component_ids, (-1,))
    mu, sigma = component_params_from_ids(gmm_state, flat_ids, latent_shape, eps=eps)
    out_shape = (batch_size, topk) + tuple(latent_shape)
    return jnp.reshape(mu, out_shape), jnp.reshape(sigma, out_shape)


def _sample_components_from_topk(key, gmm_state, component_ids, latent_shape, eps: float = 1e-6):
    batch_size, topk = component_ids.shape
    flat_ids = jnp.reshape(component_ids, (-1,))
    samples, mu, sigma = sample_components(key, gmm_state, flat_ids, latent_shape, eps=eps)
    out_shape = (batch_size, topk) + tuple(latent_shape)
    return jnp.reshape(samples, out_shape), jnp.reshape(mu, out_shape), jnp.reshape(sigma, out_shape)


def _component_params_all(gmm_state, latent_shape, eps: float = 1e-6):
    component_ids = jnp.arange(gmm_state["pi"].shape[0], dtype=jnp.int32)
    return component_params_from_ids(gmm_state, component_ids, latent_shape, eps=eps)


def _weight_view(weights, latent_shape):
    return jnp.reshape(weights, weights.shape + (1,) * len(tuple(latent_shape)))


def _apply_router(router_state, x, params=None):
    if params is None:
        params = router_state["params"]
    return router_state["model_def"].apply({"params": params}, x, train=False)


def _sample_gumbel(key, shape, dtype, eps: float = 1e-6):
    u = jax.random.uniform(key, shape, dtype=dtype, minval=eps, maxval=1.0 - eps)
    return -jnp.log(-jnp.log(u))


def _scatter_topk_weights(top_ids, top_weights, num_modes: int):
    one_hot = jax.nn.one_hot(top_ids, num_modes, dtype=top_weights.dtype)
    return jnp.sum(one_hot * top_weights[:, :, None], axis=1)


def _make_full_mixture_source(key, gmm_state, latent_shape, weights, eps: float = 1e-6):
    num_modes = int(gmm_state["pi"].shape[0])
    component_ids = jnp.tile(jnp.arange(num_modes, dtype=jnp.int32)[None, :], (weights.shape[0], 1))
    samples, mu_all, sigma_all = _sample_components_from_topk(
        key,
        gmm_state,
        component_ids,
        latent_shape,
        eps=eps,
    )
    view = _weight_view(weights, latent_shape)
    x0_tide = jnp.sum(view * samples, axis=1)
    mu_tide = jnp.sum(view * mu_all, axis=1)
    sigma_tide = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(view * sigma_all), axis=1), eps))
    return x0_tide, mu_tide, sigma_tide


def _topk_center_geometry(top_mu, top_weights, mu_tide, eps: float = 1e-8):
    top_mu_flat = jnp.reshape(top_mu, top_mu.shape[:2] + (-1,))
    top_mu_norm = jnp.sqrt(jnp.maximum(jnp.sum(top_mu_flat * top_mu_flat, axis=-1), eps))
    top_unit = top_mu_flat / top_mu_norm[..., None]
    pair_cos = jnp.einsum("bkd,bld->bkl", top_unit, top_unit)
    topk = top_mu.shape[1]
    offdiag = 1.0 - jnp.eye(topk, dtype=pair_cos.dtype)
    offdiag_count = jnp.maximum(jnp.asarray(topk * (topk - 1), dtype=pair_cos.dtype), 1.0)
    offdiag_cos_mean = jnp.sum(pair_cos * offdiag[None, ...], axis=(1, 2)) / offdiag_count
    offdiag_cos_mean = jnp.where(topk > 1, offdiag_cos_mean, jnp.ones_like(offdiag_cos_mean))
    pair_cos_min = jnp.min(pair_cos + (1.0 - offdiag[None, ...]) * 2.0, axis=(1, 2))
    pair_cos_min = jnp.where(topk > 1, pair_cos_min, jnp.ones_like(pair_cos_min))

    mu_tide_flat = flatten_samples(mu_tide)
    mu_tide_norm = jnp.sqrt(jnp.maximum(jnp.sum(mu_tide_flat * mu_tide_flat, axis=-1), eps))
    mu_tide_unit = mu_tide_flat / mu_tide_norm[:, None]
    center_cos = jnp.einsum("bkd,bd->bk", top_unit, mu_tide_unit)
    weighted_center_cos = jnp.sum(top_weights * center_cos, axis=-1)
    angular_dispersion = jnp.sum(top_weights * (1.0 - center_cos), axis=-1)

    return {
        "tide/topk_mu_pair_cosine_mean": jnp.mean(offdiag_cos_mean),
        "tide/topk_mu_pair_cosine_min": jnp.mean(pair_cos_min),
        "tide/topk_mu_to_tide_cosine_mean": jnp.mean(weighted_center_cos),
        "tide/topk_mu_angular_dispersion": jnp.mean(angular_dispersion),
    }


def make_tide_source(
    key,
    gmm_state,
    router_state,
    batch_size: int,
    latent_shape,
    topk: int,
    temperature: float = 1.0,
    eps: float = 1e-6,
    router_params=None,
    stop_router_gradient: bool = True,
    gradient_mode: str = "topk",
    gumbel_tau: float = 1.0,
    source_mode: str = "weighted",
    routing_policy: str = "router",
    shift_mixture_mean: bool = False,
    center_scale: float = 1.0,
):
    if gmm_state is None:
        raise ValueError("gmm-tide requires gmm_state loaded from --model.gmm_stats_path")
    if router_state is None:
        raise ValueError("gmm-tide requires router_state loaded from --model.gmm_router_path")

    source_mode = str(source_mode).lower().replace("-", "_")
    if source_mode not in ("weighted", "hard_top1", "sample_topk"):
        raise ValueError("source_mode must be weighted, hard_top1, or sample_topk")
    routing_policy = str(routing_policy).lower().replace("-", "_")
    if routing_policy not in ("router", "gmm_oracle", "matched_random"):
        raise ValueError("routing_policy must be router, gmm_oracle, or matched_random")
    if routing_policy != "router" and gradient_mode != "topk":
        raise ValueError("gmm_oracle and matched_random routing require gradient_mode=topk")
    center_scale = float(center_scale)
    if not np.isfinite(center_scale) or center_scale < 0.0:
        raise ValueError("center_scale must be finite and non-negative")
    base_key, sample_key, route_key, select_key = jax.random.split(key, 4)
    x0_base, base_mu, base_sigma, base_ids = sample_prior_components(
        base_key,
        gmm_state,
        batch_size,
        latent_shape,
        eps=eps,
    )

    logits = _apply_router(router_state, x0_base, params=router_params)
    logits = logits / jnp.maximum(jnp.asarray(temperature, dtype=logits.dtype), eps)
    q_phi = jax.nn.softmax(logits, axis=-1)
    q_gmm_base, _, _ = posterior_from_stats(gmm_state, flatten_latents(x0_base), eps=eps)
    if routing_policy == "gmm_oracle":
        q_route_soft = q_gmm_base
    elif routing_policy == "matched_random":
        q_route_soft = q_phi[jax.random.permutation(route_key, batch_size)]
    else:
        q_route_soft = q_phi
    if routing_policy == "router" and gradient_mode == "gumbel_st":
        tau = jnp.maximum(jnp.asarray(gumbel_tau, dtype=logits.dtype), eps)
        q_route_soft = jax.nn.softmax((logits + _sample_gumbel(route_key, logits.shape, logits.dtype, eps=eps)) / tau, axis=-1)
    if stop_router_gradient:
        q_phi = jax.lax.stop_gradient(q_phi)
        q_route_soft = jax.lax.stop_gradient(q_route_soft)

    topk = min(int(topk), int(q_phi.shape[-1]))
    top_probs, top_ids = jax.lax.top_k(q_route_soft, topk)
    top_weights = top_probs / jnp.maximum(jnp.sum(top_probs, axis=-1, keepdims=True), eps)
    top_mu, top_sigma = _component_params_from_topk(gmm_state, top_ids, latent_shape, eps=eps)

    if source_mode == "hard_top1":
        x0_tide, chosen_mu, chosen_sigma = sample_components(
            sample_key,
            gmm_state,
            top_ids[:, 0],
            latent_shape,
            eps=eps,
        )
        mu_tide = chosen_mu
        sigma_tide = chosen_sigma
    elif source_mode == "sample_topk":
        select_logits = jnp.log(jnp.maximum(top_weights, eps))
        chosen_rel = jax.random.categorical(select_key, select_logits, axis=-1)
        batch_ids = jnp.arange(batch_size)
        chosen_ids = top_ids[batch_ids, chosen_rel]
        x0_tide, chosen_mu, chosen_sigma = sample_components(
            sample_key,
            gmm_state,
            chosen_ids,
            latent_shape,
            eps=eps,
        )
        mu_tide = chosen_mu
        sigma_tide = chosen_sigma
    elif gradient_mode in ("straight_through_full", "gumbel_st"):
        hard_weights = _scatter_topk_weights(top_ids, top_weights, q_phi.shape[-1])
        route_weights = q_route_soft + jax.lax.stop_gradient(hard_weights - q_route_soft)
        if stop_router_gradient:
            route_weights = hard_weights
        x0_tide, mu_tide, sigma_tide = _make_full_mixture_source(
            sample_key,
            gmm_state,
            latent_shape,
            route_weights,
            eps=eps,
        )
    else:
        top_samples, top_mu, top_sigma = _sample_components_from_topk(
            sample_key,
            gmm_state,
            top_ids,
            latent_shape,
            eps=eps,
        )
        weights = _weight_view(top_weights, latent_shape)

        x0_tide = jnp.sum(weights * top_samples, axis=1)
        mu_tide = jnp.sum(weights * top_mu, axis=1)
        sigma_tide = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(weights * top_sigma), axis=1), eps))

    mixture_mean = mixture_mean_from_stats(gmm_state, latent_shape, eps=eps)
    if center_scale != 1.0:
        # Keep the selected component residual (and therefore covariance)
        # unchanged while scaling only its offset from the fitted mixture mean.
        source_residual = x0_tide - mu_tide
        mu_tide = mixture_mean + center_scale * (mu_tide - mixture_mean)
        x0_tide = source_residual + mu_tide
        top_mu = mixture_mean + center_scale * (top_mu - mixture_mean)
    if shift_mixture_mean:
        # Preserve the original MOE2 router input and routing decision. Shift
        # only the final TIDE source and its mean conditioning.
        x0_tide = x0_tide - mixture_mean
        mu_tide = mu_tide - mixture_mean
        top_mu = top_mu - mixture_mean

    q_gmm_tide, _, _ = posterior_from_stats(gmm_state, flatten_latents(x0_tide), eps=eps)
    logits_tide = _apply_router(router_state, x0_tide, params=router_params)
    logits_tide = logits_tide / jnp.maximum(jnp.asarray(temperature, dtype=logits_tide.dtype), eps)
    q_phi_tide = jax.nn.softmax(logits_tide, axis=-1)
    q_gmm_tide = jax.lax.stop_gradient(q_gmm_tide)
    if stop_router_gradient:
        q_phi_tide = jax.lax.stop_gradient(q_phi_tide)

    q_safe = jnp.maximum(q_phi, eps)
    q_tide_safe = jnp.maximum(q_phi_tide, eps)
    q_route_safe = jnp.maximum(q_route_soft, eps)
    q_gmm_safe = jnp.maximum(q_gmm_base, eps)
    q_gmm_tide_safe = jnp.maximum(q_gmm_tide, eps)
    top1_ids = top_ids[:, 0]
    gmm_tide_top1_ids = jnp.argmax(q_gmm_tide, axis=-1)
    counts = jnp.bincount(top1_ids, length=q_phi.shape[-1])
    usage = counts / jnp.maximum(batch_size, 1)
    usage_safe = jnp.maximum(usage, eps)
    soft_usage = jnp.mean(q_route_soft, axis=0)
    soft_usage_safe = jnp.maximum(soft_usage, eps)
    usage_entropy = -jnp.sum(usage_safe * jnp.log(usage_safe))
    usage_entropy_normalized = usage_entropy / jnp.log(jnp.asarray(q_phi.shape[-1], dtype=jnp.float32))
    usage_kl_to_uniform = jnp.log(jnp.asarray(q_phi.shape[-1], dtype=jnp.float32)) - usage_entropy
    soft_usage_entropy = -jnp.sum(soft_usage_safe * jnp.log(soft_usage_safe))
    soft_usage_entropy_normalized = soft_usage_entropy / jnp.log(jnp.asarray(q_phi.shape[-1], dtype=jnp.float32))
    soft_usage_kl_to_uniform = jnp.log(jnp.asarray(q_phi.shape[-1], dtype=jnp.float32)) - soft_usage_entropy
    router_kl_to_gmm_base = jnp.mean(jnp.sum(q_gmm_safe * (jnp.log(q_gmm_safe) - jnp.log(q_safe)), axis=-1))
    route_kl_to_gmm_base = jnp.mean(
        jnp.sum(q_gmm_safe * (jnp.log(q_gmm_safe) - jnp.log(q_route_safe)), axis=-1)
    )
    router_kl_to_gmm_tide = jnp.mean(jnp.sum(q_gmm_tide_safe * (jnp.log(q_gmm_tide_safe) - jnp.log(q_tide_safe)), axis=-1))
    route_topk_dist = _scatter_topk_weights(top_ids, top_weights, q_phi.shape[-1])
    route_topk_safe = jnp.maximum(route_topk_dist, eps)
    router_kl_gmm_tide_to_topk = jnp.mean(
        jnp.sum(q_gmm_tide_safe * (jnp.log(q_gmm_tide_safe) - jnp.log(route_topk_safe)), axis=-1)
    )
    gmm_tide_entropy = -jnp.sum(q_gmm_tide_safe * jnp.log(q_gmm_tide_safe), axis=-1)
    gmm_tide_topk_hit = jnp.max(jnp.sum(jax.nn.one_hot(top_ids, q_phi.shape[-1]) * jax.nn.one_hot(gmm_tide_top1_ids, q_phi.shape[-1])[:, None, :], axis=-1), axis=-1)

    info = {
        "router/topk": jnp.asarray(topk, dtype=jnp.float32),
        "router/topk_mass": jnp.mean(jnp.sum(top_probs, axis=-1)),
        "router/entropy": jnp.mean(-jnp.sum(q_safe * jnp.log(q_safe), axis=-1)),
        "router/route_entropy": jnp.mean(-jnp.sum(q_route_safe * jnp.log(q_route_safe), axis=-1)),
        "router/route_top1_prob_mean": jnp.mean(jnp.max(q_route_soft, axis=-1)),
        "router/gumbel_tau": jnp.asarray(gumbel_tau, dtype=jnp.float32),
        "router/mode_is_topk": jnp.asarray(gradient_mode == "topk", dtype=jnp.float32),
        "router/mode_is_st_full": jnp.asarray(gradient_mode == "straight_through_full", dtype=jnp.float32),
        "router/mode_is_gumbel_st": jnp.asarray(gradient_mode == "gumbel_st", dtype=jnp.float32),
        "router/routing_policy_is_router": jnp.asarray(routing_policy == "router", dtype=jnp.float32),
        "router/routing_policy_is_gmm_oracle": jnp.asarray(routing_policy == "gmm_oracle", dtype=jnp.float32),
        "router/routing_policy_is_matched_random": jnp.asarray(routing_policy == "matched_random", dtype=jnp.float32),
        "tide/source_mode_is_weighted": jnp.asarray(source_mode == "weighted", dtype=jnp.float32),
        "tide/source_mode_is_hard_top1": jnp.asarray(source_mode == "hard_top1", dtype=jnp.float32),
        "tide/source_mode_is_sample_topk": jnp.asarray(source_mode == "sample_topk", dtype=jnp.float32),
        "tide/source_shift_mixture_mean": jnp.asarray(float(shift_mixture_mean), dtype=jnp.float32),
        "tide/source_center_scale": jnp.asarray(center_scale, dtype=jnp.float32),
        "tide/fit_mixture_mean_magnitude": jnp.sqrt(jnp.mean(jnp.square(mixture_mean))),
        "router/top1_prob_mean": jnp.mean(jnp.max(q_phi, axis=-1)),
        "router/top1_agreement_to_gmm_base": jnp.mean(jnp.argmax(q_phi, axis=-1) == jnp.argmax(q_gmm_base, axis=-1)),
        "router/route_top1_agreement_to_gmm_base": jnp.mean(
            jnp.argmax(q_route_soft, axis=-1) == jnp.argmax(q_gmm_base, axis=-1)
        ),
        "router/kl_to_gmm_base": router_kl_to_gmm_base,
        "router/route_kl_to_gmm_base": route_kl_to_gmm_base,
        "router/kl_to_gmm_tide": router_kl_to_gmm_tide,
        "router/kl_gmm_tide_to_topk": router_kl_gmm_tide_to_topk,
        "router/tide_top1_agreement_to_gmm_tide": jnp.mean(jnp.argmax(q_phi_tide, axis=-1) == gmm_tide_top1_ids),
        "router/assign_entropy": usage_entropy,
        "router/usage_entropy_normalized": usage_entropy_normalized,
        "router/usage_kl_to_uniform": usage_kl_to_uniform,
        "router/soft_usage_entropy": soft_usage_entropy,
        "router/soft_usage_entropy_normalized": soft_usage_entropy_normalized,
        "router/soft_usage_kl_to_uniform": soft_usage_kl_to_uniform,
        "router/soft_assign_max_frac": jnp.max(soft_usage),
        "router/assign_max_frac": jnp.max(usage),
        "router/num_unique_clusters": jnp.sum(counts > 0),
        "tide/base_component_match_top1": jnp.mean(base_ids == top1_ids),
        "tide/gmm_tide_top1_in_topk": jnp.mean(gmm_tide_topk_hit),
        "tide/gmm_base_tide_top1_agreement": jnp.mean(jnp.argmax(q_gmm_base, axis=-1) == gmm_tide_top1_ids),
        "tide/gmm_tide_entropy": jnp.mean(gmm_tide_entropy),
        "tide/gmm_tide_entropy_normalized": jnp.mean(gmm_tide_entropy) / jnp.log(jnp.asarray(q_phi.shape[-1], dtype=jnp.float32)),
        "tide/gmm_tide_top1_prob_mean": jnp.mean(jnp.max(q_gmm_tide, axis=-1)),
        "tide/x0_base_magnitude": jnp.sqrt(jnp.mean(jnp.square(x0_base))),
        "tide/x0_tide_magnitude": jnp.sqrt(jnp.mean(jnp.square(x0_tide))),
        "tide/mu_tide_magnitude": jnp.sqrt(jnp.mean(jnp.square(mu_tide))),
        "tide/sigma_tide_magnitude": jnp.sqrt(jnp.mean(jnp.square(sigma_tide))),
        "tide/base_mu_magnitude": jnp.sqrt(jnp.mean(jnp.square(base_mu))),
        "tide/base_sigma_magnitude": jnp.sqrt(jnp.mean(jnp.square(base_sigma))),
    }
    info.update(_topk_center_geometry(top_mu, top_weights, mu_tide, eps=eps))
    info.update(cosine_summary("tide/x0_tide_base", batch_cosine(x0_tide, x0_base, eps=eps)))
    info.update(cosine_summary("tide/mu_tide_base_mu", batch_cosine(mu_tide, base_mu, eps=eps)))
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
    router_params=None,
    stop_router_gradient: bool = True,
):
    del train_state
    del force_dt
    label_key, time_key, source_key = jax.random.split(key, 3)
    info = {}

    labels_dropout = jax.random.bernoulli(label_key, FLAGS.model["class_dropout_prob"], (labels.shape[0],))
    labels_dropped = jnp.where(labels_dropout, FLAGS.model["num_classes"], labels)
    info["dropped_ratio"] = jnp.mean(labels_dropped == FLAGS.model["num_classes"])

    t = sample_flow_t(FLAGS, time_key, images.shape[0])
    force_t_vec = jnp.ones(images.shape[0], dtype=jnp.float32) * force_t
    t = jnp.where(force_t_vec != -1, force_t_vec, t)
    t_full = t[:, None, None, None]
    info["t_mean"] = jnp.mean(t)
    info["t_variance"] = jnp.var(t)
    info["t_min"] = jnp.min(t)
    info["t_max"] = jnp.max(t)

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
        router_params=router_params,
        stop_router_gradient=stop_router_gradient,
        gradient_mode=FLAGS.model["gmm_router_gradient_mode"],
        gumbel_tau=FLAGS.model["gmm_router_gumbel_tau"],
        source_mode=FLAGS.model.get("gmm_router_source_mode", "weighted"),
        routing_policy=FLAGS.model.get("gmm_router_routing_policy", "router"),
        shift_mixture_mean=bool(FLAGS.model.get("gmm_source_shift_mean", 0)),
        center_scale=float(FLAGS.model.get("gmm_source_center_scale", 1.0)),
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
    info.update(pair_geometry_metrics("geometry/x0_x1", x_0, x_1))
    info.update(pair_geometry_metrics("geometry/v_x1", v_t, x_1))
    info.update(pair_geometry_metrics("geometry/v_x0", v_t, x_0))

    return x_t, v_t, t, dt_base, labels_dropped, info, gmm_mu, gmm_sigma
