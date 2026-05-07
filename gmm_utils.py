import json
import os
from typing import Callable, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


LOG_2PI = np.log(2.0 * np.pi)


def chunk_slices(n_items: int, chunk_size: int) -> Iterable[slice]:
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, n_items, chunk_size):
        yield slice(start, min(start + chunk_size, n_items))


def flatten_latents(x):
    return jnp.reshape(x, (x.shape[0], -1))


def flatten_latents_np(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (x.shape[0], -1))


def _logsumexp_np(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    out = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _diag_gmm_const_np(pi: np.ndarray, mu: np.ndarray, var: np.ndarray, eps: float) -> np.ndarray:
    inv_var = 1.0 / np.maximum(var, eps)
    return (
        np.log(np.maximum(pi, eps))
        - 0.5 * np.sum(np.log(2.0 * np.pi * np.maximum(var, eps)), axis=1)
        - 0.5 * np.sum(mu * mu * inv_var, axis=1)
    )


def diag_gmm_log_prob_np(
    x: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pi = np.asarray(pi, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)
    var = np.maximum(np.asarray(var, dtype=np.float32), eps)
    inv_var = 1.0 / var
    const = _diag_gmm_const_np(pi, mu, var, eps)
    return -0.5 * (x * x) @ inv_var.T + x @ (mu * inv_var).T + const


def diag_gmm_log_prob(
    x,
    pi,
    mu,
    var,
    eps: float = 1e-6,
):
    x = jnp.asarray(x, dtype=jnp.float32)
    pi = jnp.asarray(pi, dtype=jnp.float32)
    mu = jnp.asarray(mu, dtype=jnp.float32)
    var = jnp.maximum(jnp.asarray(var, dtype=jnp.float32), eps)
    inv_var = 1.0 / var
    const = (
        jnp.log(jnp.maximum(pi, eps))
        - 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * var), axis=1)
        - 0.5 * jnp.sum(mu * mu * inv_var, axis=1)
    )
    return -0.5 * (x * x) @ inv_var.T + x @ (mu * inv_var).T + const


def posterior_from_stats(gmm_state: Dict[str, jnp.ndarray], x_flat, eps: float = 1e-6):
    mean = gmm_state["mean"]
    std = gmm_state["std"]
    x_std = (x_flat - mean) / jnp.maximum(std, eps)
    logits = diag_gmm_log_prob(x_std, gmm_state["pi"], gmm_state["mu"], gmm_state["var"], eps=eps)
    log_px = jax.nn.logsumexp(logits, axis=1)
    q = jax.nn.softmax(logits, axis=1)
    return q, log_px, x_std


def infer_component_params(gmm_state: Dict[str, jnp.ndarray], x_1, eps: float = 1e-6):
    x_flat = flatten_latents(x_1)
    q, log_px, _ = posterior_from_stats(gmm_state, x_flat, eps=eps)
    k = jnp.argmax(q, axis=1)
    mu, sigma = component_params_from_ids(gmm_state, k, x_1.shape[1:], eps=eps)
    return k, q, log_px, mu, sigma


def component_params_from_ids(
    gmm_state: Dict[str, jnp.ndarray],
    component_ids,
    latent_shape: Tuple[int, ...],
    eps: float = 1e-6,
):
    component_ids = jnp.asarray(component_ids, dtype=jnp.int32)
    mu_std = gmm_state["mu"][component_ids]
    var_std = jnp.maximum(gmm_state["var"][component_ids], eps)
    mean = gmm_state["mean"]
    std = jnp.maximum(gmm_state["std"], eps)
    mu = mu_std * std + mean
    sigma = jnp.sqrt(var_std) * std
    return jnp.reshape(mu, (component_ids.shape[0],) + tuple(latent_shape)), jnp.reshape(
        sigma, (component_ids.shape[0],) + tuple(latent_shape)
    )


def sample_components(
    key,
    gmm_state: Dict[str, jnp.ndarray],
    component_ids,
    latent_shape: Tuple[int, ...],
    eps: float = 1e-6,
):
    mu, sigma = component_params_from_ids(gmm_state, component_ids, latent_shape, eps=eps)
    return mu + sigma * jax.random.normal(key, mu.shape, dtype=mu.dtype), mu, sigma


def sample_prior_components(
    key,
    gmm_state: Dict[str, jnp.ndarray],
    batch_size: int,
    latent_shape: Tuple[int, ...],
    eps: float = 1e-6,
):
    key_k, key_x = jax.random.split(key)
    logits = jnp.log(jnp.maximum(gmm_state["pi"], eps))
    k = jax.random.categorical(key_k, logits, shape=(batch_size,))
    x_0, mu, sigma = sample_components(key_x, gmm_state, k, latent_shape, eps=eps)
    return x_0, mu, sigma, k


def compute_var_floor(
    dim: int,
    min_std: float = 0.0,
    min_std_data_frac: float = 1.0,
    data_std: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    if data_std is None:
        data_std = np.ones((dim,), dtype=np.float32)
    data_std = np.maximum(np.asarray(data_std, dtype=np.float32).reshape(-1), eps)
    abs_floor = (float(min_std) / data_std) ** 2
    rel_floor = np.ones((dim,), dtype=np.float32) * (float(min_std_data_frac) ** 2)
    return np.maximum(abs_floor, rel_floor).astype(np.float32)


def kmeanspp_init(
    x: np.ndarray,
    num_modes: int,
    rng: np.random.Generator,
    chunk_size: int = 512,
    eps: float = 1e-6,
) -> np.ndarray:
    n, dim = x.shape
    centers = np.empty((num_modes, dim), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = x[first]
    min_dist2 = np.full((n,), np.inf, dtype=np.float64)

    for i in range(1, num_modes):
        center = centers[i - 1]
        for sl in chunk_slices(n, chunk_size):
            diff = x[sl].astype(np.float32) - center
            dist2 = np.sum(diff * diff, axis=1, dtype=np.float64)
            min_dist2[sl] = np.minimum(min_dist2[sl], dist2)
        total = float(np.sum(min_dist2))
        if not np.isfinite(total) or total <= eps:
            idx = int(rng.integers(0, n))
        else:
            idx = int(rng.choice(n, p=min_dist2 / total))
        centers[i] = x[idx]
    return centers


def random_init(x: np.ndarray, num_modes: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    replace = n < num_modes
    ids = rng.choice(n, size=num_modes, replace=replace)
    return np.asarray(x[ids], dtype=np.float32).copy()


def _initial_params(
    x: np.ndarray,
    num_modes: int,
    rng: np.random.Generator,
    var_floor: np.ndarray,
    chunk_size: int,
    use_kmeanspp: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if use_kmeanspp:
        mu = kmeanspp_init(x, num_modes, rng, chunk_size=chunk_size)
    else:
        mu = random_init(x, num_modes, rng)
    global_var = np.var(x, axis=0).astype(np.float32)
    var = np.maximum(np.broadcast_to(global_var, mu.shape), var_floor[None]).astype(np.float32)
    pi = np.ones((num_modes,), dtype=np.float32) / float(num_modes)
    return pi, mu, var


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def _kl_regularized_pi(
    soft_counts: np.ndarray,
    strength: float,
    steps: int = 100,
    lr: float = 0.2,
    eps: float = 1e-8,
) -> np.ndarray:
    counts = np.asarray(soft_counts, dtype=np.float64)
    n = float(np.sum(counts))
    k = counts.shape[0]
    if n <= eps:
        return np.ones((k,), dtype=np.float32) / float(k)

    # Maximize sum_k N_k log pi_k - strength * D_KL(pi || uniform).
    pi = np.maximum(counts, eps)
    pi = pi / np.sum(pi)
    logits = np.log(np.maximum(pi, eps))
    uniform = 1.0 / float(k)
    beta = float(strength) / n

    for _ in range(max(int(steps), 1)):
        pi = _softmax_np(logits)
        grad_pi = (counts / n) / np.maximum(pi, eps)
        grad_pi -= beta * (np.log(np.maximum(pi, eps) / uniform) + 1.0)
        baseline = float(np.sum(pi * grad_pi))
        grad_logits = pi * (grad_pi - baseline)
        logits = logits + float(lr) * grad_logits

    return _softmax_np(logits).astype(np.float32)


def _update_pi(
    soft_counts: np.ndarray,
    prior_type: str,
    prior_strength: float,
    kl_steps: int,
    kl_lr: float,
    eps: float,
) -> np.ndarray:
    prior_type = str(prior_type).lower()
    total = float(np.sum(soft_counts))
    k = soft_counts.shape[0]
    if total <= eps:
        return np.ones((k,), dtype=np.float32) / float(k)

    if prior_type in ("none", "off", "ml"):
        pi = soft_counts.astype(np.float64) / total
    elif prior_type in ("dirichlet", "pseudo_count", "pseudocount"):
        pi = (soft_counts + float(prior_strength)).astype(np.float64)
        pi /= np.sum(pi)
    elif prior_type in ("kl", "dkl", "d_kl"):
        pi = _kl_regularized_pi(soft_counts, prior_strength, steps=kl_steps, lr=kl_lr, eps=eps)
    else:
        raise ValueError(f"Unknown pi prior type {prior_type}")

    pi = np.maximum(pi, eps)
    pi = pi / np.sum(pi)
    return pi.astype(np.float32)


def _em_step(
    x: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    var_floor: np.ndarray,
    pi_prior_type: str,
    pi_prior_strength: float,
    pi_kl_steps: int,
    pi_kl_lr: float,
    chunk_size: int,
    eps: float,
):
    n, dim = x.shape
    k = pi.shape[0]
    soft_counts = np.zeros((k,), dtype=np.float64)
    sum_x = np.zeros((k, dim), dtype=np.float64)
    sum_x2 = np.zeros((k, dim), dtype=np.float64)
    nll_sum = 0.0

    for sl in chunk_slices(n, chunk_size):
        xb = np.asarray(x[sl], dtype=np.float32)
        logits = diag_gmm_log_prob_np(xb, pi, mu, var, eps=eps)
        log_norm = _logsumexp_np(logits, axis=1, keepdims=True)
        resp = np.exp(logits - log_norm).astype(np.float64)
        nll_sum += -float(np.sum(log_norm))
        soft_counts += np.sum(resp, axis=0)
        sum_x += resp.T @ xb
        sum_x2 += resp.T @ (xb * xb)

    denom = np.maximum(soft_counts[:, None], eps)
    mu_new = sum_x / denom
    var_new = sum_x2 / denom - mu_new * mu_new

    dead = soft_counts <= eps
    if np.any(dead):
        mu_new[dead] = mu[dead]
        var_new[dead] = var[dead]

    var_new = np.maximum(var_new, var_floor[None])
    pi_new = _update_pi(
        soft_counts,
        prior_type=pi_prior_type,
        prior_strength=pi_prior_strength,
        kl_steps=pi_kl_steps,
        kl_lr=pi_kl_lr,
        eps=eps,
    )
    return (
        pi_new.astype(np.float32),
        mu_new.astype(np.float32),
        var_new.astype(np.float32),
        float(nll_sum / max(n, 1)),
        soft_counts.astype(np.float32),
    )


def fit_diag_gmm(
    x: np.ndarray,
    num_modes: int,
    em_iters: int,
    em_restarts: int = 1,
    seed: int = 0,
    pi_prior_type: str = "dirichlet",
    pi_prior_strength: float = 1e-2,
    pi_kl_steps: int = 100,
    pi_kl_lr: float = 0.2,
    min_std: float = 0.0,
    min_std_data_frac: float = 1.0,
    data_std: Optional[np.ndarray] = None,
    var_floor: Optional[np.ndarray] = None,
    chunk_size: int = 128,
    use_kmeanspp: bool = True,
    eps: float = 1e-6,
    em_metrics_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected flattened latents of shape [N, D], got {x.shape}")
    n, dim = x.shape
    if n == 0:
        raise ValueError("Cannot fit GMM with zero samples")
    if num_modes <= 0:
        raise ValueError("num_modes must be positive")

    if var_floor is None:
        var_floor = compute_var_floor(dim, min_std, min_std_data_frac, data_std=data_std, eps=eps)
    else:
        var_floor = np.asarray(var_floor, dtype=np.float32).reshape(-1)
        if var_floor.shape[0] != dim:
            raise ValueError(f"var_floor dim mismatch: expected {dim}, got {var_floor.shape[0]}")

    best = None
    restart_traces = []
    for restart in range(max(int(em_restarts), 1)):
        rng = np.random.default_rng(int(seed) + restart * 1009)
        pi, mu, var = _initial_params(x, num_modes, rng, var_floor, chunk_size, use_kmeanspp)
        trace = []
        counts = np.zeros((num_modes,), dtype=np.float32)
        nll = np.inf
        for it in range(int(em_iters)):
            pi, mu, var, nll, counts = _em_step(
                x,
                pi,
                mu,
                var,
                var_floor,
                pi_prior_type=pi_prior_type,
                pi_prior_strength=pi_prior_strength,
                pi_kl_steps=pi_kl_steps,
                pi_kl_lr=pi_kl_lr,
                chunk_size=chunk_size,
                eps=eps,
            )
            trace_entry = {
                "restart": int(restart),
                "iter": int(it),
                "nll": float(nll),
                "pi_min": float(np.min(pi)),
                "pi_max": float(np.max(pi)),
                "pi_entropy_normalized": float(
                    -np.sum(pi * np.log(np.maximum(pi, eps))) / max(np.log(num_modes), eps)
                ),
                "count_min": float(np.min(counts)),
                "count_max": float(np.max(counts)),
                "count_gap": float(np.max(counts) - np.min(counts)),
                "dead_components": int(np.sum(counts <= eps)),
                "var_min": float(np.min(var)),
                "var_max": float(np.max(var)),
                "var_floor_hit_rate": float(np.mean(var <= (var_floor[None] * (1.0 + 1e-5)))),
            }
            trace.append(trace_entry)
            if em_metrics_callback is not None:
                em_metrics_callback(trace_entry)
        candidate = {
            "pi": pi,
            "mu": mu,
            "var": var,
            "counts": counts,
            "nll": float(nll),
            "trace": trace,
            "restart": restart,
        }
        restart_traces.append({"restart": restart, "final_nll": float(nll), "trace": trace})
        if best is None or candidate["nll"] < best["nll"]:
            best = candidate

    best["var_floor"] = var_floor.astype(np.float32)
    best["restart_traces"] = restart_traces
    return best


def assignment_metrics_np(
    x: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    chunk_size: int = 128,
    eps: float = 1e-6,
) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    k = pi.shape[0]
    hard_counts = np.zeros((k,), dtype=np.int64)
    soft_counts = np.zeros((k,), dtype=np.float64)
    nll_sum = 0.0
    entropy_sum = 0.0
    top1_sum = 0.0
    margin_sum = 0.0

    for sl in chunk_slices(n, chunk_size):
        xb = np.asarray(x[sl], dtype=np.float32)
        logits = diag_gmm_log_prob_np(xb, pi, mu, var, eps=eps)
        log_norm = _logsumexp_np(logits, axis=1, keepdims=True)
        q = np.exp(logits - log_norm)
        hard = np.argmax(q, axis=1)
        hard_counts += np.bincount(hard, minlength=k)
        soft_counts += np.sum(q, axis=0)
        nll_sum += -float(np.sum(log_norm))
        entropy = -np.sum(q * np.log(np.maximum(q, eps)), axis=1)
        entropy_sum += float(np.sum(entropy))
        if k > 1:
            top2 = np.partition(q, kth=-2, axis=1)[:, -2:]
            top1 = top2[:, 1]
            top2_prob = top2[:, 0]
        else:
            top1 = q[:, 0]
            top2_prob = np.zeros_like(top1)
        top1_sum += float(np.sum(top1))
        margin_sum += float(np.sum(top1 - top2_prob))

    count_min = int(np.min(hard_counts)) if k > 0 else 0
    count_max = int(np.max(hard_counts)) if k > 0 else 0
    count_ratio = float(count_max / max(count_min, 1))
    entropy_mean = entropy_sum / max(n, 1)
    return {
        "nll": float(nll_sum / max(n, 1)),
        "count_min": count_min,
        "count_max": count_max,
        "count_gap": int(count_max - count_min),
        "count_ratio": count_ratio,
        "dead_components": int(np.sum(hard_counts == 0)),
        "posterior_entropy_mean": float(entropy_mean),
        "posterior_entropy_normalized": float(entropy_mean / max(np.log(k), eps)),
        "posterior_top1_prob_mean": float(top1_sum / max(n, 1)),
        "posterior_top1_margin_mean": float(margin_sum / max(n, 1)),
        "hard_counts": hard_counts.astype(np.int64).tolist(),
        "soft_counts": soft_counts.astype(np.float64).tolist(),
    }


def gmm_diagnostics(
    x_train_std: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    var_floor: np.ndarray,
    data_var: Optional[np.ndarray] = None,
    x_valid_std: Optional[np.ndarray] = None,
    chunk_size: int = 128,
    eps: float = 1e-6,
) -> Dict[str, object]:
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float32)
    var = np.asarray(var, dtype=np.float32)
    var_floor = np.asarray(var_floor, dtype=np.float32).reshape(-1)
    k = int(pi.shape[0])
    uniform = np.ones_like(pi) / max(k, 1)

    metrics: Dict[str, object] = {}
    train_metrics = assignment_metrics_np(x_train_std, pi, mu, var, chunk_size=chunk_size, eps=eps)
    metrics.update({f"train_{name}": value for name, value in train_metrics.items() if name not in ("hard_counts", "soft_counts")})
    metrics["train_hard_counts"] = train_metrics["hard_counts"]
    metrics["train_soft_counts"] = train_metrics["soft_counts"]

    if x_valid_std is not None and x_valid_std.shape[0] > 0:
        valid_metrics = assignment_metrics_np(x_valid_std, pi, mu, var, chunk_size=chunk_size, eps=eps)
        metrics.update({f"valid_{name}": value for name, value in valid_metrics.items() if name not in ("hard_counts", "soft_counts")})
        metrics["valid_hard_counts"] = valid_metrics["hard_counts"]

    metrics["pi_kl_to_uniform"] = float(np.sum(pi * (np.log(np.maximum(pi, eps)) - np.log(uniform))))
    metrics["pi_mse_to_uniform"] = float(np.mean((pi - uniform) ** 2))
    metrics["pi_min"] = float(np.min(pi))
    metrics["pi_max"] = float(np.max(pi))
    metrics["pi_entropy"] = float(-np.sum(pi * np.log(np.maximum(pi, eps))))
    metrics["pi_entropy_normalized"] = float(metrics["pi_entropy"] / max(np.log(k), eps))

    if data_var is None:
        data_var = np.var(np.asarray(x_train_std, dtype=np.float32), axis=0)
    data_var = np.asarray(data_var, dtype=np.float32).reshape(-1)
    metrics["data_variance_mean"] = float(np.mean(data_var))
    metrics["data_covariance_trace"] = float(np.sum(data_var))
    metrics["component_variance_mean"] = float(np.mean(var))
    metrics["component_variance_min"] = float(np.min(var))
    metrics["component_variance_max"] = float(np.max(var))
    metrics["component_variance_trace_mean"] = float(np.mean(np.sum(var, axis=1)))
    metrics["component_variance_trace_min"] = float(np.min(np.sum(var, axis=1)))
    metrics["component_variance_trace_max"] = float(np.max(np.sum(var, axis=1)))
    metrics["var_floor_hit_rate"] = float(np.mean(var <= (var_floor[None] * (1.0 + 1e-5))))

    if k > 1:
        mu_norm = np.sum(mu * mu, axis=1, keepdims=True)
        dist2 = np.maximum(mu_norm + mu_norm.T - 2.0 * (mu @ mu.T), 0.0)
        dist = np.sqrt(dist2)
        offdiag = ~np.eye(k, dtype=bool)
        center_dists = dist[offdiag]
        std_mean = np.sqrt(np.maximum(np.mean(var, axis=1), eps))
        scale = std_mean[:, None] + std_mean[None, :]
        overlap = np.exp(-dist / np.maximum(scale, eps))
        overlap_vals = overlap[offdiag]
        metrics["center_distance_min"] = float(np.min(center_dists))
        metrics["center_distance_mean"] = float(np.mean(center_dists))
        metrics["center_distance_max"] = float(np.max(center_dists))
        metrics["overlap_proxy_mean"] = float(np.mean(overlap_vals))
        metrics["overlap_proxy_max"] = float(np.max(overlap_vals))
        metrics["overlap_proxy_pair_fraction_gt_0_5"] = float(np.mean(overlap_vals > 0.5))
    else:
        metrics["center_distance_min"] = 0.0
        metrics["center_distance_mean"] = 0.0
        metrics["center_distance_max"] = 0.0
        metrics["overlap_proxy_mean"] = 0.0
        metrics["overlap_proxy_max"] = 0.0
        metrics["overlap_proxy_pair_fraction_gt_0_5"] = 0.0

    return metrics


def save_gmm_stats(path: str, **stats):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **stats)


def load_gmm_stats(path: str) -> Dict[str, jnp.ndarray]:
    data = np.load(path, allow_pickle=True)
    state = {
        "pi": jnp.asarray(data["pi"], dtype=jnp.float32),
        "mu": jnp.asarray(data["mu"], dtype=jnp.float32),
        "var": jnp.asarray(data["var"], dtype=jnp.float32),
        "mean": jnp.asarray(data["mean"], dtype=jnp.float32),
        "std": jnp.asarray(data["std"], dtype=jnp.float32),
    }
    if "var_floor" in data.files:
        state["var_floor"] = jnp.asarray(data["var_floor"], dtype=jnp.float32)
    return state


def json_dump(path: str, payload: Dict[str, object]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def path_straightness_metrics_np(path_states: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    path_states = np.asarray(path_states, dtype=np.float32)
    if path_states.ndim < 3 or path_states.shape[1] < 2:
        return {
            "path_length_mean": 0.0,
            "endpoint_displacement_mean": 0.0,
            "straightness_ratio_mean": 0.0,
            "curvature_proxy_mean": 0.0,
        }
    flat = path_states.reshape(path_states.shape[0], path_states.shape[1], -1)
    delta = flat[:, 1:] - flat[:, :-1]
    seg_len = np.linalg.norm(delta, axis=-1)
    path_len = np.sum(seg_len, axis=1)
    endpoint = np.linalg.norm(flat[:, -1] - flat[:, 0], axis=-1)
    unit = delta / np.maximum(seg_len[..., None], eps)
    if unit.shape[1] > 1:
        curvature = np.mean(np.linalg.norm(unit[:, 1:] - unit[:, :-1], axis=-1), axis=1)
    else:
        curvature = np.zeros((path_states.shape[0],), dtype=np.float32)
    return {
        "path_length_mean": float(np.mean(path_len)),
        "endpoint_displacement_mean": float(np.mean(endpoint)),
        "straightness_ratio_mean": float(np.mean(path_len / np.maximum(endpoint, eps))),
        "curvature_proxy_mean": float(np.mean(curvature)),
    }
