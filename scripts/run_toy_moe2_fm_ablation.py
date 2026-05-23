from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

# The repo's JAX install includes TPU support. On local CPU/GPU machines without
# a TPU runtime, probing TPU can hang for a long time, so default toy runs to CPU.
# Override with JAX_PLATFORMS=cuda,cpu when running the script on a CUDA machine.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def make_aniso_blobs(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[-3.4, -2.2], [-0.6, -2.4], [2.7, -1.8], [-2.7, 1.9], [0.6, 2.4], [3.2, 1.5]],
        dtype=np.float32,
    )
    angles = np.array([0.2, -0.9, 0.65, 1.1, -0.45, 0.35])
    scales = np.array([[0.55, 0.11], [0.28, 0.65], [0.72, 0.12], [0.40, 0.28], [0.22, 0.75], [0.62, 0.20]])
    probs = np.array([0.10, 0.20, 0.15, 0.15, 0.25, 0.15])
    labels = rng.choice(len(centers), size=n, p=probs)
    x = np.empty((n, 2), dtype=np.float32)
    for k in range(len(centers)):
        idx = labels == k
        c, s = np.cos(angles[k]), np.sin(angles[k])
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        cov = rot @ np.diag(scales[k] ** 2) @ rot.T
        x[idx] = rng.multivariate_normal(centers[k], cov, size=int(idx.sum()))
    return x, labels.astype(np.int32)


def make_nested_rings(n: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probs = np.array([0.35, 0.40, 0.25])
    labels = rng.choice(3, size=n, p=probs)
    radius = np.array([1.1, 2.15, 3.05])[labels] + rng.normal(0, 0.065 + 0.015 * labels, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    x = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    x += rng.normal(0, 0.025, size=x.shape)
    return x.astype(np.float32), labels.astype(np.int32)


def make_pinwheel(n: int, seed: int = 2, arms: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, arms, size=n)
    r = rng.gamma(shape=2.2, scale=0.55, size=n) + 0.35
    theta = labels * 2 * np.pi / arms + 0.85 * r + rng.normal(0, 0.13, size=n)
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x *= 1.35
    x += rng.normal(0, 0.035, size=x.shape)
    return x.astype(np.float32), labels.astype(np.int32)


def make_checkerboard(n: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gx = rng.integers(-3, 4, size=n)
    gy = rng.integers(-3, 4, size=n)
    keep = (gx + gy) % 2 == 0
    while np.any(~keep):
        gx[~keep] = rng.integers(-3, 4, size=int(np.sum(~keep)))
        gy[~keep] = rng.integers(-3, 4, size=int(np.sum(~keep)))
        keep = (gx + gy) % 2 == 0
    x = np.stack([gx, gy], axis=1).astype(np.float32)
    x += rng.normal(0, 0.16, size=x.shape).astype(np.float32)
    labels = ((gx + 3) * 7 + (gy + 3)).astype(np.int32)
    return x, labels


def make_spiral_blobs(n: int, seed: int = 4, arms: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, arms, size=n)
    t = rng.uniform(0.15, 1.0, size=n)
    theta = labels * 2 * np.pi / arms + 4.0 * np.pi * t
    radius = 0.5 + 2.7 * t
    x = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    x += rng.normal(0, 0.09, size=x.shape)
    return x.astype(np.float32), labels.astype(np.int32)


DATASET_FNS = {
    "aniso_blobs": make_aniso_blobs,
    "nested_rings": make_nested_rings,
    "pinwheel": make_pinwheel,
    "checkerboard": make_checkerboard,
    "spiral_blobs": make_spiral_blobs,
}


def load_keras_dataset(name: str, n_train: int, n_valid: int, seed: int):
    if name == "mnist":
        from tensorflow.keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        from tensorflow.keras.datasets import fashion_mnist

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif name == "cifar10":
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.reshape(-1)
        y_test = y_test.reshape(-1)
    else:
        raise ValueError(f"Unknown dataset {name!r}; available toy datasets {sorted(DATASET_FNS)} plus mnist/fashion_mnist/cifar10")

    rng = np.random.default_rng(seed)
    train_ids = rng.choice(len(x_train), size=min(n_train, len(x_train)), replace=False)
    valid_ids = rng.choice(len(x_test), size=min(n_valid, len(x_test)), replace=False)
    x_train = x_train[train_ids].astype(np.float32).reshape(len(train_ids), -1) / 255.0
    x_valid = x_test[valid_ids].astype(np.float32).reshape(len(valid_ids), -1) / 255.0
    return x_train, y_train[train_ids].astype(np.int32), x_valid, y_test[valid_ids].astype(np.int32)


def pca_transform(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    dim: int,
    max_samples: int,
    seed: int,
):
    if dim <= 0 or dim >= x_train.shape[1]:
        return x_train.astype(np.float32), x_valid.astype(np.float32), {"pca_dim": int(x_train.shape[1]), "pca_var_ratio": 1.0}
    rng = np.random.default_rng(seed)
    sample_size = min(max(int(max_samples), 2), len(x_train))
    sample_ids = rng.choice(len(x_train), size=sample_size, replace=False)
    mean = np.mean(x_train[sample_ids], axis=0, dtype=np.float64).astype(np.float32)
    centered = x_train[sample_ids] - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:dim].T.astype(np.float32)
    train_z = (x_train - mean) @ basis
    valid_z = (x_valid - mean) @ basis
    denom = max(float(np.sum(singular_values * singular_values)), 1e-8)
    return train_z.astype(np.float32), valid_z.astype(np.float32), {
        "pca_dim": int(dim),
        "pca_var_ratio": float(np.sum(singular_values[:dim] * singular_values[:dim]) / denom),
    }


def maybe_standardize(x_train: np.ndarray, x_valid: np.ndarray, enabled: bool):
    if not enabled:
        return x_train.astype(np.float32), x_valid.astype(np.float32), {"standardized": 0}
    mean = np.mean(x_train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(x_train, axis=0).astype(np.float32) + 1e-6
    return ((x_train - mean) / std).astype(np.float32), ((x_valid - mean) / std).astype(np.float32), {
        "standardized": 1,
        "feature_std_min": float(np.min(std)),
        "feature_std_max": float(np.max(std)),
    }


def load_dataset_for_ablation(name: str, n_train: int, n_valid: int, seed: int, pca_dim: int, pca_max_samples: int, standardize: bool):
    if name in DATASET_FNS:
        x_train, y_train = DATASET_FNS[name](n_train, seed=seed)
        x_valid, y_valid = DATASET_FNS[name](n_valid, seed=seed + 1)
        meta = {"raw_dim": int(x_train.shape[1]), "kind": "toy"}
    else:
        x_train, y_train, x_valid, y_valid = load_keras_dataset(name, n_train, n_valid, seed=seed)
        meta = {"raw_dim": int(x_train.shape[1]), "kind": "keras"}
    x_train, x_valid, pca_meta = pca_transform(x_train, x_valid, pca_dim, pca_max_samples, seed)
    x_train, x_valid, std_meta = maybe_standardize(x_train, x_valid, standardize)
    meta.update(pca_meta)
    meta.update(std_meta)
    meta["data_variance_mean"] = float(np.mean(np.var(x_train, axis=0)))
    return x_train, y_train, x_valid, y_valid, meta


def logsumexp_np(a: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-12)
    return out if keepdims else np.squeeze(out, axis=axis)


def pairwise_dist2(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(x * x, axis=1, keepdims=True) + np.sum(centers * centers, axis=1)[None] - 2 * x @ centers.T, 0.0)


def kmeanspp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(0, n)]
    dist2 = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(dist2.sum())
        idx = int(rng.integers(0, n)) if total <= 1e-12 else int(rng.choice(n, p=dist2 / total))
        centers[i] = x[idx]
        dist2 = np.minimum(dist2, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def farthest_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(0, n)]
    dist2 = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        centers[i] = x[int(np.argmax(dist2))]
        dist2 = np.minimum(dist2, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def lloyd_warmup(x: np.ndarray, centers: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    centers = centers.astype(np.float32).copy()
    for _ in range(int(steps)):
        labels = np.argmin(pairwise_dist2(x, centers), axis=1)
        for j in range(len(centers)):
            pts = x[labels == j]
            centers[j] = pts.mean(axis=0) if len(pts) else x[rng.integers(0, len(x))]
    return centers


def pca_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered[: min(len(x), 4096)], full_matrices=False)
    basis = vt[: min(2, x.shape[1])].T
    z = centered @ basis
    z_centers = kmeanspp_init(z.astype(np.float32), k, rng)
    ids = [int(np.argmin(np.sum((z - c) ** 2, axis=1))) for c in z_centers]
    return x[np.asarray(ids)]


def split_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    base = max(1, int(np.ceil(k / 2)))
    centers = lloyd_warmup(x, kmeanspp_init(x, base, rng), 3, rng)
    while len(centers) < k:
        labels = np.argmin(pairwise_dist2(x, centers), axis=1)
        scores = []
        for j in range(len(centers)):
            pts = x[labels == j]
            scores.append(0.0 if len(pts) == 0 else len(pts) * float(np.mean(np.var(pts, axis=0))))
        split_id = int(np.argmax(scores))
        pts = x[labels == split_id]
        var = np.var(pts, axis=0) if len(pts) else np.var(x, axis=0)
        dim = int(np.argmax(var))
        delta = np.zeros((x.shape[1],), dtype=np.float32)
        delta[dim] = 0.45 * math.sqrt(max(float(var[dim]), 1e-6))
        centers = np.concatenate(
            [centers[:split_id], (centers[split_id] - delta)[None], (centers[split_id] + delta)[None], centers[split_id + 1 :]],
            axis=0,
        )
    return centers[:k].astype(np.float32)


def hybrid_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    first = max(1, k // 2)
    centers = kmeanspp_init(x, first, rng)
    dist2 = np.min(pairwise_dist2(x, centers), axis=1)
    while len(centers) < k:
        center = x[int(np.argmax(dist2))][None].astype(np.float32)
        centers = np.concatenate([centers, center], axis=0)
        dist2 = np.minimum(dist2, np.sum((x - center) ** 2, axis=1))
    return centers.astype(np.float32)


def quantile_pca_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    sample = centered[rng.choice(len(x), size=min(len(x), 4096), replace=False)]
    _, _, vt = np.linalg.svd(sample, full_matrices=False)
    pc1 = centered @ vt[0]
    quantiles = np.linspace(0.02, 0.98, k)
    ids = [int(np.argmin(np.abs(pc1 - np.quantile(pc1, q)))) for q in quantiles]
    return x[np.asarray(ids)].astype(np.float32).copy()


def init_centers(x: np.ndarray, k: int, strategy: str, rng: np.random.Generator, warmup: int) -> np.ndarray:
    if strategy == "kmeans++":
        centers = kmeanspp_init(x, k, rng)
    elif strategy == "farthest":
        centers = farthest_init(x, k, rng)
    elif strategy == "pca":
        centers = pca_init(x, k, rng)
    elif strategy == "split":
        centers = split_init(x, k, rng)
    elif strategy == "hybrid":
        centers = hybrid_init(x, k, rng)
    elif strategy == "quantilepca":
        centers = quantile_pca_init(x, k, rng)
    else:
        raise ValueError(strategy)
    return lloyd_warmup(x, centers, warmup, rng) if warmup else centers.astype(np.float32)


def gmm_log_prob_np(x: np.ndarray, pi: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - mu[None, :, :]
    log_comp = -0.5 * (np.sum(diff * diff / var[None], axis=-1) + np.sum(np.log(var), axis=-1) + x.shape[1] * np.log(2 * np.pi))
    return log_comp + np.log(pi[None] + 1e-12)


def posterior_np(x: np.ndarray, fit: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    logits = gmm_log_prob_np(x, fit["pi"], fit["mu"], fit["var"])
    log_norm = logsumexp_np(logits, axis=1, keepdims=True)
    return np.exp(logits - log_norm), -np.squeeze(log_norm, axis=1)


def fit_diag_gmm_np(
    x: np.ndarray,
    k: int,
    iters: int,
    seed: int,
    init_strategy: str,
    warmup: int,
    restarts: int,
    floor_frac: float,
) -> dict[str, Any]:
    best = None
    best_nll = float("inf")
    for restart in range(int(restarts)):
        rng = np.random.default_rng(seed + 1009 * restart)
        data_var = np.var(x, axis=0).astype(np.float32) + 1e-6
        var_floor = np.maximum((floor_frac ** 2) * data_var, 1e-5)
        mu = init_centers(x, k, init_strategy, rng, warmup=warmup)
        var = np.tile(data_var[None], (k, 1)).astype(np.float32)
        pi = np.full((k,), 1.0 / k, dtype=np.float32)
        trace = []
        for it in range(iters):
            logits = gmm_log_prob_np(x, pi, mu, var)
            log_norm = logsumexp_np(logits, axis=1, keepdims=True)
            q = np.exp(logits - log_norm)
            nk = q.sum(axis=0) + 1e-8
            pi = (nk / len(x)).astype(np.float32)
            mu = ((q.T @ x) / nk[:, None]).astype(np.float32)
            diff = x[:, None, :] - mu[None]
            var = np.maximum((np.sum(q[:, :, None] * diff * diff, axis=0) / nk[:, None]).astype(np.float32), var_floor[None])
            trace.append(-float(np.mean(log_norm)))
        nll = trace[-1]
        if nll < best_nll:
            best = {
                "pi": pi,
                "mu": mu,
                "var": var,
                "var_floor": var_floor,
                "train_nll": nll,
                "trace": trace,
                "init_strategy": init_strategy,
                "warmup": int(warmup),
                "restarts": int(restarts),
            }
            best_nll = nll
    assert best is not None
    return best


def gmm_metrics(x: np.ndarray, fit: dict[str, np.ndarray]) -> dict[str, float | int]:
    q, nll = posterior_np(x, fit)
    hard = np.argmax(q, axis=1)
    counts = np.bincount(hard, minlength=fit["pi"].shape[0])
    k = fit["pi"].shape[0]
    pi = fit["pi"]
    mu = fit["mu"]
    var = fit["var"]
    center_dist = np.sqrt(np.sum((mu[:, None, :] - mu[None, :, :]) ** 2, axis=-1) + 1e-12)
    mask = ~np.eye(k, dtype=bool)
    avg_std = np.sqrt(np.mean(var, axis=1))
    overlap = np.exp(-center_dist / (avg_std[:, None] + avg_std[None, :] + 1e-6))
    return {
        "gmm_valid_nll": float(np.mean(nll)),
        "gmm_pi_entropy_norm": float(-np.sum(pi * np.log(pi + 1e-12)) / np.log(k)),
        "gmm_count_ratio": float(counts.max() / max(counts.min(), 1)),
        "gmm_dead": int(np.sum(counts == 0)),
        "gmm_overlap_max": float(np.max(overlap[mask])) if k > 1 else 0.0,
        "gmm_q_entropy_norm": float(np.mean(-np.sum(q * np.log(q + 1e-12), axis=1) / np.log(k))),
        "gmm_top1_prob": float(np.mean(np.max(q, axis=1))),
    }


def init_mlp(key, sizes: list[int]) -> list[dict[str, jnp.ndarray]]:
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
        scale = math.sqrt(2.0 / din)
        params.append({"w": jax.random.normal(keys[i], (din, dout)) * scale, "b": jnp.zeros((dout,))})
    return params


def mlp_apply(params, x: jnp.ndarray, final_tanh: bool = False) -> jnp.ndarray:
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["w"] + layer["b"])
    y = h @ params[-1]["w"] + params[-1]["b"]
    return jnp.tanh(y) if final_tanh else y


def zeros_like_tree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def adam_update(params, m, v, grads, step, lr: float, weight_decay: float = 0.0):
    b1, b2, eps = 0.9, 0.999, 1e-8
    step_f = step.astype(jnp.float32)
    grads = jax.tree_util.tree_map(lambda g, p: g + weight_decay * p, grads, params)
    new_m = jax.tree_util.tree_map(lambda mt, g: b1 * mt + (1 - b1) * g, m, grads)
    new_v = jax.tree_util.tree_map(lambda vt, g: b2 * vt + (1 - b2) * (g * g), v, grads)
    new_params = jax.tree_util.tree_map(
        lambda p, mt, vt: p - lr * (mt / (1 - b1**step_f)) / (jnp.sqrt(vt / (1 - b2**step_f)) + eps),
        params,
        new_m,
        new_v,
    )
    return new_params, new_m, new_v


@jax.jit
def posterior_jax(x: jnp.ndarray, pi: jnp.ndarray, mu: jnp.ndarray, var: jnp.ndarray):
    diff = x[:, None, :] - mu[None, :, :]
    log_comp = -0.5 * (jnp.sum(diff * diff / var[None], axis=-1) + jnp.sum(jnp.log(var), axis=-1) + x.shape[1] * jnp.log(2 * jnp.pi))
    logits = log_comp + jnp.log(pi[None] + 1e-12)
    return jax.nn.softmax(logits, axis=-1)


@jax.jit
def router_loss(params, x: jnp.ndarray, target_q: jnp.ndarray):
    logits = mlp_apply(params, x)
    logq = jax.nn.log_softmax(logits, axis=-1)
    q_safe = jnp.maximum(target_q, 1e-8)
    return jnp.mean(jnp.sum(q_safe * (jnp.log(q_safe) - logq), axis=-1))


@jax.jit
def router_step(params, m, v, step, x: jnp.ndarray, target_q: jnp.ndarray, lr: float):
    loss, grads = jax.value_and_grad(router_loss)(params, x, target_q)
    params, m, v = adam_update(params, m, v, grads, step, lr, weight_decay=1e-4)
    return params, m, v, loss


def train_router(
    key,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    fit: dict[str, np.ndarray],
    steps: int,
    batch_size: int,
    hidden: int,
    lr: float,
) -> tuple[Any, dict[str, float]]:
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
    n_mix = min(max(len(x_train), batch_size), 8192)
    ids = rng.choice(len(x_train), size=n_mix // 2, replace=len(x_train) < n_mix // 2)
    prior_ids = rng.choice(fit["pi"].shape[0], size=n_mix - len(ids), p=fit["pi"])
    dim = x_train.shape[1]
    x_prior = fit["mu"][prior_ids] + rng.normal(size=(len(prior_ids), dim)).astype(np.float32) * np.sqrt(fit["var"][prior_ids])
    x_mix = np.concatenate([x_train[ids], x_prior.astype(np.float32)], axis=0)
    q_mix, _ = posterior_np(x_mix, fit)
    q_valid, _ = posterior_np(x_valid, fit)
    xj = jnp.asarray(x_mix)
    qj = jnp.asarray(q_mix)
    key, init_key = jax.random.split(key)
    params = init_mlp(init_key, [x_train.shape[1], hidden, hidden, fit["pi"].shape[0]])
    m = zeros_like_tree(params)
    v = zeros_like_tree(params)
    for step in range(1, steps + 1):
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (batch_size,), 0, xj.shape[0])
        params, m, v, _ = router_step(params, m, v, jnp.asarray(step), xj[idx], qj[idx], lr)
    logits_valid = np.asarray(jax.device_get(mlp_apply(params, jnp.asarray(x_valid))))
    q_pred = np.asarray(jax.nn.softmax(jnp.asarray(logits_valid), axis=-1))
    hard = np.argmax(q_pred, axis=1)
    counts = np.bincount(hard, minlength=fit["pi"].shape[0])
    usage = counts / max(counts.sum(), 1)
    valid_loss = float(router_loss(params, jnp.asarray(x_valid), jnp.asarray(q_valid)))
    return params, {
        "router_valid_kl": valid_loss,
        "router_top1_agreement": float(np.mean(np.argmax(q_pred, axis=1) == np.argmax(q_valid, axis=1))),
        "router_top1_prob": float(np.mean(np.max(q_pred, axis=1))),
        "router_usage_entropy_norm": float(-np.sum(np.maximum(usage, 1e-8) * np.log(np.maximum(usage, 1e-8))) / np.log(len(usage))),
        "router_dead": int(np.sum(counts == 0)),
    }


def sample_tide_source(key, pi, mu, var, router_params, batch_size: int, topk: int):
    key_ids, key_noise, key_top = jax.random.split(key, 3)
    base_ids = jax.random.categorical(key_ids, jnp.log(pi), shape=(batch_size,))
    x0_base = mu[base_ids] + jax.random.normal(key_noise, (batch_size, mu.shape[1])) * jnp.sqrt(var[base_ids])
    q = jax.nn.softmax(mlp_apply(router_params, x0_base), axis=-1)
    top_probs, top_ids = jax.lax.top_k(q, min(topk, q.shape[-1]))
    w = top_probs / jnp.maximum(jnp.sum(top_probs, axis=-1, keepdims=True), 1e-8)
    eps = jax.random.normal(key_top, (batch_size, top_ids.shape[1], mu.shape[1]))
    top_mu = mu[top_ids]
    top_sigma = jnp.sqrt(var[top_ids])
    samples = top_mu + eps * top_sigma
    x0 = jnp.sum(w[:, :, None] * samples, axis=1)
    mu_tide = jnp.sum(w[:, :, None] * top_mu, axis=1)
    sigma_tide = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(w[:, :, None] * top_sigma), axis=1), 1e-8))
    return x0, mu_tide, sigma_tide


@partial(jax.jit, static_argnames=("batch_size", "topk", "is_gmm"))
def fm_batch(key, x_data, data_mean, data_std, pi, mu, var, router_params, batch_size: int, topk: int, is_gmm: bool):
    key_i, key_t, key_src = jax.random.split(key, 3)
    idx = jax.random.randint(key_i, (batch_size,), 0, x_data.shape[0])
    x1 = x_data[idx]
    if is_gmm:
        x0, mu_cond, sigma_cond = sample_tide_source(key_src, pi, mu, var, router_params, batch_size, topk)
    else:
        x0 = data_mean + data_std * jax.random.normal(key_src, (batch_size, x_data.shape[1]))
        mu_cond = jnp.zeros_like(x0)
        sigma_cond = jnp.zeros_like(x0)
    t = jax.random.uniform(key_t, (batch_size, 1))
    xt = (1.0 - t) * x0 + t * x1
    target = x1 - x0
    inp = jnp.concatenate([xt, t, mu_cond, sigma_cond], axis=1)
    return inp, target, x0, x1


@jax.jit
def fm_loss(params, inp: jnp.ndarray, target: jnp.ndarray):
    pred = mlp_apply(params, inp)
    return jnp.mean((pred - target) ** 2)


@jax.jit
def fm_step(params, m, v, step, inp: jnp.ndarray, target: jnp.ndarray, lr: float):
    loss, grads = jax.value_and_grad(fm_loss)(params, inp, target)
    params, m, v = adam_update(params, m, v, grads, step, lr, weight_decay=0.0)
    return params, m, v, loss


def eval_fm(params, key, x_valid, data_mean, data_std, pi, mu, var, router_params, batch_size: int, topk: int, is_gmm: bool, batches: int):
    vals = []
    xj = jnp.asarray(x_valid)
    for _ in range(batches):
        key, sub = jax.random.split(key)
        inp, target, _, _ = fm_batch(sub, xj, data_mean, data_std, pi, mu, var, router_params, batch_size, topk, is_gmm)
        vals.append(float(fm_loss(params, inp, target)))
    return float(np.mean(vals))


def rollout(params, key, n, data_mean, data_std, pi, mu, var, router_params, topk: int, is_gmm: bool, steps: int = 64):
    if is_gmm:
        x, mu_cond, sigma_cond = sample_tide_source(key, pi, mu, var, router_params, n, topk)
    else:
        x = data_mean + data_std * jax.random.normal(key, (n, mu.shape[1]))
        mu_cond = jnp.zeros_like(x)
        sigma_cond = jnp.zeros_like(x)
    dt = 1.0 / steps
    for i in range(steps):
        t = jnp.ones((n, 1), dtype=x.dtype) * (i / steps)
        inp = jnp.concatenate([x, t, mu_cond, sigma_cond], axis=1)
        x = x + dt * mlp_apply(params, inp)
    return np.asarray(jax.device_get(x), dtype=np.float32)


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, seed: int = 0, n_proj: int = 128) -> float:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_proj, x.shape[1])).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    xp = np.sort(x @ dirs.T, axis=0)
    yp = np.sort(y @ dirs.T, axis=0)
    m = min(len(xp), len(yp))
    return float(np.mean((xp[:m] - yp[:m]) ** 2))


def train_fm(
    key,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    fit: dict[str, np.ndarray],
    router_params,
    args,
    is_gmm: bool,
):
    xj = jnp.asarray(x_train)
    data_mean = jnp.asarray(x_train.mean(axis=0))
    data_std = jnp.asarray(x_train.std(axis=0) + 1e-6)
    pi = jnp.asarray(fit["pi"])
    mu = jnp.asarray(fit["mu"])
    var = jnp.asarray(fit["var"])
    key, init_key = jax.random.split(key)
    dim = x_train.shape[1]
    params = init_mlp(init_key, [dim * 3 + 1, args.hidden, args.hidden, dim])
    m = zeros_like_tree(params)
    v = zeros_like_tree(params)
    last_loss = 0.0
    for step in range(1, args.fm_steps + 1):
        key, sub = jax.random.split(key)
        inp, target, _, _ = fm_batch(sub, xj, data_mean, data_std, pi, mu, var, router_params, args.batch_size, args.topk, is_gmm)
        params, m, v, loss = fm_step(params, m, v, jnp.asarray(step), inp, target, args.fm_lr)
        last_loss = float(loss)
    key, eval_key, roll_key, source_key = jax.random.split(key, 4)
    valid_mse = eval_fm(
        params,
        eval_key,
        x_valid,
        data_mean,
        data_std,
        pi,
        mu,
        var,
        router_params,
        min(args.batch_size, len(x_valid)),
        args.topk,
        is_gmm,
        args.eval_batches,
    )
    rollout_n = min(args.rollout_samples, len(x_valid))
    rolled = rollout(params, roll_key, rollout_n, data_mean, data_std, pi, mu, var, router_params, args.topk, is_gmm)
    inp, target, x0b, x1b = fm_batch(
        source_key,
        jnp.asarray(x_valid),
        data_mean,
        data_std,
        pi,
        mu,
        var,
        router_params,
        min(args.batch_size, len(x_valid)),
        args.topk,
        is_gmm,
    )
    target_np = np.asarray(jax.device_get(target))
    x0_np = np.asarray(jax.device_get(x0b))
    x1_np = np.asarray(jax.device_get(x1b))
    return {
        "fm_train_loss": last_loss,
        "fm_valid_mse": valid_mse,
        "rollout_swd": sliced_wasserstein(rolled, x_valid[:rollout_n], seed=args.seed),
        "x0_magnitude": float(np.sqrt(np.mean(x0_np * x0_np))),
        "x1_magnitude": float(np.sqrt(np.mean(x1_np * x1_np))),
        "x0_x1_mag_ratio": float(np.sqrt(np.mean(x0_np * x0_np)) / max(np.sqrt(np.mean(x1_np * x1_np)), 1e-8)),
        "source_to_target_dist": float(np.mean(np.linalg.norm(x1_np - x0_np, axis=1))),
        "target_vector_var_trace": float(np.trace(np.cov(target_np.T))),
    }, rolled


def parse_init_config(value: str) -> dict[str, Any]:
    name, strategy, warmup, restarts = value.split(":")
    return {"name": name, "strategy": strategy, "warmup": int(warmup), "restarts": int(restarts)}


def write_outputs(rows: list[dict[str, Any]], examples: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "toy_moe2_fm_summary.json"
    csv_path = out_dir / "toy_moe2_fm_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    datasets = sorted({r["dataset"] for r in rows})
    labels = sorted({r["run_label"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, metric, title in [
        (axes[0], "fm_valid_mse", "FM valid MSE"),
        (axes[1], "rollout_swd", "Rollout sliced-W2"),
    ]:
        width = 0.8 / max(len(labels), 1)
        xs = np.arange(len(datasets))
        for i, label in enumerate(labels):
            vals = []
            for dataset in datasets:
                match = [r for r in rows if r["dataset"] == dataset and r["run_label"] == label]
                vals.append(match[0][metric] if match else np.nan)
            ax.bar(xs + (i - (len(labels) - 1) / 2) * width, vals, width=width, label=label)
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(datasets)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "toy_moe2_fm_summary.png", dpi=180)

    n_rows = len(datasets)
    selected_labels = [
        label
        for label in labels
        if label == "gaussian"
        or "kpp" in label
        or "farthest" in label
        or "pca" in label
        or "split" in label
        or "hybrid" in label
    ]
    selected_labels = selected_labels[:5]
    if selected_labels:
        fig, axes = plt.subplots(n_rows, len(selected_labels), figsize=(4 * len(selected_labels), 3.8 * n_rows), squeeze=False)
        for r_i, dataset in enumerate(datasets):
            for c_i, label in enumerate(selected_labels):
                ax = axes[r_i, c_i]
                rolled, target = examples.get((dataset, label), (None, None))
                if rolled is not None:
                    ax.scatter(target[:, 0], target[:, 1], s=2, c="black", alpha=0.18, linewidths=0)
                    ax.scatter(rolled[:, 0], rolled[:, 1], s=2, c="tab:blue", alpha=0.35, linewidths=0)
                ax.set_title(f"{dataset}\n{label}")
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_dir / "toy_moe2_fm_rollouts.png", dpi=180)

    md_lines = [
        "# Toy MOE2 FM Results",
        "",
        "| dataset | run | init | valid_mse | rollout_swd | router_kl | router_top1 | gmm_nll | dead | count_ratio | x0/x1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: (r["dataset"], r["fm_valid_mse"])):
        md_lines.append(
            f"| {row['dataset']} | {row['run_label']} | {row['init_strategy']} | "
            f"{row['fm_valid_mse']:.6f} | {row['rollout_swd']:.6f} | "
            f"{row.get('router_valid_kl', float('nan')):.6f} | {row.get('router_top1_agreement', float('nan')):.3f} | "
            f"{row.get('gmm_valid_nll', float('nan')):.4f} | {row.get('gmm_dead', 0)} | "
            f"{row.get('gmm_count_ratio', float('nan')):.2f} | {row['x0_x1_mag_ratio']:.3f} |"
        )
    (out_dir / "toy_moe2_fm_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a full toy MOE2-style GMM-router-TIDE-FM ablation.")
    parser.add_argument("--datasets", default="aniso_blobs,nested_rings,pinwheel")
    parser.add_argument("--init-configs", nargs="*", default=[
        "kpp_r3:kmeans++:0:3",
        "kpp_lw5:kmeans++:5:1",
        "farthest_lw5:farthest:5:1",
        "pca_lw5:pca:5:1",
        "split_lw5:split:5:1",
        "hybrid_lw5:hybrid:5:1",
        "quantilepca_lw5:quantilepca:5:1",
    ])
    parser.add_argument("--out-dir", default="toy_moe2_outputs")
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-valid", type=int, default=2048)
    parser.add_argument("--gmm-modes", type=int, default=16)
    parser.add_argument("--gmm-iters", type=int, default=45)
    parser.add_argument("--gmm-floor-frac", type=float, default=0.0)
    parser.add_argument("--router-steps", type=int, default=400)
    parser.add_argument("--router-lr", type=float, default=3e-4)
    parser.add_argument("--fm-steps", type=int, default=800)
    parser.add_argument("--fm-lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--rollout-samples", type=int, default=1024)
    parser.add_argument("--pca-dim", type=int, default=0)
    parser.add_argument("--pca-max-samples", type=int, default=4096)
    parser.add_argument("--standardize", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    init_configs = [parse_init_config(item) for item in args.init_configs]
    rows: list[dict[str, Any]] = []
    examples: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    print("jax", jax.__version__, jax.devices(), flush=True)
    print(vars(args), flush=True)

    for d_i, dataset in enumerate(datasets):
        x_train, y_train, x_valid, y_valid, data_meta = load_dataset_for_ablation(
            dataset,
            args.n_train,
            args.n_valid,
            seed=args.seed + d_i * 100,
            pca_dim=args.pca_dim,
            pca_max_samples=args.pca_max_samples,
            standardize=bool(args.standardize),
        )
        dim = x_train.shape[1]
        dummy_fit = {
            "pi": np.ones((args.gmm_modes,), dtype=np.float32) / args.gmm_modes,
            "mu": np.zeros((args.gmm_modes, dim), dtype=np.float32),
            "var": np.ones((args.gmm_modes, dim), dtype=np.float32),
        }
        dummy_router = init_mlp(jax.random.PRNGKey(args.seed + 17), [dim, args.hidden, args.hidden, args.gmm_modes])
        print(f"\nDataset {dataset}: gaussian baseline", flush=True)
        t0 = time.time()
        fm_metrics, rolled = train_fm(
            jax.random.PRNGKey(args.seed + 1000 + d_i),
            x_train,
            x_valid,
            dummy_fit,
            dummy_router,
            args,
            is_gmm=False,
        )
        rows.append({
            "dataset": dataset,
            "run_label": "gaussian",
            "init_name": "none",
            "init_strategy": "none",
            "init_warmup": 0,
            "init_restarts": 0,
            "gmm_modes": args.gmm_modes,
            "topk": args.topk,
            **data_meta,
            "elapsed_sec": time.time() - t0,
            **fm_metrics,
        })
        examples[(dataset, "gaussian")] = (rolled, x_valid[: len(rolled)])

        for init_i, cfg in enumerate(init_configs):
            print(f"Dataset {dataset}: init {cfg['name']}", flush=True)
            t0 = time.time()
            fit = fit_diag_gmm_np(
                x_train,
                k=args.gmm_modes,
                iters=args.gmm_iters,
                seed=args.seed + d_i * 1000 + init_i * 73,
                init_strategy=cfg["strategy"],
                warmup=cfg["warmup"],
                restarts=cfg["restarts"],
                floor_frac=args.gmm_floor_frac,
            )
            router_params, router_metrics = train_router(
                jax.random.PRNGKey(args.seed + 2000 + d_i * 100 + init_i),
                x_train,
                x_valid,
                fit,
                steps=args.router_steps,
                batch_size=args.batch_size,
                hidden=args.hidden,
                lr=args.router_lr,
            )
            fm_metrics, rolled = train_fm(
                jax.random.PRNGKey(args.seed + 3000 + d_i * 100 + init_i),
                x_train,
                x_valid,
                fit,
                router_params,
                args,
                is_gmm=True,
            )
            gmm_metric_values = gmm_metrics(x_valid, fit)
            row = {
                "dataset": dataset,
                "run_label": f"tide_{cfg['name']}",
                "init_name": cfg["name"],
                "init_strategy": cfg["strategy"],
                "init_warmup": cfg["warmup"],
                "init_restarts": cfg["restarts"],
                "gmm_modes": args.gmm_modes,
                "topk": args.topk,
                **data_meta,
                "elapsed_sec": time.time() - t0,
                **gmm_metric_values,
                **router_metrics,
                **fm_metrics,
            }
            rows.append(row)
            examples[(dataset, row["run_label"])] = (rolled, x_valid[: len(rolled)])
            print(
                f"  valid_mse={row['fm_valid_mse']:.5f} swd={row['rollout_swd']:.5f} "
                f"router_kl={row['router_valid_kl']:.5f} gmm_nll={row['gmm_valid_nll']:.4f}",
                flush=True,
            )

    write_outputs(rows, examples, out_dir)
    print(f"\nWrote {out_dir / 'toy_moe2_fm_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
