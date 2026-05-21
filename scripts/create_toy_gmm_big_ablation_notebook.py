from __future__ import annotations

import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        markdown_cell(
            """
# Bigger Toy GMM Source Ablation

Self-contained CPU/GPU-friendly notebook for stress-testing source construction
before launching CelebA latent jobs. It uses larger 2D datasets, diagonal GMM
EM, k-means/local-Gaussian baselines, and cheap FM target-complexity proxies.
"""
        ),
        code_cell(
            """
import base64
import csv
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("toy_big_outputs")
OUT.mkdir(parents=True, exist_ok=True)

N_TRAIN = int(os.environ.get("TOY_N_TRAIN", "24000"))
N_VALID = int(os.environ.get("TOY_N_VALID", "6000"))
EM_ITERS = int(os.environ.get("TOY_EM_ITERS", "45"))
SEED = int(os.environ.get("TOY_SEED", "17"))
RNG = np.random.default_rng(SEED)

print({"n_train": N_TRAIN, "n_valid": N_VALID, "em_iters": EM_ITERS, "seed": SEED})
print("numpy", np.__version__)
"""
        ),
        code_cell(
            """
def make_aniso_blobs(n, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([
        [-3.5, -2.2], [-0.7, -2.5], [2.6, -1.9],
        [-2.9, 1.9], [0.5, 2.4], [3.4, 1.6],
    ], dtype=np.float32)
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
    return x, labels


def make_nested_rings(n, seed=1):
    rng = np.random.default_rng(seed)
    probs = np.array([0.35, 0.40, 0.25])
    labels = rng.choice(3, size=n, p=probs)
    base_r = np.array([1.1, 2.15, 3.05])[labels]
    radius = base_r + rng.normal(0, 0.065 + 0.015 * labels, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    x = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    x += rng.normal(0, 0.025, size=x.shape)
    return x.astype(np.float32), labels


def make_pinwheel(n, seed=2, arms=6):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, arms, size=n)
    r = rng.gamma(shape=2.2, scale=0.55, size=n) + 0.35
    theta = labels * 2 * np.pi / arms + 0.85 * r + rng.normal(0, 0.13, size=n)
    x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x *= 1.35
    x += rng.normal(0, 0.035, size=x.shape)
    return x.astype(np.float32), labels


def make_cross_cov(n, seed=3):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=n)
    x = np.empty((n, 2), dtype=np.float32)
    for k in range(4):
        idx = labels == k
        m = int(idx.sum())
        t = rng.normal(0, 1.25, size=m)
        noise = rng.normal(0, 0.08, size=m)
        if k == 0:
            pts = np.stack([t, noise + 1.4], axis=1)
        elif k == 1:
            pts = np.stack([t, noise - 1.4], axis=1)
        elif k == 2:
            pts = np.stack([noise + 1.4, t], axis=1)
        else:
            pts = np.stack([noise - 1.4, t], axis=1)
        x[idx] = pts
    x += rng.normal(0, 0.025, size=x.shape)
    return x.astype(np.float32), labels


DATASETS = {
    "aniso_blobs": make_aniso_blobs,
    "nested_rings": make_nested_rings,
    "pinwheel": make_pinwheel,
    "cross_cov": make_cross_cov,
}

toy_data = {}
for i, (name, fn) in enumerate(DATASETS.items()):
    x_train, y_train = fn(N_TRAIN, seed=SEED + 10 * i)
    x_valid, y_valid = fn(N_VALID, seed=SEED + 10 * i + 1)
    toy_data[name] = (x_train, y_train, x_valid, y_valid)

fig, axes = plt.subplots(1, len(toy_data), figsize=(15, 3.8))
for ax, (name, (_, _, x_valid, y_valid)) in zip(axes, toy_data.items()):
    ax.scatter(x_valid[:, 0], x_valid[:, 1], c=y_valid, s=2, cmap="tab10", alpha=0.45, linewidths=0)
    ax.set_title(name)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "datasets.png", dpi=190)
"""
        ),
        code_cell(
            """
def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-12)
    return out if keepdims else np.squeeze(out, axis=axis)


def pairwise_dist2(x, centers):
    return np.maximum(np.sum(x * x, axis=1, keepdims=True) + np.sum(centers * centers, axis=1)[None] - 2 * x @ centers.T, 0.0)


def kmeanspp_init(x, k, rng):
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


def farthest_init(x, k, rng):
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(0, n)]
    dist2 = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        centers[i] = x[int(np.argmax(dist2))]
        dist2 = np.minimum(dist2, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def pca_init(x, k, rng):
    mean = x.mean(axis=0)
    centered = x - mean
    _, _, vt = np.linalg.svd(centered[: min(len(x), 4096)], full_matrices=False)
    basis = vt[: min(2, x.shape[1])].T
    z = centered @ basis
    ids = []
    z_centers = kmeanspp_init(z.astype(np.float32), k, rng)
    for c in z_centers:
        ids.append(int(np.argmin(np.sum((z - c) ** 2, axis=1))))
    return x[np.array(ids)]


def split_init(x, k, rng):
    base = max(1, int(np.ceil(k / 2)))
    centers = kmeanspp_init(x, base, rng)
    centers = lloyd_warmup(x, centers, 3, rng)
    while len(centers) < k:
        labels = np.argmin(pairwise_dist2(x, centers), axis=1)
        scores = []
        for j in range(len(centers)):
            pts = x[labels == j]
            scores.append(0.0 if len(pts) == 0 else len(pts) * float(np.mean(np.var(pts, axis=0))))
        j = int(np.argmax(scores))
        pts = x[labels == j]
        var = np.var(pts, axis=0) if len(pts) else np.var(x, axis=0)
        dim = int(np.argmax(var))
        delta = np.zeros((x.shape[1],), dtype=np.float32)
        delta[dim] = 0.45 * math.sqrt(max(float(var[dim]), 1e-6))
        new_a = centers[j] - delta
        new_b = centers[j] + delta
        centers = np.concatenate([centers[:j], new_a[None], new_b[None], centers[j + 1:]], axis=0)
    return centers[:k].astype(np.float32)


def lloyd_warmup(x, centers, steps, rng):
    centers = centers.astype(np.float32).copy()
    for _ in range(int(steps)):
        labels = np.argmin(pairwise_dist2(x, centers), axis=1)
        for j in range(len(centers)):
            pts = x[labels == j]
            centers[j] = pts.mean(axis=0) if len(pts) else x[rng.integers(0, len(x))]
    return centers


def init_centers(x, k, strategy, rng, warmup=0):
    if strategy == "kmeans++":
        centers = kmeanspp_init(x, k, rng)
    elif strategy == "farthest":
        centers = farthest_init(x, k, rng)
    elif strategy == "pca":
        centers = pca_init(x, k, rng)
    elif strategy == "split":
        centers = split_init(x, k, rng)
    elif strategy == "random":
        centers = x[rng.choice(len(x), size=k, replace=len(x) < k)].copy()
    else:
        raise ValueError(strategy)
    if warmup:
        centers = lloyd_warmup(x, centers, warmup, rng)
    return centers.astype(np.float32)
"""
        ),
        code_cell(
            """
def gmm_log_prob(x, pi, mu, var):
    dim = x.shape[1]
    diff = x[:, None, :] - mu[None, :, :]
    log_comp = -0.5 * (np.sum(diff * diff / var[None], axis=-1) + np.sum(np.log(var), axis=-1) + dim * np.log(2 * np.pi))
    return log_comp + np.log(pi[None] + 1e-12)


def posterior(x, fit):
    logits = gmm_log_prob(x, fit["pi"], fit["mu"], fit["var"])
    log_norm = logsumexp(logits, axis=1, keepdims=True)
    q = np.exp(logits - log_norm)
    return q, -np.squeeze(log_norm, axis=1)


def fit_diag_gmm(x, k, init_strategy="kmeans++", warmup=0, iters=40, seed=0, floor_frac=0.0, pi_prior=0.0, var_prior=0.0, var_target=1.0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    n, dim = x.shape
    data_var = np.var(x, axis=0).astype(np.float32) + 1e-6
    var_floor = np.maximum((floor_frac ** 2) * data_var, 1e-5)
    mu = init_centers(x, k, init_strategy, rng, warmup=warmup)
    labels = np.argmin(pairwise_dist2(x, mu), axis=1)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    pi = (counts + pi_prior) / max(float(np.sum(counts + pi_prior)), 1e-12)
    var = np.tile(data_var[None], (k, 1)).astype(np.float32)
    for j in range(k):
        pts = x[labels == j]
        if len(pts):
            mu[j] = pts.mean(axis=0)
            var[j] = np.maximum(np.var(pts, axis=0), var_floor)
    trace = []
    for it in range(int(iters)):
        logits = gmm_log_prob(x, pi.astype(np.float32), mu, var)
        log_norm = logsumexp(logits, axis=1, keepdims=True)
        q = np.exp(logits - log_norm)
        nk = q.sum(axis=0) + 1e-8
        pi = (nk + pi_prior) / max(float(np.sum(nk + pi_prior)), 1e-12)
        mu = ((q.T @ x) / nk[:, None]).astype(np.float32)
        diff = x[:, None, :] - mu[None]
        ml_var = (np.sum(q[:, :, None] * diff * diff, axis=0) / nk[:, None]).astype(np.float32)
        if var_prior > 0:
            alpha = var_prior / (nk[:, None] + var_prior)
            ml_var = (1 - alpha) * ml_var + alpha * float(var_target)
        var = np.maximum(ml_var, var_floor[None]).astype(np.float32)
        hard = np.argmax(q, axis=1)
        hard_counts = np.bincount(hard, minlength=k)
        trace.append({
            "iter": it,
            "nll": -float(np.mean(log_norm)),
            "dead": int(np.sum(hard_counts == 0)),
            "count_ratio": float(hard_counts.max() / max(hard_counts.min(), 1)),
            "pi_entropy": float(-np.sum(pi * np.log(pi + 1e-12)) / np.log(k)),
            "floor_hit": float(np.mean(var <= var_floor[None] * 1.00001)),
        })
    return {"model": "diag_gmm", "pi": pi.astype(np.float32), "mu": mu, "var": var, "var_floor": var_floor, "trace": trace, "init_strategy": init_strategy, "warmup": warmup}


def fit_kmeans_local_gaussian(x, k, init_strategy="kmeans++", warmup=10, seed=0, floor_frac=0.0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    data_var = np.var(x, axis=0).astype(np.float32) + 1e-6
    var_floor = np.maximum((floor_frac ** 2) * data_var, 1e-5)
    centers = init_centers(x, k, init_strategy, rng, warmup=warmup)
    labels = np.argmin(pairwise_dist2(x, centers), axis=1)
    pi = np.zeros((k,), dtype=np.float32)
    mu = centers.copy()
    var = np.tile(data_var[None], (k, 1)).astype(np.float32)
    for j in range(k):
        pts = x[labels == j]
        pi[j] = len(pts) / len(x)
        if len(pts):
            mu[j] = pts.mean(axis=0)
            var[j] = np.maximum(np.var(pts, axis=0), var_floor)
    pi = np.maximum(pi, 1e-8)
    pi = pi / pi.sum()
    return {"model": "kmeans_local_gaussian", "pi": pi, "mu": mu, "var": var, "var_floor": var_floor, "trace": [], "init_strategy": init_strategy, "warmup": warmup}
"""
        ),
        code_cell(
            """
def cluster_scores(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    true_vals = np.unique(labels_true)
    pred_vals = np.unique(labels_pred)
    contingency = np.zeros((len(true_vals), len(pred_vals)), dtype=np.float64)
    for i, t in enumerate(true_vals):
        for j, p in enumerate(pred_vals):
            contingency[i, j] = np.sum((labels_true == t) & (labels_pred == p))
    purity = float(np.sum(np.max(contingency, axis=0)) / max(n, 1))
    pi = contingency.sum(axis=1) / max(n, 1)
    pj = contingency.sum(axis=0) / max(n, 1)
    pij = contingency / max(n, 1)
    mi = 0.0
    for i in range(len(true_vals)):
        for j in range(len(pred_vals)):
            if pij[i, j] > 0 and pi[i] > 0 and pj[j] > 0:
                mi += pij[i, j] * math.log(pij[i, j] / (pi[i] * pj[j]))
    ht = -float(np.sum(pi[pi > 0] * np.log(pi[pi > 0])))
    hp = -float(np.sum(pj[pj > 0] * np.log(pj[pj > 0])))
    nmi = float(2 * mi / max(ht + hp, 1e-12))
    return purity, nmi


def model_metrics(x, y, fit):
    q, nll_vec = posterior(x, fit)
    hard = np.argmax(q, axis=1)
    k = len(fit["pi"])
    counts = np.bincount(hard, minlength=k)
    pi = fit["pi"]
    mu = fit["mu"]
    var = fit["var"]
    purity, nmi = cluster_scores(y, hard)
    dist = np.sqrt(pairwise_dist2(mu, mu) + 1e-12)
    mask = ~np.eye(k, dtype=bool)
    avg_std = np.sqrt(np.mean(var, axis=1))
    overlap = np.exp(-dist / (avg_std[:, None] + avg_std[None, :] + 1e-6))
    entropy = -np.sum(q * np.log(q + 1e-12), axis=1)
    return {
        "nll": float(np.mean(nll_vec)),
        "dead": int(np.sum(counts == 0)),
        "count_ratio": float(counts.max() / max(counts.min(), 1)),
        "pi_entropy": float(-np.sum(pi * np.log(pi + 1e-12)) / np.log(k)),
        "q_entropy": float(np.mean(entropy) / np.log(k)),
        "top1_prob": float(np.mean(np.max(q, axis=1))),
        "purity": purity,
        "nmi": nmi,
        "component_var_mean": float(np.mean(var)),
        "floor_hit": float(np.mean(var <= fit["var_floor"][None] * 1.00001)),
        "overlap_max": float(np.max(overlap[mask])) if k > 1 else 0.0,
    }


def nearest_data_distance(x, data, max_ref=1200):
    ref = data[np.linspace(0, len(data) - 1, min(max_ref, len(data))).astype(int)]
    outs = []
    for start in range(0, len(x), 512):
        xb = x[start:start + 512]
        d2 = np.sum((xb[:, None, :] - ref[None, :, :]) ** 2, axis=-1)
        outs.append(np.sqrt(np.min(d2, axis=1)))
    return np.concatenate(outs)


def fm_features(x_t, t, degree=2):
    t = t.reshape(-1, 1)
    feats = [np.ones((len(x_t), 1)), x_t, t]
    if degree >= 2:
        x, y = x_t[:, :1], x_t[:, 1:2]
        feats += [x * x, y * y, x * y, x * t, y * t, t * t]
    if degree >= 3:
        x, y = x_t[:, :1], x_t[:, 1:2]
        feats += [x ** 3, y ** 3, x * x * y, x * y * y, x * x * t, y * y * t, x * t * t, y * t * t, t ** 3]
    return np.concatenate(feats, axis=1).astype(np.float64)


def ridge_mse(phi, y, ridge=1e-5):
    y = y.astype(np.float64)
    a = phi.T @ phi + ridge * np.eye(phi.shape[1])
    w = np.linalg.solve(a, phi.T @ y)
    pred = phi @ w
    return float(np.mean((pred - y) ** 2))


def sample_sources(x1, fit, seed=0, topk_values=(2, 4)):
    rng = np.random.default_rng(seed)
    q, _ = posterior(x1, fit)
    data_mean = x1.mean(axis=0)
    data_std = x1.std(axis=0) + 1e-6
    out = {"gaussian": rng.normal(data_mean, data_std, size=x1.shape).astype(np.float32)}
    hard = np.argmax(q, axis=1)
    out["hard_sample"] = (fit["mu"][hard] + rng.normal(size=x1.shape) * np.sqrt(fit["var"][hard])).astype(np.float32)
    for topk in topk_values:
        top = np.argpartition(q, kth=-topk, axis=1)[:, -topk:]
        top_q = np.take_along_axis(q, top, axis=1)
        w = top_q / (top_q.sum(axis=1, keepdims=True) + 1e-12)
        top_mu = fit["mu"][top]
        out[f"top{topk}_mean"] = np.sum(w[:, :, None] * top_mu, axis=1).astype(np.float32)
        sampled = top_mu + rng.normal(size=top_mu.shape) * np.sqrt(fit["var"][top])
        out[f"top{topk}_weighted_sample"] = np.sum(w[:, :, None] * sampled, axis=1).astype(np.float32)
    return out


def source_metrics(x1, fit, seed=0):
    sources = sample_sources(x1, fit, seed=seed)
    rng = np.random.default_rng(seed + 123)
    rows = []
    for name, x0 in sources.items():
        t = rng.uniform(0, 1, size=(len(x1), 1)).astype(np.float32)
        xt = (1 - t) * x0 + t * x1
        v = x1 - x0
        nn = nearest_data_distance(x0, x1)
        rows.append({
            "source": name,
            "source_to_target_dist": float(np.mean(np.linalg.norm(v, axis=1))),
            "target_vector_var_trace": float(np.trace(np.cov(v.T))),
            "source_nn_dist": float(np.mean(nn)),
            "source_nn_dist_p90": float(np.quantile(nn, 0.90)),
            "linear_fm_mse": ridge_mse(fm_features(xt, t, degree=1), v),
            "quadratic_fm_mse": ridge_mse(fm_features(xt, t, degree=2), v),
            "cubic_fm_mse": ridge_mse(fm_features(xt, t, degree=3), v),
        })
    return rows
"""
        ),
        code_cell(
            """
CONFIGS = []
for k in [8, 16, 32]:
    CONFIGS.extend([
        {"name": f"gmm_k{k}_kpp_r3", "model": "diag_gmm", "k": k, "init": "kmeans++", "warmup": 0, "restarts": 3, "floor": 0.0, "pi_prior": 0.0, "var_prior": 0.0},
        {"name": f"gmm_k{k}_kpp_lw8", "model": "diag_gmm", "k": k, "init": "kmeans++", "warmup": 8, "restarts": 1, "floor": 0.0, "pi_prior": 0.0, "var_prior": 0.0},
        {"name": f"gmm_k{k}_far_lw8", "model": "diag_gmm", "k": k, "init": "farthest", "warmup": 8, "restarts": 1, "floor": 0.0, "pi_prior": 0.0, "var_prior": 0.0},
        {"name": f"gmm_k{k}_pca_lw8", "model": "diag_gmm", "k": k, "init": "pca", "warmup": 8, "restarts": 1, "floor": 0.0, "pi_prior": 0.0, "var_prior": 0.0},
        {"name": f"gmm_k{k}_split_lw8", "model": "diag_gmm", "k": k, "init": "split", "warmup": 8, "restarts": 1, "floor": 0.0, "pi_prior": 0.0, "var_prior": 0.0},
        {"name": f"gmm_k{k}_hard05", "model": "diag_gmm", "k": k, "init": "kmeans++", "warmup": 8, "restarts": 1, "floor": 0.5, "pi_prior": 0.0, "var_prior": 0.0},
        {"name": f"gmm_k{k}_softv1_s256", "model": "diag_gmm", "k": k, "init": "kmeans++", "warmup": 8, "restarts": 1, "floor": 0.0, "pi_prior": 0.0, "var_prior": 256.0},
        {"name": f"kmeans_gauss_k{k}_lw12", "model": "kmeans_local_gaussian", "k": k, "init": "kmeans++", "warmup": 12, "restarts": 1, "floor": 0.0, "pi_prior": 0.0, "var_prior": 0.0},
    ])

print("configs", len(CONFIGS))
all_rows = []
best_by_dataset = {}
start_time = time.time()
for d_i, (dataset_name, (x_train, y_train, x_valid, y_valid)) in enumerate(toy_data.items()):
    print("dataset", dataset_name, x_train.shape, flush=True)
    for c_i, cfg in enumerate(CONFIGS):
        best_fit = None
        best_nll = float("inf")
        for restart in range(int(cfg.get("restarts", 1))):
            seed = SEED + 1000 * d_i + 37 * c_i + restart
            if cfg["model"] == "diag_gmm":
                fit = fit_diag_gmm(
                    x_train,
                    k=cfg["k"],
                    init_strategy=cfg["init"],
                    warmup=cfg["warmup"],
                    iters=EM_ITERS,
                    seed=seed,
                    floor_frac=cfg["floor"],
                    pi_prior=cfg["pi_prior"],
                    var_prior=cfg["var_prior"],
                    var_target=1.0,
                )
            else:
                fit = fit_kmeans_local_gaussian(
                    x_train,
                    k=cfg["k"],
                    init_strategy=cfg["init"],
                    warmup=cfg["warmup"],
                    seed=seed,
                    floor_frac=cfg["floor"],
                )
            train_nll = model_metrics(x_train, y_train, fit)["nll"]
            if train_nll < best_nll:
                best_nll = train_nll
                best_fit = fit
        train_m = model_metrics(x_train, y_train, best_fit)
        valid_m = model_metrics(x_valid, y_valid, best_fit)
        src_rows = source_metrics(x_valid, best_fit, seed=SEED + c_i)
        for src in src_rows:
            row = {
                "dataset": dataset_name,
                "config": cfg["name"],
                "model": cfg["model"],
                "k": cfg["k"],
                "init": cfg["init"],
                "warmup": cfg["warmup"],
                "floor": cfg["floor"],
                "var_prior": cfg["var_prior"],
                "train_nll": train_m["nll"],
                "valid_nll": valid_m["nll"],
                "valid_dead": valid_m["dead"],
                "valid_count_ratio": valid_m["count_ratio"],
                "valid_pi_entropy": valid_m["pi_entropy"],
                "valid_q_entropy": valid_m["q_entropy"],
                "valid_top1_prob": valid_m["top1_prob"],
                "valid_purity": valid_m["purity"],
                "valid_nmi": valid_m["nmi"],
                "valid_component_var_mean": valid_m["component_var_mean"],
                "valid_floor_hit": valid_m["floor_hit"],
                "valid_overlap_max": valid_m["overlap_max"],
                **src,
            }
            all_rows.append(row)
        quality = valid_m["nll"] + 0.02 * math.log(max(valid_m["count_ratio"], 1.0)) + 1.5 * valid_m["dead"] + 0.05 * valid_m["overlap_max"]
        prev = best_by_dataset.get(dataset_name)
        if prev is None or quality < prev["quality"]:
            best_by_dataset[dataset_name] = {"quality": quality, "cfg": cfg, "fit": best_fit, "metrics": valid_m}
        print(" ", cfg["name"], "valid_nll", round(valid_m["nll"], 4), "dead", valid_m["dead"], "ratio", round(valid_m["count_ratio"], 2), flush=True)

print("elapsed_sec", round(time.time() - start_time, 2))
"""
        ),
        code_cell(
            """
metrics_path = OUT / "toy_big_metrics.csv"
with metrics_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)
(OUT / "toy_big_metrics.json").write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

summary = []
summary.append("Best config by dataset")
for dataset_name, item in best_by_dataset.items():
    m = item["metrics"]
    summary.append(
        f"{dataset_name}: {item['cfg']['name']} valid_nll={m['nll']:.4f} dead={m['dead']} "
        f"ratio={m['count_ratio']:.2f} nmi={m['nmi']:.3f} overlap={m['overlap_max']:.4f}"
    )
summary_text = "\\n".join(summary)
print(summary_text)
(OUT / "toy_big_summary.txt").write_text(summary_text + "\\n", encoding="utf-8")
"""
        ),
        code_cell(
            """
def plot_ellipse(ax, mean, var, color="black"):
    from matplotlib.patches import Ellipse
    ell = Ellipse(mean, width=2 * math.sqrt(max(var[0], 1e-8)), height=2 * math.sqrt(max(var[1], 1e-8)), fill=False, color=color, lw=1.1, alpha=0.85)
    ax.add_patch(ell)

fig, axes = plt.subplots(2, len(toy_data), figsize=(15, 7.2))
for col, (dataset_name, (_, _, x_valid, y_valid)) in enumerate(toy_data.items()):
    item = best_by_dataset[dataset_name]
    fit = item["fit"]
    q, _ = posterior(x_valid, fit)
    hard = np.argmax(q, axis=1)
    ax = axes[0, col]
    ax.scatter(x_valid[:, 0], x_valid[:, 1], c=hard, s=2, cmap="tab20", alpha=0.55, linewidths=0)
    for mean, var in zip(fit["mu"], fit["var"]):
        plot_ellipse(ax, mean, var)
    ax.set_title(f"{dataset_name}\\n{item['cfg']['name']}")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = axes[1, col]
    sources = sample_sources(x_valid, fit, seed=SEED)
    for sname, color in [("gaussian", "gray"), ("hard_sample", "tab:blue"), ("top2_mean", "tab:orange"), ("top4_mean", "tab:green")]:
        xs = sources[sname]
        ax.scatter(xs[:2500, 0], xs[:2500, 1], s=2, color=color, alpha=0.30, linewidths=0, label=sname)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "best_assignments_and_sources.png", dpi=190)

dataset_names = list(toy_data.keys())
source_names = ["gaussian", "hard_sample", "top2_mean", "top4_mean", "top2_weighted_sample", "top4_weighted_sample"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, metric, title in [
    (axes[0], "source_to_target_dist", "Mean |x1-x0|"),
    (axes[1], "source_nn_dist", "Off-manifold proxy"),
    (axes[2], "quadratic_fm_mse", "Quadratic FM proxy MSE"),
]:
    width = 0.12
    xs = np.arange(len(dataset_names))
    for i, source in enumerate(source_names):
        vals = []
        for dataset_name in dataset_names:
            best_cfg = best_by_dataset[dataset_name]["cfg"]["name"]
            subset = [r for r in all_rows if r["dataset"] == dataset_name and r["config"] == best_cfg and r["source"] == source]
            vals.append(np.mean([r[metric] for r in subset]))
        ax.bar(xs + (i - (len(source_names) - 1) / 2) * width, vals, width=width, label=source)
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_xticklabels(dataset_names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
axes[0].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "source_proxy_metrics.png", dpi=190)

fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
for ax, dataset_name in zip(axes.ravel(), dataset_names):
    rows = [r for r in all_rows if r["dataset"] == dataset_name and r["source"] == "hard_sample"]
    top = sorted(rows, key=lambda r: (r["valid_dead"], r["valid_nll"]))[:12]
    labels = [r["config"] for r in top]
    nll = [r["valid_nll"] for r in top]
    ratio = [min(r["valid_count_ratio"], 100.0) for r in top]
    ax2 = ax.twinx()
    ax.bar(np.arange(len(top)) - 0.18, nll, width=0.36, color="tab:blue", label="valid NLL")
    ax2.bar(np.arange(len(top)) + 0.18, ratio, width=0.36, color="tab:red", alpha=0.62, label="count ratio cap100")
    ax.set_title(dataset_name)
    ax.set_xticks(np.arange(len(top)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("valid NLL")
    ax2.set_ylabel("count ratio")
fig.tight_layout()
fig.savefig(OUT / "gmm_quality_tradeoff_top12.png", dpi=190)
"""
        ),
        code_cell(
            """
def image_tag(path, width=940):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" width="{width}"/>'

table = [
    "| dataset | best config | valid_nll | dead | count_ratio | NMI | overlap |",
    "|---|---|---:|---:|---:|---:|---:|",
]
for dataset_name, item in best_by_dataset.items():
    m = item["metrics"]
    table.append(f"| {dataset_name} | {item['cfg']['name']} | {m['nll']:.4f} | {m['dead']} | {m['count_ratio']:.2f} | {m['nmi']:.3f} | {m['overlap_max']:.5f} |")

artifact_nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Bigger Toy GMM Source Ablation - executed report\\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": [summary_text.replace("\\n", "\\n\\n")]},
        {"cell_type": "markdown", "metadata": {}, "source": ["\\n".join(table)]},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
for path in [
    OUT / "datasets.png",
    OUT / "best_assignments_and_sources.png",
    OUT / "source_proxy_metrics.png",
    OUT / "gmm_quality_tradeoff_top12.png",
]:
    artifact_nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [f"## {path.name}\\n\\n{image_tag(path)}"]})

artifact_path = OUT / "toy_gmm_big_executed.ipynb"
artifact_path.write_text(json.dumps(artifact_nb, indent=1) + "\\n", encoding="utf-8")
print("Saved outputs")
for p in [
    artifact_path,
    OUT / "toy_big_metrics.csv",
    OUT / "toy_big_metrics.json",
    OUT / "toy_big_summary.txt",
    OUT / "datasets.png",
    OUT / "best_assignments_and_sources.png",
    OUT / "source_proxy_metrics.png",
    OUT / "gmm_quality_tradeoff_top12.png",
]:
    print(p)
"""
        ),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path("toy-gmm-big-ablation.ipynb").write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("Wrote toy-gmm-big-ablation.ipynb")


if __name__ == "__main__":
    main()
