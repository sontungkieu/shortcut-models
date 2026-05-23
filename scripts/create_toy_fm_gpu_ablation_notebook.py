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
# Toy FM GPU Ablation

This notebook trains small JAX MLP flow-matching vector fields on toy data.
It is meant for GPU runs; CPU is fine for debugging but slower. It compares
Gaussian FM against GMM sources produced by several GMM initialization
strategies, then gives each source the same FM model and training budget.
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

import jax
import jax.numpy as jnp

OUT = Path("toy_fm_outputs")
OUT.mkdir(parents=True, exist_ok=True)

N_TRAIN = int(os.environ.get("TOY_FM_N_TRAIN", "32768"))
N_VALID = int(os.environ.get("TOY_FM_N_VALID", "8192"))
TRAIN_STEPS = int(os.environ.get("TOY_FM_STEPS", "2200"))
BATCH_SIZE = int(os.environ.get("TOY_FM_BATCH", "512"))
HIDDEN = int(os.environ.get("TOY_FM_HIDDEN", "128"))
LR = float(os.environ.get("TOY_FM_LR", "3e-4"))
SEED = int(os.environ.get("TOY_FM_SEED", "23"))

print("jax", jax.__version__)
print("devices", jax.devices())
print({"n_train": N_TRAIN, "n_valid": N_VALID, "steps": TRAIN_STEPS, "batch": BATCH_SIZE, "hidden": HIDDEN, "lr": LR})
"""
        ),
        code_cell(
            """
def make_nested_rings(n, seed=1):
    rng = np.random.default_rng(seed)
    probs = np.array([0.35, 0.40, 0.25])
    labels = rng.choice(3, size=n, p=probs)
    radius = np.array([1.1, 2.15, 3.05])[labels] + rng.normal(0, 0.065 + 0.015 * labels, size=n)
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


DATASETS = {
    "nested_rings": make_nested_rings,
    "pinwheel": make_pinwheel,
}

toy_data = {}
for i, (name, fn) in enumerate(DATASETS.items()):
    toy_data[name] = (
        *fn(N_TRAIN, seed=SEED + 100 * i),
        *fn(N_VALID, seed=SEED + 100 * i + 1),
    )
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
    z_centers = kmeanspp_init(z.astype(np.float32), k, rng)
    ids = [int(np.argmin(np.sum((z - c) ** 2, axis=1))) for c in z_centers]
    return x[np.asarray(ids)]


def lloyd_warmup(x, centers, steps, rng):
    centers = centers.astype(np.float32).copy()
    for _ in range(int(steps)):
        labels = np.argmin(pairwise_dist2(x, centers), axis=1)
        for j in range(len(centers)):
            pts = x[labels == j]
            centers[j] = pts.mean(axis=0) if len(pts) else x[rng.integers(0, len(x))]
    return centers


def split_init(x, k, rng):
    base = max(1, int(np.ceil(k / 2)))
    centers = lloyd_warmup(x, kmeanspp_init(x, base, rng), 3, rng)
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
        centers = np.concatenate([centers[:j], (centers[j] - delta)[None], (centers[j] + delta)[None], centers[j + 1:]], axis=0)
    return centers[:k].astype(np.float32)


def init_centers(x, k, init_strategy, rng, warmup=0):
    if init_strategy == "kmeans++":
        centers = kmeanspp_init(x, k, rng)
    elif init_strategy == "farthest":
        centers = farthest_init(x, k, rng)
    elif init_strategy == "pca":
        centers = pca_init(x, k, rng)
    elif init_strategy == "split":
        centers = split_init(x, k, rng)
    else:
        raise ValueError(init_strategy)
    if warmup:
        centers = lloyd_warmup(x, centers, warmup, rng)
    return centers.astype(np.float32)


def gmm_log_prob(x, pi, mu, var):
    diff = x[:, None, :] - mu[None, :, :]
    log_comp = -0.5 * (np.sum(diff * diff / var[None], axis=-1) + np.sum(np.log(var), axis=-1) + x.shape[1] * np.log(2 * np.pi))
    return log_comp + np.log(pi[None] + 1e-12)


def fit_diag_gmm_np(x, k=32, iters=40, seed=0, floor_frac=0.0, init_strategy="kmeans++", warmup=0, restarts=1):
    best = None
    best_nll = float("inf")
    for restart in range(int(restarts)):
        fit = _fit_diag_gmm_np_single(
            x,
            k=k,
            iters=iters,
            seed=seed + 1009 * restart,
            floor_frac=floor_frac,
            init_strategy=init_strategy,
            warmup=warmup,
        )
        logits = gmm_log_prob(x, fit["pi"], fit["mu"], fit["var"])
        nll = -float(np.mean(logsumexp(logits, axis=1, keepdims=True)))
        if nll < best_nll:
            best = fit
            best_nll = nll
    best["best_train_nll"] = best_nll
    best["init_strategy"] = init_strategy
    best["warmup"] = int(warmup)
    best["restarts"] = int(restarts)
    return best


def _fit_diag_gmm_np_single(x, k=32, iters=40, seed=0, floor_frac=0.0, init_strategy="kmeans++", warmup=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    data_var = np.var(x, axis=0).astype(np.float32) + 1e-6
    var_floor = np.maximum((floor_frac ** 2) * data_var, 1e-5)
    mu = init_centers(x, k, init_strategy, rng, warmup=warmup)
    var = np.tile(data_var[None], (k, 1)).astype(np.float32)
    pi = np.full((k,), 1.0 / k, dtype=np.float32)
    for _ in range(iters):
        logits = gmm_log_prob(x, pi, mu, var)
        log_norm = logsumexp(logits, axis=1, keepdims=True)
        q = np.exp(logits - log_norm)
        nk = q.sum(axis=0) + 1e-8
        pi = (nk / len(x)).astype(np.float32)
        mu = ((q.T @ x) / nk[:, None]).astype(np.float32)
        diff = x[:, None, :] - mu[None]
        var = np.maximum((np.sum(q[:, :, None] * diff * diff, axis=0) / nk[:, None]).astype(np.float32), var_floor[None])
    return {"pi": pi, "mu": mu, "var": var, "var_floor": var_floor}


def posterior_np(x, fit):
    logits = gmm_log_prob(x, fit["pi"], fit["mu"], fit["var"])
    log_norm = logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits - log_norm)


def make_source_pairs(x1, fit, source, seed=0):
    rng = np.random.default_rng(seed)
    if source == "gaussian":
        x0 = rng.normal(x1.mean(axis=0), x1.std(axis=0) + 1e-6, size=x1.shape).astype(np.float32)
    else:
        q = posterior_np(x1, fit)
        if source == "hard":
            k = np.argmax(q, axis=1)
            x0 = fit["mu"][k] + rng.normal(size=x1.shape) * np.sqrt(fit["var"][k])
        elif source.startswith("top") and source.endswith("_mean"):
            topk = int(source[3:-5])
            top = np.argpartition(q, kth=-topk, axis=1)[:, -topk:]
            w = np.take_along_axis(q, top, axis=1)
            w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
            x0 = np.sum(w[:, :, None] * fit["mu"][top], axis=1)
        elif source.startswith("top") and source.endswith("_sample"):
            topk = int(source[3:-7])
            top = np.argpartition(q, kth=-topk, axis=1)[:, -topk:]
            w = np.take_along_axis(q, top, axis=1)
            w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
            samples = fit["mu"][top] + rng.normal(size=(len(x1), topk, 2)) * np.sqrt(fit["var"][top])
            x0 = np.sum(w[:, :, None] * samples, axis=1)
        else:
            raise ValueError(source)
        x0 = x0.astype(np.float32)
    return x0, x1.astype(np.float32)
"""
        ),
        code_cell(
            """
def init_mlp(key, in_dim=3, hidden=128, out_dim=2, depth=3):
    keys = jax.random.split(key, depth + 1)
    params = []
    prev = in_dim
    for i in range(depth):
        w = jax.random.normal(keys[i], (prev, hidden)) * math.sqrt(2.0 / prev)
        b = jnp.zeros((hidden,))
        params.append({"w": w, "b": b})
        prev = hidden
    w = jax.random.normal(keys[-1], (prev, out_dim)) * math.sqrt(2.0 / prev)
    b = jnp.zeros((out_dim,))
    params.append({"w": w, "b": b})
    return params


def mlp_apply(params, x):
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["w"] + layer["b"])
    return h @ params[-1]["w"] + params[-1]["b"]


def make_batch(x0, x1, key, batch_size):
    n = x0.shape[0]
    key_i, key_t = jax.random.split(key)
    idx = jax.random.randint(key_i, (batch_size,), 0, n)
    x0b = x0[idx]
    x1b = x1[idx]
    t = jax.random.uniform(key_t, (batch_size, 1))
    xt = (1.0 - t) * x0b + t * x1b
    v = x1b - x0b
    inp = jnp.concatenate([xt, t], axis=1)
    return inp, v


@jax.jit
def loss_fn(params, inp, target):
    pred = mlp_apply(params, inp)
    return jnp.mean((pred - target) ** 2)


@jax.jit
def adam_step(params, m, v, step, inp, target, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params, inp, target)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step_f = step.astype(jnp.float32)
    new_params = []
    new_m = []
    new_v = []
    for p_layer, m_layer, v_layer, g_layer in zip(params, m, v, grads):
        p_new_layer = {}
        m_new_layer = {}
        v_new_layer = {}
        for key in p_layer:
            mt = b1 * m_layer[key] + (1 - b1) * g_layer[key]
            vt = b2 * v_layer[key] + (1 - b2) * (g_layer[key] * g_layer[key])
            mt_hat = mt / (1 - b1 ** step_f)
            vt_hat = vt / (1 - b2 ** step_f)
            p_new_layer[key] = p_layer[key] - lr * mt_hat / (jnp.sqrt(vt_hat) + eps)
            m_new_layer[key] = mt
            v_new_layer[key] = vt
        new_params.append(p_new_layer)
        new_m.append(m_new_layer)
        new_v.append(v_new_layer)
    return new_params, new_m, new_v, loss


def zeros_like_params(params):
    return [{k: jnp.zeros_like(v) for k, v in layer.items()} for layer in params]


def eval_mse(params, x0, x1, seed=0, batches=8):
    key = jax.random.PRNGKey(seed)
    vals = []
    x0j = jnp.asarray(x0)
    x1j = jnp.asarray(x1)
    for _ in range(batches):
        key, sub = jax.random.split(key)
        inp, target = make_batch(x0j, x1j, sub, min(BATCH_SIZE, len(x0)))
        vals.append(float(loss_fn(params, inp, target)))
    return float(np.mean(vals))


def rollout(params, x0, steps=64):
    x = jnp.asarray(x0)
    dt = 1.0 / steps
    for i in range(steps):
        t = jnp.ones((x.shape[0], 1), dtype=x.dtype) * (i / steps)
        inp = jnp.concatenate([x, t], axis=1)
        x = x + dt * mlp_apply(params, inp)
    return np.asarray(jax.device_get(x), dtype=np.float32)


def sliced_wasserstein(x, y, seed=0, n_proj=128):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_proj, x.shape[1])).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    xp = np.sort(x @ dirs.T, axis=0)
    yp = np.sort(y @ dirs.T, axis=0)
    m = min(len(xp), len(yp))
    return float(np.mean((xp[:m] - yp[:m]) ** 2))
"""
        ),
        code_cell(
            """
GMM_INIT_CONFIGS = [
    {"name": "kpp_r3", "init": "kmeans++", "warmup": 0, "restarts": 3},
    {"name": "kpp_lw8", "init": "kmeans++", "warmup": 8, "restarts": 1},
    {"name": "farthest_lw8", "init": "farthest", "warmup": 8, "restarts": 1},
    {"name": "pca_lw8", "init": "pca", "warmup": 8, "restarts": 1},
    {"name": "split_lw8", "init": "split", "warmup": 8, "restarts": 1},
]
GMM_SOURCES = ["hard", "top2_mean", "top4_sample"]
summary_rows = []
curve_rows = []
fig_examples = {}

for d_i, (dataset_name, packed) in enumerate(toy_data.items()):
    x_train, y_train, x_valid, y_valid = packed
    print("dataset", dataset_name, flush=True)
    planned_sources = [{"gmm_init": "none", "source": "gaussian", "fit": None}]
    for init_i, init_cfg in enumerate(GMM_INIT_CONFIGS):
        fit = fit_diag_gmm_np(
            x_train,
            k=32,
            iters=45,
            seed=SEED + 100 * d_i + 17 * init_i,
            floor_frac=0.0,
            init_strategy=init_cfg["init"],
            warmup=init_cfg["warmup"],
            restarts=init_cfg["restarts"],
        )
        planned_sources.extend({"gmm_init": init_cfg["name"], "source": source, "fit": fit} for source in GMM_SOURCES)
        print(" fit", init_cfg["name"], "train_nll", round(float(fit["best_train_nll"]), 5), flush=True)
    for s_i, plan in enumerate(planned_sources):
        source = plan["source"]
        fit = plan["fit"]
        source_label = source if fit is None else f"{plan['gmm_init']}:{source}"
        print(" source", source_label, flush=True)
        x0_train, x1_train = make_source_pairs(x_train, fit, source, seed=SEED + 101 * s_i)
        x0_valid, x1_valid = make_source_pairs(x_valid, fit, source, seed=SEED + 101 * s_i + 1)
        x0j = jnp.asarray(x0_train)
        x1j = jnp.asarray(x1_train)
        key = jax.random.PRNGKey(SEED + 1000 * d_i + s_i)
        params = init_mlp(key, hidden=HIDDEN)
        m = zeros_like_params(params)
        vv = zeros_like_params(params)
        t0 = time.time()
        last_loss = None
        for step in range(1, TRAIN_STEPS + 1):
            key, sub = jax.random.split(key)
            inp, target = make_batch(x0j, x1j, sub, BATCH_SIZE)
            params, m, vv, loss = adam_step(params, m, vv, jnp.asarray(step), inp, target, LR)
            if step == 1 or step % 250 == 0 or step == TRAIN_STEPS:
                valid_mse = eval_mse(params, x0_valid, x1_valid, seed=SEED + step + s_i, batches=8)
                last_loss = float(loss)
                curve_rows.append({
                    "dataset": dataset_name,
                    "gmm_init": plan["gmm_init"],
                    "source": source,
                    "source_label": source_label,
                    "step": step,
                    "train_loss": float(loss),
                    "valid_mse": valid_mse,
                })
        rolled = rollout(params, x0_valid[: min(4096, len(x0_valid))], steps=64)
        swd = sliced_wasserstein(rolled, x_valid[: len(rolled)], seed=SEED + s_i)
        source_dist = float(np.mean(np.linalg.norm(x1_valid - x0_valid, axis=1)))
        source_var = float(np.trace(np.cov((x1_valid - x0_valid).T)))
        summary_rows.append({
            "dataset": dataset_name,
            "gmm_init": plan["gmm_init"],
            "source": source,
            "source_label": source_label,
            "gmm_train_nll": float("nan") if fit is None else float(fit["best_train_nll"]),
            "gmm_warmup": 0 if fit is None else int(fit.get("warmup", 0)),
            "gmm_restarts": 0 if fit is None else int(fit.get("restarts", 0)),
            "train_final_loss": last_loss,
            "valid_mse": eval_mse(params, x0_valid, x1_valid, seed=SEED + 999 + s_i, batches=16),
            "rollout_swd": swd,
            "source_to_target_dist": source_dist,
            "target_vector_var_trace": source_var,
            "elapsed_sec": time.time() - t0,
        })
        if source_label in ("gaussian", "kpp_r3:hard", "farthest_lw8:hard", "pca_lw8:top2_mean", "split_lw8:top4_sample"):
            fig_examples[(dataset_name, source_label)] = (rolled, x0_valid[: len(rolled)], x_valid[: len(rolled)])

print("done", len(summary_rows), "runs")
"""
        ),
        code_cell(
            """
with (OUT / "toy_fm_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)
with (OUT / "toy_fm_curves.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(curve_rows[0].keys()))
    writer.writeheader()
    writer.writerows(curve_rows)
(OUT / "toy_fm_summary.json").write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

for dataset_name in DATASETS:
    print("\\n", dataset_name)
    for row in sorted([r for r in summary_rows if r["dataset"] == dataset_name], key=lambda r: r["valid_mse"]):
        print(row["source_label"], "valid_mse", round(row["valid_mse"], 5), "swd", round(row["rollout_swd"], 5), "dist", round(row["source_to_target_dist"], 3))
"""
        ),
        code_cell(
            """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, metric, title in [
    (axes[0], "valid_mse", "FM validation MSE lower is easier"),
    (axes[1], "rollout_swd", "Rollout sliced-W2 lower is better"),
]:
    datasets = list(DATASETS)
    source_labels = ["gaussian"] + [f"{init['name']}:hard" for init in GMM_INIT_CONFIGS]
    width = 0.11
    xs = np.arange(len(datasets))
    for i, source_label in enumerate(source_labels):
        vals = []
        for dataset_name in datasets:
            vals.append([r[metric] for r in summary_rows if r["dataset"] == dataset_name and r["source_label"] == source_label][0])
        ax.bar(xs + (i - (len(source_labels)-1)/2) * width, vals, width=width, label=source_label)
    ax.set_title(title)
    ax.set_xticks(xs)
    ax.set_xticklabels(datasets)
    ax.grid(axis="y", alpha=0.25)
axes[0].legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "toy_fm_summary_bars.png", dpi=190)

example_labels = ["gaussian", "kpp_r3:hard", "farthest_lw8:hard", "pca_lw8:top2_mean", "split_lw8:top4_sample"]
fig, axes = plt.subplots(len(DATASETS), len(example_labels), figsize=(16, 7))
for row_i, dataset_name in enumerate(DATASETS):
    for col_i, source_label in enumerate(example_labels):
        ax = axes[row_i, col_i]
        rolled, x0, x1 = fig_examples[(dataset_name, source_label)]
        ax.scatter(x1[:2000, 0], x1[:2000, 1], s=2, alpha=0.22, color="black", label="target", linewidths=0)
        ax.scatter(rolled[:2000, 0], rolled[:2000, 1], s=2, alpha=0.35, color="tab:blue", label="rollout", linewidths=0)
        ax.set_title(f"{dataset_name} {source_label}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "toy_fm_rollout_examples.png", dpi=190)

fig, ax = plt.subplots(figsize=(10, 5))
for dataset_name in DATASETS:
    for source_label in ["gaussian", "kpp_r3:hard", "farthest_lw8:hard", "pca_lw8:hard", "split_lw8:hard"]:
        rows = [r for r in curve_rows if r["dataset"] == dataset_name and r["source_label"] == source_label]
        ax.plot([r["step"] for r in rows], [r["valid_mse"] for r in rows], label=f"{dataset_name}:{source_label}", alpha=0.8)
ax.set_title("Validation MSE curves")
ax.set_xlabel("step")
ax.set_ylabel("valid MSE")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "toy_fm_valid_curves.png", dpi=190)
"""
        ),
        code_cell(
            """
def image_tag(path, width=940):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" width="{width}"/>'

table = ["| dataset | gmm_init | source | valid_mse | rollout_swd | source_dist | vector_var | gmm_nll |", "|---|---|---|---:|---:|---:|---:|---:|"]
for row in sorted(summary_rows, key=lambda r: (r["dataset"], r["valid_mse"])):
    table.append(f"| {row['dataset']} | {row['gmm_init']} | {row['source']} | {row['valid_mse']:.6f} | {row['rollout_swd']:.6f} | {row['source_to_target_dist']:.4f} | {row['target_vector_var_trace']:.4f} | {row['gmm_train_nll']:.4f} |")

artifact = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Toy FM GPU Ablation - executed report\\n"]},
        {"cell_type": "markdown", "metadata": {}, "source": ["\\n".join(table)]},
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
for path in [OUT / "toy_fm_summary_bars.png", OUT / "toy_fm_rollout_examples.png", OUT / "toy_fm_valid_curves.png"]:
    artifact["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [f"## {path.name}\\n\\n{image_tag(path)}"]})
artifact_path = OUT / "toy_fm_gpu_executed.ipynb"
artifact_path.write_text(json.dumps(artifact, indent=1) + "\\n", encoding="utf-8")
print("Saved outputs")
for p in [artifact_path, OUT / "toy_fm_summary.csv", OUT / "toy_fm_curves.csv", OUT / "toy_fm_summary.json", OUT / "toy_fm_summary_bars.png", OUT / "toy_fm_rollout_examples.png", OUT / "toy_fm_valid_curves.png"]:
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
    Path("toy-fm-gpu-ablation.ipynb").write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("Wrote toy-fm-gpu-ablation.ipynb")


if __name__ == "__main__":
    main()
