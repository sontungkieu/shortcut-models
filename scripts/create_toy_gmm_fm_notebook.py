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
# Toy GMM/FM Insight

This notebook is intentionally self-contained and CPU-friendly. It tests whether
GMM source construction makes flow matching easier or harder on simple 2D data.

Outputs are saved under `toy_outputs/`, including a self-contained
`toy_gmm_fm_executed.ipynb` with embedded plots for downloading after a Kaggle
run.
"""
        ),
        code_cell(
            """
import base64
import csv
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("toy_outputs")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
print("numpy", np.__version__)
"""
        ),
        code_cell(
            """
def make_blobs(n=600, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.3, -1.7], [2.2, -1.5], [-2.0, 1.9], [2.0, 1.7]], dtype=np.float32)
    covs = np.array([
        [[0.22, 0.05], [0.05, 0.14]],
        [[0.18, -0.04], [-0.04, 0.24]],
        [[0.26, 0.00], [0.00, 0.18]],
        [[0.20, 0.07], [0.07, 0.20]],
    ], dtype=np.float32)
    labels = rng.integers(0, len(centers), size=n)
    x = np.empty((n, 2), dtype=np.float32)
    for k in range(len(centers)):
        idx = labels == k
        x[idx] = rng.multivariate_normal(centers[k], covs[k], size=int(idx.sum()))
    return x, labels


def make_rings(n=600, seed=1):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    radius = np.where(labels == 0, 1.25, 2.55) + rng.normal(0, 0.08, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    x = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    x += rng.normal(0, 0.035, size=x.shape)
    return x.astype(np.float32), labels


def make_moons(n=600, seed=2):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    theta = rng.uniform(0, np.pi, size=n)
    x = np.empty((n, 2), dtype=np.float32)
    upper = labels == 0
    x[upper, 0] = np.cos(theta[upper])
    x[upper, 1] = np.sin(theta[upper])
    lower = ~upper
    x[lower, 0] = 1.0 - np.cos(theta[lower])
    x[lower, 1] = -np.sin(theta[lower]) - 0.45
    x *= 1.8
    x += rng.normal(0, 0.075, size=x.shape)
    return x.astype(np.float32), labels


DATASETS = {
    "blobs": make_blobs,
    "rings": make_rings,
    "moons": make_moons,
}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, (name, fn) in zip(axes, DATASETS.items()):
    x, y = fn()
    ax.scatter(x[:, 0], x[:, 1], c=y, s=4, cmap="tab10", alpha=0.55, linewidths=0)
    ax.set_title(name)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Toy target distributions x1")
fig.tight_layout()
fig.savefig(OUT / "toy_datasets.png", dpi=180)
"""
        ),
        code_cell(
            """
def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-12)
    return out if keepdims else np.squeeze(out, axis=axis)


def kmeanspp_init(x, k, rng):
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(0, n)]
    dist2 = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        probs = dist2 / max(float(dist2.sum()), 1e-12)
        centers[i] = x[rng.choice(n, p=probs)]
        dist2 = np.minimum(dist2, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def gmm_log_prob(x, pi, mu, var):
    dim = x.shape[1]
    diff = x[:, None, :] - mu[None, :, :]
    log_comp = -0.5 * (np.sum(diff * diff / var[None], axis=-1) + np.sum(np.log(var), axis=-1) + dim * np.log(2 * np.pi))
    return log_comp + np.log(pi[None] + 1e-12)


def fit_diag_gmm(x, k=8, iters=80, seed=0, floor_frac=0.0, var_prior_strength=0.0, var_prior_target=1.0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    n, dim = x.shape
    global_var = np.var(x, axis=0) + 1e-6
    var_floor = np.maximum((floor_frac ** 2) * global_var, 1e-5)
    mu = kmeanspp_init(x, k, rng)
    var = np.tile(global_var[None], (k, 1)).astype(np.float32)
    pi = np.full((k,), 1.0 / k, dtype=np.float32)
    trace = []
    for it in range(iters):
        logits = gmm_log_prob(x, pi, mu, var)
        log_norm = logsumexp(logits, axis=1, keepdims=True)
        q = np.exp(logits - log_norm)
        nk = q.sum(axis=0) + 1e-8
        pi = (nk / n).astype(np.float32)
        mu = ((q.T @ x) / nk[:, None]).astype(np.float32)
        diff = x[:, None, :] - mu[None]
        ml_var = (np.sum(q[:, :, None] * diff * diff, axis=0) / nk[:, None]).astype(np.float32)
        if var_prior_strength > 0:
            alpha = var_prior_strength / (nk[:, None] + var_prior_strength)
            ml_var = (1.0 - alpha) * ml_var + alpha * float(var_prior_target)
        var = np.maximum(ml_var, var_floor[None]).astype(np.float32)
        nll = -float(np.mean(log_norm))
        hard = np.argmax(q, axis=1)
        counts = np.bincount(hard, minlength=k)
        pi_entropy = -float(np.sum(pi * np.log(pi + 1e-12)) / np.log(k))
        trace.append({
            "iter": it,
            "nll": nll,
            "pi_entropy": pi_entropy,
            "count_ratio": float(counts.max() / max(counts.min(), 1)),
            "dead": int(np.sum(counts == 0)),
            "floor_hit": float(np.mean(var <= var_floor[None] * 1.00001)),
        })
    return {"pi": pi, "mu": mu, "var": var, "var_floor": var_floor, "trace": trace}


def posterior(x, fit):
    logits = gmm_log_prob(x, fit["pi"], fit["mu"], fit["var"])
    log_norm = logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits - log_norm), -np.squeeze(log_norm, axis=1)


def gmm_metrics(x, fit):
    q, nll_vec = posterior(x, fit)
    hard = np.argmax(q, axis=1)
    k = fit["pi"].shape[0]
    counts = np.bincount(hard, minlength=k)
    pi = fit["pi"]
    mu = fit["mu"]
    var = fit["var"]
    center_dist = np.sqrt(np.sum((mu[:, None, :] - mu[None, :, :]) ** 2, axis=-1) + 1e-12)
    mask = ~np.eye(k, dtype=bool)
    avg_std = np.sqrt(np.mean(var, axis=1))
    overlap = np.exp(-center_dist / (avg_std[:, None] + avg_std[None, :] + 1e-6))
    return {
        "nll": float(np.mean(nll_vec)),
        "pi_entropy": float(-np.sum(pi * np.log(pi + 1e-12)) / np.log(k)),
        "count_ratio": float(counts.max() / max(counts.min(), 1)),
        "dead": int(np.sum(counts == 0)),
        "component_var_mean": float(np.mean(var)),
        "floor_hit": float(np.mean(var <= fit["var_floor"][None] * 1.00001)),
        "overlap_max": float(np.max(overlap[mask])) if k > 1 else 0.0,
        "q_entropy": float(np.mean(-np.sum(q * np.log(q + 1e-12), axis=1) / np.log(k))),
        "top1_prob": float(np.mean(np.max(q, axis=1))),
    }
"""
        ),
        code_cell(
            """
def sample_sources(x1, fit, seed=0, topk=2):
    rng = np.random.default_rng(seed)
    n, dim = x1.shape
    q, _ = posterior(x1, fit)
    k = fit["pi"].shape[0]
    data_mean = x1.mean(axis=0)
    data_std = x1.std(axis=0) + 1e-6
    gaussian = rng.normal(data_mean, data_std, size=x1.shape).astype(np.float32)
    hard = np.argmax(q, axis=1)
    hard_sample = fit["mu"][hard] + rng.normal(size=x1.shape) * np.sqrt(fit["var"][hard])
    hard_sample = hard_sample.astype(np.float32)

    top = np.argpartition(q, kth=-topk, axis=1)[:, -topk:]
    top_q = np.take_along_axis(q, top, axis=1)
    top_w = top_q / (top_q.sum(axis=1, keepdims=True) + 1e-12)
    top_mu = fit["mu"][top]
    top_mean = np.sum(top_w[:, :, None] * top_mu, axis=1).astype(np.float32)
    sampled = top_mu + rng.normal(size=top_mu.shape) * np.sqrt(fit["var"][top])
    top_sample_weighted = np.sum(top_w[:, :, None] * sampled, axis=1).astype(np.float32)
    return {
        "gaussian": gaussian,
        "gmm_hard_sample": hard_sample,
        f"gmm_top{topk}_mean": top_mean,
        f"gmm_top{topk}_weighted_sample": top_sample_weighted,
    }


def nearest_data_distance(x, data, max_ref=160):
    ref = data[np.linspace(0, len(data) - 1, min(max_ref, len(data))).astype(int)]
    out = []
    for start in range(0, len(x), 256):
        xb = x[start:start+256]
        d2 = np.sum((xb[:, None, :] - ref[None, :, :]) ** 2, axis=-1)
        out.append(np.sqrt(np.min(d2, axis=1)))
    return np.concatenate(out)


def fm_features(x_t, t, degree=2):
    t = t.reshape(-1, 1)
    feats = [np.ones((len(x_t), 1)), x_t, t]
    if degree >= 2:
        x, y = x_t[:, :1], x_t[:, 1:2]
        feats += [x * x, y * y, x * y, x * t, y * t, t * t]
    return np.concatenate(feats, axis=1)


def ridge_mse(phi, y, ridge=1e-5):
    a = phi.T @ phi + ridge * np.eye(phi.shape[1])
    w = np.linalg.solve(a, phi.T @ y)
    pred = phi @ w
    return float(np.mean((pred - y) ** 2))


def source_fm_metrics(x1, sources, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, x0 in sources.items():
        t = rng.uniform(0, 1, size=(len(x1), 1)).astype(np.float32)
        xt = (1.0 - t) * x0 + t * x1
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
        })
    return rows


def plot_ellipse(ax, mean, var, color):
    from matplotlib.patches import Ellipse
    ell = Ellipse(mean, width=2*np.sqrt(var[0]), height=2*np.sqrt(var[1]), fill=False, color=color, lw=1.4, alpha=0.9)
    ax.add_patch(ell)
"""
        ),
        code_cell(
            """
GMM_CONFIGS = [
    {"name": "ml_k8", "k": 8, "floor_frac": 0.0, "var_prior_strength": 0.0, "var_prior_target": 1.0},
    {"name": "hard05_k8", "k": 8, "floor_frac": 0.5, "var_prior_strength": 0.0, "var_prior_target": 1.0},
    {"name": "soft1_s256_k8", "k": 8, "floor_frac": 0.0, "var_prior_strength": 256.0, "var_prior_target": 1.0},
    {"name": "hard05_k16", "k": 16, "floor_frac": 0.5, "var_prior_strength": 0.0, "var_prior_target": 1.0},
]

all_rows = []
best = {}
for dname, make_fn in DATASETS.items():
    x, labels = make_fn(seed=11)
    for cfg in GMM_CONFIGS:
        fit = fit_diag_gmm(
            x,
            k=cfg["k"],
            iters=25,
            seed=3,
            floor_frac=cfg["floor_frac"],
            var_prior_strength=cfg["var_prior_strength"],
            var_prior_target=cfg["var_prior_target"],
        )
        gm = gmm_metrics(x, fit)
        sources = sample_sources(x, fit, seed=5, topk=2)
        fm_rows = source_fm_metrics(x, sources, seed=13)
        for fm in fm_rows:
            row = {"dataset": dname, "gmm_config": cfg["name"], **cfg, **gm, **fm}
            all_rows.append(row)
        candidate_score = gm["nll"] + 0.02 * math.log(max(gm["count_ratio"], 1.0)) + 2.0 * gm["dead"]
        prev = best.get(dname)
        if prev is None or candidate_score < prev["score"]:
            best[dname] = {"score": candidate_score, "fit": fit, "cfg": cfg, "x": x, "labels": labels}

print("rows", len(all_rows))
print("Best GMM per dataset by simple quality score:")
for dname, item in best.items():
    print(dname, item["cfg"]["name"], "score", round(item["score"], 4), "nll", round(gmm_metrics(item["x"], item["fit"])["nll"], 4))
"""
        ),
        code_cell(
            """
# Plot best GMM ellipses and assignment for each dataset.
fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
for col, dname in enumerate(DATASETS):
    item = best[dname]
    x = item["x"]
    fit = item["fit"]
    q, _ = posterior(x, fit)
    hard = np.argmax(q, axis=1)
    ax = axes[0, col]
    ax.scatter(x[:, 0], x[:, 1], c=hard, s=4, cmap="tab20", alpha=0.6, linewidths=0)
    for i, (m, v) in enumerate(zip(fit["mu"], fit["var"])):
        plot_ellipse(ax, m, v, "black")
        ax.scatter([m[0]], [m[1]], c="white", edgecolors="black", s=35, zorder=5)
    ax.set_title(f"{dname}: {item['cfg']['name']} assignment")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    sources = sample_sources(x, fit, seed=7, topk=2)
    ax = axes[1, col]
    for sname, color in [("gaussian", "gray"), ("gmm_hard_sample", "tab:blue"), ("gmm_top2_mean", "tab:orange")]:
        xs = sources[sname]
        ax.scatter(xs[:800, 0], xs[:800, 1], s=4, alpha=0.38, label=sname, color=color, linewidths=0)
    ax.set_title(f"{dname}: source samples")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if col == 0:
        ax.legend(frameon=False, loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "toy_best_gmm_and_sources.png", dpi=180)
"""
        ),
        code_cell(
            """
# Aggregate and plot metrics.
metrics_path = OUT / "toy_metrics.csv"
with metrics_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)
(OUT / "toy_metrics.json").write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\\n")

def group_mean(rows, keys, value):
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups.setdefault(key, []).append(float(r[value]))
    return {k: float(np.mean(v)) for k, v in groups.items()}

source_order = ["gaussian", "gmm_hard_sample", "gmm_top2_mean", "gmm_top2_weighted_sample"]
dataset_order = list(DATASETS)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric, title in [
    (axes[0], "source_to_target_dist", "Mean |x1 - x0|"),
    (axes[1], "source_nn_dist", "Source off-manifold proxy"),
    (axes[2], "quadratic_fm_mse", "Quadratic FM target MSE"),
]:
    width = 0.18
    xloc = np.arange(len(dataset_order))
    for i, src in enumerate(source_order):
        vals = []
        for d in dataset_order:
            subset = [r for r in all_rows if r["dataset"] == d and r["source"] == src and r["gmm_config"] == best[d]["cfg"]["name"]]
            vals.append(float(np.mean([r[metric] for r in subset])))
        ax.bar(xloc + (i - 1.5) * width, vals, width=width, label=src)
    ax.set_xticks(xloc)
    ax.set_xticklabels(dataset_order)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "toy_source_fm_metrics.png", dpi=180)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, dname in zip(axes, dataset_order):
    rows = [r for r in all_rows if r["dataset"] == dname and r["source"] == "gmm_hard_sample"]
    cfgs = [c["name"] for c in GMM_CONFIGS]
    nll = [np.mean([r["nll"] for r in rows if r["gmm_config"] == c]) for c in cfgs]
    count = [np.mean([r["count_ratio"] for r in rows if r["gmm_config"] == c]) for c in cfgs]
    ax2 = ax.twinx()
    ax.bar(np.arange(len(cfgs)) - 0.18, nll, width=0.36, color="tab:blue", label="NLL")
    ax2.bar(np.arange(len(cfgs)) + 0.18, count, width=0.36, color="tab:red", alpha=0.65, label="count ratio")
    ax.set_title(dname)
    ax.set_xticks(np.arange(len(cfgs)))
    ax.set_xticklabels(cfgs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("NLL")
    ax2.set_ylabel("count ratio")
fig.suptitle("GMM fit quality: NLL can disagree with balance")
fig.tight_layout()
fig.savefig(OUT / "toy_gmm_quality_tradeoff.png", dpi=180)
"""
        ),
        code_cell(
            """
# Build a compact textual summary.
summary_lines = []
summary_lines.append("Toy GMM/FM takeaways")
summary_lines.append("")
summary_lines.append("1. GMM source usually shortens |x1-x0| versus a broad Gaussian source.")
summary_lines.append("2. The top-k mean source can sit between modes; the nearest-data distance plot exposes this blur.")
summary_lines.append("3. Lower GMM NLL is not automatically a better FM source; count ratio and dead components still matter.")
summary_lines.append("4. On curved data such as rings/moons, diagonal GMM needs more modes and can improve likelihood while worsening balance.")
summary_lines.append("5. A simple quadratic vector-field fit is a cheap proxy for FM target complexity before launching large runs.")
summary_text = "\\n".join(summary_lines)
print(summary_text)
(OUT / "toy_summary.txt").write_text(summary_text + "\\n", encoding="utf-8")
"""
        ),
        code_cell(
            """
# Create a self-contained notebook artifact with embedded plots for CLI download.
def image_tag(path, width=920):
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" width="{width}"/>'

plot_paths = [
    OUT / "toy_datasets.png",
    OUT / "toy_best_gmm_and_sources.png",
    OUT / "toy_source_fm_metrics.png",
    OUT / "toy_gmm_quality_tradeoff.png",
]

metric_preview = []
for dname in DATASETS:
    item = best[dname]
    subset = [
        r for r in all_rows
        if r["dataset"] == dname
        and r["gmm_config"] == item["cfg"]["name"]
        and r["source"] in ("gaussian", "gmm_hard_sample", "gmm_top2_mean")
    ]
    metric_preview.extend(subset)

table_lines = [
    "| dataset | best_gmm | source | dist | off_manifold | quad_mse | nll | pi_entropy | count_ratio | dead |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for r in metric_preview:
    table_lines.append(
        f"| {r['dataset']} | {r['gmm_config']} | {r['source']} | "
        f"{r['source_to_target_dist']:.3f} | {r['source_nn_dist']:.3f} | "
        f"{r['quadratic_fm_mse']:.4f} | {r['nll']:.3f} | "
        f"{r['pi_entropy']:.3f} | {r['count_ratio']:.2f} | {r['dead']} |"
    )

executed_nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Toy GMM/FM Insight - executed report\\n",
                "\\n",
                "This notebook was generated by the Kaggle run output. It embeds the key plots and metric table.\\n",
            ],
        },
        {"cell_type": "markdown", "metadata": {}, "source": [summary_text.replace("\\n", "\\n\\n")]},
        {"cell_type": "markdown", "metadata": {}, "source": ["\\n".join(table_lines)]},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for path in plot_paths:
    executed_nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {path.name}\\n\\n{image_tag(path)}"],
    })

executed_path = OUT / "toy_gmm_fm_executed.ipynb"
executed_path.write_text(json.dumps(executed_nb, indent=1) + "\\n", encoding="utf-8")
print("Saved outputs:")
for path in [executed_path, OUT / "toy_metrics.csv", OUT / "toy_metrics.json", OUT / "toy_summary.txt", *plot_paths]:
    print(path)
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
    Path("toy-gmm-fm-insight.ipynb").write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("Wrote toy-gmm-fm-insight.ipynb")


if __name__ == "__main__":
    main()
