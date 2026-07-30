#!/usr/bin/env python3
"""Plot learned denoising paths from one or more eval-trajectory NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def parse_trajectory_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=/path/to/trajectory.npz")
    label, path = value.split("=", 1)
    label = label.strip()
    path = Path(path).expanduser()
    if not label:
        raise argparse.ArgumentTypeError("Trajectory label cannot be empty.")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Trajectory file does not exist: {path}")
    return label, path


def load_trajectory(label: str, path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        required = {"states", "times", "save_steps"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")
        states = np.asarray(payload["states"], dtype=np.float32)
        times = np.asarray(payload["times"], dtype=np.float32)
        save_steps = np.asarray(payload["save_steps"], dtype=np.int32)
        metadata = {}
        if "metadata_json" in payload.files:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        metrics = {}
        for name in (
            "path_length",
            "endpoint_displacement",
            "straightness_ratio",
            "curvature_proxy",
        ):
            if name in payload.files:
                metrics[name] = np.asarray(payload[name], dtype=np.float32)
    if states.ndim < 3:
        raise ValueError(f"{path}: states must have shape [sample, time, ...].")
    if states.shape[1] != times.size or times.size != save_steps.size:
        raise ValueError(
            f"{path}: states/time/save-step dimensions disagree: "
            f"{states.shape[1]}, {times.size}, {save_steps.size}"
        )
    return {
        "label": label,
        "path": str(path.resolve()),
        "states": states,
        "times": times,
        "save_steps": save_steps,
        "metadata": metadata,
        "metrics": metrics,
    }


def fit_randomized_pca(
    matrix: np.ndarray,
    *,
    n_components: int = 2,
    random_seed: int = 0,
    oversample: int = 8,
    power_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("PCA input must be a two-dimensional matrix.")
    if min(matrix.shape) < n_components:
        raise ValueError(
            f"PCA requires at least {n_components} samples/features, got {matrix.shape}."
        )
    mean = np.mean(matrix, axis=0, dtype=np.float64).astype(np.float32)
    centered = matrix - mean
    rank = min(n_components + oversample, min(centered.shape))
    rng = np.random.default_rng(random_seed)
    omega = rng.standard_normal((centered.shape[1], rank), dtype=np.float32)
    projected = centered @ omega
    for _ in range(max(int(power_iterations), 0)):
        projected = centered @ (centered.T @ projected)
        projected, _ = np.linalg.qr(projected, mode="reduced")
    basis, _ = np.linalg.qr(projected, mode="reduced")
    small = basis.T @ centered
    _, singular_values, components = np.linalg.svd(small, full_matrices=False)
    components = components[:n_components].astype(np.float32)
    explained = np.square(singular_values[:n_components].astype(np.float64))
    total = float(np.sum(np.square(centered.astype(np.float64))))
    explained_ratio = explained / max(total, 1e-12)
    return mean, components, explained_ratio.astype(np.float32)


def project_trajectories(records: list[dict]) -> tuple[list[np.ndarray], dict]:
    flattened = [
        record["states"].reshape(
            record["states"].shape[0] * record["states"].shape[1],
            -1,
        )
        for record in records
    ]
    feature_dims = {matrix.shape[1] for matrix in flattened}
    if len(feature_dims) != 1:
        raise ValueError(f"All trajectory files must share one latent dimension, got {feature_dims}.")
    union = np.concatenate(flattened, axis=0)
    mean, components, explained_ratio = fit_randomized_pca(union)
    projected = []
    for record, matrix in zip(records, flattened):
        coords = (matrix - mean) @ components.T
        projected.append(coords.reshape(record["states"].shape[:2] + (2,)))
    pca_info = {
        "fit_samples": int(union.shape[0]),
        "feature_dim": int(union.shape[1]),
        "explained_variance_ratio": [float(value) for value in explained_ratio],
        "basis_fit": "union of all samples, saved times, and supplied models",
        "random_seed": 0,
    }
    return projected, pca_info


def _shared_limits(projected: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    values = np.concatenate([coords.reshape(-1, 2) for coords in projected], axis=0)
    low = np.quantile(values, 0.005, axis=0)
    high = np.quantile(values, 0.995, axis=0)
    span = np.maximum(high - low, 1e-6)
    return (
        (float(low[0] - 0.08 * span[0]), float(high[0] + 0.08 * span[0])),
        (float(low[1] - 0.08 * span[1]), float(high[1] + 0.08 * span[1])),
    )


def _add_time_colored_path(ax, coords: np.ndarray, times: np.ndarray, *, alpha: float) -> None:
    points = coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = 0.5 * (times[:-1] + times[1:])
    collection = LineCollection(
        segments,
        cmap="viridis",
        norm=plt.Normalize(float(times[0]), float(times[-1])),
        linewidths=1.0,
        alpha=alpha,
    )
    collection.set_array(colors)
    ax.add_collection(collection)


def plot_individual_paths(
    records: list[dict],
    projected: list[np.ndarray],
    output_dir: Path,
    *,
    max_paths: int,
) -> list[str]:
    columns = 2 if len(records) > 1 else 1
    rows = int(np.ceil(len(records) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(7.0 * columns, 6.2 * rows), squeeze=False)
    xlim, ylim = _shared_limits(projected)
    colorbar_collection = None
    for index, (record, coords) in enumerate(zip(records, projected)):
        ax = axes.flat[index]
        count = min(int(max_paths), coords.shape[0])
        sample_ids = np.linspace(0, coords.shape[0] - 1, count).round().astype(np.int32)
        for sample_id in sample_ids:
            _add_time_colored_path(
                ax,
                coords[sample_id],
                record["times"],
                alpha=max(0.25, min(0.8, 8.0 / max(count, 1))),
            )
        ax.scatter(coords[sample_ids, 0, 0], coords[sample_ids, 0, 1], s=24, c="#2563eb", marker="o", label="$x_0$")
        ax.scatter(coords[sample_ids, -1, 0], coords[sample_ids, -1, 1], s=30, c="#dc2626", marker="x", label="$x_1$")
        ax.set_title(record["label"])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        colorbar_collection = ax.collections[0]
    for index in range(len(records), rows * columns):
        axes.flat[index].axis("off")
    if colorbar_collection is not None:
        fig.colorbar(colorbar_collection, ax=axes.ravel().tolist(), label="ODE time $t$", shrink=0.8)
    fig.suptitle("Learned denoising trajectories in one shared PCA basis", fontsize=14)
    fig.subplots_adjust(top=0.92, right=0.92, hspace=0.28, wspace=0.22)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"denoising_trajectory_pca_paths.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_mean_paths(records: list[dict], projected: list[np.ndarray], output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    colors = plt.get_cmap("tab10")
    for index, (record, coords) in enumerate(zip(records, projected)):
        mean_path = np.mean(coords, axis=0)
        color = colors(index % 10)
        ax.plot(mean_path[:, 0], mean_path[:, 1], color=color, linewidth=2.2, label=record["label"])
        ax.plot(
            [mean_path[0, 0], mean_path[-1, 0]],
            [mean_path[0, 1], mean_path[-1, 1]],
            color=color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.7,
        )
        ax.scatter(mean_path[0, 0], mean_path[0, 1], color=color, marker="o", s=42)
        ax.scatter(mean_path[-1, 0], mean_path[-1, 1], color=color, marker="x", s=52)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Mean learned path; dashed line is the direct endpoint chord")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"denoising_trajectory_pca_mean.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_metric_boxes(records: list[dict], output_dir: Path) -> list[str]:
    metric_names = [
        ("path_length", "Path length $L$"),
        ("endpoint_displacement", "Endpoint displacement $D$"),
        ("straightness_ratio", "Straightness $L/D$"),
        ("curvature_proxy", "Curvature proxy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), squeeze=False)
    for ax, (metric_name, title) in zip(axes.flat, metric_names):
        available = [
            (record["label"], record["metrics"][metric_name])
            for record in records
            if metric_name in record["metrics"]
        ]
        if not available:
            ax.text(0.5, 0.5, "Metric absent from supplied NPZ", ha="center", va="center")
            ax.axis("off")
            continue
        labels, values = zip(*available)
        ax.boxplot(values, tick_labels=labels, showmeans=True)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Raw-latent trajectory diagnostics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    paths = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"denoising_trajectory_metrics.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def comparability_audit(records: list[dict]) -> dict:
    fields = {
        "trajectory_seed": [],
        "trajectory_num_samples": [],
        "trajectory_timesteps": [],
        "trajectory_save_steps": [],
        "checkpoint_step": [],
    }
    for record in records:
        metadata = record["metadata"]
        fields["trajectory_seed"].append(metadata.get("trajectory_seed"))
        fields["trajectory_num_samples"].append(int(record["states"].shape[0]))
        fields["trajectory_timesteps"].append(metadata.get("trajectory_timesteps"))
        fields["trajectory_save_steps"].append(tuple(int(value) for value in record["save_steps"]))
        fields["checkpoint_step"].append(metadata.get("checkpoint_step"))
    mismatches = {
        name: values
        for name, values in fields.items()
        if len({json.dumps(value) for value in values}) > 1
    }
    return {
        "status": "PASS" if not mismatches else "WARN",
        "mismatches": mismatches,
        "note": (
            "PASS means the evaluation seed, sample count, Euler budget, saved times, and "
            "checkpoint step match. It does not imply identical source samples when models "
            "use different fitted GMM/router artifacts."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        type=parse_trajectory_spec,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-paths", type=int, default=16)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [load_trajectory(label, path) for label, path in args.trajectory]
    projected, pca_info = project_trajectories(records)
    outputs = []
    outputs.extend(plot_individual_paths(records, projected, args.output_dir, max_paths=args.max_paths))
    outputs.extend(plot_mean_paths(records, projected, args.output_dir))
    outputs.extend(plot_metric_boxes(records, args.output_dir))
    audit = comparability_audit(records)
    summary = {
        "inputs": [
            {
                "label": record["label"],
                "path": record["path"],
                "num_samples": int(record["states"].shape[0]),
                "num_saved_states": int(record["states"].shape[1]),
                "metadata": record["metadata"],
            }
            for record in records
        ],
        "pca": pca_info,
        "comparability_audit": audit,
        "outputs": outputs,
    }
    summary_path = args.output_dir / "denoising_trajectory_visualization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
