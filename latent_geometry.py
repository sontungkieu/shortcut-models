from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from scipy import linalg, stats

from gmm_utils import diag_gmm_log_prob_np, fit_diag_gmm
from latent_population import load_gmm_npz, safe_label, write_json, write_rows_csv


matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class GeometryConfig:
    seed: int = 20260726
    train_fraction: float = 0.8
    split_half_repeats: int = 5
    whitening_projection_count: int = 100
    whitening_eigenvalue_floor_relative: float = 1e-6
    ppca_rank: int = 256
    local_pool_size: int = 4096
    local_query_count: int = 256
    local_neighbor_counts: tuple[int, ...] = (20, 50, 100)
    local_variance_fraction: float = 0.9
    heldout_gmm_modes: int = 16
    heldout_gmm_em_iters: int = 25
    heldout_gmm_chunk_size: int = 128
    c2st_sample_count: int = 28000
    c2st_batch_size: int = 512
    c2st_logistic_steps: int = 600
    c2st_mlp_steps: int = 1000
    c2st_mlp_hidden_size: int = 128
    c2st_learning_rate: float = 3e-4
    c2st_weight_decay: float = 1e-4
    knn_subset_size: int = 4096
    knn_k: int = 5


def _event(
    callback: Callable[..., None] | None,
    phase: str,
    **payload: object,
) -> None:
    if callback is not None:
        callback(phase=phase, **payload)


def _validate_config(config: GeometryConfig, num_samples: int) -> None:
    if not 0.5 <= config.train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    if config.split_half_repeats <= 0:
        raise ValueError("split_half_repeats must be positive")
    if config.whitening_projection_count <= 0:
        raise ValueError("whitening_projection_count must be positive")
    if config.ppca_rank <= 0:
        raise ValueError("ppca_rank must be positive")
    if not config.local_neighbor_counts:
        raise ValueError("local_neighbor_counts cannot be empty")
    if min(config.local_neighbor_counts) < 2:
        raise ValueError("local_neighbor_counts must be at least 2")
    if max(config.local_neighbor_counts) >= min(config.local_pool_size, num_samples):
        raise ValueError("local neighbor count must be smaller than local pool")
    if config.knn_k >= min(config.knn_subset_size, num_samples):
        raise ValueError("knn_k must be smaller than the kNN subset")


def deterministic_train_test_split(
    num_samples: int,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_samples)
    train_count = int(round(num_samples * train_fraction))
    train_count = min(max(train_count, 1), num_samples - 1)
    return order[:train_count], order[train_count:]


def covariance_matrix(
    samples: np.ndarray,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = (
        np.asarray(samples, dtype=np.float32)
        if indices is None
        else np.asarray(samples[indices], dtype=np.float32)
    )
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("samples must have shape [N, D] with N >= 2")
    if jax.default_backend() in ("tpu", "gpu"):
        device_values = jnp.asarray(values)
        mean = jnp.mean(device_values, axis=0)
        centered = device_values - mean
        covariance = centered.T @ centered / device_values.shape[0]
        mean_host, covariance_host = jax.device_get((mean, covariance))
        return (
            np.asarray(mean_host, dtype=np.float64),
            np.asarray(covariance_host, dtype=np.float64),
        )
    mean = np.mean(values, axis=0, dtype=np.float64)
    centered = np.asarray(values - mean, dtype=np.float64)
    covariance = centered.T @ centered / values.shape[0]
    return mean, covariance


def split_half_covariance_baseline(
    samples: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    num_samples = samples.shape[0]
    half = num_samples // 2
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for repeat in range(repeats):
        order = rng.permutation(num_samples)
        index_a = order[:half]
        index_b = order[half : 2 * half]
        _, covariance_a = covariance_matrix(samples, index_a)
        _, covariance_b = covariance_matrix(samples, index_b)
        difference = float(np.linalg.norm(covariance_a - covariance_b, ord="fro"))
        norm_a = float(np.linalg.norm(covariance_a, ord="fro"))
        norm_b = float(np.linalg.norm(covariance_b, ord="fro"))
        rows.append(
            {
                "repeat": repeat,
                "half_samples": half,
                "relative_error_a_denominator": difference / max(norm_a, 1e-30),
                "relative_error_symmetric": (
                    difference / max(0.5 * (norm_a + norm_b), 1e-30)
                ),
                "covariance_a_frobenius_norm": norm_a,
                "covariance_b_frobenius_norm": norm_b,
            }
        )
    required = np.asarray(
        [row["relative_error_a_denominator"] for row in rows],
        dtype=np.float64,
    )
    symmetric = np.asarray(
        [row["relative_error_symmetric"] for row in rows],
        dtype=np.float64,
    )
    summary: dict[str, float | int] = {
        "repeats": repeats,
        "half_samples": half,
        "relative_error_mean": float(np.mean(required)),
        "relative_error_std": float(np.std(required, ddof=1)) if repeats > 1 else 0.0,
        "relative_error_min": float(np.min(required)),
        "relative_error_max": float(np.max(required)),
        "relative_error_symmetric_mean": float(np.mean(symmetric)),
        "relative_error_symmetric_std": (
            float(np.std(symmetric, ddof=1)) if repeats > 1 else 0.0
        ),
        "definition": "||Sigma_A-Sigma_B||_F/||Sigma_A||_F",
        "covariance_divisor": "population_N",
    }
    return summary, rows


def fit_whitening(
    samples: np.ndarray,
    train_indices: np.ndarray,
    eigenvalue_floor_relative: float,
) -> dict[str, np.ndarray | float | int]:
    mean, covariance = covariance_matrix(samples, train_indices)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = linalg.eigh(
        covariance,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    max_eigenvalue = float(max(np.max(eigenvalues), 1e-30))
    floor = max(max_eigenvalue * eigenvalue_floor_relative, 1e-12)
    regularized = np.maximum(eigenvalues, floor)
    return {
        "mean": np.asarray(mean, dtype=np.float32),
        "eigenvalues": np.asarray(eigenvalues, dtype=np.float64),
        "regularized_eigenvalues": np.asarray(regularized, dtype=np.float64),
        "eigenvectors": np.asarray(eigenvectors, dtype=np.float32),
        "eigenvalue_floor": floor,
        "floor_hit_count": int(np.sum(eigenvalues < floor)),
    }


def _qq_metrics(
    values: np.ndarray,
    distribution,
    *,
    scale: float,
    theoretical_scale: float = 1.0,
    quantile_count: int = 999,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    probabilities = np.linspace(0.001, 0.999, quantile_count)
    empirical = np.quantile(np.asarray(values, dtype=np.float64), probabilities)
    theoretical = theoretical_scale * np.asarray(
        distribution.ppf(probabilities), dtype=np.float64
    )
    difference = empirical - theoretical
    metrics = {
        "qq_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "qq_rmse_scaled": float(
            np.sqrt(np.mean(np.square(difference))) / max(scale, 1e-30)
        ),
        "qq_mean_abs_error": float(np.mean(np.abs(difference))),
        "qq_mean_abs_relative_error": float(
            np.mean(np.abs(difference) / np.maximum(np.abs(theoretical), 1e-6))
        ),
    }
    return metrics, theoretical, empirical


def whitening_diagnostics(
    samples: np.ndarray,
    test_indices: np.ndarray,
    whitening: Mapping[str, np.ndarray | float | int],
    projection_count: int,
    seed: int,
    output_dir: Path,
    chunk_size: int = 512,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int]]]:
    mean = np.asarray(whitening["mean"], dtype=np.float32)
    eigenvectors = np.asarray(whitening["eigenvectors"], dtype=np.float32)
    eigenvalues = np.asarray(
        whitening["regularized_eigenvalues"], dtype=np.float64
    )
    dim = mean.size
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(dim, projection_count)).astype(np.float32)
    directions /= np.maximum(
        np.linalg.norm(directions, axis=0, keepdims=True), 1e-12
    )
    mahalanobis_sq = np.empty((test_indices.size,), dtype=np.float64)
    projections = np.empty(
        (test_indices.size, projection_count), dtype=np.float32
    )
    inv_sqrt = np.asarray(1.0 / np.sqrt(eigenvalues), dtype=np.float32)
    for start in range(0, test_indices.size, chunk_size):
        stop = min(start + chunk_size, test_indices.size)
        centered = (
            np.asarray(samples[test_indices[start:stop]], dtype=np.float32)
            - mean[None]
        )
        coefficients = centered @ eigenvectors
        whitened = coefficients * inv_sqrt[None]
        mahalanobis_sq[start:stop] = np.sum(
            np.square(whitened, dtype=np.float64), axis=1
        )
        projections[start:stop] = whitened @ directions

    chi_distribution = stats.chi2(df=dim)
    chi_metrics, chi_theoretical, chi_empirical = _qq_metrics(
        mahalanobis_sq,
        chi_distribution,
        scale=math.sqrt(2.0 * dim),
    )
    chi_ks = stats.kstest(mahalanobis_sq, chi_distribution.cdf)
    train_samples = int(samples.shape[0] - test_indices.size)
    if train_samples <= dim + 2:
        raise ValueError(
            "finite-sample Mahalanobis calibration requires "
            f"train_samples > dimension + 2, got {train_samples} <= {dim + 2}"
        )
    finite_sample_df = train_samples - dim
    finite_sample_scale = (train_samples + 1) * dim / finite_sample_df
    finite_sample_distribution = stats.f(dfn=dim, dfd=finite_sample_df)
    finite_sample_metrics, finite_theoretical, finite_empirical = _qq_metrics(
        mahalanobis_sq,
        finite_sample_distribution,
        scale=math.sqrt(2.0 * dim),
        theoretical_scale=finite_sample_scale,
    )
    finite_sample_ks = stats.kstest(
        mahalanobis_sq,
        lambda values: finite_sample_distribution.cdf(
            np.asarray(values) / finite_sample_scale
        ),
    )
    finite_sample_reference_mean = (
        dim * (train_samples + 1) / (train_samples - dim - 2)
    )

    normal_distribution = stats.norm()
    projection_rows: list[dict[str, float | int]] = []
    projection_empirical = []
    normal_theoretical = None
    for index in range(projection_count):
        values = np.asarray(projections[:, index], dtype=np.float64)
        metrics, theoretical, empirical = _qq_metrics(
            values,
            normal_distribution,
            scale=1.0,
        )
        ks = stats.kstest(values, normal_distribution.cdf)
        projection_rows.append(
            {
                "projection": index,
                **metrics,
                "ks_statistic": float(ks.statistic),
                "mean": float(np.mean(values)),
                "variance": float(np.var(values)),
                "skewness": float(stats.skew(values)),
                "excess_kurtosis": float(stats.kurtosis(values)),
            }
        )
        projection_empirical.append(empirical)
        normal_theoretical = theoretical

    projection_qq_rmse = np.asarray(
        [row["qq_rmse"] for row in projection_rows], dtype=np.float64
    )
    projection_ks = np.asarray(
        [row["ks_statistic"] for row in projection_rows], dtype=np.float64
    )
    summary: dict[str, float | int | str] = {
        "fit_samples": int(samples.shape[0] - test_indices.size),
        "test_samples": int(test_indices.size),
        "dimension": dim,
        "eigenvalue_floor": float(whitening["eigenvalue_floor"]),
        "eigenvalue_floor_relative": float(
            float(whitening["eigenvalue_floor"])
            / max(float(np.max(whitening["eigenvalues"])), 1e-30)
        ),
        "eigenvalue_floor_hit_count": int(whitening["floor_hit_count"]),
        "mahalanobis_radius_sq_mean": float(np.mean(mahalanobis_sq)),
        "mahalanobis_radius_sq_variance": float(np.var(mahalanobis_sq)),
        "mahalanobis_chi2_ks_statistic": float(chi_ks.statistic),
        "mahalanobis_chi2_ks_pvalue": float(chi_ks.pvalue),
        **{f"mahalanobis_{name}": value for name, value in chi_metrics.items()},
        "mahalanobis_reference_distribution": (
            "(n+1)*d/(n-d)*F_{d,n-d}; covariance uses divisor n"
        ),
        "mahalanobis_finite_sample_reference_mean": float(
            finite_sample_reference_mean
        ),
        "mahalanobis_radius_sq_mean_to_reference_ratio": float(
            np.mean(mahalanobis_sq) / finite_sample_reference_mean
        ),
        "mahalanobis_finite_sample_ks_statistic": float(
            finite_sample_ks.statistic
        ),
        "mahalanobis_finite_sample_ks_pvalue": float(finite_sample_ks.pvalue),
        **{
            f"mahalanobis_finite_sample_{name}": value
            for name, value in finite_sample_metrics.items()
        },
        "random_projection_count": projection_count,
        "projection_qq_rmse_median": float(np.median(projection_qq_rmse)),
        "projection_qq_rmse_q95": float(np.quantile(projection_qq_rmse, 0.95)),
        "projection_ks_statistic_median": float(np.median(projection_ks)),
        "projection_ks_statistic_q95": float(np.quantile(projection_ks, 0.95)),
        "fit_scope": "train split only; diagnostics on held-out split",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.plot(chi_theoretical, chi_empirical, linewidth=1.5, color="#2f6690")
    lower = min(float(chi_theoretical[0]), float(chi_empirical[0]))
    upper = max(float(chi_theoretical[-1]), float(chi_empirical[-1]))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="0.4")
    axis.set(
        xlabel=r"Theoretical $\chi^2_d$ quantile",
        ylabel="Held-out squared Mahalanobis quantile",
        title="Whitened Radius QQ Plot",
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "whitened_mahalanobis_qq.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.plot(
        finite_theoretical,
        finite_empirical,
        linewidth=1.5,
        color="#2f6690",
    )
    lower = min(float(finite_theoretical[0]), float(finite_empirical[0]))
    upper = max(float(finite_theoretical[-1]), float(finite_empirical[-1]))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="0.4")
    axis.set(
        xlabel="Finite-sample Gaussian reference quantile",
        ylabel="Held-out squared Mahalanobis quantile",
        title="Finite-Sample-Calibrated Whitened Radius QQ Plot",
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        output_dir / "whitened_mahalanobis_finite_sample_qq.png",
        dpi=180,
    )
    plt.close(fig)

    projection_matrix = np.asarray(projection_empirical, dtype=np.float64)
    median = np.median(projection_matrix, axis=0)
    q05 = np.quantile(projection_matrix, 0.05, axis=0)
    q95 = np.quantile(projection_matrix, 0.95, axis=0)
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    axis.fill_between(
        normal_theoretical,
        q05,
        q95,
        color="#8ecae6",
        alpha=0.45,
        label="5%-95% across projections",
    )
    axis.plot(
        normal_theoretical,
        median,
        color="#006d77",
        linewidth=1.6,
        label="Median projection",
    )
    axis.plot([-3.2, 3.2], [-3.2, 3.2], linestyle="--", color="0.4")
    axis.set(
        xlabel="Theoretical standard-normal quantile",
        ylabel="Whitened projection quantile",
        title=f"Random Projection QQ ({projection_count} Directions)",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "whitened_random_projection_qq.png", dpi=180)
    plt.close(fig)
    return summary, projection_rows


def _pairwise_squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if jax.default_backend() in ("tpu", "gpu"):
        a_device = jnp.asarray(a, dtype=jnp.float32)
        b_device = jnp.asarray(b, dtype=jnp.float32)
        distances = (
            jnp.sum(a_device * a_device, axis=1, keepdims=True)
            + jnp.sum(b_device * b_device, axis=1)[None]
            - 2.0 * (a_device @ b_device.T)
        )
        return np.maximum(np.asarray(jax.device_get(distances)), 0.0)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return np.maximum(
        np.sum(a * a, axis=1, keepdims=True)
        + np.sum(b * b, axis=1)[None]
        - 2.0 * (a @ b.T),
        0.0,
    )


def local_pca_dimensions(
    samples: np.ndarray,
    *,
    pool_size: int,
    query_count: int,
    neighbor_counts: Sequence[int],
    variance_fraction: float,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    pool_size = min(pool_size, samples.shape[0])
    query_count = min(query_count, pool_size)
    pool_indices = rng.choice(samples.shape[0], size=pool_size, replace=False)
    query_positions = rng.choice(pool_size, size=query_count, replace=False)
    pool = np.asarray(samples[pool_indices], dtype=np.float32)
    queries = pool[query_positions]
    distances = _pairwise_squared_distances(queries, pool)
    distances[np.arange(query_count), query_positions] = np.inf
    max_neighbors = max(neighbor_counts)
    nearest = np.argpartition(
        distances,
        kth=max_neighbors - 1,
        axis=1,
    )[:, :max_neighbors]
    nearest_distance = np.take_along_axis(distances, nearest, axis=1)
    nearest = np.take_along_axis(
        nearest,
        np.argsort(nearest_distance, axis=1),
        axis=1,
    )

    rows: list[dict[str, float | int]] = []
    summary: dict[str, float | int] = {
        "pool_size": pool_size,
        "query_count": query_count,
        "variance_fraction": variance_fraction,
        "distance_space": "raw scaled VAE posterior-mean latent",
    }
    figure_values = []
    for neighbors in neighbor_counts:
        local = pool[nearest[:, :neighbors]]
        if jax.default_backend() in ("tpu", "gpu"):
            local_device = jnp.asarray(local)
            centered = local_device - jnp.mean(local_device, axis=1, keepdims=True)
            gram = centered @ jnp.swapaxes(centered, 1, 2) / neighbors
            eigenvalues = np.asarray(jax.device_get(jnp.linalg.eigvalsh(gram)))
        else:
            centered = local - np.mean(local, axis=1, keepdims=True)
            gram = centered @ np.swapaxes(centered, 1, 2) / neighbors
            eigenvalues = np.linalg.eigvalsh(gram)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        eigenvalues = np.sort(eigenvalues, axis=1)[:, ::-1]
        cumulative = np.cumsum(eigenvalues, axis=1)
        cumulative /= np.maximum(cumulative[:, -1:], 1e-30)
        dimensions = (
            np.sum(cumulative < variance_fraction, axis=1).astype(np.int32) + 1
        )
        for query, dimension in enumerate(dimensions):
            rows.append(
                {
                    "query": query,
                    "neighbors": neighbors,
                    "local_dimension": int(dimension),
                }
            )
        prefix = f"k{neighbors}"
        summary.update(
            {
                f"{prefix}_median": float(np.median(dimensions)),
                f"{prefix}_q25": float(np.quantile(dimensions, 0.25)),
                f"{prefix}_q75": float(np.quantile(dimensions, 0.75)),
                f"{prefix}_mean": float(np.mean(dimensions)),
                f"{prefix}_std": float(np.std(dimensions)),
            }
        )
        figure_values.append((neighbors, dimensions))

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    bins = np.arange(
        0.5,
        max(float(np.max(values)) for _, values in figure_values) + 1.5,
        1.0,
    )
    for neighbors, dimensions in figure_values:
        axis.hist(
            dimensions,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=f"k={neighbors}",
        )
    axis.set(
        xlabel=f"Local PCA components for {variance_fraction:.0%} variance",
        ylabel="Density",
        title="Local Intrinsic Dimension",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "local_pca_dimension_histogram.png", dpi=180)
    plt.close(fig)
    return summary, rows


def diagonal_gaussian_nll(
    samples: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> float:
    values = np.asarray(samples, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 1e-12)
    nll = 0.5 * np.sum(
        LOG_2PI + np.log(variance)[None] + np.square(values - mean) / variance[None],
        axis=1,
    )
    return float(np.mean(nll))


def ppca_nll(
    samples: np.ndarray,
    mean: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    rank: int,
    chunk_size: int = 512,
) -> float:
    dim = mean.size
    rank = min(max(rank, 1), dim - 1)
    top_vectors = np.asarray(eigenvectors[:, -rank:], dtype=np.float32)
    top_values = np.maximum(
        np.asarray(eigenvalues[-rank:], dtype=np.float64), 1e-12
    )
    residual_value = max(
        float(np.mean(np.maximum(eigenvalues[:-rank], 0.0))),
        1e-12,
    )
    log_determinant = float(
        np.sum(np.log(top_values)) + (dim - rank) * math.log(residual_value)
    )
    total = 0.0
    count = 0
    for start in range(0, samples.shape[0], chunk_size):
        stop = min(start + chunk_size, samples.shape[0])
        centered = (
            np.asarray(samples[start:stop], dtype=np.float32)
            - np.asarray(mean, dtype=np.float32)[None]
        )
        projected = centered @ top_vectors
        projected_sq = np.sum(np.square(projected, dtype=np.float64), axis=1)
        total_sq = np.sum(np.square(centered, dtype=np.float64), axis=1)
        quadratic = np.sum(
            np.square(projected, dtype=np.float64) / top_values[None],
            axis=1,
        )
        quadratic += np.maximum(total_sq - projected_sq, 0.0) / residual_value
        total += float(
            np.sum(0.5 * (dim * LOG_2PI + log_determinant + quadratic))
        )
        count += stop - start
    return total / max(count, 1)


def _gmm_nll(
    samples: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    variance: np.ndarray,
    chunk_size: int,
) -> float:
    total = 0.0
    for start in range(0, samples.shape[0], chunk_size):
        stop = min(start + chunk_size, samples.shape[0])
        logits = diag_gmm_log_prob_np(
            samples[start:stop],
            pi,
            mu,
            variance,
        )
        maximum = np.max(logits, axis=1, keepdims=True)
        log_prob = maximum[:, 0] + np.log(
            np.sum(np.exp(logits - maximum), axis=1)
        )
        total += -float(np.sum(log_prob))
    return total / max(samples.shape[0], 1)


def heldout_density_diagnostics(
    samples: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    whitening: Mapping[str, np.ndarray | float | int],
    config: GeometryConfig,
    output_dir: Path,
) -> tuple[list[dict[str, float | int | str]], dict[str, np.ndarray]]:
    train = np.asarray(samples[train_indices], dtype=np.float32)
    test = np.asarray(samples[test_indices], dtype=np.float32)
    mean = np.mean(train, axis=0, dtype=np.float64)
    variance = np.maximum(np.var(train, axis=0, dtype=np.float64), 1e-12)
    dim = samples.shape[1]

    rows: list[dict[str, float | int | str]] = []

    def add_row(model: str, train_nll: float, test_nll: float, **extra: object) -> None:
        rows.append(
            {
                "model": model,
                "train_nll": train_nll,
                "test_nll": test_nll,
                "train_nll_per_dimension": train_nll / dim,
                "test_nll_per_dimension": test_nll / dim,
                "test_minus_train_nll": test_nll - train_nll,
                **extra,
            }
        )

    add_row(
        "diagonal_gaussian",
        diagonal_gaussian_nll(train, mean, variance),
        diagonal_gaussian_nll(test, mean, variance),
        covariance_type="diagonal",
    )

    eigenvectors = np.asarray(whitening["eigenvectors"], dtype=np.float32)
    eigenvalues = np.asarray(whitening["eigenvalues"], dtype=np.float64)
    ppca_rank = min(config.ppca_rank, dim - 1)
    add_row(
        f"ppca_rank_{ppca_rank}",
        ppca_nll(train, mean, eigenvectors, eigenvalues, ppca_rank),
        ppca_nll(test, mean, eigenvectors, eigenvalues, ppca_rank),
        covariance_type="low_rank_plus_isotropic_residual",
        rank=ppca_rank,
        residual_variance=float(
            np.mean(np.maximum(eigenvalues[:-ppca_rank], 0.0))
        ),
    )

    em_trace: list[dict[str, float]] = []
    fit = fit_diag_gmm(
        train,
        num_modes=config.heldout_gmm_modes,
        em_iters=config.heldout_gmm_em_iters,
        em_restarts=1,
        seed=config.seed,
        pi_prior_type="dirichlet",
        pi_prior_strength=512.0,
        var_prior_type="kl",
        var_prior_strength=128.0,
        var_prior_target_var=0.75,
        min_std=0.0,
        min_std_data_frac=0.0,
        standardized=False,
        chunk_size=config.heldout_gmm_chunk_size,
        use_kmeanspp=True,
        em_metrics_callback=lambda row: em_trace.append(dict(row)),
    )
    train_nll = _gmm_nll(
        train,
        fit["pi"],
        fit["mu"],
        fit["var"],
        config.heldout_gmm_chunk_size,
    )
    test_nll = _gmm_nll(
        test,
        fit["pi"],
        fit["mu"],
        fit["var"],
        config.heldout_gmm_chunk_size,
    )
    add_row(
        f"diagonal_gmm_{config.heldout_gmm_modes}",
        train_nll,
        test_nll,
        covariance_type="component_diagonal",
        components=config.heldout_gmm_modes,
        em_iters=config.heldout_gmm_em_iters,
        pi_prior_type="dirichlet",
        pi_prior_strength=512.0,
        var_prior_type="kl",
        var_prior_strength=128.0,
        var_prior_target_var=0.75,
    )
    write_rows_csv(output_dir / "heldout_gmm_em_trace.csv", em_trace)
    refit_state = {
        "pi": np.asarray(fit["pi"], dtype=np.float32),
        "mu": np.asarray(fit["mu"], dtype=np.float32),
        "var": np.asarray(fit["var"], dtype=np.float32),
        "mean": np.zeros((dim,), dtype=np.float32),
        "std": np.ones((dim,), dtype=np.float32),
        "latent_shape": np.asarray([32, 32, 4], dtype=np.int32),
        "transform_type": np.asarray("raw"),
    }
    np.savez_compressed(
        output_dir / "heldout_refit_gmm16.npz",
        **refit_state,
        train_indices=train_indices.astype(np.int32),
        test_indices=test_indices.astype(np.int32),
    )
    return rows, refit_state


def sample_gmm(
    state: Mapping[str, object],
    sample_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pi = np.asarray(state["pi"], dtype=np.float64).reshape(-1)
    pi /= np.sum(pi)
    mu = np.asarray(state["mu"], dtype=np.float32)
    variance = np.maximum(np.asarray(state["var"], dtype=np.float32), 1e-12)
    component = rng.choice(pi.size, size=sample_count, p=pi)
    fit_samples = (
        mu[component]
        + np.sqrt(variance[component])
        * rng.normal(size=(sample_count, mu.shape[1])).astype(np.float32)
    )
    transform = state.get("transform_type")
    if transform is None:
        transform = (
            "standardize"
            if int(np.asarray(state.get("gmm_standardize_data", 1)).item())
            else "raw"
        )
    if isinstance(transform, np.ndarray):
        transform = transform.item()
    if isinstance(transform, bytes):
        transform = transform.decode("utf-8")
    transform = str(transform)
    if transform == "raw":
        return fit_samples.astype(np.float32)
    if transform == "standardize":
        offset = np.asarray(state["mean"], dtype=np.float32).reshape(-1)
        scale = np.asarray(state["std"], dtype=np.float32).reshape(-1)
        return (fit_samples * scale[None] + offset[None]).astype(np.float32)
    if transform == "channel_whiten":
        latent_shape = tuple(
            int(value) for value in np.asarray(state["latent_shape"]).reshape(-1)
        )
        channels = latent_shape[-1]
        blocks = fit_samples.reshape(sample_count, -1, channels)
        channel_mean = np.asarray(state["channel_mean"], dtype=np.float32)
        unwhiten = np.asarray(state["channel_unwhiten"], dtype=np.float32)
        return (
            np.einsum("nsd,cd->nsc", blocks, unwhiten) + channel_mean
        ).reshape(sample_count, -1)
    raise ValueError(f"Unsupported GMM transform {transform!r}")


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    count_positive = int(np.sum(positive))
    count_negative = int(np.sum(negative))
    if count_positive == 0 or count_negative == 0:
        raise ValueError("AUC requires both classes")
    ranks = stats.rankdata(scores)
    return float(
        (
            np.sum(ranks[positive])
            - count_positive * (count_positive + 1) / 2.0
        )
        / (count_positive * count_negative)
    )


def _binary_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    loss = np.mean(
        np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    )
    return {
        "auc": binary_auc(labels, logits),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)),
        "loss": float(loss),
    }


def _predict_in_chunks(
    predict: Callable[[object, jnp.ndarray], jnp.ndarray],
    params: object,
    features: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    outputs = []
    for start in range(0, features.shape[0], chunk_size):
        stop = min(start + chunk_size, features.shape[0])
        outputs.append(
            np.asarray(
                jax.device_get(predict(params, jnp.asarray(features[start:stop])))
            )
        )
    return np.concatenate(outputs)


def _fit_binary_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    model: str,
    steps: int,
    batch_size: int,
    hidden_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, float | int | str]:
    dim = train_features.shape[1]
    key = jax.random.PRNGKey(seed)
    if model == "logistic":
        params = {
            "weight": jnp.zeros((dim,), dtype=jnp.float32),
            "bias": jnp.asarray(0.0, dtype=jnp.float32),
        }

        @jax.jit
        def predict(parameters, features):
            return features @ parameters["weight"] + parameters["bias"]

    elif model == "mlp":
        key_1, key_2 = jax.random.split(key)
        params = {
            "weight_1": jax.random.normal(key_1, (dim, hidden_size))
            * math.sqrt(2.0 / dim),
            "bias_1": jnp.zeros((hidden_size,), dtype=jnp.float32),
            "weight_2": jax.random.normal(key_2, (hidden_size,))
            * math.sqrt(2.0 / hidden_size),
            "bias_2": jnp.asarray(0.0, dtype=jnp.float32),
        }

        @jax.jit
        def predict(parameters, features):
            hidden = jax.nn.relu(
                features @ parameters["weight_1"] + parameters["bias_1"]
            )
            return hidden @ parameters["weight_2"] + parameters["bias_2"]

    else:
        raise ValueError(f"Unsupported classifier model={model!r}")

    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def update(parameters, state, features, labels):
        def objective(candidate):
            logits = predict(candidate, features)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

        loss, gradients = jax.value_and_grad(objective)(parameters)
        updates, state = optimizer.update(gradients, state, parameters)
        parameters = optax.apply_updates(parameters, updates)
        return parameters, state, loss

    rng = np.random.default_rng(seed)
    batch_size = min(batch_size, train_features.shape[0])
    last_loss = 0.0
    for _ in range(steps):
        indices = rng.integers(0, train_features.shape[0], size=batch_size)
        params, optimizer_state, loss = update(
            params,
            optimizer_state,
            jnp.asarray(train_features[indices]),
            jnp.asarray(train_labels[indices], dtype=jnp.float32),
        )
        last_loss = float(loss)
    train_logits = _predict_in_chunks(predict, params, train_features)
    test_logits = _predict_in_chunks(predict, params, test_features)
    train_metrics = _binary_metrics(train_labels, train_logits)
    test_metrics = _binary_metrics(test_labels, test_logits)
    return {
        "classifier": model,
        "steps": steps,
        "batch_size": batch_size,
        "last_batch_loss": last_loss,
        **{f"train_{name}": value for name, value in train_metrics.items()},
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }


def classifier_two_sample_test(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    config: GeometryConfig,
    seed: int,
) -> list[dict[str, float | int | str]]:
    sample_count = min(
        config.c2st_sample_count,
        real_samples.shape[0],
        generated_samples.shape[0],
    )
    rng = np.random.default_rng(seed)
    real_indices = rng.choice(real_samples.shape[0], sample_count, replace=False)
    generated_indices = rng.choice(
        generated_samples.shape[0], sample_count, replace=False
    )
    real = np.asarray(real_samples[real_indices], dtype=np.float32)
    generated = np.asarray(generated_samples[generated_indices], dtype=np.float32)

    real_order = rng.permutation(sample_count)
    generated_order = rng.permutation(sample_count)
    train_count = int(round(sample_count * config.train_fraction))
    train_features = np.concatenate(
        [real[real_order[:train_count]], generated[generated_order[:train_count]]],
        axis=0,
    )
    train_labels = np.concatenate(
        [
            np.zeros((train_count,), dtype=np.float32),
            np.ones((train_count,), dtype=np.float32),
        ]
    )
    test_features = np.concatenate(
        [real[real_order[train_count:]], generated[generated_order[train_count:]]],
        axis=0,
    )
    test_labels = np.concatenate(
        [
            np.zeros((sample_count - train_count,), dtype=np.float32),
            np.ones((sample_count - train_count,), dtype=np.float32),
        ]
    )
    shuffle = rng.permutation(train_features.shape[0])
    train_features = train_features[shuffle]
    train_labels = train_labels[shuffle]
    shuffle = rng.permutation(test_features.shape[0])
    test_features = test_features[shuffle]
    test_labels = test_labels[shuffle]

    feature_mean = np.mean(train_features, axis=0, dtype=np.float64)
    feature_std = np.std(train_features, axis=0, dtype=np.float64)
    feature_std = np.maximum(feature_std, 1e-6)
    train_features -= feature_mean.astype(np.float32)
    train_features /= feature_std.astype(np.float32)
    test_features -= feature_mean.astype(np.float32)
    test_features /= feature_std.astype(np.float32)

    rows = []
    for model, steps in (
        ("logistic", config.c2st_logistic_steps),
        ("mlp", config.c2st_mlp_steps),
    ):
        rows.append(
            {
                "sample_count_per_class": sample_count,
                "feature_standardization": "combined training split only",
                **_fit_binary_classifier(
                    train_features,
                    train_labels,
                    test_features,
                    test_labels,
                    model=model,
                    steps=steps,
                    batch_size=config.c2st_batch_size,
                    hidden_size=config.c2st_mlp_hidden_size,
                    learning_rate=config.c2st_learning_rate,
                    weight_decay=config.c2st_weight_decay,
                    seed=seed + (0 if model == "logistic" else 1),
                ),
            }
        )
    return rows


def knn_precision_recall(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    *,
    subset_size: int,
    k: int,
    seed: int,
) -> dict[str, float | int | str]:
    subset_size = min(
        subset_size,
        real_samples.shape[0],
        generated_samples.shape[0],
    )
    rng = np.random.default_rng(seed)
    real = np.asarray(
        real_samples[
            rng.choice(real_samples.shape[0], subset_size, replace=False)
        ],
        dtype=np.float32,
    )
    generated = np.asarray(
        generated_samples[
            rng.choice(generated_samples.shape[0], subset_size, replace=False)
        ],
        dtype=np.float32,
    )
    real_distances = _pairwise_squared_distances(real, real)
    generated_distances = _pairwise_squared_distances(generated, generated)
    np.fill_diagonal(real_distances, np.inf)
    np.fill_diagonal(generated_distances, np.inf)
    real_radius = np.partition(real_distances, kth=k - 1, axis=1)[:, k - 1]
    generated_radius = np.partition(
        generated_distances, kth=k - 1, axis=1
    )[:, k - 1]
    cross = _pairwise_squared_distances(real, generated)
    precision = float(np.mean(np.any(cross <= real_radius[:, None], axis=0)))
    recall = float(
        np.mean(np.any(cross <= generated_radius[None], axis=1))
    )
    return {
        "subset_size_per_distribution": subset_size,
        "k": k,
        "precision": precision,
        "recall": recall,
        "distance_space": "raw scaled VAE posterior-mean latent",
        "estimator": "kth-neighbor manifold precision/recall",
    }


def _gmm_fingerprint(state: Mapping[str, object]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in ("pi", "mu", "var"):
        array = np.ascontiguousarray(np.asarray(state[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def run_geometry_diagnostics(
    samples: np.ndarray,
    *,
    gmm_paths: Sequence[str | Path],
    gmm_labels: Sequence[str],
    output_dir: str | Path,
    config: GeometryConfig,
    event_callback: Callable[..., None] | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("samples must have shape [N, D]")
    if len(gmm_paths) != len(gmm_labels):
        raise ValueError("gmm_paths and gmm_labels must have equal lengths")
    _validate_config(config, samples.shape[0])
    write_json(output_dir / "geometry_config.json", asdict(config))
    train_indices, test_indices = deterministic_train_test_split(
        samples.shape[0],
        config.train_fraction,
        config.seed,
    )
    np.savez_compressed(
        output_dir / "heldout_split_indices.npz",
        train_indices=train_indices.astype(np.int32),
        test_indices=test_indices.astype(np.int32),
    )

    _event(event_callback, "geometry_split_half_start")
    split_summary, split_rows = split_half_covariance_baseline(
        samples,
        config.split_half_repeats,
        config.seed + 1,
    )
    write_json(output_dir / "split_half_covariance_summary.json", split_summary)
    write_rows_csv(output_dir / "split_half_covariance.csv", split_rows)
    _event(event_callback, "geometry_split_half_complete", **split_summary)

    _event(event_callback, "geometry_whitening_fit_start")
    whitening = fit_whitening(
        samples,
        train_indices,
        config.whitening_eigenvalue_floor_relative,
    )
    np.savez_compressed(
        output_dir / "whitening_spectrum.npz",
        eigenvalues=np.asarray(whitening["eigenvalues"], dtype=np.float32),
        regularized_eigenvalues=np.asarray(
            whitening["regularized_eigenvalues"], dtype=np.float32
        ),
        eigenvalue_floor=np.asarray(whitening["eigenvalue_floor"]),
    )
    whitening_summary, projection_rows = whitening_diagnostics(
        samples,
        test_indices,
        whitening,
        config.whitening_projection_count,
        config.seed + 2,
        output_dir,
    )
    write_json(output_dir / "whitening_summary.json", whitening_summary)
    write_rows_csv(
        output_dir / "whitening_random_projection_metrics.csv",
        projection_rows,
    )
    _event(event_callback, "geometry_whitening_complete", **whitening_summary)

    _event(event_callback, "geometry_local_dimension_start")
    local_summary, local_rows = local_pca_dimensions(
        samples,
        pool_size=config.local_pool_size,
        query_count=config.local_query_count,
        neighbor_counts=config.local_neighbor_counts,
        variance_fraction=config.local_variance_fraction,
        seed=config.seed + 3,
        output_dir=output_dir,
    )
    write_json(output_dir / "local_dimension_summary.json", local_summary)
    write_rows_csv(output_dir / "local_dimension.csv", local_rows)
    _event(event_callback, "geometry_local_dimension_complete", **local_summary)

    _event(event_callback, "geometry_heldout_density_start")
    density_rows, refit_state = heldout_density_diagnostics(
        samples,
        train_indices,
        test_indices,
        whitening,
        config,
        output_dir,
    )
    write_rows_csv(output_dir / "heldout_density.csv", density_rows)
    write_json(
        output_dir / "heldout_density.json",
        {"models": density_rows},
    )
    _event(event_callback, "geometry_heldout_density_complete")

    _event(event_callback, "geometry_two_sample_start")
    c2st_rows: list[dict[str, object]] = []
    knn_rows: list[dict[str, object]] = []
    states: list[tuple[str, Mapping[str, object]]] = [
        ("heldout_refit_gmm16", refit_state)
    ]
    states.extend(
        (label, load_gmm_npz(path))
        for label, path in zip(gmm_labels, gmm_paths)
    )
    cached_by_fingerprint: dict[
        str, tuple[list[dict[str, object]], dict[str, object], str]
    ] = {}
    for index, (label, state) in enumerate(states):
        fingerprint = _gmm_fingerprint(state)
        duplicate = cached_by_fingerprint.get(fingerprint)
        if duplicate is not None:
            cached_c2st, cached_knn, source_label = duplicate
            for row in cached_c2st:
                c2st_rows.append(
                    {
                        **row,
                        "gmm_label": label,
                        "duplicate_of": source_label,
                    }
                )
            knn_rows.append(
                {
                    **cached_knn,
                    "gmm_label": label,
                    "duplicate_of": source_label,
                }
            )
            continue
        generated = sample_gmm(
            state,
            config.c2st_sample_count,
            config.seed + 100 + index,
        )
        model_c2st = classifier_two_sample_test(
            samples,
            generated,
            config,
            config.seed + 200 + index,
        )
        model_knn = knn_precision_recall(
            samples,
            generated,
            subset_size=config.knn_subset_size,
            k=config.knn_k,
            seed=config.seed + 300 + index,
        )
        for row in model_c2st:
            c2st_rows.append(
                {
                    **row,
                    "gmm_label": label,
                    "duplicate_of": "",
                }
            )
        knn_rows.append(
            {
                **model_knn,
                "gmm_label": label,
                "duplicate_of": "",
            }
        )
        cached_by_fingerprint[fingerprint] = (
            [dict(row) for row in model_c2st],
            dict(model_knn),
            label,
        )
        _event(
            event_callback,
            "geometry_two_sample_model_complete",
            gmm_label=label,
        )
        del generated
    write_rows_csv(output_dir / "classifier_two_sample.csv", c2st_rows)
    write_rows_csv(output_dir / "knn_precision_recall.csv", knn_rows)

    primary_rows: list[dict[str, object]] = [
        {
            "group": "covariance_reliability",
            "metric": "split_half_covariance_relative_error_mean",
            "value": split_summary["relative_error_mean"],
        },
        {
            "group": "whitened_gaussianity",
            "metric": "mahalanobis_finite_sample_qq_rmse_scaled",
            "value": whitening_summary[
                "mahalanobis_finite_sample_qq_rmse_scaled"
            ],
        },
        {
            "group": "whitened_gaussianity",
            "metric": "projection_qq_rmse_median",
            "value": whitening_summary["projection_qq_rmse_median"],
        },
        {
            "group": "local_geometry",
            "metric": "local_dimension_k50_median",
            "value": local_summary.get("k50_median", ""),
        },
    ]
    for row in density_rows:
        primary_rows.append(
            {
                "group": "heldout_density",
                "model": row["model"],
                "metric": "test_nll_per_dimension",
                "value": row["test_nll_per_dimension"],
            }
        )
    for row in c2st_rows:
        primary_rows.append(
            {
                "group": "distribution_match",
                "model": row["gmm_label"],
                "classifier": row["classifier"],
                "metric": "c2st_test_auc",
                "value": row["test_auc"],
            }
        )
    for row in knn_rows:
        for metric in ("precision", "recall"):
            primary_rows.append(
                {
                    "group": "distribution_match",
                    "model": row["gmm_label"],
                    "metric": f"knn_{metric}",
                    "value": row[metric],
                }
            )
    write_rows_csv(output_dir / "geometry_primary_metrics.csv", primary_rows)

    summary: dict[str, object] = {
        "representation": (
            "scaled deterministic VAE posterior mean; posterior noise is excluded "
            "from pointwise geometry to avoid Monte Carlo noise"
        ),
        "num_samples": int(samples.shape[0]),
        "dimension": int(samples.shape[1]),
        "split_half_covariance": split_summary,
        "whitening": whitening_summary,
        "local_dimension": local_summary,
        "heldout_density": density_rows,
        "classifier_two_sample": c2st_rows,
        "knn_precision_recall": knn_rows,
        "elapsed_seconds": time.monotonic() - started_at,
    }
    write_json(output_dir / "geometry_diagnostics_summary.json", summary)
    _event(
        event_callback,
        "geometry_complete",
        elapsed_seconds=summary["elapsed_seconds"],
    )
    return summary
