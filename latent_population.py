from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass
class PosteriorMoments:
    count: int
    mean: np.ndarray
    between_m2: np.ndarray
    posterior_var_sum: np.ndarray

    @classmethod
    def empty(cls, dim: int) -> "PosteriorMoments":
        return cls(
            count=0,
            mean=np.zeros((dim,), dtype=np.float64),
            between_m2=np.zeros((dim, dim), dtype=np.float64),
            posterior_var_sum=np.zeros((dim,), dtype=np.float64),
        )

    def update(self, posterior_mean: np.ndarray, posterior_var: np.ndarray) -> None:
        posterior_mean = np.asarray(posterior_mean, dtype=np.float64)
        posterior_var = np.asarray(posterior_var, dtype=np.float64)
        if posterior_mean.ndim != 2 or posterior_var.shape != posterior_mean.shape:
            raise ValueError("posterior_mean and posterior_var must both have shape [batch, dim]")
        if posterior_mean.shape[0] == 0:
            return

        batch_count = int(posterior_mean.shape[0])
        batch_mean = np.mean(posterior_mean, axis=0)
        centered = posterior_mean - batch_mean
        batch_m2 = centered.T @ centered
        batch_var_sum = np.sum(posterior_var, axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.between_m2 = batch_m2
            self.posterior_var_sum = batch_var_sum
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.between_m2 += (
            batch_m2
            + np.outer(delta, delta) * (self.count * batch_count / total)
        )
        self.mean += delta * (batch_count / total)
        self.posterior_var_sum += batch_var_sum
        self.count = total

    def finalize(self) -> dict[str, np.ndarray | int]:
        if self.count <= 0:
            raise ValueError("Cannot finalize empty posterior moments")
        between_cov = self.between_m2 / self.count
        posterior_noise_var = self.posterior_var_sum / self.count
        aggregated_cov = between_cov.copy()
        aggregated_cov.flat[:: aggregated_cov.shape[0] + 1] += posterior_noise_var
        return {
            "count": self.count,
            "mean": self.mean,
            "between_covariance": between_cov,
            "posterior_noise_variance": posterior_noise_var,
            "aggregated_covariance": aggregated_cov,
        }


def finalize_posterior_moments(
    count: int,
    mean: np.ndarray,
    between_m2: np.ndarray,
    posterior_var_sum: np.ndarray,
) -> dict[str, np.ndarray | int]:
    state = PosteriorMoments(
        count=int(count),
        mean=np.asarray(mean, dtype=np.float64),
        between_m2=np.asarray(between_m2, dtype=np.float64),
        posterior_var_sum=np.asarray(posterior_var_sum, dtype=np.float64),
    )
    return state.finalize()


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, q)) if values.size else 0.0


def _effective_rank(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(np.sum(values))
    if total <= eps:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > eps]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _components_for_fraction(eigenvalues: np.ndarray, fraction: float) -> int:
    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    values = np.sort(values)[::-1]
    total = float(np.sum(values))
    if total <= 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(values) / total, fraction, side="left") + 1)


def covariance_summary(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    mean_epsilon: float,
    dead_variance_threshold: float,
    negative_eigenvalue_tolerance: float = 1e-7,
) -> tuple[dict[str, float | int], np.ndarray]:
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (mean.size, mean.size):
        raise ValueError("covariance shape must match the flattened mean dimension")

    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    covariance = (covariance + covariance.T) * 0.5
    variances = np.diag(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive_variances = np.maximum(variances, 0.0)
    active = positive_variances > max(dead_variance_threshold, 0.0)
    if np.count_nonzero(active) >= 2:
        active_cov = covariance[np.ix_(active, active)]
        active_std = np.sqrt(np.maximum(np.diag(active_cov), 1e-30))
        correlation = active_cov / np.outer(active_std, active_std)
        offdiag = np.abs(correlation[np.triu_indices(correlation.shape[0], 1)])
    else:
        offdiag = np.asarray([], dtype=np.float64)

    abs_mean = np.abs(mean)
    metrics: dict[str, float | int] = {
        "dimension": int(mean.size),
        "mean_rms": float(np.linalg.norm(mean) / math.sqrt(max(mean.size, 1))),
        "mean_abs_q95": _quantile(abs_mean, 0.95),
        "mean_abs_max": float(np.max(abs_mean)) if abs_mean.size else 0.0,
        "mean_fraction_abs_lt_epsilon": float(np.mean(abs_mean < mean_epsilon)),
        "variance_mean": float(np.mean(positive_variances)),
        "variance_median": _quantile(positive_variances, 0.5),
        "variance_q05": _quantile(positive_variances, 0.05),
        "variance_q95": _quantile(positive_variances, 0.95),
        "dead_dimension_fraction": float(
            np.mean(positive_variances < dead_variance_threshold)
        ),
        "covariance_trace": float(np.trace(covariance)),
        "mean_abs_correlation": float(np.mean(offdiag)) if offdiag.size else 0.0,
        "abs_correlation_q95": _quantile(offdiag, 0.95),
        "effective_rank": _effective_rank(eigenvalues),
        "components_90pct": _components_for_fraction(eigenvalues, 0.90),
        "components_95pct": _components_for_fraction(eigenvalues, 0.95),
        "components_99pct": _components_for_fraction(eigenvalues, 0.99),
        "covariance_min_eigenvalue": float(np.min(eigenvalues)),
        "covariance_negative_eigenvalue_count": int(
            np.sum(eigenvalues < -abs(negative_eigenvalue_tolerance))
        ),
        "covariance_symmetry_max_abs_error": symmetry_error,
        "finite_fraction": float(
            np.mean(np.isfinite(mean)) * np.mean(np.isfinite(covariance))
        ),
    }
    return metrics, eigenvalues


def posterior_population_summary(
    moments: Mapping[str, np.ndarray | int],
    radii: np.ndarray,
    *,
    mean_epsilon: float,
    dead_variance_threshold: float,
    population_mode: str = "aggregated_posterior",
) -> tuple[dict[str, float | int | str], np.ndarray]:
    mean = np.asarray(moments["mean"], dtype=np.float64)
    between_cov = np.asarray(moments["between_covariance"], dtype=np.float64)
    posterior_noise_var = np.asarray(
        moments["posterior_noise_variance"], dtype=np.float64
    )
    aggregated_cov = np.asarray(moments["aggregated_covariance"], dtype=np.float64)
    if population_mode == "aggregated_posterior":
        population_covariance = aggregated_cov
    elif population_mode == "posterior_mean":
        population_covariance = between_cov
    else:
        raise ValueError(
            "population_mode must be aggregated_posterior or posterior_mean"
        )
    metrics, eigenvalues = covariance_summary(
        mean,
        population_covariance,
        mean_epsilon=mean_epsilon,
        dead_variance_threshold=dead_variance_threshold,
    )
    radii = np.asarray(radii, dtype=np.float64)
    trace = float(np.trace(aggregated_cov))
    trace_parts = float(np.trace(between_cov) + np.sum(posterior_noise_var))
    metrics.update(
        {
            "num_samples": int(moments["count"]),
            "between_image_covariance_trace": float(np.trace(between_cov)),
            "posterior_noise_covariance_trace": float(np.sum(posterior_noise_var)),
            "selected_covariance_trace": float(np.trace(population_covariance)),
            "posterior_noise_trace_fraction": float(
                np.sum(posterior_noise_var) / max(trace, 1e-30)
            ),
            "trace_decomposition_abs_error": float(abs(trace - trace_parts)),
            "expected_radius_median": _quantile(radii, 0.5),
            "expected_radius_q05": _quantile(radii, 0.05),
            "expected_radius_q95": _quantile(radii, 0.95),
            "expected_radius_max": float(np.max(radii)) if radii.size else 0.0,
            "population_mode": population_mode,
        }
    )
    return metrics, eigenvalues


def load_gmm_npz(path: str | Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=True) as data:
        state: dict[str, np.ndarray | str] = {
            name: np.asarray(data[name]) for name in data.files
        }
    if "gmm_transform" in state:
        value = np.asarray(state["gmm_transform"]).item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        state["transform_type"] = str(value)
    else:
        standardized = int(np.asarray(state.get("gmm_standardize_data", 1)).item())
        state["transform_type"] = "standardize" if standardized else "raw"
    return state


def _latent_shape(state: Mapping[str, object], dim: int) -> tuple[int, ...]:
    if "latent_shape" not in state:
        return (dim,)
    shape = tuple(int(v) for v in np.asarray(state["latent_shape"]).reshape(-1))
    if int(np.prod(shape)) != dim:
        raise ValueError(f"latent_shape={shape} does not match GMM dimension {dim}")
    return shape


def gmm_component_moments_in_latent_space(
    state: Mapping[str, object],
    eps: float = 1e-12,
) -> dict[str, np.ndarray | str]:
    pi = np.asarray(state["pi"], dtype=np.float64).reshape(-1)
    pi = pi / np.sum(pi)
    mu_fit = np.asarray(state["mu"], dtype=np.float64)
    var_fit = np.maximum(np.asarray(state["var"], dtype=np.float64), eps)
    if mu_fit.shape != var_fit.shape or mu_fit.shape[0] != pi.size:
        raise ValueError("GMM pi, mu, and var shapes are inconsistent")

    num_components, dim = mu_fit.shape
    latent_shape = _latent_shape(state, dim)
    channels = int(latent_shape[-1]) if len(latent_shape) > 1 else 1
    spatial = dim // channels
    transform_type = str(state.get("transform_type", "standardize"))

    if transform_type == "channel_whiten":
        channel_mean = np.asarray(state["channel_mean"], dtype=np.float64)
        unwhiten = np.asarray(state["channel_unwhiten"], dtype=np.float64)
        mu_blocks = mu_fit.reshape(num_components, spatial, channels)
        var_blocks_fit = var_fit.reshape(num_components, spatial, channels)
        component_mean = (
            np.einsum("ksd,cd->ksc", mu_blocks, unwhiten) + channel_mean
        ).reshape(num_components, dim)
        covariance_blocks = np.einsum(
            "cd,ksd,ed->ksce",
            unwhiten,
            var_blocks_fit,
            unwhiten,
        )
    else:
        if transform_type == "raw":
            component_mean = mu_fit.copy()
            component_var_diag = var_fit.copy()
        elif transform_type == "standardize":
            offset = np.asarray(state["mean"], dtype=np.float64).reshape(-1)
            scale = np.asarray(state["std"], dtype=np.float64).reshape(-1)
            if offset.size != dim or scale.size != dim:
                raise ValueError("standardized GMM mean/std do not match component dimension")
            component_mean = mu_fit * scale + offset
            component_var_diag = var_fit * np.square(scale)
        else:
            raise ValueError(f"Unsupported GMM transform_type={transform_type!r}")

        diagonal_blocks = component_var_diag.reshape(
            num_components, spatial, channels
        )
        covariance_blocks = np.zeros(
            (num_components, spatial, channels, channels), dtype=np.float64
        )
        diagonal_index = np.arange(channels)
        covariance_blocks[:, :, diagonal_index, diagonal_index] = diagonal_blocks

    component_var_diag = np.diagonal(
        covariance_blocks, axis1=-2, axis2=-1
    ).reshape(num_components, dim)
    mixture_mean = np.sum(pi[:, None] * component_mean, axis=0)
    centered = component_mean - mixture_mean
    between_covariance = centered.T @ (centered * pi[:, None])

    weighted_blocks = np.sum(pi[:, None, None, None] * covariance_blocks, axis=0)
    within_covariance = np.zeros((dim, dim), dtype=np.float64)
    for spatial_index in range(spatial):
        start = spatial_index * channels
        stop = start + channels
        within_covariance[start:stop, start:stop] = weighted_blocks[spatial_index]
    mixture_covariance = within_covariance + between_covariance
    mixture_covariance = (mixture_covariance + mixture_covariance.T) * 0.5

    return {
        "transform_type": transform_type,
        "latent_shape": np.asarray(latent_shape, dtype=np.int32),
        "pi": pi,
        "component_mean": component_mean,
        "component_var_diag": component_var_diag,
        "component_covariance_blocks": covariance_blocks,
        "mixture_mean": mixture_mean,
        "within_component_covariance": within_covariance,
        "between_component_covariance": between_covariance,
        "mixture_covariance": mixture_covariance,
    }


def compare_gmm_to_population(
    population_mean: np.ndarray,
    population_covariance: np.ndarray,
    gmm_moments: Mapping[str, np.ndarray | str],
    *,
    mean_epsilon: float,
    dead_variance_threshold: float,
    population_eigenvalues: np.ndarray | None = None,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    population_mean = np.asarray(population_mean, dtype=np.float64)
    population_covariance = np.asarray(population_covariance, dtype=np.float64)
    mixture_mean = np.asarray(gmm_moments["mixture_mean"], dtype=np.float64)
    mixture_covariance = np.asarray(
        gmm_moments["mixture_covariance"], dtype=np.float64
    )
    if mixture_mean.shape != population_mean.shape:
        raise ValueError("GMM and population latent dimensions do not match")

    metrics, eigenvalues = covariance_summary(
        mixture_mean,
        mixture_covariance,
        mean_epsilon=mean_epsilon,
        dead_variance_threshold=dead_variance_threshold,
    )
    population_diag = np.maximum(np.diag(population_covariance), 1e-30)
    mixture_diag = np.maximum(np.diag(mixture_covariance), 0.0)
    variance_ratio = mixture_diag / population_diag
    if population_eigenvalues is None:
        population_eigenvalues = np.linalg.eigvalsh(
            (population_covariance + population_covariance.T) * 0.5
        )
    mean_delta = mixture_mean - population_mean
    metrics.update(
        {
            "transform_type": str(gmm_moments["transform_type"]),
            "num_components": int(np.asarray(gmm_moments["pi"]).size),
            "mean_gap_rms": float(
                np.linalg.norm(mean_delta) / math.sqrt(max(mean_delta.size, 1))
            ),
            "mean_gap_l2": float(np.linalg.norm(mean_delta)),
            "covariance_relative_frobenius_error": float(
                np.linalg.norm(mixture_covariance - population_covariance)
                / max(np.linalg.norm(population_covariance), 1e-30)
            ),
            "covariance_trace_ratio": float(
                np.trace(mixture_covariance)
                / max(np.trace(population_covariance), 1e-30)
            ),
            "variance_ratio_median": _quantile(variance_ratio, 0.5),
            "variance_ratio_q05": _quantile(variance_ratio, 0.05),
            "variance_ratio_q95": _quantile(variance_ratio, 0.95),
            "effective_rank_gap": float(
                _effective_rank(eigenvalues)
                - _effective_rank(population_eigenvalues)
            ),
        }
    )
    return metrics, eigenvalues


def gmm_moments_fingerprint(
    gmm_moments: Mapping[str, np.ndarray | str],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(gmm_moments["transform_type"]).encode("utf-8"))
    for name in (
        "latent_shape",
        "pi",
        "component_mean",
        "component_covariance_blocks",
    ):
        value = np.ascontiguousarray(np.asarray(gmm_moments[name]))
        digest.update(name.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def component_summary_rows(
    gmm_moments: Mapping[str, np.ndarray | str],
) -> list[dict[str, float | int]]:
    pi = np.asarray(gmm_moments["pi"], dtype=np.float64)
    means = np.asarray(gmm_moments["component_mean"], dtype=np.float64)
    variances = np.asarray(gmm_moments["component_var_diag"], dtype=np.float64)
    mixture_mean = np.asarray(gmm_moments["mixture_mean"], dtype=np.float64)
    rows = []
    for component in range(pi.size):
        rows.append(
            {
                "component": component,
                "pi": float(pi[component]),
                "mean_rms": float(
                    np.linalg.norm(means[component]) / math.sqrt(means.shape[1])
                ),
                "center_distance_to_mixture": float(
                    np.linalg.norm(means[component] - mixture_mean)
                ),
                "variance_mean": float(np.mean(variances[component])),
                "variance_min": float(np.min(variances[component])),
                "variance_max": float(np.max(variances[component])),
                "variance_trace": float(np.sum(variances[component])),
            }
        )
    return rows


def safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return label or "gmm"


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_rows_csv(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics_rows(
    metrics: Mapping[str, object], group: str
) -> list[dict[str, object]]:
    rows = []
    for name, value in sorted(metrics.items()):
        if isinstance(value, (int, float, np.integer, np.floating)):
            rows.append({"group": group, "metric": name, "value": value})
    return rows
