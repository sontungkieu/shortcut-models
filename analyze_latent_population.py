from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from absl import app, flags

from latent_geometry import GeometryConfig, run_geometry_diagnostics
from latent_population import (
    PosteriorMoments,
    compare_gmm_to_population,
    component_summary_rows,
    finalize_posterior_moments,
    flatten_metrics_rows,
    gmm_component_moments_in_latent_space,
    gmm_moments_fingerprint,
    load_gmm_npz,
    posterior_population_summary,
    safe_label,
    write_json,
    write_rows_csv,
)
from utils.datasets import get_dataset_for_statistics
from utils.stable_vae import StableVAE


matplotlib.use("Agg")
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_name", "celebahq256", "TFDS dataset name.")
flags.DEFINE_string("tfds_data_dir", None, "Optional TFDS data directory.")
flags.DEFINE_string("split", "train", "Finite TFDS split used for statistics.")
flags.DEFINE_integer("batch_size", 64, "VAE encoding batch size.")
flags.DEFINE_integer("max_samples", 0, "Maximum samples; 0 means the full split.")
flags.DEFINE_multi_string(
    "gmm_stats_path",
    [],
    "Optional GMM .npz path. Repeat the flag to compare multiple GMMs.",
)
flags.DEFINE_multi_string(
    "gmm_label",
    [],
    "Optional label matching each --gmm_stats_path.",
)
flags.DEFINE_string(
    "output_dir",
    "/kaggle/working/latent_population_analysis",
    "Directory for JSON, CSV, NPZ, JSONL, and plot outputs.",
)
flags.DEFINE_string(
    "cache_dir",
    "/tmp/latent_population_analysis",
    "Temporary directory for posterior mean/radius caches.",
)
flags.DEFINE_integer("keep_cache", 0, "Keep temporary memmap caches, as 1/0.")
flags.DEFINE_string(
    "moment_backend",
    "auto",
    "Moment accumulator backend: auto, numpy, or jax.",
)
flags.DEFINE_enum(
    "population_mode",
    "aggregated_posterior",
    ("aggregated_posterior", "posterior_mean"),
    "Analyze sampled posterior moments or deterministic posterior means.",
)
flags.DEFINE_float(
    "mean_epsilon",
    1e-2,
    "Threshold used by the near-zero latent-mean fraction.",
)
flags.DEFINE_float(
    "dead_variance_threshold",
    1e-6,
    "Variance threshold used to classify inactive latent dimensions.",
)
flags.DEFINE_integer(
    "progress_every",
    10,
    "Write a progress event every this many batches.",
)
flags.DEFINE_integer(
    "extended_geometry_diagnostics",
    0,
    "Run split covariance, whitening, local PCA, held-out density, and C2ST, as 1/0.",
)
flags.DEFINE_integer("geometry_seed", 20260726, "Geometry diagnostic RNG seed.")
flags.DEFINE_float(
    "geometry_train_fraction",
    0.8,
    "Train fraction for whitening, density fitting, and C2ST.",
)
flags.DEFINE_integer(
    "geometry_split_half_repeats",
    5,
    "Number of random split-half covariance comparisons.",
)
flags.DEFINE_integer(
    "geometry_whitening_projections",
    100,
    "Number of whitened random projections.",
)
flags.DEFINE_float(
    "geometry_whitening_eigen_floor_relative",
    1e-6,
    "Whitening eigenvalue floor relative to the largest train eigenvalue.",
)
flags.DEFINE_integer(
    "geometry_ppca_rank",
    256,
    "Rank of the low-rank plus isotropic Gaussian baseline.",
)
flags.DEFINE_integer(
    "geometry_local_pool_size",
    4096,
    "Reference pool size for local-PCA nearest neighbors.",
)
flags.DEFINE_integer(
    "geometry_local_query_count",
    256,
    "Number of query points for local-PCA dimensions.",
)
flags.DEFINE_string(
    "geometry_local_neighbor_counts",
    "20,50,100",
    "Comma-separated local-PCA neighbor counts.",
)
flags.DEFINE_float(
    "geometry_local_variance_fraction",
    0.9,
    "Local variance fraction used to define local PCA dimension.",
)
flags.DEFINE_integer(
    "geometry_heldout_gmm_modes",
    16,
    "Component count for the train-only held-out GMM refit.",
)
flags.DEFINE_integer(
    "geometry_heldout_gmm_em_iters",
    25,
    "EM iterations for the train-only held-out GMM refit.",
)
flags.DEFINE_integer(
    "geometry_heldout_gmm_chunk_size",
    128,
    "Chunk size for held-out GMM EM and likelihood.",
)
flags.DEFINE_integer(
    "geometry_c2st_sample_count",
    28000,
    "Real and generated samples per classifier two-sample test.",
)
flags.DEFINE_integer(
    "geometry_c2st_batch_size",
    512,
    "Classifier two-sample training batch size.",
)
flags.DEFINE_integer(
    "geometry_c2st_logistic_steps",
    600,
    "Logistic classifier optimizer steps.",
)
flags.DEFINE_integer(
    "geometry_c2st_mlp_steps",
    1000,
    "Small-MLP classifier optimizer steps.",
)
flags.DEFINE_integer(
    "geometry_c2st_mlp_hidden_size",
    128,
    "Small-MLP hidden width.",
)
flags.DEFINE_float(
    "geometry_c2st_learning_rate",
    3e-4,
    "Classifier two-sample learning rate.",
)
flags.DEFINE_float(
    "geometry_c2st_weight_decay",
    1e-4,
    "Classifier two-sample AdamW decay.",
)
flags.DEFINE_integer(
    "geometry_knn_subset_size",
    4096,
    "Per-distribution subset size for kNN precision/recall.",
)
flags.DEFINE_integer(
    "geometry_knn_k",
    5,
    "Neighbor rank for kNN precision/recall radii.",
)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def _event(
    progress_path: Path,
    phase: str,
    started_at: float,
    **payload,
) -> None:
    row = {
        "phase": phase,
        "elapsed_seconds": time.monotonic() - started_at,
        "timestamp_unix": time.time(),
        **payload,
    }
    _append_jsonl(progress_path, row)
    printable = " ".join(f"{key}={value}" for key, value in payload.items())
    print(f"LATENT_STATS phase={phase} {printable}", flush=True)


@jax.jit
def _jax_moment_update(state, posterior_mean, posterior_var):
    count, mean, between_m2, posterior_var_sum = state
    posterior_mean = jnp.reshape(posterior_mean, (posterior_mean.shape[0], -1))
    posterior_var = jnp.reshape(posterior_var, posterior_mean.shape)
    batch_count = jnp.asarray(posterior_mean.shape[0], dtype=jnp.float32)
    batch_mean = jnp.mean(posterior_mean, axis=0)
    centered = posterior_mean - batch_mean
    batch_m2 = centered.T @ centered
    total = count + batch_count
    delta = batch_mean - mean
    between_m2 = (
        between_m2
        + batch_m2
        + jnp.outer(delta, delta) * (count * batch_count / jnp.maximum(total, 1.0))
    )
    mean = mean + delta * (batch_count / jnp.maximum(total, 1.0))
    posterior_var_sum = posterior_var_sum + jnp.sum(posterior_var, axis=0)
    return total, mean, between_m2, posterior_var_sum


def _resolve_backend(value: str) -> str:
    value = value.strip().lower()
    if value == "auto":
        return "jax" if jax.default_backend() in ("gpu", "tpu") else "numpy"
    if value not in ("jax", "numpy"):
        raise ValueError("--moment_backend must be auto, numpy, or jax")
    return value


def _expected_radii(
    mean_cache: np.memmap,
    posterior_trace_cache: np.memmap,
    population_mean: np.ndarray,
    chunk_size: int,
    include_posterior_noise: bool,
) -> np.ndarray:
    radii = np.empty((mean_cache.shape[0],), dtype=np.float32)
    for start in range(0, mean_cache.shape[0], chunk_size):
        stop = min(start + chunk_size, mean_cache.shape[0])
        centered = (
            np.asarray(mean_cache[start:stop], dtype=np.float64)
            - population_mean[None]
        )
        expected_sq = np.sum(centered * centered, axis=1)
        if include_posterior_noise:
            expected_sq += np.asarray(
                posterior_trace_cache[start:stop], dtype=np.float64
            )
        radii[start:stop] = np.sqrt(np.maximum(expected_sq, 0.0))
    return radii


def _write_population_artifacts(
    output_dir: Path,
    moments: dict[str, np.ndarray | int],
    summary: dict[str, float | int],
    eigenvalues: np.ndarray,
    radii: np.ndarray,
    latent_shape: tuple[int, ...],
    population_mode: str,
) -> None:
    mean = np.asarray(moments["mean"], dtype=np.float64)
    between_cov = np.asarray(moments["between_covariance"], dtype=np.float64)
    posterior_var = np.asarray(
        moments["posterior_noise_variance"], dtype=np.float64
    )
    aggregated_cov = np.asarray(
        moments["aggregated_covariance"], dtype=np.float64
    )
    selected_covariance = (
        aggregated_cov
        if population_mode == "aggregated_posterior"
        else between_cov
    )
    np.savez_compressed(
        output_dir / "latent_population_stats.npz",
        num_samples=np.asarray(moments["count"], dtype=np.int64),
        latent_shape=np.asarray(latent_shape, dtype=np.int32),
        posterior_mean_vector=mean.astype(np.float32),
        between_image_covariance=between_cov.astype(np.float32),
        posterior_noise_variance=posterior_var.astype(np.float32),
        aggregated_posterior_covariance=aggregated_cov.astype(np.float32),
        selected_population_covariance=selected_covariance.astype(np.float32),
        population_mode=np.asarray(population_mode),
        covariance_eigenvalues=np.asarray(eigenvalues, dtype=np.float32),
        expected_radii=np.asarray(radii, dtype=np.float32),
    )
    write_json(output_dir / "latent_population_summary.json", summary)
    write_rows_csv(
        output_dir / "latent_population_metrics.csv",
        flatten_metrics_rows(summary, "aggregated_posterior"),
    )
    dimension_rows = []
    between_var = np.diag(between_cov)
    total_var = np.diag(aggregated_cov)
    for index in range(mean.size):
        dimension_rows.append(
            {
                "dimension": index,
                "mean": float(mean[index]),
                "between_image_variance": float(between_var[index]),
                "posterior_noise_variance": float(posterior_var[index]),
                "aggregated_variance": float(total_var[index]),
            }
        )
    write_rows_csv(output_dir / "latent_dimension_stats.csv", dimension_rows)


def _plot_population(
    output_dir: Path,
    eigenvalues: np.ndarray,
    variances: np.ndarray,
    population_mode: str,
) -> None:
    population_title = (
        "Aggregated VAE Posterior"
        if population_mode == "aggregated_posterior"
        else "Deterministic VAE Posterior Mean"
    )
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
    cumulative = np.cumsum(eigenvalues) / max(float(np.sum(eigenvalues)), 1e-30)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(np.arange(1, cumulative.size + 1), cumulative, linewidth=1.6)
    for fraction in (0.90, 0.95, 0.99):
        axis.axhline(fraction, color="0.75", linewidth=0.8, linestyle="--")
    axis.set(
        xlabel="Number of principal components",
        ylabel="Cumulative explained variance",
        ylim=(0.0, 1.01),
        title=f"{population_title}: Explained Variance",
    )
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_explained_variance.png", dpi=180)
    plt.close(fig)

    variances = np.maximum(np.asarray(variances, dtype=np.float64), 1e-30)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].hist(np.log10(variances), bins=60, color="#35618f", alpha=0.9)
    axes[0].set(
        xlabel="log10 variance",
        ylabel="Latent dimensions",
        title="Per-Dimension Variance",
    )
    axes[1].boxplot(variances, vert=True, showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set(ylabel="Variance", xticks=[], title="Variance Distribution")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "latent_dimension_variance.png", dpi=180)
    plt.close(fig)


def _analyze_gmm(
    gmm_path: Path,
    label: str,
    output_dir: Path,
    population_mean: np.ndarray,
    population_covariance: np.ndarray,
    population_eigenvalues: np.ndarray,
    canonical_by_fingerprint: dict[str, dict[str, object]],
) -> dict[str, object]:
    state = load_gmm_npz(gmm_path)
    moments = gmm_component_moments_in_latent_space(state)
    fingerprint = gmm_moments_fingerprint(moments)
    duplicate = canonical_by_fingerprint.get(fingerprint)
    if duplicate is None:
        metrics, eigenvalues = compare_gmm_to_population(
            population_mean,
            population_covariance,
            moments,
            mean_epsilon=FLAGS.mean_epsilon,
            dead_variance_threshold=FLAGS.dead_variance_threshold,
            population_eigenvalues=population_eigenvalues,
        )
    else:
        metrics = dict(duplicate["metrics"])
        eigenvalues = np.asarray(duplicate["eigenvalues"], dtype=np.float64)
    gmm_dir = output_dir / "gmm" / safe_label(label)
    gmm_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = gmm_dir / "gmm_moment_stats.npz"
    if duplicate is not None:
        artifact_path = output_dir / duplicate["artifact_path"]
    metrics.update(
        {
            "label": label,
            "gmm_stats_path": str(gmm_path),
            "gmm_moments_sha256": fingerprint,
            "duplicate_of": "" if duplicate is None else duplicate["label"],
            "gmm_moment_stats_path": str(artifact_path.relative_to(output_dir)),
        }
    )
    if duplicate is None:
        np.savez_compressed(
            artifact_path,
            transform_type=np.asarray(moments["transform_type"]),
            latent_shape=np.asarray(moments["latent_shape"], dtype=np.int32),
            pi=np.asarray(moments["pi"], dtype=np.float32),
            component_mean=np.asarray(moments["component_mean"], dtype=np.float32),
            component_var_diag=np.asarray(
                moments["component_var_diag"], dtype=np.float32
            ),
            component_covariance_blocks=np.asarray(
                moments["component_covariance_blocks"], dtype=np.float32
            ),
            mixture_mean=np.asarray(moments["mixture_mean"], dtype=np.float32),
            within_component_covariance=np.asarray(
                moments["within_component_covariance"], dtype=np.float32
            ),
            between_component_covariance=np.asarray(
                moments["between_component_covariance"], dtype=np.float32
            ),
            mixture_covariance=np.asarray(
                moments["mixture_covariance"], dtype=np.float32
            ),
            mixture_covariance_eigenvalues=np.asarray(
                eigenvalues, dtype=np.float32
            ),
        )
        canonical_by_fingerprint[fingerprint] = {
            "label": label,
            "artifact_path": str(artifact_path.relative_to(output_dir)),
            "metrics": dict(metrics),
            "eigenvalues": np.asarray(eigenvalues, dtype=np.float64),
        }
    write_json(gmm_dir / "gmm_summary.json", metrics)
    write_rows_csv(
        gmm_dir / "gmm_component_stats.csv",
        component_summary_rows(moments),
    )
    return metrics


def main(argv):
    del argv
    output_dir = Path(FLAGS.output_dir)
    cache_dir = Path(FLAGS.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "latent_stats_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    started_at = time.monotonic()

    if FLAGS.gmm_label and len(FLAGS.gmm_label) != len(FLAGS.gmm_stats_path):
        raise ValueError("--gmm_label count must match --gmm_stats_path count")
    labels = list(FLAGS.gmm_label) or [
        Path(path).stem for path in FLAGS.gmm_stats_path
    ]
    backend = _resolve_backend(FLAGS.moment_backend)
    _event(
        progress_path,
        "start",
        started_at,
        dataset=FLAGS.dataset_name,
        split=FLAGS.split,
        backend=backend,
        jax_backend=jax.default_backend(),
        device_count=jax.device_count(),
    )

    dataset, expected_samples = get_dataset_for_statistics(
        FLAGS.dataset_name,
        FLAGS.batch_size,
        data_dir=FLAGS.tfds_data_dir,
        split=FLAGS.split,
        max_samples=FLAGS.max_samples,
    )
    vae = StableVAE.create()
    encode_moments = jax.jit(vae.encode_posterior_moments)

    mean_cache = None
    posterior_trace_cache = None
    numpy_state = None
    jax_state = None
    latent_shape = None
    written = 0
    batch_index = 0

    for images, _ in dataset:
        if written >= expected_samples:
            break
        take = min(images.shape[0], expected_samples - written)
        images = images[:take]
        posterior_mean, posterior_var = encode_moments(images)
        if latent_shape is None:
            latent_shape = tuple(int(v) for v in posterior_mean.shape[1:])
            dim = int(np.prod(latent_shape))
            mean_cache = np.memmap(
                cache_dir / "posterior_mean.dat",
                mode="w+",
                dtype=np.float32,
                shape=(expected_samples, dim),
            )
            posterior_trace_cache = np.memmap(
                cache_dir / "posterior_variance_trace.dat",
                mode="w+",
                dtype=np.float32,
                shape=(expected_samples,),
            )
            if backend == "numpy":
                numpy_state = PosteriorMoments.empty(dim)
            else:
                jax_state = (
                    jnp.asarray(0.0, dtype=jnp.float32),
                    jnp.zeros((dim,), dtype=jnp.float32),
                    jnp.zeros((dim, dim), dtype=jnp.float32),
                    jnp.zeros((dim,), dtype=jnp.float32),
                )

        flat_mean = jnp.reshape(posterior_mean, (take, -1))
        flat_var = jnp.reshape(posterior_var, (take, -1))
        if backend == "jax":
            jax_state = _jax_moment_update(jax_state, flat_mean, flat_var)

        flat_mean_host, flat_var_host = jax.device_get((flat_mean, flat_var))
        flat_mean_host = np.asarray(flat_mean_host, dtype=np.float32)
        flat_var_host = np.asarray(flat_var_host, dtype=np.float32)
        if backend == "numpy":
            numpy_state.update(flat_mean_host, flat_var_host)
        mean_cache[written : written + take] = flat_mean_host
        posterior_trace_cache[written : written + take] = np.sum(
            flat_var_host, axis=1, dtype=np.float32
        )
        written += take
        batch_index += 1
        if batch_index % max(FLAGS.progress_every, 1) == 0 or written == expected_samples:
            _event(
                progress_path,
                "encode",
                started_at,
                batch=batch_index,
                samples=written,
                expected_samples=expected_samples,
            )

    if written != expected_samples or latent_shape is None:
        raise RuntimeError(
            f"Expected {expected_samples} samples but encoded {written}"
        )
    mean_cache.flush()
    posterior_trace_cache.flush()

    if backend == "jax":
        count, mean, between_m2, posterior_var_sum = jax.device_get(jax_state)
        moments = finalize_posterior_moments(
            int(round(float(count))),
            mean,
            between_m2,
            posterior_var_sum,
        )
    else:
        moments = numpy_state.finalize()
    _event(progress_path, "moments_finalized", started_at, samples=written)

    radii = _expected_radii(
        mean_cache,
        posterior_trace_cache,
        np.asarray(moments["mean"], dtype=np.float64),
        max(FLAGS.batch_size, 1),
        FLAGS.population_mode == "aggregated_posterior",
    )
    summary, eigenvalues = posterior_population_summary(
        moments,
        radii,
        mean_epsilon=FLAGS.mean_epsilon,
        dead_variance_threshold=FLAGS.dead_variance_threshold,
        population_mode=FLAGS.population_mode,
    )
    summary.update(
        {
            "dataset_name": FLAGS.dataset_name,
            "split": FLAGS.split,
            "latent_shape": list(latent_shape),
            "vae_scaling_factor": float(vae.module.config.scaling_factor),
            "latent_definition": (
                "scaled aggregated VAE posterior: "
                "z=s*(posterior_mean+posterior_std*epsilon)"
                if FLAGS.population_mode == "aggregated_posterior"
                else "scaled deterministic VAE posterior mean: z=s*posterior_mean"
            ),
            "covariance_divisor": "population_N",
            "radius_definition": (
                "sqrt(||posterior_mean_i-population_mean||^2"
                "+sum(posterior_var_i))"
                if FLAGS.population_mode == "aggregated_posterior"
                else "||posterior_mean_i-population_mean||"
            ),
            "moment_backend": backend,
            "jax_backend": jax.default_backend(),
        }
    )
    _write_population_artifacts(
        output_dir,
        moments,
        summary,
        eigenvalues,
        radii,
        latent_shape,
        FLAGS.population_mode,
    )
    _plot_population(
        output_dir,
        eigenvalues,
        np.diag(
            np.asarray(
                moments[
                    "aggregated_covariance"
                    if FLAGS.population_mode == "aggregated_posterior"
                    else "between_covariance"
                ]
            )
        ),
        FLAGS.population_mode,
    )
    _event(progress_path, "population_outputs_written", started_at)

    comparison_rows = []
    canonical_by_fingerprint: dict[str, dict[str, object]] = {}
    for gmm_path_value, label in zip(FLAGS.gmm_stats_path, labels):
        gmm_path = Path(gmm_path_value)
        _event(
            progress_path,
            "gmm_start",
            started_at,
            label=label,
            path=str(gmm_path),
        )
        comparison_rows.append(
            _analyze_gmm(
                gmm_path,
                label,
                output_dir,
                np.asarray(moments["mean"], dtype=np.float64),
                np.asarray(
                    moments[
                        "aggregated_covariance"
                        if FLAGS.population_mode == "aggregated_posterior"
                        else "between_covariance"
                    ],
                    dtype=np.float64,
                ),
                eigenvalues,
                canonical_by_fingerprint,
            )
        )
        _event(progress_path, "gmm_complete", started_at, label=label)
    if comparison_rows:
        write_rows_csv(output_dir / "gmm_comparison.csv", comparison_rows)
        write_json(
            output_dir / "gmm_comparison.json",
            {"population": summary, "gmms": comparison_rows},
        )

    if FLAGS.extended_geometry_diagnostics:
        neighbor_counts = tuple(
            int(value.strip())
            for value in FLAGS.geometry_local_neighbor_counts.split(",")
            if value.strip()
        )
        geometry_config = GeometryConfig(
            seed=FLAGS.geometry_seed,
            train_fraction=FLAGS.geometry_train_fraction,
            split_half_repeats=FLAGS.geometry_split_half_repeats,
            whitening_projection_count=FLAGS.geometry_whitening_projections,
            whitening_eigenvalue_floor_relative=(
                FLAGS.geometry_whitening_eigen_floor_relative
            ),
            ppca_rank=FLAGS.geometry_ppca_rank,
            local_pool_size=FLAGS.geometry_local_pool_size,
            local_query_count=FLAGS.geometry_local_query_count,
            local_neighbor_counts=neighbor_counts,
            local_variance_fraction=FLAGS.geometry_local_variance_fraction,
            heldout_gmm_modes=FLAGS.geometry_heldout_gmm_modes,
            heldout_gmm_em_iters=FLAGS.geometry_heldout_gmm_em_iters,
            heldout_gmm_chunk_size=FLAGS.geometry_heldout_gmm_chunk_size,
            c2st_sample_count=FLAGS.geometry_c2st_sample_count,
            c2st_batch_size=FLAGS.geometry_c2st_batch_size,
            c2st_logistic_steps=FLAGS.geometry_c2st_logistic_steps,
            c2st_mlp_steps=FLAGS.geometry_c2st_mlp_steps,
            c2st_mlp_hidden_size=FLAGS.geometry_c2st_mlp_hidden_size,
            c2st_learning_rate=FLAGS.geometry_c2st_learning_rate,
            c2st_weight_decay=FLAGS.geometry_c2st_weight_decay,
            knn_subset_size=FLAGS.geometry_knn_subset_size,
            knn_k=FLAGS.geometry_knn_k,
        )

        def geometry_event(*, phase: str, **payload) -> None:
            _event(progress_path, phase, started_at, **payload)

        run_geometry_diagnostics(
            mean_cache,
            gmm_paths=list(FLAGS.gmm_stats_path),
            gmm_labels=labels,
            output_dir=output_dir / "geometry_diagnostics",
            config=geometry_config,
            event_callback=geometry_event,
        )

    _event(progress_path, "complete", started_at, output_dir=str(output_dir))
    if not FLAGS.keep_cache:
        del mean_cache
        del posterior_trace_cache
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    app.run(main)
