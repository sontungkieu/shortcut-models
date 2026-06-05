import json
import os
from typing import Tuple

import jax
import numpy as np
import wandb
from absl import app, flags
from ml_collections import config_flags

from gmm_utils import fit_diag_gmm, gmm_diagnostics, json_default, json_dump, save_gmm_stats
from metrics_io import append_metrics_csv, clear_metrics_csv
from utils.datasets import get_dataset
from utils.stable_vae import StableVAE
from utils.wandb import default_wandb_config, setup_wandb


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_name", "celebahq256", "TFDS dataset name.")
flags.DEFINE_string("tfds_data_dir", None, "Optional TFDS data_dir, useful for Kaggle caches.")
flags.DEFINE_integer("batch_size", 64, "Batch size used for VAE latent extraction.")
flags.DEFINE_integer("seed", 10, "Random seed.")
flags.DEFINE_integer("debug_overfit", 0, "Use a tiny repeated dataset for debugging.")
flags.DEFINE_string("gmm_save_path", "/kaggle/working/celebahq256_gmm_stats.npz", "Output .npz path.")
flags.DEFINE_string("gmm_latent_cache_path", "/kaggle/working/gmm_latents.dat", "Memmap cache path for train latents.")
flags.DEFINE_integer("gmm_fit_samples", 32768, "Number of train latents to fit the GMM.")
flags.DEFINE_integer("gmm_valid_samples", 4096, "Number of validation latents for diagnostics.")
flags.DEFINE_integer("gmm_num_modes", 64, "Number of diagonal GMM components.")
flags.DEFINE_integer("gmm_em_iters", 25, "EM iterations per restart.")
flags.DEFINE_integer("gmm_em_restarts", 1, "Number of EM restarts.")
flags.DEFINE_integer("gmm_init_seed", 0, "Seed for GMM initialization.")
flags.DEFINE_integer("gmm_em_chunk_size", 128, "Chunk size for EM responsibility passes.")
flags.DEFINE_string("gmm_pi_prior_type", "dirichlet", "Pi prior type: none, dirichlet, or kl.")
flags.DEFINE_float("gmm_pi_prior_strength", 1e-2, "Strength for the selected pi prior.")
flags.DEFINE_integer("gmm_pi_kl_steps", 100, "Optimizer steps for KL-regularized pi M-step.")
flags.DEFINE_float("gmm_pi_kl_lr", 0.2, "Optimizer learning rate for KL-regularized pi M-step.")
flags.DEFINE_string("gmm_var_prior_type", "none", "Variance prior type: none or kl.")
flags.DEFINE_float("gmm_var_prior_strength", 0.0, "Count-scale strength for KL variance prior.")
flags.DEFINE_float("gmm_var_prior_target_var", 1.0, "Target component variance in GMM fit space.")
flags.DEFINE_float("gmm_min_std", 0.0, "Absolute latent-space std floor.")
flags.DEFINE_float("gmm_min_std_data_frac", 1.0, "Relative floor as a fraction of global data std.")
flags.DEFINE_integer("gmm_standardize_data", 0, "Fit/infer GMM on per-dimension standardized latents, as 1/0.")
flags.DEFINE_float("gmm_standardize_eps", 1e-6, "Std epsilon for optional standardization.")
flags.DEFINE_string(
    "gmm_transform",
    "auto",
    "GMM fit transform: auto, raw, standardize, or channel_whiten.",
)
flags.DEFINE_integer("gmm_kmeanspp_init", 1, "Use k-means++ initialization for component means, as 1/0.")
flags.DEFINE_string(
    "gmm_init_strategy",
    "auto",
    "GMM mean initialization: auto, random, kmeans++, farthest, pca, or split.",
)
flags.DEFINE_integer(
    "gmm_init_warmup_iters",
    0,
    "Optional Lloyd/k-means refinement iterations after mean initialization.",
)
flags.DEFINE_integer("gmm_init_pca_dims", 16, "Number of PCA dimensions for pca initialization.")
flags.DEFINE_integer(
    "gmm_init_pca_max_samples",
    2048,
    "Maximum samples used to estimate PCA basis for pca initialization.",
)
flags.DEFINE_integer("gmm_keep_latent_cache", 0, "Keep latent memmap cache files after fitting, as 1/0.")
flags.DEFINE_string("gmm_fit_data_mode", "x1", "Final GMM fit data: x1 or mix.")
flags.DEFINE_float("gmm_mix_x1_prob", 0.5, "Probability of using an x1 latent in mixed GMM fitting.")
flags.DEFINE_integer("gmm_continue_em_iters", 0, "Extra warm-start EM iterations after the initial x1 GMM fit.")
flags.DEFINE_integer("gmm_mix_seed", 0, "Seed for sampling the mixed GMM fitting set.")
flags.DEFINE_string("metrics_output_path", None, "Optional JSON diagnostics output path.")
flags.DEFINE_string("gmm_em_metrics_output_path", None, "Optional JSONL path for per-EM-iteration diagnostics.")

wandb_config = default_wandb_config()
wandb_config.update(
    {
        "project": "shortcut",
        "name": "prepare_gmm_{dataset_name}",
    }
)
config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)


def _encode_batch(vae_encode, key, dataset_name: str, batch_images: np.ndarray) -> np.ndarray:
    if "latent" in dataset_name:
        if batch_images.shape[-1] % 2 == 0 and batch_images.shape[-1] > 4:
            batch_images = batch_images[..., batch_images.shape[-1] // 2 :]
        return np.asarray(batch_images, dtype=np.float32)
    latents = vae_encode(key, batch_images)
    return np.asarray(jax.device_get(latents), dtype=np.float32)


def _collect_latents(
    dataset,
    vae_encode,
    rng,
    dataset_name: str,
    num_samples: int,
    cache_path: str,
) -> Tuple[np.memmap, Tuple[int, ...]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    written = 0
    latents_mm = None
    latent_shape = None

    while written < num_samples:
        batch_images, _ = next(dataset)
        rng, batch_key = jax.random.split(rng)
        latents = _encode_batch(vae_encode, batch_key, dataset_name, batch_images)
        if latent_shape is None:
            latent_shape = tuple(latents.shape[1:])
            latents_mm = np.memmap(
                cache_path,
                mode="w+",
                dtype=np.float32,
                shape=(num_samples,) + latent_shape,
            )
        take = min(num_samples - written, latents.shape[0])
        latents_mm[written : written + take] = latents[:take]
        written += take
        print(f"Collected {written}/{num_samples} latents at {cache_path}", flush=True)

    latents_mm.flush()
    return latents_mm, latent_shape


def _moments_from_memmap(latents_mm: np.memmap, chunk_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = latents_mm.reshape((latents_mm.shape[0], -1))
    n, dim = flat.shape
    total = np.zeros((dim,), dtype=np.float64)
    total_sq = np.zeros((dim,), dtype=np.float64)
    for start in range(0, n, chunk_size):
        xb = np.asarray(flat[start : start + chunk_size], dtype=np.float32)
        total += np.sum(xb, axis=0, dtype=np.float64)
        total_sq += np.sum(xb * xb, axis=0, dtype=np.float64)
    mean = (total / max(n, 1)).astype(np.float32)
    var = np.maximum((total_sq / max(n, 1)) - mean.astype(np.float64) ** 2, 0.0).astype(np.float32)
    std = np.sqrt(np.maximum(var, FLAGS.gmm_standardize_eps)).astype(np.float32)
    return mean, std, var


def _standardize_to_memmap(
    latents_mm: np.memmap,
    mean: np.ndarray,
    std: np.ndarray,
    cache_path: str,
    chunk_size: int,
) -> np.memmap:
    flat = latents_mm.reshape((latents_mm.shape[0], -1))
    out = np.memmap(cache_path, mode="w+", dtype=np.float32, shape=flat.shape)
    denom = np.maximum(std, FLAGS.gmm_standardize_eps)
    for start in range(0, flat.shape[0], chunk_size):
        xb = np.asarray(flat[start : start + chunk_size], dtype=np.float32)
        out[start : start + xb.shape[0]] = (xb - mean) / denom
    out.flush()
    return out


def _normalize_gmm_transform(value: str, standardize_data: bool) -> str:
    name = str(value or "auto").strip().lower().replace("-", "_")
    if name in ("", "auto", "default"):
        return "standardize" if standardize_data else "raw"
    aliases = {
        "raw": "raw",
        "latent": "raw",
        "none": "raw",
        "std": "standardize",
        "standardized": "standardize",
        "standardize": "standardize",
        "diag_standardize": "standardize",
        "channel": "channel_whiten",
        "channel_whiten": "channel_whiten",
        "channel_whitening": "channel_whiten",
    }
    if name not in aliases:
        raise ValueError(f"Unknown gmm_transform {value!r}; expected auto, raw, standardize, or channel_whiten")
    return aliases[name]


def _channel_whitening_from_memmap(
    latents_mm: np.memmap,
    latent_shape: Tuple[int, ...],
    chunk_size: int,
    eps: float,
):
    if len(latent_shape) < 1:
        raise ValueError("channel_whiten requires a latent channel dimension")
    channels = int(latent_shape[-1])
    flat = latents_mm.reshape((latents_mm.shape[0], -1, channels))
    total = np.zeros((channels,), dtype=np.float64)
    total_outer = np.zeros((channels, channels), dtype=np.float64)
    count = 0
    for start in range(0, flat.shape[0], chunk_size):
        xb = np.asarray(flat[start : start + chunk_size], dtype=np.float64).reshape(-1, channels)
        total += np.sum(xb, axis=0)
        total_outer += xb.T @ xb
        count += xb.shape[0]
    channel_mean = (total / max(count, 1)).astype(np.float32)
    cov = total_outer / max(count, 1) - np.outer(channel_mean, channel_mean)
    cov = ((cov + cov.T) * 0.5).astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    whiten = (np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)
    unwhiten = (eigvecs @ np.diag(np.sqrt(eigvals))).astype(np.float32)
    offdiag = cov - np.diag(np.diag(cov))
    cov_energy = float(np.sum(cov * cov))
    metrics = {
        "channel_cov_trace": float(np.trace(cov)),
        "channel_cov_diag_mean": float(np.mean(np.diag(cov))),
        "channel_cov_offdiag_energy_fraction": float(np.sum(offdiag * offdiag) / max(cov_energy, eps)),
        "channel_cov_eig_min": float(np.min(eigvals)),
        "channel_cov_eig_max": float(np.max(eigvals)),
        "channel_cov_condition": float(np.max(eigvals) / max(np.min(eigvals), eps)),
    }
    return channel_mean, whiten, unwhiten, metrics


def _channel_transform_to_memmap(
    latents_mm: np.memmap,
    latent_shape: Tuple[int, ...],
    channel_mean: np.ndarray,
    channel_whiten: np.ndarray,
    cache_path: str,
    chunk_size: int,
) -> np.memmap:
    channels = int(latent_shape[-1])
    flat = latents_mm.reshape((latents_mm.shape[0], -1, channels))
    out = np.memmap(cache_path, mode="w+", dtype=np.float32, shape=(latents_mm.shape[0], int(np.prod(latent_shape))))
    for start in range(0, flat.shape[0], chunk_size):
        xb = np.asarray(flat[start : start + chunk_size], dtype=np.float32)
        z = np.einsum("bsc,dc->bsd", xb - channel_mean, channel_whiten)
        out[start : start + xb.shape[0]] = z.reshape((xb.shape[0], -1))
    out.flush()
    return out


def _inverse_transform_fit_to_latent_np(
    z_flat: np.ndarray,
    transform_type: str,
    latent_shape: Tuple[int, ...],
    mean: np.ndarray,
    std: np.ndarray,
    channel_mean: np.ndarray | None,
    channel_unwhiten: np.ndarray | None,
    eps: float,
) -> np.ndarray:
    z_flat = np.asarray(z_flat, dtype=np.float32)
    if transform_type == "channel_whiten":
        channels = int(latent_shape[-1])
        z = z_flat.reshape((z_flat.shape[0], -1, channels))
        x = np.einsum("bsd,cd->bsc", z, np.asarray(channel_unwhiten, dtype=np.float32))
        x = x + np.asarray(channel_mean, dtype=np.float32)
        return x.reshape(z_flat.shape).astype(np.float32)
    if transform_type == "standardize":
        return (z_flat * np.maximum(std.reshape(1, -1), eps) + mean.reshape(1, -1)).astype(np.float32)
    return z_flat.astype(np.float32)


def _transform_geometry_metrics(
    raw_flat,
    fit_flat,
    max_pairs: int = 2048,
    seed: int = 0,
    eps: float = 1e-8,
):
    n = min(int(raw_flat.shape[0]), int(fit_flat.shape[0]))
    if n < 2:
        return {}
    rng = np.random.default_rng(seed)
    num_pairs = min(max_pairs, n)
    a = rng.integers(0, n, size=num_pairs)
    b = rng.integers(0, n, size=num_pairs)
    same = a == b
    b[same] = (b[same] + 1) % n
    raw_a = np.asarray(raw_flat[a], dtype=np.float32)
    raw_b = np.asarray(raw_flat[b], dtype=np.float32)
    fit_a = np.asarray(fit_flat[a], dtype=np.float32)
    fit_b = np.asarray(fit_flat[b], dtype=np.float32)

    def cosine(x, y):
        dot = np.sum(x * y, axis=1)
        x_norm = np.sqrt(np.maximum(np.sum(x * x, axis=1), eps))
        y_norm = np.sqrt(np.maximum(np.sum(y * y, axis=1), eps))
        return dot / np.maximum(x_norm * y_norm, eps)

    raw_cos = np.clip(cosine(raw_a, raw_b), -1.0, 1.0)
    fit_cos = np.clip(cosine(fit_a, fit_b), -1.0, 1.0)
    corr = float(np.corrcoef(raw_cos, fit_cos)[0, 1]) if np.std(raw_cos) > eps and np.std(fit_cos) > eps else 0.0
    return {
        "transform_pair_cosine_corr": corr,
        "transform_pair_cosine_delta_mean": float(np.mean(fit_cos - raw_cos)),
        "transform_pair_cosine_abs_delta_mean": float(np.mean(np.abs(fit_cos - raw_cos))),
        "transform_pair_angle_abs_delta_mean": float(np.mean(np.abs(np.arccos(fit_cos) - np.arccos(raw_cos)))),
    }


def _normalize_gmm_fit_data_mode(value: str) -> str:
    mode = str(value).lower().replace("-", "_")
    if mode in ("x1", "data", "latent"):
        return "x1"
    if mode in ("mix", "x1_prior", "x1_x0", "mixed"):
        return "mix"
    raise ValueError(f"Unknown gmm_fit_data_mode {value!r}; expected x1 or mix")


def _mix_latents_to_memmap(
    x1_latents_mm: np.memmap,
    fit: dict,
    gmm_mean: np.ndarray,
    gmm_std: np.ndarray,
    cache_path: str,
    x1_prob: float,
    seed: int,
    chunk_size: int,
    eps: float,
    transform_type: str = "raw",
    latent_shape: Tuple[int, ...] | None = None,
    channel_mean: np.ndarray | None = None,
    channel_unwhiten: np.ndarray | None = None,
) -> np.memmap:
    flat_x1 = x1_latents_mm.reshape((x1_latents_mm.shape[0], -1))
    n, dim = flat_x1.shape
    out = np.memmap(cache_path, mode="w+", dtype=np.float32, shape=(n, dim))

    rng = np.random.default_rng(int(seed))
    pi = np.asarray(fit["pi"], dtype=np.float64).reshape(-1)
    pi = np.maximum(pi, eps)
    pi = pi / np.sum(pi)
    cdf = np.cumsum(pi)
    cdf[-1] = 1.0
    mu = np.asarray(fit["mu"], dtype=np.float32)
    sigma = np.sqrt(np.maximum(np.asarray(fit["var"], dtype=np.float32), eps))
    gmm_mean = np.asarray(gmm_mean, dtype=np.float32).reshape(1, -1)
    gmm_std = np.maximum(np.asarray(gmm_std, dtype=np.float32).reshape(1, -1), eps)

    x1_prob = float(np.clip(x1_prob, 0.0, 1.0))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        take = stop - start
        choose_x1 = rng.random(take) < x1_prob
        mixed = np.empty((take, dim), dtype=np.float32)
        if np.any(choose_x1):
            mixed[choose_x1] = np.asarray(flat_x1[start:stop][choose_x1], dtype=np.float32)
        n_prior = int(np.sum(~choose_x1))
        if n_prior:
            component_ids = np.searchsorted(cdf, rng.random(n_prior), side="right")
            component_ids = np.minimum(component_ids, pi.shape[0] - 1)
            noise = rng.standard_normal((n_prior, dim), dtype=np.float32)
            prior_fit_space = mu[component_ids] + sigma[component_ids] * noise
            mixed[~choose_x1] = _inverse_transform_fit_to_latent_np(
                prior_fit_space,
                transform_type,
                tuple(latent_shape or (dim,)),
                gmm_mean.reshape(-1),
                gmm_std.reshape(-1),
                channel_mean,
                channel_unwhiten,
                eps,
            )
        out[start:stop] = mixed
    out.flush()
    return out


def _cleanup_paths(*paths):
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _append_jsonl(path: str, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True, default=json_default))
        f.write("\n")


def main(_):
    np.random.seed(FLAGS.seed)
    rng = jax.random.PRNGKey(FLAGS.seed)
    gmm_fit_data_mode = _normalize_gmm_fit_data_mode(FLAGS.gmm_fit_data_mode)
    gmm_mix_x1_prob = float(np.clip(FLAGS.gmm_mix_x1_prob, 0.0, 1.0))
    gmm_continue_em_iters = max(int(FLAGS.gmm_continue_em_iters), 0)
    gmm_transform = _normalize_gmm_transform(FLAGS.gmm_transform, bool(FLAGS.gmm_standardize_data))

    if jax.process_index() == 0:
        setup_wandb(
            {
                "dataset_name": FLAGS.dataset_name,
                "gmm_num_modes": FLAGS.gmm_num_modes,
                "gmm_em_iters": FLAGS.gmm_em_iters,
                "gmm_em_restarts": FLAGS.gmm_em_restarts,
                "gmm_pi_prior_type": FLAGS.gmm_pi_prior_type,
                "gmm_pi_prior_strength": FLAGS.gmm_pi_prior_strength,
                "gmm_pi_kl_steps": FLAGS.gmm_pi_kl_steps,
                "gmm_pi_kl_lr": FLAGS.gmm_pi_kl_lr,
                "gmm_var_prior_type": FLAGS.gmm_var_prior_type,
                "gmm_var_prior_strength": FLAGS.gmm_var_prior_strength,
                "gmm_var_prior_target_var": FLAGS.gmm_var_prior_target_var,
                "gmm_min_std": FLAGS.gmm_min_std,
                "gmm_min_std_data_frac": FLAGS.gmm_min_std_data_frac,
                "gmm_standardize_data": FLAGS.gmm_standardize_data,
                "gmm_transform": gmm_transform,
                "gmm_init_strategy": FLAGS.gmm_init_strategy,
                "gmm_init_warmup_iters": FLAGS.gmm_init_warmup_iters,
                "gmm_init_pca_dims": FLAGS.gmm_init_pca_dims,
                "gmm_init_pca_max_samples": FLAGS.gmm_init_pca_max_samples,
                "gmm_fit_data_mode": gmm_fit_data_mode,
                "gmm_mix_x1_prob": gmm_mix_x1_prob,
                "gmm_continue_em_iters": gmm_continue_em_iters,
                "gmm_mix_seed": FLAGS.gmm_mix_seed,
            },
            **FLAGS.wandb,
        )
        if FLAGS.gmm_em_metrics_output_path:
            os.makedirs(os.path.dirname(FLAGS.gmm_em_metrics_output_path) or ".", exist_ok=True)
            open(FLAGS.gmm_em_metrics_output_path, "w", encoding="utf-8").close()
            clear_metrics_csv(FLAGS.gmm_em_metrics_output_path)
        clear_metrics_csv(FLAGS.metrics_output_path)

    def make_em_metrics_callback(fit_stage: str, em_iters: int, em_restarts: int):
        def em_metrics_callback(row):
            if jax.process_index() != 0:
                return
            payload = {
                "phase": "gmm_em",
                "dataset_name": FLAGS.dataset_name,
                "num_modes": int(FLAGS.gmm_num_modes),
                "em_iters": int(em_iters),
                "em_restarts": int(em_restarts),
                "gmm_fit_stage": fit_stage,
                "gmm_fit_data_mode": gmm_fit_data_mode,
                "gmm_mix_x1_prob": gmm_mix_x1_prob,
                "gmm_continue_em_iters": gmm_continue_em_iters,
                "gmm_pi_prior_type": FLAGS.gmm_pi_prior_type,
                "gmm_pi_prior_strength": float(FLAGS.gmm_pi_prior_strength),
                "gmm_var_prior_type": FLAGS.gmm_var_prior_type,
                "gmm_var_prior_strength": float(FLAGS.gmm_var_prior_strength),
                "gmm_var_prior_target_var": float(FLAGS.gmm_var_prior_target_var),
                "gmm_standardize_data": int(FLAGS.gmm_standardize_data),
                "gmm_transform": gmm_transform,
                "gmm_init_strategy": FLAGS.gmm_init_strategy,
                "gmm_init_warmup_iters": int(FLAGS.gmm_init_warmup_iters),
                "gmm_init_pca_dims": int(FLAGS.gmm_init_pca_dims),
                "gmm_init_pca_max_samples": int(FLAGS.gmm_init_pca_max_samples),
                **row,
            }
            _append_jsonl(FLAGS.gmm_em_metrics_output_path, payload)
            append_metrics_csv(FLAGS.gmm_em_metrics_output_path, payload)
            print(
                "GMM EM "
                f"stage={payload['gmm_fit_stage']} "
                f"restart={payload['restart']} iter={payload['iter']} "
                f"nll={payload['nll']:.6f} "
                f"pi=[{payload['pi_min']:.6f},{payload['pi_max']:.6f}] "
                f"counts=[{payload['count_min']:.1f},{payload['count_max']:.1f}] "
                f"dead={payload['dead_components']}",
                flush=True,
            )

        return em_metrics_callback

    dataset = get_dataset(
        FLAGS.dataset_name,
        FLAGS.batch_size,
        True,
        FLAGS.debug_overfit,
        data_dir=FLAGS.tfds_data_dir,
    )
    dataset_valid = get_dataset(
        FLAGS.dataset_name,
        FLAGS.batch_size,
        False,
        FLAGS.debug_overfit,
        data_dir=FLAGS.tfds_data_dir,
    )

    vae = None
    vae_encode = None
    if "latent" not in FLAGS.dataset_name:
        vae = StableVAE.create()
        vae_encode = jax.jit(vae.encode)

    train_cache_path = FLAGS.gmm_latent_cache_path
    valid_cache_path = train_cache_path + ".valid"
    std_cache_path = train_cache_path + ".std"
    valid_std_cache_path = train_cache_path + ".valid.std"
    mix_cache_path = train_cache_path + ".mix"
    mix_std_cache_path = train_cache_path + ".mix.std"
    transform_cache_path = train_cache_path + f".{gmm_transform}"
    valid_transform_cache_path = valid_cache_path + f".{gmm_transform}"
    mix_transform_cache_path = mix_cache_path + f".{gmm_transform}"

    print("Collecting train latents", flush=True)
    rng, train_rng = jax.random.split(rng)
    latents_mm, latent_shape = _collect_latents(
        dataset,
        vae_encode,
        train_rng,
        FLAGS.dataset_name,
        FLAGS.gmm_fit_samples,
        train_cache_path,
    )
    mean, std, data_var = _moments_from_memmap(latents_mm, FLAGS.gmm_em_chunk_size)
    flat_train = latents_mm.reshape((latents_mm.shape[0], -1))
    channel_mean = None
    channel_whiten = None
    channel_unwhiten = None
    transform_metrics = {}
    if gmm_transform == "standardize":
        x_train_gmm = _standardize_to_memmap(latents_mm, mean, std, std_cache_path, FLAGS.gmm_em_chunk_size)
        gmm_mean = mean
        gmm_std = std
        gmm_fit_space = "standardized"
        fit_data_std = std
    elif gmm_transform == "channel_whiten":
        channel_mean, channel_whiten, channel_unwhiten, transform_metrics = _channel_whitening_from_memmap(
            latents_mm,
            latent_shape,
            FLAGS.gmm_em_chunk_size,
            FLAGS.gmm_standardize_eps,
        )
        x_train_gmm = _channel_transform_to_memmap(
            latents_mm,
            latent_shape,
            channel_mean,
            channel_whiten,
            transform_cache_path,
            FLAGS.gmm_em_chunk_size,
        )
        gmm_mean = np.zeros_like(mean, dtype=np.float32)
        gmm_std = np.ones_like(std, dtype=np.float32)
        gmm_fit_space = "channel_whiten"
        fit_data_std = np.ones_like(std, dtype=np.float32)
    else:
        x_train_gmm = flat_train
        gmm_mean = np.zeros_like(mean, dtype=np.float32)
        gmm_std = np.ones_like(std, dtype=np.float32)
        gmm_fit_space = "latent"
        fit_data_std = std
    fit_var_estimate = np.var(np.asarray(x_train_gmm[: min(x_train_gmm.shape[0], 4096)], dtype=np.float32), axis=0)
    transform_metrics.update(
        {
            "gmm_transform_fit_variance_mean_sample": float(np.mean(fit_var_estimate)),
            "gmm_transform_fit_variance_trace_sample": float(np.sum(fit_var_estimate)),
        }
    )
    transform_metrics.update(
        {
            f"gmm_{name}": value
            for name, value in _transform_geometry_metrics(flat_train, x_train_gmm, seed=FLAGS.gmm_init_seed).items()
        }
    )

    x_valid_gmm = None
    if FLAGS.gmm_valid_samples > 0:
        print("Collecting validation latents", flush=True)
        rng, valid_rng = jax.random.split(rng)
        valid_mm, _ = _collect_latents(
            dataset_valid,
            vae_encode,
            valid_rng,
            FLAGS.dataset_name,
            FLAGS.gmm_valid_samples,
            valid_cache_path,
        )
        if gmm_transform == "standardize":
            x_valid_gmm = _standardize_to_memmap(valid_mm, mean, std, valid_std_cache_path, FLAGS.gmm_em_chunk_size)
        elif gmm_transform == "channel_whiten":
            x_valid_gmm = _channel_transform_to_memmap(
                valid_mm,
                latent_shape,
                channel_mean,
                channel_whiten,
                valid_transform_cache_path,
                FLAGS.gmm_em_chunk_size,
            )
        else:
            x_valid_gmm = valid_mm.reshape((valid_mm.shape[0], -1))

    def run_gmm_fit(x_fit, *, em_iters: int, em_restarts: int, seed: int, fit_stage: str, init_params=None):
        print(
            f"Fitting diagonal GMM stage={fit_stage} "
            f"em_iters={em_iters} em_restarts={em_restarts} "
            f"init={FLAGS.gmm_init_strategy} lloyd={FLAGS.gmm_init_warmup_iters}",
            flush=True,
        )
        return fit_diag_gmm(
            x_fit,
            num_modes=FLAGS.gmm_num_modes,
            em_iters=em_iters,
            em_restarts=em_restarts,
            seed=seed,
            pi_prior_type=FLAGS.gmm_pi_prior_type,
            pi_prior_strength=FLAGS.gmm_pi_prior_strength,
            pi_kl_steps=FLAGS.gmm_pi_kl_steps,
            pi_kl_lr=FLAGS.gmm_pi_kl_lr,
            var_prior_type=FLAGS.gmm_var_prior_type,
            var_prior_strength=FLAGS.gmm_var_prior_strength,
            var_prior_target_var=FLAGS.gmm_var_prior_target_var,
            min_std=FLAGS.gmm_min_std,
            min_std_data_frac=FLAGS.gmm_min_std_data_frac,
            data_std=fit_data_std,
            standardized=(gmm_transform != "raw"),
            chunk_size=FLAGS.gmm_em_chunk_size,
            use_kmeanspp=bool(FLAGS.gmm_kmeanspp_init),
            init_strategy=FLAGS.gmm_init_strategy,
            init_warmup_iters=FLAGS.gmm_init_warmup_iters,
            init_pca_dims=FLAGS.gmm_init_pca_dims,
            init_pca_max_samples=FLAGS.gmm_init_pca_max_samples,
            eps=FLAGS.gmm_standardize_eps,
            init_params=init_params,
            em_metrics_callback=make_em_metrics_callback(fit_stage, em_iters, em_restarts),
        )

    initial_fit = run_gmm_fit(
        x_train_gmm,
        em_iters=FLAGS.gmm_em_iters,
        em_restarts=FLAGS.gmm_em_restarts,
        seed=FLAGS.gmm_init_seed,
        fit_stage="initial_x1",
    )
    fit = initial_fit
    x_final_fit_gmm = x_train_gmm
    final_fit_stage = "initial_x1"

    if gmm_fit_data_mode == "mix":
        print(
            f"Building mixed GMM fit set with x1 probability {gmm_mix_x1_prob:.3f}",
            flush=True,
        )
        mix_mm = _mix_latents_to_memmap(
            latents_mm,
            initial_fit,
            gmm_mean,
            gmm_std,
            mix_cache_path,
            x1_prob=gmm_mix_x1_prob,
            seed=FLAGS.gmm_mix_seed,
            chunk_size=FLAGS.gmm_em_chunk_size,
            eps=FLAGS.gmm_standardize_eps,
            transform_type=gmm_transform,
            latent_shape=latent_shape,
            channel_mean=channel_mean,
            channel_unwhiten=channel_unwhiten,
        )
        if gmm_transform == "standardize":
            x_final_fit_gmm = _standardize_to_memmap(
                mix_mm,
                mean,
                std,
                mix_std_cache_path,
                FLAGS.gmm_em_chunk_size,
            )
        elif gmm_transform == "channel_whiten":
            x_final_fit_gmm = _channel_transform_to_memmap(
                mix_mm,
                latent_shape,
                channel_mean,
                channel_whiten,
                mix_transform_cache_path,
                FLAGS.gmm_em_chunk_size,
            )
        else:
            x_final_fit_gmm = mix_mm.reshape((mix_mm.shape[0], -1))
        if gmm_continue_em_iters > 0:
            fit = run_gmm_fit(
                x_final_fit_gmm,
                em_iters=gmm_continue_em_iters,
                em_restarts=1,
                seed=FLAGS.gmm_init_seed + 7919,
                fit_stage="mix_continue",
                init_params=initial_fit,
            )
            final_fit_stage = "mix_continue"
        else:
            fit = run_gmm_fit(
                x_final_fit_gmm,
                em_iters=FLAGS.gmm_em_iters,
                em_restarts=FLAGS.gmm_em_restarts,
                seed=FLAGS.gmm_init_seed + 7919,
                fit_stage="mix",
            )
            final_fit_stage = "mix"
    elif gmm_continue_em_iters > 0:
        fit = run_gmm_fit(
            x_train_gmm,
            em_iters=gmm_continue_em_iters,
            em_restarts=1,
            seed=FLAGS.gmm_init_seed + 7919,
            fit_stage="x1_continue",
            init_params=initial_fit,
        )
        final_fit_stage = "x1_continue"

    metrics = gmm_diagnostics(
        x_final_fit_gmm,
        fit["pi"],
        fit["mu"],
        fit["var"],
        fit["var_floor"],
        data_var=data_var,
        x_valid_std=x_valid_gmm,
        chunk_size=FLAGS.gmm_em_chunk_size,
        eps=FLAGS.gmm_standardize_eps,
    )
    if x_final_fit_gmm is not x_train_gmm:
        x1_eval_metrics = gmm_diagnostics(
            x_train_gmm,
            fit["pi"],
            fit["mu"],
            fit["var"],
            fit["var_floor"],
            data_var=data_var,
            x_valid_std=None,
            chunk_size=FLAGS.gmm_em_chunk_size,
            eps=FLAGS.gmm_standardize_eps,
        )
        metrics.update({f"x1_eval_{name}": value for name, value in x1_eval_metrics.items()})
    metrics.update(
        {
            "dataset_name": FLAGS.dataset_name,
            "fit_samples": int(FLAGS.gmm_fit_samples),
            "valid_samples": int(FLAGS.gmm_valid_samples),
            "latent_shape": list(latent_shape),
            "num_modes": int(FLAGS.gmm_num_modes),
            "em_iters": int(FLAGS.gmm_em_iters),
            "em_restarts": int(FLAGS.gmm_em_restarts),
            "best_restart": int(fit["restart"]),
            "final_train_nll": float(fit["nll"]),
            "gmm_pi_prior_type": FLAGS.gmm_pi_prior_type,
            "gmm_pi_prior_strength": float(FLAGS.gmm_pi_prior_strength),
            "gmm_pi_kl_steps": int(FLAGS.gmm_pi_kl_steps),
            "gmm_pi_kl_lr": float(FLAGS.gmm_pi_kl_lr),
            "gmm_var_prior_type": FLAGS.gmm_var_prior_type,
            "gmm_var_prior_strength": float(FLAGS.gmm_var_prior_strength),
            "gmm_var_prior_target_var": float(FLAGS.gmm_var_prior_target_var),
            "gmm_min_std": float(FLAGS.gmm_min_std),
            "gmm_min_std_data_frac": float(FLAGS.gmm_min_std_data_frac),
            "gmm_standardize_data": int(FLAGS.gmm_standardize_data),
            "gmm_transform": gmm_transform,
            "gmm_fit_space": gmm_fit_space,
            "gmm_init_strategy": FLAGS.gmm_init_strategy,
            "gmm_init_mode": fit.get("init_mode", ""),
            "gmm_init_warmup_iters": int(FLAGS.gmm_init_warmup_iters),
            "gmm_init_pca_dims": int(FLAGS.gmm_init_pca_dims),
            "gmm_init_pca_max_samples": int(FLAGS.gmm_init_pca_max_samples),
            "gmm_fit_data_mode": gmm_fit_data_mode,
            "gmm_mix_x1_prob": gmm_mix_x1_prob,
            "gmm_continue_em_iters": gmm_continue_em_iters,
            "gmm_mix_seed": int(FLAGS.gmm_mix_seed),
            "gmm_final_fit_stage": final_fit_stage,
            "gmm_initial_train_nll": float(initial_fit["nll"]),
            "gmm_initial_best_restart": int(initial_fit["restart"]),
            "gmm_em_metrics_output_path": FLAGS.gmm_em_metrics_output_path,
            "em_restart_traces": fit["restart_traces"],
            "em_best_trace": fit["trace"],
            "em_initial_restart_traces": initial_fit["restart_traces"],
            "em_initial_best_trace": initial_fit["trace"],
        }
    )
    metrics.update(transform_metrics)

    save_gmm_stats(
        FLAGS.gmm_save_path,
        pi=fit["pi"],
        mu=fit["mu"],
        var=fit["var"],
        mean=gmm_mean,
        std=gmm_std,
        data_mean=mean,
        data_std=std,
        data_var=data_var,
        var_floor=fit["var_floor"],
        latent_shape=np.asarray(latent_shape, dtype=np.int32),
        counts=fit["counts"],
        best_restart=np.asarray(fit["restart"], dtype=np.int32),
        gmm_pi_prior_type=np.asarray(FLAGS.gmm_pi_prior_type),
        gmm_pi_prior_strength=np.asarray(FLAGS.gmm_pi_prior_strength, dtype=np.float32),
        gmm_pi_kl_steps=np.asarray(FLAGS.gmm_pi_kl_steps, dtype=np.int32),
        gmm_pi_kl_lr=np.asarray(FLAGS.gmm_pi_kl_lr, dtype=np.float32),
        gmm_var_prior_type=np.asarray(FLAGS.gmm_var_prior_type),
        gmm_var_prior_strength=np.asarray(FLAGS.gmm_var_prior_strength, dtype=np.float32),
        gmm_var_prior_target_var=np.asarray(FLAGS.gmm_var_prior_target_var, dtype=np.float32),
        gmm_min_std=np.asarray(FLAGS.gmm_min_std, dtype=np.float32),
        gmm_min_std_data_frac=np.asarray(FLAGS.gmm_min_std_data_frac, dtype=np.float32),
        gmm_standardize_data=np.asarray(FLAGS.gmm_standardize_data, dtype=np.int32),
        gmm_transform=np.asarray(gmm_transform),
        channel_mean=np.asarray(channel_mean if channel_mean is not None else np.zeros((int(latent_shape[-1]),), dtype=np.float32), dtype=np.float32),
        channel_whiten=np.asarray(channel_whiten if channel_whiten is not None else np.eye(int(latent_shape[-1]), dtype=np.float32), dtype=np.float32),
        channel_unwhiten=np.asarray(channel_unwhiten if channel_unwhiten is not None else np.eye(int(latent_shape[-1]), dtype=np.float32), dtype=np.float32),
        gmm_init_strategy=np.asarray(FLAGS.gmm_init_strategy),
        gmm_init_mode=np.asarray(fit.get("init_mode", "")),
        gmm_init_warmup_iters=np.asarray(FLAGS.gmm_init_warmup_iters, dtype=np.int32),
        gmm_init_pca_dims=np.asarray(FLAGS.gmm_init_pca_dims, dtype=np.int32),
        gmm_init_pca_max_samples=np.asarray(FLAGS.gmm_init_pca_max_samples, dtype=np.int32),
        gmm_fit_data_mode=np.asarray(gmm_fit_data_mode),
        gmm_mix_x1_prob=np.asarray(gmm_mix_x1_prob, dtype=np.float32),
        gmm_continue_em_iters=np.asarray(gmm_continue_em_iters, dtype=np.int32),
        gmm_mix_seed=np.asarray(FLAGS.gmm_mix_seed, dtype=np.int32),
        gmm_final_fit_stage=np.asarray(final_fit_stage),
        fit_samples=np.asarray(FLAGS.gmm_fit_samples, dtype=np.int32),
        valid_samples=np.asarray(FLAGS.gmm_valid_samples, dtype=np.int32),
    )
    print(f"Saved GMM stats to {FLAGS.gmm_save_path}", flush=True)

    numeric_metrics = {
        k: v for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))
    }
    gmm_wandb_metrics = {f"gmm/{k}": v for k, v in numeric_metrics.items()}

    if FLAGS.metrics_output_path:
        json_dump(FLAGS.metrics_output_path, metrics)
        final_payload = {
            "phase": "gmm_final",
            "step": int(fit["trace"][-1]["iter"]) if fit.get("trace") else int(FLAGS.gmm_em_iters),
            **gmm_wandb_metrics,
        }
        append_metrics_csv(FLAGS.metrics_output_path, final_payload)
        print(f"Saved GMM diagnostics to {FLAGS.metrics_output_path}", flush=True)

    if jax.process_index() == 0:
        wandb.log(gmm_wandb_metrics)
        wandb.summary.update(gmm_wandb_metrics)

    print(json.dumps(metrics, indent=2, sort_keys=True, default=json_default), flush=True)

    if not bool(FLAGS.gmm_keep_latent_cache):
        _cleanup_paths(
            train_cache_path,
            valid_cache_path,
            std_cache_path,
            valid_std_cache_path,
            mix_cache_path,
            mix_std_cache_path,
            transform_cache_path,
            valid_transform_cache_path,
            mix_transform_cache_path,
        )


if __name__ == "__main__":
    app.run(main)
