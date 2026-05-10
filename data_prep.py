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
flags.DEFINE_integer("gmm_kmeanspp_init", 1, "Use k-means++ initialization for component means, as 1/0.")
flags.DEFINE_integer("gmm_keep_latent_cache", 0, "Keep latent memmap cache files after fitting, as 1/0.")
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
            },
            **FLAGS.wandb,
        )
        if FLAGS.gmm_em_metrics_output_path:
            os.makedirs(os.path.dirname(FLAGS.gmm_em_metrics_output_path) or ".", exist_ok=True)
            open(FLAGS.gmm_em_metrics_output_path, "w", encoding="utf-8").close()
            clear_metrics_csv(FLAGS.gmm_em_metrics_output_path)

    def em_metrics_callback(row):
        if jax.process_index() != 0:
            return
        payload = {
            "phase": "gmm_em",
            "dataset_name": FLAGS.dataset_name,
            "num_modes": int(FLAGS.gmm_num_modes),
            "em_iters": int(FLAGS.gmm_em_iters),
            "em_restarts": int(FLAGS.gmm_em_restarts),
            "gmm_pi_prior_type": FLAGS.gmm_pi_prior_type,
            "gmm_pi_prior_strength": float(FLAGS.gmm_pi_prior_strength),
            "gmm_var_prior_type": FLAGS.gmm_var_prior_type,
            "gmm_var_prior_strength": float(FLAGS.gmm_var_prior_strength),
            "gmm_var_prior_target_var": float(FLAGS.gmm_var_prior_target_var),
            "gmm_standardize_data": int(FLAGS.gmm_standardize_data),
            **row,
        }
        _append_jsonl(FLAGS.gmm_em_metrics_output_path, payload)
        append_metrics_csv(FLAGS.gmm_em_metrics_output_path, payload)
        print(
            "GMM EM "
            f"restart={payload['restart']} iter={payload['iter']} "
            f"nll={payload['nll']:.6f} "
            f"pi=[{payload['pi_min']:.6f},{payload['pi_max']:.6f}] "
            f"counts=[{payload['count_min']:.1f},{payload['count_max']:.1f}] "
            f"dead={payload['dead_components']}",
            flush=True,
        )

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
    if bool(FLAGS.gmm_standardize_data):
        x_train_gmm = _standardize_to_memmap(latents_mm, mean, std, std_cache_path, FLAGS.gmm_em_chunk_size)
        gmm_mean = mean
        gmm_std = std
        gmm_fit_space = "standardized"
    else:
        x_train_gmm = flat_train
        gmm_mean = np.zeros_like(mean, dtype=np.float32)
        gmm_std = np.ones_like(std, dtype=np.float32)
        gmm_fit_space = "latent"

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
        if bool(FLAGS.gmm_standardize_data):
            x_valid_gmm = _standardize_to_memmap(valid_mm, mean, std, valid_std_cache_path, FLAGS.gmm_em_chunk_size)
        else:
            x_valid_gmm = valid_mm.reshape((valid_mm.shape[0], -1))

    print("Fitting diagonal GMM", flush=True)
    fit = fit_diag_gmm(
        x_train_gmm,
        num_modes=FLAGS.gmm_num_modes,
        em_iters=FLAGS.gmm_em_iters,
        em_restarts=FLAGS.gmm_em_restarts,
        seed=FLAGS.gmm_init_seed,
        pi_prior_type=FLAGS.gmm_pi_prior_type,
        pi_prior_strength=FLAGS.gmm_pi_prior_strength,
        pi_kl_steps=FLAGS.gmm_pi_kl_steps,
        pi_kl_lr=FLAGS.gmm_pi_kl_lr,
        var_prior_type=FLAGS.gmm_var_prior_type,
        var_prior_strength=FLAGS.gmm_var_prior_strength,
        var_prior_target_var=FLAGS.gmm_var_prior_target_var,
        min_std=FLAGS.gmm_min_std,
        min_std_data_frac=FLAGS.gmm_min_std_data_frac,
        data_std=std,
        standardized=bool(FLAGS.gmm_standardize_data),
        chunk_size=FLAGS.gmm_em_chunk_size,
        use_kmeanspp=bool(FLAGS.gmm_kmeanspp_init),
        eps=FLAGS.gmm_standardize_eps,
        em_metrics_callback=em_metrics_callback,
    )

    metrics = gmm_diagnostics(
        x_train_gmm,
        fit["pi"],
        fit["mu"],
        fit["var"],
        fit["var_floor"],
        data_var=data_var,
        x_valid_std=x_valid_gmm,
        chunk_size=FLAGS.gmm_em_chunk_size,
        eps=FLAGS.gmm_standardize_eps,
    )
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
            "gmm_fit_space": gmm_fit_space,
            "gmm_em_metrics_output_path": FLAGS.gmm_em_metrics_output_path,
            "em_restart_traces": fit["restart_traces"],
            "em_best_trace": fit["trace"],
        }
    )

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
        fit_samples=np.asarray(FLAGS.gmm_fit_samples, dtype=np.int32),
        valid_samples=np.asarray(FLAGS.gmm_valid_samples, dtype=np.int32),
    )
    print(f"Saved GMM stats to {FLAGS.gmm_save_path}", flush=True)

    if FLAGS.metrics_output_path:
        json_dump(FLAGS.metrics_output_path, metrics)
        print(f"Saved GMM diagnostics to {FLAGS.metrics_output_path}", flush=True)

    if jax.process_index() == 0:
        numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}
        wandb.log({f"gmm/{k}": v for k, v in numeric_metrics.items()})
        wandb.summary.update({f"gmm/{k}": v for k, v in numeric_metrics.items()})

    print(json.dumps(metrics, indent=2, sort_keys=True, default=json_default), flush=True)

    if not bool(FLAGS.gmm_keep_latent_cache):
        _cleanup_paths(train_cache_path, valid_cache_path, std_cache_path, valid_std_cache_path)


if __name__ == "__main__":
    app.run(main)
