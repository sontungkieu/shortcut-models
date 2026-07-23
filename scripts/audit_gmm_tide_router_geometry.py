from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# The Kaggle notebook embeds this script under /tmp but executes it from the
# checked-out repository. Keep repo-local imports resolvable in that mode.
REPO_ROOT = Path(os.environ.get("GMM_TIDE_REPO_ROOT", Path.cwd())).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.targets_gmm_tide import make_tide_source
from gmm_router import load_router_state
from gmm_utils import flatten_latents, load_gmm_stats, posterior_from_stats, sample_prior_components
from utils.datasets import get_dataset
from utils.stable_vae import StableVAE


EPS = 1e-8


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_batch(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(x, (x.shape[0], -1)).astype(jnp.float32)


def batch_cosine(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    a_flat = flatten_batch(a)
    b_flat = flatten_batch(b)
    numerator = jnp.sum(a_flat * b_flat, axis=-1)
    denominator = jnp.sqrt(jnp.maximum(jnp.sum(a_flat * a_flat, axis=-1), EPS))
    denominator *= jnp.sqrt(jnp.maximum(jnp.sum(b_flat * b_flat, axis=-1), EPS))
    return numerator / jnp.maximum(denominator, EPS)


def normalized_entropy(q: jnp.ndarray) -> jnp.ndarray:
    q_safe = jnp.maximum(q, EPS)
    entropy = -jnp.sum(q_safe * jnp.log(q_safe), axis=-1)
    return entropy / jnp.log(jnp.asarray(q.shape[-1], dtype=jnp.float32))


def router_posteriors(gmm_state, router_state, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    q_gmm, _, _ = posterior_from_stats(gmm_state, flatten_latents(x))
    logits = router_state["model_def"].apply({"params": router_state["params"]}, x, train=False)
    return q_gmm, jax.nn.softmax(logits, axis=-1)


def posterior_metrics(prefix: str, q_gmm: jnp.ndarray, q_phi: jnp.ndarray) -> dict[str, jnp.ndarray]:
    q_gmm_safe = jnp.maximum(q_gmm, EPS)
    q_phi_safe = jnp.maximum(q_phi, EPS)
    return {
        f"{prefix}/kl_gmm_to_phi": jnp.mean(
            jnp.sum(q_gmm_safe * (jnp.log(q_gmm_safe) - jnp.log(q_phi_safe)), axis=-1)
        ),
        f"{prefix}/brier": jnp.mean(jnp.sum(jnp.square(q_gmm - q_phi), axis=-1)),
        f"{prefix}/top1_agreement": jnp.mean(
            jnp.argmax(q_gmm, axis=-1) == jnp.argmax(q_phi, axis=-1)
        ),
        f"{prefix}/gmm_top1_prob": jnp.mean(jnp.max(q_gmm, axis=-1)),
        f"{prefix}/phi_top1_prob": jnp.mean(jnp.max(q_phi, axis=-1)),
        f"{prefix}/gmm_entropy_normalized": jnp.mean(normalized_entropy(q_gmm)),
        f"{prefix}/phi_entropy_normalized": jnp.mean(normalized_entropy(q_phi)),
    }


def js_divergence(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    p_safe = jnp.maximum(p, EPS)
    q_safe = jnp.maximum(q, EPS)
    middle = jnp.maximum(0.5 * (p_safe + q_safe), EPS)
    return 0.5 * jnp.sum(p_safe * (jnp.log(p_safe) - jnp.log(middle)), axis=-1) + 0.5 * jnp.sum(
        q_safe * (jnp.log(q_safe) - jnp.log(middle)), axis=-1
    )


def topk_jaccard(p: jnp.ndarray, q: jnp.ndarray, topk: int) -> jnp.ndarray:
    topk = min(int(topk), int(p.shape[-1]))
    p_ids = jax.lax.top_k(p, topk)[1]
    q_ids = jax.lax.top_k(q, topk)[1]
    p_hot = jnp.sum(jax.nn.one_hot(p_ids, p.shape[-1]), axis=1) > 0
    q_hot = jnp.sum(jax.nn.one_hot(q_ids, q.shape[-1]), axis=1) > 0
    intersection = jnp.sum(p_hot & q_hot, axis=-1)
    union = jnp.sum(p_hot | q_hot, axis=-1)
    return intersection / jnp.maximum(union, 1)


def stability_metrics(
    prefix: str,
    q_clean: jnp.ndarray,
    q_perturbed: jnp.ndarray,
    x_clean: jnp.ndarray,
    x_perturbed: jnp.ndarray,
    topk: int,
) -> dict[str, jnp.ndarray]:
    input_delta = jnp.sqrt(jnp.mean(jnp.square(x_perturbed - x_clean), axis=tuple(range(1, x_clean.ndim))))
    output_delta = jnp.sqrt(jnp.mean(jnp.square(q_perturbed - q_clean), axis=-1))
    return {
        f"{prefix}/js_divergence": jnp.mean(js_divergence(q_clean, q_perturbed)),
        f"{prefix}/top1_stability": jnp.mean(
            jnp.argmax(q_clean, axis=-1) == jnp.argmax(q_perturbed, axis=-1)
        ),
        f"{prefix}/topk_jaccard": jnp.mean(topk_jaccard(q_clean, q_perturbed, topk)),
        f"{prefix}/input_rms_delta": jnp.mean(input_delta),
        f"{prefix}/output_rms_delta": jnp.mean(output_delta),
        f"{prefix}/local_lipschitz_proxy": jnp.mean(output_delta / jnp.maximum(input_delta, EPS)),
    }


def channel_covariance_metrics(prefix: str, x: np.ndarray) -> dict[str, float]:
    flat = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
    flat -= flat.mean(axis=0, keepdims=True)
    covariance = flat.T @ flat / max(flat.shape[0] - 1, 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = float(eigenvalues.sum())
    probs = eigenvalues / max(total, EPS)
    entropy = float(-(probs * np.log(np.maximum(probs, EPS))).sum())
    return {
        f"{prefix}/channel_cov_trace": total,
        f"{prefix}/channel_cov_condition": float(eigenvalues.max() / max(eigenvalues.min(), EPS)),
        f"{prefix}/channel_cov_effective_rank": float(math.exp(entropy)),
    }


def scalarize(metrics: dict[str, Any]) -> dict[str, float]:
    result = {}
    for name, value in metrics.items():
        array = np.asarray(jax.device_get(value))
        if array.shape == () and np.isfinite(array):
            result[name] = float(array)
    return result


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    names = sorted({name for row in rows for name in row if name != "batch"})
    summary = {}
    for name in names:
        values = np.asarray([row[name] for row in rows if name in row], dtype=np.float64)
        if values.size == 0:
            continue
        summary[name] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="celebahq256")
    parser.add_argument("--tfds-data-dir", default="/root/tensorflow_datasets")
    parser.add_argument("--gmm-stats-path", type=Path, required=True)
    parser.add_argument("--router-path", type=Path, required=True)
    parser.add_argument("--source-mode", choices=("weighted", "sample_topk", "hard_top1"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bridge-lambdas", type=parse_float_list, default=parse_float_list("0,0.25,0.5,0.75,1"))
    parser.add_argument("--noise-scales", type=parse_float_list, default=parse_float_list("0.01,0.05,0.1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_batches <= 0 or args.batch_size <= 0 or args.topk <= 0:
        raise ValueError("num_batches, batch_size, and topk must be positive")
    if any(value < 0.0 or value > 1.0 for value in args.bridge_lambdas):
        raise ValueError("bridge lambdas must lie in [0, 1]")
    if any(value <= 0.0 for value in args.noise_scales):
        raise ValueError("noise scales must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "audit_metrics.jsonl"
    csv_path = args.output_dir / "audit_metrics.csv"
    summary_path = args.output_dir / "audit_summary.json"

    gmm_state = load_gmm_stats(str(args.gmm_stats_path))
    router_state = load_router_state(str(args.router_path))
    dataset = get_dataset(args.dataset_name, args.batch_size, False, 0, data_dir=args.tfds_data_dir)
    vae_encode = None
    if "latent" not in args.dataset_name:
        vae_encode = jax.jit(StableVAE.create().encode)

    rng = jax.random.PRNGKey(args.seed)
    rows: list[dict[str, float]] = []
    for batch_index in range(args.num_batches):
        images, _ = next(dataset)
        rng, encode_key, prior_key, source_key = jax.random.split(rng, 4)
        x1 = jnp.asarray(images, dtype=jnp.float32) if vae_encode is None else vae_encode(encode_key, images)
        x0_base, _, _, _ = sample_prior_components(prior_key, gmm_state, x1.shape[0], x1.shape[1:])

        metrics: dict[str, Any] = {"batch": float(batch_index)}
        q_gmm_x1, q_phi_x1 = router_posteriors(gmm_state, router_state, x1)
        q_gmm_x0, q_phi_x0 = router_posteriors(gmm_state, router_state, x0_base)
        metrics.update(posterior_metrics("router/x1", q_gmm_x1, q_phi_x1))
        metrics.update(posterior_metrics("router/x0", q_gmm_x0, q_phi_x0))

        for lam in args.bridge_lambdas:
            x_bridge = lam * x1 + (1.0 - lam) * x0_base
            q_gmm_bridge, q_phi_bridge = router_posteriors(gmm_state, router_state, x_bridge)
            tag = str(lam).replace(".", "p")
            metrics.update(posterior_metrics(f"router/bridge_l{tag}", q_gmm_bridge, q_phi_bridge))

        x1_rms = jnp.sqrt(jnp.mean(jnp.square(x1), axis=tuple(range(1, x1.ndim)), keepdims=True))
        for noise_scale in args.noise_scales:
            rng, noise_key = jax.random.split(rng)
            x1_noisy = x1 + noise_scale * x1_rms * jax.random.normal(noise_key, x1.shape, dtype=x1.dtype)
            _, q_phi_noisy = router_posteriors(gmm_state, router_state, x1_noisy)
            tag = str(noise_scale).replace(".", "p")
            metrics.update(
                stability_metrics(
                    f"router/noise_s{tag}",
                    q_phi_x1,
                    q_phi_noisy,
                    x1,
                    x1_noisy,
                    args.topk,
                )
            )

        x0_tide, _, _, tide_info = make_tide_source(
            source_key,
            gmm_state,
            router_state,
            x1.shape[0],
            x1.shape[1:],
            topk=args.topk,
            temperature=args.temperature,
            source_mode=args.source_mode,
        )
        velocity = x1 - x0_tide
        x0_x1_cosine = batch_cosine(x0_tide, x1)
        velocity_x1_cosine = batch_cosine(velocity, x1)
        metrics.update(tide_info)
        metrics.update(
            {
                "source/x1_magnitude": jnp.sqrt(jnp.mean(jnp.square(x1))),
                "source/x0_tide_magnitude": jnp.sqrt(jnp.mean(jnp.square(x0_tide))),
                "source/x0_x1_magnitude_ratio": jnp.sqrt(jnp.mean(jnp.square(x0_tide)))
                / jnp.maximum(jnp.sqrt(jnp.mean(jnp.square(x1))), EPS),
                "source/x0_x1_cosine_mean": jnp.mean(x0_x1_cosine),
                "source/x0_x1_cosine_std": jnp.std(x0_x1_cosine),
                "source/velocity_x1_cosine_mean": jnp.mean(velocity_x1_cosine),
                "source/velocity_magnitude": jnp.sqrt(jnp.mean(jnp.square(velocity))),
            }
        )
        scalar_row = scalarize(metrics)
        scalar_row.update(channel_covariance_metrics("source/x1", np.asarray(jax.device_get(x1))))
        scalar_row.update(channel_covariance_metrics("source/x0_tide", np.asarray(jax.device_get(x0_tide))))
        rows.append(scalar_row)
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(scalar_row, sort_keys=True) + "\n")
        print(f"AUDIT_BATCH {batch_index + 1}/{args.num_batches}", flush=True)

    fieldnames = sorted({name for row in rows for name in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "config": {
            "dataset_name": args.dataset_name,
            "batch_size": args.batch_size,
            "num_batches": args.num_batches,
            "seed": args.seed,
            "source_mode": args.source_mode,
            "topk": args.topk,
            "temperature": args.temperature,
            "bridge_lambdas": args.bridge_lambdas,
            "noise_scales": args.noise_scales,
        },
        "artifacts": {
            "gmm_stats_sha256": file_sha256(args.gmm_stats_path),
            "router_sha256": file_sha256(args.router_path),
        },
        "summary": summarize_rows(rows),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("AUDIT_SUMMARY " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
