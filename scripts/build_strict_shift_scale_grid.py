from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_BASE_GRID = Path("configs/gmm_tide_moe2_shift_scale_raw_200k_grid.json")


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one training seed is required")
    if len(values) != len(set(values)):
        raise ValueError("Training seeds must be unique")
    return values


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != len(set(values)):
        raise ValueError("Scale values must be unique")
    if any(item < 0 for item in values):
        raise ValueError("Scale values must be non-negative")
    return values


def scale_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def treatment_rows(scales: list[float]) -> list[dict]:
    rows = [
        {
            "candidate_family": "strict_no_shift",
            "gmm_source_shift_mean": 0,
            "gmm_source_center_scale": 1.0,
            "treatment_slug": "noshift",
        },
        {
            "candidate_family": "strict_shift_c1",
            "gmm_source_shift_mean": 1,
            "gmm_source_center_scale": 1.0,
            "treatment_slug": "shift-c1",
        },
    ]
    rows.extend(
        {
            "candidate_family": f"strict_shift_scale_c{scale:g}",
            "gmm_source_shift_mean": 1,
            "gmm_source_center_scale": scale,
            "treatment_slug": f"shift-c{scale_slug(scale)}",
        }
        for scale in scales
        if scale != 1.0
    )
    return rows


def build_grid(
    *,
    base_grid: Path,
    artifact_ref: str,
    artifact_run_name: str,
    artifact_block: str,
    artifact_seed: int,
    training_seeds: list[int],
    scales: list[float],
    dataset_seed: int,
    vae_seed: int,
    eval_fid_seeds: str,
    expected_gmm_hash: str = "",
    expected_router_hash: str = "",
) -> dict:
    owner, separator, slug = artifact_ref.partition("/")
    if not separator or not owner or not slug:
        raise ValueError("--artifact-ref must use owner/kernel-slug form")
    base_payload = json.loads(base_grid.read_text(encoding="utf-8"))
    defaults = dict(base_payload["defaults"])
    defaults.update(
        {
            "ablation_family": "moe2_strict_shift_scale",
            "branch": "moe2-strict-ablation",
            "dataset_seed": int(dataset_seed),
            "eval_fid_seeds": eval_fid_seeds,
            "eval_fid_timesteps": "128",
            "execution_mode": "train",
            "gmm_randomization_seed": int(artifact_seed),
            "resume_download_output": True,
            "resume_kernel_ref": artifact_ref,
            "resume_require_checkpoint": False,
            "resume_reuse_gmm_router": True,
            "resume_run_name": artifact_run_name,
            "source_run_name": artifact_run_name,
            "strict_ablation": 1,
            "strict_artifact_block": artifact_block,
            "strict_deterministic_data": 1,
            "vae_seed": int(vae_seed),
        }
    )
    if expected_gmm_hash:
        defaults["strict_expected_gmm_content_sha256"] = expected_gmm_hash
    if expected_router_hash:
        defaults["strict_expected_router_content_sha256"] = expected_router_hash

    jobs = []
    for training_seed in training_seeds:
        for treatment in treatment_rows(scales):
            row = {
                key: value
                for key, value in treatment.items()
                if key != "treatment_slug"
            }
            row.update(
                {
                    "run_name": (
                        f"tide-strict-{artifact_block}-s{training_seed}-"
                        f"{treatment['treatment_slug']}-200k"
                    ),
                    "training_seed": training_seed,
                }
            )
            jobs.append(row)
    return {"defaults": defaults, "jobs": jobs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a paired strict shift/scale grid. Every treatment at one "
            "training seed reuses the same canonical GMM/router pair."
        )
    )
    parser.add_argument("--artifact-ref", required=True, help="owner/kernel-slug")
    parser.add_argument("--artifact-run-name", required=True)
    parser.add_argument("--artifact-block", required=True)
    parser.add_argument("--artifact-seed", type=int, required=True)
    parser.add_argument("--training-seeds", default="0,1,2")
    parser.add_argument("--scales", default="0.75,0.875,1.125,1.25")
    parser.add_argument("--dataset-seed", type=int, default=52000)
    parser.add_argument("--vae-seed", type=int, default=62000)
    parser.add_argument("--eval-fid-seeds", default="101,202,303,404,505")
    parser.add_argument("--expected-gmm-hash", default="")
    parser.add_argument("--expected-router-hash", default="")
    parser.add_argument("--base-grid", type=Path, default=DEFAULT_BASE_GRID)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/gmm_tide_moe2_strict_shift_scale_grid.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_grid(
        base_grid=args.base_grid,
        artifact_ref=args.artifact_ref,
        artifact_run_name=args.artifact_run_name,
        artifact_block=args.artifact_block,
        artifact_seed=args.artifact_seed,
        training_seeds=parse_int_list(args.training_seeds),
        scales=parse_float_list(args.scales),
        dataset_seed=args.dataset_seed,
        vae_seed=args.vae_seed,
        eval_fid_seeds=args.eval_fid_seeds,
        expected_gmm_hash=args.expected_gmm_hash,
        expected_router_hash=args.expected_router_hash,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
