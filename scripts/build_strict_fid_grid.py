from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_checkpoint_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("checkpoints", payload)
    if not isinstance(rows, list):
        raise ValueError("Checkpoint manifest must be a list or contain a checkpoints list")
    result = {}
    for row in rows:
        run_name = str(row["run_name"])
        kernel_ref = str(row["kernel_ref"])
        if "/" not in kernel_ref:
            raise ValueError(f"Invalid kernel_ref for {run_name}: {kernel_ref!r}")
        if run_name in result:
            raise ValueError(f"Duplicate checkpoint run_name: {run_name}")
        result[run_name] = kernel_ref
    return result


def build_fid_grid(
    training_grid_path: Path,
    checkpoint_manifest_path: Path,
    eval_fid_seeds: str,
    eval_fid_generations: int,
) -> dict:
    training_grid = json.loads(training_grid_path.read_text(encoding="utf-8"))
    checkpoints = load_checkpoint_map(checkpoint_manifest_path)
    defaults = dict(training_grid["defaults"])
    defaults.update(
        {
            "execution_mode": "fid_repeats",
            "eval_fid_generations": int(eval_fid_generations),
            "eval_fid_seeds": eval_fid_seeds,
            "eval_fid_timesteps": "128",
            "resume_require_checkpoint": True,
            "resume_reuse_gmm_router": True,
            "train_max_steps": 0,
        }
    )
    defaults.pop("resume_kernel_ref", None)
    defaults.pop("resume_run_name", None)

    jobs = []
    missing = []
    for training_job in training_grid["jobs"]:
        source_run_name = str(training_job["run_name"])
        kernel_ref = checkpoints.get(source_run_name)
        if not kernel_ref:
            missing.append(source_run_name)
            continue
        job = dict(training_job)
        job.update(
            {
                "resume_kernel_ref": kernel_ref,
                "resume_run_name": source_run_name,
                "run_name": f"fid-{source_run_name}",
                "source_run_name": source_run_name,
            }
        )
        jobs.append(job)
    if missing:
        raise ValueError(f"Checkpoint manifest is missing training runs: {missing}")
    return {"defaults": defaults, "jobs": jobs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build repeated-FID jobs for every checkpoint in a strict paired training grid."
    )
    parser.add_argument("--training-grid", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--eval-fid-seeds", default="101,202,303,404,505")
    parser.add_argument("--eval-fid-generations", type=int, default=50048)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/gmm_tide_moe2_strict_fid_grid.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_fid_generations <= 0:
        raise ValueError("--eval-fid-generations must be positive")
    payload = build_fid_grid(
        args.training_grid,
        args.checkpoint_manifest,
        args.eval_fid_seeds,
        args.eval_fid_generations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
