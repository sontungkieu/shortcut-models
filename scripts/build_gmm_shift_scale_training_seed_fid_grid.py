from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path
from typing import Any


FAMILY_ORDER = {"naive_gaussian": 0, "top2_c01": 1, "top4_c02": 2}
FAMILY_RUN_TOKEN = {
    "naive_gaussian": "naive",
    "top2_c01": "t2c01",
    "top4_c02": "t4c02",
}
EVAL_FID_SEEDS = "101,202,303,404,505"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _artifact_gate(
    *,
    run_dir: str,
    run_name: str,
    candidate_family: str,
) -> dict[str, Any]:
    output = Path(run_dir) / "output"
    gmm_root = output / "gmm_tide_fm" / run_name
    gate: dict[str, Any] = {
        "audit": str(Path(run_dir) / "reports" / "audit_run_dir.json"),
        "checkpoint": str(output / "ckpts" / f"{run_name}.pkl"),
        "diagnostic_manifest": str(gmm_root / "diagnostics" / "train_metrics_summary.json"),
        "summary": str(Path(run_dir) / "reports" / "summary.json"),
        "terminal_status": "COMPLETE",
    }
    if candidate_family != "naive_gaussian":
        gate["gmm_stats"] = str(gmm_root / "gmm_stats.npz")
        gate["router"] = str(gmm_root / "gmm_router.pkl")
    return gate


def _accepted_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("submitted")
    if not isinstance(rows, list):
        raise ValueError("resume submit report must contain a submitted list")
    accepted: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    seen_kernels: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("submitted rows must be JSON objects")
        run_name = str(raw.get("run_name") or "")
        kernel_id = str(raw.get("kernel_id") or "")
        owner = str(raw.get("owner") or "")
        run_dir = str(raw.get("run_dir") or "")
        if not run_name or not kernel_id or not owner or not run_dir:
            raise ValueError(f"incomplete accepted resume identity: {raw!r}")
        if kernel_id.split("/", 1)[0] != owner:
            raise ValueError(f"owner/kernel mismatch for {kernel_id}")
        if run_name in seen_runs or kernel_id in seen_kernels:
            raise ValueError(f"duplicate accepted resume identity: {run_name} / {kernel_id}")
        seen_runs.add(run_name)
        seen_kernels.add(kernel_id)
        accepted.append(raw)
    return accepted


def build_fid_grid(
    resume_grid: dict[str, Any],
    resume_submit_report: dict[str, Any],
) -> dict[str, Any]:
    defaults_raw = resume_grid.get("defaults")
    jobs_raw = resume_grid.get("jobs")
    if not isinstance(defaults_raw, dict) or not isinstance(jobs_raw, list):
        raise ValueError("resume grid must contain defaults and jobs")
    source_jobs = {
        str(job.get("run_name")): job
        for job in jobs_raw
        if isinstance(job, dict) and job.get("run_name")
    }
    if len(source_jobs) != len(jobs_raw):
        raise ValueError("resume grid run_name values must be present and unique")

    defaults = copy.deepcopy(defaults_raw)
    defaults.update(
        {
            "ablation_family": "gmm_shift_scale_training_seed_fidrepeat5_400k",
            "comparison_protocol": (
                "fixed audited 400k checkpoint; paired evaluation seeds; "
                "50048 generations per seed; FID128 only"
            ),
            "evaluation_protocol_id": "gmm-shift-scale-training-seed-replication-v1-wave3",
            "eval_fid_generations": 50048,
            "eval_fid_seeds": EVAL_FID_SEEDS,
            "eval_fid_timesteps": "128",
            "execution_mode": "fid_repeats",
            "resume_attach_kernel_source": True,
            "resume_checkpoint_step": 400000,
            "resume_download_output": False,
            "resume_expected_checkpoint_step": 400000,
            "resume_require_checkpoint": True,
            "train_max_steps": 0,
            "train_resume_start_step": 400000,
            "train_target_step_abs": 400000,
        }
    )
    defaults.pop("training_seed", None)

    jobs: list[dict[str, Any]] = []
    for row in _accepted_rows(resume_submit_report):
        source_run_name = str(row["run_name"])
        source = source_jobs.get(source_run_name)
        if source is None:
            raise ValueError(f"accepted run is absent from frozen resume grid: {source_run_name}")
        candidate_family = str(source.get("candidate_family") or "")
        training_seed = int(source.get("training_seed"))
        if candidate_family not in FAMILY_ORDER or training_seed not in {1, 2, 3, 4}:
            raise ValueError(
                f"out-of-protocol resume child: family={candidate_family}, seed={training_seed}"
            )

        job = {
            key: copy.deepcopy(value)
            for key, value in source.items()
            if key
            not in {
                "artifact_source_role",
                "expected_submit_owner",
                "resume_attach_kernel_source",
                "resume_download_output",
                "resume_kernel_ref",
                "resume_parent_gate",
                "resume_run_name",
                "run_name",
                "source_run_name",
            }
        }
        owner = str(row["owner"])
        run_dir = str(row["run_dir"])
        job.update(
            {
                "artifact_source_role": (
                    f"exact_{candidate_family}_training_seed{training_seed}_400k_child"
                ),
                "expected_submit_owner": owner,
                "resume_attach_kernel_source": True,
                "resume_download_output": False,
                "resume_kernel_ref": str(row["kernel_id"]),
                "resume_parent_gate": _artifact_gate(
                    run_dir=run_dir,
                    run_name=source_run_name,
                    candidate_family=candidate_family,
                ),
                "resume_run_name": source_run_name,
                "resume_reuse_gmm_router": candidate_family != "naive_gaussian",
                "run_name": (
                    f"fidrep-{FAMILY_RUN_TOKEN[candidate_family]}-s{training_seed}-400k"
                ),
                "source_run_name": source_run_name,
                "training_seed": training_seed,
            }
        )
        jobs.append(job)

    jobs.sort(key=lambda job: (FAMILY_ORDER[str(job["candidate_family"])], int(job["training_seed"])))
    identities = {(job["candidate_family"], job["training_seed"]) for job in jobs}
    if len(identities) != len(jobs) or len(jobs) > 12:
        raise ValueError("FID grid must contain at most one job for each of the frozen 12 children")
    return {
        "defaults": defaults,
        "jobs": jobs,
        "provenance": {
            "accepted_resume_count": len(jobs),
            "build_policy": "metric-blind transform of accepted frozen wave-2 children",
            "evaluation_seeds": [101, 202, 303, 404, 505],
            "expected_complete_count": 12,
            "generations_per_seed": 50048,
            "primary_checkpoint_step": 400000,
            "timesteps": [128],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the fixed-400k five-seed FID grid from accepted wave-2 resume jobs."
    )
    parser.add_argument(
        "--resume-grid",
        type=Path,
        default=Path("configs/gmm_shift_scale_training_seed15_resume400_grid.json"),
    )
    parser.add_argument(
        "--resume-submit-report",
        type=Path,
        default=Path("reports/gmm_shift_scale_training_seed12_resume400_submit_20260813.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("configs/gmm_shift_scale_training_seed_fidrepeat5_400k_grid.json"),
    )
    parser.add_argument("--expected-accepted-count", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_fid_grid(_read_json(args.resume_grid), _read_json(args.resume_submit_report))
    accepted = len(payload["jobs"])
    if args.expected_accepted_count > 0 and accepted != args.expected_accepted_count:
        raise SystemExit(
            f"expected {args.expected_accepted_count} accepted resume jobs, found {accepted}"
        )
    _atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "accepted_resume_children": accepted,
                "fid_jobs": accepted,
                "output": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
