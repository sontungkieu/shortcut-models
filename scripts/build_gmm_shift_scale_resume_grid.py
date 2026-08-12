from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_FAMILIES = {"naive_gaussian", "top2_c01", "top4_c02"}
EXPECTED_SEEDS = {1, 2, 3, 4, 5}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_parent_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 15:
        raise ValueError(f"Expected exactly 15 accepted parent rows, got {len(rows)}")
    family_counts = Counter(str(row.get("candidate_family") or "") for row in rows)
    expected_counts = {family: 5 for family in EXPECTED_FAMILIES}
    if dict(family_counts) != expected_counts:
        raise ValueError(f"Expected five rows per family, got {dict(family_counts)}")
    identities = set()
    owners = set()
    kernel_ids = set()
    for row in rows:
        family = str(row.get("candidate_family") or "")
        seed = int(row.get("training_seed"))
        owner = str(row.get("owner") or "")
        kernel_id = str(row.get("kernel_id") or "")
        if seed not in EXPECTED_SEEDS:
            raise ValueError(f"Unexpected training seed {seed} for {kernel_id}")
        if not owner or not kernel_id.startswith(f"{owner}/"):
            raise ValueError(f"Owner/kernel mismatch: owner={owner!r}, kernel={kernel_id!r}")
        identities.add((family, seed))
        owners.add(owner)
        kernel_ids.add(kernel_id)
    expected_identities = {(family, seed) for family in EXPECTED_FAMILIES for seed in EXPECTED_SEEDS}
    if identities != expected_identities:
        raise ValueError("Parent family/seed matrix is incomplete or duplicated")
    if len(owners) != 15 or len(kernel_ids) != 15:
        raise ValueError("Each parent must have a unique owner and exact kernel id")


def _parent_gate_spec(row: dict[str, Any], parent_run_name: str) -> dict[str, Any]:
    run_dir = Path(str(row["run_dir"]))
    output_base = run_dir / "output" / "gmm_tide_fm" / parent_run_name
    reports_dir = run_dir / "reports"
    spec: dict[str, Any] = {
        "terminal_status": "COMPLETE",
        "checkpoint": str(run_dir / "output" / "ckpts" / f"{parent_run_name}.pkl"),
        "diagnostic_manifest": str(output_base / "diagnostics" / "train_metrics_summary.json"),
        "summary": str(reports_dir / "summary.json"),
        "audit": str(reports_dir / "audit_run_dir.json"),
    }
    if str(row["candidate_family"]) != "naive_gaussian":
        spec["gmm_stats"] = str(output_base / "gmm_stats.npz")
        spec["router"] = str(output_base / "gmm_router.pkl")
    return spec


def build_resume_grid(
    parent_grid: dict[str, Any],
    submit_report: dict[str, Any],
    allocation_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = submit_report.get("submitted")
    if not isinstance(rows, list):
        raise ValueError("Submit report is missing submitted rows")
    rows = [dict(row) for row in rows]
    if submit_report.get("failed") or submit_report.get("not_submitted"):
        raise ValueError("Parent submit report contains failed or not-submitted rows")
    _validate_parent_rows(rows)

    parent_jobs = parent_grid.get("jobs")
    defaults = parent_grid.get("defaults")
    if not isinstance(parent_jobs, list) or not isinstance(defaults, dict):
        raise ValueError("Parent grid must contain defaults and jobs")
    by_index = {int(row["grid_index"]): row for row in rows}
    if set(by_index) != set(range(15)) or len(parent_jobs) != 15:
        raise ValueError("Parent grid/report indices must be exactly 0..14")

    resume_defaults = dict(defaults)
    resume_defaults.update(
        {
            "ablation_family": "gmm_shift_scale_training_seed15_resume400",
            "comparison_protocol": (
                "exact 200k parent artifact resume to fixed 400k; parent artifact gate required; "
                "same-owner attach or audited cross-account download according to the frozen allocation; "
                "optimizer state reinitialized to match the registered two-stage seed-0 protocol"
            ),
            "resume_attach_kernel_source": True,
            "resume_download_output": False,
            "resume_expected_checkpoint_step": 200000,
            "resume_require_checkpoint": True,
            "save_interval": 200000,
            "save_slim_checkpoint": 1,
            "train_max_steps": 200000,
            "train_resume_start_step": 200000,
            "train_target_step_abs": 400000,
        }
    )

    included_seeds = EXPECTED_SEEDS
    destination_overrides: dict[tuple[str, int], str] = {}
    if allocation_plan is not None:
        raw_included = allocation_plan.get("included_training_seeds")
        if not isinstance(raw_included, list) or not raw_included:
            raise ValueError("Allocation plan must contain included_training_seeds")
        included_seeds = {int(seed) for seed in raw_included}
        if not included_seeds <= EXPECTED_SEEDS:
            raise ValueError(f"Allocation plan contains unexpected seeds: {sorted(included_seeds)}")
        raw_overrides = allocation_plan.get("destination_overrides", [])
        if not isinstance(raw_overrides, list):
            raise ValueError("destination_overrides must be a list")
        for override in raw_overrides:
            if not isinstance(override, dict):
                raise ValueError("destination override rows must be JSON objects")
            identity = (str(override.get("candidate_family") or ""), int(override.get("training_seed")))
            destination_owner = str(override.get("destination_owner") or "")
            if identity in destination_overrides or not destination_owner:
                raise ValueError(f"Invalid or duplicated destination override: {identity}")
            destination_overrides[identity] = destination_owner

    jobs: list[dict[str, Any]] = []
    for grid_index, parent_job in enumerate(parent_jobs):
        row = by_index[grid_index]
        family = str(row["candidate_family"])
        seed = int(row["training_seed"])
        owner = str(row["owner"])
        if str(parent_job.get("candidate_family")) != family:
            raise ValueError(f"Family mismatch at grid index {grid_index}")
        if int(parent_job.get("training_seed")) != seed:
            raise ValueError(f"Training-seed mismatch at grid index {grid_index}")
        if str(parent_job.get("expected_submit_owner")) != owner:
            raise ValueError(f"Owner mismatch at grid index {grid_index}")
        if seed not in included_seeds:
            continue

        parent_run_name = str(row["run_name"])
        if not parent_run_name.endswith("-parent200"):
            raise ValueError(f"Unexpected parent run name: {parent_run_name}")
        resume_run_name = parent_run_name.removesuffix("-parent200") + "-resume400"
        destination_owner = destination_overrides.get((family, seed), owner)
        cross_account_resume = destination_owner != owner
        job = {
            key: parent_job[key]
            for key in (
                "candidate_family",
                "gmm_router_temperature",
                "gmm_router_topk",
                "gmm_source_center_scale",
                "model_train_type",
                "training_seed",
            )
            if key in parent_job
        }
        job.update(
            {
                "artifact_source_role": f"exact_{family}_seed{seed}_200k_parent",
                "expected_submit_owner": destination_owner,
                "resume_attach_kernel_source": not cross_account_resume,
                "resume_download_output": cross_account_resume,
                "resume_kernel_ref": str(row["kernel_id"]),
                "resume_parent_gate": _parent_gate_spec(row, parent_run_name),
                "resume_reuse_gmm_router": family != "naive_gaussian",
                "resume_run_name": parent_run_name,
                "run_name": resume_run_name,
                "source_run_name": parent_run_name,
            }
        )
        jobs.append(job)

    expected_job_count = len(EXPECTED_FAMILIES) * len(included_seeds)
    if len(jobs) != expected_job_count:
        raise ValueError(f"Expected {expected_job_count} allocated resume jobs, got {len(jobs)}")
    destination_owners = [str(job["expected_submit_owner"]) for job in jobs]
    if len(set(destination_owners)) != len(destination_owners):
        raise ValueError("Allocated resume jobs must use unique destination owners in the primary wave")
    unused_overrides = set(destination_overrides) - {
        (str(job["candidate_family"]), int(job["training_seed"])) for job in jobs
    }
    if unused_overrides:
        raise ValueError(f"Destination overrides do not match selected jobs: {sorted(unused_overrides)}")

    return {"defaults": resume_defaults, "jobs": jobs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the exact 200k-to-400k seed-replication resume grid.")
    parser.add_argument(
        "--parent-grid",
        type=Path,
        default=Path("configs/gmm_shift_scale_training_seed15_parent200_grid.json"),
    )
    parser.add_argument(
        "--submit-report",
        type=Path,
        default=Path("reports/gmm_shift_scale_training_seed15_parent200_submit_20260813.json"),
    )
    parser.add_argument(
        "--allocation-plan",
        type=Path,
        default=Path("configs/gmm_shift_scale_training_seed_resume_allocation_20260813.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/gmm_shift_scale_training_seed15_resume400_grid.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_resume_grid(
        _load_json(args.parent_grid),
        _load_json(args.submit_report),
        _load_json(args.allocation_plan),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "jobs": len(payload["jobs"]),
                "owners": len({job["expected_submit_owner"] for job in payload["jobs"]}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
