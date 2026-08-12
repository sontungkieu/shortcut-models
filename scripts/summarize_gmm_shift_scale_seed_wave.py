from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalized_status(run_dir: Path) -> tuple[str, str]:
    payload = _read_json(run_dir / "status" / "status_result.json") or {}
    record = payload.get("record") if isinstance(payload.get("record"), dict) else {}
    status = str(record.get("normalized_status") or "UNKNOWN").upper()
    checked_at = str(record.get("checked_at_utc") or payload.get("generated_at_utc") or "")
    return status, checked_at


def _ok_json(path: Path) -> bool:
    payload = _read_json(path)
    return bool(payload and payload.get("ok") is True)


def _gate_cache_path(gate_root: Path, kernel_id: str) -> Path:
    return gate_root / f"{kernel_id.replace('/', '__')}.json"


def build_wave_state(
    submit_report: dict[str, Any],
    resume_grid: dict[str, Any],
    gate_root: Path,
) -> dict[str, Any]:
    parent_rows = submit_report.get("submitted")
    resume_jobs = resume_grid.get("jobs")
    if not isinstance(parent_rows, list) or not isinstance(resume_jobs, list):
        raise ValueError("Submit report and resume grid must contain submitted/jobs lists")
    by_parent = {str(job.get("resume_kernel_ref")): job for job in resume_jobs if isinstance(job, dict)}
    if len(parent_rows) != 15 or not by_parent or len(by_parent) > 15:
        raise ValueError("Expected exactly 15 parent rows and between 1 and 15 exact resume mappings")

    rows: list[dict[str, Any]] = []
    for parent in parent_rows:
        if not isinstance(parent, dict):
            raise ValueError("Parent submit rows must be JSON objects")
        kernel_id = str(parent.get("kernel_id") or "")
        resume_job = by_parent.get(kernel_id)
        run_dir = Path(str(parent.get("run_dir") or ""))
        status, checked_at = _normalized_status(run_dir)
        if resume_job is None:
            rows.append(
                {
                    "allocated_for_resume": False,
                    "audit_ok": False,
                    "candidate_family": parent.get("candidate_family"),
                    "checked_at_utc": checked_at,
                    "expected_artifacts": {},
                    "artifact_exists": {},
                    "gate_cache": "",
                    "gate_inputs_ready": False,
                    "gate_recorded": False,
                    "kernel_id": kernel_id,
                    "owner": parent.get("owner"),
                    "resume_run_name": "",
                    "resume_submit_ready": False,
                    "run_dir": str(run_dir),
                    "status": status,
                    "summary_ok": False,
                    "training_seed": parent.get("training_seed"),
                }
            )
            continue
        gate_spec = resume_job.get("resume_parent_gate")
        if not isinstance(gate_spec, dict):
            raise ValueError(f"Missing resume_parent_gate for {kernel_id}")

        artifact_paths = {
            key: Path(str(gate_spec[key]))
            for key in ("checkpoint", "diagnostic_manifest", "gmm_stats", "router")
            if gate_spec.get(key)
        }
        artifact_exists = {key: path.is_file() for key, path in artifact_paths.items()}
        summary_ok = _ok_json(Path(str(gate_spec.get("summary") or "")))
        audit_ok = _ok_json(Path(str(gate_spec.get("audit") or "")))
        gate_path = Path(str(gate_spec.get("gate") or _gate_cache_path(gate_root, kernel_id)))
        gate_payload = _read_json(gate_path) or {}
        gate_recorded = bool(gate_payload.get("ok") is True and gate_payload.get("kernel_id") == kernel_id)
        terminal_complete = status == "COMPLETE"
        gate_inputs_ready = terminal_complete and all(artifact_exists.values()) and summary_ok and audit_ok
        rows.append(
            {
                "allocated_for_resume": True,
                "audit_ok": audit_ok,
                "candidate_family": parent.get("candidate_family"),
                "checked_at_utc": checked_at,
                "expected_artifacts": {key: str(path) for key, path in artifact_paths.items()},
                "artifact_exists": artifact_exists,
                "gate_cache": str(gate_path),
                "gate_inputs_ready": gate_inputs_ready,
                "gate_recorded": gate_recorded,
                "kernel_id": kernel_id,
                "owner": parent.get("owner"),
                "resume_run_name": resume_job.get("run_name"),
                "resume_submit_ready": gate_inputs_ready and gate_recorded,
                "run_dir": str(run_dir),
                "status": status,
                "summary_ok": summary_ok,
                "training_seed": parent.get("training_seed"),
            }
        )

    status_counts = Counter(str(row["status"]) for row in rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "rows": rows,
        "summary": {
            "allocated_for_resume": sum(bool(row["allocated_for_resume"]) for row in rows),
            "excluded_from_resume": sum(not bool(row["allocated_for_resume"]) for row in rows),
            "gate_inputs_ready": sum(bool(row["gate_inputs_ready"]) for row in rows),
            "gate_recorded": sum(bool(row["gate_recorded"]) for row in rows),
            "parents": len(rows),
            "resume_submit_ready": sum(bool(row["resume_submit_ready"]) for row in rows),
            "status_counts": dict(sorted(status_counts.items())),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# GMM Shift-Scale Training-Seed Wave State",
        "",
        f"- Generated UTC: {payload['generated_at_utc']}",
        f"- Parent status counts: `{json.dumps(summary['status_counts'], sort_keys=True)}`",
        f"- Allocated for resume: {summary['allocated_for_resume']}/{summary['parents']}",
        f"- Excluded from resume: {summary['excluded_from_resume']}/{summary['parents']}",
        f"- Gate inputs ready: {summary['gate_inputs_ready']}/{summary['allocated_for_resume']}",
        f"- Recorded parent gates: {summary['gate_recorded']}/{summary['allocated_for_resume']}",
        f"- Resume submit ready: {summary['resume_submit_ready']}/{summary['allocated_for_resume']}",
        "",
        "| owner | family | seed | parent status | allocated | artifacts | summary | audit | gate | resume ready | kernel |",
        "|---|---|---:|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        artifact_count = sum(bool(value) for value in row["artifact_exists"].values())
        artifact_total = len(row["artifact_exists"])
        lines.append(
            f"| {row['owner']} | {row['candidate_family']} | {row['training_seed']} | {row['status']} | "
            f"{row['allocated_for_resume']} | "
            f"{artifact_count}/{artifact_total} | {row['summary_ok']} | {row['audit_ok']} | "
            f"{row['gate_recorded']} | {row['resume_submit_ready']} | `{row['kernel_id']}` |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize exact parent and resume-gate state for the seed wave.")
    parser.add_argument(
        "--submit-report",
        type=Path,
        default=Path("reports/gmm_shift_scale_training_seed15_parent200_submit_20260813.json"),
    )
    parser.add_argument(
        "--resume-grid",
        type=Path,
        default=Path("configs/gmm_shift_scale_training_seed15_resume400_grid.json"),
    )
    parser.add_argument(
        "--gate-root",
        type=Path,
        default=Path("outputs/kaggle_jobs/parent_resume_gates"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/gmm_shift_scale_training_seed15_wave_state.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/gmm_shift_scale_training_seed15_wave_state.md"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    submit_report = _read_json(args.submit_report)
    resume_grid = _read_json(args.resume_grid)
    if submit_report is None or resume_grid is None:
        raise ValueError("Could not load submit report or resume grid")
    payload = build_wave_state(submit_report, resume_grid, args.gate_root)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
