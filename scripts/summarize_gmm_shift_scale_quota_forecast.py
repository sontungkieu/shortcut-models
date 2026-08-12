from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _current_parent_elapsed_s(entry: dict[str, Any]) -> float | None:
    elapsed = []
    for record in entry.get("usage", {}).get("records", []):
        if not isinstance(record, dict):
            continue
        if record.get("source") != "running_status_estimate":
            continue
        if "parent200" not in str(record.get("kernel_id") or ""):
            continue
        try:
            elapsed.append(float(record["seconds"]))
        except (KeyError, TypeError, ValueError):
            continue
    return max(elapsed) if elapsed else None


def build_quota_forecast(
    quota_report: dict[str, Any],
    resume_grid: dict[str, Any],
    allocation_plan: dict[str, Any],
    *,
    fid_runtime_s: float,
) -> dict[str, Any]:
    entries = quota_report.get("entries")
    jobs = resume_grid.get("jobs")
    runtime = allocation_plan.get("runtime_estimate")
    if not isinstance(entries, list) or not isinstance(jobs, list) or not isinstance(runtime, dict):
        raise ValueError("Quota report, resume grid, or allocation runtime estimate is missing")

    parent_runtime_s = float(runtime["historical_max_seconds"])
    resume_runtime_s = float(runtime["historical_max_seconds"])
    by_owner = {
        str(entry.get("owner") or ""): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("owner")
    }
    rows: list[dict[str, Any]] = []
    seen_owners: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("Resume jobs must be JSON objects")
        owner = str(job.get("expected_submit_owner") or "")
        if not owner or owner in seen_owners:
            raise ValueError(f"Resume destination owner is missing or duplicated: {owner!r}")
        seen_owners.add(owner)
        entry = by_owner.get(owner)
        if entry is None:
            raise ValueError(f"Quota report is missing destination owner {owner}")

        current_parent_elapsed_s = _current_parent_elapsed_s(entry)
        weekly_remaining_s = float(entry["weekly_remaining_s"])
        conservative_elapsed_s = current_parent_elapsed_s or 0.0
        projected_parent_residual_s = max(0.0, parent_runtime_s - conservative_elapsed_s)
        projected_remaining_after_parent_s = weekly_remaining_s - projected_parent_residual_s
        margin_after_resume_s = projected_remaining_after_parent_s - resume_runtime_s
        margin_after_resume_and_fid_s = margin_after_resume_s - fid_runtime_s
        rows.append(
            {
                "accounting_confidence": entry.get("accounting_confidence"),
                "candidate_family": job.get("candidate_family"),
                "current_parent_elapsed_s": current_parent_elapsed_s,
                "destination_owner": owner,
                "fid_runtime_s": fid_runtime_s,
                "margin_after_resume_and_fid_s": margin_after_resume_and_fid_s,
                "margin_after_resume_s": margin_after_resume_s,
                "parent_runtime_upper_s": parent_runtime_s,
                "projected_parent_residual_s": projected_parent_residual_s,
                "projected_remaining_after_parent_s": projected_remaining_after_parent_s,
                "resume_runtime_upper_s": resume_runtime_s,
                "run_name": job.get("run_name"),
                "safe_for_resume": margin_after_resume_s >= 0.0,
                "safe_for_resume_and_fid": margin_after_resume_and_fid_s >= 0.0,
                "training_seed": job.get("training_seed"),
                "untracked_usage_possible": bool(entry.get("untracked_usage_possible")),
                "weekly_remaining_now_s": weekly_remaining_s,
            }
        )

    return {
        "accounting_scope": "local_registry_estimate_conservative_projection",
        "generated_at_utc": quota_report.get("generated_at_utc"),
        "quota_report_kjo_version": quota_report.get("kjo_version"),
        "rows": rows,
        "summary": {
            "destinations": len(rows),
            "minimum_margin_after_resume_and_fid_s": min(
                row["margin_after_resume_and_fid_s"] for row in rows
            ),
            "minimum_margin_after_resume_s": min(row["margin_after_resume_s"] for row in rows),
            "resume_and_fid_safe": sum(row["safe_for_resume_and_fid"] for row in rows),
            "resume_safe": sum(row["safe_for_resume"] for row in rows),
            "untracked_usage_possible": any(row["untracked_usage_possible"] for row in rows),
        },
    }


def _hours(seconds: float | None) -> str:
    return "unknown" if seconds is None else f"{seconds / 3600.0:.2f}"


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# GMM shift-scale TPU quota forecast",
        "",
        f"- Evidence time: `{payload.get('generated_at_utc')}`",
        f"- Scope: `{payload['accounting_scope']}`",
        f"- Resume-safe destinations: {summary['resume_safe']}/{summary['destinations']}",
        f"- Resume+FID-safe destinations: {summary['resume_and_fid_safe']}/{summary['destinations']}",
        f"- Minimum margin after resume: {_hours(summary['minimum_margin_after_resume_s'])} h",
        f"- Minimum margin after resume+FID: {_hours(summary['minimum_margin_after_resume_and_fid_s'])} h",
        f"- Untracked usage possible: `{summary['untracked_usage_possible']}`",
        "",
        "This is a conservative local-registry projection, not official Kaggle quota evidence.",
        "",
        "| run | owner | parent elapsed h | remaining now h | projected after parent h | margin after resume h | margin after resume+FID h | safe |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['run_name']} | {row['destination_owner']} | "
            f"{_hours(row['current_parent_elapsed_s'])} | {_hours(row['weekly_remaining_now_s'])} | "
            f"{_hours(row['projected_remaining_after_parent_s'])} | "
            f"{_hours(row['margin_after_resume_s'])} | "
            f"{_hours(row['margin_after_resume_and_fid_s'])} | "
            f"resume={row['safe_for_resume']}, fid={row['safe_for_resume_and_fid']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast quota after parent, resume, and FID waves.")
    parser.add_argument(
        "--quota-report",
        type=Path,
        default=Path("reports/gmm_shift_scale_tpu_quota_forecast_20260813.json"),
    )
    parser.add_argument(
        "--resume-grid",
        type=Path,
        default=Path("configs/gmm_shift_scale_training_seed15_resume400_grid.json"),
    )
    parser.add_argument(
        "--allocation-plan",
        type=Path,
        default=Path("configs/gmm_shift_scale_training_seed_resume_allocation_20260813.json"),
    )
    parser.add_argument("--fid-runtime-seconds", type=float, default=5400.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/gmm_shift_scale_training_seed12_quota_forecast.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/gmm_shift_scale_training_seed12_quota_forecast.md"),
    )
    args = parser.parse_args()
    payload = build_quota_forecast(
        _load_json(args.quota_report),
        _load_json(args.resume_grid),
        _load_json(args.allocation_plan),
        fid_runtime_s=args.fid_runtime_seconds,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
