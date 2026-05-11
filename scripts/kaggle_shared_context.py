from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from push_gmm_ablation_jobs import kaggle_command, load_kaggle_accounts, parse_kernel_status


ACTIVE_STATUSES = {"QUEUED", "RUNNING", "PENDING", "PREPARING", "STARTING", "SUBMITTED"}
TERMINAL_STATUSES = {"COMPLETE", "ERROR", "FAILED", "CANCELED", "CANCELLED", "CANCEL_ACKNOWLEDGED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_status(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().strip('"').replace("KernelWorkerStatus.", "")


def is_active_status(value: str | None) -> bool:
    return normalize_status(value).upper() in ACTIVE_STATUSES


def report_kind(path: Path, payload: dict[str, Any]) -> str:
    name = path.name
    if "tide" in name:
        return "moe2-tide"
    if "standardize" in name:
        return "gmm-standardize"
    if "ablation" in name:
        return "gmm-ablation"
    return str(payload.get("grid_config") or "unknown")


def iter_report_rows(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    kind = report_kind(path, payload)
    for section in ("jobs", "submitted", "submitted_tpu_jobs"):
        rows = payload.get(section, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            kernel_id = row.get("kernel_id") or row.get("ref")
            if not kernel_id:
                continue
            owner = row.get("owner") or str(kernel_id).split("/", 1)[0]
            raw_status = (
                row.get("latest_status")
                or row.get("kaggle_status")
                or row.get("kernel_status")
                or row.get("submitted_status")
                or row.get("status")
                or ""
            )
            yield {
                "owner": owner,
                "kernel_id": kernel_id,
                "run_name": row.get("run_name", ""),
                "grid_index": row.get("grid_index"),
                "source_report": str(path),
                "source_section": section,
                "context": kind,
                "reported_status": normalize_status(raw_status),
            }


def load_report_rows(globs: list[str]) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in globs:
        matches = sorted(Path().glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in iter_report_rows(path) or []:
            kernel_id = row["kernel_id"]
            if kernel_id in seen:
                continue
            seen.add(kernel_id)
            rows.append(row)
    return rows


def write_credential(config_dir: Path, credential: dict[str, str]) -> None:
    path = config_dir / "kaggle.json"
    path.write_text(json.dumps(credential) + "\n", encoding="utf-8")
    path.chmod(0o600)


def query_status(kernel_id: str, credential: dict[str, str]) -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix=f"kaggle-context-{credential['username']}-") as config_dir:
        write_credential(Path(config_dir), credential)
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = config_dir
        result = subprocess.run(
            [*kaggle_command(), "kernels", "status", kernel_id],
            check=False,
            env=env,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        return "STATUS_ERROR", result.stdout.strip()
    return normalize_status(parse_kernel_status(result.stdout)), result.stdout.strip()


def build_shared_context(
    *,
    report_globs: list[str],
    accounts: dict[str, dict[str, str]],
    live: bool,
) -> dict[str, Any]:
    rows = load_report_rows(report_globs)
    status_cache: dict[str, tuple[str, str]] = {}
    now = utc_now()
    resolved: list[dict[str, Any]] = []
    for row in rows:
        owner = row["owner"]
        kernel_id = row["kernel_id"]
        live_status = row["reported_status"]
        status_error = ""
        if live and owner in accounts:
            if kernel_id not in status_cache:
                status_cache[kernel_id] = query_status(kernel_id, accounts[owner])
            live_status, status_error = status_cache[kernel_id]
        elif live and owner not in accounts:
            live_status = "NO_CREDENTIAL"
            status_error = f"No Kaggle credential for {owner}"
        status = live_status or row["reported_status"]
        row = dict(row)
        row["status"] = normalize_status(status)
        row["status_error"] = status_error
        row["checked_at"] = now
        row["is_active"] = is_active_status(row["status"])
        resolved.append(row)

    active_rows = [row for row in resolved if row["is_active"]]
    active_by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in active_rows:
        active_by_owner[row["owner"]].append(row)
    status_counts = Counter(row["status"] or "UNKNOWN" for row in resolved)
    active_counts = {owner: len(rows) for owner, rows in sorted(active_by_owner.items())}
    return {
        "generated_at": now,
        "live": live,
        "report_globs": report_globs,
        "summary": {
            "kernels": len(resolved),
            "active": len(active_rows),
            "active_by_owner": active_counts,
            "status_counts": dict(sorted(status_counts.items())),
        },
        "active": active_rows,
        "kernels": resolved,
    }


def write_context(path: Path, context: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = path.with_suffix(".md")
    lines = [
        "# Kaggle Shared Context",
        "",
        f"- Generated: {context.get('generated_at', '')}",
        f"- Live status: {context.get('live')}",
        f"- Kernels scanned: {context.get('summary', {}).get('kernels', 0)}",
        f"- Active kernels: {context.get('summary', {}).get('active', 0)}",
        "",
        "## Active By Owner",
        "",
        "| owner | active |",
        "|---|---:|",
    ]
    for owner, count in context.get("summary", {}).get("active_by_owner", {}).items():
        lines.append(f"| {owner} | {count} |")
    lines.extend([
        "",
        "## Active Kernels",
        "",
        "| owner | status | context | run | kernel | report |",
        "|---|---|---|---|---|---|",
    ])
    for row in context.get("active", []):
        lines.append(
            f"| {row.get('owner', '')} | {row.get('status', '')} | "
            f"{row.get('context', '')} | {row.get('run_name', '')} | "
            f"`{row.get('kernel_id', '')}` | `{row.get('source_report', '')}` |"
        )
    lines.extend([
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "|---|---:|",
    ])
    for status, count in context.get("summary", {}).get("status_counts", {}).items():
        lines.append(f"| {status} | {count} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def active_counts_excluding(context: dict[str, Any], kernel_ids: set[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in context.get("active", []):
        if row.get("kernel_id") in kernel_ids:
            continue
        counts[row["owner"]] += 1
    return dict(counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a shared Kaggle running-context report from local submit/queue reports.")
    parser.add_argument("--accounts-file", default=str(Path("/home/tung/all-kaggle.json")))
    parser.add_argument("--report-glob", action="append", default=[])
    parser.add_argument("--output", default="reports/kaggle_shared_context.json")
    parser.add_argument("--no-live", action="store_true", help="Use only statuses already stored in reports.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    report_globs = args.report_glob or ["reports/*.json"]
    context = build_shared_context(
        report_globs=report_globs,
        accounts=accounts,
        live=not args.no_live,
    )
    write_context(Path(args.output), context)
    print(json.dumps(context["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
