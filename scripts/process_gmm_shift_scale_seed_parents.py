from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KJO_CLI = Path(
    "/mnt/c/Users/Tung/.codex/skills/kaggle-job-ops/scripts/kaggle_job_ops.py"
)
DEFAULT_KAGGLE_BIN = Path("/tmp/kaggle-cli-2.2.3-fixed/bin/kaggle")
TERMINAL_STATUSES = {"COMPLETE", "ERROR", "CANCELLED"}


@dataclass(frozen=True)
class ParentItem:
    kernel_id: str
    owner: str
    run_name: str
    run_dir: Path
    allocated_for_resume: bool
    expected_checkpoint_step: int | None
    gate_path: Path | None
    gate_spec: dict[str, Path]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _resolve(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


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
        temp_path = Path(handle.name)
    temp_path.replace(path)


def normalized_status(run_dir: Path) -> tuple[str, str]:
    payload = _read_json(run_dir / "status" / "status_result.json") or {}
    record = payload.get("record") if isinstance(payload.get("record"), dict) else {}
    raw = str(record.get("normalized_status") or "UNKNOWN").upper()
    status = raw.removeprefix("KERNELWORKERSTATUS.")
    checked_at = str(record.get("checked_at_utc") or payload.get("generated_at_utc") or "")
    return status, checked_at


def build_parent_items(
    submit_report: dict[str, Any],
    resume_grid: dict[str, Any],
    *,
    project_root: Path,
    gate_root: Path,
) -> list[ParentItem]:
    parent_rows = submit_report.get("submitted")
    resume_jobs = resume_grid.get("jobs")
    defaults = resume_grid.get("defaults")
    if not isinstance(parent_rows, list) or not isinstance(resume_jobs, list):
        raise ValueError("submit report and resume grid must contain submitted/jobs lists")
    if not isinstance(defaults, dict):
        defaults = {}

    by_parent = {
        str(job.get("resume_kernel_ref")): job
        for job in resume_jobs
        if isinstance(job, dict) and job.get("resume_kernel_ref")
    }
    expected_step_raw = defaults.get("resume_expected_checkpoint_step")
    expected_step = int(expected_step_raw) if expected_step_raw is not None else None
    items: list[ParentItem] = []
    seen: set[str] = set()
    for row in parent_rows:
        if not isinstance(row, dict):
            raise ValueError("parent submit rows must be JSON objects")
        kernel_id = str(row.get("kernel_id") or "")
        owner = str(row.get("owner") or "")
        run_name = str(row.get("run_name") or "")
        if not kernel_id or "/" not in kernel_id or not owner or not run_name:
            raise ValueError(f"invalid parent identity: {row!r}")
        if kernel_id in seen:
            raise ValueError(f"duplicate parent kernel id: {kernel_id}")
        seen.add(kernel_id)
        if kernel_id.split("/", 1)[0] != owner:
            raise ValueError(f"owner/kernel mismatch for {kernel_id}")
        run_dir = _resolve(str(row.get("run_dir") or ""), project_root)
        resume_job = by_parent.get(kernel_id)
        gate_spec: dict[str, Path] = {}
        gate_path: Path | None = None
        if resume_job is not None:
            raw_spec = resume_job.get("resume_parent_gate")
            if not isinstance(raw_spec, dict):
                raise ValueError(f"missing resume_parent_gate for {kernel_id}")
            for key in ("checkpoint", "diagnostic_manifest", "gmm_stats", "router", "summary", "audit"):
                value = raw_spec.get(key)
                if value:
                    gate_spec[key] = _resolve(str(value), project_root)
            gate_path = _resolve(
                str(raw_spec.get("gate") or gate_root / f"{kernel_id.replace('/', '__')}.json"),
                project_root,
            )
        items.append(
            ParentItem(
                kernel_id=kernel_id,
                owner=owner,
                run_name=run_name,
                run_dir=run_dir,
                allocated_for_resume=resume_job is not None,
                expected_checkpoint_step=expected_step if resume_job is not None else None,
                gate_path=gate_path,
                gate_spec=gate_spec,
            )
        )
    if set(by_parent) - seen:
        missing = sorted(set(by_parent) - seen)
        raise ValueError(f"resume grid references unknown parents: {missing}")
    return items


def _artifact_relative_path(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir / "output").as_posix()
    except ValueError as exc:
        raise ValueError(f"expected artifact is outside parent output: {path}") from exc


def heavy_download_pattern(item: ParentItem) -> str:
    if not item.allocated_for_resume:
        raise ValueError("heavy download pattern is only valid for allocated parents")
    artifact_paths = [
        _artifact_relative_path(item.gate_spec[key], item.run_dir)
        for key in ("checkpoint", "diagnostic_manifest", "gmm_stats", "router")
        if key in item.gate_spec
    ]
    diagnostics_root = re.escape(f"gmm_tide_fm/{item.run_name}/diagnostics/") + ".*"
    alternatives = [re.escape(path) for path in artifact_paths]
    alternatives.extend(
        [
            diagnostics_root,
            r"kaggle_job_ops/.*",
            r".*\.log",
        ]
    )
    return ".*(" + "|".join(alternatives) + ")$"


def diagnostics_download_pattern() -> str:
    return r".*(diagnostics/.*|kaggle_job_ops/.*|.*\.log)$"


def gate_command(item: ParentItem, kjo_cli: Path, *, record: bool) -> list[str]:
    if not item.allocated_for_resume or item.gate_path is None:
        raise ValueError("resume gate is unavailable for an excluded parent")
    required = ("checkpoint", "diagnostic_manifest")
    missing = [key for key in required if key not in item.gate_spec]
    if missing:
        raise ValueError(f"missing gate paths for {item.kernel_id}: {missing}")
    command = [
        sys.executable,
        str(kjo_cli),
        "parent-resume-gate",
        "--gate",
        str(item.gate_path),
        "--kernel-id",
        item.kernel_id,
        "--terminal-status",
        "COMPLETE",
        "--checkpoint",
        str(item.gate_spec["checkpoint"]),
        "--diagnostic-manifest",
        str(item.gate_spec["diagnostic_manifest"]),
        "--operation-timeline",
        str(item.gate_path.with_suffix(".operation_timeline.jsonl")),
    ]
    if "gmm_stats" in item.gate_spec:
        command.extend(["--gmm-stats", str(item.gate_spec["gmm_stats"])])
    if "router" in item.gate_spec:
        command.extend(["--router", str(item.gate_spec["router"])])
    if record:
        command.append("--record")
    return command


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        cp = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "timed_out": True}
    parsed: dict[str, Any] | None = None
    try:
        candidate = json.loads(cp.stdout)
        if isinstance(candidate, dict):
            parsed = candidate
    except json.JSONDecodeError:
        pass
    return {
        "ok": cp.returncode == 0,
        "returncode": cp.returncode,
        "timed_out": False,
        "json_ok": parsed.get("ok") if parsed else None,
    }


def _load_accounts(path: Path) -> dict[str, dict[str, str]]:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from push_gmm_ablation_jobs import load_kaggle_accounts

    return load_kaggle_accounts(path)


def poll_parent_status(
    item: ParentItem,
    *,
    account: dict[str, str] | None,
    registry: Path,
    kjo_cli: Path,
    kaggle_bin: Path,
    project_root: Path,
) -> dict[str, Any]:
    if account is None:
        return {"ok": False, "returncode": 98, "error": "credential_owner_missing"}
    with tempfile.TemporaryDirectory(prefix=f"kjo-parent-status-{item.owner}-", dir="/tmp") as raw_dir:
        config_dir = Path(raw_dir)
        credential_path = config_dir / "kaggle.json"
        credential_path.write_text(json.dumps(account) + "\n", encoding="utf-8")
        credential_path.chmod(0o600)
        command = [
            sys.executable,
            str(kjo_cli),
            "check-kernel-status",
            "--run-dir",
            str(item.run_dir),
            "--kernel-id",
            item.kernel_id,
            "--registry",
            str(registry),
            "--kaggle-bin",
            str(kaggle_bin),
            "--kaggle-config-dir",
            str(config_dir),
            "--timeout-s",
            "45",
        ]
        env = os.environ.copy()
        for key in ("KAGGLE_API_V1_TOKEN", "KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
            env.pop(key, None)
        env["KAGGLE_CONFIG_DIR"] = str(config_dir)
        result = _run(command, cwd=project_root, env=env, timeout_s=75)
    status, checked_at = normalized_status(item.run_dir)
    return {**result, "status": status, "checked_at_utc": checked_at}


def _download_command(
    item: ParentItem,
    *,
    status: str,
    accounts_file: Path,
    registry: Path,
    kjo_cli: Path,
    kaggle_bin: Path,
) -> list[str]:
    include_heavy = status == "COMPLETE" and item.allocated_for_resume
    command = [
        sys.executable,
        str(kjo_cli),
        "download-kernel-output",
        "--run-dir",
        str(item.run_dir),
        "--kernel-id",
        item.kernel_id,
        "--accounts-file",
        str(accounts_file),
        "--credential-owner",
        item.owner,
        "--runtime-owner",
        item.owner,
        "--require-source-owner-credential",
        "--kaggle-bin",
        str(kaggle_bin),
        "--file-pattern",
        heavy_download_pattern(item) if include_heavy else diagnostics_download_pattern(),
        "--max-attempts",
        "3",
        "--timeout-s",
        "180",
        "--registry",
        str(registry),
        "--mark-downloaded",
        "--kind",
        "artifacts" if include_heavy else "diagnostics",
    ]
    if include_heavy:
        command.append("--include-heavy")
    return command


def _summary_command(item: ParentItem, *, registry: Path, kjo_cli: Path) -> list[str]:
    summary_path = item.gate_spec.get("summary", item.run_dir / "reports" / "summary.json")
    return [
        sys.executable,
        str(kjo_cli),
        "summarize-run-dir",
        "--run-dir",
        str(item.run_dir),
        "--kernel-id",
        item.kernel_id,
        "--registry",
        str(registry),
        "--out-json",
        str(summary_path),
        "--out-md",
        str(summary_path.with_suffix(".md")),
    ]


def _audit_command(
    item: ParentItem,
    *,
    status: str,
    registry: Path,
    kjo_cli: Path,
    strict_submit_evidence: bool,
) -> list[str]:
    audit_path = item.gate_spec.get("audit", item.run_dir / "reports" / "audit_run_dir.json")
    command = [
        sys.executable,
        str(kjo_cli),
        "audit-run-dir",
        "--run-dir",
        str(item.run_dir),
        "--kernel-id",
        item.kernel_id,
        "--registry",
        str(registry),
        "--require-status-poll",
        "--require-status-summary",
        "--require-output",
        "--require-download-result",
        "--require-report-summary",
        "--require-kjo-cell-logs",
        "--require-accelerator-probe-contract",
        "--out",
        str(audit_path),
    ]
    if not strict_submit_evidence:
        command.extend(["--allow-missing-submit-result", "--allow-missing-pre-submit-audit"])
    if status != "COMPLETE":
        command.append("--allow-failed-run-summary")
    return command


def _manifest_step_check(item: ParentItem) -> dict[str, Any]:
    if not item.allocated_for_resume:
        return {"checked": False, "ok": True}
    manifest = _read_json(item.gate_spec["diagnostic_manifest"])
    actual = manifest.get("step") if manifest else None
    expected = item.expected_checkpoint_step
    return {
        "checked": True,
        "ok": manifest is not None and expected is not None and actual == expected,
        "expected": expected,
        "actual": actual,
    }


def _json_ok(path: Path) -> bool:
    payload = _read_json(path)
    return bool(payload and payload.get("ok") is True)


def process_terminal_parent(
    item: ParentItem,
    *,
    status: str,
    accounts_file: Path,
    registry: Path,
    kjo_cli: Path,
    kaggle_bin: Path,
    project_root: Path,
    dry_run: bool,
    strict_submit_evidence: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kernel_id": item.kernel_id,
        "owner": item.owner,
        "run_name": item.run_name,
        "run_dir": str(item.run_dir),
        "status": status,
        "allocated_for_resume": item.allocated_for_resume,
        "actions": [],
        "gate_recorded": False,
        "ok": True,
    }
    if status not in TERMINAL_STATUSES:
        result["actions"].append("skip_non_terminal")
        return result

    if status == "COMPLETE" and item.allocated_for_resume and item.gate_path and item.gate_path.is_file():
        gate_hit = _run(gate_command(item, kjo_cli, record=False), cwd=project_root, timeout_s=60)
        if gate_hit["ok"]:
            result["actions"].append("skip_existing_parent_gate_hit")
            result["gate_recorded"] = True
            return result

    if dry_run:
        result["actions"].extend(["would_download", "would_summarize", "would_audit"])
        if status == "COMPLETE" and item.allocated_for_resume:
            result["actions"].append("would_record_parent_gate")
        return result

    download = _run(
        _download_command(
            item,
            status=status,
            accounts_file=accounts_file,
            registry=registry,
            kjo_cli=kjo_cli,
            kaggle_bin=kaggle_bin,
        ),
        cwd=project_root,
        timeout_s=900,
    )
    result["actions"].append("download")
    result["download"] = download
    if not download["ok"]:
        result["ok"] = False
        return result

    summary = _run(_summary_command(item, registry=registry, kjo_cli=kjo_cli), cwd=project_root, timeout_s=300)
    result["actions"].append("summarize")
    result["summary"] = summary
    if not summary["ok"]:
        result["ok"] = False
        return result

    audit = _run(
        _audit_command(
            item,
            status=status,
            registry=registry,
            kjo_cli=kjo_cli,
            strict_submit_evidence=strict_submit_evidence,
        ),
        cwd=project_root,
        timeout_s=300,
    )
    result["actions"].append("audit")
    result["audit"] = audit

    summary_path = item.gate_spec.get("summary", item.run_dir / "reports" / "summary.json")
    audit_path = item.gate_spec.get("audit", item.run_dir / "reports" / "audit_run_dir.json")
    summary_ok = _json_ok(summary_path)
    audit_ok = _json_ok(audit_path)
    result["summary_ok"] = summary_ok
    result["audit_ok"] = audit_ok
    result["ok"] = bool(summary_ok and audit_ok)

    if status != "COMPLETE" or not item.allocated_for_resume:
        return result

    artifact_exists = {
        key: path.is_file()
        for key, path in item.gate_spec.items()
        if key in {"checkpoint", "diagnostic_manifest", "gmm_stats", "router"}
    }
    step_check = _manifest_step_check(item)
    result["artifact_exists"] = artifact_exists
    result["diagnostic_manifest_step"] = step_check
    if not result["ok"] or not all(artifact_exists.values()) or not step_check["ok"]:
        result["ok"] = False
        return result

    gate_record = _run(gate_command(item, kjo_cli, record=True), cwd=project_root, timeout_s=60)
    result["actions"].append("record_parent_gate")
    result["parent_gate"] = gate_record
    result["gate_recorded"] = bool(gate_record["ok"] and item.gate_path and _json_ok(item.gate_path))
    result["ok"] = result["gate_recorded"]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Poll the 15 training-seed parents and idempotently download, summarize, audit, "
            "and record resume gates for terminal runs."
        )
    )
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
    parser.add_argument("--accounts-file", type=Path, default=Path("/home/tung/all-kaggle.json"))
    parser.add_argument("--registry", type=Path, default=Path(".secrets/kaggle_notebooks.jsonl"))
    parser.add_argument(
        "--gate-root", type=Path, default=Path("outputs/kaggle_jobs/parent_resume_gates")
    )
    parser.add_argument("--kjo-cli", type=Path, default=DEFAULT_KJO_CLI)
    parser.add_argument("--kaggle-bin", type=Path, default=DEFAULT_KAGGLE_BIN)
    parser.add_argument("--max-status-workers", type=int, default=8)
    parser.add_argument("--max-terminal-workers", type=int, default=3)
    parser.add_argument(
        "--expected-parent-count",
        type=int,
        default=15,
        help="Exact expected submit rows; use 0 for a partial, incrementally growing wave.",
    )
    parser.add_argument(
        "--strict-submit-evidence",
        action="store_true",
        help="Require KJO submit_result and pre-submit audit; use for atomic child waves.",
    )
    parser.add_argument("--no-poll-status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/gmm_shift_scale_training_seed15_parent_processing.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = PROJECT_ROOT
    submit_report_path = _resolve(args.submit_report, project_root)
    resume_grid_path = _resolve(args.resume_grid, project_root)
    accounts_file = _resolve(args.accounts_file, project_root)
    registry = _resolve(args.registry, project_root)
    gate_root = _resolve(args.gate_root, project_root)
    out_path = _resolve(args.out, project_root)
    kjo_cli = _resolve(args.kjo_cli, project_root)
    kaggle_bin = _resolve(args.kaggle_bin, project_root)
    submit_report = _read_json(submit_report_path)
    resume_grid = _read_json(resume_grid_path)
    if submit_report is None or resume_grid is None:
        raise SystemExit("could not load submit report or resume grid")
    items = build_parent_items(
        submit_report,
        resume_grid,
        project_root=project_root,
        gate_root=gate_root,
    )
    if args.expected_parent_count > 0 and len(items) != args.expected_parent_count:
        raise SystemExit(
            f"expected exactly {args.expected_parent_count} parent rows, found {len(items)}"
        )

    status_results: dict[str, dict[str, Any]] = {}
    if args.no_poll_status or args.dry_run:
        for item in items:
            status, checked_at = normalized_status(item.run_dir)
            status_results[item.kernel_id] = {
                "ok": True,
                "status": status,
                "checked_at_utc": checked_at,
                "source": "existing_status_result",
            }
    else:
        accounts = _load_accounts(accounts_file)
        with ThreadPoolExecutor(max_workers=max(1, args.max_status_workers)) as pool:
            futures = {
                pool.submit(
                    poll_parent_status,
                    item,
                    account=accounts.get(item.owner),
                    registry=registry,
                    kjo_cli=kjo_cli,
                    kaggle_bin=kaggle_bin,
                    project_root=project_root,
                ): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    status_results[item.kernel_id] = future.result()
                except Exception as exc:  # preserve one bad worker without losing the wave report
                    status_results[item.kernel_id] = {
                        "ok": False,
                        "status": "UNKNOWN",
                        "error": type(exc).__name__,
                    }

    process_results: list[dict[str, Any]] = []
    terminal_items = [
        item
        for item in items
        if status_results.get(item.kernel_id, {}).get("status") in TERMINAL_STATUSES
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.max_terminal_workers)) as pool:
        futures = {
            pool.submit(
                process_terminal_parent,
                item,
                status=str(status_results[item.kernel_id]["status"]),
                accounts_file=accounts_file,
                registry=registry,
                kjo_cli=kjo_cli,
                kaggle_bin=kaggle_bin,
                project_root=project_root,
                dry_run=args.dry_run,
                strict_submit_evidence=args.strict_submit_evidence,
            ): item
            for item in terminal_items
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                process_results.append(future.result())
            except Exception as exc:
                process_results.append(
                    {
                        "kernel_id": item.kernel_id,
                        "owner": item.owner,
                        "status": status_results[item.kernel_id].get("status", "UNKNOWN"),
                        "ok": False,
                        "error": type(exc).__name__,
                    }
                )

    status_counts = Counter(
        str(status_results.get(item.kernel_id, {}).get("status") or "UNKNOWN") for item in items
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "kjo_cli": str(kjo_cli),
        "kaggle_bin": str(kaggle_bin),
        "submit_report": str(submit_report_path),
        "resume_grid": str(resume_grid_path),
        "registry": str(registry),
        "dry_run": args.dry_run,
        "status_results": [
            {"kernel_id": item.kernel_id, "owner": item.owner, **status_results[item.kernel_id]}
            for item in items
        ],
        "processed": sorted(process_results, key=lambda row: str(row.get("kernel_id"))),
        "summary": {
            "parents": len(items),
            "allocated_for_resume": sum(item.allocated_for_resume for item in items),
            "status_counts": dict(sorted(status_counts.items())),
            "terminal": len(terminal_items),
            "processed": len(process_results),
            "processed_ok": sum(row.get("ok") is True for row in process_results),
            "processed_failed": sum(row.get("ok") is not True for row in process_results),
            "gates_recorded": sum(row.get("gate_recorded") is True for row in process_results),
        },
    }
    _atomic_write_json(out_path, payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    if any(not row.get("ok") for row in process_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
