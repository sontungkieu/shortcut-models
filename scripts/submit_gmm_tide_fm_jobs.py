from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_shared_context import active_counts_excluding, build_shared_context, write_context
from stage_gmm_ablation_jobs import load_env_file, normalize_accelerator, slugify
from push_gmm_ablation_jobs import kaggle_command, load_kaggle_accounts, parse_kernel_id, parse_kernel_status


DEFAULT_ACCOUNTS_FILE = Path("/home/tung/all-kaggle.json")
KAGGLE_JOB_OPS_SCRIPT = Path("/home/tung/.codex/skills/kaggle-job-ops/scripts/kaggle_job_ops.py")
DEFAULT_JOB_ROOT = Path("outputs/kaggle_jobs/gmm_tide_fm")
DEFAULT_NOTEBOOK_REGISTRY = Path(".secrets/kaggle_notebooks.jsonl")
DEFAULT_INJECTED_NOTEBOOK_LOG = Path(".secrets/injected_notebooks.md")
BLUEPRINT_SOURCE_KERNEL_REF = "kjo-placeholder/source-kernel"
BLUEPRINT_DESTINATION_OWNER = "kjo-placeholder-owner"
BLUEPRINT_DESTINATION_SLUG = "kjo-placeholder-slug"


def load_grid(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = payload.get("defaults", {})
    jobs = []
    for index, raw_job in enumerate(payload.get("jobs", [])):
        job = dict(defaults)
        job.update(raw_job)
        job["grid_index"] = index
        job["gmm_min_std"] = math.sqrt(max(float(job.get("gmm_min_var", 0.0)), 0.0))
        job["gmm_min_std_data_frac"] = math.sqrt(max(float(job.get("gmm_min_var_data_frac", 0.0)), 0.0))
        jobs.append(job)
    if not jobs:
        raise SystemExit(f"No jobs found in {path}")
    return jobs


def selected_owners(value: str, available: list[str], exclude: str) -> list[str]:
    if value == "all":
        owners = sorted(available)
    else:
        owners = [item.strip() for item in value.split(",") if item.strip()]
    missing = [owner for owner in owners if owner not in available]
    if missing:
        raise SystemExit(f"Unknown Kaggle owner(s): {', '.join(missing)}")
    excluded = {item.strip() for item in exclude.split(",") if item.strip()}
    owners = [owner for owner in owners if owner not in excluded]
    if not owners:
        raise SystemExit("No owners selected after applying exclusions.")
    return owners


def resume_download_credential_owner(
    *,
    config: dict[str, Any],
    target_owner: str,
    accounts: dict[str, dict[str, str]],
) -> str | None:
    if not config.get("resume_kernel_ref") or not bool(config.get("resume_download_output", True)):
        return None
    resume_ref = str(config["resume_kernel_ref"])
    source_owner = resume_ref.split("/", 1)[0] if "/" in resume_ref else ""
    if not source_owner:
        raise ValueError(f"resume_kernel_ref must be owner/slug, got {resume_ref!r}")
    if source_owner not in accounts:
        raise ValueError(
            f"resume source owner {source_owner!r} is not in the accounts file; "
            "cross-account resume requires the exact source-owner credential"
        )
    return source_owner


def resume_file_pattern(config: dict[str, Any]) -> str:
    patterns = [
        r".*gmm_stats\.npz$",
        r".*gmm_router\.pkl$",
        r".*diagnostics/(gmm_metrics\.json|router_metrics_summary\.json|train_metrics_summary\.json)$",
    ]
    require_checkpoint = bool(
        config.get(
            "resume_require_checkpoint",
            str(config.get("execution_mode", "train")).strip().lower() != "router_geometry_audit",
        )
    )
    if require_checkpoint:
        # Checkpoints use a stable ckpts/<run_name>.pkl filename. The training
        # step is stored inside the checkpoint, so filtering by a step suffix
        # drops the exact artifact that resume/eval needs.
        patterns.append(r".*ckpts/.*\.pkl$")
    return "|".join(patterns)


def render_cross_account_output_source(
    staging_dir: Path,
    config: dict[str, Any],
    runtime_owner: str,
) -> str:
    if not config.get("resume_kernel_ref") or not bool(config.get("resume_download_output", True)):
        return ""
    if not KAGGLE_JOB_OPS_SCRIPT.exists():
        raise FileNotFoundError(f"Missing Kaggle Job Ops helper: {KAGGLE_JOB_OPS_SCRIPT}")
    source_path = staging_dir / "sources" / "04_cross_account_output.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(KAGGLE_JOB_OPS_SCRIPT),
        "render-cross-account-output-cell",
        "--out",
        str(source_path),
        "--kernel-id",
        str(config["resume_kernel_ref"]),
        "--runtime-owner",
        str(runtime_owner),
        "--kaggle-config-dir",
        "/tmp/.kaggle_source_owner",
        "--output-dir",
        str(config.get("resume_output_dir", "/tmp/resume_output")),
        "--file-pattern",
        resume_file_pattern(config),
        "--max-attempts",
        str(config.get("resume_download_max_attempts", 3)),
        "--timeout-s",
        str(config.get("resume_download_timeout_s", 600)),
    ]
    result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0 or not source_path.exists():
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Failed to render cross-account output cell: {detail}")
    return source_path.read_text(encoding="utf-8")


def next_owner(
    owners: list[str],
    counts: Counter[str],
    max_submit_per_owner: int,
    cursor: int,
) -> tuple[str | None, int]:
    for offset in range(len(owners)):
        index = (cursor + offset) % len(owners)
        owner = owners[index]
        if max_submit_per_owner <= 0 or counts[owner] < max_submit_per_owner:
            return owner, index + 1
    return None, cursor


def has_tracked_changes() -> bool:
    unstaged = subprocess.run(["git", "diff", "--quiet"], check=False)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
    return unstaged.returncode != 0 or staged.returncode != 0


def remote_branches_containing(commit: str) -> list[str]:
    result = subprocess.run(
        ["git", "branch", "-r", "--contains", commit],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [line.strip().lstrip("* ").strip() for line in result.stdout.splitlines() if line.strip()]


def ensure_submit_source_ready(commit: str, allow_dirty: bool, dry_run: bool) -> None:
    if dry_run:
        return
    dirty = has_tracked_changes()
    remote_branches = remote_branches_containing(commit)
    if dirty and not allow_dirty:
        raise SystemExit(
            "Refusing to submit with tracked uncommitted changes. Kaggle checks out the recorded "
            "repo_commit from GitHub, so local edits would be missing. Commit and push first, or "
            "rerun with --allow-dirty if you intentionally want to submit the current HEAD."
        )
    if not remote_branches and not allow_dirty:
        raise SystemExit(
            f"Refusing to submit commit {commit}: no remote-tracking branch contains it. "
            "Push the commit first, or rerun with --allow-dirty if you intentionally accept that risk."
        )
    if allow_dirty and dirty:
        print(
            "WARNING: --allow-dirty is set. The staged notebook will still checkout the current HEAD "
            "commit only; uncommitted local edits will not run on Kaggle.",
            flush=True,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def accelerator_kind(accelerator: str) -> str:
    lowered = accelerator.strip().lower()
    if lowered in {"cpu", "none", "noaccelerator", "no-accelerator"}:
        return "cpu"
    if lowered.startswith("tpu"):
        return "tpu"
    return "gpu"


def run_dir_for_kernel(job_root: Path, kernel_id: str) -> Path:
    return job_root / kernel_id.replace("/", "__")


def run_json_command(cmd: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = result.stdout.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Command did not return JSON: {' '.join(cmd)}\n{text[-2000:]}") from exc
    if result.returncode != 0 or not payload.get("ok", False):
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{json.dumps(payload, indent=2, sort_keys=True)}")
    return payload


def reserve_exact_owner(
    *,
    owner: str,
    accelerator: str,
    accounts_file: Path,
    project_root: Path,
    run_id: str,
    task_id: str,
    estimated_runtime_minutes: float,
    ttl_minutes: int,
) -> dict[str, Any]:
    kind = accelerator_kind(accelerator)
    command = [
        sys.executable,
        str(KAGGLE_JOB_OPS_SCRIPT),
        "reserve-owners",
        "--accounts-file",
        str(accounts_file),
        "--owners",
        owner,
        "--preferred-owners",
        owner,
        "--accelerator",
        kind,
        "--count",
        "1",
        "--estimated-runtime-minutes",
        str(estimated_runtime_minutes),
        "--ttl-minutes",
        str(ttl_minutes),
        "--reserve-project-root",
        str(project_root),
        "--run-id",
        run_id,
        "--task-id",
        task_id,
        "--registry-sync-mode",
        "db-only",
        "--live",
        "--kaggle-bin",
        str(kaggle_command()[0]),
        "--note",
        "exact-owner atomic reservation for GMM-TIDE job",
    ]
    payload = run_json_command(command)
    reserved = payload.get("reserved")
    if not isinstance(reserved, list) or len(reserved) != 1:
        raise RuntimeError(f"Expected one exact reservation for {owner}, got {reserved!r}")
    row = reserved[0]
    if not isinstance(row, dict) or str(row.get("owner") or "") != owner:
        raise RuntimeError(f"Reservation owner mismatch for {owner}: {row!r}")
    token = str(row.get("reservation_token") or "")
    if not token:
        raise RuntimeError(f"Reservation for {owner} did not contain a token")
    return {"payload": payload, "reservation": row, "reservation_token": token}


def release_unused_reservation(*, owner: str, accelerator: str, reservation: dict[str, Any]) -> dict[str, Any]:
    slot_id = reservation.get("slot_id")
    token = str(reservation.get("reservation_token") or "")
    if slot_id is None or not token:
        raise ValueError(f"Cannot release incomplete reservation for {owner}: {reservation!r}")
    return run_json_command(
        [
            sys.executable,
            str(KAGGLE_JOB_OPS_SCRIPT),
            "release-owner",
            "--owner",
            owner,
            "--accelerator",
            accelerator_kind(accelerator),
            "--slot-id",
            str(slot_id),
            "--reservation-token",
            token,
            "--reason",
            "local failure before reservation token handoff to submit-kernel",
        ]
    )


def build_atomic_submit_command(
    *,
    run_dir: Path,
    staging_dir: Path,
    owner: str,
    accelerator: str,
    reservation_token: str,
    registry: Path,
    project_root: Path,
    run_id: str,
    task_id: str,
    artifact_mode: str,
    retention_action: str,
    embedded_key_names: list[str],
    kaggle_config_dir: Path,
    runtime_dataset_source: str,
) -> list[str]:
    metadata_path = staging_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    submitted_notebook = staging_dir / str(metadata["code_file"])
    command = [
        sys.executable,
        str(KAGGLE_JOB_OPS_SCRIPT),
        "submit-kernel",
        "--run-dir",
        str(run_dir),
        "--metadata",
        str(metadata_path),
        "--submitted-notebook",
        str(submitted_notebook),
        "--expected-accelerator",
        accelerator_kind(accelerator),
        "--owner",
        owner,
        "--reservation-token",
        reservation_token,
        "--record-registry",
        str(registry),
        "--artifact-mode",
        artifact_mode,
        "--retention-action",
        retention_action,
        "--is-private",
        "--require-notebook-logging-contract",
        "--require-accelerator-probe-contract",
        "--kaggle-bin",
        str(kaggle_command()[0]),
        "--kaggle-config-dir",
        str(kaggle_config_dir),
        "--local-secret-scrub-ledger",
        str(DEFAULT_INJECTED_NOTEBOOK_LOG),
        "--project-root",
        str(project_root),
        "--run-id",
        run_id,
        "--task-id",
        task_id,
        "--operation-timeline",
        str(run_dir / "operation_timeline.jsonl"),
        "--note",
        "KJO atomic submit from GMM-TIDE helper",
    ]
    if accelerator_kind(accelerator) != "cpu":
        command.extend(["--submit-accelerator", accelerator])
    if runtime_dataset_source:
        command.extend(["--required-dataset-source", runtime_dataset_source])
    key_names = sorted({str(name) for name in embedded_key_names if str(name)})
    if key_names:
        command.extend(["--secret-mode", "embedded"])
        for key_name in key_names:
            command.extend(["--embedded-key-name", key_name])
    else:
        command.extend(["--secret-mode", "none"])
    return command


def kjo_submit_artifact_paths(run_dir: Path) -> dict[str, str]:
    return {
        "run_dir": str(run_dir),
        "submitted_notebook": str(run_dir / "submit" / "submitted_notebook.ipynb"),
        "metadata": str(run_dir / "submit" / "kernel-metadata.json"),
        "submit_stdout": str(run_dir / "submit" / "submit_stdout.txt"),
        "submit_stderr": str(run_dir / "submit" / "submit_stderr.txt"),
        "submit_result": str(run_dir / "submit" / "submit_result.json"),
        "local_secret_scrub": str(run_dir / "submit" / "local_secret_scrub_result.json"),
        "status_result": str(run_dir / "status" / "status_result.json"),
    }


def copy_atomic_submission_evidence(staging_dir: Path, run_dir: Path) -> dict[str, str]:
    submit_dir = run_dir / "submit"
    submit_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for source_name, destination_name in (
        ("gmm_tide_config.json", "gmm_tide_config.json"),
        ("stage_package_manifest.json", "stage_package_manifest.json"),
        ("staging_blueprint_result.json", "staging_blueprint_result.json"),
        ("operation_timeline.jsonl", "stage_operation_timeline.jsonl"),
    ):
        source = staging_dir / source_name
        if source.is_file():
            destination = submit_dir / destination_name
            shutil.copy2(source, destination)
            copied[source_name] = str(destination)
    return copied


def parent_resume_gate_path(gate_root: Path, kernel_id: str) -> Path:
    return gate_root / f"{kernel_id.replace('/', '__')}.json"


def _require_ok_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("ok") is not True:
        raise ValueError(f"{label} is not verified ok: {path}")
    return payload


def evaluate_parent_resume_gate(
    *,
    config: dict[str, Any],
    gate_root: Path,
    require_cache_hit: bool = False,
    record: bool = False,
) -> dict[str, Any] | None:
    kernel_id = str(config.get("resume_kernel_ref") or "")
    if not kernel_id:
        return None
    raw_spec = config.get("resume_parent_gate")
    if not isinstance(raw_spec, dict):
        if require_cache_hit:
            raise ValueError(
                f"resume_parent_gate is required for {kernel_id}; provide checkpoint, diagnostic manifest, "
                "terminal status, and optional GMM/router artifact paths"
            )
        return {
            "ok": True,
            "configured": False,
            "kernel_id": kernel_id,
            "cache_hit": False,
            "skip_parent_processing": False,
        }

    required = ("terminal_status", "checkpoint", "diagnostic_manifest")
    missing = [key for key in required if not str(raw_spec.get(key) or "")]
    if missing:
        raise ValueError(f"resume_parent_gate for {kernel_id} is missing: {', '.join(missing)}")
    gate_root.mkdir(parents=True, exist_ok=True)
    gate_path = Path(str(raw_spec.get("gate") or parent_resume_gate_path(gate_root, kernel_id)))
    timeline_path = gate_root / f"{kernel_id.replace('/', '__')}.operation_timeline.jsonl"
    command = [
        sys.executable,
        str(KAGGLE_JOB_OPS_SCRIPT),
        "parent-resume-gate",
        "--gate",
        str(gate_path),
        "--kernel-id",
        kernel_id,
        "--terminal-status",
        str(raw_spec["terminal_status"]),
        "--checkpoint",
        str(raw_spec["checkpoint"]),
        "--diagnostic-manifest",
        str(raw_spec["diagnostic_manifest"]),
        "--operation-timeline",
        str(timeline_path),
    ]
    for flag, key in (("--gmm-stats", "gmm_stats"), ("--router", "router")):
        value = str(raw_spec.get(key) or "")
        if value:
            command.extend([flag, value])
    if record:
        _require_ok_json(Path(str(raw_spec.get("summary") or "")), "parent summary")
        _require_ok_json(Path(str(raw_spec.get("audit") or "")), "parent audit")
        command.append("--record")
    payload = run_json_command(command)
    payload.update(
        {
            "configured": True,
            "record_requested": bool(record),
            "operation_timeline": str(timeline_path),
        }
    )
    if require_cache_hit and not payload.get("cache_hit"):
        raise RuntimeError(
            f"Parent resume gate miss for {kernel_id}; complete parent download/summary/audit and record the gate first"
        )
    return payload


def ensure_kaggle_cli_for_submit(accelerator: str, skip: bool, dry_run: bool) -> None:
    if skip or dry_run or accelerator_kind(accelerator) != "tpu":
        return
    payload = run_json_command([sys.executable, str(KAGGLE_JOB_OPS_SCRIPT), "ensure-cli"])
    bin_dir = payload.get("bin_dir")
    if bin_dir:
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        print(f"Using Kaggle CLI from {bin_dir}", flush=True)


def validate_staged_metadata(staging_dir: Path, owner: str, accelerator: str, skip: bool) -> dict[str, Any] | None:
    if skip:
        return None
    kind = accelerator_kind(accelerator)
    cmd = [
        sys.executable,
        str(KAGGLE_JOB_OPS_SCRIPT),
        "validate-metadata",
        "--metadata",
        str(staging_dir / "kernel-metadata.json"),
        "--expected-accelerator",
        kind,
        "--owner",
        owner,
    ]
    if kind != "cpu":
        cmd.extend(["--submit-accelerator", accelerator])
    return run_json_command(cmd)


def copy_submission_artifacts(staging_dir: Path, run_dir: Path) -> dict[str, str]:
    submit_dir = run_dir / "submit"
    submit_dir.mkdir(parents=True, exist_ok=True)
    metadata_src = staging_dir / "kernel-metadata.json"
    config_src = staging_dir / "gmm_tide_config.json"
    metadata = json.loads(metadata_src.read_text(encoding="utf-8"))
    notebook_src = staging_dir / metadata["code_file"]
    notebook_dst = submit_dir / "submitted_notebook.ipynb"
    metadata_dst = submit_dir / "kernel-metadata.json"
    config_dst = submit_dir / "gmm_tide_config.json"
    shutil.copy2(notebook_src, notebook_dst)
    shutil.copy2(metadata_src, metadata_dst)
    shutil.copy2(config_src, config_dst)
    optional_artifacts = {}
    for name, destination in (
        ("stage_package_manifest.json", submit_dir / "stage_package_manifest.json"),
        ("staging_blueprint_result.json", submit_dir / "staging_blueprint_result.json"),
        ("operation_timeline.jsonl", run_dir / "operation_timeline.jsonl"),
    ):
        source = staging_dir / name
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            optional_artifacts[name] = str(destination)
    return {
        "run_dir": str(run_dir),
        "submitted_notebook": str(notebook_dst),
        "metadata": str(metadata_dst),
        "config": str(config_dst),
        "submit_stdout": str(submit_dir / "submit_stdout.txt"),
        "status_stdout": str(submit_dir / "status_stdout.txt"),
        "status_poll": str(run_dir / "status" / "status_poll.jsonl"),
        "local_secret_scrub": str(submit_dir / "local_secret_scrub_result.json"),
        **optional_artifacts,
    }


def scrub_notebook_embedded_credentials(notebook_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(notebook_path),
        "exists": notebook_path.exists(),
        "ok": True,
        "key_names": [],
        "replacements": 0,
    }
    if not notebook_path.exists():
        return result

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    patterns = (
        (re.compile(r"^WANDB_API_KEY\s*=.*$", flags=re.MULTILINE), 'WANDB_API_KEY = ""', "WANDB_API_KEY"),
        (
            re.compile(r"^KAGGLE_CREDENTIAL\s*=\s*json\.loads\(.*\)\s*$", flags=re.MULTILINE),
            "KAGGLE_CREDENTIAL = {}",
            "KAGGLE_CREDENTIAL",
        ),
    )
    key_names: set[str] = set()
    replacements = 0
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        source_text = "".join(source) if isinstance(source, list) else str(source)
        for pattern, replacement, key_name in patterns:
            source_text, count = pattern.subn(replacement, source_text)
            if count:
                key_names.add(key_name)
                replacements += count
        cell["source"] = source_text.splitlines(keepends=True)

    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    result.update(
        {
            "key_names": sorted(key_names),
            "replacements": replacements,
        }
    )
    return result


def scrub_local_submission_notebooks(
    staging_dir: Path,
    artifact_paths: dict[str, str] | None,
) -> dict[str, Any]:
    notebook_paths: list[Path] = []
    metadata_path = staging_dir / "kernel-metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        code_file = metadata.get("code_file")
        if code_file:
            notebook_paths.append(staging_dir / str(code_file))
    if artifact_paths is not None:
        notebook_paths.append(Path(artifact_paths["submitted_notebook"]))

    results = [scrub_notebook_embedded_credentials(path) for path in notebook_paths]
    payload = {
        "scrubbed_at_utc": utc_now(),
        "ok": all(bool(item["ok"]) for item in results),
        "notebooks": results,
    }
    if artifact_paths is not None:
        receipt_path = Path(artifact_paths["local_secret_scrub"])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def record_injected_notebook(
    *,
    notebook_path: Path,
    owner: str,
    kernel_id: str,
    key_names: list[str],
    log_path: Path = DEFAULT_INJECTED_NOTEBOOK_LOG,
) -> None:
    key_names = sorted({str(name) for name in key_names if str(name)})
    if not key_names:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# Injected Kaggle Notebooks\n\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"- {utc_now()} | `{notebook_path}` | `{owner}` | `{kernel_id}` | "
            f"keys: `{','.join(key_names)}`\n"
        )


def parse_submit_stdout(log_path: Path) -> dict[str, Any]:
    return run_json_command([sys.executable, str(KAGGLE_JOB_OPS_SCRIPT), "parse-submit-log", "--log", str(log_path)])


def append_status_poll(run_dir: Path, *, owner: str, kernel_id: str, status: str, method: str, returncode: int, output: str) -> None:
    poll_dir = run_dir / "status"
    poll_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checked_at_utc": utc_now(),
        "owner": owner,
        "ref": kernel_id,
        "status": status,
        "method": method,
        "returncode": int(returncode),
        "output_tail": output[-2000:],
    }
    with (poll_dir / "status_poll.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def record_submitted_notebook(
    *,
    registry: Path,
    kernel_id: str,
    run_name: str,
    project_root: Path,
    accelerator: str,
    artifact_mode: str,
    retention_action: str,
    secret_mode: str,
    embedded_key_names: list[str],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(KAGGLE_JOB_OPS_SCRIPT),
        "record-notebook",
        "--registry",
        str(registry),
        "--kernel-id",
        kernel_id,
        "--secret-mode",
        secret_mode,
        "--artifact-mode",
        artifact_mode,
        "--retention-action",
        retention_action,
        "--is-private",
        "--run-id",
        run_name,
        "--project-root",
        str(project_root),
        "--accelerator",
        accelerator_kind(accelerator),
        "--submit-accelerator",
        accelerator,
        "--title",
        run_name,
        "--submitted-notebook",
        artifact_paths["submitted_notebook"],
        "--metadata",
        artifact_paths["metadata"],
        "--submit-stdout",
        artifact_paths["submit_stdout"],
    ]
    for key_name in embedded_key_names:
        cmd.extend(["--embedded-key-name", key_name])
    return run_json_command(cmd)


def render_accelerator_probe_source(staging_dir: Path, requested_accelerator: str) -> str:
    source_dir = staging_dir / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "01_accelerator_probe.py"
    run_json_command(
        [
            sys.executable,
            str(KAGGLE_JOB_OPS_SCRIPT),
            "render-accelerator-probe-cell",
            "--out",
            str(source_path),
            "--requested-accelerator",
            requested_accelerator,
        ]
    )
    return source_path.read_text(encoding="utf-8")


def make_code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def make_notebook(
    config: dict[str, Any],
    wandb_api_key: str = "",
    kaggle_credential: dict[str, str] | None = None,
    accelerator_probe_source: str = "",
    cross_account_output_source: str = "",
    router_geometry_audit_script_source: str = "",
) -> dict[str, Any]:
    config_json = json.dumps(config, indent=4, sort_keys=True)
    config_json_literal = json.dumps(config_json)
    wandb_key_json = json.dumps(wandb_api_key)
    kaggle_credential_json = json.dumps(kaggle_credential or {})
    cells = [
        make_code_cell(
            f"""import json
import os
from pathlib import Path

CONFIG = json.loads({config_json_literal})
RUN_NAME = CONFIG["run_name"]
WANDB_API_KEY = {wandb_key_json}
KAGGLE_CREDENTIAL = json.loads({json.dumps(kaggle_credential_json)})

if CONFIG.get("execution_mode") in {"fid_repeats", "trajectory_eval", "router_geometry_audit"}:
    os.environ["WANDB_MODE"] = "offline"
elif WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
else:
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        for secret_name in ("WANDB2", "WANDB_API_KEY"):
            try:
                value = secrets.get_secret(secret_name)
            except Exception:
                value = ""
            if value:
                os.environ["WANDB_API_KEY"] = value
                break
    except Exception:
        pass

if not os.environ.get("WANDB_API_KEY"):
    os.environ["WANDB_MODE"] = "offline"

if KAGGLE_CREDENTIAL:
    kaggle_config_dir = Path("/tmp/.kaggle_source_owner")
    kaggle_config_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_config_dir / "kaggle.json"
    kaggle_json_path.write_text(json.dumps(KAGGLE_CREDENTIAL) + "\\n", encoding="utf-8")
    kaggle_json_path.chmod(0o600)

del WANDB_API_KEY
del KAGGLE_CREDENTIAL

os.environ["MPLBACKEND"] = "agg"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ.setdefault("WANDB__SERVICE_WAIT", "120")
print(json.dumps(CONFIG, indent=2, sort_keys=True))
"""
        ),
        make_code_cell(
            """import subprocess
from pathlib import Path


def _print_log_tail(path: Path, max_chars: int = 20000) -> None:
    path = Path(path)
    if not path.exists():
        print(f"Missing log file: {path}", flush=True)
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"===== tail {path} =====", flush=True)
    print(text[-max_chars:], flush=True)


def run_logged(cmd: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_path, "w", encoding="utf-8") as out, open(stderr_path, "w", encoding="utf-8") as err:
        try:
            subprocess.run(cmd, stdout=out, stderr=err, check=True)
        except subprocess.CalledProcessError:
            print(f"Command failed: {' '.join(cmd)}", flush=True)
            _print_log_tail(stdout_path)
            _print_log_tail(stderr_path)
            raise
"""
        ),
        make_code_cell(
            """import os
import subprocess
import sys

subprocess.run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "kaggle==2.2.3",
    "kagglesdk==0.1.31",
    "protobuf<4",
    "tfds",
    "apache_beam",
    "mlcroissant",
], check=True)
kaggle_help = subprocess.run(
    ["kaggle", "kernels", "output", "--help"],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
).stdout
if "--page-size" not in kaggle_help:
    raise RuntimeError("Installed Kaggle CLI does not support kernels output --page-size")
subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)
os.environ["PATH"] += ":/root/.local/bin"
"""
        ),
        make_code_cell(
            """import os
import shutil
import subprocess
import sys
from pathlib import Path

download_dir = Path(CONFIG.get("dataset_download_dir", "/tmp/shortcut_dataset"))
download_dir.mkdir(parents=True, exist_ok=True)
kaggle_cli = shutil.which("kaggle")
if kaggle_cli:
    dataset_cmd = [kaggle_cli, "datasets", "download"]
else:
    dataset_cmd = [sys.executable, "-m", "kaggle.cli", "datasets", "download"]
subprocess.run([
    *dataset_cmd,
    "-d",
    CONFIG["dataset_ref"],
    "-p",
    str(download_dir),
    "--unzip",
], check=True)

tfds_source = download_dir / "tensorflow_datasets"
if tfds_source.exists():
    tfds_target = Path("/root/tensorflow_datasets")
    if tfds_target.exists():
        shutil.rmtree(tfds_target)
    shutil.copytree(tfds_source, tfds_target)
else:
    tfds_target = Path("/root/tensorflow_datasets")

has_built_celebahq = any(tfds_target.glob("celebahq256/*/dataset_info.json"))

tfds_builders_root = Path(CONFIG.get("tfds_builders_root", "/tmp/tfds_builders"))
if has_built_celebahq:
    print(f"Using prebuilt TFDS from {tfds_target}")
else:
    if not tfds_builders_root.exists():
        subprocess.run(["git", "clone", "https://github.com/kvfrans/tfds_builders.git", str(tfds_builders_root)], check=True)
    os.chdir(tfds_builders_root / "celebahq256")
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    subprocess.run(["tfds", "build"], check=True, env=env)
"""
        ),
        make_code_cell(
            """import os
import shutil
import subprocess
import sys
from pathlib import Path

repo_dir = Path(CONFIG.get("runtime_repo_dir", "/tmp/shortcut-models"))
if not repo_dir.exists():
    subprocess.run(["git", "clone", CONFIG["repo_url"], str(repo_dir)], check=True)
os.chdir(repo_dir)
subprocess.run(["git", "fetch", "--all"], check=True)
subprocess.run(["git", "checkout", CONFIG["branch"]], check=True)
subprocess.run(["git", "pull"], check=True)
if CONFIG.get("repo_commit"):
    subprocess.run(["git", "checkout", CONFIG["repo_commit"]], check=True)
run_logged(["uv", "sync"], Path("sync_out.txt"), Path("sync_err.txt"))
subprocess.run(["uv", "cache", "clean"], check=False)
subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
shutil.rmtree(Path.home() / ".cache" / "pip", ignore_errors=True)

source_data = Path(CONFIG.get("dataset_download_dir", "/tmp/shortcut_dataset")) / "data"
if source_data.exists():
    target_data = repo_dir / "data"
    target_data.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_data, target_data, dirs_exist_ok=True)
"""
        ),
        make_code_cell(
            """import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

base_dir = Path("/kaggle/working/gmm_tide_fm") / RUN_NAME
diag_dir = base_dir / "diagnostics"
diag_dir.mkdir(parents=True, exist_ok=True)
resume_kernel_ref = CONFIG.get("resume_kernel_ref", "")
resume_manifest_path = diag_dir / "resume_manifest.json"


def _run_kaggle_output(kernel_ref: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle_cli = shutil.which("kaggle")
    if kaggle_cli:
        cmd = [kaggle_cli, "kernels", "output"]
    else:
        cmd = [sys.executable, "-m", "kaggle.cli", "kernels", "output"]
    args = [*cmd, kernel_ref, "-p", str(output_dir), "-o", "-q"]
    file_pattern = _resume_file_pattern()
    if file_pattern:
        args.extend(["--file-pattern", file_pattern])
    subprocess.run(args, check=True)


def _resume_file_pattern() -> str:
    patterns = [
        r".*gmm_stats\\.npz$",
        r".*gmm_router\\.pkl$",
        r".*diagnostics/(gmm_metrics\\.json|router_metrics_summary\\.json|train_metrics_summary\\.json)$",
    ]
    require_checkpoint = bool(
        CONFIG.get(
            "resume_require_checkpoint",
            str(CONFIG.get("execution_mode", "train")).strip().lower() != "router_geometry_audit",
        )
    )
    if require_checkpoint:
        patterns.append(r".*ckpts/.*\.pkl$")
    return "|".join(patterns)


def _cleanup_kaggle_config() -> None:
    config_dir = os.environ.pop("KAGGLE_CONFIG_DIR", "")
    os.environ.pop("KAGGLE_USERNAME", None)
    os.environ.pop("KAGGLE_KEY", None)
    os.environ.pop("KAGGLE_API_TOKEN", None)
    if config_dir:
        shutil.rmtree(config_dir, ignore_errors=True)


def _candidate_roots(download_dir: Path) -> list[Path]:
    roots = []
    copied_root = Path(CONFIG.get("resume_copy_to", "/kaggle/working"))
    if bool(CONFIG.get("resume_copy_full_output", False)) and copied_root.exists():
        roots.append(copied_root)
    if download_dir.exists():
        roots.append(download_dir)
    input_root = Path("/kaggle/input")
    if input_root.exists():
        roots.extend(sorted(p for p in input_root.iterdir() if p.is_dir()))
    return roots


def _prefer_path(paths: list[Path], run_name: str = "") -> Path | None:
    if not paths:
        return None
    run_name = str(run_name or "")
    paths = sorted(paths, key=lambda p: (run_name not in str(p), len(str(p)), str(p)))
    return paths[0]


def _find_named_file(roots: list[Path], filename: str, run_name: str = "") -> Path | None:
    matches = []
    for root in roots:
        matches.extend(p for p in root.rglob(filename) if p.is_file())
    return _prefer_path(matches, run_name=run_name)


def _checkpoint_step(path: Path) -> int:
    numbers = [int(x) for x in re.findall(r"\\d+", str(path))]
    return max(numbers) if numbers else -1


def _find_checkpoint(roots: list[Path], run_name: str = "", target_step: int = 0) -> Path | None:
    candidates = []
    for root in roots:
        for ckpt_root in root.rglob("ckpts"):
            if not ckpt_root.is_dir():
                continue
            for path in ckpt_root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix in {".json", ".jsonl", ".csv", ".txt", ".png", ".jpg", ".jpeg", ".npz"}:
                    continue
                if path.stat().st_size <= 1024:
                    continue
                candidates.append(path)
    if not candidates:
        return None
    run_name = str(run_name or "")
    target_step = int(target_step or 0)
    if target_step > 0:
        candidates = sorted(
            candidates,
            key=lambda p: (
                run_name not in str(p),
                abs(_checkpoint_step(p) - target_step),
                -_checkpoint_step(p),
                str(p),
            ),
        )
    else:
        candidates = sorted(
            candidates,
            key=lambda p: (
                run_name not in str(p),
                -_checkpoint_step(p),
                str(p),
            ),
        )
    return candidates[0]


def _copy_full_output_to_working(download_dir: Path) -> tuple[str, list[str]]:
    target_root = Path(CONFIG.get("resume_copy_to", "/kaggle/working"))
    target_root.mkdir(parents=True, exist_ok=True)
    skipped = []
    allow_code_overwrite = bool(CONFIG.get("resume_overwrite_code", False))
    for source in sorted(download_dir.iterdir()):
        if source.name in {".kaggle_config", "resume_output"}:
            skipped.append(source.name)
            continue
        if source.name == "shortcut-models" and not allow_code_overwrite:
            skipped.append(source.name)
            continue
        target = target_root / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return str(target_root), skipped


def _copy_resume_checkpoint(checkpoint_source: Path) -> Path:
    checkpoint_target = base_dir / "resume_checkpoint.pkl"
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_source, checkpoint_target)
    return checkpoint_target


def _cleanup_resume_checkpoints(roots: list[Path]) -> list[str]:
    removed = []
    for root in roots:
        if not root.exists():
            continue
        for ckpt_root in root.rglob("ckpts"):
            if not ckpt_root.is_dir():
                continue
            try:
                shutil.rmtree(ckpt_root)
                removed.append(str(ckpt_root))
            except OSError:
                pass
    return removed


if resume_kernel_ref:
    download_dir = Path(CONFIG.get("resume_output_dir", "/tmp/resume_output"))
    if bool(CONFIG.get("resume_download_output", True)):
        if bool(CONFIG.get("resume_output_preloaded", False)):
            if not download_dir.exists() or not any(path.is_file() for path in download_dir.rglob("*")):
                raise FileNotFoundError(f"KJO cross-account output cell did not populate {download_dir}")
        else:
            try:
                _run_kaggle_output(resume_kernel_ref, download_dir)
            finally:
                _cleanup_kaggle_config()
    copied_to = ""
    skipped_copy_entries = []
    if bool(CONFIG.get("resume_copy_full_output", False)) and download_dir.exists():
        copied_to, skipped_copy_entries = _copy_full_output_to_working(download_dir)

    roots = _candidate_roots(download_dir)
    source_run_name = CONFIG.get("resume_run_name") or CONFIG.get("source_run_name") or ""
    gmm_stats_source = _find_named_file(roots, "gmm_stats.npz", run_name=source_run_name)
    router_source = _find_named_file(roots, "gmm_router.pkl", run_name=source_run_name)
    require_checkpoint = bool(
        CONFIG.get(
            "resume_require_checkpoint",
            str(CONFIG.get("execution_mode", "train")).strip().lower() != "router_geometry_audit",
        )
    )
    checkpoint_source = None
    if require_checkpoint:
        checkpoint_source = _find_checkpoint(
            roots,
            run_name=source_run_name,
            target_step=int(CONFIG.get("resume_checkpoint_step", 0) or 0),
        )

    if bool(CONFIG.get("resume_reuse_gmm_router", True)):
        if gmm_stats_source is None or router_source is None:
            raise FileNotFoundError(
                f"Could not find gmm_stats.npz/router in previous output roots: {[str(p) for p in roots]}"
            )
        shutil.copy2(gmm_stats_source, base_dir / "gmm_stats.npz")
        shutil.copy2(router_source, base_dir / "gmm_router.pkl")

    if require_checkpoint and checkpoint_source is None:
        raise FileNotFoundError(f"Could not find a checkpoint under previous output roots: {[str(p) for p in roots]}")
    checkpoint_target = _copy_resume_checkpoint(checkpoint_source) if checkpoint_source is not None else None
    removed_checkpoint_roots = _cleanup_resume_checkpoints(roots)
    if bool(CONFIG.get("resume_cleanup_download_dir", True)) and download_dir.exists():
        shutil.rmtree(download_dir, ignore_errors=True)

    manifest = {
        "resume_kernel_ref": resume_kernel_ref,
        "resume_run_name": source_run_name,
        "download_dir": str(download_dir),
        "copied_full_output_to": copied_to,
        "skipped_copy_entries": skipped_copy_entries,
        "gmm_stats_source": str(gmm_stats_source) if gmm_stats_source else "",
        "router_source": str(router_source) if router_source else "",
        "require_checkpoint": require_checkpoint,
        "load_dir": str(checkpoint_target) if checkpoint_target else "",
        "checkpoint_step_guess": _checkpoint_step(checkpoint_source) if checkpoint_source else -1,
        "checkpoint_source": str(checkpoint_source) if checkpoint_source else "",
        "removed_checkpoint_roots": removed_checkpoint_roots,
        "cleaned_download_dir": bool(CONFIG.get("resume_cleanup_download_dir", True)),
        "roots": [str(p) for p in roots],
    }
    resume_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
else:
    print("No resume_kernel_ref configured; running fresh GMM/router/FM pipeline.")
"""
        ),
        make_code_cell(
            """import json
import os
import subprocess
from pathlib import Path

base_dir = Path("/kaggle/working/gmm_tide_fm") / RUN_NAME
diag_dir = base_dir / "diagnostics"
diag_dir.mkdir(parents=True, exist_ok=True)
gmm_stats_path = base_dir / "gmm_stats.npz"
gmm_latent_cache_path = Path(CONFIG.get("gmm_latent_cache_path", f"/tmp/{RUN_NAME}_gmm_latents.dat"))
model_train_type = str(CONFIG.get("model_train_type", "gmm-tide"))

if model_train_type != "gmm-tide":
    print(f"Skipping GMM prep for model_train_type={model_train_type}")
elif bool(CONFIG.get("resume_reuse_gmm_router", True)) and gmm_stats_path.exists():
    print(f"Using resumed GMM stats at {gmm_stats_path}")
else:
    prep_cmd = [
        "uv", "run", "data_prep.py",
        "--dataset_name", CONFIG["dataset_name"],
        "--tfds_data_dir", CONFIG["tfds_data_dir"],
        "--batch_size", str(CONFIG["batch_size"]),
        "--gmm_save_path", str(gmm_stats_path),
        "--gmm_latent_cache_path", str(gmm_latent_cache_path),
        "--gmm_num_modes", str(CONFIG["gmm_num_modes"]),
        "--gmm_fit_samples", str(CONFIG["gmm_fit_samples"]),
        "--gmm_valid_samples", str(CONFIG["gmm_valid_samples"]),
        "--gmm_em_iters", str(CONFIG["gmm_em_iters"]),
        "--gmm_em_restarts", str(CONFIG["gmm_em_restarts"]),
        "--gmm_init_seed", str(CONFIG["gmm_init_seed"]),
        "--gmm_standardize_data", str(CONFIG["gmm_standardize_data"]),
        "--gmm_transform", str(CONFIG.get("gmm_transform", "auto")),
        "--gmm_standardize_eps", str(CONFIG["gmm_standardize_eps"]),
        "--gmm_fit_data_mode", CONFIG.get("gmm_fit_data_mode", "x1"),
        "--gmm_mix_x1_prob", str(CONFIG.get("gmm_mix_x1_prob", 0.5)),
        "--gmm_continue_em_iters", str(CONFIG.get("gmm_continue_em_iters", 0)),
        "--gmm_mix_seed", str(CONFIG.get("gmm_mix_seed", 0)),
        "--gmm_pi_prior_type", CONFIG["gmm_pi_prior_type"],
        "--gmm_pi_prior_strength", str(CONFIG["gmm_pi_prior_strength"]),
        "--gmm_pi_kl_steps", str(CONFIG["gmm_pi_kl_steps"]),
        "--gmm_pi_kl_lr", str(CONFIG["gmm_pi_kl_lr"]),
        "--gmm_var_prior_type", CONFIG["gmm_var_prior_type"],
        "--gmm_var_prior_strength", str(CONFIG["gmm_var_prior_strength"]),
        "--gmm_var_prior_target_var", str(CONFIG["gmm_var_prior_target_var"]),
        "--gmm_min_std", str(CONFIG["gmm_min_std"]),
        "--gmm_min_std_data_frac", str(CONFIG["gmm_min_std_data_frac"]),
        "--gmm_kmeanspp_init", str(CONFIG["gmm_kmeanspp_init"]),
        "--gmm_init_strategy", str(CONFIG.get("gmm_init_strategy", "auto")),
        "--gmm_init_warmup_iters", str(CONFIG.get("gmm_init_warmup_iters", 0)),
        "--gmm_init_pca_dims", str(CONFIG.get("gmm_init_pca_dims", 16)),
        "--gmm_init_pca_max_samples", str(CONFIG.get("gmm_init_pca_max_samples", 2048)),
        "--gmm_em_chunk_size", str(CONFIG["gmm_em_chunk_size"]),
        "--gmm_keep_latent_cache", str(CONFIG["gmm_keep_latent_cache"]),
        "--metrics_output_path", str(diag_dir / "gmm_metrics.json"),
        "--gmm_em_metrics_output_path", str(diag_dir / "gmm_em_metrics.jsonl"),
        "--wandb.name", f"prep_{RUN_NAME}",
        f"--wandb.offline={not bool(os.environ.get('WANDB_API_KEY'))}",
    ]
    run_logged(prep_cmd, diag_dir / "gmm_prep_stdout.txt", diag_dir / "gmm_prep_stderr.txt")
"""
        ),
        make_code_cell(
            """import os
import subprocess
from pathlib import Path

base_dir = Path("/kaggle/working/gmm_tide_fm") / RUN_NAME
diag_dir = base_dir / "diagnostics"
router_path = base_dir / "gmm_router.pkl"
model_train_type = str(CONFIG.get("model_train_type", "gmm-tide"))

if model_train_type != "gmm-tide":
    print(f"Skipping router training for model_train_type={model_train_type}")
elif bool(CONFIG.get("resume_reuse_gmm_router", True)) and router_path.exists():
    print(f"Using resumed router at {router_path}")
else:
    router_cmd = [
        "uv", "run", "train_gmm_router.py",
        "--dataset_name", CONFIG["dataset_name"],
        "--tfds_data_dir", CONFIG["tfds_data_dir"],
        "--batch_size", str(CONFIG["batch_size"]),
        "--gmm_stats_path", str(base_dir / "gmm_stats.npz"),
        "--router_save_path", str(router_path),
        "--router_train_data_mode", CONFIG["router_train_data_mode"],
        "--router_mix_x1_prob", str(CONFIG["router_mix_x1_prob"]),
        "--router_bridge_alpha", str(CONFIG.get("router_bridge_alpha", 2.0)),
        "--router_bridge_beta", str(CONFIG.get("router_bridge_beta", 2.0)),
        "--router_target_type", CONFIG["router_target_type"],
        "--router_target_temperature", str(CONFIG.get("router_target_temperature", 1.0)),
        "--router_entropy_floor", str(CONFIG.get("router_entropy_floor", 0.0)),
        "--router_entropy_floor_weight", str(CONFIG.get("router_entropy_floor_weight", 0.0)),
        "--router_max_steps", str(CONFIG["router_max_steps"]),
        "--router_log_interval", "100",
        "--router_valid_interval", str(CONFIG["router_valid_interval"]),
        "--router_valid_batches", str(CONFIG["router_valid_batches"]),
        "--router_lr", str(CONFIG["router_lr"]),
        "--router_weight_decay", str(CONFIG["router_weight_decay"]),
        "--router_hidden_channels", str(CONFIG["router_hidden_channels"]),
        "--router_mlp_hidden_size", str(CONFIG["router_mlp_hidden_size"]),
        "--router_depth", str(CONFIG["router_depth"]),
        "--router_dropout_rate", str(CONFIG.get("router_dropout_rate", 0.0)),
        "--router_norm_type", str(CONFIG.get("router_norm_type", "none")),
        f"--router_save_best={bool(CONFIG['router_save_best'])}",
        "--metrics_output_path", str(diag_dir / "router_metrics.jsonl"),
        "--wandb.name", f"router_{RUN_NAME}",
        f"--wandb.offline={not bool(os.environ.get('WANDB_API_KEY'))}",
    ]
    run_logged(router_cmd, diag_dir / "router_stdout.txt", diag_dir / "router_stderr.txt")
"""
        ),
        make_code_cell(
            """import json
import os
import subprocess
from pathlib import Path

base_dir = Path("/kaggle/working/gmm_tide_fm") / RUN_NAME
diag_dir = base_dir / "diagnostics"
ckpt_root = Path("/kaggle/working/ckpts")
ckpt_path = ckpt_root / f"{RUN_NAME}.pkl"
diag_dir.mkdir(parents=True, exist_ok=True)
ckpt_root.mkdir(parents=True, exist_ok=True)
resume_manifest_path = diag_dir / "resume_manifest.json"
execution_mode = str(CONFIG.get("execution_mode", "train")).strip().lower()
if execution_mode not in {"train", "fid_repeats", "trajectory_eval", "router_geometry_audit"}:
    raise ValueError(f"Unknown execution_mode={execution_mode!r}")
if execution_mode == "fid_repeats" and not resume_manifest_path.exists():
    raise ValueError("execution_mode=fid_repeats requires a resumed checkpoint")
if execution_mode == "trajectory_eval" and not resume_manifest_path.exists():
    raise ValueError("execution_mode=trajectory_eval requires a resumed checkpoint")
if execution_mode == "router_geometry_audit" and not resume_manifest_path.exists():
    raise ValueError("execution_mode=router_geometry_audit requires resumed GMM/router artifacts")


def _effective_train_max_steps() -> tuple[int, dict]:
    configured_max_steps = int(CONFIG["train_max_steps"])
    target_step_abs = int(CONFIG.get("train_target_step_abs", 0) or 0)
    resume_start_step = int(
        CONFIG.get("train_resume_start_step", 0)
        or CONFIG.get("resume_expected_checkpoint_step", 0)
        or 0
    )
    has_resume = resume_manifest_path.exists()
    if target_step_abs <= 0:
        return configured_max_steps, {
            "configured_train_max_steps": configured_max_steps,
            "effective_train_max_steps": configured_max_steps,
            "resume_manifest_exists": has_resume,
            "train_resume_start_step": resume_start_step,
            "train_target_step_abs": 0,
        }
    if has_resume and resume_start_step <= 0:
        raise ValueError(
            "train_target_step_abs was set for a resume run, but neither "
            "train_resume_start_step nor resume_expected_checkpoint_step is configured."
        )
    effective = target_step_abs - resume_start_step if has_resume else target_step_abs
    if effective <= 0:
        raise ValueError(
            f"Resolved non-positive max_steps={effective} from "
            f"train_target_step_abs={target_step_abs} and train_resume_start_step={resume_start_step}."
        )
    return effective, {
        "configured_train_max_steps": configured_max_steps,
        "effective_train_max_steps": effective,
        "resume_manifest_exists": has_resume,
        "train_resume_start_step": resume_start_step if has_resume else 0,
        "train_target_step_abs": target_step_abs,
    }


if execution_mode == "train":
    effective_train_max_steps, train_budget_summary = _effective_train_max_steps()
else:
    effective_train_max_steps = 0
    train_budget_summary = {
        "execution_mode": execution_mode,
        "effective_train_max_steps": 0,
        "resume_manifest_exists": resume_manifest_path.exists(),
        "eval_fid_seeds": str(CONFIG.get("eval_fid_seeds", "42")),
        "eval_fid_generations": int(CONFIG.get("eval_fid_generations", 50048)),
        "trajectory_seed": int(CONFIG.get("trajectory_seed", 42)),
        "trajectory_num_samples": int(CONFIG.get("trajectory_num_samples", 64)),
        "trajectory_timesteps": int(CONFIG.get("trajectory_timesteps", 128)),
    }
print("TRAIN_BUDGET_SUMMARY " + json.dumps(train_budget_summary, sort_keys=True))

if execution_mode == "router_geometry_audit":
    audit_cmd = [
        "uv", "run", "python", "/tmp/audit_gmm_tide_router_geometry.py",
        "--dataset-name", CONFIG["dataset_name"],
        "--tfds-data-dir", CONFIG["tfds_data_dir"],
        "--gmm-stats-path", str(base_dir / "gmm_stats.npz"),
        "--router-path", str(base_dir / "gmm_router.pkl"),
        "--source-mode", str(CONFIG.get("gmm_router_source_mode", "weighted")),
        "--batch-size", str(CONFIG.get("audit_batch_size", 64)),
        "--num-batches", str(CONFIG.get("audit_num_batches", 32)),
        "--seed", str(CONFIG.get("audit_seed", 0)),
        "--topk", str(CONFIG["gmm_router_topk"]),
        "--temperature", str(CONFIG["gmm_router_temperature"]),
        "--bridge-lambdas", str(CONFIG.get("audit_bridge_lambdas", "0,0.25,0.5,0.75,1")),
        "--noise-scales", str(CONFIG.get("audit_noise_scales", "0.01,0.03,0.05,0.1")),
        "--output-dir", str(diag_dir),
    ]
    run_logged(audit_cmd, diag_dir / "router_geometry_audit_stdout.txt", diag_dir / "router_geometry_audit_stderr.txt")
else:
    train_cmd = [

        "uv", "run", "train.py",
    "--model.hidden_size", "768",
    "--model.patch_size", "2",
    "--model.depth", "12",
    "--model.num_heads", "12",
    "--model.mlp_ratio", "4",
    "--model.train_type", str(CONFIG.get("model_train_type", "gmm-tide")),
    "--model.cfg_scale", "0",
    "--model.class_dropout_prob", "1",
    "--model.num_classes", "1",
    "--model.denoise_timesteps", "128",
    "--batch_size", str(CONFIG["train_batch_size"]),
    "--seed", str(CONFIG.get("training_seed", 0)),
    "--dataset_name", CONFIG["dataset_name"],
    "--tfds_data_dir", CONFIG["tfds_data_dir"],
    "--fid_stats", "data/celeba256_fidstats_ours.npz",
    "--wandb.name", RUN_NAME,
    "--model.lr", str(CONFIG.get("model_lr", 1e-4)),
    "--model.warmup", str(CONFIG.get("model_warmup", 0)),
    "--model.use_cosine", str(CONFIG.get("model_use_cosine", 0)),
    "--model.beta1", str(CONFIG.get("model_beta1", 0.9)),
    "--model.beta2", str(CONFIG.get("model_beta2", 0.999)),
    "--model.weight_decay", str(CONFIG.get("model_weight_decay", 0.01)),
    "--model.t_sampling", str(CONFIG.get("model_t_sampling", "discrete-dt")),
    "--model.t_beta_alpha", str(CONFIG.get("model_t_beta_alpha", 1.0)),
    "--model.t_beta_beta", str(CONFIG.get("model_t_beta_beta", 1.0)),
    "--model.eval_ode_schedule", str(CONFIG.get("model_eval_ode_schedule", "uniform")),
    "--model.eval_ode_power", str(CONFIG.get("model_eval_ode_power", 1.0)),
    "--model.gmm_stats_path", str(base_dir / "gmm_stats.npz"),
    "--model.gmm_router_path", str(base_dir / "gmm_router.pkl"),
    "--model.gmm_router_topk", str(CONFIG["gmm_router_topk"]),
    "--model.gmm_router_temperature", str(CONFIG["gmm_router_temperature"]),
    "--model.gmm_router_source_mode", str(CONFIG.get("gmm_router_source_mode", "weighted")),
    "--model.gmm_router_gradient_mode", str(CONFIG.get("gmm_router_gradient_mode", "topk")),
    "--model.gmm_router_gumbel_tau", str(CONFIG.get("gmm_router_gumbel_tau", 1.0)),
    "--model.gmm_router_update_policy", str(CONFIG.get("gmm_router_update_policy", "frozen")),
    "--model.gmm_router_eval_use_ema", str(CONFIG.get("gmm_router_eval_use_ema", 0)),
    "--model.gmm_router_lr", str(CONFIG.get("gmm_router_lr", 3e-5)),
    "--model.gmm_router_weight_decay", str(CONFIG.get("gmm_router_weight_decay", 1e-4)),
    "--model.gmm_router_distill_weight", str(CONFIG.get("gmm_router_distill_weight", 1.0)),
    "--model.gmm_router_tide_distill_weight", str(CONFIG.get("gmm_router_tide_distill_weight", 0.0)),
    "--model.gmm_router_usage_weight", str(CONFIG.get("gmm_router_usage_weight", 0.0)),
    "--model.gmm_router_entropy_weight", str(CONFIG.get("gmm_router_entropy_weight", 0.0)),
    "--model.gmm_router_geometry_weight", str(CONFIG.get("gmm_router_geometry_weight", 0.0)),
    "--model.gmm_source_shift_mean", str(int(CONFIG.get("gmm_source_shift_mean", 0))),
    "--model.gmm_cond_channels", str(CONFIG["model_gmm_cond_channels"]),
    "--eval_fid_timesteps", CONFIG["eval_fid_timesteps"],
    f"--wandb.offline={not bool(os.environ.get('WANDB_API_KEY'))}",
    ]
    if str(CONFIG.get("model_train_type", "gmm-tide")) in ("gmm-centered", "gmm-tide"):
        train_cmd.extend([
            "--model.gmm_source_center_scale",
            str(CONFIG.get("gmm_source_center_scale", 1.0)),
        ])
    routing_policy = str(CONFIG.get("gmm_router_routing_policy", "router"))
    if routing_policy != "router":
        train_cmd.extend(["--model.gmm_router_routing_policy", routing_policy])
    if execution_mode == "train":
        train_cmd.extend([
        "--max_steps", str(effective_train_max_steps),
        "--eval_interval", str(CONFIG["train_eval_interval"]),
        "--log_interval", str(CONFIG["train_log_interval"]),
        "--save_dir", str(ckpt_path),
        "--metrics_output_path", str(diag_dir / "train_metrics.jsonl"),
        ])
    elif execution_mode == "fid_repeats":
        train_cmd.extend([
        "--mode", "eval-fid",
        "--eval_fid_seeds", str(CONFIG.get("eval_fid_seeds", "42")),
        "--eval_fid_generations", str(CONFIG.get("eval_fid_generations", 50048)),
        "--metrics_output_path", str(diag_dir / "fid_repeat_metrics.jsonl"),
        ])
    else:
        train_cmd.extend([
        "--mode", "eval-trajectory",
        "--trajectory_seed", str(CONFIG.get("trajectory_seed", 42)),
        "--trajectory_num_samples", str(CONFIG.get("trajectory_num_samples", 64)),
        "--trajectory_timesteps", str(CONFIG.get("trajectory_timesteps", 128)),
        "--trajectory_save_steps", str(CONFIG.get("trajectory_save_steps", "")),
        "--trajectory_decode_samples", str(CONFIG.get("trajectory_decode_samples", 8)),
        "--trajectory_output_path", str(diag_dir / "denoising_trajectory.npz"),
        "--metrics_output_path", str(diag_dir / "trajectory_metrics.jsonl"),
        ])
    if resume_manifest_path.exists():
        resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        load_dir = str(resume_manifest.get("load_dir", ""))
        if load_dir:
            train_cmd.extend([
                "--load_dir", load_dir,
                "--reset_step_on_load", str(CONFIG.get("reset_step_on_load", 0)),
            ])
            if bool(CONFIG.get("delete_load_dir_after_load", True)):
                train_cmd.extend(["--delete_load_dir_after_load", "1"])
    if execution_mode == "train" and CONFIG.get("save_interval"):
        train_cmd.extend(["--save_interval", str(CONFIG["save_interval"])])
    if execution_mode == "train" and CONFIG.get("save_slim_checkpoint", 1):
        train_cmd.extend(["--save_slim_checkpoint", "1"])
    log_prefix = {
        "train": "train",
        "fid_repeats": "fid_repeat_eval",
        "trajectory_eval": "trajectory_eval",
    }.get(execution_mode, execution_mode)
    run_logged(train_cmd, diag_dir / f"{log_prefix}_stdout.txt", diag_dir / f"{log_prefix}_stderr.txt")
"""
        ),
        make_code_cell(
            """import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

base_dir = Path("/kaggle/working/gmm_tide_fm") / RUN_NAME
diag_dir = base_dir / "diagnostics"
working_dir = Path("/kaggle/working")


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def top_level_sizes(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    return {
        item.name: path_size_bytes(item)
        for item in sorted(root.iterdir(), key=lambda p: p.name)
    }


before = top_level_sizes(working_dir)
removed = []
os.chdir(working_dir)
for raw_path in [
    CONFIG.get("runtime_repo_dir", "/tmp/shortcut-models"),
    CONFIG.get("dataset_download_dir", "/tmp/shortcut_dataset"),
    CONFIG.get("tfds_builders_root", "/tmp/tfds_builders"),
    CONFIG.get("resume_output_dir", "/tmp/resume_output"),
    CONFIG.get("gmm_latent_cache_path", f"/tmp/{RUN_NAME}_gmm_latents.dat"),
    "/tmp/.kaggle_config",
    "/tmp/.kaggle_source_owner",
    "/kaggle/working/shortcut-models",
    "/kaggle/working/shortcut_dataset",
    "/kaggle/working/tfds_builders",
    "/kaggle/working/.kaggle_config",
]:
    path = Path(raw_path)
    if not path.exists():
        continue
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))
    except OSError as exc:
        removed.append(f"{path}: {exc}")

for path in base_dir.rglob("gmm_latents.dat"):
    try:
        path.unlink()
        removed.append(str(path))
    except OSError as exc:
        removed.append(f"{path}: {exc}")

if str(CONFIG.get("execution_mode", "train")).strip().lower() in {"fid_repeats", "trajectory_eval", "router_geometry_audit"}:
    for path in [
        base_dir / "gmm_stats.npz",
        base_dir / "gmm_router.pkl",
        base_dir / "resume_checkpoint.pkl",
        Path("/kaggle/working/ckpts"),
    ]:
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))
        except OSError as exc:
            removed.append(f"{path}: {exc}")

subprocess.run(["uv", "cache", "clean"], check=False)
subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
shutil.rmtree(Path.home() / ".cache" / "pip", ignore_errors=True)
shutil.rmtree(Path.home() / ".cache" / "uv", ignore_errors=True)

after = top_level_sizes(working_dir)
summary = {
    "before_top_level_bytes": before,
    "after_top_level_bytes": after,
    "after_total_bytes": sum(after.values()),
    "after_total_gib": sum(after.values()) / (1024 ** 3),
    "removed": removed,
    "kept_expected": [str(base_dir)],
}
diag_dir.mkdir(parents=True, exist_ok=True)
(diag_dir / "output_cleanup_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
"""
        ),
    ]
    insert_index = 3
    if cross_account_output_source:
        cells.insert(insert_index, make_code_cell(cross_account_output_source))
        insert_index += 1
    if router_geometry_audit_script_source:
        audit_source_literal = json.dumps(router_geometry_audit_script_source)
        cells.insert(
            insert_index,
            make_code_cell(
                "from pathlib import Path\n"
                f"Path('/tmp/audit_gmm_tide_router_geometry.py').write_text({audit_source_literal}, encoding='utf-8')\n"
            ),
        )
    if accelerator_probe_source:
        cells.insert(min(6, len(cells)), make_code_cell(accelerator_probe_source))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _stage_job_legacy(
    *,
    owner: str,
    config: dict[str, Any],
    staging_root: Path,
    accelerator: str,
    wandb_api_key: str,
    kaggle_credential: dict[str, str] | None = None,
) -> tuple[Path, str]:
    accelerator = normalize_accelerator(accelerator)
    kind = accelerator_kind(accelerator)
    is_tpu = kind == "tpu"
    is_gpu = kind == "gpu"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    slug = slugify(f"{config['run_name']}-{owner}-{timestamp}", max_length=48)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / slug
    suffix = 2
    while staging_dir.exists():
        staging_dir = staging_root / f"{slug}-{suffix}"
        suffix += 1
    staging_dir.mkdir(parents=True, exist_ok=False)
    accelerator_probe_source = render_accelerator_probe_source(staging_dir, kind)
    cross_account_output_source = render_cross_account_output_source(staging_dir, config, owner)
    if cross_account_output_source:
        config["resume_output_preloaded"] = True
        config["resume_runtime_owner"] = owner
    router_geometry_audit_script_source = ""
    if str(config.get("execution_mode", "train")).strip().lower() == "router_geometry_audit":
        audit_script_path = Path("scripts/audit_gmm_tide_router_geometry.py")
        if not audit_script_path.exists():
            raise FileNotFoundError(f"Missing router geometry audit script: {audit_script_path}")
        router_geometry_audit_script_source = audit_script_path.read_text(encoding="utf-8")

    notebook_name = f"{slug}.ipynb"
    notebook_path = staging_dir / notebook_name
    notebook_path.write_text(
        json.dumps(
            make_notebook(
                config,
                wandb_api_key=wandb_api_key,
                kaggle_credential=kaggle_credential,
                accelerator_probe_source=accelerator_probe_source,
                cross_account_output_source=cross_account_output_source,
                router_geometry_audit_script_source=router_geometry_audit_script_source,
            ),
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    kernel_sources = list(config.get("kernel_sources", []))
    attach_resume_source = bool(config.get("resume_attach_kernel_source", not bool(config.get("resume_download_output", True))))
    if config.get("resume_kernel_ref") and attach_resume_source:
        kernel_sources.append(config["resume_kernel_ref"])
    kernel_sources = sorted(set(kernel_sources))
    metadata = {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": is_gpu,
        "enable_tpu": is_tpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }
    (staging_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (staging_dir / "gmm_tide_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return staging_dir, metadata["id"]


def _stage_job_blueprint(
    *,
    owner: str,
    config: dict[str, Any],
    staging_root: Path,
    accelerator: str,
    wandb_api_key: str,
    kaggle_credential: dict[str, str] | None = None,
    instrumentation_mode: str = "none",
    runtime_dataset_source: str = "",
    runtime_version: str = "",
    runtime_module_sha256: str = "",
) -> tuple[Path, str]:
    accelerator = normalize_accelerator(accelerator)
    kind = accelerator_kind(accelerator)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    slug = slugify(f"{config['run_name']}-{owner}-{timestamp}", max_length=48)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / slug
    suffix = 2
    while staging_dir.exists():
        staging_dir = staging_root / f"{slug}-{suffix}"
        suffix += 1
    slug = staging_dir.name
    work_dir = staging_root / f".{slug}.blueprint-work"
    suffix = 2
    while work_dir.exists():
        work_dir = staging_root / f".{slug}.blueprint-work-{suffix}"
        suffix += 1
    work_dir.mkdir(parents=True, exist_ok=False)
    timeline_path = work_dir / "operation_timeline.jsonl"

    semantic_config = dict(config)
    source_kernel_id = str(config.get("resume_kernel_ref") or "")
    if source_kernel_id:
        semantic_config["resume_kernel_ref"] = BLUEPRINT_SOURCE_KERNEL_REF
        semantic_config["resume_runtime_owner"] = BLUEPRINT_DESTINATION_OWNER
    semantic_config["kaggle_destination_owner"] = BLUEPRINT_DESTINATION_OWNER
    semantic_config["kaggle_destination_slug"] = BLUEPRINT_DESTINATION_SLUG
    semantic_config["kjo_staging_mode"] = "blueprint"

    try:
        accelerator_probe_source = render_accelerator_probe_source(work_dir, kind)
        cross_account_output_source = render_cross_account_output_source(
            work_dir,
            semantic_config,
            BLUEPRINT_DESTINATION_OWNER,
        )
        if cross_account_output_source:
            semantic_config["resume_output_preloaded"] = True
        router_geometry_audit_script_source = ""
        if str(config.get("execution_mode", "train")).strip().lower() == "router_geometry_audit":
            audit_script_path = Path("scripts/audit_gmm_tide_router_geometry.py")
            if not audit_script_path.exists():
                raise FileNotFoundError(f"Missing router geometry audit script: {audit_script_path}")
            router_geometry_audit_script_source = audit_script_path.read_text(encoding="utf-8")

        semantic_notebook = work_dir / "semantic_source.ipynb"
        semantic_notebook.write_text(
            json.dumps(
                make_notebook(
                    semantic_config,
                    wandb_api_key=wandb_api_key,
                    kaggle_credential=kaggle_credential,
                    accelerator_probe_source=accelerator_probe_source,
                    cross_account_output_source=cross_account_output_source,
                    router_geometry_audit_script_source=router_geometry_audit_script_source,
                ),
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        blueprint_dir = work_dir / "blueprint"
        create_command = [
            sys.executable,
            str(KAGGLE_JOB_OPS_SCRIPT),
            "create-staging-blueprint",
            "--source-notebook",
            str(semantic_notebook),
            "--out-dir",
            str(blueprint_dir),
            "--operation-timeline",
            str(timeline_path),
        ]
        if instrumentation_mode != "none":
            create_command.extend(
                [
                    "--instrument-logging",
                    "--instrumentation-mode",
                    instrumentation_mode,
                ]
            )
            if instrumentation_mode == "runtime-dataset":
                if not runtime_dataset_source or not runtime_version or not runtime_module_sha256:
                    raise ValueError(
                        "runtime-dataset blueprint staging requires runtime dataset source, version, and module SHA256"
                    )
                create_command.extend(
                    [
                        "--runtime-version",
                        runtime_version,
                        "--runtime-module-sha256",
                        runtime_module_sha256,
                    ]
                )
        create_result = run_json_command(create_command)

        kernel_sources = list(config.get("kernel_sources", []))
        attach_resume_source = bool(
            config.get("resume_attach_kernel_source", not bool(config.get("resume_download_output", True)))
        )
        if source_kernel_id and attach_resume_source:
            kernel_sources.append(source_kernel_id)
        kernel_sources = list(dict.fromkeys(str(item) for item in kernel_sources))
        dataset_sources = list(dict.fromkeys(str(item) for item in config.get("dataset_sources", [])))
        destinations_path = work_dir / "destinations.json"
        destinations_path.write_text(
            json.dumps(
                {
                    "destinations": [
                        {
                            "owner": owner,
                            "slug": slug,
                            "title": slug,
                            "accelerator": kind,
                            "submit_accelerator": "" if kind == "cpu" else accelerator,
                            "runtime_dataset_source": runtime_dataset_source,
                            "source_kernel_id": source_kernel_id,
                            "dataset_sources": dataset_sources,
                            "kernel_sources": kernel_sources,
                            "out_dir": str(staging_dir.resolve()),
                            "is_private": True,
                            "enable_internet": True,
                        }
                    ]
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        materialize_result = run_json_command(
            [
                sys.executable,
                str(KAGGLE_JOB_OPS_SCRIPT),
                "materialize-staging-blueprint",
                "--blueprint",
                str(blueprint_dir / "staging_blueprint.json"),
                "--destinations",
                str(destinations_path),
                "--out-root",
                str(staging_root),
                "--operation-timeline",
                str(timeline_path),
            ]
        )
        if len(materialize_result.get("materialized", [])) != 1:
            raise RuntimeError("Blueprint materialization did not produce exactly one destination")
        (staging_dir / "gmm_tide_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt = {
            "ok": True,
            "staging_mode": "blueprint",
            "semantic_fingerprint": create_result.get("semantic_fingerprint"),
            "create": create_result,
            "materialize": materialize_result,
            "source_kernel_id": source_kernel_id,
            "destination_kernel_id": f"{owner}/{slug}",
            "runtime_dataset_source": runtime_dataset_source,
            "instrumentation_mode": instrumentation_mode,
        }
        (staging_dir / "staging_blueprint_result.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if timeline_path.is_file():
            shutil.copy2(timeline_path, staging_dir / "operation_timeline.jsonl")
        return staging_dir, f"{owner}/{slug}"
    finally:
        for notebook_path in (work_dir / "semantic_source.ipynb", work_dir / "blueprint" / "semantic_notebook.ipynb"):
            scrub_notebook_embedded_credentials(notebook_path)
        shutil.rmtree(work_dir, ignore_errors=True)


def stage_job(
    *,
    owner: str,
    config: dict[str, Any],
    staging_root: Path,
    accelerator: str,
    wandb_api_key: str,
    kaggle_credential: dict[str, str] | None = None,
    staging_mode: str = "blueprint",
    instrumentation_mode: str = "none",
    runtime_dataset_source: str = "",
    runtime_version: str = "",
    runtime_module_sha256: str = "",
) -> tuple[Path, str]:
    if staging_mode == "legacy":
        return _stage_job_legacy(
            owner=owner,
            config=config,
            staging_root=staging_root,
            accelerator=accelerator,
            wandb_api_key=wandb_api_key,
            kaggle_credential=kaggle_credential,
        )
    if staging_mode != "blueprint":
        raise ValueError(f"Unknown staging_mode={staging_mode!r}")
    return _stage_job_blueprint(
        owner=owner,
        config=config,
        staging_root=staging_root,
        accelerator=accelerator,
        wandb_api_key=wandb_api_key,
        kaggle_credential=kaggle_credential,
        instrumentation_mode=instrumentation_mode,
        runtime_dataset_source=runtime_dataset_source,
        runtime_version=runtime_version,
        runtime_module_sha256=runtime_module_sha256,
    )


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = path.with_suffix(".md")
    lines = [
        "# GMM-TIDE FM Submit Report",
        "",
        f"- Submitted: {len(report['submitted'])}",
        f"- Failed: {len(report['failed'])}",
        f"- Not submitted: {len(report.get('not_submitted', []))}",
        f"- Job root: `{report.get('job_root', '')}`",
        f"- Notebook registry: `{report.get('notebook_registry', '')}`",
        f"- Artifact mode: `{report.get('artifact_mode', '')}`",
        f"- Retention action: `{report.get('retention_action', '')}`",
        "",
    ]
    if report.get("shared_context"):
        lines.extend(
            [
                "## Shared Context",
                "",
                f"- Output: `{report['shared_context'].get('output', '')}`",
                f"- Live status: {report['shared_context'].get('live')}",
                f"- Active by owner: `{json.dumps(report['shared_context'].get('active_by_owner', {}), sort_keys=True)}`",
                "",
            ]
        )
    lines.extend([
        "| job | owner | exec | family | gmm seed | eval seeds | eval N | resume_cred | modes | topk | source_mode | routing | route_grad | tau | router_reg | router_train | ema_eval | target_T | entropy_floor | bridge | geom_w | tide_kl_w | transform | fit_data | cont_em | init | lloyd | t_sampling | beta | eval_ode | resume | target_abs | resume_start | max_steps | save_int | source_grid | kernel | status |",
        "|---:|---|---|---|---:|---|---:|---|---:|---:|---|---|---|---:|---|---|---:|---:|---|---|---:|---:|---|---|---:|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in report["submitted"]:
        lines.append(
            f"| {row['grid_index']} | {row['owner']} | {row.get('execution_mode', 'train')} | "
            f"{row.get('candidate_family', '')} | {row.get('gmm_randomization_seed', row.get('training_seed', ''))} | "
            f"{row.get('eval_fid_seeds', '')} | {row.get('eval_fid_generations', '')} | "
            f"{row.get('notebook_kaggle_credential_owner', '')} | "
            f"{row['gmm_num_modes']} | {row['gmm_router_topk']} | "
            f"{row.get('gmm_router_source_mode', 'weighted')} | "
            f"{row.get('gmm_router_routing_policy', 'router')} | "
            f"{row.get('gmm_router_gradient_mode', 'topk')} | {row.get('gmm_router_gumbel_tau', 1.0)} | "
            f"{row.get('router_norm_type', 'none')},drop={row.get('router_dropout_rate', 0.0)} | "
            f"{row.get('router_train_data_mode', 'mix')} | "
            f"{row.get('gmm_router_eval_use_ema', 0)} | "
            f"{row.get('router_target_temperature', 1.0)} | "
            f"{row.get('router_entropy_floor', 0.0)},w={row.get('router_entropy_floor_weight', 0.0)} | "
            f"{row.get('router_bridge_alpha', 2.0)},{row.get('router_bridge_beta', 2.0)} | "
            f"{row.get('gmm_router_geometry_weight', 0.0)} | "
            f"{row.get('gmm_router_tide_distill_weight', 0.0)} | "
            f"{row.get('gmm_transform', '')} | "
            f"{row.get('gmm_fit_data_mode', '')} | {row.get('gmm_continue_em_iters', '')} | "
            f"{row.get('gmm_init_strategy', '')} | {row.get('gmm_init_warmup_iters', '')} | "
            f"{row.get('model_t_sampling', '')} | "
            f"{row.get('model_t_beta_alpha', '')},{row.get('model_t_beta_beta', '')} | "
            f"{row.get('model_eval_ode_schedule', 'uniform')},{row.get('model_eval_ode_power', 1.0)} | "
            f"{row.get('resume_kernel_ref', '')} | "
            f"{row.get('train_target_step_abs', '')} | "
            f"{row.get('train_resume_start_step', '')} | "
            f"{row.get('train_max_steps', '')} | "
            f"{row.get('save_interval', '')} | "
            f"{row['source_grid_index']} | `{row['kernel_id']}` | {row.get('kernel_status', '')} |"
        )
    if report["failed"]:
        lines.extend(["", "## Failed", "", "| job | owner | error |", "|---:|---|---|"])
        for row in report["failed"]:
            error = str(row.get("error", "")).replace("\n", "<br>")
            lines.append(f"| {row['grid_index']} | {row['owner']} | {error} |")
    if report.get("not_submitted"):
        lines.extend(["", "## Not Submitted", "", "| job | reason |", "|---:|---|"])
        for row in report["not_submitted"]:
            reason = str(row.get("reason", "")).replace("\n", "<br>")
            lines.append(f"| {row['grid_index']} | {reason} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render and push GMM-TIDE FM Kaggle notebooks.")
    parser.add_argument("--grid-config", default="configs/gmm_tide_fm_grid.json")
    parser.add_argument("--accounts-file", default=str(DEFAULT_ACCOUNTS_FILE))
    parser.add_argument("--env-file", default=".secrets/.env")
    parser.add_argument("--owners", default="all", help="Comma-separated owners or all.")
    parser.add_argument("--exclude-owners", default="kieutung,no1ceboy")
    parser.add_argument("--accelerator", default="tpu")
    parser.add_argument("--staging-root", default="kaggle_staging/gmm_tide_fm")
    parser.add_argument("--report-path", default="reports/gmm_tide_fm_submit.json")
    parser.add_argument("--job-root", default=str(DEFAULT_JOB_ROOT), help="Root for per-kernel submit artifacts and status polls.")
    parser.add_argument("--notebook-registry", default=str(DEFAULT_NOTEBOOK_REGISTRY), help="Local JSONL registry for submitted Kaggle notebooks.")
    parser.add_argument("--artifact-mode", default="has-artifacts", choices=["logs-only", "has-artifacts", "unknown"])
    parser.add_argument(
        "--retention-action",
        default="keep-while-artifacts-needed",
        choices=["delete-after-download", "keep-while-artifacts-needed", "keep", "review"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-submit-per-owner", type=int, default=1)
    parser.add_argument("--shared-context-glob", action="append", default=[])
    parser.add_argument("--shared-context-output", default="reports/kaggle_shared_context.json")
    parser.add_argument("--no-shared-context", action="store_true")
    parser.add_argument("--no-live-shared-context", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow submit even when HEAD is dirty or not known to a remote.")
    parser.add_argument("--skip-ensure-kaggle-cli", action="store_true", help="Do not bootstrap the pinned Kaggle CLI before TPU submit.")
    parser.add_argument("--skip-kjo-validation", action="store_true", help="Skip KJO metadata validation before push.")
    parser.add_argument("--skip-kjo-registry", action="store_true", help="Skip recording successful submits in the local notebook registry.")
    parser.add_argument(
        "--kjo-atomic-submit",
        action="store_true",
        help="Reserve the exact owner through KJO and consume its token with submit-kernel.",
    )
    parser.add_argument(
        "--estimated-runtime-minutes",
        type=float,
        default=480.0,
        help="Expected accelerator runtime used for the KJO session-limit reservation gate.",
    )
    parser.add_argument("--reservation-ttl-minutes", type=int, default=30)
    parser.add_argument("--task-id", default="gmm-tide-fm-submit")
    parser.add_argument(
        "--staging-mode",
        default="blueprint",
        choices=["blueprint", "legacy"],
        help="Render one semantic notebook and materialize destination parameters, or use the legacy per-owner renderer.",
    )
    parser.add_argument(
        "--instrumentation-mode",
        default="none",
        choices=["none", "inline", "runtime-dataset"],
        help="Optional KJO logging instrumentation applied while creating the semantic blueprint.",
    )
    parser.add_argument("--runtime-dataset-slug", default="kjo-runtime-0-10-0")
    parser.add_argument("--runtime-version", default="0.10.0")
    parser.add_argument("--runtime-module-sha256", default="")
    parser.add_argument("--parent-gate-root", default="outputs/kaggle_jobs/parent_resume_gates")
    parser.add_argument("--require-parent-resume-gate", action="store_true")
    parser.add_argument(
        "--record-parent-resume-gate",
        action="store_true",
        help="Record configured parent gates only after their summary and audit JSON both report ok=true.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.kjo_atomic_submit and args.skip_kjo_registry:
        raise SystemExit("--kjo-atomic-submit requires KJO registry recording")
    accelerator = normalize_accelerator(args.accelerator)
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    env_values = load_env_file(Path(args.env_file))
    wandb_api_key = env_values.get("WANDB_API_KEY", "")
    owners = selected_owners(args.owners, sorted(accounts), args.exclude_owners)
    jobs = load_grid(Path(args.grid_config))
    if args.limit:
        jobs = jobs[: args.limit]

    repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ensure_submit_source_ready(repo_commit, allow_dirty=args.allow_dirty, dry_run=args.dry_run)
    ensure_kaggle_cli_for_submit(accelerator, skip=args.skip_ensure_kaggle_cli, dry_run=args.dry_run)
    shared_context = None
    external_running_counts: dict[str, int] = {}
    if not args.no_shared_context:
        shared_context_globs = args.shared_context_glob or ["reports/*.json"]
        shared_context = build_shared_context(
            report_globs=shared_context_globs,
            accounts=accounts,
            live=not args.no_live_shared_context,
        )
        external_running_counts = active_counts_excluding(shared_context, set())
        write_context(Path(args.shared_context_output), shared_context)
        print(
            "Shared Kaggle context active_by_owner: "
            + json.dumps(shared_context.get("summary", {}).get("active_by_owner", {}), sort_keys=True),
            flush=True,
        )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accelerator": accelerator,
        "repo_commit": repo_commit,
        "grid_config": args.grid_config,
        "job_root": args.job_root,
        "notebook_registry": args.notebook_registry,
        "artifact_mode": args.artifact_mode,
        "retention_action": args.retention_action,
        "staging_mode": args.staging_mode,
        "instrumentation_mode": args.instrumentation_mode,
        "max_submit_per_owner": args.max_submit_per_owner,
        "submit_mode": "kjo-atomic" if args.kjo_atomic_submit else "legacy-uncoordinated",
        "shared_context": (
            {
                "output": args.shared_context_output,
                "live": not args.no_live_shared_context,
                "report_globs": shared_context_globs,
                "active_by_owner": shared_context.get("summary", {}).get("active_by_owner", {}),
                "external_active_by_owner": external_running_counts,
            }
            if shared_context
            else None
        ),
        "submitted": [],
        "failed": [],
        "not_submitted": [],
    }
    running_counts: Counter[str] = Counter(external_running_counts)
    cursor = 0
    submit_attempts = 0

    for index, job in enumerate(jobs):
        expected_submit_owner = str(job.get("expected_submit_owner", "")).strip()
        if expected_submit_owner:
            if expected_submit_owner not in owners:
                report["not_submitted"].append(
                    {
                        "grid_index": int(job["grid_index"]),
                        "run_name": job["run_name"],
                        "reason": (
                            f"expected_submit_owner={expected_submit_owner} is not in the selected owner set"
                        ),
                    }
                )
                write_report(Path(args.report_path), report)
                continue
            if (
                args.max_submit_per_owner > 0
                and running_counts[expected_submit_owner] >= args.max_submit_per_owner
            ):
                report["not_submitted"].append(
                    {
                        "grid_index": int(job["grid_index"]),
                        "run_name": job["run_name"],
                        "reason": (
                            f"expected_submit_owner={expected_submit_owner} has reached "
                            "--max-submit-per-owner"
                        ),
                    }
                )
                write_report(Path(args.report_path), report)
                continue
            owner = expected_submit_owner
        else:
            owner, cursor = next_owner(owners, running_counts, args.max_submit_per_owner, cursor)
        if not owner:
            reason = "No owner below --max-submit-per-owner after shared context reconciliation."
            for remaining in jobs[index:]:
                report["not_submitted"].append(
                    {
                        "grid_index": int(remaining["grid_index"]),
                        "run_name": remaining["run_name"],
                        "source_grid_index": remaining.get("source_grid_index"),
                        "source_run_name": remaining.get("source_run_name"),
                        "gmm_num_modes": remaining["gmm_num_modes"],
                        "gmm_router_topk": remaining["gmm_router_topk"],
                        "gmm_router_gradient_mode": remaining.get("gmm_router_gradient_mode", "topk"),
                        "gmm_router_gumbel_tau": remaining.get("gmm_router_gumbel_tau", 1.0),
                        "router_dropout_rate": remaining.get("router_dropout_rate", 0.0),
                        "router_norm_type": remaining.get("router_norm_type", "none"),
                        "gmm_transform": remaining.get("gmm_transform", "auto"),
                        "reason": reason,
                    }
                )
            write_report(Path(args.report_path), report)
            print(reason, flush=True)
            break
        config = dict(job)
        config["repo_commit"] = repo_commit
        parent_gate = evaluate_parent_resume_gate(
            config=config,
            gate_root=Path(args.parent_gate_root),
            require_cache_hit=args.require_parent_resume_gate,
            record=args.record_parent_resume_gate,
        )
        notebook_kaggle_credential = None
        notebook_kaggle_credential_owner = resume_download_credential_owner(
            config=config,
            target_owner=owner,
            accounts=accounts,
        )
        if config.get("resume_kernel_ref") and bool(config.get("resume_download_output", True)):
            notebook_kaggle_credential = accounts[notebook_kaggle_credential_owner]
        job_wandb_api_key = "" if config.get("execution_mode") in {"fid_repeats", "trajectory_eval", "router_geometry_audit"} else wandb_api_key
        runtime_dataset_source = ""
        if args.instrumentation_mode == "runtime-dataset":
            runtime_dataset_source = f"{owner}/{args.runtime_dataset_slug}"
        staging_dir, kernel_id = stage_job(
            owner=owner,
            config=config,
            staging_root=Path(args.staging_root),
            accelerator=accelerator,
            wandb_api_key=job_wandb_api_key,
            kaggle_credential=notebook_kaggle_credential,
            staging_mode=args.staging_mode,
            instrumentation_mode=args.instrumentation_mode,
            runtime_dataset_source=runtime_dataset_source,
            runtime_version=args.runtime_version,
            runtime_module_sha256=args.runtime_module_sha256,
        )
        injected_key_names = []
        if job_wandb_api_key:
            injected_key_names.append("WANDB_API_KEY")
        if notebook_kaggle_credential:
            injected_key_names.append("KAGGLE_CREDENTIAL")
        staged_metadata = json.loads((staging_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        record_injected_notebook(
            notebook_path=staging_dir / staged_metadata["code_file"],
            owner=owner,
            kernel_id=kernel_id,
            key_names=injected_key_names,
        )
        print(f"Staged {kernel_id} at {staging_dir}", flush=True)
        metadata_validation = validate_staged_metadata(
            staging_dir,
            owner,
            accelerator,
            skip=args.skip_kjo_validation,
        )
        row_base = {
            "grid_index": int(config["grid_index"]),
            "owner": owner,
            "expected_submit_owner": config.get("expected_submit_owner", ""),
            "run_name": config["run_name"],
            "execution_mode": config.get("execution_mode", "train"),
            "eval_fid_seeds": config.get("eval_fid_seeds", ""),
            "eval_fid_generations": config.get("eval_fid_generations", ""),
            "candidate_family": config.get("candidate_family", ""),
            "training_seed": config.get("training_seed", ""),
            "gmm_randomization_seed": config.get("gmm_randomization_seed", config.get("training_seed", "")),
            "source_grid_index": config.get("source_grid_index"),
            "source_run_name": config.get("source_run_name"),
            "gmm_num_modes": config["gmm_num_modes"],
            "gmm_router_topk": config["gmm_router_topk"],
            "gmm_router_source_mode": config.get("gmm_router_source_mode", "weighted"),
            "gmm_router_routing_policy": config.get("gmm_router_routing_policy", "router"),
            "gmm_router_gradient_mode": config.get("gmm_router_gradient_mode", "topk"),
            "gmm_router_gumbel_tau": config.get("gmm_router_gumbel_tau", 1.0),
            "gmm_router_eval_use_ema": config.get("gmm_router_eval_use_ema", 0),
            "gmm_router_geometry_weight": config.get("gmm_router_geometry_weight", 0.0),
            "gmm_router_tide_distill_weight": config.get("gmm_router_tide_distill_weight", 0.0),
            "gmm_source_shift_mean": config.get("gmm_source_shift_mean", 0),
            "gmm_source_center_scale": config.get("gmm_source_center_scale", 1.0),
            "router_dropout_rate": config.get("router_dropout_rate", 0.0),
            "router_norm_type": config.get("router_norm_type", "none"),
            "router_train_data_mode": config.get("router_train_data_mode", "mix"),
            "router_target_temperature": config.get("router_target_temperature", 1.0),
            "router_entropy_floor": config.get("router_entropy_floor", 0.0),
            "router_entropy_floor_weight": config.get("router_entropy_floor_weight", 0.0),
            "router_bridge_alpha": config.get("router_bridge_alpha", 2.0),
            "router_bridge_beta": config.get("router_bridge_beta", 2.0),
            "gmm_transform": config.get("gmm_transform", "auto"),
            "gmm_fit_data_mode": config.get("gmm_fit_data_mode", "x1"),
            "gmm_mix_x1_prob": config.get("gmm_mix_x1_prob", 0.5),
            "gmm_continue_em_iters": config.get("gmm_continue_em_iters", 0),
            "gmm_init_strategy": config.get("gmm_init_strategy", "auto"),
            "gmm_init_warmup_iters": config.get("gmm_init_warmup_iters", 0),
            "model_t_sampling": config.get("model_t_sampling", "discrete-dt"),
            "model_t_beta_alpha": config.get("model_t_beta_alpha", 1.0),
            "model_t_beta_beta": config.get("model_t_beta_beta", 1.0),
            "model_eval_ode_schedule": config.get("model_eval_ode_schedule", "uniform"),
            "model_eval_ode_power": config.get("model_eval_ode_power", 1.0),
            "train_target_step_abs": config.get("train_target_step_abs", ""),
            "train_resume_start_step": config.get("train_resume_start_step", ""),
            "train_max_steps": config.get("train_max_steps", ""),
            "save_interval": config.get("save_interval", ""),
            "resume_kernel_ref": config.get("resume_kernel_ref", ""),
            "resume_run_name": config.get("resume_run_name", ""),
            "reset_step_on_load": config.get("reset_step_on_load", ""),
            "notebook_kaggle_credential_owner": notebook_kaggle_credential_owner or "",
            "kernel_id": kernel_id,
            "staging_dir": str(staging_dir),
            "metadata_validation": metadata_validation,
            "parent_resume_gate": parent_gate,
            "staging_mode": args.staging_mode,
            "instrumentation_mode": args.instrumentation_mode,
            "runtime_dataset_source": runtime_dataset_source,
        }
        if args.dry_run:
            report["submitted"].append({**row_base, "kernel_status": "DRY_RUN"})
            running_counts[owner] += 1
            write_report(Path(args.report_path), report)
            scrub_notebook_embedded_credentials(staging_dir / staged_metadata["code_file"])
            if not args.keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)
            continue

        run_dir: Path | None = None
        artifact_paths: dict[str, str] | None = None
        helper_scrub_path: Path | None = None
        reservation_info: dict[str, Any] | None = None
        reservation_handed_off = False
        try:
            if submit_attempts > 0 and not args.kjo_atomic_submit:
                delay = random.uniform(1.0, 4.0)
                print(f"Sleeping {delay:.2f}s before next Kaggle submit.", flush=True)
                time.sleep(delay)
            submit_attempts += 1
            run_dir = run_dir_for_kernel(Path(args.job_root), kernel_id)
            credential = accounts[owner]
            embedded_key_names = []
            if job_wandb_api_key:
                embedded_key_names.append("WANDB_API_KEY")
            if notebook_kaggle_credential:
                embedded_key_names.append("KAGGLE_CREDENTIAL")

            if args.kjo_atomic_submit:
                reservation_info = reserve_exact_owner(
                    owner=owner,
                    accelerator=accelerator,
                    accounts_file=Path(args.accounts_file),
                    project_root=Path.cwd(),
                    run_id=kernel_id.replace("/", "__"),
                    task_id=args.task_id,
                    estimated_runtime_minutes=args.estimated_runtime_minutes,
                    ttl_minutes=args.reservation_ttl_minutes,
                )
                artifact_paths = kjo_submit_artifact_paths(run_dir)
                helper_scrub_path = run_dir / "submit" / "helper_local_secret_scrub_result.json"
                with tempfile.TemporaryDirectory(prefix=f"kaggle-config-{owner}-") as config_dir:
                    config_path = Path(config_dir) / "kaggle.json"
                    config_path.write_text(json.dumps(credential) + "\n", encoding="utf-8")
                    config_path.chmod(0o600)
                    submit_command = build_atomic_submit_command(
                        run_dir=run_dir,
                        staging_dir=staging_dir,
                        owner=owner,
                    accelerator=accelerator,
                    reservation_token=reservation_info["reservation_token"],
                        registry=Path(args.notebook_registry),
                        project_root=Path.cwd(),
                        run_id=kernel_id.replace("/", "__"),
                        task_id=args.task_id,
                        artifact_mode=args.artifact_mode,
                        retention_action=args.retention_action,
                        embedded_key_names=embedded_key_names,
                        kaggle_config_dir=Path(config_dir),
                        runtime_dataset_source=runtime_dataset_source,
                    )
                    reservation_handed_off = True
                    submit_payload = run_json_command(submit_command)
                    artifact_paths.update(copy_atomic_submission_evidence(staging_dir, run_dir))
                    status_payload = run_json_command(
                        [
                            sys.executable,
                            str(KAGGLE_JOB_OPS_SCRIPT),
                            "check-kernel-status",
                            "--run-dir",
                            str(run_dir),
                            "--kernel-id",
                            kernel_id,
                            "--registry",
                            str(Path(args.notebook_registry)),
                            "--kaggle-bin",
                            str(kaggle_command()[0]),
                            "--kaggle-config-dir",
                            str(config_dir),
                            "--note",
                            "initial exact status after KJO atomic submit",
                        ]
                    )
                status_record = status_payload.get("record")
                if not isinstance(status_record, dict):
                    status_record = {}
                kernel_status = str(status_record.get("normalized_status") or "UNKNOWN").upper()
                report["submitted"].append(
                    {
                        **row_base,
                        "kernel_id": kernel_id,
                        "kernel_status": kernel_status,
                        "status_error": "",
                        "run_dir": str(run_dir),
                        "submit_artifacts": artifact_paths,
                        "submit_parse": submit_payload,
                        "reservation": reservation_info["reservation"],
                        "reservation_result": reservation_info["payload"],
                        "registry": str(args.notebook_registry),
                        "registry_result": submit_payload.get("registry_result"),
                        "url": f"https://www.kaggle.com/code/{kernel_id}",
                    }
                )
                running_counts[owner] += 1
                continue

            artifact_paths = copy_submission_artifacts(staging_dir, run_dir)
            with tempfile.TemporaryDirectory(prefix=f"kaggle-config-{owner}-") as config_dir:
                config_path = Path(config_dir) / "kaggle.json"
                config_path.write_text(json.dumps(credential) + "\n", encoding="utf-8")
                config_path.chmod(0o600)
                command_env = os.environ.copy()
                command_env["KAGGLE_CONFIG_DIR"] = config_dir
                push_cmd = [*kaggle_command(), "kernels", "push", "-p", str(staging_dir)]
                if accelerator_kind(accelerator) != "cpu":
                    push_cmd.extend(["--accelerator", accelerator])
                result = subprocess.run(
                    push_cmd,
                    check=False,
                    env=command_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                print(result.stdout, end="", flush=True)
                submit_stdout_path = Path(artifact_paths["submit_stdout"])
                submit_stdout_path.write_text(result.stdout, encoding="utf-8")
                submit_parse = parse_submit_stdout(submit_stdout_path)
                actual_kernel_id = parse_kernel_id(result.stdout, kernel_id)
                status_result = subprocess.run(
                    [*kaggle_command(), "kernels", "status", actual_kernel_id],
                    check=False,
                    env=command_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                print(status_result.stdout, end="", flush=True)
                Path(artifact_paths["status_stdout"]).write_text(status_result.stdout, encoding="utf-8")
                kernel_status = "UNKNOWN_STATUS_ERROR"
                status_error = ""
                if status_result.returncode != 0:
                    status_error = status_result.stdout.strip()
                    print(
                        f"WARNING: kaggle kernels status failed for {actual_kernel_id}: {status_error}",
                        flush=True,
                    )
                else:
                    kernel_status = parse_kernel_status(status_result.stdout)
                append_status_poll(
                    run_dir,
                    owner=owner,
                    kernel_id=actual_kernel_id,
                    status=kernel_status,
                    method="status",
                    returncode=status_result.returncode,
                    output=status_result.stdout,
                )
                registry_result = None
                if not args.skip_kjo_registry:
                    secret_mode = "embedded" if embedded_key_names else "none"
                    registry_result = record_submitted_notebook(
                        registry=Path(args.notebook_registry),
                        kernel_id=actual_kernel_id,
                        run_name=config["run_name"],
                        project_root=Path.cwd(),
                        accelerator=accelerator,
                        artifact_mode=args.artifact_mode,
                        retention_action=args.retention_action,
                        secret_mode=secret_mode,
                        embedded_key_names=embedded_key_names,
                        artifact_paths=artifact_paths,
                    )
                report["submitted"].append(
                    {
                        **row_base,
                        "kernel_id": actual_kernel_id,
                        "kernel_status": kernel_status,
                        "status_error": status_error,
                        "run_dir": str(run_dir),
                        "submit_artifacts": artifact_paths,
                        "submit_parse": submit_parse,
                        "registry": str(args.notebook_registry),
                        "registry_result": registry_result,
                        "url": f"https://www.kaggle.com/code/{actual_kernel_id}",
                    }
                )
                running_counts[owner] += 1
        except Exception as exc:
            print(f"FAILED {kernel_id}: {exc}", flush=True)
            failed_row = {**row_base, "error": str(exc)}
            if reservation_info is not None:
                failed_row["reservation"] = reservation_info.get("reservation")
                if not reservation_handed_off:
                    try:
                        failed_row["reservation_release"] = release_unused_reservation(
                            owner=owner,
                            accelerator=accelerator,
                            reservation=reservation_info["reservation"],
                        )
                    except Exception as release_exc:
                        failed_row["reservation_release_error"] = str(release_exc)
            if run_dir is not None:
                failed_row["run_dir"] = str(run_dir)
            if artifact_paths is not None:
                failed_row["submit_artifacts"] = artifact_paths
            report["failed"].append(failed_row)
            running_counts[owner] += 1
        finally:
            if args.kjo_atomic_submit:
                notebook_paths: list[Path] = []
                metadata_path = staging_dir / "kernel-metadata.json"
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    notebook_paths.append(staging_dir / str(metadata["code_file"]))
                if artifact_paths is not None:
                    notebook_paths.append(Path(artifact_paths["submitted_notebook"]))
                scrub_results = [scrub_notebook_embedded_credentials(path) for path in notebook_paths]
                scrub_result = {
                    "scrubbed_at_utc": utc_now(),
                    "ok": all(bool(item["ok"]) for item in scrub_results),
                    "notebooks": scrub_results,
                }
                if helper_scrub_path is not None:
                    helper_scrub_path.parent.mkdir(parents=True, exist_ok=True)
                    helper_scrub_path.write_text(
                        json.dumps(scrub_result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    if artifact_paths is not None:
                        artifact_paths["helper_local_secret_scrub"] = str(helper_scrub_path)
            else:
                scrub_result = scrub_local_submission_notebooks(staging_dir, artifact_paths)
                if artifact_paths is not None:
                    artifact_paths["local_secret_scrub_ok"] = str(bool(scrub_result["ok"]))
            if not args.keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)
            write_report(Path(args.report_path), report)


if __name__ == "__main__":
    main()
