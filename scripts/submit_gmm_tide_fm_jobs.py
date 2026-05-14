from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_shared_context import active_counts_excluding, build_shared_context, write_context
from stage_gmm_ablation_jobs import load_env_file, normalize_accelerator, slugify
from push_gmm_ablation_jobs import kaggle_command, load_kaggle_accounts, parse_kernel_id, parse_kernel_status


DEFAULT_ACCOUNTS_FILE = Path("/home/tung/all-kaggle.json")


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

if WANDB_API_KEY:
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
    kaggle_config_dir = Path("/kaggle/working/.kaggle_config")
    kaggle_config_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_config_dir / "kaggle.json"
    kaggle_json_path.write_text(json.dumps(KAGGLE_CREDENTIAL) + "\\n", encoding="utf-8")
    kaggle_json_path.chmod(0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)

del WANDB_API_KEY
del KAGGLE_CREDENTIAL

os.environ["MPLBACKEND"] = "agg"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
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

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle", "protobuf<4", "tfds", "apache_beam", "mlcroissant"], check=True)
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

download_dir = Path("/kaggle/working/shortcut_dataset")
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

os.chdir("/kaggle/working")
if has_built_celebahq:
    print(f"Using prebuilt TFDS from {tfds_target}")
else:
    if not Path("tfds_builders").exists():
        subprocess.run(["git", "clone", "https://github.com/kvfrans/tfds_builders.git"], check=True)
    os.chdir("/kaggle/working/tfds_builders/celebahq256")
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

os.chdir("/kaggle/working")
if not Path("shortcut-models").exists():
    subprocess.run(["git", "clone", CONFIG["repo_url"], "shortcut-models"], check=True)
os.chdir("/kaggle/working/shortcut-models")
subprocess.run(["git", "fetch", "--all"], check=True)
subprocess.run(["git", "checkout", CONFIG["branch"]], check=True)
subprocess.run(["git", "pull"], check=True)
if CONFIG.get("repo_commit"):
    subprocess.run(["git", "checkout", CONFIG["repo_commit"]], check=True)
run_logged(["uv", "sync"], Path("sync_out.txt"), Path("sync_err.txt"))
subprocess.run(["uv", "cache", "clean"], check=False)
subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], check=False)
shutil.rmtree(Path.home() / ".cache" / "pip", ignore_errors=True)

source_data = Path("/kaggle/working/shortcut_dataset/data")
if source_data.exists():
    target_data = Path("/kaggle/working/shortcut-models/data")
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
    target_step = int(CONFIG.get("resume_checkpoint_step", 0) or 0)
    if target_step > 0:
        patterns.append(rf".*ckpts.*{target_step}.*")
    else:
        patterns.append(r".*ckpts.*")
    return "|".join(patterns)


def _cleanup_kaggle_config() -> None:
    config_dir = os.environ.pop("KAGGLE_CONFIG_DIR", "")
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
                if path.suffix in {".json", ".jsonl", ".csv", ".txt", ".png", ".jpg", ".jpeg", ".npz", ".pkl"}:
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
    download_dir = Path(CONFIG.get("resume_output_dir", "/kaggle/working/resume_output"))
    if bool(CONFIG.get("resume_download_output", True)):
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

    if checkpoint_source is None:
        raise FileNotFoundError(f"Could not find a checkpoint under previous output roots: {[str(p) for p in roots]}")
    checkpoint_target = _copy_resume_checkpoint(checkpoint_source)
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
        "load_dir": str(checkpoint_target),
        "checkpoint_step_guess": _checkpoint_step(checkpoint_source),
        "checkpoint_source": str(checkpoint_source),
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

if bool(CONFIG.get("resume_reuse_gmm_router", True)) and gmm_stats_path.exists():
    print(f"Using resumed GMM stats at {gmm_stats_path}")
else:
    prep_cmd = [
        "uv", "run", "data_prep.py",
        "--dataset_name", CONFIG["dataset_name"],
        "--tfds_data_dir", CONFIG["tfds_data_dir"],
        "--batch_size", str(CONFIG["batch_size"]),
        "--gmm_save_path", str(gmm_stats_path),
        "--gmm_latent_cache_path", str(base_dir / "gmm_latents.dat"),
        "--gmm_num_modes", str(CONFIG["gmm_num_modes"]),
        "--gmm_fit_samples", str(CONFIG["gmm_fit_samples"]),
        "--gmm_valid_samples", str(CONFIG["gmm_valid_samples"]),
        "--gmm_em_iters", str(CONFIG["gmm_em_iters"]),
        "--gmm_em_restarts", str(CONFIG["gmm_em_restarts"]),
        "--gmm_init_seed", str(CONFIG["gmm_init_seed"]),
        "--gmm_standardize_data", str(CONFIG["gmm_standardize_data"]),
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

if bool(CONFIG.get("resume_reuse_gmm_router", True)) and router_path.exists():
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
        "--router_target_type", CONFIG["router_target_type"],
        "--router_max_steps", str(CONFIG["router_max_steps"]),
        "--router_log_interval", "100",
        "--router_valid_interval", str(CONFIG["router_valid_interval"]),
        "--router_valid_batches", str(CONFIG["router_valid_batches"]),
        "--router_lr", str(CONFIG["router_lr"]),
        "--router_weight_decay", str(CONFIG["router_weight_decay"]),
        "--router_hidden_channels", str(CONFIG["router_hidden_channels"]),
        "--router_mlp_hidden_size", str(CONFIG["router_mlp_hidden_size"]),
        "--router_depth", str(CONFIG["router_depth"]),
        f"--router_save_best={bool(CONFIG['router_save_best'])}",
        "--metrics_output_path", str(diag_dir / "router_metrics.jsonl"),
        "--wandb.name", f"router_{RUN_NAME}",
        f"--wandb.offline={not bool(os.environ.get('WANDB_API_KEY'))}",
    ]
    run_logged(router_cmd, diag_dir / "router_stdout.txt", diag_dir / "router_stderr.txt")
"""
        ),
        make_code_cell(
            """import os
import subprocess
from pathlib import Path

base_dir = Path("/kaggle/working/gmm_tide_fm") / RUN_NAME
diag_dir = base_dir / "diagnostics"
ckpt_root = Path("/kaggle/working/ckpts")
ckpt_path = ckpt_root / f"{RUN_NAME}.pkl"
diag_dir.mkdir(parents=True, exist_ok=True)
ckpt_root.mkdir(parents=True, exist_ok=True)

train_cmd = [
    "uv", "run", "train.py",
    "--model.hidden_size", "768",
    "--model.patch_size", "2",
    "--model.depth", "12",
    "--model.num_heads", "12",
    "--model.mlp_ratio", "4",
    "--model.train_type", "gmm-tide",
    "--model.cfg_scale", "0",
    "--model.class_dropout_prob", "1",
    "--model.num_classes", "1",
    "--model.denoise_timesteps", "128",
    "--batch_size", str(CONFIG["train_batch_size"]),
    "--dataset_name", CONFIG["dataset_name"],
    "--tfds_data_dir", CONFIG["tfds_data_dir"],
    "--fid_stats", "data/celeba256_fidstats_ours.npz",
    "--max_steps", str(CONFIG["train_max_steps"]),
    "--eval_interval", str(CONFIG["train_eval_interval"]),
    "--log_interval", str(CONFIG["train_log_interval"]),
    "--save_dir", str(ckpt_path),
    "--wandb.name", RUN_NAME,
    "--model.weight_decay", "0.01",
    "--model.gmm_stats_path", str(base_dir / "gmm_stats.npz"),
    "--model.gmm_router_path", str(base_dir / "gmm_router.pkl"),
    "--model.gmm_router_topk", str(CONFIG["gmm_router_topk"]),
    "--model.gmm_router_temperature", str(CONFIG["gmm_router_temperature"]),
    "--model.gmm_router_update_policy", "frozen",
    "--model.gmm_cond_channels", str(CONFIG["model_gmm_cond_channels"]),
    "--eval_fid_timesteps", CONFIG["eval_fid_timesteps"],
    "--metrics_output_path", str(diag_dir / "train_metrics.jsonl"),
    f"--wandb.offline={not bool(os.environ.get('WANDB_API_KEY'))}",
]
resume_manifest_path = diag_dir / "resume_manifest.json"
if resume_manifest_path.exists():
    resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
    train_cmd.extend([
        "--load_dir", resume_manifest["load_dir"],
        "--reset_step_on_load", str(CONFIG.get("reset_step_on_load", 0)),
    ])
    if bool(CONFIG.get("delete_load_dir_after_load", True)):
        train_cmd.extend(["--delete_load_dir_after_load", "1"])
if CONFIG.get("save_interval"):
    train_cmd.extend(["--save_interval", str(CONFIG["save_interval"])])
run_logged(train_cmd, diag_dir / "train_stdout.txt", diag_dir / "train_stderr.txt")
"""
        ),
    ]
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


def stage_job(
    *,
    owner: str,
    config: dict[str, Any],
    staging_root: Path,
    accelerator: str,
    wandb_api_key: str,
    kaggle_credential: dict[str, str] | None = None,
) -> tuple[Path, str]:
    accelerator = normalize_accelerator(accelerator)
    is_tpu = accelerator.lower().startswith("tpu")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    slug = slugify(f"{config['run_name']}-{owner}-{timestamp}", max_length=48)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / slug
    suffix = 2
    while staging_dir.exists():
        staging_dir = staging_root / f"{slug}-{suffix}"
        suffix += 1
    staging_dir.mkdir(parents=True, exist_ok=False)

    notebook_name = f"{slug}.ipynb"
    notebook_path = staging_dir / notebook_name
    notebook_path.write_text(
        json.dumps(
            make_notebook(config, wandb_api_key=wandb_api_key, kaggle_credential=kaggle_credential),
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
        "enable_gpu": not is_tpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }
    (staging_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (staging_dir / "gmm_tide_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return staging_dir, metadata["id"]


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
        "| job | owner | modes | topk | fit_data | cont_em | resume | source_grid | kernel | status |",
        "|---:|---|---:|---:|---|---:|---|---:|---|---|",
    ])
    for row in report["submitted"]:
        lines.append(
            f"| {row['grid_index']} | {row['owner']} | {row['gmm_num_modes']} | {row['gmm_router_topk']} | "
            f"{row.get('gmm_fit_data_mode', '')} | {row.get('gmm_continue_em_iters', '')} | "
            f"{row.get('resume_kernel_ref', '')} | "
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-submit-per-owner", type=int, default=1)
    parser.add_argument("--shared-context-glob", action="append", default=[])
    parser.add_argument("--shared-context-output", default="reports/kaggle_shared_context.json")
    parser.add_argument("--no-shared-context", action="store_true")
    parser.add_argument("--no-live-shared-context", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow submit even when HEAD is dirty or not known to a remote.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
        "max_submit_per_owner": args.max_submit_per_owner,
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

    for index, job in enumerate(jobs):
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
                        "reason": reason,
                    }
                )
            write_report(Path(args.report_path), report)
            print(reason, flush=True)
            break
        config = dict(job)
        config["repo_commit"] = repo_commit
        notebook_kaggle_credential = None
        if config.get("resume_kernel_ref") and bool(config.get("resume_download_output", True)):
            notebook_kaggle_credential = accounts[owner]
        staging_dir, kernel_id = stage_job(
            owner=owner,
            config=config,
            staging_root=Path(args.staging_root),
            accelerator=accelerator,
            wandb_api_key=wandb_api_key,
            kaggle_credential=notebook_kaggle_credential,
        )
        print(f"Staged {kernel_id} at {staging_dir}", flush=True)
        row_base = {
            "grid_index": int(config["grid_index"]),
            "owner": owner,
            "run_name": config["run_name"],
            "source_grid_index": config.get("source_grid_index"),
            "source_run_name": config.get("source_run_name"),
            "gmm_num_modes": config["gmm_num_modes"],
            "gmm_router_topk": config["gmm_router_topk"],
            "gmm_fit_data_mode": config.get("gmm_fit_data_mode", "x1"),
            "gmm_mix_x1_prob": config.get("gmm_mix_x1_prob", 0.5),
            "gmm_continue_em_iters": config.get("gmm_continue_em_iters", 0),
            "resume_kernel_ref": config.get("resume_kernel_ref", ""),
            "resume_run_name": config.get("resume_run_name", ""),
            "reset_step_on_load": config.get("reset_step_on_load", ""),
            "kernel_id": kernel_id,
            "staging_dir": str(staging_dir),
        }
        if args.dry_run:
            report["submitted"].append({**row_base, "kernel_status": "DRY_RUN"})
            running_counts[owner] += 1
            write_report(Path(args.report_path), report)
            if not args.keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)
            continue

        try:
            credential = accounts[owner]
            with tempfile.TemporaryDirectory(prefix=f"kaggle-config-{owner}-") as config_dir:
                config_path = Path(config_dir) / "kaggle.json"
                config_path.write_text(json.dumps(credential) + "\n", encoding="utf-8")
                config_path.chmod(0o600)
                command_env = os.environ.copy()
                command_env["KAGGLE_CONFIG_DIR"] = config_dir
                push_cmd = [*kaggle_command(), "kernels", "push", "-p", str(staging_dir), "--accelerator", accelerator]
                result = subprocess.run(
                    push_cmd,
                    check=False,
                    env=command_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                print(result.stdout, end="", flush=True)
                if result.returncode != 0 or "Kernel push error:" in result.stdout:
                    raise RuntimeError(f"kaggle kernels push failed: {result.stdout.strip()}")
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
                if status_result.returncode != 0:
                    raise RuntimeError(f"kaggle kernels status failed: {status_result.stdout.strip()}")
                report["submitted"].append(
                    {
                        **row_base,
                        "kernel_id": actual_kernel_id,
                        "kernel_status": parse_kernel_status(status_result.stdout),
                        "url": f"https://www.kaggle.com/code/{actual_kernel_id}",
                    }
                )
                running_counts[owner] += 1
        except Exception as exc:
            print(f"FAILED {kernel_id}: {exc}", flush=True)
            report["failed"].append({**row_base, "error": str(exc)})
            running_counts[owner] += 1
        finally:
            if not args.keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)
            write_report(Path(args.report_path), report)


if __name__ == "__main__":
    main()
