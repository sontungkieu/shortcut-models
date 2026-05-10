from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage_gmm_ablation_jobs import normalize_accelerator, slugify
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


def make_code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def make_notebook(config: dict[str, Any]) -> dict[str, Any]:
    config_json = json.dumps(config, indent=4, sort_keys=True)
    cells = [
        make_code_cell(
            f"""import json
import os

CONFIG = {config_json}
RUN_NAME = CONFIG["run_name"]

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

os.environ["MPLBACKEND"] = "agg"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["ENABLE_PJRT_COMPATIBILITY"] = "1"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
print(json.dumps(CONFIG, indent=2, sort_keys=True))
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
subprocess.run([
    sys.executable,
    "-m",
    "kaggle",
    "datasets",
    "download",
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

os.chdir("/kaggle/working")
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
with open("sync_out.txt", "w", encoding="utf-8") as out, open("sync_err.txt", "w", encoding="utf-8") as err:
    subprocess.run(["uv", "sync"], stdout=out, stderr=err, check=True)

source_data = Path("/kaggle/working/shortcut_dataset/data")
if source_data.exists():
    target_data = Path("/kaggle/working/shortcut-models/data")
    if target_data.exists():
        shutil.rmtree(target_data)
    shutil.copytree(source_data, target_data)
"""
        ),
        make_code_cell(
            """import subprocess
from pathlib import Path

base_dir = Path("/kaggle/working/gmm_tide_fm") / RUN_NAME
diag_dir = base_dir / "diagnostics"
diag_dir.mkdir(parents=True, exist_ok=True)
gmm_stats_path = base_dir / "gmm_stats.npz"

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
]
with open(diag_dir / "gmm_prep_stdout.txt", "w", encoding="utf-8") as out, open(diag_dir / "gmm_prep_stderr.txt", "w", encoding="utf-8") as err:
    subprocess.run(prep_cmd, stdout=out, stderr=err, check=True)
"""
        ),
        make_code_cell(
            """import subprocess
from pathlib import Path

base_dir = Path("/kaggle/working/gmm_tide_fm") / RUN_NAME
diag_dir = base_dir / "diagnostics"
router_path = base_dir / "gmm_router.pkl"

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
    "--router_valid_interval", "1000",
    "--router_valid_batches", str(CONFIG["router_valid_batches"]),
    "--router_lr", str(CONFIG["router_lr"]),
    "--router_hidden_channels", str(CONFIG["router_hidden_channels"]),
    "--router_mlp_hidden_size", str(CONFIG["router_mlp_hidden_size"]),
    "--metrics_output_path", str(diag_dir / "router_metrics.jsonl"),
    "--wandb.name", f"router_{RUN_NAME}",
]
with open(diag_dir / "router_stdout.txt", "w", encoding="utf-8") as out, open(diag_dir / "router_stderr.txt", "w", encoding="utf-8") as err:
    subprocess.run(router_cmd, stdout=out, stderr=err, check=True)
"""
        ),
        make_code_cell(
            """import subprocess
from pathlib import Path

base_dir = Path("/kaggle/working/gmm_tide_fm") / RUN_NAME
diag_dir = base_dir / "diagnostics"
ckpt_dir = Path("/kaggle/working/ckpts") / RUN_NAME
diag_dir.mkdir(parents=True, exist_ok=True)
ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    "--save_dir", str(ckpt_dir),
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
]
with open(diag_dir / "train_stdout.txt", "w", encoding="utf-8") as out, open(diag_dir / "train_stderr.txt", "w", encoding="utf-8") as err:
    subprocess.run(train_cmd, stdout=out, stderr=err, check=True)
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
) -> tuple[Path, str]:
    accelerator = normalize_accelerator(accelerator)
    is_tpu = accelerator.lower().startswith("tpu")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    slug = slugify(f"{config['run_name']}-{owner}-{timestamp}", max_length=63)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / slug
    suffix = 2
    while staging_dir.exists():
        staging_dir = staging_root / f"{slug}-{suffix}"
        suffix += 1
    staging_dir.mkdir(parents=True, exist_ok=False)

    notebook_name = f"{slug}.ipynb"
    notebook_path = staging_dir / notebook_name
    notebook_path.write_text(json.dumps(make_notebook(config), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not is_tpu,
        "enable_tpu": is_tpu,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
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
        "",
        "| job | owner | modes | topk | source_grid | kernel | status |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for row in report["submitted"]:
        lines.append(
            f"| {row['grid_index']} | {row['owner']} | {row['gmm_num_modes']} | {row['gmm_router_topk']} | "
            f"{row['source_grid_index']} | `{row['kernel_id']}` | {row.get('kernel_status', '')} |"
        )
    if report["failed"]:
        lines.extend(["", "## Failed", "", "| job | owner | error |", "|---:|---|---|"])
        for row in report["failed"]:
            error = str(row.get("error", "")).replace("\n", "<br>")
            lines.append(f"| {row['grid_index']} | {row['owner']} | {error} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render and push GMM-TIDE FM Kaggle notebooks.")
    parser.add_argument("--grid-config", default="configs/gmm_tide_fm_grid.json")
    parser.add_argument("--accounts-file", default=str(DEFAULT_ACCOUNTS_FILE))
    parser.add_argument("--owners", default="all", help="Comma-separated owners or all.")
    parser.add_argument("--exclude-owners", default="kieutung,no1ceboy")
    parser.add_argument("--accelerator", default="tpu")
    parser.add_argument("--staging-root", default="kaggle_staging/gmm_tide_fm")
    parser.add_argument("--report-path", default="reports/gmm_tide_fm_submit.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    accelerator = normalize_accelerator(args.accelerator)
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    owners = selected_owners(args.owners, sorted(accounts), args.exclude_owners)
    jobs = load_grid(Path(args.grid_config))
    if args.limit:
        jobs = jobs[: args.limit]

    repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "accelerator": accelerator,
        "repo_commit": repo_commit,
        "grid_config": args.grid_config,
        "submitted": [],
        "failed": [],
    }

    for index, job in enumerate(jobs):
        owner = owners[index % len(owners)]
        config = dict(job)
        config["repo_commit"] = repo_commit
        staging_dir, kernel_id = stage_job(
            owner=owner,
            config=config,
            staging_root=Path(args.staging_root),
            accelerator=accelerator,
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
            "kernel_id": kernel_id,
            "staging_dir": str(staging_dir),
        }
        if args.dry_run:
            report["submitted"].append({**row_base, "kernel_status": "DRY_RUN"})
            write_report(Path(args.report_path), report)
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
        except Exception as exc:
            print(f"FAILED {kernel_id}: {exc}", flush=True)
            report["failed"].append({**row_base, "error": str(exc)})
        finally:
            if not args.keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)
            write_report(Path(args.report_path), report)


if __name__ == "__main__":
    main()
