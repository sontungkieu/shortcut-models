from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from push_gmm_ablation_jobs import kaggle_command, load_kaggle_accounts, parse_kernel_id, parse_kernel_status
from stage_gmm_ablation_jobs import normalize_accelerator, slugify


DEFAULT_ACCOUNTS_FILE = Path("/home/tung/all-kaggle.json")


def load_grid(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = payload.get("defaults", {})
    jobs = []
    for index, raw in enumerate(payload.get("jobs", [])):
        job = dict(defaults)
        job.update(raw)
        job["grid_index"] = index
        jobs.append(job)
    if not jobs:
        raise SystemExit(f"No jobs found in {path}")
    return jobs


def selected_owners(value: str, available: list[str], exclude: str) -> list[str]:
    owners = sorted(available) if value == "all" else [item.strip() for item in value.split(",") if item.strip()]
    excluded = {item.strip() for item in exclude.split(",") if item.strip()}
    owners = [owner for owner in owners if owner not in excluded]
    missing = [owner for owner in owners if owner not in available]
    if missing:
        raise SystemExit(f"Unknown Kaggle owner(s): {', '.join(missing)}")
    if not owners:
        raise SystemExit("No owners selected after applying exclusions.")
    return owners


def make_notebook(config: dict[str, Any], repo_commit: str) -> dict[str, Any]:
    config_json = json.dumps(config, indent=4, sort_keys=True)
    init_configs_json = json.dumps(config.get("init_configs", []))
    variant_configs_json = json.dumps(config.get("variant_configs", []))
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MOE2 Toy/MNIST FM Init Ablation\n",
                    "\n",
                    "Runs a full GMM -> router distill -> TIDE source -> FM MLP pipeline on GPU.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json, os, subprocess, sys\n",
                    "from pathlib import Path\n",
                    f"CONFIG = json.loads({json.dumps(config_json)})\n",
                    f"REPO_COMMIT = {json.dumps(repo_commit)}\n",
                    "RUN_NAME = CONFIG['run_name']\n",
                    "base = Path('/kaggle/working/toy_moe2_fm') / RUN_NAME\n",
                    "base.mkdir(parents=True, exist_ok=True)\n",
                    "(base / 'config.json').write_text(json.dumps(CONFIG, indent=2, sort_keys=True) + '\\n')\n",
                    "print(json.dumps(CONFIG, indent=2, sort_keys=True))\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import importlib.util, subprocess, sys\n",
                    "pkgs = []\n",
                    "if importlib.util.find_spec('jax') is None:\n",
                    "    pkgs.append('jax[cuda12]==0.5.3')\n",
                    "if importlib.util.find_spec('tensorflow') is None:\n",
                    "    pkgs.append('tensorflow-cpu>=2.16.0')\n",
                    "if importlib.util.find_spec('matplotlib') is None:\n",
                    "    pkgs.append('matplotlib')\n",
                    "if pkgs:\n",
                    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=True)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, shutil, subprocess\n",
                    "from pathlib import Path\n",
                    "repo = Path('/kaggle/working/shortcut-models')\n",
                    "if repo.exists():\n",
                    "    shutil.rmtree(repo)\n",
                    "subprocess.run(['git', 'clone', '--depth', '1', '--branch', CONFIG.get('branch', 'moe2'), CONFIG['repo_url'], str(repo)], check=True)\n",
                    "subprocess.run(['git', 'fetch', '--depth', '1', 'origin', REPO_COMMIT], cwd=repo, check=True)\n",
                    "subprocess.run(['git', 'checkout', REPO_COMMIT], cwd=repo, check=True)\n",
                    "print(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip())\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, subprocess\n",
                    "from pathlib import Path\n",
                    "repo = Path('/kaggle/working/shortcut-models')\n",
                    "base = Path('/kaggle/working/toy_moe2_fm') / RUN_NAME\n",
                    "stdout_path = base / 'stdout.txt'\n",
                    "stderr_path = base / 'stderr.txt'\n",
                    "cmd = [\n",
                    "    sys.executable, 'scripts/run_toy_moe2_fm_ablation.py',\n",
                    "    '--datasets', CONFIG['datasets'],\n",
                    "    '--out-dir', str(base),\n",
                    "    '--n-train', str(CONFIG['n_train']),\n",
                    "    '--n-valid', str(CONFIG['n_valid']),\n",
                    "    '--gmm-modes', str(CONFIG['gmm_modes']),\n",
                    "    '--gmm-iters', str(CONFIG['gmm_iters']),\n",
                    "    '--gmm-floor-frac', str(CONFIG.get('gmm_floor_frac', 0.0)),\n",
                    "    '--router-steps', str(CONFIG['router_steps']),\n",
                    "    '--router-lr', str(CONFIG['router_lr']),\n",
                    "    '--fm-steps', str(CONFIG['fm_steps']),\n",
                    "    '--fm-lr', str(CONFIG['fm_lr']),\n",
                    "    '--hidden', str(CONFIG['hidden']),\n",
                    "    '--batch-size', str(CONFIG['batch_size']),\n",
                    "    '--topk', str(CONFIG['topk']),\n",
                    "    '--eval-batches', str(CONFIG['eval_batches']),\n",
                    "    '--rollout-samples', str(CONFIG['rollout_samples']),\n",
                    "    '--pca-dim', str(CONFIG.get('pca_dim', 0)),\n",
                    "    '--pca-max-samples', str(CONFIG.get('pca_max_samples', 4096)),\n",
                    "    '--standardize', str(CONFIG.get('standardize', 1)),\n",
                    "    '--seed', str(CONFIG['seed']),\n",
                    "    '--init-configs',\n",
                    f"    *{init_configs_json},\n",
                    "]\n",
                    f"variant_configs = {variant_configs_json}\n",
                    "if variant_configs:\n",
                    "    cmd.extend(['--variant-configs', *variant_configs])\n",
                    "env = os.environ.copy()\n",
                    "env.setdefault('JAX_PLATFORMS', 'cuda,cpu')\n",
                    "env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')\n",
                    "print(' '.join(cmd))\n",
                    "with stdout_path.open('w') as out, stderr_path.open('w') as err:\n",
                    "    result = subprocess.run(cmd, cwd=repo, env=env, stdout=out, stderr=err, text=True)\n",
                    "print('returncode', result.returncode)\n",
                    "print('--- stdout tail ---')\n",
                    "print('\\n'.join(stdout_path.read_text(errors='replace').splitlines()[-80:]))\n",
                    "print('--- stderr tail ---')\n",
                    "print('\\n'.join(stderr_path.read_text(errors='replace').splitlines()[-80:]))\n",
                    "result.check_returncode()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "base = Path('/kaggle/working/toy_moe2_fm') / RUN_NAME\n",
                    "print((base / 'toy_moe2_fm_summary.md').read_text())\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def stage_job(owner: str, config: dict[str, Any], staging_root: Path, accelerator: str, repo_commit: str):
    accelerator = normalize_accelerator(accelerator)
    is_gpu = accelerator.lower() not in ("none", "cpu")
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
    (staging_dir / notebook_name).write_text(
        json.dumps(make_notebook(config, repo_commit), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": is_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (staging_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (staging_dir / "toy_moe2_fm_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return staging_dir, metadata["id"]


def write_report(path: Path, report: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Toy MOE2 FM Submit Report",
        "",
        f"- Submitted: {len(report['submitted'])}",
        f"- Failed: {len(report['failed'])}",
        "",
        "| job | owner | run | datasets | modes | pca | kernel | status |",
        "|---:|---|---|---|---:|---:|---|---|",
    ]
    for row in report["submitted"]:
        lines.append(
            f"| {row['grid_index']} | {row['owner']} | {row['run_name']} | {row['datasets']} | "
            f"{row['gmm_modes']} | {row.get('pca_dim', 0)} | `{row['kernel_id']}` | {row.get('kernel_status', '')} |"
        )
    if report["failed"]:
        lines.extend(["", "## Failed", "", "| job | owner | run | error |", "|---:|---|---|---|"])
        for row in report["failed"]:
            lines.append(f"| {row['grid_index']} | {row['owner']} | {row['run_name']} | {str(row.get('error', '')).replace(chr(10), '<br>')} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Submit toy/MNIST MOE2 FM init ablation notebooks to Kaggle.")
    parser.add_argument("--grid-config", default="configs/toy_moe2_fm_complex_init_grid.json")
    parser.add_argument("--accounts-file", default=str(DEFAULT_ACCOUNTS_FILE))
    parser.add_argument("--owners", default="all")
    parser.add_argument("--exclude-owners", default="kieutung")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--staging-root", default="kaggle_staging/toy_moe2_fm")
    parser.add_argument("--report-path", default="reports/toy_moe2_fm_submit.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    accelerator = normalize_accelerator(args.accelerator)
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    owners = selected_owners(args.owners, sorted(accounts), args.exclude_owners)
    jobs = list(enumerate(load_grid(Path(args.grid_config))))[args.offset :]
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
    for local_index, (grid_index, config) in enumerate(jobs):
        owner = owners[local_index % len(owners)]
        staging_dir, kernel_id = stage_job(owner, config, Path(args.staging_root), accelerator, repo_commit)
        print(f"Staged {kernel_id} at {staging_dir}", flush=True)
        row = {
            "grid_index": grid_index,
            "owner": owner,
            "run_name": config["run_name"],
            "datasets": config["datasets"],
            "gmm_modes": config["gmm_modes"],
            "pca_dim": config.get("pca_dim", 0),
            "kernel_id": kernel_id,
            "staging_dir": str(staging_dir),
        }
        if args.dry_run:
            report["submitted"].append({**row, "kernel_status": "DRY_RUN"})
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
                env = os.environ.copy()
                env["KAGGLE_CONFIG_DIR"] = config_dir
                push_cmd = [*kaggle_command(), "kernels", "push", "-p", str(staging_dir), "--accelerator", accelerator]
                result = subprocess.run(push_cmd, env=env, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                print(result.stdout, end="", flush=True)
                if result.returncode != 0 or "Kernel push error:" in result.stdout:
                    raise RuntimeError(f"kaggle kernels push failed: {result.stdout.strip()}")
                actual_kernel_id = parse_kernel_id(result.stdout, kernel_id)
                status = subprocess.run(
                    [*kaggle_command(), "kernels", "status", actual_kernel_id],
                    env=env,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                print(status.stdout, end="", flush=True)
                report["submitted"].append({
                    **row,
                    "kernel_id": actual_kernel_id,
                    "kernel_status": parse_kernel_status(status.stdout),
                })
        except Exception as exc:
            print(f"FAILED {kernel_id}: {exc}", flush=True)
            report["failed"].append({**row, "error": str(exc)})
        finally:
            write_report(Path(args.report_path), report)
            if not args.keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
