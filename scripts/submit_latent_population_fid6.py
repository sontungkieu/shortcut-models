from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KJO_ROOT = Path("/home/tung/.codex/skills/kaggle-job-ops")
KJO_SCRIPT = KJO_ROOT / "scripts" / "kaggle_job_ops.py"
KJO_ACCOUNTS = KJO_ROOT / "scripts" / "kaggle_accounts.py"
DEFAULT_ACCOUNTS_FILE = Path("/home/tung/all-kaggle.json")
DEFAULT_REGISTRY = Path(".secrets/kaggle_notebooks.jsonl")
RUNTIME_FILES = (
    Path("utils/stable_vae.py"),
    Path("utils/datasets.py"),
    Path("gmm_utils.py"),
    Path("latent_population.py"),
    Path("latent_geometry.py"),
    Path("analyze_latent_population.py"),
)


def _run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Command did not return JSON: {' '.join(command)}\n"
            f"{result.stdout[-4000:]}"
        ) from exc
    if result.returncode != 0 or not payload.get("ok", False):
        raise RuntimeError(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _run_checked(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n{result.stdout[-4000:]}"
        )
    return result.stdout


def _wait_for_dataset_ready(
    *,
    kaggle_bin: str,
    kaggle_config_dir: str,
    dataset_source: str,
    expected_paths: list[str],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = kaggle_config_dir
    last_output = ""
    while time.monotonic() < deadline:
        result = subprocess.run(
            [kaggle_bin, "datasets", "files", dataset_source],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        last_output = result.stdout
        if result.returncode == 0 and all(
            path in last_output for path in expected_paths
        ):
            return
        time.sleep(5.0)
    raise RuntimeError(
        f"Dataset {dataset_source} was not ready after {timeout_seconds}s\n"
        f"{last_output[-2000:]}"
    )


def _slugify(value: str, max_length: int = 48) -> str:
    value = "".join(
        character.lower() if character.isalnum() else "-" for character in value
    )
    value = "-".join(part for part in value.split("-") if part)
    return value[:max_length].rstrip("-")


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_gmm_source(source_root: Path, kernel_id: str) -> Path:
    owner, slug = kernel_id.split("/", 1)
    run_dir = source_root / f"{owner}__{slug}"
    candidates = sorted(run_dir.rglob("gmm_stats.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No gmm_stats.npz found for {kernel_id} under {run_dir}"
        )
    preferred = [path for path in candidates if "/output/" in path.as_posix()]
    return preferred[0] if preferred else candidates[0]


def _prepare_selection(
    config: dict[str, Any],
    source_root: Path,
) -> list[dict[str, Any]]:
    selections = []
    for selection in config["selections"]:
        selection = dict(selection)
        source_path = _resolve_gmm_source(
            source_root, selection["artifact_kernel_id"]
        )
        selection["local_source_path"] = str(source_path)
        selection["gmm_stats_sha256"] = _sha256(source_path)
        selection["gmm_stats_bytes"] = source_path.stat().st_size
        selections.append(selection)
    return selections


def _stage_asset_dataset(
    *,
    repo_root: Path,
    selections: list[dict[str, Any]],
    owner: str,
    dataset_slug: str,
    asset_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    dataset_source = f"{owner}/{dataset_slug}"
    package_root = asset_root / dataset_source.replace("/", "__")
    if package_root.exists():
        shutil.rmtree(package_root)
    dataset_dir = package_root / "dataset"
    runtime_dir = dataset_dir / "runtime"
    gmm_dir = dataset_dir / "gmms"
    runtime_dir.mkdir(parents=True)
    gmm_dir.mkdir(parents=True)

    files = []
    for relative_path in RUNTIME_FILES:
        source = repo_root / relative_path
        destination = runtime_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "path": str(destination.relative_to(dataset_dir)),
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )

    public_selections = []
    for selection in selections:
        public_selection = {
            key: value
            for key, value in selection.items()
            if key != "local_source_path"
        }
        source = Path(selection["local_source_path"])
        destination = gmm_dir / f"{selection['label']}.npz"
        shutil.copy2(source, destination)
        if _sha256(destination) != selection["gmm_stats_sha256"]:
            raise RuntimeError(
                f"Staged GMM hash mismatch for {selection['label']}"
            )
        files.append(
            {
                "path": str(destination.relative_to(dataset_dir)),
                "bytes": destination.stat().st_size,
                "sha256": selection["gmm_stats_sha256"],
            }
        )
        public_selections.append(public_selection)

    selection_path = dataset_dir / "selection_manifest.json"
    selection_path.write_text(
        json.dumps(public_selections, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files.append(
        {
            "path": selection_path.name,
            "bytes": selection_path.stat().st_size,
            "sha256": _sha256(selection_path),
        }
    )
    asset_manifest = {
        "dataset_source": dataset_source,
        "files": sorted(files, key=lambda item: item["path"]),
    }
    asset_manifest_path = dataset_dir / "asset_manifest.json"
    asset_manifest_path.write_text(
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_path = dataset_dir / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "title": dataset_slug,
                "id": dataset_source,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    package_manifest = {
        "ok": True,
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_source": dataset_source,
        "message": "latent population FID6 analysis assets",
        "asset_manifest": asset_manifest,
    }
    package_manifest_path = package_root / "dataset_package_manifest.json"
    package_manifest_path.write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return package_manifest_path, dataset_dir, asset_manifest


def _make_notebook(
    *,
    config: dict[str, Any],
    selections: list[dict[str, Any]],
    asset_dataset_source: str,
    asset_manifest: dict[str, Any],
    repo_url: str,
    repo_commit: str,
    submit_accelerator: str,
) -> dict[str, Any]:
    public_selections = [
        {
            key: value
            for key, value in selection.items()
            if key != "local_source_path"
        }
        for selection in selections
    ]
    notebook_config = {
        **{key: value for key, value in config.items() if key != "selections"},
        "selections": public_selections,
        "repo_url": repo_url,
        "repo_commit": repo_commit,
        "submit_accelerator": submit_accelerator,
        "asset_dataset_source": asset_dataset_source,
        "asset_dataset_slug": asset_dataset_source.split("/", 1)[1],
    }
    cells = [
        _code_cell(
            "import json\n"
            "import os\n"
            "from pathlib import Path\n\n"
            f"CONFIG = json.loads({json.dumps(json.dumps(notebook_config))})\n"
            "OUTPUT_DIR = Path('/kaggle/working/latent_population_fid6')\n"
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
            "os.environ['MPLBACKEND'] = 'Agg'\n"
            "os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'\n"
            "os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'\n"
            "os.environ['JAX_TRACEBACK_FILTERING'] = 'off'\n"
            "print(json.dumps(CONFIG, indent=2, sort_keys=True))\n"
        ),
        _code_cell(
            "import os\n"
            "import subprocess\n\n"
            "subprocess.run(\n"
            "    'curl -LsSf https://astral.sh/uv/install.sh | sh',\n"
            "    shell=True,\n"
            "    check=True,\n"
            ")\n"
            "os.environ['PATH'] += ':/root/.local/bin'\n"
        ),
        _code_cell(
            "import hashlib\n"
            "import json\n"
            "import os\n"
            "import shutil\n"
            "import subprocess\n"
            "from pathlib import Path\n\n"
            f"ASSET_MANIFEST = json.loads({json.dumps(json.dumps(asset_manifest))})\n"
            "input_root = Path('/kaggle/input')\n"
            "asset_root = input_root / CONFIG['asset_dataset_slug']\n"
            "if not asset_root.is_dir():\n"
            "    candidates = [path.parent for path in input_root.rglob('asset_manifest.json')]\n"
            "    for candidate in candidates:\n"
            "        payload = json.loads((candidate / 'asset_manifest.json').read_text())\n"
            "        if payload.get('dataset_source') == CONFIG['asset_dataset_source']:\n"
            "            asset_root = candidate\n"
            "            break\n"
            "if not asset_root.is_dir():\n"
            "    mounted = sorted(str(path) for path in input_root.iterdir())\n"
            "    raise FileNotFoundError(\n"
            "        f'Asset dataset not mounted: {asset_root}; mounted={mounted}'\n"
            "    )\n"
            "for item in ASSET_MANIFEST['files']:\n"
            "    path = asset_root / item['path']\n"
            "    if not path.is_file():\n"
            "        raise FileNotFoundError(f'Missing asset: {path}')\n"
            "    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
            "    if digest != item['sha256']:\n"
            "        raise RuntimeError(f'Asset hash mismatch: {path}')\n"
            "repo = Path('/kaggle/working/shortcut-models')\n"
            "subprocess.run(['git', 'clone', CONFIG['repo_url'], str(repo)], check=True)\n"
            "os.chdir(repo)\n"
            "subprocess.run(['git', 'checkout', CONFIG['repo_commit']], check=True)\n"
            "for relative_path in [\n"
            "    'utils/stable_vae.py', 'utils/datasets.py',\n"
            "    'gmm_utils.py', 'latent_population.py', 'latent_geometry.py',\n"
            "    'analyze_latent_population.py',\n"
            "]:\n"
            "    destination = repo / relative_path\n"
            "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
            "    shutil.copy2(asset_root / 'runtime' / relative_path, destination)\n"
            "source_dir = Path('/kaggle/working/latent_population_sources')\n"
            "source_dir.mkdir(parents=True, exist_ok=True)\n"
            "for selection in CONFIG['selections']:\n"
            "    label = selection['label']\n"
            "    path = source_dir / f'{label}.npz'\n"
            "    shutil.copy2(asset_root / 'gmms' / f'{label}.npz', path)\n"
            "    selection['runtime_gmm_stats_path'] = str(path)\n"
            "(OUTPUT_DIR / 'selection_manifest.json').write_text(\n"
            "    json.dumps(CONFIG['selections'], indent=2, sort_keys=True) + '\\n',\n"
            "    encoding='utf-8',\n"
            ")\n"
        ),
        _code_cell(
            "import shutil\n"
            "from pathlib import Path\n\n"
            "input_root = Path('/kaggle/input')\n"
            "candidates = list(input_root.glob('shortcut-celebahq256/tensorflow_datasets'))\n"
            "if not candidates:\n"
            "    for info_path in input_root.rglob('celebahq256/*/dataset_info.json'):\n"
            "        candidates.append(info_path.parents[2])\n"
            "        break\n"
            "if not candidates:\n"
            "    raise FileNotFoundError('Attached CelebA-HQ TFDS dataset was not found')\n"
            "source = candidates[0]\n"
            "target = Path(CONFIG['tfds_data_dir'])\n"
            "if target.exists():\n"
            "    shutil.rmtree(target)\n"
            "shutil.copytree(source, target)\n"
            "if not any(target.glob('celebahq256/*/dataset_info.json')):\n"
            "    raise RuntimeError(f'Invalid TFDS root copied from {source}')\n"
            "print(f'TFDS source: {source} -> {target}')\n"
        ),
        _code_cell(
            "import json\n"
            "import os\n"
            "import subprocess\n"
            "from pathlib import Path\n\n"
            "ACCELERATOR_SHAPE_DEVICE_COUNT_HINTS = {'TpuV5E8': 8}\n"
            "repo = Path('/kaggle/working/shortcut-models')\n"
            "sync_out = OUTPUT_DIR / 'uv_sync_stdout.txt'\n"
            "sync_err = OUTPUT_DIR / 'uv_sync_stderr.txt'\n"
            "with sync_out.open('w') as out, sync_err.open('w') as err:\n"
            "    subprocess.run(\n"
            "        ['uv', 'sync', '--locked'], cwd=repo,\n"
            "        stdout=out, stderr=err, check=True,\n"
            "    )\n"
            "probe = subprocess.run(\n"
            "    ['uv', 'run', 'python', '-c',\n"
            "     'import json,jax; print(json.dumps({\"backend\":jax.default_backend(),\"devices\":[str(d) for d in jax.devices()]}))'],\n"
            "    cwd=repo, check=True, text=True, stdout=subprocess.PIPE,\n"
            ")\n"
            "accelerator = json.loads(probe.stdout.strip())\n"
            "accelerator.update({\n"
            "    'requested_accelerator': 'tpu',\n"
            "    'requested_accelerator_shape': CONFIG['submit_accelerator'],\n"
            "    'shape_device_count_hints': ACCELERATOR_SHAPE_DEVICE_COUNT_HINTS,\n"
            "    'expected_device_count_hint': ACCELERATOR_SHAPE_DEVICE_COUNT_HINTS['TpuV5E8'],\n"
            "    'visible_requested_device_count': len(accelerator['devices']),\n"
            "    'device_count_matches_shape_hint': len(accelerator['devices']) == 8,\n"
            "    'accelerator_evidence_quality': 'runtime_verified',\n"
            "    'device_counts_by_backend': {'tpu': len(accelerator['devices'])},\n"
            "})\n"
            "(OUTPUT_DIR / 'accelerator.json').write_text(\n"
            "    json.dumps(accelerator, indent=2) + '\\n', encoding='utf-8'\n"
            ")\n"
            "if accelerator['backend'] != 'tpu' or len(accelerator['devices']) < 8:\n"
            "    raise RuntimeError(f'Expected TPU v5e-8, got {accelerator}')\n"
            "print('KJO_ACCELERATOR_SUMMARY ' + json.dumps(accelerator, sort_keys=True))\n"
        ),
        _code_cell(
            "import os\n"
            "import subprocess\n"
            "from pathlib import Path\n\n"
            "repo = Path('/kaggle/working/shortcut-models')\n"
            "cmd = [\n"
            "    'uv', 'run', 'python', 'analyze_latent_population.py',\n"
            "    '--dataset_name', CONFIG['dataset_name'],\n"
            "    '--tfds_data_dir', CONFIG['tfds_data_dir'],\n"
            "    '--split', CONFIG['split'],\n"
            "    '--batch_size', str(CONFIG['batch_size']),\n"
            "    '--max_samples', str(CONFIG['max_samples']),\n"
            "    '--mean_epsilon', str(CONFIG['mean_epsilon']),\n"
            "    '--dead_variance_threshold', str(CONFIG['dead_variance_threshold']),\n"
            "    '--moment_backend', CONFIG['moment_backend'],\n"
            "    '--population_mode', CONFIG.get('population_mode', 'aggregated_posterior'),\n"
            "    '--output_dir', str(OUTPUT_DIR),\n"
            "    '--cache_dir', '/tmp/latent_population_fid6_cache',\n"
            "    '--keep_cache', '0',\n"
            "]\n"
            "for selection in CONFIG['selections']:\n"
            "    cmd.extend(['--gmm_stats_path', selection['runtime_gmm_stats_path']])\n"
            "    cmd.extend(['--gmm_label', selection['label']])\n"
            "if CONFIG.get('extended_geometry_diagnostics', 0):\n"
            "    geometry_flags = {\n"
            "        'extended_geometry_diagnostics': 1,\n"
            "        'geometry_seed': CONFIG['geometry_seed'],\n"
            "        'geometry_train_fraction': CONFIG['geometry_train_fraction'],\n"
            "        'geometry_split_half_repeats': CONFIG['geometry_split_half_repeats'],\n"
            "        'geometry_whitening_projections': CONFIG['geometry_whitening_projections'],\n"
            "        'geometry_whitening_eigen_floor_relative': CONFIG['geometry_whitening_eigen_floor_relative'],\n"
            "        'geometry_ppca_rank': CONFIG['geometry_ppca_rank'],\n"
            "        'geometry_local_pool_size': CONFIG['geometry_local_pool_size'],\n"
            "        'geometry_local_query_count': CONFIG['geometry_local_query_count'],\n"
            "        'geometry_local_neighbor_counts': CONFIG['geometry_local_neighbor_counts'],\n"
            "        'geometry_local_variance_fraction': CONFIG['geometry_local_variance_fraction'],\n"
            "        'geometry_heldout_gmm_modes': CONFIG['geometry_heldout_gmm_modes'],\n"
            "        'geometry_heldout_gmm_em_iters': CONFIG['geometry_heldout_gmm_em_iters'],\n"
            "        'geometry_heldout_gmm_chunk_size': CONFIG['geometry_heldout_gmm_chunk_size'],\n"
            "        'geometry_c2st_sample_count': CONFIG['geometry_c2st_sample_count'],\n"
            "        'geometry_c2st_batch_size': CONFIG['geometry_c2st_batch_size'],\n"
            "        'geometry_c2st_logistic_steps': CONFIG['geometry_c2st_logistic_steps'],\n"
            "        'geometry_c2st_mlp_steps': CONFIG['geometry_c2st_mlp_steps'],\n"
            "        'geometry_c2st_mlp_hidden_size': CONFIG['geometry_c2st_mlp_hidden_size'],\n"
            "        'geometry_c2st_learning_rate': CONFIG['geometry_c2st_learning_rate'],\n"
            "        'geometry_c2st_weight_decay': CONFIG['geometry_c2st_weight_decay'],\n"
            "        'geometry_knn_subset_size': CONFIG['geometry_knn_subset_size'],\n"
            "        'geometry_knn_k': CONFIG['geometry_knn_k'],\n"
            "    }\n"
            "    for name, value in geometry_flags.items():\n"
            "        cmd.extend([f'--{name}', str(value)])\n"
            "stdout_path = OUTPUT_DIR / 'analysis_stdout.txt'\n"
            "stderr_path = OUTPUT_DIR / 'analysis_stderr.txt'\n"
            "print(' '.join(cmd))\n"
            "with stdout_path.open('w') as out, stderr_path.open('w') as err:\n"
            "    result = subprocess.run(cmd, cwd=repo, stdout=out, stderr=err)\n"
            "print('\\n'.join(stdout_path.read_text(errors='replace').splitlines()[-100:]))\n"
            "print('\\n'.join(stderr_path.read_text(errors='replace').splitlines()[-100:]))\n"
            "result.check_returncode()\n"
        ),
        _code_cell(
            "import csv\n"
            "import json\n"
            "import shutil\n"
            "from pathlib import Path\n\n"
            "comparison_path = OUTPUT_DIR / 'gmm_comparison.csv'\n"
            "with comparison_path.open(newline='', encoding='utf-8') as handle:\n"
            "    rows = list(csv.DictReader(handle))\n"
            "print(json.dumps(rows, indent=2))\n"
            "shutil.rmtree('/kaggle/working/latent_population_sources', ignore_errors=True)\n"
            "shutil.rmtree('/tmp/latent_population_fid6_cache', ignore_errors=True)\n"
            "total_bytes = sum(\n"
            "    path.stat().st_size for path in OUTPUT_DIR.rglob('*') if path.is_file()\n"
            ")\n"
            "storage = {'output_bytes': total_bytes, 'output_gib': total_bytes / (1024 ** 3)}\n"
            "(OUTPUT_DIR / 'storage_summary.json').write_text(\n"
            "    json.dumps(storage, indent=2) + '\\n', encoding='utf-8'\n"
            ")\n"
            "if total_bytes >= 19 * 1024 ** 3:\n"
            "    raise RuntimeError(f'Output safety limit exceeded: {storage}')\n"
            "print(json.dumps(storage, indent=2))\n"
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
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _git_value(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _https_git_url(value: str) -> str:
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    if value.startswith("ssh://git@github.com/"):
        value = "https://github.com/" + value.removeprefix(
            "ssh://git@github.com/"
        )
    return value.removesuffix(".git")


def _tracking_repo_url(repo_root: Path) -> str:
    upstream = _git_value(
        repo_root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    remote_name = upstream.split("/", 1)[0]
    return _https_git_url(
        _git_value(repo_root, "remote", "get-url", remote_name)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage and submit the six-run latent population analysis."
    )
    parser.add_argument(
        "--config", default="configs/latent_population_fid6.json"
    )
    parser.add_argument(
        "--source-root",
        default="outputs/kaggle_jobs/latent_population_fid6_sources_20260724",
    )
    parser.add_argument("--owner", required=True)
    parser.add_argument("--accounts-file", default=str(DEFAULT_ACCOUNTS_FILE))
    parser.add_argument(
        "--job-root",
        default="outputs/kaggle_jobs/latent_population_fid6_20260724",
    )
    parser.add_argument(
        "--asset-root",
        default="outputs/kaggle_jobs/latent_population_fid6_assets_20260724",
    )
    parser.add_argument(
        "--asset-dataset-slug",
        default="latent-population-fid6-assets-20260724",
    )
    parser.add_argument(
        "--asset-mode",
        choices=("create", "version", "reuse"),
        default="create",
    )
    parser.add_argument("--asset-ready-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--registry", default=str(DEFAULT_REGISTRY)
    )
    parser.add_argument("--accelerator", default="TpuV5E8")
    parser.add_argument(
        "--kaggle-bin",
        default="/tmp/kaggle-cli-2.2.3-fixed/bin/kaggle",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config = json.loads((repo_root / args.config).read_text(encoding="utf-8"))
    selections = _prepare_selection(
        config, repo_root / args.source_root
    )
    asset_dataset_source = f"{args.owner}/{args.asset_dataset_slug}"
    package_manifest_path, asset_dataset_dir, asset_manifest = (
        _stage_asset_dataset(
            repo_root=repo_root,
            selections=selections,
            owner=args.owner,
            dataset_slug=args.asset_dataset_slug,
            asset_root=repo_root / args.asset_root,
        )
    )
    repo_url = _tracking_repo_url(repo_root)
    repo_commit = _git_value(repo_root, "rev-parse", "HEAD")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    slug_prefix = _slugify(
        f"{config['run_name']}-{args.owner}",
        max_length=34,
    )
    slug = f"{slug_prefix}-{timestamp}"
    kernel_id = f"{args.owner}/{slug}"
    run_dir = repo_root / args.job_root / kernel_id.replace("/", "__")
    submit_dir = run_dir / "submit"
    submit_dir.mkdir(parents=True, exist_ok=False)
    notebook_path = submit_dir / f"{slug}.ipynb"
    raw_notebook_path = submit_dir / f"{slug}.raw.ipynb"
    metadata_path = submit_dir / "kernel-metadata.json"
    selection_path = submit_dir / "selection_manifest.json"

    notebook = _make_notebook(
        config=config,
        selections=selections,
        asset_dataset_source=asset_dataset_source,
        asset_manifest=asset_manifest,
        repo_url=repo_url,
        repo_commit=repo_commit,
        submit_accelerator=args.accelerator,
    )
    raw_notebook_path.write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
    )
    _run_json(
        [
            "uv",
            "run",
            "python",
            str(KJO_SCRIPT),
            "instrument-notebook-logging",
            "--source-notebook",
            str(raw_notebook_path),
            "--out",
            str(notebook_path),
            "--run-id",
            config["run_name"],
            "--title",
            "Latent population and selected GMM diagnostics",
            "--overwrite",
        ]
    )
    raw_notebook_path.unlink()
    metadata = {
        "id": kernel_id,
        "title": slug,
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": True,
        "dataset_sources": [config["dataset_ref"], asset_dataset_source],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    selection_path.write_text(
        json.dumps(selections, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validation = _run_json(
        [
            "uv",
            "run",
            "python",
            str(KJO_SCRIPT),
            "validate-metadata",
            "--metadata",
            str(metadata_path),
            "--expected-accelerator",
            "tpu",
            "--submit-accelerator",
            args.accelerator,
            "--owner",
            args.owner,
            "--required-dataset-source",
            asset_dataset_source,
        ]
    )
    report = {
        "kernel_id": kernel_id,
        "run_dir": str(run_dir),
        "notebook": str(notebook_path),
        "metadata": str(metadata_path),
        "selection_manifest": str(selection_path),
        "selection_count": len(selections),
        "asset_dataset_source": asset_dataset_source,
        "asset_dataset_dir": str(asset_dataset_dir),
        "asset_manifest": asset_manifest,
        "metadata_validation": validation,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    with tempfile.TemporaryDirectory(prefix=f"kaggle-{args.owner}-") as temp_dir:
        _run_checked(
            [
                "uv",
                "run",
                "python",
                str(KJO_ACCOUNTS),
                "--accounts-file",
                args.accounts_file,
                "--owner",
                args.owner,
                "--out-dir",
                temp_dir,
            ]
        )
        asset_push = None
        if args.asset_mode != "reuse":
            asset_push = _run_json(
                [
                    "uv",
                    "run",
                    "python",
                    str(KJO_SCRIPT),
                    "push-dataset-package",
                    "--manifest",
                    str(package_manifest_path),
                    "--mode",
                    args.asset_mode,
                    "--kaggle-bin",
                    args.kaggle_bin,
                    "--kaggle-config-dir",
                    temp_dir,
                    "--timeout-s",
                    "900",
                ]
            )
            _wait_for_dataset_ready(
                kaggle_bin=args.kaggle_bin,
                kaggle_config_dir=temp_dir,
                dataset_source=asset_dataset_source,
                expected_paths=[
                    "asset_manifest.json",
                    *[
                        f"gmms/{selection['label']}.npz"
                        for selection in selections
                    ],
                ],
                timeout_seconds=args.asset_ready_timeout_s,
            )
        report["asset_push"] = asset_push
        submit = _run_json(
            [
                "uv",
                "run",
                "python",
                str(KJO_SCRIPT),
                "submit-kernel",
                "--run-dir",
                str(run_dir),
                "--metadata",
                str(metadata_path),
                "--expected-accelerator",
                "tpu",
                "--submit-accelerator",
                args.accelerator,
                "--owner",
                args.owner,
                "--submitted-notebook",
                str(notebook_path),
                "--allow-missing-kjo-markers",
                "--registry",
                args.registry,
                "--record-registry",
                args.registry,
                "--secret-mode",
                "none",
                "--is-private",
                "--artifact-mode",
                "has-artifacts",
                "--retention-action",
                "keep-while-artifacts-needed",
                "--remind-delete-after-days",
                "7",
                "--dataset-source",
                asset_dataset_source,
                "--kaggle-bin",
                args.kaggle_bin,
                "--kaggle-config-dir",
                temp_dir,
                "--run-id",
                config["run_name"],
                "--project-root",
                str(repo_root),
                "--title",
                slug,
                "--no-submit-delay",
            ]
        )
    report["submit"] = submit
    report_path = repo_root / "reports" / f"{config['run_name']}_submit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
