from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WANDB_PLACEHOLDER = '__WANDB_API_KEY_PLACEHOLDER__'


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r'[^a-z0-9]+', '-', value)
    return value.strip('-')[:48] or 'gmm-ablation'


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_grid(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    defaults = payload.get('defaults', {})
    grid = payload.get('grid', {})
    jobs = []
    for k_modes, min_std_frac, prior in itertools.product(
        grid.get('gmm_num_modes', [defaults.get('gmm_num_modes', 64)]),
        grid.get('gmm_min_std_data_frac', [defaults.get('gmm_min_std_data_frac', 1.0)]),
        grid.get('prior', [{'gmm_pi_prior_type': defaults.get('gmm_pi_prior_type', 'dirichlet'),
                            'gmm_pi_prior_strength': defaults.get('gmm_pi_prior_strength', 1e-2)}]),
    ):
        job = dict(defaults)
        job['gmm_num_modes'] = k_modes
        job['gmm_min_std_data_frac'] = min_std_frac
        job.update(prior)
        strength = str(job['gmm_pi_prior_strength']).replace('.', 'p')
        floor = str(min_std_frac).replace('.', 'p')
        job['run_name'] = slugify(
            f"gmm-k{k_modes}-floor{floor}-{job['gmm_pi_prior_type']}-s{strength}"
        )
        jobs.append(job)
    return defaults, jobs


def embed_notebook_config(notebook_path: Path, config: dict[str, Any], wandb_api_key: str) -> None:
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    config_line = (
        'EMBEDDED_ABLATION_CONFIG = '
        + json.dumps(config, indent=4, sort_keys=True)
        + '\n'
    ).splitlines(keepends=True)
    key_line = f'WANDB_API_KEY = {json.dumps(wandb_api_key)}\n'

    replaced_config = False
    replaced_key = False
    for cell in notebook.get('cells', []):
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if line == 'EMBEDDED_ABLATION_CONFIG = {}\n':
                new_source.extend(config_line)
                replaced_config = True
            elif line == f'WANDB_API_KEY = "{WANDB_PLACEHOLDER}"\n':
                new_source.append(key_line)
                replaced_key = True
            else:
                new_source.append(line)
        cell['source'] = new_source
        cell['outputs'] = []
        if cell.get('cell_type') == 'code':
            cell['execution_count'] = None
    if not replaced_config:
        raise RuntimeError('Could not find EMBEDDED_ABLATION_CONFIG placeholder')
    if wandb_api_key and not replaced_key:
        raise RuntimeError('Could not find WANDB_API_KEY placeholder')
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def stage_job(
    *,
    owner: str,
    notebook_template: Path,
    staging_root: Path,
    config: dict[str, Any],
    wandb_api_key: str,
    accelerator: str,
) -> tuple[Path, str]:
    config = dict(config)
    accelerator_lower = accelerator.lower()
    is_tpu = accelerator_lower.startswith('tpu')
    is_gpu = not is_tpu
    config['jax_runtime'] = 'tpu' if is_tpu else 'cuda12'

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')
    base_slug = slugify(f"{config['run_name']}-{owner}-{timestamp}")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / base_slug
    suffix = 2
    while staging_dir.exists():
        staging_dir = staging_root / f'{base_slug}-{suffix}'
        suffix += 1
    staging_dir.mkdir(parents=True, exist_ok=False)

    staged_notebook = staging_dir / notebook_template.name
    shutil.copy2(notebook_template, staged_notebook)
    embed_notebook_config(staged_notebook, config, wandb_api_key)

    metadata = {
        'id': f'{owner}/{staging_dir.name}',
        'title': staging_dir.name,
        'code_file': staged_notebook.name,
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': is_gpu,
        'enable_internet': True,
        'dataset_sources': [],
        'competition_sources': [],
        'kernel_sources': [],
        'model_sources': [],
    }
    (staging_dir / 'kernel-metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n',
        encoding='utf-8',
    )
    (staging_dir / 'ablation_config.json').write_text(
        json.dumps(config, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    (staging_dir / 'accelerator.txt').write_text(accelerator + '\n', encoding='utf-8')
    return staging_dir, metadata['id']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Stage GMM ablation Kaggle notebooks.')
    parser.add_argument('--owner', required=True, help='Kaggle username for kernel metadata.')
    parser.add_argument('--grid-config', default='configs/gmm_ablation_grid.json')
    parser.add_argument('--notebook', default='shortcut-model-gmm-ablation.ipynb')
    parser.add_argument('--env-file', default='.secrets/.env')
    parser.add_argument('--staging-root', default='kaggle_staging/gmm_ablation')
    parser.add_argument('--accelerator', default='NvidiaTeslaT4')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--offset', type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, jobs = load_grid(Path(args.grid_config))
    jobs = jobs[args.offset:]
    if args.limit:
        jobs = jobs[:args.limit]
    env_values = load_env_file(Path(args.env_file))
    wandb_api_key = env_values.get('WANDB_API_KEY', '')

    for job in jobs:
        staging_dir, kernel_id = stage_job(
            owner=args.owner,
            notebook_template=Path(args.notebook),
            staging_root=Path(args.staging_root),
            config=job,
            wandb_api_key=wandb_api_key,
            accelerator=args.accelerator,
        )
        print(f'{kernel_id}\t{staging_dir}')


if __name__ == '__main__':
    main()
