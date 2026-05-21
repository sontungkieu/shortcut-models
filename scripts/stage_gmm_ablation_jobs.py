from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WANDB_PLACEHOLDER = '__WANDB_API_KEY_PLACEHOLDER__'
BATCH_PLACEHOLDER = 'EMBEDDED_ABLATION_CONFIGS = []\n'


def normalize_accelerator(value: str) -> str:
    aliases = {
        'tpu': 'TpuV5E8',
        'v5e8': 'TpuV5E8',
        'tpuv5e8': 'TpuV5E8',
        'gpu': 'NvidiaTeslaT4',
        't4': 'NvidiaTeslaT4',
    }
    key = value.strip().lower().replace('-', '').replace('_', '')
    return aliases.get(key, value)


def slugify(value: str, max_length: int = 48) -> str:
    value = value.lower()
    value = re.sub(r'[^a-z0-9]+', '-', value)
    return value.strip('-')[:max_length].strip('-') or 'gmm-ablation'


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


def _append_gmm_init_suffix(run_name: str, job: dict[str, Any]) -> str:
    strategy = str(job.get('gmm_init_strategy', 'auto')).strip().lower().replace('_', '-')
    strategy_aliases = {
        'kmeanspp': 'kmeans++',
        'kpp': 'kmeans++',
        'kmeans': 'kmeans++',
        'default': 'auto',
    }
    strategy = strategy_aliases.get(strategy, strategy)
    warmup = int(job.get('gmm_init_warmup_iters', 0) or 0)
    restarts = int(job.get('gmm_em_restarts', 1) or 1)

    if strategy not in ('', 'auto', 'kmeans++'):
        run_name += f"-init{strategy.replace('+', 'p')}"
    if strategy == 'pca':
        run_name += f"-pca{int(job.get('gmm_init_pca_dims', 16) or 16)}"
    if warmup > 0:
        run_name += f"-lw{warmup}"
    if restarts > 1:
        run_name += f"-r{restarts}"
    return run_name


def load_grid(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    defaults = payload.get('defaults', {})
    explicit_jobs = payload.get('jobs', [])
    if explicit_jobs:
        jobs = []
        for raw_job in explicit_jobs:
            job = dict(defaults)
            job.update(raw_job)
            job['gmm_min_std'] = math.sqrt(max(float(job.get('gmm_min_var', 0.0)), 0.0))
            job['gmm_min_std_data_frac'] = math.sqrt(max(float(job.get('gmm_min_var_data_frac', 0.0)), 0.0))
            if 'run_name' not in job:
                k_modes = job.get('gmm_num_modes', defaults.get('gmm_num_modes', 64))
                strength = str(job.get('gmm_pi_prior_strength', defaults.get('gmm_pi_prior_strength', 1e-2))).replace('.', 'p')
                floor = str(job.get('gmm_min_var_data_frac', 0.0)).replace('.', 'p')
                run_name = f"gmm-k{k_modes}-floorv{floor}-{job.get('gmm_pi_prior_type', 'dirichlet')}-s{strength}"
                run_name += "-std" if int(job.get('gmm_standardize_data', 0)) else "-raw"
                coverage_name = str(job.get('coverage_name', '')).strip()
                if coverage_name and coverage_name != 'default':
                    run_name += f"-{coverage_name}"
                run_name = _append_gmm_init_suffix(run_name, job)
                job['run_name'] = slugify(run_name, max_length=96)
            else:
                job['run_name'] = slugify(str(job['run_name']), max_length=96)
            run_name = job['run_name']
            for suffix_key in ('ablation_tag', 'run_name_suffix'):
                suffix = str(job.get(suffix_key, '')).strip().strip('-')
                if suffix and suffix not in run_name:
                    run_name += f"-{suffix}"
            job['run_name'] = slugify(run_name, max_length=96)
            jobs.append(job)
        return defaults, jobs

    grid = payload.get('grid', {})
    jobs = []
    default_coverage = {
        'coverage_name': 'default',
        'gmm_min_var': defaults.get('gmm_min_var', defaults.get('gmm_min_std', 0.0) ** 2),
        'gmm_min_var_data_frac': defaults.get(
            'gmm_min_var_data_frac',
            defaults.get('gmm_min_std_data_frac', 1.0) ** 2,
        ),
        'gmm_var_prior_type': defaults.get('gmm_var_prior_type', 'none'),
        'gmm_var_prior_strength': defaults.get('gmm_var_prior_strength', 0.0),
        'gmm_var_prior_target_var': defaults.get('gmm_var_prior_target_var', 1.0),
        'gmm_standardize_data': defaults.get('gmm_standardize_data', 0),
    }
    if 'coverage' in grid:
        coverage_grid = grid['coverage']
    else:
        var_priors = grid.get(
            'var_prior',
            [{
                'gmm_var_prior_type': defaults.get('gmm_var_prior_type', 'none'),
                'gmm_var_prior_strength': defaults.get('gmm_var_prior_strength', 0.0),
                'gmm_var_prior_target_var': defaults.get('gmm_var_prior_target_var', 1.0),
                'gmm_standardize_data': defaults.get('gmm_standardize_data', 0),
            }],
        )
        coverage_grid = []
        for min_std_frac, var_prior in itertools.product(
            grid.get('gmm_min_std_data_frac', [defaults.get('gmm_min_std_data_frac', 1.0)]),
            var_priors,
        ):
            coverage = dict(default_coverage)
            coverage['gmm_min_var_data_frac'] = float(min_std_frac) ** 2
            coverage.update(var_prior)
            coverage_grid.append(coverage)

    for k_modes, prior, coverage in itertools.product(
        grid.get('gmm_num_modes', [defaults.get('gmm_num_modes', 64)]),
        grid.get('prior', [{'gmm_pi_prior_type': defaults.get('gmm_pi_prior_type', 'dirichlet'),
                            'gmm_pi_prior_strength': defaults.get('gmm_pi_prior_strength', 1e-2)}]),
        coverage_grid,
    ):
        job = dict(defaults)
        job['gmm_num_modes'] = k_modes
        job.update(default_coverage)
        job.update(coverage)
        job.update(prior)
        job['gmm_min_std'] = math.sqrt(max(float(job.get('gmm_min_var', 0.0)), 0.0))
        job['gmm_min_std_data_frac'] = math.sqrt(max(float(job.get('gmm_min_var_data_frac', 0.0)), 0.0))
        strength = str(job['gmm_pi_prior_strength']).replace('.', 'p')
        floor = str(job.get('gmm_min_var_data_frac', 0.0)).replace('.', 'p')
        run_name = f"gmm-k{k_modes}-floorv{floor}-{job['gmm_pi_prior_type']}-s{strength}"
        run_name += "-std" if int(job.get('gmm_standardize_data', 0)) else "-raw"
        coverage_name = str(job.get('coverage_name', '')).strip()
        if coverage_name and coverage_name != 'default':
            run_name += f"-{coverage_name}"
        var_type = str(job.get('gmm_var_prior_type', 'none')).lower()
        var_strength = float(job.get('gmm_var_prior_strength', 0.0))
        if var_type not in ('none', 'off', 'ml') and var_strength > 0:
            var_s = str(var_strength).replace('.', 'p')
            var_v = str(job.get('gmm_var_prior_target_var', 1.0)).replace('.', 'p')
            run_name += f"-var{var_type}-s{var_s}-v{var_v}"
        run_name = _append_gmm_init_suffix(run_name, job)
        for suffix_key in ('ablation_tag', 'run_name_suffix'):
            suffix = str(job.get(suffix_key, '')).strip().strip('-')
            if suffix and suffix not in run_name:
                run_name += f"-{suffix}"
        job['run_name'] = slugify(run_name, max_length=96)
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


def embed_notebook_batch_config(notebook_path: Path, configs: list[dict[str, Any]], wandb_api_key: str) -> None:
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    config_line = (
        'EMBEDDED_ABLATION_CONFIGS = '
        + json.dumps(configs, indent=4, sort_keys=True)
        + '\n'
    ).splitlines(keepends=True)
    key_line = f'WANDB_API_KEY = {json.dumps(wandb_api_key)}\n'

    replaced_config = False
    replaced_key = False
    for cell in notebook.get('cells', []):
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if line == BATCH_PLACEHOLDER:
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
        raise RuntimeError('Could not find EMBEDDED_ABLATION_CONFIGS placeholder')
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
    accelerator = normalize_accelerator(accelerator)
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


def stage_batch_job(
    *,
    owner: str,
    notebook_template: Path,
    staging_root: Path,
    configs: list[dict[str, Any]],
    wandb_api_key: str,
    accelerator: str,
) -> tuple[Path, str]:
    accelerator = normalize_accelerator(accelerator)
    accelerator_lower = accelerator.lower()
    is_tpu = accelerator_lower.startswith('tpu')
    is_gpu = not is_tpu

    staged_configs = []
    for config in configs:
        config = dict(config)
        config['jax_runtime'] = 'tpu' if is_tpu else 'cuda12'
        staged_configs.append(config)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')
    first_name = staged_configs[0].get('run_name', 'gmm-ablation')
    base_slug = slugify(f"batch{len(staged_configs)}-{first_name}-{owner}-{timestamp}")
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / base_slug
    suffix = 2
    while staging_dir.exists():
        staging_dir = staging_root / f'{base_slug}-{suffix}'
        suffix += 1
    staging_dir.mkdir(parents=True, exist_ok=False)

    staged_notebook = staging_dir / notebook_template.name
    shutil.copy2(notebook_template, staged_notebook)
    embed_notebook_batch_config(staged_notebook, staged_configs, wandb_api_key)

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
    (staging_dir / 'ablation_configs.json').write_text(
        json.dumps(staged_configs, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    (staging_dir / 'accelerator.txt').write_text(accelerator + '\n', encoding='utf-8')
    return staging_dir, metadata['id']


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'batches': rows,
        'summary': {
            'notebooks': len(rows),
            'configs': sum(len(row.get('grid_indexes', [])) for row in rows),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    md_path = path.with_suffix('.md')
    lines = [
        '# GMM Ablation Batch Manifest',
        '',
        f"- Notebooks: {payload['summary']['notebooks']}",
        f"- Configs: {payload['summary']['configs']}",
        '',
        '| batch | kernel | configs | grids | runs | staging |',
        '|---:|---|---:|---|---|---|',
    ]
    for index, row in enumerate(rows):
        runs = ', '.join(row.get('run_names', []))
        grids = ', '.join(str(item) for item in row.get('grid_indexes', []))
        lines.append(
            f"| {index} | `{row.get('kernel_id', '')}` | {len(row.get('grid_indexes', []))} | "
            f"{grids} | {runs} | `{row.get('staging_dir', '')}` |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Stage GMM ablation Kaggle notebooks.')
    parser.add_argument('--owner', required=True, help='Kaggle username for kernel metadata.')
    parser.add_argument('--grid-config', default='configs/gmm_ablation_grid.json')
    parser.add_argument('--notebook', default='shortcut-model-gmm-ablation.ipynb')
    parser.add_argument('--batch-notebook', default='shortcut-model-gmm-ablation-batch.ipynb')
    parser.add_argument('--env-file', default='.secrets/.env')
    parser.add_argument('--staging-root', default='kaggle_staging/gmm_ablation')
    parser.add_argument('--accelerator', default='NvidiaTeslaT4')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--manifest-path', default='', help='Optional JSON manifest for rendered notebooks.')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, jobs = load_grid(Path(args.grid_config))
    selected_jobs = list(enumerate(jobs))[args.offset:]
    if args.limit:
        selected_jobs = selected_jobs[:args.limit]
    env_values = load_env_file(Path(args.env_file))
    wandb_api_key = env_values.get('WANDB_API_KEY', '')

    manifest_rows: list[dict[str, Any]] = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(selected_jobs), batch_size):
        batch_items = selected_jobs[start:start + batch_size]
        grid_indexes = [grid_index for grid_index, _ in batch_items]
        batch = []
        for grid_index, job in batch_items:
            job = dict(job)
            job['grid_index'] = grid_index
            batch.append(job)
        if len(batch) == 1:
            staging_dir, kernel_id = stage_job(
                owner=args.owner,
                notebook_template=Path(args.notebook),
                staging_root=Path(args.staging_root),
                config=batch[0],
                wandb_api_key=wandb_api_key,
                accelerator=args.accelerator,
            )
        else:
            staging_dir, kernel_id = stage_batch_job(
                owner=args.owner,
                notebook_template=Path(args.batch_notebook),
                staging_root=Path(args.staging_root),
                configs=batch,
                wandb_api_key=wandb_api_key,
                accelerator=args.accelerator,
            )
        print(f'{kernel_id}\t{staging_dir}')
        manifest_rows.append(
            {
                'kernel_id': kernel_id,
                'owner': args.owner,
                'accelerator': normalize_accelerator(args.accelerator),
                'staging_dir': str(staging_dir),
                'grid_indexes': grid_indexes,
                'run_names': [job['run_name'] for job in batch],
                'configs': batch,
            }
        )

    if args.manifest_path:
        write_manifest(Path(args.manifest_path), manifest_rows)
        print(f'Wrote {args.manifest_path}')


if __name__ == '__main__':
    main()
