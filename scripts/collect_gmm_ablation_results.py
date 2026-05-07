from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from push_gmm_ablation_jobs import (
    DEFAULT_ACCOUNTS_FILE,
    kaggle_command,
    load_kaggle_accounts,
    parse_kernel_status,
)
from stage_gmm_ablation_jobs import load_grid


DEFAULT_FILE_PATTERN = (
    r'.*(gmm_metrics\.json|gmm_em_metrics\.jsonl|'
    r'gmm_prep_stdout\.txt|gmm_prep_stderr\.txt|batch_summary\.jsonl)$'
)


def normalize_status(value: str) -> str:
    value = value.strip().strip('"')
    if not value:
        return ''
    return value.replace('KernelWorkerStatus.', '')


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r'[^a-z0-9]+', '-', value)
    return value.strip('-') or 'kernel'


def report_jobs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    jobs = []
    for row in payload.get('jobs', []):
        kernel_id = row.get('kernel_id') or row.get('ref')
        if not kernel_id:
            continue
        jobs.append(
            {
                'kernel_id': kernel_id,
                'owner': row.get('owner') or kernel_id.split('/', 1)[0],
                'run_name': row.get('run_name', ''),
                'grid_index': row.get('grid_index'),
                'source_report': str(path),
                'submitted_status': normalize_status(str(row.get('kaggle_status', ''))),
                'url': row.get('url') or f'https://www.kaggle.com/code/{kernel_id}',
                'batch_size': row.get('batch_size'),
                'batch_member_index': row.get('batch_member_index'),
            }
        )
    for row in payload.get('submitted_tpu_jobs', []):
        kernel_id = row.get('ref') or row.get('kernel_id')
        if not kernel_id:
            continue
        jobs.append(
            {
                'kernel_id': kernel_id,
                'owner': row.get('owner') or kernel_id.split('/', 1)[0],
                'run_name': row.get('run_name', ''),
                'grid_index': row.get('grid_index'),
                'source_report': str(path),
                'submitted_status': normalize_status(str(row.get('status', ''))),
                'url': row.get('url') or f'https://www.kaggle.com/code/{kernel_id}',
            }
        )
    for row in payload.get('submitted', []):
        kernel_id = row.get('kernel_id') or row.get('ref')
        if not kernel_id:
            continue
        jobs.append(
            {
                'kernel_id': kernel_id,
                'owner': row.get('owner') or kernel_id.split('/', 1)[0],
                'run_name': row.get('run_name', ''),
                'grid_index': row.get('grid_index'),
                'source_report': str(path),
                'submitted_status': normalize_status(str(row.get('kernel_status', ''))),
                'url': row.get('url') or f'https://www.kaggle.com/code/{kernel_id}',
            }
        )
    return jobs


def collect_jobs(paths: list[Path], grid_path: Path) -> list[dict[str, Any]]:
    _, grid_jobs = load_grid(grid_path)
    by_job: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for job in report_jobs(path):
            dedupe_key = (job['kernel_id'], str(job.get('grid_index') or job.get('run_name') or ''))
            existing = by_job.get(dedupe_key, {})
            merged = {**existing, **job}
            grid_index = merged.get('grid_index')
            if isinstance(grid_index, int) and 0 <= grid_index < len(grid_jobs):
                config = grid_jobs[grid_index]
                merged.setdefault('run_name', config.get('run_name', ''))
                for key in (
                    'gmm_num_modes',
                    'gmm_min_var',
                    'gmm_min_var_data_frac',
                    'gmm_min_std_data_frac',
                    'gmm_pi_prior_type',
                    'gmm_pi_prior_strength',
                    'gmm_var_prior_type',
                    'gmm_var_prior_strength',
                    'gmm_var_prior_target_var',
                    'gmm_standardize_data',
                ):
                    merged[key] = config.get(key)
            by_job[dedupe_key] = merged
    return sorted(by_job.values(), key=lambda row: (row['owner'], row['kernel_id'], row.get('grid_index') or -1))


def credential_path(credential: dict[str, str], config_dir: Path) -> Path:
    config_path = config_dir / 'kaggle.json'
    config_path.write_text(json.dumps(credential) + '\n', encoding='utf-8')
    config_path.chmod(0o600)
    return config_path


def run_kaggle(args: list[str], credential: dict[str, str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix=f"kaggle-config-{credential['username']}-") as config_dir:
        credential_path(credential, Path(config_dir))
        env = os.environ.copy()
        env['KAGGLE_CONFIG_DIR'] = config_dir
        return subprocess.run(
            [*kaggle_command(), *args],
            check=False,
            env=env,
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True,
        )


def kernel_status(kernel_id: str, credential: dict[str, str]) -> tuple[str, str]:
    result = run_kaggle(['kernels', 'status', kernel_id], credential)
    if result.returncode != 0:
        return 'STATUS_FAILED', result.stdout.strip()
    return normalize_status(parse_kernel_status(result.stdout)), result.stdout.strip()


def has_required_logs(output_dir: Path) -> bool:
    return (find_first(output_dir, 'gmm_metrics.json') is not None
            and find_first(output_dir, 'gmm_em_metrics.jsonl') is not None)


def download_logs(
    *,
    kernel_id: str,
    credential: dict[str, str],
    output_dir: Path,
    file_pattern: str,
    force: bool,
) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if has_required_logs(output_dir) and not force:
        return True, 'already_downloaded'
    result = run_kaggle(
        [
            'kernels',
            'output',
            kernel_id,
            '-p',
            str(output_dir),
            '--file-pattern',
            file_pattern,
            '-o',
            '-q',
        ],
        credential,
    )
    return result.returncode == 0, result.stdout.strip()


def find_first(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def find_for_run(root: Path, name: str, run_name: str | None) -> Path | None:
    matches = sorted(root.rglob(name))
    if not matches:
        return None
    if not run_name:
        return matches[0]
    for path in matches:
        if run_name in path.parts:
            return path
    for path in matches:
        if run_name in str(path):
            return path
    return matches[0]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def extract_wandb_url(stdout_path: Path | None, stderr_path: Path | None) -> str:
    pattern = re.compile(r'https://wandb\.ai/\S+')
    for path in [stdout_path, stderr_path]:
        if not path or not path.exists():
            continue
        for match in pattern.findall(path.read_text(encoding='utf-8', errors='replace')):
            return match.rstrip('\x1b[0m')
    return ''


def count_critical_matches(stdout_path: Path | None, stderr_path: Path | None) -> int:
    pattern = re.compile(r'Traceback|Exception|TypeError|ValueError|RuntimeError|ERROR|FAILED', re.I)
    count = 0
    for path in [stdout_path, stderr_path]:
        if not path or not path.exists():
            continue
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            if pattern.search(line):
                count += 1
    return count


def parse_result(output_dir: Path, run_name: str | None = None) -> dict[str, Any]:
    metrics_path = find_for_run(output_dir, 'gmm_metrics.json', run_name)
    em_path = find_for_run(output_dir, 'gmm_em_metrics.jsonl', run_name)
    stdout_path = find_for_run(output_dir, 'gmm_prep_stdout.txt', run_name)
    stderr_path = find_for_run(output_dir, 'gmm_prep_stderr.txt', run_name)
    if not metrics_path:
        return {'parse_status': 'missing_gmm_metrics'}
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    trace = load_jsonl(em_path) if em_path else []
    first_trace = trace[0] if trace else {}
    last_trace = trace[-1] if trace else {}
    min_std = float(metrics.get('gmm_min_std') or 0.0)
    min_std_data_frac = float(metrics.get('gmm_min_std_data_frac') or 0.0)
    data_variance_mean = metrics.get('data_variance_mean')
    standardize_data = bool(metrics.get('gmm_standardize_data', 0))
    if standardize_data:
        rel_floor_var = min_std_data_frac ** 2
        if data_variance_mean is not None and data_variance_mean > 0:
            abs_floor_var_std_approx = (min_std ** 2) / float(data_variance_mean)
        else:
            abs_floor_var_std_approx = min_std ** 2
        floor_var_std_approx = max(abs_floor_var_std_approx, rel_floor_var)
        floor_var_latent_mean_approx = (
            floor_var_std_approx * float(data_variance_mean)
            if data_variance_mean is not None
            else None
        )
    else:
        floor_var_latent_mean_approx = None
        floor_var_std_approx = None
        if data_variance_mean is not None and data_variance_mean > 0:
            floor_var_latent_mean_approx = max(min_std ** 2, (min_std_data_frac ** 2) * float(data_variance_mean))
            floor_var_std_approx = floor_var_latent_mean_approx / float(data_variance_mean)

    return {
        'parse_status': 'ok',
        'metrics_path': str(metrics_path),
        'em_metrics_path': str(em_path) if em_path else '',
        'stdout_path': str(stdout_path) if stdout_path else '',
        'stderr_path': str(stderr_path) if stderr_path else '',
        'wandb_url': extract_wandb_url(stdout_path, stderr_path),
        'critical_log_match_count': count_critical_matches(stdout_path, stderr_path),
        'dataset_name': metrics.get('dataset_name'),
        'num_modes': metrics.get('num_modes'),
        'fit_samples': metrics.get('fit_samples'),
        'valid_samples': metrics.get('valid_samples'),
        'gmm_min_std_data_frac': metrics.get('gmm_min_std_data_frac'),
        'gmm_min_std': metrics.get('gmm_min_std'),
        'floor_var_std_approx': floor_var_std_approx,
        'floor_var_latent_mean_approx': floor_var_latent_mean_approx,
        'gmm_pi_prior_type': metrics.get('gmm_pi_prior_type'),
        'gmm_pi_prior_strength': metrics.get('gmm_pi_prior_strength'),
        'gmm_var_prior_type': metrics.get('gmm_var_prior_type'),
        'gmm_var_prior_strength': metrics.get('gmm_var_prior_strength'),
        'gmm_var_prior_target_var': metrics.get('gmm_var_prior_target_var'),
        'gmm_standardize_data': metrics.get('gmm_standardize_data'),
        'gmm_fit_space': metrics.get('gmm_fit_space'),
        'train_nll': metrics.get('train_nll'),
        'valid_nll': metrics.get('valid_nll'),
        'final_train_nll': metrics.get('final_train_nll'),
        'em_first_nll': first_trace.get('nll'),
        'em_last_nll': last_trace.get('nll'),
        'em_nll_delta': (
            first_trace.get('nll') - last_trace.get('nll')
            if first_trace.get('nll') is not None and last_trace.get('nll') is not None
            else None
        ),
        'pi_min': metrics.get('pi_min'),
        'pi_max': metrics.get('pi_max'),
        'pi_entropy_normalized': metrics.get('pi_entropy_normalized'),
        'pi_kl_to_uniform': metrics.get('pi_kl_to_uniform'),
        'pi_mse_to_uniform': metrics.get('pi_mse_to_uniform'),
        'train_count_min': metrics.get('train_count_min'),
        'train_count_max': metrics.get('train_count_max'),
        'train_count_gap': metrics.get('train_count_gap'),
        'train_count_ratio': metrics.get('train_count_ratio'),
        'valid_count_min': metrics.get('valid_count_min'),
        'valid_count_max': metrics.get('valid_count_max'),
        'valid_count_gap': metrics.get('valid_count_gap'),
        'valid_count_ratio': metrics.get('valid_count_ratio'),
        'train_dead_components': metrics.get('train_dead_components'),
        'valid_dead_components': metrics.get('valid_dead_components'),
        'var_floor_hit_rate': metrics.get('var_floor_hit_rate'),
        'data_variance_mean': data_variance_mean,
        'component_variance_min': metrics.get('component_variance_min'),
        'component_variance_mean': metrics.get('component_variance_mean'),
        'component_variance_max': metrics.get('component_variance_max'),
        'center_distance_min': metrics.get('center_distance_min'),
        'center_distance_mean': metrics.get('center_distance_mean'),
        'center_distance_max': metrics.get('center_distance_max'),
        'overlap_proxy_mean': metrics.get('overlap_proxy_mean'),
        'overlap_proxy_max': metrics.get('overlap_proxy_max'),
        'overlap_proxy_pair_fraction_gt_0_5': metrics.get('overlap_proxy_pair_fraction_gt_0_5'),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    md_path = path.with_suffix('.md')
    lines = [
        '# GMM Ablation Results',
        '',
        f"- Generated: {payload['generated_at_utc']}",
        f"- Jobs: {payload['summary']['jobs']}",
        f"- COMPLETE: {payload['summary']['status_counts'].get('COMPLETE', 0)}",
        f"- Parsed: {payload['summary']['parsed']}",
        f"- Missing metrics: {payload['summary']['missing_metrics']}",
        '',
        '## Job Status',
        '',
        '| owner | grid | run | status | parsed | kernel |',
        '|---|---:|---|---|---|---|',
    ]
    for row in payload['jobs']:
        lines.append(
            f"| {row['owner']} | {row.get('grid_index', '')} | {row.get('run_name', '')} | "
            f"{row.get('latest_status', '')} | {row.get('parse_status', '')} | "
            f"`{row['kernel_id']}` |"
        )
    lines.extend([
        '',
        '## Parsed Metrics',
        '',
        '| owner | grid | run | fit_space | var_prior | train_nll | valid_nll | data_var | floor_var_std | floor_var_latent | comp_var_mean | pi_entropy_norm | pi_kl | pi_min | pi_max | dead(train/valid) | count_ratio(train/valid) | floor_hit | overlap_max |',
        '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ])
    for row in payload['jobs']:
        if row.get('parse_status') != 'ok':
            continue
        lines.append(
            f"| {row['owner']} | {row.get('grid_index', '')} | {row.get('run_name', '')} | "
            f"{row.get('gmm_fit_space', '')} | "
            f"{row.get('gmm_var_prior_type', '')}:{fmt(row.get('gmm_var_prior_strength'), 1)}@{fmt(row.get('gmm_var_prior_target_var'), 2)} | "
            f"{fmt(row.get('train_nll'), 2)} | {fmt(row.get('valid_nll'), 2)} | "
            f"{fmt(row.get('data_variance_mean'), 6)} | "
            f"{fmt(row.get('floor_var_std_approx'), 6)} | "
            f"{fmt(row.get('floor_var_latent_mean_approx'), 6)} | "
            f"{fmt(row.get('component_variance_mean'), 6)} | "
            f"{fmt(row.get('pi_entropy_normalized'), 6)} | {fmt(row.get('pi_kl_to_uniform'), 6)} | "
            f"{fmt(row.get('pi_min'))} | {fmt(row.get('pi_max'))} | "
            f"{row.get('train_dead_components', '')}/{row.get('valid_dead_components', '')} | "
            f"{fmt(row.get('train_count_ratio'))}/{fmt(row.get('valid_count_ratio'))} | "
            f"{fmt(row.get('var_floor_hit_rate'))} | {fmt(row.get('overlap_proxy_max'), 6)} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Check Kaggle GMM ablation jobs, download diagnostic logs, and summarize metrics.'
    )
    parser.add_argument(
        '--submit-report',
        action='append',
        default=[],
        help='Submit/reconcile report JSON. Can be passed multiple times.',
    )
    parser.add_argument('--accounts-file', default=str(DEFAULT_ACCOUNTS_FILE))
    parser.add_argument('--grid-config', default='configs/gmm_ablation_grid.json')
    parser.add_argument('--output-root', default='outputs/kaggle/gmm_ablation_results')
    parser.add_argument('--report-path', default='reports/gmm_ablation_results.json')
    parser.add_argument('--file-pattern', default=DEFAULT_FILE_PATTERN)
    parser.add_argument('--force-download', action='store_true')
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--complete-status', default='COMPLETE')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report_paths = [Path(item) for item in args.submit_report]
    if not report_paths:
        report_paths = sorted(Path('reports').glob('gmm_ablation*_20260507.json'))
    if not report_paths:
        raise SystemExit('No submit reports provided or found.')

    accounts = load_kaggle_accounts(Path(args.accounts_file))
    jobs = collect_jobs(report_paths, Path(args.grid_config))
    output_root = Path(args.output_root)
    complete_status = normalize_status(args.complete_status)

    result_rows = []
    for job in jobs:
        row = dict(job)
        credential = accounts.get(row['owner'])
        output_dir = output_root / slugify(row['kernel_id'])
        row['output_dir'] = str(output_dir)
        if not credential:
            row['latest_status'] = 'NO_CREDENTIAL'
            row['status_output'] = ''
            row['parse_status'] = 'not_checked'
            result_rows.append(row)
            continue
        status, status_output = kernel_status(row['kernel_id'], credential)
        row['latest_status'] = status
        row['status_output'] = status_output
        if status == complete_status and not args.skip_download:
            ok, download_output = download_logs(
                kernel_id=row['kernel_id'],
                credential=credential,
                output_dir=output_dir,
                file_pattern=args.file_pattern,
                force=args.force_download,
            )
            row['download_status'] = 'ok' if ok else 'failed'
            row['download_output'] = download_output
        elif status == complete_status:
            row['download_status'] = 'skipped'
        else:
            row['download_status'] = 'not_complete'

        if status == complete_status:
            row.update(parse_result(output_dir, row.get('run_name')))
        else:
            row['parse_status'] = 'not_complete'
        result_rows.append(row)
        print(
            f"{row['owner']} {row['kernel_id']} status={row['latest_status']} "
            f"download={row.get('download_status')} parse={row.get('parse_status')}",
            flush=True,
        )

    status_counts = Counter(row.get('latest_status', '') for row in result_rows)
    parse_counts = Counter(row.get('parse_status', '') for row in result_rows)
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_reports': [str(path) for path in report_paths],
        'output_root': str(output_root),
        'summary': {
            'jobs': len(result_rows),
            'status_counts': dict(status_counts),
            'parse_counts': dict(parse_counts),
            'parsed': parse_counts.get('ok', 0),
            'missing_metrics': parse_counts.get('missing_gmm_metrics', 0),
        },
        'jobs': result_rows,
    }
    write_report(Path(args.report_path), payload)
    print(f"Wrote {args.report_path}")
    print(f"Wrote {Path(args.report_path).with_suffix('.md')}")


if __name__ == '__main__':
    main()
