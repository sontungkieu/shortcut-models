from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from stage_gmm_ablation_jobs import load_env_file, load_grid, normalize_accelerator, stage_job


DEFAULT_ACCOUNTS_FILE = Path('.secrets/all-kaggle.json')
KAGGLE_CODE_URL_RE = re.compile(r'https://www\.kaggle\.com/code/([^\s/]+)/([^\s]+)')


def load_kaggle_accounts(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise SystemExit(f'Kaggle account file not found: {path}')
    text = path.read_text(encoding='utf-8')
    accounts: dict[str, dict[str, str]] = {}
    try:
        parsed = json.loads(text)
        records = parsed.values() if isinstance(parsed, dict) else parsed
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            records.append(json.loads(line))
    for record in records:
        if not isinstance(record, dict):
            continue
        username = str(record.get('username') or record.get('KAGGLE_USERNAME') or '')
        key = str(record.get('key') or record.get('KAGGLE_KEY') or '')
        if username and key:
            accounts[username] = {'username': username, 'key': key}
    return accounts


def kaggle_command() -> list[str]:
    executable = shutil.which('kaggle')
    if executable:
        return [executable]
    uvx = shutil.which('uvx')
    if uvx:
        return [uvx, '--from', 'kaggle', 'kaggle']
    uv = shutil.which('uv')
    if uv:
        return [uv, 'tool', 'run', '--from', 'kaggle', 'kaggle']
    raise SystemExit('Neither kaggle CLI nor uv/uvx is available in PATH.')


def selected_owners(value: str, available: list[str]) -> list[str]:
    if value == 'all':
        return sorted(available)
    owners = [item.strip() for item in value.split(',') if item.strip()]
    missing = [owner for owner in owners if owner not in available]
    if missing:
        raise SystemExit(f"Unknown Kaggle owner(s): {', '.join(missing)}")
    return owners


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attempted = report['submitted'] + report['failed']
    attempted_ids = {row['grid_index'] for row in attempted}
    report['not_submitted'] = [
        row for row in report['planned'] if row['grid_index'] not in attempted_ids
    ]
    report['summary'] = {
        'planned': len(report['planned']),
        'submitted': len(report['submitted']),
        'failed': len(report['failed']),
        'not_submitted': len(report['not_submitted']),
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    md_path = path.with_suffix('.md')
    lines = [
        '# GMM Ablation Submit Report',
        '',
        f"- Planned: {report['summary']['planned']}",
        f"- Submitted: {report['summary']['submitted']}",
        f"- Failed: {report['summary']['failed']}",
        f"- Not submitted: {report['summary']['not_submitted']}",
        '',
        '## Submitted',
        '',
        '| grid_index | owner | accelerator | run_name | kernel | status |',
        '|---:|---|---|---|---|---|',
    ]
    for row in report['submitted']:
        lines.append(
            f"| {row['grid_index']} | {row['owner']} | {row['accelerator']} | "
            f"{row['run_name']} | `{row['kernel_id']}` | {row.get('kernel_status', '')} |"
        )
    lines.extend(['', '## Failed', '', '| grid_index | owner | run_name | error |', '|---:|---|---|---|'])
    for row in report['failed']:
        error = str(row.get('error', '')).replace('\n', '<br>')
        lines.append(f"| {row['grid_index']} | {row['owner']} | {row['run_name']} | {error} |")
    lines.extend(['', '## Not Submitted', '', '| grid_index | owner | run_name |', '|---:|---|---|'])
    for row in report['not_submitted']:
        lines.append(f"| {row['grid_index']} | {row['owner']} | {row['run_name']} |")
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_kernel_status(output: str) -> str:
    match = re.search(r'has status "([^"]+)"', output)
    return match.group(1) if match else ''


def parse_kernel_id(output: str, fallback: str) -> str:
    match = KAGGLE_CODE_URL_RE.search(output)
    return f'{match.group(1)}/{match.group(2)}' if match else fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Stage and push GMM ablation Kaggle GPU notebooks.')
    parser.add_argument('--owners', default='codemaivanngu', help="Comma-separated owners or 'all'.")
    parser.add_argument('--exclude-owners', default='', help='Comma-separated owners to remove after selection.')
    parser.add_argument('--accounts-file', default=str(DEFAULT_ACCOUNTS_FILE))
    parser.add_argument('--env-file', default='.secrets/.env')
    parser.add_argument('--grid-config', default='configs/gmm_ablation_grid.json')
    parser.add_argument('--notebook', default='shortcut-model-gmm-ablation.ipynb')
    parser.add_argument('--staging-root', default='kaggle_staging/gmm_ablation')
    parser.add_argument('--accelerator', default='NvidiaTeslaT4')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--keep-staging', action='store_true')
    parser.add_argument('--report-path', default='kaggle_staging/gmm_ablation_submit_report.json')
    parser.add_argument('--stop-on-error', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.accelerator = normalize_accelerator(args.accelerator)
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    owners = selected_owners(args.owners, sorted(accounts))
    excluded = {item.strip() for item in args.exclude_owners.split(',') if item.strip()}
    owners = [owner for owner in owners if owner not in excluded]
    if not owners:
        raise SystemExit('No owners selected after applying --exclude-owners.')
    _, jobs = load_grid(Path(args.grid_config))
    selected_jobs = list(enumerate(jobs))[args.offset:]
    if args.limit:
        selected_jobs = selected_jobs[:args.limit]
    if not selected_jobs:
        raise SystemExit('No jobs selected.')

    env_values = load_env_file(Path(args.env_file))
    wandb_api_key = env_values.get('WANDB_API_KEY', '')
    report_path = Path(args.report_path)
    report: dict[str, Any] = {
        'accelerator': args.accelerator,
        'owners': owners,
        'excluded_owners': sorted(excluded),
        'planned': [],
        'submitted': [],
        'failed': [],
        'not_submitted': [],
        'summary': {},
    }

    for local_index, (grid_index, job) in enumerate(selected_jobs):
        owner = owners[local_index % len(owners)]
        report['planned'].append(
            {
                'grid_index': grid_index,
                'owner': owner,
                'run_name': job['run_name'],
                'accelerator': args.accelerator,
                'gmm_num_modes': job['gmm_num_modes'],
                'gmm_min_var': job.get('gmm_min_var'),
                'gmm_min_var_data_frac': job.get('gmm_min_var_data_frac'),
                'gmm_min_std_data_frac': job['gmm_min_std_data_frac'],
                'gmm_pi_prior_type': job['gmm_pi_prior_type'],
                'gmm_pi_prior_strength': job['gmm_pi_prior_strength'],
                'gmm_var_prior_type': job.get('gmm_var_prior_type'),
                'gmm_var_prior_strength': job.get('gmm_var_prior_strength'),
                'gmm_var_prior_target_var': job.get('gmm_var_prior_target_var'),
                'gmm_standardize_data': job.get('gmm_standardize_data'),
            }
        )
    write_report(report_path, report)

    for local_index, (grid_index, job) in enumerate(selected_jobs):
        owner = owners[local_index % len(owners)]
        staging_dir, kernel_id = stage_job(
            owner=owner,
            notebook_template=Path(args.notebook),
            staging_root=Path(args.staging_root),
            config=job,
            wandb_api_key=wandb_api_key,
            accelerator=args.accelerator,
        )
        print(f'Staged {kernel_id} at {staging_dir}', flush=True)
        if args.dry_run:
            try:
                report['submitted'].append(
                    {
                        'grid_index': grid_index,
                        'owner': owner,
                        'run_name': job['run_name'],
                        'accelerator': args.accelerator,
                        'gmm_var_prior_type': job.get('gmm_var_prior_type'),
                        'gmm_var_prior_strength': job.get('gmm_var_prior_strength'),
                        'gmm_var_prior_target_var': job.get('gmm_var_prior_target_var'),
                        'gmm_standardize_data': job.get('gmm_standardize_data'),
                        'kernel_id': kernel_id,
                        'kernel_status': 'DRY_RUN',
                    }
                )
                write_report(report_path, report)
            finally:
                if not args.keep_staging:
                    shutil.rmtree(staging_dir, ignore_errors=True)
            continue

        try:
            credential = accounts[owner]
            with tempfile.TemporaryDirectory(prefix=f'kaggle-config-{owner}-') as config_dir:
                config_path = Path(config_dir) / 'kaggle.json'
                config_path.write_text(json.dumps(credential) + '\n', encoding='utf-8')
                config_path.chmod(0o600)
                command_env = os.environ.copy()
                command_env['KAGGLE_CONFIG_DIR'] = config_dir
                push_cmd = [
                    *kaggle_command(),
                    'kernels',
                    'push',
                    '-p',
                    str(staging_dir),
                    '--accelerator',
                    args.accelerator,
                ]
                result = subprocess.run(
                    push_cmd,
                    check=False,
                    env=command_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                print(result.stdout, end='', flush=True)
                if result.returncode != 0 or 'Kernel push error:' in result.stdout:
                    raise RuntimeError(
                        f'kaggle kernels push failed with exit code {result.returncode}: '
                        f'{result.stdout.strip()}'
                    )
                actual_kernel_id = parse_kernel_id(result.stdout, kernel_id)
                status_result = subprocess.run(
                    [*kaggle_command(), 'kernels', 'status', actual_kernel_id],
                    check=False,
                    env=command_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                print(status_result.stdout, end='', flush=True)
                if status_result.returncode != 0:
                    raise RuntimeError(
                        f'kaggle kernels status failed with exit code {status_result.returncode}: '
                        f'{status_result.stdout.strip()}'
                    )
                report['submitted'].append(
                    {
                        'grid_index': grid_index,
                        'owner': owner,
                        'run_name': job['run_name'],
                        'accelerator': args.accelerator,
                        'gmm_var_prior_type': job.get('gmm_var_prior_type'),
                        'gmm_var_prior_strength': job.get('gmm_var_prior_strength'),
                        'gmm_var_prior_target_var': job.get('gmm_var_prior_target_var'),
                        'gmm_standardize_data': job.get('gmm_standardize_data'),
                        'kernel_id': actual_kernel_id,
                        'kernel_status': parse_kernel_status(status_result.stdout),
                        'url': f'https://www.kaggle.com/code/{actual_kernel_id}',
                    }
                )
        except Exception as exc:
            print(f'FAILED {kernel_id}: {exc}', flush=True)
            report['failed'].append(
                {
                    'grid_index': grid_index,
                    'owner': owner,
                    'run_name': job['run_name'],
                    'accelerator': args.accelerator,
                    'gmm_var_prior_type': job.get('gmm_var_prior_type'),
                    'gmm_var_prior_strength': job.get('gmm_var_prior_strength'),
                    'gmm_var_prior_target_var': job.get('gmm_var_prior_target_var'),
                    'gmm_standardize_data': job.get('gmm_standardize_data'),
                    'kernel_id': kernel_id,
                    'error': str(exc),
                }
            )
            if args.stop_on_error:
                write_report(report_path, report)
                raise
        finally:
            if not args.keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)
            write_report(report_path, report)


if __name__ == '__main__':
    main()
