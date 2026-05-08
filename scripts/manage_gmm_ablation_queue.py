from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from push_gmm_ablation_jobs import (
    DEFAULT_ACCOUNTS_FILE,
    kaggle_command,
    load_kaggle_accounts,
    parse_kernel_id,
    parse_kernel_status,
    selected_owners,
)
from stage_gmm_ablation_jobs import (
    load_env_file,
    load_grid,
    normalize_accelerator,
    stage_batch_job,
    stage_job,
)


PENDING = 'pending'
RUNNING = 'running'
COMPLETE = 'complete'
FAILED = 'failed'
RETIRED = 'retired'

KEY_FIELDS = (
    'dataset_name',
    'gmm_num_modes',
    'gmm_min_var',
    'gmm_min_var_data_frac',
    'gmm_min_std',
    'gmm_min_std_data_frac',
    'coverage_name',
    'gmm_pi_prior_type',
    'gmm_pi_prior_strength',
    'gmm_pi_kl_steps',
    'gmm_pi_kl_lr',
    'gmm_var_prior_type',
    'gmm_var_prior_strength',
    'gmm_var_prior_target_var',
    'gmm_standardize_data',
    'gmm_fit_samples',
    'gmm_valid_samples',
    'gmm_em_iters',
    'gmm_em_restarts',
    'gmm_em_chunk_size',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_status(value: str) -> str:
    return value.strip().strip('"').replace('KernelWorkerStatus.', '')


def kaggle_to_queue_status(value: str) -> str:
    status = normalize_status(value).upper()
    if status == 'COMPLETE':
        return COMPLETE
    if status in {'QUEUED', 'RUNNING', 'PENDING', 'PREPARING', 'STARTING'}:
        return RUNNING
    if status in {'CANCELED', 'CANCELLED', 'ERROR', 'FAILED'}:
        return FAILED
    return RUNNING if status else RUNNING


def job_config_key(config: dict[str, Any]) -> str:
    payload = {key: config.get(key) for key in KEY_FIELDS if key in config}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha1(encoded).hexdigest()[:16]


def public_config(job: dict[str, Any]) -> dict[str, Any]:
    keys = [
        'run_name',
        'dataset_name',
        'gmm_num_modes',
        'gmm_min_var',
        'gmm_min_var_data_frac',
        'gmm_min_std',
        'gmm_min_std_data_frac',
        'coverage_name',
        'gmm_pi_prior_type',
        'gmm_pi_prior_strength',
        'gmm_pi_kl_steps',
        'gmm_pi_kl_lr',
        'gmm_var_prior_type',
        'gmm_var_prior_strength',
        'gmm_var_prior_target_var',
        'gmm_standardize_data',
        'gmm_fit_samples',
        'gmm_valid_samples',
        'gmm_em_iters',
        'gmm_em_restarts',
        'gmm_em_chunk_size',
        'jax_runtime',
    ]
    return {key: job.get(key) for key in keys if key in job}


def load_queue(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'jobs': [], 'retired_jobs': [], 'history': []}
    return json.loads(path.read_text(encoding='utf-8'))


def build_queue(grid_path: Path, old_queue: dict[str, Any]) -> dict[str, Any]:
    _, grid_jobs = load_grid(grid_path)
    old_by_key = {row.get('job_key'): row for row in old_queue.get('jobs', []) if row.get('job_key')}
    old_by_run = {row.get('run_name'): row for row in old_queue.get('jobs', []) if row.get('run_name')}
    now = utc_now()

    jobs = []
    active_keys = set()
    active_run_names = set()
    for grid_index, config in enumerate(grid_jobs):
        job_key = job_config_key(config)
        active_keys.add(job_key)
        active_run_names.add(config.get('run_name'))
        previous = old_by_key.get(job_key) or old_by_run.get(config.get('run_name'))
        row = {
            'job_key': job_key,
            'grid_index': grid_index,
            'run_name': config['run_name'],
            'status': PENDING,
            'config': dict(config),
            'display_config': public_config(config),
            'created_at': now,
            'updated_at': now,
            'attempts': [],
        }
        if previous:
            for field in (
                'status',
                'owner',
                'kernel_id',
                'url',
                'kaggle_status',
                'submitted_at',
                'last_checked_at',
                'completed_at',
                'failed_at',
                'last_error',
                'attempts',
            ):
                if field in previous:
                    row[field] = previous[field]
            row['created_at'] = previous.get('created_at', row['created_at'])
            row['updated_at'] = previous.get('updated_at', row['updated_at'])
        jobs.append(row)

    retired = [
        old for old in old_queue.get('retired_jobs', [])
        if old.get('job_key') not in active_keys and old.get('run_name') not in active_run_names
    ]
    for old in old_queue.get('jobs', []):
        if old.get('job_key') not in active_keys and old.get('run_name') not in active_run_names:
            old = dict(old)
            old['status'] = RETIRED
            old['retired_at'] = now
            retired.append(old)

    return {
        'grid_config': str(grid_path),
        'generated_at': now,
        'jobs': jobs,
        'retired_jobs': retired,
        'history': old_queue.get('history', []),
    }


def merge_report_row(queue: dict[str, Any], row: dict[str, Any], source: str) -> bool:
    grid_index = row.get('grid_index')
    run_name = row.get('run_name')
    target = None
    for job in queue['jobs']:
        if run_name and job.get('run_name') == run_name:
            target = job
            break
    if run_name and target is None:
        return False
    if target is None:
        for job in queue['jobs']:
            if isinstance(grid_index, int) and job.get('grid_index') == grid_index:
                target = job
                break
    if target is None and isinstance(grid_index, int):
        _, grid_jobs = load_grid(Path(queue.get('grid_config', 'configs/gmm_ablation_grid.json')))
        if 0 <= grid_index < len(grid_jobs):
            expected_run = grid_jobs[grid_index].get('run_name')
            for job in queue['jobs']:
                if job.get('run_name') == expected_run:
                    target = job
                    break
    if not target:
        return False

    kernel_id = row.get('kernel_id') or row.get('ref')
    owner = row.get('owner') or (kernel_id.split('/', 1)[0] if kernel_id else None)
    kaggle_status = (
        row.get('latest_status')
        or row.get('status')
        or row.get('kernel_status')
        or row.get('submitted_status')
        or ''
    )
    queue_status = kaggle_to_queue_status(kaggle_status) if kaggle_status else RUNNING
    if row.get('parse_status') == 'ok':
        queue_status = COMPLETE
    if owner:
        target['owner'] = owner
    if kernel_id:
        target['kernel_id'] = kernel_id
        target['url'] = row.get('url') or f'https://www.kaggle.com/code/{kernel_id}'
    if kaggle_status:
        target['kaggle_status'] = normalize_status(str(kaggle_status))
    target['status'] = queue_status
    target['updated_at'] = utc_now()
    target['seed_source'] = source
    if queue_status == COMPLETE:
        target.setdefault('completed_at', target['updated_at'])
    elif queue_status == RUNNING:
        target.setdefault('submitted_at', target['updated_at'])
    return True


def seed_queue_from_reports(queue: dict[str, Any], paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        rows: list[dict[str, Any]] = []
        rows.extend(payload.get('jobs', []))
        rows.extend(payload.get('submitted_tpu_jobs', []))
        rows.extend(payload.get('submitted', []))
        for row in rows:
            merge_report_row(queue, row, str(path))


def write_queue(path: Path, queue: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter(row.get('status', PENDING) for row in queue.get('jobs', []))
    queue['summary'] = {
        'total': len(queue.get('jobs', [])),
        'pending': counts.get(PENDING, 0),
        'running': counts.get(RUNNING, 0),
        'complete': counts.get(COMPLETE, 0),
        'failed': counts.get(FAILED, 0),
        'retired': len(queue.get('retired_jobs', [])),
    }
    queue['updated_at'] = utc_now()
    path.write_text(json.dumps(queue, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    write_queue_md(path.with_suffix('.md'), queue)


def write_queue_md(path: Path, queue: dict[str, Any]) -> None:
    summary = queue.get('summary', {})
    lines = [
        '# GMM Ablation Queue',
        '',
        f"- Total: {summary.get('total', 0)}",
        f"- Pending: {summary.get('pending', 0)}",
        f"- Running: {summary.get('running', 0)}",
        f"- Complete: {summary.get('complete', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Retired: {summary.get('retired', 0)}",
        '',
        '## Jobs',
        '',
        '| grid | status | owner | run | batch | kaggle | kernel |',
        '|---:|---|---|---|---:|---|---|',
    ]
    for row in queue.get('jobs', []):
        lines.append(
            f"| {row.get('grid_index', '')} | {row.get('status', '')} | "
            f"{row.get('owner', '')} | {row.get('run_name', '')} | "
            f"{row.get('batch_size', '')} | "
            f"{row.get('kaggle_status', '')} | `{row.get('kernel_id', '')}` |"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def credential_file(config_dir: Path, credential: dict[str, str]) -> None:
    config_path = config_dir / 'kaggle.json'
    config_path.write_text(json.dumps(credential) + '\n', encoding='utf-8')
    config_path.chmod(0o600)


def kaggle_run(args: list[str], credential: dict[str, str]) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix=f"kaggle-config-{credential['username']}-") as config_dir:
        credential_file(Path(config_dir), credential)
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


def sync_running_jobs(queue: dict[str, Any], accounts: dict[str, dict[str, str]]) -> None:
    status_cache: dict[str, dict[str, Any]] = {}
    for row in queue.get('jobs', []):
        if row.get('status') != RUNNING or not row.get('kernel_id'):
            continue
        owner = row.get('owner') or row['kernel_id'].split('/', 1)[0]
        kernel_id = row['kernel_id']
        if kernel_id not in status_cache:
            credential = accounts.get(owner)
            checked_at = utc_now()
            if not credential:
                status_cache[kernel_id] = {
                    'ok': False,
                    'checked_at': checked_at,
                    'error': f'No Kaggle credential for {owner}',
                }
            else:
                result = kaggle_run(['kernels', 'status', kernel_id], credential)
                if result.returncode != 0:
                    status_cache[kernel_id] = {
                        'ok': False,
                        'checked_at': checked_at,
                        'error': result.stdout.strip(),
                    }
                else:
                    status_cache[kernel_id] = {
                        'ok': True,
                        'checked_at': checked_at,
                        'kaggle_status': normalize_status(parse_kernel_status(result.stdout)),
                    }
        cached = status_cache[kernel_id]
        row['last_checked_at'] = cached['checked_at']
        if not cached['ok']:
            row['last_error'] = cached['error']
            row['updated_at'] = cached['checked_at']
            continue
        kaggle_status = cached['kaggle_status']
        row['kaggle_status'] = kaggle_status
        row['status'] = kaggle_to_queue_status(kaggle_status)
        row['updated_at'] = cached['checked_at']
        if row['status'] == COMPLETE:
            row['completed_at'] = cached['checked_at']
        elif row['status'] == FAILED:
            row['failed_at'] = cached['checked_at']


def owner_running_counts(queue: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in queue.get('jobs', []):
        if row.get('status') == RUNNING and row.get('owner'):
            owner = row['owner']
            kernel_marker = row.get('kernel_id') or row.get('job_key') or row.get('run_name')
            if kernel_marker in seen[owner]:
                continue
            seen[owner].add(kernel_marker)
            counts[owner] += 1
    return counts


def next_owner(
    owners: list[str],
    counts: dict[str, int],
    max_submit_per_owner: int,
    cursor: int,
) -> tuple[str | None, int]:
    for offset in range(len(owners)):
        idx = (cursor + offset) % len(owners)
        owner = owners[idx]
        if max_submit_per_owner <= 0 or counts.get(owner, 0) < max_submit_per_owner:
            return owner, idx + 1
    return None, cursor


def push_pending_jobs(
    *,
    queue: dict[str, Any],
    accounts: dict[str, dict[str, str]],
    owners: list[str],
    env_file: Path,
    notebook: Path,
    batch_notebook: Path,
    staging_root: Path,
    accelerator: str,
    limit: int,
    batch_size: int,
    max_submit_per_owner: int,
    keep_staging: bool,
    dry_run: bool,
) -> int:
    env_values = load_env_file(env_file)
    wandb_api_key = env_values.get('WANDB_API_KEY', '')
    running_counts = owner_running_counts(queue)
    cursor = 0
    submitted = 0
    pending_jobs = [row for row in queue.get('jobs', []) if row.get('status') == PENDING]
    batch_size = max(1, int(batch_size))

    index = 0
    while index < len(pending_jobs):
        if limit and submitted >= limit:
            break
        owner, cursor = next_owner(owners, running_counts, max_submit_per_owner, cursor)
        if not owner:
            break
        credential = accounts[owner]
        remaining_limit = len(pending_jobs) - index
        if limit:
            remaining_limit = min(remaining_limit, limit - submitted)
        take = min(batch_size, remaining_limit)
        batch_rows = pending_jobs[index:index + take]
        index += take
        configs = [dict(row['config']) for row in batch_rows]
        if len(batch_rows) == 1:
            staging_dir, staged_kernel_id = stage_job(
                owner=owner,
                notebook_template=notebook,
                staging_root=staging_root,
                config=configs[0],
                wandb_api_key=wandb_api_key,
                accelerator=accelerator,
            )
        else:
            staging_dir, staged_kernel_id = stage_batch_job(
                owner=owner,
                notebook_template=batch_notebook,
                staging_root=staging_root,
                configs=configs,
                wandb_api_key=wandb_api_key,
                accelerator=accelerator,
            )
        now = utc_now()
        attempt = {
            'owner': owner,
            'accelerator': accelerator,
            'staged_kernel_id': staged_kernel_id,
            'started_at': now,
            'dry_run': dry_run,
            'batch_size': len(batch_rows),
            'grid_indexes': [row.get('grid_index') for row in batch_rows],
        }
        for member_index, row in enumerate(batch_rows):
            member_attempt = dict(attempt)
            member_attempt['batch_member_index'] = member_index
            row.setdefault('attempts', []).append(member_attempt)
        print(
            f'Staged {staged_kernel_id} for grids '
            f'{[row["grid_index"] for row in batch_rows]}',
            flush=True,
        )

        try:
            if dry_run:
                for member_index, row in enumerate(batch_rows):
                    row['owner'] = owner
                    row['kernel_id'] = staged_kernel_id
                    row['url'] = f'https://www.kaggle.com/code/{staged_kernel_id}'
                    row['status'] = RUNNING
                    row['kaggle_status'] = 'DRY_RUN'
                    row['submitted_at'] = now
                    row['updated_at'] = now
                    row['batch_size'] = len(batch_rows)
                    row['batch_member_index'] = member_index
                running_counts[owner] += 1
                submitted += len(batch_rows)
                continue

            with tempfile.TemporaryDirectory(prefix=f'kaggle-config-{owner}-') as config_dir:
                credential_file(Path(config_dir), credential)
                command_env = os.environ.copy()
                command_env['KAGGLE_CONFIG_DIR'] = config_dir
                result = subprocess.run(
                    [
                        *kaggle_command(),
                        'kernels',
                        'push',
                        '-p',
                        str(staging_dir),
                        '--accelerator',
                        accelerator,
                    ],
                    check=False,
                    env=command_env,
                    stderr=subprocess.STDOUT,
                    stdout=subprocess.PIPE,
                    text=True,
            )
            finished_at = utc_now()
            push_output_tail = result.stdout.strip()[-1000:]
            for row in batch_rows:
                row['attempts'][-1]['finished_at'] = finished_at
                row['attempts'][-1]['push_output_tail'] = push_output_tail
            if result.returncode != 0 or 'Kernel push error:' in result.stdout:
                for row in batch_rows:
                    row['last_error'] = result.stdout.strip()
                    row['updated_at'] = finished_at
                print(
                    f'Push failed for grids {[row["grid_index"] for row in batch_rows]}: '
                    f'{result.stdout.strip()[-300:]}',
                    flush=True,
                )
                continue

            kernel_id = parse_kernel_id(result.stdout, staged_kernel_id)
            for member_index, row in enumerate(batch_rows):
                row['owner'] = owner
                row['kernel_id'] = kernel_id
                row['url'] = f'https://www.kaggle.com/code/{kernel_id}'
                row['status'] = RUNNING
                row['kaggle_status'] = 'SUBMITTED'
                row['submitted_at'] = finished_at
                row['updated_at'] = finished_at
                row['batch_size'] = len(batch_rows)
                row['batch_member_index'] = member_index
            running_counts[owner] += 1
            submitted += len(batch_rows)
            print(
                f'Submitted grids {[row["grid_index"] for row in batch_rows]}: {kernel_id}',
                flush=True,
            )
        finally:
            if not keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)
    return submitted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Render, sync, and submit the GMM ablation queue.')
    parser.add_argument('--queue-path', default='reports/gmm_ablation_queue.json')
    parser.add_argument('--grid-config', default='configs/gmm_ablation_grid.json')
    parser.add_argument('--accounts-file', default=str(DEFAULT_ACCOUNTS_FILE))
    parser.add_argument('--env-file', default='.secrets/.env')
    parser.add_argument('--notebook', default='shortcut-model-gmm-ablation.ipynb')
    parser.add_argument('--batch-notebook', default='shortcut-model-gmm-ablation-batch.ipynb')
    parser.add_argument('--staging-root', default='kaggle_staging/gmm_ablation_queue')
    parser.add_argument('--owners', default='all')
    parser.add_argument('--exclude-owners', default='')
    parser.add_argument('--accelerator', default='TpuV5E8')
    parser.add_argument('--seed-report', action='append', default=[])
    parser.add_argument('--reset', action='store_true', help='Ignore an existing queue file and rebuild from the grid.')
    parser.add_argument('--sync-status', action='store_true')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-submit-per-owner', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--keep-staging', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.accelerator = normalize_accelerator(args.accelerator)
    queue_path = Path(args.queue_path)
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    owners = selected_owners(args.owners, sorted(accounts))
    excluded = {item.strip() for item in args.exclude_owners.split(',') if item.strip()}
    owners = [owner for owner in owners if owner not in excluded]
    if not owners:
        raise SystemExit('No owners selected after applying --exclude-owners.')

    old_queue = {'jobs': [], 'retired_jobs': [], 'history': []} if args.reset else load_queue(queue_path)
    queue = build_queue(Path(args.grid_config), old_queue)
    seed_reports = [Path(item) for item in args.seed_report]
    if seed_reports:
        seed_queue_from_reports(queue, seed_reports)
    if args.sync_status:
        sync_running_jobs(queue, accounts)
    submitted = 0
    if args.push:
        submitted = push_pending_jobs(
            queue=queue,
            accounts=accounts,
            owners=owners,
            env_file=Path(args.env_file),
            notebook=Path(args.notebook),
            batch_notebook=Path(args.batch_notebook),
            staging_root=Path(args.staging_root),
            accelerator=args.accelerator,
            limit=args.limit,
            batch_size=args.batch_size,
            max_submit_per_owner=args.max_submit_per_owner,
            keep_staging=args.keep_staging,
            dry_run=args.dry_run,
        )
    queue.setdefault('history', []).append(
        {
            'at': utc_now(),
            'grid_config': args.grid_config,
            'owners': owners,
            'excluded_owners': sorted(excluded),
            'accelerator': args.accelerator,
            'seed_reports': [str(path) for path in seed_reports],
            'reset': args.reset,
            'sync_status': args.sync_status,
            'push': args.push,
            'limit': args.limit,
            'batch_size': args.batch_size,
            'max_submit_per_owner': args.max_submit_per_owner,
            'dry_run': args.dry_run,
            'submitted': submitted,
        }
    )
    write_queue(queue_path, queue)
    print(f'Wrote {queue_path}')
    print(f'Wrote {queue_path.with_suffix(".md")}')
    print(json.dumps(queue['summary'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
