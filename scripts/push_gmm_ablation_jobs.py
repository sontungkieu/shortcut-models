from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from stage_gmm_ablation_jobs import load_env_file, load_grid, stage_job


DEFAULT_ACCOUNTS_FILE = Path('.secrets/all-kaggle.json')


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Stage and push GMM ablation Kaggle GPU notebooks.')
    parser.add_argument('--owners', default='codemaivanngu', help="Comma-separated owners or 'all'.")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    owners = selected_owners(args.owners, sorted(accounts))
    _, jobs = load_grid(Path(args.grid_config))
    jobs = jobs[args.offset:]
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        raise SystemExit('No jobs selected.')

    env_values = load_env_file(Path(args.env_file))
    wandb_api_key = env_values.get('WANDB_API_KEY', '')

    for index, job in enumerate(jobs):
        owner = owners[index % len(owners)]
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
                if result.returncode != 0:
                    raise SystemExit(f'kaggle kernels push failed with exit code {result.returncode}')
                subprocess.run(
                    [*kaggle_command(), 'kernels', 'status', kernel_id],
                    check=True,
                    env=command_env,
                )
        finally:
            if not args.keep_staging:
                shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
