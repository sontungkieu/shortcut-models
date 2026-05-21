from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IGNORE_CONFIG_KEYS = {
    'gmm_em_iters',
    'run_name',
    'run_name_suffix',
    'ablation_tag',
}


def num(value: Any) -> float | None:
    if value is None or value == '':
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def int_or_none(value: Any) -> int | None:
    value = num(value)
    return None if value is None else int(value)


def fmt(value: Any, digits: int = 4) -> str:
    value = num(value)
    if value is None:
        return ''
    return f'{value:.{digits}f}'


def load_ok_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = []
    for row in payload.get('jobs', []):
        if row.get('parse_status') == 'ok':
            rows.append(row)
    return rows


def config_key(row: dict[str, Any]) -> str:
    config = row.get('display_config') or row.get('config') or {}
    if not isinstance(config, dict):
        config = {}
    if not config:
        config = {
            key: row.get(key)
            for key in (
                'dataset_name',
                'num_modes',
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
                'gmm_em_restarts',
                'gmm_em_chunk_size',
            )
            if row.get(key) is not None
        }
    normalized = {
        key: value
        for key, value in config.items()
        if key not in IGNORE_CONFIG_KEYS and value is not None
    }
    if 'gmm_num_modes' not in normalized and row.get('num_modes') is not None:
        normalized['gmm_num_modes'] = row.get('num_modes')
    return json.dumps(normalized, sort_keys=True, separators=(',', ':'))


def row_metric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = num(row.get(key))
        if value is not None:
            return value
    return None


def safe_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return before - after


def safe_rel(delta: float | None, denom: float | None, eps: float) -> float | None:
    if delta is None or denom is None:
        return None
    return delta / max(abs(denom), eps)


def compare_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    source_grid_indexes: set[int],
    min_valid_nll_improve_frac: float,
    max_pi_entropy_drop: float,
    max_count_ratio_rel_increase: float,
    max_overlap_increase: float,
    eps: float,
) -> list[dict[str, Any]]:
    baseline_by_grid = {
        row.get('grid_index'): row
        for row in baseline_rows
        if isinstance(row.get('grid_index'), int)
    }
    baseline_by_key = {config_key(row): row for row in baseline_rows}
    comparisons = []
    for cand in candidate_rows:
        grid_index = cand.get('grid_index')
        base = baseline_by_grid.get(grid_index) if isinstance(grid_index, int) else None
        if base is None:
            base = baseline_by_key.get(config_key(cand))
        if base is None:
            comparisons.append(
                {
                    'grid_index': grid_index,
                    'run_name': cand.get('run_name', ''),
                    'compare_status': 'missing_baseline',
                    'candidate': cand,
                }
            )
            continue

        base_valid = row_metric(base, 'latent_valid_nll', 'valid_nll')
        cand_valid = row_metric(cand, 'latent_valid_nll', 'valid_nll')
        valid_delta = safe_delta(base_valid, cand_valid)
        valid_delta_frac = safe_rel(valid_delta, base_valid, eps)
        pi_drop = safe_delta(
            row_metric(base, 'pi_entropy_normalized'),
            row_metric(cand, 'pi_entropy_normalized'),
        )
        base_count = row_metric(base, 'valid_count_ratio')
        cand_count = row_metric(cand, 'valid_count_ratio')
        count_ratio_rel_increase = (
            (cand_count - base_count) / max(abs(base_count), eps)
            if base_count is not None and cand_count is not None
            else None
        )
        overlap_increase = (
            row_metric(cand, 'latent_overlap_proxy_max', 'overlap_proxy_max')
            - row_metric(base, 'latent_overlap_proxy_max', 'overlap_proxy_max')
            if row_metric(cand, 'latent_overlap_proxy_max', 'overlap_proxy_max') is not None
            and row_metric(base, 'latent_overlap_proxy_max', 'overlap_proxy_max') is not None
            else None
        )
        train_dead = int_or_none(cand.get('train_dead_components'))
        valid_dead = int_or_none(cand.get('valid_dead_components'))
        no_dead = train_dead == 0 and valid_dead == 0
        source_candidate = isinstance(grid_index, int) and grid_index in source_grid_indexes
        source_rerun_recommended = bool(
            source_candidate
            and valid_delta_frac is not None
            and valid_delta_frac >= min_valid_nll_improve_frac
            and no_dead
            and (pi_drop is None or pi_drop <= max_pi_entropy_drop)
            and (
                count_ratio_rel_increase is None
                or count_ratio_rel_increase <= max_count_ratio_rel_increase
            )
            and (overlap_increase is None or overlap_increase <= max_overlap_increase)
        )
        profile_ok = bool(
            no_dead
            and (row_metric(cand, 'pi_entropy_normalized') or 0.0) >= 0.90
            and (
                cand_count is None
                or base_count is None
                or cand_count <= base_count * (1.0 + max_count_ratio_rel_increase)
            )
            and (overlap_increase is None or overlap_increase <= max_overlap_increase)
        )
        still_improving_after25 = bool(
            (row_metric(cand, 'nll_delta_25_to_final') or 0.0) > 0.0
            and (row_metric(cand, 'nll_delta_last10_rel') or 0.0) > 1e-5
        )
        comparisons.append(
            {
                'compare_status': 'ok',
                'grid_index': grid_index,
                'run_name': cand.get('run_name', ''),
                'baseline_run_name': base.get('run_name', ''),
                'num_modes': cand.get('num_modes') or cand.get('gmm_num_modes'),
                'coverage_name': cand.get('coverage_name'),
                'gmm_pi_prior_type': cand.get('gmm_pi_prior_type'),
                'gmm_pi_prior_strength': cand.get('gmm_pi_prior_strength'),
                'gmm_var_prior_type': cand.get('gmm_var_prior_type'),
                'gmm_var_prior_strength': cand.get('gmm_var_prior_strength'),
                'gmm_var_prior_target_var': cand.get('gmm_var_prior_target_var'),
                'baseline_valid_nll': base_valid,
                'candidate_valid_nll': cand_valid,
                'valid_nll_delta': valid_delta,
                'valid_nll_delta_frac': valid_delta_frac,
                'baseline_pi_entropy_normalized': row_metric(base, 'pi_entropy_normalized'),
                'candidate_pi_entropy_normalized': row_metric(cand, 'pi_entropy_normalized'),
                'pi_entropy_drop': pi_drop,
                'baseline_valid_count_ratio': base_count,
                'candidate_valid_count_ratio': cand_count,
                'valid_count_ratio_rel_increase': count_ratio_rel_increase,
                'baseline_overlap_proxy_max': row_metric(base, 'latent_overlap_proxy_max', 'overlap_proxy_max'),
                'candidate_overlap_proxy_max': row_metric(cand, 'latent_overlap_proxy_max', 'overlap_proxy_max'),
                'overlap_proxy_max_increase': overlap_increase,
                'candidate_train_dead_components': train_dead,
                'candidate_valid_dead_components': valid_dead,
                'candidate_train_valid_nll_gap': row_metric(cand, 'train_valid_nll_gap'),
                'candidate_nll_delta_last10': row_metric(cand, 'nll_delta_last10'),
                'candidate_nll_delta_last10_rel': row_metric(cand, 'nll_delta_last10_rel'),
                'candidate_nll_delta_25_to_final': row_metric(cand, 'nll_delta_25_to_final'),
                'candidate_nll_delta_50_to_final': row_metric(cand, 'nll_delta_50_to_final'),
                'candidate_final_minus_best_train_nll': row_metric(cand, 'final_minus_best_train_nll'),
                'candidate_var_floor_hit_rate': row_metric(cand, 'latent_var_floor_hit_rate', 'var_floor_hit_rate'),
                'source_candidate': source_candidate,
                'profile_ok': profile_ok,
                'still_improving_after25': still_improving_after25,
                'source_rerun_recommended': source_rerun_recommended,
            }
        )
    return comparisons


def sort_by_metric(rows: list[dict[str, Any]], key: str, reverse: bool) -> list[dict[str, Any]]:
    present = [row for row in rows if row.get(key) is not None]
    missing = [row for row in rows if row.get(key) is None]
    return sorted(present, key=lambda row: row[key], reverse=reverse) + missing


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    md_path = path.with_suffix('.md')
    ok_rows = [row for row in payload['comparisons'] if row.get('compare_status') == 'ok']
    recommended = [row for row in ok_rows if row.get('source_rerun_recommended')]
    top_improved = sort_by_metric(ok_rows, 'valid_nll_delta_frac', True)[:20]
    worst = sort_by_metric(ok_rows, 'valid_nll_delta_frac', False)[:20]
    still_improving = [row for row in ok_rows if row.get('still_improving_after25')]
    lines = [
        '# GMM EM25 vs EM100',
        '',
        f"- Generated: {payload['generated_at_utc']}",
        f"- Joined rows: {payload['summary']['joined']}",
        f"- Missing baseline: {payload['summary']['missing_baseline']}",
        f"- Source rerun recommendations: {payload['summary']['source_rerun_recommended']}",
        f"- Profile-ok EM100 rows: {payload['summary']['profile_ok']}",
        f"- Still improving after iter 25: {payload['summary']['still_improving_after25']}",
        '',
        '## Decision',
        '',
        payload['decision'],
        '',
        '## Recommended Source Reruns',
        '',
        '| grid | run | K | valid nll delta % | pi entropy drop | count ratio rel inc | overlap inc | last10 rel |',
        '|---:|---|---:|---:|---:|---:|---:|---:|',
    ]
    for row in recommended:
        lines.append(
            f"| {row.get('grid_index', '')} | {row.get('run_name', '')} | {row.get('num_modes', '')} | "
            f"{fmt((row.get('valid_nll_delta_frac') or 0.0) * 100.0, 3)} | "
            f"{fmt(row.get('pi_entropy_drop'), 4)} | "
            f"{fmt(row.get('valid_count_ratio_rel_increase'), 4)} | "
            f"{fmt(row.get('overlap_proxy_max_increase'), 4)} | "
            f"{fmt(row.get('candidate_nll_delta_last10_rel'), 6)} |"
        )
    lines.extend([
        '',
        '## Top Valid NLL Improvements',
        '',
        '| grid | run | K | EM25 valid | EM100 valid | delta % | pi entropy | dead | count ratio | overlap |',
        '|---:|---|---:|---:|---:|---:|---:|---|---:|---:|',
    ])
    for row in top_improved:
        lines.append(
            f"| {row.get('grid_index', '')} | {row.get('run_name', '')} | {row.get('num_modes', '')} | "
            f"{fmt(row.get('baseline_valid_nll'), 3)} | {fmt(row.get('candidate_valid_nll'), 3)} | "
            f"{fmt((row.get('valid_nll_delta_frac') or 0.0) * 100.0, 3)} | "
            f"{fmt(row.get('candidate_pi_entropy_normalized'), 4)} | "
            f"{row.get('candidate_train_dead_components', '')}/{row.get('candidate_valid_dead_components', '')} | "
            f"{fmt(row.get('candidate_valid_count_ratio'), 4)} | "
            f"{fmt(row.get('candidate_overlap_proxy_max'), 4)} |"
        )
    lines.extend([
        '',
        '## Worst Valid NLL Changes',
        '',
        '| grid | run | K | EM25 valid | EM100 valid | delta % | pi entropy | dead | count ratio | overlap |',
        '|---:|---|---:|---:|---:|---:|---:|---|---:|---:|',
    ])
    for row in worst:
        lines.append(
            f"| {row.get('grid_index', '')} | {row.get('run_name', '')} | {row.get('num_modes', '')} | "
            f"{fmt(row.get('baseline_valid_nll'), 3)} | {fmt(row.get('candidate_valid_nll'), 3)} | "
            f"{fmt((row.get('valid_nll_delta_frac') or 0.0) * 100.0, 3)} | "
            f"{fmt(row.get('candidate_pi_entropy_normalized'), 4)} | "
            f"{row.get('candidate_train_dead_components', '')}/{row.get('candidate_valid_dead_components', '')} | "
            f"{fmt(row.get('candidate_valid_count_ratio'), 4)} | "
            f"{fmt(row.get('candidate_overlap_proxy_max'), 4)} |"
        )
    lines.extend([
        '',
        '## Convergence Flags',
        '',
        '| grid | run | delta 25->final | delta 50->final | last10 rel | train-valid gap | final-best |',
        '|---:|---|---:|---:|---:|---:|---:|',
    ])
    for row in still_improving[:30]:
        lines.append(
            f"| {row.get('grid_index', '')} | {row.get('run_name', '')} | "
            f"{fmt(row.get('candidate_nll_delta_25_to_final'), 6)} | "
            f"{fmt(row.get('candidate_nll_delta_50_to_final'), 6)} | "
            f"{fmt(row.get('candidate_nll_delta_last10_rel'), 8)} | "
            f"{fmt(row.get('candidate_train_valid_nll_gap'), 6)} | "
            f"{fmt(row.get('candidate_final_minus_best_train_nll'), 8)} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare GMM EM=25 and EM=100 ablation result reports.')
    parser.add_argument('--baseline-json', required=True)
    parser.add_argument('--candidate-json', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument(
        '--source-grid-indexes',
        default='108,126,136,145,146,154,162',
        help='Comma-separated grid indexes previously used as FM/TIDE sources.',
    )
    parser.add_argument('--min-valid-nll-improve-frac', type=float, default=0.005)
    parser.add_argument('--max-pi-entropy-drop', type=float, default=0.02)
    parser.add_argument('--max-count-ratio-rel-increase', type=float, default=0.20)
    parser.add_argument('--max-overlap-increase', type=float, default=0.02)
    parser.add_argument('--eps', type=float, default=1e-12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_grid_indexes = {
        int(item)
        for item in args.source_grid_indexes.split(',')
        if item.strip()
    }
    comparisons = compare_rows(
        load_ok_rows(Path(args.baseline_json)),
        load_ok_rows(Path(args.candidate_json)),
        source_grid_indexes=source_grid_indexes,
        min_valid_nll_improve_frac=args.min_valid_nll_improve_frac,
        max_pi_entropy_drop=args.max_pi_entropy_drop,
        max_count_ratio_rel_increase=args.max_count_ratio_rel_increase,
        max_overlap_increase=args.max_overlap_increase,
        eps=args.eps,
    )
    ok_rows = [row for row in comparisons if row.get('compare_status') == 'ok']
    recommended = [row for row in ok_rows if row.get('source_rerun_recommended')]
    decision = (
        'Recommend rerunning FM for the listed source configs.'
        if recommended
        else 'No automatic FM rerun recommendation from the configured EM100 criteria.'
    )
    payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_json': args.baseline_json,
        'candidate_json': args.candidate_json,
        'criteria': {
            'source_grid_indexes': sorted(source_grid_indexes),
            'min_valid_nll_improve_frac': args.min_valid_nll_improve_frac,
            'max_pi_entropy_drop': args.max_pi_entropy_drop,
            'max_count_ratio_rel_increase': args.max_count_ratio_rel_increase,
            'max_overlap_increase': args.max_overlap_increase,
        },
        'summary': {
            'candidate_rows': len(comparisons),
            'joined': len(ok_rows),
            'missing_baseline': sum(1 for row in comparisons if row.get('compare_status') != 'ok'),
            'source_rerun_recommended': len(recommended),
            'profile_ok': sum(1 for row in ok_rows if row.get('profile_ok')),
            'still_improving_after25': sum(1 for row in ok_rows if row.get('still_improving_after25')),
        },
        'decision': decision,
        'comparisons': comparisons,
    }
    write_report(Path(args.output_json), payload)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {Path(args.output_json).with_suffix('.md')}")


if __name__ == '__main__':
    main()
