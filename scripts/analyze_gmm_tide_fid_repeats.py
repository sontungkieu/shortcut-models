from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fid_repeat_utils import parse_eval_fid_seeds, sample_mean_std


PRIMARY_METRIC = "fid/timesteps/128"
DEFAULT_PROTOCOL = Path("configs/gmm_tide_fid_repeat_analysis_protocol.json")
DEFAULT_EVAL_SECONDARY_METRICS = (
    "flow/curvature_proxy_mean",
    "flow/straightness_ratio_mean",
    "flow/path_length_mean",
    "flow/endpoint_displacement_mean",
)


def load_jobs(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    defaults = payload.get("defaults", {})
    jobs = {}
    for index, raw_job in enumerate(payload.get("jobs", [])):
        job = {**defaults, **raw_job, "grid_index": index}
        jobs[job["run_name"]] = job
    if not jobs:
        raise ValueError(f"No jobs found in {path}")
    return jobs


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_id",
        "primary_metric",
        "control_family",
        "candidate_family",
        "training_seeds",
        "evaluation_seeds",
        "generations_per_evaluation_seed",
        "expected_checkpoint_step",
        "decision_gate",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"Protocol {path} is missing required fields: {', '.join(missing)}")
    return protocol


def validate_protocol_against_jobs(
    protocol: dict[str, Any],
    jobs: dict[str, dict[str, Any]],
) -> list[str]:
    errors = []
    expected_families = {str(protocol["control_family"]), str(protocol["candidate_family"])}
    actual_families = {str(job["candidate_family"]) for job in jobs.values()}
    if actual_families != expected_families:
        errors.append(f"families={sorted(actual_families)} expected={sorted(expected_families)}")

    expected_training_seeds = sorted(int(seed) for seed in protocol["training_seeds"])
    expected_eval_seeds = [int(seed) for seed in protocol["evaluation_seeds"]]
    expected_generations = int(protocol["generations_per_evaluation_seed"])
    expected_step = int(protocol["expected_checkpoint_step"])
    for family in sorted(expected_families):
        family_seeds = sorted(
            int(job["training_seed"])
            for job in jobs.values()
            if str(job["candidate_family"]) == family
        )
        if family_seeds != expected_training_seeds:
            errors.append(f"{family}.training_seeds={family_seeds} expected={expected_training_seeds}")
    for run_name, job in jobs.items():
        job_eval_seeds = parse_eval_fid_seeds(str(job["eval_fid_seeds"]))
        if job_eval_seeds != expected_eval_seeds:
            errors.append(f"{run_name}.eval_seeds={job_eval_seeds} expected={expected_eval_seeds}")
        if int(job["eval_fid_generations"]) != expected_generations:
            errors.append(
                f"{run_name}.generations={job['eval_fid_generations']} expected={expected_generations}"
            )
        if int(job.get("resume_expected_checkpoint_step", 0)) != expected_step:
            errors.append(
                f"{run_name}.checkpoint_step={job.get('resume_expected_checkpoint_step')} expected={expected_step}"
            )
    return errors


def discover_metric_rows(search_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    files = []
    for path in sorted(search_root.rglob("fid_repeat_metrics.jsonl")):
        files.append(str(path))
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("phase") != "eval_fid_repeat" or PRIMARY_METRIC not in payload:
                continue
            run_name = str(payload.get("run_name") or path.parent.parent.name)
            rows.append(
                {
                    "run_name": run_name,
                    "eval_seed": int(payload["eval_seed"]),
                    "value": float(payload[PRIMARY_METRIC]),
                    "step": int(payload["step"]),
                    "eval_fid_generations": int(payload["eval_fid_generations"]),
                    "secondary_metrics": {
                        metric: float(payload[metric])
                        for metric in DEFAULT_EVAL_SECONDARY_METRICS
                        if isinstance(payload.get(metric), (int, float))
                        and math.isfinite(float(payload[metric]))
                    },
                    "path": str(path),
                    "line": line_number,
                }
            )
    return rows, files


def _critical_t95(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        15: 2.131,
        20: 2.086,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 1:
        return 0.0
    if df < 15:
        return table[max(key for key in table if key <= df)]
    if df < 20:
        return table[15]
    if df < 30:
        return table[20]
    return 1.96


def summarize_values(values: Iterable[float]) -> dict[str, float | int]:
    values = [float(value) for value in values]
    mean, sample_std = sample_mean_std(values)
    standard_error = sample_std / math.sqrt(len(values))
    half_width = _critical_t95(len(values) - 1) * standard_error
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "min": min(values),
        "max": max(values),
    }


def audit_comparability(
    jobs: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    errors = validate_protocol_against_jobs(protocol, jobs)
    warnings = []
    control_family = str(protocol["control_family"])
    candidate_family = str(protocol["candidate_family"])
    expected_eval_seeds = [int(seed) for seed in protocol["evaluation_seeds"]]
    expected_generations = int(protocol["generations_per_evaluation_seed"])
    expected_step = int(protocol["expected_checkpoint_step"])
    allowed_differences = set(protocol.get("allowed_pairwise_config_differences", []))

    pairwise_config_differences = []
    for training_seed in sorted(int(seed) for seed in protocol["training_seeds"]):
        control_jobs = [
            job
            for job in jobs.values()
            if str(job["candidate_family"]) == control_family
            and int(job["training_seed"]) == training_seed
        ]
        candidate_jobs = [
            job
            for job in jobs.values()
            if str(job["candidate_family"]) == candidate_family
            and int(job["training_seed"]) == training_seed
        ]
        if len(control_jobs) != 1 or len(candidate_jobs) != 1:
            errors.append(
                f"training_seed={training_seed} requires one {control_family} and one {candidate_family} job"
            )
            continue
        control_job = control_jobs[0]
        candidate_job = candidate_jobs[0]
        differences = {
            key: {"control": control_job.get(key), "candidate": candidate_job.get(key)}
            for key in sorted(set(control_job) | set(candidate_job))
            if control_job.get(key) != candidate_job.get(key)
        }
        unexpected = sorted(set(differences) - allowed_differences)
        if unexpected:
            errors.append(
                f"training_seed={training_seed} has unexpected config differences: {', '.join(unexpected)}"
            )
        pairwise_config_differences.append(
            {
                "training_seed": training_seed,
                "differences": differences,
                "unexpected_fields": unexpected,
            }
        )

    rows_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["run_name"] in jobs:
            rows_by_run[row["run_name"]].append(row)
    run_checks = []
    for run_name in jobs:
        run_rows = rows_by_run[run_name]
        actual_eval_seeds = sorted(int(row["eval_seed"]) for row in run_rows)
        unique_eval_seeds = sorted(set(actual_eval_seeds))
        actual_steps = sorted(set(int(row["step"]) for row in run_rows))
        actual_generations = sorted(set(int(row["eval_fid_generations"]) for row in run_rows))
        run_errors = []
        if unique_eval_seeds != expected_eval_seeds:
            run_errors.append(f"eval_seeds={unique_eval_seeds} expected={expected_eval_seeds}")
        if len(actual_eval_seeds) != len(unique_eval_seeds):
            run_errors.append("duplicate eval seed rows")
        if actual_steps != [expected_step]:
            run_errors.append(f"steps={actual_steps} expected={[expected_step]}")
        if actual_generations != [expected_generations]:
            run_errors.append(
                f"generations={actual_generations} expected={[expected_generations]}"
            )
        errors.extend(f"{run_name}: {message}" for message in run_errors)
        run_checks.append(
            {
                "run_name": run_name,
                "eval_seeds": unique_eval_seeds,
                "steps": actual_steps,
                "generations": actual_generations,
                "ok": not run_errors,
                "errors": run_errors,
            }
        )

    if not bool(protocol.get("comparability_requirements", {}).get("checkpoint_hash_available", False)):
        warnings.append(
            "Checkpoint/router/GMM hashes were not instrumented; audit verifies source references, config, and loaded step only."
        )
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "pairwise_config_differences": pairwise_config_differences,
        "run_checks": run_checks,
    }


def analyze_rows(
    jobs: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = protocol or {
        "control_family": "C0",
        "candidate_family": "C4",
        "training_seeds": sorted({int(job["training_seed"]) for job in jobs.values()}),
        "evaluation_seeds": sorted(
            {
                seed
                for job in jobs.values()
                for seed in parse_eval_fid_seeds(str(job["eval_fid_seeds"]))
            }
        ),
        "decision_gate": {
            "minimum_absolute_fid_improvement": 0.1,
            "pooled_evaluation_sd_multiplier": 2.0,
            "require_candidate_to_win_every_training_seed": True,
            "require_complete_evaluation_seed_pairs": True,
        },
    }
    control_family = str(protocol["control_family"])
    candidate_family = str(protocol["candidate_family"])
    expected_training_seeds = sorted(int(seed) for seed in protocol["training_seeds"])
    expected_eval_seeds = [int(seed) for seed in protocol["evaluation_seeds"]]
    gate = protocol["decision_gate"]
    by_run_seed = {}
    for row in rows:
        if row["run_name"] not in jobs:
            continue
        by_run_seed[(row["run_name"], row["eval_seed"])] = row

    run_summaries = []
    run_values = {}
    protocol_secondary_metrics = [str(metric) for metric in protocol.get("secondary_diagnostics", [])]
    eval_secondary_metrics = [
        metric for metric in protocol_secondary_metrics if metric in DEFAULT_EVAL_SECONDARY_METRICS
    ]
    run_secondary_values: dict[tuple[str, int, str], dict[int, float]] = {}
    for run_name, job in jobs.items():
        seed_rows = sorted(
            (
                row
                for (row_run, _), row in by_run_seed.items()
                if row_run == run_name
            ),
            key=lambda row: row["eval_seed"],
        )
        values = {row["eval_seed"]: row["value"] for row in seed_rows}
        run_values[(str(job["candidate_family"]), int(job["training_seed"]))] = values
        secondary_values_by_metric = {
            metric: {
                int(row["eval_seed"]): float(row.get("secondary_metrics", {})[metric])
                for row in seed_rows
                if metric in row.get("secondary_metrics", {})
            }
            for metric in eval_secondary_metrics
        }
        for metric, metric_values in secondary_values_by_metric.items():
            run_secondary_values[
                (str(job["candidate_family"]), int(job["training_seed"]), metric)
            ] = metric_values
        summary = summarize_values(values.values()) if values else None
        run_summaries.append(
            {
                "run_name": run_name,
                "candidate_family": str(job["candidate_family"]),
                "training_seed": int(job["training_seed"]),
                "checkpoint_source": str(job["resume_kernel_ref"]),
                "expected_eval_seeds": str(job["eval_fid_seeds"]),
                "expected_generations": int(job["eval_fid_generations"]),
                "values_by_eval_seed": values,
                "summary": summary,
                "secondary_values_by_eval_seed": secondary_values_by_metric,
                "secondary_summaries": {
                    metric: summarize_values(metric_values.values()) if metric_values else None
                    for metric, metric_values in secondary_values_by_metric.items()
                },
            }
        )

    paired_by_training_seed = []
    all_eval_deltas = []
    complete_pairs = True
    for training_seed in expected_training_seeds:
        control = run_values.get((control_family, training_seed), {})
        candidate = run_values.get((candidate_family, training_seed), {})
        common = sorted(set(control) & set(candidate))
        if common != expected_eval_seeds:
            complete_pairs = False
        deltas = [candidate[eval_seed] - control[eval_seed] for eval_seed in common]
        all_eval_deltas.extend(deltas)
        paired_by_training_seed.append(
            {
                "training_seed": training_seed,
                "common_eval_seeds": common,
                "control_mean": summarize_values(control.values())["mean"] if control else None,
                "candidate_mean": summarize_values(candidate.values())["mean"] if candidate else None,
                "c0_mean": summarize_values(control.values())["mean"] if control else None,
                "c4_mean": summarize_values(candidate.values())["mean"] if candidate else None,
                "delta_candidate_minus_control": summarize_values(deltas) if deltas else None,
                "delta_c4_minus_c0": summarize_values(deltas) if deltas else None,
            }
        )

    pooled_numerator = 0.0
    pooled_df = 0
    for run in run_summaries:
        summary = run["summary"]
        if not summary or int(summary["n"]) < 2:
            continue
        df = int(summary["n"]) - 1
        pooled_numerator += df * float(summary["sample_std"]) ** 2
        pooled_df += df
    pooled_eval_sd = math.sqrt(pooled_numerator / pooled_df) if pooled_df else 0.0
    minimum_improvement = float(gate["minimum_absolute_fid_improvement"])
    sd_multiplier = float(gate["pooled_evaluation_sd_multiplier"])
    practical_threshold = max(minimum_improvement, sd_multiplier * pooled_eval_sd)
    paired_summary = summarize_values(all_eval_deltas) if all_eval_deltas else None
    all_training_seeds_favor_candidate = bool(paired_by_training_seed) and all(
        item["delta_candidate_minus_control"] is not None
        and float(item["delta_candidate_minus_control"]["mean"]) < 0
        for item in paired_by_training_seed
    )
    direction_requirement = (
        all_training_seeds_favor_candidate
        if bool(gate.get("require_candidate_to_win_every_training_seed", True))
        else True
    )
    completeness_requirement = (
        complete_pairs
        if bool(gate.get("require_complete_evaluation_seed_pairs", True))
        else True
    )
    measurement_gate_passed = bool(
        paired_summary
        and float(paired_summary["mean"]) <= -practical_threshold
        and direction_requirement
        and completeness_requirement
    )
    seed_delta_means = [
        float(item["delta_candidate_minus_control"]["mean"])
        for item in paired_by_training_seed
        if item["delta_candidate_minus_control"] is not None
    ]
    all_training_seeds_lose = bool(seed_delta_means) and all(delta > 0 for delta in seed_delta_means)
    if not complete_pairs:
        outcome_code = "missing_or_incomparable_artifacts"
    elif measurement_gate_passed:
        outcome_code = "gate_passed"
    elif all_training_seeds_favor_candidate:
        outcome_code = "direction_consistent_but_below_threshold"
    elif all_training_seeds_lose:
        outcome_code = "candidate_loses_every_training_seed"
    else:
        outcome_code = "training_seed_directions_disagree"
    evaluation_noise_exceeds_fixed_floor = sd_multiplier * pooled_eval_sd > minimum_improvement

    secondary_paired_by_training_seed = []
    for training_seed in expected_training_seeds:
        metric_rows = []
        for metric in eval_secondary_metrics:
            control = run_secondary_values.get((control_family, training_seed, metric), {})
            candidate = run_secondary_values.get((candidate_family, training_seed, metric), {})
            common = sorted(set(control) & set(candidate))
            deltas = [candidate[eval_seed] - control[eval_seed] for eval_seed in common]
            metric_rows.append(
                {
                    "metric": metric,
                    "common_eval_seeds": common,
                    "control_mean": summarize_values(control.values())["mean"] if control else None,
                    "candidate_mean": summarize_values(candidate.values())["mean"] if candidate else None,
                    "delta_candidate_minus_control": summarize_values(deltas) if deltas else None,
                }
            )
        secondary_paired_by_training_seed.append(
            {"training_seed": training_seed, "metrics": metric_rows}
        )

    available_secondary_metrics = sorted(
        {
            metric
            for run in run_summaries
            for metric, summary in run["secondary_summaries"].items()
            if summary is not None
        }
    )
    unavailable_secondary_metrics = sorted(
        set(protocol_secondary_metrics) - set(available_secondary_metrics)
    )

    return {
        "primary_metric": PRIMARY_METRIC,
        "run_summaries": run_summaries,
        "paired_by_training_seed": paired_by_training_seed,
        "paired_eval_delta_c4_minus_c0": paired_summary,
        "pooled_within_checkpoint_eval_sd": pooled_eval_sd,
        "practical_improvement_threshold": practical_threshold,
        "minimum_absolute_fid_improvement": minimum_improvement,
        "pooled_evaluation_sd_multiplier": sd_multiplier,
        "complete_evaluation_seed_pairs": complete_pairs,
        "all_training_seeds_favor_candidate": all_training_seeds_favor_candidate,
        "all_training_seeds_favor_c4": all_training_seeds_favor_candidate,
        "all_training_seeds_lose": all_training_seeds_lose,
        "evaluation_noise_exceeds_fixed_floor": evaluation_noise_exceeds_fixed_floor,
        "secondary_diagnostics": {
            "role": "mechanism_diagnostics_only_not_part_of_the_decision_gate",
            "available_metrics": available_secondary_metrics,
            "unavailable_metrics": unavailable_secondary_metrics,
            "paired_by_training_seed": secondary_paired_by_training_seed,
        },
        "outcome_code": outcome_code,
        "outcome_action": protocol.get("outcome_actions", {}).get(outcome_code, ""),
        "measurement_gate_passed": measurement_gate_passed,
    }


def write_outputs(
    analysis: dict[str, Any],
    *,
    grid_config: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    metric_files: list[str],
    output_json: Path,
    output_md: Path,
    output_csv: Path,
) -> None:
    payload = {
        "material_passport": {
            "type": "Repeated FID measurement audit",
            "verification_status": "ANALYZED",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "grid_config": str(grid_config),
            "protocol_path": str(protocol_path),
            "protocol_id": str(protocol["protocol_id"]),
            "registration_status": str(protocol.get("registration_status", "")),
            "metric_files": metric_files,
        },
        "analysis_protocol": protocol,
        **analysis,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        secondary_metrics = analysis["secondary_diagnostics"]["available_metrics"]
        secondary_columns = [
            column
            for metric in secondary_metrics
            for column in (f"mean_{metric}", f"sample_sd_{metric}")
        ]
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "candidate_family",
                "training_seed",
                "n",
                "mean_fid128",
                "sample_std",
                "ci95_low",
                "ci95_high",
                "checkpoint_source",
                *secondary_columns,
            ],
        )
        writer.writeheader()
        for run in analysis["run_summaries"]:
            summary = run["summary"] or {}
            row = {
                    "run_name": run["run_name"],
                    "candidate_family": run["candidate_family"],
                    "training_seed": run["training_seed"],
                    "n": summary.get("n", 0),
                    "mean_fid128": summary.get("mean", ""),
                    "sample_std": summary.get("sample_std", ""),
                    "ci95_low": summary.get("ci95_low", ""),
                    "ci95_high": summary.get("ci95_high", ""),
                    "checkpoint_source": run["checkpoint_source"],
            }
            for metric in secondary_metrics:
                metric_summary = run["secondary_summaries"].get(metric) or {}
                row[f"mean_{metric}"] = metric_summary.get("mean", "")
                row[f"sample_sd_{metric}"] = metric_summary.get("sample_std", "")
            writer.writerow(row)

    lines = [
        "# Repeated FID Measurement Audit",
        "",
        "## Material Passport",
        "",
        "- Verification Status: `ANALYZED`",
        f"- Primary metric: `{PRIMARY_METRIC}`",
        f"- Grid: `{grid_config}`",
        f"- Protocol: `{protocol_path}` (`{protocol['protocol_id']}`)",
        f"- Registration status: `{protocol.get('registration_status', '')}`",
        f"- Parsed metric files: {len(metric_files)}",
        "- Checkpoints downloaded locally: no",
        "",
        "## Per-checkpoint Results",
        "",
        "| family | training seed | repeats | mean FID128 | sample SD | 95% CI | run |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for run in analysis["run_summaries"]:
        summary = run["summary"]
        if summary:
            ci = f"{summary['ci95_low']:.4f}--{summary['ci95_high']:.4f}"
            mean = f"{summary['mean']:.4f}"
            sample_std = f"{summary['sample_std']:.4f}"
            n = summary["n"]
        else:
            ci = mean = sample_std = "missing"
            n = 0
        lines.append(
            f"| {run['candidate_family']} | {run['training_seed']} | {n} | {mean} | "
            f"{sample_std} | {ci} | `{run['run_name']}` |"
        )

    lines.extend(
        [
            "",
            "## Paired C4 minus C0",
            "",
            "| training seed | paired eval seeds | C0 mean | C4 mean | delta mean | delta 95% CI |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in analysis["paired_by_training_seed"]:
        delta = item["delta_c4_minus_c0"]
        c0_mean = f"{item['c0_mean']:.4f}" if item["c0_mean"] is not None else "missing"
        c4_mean = f"{item['c4_mean']:.4f}" if item["c4_mean"] is not None else "missing"
        if delta:
            delta_mean = f"{delta['mean']:.4f}"
            delta_ci = f"{delta['ci95_low']:.4f}--{delta['ci95_high']:.4f}"
        else:
            delta_mean = delta_ci = "missing"
        lines.append(
            f"| {item['training_seed']} | {len(item['common_eval_seeds'])} | "
            f"{c0_mean} | {c4_mean} | "
            f"{delta_mean} | {delta_ci} |"
        )

    paired = analysis["paired_eval_delta_c4_minus_c0"]
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            f"- Pooled within-checkpoint eval SD: `{analysis['pooled_within_checkpoint_eval_sd']:.4f}`.",
            f"- Practical threshold `max(0.1, 2 x eval-SD)`: `{analysis['practical_improvement_threshold']:.4f}`.",
            f"- Overall paired eval delta C4-C0: `{paired['mean']:.4f}`." if paired else "- Overall paired delta: missing.",
            f"- All training seeds favor C4: `{analysis['all_training_seeds_favor_c4']}`.",
            f"- Complete paired evaluation seeds: `{analysis['complete_evaluation_seed_pairs']}`.",
            f"- Measurement gate passed: `{analysis['measurement_gate_passed']}`.",
            f"- Frozen outcome code: `{analysis['outcome_code']}`.",
            f"- Frozen next action: `{analysis['outcome_action']}`.",
            "",
            "The repeated generation seeds estimate evaluation noise only. The two training seeds remain the independent unit for model-family claims.",
            "",
            "## Comparability Audit",
            "",
            f"- Status: `{analysis['comparability_audit']['status']}`.",
            f"- Errors: `{len(analysis['comparability_audit']['errors'])}`.",
            f"- Warnings: `{len(analysis['comparability_audit']['warnings'])}`.",
        ]
    )
    for warning in analysis["comparability_audit"]["warnings"]:
        lines.append(f"- Warning: {warning}")
    for error in analysis["comparability_audit"]["errors"]:
        lines.append(f"- Error: {error}")

    secondary = analysis["secondary_diagnostics"]
    lines.extend(
        [
            "",
            "## Mechanism Diagnostics",
            "",
            "These metrics are descriptive diagnostics only and do not enter the frozen FID128 decision gate.",
            f"Available: `{', '.join(secondary['available_metrics']) or 'none'}`.",
            f"Unavailable from eval-only artifacts: `{', '.join(secondary['unavailable_metrics']) or 'none'}`.",
        ]
    )
    if secondary["available_metrics"]:
        lines.extend(
            [
                "",
                "| family | training seed | metric | repeats | mean | sample SD |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for run in analysis["run_summaries"]:
            for metric in secondary["available_metrics"]:
                metric_summary = run["secondary_summaries"].get(metric)
                if metric_summary:
                    lines.append(
                        f"| {run['candidate_family']} | {run['training_seed']} | `{metric}` | "
                        f"{metric_summary['n']} | {metric_summary['mean']:.6f} | "
                        f"{metric_summary['sample_std']:.6f} |"
                    )
        lines.extend(
            [
                "",
                "| training seed | metric | paired seeds | C0 mean | C4 mean | delta C4-C0 |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for seed_group in secondary["paired_by_training_seed"]:
            for metric_row in seed_group["metrics"]:
                delta = metric_row["delta_candidate_minus_control"]
                control_mean = metric_row["control_mean"]
                candidate_mean = metric_row["candidate_mean"]
                control_text = f"{control_mean:.6f}" if control_mean is not None else "missing"
                candidate_text = (
                    f"{candidate_mean:.6f}" if candidate_mean is not None else "missing"
                )
                delta_text = f"{delta['mean']:.6f}" if delta else "missing"
                lines.append(
                    f"| {seed_group['training_seed']} | `{metric_row['metric']}` | "
                    f"{len(metric_row['common_eval_seeds'])} | {control_text} | "
                    f"{candidate_text} | {delta_text} |"
                )
    lines.extend(
        [
            "",
            "## Fallacy Scan",
            "",
            "- Simpson's paradox: checked across training seeds; report per-seed and aggregate directions separately.",
            "- Ecological fallacy: not applicable; no individual-level inference is made.",
            "- Berkson's paradox: not applicable to this paired checkpoint audit.",
            "- Collider bias: not applicable; no adjusted causal model is fit.",
            "- Base-rate neglect: not applicable to FID.",
            "- Regression to the mean: caution; checkpoints were selected after observing earlier trajectories.",
            "- Survivorship bias: caution if failed or missing jobs are omitted.",
            "- Look-elsewhere effect: caution; prior checkpoint/config exploration was extensive.",
            "- Garden of forking paths: controlled partially by the predeclared 400k checkpoint and FID128 endpoint.",
            "- Correlation versus causation: no causal claim is made.",
            "- Reverse causality: not applicable.",
            "- Coverage: 11/11 checked.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze repeated FID128 evaluations for matched C0/C4 checkpoints.")
    parser.add_argument("--grid-config", default="configs/gmm_tide_fid_repeat4_grid.json")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--search-root", default="outputs/kaggle_jobs/gmm_tide_fm")
    parser.add_argument("--output-json", default="reports/gmm_tide_fid_repeat_audit.json")
    parser.add_argument("--output-md", default="reports/gmm_tide_fid_repeat_audit.md")
    parser.add_argument("--output-csv", default="reports/gmm_tide_fid_repeat_audit.csv")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grid_config = Path(args.grid_config)
    protocol_path = Path(args.protocol)
    jobs = load_jobs(grid_config)
    protocol = load_protocol(protocol_path)
    rows, metric_files = discover_metric_rows(Path(args.search_root))
    analysis = analyze_rows(jobs, rows, protocol=protocol)
    analysis["comparability_audit"] = audit_comparability(jobs, rows, protocol)
    if args.strict and analysis["comparability_audit"]["status"] != "PASS":
        raise SystemExit(
            "Comparability audit failed:\n- "
            + "\n- ".join(analysis["comparability_audit"]["errors"])
        )
    write_outputs(
        analysis,
        grid_config=grid_config,
        protocol_path=protocol_path,
        protocol=protocol,
        metric_files=metric_files,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        output_csv=Path(args.output_csv),
    )
    print(json.dumps({"ok": True, "runs": len(analysis["run_summaries"]), "metric_files": len(metric_files)}))


if __name__ == "__main__":
    main()
