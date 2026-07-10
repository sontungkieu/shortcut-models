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


def analyze_rows(jobs: dict[str, dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_run_seed = {}
    for row in rows:
        if row["run_name"] not in jobs:
            continue
        by_run_seed[(row["run_name"], row["eval_seed"])] = row

    run_summaries = []
    run_values = {}
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
            }
        )

    paired_by_training_seed = []
    all_eval_deltas = []
    for training_seed in sorted(
        set(seed for family, seed in run_values if family in {"C0", "C4"})
    ):
        c0 = run_values.get(("C0", training_seed), {})
        c4 = run_values.get(("C4", training_seed), {})
        common = sorted(set(c0) & set(c4))
        deltas = [c4[eval_seed] - c0[eval_seed] for eval_seed in common]
        all_eval_deltas.extend(deltas)
        paired_by_training_seed.append(
            {
                "training_seed": training_seed,
                "common_eval_seeds": common,
                "c0_mean": summarize_values(c0.values())["mean"] if c0 else None,
                "c4_mean": summarize_values(c4.values())["mean"] if c4 else None,
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
    practical_threshold = max(0.1, 2.0 * pooled_eval_sd)
    paired_summary = summarize_values(all_eval_deltas) if all_eval_deltas else None
    all_training_seeds_favor_c4 = bool(paired_by_training_seed) and all(
        item["delta_c4_minus_c0"] is not None
        and float(item["delta_c4_minus_c0"]["mean"]) < 0
        for item in paired_by_training_seed
    )
    measurement_gate_passed = bool(
        paired_summary
        and float(paired_summary["mean"]) <= -practical_threshold
        and all_training_seeds_favor_c4
    )

    return {
        "primary_metric": PRIMARY_METRIC,
        "run_summaries": run_summaries,
        "paired_by_training_seed": paired_by_training_seed,
        "paired_eval_delta_c4_minus_c0": paired_summary,
        "pooled_within_checkpoint_eval_sd": pooled_eval_sd,
        "practical_improvement_threshold": practical_threshold,
        "all_training_seeds_favor_c4": all_training_seeds_favor_c4,
        "measurement_gate_passed": measurement_gate_passed,
    }


def write_outputs(
    analysis: dict[str, Any],
    *,
    grid_config: Path,
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
            "metric_files": metric_files,
        },
        **analysis,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
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
            ],
        )
        writer.writeheader()
        for run in analysis["run_summaries"]:
            summary = run["summary"] or {}
            writer.writerow(
                {
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
            )

    lines = [
        "# Repeated FID Measurement Audit",
        "",
        "## Material Passport",
        "",
        "- Verification Status: `ANALYZED`",
        f"- Primary metric: `{PRIMARY_METRIC}`",
        f"- Grid: `{grid_config}`",
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
            f"- Measurement gate passed: `{analysis['measurement_gate_passed']}`.",
            "",
            "The repeated generation seeds estimate evaluation noise only. The two training seeds remain the independent unit for model-family claims.",
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
    parser.add_argument("--search-root", default="outputs/kaggle_jobs/gmm_tide_fm")
    parser.add_argument("--output-json", default="reports/gmm_tide_fid_repeat_audit.json")
    parser.add_argument("--output-md", default="reports/gmm_tide_fid_repeat_audit.md")
    parser.add_argument("--output-csv", default="reports/gmm_tide_fid_repeat_audit.csv")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grid_config = Path(args.grid_config)
    jobs = load_jobs(grid_config)
    rows, metric_files = discover_metric_rows(Path(args.search_root))
    analysis = analyze_rows(jobs, rows)
    if args.strict:
        missing = [
            run["run_name"]
            for run in analysis["run_summaries"]
            if not run["summary"]
            or int(run["summary"]["n"])
            != len(parse_eval_fid_seeds(str(jobs[run["run_name"]]["eval_fid_seeds"])))
        ]
        if missing:
            raise SystemExit(f"Missing or incomplete repeated FID rows: {', '.join(missing)}")
    write_outputs(
        analysis,
        grid_config=grid_config,
        metric_files=metric_files,
        output_json=Path(args.output_json),
        output_md=Path(args.output_md),
        output_csv=Path(args.output_csv),
    )
    print(json.dumps({"ok": True, "runs": len(analysis["run_summaries"]), "metric_files": len(metric_files)}))


if __name__ == "__main__":
    main()
