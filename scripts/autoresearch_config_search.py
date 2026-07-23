from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_GLOBS = ("configs/gmm_tide_fm*_grid.json",)
DEFAULT_RESULT_GLOBS = (
    "reports/*results*.json",
    "reports/*metrics*.json",
    "reports/latest_*.json",
    "reports/tide_selected*.json",
)
DEFAULT_OBJECTIVE = "fid128_best"

FINGERPRINT_IGNORE_KEYS = {
    "ablation_family",
    "baseline_source",
    "grid_index",
    "repo_commit",
    "run_name",
    "source_id",
    "source_run_name",
}

RANK_COLUMNS = (
    "fid128_best",
    "fid32_best",
    "fid4_best",
    "valid_loss",
    "train_loss",
    "train_step",
)

SEED_CONFIG_PREFIXES = ("gmm_", "model_", "router_")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expand_paths(patterns: list[str] | tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        paths.extend(matches)
    return sorted(set(paths))


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def slugify(value: str, max_length: int = 90) -> str:
    value = value.lower()
    value = value.replace(".", "p").replace("+", "plus")
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value[:max_length].strip("-") or "run"


def float_token(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return str(value)
    if number == 0:
        return "0"
    if abs(number) < 1e-3:
        return f"{number:.0e}".replace("-", "m").replace("+", "")
    return f"{number:g}".replace(".", "p").replace("-", "m")


def iter_result_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("runs", []):
            if not isinstance(row, dict):
                continue
            row = dict(row)
            row["_source_report"] = str(path)
            rows.append(row)
    return rows


def merged_grid_jobs(paths: list[Path]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        defaults = payload.get("defaults", {})
        if not isinstance(defaults, dict):
            defaults = {}
        for index, raw_job in enumerate(payload.get("jobs", [])):
            if not isinstance(raw_job, dict):
                continue
            job = dict(defaults)
            job.update(raw_job)
            job.setdefault("grid_index", index)
            job["_source_grid"] = str(path)
            jobs.append(job)
    return jobs


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in config.items()
        if not key.startswith("_") and key not in FINGERPRINT_IGNORE_KEYS
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def rank_key(row: dict[str, Any], objective: str) -> tuple[float, float, float, float, float, str]:
    primary = finite_float(row.get(objective))
    if primary is None:
        primary = math.inf
    fid32 = finite_float(row.get("fid32_best"))
    fid4 = finite_float(row.get("fid4_best"))
    valid_loss = finite_float(row.get("valid_loss"))
    train_loss = finite_float(row.get("train_loss"))
    return (
        primary,
        fid32 if fid32 is not None else math.inf,
        fid4 if fid4 is not None else math.inf,
        valid_loss if valid_loss is not None else math.inf,
        train_loss if train_loss is not None else math.inf,
        str(row.get("run_name", "")),
    )


def ranked_rows(rows: list[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    best_by_run: dict[str, dict[str, Any]] = {}
    for row in rows:
        if finite_float(row.get(objective)) is None:
            continue
        run_name = str(row.get("run_name") or row.get("diagnostics_dir") or "")
        if not run_name:
            continue
        previous = best_by_run.get(run_name)
        if previous is None or rank_key(row, objective) < rank_key(previous, objective):
            best_by_run[run_name] = row
    return sorted(best_by_run.values(), key=lambda item: rank_key(item, objective))


def enrich_metric_row(row: dict[str, Any], known_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run_name = str(row.get("run_name", ""))
    out = dict(known_by_name.get(run_name, {}))
    out.update({key: value for key, value in row.items() if not key.startswith("_")})

    if "gmm_num_modes" not in out and finite_float(out.get("num_modes")) is not None:
        out["gmm_num_modes"] = int(float(out["num_modes"]))
    if "gmm_min_var_data_frac" not in out and finite_float(out.get("gmm_min_std_data_frac")) is not None:
        out["gmm_min_var_data_frac"] = float(out["gmm_min_std_data_frac"]) ** 2
    if "gmm_router_topk" not in out:
        match = re.search(r"top(\d+)", run_name)
        if match:
            out["gmm_router_topk"] = int(match.group(1))
    if "model_lr" not in out:
        match = re.search(r"lr(\d+)e(\d+)", run_name)
        if match:
            out["model_lr"] = float(f"{match.group(1)}e-{match.group(2)}")
    if "model_t_sampling" not in out and "beta" in run_name:
        out["model_t_sampling"] = "beta"
    if "gmm_router_update_policy" not in out and "joint" in run_name:
        out["gmm_router_update_policy"] = "joint"
    return out


def load_template_grid(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    jobs = []
    for raw_job in payload.get("jobs", []):
        if not isinstance(raw_job, dict):
            continue
        job = dict(defaults)
        job.update(raw_job)
        jobs.append(job)
    return dict(defaults), jobs


def compact_job(job: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in sorted(job):
        if key.startswith("_"):
            continue
        if key == "run_name" or defaults.get(key) != job[key]:
            compact[key] = job[key]
    return compact


def mutation_specs(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    alpha = finite_float(base.get("model_t_beta_alpha")) or 3.0
    beta = finite_float(base.get("model_t_beta_beta")) or 1.0
    lr = finite_float(base.get("model_lr")) or 1e-4
    topk = int(finite_float(base.get("gmm_router_topk")) or 2)
    modes = int(finite_float(base.get("gmm_num_modes")) or 16)
    usage = finite_float(base.get("gmm_router_usage_weight"))
    if usage is None:
        usage = 0.01

    specs: list[tuple[str, dict[str, Any]]] = [
        ("beta-alpha-up", {"model_t_sampling": "beta", "model_t_beta_alpha": round(alpha + 0.5, 3), "model_t_beta_beta": beta}),
        ("beta-alpha-down", {"model_t_sampling": "beta", "model_t_beta_alpha": round(max(0.5, alpha - 0.5), 3), "model_t_beta_beta": beta}),
        ("beta-beta-up", {"model_t_sampling": "beta", "model_t_beta_alpha": alpha, "model_t_beta_beta": round(beta + 0.3, 3)}),
        ("enddense-p06", {"model_eval_ode_schedule": "end_dense", "model_eval_ode_power": 0.6}),
        ("enddense-p08", {"model_eval_ode_schedule": "end_dense", "model_eval_ode_power": 0.8}),
        ("lr-half", {"model_lr": lr / 2}),
        ("usage-zero", {"gmm_router_usage_weight": 0.0}),
        ("usage-up", {"gmm_router_usage_weight": round(max(0.03, usage * 3), 4)}),
    ]
    if topk < modes:
        specs.insert(3, ("topk-up", {"gmm_router_topk": min(modes, max(topk + 2, topk * 2))}))
    return specs


def make_run_name(label: str, seed_rank: int, seed: dict[str, Any], strategy: str, job: dict[str, Any]) -> str:
    alpha = job.get("model_t_beta_alpha", seed.get("model_t_beta_alpha", ""))
    beta = job.get("model_t_beta_beta", seed.get("model_t_beta_beta", ""))
    lr = job.get("model_lr", seed.get("model_lr", ""))
    parts = [
        "ar",
        label,
        f"r{seed_rank}",
        strategy,
        f"k{job.get('gmm_num_modes', seed.get('gmm_num_modes', 'x'))}",
        f"top{job.get('gmm_router_topk', seed.get('gmm_router_topk', 'x'))}",
    ]
    if job.get("model_t_sampling") == "beta" or alpha:
        parts.append(f"b{float_token(alpha)}-{float_token(beta)}")
    if lr:
        parts.append(f"lr{float_token(lr)}")
    return slugify("-".join(str(part) for part in parts))


def write_rank_report(path: Path, rows: list[dict[str, Any]], objective: str) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": objective,
        "runs": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Autoresearch Rank Report",
        "",
        f"- Objective: `{objective}` (lower is better)",
        f"- Runs: {len(rows)}",
        "",
        "| rank | run | fid128 | fid32 | fid4 | valid | step | source |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {row.get('run_name', '')} | {fmt(row.get('fid128_best'))} | "
            f"{fmt(row.get('fid32_best'))} | {fmt(row.get('fid4_best'))} | "
            f"{fmt(row.get('valid_loss'), digits=4)} | {row.get('train_step', '')} | "
            f"`{row.get('_source_report', '')}` |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 3) -> str:
    number = finite_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def command_rank(args: argparse.Namespace) -> None:
    result_paths = expand_paths(args.results or list(DEFAULT_RESULT_GLOBS))
    rows = ranked_rows(iter_result_rows(result_paths), args.objective)
    if args.top:
        rows = rows[: args.top]
    if args.output:
        write_rank_report(Path(args.output), rows, args.objective)
    print(f"Found {len(rows)} ranked run(s) for objective {args.objective}")
    for index, row in enumerate(rows[: args.print_top], start=1):
        print(
            f"{index:>2}. {row.get('run_name', '')} "
            f"{args.objective}={fmt(row.get(args.objective))} "
            f"fid32={fmt(row.get('fid32_best'))} step={row.get('train_step', '')}"
        )


def command_propose(args: argparse.Namespace) -> None:
    result_paths = expand_paths(args.results or list(DEFAULT_RESULT_GLOBS))
    config_paths = expand_paths(args.config_glob or list(DEFAULT_CONFIG_GLOBS))
    known_jobs = merged_grid_jobs(config_paths)
    known_by_name = {str(job.get("run_name")): job for job in known_jobs if job.get("run_name")}
    known_config_keys = {
        key
        for job in known_jobs
        for key in job
        if not key.startswith("_") and key not in FINGERPRINT_IGNORE_KEYS
    }
    existing_fingerprints = {config_fingerprint(job) for job in known_jobs}

    defaults, template_jobs = load_template_grid(Path(args.template_grid))
    known_config_keys.update(defaults)
    if not template_jobs:
        raise SystemExit(f"Template grid has no jobs: {args.template_grid}")
    template = template_jobs[0]

    ranked = ranked_rows(iter_result_rows(result_paths), args.objective)
    if not ranked:
        raise SystemExit(f"No result rows found with finite {args.objective}")
    seeds = [enrich_metric_row(row, known_by_name) for row in ranked[: args.seed_top]]

    candidates: list[dict[str, Any]] = []
    candidate_notes: list[dict[str, Any]] = []
    seen = set(existing_fingerprints)
    for seed_rank, seed in enumerate(seeds, start=1):
        base = copy.deepcopy(template)
        for key, value in seed.items():
            if key.startswith("_") or key in RANK_COLUMNS or key in {"diagnostics_dir", "eval_count", "last_eval_step"}:
                continue
            if key in known_config_keys and key.startswith(SEED_CONFIG_PREFIXES):
                base[key] = value
        base["ablation_family"] = args.family
        base["source_run_name"] = seed.get("run_name", base.get("source_run_name", ""))

        for strategy, changes in mutation_specs(base):
            job = copy.deepcopy(base)
            job.update(changes)
            job["run_name"] = make_run_name(args.label, seed_rank, seed, strategy, job)
            fingerprint = config_fingerprint(job)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            candidates.append(compact_job(job, defaults))
            candidate_notes.append(
                {
                    "run_name": job["run_name"],
                    "seed_rank": seed_rank,
                    "seed_run_name": seed.get("run_name", ""),
                    "strategy": strategy,
                    "seed_objective": seed.get(args.objective),
                    "changes": changes,
                }
            )
            if len(candidates) >= args.budget:
                break
        if len(candidates) >= args.budget:
            break

    output_path = Path(args.output_grid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = {
        "defaults": defaults,
        "jobs": candidates,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/autoresearch_config_search.py",
            "objective": args.objective,
            "template_grid": args.template_grid,
            "seed_top": args.seed_top,
            "source_reports": [str(path) for path in result_paths],
            "source_grids": [str(path) for path in config_paths],
        },
    }
    output_path.write_text(json.dumps(grid, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_proposal_report(output_path.with_suffix(".md"), candidate_notes, ranked[: args.seed_top], args.objective)
    print(f"Wrote {len(candidates)} candidate job(s) to {output_path}")


def write_proposal_report(
    path: Path,
    notes: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    objective: str,
) -> None:
    lines = [
        "# Autoresearch Candidate Grid",
        "",
        f"- Objective: `{objective}` (lower is better)",
        f"- Candidates: {len(notes)}",
        "",
        "## Seeds",
        "",
        "| rank | run | fid128 | fid32 | fid4 | step |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, seed in enumerate(seeds, start=1):
        lines.append(
            f"| {index} | {seed.get('run_name', '')} | {fmt(seed.get('fid128_best'))} | "
            f"{fmt(seed.get('fid32_best'))} | {fmt(seed.get('fid4_best'))} | {seed.get('train_step', '')} |"
        )
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| job | seed | strategy | run | seed objective | changes |",
            "|---:|---:|---|---|---:|---|",
        ]
    )
    for index, note in enumerate(notes, start=1):
        changes = ", ".join(f"{key}={value}" for key, value in sorted(note["changes"].items()))
        lines.append(
            f"| {index} | {note['seed_rank']} | {note['strategy']} | {note['run_name']} | "
            f"{fmt(note.get('seed_objective'))} | `{changes}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank completed GMM-TIDE runs and generate Karpathy-style autoresearch config grids."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rank = subparsers.add_parser("rank", help="Rank existing result reports.")
    rank.add_argument("--results", action="append", default=[], help="Result JSON path or glob. Repeatable.")
    rank.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    rank.add_argument("--top", type=int, default=0, help="Limit rows written to --output.")
    rank.add_argument("--print-top", type=int, default=12)
    rank.add_argument("--output", default="", help="Optional JSON report path.")
    rank.set_defaults(func=command_rank)

    propose = subparsers.add_parser("propose", help="Generate a bounded next-candidate grid.")
    propose.add_argument("--results", action="append", default=[], help="Result JSON path or glob. Repeatable.")
    propose.add_argument("--config-glob", action="append", default=[], help="Known config grid path/glob. Repeatable.")
    propose.add_argument("--template-grid", default="configs/gmm_tide_fm_next10_grid.json")
    propose.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    propose.add_argument("--seed-top", type=int, default=4)
    propose.add_argument("--budget", type=int, default=6)
    propose.add_argument("--label", default=datetime.now().strftime("%Y%m%d"))
    propose.add_argument("--family", default="autoresearch")
    propose.add_argument("--output-grid", default="configs/autoresearch/gmm_tide_fm_autoresearch_grid.json")
    propose.set_defaults(func=command_propose)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
