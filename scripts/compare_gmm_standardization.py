from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAIR_FIELDS = (
    "dataset_name",
    "num_modes",
    "gmm_min_std_data_frac",
    "gmm_min_std",
    "gmm_pi_prior_type",
    "gmm_pi_prior_strength",
    "gmm_var_prior_type",
    "gmm_var_prior_strength",
    "gmm_var_prior_target_var",
    "fit_samples",
    "valid_samples",
)


def pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in PAIR_FIELDS)


def fnum(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value, digits=4):
    value = fnum(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def delta(std: dict[str, Any], raw: dict[str, Any], field: str):
    lhs = fnum(std.get(field))
    rhs = fnum(raw.get(field))
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.setdefault("latent_train_nll", row.get("train_nll"))
    row.setdefault("latent_valid_nll", row.get("valid_nll"))
    row.setdefault("latent_component_variance_mean", row.get("component_variance_mean"))
    row.setdefault("latent_overlap_proxy_max", row.get("overlap_proxy_max"))
    row.setdefault("latent_var_floor_hit_rate", row.get("var_floor_hit_rate"))
    if "num_modes" not in row and "gmm_num_modes" in row:
        row["num_modes"] = row["gmm_num_modes"]
    return row


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "jobs" in payload:
        return [normalize_row(row) for row in payload["jobs"] if row.get("parse_status") == "ok"]
    if isinstance(payload, dict) and "runs" in payload:
        return [normalize_row(row) for row in payload["runs"]]
    if isinstance(payload, list):
        return [normalize_row(row) for row in payload]
    raise SystemExit(f"Unsupported result schema in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw vs standardized GMM ablation pairs.")
    parser.add_argument("--results-json", required=True)
    parser.add_argument(
        "--baseline-json",
        default="",
        help="Optional raw baseline results JSON. When set, raw rows come from this file and standardized rows from --results-json.",
    )
    parser.add_argument("--output-md", default="reports/gmm_standardize_comparison.md")
    args = parser.parse_args()

    result_rows = load_rows(Path(args.results_json))
    baseline_rows = load_rows(Path(args.baseline_json)) if args.baseline_json else []
    grouped: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = {}
    for row in baseline_rows:
        if int(row.get("gmm_standardize_data", 0)) == 0:
            grouped.setdefault(pair_key(row), {})[0] = row
    for row in result_rows:
        grouped.setdefault(pair_key(row), {})[int(row.get("gmm_standardize_data", 0))] = row

    pairs = []
    for variants in grouped.values():
        raw = variants.get(0)
        std = variants.get(1)
        if raw and std:
            pairs.append((raw, std))

    pairs.sort(
        key=lambda item: (
            delta(item[1], item[0], "latent_valid_nll") if delta(item[1], item[0], "latent_valid_nll") is not None else 1e9,
            item[0].get("num_modes", 0),
            item[0].get("run_name", ""),
        )
    )

    lines = [
        "# GMM Standardization Comparison",
        "",
        "Negative deltas mean the standardized-fit GMM is better after unnormalizing back to latent space.",
        f"Raw baseline: `{args.baseline_json or args.results_json}`.",
        f"Standardized results: `{args.results_json}`.",
        "",
        "| K | prior | coverage std-frac | var prior | raw run | std run | raw latent valid nll | std latent valid nll | d latent valid nll | d valid count ratio | d valid dead | d latent comp var | d latent floor hit | d latent overlap max |",
        "|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for raw, std in pairs:
        lines.append(
            f"| {raw.get('num_modes', '')} | "
            f"{raw.get('gmm_pi_prior_type', '')}:{fmt(raw.get('gmm_pi_prior_strength'), 2)} | "
            f"{fmt(raw.get('gmm_min_std_data_frac'), 3)} | "
            f"{raw.get('gmm_var_prior_type', '')}:{fmt(raw.get('gmm_var_prior_strength'), 1)}@{fmt(raw.get('gmm_var_prior_target_var'), 2)} | "
            f"{raw.get('run_name', '')} | {std.get('run_name', '')} | "
            f"{fmt(raw.get('latent_valid_nll'), 2)} | "
            f"{fmt(std.get('latent_valid_nll'), 2)} | "
            f"{fmt(delta(std, raw, 'latent_valid_nll'), 2)} | "
            f"{fmt(delta(std, raw, 'valid_count_ratio'), 3)} | "
            f"{fmt(delta(std, raw, 'valid_dead_components'), 0)} | "
            f"{fmt(delta(std, raw, 'latent_component_variance_mean'), 6)} | "
            f"{fmt(delta(std, raw, 'latent_var_floor_hit_rate'), 6)} | "
            f"{fmt(delta(std, raw, 'latent_overlap_proxy_max'), 6)} |"
        )

    output = Path(args.output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output} with {len(pairs)} raw/std pair(s)")


if __name__ == "__main__":
    main()
