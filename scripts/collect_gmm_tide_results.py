from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def finite(value, default=None):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def best_and_last_eval(train_jsonl: Path) -> dict[str, Any]:
    eval_rows = [row for row in iter_jsonl(train_jsonl) or [] if row.get("phase") == "eval"]
    out: dict[str, Any] = {"eval_count": len(eval_rows)}
    if not eval_rows:
        return out
    last = eval_rows[-1]
    out["last_eval_step"] = last.get("step")
    for timestep in ("1", "4", "32", "128"):
        key = f"fid/timesteps/{timestep}"
        rows = [(row.get("step"), finite(row.get(key))) for row in eval_rows if finite(row.get(key)) is not None]
        if rows:
            best = min(rows, key=lambda item: item[1])
            out[f"fid{timestep}_best"] = best[1]
            out[f"fid{timestep}_best_step"] = best[0]
            out[f"fid{timestep}_last"] = rows[-1][1]
    for metric in (
        "flow/straightness_ratio_mean",
        "flow/curvature_proxy_mean",
        "flow/path_length_mean",
        "flow/endpoint_displacement_mean",
    ):
        if metric in last:
            out[metric.replace("/", "_") + "_last"] = finite(last.get(metric))
    return out


def derive_overfit(router_summary: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: router_summary[key]
        for key in router_summary
        if key.startswith("router_overfit/")
    }
    pairs = {
        "loss": "loss",
        "kl_to_gmm": "router/kl_to_gmm",
        "cross_entropy": "router/cross_entropy",
        "top1_agreement": "router/top1_agreement",
        "usage_entropy_normalized": "router/usage_entropy_normalized",
    }
    for short, metric in pairs.items():
        out_key = f"router_overfit/{short}_gap"
        if out_key in out:
            continue
        train = finite(router_summary.get(f"router_train/{metric}"))
        valid = finite(router_summary.get(f"router_valid/{metric}"))
        if train is None or valid is None:
            continue
        if short in ("top1_agreement",):
            out[out_key] = train - valid
        elif short == "usage_entropy_normalized":
            out[out_key] = abs(train - valid)
        else:
            out[out_key] = valid - train
    return out


def collect_one(diag_dir: Path) -> dict[str, Any]:
    gmm = load_json(diag_dir / "gmm_metrics.json")
    router = load_json(diag_dir / "router_metrics_summary.json")
    train = load_json(diag_dir / "train_metrics_summary.json")
    evals = best_and_last_eval(diag_dir / "train_metrics.jsonl")
    overfit = derive_overfit(router)

    row: dict[str, Any] = {
        "diagnostics_dir": str(diag_dir),
        "run_name": diag_dir.parent.name,
        "num_modes": gmm.get("num_modes"),
        "gmm_pi_prior_type": gmm.get("gmm_pi_prior_type"),
        "gmm_pi_prior_strength": gmm.get("gmm_pi_prior_strength"),
        "gmm_var_prior_type": gmm.get("gmm_var_prior_type"),
        "gmm_var_prior_strength": gmm.get("gmm_var_prior_strength"),
        "gmm_var_prior_target_var": gmm.get("gmm_var_prior_target_var"),
        "gmm_min_std_data_frac": gmm.get("gmm_min_std_data_frac"),
        "train_nll": gmm.get("train_nll"),
        "valid_nll": gmm.get("valid_nll"),
        "pi_entropy_normalized": gmm.get("pi_entropy_normalized"),
        "data_variance_mean": gmm.get("data_variance_mean"),
        "component_variance_mean": gmm.get("component_variance_mean"),
        "var_floor_hit_rate": gmm.get("var_floor_hit_rate"),
        "router_step": router.get("step"),
        "router_train_loss": router.get("router_train/loss"),
        "router_valid_loss": router.get("router_valid/loss"),
        "router_valid_top1_agreement": router.get("router_valid/router/top1_agreement"),
        "router_valid_usage_entropy_normalized": router.get("router_valid/router/usage_entropy_normalized"),
        "train_step": train.get("step"),
        "train_loss": train.get("training/loss"),
        "valid_loss": train.get("training/loss_valid"),
        "fm_target_variance": train.get("training/fm/target_variance"),
        "fm_pred_variance": train.get("training/fm/pred_variance"),
        "fm_residual_variance": train.get("training/fm/loss_residual_variance"),
        "x0_magnitude": train.get("training/x0_magnitude"),
        "x1_magnitude": train.get("training/x1_magnitude"),
        "router_topk_mass": train.get("training/router/topk_mass"),
        "router_top1_prob_mean": train.get("training/router/top1_prob_mean"),
    }
    row.update({key.replace("/", "_"): value for key, value in overfit.items()})
    row.update(evals)
    return row


def fmt(value, digits=4):
    value = finite(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_outputs(rows: list[dict[str, Any]], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# GMM-TIDE Results",
        "",
        "| run | K | step | FID128 best/last | FID32 best/last | router valid/top1 | router loss gap | FM pred/target var | x0/x1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (finite(item.get("fid128_best"), 1e9), item.get("run_name", ""))):
        pred = finite(row.get("fm_pred_variance"))
        target = finite(row.get("fm_target_variance"))
        x0 = finite(row.get("x0_magnitude"))
        x1 = finite(row.get("x1_magnitude"))
        lines.append(
            f"| {row.get('run_name', '')} | {row.get('num_modes', '')} | {row.get('train_step', '')} | "
            f"{fmt(row.get('fid128_best'), 2)}/{fmt(row.get('fid128_last'), 2)} | "
            f"{fmt(row.get('fid32_best'), 2)}/{fmt(row.get('fid32_last'), 2)} | "
            f"{fmt(row.get('router_valid_loss'), 3)}/{fmt(row.get('router_valid_top1_agreement'), 3)} | "
            f"{fmt(row.get('router_overfit_loss_gap'), 3)} | "
            f"{fmt(pred / target if pred is not None and target else None, 3)} | "
            f"{fmt(x0 / x1 if x0 is not None and x1 else None, 3)} |"
        )
    output_json.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GMM-TIDE diagnostic summaries.")
    parser.add_argument("--input-root", default="outputs/kaggle", help="Root containing downloaded Kaggle outputs.")
    parser.add_argument("--output-json", default="reports/gmm_tide_results.json")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    diag_dirs = sorted(
        path for path in input_root.rglob("diagnostics")
        if (path / "train_metrics_summary.json").exists()
    )
    rows = [collect_one(path) for path in diag_dirs]
    write_outputs(rows, Path(args.output_json))
    print(f"Collected {len(rows)} GMM-TIDE run(s) into {args.output_json}")


if __name__ == "__main__":
    main()
