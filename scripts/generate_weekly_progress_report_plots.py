from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


BASELINE = {
    "batch": "baseline",
    "period": "15/05-13/06",
    "protocol": "single_best",
    "label": "baseline K16 top2 soft0.75 dir512",
    "run_name": "tide-k16-top2-softv0p75-s128-dir512",
    "fid128": 6.969,
    "step": 365600,
    "pred_target_var_ratio": 0.674,
    "x0_x1_mag_ratio": 1.021,
    "router_usage_entropy": 0.948,
    "x0_x1_cosine": None,
    "topk_angular_dispersion": None,
    "curvature": None,
    "fid_sd": None,
    "candidate_family": "baseline",
    "training_seed": None,
    "metric_source": "20260515 diagnostics",
}


ALGORITHM_ORDER = [
    "baseline",
    "source_topk_beta",
    "whitening",
    "router_reg_capacity",
    "target_temp_depth",
    "router_smoothing_source",
    "bridge_tide_kl",
    "router_ema",
    "phase2_other",
    "c0_weighted_mix",
    "c4_sample_bridge",
    "factorial_weighted_bridge",
    "factorial_sample_mix",
]
ALGORITHM_LABELS = {
    "baseline": "Baseline",
    "source_topk_beta": "Source / top-k / beta",
    "whitening": "Channel whitening",
    "router_reg_capacity": "Router reg. / capacity",
    "target_temp_depth": "Target temp. / depth",
    "router_smoothing_source": "Router smoothing / source",
    "bridge_tide_kl": "Bridge / Tide-KL",
    "router_ema": "Router EMA",
    "phase2_other": "C1/C2/C3/C5 variants",
    "c0_weighted_mix": "C0: weighted + mix",
    "c4_sample_bridge": "C4: sample + bridge",
    "factorial_weighted_bridge": "Factorial: weighted + bridge",
    "factorial_sample_mix": "Factorial: sample + mix",
}
ALGORITHM_TICK_LABELS = {
    key: value.replace(" / ", " /\n").replace(": ", ":\n")
    for key, value in ALGORITHM_LABELS.items()
}
ALGORITHM_COLORS = {
    "baseline": "#111111",
    "source_topk_beta": "#3b82f6",
    "whitening": "#06b6d4",
    "router_reg_capacity": "#8b5cf6",
    "target_temp_depth": "#f59e0b",
    "router_smoothing_source": "#84cc16",
    "bridge_tide_kl": "#ec4899",
    "router_ema": "#f97316",
    "phase2_other": "#64748b",
    "c0_weighted_mix": "#2563eb",
    "c4_sample_bridge": "#dc2626",
    "factorial_weighted_bridge": "#0f766e",
    "factorial_sample_mix": "#a16207",
}


def finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_csv_by_target(path: str) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8", newline="") as f:
        return {str(row.get("target")): row for row in csv.DictReader(f) if row.get("target")}


def first_float(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = finite(row.get(key))
        if value is not None:
            return value
    return None


def short_label(text: str, max_len: int = 42) -> str:
    text = text.replace("tide-", "")
    text = text.replace("k16-top2-soft075-dir512", "K16 top2")
    text = text.replace("k16-top2-softv0p75-s128-dir512", "K16 top2")
    text = text.replace("router", "rtr")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def normalize_row(
    row: dict[str, Any],
    batch: str,
    label: str | None = None,
    *,
    period: str = "15/05-13/06",
    protocol: str = "single_best",
) -> dict[str, Any] | None:
    fid128 = first_float(row, ["fid128", "fid128_best", "best_fid128"])
    if fid128 is None:
        return None
    run_name = str(row.get("run_name") or row.get("config_label") or label or "")
    return {
        "batch": batch,
        "period": period,
        "protocol": protocol,
        "label": label or str(row.get("label") or row.get("config_label") or short_label(run_name)),
        "run_name": run_name,
        "fid128": fid128,
        "fid_sd": first_float(row, ["fid_sd", "sample_sd"]),
        "step": first_float(
            row,
            ["fid128_step", "fid128_best_step", "best_fid128_step", "train_step", "step"],
        ),
        "pred_target_var_ratio": first_float(
            row,
            [
                "fm_pred_target_var_ratio",
                "pred_target_var_ratio",
                "pred_target_var_ratio_at_best_fid",
                "pred_target_var_ratio_last",
                "fm_pred_target_ratio",
            ],
        ),
        "curvature": first_float(
            row,
            [
                "flow_curvature",
                "curvature128_at_best_fid",
                "curvature_last",
                "flow_curvature_proxy_mean_last",
                "flow_curvature_last",
            ],
        ),
        "x0_x1_mag_ratio": first_float(
            row,
            [
                "x0_x1_mag_ratio",
                "x0_x1_mag_ratio_last",
                "x0_x1_magnitude_ratio",
                "x0_x1_ratio",
            ],
        ),
        "x0_x1_cosine": first_float(
            row,
            [
                "x0_x1_cosine_mean",
                "geometry_x0_x1_cosine_mean",
            ],
        ),
        "topk_angular_dispersion": first_float(
            row,
            [
                "topk_mu_angular_dispersion",
                "tide_topk_mu_angular_dispersion",
            ],
        ),
        "router_usage_entropy": first_float(
            row,
            [
                "router_usage_entropy",
                "router_usage_entropy_norm",
                "router_usage_entropy_norm_last",
                "router_valid_usage_entropy_normalized",
            ],
        ),
        "valid_loss": first_float(row, ["valid_loss", "training_loss_valid"]),
        "router_valid_loss": first_float(row, ["router_valid_loss"]),
        "router_valid_top1": first_float(row, ["router_valid_top1", "router_valid_top1_agreement"]),
        "candidate_family": row.get("candidate_family") or row.get("candidate_id") or row.get("candidate"),
        "training_seed": row.get("training_seed"),
        "metric_source": str(row.get("metric_source") or row.get("status") or row.get("kaggle_status_latest") or ""),
    }


def wandb_20260606_label(target: str) -> tuple[str, str] | None:
    if "hardtop1" in target:
        return "06/06 source/topk/beta", "hard_top1, x1 frozen, beta(3,1.4)"
    if "sampletopk-t05" in target:
        return "06/06 source/topk/beta", "sample_topk tau0.5, x1 frozen, beta(3,1.4)"
    if "sampletopk" in target:
        return "06/06 source/topk/beta", "sample_topk tau1.0, x1 frozen, beta(3,1.4)"
    if "whiten" not in target:
        return None
    if "mixjoint-gumbel075" in target:
        return "06/06 channel whitening", "whiten mix joint gumbel0.75, beta(3,1.4)"
    if "mixfrozen" in target:
        return "06/06 channel whitening", "whiten mix frozen, beta(3,1.4)"
    if "top1-tau1" in target:
        return "06/06 channel whitening", "whiten x1 frozen, beta(3,1.4), top1"
    if "beta31p4" in target:
        return "06/06 channel whitening", "whiten x1 frozen, beta(3,1.4), top2"
    if "beta31" in target:
        return "06/06 channel whitening", "whiten x1 frozen, beta(3,1), top2"
    return "06/06 channel whitening", short_label(target)


def collect_wandb_20260606_rows() -> list[dict[str, Any]]:
    fid_rows = load_csv_by_target("outputs/kaggle_metrics_20260606/wandb/wandb_fid_summary.csv")
    target_rows = load_csv_by_target("outputs/kaggle_metrics_20260606/wandb/wandb_target_summary.csv")
    out: list[dict[str, Any]] = []
    for target, fid_row in fid_rows.items():
        labelled = wandb_20260606_label(target)
        if labelled is None:
            continue
        fid128 = finite(fid_row.get("fid128_best"))
        if fid128 is None:
            continue
        batch, label = labelled
        target_row = target_rows.get(target, {})
        pred_var = finite(target_row.get("training__fm__pred_variance"))
        target_var = finite(target_row.get("training__fm__target_variance"))
        ratio = pred_var / target_var if pred_var is not None and target_var not in (None, 0.0) else None
        out.append(
            {
                "batch": batch,
                "label": label,
                "run_name": target,
                "fid128": fid128,
                "step": finite(fid_row.get("fid128_best_step")),
                "pred_target_var_ratio": ratio,
                "x0_x1_mag_ratio": None,
                "x0_x1_cosine": None,
                "topk_angular_dispersion": None,
                "router_usage_entropy": finite(target_row.get("training__router__usage_entropy_normalized")),
                "curvature": finite(fid_row.get("flow__curvature_proxy_mean_at_best128")),
                "valid_loss": finite(fid_row.get("training__loss_valid_at_best128")),
                "router_valid_loss": None,
                "router_valid_top1": finite(target_row.get("training__router__top1_agreement_to_gmm_base")),
                "metric_source": "wandb_20260606",
            }
        )
    return out


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [dict(BASELINE)]

    source10 = load_json("reports/gmm_tide_fm_beta_source10_data_available_20260609.json")
    for row in source10.get("runs", []):
        norm = normalize_row(row, "06/06 source/topk/beta")
        if norm:
            rows.append(norm)

    combined = load_json("reports/gmm_tide_router_all_combined_20260611.json")
    for row in combined.get("runs", []):
        norm = normalize_row(row, str(row.get("batch") or "router combined"))
        if norm:
            rows.append(norm)

    deep4 = load_json("reports/gmm_tide_router_deep4_analysis_20260609.json")
    for row in deep4.get("runs", []):
        label = f"d{row.get('router_depth')} {row.get('router_norm_type')} dropout {row.get('router_dropout_rate')}"
        norm = normalize_row(row, "router_deep4", label=label)
        if norm:
            rows.append(norm)

    bridge = load_json("reports/bridge_tidekl_analysis_20260612.json")
    for row in bridge.get("runs", []):
        family = str(row.get("family", ""))
        config_label = str(row.get("config_label", ""))
        if family == "router bridge lambda":
            label = config_label.replace("bridge lambda ", "Bridge ").replace(
                ", FM t uniform/discrete",
                ", FM uniform",
            )
        elif family == "q_GMM(x0_tide) KL":
            label = config_label.replace("tide-KL", "Tide-KL").replace("FM t ", "FM ")
        else:
            label = f"{family}: {config_label}".strip(": ")
        norm = normalize_row(row, "12/06 bridge/tide-KL", label=label)
        if norm:
            rows.append(norm)

    ema = load_json("reports/router_ema_beta35_resume_diagnostics_20260613.json")
    if ema:
        curve = ema.get("eval_curve", [])
        best = min(curve, key=lambda r: finite(r.get("fid128")) or 1e9) if curve else {}
        result = load_json("reports/router_ema_beta35_resume_result_20260613.json")
        pred_target = None
        for row in result.get("runs", []):
            pred_target = first_float(row, ["fm_pred_target_var_ratio"])
            break
        rows.append(
            {
                "batch": "13/06 EMA resume from 150k",
                "label": "EMA beta(3.5,1.3), resume 150k; best 280k",
                "run_name": ema.get("resume", {}).get("source_run_name", "router EMA resume"),
                "fid128": finite(best.get("fid128")),
                "step": finite(best.get("step")),
                "pred_target_var_ratio": pred_target,
                "x0_x1_mag_ratio": None,
                "x0_x1_cosine": None,
                "topk_angular_dispersion": None,
                "router_usage_entropy": None,
                "curvature": finite(best.get("curvature128")),
                "valid_loss": finite(ema.get("train", {}).get("last_valid_loss")),
                "router_valid_loss": None,
                "router_valid_top1": None,
                "metric_source": "kaggle diagnostics",
            }
        )

    rows.extend(collect_wandb_20260606_rows())

    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("fid128") is None:
            continue
        key = (str(row.get("batch")), str(row.get("run_name") or row.get("label")))
        previous = dedup.get(key)
        if previous is None or float(row["fid128"]) < float(previous["fid128"]):
            if previous is not None:
                for field, value in previous.items():
                    if row.get(field) is None and value is not None:
                        row[field] = value
            dedup[key] = row
        elif previous is not None:
            for field, value in row.items():
                if previous.get(field) is None and value is not None:
                    previous[field] = value
    return list(dedup.values())


def nested_metric(metrics: dict[str, Any], key: str, summary: str, index: int = 1) -> float | None:
    value = metrics.get(key, {}).get(summary)
    if isinstance(value, list) and len(value) > index:
        return finite(value[index])
    return None


def gmm_seed_from_name(run_name: str) -> int | None:
    match = re.search(r"gmmseed(\d+)", run_name)
    return int(match.group(1)) if match else None


def collect_late_single_eval_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    phase2 = load_json("reports/gmm_tide_phase2_aware_c0_c5_analysis_20260623.json")
    for row in phase2.get("runs", []):
        candidate = str(row.get("candidate_id") or "C?")
        norm = normalize_row(
            row,
            "22/06 C0-C5",
            label=f"{candidate}: {row.get('label', '')}",
            period="22-29/06",
        )
        if norm:
            norm["metric_source"] = "phase2 C0-C5 analysis"
            rows.append(norm)

    confirm = load_json("reports/gmm_tide_phase2_aware_confirm_c0_c4_200k_analysis_20260629.json")
    for row in confirm.get("runs", []):
        candidate = str(row.get("candidate_id") or "C?")
        norm = normalize_row(
            row,
            "29/06 C0/C4 confirm 200k",
            label=f"{candidate}: {row.get('label', '')}",
            period="22-29/06",
        )
        if norm:
            norm["metric_source"] = "clean C0/C4 confirmation"
            rows.append(norm)

    repeats = load_json("reports/gmm_tide_phase2_variance_repeats4_analysis_20260707.json")
    for row in repeats.get("rows", []):
        norm = normalize_row(
            row,
            "07/07 independent 200k repeats",
            label=str(row.get("candidate_label") or row.get("candidate") or "repeat 200k"),
            period="01-10/07",
        )
        if norm:
            norm["metric_source"] = "independent 200k repeats"
            rows.append(norm)

    resume_seed1 = load_json("reports/gmm_tide_phase2_variance_resume2_420k_results_20260710.json")
    for row in resume_seed1.get("runs", []):
        fid = row.get("fid128", {}).get("best_min", {})
        curvature = row.get("curvature", {}).get("last", {})
        usage = row.get("router_usage_entropy", {}).get("last", {})
        candidate = "C4" if str(row.get("label", "")).startswith("C4") else "C0"
        rows.append(
            {
                "batch": "10/07 resume to 400k",
                "period": "01-10/07",
                "protocol": "single_best",
                "label": f"{candidate} GMM seed1, resumed to 400k",
                "run_name": str(row.get("kernel") or row.get("resume_source") or row.get("label")),
                "fid128": finite(fid.get("value")),
                "fid_sd": None,
                "step": finite(fid.get("step")),
                "pred_target_var_ratio": finite(row.get("pred_target_var_ratio_last")),
                "curvature": finite(curvature.get("value")),
                "x0_x1_mag_ratio": None,
                "x0_x1_cosine": None,
                "topk_angular_dispersion": None,
                "router_usage_entropy": finite(usage.get("value")),
                "valid_loss": finite(row.get("valid_loss", {}).get("last", {}).get("value")),
                "router_valid_loss": None,
                "router_valid_top1": None,
                "candidate_family": candidate,
                "training_seed": 1,
                "metric_source": "resume diagnostics; auxiliary metrics are last logged values",
            }
        )

    resume_seed0 = load_json("reports/gmm_tide_phase2_variance_seed0_resume400_metrics_20260710.json")
    for row in resume_seed0.get("runs", []):
        run_name = str(row.get("run") or "")
        candidate = "C4" if "c4" in run_name.lower() else "C0"
        metrics = row.get("metrics", {})
        pred = nested_metric(metrics, "training/fm/pred_variance", "last")
        target = nested_metric(metrics, "training/fm/target_variance", "last")
        ratio = pred / target if pred is not None and target not in (None, 0.0) else None
        fid_summary = metrics.get("fid/timesteps/128", {}).get("best_min", [])
        rows.append(
            {
                "batch": "10/07 resume to 400k",
                "period": "01-10/07",
                "protocol": "single_best",
                "label": f"{candidate} GMM seed0, resumed to 400k",
                "run_name": run_name,
                "fid128": finite(fid_summary[1]) if len(fid_summary) > 1 else None,
                "fid_sd": None,
                "step": finite(fid_summary[0]) if fid_summary else None,
                "pred_target_var_ratio": ratio,
                "curvature": nested_metric(metrics, "flow/curvature_proxy_mean", "last"),
                "x0_x1_mag_ratio": None,
                "x0_x1_cosine": None,
                "topk_angular_dispersion": None,
                "router_usage_entropy": None,
                "valid_loss": None,
                "router_valid_loss": None,
                "router_valid_top1": None,
                "candidate_family": candidate,
                "training_seed": 0,
                "metric_source": "resume diagnostics; auxiliary metrics are last logged values",
            }
        )

    for report_path in (
        "reports/gmm_tide_factorial_seed01_resume400_retry2_results_20260720.json",
        "reports/gmm_tide_factorial_seed2_resume400_retry2_results_20260720.json",
    ):
        report = load_json(report_path)
        for row in report.get("runs", []):
            run_name = str(row.get("run_name") or "")
            seed = gmm_seed_from_name(run_name)
            if "wbridge" in run_name:
                family = "W+B"
                label = f"weighted + bridge, GMM seed {seed}"
            else:
                family = "S+M"
                label = f"sample_topk + mix, GMM seed {seed}"
            norm = normalize_row(
                row,
                "20/07 factorial 400k",
                label=label,
                period="15-20/07",
            )
            if norm:
                norm["candidate_family"] = family
                norm["training_seed"] = seed
                norm["metric_source"] = "factorial resume 400k"
                rows.append(norm)

    return [row for row in rows if row.get("fid128") is not None]


def collect_repeated_fid_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = load_json("reports/gmm_tide_fid_repeat_audit_complete_20260715.json")
    for row in audit.get("run_summaries", []):
        family = str(row.get("candidate_family") or "C?")
        seed = row.get("training_seed")
        summary = row.get("summary", {})
        curvature = row.get("secondary_summaries", {}).get("flow/curvature_proxy_mean", {})
        rows.append(
            {
                "batch": "15/07 repeated FID",
                "period": "15-20/07",
                "protocol": "repeated_mean",
                "label": f"{family} GMM seed {seed}, repeated-FID mean",
                "run_name": str(row.get("run_name") or ""),
                "fid128": finite(summary.get("mean")),
                "fid_sd": finite(summary.get("sample_std")),
                "step": 400000.0,
                "pred_target_var_ratio": None,
                "curvature": finite(curvature.get("mean")),
                "x0_x1_mag_ratio": None,
                "x0_x1_cosine": None,
                "topk_angular_dispersion": None,
                "router_usage_entropy": None,
                "valid_loss": None,
                "router_valid_loss": None,
                "router_valid_top1": None,
                "candidate_family": family,
                "training_seed": seed,
                "metric_source": "5 matched evaluation seeds; 50048 generations each",
            }
        )

    seed2 = load_json("reports/gmm_tide_fid_confirmation_gmmseed2_result_20260718.json")
    for family, summary in seed2.get("runs", {}).items():
        rows.append(
            {
                "batch": "18/07 repeated FID",
                "period": "15-20/07",
                "protocol": "repeated_mean",
                "label": f"{family} GMM seed 2, repeated-FID mean",
                "run_name": f"tide-fidconfirm-{family.lower()}-gmmseed2-400k",
                "fid128": finite(summary.get("mean")),
                "fid_sd": finite(summary.get("sample_sd")),
                "step": 400000.0,
                "pred_target_var_ratio": None,
                "curvature": None,
                "x0_x1_mag_ratio": None,
                "x0_x1_cosine": None,
                "topk_angular_dispersion": None,
                "router_usage_entropy": None,
                "valid_loss": None,
                "router_valid_loss": None,
                "router_valid_top1": None,
                "candidate_family": family,
                "training_seed": 2,
                "metric_source": "5 matched evaluation seeds; 50048 generations each",
            }
        )
    return [row for row in rows if row.get("fid128") is not None]


def algorithm_group(row: dict[str, Any]) -> str:
    batch = str(row.get("batch") or "")
    family = str(row.get("candidate_family") or "")
    if batch == "baseline":
        return "baseline"
    if family.startswith("C0"):
        return "c0_weighted_mix"
    if family.startswith("C4"):
        return "c4_sample_bridge"
    if family == "W+B":
        return "factorial_weighted_bridge"
    if family == "S+M":
        return "factorial_sample_mix"
    if batch == "06/06 source/topk/beta":
        return "source_topk_beta"
    if batch == "06/06 channel whitening":
        return "whitening"
    if batch in ("router_reg_capacity", "router_deep4"):
        return "router_reg_capacity"
    if batch == "router_temp_depth10":
        return "target_temp_depth"
    if batch == "router_smooth5":
        return "router_smoothing_source"
    if batch == "12/06 bridge/tide-KL":
        return "bridge_tide_kl"
    if batch == "13/06 EMA resume from 150k":
        return "router_ema"
    if batch == "22/06 C0-C5":
        return "phase2_other"
    raise ValueError(f"No algorithm group for row: batch={batch!r}, family={family!r}")


def collect_full_period_rows() -> list[dict[str, Any]]:
    single = collect_rows() + collect_late_single_eval_rows()
    repeated = collect_repeated_fid_rows()
    for row in single:
        row.setdefault("period", "15/05-13/06")
        row.setdefault("protocol", "single_best")
        row.setdefault("fid_sd", None)
        row.setdefault("candidate_family", None)
        row.setdefault("training_seed", None)
    for row in single + repeated:
        row["algorithm_group"] = algorithm_group(row)
    return single + repeated


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch",
        "period",
        "protocol",
        "algorithm_group",
        "label",
        "run_name",
        "fid128",
        "fid_sd",
        "step",
        "pred_target_var_ratio",
        "x0_x1_mag_ratio",
        "x0_x1_cosine",
        "topk_angular_dispersion",
        "router_usage_entropy",
        "curvature",
        "valid_loss",
        "router_valid_loss",
        "router_valid_top1",
        "candidate_family",
        "training_seed",
        "metric_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def color_for(batch: str) -> str:
    palette = {
        "baseline": "#111111",
        "06/06 source/topk/beta": "#8e44ad",
        "06/06 channel whitening": "#1f77b4",
        "router_reg_capacity": "#2e86de",
        "router_deep4": "#16a085",
        "router_temp_depth10": "#f39c12",
        "router_smooth5": "#7f8c8d",
        "12/06 bridge/tide-KL": "#c0392b",
        "13/06 EMA resume from 150k": "#d35400",
    }
    return palette.get(batch, "#4c72b0")


def algorithm_color(group: str) -> str:
    return ALGORITHM_COLORS.get(group, "#64748b")


def spread_positions(center: float, count: int, width: float = 0.58) -> list[float]:
    if count <= 1:
        return [center]
    return [center - width / 2 + width * index / (count - 1) for index in range(count)]


def save_fid_rank(rows: list[dict[str, Any]], out: Path) -> None:
    single = [r for r in rows if r.get("protocol") == "single_best"]
    repeated = sorted(
        [r for r in rows if r.get("protocol") == "repeated_mean"],
        key=lambda r: (int(r.get("training_seed") or 0), str(r.get("candidate_family"))),
    )
    all_values = [float(r["fid128"]) for r in rows]
    y_min = min(all_values) - 0.2
    y_max = max(all_values) + 0.35

    fig, (ax, ax_repeat) = plt.subplots(
        2,
        1,
        figsize=(13.0, 8.8),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for x_index, algorithm in enumerate(ALGORITHM_ORDER):
        group = sorted(
            [r for r in single if r.get("algorithm_group") == algorithm],
            key=lambda r: float(r["fid128"]),
        )
        x_values = spread_positions(float(x_index), len(group))
        ax.scatter(
            x_values,
            [float(r["fid128"]) for r in group],
            s=32,
            alpha=0.78,
            color=algorithm_color(algorithm),
            edgecolor="white",
            linewidth=0.35,
        )
        ax.text(x_index, y_max - 0.1, f"n={len(group)}", ha="center", va="top", fontsize=8)
    ax.axhline(BASELINE["fid128"], color="#111111", linestyle="--", linewidth=1.3, label="baseline 6.969")
    ax.set_xticks(range(len(ALGORITHM_ORDER)))
    ax.set_xticklabels([ALGORITHM_TICK_LABELS[group] for group in ALGORITHM_ORDER], fontsize=7)
    ax.set_ylabel("Best single-eval FID128 (lower is better)")
    ax.set_title(f"All single-eval results grouped by algorithm (n={len(single)})")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=8, loc="upper right")

    repeat_labels: list[str] = []
    for index, row in enumerate(repeated):
        family = str(row.get("candidate_family") or "?")
        seed = row.get("training_seed")
        color = "#2563eb" if family == "C0" else "#dc2626"
        ax_repeat.errorbar(
            index,
            float(row["fid128"]),
            yerr=float(row.get("fid_sd") or 0.0),
            fmt="D",
            markersize=5.5,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
        )
        repeat_labels.append(f"{family}\ns{seed}")
    ax_repeat.axhline(BASELINE["fid128"], color="#111111", linestyle="--", linewidth=1.3)
    ax_repeat.set_ylim(y_min, y_max)
    ax_repeat.set_xticks(range(len(repeated)))
    ax_repeat.set_xticklabels(repeat_labels, fontsize=8)
    ax_repeat.set_title(f"Repeated-FID mean +/- SD (n={len(repeated)})")
    ax_repeat.set_xlabel("Candidate / GMM seed")
    ax_repeat.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_var_ratio(rows: list[dict[str, Any]], out: Path) -> None:
    eligible = [r for r in rows if r.get("protocol") == "single_best"]
    plotted = [r for r in eligible if r.get("pred_target_var_ratio") is not None]
    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    for algorithm in ALGORITHM_ORDER:
        group = [r for r in plotted if r.get("algorithm_group") == algorithm]
        if not group:
            continue
        ax.scatter(
            [float(r["pred_target_var_ratio"]) for r in group],
            [float(r["fid128"]) for r in group],
            s=40,
            alpha=0.75,
            color=algorithm_color(algorithm),
            edgecolor="white",
            linewidth=0.35,
            label=f"{ALGORITHM_LABELS[algorithm]} (n={len(group)})",
        )
    ax.axvline(1.0, color="#111111", linestyle="--", linewidth=1.2, label="matched variance = 1")
    ax.axhline(BASELINE["fid128"], color="#555555", linestyle=":", linewidth=1.0, label="baseline FID128")
    ax.set_xlabel("Predicted / target velocity variance")
    ax.set_ylabel("Best single-eval FID128 (lower is better)")
    ax.set_title(f"All available variance diagnostics (n={len(plotted)}/{len(eligible)} single-eval rows)")
    ax.set_xlim(0.60, 1.02)
    ax.grid(alpha=0.16)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_curvature_scatter(rows: list[dict[str, Any]], out: Path) -> None:
    plotted = [r for r in rows if r.get("curvature") is not None and r.get("fid128") is not None]
    normal = [r for r in plotted if float(r["curvature"]) <= 0.03]
    outliers = [r for r in plotted if float(r["curvature"]) > 0.03]
    fig, (ax, ax_outlier) = plt.subplots(
        1,
        2,
        figsize=(10.4, 6.4),
        gridspec_kw={"width_ratios": [4.6, 1.0]},
        sharey=True,
    )
    for algorithm in ALGORITHM_ORDER:
        group = [
            r
            for r in normal
            if r.get("protocol") == "single_best" and r.get("algorithm_group") == algorithm
        ]
        if not group:
            continue
        ax.scatter(
            [float(r["curvature"]) for r in group],
            [float(r["fid128"]) for r in group],
            s=40,
            alpha=0.72,
            color=algorithm_color(algorithm),
            edgecolor="white",
            linewidth=0.35,
            label=f"{ALGORITHM_LABELS[algorithm]} (n={len(group)})",
        )
    repeated = [r for r in normal if r.get("protocol") == "repeated_mean"]
    for family in ("C0", "C4"):
        group = [r for r in repeated if str(r.get("candidate_family")) == family]
        if not group:
            continue
        color = "#2563eb" if family == "C0" else "#dc2626"
        ax.errorbar(
            [float(row["curvature"]) for row in group],
            [float(row["fid128"]) for row in group],
            yerr=[float(row.get("fid_sd") or 0.0) for row in group],
            fmt="D",
            markersize=5.5,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            label=f"repeated mean {family} (n={len(group)})",
        )
    for current_ax in (ax, ax_outlier):
        current_ax.axhline(BASELINE["fid128"], color="#111111", linestyle="--", linewidth=1.2)
        current_ax.grid(alpha=0.16)
    for row in outliers:
        ax_outlier.scatter(
            float(row["curvature"]),
            float(row["fid128"]),
            s=48,
            alpha=0.8,
            color=algorithm_color(str(row.get("algorithm_group"))),
            edgecolor="white",
            linewidth=0.35,
        )
        ax_outlier.annotate(
            short_label(str(row.get("label")), 24),
            (float(row["curvature"]), float(row["fid128"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax.plot([], [], color="#111111", linestyle="--", linewidth=1.2, label="baseline FID128")
    ax.set_xlabel("Flow curvature proxy at eval")
    ax.set_ylabel("FID128 (single best or repeated mean)")
    ax_outlier.set_xlabel("Outlier")
    ax.set_title(f"Main range (n={len(normal)})")
    ax_outlier.set_title(f"> 0.03 (n={len(outliers)})")
    fig.suptitle(f"All available curvature diagnostics (n={len(plotted)}/{len(rows)} rows)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.62, -0.16))
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_usage_entropy_scatter(rows: list[dict[str, Any]], out: Path) -> None:
    plotted = [r for r in rows if r.get("router_usage_entropy") is not None and r.get("fid128") is not None]
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    batches = sorted({str(r["batch"]) for r in plotted})
    for batch in batches:
        group = [r for r in plotted if str(r["batch"]) == batch]
        ax.scatter(
            [float(r["router_usage_entropy"]) for r in group],
            [float(r["fid128"]) for r in group],
            s=44,
            alpha=0.85,
            color=color_for(batch),
            label=batch,
        )
    ax.axhline(BASELINE["fid128"], color="#111111", linestyle="--", linewidth=1.2, label="baseline FID128")
    ax.axvline(BASELINE["router_usage_entropy"], color="#555555", linestyle=":", linewidth=1.0, label="baseline usage H")
    ax.set_xlabel("Normalized hard-usage entropy in [0, 1] (view zoomed to 0.82-0.99)")
    ax.set_ylabel("Best FID128 (lower is better)")
    ax.set_title("Balanced routing alone did not explain FID")
    ax.set_xlim(0.82, 0.99)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_source_scale_ratio(rows: list[dict[str, Any]], out: Path, top_n: int = 18) -> None:
    available = [r for r in rows if r.get("x0_x1_mag_ratio") is not None]
    required = [
        r
        for r in available
        if str(r.get("batch")) in ("baseline", "12/06 bridge/tide-KL")
    ]
    required_keys = {(str(r.get("batch")), str(r.get("run_name"))) for r in required}
    extras = [
        r
        for r in sorted(available, key=lambda r: float(r["fid128"]))
        if (str(r.get("batch")), str(r.get("run_name"))) not in required_keys
    ]
    ranked = sorted(required + extras[: max(top_n - len(required), 0)], key=lambda r: float(r["fid128"]))
    fig, ax = plt.subplots(figsize=(10, 6.8))
    labels = [short_label(str(r["label"]), 50) for r in ranked]
    values = [float(r["x0_x1_mag_ratio"]) for r in ranked]
    colors = [color_for(str(r["batch"])) for r in ranked]
    ypos = list(range(len(ranked)))
    ax.barh(ypos, values, color=colors)
    ax.axvline(1.0, color="#111111", linestyle="--", linewidth=1.2, label="matched x0/x1 magnitude")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("x0 magnitude / x1 magnitude")
    ax.set_title("Source scale, including every 12/06 bridge and Tide-KL run")
    for y, value in zip(ypos, values):
        ax.text(value + 0.003, y, f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.94, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    rows = collect_full_period_rows()
    out_dir = Path("pdf/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: float(r["fid128"]))
    single_count = sum(row.get("protocol") == "single_best" for row in rows_sorted)
    repeated_count = sum(row.get("protocol") == "repeated_mean" for row in rows_sorted)
    coverage = {
        "all_rows": len(rows_sorted),
        "single_best_rows": single_count,
        "repeated_fid_mean_rows": repeated_count,
        "pred_target_var_ratio_rows": sum(row.get("pred_target_var_ratio") is not None for row in rows_sorted),
        "curvature_rows": sum(row.get("curvature") is not None for row in rows_sorted),
        "router_usage_entropy_rows": sum(row.get("router_usage_entropy") is not None for row in rows_sorted),
        "x0_x1_mag_ratio_rows": sum(row.get("x0_x1_mag_ratio") is not None for row in rows_sorted),
        "period_counts_single_best": {
            period: sum(
                row.get("protocol") == "single_best" and row.get("period") == period
                for row in rows_sorted
            )
            for period in sorted({str(row.get("period")) for row in rows_sorted})
        },
        "algorithm_counts_single_best": {
            algorithm: sum(
                row.get("protocol") == "single_best" and row.get("algorithm_group") == algorithm
                for row in rows_sorted
            )
            for algorithm in ALGORITHM_ORDER
        },
    }
    write_csv(rows_sorted, out_dir / "weekly_plot_data.csv")
    (out_dir / "weekly_plot_data.json").write_text(
        json.dumps({"baseline": BASELINE, "coverage": coverage, "runs": rows_sorted}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    save_fid_rank(rows_sorted, out_dir / "weekly_fid128_rank.png")
    save_var_ratio(rows_sorted, out_dir / "weekly_variance_ratio.png")
    save_curvature_scatter(rows_sorted, out_dir / "weekly_curvature_vs_fid.png")
    save_usage_entropy_scatter(rows_sorted, out_dir / "weekly_router_usage_vs_fid.png")
    save_source_scale_ratio(rows_sorted, out_dir / "weekly_source_scale_ratio.png")
    print(
        f"Wrote {len(rows_sorted)} rows ({single_count} single-best, "
        f"{repeated_count} repeated means) and 5 plots to {out_dir}"
    )


if __name__ == "__main__":
    main()
