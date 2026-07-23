from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = ROOT / "pdf" / "figures"
BASELINE_FID128 = 6.969


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def plot_phase2_confirmation(confirm: dict[str, Any]) -> None:
    colors = {"C0-confirm": "#0072B2", "C4-confirm": "#D55E00"}
    labels = {"C0-confirm": "C0: weighted + mix", "C4-confirm": "C4: sample-topk + bridge"}
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    for run in sorted(confirm["runs"], key=lambda row: row["candidate_id"]):
        candidate = run["candidate_id"]
        steps = [row["step"] / 1000 for row in run["evals"]]
        fids = [row["fid128"] for row in run["evals"]]
        ax.plot(steps, fids, marker="o", linewidth=2, color=colors[candidate], label=labels[candidate])
    ax.axhline(BASELINE_FID128, linestyle="--", linewidth=1.5, color="#333333", label="baseline lịch sử 6.969")
    ax.set_title("Xác nhận sạch C0/C4 trong 200k bước")
    ax.set_xlabel("Bước huấn luyện (nghìn)")
    ax.set_ylabel("FID128 (thấp hơn tốt hơn)")
    ax.set_xticks(range(20, 201, 20))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "progress_phase2_confirmation.png", bbox_inches="tight")
    plt.close(fig)


def repeated_fid_rows(
    repeat: dict[str, Any], seed2: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for row in repeat["run_summaries"]:
        rows.append(
            {
                "gmm_seed": int(row["training_seed"]),
                "family": row["candidate_family"],
                "mean": float(row["summary"]["mean"]),
                "sd": float(row["summary"]["sample_std"]),
                "ci95_low": float(row["summary"]["ci95_low"]),
                "ci95_high": float(row["summary"]["ci95_high"]),
                "eval_seeds": int(row["summary"]["n"]),
            }
        )
    for family, values in seed2["runs"].items():
        rows.append(
            {
                "gmm_seed": 2,
                "family": family,
                "mean": float(values["mean"]),
                "sd": float(values["sample_sd"]),
                "ci95_low": None,
                "ci95_high": None,
                "eval_seeds": 5,
            }
        )

    paired: list[dict[str, Any]] = []
    for row in repeat["paired_by_training_seed"]:
        delta = row["delta_c4_minus_c0"]
        paired.append(
            {
                "gmm_seed": int(row["training_seed"]),
                "mean": float(delta["mean"]),
                "ci95_low": float(delta["ci95_low"]),
                "ci95_high": float(delta["ci95_high"]),
            }
        )
    seed2_delta = seed2["paired_c4_minus_c0"]
    paired.append(
        {
            "gmm_seed": 2,
            "mean": float(seed2_delta["mean"]),
            "ci95_low": float(seed2_delta["ci95_low"]),
            "ci95_high": float(seed2_delta["ci95_high"]),
        }
    )
    return sorted(rows, key=lambda row: (row["gmm_seed"], row["family"])), sorted(
        paired, key=lambda row: row["gmm_seed"]
    )


def plot_repeated_fid(rows: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    colors = {"C0": "#0072B2", "C4": "#D55E00"}
    markers = {"C0": "o", "C4": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    for family in ("C0", "C4"):
        subset = [row for row in rows if row["family"] == family]
        axes[0].errorbar(
            [row["gmm_seed"] for row in subset],
            [row["mean"] for row in subset],
            yerr=[row["sd"] for row in subset],
            marker=markers[family],
            capsize=4,
            linewidth=2,
            color=colors[family],
            label=f"{family}: mean +/- SD eval seed",
        )
    axes[0].axhline(BASELINE_FID128, linestyle="--", color="#333333", linewidth=1.3, label="baseline 6.969")
    axes[0].set_title("Repeated-FID tại checkpoint 400k")
    axes[0].set_xlabel("GMM randomization seed")
    axes[0].set_ylabel("FID128")
    axes[0].set_xticks([0, 1, 2])
    axes[0].legend()

    x = [row["gmm_seed"] for row in paired]
    y = [row["mean"] for row in paired]
    low = [row["mean"] - row["ci95_low"] for row in paired]
    high = [row["ci95_high"] - row["mean"] for row in paired]
    bar_colors = ["#009E73" if value < 0 else "#CC79A7" for value in y]
    axes[1].bar(x, y, color=bar_colors, width=0.62)
    axes[1].errorbar(x, y, yerr=[low, high], fmt="none", color="#222222", capsize=4)
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_title(r"Hiệu ứng ghép cặp C4 $-$ C0")
    axes[1].set_xlabel("GMM randomization seed")
    axes[1].set_ylabel(r"$\Delta$ FID128")
    axes[1].set_xticks([0, 1, 2])
    axes[1].text(0.03, 0.04, "Âm: C4 tốt hơn", transform=axes[1].transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "progress_repeated_fid_gmm_seeds.png", bbox_inches="tight")
    plt.close(fig)


def plot_factorial(factorial: dict[str, Any]) -> None:
    families = factorial["families"]
    labels = ["W+B", "S+M"]
    best = [families["weighted_bridge"]["best_fid128_mean"], families["sample_topk_mix"]["best_fid128_mean"]]
    best_sd = [
        families["weighted_bridge"]["best_fid128_sample_sd"],
        families["sample_topk_mix"]["best_fid128_sample_sd"],
    ]
    last = [families["weighted_bridge"]["last_fid128_mean"], families["sample_topk_mix"]["last_fid128_mean"]]
    last_sd = [
        families["weighted_bridge"]["last_fid128_sample_sd"],
        families["sample_topk_mix"]["last_fid128_sample_sd"],
    ]
    drift = [families["weighted_bridge"]["last_minus_best_mean"], families["sample_topk_mix"]["last_minus_best_mean"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    x = [0, 1]
    width = 0.34
    axes[0].bar([value - width / 2 for value in x], best, width, yerr=best_sd, capsize=4, label="best 220k-400k", color="#56B4E9")
    axes[0].bar([value + width / 2 for value in x], last, width, yerr=last_sd, capsize=4, label="tại 400k", color="#E69F00")
    axes[0].axhline(BASELINE_FID128, linestyle="--", color="#333333", linewidth=1.3, label="baseline 6.969")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("FID128, mean +/- SD qua 3 GMM seed")
    axes[0].set_title("Factorial screening ở 400k")
    axes[0].legend()

    axes[1].bar(labels, drift, color=["#CC79A7", "#009E73"])
    axes[1].axhline(0.0, color="#222222", linewidth=1)
    axes[1].set_ylabel("FID(400k) - best FID")
    axes[1].set_title("Mức suy giảm sau điểm tốt nhất")
    for index, value in enumerate(drift):
        axes[1].text(index, value + 0.006, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "progress_factorial_screening.png", bbox_inches="tight")
    plt.close(fig)


def load_geometry() -> dict[str, dict[str, Any]]:
    root = ROOT / "outputs" / "kaggle_jobs" / "gmm_tide_router_geometry_audit_retry_path_20260719"
    result: dict[str, dict[str, Any]] = {}
    for path in root.glob("**/audit_summary.json"):
        payload = load_json(path)
        family = "C4" if payload["config"]["source_mode"] == "sample_topk" else "C0"
        result[family] = payload
    if set(result) != {"C0", "C4"}:
        raise RuntimeError(f"Expected C0/C4 geometry summaries, got {sorted(result)}")
    return result


def scalar(summary: dict[str, Any], key: str) -> float:
    value = summary[key]
    if isinstance(value, dict):
        for candidate in ("mean", "value", "last"):
            if candidate in value:
                value = value[candidate]
                break
    return float(value)


def plot_geometry(geometry: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for family, payload in geometry.items():
        summary = payload["summary"]
        values[family] = {
            "top1_x0": scalar(summary, "router/bridge_l0p0/top1_agreement"),
            "top1_mid": scalar(summary, "router/bridge_l0p5/top1_agreement"),
            "top1_x1": scalar(summary, "router/bridge_l1p0/top1_agreement"),
            "kl_x0": scalar(summary, "router/bridge_l0p0/kl_gmm_to_phi"),
            "kl_mid": scalar(summary, "router/bridge_l0p5/kl_gmm_to_phi"),
            "kl_x1": scalar(summary, "router/bridge_l1p0/kl_gmm_to_phi"),
            "angular_dispersion": scalar(summary, "tide/topk_mu_angular_dispersion"),
            "pair_cosine": scalar(summary, "tide/topk_mu_pair_cosine_mean"),
            "usage_entropy": scalar(summary, "router/usage_entropy_normalized"),
        }

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    labels = [r"$x_0$", "midpoint", r"$x_1$"]
    x = [0, 1, 2]
    width = 0.34
    colors = {"C0": "#0072B2", "C4": "#D55E00"}
    for offset, family in ((-width / 2, "C0"), (width / 2, "C4")):
        axes[0].bar(
            [value + offset for value in x],
            [values[family]["top1_x0"], values[family]["top1_mid"], values[family]["top1_x1"]],
            width,
            label=family,
            color=colors[family],
        )
        axes[1].bar(
            [value + offset for value in x],
            [values[family]["kl_x0"], values[family]["kl_mid"], values[family]["kl_x1"]],
            width,
            label=family,
            color=colors[family],
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.75, 1.01)
    axes[0].set_ylabel("Top-1 agreement")
    axes[0].set_title("Router agreement theo miền input")
    axes[0].legend()
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel(r"$D_{KL}(q_{GMM}\|q_\phi)$")
    axes[1].set_title("Sai lệch distill theo miền input")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "progress_router_geometry_audit.png", bbox_inches="tight")
    plt.close(fig)
    return values


def report_date(path: Path) -> str:
    match = re.search(r"(2026\d{4})", path.name)
    return match.group(1) if match else ""


def classify_run(kernel_id: str) -> str:
    if "output-quota-smoke" in kernel_id:
        return "output_quota_smoke"
    if "audit-" in kernel_id and "router-geometry" in kernel_id:
        return "router_geometry_audit"
    if "fidrep-factorial" in kernel_id:
        return "factorial_repeated_fid"
    if "factorial-" in kernel_id:
        return "factorial_training"
    if "fidconfirm" in kernel_id:
        return "gmm_seed2_repeated_fid"
    if "confirm-c" in kernel_id and "gmmseed2" in kernel_id:
        return "gmm_seed2_training"
    if "fidrep-c" in kernel_id:
        return "c0_c4_repeated_fid"
    if "var-c" in kernel_id and ("resume" in kernel_id or "recover" in kernel_id):
        return "c0_c4_resume"
    if "var-c" in kernel_id and "repeat" in kernel_id:
        return "c0_c4_200k_repeats"
    if "long240" in kernel_id:
        return "c4_long240"
    if "ars-confirm" in kernel_id:
        return "c0_c4_clean_confirm"
    if "ars-c" in kernel_id:
        return "phase2_c0_c5_exploration"
    return "other"


def evidence_status(kernel_id: str, source_report: str) -> str:
    if "phase2_aware_c0_c5" in source_report:
        return "partial_metrics_timeout"
    if "phase2_aware_confirm" in source_report or "output_quota_smoke" in source_report:
        return "complete_metrics"
    if "long240" in source_report:
        return "error_no_model_metrics"
    if "variance_repeats4" in source_report or "variance_seed0_resume400" in source_report:
        return "complete_metrics"
    if "variance_resume2" in source_report:
        return "partial_metrics_timeout"
    if "c4_seed1_recover400" in source_report:
        return "complete_metrics"
    if "fid_repeat4_submit" in source_report:
        return "complete_metrics" if kernel_id.startswith("bangchi/") else "error_cross_account"
    if "fid_repeat_same_owner_retry2" in source_report or "fid_repeat_c4_s1_recovered" in source_report:
        return "complete_metrics"
    if "confirm_gmmseed2" in source_report or "fidconfirm_gmmseed2" in source_report:
        return "complete_metrics"
    if "factorial_seed01_200k" in source_report or "factorial_seed2_submit" in source_report:
        return "complete_metrics"
    if "factorial_seed01_resume400_retry2" in source_report or "factorial_seed2_resume400_retry2" in source_report:
        return "complete_metrics"
    if "factorial_seed01_resume400" in source_report or "factorial_seed2_resume400_submit" in source_report:
        return "superseded_resume_error"
    if "router_geometry_audit_retry_path" in source_report:
        return "complete_metrics"
    if "router_geometry_audit" in source_report:
        return "error_then_retried"
    if "factorial_wb_sm_fidrepeat6" in source_report:
        return "error_missing_checkpoint_in_relay"
    return "submitted_status_only"


def is_dry_run_report(path: Path, payload: dict[str, Any]) -> bool:
    name = path.name.lower()
    artifact_mode = str(payload.get("artifact_mode", "")).lower()
    return "dry_run" in name or "dryrun" in name or "dry" in artifact_mode or "sensitive_audit" in name


def collect_submission_attempts() -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for path in sorted(REPORTS.glob("*.json")):
        date = report_date(path)
        if date < "20260622":
            continue
        try:
            payload = load_json(path)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or is_dry_run_report(path, payload):
            continue
        submitted = payload.get("submitted", [])
        if not isinstance(submitted, list):
            continue
        submitted_at = str(payload.get("generated_at_utc") or date)
        for row_index, row in enumerate(submitted):
            if not isinstance(row, dict):
                continue
            kernel_id = row.get("kernel_id") or row.get("kernel")
            if not kernel_id:
                continue
            attempts.append(
                {
                    "attempt_id": f"{path.name}#{row_index}",
                    "kernel_id": str(kernel_id),
                    "owner": str(row.get("owner") or str(kernel_id).split("/", 1)[0]),
                    "run_name": str(row.get("run_name") or ""),
                    "submitted_on": date,
                    "submitted_at_utc": submitted_at,
                    "experiment_family": classify_run(str(kernel_id)),
                    "submission_status": str(row.get("kernel_status") or row.get("status") or ""),
                    "evidence_status": evidence_status(str(kernel_id), path.name),
                    "source_submit_report": path.name,
                    "url": str(row.get("url") or f"https://www.kaggle.com/code/{kernel_id}"),
                }
            )
    attempts.sort(key=lambda row: (row["submitted_at_utc"], row["attempt_id"]))
    seen: Counter[str] = Counter()
    for row in attempts:
        seen[row["kernel_id"]] += 1
        row["attempt_number_for_slug"] = seen[row["kernel_id"]]
    return attempts


def collect_kernel_inventory(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    for attempt in attempts:
        kernel_id = attempt["kernel_id"]
        latest[kernel_id] = dict(attempt)
        histories.setdefault(kernel_id, []).append(attempt)

    inventory: list[dict[str, Any]] = []
    for kernel_id, row in latest.items():
        history = histories[kernel_id]
        row.update(
            {
                "attempt_count_for_slug": len(history),
                "attempt_evidence_history": [attempt["evidence_status"] for attempt in history],
                "attempt_report_history": [attempt["source_submit_report"] for attempt in history],
            }
        )
        inventory.append(row)
    return sorted(inventory, key=lambda row: (row["submitted_at_utc"], row["kernel_id"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_inventory(rows: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    json_path = REPORTS / "progress_run_inventory_20260721.json"
    csv_path = REPORTS / "progress_run_inventory_20260721.csv"
    attempts_csv_path = REPORTS / "progress_submission_attempts_20260721.csv"
    md_path = REPORTS / "progress_run_inventory_20260721.md"
    evidence_counts = Counter(row["evidence_status"] for row in rows)
    attempt_evidence_counts = Counter(row["evidence_status"] for row in attempts)
    family_counts = Counter(row["experiment_family"] for row in rows)
    distinct_run_names = {row["run_name"] for row in attempts if row["run_name"]}
    source_reports = sorted({row["source_submit_report"] for row in attempts})
    duplicate_slugs = [row for row in rows if row["attempt_count_for_slug"] > 1]
    payload = {
        "scope": "Kaggle submissions from 2026-06-22 through 2026-07-20",
        "count": len(rows),
        "submission_event_count": len(attempts),
        "distinct_kernel_slug_count": len(rows),
        "distinct_run_name_count": len(distinct_run_names),
        "submit_report_count": len(source_reports),
        "evidence_status_counts": dict(sorted(evidence_counts.items())),
        "latest_slug_evidence_status_counts": dict(sorted(evidence_counts.items())),
        "attempt_evidence_status_counts": dict(sorted(attempt_evidence_counts.items())),
        "experiment_family_counts": dict(sorted(family_counts.items())),
        "notes": [
            "Submission events count every successful push recorded in a non-dry-run report.",
            "Distinct kernel slugs collapse pushes that overwrite the same Kaggle slug; latest state is selected by generated_at_utc.",
            "Distinct run names are labels, not independent scientific replicates; several labels are intentionally retried.",
            "Historical runs through 2026-06-13 remain documented in pdf/main.tex and are not duplicated here.",
        ],
        "source_submit_reports": source_reports,
        "runs": rows,
        "submission_attempts": attempts,
    }
    save_json(json_path, payload)
    write_csv(csv_path, rows)
    write_csv(attempts_csv_path, attempts)

    statuses = sorted(set(evidence_counts) | set(attempt_evidence_counts))
    lines = [
        "# Progress Run Inventory 2026-07-21",
        "",
        f"- Submission events: **{len(attempts)}**",
        f"- Distinct kernel slugs: **{len(rows)}**",
        f"- Distinct run-name labels: **{len(distinct_run_names)}**",
        f"- Non-dry-run submit reports scanned: **{len(source_reports)}**",
        "- Scope: submissions from 2026-06-22 through 2026-07-20.",
        "- A retry that reuses a slug is one additional submission event but not a new distinct slug.",
        "",
        "## Evidence Status",
        "",
        "| Status | Submission events | Latest distinct slugs |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| `{key}` | {attempt_evidence_counts.get(key, 0)} | {evidence_counts.get(key, 0)} |" for key in statuses
    )
    lines.extend(
        [
            "",
            "## Reused Slugs",
            "",
            "These six slugs were pushed twice. The first resume attempt failed and the later retry produced complete metrics.",
            "",
            "| Kernel | Attempts | Evidence history | Submit reports |",
            "|---|---:|---|---|",
        ]
    )
    for row in duplicate_slugs:
        status_history = " -> ".join(f"`{value}`" for value in row["attempt_evidence_history"])
        report_history = "<br>".join(f"`{value}`" for value in row["attempt_report_history"])
        lines.append(f"| `{row['kernel_id']}` | {row['attempt_count_for_slug']} | {status_history} | {report_history} |")
    lines.extend(
        [
            "",
            "## Latest State By Slug",
            "",
            "| # | Date | Family | Evidence | Attempts | Kernel |",
            "|---:|---|---|---|---:|---|",
        ]
    )
    for index, row in enumerate(rows, 1):
        lines.append(
            f"| {index} | {row['submitted_on']} | `{row['experiment_family']}` | "
            f"`{row['evidence_status']}` | {row['attempt_count_for_slug']} | `{row['kernel_id']}` |"
        )
    lines.extend(
        [
            "",
            "## Submission Events",
            "",
            "| # | Timestamp | Family | Evidence | Attempt | Kernel | Source report |",
            "|---:|---|---|---|---:|---|---|",
        ]
    )
    for index, row in enumerate(attempts, 1):
        lines.append(
            f"| {index} | `{row['submitted_at_utc']}` | `{row['experiment_family']}` | "
            f"`{row['evidence_status']}` | {row['attempt_number_for_slug']} | `{row['kernel_id']}` | "
            f"`{row['source_submit_report']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    confirm = load_json(REPORTS / "gmm_tide_phase2_aware_confirm_c0_c4_200k_analysis_20260629.json")
    repeat = load_json(REPORTS / "gmm_tide_fid_repeat_audit_complete_20260715.json")
    seed2 = load_json(REPORTS / "gmm_tide_fid_confirmation_gmmseed2_result_20260718.json")
    factorial = load_json(REPORTS / "gmm_tide_factorial_resume400_retry2_analysis_20260720.json")

    plot_phase2_confirmation(confirm)
    repeated_rows, paired_rows = repeated_fid_rows(repeat, seed2)
    plot_repeated_fid(repeated_rows, paired_rows)
    plot_factorial(factorial)
    geometry_values = plot_geometry(load_geometry())
    submission_attempts = collect_submission_attempts()
    inventory = write_inventory(collect_kernel_inventory(submission_attempts), submission_attempts)

    evidence = {
        "generated_from": [
            "reports/gmm_tide_phase2_aware_confirm_c0_c4_200k_analysis_20260629.json",
            "reports/gmm_tide_fid_repeat_audit_complete_20260715.json",
            "reports/gmm_tide_fid_confirmation_gmmseed2_result_20260718.json",
            "reports/gmm_tide_factorial_resume400_retry2_analysis_20260720.json",
            "outputs/kaggle_jobs/gmm_tide_router_geometry_audit_retry_path_20260719/**/audit_summary.json",
        ],
        "historical_baseline_fid128": BASELINE_FID128,
        "repeated_fid": repeated_rows,
        "paired_c4_minus_c0": paired_rows,
        "cross_gmm_seed": seed2["cross_gmm_seed_summary"],
        "factorial_families": factorial["families"],
        "factorial_paired_deltas": factorial["paired_deltas_weighted_minus_sample"],
        "router_geometry_seed2": geometry_values,
        "kernel_inventory_summary": {
            "count": inventory["count"],
            "submission_event_count": inventory["submission_event_count"],
            "distinct_kernel_slug_count": inventory["distinct_kernel_slug_count"],
            "distinct_run_name_count": inventory["distinct_run_name_count"],
            "evidence_status_counts": inventory["evidence_status_counts"],
            "attempt_evidence_status_counts": inventory["attempt_evidence_status_counts"],
            "experiment_family_counts": inventory["experiment_family_counts"],
        },
    }
    save_json(REPORTS / "progress_report_evidence_20260721.json", evidence)
    print(
        f"Wrote {inventory['submission_event_count']} submissions / "
        f"{inventory['distinct_kernel_slug_count']} slugs and "
        f"{len(list(FIGURES.glob('progress_*.png')))} figures"
    )


if __name__ == "__main__":
    main()
