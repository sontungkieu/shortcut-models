from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CURRENT_METRICS = Path(
    "outputs/kaggle_jobs/gmm_tide_moe2_shift_only_raw_20260726/"
    "johnntlhudson__tide-moe2-shiftmean-raw-s0-k16-top2-soft075-john/"
    "output/gmm_tide_fm/tide-moe2-shiftmean-raw-s0-k16-top2-soft075/"
    "diagnostics/train_metrics.csv"
)
DEFAULT_HISTORICAL_MANIFEST = Path(
    "/home/tung/shortcut-models-document/pdf/latent_population_report/"
    "data/aggregated_selection_manifest.json"
)
DEFAULT_OUTPUT = Path(
    "reports/figures/gmm_tide_shift_only_fid128_vs_historical.png"
)

HISTORICAL_RUNS = {
    "best_wbridge_seed0": {
        "id": "B1",
        "name": "W+B s0",
        "protocol": "best single ≤400k",
    },
    "best_samplemix_seed2": {
        "id": "B2",
        "name": "S+M s2",
        "protocol": "best single ≤400k",
    },
    "best_c4_sampletopk_bridge_seed1": {
        "id": "B3",
        "name": "C4 S+B s1",
        "protocol": "5-seed mean @400k",
    },
    "fid8_c0_control_200k": {
        "id": "C1",
        "name": "C0 W+M",
        "protocol": "best single @200k",
    },
    "fid8_beta31p4_x1frozen": {
        "id": "C2",
        "name": "B(3,1.4) x1-frozen",
        "protocol": "best single @220k",
    },
    "fid8_fmretune_s1": {
        "id": "C3",
        "name": "FM-retune S1",
        "protocol": "best single @320k",
    },
}


def load_current_fid128(path: Path) -> list[tuple[int, float]]:
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != "eval":
                continue
            if row.get("metric") != "fid/timesteps/128":
                continue
            values[int(row["step"])] = float(row["value"])
    if not values:
        raise ValueError(f"No complete eval FID128 rows found in {path}")
    return sorted(values.items())


def load_historical_fid128(path: Path) -> list[dict[str, object]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in manifest}
    missing = sorted(set(HISTORICAL_RUNS) - set(by_label))
    if missing:
        raise ValueError(f"Historical manifest is missing labels: {missing}")
    rows = []
    for label, display in HISTORICAL_RUNS.items():
        source = by_label[label]
        rows.append(
            {
                **display,
                "label": label,
                "fid128": float(source["fid128"]),
                "fid_group": source["fid_group"],
                "fid_statistic": source["fid_statistic"],
            }
        )
    return rows


def render(
    current: list[tuple[int, float]],
    historical: list[dict[str, object]],
    output: Path,
) -> None:
    current_step, current_fid = current[-1]
    historical_fids = [float(row["fid128"]) for row in historical]
    c1 = next(row for row in historical if row["id"] == "C1")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_curve, ax_compare) = plt.subplots(
        1,
        2,
        figsize=(14.4, 6.4),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )

    steps_k = [step / 1000 for step, _ in current]
    curve = [fid for _, fid in current]
    current_color = "#D55E00"
    best_color = "#009E73"
    control_color = "#4C78A8"

    ax_curve.axhspan(
        min(historical_fids),
        max(historical_fids),
        color="#A0A0A0",
        alpha=0.12,
        label="Historical FID range: 6.88–8.83",
        zorder=0,
    )
    ax_curve.axhline(
        float(c1["fid128"]),
        color=control_color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        label=f"C1 matched checkpoint @200k: {float(c1['fid128']):.3f}",
    )
    ax_curve.plot(
        steps_k,
        curve,
        color=current_color,
        marker="o",
        markersize=5.5,
        linewidth=2.4,
        label="Shift-only raw GMM · FID128",
        zorder=3,
    )
    ax_curve.scatter(
        [current_step / 1000],
        [current_fid],
        s=95,
        color=current_color,
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
    )
    ax_curve.annotate(
        f"{current_fid:.3f}",
        xy=(current_step / 1000, current_fid),
        xytext=(-10, 13),
        textcoords="offset points",
        ha="right",
        color=current_color,
        fontweight="bold",
    )
    ax_curve.set_title("Shift-only raw GMM training curve", loc="left", fontweight="bold")
    ax_curve.set_xlabel("Training step (thousands)")
    ax_curve.set_ylabel("FID128 · lower is better")
    ax_curve.set_xticks(steps_k)
    ax_curve.set_ylim(6.2, 28.5)
    ax_curve.legend(loc="upper right", frameon=True, fontsize=9)
    ax_curve.grid(True, color="#D9D9D9", linewidth=0.8)
    ax_curve.spines[["top", "right"]].set_visible(False)

    comparison = [
        *historical,
        {
            "id": "NEW",
            "name": "Shift-only raw",
            "protocol": f"single @ {current_step // 1000}k",
            "fid128": current_fid,
            "fid_group": "current",
        },
    ]
    comparison.sort(key=lambda row: float(row["fid128"]))
    y_positions = list(range(len(comparison)))
    colors = [
        current_color
        if row["fid_group"] == "current"
        else best_color
        if row["fid_group"] == "best"
        else control_color
        for row in comparison
    ]
    markers = [
        "D" if row["fid_group"] == "current" else "o"
        for row in comparison
    ]
    for y, row, color, marker in zip(y_positions, comparison, colors, markers):
        fid = float(row["fid128"])
        ax_compare.hlines(y, 6.5, fid, color=color, alpha=0.28, linewidth=2)
        ax_compare.scatter(
            fid,
            y,
            s=95 if row["fid_group"] == "current" else 68,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax_compare.text(
            fid + 0.045,
            y,
            f"{fid:.3f}",
            va="center",
            ha="left",
            color=color,
            fontweight="bold" if row["fid_group"] == "current" else "normal",
        )
    ax_compare.set_yticks(
        y_positions,
        [
            f"{row['id']} · {row['name']}\n{row['protocol']}"
            for row in comparison
        ],
    )
    ax_compare.invert_yaxis()
    ax_compare.set_xlim(6.5, 9.2)
    ax_compare.set_xlabel("Reported historical FID128")
    ax_compare.set_title(
        "Historical reference points",
        loc="left",
        fontweight="bold",
    )
    ax_compare.grid(True, axis="x", color="#D9D9D9", linewidth=0.8)
    ax_compare.grid(False, axis="y")
    ax_compare.spines[["top", "right", "left"]].set_visible(False)
    ax_compare.tick_params(axis="y", length=0, labelsize=9)
    ax_compare.text(
        0.99,
        -0.17,
        "Historical points mix checkpoints and FID aggregation rules;\n"
        "use as descriptive references, not a matched significance test.",
        transform=ax_compare.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#555555",
    )

    improvement = (curve[0] - current_fid) / curve[0] * 100.0
    fig.suptitle(
        "FID128: mean-shift-only run versus historical GMM-TIDE runs",
        fontsize=15,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    fig.text(
        0.04,
        0.925,
        f"Current run improves {improvement:.1f}% from 20k to {current_step // 1000}k; "
        "the 400k resume has no completed FID128 yet.",
        fontsize=10,
        color="#555555",
        ha="left",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.90), w_pad=3.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-metrics", type=Path, default=DEFAULT_CURRENT_METRICS)
    parser.add_argument(
        "--historical-manifest",
        type=Path,
        default=DEFAULT_HISTORICAL_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    current = load_current_fid128(args.current_metrics)
    historical = load_historical_fid128(args.historical_manifest)
    render(current, historical, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
