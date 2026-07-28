from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SHIFT_SCALE_ROOT = Path(
    "outputs/kaggle_jobs/gmm_tide_moe2_shift_scale_raw_200k_20260727"
)
DEFAULT_VARIANTS = {
    0.75: (
        SHIFT_SCALE_ROOT
        / "kieuhongquan__tide-sc075-shift-raw-s0-200k-k16-top2-soft075-ki"
        / "output/gmm_tide_fm/tide-sc075-shift-raw-s0-200k-k16-top2-soft075"
        / "diagnostics/train_metrics.csv"
    ),
    0.875: (
        SHIFT_SCALE_ROOT
        / "nguyncmnhda__tide-sc0875-shift-raw-s0-200k-k16-top2-soft075-n"
        / "output/gmm_tide_fm/tide-sc0875-shift-raw-s0-200k-k16-top2-soft075"
        / "diagnostics/train_metrics.csv"
    ),
    1.25: (
        SHIFT_SCALE_ROOT
        / "phamdotuandng__tide-sc125-shift-raw-s0-200k-k16-top2-soft075-ph"
        / "output/gmm_tide_fm/tide-sc125-shift-raw-s0-200k-k16-top2-soft075"
        / "diagnostics/train_metrics.csv"
    ),
}
DEFAULT_SHIFT_C1 = Path(
    "outputs/kaggle_jobs/gmm_tide_moe2_shift_only_raw_20260726/"
    "johnntlhudson__tide-moe2-shiftmean-raw-s0-k16-top2-soft075-john/"
    "output/gmm_tide_fm/tide-moe2-shiftmean-raw-s0-k16-top2-soft075/"
    "diagnostics/train_metrics.csv"
)
DEFAULT_NO_SHIFT = Path(
    "outputs/kaggle/moe2_resubmit_fixed_20260512/"
    "iamlonely__tide-k16-top2-softv0p75-s128-dir512-iamlonely-20/"
    "gmm_tide_fm/tide-k16-top2-softv0p75-s128-dir512/"
    "diagnostics/train_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "reports/figures/gmm_tide_shift_scale_three_variants_fid128.png"
)


def load_fid128(path: Path, *, max_step: int = 200_000) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != "eval":
                continue
            if row.get("metric") != "fid/timesteps/128":
                continue
            step = int(row["step"])
            if step <= max_step:
                values[step] = float(row["value"])
    if not values:
        raise ValueError(f"No eval FID128 rows found in {path}")
    return values


def plot_curve(
    ax: plt.Axes,
    values: dict[int, float],
    *,
    label: str,
    color: str,
    marker: str,
    linewidth: float,
    linestyle: str = "-",
    zorder: int = 3,
) -> None:
    rows = sorted(values.items())
    ax.plot(
        [step / 1000 for step, _ in rows],
        [fid for _, fid in rows],
        label=label,
        color=color,
        marker=marker,
        markersize=5.5,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )


def render(
    variants: dict[float, dict[int, float]],
    shift_c1: dict[int, float],
    no_shift: dict[int, float],
    output: Path,
) -> None:
    required_step = 200_000
    all_series = [*variants.values(), shift_c1, no_shift]
    if any(required_step not in series for series in all_series):
        raise ValueError("Every series must contain the shared 200k evaluation")

    colors = {
        0.75: "#4C78A8",
        0.875: "#2A9D8F",
        1.25: "#F28E2B",
    }
    markers = {0.75: "o", 0.875: "D", 1.25: "^"}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11.4, 6.6))

    plot_curve(
        ax,
        no_shift,
        label="No shift · historical raw-MOE2",
        color="#555555",
        marker="s",
        linewidth=2.0,
        linestyle="--",
        zorder=2,
    )
    plot_curve(
        ax,
        shift_c1,
        label="Shift only · c=1",
        color="#999999",
        marker="x",
        linewidth=2.0,
        linestyle=":",
        zorder=2,
    )
    for scale, values in sorted(variants.items()):
        plot_curve(
            ax,
            values,
            label=f"Shift + scale · c={scale:g}",
            color=colors[scale],
            marker=markers[scale],
            linewidth=2.6,
        )

    endpoint_order = [
        ("No shift", no_shift[required_step], "#555555", 7.65),
        ("c=0.875", variants[0.875][required_step], colors[0.875], 8.10),
        ("c=1", shift_c1[required_step], "#777777", 8.55),
        ("c=0.75", variants[0.75][required_step], colors[0.75], 9.00),
        ("c=1.25", variants[1.25][required_step], colors[1.25], 9.45),
    ]
    for label, fid, color, label_y in endpoint_order:
        ax.annotate(
            f"{label}: {fid:.4f}",
            xy=(200, fid),
            xytext=(207, label_y),
            textcoords="data",
            color=color,
            fontsize=9.2,
            fontweight="bold",
            va="center",
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 0.9,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    best_scale = min(variants, key=lambda scale: variants[scale][required_step])
    best_fid = variants[best_scale][required_step]
    shift_c1_fid = shift_c1[required_step]
    no_shift_fid = no_shift[required_step]

    fig.suptitle(
        "FID128: three completed shift+scale variants",
        x=0.075,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.925,
        f"At 200k, best of the three is c={best_scale:g}: {best_fid:.4f} "
        f"(c=1 shift-only: {shift_c1_fid:.4f}; no shift: {no_shift_fid:.4f}).",
        color="#555555",
        fontsize=10,
        ha="left",
        va="top",
    )
    fig.text(
        0.075,
        0.888,
        "No shift is the closest historical raw-MOE2 reference, not a paired "
        "same-commit rerun. Lower is better.",
        color="#777777",
        fontsize=9,
        ha="left",
        va="top",
    )

    ax.set_xlabel("Training step (thousands)")
    ax.set_ylabel("FID128 · lower is better")
    ax.set_xlim(10, 235)
    ax.set_ylim(7.0, 30.5)
    ax.set_xticks([20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
    ax.grid(True, color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.84))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift-c1", type=Path, default=DEFAULT_SHIFT_C1)
    parser.add_argument("--no-shift", type=Path, default=DEFAULT_NO_SHIFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    render(
        variants={
            scale: load_fid128(path)
            for scale, path in DEFAULT_VARIANTS.items()
        },
        shift_c1=load_fid128(args.shift_c1),
        no_shift=load_fid128(args.no_shift),
        output=args.output,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
