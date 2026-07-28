from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SHIFT_SOURCE = Path(
    "outputs/kaggle_jobs/gmm_tide_moe2_shift_only_raw_20260726/"
    "johnntlhudson__tide-moe2-shiftmean-raw-s0-k16-top2-soft075-john/"
    "output/gmm_tide_fm/tide-moe2-shiftmean-raw-s0-k16-top2-soft075/"
    "diagnostics/train_metrics.csv"
)
DEFAULT_SHIFT_RESUME = Path(
    "outputs/kaggle_jobs/gmm_tide_moe2_shift_only_raw_resume400_r1_20260727/"
    "johnntlhudson__tide-moe2-shiftmean-raw-s0-resume400-r1-k16-top2/"
    "output/gmm_tide_fm/"
    "tide-moe2-shiftmean-raw-s0-resume400-r1-k16-top2-soft075/"
    "diagnostics/train_metrics.csv"
)
DEFAULT_NO_SHIFT = Path(
    "outputs/kaggle/moe2_resubmit_fixed_20260512/"
    "iamlonely__tide-k16-top2-softv0p75-s128-dir512-iamlonely-20/"
    "gmm_tide_fm/tide-k16-top2-softv0p75-s128-dir512/"
    "diagnostics/train_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "reports/figures/gmm_tide_c1_shift_vs_no_shift_fid128.png"
)


def load_fid128(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != "eval":
                continue
            if row.get("metric") != "fid/timesteps/128":
                continue
            values[int(row["step"])] = float(row["value"])
    if not values:
        raise ValueError(f"No eval FID128 rows found in {path}")
    return values


def render(
    shift_source: dict[int, float],
    shift_resume: dict[int, float],
    no_shift: dict[int, float],
    output: Path,
) -> None:
    shift = dict(shift_source)
    shift.update(shift_resume)
    shift_rows = sorted(shift.items())
    no_shift_rows = sorted(no_shift.items())
    common_steps = sorted(set(shift) & set(no_shift))
    if 200_000 not in common_steps:
        raise ValueError("Expected a matched 200k evaluation point")

    shift_x = [step / 1000 for step, _ in shift_rows]
    shift_y = [fid for _, fid in shift_rows]
    no_shift_x = [step / 1000 for step, _ in no_shift_rows]
    no_shift_y = [fid for _, fid in no_shift_rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11.3, 6.4))
    shift_color = "#E45756"
    no_shift_color = "#4C78A8"

    ax.plot(
        no_shift_x,
        no_shift_y,
        color=no_shift_color,
        marker="s",
        markersize=6,
        linewidth=2.5,
        label="No shift · closest historical raw-MOE2",
        zorder=2,
    )
    ax.plot(
        shift_x,
        shift_y,
        color=shift_color,
        marker="o",
        markersize=5.5,
        linewidth=2.5,
        label="Shift global mean · c=1",
        zorder=3,
    )
    ax.axvline(
        200,
        color="#666666",
        linestyle="--",
        linewidth=1.3,
        alpha=0.8,
    )
    ax.text(
        204,
        26.1,
        "Shift run resumes at 200k",
        color="#555555",
        fontsize=9.5,
        va="top",
    )

    shift_200 = shift[200_000]
    no_shift_200 = no_shift[200_000]
    delta_200 = shift_200 - no_shift_200
    ax.scatter(
        [200, 200],
        [shift_200, no_shift_200],
        s=105,
        color=[shift_color, no_shift_color],
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
    )
    ax.annotate(
        f"Shift: {shift_200:.4f}",
        (200, shift_200),
        xytext=(12, 16),
        textcoords="offset points",
        color=shift_color,
        fontweight="bold",
    )
    ax.annotate(
        f"No shift: {no_shift_200:.4f}",
        (200, no_shift_200),
        xytext=(12, -22),
        textcoords="offset points",
        color=no_shift_color,
        fontweight="bold",
    )

    shift_last_step, shift_last_fid = shift_rows[-1]
    no_shift_last_step, no_shift_last_fid = no_shift_rows[-1]
    ax.annotate(
        f"{shift_last_step // 1000}k: {shift_last_fid:.4f}",
        (shift_last_step / 1000, shift_last_fid),
        xytext=(-10, 15),
        textcoords="offset points",
        ha="right",
        color=shift_color,
        fontweight="bold",
    )
    ax.annotate(
        f"{no_shift_last_step // 1000}k: {no_shift_last_fid:.4f}",
        (no_shift_last_step / 1000, no_shift_last_fid),
        xytext=(-10, -23),
        textcoords="offset points",
        ha="right",
        color=no_shift_color,
        fontweight="bold",
    )

    fig.suptitle(
        "FID128: global-mean shift (c=1) versus no shift",
        x=0.075,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.925,
        f"At the shared 200k checkpoint: shift {shift_200:.4f} vs "
        f"no shift {no_shift_200:.4f} (shift +{delta_200:.4f}; lower is better).",
        color="#555555",
        fontsize=10,
        ha="left",
        va="top",
    )
    fig.text(
        0.075,
        0.888,
        "No-shift curve is the closest historical raw-MOE2 reference, "
        "not a paired rerun on the same commit.",
        color="#777777",
        fontsize=9,
        ha="left",
        va="top",
    )

    ax.set_xlabel("Training step (thousands)")
    ax.set_ylabel("FID128 · lower is better")
    ax.set_xlim(10, 410)
    ax.set_ylim(6.2, 28.5)
    ax.set_xticks([20, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.grid(True, color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.84))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift-source", type=Path, default=DEFAULT_SHIFT_SOURCE)
    parser.add_argument("--shift-resume", type=Path, default=DEFAULT_SHIFT_RESUME)
    parser.add_argument("--no-shift", type=Path, default=DEFAULT_NO_SHIFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(
        shift_source=load_fid128(args.shift_source),
        shift_resume=load_fid128(args.shift_resume),
        no_shift=load_fid128(args.no_shift),
        output=args.output,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
