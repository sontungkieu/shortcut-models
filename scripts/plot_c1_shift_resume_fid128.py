from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SOURCE_METRICS = Path(
    "outputs/kaggle_jobs/gmm_tide_moe2_shift_only_raw_20260726/"
    "johnntlhudson__tide-moe2-shiftmean-raw-s0-k16-top2-soft075-john/"
    "output/gmm_tide_fm/tide-moe2-shiftmean-raw-s0-k16-top2-soft075/"
    "diagnostics/train_metrics.csv"
)
DEFAULT_RESUME_METRICS = Path(
    "outputs/kaggle_jobs/gmm_tide_moe2_shift_only_raw_resume400_r1_20260727/"
    "johnntlhudson__tide-moe2-shiftmean-raw-s0-resume400-r1-k16-top2/"
    "output/gmm_tide_fm/"
    "tide-moe2-shiftmean-raw-s0-resume400-r1-k16-top2-soft075/"
    "diagnostics/train_metrics.csv"
)
DEFAULT_OUTPUT = Path("reports/figures/gmm_tide_shift_only_c1_resume_fid128.png")


def load_fid128(path: Path) -> list[tuple[int, float]]:
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
    return sorted(values.items())


def render(
    source: list[tuple[int, float]],
    resume: list[tuple[int, float]],
    output: Path,
) -> None:
    source_end_step, source_end_fid = source[-1]
    resume_end_step, resume_end_fid = resume[-1]
    if source_end_step != 200_000 or resume_end_step != 400_000:
        raise ValueError(
            f"Expected source/resume endpoints 200k/400k, got "
            f"{source_end_step}/{resume_end_step}"
        )

    source_x = [step / 1000 for step, _ in source]
    source_y = [fid for _, fid in source]
    resume_with_anchor = [(source_end_step, source_end_fid), *resume]
    resume_x = [step / 1000 for step, _ in resume_with_anchor]
    resume_y = [fid for _, fid in resume_with_anchor]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    source_color = "#4C78A8"
    resume_color = "#E45756"

    ax.plot(
        source_x,
        source_y,
        color=source_color,
        marker="o",
        markersize=5.5,
        linewidth=2.5,
        label="Run gốc · 0–200k",
    )
    ax.plot(
        resume_x,
        resume_y,
        color=resume_color,
        marker="o",
        markersize=5.5,
        linewidth=2.5,
        label="Resume · 200–400k",
    )
    ax.axvline(
        200,
        color="#666666",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
    )
    ax.text(
        204,
        26.1,
        "Resume from 200k checkpoint",
        color="#555555",
        fontsize=9.5,
        va="top",
    )

    ax.scatter(
        [200, 400],
        [source_end_fid, resume_end_fid],
        s=100,
        color=[source_color, resume_color],
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
    )
    ax.annotate(
        f"200k: {source_end_fid:.4f}",
        (200, source_end_fid),
        xytext=(-12, 14),
        textcoords="offset points",
        ha="right",
        color=source_color,
        fontweight="bold",
    )
    ax.annotate(
        f"400k: {resume_end_fid:.4f}",
        (400, resume_end_fid),
        xytext=(-10, 14),
        textcoords="offset points",
        ha="right",
        color=resume_color,
        fontweight="bold",
    )

    absolute_gain = source_end_fid - resume_end_fid
    relative_gain = absolute_gain / source_end_fid * 100
    fig.suptitle(
        "GMM-TIDE shift-only (c=1): FID128 through resume training",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.925,
        f"200k → 400k: {source_end_fid:.4f} → {resume_end_fid:.4f} "
        f"(−{absolute_gain:.4f}, −{relative_gain:.1f}%)",
        color="#555555",
        fontsize=10,
        ha="left",
        va="top",
    )
    ax.set_xlabel("Training step (thousands)")
    ax.set_ylabel("FID128 · lower is better")
    ax.set_xlim(10, 410)
    ax.set_ylim(6.5, 28.5)
    ax.set_xticks(range(20, 401, 20))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument("--resume-metrics", type=Path, default=DEFAULT_RESUME_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(
        source=load_fid128(args.source_metrics),
        resume=load_fid128(args.resume_metrics),
        output=args.output,
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
