# GMM-TIDE Mix/Continue 12 Run Report

- Date: 2026-05-13
- Source submit report: `reports/gmm_tide_fm_mix_continue12_combined_20260512.json`
- Downloaded diagnostics root: `outputs/kaggle/gmm_tide_mix_continue12_20260513/`
- Status: `{"CANCEL_ACKNOWLEDGED": 12}`; all 12 stopped by Kaggle max execution duration after producing 8 eval points.
- Last eval step for all runs: `320000`; train logs reached roughly `333k-345k`. Baseline runs reached later steps, so deltas are directional, not perfectly step-matched.

## Overall Rank By FID128

| rank | baseline | family | run | K | step | FID128 | ΔFID128 vs base | FID32 | router val loss/top1 | FM pred/target var | x0/x1 |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | mix+cont10 | `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` | 16 | 320000 | 7.315 | 0.346 | 9.572 | 0.195/0.921 | 0.656 | 1.006 |
| 2 | 1 | mix | `tide-mc-r1-k16-top2-soft075-dir512-mix` | 16 | 320000 | 7.466 | 0.498 | 9.667 | 0.194/0.922 | 0.648 | 1.025 |
| 3 | 2 | mix | `tide-mc-r2-k32-top2-soft075-dir001-mix` | 32 | 320000 | 7.617 | 0.499 | 9.882 | 0.263/0.903 | 0.644 | 1.013 |
| 4 | 3 | x1+cont10 | `tide-mc-r3-k32-top2-none-hard05-x1cont10` | 32 | 320000 | 7.622 | 0.367 | 10.061 | 0.239/0.905 | 0.642 | 1.000 |
| 5 | 4 | mix | `tide-mc-r4-k32-top2-dir001-hard05-mix` | 32 | 320000 | 7.740 | 0.213 | 10.037 | 0.275/0.892 | 0.665 | 1.040 |
| 6 | 2 | mix+cont10 | `tide-mc-r2-k32-top2-soft075-dir001-mixcont10` | 32 | 320000 | 7.747 | 0.629 | 10.034 | 0.276/0.887 | 0.691 | 1.009 |
| 7 | 1 | x1+cont10 | `tide-mc-r1-k16-top2-soft075-dir512-x1cont10` | 16 | 320000 | 7.797 | 0.828 | 10.039 | 0.171/0.930 | 0.637 | 1.000 |
| 8 | 3 | mix | `tide-mc-r3-k32-top2-none-hard05-mix` | 32 | 320000 | 7.806 | 0.550 | 10.217 | 0.252/0.891 | 0.684 | 0.987 |
| 9 | 2 | x1+cont10 | `tide-mc-r2-k32-top2-soft075-dir001-x1cont10` | 32 | 320000 | 7.815 | 0.698 | 10.087 | 0.264/0.900 | 0.643 | 1.000 |
| 10 | 4 | x1+cont10 | `tide-mc-r4-k32-top2-dir001-hard05-x1cont10` | 32 | 320000 | 7.880 | 0.353 | 10.210 | 0.238/0.903 | 0.663 | 0.994 |
| 11 | 3 | mix+cont10 | `tide-mc-r3-k32-top2-none-hard05-mixcont10` | 32 | 320000 | 7.892 | 0.636 | 10.228 | 0.218/0.921 | 0.642 | 0.997 |
| 12 | 4 | mix+cont10 | `tide-mc-r4-k32-top2-dir001-hard05-mixcont10` | 32 | 320000 | 8.033 | 0.506 | 10.379 | 0.263/0.911 | 0.627 | 1.012 |

## Comparison Within Each Baseline

| baseline rank | baseline run | baseline FID128 | best new family | best new FID128 | delta | all variants FID128 |
|---:|---|---:|---|---:|---:|---|
| 1 | `tide-k16-top2-softv0p75-s128-dir512` | 6.969 | mix+cont10 | 7.315 | 0.346 | mix: 7.466, mix+cont10: 7.315, x1+cont10: 7.797 |
| 2 | `tide-k32-top2-softv0p75-s128-dir001` | 7.118 | mix | 7.617 | 0.499 | mix: 7.617, mix+cont10: 7.747, x1+cont10: 7.815 |
| 3 | `tide-k32-top2-g136-none-hardv0p5` | 7.256 | x1+cont10 | 7.622 | 0.367 | mix: 7.806, mix+cont10: 7.892, x1+cont10: 7.622 |
| 4 | `tide-k32-top2-g145-dir001-hardv0p5` | 7.528 | mix | 7.740 | 0.213 | mix: 7.740, mix+cont10: 8.033, x1+cont10: 7.880 |

## GMM And Router Diagnostics

| run | final GMM stage | train/valid NLL | pi entropy | comp/data var | floor hit | router overfit gap | straightness |
|---|---|---:|---:|---:|---:|---:|---:|
| `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` | mix_continue | 4273.1/4259.7 | 0.996 | 0.793 | 0.000 | 0.040 | 1.110 |
| `tide-mc-r1-k16-top2-soft075-dir512-mix` | mix | 4285.3/4265.5 | 0.988 | 0.792 | 0.000 | -0.060 | 1.111 |
| `tide-mc-r2-k32-top2-soft075-dir001-mix` | mix | 4262.1/4218.2 | 0.922 | 0.844 | 0.000 | 0.070 | 1.110 |
| `tide-mc-r3-k32-top2-none-hard05-x1cont10` | x1_continue | 4182.4/4188.7 | 0.985 | 0.765 | 0.119 | -0.169 | 1.108 |
| `tide-mc-r4-k32-top2-dir001-hard05-mix` | mix | 4220.2/4208.6 | 0.943 | 0.776 | 0.119 | -0.049 | 1.108 |
| `tide-mc-r2-k32-top2-soft075-dir001-mixcont10` | mix_continue | 4214.7/4189.9 | 0.990 | 0.790 | 0.000 | -0.141 | 1.109 |
| `tide-mc-r1-k16-top2-soft075-dir512-x1cont10` | x1_continue | 4253.6/4259.5 | 0.992 | 0.790 | 0.000 | 0.040 | 1.111 |
| `tide-mc-r3-k32-top2-none-hard05-mix` | mix | 4223.0/4206.4 | 0.944 | 0.782 | 0.103 | -0.082 | 1.109 |
| `tide-mc-r2-k32-top2-soft075-dir001-x1cont10` | x1_continue | 4176.0/4179.9 | 0.990 | 0.778 | 0.000 | 0.169 | 1.108 |
| `tide-mc-r4-k32-top2-dir001-hard05-x1cont10` | x1_continue | 4179.5/4186.8 | 0.994 | 0.769 | 0.099 | 0.041 | 1.108 |
| `tide-mc-r3-k32-top2-none-hard05-mixcont10` | mix_continue | 4192.4/4186.5 | 0.994 | 0.767 | 0.098 | -0.045 | 1.108 |
| `tide-mc-r4-k32-top2-dir001-hard05-mixcont10` | mix_continue | 4198.4/4195.7 | 0.975 | 0.765 | 0.131 | -0.084 | 1.109 |

## Takeaways

- Best of the 12 is `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` with FID128 `7.315` and FID32 `9.572`.
- None of the mix/continue variants beats its corresponding no-mix/no-continue baseline from the previous 6-run batch. Best delta is still positive, so this batch does not support replacing the old baseline yet.
- Rank-1 soft K16 remains the strongest family. `mix+cont10` is better than plain `mix` and `x1+cont10` for that family, but still worse than the original baseline.
- For the hard-floor K32 families, `mix+cont10` is consistently not helpful here; rank 3 prefers `x1+cont10`, rank 4 prefers `mix` without continuation.
- All new runs stopped at eval step 320k due Kaggle timeout; for strict comparison, the clean next test should resume the top 2-3 candidates to the same later eval budget as the baseline.

## Files

- JSON: `reports/gmm_tide_mix_continue12_report_20260513.json`
- CSV: `reports/gmm_tide_mix_continue12_report_20260513.csv`
- Markdown: `reports/gmm_tide_mix_continue12_report_20260513.md`
