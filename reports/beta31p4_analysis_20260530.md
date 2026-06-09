# Beta(3,1.4) GMM-TIDE analysis - 2026-05-30

## Scope

- New jobs: 2 x `x1+frozen`, 2 x `jointmix` for `Beta(3,1.4)`.
- All 4 Kaggle kernels stopped by max-duration timeout, so this is partial-training analysis.
- `kiuvithong` jointmix output has GMM/router diagnostics but `train_metrics.csv` is 0 bytes, so FM metrics use the `iamlonely` jointmix replicate.

## New Runs

| run | owner | train step | eval step | FID128 best@step | FID32 best | valid loss tail | flow straight/curv last | pred/target var | x0/x1 | router valid loss/top1 | GMM valid NLL/count ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tide-beta31p4-x1frozen-k16-top2-soft075-dir512 | huynhtule | 240000 | 240000 | 8.05@220000 | 9.70 | 0.4586 | 1.109/0.068 | 0.657 | 1.003 | 0.299/0.872 | 4254.3/1.87 |
| tide-beta31p4-jointmix-k16-top2-soft075-dir512-r2 | iamlonely | 220000 | 220000 | 8.48@200000 | 9.49 | 0.4641 | 1.104/0.066 | 0.654 | 1.008 | 0.219/0.914 | 4263.5/5.02 |
|  | kiuvithong | | | missing | | | | | | | |
| tide-beta31p4-x1frozen-k16-top2-soft075-dir512-r2 | victorharvey27 | 240000 | 240000 | 7.93@220000 | 9.44 | 0.4573 | 1.091/0.393 | 0.661 | 1.002 | 0.276/0.886 | 4257.5/2.21 |

## Context Comparison

| key | label | train step | FID128 best@step | FID128 last | FID32 best | valid loss tail | flow straight/curv last |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_old | baseline old uniform-ish | 365600 | 6.97@350000 | 6.97 | 9.29 | 0.4741 | 1.109/0.021 |
| beta31_resume | Beta(3,1) resumed | 376600 | 7.33@360000 | 7.33 | 8.42 | 0.4678 | 1.097/0.019 |
| beta31p4_victorharvey27 | Beta(3,1.4) tide-beta31p4-x1frozen-k16-top2-soft075-dir512-r2 | 240000 | 7.93@220000 | 7.93 | 9.44 | 0.4573 | 1.091/0.393 |
| beta31p4_huynhtule | Beta(3,1.4) tide-beta31p4-x1frozen-k16-top2-soft075-dir512 | 240000 | 8.05@220000 | 8.05 | 9.70 | 0.4586 | 1.109/0.068 |
| beta31p4_iamlonely | Beta(3,1.4) tide-beta31p4-jointmix-k16-top2-soft075-dir512-r2 | 220000 | 8.48@200000 | 8.48 | 9.49 | 0.4641 | 1.104/0.066 |
| beta35_A | Beta(3.5,1.3) jointmix uniform ODE | 220000 | 8.51@180000 | 9.00 | 9.36 | 0.4722 | 1.104/0.065 |
| beta35_B | Beta(3.5,1.3) jointmix end-dense ODE | 220000 | 9.12@180000 | 9.72 | 11.88 | 0.4780 | 1.118/0.067 |
| beta41 | Beta(4,1) jointmix uniform ODE | 220000 | 9.35@180000 | 11.27 | 10.08 | 0.4823 | 1.096/0.062 |

## Plots

- `reports/figures/beta31p4_fid128_20260530.png`
- `reports/figures/beta31p4_fid32_20260530.png`
- `reports/figures/beta31p4_valid_loss_20260530.png`
- `reports/figures/beta31p4_flow_curvature_20260530.png`

## Notes

- Best new partial run is `tide-beta31p4-x1frozen-k16-top2-soft075-dir512-r2` with FID128 `7.93` at step `220000`.
- Among the complete new logs, `x1+frozen` beats `jointmix` on FID128. This suggests the mix/joint change is still not helping this source at this compute budget.
- `Beta(3,1.4)` improves over the previous `Beta(3.5,1.3)` jointmix run at similar step range, but it has not beaten the old baseline or the resumed `Beta(3,1)` run yet.
- Valid FM loss alone is misleading: the jointmix replicate has the lowest tail valid loss among the new runs, but worse FID than `x1+frozen`.
- The checkpoint interval is 150k, while best FID appears around 220k for the new x1-frozen runs; resuming these exact kernels would likely restart from 150k unless a newer checkpoint was saved elsewhere.
