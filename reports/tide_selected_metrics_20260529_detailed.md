# Selected TIDE/GMM FM Metric Pull - 2026-05-29

Scope: six Kaggle runs requested from W&B screenshot. Outputs were downloaded with Kaggle `file_pattern` limited to diagnostics/log files; no checkpoint files were downloaded.

All six kernels are `KernelWorkerStatus.CANCEL_ACKNOWLEDGED` because they hit Kaggle max runtime. Each still produced diagnostics through train step 220k and eval step 200k.

## Ranking By FID128

| rank | run | config | step train/eval | FID128 best@step | FID128 last | FID32 best | valid loss | flow straight | pred/target var | router valid loss/top1 | router usage H |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tide-beta31-k16-top2-soft075-dir512-jointmix` | Beta(3,1), mean t=0.75 | 220000/200000 | 8.35 @ 180000 | 9.20 | 9.63 | 0.471 | 1.101 | 0.651 | 0.212/0.910 | 0.956 |
| 2 | `tide-beta22-k16-top2-soft075-dir512-jointmix` | Beta(2,2), mid-heavy | 220000/200000 | 9.52 @ 200000 | 9.52 | 11.70 | 0.448 | 1.110 | 0.660 | 0.168/0.936 | 0.928 |
| 3 | `tide-farthestlw5-k16-top2-soft075-dir512-joint` | uniform/default t | 220000/200000 | 9.97 @ 200000 | 9.97 | 12.41 | 0.476 | 1.115 | 0.647 | 0.189/0.923 | 0.949 |
| 4 | `tide-farthestlw5-k32-top2-none-hard05-joint` | uniform/default t | 220000/200000 | 10.41 @ 200000 | 10.41 | 12.95 | 0.475 | 1.112 | 0.651 | 0.269/0.899 | 0.907 |
| 5 | `tide-beta0p5-0p5-k16-top2-soft075-dir512-jointmix` | Beta(0.5,0.5), endpoints-heavy | 220000/200000 | 11.24 @ 200000 | 11.24 | 13.76 | 0.506 | 1.118 | 0.601 | 0.205/0.914 | 0.937 |
| 6 | `tide-beta13-k16-top2-soft075-dir512-jointmix` | Beta(1,3), mean t=0.25 | 220000/200000 | 13.92 @ 200000 | 13.92 | 16.15 | 0.510 | 1.114 | 0.622 | 0.222/0.913 | 0.934 |

## FM And Router Diagnostics

| run | train loss | target var | pred var | residual var frac | x0/x1 mag | router grad/update | usage KL | unique clusters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tide-beta31-k16-top2-soft075-dir512-jointmix` | 0.476 | 1.331 | 0.867 | 0.984 | 0.976 | 0.00/0.000003 | 0.142 | 16 |
| `tide-beta22-k16-top2-soft075-dir512-jointmix` | 0.463 | 1.356 | 0.895 | 0.984 | 0.973 | 0.00/0.000003 | 0.115 | 16 |
| `tide-farthestlw5-k16-top2-soft075-dir512-joint` | 0.478 | 1.377 | 0.891 | 0.984 | 0.966 | 0.00/0.000003 | 0.112 | 16 |
| `tide-farthestlw5-k32-top2-none-hard05-joint` | 0.467 | 1.392 | 0.906 | 0.983 | 0.964 | 0.00/0.000010 | 0.243 | 28 |
| `tide-beta0p5-0p5-k16-top2-soft075-dir512-jointmix` | 0.531 | 1.332 | 0.801 | 0.985 | 0.978 | 0.00/0.000003 | 0.177 | 15 |
| `tide-beta13-k16-top2-soft075-dir512-jointmix` | 0.512 | 1.362 | 0.848 | 0.983 | 0.974 | 0.00/0.000000 | 0.152 | 16 |

## Plots

- [tide_selected_fid128_20260529.png](reports/plots/tide_selected_fid128_20260529.png)
- [tide_selected_fid32_20260529.png](reports/plots/tide_selected_fid32_20260529.png)
- [tide_selected_flow_straightness_20260529.png](reports/plots/tide_selected_flow_straightness_20260529.png)
- [tide_selected_valid_loss_20260529.png](reports/plots/tide_selected_valid_loss_20260529.png)
- [tide_selected_pred_variance_20260529.png](reports/plots/tide_selected_pred_variance_20260529.png)
- [tide_selected_router_usage_entropy_20260529.png](reports/plots/tide_selected_router_usage_entropy_20260529.png)

## Main Insights

- `Beta(3,1)` is the strongest run in this batch: best FID128 = 8.35 at 180k, then worsens to 9.20 at 200k. This is better than the other five selected runs, but the last-point regression means it should be resumed/replicated with checkpoint selection or lower LR after ~160k.
- Time sampling matters more than farthest init in this batch. `Beta(2,2)` is second by final FID128 (9.52) and has the lowest valid FM loss (0.448). `Beta(1,3)` is clearly bad here: FID128 13.92, FID32 16.15.
- Interpreting beta schedules: `Beta(3,1)` samples larger t, closer to x1; `Beta(1,3)` samples smaller t, closer to x0; `Beta(0.5,0.5)` over-samples both endpoints. For this GMM/TIDE source, focusing more near x1 seems helpful.
- Farthest+Lloyd does not win downstream here. K16 farthest reaches FID128 9.97, K32 hard05 farthest reaches 10.41. That is usable but behind beta31/beta22 on the same selected comparison set.
- Router is effectively frozen in the FM phase for these top-k runs: `loss_distill=0`, `kl_to_gmm_base=0`, `top1_prob=1`, and `grad_norm_joint` is ~1e-8 or lower. So the observed differences mainly come from t-sampling/init/source stats, not meaningful joint router learning.
- All runs under-predict velocity variance: pred/target variance is only 0.60-0.66. `Beta(0.5,0.5)` has the worst ratio (0.601) and also poor FID. This supports retuning FM to increase output variance or reduce late-stage under-dispersion.
- x0 magnitude is close to x1 magnitude in every run (x0/x1 ~0.964-0.978), so this batch does not show a severe x0 norm mismatch. The bigger issue is velocity prediction variance and time-sampling.
- Flow straightness correlates with FID in this subset: beta31 has the best last straightness ratio (1.101) and curvature (0.0185); beta0.5 has worse straightness (1.118) and worse FID.

## Suggested Next Runs

- Replicate `Beta(3,1)` once with the same source and eval every 20k; keep best checkpoint around 160k-200k.
- Try `Beta(3,1)` with slower/lower LR after 160k, because FID worsened from 180k to 200k despite good best score.
- Try intermediate schedules between beta22 and beta31, e.g. `Beta(2.5,1.5)` or a warm schedule `Beta(3,1)` early then uniform, to keep the FID gain while avoiding late drift.
- Do not prioritize farthest+Lloyd for full CelebA FM until it beats beta31/beta22 in a replicated downstream run.

## Local Artifacts

- Summary CSV: [tide_selected_metrics_20260529_summary.csv](reports/tide_selected_metrics_20260529_summary.csv)
- Raw selective outputs: [outputs/kaggle/tide_selected_20260529](outputs/kaggle/tide_selected_20260529)
- Collector JSON: [tide_selected_metrics_20260529.json](reports/tide_selected_metrics_20260529.json)
