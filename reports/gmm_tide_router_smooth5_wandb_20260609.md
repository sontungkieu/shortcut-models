# Router Smooth5 W&B Snapshot

Kaggle output files were unavailable because the kernels still report `RUNNING`; this report uses W&B history snapshot. `samples=10000` should include all current log rows for these runs.

Output root: `outputs/kaggle_metrics_20260609/router_smooth5_wandb`

| run | state | step | FID128 best/last | FID32 best/last | valid loss best/last | variant | pred/target var | straight/curv last | x0/x1 |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| `tide-routersmooth-bridgeb22-k16-top2-soft075-dir512` | running | 349900 | 7.28/7.28 | 9.44/9.44 | 0.4343/0.4592 | bridge | 0.650 | 1.110/0.0209 | 1.036 |
| `tide-routersmooth-sampletopk-ln-drop02-k16-top2-soft075-dir512` | running | 349900 | 7.30/7.30 | 9.58/9.58 | 0.4262/0.4576 | sample_topk, layer_norm, drop=0.2 | 0.656 | 1.110/0.0207 | 1.039 |
| `tide-routersmooth-temp20-k16-top2-soft075-dir512` | running | 349900 | 7.40/7.40 | 9.61/9.61 | 0.4316/0.4562 | T=2.0 | 0.643 | 1.110/0.0209 | 1.031 |
| `tide-routersmooth-temp15-k16-top2-soft075-dir512` | running | 349900 | 7.51/7.51 | 9.71/9.71 | 0.4382/0.4573 | T=1.5 | 0.644 | 1.111/0.0208 | 1.037 |
| `tide-routersmooth-entropyfloor025w005-k16-top2-soft075-dir512` | running | 349900 | 7.53/7.53 | 9.70/9.70 | 0.4360/0.4548 | efloor=0.25,w=0.05 | 0.649 | 1.111/0.0207 | 1.034 |
