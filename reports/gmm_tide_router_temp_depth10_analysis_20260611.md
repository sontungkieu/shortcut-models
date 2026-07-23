# Router Temp/Depth10 Analysis

- Generated: 2026-06-11T05:12:06.919492+00:00
- Download root: `outputs/kaggle/router_temp_depth10_20260611`
- Parsed runs: 10
- Status: {'CANCEL_ACKNOWLEDGED': 10}

## Ranking

| rank | config | step | FID128 | FID32 | router valid/top1 | overfit gap | pred/target var | x0/x1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `d5 drop0.2 none T=1.25` | 356200 | 7.222 | 9.523 | 0.101/0.961 | 0.028 | 0.635 | 1.004 |
| 2 | `d5 drop0.2 none T=1.0` | 350000 | 7.245 | 9.436 | 0.080/0.976 | -0.107 | 0.665 | 0.990 |
| 3 | `d7 drop0.2 layer_norm T=1.25` | 344200 | 7.282 | 9.504 | 0.158/0.936 | -0.004 | 0.643 | 0.994 |
| 4 | `d3 drop0.2 none T=2.5` | 350000 | 7.307 | 9.441 | 0.206/0.917 | 0.071 | 0.653 | 0.986 |
| 5 | `d3 drop0.2 none T=3.0` | 350000 | 7.319 | 9.457 | 0.235/0.914 | 0.120 | 0.652 | 0.985 |
| 6 | `d3 drop0.0 none T=2.5` | 350000 | 7.417 | 9.599 | 0.197/0.928 | -0.085 | 0.654 | 0.991 |
| 7 | `d7 drop0.2 none T=1.25` | 350000 | 7.475 | 9.719 | 0.132/0.953 | -0.006 | 0.643 | 0.986 |
| 8 | `d3 drop0.0 none T=3.0` | 350000 | 7.494 | 9.718 | 0.195/0.925 | 0.051 | 0.640 | 0.990 |
| 9 | `d7 drop0.2 none T=1.0` | 350000 | 7.503 | 9.694 | 0.113/0.952 | 0.032 | 0.655 | 1.002 |
| 10 | `d3 drop0.2 none T=2.0` | 350000 | 7.542 | 9.849 | 0.181/0.929 | 0.077 | 0.643 | 0.987 |

## Notes

- All 10 kernels are `CANCEL_ACKNOWLEDGED`, but each produced parseable diagnostics and train summaries.
- No heavy checkpoint/archive artifact was downloaded; downloaded files are diagnostics/log/notebook output only.
- Best in this batch is depth 5, dropout 0.2, target temperature 1.25 with FID128 7.222.
- Depth 5 beats depth 3 and depth 7 in this batch; depth 7 benefits from LayerNorm at T=1.25, but still trails depth 5.
- Higher target temperature in depth 3 does not monotonically help: T=2.5/3.0 with dropout 0.2 is better than T=2.0, but no-drop high T is worse.
