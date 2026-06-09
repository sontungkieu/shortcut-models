# Router Deep4 Analysis

## Metric Source Policy

- Primary source: Kaggle diagnostics (`diagnostics/*.json/jsonl/csv`).
- Fallback source: W&B history snapshot only when Kaggle output is unavailable or has no diagnostics.
- This report records `metric_source` per run, so Kaggle and W&B are not silently mixed.
- No checkpoint files were downloaded.

## Ranking

| rank | config | source | FID128 | FID32 | delta vs 6.969 | delta vs reg-best 7.147 | valid loss | pred/target var | flow straight/curv |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `d4, none, drop=0.2` | kaggle_diagnostics | 7.094 | 9.286 | 0.125 | -0.053 | 0.4562 | 0.651 | 1.111/0.0208 |
| 2 | `d5, layer_norm, drop=0.2` | kaggle_diagnostics | 7.164 | 9.238 | 0.195 | 0.017 | 0.4610 | 0.661 | 1.111/0.0209 |
| 3 | `d4, layer_norm, drop=0.2` | kaggle_diagnostics | 7.294 | 9.519 | 0.325 | 0.147 | 0.4644 | 0.649 | 1.111/0.0208 |
| 4 | `d4, none, drop=0.3` | kaggle_diagnostics | 7.335 | 9.569 | 0.366 | 0.188 | 0.4607 | 0.642 | 1.110/0.0207 |

## Geometry / Router Signals

| config | usage H | router top1/loss | x0-x1 cos | v-x1 cos | v-x0 cos | topk mu cos | angular disp |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d4, none, drop=0.2` | 0.949 | 0.943/0.144 | 0.163 | 0.650 | -0.636 | 0.766 | 0.000005 |
| `d5, layer_norm, drop=0.2` | 0.948 | 0.964/0.111 | 0.174 | 0.637 | -0.640 | 0.762 | 0.000065 |
| `d4, layer_norm, drop=0.2` | 0.947 | 0.926/0.170 | 0.175 | 0.633 | -0.641 | 0.762 | 0.000066 |
| `d4, none, drop=0.3` | 0.943 | 0.944/0.147 | 0.184 | 0.635 | -0.634 | 0.788 | 0.000008 |

## Insights

1. Best deep-router run is `d4, none, drop=0.2` with FID128 `7.094`. It is still worse than historical baseline `6.969` by `0.125`, but it is better than the previous router-reg best `7.147` by `0.053`.
2. Depth 4 + dropout 0.2 is the strongest result in this batch and improves over the earlier shallow dropout/LN sweep. This suggests moderate extra router depth can help, but only in the plain dropout setting.
3. More depth or more regularization is not automatically better: d5 + LayerNorm + dropout 0.2 is second, while d4 + LayerNorm + dropout 0.2 and d4 + dropout 0.3 are clearly worse.
4. Better router distillation metrics do not guarantee best FID. The d5 LayerNorm run has the best router top1/loss, but d4 dropout 0.2 has better FID128.
5. `pred/target variance` remains around 0.64-0.66 and flow metrics are almost tied, so deeper router still does not solve the FM variance/source-geometry issue.
6. The practical next step is not a broad depth sweep. Keep d4 dropout 0.2 as a candidate and combine it with the stronger time/source geometry settings; avoid d4 dropout 0.3 and do not expand LayerNorm-depth blindly.

## Files

- Pull report: `reports/gmm_tide_router_deep4_download_20260609.json`
- Analysis JSON: `reports/gmm_tide_router_deep4_analysis_20260609.json`
- Analysis CSV: `reports/gmm_tide_router_deep4_analysis_20260609.csv`
- Kaggle output root: `outputs/kaggle/router_deep4_20260609`
- W&B fallback root: `outputs/kaggle_metrics_20260609/router_deep4_wandb`
