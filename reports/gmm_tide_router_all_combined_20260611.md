# Router Experiments Combined

## Scope

- Gộp report router regularization/capacity, router-deep, router-smooth/source-geometry và batch router temp/depth10 mới.
- Batch mới thêm 10 run từ `reports/gmm_tide_router_temp_depth10_analysis_20260611.json`.
- Baseline lịch sử để tham chiếu: `FID128 = 6.969`.

## Overall Ranking

| rank | config | batch | source | FID128 | FID32 | delta vs 6.969 | valid loss | router loss/top1 | pred/target var | flow straight/curv |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `d4 none dropout 0.2` | router_deep4 | kaggle_diagnostics | 7.094 | 9.286 | 0.125 | 0.4562 | 0.144/0.943 | 0.651 | 1.111/0.0208 |
| 2 | `LayerNorm + dropout 0.2` | router_reg_capacity | kaggle_diagnostics | 7.147 | 9.414 | 0.178 | 0.4664 | 0.230/0.912 | 0.669 | 1.110/0.0208 |
| 3 | `dropout 0.3` | router_reg_capacity | kaggle_diagnostics | 7.153 | 9.419 | 0.184 | 0.4684 | 0.204/0.914 | 0.628 | 1.109/0.0208 |
| 4 | `d5 layer_norm dropout 0.2` | router_deep4 | kaggle_diagnostics | 7.164 | 9.238 | 0.195 | 0.4610 | 0.111/0.964 | 0.661 | 1.111/0.0209 |
| 5 | `dropout 0.2` | router_reg_capacity | kaggle_diagnostics | 7.175 | 9.431 | 0.206 | 0.4611 | 0.233/0.905 | 0.654 | 1.110/0.0209 |
| 6 | `d5 none dropout 0.2 targetT 1.25` | router_temp_depth10 | kaggle_diagnostics | 7.222 | 9.523 | 0.253 | 0.4678 | 0.101/0.961 | 0.635 | 1.109/0.0208 |
| 7 | `d5 none dropout 0.2 targetT 1.0` | router_temp_depth10 | kaggle_diagnostics | 7.245 | 9.436 | 0.276 | 0.4599 | 0.080/0.976 | 0.665 | 1.110/0.0208 |
| 8 | `low-cap plain` | router_reg_capacity | kaggle_diagnostics | 7.255 | 9.438 | 0.286 | 0.4479 | 0.479/0.820 | 0.674 | 1.110/0.0207 |
| 9 | `bridge Beta(2,2)` | router_smooth5 | wandb_snapshot | 7.277 | 9.435 | 0.308 | 0.4592 | / | 0.650 | 1.110/0.0209 |
| 10 | `d7 layer_norm dropout 0.2 targetT 1.25` | router_temp_depth10 | kaggle_diagnostics | 7.282 | 9.504 | 0.313 | 0.4644 | 0.158/0.936 | 0.643 | 1.110/0.0209 |
| 11 | `d4 layer_norm dropout 0.2` | router_deep4 | kaggle_diagnostics | 7.294 | 9.519 | 0.325 | 0.4644 | 0.170/0.926 | 0.649 | 1.111/0.0208 |
| 12 | `sample_topk source + layer_norm + dropout 0.2` | router_smooth5 | wandb_snapshot | 7.302 | 9.577 | 0.333 | 0.4576 | / | 0.656 | 1.110/0.0207 |
| 13 | `LayerNorm + dropout 0.3` | router_reg_capacity | kaggle_diagnostics | 7.305 | 9.629 | 0.336 |  | 0.230/0.916 | 0.650 | 1.110/0.0209 |
| 14 | `d3 none dropout 0.2 targetT 2.5` | router_temp_depth10 | kaggle_diagnostics | 7.307 | 9.441 | 0.338 | 0.4540 | 0.206/0.917 | 0.653 | 1.109/0.0207 |
| 15 | `d3 none dropout 0.2 targetT 3.0` | router_temp_depth10 | kaggle_diagnostics | 7.319 | 9.457 | 0.350 | 0.4596 | 0.235/0.914 | 0.652 | 1.110/0.0208 |
| 16 | `GroupNorm + dropout 0.2` | router_reg_capacity | kaggle_diagnostics | 7.325 | 9.405 | 0.356 | 0.4565 | 0.201/0.914 | 0.633 | 1.110/0.0207 |
| 17 | `d4 none dropout 0.3` | router_deep4 | kaggle_diagnostics | 7.335 | 9.569 | 0.366 | 0.4607 | 0.147/0.944 | 0.642 | 1.110/0.0207 |
| 18 | `LayerNorm + dropout 0.2` | router_reg_capacity | kaggle_diagnostics | 7.349 | 9.365 | 0.380 | 0.4543 | 0.433/0.837 | 0.646 | 1.109/0.0207 |
| 19 | `GroupNorm + dropout 0.3` | router_reg_capacity | kaggle_diagnostics | 7.381 | 9.532 | 0.412 | 0.4587 | 0.228/0.904 | 0.647 | 1.111/0.0208 |
| 20 | `target T=2.0` | router_smooth5 | wandb_snapshot | 7.396 | 9.613 | 0.427 | 0.4562 | / | 0.643 | 1.110/0.0209 |
| 21 | `LayerNorm + dropout 0.1` | router_reg_capacity | kaggle_diagnostics | 7.407 | 9.593 | 0.438 | 0.4609 | 0.210/0.911 | 0.658 | 1.110/0.0206 |
| 22 | `LayerNorm only` | router_reg_capacity | kaggle_diagnostics | 7.417 | 9.458 | 0.448 | 0.4660 | 0.218/0.907 | 0.650 | 1.110/0.0206 |
| 23 | `d3 none dropout 0.0 targetT 2.5` | router_temp_depth10 | kaggle_diagnostics | 7.417 | 9.599 | 0.448 | 0.4594 | 0.197/0.928 | 0.654 | 1.110/0.0207 |
| 24 | `dropout 0.1` | router_reg_capacity | kaggle_diagnostics | 7.422 | 9.491 | 0.453 | 0.4606 | 0.217/0.915 | 0.662 | 1.110/0.0207 |
| 25 | `d7 none dropout 0.2 targetT 1.25` | router_temp_depth10 | kaggle_diagnostics | 7.475 | 9.719 | 0.506 | 0.4617 | 0.132/0.953 | 0.643 | 1.110/0.0208 |
| 26 | `d3 none dropout 0.0 targetT 3.0` | router_temp_depth10 | kaggle_diagnostics | 7.494 | 9.718 | 0.525 | 0.4538 | 0.195/0.925 | 0.640 | 1.111/0.0207 |
| 27 | `d7 none dropout 0.2 targetT 1.0` | router_temp_depth10 | kaggle_diagnostics | 7.503 | 9.694 | 0.534 | 0.4585 | 0.113/0.952 | 0.655 | 1.110/0.0208 |
| 28 | `target T=1.5` | router_smooth5 | wandb_snapshot | 7.506 | 9.712 | 0.537 | 0.4573 | / | 0.644 | 1.111/0.0208 |
| 29 | `entropy floor 0.25 w=0.05` | router_smooth5 | wandb_snapshot | 7.533 | 9.697 | 0.564 | 0.4548 | / | 0.649 | 1.111/0.0207 |
| 30 | `d3 none dropout 0.2 targetT 2.0` | router_temp_depth10 | kaggle_diagnostics | 7.542 | 9.849 | 0.573 | 0.4618 | 0.181/0.929 | 0.643 | 1.111/0.0208 |

## Batch Summary

| batch | n | best config | best FID128 | mean FID128 |
|---|---:|---|---:|---:|
| router_deep4 | 4 | `d4 none dropout 0.2` | 7.094 | 7.222 |
| router_reg_capacity | 11 | `LayerNorm + dropout 0.2` | 7.147 | 7.303 |
| router_temp_depth10 | 10 | `d5 none dropout 0.2 targetT 1.25` | 7.222 | 7.381 |
| router_smooth5 | 5 | `bridge Beta(2,2)` | 7.277 | 7.403 |

## Group Summary

| group | n | best config | best FID128 | mean FID128 |
|---|---:|---|---:|---:|
| deep_router | 4 | `d4 none dropout 0.2` | 7.094 | 7.222 |
| layernorm | 4 | `LayerNorm + dropout 0.2` | 7.147 | 7.319 |
| dropout | 3 | `dropout 0.3` | 7.153 | 7.250 |
| router_temp_depth10 | 10 | `d5 none dropout 0.2 targetT 1.25` | 7.222 | 7.381 |
| low capacity | 2 | `low-cap plain` | 7.255 | 7.302 |
| smooth_source_router | 5 | `bridge Beta(2,2)` | 7.277 | 7.403 |
| groupnorm | 2 | `GroupNorm + dropout 0.2` | 7.325 | 7.353 |

## Updated Insights

1. Best chung vẫn là `d4 none dropout 0.2` với FID128 `7.094`; batch temp/depth10 chưa vượt được mốc này.
2. Best của batch mới là `d5 none dropout 0.2 targetT 1.25` với FID128 `7.222`, xếp hạng #6 chung.
3. Depth 5 + dropout 0.2 là điểm sáng mới: `T=1.25` và `T=1.0` đều vào top của batch mới, nhưng vẫn kém d4 dropout 0.2 cũ.
4. Depth 7 plain không tốt; LayerNorm cứu d7 đáng kể, nhưng chưa đủ để vượt d5.
5. Target temperature lớn ở d3 cần dropout đi kèm: d3 drop0.2 T=2.5/3.0 khá hơn no-drop, nhưng T=2.0 yếu và high temperature không tạo breakthrough.
6. Các run mới vẫn có `pred/target variance` quanh 0.64-0.67 và flow straightness quanh 1.110, cùng pattern với report cũ; nghĩa là chỉnh router tiếp tục chưa xử lý lõi variance/geometry.

## Files

- JSON: `reports/gmm_tide_router_all_combined_20260611.json`
- CSV: `reports/gmm_tide_router_all_combined_20260611.csv`
- Previous combined source: `reports/gmm_tide_router_all_combined_20260609.json`
- New batch source: `reports/gmm_tide_router_temp_depth10_analysis_20260611.json`
