# Router Experiments Combined

## Scope

- Gộp 11 run router regularization/capacity, 4 run router-deep và 5 run router-smooth/source-geometry.
- `metric_source` được giữ rõ: deep/reg là Kaggle diagnostics; smooth5 hiện là W&B snapshot vì Kaggle diagnostics chưa publish khi kéo.
- Baseline lịch sử để tham chiếu: `FID128 = 6.969`.

## Overall Ranking

| rank | config | batch | source | FID128 | FID32 | delta vs 6.969 | valid loss | router loss/top1 | pred/target var | flow straight/curv |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `d4 none dropout 0.2` | router_deep4 | kaggle_diagnostics | 7.094 | 9.286 | 0.125 | 0.4562 | 0.144/0.943 | 0.651 | 1.111/0.0208 |
| 2 | `LayerNorm + dropout 0.2` | router_reg_capacity | kaggle_diagnostics | 7.147 | 9.414 | 0.178 | 0.4664 | 0.230/0.912 | 0.669 | 1.110/0.0208 |
| 3 | `dropout 0.3` | router_reg_capacity | kaggle_diagnostics | 7.153 | 9.419 | 0.184 | 0.4684 | 0.204/0.914 | 0.628 | 1.109/0.0208 |
| 4 | `d5 layer_norm dropout 0.2` | router_deep4 | kaggle_diagnostics | 7.164 | 9.238 | 0.195 | 0.4610 | 0.111/0.964 | 0.661 | 1.111/0.0209 |
| 5 | `dropout 0.2` | router_reg_capacity | kaggle_diagnostics | 7.175 | 9.431 | 0.206 | 0.4611 | 0.233/0.905 | 0.654 | 1.110/0.0209 |
| 6 | `low-cap plain` | router_reg_capacity | kaggle_diagnostics | 7.255 | 9.438 | 0.286 | 0.4479 | 0.479/0.820 | 0.674 | 1.110/0.0207 |
| 7 | `bridge Beta(2,2)` | router_smooth5 | wandb_snapshot | 7.277 | 9.435 | 0.308 | 0.4592 |  | 0.650 | 1.110/0.0209 |
| 8 | `d4 layer_norm dropout 0.2` | router_deep4 | kaggle_diagnostics | 7.294 | 9.519 | 0.325 | 0.4644 | 0.170/0.926 | 0.649 | 1.111/0.0208 |
| 9 | `sample_topk source + layer_norm + dropout 0.2` | router_smooth5 | wandb_snapshot | 7.302 | 9.577 | 0.333 | 0.4576 |  | 0.656 | 1.110/0.0207 |
| 10 | `LayerNorm + dropout 0.3` | router_reg_capacity | kaggle_diagnostics | 7.305 | 9.629 | 0.336 |  | 0.230/0.916 | 0.650 | 1.110/0.0209 |
| 11 | `GroupNorm + dropout 0.2` | router_reg_capacity | kaggle_diagnostics | 7.325 | 9.405 | 0.356 | 0.4565 | 0.201/0.914 | 0.633 | 1.110/0.0207 |
| 12 | `d4 none dropout 0.3` | router_deep4 | kaggle_diagnostics | 7.335 | 9.569 | 0.366 | 0.4607 | 0.147/0.944 | 0.642 | 1.110/0.0207 |
| 13 | `LayerNorm + dropout 0.2` | router_reg_capacity | kaggle_diagnostics | 7.349 | 9.365 | 0.380 | 0.4543 | 0.433/0.837 | 0.646 | 1.109/0.0207 |
| 14 | `GroupNorm + dropout 0.3` | router_reg_capacity | kaggle_diagnostics | 7.381 | 9.532 | 0.412 | 0.4587 | 0.228/0.904 | 0.647 | 1.111/0.0208 |
| 15 | `target T=2.0` | router_smooth5 | wandb_snapshot | 7.396 | 9.613 | 0.427 | 0.4562 |  | 0.643 | 1.110/0.0209 |
| 16 | `LayerNorm + dropout 0.1` | router_reg_capacity | kaggle_diagnostics | 7.407 | 9.593 | 0.438 | 0.4609 | 0.210/0.911 | 0.658 | 1.110/0.0206 |
| 17 | `LayerNorm only` | router_reg_capacity | kaggle_diagnostics | 7.417 | 9.458 | 0.448 | 0.4660 | 0.218/0.907 | 0.650 | 1.110/0.0206 |
| 18 | `dropout 0.1` | router_reg_capacity | kaggle_diagnostics | 7.422 | 9.491 | 0.453 | 0.4606 | 0.217/0.915 | 0.662 | 1.110/0.0207 |
| 19 | `target T=1.5` | router_smooth5 | wandb_snapshot | 7.506 | 9.712 | 0.537 | 0.4573 |  | 0.644 | 1.111/0.0208 |
| 20 | `entropy floor 0.25 w=0.05` | router_smooth5 | wandb_snapshot | 7.533 | 9.697 | 0.564 | 0.4548 |  | 0.649 | 1.111/0.0207 |

## Group Summary

| group | n | best config | best FID128 | mean FID128 |
|---|---:|---|---:|---:|
| deep_router | 4 | `d4 none dropout 0.2` | 7.094 | 7.222 |
| layernorm | 4 | `LayerNorm + dropout 0.2` | 7.147 | 7.319 |
| dropout | 3 | `dropout 0.3` | 7.153 | 7.250 |
| low capacity | 2 | `low-cap plain` | 7.255 | 7.302 |
| smooth_source_router | 5 | `bridge Beta(2,2)` | 7.277 | 7.403 |
| groupnorm | 2 | `GroupNorm + dropout 0.2` | 7.325 | 7.353 |

## Main Insights

1. Best chung vẫn là `d4 none dropout 0.2` với FID128 `7.094`. Đây là cải thiện thật so với batch router-reg cũ, nhưng vẫn chưa vượt baseline lịch sử `6.969`.
2. Nhóm smooth/source-geometry mới không vượt deep d4 dropout 0.2. Best smooth là `bridge Beta(2,2)` với FID128 `7.277`, xếp sau low-cap plain `7.255` và sau d4/d5 deep tốt nhất.
3. `sample_topk + layer_norm + dropout 0.2` ở `7.302` gần bridge nhưng không vượt các cấu hình dropout/router-depth tốt. Nó đáng giữ như hướng source stochastic, nhưng chưa phải default.
4. Target temperature (`T=1.5/2.0`) và entropy floor đều kém. Entropy floor có valid loss thấp nhưng FID tệ, tiếp tục xác nhận valid FM loss không đủ để rank ảnh.
5. Router sâu giúp khi chỉ thêm depth vừa phải + dropout 0.2. Nhưng thêm LayerNorm hoặc dropout 0.3 không ổn định, và router metric tốt hơn không đảm bảo FID tốt hơn.
6. Các nhóm vẫn chung một bệnh: `pred/target variance` quanh `0.64-0.67`, flow straightness quanh `1.109-1.111`. Nên vấn đề chính nhiều khả năng nằm ở source geometry/variance/time schedule hơn là router capacity thuần.

## Practical Decision

- Candidate chính: `d4 none dropout 0.2`.
- Candidate phụ để kết hợp với hướng geometry/time: `bridge Beta(2,2)` và `sample_topk + LN + dropout 0.2`.
- Không ưu tiên tiếp: entropy floor, target-temperature-only, GroupNorm, LayerNorm-only, dropout 0.1, router sâu dropout 0.3.
- Batch tiếp theo nên ghép `d4 none dropout 0.2` với các time/source setting tốt nhất hiện có, thay vì tiếp tục chỉ chỉnh router.

## Files

- JSON: `reports/gmm_tide_router_all_combined_20260609.json`
- CSV: `reports/gmm_tide_router_all_combined_20260609.csv`
- Sources: `reports/gmm_tide_router_regularization_analysis_20260608.json`, `reports/gmm_tide_router_deep4_analysis_20260609.json`, `reports/gmm_tide_router_smooth5_wandb_20260609.json`
