# Router Regularization + Deep Router Combined

## Scope

- Gộp 11 run router regularization/capacity cũ với 4 run router sâu mới.
- Tất cả run trong bảng này dùng Kaggle diagnostics, không dùng W&B fallback.
- Baseline lịch sử để tham chiếu: `FID128 = 6.969`.

## Overall Ranking

| rank | config | batch | capacity | FID128 | FID32 | delta vs 6.969 | router loss/top1 | pred/target var | flow straight/curv |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `d4 none dropout 0.2` | router_deep4 | d4/h128/m256 | 7.094 | 9.286 | 0.125 | 0.144/0.943 | 0.651 | 1.111/0.0208 |
| 2 | `LayerNorm + dropout 0.2` | router_reg_capacity | d3/h128/m256 | 7.147 | 9.414 | 0.178 | 0.230/0.912 | 0.669 | 1.110/0.0208 |
| 3 | `dropout 0.3` | router_reg_capacity | d3/h128/m256 | 7.153 | 9.419 | 0.184 | 0.204/0.914 | 0.628 | 1.109/0.0208 |
| 4 | `d5 layer_norm dropout 0.2` | router_deep4 | d5/h128/m256 | 7.164 | 9.238 | 0.195 | 0.111/0.964 | 0.661 | 1.111/0.0209 |
| 5 | `dropout 0.2` | router_reg_capacity | d3/h128/m256 | 7.175 | 9.431 | 0.206 | 0.233/0.905 | 0.654 | 1.110/0.0209 |
| 6 | `low-cap plain` | router_reg_capacity | d2/h64/m128 | 7.255 | 9.438 | 0.286 | 0.479/0.820 | 0.674 | 1.110/0.0207 |
| 7 | `d4 layer_norm dropout 0.2` | router_deep4 | d4/h128/m256 | 7.294 | 9.519 | 0.325 | 0.170/0.926 | 0.649 | 1.111/0.0208 |
| 8 | `LayerNorm + dropout 0.3` | router_reg_capacity | d3/h128/m256 | 7.305 | 9.629 | 0.336 | 0.230/0.916 | 0.650 | 1.110/0.0209 |
| 9 | `GroupNorm + dropout 0.2` | router_reg_capacity | d3/h128/m256 | 7.325 | 9.405 | 0.356 | 0.201/0.914 | 0.633 | 1.110/0.0207 |
| 10 | `d4 none dropout 0.3` | router_deep4 | d4/h128/m256 | 7.335 | 9.569 | 0.366 | 0.147/0.944 | 0.642 | 1.110/0.0207 |
| 11 | `LayerNorm + dropout 0.2` | router_reg_capacity | d2/h64/m128 | 7.349 | 9.365 | 0.380 | 0.433/0.837 | 0.646 | 1.109/0.0207 |
| 12 | `GroupNorm + dropout 0.3` | router_reg_capacity | d3/h128/m256 | 7.381 | 9.532 | 0.412 | 0.228/0.904 | 0.647 | 1.111/0.0208 |
| 13 | `LayerNorm + dropout 0.1` | router_reg_capacity | d3/h128/m256 | 7.407 | 9.593 | 0.438 | 0.210/0.911 | 0.658 | 1.110/0.0206 |
| 14 | `LayerNorm only` | router_reg_capacity | d3/h128/m256 | 7.417 | 9.458 | 0.448 | 0.218/0.907 | 0.650 | 1.110/0.0206 |
| 15 | `dropout 0.1` | router_reg_capacity | d3/h128/m256 | 7.422 | 9.491 | 0.453 | 0.217/0.915 | 0.662 | 1.110/0.0207 |

## Direct Comparisons

| comparison | old/baseline | FID128 | new | FID128 | delta new-old |
|---|---|---:|---|---:|---:|
| plain dropout 0.2: d3 -> d4 | `dropout 0.2` | 7.175 | `d4 none dropout 0.2` | 7.094 | -0.081 |
| plain dropout 0.3: d3 -> d4 | `dropout 0.3` | 7.153 | `d4 none dropout 0.3` | 7.335 | 0.183 |
| LayerNorm dropout 0.2: d3 -> d4 | `LayerNorm + dropout 0.2` | 7.147 | `d4 layer_norm dropout 0.2` | 7.294 | 0.147 |
| LayerNorm dropout 0.2: d3 -> d5 | `LayerNorm + dropout 0.2` | 7.147 | `d5 layer_norm dropout 0.2` | 7.164 | 0.017 |

## Group Summary

| group | n | best config | best FID128 | mean FID128 |
|---|---:|---|---:|---:|
| deep_router | 4 | `d4 none dropout 0.2` | 7.094 | 7.222 |
| layernorm | 4 | `LayerNorm + dropout 0.2` | 7.147 | 7.319 |
| dropout | 3 | `dropout 0.3` | 7.153 | 7.250 |
| low capacity | 2 | `low-cap plain` | 7.255 | 7.302 |
| groupnorm | 2 | `GroupNorm + dropout 0.2` | 7.325 | 7.353 |

## Insights

1. Best chung hiện tại là `d4 none dropout 0.2` với FID128 `7.094`. Nó chưa vượt baseline lịch sử `6.969`, nhưng đã vượt best router-reg cũ `LayerNorm + dropout 0.2` ở `7.147` khoảng `0.053`.
2. Depth chỉ có ích trong cấu hình rất cụ thể: `plain dropout 0.2`. So với `d3 dropout 0.2`, lên `d4 dropout 0.2` cải thiện `7.175 -> 7.094`.
3. Tăng depth không phải hướng đơn điệu: `d3 dropout 0.3` tốt hơn `d4 dropout 0.3` (`7.153 -> 7.335`, xấu đi rõ), còn `LayerNorm + dropout 0.2` lên d4 cũng xấu (`7.147 -> 7.294`).
4. `d5 + LayerNorm + dropout 0.2` có router loss/top1 rất tốt (`0.111/0.964`) nhưng FID chỉ `7.164`, kém `d4 plain dropout 0.2`. Vậy router distill tốt hơn vẫn không đồng nghĩa FID tốt hơn.
5. Low-cap vẫn là diagnostic quan trọng: router yếu nhưng FID không sập (`7.255/7.349`). Điều này củng cố rằng bottleneck không chỉ là overfit/capacity của router.
6. GroupNorm tiếp tục không đáng mở rộng: best GroupNorm `7.325`, kém dropout-only và kém deep d4 dropout 0.2.
7. `pred/target variance` của mọi nhóm vẫn quanh `0.63-0.67`; flow straightness quanh `1.109-1.111`. Những sweep router này không xử lý triệt để mismatch variance/geometry của FM source.

## Practical Takeaway

- Candidate nên giữ: `d4 none dropout 0.2`.
- Candidate phụ nếu cần repeat: `d5 LayerNorm dropout 0.2`, vì FID32 tốt nhất trong nhóm deep nhưng FID128 chưa bằng d4 plain.
- Không nên mở rộng: dropout 0.3 với router sâu, GroupNorm, LayerNorm-only/LayerNorm+dropout nhẹ.
- Next ablation hợp lý: lấy `d4 none dropout 0.2` ghép với các setting đã tốt về time/source geometry, thay vì tiếp tục tăng depth/norm.

## Files

- JSON: `reports/gmm_tide_router_reg_deep_combined_20260609.json`
- CSV: `reports/gmm_tide_router_reg_deep_combined_20260609.csv`
- Old source: `reports/gmm_tide_router_regularization_analysis_20260608.json`
- Deep source: `reports/gmm_tide_router_deep4_analysis_20260609.json`
