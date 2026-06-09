# Router Regularization / Capacity Analysis

## Scope

- Pulled diagnostics-only outputs for 4 `uniform4_more` jobs and 2 `capacity2` jobs on 2026-06-08.
- Reused already-downloaded diagnostics for the 5 `uniform5` jobs.
- No checkpoints were downloaded; output roots are `outputs/kaggle/router_reg_uniform5_20260607` and `outputs/kaggle/router_reg_more_capacity_20260608`.
- Historical baseline for comparison: `tide-k16-top2-softv0p75-s128-dir512`, FID128 `6.969`, router valid loss `0.185`, top1 `0.930`.

## Ranking By FID128

| rank | regularization | capacity | run | step | evals | FID128 | delta vs 6.969 | FID32 | router loss/top1 | usage H | pred/target var | flow straight/curv |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | LayerNorm + dropout 0.2 | d3/h128/m256 | `tide-routerreg-uniform-ln-drop02-k16-top2-soft075-dir512` | 351400 | 7 | 7.147 | 0.178 | 9.414 | 0.2301/0.912 | 0.944 | 0.669 | 1.110/0.0208 |
| 2 | dropout 0.3 | d3/h128/m256 | `tide-routerreg-uniform-drop03-k16-top2-soft075-dir512` | 356200 | 7 | 7.153 | 0.184 | 9.419 | 0.2044/0.914 | 0.950 | 0.628 | 1.109/0.0208 |
| 3 | dropout 0.2 | d3/h128/m256 | `tide-routerreg-uniform-drop02-k16-top2-soft075-dir512` | 355700 | 7 | 7.175 | 0.206 | 9.431 | 0.2325/0.905 | 0.949 | 0.654 | 1.110/0.0209 |
| 4 | low-cap plain | d2/h64/m128 | `tide-routercap-uniform-low-k16-top2-soft075-dir512` | 350000 | 6 | 7.255 | 0.286 | 9.438 | 0.4789/0.820 | 0.940 | 0.674 | 1.110/0.0207 |
| 5 | LayerNorm + dropout 0.3 | d3/h128/m256 | `tide-routerreg-uniform-ln-drop03-k16-top2-soft075-dir512` | 350000 | 7 | 7.305 | 0.336 | 9.629 | 0.2297/0.916 | 0.943 | 0.650 | 1.110/0.0209 |
| 6 | GroupNorm + dropout 0.2 | d3/h128/m256 | `tide-routerreg-uniform-gn-drop02-k16-top2-soft075-dir512` | 350000 | 6 | 7.325 | 0.356 | 9.405 | 0.2012/0.914 | 0.946 | 0.633 | 1.110/0.0207 |
| 7 | LayerNorm + dropout 0.2 | d2/h64/m128 | `tide-routercap-uniform-low-ln-drop02-k16-top2-soft075-dir512` | 350000 | 6 | 7.349 | 0.380 | 9.365 | 0.4332/0.837 | 0.948 | 0.646 | 1.109/0.0207 |
| 8 | GroupNorm + dropout 0.3 | d3/h128/m256 | `tide-routerreg-uniform-gn-drop03-k16-top2-soft075-dir512` | 350000 | 6 | 7.381 | 0.412 | 9.532 | 0.2283/0.904 | 0.949 | 0.647 | 1.111/0.0208 |
| 9 | LayerNorm + dropout 0.1 | d3/h128/m256 | `tide-routerreg-uniform-ln-drop01-k16-top2-soft075-dir512` | 350000 | 6 | 7.407 | 0.438 | 9.593 | 0.2105/0.911 | 0.945 | 0.658 | 1.110/0.0206 |
| 10 | LayerNorm only | d3/h128/m256 | `tide-routerreg-uniform-ln-k16-top2-soft075-dir512` | 350000 | 6 | 7.417 | 0.448 | 9.458 | 0.2176/0.907 | 0.947 | 0.650 | 1.110/0.0206 |
| 11 | dropout 0.1 | d3/h128/m256 | `tide-routerreg-uniform-drop01-k16-top2-soft075-dir512` | 350000 | 6 | 7.422 | 0.453 | 9.491 | 0.2171/0.915 | 0.945 | 0.662 | 1.110/0.0207 |

## Group Summary

| group | n | best config | best FID128 | mean FID128 | mean router loss | mean top1 | note |
|---|---:|---|---:|---:|---:|---:|---|
| dropout | 3 | dropout 0.3 | 7.153 | 7.250 | 0.2180 | 0.911 | Dropout alone is the strongest family in this sweep. |
| groupnorm | 2 | GroupNorm + dropout 0.2 | 7.325 | 7.353 | 0.2148 | 0.909 | GroupNorm does not beat dropout alone or LayerNorm+dropout 0.2. |
| layernorm | 4 | LayerNorm + dropout 0.2 | 7.147 | 7.319 | 0.2220 | 0.912 | LayerNorm helps only around dropout 0.2; too much dropout hurts. |
| low capacity | 2 | low-cap plain | 7.255 | 7.302 | 0.4561 | 0.829 | Lower capacity greatly worsens distill metrics but FID degrades only moderately. |

## Main Insights

1. Không cấu hình router regularization/capacity nào vượt baseline lịch sử FID128 `6.969`. Best mới là LayerNorm+dropout `0.2` ở `7.147`, rất sát dropout `0.3` ở `7.153` và dropout `0.2` ở `7.175`.
2. Dropout có vùng hữu ích: `0.1` yếu (`7.422`), `0.2/0.3` tốt hơn nhiều (`7.175/7.153`). Chênh giữa `0.2` và `0.3` nhỏ, nên nên coi là tín hiệu yếu/chờ repeat.
3. LayerNorm không phải mặc định tốt: LayerNorm-only `7.417`, LayerNorm+dropout `0.1` `7.407`, LayerNorm+dropout `0.2` tốt nhất, nhưng LayerNorm+dropout `0.3` tụt xuống `7.305`.
4. GroupNorm chưa đáng mở rộng: GroupNorm+dropout `0.2/0.3` là `7.325/7.381`, kém dropout-only và kém LayerNorm+dropout `0.2`.
5. Hạ capacity cho thấy router overfit không phải nguyên nhân duy nhất. Router loss/top1 xấu mạnh (`0.43-0.48`, top1 `0.82-0.84`) nhưng FID chỉ tụt vừa phải (`7.26-7.35`). Vì vậy distill q_theta khớp GMM tốt không đủ để đảm bảo FID tốt.
6. `pred/target var` vẫn quanh `0.63-0.67`, tức DiT vẫn under-dispersed so với target velocity. Router regularization không xử lý gốc vấn đề variance/source-geometry.
7. Flow straightness/curvature của các run khá sát nhau; batch này không tách biệt rõ bằng flow metric, FID vẫn là tín hiệu downstream chính.

## Recommendation

- Giữ lại hai candidate nếu muốn retry/repeat: dropout `0.3` và LayerNorm+dropout `0.2`.
- Không mở rộng GroupNorm thêm ở source/time-sampling này.
- Không dùng low-capacity router làm default; nó hữu ích như diagnostic overfit, không phải hướng cải thiện chính.
- Nên quay lại source geometry/time schedule/FM config thay vì tiếp tục sweep normalization router rộng hơn.

## Files

- JSON: `reports/gmm_tide_router_regularization_analysis_20260608.json`
- CSV: `reports/gmm_tide_router_regularization_analysis_20260608.csv`
- Download report: `reports/gmm_tide_router_reg_more_capacity_download_20260608.json`
- Check report: `reports/gmm_tide_router_reg_more_capacity_check_20260608.md`
