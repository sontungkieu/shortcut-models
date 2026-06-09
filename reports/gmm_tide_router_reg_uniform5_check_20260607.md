# Router Regularization Uniform5 Check

## Kết Luận Nhanh

- Đã tải được diagnostics cho cả 5 job; không tải checkpoint/model artifact.
- Không thấy traceback/runtime error trong diagnostics. Các run dừng quanh `350k/500k` step, khả năng do giới hạn thời gian Kaggle, không phải lỗi code.
- Cả 5 cấu hình dropout/LayerNorm đều chưa vượt baseline uniform cũ `tide-k16-top2-softv0p75-s128-dir512` có FID128 `6.969`.
- Trong nhóm này, tốt nhất là `LayerNorm + dropout 0.2` với FID128 `7.147`, kế tiếp là `dropout 0.2` với FID128 `7.175`.
- Dropout 0.2 giúp FID hơn dropout 0.1; LayerNorm đơn lẻ không giúp. Nhưng router valid loss/top1 lại xấu hơn baseline, nên regularization đang cải thiện nhẹ downstream trong nhóm này chứ không cải thiện distill quality.

## Bảng Chính

| rank | regularization | run | step | evals | FID128 best@step | FID32 best | valid loss | router loss/top1 | usage entropy | overfit loss gap | pred/target var | x0/x1 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | LayerNorm + dropout 0.2 | `tide-routerreg-uniform-ln-drop02-k16-top2-soft075-dir512` | 351400 | 7 | 7.147 @ 350000 | 9.414 | 0.4664 | 0.2301/0.912 | 0.944 | -0.0269 | 0.669 | 0.991 |
| 2 | dropout 0.2 | `tide-routerreg-uniform-drop02-k16-top2-soft075-dir512` | 355700 | 7 | 7.175 @ 350000 | 9.431 | 0.4611 | 0.2325/0.905 | 0.949 | 0.1129 | 0.654 | 0.998 |
| 3 | LayerNorm + dropout 0.1 | `tide-routerreg-uniform-ln-drop01-k16-top2-soft075-dir512` | 350000 | 6 | 7.407 @ 300000 | 9.669 | 0.4609 | 0.2105/0.911 | 0.945 | 0.0623 | 0.658 | 0.995 |
| 4 | LayerNorm | `tide-routerreg-uniform-ln-k16-top2-soft075-dir512` | 350000 | 6 | 7.417 @ 300000 | 9.738 | 0.4660 | 0.2176/0.907 | 0.947 | -0.0208 | 0.650 | 0.983 |
| 5 | dropout 0.1 | `tide-routerreg-uniform-drop01-k16-top2-soft075-dir512` | 350000 | 6 | 7.422 @ 300000 | 9.618 | 0.4606 | 0.2171/0.915 | 0.945 | 0.0662 | 0.662 | 1.007 |

## So Với Baseline Cũ

| run | FID128 | delta vs 6.969 | router valid loss | delta vs 0.185 | router top1 | delta vs 0.930 |
|---|---:|---:|---:|---:|---:|---:|
| `tide-routerreg-uniform-ln-drop02-k16-top2-soft075-dir512` | 7.147 | +0.178 | 0.2301 | +0.0451 | 0.912 | -0.018 |
| `tide-routerreg-uniform-drop02-k16-top2-soft075-dir512` | 7.175 | +0.206 | 0.2325 | +0.0475 | 0.905 | -0.025 |
| `tide-routerreg-uniform-ln-drop01-k16-top2-soft075-dir512` | 7.407 | +0.438 | 0.2105 | +0.0255 | 0.911 | -0.019 |
| `tide-routerreg-uniform-ln-k16-top2-soft075-dir512` | 7.417 | +0.448 | 0.2176 | +0.0326 | 0.907 | -0.023 |
| `tide-routerreg-uniform-drop01-k16-top2-soft075-dir512` | 7.422 | +0.453 | 0.2171 | +0.0321 | 0.915 | -0.015 |

## Nhận Xét

- `dropout=0.2` tốt hơn `dropout=0.1` cả khi có và không có LayerNorm: FID128 giảm khoảng `0.24-0.25` trong từng cặp.
- `LayerNorm` đơn lẻ gần như ngang hoặc kém dropout nhẹ: FID128 `7.417`, không tạo lợi ích rõ.
- `LayerNorm + dropout=0.2` là tốt nhất trong batch này, nhưng vẫn kém baseline cũ khoảng `+0.178` FID128.
- Router overfit gap giảm/âm ở LayerNorm, nhưng FID không tự động tốt hơn. Điều này tiếp tục cho thấy metric distill/overfit của router chỉ là diagnostic, không đủ để rank ảnh sinh.
- `pred/target variance` khoảng `0.65-0.67`; model vẫn under-dispersed so với target velocity. Regularization router không sửa vấn đề variance này.
