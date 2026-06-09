# GMM Distill / Router Report

Mục này chỉ nhìn phần distill `q_theta(k|x)` từ posterior GMM `q_GMM(k|x)`, tách khỏi FID/FM.

## Metric Meanings

- `router_valid/loss`: loss distill trên validation, hiện gần như KL từ `q_GMM` sang `q_theta`; thấp hơn là router match GMM tốt hơn.
- `router_valid/router/kl_to_gmm`: KL target `q_GMM(k|x)` so với output router; thấp hơn tốt hơn.
- `router_valid/router/cross_entropy`: cross entropy với target GMM; gồm entropy target + KL, nên target càng mềm thì CE tự nhiên cao hơn.
- `router_valid/router/top1_agreement`: tỉ lệ argmax của router trùng cluster argmax của GMM; cao hơn tốt hơn, nhưng không đo đúng xác suất mềm.
- `router_valid/router/top1_prob_mean`: độ tự tin trung bình của router ở cụm top1; quá thấp là mơ hồ, quá cao trong khi agreement thấp là tự tin sai.
- `router_valid/router/usage_entropy_normalized`: entropy phân bố cụm mà router dùng, chuẩn hóa theo `log(K)`; gần 1 nghĩa là dùng nhiều cụm đều, thấp nghĩa là collapse vào ít cụm.
- `router_valid/router/assign_max_frac`: fraction lớn nhất của một cụm trong batch; càng cao càng có nguy cơ cụm dominate. Với K=16 kỳ vọng đều là 0.0625, K=32 là 0.03125, nhưng batch nhỏ nên thực tế cao hơn.
- `router_valid/router/num_unique_clusters`: số cụm xuất hiện trong batch validation; càng gần K càng ít collapse.
- `router_valid/router/target_entropy` và `target_top1_prob_mean`: độ cứng/mềm của posterior GMM target. Target entropy gần 0 và top1 gần 1 nghĩa là GMM assignment gần hard.
- `router_overfit/loss_gap = valid_loss - train_loss`: dương lớn là overfit train; âm nghĩa validation dễ hơn hoặc sample noise.
- `training/router/kl_to_gmm_base`, `top1_agreement_to_gmm_base`, `topk_mass`: router được dùng trong FM còn bám GMM base không; top1=1 và KL rất nhỏ nghĩa là khi train FM, router hầu như giữ nguyên quyết định GMM.

## Distill Table

| rank FID | run | K | router valid loss | valid top1 | usage entropy | unique | max frac | overfit gap | FM-time KL to GMM | FM-time top1 | FID128 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tide-k16-top2-softv0p75-s128-dir512` | 16 | 0.185 | 0.930 | 0.948 | 15.6 | 0.134 | -0.033 | 0.0007 | 1.000 | 6.969 |
| 2 | `tide-k32-top2-softv0p75-s128-dir001` | 32 | 0.240 | 0.907 | 0.904 | 26.9 | 0.096 | -0.043 | 0.0010 | 1.000 | 7.118 |
| 3 | `tide-k16-top2-g108` | 16 | 0.176 | 0.938 | 0.954 | 15.5 | 0.123 |  | 0.0005 | 1.000 | 7.219 |
| 4 | `tide-k32-top2-g136-none-hardv0p5` | 32 | 0.261 | 0.899 | 0.903 | 26.7 | 0.092 | 0.016 | 0.0016 | 1.000 | 7.256 |
| 5 | `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` | 16 | 0.195 | 0.921 | 0.949 | 15.7 | 0.131 | 0.040 | 0.0007 | 1.000 | 7.315 |
| 6 | `tide-mc-r1-k16-top2-soft075-dir512-mix` | 16 | 0.194 | 0.922 | 0.939 | 15.6 | 0.131 | -0.060 | 0.0013 | 1.000 | 7.466 |
| 7 | `tide-k32-top2-g145-dir001-hardv0p5` | 32 | 0.241 | 0.909 | 0.924 | 28.5 | 0.082 | -0.159 | 0.0011 | 1.000 | 7.528 |
| 8 | `tide-k32-top4-g136-none-hardv0p5` | 32 | 0.232 | 0.911 | 0.913 | 27.6 | 0.090 | -0.088 | 0.0013 | 1.000 | 7.598 |
| 9 | `tide-mc-r2-k32-top2-soft075-dir001-mix` | 32 | 0.263 | 0.903 | 0.869 | 24.1 | 0.106 | 0.070 | 0.0015 | 1.000 | 7.617 |
| 10 | `tide-mc-r3-k32-top2-none-hard05-x1cont10` | 32 | 0.239 | 0.905 | 0.918 | 27.4 | 0.084 | -0.169 | 0.0017 | 1.000 | 7.622 |
| 11 | `tide-k32-top2-g146-dir001-hardv1p0` | 32 | 0.275 | 0.886 | 0.912 | 27.3 | 0.086 | 0.054 | 0.0024 | 1.000 | 7.623 |
| 12 | `tide-mc-r4-k32-top2-dir001-hard05-mix` | 32 | 0.275 | 0.892 | 0.882 | 25.2 | 0.107 | -0.049 | 0.0039 | 1.000 | 7.740 |
| 13 | `tide-mc-r2-k32-top2-soft075-dir001-mixcont10` | 32 | 0.276 | 0.887 | 0.915 | 27.7 | 0.092 | -0.141 | 0.0042 | 1.000 | 7.747 |
| 14 | `tide-mc-r1-k16-top2-soft075-dir512-x1cont10` | 16 | 0.171 | 0.930 | 0.946 | 15.6 | 0.133 | 0.040 | 0.0005 | 1.000 | 7.797 |
| 15 | `tide-mc-r3-k32-top2-none-hard05-mix` | 32 | 0.252 | 0.891 | 0.875 | 24.6 | 0.103 | -0.082 | 0.0030 | 1.000 | 7.806 |
| 16 | `tide-mc-r2-k32-top2-soft075-dir001-x1cont10` | 32 | 0.264 | 0.900 | 0.924 | 28.4 | 0.091 | 0.169 | 0.0016 | 1.000 | 7.815 |
| 17 | `tide-mc-r4-k32-top2-dir001-hard05-x1cont10` | 32 | 0.238 | 0.903 | 0.915 | 27.7 | 0.084 | 0.041 | 0.0014 | 1.000 | 7.880 |
| 18 | `tide-mc-r3-k32-top2-none-hard05-mixcont10` | 32 | 0.218 | 0.921 | 0.907 | 27.3 | 0.099 | -0.045 | 0.0019 | 1.000 | 7.892 |
| 19 | `tide-mc-r4-k32-top2-dir001-hard05-mixcont10` | 32 | 0.263 | 0.911 | 0.898 | 25.8 | 0.086 | -0.084 | 0.0017 | 1.000 | 8.033 |

## Distill Insights

- valid loss: range `0.1706 -> 0.2765`, mean `0.2348`.
- valid top1: range `0.8857 -> 0.9375`, mean `0.9087`.
- usage entropy: range `0.8692 -> 0.9539`, mean `0.9154`.
- assign max frac: range `0.0820 -> 0.1338`, mean `0.1025`.
- FM-time KL: range `0.0005 -> 0.0042`, mean `0.0017`.
- Router distill nhìn chung không collapse: valid usage entropy thường khoảng `0.86-0.94`, số cụm dùng trong batch cao, và top1 agreement quanh `0.89-0.93` ở các run có summary.
- Trong phase FM, `training/router/top1_agreement_to_gmm_base` thường bằng `1.0`, `training/router/kl_to_gmm_base` rất nhỏ khoảng `1e-3`; nghĩa là sau distill, router gần như copy quyết định GMM base khi sinh top-k.
- Correlation router-valid-top1 với FID không mạnh; những run agreement cao vẫn có thể FID xấu. Vì vậy distill chưa phải bottleneck chính, bottleneck hiện nghiêng về chất lượng GMM/source distribution và FM variance.
- Cần chú ý target GMM khá hard (`target_entropy` rất thấp, `target_top1_prob_mean` gần 1). Nếu muốn top-k mixture có tác dụng hơn, có thể cần soften posterior/temperature hoặc train router trên target mềm hơn thay vì chỉ hard top1.

## Correlation With FID128

| metric | corr vs FID128 | note |
|---|---:|---|
| `_router_valid_loss` | 0.423 | dương nghĩa loss router cao đi kèm FID xấu hơn |
| `_router_valid_top1` | -0.386 | âm nghĩa agreement cao đi kèm FID tốt hơn |
| `_router_loss_gap` | 0.012 | dương lớn là overfit distill |
| `_router_valid_usage_entropy` | -0.426 | âm nếu dùng cụm đều hơn giúp FID |
| `_router_assign_max_frac` | -0.440 | dương nếu cụm dominate làm FID xấu |
| `_train_router_kl_base` | 0.453 | FM-time router lệch GMM base |
| `_train_router_topk_mass` | -0.338 | mass top-k được giữ lại |
| `_pred_target_var_ratio` | -0.164 | FM pred/target variance ratio |
