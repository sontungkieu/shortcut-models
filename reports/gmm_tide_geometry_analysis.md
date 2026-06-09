# GMM/TIDE Geometry Analysis

Mục tiêu: kiểm tra giả thuyết source GMM/TIDE có thể giữ khoảng cách nhưng phá hình học góc của latent.

## Kết Luận Ngắn

- Có 480 `gmm_metrics.json` có đủ `center_distance` và `component_variance_trace`.
- 476/480 run có `center_distance_mean / sqrt(component_variance_trace_mean) < 1`. Khi tỷ lệ này nhỏ hơn 1, khoảng cách tâm component nhỏ hơn độ nhiễu RMS trong component; source sample dễ bị nhiễu hướng và cosine/angle trở thành metric bắt buộc.
- Tỷ lệ này có min/median/max = `0.4343` / `0.7049` / `1.4886`.
- Có 3 `gmm_stats.npz` đủ `mu/var`; mean center/noise SNR = `0.6487`, nearest-2 center cosine mean = `0.4741`.
- SNR tâm/noise thấp nghĩa là mỗi Gaussian component không định nghĩa một hướng latent sắc; weighted top-k càng dễ sinh `x0_tide` nằm giữa nhiều hướng.
- Không tìm thấy geometry cosine metrics trong các `train_metrics.csv` cũ. Điều này xác nhận các run cũ chưa trực tiếp đo góc `x0`-`x1` hoặc angular dispersion top-k.

## GMM Center/Noise Proxy Tệ Nhất

| run | K | center/noise | min center/noise | center mean | noise mean | pi entropy | dead valid | count ratio | floor hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gmm-k4-floor1p25-none-s0p0` | 4 | 0.4343 | 0.3066 | 34.7459 | 80.0000 | 0.9989 | 0 | 1.0984 | 1.0000 |
| `gmm-k4-floor1p25-dirichlet-s0p01` | 4 | 0.4348 | 0.3066 | 34.7856 | 80.0000 | 0.9989 | 0 | 1.1011 | 1.0000 |
| `gmm-k4-floorv2p0-kl-s512p0-raw-hardv2p0` | 4 | 0.4403 | 0.2757 | 32.5945 | 74.0263 | 0.9990 | 0 | 1.1027 | 1.0000 |
| `gmm-k4-floorv2p0-dirichlet-s0p01-raw-hardv2p0` | 4 | 0.4403 | 0.2773 | 32.5936 | 74.0224 | 0.9987 | 0 | 1.1226 | 1.0000 |
| `gmm-k4-floorv2p0-none-s0p0-raw-hardv2p0-em100` | 4 | 0.4404 | 0.2783 | 32.5960 | 74.0229 | 0.9989 | 0 | 1.0908 | 1.0000 |
| `gmm-k4-floorv2p0-kl-s2048p0-raw-hardv2p0-em100` | 4 | 0.4404 | 0.2775 | 32.5992 | 74.0239 | 0.9989 | 0 | 1.1238 | 1.0000 |
| `gmm-k4-floorv2p0-none-s0p0-raw-hardv2p0` | 4 | 0.4404 | 0.2779 | 32.5989 | 74.0222 | 0.9988 | 0 | 1.1050 | 1.0000 |
| `gmm-k4-floorv2p0-kl-s512p0-raw-hardv2p0-em100` | 4 | 0.4404 | 0.2781 | 32.6029 | 74.0229 | 0.9991 | 0 | 1.0870 | 1.0000 |
| `gmm-k4-floorv2p0-dirichlet-s512p0-raw-hardv2p0-em100` | 4 | 0.4405 | 0.2778 | 32.6049 | 74.0222 | 0.9990 | 0 | 1.0995 | 1.0000 |
| `gmm-k4-floorv2p0-dirichlet-s512p0-raw-hardv2p0` | 4 | 0.4406 | 0.2775 | 32.6105 | 74.0217 | 0.9989 | 0 | 1.1138 | 1.0000 |
| `gmm-k4-floorv2p0-dirichlet-s0p01-raw-hardv2p0-em100` | 4 | 0.4406 | 0.2774 | 32.6109 | 74.0195 | 0.9988 | 0 | 1.1155 | 1.0000 |
| `gmm-k4-floorv2p0-kl-s2048p0-raw-hardv2p0` | 4 | 0.4406 | 0.2791 | 32.6170 | 74.0238 | 0.9990 | 0 | 1.1004 | 1.0000 |
| `gmm-k8-floorv2p0-kl-s512p0-raw-hardv2p0` | 8 | 0.4509 | 0.2895 | 33.3789 | 74.0215 | 0.9951 | 0 | 1.5801 | 1.0000 |
| `gmm-k8-floorv2p0-none-s0p0-raw-hardv2p0` | 8 | 0.4516 | 0.2570 | 33.4289 | 74.0215 | 0.9926 | 0 | 1.6830 | 0.9999 |
| `gmm-k8-floorv2p0-dirichlet-s512p0-raw-hardv2p0` | 8 | 0.4534 | 0.2804 | 33.5630 | 74.0248 | 0.9961 | 0 | 1.6192 | 1.0000 |
| `gmm-k8-floorv2p0-dirichlet-s512p0-raw-hardv2p0-em100` | 8 | 0.4541 | 0.2963 | 33.6161 | 74.0260 | 0.9956 | 0 | 1.5718 | 1.0000 |
| `gmm-k8-floorv2p0-dirichlet-s0p01-raw-hardv2p0-em100` | 8 | 0.4558 | 0.2898 | 33.7396 | 74.0233 | 0.9932 | 0 | 1.9528 | 1.0000 |
| `gmm-k8-floorv2p0-kl-s2048p0-raw-hardv2p0` | 8 | 0.4578 | 0.2728 | 33.8888 | 74.0240 | 0.9904 | 0 | 2.0519 | 1.0000 |
| `gmm-k8-floorv2p0-kl-s512p0-raw-hardv2p0-em100` | 8 | 0.4580 | 0.2853 | 33.9062 | 74.0230 | 0.9935 | 0 | 1.7038 | 1.0000 |
| `gmm-k8-floorv2p0-kl-s2048p0-raw-hardv2p0-em100` | 8 | 0.4584 | 0.2658 | 33.9305 | 74.0248 | 0.9930 | 0 | 1.8010 | 1.0000 |
| `gmm-k8-floorv2p0-none-s0p0-raw-hardv2p0-em100` | 8 | 0.4606 | 0.2663 | 34.0973 | 74.0231 | 0.9902 | 0 | 1.8868 | 1.0000 |
| `gmm-k8-floorv2p0-dirichlet-s0p01-raw-hardv2p0` | 8 | 0.4626 | 0.2895 | 34.2403 | 74.0219 | 0.9921 | 0 | 1.9006 | 1.0000 |
| `gmm-k16-floorv2p0-none-s0p0-raw-hardv2p0` | 16 | 0.4734 | 0.2475 | 35.0402 | 74.0215 | 0.9919 | 0 | 2.0899 | 0.9999 |
| `gmm-k16-floorv2p0-dirichlet-s512p0-raw-hardv2p0-em100` | 16 | 0.4757 | 0.2362 | 35.2170 | 74.0248 | 0.9949 | 0 | 2.0819 | 0.9998 |
| `gmm-k16-floorv2p0-kl-s2048p0-raw-hardv2p0` | 16 | 0.4766 | 0.2418 | 35.2786 | 74.0217 | 0.9924 | 0 | 2.0723 | 0.9999 |
| `gmm-k16-floorv2p0-kl-s512p0-raw-hardv2p0` | 16 | 0.4773 | 0.2367 | 35.3324 | 74.0236 | 0.9820 | 0 | 3.9194 | 0.9998 |
| `gmm-k16-floorv2p0-dirichlet-s0p01-raw-hardv2p0-em100` | 16 | 0.4789 | 0.2289 | 35.4504 | 74.0242 | 0.9868 | 0 | 2.8310 | 0.9998 |
| `gmm-k16-floorv2p0-none-s0p0-raw-hardv2p0-em100` | 16 | 0.4810 | 0.2460 | 35.6065 | 74.0237 | 0.9906 | 0 | 2.6067 | 0.9997 |
| `gmm-k16-floorv2p0-dirichlet-s512p0-raw-hardv2p0` | 16 | 0.4816 | 0.2567 | 35.6491 | 74.0235 | 0.9921 | 0 | 2.7021 | 0.9997 |
| `gmm-k16-floorv2p0-kl-s512p0-raw-hardv2p0-em100` | 16 | 0.4817 | 0.2524 | 35.6550 | 74.0242 | 0.9886 | 0 | 2.6169 | 0.9998 |

## GMM Stats Angular Metrics

| run | K | center/noise SNR | center/data RMS | pair cos mean | pair cos p05 | nearest2 cos | nearest4 cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gmm-k4-floor0p5-none-s0p0` | 4 | 0.3443 | 0.3940 | -0.3296 | -0.9856 | -0.1939 | -0.3296 |
| `tide-topk-g136-k32-top24-none-hard05` | 32 | 0.7855 | 0.6212 | 0.4395 | -0.1741 | 0.8165 | 0.7902 |
| `fm-r1-g162-gmm-k32-floorv0p0-kl-s512p0-raw-ml-no-coverage` | 32 | 0.8165 | 0.6267 | 0.4290 | -0.1994 | 0.7997 | 0.7763 |

## Cách Đọc

- `center/noise`: khoảng cách trung bình giữa tâm GMM chia cho RMS noise trong một component. Nhỏ hơn 1 là cảnh báo hình học: các component tách theo Euclidean nhưng sample trong component có độ nhiễu đủ lớn để làm hướng bị mờ.
- `center/noise SNR`: norm tâm component chia cho RMS noise component. Thấp nghĩa là hướng từ gốc tới component không sắc.
- `nearest-k cos`: cosine giữa tâm component và các tâm gần nhất theo Euclidean. Nếu thấp hoặc âm, weighted top-k có thể trộn các hướng khác nhau và kéo source vào vùng giữa mode.

## Khắc Phục Đề Xuất

1. Log trực tiếp cosine/angle trong training: `geometry/x0_x1/*`, `geometry/v_x1/*`, `tide/topk_mu_angular_dispersion`, `tide/x0_tide_base/*`.
2. Khi `topk_mu_angular_dispersion` cao, ưu tiên source sparse hơn: `topk=1`, `topk=2` với temperature thấp, hoặc hard-sample một component thay vì weighted mean nhiều component.
3. Không rank source bằng NLL/khoảng cách đơn lẻ. Rank theo FID + flow curvature + geometry cosine + collapse metrics.
4. Nếu muốn sửa GMM fit, cân nhắc spherical/cosine-aware preprocessing hoặc angular penalty cho centers; nhưng thay đổi này nên sau khi đã có logs cosine trên CelebA.
