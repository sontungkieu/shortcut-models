# Router EMA Beta35 Resume Diagnostics

- Kernel: `veilwings/tide-resume-routerema-beta35-13-jointmix-k16-top`
- Kaggle status: `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`. Đây là dừng/cancel sau khi đã chạy và ghi output, không phải lỗi Python trong diagnostics.
- Output root: `outputs/kaggle/router_ema_beta35_resume_20260613`
- Diagnostics: `outputs/kaggle/router_ema_beta35_resume_20260613/gmm_tide_fm/tide-resume-routerema-beta35-13-jointmix-k16-top2-soft075-dir512/diagnostics`
- Resume source: `veilwings/tide-routerema-beta35-13-jointmix-k16-top2-soft0`
- Loaded model/router step: `150000`
- Last train step: `351400`
- Best FID128: `7.2723` at step `280000`
- Last FID128: `8.7577` at eval step `340000`
- Heavy files found locally: `0`

## Eval Curve

| step | FID128 | FID32 | straightness | curvature |
|---:|---:|---:|---:|---:|
| 160000 | 8.3084 | 9.1046 | 1.1009 | 0.01846 |
| 180000 | 8.7490 | 10.2048 | 1.1006 | 0.01833 |
| 200000 | 8.1926 | 9.3372 | 1.1014 | 0.01861 |
| 220000 | 7.7750 | 8.6574 | 1.0944 | 0.01822 |
| 240000 | 7.6887 | 8.8319 | 1.0999 | 0.01856 |
| 260000 | 8.2513 | 9.9032 | 1.1011 | 0.01897 |
| 280000 | 7.2723 | 8.3256 | 1.0956 | 0.01874 |
| 300000 | 7.3606 | 8.4577 | 1.0998 | 0.01923 |
| 320000 | 8.5050 | 10.2400 | 1.0949 | 0.01871 |
| 340000 | 8.7577 | 10.4585 | 1.0994 | 0.01930 |

## Notes

- Resume thành công: `train_stdout.txt` có `Loaded GMM router train state with step 150000` và `Loaded model with step 150000`.
- Run cải thiện tốt nhất ở step 280k, sau đó FID xấu đi ở 320k và 340k. Nếu dùng kết quả này làm nguồn tiếp theo, nên coi 280k là điểm tốt nhất theo metric đã log; checkpoint local cuối cùng trong notebook là file cố định ghi tại 300k.
- Không tải checkpoint/model artifact về local trong lần kéo này.
