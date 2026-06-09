# GMM-TIDE Kaggle Metric Insights

- Live Kaggle status scanned: 51 kernels; status counts `{'CANCEL_ACKNOWLEDGED': 22, 'COMPLETE': 3, 'ERROR': 13, 'STATUS_ERROR': 13}`.
- Local diagnostics collected: 24 rows, deduped to 22 Tide runs.
- Source aggregate: `reports/gmm_tide_all_downloaded_metrics_20260515.json`.

## Top Runs By FID128

| rank | category | run | step | FID128 best/last | FID32 best/last | var pred/target | router valid top1 | diag |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | base_fm | `tide-k16-top2-softv0p75-s128-dir512` | 365600 | 6.969/6.969 | 9.295/9.295 | 0.674 | 0.930 | `outputs/kaggle/moe2_resubmit_fixed_20260512/iamlonely__tide-k16-top2-softv0p75-s128-dir512-iamlonely-20/gmm_tide_fm/tide-k16-top2-softv0p75-s128-dir512/diagnostics` |
| 2 | resume | `tide-resume-r1-k16-top2-soft075-dir512-mixcont10` | 620000 | 7.091/7.347 | 9.253/9.577 | 0.680 |  | `outputs/kaggle/tide_resume_metrics_20260515/mix_continue__victorharvey27__tide-resume-r1-k16-top2-soft075-dir512-mixcont10/output/gmm_tide_fm/tide-resume-r1-k16-top2-soft075-dir512-mixcont10/diagnostics` |
| 3 | base_fm | `tide-k32-top2-softv0p75-s128-dir001` | 367400 | 7.118/7.118 | 9.380/9.380 | 0.649 | 0.907 | `outputs/kaggle/moe2_resubmit_fixed_20260512/veilwings__tide-k32-top2-softv0p75-s128-dir001-veilwings-20/gmm_tide_fm/tide-k32-top2-softv0p75-s128-dir001/diagnostics` |
| 4 | resume | `tide-resume-best-mix-r1-k16-top2-soft075-dir512` | 620000 | 7.131/7.552 | 9.348/9.806 | 0.656 |  | `outputs/kaggle/tide_resume_metrics_20260515/mix_only__no1ceboy__tide-resume-best-mix-r1-k16-top2-soft075-dir512/output/gmm_tide_fm/tide-resume-best-mix-r1-k16-top2-soft075-dir512/diagnostics` |
| 5 | resume | `tide-resume-best-x1cont-r3-k32-hard05` | 620000 | 7.151/7.480 | 9.346/9.789 | 0.664 |  | `outputs/kaggle/tide_resume_metrics_20260515/x1_continue__kiuvithong__tide-resume-best-x1cont-r3-k32-hard05-kiuvithong/output/gmm_tide_fm/tide-resume-best-x1cont-r3-k32-hard05/diagnostics` |
| 6 | base_fm | `tide-k16-top2-g108` | 369100 | 7.219/7.219 | 9.573/9.573 | 0.653 | 0.938 | `outputs/kaggle/gmm_tide_fm_resubmit6_diag_check/casihoavinh_tide-k16-top2-g108-casihoavinh-20260510-1821/gmm_tide_fm/tide-k16-top2-g108/diagnostics` |
| 7 | base_fm | `tide-k32-top2-g136-none-hardv0p5` | 364700 | 7.256/7.256 | 9.669/9.669 | 0.654 | 0.899 | `outputs/kaggle/moe2_resubmit_fixed_20260512/kieuhongquan__tide-k32-top2-g136-none-hardv0p5-kieuhongquan-20/gmm_tide_fm/tide-k32-top2-g136-none-hardv0p5/diagnostics` |
| 8 | mix_continue_ablation | `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` | 333200 | 7.315/7.315 | 9.572/9.597 | 0.656 | 0.921 | `outputs/kaggle/gmm_tide_resume_best_320k_20260513/gmm_tide_fm/tide-mc-r1-k16-top2-soft075-dir512-mixcont10/diagnostics` |
| 9 | mix_continue_ablation | `tide-mc-r1-k16-top2-soft075-dir512-mix` | 335700 | 7.466/7.565 | 9.667/9.887 | 0.648 | 0.922 | `outputs/kaggle/gmm_tide_mix_continue12_20260513/no1ceboy__tide-mc-r1-k16-top2-soft075-dir512-mix-no1ceboy/gmm_tide_fm/tide-mc-r1-k16-top2-soft075-dir512-mix/diagnostics` |
| 10 | base_fm | `tide-k32-top2-g145-dir001-hardv0p5` | 373700 | 7.528/7.528 | 9.989/9.989 | 0.638 | 0.909 | `outputs/kaggle/moe2_resubmit_fixed_20260512/manh1904__tide-k32-top2-g145-dir001-hardv0p5-manh1904-2026/gmm_tide_fm/tide-k32-top2-g145-dir001-hardv0p5/diagnostics` |
| 11 | base_fm | `tide-k32-top4-g136-none-hardv0p5` | 363800 | 7.598/7.598 | 9.984/9.984 | 0.662 | 0.911 | `outputs/kaggle/moe2_resubmit_fixed_20260512/kiuvithong__tide-k32-top4-g136-none-hardv0p5-kiuvithong-2026/gmm_tide_fm/tide-k32-top4-g136-none-hardv0p5/diagnostics` |
| 12 | mix_continue_ablation | `tide-mc-r2-k32-top2-soft075-dir001-mix` | 339400 | 7.617/7.617 | 9.882/9.882 | 0.644 | 0.903 | `outputs/kaggle/gmm_tide_mix_continue12_20260513/codemaivanngu__tide-mc-r2-k32-top2-soft075-dir001-mix-codemaiva/gmm_tide_fm/tide-mc-r2-k32-top2-soft075-dir001-mix/diagnostics` |
| 13 | mix_continue_ablation | `tide-mc-r3-k32-top2-none-hard05-x1cont10` | 342200 | 7.622/7.622 | 10.061/10.061 | 0.642 | 0.905 | `outputs/kaggle/gmm_tide_mix_continue12_20260513/kiuvithong__tide-mc-r3-k32-top2-none-hard05-x1cont10-kiuvith/gmm_tide_fm/tide-mc-r3-k32-top2-none-hard05-x1cont10/diagnostics` |
| 14 | base_fm | `tide-k32-top2-g146-dir001-hardv1p0` | 365700 | 7.623/7.623 | 9.728/9.728 | 0.648 | 0.886 | `outputs/kaggle/moe2_resubmit_fixed_20260512/nguyncmnhda__tide-k32-top2-g146-dir001-hardv1p0-nguyncmnhda-2/gmm_tide_fm/tide-k32-top2-g146-dir001-hardv1p0/diagnostics` |
| 15 | mix_continue_ablation | `tide-mc-r4-k32-top2-dir001-hard05-mix` | 344900 | 7.740/7.831 | 10.037/10.169 | 0.665 | 0.892 | `outputs/kaggle/gmm_tide_mix_continue12_20260513/manh1904__tide-mc-r4-k32-top2-dir001-hard05-mix-manh1904-2/gmm_tide_fm/tide-mc-r4-k32-top2-dir001-hard05-mix/diagnostics` |

## Family Summary

| family | n | best FID128 | mean FID128 | best run |
|---|---:|---:|---:|---|
| base_fm | 7 | 6.969 | 7.330 | `tide-k16-top2-softv0p75-s128-dir512` |
| mix_continue_ablation | 12 | 7.315 | 7.728 | `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` |
| resume | 3 | 7.091 | 7.125 | `tide-resume-r1-k16-top2-soft075-dir512-mixcont10` |

## Insights

- Best non-resume vẫn là `tide-k16-top2-softv0p75-s128-dir512` với FID128 `6.969` tại step 350k.
- Các resume run tốt nhất đạt FID128 khoảng `7.09-7.15`, nhưng last FID xấu hơn best khoảng `+0.25` đến `+0.42`; tức chạy tiếp quá lâu có dấu hiệu drift/overtrain. Checkpoint tốt nhất nằm quanh step 480k trong các resume run.
- `mix_continue` resume là tốt nhất trong nhóm resume: FID128 best `7.091`, tốt hơn `mix_only` và `x1_continue`, nhưng vẫn chưa vượt baseline non-resume tốt nhất.
- `fm_pred_target_var_ratio` phần lớn chỉ khoảng `0.63-0.70` ở các run non-resume và khoảng `0.88-0.95` ở resume; model vẫn dự đoán velocity có variance thấp hơn target. Đây là tín hiệu under-dispersion của vector field, không phải lỗi router đơn thuần.
- `x0/x1` magnitude ratio gần 1 ở các run có log, nên x0 sample không bị lệch norm nghiêm trọng so với x1 trong các run hiện tại.
- Live context hiện chỉ còn 3 kernel COMPLETE có metric đầy đủ; các kernel cũ nhiều cái `CANCEL_ACKNOWLEDGED` hoặc `ERROR`, nhưng vẫn có một số diagnostics local từ lần tải trước nên vẫn được đưa vào bảng local.
