# Bridge + Tide-KL Analysis

## Scope

- Pulled diagnostics/log outputs for 6 Kaggle runs on 2026-06-12.
- Output root: `/home/tung/shortcut-models-1/outputs/kaggle/bridge_tidekl_20260612`.
- Checkpoints were not downloaded; only diagnostics/log/notebook output files were kept.
- All 6 kernels report `CANCEL_ACKNOWLEDGED`, but each produced parseable diagnostics.
- Historical reference FID128: plain best `6.969`; deep-router d4/drop0.2 best `7.094`; shallow regularization best `7.147`.

## Ranking By FID128

| rank | family | config | step | evals | FID128 | FID32 | delta vs 6.969 | valid loss | router loss/top1 | usage H | pred/target var | flow straight/curv | status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | router bridge lambda | `bridge lambda B(1,1), FM t uniform/discrete` | 351900 | 7 | 7.033 | 9.334 | 0.064 | 0.4749 | 0.229/0.907 | 0.884 | 0.672 | 1.109/0.0209 | CANCEL_ACKNOWLEDGED |
| 2 | router bridge lambda | `bridge lambda B(3,1.4), FM t uniform/discrete` | 350000 | 6 | 7.404 | 9.566 | 0.435 | 0.4556 | 0.296/0.868 | 0.864 | 0.665 | 1.109/0.0206 | CANCEL_ACKNOWLEDGED |
| 3 | router bridge lambda | `bridge lambda B(2.2,1.2), FM t uniform/discrete` | 350000 | 6 | 7.612 | 9.894 | 0.643 | 0.4555 | 0.278/0.888 | 0.896 | 0.660 | 1.111/0.0208 | CANCEL_ACKNOWLEDGED |
| 4 | q_GMM(x0_tide) KL | `tide-KL w=0.10, FM t Beta(3,1.4)` | 200000 | 9 | 8.339 | 10.087 | 1.370 | 0.4497 | 0.250/0.900 | 0.947 | 0.662 | 1.106/0.0190 | CANCEL_ACKNOWLEDGED |
| 5 | q_GMM(x0_tide) KL | `tide-KL w=0.05, FM t Beta(3,1.4)` | 200000 | 9 | 8.366 | 9.858 | 1.397 | 0.4546 | 0.201/0.917 | 0.926 | 0.659 | 1.104/0.0187 | CANCEL_ACKNOWLEDGED |
| 6 | q_GMM(x0_tide) KL | `tide-KL w=0.30, FM t Beta(3,1.4)` | 200000 | 9 | 8.645 | 10.261 | 1.676 | 0.4592 | 0.261/0.897 | 0.941 | 0.641 | 1.104/0.0187 | CANCEL_ACKNOWLEDGED |

## Family Takeaways

- `bridge lambda B(1,1)` is the strongest new run: FID128 `7.033` at step 350k. It does not beat the historical best `6.969`, but it beats the deep-router d4/drop0.2 result `7.094` by about `0.061` and all previous router-regularization variants in the reports.
- Bridge lambda shape matters a lot. Moving bridge lambda from uniform `B(1,1)` to endpoint-biased `B(2.2,1.2)` or `B(3,1.4)` worsens FID to `7.612` and `7.404`. So the useful effect is not simply “more samples near x1”; it is likely the smoother coverage of the segment between source and data.
- Tide-KL to `q_GMM(x0_tide)` is not promising in this form. Weights `0.05/0.10/0.30` give FID128 `8.366/8.339/8.645` at 180k best eval. They also stop at 200k, so they are not fully comparable with 350k runs, but the early curve is clearly behind the bridge family.
- Tide-KL does reduce curvature (`~0.0187-0.0190`) compared with bridge (`~0.0206-0.0209`), but FID is much worse. This reinforces the earlier observation: straighter flow alone is not enough if the source/router distribution becomes too sharp or mismatched.
- Bridge-uniform has higher valid loss (`0.475`) than bridge endpoint-biased variants (`~0.456`), yet much better FID. This is another case where FM validation MSE is not a reliable ranking metric for image quality.
- Router usage entropy is lower for bridge-uniform (`0.884`) than bridge B(2.2,1.2) (`0.896`) and lower than Tide-KL (`0.926-0.947`), but FID is better. For this source family, forcing more uniform routing is not the main bottleneck.
- `pred/target variance` remains under-dispersed in all runs (`0.641-0.672`), so none of these modifications fixes the DiT velocity variance mismatch.

## Recommendation

- Keep `router bridge lambda B(1,1)` as the best new candidate from this batch. It is worth repeating or combining with the best time schedule/source setting, but it still has not surpassed the original 6.969 baseline.
- Do not expand Tide-KL weights yet. If revisited, use a gentler schedule/ramp or log whether `q_GMM(x0_tide)` target is too sharp; current fixed weights hurt FID.
- For bridge, the next useful ablation is not stronger endpoint bias. Try changing bridge target construction or bridge probability/frequency while keeping lambda broad, e.g. B(1,1) vs B(0.7,0.7) vs B(1.5,1.5), or combine bridge-uniform with sample-topk.
- Continue ranking by FID + geometry diagnostics, not by valid loss alone.

## Files

- JSON: `/home/tung/shortcut-models-1/reports/bridge_tidekl_analysis_20260612.json`
- CSV: `/home/tung/shortcut-models-1/reports/bridge_tidekl_analysis_20260612.csv`
- Raw collect summary: `reports/bridge_tidekl_results_20260612.json`
- Download report: `/home/tung/shortcut-models-1/reports/bridge_tidekl_download_20260612.json`
