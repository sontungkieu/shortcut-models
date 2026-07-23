# Phase2 Variance Repeats Analysis

## Material Passport

- Type: Experiment validation report
- Verification Status: ANALYZED
- Scope: four completed Kaggle TPU GMM-TIDE variance-repeat runs, diagnostics-only download
- Primary metric: FID128, lower is better
- Historical baseline: `tide-k16-top2-softv0p75-s128-dir512`, FID128 `6.969`
- Reproducibility: no rerun was launched in this validation; two seeds per family were compared from completed Kaggle outputs
- Heavy artifacts: checkpoints and model blobs were not downloaded

## Artifacts

- Output root: `outputs/kaggle/phase2_variance_repeats4_20260707`
- Status report: `reports/gmm_tide_phase2_variance_repeats4_status_20260707.json`
- Machine JSON: `reports/gmm_tide_phase2_variance_repeats4_analysis_20260707.json`

## Completed Runs

| candidate | source | seed | FID128 best | best step | delta vs 6.969 | FID128 last | valid loss | var ratio | curvature | x0/x1 mag | x0-x1 cos | router top1 | router usage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4-repeat-s1 | sample_topk + bridge | 1 | 7.631 | 200000 | 0.662 | 7.631 | 0.4618 | 0.655 | 0.02049 | 1.025 | 0.181 | 0.900 | 0.907 |
| C0-repeat-s0 | weighted + mix | 0 | 7.738 | 180000 | 0.769 | 7.892 | 0.4641 | 0.637 | 0.02041 | 1.000 | 0.193 | 0.907 | 0.941 |
| C0-repeat-s1 | weighted + mix | 1 | 7.777 | 200000 | 0.808 | 7.777 | 0.4641 | 0.623 | 0.02051 | 0.993 | 0.197 | 0.944 | 0.952 |
| C4-repeat-s0 | sample_topk + bridge | 0 | 7.918 | 200000 | 0.949 | 7.918 | 0.4596 | 0.635 | 0.02047 | 1.010 | 0.198 | 0.894 | 0.896 |

## Family Summary

| family | n | mean FID128 | std | min-max | mean delta | mean valid loss | mean var ratio | mean curvature | mean x0/x1 | mean cos | mean router usage |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 2 | 7.757 | 0.019 | 7.738-7.777 | 0.788 | 0.4641 | 0.630 | 0.02046 | 0.997 | 0.195 | 0.947 |
| C4 | 2 | 7.775 | 0.143 | 7.631-7.918 | 0.806 | 0.4607 | 0.645 | 0.02048 | 1.017 | 0.189 | 0.901 |

## Interpretation

- All four kernels completed and diagnostics-only downloads succeeded. No fatal traceback was found; the only `ERROR` matches were pip dependency-resolver warnings in notebook logs.
- C4 has the best single seed: `C4-repeat-s1` reaches FID128 `7.631` at 200k.
- C0 is more stable across two seeds: FID128 best is `7.738` and `7.777`; C4 spans `7.631` to `7.918`.
- Mean FID128 is effectively tied at this sample size: C0 mean `7.757`, C4 mean `7.775`. With only two seeds per family, this does not justify claiming C4 dominates C0.
- C4 has slightly better mean valid loss and pred/target variance ratio, but slightly worse router usage entropy and x0-x1 cosine. The best C4 seed is therefore not explained by a clean improvement in the currently logged router/geometry diagnostics.
- Both families remain worse than the historical baseline 6.969. This batch supports a cautious interpretation: phase2-aware source variants are plausible and can produce strong single seeds, but are not yet better than the old GMM-TIDE baseline.

## Fallacy Scan

- Small-n fallacy: active risk. n=2 per family is only a seed check, not a stable ranking.
- Best-seed selection fallacy: active risk. Choosing C4 only because of its best seed would ignore its worse second seed.
- Proxy-metric fallacy: active risk. Valid loss and variance ratio do not align cleanly with FID here.
- Baseline mismatch: controlled partially. These four runs are matched to each other at 200k, but the historical baseline is a longer run, so comparison to 6.969 is contextual rather than a strict same-budget test.

## Next Gate

- Treat C4 as a high-variance candidate, not a promoted winner.
- If continuing this line, compare either more seeds at 200k or resume only the best C4 and best C0 to the same later budget. Do not broaden the grid based solely on `C4-repeat-s1`.
