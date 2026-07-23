# Router Smooth5 Analysis

## Data Status

- Kaggle `kernels output` currently returns no files because all 5 kernels still report `KernelWorkerStatus.RUNNING`.
- W&B also marks all 5 train runs as `running`; analysis below uses W&B history snapshot exported with `samples=10000`.
- All runs are around train step `349900`, but the latest FID available in this snapshot is at step `300000`. The 350k eval had not appeared yet.
- No checkpoints were downloaded.

## Ranking

| rank | variant | FID128 @step | FID32 | delta vs 6.969 | delta vs reg-best 7.147 | valid loss last | pred/target var | flow straight/curv |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `router bridge Beta(2,2)` | 7.277 @300000 | 9.435 | 0.308 | 0.130 | 0.4592 | 0.650 | 1.110/0.0209 |
| 2 | `sample_topk source, layer_norm, dropout=0.2` | 7.302 @300000 | 9.577 | 0.333 | 0.155 | 0.4576 | 0.656 | 1.110/0.0207 |
| 3 | `target T=2.0` | 7.396 @300000 | 9.613 | 0.427 | 0.249 | 0.4562 | 0.643 | 1.110/0.0209 |
| 4 | `target T=1.5` | 7.506 @300000 | 9.712 | 0.537 | 0.359 | 0.4573 | 0.644 | 1.111/0.0208 |
| 5 | `entropy floor=0.25 w=0.05` | 7.533 @300000 | 9.697 | 0.564 | 0.385 | 0.4548 | 0.649 | 1.111/0.0207 |

## Detailed Signals

| variant | router usage H | assign max frac | KL to GMM base | x0-x1 cosine | v-x1 cosine | v-x0 cosine | topk mu cosine | angular disp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `router bridge Beta(2,2)` | 0.939 | 0.141 | 0.0046 | 0.145 | 0.633 | -0.668 | 0.721 | 0.000670 |
| `sample_topk source, layer_norm, dropout=0.2` | 0.952 | 0.141 | 0.0073 | 0.166 | 0.624 | -0.661 | 0.743 | 0.000761 |
| `target T=2.0` | 0.939 | 0.141 | 0.0009 | 0.154 | 0.631 | -0.663 | 0.664 | 0.000136 |
| `target T=1.5` | 0.947 | 0.141 | 0.0008 | 0.185 | 0.616 | -0.652 | 0.781 | 0.000123 |
| `entropy floor=0.25 w=0.05` | 0.946 | 0.125 | 0.0013 | 0.144 | 0.634 | -0.668 | 0.795 | 0.000169 |

## Insights

1. Best in this batch is `router bridge Beta(2,2)` with FID128 `7.277` at 300k. It is better than temperature smoothing and entropy floor, but still worse than historical baseline `6.969` by about `0.308`.
2. `sample_topk + LayerNorm + dropout 0.2` is second at FID128 `7.302`, very close to bridge. This suggests stochastic source selection is viable, but not enough to beat the earlier plain/regularized recipes.
3. Target-temperature smoothing is not helping in this form: T=2.0 beats T=1.5, but both are behind bridge/sample_topk and behind prior regularization best. Softening q_GMM target alone likely smooths the router target without fixing source geometry.
4. Entropy floor is worst in the batch despite the lowest last valid loss among these rows. This repeats the pattern we have seen: FM MSE/valid loss does not rank image quality reliably.
5. Flow metrics are almost tied across all five: straightness around `1.110` and curvature around `0.0207-0.0209`. They do not explain FID separation inside this narrow batch.
6. `pred/target variance` stays around `0.64-0.66`, similar to prior runs. The DiT is still under-dispersed relative to target velocity, so router smoothing does not solve the variance mismatch.
7. Geometry metrics are more informative than loss here, but not all are monotonic. Bridge and sample-topk have the highest top-k angular dispersion in this batch, which is consistent with exposing FM to a slightly less collapsed source. Top-k pair cosine alone is not enough: T=2.0 has low pair cosine but worse FID.

## Recommendation

- Keep `router bridge Beta(2,2)` and `sample_topk + LN + dropout 0.2` as candidates for one repeat or for combining with the better time schedules, but do not make them default yet.
- Do not continue target-temperature-only or entropy-floor-only sweep unless the 350k/400k eval unexpectedly improves; current evidence is weaker than router dropout/LN and weaker than baseline.
- Wait for 350k eval or Kaggle completion before final ranking; current FID is from 300k while training has already progressed to 349900.
- If running next ablation, prioritize source geometry changes: bridge ratios/Beta shape, sample_topk temperature, or angular/variance regularizers. Pure router target smoothing seems low value.

## Files

- W&B snapshot JSON: `reports/gmm_tide_router_smooth5_wandb_20260609.json`
- W&B snapshot table: `reports/gmm_tide_router_smooth5_wandb_20260609.md`
- Analysis CSV: `reports/gmm_tide_router_smooth5_analysis_20260609.csv`
- Export root: `outputs/kaggle_metrics_20260609/router_smooth5_wandb`
## 2026-06-12 Bridge + Tide-KL Update

- New diagnostics-only pull: `reports/bridge_tidekl_analysis_20260612.md`.
- Best new result is `router bridge lambda B(1,1)` with FID128 `7.033` at 350k. This beats the previous deep-router d4/drop0.2 result `7.094`, but still trails the historical baseline `6.969`.
- Endpoint-biased bridge lambda is worse: B(3,1.4) `7.404`, B(2.2,1.2) `7.612`. The bridge benefit appears to come from broad segment smoothing, not from pushing bridge samples near x1.
- Tide-KL to `q_GMM(x0_tide)` is currently weak: FID128 `8.34-8.64` at best eval. It lowers curvature but hurts FID, so it should not be expanded without a gentler schedule.
- This strengthens the earlier conclusion: validation loss and flow straightness are insufficient ranking signals; source geometry and image FID still dominate.
