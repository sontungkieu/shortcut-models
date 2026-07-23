# GMM-TIDE factorial resume-to-400k analysis

Date: 2026-07-20

## Scope and evidence

This report compares two coupled GMM-TIDE recipes across GMM initialization seeds 0, 1, and 2:

- **W+B**: `weighted` source construction + router distillation on `bridge` inputs.
- **S+M**: `sample_topk` source construction + router distillation on `mix` inputs.

All runs use K16, top-2 routing, a Dirichlet pi prior with strength 512, and a soft KL variance prior with strength 128 and target variance 0.75. Router distillation uses `soft_kl` targets at temperature 1.0; the router is frozen during FM. Time sampling uses the existing discrete-dt/uniform setup. Each run resumed from its 200k checkpoint and was evaluated every 20k steps through cumulative step 400k. Every FID128 evaluation generated 50,048 samples.

Operational evidence:

- 6/6 Kaggle kernels reached `COMPLETE`.
- 6/6 diagnostics downloads succeeded: 72 files, 226.0 MB total.
- Downloads were diagnostics-only; no checkpoints were downloaded.
- All six runtime probes reported TPU v5 lite with 8 visible JAX TPU devices and `runtime_matches_requested=true`.
- All six metric streams reached step 400,000 and contain ten evaluations from 220k through 400k.
- No terminal traceback was found. Kaggle logs contain non-fatal package resolver and notebook-format warnings.

Raw collector reports:

- `reports/gmm_tide_factorial_seed01_resume400_retry2_results_20260720.json`
- `reports/gmm_tide_factorial_seed2_resume400_retry2_results_20260720.json`

Kaggle operational summaries:

- `outputs/kaggle_jobs/gmm_tide_factorial_seed01_resume400_retry2_20260719/reports/aggregate_summary.{json,md,csv}`
- `outputs/kaggle_jobs/gmm_tide_factorial_seed2_resume400_retry2_20260719/reports/aggregate_summary.{json,md,csv}`

The operational summaries count five `ERROR` text hints per run. Manual inspection shows these are false positives from pip's non-fatal dependency-resolver warning and nbformat warning text. The aggregate failure count is zero, runtime accelerator checks pass, and the training diagnostics contain no terminal exception.

## Primary results

Lower FID128 is better. `last-best` measures deterioration between the best checkpoint and step 400k.

| GMM seed | W+B best (step) | W+B @400k | S+M best (step) | S+M @400k | paired best delta W-S | paired @400k delta W-S |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.8810 (300k) | 7.1042 | 7.0682 (380k) | 7.1558 | -0.1872 | -0.0517 |
| 1 | 7.2085 (380k) | 7.2561 | 7.0625 (400k) | 7.0625 | +0.1460 | +0.1937 |
| 2 | 7.3239 (380k) | 7.4259 | 6.9126 (400k) | 6.9126 | +0.4113 | +0.5133 |
| mean | 7.1378 | 7.2621 | 7.0144 | 7.0436 | +0.1234 | +0.2184 |
| sample SD | 0.2298 | 0.1610 | 0.0882 | 0.1227 | 0.2999 | 0.2833 |

Positive W-S delta favors S+M. S+M wins 2/3 paired GMM seeds, while W+B wins seed 0. The paired mean favors S+M, but the 95% t intervals are wide because there are only three GMM seeds:

- Best-checkpoint delta W-S: `+0.1234`, 95% CI `[-0.6216, +0.8684]`.
- Step-400k delta W-S: `+0.2184`, 95% CI `[-0.4853, +0.9222]`.

These intervals include zero. This is directional evidence, not a statistically resolved win.

The historical one-run baseline is FID128 about 6.969. The S+M best-FID mean is 7.014, 0.045 worse than that reference. Two individual runs cross the reference (W+B seed 0 at 6.881 and S+M seed 2 at 6.913), but neither is a repeated-FID confirmation, so they should not be reported as a robust baseline improvement.

## Training trajectory

Mean FID128 over the three GMM seeds at each matched checkpoint:

| step | W+B mean | S+M mean | delta W-S |
|---:|---:|---:|---:|
| 220k | 7.8980 | 7.7635 | +0.1345 |
| 240k | 7.7455 | 7.7113 | +0.0341 |
| 260k | 7.6233 | 7.4486 | +0.1748 |
| 280k | 7.4740 | 7.2432 | +0.2309 |
| 300k | 7.3063 | 7.3559 | -0.0496 |
| 320k | 7.2325 | 7.0526 | +0.1799 |
| 340k | 7.4212 | 7.3049 | +0.1163 |
| 360k | 7.3351 | 7.2280 | +0.1071 |
| 380k | 7.1425 | 7.0686 | +0.0738 |
| 400k | 7.2621 | 7.0436 | +0.2184 |

S+M has the lower mean at 9/10 matched checkpoints. Its mean `last-best` deterioration is 0.029, compared with 0.124 for W+B. Seeds 1 and 2 of S+M reach their best measured FID at the final 400k checkpoint, whereas every W+B run peaks earlier and worsens at 400k. The defensible interpretation is that S+M tolerates extended training better in this small sample; it does not prove that further training will keep improving.

## Mechanism diagnostics at 400k

| family | pred/target variance | curvature proxy | straightness ratio | mean last-best FID |
|---|---:|---:|---:|---:|
| W+B | 0.6607 +/- 0.0116 | 0.020942 +/- 0.000007 | 1.10968 +/- 0.00024 | 0.1243 |
| S+M | 0.6490 +/- 0.0111 | 0.021002 +/- 0.000039 | 1.10885 +/- 0.00099 | 0.0292 |

Both families under-predict FM target variance by roughly one third. S+M does not improve that variance ratio. Flow curvature and straightness are almost identical, so these diagnostics do not explain its lower mean FID.

At the final training batch, `x0_magnitude` and `x1_magnitude` remain closely matched (roughly 0.89-0.91). Router top-1 probabilities are 0.9977-0.9994, top-k mass is above 0.9990, and top-1 component agreement with the base GMM is 1.0. The router is therefore extremely sharp. Usage entropy remains 0.904-0.941 normalized and 15/16 clusters appear in each sampled final batch, so the logs do not show simple global one-component collapse. Because these runs use a frozen router, these values diagnose the fixed source policy rather than ongoing joint-router optimization.

## Interpretation and next gate

The current evidence promotes S+M as the better candidate for confirmation, mainly because it wins seeds 1 and 2, has much lower across-seed spread at the best checkpoint, and degrades less by 400k. It does not yet establish a robust improvement over the historical 6.969 baseline.

There are two unresolved measurement/design issues:

1. Each checkpoint has only one standard FID evaluation stream. Repeated generation seeds are required to separate FID sampling noise from training/GMM-seed variation.
2. The comparison changes source construction and router distillation data together. It cannot attribute the result to `sample_topk` or `mix` independently.

The next high-information experiment is a full 2x2 at matched GMM seeds:

| source mode | router data | current status |
|---|---|---|
| weighted | bridge | measured |
| weighted | mix | missing |
| sample_topk | bridge | missing |
| sample_topk | mix | measured |

Before expanding training, repeat FID128 on the six 400k checkpoints with the same evaluation-seed set. If compute must be minimized, prioritize S+M seeds 1/2 and W+B seed 0, because those are the candidate winners selected by the current noisy evaluation.

## Reproduction notes

The per-run summaries were produced with `scripts/collect_gmm_tide_results.py` from each downloaded `diagnostics/train_metrics.jsonl`. Aggregate values above use paired GMM seed as the unit: family means/SDs are computed over seeds 0, 1, and 2; the paired delta is `FID(W+B) - FID(S+M)` at the same seed. The confidence interval uses Student's t with 2 degrees of freedom. No FID32 result is used for ranking.
