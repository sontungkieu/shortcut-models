# Factorial W+B / S+M repeated-FID results

- Status: **PASS** (6/6 terminal COMPLETE, 30/30 FID records).
- Protocol: FID128 only, checkpoint 400k, seeds 101/202/303/404/505, 50,048 generations per seed.
- Checkpoints were consumed inside Kaggle; no checkpoint was downloaded locally.

## New repeated-FID runs

| GMM seed | Cell | FID128 mean | sample SD |
|---:|---|---:|---:|
| 0 | S_M | 7.1698 | 0.0365 |
| 0 | W_B | 7.0982 | 0.0202 |
| 1 | S_M | 7.0375 | 0.0308 |
| 1 | W_B | 7.2522 | 0.0306 |
| 2 | S_M | 6.9306 | 0.0344 |
| 2 | W_B | 7.3940 | 0.0255 |

## Full 2x2 factorial

| GMM seed | W+M | W+B | S+M | S+B | Best |
|---:|---:|---:|---:|---:|---|
| 0 | 7.0749 | 7.0982 | 7.1698 | 7.0400 | S_B |
| 1 | 7.2660 | 7.2522 | 7.0375 | 6.9132 | S_B |
| 2 | 7.1015 | 7.3940 | 6.9306 | 7.3965 | S_M |
| **mean** | **7.1475** | **7.2481** | **7.0460** | **7.1166** | **S_M** |

## Descriptive effects

| Effect (lower is better) | Mean | SD across 3 GMM seeds |
|---|---:|---:|
| `source_main_sample_minus_weighted` | -0.1165 | 0.1536 |
| `router_main_bridge_minus_mix` | +0.0856 | 0.2544 |
| `interaction` | -0.0300 | 0.1775 |

## Interpretation

- S_M has the best mean across the three GMM seeds, but this is only three GMM initializations.
- Sample-topk is better than weighted on average and in GMM seeds 1 and 2, but not seed 0.
- Bridge helps in seeds 0 and 1 but is strongly harmful in seed 2, so its effect is not stable.
- The interaction changes sign across seeds; no robust source-by-router-data interaction is established.
