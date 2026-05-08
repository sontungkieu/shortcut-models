# GMM Ablation Summary 2026-05-08

- Source: `reports/gmm_ablation_results_20260508.json`
- Queue: `reports/gmm_ablation_queue_20260507.json`
- Rows: 180
- COMPLETE: 180
- Parsed OK: 180
- Missing metrics: 0
- Train dead-component rows: 1
- Valid dead-component rows: 37
- Rows with critical log matches: 0

## Top 20 by Valid NLL

| rank | grid | k | coverage | pi prior | var prior | valid_nll | train_nll | pi_entropy | dead t/v | count_ratio t/v | comp_var | floor_hit | overlap_max |
|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 162 | 32 | ml-no-coverage | kl:512.00 | none:0.0@1.00 | 4180.63 | 4173.39 | 0.9848 | 0/0 | 230.00/199.00 | 0.5032 | 0.0000 | 0.0000 |
| 2 | 171 | 32 | ml-no-coverage | kl:2048.00 | none:0.0@1.00 | 4180.91 | 4174.68 | 0.9840 | 0/0 | 190.00/101.50 | 0.5027 | 0.0000 | 0.0000 |
| 3 | 144 | 32 | ml-no-coverage | dirichlet:0.01 | none:0.0@1.00 | 4181.87 | 4176.84 | 0.9831 | 0/0 | 232.86/213.00 | 0.5019 | 0.0000 | 0.0000 |
| 4 | 136 | 32 | hardv0p5 | none:0.00 | none:0.0@1.00 | 4187.19 | 4179.75 | 0.9932 | 0/0 | 2.45/2.36 | 0.5152 | 0.1013 | 0.0000 |
| 5 | 145 | 32 | hardv0p5 | dirichlet:0.01 | none:0.0@1.00 | 4187.28 | 4181.78 | 0.9928 | 0/0 | 2.62/2.40 | 0.5143 | 0.0976 | 0.0000 |
| 6 | 135 | 32 | ml-no-coverage | none:0.00 | none:0.0@1.00 | 4187.83 | 4182.37 | 0.9656 | 0/2 | 435.75/208.00 | 0.4995 | 0.0000 | 0.0001 |
| 7 | 154 | 32 | hardv0p5 | dirichlet:512.00 | none:0.0@1.00 | 4188.12 | 4183.63 | 0.9949 | 0/0 | 1513.00/184.00 | 0.5083 | 0.1256 | 0.0000 |
| 8 | 153 | 32 | ml-no-coverage | dirichlet:512.00 | none:0.0@1.00 | 4189.00 | 4179.47 | 0.9912 | 0/2 | 1839.00/202.00 | 0.4861 | 0.0160 | 0.0000 |
| 9 | 163 | 32 | hardv0p5 | kl:512.00 | none:0.0@1.00 | 4189.97 | 4184.68 | 0.9841 | 0/1 | 1549.00/203.00 | 0.5092 | 0.1233 | 0.0000 |
| 10 | 172 | 32 | hardv0p5 | kl:2048.00 | none:0.0@1.00 | 4192.60 | 4183.24 | 0.9861 | 0/1 | 782.00/181.00 | 0.5129 | 0.1149 | 0.0000 |
| 11 | 108 | 16 | ml-no-coverage | dirichlet:512.00 | none:0.0@1.00 | 4257.40 | 4254.39 | 0.9949 | 0/0 | 2.21/2.05 | 0.5207 | 0.0000 | 0.0000 |
| 12 | 109 | 16 | hardv0p5 | dirichlet:512.00 | none:0.0@1.00 | 4258.62 | 4256.75 | 0.9932 | 0/0 | 2.18/2.30 | 0.5273 | 0.0743 | 0.0000 |
| 13 | 126 | 16 | ml-no-coverage | kl:2048.00 | none:0.0@1.00 | 4258.89 | 4254.30 | 0.9888 | 0/0 | 2.85/2.80 | 0.5204 | 0.0000 | 0.0000 |
| 14 | 99 | 16 | ml-no-coverage | dirichlet:0.01 | none:0.0@1.00 | 4259.22 | 4253.05 | 0.9917 | 0/0 | 2.47/2.57 | 0.5216 | 0.0000 | 0.0000 |
| 15 | 90 | 16 | ml-no-coverage | none:0.00 | none:0.0@1.00 | 4259.48 | 4256.52 | 0.9947 | 0/0 | 1.93/1.84 | 0.5194 | 0.0000 | 0.0000 |
| 16 | 100 | 16 | hardv0p5 | dirichlet:0.01 | none:0.0@1.00 | 4259.65 | 4256.66 | 0.9933 | 0/0 | 1.92/1.96 | 0.5281 | 0.0651 | 0.0000 |
| 17 | 117 | 16 | ml-no-coverage | kl:512.00 | none:0.0@1.00 | 4260.13 | 4254.97 | 0.9904 | 0/0 | 2.58/2.70 | 0.5212 | 0.0000 | 0.0000 |
| 18 | 118 | 16 | hardv0p5 | kl:512.00 | none:0.0@1.00 | 4261.32 | 4259.20 | 0.9910 | 0/0 | 2.26/2.21 | 0.5286 | 0.0685 | 0.0000 |
| 19 | 91 | 16 | hardv0p5 | none:0.00 | none:0.0@1.00 | 4261.42 | 4256.18 | 0.9909 | 0/0 | 2.43/2.24 | 0.5279 | 0.0726 | 0.0000 |
| 20 | 176 | 32 | soft-v1p0-s512 | kl:2048.00 | kl:512.0@1.00 | 4263.01 | 4259.61 | 0.8404 | 0/1 | 996.67/380.00 | 0.7514 | 0.0000 | 0.0000 |

## Best by Number of Modes

| k | rows | grid | coverage | pi prior | valid_nll | dead t/v | count_ratio t/v |
|---:|---:|---:|---|---|---:|---:|---:|
| 32 | 45 | 162 | ml-no-coverage | kl:512.00 | 4180.63 | 0/0 | 230.00/199.00 |
| 16 | 45 | 108 | ml-no-coverage | dirichlet:512.00 | 4257.40 | 0/0 | 2.21/2.05 |
| 8 | 45 | 45 | ml-no-coverage | none:0.00 | 4343.91 | 0/0 | 1.74/1.73 |
| 4 | 45 | 18 | ml-no-coverage | dirichlet:512.00 | 4442.41 | 0/0 | 1.22/1.17 |

## Best by Coverage Regime

| coverage | rows | grid | k | pi prior | valid_nll | comp_var | floor_hit |
|---|---:|---:|---:|---|---:|---:|---:|
| ml-no-coverage | 20 | 162 | 32 | kl:512.00 | 4180.63 | 0.5032 | 0.0000 |
| hardv0p5 | 20 | 136 | 32 | none:0.00 | 4187.19 | 0.5152 | 0.1013 |
| soft-v1p0-s512 | 20 | 176 | 32 | kl:2048.00 | 4263.01 | 0.7514 | 0.0000 |
| hardv1p0 | 20 | 146 | 32 | dirichlet:0.01 | 4310.04 | 0.6820 | 0.8396 |
| soft-v1p5-s512 | 20 | 142 | 32 | none:0.00 | 4312.22 | 1.1567 | 0.0000 |
| soft-v1p0-s2048 | 20 | 150 | 32 | dirichlet:0.01 | 4404.31 | 0.9109 | 0.0000 |
| soft-v1p5-s2048 | 20 | 152 | 32 | dirichlet:0.01 | 4462.65 | 1.3850 | 0.0000 |
| hardv1p5 | 20 | 174 | 32 | kl:2048.00 | 4604.43 | 1.0038 | 0.9927 |
| hardv2p0 | 20 | 157 | 32 | dirichlet:512.00 | 4923.10 | 1.3378 | 0.9995 |

## Notes

- Best valid NLL is grid 162 (`gmm-k32-floorv0p0-kl-s512p0-raw-ml-no-coverage`), k=32, coverage `ml-no-coverage`, valid_nll=4180.63.
- All rows parsed successfully; no missing `gmm_metrics.json` files were detected.
- No rows had critical log matches in downloaded stdout/stderr diagnostics.
