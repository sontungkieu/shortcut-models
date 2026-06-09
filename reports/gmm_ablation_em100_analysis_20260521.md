# GMM EM100 Analysis

- Parsed EM100: 180/180
- Joined EM25/EM100: 180/180
- Median valid NLL improvement: 0.0143%
- Mean valid NLL improvement: 0.0446%
- Improved rows: 121/180; >=0.5%: 5; >=1%: 2; worse: 59
- Profile-ok rows: 108/180; rows with dead components: 37/180
- Known-source rerun recommendations: 0

## Plots
- [gmm_em100_valid_nll_em25_vs_em100.png](reports/plots/gmm_em100_valid_nll_em25_vs_em100.png)
- [gmm_em100_delta_by_coverage.png](reports/plots/gmm_em100_delta_by_coverage.png)
- [gmm_em100_delta_vs_count_ratio.png](reports/plots/gmm_em100_delta_vs_count_ratio.png)
- [gmm_em100_convergence_after25_by_k.png](reports/plots/gmm_em100_convergence_after25_by_k.png)
- [gmm_em100_source_grid_delta.png](reports/plots/gmm_em100_source_grid_delta.png)

## Top Improvements
| grid | K | coverage | prior | delta % | EM100 valid NLL | pi entropy | count ratio | dead | profile ok |
|---:|---:|---|---|---:|---:|---:|---:|---|---|
| 89 | 8 | soft-v1p5-s2048 | kl:2048.0 | 1.632 | 4467.176 | 0.6846 | 317.750 | 0/0 | False |
| 143 | 32 | soft-v1p5-s2048 | none:0.0 | 1.091 | 4463.881 | 0.4273 | 1187.000 | 1/14 | False |
| 107 | 16 | soft-v1p5-s2048 | dirichlet:0.01 | 0.624 | 4464.929 | 0.5260 | 1211.000 | 0/3 | False |
| 80 | 8 | soft-v1p5-s2048 | kl:512.0 | 0.610 | 4465.976 | 0.6906 | 301.500 | 0/0 | False |
| 53 | 8 | soft-v1p5-s2048 | none:0.0 | 0.600 | 4467.225 | 0.6866 | 181.571 | 0/0 | False |
| 124 | 16 | soft-v1p5-s512 | kl:512.0 | 0.475 | 4337.595 | 0.7897 | 598.000 | 0/2 | False |
| 88 | 8 | soft-v1p5-s512 | kl:2048.0 | 0.381 | 4354.713 | 0.9865 | 2.115 | 0/0 | True |
| 24 | 4 | soft-v1p0-s2048 | dirichlet:512.0 | 0.354 | 4457.530 | 0.9976 | 1.228 | 0/0 | True |
| 158 | 32 | soft-v1p0-s512 | dirichlet:512.0 | 0.301 | 4260.599 | 0.9366 | 413.000 | 0/2 | False |
| 153 | 32 | ml-no-coverage | dirichlet:512.0 | 0.279 | 4177.292 | 0.9974 | 2.590 | 0/0 | True |
| 127 | 16 | hardv0p5 | kl:2048.0 | 0.229 | 4255.382 | 0.9938 | 2.110 | 0/0 | True |
| 104 | 16 | soft-v1p0-s512 | dirichlet:0.01 | 0.219 | 4275.869 | 0.9719 | 4.085 | 0/0 | True |

## Known Source Configs
| grid | K | run | delta % | EM25 valid | EM100 valid | pi entropy | count ratio | dead | rerun? |
|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| 108 | 16 | gmm-k16-floorv0p0-dirichlet-s512p0-raw-ml-no-coverage-em100 | 0.013 | 4257.401 | 4256.854 | 0.9922 | 2.130 | 0/0 | False |
| 109 | 16 | gmm-k16-floorv0p5-dirichlet-s512p0-raw-hardv0p5-em100 | 0.023 | 4258.617 | 4257.638 | 0.9922 | 2.591 | 0/0 | False |
| 117 | 16 | gmm-k16-floorv0p0-kl-s512p0-raw-ml-no-coverage-em100 | 0.181 | 4260.125 | 4252.395 | 0.9899 | 2.601 | 0/0 | False |
| 126 | 16 | gmm-k16-floorv0p0-kl-s2048p0-raw-ml-no-coverage-em100 | 0.149 | 4258.893 | 4252.556 | 0.9901 | 2.300 | 0/0 | False |
| 136 | 32 | gmm-k32-floorv0p5-none-s0p0-raw-hardv0p5-em100 | 0.080 | 4187.194 | 4183.845 | 0.9935 | 2.852 | 0/0 | False |
| 145 | 32 | gmm-k32-floorv0p5-dirichlet-s0p01-raw-hardv0p5-em100 | 0.100 | 4187.277 | 4183.072 | 0.9932 | 2.795 | 0/0 | False |
| 146 | 32 | gmm-k32-floorv1p0-dirichlet-s0p01-raw-hardv1p0-em100 | 0.061 | 4310.036 | 4307.403 | 0.9916 | 2.494 | 0/0 | False |
| 154 | 32 | gmm-k32-floorv0p5-dirichlet-s512p0-raw-hardv0p5-em100 | -0.034 | 4188.120 | 4189.524 | 0.9950 | 192.000 | 0/1 | False |
| 162 | 32 | gmm-k32-floorv0p0-kl-s512p0-raw-ml-no-coverage-em100 | 0.116 | 4180.631 | 4175.799 | 0.9921 | 2.706 | 0/0 | False |

## By K
| K | n | median delta % | mean delta % | >=0.5% count | profile ok | dead rows | median delta25->final |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 45 | 0.0018 | 0.0050 | 0 | 40 | 0 | 0.0278 |
| 8 | 45 | 0.0174 | 0.0807 | 3 | 32 | 0 | 1.1501 |
| 16 | 45 | 0.0230 | 0.0565 | 1 | 22 | 10 | 0.8574 |
| 32 | 45 | 0.0290 | 0.0362 | 1 | 14 | 27 | 1.1467 |

## By Coverage
| coverage | n | median delta % | max delta % | profile ok | dead rows |
|---|---:|---:|---:|---:|---:|
| ml-no-coverage | 20 | 0.0186 | 0.2794 | 17 | 1 |
| hardv0p5 | 20 | 0.0533 | 0.2294 | 16 | 3 |
| hardv1p0 | 20 | 0.0090 | 0.0734 | 15 | 2 |
| hardv1p5 | 20 | 0.0182 | 0.1041 | 17 | 1 |
| hardv2p0 | 20 | 0.0079 | 0.0332 | 15 | 3 |
| soft-v1p0-s512 | 20 | 0.0673 | 0.3011 | 14 | 3 |
| soft-v1p0-s2048 | 20 | -0.0120 | 0.3540 | 5 | 4 |
| soft-v1p5-s512 | 20 | 0.0225 | 0.4747 | 9 | 10 |
| soft-v1p5-s2048 | 20 | -0.0024 | 1.6318 | 0 | 10 |
