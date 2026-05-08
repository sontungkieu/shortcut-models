# GMM Ablation Results

- Generated: 2026-05-07T14:19:14.966199+00:00
- Jobs: 11
- COMPLETE: 11
- Parsed: 11
- Missing metrics: 0

## Job Status

| owner | grid | run | status | parsed | kernel |
|---|---:|---|---|---|---|
| casihoavinh | 0 | gmm-k4-floor0p5-none-s0p0 | COMPLETE | ok | `casihoavinh/gmm-k4-floor0p5-none-s0p0-casihoavinh-20260507-0` |
| codemaivanngu | 1 | gmm-k4-floor0p5-dirichlet-s0p01 | COMPLETE | ok | `codemaivanngu/gmm-k4-floor0p5-dirichlet-s0p01-codemaivanngu-20` |
| hoanganpham123 | 0 | gmm-k4-floor0p5-none-s0p0 | COMPLETE | ok | `hoanganpham123/gmm-k4-floor0p5-none-s0p0-hoanganpham123-2026050` |
| huynhtule | 3 | gmm-k4-floor0p5-kl-s512p0 | COMPLETE | ok | `huynhtule/gmm-k4-floor0p5-kl-s512p0-huynhtule-20260507-064` |
| iamlonely | 4 | gmm-k4-floor0p5-kl-s2048p0 | COMPLETE | ok | `iamlonely/gmm-k4-floor0p5-kl-s2048p0-iamlonely-20260507-06` |
| kieuhongquan | 5 | gmm-k4-floor1p0-none-s0p0 | COMPLETE | ok | `kieuhongquan/gmm-k4-floor1p0-none-s0p0-kieuhongquan-20260507` |
| kiuvithong | 6 | gmm-k4-floor1p0-dirichlet-s0p01 | COMPLETE | ok | `kiuvithong/gmm-k4-floor1p0-dirichlet-s0p01-kiuvithong-20260` |
| manh1904 | 0 | gmm-k4-floor0p5-none-s0p0 | COMPLETE | ok | `manh1904/gmm-k4-floor0p5-none-s0p0-manh1904-20260507-0633` |
| nguyncmnhda | 8 | gmm-k4-floor1p0-kl-s512p0 | COMPLETE | ok | `nguyncmnhda/gmm-k4-floor1p0-kl-s512p0-nguyncmnhda-20260507-0` |
| veilwings | 10 | gmm-k4-floor1p25-none-s0p0 | COMPLETE | ok | `veilwings/gmm-k4-floor1p25-none-s0p0-veilwings-20260507-06` |
| victorharvey27 | 11 | gmm-k4-floor1p25-dirichlet-s0p01 | COMPLETE | ok | `victorharvey27/gmm-k4-floor1p25-dirichlet-s0p01-victorharvey27` |

## Parsed Metrics

| owner | grid | run | train_nll | valid_nll | data_var | floor_var_std | floor_var_latent | comp_var_mean | pi_entropy_norm | pi_kl | pi_min | pi_max | dead(train/valid) | count_ratio(train/valid) | floor_hit | overlap_max |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| casihoavinh | 0 | gmm-k4-floor0p5-none-s0p0 | 5516.43 | 5517.47 | 0.668858 | 0.250000 | 0.167214 | 0.881890 | 0.998271 | 0.002397 | 0.2279 | 0.2750 | 0/0 | 1.2117/1.1825 | 0.0000 | 0.000003 |
| codemaivanngu | 1 | gmm-k4-floor0p5-dirichlet-s0p01 | 5516.60 | 5518.75 | 0.668844 | 0.250000 | 0.167211 | 0.881701 | 0.998306 | 0.002349 | 0.2309 | 0.2737 | 0/0 | 1.1830/1.2448 | 0.0000 | 0.000002 |
| hoanganpham123 | 0 | gmm-k4-floor0p5-none-s0p0 | 5516.62 | 5517.55 | 0.668848 | 0.250000 | 0.167212 | 0.881947 | 0.999063 | 0.001298 | 0.2396 | 0.2711 | 0/0 | 1.1384/1.1806 | 0.0000 | 0.000003 |
| huynhtule | 3 | gmm-k4-floor0p5-kl-s512p0 | 5516.47 | 5517.72 | 0.668885 | 0.250000 | 0.167221 | 0.881955 | 0.998818 | 0.001638 | 0.2333 | 0.2714 | 0/0 | 1.1668/1.1593 | 0.0000 | 0.000003 |
| iamlonely | 4 | gmm-k4-floor0p5-kl-s2048p0 | 5516.48 | 5517.65 | 0.668825 | 0.250000 | 0.167206 | 0.881948 | 0.998501 | 0.002079 | 0.2296 | 0.2736 | 0/0 | 1.2025/1.1728 | 0.0000 | 0.000003 |
| kieuhongquan | 5 | gmm-k4-floor1p0-none-s0p0 | 5565.42 | 5566.59 | 0.668866 | 1.000000 | 0.668866 | 1.015105 | 0.998730 | 0.001760 | 0.2310 | 0.2710 | 0/0 | 1.1777/1.1040 | 0.7963 | 0.000005 |
| kiuvithong | 6 | gmm-k4-floor1p0-dirichlet-s0p01 | 5565.60 | 5567.39 | 0.668845 | 1.000000 | 0.668845 | 1.015371 | 0.999240 | 0.001054 | 0.2356 | 0.2624 | 0/0 | 1.1083/1.0976 | 0.8003 | 0.000005 |
| manh1904 | 0 | gmm-k4-floor0p5-none-s0p0 | 5516.57 | 5518.63 | 0.668825 | 0.250000 | 0.167206 | 0.881814 | 0.997788 | 0.003066 | 0.2255 | 0.2770 | 0/0 | 1.2246/1.1974 | 0.0000 | 0.000003 |
| nguyncmnhda | 8 | gmm-k4-floor1p0-kl-s512p0 | 5565.35 | 5567.55 | 0.668887 | 1.000000 | 0.668887 | 1.014958 | 0.998749 | 0.001734 | 0.2298 | 0.2713 | 0/0 | 1.1840/1.1107 | 0.7975 | 0.000005 |
| veilwings | 10 | gmm-k4-floor1p25-none-s0p0 | 5832.24 | 5833.29 | 0.668877 | 1.562500 | 1.045120 | 1.562500 | 0.998861 | 0.001579 | 0.2300 | 0.2697 | 0/0 | 1.1714/1.0984 | 1.0000 | 0.000055 |
| victorharvey27 | 11 | gmm-k4-floor1p25-dirichlet-s0p01 | 5832.23 | 5833.41 | 0.668882 | 1.562500 | 1.045128 | 1.562500 | 0.998851 | 0.001593 | 0.2311 | 0.2706 | 0/0 | 1.1685/1.1011 | 1.0000 | 0.000055 |
