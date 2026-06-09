# GMM EM25 vs EM100

- Generated: 2026-05-21T09:33:35.854768+00:00
- Joined rows: 180
- Missing baseline: 0
- Source rerun recommendations: 0
- Profile-ok EM100 rows: 108
- Still improving after iter 25: 38

## Decision

No automatic FM rerun recommendation from the configured EM100 criteria.

## Recommended Source Reruns

| grid | run | K | valid nll delta % | pi entropy drop | count ratio rel inc | overlap inc | last10 rel |
|---:|---|---:|---:|---:|---:|---:|---:|

## Top Valid NLL Improvements

| grid | run | K | EM25 valid | EM100 valid | delta % | pi entropy | dead | count ratio | overlap |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 89 | gmm-k8-floorv0p0-kl-s2048p0-raw-soft-v1p5-s2048-varkl-s2048p0-v1p5-em100 | 8 | 4541.280 | 4467.176 | 1.632 | 0.6846 | 0/0 | 317.7500 | 0.0000 |
| 143 | gmm-k32-floorv0p0-none-s0p0-raw-soft-v1p5-s2048-varkl-s2048p0-v1p5-em100 | 32 | 4513.116 | 4463.881 | 1.091 | 0.4273 | 1/14 | 1187.0000 | 0.0000 |
| 107 | gmm-k16-floorv0p0-dirichlet-s0p01-raw-soft-v1p5-s2048-varkl-s2048p0-v1p5-em100 | 16 | 4492.951 | 4464.929 | 0.624 | 0.5260 | 0/3 | 1211.0000 | 0.0000 |
| 80 | gmm-k8-floorv0p0-kl-s512p0-raw-soft-v1p5-s2048-varkl-s2048p0-v1p5-em100 | 8 | 4493.389 | 4465.976 | 0.610 | 0.6906 | 0/0 | 301.5000 | 0.0000 |
| 53 | gmm-k8-floorv0p0-none-s0p0-raw-soft-v1p5-s2048-varkl-s2048p0-v1p5-em100 | 8 | 4494.211 | 4467.225 | 0.600 | 0.6866 | 0/0 | 181.5714 | 0.0000 |
| 124 | gmm-k16-floorv0p0-kl-s512p0-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 16 | 4358.284 | 4337.595 | 0.475 | 0.7897 | 0/2 | 598.0000 | 0.0000 |
| 88 | gmm-k8-floorv0p0-kl-s2048p0-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 8 | 4371.383 | 4354.713 | 0.381 | 0.9865 | 0/0 | 2.1148 | 0.0000 |
| 24 | gmm-k4-floorv0p0-dirichlet-s512p0-raw-soft-v1p0-s2048-varkl-s2048p0-v1p0-em100 | 4 | 4473.366 | 4457.530 | 0.354 | 0.9976 | 0/0 | 1.2275 | 0.0000 |
| 158 | gmm-k32-floorv0p0-dirichlet-s512p0-raw-soft-v1p0-s512-varkl-s512p0-v1p0-em100 | 32 | 4273.468 | 4260.599 | 0.301 | 0.9366 | 0/2 | 413.0000 | 0.0000 |
| 153 | gmm-k32-floorv0p0-dirichlet-s512p0-raw-ml-no-coverage-em100 | 32 | 4188.997 | 4177.292 | 0.279 | 0.9974 | 0/0 | 2.5897 | 0.0000 |
| 127 | gmm-k16-floorv0p5-kl-s2048p0-raw-hardv0p5-em100 | 16 | 4265.167 | 4255.382 | 0.229 | 0.9938 | 0/0 | 2.1098 | 0.0000 |
| 104 | gmm-k16-floorv0p0-dirichlet-s0p01-raw-soft-v1p0-s512-varkl-s512p0-v1p0-em100 | 16 | 4285.248 | 4275.869 | 0.219 | 0.9719 | 0/0 | 4.0849 | 0.0000 |
| 122 | gmm-k16-floorv0p0-kl-s512p0-raw-soft-v1p0-s512-varkl-s512p0-v1p0-em100 | 16 | 4285.500 | 4276.527 | 0.209 | 0.9639 | 0/0 | 20.1429 | 0.0000 |
| 177 | gmm-k32-floorv0p0-kl-s2048p0-raw-soft-v1p0-s2048-varkl-s2048p0-v1p0-em100 | 32 | 4413.749 | 4405.114 | 0.196 | 0.6284 | 0/1 | 888.0000 | 0.0000 |
| 1 | gmm-k4-floorv0p5-none-s0p0-raw-hardv0p5-em100 | 4 | 4450.671 | 4442.067 | 0.193 | 0.9985 | 0/0 | 1.1632 | 0.0000 |
| 149 | gmm-k32-floorv0p0-dirichlet-s0p01-raw-soft-v1p0-s512-varkl-s512p0-v1p0-em100 | 32 | 4272.033 | 4264.206 | 0.183 | 0.8269 | 0/0 | 436.0000 | 0.0000 |
| 117 | gmm-k16-floorv0p0-kl-s512p0-raw-ml-no-coverage-em100 | 16 | 4260.125 | 4252.395 | 0.181 | 0.9899 | 0/0 | 2.6012 | 0.0000 |
| 28 | gmm-k4-floorv0p5-kl-s512p0-raw-hardv0p5-em100 | 4 | 4451.619 | 4443.655 | 0.179 | 0.9984 | 0/0 | 1.1698 | 0.0000 |
| 160 | gmm-k32-floorv0p0-dirichlet-s512p0-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 32 | 4329.272 | 4321.868 | 0.171 | 0.8751 | 0/11 | 549.0000 | 0.0000 |
| 90 | gmm-k16-floorv0p0-none-s0p0-raw-ml-no-coverage-em100 | 16 | 4259.482 | 4252.403 | 0.166 | 0.9896 | 0/0 | 2.2842 | 0.0000 |

## Worst Valid NLL Changes

| grid | run | K | EM25 valid | EM100 valid | delta % | pi entropy | dead | count ratio | overlap |
|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| 179 | gmm-k32-floorv0p0-kl-s2048p0-raw-soft-v1p5-s2048-varkl-s2048p0-v1p5-em100 | 32 | 4491.787 | 4538.054 | -1.030 | 0.2699 | 0/7 | 2049.0000 | 0.0000 |
| 44 | gmm-k4-floorv0p0-kl-s2048p0-raw-soft-v1p5-s2048-varkl-s2048p0-v1p5-em100 | 4 | 4470.453 | 4497.495 | -0.605 | 0.8022 | 0/0 | 70.0000 | 0.0000 |
| 151 | gmm-k32-floorv0p0-dirichlet-s0p01-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 32 | 4312.581 | 4326.689 | -0.327 | 0.6638 | 0/15 | 546.0000 | 0.0000 |
| 142 | gmm-k32-floorv0p0-none-s0p0-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 32 | 4312.222 | 4325.275 | -0.303 | 0.6622 | 0/11 | 620.0000 | 0.0000 |
| 150 | gmm-k32-floorv0p0-dirichlet-s0p01-raw-soft-v1p0-s2048-varkl-s2048p0-v1p0-em100 | 32 | 4404.313 | 4414.664 | -0.235 | 0.5840 | 0/1 | 963.0000 | 0.0000 |
| 132 | gmm-k16-floorv0p0-kl-s2048p0-raw-soft-v1p0-s2048-varkl-s2048p0-v1p0-em100 | 16 | 4407.997 | 4417.600 | -0.218 | 0.6955 | 0/0 | 53.5556 | 0.0000 |
| 123 | gmm-k16-floorv0p0-kl-s512p0-raw-soft-v1p0-s2048-varkl-s2048p0-v1p0-em100 | 16 | 4408.854 | 4417.814 | -0.203 | 0.6849 | 0/0 | 75.2308 | 0.0000 |
| 87 | gmm-k8-floorv0p0-kl-s2048p0-raw-soft-v1p0-s2048-varkl-s2048p0-v1p0-em100 | 8 | 4417.961 | 4425.604 | -0.173 | 0.8382 | 0/0 | 13.1923 | 0.0000 |
| 52 | gmm-k8-floorv0p0-none-s0p0-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 8 | 4364.237 | 4369.461 | -0.120 | 0.9278 | 0/0 | 84.7000 | 0.0000 |
| 168 | gmm-k32-floorv0p0-kl-s512p0-raw-soft-v1p0-s2048-varkl-s2048p0-v1p0-em100 | 32 | 4408.660 | 4413.838 | -0.117 | 0.5889 | 0/1 | 952.0000 | 0.0000 |
| 83 | gmm-k8-floorv1p0-kl-s2048p0-raw-hardv1p0-em100 | 8 | 4422.369 | 4427.517 | -0.116 | 0.9900 | 0/0 | 2.1064 | 0.0000 |
| 78 | gmm-k8-floorv0p0-kl-s512p0-raw-soft-v1p0-s2048-varkl-s2048p0-v1p0-em100 | 8 | 4420.437 | 4424.615 | -0.095 | 0.8331 | 0/0 | 12.3494 | 0.0000 |
| 97 | gmm-k16-floorv0p0-none-s0p0-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 16 | 4334.470 | 4337.954 | -0.080 | 0.7895 | 0/1 | 618.0000 | 0.0000 |
| 176 | gmm-k32-floorv0p0-kl-s2048p0-raw-soft-v1p0-s512-varkl-s512p0-v1p0-em100 | 32 | 4263.013 | 4266.167 | -0.074 | 0.8320 | 0/0 | 397.0000 | 0.0000 |
| 119 | gmm-k16-floorv1p0-kl-s512p0-raw-hardv1p0-em100 | 16 | 4358.310 | 4361.364 | -0.070 | 0.9910 | 0/0 | 2.3671 | 0.0000 |
| 55 | gmm-k8-floorv0p5-dirichlet-s0p01-raw-hardv0p5-em100 | 8 | 4348.101 | 4351.137 | -0.070 | 0.9905 | 0/0 | 1.9340 | 0.0000 |
| 69 | gmm-k8-floorv0p0-dirichlet-s512p0-raw-soft-v1p0-s2048-varkl-s2048p0-v1p0-em100 | 8 | 4421.610 | 4424.452 | -0.064 | 0.8712 | 0/0 | 13.8514 | 0.0000 |
| 133 | gmm-k16-floorv0p0-kl-s2048p0-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 16 | 4324.642 | 4327.337 | -0.062 | 0.8236 | 0/2 | 598.0000 | 0.0000 |
| 99 | gmm-k16-floorv0p0-dirichlet-s0p01-raw-ml-no-coverage-em100 | 16 | 4259.216 | 4261.443 | -0.052 | 0.9684 | 0/0 | 416.0000 | 0.0000 |
| 164 | gmm-k32-floorv1p0-kl-s512p0-raw-hardv1p0-em100 | 32 | 4311.681 | 4313.317 | -0.038 | 0.9723 | 0/2 | 224.0000 | 0.0000 |

## Convergence Flags

| grid | run | delta 25->final | delta 50->final | last10 rel | train-valid gap | final-best |
|---:|---|---:|---:|---:|---:|---:|
| 48 | gmm-k8-floorv1p5-none-s0p0-raw-hardv1p5-em100 | 2.274389 | 0.323410 | 0.00001511 | 1.403717 | 0.00000000 |
| 52 | gmm-k8-floorv0p0-none-s0p0-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 3.143448 | 2.619095 | 0.00018474 | 0.946707 | 0.00000000 |
| 55 | gmm-k8-floorv0p5-dirichlet-s0p01-raw-hardv0p5-em100 | 4.663210 | 3.265026 | 0.00017283 | 3.610649 | 0.00000000 |
| 61 | gmm-k8-floorv0p0-dirichlet-s0p01-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 4.064434 | 1.370020 | 0.00001144 | 1.036795 | 0.00589180 |
| 64 | gmm-k8-floorv0p5-dirichlet-s512p0-raw-hardv0p5-em100 | 3.301882 | 2.033579 | 0.00008434 | 2.066370 | 0.00000000 |
| 82 | gmm-k8-floorv0p5-kl-s2048p0-raw-hardv0p5-em100 | 1.757193 | 0.329699 | 0.00001178 | -0.449436 | 0.00000000 |
| 83 | gmm-k8-floorv1p0-kl-s2048p0-raw-hardv1p0-em100 | 2.002581 | 0.420057 | 0.00001689 | 4.328888 | 0.00000000 |
| 84 | gmm-k8-floorv1p5-kl-s2048p0-raw-hardv1p5-em100 | 0.895771 | 0.268528 | 0.00001077 | 2.091982 | 0.00000000 |
| 85 | gmm-k8-floorv2p0-kl-s2048p0-raw-hardv2p0-em100 | 0.674250 | 0.523149 | 0.00001120 | 1.081224 | 0.00000000 |
| 88 | gmm-k8-floorv0p0-kl-s2048p0-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 1.314783 | 1.104778 | 0.00008720 | 2.463430 | 0.00000000 |
| 90 | gmm-k16-floorv0p0-none-s0p0-raw-ml-no-coverage-em100 | 2.998903 | 1.308985 | 0.00002812 | 3.280888 | 0.00000000 |
| 95 | gmm-k16-floorv0p0-none-s0p0-raw-soft-v1p0-s512-varkl-s512p0-v1p0-em100 | 6.630663 | 3.152649 | 0.00004773 | 2.153557 | 0.00000000 |
| 99 | gmm-k16-floorv0p0-dirichlet-s0p01-raw-ml-no-coverage-em100 | 4.367666 | 1.326105 | 0.00002411 | 2.429474 | 0.00000000 |
| 102 | gmm-k16-floorv1p5-dirichlet-s0p01-raw-hardv1p5-em100 | 1.557255 | 0.311049 | 0.00001448 | 0.819349 | 0.00000000 |
| 109 | gmm-k16-floorv0p5-dirichlet-s512p0-raw-hardv0p5-em100 | 1.991137 | 1.141771 | 0.00004942 | 3.153341 | 0.00000000 |
| 110 | gmm-k16-floorv1p0-dirichlet-s512p0-raw-hardv1p0-em100 | 1.756689 | 1.205624 | 0.00006897 | 1.379763 | 0.00000000 |
| 112 | gmm-k16-floorv2p0-dirichlet-s512p0-raw-hardv2p0-em100 | 0.570055 | 0.323549 | 0.00001493 | 0.562828 | 0.00000000 |
| 113 | gmm-k16-floorv0p0-dirichlet-s512p0-raw-soft-v1p0-s512-varkl-s512p0-v1p0-em100 | 5.782576 | 2.086182 | 0.00010812 | 4.376595 | 0.00000000 |
| 117 | gmm-k16-floorv0p0-kl-s512p0-raw-ml-no-coverage-em100 | 5.829830 | 2.417164 | 0.00006250 | 2.889435 | 0.00000000 |
| 126 | gmm-k16-floorv0p0-kl-s2048p0-raw-ml-no-coverage-em100 | 3.306065 | 1.031612 | 0.00002602 | 1.380285 | 0.00000000 |
| 128 | gmm-k16-floorv1p0-kl-s2048p0-raw-hardv1p0-em100 | 2.180666 | 1.805883 | 0.00001576 | 2.594858 | 0.00000000 |
| 131 | gmm-k16-floorv0p0-kl-s2048p0-raw-soft-v1p0-s512-varkl-s512p0-v1p0-em100 | 4.357752 | 2.259840 | 0.00001734 | 2.519751 | 0.01059914 |
| 136 | gmm-k32-floorv0p5-none-s0p0-raw-hardv0p5-em100 | 1.204300 | 0.576787 | 0.00001957 | 4.261241 | 0.00000000 |
| 138 | gmm-k32-floorv1p5-none-s0p0-raw-hardv1p5-em100 | 1.313456 | 0.757029 | 0.00002852 | 1.772467 | 0.00000000 |
| 140 | gmm-k32-floorv0p0-none-s0p0-raw-soft-v1p0-s512-varkl-s512p0-v1p0-em100 | 1.834625 | 1.059317 | 0.00001202 | 3.460573 | 0.03007317 |
| 144 | gmm-k32-floorv0p0-dirichlet-s0p01-raw-ml-no-coverage-em100 | 2.384769 | 0.861097 | 0.00003288 | 5.667348 | 0.00000000 |
| 145 | gmm-k32-floorv0p5-dirichlet-s0p01-raw-hardv0p5-em100 | 3.490935 | 0.774990 | 0.00002120 | 4.801844 | 0.00000000 |
| 147 | gmm-k32-floorv1p5-dirichlet-s0p01-raw-hardv1p5-em100 | 0.658810 | 0.342346 | 0.00001091 | 2.674566 | 0.00000000 |
| 151 | gmm-k32-floorv0p0-dirichlet-s0p01-raw-soft-v1p5-s512-varkl-s512p0-v1p5-em100 | 0.862444 | 0.268845 | 0.00001244 | 4.222240 | 0.00109100 |
| 153 | gmm-k32-floorv0p0-dirichlet-s512p0-raw-ml-no-coverage-em100 | 2.289090 | 1.046303 | 0.00005698 | 7.435763 | 0.00000000 |
