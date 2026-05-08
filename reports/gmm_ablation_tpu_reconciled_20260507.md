# GMM TPU Ablation Reconciled Submit Report

- Accelerator: `TpuV5E8`
- Requested accounts: 12
- Accounts with TPU job: 11
- Accounts without TPU job: 1
- Unique grid configs submitted: 9 / 150
- Unique grid configs not submitted: 141
- Status counts: {"QUEUED": 11}
- Raw batch report: `reports/gmm_ablation_tpu_submit_20260507.json`

## Submitted TPU Jobs

| owner | grid_index | run_name | status | kernel |
|---|---:|---|---|---|
| casihoavinh | 0 | gmm-k4-floor0p5-none-s0p0 | QUEUED | `casihoavinh/gmm-k4-floor0p5-none-s0p0-casihoavinh-20260507-0` |
| codemaivanngu | 1 | gmm-k4-floor0p5-dirichlet-s0p01 | QUEUED | `codemaivanngu/gmm-k4-floor0p5-dirichlet-s0p01-codemaivanngu-20` |
| hoanganpham123 | 0 | gmm-k4-floor0p5-none-s0p0 | QUEUED | `hoanganpham123/gmm-k4-floor0p5-none-s0p0-hoanganpham123-2026050` |
| huynhtule | 3 | gmm-k4-floor0p5-kl-s512p0 | QUEUED | `huynhtule/gmm-k4-floor0p5-kl-s512p0-huynhtule-20260507-064` |
| iamlonely | 4 | gmm-k4-floor0p5-kl-s2048p0 | QUEUED | `iamlonely/gmm-k4-floor0p5-kl-s2048p0-iamlonely-20260507-06` |
| kieuhongquan | 5 | gmm-k4-floor1p0-none-s0p0 | QUEUED | `kieuhongquan/gmm-k4-floor1p0-none-s0p0-kieuhongquan-20260507` |
| kiuvithong | 6 | gmm-k4-floor1p0-dirichlet-s0p01 | QUEUED | `kiuvithong/gmm-k4-floor1p0-dirichlet-s0p01-kiuvithong-20260` |
| manh1904 | 0 | gmm-k4-floor0p5-none-s0p0 | QUEUED | `manh1904/gmm-k4-floor0p5-none-s0p0-manh1904-20260507-0633` |
| nguyncmnhda | 8 | gmm-k4-floor1p0-kl-s512p0 | QUEUED | `nguyncmnhda/gmm-k4-floor1p0-kl-s512p0-nguyncmnhda-20260507-0` |
| veilwings | 10 | gmm-k4-floor1p25-none-s0p0 | QUEUED | `veilwings/gmm-k4-floor1p25-none-s0p0-veilwings-20260507-06` |
| victorharvey27 | 11 | gmm-k4-floor1p25-dirichlet-s0p01 | QUEUED | `victorharvey27/gmm-k4-floor1p25-dirichlet-s0p01-victorharvey27` |

## Accounts Not Submitted

| owner | reason |
|---|---|
| no1ceboy | Kaggle reported maximum weekly TPU quota reached during push attempts |

## Grid Configs Not Submitted

| grid_index | run_name | k | min_std_data_frac | prior_type | prior_strength |
|---:|---|---:|---:|---|---:|
| 2 | gmm-k4-floor0p5-dirichlet-s512p0 | 4 | 0.5 | dirichlet | 512.0 |
| 7 | gmm-k4-floor1p0-dirichlet-s512p0 | 4 | 1.0 | dirichlet | 512.0 |
| 9 | gmm-k4-floor1p0-kl-s2048p0 | 4 | 1.0 | kl | 2048.0 |
| 12 | gmm-k4-floor1p25-dirichlet-s512p0 | 4 | 1.25 | dirichlet | 512.0 |
| 13 | gmm-k4-floor1p25-kl-s512p0 | 4 | 1.25 | kl | 512.0 |
| 14 | gmm-k4-floor1p25-kl-s2048p0 | 4 | 1.25 | kl | 2048.0 |
| 15 | gmm-k4-floor1p5-none-s0p0 | 4 | 1.5 | none | 0.0 |
| 16 | gmm-k4-floor1p5-dirichlet-s0p01 | 4 | 1.5 | dirichlet | 0.01 |
| 17 | gmm-k4-floor1p5-dirichlet-s512p0 | 4 | 1.5 | dirichlet | 512.0 |
| 18 | gmm-k4-floor1p5-kl-s512p0 | 4 | 1.5 | kl | 512.0 |
| 19 | gmm-k4-floor1p5-kl-s2048p0 | 4 | 1.5 | kl | 2048.0 |
| 20 | gmm-k4-floor1p75-none-s0p0 | 4 | 1.75 | none | 0.0 |
| 21 | gmm-k4-floor1p75-dirichlet-s0p01 | 4 | 1.75 | dirichlet | 0.01 |
| 22 | gmm-k4-floor1p75-dirichlet-s512p0 | 4 | 1.75 | dirichlet | 512.0 |
| 23 | gmm-k4-floor1p75-kl-s512p0 | 4 | 1.75 | kl | 512.0 |
| 24 | gmm-k4-floor1p75-kl-s2048p0 | 4 | 1.75 | kl | 2048.0 |
| 25 | gmm-k8-floor0p5-none-s0p0 | 8 | 0.5 | none | 0.0 |
| 26 | gmm-k8-floor0p5-dirichlet-s0p01 | 8 | 0.5 | dirichlet | 0.01 |
| 27 | gmm-k8-floor0p5-dirichlet-s512p0 | 8 | 0.5 | dirichlet | 512.0 |
| 28 | gmm-k8-floor0p5-kl-s512p0 | 8 | 0.5 | kl | 512.0 |
| 29 | gmm-k8-floor0p5-kl-s2048p0 | 8 | 0.5 | kl | 2048.0 |
| 30 | gmm-k8-floor1p0-none-s0p0 | 8 | 1.0 | none | 0.0 |
| 31 | gmm-k8-floor1p0-dirichlet-s0p01 | 8 | 1.0 | dirichlet | 0.01 |
| 32 | gmm-k8-floor1p0-dirichlet-s512p0 | 8 | 1.0 | dirichlet | 512.0 |
| 33 | gmm-k8-floor1p0-kl-s512p0 | 8 | 1.0 | kl | 512.0 |
| 34 | gmm-k8-floor1p0-kl-s2048p0 | 8 | 1.0 | kl | 2048.0 |
| 35 | gmm-k8-floor1p25-none-s0p0 | 8 | 1.25 | none | 0.0 |
| 36 | gmm-k8-floor1p25-dirichlet-s0p01 | 8 | 1.25 | dirichlet | 0.01 |
| 37 | gmm-k8-floor1p25-dirichlet-s512p0 | 8 | 1.25 | dirichlet | 512.0 |
| 38 | gmm-k8-floor1p25-kl-s512p0 | 8 | 1.25 | kl | 512.0 |
| 39 | gmm-k8-floor1p25-kl-s2048p0 | 8 | 1.25 | kl | 2048.0 |
| 40 | gmm-k8-floor1p5-none-s0p0 | 8 | 1.5 | none | 0.0 |
| 41 | gmm-k8-floor1p5-dirichlet-s0p01 | 8 | 1.5 | dirichlet | 0.01 |
| 42 | gmm-k8-floor1p5-dirichlet-s512p0 | 8 | 1.5 | dirichlet | 512.0 |
| 43 | gmm-k8-floor1p5-kl-s512p0 | 8 | 1.5 | kl | 512.0 |
| 44 | gmm-k8-floor1p5-kl-s2048p0 | 8 | 1.5 | kl | 2048.0 |
| 45 | gmm-k8-floor1p75-none-s0p0 | 8 | 1.75 | none | 0.0 |
| 46 | gmm-k8-floor1p75-dirichlet-s0p01 | 8 | 1.75 | dirichlet | 0.01 |
| 47 | gmm-k8-floor1p75-dirichlet-s512p0 | 8 | 1.75 | dirichlet | 512.0 |
| 48 | gmm-k8-floor1p75-kl-s512p0 | 8 | 1.75 | kl | 512.0 |
| 49 | gmm-k8-floor1p75-kl-s2048p0 | 8 | 1.75 | kl | 2048.0 |
| 50 | gmm-k16-floor0p5-none-s0p0 | 16 | 0.5 | none | 0.0 |
| 51 | gmm-k16-floor0p5-dirichlet-s0p01 | 16 | 0.5 | dirichlet | 0.01 |
| 52 | gmm-k16-floor0p5-dirichlet-s512p0 | 16 | 0.5 | dirichlet | 512.0 |
| 53 | gmm-k16-floor0p5-kl-s512p0 | 16 | 0.5 | kl | 512.0 |
| 54 | gmm-k16-floor0p5-kl-s2048p0 | 16 | 0.5 | kl | 2048.0 |
| 55 | gmm-k16-floor1p0-none-s0p0 | 16 | 1.0 | none | 0.0 |
| 56 | gmm-k16-floor1p0-dirichlet-s0p01 | 16 | 1.0 | dirichlet | 0.01 |
| 57 | gmm-k16-floor1p0-dirichlet-s512p0 | 16 | 1.0 | dirichlet | 512.0 |
| 58 | gmm-k16-floor1p0-kl-s512p0 | 16 | 1.0 | kl | 512.0 |
| 59 | gmm-k16-floor1p0-kl-s2048p0 | 16 | 1.0 | kl | 2048.0 |
| 60 | gmm-k16-floor1p25-none-s0p0 | 16 | 1.25 | none | 0.0 |
| 61 | gmm-k16-floor1p25-dirichlet-s0p01 | 16 | 1.25 | dirichlet | 0.01 |
| 62 | gmm-k16-floor1p25-dirichlet-s512p0 | 16 | 1.25 | dirichlet | 512.0 |
| 63 | gmm-k16-floor1p25-kl-s512p0 | 16 | 1.25 | kl | 512.0 |
| 64 | gmm-k16-floor1p25-kl-s2048p0 | 16 | 1.25 | kl | 2048.0 |
| 65 | gmm-k16-floor1p5-none-s0p0 | 16 | 1.5 | none | 0.0 |
| 66 | gmm-k16-floor1p5-dirichlet-s0p01 | 16 | 1.5 | dirichlet | 0.01 |
| 67 | gmm-k16-floor1p5-dirichlet-s512p0 | 16 | 1.5 | dirichlet | 512.0 |
| 68 | gmm-k16-floor1p5-kl-s512p0 | 16 | 1.5 | kl | 512.0 |
| 69 | gmm-k16-floor1p5-kl-s2048p0 | 16 | 1.5 | kl | 2048.0 |
| 70 | gmm-k16-floor1p75-none-s0p0 | 16 | 1.75 | none | 0.0 |
| 71 | gmm-k16-floor1p75-dirichlet-s0p01 | 16 | 1.75 | dirichlet | 0.01 |
| 72 | gmm-k16-floor1p75-dirichlet-s512p0 | 16 | 1.75 | dirichlet | 512.0 |
| 73 | gmm-k16-floor1p75-kl-s512p0 | 16 | 1.75 | kl | 512.0 |
| 74 | gmm-k16-floor1p75-kl-s2048p0 | 16 | 1.75 | kl | 2048.0 |
| 75 | gmm-k32-floor0p5-none-s0p0 | 32 | 0.5 | none | 0.0 |
| 76 | gmm-k32-floor0p5-dirichlet-s0p01 | 32 | 0.5 | dirichlet | 0.01 |
| 77 | gmm-k32-floor0p5-dirichlet-s512p0 | 32 | 0.5 | dirichlet | 512.0 |
| 78 | gmm-k32-floor0p5-kl-s512p0 | 32 | 0.5 | kl | 512.0 |
| 79 | gmm-k32-floor0p5-kl-s2048p0 | 32 | 0.5 | kl | 2048.0 |
| 80 | gmm-k32-floor1p0-none-s0p0 | 32 | 1.0 | none | 0.0 |
| 81 | gmm-k32-floor1p0-dirichlet-s0p01 | 32 | 1.0 | dirichlet | 0.01 |
| 82 | gmm-k32-floor1p0-dirichlet-s512p0 | 32 | 1.0 | dirichlet | 512.0 |
| 83 | gmm-k32-floor1p0-kl-s512p0 | 32 | 1.0 | kl | 512.0 |
| 84 | gmm-k32-floor1p0-kl-s2048p0 | 32 | 1.0 | kl | 2048.0 |
| 85 | gmm-k32-floor1p25-none-s0p0 | 32 | 1.25 | none | 0.0 |
| 86 | gmm-k32-floor1p25-dirichlet-s0p01 | 32 | 1.25 | dirichlet | 0.01 |
| 87 | gmm-k32-floor1p25-dirichlet-s512p0 | 32 | 1.25 | dirichlet | 512.0 |
| 88 | gmm-k32-floor1p25-kl-s512p0 | 32 | 1.25 | kl | 512.0 |
| 89 | gmm-k32-floor1p25-kl-s2048p0 | 32 | 1.25 | kl | 2048.0 |
| 90 | gmm-k32-floor1p5-none-s0p0 | 32 | 1.5 | none | 0.0 |
| 91 | gmm-k32-floor1p5-dirichlet-s0p01 | 32 | 1.5 | dirichlet | 0.01 |
| 92 | gmm-k32-floor1p5-dirichlet-s512p0 | 32 | 1.5 | dirichlet | 512.0 |
| 93 | gmm-k32-floor1p5-kl-s512p0 | 32 | 1.5 | kl | 512.0 |
| 94 | gmm-k32-floor1p5-kl-s2048p0 | 32 | 1.5 | kl | 2048.0 |
| 95 | gmm-k32-floor1p75-none-s0p0 | 32 | 1.75 | none | 0.0 |
| 96 | gmm-k32-floor1p75-dirichlet-s0p01 | 32 | 1.75 | dirichlet | 0.01 |
| 97 | gmm-k32-floor1p75-dirichlet-s512p0 | 32 | 1.75 | dirichlet | 512.0 |
| 98 | gmm-k32-floor1p75-kl-s512p0 | 32 | 1.75 | kl | 512.0 |
| 99 | gmm-k32-floor1p75-kl-s2048p0 | 32 | 1.75 | kl | 2048.0 |
| 100 | gmm-k64-floor0p5-none-s0p0 | 64 | 0.5 | none | 0.0 |
| 101 | gmm-k64-floor0p5-dirichlet-s0p01 | 64 | 0.5 | dirichlet | 0.01 |
| 102 | gmm-k64-floor0p5-dirichlet-s512p0 | 64 | 0.5 | dirichlet | 512.0 |
| 103 | gmm-k64-floor0p5-kl-s512p0 | 64 | 0.5 | kl | 512.0 |
| 104 | gmm-k64-floor0p5-kl-s2048p0 | 64 | 0.5 | kl | 2048.0 |
| 105 | gmm-k64-floor1p0-none-s0p0 | 64 | 1.0 | none | 0.0 |
| 106 | gmm-k64-floor1p0-dirichlet-s0p01 | 64 | 1.0 | dirichlet | 0.01 |
| 107 | gmm-k64-floor1p0-dirichlet-s512p0 | 64 | 1.0 | dirichlet | 512.0 |
| 108 | gmm-k64-floor1p0-kl-s512p0 | 64 | 1.0 | kl | 512.0 |
| 109 | gmm-k64-floor1p0-kl-s2048p0 | 64 | 1.0 | kl | 2048.0 |
| 110 | gmm-k64-floor1p25-none-s0p0 | 64 | 1.25 | none | 0.0 |
| 111 | gmm-k64-floor1p25-dirichlet-s0p01 | 64 | 1.25 | dirichlet | 0.01 |
| 112 | gmm-k64-floor1p25-dirichlet-s512p0 | 64 | 1.25 | dirichlet | 512.0 |
| 113 | gmm-k64-floor1p25-kl-s512p0 | 64 | 1.25 | kl | 512.0 |
| 114 | gmm-k64-floor1p25-kl-s2048p0 | 64 | 1.25 | kl | 2048.0 |
| 115 | gmm-k64-floor1p5-none-s0p0 | 64 | 1.5 | none | 0.0 |
| 116 | gmm-k64-floor1p5-dirichlet-s0p01 | 64 | 1.5 | dirichlet | 0.01 |
| 117 | gmm-k64-floor1p5-dirichlet-s512p0 | 64 | 1.5 | dirichlet | 512.0 |
| 118 | gmm-k64-floor1p5-kl-s512p0 | 64 | 1.5 | kl | 512.0 |
| 119 | gmm-k64-floor1p5-kl-s2048p0 | 64 | 1.5 | kl | 2048.0 |
| 120 | gmm-k64-floor1p75-none-s0p0 | 64 | 1.75 | none | 0.0 |
| 121 | gmm-k64-floor1p75-dirichlet-s0p01 | 64 | 1.75 | dirichlet | 0.01 |
| 122 | gmm-k64-floor1p75-dirichlet-s512p0 | 64 | 1.75 | dirichlet | 512.0 |
| 123 | gmm-k64-floor1p75-kl-s512p0 | 64 | 1.75 | kl | 512.0 |
| 124 | gmm-k64-floor1p75-kl-s2048p0 | 64 | 1.75 | kl | 2048.0 |
| 125 | gmm-k128-floor0p5-none-s0p0 | 128 | 0.5 | none | 0.0 |
| 126 | gmm-k128-floor0p5-dirichlet-s0p01 | 128 | 0.5 | dirichlet | 0.01 |
| 127 | gmm-k128-floor0p5-dirichlet-s512p0 | 128 | 0.5 | dirichlet | 512.0 |
| 128 | gmm-k128-floor0p5-kl-s512p0 | 128 | 0.5 | kl | 512.0 |
| 129 | gmm-k128-floor0p5-kl-s2048p0 | 128 | 0.5 | kl | 2048.0 |
| 130 | gmm-k128-floor1p0-none-s0p0 | 128 | 1.0 | none | 0.0 |
| 131 | gmm-k128-floor1p0-dirichlet-s0p01 | 128 | 1.0 | dirichlet | 0.01 |
| 132 | gmm-k128-floor1p0-dirichlet-s512p0 | 128 | 1.0 | dirichlet | 512.0 |
| 133 | gmm-k128-floor1p0-kl-s512p0 | 128 | 1.0 | kl | 512.0 |
| 134 | gmm-k128-floor1p0-kl-s2048p0 | 128 | 1.0 | kl | 2048.0 |
| 135 | gmm-k128-floor1p25-none-s0p0 | 128 | 1.25 | none | 0.0 |
| 136 | gmm-k128-floor1p25-dirichlet-s0p01 | 128 | 1.25 | dirichlet | 0.01 |
| 137 | gmm-k128-floor1p25-dirichlet-s512p0 | 128 | 1.25 | dirichlet | 512.0 |
| 138 | gmm-k128-floor1p25-kl-s512p0 | 128 | 1.25 | kl | 512.0 |
| 139 | gmm-k128-floor1p25-kl-s2048p0 | 128 | 1.25 | kl | 2048.0 |
| 140 | gmm-k128-floor1p5-none-s0p0 | 128 | 1.5 | none | 0.0 |
| 141 | gmm-k128-floor1p5-dirichlet-s0p01 | 128 | 1.5 | dirichlet | 0.01 |
| 142 | gmm-k128-floor1p5-dirichlet-s512p0 | 128 | 1.5 | dirichlet | 512.0 |
| 143 | gmm-k128-floor1p5-kl-s512p0 | 128 | 1.5 | kl | 512.0 |
| 144 | gmm-k128-floor1p5-kl-s2048p0 | 128 | 1.5 | kl | 2048.0 |
| 145 | gmm-k128-floor1p75-none-s0p0 | 128 | 1.75 | none | 0.0 |
| 146 | gmm-k128-floor1p75-dirichlet-s0p01 | 128 | 1.75 | dirichlet | 0.01 |
| 147 | gmm-k128-floor1p75-dirichlet-s512p0 | 128 | 1.75 | dirichlet | 512.0 |
| 148 | gmm-k128-floor1p75-kl-s512p0 | 128 | 1.75 | kl | 512.0 |
| 149 | gmm-k128-floor1p75-kl-s2048p0 | 128 | 1.75 | kl | 2048.0 |
