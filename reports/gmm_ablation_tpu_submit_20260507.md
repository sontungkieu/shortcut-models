# GMM Ablation Submit Report

- Planned: 150
- Submitted: 7
- Failed: 51
- Not submitted: 92

## Submitted

| grid_index | owner | accelerator | run_name | kernel | status |
|---:|---|---|---|---|---|
| 0 | casihoavinh | TpuV5E8 | gmm-k4-floor0p5-none-s0p0 | `casihoavinh/gmm-k4-floor0p5-none-s0p0-casihoavinh-20260507-0` | KernelWorkerStatus.QUEUED |
| 1 | codemaivanngu | TpuV5E8 | gmm-k4-floor0p5-dirichlet-s0p01 | `codemaivanngu/gmm-k4-floor0p5-dirichlet-s0p01-codemaivanngu-20` | KernelWorkerStatus.QUEUED |
| 3 | huynhtule | TpuV5E8 | gmm-k4-floor0p5-kl-s512p0 | `huynhtule/gmm-k4-floor0p5-kl-s512p0-huynhtule-20260507-064` | KernelWorkerStatus.QUEUED |
| 4 | iamlonely | TpuV5E8 | gmm-k4-floor0p5-kl-s2048p0 | `iamlonely/gmm-k4-floor0p5-kl-s2048p0-iamlonely-20260507-06` | KernelWorkerStatus.QUEUED |
| 6 | kiuvithong | TpuV5E8 | gmm-k4-floor1p0-dirichlet-s0p01 | `kiuvithong/gmm-k4-floor1p0-dirichlet-s0p01-kiuvithong-20260` | KernelWorkerStatus.QUEUED |
| 8 | nguyncmnhda | TpuV5E8 | gmm-k4-floor1p0-kl-s512p0 | `nguyncmnhda/gmm-k4-floor1p0-kl-s512p0-nguyncmnhda-20260507-0` | KernelWorkerStatus.QUEUED |
| 10 | veilwings | TpuV5E8 | gmm-k4-floor1p25-none-s0p0 | `veilwings/gmm-k4-floor1p25-none-s0p0-veilwings-20260507-06` | KernelWorkerStatus.QUEUED |

## Failed

| grid_index | owner | run_name | error |
|---:|---|---|---|
| 2 | hoanganpham123 | gmm-k4-floor0p5-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: Cannot access kernel 'hoanganpham123/gmm-k4-floor0p5-dirichlet-s512p0-hoanganpham123-' (Permission 'kernels.get' was denied). The most likely cause is a wrong kernel slug. The benchmark_task_slug returned by get_benchmark_leaderboard differs from the actual kernel slug — use the slug from the notebook URL (kaggle.com/code/owner/KERNEL-SLUG), not from the leaderboard. It can also occur if the notebook is private. |
| 5 | kieuhongquan | gmm-k4-floor1p0-none-s0p0 | kaggle kernels status failed with exit code 1: Cannot access kernel 'kieuhongquan/gmm-k4-floor1p0-none-s0p0-kieuhongquan-20260507-' (Permission 'kernels.get' was denied). The most likely cause is a wrong kernel slug. The benchmark_task_slug returned by get_benchmark_leaderboard differs from the actual kernel slug — use the slug from the notebook URL (kaggle.com/code/owner/KERNEL-SLUG), not from the leaderboard. It can also occur if the notebook is private. |
| 7 | manh1904 | gmm-k4-floor1p0-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 9 | no1ceboy | gmm-k4-floor1p0-kl-s2048p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 11 | victorharvey27 | gmm-k4-floor1p25-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: Cannot access kernel 'victorharvey27/gmm-k4-floor1p25-dirichlet-s0p01-victorharvey27-' (Permission 'kernels.get' was denied). The most likely cause is a wrong kernel slug. The benchmark_task_slug returned by get_benchmark_leaderboard differs from the actual kernel slug — use the slug from the notebook URL (kaggle.com/code/owner/KERNEL-SLUG), not from the leaderboard. It can also occur if the notebook is private. |
| 12 | casihoavinh | gmm-k4-floor1p25-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 13 | codemaivanngu | gmm-k4-floor1p25-kl-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 14 | hoanganpham123 | gmm-k4-floor1p25-kl-s2048p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 15 | huynhtule | gmm-k4-floor1p5-none-s0p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 16 | iamlonely | gmm-k4-floor1p5-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 17 | kieuhongquan | gmm-k4-floor1p5-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 18 | kiuvithong | gmm-k4-floor1p5-kl-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 19 | manh1904 | gmm-k4-floor1p5-kl-s2048p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 20 | nguyncmnhda | gmm-k4-floor1p75-none-s0p0 | kaggle kernels status failed with exit code 1: Cannot access kernel 'nguyncmnhda/gmm-k4-floor1p75-none-s0p0-nguyncmnhda-20260507-' (Permission 'kernels.get' was denied). The most likely cause is a wrong kernel slug. The benchmark_task_slug returned by get_benchmark_leaderboard differs from the actual kernel slug — use the slug from the notebook URL (kaggle.com/code/owner/KERNEL-SLUG), not from the leaderboard. It can also occur if the notebook is private. |
| 21 | no1ceboy | gmm-k4-floor1p75-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 22 | veilwings | gmm-k4-floor1p75-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 23 | victorharvey27 | gmm-k4-floor1p75-kl-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 24 | casihoavinh | gmm-k4-floor1p75-kl-s2048p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 25 | codemaivanngu | gmm-k8-floor0p5-none-s0p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 26 | hoanganpham123 | gmm-k8-floor0p5-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 27 | huynhtule | gmm-k8-floor0p5-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 28 | iamlonely | gmm-k8-floor0p5-kl-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 29 | kieuhongquan | gmm-k8-floor0p5-kl-s2048p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 30 | kiuvithong | gmm-k8-floor1p0-none-s0p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 31 | manh1904 | gmm-k8-floor1p0-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 32 | nguyncmnhda | gmm-k8-floor1p0-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 33 | no1ceboy | gmm-k8-floor1p0-kl-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 34 | veilwings | gmm-k8-floor1p0-kl-s2048p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 35 | victorharvey27 | gmm-k8-floor1p25-none-s0p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 36 | casihoavinh | gmm-k8-floor1p25-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 37 | codemaivanngu | gmm-k8-floor1p25-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: Cannot access kernel 'codemaivanngu/gmm-k8-floor1p25-dirichlet-s512p0-codemaivanngu-' (Permission 'kernels.get' was denied). The most likely cause is a wrong kernel slug. The benchmark_task_slug returned by get_benchmark_leaderboard differs from the actual kernel slug — use the slug from the notebook URL (kaggle.com/code/owner/KERNEL-SLUG), not from the leaderboard. It can also occur if the notebook is private. |
| 38 | hoanganpham123 | gmm-k8-floor1p25-kl-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 39 | huynhtule | gmm-k8-floor1p25-kl-s2048p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 40 | iamlonely | gmm-k8-floor1p5-none-s0p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 41 | kieuhongquan | gmm-k8-floor1p5-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 42 | kiuvithong | gmm-k8-floor1p5-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 43 | manh1904 | gmm-k8-floor1p5-kl-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 44 | nguyncmnhda | gmm-k8-floor1p5-kl-s2048p0 | kaggle kernels status failed with exit code 1: Cannot access kernel 'nguyncmnhda/gmm-k8-floor1p5-kl-s2048p0-nguyncmnhda-20260507-' (Permission 'kernels.get' was denied). The most likely cause is a wrong kernel slug. The benchmark_task_slug returned by get_benchmark_leaderboard differs from the actual kernel slug — use the slug from the notebook URL (kaggle.com/code/owner/KERNEL-SLUG), not from the leaderboard. It can also occur if the notebook is private. |
| 45 | no1ceboy | gmm-k8-floor1p75-none-s0p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 46 | veilwings | gmm-k8-floor1p75-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 47 | victorharvey27 | gmm-k8-floor1p75-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 48 | casihoavinh | gmm-k8-floor1p75-kl-s512p0 | kaggle kernels status failed with exit code 1: Cannot access kernel 'casihoavinh/gmm-k8-floor1p75-kl-s512p0-casihoavinh-20260507-' (Permission 'kernels.get' was denied). The most likely cause is a wrong kernel slug. The benchmark_task_slug returned by get_benchmark_leaderboard differs from the actual kernel slug — use the slug from the notebook URL (kaggle.com/code/owner/KERNEL-SLUG), not from the leaderboard. It can also occur if the notebook is private. |
| 49 | codemaivanngu | gmm-k8-floor1p75-kl-s2048p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 50 | hoanganpham123 | gmm-k16-floor0p5-none-s0p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 51 | huynhtule | gmm-k16-floor0p5-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 52 | iamlonely | gmm-k16-floor0p5-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 53 | kieuhongquan | gmm-k16-floor0p5-kl-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 54 | kiuvithong | gmm-k16-floor0p5-kl-s2048p0 | kaggle kernels status failed with exit code 1: Cannot access kernel 'kiuvithong/gmm-k16-floor0p5-kl-s2048p0-kiuvithong-20260507-' (Permission 'kernels.get' was denied). The most likely cause is a wrong kernel slug. The benchmark_task_slug returned by get_benchmark_leaderboard differs from the actual kernel slug — use the slug from the notebook URL (kaggle.com/code/owner/KERNEL-SLUG), not from the leaderboard. It can also occur if the notebook is private. |
| 55 | manh1904 | gmm-k16-floor1p0-none-s0p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 56 | nguyncmnhda | gmm-k16-floor1p0-dirichlet-s0p01 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |
| 57 | no1ceboy | gmm-k16-floor1p0-dirichlet-s512p0 | kaggle kernels status failed with exit code 1: 404 Client Error: Not Found for url: https://api.kaggle.com/v1/kernels.KernelsApiService/GetKernelSessionStatus |

## Not Submitted

| grid_index | owner | run_name |
|---:|---|---|
| 58 | veilwings | gmm-k16-floor1p0-kl-s512p0 |
| 59 | victorharvey27 | gmm-k16-floor1p0-kl-s2048p0 |
| 60 | casihoavinh | gmm-k16-floor1p25-none-s0p0 |
| 61 | codemaivanngu | gmm-k16-floor1p25-dirichlet-s0p01 |
| 62 | hoanganpham123 | gmm-k16-floor1p25-dirichlet-s512p0 |
| 63 | huynhtule | gmm-k16-floor1p25-kl-s512p0 |
| 64 | iamlonely | gmm-k16-floor1p25-kl-s2048p0 |
| 65 | kieuhongquan | gmm-k16-floor1p5-none-s0p0 |
| 66 | kiuvithong | gmm-k16-floor1p5-dirichlet-s0p01 |
| 67 | manh1904 | gmm-k16-floor1p5-dirichlet-s512p0 |
| 68 | nguyncmnhda | gmm-k16-floor1p5-kl-s512p0 |
| 69 | no1ceboy | gmm-k16-floor1p5-kl-s2048p0 |
| 70 | veilwings | gmm-k16-floor1p75-none-s0p0 |
| 71 | victorharvey27 | gmm-k16-floor1p75-dirichlet-s0p01 |
| 72 | casihoavinh | gmm-k16-floor1p75-dirichlet-s512p0 |
| 73 | codemaivanngu | gmm-k16-floor1p75-kl-s512p0 |
| 74 | hoanganpham123 | gmm-k16-floor1p75-kl-s2048p0 |
| 75 | huynhtule | gmm-k32-floor0p5-none-s0p0 |
| 76 | iamlonely | gmm-k32-floor0p5-dirichlet-s0p01 |
| 77 | kieuhongquan | gmm-k32-floor0p5-dirichlet-s512p0 |
| 78 | kiuvithong | gmm-k32-floor0p5-kl-s512p0 |
| 79 | manh1904 | gmm-k32-floor0p5-kl-s2048p0 |
| 80 | nguyncmnhda | gmm-k32-floor1p0-none-s0p0 |
| 81 | no1ceboy | gmm-k32-floor1p0-dirichlet-s0p01 |
| 82 | veilwings | gmm-k32-floor1p0-dirichlet-s512p0 |
| 83 | victorharvey27 | gmm-k32-floor1p0-kl-s512p0 |
| 84 | casihoavinh | gmm-k32-floor1p0-kl-s2048p0 |
| 85 | codemaivanngu | gmm-k32-floor1p25-none-s0p0 |
| 86 | hoanganpham123 | gmm-k32-floor1p25-dirichlet-s0p01 |
| 87 | huynhtule | gmm-k32-floor1p25-dirichlet-s512p0 |
| 88 | iamlonely | gmm-k32-floor1p25-kl-s512p0 |
| 89 | kieuhongquan | gmm-k32-floor1p25-kl-s2048p0 |
| 90 | kiuvithong | gmm-k32-floor1p5-none-s0p0 |
| 91 | manh1904 | gmm-k32-floor1p5-dirichlet-s0p01 |
| 92 | nguyncmnhda | gmm-k32-floor1p5-dirichlet-s512p0 |
| 93 | no1ceboy | gmm-k32-floor1p5-kl-s512p0 |
| 94 | veilwings | gmm-k32-floor1p5-kl-s2048p0 |
| 95 | victorharvey27 | gmm-k32-floor1p75-none-s0p0 |
| 96 | casihoavinh | gmm-k32-floor1p75-dirichlet-s0p01 |
| 97 | codemaivanngu | gmm-k32-floor1p75-dirichlet-s512p0 |
| 98 | hoanganpham123 | gmm-k32-floor1p75-kl-s512p0 |
| 99 | huynhtule | gmm-k32-floor1p75-kl-s2048p0 |
| 100 | iamlonely | gmm-k64-floor0p5-none-s0p0 |
| 101 | kieuhongquan | gmm-k64-floor0p5-dirichlet-s0p01 |
| 102 | kiuvithong | gmm-k64-floor0p5-dirichlet-s512p0 |
| 103 | manh1904 | gmm-k64-floor0p5-kl-s512p0 |
| 104 | nguyncmnhda | gmm-k64-floor0p5-kl-s2048p0 |
| 105 | no1ceboy | gmm-k64-floor1p0-none-s0p0 |
| 106 | veilwings | gmm-k64-floor1p0-dirichlet-s0p01 |
| 107 | victorharvey27 | gmm-k64-floor1p0-dirichlet-s512p0 |
| 108 | casihoavinh | gmm-k64-floor1p0-kl-s512p0 |
| 109 | codemaivanngu | gmm-k64-floor1p0-kl-s2048p0 |
| 110 | hoanganpham123 | gmm-k64-floor1p25-none-s0p0 |
| 111 | huynhtule | gmm-k64-floor1p25-dirichlet-s0p01 |
| 112 | iamlonely | gmm-k64-floor1p25-dirichlet-s512p0 |
| 113 | kieuhongquan | gmm-k64-floor1p25-kl-s512p0 |
| 114 | kiuvithong | gmm-k64-floor1p25-kl-s2048p0 |
| 115 | manh1904 | gmm-k64-floor1p5-none-s0p0 |
| 116 | nguyncmnhda | gmm-k64-floor1p5-dirichlet-s0p01 |
| 117 | no1ceboy | gmm-k64-floor1p5-dirichlet-s512p0 |
| 118 | veilwings | gmm-k64-floor1p5-kl-s512p0 |
| 119 | victorharvey27 | gmm-k64-floor1p5-kl-s2048p0 |
| 120 | casihoavinh | gmm-k64-floor1p75-none-s0p0 |
| 121 | codemaivanngu | gmm-k64-floor1p75-dirichlet-s0p01 |
| 122 | hoanganpham123 | gmm-k64-floor1p75-dirichlet-s512p0 |
| 123 | huynhtule | gmm-k64-floor1p75-kl-s512p0 |
| 124 | iamlonely | gmm-k64-floor1p75-kl-s2048p0 |
| 125 | kieuhongquan | gmm-k128-floor0p5-none-s0p0 |
| 126 | kiuvithong | gmm-k128-floor0p5-dirichlet-s0p01 |
| 127 | manh1904 | gmm-k128-floor0p5-dirichlet-s512p0 |
| 128 | nguyncmnhda | gmm-k128-floor0p5-kl-s512p0 |
| 129 | no1ceboy | gmm-k128-floor0p5-kl-s2048p0 |
| 130 | veilwings | gmm-k128-floor1p0-none-s0p0 |
| 131 | victorharvey27 | gmm-k128-floor1p0-dirichlet-s0p01 |
| 132 | casihoavinh | gmm-k128-floor1p0-dirichlet-s512p0 |
| 133 | codemaivanngu | gmm-k128-floor1p0-kl-s512p0 |
| 134 | hoanganpham123 | gmm-k128-floor1p0-kl-s2048p0 |
| 135 | huynhtule | gmm-k128-floor1p25-none-s0p0 |
| 136 | iamlonely | gmm-k128-floor1p25-dirichlet-s0p01 |
| 137 | kieuhongquan | gmm-k128-floor1p25-dirichlet-s512p0 |
| 138 | kiuvithong | gmm-k128-floor1p25-kl-s512p0 |
| 139 | manh1904 | gmm-k128-floor1p25-kl-s2048p0 |
| 140 | nguyncmnhda | gmm-k128-floor1p5-none-s0p0 |
| 141 | no1ceboy | gmm-k128-floor1p5-dirichlet-s0p01 |
| 142 | veilwings | gmm-k128-floor1p5-dirichlet-s512p0 |
| 143 | victorharvey27 | gmm-k128-floor1p5-kl-s512p0 |
| 144 | casihoavinh | gmm-k128-floor1p5-kl-s2048p0 |
| 145 | codemaivanngu | gmm-k128-floor1p75-none-s0p0 |
| 146 | hoanganpham123 | gmm-k128-floor1p75-dirichlet-s0p01 |
| 147 | huynhtule | gmm-k128-floor1p75-dirichlet-s512p0 |
| 148 | iamlonely | gmm-k128-floor1p75-kl-s512p0 |
| 149 | kieuhongquan | gmm-k128-floor1p75-kl-s2048p0 |
