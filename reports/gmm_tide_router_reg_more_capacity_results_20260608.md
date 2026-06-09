# GMM-TIDE Results

| run | K | step | FID128 best/last | FID32 best/last | router valid/top1 | router loss gap | FM pred/target var | x0/x1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tide-routerreg-uniform-drop03-k16-top2-soft075-dir512 | 16 | 356200 | 7.15/7.15 | 9.42/9.42 | 0.204/0.914 | 0.018 | 0.628 | 0.987 |
| tide-routercap-uniform-low-k16-top2-soft075-dir512 | 16 | 350000 | 7.26/7.26 | 9.57/9.57 | 0.479/0.820 | 0.140 | 0.674 | 0.977 |
| tide-routerreg-uniform-ln-drop03-k16-top2-soft075-dir512 | 16 | 350000 | 7.30/7.30 | 9.63/9.63 | 0.230/0.916 | 0.049 |  |  |
| tide-routerreg-uniform-gn-drop02-k16-top2-soft075-dir512 | 16 | 350000 | 7.33/7.35 | 9.61/9.61 | 0.201/0.914 | 0.016 | 0.633 | 1.002 |
| tide-routercap-uniform-low-ln-drop02-k16-top2-soft075-dir512 | 16 | 350000 | 7.35/7.35 | 9.51/9.51 | 0.433/0.837 | 0.066 | 0.646 | 0.985 |
| tide-routerreg-uniform-gn-drop03-k16-top2-soft075-dir512 | 16 | 350000 | 7.38/7.38 | 9.59/9.59 | 0.228/0.904 | -0.058 | 0.647 | 0.987 |
