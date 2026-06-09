# GMM-TIDE Results

| run | K | step | FID128 best/last | FID32 best/last | router valid/top1 | router loss gap | FM pred/target var | x0/x1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tide-resume150-beta31-k16-top2-soft075-dir512-jointmix | None | 376600 | 7.33/7.33 | 8.42/8.42 | / |  | 0.649 | 1.013 |
| tide-beta35-13-k16-top2-soft075-dir512-jointmix | 16 | 220000 | 8.51/9.00 | 9.77/10.60 | 0.172/0.930 | 0.074 | 0.661 | 0.983 |
| tide-beta35-13-enddense07-k16-top2-soft075-dir512-jointmix | 16 | 220000 | 9.12/9.72 | 12.18/12.94 | 0.167/0.930 | -0.105 | 0.666 | 0.991 |
| tide-beta41fresh-k16-top2-soft075-dir512-jointmix | 16 | 220000 | 9.35/11.27 | 10.08/12.78 | 0.228/0.915 | -0.017 | 0.654 | 0.981 |
