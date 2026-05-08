# GMM FM Variance > 1 Candidates vs Current 9

parse ok, no critical logs, train/valid dead components = 0, variance coverage > 1, sorted by valid_nll then valid_count_ratio

## Top 14 Variance > 1

| rank | grid | k | coverage | pi | var penalty | valid nll | floor hit rate | floor hit count approx | count ratio |
|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| 1 | 79 | 8 | soft-v1p5-s512-varkl-s512p0-v1p5 | kl:512.0 | hard=0.0 soft=kl:512.0@1.5 | 4351.24 | 0.0000 | 0 | 1.93 |
| 2 | 61 | 8 | soft-v1p5-s512-varkl-s512p0-v1p5 | dirichlet:0.01 | hard=0.0 soft=kl:512.0@1.5 | 4355.82 | 0.0000 | 0 | 2.67 |
| 3 | 70 | 8 | soft-v1p5-s512-varkl-s512p0-v1p5 | dirichlet:512.0 | hard=0.0 soft=kl:512.0@1.5 | 4355.85 | 0.0000 | 0 | 2.20 |
| 4 | 52 | 8 | soft-v1p5-s512-varkl-s512p0-v1p5 | none:0.0 | hard=0.0 soft=kl:512.0@1.5 | 4364.24 | 0.0000 | 0 | 6.26 |
| 5 | 88 | 8 | soft-v1p5-s512-varkl-s512p0-v1p5 | kl:2048.0 | hard=0.0 soft=kl:512.0@1.5 | 4371.38 | 0.0000 | 0 | 99.38 |
| 6 | 43 | 4 | soft-v1p5-s512-varkl-s512p0-v1p5 | kl:2048.0 | hard=0.0 soft=kl:512.0@1.5 | 4444.79 | 0.0000 | 0 | 1.19 |
| 7 | 25 | 4 | soft-v1p5-s512-varkl-s512p0-v1p5 | dirichlet:512.0 | hard=0.0 soft=kl:512.0@1.5 | 4444.96 | 0.0000 | 0 | 1.16 |
| 8 | 7 | 4 | soft-v1p5-s512-varkl-s512p0-v1p5 | none:0.0 | hard=0.0 soft=kl:512.0@1.5 | 4445.20 | 0.0000 | 0 | 1.19 |
| 9 | 16 | 4 | soft-v1p5-s512-varkl-s512p0-v1p5 | dirichlet:0.01 | hard=0.0 soft=kl:512.0@1.5 | 4445.26 | 0.0000 | 0 | 1.17 |
| 10 | 34 | 4 | soft-v1p5-s512-varkl-s512p0-v1p5 | kl:512.0 | hard=0.0 soft=kl:512.0@1.5 | 4445.32 | 0.0000 | 0 | 1.19 |
| 11 | 62 | 8 | soft-v1p5-s2048-varkl-s2048p0-v1p5 | dirichlet:0.01 | hard=0.0 soft=kl:2048.0@1.5 | 4466.04 | 0.0000 | 0 | 300.75 |
| 12 | 44 | 4 | soft-v1p5-s2048-varkl-s2048p0-v1p5 | kl:2048.0 | hard=0.0 soft=kl:2048.0@1.5 | 4470.45 | 0.0000 | 0 | 1.71 |
| 13 | 71 | 8 | soft-v1p5-s2048-varkl-s2048p0-v1p5 | dirichlet:512.0 | hard=0.0 soft=kl:2048.0@1.5 | 4493.34 | 0.0000 | 0 | 454.50 |
| 14 | 80 | 8 | soft-v1p5-s2048-varkl-s2048p0-v1p5 | kl:512.0 | hard=0.0 soft=kl:2048.0@1.5 | 4493.39 | 0.0000 | 0 | 258.43 |

## Current 9 Running Selection

| rank | grid | k | coverage | pi | var penalty | valid nll | floor hit rate | floor hit count approx | count ratio |
|---:|---:|---:|---|---|---|---:|---:|---:|---:|
| 1 | 162 | 32 | ml-no-coverage | kl:512.0 | hard=0.0 soft=none:0.0@1.0 | 4180.63 | 0.0000 | 0 | 199.00 |
| 2 | 171 | 32 | ml-no-coverage | kl:2048.0 | hard=0.0 soft=none:0.0@1.0 | 4180.91 | 0.0000 | 0 | 101.50 |
| 3 | 144 | 32 | ml-no-coverage | dirichlet:0.01 | hard=0.0 soft=none:0.0@1.0 | 4181.87 | 0.0000 | 0 | 213.00 |
| 4 | 136 | 32 | hardv0p5 | none:0.0 | hard=0.5 soft=none:0.0@1.0 | 4187.19 | 0.1013 | 13274 | 2.36 |
| 5 | 145 | 32 | hardv0p5 | dirichlet:0.01 | hard=0.5 soft=none:0.0@1.0 | 4187.28 | 0.0976 | 12791 | 2.40 |
| 6 | 154 | 32 | hardv0p5 | dirichlet:512.0 | hard=0.5 soft=none:0.0@1.0 | 4188.12 | 0.1256 | 16465 | 184.00 |
| 7 | 108 | 16 | ml-no-coverage | dirichlet:512.0 | hard=0.0 soft=none:0.0@1.0 | 4257.40 | 0.0000 | 0 | 2.05 |
| 8 | 109 | 16 | hardv0p5 | dirichlet:512.0 | hard=0.5 soft=none:0.0@1.0 | 4258.62 | 0.0743 | 4869 | 2.30 |
| 9 | 126 | 16 | ml-no-coverage | kl:2048.0 | hard=0.0 soft=none:0.0@1.0 | 4258.89 | 0.0000 | 0 | 2.80 |
