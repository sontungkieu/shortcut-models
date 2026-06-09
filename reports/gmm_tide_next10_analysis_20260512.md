# GMM-TIDE Timeout Run Analysis 2026-05-12

Downloaded diagnostics from `reports/gmm_tide_fm_resubmit_fixed_20260511.json` into `outputs/kaggle/moe2_resubmit_fixed_20260512/`.

## Ranking

| rank | run | K | FID128 | FID32 | FID4 | step | router valid loss/top1/use | pred/target var | x0/x1 | floor hit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `tide-k16-top2-softv0p75-s128-dir512` | 16 top2 | 6.969 | 9.295 | 68.76 | 365600 | 0.185/0.930/0.948 | 0.674 | 1.021 | 0.000 |
| 2 | `tide-k32-top2-softv0p75-s128-dir001` | 32 top2 | 7.118 | 9.380 | 69.27 | 367400 | 0.240/0.907/0.904 | 0.649 | 1.014 | 0.000 |
| 3 | `tide-k32-top2-g136-none-hardv0p5` | 32 top2 | 7.256 | 9.669 | 70.36 | 364700 | 0.261/0.899/0.903 | 0.654 | 0.975 | 0.113 |
| 4 | `tide-k32-top2-g145-dir001-hardv0p5` | 32 top2 | 7.528 | 9.989 | 70.02 | 373700 | 0.241/0.909/0.924 | 0.638 | 1.008 | 0.099 |
| 5 | `tide-k32-top4-g136-none-hardv0p5` | 32 top4 | 7.598 | 9.984 | 70.39 | 363800 | 0.232/0.911/0.913 | 0.662 | 1.004 | 0.102 |
| 6 | `tide-k32-top2-g146-dir001-hardv1p0` | 32 top2 | 7.623 | 9.728 | 65.96 | 365700 | 0.275/0.886/0.912 | 0.648 | 1.088 | 0.842 |

## Observations

- Best run is `tide-k16-top2-softv0p75-s128-dir512`: FID128 `6.969`, FID32 `9.295`, router valid top1 `0.930`, and no hard floor hits.
- Soft variance pressure around target variance `0.75` beats hard floors in this sample. K32 soft is second; K32 hard floor `0.5` trails; hard floor `1.0` is worst for FID128 and has floor hit rate `0.842`.
- Router top-k mass is almost saturated (`~0.999`), so top-k does little at temperature `1.0`. The next grid tests router temperature `1.5` to make top-k mixing real.
- Router overfit is not severe: valid/train gaps are small or negative. The next grid keeps router capacity/training fixed and sweeps GMM variance pressure instead.
- The runs timed out/cancelled after roughly 364k-374k train steps, but every run had already logged eval through step 350k. The next grid sets `train_max_steps=350000` to finish cleanly before timeout.

## Recommended Next Grid

Use `configs/gmm_tide_fm_next10_grid.json`. It contains 10 configs:

| idx | run | purpose |
|---:|---|---|
| 1 | `tide-next-k16-top2-softv0p65-s128-dir512` | lower target variance around best K16 soft run |
| 2 | `tide-next-k16-top2-softv0p75-s64-dir512` | weaker soft variance strength around best K16 run |
| 3 | `tide-next-k16-top2-softv0p75-s256-dir512` | stronger soft variance strength around best K16 run |
| 4 | `tide-next-k16-top2-softv0p85-s128-dir512` | higher target variance around best K16 soft run |
| 5 | `tide-next-k16-top2-softv0p75-s128-dir512-t1p5` | temperature check: soften deterministic router without changing top2 |
| 6 | `tide-next-k16-top4-softv0p75-s128-dir512-t1p5` | temperature plus top4 check: make weighted top-k mixing nontrivial |
| 7 | `tide-next-k16-top2-softv0p75-s128-kl512` | compare KL pi prior against best K16 Dirichlet prior |
| 8 | `tide-next-k32-top2-softv0p65-s128-dir001` | K32 soft lower target variance |
| 9 | `tide-next-k32-top2-softv0p75-s64-dir001` | K32 soft weaker variance strength |
| 10 | `tide-next-k32-top2-softv0p75-s128-dir512` | K32 soft with strong Dirichlet prior like the K16 winner |

Submit example:

```bash
python3 scripts/submit_gmm_tide_fm_jobs.py \
  --grid-config configs/gmm_tide_fm_next10_grid.json \
  --owners all \
  --exclude-owners kieutung,no1ceboy \
  --accelerator tpu \
  --max-submit-per-owner 1 \
  --report-path reports/gmm_tide_fm_next10_submit_20260512.json
```
