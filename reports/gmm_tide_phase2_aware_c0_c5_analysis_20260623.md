# Phase-2-Aware C0-C5 Analysis

Source: W&B sampled history because Kaggle output file lists are empty for all six canceled kernels. Checkpoints were not downloaded.

## Status

| ID | run | owner | Kaggle status | W&B train state | last sampled step |
|---|---|---|---|---|---:|
| C0 | `tide-ars-c0-control-k16-top2-soft075-dir512` | anhhaphan | CANCEL_ACKNOWLEDGED | crashed | 232800 |
| C1 | `tide-ars-c1-bridge11-k16-top2-soft075-dir512` | bangchi | CANCEL_ACKNOWLEDGED | crashed | 219900 |
| C2 | `tide-ars-c2-d4drop02-k16-top2-soft075-dir512` | casihoavinh | CANCEL_ACKNOWLEDGED | crashed | 219900 |
| C3 | `tide-ars-c3-bridge11-d4drop02-k16-top2-soft075-dir512` | codemaivanngu | CANCEL_ACKNOWLEDGED | crashed | 219900 |
| C4 | `tide-ars-c4-bridge11-sampletopk-k16-top2-soft075-dir512` | ctlcmleon | CANCEL_ACKNOWLEDGED | crashed | 219900 |
| C5 | `tide-ars-c5-bridge11-pcalw5-k16-top2-soft075-dir512` | damtrunghieu | CANCEL_ACKNOWLEDGED | crashed | 219900 |

## FID128 Ranking

| rank | ID | config | best FID128 | best step | last FID128 | last eval step | pred/target var last | curvature@best | straightness@best | router valid loss/top1 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | C4 | bridge + sample_topk | 7.698 | 200000 | 7.698 | 200000 | 0.635 | 0.0204 | 1.112 | 0.255/0.885 |
| 2 | C0 | matched control | 7.873 | 220000 | 7.873 | 220000 | 0.626 | 0.0205 | 1.111 | 0.198/0.909 |
| 3 | C3 | bridge + d4 dropout 0.2 | 7.955 | 200000 | 7.955 | 200000 | 0.637 | 0.0203 | 1.112 | 0.210/0.911 |
| 4 | C5 | bridge + PCA init + Lloyd5 | 7.957 | 180000 | 8.031 | 200000 | 0.648 | 0.0204 | 1.113 | 0.231/0.899 |
| 5 | C1 | bridge B(1,1) | 8.034 | 200000 | 8.034 | 200000 | 0.647 | 0.0205 | 1.112 | 0.288/0.881 |
| 6 | C2 | d4 dropout 0.2 | 8.323 | 200000 | 8.323 | 200000 | 0.626 | 0.0205 | 1.113 | 0.118/0.962 |

## Main Observations

- Matched control C0 reached best FID128 7.873 at step 220000, still worse than historical baseline 6.969.
- Best in this batch is C4 (bridge + sample_topk) with FID128 7.698; delta vs C0 is -0.175.
- C4 vs C0: delta FID128 -0.175; router valid loss/top1 0.255/0.885; var ratio last 0.635.
- C3 vs C0: delta FID128 0.081; router valid loss/top1 0.210/0.911; var ratio last 0.637.
- C5 vs C0: delta FID128 0.084; router valid loss/top1 0.231/0.899; var ratio last 0.648.
- C1 vs C0: delta FID128 0.161; router valid loss/top1 0.288/0.881; var ratio last 0.647.
- C2 vs C0: delta FID128 0.450; router valid loss/top1 0.118/0.962; var ratio last 0.626.

## Decision

- No run clearly beats the historical baseline. Treat these as partial 230k-step evidence, not final completed runs.
- Because all kernels ended as canceled and published no diagnostics files, future targeted jobs should either finish under the Kaggle wall-time or save/publish shorter runs. A practical next batch is to cap at 220k or 240k steps, keep eval every 20k, and rerun only C0 plus the best non-control candidate(s).
- Do not promote router/GMM phase-1 metrics alone; this batch again needs FID/flow diagnostics as the gate.
