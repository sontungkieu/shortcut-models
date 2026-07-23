# Phase-2-Aware C0/C4 200k Confirmation

Source: downloaded diagnostics CSV/JSONL plus W&B history cross-check. Kaggle kernels are COMPLETE. Checkpoints were not downloaded.

Local outputs:

- diagnostics root: `outputs/kaggle/phase2_confirm_c0_c4_200k_20260629/`
- detailed diagnostics table: `reports/gmm_tide_phase2_aware_confirm_c0_c4_200k_diagnostics_detail_20260629.md`

## Result Table

| rank | ID | config | best FID128 | best step | valid loss | pred/target var | curvature128 | straightness128 | router valid loss/top1 | usage H |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | C4-confirm | bridge + sample_topk 200k | 7.751 | 200000 | 0.458 | 0.656 | 0.0204 | 1.112 | 0.236/0.896 | 0.946 |
| 2 | C0-confirm | matched control 200k | 8.044 | 200000 | 0.468 | 0.657 | 0.0205 | 1.113 | 0.209/0.919 | 0.958 |

## FID128 by Eval Step

| step | C0 confirm | C4 confirm | C4 - C0 |
|---:|---:|---:|---:|
| 20000 | 27.693 | 26.152 | -1.540 |
| 40000 | 18.241 | 17.272 | -0.969 |
| 60000 | 14.576 | 13.016 | -1.560 |
| 80000 | 12.367 | 10.863 | -1.504 |
| 100000 | 11.042 | 9.934 | -1.108 |
| 120000 | 9.995 | 9.008 | -0.987 |
| 140000 | 9.421 | 8.890 | -0.531 |
| 160000 | 8.844 | 8.537 | -0.307 |
| 180000 | 8.206 | 8.105 | -0.102 |
| 200000 | 8.044 | 7.751 | -0.292 |

## Interpretation

- Confirm run supports the earlier partial signal: C4 best FID128 is 7.751 at 200000 steps, better than C0 best FID128 8.044 at the same step.
- C4 improves over C0 by 0.292 FID128 in the clean 200k confirmation.
- Both are still worse than the historical baseline 6.969: C0 delta 1.075, C4 delta 0.782.
- Router phase-1 quality again does not explain FID cleanly: C0 router valid loss/top1 0.209/0.919, C4 0.236/0.896, yet C4 has better FID.
- FM diagnostics are nearly matched, so the main isolated difference is source construction: C0 uses weighted source, C4 uses sample-topk source. At 200k, C4 has slightly better valid FM loss (0.458 vs 0.468) and slightly lower curvature/straightness.
- Recommendation: promote bridge+sample_topk to the next matched-control budget/seed check, but do not claim it beats the historical baseline yet.
