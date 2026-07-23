# Source10 Data Availability

- Source report: `reports/gmm_tide_fm_beta_source10_combined_20260606.json`
- Kaggle pull report: `reports/gmm_tide_fm_beta_source10_diag_pull_20260609.json`
- W&B summaries: `outputs/kaggle_metrics_20260606/wandb/wandb_fid_summary.csv`, `outputs/kaggle_metrics_20260606/wandb/wandb_target_summary.csv`
- W&B raw history: `outputs/kaggle_metrics_20260609/wandb_source10_history/summary.json`

## Summary

- Runs: 10
- Kaggle latest status: 10 `CANCEL_ACKNOWLEDGED`, 0 `COMPLETE`.
- W&B FID rows: 10/10.
- W&B target rows: 10/10.
- Raw history all phases: 10/10, 30 CSV files, 22835 train-history rows.
- No checkpoint, `.pkl`, `.npz`, `ckpts`, or `.venv` files were requested by the diagnostics/W&B pulls.

## FID And Training Data

| run | kg status | FID128 best@step | FID32 best | FID4 best | train step/state | valid loss | pred/target var | topk mass | angle disp | hist rows prep/router/train |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tide-srcsweep-x1frozen-beta31p4-k16-top2-tau125-soft075-dir512 | CANCEL_ACKNOWLEDGED | 7.986@180000 | 9.515 | 59.121 | 197600/ | 0.4508 | 0.637 | 0.9987 | 0.000455 | 1/51/2334 |
| tide-srcsweep-x1frozen-beta31p4-k16-top4-tau075-soft075-dir512 | CANCEL_ACKNOWLEDGED | 8.006@180000 | 9.555 | 59.045 | 198000/ | 0.4607 | 0.635 | 1.0000 | 0.000056 | 1/51/2341 |
| tide-srcsweep-x1frozen-beta31p4-k16-top1-tau1-soft075-dir512 | CANCEL_ACKNOWLEDGED | 8.207@180000 | 9.526 | 59.861 | 199900/ | 0.4692 | 0.651 | 0.9982 | -0.000000 | 1/51/2373 |
| tide-srcmain-x1frozen-beta31p4-k16-top2-soft075-dir512 | CANCEL_ACKNOWLEDGED | 8.386@180000 | 9.922 | 59.996 | 199900/ | 0.4719 | 0.640 | 0.9996 | 0.000491 | 1/51/2399 |
| tide-srcmain-mixfrozen-beta31p4-k16-top2-soft075-dir512 | CANCEL_ACKNOWLEDGED | 8.571@180000 | 10.042 | 58.589 | 199900/ | 0.4751 | 0.665 | 0.9997 | 0.000069 | 1/51/2302 |
| tide-srcsweep-x1frozen-beta31p4-k16-top2-tau075-soft075-dir512 | CANCEL_ACKNOWLEDGED | 8.665@180000 | 10.192 | 59.758 | 199900/ | 0.4705 | 0.649 | 1.0000 | 0.000095 | 1/51/2263 |
| tide-risk-x1joint-gumbel075-beta31p4-k16-top2-soft075-dir512 | CANCEL_ACKNOWLEDGED | 9.007@140000 | 10.710 | 59.825 | 179900/ | 0.4463 | 0.661 | 0.9932 | 0.014216 | 1/51/2080 |
| tide-risk-mixjoint-gumbel075-beta31p4-k16-top2-soft075-dir512 | CANCEL_ACKNOWLEDGED | 9.086@140000 | 10.542 | 58.497 | 179900/running | 0.4495 | 0.657 | 0.9922 | 0.012281 | 1/51/2055 |
| tide-srcmain-x1frozen-beta31-k16-top2-soft075-dir512 | CANCEL_ACKNOWLEDGED | 9.106@180000 | 10.777 | 58.228 | 197400/ | 0.4752 | 0.655 | 0.9997 | 0.000401 | 1/51/2399 |
| tide-srcmain-mixfrozen-beta31-k16-top2-soft075-dir512 | CANCEL_ACKNOWLEDGED | 9.484@160000 | 11.013 | 58.287 | 179900/ | 0.4729 | 0.632 | 0.9998 | 0.000159 | 1/51/2289 |

## Per-Run Files

| run | prep csv | router csv | train csv |
|---|---|---|---|
| tide-srcmain-x1frozen-beta31-k16-top2-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-x1frozen-beta31-k16-top2-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-x1frozen-beta31-k16-top2-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-x1frozen-beta31-k16-top2-soft075-dir512__train.csv` |
| tide-srcmain-x1frozen-beta31p4-k16-top2-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-x1frozen-beta31p4-k16-top2-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-x1frozen-beta31p4-k16-top2-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-x1frozen-beta31p4-k16-top2-soft075-dir512__train.csv` |
| tide-srcmain-mixfrozen-beta31-k16-top2-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-mixfrozen-beta31-k16-top2-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-mixfrozen-beta31-k16-top2-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-mixfrozen-beta31-k16-top2-soft075-dir512__train.csv` |
| tide-srcmain-mixfrozen-beta31p4-k16-top2-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-mixfrozen-beta31p4-k16-top2-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-mixfrozen-beta31p4-k16-top2-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcmain-mixfrozen-beta31p4-k16-top2-soft075-dir512__train.csv` |
| tide-srcsweep-x1frozen-beta31p4-k16-top1-tau1-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top1-tau1-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top1-tau1-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top1-tau1-soft075-dir512__train.csv` |
| tide-srcsweep-x1frozen-beta31p4-k16-top2-tau075-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top2-tau075-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top2-tau075-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top2-tau075-soft075-dir512__train.csv` |
| tide-srcsweep-x1frozen-beta31p4-k16-top2-tau125-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top2-tau125-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top2-tau125-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top2-tau125-soft075-dir512__train.csv` |
| tide-srcsweep-x1frozen-beta31p4-k16-top4-tau075-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top4-tau075-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top4-tau075-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-srcsweep-x1frozen-beta31p4-k16-top4-tau075-soft075-dir512__train.csv` |
| tide-risk-x1joint-gumbel075-beta31p4-k16-top2-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-risk-x1joint-gumbel075-beta31p4-k16-top2-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-risk-x1joint-gumbel075-beta31p4-k16-top2-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-risk-x1joint-gumbel075-beta31p4-k16-top2-soft075-dir512__train.csv` |
| tide-risk-mixjoint-gumbel075-beta31p4-k16-top2-soft075-dir512 | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-risk-mixjoint-gumbel075-beta31p4-k16-top2-soft075-dir512__prep.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-risk-mixjoint-gumbel075-beta31p4-k16-top2-soft075-dir512__router.csv` | `outputs/kaggle_metrics_20260609/wandb_source10_history/tide-risk-mixjoint-gumbel075-beta31p4-k16-top2-soft075-dir512__train.csv` |
