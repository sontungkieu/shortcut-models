# End-Dense Runs Inventory

Nguồn rà soát: `configs/`, `reports/`, `outputs/kaggle`.

| run | train t | source/router | ODE split | kernel | status/output | FID128 best | FID32 best | note |
|---|---|---|---|---|---|---:|---:|---|
| `tide-beta35-13-enddense07-k16-top2-soft075-dir512-jointmix` | Beta(3.5,1.3) | `mix+joint` | `end_dense p=0.7` | `huynhtule/tide-beta35-13-enddense07-k16-top2-soft075-dir51` | CANCEL_ACKNOWLEDGED / downloaded | 9.12 | 11.88 | best step 180000; last FID128 9.72 |
| `tide-beta31p4-enddense07-k16-top2-soft075-dir512-jointmix` | Beta(3,1.4) | `mix+joint` | `end_dense p=0.7` | `nguyncmnhda/tide-beta31p4-enddense07-k16-top2-soft075-dir512` | CANCEL_ACKNOWLEDGED / downloaded | 9.01 | 11.97 | best step 180000; last FID128 9.33 |
| `tide-beta22-enddense07-k16-top2-soft075-dir512-jointmix` | Beta(2,2) | `mix+joint` | `end_dense p=0.7` | `manh1904/tide-beta22-enddense07-k16-top2-soft075-dir512-j` | CANCEL_ACKNOWLEDGED / downloaded | 10.93 | 15.94 | best step 200000; last FID128 10.93 |
| `tide-beta22-enddense08-k16-top2-soft075-dir512-jointmix` | Beta(2,2) | `mix+joint` | `end_dense p=0.8` | `veilwings/tide-beta22-enddense08-k16-top2-soft075-dir512-j` | CANCEL_ACKNOWLEDGED / downloaded | 10.12 | 13.19 | best step 200000; last FID128 10.12 |
| `ar-20260529-r1-enddense-p06-k16-top2` | from autoresearch grid | `see configs/autoresearch` | `end_dense p=0.6` | `` | config present; no local Kaggle output found / not found |  |  |  |
| `ar-20260529-r1-enddense-p08-k16-top2` | from autoresearch grid | `see configs/autoresearch` | `end_dense p=0.8` | `` | config present; no local Kaggle output found / not found |  |  |  |

## Summary

- Có 4 end-dense run có metrics local hiện tại.
- `Beta(3,1.4)+mix+joint+end_dense p=0.7`: FID128 best `9.01`, kém uniform cùng source/router `8.48`.
- `Beta(3.5,1.3)+mix+joint+end_dense p=0.7`: FID128 best `9.12`, kém uniform cùng source/router `8.51`.
- `Beta(2,2)+end_dense`: p=0.8 tốt hơn p=0.7 (`10.12` vs `10.93`) nhưng vẫn yếu.
- Các run không phải end-dense có metric `flow/eval_ode_is_end_dense=0` trong CSV; không tính vào inventory này.
