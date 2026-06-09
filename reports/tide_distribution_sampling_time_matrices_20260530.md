# TIDE Distribution / Sampling / Timestep Matrices

Nguồn: `reports/tide_recent_analysis_20260530.csv`. Các run đều là partial/timeout trừ baseline lịch sử; so sánh các ô không phải lúc nào cũng cùng số step.

## Matrix 1: Best Observed By Training `t` Distribution

| training t distribution | mean(t) | best observed config | FID128 best | FID32 best | valid loss | note |
|---|---:|---|---:|---:|---:|---|
| `default/old` |  | `tide-k16-top2-softv0p75-s128-dir512` | 6.97@350000 | 9.29 | 0.4741 | baseline cũ |
| `Beta(3,1)` | 0.750 | `tide-resume150-beta31-k16-top2-soft075-dir512-jointmix` | 7.33@360000 | 8.42 | 0.4678 | jointmix, uniform ODE; có resume dài hơn |
| `Beta(3,1.4)` | 0.682 | `tide-beta31p4-x1frozen-k16-top2-soft075-dir512-r2` | 7.93@220000 | 9.44 | 0.4573 | x1+frozen |
| `Beta(3.5,1.3)` | 0.729 | `tide-beta35-13-mixfrozen-k16-top2-soft075-dir512` | 8.06@220000 | 9.59 | 0.4643 | mix+frozen |
| `Beta(4,1)` | 0.800 | `tide-beta41fresh-k16-top2-soft075-dir512-jointmix` | 9.35@180000 | 10.08 | 0.4823 | mix+joint uniform |
| `Beta(2,2)` | 0.500 | `tide-beta22-enddense08-k16-top2-soft075-dir512-jointmix` | 10.12@200000 | 13.19 | 0.4627 | mix+joint end_dense p=0.8 |

## Matrix 2: Source/Sampling x Router Update

Ô là `FID128 best@step / last`. `mix` nghĩa là GMM/router data mode mix `x1/x0`; `x1` nghĩa là fit/train router trên `x1` thuần.

### Beta(3.5,1.3)

| source/sample | router frozen | router joint |
|---|---:|---:|
| `x1` | 8.92@220k / last 8.92 | 8.54@180k / last 9.52 |
| `mix` | 8.06@220k / last 8.06 | 8.51@180k / last 9.00 |

### Beta(3,1.4)

| source/sample | router frozen | router joint |
|---|---:|---:|
| `x1` | 7.93@220k / last 7.93 / 8.05@220k / last 8.05 | |
| `mix` | | 8.48@200k / last 8.48 |

### Beta(2,2)

| source/sample | router frozen | router joint |
|---|---:|---:|
| `mix` | | 10.93@200k / last 10.93 / 10.12@200k / last 10.12 |

## Matrix 3: Same/Related Distribution, Different Eval ODE Timestep Split

| train t distribution | source/router | ODE timestep split | FID128 best | FID32 best | valid loss | straight128/curv128 |
|---|---|---|---:|---:|---:|---:|
| `Beta(3.5,1.3)` | `mix+joint` | `uniform` | 8.51 | 9.36 | 0.4722 | 1.107/0.0189 |
| `Beta(3.5,1.3)` | `mix+joint` | `end_dense p=0.7` | 9.12 | 11.88 | 0.4780 | 1.107/0.0192 |
| `Beta(3,1.4)` | `mix+joint` | `uniform` | 8.48 | 9.49 | 0.4641 | 1.104/0.0189 |
| `Beta(3,1.4)` | `mix+joint` | `end_dense p=0.7` | 9.01 | 11.97 | 0.4698 | 1.111/0.0195 |
| `Beta(2,2)` | `mix+joint` | `end_dense p=0.7` | 10.93 | 15.94 | 0.4603 | 1.120/0.0203 |
| `Beta(2,2)` | `mix+joint` | `end_dense p=0.8` | 10.12 | 13.19 | 0.4627 | 1.116/0.0202 |

## Matrix 4: Comparable Metric Table

| config | run | FID128 best | best step | FID128 last | FID32 best | valid loss | straight128 | curv128 | ODE p | pred/target var |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline old` | tide-k16-top2-softv0p75-s128-dir512 | 6.97 | 350000 | 6.97 | 9.29 | 0.4741 | 1.109 | 0.0208 |  | 0.653 |
| `Beta(3,1) jointmix resume` | tide-resume150-beta31-k16-top2-soft075-dir512-jointmix | 7.33 | 360000 | 7.33 | 8.42 | 0.4678 | 1.097 | 0.0190 |  | 0.657 |
| `Beta(3,1.4) x1+frozen best` | tide-beta31p4-x1frozen-k16-top2-soft075-dir512-r2 | 7.93 | 220000 | 7.93 | 9.44 | 0.4573 | 1.102 | 0.0189 | 1.00 | 0.661 |
| `Beta(3,1.4) mix+joint uniform` | tide-beta31p4-jointmix-k16-top2-soft075-dir512-r2 | 8.48 | 200000 | 8.48 | 9.49 | 0.4641 | 1.104 | 0.0189 | 1.00 | 0.654 |
| `Beta(3,1.4) mix+joint enddense07` | tide-beta31p4-enddense07-k16-top2-soft075-dir512-jointmix | 9.01 | 180000 | 9.33 | 11.97 | 0.4698 | 1.111 | 0.0195 | 0.70 | 0.656 |
| `Beta(3.5,1.3) mix+frozen` | tide-beta35-13-mixfrozen-k16-top2-soft075-dir512 | 8.06 | 220000 | 8.06 | 9.59 | 0.4643 | 1.099 | 0.0186 | 1.00 | 0.655 |
| `Beta(3.5,1.3) mix+joint uniform` | tide-beta35-13-k16-top2-soft075-dir512-jointmix | 8.51 | 180000 | 9.00 | 9.36 | 0.4722 | 1.107 | 0.0189 |  | 0.648 |
| `Beta(3.5,1.3) mix+joint enddense07` | tide-beta35-13-enddense07-k16-top2-soft075-dir512-jointmix | 9.12 | 180000 | 9.72 | 11.88 | 0.4780 | 1.107 | 0.0192 | 0.70 | 0.651 |
| `Beta(2,2) mix+joint enddense07` | tide-beta22-enddense07-k16-top2-soft075-dir512-jointmix | 10.93 | 200000 | 10.93 | 15.94 | 0.4603 | 1.120 | 0.0203 | 0.70 | 0.657 |
| `Beta(2,2) mix+joint enddense08` | tide-beta22-enddense08-k16-top2-soft075-dir512-jointmix | 10.12 | 200000 | 10.12 | 13.19 | 0.4627 | 1.116 | 0.0202 | 0.80 | 0.654 |
| `Beta(4,1) mix+joint` | tide-beta41fresh-k16-top2-soft075-dir512-jointmix | 9.35 | 180000 | 11.27 | 10.08 | 0.4823 | 1.104 | 0.0181 |  | 0.642 |

## Delta Notes

- `Beta(3,1.4)+mix+joint+end_dense p=0.7` kém uniform cùng source/router: `0.53` FID128.
- `Beta(3.5,1.3)+mix+joint+end_dense p=0.7` kém uniform cùng source/router: `0.61` FID128.
- `Beta(2,2)+end_dense p=0.8` tốt hơn p=0.7 khoảng `0.81` FID128, nhưng vẫn tệ hơn các beta lệch phải.
- Kết luận hiện tại: end-dense không có tín hiệu tốt trong các ô đã thử; nếu muốn cải thiện rollout, nên thử hướng train schedule hoặc solver khác hơn là end_dense p=0.7/0.8 hiện tại.
