# Autoresearch Candidate Grid

- Objective: `fid128_best` (lower is better)
- Candidates: 6

## Seeds

| rank | run | fid128 | fid32 | fid4 | step |
|---:|---|---:|---:|---:|---:|
| 1 | tide-k16-top2-softv0p75-s128-dir512 | 6.969 | 9.295 | 68.762 | 365600 |
| 2 | tide-resume5-cli-g136-k32-top24-none-hard05 | 7.062 | 9.314 | 68.402 | 640000 |
| 3 | tide-resume-r1-k16-top2-soft075-dir512-mixcont10 | 7.091 | 9.253 | 66.769 | 620000 |
| 4 | tide-k32-top2-softv0p75-s128-dir001 | 7.118 | 9.380 | 69.272 | 367400 |

## Candidates

| job | seed | strategy | run | seed objective | changes |
|---:|---:|---|---|---:|---|
| 1 | 1 | beta-alpha-up | ar-20260529-r1-beta-alpha-up-k16-top2-b3p5-1 | 6.969 | `model_t_beta_alpha=3.5, model_t_beta_beta=1.0, model_t_sampling=beta` |
| 2 | 1 | beta-alpha-down | ar-20260529-r1-beta-alpha-down-k16-top2-b2p5-1 | 6.969 | `model_t_beta_alpha=2.5, model_t_beta_beta=1.0, model_t_sampling=beta` |
| 3 | 1 | beta-beta-up | ar-20260529-r1-beta-beta-up-k16-top2-b3-1p3 | 6.969 | `model_t_beta_alpha=3.0, model_t_beta_beta=1.3, model_t_sampling=beta` |
| 4 | 1 | topk-up | ar-20260529-r1-topk-up-k16-top4 | 6.969 | `gmm_router_topk=4` |
| 5 | 1 | enddense-p06 | ar-20260529-r1-enddense-p06-k16-top2 | 6.969 | `model_eval_ode_power=0.6, model_eval_ode_schedule=end_dense` |
| 6 | 1 | enddense-p08 | ar-20260529-r1-enddense-p08-k16-top2 | 6.969 | `model_eval_ode_power=0.8, model_eval_ode_schedule=end_dense` |
