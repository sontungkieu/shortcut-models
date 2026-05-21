# Ablations

This file records the ablation plan and current experiment inventory for the
GMM, GMM-FM, and GMM-TIDE/FM branches. The goal is to make it possible to
reproduce which knobs were changed, why each batch exists, and where to find the
local reports.

Last updated: 2026-05-21.

## Common Setup

- Dataset: `celebahq256` from Kaggle dataset `codemaivanngu/shortcut-celebahq256`.
- Main data space: StableVAE latent space.
- Default GMM fit:
  - `gmm_fit_samples=32768`
  - `gmm_valid_samples=4096`
  - `gmm_em_iters=25`
  - `gmm_em_restarts=1`
  - `gmm_em_chunk_size=128`
  - `gmm_kmeanspp_init=1`
- Default raw/standardized behavior:
  - Main runs use `gmm_standardize_data=0`, so GMM is fit directly in latent space.
  - Standardization ablations explicitly use `gmm_standardize_data=1`; when used for FM, samples must be unstandardized back before feeding FM unless the run is intentionally testing standardized-space training.
- Default FM eval:
  - `eval_fid_timesteps=1,4,32,128`
  - Main later TIDE/FM runs use `train_eval_interval=40000`.
- GMM diagnostics:
  - train/valid NLL
  - `pi` KL/MSE/entropy against uniform
  - cluster count min/max/gap/ratio
  - dead components
  - data variance and per-component variance stats
  - center-distance and overlap proxy
  - variance floor hit rate
- TIDE/FM diagnostics:
  - FID at timesteps 1/4/32/128
  - flow straightness metrics
  - `x0/x1/v_target` magnitude and variance
  - FM residual decomposition
  - router distillation train/valid metrics
  - router usage/collapse metrics
  - JSONL and CSV mirrors for metrics.

## GMM-Only Mesh

Primary config: [configs/gmm_ablation_grid.json](configs/gmm_ablation_grid.json)

Queue/result reports:

- [reports/gmm_ablation_queue_20260507.json](reports/gmm_ablation_queue_20260507.json)
- [reports/gmm_ablation_results_20260508.json](reports/gmm_ablation_results_20260508.json)
- [reports/gmm_ablation_summary_20260508.json](reports/gmm_ablation_summary_20260508.json)

Grid size: `4 K values * 5 prior settings * 9 coverage settings = 180` GMM-only runs.
All 180 completed successfully in the reconciled queue.

### GMM Axes

| Axis | Values |
|---|---|
| Number of modes | `4`, `8`, `16`, `32` |
| Prior type/strength | `none:0`, `dirichlet:0.01`, `dirichlet:512`, `kl:512`, `kl:2048` |
| Fit samples | `32768` |
| Valid samples | `4096` |
| EM iterations | `25` |
| Standardization | `0` in the main 180-run mesh |

### Coverage / Variance Pressure

| Coverage name | Hard floor `gmm_min_var_data_frac` | Soft prior | Target variance | Strength |
|---|---:|---|---:|---:|
| `ml-no-coverage` | `0.0` | `none` | `1.0` | `0` |
| `hardv0p5` | `0.5` | `none` | `1.0` | `0` |
| `hardv1p0` | `1.0` | `none` | `1.0` | `0` |
| `hardv1p5` | `1.5` | `none` | `1.0` | `0` |
| `hardv2p0` | `2.0` | `none` | `1.0` | `0` |
| `soft-v1p0-s512` | `0.0` | `kl` | `1.0` | `512` |
| `soft-v1p0-s2048` | `0.0` | `kl` | `1.0` | `2048` |
| `soft-v1p5-s512` | `0.0` | `kl` | `1.5` | `512` |
| `soft-v1p5-s2048` | `0.0` | `kl` | `1.5` | `2048` |

Hard floor and soft variance prior were intentionally separated in this mesh.
Hard floor changes the M-step variance by clipping to a minimum variance.
Soft variance prior adds a KL-style penalty toward a target component variance.

### GMM-Only Observations

- Dataset latent variance mean in these logs is consistently around `0.668`.
- Ranking only by NLL favors flexible/no-coverage K32 models, but this can reward collapsed or highly imbalanced components.
- Top raw-NLL candidates included K32 `kl` prior with no coverage, but count ratios were often very large.
- For FM source quality, NLL must be read together with:
  - dead component count
  - count ratio/gap
  - `pi_entropy_normalized`
  - component variance mean/min
  - floor hit rate
  - overlap proxy.
- Later FM/TIDE selections used a mixture of NLL, variance coverage, dead component avoidance, and practical FID.

## Early GMM-FM Runs On Branch `gmm`

These runs tested the direct GMM-FM source before moving to the TIDE/router
formulation.

Key reports:

- [reports/gmm_fm_ranked_top15_20260508.json](reports/gmm_fm_ranked_top15_20260508.json)
- [reports/gmm_fm_hard_gt1_top5_submit_20260508.json](reports/gmm_fm_hard_gt1_top5_submit_20260508.json)
- [reports/gmm_fm_variance_gt1_top14_20260508.json](reports/gmm_fm_variance_gt1_top14_20260508.json)
- [reports/gmm_fm_standardize_top4_submit_20260511.json](reports/gmm_fm_standardize_top4_submit_20260511.json)
- [reports/gmm_standardize_top4_queue_20260511.json](reports/gmm_standardize_top4_queue_20260511.json)

### GMM-FM Selection Batches

| Batch | Purpose | Selection |
|---|---|---|
| Ranked top15 | Pick GMM candidates from the 180 GMM-only mesh for FM testing | mostly valid NLL, with later caution about collapse |
| Hard `var > 1` top5 | Test stronger minimum coverage candidates | selected hard-floor runs with no dead components |
| Variance `> 1` top14 | Broader coverage-oriented candidates | candidates where component coverage was larger than baseline |
| Standardize top4 | Test fitting GMM in standardized latent space | top four previously useful GMM configs with `gmm_standardize_data=1` |

### Standardize Top4 Candidates

| Run | Source grid | K | Prior / coverage | Standardize |
|---|---:|---:|---|---:|
| `fm-gmm-std-top4-g162-k32-kl512-ml` | `162` | `32` | `kl:512`, no coverage | `1` |
| `fm-gmm-std-top4-g136-k32-none-hardv0p5` | `136` | `32` | `none`, hard floor `0.5` | `1` |
| `fm-gmm-std-top4-g108-k16-dir512-ml` | `108` | `16` | `dirichlet:512`, no coverage | `1` |
| `fm-gmm-std-top4-g109-k16-dir512-hardv0p5` | `109` | `16` | `dirichlet:512`, hard floor `0.5` | `1` |

## GMM-TIDE/FM Baseline Grid

Primary config: [configs/gmm_tide_fm_grid.json](configs/gmm_tide_fm_grid.json)

Purpose: replace direct GMM assignment in FM with a learned router/distillation
network that predicts GMM responsibilities from the source state, then uses
top-k mixture components to construct the FM source.

Default router:

- `router_target_type=soft_kl`
- `router_train_data_mode=mix`
- `router_mix_x1_prob=0.5`
- `router_max_steps=5000`
- `router_valid_batches=16`
- `router_save_best=true`

### Baseline Jobs

| Run | K | top-k | Source GMM | Prior | Variance handling |
|---|---:|---:|---|---|---|
| `tide-k16-top2-g108-dir512-ml` | 16 | 2 | grid `108` | `dirichlet:512` | no coverage |
| `tide-k16-top4-g108-dir512-ml` | 16 | 4 | grid `108` | `dirichlet:512` | no coverage |
| `tide-k16-top2-g117-kl512-ml` | 16 | 2 | grid `117` | `kl:512` | no coverage |
| `tide-k16-top2-g109-dir512-hardv0p5` | 16 | 2 | grid `109` | `dirichlet:512` | hard floor `0.5` |
| `tide-k16-top2-softv0p75-s128-dir512` | 16 | 2 | newly fit | `dirichlet:512` | soft target variance `0.75`, strength `128` |
| `tide-k32-top2-g136-none-hardv0p5` | 32 | 2 | grid `136` | `none` | hard floor `0.5` |
| `tide-k32-top4-g136-none-hardv0p5` | 32 | 4 | grid `136` | `none` | hard floor `0.5` |
| `tide-k32-top2-g145-dir001-hardv0p5` | 32 | 2 | grid `145` | `dirichlet:0.01` | hard floor `0.5` |
| `tide-k32-top2-g146-dir001-hardv1p0` | 32 | 2 | grid `146` | `dirichlet:0.01` | hard floor `1.0` |
| `tide-k32-top2-softv0p75-s128-dir001` | 32 | 2 | newly fit | `dirichlet:0.01` | soft target variance `0.75`, strength `128` |

Notable earlier metric reports:

- [reports/gmm_tide_all_downloaded_metrics_20260515.json](reports/gmm_tide_all_downloaded_metrics_20260515.json)
- [reports/gmm_tide_kaggle_metric_insights_20260515.json](reports/gmm_tide_kaggle_metric_insights_20260515.json)
- [reports/gmm_tide_distill_gmm_report_20260515.json](reports/gmm_tide_distill_gmm_report_20260515.json)

Observed strong base runs included:

- `tide-k16-top2-softv0p75-s128-dir512`: FID128 around `6.97` at 350k in downloaded metrics.
- `tide-k32-top2-softv0p75-s128-dir001`: FID128 around `7.12` at 350k.
- `tide-k16-top2-g108`: FID128 around `7.22` at 350k.

## Next10 TIDE/FM Mesh

Primary config: [configs/gmm_tide_fm_next10_grid.json](configs/gmm_tide_fm_next10_grid.json)

Purpose: refine around the best soft-variance TIDE/FM configs, especially K16/K32
with top-k 2 and soft variance target near `0.75`.

Axes:

- target variance: `0.65`, `0.75`, `0.85`
- variance prior strength: `64`, `128`, `256`
- K: `16`, `32`
- prior: `dirichlet:512`, `dirichlet:0.01`, and one `kl:512` alternative
- router top-k: mostly `2`, one `top4`
- router temperature: base `1.0`, selected `1.5` probes

Representative runs:

| Run | K | top-k | Prior | Soft variance |
|---|---:|---:|---|---|
| `tide-next-k16-top2-softv0p65-s128-dir512` | 16 | 2 | `dirichlet:512` | target `0.65`, strength `128` |
| `tide-next-k16-top2-softv0p75-s64-dir512` | 16 | 2 | `dirichlet:512` | target `0.75`, strength `64` |
| `tide-next-k16-top2-softv0p75-s256-dir512` | 16 | 2 | `dirichlet:512` | target `0.75`, strength `256` |
| `tide-next-k16-top2-softv0p85-s128-dir512` | 16 | 2 | `dirichlet:512` | target `0.85`, strength `128` |
| `tide-next-k16-top4-softv0p75-s128-dir512-t1p5` | 16 | 4 | `dirichlet:512` | target `0.75`, strength `128`, temp `1.5` |
| `tide-next-k32-top2-softv0p75-s128-dir512` | 32 | 2 | `dirichlet:512` | target `0.75`, strength `128` |

## Mix / Continue EM Mesh

Primary config: [configs/gmm_tide_fm_mix_continue12_grid.json](configs/gmm_tide_fm_mix_continue12_grid.json)

Reports:

- [reports/gmm_tide_mix_continue12_report_20260513.json](reports/gmm_tide_mix_continue12_report_20260513.json)
- [reports/gmm_tide_mix_continue12_results_20260513.json](reports/gmm_tide_mix_continue12_results_20260513.json)

Purpose: test whether GMM quality improves when the GMM is fit on mixed
`x1/x0` samples or continued for extra EM iterations before FM. In this mesh,
`continue` means extra GMM EM iterations only; it does not mean router joint
training during FM.

Base families:

| Base | K | top-k | Prior / variance |
|---|---:|---:|---|
| r1 | 16 | 2 | soft target variance `0.75`, strength `128`, `dirichlet:512` |
| r2 | 32 | 2 | soft target variance `0.75`, strength `128`, `dirichlet:0.01` |
| r3 | 32 | 2 | hard floor `0.5`, no `pi` prior |
| r4 | 32 | 2 | hard floor `0.5`, `dirichlet:0.01` |

Variants per base:

| Variant | `gmm_fit_data_mode` | `gmm_mix_x1_prob` | `gmm_continue_em_iters` | Meaning |
|---|---|---:|---:|---|
| `mix` | `mix` | `0.5` | `0` | fit GMM on a 50/50 style x1/x0 mixture |
| `mixcont10` | `mix` | `0.5` | `10` | mixed fit, then 10 extra EM iterations |
| `x1cont10` | `x1` | default | `10` | x1-only fit, then 10 extra EM iterations |

Notable observation:

- `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` was a strong candidate with FID128 around `7.31` at 320k in the mix/continue report.
- Several resume attempts later showed that continuing beyond the earlier stopping point can degrade best-to-last FID, so best checkpoint selection matters.

## Joint Router Update Mesh

Primary config: [configs/gmm_tide_fm_joint9_grid.json](configs/gmm_tide_fm_joint9_grid.json)

Reports:

- [reports/joint9_comparison_20260520.json](reports/joint9_comparison_20260520.json)
- [reports/kaggle_cancelled_step_check_20260520.json](reports/kaggle_cancelled_step_check_20260520.json)

Purpose: test whether the distilled router should keep learning during FM.

Factor view:

- `M=0, J=0`: x1-only GMM, frozen router (`frozen`)
- `M=0, J=1`: x1-only GMM, router jointly updated by FM-time losses (`jointx1`)
- `M=1, J=1`: mixed GMM, router jointly updated (`jointmix`)
- The missing cell `M=1, J=0` was later covered by the `mix_only3` mesh.

The three base families were:

| Rank | Base | K | Prior / variance |
|---|---|---:|---|
| r1 | `k16-top2-soft075-dir512` | 16 | soft target variance `0.75`, strength `128`, `dirichlet:512` |
| r2 | `k32-top2-soft075-dir001` | 32 | soft target variance `0.75`, strength `128`, `dirichlet:0.01` |
| r3 | `k32-top2-none-hard05` | 32 | hard floor `0.5`, no prior |

Summary from the comparison report:

- r1: `jointx1` slightly beat frozen at best FID128 (`7.54` vs `7.58`), while `jointmix` was worse (`7.64`).
- r2: frozen beat joint variants (`7.75` vs about `7.86`/`8.08`).
- r3: frozen beat joint variants (`7.81` vs about `8.07`/`8.11`).
- `router_grad_norm_joint` was extremely small in several joint runs, so the joint update path may be too weak or dominated by the current loss scaling.

## Mix-Only Mesh

Primary config: [configs/gmm_tide_fm_mix_only3_grid.json](configs/gmm_tide_fm_mix_only3_grid.json)

Purpose: complete the missing `M=1, J=0` cell from the joint-factor view.
These use mixed GMM fitting but keep the router frozen during FM.

| Run | K | Prior / variance | Current status in latest shared context |
|---|---:|---|---|
| `tide-mixonly-r1-k16-top2-soft075-dir512` | 16 | soft `0.75`, strength `128`, `dirichlet:512` | `CANCEL_ACKNOWLEDGED` after timeout |
| `tide-mixonly-r2-k32-top2-soft075-dir001` | 32 | soft `0.75`, strength `128`, `dirichlet:0.01` | `CANCEL_ACKNOWLEDGED` after timeout |
| `tide-mixonly-r3-k32-top2-none-hard05` | 32 | hard floor `0.5`, no prior | `CANCEL_ACKNOWLEDGED` after timeout |

Timeout here did not mean training failure. Diagnostics were downloaded on
2026-05-21 and all three reached the 320k eval point.

## Top-K Baseline Mesh

Primary config: [configs/gmm_tide_fm_topk_baselines_grid.json](configs/gmm_tide_fm_topk_baselines_grid.json)

Reports:

- [reports/gmm_tide_fm_topk_baselines_submit_20260520.json](reports/gmm_tide_fm_topk_baselines_submit_20260520.json)
- [reports/gmm_tide_fm_topk_baselines_remaining4_submit_20260520.json](reports/gmm_tide_fm_topk_baselines_remaining4_submit_20260520.json)
- [reports/gmm_tide_fm_topk_baselines_idle2_submit_20260520.json](reports/gmm_tide_fm_topk_baselines_idle2_submit_20260520.json)
- [reports/kaggle_cancelled_step_check_20260520.json](reports/kaggle_cancelled_step_check_20260520.json)

Purpose: test larger router top-k values after observing that source replacement
may require different FM settings.

| Family | Source | K | top-k values |
|---|---|---:|---|
| g136 hard floor | `gmm-k32-floorv0p5-none-s0p0-raw-hardv0p5` | 32 | `8`, `12`, `16`, `24` |
| g108 no coverage | `gmm-k16-floorv0p0-dirichlet-s512p0-raw-ml-no-coverage` | 16 | `8`, `12` |

These runs reached roughly 330k-344k training steps before timeout/cancel:

- g136 top8: about `333200`
- g136 top12: about `333300`
- g136 top16: about `336900`
- g136 top24: about `330000`
- g108 top8: about `343700`
- g108 top12: about `343000`

Resume attempts were opened for:

- g136 K32 top24
- g108 K16 top12

Cross-account resume failed because Kaggle runtime returns `403 Forbidden` for
private kernel output access even when a source-owner `kaggle.json` is injected.
A same-owner probe succeeded. The later resume5 same-owner jobs ran to Kaggle
timeout and their diagnostics are parsed below.

## FM Retune 8 Mesh

Primary config: [configs/gmm_tide_fm_fmretune8_grid.json](configs/gmm_tide_fm_fmretune8_grid.json)

Purpose: test the hypothesis that changing the source distribution requires
retuning FM optimizer settings. This isolates source/top-k choices from two
simple FM learning-rate schedules.

FM variants:

| FM variant | LR | Warmup | Cosine | beta1 | beta2 | Weight decay |
|---|---:|---:|---:|---:|---:|---:|
| F1 | `1e-4` | `20000` | `0` | `0.9` | `0.999` | `0.01` |
| F2 | `5e-5` | `20000` | `0` | `0.9` | `0.999` | `0.01` |

Submitted jobs:

| Run | Source | K | top-k | Router/GMM policy | FM variant | Latest status |
|---|---|---:|---:|---|---|---|
| `tide-fmretune-s1-k16-top2-soft075-dir512-f1-w20k` | soft K16 top2 | 16 | 2 | frozen, x1 GMM | F1 | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |
| `tide-fmretune-s1-k16-top2-soft075-dir512-f2-lr5e5-w20k` | soft K16 top2 | 16 | 2 | frozen, x1 GMM | F2 | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |
| `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f1-w20k` | soft K16 top2 | 16 | 2 | joint router, mix GMM | F1 | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |
| `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f2-lr5e5-w20k` | soft K16 top2 | 16 | 2 | joint router, mix GMM | F2 | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |
| `tide-fmretune-s3-k16-top8-soft075-dir512-f1-w20k` | soft K16 top8 | 16 | 8 | frozen | F1 | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |
| `tide-fmretune-s4-k16-top16-soft075-dir512-f1-w20k` | soft K16 top16 | 16 | 16 | frozen | F1 | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |
| `tide-fmretune-s5-g136-k32-top24-none-hard05-f1-w20k` | hard K32 top24 | 32 | 24 | frozen | F1 | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |
| `tide-fmretune-s6-g136-k32-top32-none-hard05-f1-w20k` | hard K32 top32 | 32 | 32 | frozen | F1 | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |

Latest status source:
[reports/kaggle_shared_context_status_20260521_latest.json](reports/kaggle_shared_context_status_20260521_latest.json)

## Resume / Credential Probes

Reports:

- [reports/gmm_tide_fm_topk_resume2_srcjson_resubmit_20260520.json](reports/gmm_tide_fm_topk_resume2_srcjson_resubmit_20260520.json)
- [reports/gmm_tide_fm_topk_resume2_kernelsrc_resubmit_20260520.json](reports/gmm_tide_fm_topk_resume2_kernelsrc_resubmit_20260520.json)
- [reports/gmm_tide_fm_topk_resume5_sameowner_cli_submit_20260520.json](reports/gmm_tide_fm_topk_resume5_sameowner_cli_submit_20260520.json)
- [reports/kaggle_probe_kernel_output_gpu_20260520_retry.json](reports/kaggle_probe_kernel_output_gpu_20260520_retry.json)

Credential probe conclusion:

- Same-owner notebook using same-owner credentials can call `kaggle kernels status` and `kaggle kernels output` for a private canceled source kernel.
- Cross-account notebook injecting the source owner's `kaggle.json` still receives `403 Forbidden` in Kaggle runtime.
- Therefore private-kernel resume should run under the source owner when using `kaggle kernels output`.

Current resume5 same-owner jobs:

| Run | Owner | Source kernel | Latest status |
|---|---|---|---|
| `tide-resume5-cli-g136-k32-top24-none-hard05` | `anhhaphan` | `anhhaphan/tide-topk-g136-k32-top24-none-hard05-anhhaphan-2` | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |
| `tide-resume5-cli-g108-k16-top12-dir512-ml` | `damtrunghieu` | `damtrunghieu/tide-topk-g108-k16-top12-dir512-ml-damtrunghieu` | `CANCEL_ACKNOWLEDGED`, diagnostics parsed |

## Latest Downloaded Results: 2026-05-21

Downloaded diagnostics:

- Output root: `outputs/kaggle/latest_13_20260521`
- Download manifest: [reports/latest_13_download_20260521.json](reports/latest_13_download_20260521.json)
- Parsed metrics: [reports/latest_13_results_20260521.json](reports/latest_13_results_20260521.json)
- Status source: [reports/kaggle_shared_context_status_20260521_latest.json](reports/kaggle_shared_context_status_20260521_latest.json)
- Download result: `13/13` kernels downloaded with Kaggle CLI return code `0`.

All 13 jobs ended as `CANCEL_ACKNOWLEDGED` because they exceeded the Kaggle
runtime limit. That status is expected for these long runs; the useful question
is how far they trained and which eval checkpoint was best.

### Main Result Table

| Group | Run | Owner | Train step | Last eval | Best FID128 | Last FID128 | Best FID32 | Last FID32 | x0/x1 | Target var | Residual var | Straight last |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fm-retune | `tide-fmretune-s1-k16-top2-soft075-dir512-f1-w20k` | `huynhtule` | 337000 | 320000 | 8.833 @ 320000 | 8.833 | 11.254 @ 320000 | 11.254 | 0.995 | 1.292 | 0.484 | 1.112 |
| fm-retune | `tide-fmretune-s1-k16-top2-soft075-dir512-f2-lr5e5-w20k` | `iamlonely` | 345100 | 320000 | 11.122 @ 320000 | 11.122 | 13.515 @ 320000 | 13.515 | 0.996 | 1.349 | 0.482 | 1.115 |
| fm-retune | `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f1-w20k` | `kieuhongquan` | 320000 | 280000 | 9.051 @ 280000 | 9.051 | 11.371 @ 280000 | 11.371 | 1.025 | 1.236 | 0.480 | 1.113 |
| fm-retune | `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f2-lr5e5-w20k` | `kiuvithong` | 320000 | 280000 | 11.333 @ 280000 | 11.333 | 13.595 @ 280000 | 13.595 | 1.029 | 1.256 | 0.488 | 1.116 |
| fm-retune | `tide-fmretune-s3-k16-top8-soft075-dir512-f1-w20k` | `manh1904` | 346200 | 320000 | 9.380 @ 320000 | 9.380 | 11.862 @ 320000 | 11.862 | 1.011 | 1.347 | 0.448 | 1.113 |
| fm-retune | `tide-fmretune-s4-k16-top16-soft075-dir512-f1-w20k` | `nguyncmnhda` | 340900 | 320000 | 8.903 @ 320000 | 8.903 | 11.463 @ 320000 | 11.463 | 1.004 | 1.291 | 0.469 | 1.113 |
| fm-retune | `tide-fmretune-s5-g136-k32-top24-none-hard05-f1-w20k` | `veilwings` | 333800 | 320000 | 9.251 @ 320000 | 9.251 | 11.771 @ 320000 | 11.771 | 1.023 | 1.290 | 0.459 | 1.110 |
| fm-retune | `tide-fmretune-s6-g136-k32-top32-none-hard05-f1-w20k` | `victorharvey27` | 337400 | 320000 | 9.389 @ 320000 | 9.389 | 11.933 @ 320000 | 11.933 | 1.001 | 1.320 | 0.465 | 1.111 |
| mix-only | `tide-mixonly-r1-k16-top2-soft075-dir512` | `casihoavinh` | 338400 | 320000 | 7.473 @ 320000 | 7.473 | 9.705 @ 320000 | 9.705 | 1.025 | 1.290 | 0.483 | 1.110 |
| mix-only | `tide-mixonly-r2-k32-top2-soft075-dir001` | `codemaivanngu` | 340000 | 320000 | 7.662 @ 320000 | 7.662 | 10.017 @ 280000 | 10.147 | 1.003 | 1.308 | 0.482 | 1.110 |
| mix-only | `tide-mixonly-r3-k32-top2-none-hard05` | `hoanganpham123` | 338600 | 320000 | 7.687 @ 320000 | 7.687 | 9.941 @ 320000 | 9.941 | 0.975 | 1.344 | 0.451 | 1.108 |
| resume5 | `tide-resume5-cli-g108-k16-top12-dir512-ml` | `damtrunghieu` | 640000 | 600000 | 7.214 @ 560000 | 7.503 | 9.270 @ 560000 | 9.821 | 1.000 | 1.302 | 0.454 | 1.109 |
| resume5 | `tide-resume5-cli-g136-k32-top24-none-hard05` | `anhhaphan` | 640000 | 600000 | 7.062 @ 360000 | 7.415 | 9.314 @ 360000 | 9.764 | 1.017 | 1.296 | 0.442 | 1.105 |

### FID128 Curves

| Run | FID128 by eval step |
|---|---|
| `tide-mixonly-r1-k16-top2-soft075-dir512` | 40k: 17.166, 80k: 11.443, 120k: 9.323, 160k: 8.547, 200k: 8.148, 240k: 7.693, 280k: 7.507, 320k: 7.473 |
| `tide-mixonly-r2-k32-top2-soft075-dir001` | 40k: 17.720, 80k: 12.584, 120k: 9.857, 160k: 9.032, 200k: 8.539, 240k: 7.879, 280k: 7.775, 320k: 7.662 |
| `tide-mixonly-r3-k32-top2-none-hard05` | 40k: 18.702, 80k: 12.786, 120k: 9.916, 160k: 9.080, 200k: 8.603, 240k: 8.109, 280k: 7.841, 320k: 7.687 |
| `tide-fmretune-s1-k16-top2-soft075-dir512-f1-w20k` | 40k: 25.673, 80k: 16.152, 120k: 12.745, 160k: 11.591, 200k: 10.800, 240k: 9.709, 280k: 9.097, 320k: 8.833 |
| `tide-fmretune-s1-k16-top2-soft075-dir512-f2-lr5e5-w20k` | 40k: 37.110, 80k: 21.680, 120k: 16.894, 160k: 14.769, 200k: 13.401, 240k: 12.299, 280k: 11.434, 320k: 11.122 |
| `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f1-w20k` | 40k: 28.124, 80k: 16.802, 120k: 12.932, 160k: 11.331, 200k: 10.385, 240k: 9.633, 280k: 9.051 |
| `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f2-lr5e5-w20k` | 40k: 39.381, 80k: 23.916, 120k: 18.531, 160k: 15.744, 200k: 13.809, 240k: 12.331, 280k: 11.333 |
| `tide-fmretune-s3-k16-top8-soft075-dir512-f1-w20k` | 40k: 25.665, 80k: 15.957, 120k: 13.046, 160k: 11.851, 200k: 10.873, 240k: 9.936, 280k: 9.835, 320k: 9.380 |
| `tide-fmretune-s4-k16-top16-soft075-dir512-f1-w20k` | 40k: 24.061, 80k: 14.333, 120k: 11.749, 160k: 10.706, 200k: 9.978, 240k: 9.296, 280k: 9.133, 320k: 8.903 |
| `tide-fmretune-s5-g136-k32-top24-none-hard05-f1-w20k` | 40k: 27.056, 80k: 15.768, 120k: 12.955, 160k: 11.260, 200k: 10.558, 240k: 9.963, 280k: 9.578, 320k: 9.251 |
| `tide-fmretune-s6-g136-k32-top32-none-hard05-f1-w20k` | 40k: 27.817, 80k: 17.099, 120k: 14.004, 160k: 12.162, 200k: 10.891, 240k: 10.480, 280k: 9.805, 320k: 9.389 |
| `tide-resume5-cli-g136-k32-top24-none-hard05` | 320k: 7.424, 360k: 7.062, 400k: 7.467, 440k: 7.375, 480k: 7.268, 520k: 7.330, 560k: 7.446, 600k: 7.415 |
| `tide-resume5-cli-g108-k16-top12-dir512-ml` | 320k: 7.500, 360k: 7.504, 400k: 7.326, 440k: 7.412, 480k: 7.280, 520k: 7.258, 560k: 7.214, 600k: 7.503 |

### Router Distillation Diagnostics

| Run | Router valid loss | Valid top1 | Top-k mass | Usage entropy | Overfit note |
|---|---:|---:|---:|---:|---|
| `tide-fmretune-s1-k16-top2-soft075-dir512-f1-w20k` | 0.1879 | 0.9141 | 0.9998 | 0.9419 | gap -0.0250, ratio 0.883, best@5000 |
| `tide-fmretune-s1-k16-top2-soft075-dir512-f2-lr5e5-w20k` | 0.2137 | 0.9102 | 0.9996 | 0.9306 | gap -0.0499, ratio 0.811, best@4000 |
| `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f1-w20k` | 0.1982 | 0.9180 | 1.0000 | 0.9250 | gap 0.1092, ratio 2.226, best@4000 |
| `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f2-lr5e5-w20k` | 0.1937 | 0.9199 | 1.0000 | 0.9219 | gap -0.0608, ratio 0.761, best@4500 |
| `tide-fmretune-s3-k16-top8-soft075-dir512-f1-w20k` | 0.2134 | 0.9131 | 1.0000 | 0.9465 | gap -0.0975, ratio 0.686, best@4000 |
| `tide-fmretune-s4-k16-top16-soft075-dir512-f1-w20k` | 0.1968 | 0.9248 | 1.0000 | 0.9469 | gap -0.0998, ratio 0.663, best@4500 |
| `tide-fmretune-s5-g136-k32-top24-none-hard05-f1-w20k` | 0.2343 | 0.9131 | 1.0000 | 0.9183 | gap 0.0448, ratio 1.236, best@5000 |
| `tide-fmretune-s6-g136-k32-top32-none-hard05-f1-w20k` | 0.2594 | 0.9004 | 1.0000 | 0.9059 | gap 0.0397, ratio 1.181, best@4500 |
| `tide-mixonly-r1-k16-top2-soft075-dir512` | 0.2094 | 0.9131 | 0.9996 | 0.9482 | gap 0.0790, ratio 1.605, best@4000 |
| `tide-mixonly-r2-k32-top2-soft075-dir001` | 0.2266 | 0.9131 | 0.9992 | 0.8584 | gap -0.0791, ratio 0.741, best@4500 |
| `tide-mixonly-r3-k32-top2-none-hard05` | 0.2601 | 0.8994 | 0.9997 | 0.8937 | gap -0.0133, ratio 0.951, best@4500 |

### GMM Diagnostics For New Non-Resume Runs

| Run | K | GMM prior | Var policy | Valid NLL | pi entropy | Comp var | Floor hit |
|---|---:|---|---|---:|---:|---:|---:|
| `tide-fmretune-s1-k16-top2-soft075-dir512-f1-w20k` | 16 | `dirichlet:512` | soft KL target `0.75`, strength `128` | 4261.2 | 0.9936 | 0.5273 | 0.0000 |
| `tide-fmretune-s1-k16-top2-soft075-dir512-f2-lr5e5-w20k` | 16 | `dirichlet:512` | soft KL target `0.75`, strength `128` | 4260.2 | 0.9931 | 0.5274 | 0.0000 |
| `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f1-w20k` | 16 | `dirichlet:512` | soft KL target `0.75`, strength `128` | 4273.5 | 0.9705 | 0.5438 | 0.0000 |
| `tide-fmretune-s2-k16-top2-soft075-dir512-jointmix-f2-lr5e5-w20k` | 16 | `dirichlet:512` | soft KL target `0.75`, strength `128` | 4272.3 | 0.9705 | 0.5471 | 0.0000 |
| `tide-fmretune-s3-k16-top8-soft075-dir512-f1-w20k` | 16 | `dirichlet:512` | soft KL target `0.75`, strength `128` | 4257.1 | 0.9948 | 0.5269 | 0.0000 |
| `tide-fmretune-s4-k16-top16-soft075-dir512-f1-w20k` | 16 | `dirichlet:512` | soft KL target `0.75`, strength `128` | 4261.1 | 0.9950 | 0.5257 | 0.0000 |
| `tide-fmretune-s5-g136-k32-top24-none-hard05-f1-w20k` | 32 | `none:0` | hard std-frac `0.707` | 4187.1 | 0.9938 | 0.5135 | 0.0990 |
| `tide-fmretune-s6-g136-k32-top32-none-hard05-f1-w20k` | 32 | `none:0` | hard std-frac `0.707` | 4190.8 | 0.9843 | 0.5156 | 0.1124 |
| `tide-mixonly-r1-k16-top2-soft075-dir512` | 16 | `dirichlet:512` | soft KL target `0.75`, strength `128` | 4259.5 | 0.9949 | 0.5297 | 0.0000 |
| `tide-mixonly-r2-k32-top2-soft075-dir001` | 32 | `dirichlet:0.01` | soft KL target `0.75`, strength `128` | 4221.4 | 0.9190 | 0.5701 | 0.0000 |
| `tide-mixonly-r3-k32-top2-none-hard05` | 32 | `none:0` | hard std-frac `0.707` | 4200.1 | 0.9554 | 0.5093 | 0.1474 |

### 2026-05-21 Interpretation

- The strongest new non-resume result is the missing factorial cell
  `M=1, J=0`: mixed GMM fit with frozen router. It improves all three matched
  bases against the earlier joint9 comparison:
  - r1: mix-only `7.473` vs frozen `7.58` vs jointmix `7.64`.
  - r2: mix-only `7.662` vs frozen `7.75` vs jointmix about `8.08`.
  - r3: mix-only `7.687` vs frozen `7.81` vs jointmix about `8.11`.
- For the current implementation, mixed GMM fitting helps more reliably than
  updating the router during FM. Joint updates are not yet buying improvement.
- The FM retune schedule tested here is bad for this setup. F1 (`1e-4`,
  20k warmup, no cosine) is consistently much worse than the earlier default
  schedule, and F2 (`5e-5`) is much slower/worse. The F2 runs also have roughly
  half the last update norm (`~0.12`) compared with F1 (`~0.24`), matching the
  slow FID curves.
- Larger top-k alone does not fix the source problem under the retuned FM
  schedule. In the K16 soft family, top16 (`8.903`) beats top8 (`9.380`) but
  still loses to top2 mix-only (`7.473`) and to earlier default-schedule runs.
- Same-owner resume now works: both resume5 notebooks loaded a 300k checkpoint,
  reused GMM/router files, deleted the temporary loaded checkpoint, and trained
  to 640k. The manifest `checkpoint_step_guess` field is not reliable because
  it is parsed from source names like `g136`; the actual train log says
  `Loaded model with step 300000`.
- Long training needs best-checkpoint selection. Resume g136 top24 reaches
  FID128 `7.062` at 360k but degrades to `7.415` by 600k. Resume g108 top12
  reaches `7.214` at 560k and then degrades to `7.503` by 600k.
- The FM variance diagnostics look stable rather than explosive:
  `x0/x1` is roughly `0.975-1.029`, target variance is roughly `1.24-1.35`,
  residual variance is roughly `0.44-0.49`, and straightness ratio stays close
  to `1.105-1.116`. The difference between good and bad runs is therefore more
  likely source quality plus optimization schedule than a visible variance
  blow-up.
- Router distillation is not obviously collapsing. Valid top-1 agreement is
  around `0.90-0.925`, top-k mass is essentially `1.0`, and usage entropy is
  usually high. K32 hard-floor runs have lower usage entropy (`~0.88-0.91`)
  than K16 soft runs (`~0.94-0.97`), which may be another sign that larger
  top-k/hard-floor sources are less clean.

### Recommended Next Moves From This Batch

- Keep the original/default FM schedule for near-term source ablations. The
  simple F1/F2 retunes were worse and should not be used as defaults.
- Promote `mix-only` as the next main axis: run mixed GMM with frozen router on
  the strongest soft and hard candidates, then compare against x1-only frozen at
  the same eval budget.
- If testing joint router updates again, increase or redesign the router update
  signal instead of treating the current joint setting as a negative result for
  the idea. Current joint runs do not show enough benefit.
- Add best-checkpoint preservation or explicit best-FID checkpoint selection
  before more resume jobs. Last checkpoint is not a good proxy in timeout-prone
  Kaggle runs.
- For top-k, test only after locking a working FM schedule. Current top8/16/24/32
  evidence is confounded by the bad retune schedule.

## Reading Results

FID ranking is useful but should not be the only selection criterion:

- NLL can be misleading when hard variance floors or soft variance priors are used.
- A good GMM source should also preserve cluster coverage and avoid dead modes.
- A lower flow path length is not automatically better if top-k mixing blurs source samples between modes.
- For router quality, inspect both train and valid distillation metrics:
  - valid loss/KL
  - train-valid gap
  - top-1 agreement to GMM assignment
  - top-k mass
  - usage entropy and assign max fraction
  - unique clusters used.
- For FM compatibility, inspect:
  - `x0/x1` magnitude ratio
  - FM target/prediction variance ratio
  - residual variance decomposition
  - flow straightness metrics
  - FID trend across eval checkpoints.

## Active Questions

- Whether larger top-k values help under the original/default FM schedule.
- Whether joint router updates need stronger gradients or better loss scaling.
- Whether mixed GMM fitting plus frozen router remains better across a broader
  source set.
- Whether standardized GMM fitting plus unstandardized FM source improves coverage without changing geometric angles too much.
- Whether best-checkpoint selection should replace last-checkpoint reporting for timeout-prone Kaggle runs.
