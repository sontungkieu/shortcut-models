# Repeated FID Measurement Audit

## Material Passport

- Verification Status: `ANALYZED`
- Primary metric: `fid/timesteps/128`
- Grid: `configs/gmm_tide_fid_repeat4_grid.json`
- Protocol: `configs/gmm_tide_fid_repeat_analysis_protocol.json` (`gmm-tide-phase2-fid-repeat-v1`)
- Registration status: `frozen_before_local_result_retrieval_not_external_preregistration`
- Parsed metric files: 4
- Checkpoints downloaded locally: no

## Per-checkpoint Results

| family | training seed | repeats | mean FID128 | sample SD | 95% CI | run |
|---|---:|---:|---:|---:|---|---|
| C0 | 0 | 5 | 7.0749 | 0.0189 | 7.0514--7.0983 | `tide-fidrep-c0-s0-400k` |
| C4 | 0 | 5 | 7.0400 | 0.0249 | 7.0092--7.0709 | `tide-fidrep-c4-s0-400k` |
| C0 | 1 | 5 | 7.2660 | 0.0407 | 7.2155--7.3165 | `tide-fidrep-c0-s1-400k` |
| C4 | 1 | 5 | 6.9132 | 0.0426 | 6.8604--6.9661 | `tide-fidrep-c4-s1-400k` |

## Paired C4 minus C0

| training seed | paired eval seeds | C0 mean | C4 mean | delta mean | delta 95% CI |
|---:|---:|---:|---:|---:|---|
| 0 | 5 | 7.0749 | 7.0400 | -0.0348 | -0.0758--0.0062 |
| 1 | 5 | 7.2660 | 6.9132 | -0.3528 | -0.4317---0.2739 |

## Decision Gate

- Pooled within-checkpoint eval SD: `0.0333`.
- Practical threshold `max(0.1, 2 x eval-SD)`: `0.1000`.
- Overall paired eval delta C4-C0: `-0.1938`.
- All training seeds favor C4: `True`.
- Complete paired evaluation seeds: `True`.
- Measurement gate passed: `True`.
- Frozen outcome code: `gate_passed`.
- Frozen next action: `run_at_least_one_additional_independent_training_seed_at_matched_budget`.

The repeated generation seeds estimate evaluation noise only. A post-audit wiring
check found that the labels `training seed 0/1` changed only `gmm_init_seed` and
`gmm_mix_seed`; router and FM runtime RNG remained at default seed 0. Therefore,
the independent unit supported by this audit is the GMM initialization/mix seed,
not a full end-to-end training seed. The frozen next action is preserved as a
third matched GMM-seed replication and is registered separately in
`configs/gmm_tide_fid_confirmation_gmmseed2_protocol.json`.

## Comparability Audit

- Status: `PASS`.
- Errors: `0`.
- Warnings: `1`.
- Warning: Checkpoint/router/GMM hashes were not instrumented; audit verifies source references, config, and loaded step only.

## Mechanism Diagnostics

These metrics are descriptive diagnostics only and do not enter the frozen FID128 decision gate.
Available: `flow/curvature_proxy_mean, flow/straightness_ratio_mean`.
Unavailable from eval-only artifacts: `training/fm/pred_variance, training/fm/target_variance, training/router/assign_max_frac, training/router/usage_entropy_normalized, training/tide/topk_mu_angular_dispersion`.

| family | training seed | metric | repeats | mean | sample SD |
|---|---:|---|---:|---:|---:|
| C0 | 0 | `flow/curvature_proxy_mean` | 5 | 0.020989 | 0.000012 |
| C0 | 0 | `flow/straightness_ratio_mean` | 5 | 1.109217 | 0.000232 |
| C4 | 0 | `flow/curvature_proxy_mean` | 5 | 0.021035 | 0.000009 |
| C4 | 0 | `flow/straightness_ratio_mean` | 5 | 1.109390 | 0.000121 |
| C0 | 1 | `flow/curvature_proxy_mean` | 5 | 0.021055 | 0.000014 |
| C0 | 1 | `flow/straightness_ratio_mean` | 5 | 1.109669 | 0.000143 |
| C4 | 1 | `flow/curvature_proxy_mean` | 5 | 0.021093 | 0.000015 |
| C4 | 1 | `flow/straightness_ratio_mean` | 5 | 1.109385 | 0.000263 |

| training seed | metric | paired seeds | C0 mean | C4 mean | delta C4-C0 |
|---:|---|---:|---:|---:|---:|
| 0 | `flow/curvature_proxy_mean` | 5 | 0.020989 | 0.021035 | 0.000046 |
| 0 | `flow/straightness_ratio_mean` | 5 | 1.109217 | 1.109390 | 0.000173 |
| 1 | `flow/curvature_proxy_mean` | 5 | 0.021055 | 0.021093 | 0.000039 |
| 1 | `flow/straightness_ratio_mean` | 5 | 1.109669 | 1.109385 | -0.000285 |

## Fallacy Scan

- Simpson's paradox: checked across training seeds; report per-seed and aggregate directions separately.
- Ecological fallacy: not applicable; no individual-level inference is made.
- Berkson's paradox: not applicable to this paired checkpoint audit.
- Collider bias: not applicable; no adjusted causal model is fit.
- Base-rate neglect: not applicable to FID.
- Regression to the mean: caution; checkpoints were selected after observing earlier trajectories.
- Survivorship bias: caution if failed or missing jobs are omitted.
- Look-elsewhere effect: caution; prior checkpoint/config exploration was extensive.
- Garden of forking paths: controlled partially by the predeclared 400k checkpoint and FID128 endpoint.
- Correlation versus causation: no causal claim is made.
- Reverse causality: not applicable.
- Coverage: 11/11 checked.
