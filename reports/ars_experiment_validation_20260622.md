## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate + experiment planning handoff
- Origin Date: 2026-06-22
- Verification Status: ANALYZED
- Version Label: validation_v1
- Repo: `/home/tung/shortcut-models-1`
- Branch: `moe2`
- Remote Pull: not needed for this pass; local Kaggle/W&B exports were sufficient
- Checkpoints Downloaded: no

## Validation Report

- **Source**: local reports, local Kaggle diagnostics, and local W&B CSV/JSON exports listed in `reports/ars_experiment_validation_20260622.json`.
- **Overall Confidence**: CAUTION
- **Primary Metric**: FID128, lower is better.
- **Control Baseline**: `tide-k16-top2-softv0p75-s128-dir512`, FID128 = 6.969.

### Executive Finding

To date, the best MoE2/router ablations have approached but not beaten the historical GMM-TIDE baseline. The strongest new evidence is not that a particular router is best, but that the current phase-1 objective is misaligned with the phase-2 objective: router/GMM fit metrics improve locally, yet FID often does not improve. This makes direct grid search over router/GMM settings inefficient unless the selection loop uses phase-2-aware signals.

### Statistical Findings

| Finding | Evidence | Value | Confidence |
|---|---:|---:|---|
| Historical baseline remains best | baseline FID128 | 6.969 | SOLID |
| Best new weekly run is close but worse | bridge B(1,1) FID128 | 7.033 | SOLID |
| Best router capacity/reg run | router d4 dropout 0.2 FID128 | 7.094 | SOLID |
| EM100 does not justify automatic FM rerun | source rerun recommended | 0 / 180 | SOLID |
| EM100 average valid-NLL gain is tiny | mean relative gain | 0.0446% | SOLID |
| Toy FM loss is not rollout quality | toy source best mean SWD ratio | 1.354 | CAUTION |
| Whitening has metrics but poor FID | best whitening FID128 from W&B exports | 10.037 | SOLID |

### Best Observed Runs

| Rank | FID128 | Step | Batch | Label | Var Ratio | Curvature | Usage H |
|---:|---:|---:|---|---|---:|---:|---:|
| 1 | 6.969 | 365600 | baseline | `baseline K16 top2 soft0.75 dir512` | 0.674 | N/A | 0.948 |
| 2 | 7.033 | 350000 | 12/06 bridge/tide-KL | `router bridge lambda: bridge lambda B(1,1), FM t uniform/discrete` | 0.672 | 0.02093 | 0.884 |
| 3 | 7.094 | 300000 | router_deep4 | `d4 none dropout 0.2` | 0.651 | 0.02084 | 0.949 |
| 4 | 7.147 | 350000 | router_reg_capacity | `LayerNorm + dropout 0.2` | 0.669 | 0.02082 | 0.944 |
| 5 | 7.153 | 350000 | router_reg_capacity | `dropout 0.3` | 0.628 | 0.02083 | 0.950 |
| 6 | 7.164 | 300000 | router_deep4 | `d5 layer_norm dropout 0.2` | 0.661 | 0.02091 | 0.948 |
| 7 | 7.175 | 350000 | router_reg_capacity | `dropout 0.2` | 0.654 | 0.02092 | 0.949 |
| 8 | 7.222 | 350000 | router_temp_depth10 | `d5 none dropout 0.2 targetT 1.25` | 0.635 | 0.02080 | 0.950 |
| 9 | 7.245 | 300000 | router_temp_depth10 | `d5 none dropout 0.2 targetT 1.0` | 0.665 | 0.02078 | 0.942 |
| 10 | 7.255 | 300000 | router_reg_capacity | `low-cap plain` | 0.674 | 0.02072 | 0.940 |
| 11 | 7.272 | 280000 | 13/06 EMA resume | `EMA beta(3.5,1.3) resume` | N/A | 0.01874 | N/A |
| 12 | 7.277 | 300000 | router_smooth5 | `bridge Beta(2,2)` | 0.650 | 0.02088 | 0.939 |

### Historical Context Before Router-Regularization Week

| Rank | FID128 | Step | Run |
|---:|---:|---:|---|
| 1 | 6.969 | 350000 | `tide-k16-top2-softv0p75-s128-dir512` |
| 2 | 7.091 | 480000 | `tide-resume-r1-k16-top2-soft075-dir512-mixcont10` |
| 3 | 7.118 | 350000 | `tide-k32-top2-softv0p75-s128-dir001` |
| 4 | 7.131 | 480000 | `tide-resume-best-mix-r1-k16-top2-soft075-dir512` |
| 5 | 7.151 | 480000 | `tide-resume-best-x1cont-r3-k32-hard05` |
| 6 | 7.219 | 350000 | `tide-k16-top2-g108` |
| 7 | 7.256 | 350000 | `tide-k32-top2-g136-none-hardv0p5` |
| 8 | 7.315 | 320000 | `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` |
| 9 | 7.315 | 320000 | `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` |
| 10 | 7.315 | 320000 | `tide-mc-r1-k16-top2-soft075-dir512-mixcont10` |

### GMM / EM Validation

- EM100 versus EM25 joined 180/180 configs.
- Decision from configured criteria: **No automatic FM rerun recommendation from the configured EM100 criteria.**
- Profile-ok configs: 108 / 180.
- Still improving after EM25: 38 / 180.
- Mean relative valid-NLL improvement: 0.0446%; max 1.632%; min -1.030%.

Interpretation: EM100 mildly optimizes likelihood but did not pass the source-quality criteria for automatic FM rerun. This supports keeping compute on phase-2-aware source/router objectives rather than rerunning all FM sources just because GMM NLL improved.

### GMM Initialization Validation

Important data-quality note: `gmm_init_strategy` is unreliable in the init report because it records `kmeans++` for all rows. The table below infers the actual init from `run_name`.

| Init | n | Mean valid NLL | Best valid NLL | Dead Sum | Bad count-ratio > 50 | Mean count ratio |
|---|---:|---:|---:|---:|---:|---:|
| PCA + Lloyd5 | 4 | 4220.002 | 4180.718 | 0 | 0 | 2.632 |
| farthest + Lloyd5 | 4 | 4223.287 | 4179.364 | 2 | 1 | 50.012 |
| kmeans++ + Lloyd5 | 4 | 4222.964 | 4180.923 | 1 | 1 | 55.233 |
| kmeans++ r4 | 4 | 4218.343 | 4176.326 | 0 | 0 | 2.400 |
| split + Lloyd5 | 4 | 4221.562 | 4186.375 | 2 | 2 | 113.872 |

Interpretation: PCA + Lloyd and kmeans++ r4 are comparatively stable in this small CelebA latent init ablation. Split and farthest can create collapse/count-ratio failures in some source groups, so they should not be selected by NLL alone.

### Router / Source Validation

| Family | Best FID128 | Best Label | n | Mean FID128 |
|---|---:|---|---:|---:|
| deep_router | 7.094 | `d4 none dropout 0.2` | 4 | 7.222 |
| layernorm | 7.147 | `LayerNorm + dropout 0.2` | 4 | 7.319 |
| dropout | 7.153 | `dropout 0.3` | 3 | 7.250 |
| router_temp_depth10 | 7.222 | `d5 none dropout 0.2 targetT 1.25` | 10 | 7.381 |
| low capacity | 7.255 | `low-cap plain` | 2 | 7.302 |
| smooth_source_router | 7.277 | `bridge Beta(2,2)` | 5 | 7.403 |
| groupnorm | 7.325 | `GroupNorm + dropout 0.2` | 2 | 7.353 |

Interpretation: router depth/capacity regularization helps locally, with d4 dropout 0.2 strongest among regularizers, but even the best router-regularized run remains above the 6.969 baseline. Bigger/cleaner router fit is therefore not sufficient.

### Bridge and Tide-KL Validation

| Family | Config | FID128 | Step |
|---|---|---:|---:|
| router bridge lambda | `bridge lambda B(1,1), FM t uniform/discrete` | 7.033 | 350000 |
| router bridge lambda | `bridge lambda B(3,1.4), FM t uniform/discrete` | 7.404 | 300000 |
| router bridge lambda | `bridge lambda B(2.2,1.2), FM t uniform/discrete` | 7.612 | 300000 |
| q_GMM(x0_tide) KL | `tide-KL w=0.10, FM t Beta(3,1.4)` | 8.339 | 180000 |
| q_GMM(x0_tide) KL | `tide-KL w=0.05, FM t Beta(3,1.4)` | 8.366 | 180000 |
| q_GMM(x0_tide) KL | `tide-KL w=0.30, FM t Beta(3,1.4)` | 8.645 | 180000 |

Interpretation: bridge B(1,1) is the closest new result to the baseline. Tide-KL on `q_GMM(x0_tide)` worsens FID despite being conceptually aligned with geometry; this suggests over-constraining router/source can harm image quality.

### Toy Experiment Cross-Check

| Toy Group | Best/Lowest by rollout SWD | Mean SWD Ratio | Mean FM-MSE Ratio | Note |
|---|---|---:|---:|---|
| source | `nearest` | 1.354 | 0.691 | wins_swd=1, wins_mse=0 |
| source | `uniform` | 1.363 | 0.414 | wins_swd=0, wins_mse=5 |
| source | `direct` | 1.456 | 0.715 | wins_swd=1, wins_mse=0 |
| source | `oracle` | 1.646 | 0.627 | wins_swd=2, wins_mse=0 |
| source | `distilled` | 3.517 | 0.569 | wins_swd=12, wins_mse=11 |
| init | `kpp_r5` | 2.464 | 0.493 | wins_swd=3, wins_fm=2 |
| init | `split_lw8` | 2.605 | 0.496 | wins_swd=3, wins_fm=3 |
| init | `kpp_lw8` | 2.654 | 0.498 | wins_swd=3, wins_fm=1 |
| init | `quantilepca_lw8` | 2.697 | 0.497 | wins_swd=2, wins_fm=3 |
| init | `pca_lw8` | 2.712 | 0.506 | wins_swd=1, wins_fm=0 |

Interpretation: toy datasets show the same warning as CelebA: FM valid MSE can become much better while rollout SWD gets worse. This is direct evidence against using FM loss/router KL alone as a selection objective.

### Warnings

| Type | Detail | Affected |
|---|---|---|
| Surrogate mismatch | GMM NLL, router KL/top1, and FM valid MSE are weak surrogates for FID/rollout quality. | GMM grid, router distill, FM selection |
| Multiple comparisons | Many configs were tried across GMM, router, source, time sampling, init, toy settings without formal correction. | All ablation rankings |
| Selection bias | Most analysis uses completed/downloaded/logged runs; missing/timeout runs may be underrepresented. | Kaggle queue outputs |
| Non-independent trials | Many runs share base source, checkpoint, code revision, and data split. | Pairwise comparisons |
| Data-quality issue | Init report stores `gmm_init_strategy=kmeans++` for all rows; actual init inferred from names. | GMM init ablation |
| Early-stop comparability | Some 06/06 source/whitening/Gumbel runs are 120k--180k while later runs reach 300k--350k. | Cross-batch FID ranking |

### Fallacy Scan

- **Coverage**: 11/11 fallacy types checked.

| Fallacy | Severity | Detail |
|---|---|---|
| Simpson's Paradox | NOTE | No subgroup reversal test available; source family/K groups should be stratified before claims. |
| Ecological Fallacy | NOTE | Mostly run-level analysis; do not infer per-sample image behavior from aggregate FID alone. |
| Berkson's Paradox | CAUTION | Analysis is conditioned on runs with available logs/downloads; failed/missing runs may bias trends. |
| Collider Bias | CAUTION | Conditioning on completed Kaggle outputs and best-FID checkpoints can distort config-quality associations. |
| Base Rate Neglect | N/A | No diagnostic classification metrics such as sensitivity/specificity. |
| Regression to the Mean | CAUTION | Best-FID checkpoint selection across noisy evals can overstate true config quality without repeated seeds. |
| Survivorship Bias | CAUTION | Timeout/cancelled/error notebooks are not always fully represented in metric tables. |
| Look-Elsewhere Effect | CAUTION | Large config search; best run may be a search artifact unless confirmed by rerun/seed. |
| Garden of Forking Paths | CAUTION | Ablation direction evolved interactively; report should mark exploratory status. |
| Correlation != Causation | CAUTION | Correlations between diagnostics and FID do not establish causality; use targeted ablations. |
| Reverse Causality | NOTE | Less relevant, but good FID may co-occur with diagnostics rather than be caused by them. |

### Reproducibility

- **Method**: not rerun in this pass; validation is based on existing structured outputs.
- **Verdict**: CANNOT_VERIFY as a reproducibility rerun, ANALYZED as a statistics/results audit.
- **Remote Pull Decision**: no additional Kaggle/W&B pull was necessary for this pass because the main evidence tables and W&B exports are already local. Pulling remote would be useful only to fill missing failed-run logs or to update runs after 2026-06-13.

### Research/Experiment Handoff

The next experiment design should not search for a single “best router” by phase-1 scores. It should treat router/source construction as a policy/source proposal problem and select configs by phase-2-aware signals. A practical next-stage loop is:

1. Define a small candidate set from families that are interpretable: baseline GMM-TIDE, bridge B(1,1), d4 dropout 0.2, sample-topk, and one stable GMM init.
2. Run a resource-allocation schedule: 40k -> 120k -> 320k, selecting by early FID128 slope plus variance/geometry diagnostics, not by router KL alone.
3. Add a surrogate/ranker trained on this report JSON plus historical reports to predict final FID category from early metrics.
4. Only consider RL-lite/DDPO-like router updates after the action/reward interface is clear: sampled component/top-k action, logged probability, reward proxy, KL-to-GMM regularizer, and repeated-seed validation.

### Files Produced

- Machine-readable summary: `reports/ars_experiment_validation_20260622.json`
- This report: `reports/ars_experiment_validation_20260622.md`
