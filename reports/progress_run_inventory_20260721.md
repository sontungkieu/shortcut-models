# Progress Run Inventory 2026-07-21

- Submission events: **62**
- Distinct kernel slugs: **56**
- Distinct run-name labels: **53**
- Non-dry-run submit reports scanned: **25**
- Scope: submissions from 2026-06-22 through 2026-07-20.
- A retry that reuses a slug is one additional submission event but not a new distinct slug.

## Evidence Status

| Status | Submission events | Latest distinct slugs |
|---|---:|---:|
| `complete_metrics` | 34 | 34 |
| `error_cross_account` | 3 | 3 |
| `error_missing_checkpoint_in_relay` | 6 | 6 |
| `error_no_model_metrics` | 1 | 1 |
| `error_then_retried` | 4 | 4 |
| `partial_metrics_timeout` | 8 | 8 |
| `superseded_resume_error` | 6 | 0 |

## Reused Slugs

These six slugs were pushed twice. The first resume attempt failed and the later retry produced complete metrics.

| Kernel | Attempts | Evidence history | Submit reports |
|---|---:|---|---|
| `johnntlhudson/tide-factorial-wbridge-gmmseed0-resume400-k16-to` | 2 | `superseded_resume_error` -> `complete_metrics` | `gmm_tide_factorial_seed01_resume400_crossaccount_submit_20260719.json`<br>`gmm_tide_factorial_seed01_resume400_retry2_submit_20260719.json` |
| `kieuhongquan/tide-factorial-samplemix-gmmseed0-resume400-k16` | 2 | `superseded_resume_error` -> `complete_metrics` | `gmm_tide_factorial_seed01_resume400_crossaccount_submit_20260719.json`<br>`gmm_tide_factorial_seed01_resume400_retry2_submit_20260719.json` |
| `nguyncmnhda/tide-factorial-wbridge-gmmseed1-resume400-k16-to` | 2 | `superseded_resume_error` -> `complete_metrics` | `gmm_tide_factorial_seed01_resume400_crossaccount_submit_20260719.json`<br>`gmm_tide_factorial_seed01_resume400_retry2_submit_20260719.json` |
| `no1ceboy/tide-factorial-samplemix-gmmseed1-resume400-k16` | 2 | `superseded_resume_error` -> `complete_metrics` | `gmm_tide_factorial_seed01_resume400_no1ceboy_submit_20260719.json`<br>`gmm_tide_factorial_seed01_resume400_retry2_submit_20260719.json` |
| `bangchi/tide-factorial-wbridge-gmmseed2-resume400-k16-to` | 2 | `superseded_resume_error` -> `complete_metrics` | `gmm_tide_factorial_seed2_resume400_submit_20260719.json`<br>`gmm_tide_factorial_seed2_resume400_retry2_submit_20260719.json` |
| `ctlcmleon/tide-factorial-samplemix-gmmseed2-resume400-k16` | 2 | `superseded_resume_error` -> `complete_metrics` | `gmm_tide_factorial_seed2_resume400_submit_20260719.json`<br>`gmm_tide_factorial_seed2_resume400_retry2_submit_20260719.json` |

## Latest State By Slug

| # | Date | Family | Evidence | Attempts | Kernel |
|---:|---|---|---|---:|---|
| 1 | 20260622 | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `anhhaphan/tide-ars-c0-control-k16-top2-soft075-dir512-anhh` |
| 2 | 20260622 | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `bangchi/tide-ars-c1-bridge11-k16-top2-soft075-dir512-ban` |
| 3 | 20260622 | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `casihoavinh/tide-ars-c2-d4drop02-k16-top2-soft075-dir512-cas` |
| 4 | 20260622 | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `codemaivanngu/tide-ars-c3-bridge11-d4drop02-k16-top2-soft075-d` |
| 5 | 20260622 | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `ctlcmleon/tide-ars-c4-bridge11-sampletopk-k16-top2-soft075` |
| 6 | 20260622 | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `damtrunghieu/tide-ars-c5-bridge11-pcalw5-k16-top2-soft075-dir` |
| 7 | 20260623 | `c0_c4_clean_confirm` | `complete_metrics` | 1 | `anhhaphan/tide-ars-confirm-c0-control-200k-k16-top2-soft07` |
| 8 | 20260623 | `c0_c4_clean_confirm` | `complete_metrics` | 1 | `bangchi/tide-ars-confirm-c4-bridge11-sampletopk-200k-k16` |
| 9 | 20260629 | `output_quota_smoke` | `complete_metrics` | 1 | `anhhaphan/tide-output-quota-smoke-k4-top2-sampletopk-anhha` |
| 10 | 20260630 | `c4_long240` | `error_no_model_metrics` | 1 | `no1ceboy/tide-ars-c4-long240-bridge11-sampletopk-k16-top2` |
| 11 | 20260701 | `c0_c4_200k_repeats` | `complete_metrics` | 1 | `anhhaphan/tide-ars-var-c0-repeat-s0-200k-k16-top2-soft075` |
| 12 | 20260701 | `c0_c4_200k_repeats` | `complete_metrics` | 1 | `bangchi/tide-ars-var-c0-repeat-s1-200k-k16-top2-soft075` |
| 13 | 20260701 | `c0_c4_200k_repeats` | `complete_metrics` | 1 | `casihoavinh/tide-ars-var-c4-repeat-s0-200k-k16-top2-soft075` |
| 14 | 20260701 | `c0_c4_200k_repeats` | `complete_metrics` | 1 | `codemaivanngu/tide-ars-var-c4-repeat-s1-200k-k16-top2-soft075` |
| 15 | 20260707 | `c0_c4_resume` | `partial_metrics_timeout` | 1 | `bangchi/tide-ars-var-c0-s1-resume420-k16-top2-soft075-ba` |
| 16 | 20260707 | `c0_c4_resume` | `partial_metrics_timeout` | 1 | `codemaivanngu/tide-ars-var-c4-s1-resume420-k16-top2-soft075-co` |
| 17 | 20260710 | `c0_c4_resume` | `complete_metrics` | 1 | `anhhaphan/tide-ars-var-c0-s0-resume400-k16-top2-soft075-an` |
| 18 | 20260710 | `c0_c4_resume` | `complete_metrics` | 1 | `casihoavinh/tide-ars-var-c4-s0-resume400-k16-top2-soft075-ca` |
| 19 | 20260710 | `c0_c4_repeated_fid` | `complete_metrics` | 1 | `bangchi/tide-fidrep-c0-s1-400k-bangchi-20260710-1120` |
| 20 | 20260710 | `c0_c4_repeated_fid` | `error_cross_account` | 1 | `ctlcmleon/tide-fidrep-c0-s0-400k-ctlcmleon-20260710-1119` |
| 21 | 20260710 | `c0_c4_repeated_fid` | `error_cross_account` | 1 | `damtrunghieu/tide-fidrep-c4-s0-400k-damtrunghieu-20260710-111` |
| 22 | 20260710 | `c0_c4_repeated_fid` | `error_cross_account` | 1 | `hoanganpham123/tide-fidrep-c4-s1-400k-hoanganpham123-20260710-1` |
| 23 | 20260713 | `c0_c4_repeated_fid` | `complete_metrics` | 1 | `anhhaphan/tide-fidrep-c0-s0-400k-anhhaphan-20260713-1059` |
| 24 | 20260713 | `c0_c4_repeated_fid` | `complete_metrics` | 1 | `casihoavinh/tide-fidrep-c4-s0-400k-casihoavinh-20260713-1059` |
| 25 | 20260713 | `c0_c4_resume` | `complete_metrics` | 1 | `codemaivanngu/tide-ars-var-c4-s1-recover400-k16-top2-soft075-c` |
| 26 | 20260714 | `c0_c4_repeated_fid` | `complete_metrics` | 1 | `codemaivanngu/tide-fidrep-c4-s1-400k-codemaivanngu-20260714-03` |
| 27 | 20260715 | `gmm_seed2_training` | `complete_metrics` | 1 | `bangchi/tide-ars-confirm-c0-gmmseed2-200k-k16-top2-soft0` |
| 28 | 20260715 | `gmm_seed2_training` | `complete_metrics` | 1 | `ctlcmleon/tide-ars-confirm-c4-gmmseed2-200k-k16-top2-soft0` |
| 29 | 20260716 | `gmm_seed2_training` | `complete_metrics` | 1 | `bangchi/tide-ars-confirm-c0-gmmseed2-resume400-k16-top2` |
| 30 | 20260716 | `gmm_seed2_training` | `complete_metrics` | 1 | `ctlcmleon/tide-ars-confirm-c4-gmmseed2-resume400-k16-top2` |
| 31 | 20260716 | `gmm_seed2_repeated_fid` | `complete_metrics` | 1 | `bangchi/tide-fidconfirm-c0-gmmseed2-400k-bangchi-2026071` |
| 32 | 20260716 | `gmm_seed2_repeated_fid` | `complete_metrics` | 1 | `ctlcmleon/tide-fidconfirm-c4-gmmseed2-400k-ctlcmleon-20260` |
| 33 | 20260719 | `factorial_training` | `complete_metrics` | 1 | `bangchi/tide-factorial-wbridge-gmmseed2-200k-k16-top2-so` |
| 34 | 20260719 | `factorial_training` | `complete_metrics` | 1 | `ctlcmleon/tide-factorial-samplemix-gmmseed2-200k-k16-top2` |
| 35 | 20260719 | `factorial_training` | `complete_metrics` | 1 | `codemaivanngu/tide-factorial-wbridge-gmmseed0-200k-k16-top2-so` |
| 36 | 20260719 | `factorial_training` | `complete_metrics` | 1 | `damtrunghieu/tide-factorial-samplemix-gmmseed0-200k-k16-top2` |
| 37 | 20260719 | `factorial_training` | `complete_metrics` | 1 | `hoanganpham123/tide-factorial-wbridge-gmmseed1-200k-k16-top2-so` |
| 38 | 20260719 | `factorial_training` | `complete_metrics` | 1 | `iamlonely/tide-factorial-samplemix-gmmseed1-200k-k16-top2` |
| 39 | 20260719 | `router_geometry_audit` | `error_then_retried` | 1 | `phamdotuandng/tide-audit-c0-gmmseed2-router-geometry-phamdotua` |
| 40 | 20260719 | `router_geometry_audit` | `error_then_retried` | 1 | `veilwings/tide-audit-c4-gmmseed2-router-geometry-veilwings` |
| 41 | 20260719 | `router_geometry_audit` | `error_then_retried` | 1 | `phamdotuandng/tide-audit-c0-gmmseed2-router-geometry-cli223-re` |
| 42 | 20260719 | `router_geometry_audit` | `error_then_retried` | 1 | `veilwings/tide-audit-c4-gmmseed2-router-geometry-cli223-re` |
| 43 | 20260719 | `router_geometry_audit` | `complete_metrics` | 1 | `phamdotuandng/tide-audit-c0-gmmseed2-router-geometry-path-retr` |
| 44 | 20260719 | `router_geometry_audit` | `complete_metrics` | 1 | `veilwings/tide-audit-c4-gmmseed2-router-geometry-path-retr` |
| 45 | 20260719 | `factorial_training` | `complete_metrics` | 2 | `johnntlhudson/tide-factorial-wbridge-gmmseed0-resume400-k16-to` |
| 46 | 20260719 | `factorial_training` | `complete_metrics` | 2 | `kieuhongquan/tide-factorial-samplemix-gmmseed0-resume400-k16` |
| 47 | 20260719 | `factorial_training` | `complete_metrics` | 2 | `nguyncmnhda/tide-factorial-wbridge-gmmseed1-resume400-k16-to` |
| 48 | 20260719 | `factorial_training` | `complete_metrics` | 2 | `no1ceboy/tide-factorial-samplemix-gmmseed1-resume400-k16` |
| 49 | 20260719 | `factorial_training` | `complete_metrics` | 2 | `bangchi/tide-factorial-wbridge-gmmseed2-resume400-k16-to` |
| 50 | 20260719 | `factorial_training` | `complete_metrics` | 2 | `ctlcmleon/tide-factorial-samplemix-gmmseed2-resume400-k16` |
| 51 | 20260720 | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `bangchi/tide-fidrep-factorial-wb-gmmseed2-400k-bangchi-2` |
| 52 | 20260720 | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `ctlcmleon/tide-fidrep-factorial-sm-gmmseed2-400k-ctlcmleon` |
| 53 | 20260720 | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `johnntlhudson/tide-fidrep-factorial-wb-gmmseed0-400k-johnntlhu` |
| 54 | 20260720 | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `kieuhongquan/tide-fidrep-factorial-sm-gmmseed0-400k-kieuhongq` |
| 55 | 20260720 | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `nguyncmnhda/tide-fidrep-factorial-wb-gmmseed1-400k-nguyncmnh` |
| 56 | 20260720 | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `no1ceboy/tide-fidrep-factorial-sm-gmmseed1-400k-no1ceboy` |

## Submission Events

| # | Timestamp | Family | Evidence | Attempt | Kernel | Source report |
|---:|---|---|---|---:|---|---|
| 1 | `2026-06-22T08:25:38.402462+00:00` | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `anhhaphan/tide-ars-c0-control-k16-top2-soft075-dir512-anhh` | `gmm_tide_phase2_aware_c0_c5_submit_20260622.json` |
| 2 | `2026-06-22T08:25:38.402462+00:00` | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `bangchi/tide-ars-c1-bridge11-k16-top2-soft075-dir512-ban` | `gmm_tide_phase2_aware_c0_c5_submit_20260622.json` |
| 3 | `2026-06-22T08:25:38.402462+00:00` | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `casihoavinh/tide-ars-c2-d4drop02-k16-top2-soft075-dir512-cas` | `gmm_tide_phase2_aware_c0_c5_submit_20260622.json` |
| 4 | `2026-06-22T08:25:38.402462+00:00` | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `codemaivanngu/tide-ars-c3-bridge11-d4drop02-k16-top2-soft075-d` | `gmm_tide_phase2_aware_c0_c5_submit_20260622.json` |
| 5 | `2026-06-22T08:25:38.402462+00:00` | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `ctlcmleon/tide-ars-c4-bridge11-sampletopk-k16-top2-soft075` | `gmm_tide_phase2_aware_c0_c5_submit_20260622.json` |
| 6 | `2026-06-22T08:25:38.402462+00:00` | `phase2_c0_c5_exploration` | `partial_metrics_timeout` | 1 | `damtrunghieu/tide-ars-c5-bridge11-pcalw5-k16-top2-soft075-dir` | `gmm_tide_phase2_aware_c0_c5_submit_20260622.json` |
| 7 | `2026-06-22T20:09:09.450560+00:00` | `c0_c4_clean_confirm` | `complete_metrics` | 1 | `anhhaphan/tide-ars-confirm-c0-control-200k-k16-top2-soft07` | `gmm_tide_phase2_aware_confirm_c0_c4_200k_submit_20260623.json` |
| 8 | `2026-06-22T20:09:09.450560+00:00` | `c0_c4_clean_confirm` | `complete_metrics` | 1 | `bangchi/tide-ars-confirm-c4-bridge11-sampletopk-200k-k16` | `gmm_tide_phase2_aware_confirm_c0_c4_200k_submit_20260623.json` |
| 9 | `2026-06-29T16:59:04.256229+00:00` | `output_quota_smoke` | `complete_metrics` | 1 | `anhhaphan/tide-output-quota-smoke-k4-top2-sampletopk-anhha` | `gmm_tide_output_quota_smoke_submit_20260629.json` |
| 10 | `2026-06-30T16:54:59.550313+00:00` | `c4_long240` | `error_no_model_metrics` | 1 | `no1ceboy/tide-ars-c4-long240-bridge11-sampletopk-k16-top2` | `gmm_tide_phase2_c4_long240_submit_20260630.json` |
| 11 | `2026-07-01T08:39:47.175647+00:00` | `c0_c4_200k_repeats` | `complete_metrics` | 1 | `anhhaphan/tide-ars-var-c0-repeat-s0-200k-k16-top2-soft075` | `gmm_tide_phase2_variance_repeats4_submit_20260701.json` |
| 12 | `2026-07-01T08:39:47.175647+00:00` | `c0_c4_200k_repeats` | `complete_metrics` | 1 | `bangchi/tide-ars-var-c0-repeat-s1-200k-k16-top2-soft075` | `gmm_tide_phase2_variance_repeats4_submit_20260701.json` |
| 13 | `2026-07-01T08:39:47.175647+00:00` | `c0_c4_200k_repeats` | `complete_metrics` | 1 | `casihoavinh/tide-ars-var-c4-repeat-s0-200k-k16-top2-soft075` | `gmm_tide_phase2_variance_repeats4_submit_20260701.json` |
| 14 | `2026-07-01T08:39:47.175647+00:00` | `c0_c4_200k_repeats` | `complete_metrics` | 1 | `codemaivanngu/tide-ars-var-c4-repeat-s1-200k-k16-top2-soft075` | `gmm_tide_phase2_variance_repeats4_submit_20260701.json` |
| 15 | `2026-07-07T12:23:59.226191+00:00` | `c0_c4_resume` | `partial_metrics_timeout` | 1 | `bangchi/tide-ars-var-c0-s1-resume420-k16-top2-soft075-ba` | `gmm_tide_phase2_variance_resume2_420k_submit_20260707.json` |
| 16 | `2026-07-07T12:23:59.226191+00:00` | `c0_c4_resume` | `partial_metrics_timeout` | 1 | `codemaivanngu/tide-ars-var-c4-s1-resume420-k16-top2-soft075-co` | `gmm_tide_phase2_variance_resume2_420k_submit_20260707.json` |
| 17 | `2026-07-09T17:42:05.314059+00:00` | `c0_c4_resume` | `complete_metrics` | 1 | `anhhaphan/tide-ars-var-c0-s0-resume400-k16-top2-soft075-an` | `gmm_tide_phase2_variance_seed0_resume400_submit_20260710.json` |
| 18 | `2026-07-09T17:42:05.314059+00:00` | `c0_c4_resume` | `complete_metrics` | 1 | `casihoavinh/tide-ars-var-c4-s0-resume400-k16-top2-soft075-ca` | `gmm_tide_phase2_variance_seed0_resume400_submit_20260710.json` |
| 19 | `2026-07-10T11:19:52.523950+00:00` | `c0_c4_repeated_fid` | `error_cross_account` | 1 | `ctlcmleon/tide-fidrep-c0-s0-400k-ctlcmleon-20260710-1119` | `gmm_tide_fid_repeat4_submit_20260710.json` |
| 20 | `2026-07-10T11:19:52.523950+00:00` | `c0_c4_repeated_fid` | `error_cross_account` | 1 | `damtrunghieu/tide-fidrep-c4-s0-400k-damtrunghieu-20260710-111` | `gmm_tide_fid_repeat4_submit_20260710.json` |
| 21 | `2026-07-10T11:19:52.523950+00:00` | `c0_c4_repeated_fid` | `complete_metrics` | 1 | `bangchi/tide-fidrep-c0-s1-400k-bangchi-20260710-1120` | `gmm_tide_fid_repeat4_submit_20260710.json` |
| 22 | `2026-07-10T11:19:52.523950+00:00` | `c0_c4_repeated_fid` | `error_cross_account` | 1 | `hoanganpham123/tide-fidrep-c4-s1-400k-hoanganpham123-20260710-1` | `gmm_tide_fid_repeat4_submit_20260710.json` |
| 23 | `2026-07-13T10:59:52.829139+00:00` | `c0_c4_repeated_fid` | `complete_metrics` | 1 | `anhhaphan/tide-fidrep-c0-s0-400k-anhhaphan-20260713-1059` | `gmm_tide_fid_repeat_same_owner_retry2_submit_20260713.json` |
| 24 | `2026-07-13T10:59:52.829139+00:00` | `c0_c4_repeated_fid` | `complete_metrics` | 1 | `casihoavinh/tide-fidrep-c4-s0-400k-casihoavinh-20260713-1059` | `gmm_tide_fid_repeat_same_owner_retry2_submit_20260713.json` |
| 25 | `2026-07-13T11:03:16.449596+00:00` | `c0_c4_resume` | `complete_metrics` | 1 | `codemaivanngu/tide-ars-var-c4-s1-recover400-k16-top2-soft075-c` | `gmm_tide_c4_seed1_recover400_submit_20260713.json` |
| 26 | `2026-07-14T03:00:00.463428+00:00` | `c0_c4_repeated_fid` | `complete_metrics` | 1 | `codemaivanngu/tide-fidrep-c4-s1-400k-codemaivanngu-20260714-03` | `gmm_tide_fid_repeat_c4_s1_recovered400_submit_20260714.json` |
| 27 | `2026-07-15T07:39:02.932574+00:00` | `gmm_seed2_training` | `complete_metrics` | 1 | `bangchi/tide-ars-confirm-c0-gmmseed2-200k-k16-top2-soft0` | `gmm_tide_confirm_gmmseed2_submit_20260715.json` |
| 28 | `2026-07-15T07:39:02.932574+00:00` | `gmm_seed2_training` | `complete_metrics` | 1 | `ctlcmleon/tide-ars-confirm-c4-gmmseed2-200k-k16-top2-soft0` | `gmm_tide_confirm_gmmseed2_submit_20260715.json` |
| 29 | `2026-07-16T03:13:40.902872+00:00` | `gmm_seed2_training` | `complete_metrics` | 1 | `bangchi/tide-ars-confirm-c0-gmmseed2-resume400-k16-top2` | `gmm_tide_confirm_gmmseed2_resume400_submit_20260716.json` |
| 30 | `2026-07-16T03:13:40.902872+00:00` | `gmm_seed2_training` | `complete_metrics` | 1 | `ctlcmleon/tide-ars-confirm-c4-gmmseed2-resume400-k16-top2` | `gmm_tide_confirm_gmmseed2_resume400_submit_20260716.json` |
| 31 | `2026-07-16T16:15:35.113058+00:00` | `gmm_seed2_repeated_fid` | `complete_metrics` | 1 | `bangchi/tide-fidconfirm-c0-gmmseed2-400k-bangchi-2026071` | `gmm_tide_fidconfirm_gmmseed2_submit_20260716.json` |
| 32 | `2026-07-16T16:15:35.113058+00:00` | `gmm_seed2_repeated_fid` | `complete_metrics` | 1 | `ctlcmleon/tide-fidconfirm-c4-gmmseed2-400k-ctlcmleon-20260` | `gmm_tide_fidconfirm_gmmseed2_submit_20260716.json` |
| 33 | `2026-07-18T17:24:16.022075+00:00` | `factorial_training` | `complete_metrics` | 1 | `bangchi/tide-factorial-wbridge-gmmseed2-200k-k16-top2-so` | `gmm_tide_factorial_seed2_submit_20260719.json` |
| 34 | `2026-07-18T17:24:16.022075+00:00` | `factorial_training` | `complete_metrics` | 1 | `ctlcmleon/tide-factorial-samplemix-gmmseed2-200k-k16-top2` | `gmm_tide_factorial_seed2_submit_20260719.json` |
| 35 | `2026-07-18T18:51:35.721387+00:00` | `factorial_training` | `complete_metrics` | 1 | `codemaivanngu/tide-factorial-wbridge-gmmseed0-200k-k16-top2-so` | `gmm_tide_factorial_seed01_200k_submit_20260719.json` |
| 36 | `2026-07-18T18:51:35.721387+00:00` | `factorial_training` | `complete_metrics` | 1 | `damtrunghieu/tide-factorial-samplemix-gmmseed0-200k-k16-top2` | `gmm_tide_factorial_seed01_200k_submit_20260719.json` |
| 37 | `2026-07-18T18:51:35.721387+00:00` | `factorial_training` | `complete_metrics` | 1 | `hoanganpham123/tide-factorial-wbridge-gmmseed1-200k-k16-top2-so` | `gmm_tide_factorial_seed01_200k_submit_20260719.json` |
| 38 | `2026-07-18T18:51:35.721387+00:00` | `factorial_training` | `complete_metrics` | 1 | `iamlonely/tide-factorial-samplemix-gmmseed1-200k-k16-top2` | `gmm_tide_factorial_seed01_200k_submit_20260719.json` |
| 39 | `2026-07-18T18:53:07.764746+00:00` | `router_geometry_audit` | `error_then_retried` | 1 | `phamdotuandng/tide-audit-c0-gmmseed2-router-geometry-phamdotua` | `gmm_tide_router_geometry_audit_submit_20260719.json` |
| 40 | `2026-07-18T18:53:07.764746+00:00` | `router_geometry_audit` | `error_then_retried` | 1 | `veilwings/tide-audit-c4-gmmseed2-router-geometry-veilwings` | `gmm_tide_router_geometry_audit_submit_20260719.json` |
| 41 | `2026-07-18T19:09:35.582954+00:00` | `router_geometry_audit` | `error_then_retried` | 1 | `phamdotuandng/tide-audit-c0-gmmseed2-router-geometry-cli223-re` | `gmm_tide_router_geometry_audit_retry_cli223_submit_20260719.json` |
| 42 | `2026-07-18T19:09:35.582954+00:00` | `router_geometry_audit` | `error_then_retried` | 1 | `veilwings/tide-audit-c4-gmmseed2-router-geometry-cli223-re` | `gmm_tide_router_geometry_audit_retry_cli223_submit_20260719.json` |
| 43 | `2026-07-18T19:15:39.751207+00:00` | `router_geometry_audit` | `complete_metrics` | 1 | `phamdotuandng/tide-audit-c0-gmmseed2-router-geometry-path-retr` | `gmm_tide_router_geometry_audit_retry_path_submit_20260719.json` |
| 44 | `2026-07-18T19:15:39.751207+00:00` | `router_geometry_audit` | `complete_metrics` | 1 | `veilwings/tide-audit-c4-gmmseed2-router-geometry-path-retr` | `gmm_tide_router_geometry_audit_retry_path_submit_20260719.json` |
| 45 | `2026-07-19T07:52:18.474351+00:00` | `factorial_training` | `superseded_resume_error` | 1 | `johnntlhudson/tide-factorial-wbridge-gmmseed0-resume400-k16-to` | `gmm_tide_factorial_seed01_resume400_crossaccount_submit_20260719.json` |
| 46 | `2026-07-19T07:52:18.474351+00:00` | `factorial_training` | `superseded_resume_error` | 1 | `kieuhongquan/tide-factorial-samplemix-gmmseed0-resume400-k16` | `gmm_tide_factorial_seed01_resume400_crossaccount_submit_20260719.json` |
| 47 | `2026-07-19T07:52:18.474351+00:00` | `factorial_training` | `superseded_resume_error` | 1 | `nguyncmnhda/tide-factorial-wbridge-gmmseed1-resume400-k16-to` | `gmm_tide_factorial_seed01_resume400_crossaccount_submit_20260719.json` |
| 48 | `2026-07-19T07:53:19.559970+00:00` | `factorial_training` | `superseded_resume_error` | 1 | `no1ceboy/tide-factorial-samplemix-gmmseed1-resume400-k16` | `gmm_tide_factorial_seed01_resume400_no1ceboy_submit_20260719.json` |
| 49 | `2026-07-19T10:56:57.590475+00:00` | `factorial_training` | `superseded_resume_error` | 1 | `bangchi/tide-factorial-wbridge-gmmseed2-resume400-k16-to` | `gmm_tide_factorial_seed2_resume400_submit_20260719.json` |
| 50 | `2026-07-19T10:56:57.590475+00:00` | `factorial_training` | `superseded_resume_error` | 1 | `ctlcmleon/tide-factorial-samplemix-gmmseed2-resume400-k16` | `gmm_tide_factorial_seed2_resume400_submit_20260719.json` |
| 51 | `2026-07-19T11:06:54.289682+00:00` | `factorial_training` | `complete_metrics` | 2 | `johnntlhudson/tide-factorial-wbridge-gmmseed0-resume400-k16-to` | `gmm_tide_factorial_seed01_resume400_retry2_submit_20260719.json` |
| 52 | `2026-07-19T11:06:54.289682+00:00` | `factorial_training` | `complete_metrics` | 2 | `kieuhongquan/tide-factorial-samplemix-gmmseed0-resume400-k16` | `gmm_tide_factorial_seed01_resume400_retry2_submit_20260719.json` |
| 53 | `2026-07-19T11:06:54.289682+00:00` | `factorial_training` | `complete_metrics` | 2 | `nguyncmnhda/tide-factorial-wbridge-gmmseed1-resume400-k16-to` | `gmm_tide_factorial_seed01_resume400_retry2_submit_20260719.json` |
| 54 | `2026-07-19T11:06:54.289682+00:00` | `factorial_training` | `complete_metrics` | 2 | `no1ceboy/tide-factorial-samplemix-gmmseed1-resume400-k16` | `gmm_tide_factorial_seed01_resume400_retry2_submit_20260719.json` |
| 55 | `2026-07-19T11:07:50.446042+00:00` | `factorial_training` | `complete_metrics` | 2 | `bangchi/tide-factorial-wbridge-gmmseed2-resume400-k16-to` | `gmm_tide_factorial_seed2_resume400_retry2_submit_20260719.json` |
| 56 | `2026-07-19T11:07:50.446042+00:00` | `factorial_training` | `complete_metrics` | 2 | `ctlcmleon/tide-factorial-samplemix-gmmseed2-resume400-k16` | `gmm_tide_factorial_seed2_resume400_retry2_submit_20260719.json` |
| 57 | `2026-07-20T14:19:32.415613+00:00` | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `johnntlhudson/tide-fidrep-factorial-wb-gmmseed0-400k-johnntlhu` | `gmm_tide_factorial_wb_sm_fidrepeat6_submit_20260720.json` |
| 58 | `2026-07-20T14:19:32.415613+00:00` | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `kieuhongquan/tide-fidrep-factorial-sm-gmmseed0-400k-kieuhongq` | `gmm_tide_factorial_wb_sm_fidrepeat6_submit_20260720.json` |
| 59 | `2026-07-20T14:19:32.415613+00:00` | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `nguyncmnhda/tide-fidrep-factorial-wb-gmmseed1-400k-nguyncmnh` | `gmm_tide_factorial_wb_sm_fidrepeat6_submit_20260720.json` |
| 60 | `2026-07-20T14:19:32.415613+00:00` | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `no1ceboy/tide-fidrep-factorial-sm-gmmseed1-400k-no1ceboy` | `gmm_tide_factorial_wb_sm_fidrepeat6_submit_20260720.json` |
| 61 | `2026-07-20T14:19:32.415613+00:00` | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `bangchi/tide-fidrep-factorial-wb-gmmseed2-400k-bangchi-2` | `gmm_tide_factorial_wb_sm_fidrepeat6_submit_20260720.json` |
| 62 | `2026-07-20T14:19:32.415613+00:00` | `factorial_repeated_fid` | `error_missing_checkpoint_in_relay` | 1 | `ctlcmleon/tide-fidrep-factorial-sm-gmmseed2-400k-ctlcmleon` | `gmm_tide_factorial_wb_sm_fidrepeat6_submit_20260720.json` |
