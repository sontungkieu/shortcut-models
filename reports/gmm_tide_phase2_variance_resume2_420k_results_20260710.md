# Phase2 Variance Resume2 420k Results

- Status checked: both kernels are `CANCEL_ACKNOWLEDGED`. Diagnostics were downloaded only; no checkpoint/model artifacts were pulled locally.
- Runtime accelerator: both runs reported TPU runtime match with `tpu_device_count=8`.
- Historical baseline used for quick comparison: FID128 `6.969`.
- Note: `train.py` treats `max_steps` as additional steps after resume when `reset_step_on_load=0`; these jobs loaded step 200000 and were terminated by Kaggle around 420k absolute step, before the intended additional-step loop finished.

| run | family | status | loaded step | max logged step | best FID128 | best step | last complete FID128 | last FID128 step | valid loss best | curvature last | var ratio last | router usage last | ckpt saves | tracebacks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C0 resume | weighted + mix | CANCEL_ACKNOWLEDGED | 200000 | 424800 | 7.088 | 380000 | 7.243 | 420000 | 0.432 | 0.02094 | 0.650 | 0.944 | 2 | no |
| C4 resume | sample_topk + bridge | CANCEL_ACKNOWLEDGED | 200000 | 420000 | 6.864 | 320000 | 6.966 | 400000 | 0.432 | 0.07332 | 0.654 | 0.951 | 2 | no |

## Interpretation

- C4 resume is the important result: best FID128 `6.864` at step `320000`, which is slightly better than the historical `6.969` baseline.
- C4 later drifted back: the last completed FID128 is `6.966` at `400000`; the 420k eval appears incomplete for FID128.
- C0 resume did not beat baseline: best FID128 `7.088` at step `380000`, last completed FID128 `7.243` at `420000`.
- No fatal traceback/OOM/disk error was found in downloaded stdout/stderr. The visible `ERROR` hints are pip resolver warnings and accelerator-probe lines, not train crashes.
- Both resumes successfully loaded checkpoint step `200000`, deleted the temporary loaded checkpoint, and wrote checkpoint twice, consistent with save events at around 300k and 400k.

## Local Paths

- `C0 resume` run dir: `outputs/kaggle_jobs/gmm_tide_fm/bangchi__tide-ars-var-c0-s1-resume420-k16-top2-soft075-ba`
- `C4 resume` run dir: `outputs/kaggle_jobs/gmm_tide_fm/codemaivanngu__tide-ars-var-c4-s1-resume420-k16-top2-soft075-co`
- Machine JSON: `reports/gmm_tide_phase2_variance_resume2_420k_results_20260710.json`
