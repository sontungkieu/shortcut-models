# GMM-TIDE Output Quota Smoke Result 2026-06-30

- Kernel: `anhhaphan/tide-output-quota-smoke-k4-top2-sampletopk-anhha`
- Status: `KernelWorkerStatus.COMPLETE`
- Commit: `50350fc9099d607465318254235db47ebf5036ed`
- Local diagnostics: `outputs/kaggle/output_quota_smoke_20260630/`

## Result

The real Kaggle TPU smoke pipeline completed: GMM prep, router distillation, FM training, checkpoint save, eval, and final cleanup all ran.

Only diagnostics/log files were downloaded locally. Checkpoints, GMM stats, and router pickle were not downloaded.

## Output Quota Check

`output_cleanup_summary.json` reports:

- Total `/kaggle/working` output after cleanup: `1.0019 GiB`
- `ckpts`: `1.0732 GB`
- `gmm_tide_fm`: `2.48 MB`
- `__notebook__.ipynb`: `70 KB`

Cleanup removed:

- `/tmp/shortcut-models`
- `/tmp/shortcut_dataset`

The GMM latent cache was written under `/tmp`, confirmed by `gmm_prep_stdout.txt`.

## Diagnostics

- GMM smoke fit: `gmm_em_iters=2`, `fit_samples=512`, `valid_samples=128`
- GMM final train NLL: `4474.9694`
- GMM dead components: `0`
- Router best valid loss: `0.60297` at step `20`
- Train/eval summary step: `20`
- FID smoke value is not meaningful: `fid/timesteps/128 = 341.2968`
- Flow straightness smoke check: `flow/straightness_ratio_mean = 1.0000055`

## Error Check

No runtime traceback or quota/disk error was found in downloaded diagnostics. The only `ERROR` lines in the Kaggle notebook log are pip dependency resolver warnings during install, not fatal pipeline errors.
