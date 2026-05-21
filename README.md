## One-Step Diffusion via Shortcut Models 

Kevin Frans, Danijar Hafner, Sergey Levine, Pieter Abbeel

[Paper Link](https://arxiv.org/abs/2410.12557)
[Website Link](https://kvfrans.com/shortcut-models/)

### Abstract
Diffusion models and flow-matching models have enabled generating diverse and realistic images by learning to transfer noise to data.
However, sampling from these models involves iterative denoising over many neural network passes, making generation slow and expensive.
Previous approaches for speeding up sampling require complex training regimes, such as multiple training phases, multiple networks, or fragile scheduling.
We introduce shortcut models, a family of generative models that use a single network and training phase to produce high-quality samples in a single or multiple sampling steps.
Shortcut models condition the network not only on the current noise level but also on the desired step size, allowing the model to skip ahead in the generation process.
Across a wide range of sampling step budgets, shortcut models consistently produce higher quality samples than previous approaches, such as consistency models and reflow.
Compared to distillation, shortcut models reduce complexity to a single network and training phase and additionally allow varying step budgets at inference time.

![Showcase Figire](data/fig-showcase4.png)

### Overview

Shortcut models can utilize standard diffusion architectures (e.g. DiT), and condition on both `t` and `d`. At `d ≈ 0`, the shortcut objective is equivalent to the flow-matching objective, and can be trained by regressing onto empirical `E[vt|xt]` samples. Targets for larger `d` shortcuts are constructed by concatenating a sequence of two `d/2` shortcuts. Both objectives can be trained jointly; shortcut models do not require a two-stage procedure or discretization schedule.

![Showcase Figire](data/fig-method5.png)

### Using the code

This codebase is written in JAX, and was developed on TPU-v3 machines. You should start by installing the conda dependencies from `environment.yml` and `requirements.txt`. To load datasets, we use TFDS, and you can see our specific dataloaders at [https://github.com/kvfrans/tfds_builders](https://github.com/kvfrans/tfds_builders), of course you are free to use your own dataloader as well. 

To train a DiT-B scale model on CelebA:
```
python train.py --model.hidden_size 768 --model.patch_size 2 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4 --dataset_name celebahq256 --fid_stats data/celeba256_fidstats_ours.npz --model.cfg_scale 0 --model.class_dropout_prob 1 --model.num_classes 1 --batch_size 64 --max_steps 410_000 --model.train_type shortcut
```
or on Imagenet-256:
``` 
python train.py --model.hidden_size 768 --model.patch_size 2 --model.depth 12 --model.num_heads 12 --model.mlp_ratio 4 --dataset_name imagenet256 --fid_stats data/imagenet256_fidstats_ours.npz --model.cfg_scale 1.5 --model.class_dropout_prob 0.1 --model.bootstrap_cfg 1 --batch_size 256 --max_steps 810_000 --model.train_type shortcut
```

A larger DiT-XL scale model can be trained via:
``` 
python train.py --model.hidden_size 1152 --model.patch_size 2 --model.depth 28 --model.num_heads 16 --model.mlp_ratio 4 --dataset_name imagenet256 --fid_stats data/imagenet256_fidstats_ours.npz --model.cfg_scale 1.5 --model.class_dropout_prob 0.1 --model.bootstrap_cfg 1 --batch_size 256 --max_steps 810_000 --model.train_type shortcut
```

This code also supports `--model.sharding fsdp` for fully-sharded data parallelism, which is recommended if you are training on a multi-GPU or TPU machine.

### GMM-Based Naive Flow Matching

On the `gmm` branch, `--model.train_type naive` is a GMM-conditioned flow-matching path. It first fits a diagonal GMM on StableVAE latents, then for every training latent `x_1` infers the hard component `k = argmax q(k|x_1)`, samples `x_0 ~ N(mu_k, sigma_k)`, and conditions the DiT on the selected component `mu_k/sigma_k`.

Fit the GMM:

```bash
python data_prep.py \
  --dataset_name celebahq256 \
  --tfds_data_dir /kaggle/working/tfds \
  --batch_size 64 \
  --gmm_save_path /kaggle/working/celebahq256_gmm_stats.npz \
  --gmm_num_modes 64 \
  --gmm_em_iters 25 \
  --gmm_em_restarts 1 \
  --gmm_pi_prior_type dirichlet \
  --gmm_pi_prior_strength 1e-2 \
  --gmm_var_prior_type none \
  --gmm_var_prior_strength 0 \
  --gmm_var_prior_target_var 1.0 \
  --gmm_min_std 0.0 \
  --gmm_min_std_data_frac 1.0 \
  --gmm_standardize_data 0 \
  --metrics_output_path /kaggle/working/gmm_diagnostics/gmm_metrics.json \
  --gmm_em_metrics_output_path /kaggle/working/gmm_diagnostics/gmm_em_metrics.jsonl
```

`--gmm_pi_prior_type` controls how component weights are pulled toward uniform during EM:
`dirichlet` uses the original symmetric pseudo-count update, `kl` optimizes the pi M-step with a `D_KL(pi || uniform)` penalty, and `none` uses the maximum-likelihood count update. Increase `--gmm_pi_prior_strength` to make either regularizer stronger. For KL mode, the strength is on the same count scale as the EM soft counts; for example, with 32768 fit samples and 64 modes, `512` is roughly one ideal component count.

By default `--gmm_standardize_data 0`, so the GMM is fit and queried directly in the original StableVAE latent space. This avoids per-dimension rescaling of the latent coordinates. If `--gmm_standardize_data 1` is set, the GMM fit uses `(x - mean) / std` internally and stores the inverse transform in the `.npz`.
At train/eval time the stored transform is inverted automatically: posterior inference first standardizes incoming `x_1`, while gathered `mu_k`, `sigma_k`, and sampled `x_0` are returned in the original latent space before they enter flow matching. Diagnostics report both `fit_space_*` metrics and `latent_*` metrics so raw-vs-standardized runs can be compared in the original latent coordinates.

`--gmm_min_std` and `--gmm_min_std_data_frac` are hard variance floors: after the variance M-step, every diagonal component variance is clamped to at least the effective floor in the active GMM fit space. With the default unscaled fit, `gmm_min_std_data_frac` means a fraction of each latent dimension's original data std. `--gmm_var_prior_type kl` adds a softer variance regularizer before that clamp. It pulls each component variance toward `--gmm_var_prior_target_var` in the active GMM fit space with strength `--gmm_var_prior_strength`; use this to tune coverage pressure without relying only on the hard floor. `none` leaves the variance M-step at maximum likelihood plus the hard floor.
`gmm_metrics.json` contains the final diagnostics plus the full EM trace after fitting completes, while `gmm_em_metrics.jsonl` is streamed once per EM iteration during fitting. Both outputs also get CSV companions (`gmm_metrics.csv`, `gmm_em_metrics.csv`) with long-form rows `phase,step,metric,value`; the final CSV uses the same `gmm/...` numeric metric names that are sent to W&B.

GMM initialization is configurable. The default remains the previous behavior: `--gmm_init_strategy auto` uses k-means++ when `--gmm_kmeanspp_init 1` is set. New explicit strategies are `random`, `kmeans++`, `farthest`, `pca`, and `split`; `--gmm_init_warmup_iters` adds Lloyd/k-means refinement before EM, while `--gmm_em_restarts` runs multiple EM restarts and keeps the best final train NLL. PCA initialization is controlled by `--gmm_init_pca_dims` and `--gmm_init_pca_max_samples`. See [ablations_gmm_vi.md](ablations_gmm_vi.md) and [configs/gmm_init_ablation_grid.json](configs/gmm_init_ablation_grid.json) for the focused five-way init ablation.

Train GMM-conditioned FM:

```bash
python train.py \
  --dataset_name celebahq256 \
  --tfds_data_dir /kaggle/working/tfds \
  --fid_stats data/celeba256_fidstats_ours.npz \
  --model.train_type naive \
  --model.gmm_stats_path /kaggle/working/celebahq256_gmm_stats.npz \
  --model.gmm_cond_channels 64 \
  --eval_fid_timesteps 1,4,32,128 \
  --metrics_output_path /kaggle/working/gmm_diagnostics/train_metrics.jsonl
```

The old Gaussian flow-matching baseline remains available as `--model.train_type naive-gaussian`.

Numeric metrics sent to W&B are also appended to the CSV companion of `--metrics_output_path`. For example, `train_metrics.jsonl` creates `train_metrics.csv` with `phase,step,metric,value` rows for train loss, valid loss, activation norms, GMM assignment diagnostics, `x0/x1/v_target` magnitude and variance diagnostics, FID/flow metrics, and the FM loss decomposition (`training/fm/loss_residual_variance`, `training/fm/loss_residual_mean_sq`, per-sample loss variance/std, target variance, and prediction variance). Image-only W&B artifacts are not written to CSV.

### GMM Ablations on Kaggle

The ablation template [shortcut-model-gmm-ablation.ipynb](shortcut-model-gmm-ablation.ipynb) runs one GMM-only diagnostics job from an embedded config. It fits the GMM and writes diagnostics; it does not train the FM model. It downloads the dataset inside the notebook with:

```bash
kaggle datasets download -d codemaivanngu/shortcut-celebahq256 --unzip
```

The grid in [configs/gmm_ablation_grid.json](configs/gmm_ablation_grid.json) sweeps:

- `gmm_num_modes`
- `gmm_min_var`
- `gmm_min_var_data_frac`
- `gmm_pi_prior_type`
- `gmm_pi_prior_strength`
- `gmm_var_prior_type`
- `gmm_var_prior_strength`
- `gmm_var_prior_target_var`
- `gmm_standardize_data`

When the grid contains a `coverage` list, each entry explicitly selects one coverage regime. This avoids accidentally testing only the Cartesian product of every floor with every soft prior. The current mesh includes `ml-no-coverage`, `hard*`, and `soft-*` regimes only, so hard-only and soft-only coverage pressure can be compared directly without combined hard+soft runs.
The grid names hard floors in variance units (`gmm_min_var`, `gmm_min_var_data_frac`). The staging script converts those values to the current runtime `data_prep.py` std-floor flags when rendering notebooks.
For a focused raw-vs-standardized comparison, use [configs/gmm_standardize_ablation_grid.json](configs/gmm_standardize_ablation_grid.json). It runs only `gmm_standardize_data=1` jobs, then reuses the previous raw GMM ablation report as the baseline. The standardized jobs match existing raw configs across `K in {16,32}`, selected pi priors, no-coverage, hard floor `0.5/1.0`, and soft variance `target_var=1.0,strength=512`. Rank these runs primarily by latent-space metrics such as `latent_component_variance_mean`, `latent_var_floor_hit_rate`, `latent_overlap_proxy_max`, dead components, count ratio, and downstream FM/FID if trained; standardized-space NLL is not directly comparable to raw-space NLL.
For standardized runs, `latent_train_nll` and `latent_valid_nll` add the diagonal change-of-variables term `sum(log(std))`, so these are the NLL fields to compare against raw fits. After collecting results, generate a paired raw/std comparison with:

```bash
python scripts/compare_gmm_standardization.py \
  --results-json reports/gmm_standardize_results.json \
  --baseline-json reports/gmm_ablation_results_20260508.json \
  --output-md reports/gmm_standardize_comparison.md
```

For the short TPU check, [configs/gmm_standardize_top4_grid.json](configs/gmm_standardize_top4_grid.json) contains four standardized reruns matched to strong previous raw baselines: the best-NLL K32 run, the best balanced hard-floor K32 run, and the best K16 no-floor/hard-floor pair.

To run those four standardized configs through the full GMM-FM path, submit generated private notebooks with:

```bash
python scripts/submit_gmm_fm_jobs.py \
  --grid-config configs/gmm_standardize_top4_grid.json \
  --owners all \
  --exclude-owners kieutung,no1ceboy \
  --accelerator tpu \
  --report-path reports/gmm_fm_standardize_top4_submit.json
```

Each generated notebook downloads the CelebA-HQ payload, reuses prebuilt TFDS files when available, fits the selected GMM config, then runs `train.py --model.train_type naive` with the produced `gmm_stats.npz`. Outputs are written under `/kaggle/working/gmm_fm/<run>/diagnostics`, including `gmm_metrics.json`, `gmm_em_metrics.jsonl`, `train_metrics.jsonl`, and the matching CSV metric files. The submit helper uses the shared Kaggle context by default, so active kernels from other reports/branches count against `--max-submit-per-owner` before selecting an account.

Stage notebooks without pushing:

```bash
python scripts/stage_gmm_ablation_jobs.py \
  --owner codemaivanngu \
  --batch-size 8 \
  --manifest-path reports/gmm_ablation_stage_manifest.json \
  --limit 2
```

This render step is deterministic from [configs/gmm_ablation_grid.json](configs/gmm_ablation_grid.json): changing the mesh and rerunning the command creates a new set of staged notebooks plus a JSON/Markdown manifest that records which grid indexes and run names were packed into each notebook.

Push GPU or TPU jobs directly to Kaggle:

```bash
python scripts/push_gmm_ablation_jobs.py \
  --owners codemaivanngu \
  --accelerator NvidiaTeslaT4 \
  --report-path reports/gmm_ablation_submit.json \
  --limit 2
```

Use `--owners all` to distribute jobs round-robin across accounts listed in `.secrets/all-kaggle.json`; use `--exclude-owners` to skip accounts that should not receive a job, and set `--accelerator TpuV5E8` for Kaggle TPU sessions. The push helper writes JSON and Markdown submit reports to `--report-path`, and it records submitted, failed, and not-submitted rows as the batch progresses. Staged notebooks are written under `kaggle_staging/`, which is ignored by git; the source notebook only contains a W&B placeholder, while the staging script injects `WANDB_API_KEY` from `.secrets/.env` into the pushed private notebook.

When notebook dataset payloads include a `data/` directory, the notebooks merge those files into the cloned repo `data/` directory instead of replacing it. This keeps repo-local auxiliary files such as `data/imagenet_labels.txt` available for train/eval while still allowing dataset-provided FID stats or cached assets to override/add files.

Kaggle Python 3.12 images can combine older TFDS metadata protos with a newer protobuf runtime. The notebooks pin `protobuf<4` for TFDS tooling and run `tfds build` with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`; this is intentionally scoped to notebook dataset preparation and does not change the repo TPU/JAX environment in `pyproject.toml`.

After jobs have been submitted, collect completed diagnostics without downloading the full Kaggle output:

```bash
python scripts/collect_gmm_ablation_results.py \
  --submit-report reports/gmm_ablation_tpu_reconciled_20260507.json \
  --output-root outputs/kaggle/gmm_ablation_results \
  --report-path reports/gmm_ablation_results.json
```

The collector checks each kernel status, downloads only `gmm_metrics.json`, `gmm_em_metrics.jsonl`, `gmm_prep_stdout.txt`, and `gmm_prep_stderr.txt` for completed jobs, then writes aggregate JSON and Markdown reports with NLL, cluster balance, dead-component, variance-floor, and overlap metrics. For batch notebooks, status and downloads are cached by `kernel_id` so each Kaggle kernel is queried once even when it contains many config rows.

For longer ablation sweeps, use a persistent queue instead of manually tracking offsets:

```bash
python scripts/manage_gmm_ablation_queue.py \
  --queue-path reports/gmm_ablation_queue.json \
  --seed-report reports/gmm_ablation_results_20260507.json \
  --owners all \
  --exclude-owners kieutung \
  --accelerator TpuV5E8 \
  --sync-status \
  --push \
  --limit 12
```

The queue is rendered from the current grid every run. If new grid rows are added later, they appear as `pending`; existing `pending`, `running`, `complete`, and `failed` rows are carried forward by a stable config key. A successful Kaggle submit is marked `running` immediately, and Kaggle `QUEUED`/`RUNNING` statuses also remain `running` until a later `--sync-status` changes them to `complete` or `failed`.
Use `--reset` when the grid has been intentionally reshaped and old grid-index-only reports should not be carried into the new queue.

The queue manager also builds a shared Kaggle running-context report before pushing. By default it scans `reports/*.json`, queries live Kaggle kernel status, writes `reports/kaggle_shared_context.json` plus a Markdown companion, and counts active kernels from other reports/branches against each owner before selecting the next account. This prevents the GMM queue from blindly submitting to an account that is already busy with a `moe2` TIDE/FM job recorded in another report:

```bash
python scripts/kaggle_shared_context.py \
  --accounts-file /home/tung/all-kaggle.json \
  --report-glob 'reports/*.json' \
  --output reports/kaggle_shared_context.json
```

Use `--shared-context-glob` on `manage_gmm_ablation_queue.py` to add or narrow report sources, `--no-live-shared-context` for a fast local-only reconciliation, and `--no-shared-context` only when intentionally ignoring activity from other branch reports.

Because Kaggle TPU queue time can dominate the 5-6 minute GMM-only fit, the queue manager can pack multiple configs into one notebook with `--batch-size`. The batch notebook downloads/builds the dataset once, syncs the repo once, then runs `data_prep.py` sequentially for each embedded config and writes per-run diagnostics plus `/kaggle/working/gmm_ablation_batch/batch_summary.jsonl`:

```bash
python scripts/manage_gmm_ablation_queue.py \
  --queue-path reports/gmm_ablation_queue.json \
  --owners all \
  --exclude-owners kieutung,no1ceboy \
  --accelerator tpu \
  --sync-status \
  --push \
  --batch-size 8 \
  --limit 80
```

To rerun the 180 raw GMM configs with a longer EM schedule, use [configs/gmm_ablation_em100_grid.json](configs/gmm_ablation_em100_grid.json). It is the same raw mesh as the EM25 grid, but sets `gmm_em_iters=100` and appends `em100` to run names so W&B and Kaggle outputs do not collide with earlier runs. The stable queue key includes `gmm_em_iters`, `ablation_tag`, and `run_name_suffix`, so EM25 and EM100 rows can coexist in separate queue reports:

```bash
python scripts/manage_gmm_ablation_queue.py \
  --queue-path reports/gmm_ablation_em100_queue_20260521.json \
  --grid-config configs/gmm_ablation_em100_grid.json \
  --accounts-file .secrets/all-kaggle.json \
  --owners all \
  --exclude-owners kieutung \
  --accelerator tpu \
  --reset \
  --sync-status \
  --push \
  --batch-size 10 \
  --max-submit-per-owner 1 \
  --shared-context-output reports/kaggle_shared_context_em100_20260521.json
```

Collect EM100 diagnostics, including timed-out partial logs when Kaggle reports `CANCEL_ACKNOWLEDGED`, with:

```bash
python scripts/collect_gmm_ablation_results.py \
  --submit-report reports/gmm_ablation_em100_queue_20260521.json \
  --grid-config configs/gmm_ablation_em100_grid.json \
  --accounts-file .secrets/all-kaggle.json \
  --output-root outputs/kaggle/gmm_ablation_em100_20260521 \
  --report-path reports/gmm_ablation_em100_results_20260521.json \
  --download-statuses COMPLETE,CANCEL_ACKNOWLEDGED
```

Then compare EM100 against the EM25 baseline report before deciding whether any FM/TIDE source should be rerun:

```bash
python scripts/compare_gmm_em_iters.py \
  --baseline-json reports/gmm_ablation_results_20260508.json \
  --candidate-json reports/gmm_ablation_em100_results_20260521.json \
  --output-json reports/gmm_ablation_em25_vs_em100_20260521.json
```

The compare report recommends FM reruns only when EM100 improves latent valid NLL by at least 0.5% on a known source config without introducing dead components, a large pi-entropy drop, a large count-ratio increase, or a larger overlap proxy. The GMM final metrics also include EM convergence fields such as `nll_delta_last10`, `nll_delta_25_to_final`, `nll_delta_50_to_final`, `final_minus_best_train_nll`, and `train_valid_nll_gap`.

For fast qualitative debugging before launching CelebA jobs, use the self-contained toy notebook [toy-gmm-fm-insight.ipynb](toy-gmm-fm-insight.ipynb). It creates 2D blobs/rings/moons, fits small diagonal GMMs, compares Gaussian, hard-GMM, and top-k GMM source constructions, and writes `toy_outputs/toy_gmm_fm_executed.ipynb` with embedded plots for downloading from Kaggle:

```bash
python scripts/create_toy_gmm_fm_notebook.py
```

For larger toy sweeps, use CPU for the clustering/source-geometry notebook and GPU only when training toy FM models:

```bash
python scripts/create_toy_gmm_big_ablation_notebook.py
python scripts/create_toy_fm_gpu_ablation_notebook.py
```

`toy-gmm-big-ablation.ipynb` runs larger NumPy-only clustering/source diagnostics. `toy-fm-gpu-ablation.ipynb` trains small JAX MLP flow-matching vector fields for each source construction, so GPU acceleration is useful there.

### Sanity Checking

Shorcut models trained with the provided functions should achieve the following FID-50k performance.

|                           | 128-Step| 4-Step  | 1-Step  |
| --------                  | ------- | ------- | ------- |
| CelebA (DiT-B)            | 6.9     | 13.8    | 20.5    |
| Imagenet-256 (DiT-B)      | 15.5    | 28.3    | 40.3    |
| Imagenet-256 (DiT-XL)     | 3.8     | 7.8     | 10.6    |

### Checkpoints and FID Stats

Pretrained model checkpoints, and pre-computed reference FID stats for CelebA and Imagenet can be downloaded from [this drive](https://drive.google.com/drive/folders/1g665i0vMxm8qqqcp5mAiexnL919-gMwW?usp=sharing). To load a checkpoint, use the `--load_dir` flag. 
