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

`--gmm_min_std` and `--gmm_min_std_data_frac` are hard variance floors: after the variance M-step, every diagonal component variance is clamped to at least the effective floor in the active GMM fit space. With the default unscaled fit, `gmm_min_std_data_frac` means a fraction of each latent dimension's original data std. `--gmm_var_prior_type kl` adds a softer variance regularizer before that clamp. It pulls each component variance toward `--gmm_var_prior_target_var` in the active GMM fit space with strength `--gmm_var_prior_strength`; use this to tune coverage pressure without relying only on the hard floor. `none` leaves the variance M-step at maximum likelihood plus the hard floor.
`gmm_metrics.json` contains the final diagnostics plus the full EM trace after fitting completes, while `gmm_em_metrics.jsonl` is streamed once per EM iteration during fitting.

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

### GMM-TIDE Router Flow Matching

`--model.train_type gmm-tide` is the V1 router variant. It keeps the diagonal GMM fixed, distills the GMM posterior `q_GMM(k|x)` into a CNN router `f_phi`, then freezes that router during FM training. During FM, the router reads a GMM-prior base sample, selects top-k components, samples one latent from each selected component, and forms a weighted source latent `x0_tide`. The DiT is trained from `x0_tide` to the data latent and is conditioned on the weighted `mu/sigma` aggregate.

Distill the router after fitting the GMM:

```bash
python train_gmm_router.py \
  --dataset_name celebahq256 \
  --tfds_data_dir /kaggle/working/tfds \
  --batch_size 64 \
  --gmm_stats_path /kaggle/working/celebahq256_gmm_stats.npz \
  --router_save_path /kaggle/working/celebahq256_gmm_router.pkl \
  --router_train_data_mode mix \
  --router_mix_x1_prob 0.5 \
  --router_target_type soft_kl \
  --router_max_steps 10000 \
  --metrics_output_path /kaggle/working/gmm_diagnostics/router_metrics.jsonl
```

`--router_train_data_mode x1` trains the router only on data latents. `mix` blends data latents and GMM-prior samples according to `--router_mix_x1_prob`; this is the default for V1 because the FM source path queries the router on prior-side latents. `--router_target_type soft_kl` matches the full posterior distribution, while `hard_ce` trains only against `argmax q_GMM(k|x)`.

Train FM with the frozen router:

```bash
python train.py \
  --dataset_name celebahq256 \
  --tfds_data_dir /kaggle/working/tfds \
  --fid_stats data/celeba256_fidstats_ours.npz \
  --model.train_type gmm-tide \
  --model.gmm_stats_path /kaggle/working/celebahq256_gmm_stats.npz \
  --model.gmm_router_path /kaggle/working/celebahq256_gmm_router.pkl \
  --model.gmm_router_topk 4 \
  --model.gmm_router_temperature 1.0 \
  --model.gmm_router_update_policy frozen \
  --model.gmm_cond_channels 64 \
  --eval_fid_timesteps 1,4,32,128 \
  --metrics_output_path /kaggle/working/gmm_diagnostics/train_metrics.jsonl
```

V1 intentionally supports only `--model.gmm_router_update_policy frozen`. Joint FM updates to `f_phi` should be added as a separate training state because the current optimizer state belongs only to the DiT.

Render and submit the four default Kaggle V1 runs from [configs/gmm_tide_fm_grid.json](configs/gmm_tide_fm_grid.json):

```bash
python scripts/submit_gmm_tide_fm_jobs.py \
  --owners all \
  --exclude-owners kieutung,no1ceboy \
  --accelerator tpu \
  --report-path reports/gmm_tide_fm_submit.json
```

The default mesh covers `gmm_num_modes in {16, 32}` and `gmm_router_topk in {2, 4}`. Each notebook downloads the CelebA-HQ payload with the Kaggle CLI, then runs GMM fitting, router distillation, then `train.py --model.train_type gmm-tide`, and writes GMM/router/train diagnostics under `/kaggle/working/gmm_tide_fm/<run>/diagnostics`.

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

### Sanity Checking

Shorcut models trained with the provided functions should achieve the following FID-50k performance.

|                           | 128-Step| 4-Step  | 1-Step  |
| --------                  | ------- | ------- | ------- |
| CelebA (DiT-B)            | 6.9     | 13.8    | 20.5    |
| Imagenet-256 (DiT-B)      | 15.5    | 28.3    | 40.3    |
| Imagenet-256 (DiT-XL)     | 3.8     | 7.8     | 10.6    |

### Checkpoints and FID Stats

Pretrained model checkpoints, and pre-computed reference FID stats for CelebA and Imagenet can be downloaded from [this drive](https://drive.google.com/drive/folders/1g665i0vMxm8qqqcp5mAiexnL919-gMwW?usp=sharing). To load a checkpoint, use the `--load_dir` flag. 
