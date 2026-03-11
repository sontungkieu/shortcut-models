## One-Step Diffusion via Shortcut Models

Kevin Frans, Danijar Hafner, Sergey Levine, Pieter Abbeel

[Paper](https://arxiv.org/abs/2410.12557)  
[Project Website](https://kvfrans.com/shortcut-models/)  
[PDF Guide](pdf/main.pdf)  
[LaTeX Source](pdf/main.tex)  
[Vietnamese Repo Guide](docs/repo-guide.md)

### Abstract

Diffusion models and flow-matching models generate high-quality images, but they usually require many denoising steps at inference time. Shortcut Models train a single DiT-based network to predict both standard flow targets and larger shortcut targets, so the same model can be sampled with different step budgets, including one-step generation.

![Showcase Figure](data/fig-showcase4.png)

### Overview

This repository contains a JAX/Flax implementation of Shortcut Models plus several baselines:

- `shortcut`: the main method from the paper
- `naive`: regular flow-matching baseline
- `sit`: Scalable Interpolant Transformers baseline with configurable transport path, prediction target, and ODE/SDE sampler
- `progressive`: teacher-assisted progressive shortening baseline
- `consistency`: consistency training baseline
- `consistency-distillation`: teacher-student consistency distillation baseline
- `livereflow`: online reflow-style baseline

The main entrypoint is `train.py`. The same script is used for:

- training
- evaluation during training
- offline sampling / FID runs when `--mode` is not `train`

![Method Figure](data/fig-method5.png)

### Repository Layout

- `train.py`: flag definitions, model setup, training loop, checkpoint I/O
- `model.py`: DiT backbone with timestep, label, and shortcut-step conditioning
- `targets_shortcut.py`: shortcut target construction
- `baselines/`: alternative target builders for baseline methods
- `utils/sit_transport.py`: shared SiT transport math, target construction, and fixed-step ODE/SDE samplers
- `helper_eval.py`: visualization and periodic FID evaluation during training
- `helper_inference.py`: standalone sampling / FID path for non-training modes
- `utils/datasets.py`: TFDS input pipelines
- `utils/stable_vae.py`: Stable VAE encode/decode wrapper
- `utils/checkpoint.py`: lightweight pickle-based checkpoint helper

### Environment Setup

This codebase was developed for JAX on TPU-v3, but it also includes `dp` and `fsdp` sharding modes for multi-device runs.

Preferred setup with `uv`:

```bash
uv sync
```

Alternative setup from the legacy conda files:

```bash
conda env create -f environment.yml
conda activate project-brc
pip install -r requirements.txt
```

Notes:

- `pyproject.toml` pins Python `3.11.6`, while `environment.yml` uses Python `3.10`.
- README command examples below use `uv run ...` because the repo already includes `uv.lock`.
- `utils/stable_vae.py` downloads `pcuenq/sd-vae-ft-mse-flax` from Hugging Face on first use.
- `utils/fid.py` downloads Inception weights into `data/` on first FID run.

### Supported Datasets

`utils/datasets.py` currently supports these dataset names:

- `imagenet256`
- `celebahq256`
- `lsunchurch`

Important details:

- ImageNet is loaded from TFDS `imagenet2012`.
- `celebahq256` may require the custom TFDS builder referenced in the original paper README.
- The old `celeb_a_hq` example does not match the current loader; use `celebahq256`.

### Common Flags

Top-level flags from `train.py` and `helper_inference.py`:

| Flag | Purpose |
| --- | --- |
| `--dataset_name` | Selects the TFDS pipeline |
| `--fid_stats` | Reference statistics for FID evaluation |
| `--load_dir` | Path to a serialized checkpoint to resume or sample from |
| `--save_dir` | Output prefix for checkpoints and generated arrays |
| `--mode` | `train` by default; any other value enters the inference path |
| `--batch_size` | Global batch size |
| `--max_steps` | Number of optimization steps |
| `--model.train_type` | `shortcut`, `naive`, `sit`, `progressive`, `consistency`, `consistency-distillation`, or `livereflow` |
| `--model.sharding` | `dp` or `fsdp` |
| `--model.denoise_timesteps` | Training-time flow horizon, default `128` |
| `--inference_timesteps` | Sampling step budget for non-training modes |
| `--inference_generations` | Number of generated samples in the inference path |
| `--inference_cfg_scale` | CFG scale used during sampling |

SiT-specific flags:

| Flag | Purpose |
| --- | --- |
| `--model.transport_path_type` | Interpolant path: `linear`, `gvp`, or `vp` |
| `--model.transport_prediction` | Model target: `velocity`, `score`, or `noise` |
| `--model.transport_loss_weight` | SiT weighting rule: `none`, `velocity`, or `likelihood` |
| `--model.transport_train_eps` | Optional training epsilon override; omit to use the SiT default for the selected path/prediction |
| `--model.transport_sample_eps` | Optional sampling epsilon override; omit to use the SiT default for the selected path/prediction |
| `--inference_transport` | SiT sampler family: `ode` or `sde` |
| `--inference_sampling_method` | Fixed-step solver: `euler` or `heun` |
| `--inference_diffusion_form` | SDE diffusion schedule: `constant`, `sbdm`, `sigma`, `linear`, `decreasing`, or `increasing-decreasing` |
| `--inference_diffusion_norm` | Multiplier for the SiT SDE diffusion term |
| `--inference_last_step` | Optional SiT SDE last-step correction: `none`, `mean`, `tweedie`, or `euler` |
| `--inference_last_step_size` | Step size for the optional SiT SDE last-step correction |

Operational notes:

- When loading a checkpoint, you must pass the same architecture flags used for training.
- Training concatenates `save_dir` and the step number directly, so using a trailing slash is the safest convention, for example `--save_dir checkpoints/imagenet256-shortcut/`.
- Non-training modes rely on the inference helper and are most useful when `--fid_stats` is provided.
- The local SiT defaults match the official transport rules: `linear/gvp + velocity => eps=0`, `linear/gvp + score|noise => eps=1e-3`, and `vp => train_eps=1e-5`, `sample_eps=1e-3`.

### Training Examples

Train a DiT-B shortcut model on CelebA-HQ:

```bash
uv run train.py \
  --dataset_name celebahq256 \
  --fid_stats data/celeba256_fidstats_ours.npz \
  --save_dir checkpoints/celebahq256-shortcut/ \
  --batch_size 64 \
  --max_steps 410000 \
  --model.hidden_size 768 \
  --model.patch_size 2 \
  --model.depth 12 \
  --model.num_heads 12 \
  --model.mlp_ratio 4 \
  --model.cfg_scale 0 \
  --model.class_dropout_prob 1 \
  --model.num_classes 1 \
  --model.train_type shortcut
```

Train a DiT-B shortcut model on ImageNet-256:

```bash
uv run train.py \
  --dataset_name imagenet256 \
  --fid_stats data/imagenet256_fidstats_ours.npz \
  --save_dir checkpoints/imagenet256-shortcut-b/ \
  --batch_size 256 \
  --max_steps 810000 \
  --model.hidden_size 768 \
  --model.patch_size 2 \
  --model.depth 12 \
  --model.num_heads 12 \
  --model.mlp_ratio 4 \
  --model.cfg_scale 1.5 \
  --model.class_dropout_prob 0.1 \
  --model.bootstrap_cfg 1 \
  --model.train_type shortcut
```

Train a DiT-XL shortcut model on ImageNet-256:

```bash
uv run train.py \
  --dataset_name imagenet256 \
  --fid_stats data/imagenet256_fidstats_ours.npz \
  --save_dir checkpoints/imagenet256-shortcut-xl/ \
  --batch_size 256 \
  --max_steps 810000 \
  --model.hidden_size 1152 \
  --model.patch_size 2 \
  --model.depth 28 \
  --model.num_heads 16 \
  --model.mlp_ratio 4 \
  --model.cfg_scale 1.5 \
  --model.class_dropout_prob 0.1 \
  --model.bootstrap_cfg 1 \
  --model.train_type shortcut
```

Useful variants:

- Use `--model.train_type naive` for a regular flow model.
- Use `--model.train_type sit` to train the SiT-style transport baseline with the same DiT backbone.
- Use `--model.sharding fsdp` for fully sharded data parallelism.
- Use `--debug_overfit 1` to loop over a tiny repeated dataset for debugging.
- Use `--wandb.offline true` if you want local W&B logging only.

Train a SiT baseline on ImageNet-256 with the default linear-velocity transport:

```bash
uv run train.py \
  --dataset_name imagenet256 \
  --fid_stats data/imagenet256_fidstats_ours.npz \
  --save_dir checkpoints/imagenet256-sit-b/ \
  --batch_size 256 \
  --max_steps 810000 \
  --model.hidden_size 768 \
  --model.patch_size 2 \
  --model.depth 12 \
  --model.num_heads 12 \
  --model.mlp_ratio 4 \
  --model.cfg_scale 1.5 \
  --model.class_dropout_prob 0.1 \
  --model.train_type sit \
  --model.transport_path_type linear \
  --model.transport_prediction velocity \
  --model.transport_loss_weight none
```

### Sampling and FID Runs

Any `--mode` other than `train` routes into `helper_inference.py`.

Example: sample from a trained ImageNet checkpoint with 4 denoising steps:

```bash
uv run train.py \
  --mode inference \
  --dataset_name imagenet256 \
  --fid_stats data/imagenet256_fidstats_ours.npz \
  --load_dir checkpoints/imagenet256-shortcut-b/810001 \
  --save_dir outputs/imagenet256-shortcut-b-4step \
  --batch_size 256 \
  --model.hidden_size 768 \
  --model.patch_size 2 \
  --model.depth 12 \
  --model.num_heads 12 \
  --model.mlp_ratio 4 \
  --model.cfg_scale 1.5 \
  --model.class_dropout_prob 0.1 \
  --model.bootstrap_cfg 1 \
  --model.train_type shortcut \
  --inference_timesteps 4 \
  --inference_generations 4096 \
  --inference_cfg_scale 1.5
```

Outputs from the inference path:

- rendered samples, when collected by the helper, are saved as `x_render.npy` under `save_dir`
- console FID report against `--fid_stats`

Example: sample from a SiT checkpoint with the default ODE + Heun solver:

```bash
uv run train.py \
  --mode inference \
  --dataset_name imagenet256 \
  --fid_stats data/imagenet256_fidstats_ours.npz \
  --load_dir checkpoints/imagenet256-sit-b/810001 \
  --save_dir outputs/imagenet256-sit-b-ode \
  --batch_size 256 \
  --model.hidden_size 768 \
  --model.patch_size 2 \
  --model.depth 12 \
  --model.num_heads 12 \
  --model.mlp_ratio 4 \
  --model.cfg_scale 1.5 \
  --model.class_dropout_prob 0.1 \
  --model.train_type sit \
  --model.transport_path_type linear \
  --model.transport_prediction velocity \
  --model.transport_loss_weight none \
  --inference_transport ode \
  --inference_sampling_method heun \
  --inference_timesteps 32 \
  --inference_generations 4096 \
  --inference_cfg_scale 1.5
```

Known caveat:

- `--mode interpolate` enters a special interpolation path in `helper_inference.py`, but the code currently drops into `breakpoint()` before continuing.

### Sanity Check Targets

Shortcut models trained with the provided setup should approximately match the reported FID-50k numbers:

| Model | 128-step | 4-step | 1-step |
| --- | ---: | ---: | ---: |
| CelebA-HQ (DiT-B) | 6.9 | 13.8 | 20.5 |
| ImageNet-256 (DiT-B) | 15.5 | 28.3 | 40.3 |
| ImageNet-256 (DiT-XL) | 3.8 | 7.8 | 10.6 |

### Checkpoints and External Assets

Pretrained checkpoints and reference FID statistics from the authors are linked in the original project materials:

- [Google Drive with checkpoints and FID stats](https://drive.google.com/drive/folders/1g665i0vMxm8qqqcp5mAiexnL919-gMwW?usp=sharing)

If you use this repo regularly, the Vietnamese guide in [docs/repo-guide.md](docs/repo-guide.md) gives a more detailed walkthrough of the training flow, file structure, and operational caveats.
