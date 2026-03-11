# Hướng dẫn repo `shortcut-models`

Tài liệu này mô tả repo theo góc nhìn vận hành: entrypoint nào dùng để làm gì, các mode train hiện có, dữ liệu nào được hỗ trợ, và những điểm cần cẩn thận khi chạy thật.

## 1. Mục tiêu của repo

Repo này hiện thực bài toán sinh ảnh bằng Shortcut Models trên JAX/Flax. Một mô hình DiT duy nhất được huấn luyện để:

- học flow-matching objective chuẩn
- học thêm các shortcut target với bước nhảy lớn hơn
- cho phép suy luận với nhiều ngân sách bước lấy mẫu khác nhau, kể cả 1 bước

Ngoài mode `shortcut`, repo còn giữ các baseline để so sánh như `naive`, `sit`, `progressive`, `consistency`, `consistency-distillation`, và `livereflow`.

## 2. Cấu trúc quan trọng

### Entry points

- `train.py`: file chính. Dùng cho train, eval định kỳ, load checkpoint, và cả inference khi `--mode` khác `train`.
- `helper_eval.py`: vẽ reconstruction, sample grids, activation norms, và tính FID trong quá trình train.
- `helper_inference.py`: chạy sampling/FID khi không train.

### Model và target

- `model.py`: DiT backbone, có conditioning theo `t`, `dt`, và nhãn class.
- `targets_shortcut.py`: sinh target cho thuật toán shortcut.
- `baselines/targets_naive.py`: flow matching chuẩn.
- `baselines/targets_sit.py`: sinh target continuous-time cho baseline SiT.
- `baselines/targets_progressive.py`: progressive teacher bootstrap.
- `baselines/targets_consistency_training.py`: consistency training.
- `baselines/targets_consistency_distillation.py`: consistency distillation.
- `baselines/targets_livereflow.py`: live reflow target.
- `utils/sit_transport.py`: toán transport dùng chung cho SiT, gồm path `linear/gvp/vp`, target `velocity/score/noise`, weighting, và sampler ODE/SDE fixed-step.

### Hạ tầng

- `utils/datasets.py`: pipeline TFDS.
- `utils/sharding.py`: `dp` và `fsdp`.
- `utils/train_state.py`: train state + EMA params.
- `utils/checkpoint.py`: checkpoint pickle đơn giản.
- `utils/stable_vae.py`: encode/decode latent bằng Stable VAE.
- `utils/fid.py`: mạng Inception và phép tính FID.
- `utils/wandb.py`: khởi tạo W&B.

## 3. Luồng chạy thực tế

### Train

`train.py` làm các bước chính sau:

1. Parse flags và config.
2. Tạo dataset train/validation từ TFDS.
3. Nếu `model.use_stable_vae=1`, ảnh sẽ được encode sang latent trước khi vào DiT.
4. Khởi tạo model DiT và optimizer `adamw`.
5. Chọn hàm target theo `model.train_type`.
6. Mỗi step:
   - lấy batch
   - encode latent nếu cần
   - tính target
   - backprop
   - update EMA
7. Định kỳ:
   - log loss lên W&B
   - gọi `helper_eval.py`
   - save checkpoint nếu có `save_dir`

### Inference

Nếu `--mode` khác `train`, `train.py` bỏ qua vòng lặp train và chuyển sang `do_inference(...)` trong `helper_inference.py`.

Mode này sẽ:

- load checkpoint nếu có `--load_dir`
- tạo noise đầu vào
- chạy Euler sampling cho các mode legacy, hoặc transport ODE/SDE fixed-step cho `sit`
- tính FID với `--fid_stats`
- lưu `x_render.npy` vào `save_dir` nếu được chỉ định

Lưu ý quan trọng:

- `--mode interpolate` có branch riêng nhưng trong code hiện còn `breakpoint()`.
- inference path gần như giả định rằng bạn có `fid_stats`; nếu thiếu, đường chạy này không phải lựa chọn an toàn.

## 4. Dataset đang hỗ trợ

Theo `utils/datasets.py`, repo hiện hỗ trợ:

| `dataset_name` | Nguồn TFDS | Ghi chú |
| --- | --- | --- |
| `imagenet256` | `imagenet2012` | crop vuông rồi resize `256x256` |
| `celebahq256` | `celebahq256` | thường cần builder TFDS riêng |
| `lsunchurch` | `lsunc` | dùng split `church-train` / `church-test` |

Điểm dễ nhầm:

- README cũ dùng `celeb_a_hq`, nhưng code hiện tại nhận `celebahq256`.
- Repo gốc có nhắc tới `tfds_builders`; với `celebahq256`, bạn nên chuẩn bị builder tương ứng trước.

## 5. Các cờ quan trọng

### Cờ top-level

| Cờ | Ý nghĩa |
| --- | --- |
| `--dataset_name` | Chọn dataset |
| `--load_dir` | File checkpoint để load |
| `--save_dir` | Prefix/path đầu ra cho checkpoint hoặc ảnh sinh |
| `--fid_stats` | File `.npz` chứa `mu` và `sigma` cho FID |
| `--batch_size` | Global batch size |
| `--max_steps` | Số step train |
| `--mode` | `train` hoặc bất kỳ giá trị nào khác để chạy inference |
| `--debug_overfit` | ép dataset nhỏ để debug |

### Cờ trong `model.*`

| Cờ | Ý nghĩa |
| --- | --- |
| `--model.hidden_size` | chiều ẩn của DiT |
| `--model.patch_size` | kích thước patch embedding |
| `--model.depth` | số block Transformer |
| `--model.num_heads` | số attention heads |
| `--model.mlp_ratio` | MLP ratio của block |
| `--model.cfg_scale` | CFG scale cho conditional model |
| `--model.class_dropout_prob` | dropout nhãn phục vụ classifier-free guidance |
| `--model.num_classes` | số lớp; với unconditional thường đặt `1` |
| `--model.denoise_timesteps` | số bước chuẩn của flow horizon |
| `--model.train_type` | chọn thuật toán train |
| `--model.sharding` | `dp` hoặc `fsdp` |
| `--model.use_stable_vae` | encode ảnh sang latent trước khi train/sample |
| `--model.bootstrap_cfg` | bật CFG trong bootstrap target của shortcut/progressive |

### Cờ `model.*` riêng cho `sit`

| Cờ | Ý nghĩa |
| --- | --- |
| `--model.transport_path_type` | chọn interpolant path: `linear`, `gvp`, hoặc `vp` |
| `--model.transport_prediction` | chọn target model học: `velocity`, `score`, hoặc `noise` |
| `--model.transport_loss_weight` | chọn weighting: `none`, `velocity`, hoặc `likelihood` |
| `--model.transport_train_eps` | override epsilon train; bỏ qua để dùng mặc định theo SiT |
| `--model.transport_sample_eps` | override epsilon sample; bỏ qua để dùng mặc định theo SiT |

Mặc định đang bám logic official của SiT:

- `linear/gvp + velocity` dùng `train_eps=0`, `sample_eps=0`
- `linear/gvp + score|noise` dùng `train_eps=1e-3`, `sample_eps=1e-3`
- `vp` dùng `train_eps=1e-5`, `sample_eps=1e-3`

### Cờ inference

| Cờ | Ý nghĩa |
| --- | --- |
| `--inference_timesteps` | số bước sample |
| `--inference_generations` | số lượng mẫu để tính FID |
| `--inference_cfg_scale` | CFG scale lúc sample |
| `--inference_transport` | với `sit`: chọn `ode` hoặc `sde` |
| `--inference_sampling_method` | với `sit`: chọn solver `euler` hoặc `heun` |
| `--inference_diffusion_form` | với `sit` và `sde`: dạng diffusion coefficient |
| `--inference_diffusion_norm` | với `sit` và `sde`: hệ số nhân diffusion |
| `--inference_last_step` | với `sit` và `sde`: bước hiệu chỉnh cuối `none/mean/tweedie/euler` |
| `--inference_last_step_size` | với `sit` và `sde`: độ lớn bước hiệu chỉnh cuối |

## 6. Ý nghĩa các `train_type`

### `shortcut`

Mode chính của bài báo. Batch được chia thành hai phần:

- phần bootstrap target ở nhiều `dt`
- phần flow-matching target chuẩn

Sau đó hai phần được ghép lại để train chung.

### `naive`

Flow matching cơ bản, không dùng shortcut target.

### `sit`

Baseline Scalable Interpolant Transformers dùng cùng backbone DiT nhưng đổi sang continuous-time transport. Mode này:

- bỏ conditioning `dt` ở backbone
- cho phép chọn path `linear`, `gvp`, hoặc `vp`
- cho phép học `velocity`, `score`, hoặc `noise`
- dùng cùng entrypoint inference nhưng có thêm sampler `ode/sde` fixed-step

### `progressive`

Dùng một teacher state để bootstrap dần từ bước lớn sang bước nhỏ theo tiến độ train.

### `consistency`

Dùng EMA model để sinh consistency target.

### `consistency-distillation`

Distill từ teacher EMA sang student consistency objective.

### `livereflow`

Sinh một phần reflow target online bằng cách tự rollout model hiện tại.

## 7. Thiết lập môi trường

Repo hiện có hai hướng setup:

### Khuyến nghị: `uv`

```bash
uv sync
```

Ưu điểm:

- đã có `uv.lock`
- README hiện dùng `uv run`
- khớp hơn với `pyproject.toml`

### Legacy: conda + pip

```bash
conda env create -f environment.yml
conda activate project-brc
pip install -r requirements.txt
```

Lưu ý:

- `pyproject.toml` yêu cầu Python `3.11.6`
- `environment.yml` đang ghi Python `3.10`
- nếu bạn muốn một đường setup duy nhất, nên ưu tiên `uv`

## 8. Ví dụ lệnh nên dùng

### Train shortcut trên CelebA-HQ

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

### Train shortcut trên ImageNet-256

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

### Train SiT trên ImageNet-256

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

### Sample/FID từ checkpoint đã train

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

### Sample/FID từ checkpoint SiT

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

## 9. Checkpoint, logging và output

### Checkpoint

- Code save checkpoint qua `utils/checkpoint.py`.
- Trong loop train, đường dẫn được tạo bằng cách nối `save_dir` và step number trực tiếp.
- Vì vậy nên truyền `save_dir` theo kiểu có dấu `/` ở cuối, ví dụ `checkpoints/run_a/`, để path tạo ra rõ ràng hơn.

### W&B

- Chỉ process `0` khởi tạo W&B khi `mode=train`.
- Có thể dùng `--wandb.offline true` để log local.

### Output inference

- `helper_inference.py` lưu `x_render.npy` vào thư mục `save_dir`.
- Nếu dùng Stable VAE, latent sẽ được decode về ảnh trước khi resize cho FID.

## 10. Các điểm cần cẩn thận

- Khi load checkpoint, bạn phải truyền lại đúng kiến trúc model đã train.
- `celebahq256` khác với `celeb_a_hq`; dùng sai tên dataset sẽ fail ngay ở loader.
- `helper_inference.py` dùng FID network trong đường chạy chính, nên chuẩn bị `--fid_stats`.
- `utils/stable_vae.py` sẽ tải VAE từ Hugging Face ở lần chạy đầu tiên.
- `utils/fid.py` sẽ tự tải trọng số Inception vào `data/` khi cần.

## 11. Bắt đầu đọc code từ đâu

Nếu bạn mới vào repo, thứ tự đọc nên là:

1. `train.py`
2. `targets_shortcut.py`
3. `model.py`
4. `helper_eval.py`
5. `helper_inference.py`
6. `utils/datasets.py`

Thứ tự này giúp thấy rõ luồng train trước, rồi mới đi vào chi tiết target và kiến trúc.
