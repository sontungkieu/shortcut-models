# Toy GMM/FM Ablations

Tài liệu này ghi lại các notebook toy đã port từ nhánh `gmm` sang `moe2`. Mục tiêu là kiểm tra nhanh các giả thuyết về GMM source và TIDE source trước khi tiêu tốn queue TPU cho CelebA-HQ latent.

## File Đã Port

- `toy-gmm-fm-insight.ipynb`: notebook CPU nhỏ, tự chứa dữ liệu 2D đơn giản như blobs, rings, moons. Nó so sánh Gaussian source với GMM source bằng các proxy như NLL, entropy, distance từ source tới target, vector variance và các hình scatter.
- `toy-gmm-big-ablation.ipynb`: notebook CPU/GPU nhẹ cho toy dataset lớn hơn, gồm aniso blobs, nested rings, pinwheel và cross-cov. Nó stress-test diagonal GMM, variance floor, overlap, count balance và các chiến lược khởi tạo trong toy.
- `toy-fm-gpu-ablation.ipynb`: notebook GPU/JAX nhỏ, train MLP flow-matching trên toy data để kiểm tra downstream thật hơn proxy. Nó so sánh Gaussian FM với GMM source sinh từ nhiều init strategy, cùng budget FM.
- `scripts/run_toy_moe2_fm_ablation.py`: runner full-pipeline theo kiểu `moe2`: fit diagonal GMM, distill router `q_phi(k|x)` từ `q_GMM(k|x)`, sinh TIDE source top-k từ router, rồi train FM MLP có conditioning `mu_tide/sigma_tide`.
- `configs/toy_moe2_fm_complex_init_grid.json`: mesh GPU mới cho full pipeline, gồm toy 2D khó hơn và dataset ảnh PCA như `mnist`, `fashion_mnist`, `cifar10`.
- `scripts/submit_toy_moe2_fm_jobs.py`: submit mỗi job trong mesh trên thành một notebook Kaggle GPU riêng, dùng credential local theo account nhưng không ghi secret vào notebook.
- `scripts/create_toy_gmm_fm_notebook.py`, `scripts/create_toy_gmm_big_ablation_notebook.py`, `scripts/create_toy_fm_gpu_ablation_notebook.py`: script sinh lại ba notebook trên để thay đổi mesh/toy dễ hơn thay vì sửa notebook thủ công.

## Cách Sinh Lại Notebook

```bash
python scripts/create_toy_gmm_fm_notebook.py
python scripts/create_toy_gmm_big_ablation_notebook.py
python scripts/create_toy_fm_gpu_ablation_notebook.py
```

Các notebook là self-contained, không phụ thuộc `data_prep.py` hay pipeline CelebA. Vì vậy có thể chạy nhanh trên CPU/GPU Kaggle hoặc local để lấy insight trước khi sửa grid thật.

## Chạy Full Pipeline MOE2 Toy

```bash
./.venv/bin/python scripts/run_toy_moe2_fm_ablation.py \
  --datasets aniso_blobs,nested_rings,pinwheel \
  --gmm-modes 16 \
  --topk 2 \
  --router-steps 400 \
  --fm-steps 800 \
  --out-dir toy_moe2_outputs
```

Runner này tạo một Gaussian baseline và các TIDE runs cho nhiều cách khởi tạo:

- `kpp_r3`: k-means++ với 3 restart.
- `kpp_lw5`: k-means++ rồi Lloyd warmup 5 vòng.
- `farthest_lw5`: farthest-point rồi Lloyd warmup.
- `pca_lw5`: PCA/subspace k-means++ rồi Lloyd warmup.
- `split_lw5`: split initialization rồi Lloyd warmup.
- `hybrid_lw5`: nửa đầu k-means++, nửa sau farthest-point trên khoảng cách còn lại, rồi Lloyd warmup.
- `quantilepca_lw5`: chọn seed theo các quantile trên trục PC1, rồi Lloyd warmup.

Metric chính:

- `gmm_valid_nll`, `gmm_dead`, `gmm_count_ratio`, `gmm_overlap_max`: chất lượng GMM.
- `router_valid_kl`, `router_top1_agreement`, `router_usage_entropy_norm`: chất lượng distill router.
- `fm_valid_mse`, `rollout_swd`: downstream FM toy.
- `x0_x1_mag_ratio`, `source_to_target_dist`, `target_vector_var_trace`: hình học source/target.

## Output

- `toy-gmm-fm-insight.ipynb` ghi vào `toy_outputs/`.
- `toy-gmm-big-ablation.ipynb` ghi vào `toy_big_outputs/`.
- `toy-fm-gpu-ablation.ipynb` ghi vào `toy_fm_outputs/`.
- `scripts/run_toy_moe2_fm_ablation.py` ghi vào `toy_moe2_outputs/`.
- Kaggle jobs từ `scripts/submit_toy_moe2_fm_jobs.py` ghi vào `/kaggle/working/toy_moe2_fm/<run_name>/`.

Các thư mục này đã được ignore trong `.gitignore`. Notebook khi chạy xong tự tạo notebook executed rút gọn, CSV/JSON summary và plot PNG để tải về từ Kaggle output.

## Mesh GPU Toy/Image Full-FM 2026-05-24

Mục tiêu của mesh này là không chỉ test chất lượng seed GMM, mà còn chạy hết downstream `GMM -> router distill -> TIDE source -> FM MLP` để xem init nào thực sự giúp vector field/rollout.

Grid chính:

- `toyfm-init-2d-complex-k16`: `nested_rings,pinwheel,checkerboard,spiral_blobs`, K=16.
- `toyfm-init-2d-complex-k32`: cùng bộ 2D, K=32.
- `toyfm-init-mnist-pca32-k32`: MNIST, PCA 32, K=32.
- `toyfm-init-fashion-pca32-k32`: Fashion-MNIST, PCA 32, K=32.
- `toyfm-init-mnist-fashion-pca64-k64`: MNIST + Fashion-MNIST, PCA 64, K=64.
- `toyfm-init-cifar10-pca48-k64`: CIFAR-10, PCA 48, K=64.

Init được so sánh trong từng job:

- `kpp_r5`
- `kpp_lw8`
- `farthest_lw8`
- `pca_lw8`
- `split_lw8`
- `hybrid_lw8`
- `quantilepca_lw8`

Submit GPU:

```bash
./.venv/bin/python scripts/submit_toy_moe2_fm_jobs.py \
  --grid-config configs/toy_moe2_fm_complex_init_grid.json \
  --owners all \
  --exclude-owners kieutung \
  --accelerator gpu \
  --report-path reports/toy_moe2_fm_complex_init_submit.json
```

Đọc kết quả cần ưu tiên song song:

- `fm_valid_mse`: field có dễ học hơn không.
- `rollout_swd`: rollout distribution có thật sự gần data không.
- `router_valid_kl` và `router_top1_agreement`: router có distill được posterior GMM không.
- `gmm_dead`, `gmm_count_ratio`, `gmm_overlap_max`: init có tạo cụm chết/đè cụm không.
- `x0_x1_mag_ratio`, `source_to_target_dist`, `target_vector_var_trace`: source TIDE có làm bài toán quá ngắn/quá mờ không.

## Insight Cần Lấy Trước Khi Áp Vào `moe2`

- Nếu GMM chỉ tốt hơn Gaussian trên blobs rõ cụm nhưng kém trên rings/pinwheel, nghĩa là diagonal GMM đang mô hình hóa sai manifold cong. Khi đó không nên kỳ vọng top-k rộng giúp CelebA nếu latent có cấu trúc phi tuyến tương tự.
- Nếu top-k/weighted mixture giảm khoảng cách source-target nhưng tăng rollout SWD hoặc valid MSE, source đang bị mờ giữa nhiều mode. Điều này khớp với các run CelebA nơi topk lớn không thắng top2.
- Nếu init strategy làm NLL tốt hơn nhưng proxy FM hoặc rollout xấu hơn, không nên chọn source chỉ theo likelihood. Cần giữ thêm count balance, overlap, floor hit, vector variance và downstream FM loss/FID.
- Nếu toy FM cho thấy GMM source tốt hơn Gaussian chỉ khi router/source chọn đúng local mode, hướng đáng thử trên `moe2` là routing gradient relax hoặc temperature/top-k nhỏ, không phải tăng top-k rộng.

## Kết Quả Quick Sweep 2026-05-23

Lệnh đã chạy local CPU:

```bash
./.venv/bin/python -u scripts/run_toy_moe2_fm_ablation.py \
  --datasets aniso_blobs,nested_rings \
  --n-train 1024 \
  --n-valid 512 \
  --gmm-modes 12 \
  --gmm-iters 20 \
  --router-steps 80 \
  --fm-steps 180 \
  --batch-size 128 \
  --hidden 48 \
  --eval-batches 4 \
  --rollout-samples 512 \
  --out-dir toy_moe2_outputs/quick_20260523
```

Output:

- `toy_moe2_outputs/quick_20260523/toy_moe2_fm_summary.md`
- `toy_moe2_outputs/quick_20260523/toy_moe2_fm_summary.csv`
- `toy_moe2_outputs/quick_20260523/toy_moe2_fm_summary.json`
- `toy_moe2_outputs/quick_20260523/toy_moe2_fm_summary.png`
- `toy_moe2_outputs/quick_20260523/toy_moe2_fm_rollouts.png`

Tóm tắt theo `fm_valid_mse`:

| dataset | best GMM init | best GMM valid MSE | Gaussian valid MSE | Ghi chú |
|---|---|---:|---:|---|
| `aniso_blobs` | `kpp_r3` | 2.191 | 9.271 | GMM/TIDE làm bài toán vector field dễ hơn rất rõ. |
| `nested_rings` | `farthest_lw5` | 1.695 | 4.694 | GMM/TIDE cũng giảm MSE, nhưng GMM có dead components và count ratio rất xấu. |

Tóm tắt theo rollout sliced-W2:

| dataset | best rollout | rollout SWD | Gaussian SWD | Ghi chú |
|---|---|---:|---:|---|
| `aniso_blobs` | `gaussian` | 0.331 | 0.331 | Dù Gaussian MSE cao, rollout distribution lại tốt nhất ở budget nhỏ này. |
| `nested_rings` | `kpp_lw5` | 0.037 | 0.057 | GMM/TIDE có lợi nhẹ cho rollout trên ring. |

Insight chính:

- Toy full-pipe xác nhận một điểm quan trọng: **MSE thấp hơn không luôn đồng nghĩa rollout tốt hơn**. Trên `aniso_blobs`, các GMM/TIDE init giảm `fm_valid_mse` từ 9.27 xuống khoảng 2.19-2.42, nhưng rollout SWD vẫn kém Gaussian.
- NLL cũng không đủ để chọn init. Trên `aniso_blobs`, `pca_lw5` có NLL tốt nhất `2.718`, nhưng `kpp_r3` có valid MSE tốt nhất, còn `farthest_lw5` có rollout tốt nhất trong nhóm GMM.
- `nested_rings` cho thấy diagonal GMM có thể tạo source giúp FM dễ hơn, nhưng cụm bị mất cân bằng mạnh: các GMM runs có `dead=1-2`, `count_ratio=70-120`. Đây là cảnh báo rõ cho CelebA latent: nếu metric GMM báo dead/count-ratio xấu, downstream có thể vẫn giảm MSE nhưng source không chắc tốt.
- `farthest_lw5` đáng chú ý: trên `aniso_blobs` rollout tốt nhất trong nhóm GMM, và trên `nested_rings` valid MSE tốt nhất. Nó không có NLL tốt nhất, nên nếu chỉ rank bằng likelihood sẽ bỏ lỡ ứng viên này.
- `kpp_lw5` trên `nested_rings` có rollout tốt nhất và router KL thấp, nhưng valid MSE không thấp nhất. Đây là ví dụ khác cho thấy cần rank nhiều metric cùng lúc.

Kết luận tạm thời:

- Với `moe2`, không nên chọn GMM init chỉ bằng `valid_nll`.
- Cần đưa `farthest + Lloyd` và `kmeans++ + Lloyd` vào shortlist nếu muốn thử lại CelebA/TIDE.
- Khi đọc kết quả thật, ưu tiên bảng nhiều tiêu chí: FID/rollout proxy, FM variance, router KL/top1, count ratio/dead, overlap, chứ không dùng một metric đơn.

## Áp Dụng Vào Pipeline Thật

`moe2` đã có option init strategy cho `data_prep.py`:

- `--gmm_init_strategy auto|random|kmeans++|farthest|pca|split`
- `--gmm_init_warmup_iters N` để chạy Lloyd/k-means refinement sau khi seed mean.
- `--gmm_init_pca_dims` và `--gmm_init_pca_max_samples` cho riêng `pca`.

Default `auto` giữ nguyên hành vi cũ, nên các grid cũ vẫn chạy như trước. Grid [configs/gmm_tide_fm_farthest_lloyd2_grid.json](configs/gmm_tide_fm_farthest_lloyd2_grid.json) là bước chuyển trực tiếp từ toy sang CelebA: `farthest + Lloyd 5` với một nguồn K16 soft variance và một nguồn K32 hard floor.
