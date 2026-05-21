# Ablation GMM

Tài liệu này tách riêng các thí nghiệm chỉ liên quan đến GMM source. Mục tiêu là đánh giá chất lượng GMM trước khi dùng nó làm source cho FM/TIDE, vì downstream FID đang cho thấy source tốt theo likelihood chưa chắc đã tạo đường flow tốt.

## Hiện trạng

- EM25 trên CelebA-HQ latent chưa cho tín hiệu đủ mạnh rằng GMM source luôn tốt hơn Gaussian source.
- EM100 cải thiện likelihood ở một số cấu hình, nhưng cải thiện likelihood đơn thuần không đủ để kết luận nên rerun FM. Cần nhìn thêm dead component, cân bằng cụm, overlap, floor hit và downstream FID.
- Toy notebook cho thấy GMM hữu ích khi dữ liệu thật sự có cụm rõ và source được chọn đúng mode, nhưng top-k/weighted mixture quá rộng có thể làm source bị mờ và kéo điểm vào giữa các mode.
- Vì vậy hướng tiếp theo nên tập trung vào chất lượng khởi tạo và hội tụ của GMM, trước khi thêm nhiều biến FM.

## 5 hướng khởi tạo mới

Các hướng này đã được đưa vào `data_prep.py` qua flag `--gmm_init_strategy`, `--gmm_init_warmup_iters`, `--gmm_init_pca_dims`, `--gmm_init_pca_max_samples`, và vẫn giữ mặc định cũ.

### 1. Multi-restart k-means++

Ý tưởng: chạy nhiều restart EM từ k-means++ khác nhau, chọn restart có final train NLL thấp nhất.

Flag chính:

```bash
--gmm_init_strategy kmeans++ \
--gmm_em_restarts 4
```

Kỳ vọng:

- Ít rủi ro nhất vì vẫn dùng cùng thuật toán khởi tạo cũ.
- Tốn thời gian tuyến tính theo số restart.
- Dùng làm baseline mạnh hơn cho EM25/EM100.

Metric cần nhìn:

- `valid_nll`, `latent_valid_nll`
- `train_valid_nll_gap`
- `valid_dead_components`
- `valid_count_ratio`
- `overlap_proxy_max`

### 2. Lloyd/k-means warmup

Ý tưởng: sau khi chọn center ban đầu, chạy vài vòng hard k-means trước EM để center đi vào vùng mật độ cao hơn, rồi lấy hard assignment ban đầu để khởi tạo `pi`, `mu`, `var`.

Flag chính:

```bash
--gmm_init_strategy kmeans++ \
--gmm_init_warmup_iters 5
```

Kỳ vọng:

- Giảm biến động do center ban đầu nằm lệch.
- Có thể giảm dead component sớm.
- Có thể làm component quá cứng nếu dữ liệu không thật sự cụm rõ.

Metric cần nhìn:

- `em_nll_start`: nếu giảm mạnh so với k-means++ thường thì warmup giúp khởi điểm tốt hơn.
- `nll_delta_last10`: nếu còn lớn, EM vẫn chưa hội tụ.
- `var_floor_hit_rate`: nếu tăng quá cao, warmup có thể tạo cụm hẹp rồi bị floor ép.

### 3. Split initialization

Ý tưởng: khởi tạo ít center hơn, sau đó tách cụm có score lớn nhất theo `count * variance_mean`. Cụm rộng/dày sẽ được chia làm hai theo chiều có variance lớn nhất.

Flag chính:

```bash
--gmm_init_strategy split \
--gmm_init_warmup_iters 5
```

Kỳ vọng:

- Hữu ích khi một mode lớn bị k-means++ bỏ sót cấu trúc con.
- Có thể tạo coverage đều hơn trong vùng mật độ lớn.
- Rủi ro là tách theo trục diagonal có thể không khớp cấu trúc cong/xiên của latent.

Metric cần nhìn:

- `valid_count_ratio`: giảm là tốt.
- `pi_entropy_normalized`: không nên giảm nhiều.
- `overlap_proxy_max`: không nên tăng mạnh.
- `component_variance_mean` và `var_floor_hit_rate`: tránh collapse thành nhiều cụm quá nhỏ.

### 4. PCA/subspace-aware initialization

Ý tưởng: ước lượng PCA basis từ sample nhỏ, chọn k-means++ trong không gian PCA thấp chiều, rồi lấy center gốc tương ứng trong latent space.

Flag chính:

```bash
--gmm_init_strategy pca \
--gmm_init_pca_dims 16 \
--gmm_init_pca_max_samples 2048 \
--gmm_init_warmup_iters 5
```

Kỳ vọng:

- Giảm nhiễu từ chiều latent quá lớn.
- Có thể tốt hơn nếu khác biệt mode nằm trong vài hướng chính.
- Rủi ro là bỏ mất mode hiếm nằm ở subspace không chiếm nhiều variance.

Metric cần nhìn:

- `valid_count_min`, `valid_dead_components`: mode hiếm có bị mất không.
- `pi_entropy_normalized`: có tụ về vài mode lớn không.
- `latent_valid_nll`: so sánh trong latent space nếu chạy chuẩn hóa.

### 5. Farthest-point coverage initialization

Ý tưởng: chọn center đầu tiên ngẫu nhiên, sau đó luôn chọn điểm xa nhất so với các center hiện tại. Đây là khởi tạo thiên về phủ không gian hơn likelihood.

Flag chính:

```bash
--gmm_init_strategy farthest \
--gmm_init_warmup_iters 5
```

Kỳ vọng:

- Có thể giảm dead component và tăng coverage.
- Hữu ích khi k-means++ vẫn tập trung quá nhiều vào cụm lớn.
- Rủi ro là chọn outlier làm center, NLL xấu hơn.

Metric cần nhìn:

- `center_distance_min`, `center_distance_mean`: coverage center có tăng không.
- `valid_count_ratio`: có cân bằng hơn không.
- `valid_nll`: không nên xấu quá nhiều.
- `overlap_proxy_max`: giảm là tốt, nhưng giảm vì center chạy ra outlier thì không tốt.

## Grid thử nghiệm

File `configs/gmm_init_ablation_grid.json` chứa 20 cấu hình GMM-only:

- 4 nguồn GMM đại diện:
  - `K=16`, Dirichlet `512`, no coverage.
  - `K=16`, KL pi `512`, no coverage.
  - `K=32`, no pi prior, hard floor variance `0.5`.
  - `K=32`, KL pi `512`, no coverage.
- 5 hướng khởi tạo:
  - `kmeans++` với `gmm_em_restarts=4`.
  - `kmeans++` với `gmm_init_warmup_iters=5`.
  - `farthest` với Lloyd warmup 5.
  - `pca` với 16 PCA dims và Lloyd warmup 5.
  - `split` với Lloyd warmup 5.

Submit bằng queue manager:

```bash
python scripts/manage_gmm_ablation_queue.py \
  --queue-path reports/gmm_init_ablation_queue_20260521.json \
  --grid-config configs/gmm_init_ablation_grid.json \
  --accounts-file .secrets/all-kaggle.json \
  --owners all \
  --exclude-owners kieutung \
  --accelerator tpu \
  --sync-status \
  --push \
  --batch-size 5 \
  --max-submit-per-owner 1
```

Thu log:

```bash
python scripts/collect_gmm_ablation_results.py \
  --submit-report reports/gmm_init_ablation_queue_20260521.json \
  --grid-config configs/gmm_init_ablation_grid.json \
  --accounts-file .secrets/all-kaggle.json \
  --output-root outputs/kaggle/gmm_init_ablation_20260521 \
  --report-path reports/gmm_init_ablation_results_20260521.json \
  --download-statuses COMPLETE,CANCEL_ACKNOWLEDGED
```

## Tiêu chí chọn shortlist

Không chọn theo NLL một mình. Một cấu hình GMM nên lọt shortlist nếu:

- `valid_dead_components = 0`.
- `pi_entropy_normalized >= 0.90`, hoặc ít nhất không giảm đáng kể so với baseline cùng K/prior/coverage.
- `valid_count_ratio` không tăng quá mạnh.
- `overlap_proxy_max` không tăng.
- `var_floor_hit_rate` không cho thấy nhiều component bị ép floor hàng loạt.
- `latent_valid_nll` hoặc `valid_nll` tốt hơn baseline cùng nhóm, nhưng không đánh đổi bằng collapse.

## Khi nào mới rerun FM

Chỉ nên rerun FM nếu init mới cải thiện đồng thời cả hai nhóm:

- Nhóm GMM likelihood/convergence: `valid_nll`, `train_valid_nll_gap`, `nll_delta_last10`.
- Nhóm source quality: dead component, count ratio, pi entropy, overlap proxy, floor hit.

Nếu chỉ có NLL tốt hơn nhưng count ratio xấu hơn, overlap tăng, hoặc variance bị floor ép nhiều hơn, nên coi đó là cải thiện likelihood nội bộ của GMM chứ chưa phải source tốt hơn cho FM.

