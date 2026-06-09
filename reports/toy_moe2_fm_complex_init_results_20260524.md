# Toy MOE2 FM Complex Init Results 2026-05-24

- Output root: `outputs/kaggle/toy_moe2_fm_complex_init_20260524`
- Summary CSV files: 6
- Total rows: 104
- All 6 Kaggle notebooks completed.

## Per Dataset/Job

| job | dataset | K | pca | gaussian MSE | best GMM MSE | ratio | best MSE init | gaussian SWD | best GMM SWD | ratio | best SWD init | dead(MSE/SWD) |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|
| toyfm-init-2d-complex-k16 | checkerboard | 16 | 2 | 1.5410 | 0.4684 | 0.304 | tide_hybrid_lw8 | 0.0091 | 0.0283 | 3.115 | tide_split_lw8 | 0/0 |
| toyfm-init-2d-complex-k16 | nested_rings | 16 | 2 | 1.6009 | 0.6394 | 0.399 | tide_kpp_r5 | 0.0145 | 0.0077 | 0.530 | tide_quantilepca_lw8 | 2/1 |
| toyfm-init-2d-complex-k16 | pinwheel | 16 | 2 | 1.6964 | 0.7298 | 0.430 | tide_kpp_r5 | 0.0186 | 0.0126 | 0.681 | tide_kpp_lw8 | 0/0 |
| toyfm-init-2d-complex-k16 | spiral_blobs | 16 | 2 | 1.5175 | 0.5256 | 0.346 | tide_hybrid_lw8 | 0.0111 | 0.0071 | 0.639 | tide_split_lw8 | 0/0 |
| toyfm-init-2d-complex-k32 | checkerboard | 32 | 2 | 1.5053 | 0.1706 | 0.113 | tide_split_lw8 | 0.0106 | 0.1236 | 11.673 | tide_kpp_r5 | 4/7 |
| toyfm-init-2d-complex-k32 | nested_rings | 32 | 2 | 1.6073 | 0.4810 | 0.299 | tide_quantilepca_lw8 | 0.0090 | 0.0094 | 1.045 | tide_split_lw8 | 8/8 |
| toyfm-init-2d-complex-k32 | pinwheel | 32 | 2 | 1.4910 | 0.6130 | 0.411 | tide_hybrid_lw8 | 0.0115 | 0.0124 | 1.074 | tide_pca_lw8 | 3/2 |
| toyfm-init-2d-complex-k32 | spiral_blobs | 32 | 2 | 1.5250 | 0.4935 | 0.324 | tide_kpp_lw8 | 0.0135 | 0.0078 | 0.578 | tide_hybrid_lw8 | 1/3 |
| toyfm-init-cifar10-pca48-k64 | cifar10 | 64 | 48 | 2.0535 | 1.3202 | 0.643 | tide_split_lw8 | 0.0101 | 0.0232 | 2.290 | tide_kpp_r5 | 2/9 |
| toyfm-init-fashion-pca32-k32 | fashion_mnist | 32 | 32 | 1.8163 | 1.3150 | 0.724 | tide_farthest_lw8 | 0.0169 | 0.0157 | 0.929 | tide_kpp_lw8 | 0/0 |
| toyfm-init-mnist-fashion-pca64-k64 | fashion_mnist | 64 | 64 | 1.9473 | 1.4072 | 0.723 | tide_quantilepca_lw8 | 0.0094 | 0.0123 | 1.305 | tide_kpp_r5 | 0/1 |
| toyfm-init-mnist-fashion-pca64-k64 | mnist | 64 | 64 | 1.9686 | 1.4934 | 0.759 | tide_quantilepca_lw8 | 0.0097 | 0.0150 | 1.557 | tide_quantilepca_lw8 | 0/0 |
| toyfm-init-mnist-pca32-k32 | mnist | 32 | 32 | 1.9353 | 1.4041 | 0.726 | tide_split_lw8 | 0.0099 | 0.0180 | 1.817 | tide_kpp_lw8 | 0/0 |

## Init Aggregate

| init | n | mean MSE ratio | median MSE ratio | mean SWD ratio | median SWD ratio | FM wins | SWD wins | mean router KL | mean dead | mean count ratio | mean overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kpp_r5 | 13 | 0.493 | 0.430 | 2.464 | 1.394 | 2 | 3 | 0.3905 | 2.69 | 250.0 | 0.6924 |
| split_lw8 | 13 | 0.496 | 0.439 | 2.605 | 1.714 | 3 | 3 | 0.3756 | 1.62 | 193.8 | 0.6514 |
| hybrid_lw8 | 13 | 0.497 | 0.414 | 2.764 | 1.693 | 3 | 1 | 0.3569 | 4.62 | 230.7 | 0.6271 |
| quantilepca_lw8 | 13 | 0.497 | 0.460 | 2.697 | 1.361 | 3 | 2 | 0.3754 | 1.54 | 232.6 | 0.6501 |
| farthest_lw8 | 13 | 0.498 | 0.429 | 2.834 | 1.853 | 1 | 0 | 0.3699 | 2.77 | 242.9 | 0.6196 |
| kpp_lw8 | 13 | 0.498 | 0.426 | 2.654 | 1.511 | 1 | 3 | 0.3810 | 1.92 | 231.6 | 0.6529 |
| pca_lw8 | 13 | 0.506 | 0.418 | 2.712 | 1.605 | 0 | 1 | 0.3818 | 1.77 | 244.8 | 0.6479 |

## Top FM MSE Ratios

| job | dataset | init | ratio | MSE | SWD | dead | count ratio | router KL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| toyfm-init-2d-complex-k32 | checkerboard | split_lw8 | 0.113 | 0.1706 | 0.1554 | 4 | 308.0 | 0.4693 |
| toyfm-init-2d-complex-k32 | checkerboard | farthest_lw8 | 0.121 | 0.1825 | 0.1740 | 3 | 226.0 | 0.5089 |
| toyfm-init-2d-complex-k32 | checkerboard | quantilepca_lw8 | 0.124 | 0.1872 | 0.1786 | 3 | 315.0 | 0.4659 |
| toyfm-init-2d-complex-k32 | checkerboard | kpp_r5 | 0.127 | 0.1911 | 0.1236 | 7 | 359.0 | 0.5195 |
| toyfm-init-2d-complex-k32 | checkerboard | kpp_lw8 | 0.140 | 0.2102 | 0.1357 | 5 | 315.0 | 0.5283 |
| toyfm-init-2d-complex-k32 | checkerboard | hybrid_lw8 | 0.143 | 0.2154 | 0.1564 | 8 | 340.0 | 0.4596 |
| toyfm-init-2d-complex-k32 | checkerboard | pca_lw8 | 0.187 | 0.2820 | 0.1562 | 5 | 359.0 | 0.5590 |
| toyfm-init-2d-complex-k32 | nested_rings | quantilepca_lw8 | 0.299 | 0.4810 | 0.0148 | 8 | 463.0 | 0.1837 |
| toyfm-init-2d-complex-k16 | checkerboard | hybrid_lw8 | 0.304 | 0.4684 | 0.0376 | 0 | 36.6 | 0.4865 |
| toyfm-init-2d-complex-k16 | checkerboard | farthest_lw8 | 0.310 | 0.4780 | 0.0323 | 0 | 6.9 | 0.4793 |
| toyfm-init-2d-complex-k16 | checkerboard | kpp_lw8 | 0.319 | 0.4918 | 0.0628 | 0 | 50.4 | 0.5934 |
| toyfm-init-2d-complex-k32 | spiral_blobs | kpp_lw8 | 0.324 | 0.4935 | 0.0084 | 1 | 306.0 | 0.1279 |

## Top Rollout SWD Ratios

| job | dataset | init | ratio | SWD | MSE | dead | count ratio | router KL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| toyfm-init-2d-complex-k16 | nested_rings | quantilepca_lw8 | 0.530 | 0.0077 | 0.6628 | 1 | 525.0 | 0.0373 |
| toyfm-init-2d-complex-k32 | spiral_blobs | hybrid_lw8 | 0.578 | 0.0078 | 0.5038 | 3 | 321.0 | 0.1508 |
| toyfm-init-2d-complex-k16 | nested_rings | split_lw8 | 0.583 | 0.0085 | 0.6547 | 1 | 442.0 | 0.0783 |
| toyfm-init-2d-complex-k32 | spiral_blobs | kpp_lw8 | 0.619 | 0.0084 | 0.4935 | 1 | 306.0 | 0.1279 |
| toyfm-init-2d-complex-k16 | nested_rings | pca_lw8 | 0.626 | 0.0091 | 0.6696 | 1 | 470.0 | 0.0697 |
| toyfm-init-2d-complex-k16 | nested_rings | hybrid_lw8 | 0.627 | 0.0091 | 0.6626 | 0 | 4.7 | 0.0738 |
| toyfm-init-2d-complex-k16 | spiral_blobs | split_lw8 | 0.639 | 0.0071 | 0.5727 | 0 | 8.3 | 0.1097 |
| toyfm-init-2d-complex-k16 | spiral_blobs | hybrid_lw8 | 0.648 | 0.0072 | 0.5256 | 0 | 42.7 | 0.1317 |
| toyfm-init-2d-complex-k16 | pinwheel | kpp_lw8 | 0.681 | 0.0126 | 0.7612 | 0 | 67.4 | 0.0746 |
| toyfm-init-2d-complex-k16 | spiral_blobs | quantilepca_lw8 | 0.708 | 0.0079 | 0.5813 | 1 | 415.0 | 0.1112 |
| toyfm-init-2d-complex-k16 | nested_rings | farthest_lw8 | 0.714 | 0.0104 | 0.6796 | 0 | 5.5 | 0.0519 |
| toyfm-init-2d-complex-k16 | nested_rings | kpp_lw8 | 0.738 | 0.0107 | 0.6421 | 1 | 469.0 | 0.0836 |

## Notes

- Best GMM/TIDE improves FM valid MSE over Gaussian in 13/13 dataset/job groups.
- Best GMM/TIDE improves rollout SWD over Gaussian in 5/13 dataset/job groups.
- Use SWD together with MSE: several GMM sources make the supervised vector field easier but do not necessarily improve rollout distribution.
- Dead components are common on ring/checkerboard-style data and should not be ignored even when FM MSE improves.
