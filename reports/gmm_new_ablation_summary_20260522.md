# GMM New Ablation Summary 20260522

## Inputs

- gmm_fixed_report: `reports/gmm_init_ablation_fixed_results_20260522.json`
- toy_big_csv: `outputs/kaggle/toy_ablation_20260522/kieutung-toy-gmm-big-ablation-kieutung-cpu-20260521-1713/toy_big_outputs/toy_big_metrics.csv`
- toy_fm_old_csv: `outputs/kaggle/toy_ablation_20260522/no1ceboy-toy-fm-gpu-ablation-no1ceboy-gpu-20260521-1713/toy_fm_outputs/toy_fm_summary.csv`
- toy_fm_init_sweep_csv: `outputs/kaggle/toy_ablation_20260522/victorharvey27-toy-fm-init-0522-0049-victorharvey27/toy_fm_outputs/toy_fm_summary.csv`

## CelebA Latent GMM Init Fixed

Init flags were verified in `gmm_metrics.json`: 4 rows each for kmeans++ r4, kmeans++ lw5, farthest lw5, pca lw5, split lw5.

### Init Aggregate

| init | mean latent valid NLL | total valid dead | mean count ratio | mean pi entropy |
|---|---:|---:|---:|---:|
| kmeans++ lw5 | 4218.41 | 0 | 2.66 | 0.9915 |
| farthest lw5 | 4219.75 | 0 | 2.57 | 0.9921 |
| kmeans++ r4 | 4219.82 | 0 | 2.47 | 0.9936 |
| pca lw5 | 4220.00 | 1 | 52.75 | 0.9910 |
| split lw5 | 4271.74 | 33 | 486.25 | 0.8403 |

### Ranking Per Source Group

| group | best by NLL | NLL | dead | ratio | pi_ent | best clean/balanced | NLL | dead | ratio | pi_ent |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| K16 dirichlet512.0 ml-no-coverage | kmeans++ lw5 | 4253.76 | 0 | 2.56 | 0.9926 | kmeans++ lw5 | 4253.76 | 0 | 2.56 | 0.9926 |
| K16 kl512.0 ml-no-coverage | kmeans++ r4 | 4257.07 | 0 | 1.69 | 0.9952 | kmeans++ r4 | 4257.07 | 0 | 1.69 | 0.9952 |
| K32 none0.0 hardv0p5 | kmeans++ lw5 | 4184.48 | 0 | 2.38 | 0.9933 | kmeans++ lw5 | 4184.48 | 0 | 2.38 | 0.9933 |
| K32 kl512.0 ml-no-coverage | kmeans++ lw5 | 4176.77 | 0 | 3.08 | 0.9932 | kmeans++ lw5 | 4176.77 | 0 | 3.08 | 0.9932 |

## Toy GMM Big CPU

| dataset | best NLL config | NLL | dead | ratio | NMI | overlap | best balance config | NLL | dead | ratio | NMI |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| aniso_blobs | gmm_k32_split_lw8 | 2.5495 | 0 | 7.54 | 0.686 | 0.587 | gmm_k8_kpp_r3 | 2.8440 | 0 | 1.58 | 0.913 |
| cross_cov | gmm_k8_pca_lw8 | 1.8841 | 0 | 1557.00 | 0.764 | 0.681 | kmeans_gauss_k8_lw12 | 2.5646 | 0 | 1.49 | 0.568 |
| nested_rings | gmm_k32_kpp_lw8 | 2.8114 | 0 | 8.19 | 0.466 | 0.393 | gmm_k16_split_lw8 | 3.2690 | 0 | 2.23 | 0.344 |
| pinwheel | gmm_k32_kpp_lw8 | 3.3452 | 0 | 60.33 | 0.695 | 0.789 | gmm_k8_far_lw8 | 3.8189 | 0 | 2.44 | 0.462 |

## Toy FM Init Sweep GPU

| dataset | gaussian valid_mse/swd | best hard by MSE | valid_mse | swd | best hard by SWD | valid_mse | swd |
|---|---:|---|---:|---:|---|---:|---:|
| nested_rings | 3.5830/0.01079 | kpp_r3:hard | 0.4025 | 0.00472 | farthest_lw8:hard | 0.4251 | 0.00362 |
| pinwheel | 4.5510/0.07041 | farthest_lw8:hard | 0.9523 | 0.00822 | kpp_lw8:hard | 1.0952 | 0.00705 |

### Top By Validation MSE

#### nested_rings
| rank | source | valid_mse | rollout_swd | dist | gmm_nll |
|---:|---|---:|---:|---:|---:|
| 1 | kpp_r3:top2_mean | 0.023440 | 0.020662 | 0.4887 | 3.1284 |
| 2 | pca_lw8:top2_mean | 0.025064 | 0.014938 | 0.4750 | 3.2295 |
| 3 | kpp_lw8:top2_mean | 0.028410 | 0.016650 | 0.4931 | 3.2399 |
| 4 | farthest_lw8:top2_mean | 0.029716 | 0.020528 | 0.4676 | 3.2046 |
| 5 | split_lw8:top2_mean | 0.032378 | 0.026601 | 0.4877 | 3.2575 |
| 6 | pca_lw8:top4_sample | 0.240646 | 0.005213 | 0.6264 | 3.2295 |

#### pinwheel
| rank | source | valid_mse | rollout_swd | dist | gmm_nll |
|---:|---|---:|---:|---:|---:|
| 1 | kpp_r3:top2_mean | 0.062680 | 0.030293 | 0.8570 | 3.7807 |
| 2 | farthest_lw8:top2_mean | 0.063423 | 0.023921 | 0.7588 | 3.7720 |
| 3 | split_lw8:top2_mean | 0.069525 | 0.047862 | 0.8870 | 3.7946 |
| 4 | kpp_lw8:top2_mean | 0.069986 | 0.044935 | 0.8139 | 3.7770 |
| 5 | pca_lw8:top2_mean | 0.071977 | 0.033060 | 0.8192 | 3.7826 |
| 6 | farthest_lw8:top4_sample | 0.489919 | 0.008199 | 0.9580 | 3.7720 |


## Main Insights

- CelebA fixed run is valid: init metadata differs exactly as intended. The previous non-fixed report remains invalid for init comparison.
- Split initialization is consistently bad on CelebA latent here: high NLL and many dead validation components in every source group.
- Lloyd warmup helps or ties k-means++ on several CelebA groups, especially K16 dirichlet and K32 KL, without dead components.
- Farthest and PCA are competitive in some groups, but PCA creates dead components for K32 hardv0p5 and neither dominates k-means++ warmup.
- Toy GMM shows the same caution: diagonal GMM likelihood can improve while count ratio/overlap remain poor on curved datasets.
- Toy FM init sweep: GMM sources dramatically reduce FM validation MSE versus Gaussian, but the source with lowest vector-field MSE is often top-k mean; rollout SWD can prefer hard sample instead. This means low FM loss alone can be misleading.
- For real FM reruns, the safest next candidates are k-means++ Lloyd warmup or k-means++ multi-restart. Avoid split init for CelebA latent unless fixed with a better split criterion.
