# Toy MOE2 FM Source Sweep Results 2026-05-24

- Output root: `outputs/kaggle/toy_moe2_fm_source_20260524`
- Flat CSV: `reports/toy_moe2_fm_source_results_20260524.csv`
- Plot: `reports/toy_moe2_fm_source_ratios_20260524.png`
- Summary CSV files: 7
- Total rows: 111

## Per Dataset Best

| job | dataset | K | pca | Gaussian MSE | best MSE | ratio | best MSE variant | Gaussian SWD | best SWD | ratio | best SWD variant |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| toyfm-cifar-source-pca48-k64 | cifar10 | 64 | 48 | 2.0228 | 1.0331 | 0.511 | lr1e4_beta31 | 0.0108 | 0.0222 | 2.053 | t1_tau10 |
| toyfm-retune-2d-k16 | nested_rings | 16 | 2 | 1.4844 | 0.3805 | 0.256 | lr1e4_beta31 | 0.0155 | 0.0078 | 0.504 | lr3e4_uniform |
| toyfm-retune-2d-k16 | pinwheel | 16 | 2 | 1.5805 | 0.4547 | 0.288 | lr1e4_beta31 | 0.0130 | 0.0119 | 0.912 | lr3e4_uniform |
| toyfm-retune-2d-k16 | spiral_blobs | 16 | 2 | 1.5924 | 0.2969 | 0.186 | lr1e4_beta31 | 0.0108 | 0.0095 | 0.880 | lr3e4_uniform |
| toyfm-retune-image-pca32-k32 | fashion_mnist | 32 | 32 | 1.8215 | 1.0159 | 0.558 | lr1e4_beta31 | 0.0185 | 0.0119 | 0.644 | lr1e4_uniform |
| toyfm-retune-image-pca32-k32 | mnist | 32 | 32 | 1.9367 | 1.0717 | 0.553 | lr1e4_beta31 | 0.0096 | 0.0107 | 1.116 | lr5e5_uniform |
| toyfm-router-2d-k16 | nested_rings | 16 | 2 | 1.5315 | 0.6108 | 0.399 | uniform_hybrid_t2 | 0.0077 | 0.0085 | 1.108 | direct_hybrid |
| toyfm-router-2d-k16 | pinwheel | 16 | 2 | 1.6370 | 0.7865 | 0.480 | uniform_hybrid_t2 | 0.0163 | 0.0085 | 0.520 | oracle_hybrid_t2 |
| toyfm-router-2d-k16 | spiral_blobs | 16 | 2 | 1.5187 | 0.4621 | 0.304 | uniform_hybrid_t2 | 0.0070 | 0.0070 | 0.996 | distill_hybrid_t2 |
| toyfm-router-image-pca32-k32 | fashion_mnist | 32 | 32 | 1.7511 | 0.8378 | 0.478 | uniform_split_t2 | 0.0202 | 0.0158 | 0.783 | nearest_split_t2 |
| toyfm-router-image-pca32-k32 | mnist | 32 | 32 | 1.9293 | 0.7879 | 0.408 | uniform_split_t2 | 0.0112 | 0.0160 | 1.432 | oracle_split_t2 |
| toyfm-topk-temp-2d-k16 | nested_rings | 16 | 2 | 1.5610 | 0.4747 | 0.304 | t8_tau10 | 0.0061 | 0.0076 | 1.259 | t4_tau10 |
| toyfm-topk-temp-2d-k16 | pinwheel | 16 | 2 | 1.5608 | 0.5674 | 0.364 | t8_tau10 | 0.0170 | 0.0107 | 0.626 | t2_tau10 |
| toyfm-topk-temp-2d-k16 | spiral_blobs | 16 | 2 | 1.6092 | 0.4394 | 0.273 | t8_tau10 | 0.0069 | 0.0061 | 0.883 | t2_tau10 |
| toyfm-topk-temp-image-pca32-k32 | fashion_mnist | 32 | 32 | 1.8205 | 1.2682 | 0.697 | t2_tau15 | 0.0167 | 0.0144 | 0.861 | t2_tau05 |
| toyfm-topk-temp-image-pca32-k32 | mnist | 32 | 32 | 1.9215 | 1.3194 | 0.687 | t8_tau10 | 0.0118 | 0.0160 | 1.362 | t2_tau05 |

## Source Mode Aggregate

| key | n | mean MSE ratio | median MSE ratio | mean SWD ratio | median SWD ratio | MSE wins | SWD wins | mean router KL | mean dead | mean count ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 5 | 0.414 | 0.408 | 1.363 | 1.238 | 5 | 0 | 0.1435 | 0.60 | 150.3 |
| distilled | 73 | 0.569 | 0.553 | 3.517 | 1.454 | 11 | 12 | 0.2243 | 0.53 | 148.3 |
| oracle | 6 | 0.627 | 0.660 | 1.646 | 1.403 | 0 | 2 | 0.3580 | 0.83 | 172.8 |
| nearest | 5 | 0.691 | 0.796 | 1.354 | 1.170 | 0 | 1 | 0.1435 | 0.60 | 150.3 |
| direct | 6 | 0.715 | 0.755 | 1.456 | 1.109 | 0 | 1 | 0.3580 | 0.83 | 172.8 |

## Top-k Aggregate (distilled only)

| key | n | mean MSE ratio | median MSE ratio | mean SWD ratio | median SWD ratio | MSE wins | SWD wins | mean router KL | mean dead | mean count ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| topk=8 | 5 | 0.466 | 0.364 | 1.553 | 1.477 | 4 | 0 | 0.1394 | 0.40 | 162.0 |
| topk=4 | 6 | 0.500 | 0.468 | 1.710 | 1.391 | 0 | 1 | 0.3546 | 0.67 | 182.5 |
| topk=2 | 53 | 0.572 | 0.543 | 4.271 | 1.469 | 7 | 10 | 0.2135 | 0.51 | 132.7 |
| topk=1 | 9 | 0.654 | 0.634 | 1.371 | 1.273 | 0 | 1 | 0.2486 | 0.67 | 210.0 |

## Temperature Aggregate (distilled only)

| key | n | mean MSE ratio | median MSE ratio | mean SWD ratio | median SWD ratio | MSE wins | SWD wins | mean router KL | mean dead | mean count ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tau=1.5 | 5 | 0.540 | 0.491 | 1.277 | 1.082 | 1 | 0 | 0.1394 | 0.40 | 162.0 |
| tau=1.0 | 60 | 0.570 | 0.554 | 3.986 | 1.494 | 10 | 10 | 0.2479 | 0.55 | 140.2 |
| tau=0.5 | 8 | 0.578 | 0.550 | 1.396 | 1.317 | 0 | 2 | 0.1008 | 0.50 | 200.6 |

## Time Sampling Aggregate (distilled only)

| key | n | mean MSE ratio | median MSE ratio | mean SWD ratio | median SWD ratio | MSE wins | SWD wins | mean router KL | mean dead | mean count ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| beta31 | 6 | 0.392 | 0.399 | 18.332 | 16.407 | 6 | 0 | 0.3555 | 0.67 | 130.7 |
| uniform | 56 | 0.570 | 0.539 | 1.343 | 1.099 | 5 | 12 | 0.2037 | 0.52 | 156.4 |
| beta22 | 5 | 0.574 | 0.526 | 2.323 | 2.431 | 0 | 0 | 0.1405 | 0.40 | 99.8 |
| beta13 | 6 | 0.732 | 0.733 | 9.990 | 8.117 | 0 | 0 | 0.3555 | 0.67 | 130.7 |

## FM LR Aggregate (distilled only)

| key | n | mean MSE ratio | median MSE ratio | mean SWD ratio | median SWD ratio | MSE wins | SWD wins | mean router KL | mean dead | mean count ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lr=0.0003 | 46 | 0.561 | 0.529 | 1.398 | 1.171 | 5 | 10 | 0.2174 | 0.54 | 168.7 |
| lr=0.0001 | 22 | 0.572 | 0.556 | 8.484 | 3.887 | 6 | 1 | 0.2578 | 0.55 | 116.7 |
| lr=5e-05 | 5 | 0.623 | 0.554 | 1.154 | 1.116 | 0 | 1 | 0.1405 | 0.40 | 99.8 |

## Most Useful Individual Variants

| job | dataset | variant | source | topk | tau | t | lr | MSE ratio | SWD ratio | MSE | SWD |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| toyfm-retune-2d-k16 | spiral_blobs | lr1e4_beta31 | distilled | 2 | 1.00 | beta31 | 0.00010 | 0.186 | 5.807 | 0.2969 | 0.0625 |
| toyfm-retune-2d-k16 | nested_rings | lr1e4_beta31 | distilled | 2 | 1.00 | beta31 | 0.00010 | 0.256 | 6.469 | 0.3805 | 0.1005 |
| toyfm-topk-temp-2d-k16 | spiral_blobs | t8_tau10 | distilled | 8 | 1.00 | uniform | 0.00030 | 0.273 | 2.599 | 0.4394 | 0.0179 |
| toyfm-retune-2d-k16 | pinwheel | lr1e4_beta31 | distilled | 2 | 1.00 | beta31 | 0.00010 | 0.288 | 14.502 | 0.4547 | 0.1891 |
| toyfm-topk-temp-2d-k16 | nested_rings | t8_tau10 | distilled | 8 | 1.00 | uniform | 0.00030 | 0.304 | 1.477 | 0.4747 | 0.0089 |
| toyfm-router-2d-k16 | spiral_blobs | uniform_hybrid_t2 | uniform | 2 | 1.00 | uniform | 0.00030 | 0.304 | 1.238 | 0.4621 | 0.0087 |
| toyfm-topk-temp-2d-k16 | spiral_blobs | t4_tau10 | distilled | 4 | 1.00 | uniform | 0.00030 | 0.317 | 1.522 | 0.5094 | 0.0105 |
| toyfm-topk-temp-2d-k16 | nested_rings | t4_tau10 | distilled | 4 | 1.00 | uniform | 0.00030 | 0.333 | 1.259 | 0.5204 | 0.0076 |
| toyfm-topk-temp-2d-k16 | spiral_blobs | t2_tau10 | distilled | 2 | 1.00 | uniform | 0.00030 | 0.342 | 0.883 | 0.5507 | 0.0061 |
| toyfm-retune-2d-k16 | spiral_blobs | lr1e4_beta22 | distilled | 2 | 1.00 | beta22 | 0.00010 | 0.343 | 2.431 | 0.5468 | 0.0262 |
| toyfm-router-2d-k16 | spiral_blobs | distill_hybrid_t2 | distilled | 2 | 1.00 | uniform | 0.00030 | 0.347 | 0.996 | 0.5266 | 0.0070 |
| toyfm-topk-temp-2d-k16 | spiral_blobs | t2_tau15 | distilled | 2 | 1.50 | uniform | 0.00030 | 0.361 | 1.082 | 0.5815 | 0.0075 |
| toyfm-topk-temp-2d-k16 | pinwheel | t8_tau10 | distilled | 8 | 1.00 | uniform | 0.00030 | 0.364 | 0.720 | 0.5674 | 0.0123 |
| toyfm-router-2d-k16 | spiral_blobs | oracle_hybrid_t2 | oracle | 2 | 1.00 | uniform | 0.00030 | 0.364 | 2.447 | 0.5530 | 0.0172 |
| toyfm-retune-2d-k16 | spiral_blobs | lr3e4_uniform | distilled | 2 | 1.00 | uniform | 0.00030 | 0.374 | 0.880 | 0.5961 | 0.0095 |
| toyfm-topk-temp-2d-k16 | spiral_blobs | t2_tau05 | distilled | 2 | 0.50 | uniform | 0.00030 | 0.381 | 1.462 | 0.6127 | 0.0101 |

## Notes

- Best non-Gaussian source improves FM valid MSE in 16/16 job/dataset groups.
- Best non-Gaussian source improves rollout SWD in 10/16 job/dataset groups.
- Source mode ranking by mean MSE ratio starts with: uniform MSE 0.414/SWD 1.363, distilled MSE 0.569/SWD 3.517, oracle MSE 0.627/SWD 1.646, nearest MSE 0.691/SWD 1.354.
- Time-sampling aggregate: beta31 MSE 0.392/SWD 18.332, uniform MSE 0.570/SWD 1.343, beta22 MSE 0.574/SWD 2.323, beta13 MSE 0.732/SWD 9.990.
- Ratios are always normalized to the Gaussian row in the same notebook and dataset, so they are comparable despite different seeds/jobs.
