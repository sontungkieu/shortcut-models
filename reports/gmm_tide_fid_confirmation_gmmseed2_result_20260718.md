# GMM-TIDE FID Confirmation: GMM Seed 2

## Material Passport

- Verification status: `ANALYZED`
- Primary metric: `fid/timesteps/128`
- Checkpoint step: `400000`
- Evaluation seeds: `101,202,303,404,505`
- Generations per evaluation seed: `50048`
- Protocol: `configs/gmm_tide_fid_confirmation_gmmseed2_protocol.json`
- Checkpoints downloaded locally: no
- Kaggle status: both evaluation jobs `COMPLETE`
- Retention: diagnostics downloaded; both private log-only evaluation notebooks deleted remotely

## Repeated FID128

| family | source construction | router data | repeats | mean FID128 | sample SD | 95% CI |
|---|---|---|---:|---:|---:|---|
| C0 | weighted | mix | 5 | 7.1015 | 0.0288 | 7.0658--7.1373 |
| C4 | sample-top-k | bridge | 5 | 7.3965 | 0.0426 | 7.3436--7.4494 |

Paired differences use the same evaluation seed for both checkpoints:

| evaluation seed | C0 | C4 | C4 - C0 |
|---:|---:|---:|---:|
| 101 | 7.0512 | 7.3899 | +0.3387 |
| 202 | 7.1118 | 7.4704 | +0.3586 |
| 303 | 7.1202 | 7.3630 | +0.2427 |
| 404 | 7.1055 | 7.3737 | +0.2682 |
| 505 | 7.1191 | 7.3857 | +0.2666 |

The paired mean is `+0.2950` FID128 with sample SD `0.0505` and 95% CI
`[+0.2322, +0.3577]`. Lower FID is better, so every evaluation seed favors C0.
The frozen rule required `C4 - C0 <= -0.1`; therefore C4 fails the seed-2
confirmation.

## Mechanism Diagnostics

| metric | C0 mean | C4 mean | C4 - C0 |
|---|---:|---:|---:|
| flow curvature proxy | 0.020970 | 0.020957 | -0.000013 |
| flow straightness ratio | 1.109879 | 1.108810 | -0.001069 |

C4 is marginally better on these geometric proxies but substantially worse on
FID128. In this pair, the logged straightness diagnostics are not sufficient
surrogates for generation quality.

## Across GMM Seeds 0, 1, and 2

The historical labels called these training seeds, but only GMM initialization
and mix seeds changed; router and FM runtime RNG remained fixed.

| GMM seed | C0 mean FID128 | C4 mean FID128 | C4 - C0 |
|---:|---:|---:|---:|
| 0 | 7.0749 | 7.0400 | -0.0348 |
| 1 | 7.2660 | 6.9132 | -0.3528 |
| 2 | 7.1015 | 7.3965 | +0.2950 |

Across the three GMM-seed paired effects, the mean `C4 - C0` is `-0.0309`, the
sample SD is `0.3239`, and the 95% t interval is `[-0.8355, +0.7738]`. C4 wins
on two seeds but loses badly on one. The earlier two-seed aggregate improvement
does not replicate robustly.

## Conclusion

- Do not claim that C4 beats C0 under the current evidence.
- The dominant uncertainty is between-GMM-initialization variation, not the
  within-checkpoint FID generation noise.
- The seed-2 pair is internally decisive for its two checkpoints, but three GMM
  seeds are still too few for a precise family-level effect estimate.
- Any next experiment should address source sensitivity or evaluate more
  independent GMM seeds before further tuning against the best observed seed.
