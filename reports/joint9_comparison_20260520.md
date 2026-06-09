# Joint9 Comparison

| rank | variant | run | last step | eval step | FID128 best/last | FID32 best/last | train/valid loss | pred/target var | router joint grad |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | neither_frozen_x1 | `tide-joint9-r1-k16-top2-soft075-dir512-frozen` | 341700 | 320000 | 7.580/7.580 | 9.811/9.811 | 0.4701/0.4660 | 0.672 |  |
| r1 | joint_only_x1 | `tide-joint9-r1-k16-top2-soft075-dir512-jointx1` | 320000 | 320000 | 7.541/7.541 | 9.768/10.013 | 0.4802/0.4637 | 0.645 | 0.000 |
| r1 | both_joint_mix | `tide-joint9-r1-k16-top2-soft075-dir512-jointmix` | 320000 | 320000 | 7.643/7.643 | 9.864/9.961 | 0.4825/0.4756 | 0.631 | 0.000 |
| r2 | neither_frozen_x1 | `tide-joint9-r2-k32-top2-soft075-dir001-frozen` | 331100 | 320000 | 7.755/7.755 | 10.097/10.097 | 0.4724/0.4646 | 0.646 |  |
| r2 | joint_only_x1 | `tide-joint9-r2-k32-top2-soft075-dir001-jointx1` | 320000 | 320000 | 8.082/8.082 | 10.369/10.369 | 0.4719/0.4668 | 0.635 | 0.000 |
| r2 | both_joint_mix | `tide-joint9-r2-k32-top2-soft075-dir001-jointmix` | 320000 | 320000 | 7.863/7.867 | 10.019/10.109 | 0.4808/0.4772 | 0.617 | 0.000 |
| r3 | neither_frozen_x1 | `tide-joint9-r3-k32-top2-none-hard05-frozen` | 339100 | 320000 | 7.811/7.811 | 10.178/10.178 | 0.4735/0.4782 | 0.642 |  |
| r3 | joint_only_x1 | `tide-joint9-r3-k32-top2-none-hard05-jointx1` | 320000 | 320000 | 8.110/8.110 | 10.299/10.299 | 0.4719/0.4562 | 0.632 | 0.000 |
| r3 | both_joint_mix | `tide-joint9-r3-k32-top2-none-hard05-jointmix` | 320000 | 320000 | 8.068/8.068 | 10.331/10.520 | 0.4744/0.4696 | 0.620 | 0.000 |
