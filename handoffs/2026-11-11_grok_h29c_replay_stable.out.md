I'll read the project instructions and the handoff, then follow that handoff exactly.The handoff is H29c: three stable-learning-rate replay arms on enwik8 E10M. I'll read the H29b results and runner first, then smoke-test before any Kaggle kernel.The 4,096-byte smoke is running. I'll prepare the kernel folders and check the Kaggle CLI while it finishes.The smoke matches H29b: `nr_3e4` is 23836.498040661 bits, within 1e-6 of 23836.498041, and the three new arms are finite. Next I’ll upload the code dataset and start two GPU kernels.`r1_3e4` finished and `r8_3e4` is still running. I'll start `nr_1e4` in the free GPU slot and pull the `r1` result.H29c is in `results/h29c/RESULTS.md` and `results/h29c/results.json`. `h29b_enwik8.py` was not changed. No kernel was rerun. Total GPU time is 2.2538 hours.

The CPU smoke on the first 4,096 bytes reproduced the H29b `nr_3e4` value: 23836.498040661034 bits, 3.390e-07 from 23836.498041. `r1_3e4`, `r8_3e4`, and `nr_1e4` were finite.

E10M bits/byte, against the H29b `nr_3e4` value 1.973638, xz -9e 2.176205, and PPMd order 16 1.837254:

| Arm | lr | Replay batch | Bits/byte | Replay steps | Traffic | Wall seconds | GPU | Minus nr_3e4 | Minus xz -9e | Minus PPMd order 16 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| r1_3e4 | 0.0003 | 1 | 1.874187 | 78125 | 1 | 2798.995 | Tesla T4 | -0.099451 | -0.302018 | 0.036933 |
| r8_3e4 | 0.0003 | 8 | 1.831624 | 78125 | 8 | 3783.846 | Tesla T4 | -0.142014 | -0.344581 | -0.005630 |
| nr_1e4 | 0.0001 | 0 | 2.150570 | 0 | 0 | 1430.005 | Tesla T4 | 0.176932 | -0.025635 | 0.313316 |

nr_1e4 − nr_3e4 = 0.176932. Last 1 MB slice: r1_3e4 1.812694, r8_3e4 1.761981, nr_1e4 1.900792.

Each kernel printed E10M sha256 `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.

| Kernel | Version | Status | Elapsed seconds | GPU hours |
|---|---:|---|---:|---:|
| ezzzzzx/h29c-r1-3e4 | 1 | COMPLETE | 2836.048 | 0.7878 |
| ezzzzzx/h29c-r8-3e4 | 1 | COMPLETE | 3810.420 | 1.0584 |
| ezzzzzx/h29c-nr-1e4 | 1 | COMPLETE | 1467.049 | 0.4075 |

`r1_3e4` and `r8_3e4` ran together. `nr_1e4` was pushed after `r1_3e4` reached COMPLETE.
