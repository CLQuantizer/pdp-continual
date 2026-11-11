I'll read the project instructions and the handoff file first, then follow the handoff exactly.The handoff is a seed-spread run on enwik8. I'll read the prior results and the H29b/H29c code before changing anything.H29c’s replay gain and the PPMd margin are single-seed. I’ll add a `--seed` flag, smoke-test it on CPU, then run the four private GPU kernels.Seed 0 with no replay matches the H29b smoke: 23836.498040661 bits, within 1e-6 of 23836.498041. Next I’ll check that seed 1 with replay actually changes the bits.The CPU smoke passed, and the two replay kernels are running on Kaggle. The no-replay kernels wait for a free GPU slot.

Seed 0 with no replay on the first 4,096 bytes scored 23836.498040661 bits, 3.39e-07 from the H29b value 23836.498041. Seed 0 with replay batch 8 matched the H29c smoke exactly at 20875.886154200 bits. Seed 1 with the same replay dose was finite and 82.337 bits away, with replay seed 30001.

`ezzzzzx/h29d-r8-s1` and `ezzzzzx/h29d-r8-s2` are both version 1 and running, private GPU, no internet. Code is the private dataset `ezzzzzx/pdp-code-h29d`. E10M sha256 checked as `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`. `h29b_enwik8.py` was not changed. Nothing is committed.

`r8-s2` finished cleanly: seed 2, 1.840442 bits/byte, sha256 matches. One GPU slot is free, so I’ll start the first no-replay kernel.`ezzzzzx/h29d-r8-s2` finished. Seed 2, replay batch 8, E10M is 1.840442 bits/byte, replay steps 78125, traffic 8, Tesla T4, sha256 matched. `ezzzzzx/h29d-nr-s1` is now running in the free GPU slot. `ezzzzzx/h29d-r8-s1` is still running, so `nr-s2` waits for that slot.

`r8-s1` finished while `nr-s1` is still running. I'll pull that result and start `nr-s2` in the free slot.`ezzzzzx/h29d-r8-s1` finished. Seed 1, replay batch 8, E10M is 1.836391 bits/byte, replay steps 78125, traffic 8, Tesla T4, sha256 matched. Both no-replay kernels are running: `ezzzzzx/h29d-nr-s1` and `ezzzzzx/h29d-nr-s2`, each version 1.

`nr-s1` finished. I'll pull that result and leave `nr-s2` running.`ezzzzzx/h29d-nr-s1` finished. Seed 1, no replay, E10M is 1.973554 bits/byte, replay steps 0, Tesla T4, sha256 matched. `ezzzzzx/h29d-nr-s2` is still running.

Both no-replay kernels are done. I'll fetch `nr-s2` and write the results.H29d is in `results/h29d/RESULTS.md` and `results/h29d/results.json`. Four private GPU kernels finished, version 1, no rerun. Total GPU time is 2.9410 hours. Nothing was committed.

E10M bits/byte at lr 3e-4. Seed 0 is H29c `r8_3e4` and H29b `nr_3e4`. Population std uses ddof = 0. PPMd order 16 is 1.837254. xz -9e is 2.176205.

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean | Std | Mean − PPMd order 16 | Mean − xz -9e |
|---|---:|---:|---:|---:|---:|---:|---:|
| r8_3e4 | 1.831624 | 1.836391 | 1.840442 | 1.836152 | 0.003604 | -0.001102 | -0.340053 |
| nr_3e4 | 1.973638 | 1.973554 | 1.974520 | 1.973904 | 0.000437 | 0.136650 | -0.202301 |

Paired r8 − nr: seed 0 is -0.142014, seed 1 is -0.137162, seed 2 is -0.134078. Mean -0.137752, std 0.003266.

r8 mean + std is 1.839756. |r8 mean − 1.837254| is 0.001102.

| Kernel | Replay batch | Seed | Bits/byte | Replay steps | Traffic | Wall seconds | GPU | Elapsed seconds | GPU hours |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| ezzzzzx/h29d-r8-s1 | 8 | 1 | 1.836391 | 78125 | 8 | 3853.506 | Tesla T4 | 3889.943 | 1.0805 |
| ezzzzzx/h29d-r8-s2 | 8 | 2 | 1.840442 | 78125 | 8 | 3723.786 | Tesla T4 | 3757.651 | 1.0438 |
| ezzzzzx/h29d-nr-s1 | 0 | 1 | 1.973554 | 0 | 0 | 1431.730 | Tesla T4 | 1467.003 | 0.4075 |
| ezzzzzx/h29d-nr-s2 | 0 | 2 | 1.974520 | 0 | 0 | 1428.805 | Tesla T4 | 1473.050 | 0.4092 |

The two r8 kernels ran together. `nr-s1` started after `r8-s2` finished, and `nr-s2` started after `r8-s1` finished. Each kernel checked sha256 `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`. Replay seeds were 30001 and 30002.

CPU smoke on the first 4,096 bytes: seed 0, no replay, 23836.498040661 bits, 3.390e-07 from 23836.498041. Seed 1 with replay batch 8 was finite and 82.337 bits from seed 0.
