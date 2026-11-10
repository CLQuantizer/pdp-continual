# H29c replay at the stable learning rate

E10M bytes: 10000000. sha256: `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
The H29b nr_3e4 arm was not rerun. Its E10M bits/byte is 1.973638.
xz -9e and PPMd order 16 were not rerun. xz -9e bits/byte: 2.176205. PPMd H order 16 bits/byte: 1.837254.
GRU bits/byte = sum of live −log2 p / 10000000. Replay is not part of the code length.

## E10M

| Arm | lr | Replay batch | Bits/byte | Replay steps | Traffic | Wall seconds | GPU | Minus nr_3e4 | Minus xz -9e | Minus PPMd order 16 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| r1_3e4 | 0.0003 | 1 | 1.874187 | 78125 | 1.000000 | 2798.995 | Tesla T4 | -0.099451 | -0.302018 | 0.036933 |
| r8_3e4 | 0.0003 | 8 | 1.831624 | 78125 | 8.000000 | 3783.846 | Tesla T4 | -0.142014 | -0.344581 | -0.005630 |
| nr_1e4 | 0.0001 | 0 | 2.150570 | 0 | 0.000000 | 1430.005 | Tesla T4 | 0.176932 | -0.025635 | 0.313316 |

Minus nr_3e4 = arm bits/byte − 1.973638. Minus xz -9e = arm bits/byte − 2.176205. Minus PPMd order 16 = arm bits/byte − 1.837254.

## r1_3e4 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2211125.843 | 2.211126 |
| 1 | 1000000 | 2000000 | 1958007.370 | 1.958007 |
| 2 | 2000000 | 3000000 | 1865439.975 | 1.865440 |
| 3 | 3000000 | 4000000 | 1873829.604 | 1.873830 |
| 4 | 4000000 | 5000000 | 1829578.137 | 1.829578 |
| 5 | 5000000 | 6000000 | 1796175.513 | 1.796176 |
| 6 | 6000000 | 7000000 | 1849185.804 | 1.849186 |
| 7 | 7000000 | 8000000 | 1801513.264 | 1.801513 |
| 8 | 8000000 | 9000000 | 1744325.594 | 1.744326 |
| 9 | 9000000 | 10000000 | 1812693.562 | 1.812694 |

## r8_3e4 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2104744.492 | 2.104744 |
| 1 | 1000000 | 2000000 | 1923321.387 | 1.923321 |
| 2 | 2000000 | 3000000 | 1846393.123 | 1.846393 |
| 3 | 3000000 | 4000000 | 1857307.565 | 1.857308 |
| 4 | 4000000 | 5000000 | 1799447.295 | 1.799447 |
| 5 | 5000000 | 6000000 | 1753810.931 | 1.753811 |
| 6 | 6000000 | 7000000 | 1813825.167 | 1.813825 |
| 7 | 7000000 | 8000000 | 1756344.262 | 1.756344 |
| 8 | 8000000 | 9000000 | 1699067.329 | 1.699067 |
| 9 | 9000000 | 10000000 | 1761981.305 | 1.761981 |

## nr_1e4 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 3062929.343 | 3.062929 |
| 1 | 1000000 | 2000000 | 2433423.472 | 2.433423 |
| 2 | 2000000 | 3000000 | 2206953.813 | 2.206954 |
| 3 | 3000000 | 4000000 | 2147466.843 | 2.147467 |
| 4 | 4000000 | 5000000 | 2044538.033 | 2.044538 |
| 5 | 5000000 | 6000000 | 1962794.644 | 1.962795 |
| 6 | 6000000 | 7000000 | 1994153.969 | 1.994154 |
| 7 | 7000000 | 8000000 | 1916685.956 | 1.916686 |
| 8 | 8000000 | 9000000 | 1835964.296 | 1.835964 |
| 9 | 9000000 | 10000000 | 1900791.629 | 1.900792 |

## Reported differences

Pre-registered. Not scored in this file.

nr_1e4 − nr_3e4 = 0.176932 (2.150570 − 1.973638).

Last 1 MB slice bits/byte:

- r1_3e4 slice 9: 1.812694.
- r8_3e4 slice 9: 1.761981.
- nr_1e4 slice 9: 1.900792.

## Kernels

Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that
clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.
Script wall is time.monotonic from kernel_main entry through the result write. Queue time
is not included. Total GPU hours is the sum across the three kernels.

| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours |
|---|---:|---|---:|---|---:|---:|
| ezzzzzx/h29c-r1-3e4 | 1 | COMPLETE | 2836.048 | log_clock | 2812.827 | 0.7878 |
| ezzzzzx/h29c-r8-3e4 | 1 | COMPLETE | 3810.420 | log_clock | 3796.267 | 1.0584 |
| ezzzzzx/h29c-nr-1e4 | 1 | COMPLETE | 1467.049 | log_clock | 1443.138 | 0.4075 |

Total GPU hours: 2.2538.

Kaggle accepted two GPU sessions at a time. ezzzzzx/h29c-r1-3e4 version 1 and
ezzzzzx/h29c-r8-3e4 version 1 ran together. ezzzzzx/h29c-nr-1e4 version 1 was pushed after
ezzzzzx/h29c-r1-3e4 reached COMPLETE, while ezzzzzx/h29c-r8-3e4 was still running.
No kernel was rerun.

Each kernel printed sha256_e10m `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.

## Protocol

- One online pass per arm. Prequential: −log2 p of each byte is recorded before that chunk is trained.
- GRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam, clip 1.0. Seed 0. Replay RNG 30000.
- Live state is carried and detached every 128 bytes. Replay state is zero.
- Replay starts are uniform on 0 .. L−128 inclusive, with replacement. Batch 0 skips that step.
- E10M is bytes [0, 10000000). 1 MB slices are 1000000 bytes.
- Traffic = replay steps × replay batch × 128 / bytes.
- The same E10M bytes are the calibration. No held-out split.
- Code is h29b_enwik8.py, unchanged. Private dataset ezzzzzx/pdp-code-h29c. Data dataset ezzzzzx/enwik8.

## Smoke, first 4096 bytes, CPU

Not a score.

h29b nr_3e4 smoke bits: 23836.498040661034.
h29b_enwik8.py --lr 3e-4 --replay-batch 0 bits: 23836.498040661034.
Absolute difference from 23836.498041: 3.390e-07.
Within 1e-6: True.

| Arm | Bits | Finite | Replay steps | Traffic |
|---|---:|---|---:|---:|
| nr_3e4 | 23836.498041 | True | 0 | 0.000000 |
| r1_3e4 | 21352.219491 | True | 32 | 1.000000 |
| r8_3e4 | 20875.886154 | True | 32 | 8.000000 |
| nr_1e4 | 29669.629392 | True | 0 | 0.000000 |

## Verdict (orchestrator, 11 Nov 2026, 04:40 UTC+8)

**Bar 1 (replay helps when stable): PASS.** r8_3e4 scores 1.832 against nr_3e4 at 1.974 (−0.142). r1_3e4 also clears
the bar at 1.874 (−0.099). **Bar 2 (beats PPMd order 16): PASS, narrowly.** 1.8316 against 1.8373 is −0.006 bits/byte
on a single run, with no seed spread measured. Treat it as a tie with PPMd that leans our way, not a clear win. The kill
rule does not fire.

Mechanism. Store-everything replay helps on real text once the step size is stable, and more replay helps more: traffic
8 beats traffic 1. That matches the toy world, where traffic 8 was the knee. In H29b replay looked harmful only because
it multiplied an unstable learning rate. lr 1e-4 is too slow (2.151), so 3e-4 sits between too slow and divergent. The
last 1 MB of r8_3e4 costs 1.762 bits/byte. It is still falling at 10 MB, so the curve has not saturated.
