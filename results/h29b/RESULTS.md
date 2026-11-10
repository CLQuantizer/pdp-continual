# H29b enwik8 dose

E10M bytes: 10000000. sha256: `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
Baselines are the H29 measurements. They were not rerun.
xz -9e bits/byte: 2.176205. PPMd H order 16 bits/byte: 1.837254.
GRU bits/byte = sum of live −log2 p / 10000000. Replay is not part of the code length.

## E10M

| Arm | lr | Replay batch | Bits/byte | Replay steps | Traffic | Wall seconds | GPU | Minus xz -9e | Minus PPMd order 16 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| nr_1e3 | 0.001 | 0 | 2.794439 | 0 | 0.000000 | 1448.785 | Tesla T4 | 0.618234 | 0.957185 |
| r8_1e3 | 0.001 | 8 | 3.220097 | 78125 | 8.000000 | 3849.497 | Tesla T4 | 1.043892 | 1.382843 |
| nr_3e4 | 0.0003 | 0 | 1.973638 | 0 | 0.000000 | 1454.947 | Tesla T4 | -0.202567 | 0.136384 |

Minus xz -9e = arm bits/byte − 2.176205. Minus PPMd order 16 = arm bits/byte − 1.837254.

## nr_1e3 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2329297.413 | 2.329297 |
| 1 | 1000000 | 2000000 | 2218159.937 | 2.218160 |
| 2 | 2000000 | 3000000 | 2194800.650 | 2.194801 |
| 3 | 3000000 | 4000000 | 2296690.797 | 2.296691 |
| 4 | 4000000 | 5000000 | 2371822.464 | 2.371822 |
| 5 | 5000000 | 6000000 | 2436663.260 | 2.436663 |
| 6 | 6000000 | 7000000 | 2690608.048 | 2.690608 |
| 7 | 7000000 | 8000000 | 2942344.787 | 2.942345 |
| 8 | 8000000 | 9000000 | 3886366.668 | 3.886367 |
| 9 | 9000000 | 10000000 | 4577637.527 | 4.577638 |

## r8_1e3 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2213050.206 | 2.213050 |
| 1 | 1000000 | 2000000 | 2438259.012 | 2.438259 |
| 2 | 2000000 | 3000000 | 3755486.387 | 3.755486 |
| 3 | 3000000 | 4000000 | 3367998.133 | 3.367998 |
| 4 | 4000000 | 5000000 | 3475319.941 | 3.475320 |
| 5 | 5000000 | 6000000 | 3350598.521 | 3.350599 |
| 6 | 6000000 | 7000000 | 3315431.517 | 3.315432 |
| 7 | 7000000 | 8000000 | 3385662.661 | 3.385663 |
| 8 | 8000000 | 9000000 | 3416306.947 | 3.416307 |
| 9 | 9000000 | 10000000 | 3482853.166 | 3.482853 |

## nr_3e4 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2511646.944 | 2.511647 |
| 1 | 1000000 | 2000000 | 2109053.234 | 2.109053 |
| 2 | 2000000 | 3000000 | 1973250.047 | 1.973250 |
| 3 | 3000000 | 4000000 | 1964439.138 | 1.964439 |
| 4 | 4000000 | 5000000 | 1914244.770 | 1.914245 |
| 5 | 5000000 | 6000000 | 1870552.307 | 1.870552 |
| 6 | 6000000 | 7000000 | 1909113.441 | 1.909113 |
| 7 | 7000000 | 8000000 | 1851397.049 | 1.851397 |
| 8 | 8000000 | 9000000 | 1777629.070 | 1.777629 |
| 9 | 9000000 | 10000000 | 1855058.876 | 1.855059 |

## Reported differences

Pre-registered. Not scored in this file.

r8_1e3 − nr_1e3 = 0.425657 (3.220097 − 2.794439).
nr_3e4 − nr_1e3 = -0.820801 (1.973638 − 2.794439).

Lowest E10M bits/byte: nr_3e4 1.973638.
That arm, slice 0: 2.511647. Slice 9: 1.855059. Slice 9 − slice 0: -0.656588.

## Kernels

Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that
clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.
Script wall is time.monotonic from kernel_main entry through the result write. Queue time
is not included. Total GPU hours is the sum across the three kernels.

| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours |
|---|---:|---|---:|---|---:|---:|
| ezzzzzx/h29b-nr-1e3 | 1 | COMPLETE | 1482.705 | log_clock | 1462.741 | 0.4119 |
| ezzzzzx/h29b-r8-1e3 | 1 | COMPLETE | 3885.196 | log_clock | 3864.765 | 1.0792 |
| ezzzzzx/h29b-nr-3e4 | 1 | COMPLETE | 1507.173 | log_clock | 1472.424 | 0.4187 |

Total GPU hours: 1.9097.

Kaggle accepted two GPU sessions at a time. ezzzzzx/h29b-nr-3e4 version 1 was pushed after
ezzzzzx/h29b-nr-1e3 reached COMPLETE. No kernel was rerun.

## Protocol

- One online pass per arm. Prequential: −log2 p of each byte is recorded before that chunk is trained.
- GRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam, clip 1.0. Seed 0. Replay RNG 30000.
- Live state is carried and detached every 128 bytes. Replay state is zero.
- Replay starts are uniform on 0 .. L−128 inclusive, with replacement. Batch 0 skips that step.
- E10M is bytes [0, 10000000). 1 MB slices are 1000000 bytes.
- Traffic = replay steps × replay batch × 128 / bytes.
- The same E10M bytes are the calibration. No held-out split.

## nr_1e3 against the H29 no-replay slices

H29 no-replay, lr 0.001, first 2 MB. Slice bits, not rerun.

| Slice | H29b bits | H29 bits | Difference |
|---:|---:|---:|---:|
| 0 | 2329297.412924 | 2329297.412924 | 0.000e+00 |
| 1 | 2218159.937006 | 2218159.937006 | 0.000e+00 |

## Smoke, first 4096 bytes, CPU

Not a score.

h29 model path bits: 16784.078925371294.
h29b --lr 1e-3 --replay-batch 32 bits: 16784.078925371294.
Absolute difference: 0.000e+00.

| Arm | Bits | Finite | Replay steps | Traffic |
|---|---:|---|---:|---:|
| nr_1e3 | 19611.760897 | True | 0 | 0.000000 |
| r8_1e3 | 17019.407864 | True | 32 | 8.000000 |
| nr_3e4 | 23836.498041 | True | 0 | 0.000000 |

## Verdict (orchestrator, 11 Nov 2026, 03:10 UTC+8)

**Bar 1 (best arm beats xz on E10M): PASS.** nr_3e4 scores 1.974 bits/byte against xz -9e at 2.176 (−0.203). It
still trails PPMd order 16 (1.837) by 0.136. **Bar 2 (learns as it reads): PASS.** nr_3e4's slices fall from 2.51
to 1.86. The last slices sit around 1.78–1.86, which is near PPMd.

Mechanism. H29's failure was mainly instability. At lr 1e-3, online Adam degrades after about 5 MB even with no replay
(nr_1e3 slices go 2.19 → 4.58). At lr 3e-4 it keeps improving for the whole 10 MB. The smaller step size is what lets
the learner live for 10^7 steps. Replay at the toy-world knee (traffic 8) still cost +0.43 bits/byte at lr 1e-3, and it
broke the curve earlier (slice 2: 3.76). Every replay step is an extra Adam step at the same learning rate, so replay
multiplies the instability. Replay at the stable learning rate is untested, so replay on real text is not closed. The
smoke matched H29 bit for bit, and nr_1e3's first 2 MB equal H29's no-replay run.
