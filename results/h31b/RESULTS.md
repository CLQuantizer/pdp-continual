# H31b repeater spacing on enwik8 E3M

E3M bytes: 3000000. sha256: `72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac`.
E10M sha256, checked and not scored: `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
GRU bits/byte = sum of live −log2 p / 3000000. Replay is not part of the code length.
No bytes are scored during sleep.

Fixed settings: seed 0, replay seed 30000, embedding 64, hidden 1024,
chunk 128, clip 1.0, Adam, replay batch 8. Learning rate and sleep block vary by arm.
The lr 3e-4 pair is H31 and was not rerun.

## E3M

| Arm | lr | Sleep block | Bits/byte | Replay steps | Wall seconds | GPU |
|---|---:|---:|---:|---:|---:|---|
| inter_lo | 0.00015 | 1 | 2.022717 | 23438 | 1101.413 | Tesla T4 |
| sleep1k_lo | 0.00015 | 1024 | 2.122769 | 23438 | 1149.500 | Tesla T4 |
| inter_hi | 0.0006 | 1 | 2.031624 | 23438 | 1193.622 | Tesla T4 |
| sleep1k_hi | 0.0006 | 1024 | 2.168835 | 23438 | 1109.052 | Tesla T4 |

## 1 MB slices

### inter_lo

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2275002.820 | 2.275003 |
| 1 | 1000000 | 2000000 | 1950605.241 | 1.950605 |
| 2 | 2000000 | 3000000 | 1842543.762 | 1.842544 |

### sleep1k_lo

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2447523.554 | 2.447524 |
| 1 | 1000000 | 2000000 | 2022027.658 | 2.022028 |
| 2 | 2000000 | 3000000 | 1898755.051 | 1.898755 |

### inter_hi

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2123755.649 | 2.123756 |
| 1 | 1000000 | 2000000 | 2014503.524 | 2.014504 |
| 2 | 2000000 | 3000000 | 1956611.482 | 1.956611 |

### sleep1k_hi

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2351971.773 | 2.351972 |
| 1 | 1000000 | 2000000 | 2128584.333 | 2.128584 |
| 2 | 2000000 | 3000000 | 2025949.801 | 2.025950 |

## Gaps

sleep1k − inter at each learning rate. The H31b gaps are unrounded. The E3M table rounds
bits/byte to 6 decimals. H31's gap is the published figure and was not rerun.
Not scored in this file.

gap(1.5e-4) = sleep1k_lo − inter_lo = 0.10005147972085071 (2.122768754133626 − 2.0227172744127753).
gap(3e-4) = H31 sleep_1k − inter = 0.132530 (2.090683 − 1.958153).
gap(6e-4) = sleep1k_hi − inter_hi = 0.13721175101118188 (2.168835302546588 − 2.031623551535406).
gap(6e-4) − gap(1.5e-4) = 0.03716027129033117.

No arm was nonfinite. No arm was rerun.

## Morning after

128 KB = 131072 bytes = 1024 chunks. For each sleep that has a full 128 KB awake
stretch immediately before it and a full 128 KB awake stretch immediately after it,
bits/byte of those stretches. The mean is over those sleeps. Awake bits only.

### sleep1k_lo

Sleeps in the mean: 21.
Mean bits/byte before: 2.141547.
Mean bits/byte after: 2.044730.
Mean after − before: -0.096817.

| Sleep | Bytes seen | Bits/byte before | Bits/byte after | After − before |
|---:|---:|---:|---:|---:|
| 0 | 131072 | 4.030444 | 2.484680 | -1.545764 |
| 1 | 262144 | 2.484680 | 2.350559 | -0.134121 |
| 2 | 393216 | 2.350559 | 2.239795 | -0.110764 |
| 3 | 524288 | 2.239795 | 1.938332 | -0.301463 |
| 4 | 655360 | 1.938332 | 2.195460 | 0.257128 |
| 5 | 786432 | 2.195460 | 2.106767 | -0.088693 |
| 6 | 917504 | 2.106767 | 2.093262 | -0.013506 |
| 7 | 1048576 | 2.093262 | 2.041549 | -0.051712 |
| 8 | 1179648 | 2.041549 | 2.037987 | -0.003563 |
| 9 | 1310720 | 2.037987 | 1.998475 | -0.039512 |
| 10 | 1441792 | 1.998475 | 2.038303 | 0.039828 |
| 11 | 1572864 | 2.038303 | 1.953010 | -0.085294 |
| 12 | 1703936 | 1.953010 | 1.997235 | 0.044225 |
| 13 | 1835008 | 1.997235 | 2.030780 | 0.033545 |
| 14 | 1966080 | 2.030780 | 1.992390 | -0.038390 |
| 15 | 2097152 | 1.992390 | 1.882175 | -0.110215 |
| 16 | 2228224 | 1.882175 | 1.852737 | -0.029438 |
| 17 | 2359296 | 1.852737 | 1.898868 | 0.046132 |
| 18 | 2490368 | 1.898868 | 1.852256 | -0.046613 |
| 19 | 2621440 | 1.852256 | 1.957425 | 0.105169 |
| 20 | 2752512 | 1.957425 | 1.997286 | 0.039861 |

### sleep1k_hi

Sleeps in the mean: 21.
Mean bits/byte before: 2.180147.
Mean bits/byte after: 2.120048.
Mean after − before: -0.060099.

| Sleep | Bytes seen | Bits/byte before | Bits/byte after | After − before |
|---:|---:|---:|---:|---:|
| 0 | 131072 | 3.400072 | 2.253054 | -1.147018 |
| 1 | 262144 | 2.253054 | 2.252563 | -0.000491 |
| 2 | 393216 | 2.252563 | 2.209861 | -0.042702 |
| 3 | 524288 | 2.209861 | 1.997646 | -0.212216 |
| 4 | 655360 | 1.997646 | 2.254591 | 0.256946 |
| 5 | 786432 | 2.254591 | 2.197385 | -0.057206 |
| 6 | 917504 | 2.197385 | 2.138197 | -0.059188 |
| 7 | 1048576 | 2.138197 | 2.150350 | 0.012153 |
| 8 | 1179648 | 2.150350 | 2.143957 | -0.006394 |
| 9 | 1310720 | 2.143957 | 2.107458 | -0.036498 |
| 10 | 1441792 | 2.107458 | 2.140116 | 0.032658 |
| 11 | 1572864 | 2.140116 | 2.075096 | -0.065020 |
| 12 | 1703936 | 2.075096 | 2.123063 | 0.047967 |
| 13 | 1835008 | 2.123063 | 2.146151 | 0.023088 |
| 14 | 1966080 | 2.146151 | 2.119956 | -0.026195 |
| 15 | 2097152 | 2.119956 | 2.018183 | -0.101773 |
| 16 | 2228224 | 2.018183 | 1.985165 | -0.033018 |
| 17 | 2359296 | 1.985165 | 2.023360 | 0.038195 |
| 18 | 2490368 | 2.023360 | 1.956521 | -0.066839 |
| 19 | 2621440 | 1.956521 | 2.090331 | 0.133810 |
| 20 | 2752512 | 2.090331 | 2.137999 | 0.047668 |

## Kernels

Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that
clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.
Script wall is time.monotonic from kernel_main entry through the result write. Queue time
is not included. Total GPU hours is the sum across the four kernels.

| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours |
|---|---:|---|---:|---|---:|---:|
| ezzzzzx/h31b-inter-lo | 1 | COMPLETE | 1137.788 | log_clock | 1115.917 | 0.3161 |
| ezzzzzx/h31b-sleep1k-lo | 1 | COMPLETE | 1184.458 | log_clock | 1163.272 | 0.3290 |
| ezzzzzx/h31b-inter-hi | 1 | COMPLETE | 1236.735 | log_clock | 1207.195 | 0.3435 |
| ezzzzzx/h31b-sleep1k-hi | 1 | COMPLETE | 1141.330 | log_clock | 1121.974 | 0.3170 |

Total GPU hours: 1.3056.

Kaggle accepted two GPU sessions at a time. ezzzzzx/h31b-inter-lo version 1 and ezzzzzx/h31b-sleep1k-lo version 1 ran together. ezzzzzx/h31b-inter-hi version 1 and ezzzzzx/h31b-sleep1k-hi version 1 were pushed after both of those reached COMPLETE. No kernel was rerun.

## Protocol

- One online pass per arm. Prequential: −log2 p of each byte is recorded before that chunk is trained.
- Awake: score the 128-byte chunk, then one Adam step on it. No replay step while awake, except sleep block 1.
- Sleep: after every B live chunks, B replay steps back to back. Each step is batch 8, window 128, hidden state 0.
- Replay starts are uniform on bytes seen so far, the same sampler as H29d. One rng draw per replay step.
- Sleep block 1 replays once after every live chunk. That is the H29d path.
- A final sleep covers a remainder shorter than B. Replay steps match across the four arms.
- The live hidden state is detached after the live score and is not replaced by replay.
- E3M is bytes [0, 3000000). 1 MB slices are 1000000 bytes.
- Traffic = replay steps × replay batch × 128 / bytes.
- Code is h31_sleep.py unchanged. Private dataset ezzzzzx/pdp-code-h31b. Data dataset ezzzzzx/enwik8.

## Smoke

CPU. Not a score. First 8192 bytes. Replay batch 8. Seed 0.

h31_sleep.py --lr 3e-4 --sleep-block 1 bits: 42063.298164869797.
H31 smoke bits: 42063.298164869797.
Absolute difference: 0.000e+00.
Within 1e-6: True.
Reference replay steps: 64.

| Arm | lr | Sleep block | Bits | Finite | Replay steps |
|---|---:|---:|---:|---|---:|
| inter_lo | 0.00015 | 1 | 45162.948127424774 | True | 64 |
| sleep1k_lo | 0.00015 | 1024 | 49683.635914933133 | True | 64 |
| inter_hi | 0.0006 | 1 | 38948.062321467165 | True | 64 |
| sleep1k_hi | 0.0006 | 1024 | 44227.316168456564 | True | 64 |

Replay steps for --sleep-block 1 and --sleep-block 1024: 64 and 64 at lr 1.5e-4, 64 and 64 at lr 6e-4.
Counts equal: True. All four finite: True.

## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (gap(1.5e-4) < gap(3e-4) < gap(6e-4)): PASS.** The gaps are 0.100 < 0.133 < 0.137.

**Bar 2 (gap(6e-4) − gap(1.5e-4) ≥ 0.05): FAIL.** The difference is 0.037.

**Mechanism.** Spacing costs more when the cortex forgets faster, so the repeater picture points the right way. But the
effect is small, and it saturates: doubling lr from 1.5e-4 to 3e-4 adds 0.033, while doubling again to 6e-4 adds only
0.005. That step is one seed and within noise. This fits a repeater spaced beyond the line's decay length. Once 1,024
chunks are longer than the weights' memory at either lr, the signal between repeats is already mostly lost, and faster
loss changes little. So the spacing cost at B = 1,024 is set by the spacing, not by the loss rate. The lossy-line
picture survives only in weak form. It predicts that the lr dependence would show at spacings near the decay length
(B ≈ 16–128), not at 1,024. One seed: treat this as a direction, not a size. The morning-after effect is smaller at the
high lr (−0.060 vs −0.097), which is consistent with faster wear between sleeps.

**Side finding.** The low lr compresses worse when replay is interleaved (2.023 vs H31's 1.958 at 3e-4). The best lr
for the dose is not the low one.

Line status: the repeater question is parked. Organic sleep (choosing when to sleep) is the live version of it.
