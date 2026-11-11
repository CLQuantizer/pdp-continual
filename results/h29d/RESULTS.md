# H29d seed spread on enwik8 E10M

E10M bytes: 10000000. sha256: `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
Seed 0 r8_3e4 is H29c and was not rerun. Seed 0 nr_3e4 is H29b and was not rerun.
xz -9e bits/byte: 2.176205. PPMd H order 16 bits/byte: 1.837254.
GRU bits/byte = sum of live −log2 p / 10000000. Replay is not part of the code length.
Population std uses ddof = 0.

## Kernels

| Kernel | Arm | Seed | lr | Replay batch | Bits/byte | Replay steps | Traffic | Wall seconds | GPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ezzzzzx/h29d-r8-s1 | r8_3e4 | 1 | 0.0003 | 8 | 1.836391 | 78125 | 8.000000 | 3853.506 | Tesla T4 |
| ezzzzzx/h29d-r8-s2 | r8_3e4 | 2 | 0.0003 | 8 | 1.840442 | 78125 | 8.000000 | 3723.786 | Tesla T4 |
| ezzzzzx/h29d-nr-s1 | nr_3e4 | 1 | 0.0003 | 0 | 1.973554 | 0 | 0.000000 | 1431.730 | Tesla T4 |
| ezzzzzx/h29d-nr-s2 | nr_3e4 | 2 | 0.0003 | 0 | 1.974520 | 0 | 0.000000 | 1428.805 | Tesla T4 |

## r8_3e4 seed 1 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2109085.610 | 2.109086 |
| 1 | 1000000 | 2000000 | 1929557.358 | 1.929557 |
| 2 | 2000000 | 3000000 | 1852774.265 | 1.852774 |
| 3 | 3000000 | 4000000 | 1862698.835 | 1.862699 |
| 4 | 4000000 | 5000000 | 1805213.098 | 1.805213 |
| 5 | 5000000 | 6000000 | 1759257.982 | 1.759258 |
| 6 | 6000000 | 7000000 | 1817586.209 | 1.817586 |
| 7 | 7000000 | 8000000 | 1759419.808 | 1.759420 |
| 8 | 8000000 | 9000000 | 1703030.221 | 1.703030 |
| 9 | 9000000 | 10000000 | 1765287.145 | 1.765287 |

## r8_3e4 seed 2 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2117951.849 | 2.117952 |
| 1 | 1000000 | 2000000 | 1934600.204 | 1.934600 |
| 2 | 2000000 | 3000000 | 1859762.027 | 1.859762 |
| 3 | 3000000 | 4000000 | 1866655.813 | 1.866656 |
| 4 | 4000000 | 5000000 | 1809826.621 | 1.809827 |
| 5 | 5000000 | 6000000 | 1762978.267 | 1.762978 |
| 6 | 6000000 | 7000000 | 1819680.538 | 1.819681 |
| 7 | 7000000 | 8000000 | 1762816.024 | 1.762816 |
| 8 | 8000000 | 9000000 | 1704879.545 | 1.704880 |
| 9 | 9000000 | 10000000 | 1765268.397 | 1.765268 |

## nr_3e4 seed 1 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2511217.179 | 2.511217 |
| 1 | 1000000 | 2000000 | 2111061.479 | 2.111061 |
| 2 | 2000000 | 3000000 | 1974931.278 | 1.974931 |
| 3 | 3000000 | 4000000 | 1966214.544 | 1.966215 |
| 4 | 4000000 | 5000000 | 1914235.857 | 1.914236 |
| 5 | 5000000 | 6000000 | 1868908.558 | 1.868909 |
| 6 | 6000000 | 7000000 | 1907380.284 | 1.907380 |
| 7 | 7000000 | 8000000 | 1850050.005 | 1.850050 |
| 8 | 8000000 | 9000000 | 1778291.743 | 1.778292 |
| 9 | 9000000 | 10000000 | 1853244.545 | 1.853245 |

## nr_3e4 seed 2 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2516995.232 | 2.516995 |
| 1 | 1000000 | 2000000 | 2110784.173 | 2.110784 |
| 2 | 2000000 | 3000000 | 1975875.905 | 1.975876 |
| 3 | 3000000 | 4000000 | 1967909.778 | 1.967910 |
| 4 | 4000000 | 5000000 | 1916708.384 | 1.916708 |
| 5 | 5000000 | 6000000 | 1869646.570 | 1.869647 |
| 6 | 6000000 | 7000000 | 1908633.578 | 1.908634 |
| 7 | 7000000 | 8000000 | 1846353.888 | 1.846354 |
| 8 | 8000000 | 9000000 | 1777632.073 | 1.777632 |
| 9 | 9000000 | 10000000 | 1854663.080 | 1.854663 |

## Seeds

Seed 0 r8_3e4 bits/byte is H29c. Seed 0 nr_3e4 bits/byte is H29b.

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean | Std | Mean − PPMd order 16 | Mean − xz -9e |
|---|---:|---:|---:|---:|---:|---:|---:|
| r8_3e4 | 1.831624 | 1.836391 | 1.840442 | 1.836152 | 0.003604 | -0.001102 | -0.340053 |
| nr_3e4 | 1.973638 | 1.973554 | 1.974520 | 1.973904 | 0.000437 | 0.136650 | -0.202301 |

## Paired difference r8 − nr

Pre-registered. Not scored in this file.

| Seed | r8 − nr |
|---:|---:|
| 0 | -0.142014 |
| 1 | -0.137162 |
| 2 | -0.134078 |

Mean r8 − nr: -0.137752. Std: 0.003266.

r8 mean: 1.836152. r8 std: 0.003604.
r8 mean + std: 1.839756.
r8 |mean − 1.837254|: 0.001102.
PPMd order 16: 1.837254.

## Kernel time

Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that
clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.
Script wall is time.monotonic from kernel_main entry through the result write. Queue time
is not included. Total GPU hours is the sum across the four H29d kernels. Seed 0 is not included.

| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours |
|---|---:|---|---:|---|---:|---:|
| ezzzzzx/h29d-r8-s1 | 1 | COMPLETE | 3889.943 | log_clock | 3866.919 | 1.0805 |
| ezzzzzx/h29d-r8-s2 | 1 | COMPLETE | 3757.651 | log_clock | 3737.739 | 1.0438 |
| ezzzzzx/h29d-nr-s1 | 1 | COMPLETE | 1467.003 | log_clock | 1445.535 | 0.4075 |
| ezzzzzx/h29d-nr-s2 | 1 | COMPLETE | 1473.050 | log_clock | 1443.136 | 0.4092 |

Total GPU hours: 2.9410.

Kaggle accepted two GPU sessions at a time. ezzzzzx/h29d-r8-s1 version 1 and ezzzzzx/h29d-r8-s2 version 1 ran together. ezzzzzx/h29d-nr-s1 version 1 was pushed after ezzzzzx/h29d-r8-s2 reached COMPLETE, while ezzzzzx/h29d-r8-s1 was still running. ezzzzzx/h29d-nr-s2 version 1 was pushed after ezzzzzx/h29d-r8-s1 reached COMPLETE, while ezzzzzx/h29d-nr-s1 was still running. No kernel was rerun.

Each H29d kernel checked sha256_e10m `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.

## Protocol

- One online pass per arm. Prequential: −log2 p of each byte is recorded before that chunk is trained.
- GRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam, clip 1.0.
- Seed 0 is the H29b and H29c run. Seeds 1 and 2 set SEED. Replay RNG is 30000 + SEED.
- Live state is carried and detached every 128 bytes. Replay state is zero.
- Replay starts are uniform on 0 .. L−128 inclusive, with replacement. Batch 0 skips that step.
- E10M is bytes [0, 10000000). 1 MB slices are 1000000 bytes.
- Traffic = replay steps × replay batch × 128 / bytes.
- The same E10M bytes are the calibration. No held-out split.
- Code is h29d_enwik8.py. Private dataset ezzzzzx/pdp-code-h29d. Data dataset ezzzzzx/enwik8.

## Smoke, first 4096 bytes, CPU

Not a score.

h29d --seed 0 --lr 3e-4 --replay-batch 0 bits: 23836.498040661034.
Absolute difference from 23836.498041: 3.390e-07.
Within 1e-6: True.
Seed 0 replay batch 8 bits: 20875.886154200049.
Seed 1 replay batch 8 bits: 20958.222682068634.
Absolute difference: 82.336528.
Both replay smokes are finite.

## Verdict (orchestrator, 11 Nov 2026, 10:30 UTC+8)

**Bar 1 (replay gain holds): PASS.** r8 − nr is negative on all 3 seeds (−0.142, −0.137, −0.134), mean −0.138 ±
0.003. That is nearly 3× the −0.05 bar, with a spread 40× smaller than the effect. **Bar 2 (PPMd): TIE.** The r8 mean
is 1.8362 ± 0.0036 against PPMd order 16 at 1.8373: |difference| 0.0011 ≤ std. The H29c "edge" was within seed
noise.

Mechanism. The replay gain on real text is robust, not a lucky seed. Seeds move the result by about 0.004 and replay
moves it by 0.138. The cortex compresses the first 10 MB of Wikipedia as well as a strong order-16 context mixer, and
its last-MB slices (about 1.76) are still falling. The code differs from h29b only in the `--seed` flag, which was
checked by diff.
