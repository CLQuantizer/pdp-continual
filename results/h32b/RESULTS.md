# H32b cerebellar mixer on enwik8 E10M

E10M bytes: 10000000. sha256: `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
GRU bits/byte = sum of live −log2 p / 10000000. The mixers and the count
models are scored the same way. PPMd bits/byte = 8 × compressed bytes / 10000000.
Replay is not part of the code length. Mixer gradients do not enter the cortex.

Cortex: H26 uniform, lr 0.0003, seed 0, replay seed 30000, hidden 1024,
chunk 128, replay batch 8, sleep block 1. One cortex pass feeds every arm.
Mixers: Adam lr 0.003, one step per chunk. mix4 and mix6 rows start at
[1, 0, …, 0]. temp4 starts at 1 on every row. Count smoothing 0.05.
Orders 1–6 share one count store. Counts of a chunk are added after every
mixer has scored that chunk and taken its Adam step.

## E10M

| Arm | Bits/byte | Bits |
|---|---:|---:|
| gru | 1.831624 | 18316242.856 |
| count_1 | 3.874791 | 38747909.593 |
| count_2 | 3.126652 | 31266523.995 |
| count_3 | 2.682226 | 26822256.595 |
| count_4 | 2.642897 | 26428973.848 |
| count_5 | 2.985393 | 29853925.055 |
| count_6 | 3.528447 | 35284472.420 |
| mix4 | 1.750253 | 17502529.252 |
| temp4 | 1.818013 | 18180127.900 |
| mix6 | 1.712450 | 17124499.982 |
| ppmd_order_16 | 1.837254 | 18372544.000 |

## 1 MB slices

### gru

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

### count_1

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 3874483.318 | 3.874483 |
| 1 | 1000000 | 2000000 | 3880345.051 | 3.880345 |
| 2 | 2000000 | 3000000 | 3875507.899 | 3.875508 |
| 3 | 3000000 | 4000000 | 3915587.742 | 3.915588 |
| 4 | 4000000 | 5000000 | 3881927.482 | 3.881927 |
| 5 | 5000000 | 6000000 | 3846330.512 | 3.846331 |
| 6 | 6000000 | 7000000 | 3868820.286 | 3.868820 |
| 7 | 7000000 | 8000000 | 3841539.676 | 3.841540 |
| 8 | 8000000 | 9000000 | 3840175.330 | 3.840175 |
| 9 | 9000000 | 10000000 | 3923192.296 | 3.923192 |

### count_2

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 3166793.210 | 3.166793 |
| 1 | 1000000 | 2000000 | 3148434.798 | 3.148435 |
| 2 | 2000000 | 3000000 | 3140329.014 | 3.140329 |
| 3 | 3000000 | 4000000 | 3179698.676 | 3.179699 |
| 4 | 4000000 | 5000000 | 3132753.984 | 3.132754 |
| 5 | 5000000 | 6000000 | 3077127.190 | 3.077127 |
| 6 | 6000000 | 7000000 | 3119003.579 | 3.119004 |
| 7 | 7000000 | 8000000 | 3088511.758 | 3.088512 |
| 8 | 8000000 | 9000000 | 3070198.873 | 3.070199 |
| 9 | 9000000 | 10000000 | 3143672.913 | 3.143673 |

### count_3

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2961961.115 | 2.961961 |
| 1 | 1000000 | 2000000 | 2773872.038 | 2.773872 |
| 2 | 2000000 | 3000000 | 2695585.193 | 2.695585 |
| 3 | 3000000 | 4000000 | 2722477.600 | 2.722478 |
| 4 | 4000000 | 5000000 | 2665329.156 | 2.665329 |
| 5 | 5000000 | 6000000 | 2594568.694 | 2.594569 |
| 6 | 6000000 | 7000000 | 2642605.512 | 2.642606 |
| 7 | 7000000 | 8000000 | 2595894.744 | 2.595895 |
| 8 | 8000000 | 9000000 | 2552116.153 | 2.552116 |
| 9 | 9000000 | 10000000 | 2617846.390 | 2.617846 |

### count_4

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 3297434.917 | 3.297435 |
| 1 | 1000000 | 2000000 | 2876494.486 | 2.876494 |
| 2 | 2000000 | 3000000 | 2694611.554 | 2.694612 |
| 3 | 3000000 | 4000000 | 2687912.419 | 2.687912 |
| 4 | 4000000 | 5000000 | 2587274.437 | 2.587274 |
| 5 | 5000000 | 6000000 | 2479200.778 | 2.479201 |
| 6 | 6000000 | 7000000 | 2532770.050 | 2.532770 |
| 7 | 7000000 | 8000000 | 2449932.596 | 2.449933 |
| 8 | 8000000 | 9000000 | 2387206.496 | 2.387206 |
| 9 | 9000000 | 10000000 | 2436136.114 | 2.436136 |

### count_5

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 3936432.606 | 3.936433 |
| 1 | 1000000 | 2000000 | 3382315.973 | 3.382316 |
| 2 | 2000000 | 3000000 | 3110716.836 | 3.110717 |
| 3 | 3000000 | 4000000 | 3057416.253 | 3.057416 |
| 4 | 4000000 | 5000000 | 2903569.052 | 2.903569 |
| 5 | 5000000 | 6000000 | 2752178.611 | 2.752179 |
| 6 | 6000000 | 7000000 | 2804322.104 | 2.804322 |
| 7 | 7000000 | 8000000 | 2683374.081 | 2.683374 |
| 8 | 8000000 | 9000000 | 2592907.427 | 2.592907 |
| 9 | 9000000 | 10000000 | 2630692.114 | 2.630692 |

### count_6

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 4611376.155 | 4.611376 |
| 1 | 1000000 | 2000000 | 4042222.475 | 4.042222 |
| 2 | 2000000 | 3000000 | 3730190.383 | 3.730190 |
| 3 | 3000000 | 4000000 | 3638005.108 | 3.638005 |
| 4 | 4000000 | 5000000 | 3441151.357 | 3.441151 |
| 5 | 5000000 | 6000000 | 3263139.245 | 3.263139 |
| 6 | 6000000 | 7000000 | 3310073.015 | 3.310073 |
| 7 | 7000000 | 8000000 | 3151982.164 | 3.151982 |
| 8 | 8000000 | 9000000 | 3034594.851 | 3.034595 |
| 9 | 9000000 | 10000000 | 3061737.665 | 3.061738 |

### mix4

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 1951635.909 | 1.951636 |
| 1 | 1000000 | 2000000 | 1824186.192 | 1.824186 |
| 2 | 2000000 | 3000000 | 1763414.944 | 1.763415 |
| 3 | 3000000 | 4000000 | 1777164.053 | 1.777164 |
| 4 | 4000000 | 5000000 | 1724145.509 | 1.724146 |
| 5 | 5000000 | 6000000 | 1688927.494 | 1.688927 |
| 6 | 6000000 | 7000000 | 1746684.700 | 1.746685 |
| 7 | 7000000 | 8000000 | 1693518.859 | 1.693519 |
| 8 | 8000000 | 9000000 | 1641487.083 | 1.641487 |
| 9 | 9000000 | 10000000 | 1691364.509 | 1.691365 |

### temp4

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2088270.979 | 2.088271 |
| 1 | 1000000 | 2000000 | 1899728.068 | 1.899728 |
| 2 | 2000000 | 3000000 | 1827179.267 | 1.827179 |
| 3 | 3000000 | 4000000 | 1840033.874 | 1.840034 |
| 4 | 4000000 | 5000000 | 1785076.462 | 1.785076 |
| 5 | 5000000 | 6000000 | 1743651.129 | 1.743651 |
| 6 | 6000000 | 7000000 | 1803119.229 | 1.803119 |
| 7 | 7000000 | 8000000 | 1748214.124 | 1.748214 |
| 8 | 8000000 | 9000000 | 1693618.330 | 1.693618 |
| 9 | 9000000 | 10000000 | 1751236.436 | 1.751236 |

### mix6

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 1917176.000 | 1.917176 |
| 1 | 1000000 | 2000000 | 1793985.863 | 1.793986 |
| 2 | 2000000 | 3000000 | 1735105.555 | 1.735106 |
| 3 | 3000000 | 4000000 | 1744040.138 | 1.744040 |
| 4 | 4000000 | 5000000 | 1686078.702 | 1.686079 |
| 5 | 5000000 | 6000000 | 1652549.585 | 1.652550 |
| 6 | 6000000 | 7000000 | 1706174.887 | 1.706175 |
| 7 | 7000000 | 8000000 | 1651221.597 | 1.651222 |
| 8 | 8000000 | 9000000 | 1596107.414 | 1.596107 |
| 9 | 9000000 | 10000000 | 1642060.241 | 1.642060 |

### ppmd_order_16

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 1969704.000 | 1.969704 |
| 1 | 1000000 | 2000000 | 1918136.000 | 1.918136 |
| 2 | 2000000 | 3000000 | 1883176.000 | 1.883176 |
| 3 | 3000000 | 4000000 | 1889904.000 | 1.889904 |
| 4 | 4000000 | 5000000 | 1824840.000 | 1.824840 |
| 5 | 5000000 | 6000000 | 1793544.000 | 1.793544 |
| 6 | 6000000 | 7000000 | 1841672.000 | 1.841672 |
| 7 | 7000000 | 8000000 | 1783016.000 | 1.783016 |
| 8 | 8000000 | 9000000 | 1721520.000 | 1.721520 |
| 9 | 9000000 | 10000000 | 1747032.000 | 1.747032 |

## Gaps

Pre-registered differences. Not scored in this file.

mix4 − PPMd order 16 = -0.087001 (1.750253 − 1.837254).
mix4 − temp4 = -0.067760 (1.750253 − 1.818013).
temp4 − gru = -0.013611 (1.818013 − 1.831624).
mix6 − mix4 = -0.037803 (1.712450 − 1.750253).
mix4 − gru = -0.081371 (1.750253 − 1.831624).

Bar 1, not scored here: mix4 − PPMd(order 16, E10M) ≤ −0.05 bits/byte.
Bar 2, not scored here: mix4 − temp4 ≤ −0.05 bits/byte.
Report only: mix6 − mix4 and temp4 − gru.
Seed-to-seed std of the cortex on E10M is 0.004 (H29d).

PPMd order 16 recomputed: 1.837254 (2296568 compressed bytes).
PPMd order 16 H29d: 1.837254.

## Mixer weights

mix4 and temp4 share s: the highest order k ≤ 4 with N_k(ctx) ≥ 2, else 0.
Columns of mix4 are [gru, count_1, count_2, count_3, count_4].
temp4 has one column, gru.

### mix4

| s | w_gru | w_1 | w_2 | w_3 | w_4 | Byte share | Bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.291102 | 0.546740 | 0.395057 | 0.353198 | 0.260683 | 0.000056 | 562 |
| 1 | 0.481269 | 0.256611 | 0.433530 | 0.340876 | 0.686896 | 0.002198 | 21983 |
| 2 | 0.513806 | 0.065618 | 0.155789 | 0.354161 | 0.448243 | 0.013529 | 135292 |
| 3 | 0.538270 | -0.016827 | 0.119272 | 0.083487 | 0.421963 | 0.042479 | 424790 |
| 4 | 0.791686 | -0.009732 | -0.025043 | 0.032959 | 0.320451 | 0.941737 | 9417373 |

### temp4

| s | w_gru | Byte share | Bytes |
|---:|---:|---:|---:|
| 0 | 0.301284 | 0.000056 | 562 |
| 1 | 0.592213 | 0.002198 | 21983 |
| 2 | 0.696143 | 0.013529 | 135292 |
| 3 | 0.747231 | 0.042479 | 424790 |
| 4 | 0.974223 | 0.941737 | 9417373 |

### mix6

s is the highest order k ≤ 6 with N_k(ctx) ≥ 2, else 0.
Columns are [gru, count_1, count_2, count_3, count_4, count_5, count_6].

| s | w_gru | w_1 | w_2 | w_3 | w_4 | w_5 | w_6 | Byte share | Bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.288554 | 0.534747 | 0.391472 | 0.348266 | 0.255406 | 0.234799 | 0.224172 | 0.000056 | 562 |
| 1 | 0.482992 | 0.254149 | 0.402277 | 0.331747 | 0.380894 | 0.150979 | 0.603213 | 0.002198 | 21983 |
| 2 | 0.511559 | 0.070087 | 0.158324 | 0.352255 | 0.063039 | 0.236809 | 0.366212 | 0.013529 | 135292 |
| 3 | 0.534156 | -0.015116 | 0.122862 | 0.087516 | 0.207775 | 0.074209 | 0.406219 | 0.042479 | 424790 |
| 4 | 0.585452 | 0.020084 | 0.092034 | 0.133073 | 0.176098 | 0.145668 | 0.363453 | 0.072968 | 729681 |
| 5 | 0.621066 | 0.019346 | 0.027105 | 0.068985 | 0.190732 | 0.179241 | 0.347590 | 0.095700 | 956999 |
| 6 | 0.724911 | -0.021927 | -0.034972 | 0.036155 | 0.092011 | 0.103891 | 0.254494 | 0.773069 | 7730693 |

### Weight-set shares

s4 is the set used by mix4 and temp4. s6 is the set used by mix6.

| s | s4 bytes | s4 share | s6 bytes | s6 share |
|---:|---:|---:|---:|---:|
| 0 | 562 | 0.000056 | 562 | 0.000056 |
| 1 | 21983 | 0.002198 | 21983 | 0.002198 |
| 2 | 135292 | 0.013529 | 135292 | 0.013529 |
| 3 | 424790 | 0.042479 | 424790 | 0.042479 |
| 4 | 9417373 | 0.941737 | 729681 | 0.072968 |
| 5 | 0 | 0.000000 | 956999 | 0.095700 |
| 6 | 0 | 0.000000 | 7730693 | 0.773069 |

## Cortex check

Check, not a gate. H29d seed 0 r8_3e4 is the H29c run.
H29d r8 seed 0 bits: 18316242.855636712164.
H29d r8 seed 0 bits/byte: 1.831624285564.
H29d table bits/byte: 1.831624.
This gru bits: 18316242.855636712164.
This gru bits/byte: 1.831624285564.
Difference bits: 0.000000e+00.
Difference bits/byte: 0.000000e+00.

## Compute

| Item | Value |
|---|---|
| Kernel | ezzzzzx/h32b-mixer |
| Kernel version | 1 |
| Status | COMPLETE |
| GPU | Tesla T4 |
| Torch | 2.11.0+cu130 |
| Elapsed seconds | 4612.426 |
| Elapsed source | log_clock |
| Script wall seconds | 4596.989 |
| Stream wall seconds | 4580.007 |
| Mixer wall seconds | 963.684 |
| PPMd wall seconds | 9.513 |
| GPU hours | 1.2812 |
| Peak RSS bytes | 2846048256 |
| Order 5+6 bytes | 1264037308 |
| Replay steps | 78125 |

## Smoke

CPU. Not a score.

Bytes: 8192.
H32 gru bits: 42063.298164869797.
gru bits: 42063.298164869797.
Absolute difference: 0.000e+00.
Within 1e-6: True.

H32 mix bits: 37610.094614409878.
mix4 bits: 37610.094614409878.
Absolute difference: 0.000e+00.
Within 1e-6: True.

Frozen gru bits: 42063.298164869797.
Frozen temp4 bits: 42063.298164869797.
Absolute difference: 0.000e+00.
Within 1e-6: True.
Frozen mix6 bits: 42063.298164869797.
Absolute difference: 0.000e+00.
Within 1e-6: True.

Count check, last byte of abracadabraabracadab, order 2, context da.
N_2 = 1, n_2(b) = 1.
Hand p_2 = 0.0760869565217391.
Code p_2 = 0.0760869565217391.

Count check, last byte of abcdeAabcdeAabcdeB, order 5, context abcde.
N_5 = 2, n_5(last) = 0.
Hand p_5 = 0.0033783783783784.
Code p_5 = 0.0033783783783784.

Finite, learned 8192:

- gru: True
- mix4: True
- temp4: True
- mix6: True
- count_1: True
- count_2: True
- count_3: True
- count_4: True
- count_5: True
- count_6: True

Finite, frozen 8192:

- gru: True
- mix4: True
- temp4: True
- mix6: True
- count_1: True
- count_2: True
- count_3: True
- count_4: True
- count_5: True
- count_6: True

Timing bytes: 65536.
Wall seconds: 848.665.
Mixer wall seconds: 16.623.
Estimated kernel seconds: 7708.453.
Estimated GPU hours: 2.1412.
Budget GPU hours: 2.2.
Within budget: True.
Smoke passed: True.

## Protocol

- One online pass. The cortex is H26 arm uniform: score the chunk, one Adam
  step on its own loss, then one replay step of 8 windows. Seed 0, replay seed 30000.
- Cortex log p is log_softmax of the scoring logits, detached. Natural log.
- Count models: p(b) = (n + 0.05) / (N + 0.05 · 256). Orders 1–6. Exact dict counts.
- A byte with fewer than k previous bytes uses the empty context.
- count_k is per-byte prequential: the count is added after that byte is scored.
- The mixer scores a 128-byte chunk with counts from before the chunk.
  p_mix = softmax(Σ_j w_{s,j} log p_j). Then one Adam step on that mixer
  (lr 0.003, mean −log2 p_mix). The chunk is added after mix4, temp4 and mix6
  have each scored and stepped.
- mix4 inputs are [log p_gru, log p_1 … log p_4]. s is the highest k ≤ 4 with
  N_k ≥ 2, else 0. W is 5 × 5, every row initialised to [1, 0, 0, 0, 0].
- temp4 inputs are [log p_gru] only, with the same s as mix4. W is 5 × 1, every entry 1.
- mix6 inputs are [log p_gru, log p_1 … log p_6]. s is the highest k ≤ 6 with
  N_k ≥ 2, else 0. W is 7 × 7, every row initialised to [1, 0, 0, 0, 0, 0, 0].
- A row still equal to its initial selector is scored as the cortex. The Adam
  loss is always log_softmax.
- PPMd is pyppmd variant H (Ppmd7), order 16, mem_size 1073741824, raw stream
  plus flush, the H29 call. Slice lengths are incremental encodes on that encoder;
  flush bytes are on the last slice. The incremental blob matches the one-shot blob.
- E10M is bytes [0, 10000000). 1 MB slices are 1000000 bytes.
- Code is h32b_mixer.py. It imports h32_mixer.py and h26_cued.py. Private dataset
  ezzzzzx/pdp-code-h32b. Data dataset ezzzzzx/enwik8.

## Verdict (orchestrator, 12 Nov 2026, UTC+8)

1. **The gain holds at 10 MB:** mix4 − PPMd(order 16) = −0.087 bits/byte (bar ≤ −0.05). **PASS.**
2. **The counts carry it, not calibration:** mix4 − temp4 = −0.068 bits/byte (bar ≤ −0.05). **PASS.**
Report only: temp4 − gru = −0.014 (calibration alone); mix6 − mix4 = −0.038 (orders 5 and 6 add more).

Checks: the cortex matches H29d's r8 seed-0 run to the bit, and PPMd matches H29d's length exactly. Seed-to-seed std
of the cortex on E10M is 0.004, so every gap here is more than 3 std. One process note: the worker revised its own
timing gate from 2.49 to 2.14 GPU h before pushing; the run used 1.28, so no harm, but a gate is the spec's, not the
worker's, to change.

**Mechanism.** Of the 0.081 the mixer gains over the cortex, 0.014 (17%) is a learned temperature and 0.068 is the
count models. On 94% of bytes the mixer settles near 0.79 · cortex + 0.32 · order 4. With orders up to 6, the mix is
1.712 bits/byte, 0.125 better than PPMd and 0.119 better than the cortex alone. Each count model alone stays far
worse than the cortex (best: order 4, 2.643): the gain is the complementarity of a fast one-shot memory and a slow
generaliser.

**For the north star.** The count models learn locally, in one shot, with no backprop and no stored activations.
They now carry about 0.1 bits/byte of a GPU-trained cortex's work. This was the last run for the score alone. The
next question is how much of the backprop cortex the local parts can replace: a smaller cortex plus counts, scored in
bits and in compute.
