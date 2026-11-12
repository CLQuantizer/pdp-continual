# H32 cerebellar mixer on enwik8 E3M

E3M bytes: 3000000. sha256: `72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac`.
GRU bits/byte = sum of live −log2 p / 3000000. The mixer and the count
models are scored the same way. PPMd bits/byte = 8 × compressed bytes / 3000000.
Replay is not part of the code length. The mixer gradient does not enter the cortex.

Cortex: H26 uniform, lr 0.0003, seed 0, replay seed 30000, hidden 1024,
chunk 128, replay batch 8, sleep block 1. Mixer: Adam lr 0.003, one step
per chunk, W initialised to [1, 0, 0, 0, 0] on every row. Count smoothing 0.05.

## E3M

| Arm | Bits/byte | Bits |
|---|---:|---:|
| gru | 1.958153 | 5874459.001 |
| count_1 | 3.876779 | 11630336.268 |
| count_2 | 3.151852 | 9455557.022 |
| count_3 | 2.810473 | 8431418.347 |
| count_4 | 2.956180 | 8868540.958 |
| mix | 1.846412 | 5539237.045 |
| ppmd_order_16 | 1.923685 | 5771056.000 |

## 1 MB slices

### gru

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2104744.492 | 2.104744 |
| 1 | 1000000 | 2000000 | 1923321.387 | 1.923321 |
| 2 | 2000000 | 3000000 | 1846393.123 | 1.846393 |

### count_1

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 3874483.318 | 3.874483 |
| 1 | 1000000 | 2000000 | 3880345.051 | 3.880345 |
| 2 | 2000000 | 3000000 | 3875507.899 | 3.875508 |

### count_2

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 3166793.210 | 3.166793 |
| 1 | 1000000 | 2000000 | 3148434.798 | 3.148435 |
| 2 | 2000000 | 3000000 | 3140329.014 | 3.140329 |

### count_3

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2961961.115 | 2.961961 |
| 1 | 1000000 | 2000000 | 2773872.038 | 2.773872 |
| 2 | 2000000 | 3000000 | 2695585.193 | 2.695585 |

### count_4

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 3297434.917 | 3.297435 |
| 1 | 1000000 | 2000000 | 2876494.486 | 2.876494 |
| 2 | 2000000 | 3000000 | 2694611.554 | 2.694612 |

### mix

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 1951635.909 | 1.951636 |
| 1 | 1000000 | 2000000 | 1824186.192 | 1.824186 |
| 2 | 2000000 | 3000000 | 1763414.944 | 1.763415 |

### ppmd_order_16

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 1969704.000 | 1.969704 |
| 1 | 1000000 | 2000000 | 1918136.000 | 1.918136 |
| 2 | 2000000 | 3000000 | 1883216.000 | 1.883216 |

## Gaps

Pre-registered differences. Not scored in this file.

mix − gru = -0.111741 (1.846412 − 1.958153).
mix − PPMd order 16 = -0.077273 (1.846412 − 1.923685).

Bar 1, not scored here: mix − gru ≤ −0.05 bits/byte on E3M.
Bar 2, not scored here: mix − PPMd(order 16, E3M) ≤ −0.05 bits/byte.

Best single count order: 3 at 2.810473 bits/byte.

## Mixer weights

Rows are weight sets s = 0..4. Columns are experts [gru, count_1, count_2, count_3, count_4].

| s | w_gru | w_1 | w_2 | w_3 | w_4 | Byte share | Bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.291102 | 0.546740 | 0.395057 | 0.353198 | 0.260683 | 0.000187 | 562 |
| 1 | 0.463475 | 0.291837 | 0.466569 | 0.273819 | 0.453305 | 0.005082 | 15247 |
| 2 | 0.431789 | 0.164648 | 0.246106 | 0.289557 | 0.511453 | 0.025428 | 76285 |
| 3 | 0.457138 | 0.124771 | 0.142128 | 0.131386 | 0.477860 | 0.066628 | 199885 |
| 4 | 0.649131 | 0.001390 | 0.015730 | 0.042377 | 0.247996 | 0.902674 | 2708021 |

## Cortex check

H26 uniform E3M bits: 5874459.001424252987.
This gru bits: 5874459.001424252987.
Difference: 0.000000e+00.

## Compute

| Item | Value |
|---|---|
| Kernel | ezzzzzx/h32-mixer |
| Kernel version | 2 |
| Status | COMPLETE |
| GPU | Tesla T4 |
| Torch | 2.11.0+cu130 |
| Elapsed seconds | 1344.241 |
| Elapsed source | log_clock |
| Script wall seconds | 1327.217 |
| Stream wall seconds | 1317.900 |
| PPMd wall seconds | 2.268 |
| GPU hours | 0.3734 |
| Replay steps | 23438 |

## Smoke

CPU. Not a score.

Bytes: 8192.
H26 uniform bits: 42063.298164869797.
gru bits: 42063.298164869797.
Absolute difference: 0.000e+00.
Within 1e-6: True.

Frozen mix bits: 42063.298164869797.
Absolute difference versus gru: 0.000e+00.
Bits/byte difference: 0.000e+00.
Within 1e-6: True.

Count check, last byte of abracadabraabracadab, order 2, context da.
N_2 = 1, n_2(b) = 1.
Hand p_2 = 0.0760869565217391.
Code p_2 = 0.0760869565217391.

Finite:

- gru: True
- mix: True
- count_1: True
- count_2: True
- count_3: True
- count_4: True
- mix_frozen: True
- gru_frozen: True

## Protocol

- One online pass. The cortex is H26 arm uniform: score the chunk, one Adam
  step on its own loss, then one replay step of 8 windows. Seed 0, replay seed 30000.
- Cortex log p is log_softmax of the scoring logits, detached. Natural log.
- Count models: p(b) = (n + 0.05) / (N + 0.05 · 256). Orders 1–4. Exact dict counts.
- A byte with fewer than k previous bytes uses the empty context.
- count_k is per-byte prequential: the count is added after that byte is scored.
- The mixer scores a 128-byte chunk with counts from before the chunk, with
  p_mix = softmax(Σ_j w_{s,j} log p_j). s is the highest order whose context
  has N ≥ 2, else 0. Then one Adam step on W (lr 0.003, mean −log2 p_mix).
  Then the chunk is added to the counts.
- W is 5 × 5, every row initialised to [1, 0, 0, 0, 0].
- PPMd is pyppmd variant H (Ppmd7), order 16, mem_size 1073741824, raw stream
  plus flush, the H29 call. Slice lengths are incremental encodes on that encoder;
  flush bytes are on the last slice. The incremental blob matches the one-shot blob.
- E3M is bytes [0, 3000000). 1 MB slices are 1000000 bytes.
- Code is h32_mixer.py. Private dataset ezzzzzx/pdp-code-h32. Data dataset ezzzzzx/enwik8.

## Verdict (orchestrator, 12 Nov 2026)

1. **The fast learner helps the cortex:** mix − gru = −0.112 bits/byte (bar ≤ −0.05). **PASS.**
2. **Beats PPMd clearly:** mix − PPMd(order 16) = −0.077 bits/byte (bar ≤ −0.05). **PASS.**
Report only: the best single count order is 3 (2.810 bits/byte). Every count model alone is far worse than the cortex.

Checks: the cortex reproduces H26 `uniform` to the bit; the frozen mixer equals the cortex; the code scores each chunk
before the Adam step and adds counts after it (spot-checked `consume`). The score is prequential.

**Mechanism.** Two predictors that are each worse than PPMd (the cortex at +0.034, the best count model at +0.89) beat it
by 0.077 when mixed. The cortex generalises slowly; the count models remember an exact context after one sighting. On
90% of bytes (weight set 4: the order-4 context was seen at least twice) the mixer settles on 0.65 · cortex + 0.25 ·
order 4. Where contexts are rare (sets 0–2) it leans on the low orders. The gap to PPMd grows with data: −0.018 in the
first MB, −0.094 in the second, −0.120 in the third.

**Caveat.** The rows of W are not normalised: the row for set 4 sums to 0.96, the rows for sets 0–2 to 1.3–1.8, so part
of the gain may be calibration (a learned temperature on the cortex) rather than the counts. H32b separates the two with
a cortex-only mixer arm, and checks the result on E10M.
