# W1c shared cortex

One cortex, seven channels. Weights are shared. Each channel has its own hidden state.

## Streams

| Channel | Stream bytes | Stream sha256 | Train bytes | Train sha256 | Held-out bytes | Held-out sha256 |
|---|---:|---|---:|---|---:|---|
| bible | 4432490 | `ef0b6942e0ac1174072d3e4420cd5e05aca97fe86af2088b986c5a6f493f77a6` | 2000000 | `fa9eb395ac956821a48432d7337ee35e78d7611dbdc75240445e945700737642` | 100000 | `de2429965badbd30dc4eac76f4df2fc843e5ff96e3bbfafd05d1fee608a0dde5` |
| chess | 5027572 | `7791c438b0f21b8f8ec6b3b1f421eaa29d526647ed21eaddc2da9795c29d5b38` | 2000000 | `4986e78f72b9cc7fe1fa85229c56c7359df8afaa7aa426df2be0614cf723ef0b` | 100000 | `c0b520a6b4b43e22db26c1555e288bac280e5fc947ae987d1ecc42176ee23111` |
| squad | 2466240 | `1348449a968759680bb85a2803250a8b904d1c2a572d6ddb9ed757a2eafd1cb0` | 2000000 | `42003a1ff6ab605a2f0afe8080e233718cc0b08be61b3b155ec5f647e1c00a0a` | 100000 | `fb97f239566374dbea842207371fa2477938b9cbedf6d2f0283b7878586d1c32` |
| btc | 6606836 | `095ee45ed97240a12e64cb2e038e04eef537da520851b1be4f05e1eee91395b9` | 2000000 | `3ba760b1a58801976c68e834954a16f737e4cfcac1132b78fc667a80efbb713a` | 100000 | `4ba57af0b7dc6b1bb237bd9ddf6d0249496b19e10a9cc803c29e2281e8263a51` |
| dna | 5538942 | `e42a27d03e6b6248fa4c34bcb0cd2c68fd96e28600971edd0d9fb585fbdbbba7` | 2000000 | `449c7a12825c53e0a5343c4eb05467245681ac8feb94687f8c19f18ad71bf229` | 100000 | `381022715867364fda2ced39782996096abb085a26bcb4f38b1e06228ec8b0c3` |
| noise | 2100000 | `ee85587d25b9ad4044d603b2a87a46fc07c3341bfd61bcba287623e6a6be4e2e` | 2000000 | `1e1bf139cca67a5858d2662b1f0639fefbf988c4f18bb0ff1ffdf22033c8c3f9` | 100000 | `c40fe11aec22ef630945c1ce6a17f8f7c8814bdc06c5509258dd803e95102b08` |
| dark | 2100000 | `c2f2ff9f44898670c8a5020b4987f78845bcfce6f6678594c8559a64c65346ed` | 2000000 | `13aea96040f2133033d103008d5d96cfe98b3361f7202d77bea97b2424a7a6cd` | 100000 | `9192c25b734fcbadbe32dadc28089c60db0e39f90cc20ce2e5733f57261acc0c` |

## Quotas

| Arm | bible | chess | squad | btc | dna | noise | dark | Sum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U7 | 500000 | 500000 | 500000 | 500000 | 500000 | 500000 | 500000 | 3500000 |
| N0 | 583333 | 583333 | 583334 | 583333 | 583333 | 0 | 583334 | 3500000 |
| OR | 1000000 | 1000000 | 1000000 | 250000 | 125000 | 0 | 125000 | 3500000 |

## U7

### Bytes read

| Channel | Quota | Bytes read |
|---|---:|---:|
| bible | 500000 | 500000 |
| chess | 500000 | 500000 |
| squad | 500000 | 500000 |
| btc | 500000 | 500000 |
| dna | 500000 | 500000 |
| noise | 500000 | 500000 |
| dark | 500000 | 500000 |

### Prequential

Total bits 8596866.693281. Bits/byte 2.456248. Replay steps 27349. Traffic 8.001536. Chunks 27349. Run seconds 1411.586.

| Channel | Bytes | Bits | Bits/byte |
|---|---:|---:|---:|
| bible | 500000 | 813044.984304 | 1.626090 |
| chess | 500000 | 906726.732775 | 1.813453 |
| squad | 500000 | 1100344.519559 | 2.200689 |
| btc | 500000 | 868875.951412 | 1.737752 |
| dna | 500000 | 816078.026887 | 1.632156 |
| noise | 500000 | 4090289.038325 | 8.180578 |
| dark | 500000 | 1507.440018 | 0.003015 |

### Held-out L_c, bits/byte

| n | bible | chess | squad | btc | dna | noise | dark | M7 | learnable5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 875000 | 2.429362 | 1.864389 | 3.582022 | 1.923939 | 3.058909 | 8.312949 | 0.000086 | 3.024522 | 2.571724 |
| 1750000 | 2.321536 | 1.820041 | 3.429903 | 1.875704 | 2.980405 | 8.446814 | 0.000082 | 2.982069 | 2.485518 |
| 3500000 | 2.208598 | 1.789992 | 3.081085 | 1.843867 | 2.170288 | 8.384786 | 0.000091 | 2.782672 | 2.218766 |

Held-out bits:

| n | bible | chess | squad | btc | dna | noise | dark |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 875000 | 242936.208 | 186438.895 | 358202.174 | 192393.853 | 305890.859 | 831294.940 | 8.602 |
| 1750000 | 232153.560 | 182004.117 | 342990.337 | 187570.390 | 298040.536 | 844681.355 | 8.165 |
| 3500000 | 220859.773 | 178999.166 | 308108.518 | 184386.660 | 217028.845 | 838478.608 | 9.059 |

## N0

### Bytes read

| Channel | Quota | Bytes read |
|---|---:|---:|
| bible | 583333 | 583333 |
| chess | 583333 | 583333 |
| squad | 583334 | 583334 |
| btc | 583333 | 583333 |
| dna | 583333 | 583333 |
| noise | 0 | 0 |
| dark | 583334 | 583334 |

### Prequential

Total bits 5168056.909310. Bits/byte 1.476588. Replay steps 27348. Traffic 8.001243. Chunks 27348. Run seconds 1446.304.

| Channel | Bytes | Bits | Bits/byte |
|---|---:|---:|---:|
| bible | 583333 | 898576.742436 | 1.540418 |
| chess | 583333 | 1047119.642090 | 1.795063 |
| squad | 583334 | 1268450.895029 | 2.174485 |
| btc | 583333 | 997980.184724 | 1.710824 |
| dna | 583333 | 953736.879905 | 1.634978 |
| noise | 0 | 0.000000 |  |
| dark | 583334 | 2192.565126 | 0.003759 |

### Held-out L_c, bits/byte

| n | bible | chess | squad | btc | dna | noise | dark | M7 | learnable5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 875000 | 2.393935 | 1.857072 | 3.540047 | 1.927409 | 3.281833 | 18.466379 | 0.000081 | 4.495251 | 2.600060 |
| 1750000 | 2.264674 | 1.807742 | 3.349016 | 1.869294 | 3.050097 | 20.902370 | 0.000106 | 4.749043 | 2.468165 |
| 3500000 | 2.194527 | 1.779179 | 3.194564 | 1.817644 | 2.851855 | 21.572233 | 0.000129 | 4.772876 | 2.367554 |

Held-out bits:

| n | bible | chess | squad | btc | dna | noise | dark |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 875000 | 239393.544 | 185707.225 | 354004.714 | 192740.944 | 328183.324 | 1846637.867 | 8.053 |
| 1750000 | 226467.386 | 180774.194 | 334901.650 | 186929.446 | 305009.677 | 2090237.026 | 10.563 |
| 3500000 | 219452.676 | 177917.858 | 319456.409 | 181764.352 | 285185.477 | 2157223.275 | 12.865 |

## OR

### Bytes read

| Channel | Quota | Bytes read |
|---|---:|---:|
| bible | 1000000 | 1000000 |
| chess | 1000000 | 1000000 |
| squad | 1000000 | 1000000 |
| btc | 250000 | 250000 |
| dna | 125000 | 125000 |
| noise | 0 | 0 |
| dark | 125000 | 125000 |

### Prequential

Total bits 5906233.491679. Bits/byte 1.687495. Replay steps 27347. Traffic 8.000951. Chunks 27347. Run seconds 1430.982.

| Channel | Bytes | Bits | Bits/byte |
|---|---:|---:|---:|
| bible | 1000000 | 1453595.718631 | 1.453596 |
| chess | 1000000 | 1736098.332883 | 1.736098 |
| squad | 1000000 | 2038749.360465 | 2.038749 |
| btc | 250000 | 466935.860973 | 1.867743 |
| dna | 125000 | 209798.723470 | 1.678390 |
| noise | 0 | 0.000000 |  |
| dark | 125000 | 1055.495256 | 0.008444 |

### Held-out L_c, bits/byte

| n | bible | chess | squad | btc | dna | noise | dark | M7 | learnable5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 875000 | 2.233966 | 1.785114 | 3.291181 | 2.039383 | 3.193963 | 17.665715 | 0.000084 | 4.315630 | 2.508722 |
| 1750000 | 2.129383 | 1.746220 | 3.032516 | 1.996466 | 2.855014 | 19.586983 | 0.000108 | 4.478099 | 2.351920 |
| 3500000 | 2.005808 | 1.701802 | 2.773464 | 1.941298 | 3.095523 | 21.202893 | 0.000135 | 4.674418 | 2.303579 |

Held-out bits:

| n | bible | chess | squad | btc | dna | noise | dark |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 875000 | 223396.637 | 178511.368 | 329118.127 | 203938.346 | 319396.324 | 1766571.517 | 8.391 |
| 1750000 | 212938.314 | 174621.987 | 303251.597 | 199646.572 | 285501.425 | 1958698.347 | 10.764 |
| 3500000 | 200580.836 | 170180.249 | 277346.413 | 194129.823 | 309552.320 | 2120289.341 | 13.539 |

## Headroom

At 3500000 bytes.

| Arm | M7 | learnable5 |
|---|---:|---:|
| U7 | 2.782672 | 2.218766 |
| N0 | 4.772876 | 2.367554 |
| OR | 4.674418 | 2.303579 |

Minimum of M7(N0) and M7(OR): 4.674418 (OR).
headroom_shared = M7(U7) − min(M7(N0), M7(OR)) = -1.891746.
M7(U7) − M7(N0) = -1.990203.

## Kernels

Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that
clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.
Script wall is time.monotonic from kernel_main entry through the result write. Queue time
is not included.

| Kernel | Arm | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours | GPU |
|---|---|---:|---|---:|---|---:|---:|---|
| ezzzzzx/w1c-u7 | U7 | 1 | COMPLETE | 1428.597 | log_clock | 1413.560 | 0.3968 | Tesla T4 |
| ezzzzzx/w1c-n0 | N0 | 1 | COMPLETE | 1464.949 | log_clock | 1448.899 | 0.4069 | Tesla T4 |
| ezzzzzx/w1c-or | OR | 1 | COMPLETE | 1447.924 | log_clock | 1433.018 | 0.4022 | Tesla T4 |

Total GPU hours: 1.2060.

## Smoke

CPU. U7 budget 3584, quota 512 per channel, held-out cut to 1024. N0 is the N0 quota row scaled to the same budget by largest remainder.

U7 bytes read: bible=512, chess=512, squad=512, btc=512, dna=512, noise=512, dark=512.
U7 finite: True. U7 bits: 21699.655499. U7 traffic: 8.000000. U7 replay outside read: 0.
N0 noise bytes: 0. N0 finite: True. N0 bits: 19896.300564. N0 replay outside read: 0.
Smoke wall seconds: 131.148.

## Protocol

- One H29c cortex per arm. ByteGRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam lr 0.0003, clip 1. 3627264 parameters. Seed 0. Replay RNG 30000.
- Chunks of 128. A chunk is min(128, remaining quota, bytes until the next checkpoint). Score the chunk (live −log2 p) before its Adam step. Detach the channel hidden state at the chunk boundary. Weights are shared. Hidden state is per channel, zero at the channel's first byte.
- Before each chunk, read the channel with the smallest (bytes read) / quota among channels with quota > 0 and bytes still remaining. Ties go to the lowest index in bible, chess, squad, btc, dna, noise, dark.
- After each live chunk, one Adam step on a batch of 8 windows of 128 bytes, zero initial state. The channel is drawn with probability proportional to bytes read so far. The start is uniform on 0 .. bytes_read−128 inclusive. A window that would leave the read prefix raises.
- Held-out score at 875000, 1750000 and 3500000 total bytes. No gradient, zero initial state, chunks of 128, state carried. L_c is bits / held-out bytes. M7 is the mean of the 7 L_c values. learnable5 is the mean of bible, chess, squad, btc and dna.
- headroom_shared = M7(U7) − min(M7(N0), M7(OR)) at 3500000 bytes. M7(U7) − M7(N0) is the noisy-TV comparison.
- Training bytes are the first 2000000 of each W1b stream. Held-out bytes are the next 100000. Noise is numpy.random.default_rng(777).integers(0, 256, size=2100000, dtype=uint8). Dark is 2100000 zero bytes. DNA is the sequence column of human.txt.
- Quotas: U7 is 500000 on every channel. N0 is 0 on noise and the remaining 3500000 split across the other six (583333, 583333, 583334, 583333, 583333, 583334). OR is 1000000, 1000000, 1000000, 250000, 125000, 0, 125000.

## Verdict (orchestrator, 11 Nov 2026, 13:30 UTC+8)

**Bar (headroom_shared ≥ 0.10): FAIL.** headroom_shared = −1.89: uniform U7 beats both alternatives. The
orchestrator's written prediction (0.03 to 0.10) was wrong in sign.

The size of that number is a metric artefact. M7 averages in the noise channel's held-out loss. A shared cortex that
never read noise is confidently wrong on it: 21.6 bits/byte for N0, against 8.38 for U7 and 8.0 for knowing nothing.
That single term accounts for the −1.99. The fair comparison is the learnable channels:
- **learnable5:** U7 2.219, OR 2.304, N0 2.368. Uniform is still best.
- **Excluding DNA, whose learning is erratic (2.17 vs 2.85 on one seed):** U7 2.231 vs N0 2.247. N0 is better on bible,
  chess and btc, where it read 17% more bytes, and worse on squad.

Mechanism.
1. In a shared cortex, reading the noisy TV did **not** measurably damage the learnable channels at this budget. The
   interference I expected is absent. The cost of the TV is opportunity, as with separate learners.
2. **Knowing that something is unpredictable is itself knowledge.** A model that never samples noise mis-predicts it
   by 13 bits/byte. So a well-designed curious agent should visit the TV briefly, learn that it is random (progress
   then falls to zero), and leave. Schmidhuber's principle predicts that, and an agent that avoids noise entirely is
   worse calibrated.
3. Lesson for metrics: a probe must not reward never looking at a region, nor punish it unboundedly. W2's probe keeps
   the TV out of the mean, and will report a capped variant.

The breadth-world line (W1, W1b, W1c) is closed as a curiosity test. M4 waits for W2. Single seed per arm.
