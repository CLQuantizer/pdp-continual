# W1b headroom

These are separate learners per channel. The real machine shares one cortex, so this is an estimate.

Bible, chess, squad and btc are the W1-0 runs. They were not rerun.

## New streams

| Channel | Source bytes | Source sha256 | Stream bytes | Stream sha256 | Split | Train bytes | Train sha256 | Held-out bytes | Held-out sha256 |
|---|---:|---|---:|---|---|---:|---|---:|---|
| dna | 5547716 | `82794876b4687104f431142f60c0e5ef2c1903ff0f95ddf71be4043f3e5b506f` | 5538942 | `e42a27d03e6b6248fa4c34bcb0cd2c68fd96e28600971edd0d9fb585fbdbbba7` | prefix_2000000_100000 | 2000000 | `449c7a12825c53e0a5343c4eb05467245681ac8feb94687f8c19f18ad71bf229` | 100000 | `381022715867364fda2ced39782996096abb085a26bcb4f38b1e06228ec8b0c3` |
| noise | 2100000 | `ee85587d25b9ad4044d603b2a87a46fc07c3341bfd61bcba287623e6a6be4e2e` | 2100000 | `ee85587d25b9ad4044d603b2a87a46fc07c3341bfd61bcba287623e6a6be4e2e` | prefix_2000000_100000 | 2000000 | `1e1bf139cca67a5858d2662b1f0639fefbf988c4f18bb0ff1ffdf22033c8c3f9` | 100000 | `c40fe11aec22ef630945c1ce6a17f8f7c8814bdc06c5509258dd803e95102b08` |
| dark | 2100000 | `c2f2ff9f44898670c8a5020b4987f78845bcfce6f6678594c8559a64c65346ed` | 2100000 | `c2f2ff9f44898670c8a5020b4987f78845bcfce6f6678594c8559a64c65346ed` | prefix_2000000_100000 | 2000000 | `13aea96040f2133033d103008d5d96cfe98b3361f7202d77bea97b2424a7a6cd` | 100000 | `9192c25b734fcbadbe32dadc28089c60db0e39f90cc20ce2e5733f57261acc0c` |

Noise and dark are synthetic. Their source bytes are the stream.

## Excerpt

### dna

300 bytes.

```
ATGCCCCAACTAAATACTACCGTATGGCCCACCATAATTACCCCCATACTCCTTACACTATTCCTCATCACCCAACTAAAAATATTAAACACAAACTACCACCTACCTCCCTCACCAAAGCCCATAAAAATAAAAAATTATAACAAACCCTGAGAACCAAAATGAACGAAAATCTGTTCGCTTCATTCATTGCCCCCACAATCCTAG
ATGAACGAAAATCTGTTCGCTTCATTCATTGCCCCCACAATCCTAGGCCTACCCGCCGCAGTACTGATCATTCTATTTCCCCCTCTATTGAT
```

## Held-out L_c(n), bits/byte

| n | bible | chess | squad | btc | dna | noise | dark |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.977886 | 8.016386 | 7.980837 | 8.011427 | 8.035820 | 8.001894 | 7.875209 |
| 62500 | 2.642834 | 1.923204 | 3.643461 | 2.079862 | 1.929631 | 8.117100 | 0.000077 |
| 125000 | 2.366760 | 1.822484 | 3.378627 | 1.883045 | 1.927727 | 8.107212 | 0.000077 |
| 250000 | 2.200340 | 1.735391 | 3.172959 | 1.780715 | 1.915025 | 8.106891 | 0.000076 |
| 500000 | 2.116726 | 1.670179 | 2.971317 | 1.735002 | 1.892069 | 8.149331 | 0.000076 |
| 1000000 | 1.958147 | 1.596832 | 2.702754 | 1.701187 | 1.903767 | 8.283232 | 0.000075 |
| 2000000 | 1.805729 | 1.531396 | 2.600601 | 1.615324 | 1.901709 | 8.338520 | 0.000074 |

Held-out bits:

| n | bible | chess | squad | btc | dna | noise | dark |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 797788.642 | 801638.554 | 798083.650 | 801142.736 | 803582.012 | 800189.399 | 787520.864 |
| 62500 | 264283.439 | 192320.440 | 364346.121 | 207986.182 | 192963.133 | 811709.997 | 7.706 |
| 125000 | 236676.007 | 182248.401 | 337862.732 | 188304.545 | 192772.679 | 810721.186 | 7.669 |
| 250000 | 220034.030 | 173539.119 | 317295.896 | 178071.529 | 191502.473 | 810689.073 | 7.630 |
| 500000 | 211672.593 | 167017.912 | 297131.656 | 173500.158 | 189206.944 | 814933.089 | 7.572 |
| 1000000 | 195814.709 | 159683.182 | 270275.363 | 170118.728 | 190376.672 | 828323.174 | 7.469 |
| 2000000 | 180572.885 | 153139.556 | 260060.103 | 161532.363 | 190170.946 | 833852.021 | 7.410 |

## Prequential 1 MB slices

### dna

Bits 3732704.285486. Bits/byte 1.866352. Replay steps 15629. Traffic 8.002048. Wall seconds 771.529.

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 1880461.365 | 1.880461 |
| 1 | 1000000 | 2000000 | 1852242.921 | 1.852243 |

### noise

Bits 16505004.575346. Bits/byte 8.252502. Replay steps 15629. Traffic 8.002048. Wall seconds 762.036.

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 8167607.825 | 8.167608 |
| 1 | 1000000 | 2000000 | 8337396.750 | 8.337397 |

### dark

Bits 2232.902719. Bits/byte 0.001116. Replay steps 15629. Traffic 8.002048. Wall seconds 761.699.

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2232.903 | 0.002233 |
| 1 | 1000000 | 2000000 | 0.000 | 0.000000 |

## xz -9e and PPMd order 16

Training bytes of the new channels. xz is `xz -9e -T1`. PPMd is variant H, order 16, mem 1073741824.
PPMd is encoded in pieces of 65536 bytes on one encoder, then flushed.

| Channel | xz bytes | xz bits/byte | PPMd bytes | PPMd bits/byte |
|---|---:|---:|---:|---:|
| dna | 163436 | 0.653744 | 155730 | 0.622920 |
| noise | 2000160 | 8.000640 | 2043066 | 8.172264 |
| dark | 416 | 0.001664 | 742 | 0.002968 |

## Headroom

### full

Channels: bible, chess, squad, btc, dna, noise, dark.
Budget 3500000 bytes. Grid [0, 62500, 125000, 250000, 500000, 1000000, 2000000]. Feasible allocations 365573 of 823543.

Uniform mean L_c(500000): 2.647814 bits/byte.
Oracle mean: 2.566878 bits/byte.
Headroom (uniform − oracle): 0.080936 bits/byte.

Oracle allocations: 1.

| bible | chess | squad | btc | dna | noise | dark | Sum | Mean bits/byte |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000000 | 1000000 | 1000000 | 250000 | 125000 | 0 | 125000 | 3500000 | 2.566878 |

### learnable

Channels: bible, chess, squad, btc, dna.
Budget 2500000 bytes. Grid [0, 62500, 125000, 250000, 500000, 1000000, 2000000]. Feasible allocations 7700 of 16807.

Uniform mean L_c(500000): 2.077059 bits/byte.
Oracle mean: 2.037080 bits/byte.
Headroom (uniform − oracle): 0.039979 bits/byte.

Oracle allocations: 1.

| bible | chess | squad | btc | dna | Sum | Mean bits/byte |
|---:|---:|---:|---:|---:|---:|---:|
| 500000 | 500000 | 1000000 | 250000 | 250000 | 2500000 | 2.037080 |

Most of the gain comes from avoiding noise and dark rather than from choosing between learnable streams.

## Kernel

Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that
clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.
Script wall is time.monotonic from kernel_main entry through the result write. Queue time
is not included.

| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours | GPU |
|---|---:|---|---:|---|---:|---:|---|
| ezzzzzx/w1b-headroom | 1 | COMPLETE | 2316.191 | log_clock | 2296.058 | 0.6434 | Tesla T4 |

Total GPU hours: 0.6434.

## Smoke

CPU. Noise and dark: 4096 training bytes, 1024 held-out bytes, checkpoints 0 and 4096.
Dark held-out bits at n=0: 8064.478044.
Dark held-out bits at n=4096: 9.045865.
Dark held-out bits fell: True.
Noise training finite: True. Dark training finite: True.
Bible training bits: 19362.707616529544.
W1-0 smoke bits: 19362.707616529544.
Absolute difference: 0.000e+00.
Within 1e-6: True.

## Protocol

- One fresh learner per channel. lr 0.0003. Replay batch 8. Seed 0. Replay RNG 30000.
- ByteGRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam, clip 1.0. 3627264 parameters.
- Prequential: each chunk is scored before that chunk is trained. Live state is carried and detached.
- Replay state is zero. Replay starts are uniform on 0 .. n_seen−128 inclusive, with replacement.
- The learner advances to each checkpoint with that loop. A checkpoint that is not a multiple of 128 ends on a short chunk, then training continues.
- Held-out evaluation uses no gradient, a zero initial state, chunks of 128, and carried state. L_c(n) is live −log2 p bits divided by held-out bytes.
- The training stream is the first 2000000 bytes. The held-out block is the next 100000 bytes.
- Bible, chess, squad and btc L_c(n) are the W1-0 curves. Those channels were not rerun.
- DNA bytes are the sequence column of human.txt, one sequence per line, in file order. The header and the class labels are dropped. Bases are A, C, G, T and N.
- Noise bytes are numpy.random.default_rng(777).integers(0, 256, size=2100000, dtype=uint8).
- Dark bytes are 2100000 zero bytes.
- xz -9e -T1 and PPMd variant H, order 16, mem 1073741824, on the training bytes of dna, noise and dark. PPMd is encoded in 65536-byte pieces on one encoder, then flushed.
- For a set of k channels the budget is 500000 times k bytes. Each n_c is on the checkpoint grid and the sum of n_c is at most the budget.
- Uniform is the mean of L_c(500000). Oracle is the minimum mean of L_c(n_c) over that grid, by exhaustive search of the 7^k tuples. Headroom is uniform minus oracle.
- Full set: bible, chess, squad, btc, dna, noise, dark. Learnable set: bible, chess, squad, btc, dna.

## Verdict (orchestrator, 11 Nov 2026, 12:10 UTC+8)

**Bar 1 (full headroom ≥ 0.10): FAIL**, at 0.081. The learnable-set headroom is 0.040, so as the spec anticipated,
almost all of the gain comes from skipping noise and spending little on dark.

Mechanism. With a separate learner per channel, the only cost of reading the noisy TV is opportunity: those bytes
don't go to a learnable channel. Noise's own held-out loss sits near 8 bits whatever is read (8.15 at 500K, slightly
overfit). The cost Schmidhuber's argument and our toy worlds point to is different. When one shared cortex reads noise,
that spends its capacity and replay on it and interferes with what it has learned. A per-channel static oracle cannot
see that cost, so this measurement is a lower bound for the real machine. A side finding: the cortex is poor on DNA
(1.89 at 500K, against PPMd 0.62), because a byte GRU at lr 3e-4 learns a 5-symbol alphabet slowly.
