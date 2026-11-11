# W1 headroom

These are separate learners per channel. The real machine shares one cortex, so this is an estimate.

## Streams

| Channel | Source bytes | Source sha256 | Stream bytes | Stream sha256 | Split | Train bytes | Train sha256 | Held-out bytes | Held-out sha256 |
|---|---:|---|---:|---|---|---:|---|---:|---|
| bible | 4452070 | `ef4a0a6f3bf7254c2304a0d7b577200f2b13c5c37c9bb5c28875e4017bbf14e9` | 4432490 | `ef0b6942e0ac1174072d3e4420cd5e05aca97fe86af2088b986c5a6f493f77a6` | prefix_2000000_100000 | 2000000 | `fa9eb395ac956821a48432d7337ee35e78d7611dbdc75240445e945700737642` | 100000 | `de2429965badbd30dc4eac76f4df2fc843e5ff96e3bbfafd05d1fee608a0dde5` |
| chess | 7672655 | `e7aadff104a610afb5403caf81c1461babecb0dc87760ff5eea7dc0e7a4a8129` | 5027572 | `7791c438b0f21b8f8ec6b3b1f421eaa29d526647ed21eaddc2da9795c29d5b38` | prefix_2000000_100000 | 2000000 | `4986e78f72b9cc7fe1fa85229c56c7359df8afaa7aa426df2be0614cf723ef0b` | 100000 | `c0b520a6b4b43e22db26c1555e288bac280e5fc947ae987d1ecc42176ee23111` |
| squad | 4854279 | `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9` | 2466240 | `1348449a968759680bb85a2803250a8b904d1c2a572d6ddb9ed757a2eafd1cb0` | prefix_2000000_100000 | 2000000 | `42003a1ff6ab605a2f0afe8080e233718cc0b08be61b3b155ec5f647e1c00a0a` | 100000 | `fb97f239566374dbea842207371fa2477938b9cbedf6d2f0283b7878586d1c32` |
| btc | 6606836 | `095ee45ed97240a12e64cb2e038e04eef537da520851b1be4f05e1eee91395b9` | 6606836 | `095ee45ed97240a12e64cb2e038e04eef537da520851b1be4f05e1eee91395b9` | prefix_2000000_100000 | 2000000 | `3ba760b1a58801976c68e834954a16f737e4cfcac1132b78fc667a80efbb713a` | 100000 | `4ba57af0b7dc6b1bb237bd9ddf6d0249496b19e10a9cc803c29e2281e8263a51` |

No stream was shorter than 2100000 bytes.

## Excerpts

### bible

300 bytes.

```














The Old Testament of the King James Version of the Bible




The First Book of Moses:  Called Genesis


1:1 In the beginning God created the heavens and the earth.

1:2 And the earth was without form, and void; and darkness was upon
the face of the deep. And th
```

### chess

300 bytes.

```
d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5 Bf4
d4 Nc6 e4 e5 f4 f6 dxe5 fxe5 fxe5 Nxe5 Qd4 Nc6 Qe5+ Nxe5 c4 Bb4+
e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc6 bxc6 Ra6 Nc4 a4 c3 a3 Nxa3 Rxa3 Rxa3 c4 dxc4 d5 cxd5 Qxd5 exd5 Be6 Ra8+ Ke7 Bc5+ Kf6 Bxf8 Kg6 Bxg7 Kxg7 dxe6 Kh6 exf7 Nf6 Rxh8 Nh5 Bxh5 
```

### squad

300 bytes.

```
Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super
```

### btc

300 bytes.

```
,system_time,midpoint,spread,buys,sells,bids_distance_0,bids_distance_1,bids_distance_2,bids_distance_3,bids_distance_4,bids_distance_5,bids_distance_6,bids_distance_7,bids_distance_8,bids_distance_9,bids_distance_10,bids_distance_11,bids_distance_12,bids_distance_13,bids_distance_14,bids_notional_0
```

Chess excerpt is the moves column only.

## Held-out L_c(n), bits/byte

| n | bible | chess | squad | btc |
|---:|---:|---:|---:|---:|
| 0 | 7.977886 | 8.016386 | 7.980837 | 8.011427 |
| 62500 | 2.642834 | 1.923204 | 3.643461 | 2.079862 |
| 125000 | 2.366760 | 1.822484 | 3.378627 | 1.883045 |
| 250000 | 2.200340 | 1.735391 | 3.172959 | 1.780715 |
| 500000 | 2.116726 | 1.670179 | 2.971317 | 1.735002 |
| 1000000 | 1.958147 | 1.596832 | 2.702754 | 1.701187 |
| 2000000 | 1.805729 | 1.531396 | 2.600601 | 1.615324 |

Held-out bits:

| n | bible | chess | squad | btc |
|---:|---:|---:|---:|---:|
| 0 | 797788.642 | 801638.554 | 798083.650 | 801142.736 |
| 62500 | 264283.439 | 192320.440 | 364346.121 | 207986.182 |
| 125000 | 236676.007 | 182248.401 | 337862.732 | 188304.545 |
| 250000 | 220034.030 | 173539.119 | 317295.896 | 178071.529 |
| 500000 | 211672.593 | 167017.912 | 297131.656 | 173500.158 |
| 1000000 | 195814.709 | 159683.182 | 270275.363 | 170118.728 |
| 2000000 | 180572.885 | 153139.556 | 260060.103 | 161532.363 |

## Prequential 1 MB slices

### bible

Bits 2771491.379673. Bits/byte 1.385746. Replay steps 15629. Traffic 8.002048. Wall seconds 801.387.

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 1456632.485 | 1.456632 |
| 1 | 1000000 | 2000000 | 1314858.895 | 1.314859 |

### chess

Bits 3206589.476955. Bits/byte 1.603295. Replay steps 15629. Traffic 8.002048. Wall seconds 785.629.

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 1671388.273 | 1.671388 |
| 1 | 1000000 | 2000000 | 1535201.204 | 1.535201 |

### squad

Bits 3872055.700862. Bits/byte 1.936028. Replay steps 15629. Traffic 8.002048. Wall seconds 798.432.

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2066438.665 | 2.066439 |
| 1 | 1000000 | 2000000 | 1805617.035 | 1.805617 |

### btc

Bits 3255073.307818. Bits/byte 1.627537. Replay steps 15629. Traffic 8.002048. Wall seconds 790.886.

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 1680078.828 | 1.680079 |
| 1 | 1000000 | 2000000 | 1574994.480 | 1.574994 |

## xz -9e and PPMd order 16

Training stream only. xz is `xz -9e -T1`. PPMd is variant H, order 16, mem 1073741824.
PPMd is encoded in pieces of 65536 bytes on one encoder, then flushed.
A single pyppmd.compress call does not roundtrip on the chess stream. Piece encoding does.
On bible, squad and btc the piece encoding matches that one-shot byte string.

| Channel | xz bytes | xz bits/byte | PPMd bytes | PPMd bits/byte |
|---|---:|---:|---:|---:|
| bible | 456136 | 1.824544 | 365253 | 1.461012 |
| chess | 568400 | 2.273600 | 473059 | 1.892236 |
| squad | 529328 | 2.117312 | 455285 | 1.821140 |
| btc | 477188 | 1.908752 | 456101 | 1.824404 |

## Headroom

Budget 2000000 bytes. Grid [0, 62500, 125000, 250000, 500000, 1000000, 2000000]. Feasible allocations 1095 of 2401.

Uniform mean L_c(500000): 2.123306 bits/byte.
Oracle mean: 2.083897 bits/byte.
Headroom (uniform − oracle): 0.039409 bits/byte.

Oracle allocations: 1.

| bible | chess | squad | btc | Sum | Mean bits/byte |
|---:|---:|---:|---:|---:|---:|
| 500000 | 250000 | 1000000 | 250000 | 2000000 | 2.083897 |

## Kernels

Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that
clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.
Script wall is time.monotonic from kernel_main entry through the result write. Queue time
is not included. Total GPU hours is the sum across the two kernels.

| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours | GPU |
|---|---:|---|---:|---|---:|---:|---|
| ezzzzzx/w1-headroom-a | 1 | COMPLETE | 1621.370 | log_clock | 1588.405 | 0.4504 | Tesla T4 |
| ezzzzzx/w1-headroom-b | 1 | COMPLETE | 1617.285 | log_clock | 1591.143 | 0.4492 | Tesla T4 |

Total GPU hours: 0.8996.

## Smoke

CPU. Bible training bytes 4096. Held-out bytes 1024. Checkpoints 0 and 4096.
W1 bits: 19362.707616529544.
H29c-equivalent bits: 19362.707616529544.
Absolute difference: 0.000e+00.
Within 1e-6: True.
Training finite: True. Held-out finite: True.
Replay steps: 32.
Mid-eval absolute difference: 0.000e+00.

| n | Held-out bits | Bits/byte |
|---:|---:|---:|
| 0 | 8164.811723 | 7.973449 |
| 4096 | 3630.370539 | 3.545284 |

## Protocol

- One fresh learner per channel. lr 0.0003. Replay batch 8. Seed 0. Replay RNG 30000.
- ByteGRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam, clip 1.0. 3627264 parameters.
- Prequential: each chunk is scored before that chunk is trained. Live state is carried and detached.
- Replay state is zero. Replay starts are uniform on 0 .. n_seen−128 inclusive, with replacement.
- The learner advances to each checkpoint with that loop. A checkpoint that is not a multiple of 128 ends on a short chunk, then training continues.
- Held-out evaluation uses no gradient, a zero initial state, chunks of 128, and carried state. L_c(n) is live −log2 p bits divided by held-out bytes.
- The training stream is the first 2000000 bytes. The held-out block is the next 100000 bytes.
- Bible bytes are the text between the *** START line and the *** END line.
- Chess bytes are the moves column, one game per line. Other columns are dropped.
- SQuAD bytes are, per paragraph, the context, then each question and its first answer text, newline-separated.
- BTC bytes are the raw csv, header included.
- xz -9e -T1 and PPMd variant H, order 16, mem 1073741824, on the training bytes. PPMd is encoded in 65536-byte pieces on one encoder.
- Uniform is the mean of L_c(500000). Oracle is the minimum mean of L_c(n_c) over the 7^4 grid with the sum of n_c at most 2000000. Headroom is uniform minus oracle.

## Verdict (orchestrator, 11 Nov 2026, 11:00 UTC+8)

**Bar (headroom ≥ 0.10 bits/byte): FAIL.** Headroom is 0.039. Uniform is 2.1233 and the oracle is 2.0839; the
orchestrator recomputed both from the L_c(n) table. W1 as built does not qualify for the curiosity run.

Mechanism. All four real streams are far from saturation at 2 MB. They follow similar log-linear learning curves,
gaining roughly 0.07–0.25 bits/byte per doubling. With concave curves of similar slope, the marginal gain per byte is
nearly equal across channels, so any allocation near uniform is near optimal. The oracle only shifts budget toward
SQuAD, the steepest curve. Allocation pays when channels differ in kind: some are unlearnable (noise), some are already
learned (saturated), some are learnable. Four rich text-like streams are all the third kind. This holds for separate
learners. A shared cortex could add transfer or interference, which this measurement cannot see.
