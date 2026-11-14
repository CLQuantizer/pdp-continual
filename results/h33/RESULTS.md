# H33 bank under the count mixer on the first 1 MB of enwik8

Bytes: 1000000. sha256 of the first 10 MB: `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
Each byte is scored before any arm learns from it. Chunk 128.
Bank: roots_learn, N = 256, θ_k = 2πk/256, |λ| = 0.98, order-2 readout,
h30x trace rule, Adam lr 0.001, clip 1.0, float32, embedding 32.
bank_mix6: H32b mix6 with the bank in the cortex slot. Init 1 on the bank.
counts_mix6: orders 1–6 only. Init 1 on order 4. Weight sets are H32b's,
highest order k with N_k ≥ 2, else 0.
Smoothing 0.05. Mixer Adam lr 0.003. One count store. Counts of a chunk are
added after both mixers have scored that chunk and stepped. Mixer gradients
do not enter the bank. Count arms do not use the model seed.

## Bits/byte

| Arm | Seed 0 | Seed 1 | Mean |
|---|---:|---:|---:|
| bank | 3.091648 | 3.104004 | 3.097826 |
| counts_mix6 | 2.177547 |  |  |
| bank_mix6 | 2.160058 | 2.162230 | 2.161144 |
| count_1 | 3.874483 |  |  |
| count_2 | 3.166793 |  |  |
| count_3 | 2.961961 |  |  |
| count_4 | 3.297435 |  |  |
| count_5 | 3.936433 |  |  |
| count_6 | 4.611376 |  |  |

Count-arm absolute bit difference between seed 0 and seed 1:

| Arm | Abs bits |
|---|---:|
| counts_mix6 | 0.00000000e+00 |
| count_1 | 0.00000000e+00 |
| count_2 | 0.00000000e+00 |
| count_3 | 0.00000000e+00 |
| count_4 | 0.00000000e+00 |
| count_5 | 0.00000000e+00 |
| count_6 | 0.00000000e+00 |

## Per 100 KB slice

### bank seed 0

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 485897.758 | 4.858978 |
| 1 | 100000 | 200000 | 303617.410 | 3.036174 |
| 2 | 200000 | 300000 | 330161.214 | 3.301612 |
| 3 | 300000 | 400000 | 297989.999 | 2.979900 |
| 4 | 400000 | 500000 | 299073.133 | 2.990731 |
| 5 | 500000 | 600000 | 258587.437 | 2.585874 |
| 6 | 600000 | 700000 | 271205.305 | 2.712053 |
| 7 | 700000 | 800000 | 295540.292 | 2.955403 |
| 8 | 800000 | 900000 | 280505.139 | 2.805051 |
| 9 | 900000 | 1000000 | 269069.922 | 2.690699 |

### bank seed 1

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 492796.122 | 4.927961 |
| 1 | 100000 | 200000 | 306481.603 | 3.064816 |
| 2 | 200000 | 300000 | 331774.509 | 3.317745 |
| 3 | 300000 | 400000 | 299865.870 | 2.998659 |
| 4 | 400000 | 500000 | 299865.305 | 2.998653 |
| 5 | 500000 | 600000 | 258125.588 | 2.581256 |
| 6 | 600000 | 700000 | 271940.339 | 2.719403 |
| 7 | 700000 | 800000 | 294370.152 | 2.943702 |
| 8 | 800000 | 900000 | 280914.759 | 2.809148 |
| 9 | 900000 | 1000000 | 267869.269 | 2.678693 |

### bank_mix6 seed 0

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 331950.188 | 3.319502 |
| 1 | 100000 | 200000 | 195053.229 | 1.950532 |
| 2 | 200000 | 300000 | 231359.750 | 2.313597 |
| 3 | 300000 | 400000 | 202003.354 | 2.020034 |
| 4 | 400000 | 500000 | 207627.843 | 2.076278 |
| 5 | 500000 | 600000 | 182614.107 | 1.826141 |
| 6 | 600000 | 700000 | 194467.261 | 1.944673 |
| 7 | 700000 | 800000 | 208748.048 | 2.087480 |
| 8 | 800000 | 900000 | 204939.761 | 2.049398 |
| 9 | 900000 | 1000000 | 201294.110 | 2.012941 |

### bank_mix6 seed 1

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 332902.639 | 3.329026 |
| 1 | 100000 | 200000 | 195107.941 | 1.951079 |
| 2 | 200000 | 300000 | 231741.604 | 2.317416 |
| 3 | 300000 | 400000 | 202282.575 | 2.022826 |
| 4 | 400000 | 500000 | 207858.654 | 2.078587 |
| 5 | 500000 | 600000 | 182569.931 | 1.825699 |
| 6 | 600000 | 700000 | 194577.447 | 1.945774 |
| 7 | 700000 | 800000 | 208519.754 | 2.085198 |
| 8 | 800000 | 900000 | 205397.183 | 2.053972 |
| 9 | 900000 | 1000000 | 201272.026 | 2.012720 |

### counts_mix6

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 269707.931 | 2.697079 |
| 1 | 100000 | 200000 | 204690.444 | 2.046904 |
| 2 | 200000 | 300000 | 237229.101 | 2.372291 |
| 3 | 300000 | 400000 | 209504.536 | 2.095045 |
| 4 | 400000 | 500000 | 216551.607 | 2.165516 |
| 5 | 500000 | 600000 | 190571.209 | 1.905712 |
| 6 | 600000 | 700000 | 201390.486 | 2.013905 |
| 7 | 700000 | 800000 | 216736.067 | 2.167361 |
| 8 | 800000 | 900000 | 217620.916 | 2.176209 |
| 9 | 900000 | 1000000 | 213544.546 | 2.135445 |

### count_1

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 386296.313 | 3.862963 |
| 1 | 100000 | 200000 | 388010.087 | 3.880101 |
| 2 | 200000 | 300000 | 387008.740 | 3.870087 |
| 3 | 300000 | 400000 | 382891.658 | 3.828917 |
| 4 | 400000 | 500000 | 391741.582 | 3.917416 |
| 5 | 500000 | 600000 | 378797.161 | 3.787972 |
| 6 | 600000 | 700000 | 377605.633 | 3.776056 |
| 7 | 700000 | 800000 | 393012.448 | 3.930124 |
| 8 | 800000 | 900000 | 403008.510 | 4.030085 |
| 9 | 900000 | 1000000 | 386111.187 | 3.861112 |

### count_2

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 341969.669 | 3.419697 |
| 1 | 100000 | 200000 | 304296.708 | 3.042967 |
| 2 | 200000 | 300000 | 328061.514 | 3.280615 |
| 3 | 300000 | 400000 | 309930.607 | 3.099306 |
| 4 | 400000 | 500000 | 317048.516 | 3.170485 |
| 5 | 500000 | 600000 | 301326.541 | 3.013265 |
| 6 | 600000 | 700000 | 304354.297 | 3.043543 |
| 7 | 700000 | 800000 | 323458.662 | 3.234587 |
| 8 | 800000 | 900000 | 323538.455 | 3.235385 |
| 9 | 900000 | 1000000 | 312808.241 | 3.128082 |

### count_3

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 369545.001 | 3.695450 |
| 1 | 100000 | 200000 | 296913.253 | 2.969133 |
| 2 | 200000 | 300000 | 321489.533 | 3.214895 |
| 3 | 300000 | 400000 | 283820.047 | 2.838200 |
| 4 | 400000 | 500000 | 295041.948 | 2.950419 |
| 5 | 500000 | 600000 | 260287.245 | 2.602872 |
| 6 | 600000 | 700000 | 265563.005 | 2.655630 |
| 7 | 700000 | 800000 | 291534.916 | 2.915349 |
| 8 | 800000 | 900000 | 291799.523 | 2.917995 |
| 9 | 900000 | 1000000 | 285966.644 | 2.859666 |

### count_4

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 440255.560 | 4.402556 |
| 1 | 100000 | 200000 | 339685.381 | 3.396854 |
| 2 | 200000 | 300000 | 375964.426 | 3.759644 |
| 3 | 300000 | 400000 | 318796.731 | 3.187967 |
| 4 | 400000 | 500000 | 328162.108 | 3.281621 |
| 5 | 500000 | 600000 | 279129.015 | 2.791290 |
| 6 | 600000 | 700000 | 282319.931 | 2.823199 |
| 7 | 700000 | 800000 | 312585.732 | 3.125857 |
| 8 | 800000 | 900000 | 314268.306 | 3.142683 |
| 9 | 900000 | 1000000 | 306267.728 | 3.062677 |

### count_5

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 511842.878 | 5.118429 |
| 1 | 100000 | 200000 | 396289.692 | 3.962897 |
| 2 | 200000 | 300000 | 456481.311 | 4.564813 |
| 3 | 300000 | 400000 | 388196.491 | 3.881965 |
| 4 | 400000 | 500000 | 393621.151 | 3.936212 |
| 5 | 500000 | 600000 | 339546.348 | 3.395463 |
| 6 | 600000 | 700000 | 343264.623 | 3.432646 |
| 7 | 700000 | 800000 | 371794.866 | 3.717949 |
| 8 | 800000 | 900000 | 370807.552 | 3.708076 |
| 9 | 900000 | 1000000 | 364587.695 | 3.645877 |

### count_6

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 100000 | 571072.896 | 5.710729 |
| 1 | 100000 | 200000 | 446000.904 | 4.460009 |
| 2 | 200000 | 300000 | 532518.249 | 5.325182 |
| 3 | 300000 | 400000 | 459952.576 | 4.599526 |
| 4 | 400000 | 500000 | 462400.616 | 4.624006 |
| 5 | 500000 | 600000 | 409820.359 | 4.098204 |
| 6 | 600000 | 700000 | 418850.939 | 4.188509 |
| 7 | 700000 | 800000 | 442278.861 | 4.422789 |
| 8 | 800000 | 900000 | 435039.575 | 4.350396 |
| 9 | 900000 | 1000000 | 433441.179 | 4.334412 |

## H32b slice 0

| Arm | Bits/byte |
|---|---:|
| gru | 2.104744 |
| mix6 | 1.917176 |
| mix4 | 1.951636 |
| count_3 | 2.961961 |
| ppmd_order_16 | 1.969704 |

## Gaps

| Gap | Seed 0 | Seed 1 | Mean |
|---|---:|---:|---:|
| bank_mix6_minus_h32b_mix6 | 0.242882 | 0.245054 | 0.243968 |
| bank_mix6_minus_counts_mix6 | -0.017489 | -0.015317 | -0.016403 |
| bank_minus_h32b_gru | 0.986904 | 0.999260 | 0.993082 |

substitution threshold: 0.03. slot threshold: -0.05.

## Mean selected weights, last 100 KB

### seed 0

| Mixer | Expert | Mean weight |
|---|---|---:|
| bank_mix6 | bank | 0.374622 |
| bank_mix6 | count_1 | 0.087137 |
| bank_mix6 | count_2 | 0.058686 |
| bank_mix6 | count_3 | 0.149923 |
| bank_mix6 | count_4 | 0.234761 |
| bank_mix6 | count_5 | 0.175912 |
| bank_mix6 | count_6 | 0.334646 |
| counts_mix6 | count_1 | 0.281260 |
| counts_mix6 | count_2 | 0.177473 |
| counts_mix6 | count_3 | 0.207182 |
| counts_mix6 | count_4 | 0.266929 |
| counts_mix6 | count_5 | 0.189651 |
| counts_mix6 | count_6 | 0.369807 |
| bytes |  | 100000 |

### seed 1

| Mixer | Expert | Mean weight |
|---|---|---:|
| bank_mix6 | bank | 0.375954 |
| bank_mix6 | count_1 | 0.088522 |
| bank_mix6 | count_2 | 0.059906 |
| bank_mix6 | count_3 | 0.148426 |
| bank_mix6 | count_4 | 0.233884 |
| bank_mix6 | count_5 | 0.175385 |
| bank_mix6 | count_6 | 0.334140 |
| counts_mix6 | count_1 | 0.281260 |
| counts_mix6 | count_2 | 0.177473 |
| counts_mix6 | count_3 | 0.207182 |
| counts_mix6 | count_4 | 0.266929 |
| counts_mix6 | count_5 | 0.189651 |
| counts_mix6 | count_6 | 0.369807 |
| bytes |  | 100000 |

## Compute

Real multiply-adds. A complex multiply counts as 4. A Linear counts as
3 × in × out per byte (forward, input gradient, weight gradient). Adam counts
as 6 real multiplies per parameter per step. GRU replay is the H32b uniform
cortex: h26 hidden 1024, chunk 128, replay batch 8, each window from state 0.

| Cortex | Parameters | Real MACs/byte | History state | State bytes | Per history byte |
|---|---:|---:|---|---:|---:|
| bank | 238080 | 9684832 | traces h, e^λ, E^B, z | 593920 | 0 |
| gru | 3627264 | 97811734.502 | BPTT window 128 | 2129920 | 16640 |

compute_ratio bank/gru = 0.099015.
compute_ratio bank/gru_live = 0.880471.
gru live MACs/byte = 10999606.882.
gru replay MACs/byte = 86812127.621.

Bank terms, real MACs/byte:

| Term | Real MACs | Formula |
|---|---:|---|
| e_lam | 1024 | 4 N  —  e^λ ← λ e^λ + h |
| e_B | 32768 | 4 N E  —  E^B ← λ E^B + x |
| z_decay | 262144 | 4 N A  —  z ← λ z |
| h | 1024 | 4 N  —  h ← λ h + Bx |
| inject | 16384 | 2 N E  —  Bx real and imag |
| W | 393216 | 3 (2N) A  —  readout W forward + backward |
| P1 | 98304 | 3 (2N) R  —  P1 forward + backward |
| P2 | 98304 | 3 (2N) R  —  P2 forward + backward |
| quad | 192 | 3 R  —  P1⊙P2 and both backward factors |
| V | 49152 | 3 R A  —  V forward + backward |
| grad_nu | 2048 | 8 N  —  conj(δ) (e^λ ∂λ/∂ν) |
| grad_theta | 2048 | 8 N  —  conj(δ) (e^λ ∂λ/∂θ) |
| grad_B_re | 32768 | 4 N E  —  conj(δ) E^B |
| grad_B_im | 32768 | 4 N E  —  conj(δ) (i E^B) |
| grad_coeff | 262144 | 4 N A  —  conj(δ) z |
| grad_emb | 8388608 | 4 N A E  —  einsum into the embedding |
| dlam | 8 | 4 N / 128  —  ∂λ/∂ν once per chunk |
| softmax | 768 | 3 A  —  log-softmax backward |
| adam | 11160 | 6 P / 128  —  one Adam step per chunk |

GRU terms, real MACs/byte, averaged over the 1 MB stream:

| Term | Real MACs | Formula |
|---|---:|---|
| live_gru_gemm | 10027008.000 | 9 H (E+H)  —  3 gates, input and recurrent, forward + 2 backward |
| live_readout | 786432.000 | 3 H A  —  Linear forward + backward |
| live_softmax | 768.000 | 3 A |
| live_element | 15360.000 | 15 H  —  three interpolations, gate backward, product rule |
| live_adam | 170038.882 | n_chunks * 6 P / n  —  one Adam step on the live chunk |
| replay_body | 86642088.739 | n_chunks * 8 * 128 / n * live body  —  H32b replay batch 8, state 0 |
| replay_adam | 170038.882 | n_chunks * 6 P / n  —  one Adam step on the replay batch |

Bank trace bytes = 8 N (2 + E + A)  —  h, e^λ, E^B, z as complex64.
GRU activations = 128 * (H + 3H + E) * 4.
h26.CHUNK = 128. Replay windows are 128 and start at state 0. H32b sets h26.REPLAY_BATCH = 8. Hidden 1024 and chunk 128 are h26.

## Smoke

threads: 2. torch: 2.15.0+cpu.

bank match H20 slow seed 6, model seed 0, alphabet 41, n=4096, chunk 64: h30y 1.190395 h33 1.190395 max_abs_delta 0.000e+00 seconds 2.563 ok True.
count_1..count_4 vs h32.CountBank on 20000 bytes:
- count_1: h33 82426.490827 h32 82426.490827 abs_delta 0.000e+00
- count_2: h33 81345.525952 h32 81345.525952 abs_delta 0.000e+00
- count_3: h33 93761.498281 h32 93761.498281 abs_delta 0.000e+00
- count_4: h33 106744.269327 h32 106744.269327 abs_delta 0.000e+00
prequential first 1024 of a 1024-byte run vs a 2048-byte run, seed 0, ok True.
- bank: identical True max_abs_delta 0.000e+00
- counts_mix6: identical True max_abs_delta 0.000e+00
- bank_mix6: identical True max_abs_delta 0.000e+00
- count_1: identical True max_abs_delta 0.000e+00
- count_2: identical True max_abs_delta 0.000e+00
- count_3: identical True max_abs_delta 0.000e+00
- count_4: identical True max_abs_delta 0.000e+00
- count_5: identical True max_abs_delta 0.000e+00
- count_6: identical True max_abs_delta 0.000e+00
finite first 8192: True. finite 32768: True.
32768 wall seconds 90.631, bank 83.067, mixer 7.509.
kernel estimate seconds 1707.737 (vps threads 2, speedup 1.750, count growth 1.5, late/mean chunk 1.038, limit 5400, fits True).

## Kernels

| Seed | Kernel | Wall seconds |
|---:|---|---:|
| 0 | ezzzzzx/h33-bank-text-s0 | 2568.825 |
| 1 | ezzzzzx/h33-bank-text-s1 | 2260.810 |

## Verdict (orchestrator, 14 Nov 2026, UTC+8)

1. **Substitution holds:** bank_mix6 − 1.917 (H32b's GRU mix6) = +0.244 bits/byte (bar ≤ +0.03). **FAIL.**
2. **The bank earns its slot:** bank_mix6 − counts_mix6 = −0.016 bits/byte (bar ≤ −0.05). **FAIL.**
Report only: the bank alone scores 3.098 bits/byte against the GRU's 2.105 (+0.99). Per byte, the bank costs 0.88× the GRU's live
forward-and-learning compute, and 0.10× once the GRU's replay is counted.

Checks: the bank matches H30y and the counts match H32b's store exactly (delta 0). The count arms are identical across
seeds. The mixer's weights are read from the rows that scored each byte.

**Mechanism.** The bank is a linear diagonal recurrence of 256 complex units with an order-2 readout and a 32-wide embedding,
238k parameters in all. On text it learns little beyond short context, and short context is what the order-1 to order-6 counts already
hold exactly. The mixer still gives it weight 0.37, but what it adds over the counts is only 0.016. The GRU is a nonlinear recurrence
with 3.6 M parameters trained by BPTT. It learns features the counts lack, such as word shape and the open tag, so it carries
0.26 more than the counts (1.917 vs 2.178). At about the same live compute, the local bank cannot yet stand in for backprop on real
data. As the kill rule says, mixing is not the lever. What is missing is a richer local cortex: nonlinear depth, or many banks with a
nonlinear mixer between them, learned by local rules. That is a substrate question, ranked behind W5.
