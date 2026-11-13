# M5b-W3 life length

Visit controllers on the W3 ring at three life lengths. W is bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. W9 is that mean over all 9 room channels (the 7, plus N and Z). Lower is better. Numbers only.

## W

### L = 512

| controller | seed 12 | seed 13 | seed 14 | seed 15 | seed 16 | seed 17 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 2.855491 | 2.912797 | 2.923366 | 2.789267 | 2.827362 | 2.873024 | 2.863551 ± 0.046522 |
| oracle7 | 2.644092 | 2.738035 | 2.783926 | 2.718850 | 2.644832 | 2.605103 | 2.689140 ± 0.062333 |
| prog | 2.742358 | 2.743384 | 2.651623 | 2.723879 | 2.740908 | 2.735981 | 2.723022 ± 0.032596 |

### L = 1024

| controller | seed 12 | seed 13 | seed 14 | seed 15 | seed 16 | seed 17 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 2.313721 | 2.297701 | 2.390992 | 2.215562 | 2.302654 | 2.285117 | 2.300958 ± 0.051358 |
| oracle7 | 2.309781 | 2.338945 | 2.290228 | 2.219536 | 2.170851 | 2.166651 | 2.249332 ± 0.067380 |
| prog | 2.116171 | 2.240669 | 2.285091 | 2.168741 | 2.136868 | 2.236815 | 2.197392 ± 0.060826 |

### L = 2048

| controller | seed 12 | seed 13 | seed 14 | seed 15 | seed 16 | seed 17 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 1.979503 | 1.939061 | 1.992578 | 1.944813 | 2.100946 | 1.961694 | 1.986432 ± 0.054448 |
| oracle7 | 1.824375 | 1.925945 | 1.878755 | 1.914727 | 1.959996 | 1.831212 | 1.889168 ± 0.049477 |
| prog | 1.737457 | 1.753690 | 1.734712 | 1.851563 | 1.761203 | 1.726191 | 1.760803 ± 0.042253 |

## W9

### L = 512

| controller | seed 12 | seed 13 | seed 14 | seed 15 | seed 16 | seed 17 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 2.818604 | 2.863748 | 2.871281 | 2.766959 | 2.804441 | 2.831884 | 2.826153 ± 0.035391 |
| oracle7 | 3.248534 | 3.320195 | 3.355923 | 3.305261 | 3.248090 | 3.216890 | 3.282482 ± 0.048257 |
| prog | 2.730350 | 2.731096 | 2.659732 | 2.715774 | 2.729120 | 2.725317 | 2.715232 ± 0.025349 |

### L = 1024

| controller | seed 12 | seed 13 | seed 14 | seed 15 | seed 16 | seed 17 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 2.396211 | 2.384976 | 2.456142 | 2.319632 | 2.390035 | 2.374046 | 2.386840 ± 0.039968 |
| oracle7 | 2.990015 | 3.009859 | 2.972002 | 2.916977 | 2.880192 | 2.875944 | 2.940831 ± 0.052618 |
| prog | 2.242570 | 2.339492 | 2.374109 | 2.283435 | 2.258746 | 2.336508 | 2.305810 ± 0.047358 |

### L = 2048

| controller | seed 12 | seed 13 | seed 14 | seed 15 | seed 16 | seed 17 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 2.135669 | 2.104623 | 2.145593 | 2.108565 | 2.230599 | 2.121881 | 2.141155 ± 0.042462 |
| oracle7 | 2.614618 | 2.688707 | 2.652051 | 2.680026 | 2.717141 | 2.615158 | 2.661284 ± 0.037890 |
| prog | 1.947638 | 1.960192 | 1.945573 | 2.036360 | 1.966125 | 1.939081 | 1.965828 ± 0.032821 |

## Gaps

prog minus the other controller, on W, per seed and L.

### L = 512

| seed | prog − rand_dwell | prog − oracle7 |
|---:|---:|---:|
| 12 | -0.113133 | 0.098266 |
| 13 | -0.169413 | 0.005350 |
| 14 | -0.271743 | -0.132303 |
| 15 | -0.065388 | 0.005029 |
| 16 | -0.086453 | 0.096076 |
| 17 | -0.137044 | 0.130877 |
| mean | -0.140529 | 0.033883 |
| std ddof=0 | 0.067558 | 0.088210 |
| prog lower (count) | 6 of 6 | 1 of 6 |

### L = 1024

| seed | prog − rand_dwell | prog − oracle7 |
|---:|---:|---:|
| 12 | -0.197551 | -0.193611 |
| 13 | -0.057032 | -0.098276 |
| 14 | -0.105901 | -0.005137 |
| 15 | -0.046821 | -0.050796 |
| 16 | -0.165786 | -0.033983 |
| 17 | -0.048302 | 0.070164 |
| mean | -0.103565 | -0.051940 |
| std ddof=0 | 0.059371 | 0.081283 |
| prog lower (count) | 6 of 6 | 5 of 6 |

### L = 2048

| seed | prog − rand_dwell | prog − oracle7 |
|---:|---:|---:|
| 12 | -0.242046 | -0.086918 |
| 13 | -0.185371 | -0.172255 |
| 14 | -0.257866 | -0.144043 |
| 15 | -0.093250 | -0.063164 |
| 16 | -0.339743 | -0.198793 |
| 17 | -0.235503 | -0.105021 |
| mean | -0.225630 | -0.128366 |
| std ddof=0 | 0.074803 | 0.047690 |
| prog lower (count) | 6 of 6 | 6 of 6 |

## Decision shares

Mean over seeds. A stay counts for the room's channel. A move counts for the corridor. Shares sum to 1.

| controller | L | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 512 | 0.0859 | 0.0889 | 0.0941 | 0.0755 | 0.0938 | 0.0938 | 0.0729 | 0.0788 | 0.0853 | 0.2311 |
| oracle7 | 512 | 0.0885 | 0.1243 | 0.0729 | 0.1042 | 0.1296 | 0.1423 | 0.1162 | 0.0000 | 0.0000 | 0.2220 |
| prog | 512 | 0.0234 | 0.0859 | 0.0391 | 0.1172 | 0.1198 | 0.1302 | 0.1618 | 0.0208 | 0.1094 | 0.1924 |
| rand_dwell | 1024 | 0.0911 | 0.0990 | 0.0918 | 0.0781 | 0.0846 | 0.0895 | 0.0872 | 0.0781 | 0.0776 | 0.2228 |
| oracle7 | 1024 | 0.0833 | 0.1237 | 0.0907 | 0.1073 | 0.1283 | 0.1388 | 0.1133 | 0.0000 | 0.0000 | 0.2147 |
| prog | 1024 | 0.1230 | 0.0846 | 0.0768 | 0.0885 | 0.0885 | 0.0938 | 0.1484 | 0.0153 | 0.0796 | 0.2013 |
| rand_dwell | 2048 | 0.0819 | 0.0964 | 0.0802 | 0.0846 | 0.0942 | 0.0918 | 0.0885 | 0.0769 | 0.0835 | 0.2219 |
| oracle7 | 2048 | 0.1003 | 0.1188 | 0.1048 | 0.1068 | 0.1237 | 0.1187 | 0.1147 | 0.0000 | 0.0000 | 0.2122 |
| prog | 2048 | 0.2339 | 0.0658 | 0.0898 | 0.0671 | 0.0701 | 0.0707 | 0.1304 | 0.0117 | 0.0587 | 0.2018 |

## prog shares by quarter of life

Mean over seeds. Quarter i is decisions [(i−1)L/4, iL/4).

| L | quarter | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 0.0729 | 0.0625 | 0.0729 | 0.1003 | 0.1341 | 0.1289 | 0.0625 | 0.0729 | 0.1354 | 0.1576 |
| 512 | 2 | 0.0000 | 0.0000 | 0.0104 | 0.2031 | 0.1784 | 0.1927 | 0.0000 | 0.0000 | 0.1667 | 0.2487 |
| 512 | 3 | 0.0000 | 0.1979 | 0.0130 | 0.1445 | 0.1250 | 0.1784 | 0.0247 | 0.0000 | 0.0938 | 0.2227 |
| 512 | 4 | 0.0208 | 0.0833 | 0.0599 | 0.0208 | 0.0417 | 0.0208 | 0.5599 | 0.0104 | 0.0417 | 0.1406 |
| 1024 | 1 | 0.0365 | 0.0312 | 0.0417 | 0.1517 | 0.1562 | 0.1608 | 0.0312 | 0.0365 | 0.1510 | 0.2031 |
| 1024 | 2 | 0.0104 | 0.1406 | 0.0365 | 0.0827 | 0.0833 | 0.0996 | 0.2923 | 0.0052 | 0.0677 | 0.1816 |
| 1024 | 3 | 0.2155 | 0.0788 | 0.1283 | 0.0573 | 0.0508 | 0.0573 | 0.1348 | 0.0052 | 0.0625 | 0.2096 |
| 1024 | 4 | 0.2298 | 0.0879 | 0.1009 | 0.0625 | 0.0638 | 0.0573 | 0.1354 | 0.0143 | 0.0371 | 0.2109 |
| 2048 | 1 | 0.0234 | 0.0859 | 0.0391 | 0.1172 | 0.1198 | 0.1302 | 0.1618 | 0.0208 | 0.1094 | 0.1924 |
| 2048 | 2 | 0.2227 | 0.0833 | 0.1146 | 0.0599 | 0.0573 | 0.0573 | 0.1351 | 0.0098 | 0.0498 | 0.2103 |
| 2048 | 3 | 0.3219 | 0.0599 | 0.0986 | 0.0417 | 0.0469 | 0.0521 | 0.1094 | 0.0088 | 0.0492 | 0.2116 |
| 2048 | 4 | 0.3675 | 0.0339 | 0.1071 | 0.0495 | 0.0566 | 0.0433 | 0.1152 | 0.0075 | 0.0264 | 0.1930 |

## Mean held-out bits

Mean over seeds. W is the mean of the 7 learnable channels. W9 is the mean of all 9.

| controller | L | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | W | W9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 512 | 4.222353 | 4.237524 | 4.443473 | 1.535742 | 1.401968 | 0.694691 | 3.509108 | 5.358198 | 0.032319 | 2.863551 | 2.826153 |
| oracle7 | 512 | 4.183159 | 4.105416 | 4.624236 | 1.446822 | 1.326850 | 0.471632 | 2.665863 | 5.358041 | 5.360321 | 2.689140 | 3.282482 |
| prog | 512 | 4.986583 | 4.036443 | 4.854480 | 1.387344 | 1.326470 | 0.462162 | 2.007673 | 5.357923 | 0.018005 | 2.723022 | 2.715232 |
| rand_dwell | 1024 | 3.465015 | 3.688062 | 4.046552 | 1.341085 | 1.287376 | 0.394089 | 1.884526 | 5.359005 | 0.015854 | 2.300958 | 2.386840 |
| oracle7 | 1024 | 3.580599 | 3.661768 | 4.143976 | 1.305659 | 1.236638 | 0.375568 | 1.441116 | 5.358437 | 5.363721 | 2.249332 | 2.940831 |
| prog | 1024 | 3.119834 | 4.030292 | 4.190228 | 1.324529 | 1.267567 | 0.409481 | 1.039814 | 5.358451 | 0.012095 | 2.197392 | 2.305810 |
| rand_dwell | 2048 | 2.751733 | 3.598143 | 3.774769 | 1.256344 | 1.188393 | 0.377795 | 0.957851 | 5.359465 | 0.005902 | 1.986432 | 2.141155 |
| oracle7 | 2048 | 2.495171 | 3.567226 | 3.695206 | 1.235553 | 1.154963 | 0.355411 | 0.720649 | 5.359049 | 5.368325 | 1.889168 | 2.661284 |
| prog | 2048 | 1.490384 | 3.583306 | 3.732319 | 1.276181 | 1.204562 | 0.410567 | 0.628299 | 5.358746 | 0.008088 | 1.760803 | 1.965828 |

## Ring order

Channel in rooms 0..8, seeds 12–17. One order per seed; controllers share it.

| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---:|---|---|---|---|---|---|---|---|---|
| 12 | N | C7 | M7 | E110 | E30 | M37 | G | C1 | Z |
| 13 | E110 | M37 | C7 | C1 | E30 | N | M7 | Z | G |
| 14 | C7 | Z | M7 | G | E110 | M37 | N | E30 | C1 |
| 15 | C7 | M37 | Z | E110 | G | M7 | C1 | N | E30 |
| 16 | M37 | C7 | M7 | E110 | C1 | G | N | Z | E30 |
| 17 | M37 | E30 | C1 | Z | N | C7 | G | E110 | M7 |

## Walls

Scored run wall 520.488 s. Pool of 2. Smoke wall 21.712 s.

| controller | L | seed | wall seconds |
|---|---:|---:|---:|
| rand_dwell | 512 | 12 | 4.752 |
| rand_dwell | 512 | 13 | 4.788 |
| rand_dwell | 512 | 14 | 4.536 |
| rand_dwell | 512 | 15 | 4.673 |
| rand_dwell | 512 | 16 | 4.767 |
| rand_dwell | 512 | 17 | 4.457 |
| oracle7 | 512 | 12 | 4.982 |
| oracle7 | 512 | 13 | 4.535 |
| oracle7 | 512 | 14 | 4.586 |
| oracle7 | 512 | 15 | 4.650 |
| oracle7 | 512 | 16 | 4.552 |
| oracle7 | 512 | 17 | 4.579 |
| prog | 512 | 12 | 16.927 |
| prog | 512 | 13 | 16.589 |
| prog | 512 | 14 | 19.989 |
| prog | 512 | 15 | 20.525 |
| prog | 512 | 16 | 16.397 |
| prog | 512 | 17 | 16.625 |
| rand_dwell | 1024 | 12 | 8.004 |
| rand_dwell | 1024 | 13 | 8.014 |
| rand_dwell | 1024 | 14 | 8.122 |
| rand_dwell | 1024 | 15 | 8.162 |
| rand_dwell | 1024 | 16 | 8.107 |
| rand_dwell | 1024 | 17 | 8.025 |
| oracle7 | 1024 | 12 | 8.088 |
| oracle7 | 1024 | 13 | 8.235 |
| oracle7 | 1024 | 14 | 8.433 |
| oracle7 | 1024 | 15 | 8.079 |
| oracle7 | 1024 | 16 | 8.848 |
| oracle7 | 1024 | 17 | 8.326 |
| prog | 1024 | 12 | 33.027 |
| prog | 1024 | 13 | 33.528 |
| prog | 1024 | 14 | 33.789 |
| prog | 1024 | 15 | 32.816 |
| prog | 1024 | 16 | 32.233 |
| prog | 1024 | 17 | 32.422 |
| rand_dwell | 2048 | 12 | 15.029 |
| rand_dwell | 2048 | 13 | 14.822 |
| rand_dwell | 2048 | 14 | 15.015 |
| rand_dwell | 2048 | 15 | 14.618 |
| rand_dwell | 2048 | 16 | 15.334 |
| rand_dwell | 2048 | 17 | 15.325 |
| oracle7 | 2048 | 12 | 14.986 |
| oracle7 | 2048 | 13 | 14.941 |
| oracle7 | 2048 | 14 | 14.964 |
| oracle7 | 2048 | 15 | 14.839 |
| oracle7 | 2048 | 16 | 15.130 |
| oracle7 | 2048 | 17 | 15.171 |
| prog | 2048 | 12 | 67.776 |
| prog | 2048 | 13 | 66.639 |
| prog | 2048 | 14 | 66.809 |
| prog | 2048 | 15 | 64.714 |
| prog | 2048 | 16 | 67.343 |
| prog | 2048 | 17 | 67.848 |

## Smoke test

```
smoke check 1: prog seed 6 L=512 vs results/m5w3/results.json
  prog W 2.846580675880 ref 2.846580675880 abs 0.000e+00 W9 2.811324 finite True wall 15.61s
smoke check 2: seed 12 L=64, rand_dwell oracle7 prog, W and W9 finite
  rand_dwell W 4.695331 W9 4.842435 finite True wall 1.58s learn 0.44s eval 1.15s N 0.000 Z 0.000 corridor 0.250
  oracle7 W 4.445962 W9 4.648489 finite True wall 1.72s learn 0.44s eval 1.28s N 0.000 Z 0.000 corridor 0.250
  prog W 4.324942 W9 4.554542 finite True wall 2.78s learn 1.70s eval 1.07s N 0.125 Z 0.000 corridor 0.109
estimate serial raw 930s conservative 1191s pool2 raw 465s (7.8 min) conservative 595s (9.9 min)
estimate is within 30 min; scored run stays on this machine, pool of 2
```

## Protocol

- Controllers `rand_dwell`, `oracle7` and `prog` are `m5_w3.run_controller`. `m5_w3.py` is imported and not edited. World, corridor glances, compressor, visit length and W are the M5-W3 ones.
- W is `w3_headroom.evaluate`: mean bits/symbol of the frozen compressor on EVAL=2048 fresh symbols of the 7 learnable channels, from a copy of the H20 world, channels in `curio.LEARN` order.
- W9 is the mean of those 7 rates plus N and Z, scored on the same frozen compressor with learning off. Z is EVAL copies of the space symbol. N is EVAL uniform symbols from `default_rng(30000 + seed)`. That stream is not the life rng, so the three controllers on a seed share one N continuation. Corridor is not in W9.
- A stay counts for the room channel. A move counts as corridor. Quarter i is decisions [(i−1)L/4, iL/4). Shares are means over seeds.
- Seeds 12–17. L ∈ {512, 1024, 2048}. Nothing retuned.
- PYTHONHASHSEED=0. CURIO_LR=0.004. CPU. Pool of 2. No GPU.

## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (at L = 2,048, prog − rand_dwell ≤ −0.10 with prog lower on ≥ 5 of 6): PASS.** The gap is −0.226 ± 0.075, and
prog is lower on 6 of 6.
**Bar 2 (prog − oracle7 falls by ≥ 0.10 from L = 512 to 2,048): PASS.** It goes from +0.034 to −0.128, a fall of 0.162. At
L = 2,048 prog beats the learnability oracle on 6 of 6 fresh seeds.

**What it shows.** On fresh seeds, the progress controller beats random dwelling at every life length (6 of 6 each
time). It overtakes the oracle that knows which rooms are worthless, and its lead grows with the life. An oracle that
spreads its time evenly over the learnable rooms is not the best allocation. Curiosity finds an order: easy sources
first, the grammar last. That order is worth more than knowing what to ignore, once the life is long enough to follow
it. M5-W3's 4-of-6 at L = 512 was the short life plus a noisy baseline, not a failing controller.

**W9.** Counting every room, prog leads by more (1.97 vs 2.14 for rand_dwell and 2.66 for oracle7 at L = 2,048). It learns the dark
room to 0.008 bits, while the oracle leaves it at 5.37. Its time in Z (11% falling to 6%) is real compression, which W
does not count.

**Milestone.** This is the north star's core loop working in an embodied world. A machine acts, learns while it acts,
and chooses where to go by its own compression progress. It pays travel and finds a curriculum that beats a privileged
oracle. It is H23c, no longer free to look anywhere. The learner is still a hashed linear model trained by SGD. The next
step for this line is a richer cortex and sleep inside the same loop.
