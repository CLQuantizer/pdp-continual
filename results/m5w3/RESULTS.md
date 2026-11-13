# M5-W3 progress controller

Visit controllers on the W3 ring. Bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. Lower is better. Numbers only.

## W at L = 512

| controller | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | seed 11 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 3.103469 | 3.236270 | 2.820434 | 2.732186 | 2.931940 | 2.679422 | 2.917287 ± 0.198684 |
| oracle7 | 2.859435 | 2.507232 | 2.580938 | 2.532208 | 2.763378 | 2.741224 | 2.664069 ± 0.130945 |
| err | 3.447341 | 3.963692 | 3.562034 | 3.633276 | 3.367021 | 3.484724 | 3.576348 ± 0.192500 |
| prog | 2.846581 | 2.748193 | 2.729443 | 2.736738 | 2.817366 | 2.724786 | 2.767184 ± 0.047137 |

## Per-channel held-out bits at L = 512

| controller | seed | G | C1 | C7 | E30 | E110 | M7 | M37 | W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 6 | 4.844448 | 4.628294 | 4.262930 | 1.504143 | 1.262469 | 1.049764 | 4.172236 | 3.103469 |
| rand_dwell | 7 | 3.659610 | 4.045679 | 4.426294 | 1.755406 | 5.354855 | 0.873514 | 2.538531 | 3.236270 |
| rand_dwell | 8 | 4.434967 | 4.286803 | 4.444673 | 1.392870 | 1.525212 | 0.488825 | 3.169691 | 2.820434 |
| rand_dwell | 9 | 4.337111 | 3.637663 | 4.363514 | 1.579241 | 1.428546 | 0.502086 | 3.277138 | 2.732186 |
| rand_dwell | 10 | 4.559524 | 4.198205 | 4.455070 | 1.397098 | 1.644029 | 0.552136 | 3.717517 | 2.931940 |
| rand_dwell | 11 | 3.970392 | 4.576384 | 4.637561 | 1.352355 | 1.434344 | 0.554111 | 2.230810 | 2.679422 |
| rand_dwell | mean | 4.301009 | 4.228838 | 4.431673 | 1.496852 | 2.108242 | 0.670073 | 3.184320 | 2.917287 |
| oracle7 | 6 | 4.237693 | 4.557514 | 4.170736 | 1.338485 | 1.279033 | 1.283411 | 3.149176 | 2.859435 |
| oracle7 | 7 | 3.438731 | 3.781608 | 4.242259 | 1.500503 | 1.431517 | 0.627381 | 2.528628 | 2.507232 |
| oracle7 | 8 | 4.233845 | 4.286795 | 3.836667 | 1.368849 | 1.710887 | 0.678037 | 1.951486 | 2.580938 |
| oracle7 | 9 | 3.887564 | 4.019863 | 4.363537 | 1.400519 | 1.305345 | 0.707076 | 2.041552 | 2.532208 |
| oracle7 | 10 | 3.746580 | 4.542164 | 4.296703 | 1.291068 | 1.380155 | 0.715074 | 3.371902 | 2.763378 |
| oracle7 | 11 | 3.875469 | 4.275628 | 4.351359 | 1.362271 | 1.320177 | 0.380121 | 3.623540 | 2.741224 |
| oracle7 | mean | 3.903314 | 4.243929 | 4.210210 | 1.376949 | 1.404519 | 0.731850 | 2.777714 | 2.664069 |
| err | 6 | 4.849299 | 4.668925 | 4.937729 | 2.183608 | 1.706857 | 1.988089 | 3.796877 | 3.447341 |
| err | 7 | 4.838253 | 4.605344 | 5.122465 | 3.274405 | 3.183933 | 3.285865 | 3.435582 | 3.963692 |
| err | 8 | 4.622949 | 4.286791 | 4.534333 | 3.268235 | 2.104703 | 1.964376 | 4.152848 | 3.562034 |
| err | 9 | 3.742869 | 4.019492 | 4.282479 | 3.273772 | 3.189749 | 3.265793 | 3.658775 | 3.633276 |
| err | 10 | 4.203187 | 4.461200 | 4.623910 | 2.180479 | 3.187715 | 1.235324 | 3.677328 | 3.367021 |
| err | 11 | 4.393442 | 4.576869 | 4.551899 | 2.216109 | 3.210403 | 1.812321 | 3.632022 | 3.484724 |
| err | mean | 4.441667 | 4.436437 | 4.675469 | 2.732768 | 2.763893 | 2.258628 | 3.725572 | 3.576348 |
| prog | 6 | 5.091828 | 4.557690 | 5.122767 | 1.373210 | 1.307217 | 0.492782 | 1.980572 | 2.846581 |
| prog | 7 | 4.849769 | 4.406291 | 4.760451 | 1.402387 | 1.292578 | 0.513621 | 2.012253 | 2.748193 |
| prog | 8 | 4.837748 | 4.117482 | 5.106339 | 1.388277 | 1.326072 | 0.590574 | 1.739607 | 2.729443 |
| prog | 9 | 5.057393 | 4.019848 | 4.852441 | 1.372750 | 1.308076 | 0.540422 | 2.006235 | 2.736738 |
| prog | 10 | 5.068082 | 4.542531 | 4.949581 | 1.374432 | 1.348984 | 0.456382 | 1.981566 | 2.817366 |
| prog | 11 | 4.821960 | 4.124337 | 4.760581 | 1.380160 | 1.334547 | 0.450695 | 2.201222 | 2.724786 |
| prog | mean | 4.954463 | 4.294697 | 4.925360 | 1.381869 | 1.319579 | 0.507413 | 1.986909 | 2.767184 |

## Decision shares at L = 512

Mean over seeds. A stay counts for the room's channel. A move counts for the corridor. Shares sum to 1.

| controller | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 0.0755 | 0.0807 | 0.1003 | 0.0911 | 0.0736 | 0.1025 | 0.0911 | 0.1081 | 0.0495 | 0.2275 |
| oracle7 | 0.1152 | 0.0833 | 0.1458 | 0.1406 | 0.1042 | 0.0895 | 0.1130 | 0.0000 | 0.0000 | 0.2083 |
| err | 0.0625 | 0.0430 | 0.0651 | 0.0234 | 0.0234 | 0.0286 | 0.0625 | 0.5749 | 0.0156 | 0.1009 |
| prog | 0.0234 | 0.0885 | 0.0322 | 0.1227 | 0.1198 | 0.1302 | 0.1631 | 0.0260 | 0.1094 | 0.1846 |

## prog shares by quarter of life

Mean over seeds. Quarter i is decisions [(i−1)L/4, iL/4).

| L | quarter | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 0.0625 | 0.0690 | 0.0625 | 0.1211 | 0.1289 | 0.1224 | 0.0625 | 0.0729 | 0.1458 | 0.1523 |
| 512 | 2 | 0.0208 | 0.0247 | 0.0000 | 0.2018 | 0.1836 | 0.2057 | 0.0000 | 0.0104 | 0.1406 | 0.2122 |
| 512 | 3 | 0.0000 | 0.1862 | 0.0104 | 0.1146 | 0.1250 | 0.1406 | 0.0638 | 0.0208 | 0.1276 | 0.2109 |
| 512 | 4 | 0.0104 | 0.0742 | 0.0560 | 0.0534 | 0.0417 | 0.0521 | 0.5260 | 0.0000 | 0.0234 | 0.1628 |
| 2048 | 1 | 0.0234 | 0.0885 | 0.0322 | 0.1227 | 0.1198 | 0.1302 | 0.1631 | 0.0260 | 0.1094 | 0.1846 |
| 2048 | 2 | 0.2201 | 0.0703 | 0.1279 | 0.0518 | 0.0622 | 0.0573 | 0.1364 | 0.0026 | 0.0550 | 0.2165 |
| 2048 | 3 | 0.3639 | 0.0573 | 0.0856 | 0.0521 | 0.0420 | 0.0495 | 0.1094 | 0.0104 | 0.0387 | 0.1911 |
| 2048 | 4 | 0.3773 | 0.0391 | 0.1058 | 0.0465 | 0.0547 | 0.0443 | 0.1016 | 0.0104 | 0.0365 | 0.1839 |

## Gaps at L = 512

prog minus the other controller, per seed.

| seed | prog − rand_dwell | prog − oracle7 | prog − err |
|---:|---:|---:|---:|
| 6 | -0.256888 | -0.012855 | -0.600760 |
| 7 | -0.488077 | 0.240960 | -1.215500 |
| 8 | -0.090992 | 0.148505 | -0.832591 |
| 9 | 0.004552 | 0.204530 | -0.896538 |
| 10 | -0.114574 | 0.053988 | -0.549655 |
| 11 | 0.045364 | -0.016438 | -0.759938 |
| mean | -0.150103 | 0.103115 | -0.809163 |
| std ddof=0 | 0.179124 | 0.101244 | 0.218515 |
| prog lower (count) | 4 of 6 | 2 of 6 | 6 of 6 |

## L = 2048

| controller | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | seed 11 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 1.747728 | 1.740639 | 1.825672 | 1.724590 | 1.874493 | 1.762745 | 1.779311 ± 0.053183 |
| oracle7 | 1.853719 | 1.780228 | 1.851564 | 1.811098 | 1.835125 | 1.823442 | 1.825863 ± 0.025261 |

| seed | prog − oracle7 |
|---:|---:|
| 6 | -0.105991 |
| 7 | -0.039589 |
| 8 | -0.025892 |
| 9 | -0.086508 |
| 10 | 0.039368 |
| 11 | -0.060696 |
| mean | -0.046551 |
| std ddof=0 | 0.046881 |
| prog lower (count) | 5 of 6 |

Mean held-out bits at L = 2048.

| controller | G | C1 | C7 | E30 | E110 | M7 | M37 | W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prog | 1.435200 | 3.745901 | 3.757649 | 1.276901 | 1.215842 | 0.375183 | 0.648503 | 1.779311 |
| oracle7 | 2.297636 | 3.413054 | 3.649723 | 1.235230 | 1.158473 | 0.347032 | 0.679891 | 1.825863 |

## Noise and dark shares

Mean over seeds of the fraction of decisions whose glance is N (noise) or Z (dark).

| controller | L | N | Z | corridor |
|---|---:|---:|---:|---:|
| rand_dwell | 512 | 0.1081 | 0.0495 | 0.2275 |
| oracle7 | 512 | 0.0000 | 0.0000 | 0.2083 |
| err | 512 | 0.5749 | 0.0156 | 0.1009 |
| prog | 512 | 0.0260 | 0.1094 | 0.1846 |
| prog | 2048 | 0.0124 | 0.0599 | 0.1940 |
| oracle7 | 2048 | 0.0000 | 0.0000 | 0.2030 |

## W3 headroom reference (seeds 0–2)

From results/w3/results.json. Not recomputed in this run.

| policy | L | seed 0 | seed 1 | seed 2 | mean | std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|
| rand_walk | 512 | 3.720991 | 3.754106 | 3.491642 | 3.655580 | 0.116707 |
| rand_dwell | 512 | 2.961286 | 2.955954 | 2.772922 | 2.896721 | 0.087566 |
| oracle7 | 512 | 2.857721 | 2.747134 | 2.489710 | 2.698189 | 0.154175 |
| oracle_hard | 512 | 2.922988 | 3.004731 | 3.448089 | 3.125269 | 0.230694 |
| oracle7 | 2048 | 1.822230 | 1.840934 | 1.837338 | 1.833501 | 0.008104 |

| seed | rand_dwell − oracle7 | oracle7 L512 − L2048 |
|---:|---:|---:|
| 0 | 0.103565 | 1.035492 |
| 1 | 0.208820 | 0.906200 |
| 2 | 0.283211 | 0.652372 |
| mean | 0.198532 | 0.864688 |
| std ddof=0 | 0.073700 | 0.159138 |

## Ring order

Channel in rooms 0..8, seeds 6–11. One order per seed; controllers share it.

| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---:|---|---|---|---|---|---|---|---|---|
| 6 | M37 | M7 | C7 | E110 | C1 | G | E30 | Z | N |
| 7 | M37 | N | E30 | M7 | E110 | Z | C7 | G | C1 |
| 8 | E110 | M37 | C7 | G | C1 | N | Z | E30 | M7 |
| 9 | C7 | M37 | Z | C1 | E30 | N | M7 | E110 | G |
| 10 | E30 | M7 | M37 | N | G | E110 | Z | C1 | C7 |
| 11 | M7 | C1 | G | E30 | E110 | M37 | N | Z | C7 |

## Walls

Scored run wall 346.008 s. Pool of 2. Smoke wall 28.635 s.

| controller | L | seed | wall seconds |
|---|---:|---:|---:|
| rand_dwell | 512 | 6 | 4.328 |
| rand_dwell | 512 | 7 | 4.227 |
| rand_dwell | 512 | 8 | 4.232 |
| rand_dwell | 512 | 9 | 4.303 |
| rand_dwell | 512 | 10 | 4.322 |
| rand_dwell | 512 | 11 | 4.190 |
| oracle7 | 512 | 6 | 5.053 |
| oracle7 | 512 | 7 | 4.996 |
| oracle7 | 512 | 8 | 4.413 |
| oracle7 | 512 | 9 | 4.558 |
| oracle7 | 512 | 10 | 4.465 |
| oracle7 | 512 | 11 | 4.404 |
| err | 512 | 6 | 4.468 |
| err | 512 | 7 | 4.315 |
| err | 512 | 8 | 4.388 |
| err | 512 | 9 | 4.586 |
| err | 512 | 10 | 4.675 |
| err | 512 | 11 | 4.531 |
| prog | 512 | 6 | 16.468 |
| prog | 512 | 7 | 17.695 |
| prog | 512 | 8 | 17.115 |
| prog | 512 | 9 | 16.639 |
| prog | 512 | 10 | 17.316 |
| prog | 512 | 11 | 16.011 |
| prog | 2048 | 6 | 72.725 |
| prog | 2048 | 7 | 73.743 |
| prog | 2048 | 8 | 70.007 |
| prog | 2048 | 9 | 70.403 |
| prog | 2048 | 10 | 65.822 |
| prog | 2048 | 11 | 65.497 |
| oracle7 | 2048 | 6 | 15.637 |
| oracle7 | 2048 | 7 | 15.757 |
| oracle7 | 2048 | 8 | 14.335 |
| oracle7 | 2048 | 9 | 14.852 |
| oracle7 | 2048 | 10 | 14.651 |
| oracle7 | 2048 | 11 | 15.188 |

## Smoke test

```
smoke check 3 first: L=64 seed 0, four controllers
  rand_dwell W 4.490496 finite True wall 1.34s learn 0.47s eval 0.88s corridor 0.250
  oracle7 W 4.490496 finite True wall 1.32s learn 0.44s eval 0.88s corridor 0.250
  err W 4.819563 finite True wall 1.38s learn 0.46s eval 0.92s corridor 0.109
  prog W 4.819563 finite True wall 2.54s learn 1.61s eval 0.93s corridor 0.109
smoke check 2: 6-visit prog trace, seed 0
prog seed 0, first 6 visits (target, travel, summed progress, visit reward)
  visit 1 target 0 M37 travel 0 [-] dwell 8 sum_progress 0.311619 visit_reward 0.038952 decisions 8
  visit 2 target 1 C7 travel 1 [right] dwell 8 sum_progress 0.229483 visit_reward 0.025498 decisions 9
  visit 3 target 2 C1 travel 1 [right] dwell 8 sum_progress 0.474410 visit_reward 0.052712 decisions 9
  visit 4 target 3 N travel 1 [right] dwell 8 sum_progress 0.000229 visit_reward 0.000025 decisions 9
  visit 5 target 4 G travel 1 [right] dwell 8 sum_progress 0.202411 visit_reward 0.022490 decisions 9
  visit 6 target 5 E30 travel 1 [right] dwell 8 sum_progress 1.476994 visit_reward 0.164110 decisions 9
smoke check 1: rand_dwell and oracle7 vs w3_headroom.run_life, seed 0, L=512
  rand_dwell m5 W 2.961286438354 w3 W 2.961286438354 abs 0.000e+00 corridor_abs 0.000e+00
  held abs G 0.000e+00 C1 0.000e+00 C7 0.000e+00 E30 0.000e+00 E110 0.000e+00 M7 0.000e+00 M37 0.000e+00
  oracle7 m5 W 2.857721285052 w3 W 2.857721285052 abs 0.000e+00 corridor_abs 0.000e+00
  held abs G 0.000e+00 C1 0.000e+00 C7 0.000e+00 E30 0.000e+00 E110 0.000e+00 M7 0.000e+00 M37 0.000e+00
  rand_dwell vs published results/w3 abs 0.000e+00
  oracle7 vs published results/w3 abs 0.000e+00
estimate serial raw 569s conservative 745s pool2 raw 285s (4.7 min) conservative 373s (6.2 min)
estimate is within 20 min; scored run stays on this machine, pool of 2
```

## Protocol

- World, corridor glances, `curio.SGDCompressor` at lr 0.004, life length and W are `w3_headroom.py`. W is the mean over the 7 learnable channels of bits/symbol on EVAL=2048 fresh symbols. The H20 world is copied before the continuation. Channels are scored in `curio.LEARN` order.
- A visit chooses a target room, walks the shortest path (each move is one corridor glance), then stays 8 decisions. A visit cut short by the end of life ends there.
- Per-glance rewards follow `curio.live` with `prog_kind=hist`. prog scores the old and new compressor on the channel reservoir (`R_RES`, rng `default_rng(20_000 + seed)`), and the denominator is the reservoir length after that glance is stored, because `curio.live` binds `hs = res[c]` before the append. err is the glance's bits per symbol. Corridor glances are learnt and earn nothing.
- The visit reward is the sum of the dwell glances' per-glance rewards, divided by travel decisions plus dwell decisions.
- `rand_dwell` and `oracle7` use `w3_headroom.next_action` and `default_rng(1_000 + seed)` for both targets and N-channel noise. The smoke test requires their W on seed 0, L=512, to match `w3_headroom.run_life` to 1e-9, which fixes that stream. The spec's `default_rng(10_000 + seed)` is the choice rng of `err` and `prog`.
- `err` and `prog` share one bandit. The first pass visits every room once, always the nearest unvisited room, ties broken to the right. Then, with probability EPS=0.1, a uniform room; otherwise the room with the highest EMA, ties broken by `max` on `(value, rng.random())` as in `curio.live`. The EMA uses BETA=0.3 once per visit. Their N-channel noise is `default_rng(1_000 + seed)`, separate from choices.
- Seeds 6–11. L=512 for every controller. L=2048 for prog and oracle7.
- PYTHONHASHSEED=0. CPU. Pool of 2. No GPU.

## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (prog − rand_dwell ≤ −0.10 on the mean, and prog lower on ≥ 5 of 6 seeds): FAIL.** The mean passes at −0.150,
but prog is lower on only 4 of 6 seeds (seeds 9 and 11: +0.005 and +0.045).
**Bar 2 (prog − err ≤ −0.20): PASS.** The gap is −0.81, and prog is lower on 6 of 6. The error-seeker spends 57% of its
life on noise. prog spends 2.6% there.

**What the run shows.**
- **prog is consistent and rand_dwell is not:** the std is 0.047 vs 0.199. The seeds where prog does not win are ones
  where random dwelling got lucky (2.62 on seed 11). Random dwelling is a noisy baseline, so 5 of 6 was a demanding
  bar.
- **It loses the early life.** In the first three quarters of a 512 life, prog spends 13–15% of its decisions in the
  dark room Z. Corridors take another 15–21%. The SGD compressor learns even a constant symbol slowly at lr 0.004, so Z
  gives real, slow progress. W does not score Z, though, and oracle7 excludes it by fiat. Part of the gap to oracle7
  (+0.10) is the agent learning something the score ignores.
- **Over a longer life it wins.** At L = 2,048, prog beats oracle7 on 5 of 6 seeds (−0.047). Its curriculum runs from
  the easy CA and short motif (quarter 1), to M37 and C7, to the grammar G (36–38% of the last half). That is H23c's
  curriculum, now with travel. Curiosity's advantage grows with life length, because what it buys is an order of
  study, and an order pays off only once there is time to follow it.

**The kill rule says "planning, not a bandit".** I am not taking that as given. The failure is in consistency, not in
the mean, and the long-life result goes the other way. The follow-up is on fresh seeds (12–17), with life length as
the factor (M5b), and it pre-registers the claim that the advantage grows with L. A planner is next only if M5b also
fails.
