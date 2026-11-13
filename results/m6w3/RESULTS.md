# M6-W3 sleep

Sleep is an action on the W3 ring. W is bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. W9 is that mean over all 9 room channels. Lower is better. Numbers only.

## W

### L = 512

| controller | seed 18 | seed 19 | seed 20 | seed 21 | seed 22 | seed 23 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 2.771504 | 2.777528 | 2.701270 | 2.785545 | 2.711746 | 2.860953 | 2.768091 ± 0.052708 |
| prog_total | 2.738062 | 2.780412 | 2.714848 | 2.736407 | 2.711746 | 2.843951 | 2.754237 ± 0.045953 |
| prog_sleep | 2.696407 | 2.792658 | 2.749220 | 2.757574 | 2.703331 | 2.864109 | 2.760550 ± 0.056688 |
| fixed_sleep | 2.733488 | 2.758857 | 2.776794 | 2.702415 | 2.689770 | 2.895765 | 2.759515 ± 0.067903 |

### L = 2048

| controller | seed 18 | seed 19 | seed 20 | seed 21 | seed 22 | seed 23 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 1.745005 | 1.724221 | 1.730304 | 1.761021 | 1.729390 | 1.781761 | 1.745284 ± 0.020408 |
| prog_total | 1.730566 | 1.765730 | 1.732510 | 1.724529 | 1.714514 | 1.766718 | 1.739094 ± 0.020021 |
| prog_sleep | 1.824140 | 1.873554 | 1.917638 | 1.778354 | 1.801790 | 1.825488 | 1.836827 ± 0.046216 |
| fixed_sleep | 1.809760 | 1.785788 | 1.792632 | 1.750703 | 1.849135 | 1.787718 | 1.795956 ± 0.029577 |

## W9

### L = 512

| controller | seed 18 | seed 19 | seed 20 | seed 21 | seed 22 | seed 23 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 2.752939 | 2.757682 | 2.698378 | 2.763808 | 2.706398 | 2.822153 | 2.750226 ± 0.040886 |
| prog_total | 2.727288 | 2.759926 | 2.709302 | 2.725949 | 2.706398 | 2.809194 | 2.739676 ± 0.035625 |
| prog_sleep | 2.694355 | 2.769539 | 2.735784 | 2.742186 | 2.699651 | 2.824960 | 2.744413 ± 0.044174 |
| fixed_sleep | 2.723467 | 2.743239 | 2.757260 | 2.699308 | 2.689236 | 2.849470 | 2.743663 ± 0.052785 |

### L = 2048

| controller | seed 18 | seed 19 | seed 20 | seed 21 | seed 22 | seed 23 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 1.953521 | 1.937535 | 1.942157 | 1.965886 | 1.941326 | 1.982056 | 1.953747 ± 0.015813 |
| prog_total | 1.942358 | 1.969828 | 1.943926 | 1.937630 | 1.929757 | 1.970417 | 1.948986 ± 0.015609 |
| prog_sleep | 2.015522 | 2.055436 | 2.090253 | 1.981735 | 2.000105 | 2.016716 | 2.026628 ± 0.036098 |
| fixed_sleep | 2.004791 | 1.986960 | 1.992545 | 1.959853 | 2.036093 | 1.987467 | 1.994618 ± 0.022900 |

## Gaps

On W. prog_sleep minus prog_total, prog_sleep minus fixed_sleep, prog_total minus prog.

### L = 512

| seed | prog_sleep − prog_total | prog_sleep − fixed_sleep | prog_total − prog |
|---:|---:|---:|---:|
| 18 | -0.041655 | -0.037081 | -0.033443 |
| 19 | 0.012246 | 0.033802 | 0.002884 |
| 20 | 0.034372 | -0.027574 | 0.013578 |
| 21 | 0.021167 | 0.055159 | -0.049139 |
| 22 | -0.008415 | 0.013561 | -0.000000 |
| 23 | 0.020158 | -0.031656 | -0.017002 |
| mean | 0.006312 | 0.001035 | -0.013854 |
| std ddof=0 | 0.025009 | 0.035355 | 0.021845 |
| left lower (count) | 2 of 6 | 3 of 6 | 4 of 6 |

### L = 2048

| seed | prog_sleep − prog_total | prog_sleep − fixed_sleep | prog_total − prog |
|---:|---:|---:|---:|
| 18 | 0.093574 | 0.014380 | -0.014439 |
| 19 | 0.107824 | 0.087767 | 0.041509 |
| 20 | 0.185128 | 0.125005 | 0.002206 |
| 21 | 0.053825 | 0.027651 | -0.036492 |
| 22 | 0.087276 | -0.047345 | -0.014876 |
| 23 | 0.058771 | 0.037770 | -0.015043 |
| mean | 0.097733 | 0.040871 | -0.006189 |
| std ddof=0 | 0.043421 | 0.054690 | 0.024103 |
| left lower (count) | 0 of 6 | 1 of 6 | 4 of 6 |

## Sleep share

Sleep decisions divided by L. fixed_sleep uses prog_sleep's count on the same seed and L.

### L = 512

| controller | seed 18 | seed 19 | seed 20 | seed 21 | seed 22 | seed 23 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_total | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_sleep | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.140625 | 0.127604 ± 0.005823 |
| fixed_sleep | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.125000 | 0.140625 | 0.127604 ± 0.005823 |

### L = 2048

| controller | seed 18 | seed 19 | seed 20 | seed 21 | seed 22 | seed 23 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_total | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_sleep | 0.324707 | 0.312500 | 0.354004 | 0.332031 | 0.332520 | 0.342285 | 0.333008 ± 0.013023 |
| fixed_sleep | 0.324707 | 0.312500 | 0.354004 | 0.332031 | 0.332520 | 0.342285 | 0.333008 ± 0.013023 |

## prog_sleep sleep share by quarter

Quarter i is decisions [(i−1)L/4, iL/4). The share is sleep decisions in the quarter over L/4.

### L = 512

| seed | quarter 1 | quarter 2 | quarter 3 | quarter 4 |
|---:|---:|---:|---:|---:|
| 18 | 0.062500 | 0.125000 | 0.250000 | 0.062500 |
| 19 | 0.062500 | 0.125000 | 0.187500 | 0.125000 |
| 20 | 0.062500 | 0.125000 | 0.187500 | 0.125000 |
| 21 | 0.062500 | 0.125000 | 0.250000 | 0.062500 |
| 22 | 0.062500 | 0.125000 | 0.187500 | 0.125000 |
| 23 | 0.062500 | 0.125000 | 0.250000 | 0.125000 |
| mean | 0.062500 | 0.125000 | 0.218750 | 0.104167 |

### L = 2048

| seed | quarter 1 | quarter 2 | quarter 3 | quarter 4 |
|---:|---:|---:|---:|---:|
| 18 | 0.125000 | 0.140625 | 0.328125 | 0.705078 |
| 19 | 0.125000 | 0.140625 | 0.310547 | 0.673828 |
| 20 | 0.125000 | 0.156250 | 0.421875 | 0.712891 |
| 21 | 0.125000 | 0.125000 | 0.359375 | 0.718750 |
| 22 | 0.125000 | 0.140625 | 0.343750 | 0.720703 |
| 23 | 0.140625 | 0.125000 | 0.328125 | 0.775391 |
| mean | 0.127604 | 0.138021 | 0.348633 | 0.717773 |

## Decision shares

Mean over seeds. A stay counts for the room channel. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.

| controller | L | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prog | 512 | 0.0234 | 0.0859 | 0.0365 | 0.1227 | 0.1230 | 0.1354 | 0.1559 | 0.0234 | 0.1120 | 0.1816 | 0.0000 |
| prog_total | 512 | 0.0234 | 0.0853 | 0.0365 | 0.1224 | 0.1214 | 0.1380 | 0.1624 | 0.0234 | 0.1016 | 0.1855 | 0.0000 |
| prog_sleep | 512 | 0.0182 | 0.0755 | 0.0208 | 0.1126 | 0.1120 | 0.1276 | 0.1354 | 0.0156 | 0.0964 | 0.1582 | 0.1276 |
| fixed_sleep | 512 | 0.0234 | 0.0745 | 0.0267 | 0.1094 | 0.1120 | 0.1250 | 0.1286 | 0.0234 | 0.0951 | 0.1543 | 0.1276 |
| prog | 2048 | 0.2413 | 0.0697 | 0.0879 | 0.0684 | 0.0684 | 0.0703 | 0.1315 | 0.0124 | 0.0599 | 0.1903 | 0.0000 |
| prog_total | 2048 | 0.2402 | 0.0690 | 0.0924 | 0.0663 | 0.0677 | 0.0697 | 0.1309 | 0.0124 | 0.0566 | 0.1948 | 0.0000 |
| prog_sleep | 2048 | 0.1145 | 0.0508 | 0.0501 | 0.0534 | 0.0540 | 0.0592 | 0.0918 | 0.0072 | 0.0495 | 0.1365 | 0.3330 |
| fixed_sleep | 2048 | 0.1343 | 0.0462 | 0.0472 | 0.0521 | 0.0525 | 0.0553 | 0.0853 | 0.0085 | 0.0508 | 0.1348 | 0.3330 |

## prog_sleep visit rewards by quarter

Whole-store bits saved per decision, mean over the visits whose first decision falls in the quarter. A quarter with no visits of that kind is blank. The mean row is the mean of the per-seed means, over seeds that have at least one such visit.

### L = 512

| seed | quarter | sleep visits | sleep mean reward | room visits | room mean reward |
|---:|---:|---:|---:|---:|---:|
| 18 | 1 | 1 | 27.679225 | 14 | 21.621007 |
| 18 | 2 | 2 | 8.490740 | 10 | 3.877102 |
| 18 | 3 | 4 | 7.076713 | 10 | 3.166456 |
| 18 | 4 | 1 | 5.050426 | 13 | 6.207177 |
| 19 | 1 | 1 | 24.292507 | 13 | 21.292761 |
| 19 | 2 | 2 | 7.883259 | 11 | 6.054190 |
| 19 | 3 | 3 | 6.635148 | 10 | 3.235543 |
| 19 | 4 | 2 | 4.906259 | 12 | 6.790443 |
| 20 | 1 | 1 | 25.964269 | 13 | 21.050876 |
| 20 | 2 | 2 | 8.023846 | 11 | 6.010604 |
| 20 | 3 | 3 | 7.044903 | 10 | 2.193767 |
| 20 | 4 | 2 | 6.037210 | 12 | 7.637049 |
| 21 | 1 | 1 | 26.883419 | 13 | 20.325103 |
| 21 | 2 | 2 | 8.828319 | 12 | 7.867839 |
| 21 | 3 | 4 | 7.009907 | 9 | 2.064282 |
| 21 | 4 | 1 | 6.008961 | 13 | 6.384830 |
| 22 | 1 | 1 | 24.955962 | 13 | 21.071610 |
| 22 | 2 | 2 | 7.294409 | 12 | 5.456228 |
| 22 | 3 | 3 | 6.825860 | 10 | 2.969116 |
| 22 | 4 | 2 | 5.216364 | 12 | 5.888669 |
| 23 | 1 | 1 | 26.976669 | 13 | 21.008825 |
| 23 | 2 | 2 | 8.765486 | 11 | 5.239563 |
| 23 | 3 | 4 | 7.288697 | 9 | 3.302166 |
| 23 | 4 | 2 | 5.734015 | 11 | 5.411714 |
| mean | 1 | 6 | 26.125342 | 79 | 21.061697 |
| mean | 2 | 12 | 8.214343 | 67 | 5.750921 |
| mean | 3 | 21 | 6.980205 | 58 | 2.821888 |
| mean | 4 | 10 | 5.492206 | 73 | 6.386647 |

### L = 2048

| seed | quarter | sleep visits | sleep mean reward | room visits | room mean reward |
|---:|---:|---:|---:|---:|---:|
| 18 | 1 | 8 | 9.752248 | 47 | 9.698907 |
| 18 | 2 | 9 | 2.974322 | 42 | 2.023158 |
| 18 | 3 | 21 | 1.633960 | 34 | 0.859517 |
| 18 | 4 | 46 | 1.122138 | 16 | 0.463273 |
| 19 | 1 | 8 | 8.722123 | 46 | 9.940059 |
| 19 | 2 | 9 | 2.836888 | 44 | 2.041612 |
| 19 | 3 | 20 | 1.604288 | 35 | 1.038707 |
| 19 | 4 | 43 | 1.157442 | 16 | 0.294962 |
| 20 | 1 | 8 | 9.402636 | 46 | 9.855659 |
| 20 | 2 | 10 | 2.927014 | 42 | 2.014033 |
| 20 | 3 | 27 | 1.668929 | 29 | 0.796601 |
| 20 | 4 | 46 | 1.176565 | 14 | 0.354127 |
| 21 | 1 | 8 | 9.823581 | 47 | 9.885525 |
| 21 | 2 | 8 | 2.840084 | 43 | 1.906769 |
| 21 | 3 | 23 | 1.706059 | 32 | 0.808701 |
| 21 | 4 | 46 | 1.154347 | 14 | 0.398882 |
| 22 | 1 | 8 | 8.806886 | 47 | 9.360200 |
| 22 | 2 | 9 | 2.986694 | 43 | 2.110636 |
| 22 | 3 | 22 | 1.579976 | 33 | 1.003404 |
| 22 | 4 | 47 | 1.188498 | 13 | 0.251892 |
| 23 | 1 | 9 | 9.458940 | 44 | 9.545415 |
| 23 | 2 | 8 | 3.028746 | 44 | 2.228126 |
| 23 | 3 | 21 | 1.715256 | 34 | 0.929932 |
| 23 | 4 | 50 | 1.186336 | 10 | 0.313556 |
| mean | 1 | 49 | 9.327736 | 277 | 9.714294 |
| mean | 2 | 53 | 2.932291 | 258 | 2.054055 |
| mean | 3 | 134 | 1.651411 | 197 | 0.906144 |
| mean | 4 | 278 | 1.164221 | 83 | 0.346115 |

## Mean held-out bits

Mean over seeds. W is the mean of the 7 learnable channels. W9 is the mean of all 9.

| controller | L | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | W | W9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prog | 512 | 4.972649 | 4.278950 | 4.879548 | 1.378415 | 1.320090 | 0.463464 | 2.083522 | 5.357895 | 0.017503 | 2.768091 | 2.750226 |
| prog_total | 512 | 4.970967 | 4.257840 | 4.889297 | 1.380294 | 1.327015 | 0.458579 | 1.995670 | 5.357887 | 0.019536 | 2.754237 | 2.739676 |
| prog_sleep | 512 | 4.834971 | 4.292310 | 4.881739 | 1.372338 | 1.312777 | 0.438288 | 2.191426 | 5.358184 | 0.017680 | 2.760550 | 2.744413 |
| fixed_sleep | 512 | 4.762840 | 4.286071 | 4.823748 | 1.376052 | 1.323686 | 0.451124 | 2.293082 | 5.358483 | 0.017883 | 2.759515 | 2.743663 |
| prog | 2048 | 1.462544 | 3.435038 | 3.761459 | 1.279822 | 1.219388 | 0.421249 | 0.637486 | 5.358853 | 0.007884 | 1.745284 | 1.953747 |
| prog_total | 2048 | 1.454588 | 3.473417 | 3.719417 | 1.281471 | 1.218265 | 0.402438 | 0.624064 | 5.358862 | 0.008353 | 1.739094 | 1.948986 |
| prog_sleep | 2048 | 1.985390 | 3.737615 | 3.667176 | 1.274740 | 1.214221 | 0.325776 | 0.652874 | 5.376508 | 0.005351 | 1.836827 | 2.026628 |
| fixed_sleep | 2048 | 1.811114 | 3.600633 | 3.707365 | 1.268178 | 1.204078 | 0.335595 | 0.644729 | 5.374601 | 0.005269 | 1.795956 | 1.994618 |

## Ring order

Channel in rooms 0..8, seeds 18–23. One order per seed; controllers share it.

| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---:|---|---|---|---|---|---|---|---|---|
| 18 | C1 | E110 | M7 | C7 | G | M37 | N | E30 | Z |
| 19 | Z | M7 | M37 | E110 | C7 | E30 | C1 | N | G |
| 20 | M7 | C7 | G | E30 | M37 | E110 | C1 | Z | N |
| 21 | E30 | G | C1 | M37 | C7 | Z | N | M7 | E110 |
| 22 | N | E110 | C7 | E30 | M7 | Z | M37 | G | C1 |
| 23 | N | G | M7 | E30 | Z | C1 | M37 | E110 | C7 |

## Walls

Smoke wall 45.894 s. Kernel wall 371.639 s.

| kernel | seed | wall seconds |
|---|---:|---:|
| ezzzzzx/m6-w3 | 18 | 61.284 |
| ezzzzzx/m6-w3 | 19 | 62.318 |
| ezzzzzx/m6-w3 | 20 | 60.333 |
| ezzzzzx/m6-w3 | 21 | 62.646 |
| ezzzzzx/m6-w3 | 22 | 62.542 |
| ezzzzzx/m6-w3 | 23 | 62.497 |

| controller | L | seed | wall seconds |
|---|---:|---:|---:|
| prog | 512 | 18 | 10.486 |
| prog | 512 | 19 | 9.314 |
| prog | 512 | 20 | 9.424 |
| prog | 512 | 21 | 9.604 |
| prog | 512 | 22 | 9.282 |
| prog | 512 | 23 | 9.023 |
| prog_total | 512 | 18 | 10.442 |
| prog_total | 512 | 19 | 10.283 |
| prog_total | 512 | 20 | 10.374 |
| prog_total | 512 | 21 | 10.446 |
| prog_total | 512 | 22 | 10.437 |
| prog_total | 512 | 23 | 10.405 |
| prog_sleep | 512 | 18 | 10.322 |
| prog_sleep | 512 | 19 | 10.162 |
| prog_sleep | 512 | 20 | 10.322 |
| prog_sleep | 512 | 21 | 10.301 |
| prog_sleep | 512 | 22 | 10.285 |
| prog_sleep | 512 | 23 | 10.262 |
| fixed_sleep | 512 | 18 | 10.468 |
| fixed_sleep | 512 | 19 | 9.992 |
| fixed_sleep | 512 | 20 | 10.125 |
| fixed_sleep | 512 | 21 | 10.247 |
| fixed_sleep | 512 | 22 | 10.112 |
| fixed_sleep | 512 | 23 | 10.144 |
| prog | 2048 | 18 | 29.530 |
| prog | 2048 | 19 | 31.527 |
| prog | 2048 | 20 | 32.708 |
| prog | 2048 | 21 | 33.589 |
| prog | 2048 | 22 | 33.062 |
| prog | 2048 | 23 | 32.377 |
| prog_total | 2048 | 18 | 40.899 |
| prog_total | 2048 | 19 | 37.019 |
| prog_total | 2048 | 20 | 38.848 |
| prog_total | 2048 | 21 | 40.827 |
| prog_total | 2048 | 22 | 39.926 |
| prog_total | 2048 | 23 | 42.670 |
| prog_sleep | 2048 | 18 | 39.440 |
| prog_sleep | 2048 | 19 | 40.099 |
| prog_sleep | 2048 | 20 | 37.529 |
| prog_sleep | 2048 | 21 | 40.117 |
| prog_sleep | 2048 | 22 | 40.354 |
| prog_sleep | 2048 | 23 | 39.197 |
| fixed_sleep | 2048 | 18 | 21.801 |
| fixed_sleep | 2048 | 19 | 22.202 |
| fixed_sleep | 2048 | 20 | 22.785 |
| fixed_sleep | 2048 | 21 | 22.509 |
| fixed_sleep | 2048 | 22 | 22.166 |
| fixed_sleep | 2048 | 23 | 23.279 |

## Smoke test

```
smoke check 0: packed score matches score_pairs, and the sleep plan fits
  packed score abs 0.000e+00 on 20 random reservoirs
smoke check 1: prog seed 12 L=512 vs results/m5bw3/results.json
  prog W 2.742358251650 ref 2.742358251650 abs 0.000e+00 W9 2.730350 finite True wall 15.83s
smoke check 2: a sleep visit changes no reservoir and no world stream
  reservoir identical True world identical True weights changed True score matches True
smoke check 3: prog_sleep seed 18 L=256 visit trace
  visit 1 arm 0 C1 decisions 8 reward 13.333785
  visit 2 arm 1 E110 decisions 9 reward 40.033632
  visit 3 arm 2 M7 decisions 9 reward 39.665422
  visit 4 arm 3 C7 decisions 9 reward 8.326143
  visit 5 arm 4 G decisions 9 reward 4.964389
  visit 6 arm 5 M37 decisions 9 reward 8.624992
  visit 7 arm 6 N decisions 9 reward -0.036305
  visit 8 arm 7 E30 decisions 9 reward 38.599526
  visit 9 arm 8 Z decisions 9 reward 42.769514
  visit 10 arm sleep sleep decisions 8 reward 27.679225
  visit 11 arm 8 Z decisions 8 reward 3.098607
  visit 12 arm 1 E110 decisions 10 reward 24.454071
  visit 13 arm 2 M7 decisions 9 reward 39.110522
  visit 14 arm 2 M7 decisions 8 reward 19.009360
  visit 15 arm 7 E30 decisions 12 reward 20.740445
  visit 16 arm 1 E110 decisions 11 reward 8.343117
  visit 17 arm 2 M7 decisions 9 reward 7.545544
  visit 18 arm 7 E30 decisions 12 reward 8.259619
  visit 19 arm 8 Z decisions 9 reward 0.759719
  visit 20 arm sleep sleep decisions 8 reward 9.463120
  visit 21 arm 1 E110 decisions 10 reward 3.501217
  visit 22 arm 7 E30 decisions 11 reward 3.674350
  visit 23 arm 2 M7 decisions 12 reward 2.772136
  visit 24 arm sleep sleep decisions 8 reward 7.518361
  visit 25 arm 8 Z decisions 11 reward 0.249898
  visit 26 arm 1 E110 decisions 10 reward 1.890222
  visit 27 arm 7 E30 decisions 10 reward 1.790301
  sleep_share 0.093750 sleep_decisions 24 / 256 W 3.353079 W9 3.206470 wall 9.71s
smoke check 4: seed 18 L=128, four controllers, W and W9 finite
  prog W 3.816928 W9 3.572406 finite True sleep 0.0000 corridor 0.1250 wall 4.37s learn 3.29s eval 1.08s
  prog_total W 3.816928 W9 3.572406 finite True sleep 0.0000 corridor 0.1250 wall 5.02s learn 3.81s eval 1.22s
  prog_sleep W 3.730575 W9 3.504540 finite True sleep 0.0625 corridor 0.1172 wall 5.25s learn 4.14s eval 1.11s
  fixed_sleep W 3.775554 W9 3.540182 finite True sleep 0.0625 corridor 0.1172 wall 4.86s learn 3.71s eval 1.14s
estimate one-seed kernel wall 162s (2.7 min) pool 4 limit 90 min
estimate is within 90 min per kernel
```

## Protocol

- `prog` is `m5_w3.run_controller` with the M5 per-channel reward and no sleep. W9 is `m5b_w3.score_nz` on that frozen compressor. `m5_w3.py`, `m5b_w3.py`, `curio.py` and `w3_world.py` are imported and not edited.
- `prog_total` is M5's 9-room bandit. `prog_sleep` is the same bandit over 10 arms. Sleep is the last arm of the optimistic first pass, then eps-greedy with EPS=0.1 and BETA=0.3. Choice rng is `default_rng(10_000 + seed)`. N-channel noise and corridor steps use `default_rng(1_000 + seed)`. The reservoir uses `default_rng(20_000 + seed)`.
- A decision's progress is the sum over the 9 room channels of `score_pairs` before the learning step minus `score_pairs` after it, on the reservoirs as they stand before the decision's own pairs enter. The visit reward is that sum over the visit's decisions, divided by the decisions the visit took. Travel counts. Corridor glances are learned and do not enter a room reservoir.
- The packed score used for that difference was checked against `SGDCompressor.score_pairs`. The smoke check reports the absolute gap.
- A sleep decision draws 64 pairs uniformly with replacement from the union of the room reservoirs, rng `default_rng(30_000 + seed)`, and applies the compressor's learning step to each pair on its own channel. Pairs enter no reservoir. An empty union learns nothing. A sleep visit is 8 such decisions where the agent stands, with no travel. Its reward is the summed progress divided by the decisions it took.
- `fixed_sleep` is the `prog_total` controller plus planned sleep visits. The visit lengths are 8, with a shorter last visit when the count is not a multiple of 8, and the lengths sum to `prog_sleep`'s sleep decisions on that seed and L. Awake gaps before, between and after the visits differ by at most one decision. A due sleep that falls inside a room visit starts on the next decision after that visit. A room visit is not started when the decisions left would not cover the sleeps still owed plus one maximum room visit (4 moves + 8 dwells).
- W is `w3_headroom.evaluate`. W9 adds N and Z on the same frozen compressor. N is EVAL uniform symbols from `default_rng(30000 + seed)`, a fresh generator, not the sleep rng. Z is EVAL space symbols. Corridor is not in W or W9.
- Quarter i is decisions [(i−1)L/4, iL/4). A visit reward is counted in the quarter of its first decision. Seeds 18–23. L = 2048 is the scored life. L = 512 is the report life. Nothing retuned.
- PYTHONHASHSEED=0. CURIO_LR=0.004. CPU. Private Kaggle kernels, pool of 4, no GPU, no internet.

## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (prog_sleep − prog_total ≤ −0.05, 5 of 6 seeds): FAIL.** The gap is +0.098; sleep makes the agent worse, on 6 of 6 seeds.
**Bar 2 (prog_sleep − fixed_sleep ≤ −0.03, 4 of 6): FAIL.** The gap is +0.041; choosing when to sleep is worse than a schedule.

**Mechanism: the agent was sleeping on its own exam.** The spec scored every reward on the reservoirs, and sleep
replayed pairs from those same reservoirs. A sleep visit therefore trains on the measurement set, and its measured
"progress" is memorisation. From quarter 2 on, sleep's reward runs 1.5–2.4× the room reward. The bandit follows that
reward, and sleep's share climbs to 72% of the last quarter. Frozen and scored on fresh data, the compressor is worse.
This is H23's lesson (a progress estimator that can be memorised gets memorised) in a new place. The spec error is
mine.

**A second cost:** a matched dose on a fixed schedule also loses to no sleep (1.796 vs 1.739). Replaying a 256-pair
reservoir per channel is not worth a third of the life's decisions to this learner. The store is too small to teach
more than the world does.

**Next (M6b, fresh seeds).** Split the store: each pair goes by coin flip either into a replay reservoir, which sleep
draws from, or into a disjoint exam reservoir, which all progress is scored on. Make the store larger (R = 2,048 per
channel), so that replay holds more than the agent can re-observe cheaply. The bars stay the same.
