# M7b-W3 per-channel curiosity, fixed sleep

The W3 loop on the trace-learned roots bank. W is bits/symbol of the frozen bank on a fresh continuation of the 7 learnable channels, from a zero state, with that channel's embedding. W9 is that mean over all 9 room channels. Lower is better. Forgetting is end-of-life exam bits/symbol minus the exam bits/symbol just after the last visit to that channel. Positive forgetting is a worse code. Numbers only.

Sleep dose for every controller whose name ends in `_s`: 408 decisions, share 0.199219, in visits of 8. `rand_dwell` does not sleep. `prog_chan_s` room reward is the sum, over that room's exam glances, of bits before minus bits after, summed over the visit's decisions and divided by those decisions. Travel counts. Sleep visits are not bandit updates.

## W

| controller | seed 36 | seed 37 | seed 38 | seed 39 | seed 40 | seed 41 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 0.756620 | 1.106138 | 0.814944 | 0.911582 | 1.103841 | 0.879686 | 0.928802 ± 0.133795 |
| rand_dwell_s | 0.656358 | 0.588062 | 0.720817 | 0.713926 | 0.799436 | 0.587246 | 0.677641 ± 0.076008 |
| oracle7_s | 0.619467 | 0.588043 | 0.536813 | 0.561806 | 0.638884 | 0.509228 | 0.575707 ± 0.045063 |
| prog_total_s | 0.566470 | 0.786109 | 0.511068 | 0.659119 | 0.803390 | 0.883011 | 0.701528 ± 0.133404 |
| prog_chan_s | 0.746707 | 0.683475 | 0.472447 | 0.568460 | 0.519784 | 0.673535 | 0.610735 ± 0.097388 |

## W9

| controller | seed 36 | seed 37 | seed 38 | seed 39 | seed 40 | seed 41 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 1.748322 | 1.965512 | 1.810945 | 1.665967 | 1.787984 | 1.838085 | 1.802802 ± 0.091011 |
| rand_dwell_s | 1.773508 | 1.635419 | 1.820021 | 1.766243 | 1.641754 | 1.568780 | 1.700954 ± 0.090345 |
| oracle7_s | 2.336681 | 2.448919 | 2.289034 | 2.234924 | 2.320297 | 2.377211 | 2.334511 ± 0.067212 |
| prog_total_s | 1.297571 | 1.827359 | 1.354171 | 1.482628 | 1.441409 | 1.651365 | 1.509084 ± 0.180536 |
| prog_chan_s | 1.332421 | 1.486023 | 1.253415 | 1.324438 | 1.312296 | 1.417436 | 1.354338 ± 0.075993 |

## Gaps

On W. A negative gap means the first controller is lower.

| seed | prog_chan_s − rand_dwell_s | prog_chan_s − prog_total_s | prog_chan_s − oracle7_s | rand_dwell_s − rand_dwell |
|---:|---:|---:|---:|---:|
| 36 | 0.090350 | 0.180237 | 0.127240 | -0.100262 |
| 37 | 0.095413 | -0.102634 | 0.095432 | -0.518075 |
| 38 | -0.248369 | -0.038620 | -0.064366 | -0.094127 |
| 39 | -0.145466 | -0.090659 | 0.006654 | -0.197656 |
| 40 | -0.279652 | -0.283605 | -0.119100 | -0.304405 |
| 41 | 0.086289 | -0.209476 | 0.164307 | -0.292440 |
| mean | -0.066906 | -0.090793 | 0.035028 | -0.251161 |
| std ddof=0 | 0.162741 | 0.145821 | 0.102717 | 0.144957 |
| left lower (count) | 3 of 6 | 5 of 6 | 2 of 6 | 6 of 6 |

## Forgetting

Exam-store bits/symbol just after the last visit that dwelled in the channel, minus nothing: the difference is end minus that snapshot. A channel with no exam glance at that visit is left out of the mean. n is the number of seed-channel pairs in the mean.

| controller | mean delta bps | std ddof=0 | n |
|---|---:|---:|---:|
| rand_dwell | 0.356478 | 0.317268 | 42 |
| rand_dwell_s | 0.170689 | 0.272565 | 42 |
| oracle7_s | 0.083750 | 0.133321 | 42 |
| prog_total_s | 0.137259 | 0.216527 | 42 |
| prog_chan_s | 0.189323 | 0.254994 | 42 |

| controller | channel | n | after bps | end bps | delta bps | std ddof=0 |
|---|---|---:|---:|---:|---:|---:|
| rand_dwell | G | 6 | 1.249069 | 1.536786 | 0.287718 | 0.180510 |
| rand_dwell | C1 | 6 | 0.456222 | 0.564089 | 0.107867 | 0.112991 |
| rand_dwell | C7 | 6 | 0.604636 | 0.789487 | 0.184850 | 0.187446 |
| rand_dwell | E30 | 6 | 0.943576 | 1.475494 | 0.531917 | 0.362542 |
| rand_dwell | E110 | 6 | 0.972900 | 1.409571 | 0.436671 | 0.346019 |
| rand_dwell | M7 | 6 | 0.376396 | 0.827070 | 0.450674 | 0.438295 |
| rand_dwell | M37 | 6 | 0.515288 | 1.010935 | 0.495646 | 0.139347 |
| rand_dwell_s | G | 6 | 1.166429 | 1.553954 | 0.387525 | 0.267797 |
| rand_dwell_s | C1 | 6 | 0.375941 | 0.389273 | 0.013332 | 0.014153 |
| rand_dwell_s | C7 | 6 | 0.536653 | 0.613782 | 0.077129 | 0.060738 |
| rand_dwell_s | E30 | 6 | 0.797756 | 1.319965 | 0.522209 | 0.350521 |
| rand_dwell_s | E110 | 6 | 0.991791 | 1.114477 | 0.122686 | 0.272811 |
| rand_dwell_s | M7 | 6 | 0.309345 | 0.337670 | 0.028324 | 0.024731 |
| rand_dwell_s | M37 | 6 | 0.429146 | 0.472762 | 0.043617 | 0.060956 |
| oracle7_s | G | 6 | 1.092392 | 1.179885 | 0.087493 | 0.123142 |
| oracle7_s | C1 | 6 | 0.463897 | 0.485549 | 0.021652 | 0.033100 |
| oracle7_s | C7 | 6 | 0.599012 | 0.654073 | 0.055061 | 0.070727 |
| oracle7_s | E30 | 6 | 0.796853 | 0.909477 | 0.112624 | 0.124148 |
| oracle7_s | E110 | 6 | 0.839090 | 0.839736 | 0.000646 | 0.075111 |
| oracle7_s | M7 | 6 | 0.307153 | 0.388047 | 0.080894 | 0.108354 |
| oracle7_s | M37 | 6 | 0.433144 | 0.661024 | 0.227880 | 0.192741 |
| prog_total_s | G | 6 | 1.407364 | 1.783745 | 0.376380 | 0.271130 |
| prog_total_s | C1 | 6 | 0.478946 | 0.517992 | 0.039046 | 0.136457 |
| prog_total_s | C7 | 6 | 0.569080 | 0.645751 | 0.076671 | 0.049676 |
| prog_total_s | E30 | 6 | 0.794211 | 1.073366 | 0.279155 | 0.265035 |
| prog_total_s | E110 | 6 | 1.023775 | 1.114908 | 0.091133 | 0.215366 |
| prog_total_s | M7 | 6 | 0.327371 | 0.372331 | 0.044960 | 0.068883 |
| prog_total_s | M37 | 6 | 0.431681 | 0.485146 | 0.053466 | 0.062595 |
| prog_chan_s | G | 6 | 0.894180 | 1.120558 | 0.226378 | 0.160878 |
| prog_chan_s | C1 | 6 | 0.379431 | 0.478370 | 0.098940 | 0.106804 |
| prog_chan_s | C7 | 6 | 0.509029 | 0.681720 | 0.172690 | 0.105554 |
| prog_chan_s | E30 | 6 | 0.903135 | 1.359977 | 0.456842 | 0.410729 |
| prog_chan_s | E110 | 6 | 1.098056 | 1.322258 | 0.224203 | 0.297263 |
| prog_chan_s | M7 | 6 | 0.328400 | 0.460321 | 0.131921 | 0.174503 |
| prog_chan_s | M37 | 6 | 0.467829 | 0.482117 | 0.014288 | 0.051716 |

## Sleep share

Sleep decisions divided by L.

| controller | seed 36 | seed 37 | seed 38 | seed 39 | seed 40 | seed 41 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| rand_dwell_s | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 ± 0.000000 |
| oracle7_s | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 ± 0.000000 |
| prog_total_s | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 ± 0.000000 |
| prog_chan_s | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 | 0.199219 ± 0.000000 |

## Decision shares

Mean over seeds. A stay counts for the room channel. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.

| controller | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 0.0833 | 0.0978 | 0.0833 | 0.0885 | 0.0900 | 0.0879 | 0.0820 | 0.0822 | 0.0846 | 0.2202 | 0.0000 |
| rand_dwell_s | 0.0663 | 0.0771 | 0.0671 | 0.0710 | 0.0749 | 0.0715 | 0.0671 | 0.0677 | 0.0634 | 0.1748 | 0.1992 |
| oracle7_s | 0.0915 | 0.0876 | 0.0892 | 0.0898 | 0.0965 | 0.0872 | 0.0846 | 0.0000 | 0.0000 | 0.1743 | 0.1992 |
| prog_total_s | 0.0898 | 0.0391 | 0.0684 | 0.1227 | 0.0827 | 0.0749 | 0.0671 | 0.1393 | 0.0182 | 0.0986 | 0.1992 |
| prog_chan_s | 0.1595 | 0.0456 | 0.0703 | 0.0749 | 0.0546 | 0.0393 | 0.0462 | 0.1925 | 0.0137 | 0.1042 | 0.1992 |

## prog_chan_s shares by quarter

Quarter i is decisions [(i−1)L/4, iL/4). A stay counts for the room channel. A move counts for the corridor. A sleep decision counts for sleep.

| seed | quarter | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor | sleep |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1 | 0.1250 | 0.0156 | 0.0781 | 0.0156 | 0.0156 | 0.1406 | 0.0938 | 0.1719 | 0.0156 | 0.1406 | 0.1875 |
| 36 | 2 | 0.1719 | 0.0000 | 0.1582 | 0.0000 | 0.0000 | 0.0469 | 0.0625 | 0.2500 | 0.0156 | 0.0918 | 0.2031 |
| 36 | 3 | 0.2207 | 0.0781 | 0.1230 | 0.0000 | 0.0000 | 0.0312 | 0.1094 | 0.1406 | 0.0156 | 0.0781 | 0.2031 |
| 36 | 4 | 0.0762 | 0.0469 | 0.0469 | 0.1562 | 0.0156 | 0.0215 | 0.0156 | 0.3438 | 0.0156 | 0.0586 | 0.2031 |
| 37 | 1 | 0.1094 | 0.0625 | 0.0781 | 0.0156 | 0.0625 | 0.0938 | 0.0410 | 0.1875 | 0.0156 | 0.1465 | 0.1875 |
| 37 | 2 | 0.1562 | 0.0469 | 0.0625 | 0.0312 | 0.1250 | 0.0312 | 0.0215 | 0.2109 | 0.0000 | 0.1113 | 0.2031 |
| 37 | 3 | 0.1562 | 0.0625 | 0.0625 | 0.0938 | 0.0312 | 0.0156 | 0.0469 | 0.2266 | 0.0000 | 0.1016 | 0.2031 |
| 37 | 4 | 0.1094 | 0.0469 | 0.0312 | 0.0781 | 0.1406 | 0.0156 | 0.1406 | 0.0938 | 0.0156 | 0.1250 | 0.2031 |
| 38 | 1 | 0.1562 | 0.0156 | 0.0781 | 0.0312 | 0.0625 | 0.0938 | 0.0312 | 0.2324 | 0.0156 | 0.0957 | 0.1875 |
| 38 | 2 | 0.2188 | 0.0312 | 0.0938 | 0.0156 | 0.0625 | 0.0312 | 0.0000 | 0.2363 | 0.0156 | 0.0918 | 0.2031 |
| 38 | 3 | 0.1562 | 0.0312 | 0.1094 | 0.0156 | 0.1406 | 0.0469 | 0.1543 | 0.0312 | 0.0000 | 0.1113 | 0.2031 |
| 38 | 4 | 0.2344 | 0.0312 | 0.1094 | 0.0566 | 0.0156 | 0.0000 | 0.0176 | 0.2500 | 0.0000 | 0.0820 | 0.2031 |
| 39 | 1 | 0.2344 | 0.0781 | 0.0469 | 0.0156 | 0.0156 | 0.0469 | 0.0469 | 0.1992 | 0.0156 | 0.1133 | 0.1875 |
| 39 | 2 | 0.1250 | 0.0469 | 0.0781 | 0.0312 | 0.0938 | 0.0938 | 0.0625 | 0.1094 | 0.0312 | 0.1250 | 0.2031 |
| 39 | 3 | 0.1562 | 0.0469 | 0.0469 | 0.1992 | 0.0000 | 0.0156 | 0.0000 | 0.1758 | 0.0469 | 0.1094 | 0.2031 |
| 39 | 4 | 0.1719 | 0.0469 | 0.0469 | 0.0039 | 0.0938 | 0.0312 | 0.0312 | 0.2598 | 0.0156 | 0.0957 | 0.2031 |
| 40 | 1 | 0.2031 | 0.1250 | 0.0469 | 0.0312 | 0.0156 | 0.0156 | 0.0156 | 0.2031 | 0.0156 | 0.1406 | 0.1875 |
| 40 | 2 | 0.1406 | 0.0625 | 0.0781 | 0.1113 | 0.0000 | 0.0000 | 0.0625 | 0.2188 | 0.0000 | 0.1230 | 0.2031 |
| 40 | 3 | 0.0781 | 0.0469 | 0.0781 | 0.1699 | 0.0469 | 0.0000 | 0.0469 | 0.2168 | 0.0156 | 0.0977 | 0.2031 |
| 40 | 4 | 0.1719 | 0.0156 | 0.0156 | 0.2656 | 0.0605 | 0.0156 | 0.0156 | 0.1582 | 0.0000 | 0.0781 | 0.2031 |
| 41 | 1 | 0.1875 | 0.0781 | 0.0781 | 0.0156 | 0.0156 | 0.0156 | 0.0469 | 0.2344 | 0.0312 | 0.1094 | 0.1875 |
| 41 | 2 | 0.0938 | 0.0469 | 0.0801 | 0.1094 | 0.0156 | 0.0781 | 0.0312 | 0.2188 | 0.0000 | 0.1230 | 0.2031 |
| 41 | 3 | 0.2031 | 0.0176 | 0.0605 | 0.1406 | 0.2031 | 0.0312 | 0.0156 | 0.0000 | 0.0312 | 0.0938 | 0.2031 |
| 41 | 4 | 0.1719 | 0.0137 | 0.0000 | 0.1934 | 0.0781 | 0.0312 | 0.0000 | 0.2500 | 0.0000 | 0.0586 | 0.2031 |
| mean | 1 | 0.1693 | 0.0625 | 0.0677 | 0.0208 | 0.0312 | 0.0677 | 0.0459 | 0.2048 | 0.0182 | 0.1243 | 0.1875 |
| mean | 2 | 0.1510 | 0.0391 | 0.0918 | 0.0498 | 0.0495 | 0.0469 | 0.0400 | 0.2074 | 0.0104 | 0.1110 | 0.2031 |
| mean | 3 | 0.1618 | 0.0472 | 0.0801 | 0.1032 | 0.0703 | 0.0234 | 0.0622 | 0.1318 | 0.0182 | 0.0986 | 0.2031 |
| mean | 4 | 0.1559 | 0.0335 | 0.0417 | 0.1257 | 0.0674 | 0.0192 | 0.0368 | 0.2259 | 0.0078 | 0.0830 | 0.2031 |

## prog_chan_s per-channel reward by quarter

Room reward is per-channel exam bits saved per decision, mean over the room visits whose first decision falls in the quarter. Sleep reward is whole-store exam bits saved per decision on sleep visits. Sleep visits do not update the bandit. A quarter with no visits of that kind is blank. The mean row's visit count is the sum across seeds. The mean reward is the mean of the per-seed means, over seeds that have at least one such visit.

| seed | quarter | sleep visits | sleep mean reward | room visits | room mean per-channel reward |
|---:|---:|---:|---:|---:|---:|
| 36 | 1 | 12 | 134.357093 | 43 | 63.281637 |
| 36 | 2 | 13 | 84.728776 | 46 | 28.516673 |
| 36 | 3 | 13 | 40.290905 | 46 | 16.318372 |
| 36 | 4 | 13 | 49.444355 | 47 | 16.315928 |
| 37 | 1 | 12 | 63.186356 | 43 | 62.557547 |
| 37 | 2 | 13 | 119.662862 | 44 | 33.783828 |
| 37 | 3 | 13 | 77.775139 | 44 | 24.600407 |
| 37 | 4 | 13 | 142.504485 | 44 | 20.497321 |
| 38 | 1 | 12 | 123.730941 | 46 | 59.959714 |
| 38 | 2 | 13 | 65.758193 | 45 | 29.392191 |
| 38 | 3 | 13 | 59.545152 | 44 | 20.443494 |
| 38 | 4 | 13 | 94.677553 | 46 | 23.039176 |
| 39 | 1 | 12 | 96.680261 | 45 | 59.930653 |
| 39 | 2 | 13 | 53.003013 | 43 | 24.129732 |
| 39 | 3 | 13 | 87.065975 | 44 | 22.727410 |
| 39 | 4 | 13 | 153.380567 | 45 | 22.719526 |
| 40 | 1 | 12 | 98.232705 | 43 | 51.387709 |
| 40 | 2 | 13 | 86.508396 | 44 | 29.666001 |
| 40 | 3 | 13 | 19.910816 | 44 | 22.806814 |
| 40 | 4 | 13 | 12.863524 | 46 | 28.161290 |
| 41 | 1 | 12 | 91.578235 | 46 | 56.335417 |
| 41 | 2 | 13 | 96.445502 | 43 | 27.606957 |
| 41 | 3 | 13 | 99.056256 | 45 | 24.291548 |
| 41 | 4 | 13 | 70.791464 | 47 | 30.750083 |
| mean | 1 | 72 | 101.294265 | 266 | 58.908779 |
| mean | 2 | 78 | 84.351124 | 265 | 28.849230 |
| mean | 3 | 78 | 63.940707 | 267 | 21.864674 |
| mean | 4 | 78 | 87.276991 | 275 | 23.580554 |

## Mean held-out bits

Mean over seeds. W is the mean of the 7 learnable channels. W9 is the mean of all 9.

| controller | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | W | W9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 1.863195 | 0.167756 | 0.330588 | 1.145667 | 1.245919 | 0.798319 | 0.950167 | 9.721751 | 0.001860 | 0.928802 | 1.802802 |
| rand_dwell_s | 1.981188 | 0.090956 | 0.167324 | 1.083188 | 0.770224 | 0.306192 | 0.344413 | 10.563426 | 0.001677 | 0.677641 | 1.700954 |
| oracle7_s | 1.652984 | 0.107133 | 0.110517 | 0.578153 | 0.532454 | 0.440219 | 0.608489 | 16.851202 | 0.129450 | 0.575707 | 2.334511 |
| prog_total_s | 2.089922 | 0.358760 | 0.191773 | 0.695574 | 0.880406 | 0.347104 | 0.347155 | 8.668154 | 0.002906 | 0.701528 | 1.509084 |
| prog_chan_s | 1.152192 | 0.157071 | 0.201450 | 0.923315 | 1.036760 | 0.440152 | 0.364205 | 7.908528 | 0.005370 | 0.610735 | 1.354338 |

## Per-seed held-out bits

### rand_dwell

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.984969 | 0.221679 | 0.216776 | 0.856054 | 0.805804 | 0.297314 | 0.913742 | 10.436519 | 0.002042 |
| 37 | 1.339608 | 0.062205 | 0.500987 | 0.674252 | 2.314240 | 1.729606 | 1.122065 | 9.944323 | 0.002326 |
| 38 | 2.007637 | 0.148327 | 0.104553 | 0.473276 | 1.086054 | 0.729262 | 1.155497 | 10.592285 | 0.001612 |
| 39 | 2.213583 | 0.242242 | 0.193284 | 1.473141 | 1.017496 | 0.630667 | 0.610657 | 8.610198 | 0.002429 |
| 40 | 2.417412 | 0.115997 | 0.148961 | 1.817955 | 1.509267 | 1.083954 | 0.633342 | 8.363069 | 0.001896 |
| 41 | 1.215959 | 0.216088 | 0.818968 | 1.579324 | 0.742656 | 0.319111 | 1.265696 | 10.384110 | 0.000855 |

### rand_dwell_s

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.674243 | 0.095171 | 0.063672 | 0.450394 | 1.641126 | 0.264148 | 0.405749 | 11.364711 | 0.002354 |
| 37 | 1.268457 | 0.091334 | 0.109392 | 1.602952 | 0.417016 | 0.318016 | 0.309271 | 10.599241 | 0.003094 |
| 38 | 2.515847 | 0.053293 | 0.244577 | 1.001251 | 0.644565 | 0.314014 | 0.272168 | 11.332918 | 0.001557 |
| 39 | 2.403769 | 0.119241 | 0.158135 | 1.161769 | 0.505014 | 0.340030 | 0.309522 | 10.897700 | 0.001004 |
| 40 | 2.633497 | 0.061027 | 0.150811 | 1.340404 | 0.870959 | 0.240605 | 0.298749 | 9.178147 | 0.001586 |
| 41 | 1.391315 | 0.125669 | 0.277356 | 0.942356 | 0.542666 | 0.360339 | 0.471018 | 10.007838 | 0.000467 |

### oracle7_s

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.915393 | 0.096299 | 0.132968 | 0.258198 | 0.704409 | 0.331619 | 0.897386 | 16.575135 | 0.118726 |
| 37 | 1.805908 | 0.066515 | 0.076234 | 0.814190 | 0.266229 | 0.464333 | 0.622892 | 17.865133 | 0.058834 |
| 38 | 1.765765 | 0.101817 | 0.116959 | 0.408929 | 0.546608 | 0.398324 | 0.419288 | 16.669172 | 0.174447 |
| 39 | 1.527849 | 0.225279 | 0.113793 | 0.656921 | 0.389283 | 0.532697 | 0.486819 | 16.123835 | 0.057843 |
| 40 | 1.582376 | 0.080439 | 0.071427 | 0.920591 | 0.936018 | 0.563773 | 0.317566 | 16.403224 | 0.007262 |
| 41 | 1.320613 | 0.072450 | 0.151718 | 0.410087 | 0.352176 | 0.350568 | 0.906983 | 17.470716 | 0.359587 |

### prog_total_s

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.408323 | 0.383954 | 0.157945 | 0.618465 | 0.771039 | 0.261557 | 0.364008 | 7.707709 | 0.005136 |
| 37 | 3.210711 | 0.070405 | 0.118015 | 0.180603 | 1.130250 | 0.353980 | 0.438802 | 10.941478 | 0.001986 |
| 38 | 1.206215 | 0.148009 | 0.138510 | 0.942952 | 0.490448 | 0.201921 | 0.449417 | 8.607110 | 0.002957 |
| 39 | 2.718681 | 0.176181 | 0.073864 | 0.561592 | 0.506430 | 0.366448 | 0.210637 | 8.727846 | 0.001973 |
| 40 | 1.878140 | 0.888312 | 0.348423 | 1.684736 | 0.172334 | 0.294273 | 0.357511 | 7.345776 | 0.003181 |
| 41 | 2.117465 | 0.485702 | 0.313879 | 0.185095 | 2.211935 | 0.604445 | 0.262558 | 8.679005 | 0.002203 |

### prog_chan_s

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1.460832 | 0.101462 | 0.233055 | 1.231786 | 1.472862 | 0.329515 | 0.397439 | 6.762216 | 0.002619 |
| 37 | 1.113473 | 0.097765 | 0.145357 | 1.231021 | 1.237070 | 0.584705 | 0.374936 | 8.571101 | 0.018777 |
| 38 | 1.153574 | 0.315955 | 0.231183 | 0.491349 | 0.455277 | 0.256005 | 0.403788 | 7.970751 | 0.002854 |
| 39 | 1.157860 | 0.171235 | 0.244583 | 1.426147 | 0.394792 | 0.289128 | 0.295475 | 7.938563 | 0.002160 |
| 40 | 1.040499 | 0.050159 | 0.059904 | 1.025749 | 0.431625 | 0.698304 | 0.332251 | 8.168709 | 0.003464 |
| 41 | 0.986914 | 0.205852 | 0.294616 | 0.133838 | 2.228933 | 0.483253 | 0.381339 | 8.039830 | 0.002346 |

## M7 reference

M7-W3, seeds 30–35, L = 2048, same cortex. `prog_sleep` chose its sleep. `fixed_sleep` used that seed's prog_sleep count. This run's dose is 0.20 for every `_s` controller.

### M7 W

| controller | seed 30 | seed 31 | seed 32 | seed 33 | seed 34 | seed 35 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 0.804654 | 0.704805 | 0.671865 | 0.649118 | 0.710645 | 1.268567 | 0.801609 ± 0.214395 |
| oracle7 | 0.868493 | 1.222725 | 1.124020 | 0.827258 | 0.905792 | 0.802698 | 0.958498 ± 0.157898 |
| prog_total | 0.894627 | 0.931975 | 1.278002 | 1.408650 | 0.814759 | 1.058003 | 1.064336 ± 0.213251 |
| prog_sleep | 0.495599 | 0.533178 | 1.145291 | 1.200333 | 0.666642 | 0.539888 | 0.763488 ± 0.294616 |
| fixed_sleep | 0.897702 | 0.530045 | 0.807006 | 0.745396 | 0.605543 | 0.811784 | 0.732913 ± 0.126755 |

### M7 W9

| controller | seed 30 | seed 31 | seed 32 | seed 33 | seed 34 | seed 35 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 1.683813 | 1.592825 | 1.608719 | 1.645277 | 1.649447 | 1.928649 | 1.684788 ± 0.112933 |
| oracle7 | 2.464598 | 2.710732 | 2.645289 | 2.459652 | 2.481604 | 2.337826 | 2.516617 ± 0.124753 |
| prog_total | 2.222954 | 1.597342 | 1.804555 | 2.014114 | 1.717394 | 1.708770 | 1.844188 ± 0.211840 |
| prog_sleep | 1.506560 | 1.807790 | 2.444152 | 1.917340 | 1.613687 | 1.436921 | 1.787741 ± 0.336775 |
| fixed_sleep | 1.472984 | 1.608655 | 1.743396 | 1.894097 | 1.319173 | 1.438382 | 1.579448 ± 0.193956 |

### M7 gaps, forgetting, sleep share

| gap | mean | std ddof=0 | left lower |
|---|---:|---:|---:|
| prog_total_minus_rand_dwell | 0.262727 | 0.327876 | 1 of 6 |
| prog_total_minus_oracle7 | 0.105838 | 0.274834 | 2 of 6 |
| prog_sleep_minus_prog_total | -0.300848 | 0.145257 | 6 of 6 |
| prog_sleep_minus_fixed_sleep | 0.030576 | 0.304179 | 2 of 6 |

| controller | forgetting mean | forgetting std | n | sleep share |
|---|---:|---:|---:|---:|
| rand_dwell | 0.284698 | 0.468117 | 42 | 0.000000 ± 0.000000 |
| oracle7 | 0.381058 | 0.547088 | 42 | 0.000000 ± 0.000000 |
| prog_total | 0.611238 | 0.746440 | 42 | 0.000000 ± 0.000000 |
| prog_sleep | 0.234550 | 0.637066 | 42 | 0.211019 ± 0.107250 |
| fixed_sleep | 0.239063 | 0.372335 | 42 | 0.211019 ± 0.107250 |

## Ring order

Channel in rooms 0..8. One order per seed; controllers share it.

| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---:|---|---|---|---|---|---|---|---|---|
| 36 | Z | G | C7 | M37 | M7 | E110 | C1 | E30 | N |
| 37 | E30 | G | C7 | Z | C1 | M37 | E110 | M7 | N |
| 38 | C1 | E110 | E30 | Z | M7 | C7 | M37 | G | N |
| 39 | Z | M37 | M7 | E30 | N | E110 | C1 | C7 | G |
| 40 | M7 | G | C1 | E30 | C7 | M37 | E110 | N | Z |
| 41 | E110 | G | C7 | E30 | M7 | M37 | Z | C1 | N |

## Walls

| kernel | seed | wall seconds |
|---|---:|---:|
| ezzzzzx/m7b-w3-s36 | 36 | 229.694 |
| ezzzzzx/m7b-w3-s37 | 37 | 538.134 |
| ezzzzzx/m7b-w3-s38 | 38 | 375.838 |
| ezzzzzx/m7b-w3-s39 | 39 | 439.111 |
| ezzzzzx/m7b-w3-s40 | 40 | 396.162 |
| ezzzzzx/m7b-w3-s41 | 41 | 280.320 |

| controller | seed | wall seconds | learn seconds | score seconds |
|---|---:|---:|---:|---:|
| rand_dwell | 36 | 48.950 | 38.169 | 9.707 |
| rand_dwell | 37 | 121.346 | 105.110 | 14.791 |
| rand_dwell | 38 | 101.192 | 87.780 | 12.352 |
| rand_dwell | 39 | 102.686 | 88.824 | 12.726 |
| rand_dwell | 40 | 94.209 | 81.195 | 11.945 |
| rand_dwell | 41 | 53.398 | 42.555 | 9.616 |
| rand_dwell_s | 36 | 117.199 | 47.297 | 61.666 |
| rand_dwell_s | 37 | 251.461 | 127.844 | 109.404 |
| rand_dwell_s | 38 | 187.722 | 97.977 | 79.646 |
| rand_dwell_s | 39 | 209.587 | 103.548 | 94.605 |
| rand_dwell_s | 40 | 190.059 | 93.845 | 84.183 |
| rand_dwell_s | 41 | 131.927 | 54.307 | 68.316 |
| oracle7_s | 36 | 112.556 | 48.212 | 56.134 |
| oracle7_s | 37 | 251.271 | 135.097 | 101.991 |
| oracle7_s | 38 | 180.353 | 98.473 | 71.724 |
| oracle7_s | 39 | 206.792 | 108.947 | 86.323 |
| oracle7_s | 40 | 188.864 | 103.460 | 73.302 |
| oracle7_s | 41 | 133.018 | 59.780 | 63.882 |
| prog_total_s | 36 | 227.661 | 43.614 | 176.059 |
| prog_total_s | 37 | 534.144 | 148.385 | 372.095 |
| prog_total_s | 38 | 373.139 | 107.990 | 255.213 |
| prog_total_s | 39 | 436.003 | 102.471 | 322.387 |
| prog_total_s | 40 | 393.116 | 109.273 | 272.054 |
| prog_total_s | 41 | 277.642 | 61.881 | 206.533 |
| prog_chan_s | 36 | 180.287 | 47.348 | 124.797 |
| prog_chan_s | 37 | 371.625 | 127.041 | 230.180 |
| prog_chan_s | 38 | 268.956 | 96.904 | 162.046 |
| prog_chan_s | 39 | 306.446 | 100.374 | 194.320 |
| prog_chan_s | 40 | 277.203 | 97.776 | 167.133 |
| prog_chan_s | 41 | 201.961 | 53.073 | 139.593 |

## Smoke test

```
torch=2.15.0+cpu threads=1 model_seed=0
dose L2048 408/2048 share 0.199219; L512 104/512 share 0.203125; L256 48/256 share 0.187500; L64 16/64 share 0.250000
smoke check 1: rand_dwell seed 30 L=64 versus m7_w3.run_controller
  m7 W 9.92232653 W9 9.47491701 m7b W 9.92232653 W9 9.47491701 abs_dW 0.000e+00 abs_dW9 0.000e+00 limit 1.0e-06 ok True
  published M7 smoke W 9.922327 W9 9.474917; live m7 minus published W -4.728e-07
smoke check 2: rand_dwell_s seed 36 L=512 sleep share
  sleep_decisions 104 target 104 share 0.203125 target 0.20 tol 0.02 intact True wall 40.56s
smoke check 3: prog_chan_s seed 36 L=256 visit trace
  visits 29 sleep_decisions 48 share 0.187500 intact True wall 40.82s
  visit 0 room Z decisions 8 travel 0 dwell 8 per_channel_reward -0.026663 whole_store_progress -0.026663
  visit 1 room G decisions 9 travel 1 dwell 8 per_channel_reward 104.139946 whole_store_progress 95.972628
  visit 2 room C7 decisions 9 travel 1 dwell 8 per_channel_reward 100.696021 whole_store_progress 166.118071
  visit 3 room M37 decisions 9 travel 1 dwell 8 per_channel_reward 169.673211 whole_store_progress 260.790805
  visit 4 room sleep decisions 8 per_channel_reward none whole_store_progress 8.290987
  visit 5 room M7 decisions 9 travel 1 dwell 8 per_channel_reward 107.418582 whole_store_progress 65.035301
  visit 6 room E110 decisions 9 travel 1 dwell 8 per_channel_reward 6.328958 whole_store_progress -62.297664
  visit 7 room C1 decisions 9 travel 1 dwell 8 per_channel_reward 21.770727 whole_store_progress -14.559384
  visit 8 room sleep decisions 8 per_channel_reward none whole_store_progress 31.701538
  visit 9 room E30 decisions 9 travel 1 dwell 8 per_channel_reward 9.018484 whole_store_progress -1.392310
  visit 10 room N decisions 9 travel 1 dwell 8 per_channel_reward 100.384703 whole_store_progress 94.010705
  visit 11 room C7 decisions 11 travel 3 dwell 8 per_channel_reward 53.704233 whole_store_progress 100.036948
  visit 12 room sleep decisions 8 per_channel_reward none whole_store_progress 116.721351
  visit 13 room M37 decisions 9 travel 1 dwell 8 per_channel_reward 23.472123 whole_store_progress -10.309344
  visit 14 room M37 decisions 8 travel 0 dwell 8 per_channel_reward 3.013545 whole_store_progress -19.321021
  visit 15 room M7 decisions 9 travel 1 dwell 8 per_channel_reward 11.436953 whole_store_progress -143.071910
  visit 16 room G decisions 11 travel 3 dwell 8 per_channel_reward 78.190886 whole_store_progress -338.319099
  visit 17 room sleep decisions 8 per_channel_reward none whole_store_progress 517.785446
  visit 18 room N decisions 10 travel 2 dwell 8 per_channel_reward 28.442410 whole_store_progress 254.900095
  visit 19 room G decisions 10 travel 2 dwell 8 per_channel_reward 76.933380 whole_store_progress 128.882230
  visit 20 room G decisions 8 travel 0 dwell 8 per_channel_reward 90.290774 whole_store_progress 93.472603
  visit 21 room sleep decisions 8 per_channel_reward none whole_store_progress 52.766288
  visit 22 room G decisions 8 travel 0 dwell 8 per_channel_reward 22.749961 whole_store_progress -33.383466
  visit 23 room M37 decisions 10 travel 2 dwell 8 per_channel_reward 12.826191 whole_store_progress -13.420999
  visit 24 room C7 decisions 9 travel 1 dwell 8 per_channel_reward 37.506574 whole_store_progress -75.156152
  visit 25 room sleep decisions 8 per_channel_reward none whole_store_progress 147.477051
  visit 26 room N decisions 11 travel 3 dwell 8 per_channel_reward 51.753864 whole_store_progress 20.999546
  visit 27 room M7 decisions 12 travel 4 dwell 8 per_channel_reward 1.119564 whole_store_progress 9.460374
  visit 28 room C7 decisions 2 travel 2 dwell 0 per_channel_reward -8.133926 whole_store_progress -355.975626
smoke check 4: five controllers, seed 36, L=64, W and W9 finite
  rand_dwell W 7.548252 W9 7.972180 finite True sleep 0.0000 decisions 0 corridor 0.2500 wall 4.86s learn 2.80s score 0.73s
  rand_dwell_s W 13.654479 W9 12.786797 finite True sleep 0.2500 decisions 16 corridor 0.1875 wall 5.47s learn 2.86s score 1.34s
  oracle7_s W 12.284270 W9 11.593358 finite True sleep 0.2500 decisions 16 corridor 0.1875 wall 5.27s learn 2.81s score 1.26s
  prog_total_s W 7.461253 W9 7.869383 finite True sleep 0.2500 decisions 16 corridor 0.0781 wall 7.09s learn 2.79s score 2.99s
  prog_chan_s W 7.461253 W9 7.869383 finite True sleep 0.2500 decisions 16 corridor 0.0781 wall 7.02s learn 2.73s score 3.04s
  learn_per_decision 0.0447s full_rescore_288 0.2063s channel_rescore_32 0.0496s
  pessimistic prog_total_s 514.0s prog_chan_s 349.1s rand_dwell_s 175.7s queued rand_dwell 91.6s kernel 514.0s (8.57 min) limit 5400s
  estimate runs prog_total_s and prog_chan_s in a pool of 4 and charges a full exam store on every whole-store rescore; rand_dwell starts when the first wave frees a worker
```

## Protocol

- `BankCortex`, the stores, the sleep decision, W, W9 and forgetting are `m7_w3`'s. Model seed 0. Glance length 64. Reservoirs hold 32 glances per channel. Exam coin `default_rng(40_000 + seed)`, exam replacement `default_rng(20_000 + seed)`, replay replacement `default_rng(50_000 + seed)`, sleep draw `default_rng(30_000 + seed)`.
- `rand_dwell` is `m7_w3.run_controller(seed, 'rand_dwell', L)`. `rand_dwell_s` and `oracle7_s` use `w3_headroom.next_action` and `default_rng(1_000 + seed)`. A sleep visit starts only when the current plan is empty, which is the end of a visit.
- `prog_total_s` is `m7_w3.run_bandit` kind `fixed_sleep` with this dose, then renamed. Its room reward is the whole exam store. `prog_chan_s` uses the same 9-room bandit (`default_rng(10_000 + seed)`, optimistic first pass, EPS 0.1, BETA 0.3) and the same dose. Its room reward sums bits saved on the destination room's exam glances over the visit, including corridor steps, and divides by the visit's decisions. Sleep does not call the EMA.
- The dose is 8 times the integer nearest to 0.20 L / 8. At L = 2048 that is 408 decisions, share 0.199219. Placement is `m7_w3.sleep_plan` and `m7_w3.sleep_due`.
- A forgetting snapshot is taken at the end of a dwell visit, before a sleep that follows it. World seeds 36, 37, 38, 39, 40, 41. L = 2048. CPU. Private Kaggle kernels, one seed each, pool of 4, no GPU, no internet. Nothing retuned.

## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (prog_chan_s − rand_dwell_s ≤ −0.10, 5 of 6): FAIL.** −0.067 ± 0.163, 3 of 6.
**Bar 2 (prog_chan_s − prog_total_s ≤ −0.10): FAIL, narrowly.** −0.091, 5 of 6.

**What holds.**
- **The sleep gain replicates on fresh seeds:** rand_dwell_s − rand_dwell = −0.251, 6 of 6. Forgetting halves (0.36 →
  0.17).
- **With sleep in place, allocation matters again.** In M7 the learnability oracle lost to random dwelling, because
  retention dominated. Here oracle7_s is the best controller (0.576), 0.10 ahead of rand_dwell_s. prog_chan_s sits
  between them (0.611), and the per-room reward recovers most of what the whole-store reward lost (5 of 6 seeds
  better than prog_total_s).
- Over all nine rooms (W9), prog_chan_s is clearly best (1.35, against 1.70 for rand_dwell_s and 2.33 for oracle7_s). Part of its time
  goes to rooms that W does not score.

**Process error (mine):** I set bar 1 at 0.10 without a headroom check on this cortex. The oracle's own gain over
rand_dwell_s turned out to be 0.10, so the bar asked prog to match the oracle on 5 of 6 seeds with a seed std of 0.16.
That was a near-certain kill. Our protocol says to check oracle headroom first, and I skipped it.

**Next (M7c).** In M5b, curiosity's edge grew with life length, so headroom and power come first: L ∈ {2,048, 4,096},
12 fresh seeds, and a bar that is void unless the oracle's headroom at L = 4,096 is at least 0.15.
