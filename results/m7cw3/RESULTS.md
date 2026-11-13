# M7c-W3 curiosity on the bank, fixed sleep, longer lives

The W3 loop on the trace-learned roots bank. W is bits/symbol of the frozen bank on a fresh continuation of the 7 learnable channels, from a zero state, with that channel's embedding. W9 is that mean over all 9 room channels. Lower is better. Forgetting is end-of-life exam bits/symbol minus the exam bits/symbol just after the last visit to that channel. Positive forgetting is a worse code. Numbers only.

Controllers are `m7b_w3`'s `rand_dwell_s`, `oracle7_s` and `prog_chan_s`. Sleep dose is 8 times the integer nearest to 0.20 L / 8, in visits of 8. At L = 2048 that is 408 decisions, share 0.199219. At L = 4096 that is 816 decisions, share 0.199219. `prog_chan_s` room reward is the sum, over that room's exam glances, of bits before minus bits after, summed over the visit's decisions and divided by those decisions. Travel counts. Sleep visits are not bandit updates.

## W, L = 2048

| controller | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | seed 47 | seed 48 | seed 49 | seed 50 | seed 51 | seed 52 | seed 53 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 0.636124 | 0.689163 | 0.664412 | 0.795032 | 0.594871 | 0.792141 | 0.741897 | 0.720146 | 0.703650 | 0.500778 | 0.562864 | 0.565778 | 0.663905 ± 0.089948 |
| oracle7_s | 0.863302 | 0.617026 | 0.547742 | 0.483488 | 0.740540 | 0.664137 | 0.718942 | 0.715751 | 0.511838 | 0.749339 | 0.518984 | 0.494000 | 0.635424 ± 0.119076 |
| prog_chan_s | 0.844562 | 0.705678 | 0.488882 | 0.744687 | 0.568635 | 0.445198 | 0.590340 | 0.526793 | 0.841722 | 0.911758 | 0.769425 | 0.617745 | 0.671285 ± 0.146721 |

## W9, L = 2048

| controller | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | seed 47 | seed 48 | seed 49 | seed 50 | seed 51 | seed 52 | seed 53 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 1.684091 | 1.585025 | 1.696495 | 1.635441 | 1.702572 | 1.781541 | 1.792826 | 1.681685 | 1.760506 | 1.515149 | 1.460921 | 1.668821 | 1.663756 ± 0.097236 |
| oracle7_s | 2.562505 | 2.482371 | 2.250556 | 2.216731 | 2.416665 | 2.486787 | 2.495922 | 2.367800 | 2.143938 | 2.438428 | 2.295594 | 2.188870 | 2.362181 ± 0.133079 |
| prog_chan_s | 1.397336 | 1.348643 | 1.331842 | 1.331667 | 1.210237 | 1.168347 | 1.227204 | 1.191023 | 1.479382 | 1.599569 | 1.391549 | 1.290086 | 1.330574 ± 0.120970 |

## Gaps, L = 2048

On W. A negative gap means the first controller is lower.

| seed | prog_chan_s − rand_dwell_s | oracle7_s − rand_dwell_s | prog_chan_s − oracle7_s |
|---:|---:|---:|---:|
| 42 | 0.208438 | 0.227178 | -0.018740 |
| 43 | 0.016515 | -0.072137 | 0.088653 |
| 44 | -0.175530 | -0.116670 | -0.058860 |
| 45 | -0.050345 | -0.311544 | 0.261199 |
| 46 | -0.026236 | 0.145669 | -0.171905 |
| 47 | -0.346943 | -0.128004 | -0.218939 |
| 48 | -0.151557 | -0.022955 | -0.128602 |
| 49 | -0.193353 | -0.004395 | -0.188958 |
| 50 | 0.138072 | -0.191813 | 0.329884 |
| 51 | 0.410980 | 0.248561 | 0.162419 |
| 52 | 0.206561 | -0.043880 | 0.250441 |
| 53 | 0.051967 | -0.071778 | 0.123744 |
| mean | 0.007381 | -0.028481 | 0.035861 |
| std ddof=0 | 0.202312 | 0.158523 | 0.184263 |
| left lower (count) | 6 of 12 | 9 of 12 | 6 of 12 |

## Forgetting, L = 2048

Exam-store bits/symbol just after the last visit that dwelled in the channel, minus nothing: the difference is end minus that snapshot. A channel with no exam glance at that visit is left out of the mean. n is the number of seed-channel pairs in the mean.

| controller | mean delta bps | std ddof=0 | n |
|---|---:|---:|---:|
| rand_dwell_s | 0.141227 | 0.193331 | 84 |
| oracle7_s | 0.156371 | 0.252135 | 84 |
| prog_chan_s | 0.206073 | 0.312937 | 84 |

| controller | channel | n | after bps | end bps | delta bps | std ddof=0 |
|---|---|---:|---:|---:|---:|---:|
| rand_dwell_s | G | 12 | 1.216880 | 1.369471 | 0.152591 | 0.095382 |
| rand_dwell_s | C1 | 12 | 0.357706 | 0.388681 | 0.030974 | 0.045297 |
| rand_dwell_s | C7 | 12 | 0.519581 | 0.561546 | 0.041965 | 0.051049 |
| rand_dwell_s | E30 | 12 | 0.836656 | 1.159414 | 0.322758 | 0.279218 |
| rand_dwell_s | E110 | 12 | 0.885927 | 1.057057 | 0.171130 | 0.129379 |
| rand_dwell_s | M7 | 12 | 0.322256 | 0.401904 | 0.079647 | 0.141420 |
| rand_dwell_s | M37 | 12 | 0.433973 | 0.623494 | 0.189522 | 0.266591 |
| oracle7_s | G | 12 | 1.148420 | 1.254283 | 0.105863 | 0.104180 |
| oracle7_s | C1 | 12 | 0.437766 | 0.480884 | 0.043118 | 0.066177 |
| oracle7_s | C7 | 12 | 0.597449 | 0.671351 | 0.073902 | 0.061803 |
| oracle7_s | E30 | 12 | 0.819273 | 1.145612 | 0.326339 | 0.349719 |
| oracle7_s | E110 | 12 | 0.842586 | 1.049831 | 0.207245 | 0.321993 |
| oracle7_s | M7 | 12 | 0.312142 | 0.415411 | 0.103269 | 0.129973 |
| oracle7_s | M37 | 12 | 0.410935 | 0.645794 | 0.234858 | 0.347047 |
| prog_chan_s | G | 12 | 0.920157 | 1.277753 | 0.357596 | 0.304197 |
| prog_chan_s | C1 | 12 | 0.418641 | 0.531874 | 0.113233 | 0.120935 |
| prog_chan_s | C7 | 12 | 0.547752 | 0.684565 | 0.136813 | 0.114442 |
| prog_chan_s | E30 | 12 | 0.890184 | 1.217076 | 0.326892 | 0.423368 |
| prog_chan_s | E110 | 12 | 1.087369 | 1.342530 | 0.255161 | 0.484757 |
| prog_chan_s | M7 | 12 | 0.406524 | 0.517458 | 0.110933 | 0.269034 |
| prog_chan_s | M37 | 12 | 0.454787 | 0.596673 | 0.141886 | 0.109925 |

## Decision shares, L = 2048

Mean over seeds. A stay counts for the room channel. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.

| controller | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 0.0690 | 0.0665 | 0.0741 | 0.0716 | 0.0739 | 0.0622 | 0.0701 | 0.0738 | 0.0651 | 0.1744 | 0.1992 |
| oracle7_s | 0.0884 | 0.0835 | 0.0955 | 0.0911 | 0.0896 | 0.0878 | 0.0978 | 0.0000 | 0.0000 | 0.1671 | 0.1992 |
| prog_chan_s | 0.1399 | 0.0443 | 0.0729 | 0.0703 | 0.0518 | 0.0394 | 0.0618 | 0.2015 | 0.0127 | 0.1061 | 0.1992 |

## W, L = 4096

| controller | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | seed 47 | seed 48 | seed 49 | seed 50 | seed 51 | seed 52 | seed 53 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 0.389725 | 0.428852 | 0.476761 | 0.458876 | 0.496975 | 0.460069 | 0.461199 | 0.429597 | 0.519873 | 0.639998 | 0.357328 | 0.587688 | 0.475578 ± 0.075499 |
| oracle7_s | 0.515642 | 0.505272 | 0.483099 | 0.460620 | 0.449485 | 0.532669 | 0.422023 | 0.417530 | 0.416244 | 0.413774 | 0.400041 | 0.451155 | 0.455629 ± 0.042635 |
| prog_chan_s | 0.623142 | 0.715025 | 0.416408 | 0.433324 | 0.537278 | 0.457102 | 0.429211 | 0.483504 | 0.527081 | 0.433021 | 0.426319 | 0.425891 | 0.492275 ± 0.089801 |

## W9, L = 4096

| controller | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | seed 47 | seed 48 | seed 49 | seed 50 | seed 51 | seed 52 | seed 53 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 1.190477 | 1.231297 | 1.309445 | 1.322989 | 1.402427 | 1.460692 | 1.425285 | 1.281715 | 1.411393 | 1.511799 | 1.362005 | 1.434273 | 1.361983 ± 0.092518 |
| oracle7_s | 2.365489 | 2.385678 | 2.274323 | 2.260745 | 2.212792 | 2.413365 | 2.332317 | 2.170017 | 2.139220 | 2.206697 | 2.209515 | 2.230064 | 2.266685 ± 0.084923 |
| prog_chan_s | 1.231428 | 1.259170 | 1.048645 | 1.139008 | 1.189873 | 1.231735 | 1.175971 | 1.140770 | 1.082618 | 1.044171 | 1.017909 | 1.028536 | 1.132486 ± 0.082781 |

## Gaps, L = 4096

On W. A negative gap means the first controller is lower.

| seed | prog_chan_s − rand_dwell_s | oracle7_s − rand_dwell_s | prog_chan_s − oracle7_s |
|---:|---:|---:|---:|
| 42 | 0.233417 | 0.125917 | 0.107499 |
| 43 | 0.286173 | 0.076421 | 0.209752 |
| 44 | -0.060353 | 0.006338 | -0.066691 |
| 45 | -0.025552 | 0.001744 | -0.027296 |
| 46 | 0.040303 | -0.047490 | 0.087793 |
| 47 | -0.002966 | 0.072600 | -0.075566 |
| 48 | -0.031988 | -0.039176 | 0.007188 |
| 49 | 0.053907 | -0.012067 | 0.065974 |
| 50 | 0.007208 | -0.103629 | 0.110838 |
| 51 | -0.206977 | -0.226225 | 0.019247 |
| 52 | 0.068991 | 0.042712 | 0.026279 |
| 53 | -0.161798 | -0.136533 | -0.025264 |
| mean | 0.016697 | -0.019949 | 0.036646 |
| std ddof=0 | 0.134388 | 0.095110 | 0.079810 |
| left lower (count) | 6 of 12 | 6 of 12 | 4 of 12 |

## Forgetting, L = 4096

Exam-store bits/symbol just after the last visit that dwelled in the channel, minus nothing: the difference is end minus that snapshot. A channel with no exam glance at that visit is left out of the mean. n is the number of seed-channel pairs in the mean.

| controller | mean delta bps | std ddof=0 | n |
|---|---:|---:|---:|
| rand_dwell_s | 0.138138 | 0.194598 | 84 |
| oracle7_s | 0.081172 | 0.115718 | 84 |
| prog_chan_s | 0.146783 | 0.254195 | 84 |

| controller | channel | n | after bps | end bps | delta bps | std ddof=0 |
|---|---|---:|---:|---:|---:|---:|
| rand_dwell_s | G | 12 | 0.949684 | 1.080489 | 0.130805 | 0.086587 |
| rand_dwell_s | C1 | 12 | 0.333465 | 0.370840 | 0.037376 | 0.034551 |
| rand_dwell_s | C7 | 12 | 0.422967 | 0.437937 | 0.014970 | 0.026599 |
| rand_dwell_s | E30 | 12 | 0.793606 | 1.118735 | 0.325129 | 0.282101 |
| rand_dwell_s | E110 | 12 | 0.810124 | 1.158902 | 0.348778 | 0.224054 |
| rand_dwell_s | M7 | 12 | 0.303752 | 0.343103 | 0.039351 | 0.039867 |
| rand_dwell_s | M37 | 12 | 0.403192 | 0.473751 | 0.070560 | 0.072611 |
| oracle7_s | G | 12 | 1.006625 | 1.105249 | 0.098624 | 0.091875 |
| oracle7_s | C1 | 12 | 0.457554 | 0.464797 | 0.007243 | 0.035341 |
| oracle7_s | C7 | 12 | 0.533224 | 0.540852 | 0.007628 | 0.020288 |
| oracle7_s | E30 | 12 | 0.727404 | 0.885139 | 0.157735 | 0.157970 |
| oracle7_s | E110 | 12 | 0.738130 | 0.906222 | 0.168092 | 0.136486 |
| oracle7_s | M7 | 12 | 0.275519 | 0.307503 | 0.031984 | 0.046033 |
| oracle7_s | M37 | 12 | 0.402366 | 0.499263 | 0.096897 | 0.103290 |
| prog_chan_s | G | 12 | 0.834703 | 0.929528 | 0.094824 | 0.115560 |
| prog_chan_s | C1 | 12 | 0.343531 | 0.432351 | 0.088820 | 0.126582 |
| prog_chan_s | C7 | 12 | 0.405355 | 0.515725 | 0.110370 | 0.100199 |
| prog_chan_s | E30 | 12 | 0.807263 | 1.135840 | 0.328577 | 0.430914 |
| prog_chan_s | E110 | 12 | 0.963277 | 1.244738 | 0.281461 | 0.376793 |
| prog_chan_s | M7 | 12 | 0.309592 | 0.361038 | 0.051447 | 0.067660 |
| prog_chan_s | M37 | 12 | 0.438548 | 0.510527 | 0.071978 | 0.085993 |

## Decision shares, L = 4096

Mean over seeds. A stay counts for the room channel. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.

| controller | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 0.0693 | 0.0683 | 0.0765 | 0.0729 | 0.0704 | 0.0635 | 0.0716 | 0.0703 | 0.0638 | 0.1741 | 0.1992 |
| oracle7_s | 0.0906 | 0.0868 | 0.0948 | 0.0899 | 0.0889 | 0.0913 | 0.0907 | 0.0000 | 0.0000 | 0.1679 | 0.1992 |
| prog_chan_s | 0.1543 | 0.0480 | 0.0706 | 0.0804 | 0.0519 | 0.0350 | 0.0576 | 0.2017 | 0.0098 | 0.0915 | 0.1992 |

## Walls

| kernel | seed | lives | wall seconds |
|---|---:|---|---:|
| ezzzzzx/m7c-w3-s42 | 42 | 2048, 4096 | 613.226 |
| ezzzzzx/m7c-w3-s43 | 43 | 2048, 4096 | 547.269 |
| ezzzzzx/m7c-w3-s44 | 44 | 2048, 4096 | 559.066 |
| ezzzzzx/m7c-w3-s45 | 45 | 2048, 4096 | 716.484 |
| ezzzzzx/m7c-w3-s46 | 46 | 2048, 4096 | 554.270 |
| ezzzzzx/m7c-w3-s47 | 47 | 2048, 4096 | 635.177 |
| ezzzzzx/m7c-w3-s48 | 48 | 2048, 4096 | 620.347 |
| ezzzzzx/m7c-w3-s49 | 49 | 2048, 4096 | 408.656 |
| ezzzzzx/m7c-w3-s50 | 50 | 2048, 4096 | 613.441 |
| ezzzzzx/m7c-w3-s51 | 51 | 2048, 4096 | 609.144 |
| ezzzzzx/m7c-w3-s52 | 52 | 2048, 4096 | 333.831 |
| ezzzzzx/m7c-w3-s53 | 53 | 2048, 4096 | 625.304 |

| controller | L | seed | wall seconds | learn seconds | score seconds |
|---|---:|---:|---:|---:|---:|
| rand_dwell_s | 2048 | 42 | 145.823 | 81.755 | 62.897 |
| oracle7_s | 2048 | 42 | 168.611 | 103.098 | 63.775 |
| prog_chan_s | 2048 | 42 | 342.315 | 118.214 | 211.678 |
| rand_dwell_s | 4096 | 42 | 424.367 | 216.740 | 194.982 |
| oracle7_s | 4096 | 42 | 425.069 | 236.787 | 175.497 |
| prog_chan_s | 4096 | 42 | 609.889 | 213.260 | 383.780 |
| rand_dwell_s | 2048 | 43 | 129.079 | 69.209 | 58.779 |
| oracle7_s | 2048 | 43 | 142.631 | 85.425 | 56.087 |
| prog_chan_s | 2048 | 43 | 293.094 | 97.158 | 185.886 |
| rand_dwell_s | 4096 | 43 | 366.842 | 181.660 | 174.984 |
| oracle7_s | 4096 | 43 | 373.600 | 203.444 | 159.741 |
| prog_chan_s | 4096 | 43 | 544.663 | 184.645 | 349.618 |
| rand_dwell_s | 2048 | 44 | 136.212 | 75.463 | 59.640 |
| oracle7_s | 2048 | 44 | 149.977 | 90.160 | 58.236 |
| prog_chan_s | 2048 | 44 | 307.684 | 105.891 | 190.316 |
| rand_dwell_s | 4096 | 44 | 384.094 | 192.454 | 179.908 |
| oracle7_s | 4096 | 44 | 388.198 | 215.695 | 160.526 |
| prog_chan_s | 4096 | 44 | 556.007 | 186.750 | 357.502 |
| rand_dwell_s | 2048 | 45 | 167.546 | 86.133 | 79.997 |
| oracle7_s | 2048 | 45 | 194.360 | 110.216 | 82.037 |
| prog_chan_s | 2048 | 45 | 381.910 | 120.289 | 248.344 |
| rand_dwell_s | 4096 | 45 | 483.504 | 231.083 | 238.892 |
| oracle7_s | 4096 | 45 | 461.399 | 234.793 | 212.206 |
| prog_chan_s | 4096 | 45 | 712.925 | 222.990 | 476.008 |
| rand_dwell_s | 2048 | 46 | 127.830 | 72.623 | 54.170 |
| oracle7_s | 2048 | 46 | 147.766 | 88.791 | 57.747 |
| prog_chan_s | 2048 | 46 | 292.087 | 97.121 | 183.625 |
| rand_dwell_s | 4096 | 46 | 377.788 | 193.252 | 172.869 |
| oracle7_s | 4096 | 46 | 365.351 | 202.240 | 151.317 |
| prog_chan_s | 4096 | 46 | 551.345 | 185.737 | 353.889 |
| rand_dwell_s | 2048 | 47 | 145.525 | 79.495 | 64.842 |
| oracle7_s | 2048 | 47 | 168.570 | 101.581 | 65.666 |
| prog_chan_s | 2048 | 47 | 341.677 | 116.117 | 214.676 |
| rand_dwell_s | 4096 | 47 | 433.058 | 223.016 | 198.891 |
| oracle7_s | 4096 | 47 | 422.792 | 232.938 | 178.771 |
| prog_chan_s | 4096 | 47 | 632.223 | 221.572 | 399.492 |
| rand_dwell_s | 2048 | 48 | 137.328 | 78.127 | 58.111 |
| oracle7_s | 2048 | 48 | 157.665 | 97.681 | 58.735 |
| prog_chan_s | 2048 | 48 | 333.893 | 111.706 | 209.020 |
| rand_dwell_s | 4096 | 48 | 419.156 | 215.468 | 190.450 |
| oracle7_s | 4096 | 48 | 416.429 | 230.169 | 172.668 |
| prog_chan_s | 4096 | 48 | 617.007 | 208.622 | 394.904 |
| rand_dwell_s | 2048 | 49 | 97.870 | 43.794 | 53.165 |
| oracle7_s | 2048 | 49 | 95.738 | 47.253 | 47.094 |
| prog_chan_s | 2048 | 49 | 227.377 | 58.710 | 159.058 |
| rand_dwell_s | 4096 | 49 | 270.206 | 107.472 | 152.829 |
| oracle7_s | 4096 | 49 | 262.598 | 116.583 | 136.072 |
| prog_chan_s | 4096 | 49 | 406.188 | 98.802 | 297.284 |
| rand_dwell_s | 2048 | 50 | 147.157 | 82.196 | 63.790 |
| oracle7_s | 2048 | 50 | 171.377 | 94.988 | 74.526 |
| prog_chan_s | 2048 | 50 | 328.984 | 115.827 | 201.412 |
| rand_dwell_s | 4096 | 50 | 423.791 | 213.435 | 198.291 |
| oracle7_s | 4096 | 50 | 413.092 | 220.435 | 180.483 |
| prog_chan_s | 4096 | 50 | 610.109 | 215.740 | 382.319 |
| rand_dwell_s | 2048 | 51 | 139.107 | 74.911 | 63.084 |
| oracle7_s | 2048 | 51 | 150.052 | 86.725 | 61.595 |
| prog_chan_s | 2048 | 51 | 332.667 | 108.764 | 213.469 |
| rand_dwell_s | 4096 | 51 | 410.461 | 206.851 | 192.907 |
| oracle7_s | 4096 | 51 | 399.181 | 216.255 | 172.095 |
| prog_chan_s | 4096 | 51 | 606.275 | 202.012 | 393.571 |
| rand_dwell_s | 2048 | 52 | 82.608 | 35.846 | 45.958 |
| oracle7_s | 2048 | 52 | 87.221 | 41.006 | 45.328 |
| prog_chan_s | 2048 | 52 | 186.612 | 47.888 | 131.030 |
| rand_dwell_s | 4096 | 52 | 222.678 | 88.540 | 126.242 |
| oracle7_s | 4096 | 52 | 215.688 | 96.295 | 111.392 |
| prog_chan_s | 4096 | 52 | 331.797 | 79.685 | 244.126 |
| rand_dwell_s | 2048 | 53 | 136.811 | 73.442 | 62.284 |
| oracle7_s | 2048 | 53 | 162.796 | 90.939 | 70.116 |
| prog_chan_s | 2048 | 53 | 336.064 | 102.160 | 221.493 |
| rand_dwell_s | 4096 | 53 | 422.190 | 199.755 | 209.570 |
| oracle7_s | 4096 | 53 | 414.546 | 215.700 | 185.834 |
| prog_chan_s | 4096 | 53 | 621.997 | 194.758 | 414.456 |

## Smoke test

```
torch=2.15.0+cpu threads=1 model_seed=0
dose L64 16/64 share 0.250000
dose L512 104/512 share 0.203125
dose L2048 408/2048 share 0.199219
dose L4096 816/4096 share 0.199219
smoke check 1: rand_dwell_s seed 36 L=64 versus M7b's W
  published line: rand_dwell_s W 13.654479 W9 12.786797 finite True sleep 0.2500 decisions 16 corridor 0.1875 wall 5.47s learn 2.86s score 1.34s
  source results/m7bw3/smoke.json ref W 13.654479 live W 13.6544791891 W9 12.7867969106 abs_dW 1.891e-07 limit 1.0e-06 ok True
smoke check 2: rand_dwell_s, oracle7_s, prog_chan_s, seed 42, L=64, finite W
  rand_dwell_s W 12.992897 W9 12.162775 finite True sleep 0.2500 decisions 16 corridor 0.2188 wall 5.47s
  oracle7_s W 11.805389 W9 11.178754 finite True sleep 0.2500 decisions 16 corridor 0.1875 wall 5.54s
  prog_chan_s W 8.522759 W9 8.688424 finite True sleep 0.2500 decisions 16 corridor 0.0781 wall 7.48s
smoke check 3: time prog_chan_s seed 42 L=512 and estimate one kernel
  prog_chan_s seed 42 L=512 W 2.193410 W9 2.668839 wall 80.45s learn 30.81s score 47.87s eval 1.41s sleep 104
  learn_per_decision 0.0602s full_rescore_288 0.2502s channel_rescore_32 0.0497s scale 1.000
  model kernel 803.9s linear kernel 965.4s kernel 803.9s (13.40 min) slow-seed 1365.5s (22.76 min) limit 5400s split False
  one-life kernel L=2048 402.7s
  one-life kernel L=4096 803.9s
  estimate is under 90 min; one kernel per seed runs both lives
```

## Protocol

- `BankCortex`, the stores, the sleep decision, W, W9 and forgetting are `m7_w3`'s, reached through `m7b_w3`. Model seed 0. Glance length 64. Reservoirs hold 32 glances per channel. Exam coin `default_rng(40_000 + seed)`, exam replacement `default_rng(20_000 + seed)`, replay replacement `default_rng(50_000 + seed)`, sleep draw `default_rng(30_000 + seed)`.
- `rand_dwell_s` and `oracle7_s` are `m7b_w3.run_fixed_sleep` with `w3_headroom.next_action` and `default_rng(1_000 + seed)`. A sleep visit starts only when the current plan is empty, which is the end of a visit.
- `prog_chan_s` uses the 9-room bandit (`default_rng(10_000 + seed)`, optimistic first pass, EPS 0.1, BETA 0.3). Its room reward sums bits saved on the destination room's exam glances over the visit, including corridor steps, and divides by the visit's decisions. Sleep does not call the EMA.
- The dose is 8 times the integer nearest to 0.20 L / 8. Placement is `m7_w3.sleep_plan` and `m7_w3.sleep_due`. World seeds 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53. L = 2048 and L = 4096. CPU. Private Kaggle kernels, one seed each, pool of 4, no GPU, no internet. Nothing retuned.

## Verdict (orchestrator, 13 Nov 2026)

**Precondition (oracle7_s − rand_dwell_s ≤ −0.15 at L = 4,096): NOT MET.** The gap is −0.020 (6 of 12 seeds). The bars are void: **no headroom**.

**What it means.** With the bank cortex and a fixed sleep dose, W3 has almost nothing left to allocate. Every
controller learns the seven scored rooms to about the same level: W 0.46–0.49 at L = 4,096, against 1.76–1.89 for the
hashed learner at L = 2,048. The oracle's 0.10 in M7b was noise from 6 seeds; with 12 seeds it is 0.03 at L = 2,048 and 0.02 at
L = 4,096. prog_chan_s is level with random dwelling (+0.007 and +0.017, 6 of 12 each). This is W2-1's lesson at a
higher level: headroom belongs to a world *and a learner* together. W3 had room for curiosity with a weak cortex (M5b),
and a strong local cortex with sleep fills it. W9 favours prog_chan_s only because it visits the dark room, which the
oracle skips by fiat.

**Line status.** Settled on W3: curiosity pays with a weak learner (M5b), and sleep pays on a cortex that forgets (M7,
M7b, M7c, where every controller sleeps). Open: curiosity with the strong cortex needs a world sized to it, with more
sources, harder learnable ones and a life that is scarce for the bank. Headroom is to be checked on the bank and sleep
before any controller runs (W4).
