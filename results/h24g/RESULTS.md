# H24g measurements

Started: 2026-11-10 12:13:32 UTC+8. Updated: 2026-11-10 13:27:35 UTC+8.

Slow world; N = 131072; seeds = [6, 7, 8]; alphabet = 42; CPU threads = 2.
Budget B/N = 0.883848 (0.15 × full5 peak M/N). Bar reference M/N ≤ 0.884 is not scored here.

| Condition | Seed | P | Peak M/N | Final M/N | Fraction stored | Stored symbols | Final windows | Evicted windows | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | 6 | 1.339368 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | NA | 193.633 |
| gru | 7 | 2.093612 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | NA | 190.770 |
| gru | 8 | 1.760025 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | NA | 183.070 |
| full | 6 | 0.622545 | 5.892317 | 5.892317 | 1.000000 | 131072 | 2048 | NA | 684.836 |
| full | 7 | 1.272562 | 5.892317 | 5.892317 | 1.000000 | 131072 | 2048 | NA | 674.432 |
| full | 8 | 1.079035 | 5.892317 | 5.892317 | 1.000000 | 131072 | 2048 | NA | 659.480 |
| full5 | 6 | 0.620883 | 5.892317 | 5.892317 | 1.000000 | 131072 | 2048 | NA | 714.804 |
| full5 | 7 | 1.226276 | 5.892317 | 5.892317 | 1.000000 | 131072 | 2048 | NA | 702.664 |
| full5 | 8 | 1.075444 | 5.892317 | 5.892317 | 1.000000 | 131072 | 2048 | NA | 696.916 |
| fine | 6 | 0.663810 | 1.723501 | 1.723501 | 0.270950 | 35514 | 1075 | NA | 707.731 |
| fine | 7 | 1.313592 | 2.686861 | 2.686861 | 0.451370 | 59162 | 1036 | NA | 680.204 |
| fine | 8 | 1.102926 | 2.830140 | 2.830140 | 0.485321 | 63612 | 873 | NA | 671.410 |
| fine3 | 6 | 0.754674 | 2.383891 | 2.383891 | 0.381828 | 50047 | 1331 | NA | 671.812 |
| fine3 | 7 | 1.376289 | 3.409629 | 3.409629 | 0.586403 | 76861 | 1014 | NA | 662.185 |
| fine3 | 8 | 1.130408 | 3.357808 | 3.357808 | 0.572220 | 75002 | 1115 | NA | 685.580 |
| util | 6 | 0.663219 | 0.883822 | 0.883122 | 0.145981 | 19134 | 393 | 745 | 744.928 |
| util | 7 | 1.319355 | 0.883847 | 0.882993 | 0.161034 | 21107 | 60 | 923 | 744.471 |
| util | 8 | 1.040163 | 0.883845 | 0.883657 | 0.161247 | 21135 | 58 | 818 | 768.372 |
| util_m | 6 | 0.660552 | 0.883847 | 0.883070 | 0.143707 | 18836 | 443 | 583 | 691.787 |
| util_m | 7 | 1.293094 | 0.883846 | 0.883062 | 0.160141 | 20990 | 80 | 894 | 692.731 |
| util_m | 8 | 1.039472 | 0.883845 | 0.883297 | 0.161362 | 21150 | 54 | 884 | 680.230 |

| Condition | Seeds | P mean ± population std | Peak M/N mean ± std | Final M/N mean ± std | Fraction stored mean ± std |
|---|---:|---:|---:|---:|---:|
| gru | 3 | 1.731002 ± 0.308602 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| full | 3 | 0.991381 ± 0.272510 | 5.892317 ± 0.000000 | 5.892317 ± 0.000000 | 1.000000 ± 0.000000 |
| full5 | 3 | 0.974201 ± 0.257310 | 5.892317 ± 0.000000 | 5.892317 ± 0.000000 | 1.000000 ± 0.000000 |
| fine | 3 | 1.026776 ± 0.270682 | 2.413501 ± 0.491397 | 2.413501 ± 0.491397 | 0.402547 ± 0.094080 |
| fine3 | 3 | 1.087124 ± 0.255612 | 3.050443 ± 0.471798 | 3.050443 ± 0.471798 | 0.513484 ± 0.093274 |
| util | 3 | 1.007579 ± 0.268855 | 0.883838 ± 0.000012 | 0.883257 ± 0.000287 | 0.156087 ± 0.007147 |
| util_m | 3 | 0.997706 ± 0.259917 | 0.883846 ± 0.000001 | 0.883143 ± 0.000109 | 0.155070 ± 0.008050 |

| Condition | Seed | G | C1 | C7 | E30 | E110 | M7 | M37 | F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | 6 | NA | 1.570243 | 2.405219 | 1.238582 | NA | 0.654884 | 0.466952 | NA |
| gru | 7 | 1.751203 | 3.395918 | 3.144010 | NA | 1.379465 | 0.585471 | NA | 3.733900 |
| gru | 8 | NA | 0.852376 | 2.023701 | NA | 1.173049 | 0.445717 | NA | 3.194302 |
| full | 6 | NA | 0.366176 | 0.733541 | 1.243366 | NA | 0.345031 | 0.316530 | NA |
| full | 7 | 0.930272 | 1.416133 | 1.002985 | NA | 1.043214 | 0.353308 | NA | 3.348156 |
| full | 8 | NA | 0.674508 | 0.841705 | NA | 1.195194 | 0.331688 | NA | 1.776144 |
| full5 | 6 | NA | 0.366252 | 0.733517 | 1.241260 | NA | 0.337491 | 0.314780 | NA |
| full5 | 7 | 0.897395 | 1.416126 | 1.002803 | NA | 1.043406 | 0.350726 | NA | 3.012939 |
| full5 | 8 | NA | 0.671866 | 0.841867 | NA | 1.195484 | 0.329347 | NA | 1.763821 |
| fine | 6 | NA | 0.451478 | 0.836882 | 1.212071 | NA | 0.335805 | 0.359217 | NA |
| fine | 7 | 1.097534 | 1.762654 | 1.002268 | NA | 1.072507 | 0.370995 | NA | 3.057764 |
| fine | 8 | NA | 0.578198 | 0.862788 | NA | 1.159394 | 0.320902 | NA | 1.939971 |
| fine3 | 6 | NA | 0.644197 | 0.995423 | 1.207077 | NA | 0.335742 | 0.359485 | NA |
| fine3 | 7 | 1.178459 | 1.729402 | 1.537538 | NA | 1.064132 | 0.346664 | NA | 3.025853 |
| fine3 | 8 | NA | 0.595474 | 0.830572 | NA | 1.157559 | 0.319889 | NA | 2.075821 |
| util | 6 | NA | 0.437532 | 0.830664 | 1.232802 | NA | 0.337673 | 0.359217 | NA |
| util | 7 | 1.097534 | 1.802115 | 0.997669 | NA | 1.084520 | 0.353271 | NA | 3.050691 |
| util | 8 | NA | 0.467948 | 0.900464 | NA | 1.127160 | 0.333199 | NA | 1.732455 |
| util_m | 6 | NA | 0.450488 | 0.835587 | 1.200766 | NA | 0.336619 | 0.359217 | NA |
| util_m | 7 | 1.097534 | 1.710552 | 0.877949 | NA | 1.085447 | 0.351769 | NA | 3.050610 |
| util_m | 8 | NA | 0.476804 | 0.871831 | NA | 1.157682 | 0.338043 | NA | 1.720950 |

| Condition | Seed | Prior key appearances | Value digits | Bits | Bits/digit |
|---|---:|---|---:|---:|---:|
| gru | 6 | 0 | 0 | 0.000000 | NA |
| gru | 6 | 1 | 0 | 0.000000 | NA |
| gru | 6 | 2–4 | 0 | 0.000000 | NA |
| gru | 6 | 5+ | 0 | 0.000000 | NA |
| gru | 7 | 0 | 1188 | 5358.192638 | 4.510263 |
| gru | 7 | 1 | 1184 | 4406.571892 | 3.721767 |
| gru | 7 | 2–4 | 2744 | 9502.616450 | 3.463053 |
| gru | 7 | 5+ | 1436 | 4665.245333 | 3.248778 |
| gru | 8 | 0 | 1200 | 4625.701945 | 3.854752 |
| gru | 8 | 1 | 1200 | 4226.804125 | 3.522337 |
| gru | 8 | 2–4 | 3568 | 12267.201016 | 3.438117 |
| gru | 8 | 5+ | 7139 | 20974.631387 | 2.938035 |
| full | 6 | 0 | 0 | 0.000000 | NA |
| full | 6 | 1 | 0 | 0.000000 | NA |
| full | 6 | 2–4 | 0 | 0.000000 | NA |
| full | 6 | 5+ | 0 | 0.000000 | NA |
| full | 7 | 0 | 1188 | 5759.676865 | 4.848213 |
| full | 7 | 1 | 1184 | 4347.093211 | 3.671531 |
| full | 7 | 2–4 | 2744 | 8083.955914 | 2.946048 |
| full | 7 | 5+ | 1436 | 2605.540787 | 1.814443 |
| full | 8 | 0 | 1200 | 4941.403096 | 4.117836 |
| full | 8 | 1 | 1200 | 3005.172778 | 2.504311 |
| full | 8 | 2–4 | 3568 | 2852.770056 | 0.799543 |
| full | 8 | 5+ | 7139 | 1790.364661 | 0.250786 |
| full5 | 6 | 0 | 0 | 0.000000 | NA |
| full5 | 6 | 1 | 0 | 0.000000 | NA |
| full5 | 6 | 2–4 | 0 | 0.000000 | NA |
| full5 | 6 | 5+ | 0 | 0.000000 | NA |
| full5 | 7 | 0 | 1188 | 5886.563917 | 4.955020 |
| full5 | 7 | 1 | 1184 | 1723.478685 | 1.455641 |
| full5 | 7 | 2–4 | 2744 | 3251.164212 | 1.184827 |
| full5 | 7 | 5+ | 1436 | 1825.991016 | 1.271581 |
| full5 | 8 | 0 | 1200 | 4995.992299 | 4.163327 |
| full5 | 8 | 1 | 1200 | 1996.476825 | 1.663731 |
| full5 | 8 | 2–4 | 3568 | 2443.862264 | 0.684939 |
| full5 | 8 | 5+ | 7139 | 1739.061272 | 0.243600 |
| fine | 6 | 0 | 0 | 0.000000 | NA |
| fine | 6 | 1 | 0 | 0.000000 | NA |
| fine | 6 | 2–4 | 0 | 0.000000 | NA |
| fine | 6 | 5+ | 0 | 0.000000 | NA |
| fine | 7 | 0 | 1188 | 6118.462945 | 5.150221 |
| fine | 7 | 1 | 1184 | 983.672419 | 0.830804 |
| fine | 7 | 2–4 | 2744 | 1976.172441 | 0.720179 |
| fine | 7 | 5+ | 1436 | 1132.944483 | 0.788959 |
| fine | 8 | 0 | 1200 | 4973.291063 | 4.144409 |
| fine | 8 | 1 | 1200 | 1760.657106 | 1.467214 |
| fine | 8 | 2–4 | 3568 | 2836.279521 | 0.794921 |
| fine | 8 | 5+ | 7139 | 4138.735622 | 0.579736 |
| fine3 | 6 | 0 | 0 | 0.000000 | NA |
| fine3 | 6 | 1 | 0 | 0.000000 | NA |
| fine3 | 6 | 2–4 | 0 | 0.000000 | NA |
| fine3 | 6 | 5+ | 0 | 0.000000 | NA |
| fine3 | 7 | 0 | 1188 | 5640.754499 | 4.748110 |
| fine3 | 7 | 1 | 1184 | 1175.259219 | 0.992618 |
| fine3 | 7 | 2–4 | 2744 | 2247.762095 | 0.819155 |
| fine3 | 7 | 5+ | 1436 | 1221.884075 | 0.850894 |
| fine3 | 8 | 0 | 1200 | 4955.762964 | 4.129802 |
| fine3 | 8 | 1 | 1200 | 1729.857516 | 1.441548 |
| fine3 | 8 | 2–4 | 3568 | 2238.225346 | 0.627305 |
| fine3 | 8 | 5+ | 7139 | 4741.206926 | 0.664128 |
| util | 6 | 0 | 0 | 0.000000 | NA |
| util | 6 | 1 | 0 | 0.000000 | NA |
| util | 6 | 2–4 | 0 | 0.000000 | NA |
| util | 6 | 5+ | 0 | 0.000000 | NA |
| util | 7 | 0 | 1188 | 6012.730964 | 5.061221 |
| util | 7 | 1 | 1184 | 976.684775 | 0.824903 |
| util | 7 | 2–4 | 2744 | 2128.617119 | 0.775735 |
| util | 7 | 5+ | 1436 | 1236.294474 | 0.860929 |
| util | 8 | 0 | 1200 | 4977.497473 | 4.147915 |
| util | 8 | 1 | 1200 | 1785.390747 | 1.487826 |
| util | 8 | 2–4 | 3568 | 2466.341308 | 0.691239 |
| util | 8 | 5+ | 7139 | 1108.706434 | 0.155303 |
| util_m | 6 | 0 | 0 | 0.000000 | NA |
| util_m | 6 | 1 | 0 | 0.000000 | NA |
| util_m | 6 | 2–4 | 0 | 0.000000 | NA |
| util_m | 6 | 5+ | 0 | 0.000000 | NA |
| util_m | 7 | 0 | 1188 | 6053.307405 | 5.095377 |
| util_m | 7 | 1 | 1184 | 1000.206370 | 0.844769 |
| util_m | 7 | 2–4 | 2744 | 2149.835310 | 0.783468 |
| util_m | 7 | 5+ | 1436 | 1313.975925 | 0.915025 |
| util_m | 8 | 0 | 1200 | 4977.300780 | 4.147751 |
| util_m | 8 | 1 | 1200 | 1773.877345 | 1.478231 |
| util_m | 8 | 2–4 | 3568 | 2414.012728 | 0.676573 |
| util_m | 8 | 5+ | 7139 | 1155.472461 | 0.161854 |

| Condition | Seed | F value bits/digit, ≥1 prior |
|---|---:|---:|
| gru | 6 | NA |
| gru | 7 | 3.462795 |
| gru | 8 | 3.146774 |
| full | 6 | NA |
| full | 7 | 2.803242 |
| full | 8 | 0.642337 |
| full5 | 6 | NA |
| full5 | 7 | 1.267829 |
| full5 | 8 | 0.518972 |
| fine | 6 | NA |
| fine | 7 | 0.763011 |
| fine | 8 | 0.733659 |
| fine3 | 6 | NA |
| fine3 | 7 | 0.865941 |
| fine3 | 8 | 0.731443 |
| util | 6 | NA |
| util | 7 | 0.809395 |
| util | 8 | 0.450192 |
| util_m | 6 | NA |
| util_m | 7 | 0.832218 |
| util_m | 8 | 0.448758 |

| Condition | F ≥1 prior mean over observed seeds | F ≥1 prior pooled bits/digit |
|---|---:|---:|
| gru | 3.304785 | 3.244923 |
| full | 1.722790 | 1.313468 |
| full5 | 0.893400 | 0.751551 |
| fine | 0.748335 | 0.742775 |
| fine3 | 0.798692 | 0.773215 |
| util | 0.629794 | 0.561753 |
| util_m | 0.640488 | 0.567852 |

| Condition | Seed | Evicted windows | G | C1 | C7 | E30 | E110 | M7 | M37 | F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| util | 6 | 745 | 0.000000 | 0.081690 | 0.147148 | 0.529871 | 0.000000 | 0.241291 | 0.000000 | 0.000000 |
| util | 7 | 923 | 0.235921 | 0.126649 | 0.119498 | 0.000000 | 0.331584 | 0.186349 | 0.000000 | 0.000000 |
| util | 8 | 818 | 0.000000 | 0.128699 | 0.277860 | 0.000000 | 0.335848 | 0.241717 | 0.000000 | 0.015876 |
| util_m | 6 | 583 | 0.000000 | 0.173885 | 0.198328 | 0.283019 | 0.000000 | 0.000000 | 0.344768 | 0.000000 |
| util_m | 7 | 894 | 0.243574 | 0.117334 | 0.140037 | 0.000000 | 0.341337 | 0.157718 | 0.000000 | 0.000000 |
| util_m | 8 | 884 | 0.000000 | 0.117989 | 0.225550 | 0.000000 | 0.413686 | 0.223571 | 0.000000 | 0.019205 |

| Condition | Seed | Match credit bits | Replay credit bits | Measure seconds | Extra wall fraction | Measure max abs bits | Final utility min | median | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| util | 6 | 1994.866520 | 6373.618980 | 83.270 | 0.125850 | 0.000000 | 0.987183 | 4.916868 | 244.493056 |
| util | 7 | 22991.449353 | 8095.334892 | 83.001 | 0.125480 | 0.000000 | 1.000000 | 3.498818 | 7568.850175 |
| util | 8 | 2899.353560 | 5130.961223 | 84.672 | 0.123844 | 0.000000 | 1.008569 | 3.687820 | 416.479985 |
| util_m | 6 | 1952.729163 | 0.000000 | 0.000 | 0.000000 | 0.000000 | 0.461133 | 0.668140 | 251.552666 |
| util_m | 7 | 23206.037174 | 0.000000 | 0.000 | 0.000000 | 0.000000 | 0.972375 | 2.443973 | 7485.872545 |
| util_m | 8 | 3134.025667 | 0.000000 | 0.000 | 0.000000 | 0.000000 | 0.980153 | 1.181401 | 296.430794 |

Pareto: P against peak M/N. gru, full, full5, fine and fine3 are carried from H24f (no rerun). gru has M = 0. full is H24e via H24f.

| Condition | Seed | P | Peak M/N |
|---|---:|---:|---:|
| gru | 6 | 1.339368 | 0.000000 |
| gru | 7 | 2.093612 | 0.000000 |
| gru | 8 | 1.760025 | 0.000000 |
| full | 6 | 0.622545 | 5.892317 |
| full | 7 | 1.272562 | 5.892317 |
| full | 8 | 1.079035 | 5.892317 |
| full5 | 6 | 0.620883 | 5.892317 |
| full5 | 7 | 1.226276 | 5.892317 |
| full5 | 8 | 1.075444 | 5.892317 |
| fine | 6 | 0.663810 | 1.723501 |
| fine | 7 | 1.313592 | 2.686861 |
| fine | 8 | 1.102926 | 2.830140 |
| fine3 | 6 | 0.754674 | 2.383891 |
| fine3 | 7 | 1.376289 | 3.409629 |
| fine3 | 8 | 1.130408 | 3.357808 |
| util | 6 | 0.663219 | 0.883822 |
| util | 7 | 1.319355 | 0.883847 |
| util | 8 | 1.040163 | 0.883845 |
| util_m | 6 | 0.660552 | 0.883847 |
| util_m | 7 | 1.293094 | 0.883846 |
| util_m | 8 | 1.039472 | 0.883845 |

| Condition | Seeds | P mean ± population std | Peak M/N mean ± population std |
|---|---:|---:|---:|
| gru | 3 | 1.731002 ± 0.308602 | 0.000000 ± 0.000000 |
| full | 3 | 0.991381 ± 0.272510 | 5.892317 ± 0.000000 |
| full5 | 3 | 0.974201 ± 0.257310 | 5.892317 ± 0.000000 |
| fine | 3 | 1.026776 ± 0.270682 | 2.413501 ± 0.491397 |
| fine3 | 3 | 1.087124 ± 0.255612 | 3.050443 ± 0.471798 |
| util | 3 | 1.007579 ± 0.268855 | 0.883838 ± 0.000012 |
| util_m | 3 | 0.997706 ± 0.259917 | 0.883846 ± 0.000001 |

P: prequential observation bits/symbol under the fixed-share mixture of the GRU and the order-5 match. M = stored symbols × log2(42) + 32 bits/window. Peak M is the maximum memory retained after utility eviction; a write that crosses B is cut back by evicting whole unprotected windows before the peak is updated. Final M is memory at the last symbol. Denominator is final N.
gru, full, full5, fine and fine3 rows are copied from H24f, including their historical wall seconds. No reruns. full5 keeps H24e aligned 64-symbol windows. fine keeps H24f surprise windows.
Match order 5, latest stored target; backwards extension up to 32 inside the stored window. Confidence buckets: 5–7, 8–11, 12–15, 16–23, 24–32. Fixed share 0.001, initial weights (0.5, 0.5).
util and util_m write like fine: mixture bits strictly > 4.0, inclusive window [i−16, i+4] clipped to the life, overlapping windows merged, touching windows kept separate. The unscored right tail is admitted after the next chunk is scored and joined to its own head (join_touching). That tail adds no 1-bit prior and does not restart protection.
A newly written window (including one merged with older windows) starts at utility 1 bit plus the utilities of the windows it absorbed, and is protected for 32 chunks: the birth chunk and the next 31. Chunk indices are 0-based. At the start of each later chunk, every already stored window is multiplied by 0.999. Credits earned during a chunk are decayed on later chunks only.
Match credit: when the order-5 match hits a stored window, add max(0, GRU-only bits of that symbol − mixture bits of that symbol) to that window. GRU-only bits are the live cross-entropy. Mixture bits are the prequential fixed-share bits, using mixture weights from before the per-symbol weight update. A miss credits nothing.
Replay credit (util only; util_m is match credit only): after the live Adam step, two batch-8 replay steps follow H24f. When a next 64-symbol chunk exists, each replay step is charged max(0, GRU bits of that next chunk before the step − GRU bits after the step) / (number of segments in the batch). GRU bits are the sum of cross-entropy bits from the detached live hidden state, with the same input embedding as live scoring, under torch.no_grad. The hidden state is not updated by replay or by the measurement. The after-measurement of step 1 is the before-measurement of step 2. A segment is one contiguous draw from one window while filling the 8×64 batch (short windows are concatenated; a window drawn twice is two segments). Each segment adds that share to its window. The last chunk has no next chunk, so its replay steps add no replay credit. Replay still trains the GRU whenever the store is non-empty.
Eviction, after every write: while M > B, drop the unprotected window with the lowest utility, breaking ties by earliest start. The order-5 index is rebuilt from the surviving windows at the end of the chunk, so an older stored copy of a context becomes the latest target after a newer copy is evicted. Evicted-window fractions use the true process ids of symbols inside each evicted window, weighted by symbol count, for analysis only. The learner never reads process ids.
Live GRU and replay schedule: H24e Cortex; two batch-8 Adam replay steps after each scored chunk when the store is non-empty, live update then replay. Scoring and match credit happen before any current-chunk admission or weight update. Neural parameters, the optimizer, the match index and the transient live context are outside M.
F exposure counts use all observations. NA denotes absent observations. Population standard deviation uses ddof=0.
New-run wall seconds (sum of util and util_m wall times; carried rows excluded): 4322.519.

Runtime note: Replay credit (b) kept. Extra wall fraction was 0.117 on the N=8192 smoke and 0.126, 0.125, 0.124 on production util seeds 6, 7, 8, all below 0.30. The next-chunk GRU-bit measurement matched that chunk's later live GRU bits within 1e-12 bits on all 2047 measured chunks of each util seed. Retained memory stayed at or under B on every chunk (chunks_over_budget = 0). Sequential, one process, Torch threads 2.

## Verdict (orchestrator, 10 Nov 2026 UTC+8): PASS, M3 in the symbol world

1. **M3 (Pareto): PASS.** util_m P 0.998 ≤ 1.074 with peak M/N 0.8838 ≤ 0.884 on all 3 seeds: within 0.024 bits of the
   full store's P (0.974) at 15% of its memory. util also passes (P 1.008).
2. **PASS:** F value digits at ≥ 1 prior appearance 0.640 (util_m), better than the full store's 0.893.
3. Evictions: mostly CA rows (E30 / E110), then M7 and counters, and grammar on seed 7; F windows are almost never
   evicted (1.6–1.9%).

Mechanism: utility (bits a window actually saved) is compression progress applied to memory, and it beats surprise as
a retention rule (memory 0.88 vs 2.41, P 1.00 vs 1.03). Evictions sort themselves out. Irreducible content (CA noise)
gives memory nothing; content the cortex has learnt (motifs, counters) makes its memories redundant, which is
consolidation; and arbitrary facts are what memory is for, so they stay. Match credit alone suffices (util_m ≥ util).
Code check: memory stayed ≤ B on every chunk; replay-credit bit checks matched the live GRU within 1e-12.
