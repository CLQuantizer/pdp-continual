# M6b-W3 sleep

Sleep is an action on the W3 ring. Replay and exam are disjoint reservoirs of 2,048 pairs per channel. Progress is scored on the exam stores. W is bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. W9 is that mean over all 9 room channels. Lower is better. Numbers only.

## W

### L = 512

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 2.736655 | 2.800518 | 2.752136 | 2.710016 | 2.745887 | 2.786315 | 2.755254 ± 0.030286 |
| prog_total | 3.081848 | 3.091682 | 3.165153 | 2.944277 | 3.168777 | 3.144109 | 3.099308 ± 0.076991 |
| prog_sleep | 3.010436 | 3.057268 | 3.021819 | 3.173127 | 3.108728 | 3.121912 | 3.082215 ± 0.057686 |
| fixed_sleep | 3.049673 | 3.190153 | 3.043940 | 3.031179 | 3.060492 | 3.086449 | 3.076981 ± 0.053393 |

### L = 2048

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 1.777857 | 1.912109 | 1.740865 | 1.748879 | 1.728790 | 1.776491 | 1.780832 ± 0.061357 |
| prog_total | 1.795986 | 1.741791 | 1.692497 | 1.685354 | 1.657242 | 1.707991 | 1.713477 ± 0.044783 |
| prog_sleep | 1.768572 | 1.751245 | 1.711638 | 1.699932 | 1.693468 | 1.725618 | 1.725079 ± 0.027046 |
| fixed_sleep | 1.691541 | 1.728918 | 1.705069 | 1.718443 | 1.697241 | 1.739289 | 1.713417 ± 0.017053 |

## W9

### L = 512

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 2.725944 | 2.775416 | 2.738019 | 2.705137 | 2.733068 | 2.764453 | 2.740339 ± 0.023510 |
| prog_total | 2.995308 | 3.002814 | 3.060084 | 2.888194 | 3.062834 | 3.043673 | 3.008818 ± 0.059902 |
| prog_sleep | 2.938949 | 2.975270 | 2.948197 | 3.065861 | 3.015828 | 3.026131 | 2.995039 ± 0.044992 |
| fixed_sleep | 2.970598 | 3.079104 | 2.965488 | 2.955950 | 2.978361 | 2.999165 | 2.991444 ± 0.041399 |

### L = 2048

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 1.979149 | 2.083224 | 1.950666 | 1.956580 | 1.940894 | 1.978105 | 1.981437 ± 0.047584 |
| prog_total | 1.993968 | 1.951238 | 1.913721 | 1.907883 | 1.885898 | 1.925526 | 1.929705 ± 0.034789 |
| prog_sleep | 1.972269 | 1.958721 | 1.928338 | 1.919030 | 1.913932 | 1.939067 | 1.938560 ± 0.020958 |
| fixed_sleep | 1.912410 | 1.941174 | 1.923379 | 1.933301 | 1.916845 | 1.949727 | 1.929472 ± 0.013226 |

## Gaps

On W. prog_sleep minus prog_total, prog_sleep minus fixed_sleep, prog_total minus prog.

### L = 512

| seed | prog_sleep − prog_total | prog_sleep − fixed_sleep | prog_total − prog |
|---:|---:|---:|---:|
| 24 | -0.071412 | -0.039237 | 0.345193 |
| 25 | -0.034414 | -0.132885 | 0.291164 |
| 26 | -0.143334 | -0.022121 | 0.413017 |
| 27 | 0.228849 | 0.141948 | 0.234262 |
| 28 | -0.060049 | 0.048236 | 0.422890 |
| 29 | -0.022198 | 0.035462 | 0.357795 |
| mean | -0.017093 | 0.005234 | 0.344053 |
| std ddof=0 | 0.116581 | 0.084908 | 0.065844 |
| left lower (count) | 5 of 6 | 3 of 6 | 0 of 6 |

### L = 2048

| seed | prog_sleep − prog_total | prog_sleep − fixed_sleep | prog_total − prog |
|---:|---:|---:|---:|
| 24 | -0.027414 | 0.077032 | 0.018129 |
| 25 | 0.009454 | 0.022327 | -0.170318 |
| 26 | 0.019142 | 0.006569 | -0.048368 |
| 27 | 0.014579 | -0.018510 | -0.063525 |
| 28 | 0.036226 | -0.003773 | -0.071549 |
| 29 | 0.017628 | -0.013671 | -0.068500 |
| mean | 0.011602 | 0.011662 | -0.067355 |
| std ddof=0 | 0.019301 | 0.032154 | 0.055231 |
| left lower (count) | 1 of 6 | 3 of 6 | 5 of 6 |

## Sleep share

Sleep decisions divided by L. fixed_sleep uses prog_sleep's count on the same seed and L.

### L = 512

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_total | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_sleep | 0.109375 | 0.109375 | 0.125000 | 0.109375 | 0.109375 | 0.109375 | 0.111979 ± 0.005823 |
| fixed_sleep | 0.109375 | 0.109375 | 0.125000 | 0.109375 | 0.109375 | 0.109375 | 0.111979 ± 0.005823 |

### L = 2048

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_total | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_sleep | 0.136719 | 0.101562 | 0.132812 | 0.109375 | 0.099609 | 0.125000 | 0.117513 ± 0.014721 |
| fixed_sleep | 0.136719 | 0.101562 | 0.132812 | 0.109375 | 0.099609 | 0.125000 | 0.117513 ± 0.014721 |

## prog_sleep sleep share by quarter

Quarter i is decisions [(i−1)L/4, iL/4). The share is sleep decisions in the quarter over L/4.

### L = 512

| seed | quarter 1 | quarter 2 | quarter 3 | quarter 4 |
|---:|---:|---:|---:|---:|
| 24 | 0.062500 | 0.062500 | 0.187500 | 0.125000 |
| 25 | 0.125000 | 0.000000 | 0.187500 | 0.125000 |
| 26 | 0.062500 | 0.062500 | 0.250000 | 0.125000 |
| 27 | 0.125000 | 0.062500 | 0.187500 | 0.062500 |
| 28 | 0.062500 | 0.062500 | 0.187500 | 0.125000 |
| 29 | 0.125000 | 0.062500 | 0.187500 | 0.062500 |
| mean | 0.093750 | 0.052083 | 0.197917 | 0.104167 |

### L = 2048

| seed | quarter 1 | quarter 2 | quarter 3 | quarter 4 |
|---:|---:|---:|---:|---:|
| 24 | 0.109375 | 0.187500 | 0.046875 | 0.203125 |
| 25 | 0.109375 | 0.015625 | 0.093750 | 0.187500 |
| 26 | 0.125000 | 0.078125 | 0.078125 | 0.250000 |
| 27 | 0.109375 | 0.125000 | 0.031250 | 0.171875 |
| 28 | 0.109375 | 0.078125 | 0.000000 | 0.210938 |
| 29 | 0.109375 | 0.031250 | 0.187500 | 0.171875 |
| mean | 0.111979 | 0.085938 | 0.072917 | 0.199219 |

## Decision shares

Mean over seeds. A stay counts for the room channel. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.

| controller | L | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prog | 512 | 0.0208 | 0.0885 | 0.0361 | 0.1276 | 0.1250 | 0.1296 | 0.1634 | 0.0182 | 0.1094 | 0.1813 | 0.0000 |
| prog_total | 512 | 0.0208 | 0.1976 | 0.0182 | 0.1589 | 0.1589 | 0.1911 | 0.0234 | 0.0182 | 0.0781 | 0.1348 | 0.0000 |
| prog_sleep | 512 | 0.0182 | 0.0951 | 0.0365 | 0.1458 | 0.1432 | 0.1745 | 0.0316 | 0.0234 | 0.0833 | 0.1364 | 0.1120 |
| fixed_sleep | 512 | 0.0208 | 0.1462 | 0.0182 | 0.1484 | 0.1432 | 0.1755 | 0.0182 | 0.0182 | 0.0703 | 0.1289 | 0.1120 |
| prog | 2048 | 0.2415 | 0.0605 | 0.0908 | 0.0671 | 0.0677 | 0.0717 | 0.1328 | 0.0111 | 0.0618 | 0.1949 | 0.0000 |
| prog_total | 2048 | 0.3098 | 0.0671 | 0.1054 | 0.0612 | 0.0664 | 0.0684 | 0.1536 | 0.0111 | 0.0365 | 0.1206 | 0.0000 |
| prog_sleep | 2048 | 0.2648 | 0.0658 | 0.0951 | 0.0562 | 0.0547 | 0.0630 | 0.1396 | 0.0117 | 0.0319 | 0.0998 | 0.1175 |
| fixed_sleep | 2048 | 0.2720 | 0.0618 | 0.0957 | 0.0551 | 0.0568 | 0.0612 | 0.1395 | 0.0098 | 0.0332 | 0.0974 | 0.1175 |

## prog_sleep visit rewards by quarter

Exam-store bits saved per decision, mean over the visits whose first decision falls in the quarter. A quarter with no visits of that kind is blank. The mean row is the mean of the per-seed means, over seeds that have at least one such visit.

### L = 512

| seed | quarter | sleep visits | sleep mean reward | room visits | room mean reward |
|---:|---:|---:|---:|---:|---:|
| 24 | 1 | 1 | 22.487975 | 14 | 20.627943 |
| 24 | 2 | 1 | 9.288905 | 11 | 13.788175 |
| 24 | 3 | 3 | 6.201790 | 10 | 4.267861 |
| 24 | 4 | 2 | 4.312249 | 11 | 14.321388 |
| 25 | 1 | 2 | 18.642111 | 13 | 16.196272 |
| 25 | 2 | 0 | — | 14 | 18.750351 |
| 25 | 3 | 3 | 7.220420 | 10 | 4.609809 |
| 25 | 4 | 2 | 4.707832 | 12 | 9.349019 |
| 26 | 1 | 1 | 23.172418 | 14 | 16.463017 |
| 26 | 2 | 1 | 13.389881 | 13 | 21.467044 |
| 26 | 3 | 4 | 6.260342 | 10 | 3.676994 |
| 26 | 4 | 2 | 5.587006 | 12 | 10.235110 |
| 27 | 1 | 2 | 21.339201 | 13 | 15.005543 |
| 27 | 2 | 1 | 8.751913 | 14 | 19.966178 |
| 27 | 3 | 3 | 4.698372 | 9 | 3.291833 |
| 27 | 4 | 1 | 4.181879 | 14 | 12.913772 |
| 28 | 1 | 1 | 21.534051 | 14 | 18.204671 |
| 28 | 2 | 1 | 7.829150 | 13 | 18.837566 |
| 28 | 3 | 3 | 5.826276 | 10 | 4.309343 |
| 28 | 4 | 2 | 3.945979 | 12 | 8.887572 |
| 29 | 1 | 2 | 21.817601 | 13 | 14.288936 |
| 29 | 2 | 1 | 11.434035 | 13 | 20.774829 |
| 29 | 3 | 3 | 5.776976 | 9 | 3.221133 |
| 29 | 4 | 1 | 4.139617 | 14 | 12.330719 |
| mean | 1 | 9 | 21.498893 | 81 | 16.797730 |
| mean | 2 | 5 | 10.138777 | 78 | 18.930691 |
| mean | 3 | 19 | 5.997363 | 58 | 3.896162 |
| mean | 4 | 10 | 4.479094 | 75 | 11.339597 |

### L = 2048

| seed | quarter | sleep visits | sleep mean reward | room visits | room mean reward |
|---:|---:|---:|---:|---:|---:|
| 24 | 1 | 7 | 8.429535 | 46 | 13.986840 |
| 24 | 2 | 12 | 5.087105 | 47 | 15.745836 |
| 24 | 3 | 3 | 4.094713 | 59 | 13.119609 |
| 24 | 4 | 13 | 3.220960 | 44 | 3.043635 |
| 25 | 1 | 7 | 9.765878 | 49 | 12.884546 |
| 25 | 2 | 1 | 4.538528 | 59 | 19.867155 |
| 25 | 3 | 6 | 4.627109 | 52 | 7.821126 |
| 25 | 4 | 12 | 2.956368 | 43 | 2.724130 |
| 26 | 1 | 8 | 9.097210 | 49 | 13.651797 |
| 26 | 2 | 5 | 6.262958 | 55 | 16.648286 |
| 26 | 3 | 5 | 4.404433 | 54 | 11.623022 |
| 26 | 4 | 16 | 3.388828 | 41 | 3.113555 |
| 27 | 1 | 7 | 9.958187 | 50 | 13.635660 |
| 27 | 2 | 8 | 4.914751 | 51 | 17.421447 |
| 27 | 3 | 2 | 7.091790 | 57 | 10.490204 |
| 27 | 4 | 11 | 3.766112 | 46 | 3.652391 |
| 28 | 1 | 7 | 7.819141 | 49 | 13.151976 |
| 28 | 2 | 5 | 4.456213 | 55 | 17.787847 |
| 28 | 3 | 0 | — | 60 | 11.878439 |
| 28 | 4 | 14 | 3.085795 | 42 | 2.693562 |
| 29 | 1 | 7 | 10.934255 | 49 | 13.111886 |
| 29 | 2 | 2 | 6.745707 | 56 | 21.577928 |
| 29 | 3 | 12 | 5.309256 | 45 | 5.277061 |
| 29 | 4 | 11 | 3.357653 | 47 | 5.376780 |
| mean | 1 | 43 | 9.334034 | 292 | 13.403784 |
| mean | 2 | 33 | 5.334210 | 323 | 18.174750 |
| mean | 3 | 28 | 5.105460 | 327 | 10.034910 |
| mean | 4 | 77 | 3.295953 | 263 | 3.434009 |

## Mean held-out bits

Mean over seeds. W is the mean of the 7 learnable channels. W9 is the mean of all 9.

| controller | L | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | W | W9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prog | 512 | 4.985308 | 4.214086 | 4.890399 | 1.372426 | 1.322840 | 0.478978 | 2.022745 | 5.358370 | 0.017905 | 2.755254 | 2.740339 |
| prog_total | 512 | 4.985242 | 3.858511 | 5.083599 | 1.338823 | 1.287932 | 0.399575 | 4.741473 | 5.358337 | 0.025869 | 3.099308 | 3.008818 |
| prog_sleep | 512 | 4.936303 | 4.338086 | 4.910096 | 1.336686 | 1.275372 | 0.394279 | 4.384682 | 5.358667 | 0.021184 | 3.082215 | 2.995039 |
| fixed_sleep | 512 | 4.871173 | 3.938083 | 5.007117 | 1.331576 | 1.280270 | 0.424454 | 4.686193 | 5.358934 | 0.025199 | 3.076981 | 2.991444 |
| prog | 2048 | 1.453719 | 3.719013 | 3.740909 | 1.280062 | 1.210299 | 0.441241 | 0.620579 | 5.359530 | 0.007577 | 1.780832 | 1.981437 |
| prog_total | 2048 | 1.242231 | 3.579863 | 3.661828 | 1.290113 | 1.212738 | 0.419153 | 0.588411 | 5.359763 | 0.013250 | 1.713477 | 1.929705 |
| prog_sleep | 2048 | 1.337334 | 3.592099 | 3.692004 | 1.272033 | 1.212871 | 0.379969 | 0.589243 | 5.360889 | 0.010593 | 1.725079 | 1.938560 |
| fixed_sleep | 2048 | 1.316248 | 3.541367 | 3.680995 | 1.274129 | 1.216869 | 0.373794 | 0.590515 | 5.360854 | 0.010482 | 1.713417 | 1.929472 |

## Store sizes

Pairs stored at the end of life. R = 2,048 per channel. prog keeps M5's own reservoir and has no split store. A total is the sum over the 9 room channels.

### L = 512

| controller | seed | store | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | total |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prog_total | 24 | exam | 264 | 2048 | 271 | 2048 | 2048 | 2048 | 254 | 255 | 1262 | 10498 |
| prog_total | 24 | replay | 248 | 2048 | 241 | 2048 | 2048 | 2048 | 258 | 257 | 1298 | 10494 |
| prog_total | 25 | exam | 270 | 2048 | 513 | 2048 | 2048 | 2048 | 255 | 248 | 1278 | 10756 |
| prog_total | 25 | replay | 242 | 2048 | 511 | 2048 | 2048 | 2048 | 257 | 264 | 1282 | 10748 |
| prog_total | 26 | exam | 526 | 2048 | 257 | 2048 | 2048 | 2048 | 241 | 252 | 1299 | 10767 |
| prog_total | 26 | replay | 498 | 2048 | 255 | 2048 | 2048 | 2048 | 271 | 260 | 1261 | 10737 |
| prog_total | 27 | exam | 491 | 2048 | 255 | 2048 | 2048 | 2048 | 1031 | 267 | 1304 | 11540 |
| prog_total | 27 | replay | 533 | 2048 | 257 | 2048 | 2048 | 2048 | 1017 | 245 | 1256 | 11500 |
| prog_total | 28 | exam | 259 | 2048 | 246 | 2048 | 2048 | 2048 | 246 | 256 | 1278 | 10477 |
| prog_total | 28 | replay | 253 | 2048 | 266 | 2048 | 2048 | 2048 | 266 | 256 | 1282 | 10515 |
| prog_total | 29 | exam | 271 | 2048 | 261 | 2048 | 2048 | 2048 | 250 | 515 | 1298 | 10787 |
| prog_total | 29 | replay | 241 | 2048 | 251 | 2048 | 2048 | 2048 | 262 | 509 | 1262 | 10717 |
| prog_sleep | 24 | exam | 264 | 234 | 271 | 2048 | 2048 | 2048 | 1281 | 255 | 1561 | 10010 |
| prog_sleep | 24 | replay | 248 | 278 | 241 | 2048 | 2048 | 2048 | 1343 | 257 | 1511 | 10022 |
| prog_sleep | 25 | exam | 270 | 261 | 2048 | 2048 | 2048 | 2048 | 502 | 248 | 1551 | 11024 |
| prog_sleep | 25 | replay | 242 | 251 | 2048 | 2048 | 2048 | 2048 | 522 | 264 | 1521 | 10992 |
| prog_sleep | 26 | exam | 266 | 2000 | 257 | 2048 | 2048 | 2048 | 471 | 512 | 1256 | 10906 |
| prog_sleep | 26 | replay | 246 | 1968 | 255 | 2048 | 2048 | 2048 | 553 | 512 | 1304 | 10982 |
| prog_sleep | 27 | exam | 234 | 2048 | 255 | 2048 | 2048 | 2048 | 256 | 267 | 1296 | 10500 |
| prog_sleep | 27 | replay | 278 | 2048 | 257 | 2048 | 2048 | 2048 | 256 | 245 | 1264 | 10492 |
| prog_sleep | 28 | exam | 541 | 1328 | 246 | 2048 | 2048 | 2048 | 246 | 518 | 1296 | 10319 |
| prog_sleep | 28 | replay | 483 | 1360 | 266 | 2048 | 2048 | 2048 | 266 | 506 | 1264 | 10289 |
| prog_sleep | 29 | exam | 271 | 2048 | 261 | 2048 | 2048 | 2048 | 250 | 515 | 1299 | 10788 |
| prog_sleep | 29 | replay | 241 | 2048 | 251 | 2034 | 2039 | 2048 | 262 | 509 | 1261 | 10693 |
| fixed_sleep | 24 | exam | 264 | 2048 | 271 | 2048 | 2048 | 2048 | 254 | 255 | 1046 | 10282 |
| fixed_sleep | 24 | replay | 248 | 2048 | 241 | 2048 | 2048 | 2048 | 258 | 257 | 1002 | 10198 |
| fixed_sleep | 25 | exam | 270 | 1214 | 513 | 2048 | 2048 | 2048 | 255 | 248 | 1280 | 9924 |
| fixed_sleep | 25 | replay | 242 | 1218 | 511 | 2048 | 2048 | 2048 | 257 | 264 | 1280 | 9916 |
| fixed_sleep | 26 | exam | 526 | 2048 | 257 | 2048 | 2048 | 2048 | 241 | 252 | 1274 | 10742 |
| fixed_sleep | 26 | replay | 498 | 2028 | 255 | 2048 | 2048 | 2048 | 271 | 260 | 1286 | 10742 |
| fixed_sleep | 27 | exam | 491 | 2048 | 255 | 2048 | 2048 | 2048 | 516 | 267 | 1010 | 10731 |
| fixed_sleep | 27 | replay | 533 | 2048 | 257 | 2042 | 2034 | 2048 | 508 | 245 | 1038 | 10753 |
| fixed_sleep | 28 | exam | 259 | 2048 | 246 | 2048 | 2048 | 2048 | 246 | 256 | 1255 | 10454 |
| fixed_sleep | 28 | replay | 253 | 2048 | 266 | 2048 | 2048 | 2048 | 266 | 256 | 1305 | 10538 |
| fixed_sleep | 29 | exam | 271 | 2048 | 261 | 2048 | 2033 | 2048 | 250 | 515 | 1053 | 10527 |
| fixed_sleep | 29 | replay | 241 | 2048 | 251 | 2048 | 2048 | 2048 | 262 | 509 | 995 | 10450 |

### L = 2048

| controller | seed | store | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | total |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prog_total | 24 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 995 | 2048 | 17379 |
| prog_total | 24 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 1053 | 2048 | 17437 |
| prog_total | 25 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 741 | 2048 | 17125 |
| prog_total | 25 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 795 | 2048 | 17179 |
| prog_total | 26 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 252 | 2048 | 16636 |
| prog_total | 26 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 260 | 2048 | 16644 |
| prog_total | 27 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 787 | 2048 | 17171 |
| prog_total | 27 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 749 | 2048 | 17133 |
| prog_total | 28 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 519 | 2048 | 16903 |
| prog_total | 28 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 505 | 2048 | 16889 |
| prog_total | 29 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 1018 | 2048 | 17402 |
| prog_total | 29 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 1030 | 2048 | 17414 |
| prog_sleep | 24 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 773 | 2048 | 17157 |
| prog_sleep | 24 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 763 | 2048 | 17147 |
| prog_sleep | 25 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 765 | 2048 | 17149 |
| prog_sleep | 25 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 771 | 2044 | 17151 |
| prog_sleep | 26 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 1036 | 2038 | 17410 |
| prog_sleep | 26 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 1012 | 2048 | 17396 |
| prog_sleep | 27 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 533 | 2048 | 16917 |
| prog_sleep | 27 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 491 | 2018 | 16845 |
| prog_sleep | 28 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 518 | 2040 | 16894 |
| prog_sleep | 28 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 506 | 2048 | 16890 |
| prog_sleep | 29 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 1032 | 2048 | 17416 |
| prog_sleep | 29 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 1016 | 2048 | 17400 |
| fixed_sleep | 24 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 519 | 2048 | 16903 |
| fixed_sleep | 24 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 505 | 2024 | 16865 |
| fixed_sleep | 25 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 741 | 2048 | 17125 |
| fixed_sleep | 25 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 795 | 2048 | 17179 |
| fixed_sleep | 26 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 252 | 2017 | 16605 |
| fixed_sleep | 26 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 260 | 2048 | 16644 |
| fixed_sleep | 27 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 787 | 2048 | 17171 |
| fixed_sleep | 27 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 749 | 2048 | 17133 |
| fixed_sleep | 28 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 519 | 2048 | 16903 |
| fixed_sleep | 28 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 505 | 2027 | 16868 |
| fixed_sleep | 29 | exam | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 1018 | 2048 | 17402 |
| fixed_sleep | 29 | replay | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | 1030 | 1988 | 17354 |

## Ring order

Channel in rooms 0..8, seeds 24–29. One order per seed; controllers share it.

| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---:|---|---|---|---|---|---|---|---|---|
| 24 | N | M7 | G | C7 | M37 | E30 | C1 | E110 | Z |
| 25 | G | N | E30 | C1 | M7 | C7 | M37 | Z | E110 |
| 26 | C1 | N | C7 | G | Z | M7 | M37 | E110 | E30 |
| 27 | M37 | E30 | C1 | E110 | G | Z | C7 | N | M7 |
| 28 | M37 | G | E30 | N | E110 | M7 | C1 | Z | C7 |
| 29 | M37 | Z | E110 | C7 | M7 | N | G | E30 | C1 |

## Walls

Smoke wall 53.405 s. Kernel wall 1688.134 s.

| kernel | seed | wall seconds |
|---|---:|---:|
| ezzzzzx/m6b-w3 | 24 | 290.998 |
| ezzzzzx/m6b-w3 | 25 | 263.265 |
| ezzzzzx/m6b-w3 | 26 | 297.241 |
| ezzzzzx/m6b-w3 | 27 | 274.982 |
| ezzzzzx/m6b-w3 | 28 | 274.968 |
| ezzzzzx/m6b-w3 | 29 | 286.655 |

| controller | L | seed | wall seconds |
|---|---:|---:|---:|
| prog | 512 | 24 | 11.950 |
| prog | 512 | 25 | 9.399 |
| prog | 512 | 26 | 12.192 |
| prog | 512 | 27 | 9.967 |
| prog | 512 | 28 | 12.151 |
| prog | 512 | 29 | 12.229 |
| prog_total | 512 | 24 | 26.679 |
| prog_total | 512 | 25 | 25.711 |
| prog_total | 512 | 26 | 26.872 |
| prog_total | 512 | 27 | 26.377 |
| prog_total | 512 | 28 | 27.318 |
| prog_total | 512 | 29 | 25.773 |
| prog_sleep | 512 | 24 | 33.881 |
| prog_sleep | 512 | 25 | 35.317 |
| prog_sleep | 512 | 26 | 38.202 |
| prog_sleep | 512 | 27 | 36.183 |
| prog_sleep | 512 | 28 | 36.477 |
| prog_sleep | 512 | 29 | 33.796 |
| fixed_sleep | 512 | 24 | 34.252 |
| fixed_sleep | 512 | 25 | 32.879 |
| fixed_sleep | 512 | 26 | 35.787 |
| fixed_sleep | 512 | 27 | 35.697 |
| fixed_sleep | 512 | 28 | 34.720 |
| fixed_sleep | 512 | 29 | 35.353 |
| prog | 2048 | 24 | 47.942 |
| prog | 2048 | 25 | 47.309 |
| prog | 2048 | 26 | 47.678 |
| prog | 2048 | 27 | 47.608 |
| prog | 2048 | 28 | 48.431 |
| prog | 2048 | 29 | 47.938 |
| prog_total | 2048 | 24 | 115.541 |
| prog_total | 2048 | 25 | 112.280 |
| prog_total | 2048 | 26 | 114.869 |
| prog_total | 2048 | 27 | 122.065 |
| prog_total | 2048 | 28 | 125.013 |
| prog_total | 2048 | 29 | 116.263 |
| prog_sleep | 2048 | 24 | 168.391 |
| prog_sleep | 2048 | 25 | 156.000 |
| prog_sleep | 2048 | 26 | 175.677 |
| prog_sleep | 2048 | 27 | 159.617 |
| prog_sleep | 2048 | 28 | 161.666 |
| prog_sleep | 2048 | 29 | 166.990 |
| fixed_sleep | 2048 | 24 | 122.525 |
| fixed_sleep | 2048 | 25 | 107.218 |
| fixed_sleep | 2048 | 26 | 121.515 |
| fixed_sleep | 2048 | 27 | 115.316 |
| fixed_sleep | 2048 | 28 | 113.252 |
| fixed_sleep | 2048 | 29 | 119.616 |

## Smoke test

```
smoke check 0: packed score matches score_pairs
  packed score abs 0.000e+00 on 8 random reservoirs
smoke check 1: prog seed 12 L=512 vs results/m5bw3/results.json
  prog W 2.742358251650 ref 2.742358251650 abs 0.000e+00 W9 2.730350 finite True wall 16.11s
smoke check 2: prog_sleep seed 24 L=256, disjoint stores
  stream index is the symbol position in that channel's observed stream
  G exam 264 (offered 264) replay 248 (offered 248)
  C1 exam 234 (offered 234) replay 278 (offered 278)
  C7 exam 271 (offered 271) replay 241 (offered 241)
  E30 exam 1266 (offered 1266) replay 1294 (offered 1294)
  E110 exam 1541 (offered 1541) replay 1531 (offered 1531)
  M7 exam 1774 (offered 1774) replay 1810 (offered 1810)
  M37 exam 254 (offered 254) replay 258 (offered 258)
  N exam 255 (offered 255) replay 257 (offered 257)
  Z exam 512 (offered 512) replay 512 (offered 512)
  overlap 0 score_gap 0.000e+00 W 3.301383 finite True wall 13.12s
smoke check 3: exam pairs drawn by sleep during that life
  exam_pairs_drawn 0
smoke check 4: seed 24 L=128, four controllers, W and W9 finite
  prog W 3.727773 W9 3.503168 finite True sleep 0.0000 corridor 0.1484 wall 4.41s learn 3.29s eval 1.12s score 0.00s
  prog_total W 3.817922 W9 3.586428 finite True sleep 0.0000 corridor 0.1094 wall 4.76s learn 3.57s eval 1.19s score 1.89s
  prog_sleep W 3.695197 W9 3.487397 finite True sleep 0.0625 corridor 0.0938 wall 4.85s learn 3.75s eval 1.10s score 2.17s
  fixed_sleep W 3.781550 W9 3.558138 finite True sleep 0.0625 corridor 0.1094 wall 4.85s learn 3.73s eval 1.12s score 2.12s
  pessimistic exam rescore, one repetitive channel of R pairs
  pessimistic exam rescore per decision 0.0994s
estimate one-seed kernel wall 460s (7.7 min), six seeds sequential 2763s (46.0 min), pool 4 limit 90 min
estimate charges every decision a full exam store and is an upper bound from this VPS
estimate is within 90 min for one kernel of six seeds
```

## Protocol

- `prog` is `m5_w3.run_controller` with the M5 per-channel reward and no sleep. W9 is `m5b_w3.score_nz` on that frozen compressor. `m6_w3.py`, `m5_w3.py`, `m5b_w3.py`, `curio.py` and `w3_world.py` are imported and not edited.
- `prog_total` is M5's 9-room bandit. `prog_sleep` is the same bandit over 10 arms. Sleep is the last arm of the optimistic first pass, then eps-greedy with EPS=0.1 and BETA=0.3. Choice rng is `default_rng(10_000 + seed)`. N-channel noise and corridor steps use `default_rng(1_000 + seed)`.
- Every (context, symbol) pair from a room glance is scored and learned, then assigned by a fair coin, `default_rng(40_000 + seed).integers(2) == 0`, to the exam store, and otherwise to the replay store. Each store is a reservoir of R = 2,048 pairs per channel with its own offer counter. Exam replacement uses `default_rng(20_000 + seed)`. Replay replacement uses `default_rng(50_000 + seed)`. The replacement rng is drawn only when that channel's reservoir is full. Corridor glances are learned and enter neither store.
- A decision's progress is the sum over the 9 room channels of `score_pairs` on the exam store before the learning step minus `score_pairs` after it, on the pairs stored before the decision's own pairs enter. The visit reward is that sum over the visit's decisions, divided by the decisions the visit took. Travel counts.
- The exam score is the sum of per-pair codelengths. A pair is recomputed when a learning step writes one of its feature rows. The sum is checked against `SGDCompressor.score_pairs`. The smoke check reports the absolute gap. A stream index is the symbol's position in that channel's observed stream. Smoke check 2 prints the store sizes after a 256-decision life.
- A sleep decision draws 64 pairs uniformly with replacement from the union of the replay stores, rng `default_rng(30_000 + seed)`, and applies the compressor's learning step to each pair on its own channel. It does not draw from an exam store. Drawn pairs enter no reservoir. An empty union learns nothing. A sleep visit is 8 such decisions where the agent stands, with no travel. Its reward is the summed exam-store progress divided by the decisions it took. N is EVAL uniform symbols from `default_rng(30000 + seed)`, a fresh generator, not the sleep rng.
- `fixed_sleep` is the `prog_total` controller plus planned sleep visits. The visit lengths are 8, with a shorter last visit when the count is not a multiple of 8, and the lengths sum to `prog_sleep`'s sleep decisions on that seed and L. Awake gaps before, between and after the visits differ by at most one decision. A due sleep that falls inside a room visit starts on the next decision after that visit. A room visit is not started when the decisions left would not cover the sleeps still owed plus one maximum room visit (4 moves + 8 dwells).
- W is `w3_headroom.evaluate`. W9 adds N and Z on the same frozen compressor. Z is EVAL space symbols. Corridor is not in W or W9.
- Quarter i is decisions [(i−1)L/4, iL/4). A visit reward is counted in the quarter of its first decision. Seeds 24–29. L = 2048 is the scored life. L = 512 is the report life. Nothing retuned.
- PYTHONHASHSEED=0. CURIO_LR=0.004. CPU. Private Kaggle kernels, pool of 4, no GPU, no internet.

## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (prog_sleep − prog_total ≤ −0.05, 5 of 6): FAIL.** +0.012 ± 0.019; prog_sleep is lower on 1 of 6 seeds.
**Bar 2 (prog_sleep − fixed_sleep ≤ −0.03, 4 of 6): FAIL.** +0.012 ± 0.032, 3 of 6.
Kill rule 1 fires.

**Mechanism.** With an honest yardstick, the agent judges sleep correctly. Its share falls from 33% (M6) to 12%. Sleep's
reward sits below the room reward until the last quarter, when the world's easy progress is spent (sleep 3.30 vs rooms
3.43). Then the share rises to 20%. But sleep buys nothing on fresh data, and a matched fixed dose buys nothing
either. The reason is the learner. `SGDCompressor` is a hashed table of per-context weights. An update to one context
barely touches the others, so it hardly forgets, and a repeater has nothing to repair. This is the theory's own
prediction (`docs/compute-vs-data-movement.md`): replay is a repeater against *lossy* carriage, and slot-wise storage is
nearly lossless. Sleep can only earn its cost in a cortex whose shared weights interfere, such as H31's GRU, or the
diagonal bank.

**Side finding.** The whole-store yardstick helps the controller over a long life (prog_total − prog = −0.067, 5 of 6
at L = 2,048). Over a short one it hurts (+0.344 at L = 512): early on, everything is progress on the whole store, and
the per-channel reward separates sources better.

**Next for this line.** Give W3 a cortex that forgets: the trace-learned roots bank (H30y), with the curious controller,
honest split stores and sleep. That one run puts the north star's whole loop on a local substrate.
