# M8-W5 curiosity on the drifting ring

W5's 18-room ring with period 128. One learnable room's rule is redrawn every 128 decisions. T14 is the mean of W14 at decisions L/8, 2L/8, …, L. W14 is bits/symbol of the frozen bank on a fresh EVAL=2048 continuation of the 14 learnable rooms. The final W14 is the checkpoint at L. Lower is better. Every controller sleeps for the fixed share 0.20, in visits of 8, placed as in W5. At L = 2048 that is 408 decisions, share 0.199219.

`prog_chan_s` ranks 18 rooms by the stored EMA. `prog_age_s` uses the same EMA and, on the greedy step only, ranks by v + (v_max − v) · (1 − exp(−age / 256)). Age is the decisions since that room's last visit ended. v_max is the largest single-visit reward so far. The stored EMA is not aged. The room reward is the sum, over the visited room's exam glances, of bits before minus bits after, summed over the visit's decisions and divided by those decisions. Travel counts. Sleep visits are not bandit updates. A swap's latency is the decisions until the next stay in that room. If the life ends first, the latency is L minus the swap decision and the swap is censored. within 128 counts a stay whose latency is at most 128. The median includes censored latencies.

## T14

| controller | seed 12 | seed 13 | seed 14 | seed 15 | seed 16 | seed 17 | seed 18 | seed 19 | seed 20 | seed 21 | seed 22 | seed 23 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 2.553480 | 2.274791 | 1.958148 | 2.066528 | 2.337884 | 2.407411 | 2.270295 | 1.991909 | 2.691968 | 2.417047 | 2.470723 | 2.548606 | 2.332399 ± 0.221872 |
| oracle_fresh_s | 2.217438 | 2.052588 | 1.875702 | 1.693669 | 2.147240 | 2.087998 | 2.024485 | 1.756401 | 2.027284 | 2.043589 | 1.987201 | 1.979534 | 1.991094 ± 0.144618 |
| prog_chan_s | 1.944190 | 1.949176 | 1.587206 | 1.706560 | 1.960518 | 2.110890 | 1.981737 | 2.078470 | 2.056683 | 1.563020 | 2.054383 | 2.131306 | 1.927012 ± 0.189953 |
| prog_age_s | 1.785188 | 1.918485 | 1.581165 | 1.570576 | 1.736055 | 1.596326 | 1.736144 | 1.880330 | 1.748226 | 1.698552 | 1.878288 | 1.819841 | 1.745765 ± 0.113682 |

## Final W14

| controller | seed 12 | seed 13 | seed 14 | seed 15 | seed 16 | seed 17 | seed 18 | seed 19 | seed 20 | seed 21 | seed 22 | seed 23 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 1.170735 | 1.656162 | 1.179882 | 1.225463 | 1.853635 | 1.069073 | 1.030575 | 1.471300 | 1.232120 | 1.132102 | 1.574698 | 1.598420 | 1.349514 ± 0.257101 |
| oracle_fresh_s | 1.479607 | 1.102714 | 1.525876 | 0.885186 | 1.450339 | 1.771249 | 0.850446 | 1.870601 | 1.053786 | 1.372413 | 1.348555 | 1.383463 | 1.341186 ± 0.305705 |
| prog_chan_s | 1.723677 | 1.478746 | 1.062432 | 1.436207 | 1.701427 | 1.224982 | 1.368419 | 2.041296 | 1.329347 | 1.269055 | 1.584265 | 1.915747 | 1.511300 ± 0.278899 |
| prog_age_s | 1.759845 | 1.282478 | 0.865502 | 0.850855 | 1.361845 | 0.985695 | 0.915535 | 1.469353 | 0.996176 | 1.183130 | 1.670411 | 1.540619 | 1.240120 ± 0.308144 |

## T14 per checkpoint

| controller | seed | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 | T14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 12 | 5.762251 | 3.104824 | 2.640634 | 1.839998 | 2.352359 | 2.200443 | 1.356600 | 1.170735 | 2.553480 |
| rand_dwell_s | 13 | 5.440902 | 2.473546 | 1.332550 | 1.929595 | 1.427951 | 1.961253 | 1.976367 | 1.656162 | 2.274791 |
| rand_dwell_s | 14 | 5.016219 | 2.262681 | 1.846860 | 1.833498 | 1.410650 | 1.187547 | 0.927845 | 1.179882 | 1.958148 |
| rand_dwell_s | 15 | 4.071788 | 2.525566 | 1.701716 | 1.435654 | 2.067903 | 2.118716 | 1.385417 | 1.225463 | 2.066528 |
| rand_dwell_s | 16 | 4.895185 | 2.629902 | 1.990953 | 1.341160 | 1.891525 | 2.072855 | 2.027858 | 1.853635 | 2.337884 |
| rand_dwell_s | 17 | 5.791007 | 4.261880 | 2.150588 | 1.870445 | 1.313799 | 1.600819 | 1.201674 | 1.069073 | 2.407411 |
| rand_dwell_s | 18 | 6.089862 | 2.670260 | 2.313935 | 1.957661 | 1.624934 | 1.305727 | 1.169409 | 1.030575 | 2.270295 |
| rand_dwell_s | 19 | 3.943272 | 2.503024 | 2.539337 | 1.854323 | 0.980517 | 1.195864 | 1.447636 | 1.471300 | 1.991909 |
| rand_dwell_s | 20 | 7.612769 | 3.702187 | 2.137350 | 1.909320 | 1.806951 | 1.804887 | 1.330162 | 1.232120 | 2.691968 |
| rand_dwell_s | 21 | 6.284828 | 3.508236 | 2.554260 | 1.585456 | 1.906070 | 1.189232 | 1.176195 | 1.132102 | 2.417047 |
| rand_dwell_s | 22 | 4.257085 | 3.285166 | 2.586030 | 3.033306 | 1.718787 | 1.715863 | 1.594853 | 1.574698 | 2.470723 |
| rand_dwell_s | 23 | 5.056602 | 3.168361 | 2.510738 | 2.196706 | 2.319259 | 2.199176 | 1.339583 | 1.598420 | 2.548606 |
| oracle_fresh_s | 12 | 4.694007 | 3.013599 | 2.189368 | 1.241284 | 2.070252 | 1.755765 | 1.295620 | 1.479607 | 2.217438 |
| oracle_fresh_s | 13 | 4.422515 | 2.482155 | 1.804932 | 1.608641 | 1.280276 | 1.875158 | 1.844308 | 1.102714 | 2.052588 |
| oracle_fresh_s | 14 | 3.673969 | 2.107986 | 2.183635 | 1.363826 | 1.304746 | 1.382342 | 1.463235 | 1.525876 | 1.875702 |
| oracle_fresh_s | 15 | 2.726488 | 2.141369 | 1.481848 | 1.643164 | 1.862546 | 1.441384 | 1.367367 | 0.885186 | 1.693669 |
| oracle_fresh_s | 16 | 3.312465 | 2.625649 | 2.241660 | 1.860869 | 1.995736 | 1.731337 | 1.959862 | 1.450339 | 2.147240 |
| oracle_fresh_s | 17 | 2.864816 | 2.643896 | 2.139765 | 2.274314 | 1.414699 | 1.759304 | 1.835942 | 1.771249 | 2.087998 |
| oracle_fresh_s | 18 | 4.167934 | 2.433077 | 2.171561 | 2.003071 | 1.424389 | 1.980167 | 1.165235 | 0.850446 | 2.024485 |
| oracle_fresh_s | 19 | 2.758646 | 1.882054 | 2.262170 | 2.099337 | 1.133452 | 0.984474 | 1.060477 | 1.870601 | 1.756401 |
| oracle_fresh_s | 20 | 3.799824 | 3.111432 | 1.191352 | 1.765476 | 1.935859 | 1.798142 | 1.562399 | 1.053786 | 2.027284 |
| oracle_fresh_s | 21 | 3.014215 | 3.649974 | 2.182886 | 1.440056 | 2.139107 | 1.642049 | 0.908011 | 1.372413 | 2.043589 |
| oracle_fresh_s | 22 | 3.351895 | 2.656222 | 1.887350 | 1.976041 | 1.959620 | 1.200710 | 1.517215 | 1.348555 | 1.987201 |
| oracle_fresh_s | 23 | 3.558039 | 2.068467 | 1.817136 | 2.104836 | 2.085675 | 1.333236 | 1.485423 | 1.383463 | 1.979534 |
| prog_chan_s | 12 | 2.809471 | 1.675348 | 1.690362 | 1.534509 | 2.060528 | 2.378507 | 1.681117 | 1.723677 | 1.944190 |
| prog_chan_s | 13 | 3.139805 | 2.140414 | 1.735319 | 1.632208 | 2.320815 | 1.488318 | 1.657786 | 1.478746 | 1.949176 |
| prog_chan_s | 14 | 3.367771 | 1.836672 | 1.608231 | 1.415269 | 1.244563 | 1.130096 | 1.032612 | 1.062432 | 1.587206 |
| prog_chan_s | 15 | 2.361726 | 1.598877 | 2.042025 | 1.338828 | 1.817969 | 1.559856 | 1.496990 | 1.436207 | 1.706560 |
| prog_chan_s | 16 | 2.648042 | 2.671719 | 2.207301 | 1.177011 | 1.303012 | 1.480135 | 2.495502 | 1.701427 | 1.960518 |
| prog_chan_s | 17 | 2.409196 | 3.044206 | 2.869684 | 2.197778 | 1.284150 | 1.800403 | 2.056719 | 1.224982 | 2.110890 |
| prog_chan_s | 18 | 3.397148 | 2.575372 | 2.226173 | 1.994753 | 1.540098 | 1.474069 | 1.277864 | 1.368419 | 1.981737 |
| prog_chan_s | 19 | 2.357426 | 2.009017 | 1.899975 | 3.128953 | 1.860445 | 1.573346 | 1.757300 | 2.041296 | 2.078470 |
| prog_chan_s | 20 | 3.278146 | 2.855074 | 2.315442 | 1.917231 | 1.704187 | 1.483183 | 1.570854 | 1.329347 | 2.056683 |
| prog_chan_s | 21 | 2.268613 | 2.148732 | 2.145918 | 1.153113 | 1.249772 | 1.216770 | 1.052186 | 1.269055 | 1.563020 |
| prog_chan_s | 22 | 3.800560 | 2.603942 | 1.795652 | 1.392704 | 1.758505 | 2.076404 | 1.423029 | 1.584265 | 2.054383 |
| prog_chan_s | 23 | 3.214217 | 2.006711 | 1.746541 | 2.690729 | 2.156520 | 1.596637 | 1.723348 | 1.915747 | 2.131306 |
| prog_age_s | 12 | 2.809471 | 2.101224 | 1.645269 | 1.392495 | 1.678460 | 1.874394 | 1.020345 | 1.759845 | 1.785188 |
| prog_age_s | 13 | 3.126372 | 3.519336 | 1.270076 | 1.229860 | 1.363967 | 1.659483 | 1.896309 | 1.282478 | 1.918485 |
| prog_age_s | 14 | 3.367771 | 1.870450 | 1.566174 | 1.436798 | 1.404923 | 1.126062 | 1.011640 | 0.865502 | 1.581165 |
| prog_age_s | 15 | 2.063324 | 1.555108 | 1.679216 | 1.387192 | 2.151414 | 1.424722 | 1.452780 | 0.850855 | 1.570576 |
| prog_age_s | 16 | 2.439963 | 3.249696 | 1.630320 | 1.238260 | 1.408669 | 1.113504 | 1.446179 | 1.361845 | 1.736055 |
| prog_age_s | 17 | 2.410451 | 2.681888 | 1.633893 | 1.593738 | 1.033225 | 1.137887 | 1.293831 | 0.985695 | 1.596326 |
| prog_age_s | 18 | 3.493483 | 2.062060 | 1.461584 | 2.190248 | 1.601384 | 1.223459 | 0.941403 | 0.915535 | 1.736144 |
| prog_age_s | 19 | 2.687387 | 1.874637 | 1.984233 | 3.226246 | 0.973455 | 1.239880 | 1.587446 | 1.469353 | 1.880330 |
| prog_age_s | 20 | 3.244956 | 2.327334 | 1.285565 | 1.719921 | 1.808621 | 1.258196 | 1.345038 | 0.996176 | 1.748226 |
| prog_age_s | 21 | 2.408828 | 2.447982 | 2.221049 | 1.580636 | 1.554899 | 1.408063 | 0.783825 | 1.183130 | 1.698552 |
| prog_age_s | 22 | 3.180370 | 1.988852 | 1.568319 | 1.558247 | 1.855429 | 1.574976 | 1.629701 | 1.670411 | 1.878288 |
| prog_age_s | 23 | 3.226996 | 2.034213 | 1.404751 | 2.154568 | 2.064680 | 0.882803 | 1.250102 | 1.540619 | 1.819841 |

Mean over seeds at each checkpoint.

| controller | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 | T14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 5.351814 | 3.007969 | 2.192079 | 1.898927 | 1.735059 | 1.712698 | 1.411133 | 1.349514 | 2.332399 |
| oracle_fresh_s | 3.528734 | 2.567990 | 1.962805 | 1.781743 | 1.717197 | 1.573672 | 1.455425 | 1.341186 | 1.991094 |
| prog_chan_s | 2.921010 | 2.263840 | 2.023552 | 1.797757 | 1.691714 | 1.604810 | 1.602109 | 1.511300 | 1.927012 |
| prog_age_s | 2.871614 | 2.309398 | 1.612537 | 1.725684 | 1.574927 | 1.326953 | 1.304883 | 1.240120 | 1.745765 |

## Gaps

On T14. A negative gap means the first controller is lower. left lower counts seeds where the gap is negative.

| seed | prog_chan_s − rand_dwell_s | prog_age_s − rand_dwell_s | prog_age_s − prog_chan_s | oracle_fresh_s − rand_dwell_s |
|---|---:|---:|---:|---:|
| 12 | -0.609291 | -0.768293 | -0.159002 | -0.336043 |
| 13 | -0.325615 | -0.356306 | -0.030691 | -0.222203 |
| 14 | -0.370942 | -0.376983 | -0.006041 | -0.082446 |
| 15 | -0.359968 | -0.495951 | -0.135983 | -0.372859 |
| 16 | -0.377366 | -0.601830 | -0.224464 | -0.190645 |
| 17 | -0.296521 | -0.811085 | -0.514564 | -0.319412 |
| 18 | -0.288558 | -0.534151 | -0.245593 | -0.245810 |
| 19 | 0.086561 | -0.111580 | -0.198140 | -0.235508 |
| 20 | -0.635285 | -0.943742 | -0.308457 | -0.664684 |
| 21 | -0.854028 | -0.718496 | 0.135532 | -0.373458 |
| 22 | -0.416341 | -0.592435 | -0.176095 | -0.483522 |
| 23 | -0.417299 | -0.728764 | -0.311465 | -0.569071 |
| mean | -0.405388 | -0.586635 | -0.181247 | -0.341305 |
| std ddof=0 | 0.218686 | 0.220306 | 0.159900 | 0.159012 |
| left lower (count) | 11 of 12 | 12 of 12 | 11 of 12 | 12 of 12 |

## Swap-finding latency

Median over swaps. Censored swaps use L minus the swap decision. Share within 128 is the fraction of swaps with a later stay at most 128 decisions after the swap.

| controller | n | found | censored | median | share within 128 |
|---|---:|---:|---:|---:|---:|
| rand_dwell_s | 192 | 160 | 32 | 136.500000 | 0.380208 |
| oracle_fresh_s | 192 | 180 | 12 | 12.000000 | 0.937500 |
| prog_chan_s | 192 | 135 | 57 | 238.500000 | 0.223958 |
| prog_age_s | 192 | 174 | 18 | 125.500000 | 0.442708 |

| controller | seed | n | found | censored | median | share within 128 |
|---|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 12 | 16 | 14 | 2 | 152.500000 | 0.375000 |
| rand_dwell_s | 13 | 16 | 14 | 2 | 149.500000 | 0.312500 |
| rand_dwell_s | 14 | 16 | 11 | 5 | 163.000000 | 0.187500 |
| rand_dwell_s | 15 | 16 | 14 | 2 | 117.000000 | 0.500000 |
| rand_dwell_s | 16 | 16 | 12 | 4 | 123.500000 | 0.437500 |
| rand_dwell_s | 17 | 16 | 13 | 3 | 235.000000 | 0.250000 |
| rand_dwell_s | 18 | 16 | 14 | 2 | 151.500000 | 0.312500 |
| rand_dwell_s | 19 | 16 | 13 | 3 | 101.500000 | 0.562500 |
| rand_dwell_s | 20 | 16 | 14 | 2 | 120.500000 | 0.437500 |
| rand_dwell_s | 21 | 16 | 14 | 2 | 168.000000 | 0.437500 |
| rand_dwell_s | 22 | 16 | 14 | 2 | 147.500000 | 0.375000 |
| rand_dwell_s | 23 | 16 | 13 | 3 | 136.000000 | 0.375000 |
| oracle_fresh_s | 12 | 16 | 15 | 1 | 12.500000 | 0.937500 |
| oracle_fresh_s | 13 | 16 | 15 | 1 | 10.500000 | 0.937500 |
| oracle_fresh_s | 14 | 16 | 15 | 1 | 10.500000 | 0.937500 |
| oracle_fresh_s | 15 | 16 | 15 | 1 | 13.500000 | 0.937500 |
| oracle_fresh_s | 16 | 16 | 15 | 1 | 10.000000 | 0.937500 |
| oracle_fresh_s | 17 | 16 | 15 | 1 | 11.000000 | 0.937500 |
| oracle_fresh_s | 18 | 16 | 15 | 1 | 13.500000 | 0.937500 |
| oracle_fresh_s | 19 | 16 | 15 | 1 | 10.000000 | 0.937500 |
| oracle_fresh_s | 20 | 16 | 15 | 1 | 15.000000 | 0.937500 |
| oracle_fresh_s | 21 | 16 | 15 | 1 | 13.000000 | 0.937500 |
| oracle_fresh_s | 22 | 16 | 15 | 1 | 9.500000 | 0.937500 |
| oracle_fresh_s | 23 | 16 | 15 | 1 | 14.500000 | 0.937500 |
| prog_chan_s | 12 | 16 | 12 | 4 | 184.500000 | 0.312500 |
| prog_chan_s | 13 | 16 | 10 | 6 | 332.500000 | 0.187500 |
| prog_chan_s | 14 | 16 | 13 | 3 | 207.000000 | 0.187500 |
| prog_chan_s | 15 | 16 | 12 | 4 | 251.500000 | 0.250000 |
| prog_chan_s | 16 | 16 | 10 | 6 | 220.000000 | 0.187500 |
| prog_chan_s | 17 | 16 | 14 | 2 | 229.500000 | 0.312500 |
| prog_chan_s | 18 | 16 | 11 | 5 | 269.000000 | 0.125000 |
| prog_chan_s | 19 | 16 | 5 | 11 | 448.000000 | 0.125000 |
| prog_chan_s | 20 | 16 | 14 | 2 | 183.000000 | 0.375000 |
| prog_chan_s | 21 | 16 | 13 | 3 | 245.000000 | 0.187500 |
| prog_chan_s | 22 | 16 | 10 | 6 | 238.000000 | 0.187500 |
| prog_chan_s | 23 | 16 | 11 | 5 | 214.000000 | 0.250000 |
| prog_age_s | 12 | 16 | 14 | 2 | 98.500000 | 0.500000 |
| prog_age_s | 13 | 16 | 15 | 1 | 121.000000 | 0.500000 |
| prog_age_s | 14 | 16 | 15 | 1 | 118.000000 | 0.500000 |
| prog_age_s | 15 | 16 | 14 | 2 | 174.500000 | 0.250000 |
| prog_age_s | 16 | 16 | 14 | 2 | 101.000000 | 0.625000 |
| prog_age_s | 17 | 16 | 15 | 1 | 158.000000 | 0.375000 |
| prog_age_s | 18 | 16 | 14 | 2 | 149.000000 | 0.312500 |
| prog_age_s | 19 | 16 | 14 | 2 | 149.000000 | 0.375000 |
| prog_age_s | 20 | 16 | 15 | 1 | 84.000000 | 0.687500 |
| prog_age_s | 21 | 16 | 15 | 1 | 187.500000 | 0.250000 |
| prog_age_s | 22 | 16 | 14 | 2 | 202.500000 | 0.312500 |
| prog_age_s | 23 | 16 | 15 | 1 | 51.000000 | 0.625000 |

| controller | seed | decision | room | k | next dwell | latency | censored | within 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 12 | 128 | E30b | 0 | 380 | 252 | false | false |
| rand_dwell_s | 12 | 256 | M37a | 1 | 509 | 253 | false | false |
| rand_dwell_s | 12 | 384 | Ga | 2 | 555 | 171 | false | false |
| rand_dwell_s | 12 | 512 | E30b | 3 | 774 | 262 | false | false |
| rand_dwell_s | 12 | 640 | E30b | 4 | 774 | 134 | false | false |
| rand_dwell_s | 12 | 768 | Ga | 5 | 789 | 21 | false | true |
| rand_dwell_s | 12 | 896 | C1a | 6 | 1261 | 365 | false | false |
| rand_dwell_s | 12 | 1024 | C7b | 7 | 1034 | 10 | false | true |
| rand_dwell_s | 12 | 1152 | M7a | 8 | 1569 | 417 | false | false |
| rand_dwell_s | 12 | 1280 | M37b | 9 | 1345 | 65 | false | true |
| rand_dwell_s | 12 | 1408 | M7b | 10 | 1424 | 16 | false | true |
| rand_dwell_s | 12 | 1536 | Gb | 11 | 1806 | 270 | false | false |
| rand_dwell_s | 12 | 1664 | C1a | 12 |  | 384 | true | false |
| rand_dwell_s | 12 | 1792 | C1b | 13 | 1874 | 82 | false | true |
| rand_dwell_s | 12 | 1920 | E30b | 14 | 2038 | 118 | false | true |
| rand_dwell_s | 12 | 2048 | Ga | 15 |  | 0 | true | false |
| rand_dwell_s | 13 | 128 | E110b | 0 | 615 | 487 | false | false |
| rand_dwell_s | 13 | 256 | M37b | 1 | 294 | 38 | false | true |
| rand_dwell_s | 13 | 384 | C7a | 2 | 385 | 1 | false | true |
| rand_dwell_s | 13 | 512 | M37a | 3 | 711 | 199 | false | false |
| rand_dwell_s | 13 | 640 | E30a | 4 | 1087 | 447 | false | false |
| rand_dwell_s | 13 | 768 | C7a | 5 | 925 | 157 | false | false |
| rand_dwell_s | 13 | 896 | M7b | 6 | 1176 | 280 | false | false |
| rand_dwell_s | 13 | 1024 | C7a | 7 | 1031 | 7 | false | true |
| rand_dwell_s | 13 | 1152 | Ga | 8 | 1582 | 430 | false | false |
| rand_dwell_s | 13 | 1280 | C1b | 9 | 1889 | 609 | false | false |
| rand_dwell_s | 13 | 1408 | E30a | 10 | 1427 | 19 | false | true |
| rand_dwell_s | 13 | 1536 | M7b | 11 | 1720 | 184 | false | false |
| rand_dwell_s | 13 | 1664 | Ga | 12 | 1806 | 142 | false | false |
| rand_dwell_s | 13 | 1792 | M37b | 13 | 1822 | 30 | false | true |
| rand_dwell_s | 13 | 1920 | M7b | 14 |  | 128 | true | false |
| rand_dwell_s | 13 | 2048 | C1b | 15 |  | 0 | true | false |
| rand_dwell_s | 14 | 128 | C7b | 0 | 632 | 504 | false | false |
| rand_dwell_s | 14 | 256 | M37a | 1 | 395 | 139 | false | false |
| rand_dwell_s | 14 | 384 | C7a | 2 | 822 | 438 | false | false |
| rand_dwell_s | 14 | 512 | E30b | 3 | 651 | 139 | false | false |
| rand_dwell_s | 14 | 640 | C7a | 4 | 822 | 182 | false | false |
| rand_dwell_s | 14 | 768 | Gb | 5 | 1073 | 305 | false | false |
| rand_dwell_s | 14 | 896 | E30b | 6 | 905 | 9 | false | true |
| rand_dwell_s | 14 | 1024 | C1a | 7 | 1168 | 144 | false | false |
| rand_dwell_s | 14 | 1152 | Ga | 8 | 1204 | 52 | false | true |
| rand_dwell_s | 14 | 1280 | E30b | 9 | 1369 | 89 | false | true |
| rand_dwell_s | 14 | 1408 | C7a | 10 |  | 640 | true | false |
| rand_dwell_s | 14 | 1536 | E110a | 11 | 1814 | 278 | false | false |
| rand_dwell_s | 14 | 1664 | C7a | 12 |  | 384 | true | false |
| rand_dwell_s | 14 | 1792 | E30b | 13 |  | 256 | true | false |
| rand_dwell_s | 14 | 1920 | C1a | 14 |  | 128 | true | false |
| rand_dwell_s | 14 | 2048 | C7a | 15 |  | 0 | true | false |
| rand_dwell_s | 15 | 128 | E110a | 0 | 553 | 425 | false | false |
| rand_dwell_s | 15 | 256 | E110a | 1 | 553 | 297 | false | false |
| rand_dwell_s | 15 | 384 | C7b | 2 | 385 | 1 | false | true |
| rand_dwell_s | 15 | 512 | C7b | 3 | 587 | 75 | false | true |
| rand_dwell_s | 15 | 640 | Ga | 4 | 671 | 31 | false | true |
| rand_dwell_s | 15 | 768 | E110a | 5 | 1289 | 521 | false | false |
| rand_dwell_s | 15 | 896 | C7a | 6 | 1372 | 476 | false | false |
| rand_dwell_s | 15 | 1024 | Ga | 7 | 1091 | 67 | false | true |
| rand_dwell_s | 15 | 1152 | M37b | 8 | 1674 | 522 | false | false |
| rand_dwell_s | 15 | 1280 | M7a | 9 | 1307 | 27 | false | true |
| rand_dwell_s | 15 | 1408 | E110b | 10 | 1524 | 116 | false | true |
| rand_dwell_s | 15 | 1536 | M7a | 11 | 1567 | 31 | false | true |
| rand_dwell_s | 15 | 1664 | C7a | 12 |  | 384 | true | false |
| rand_dwell_s | 15 | 1792 | M7b | 13 | 2026 | 234 | false | false |
| rand_dwell_s | 15 | 1920 | C1b | 14 | 2038 | 118 | false | true |
| rand_dwell_s | 15 | 2048 | C7a | 15 |  | 0 | true | false |
| rand_dwell_s | 16 | 128 | C7a | 0 | 302 | 174 | false | false |
| rand_dwell_s | 16 | 256 | E30b | 1 | 530 | 274 | false | false |
| rand_dwell_s | 16 | 384 | Ga | 2 | 456 | 72 | false | true |
| rand_dwell_s | 16 | 512 | M37b | 3 | 625 | 113 | false | true |
| rand_dwell_s | 16 | 640 | C1a | 4 | 1217 | 577 | false | false |
| rand_dwell_s | 16 | 768 | Ga | 5 | 902 | 134 | false | false |
| rand_dwell_s | 16 | 896 | E30a | 6 | 975 | 79 | false | true |
| rand_dwell_s | 16 | 1024 | C7b | 7 | 1290 | 266 | false | false |
| rand_dwell_s | 16 | 1152 | C1a | 8 | 1217 | 65 | false | true |
| rand_dwell_s | 16 | 1280 | M7b | 9 |  | 768 | true | false |
| rand_dwell_s | 16 | 1408 | M7a | 10 | 1460 | 52 | false | true |
| rand_dwell_s | 16 | 1536 | E30a | 11 | 1569 | 33 | false | true |
| rand_dwell_s | 16 | 1664 | M37a | 12 |  | 384 | true | false |
| rand_dwell_s | 16 | 1792 | M7b | 13 |  | 256 | true | false |
| rand_dwell_s | 16 | 1920 | E30a | 14 | 2029 | 109 | false | true |
| rand_dwell_s | 16 | 2048 | C1a | 15 |  | 0 | true | false |
| rand_dwell_s | 17 | 128 | C7b | 0 | 186 | 58 | false | true |
| rand_dwell_s | 17 | 256 | E110a | 1 | 461 | 205 | false | false |
| rand_dwell_s | 17 | 384 | M7b | 2 | 623 | 239 | false | false |
| rand_dwell_s | 17 | 512 | M7a | 3 | 632 | 120 | false | true |
| rand_dwell_s | 17 | 640 | E30a | 4 | 903 | 263 | false | false |
| rand_dwell_s | 17 | 768 | Ga | 5 | 826 | 58 | false | true |
| rand_dwell_s | 17 | 896 | M37a | 6 | 1255 | 359 | false | false |
| rand_dwell_s | 17 | 1024 | M37a | 7 | 1255 | 231 | false | false |
| rand_dwell_s | 17 | 1152 | E110b | 8 | 1412 | 260 | false | false |
| rand_dwell_s | 17 | 1280 | C7b | 9 | 1714 | 434 | false | false |
| rand_dwell_s | 17 | 1408 | Gb | 10 | 1957 | 549 | false | false |
| rand_dwell_s | 17 | 1536 | C1a | 11 | 1577 | 41 | false | true |
| rand_dwell_s | 17 | 1664 | Gb | 12 | 1957 | 293 | false | false |
| rand_dwell_s | 17 | 1792 | E30a | 13 |  | 256 | true | false |
| rand_dwell_s | 17 | 1920 | E30a | 14 |  | 128 | true | false |
| rand_dwell_s | 17 | 2048 | E110b | 15 |  | 0 | true | false |
| rand_dwell_s | 18 | 128 | C1b | 0 | 919 | 791 | false | false |
| rand_dwell_s | 18 | 256 | M7b | 1 | 371 | 115 | false | true |
| rand_dwell_s | 18 | 384 | E30a | 2 | 661 | 277 | false | false |
| rand_dwell_s | 18 | 512 | E110a | 3 | 619 | 107 | false | true |
| rand_dwell_s | 18 | 640 | M7b | 4 | 809 | 169 | false | false |
| rand_dwell_s | 18 | 768 | Ga | 5 | 902 | 134 | false | false |
| rand_dwell_s | 18 | 896 | C1b | 6 | 919 | 23 | false | true |
| rand_dwell_s | 18 | 1024 | Gb | 7 | 1716 | 692 | false | false |
| rand_dwell_s | 18 | 1152 | Gb | 8 | 1716 | 564 | false | false |
| rand_dwell_s | 18 | 1280 | E110a | 9 | 1897 | 617 | false | false |
| rand_dwell_s | 18 | 1408 | C1a | 10 | 1731 | 323 | false | false |
| rand_dwell_s | 18 | 1536 | E30a | 11 | 1595 | 59 | false | true |
| rand_dwell_s | 18 | 1664 | E110b | 12 | 1883 | 219 | false | false |
| rand_dwell_s | 18 | 1792 | E30b | 13 | 1854 | 62 | false | true |
| rand_dwell_s | 18 | 1920 | C7b | 14 |  | 128 | true | false |
| rand_dwell_s | 18 | 2048 | C7a | 15 |  | 0 | true | false |
| rand_dwell_s | 19 | 128 | C7a | 0 | 157 | 29 | false | true |
| rand_dwell_s | 19 | 256 | C7b | 1 | 614 | 358 | false | false |
| rand_dwell_s | 19 | 384 | C7b | 2 | 614 | 230 | false | false |
| rand_dwell_s | 19 | 512 | C7b | 3 | 614 | 102 | false | true |
| rand_dwell_s | 19 | 640 | C1b | 4 | 736 | 96 | false | true |
| rand_dwell_s | 19 | 768 | M7a | 5 | 851 | 83 | false | true |
| rand_dwell_s | 19 | 896 | M37b | 6 | 1012 | 116 | false | true |
| rand_dwell_s | 19 | 1024 | M37a | 7 | 1098 | 74 | false | true |
| rand_dwell_s | 19 | 1152 | E110b | 8 | 1740 | 588 | false | false |
| rand_dwell_s | 19 | 1280 | E110a | 9 | 1441 | 161 | false | false |
| rand_dwell_s | 19 | 1408 | C7a | 10 | 1463 | 55 | false | true |
| rand_dwell_s | 19 | 1536 | C7b | 11 |  | 512 | true | false |
| rand_dwell_s | 19 | 1664 | M7a | 12 | 1765 | 101 | false | true |
| rand_dwell_s | 19 | 1792 | M7b | 13 | 1804 | 12 | false | true |
| rand_dwell_s | 19 | 1920 | C7a | 14 |  | 128 | true | false |
| rand_dwell_s | 19 | 2048 | Ga | 15 |  | 0 | true | false |
| rand_dwell_s | 20 | 128 | M37a | 0 | 344 | 216 | false | false |
| rand_dwell_s | 20 | 256 | M37b | 1 | 257 | 1 | false | true |
| rand_dwell_s | 20 | 384 | E30a | 2 | 442 | 58 | false | true |
| rand_dwell_s | 20 | 512 | M7b | 3 | 513 | 1 | false | true |
| rand_dwell_s | 20 | 640 | C1a | 4 | 1265 | 625 | false | false |
| rand_dwell_s | 20 | 768 | E30a | 5 | 1772 | 1004 | false | false |
| rand_dwell_s | 20 | 896 | C1b | 6 | 1062 | 166 | false | false |
| rand_dwell_s | 20 | 1024 | Ga | 7 | 1181 | 157 | false | false |
| rand_dwell_s | 20 | 1152 | C1a | 8 | 1265 | 113 | false | true |
| rand_dwell_s | 20 | 1280 | M37b | 9 | 1642 | 362 | false | false |
| rand_dwell_s | 20 | 1408 | E30a | 10 | 1772 | 364 | false | false |
| rand_dwell_s | 20 | 1536 | M7b | 11 | 1570 | 34 | false | true |
| rand_dwell_s | 20 | 1664 | Gb | 12 | 1668 | 4 | false | true |
| rand_dwell_s | 20 | 1792 | M7a | 13 | 1817 | 25 | false | true |
| rand_dwell_s | 20 | 1920 | Gb | 14 |  | 128 | true | false |
| rand_dwell_s | 20 | 2048 | C7b | 15 |  | 0 | true | false |
| rand_dwell_s | 21 | 128 | C1b | 0 | 150 | 22 | false | true |
| rand_dwell_s | 21 | 256 | E110b | 1 | 617 | 361 | false | false |
| rand_dwell_s | 21 | 384 | E30a | 2 | 388 | 4 | false | true |
| rand_dwell_s | 21 | 512 | M7a | 3 | 1111 | 599 | false | false |
| rand_dwell_s | 21 | 640 | M7a | 4 | 1111 | 471 | false | false |
| rand_dwell_s | 21 | 768 | M7b | 5 | 787 | 19 | false | true |
| rand_dwell_s | 21 | 896 | M7a | 6 | 1111 | 215 | false | false |
| rand_dwell_s | 21 | 1024 | E110b | 7 | 1673 | 649 | false | false |
| rand_dwell_s | 21 | 1152 | Ga | 8 | 1491 | 339 | false | false |
| rand_dwell_s | 21 | 1280 | M7b | 9 | 1401 | 121 | false | true |
| rand_dwell_s | 21 | 1408 | C7a | 10 | 1642 | 234 | false | false |
| rand_dwell_s | 21 | 1536 | C1b | 11 | 1652 | 116 | false | true |
| rand_dwell_s | 21 | 1664 | E30b | 12 | 1730 | 66 | false | true |
| rand_dwell_s | 21 | 1792 | E110a | 13 |  | 256 | true | false |
| rand_dwell_s | 21 | 1920 | Gb | 14 | 1927 | 7 | false | true |
| rand_dwell_s | 21 | 2048 | E110a | 15 |  | 0 | true | false |
| rand_dwell_s | 22 | 128 | C7a | 0 | 248 | 120 | false | true |
| rand_dwell_s | 22 | 256 | M7b | 1 | 1761 | 1505 | false | false |
| rand_dwell_s | 22 | 384 | M7b | 2 | 1761 | 1377 | false | false |
| rand_dwell_s | 22 | 512 | Gb | 3 | 513 | 1 | false | true |
| rand_dwell_s | 22 | 640 | Ga | 4 | 1188 | 548 | false | false |
| rand_dwell_s | 22 | 768 | E110b | 5 | 1425 | 657 | false | false |
| rand_dwell_s | 22 | 896 | E30b | 6 | 969 | 73 | false | true |
| rand_dwell_s | 22 | 1024 | M7a | 7 | 1154 | 130 | false | false |
| rand_dwell_s | 22 | 1152 | Gb | 8 | 1324 | 172 | false | false |
| rand_dwell_s | 22 | 1280 | M7b | 9 | 1761 | 481 | false | false |
| rand_dwell_s | 22 | 1408 | Ga | 10 |  | 640 | true | false |
| rand_dwell_s | 22 | 1536 | E110a | 11 | 1701 | 165 | false | false |
| rand_dwell_s | 22 | 1664 | C1a | 12 | 1719 | 55 | false | true |
| rand_dwell_s | 22 | 1792 | M7b | 13 | 1885 | 93 | false | true |
| rand_dwell_s | 22 | 1920 | E30a | 14 | 1945 | 25 | false | true |
| rand_dwell_s | 22 | 2048 | Gb | 15 |  | 0 | true | false |
| rand_dwell_s | 23 | 128 | M7a | 0 | 224 | 96 | false | true |
| rand_dwell_s | 23 | 256 | M37a | 1 | 299 | 43 | false | true |
| rand_dwell_s | 23 | 384 | Ga | 2 | 943 | 559 | false | false |
| rand_dwell_s | 23 | 512 | C1a | 3 | 651 | 139 | false | false |
| rand_dwell_s | 23 | 640 | M37b | 4 | 920 | 280 | false | false |
| rand_dwell_s | 23 | 768 | C7b | 5 | 1412 | 644 | false | false |
| rand_dwell_s | 23 | 896 | M37a | 6 | 1013 | 117 | false | true |
| rand_dwell_s | 23 | 1024 | M7a | 7 | 1718 | 694 | false | false |
| rand_dwell_s | 23 | 1152 | E110b | 8 | 1285 | 133 | false | false |
| rand_dwell_s | 23 | 1280 | M37a | 9 | 1333 | 53 | false | true |
| rand_dwell_s | 23 | 1408 | M37b | 10 | 1696 | 288 | false | false |
| rand_dwell_s | 23 | 1536 | E30b | 11 | 1570 | 34 | false | true |
| rand_dwell_s | 23 | 1664 | Gb | 12 |  | 384 | true | false |
| rand_dwell_s | 23 | 1792 | Gb | 13 |  | 256 | true | false |
| rand_dwell_s | 23 | 1920 | E30a | 14 | 2040 | 120 | false | true |
| rand_dwell_s | 23 | 2048 | M7a | 15 |  | 0 | true | false |
| oracle_fresh_s | 12 | 128 | E30b | 0 | 196 | 68 | false | true |
| oracle_fresh_s | 12 | 256 | M37a | 1 | 269 | 13 | false | true |
| oracle_fresh_s | 12 | 384 | Ga | 2 | 391 | 7 | false | true |
| oracle_fresh_s | 12 | 512 | E30b | 3 | 530 | 18 | false | true |
| oracle_fresh_s | 12 | 640 | E30b | 4 | 656 | 16 | false | true |
| oracle_fresh_s | 12 | 768 | Ga | 5 | 777 | 9 | false | true |
| oracle_fresh_s | 12 | 896 | C1a | 6 | 919 | 23 | false | true |
| oracle_fresh_s | 12 | 1024 | C7b | 7 | 1032 | 8 | false | true |
| oracle_fresh_s | 12 | 1152 | M7a | 8 | 1165 | 13 | false | true |
| oracle_fresh_s | 12 | 1280 | M37b | 9 | 1289 | 9 | false | true |
| oracle_fresh_s | 12 | 1408 | M7b | 10 | 1414 | 6 | false | true |
| oracle_fresh_s | 12 | 1536 | Gb | 11 | 1560 | 24 | false | true |
| oracle_fresh_s | 12 | 1664 | C1a | 12 | 1665 | 1 | false | true |
| oracle_fresh_s | 12 | 1792 | C1b | 13 | 1804 | 12 | false | true |
| oracle_fresh_s | 12 | 1920 | E30b | 14 | 1936 | 16 | false | true |
| oracle_fresh_s | 12 | 2048 | Ga | 15 |  | 0 | true | false |
| oracle_fresh_s | 13 | 128 | E110b | 0 | 177 | 49 | false | true |
| oracle_fresh_s | 13 | 256 | M37b | 1 | 257 | 1 | false | true |
| oracle_fresh_s | 13 | 384 | C7a | 2 | 397 | 13 | false | true |
| oracle_fresh_s | 13 | 512 | M37a | 3 | 531 | 19 | false | true |
| oracle_fresh_s | 13 | 640 | E30a | 4 | 653 | 13 | false | true |
| oracle_fresh_s | 13 | 768 | C7a | 5 | 771 | 3 | false | true |
| oracle_fresh_s | 13 | 896 | M7b | 6 | 898 | 2 | false | true |
| oracle_fresh_s | 13 | 1024 | C7a | 7 | 1043 | 19 | false | true |
| oracle_fresh_s | 13 | 1152 | Ga | 8 | 1168 | 16 | false | true |
| oracle_fresh_s | 13 | 1280 | C1b | 9 | 1295 | 15 | false | true |
| oracle_fresh_s | 13 | 1408 | E30a | 10 | 1411 | 3 | false | true |
| oracle_fresh_s | 13 | 1536 | M7b | 11 | 1544 | 8 | false | true |
| oracle_fresh_s | 13 | 1664 | Ga | 12 | 1682 | 18 | false | true |
| oracle_fresh_s | 13 | 1792 | M37b | 13 | 1800 | 8 | false | true |
| oracle_fresh_s | 13 | 1920 | M7b | 14 | 1926 | 6 | false | true |
| oracle_fresh_s | 13 | 2048 | C1b | 15 |  | 0 | true | false |
| oracle_fresh_s | 14 | 128 | C7b | 0 | 134 | 6 | false | true |
| oracle_fresh_s | 14 | 256 | M37a | 1 | 269 | 13 | false | true |
| oracle_fresh_s | 14 | 384 | C7a | 2 | 404 | 20 | false | true |
| oracle_fresh_s | 14 | 512 | E30b | 3 | 523 | 11 | false | true |
| oracle_fresh_s | 14 | 640 | C7a | 4 | 656 | 16 | false | true |
| oracle_fresh_s | 14 | 768 | Gb | 5 | 785 | 17 | false | true |
| oracle_fresh_s | 14 | 896 | E30b | 6 | 909 | 13 | false | true |
| oracle_fresh_s | 14 | 1024 | C1a | 7 | 1030 | 6 | false | true |
| oracle_fresh_s | 14 | 1152 | Ga | 8 | 1172 | 20 | false | true |
| oracle_fresh_s | 14 | 1280 | E30b | 9 | 1285 | 5 | false | true |
| oracle_fresh_s | 14 | 1408 | C7a | 10 | 1418 | 10 | false | true |
| oracle_fresh_s | 14 | 1536 | E110a | 11 | 1560 | 24 | false | true |
| oracle_fresh_s | 14 | 1664 | C7a | 12 | 1668 | 4 | false | true |
| oracle_fresh_s | 14 | 1792 | E30b | 13 | 1797 | 5 | false | true |
| oracle_fresh_s | 14 | 1920 | C1a | 14 | 1926 | 6 | false | true |
| oracle_fresh_s | 14 | 2048 | C7a | 15 |  | 0 | true | false |
| oracle_fresh_s | 15 | 128 | E110a | 0 | 209 | 81 | false | true |
| oracle_fresh_s | 15 | 256 | E110a | 1 | 267 | 11 | false | true |
| oracle_fresh_s | 15 | 384 | C7b | 2 | 405 | 21 | false | true |
| oracle_fresh_s | 15 | 512 | C7b | 3 | 533 | 21 | false | true |
| oracle_fresh_s | 15 | 640 | Ga | 4 | 655 | 15 | false | true |
| oracle_fresh_s | 15 | 768 | E110a | 5 | 777 | 9 | false | true |
| oracle_fresh_s | 15 | 896 | C7a | 6 | 902 | 6 | false | true |
| oracle_fresh_s | 15 | 1024 | Ga | 7 | 1037 | 13 | false | true |
| oracle_fresh_s | 15 | 1152 | M37b | 8 | 1170 | 18 | false | true |
| oracle_fresh_s | 15 | 1280 | M7a | 9 | 1285 | 5 | false | true |
| oracle_fresh_s | 15 | 1408 | E110b | 10 | 1422 | 14 | false | true |
| oracle_fresh_s | 15 | 1536 | M7a | 11 | 1567 | 31 | false | true |
| oracle_fresh_s | 15 | 1664 | C7a | 12 | 1684 | 20 | false | true |
| oracle_fresh_s | 15 | 1792 | M7b | 13 | 1802 | 10 | false | true |
| oracle_fresh_s | 15 | 1920 | C1b | 14 | 1932 | 12 | false | true |
| oracle_fresh_s | 15 | 2048 | C7a | 15 |  | 0 | true | false |
| oracle_fresh_s | 16 | 128 | C7a | 0 | 250 | 122 | false | true |
| oracle_fresh_s | 16 | 256 | E30b | 1 | 260 | 4 | false | true |
| oracle_fresh_s | 16 | 384 | Ga | 2 | 385 | 1 | false | true |
| oracle_fresh_s | 16 | 512 | M37b | 3 | 527 | 15 | false | true |
| oracle_fresh_s | 16 | 640 | C1a | 4 | 649 | 9 | false | true |
| oracle_fresh_s | 16 | 768 | Ga | 5 | 774 | 6 | false | true |
| oracle_fresh_s | 16 | 896 | E30a | 6 | 905 | 9 | false | true |
| oracle_fresh_s | 16 | 1024 | C7b | 7 | 1036 | 12 | false | true |
| oracle_fresh_s | 16 | 1152 | C1a | 8 | 1171 | 19 | false | true |
| oracle_fresh_s | 16 | 1280 | M7b | 9 | 1287 | 7 | false | true |
| oracle_fresh_s | 16 | 1408 | M7a | 10 | 1424 | 16 | false | true |
| oracle_fresh_s | 16 | 1536 | E30a | 11 | 1559 | 23 | false | true |
| oracle_fresh_s | 16 | 1664 | M37a | 12 | 1681 | 17 | false | true |
| oracle_fresh_s | 16 | 1792 | M7b | 13 | 1803 | 11 | false | true |
| oracle_fresh_s | 16 | 1920 | E30a | 14 | 1929 | 9 | false | true |
| oracle_fresh_s | 16 | 2048 | C1a | 15 |  | 0 | true | false |
| oracle_fresh_s | 17 | 128 | C7b | 0 | 180 | 52 | false | true |
| oracle_fresh_s | 17 | 256 | E110a | 1 | 263 | 7 | false | true |
| oracle_fresh_s | 17 | 384 | M7b | 2 | 395 | 11 | false | true |
| oracle_fresh_s | 17 | 512 | M7a | 3 | 532 | 20 | false | true |
| oracle_fresh_s | 17 | 640 | E30a | 4 | 649 | 9 | false | true |
| oracle_fresh_s | 17 | 768 | Ga | 5 | 778 | 10 | false | true |
| oracle_fresh_s | 17 | 896 | M37a | 6 | 919 | 23 | false | true |
| oracle_fresh_s | 17 | 1024 | M37a | 7 | 1033 | 9 | false | true |
| oracle_fresh_s | 17 | 1152 | E110b | 8 | 1170 | 18 | false | true |
| oracle_fresh_s | 17 | 1280 | C7b | 9 | 1298 | 18 | false | true |
| oracle_fresh_s | 17 | 1408 | Gb | 10 | 1419 | 11 | false | true |
| oracle_fresh_s | 17 | 1536 | C1a | 11 | 1543 | 7 | false | true |
| oracle_fresh_s | 17 | 1664 | Gb | 12 | 1679 | 15 | false | true |
| oracle_fresh_s | 17 | 1792 | E30a | 13 | 1801 | 9 | false | true |
| oracle_fresh_s | 17 | 1920 | E30a | 14 | 1935 | 15 | false | true |
| oracle_fresh_s | 17 | 2048 | E110b | 15 |  | 0 | true | false |
| oracle_fresh_s | 18 | 128 | C1b | 0 | 215 | 87 | false | true |
| oracle_fresh_s | 18 | 256 | M7b | 1 | 257 | 1 | false | true |
| oracle_fresh_s | 18 | 384 | E30a | 2 | 405 | 21 | false | true |
| oracle_fresh_s | 18 | 512 | E110a | 3 | 533 | 21 | false | true |
| oracle_fresh_s | 18 | 640 | M7b | 4 | 649 | 9 | false | true |
| oracle_fresh_s | 18 | 768 | Ga | 5 | 784 | 16 | false | true |
| oracle_fresh_s | 18 | 896 | C1b | 6 | 907 | 11 | false | true |
| oracle_fresh_s | 18 | 1024 | Gb | 7 | 1034 | 10 | false | true |
| oracle_fresh_s | 18 | 1152 | Gb | 8 | 1174 | 22 | false | true |
| oracle_fresh_s | 18 | 1280 | E110a | 9 | 1293 | 13 | false | true |
| oracle_fresh_s | 18 | 1408 | C1a | 10 | 1417 | 9 | false | true |
| oracle_fresh_s | 18 | 1536 | E30a | 11 | 1561 | 25 | false | true |
| oracle_fresh_s | 18 | 1664 | E110b | 12 | 1683 | 19 | false | true |
| oracle_fresh_s | 18 | 1792 | E30b | 13 | 1806 | 14 | false | true |
| oracle_fresh_s | 18 | 1920 | C7b | 14 | 1927 | 7 | false | true |
| oracle_fresh_s | 18 | 2048 | C7a | 15 |  | 0 | true | false |
| oracle_fresh_s | 19 | 128 | C7a | 0 | 213 | 85 | false | true |
| oracle_fresh_s | 19 | 256 | C7b | 1 | 266 | 10 | false | true |
| oracle_fresh_s | 19 | 384 | C7b | 2 | 394 | 10 | false | true |
| oracle_fresh_s | 19 | 512 | C7b | 3 | 513 | 1 | false | true |
| oracle_fresh_s | 19 | 640 | C1b | 4 | 654 | 14 | false | true |
| oracle_fresh_s | 19 | 768 | M7a | 5 | 771 | 3 | false | true |
| oracle_fresh_s | 19 | 896 | M37b | 6 | 912 | 16 | false | true |
| oracle_fresh_s | 19 | 1024 | M37a | 7 | 1044 | 20 | false | true |
| oracle_fresh_s | 19 | 1152 | E110b | 8 | 1168 | 16 | false | true |
| oracle_fresh_s | 19 | 1280 | E110a | 9 | 1283 | 3 | false | true |
| oracle_fresh_s | 19 | 1408 | C7a | 10 | 1417 | 9 | false | true |
| oracle_fresh_s | 19 | 1536 | C7b | 11 | 1544 | 8 | false | true |
| oracle_fresh_s | 19 | 1664 | M7a | 12 | 1679 | 15 | false | true |
| oracle_fresh_s | 19 | 1792 | M7b | 13 | 1810 | 18 | false | true |
| oracle_fresh_s | 19 | 1920 | C7a | 14 | 1921 | 1 | false | true |
| oracle_fresh_s | 19 | 2048 | Ga | 15 |  | 0 | true | false |
| oracle_fresh_s | 20 | 128 | M37a | 0 | 184 | 56 | false | true |
| oracle_fresh_s | 20 | 256 | M37b | 1 | 266 | 10 | false | true |
| oracle_fresh_s | 20 | 384 | E30a | 2 | 406 | 22 | false | true |
| oracle_fresh_s | 20 | 512 | M7b | 3 | 530 | 18 | false | true |
| oracle_fresh_s | 20 | 640 | C1a | 4 | 649 | 9 | false | true |
| oracle_fresh_s | 20 | 768 | E30a | 5 | 784 | 16 | false | true |
| oracle_fresh_s | 20 | 896 | C1b | 6 | 918 | 22 | false | true |
| oracle_fresh_s | 20 | 1024 | Ga | 7 | 1053 | 29 | false | true |
| oracle_fresh_s | 20 | 1152 | C1a | 8 | 1167 | 15 | false | true |
| oracle_fresh_s | 20 | 1280 | M37b | 9 | 1288 | 8 | false | true |
| oracle_fresh_s | 20 | 1408 | E30a | 10 | 1416 | 8 | false | true |
| oracle_fresh_s | 20 | 1536 | M7b | 11 | 1558 | 22 | false | true |
| oracle_fresh_s | 20 | 1664 | Gb | 12 | 1674 | 10 | false | true |
| oracle_fresh_s | 20 | 1792 | M7a | 13 | 1807 | 15 | false | true |
| oracle_fresh_s | 20 | 1920 | Gb | 14 | 1921 | 1 | false | true |
| oracle_fresh_s | 20 | 2048 | C7b | 15 |  | 0 | true | false |
| oracle_fresh_s | 21 | 128 | C1b | 0 | 228 | 100 | false | true |
| oracle_fresh_s | 21 | 256 | E110b | 1 | 273 | 17 | false | true |
| oracle_fresh_s | 21 | 384 | E30a | 2 | 394 | 10 | false | true |
| oracle_fresh_s | 21 | 512 | M7a | 3 | 539 | 27 | false | true |
| oracle_fresh_s | 21 | 640 | M7a | 4 | 659 | 19 | false | true |
| oracle_fresh_s | 21 | 768 | M7b | 5 | 773 | 5 | false | true |
| oracle_fresh_s | 21 | 896 | M7a | 6 | 905 | 9 | false | true |
| oracle_fresh_s | 21 | 1024 | E110b | 7 | 1049 | 25 | false | true |
| oracle_fresh_s | 21 | 1152 | Ga | 8 | 1179 | 27 | false | true |
| oracle_fresh_s | 21 | 1280 | M7b | 9 | 1289 | 9 | false | true |
| oracle_fresh_s | 21 | 1408 | C7a | 10 | 1426 | 18 | false | true |
| oracle_fresh_s | 21 | 1536 | C1b | 11 | 1544 | 8 | false | true |
| oracle_fresh_s | 21 | 1664 | E30b | 12 | 1674 | 10 | false | true |
| oracle_fresh_s | 21 | 1792 | E110a | 13 | 1808 | 16 | false | true |
| oracle_fresh_s | 21 | 1920 | Gb | 14 | 1921 | 1 | false | true |
| oracle_fresh_s | 21 | 2048 | E110a | 15 |  | 0 | true | false |
| oracle_fresh_s | 22 | 128 | C7a | 0 | 134 | 6 | false | true |
| oracle_fresh_s | 22 | 256 | M7b | 1 | 271 | 15 | false | true |
| oracle_fresh_s | 22 | 384 | M7b | 2 | 405 | 21 | false | true |
| oracle_fresh_s | 22 | 512 | Gb | 3 | 534 | 22 | false | true |
| oracle_fresh_s | 22 | 640 | Ga | 4 | 648 | 8 | false | true |
| oracle_fresh_s | 22 | 768 | E110b | 5 | 779 | 11 | false | true |
| oracle_fresh_s | 22 | 896 | E30b | 6 | 905 | 9 | false | true |
| oracle_fresh_s | 22 | 1024 | M7a | 7 | 1034 | 10 | false | true |
| oracle_fresh_s | 22 | 1152 | Gb | 8 | 1162 | 10 | false | true |
| oracle_fresh_s | 22 | 1280 | M7b | 9 | 1289 | 9 | false | true |
| oracle_fresh_s | 22 | 1408 | Ga | 10 | 1410 | 2 | false | true |
| oracle_fresh_s | 22 | 1536 | E110a | 11 | 1559 | 23 | false | true |
| oracle_fresh_s | 22 | 1664 | C1a | 12 | 1673 | 9 | false | true |
| oracle_fresh_s | 22 | 1792 | M7b | 13 | 1805 | 13 | false | true |
| oracle_fresh_s | 22 | 1920 | E30a | 14 | 1929 | 9 | false | true |
| oracle_fresh_s | 22 | 2048 | Gb | 15 |  | 0 | true | false |
| oracle_fresh_s | 23 | 128 | M7a | 0 | 148 | 20 | false | true |
| oracle_fresh_s | 23 | 256 | M37a | 1 | 271 | 15 | false | true |
| oracle_fresh_s | 23 | 384 | Ga | 2 | 387 | 3 | false | true |
| oracle_fresh_s | 23 | 512 | C1a | 3 | 525 | 13 | false | true |
| oracle_fresh_s | 23 | 640 | M37b | 4 | 658 | 18 | false | true |
| oracle_fresh_s | 23 | 768 | C7b | 5 | 784 | 16 | false | true |
| oracle_fresh_s | 23 | 896 | M37a | 6 | 911 | 15 | false | true |
| oracle_fresh_s | 23 | 1024 | M7a | 7 | 1030 | 6 | false | true |
| oracle_fresh_s | 23 | 1152 | E110b | 8 | 1177 | 25 | false | true |
| oracle_fresh_s | 23 | 1280 | M37a | 9 | 1295 | 15 | false | true |
| oracle_fresh_s | 23 | 1408 | M37b | 10 | 1416 | 8 | false | true |
| oracle_fresh_s | 23 | 1536 | E30b | 11 | 1548 | 12 | false | true |
| oracle_fresh_s | 23 | 1664 | Gb | 12 | 1685 | 21 | false | true |
| oracle_fresh_s | 23 | 1792 | Gb | 13 | 1803 | 11 | false | true |
| oracle_fresh_s | 23 | 1920 | E30a | 14 | 1934 | 14 | false | true |
| oracle_fresh_s | 23 | 2048 | M7a | 15 |  | 0 | true | false |
| prog_chan_s | 12 | 128 | E30b | 0 | 314 | 186 | false | false |
| prog_chan_s | 12 | 256 | M37a | 1 | 285 | 29 | false | true |
| prog_chan_s | 12 | 384 | Ga | 2 | 457 | 73 | false | true |
| prog_chan_s | 12 | 512 | E30b | 3 | 1338 | 826 | false | false |
| prog_chan_s | 12 | 640 | E30b | 4 | 1338 | 698 | false | false |
| prog_chan_s | 12 | 768 | Ga | 5 | 769 | 1 | false | true |
| prog_chan_s | 12 | 896 | C1a | 6 | 1267 | 371 | false | false |
| prog_chan_s | 12 | 1024 | C7b | 7 | 1878 | 854 | false | false |
| prog_chan_s | 12 | 1152 | M7a | 8 |  | 896 | true | false |
| prog_chan_s | 12 | 1280 | M37b | 9 | 1321 | 41 | false | true |
| prog_chan_s | 12 | 1408 | M7b | 10 | 1760 | 352 | false | false |
| prog_chan_s | 12 | 1536 | Gb | 11 | 1537 | 1 | false | true |
| prog_chan_s | 12 | 1664 | C1a | 12 | 1847 | 183 | false | false |
| prog_chan_s | 12 | 1792 | C1b | 13 |  | 256 | true | false |
| prog_chan_s | 12 | 1920 | E30b | 14 |  | 128 | true | false |
| prog_chan_s | 12 | 2048 | Ga | 15 |  | 0 | true | false |
| prog_chan_s | 13 | 128 | E110b | 0 | 353 | 225 | false | false |
| prog_chan_s | 13 | 256 | M37b | 1 | 492 | 236 | false | false |
| prog_chan_s | 13 | 384 | C7a | 2 | 1341 | 957 | false | false |
| prog_chan_s | 13 | 512 | M37a | 3 | 1327 | 815 | false | false |
| prog_chan_s | 13 | 640 | E30a | 4 |  | 1408 | true | false |
| prog_chan_s | 13 | 768 | C7a | 5 | 1341 | 573 | false | false |
| prog_chan_s | 13 | 896 | M7b | 6 | 932 | 36 | false | true |
| prog_chan_s | 13 | 1024 | C7a | 7 | 1341 | 317 | false | false |
| prog_chan_s | 13 | 1152 | Ga | 8 | 1500 | 348 | false | false |
| prog_chan_s | 13 | 1280 | C1b | 9 |  | 768 | true | false |
| prog_chan_s | 13 | 1408 | E30a | 10 |  | 640 | true | false |
| prog_chan_s | 13 | 1536 | M7b | 11 | 1618 | 82 | false | true |
| prog_chan_s | 13 | 1664 | Ga | 12 |  | 384 | true | false |
| prog_chan_s | 13 | 1792 | M37b | 13 | 1793 | 1 | false | true |
| prog_chan_s | 13 | 1920 | M7b | 14 |  | 128 | true | false |
| prog_chan_s | 13 | 2048 | C1b | 15 |  | 0 | true | false |
| prog_chan_s | 14 | 128 | C7b | 0 | 234 | 106 | false | true |
| prog_chan_s | 14 | 256 | M37a | 1 | 427 | 171 | false | false |
| prog_chan_s | 14 | 384 | C7a | 2 | 1250 | 866 | false | false |
| prog_chan_s | 14 | 512 | E30b | 3 | 765 | 253 | false | false |
| prog_chan_s | 14 | 640 | C7a | 4 | 1250 | 610 | false | false |
| prog_chan_s | 14 | 768 | Gb | 5 | 817 | 49 | false | true |
| prog_chan_s | 14 | 896 | E30b | 6 | 1437 | 541 | false | false |
| prog_chan_s | 14 | 1024 | C1a | 7 | 1228 | 204 | false | false |
| prog_chan_s | 14 | 1152 | Ga | 8 | 1154 | 2 | false | true |
| prog_chan_s | 14 | 1280 | E30b | 9 | 1437 | 157 | false | false |
| prog_chan_s | 14 | 1408 | C7a | 10 | 1636 | 228 | false | false |
| prog_chan_s | 14 | 1536 | E110a | 11 | 1934 | 398 | false | false |
| prog_chan_s | 14 | 1664 | C7a | 12 | 1874 | 210 | false | false |
| prog_chan_s | 14 | 1792 | E30b | 13 |  | 256 | true | false |
| prog_chan_s | 14 | 1920 | C1a | 14 |  | 128 | true | false |
| prog_chan_s | 14 | 2048 | C7a | 15 |  | 0 | true | false |
| prog_chan_s | 15 | 128 | E110a | 0 | 1875 | 1747 | false | false |
| prog_chan_s | 15 | 256 | E110a | 1 | 1875 | 1619 | false | false |
| prog_chan_s | 15 | 384 | C7b | 2 | 469 | 85 | false | true |
| prog_chan_s | 15 | 512 | C7b | 3 | 759 | 247 | false | false |
| prog_chan_s | 15 | 640 | Ga | 4 | 709 | 69 | false | true |
| prog_chan_s | 15 | 768 | E110a | 5 | 1875 | 1107 | false | false |
| prog_chan_s | 15 | 896 | C7a | 6 | 952 | 56 | false | true |
| prog_chan_s | 15 | 1024 | Ga | 7 | 1613 | 589 | false | false |
| prog_chan_s | 15 | 1152 | M37b | 8 | 1174 | 22 | false | true |
| prog_chan_s | 15 | 1280 | M7a | 9 | 1521 | 241 | false | false |
| prog_chan_s | 15 | 1408 | E110b | 10 | 1990 | 582 | false | false |
| prog_chan_s | 15 | 1536 | M7a | 11 |  | 512 | true | false |
| prog_chan_s | 15 | 1664 | C7a | 12 | 1968 | 304 | false | false |
| prog_chan_s | 15 | 1792 | M7b | 13 |  | 256 | true | false |
| prog_chan_s | 15 | 1920 | C1b | 14 |  | 128 | true | false |
| prog_chan_s | 15 | 2048 | C7a | 15 |  | 0 | true | false |
| prog_chan_s | 16 | 128 | C7a | 0 | 374 | 246 | false | false |
| prog_chan_s | 16 | 256 | E30b | 1 | 294 | 38 | false | true |
| prog_chan_s | 16 | 384 | Ga | 2 | 462 | 78 | false | true |
| prog_chan_s | 16 | 512 | M37b | 3 | 811 | 299 | false | false |
| prog_chan_s | 16 | 640 | C1a | 4 | 823 | 183 | false | false |
| prog_chan_s | 16 | 768 | Ga | 5 | 920 | 152 | false | false |
| prog_chan_s | 16 | 896 | E30a | 6 |  | 1152 | true | false |
| prog_chan_s | 16 | 1024 | C7b | 7 | 1328 | 304 | false | false |
| prog_chan_s | 16 | 1152 | C1a | 8 | 1223 | 71 | false | true |
| prog_chan_s | 16 | 1280 | M7b | 9 | 1505 | 225 | false | false |
| prog_chan_s | 16 | 1408 | M7a | 10 |  | 640 | true | false |
| prog_chan_s | 16 | 1536 | E30a | 11 |  | 512 | true | false |
| prog_chan_s | 16 | 1664 | M37a | 12 |  | 384 | true | false |
| prog_chan_s | 16 | 1792 | M7b | 13 | 2007 | 215 | false | false |
| prog_chan_s | 16 | 1920 | E30a | 14 |  | 128 | true | false |
| prog_chan_s | 16 | 2048 | C1a | 15 |  | 0 | true | false |
| prog_chan_s | 17 | 128 | C7b | 0 | 996 | 868 | false | false |
| prog_chan_s | 17 | 256 | E110a | 1 | 1299 | 1043 | false | false |
| prog_chan_s | 17 | 384 | M7b | 2 | 859 | 475 | false | false |
| prog_chan_s | 17 | 512 | M7a | 3 | 812 | 300 | false | false |
| prog_chan_s | 17 | 640 | E30a | 4 | 1949 | 1309 | false | false |
| prog_chan_s | 17 | 768 | Ga | 5 | 826 | 58 | false | true |
| prog_chan_s | 17 | 896 | M37a | 6 | 897 | 1 | false | true |
| prog_chan_s | 17 | 1024 | M37a | 7 | 1071 | 47 | false | true |
| prog_chan_s | 17 | 1152 | E110b | 8 | 1616 | 464 | false | false |
| prog_chan_s | 17 | 1280 | C7b | 9 | 2040 | 760 | false | false |
| prog_chan_s | 17 | 1408 | Gb | 10 | 1567 | 159 | false | false |
| prog_chan_s | 17 | 1536 | C1a | 11 | 1633 | 97 | false | true |
| prog_chan_s | 17 | 1664 | Gb | 12 |  | 384 | true | false |
| prog_chan_s | 17 | 1792 | E30a | 13 | 1949 | 157 | false | false |
| prog_chan_s | 17 | 1920 | E30a | 14 | 1949 | 29 | false | true |
| prog_chan_s | 17 | 2048 | E110b | 15 |  | 0 | true | false |
| prog_chan_s | 18 | 128 | C1b | 0 | 867 | 739 | false | false |
| prog_chan_s | 18 | 256 | M7b | 1 | 559 | 303 | false | false |
| prog_chan_s | 18 | 384 | E30a | 2 | 1289 | 905 | false | false |
| prog_chan_s | 18 | 512 | E110a | 3 |  | 1536 | true | false |
| prog_chan_s | 18 | 640 | M7b | 4 | 1137 | 497 | false | false |
| prog_chan_s | 18 | 768 | Ga | 5 | 1050 | 282 | false | false |
| prog_chan_s | 18 | 896 | C1b | 6 | 949 | 53 | false | true |
| prog_chan_s | 18 | 1024 | Gb | 7 | 1030 | 6 | false | true |
| prog_chan_s | 18 | 1152 | Gb | 8 | 1376 | 224 | false | false |
| prog_chan_s | 18 | 1280 | E110a | 9 |  | 768 | true | false |
| prog_chan_s | 18 | 1408 | C1a | 10 | 1911 | 503 | false | false |
| prog_chan_s | 18 | 1536 | E30a | 11 | 1717 | 181 | false | false |
| prog_chan_s | 18 | 1664 | E110b | 12 | 1817 | 153 | false | false |
| prog_chan_s | 18 | 1792 | E30b | 13 |  | 256 | true | false |
| prog_chan_s | 18 | 1920 | C7b | 14 |  | 128 | true | false |
| prog_chan_s | 18 | 2048 | C7a | 15 |  | 0 | true | false |
| prog_chan_s | 19 | 128 | C7a | 0 | 133 | 5 | false | true |
| prog_chan_s | 19 | 256 | C7b | 1 |  | 1792 | true | false |
| prog_chan_s | 19 | 384 | C7b | 2 |  | 1664 | true | false |
| prog_chan_s | 19 | 512 | C7b | 3 |  | 1536 | true | false |
| prog_chan_s | 19 | 640 | C1b | 4 | 1886 | 1246 | false | false |
| prog_chan_s | 19 | 768 | M7a | 5 |  | 1280 | true | false |
| prog_chan_s | 19 | 896 | M37b | 6 | 1074 | 178 | false | false |
| prog_chan_s | 19 | 1024 | M37a | 7 | 1224 | 200 | false | false |
| prog_chan_s | 19 | 1152 | E110b | 8 |  | 896 | true | false |
| prog_chan_s | 19 | 1280 | E110a | 9 |  | 768 | true | false |
| prog_chan_s | 19 | 1408 | C7a | 10 | 1495 | 87 | false | true |
| prog_chan_s | 19 | 1536 | C7b | 11 |  | 512 | true | false |
| prog_chan_s | 19 | 1664 | M7a | 12 |  | 384 | true | false |
| prog_chan_s | 19 | 1792 | M7b | 13 |  | 256 | true | false |
| prog_chan_s | 19 | 1920 | C7a | 14 |  | 128 | true | false |
| prog_chan_s | 19 | 2048 | Ga | 15 |  | 0 | true | false |
| prog_chan_s | 20 | 128 | M37a | 0 | 142 | 14 | false | true |
| prog_chan_s | 20 | 256 | M37b | 1 | 1130 | 874 | false | false |
| prog_chan_s | 20 | 384 | E30a | 2 | 1798 | 1414 | false | false |
| prog_chan_s | 20 | 512 | M7b | 3 | 566 | 54 | false | true |
| prog_chan_s | 20 | 640 | C1a | 4 | 1059 | 419 | false | false |
| prog_chan_s | 20 | 768 | E30a | 5 | 1798 | 1030 | false | false |
| prog_chan_s | 20 | 896 | C1b | 6 | 1652 | 756 | false | false |
| prog_chan_s | 20 | 1024 | Ga | 7 | 1097 | 73 | false | true |
| prog_chan_s | 20 | 1152 | C1a | 8 | 1425 | 273 | false | false |
| prog_chan_s | 20 | 1280 | M37b | 9 | 1358 | 78 | false | true |
| prog_chan_s | 20 | 1408 | E30a | 10 | 1798 | 390 | false | false |
| prog_chan_s | 20 | 1536 | M7b | 11 | 1774 | 238 | false | false |
| prog_chan_s | 20 | 1664 | Gb | 12 | 1762 | 98 | false | true |
| prog_chan_s | 20 | 1792 | M7a | 13 | 1865 | 73 | false | true |
| prog_chan_s | 20 | 1920 | Gb | 14 |  | 128 | true | false |
| prog_chan_s | 20 | 2048 | C7b | 15 |  | 0 | true | false |
| prog_chan_s | 21 | 128 | C1b | 0 | 426 | 298 | false | false |
| prog_chan_s | 21 | 256 | E110b | 1 | 413 | 157 | false | false |
| prog_chan_s | 21 | 384 | E30a | 2 | 528 | 144 | false | false |
| prog_chan_s | 21 | 512 | M7a | 3 | 935 | 423 | false | false |
| prog_chan_s | 21 | 640 | M7a | 4 | 935 | 295 | false | false |
| prog_chan_s | 21 | 768 | M7b | 5 | 1019 | 251 | false | false |
| prog_chan_s | 21 | 896 | M7a | 6 | 935 | 39 | false | true |
| prog_chan_s | 21 | 1024 | E110b | 7 | 1163 | 139 | false | false |
| prog_chan_s | 21 | 1152 | Ga | 8 | 1175 | 23 | false | true |
| prog_chan_s | 21 | 1280 | M7b | 9 | 1519 | 239 | false | false |
| prog_chan_s | 21 | 1408 | C7a | 10 | 1732 | 324 | false | false |
| prog_chan_s | 21 | 1536 | C1b | 11 | 1898 | 362 | false | false |
| prog_chan_s | 21 | 1664 | E30b | 12 |  | 384 | true | false |
| prog_chan_s | 21 | 1792 | E110a | 13 |  | 256 | true | false |
| prog_chan_s | 21 | 1920 | Gb | 14 | 1963 | 43 | false | true |
| prog_chan_s | 21 | 2048 | E110a | 15 |  | 0 | true | false |
| prog_chan_s | 22 | 128 | C7a | 0 | 348 | 220 | false | false |
| prog_chan_s | 22 | 256 | M7b | 1 | 915 | 659 | false | false |
| prog_chan_s | 22 | 384 | M7b | 2 | 915 | 531 | false | false |
| prog_chan_s | 22 | 512 | Gb | 3 | 588 | 76 | false | true |
| prog_chan_s | 22 | 640 | Ga | 4 | 832 | 192 | false | false |
| prog_chan_s | 22 | 768 | E110b | 5 |  | 1280 | true | false |
| prog_chan_s | 22 | 896 | E30b | 6 |  | 1152 | true | false |
| prog_chan_s | 22 | 1024 | M7a | 7 | 1140 | 116 | false | true |
| prog_chan_s | 22 | 1152 | Gb | 8 | 1536 | 384 | false | false |
| prog_chan_s | 22 | 1280 | M7b | 9 |  | 768 | true | false |
| prog_chan_s | 22 | 1408 | Ga | 10 | 1458 | 50 | false | true |
| prog_chan_s | 22 | 1536 | E110a | 11 | 1881 | 345 | false | false |
| prog_chan_s | 22 | 1664 | C1a | 12 | 1815 | 151 | false | false |
| prog_chan_s | 22 | 1792 | M7b | 13 |  | 256 | true | false |
| prog_chan_s | 22 | 1920 | E30a | 14 |  | 128 | true | false |
| prog_chan_s | 22 | 2048 | Gb | 15 |  | 0 | true | false |
| prog_chan_s | 23 | 128 | M7a | 0 | 142 | 14 | false | true |
| prog_chan_s | 23 | 256 | M37a | 1 | 485 | 229 | false | false |
| prog_chan_s | 23 | 384 | Ga | 2 | 419 | 35 | false | true |
| prog_chan_s | 23 | 512 | C1a | 3 | 713 | 201 | false | false |
| prog_chan_s | 23 | 640 | M37b | 4 | 702 | 62 | false | true |
| prog_chan_s | 23 | 768 | C7b | 5 | 1058 | 290 | false | false |
| prog_chan_s | 23 | 896 | M37a | 6 | 1507 | 611 | false | false |
| prog_chan_s | 23 | 1024 | M7a | 7 | 1296 | 272 | false | false |
| prog_chan_s | 23 | 1152 | E110b | 8 | 1313 | 161 | false | false |
| prog_chan_s | 23 | 1280 | M37a | 9 | 1507 | 227 | false | false |
| prog_chan_s | 23 | 1408 | M37b | 10 |  | 640 | true | false |
| prog_chan_s | 23 | 1536 | E30b | 11 |  | 512 | true | false |
| prog_chan_s | 23 | 1664 | Gb | 12 | 1693 | 29 | false | true |
| prog_chan_s | 23 | 1792 | Gb | 13 |  | 256 | true | false |
| prog_chan_s | 23 | 1920 | E30a | 14 |  | 128 | true | false |
| prog_chan_s | 23 | 2048 | M7a | 15 |  | 0 | true | false |
| prog_age_s | 12 | 128 | E30b | 0 | 304 | 176 | false | false |
| prog_age_s | 12 | 256 | M37a | 1 | 263 | 7 | false | true |
| prog_age_s | 12 | 384 | Ga | 2 | 559 | 175 | false | false |
| prog_age_s | 12 | 512 | E30b | 3 | 704 | 192 | false | false |
| prog_age_s | 12 | 640 | E30b | 4 | 704 | 64 | false | true |
| prog_age_s | 12 | 768 | Ga | 5 | 815 | 47 | false | true |
| prog_age_s | 12 | 896 | C1a | 6 | 955 | 59 | false | true |
| prog_age_s | 12 | 1024 | C7b | 7 | 1122 | 98 | false | true |
| prog_age_s | 12 | 1152 | M7a | 8 | 1251 | 99 | false | true |
| prog_age_s | 12 | 1280 | M37b | 9 | 1461 | 181 | false | false |
| prog_age_s | 12 | 1408 | M7b | 10 | 1692 | 284 | false | false |
| prog_age_s | 12 | 1536 | Gb | 11 | 1580 | 44 | false | true |
| prog_age_s | 12 | 1664 | C1a | 12 | 1945 | 281 | false | false |
| prog_age_s | 12 | 1792 | C1b | 13 | 1880 | 88 | false | true |
| prog_age_s | 12 | 1920 | E30b | 14 |  | 128 | true | false |
| prog_age_s | 12 | 2048 | Ga | 15 |  | 0 | true | false |
| prog_age_s | 13 | 128 | E110b | 0 | 267 | 139 | false | false |
| prog_age_s | 13 | 256 | M37b | 1 | 380 | 124 | false | true |
| prog_age_s | 13 | 384 | C7a | 2 | 389 | 5 | false | true |
| prog_age_s | 13 | 512 | M37a | 3 | 709 | 197 | false | false |
| prog_age_s | 13 | 640 | E30a | 4 | 905 | 265 | false | false |
| prog_age_s | 13 | 768 | C7a | 5 | 1071 | 303 | false | false |
| prog_age_s | 13 | 896 | M7b | 6 | 958 | 62 | false | true |
| prog_age_s | 13 | 1024 | C7a | 7 | 1071 | 47 | false | true |
| prog_age_s | 13 | 1152 | Ga | 8 | 1270 | 118 | false | true |
| prog_age_s | 13 | 1280 | C1b | 9 | 1383 | 103 | false | true |
| prog_age_s | 13 | 1408 | E30a | 10 | 1577 | 169 | false | false |
| prog_age_s | 13 | 1536 | M7b | 11 | 1598 | 62 | false | true |
| prog_age_s | 13 | 1664 | Ga | 12 | 1886 | 222 | false | false |
| prog_age_s | 13 | 1792 | M37b | 13 | 2010 | 218 | false | false |
| prog_age_s | 13 | 1920 | M7b | 14 | 1972 | 52 | false | true |
| prog_age_s | 13 | 2048 | C1b | 15 |  | 0 | true | false |
| prog_age_s | 14 | 128 | C7b | 0 | 234 | 106 | false | true |
| prog_age_s | 14 | 256 | M37a | 1 | 379 | 123 | false | true |
| prog_age_s | 14 | 384 | C7a | 2 | 496 | 112 | false | true |
| prog_age_s | 14 | 512 | E30b | 3 | 533 | 21 | false | true |
| prog_age_s | 14 | 640 | C7a | 4 | 810 | 170 | false | false |
| prog_age_s | 14 | 768 | Gb | 5 | 777 | 9 | false | true |
| prog_age_s | 14 | 896 | E30b | 6 | 1217 | 321 | false | false |
| prog_age_s | 14 | 1024 | C1a | 7 | 1232 | 208 | false | false |
| prog_age_s | 14 | 1152 | Ga | 8 | 1296 | 144 | false | false |
| prog_age_s | 14 | 1280 | E30b | 9 | 1555 | 275 | false | false |
| prog_age_s | 14 | 1408 | C7a | 10 | 1444 | 36 | false | true |
| prog_age_s | 14 | 1536 | E110a | 11 | 1748 | 212 | false | false |
| prog_age_s | 14 | 1664 | C7a | 12 | 1820 | 156 | false | false |
| prog_age_s | 14 | 1792 | E30b | 13 | 1905 | 113 | false | true |
| prog_age_s | 14 | 1920 | C1a | 14 | 1921 | 1 | false | true |
| prog_age_s | 14 | 2048 | C7a | 15 |  | 0 | true | false |
| prog_age_s | 15 | 128 | E110a | 0 | 537 | 409 | false | false |
| prog_age_s | 15 | 256 | E110a | 1 | 537 | 281 | false | false |
| prog_age_s | 15 | 384 | C7b | 2 | 485 | 101 | false | true |
| prog_age_s | 15 | 512 | C7b | 3 | 751 | 239 | false | false |
| prog_age_s | 15 | 640 | Ga | 4 | 855 | 215 | false | false |
| prog_age_s | 15 | 768 | E110a | 5 | 897 | 129 | false | false |
| prog_age_s | 15 | 896 | C7a | 6 | 968 | 72 | false | true |
| prog_age_s | 15 | 1024 | Ga | 7 | 1111 | 87 | false | true |
| prog_age_s | 15 | 1152 | M37b | 8 | 1422 | 270 | false | false |
| prog_age_s | 15 | 1280 | M7a | 9 | 1529 | 249 | false | false |
| prog_age_s | 15 | 1408 | E110b | 10 | 1696 | 288 | false | false |
| prog_age_s | 15 | 1536 | M7a | 11 | 1883 | 347 | false | false |
| prog_age_s | 15 | 1664 | C7a | 12 | 1798 | 134 | false | false |
| prog_age_s | 15 | 1792 | M7b | 13 | 1842 | 50 | false | true |
| prog_age_s | 15 | 1920 | C1b | 14 |  | 128 | true | false |
| prog_age_s | 15 | 2048 | C7a | 15 |  | 0 | true | false |
| prog_age_s | 16 | 128 | C7a | 0 | 286 | 158 | false | false |
| prog_age_s | 16 | 256 | E30b | 1 | 268 | 12 | false | true |
| prog_age_s | 16 | 384 | Ga | 2 | 385 | 1 | false | true |
| prog_age_s | 16 | 512 | M37b | 3 | 615 | 103 | false | true |
| prog_age_s | 16 | 640 | C1a | 4 | 687 | 47 | false | true |
| prog_age_s | 16 | 768 | Ga | 5 | 886 | 118 | false | true |
| prog_age_s | 16 | 896 | E30a | 6 | 933 | 37 | false | true |
| prog_age_s | 16 | 1024 | C7b | 7 | 1204 | 180 | false | false |
| prog_age_s | 16 | 1152 | C1a | 8 | 1257 | 105 | false | true |
| prog_age_s | 16 | 1280 | M7b | 9 | 1443 | 163 | false | false |
| prog_age_s | 16 | 1408 | M7a | 10 | 1498 | 90 | false | true |
| prog_age_s | 16 | 1536 | E30a | 11 | 1581 | 45 | false | true |
| prog_age_s | 16 | 1664 | M37a | 12 | 1763 | 99 | false | true |
| prog_age_s | 16 | 1792 | M7b | 13 | 1999 | 207 | false | false |
| prog_age_s | 16 | 1920 | E30a | 14 |  | 128 | true | false |
| prog_age_s | 16 | 2048 | C1a | 15 |  | 0 | true | false |
| prog_age_s | 17 | 128 | C7b | 0 | 382 | 254 | false | false |
| prog_age_s | 17 | 256 | E110a | 1 | 429 | 173 | false | false |
| prog_age_s | 17 | 384 | M7b | 2 | 549 | 165 | false | false |
| prog_age_s | 17 | 512 | M7a | 3 | 690 | 178 | false | false |
| prog_age_s | 17 | 640 | E30a | 4 | 713 | 73 | false | true |
| prog_age_s | 17 | 768 | Ga | 5 | 810 | 42 | false | true |
| prog_age_s | 17 | 896 | M37a | 6 | 1113 | 217 | false | false |
| prog_age_s | 17 | 1024 | M37a | 7 | 1113 | 89 | false | true |
| prog_age_s | 17 | 1152 | E110b | 8 | 1332 | 180 | false | false |
| prog_age_s | 17 | 1280 | C7b | 9 | 1574 | 294 | false | false |
| prog_age_s | 17 | 1408 | Gb | 10 | 1525 | 117 | false | true |
| prog_age_s | 17 | 1536 | C1a | 11 | 1565 | 29 | false | true |
| prog_age_s | 17 | 1664 | Gb | 12 | 1815 | 151 | false | false |
| prog_age_s | 17 | 1792 | E30a | 13 | 2035 | 243 | false | false |
| prog_age_s | 17 | 1920 | E30a | 14 | 2035 | 115 | false | true |
| prog_age_s | 17 | 2048 | E110b | 15 |  | 0 | true | false |
| prog_age_s | 18 | 128 | C1b | 0 | 333 | 205 | false | false |
| prog_age_s | 18 | 256 | M7b | 1 | 409 | 153 | false | false |
| prog_age_s | 18 | 384 | E30a | 2 | 435 | 51 | false | true |
| prog_age_s | 18 | 512 | E110a | 3 | 773 | 261 | false | false |
| prog_age_s | 18 | 640 | M7b | 4 | 673 | 33 | false | true |
| prog_age_s | 18 | 768 | Ga | 5 | 1066 | 298 | false | false |
| prog_age_s | 18 | 896 | C1b | 6 | 913 | 17 | false | true |
| prog_age_s | 18 | 1024 | Gb | 7 | 1136 | 112 | false | true |
| prog_age_s | 18 | 1152 | Gb | 8 | 1376 | 224 | false | false |
| prog_age_s | 18 | 1280 | E110a | 9 | 1425 | 145 | false | false |
| prog_age_s | 18 | 1408 | C1a | 10 | 1415 | 7 | false | true |
| prog_age_s | 18 | 1536 | E30a | 11 | 1765 | 229 | false | false |
| prog_age_s | 18 | 1664 | E110b | 12 | 1853 | 189 | false | false |
| prog_age_s | 18 | 1792 | E30b | 13 | 1994 | 202 | false | false |
| prog_age_s | 18 | 1920 | C7b | 14 |  | 128 | true | false |
| prog_age_s | 18 | 2048 | C7a | 15 |  | 0 | true | false |
| prog_age_s | 19 | 128 | C7a | 0 | 133 | 5 | false | true |
| prog_age_s | 19 | 256 | C7b | 1 | 582 | 326 | false | false |
| prog_age_s | 19 | 384 | C7b | 2 | 582 | 198 | false | false |
| prog_age_s | 19 | 512 | C7b | 3 | 582 | 70 | false | true |
| prog_age_s | 19 | 640 | C1b | 4 | 664 | 24 | false | true |
| prog_age_s | 19 | 768 | M7a | 5 | 1055 | 287 | false | false |
| prog_age_s | 19 | 896 | M37b | 6 | 1066 | 170 | false | false |
| prog_age_s | 19 | 1024 | M37a | 7 | 1094 | 70 | false | true |
| prog_age_s | 19 | 1152 | E110b | 8 | 1206 | 54 | false | true |
| prog_age_s | 19 | 1280 | E110a | 9 | 1479 | 199 | false | false |
| prog_age_s | 19 | 1408 | C7a | 10 | 1531 | 123 | false | true |
| prog_age_s | 19 | 1536 | C7b | 11 | 1888 | 352 | false | false |
| prog_age_s | 19 | 1664 | M7a | 12 | 1995 | 331 | false | false |
| prog_age_s | 19 | 1792 | M7b | 13 | 1968 | 176 | false | false |
| prog_age_s | 19 | 1920 | C7a | 14 |  | 128 | true | false |
| prog_age_s | 19 | 2048 | Ga | 15 |  | 0 | true | false |
| prog_age_s | 20 | 128 | M37a | 0 | 142 | 14 | false | true |
| prog_age_s | 20 | 256 | M37b | 1 | 468 | 212 | false | false |
| prog_age_s | 20 | 384 | E30a | 2 | 506 | 122 | false | true |
| prog_age_s | 20 | 512 | M7b | 3 | 612 | 100 | false | true |
| prog_age_s | 20 | 640 | C1a | 4 | 653 | 13 | false | true |
| prog_age_s | 20 | 768 | E30a | 5 | 860 | 92 | false | true |
| prog_age_s | 20 | 896 | C1b | 6 | 1168 | 272 | false | false |
| prog_age_s | 20 | 1024 | Ga | 7 | 1211 | 187 | false | false |
| prog_age_s | 20 | 1152 | C1a | 8 | 1255 | 103 | false | true |
| prog_age_s | 20 | 1280 | M37b | 9 | 1328 | 48 | false | true |
| prog_age_s | 20 | 1408 | E30a | 10 | 1484 | 76 | false | true |
| prog_age_s | 20 | 1536 | M7b | 11 | 1540 | 4 | false | true |
| prog_age_s | 20 | 1664 | Gb | 12 | 1696 | 32 | false | true |
| prog_age_s | 20 | 1792 | M7a | 13 | 2003 | 211 | false | false |
| prog_age_s | 20 | 1920 | Gb | 14 | 1944 | 24 | false | true |
| prog_age_s | 20 | 2048 | C7b | 15 |  | 0 | true | false |
| prog_age_s | 21 | 128 | C1b | 0 | 268 | 140 | false | false |
| prog_age_s | 21 | 256 | E110b | 1 | 471 | 215 | false | false |
| prog_age_s | 21 | 384 | E30a | 2 | 548 | 164 | false | false |
| prog_age_s | 21 | 512 | M7a | 3 | 811 | 299 | false | false |
| prog_age_s | 21 | 640 | M7a | 4 | 811 | 171 | false | false |
| prog_age_s | 21 | 768 | M7b | 5 | 895 | 127 | false | true |
| prog_age_s | 21 | 896 | M7a | 6 | 993 | 97 | false | true |
| prog_age_s | 21 | 1024 | E110b | 7 | 1265 | 241 | false | false |
| prog_age_s | 21 | 1152 | Ga | 8 | 1153 | 1 | false | true |
| prog_age_s | 21 | 1280 | M7b | 9 | 1543 | 263 | false | false |
| prog_age_s | 21 | 1408 | C7a | 10 | 1612 | 204 | false | false |
| prog_age_s | 21 | 1536 | C1b | 11 | 1830 | 294 | false | false |
| prog_age_s | 21 | 1664 | E30b | 12 | 1990 | 326 | false | false |
| prog_age_s | 21 | 1792 | E110a | 13 | 2028 | 236 | false | false |
| prog_age_s | 21 | 1920 | Gb | 14 | 2005 | 85 | false | true |
| prog_age_s | 21 | 2048 | E110a | 15 |  | 0 | true | false |
| prog_age_s | 22 | 128 | C7a | 0 | 328 | 200 | false | false |
| prog_age_s | 22 | 256 | M7b | 1 | 461 | 205 | false | false |
| prog_age_s | 22 | 384 | M7b | 2 | 461 | 77 | false | true |
| prog_age_s | 22 | 512 | Gb | 3 | 660 | 148 | false | false |
| prog_age_s | 22 | 640 | Ga | 4 | 648 | 8 | false | true |
| prog_age_s | 22 | 768 | E110b | 5 | 783 | 15 | false | true |
| prog_age_s | 22 | 896 | E30b | 6 | 1209 | 313 | false | false |
| prog_age_s | 22 | 1024 | M7a | 7 | 1242 | 218 | false | false |
| prog_age_s | 22 | 1152 | Gb | 8 | 1368 | 216 | false | false |
| prog_age_s | 22 | 1280 | M7b | 9 | 1379 | 99 | false | true |
| prog_age_s | 22 | 1408 | Ga | 10 | 1674 | 266 | false | false |
| prog_age_s | 22 | 1536 | E110a | 11 | 1805 | 269 | false | false |
| prog_age_s | 22 | 1664 | C1a | 12 | 1933 | 269 | false | false |
| prog_age_s | 22 | 1792 | M7b | 13 |  | 256 | true | false |
| prog_age_s | 22 | 1920 | E30a | 14 | 2025 | 105 | false | true |
| prog_age_s | 22 | 2048 | Gb | 15 |  | 0 | true | false |
| prog_age_s | 23 | 128 | M7a | 0 | 142 | 14 | false | true |
| prog_age_s | 23 | 256 | M37a | 1 | 257 | 1 | false | true |
| prog_age_s | 23 | 384 | Ga | 2 | 391 | 7 | false | true |
| prog_age_s | 23 | 512 | C1a | 3 | 553 | 41 | false | true |
| prog_age_s | 23 | 640 | M37b | 4 | 752 | 112 | false | true |
| prog_age_s | 23 | 768 | C7b | 5 | 992 | 224 | false | false |
| prog_age_s | 23 | 896 | M37a | 6 | 1047 | 151 | false | false |
| prog_age_s | 23 | 1024 | M7a | 7 | 1302 | 278 | false | false |
| prog_age_s | 23 | 1152 | E110b | 8 | 1253 | 101 | false | true |
| prog_age_s | 23 | 1280 | M37a | 9 | 1341 | 61 | false | true |
| prog_age_s | 23 | 1408 | M37b | 10 | 1410 | 2 | false | true |
| prog_age_s | 23 | 1536 | E30b | 11 | 1702 | 166 | false | false |
| prog_age_s | 23 | 1664 | Gb | 12 | 1665 | 1 | false | true |
| prog_age_s | 23 | 1792 | Gb | 13 | 1945 | 153 | false | false |
| prog_age_s | 23 | 1920 | E30a | 14 | 1958 | 38 | false | true |
| prog_age_s | 23 | 2048 | M7a | 15 |  | 0 | true | false |

## Decision shares

Mean over seeds. A stay counts for the room. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.

| controller | Ga | C1a | C7a | E30a | E110a | M7a | M37a | Gb | C1b | C7b | E30b | E110b | M7b | M37b | N1 | N2 | N3 | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 0.0319 | 0.0290 | 0.0228 | 0.0316 | 0.0251 | 0.0310 | 0.0283 | 0.0277 | 0.0322 | 0.0254 | 0.0286 | 0.0239 | 0.0254 | 0.0326 | 0.0302 | 0.0270 | 0.0260 | 0.0283 | 0.2939 | 0.1992 |
| oracle_fresh_s | 0.0342 | 0.0332 | 0.0343 | 0.0439 | 0.0326 | 0.0391 | 0.0387 | 0.0411 | 0.0320 | 0.0339 | 0.0385 | 0.0328 | 0.0345 | 0.0410 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2909 | 0.1992 |
| prog_chan_s | 0.0615 | 0.0215 | 0.0306 | 0.0146 | 0.0166 | 0.0212 | 0.0368 | 0.0804 | 0.0127 | 0.0231 | 0.0195 | 0.0228 | 0.0204 | 0.0329 | 0.0584 | 0.0645 | 0.0462 | 0.0078 | 0.2093 | 0.1992 |
| prog_age_s | 0.0358 | 0.0280 | 0.0286 | 0.0267 | 0.0252 | 0.0277 | 0.0319 | 0.0422 | 0.0257 | 0.0283 | 0.0260 | 0.0280 | 0.0273 | 0.0290 | 0.0283 | 0.0303 | 0.0309 | 0.0250 | 0.2757 | 0.1992 |

| controller | seed | Ga | C1a | C7a | E30a | E110a | M7a | M37a | Gb | C1b | C7b | E30b | E110b | M7b | M37b | N1 | N2 | N3 | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 12 | 0.0273 | 0.0312 | 0.0273 | 0.0234 | 0.0234 | 0.0156 | 0.0234 | 0.0352 | 0.0195 | 0.0273 | 0.0273 | 0.0234 | 0.0430 | 0.0352 | 0.0391 | 0.0234 | 0.0430 | 0.0234 | 0.2891 | 0.1992 |
| rand_dwell_s | 13 | 0.0391 | 0.0352 | 0.0391 | 0.0469 | 0.0195 | 0.0391 | 0.0352 | 0.0430 | 0.0195 | 0.0234 | 0.0000 | 0.0273 | 0.0234 | 0.0352 | 0.0117 | 0.0078 | 0.0273 | 0.0234 | 0.3047 | 0.1992 |
| rand_dwell_s | 14 | 0.0347 | 0.0391 | 0.0117 | 0.0273 | 0.0195 | 0.0195 | 0.0195 | 0.0156 | 0.0430 | 0.0234 | 0.0508 | 0.0312 | 0.0352 | 0.0273 | 0.0234 | 0.0039 | 0.0273 | 0.0508 | 0.2974 | 0.1992 |
| rand_dwell_s | 15 | 0.0312 | 0.0312 | 0.0195 | 0.0234 | 0.0234 | 0.0508 | 0.0469 | 0.0195 | 0.0508 | 0.0391 | 0.0234 | 0.0195 | 0.0156 | 0.0352 | 0.0156 | 0.0156 | 0.0273 | 0.0156 | 0.2969 | 0.1992 |
| rand_dwell_s | 16 | 0.0200 | 0.0117 | 0.0195 | 0.0469 | 0.0352 | 0.0352 | 0.0234 | 0.0234 | 0.0273 | 0.0234 | 0.0547 | 0.0391 | 0.0117 | 0.0312 | 0.0273 | 0.0273 | 0.0234 | 0.0156 | 0.3042 | 0.1992 |
| rand_dwell_s | 17 | 0.0352 | 0.0469 | 0.0039 | 0.0273 | 0.0352 | 0.0317 | 0.0117 | 0.0156 | 0.0312 | 0.0156 | 0.0273 | 0.0352 | 0.0391 | 0.0195 | 0.0195 | 0.0508 | 0.0234 | 0.0430 | 0.2886 | 0.1992 |
| rand_dwell_s | 18 | 0.0312 | 0.0195 | 0.0234 | 0.0391 | 0.0156 | 0.0430 | 0.0234 | 0.0234 | 0.0352 | 0.0156 | 0.0234 | 0.0195 | 0.0352 | 0.0352 | 0.0337 | 0.0352 | 0.0273 | 0.0195 | 0.3022 | 0.1992 |
| rand_dwell_s | 19 | 0.0391 | 0.0508 | 0.0391 | 0.0195 | 0.0352 | 0.0273 | 0.0156 | 0.0391 | 0.0352 | 0.0234 | 0.0273 | 0.0078 | 0.0234 | 0.0352 | 0.0469 | 0.0156 | 0.0195 | 0.0195 | 0.2812 | 0.1992 |
| rand_dwell_s | 20 | 0.0430 | 0.0078 | 0.0234 | 0.0156 | 0.0352 | 0.0156 | 0.0352 | 0.0234 | 0.0234 | 0.0312 | 0.0391 | 0.0312 | 0.0195 | 0.0469 | 0.0312 | 0.0391 | 0.0195 | 0.0352 | 0.2852 | 0.1992 |
| rand_dwell_s | 21 | 0.0195 | 0.0195 | 0.0195 | 0.0352 | 0.0234 | 0.0156 | 0.0312 | 0.0469 | 0.0352 | 0.0312 | 0.0312 | 0.0195 | 0.0234 | 0.0352 | 0.0312 | 0.0469 | 0.0234 | 0.0273 | 0.2852 | 0.1992 |
| rand_dwell_s | 22 | 0.0234 | 0.0234 | 0.0312 | 0.0352 | 0.0273 | 0.0352 | 0.0391 | 0.0273 | 0.0273 | 0.0312 | 0.0312 | 0.0059 | 0.0078 | 0.0195 | 0.0391 | 0.0352 | 0.0312 | 0.0430 | 0.2871 | 0.1992 |
| rand_dwell_s | 23 | 0.0391 | 0.0312 | 0.0156 | 0.0391 | 0.0078 | 0.0430 | 0.0352 | 0.0195 | 0.0391 | 0.0195 | 0.0078 | 0.0273 | 0.0273 | 0.0352 | 0.0430 | 0.0234 | 0.0195 | 0.0234 | 0.3047 | 0.1992 |
| oracle_fresh_s | 12 | 0.0352 | 0.0391 | 0.0469 | 0.0352 | 0.0352 | 0.0195 | 0.0391 | 0.0391 | 0.0312 | 0.0352 | 0.0586 | 0.0234 | 0.0386 | 0.0352 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2896 | 0.1992 |
| oracle_fresh_s | 13 | 0.0508 | 0.0352 | 0.0586 | 0.0586 | 0.0234 | 0.0352 | 0.0391 | 0.0498 | 0.0156 | 0.0156 | 0.0156 | 0.0391 | 0.0195 | 0.0469 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2979 | 0.1992 |
| oracle_fresh_s | 14 | 0.0156 | 0.0430 | 0.0547 | 0.0391 | 0.0352 | 0.0234 | 0.0469 | 0.0312 | 0.0352 | 0.0312 | 0.0547 | 0.0127 | 0.0469 | 0.0430 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2881 | 0.1992 |
| oracle_fresh_s | 15 | 0.0469 | 0.0312 | 0.0430 | 0.0273 | 0.0352 | 0.0781 | 0.0312 | 0.0352 | 0.0273 | 0.0234 | 0.0352 | 0.0273 | 0.0195 | 0.0430 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2969 | 0.1992 |
| oracle_fresh_s | 16 | 0.0234 | 0.0273 | 0.0352 | 0.0469 | 0.0312 | 0.0547 | 0.0352 | 0.0234 | 0.0234 | 0.0391 | 0.0601 | 0.0312 | 0.0312 | 0.0391 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2993 | 0.1992 |
| oracle_fresh_s | 17 | 0.0156 | 0.0273 | 0.0156 | 0.0391 | 0.0312 | 0.0391 | 0.0430 | 0.0312 | 0.0312 | 0.0625 | 0.0469 | 0.0454 | 0.0352 | 0.0430 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2944 | 0.1992 |
| oracle_fresh_s | 18 | 0.0469 | 0.0391 | 0.0273 | 0.0352 | 0.0312 | 0.0469 | 0.0273 | 0.0273 | 0.0508 | 0.0312 | 0.0352 | 0.0352 | 0.0508 | 0.0352 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2812 | 0.1992 |
| oracle_fresh_s | 19 | 0.0469 | 0.0312 | 0.0273 | 0.0430 | 0.0391 | 0.0312 | 0.0391 | 0.0391 | 0.0405 | 0.0391 | 0.0273 | 0.0547 | 0.0234 | 0.0430 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2759 | 0.1992 |
| oracle_fresh_s | 20 | 0.0430 | 0.0312 | 0.0430 | 0.0508 | 0.0547 | 0.0234 | 0.0430 | 0.0469 | 0.0273 | 0.0234 | 0.0234 | 0.0195 | 0.0312 | 0.0508 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2891 | 0.1992 |
| oracle_fresh_s | 21 | 0.0352 | 0.0352 | 0.0234 | 0.0508 | 0.0234 | 0.0352 | 0.0273 | 0.0586 | 0.0391 | 0.0312 | 0.0508 | 0.0430 | 0.0195 | 0.0469 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2812 | 0.1992 |
| oracle_fresh_s | 22 | 0.0312 | 0.0312 | 0.0156 | 0.0469 | 0.0352 | 0.0430 | 0.0469 | 0.0571 | 0.0195 | 0.0391 | 0.0312 | 0.0312 | 0.0469 | 0.0312 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2944 | 0.1992 |
| oracle_fresh_s | 23 | 0.0195 | 0.0273 | 0.0210 | 0.0547 | 0.0156 | 0.0391 | 0.0469 | 0.0547 | 0.0430 | 0.0352 | 0.0234 | 0.0312 | 0.0508 | 0.0352 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3032 | 0.1992 |
| prog_chan_s | 12 | 0.0586 | 0.0156 | 0.0273 | 0.0068 | 0.0352 | 0.0039 | 0.0430 | 0.0938 | 0.0078 | 0.0273 | 0.0117 | 0.0078 | 0.0273 | 0.0391 | 0.0625 | 0.0703 | 0.0508 | 0.0117 | 0.2002 | 0.1992 |
| prog_chan_s | 13 | 0.0664 | 0.0469 | 0.0234 | 0.0039 | 0.0039 | 0.0234 | 0.0156 | 0.1016 | 0.0117 | 0.0078 | 0.0039 | 0.0742 | 0.0195 | 0.0352 | 0.0742 | 0.0630 | 0.0234 | 0.0039 | 0.1987 | 0.1992 |
| prog_chan_s | 14 | 0.0547 | 0.0117 | 0.0156 | 0.0039 | 0.0391 | 0.0078 | 0.0469 | 0.0703 | 0.0195 | 0.0586 | 0.0117 | 0.0039 | 0.0039 | 0.0586 | 0.0273 | 0.0938 | 0.0625 | 0.0039 | 0.2070 | 0.1992 |
| prog_chan_s | 15 | 0.0703 | 0.0117 | 0.0352 | 0.0078 | 0.0078 | 0.0195 | 0.0547 | 0.0859 | 0.0117 | 0.0234 | 0.0586 | 0.0156 | 0.0078 | 0.0356 | 0.0586 | 0.0625 | 0.0117 | 0.0078 | 0.2144 | 0.1992 |
| prog_chan_s | 16 | 0.0703 | 0.0234 | 0.0391 | 0.0039 | 0.0156 | 0.0039 | 0.0430 | 0.0703 | 0.0117 | 0.0117 | 0.0703 | 0.0039 | 0.0264 | 0.0195 | 0.0312 | 0.0391 | 0.0781 | 0.0078 | 0.2314 | 0.1992 |
| prog_chan_s | 17 | 0.0586 | 0.0273 | 0.0195 | 0.0078 | 0.0117 | 0.0273 | 0.0625 | 0.0859 | 0.0273 | 0.0195 | 0.0430 | 0.0117 | 0.0078 | 0.0312 | 0.0664 | 0.0625 | 0.0273 | 0.0039 | 0.1992 | 0.1992 |
| prog_chan_s | 18 | 0.0508 | 0.0273 | 0.0195 | 0.0117 | 0.0039 | 0.0312 | 0.0508 | 0.0508 | 0.0117 | 0.0273 | 0.0117 | 0.0273 | 0.0430 | 0.0312 | 0.0664 | 0.0508 | 0.0503 | 0.0078 | 0.2271 | 0.1992 |
| prog_chan_s | 19 | 0.0938 | 0.0234 | 0.0195 | 0.0664 | 0.0039 | 0.0039 | 0.0195 | 0.0938 | 0.0078 | 0.0039 | 0.0078 | 0.0039 | 0.0430 | 0.0547 | 0.0586 | 0.0586 | 0.0234 | 0.0078 | 0.2070 | 0.1992 |
| prog_chan_s | 20 | 0.0664 | 0.0234 | 0.0469 | 0.0078 | 0.0117 | 0.0391 | 0.0195 | 0.0742 | 0.0117 | 0.0195 | 0.0039 | 0.0234 | 0.0195 | 0.0430 | 0.0820 | 0.0234 | 0.0742 | 0.0078 | 0.2031 | 0.1992 |
| prog_chan_s | 21 | 0.0078 | 0.0078 | 0.0508 | 0.0391 | 0.0039 | 0.0117 | 0.0352 | 0.1016 | 0.0234 | 0.0117 | 0.0039 | 0.0312 | 0.0195 | 0.0273 | 0.0156 | 0.0977 | 0.0820 | 0.0117 | 0.2188 | 0.1992 |
| prog_chan_s | 22 | 0.0625 | 0.0273 | 0.0625 | 0.0078 | 0.0547 | 0.0430 | 0.0195 | 0.0625 | 0.0039 | 0.0117 | 0.0039 | 0.0078 | 0.0078 | 0.0117 | 0.0859 | 0.0586 | 0.0669 | 0.0039 | 0.1987 | 0.1992 |
| prog_chan_s | 23 | 0.0781 | 0.0117 | 0.0078 | 0.0078 | 0.0078 | 0.0391 | 0.0312 | 0.0742 | 0.0039 | 0.0547 | 0.0039 | 0.0625 | 0.0195 | 0.0078 | 0.0718 | 0.0938 | 0.0039 | 0.0156 | 0.2056 | 0.1992 |
| prog_age_s | 12 | 0.0391 | 0.0273 | 0.0273 | 0.0234 | 0.0273 | 0.0234 | 0.0312 | 0.0547 | 0.0234 | 0.0312 | 0.0273 | 0.0234 | 0.0273 | 0.0312 | 0.0273 | 0.0352 | 0.0273 | 0.0234 | 0.2695 | 0.1992 |
| prog_age_s | 13 | 0.0273 | 0.0273 | 0.0273 | 0.0234 | 0.0234 | 0.0273 | 0.0273 | 0.0508 | 0.0273 | 0.0273 | 0.0234 | 0.0312 | 0.0273 | 0.0273 | 0.0312 | 0.0273 | 0.0273 | 0.0264 | 0.2900 | 0.1992 |
| prog_age_s | 14 | 0.0352 | 0.0234 | 0.0234 | 0.0234 | 0.0273 | 0.0234 | 0.0430 | 0.0469 | 0.0234 | 0.0352 | 0.0273 | 0.0234 | 0.0195 | 0.0352 | 0.0273 | 0.0273 | 0.0352 | 0.0234 | 0.2773 | 0.1992 |
| prog_age_s | 15 | 0.0391 | 0.0273 | 0.0312 | 0.0273 | 0.0234 | 0.0273 | 0.0352 | 0.0391 | 0.0273 | 0.0312 | 0.0312 | 0.0273 | 0.0273 | 0.0273 | 0.0273 | 0.0273 | 0.0312 | 0.0273 | 0.2656 | 0.1992 |
| prog_age_s | 16 | 0.0391 | 0.0273 | 0.0312 | 0.0234 | 0.0234 | 0.0234 | 0.0352 | 0.0312 | 0.0273 | 0.0234 | 0.0312 | 0.0234 | 0.0273 | 0.0312 | 0.0273 | 0.0352 | 0.0312 | 0.0195 | 0.2891 | 0.1992 |
| prog_age_s | 17 | 0.0352 | 0.0273 | 0.0312 | 0.0273 | 0.0254 | 0.0273 | 0.0391 | 0.0508 | 0.0234 | 0.0312 | 0.0273 | 0.0273 | 0.0273 | 0.0312 | 0.0312 | 0.0273 | 0.0312 | 0.0234 | 0.2559 | 0.1992 |
| prog_age_s | 18 | 0.0312 | 0.0352 | 0.0234 | 0.0234 | 0.0234 | 0.0273 | 0.0312 | 0.0430 | 0.0273 | 0.0273 | 0.0273 | 0.0273 | 0.0273 | 0.0273 | 0.0273 | 0.0391 | 0.0273 | 0.0234 | 0.2812 | 0.1992 |
| prog_age_s | 19 | 0.0352 | 0.0312 | 0.0273 | 0.0312 | 0.0234 | 0.0273 | 0.0273 | 0.0420 | 0.0273 | 0.0234 | 0.0234 | 0.0234 | 0.0312 | 0.0352 | 0.0273 | 0.0273 | 0.0273 | 0.0273 | 0.2822 | 0.1992 |
| prog_age_s | 20 | 0.0430 | 0.0312 | 0.0312 | 0.0273 | 0.0234 | 0.0352 | 0.0312 | 0.0391 | 0.0273 | 0.0273 | 0.0273 | 0.0312 | 0.0312 | 0.0273 | 0.0312 | 0.0273 | 0.0312 | 0.0273 | 0.2500 | 0.1992 |
| prog_age_s | 21 | 0.0312 | 0.0234 | 0.0352 | 0.0352 | 0.0234 | 0.0273 | 0.0273 | 0.0430 | 0.0273 | 0.0234 | 0.0234 | 0.0312 | 0.0273 | 0.0234 | 0.0273 | 0.0312 | 0.0352 | 0.0273 | 0.2773 | 0.1992 |
| prog_age_s | 22 | 0.0312 | 0.0273 | 0.0312 | 0.0273 | 0.0312 | 0.0352 | 0.0234 | 0.0312 | 0.0234 | 0.0234 | 0.0234 | 0.0273 | 0.0234 | 0.0273 | 0.0273 | 0.0273 | 0.0430 | 0.0234 | 0.2930 | 0.1992 |
| prog_age_s | 23 | 0.0430 | 0.0273 | 0.0234 | 0.0273 | 0.0273 | 0.0273 | 0.0312 | 0.0352 | 0.0234 | 0.0352 | 0.0195 | 0.0391 | 0.0312 | 0.0234 | 0.0273 | 0.0312 | 0.0234 | 0.0273 | 0.2773 | 0.1992 |

## Bandit choices

| controller | seed | choices | first | eps | greedy | ema updates |
|---|---:|---:|---:|---:|---:|---:|
| prog_chan_s | 12 | 154 | 18 | 15 | 121 | 154 |
| prog_chan_s | 13 | 155 | 18 | 13 | 124 | 155 |
| prog_chan_s | 14 | 153 | 18 | 11 | 124 | 153 |
| prog_chan_s | 15 | 151 | 18 | 13 | 120 | 151 |
| prog_chan_s | 16 | 146 | 18 | 11 | 117 | 146 |
| prog_chan_s | 17 | 155 | 18 | 15 | 122 | 155 |
| prog_chan_s | 18 | 147 | 18 | 12 | 117 | 147 |
| prog_chan_s | 19 | 152 | 18 | 9 | 125 | 152 |
| prog_chan_s | 20 | 153 | 18 | 18 | 117 | 153 |
| prog_chan_s | 21 | 150 | 18 | 14 | 118 | 150 |
| prog_chan_s | 22 | 155 | 18 | 19 | 118 | 155 |
| prog_chan_s | 23 | 153 | 18 | 14 | 121 | 153 |
| prog_age_s | 12 | 137 | 18 | 12 | 107 | 137 |
| prog_age_s | 13 | 131 | 18 | 11 | 102 | 131 |
| prog_age_s | 14 | 135 | 18 | 9 | 108 | 135 |
| prog_age_s | 15 | 138 | 18 | 10 | 110 | 138 |
| prog_age_s | 16 | 132 | 18 | 9 | 105 | 132 |
| prog_age_s | 17 | 140 | 18 | 15 | 107 | 140 |
| prog_age_s | 18 | 134 | 18 | 11 | 105 | 134 |
| prog_age_s | 19 | 133 | 18 | 6 | 109 | 133 |
| prog_age_s | 20 | 141 | 18 | 16 | 107 | 141 |
| prog_age_s | 21 | 134 | 18 | 14 | 102 | 134 |
| prog_age_s | 22 | 130 | 18 | 15 | 97 | 130 |
| prog_age_s | 23 | 134 | 18 | 13 | 103 | 134 |

## Swap log

One log per seed. The same log was checked against every controller. k counts swaps from 0.

| seed | decision | room | k |
|---|---:|---:|---:|
| 12 | 128 | E30b | 0 |
| 12 | 256 | M37a | 1 |
| 12 | 384 | Ga | 2 |
| 12 | 512 | E30b | 3 |
| 12 | 640 | E30b | 4 |
| 12 | 768 | Ga | 5 |
| 12 | 896 | C1a | 6 |
| 12 | 1024 | C7b | 7 |
| 12 | 1152 | M7a | 8 |
| 12 | 1280 | M37b | 9 |
| 12 | 1408 | M7b | 10 |
| 12 | 1536 | Gb | 11 |
| 12 | 1664 | C1a | 12 |
| 12 | 1792 | C1b | 13 |
| 12 | 1920 | E30b | 14 |
| 12 | 2048 | Ga | 15 |
| 13 | 128 | E110b | 0 |
| 13 | 256 | M37b | 1 |
| 13 | 384 | C7a | 2 |
| 13 | 512 | M37a | 3 |
| 13 | 640 | E30a | 4 |
| 13 | 768 | C7a | 5 |
| 13 | 896 | M7b | 6 |
| 13 | 1024 | C7a | 7 |
| 13 | 1152 | Ga | 8 |
| 13 | 1280 | C1b | 9 |
| 13 | 1408 | E30a | 10 |
| 13 | 1536 | M7b | 11 |
| 13 | 1664 | Ga | 12 |
| 13 | 1792 | M37b | 13 |
| 13 | 1920 | M7b | 14 |
| 13 | 2048 | C1b | 15 |
| 14 | 128 | C7b | 0 |
| 14 | 256 | M37a | 1 |
| 14 | 384 | C7a | 2 |
| 14 | 512 | E30b | 3 |
| 14 | 640 | C7a | 4 |
| 14 | 768 | Gb | 5 |
| 14 | 896 | E30b | 6 |
| 14 | 1024 | C1a | 7 |
| 14 | 1152 | Ga | 8 |
| 14 | 1280 | E30b | 9 |
| 14 | 1408 | C7a | 10 |
| 14 | 1536 | E110a | 11 |
| 14 | 1664 | C7a | 12 |
| 14 | 1792 | E30b | 13 |
| 14 | 1920 | C1a | 14 |
| 14 | 2048 | C7a | 15 |
| 15 | 128 | E110a | 0 |
| 15 | 256 | E110a | 1 |
| 15 | 384 | C7b | 2 |
| 15 | 512 | C7b | 3 |
| 15 | 640 | Ga | 4 |
| 15 | 768 | E110a | 5 |
| 15 | 896 | C7a | 6 |
| 15 | 1024 | Ga | 7 |
| 15 | 1152 | M37b | 8 |
| 15 | 1280 | M7a | 9 |
| 15 | 1408 | E110b | 10 |
| 15 | 1536 | M7a | 11 |
| 15 | 1664 | C7a | 12 |
| 15 | 1792 | M7b | 13 |
| 15 | 1920 | C1b | 14 |
| 15 | 2048 | C7a | 15 |
| 16 | 128 | C7a | 0 |
| 16 | 256 | E30b | 1 |
| 16 | 384 | Ga | 2 |
| 16 | 512 | M37b | 3 |
| 16 | 640 | C1a | 4 |
| 16 | 768 | Ga | 5 |
| 16 | 896 | E30a | 6 |
| 16 | 1024 | C7b | 7 |
| 16 | 1152 | C1a | 8 |
| 16 | 1280 | M7b | 9 |
| 16 | 1408 | M7a | 10 |
| 16 | 1536 | E30a | 11 |
| 16 | 1664 | M37a | 12 |
| 16 | 1792 | M7b | 13 |
| 16 | 1920 | E30a | 14 |
| 16 | 2048 | C1a | 15 |
| 17 | 128 | C7b | 0 |
| 17 | 256 | E110a | 1 |
| 17 | 384 | M7b | 2 |
| 17 | 512 | M7a | 3 |
| 17 | 640 | E30a | 4 |
| 17 | 768 | Ga | 5 |
| 17 | 896 | M37a | 6 |
| 17 | 1024 | M37a | 7 |
| 17 | 1152 | E110b | 8 |
| 17 | 1280 | C7b | 9 |
| 17 | 1408 | Gb | 10 |
| 17 | 1536 | C1a | 11 |
| 17 | 1664 | Gb | 12 |
| 17 | 1792 | E30a | 13 |
| 17 | 1920 | E30a | 14 |
| 17 | 2048 | E110b | 15 |
| 18 | 128 | C1b | 0 |
| 18 | 256 | M7b | 1 |
| 18 | 384 | E30a | 2 |
| 18 | 512 | E110a | 3 |
| 18 | 640 | M7b | 4 |
| 18 | 768 | Ga | 5 |
| 18 | 896 | C1b | 6 |
| 18 | 1024 | Gb | 7 |
| 18 | 1152 | Gb | 8 |
| 18 | 1280 | E110a | 9 |
| 18 | 1408 | C1a | 10 |
| 18 | 1536 | E30a | 11 |
| 18 | 1664 | E110b | 12 |
| 18 | 1792 | E30b | 13 |
| 18 | 1920 | C7b | 14 |
| 18 | 2048 | C7a | 15 |
| 19 | 128 | C7a | 0 |
| 19 | 256 | C7b | 1 |
| 19 | 384 | C7b | 2 |
| 19 | 512 | C7b | 3 |
| 19 | 640 | C1b | 4 |
| 19 | 768 | M7a | 5 |
| 19 | 896 | M37b | 6 |
| 19 | 1024 | M37a | 7 |
| 19 | 1152 | E110b | 8 |
| 19 | 1280 | E110a | 9 |
| 19 | 1408 | C7a | 10 |
| 19 | 1536 | C7b | 11 |
| 19 | 1664 | M7a | 12 |
| 19 | 1792 | M7b | 13 |
| 19 | 1920 | C7a | 14 |
| 19 | 2048 | Ga | 15 |
| 20 | 128 | M37a | 0 |
| 20 | 256 | M37b | 1 |
| 20 | 384 | E30a | 2 |
| 20 | 512 | M7b | 3 |
| 20 | 640 | C1a | 4 |
| 20 | 768 | E30a | 5 |
| 20 | 896 | C1b | 6 |
| 20 | 1024 | Ga | 7 |
| 20 | 1152 | C1a | 8 |
| 20 | 1280 | M37b | 9 |
| 20 | 1408 | E30a | 10 |
| 20 | 1536 | M7b | 11 |
| 20 | 1664 | Gb | 12 |
| 20 | 1792 | M7a | 13 |
| 20 | 1920 | Gb | 14 |
| 20 | 2048 | C7b | 15 |
| 21 | 128 | C1b | 0 |
| 21 | 256 | E110b | 1 |
| 21 | 384 | E30a | 2 |
| 21 | 512 | M7a | 3 |
| 21 | 640 | M7a | 4 |
| 21 | 768 | M7b | 5 |
| 21 | 896 | M7a | 6 |
| 21 | 1024 | E110b | 7 |
| 21 | 1152 | Ga | 8 |
| 21 | 1280 | M7b | 9 |
| 21 | 1408 | C7a | 10 |
| 21 | 1536 | C1b | 11 |
| 21 | 1664 | E30b | 12 |
| 21 | 1792 | E110a | 13 |
| 21 | 1920 | Gb | 14 |
| 21 | 2048 | E110a | 15 |
| 22 | 128 | C7a | 0 |
| 22 | 256 | M7b | 1 |
| 22 | 384 | M7b | 2 |
| 22 | 512 | Gb | 3 |
| 22 | 640 | Ga | 4 |
| 22 | 768 | E110b | 5 |
| 22 | 896 | E30b | 6 |
| 22 | 1024 | M7a | 7 |
| 22 | 1152 | Gb | 8 |
| 22 | 1280 | M7b | 9 |
| 22 | 1408 | Ga | 10 |
| 22 | 1536 | E110a | 11 |
| 22 | 1664 | C1a | 12 |
| 22 | 1792 | M7b | 13 |
| 22 | 1920 | E30a | 14 |
| 22 | 2048 | Gb | 15 |
| 23 | 128 | M7a | 0 |
| 23 | 256 | M37a | 1 |
| 23 | 384 | Ga | 2 |
| 23 | 512 | C1a | 3 |
| 23 | 640 | M37b | 4 |
| 23 | 768 | C7b | 5 |
| 23 | 896 | M37a | 6 |
| 23 | 1024 | M7a | 7 |
| 23 | 1152 | E110b | 8 |
| 23 | 1280 | M37a | 9 |
| 23 | 1408 | M37b | 10 |
| 23 | 1536 | E30b | 11 |
| 23 | 1664 | Gb | 12 |
| 23 | 1792 | Gb | 13 |
| 23 | 1920 | E30a | 14 |
| 23 | 2048 | M7a | 15 |

## Ring order

Room name in ring positions 0..17. One order per seed. Controllers share it.

| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 | r16 | r17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | M7a | E110a | N3 | E30b | N2 | Gb | E30a | M7b | M37a | C7b | C7a | N1 | M37b | E110b | Ga | Z | C1a | C1b |
| 13 | Z | C1a | E110b | E110a | M37a | N1 | E30b | M7b | Gb | M37b | C7a | C7b | E30a | N3 | M7a | N2 | C1b | Ga |
| 14 | M7a | E110a | Gb | Z | N3 | N1 | M37a | C7b | E110b | E30a | E30b | C7a | C1b | M37b | N2 | Ga | M7b | C1a |
| 15 | Z | N3 | C7b | C7a | E30b | M37a | Ga | M7b | N1 | E110b | E110a | C1b | M7a | N2 | C1a | Gb | E30a | M37b |
| 16 | N2 | C7a | C1a | E30b | E110b | N3 | M37b | N1 | M7b | Ga | E110a | Gb | E30a | M7a | Z | C1b | M37a | C7b |
| 17 | Gb | N2 | N1 | N3 | M37b | Ga | M37a | E110b | C1a | C7b | E30a | C7a | E110a | Z | E30b | C1b | M7b | M7a |
| 18 | E110b | N2 | C1b | N1 | C1a | M37a | E110a | M7a | C7b | M37b | M7b | Ga | E30a | E30b | Z | Gb | C7a | N3 |
| 19 | Z | C1a | E30a | M7b | M7a | C1b | E110a | M37b | Gb | E30b | N3 | M37a | C7a | E110b | N1 | C7b | N2 | Ga |
| 20 | N3 | Z | C1a | M7b | M7a | E110b | Ga | N1 | C7a | C1b | E30b | E30a | C7b | M37a | N2 | M37b | E110a | Gb |
| 21 | M7a | E30a | Z | C1b | Ga | C7a | Gb | N2 | E110b | C7b | N1 | C1a | M37a | E30b | M7b | N3 | M37b | E110a |
| 22 | C1a | Ga | E110a | C1b | E110b | Gb | N2 | C7a | M7b | M7a | N1 | M37a | C7b | Z | E30b | N3 | E30a | M37b |
| 23 | Z | M37b | M37a | C7b | E110b | N1 | M7b | C1b | E110a | E30a | N3 | N2 | Ga | M7a | Gb | E30b | C1a | C7a |

## W5 reference

Period 128, seeds 0–5, L = 2048. Source: results/w5/results.json.

| policy | T14 mean | T14 std ddof=0 | W14 mean | W14 std ddof=0 |
|---|---:|---:|---:|---:|
| rand_dwell_s | 2.162024 | 0.153050 | 1.463722 | 0.403017 |
| oracle14_s | 2.320627 | 0.158066 | 1.933755 | 0.568730 |
| oracle_fresh_s | 1.909463 | 0.177389 | 1.594566 | 0.329902 |

oracle_fresh_s − rand_dwell_s on T14: mean -0.252561, std ddof=0 0.218194, left lower 5 of 6.

| policy | seed | T14 | W14 |
|---|---:|---:|---:|
| rand_dwell_s | 0 | 2.215448 | 1.429577 |
| oracle14_s | 0 | 2.383048 | 1.549045 |
| oracle_fresh_s | 0 | 1.802812 | 1.724323 |
| rand_dwell_s | 1 | 1.981291 | 1.029645 |
| oracle14_s | 1 | 2.189115 | 1.037400 |
| oracle_fresh_s | 1 | 1.692186 | 1.121380 |
| rand_dwell_s | 2 | 2.143855 | 2.134280 |
| oracle14_s | 2 | 2.076690 | 2.836758 |
| oracle_fresh_s | 2 | 1.796393 | 1.599950 |
| rand_dwell_s | 3 | 2.039866 | 1.031553 |
| oracle14_s | 3 | 2.562918 | 1.834800 |
| oracle_fresh_s | 3 | 2.239526 | 1.309556 |
| rand_dwell_s | 4 | 2.131586 | 1.331539 |
| oracle14_s | 4 | 2.419263 | 2.013694 |
| oracle_fresh_s | 4 | 1.925698 | 1.644449 |
| rand_dwell_s | 5 | 2.460099 | 1.825739 |
| oracle14_s | 5 | 2.292729 | 2.330834 |
| oracle_fresh_s | 5 | 2.000165 | 2.167736 |

## Walls and kernels

| kernel | seed | wall seconds | workers |
|---|---:|---:|---:|
| ezzzzzx/m8-w5-s12 | 12 | 272.634 | 4 |
| ezzzzzx/m8-w5-s13 | 13 | 308.675 | 4 |
| ezzzzzx/m8-w5-s14 | 14 | 386.084 | 4 |
| ezzzzzx/m8-w5-s15 | 15 | 309.404 | 4 |
| ezzzzzx/m8-w5-s16 | 16 | 321.484 | 4 |
| ezzzzzx/m8-w5-s17 | 17 | 348.225 | 4 |
| ezzzzzx/m8-w5-s18 | 18 | 299.388 | 4 |
| ezzzzzx/m8-w5-s19 | 19 | 264.176 | 4 |
| ezzzzzx/m8-w5-s20 | 20 | 306.623 | 4 |
| ezzzzzx/m8-w5-s21 | 21 | 279.568 | 4 |
| ezzzzzx/m8-w5-s22 | 22 | 337.138 | 4 |
| ezzzzzx/m8-w5-s23 | 23 | 316.875 | 4 |

| controller | seed | wall seconds | learn seconds | score seconds | eval seconds |
|---|---:|---:|---:|---:|---:|
| rand_dwell_s | 12 | 209.811 | 104.876 | 87.724 | 7.615 |
| rand_dwell_s | 13 | 238.492 | 123.635 | 95.974 | 8.634 |
| rand_dwell_s | 14 | 304.735 | 159.899 | 121.451 | 12.350 |
| rand_dwell_s | 15 | 237.456 | 124.836 | 93.902 | 8.679 |
| rand_dwell_s | 16 | 247.660 | 127.161 | 101.025 | 8.926 |
| rand_dwell_s | 17 | 261.662 | 133.384 | 108.264 | 9.075 |
| rand_dwell_s | 18 | 229.037 | 116.861 | 93.871 | 8.583 |
| rand_dwell_s | 19 | 202.817 | 103.339 | 83.264 | 7.336 |
| rand_dwell_s | 20 | 232.637 | 117.390 | 96.941 | 8.388 |
| rand_dwell_s | 21 | 215.519 | 106.417 | 91.531 | 7.792 |
| rand_dwell_s | 22 | 267.020 | 137.738 | 110.020 | 9.430 |
| rand_dwell_s | 23 | 237.046 | 122.675 | 95.289 | 8.607 |
| oracle_fresh_s | 12 | 212.386 | 113.768 | 81.667 | 7.322 |
| oracle_fresh_s | 13 | 239.762 | 128.627 | 92.332 | 8.534 |
| oracle_fresh_s | 14 | 304.095 | 162.015 | 119.773 | 11.216 |
| oracle_fresh_s | 15 | 238.424 | 130.081 | 90.077 | 8.160 |
| oracle_fresh_s | 16 | 254.506 | 135.580 | 99.535 | 8.797 |
| oracle_fresh_s | 17 | 272.177 | 141.970 | 109.916 | 9.323 |
| oracle_fresh_s | 18 | 237.750 | 127.031 | 92.444 | 8.504 |
| oracle_fresh_s | 19 | 206.594 | 110.470 | 80.258 | 6.943 |
| oracle_fresh_s | 20 | 237.754 | 126.329 | 93.262 | 8.180 |
| oracle_fresh_s | 21 | 219.733 | 115.577 | 86.516 | 7.808 |
| oracle_fresh_s | 22 | 260.046 | 143.064 | 97.336 | 9.465 |
| oracle_fresh_s | 23 | 246.946 | 134.389 | 93.092 | 9.026 |
| prog_chan_s | 12 | 269.707 | 99.893 | 153.598 | 6.355 |
| prog_chan_s | 13 | 305.448 | 112.626 | 174.793 | 7.517 |
| prog_chan_s | 14 | 377.844 | 135.736 | 222.371 | 8.383 |
| prog_chan_s | 15 | 302.460 | 116.898 | 167.878 | 7.383 |
| prog_chan_s | 16 | 316.275 | 116.417 | 181.460 | 7.625 |
| prog_chan_s | 17 | 333.155 | 123.049 | 191.090 | 7.827 |
| prog_chan_s | 18 | 296.191 | 111.673 | 167.098 | 7.469 |
| prog_chan_s | 19 | 261.340 | 98.329 | 147.534 | 6.358 |
| prog_chan_s | 20 | 303.104 | 112.096 | 173.546 | 7.322 |
| prog_chan_s | 21 | 270.490 | 99.526 | 154.325 | 6.666 |
| prog_chan_s | 22 | 333.117 | 127.971 | 187.370 | 7.810 |
| prog_chan_s | 23 | 310.341 | 117.408 | 174.438 | 7.858 |
| prog_age_s | 12 | 264.336 | 95.611 | 152.265 | 6.699 |
| prog_age_s | 13 | 304.034 | 113.032 | 172.864 | 7.772 |
| prog_age_s | 14 | 382.508 | 137.555 | 224.053 | 9.681 |
| prog_age_s | 15 | 305.945 | 111.726 | 176.320 | 7.723 |
| prog_age_s | 16 | 318.006 | 111.774 | 187.531 | 7.982 |
| prog_age_s | 17 | 344.638 | 119.859 | 205.703 | 7.985 |
| prog_age_s | 18 | 296.290 | 106.727 | 172.095 | 7.593 |
| prog_age_s | 19 | 259.307 | 95.005 | 148.668 | 6.635 |
| prog_age_s | 20 | 303.492 | 106.457 | 179.226 | 7.717 |
| prog_age_s | 21 | 276.454 | 99.262 | 160.191 | 7.077 |
| prog_age_s | 22 | 323.839 | 123.279 | 182.393 | 8.047 |
| prog_age_s | 23 | 313.331 | 115.927 | 179.218 | 7.656 |

## Smoke test

```
torch=2.15.0+cpu threads=1 model_seed=0 n_ch=19 tau=256 period=128
dose L64 16/64 share 0.250000 checkpoints [8, 16, 24, 32, 40, 48, 56, 64]
dose L128 24/128 share 0.187500 checkpoints [16, 32, 48, 64, 80, 96, 112, 128]
dose L256 48/256 share 0.187500 checkpoints [32, 64, 96, 128, 160, 192, 224, 256]
dose L512 104/512 share 0.203125 checkpoints [64, 128, 192, 256, 320, 384, 448, 512]
dose L2048 408/2048 share 0.199219 checkpoints [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
formula: age 0 returns v; tau=256 moves 0 toward 1 by 1-exp(-1); aged rank differs from raw rank
smoke check 1: rand_dwell_s and oracle_fresh_s seed 0 period 128 L=256 vs W5
  rand_dwell_s m8 T14 10.444404 W14 6.268811 w5 T14 10.444404 W14 6.268811 abs_dT14 0.000e+00 published_dT14 4.968e-07 published_dW14 1.768e-07 limit 1.0e-06
  oracle_fresh_s m8 T14 7.876722 W14 3.486887 w5 T14 7.876722 W14 3.486887 abs_dT14 0.000e+00 published_dT14 2.628e-07 published_dW14 3.860e-07 limit 1.0e-06
smoke check 2: prog_age_s seed 12 L=512, first 5 choices, with checkpoints
choice 1 decision 1 t 0 mode first chosen M7a index 0 v_max none
  M7a index 0 v 0.000000 age none aged none v_max none
  E110a index 1 v 0.000000 age none aged none v_max none
  N3 index 2 v 0.000000 age none aged none v_max none
  E30b index 3 v 0.000000 age none aged none v_max none
  N2 index 4 v 0.000000 age none aged none v_max none
  Gb index 5 v 0.000000 age none aged none v_max none
  E30a index 6 v 0.000000 age none aged none v_max none
  M7b index 7 v 0.000000 age none aged none v_max none
  M37a index 8 v 0.000000 age none aged none v_max none
  C7b index 9 v 0.000000 age none aged none v_max none
  C7a index 10 v 0.000000 age none aged none v_max none
  N1 index 11 v 0.000000 age none aged none v_max none
  M37b index 12 v 0.000000 age none aged none v_max none
  E110b index 13 v 0.000000 age none aged none v_max none
  Ga index 14 v 0.000000 age none aged none v_max none
  Z index 15 v 0.000000 age none aged none v_max none
  C1a index 16 v 0.000000 age none aged none v_max none
  C1b index 17 v 0.000000 age none aged none v_max none
choice 2 decision 9 t 8 mode first chosen E110a index 1 v_max 18.798110
  M7a index 0 v 18.798110 age 0 aged 18.798110 v_max 18.798110
  E110a index 1 v 0.000000 age none aged none v_max 18.798110
  N3 index 2 v 0.000000 age none aged none v_max 18.798110
  E30b index 3 v 0.000000 age none aged none v_max 18.798110
  N2 index 4 v 0.000000 age none aged none v_max 18.798110
  Gb index 5 v 0.000000 age none aged none v_max 18.798110
  E30a index 6 v 0.000000 age none aged none v_max 18.798110
  M7b index 7 v 0.000000 age none aged none v_max 18.798110
  M37a index 8 v 0.000000 age none aged none v_max 18.798110
  C7b index 9 v 0.000000 age none aged none v_max 18.798110
  C7a index 10 v 0.000000 age none aged none v_max 18.798110
  N1 index 11 v 0.000000 age none aged none v_max 18.798110
  M37b index 12 v 0.000000 age none aged none v_max 18.798110
  E110b index 13 v 0.000000 age none aged none v_max 18.798110
  Ga index 14 v 0.000000 age none aged none v_max 18.798110
  Z index 15 v 0.000000 age none aged none v_max 18.798110
  C1a index 16 v 0.000000 age none aged none v_max 18.798110
  C1b index 17 v 0.000000 age none aged none v_max 18.798110
choice 3 decision 18 t 17 mode first chosen N3 index 2 v_max 54.935438
  M7a index 0 v 18.798110 age 9 aged 20.046491 v_max 54.935438
  E110a index 1 v 54.935438 age 0 aged 54.935438 v_max 54.935438
  N3 index 2 v 0.000000 age none aged none v_max 54.935438
  E30b index 3 v 0.000000 age none aged none v_max 54.935438
  N2 index 4 v 0.000000 age none aged none v_max 54.935438
  Gb index 5 v 0.000000 age none aged none v_max 54.935438
  E30a index 6 v 0.000000 age none aged none v_max 54.935438
  M7b index 7 v 0.000000 age none aged none v_max 54.935438
  M37a index 8 v 0.000000 age none aged none v_max 54.935438
  C7b index 9 v 0.000000 age none aged none v_max 54.935438
  C7a index 10 v 0.000000 age none aged none v_max 54.935438
  N1 index 11 v 0.000000 age none aged none v_max 54.935438
  M37b index 12 v 0.000000 age none aged none v_max 54.935438
  E110b index 13 v 0.000000 age none aged none v_max 54.935438
  Ga index 14 v 0.000000 age none aged none v_max 54.935438
  Z index 15 v 0.000000 age none aged none v_max 54.935438
  C1a index 16 v 0.000000 age none aged none v_max 54.935438
  C1b index 17 v 0.000000 age none aged none v_max 54.935438
choice 4 decision 27 t 26 mode first chosen E30b index 3 v_max 54.935438
  M7a index 0 v 18.798110 age 18 aged 21.251745 v_max 54.935438
  E110a index 1 v 54.935438 age 9 aged 54.935438 v_max 54.935438
  N3 index 2 v 37.915349 age 0 aged 37.915349 v_max 54.935438
  E30b index 3 v 0.000000 age none aged none v_max 54.935438
  N2 index 4 v 0.000000 age none aged none v_max 54.935438
  Gb index 5 v 0.000000 age none aged none v_max 54.935438
  E30a index 6 v 0.000000 age none aged none v_max 54.935438
  M7b index 7 v 0.000000 age none aged none v_max 54.935438
  M37a index 8 v 0.000000 age none aged none v_max 54.935438
  C7b index 9 v 0.000000 age none aged none v_max 54.935438
  C7a index 10 v 0.000000 age none aged none v_max 54.935438
  N1 index 11 v 0.000000 age none aged none v_max 54.935438
  M37b index 12 v 0.000000 age none aged none v_max 54.935438
  E110b index 13 v 0.000000 age none aged none v_max 54.935438
  Ga index 14 v 0.000000 age none aged none v_max 54.935438
  Z index 15 v 0.000000 age none aged none v_max 54.935438
  C1a index 16 v 0.000000 age none aged none v_max 54.935438
  C1b index 17 v 0.000000 age none aged none v_max 54.935438
choice 5 decision 44 t 43 mode first chosen N2 index 4 v_max 54.935438
  M7a index 0 v 18.798110 age 35 aged 23.415901 v_max 54.935438
  E110a index 1 v 54.935438 age 26 aged 54.935438 v_max 54.935438
  N3 index 2 v 37.915349 age 17 aged 39.008879 v_max 54.935438
  E30b index 3 v 11.319932 age 8 aged 12.661840 v_max 54.935438
  N2 index 4 v 0.000000 age none aged none v_max 54.935438
  Gb index 5 v 0.000000 age none aged none v_max 54.935438
  E30a index 6 v 0.000000 age none aged none v_max 54.935438
  M7b index 7 v 0.000000 age none aged none v_max 54.935438
  M37a index 8 v 0.000000 age none aged none v_max 54.935438
  C7b index 9 v 0.000000 age none aged none v_max 54.935438
  C7a index 10 v 0.000000 age none aged none v_max 54.935438
  N1 index 11 v 0.000000 age none aged none v_max 54.935438
  M37b index 12 v 0.000000 age none aged none v_max 54.935438
  E110b index 13 v 0.000000 age none aged none v_max 54.935438
  Ga index 14 v 0.000000 age none aged none v_max 54.935438
  Z index 15 v 0.000000 age none aged none v_max 54.935438
  C1a index 16 v 0.000000 age none aged none v_max 54.935438
  C1b index 17 v 0.000000 age none aged none v_max 54.935438
  age 0 rooms: choice 2 M7a v 18.798110 aged 18.798110, choice 3 E110a v 54.935438 aged 54.935438, choice 4 N3 v 37.915349 aged 37.915349
  prog_age_s seed 12 L=512 T14 4.156383 W14 1.886349 wall 77.69s learn 29.79s score 34.11s eval 10.89s sleep 104 swaps 4 choices 39 first 18 eps 4 greedy 17
  one L2048 prog_age_s 266.5s waves 1 jobs 4 workers 4 kernel 266.5s guarded 533.0s limit 5400s
smoke check 3: prog_chan_s seed 12 L=512 visit trace and swap log
  visit 1-8 room M7a decisions 8 reward 18.798110
  visit 9-17 room E110a decisions 9 reward 54.935438
  visit 18-26 room N3 decisions 9 reward 37.915349
  visit 27-35 room E30b decisions 9 reward 11.319932
  visit 44-52 room N2 decisions 9 reward 37.386175
  visit 53-61 room Gb decisions 9 reward 122.399588
  visit 62-70 room E30a decisions 9 reward 22.260216
  visit 79-87 room M7b decisions 9 reward 52.382521
  visit 88-96 room M37a decisions 9 reward 109.093892
  visit 97-105 room C7b decisions 9 reward 132.959081
  visit 114-122 room C7a decisions 9 reward 49.946289
  visit 123-131 room N1 decisions 9 reward 47.537853
  swap decision 128 room E30b k 0 inside visit 123-131
  visit 132-140 room M37b decisions 9 reward 109.818092
  visit 141-149 room E110b decisions 9 reward 20.648221
  visit 158-166 room Ga decisions 9 reward 110.553471
  visit 167-175 room Z decisions 9 reward -0.073719
  visit 176-184 room C1a decisions 9 reward 16.101302
  visit 193-201 room C1b decisions 9 reward 9.106440
  visit 202-215 room Gb decisions 14 reward 82.785564
  visit 216-227 room C7b decisions 12 reward 16.213280
  visit 236-248 room Ga decisions 13 reward 27.577361
  visit 249-265 room Gb decisions 17 reward 14.478319
  swap decision 256 room M37a k 1 inside visit 249-265
  visit 274-288 room M37b decisions 15 reward 19.542478
  visit 289-300 room M37a decisions 12 reward 65.124378
  visit 309-317 room C7b decisions 9 reward 29.524035
  visit 318-331 room E30b decisions 14 reward 10.925305
  visit 340-352 room M37a decisions 13 reward 6.623281
  visit 353-366 room Ga decisions 14 reward 63.342665
  visit 375-384 room M37b decisions 10 reward 13.063488
  swap decision 384 room Ga k 2 inside visit 375-384
  visit 385-399 room Gb decisions 15 reward 53.730918
  visit 400-416 room Ga decisions 17 reward 35.132693
  visit 425-437 room C7b decisions 13 reward 16.572099
  visit 438-453 room C1b decisions 16 reward 7.728031
  visit 462-475 room Gb decisions 14 reward 39.603886
  visit 484-494 room M37a decisions 11 reward 6.385826
  visit 495-508 room Ga decisions 14 reward 53.997402
  visit 509-512 room Gb decisions 4 reward -25.453493
  swap decision 512 room E30b k 3 inside visit 509-512
  visits 37 sleep 104 swaps 4 wall 66.74s choices 37 first 18 eps 3 greedy 16
smoke check 4: four controllers, seed 12, L=128, finite T14
  rand_dwell_s T14 12.177813 W14 9.534154 finite True sleep 0.1875 decisions 24 corridor 0.3203 swaps 1 wall 23.23s
  oracle_fresh_s T14 11.978777 W14 7.705505 finite True sleep 0.1875 decisions 24 corridor 0.3359 swaps 1 wall 23.17s
  prog_chan_s T14 9.694312 W14 5.068424 finite True sleep 0.1875 decisions 24 corridor 0.0859 swaps 1 wall 24.78s
  prog_age_s T14 9.694312 W14 5.068424 finite True sleep 0.1875 decisions 24 corridor 0.0859 swaps 1 wall 24.88s
```

```
2026-11-14T04:08:30Z dataset create ezzzzzx/pdp-code-m8w5
2026-11-14T04:08:56Z dataset ready ezzzzzx/pdp-code-m8w5
2026-11-14T04:08:56Z push ezzzzzx/m8-w5-s12 try 1
2026-11-14T04:08:59Z pushed ezzzzzx/m8-w5-s12 try 1
2026-11-14T04:08:59Z push ezzzzzx/m8-w5-s13 try 1
2026-11-14T04:09:03Z pushed ezzzzzx/m8-w5-s13 try 1
2026-11-14T04:09:03Z push ezzzzzx/m8-w5-s14 try 1
2026-11-14T04:09:06Z pushed ezzzzzx/m8-w5-s14 try 1
2026-11-14T04:09:06Z push ezzzzzx/m8-w5-s15 try 1
2026-11-14T04:09:10Z pushed ezzzzzx/m8-w5-s15 try 1
2026-11-14T04:09:10Z push ezzzzzx/m8-w5-s16 try 1
2026-11-14T04:09:13Z pushed ezzzzzx/m8-w5-s16 try 1
2026-11-14T04:09:15Z status ezzzzzx/m8-w5-s12 pushed -> RUNNING
2026-11-14T04:09:17Z status ezzzzzx/m8-w5-s13 pushed -> RUNNING
2026-11-14T04:09:19Z status ezzzzzx/m8-w5-s14 pushed -> RUNNING
2026-11-14T04:09:21Z status ezzzzzx/m8-w5-s15 pushed -> RUNNING
2026-11-14T04:09:23Z status ezzzzzx/m8-w5-s16 pushed -> RUNNING
2026-11-14T04:14:37Z status ezzzzzx/m8-w5-s12 RUNNING -> COMPLETE
2026-11-14T04:14:40Z downloaded ezzzzzx/m8-w5-s12 -> results/m8w5/s12/m8_s12.json
2026-11-14T04:14:42Z status ezzzzzx/m8-w5-s13 RUNNING -> COMPLETE
2026-11-14T04:14:45Z downloaded ezzzzzx/m8-w5-s13 -> results/m8w5/s13/m8_s13.json
2026-11-14T04:14:49Z status ezzzzzx/m8-w5-s15 RUNNING -> COMPLETE
2026-11-14T04:14:52Z downloaded ezzzzzx/m8-w5-s15 -> results/m8w5/s15/m8_s15.json
2026-11-14T04:14:54Z push ezzzzzx/m8-w5-s17 try 1
2026-11-14T04:14:58Z pushed ezzzzzx/m8-w5-s17 try 1
2026-11-14T04:14:58Z push ezzzzzx/m8-w5-s18 try 1
2026-11-14T04:15:01Z pushed ezzzzzx/m8-w5-s18 try 1
2026-11-14T04:15:01Z push ezzzzzx/m8-w5-s19 try 1
2026-11-14T04:15:04Z pushed ezzzzzx/m8-w5-s19 try 1
2026-11-14T04:15:39Z status ezzzzzx/m8-w5-s16 RUNNING -> COMPLETE
2026-11-14T04:15:42Z downloaded ezzzzzx/m8-w5-s16 -> results/m8w5/s16/m8_s16.json
2026-11-14T04:16:25Z status ezzzzzx/m8-w5-s17 pushed -> RUNNING
2026-11-14T04:16:27Z status ezzzzzx/m8-w5-s18 pushed -> RUNNING
2026-11-14T04:16:29Z status ezzzzzx/m8-w5-s19 pushed -> RUNNING
2026-11-14T04:16:29Z push ezzzzzx/m8-w5-s20 try 1
2026-11-14T04:16:32Z pushed ezzzzzx/m8-w5-s20 try 1
2026-11-14T04:17:04Z status ezzzzzx/m8-w5-s14 RUNNING -> COMPLETE
2026-11-14T04:17:07Z downloaded ezzzzzx/m8-w5-s14 -> results/m8w5/s14/m8_s14.json
2026-11-14T04:17:15Z status ezzzzzx/m8-w5-s20 pushed -> RUNNING
2026-11-14T04:17:15Z push ezzzzzx/m8-w5-s21 try 1
2026-11-14T04:17:18Z pushed ezzzzzx/m8-w5-s21 try 1
2026-11-14T04:18:00Z status ezzzzzx/m8-w5-s21 pushed -> RUNNING
2026-11-14T04:19:56Z status ezzzzzx/m8-w5-s19 RUNNING -> COMPLETE
2026-11-14T04:19:59Z downloaded ezzzzzx/m8-w5-s19 -> results/m8w5/s19/m8_s19.json
2026-11-14T04:20:04Z push ezzzzzx/m8-w5-s22 try 1
2026-11-14T04:20:07Z pushed ezzzzzx/m8-w5-s22 try 1
2026-11-14T04:20:41Z status ezzzzzx/m8-w5-s18 RUNNING -> COMPLETE
2026-11-14T04:20:45Z downloaded ezzzzzx/m8-w5-s18 -> results/m8w5/s18/m8_s18.json
2026-11-14T04:20:51Z status ezzzzzx/m8-w5-s22 pushed -> RUNNING
2026-11-14T04:20:51Z push ezzzzzx/m8-w5-s23 try 1
2026-11-14T04:20:55Z pushed ezzzzzx/m8-w5-s23 try 1
2026-11-14T04:21:26Z status ezzzzzx/m8-w5-s17 RUNNING -> COMPLETE
2026-11-14T04:21:29Z downloaded ezzzzzx/m8-w5-s17 -> results/m8w5/s17/m8_s17.json
2026-11-14T04:21:37Z status ezzzzzx/m8-w5-s23 pushed -> RUNNING
2026-11-14T04:22:09Z status ezzzzzx/m8-w5-s20 RUNNING -> COMPLETE
2026-11-14T04:22:13Z downloaded ezzzzzx/m8-w5-s20 -> results/m8w5/s20/m8_s20.json
2026-11-14T04:22:51Z status ezzzzzx/m8-w5-s21 RUNNING -> COMPLETE
2026-11-14T04:22:54Z downloaded ezzzzzx/m8-w5-s21 -> results/m8w5/s21/m8_s21.json
2026-11-14T04:26:21Z status ezzzzzx/m8-w5-s22 RUNNING -> COMPLETE
2026-11-14T04:26:23Z downloaded ezzzzzx/m8-w5-s22 -> results/m8w5/s22/m8_s22.json
2026-11-14T04:26:58Z status ezzzzzx/m8-w5-s23 RUNNING -> COMPLETE
2026-11-14T04:27:01Z downloaded ezzzzzx/m8-w5-s23 -> results/m8w5/s23/m8_s23.json
2026-11-14T04:27:31Z all kernels downloaded
```

## Protocol

- World5(world_seed, period=128). Room names, ring order, noise rooms and Z are W4's. After every positive multiple of 128 decisions, one of the 14 learnable rooms is redrawn with default_rng(world_seed + 80000), as in W5.
- The cortex, stores, coin, sleep dose and sleep placement are W5's, which are W4's. Policy movement for rand_dwell_s and oracle_fresh_s is w5.take_action with default_rng(1000 + world_seed). A tie on the ring goes left. Dwell is 8 stays.
- prog_chan_s and prog_age_s choose with default_rng(10000 + world_seed). The optimistic first pass is m7.nearest_unvisited over the 18 ring positions, with n=18. Then eps-greedy with EPS 0.1. The EMA uses BETA 0.3 and m7.ema. Sleep is not an arm.
- prog_age_s ranks the greedy step by v_r + (v_max − v_r) · (1 − exp(−age_r / 256)). age_r is the decisions since that room's last visit ended, measured at the choice. v_max is the largest room-visit reward seen so far. Tau is 256. The stored EMA is not changed by aging.
- The room reward sums bits saved on the destination room's exam glances over the visit, including corridor steps, and divides by the visit's decisions. The glance inserted by the decision is not part of that decision's sum. Sleep does not call the EMA.
- A checkpoint scores w4.evaluate_w14 after that decision's swap. T14 is the mean of the eight checkpoints.
- Swap latency uses the next stay in the swapped room after the swap decision. A missing stay is censored at L − decision.
- World seeds 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23. L = 2048. Period 128. CPU. Private Kaggle kernels, one seed each, pool of 4, no GPU, no internet. Nothing retuned.

## Verdict (orchestrator, 14 Nov 2026, UTC+8)

**Precondition (headroom on these seeds):** oracle_fresh_s − rand_dwell_s = −0.341 on T14 (12 of 12). **MET.**
1. **Curiosity finds the change:** prog_chan_s − rand_dwell_s = −0.405 (11 of 12). **PASS.**
2. **Stale beliefs help:** prog_age_s − rand_dwell_s = −0.587 (12 of 12). **PASS.**
Report only: prog_age_s − prog_chan_s = −0.181 (11 of 12). Both curious controllers beat the oracle that knows where the swaps are.
Median swap-finding latency: oracle 12, prog_age_s 126, rand_dwell_s 137, prog_chan_s 239 decisions.

Checks (orchestrator, from the per-checkpoint table): T14 weights early life heavily, because the checkpoint at decision 256 sits
at 2.8–5.2 bits. Leaving that checkpoint out, the gaps to random dwelling are prog_chan_s −0.112 (10 of 12), prog_age_s −0.315 (12 of 12) and
oracle_fresh_s −0.129 (9 of 12). **Bar 1's pass rests mostly on a fast start.** The bandit lingers in rooms that pay, so it spends 0.21 of its life in
corridors, against random dwelling's 0.29. After the first quarter it is level with the oracle, and in the last quarter it falls behind random dwelling,
because rooms it has stopped visiting keep low values while their rules change (latency 239). **Bar 2's pass holds at every checkpoint.**

**Mechanism.** Progress knows two things the swap oracle does not: how fast each room is still paying, for example the grammar rooms
G get about twice their share, and when to stop travelling. The oracle knows only where the change is. Ageing the bandit's beliefs (τ = 256)
adds the third thing: a room not seen for a while may have changed. With it, the curious agent finds swaps about as quickly as random dwelling
does, but dwells where learning is fast. The two together beat the oracle by 0.25. prog_chan_s also spends 0.17 of its life in the noise rooms,
where early calibration toward uniform counts as progress. prog_age_s spreads its time more evenly and does better.
