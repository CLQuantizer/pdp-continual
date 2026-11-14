# M9-W5 sleep timing on the drifting ring

W5's 18-room ring with period 128. One learnable room's rule is redrawn every 128 decisions. T14 is the mean of W14 at decisions L/8, 2L/8, …, L. W14 is bits/symbol of the frozen bank on a fresh EVAL=2048 continuation of the 14 learnable rooms. The final W14 is the checkpoint at L. Lower is better. The tail leaves out the checkpoint at decision 256 and averages the other seven.

Every controller uses prog_age_s's room choice: per-room exam progress, EPS 0.1, BETA 0.3, and on the greedy step v + (v_max − v) · (1 − exp(−age / 256)). Age is the decisions since that arm's last visit ended. v_max is the largest visit reward so far. For age_sleeparm the visits include sleep, so v_max includes sleep rewards. The stored EMA is not aged. `age_fixed20` is prog_age_s: 408 sleep decisions, share 0.199219, placed as in M8. `age_nosleep` uses the same room choice and never sleeps. `age_sleeparm` is the same bandit over 19 arms, the 18 rooms plus sleep. Sleep is last in the optimistic first pass. A sleep visit is 8 decisions and no travel. Its reward is the whole exam store's bits saved over those decisions, divided by the decisions in the visit (8, unless the life ends first). There is no forced sleep. `age_fixed_matched` is prog_age_s at the share age_sleeparm slept on that seed, rounded half up to a whole number of 8-decision visits, placed with the same plan and due rule as age_fixed20.

A replay draw is stale when the room's rule at the draw is not the rule that produced the glance. The rule id is the latest swap index of that room, or −1 before its first swap. Decisions since the last swap use the latest swap whose decision is strictly earlier than the visit's first decision. Before the first swap the origin is decision 0. A swap's latency is the decisions until the next stay in that room. If the life ends first, the latency is L minus the swap decision and the swap is censored. within 128 counts a stay whose latency is at most 128. The median includes censored latencies.

## T14

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | seed 30 | seed 31 | seed 32 | seed 33 | seed 34 | seed 35 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| age_nosleep | 1.932101 | 2.239328 | 2.314645 | 2.179936 | 2.095051 | 1.872280 | 1.895177 | 1.818747 | 1.831845 | 2.109578 | 2.055440 | 1.808040 | 2.012681 ± 0.168506 |
| age_fixed20 | 1.513683 | 1.962629 | 1.805104 | 1.742038 | 1.628528 | 1.570794 | 1.521246 | 1.777956 | 1.738938 | 1.955068 | 1.451871 | 1.659598 | 1.693955 ± 0.159602 |
| age_sleeparm | 1.829763 | 2.146859 | 1.979690 | 1.955278 | 1.906276 | 1.646275 | 1.580190 | 1.802950 | 1.777105 | 1.836016 | 1.988115 | 1.844261 | 1.857732 ± 0.147202 |
| age_fixed_matched | 1.880047 | 1.927227 | 1.849304 | 2.058512 | 1.864030 | 1.708789 | 1.731000 | 1.566853 | 1.801557 | 2.105080 | 1.748276 | 1.858013 | 1.841557 ± 0.141946 |

## Final W14

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | seed 30 | seed 31 | seed 32 | seed 33 | seed 34 | seed 35 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| age_nosleep | 1.435642 | 1.769748 | 1.349244 | 1.640461 | 1.311647 | 1.508058 | 1.483306 | 1.418555 | 1.536327 | 1.568735 | 1.609765 | 1.388998 | 1.501707 ± 0.126556 |
| age_fixed20 | 0.948800 | 1.701798 | 0.899542 | 1.597660 | 1.061505 | 1.361597 | 0.866740 | 1.523311 | 1.880658 | 1.942029 | 1.495621 | 1.531042 | 1.400859 ± 0.359508 |
| age_sleeparm | 1.255434 | 1.589128 | 1.233929 | 1.588921 | 1.090051 | 1.389377 | 0.903101 | 2.152903 | 1.416156 | 1.675546 | 1.553795 | 1.550460 | 1.449900 ± 0.305844 |
| age_fixed_matched | 1.127096 | 1.582477 | 1.281710 | 1.947844 | 0.995751 | 1.509920 | 0.886183 | 1.513778 | 1.352167 | 1.456697 | 1.529744 | 1.590839 | 1.397850 ± 0.278845 |

## T14 per checkpoint

| controller | seed | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 | T14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| age_nosleep | 24 | 3.536690 | 2.537647 | 2.042688 | 1.544868 | 1.403252 | 1.241385 | 1.714636 | 1.435642 | 1.932101 |
| age_nosleep | 25 | 2.674116 | 2.034952 | 2.687369 | 3.126426 | 1.905585 | 1.690516 | 2.025911 | 1.769748 | 2.239328 |
| age_nosleep | 26 | 4.878763 | 2.322634 | 2.663579 | 2.009991 | 1.886171 | 1.603406 | 1.803367 | 1.349244 | 2.314645 |
| age_nosleep | 27 | 3.312139 | 2.972547 | 2.296036 | 2.400903 | 1.729792 | 1.418918 | 1.668692 | 1.640461 | 2.179936 |
| age_nosleep | 28 | 3.820994 | 2.835009 | 2.181295 | 1.587346 | 1.674557 | 1.498364 | 1.851198 | 1.311647 | 2.095051 |
| age_nosleep | 29 | 3.676373 | 2.722606 | 1.832500 | 1.757202 | 1.289753 | 1.257599 | 0.934151 | 1.508058 | 1.872280 |
| age_nosleep | 30 | 3.532387 | 2.177261 | 1.722031 | 1.389309 | 2.166439 | 1.316270 | 1.374411 | 1.483306 | 1.895177 |
| age_nosleep | 31 | 3.015596 | 1.931543 | 1.558113 | 1.454509 | 1.546608 | 1.973120 | 1.651932 | 1.418555 | 1.818747 |
| age_nosleep | 32 | 2.932615 | 2.537415 | 1.777996 | 1.342582 | 1.453968 | 1.458033 | 1.615823 | 1.536327 | 1.831845 |
| age_nosleep | 33 | 3.349087 | 2.916031 | 2.140608 | 2.042497 | 1.777888 | 1.586043 | 1.495737 | 1.568735 | 2.109578 |
| age_nosleep | 34 | 3.488354 | 2.487019 | 1.723723 | 1.579231 | 1.523726 | 1.529745 | 2.501956 | 1.609765 | 2.055440 |
| age_nosleep | 35 | 2.942738 | 2.547054 | 1.616160 | 1.978960 | 1.224406 | 1.447970 | 1.318037 | 1.388998 | 1.808040 |
| age_fixed20 | 24 | 2.547735 | 1.844934 | 1.260924 | 1.457617 | 1.194649 | 1.595632 | 1.259173 | 0.948800 | 1.513683 |
| age_fixed20 | 25 | 2.739393 | 1.674721 | 1.575860 | 1.986245 | 2.157310 | 1.753367 | 2.112342 | 1.701798 | 1.962629 |
| age_fixed20 | 26 | 3.217499 | 1.796438 | 1.293208 | 2.088499 | 1.998439 | 1.430688 | 1.716519 | 0.899542 | 1.805104 |
| age_fixed20 | 27 | 3.002196 | 1.871653 | 1.390547 | 1.602364 | 2.043459 | 1.100552 | 1.327873 | 1.597660 | 1.742038 |
| age_fixed20 | 28 | 3.124773 | 1.785256 | 2.336850 | 1.432141 | 0.992395 | 1.251058 | 1.044250 | 1.061505 | 1.628528 |
| age_fixed20 | 29 | 3.164120 | 1.870695 | 1.446564 | 1.560995 | 1.271586 | 0.946146 | 0.944652 | 1.361597 | 1.570794 |
| age_fixed20 | 30 | 3.315940 | 1.624596 | 1.298400 | 1.152283 | 1.880505 | 1.115102 | 0.916403 | 0.866740 | 1.521246 |
| age_fixed20 | 31 | 2.749120 | 2.114875 | 1.062030 | 1.960936 | 1.711376 | 1.904840 | 1.197161 | 1.523311 | 1.777956 |
| age_fixed20 | 32 | 3.279877 | 1.970580 | 1.902622 | 1.449676 | 1.118019 | 1.032623 | 1.277444 | 1.880658 | 1.738938 |
| age_fixed20 | 33 | 3.675736 | 1.907116 | 2.010269 | 1.296975 | 1.770348 | 1.110790 | 1.927286 | 1.942029 | 1.955068 |
| age_fixed20 | 34 | 2.010713 | 2.141065 | 1.444465 | 1.234493 | 1.227002 | 0.908020 | 1.153589 | 1.495621 | 1.451871 |
| age_fixed20 | 35 | 2.213706 | 2.478170 | 1.317957 | 2.025821 | 1.094893 | 1.593794 | 1.021403 | 1.531042 | 1.659598 |
| age_sleeparm | 24 | 3.043254 | 2.284953 | 1.965490 | 1.529184 | 1.309689 | 1.768651 | 1.481446 | 1.255434 | 1.829763 |
| age_sleeparm | 25 | 2.485881 | 2.835222 | 2.282045 | 2.372054 | 1.619442 | 1.974969 | 2.016135 | 1.589128 | 2.146859 |
| age_sleeparm | 26 | 3.959163 | 2.165937 | 1.410547 | 2.099119 | 1.940668 | 1.692388 | 1.335768 | 1.233929 | 1.979690 |
| age_sleeparm | 27 | 3.383471 | 2.996904 | 1.857163 | 1.687326 | 1.573911 | 1.062017 | 1.492513 | 1.588921 | 1.955278 |
| age_sleeparm | 28 | 3.628922 | 1.983136 | 2.674384 | 1.707683 | 1.304972 | 1.409746 | 1.451317 | 1.090051 | 1.906276 |
| age_sleeparm | 29 | 3.706776 | 1.959204 | 1.388182 | 1.317097 | 1.291162 | 1.136505 | 0.981900 | 1.389377 | 1.646275 |
| age_sleeparm | 30 | 3.129338 | 1.612075 | 1.793833 | 1.221915 | 1.702911 | 1.093530 | 1.184816 | 0.903101 | 1.580190 |
| age_sleeparm | 31 | 2.727110 | 1.753003 | 1.622345 | 1.273821 | 1.625146 | 2.005681 | 1.263594 | 2.152903 | 1.802950 |
| age_sleeparm | 32 | 2.843449 | 2.331136 | 1.758787 | 1.541399 | 1.417655 | 1.084860 | 1.823399 | 1.416156 | 1.777105 |
| age_sleeparm | 33 | 3.267322 | 2.396475 | 1.859731 | 1.398998 | 1.343416 | 1.448737 | 1.297902 | 1.675546 | 1.836016 |
| age_sleeparm | 34 | 3.424863 | 2.697085 | 1.907570 | 1.927013 | 2.003600 | 1.125547 | 1.265447 | 1.553795 | 1.988115 |
| age_sleeparm | 35 | 3.062598 | 2.447646 | 1.770041 | 1.675755 | 1.375694 | 1.634893 | 1.237000 | 1.550460 | 1.844261 |
| age_fixed_matched | 24 | 3.117503 | 2.562505 | 1.839639 | 1.530727 | 1.517469 | 1.699423 | 1.646015 | 1.127096 | 1.880047 |
| age_fixed_matched | 25 | 2.818344 | 2.225915 | 1.838991 | 2.023344 | 1.076962 | 1.739647 | 2.112135 | 1.582477 | 1.927227 |
| age_fixed_matched | 26 | 3.259224 | 1.570156 | 1.438941 | 1.652993 | 2.023586 | 1.747908 | 1.819912 | 1.281710 | 1.849304 |
| age_fixed_matched | 27 | 3.478695 | 1.925601 | 1.876525 | 1.906630 | 1.867797 | 1.140071 | 2.324929 | 1.947844 | 2.058512 |
| age_fixed_matched | 28 | 3.581875 | 2.362620 | 2.205222 | 1.308581 | 1.370556 | 1.561449 | 1.526184 | 0.995751 | 1.864030 |
| age_fixed_matched | 29 | 3.523668 | 1.989960 | 1.633018 | 1.337568 | 1.370138 | 1.136986 | 1.169050 | 1.509920 | 1.708789 |
| age_fixed_matched | 30 | 3.325177 | 2.109459 | 1.585753 | 1.517465 | 2.004533 | 1.136410 | 1.283019 | 0.886183 | 1.731000 |
| age_fixed_matched | 31 | 2.291397 | 2.123776 | 1.258611 | 1.349780 | 1.290817 | 1.724232 | 0.982436 | 1.513778 | 1.566853 |
| age_fixed_matched | 32 | 3.103039 | 2.806374 | 1.881178 | 1.462034 | 1.201948 | 1.314262 | 1.291452 | 1.352167 | 1.801557 |
| age_fixed_matched | 33 | 4.199349 | 2.924031 | 1.859134 | 1.803935 | 1.606762 | 1.619396 | 1.371335 | 1.456697 | 2.105080 |
| age_fixed_matched | 34 | 3.414087 | 2.478649 | 1.438055 | 1.226520 | 1.435766 | 1.080236 | 1.383147 | 1.529744 | 1.748276 |
| age_fixed_matched | 35 | 2.760968 | 2.332231 | 1.680267 | 2.397913 | 1.041073 | 1.948100 | 1.112712 | 1.590839 | 1.858013 |

Mean over seeds at each checkpoint.

| controller | 256 | 512 | 768 | 1024 | 1280 | 1536 | 1792 | 2048 | T14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| age_nosleep | 3.429988 | 2.501810 | 2.020175 | 1.851152 | 1.631846 | 1.501781 | 1.662988 | 1.501707 | 2.012681 |
| age_fixed20 | 2.920067 | 1.923342 | 1.528308 | 1.604004 | 1.538332 | 1.311884 | 1.324841 | 1.400859 | 1.693955 |
| age_sleeparm | 3.221846 | 2.288565 | 1.857510 | 1.645947 | 1.542356 | 1.453127 | 1.402603 | 1.449900 | 1.857732 |
| age_fixed_matched | 3.239444 | 2.284273 | 1.711278 | 1.626458 | 1.483951 | 1.487343 | 1.501861 | 1.397850 | 1.841557 |

## T14 without decision 256

Mean of the checkpoints at 512, 768, 1024, 1280, 1536, 1792 and 2048.

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | seed 30 | seed 31 | seed 32 | seed 33 | seed 34 | seed 35 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| age_nosleep | 1.702874 | 2.177215 | 1.948342 | 2.018193 | 1.848488 | 1.614553 | 1.661290 | 1.647769 | 1.674592 | 1.932506 | 1.850738 | 1.645941 | 1.810208 ± 0.172925 |
| age_fixed20 | 1.365961 | 1.851663 | 1.603334 | 1.562015 | 1.414779 | 1.343176 | 1.264861 | 1.639218 | 1.518803 | 1.709259 | 1.372037 | 1.580440 | 1.518796 ± 0.164680 |
| age_sleeparm | 1.656407 | 2.098428 | 1.696908 | 1.751251 | 1.660184 | 1.351918 | 1.358883 | 1.670927 | 1.624770 | 1.631544 | 1.782865 | 1.670213 | 1.662858 ± 0.183662 |
| age_fixed_matched | 1.703268 | 1.799924 | 1.647887 | 1.855628 | 1.618623 | 1.449520 | 1.503260 | 1.463347 | 1.615631 | 1.805899 | 1.510303 | 1.729019 | 1.641859 ± 0.134040 |

## Gaps

On T14. A negative gap means the first controller is lower. left lower counts seeds where the gap is negative.

| seed | age_fixed20 − age_nosleep | age_sleeparm − age_nosleep | age_sleeparm − age_fixed_matched | age_sleeparm − age_fixed20 |
|---|---:|---:|---:|---:|
| 24 | -0.418418 | -0.102338 | -0.050285 | 0.316080 |
| 25 | -0.276698 | -0.092468 | 0.219633 | 0.184230 |
| 26 | -0.509540 | -0.334955 | 0.130386 | 0.174586 |
| 27 | -0.437898 | -0.224658 | -0.103233 | 0.213240 |
| 28 | -0.466523 | -0.188775 | 0.042247 | 0.277748 |
| 29 | -0.301486 | -0.226005 | -0.062513 | 0.075481 |
| 30 | -0.373931 | -0.314987 | -0.150810 | 0.058944 |
| 31 | -0.040791 | -0.015797 | 0.236097 | 0.024994 |
| 32 | -0.092907 | -0.054740 | -0.024452 | 0.038168 |
| 33 | -0.154510 | -0.273562 | -0.269064 | -0.119052 |
| 34 | -0.603569 | -0.067325 | 0.239839 | 0.536244 |
| 35 | -0.148442 | 0.036220 | -0.013752 | 0.184663 |
| mean | -0.318726 | -0.154949 | 0.016174 | 0.163777 |
| std ddof=0 | 0.171476 | 0.116703 | 0.155502 | 0.161284 |
| left lower (count) | 12 of 12 | 11 of 12 | 7 of 12 | 1 of 12 |

## Gaps without decision 256

Same gaps on the tail mean. A negative gap means the first controller is lower.

| seed | age_fixed20 − age_nosleep | age_sleeparm − age_nosleep | age_sleeparm − age_fixed_matched | age_sleeparm − age_fixed20 |
|---|---:|---:|---:|---:|
| 24 | -0.336913 | -0.046467 | -0.046861 | 0.290446 |
| 25 | -0.325552 | -0.078787 | 0.298503 | 0.246765 |
| 26 | -0.345008 | -0.251434 | 0.049021 | 0.093574 |
| 27 | -0.456177 | -0.266942 | -0.104377 | 0.189235 |
| 28 | -0.433709 | -0.188304 | 0.041561 | 0.245405 |
| 29 | -0.271376 | -0.262635 | -0.097602 | 0.008742 |
| 30 | -0.396428 | -0.302407 | -0.144377 | 0.094022 |
| 31 | -0.008550 | 0.023159 | 0.207580 | 0.031709 |
| 32 | -0.155789 | -0.049822 | 0.009139 | 0.105967 |
| 33 | -0.223247 | -0.300962 | -0.174355 | -0.077715 |
| 34 | -0.478701 | -0.067873 | 0.272563 | 0.410829 |
| 35 | -0.065501 | 0.024272 | -0.058807 | 0.089773 |
| mean | -0.291413 | -0.147350 | 0.020999 | 0.144063 |
| std ddof=0 | 0.145516 | 0.121442 | 0.153494 | 0.130711 |
| left lower (count) | 12 of 12 | 10 of 12 | 6 of 12 | 1 of 12 |

## age_sleeparm sleep share

Sleep decisions in the quarter divided by L/4. Quarter i is decisions [(i−1)L/4, iL/4).

| seed | overall | quarter 1 | quarter 2 | quarter 3 | quarter 4 | sleep decisions | matched dose |
|---|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.039062 | 0.031250 | 0.031250 | 0.031250 | 0.062500 | 80 | 80 |
| 25 | 0.039062 | 0.015625 | 0.031250 | 0.062500 | 0.046875 | 80 | 80 |
| 26 | 0.060547 | 0.015625 | 0.093750 | 0.062500 | 0.070312 | 124 | 128 |
| 27 | 0.042969 | 0.015625 | 0.031250 | 0.062500 | 0.062500 | 88 | 88 |
| 28 | 0.050781 | 0.015625 | 0.062500 | 0.062500 | 0.062500 | 104 | 104 |
| 29 | 0.085938 | 0.109375 | 0.078125 | 0.072266 | 0.083984 | 176 | 176 |
| 30 | 0.074219 | 0.093750 | 0.078125 | 0.078125 | 0.046875 | 152 | 152 |
| 31 | 0.070312 | 0.125000 | 0.062500 | 0.062500 | 0.031250 | 144 | 144 |
| 32 | 0.054688 | 0.031250 | 0.031250 | 0.078125 | 0.078125 | 112 | 112 |
| 33 | 0.074219 | 0.062500 | 0.062500 | 0.078125 | 0.093750 | 152 | 152 |
| 34 | 0.054688 | 0.078125 | 0.046875 | 0.046875 | 0.046875 | 112 | 112 |
| 35 | 0.058594 | 0.015625 | 0.062500 | 0.062500 | 0.093750 | 120 | 120 |
| mean ± std ddof=0 | 0.058757 ± 0.014363 | 0.050781 ± 0.039387 | 0.055990 ± 0.020629 | 0.063314 ± 0.013040 | 0.064941 ± 0.019055 |  |  |

## Sleep position relative to swaps

Mean decisions from the last earlier swap to the visit onset. The mean column pools every onset. The mean-of-seeds row averages the per-seed means.

| seed | sleep onsets | sleep mean | room onsets | room mean |
|---|---:|---:|---:|---:|
| 24 | 10 | 43.400000 | 160 | 64.106250 |
| 25 | 10 | 61.100000 | 157 | 65.560510 |
| 26 | 16 | 63.875000 | 158 | 63.430380 |
| 27 | 11 | 66.727273 | 160 | 64.343750 |
| 28 | 13 | 66.846154 | 162 | 63.851852 |
| 29 | 22 | 63.409091 | 154 | 62.629870 |
| 30 | 19 | 65.421053 | 156 | 64.551282 |
| 31 | 18 | 69.000000 | 161 | 63.720497 |
| 32 | 14 | 68.357143 | 159 | 64.408805 |
| 33 | 19 | 55.684211 | 154 | 65.259740 |
| 34 | 14 | 46.142857 | 159 | 65.081761 |
| 35 | 15 | 64.733333 | 158 | 65.405063 |
| pooled | 181 | 61.779006 | 1898 | 64.361433 |
| mean of seeds |  | 61.224676 ± 8.117964 |  | 64.362480 ± 0.841235 |

## age_sleeparm visit rewards by quarter

Exam bits saved per decision, mean over the visits whose first decision falls in the quarter. Sleep reward is the whole exam store. Room reward is that room's exam glances, travel included. A quarter with no visits of that kind is blank. The mean row's visit count is the sum across seeds. The mean reward is the mean of the per-seed means, over seeds that have at least one such visit.

| seed | quarter | sleep visits | sleep mean reward | room visits | room mean reward |
|---|---:|---:|---:|---:|---:|
| 24 | 1 | 2 | 15.684861 | 46 | 44.582585 |
| 24 | 2 | 2 | 109.250384 | 40 | 57.995039 |
| 24 | 3 | 2 | 241.444256 | 37 | 41.600666 |
| 24 | 4 | 4 | 320.171992 | 37 | 39.275260 |
| 25 | 1 | 1 | -58.927867 | 46 | 44.915146 |
| 25 | 2 | 2 | 363.518046 | 40 | 42.741362 |
| 25 | 3 | 4 | 684.519351 | 35 | 30.400038 |
| 25 | 4 | 3 | 232.399541 | 36 | 51.088064 |
| 26 | 1 | 1 | 16.961717 | 45 | 48.106463 |
| 26 | 2 | 6 | 319.347021 | 36 | 30.512141 |
| 26 | 3 | 4 | 156.913045 | 37 | 42.509384 |
| 26 | 4 | 5 | 283.997605 | 40 | 34.871020 |
| 27 | 1 | 1 | -15.141300 | 45 | 46.388297 |
| 27 | 2 | 2 | 240.297742 | 39 | 55.536670 |
| 27 | 3 | 4 | 445.024172 | 38 | 45.696040 |
| 27 | 4 | 4 | 394.985102 | 38 | 34.586746 |
| 28 | 1 | 1 | -37.176818 | 47 | 45.253856 |
| 28 | 2 | 4 | 290.556072 | 39 | 48.017284 |
| 28 | 3 | 4 | 269.039274 | 39 | 39.097275 |
| 28 | 4 | 4 | 723.421607 | 37 | 33.076521 |
| 29 | 1 | 7 | 155.167955 | 42 | 41.208809 |
| 29 | 2 | 5 | 233.897812 | 36 | 30.397238 |
| 29 | 3 | 5 | 261.752645 | 38 | 47.517688 |
| 29 | 4 | 5 | 257.597371 | 38 | 37.565537 |
| 30 | 1 | 6 | 148.579650 | 43 | 51.488296 |
| 30 | 2 | 5 | 161.284311 | 38 | 33.441369 |
| 30 | 3 | 5 | 326.915405 | 36 | 52.651940 |
| 30 | 4 | 3 | 31.396885 | 39 | 48.449427 |
| 31 | 1 | 8 | 122.240693 | 43 | 40.148530 |
| 31 | 2 | 4 | 187.702650 | 38 | 39.351717 |
| 31 | 3 | 4 | 890.202028 | 42 | 43.065916 |
| 31 | 4 | 2 | 619.478565 | 38 | 36.117840 |
| 32 | 1 | 2 | 97.295883 | 45 | 48.374281 |
| 32 | 2 | 2 | 92.334785 | 39 | 26.411492 |
| 32 | 3 | 5 | 279.796698 | 37 | 34.626203 |
| 32 | 4 | 5 | 250.224560 | 38 | 41.320260 |
| 33 | 1 | 4 | 109.039923 | 44 | 56.428984 |
| 33 | 2 | 4 | 140.847222 | 37 | 47.393738 |
| 33 | 3 | 5 | 163.449736 | 37 | 35.883398 |
| 33 | 4 | 6 | 491.533196 | 36 | 30.842327 |
| 34 | 1 | 5 | 72.721686 | 44 | 42.460864 |
| 34 | 2 | 3 | 84.756668 | 38 | 41.808460 |
| 34 | 3 | 3 | 676.451277 | 40 | 53.844868 |
| 34 | 4 | 3 | 364.279974 | 37 | 35.831105 |
| 35 | 1 | 1 | -75.445424 | 47 | 53.408102 |
| 35 | 2 | 4 | 223.805950 | 37 | 39.304851 |
| 35 | 3 | 4 | 393.385805 | 37 | 45.777624 |
| 35 | 4 | 6 | 275.112167 | 37 | 40.777975 |
| mean | 1 | 39 | 45.916747 | 537 | 46.897018 |
| mean | 2 | 43 | 203.966555 | 457 | 41.075947 |
| mean | 3 | 49 | 399.074474 | 453 | 42.722587 |
| mean | 4 | 50 | 353.716547 | 451 | 38.650174 |

## Stale replay

Share of replay draws whose glance came from a rule that had already been swapped out. age_nosleep does not draw.

| controller | seed | draws | stale | share |
|---|---:|---:|---:|---:|
| age_fixed20 | 24 | 408 | 77 | 0.188725 |
| age_fixed20 | 25 | 408 | 100 | 0.245098 |
| age_fixed20 | 26 | 408 | 71 | 0.174020 |
| age_fixed20 | 27 | 408 | 75 | 0.183824 |
| age_fixed20 | 28 | 408 | 93 | 0.227941 |
| age_fixed20 | 29 | 408 | 74 | 0.181373 |
| age_fixed20 | 30 | 408 | 66 | 0.161765 |
| age_fixed20 | 31 | 408 | 87 | 0.213235 |
| age_fixed20 | 32 | 408 | 69 | 0.169118 |
| age_fixed20 | 33 | 408 | 83 | 0.203431 |
| age_fixed20 | 34 | 408 | 83 | 0.203431 |
| age_fixed20 | 35 | 408 | 92 | 0.225490 |
| age_fixed20 | pooled | 4896 | 970 | 0.198121 |
| age_fixed20 | mean ± std ddof=0 |  |  | 0.198121 ± 0.024912 |
| age_sleeparm | 24 | 80 | 18 | 0.225000 |
| age_sleeparm | 25 | 80 | 19 | 0.237500 |
| age_sleeparm | 26 | 124 | 24 | 0.193548 |
| age_sleeparm | 27 | 88 | 19 | 0.215909 |
| age_sleeparm | 28 | 104 | 25 | 0.240385 |
| age_sleeparm | 29 | 176 | 30 | 0.170455 |
| age_sleeparm | 30 | 152 | 24 | 0.157895 |
| age_sleeparm | 31 | 144 | 26 | 0.180556 |
| age_sleeparm | 32 | 112 | 23 | 0.205357 |
| age_sleeparm | 33 | 152 | 38 | 0.250000 |
| age_sleeparm | 34 | 112 | 31 | 0.276786 |
| age_sleeparm | 35 | 120 | 34 | 0.283333 |
| age_sleeparm | pooled | 1444 | 311 | 0.215374 |
| age_sleeparm | mean ± std ddof=0 |  |  | 0.219727 ± 0.038384 |
| age_fixed_matched | 24 | 80 | 17 | 0.212500 |
| age_fixed_matched | 25 | 80 | 20 | 0.250000 |
| age_fixed_matched | 26 | 128 | 26 | 0.203125 |
| age_fixed_matched | 27 | 88 | 22 | 0.250000 |
| age_fixed_matched | 28 | 104 | 16 | 0.153846 |
| age_fixed_matched | 29 | 176 | 25 | 0.142045 |
| age_fixed_matched | 30 | 152 | 27 | 0.177632 |
| age_fixed_matched | 31 | 144 | 34 | 0.236111 |
| age_fixed_matched | 32 | 112 | 20 | 0.178571 |
| age_fixed_matched | 33 | 152 | 28 | 0.184211 |
| age_fixed_matched | 34 | 112 | 24 | 0.214286 |
| age_fixed_matched | 35 | 120 | 31 | 0.258333 |
| age_fixed_matched | pooled | 1448 | 290 | 0.200276 |
| age_fixed_matched | mean ± std ddof=0 |  |  | 0.205055 ± 0.037045 |

## Swap-finding latency

Median over swaps. Censored swaps use L minus the swap decision. Share within 128 is the fraction of swaps with a later stay at most 128 decisions after the swap.

| controller | n | found | censored | median | share within 128 |
|---|---:|---:|---:|---:|---:|
| age_nosleep | 192 | 177 | 15 | 100.000000 | 0.552083 |
| age_fixed20 | 192 | 172 | 20 | 114.500000 | 0.484375 |
| age_sleeparm | 192 | 174 | 18 | 120.000000 | 0.500000 |
| age_fixed_matched | 192 | 175 | 17 | 103.000000 | 0.515625 |

| controller | seed | n | found | censored | median | share within 128 |
|---|---:|---:|---:|---:|---:|---:|
| age_nosleep | 24 | 16 | 14 | 2 | 114.000000 | 0.500000 |
| age_nosleep | 25 | 16 | 14 | 2 | 123.000000 | 0.437500 |
| age_nosleep | 26 | 16 | 15 | 1 | 100.000000 | 0.562500 |
| age_nosleep | 27 | 16 | 14 | 2 | 110.500000 | 0.562500 |
| age_nosleep | 28 | 16 | 15 | 1 | 102.500000 | 0.500000 |
| age_nosleep | 29 | 16 | 15 | 1 | 128.000000 | 0.500000 |
| age_nosleep | 30 | 16 | 15 | 1 | 93.000000 | 0.562500 |
| age_nosleep | 31 | 16 | 15 | 1 | 81.500000 | 0.687500 |
| age_nosleep | 32 | 16 | 15 | 1 | 58.000000 | 0.625000 |
| age_nosleep | 33 | 16 | 15 | 1 | 99.000000 | 0.500000 |
| age_nosleep | 34 | 16 | 15 | 1 | 86.500000 | 0.625000 |
| age_nosleep | 35 | 16 | 15 | 1 | 118.500000 | 0.562500 |
| age_fixed20 | 24 | 16 | 15 | 1 | 140.500000 | 0.375000 |
| age_fixed20 | 25 | 16 | 15 | 1 | 84.000000 | 0.687500 |
| age_fixed20 | 26 | 16 | 14 | 2 | 72.500000 | 0.625000 |
| age_fixed20 | 27 | 16 | 15 | 1 | 112.500000 | 0.500000 |
| age_fixed20 | 28 | 16 | 13 | 3 | 95.000000 | 0.625000 |
| age_fixed20 | 29 | 16 | 15 | 1 | 43.000000 | 0.562500 |
| age_fixed20 | 30 | 16 | 15 | 1 | 125.000000 | 0.437500 |
| age_fixed20 | 31 | 16 | 14 | 2 | 152.500000 | 0.312500 |
| age_fixed20 | 32 | 16 | 13 | 3 | 127.000000 | 0.500000 |
| age_fixed20 | 33 | 16 | 14 | 2 | 78.000000 | 0.625000 |
| age_fixed20 | 34 | 16 | 14 | 2 | 125.000000 | 0.437500 |
| age_fixed20 | 35 | 16 | 15 | 1 | 187.000000 | 0.125000 |
| age_sleeparm | 24 | 16 | 14 | 2 | 124.500000 | 0.437500 |
| age_sleeparm | 25 | 16 | 15 | 1 | 130.500000 | 0.437500 |
| age_sleeparm | 26 | 16 | 15 | 1 | 140.000000 | 0.375000 |
| age_sleeparm | 27 | 16 | 14 | 2 | 148.000000 | 0.250000 |
| age_sleeparm | 28 | 16 | 15 | 1 | 116.000000 | 0.500000 |
| age_sleeparm | 29 | 16 | 15 | 1 | 104.000000 | 0.625000 |
| age_sleeparm | 30 | 16 | 14 | 2 | 97.500000 | 0.562500 |
| age_sleeparm | 31 | 16 | 15 | 1 | 58.000000 | 0.625000 |
| age_sleeparm | 32 | 16 | 15 | 1 | 69.500000 | 0.687500 |
| age_sleeparm | 33 | 16 | 14 | 2 | 53.500000 | 0.625000 |
| age_sleeparm | 34 | 16 | 14 | 2 | 115.000000 | 0.500000 |
| age_sleeparm | 35 | 16 | 14 | 2 | 135.000000 | 0.375000 |
| age_fixed_matched | 24 | 16 | 15 | 1 | 102.000000 | 0.500000 |
| age_fixed_matched | 25 | 16 | 14 | 2 | 124.500000 | 0.437500 |
| age_fixed_matched | 26 | 16 | 14 | 2 | 123.000000 | 0.437500 |
| age_fixed_matched | 27 | 16 | 15 | 1 | 157.000000 | 0.375000 |
| age_fixed_matched | 28 | 16 | 15 | 1 | 116.500000 | 0.437500 |
| age_fixed_matched | 29 | 16 | 14 | 2 | 57.000000 | 0.562500 |
| age_fixed_matched | 30 | 16 | 14 | 2 | 86.500000 | 0.625000 |
| age_fixed_matched | 31 | 16 | 15 | 1 | 85.000000 | 0.687500 |
| age_fixed_matched | 32 | 16 | 15 | 1 | 83.000000 | 0.687500 |
| age_fixed_matched | 33 | 16 | 15 | 1 | 83.000000 | 0.500000 |
| age_fixed_matched | 34 | 16 | 15 | 1 | 48.000000 | 0.625000 |
| age_fixed_matched | 35 | 16 | 14 | 2 | 151.500000 | 0.312500 |

| controller | seed | decision | room | k | next dwell | latency | censored | within 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| age_nosleep | 24 | 128 | E110b | 0 | 318 | 190 | false | false |
| age_nosleep | 24 | 256 | C7a | 1 | 259 | 3 | false | true |
| age_nosleep | 24 | 384 | C7b | 2 | 662 | 278 | false | false |
| age_nosleep | 24 | 512 | E110a | 3 | 634 | 122 | false | true |
| age_nosleep | 24 | 640 | E30b | 4 | 925 | 285 | false | false |
| age_nosleep | 24 | 768 | E30b | 5 | 925 | 157 | false | false |
| age_nosleep | 24 | 896 | C7b | 6 | 996 | 100 | false | true |
| age_nosleep | 24 | 1024 | E30a | 7 | 1226 | 202 | false | false |
| age_nosleep | 24 | 1152 | Gb | 8 | 1183 | 31 | false | true |
| age_nosleep | 24 | 1280 | C1a | 9 | 1365 | 85 | false | true |
| age_nosleep | 24 | 1408 | M37a | 10 | 1512 | 104 | false | true |
| age_nosleep | 24 | 1536 | E110b | 11 | 1770 | 234 | false | false |
| age_nosleep | 24 | 1664 | E110b | 12 | 1770 | 106 | false | true |
| age_nosleep | 24 | 1792 | M7a | 13 | 1885 | 93 | false | true |
| age_nosleep | 24 | 1920 | M7a | 14 |  | 128 | true | false |
| age_nosleep | 24 | 2048 | C7b | 15 |  | 0 | true | false |
| age_nosleep | 25 | 128 | Ga | 0 | 227 | 99 | false | true |
| age_nosleep | 25 | 256 | Ga | 1 | 389 | 133 | false | false |
| age_nosleep | 25 | 384 | C1b | 2 | 535 | 151 | false | false |
| age_nosleep | 25 | 512 | E110a | 3 | 774 | 262 | false | false |
| age_nosleep | 25 | 640 | E30b | 4 | 758 | 118 | false | true |
| age_nosleep | 25 | 768 | Gb | 5 | 820 | 52 | false | true |
| age_nosleep | 25 | 896 | C7a | 6 | 1127 | 231 | false | false |
| age_nosleep | 25 | 1024 | M37a | 7 | 1187 | 163 | false | false |
| age_nosleep | 25 | 1152 | M37b | 8 | 1162 | 10 | false | true |
| age_nosleep | 25 | 1280 | C7b | 9 | 1307 | 27 | false | true |
| age_nosleep | 25 | 1408 | E30a | 10 | 1514 | 106 | false | true |
| age_nosleep | 25 | 1536 | M37a | 11 | 1717 | 181 | false | false |
| age_nosleep | 25 | 1664 | Ga | 12 | 1683 | 19 | false | true |
| age_nosleep | 25 | 1792 | M7b | 13 | 1984 | 192 | false | false |
| age_nosleep | 25 | 1920 | E30b | 14 |  | 128 | true | false |
| age_nosleep | 25 | 2048 | M7a | 15 |  | 0 | true | false |
| age_nosleep | 26 | 128 | Gb | 0 | 129 | 1 | false | true |
| age_nosleep | 26 | 256 | M7a | 1 | 271 | 15 | false | true |
| age_nosleep | 26 | 384 | C7a | 2 | 493 | 109 | false | true |
| age_nosleep | 26 | 512 | C1a | 3 | 709 | 197 | false | false |
| age_nosleep | 26 | 640 | C1b | 4 | 731 | 91 | false | true |
| age_nosleep | 26 | 768 | C7a | 5 | 949 | 181 | false | false |
| age_nosleep | 26 | 896 | M7a | 6 | 965 | 69 | false | true |
| age_nosleep | 26 | 1024 | Gb | 7 | 1225 | 201 | false | false |
| age_nosleep | 26 | 1152 | E110a | 8 | 1182 | 30 | false | true |
| age_nosleep | 26 | 1280 | M37b | 9 | 1286 | 6 | false | true |
| age_nosleep | 26 | 1408 | C7a | 10 | 1451 | 43 | false | true |
| age_nosleep | 26 | 1536 | M37b | 11 | 1732 | 196 | false | false |
| age_nosleep | 26 | 1664 | Ga | 12 | 1890 | 226 | false | false |
| age_nosleep | 26 | 1792 | Gb | 13 | 1939 | 147 | false | false |
| age_nosleep | 26 | 1920 | C7a | 14 | 2041 | 121 | false | true |
| age_nosleep | 26 | 2048 | E110b | 15 |  | 0 | true | false |
| age_nosleep | 27 | 128 | M7b | 0 | 340 | 212 | false | false |
| age_nosleep | 27 | 256 | C7a | 1 | 326 | 70 | false | true |
| age_nosleep | 27 | 384 | Gb | 2 | 385 | 1 | false | true |
| age_nosleep | 27 | 512 | E30a | 3 | 682 | 170 | false | false |
| age_nosleep | 27 | 640 | C7a | 4 | 884 | 244 | false | false |
| age_nosleep | 27 | 768 | C7a | 5 | 884 | 116 | false | true |
| age_nosleep | 27 | 896 | C7b | 6 | 996 | 100 | false | true |
| age_nosleep | 27 | 1024 | M7b | 7 | 1152 | 128 | false | true |
| age_nosleep | 27 | 1152 | C7a | 8 | 1166 | 14 | false | true |
| age_nosleep | 27 | 1280 | M37a | 9 | 1281 | 1 | false | true |
| age_nosleep | 27 | 1408 | E30b | 10 | 1562 | 154 | false | false |
| age_nosleep | 27 | 1536 | C7a | 11 | 1674 | 138 | false | false |
| age_nosleep | 27 | 1664 | E110b | 12 | 1769 | 105 | false | true |
| age_nosleep | 27 | 1792 | Ga | 13 | 1866 | 74 | false | true |
| age_nosleep | 27 | 1920 | C1a | 14 |  | 128 | true | false |
| age_nosleep | 27 | 2048 | M37a | 15 |  | 0 | true | false |
| age_nosleep | 28 | 128 | E110b | 0 | 412 | 284 | false | false |
| age_nosleep | 28 | 256 | M37b | 1 | 304 | 48 | false | true |
| age_nosleep | 28 | 384 | Gb | 2 | 399 | 15 | false | true |
| age_nosleep | 28 | 512 | E110b | 3 | 686 | 174 | false | false |
| age_nosleep | 28 | 640 | E30a | 4 | 671 | 31 | false | true |
| age_nosleep | 28 | 768 | M37b | 5 | 848 | 80 | false | true |
| age_nosleep | 28 | 896 | C1b | 6 | 1187 | 291 | false | false |
| age_nosleep | 28 | 1024 | C1b | 7 | 1187 | 163 | false | false |
| age_nosleep | 28 | 1152 | E110a | 8 | 1310 | 158 | false | false |
| age_nosleep | 28 | 1280 | E110b | 9 | 1528 | 248 | false | false |
| age_nosleep | 28 | 1408 | Ga | 10 | 1422 | 14 | false | true |
| age_nosleep | 28 | 1536 | Ga | 11 | 1618 | 82 | false | true |
| age_nosleep | 28 | 1664 | M37a | 12 | 1795 | 131 | false | false |
| age_nosleep | 28 | 1792 | E30a | 13 | 1913 | 121 | false | true |
| age_nosleep | 28 | 1920 | C7b | 14 | 2004 | 84 | false | true |
| age_nosleep | 28 | 2048 | C7b | 15 |  | 0 | true | false |
| age_nosleep | 29 | 128 | Ga | 0 | 129 | 1 | false | true |
| age_nosleep | 29 | 256 | M37a | 1 | 440 | 184 | false | false |
| age_nosleep | 29 | 384 | C1a | 2 | 637 | 253 | false | false |
| age_nosleep | 29 | 512 | C1b | 3 | 650 | 138 | false | false |
| age_nosleep | 29 | 640 | C7a | 4 | 768 | 128 | false | true |
| age_nosleep | 29 | 768 | C1b | 5 | 938 | 170 | false | false |
| age_nosleep | 29 | 896 | Ga | 6 | 1053 | 157 | false | false |
| age_nosleep | 29 | 1024 | E30a | 7 | 1062 | 38 | false | true |
| age_nosleep | 29 | 1152 | E30a | 8 | 1356 | 204 | false | false |
| age_nosleep | 29 | 1280 | C7a | 9 | 1380 | 100 | false | true |
| age_nosleep | 29 | 1408 | C7b | 10 | 1536 | 128 | false | true |
| age_nosleep | 29 | 1536 | E30b | 11 | 1596 | 60 | false | true |
| age_nosleep | 29 | 1664 | C7b | 12 | 1830 | 166 | false | false |
| age_nosleep | 29 | 1792 | E30b | 13 | 1854 | 62 | false | true |
| age_nosleep | 29 | 1920 | C7a | 14 | 1922 | 2 | false | true |
| age_nosleep | 29 | 2048 | M37b | 15 |  | 0 | true | false |
| age_nosleep | 30 | 128 | E30a | 0 | 145 | 17 | false | true |
| age_nosleep | 30 | 256 | E30a | 1 | 411 | 155 | false | false |
| age_nosleep | 30 | 384 | E30a | 2 | 411 | 27 | false | true |
| age_nosleep | 30 | 512 | E110b | 3 | 629 | 117 | false | true |
| age_nosleep | 30 | 640 | E30b | 4 | 805 | 165 | false | false |
| age_nosleep | 30 | 768 | C1b | 5 | 919 | 151 | false | false |
| age_nosleep | 30 | 896 | C7a | 6 | 984 | 88 | false | true |
| age_nosleep | 30 | 1024 | E30a | 7 | 1105 | 81 | false | true |
| age_nosleep | 30 | 1152 | M7a | 8 | 1298 | 146 | false | false |
| age_nosleep | 30 | 1280 | M37a | 9 | 1416 | 136 | false | false |
| age_nosleep | 30 | 1408 | E30b | 10 | 1481 | 73 | false | true |
| age_nosleep | 30 | 1536 | E110b | 11 | 1571 | 35 | false | true |
| age_nosleep | 30 | 1664 | E30a | 12 | 1911 | 247 | false | false |
| age_nosleep | 30 | 1792 | E110b | 13 | 1793 | 1 | false | true |
| age_nosleep | 30 | 1920 | C1a | 14 | 2018 | 98 | false | true |
| age_nosleep | 30 | 2048 | C1b | 15 |  | 0 | true | false |
| age_nosleep | 31 | 128 | E110a | 0 | 129 | 1 | false | true |
| age_nosleep | 31 | 256 | C7b | 1 | 332 | 76 | false | true |
| age_nosleep | 31 | 384 | M7b | 2 | 472 | 88 | false | true |
| age_nosleep | 31 | 512 | E30a | 3 | 561 | 49 | false | true |
| age_nosleep | 31 | 640 | C7b | 4 | 886 | 246 | false | false |
| age_nosleep | 31 | 768 | E30b | 5 | 947 | 179 | false | false |
| age_nosleep | 31 | 896 | M37b | 6 | 983 | 87 | false | true |
| age_nosleep | 31 | 1024 | C1a | 7 | 1095 | 71 | false | true |
| age_nosleep | 31 | 1152 | Gb | 8 | 1282 | 130 | false | false |
| age_nosleep | 31 | 1280 | E30a | 9 | 1485 | 205 | false | false |
| age_nosleep | 31 | 1408 | Ga | 10 | 1454 | 46 | false | true |
| age_nosleep | 31 | 1536 | M7b | 11 | 1537 | 1 | false | true |
| age_nosleep | 31 | 1664 | E110a | 12 | 1755 | 91 | false | true |
| age_nosleep | 31 | 1792 | C7a | 13 | 1827 | 35 | false | true |
| age_nosleep | 31 | 1920 | M7a | 14 | 2014 | 94 | false | true |
| age_nosleep | 31 | 2048 | M7a | 15 |  | 0 | true | false |
| age_nosleep | 32 | 128 | E110a | 0 | 165 | 37 | false | true |
| age_nosleep | 32 | 256 | M7b | 1 | 324 | 68 | false | true |
| age_nosleep | 32 | 384 | M7b | 2 | 610 | 226 | false | false |
| age_nosleep | 32 | 512 | Ga | 3 | 538 | 26 | false | true |
| age_nosleep | 32 | 640 | M37b | 4 | 641 | 1 | false | true |
| age_nosleep | 32 | 768 | C1a | 5 | 899 | 131 | false | false |
| age_nosleep | 32 | 896 | C7b | 6 | 946 | 50 | false | true |
| age_nosleep | 32 | 1024 | E30a | 7 | 1234 | 210 | false | false |
| age_nosleep | 32 | 1152 | E30a | 8 | 1234 | 82 | false | true |
| age_nosleep | 32 | 1280 | E30a | 9 | 1552 | 272 | false | false |
| age_nosleep | 32 | 1408 | C7b | 10 | 1436 | 28 | false | true |
| age_nosleep | 32 | 1536 | E30a | 11 | 1552 | 16 | false | true |
| age_nosleep | 32 | 1664 | C7b | 12 | 1724 | 60 | false | true |
| age_nosleep | 32 | 1792 | M7a | 13 | 1947 | 155 | false | false |
| age_nosleep | 32 | 1920 | C7b | 14 | 1976 | 56 | false | true |
| age_nosleep | 32 | 2048 | M37b | 15 |  | 0 | true | false |
| age_nosleep | 33 | 128 | E30b | 0 | 304 | 176 | false | false |
| age_nosleep | 33 | 256 | M37a | 1 | 262 | 6 | false | true |
| age_nosleep | 33 | 384 | Gb | 2 | 536 | 152 | false | false |
| age_nosleep | 33 | 512 | E110b | 3 | 550 | 38 | false | true |
| age_nosleep | 33 | 640 | M7a | 4 | 787 | 147 | false | false |
| age_nosleep | 33 | 768 | E30a | 5 | 981 | 213 | false | false |
| age_nosleep | 33 | 896 | C7b | 6 | 1092 | 196 | false | false |
| age_nosleep | 33 | 1024 | C1a | 7 | 1255 | 231 | false | false |
| age_nosleep | 33 | 1152 | Ga | 8 | 1305 | 153 | false | false |
| age_nosleep | 33 | 1280 | Ga | 9 | 1305 | 25 | false | true |
| age_nosleep | 33 | 1408 | C1a | 10 | 1489 | 81 | false | true |
| age_nosleep | 33 | 1536 | Ga | 11 | 1539 | 3 | false | true |
| age_nosleep | 33 | 1664 | M7b | 12 | 1781 | 117 | false | true |
| age_nosleep | 33 | 1792 | C7b | 13 | 1808 | 16 | false | true |
| age_nosleep | 33 | 1920 | M37a | 14 | 1952 | 32 | false | true |
| age_nosleep | 33 | 2048 | Gb | 15 |  | 0 | true | false |
| age_nosleep | 34 | 128 | M37b | 0 | 310 | 182 | false | false |
| age_nosleep | 34 | 256 | C1a | 1 | 285 | 29 | false | true |
| age_nosleep | 34 | 384 | Ga | 2 | 487 | 103 | false | true |
| age_nosleep | 34 | 512 | M37a | 3 | 552 | 40 | false | true |
| age_nosleep | 34 | 640 | C1a | 4 | 815 | 175 | false | false |
| age_nosleep | 34 | 768 | E110a | 5 | 827 | 59 | false | true |
| age_nosleep | 34 | 896 | C7a | 6 | 943 | 47 | false | true |
| age_nosleep | 34 | 1024 | C1a | 7 | 1095 | 71 | false | true |
| age_nosleep | 34 | 1152 | Gb | 8 | 1247 | 95 | false | true |
| age_nosleep | 34 | 1280 | C7b | 9 | 1389 | 109 | false | true |
| age_nosleep | 34 | 1408 | C7b | 10 | 1633 | 225 | false | false |
| age_nosleep | 34 | 1536 | E30a | 11 | 1669 | 133 | false | false |
| age_nosleep | 34 | 1664 | E30b | 12 | 1814 | 150 | false | false |
| age_nosleep | 34 | 1792 | Ga | 13 | 1803 | 11 | false | true |
| age_nosleep | 34 | 1920 | E30b | 14 | 1998 | 78 | false | true |
| age_nosleep | 34 | 2048 | M7b | 15 |  | 0 | true | false |
| age_nosleep | 35 | 128 | E30a | 0 | 428 | 300 | false | false |
| age_nosleep | 35 | 256 | C7a | 1 | 378 | 122 | false | true |
| age_nosleep | 35 | 384 | M7a | 2 | 529 | 145 | false | false |
| age_nosleep | 35 | 512 | C7b | 3 | 584 | 72 | false | true |
| age_nosleep | 35 | 640 | C1b | 4 | 863 | 223 | false | false |
| age_nosleep | 35 | 768 | E110b | 5 | 814 | 46 | false | true |
| age_nosleep | 35 | 896 | M37b | 6 | 932 | 36 | false | true |
| age_nosleep | 35 | 1024 | M7a | 7 | 1205 | 181 | false | false |
| age_nosleep | 35 | 1152 | C7a | 8 | 1220 | 68 | false | true |
| age_nosleep | 35 | 1280 | C1b | 9 | 1405 | 125 | false | true |
| age_nosleep | 35 | 1408 | M7b | 10 | 1523 | 115 | false | true |
| age_nosleep | 35 | 1536 | Gb | 11 | 1595 | 59 | false | true |
| age_nosleep | 35 | 1664 | E30a | 12 | 1918 | 254 | false | false |
| age_nosleep | 35 | 1792 | C1b | 13 | 1983 | 191 | false | false |
| age_nosleep | 35 | 1920 | E30b | 14 | 2003 | 83 | false | true |
| age_nosleep | 35 | 2048 | M37b | 15 |  | 0 | true | false |
| age_fixed20 | 24 | 128 | E110b | 0 | 342 | 214 | false | false |
| age_fixed20 | 24 | 256 | C7a | 1 | 327 | 71 | false | true |
| age_fixed20 | 24 | 384 | C7b | 2 | 392 | 8 | false | true |
| age_fixed20 | 24 | 512 | E110a | 3 | 838 | 326 | false | false |
| age_fixed20 | 24 | 640 | E30b | 4 | 773 | 133 | false | false |
| age_fixed20 | 24 | 768 | E30b | 5 | 773 | 5 | false | true |
| age_fixed20 | 24 | 896 | C7b | 6 | 1062 | 166 | false | false |
| age_fixed20 | 24 | 1024 | E30a | 7 | 1172 | 148 | false | false |
| age_fixed20 | 24 | 1152 | Gb | 8 | 1231 | 79 | false | true |
| age_fixed20 | 24 | 1280 | C1a | 9 | 1481 | 201 | false | false |
| age_fixed20 | 24 | 1408 | M37a | 10 | 1698 | 290 | false | false |
| age_fixed20 | 24 | 1536 | E110b | 11 | 1580 | 44 | false | true |
| age_fixed20 | 24 | 1664 | E110b | 12 | 1890 | 226 | false | false |
| age_fixed20 | 24 | 1792 | M7a | 13 | 1963 | 171 | false | false |
| age_fixed20 | 24 | 1920 | M7a | 14 | 1963 | 43 | false | true |
| age_fixed20 | 24 | 2048 | C7b | 15 |  | 0 | true | false |
| age_fixed20 | 25 | 128 | Ga | 0 | 133 | 5 | false | true |
| age_fixed20 | 25 | 256 | Ga | 1 | 275 | 19 | false | true |
| age_fixed20 | 25 | 384 | C1b | 2 | 415 | 31 | false | true |
| age_fixed20 | 25 | 512 | E110a | 3 | 606 | 94 | false | true |
| age_fixed20 | 25 | 640 | E30b | 4 | 662 | 22 | false | true |
| age_fixed20 | 25 | 768 | Gb | 5 | 808 | 40 | false | true |
| age_fixed20 | 25 | 896 | C7a | 6 | 1023 | 127 | false | true |
| age_fixed20 | 25 | 1024 | M37a | 7 | 1145 | 121 | false | true |
| age_fixed20 | 25 | 1152 | M37b | 8 | 1474 | 322 | false | false |
| age_fixed20 | 25 | 1280 | C7b | 9 | 1539 | 259 | false | false |
| age_fixed20 | 25 | 1408 | E30a | 10 | 1522 | 114 | false | true |
| age_fixed20 | 25 | 1536 | M37a | 11 | 1813 | 277 | false | false |
| age_fixed20 | 25 | 1664 | Ga | 12 | 1837 | 173 | false | false |
| age_fixed20 | 25 | 1792 | M7b | 13 | 1866 | 74 | false | true |
| age_fixed20 | 25 | 1920 | E30b | 14 | 1928 | 8 | false | true |
| age_fixed20 | 25 | 2048 | M7a | 15 |  | 0 | true | false |
| age_fixed20 | 26 | 128 | Gb | 0 | 151 | 23 | false | true |
| age_fixed20 | 26 | 256 | M7a | 1 | 309 | 53 | false | true |
| age_fixed20 | 26 | 384 | C7a | 2 | 627 | 243 | false | false |
| age_fixed20 | 26 | 512 | C1a | 3 | 849 | 337 | false | false |
| age_fixed20 | 26 | 640 | C1b | 4 | 973 | 333 | false | false |
| age_fixed20 | 26 | 768 | C7a | 5 | 893 | 125 | false | true |
| age_fixed20 | 26 | 896 | M7a | 6 | 949 | 53 | false | true |
| age_fixed20 | 26 | 1024 | Gb | 7 | 1035 | 11 | false | true |
| age_fixed20 | 26 | 1152 | E110a | 8 | 1414 | 262 | false | false |
| age_fixed20 | 26 | 1280 | M37b | 9 | 1372 | 92 | false | true |
| age_fixed20 | 26 | 1408 | C7a | 10 | 1455 | 47 | false | true |
| age_fixed20 | 26 | 1536 | M37b | 11 | 1634 | 98 | false | true |
| age_fixed20 | 26 | 1664 | Ga | 12 | 1682 | 18 | false | true |
| age_fixed20 | 26 | 1792 | Gb | 13 | 1807 | 15 | false | true |
| age_fixed20 | 26 | 1920 | C7a | 14 |  | 128 | true | false |
| age_fixed20 | 26 | 2048 | E110b | 15 |  | 0 | true | false |
| age_fixed20 | 27 | 128 | M7b | 0 | 334 | 206 | false | false |
| age_fixed20 | 27 | 256 | C7a | 1 | 378 | 122 | false | true |
| age_fixed20 | 27 | 384 | Gb | 2 | 487 | 103 | false | true |
| age_fixed20 | 27 | 512 | E30a | 3 | 772 | 260 | false | false |
| age_fixed20 | 27 | 640 | C7a | 4 | 710 | 70 | false | true |
| age_fixed20 | 27 | 768 | C7a | 5 | 1032 | 264 | false | false |
| age_fixed20 | 27 | 896 | C7b | 6 | 897 | 1 | false | true |
| age_fixed20 | 27 | 1024 | M7b | 7 | 1332 | 308 | false | false |
| age_fixed20 | 27 | 1152 | C7a | 8 | 1364 | 212 | false | false |
| age_fixed20 | 27 | 1280 | M37a | 9 | 1341 | 61 | false | true |
| age_fixed20 | 27 | 1408 | E30b | 10 | 1506 | 98 | false | true |
| age_fixed20 | 27 | 1536 | C7a | 11 | 1764 | 228 | false | false |
| age_fixed20 | 27 | 1664 | E110b | 12 | 1747 | 83 | false | true |
| age_fixed20 | 27 | 1792 | Ga | 13 | 2014 | 222 | false | false |
| age_fixed20 | 27 | 1920 | C1a | 14 | 1974 | 54 | false | true |
| age_fixed20 | 27 | 2048 | M37a | 15 |  | 0 | true | false |
| age_fixed20 | 28 | 128 | E110b | 0 | 129 | 1 | false | true |
| age_fixed20 | 28 | 256 | M37b | 1 | 302 | 46 | false | true |
| age_fixed20 | 28 | 384 | Gb | 2 | 487 | 103 | false | true |
| age_fixed20 | 28 | 512 | E110b | 3 | 792 | 280 | false | false |
| age_fixed20 | 28 | 640 | E30a | 4 | 727 | 87 | false | true |
| age_fixed20 | 28 | 768 | M37b | 5 | 876 | 108 | false | true |
| age_fixed20 | 28 | 896 | C1b | 6 | 1023 | 127 | false | true |
| age_fixed20 | 28 | 1024 | C1b | 7 | 1025 | 1 | false | true |
| age_fixed20 | 28 | 1152 | E110a | 8 | 1222 | 70 | false | true |
| age_fixed20 | 28 | 1280 | E110b | 9 | 1494 | 214 | false | false |
| age_fixed20 | 28 | 1408 | Ga | 10 | 1582 | 174 | false | false |
| age_fixed20 | 28 | 1536 | Ga | 11 | 1582 | 46 | false | true |
| age_fixed20 | 28 | 1664 | M37a | 12 | 1701 | 37 | false | true |
| age_fixed20 | 28 | 1792 | E30a | 13 |  | 256 | true | false |
| age_fixed20 | 28 | 1920 | C7b | 14 |  | 128 | true | false |
| age_fixed20 | 28 | 2048 | C7b | 15 |  | 0 | true | false |
| age_fixed20 | 29 | 128 | Ga | 0 | 151 | 23 | false | true |
| age_fixed20 | 29 | 256 | M37a | 1 | 298 | 42 | false | true |
| age_fixed20 | 29 | 384 | C1a | 2 | 391 | 7 | false | true |
| age_fixed20 | 29 | 512 | C1b | 3 | 812 | 300 | false | false |
| age_fixed20 | 29 | 640 | C7a | 4 | 982 | 342 | false | false |
| age_fixed20 | 29 | 768 | C1b | 5 | 812 | 44 | false | true |
| age_fixed20 | 29 | 896 | Ga | 6 | 927 | 31 | false | true |
| age_fixed20 | 29 | 1024 | E30a | 7 | 1288 | 264 | false | false |
| age_fixed20 | 29 | 1152 | E30a | 8 | 1288 | 136 | false | false |
| age_fixed20 | 29 | 1280 | C7a | 9 | 1304 | 24 | false | true |
| age_fixed20 | 29 | 1408 | C7b | 10 | 1554 | 146 | false | false |
| age_fixed20 | 29 | 1536 | E30b | 11 | 1564 | 28 | false | true |
| age_fixed20 | 29 | 1664 | C7b | 12 | 1878 | 214 | false | false |
| age_fixed20 | 29 | 1792 | E30b | 13 | 1868 | 76 | false | true |
| age_fixed20 | 29 | 1920 | C7a | 14 | 1924 | 4 | false | true |
| age_fixed20 | 29 | 2048 | M37b | 15 |  | 0 | true | false |
| age_fixed20 | 30 | 128 | E30a | 0 | 177 | 49 | false | true |
| age_fixed20 | 30 | 256 | E30a | 1 | 499 | 243 | false | false |
| age_fixed20 | 30 | 384 | E30a | 2 | 499 | 115 | false | true |
| age_fixed20 | 30 | 512 | E110b | 3 | 733 | 221 | false | false |
| age_fixed20 | 30 | 640 | E30b | 4 | 775 | 135 | false | false |
| age_fixed20 | 30 | 768 | C1b | 5 | 913 | 145 | false | false |
| age_fixed20 | 30 | 896 | C7a | 6 | 902 | 6 | false | true |
| age_fixed20 | 30 | 1024 | E30a | 7 | 1297 | 273 | false | false |
| age_fixed20 | 30 | 1152 | M7a | 8 | 1448 | 296 | false | false |
| age_fixed20 | 30 | 1280 | M37a | 9 | 1380 | 100 | false | true |
| age_fixed20 | 30 | 1408 | E30b | 10 | 1635 | 227 | false | false |
| age_fixed20 | 30 | 1536 | E110b | 11 | 1613 | 77 | false | true |
| age_fixed20 | 30 | 1664 | E30a | 12 | 1665 | 1 | false | true |
| age_fixed20 | 30 | 1792 | E110b | 13 | 1941 | 149 | false | false |
| age_fixed20 | 30 | 1920 | C1a | 14 | 1924 | 4 | false | true |
| age_fixed20 | 30 | 2048 | C1b | 15 |  | 0 | true | false |
| age_fixed20 | 31 | 128 | E110a | 0 | 151 | 23 | false | true |
| age_fixed20 | 31 | 256 | C7b | 1 | 418 | 162 | false | false |
| age_fixed20 | 31 | 384 | M7b | 2 | 562 | 178 | false | false |
| age_fixed20 | 31 | 512 | E30a | 3 | 655 | 143 | false | false |
| age_fixed20 | 31 | 640 | C7b | 4 | 732 | 92 | false | true |
| age_fixed20 | 31 | 768 | E30b | 5 | 813 | 45 | false | true |
| age_fixed20 | 31 | 896 | M37b | 6 | 1113 | 217 | false | false |
| age_fixed20 | 31 | 1024 | C1a | 7 | 1301 | 277 | false | false |
| age_fixed20 | 31 | 1152 | Gb | 8 | 1368 | 216 | false | false |
| age_fixed20 | 31 | 1280 | E30a | 9 | 1479 | 199 | false | false |
| age_fixed20 | 31 | 1408 | Ga | 10 | 1498 | 90 | false | true |
| age_fixed20 | 31 | 1536 | M7b | 11 | 1568 | 32 | false | true |
| age_fixed20 | 31 | 1664 | E110a | 12 | 1953 | 289 | false | false |
| age_fixed20 | 31 | 1792 | C7a | 13 | 2047 | 255 | false | false |
| age_fixed20 | 31 | 1920 | M7a | 14 |  | 128 | true | false |
| age_fixed20 | 31 | 2048 | M7a | 15 |  | 0 | true | false |
| age_fixed20 | 32 | 128 | E110a | 0 | 205 | 77 | false | true |
| age_fixed20 | 32 | 256 | M7b | 1 | 510 | 254 | false | false |
| age_fixed20 | 32 | 384 | M7b | 2 | 510 | 126 | false | true |
| age_fixed20 | 32 | 512 | Ga | 3 | 666 | 154 | false | false |
| age_fixed20 | 32 | 640 | M37b | 4 | 795 | 155 | false | false |
| age_fixed20 | 32 | 768 | C1a | 5 | 831 | 63 | false | true |
| age_fixed20 | 32 | 896 | C7b | 6 | 1148 | 252 | false | false |
| age_fixed20 | 32 | 1024 | E30a | 7 | 1280 | 256 | false | false |
| age_fixed20 | 32 | 1152 | E30a | 8 | 1280 | 128 | false | true |
| age_fixed20 | 32 | 1280 | E30a | 9 | 1281 | 1 | false | true |
| age_fixed20 | 32 | 1408 | C7b | 10 | 1480 | 72 | false | true |
| age_fixed20 | 32 | 1536 | E30a | 11 | 1632 | 96 | false | true |
| age_fixed20 | 32 | 1664 | C7b | 12 | 1762 | 98 | false | true |
| age_fixed20 | 32 | 1792 | M7a | 13 |  | 256 | true | false |
| age_fixed20 | 32 | 1920 | C7b | 14 |  | 128 | true | false |
| age_fixed20 | 32 | 2048 | M37b | 15 |  | 0 | true | false |
| age_fixed20 | 33 | 128 | E30b | 0 | 340 | 212 | false | false |
| age_fixed20 | 33 | 256 | M37a | 1 | 278 | 22 | false | true |
| age_fixed20 | 33 | 384 | Gb | 2 | 470 | 86 | false | true |
| age_fixed20 | 33 | 512 | E110b | 3 | 582 | 70 | false | true |
| age_fixed20 | 33 | 640 | M7a | 4 | 873 | 233 | false | false |
| age_fixed20 | 33 | 768 | E30a | 5 | 827 | 59 | false | true |
| age_fixed20 | 33 | 896 | C7b | 6 | 1006 | 110 | false | true |
| age_fixed20 | 33 | 1024 | C1a | 7 | 1215 | 191 | false | false |
| age_fixed20 | 33 | 1152 | Ga | 8 | 1299 | 147 | false | false |
| age_fixed20 | 33 | 1280 | Ga | 9 | 1299 | 19 | false | true |
| age_fixed20 | 33 | 1408 | C1a | 10 | 1495 | 87 | false | true |
| age_fixed20 | 33 | 1536 | Ga | 11 | 1555 | 19 | false | true |
| age_fixed20 | 33 | 1664 | M7b | 12 | 1671 | 7 | false | true |
| age_fixed20 | 33 | 1792 | C7b | 13 | 1854 | 62 | false | true |
| age_fixed20 | 33 | 1920 | M37a | 14 |  | 128 | true | false |
| age_fixed20 | 33 | 2048 | Gb | 15 |  | 0 | true | false |
| age_fixed20 | 34 | 128 | M37b | 0 | 129 | 1 | false | true |
| age_fixed20 | 34 | 256 | C1a | 1 | 353 | 97 | false | true |
| age_fixed20 | 34 | 384 | Ga | 2 | 469 | 85 | false | true |
| age_fixed20 | 34 | 512 | M37a | 3 | 634 | 122 | false | true |
| age_fixed20 | 34 | 640 | C1a | 4 | 953 | 313 | false | false |
| age_fixed20 | 34 | 768 | E110a | 5 | 993 | 225 | false | false |
| age_fixed20 | 34 | 896 | C7a | 6 | 1107 | 211 | false | false |
| age_fixed20 | 34 | 1024 | C1a | 7 | 1217 | 193 | false | false |
| age_fixed20 | 34 | 1152 | Gb | 8 | 1203 | 51 | false | true |
| age_fixed20 | 34 | 1280 | C7b | 9 | 1389 | 109 | false | true |
| age_fixed20 | 34 | 1408 | C7b | 10 | 1741 | 333 | false | false |
| age_fixed20 | 34 | 1536 | E30a | 11 | 1595 | 59 | false | true |
| age_fixed20 | 34 | 1664 | E30b | 12 | 1814 | 150 | false | false |
| age_fixed20 | 34 | 1792 | Ga | 13 | 1947 | 155 | false | false |
| age_fixed20 | 34 | 1920 | E30b | 14 |  | 128 | true | false |
| age_fixed20 | 34 | 2048 | M7b | 15 |  | 0 | true | false |
| age_fixed20 | 35 | 128 | E30a | 0 | 492 | 364 | false | false |
| age_fixed20 | 35 | 256 | C7a | 1 | 470 | 214 | false | false |
| age_fixed20 | 35 | 384 | M7a | 2 | 663 | 279 | false | false |
| age_fixed20 | 35 | 512 | C7b | 3 | 696 | 184 | false | false |
| age_fixed20 | 35 | 640 | C1b | 4 | 853 | 213 | false | false |
| age_fixed20 | 35 | 768 | E110b | 5 | 1018 | 250 | false | false |
| age_fixed20 | 35 | 896 | M37b | 6 | 1052 | 156 | false | false |
| age_fixed20 | 35 | 1024 | M7a | 7 | 1155 | 131 | false | false |
| age_fixed20 | 35 | 1152 | C7a | 8 | 1502 | 350 | false | false |
| age_fixed20 | 35 | 1280 | C1b | 9 | 1465 | 185 | false | false |
| age_fixed20 | 35 | 1408 | M7b | 10 | 1693 | 285 | false | false |
| age_fixed20 | 35 | 1536 | Gb | 11 | 1725 | 189 | false | false |
| age_fixed20 | 35 | 1664 | E30a | 12 | 1846 | 182 | false | false |
| age_fixed20 | 35 | 1792 | C1b | 13 | 1797 | 5 | false | true |
| age_fixed20 | 35 | 1920 | E30b | 14 | 1923 | 3 | false | true |
| age_fixed20 | 35 | 2048 | M37b | 15 |  | 0 | true | false |
| age_sleeparm | 24 | 128 | E110b | 0 | 288 | 160 | false | false |
| age_sleeparm | 24 | 256 | C7a | 1 | 265 | 9 | false | true |
| age_sleeparm | 24 | 384 | C7b | 2 | 664 | 280 | false | false |
| age_sleeparm | 24 | 512 | E110a | 3 | 624 | 112 | false | true |
| age_sleeparm | 24 | 640 | E30b | 4 | 923 | 283 | false | false |
| age_sleeparm | 24 | 768 | E30b | 5 | 923 | 155 | false | false |
| age_sleeparm | 24 | 896 | C7b | 6 | 992 | 96 | false | true |
| age_sleeparm | 24 | 1024 | E30a | 7 | 1264 | 240 | false | false |
| age_sleeparm | 24 | 1152 | Gb | 8 | 1247 | 95 | false | true |
| age_sleeparm | 24 | 1280 | C1a | 9 | 1401 | 121 | false | true |
| age_sleeparm | 24 | 1408 | M37a | 10 | 1570 | 162 | false | false |
| age_sleeparm | 24 | 1536 | E110b | 11 | 1538 | 2 | false | true |
| age_sleeparm | 24 | 1664 | E110b | 12 | 1800 | 136 | false | false |
| age_sleeparm | 24 | 1792 | M7a | 13 | 1847 | 55 | false | true |
| age_sleeparm | 24 | 1920 | M7a | 14 |  | 128 | true | false |
| age_sleeparm | 24 | 2048 | C7b | 15 |  | 0 | true | false |
| age_sleeparm | 25 | 128 | Ga | 0 | 227 | 99 | false | true |
| age_sleeparm | 25 | 256 | Ga | 1 | 331 | 75 | false | true |
| age_sleeparm | 25 | 384 | C1b | 2 | 569 | 185 | false | false |
| age_sleeparm | 25 | 512 | E110a | 3 | 748 | 236 | false | false |
| age_sleeparm | 25 | 640 | E30b | 4 | 822 | 182 | false | false |
| age_sleeparm | 25 | 768 | Gb | 5 | 892 | 124 | false | true |
| age_sleeparm | 25 | 896 | C7a | 6 | 1143 | 247 | false | false |
| age_sleeparm | 25 | 1024 | M37a | 7 | 1223 | 199 | false | false |
| age_sleeparm | 25 | 1152 | M37b | 8 | 1272 | 120 | false | true |
| age_sleeparm | 25 | 1280 | C7b | 9 | 1405 | 125 | false | true |
| age_sleeparm | 25 | 1408 | E30a | 10 | 1544 | 136 | false | false |
| age_sleeparm | 25 | 1536 | M37a | 11 | 1777 | 241 | false | false |
| age_sleeparm | 25 | 1664 | Ga | 12 | 1853 | 189 | false | false |
| age_sleeparm | 25 | 1792 | M7b | 13 | 1794 | 2 | false | true |
| age_sleeparm | 25 | 1920 | E30b | 14 | 1966 | 46 | false | true |
| age_sleeparm | 25 | 2048 | M7a | 15 |  | 0 | true | false |
| age_sleeparm | 26 | 128 | Gb | 0 | 129 | 1 | false | true |
| age_sleeparm | 26 | 256 | M7a | 1 | 505 | 249 | false | false |
| age_sleeparm | 26 | 384 | C7a | 2 | 419 | 35 | false | true |
| age_sleeparm | 26 | 512 | C1a | 3 | 669 | 157 | false | false |
| age_sleeparm | 26 | 640 | C1b | 4 | 811 | 171 | false | false |
| age_sleeparm | 26 | 768 | C7a | 5 | 905 | 137 | false | false |
| age_sleeparm | 26 | 896 | M7a | 6 | 1039 | 143 | false | false |
| age_sleeparm | 26 | 1024 | Gb | 7 | 1067 | 43 | false | true |
| age_sleeparm | 26 | 1152 | E110a | 8 | 1332 | 180 | false | false |
| age_sleeparm | 26 | 1280 | M37b | 9 | 1472 | 192 | false | false |
| age_sleeparm | 26 | 1408 | C7a | 10 | 1457 | 49 | false | true |
| age_sleeparm | 26 | 1536 | M37b | 11 | 1720 | 184 | false | false |
| age_sleeparm | 26 | 1664 | Ga | 12 | 1686 | 22 | false | true |
| age_sleeparm | 26 | 1792 | Gb | 13 | 1993 | 201 | false | false |
| age_sleeparm | 26 | 1920 | C7a | 14 | 1957 | 37 | false | true |
| age_sleeparm | 26 | 2048 | E110b | 15 |  | 0 | true | false |
| age_sleeparm | 27 | 128 | M7b | 0 | 310 | 182 | false | false |
| age_sleeparm | 27 | 256 | C7a | 1 | 338 | 82 | false | true |
| age_sleeparm | 27 | 384 | Gb | 2 | 561 | 177 | false | false |
| age_sleeparm | 27 | 512 | E30a | 3 | 654 | 142 | false | false |
| age_sleeparm | 27 | 640 | C7a | 4 | 642 | 2 | false | true |
| age_sleeparm | 27 | 768 | C7a | 5 | 950 | 182 | false | false |
| age_sleeparm | 27 | 896 | C7b | 6 | 1136 | 240 | false | false |
| age_sleeparm | 27 | 1024 | M7b | 7 | 1178 | 154 | false | false |
| age_sleeparm | 27 | 1152 | C7a | 8 | 1192 | 40 | false | true |
| age_sleeparm | 27 | 1280 | M37a | 9 | 1409 | 129 | false | false |
| age_sleeparm | 27 | 1408 | E30b | 10 | 1494 | 86 | false | true |
| age_sleeparm | 27 | 1536 | C7a | 11 | 1726 | 190 | false | false |
| age_sleeparm | 27 | 1664 | E110b | 12 | 1895 | 231 | false | false |
| age_sleeparm | 27 | 1792 | Ga | 13 | 1972 | 180 | false | false |
| age_sleeparm | 27 | 1920 | C1a | 14 |  | 128 | true | false |
| age_sleeparm | 27 | 2048 | M37a | 15 |  | 0 | true | false |
| age_sleeparm | 28 | 128 | E110b | 0 | 410 | 282 | false | false |
| age_sleeparm | 28 | 256 | M37b | 1 | 264 | 8 | false | true |
| age_sleeparm | 28 | 384 | Gb | 2 | 389 | 5 | false | true |
| age_sleeparm | 28 | 512 | E110b | 3 | 692 | 180 | false | false |
| age_sleeparm | 28 | 640 | E30a | 4 | 765 | 125 | false | true |
| age_sleeparm | 28 | 768 | M37b | 5 | 900 | 132 | false | false |
| age_sleeparm | 28 | 896 | C1b | 6 | 1131 | 235 | false | false |
| age_sleeparm | 28 | 1024 | C1b | 7 | 1131 | 107 | false | true |
| age_sleeparm | 28 | 1152 | E110a | 8 | 1246 | 94 | false | true |
| age_sleeparm | 28 | 1280 | E110b | 9 | 1304 | 24 | false | true |
| age_sleeparm | 28 | 1408 | Ga | 10 | 1586 | 178 | false | false |
| age_sleeparm | 28 | 1536 | Ga | 11 | 1586 | 50 | false | true |
| age_sleeparm | 28 | 1664 | M37a | 12 | 1797 | 133 | false | false |
| age_sleeparm | 28 | 1792 | E30a | 13 | 2023 | 231 | false | false |
| age_sleeparm | 28 | 1920 | C7b | 14 | 1970 | 50 | false | true |
| age_sleeparm | 28 | 2048 | C7b | 15 |  | 0 | true | false |
| age_sleeparm | 29 | 128 | Ga | 0 | 129 | 1 | false | true |
| age_sleeparm | 29 | 256 | M37a | 1 | 498 | 242 | false | false |
| age_sleeparm | 29 | 384 | C1a | 2 | 593 | 209 | false | false |
| age_sleeparm | 29 | 512 | C1b | 3 | 634 | 122 | false | true |
| age_sleeparm | 29 | 640 | C7a | 4 | 866 | 226 | false | false |
| age_sleeparm | 29 | 768 | C1b | 5 | 894 | 126 | false | true |
| age_sleeparm | 29 | 896 | Ga | 6 | 1081 | 185 | false | false |
| age_sleeparm | 29 | 1024 | E30a | 7 | 1210 | 186 | false | false |
| age_sleeparm | 29 | 1152 | E30a | 8 | 1210 | 58 | false | true |
| age_sleeparm | 29 | 1280 | C7a | 9 | 1402 | 122 | false | true |
| age_sleeparm | 29 | 1408 | C7b | 10 | 1478 | 70 | false | true |
| age_sleeparm | 29 | 1536 | E30b | 11 | 1576 | 40 | false | true |
| age_sleeparm | 29 | 1664 | C7b | 12 | 1750 | 86 | false | true |
| age_sleeparm | 29 | 1792 | E30b | 13 | 1836 | 44 | false | true |
| age_sleeparm | 29 | 1920 | C7a | 14 | 1921 | 1 | false | true |
| age_sleeparm | 29 | 2048 | M37b | 15 |  | 0 | true | false |
| age_sleeparm | 30 | 128 | E30a | 0 | 145 | 17 | false | true |
| age_sleeparm | 30 | 256 | E30a | 1 | 463 | 207 | false | false |
| age_sleeparm | 30 | 384 | E30a | 2 | 463 | 79 | false | true |
| age_sleeparm | 30 | 512 | E110b | 3 | 537 | 25 | false | true |
| age_sleeparm | 30 | 640 | E30b | 4 | 699 | 59 | false | true |
| age_sleeparm | 30 | 768 | C1b | 5 | 1025 | 257 | false | false |
| age_sleeparm | 30 | 896 | C7a | 6 | 974 | 78 | false | true |
| age_sleeparm | 30 | 1024 | E30a | 7 | 1117 | 93 | false | true |
| age_sleeparm | 30 | 1152 | M7a | 8 | 1196 | 44 | false | true |
| age_sleeparm | 30 | 1280 | M37a | 9 | 1382 | 102 | false | true |
| age_sleeparm | 30 | 1408 | E30b | 10 | 1515 | 107 | false | true |
| age_sleeparm | 30 | 1536 | E110b | 11 | 1749 | 213 | false | false |
| age_sleeparm | 30 | 1664 | E30a | 12 | 1879 | 215 | false | false |
| age_sleeparm | 30 | 1792 | E110b | 13 | 1999 | 207 | false | false |
| age_sleeparm | 30 | 1920 | C1a | 14 |  | 128 | true | false |
| age_sleeparm | 30 | 2048 | C1b | 15 |  | 0 | true | false |
| age_sleeparm | 31 | 128 | E110a | 0 | 129 | 1 | false | true |
| age_sleeparm | 31 | 256 | C7b | 1 | 340 | 84 | false | true |
| age_sleeparm | 31 | 384 | M7b | 2 | 464 | 80 | false | true |
| age_sleeparm | 31 | 512 | E30a | 3 | 517 | 5 | false | true |
| age_sleeparm | 31 | 640 | C7b | 4 | 864 | 224 | false | false |
| age_sleeparm | 31 | 768 | E30b | 5 | 1041 | 273 | false | false |
| age_sleeparm | 31 | 896 | M37b | 6 | 943 | 47 | false | true |
| age_sleeparm | 31 | 1024 | C1a | 7 | 1093 | 69 | false | true |
| age_sleeparm | 31 | 1152 | Gb | 8 | 1342 | 190 | false | false |
| age_sleeparm | 31 | 1280 | E30a | 9 | 1299 | 19 | false | true |
| age_sleeparm | 31 | 1408 | Ga | 10 | 1624 | 216 | false | false |
| age_sleeparm | 31 | 1536 | M7b | 11 | 1712 | 176 | false | false |
| age_sleeparm | 31 | 1664 | E110a | 12 | 1703 | 39 | false | true |
| age_sleeparm | 31 | 1792 | C7a | 13 | 1823 | 31 | false | true |
| age_sleeparm | 31 | 1920 | M7a | 14 | 1966 | 46 | false | true |
| age_sleeparm | 31 | 2048 | M7a | 15 |  | 0 | true | false |
| age_sleeparm | 32 | 128 | E110a | 0 | 173 | 45 | false | true |
| age_sleeparm | 32 | 256 | M7b | 1 | 414 | 158 | false | false |
| age_sleeparm | 32 | 384 | M7b | 2 | 414 | 30 | false | true |
| age_sleeparm | 32 | 512 | Ga | 3 | 668 | 156 | false | false |
| age_sleeparm | 32 | 640 | M37b | 4 | 643 | 3 | false | true |
| age_sleeparm | 32 | 768 | C1a | 5 | 795 | 27 | false | true |
| age_sleeparm | 32 | 896 | C7b | 6 | 934 | 38 | false | true |
| age_sleeparm | 32 | 1024 | E30a | 7 | 1096 | 72 | false | true |
| age_sleeparm | 32 | 1152 | E30a | 8 | 1400 | 248 | false | false |
| age_sleeparm | 32 | 1280 | E30a | 9 | 1400 | 120 | false | true |
| age_sleeparm | 32 | 1408 | C7b | 10 | 1456 | 48 | false | true |
| age_sleeparm | 32 | 1536 | E30a | 11 | 1672 | 136 | false | false |
| age_sleeparm | 32 | 1664 | C7b | 12 | 1744 | 80 | false | true |
| age_sleeparm | 32 | 1792 | M7a | 13 | 1859 | 67 | false | true |
| age_sleeparm | 32 | 1920 | C7b | 14 | 2008 | 88 | false | true |
| age_sleeparm | 32 | 2048 | M37b | 15 |  | 0 | true | false |
| age_sleeparm | 33 | 128 | E30b | 0 | 350 | 222 | false | false |
| age_sleeparm | 33 | 256 | M37a | 1 | 280 | 24 | false | true |
| age_sleeparm | 33 | 384 | Gb | 2 | 486 | 102 | false | true |
| age_sleeparm | 33 | 512 | E110b | 3 | 530 | 18 | false | true |
| age_sleeparm | 33 | 640 | M7a | 4 | 763 | 123 | false | true |
| age_sleeparm | 33 | 768 | E30a | 5 | 1077 | 309 | false | false |
| age_sleeparm | 33 | 896 | C7b | 6 | 918 | 22 | false | true |
| age_sleeparm | 33 | 1024 | C1a | 7 | 1105 | 81 | false | true |
| age_sleeparm | 33 | 1152 | Ga | 8 | 1285 | 133 | false | false |
| age_sleeparm | 33 | 1280 | Ga | 9 | 1285 | 5 | false | true |
| age_sleeparm | 33 | 1408 | C1a | 10 | 1631 | 223 | false | false |
| age_sleeparm | 33 | 1536 | Ga | 11 | 1541 | 5 | false | true |
| age_sleeparm | 33 | 1664 | M7b | 12 | 1679 | 15 | false | true |
| age_sleeparm | 33 | 1792 | C7b | 13 |  | 256 | true | false |
| age_sleeparm | 33 | 1920 | M37a | 14 | 1946 | 26 | false | true |
| age_sleeparm | 33 | 2048 | Gb | 15 |  | 0 | true | false |
| age_sleeparm | 34 | 128 | M37b | 0 | 262 | 134 | false | false |
| age_sleeparm | 34 | 256 | C1a | 1 | 303 | 47 | false | true |
| age_sleeparm | 34 | 384 | Ga | 2 | 465 | 81 | false | true |
| age_sleeparm | 34 | 512 | M37a | 3 | 513 | 1 | false | true |
| age_sleeparm | 34 | 640 | C1a | 4 | 861 | 221 | false | false |
| age_sleeparm | 34 | 768 | E110a | 5 | 873 | 105 | false | true |
| age_sleeparm | 34 | 896 | C7a | 6 | 1081 | 185 | false | false |
| age_sleeparm | 34 | 1024 | C1a | 7 | 1149 | 125 | false | true |
| age_sleeparm | 34 | 1152 | Gb | 8 | 1255 | 103 | false | true |
| age_sleeparm | 34 | 1280 | C7b | 9 | 1413 | 133 | false | false |
| age_sleeparm | 34 | 1408 | C7b | 10 | 1413 | 5 | false | true |
| age_sleeparm | 34 | 1536 | E30a | 11 | 1607 | 71 | false | true |
| age_sleeparm | 34 | 1664 | E30b | 12 | 1882 | 218 | false | false |
| age_sleeparm | 34 | 1792 | Ga | 13 | 2039 | 247 | false | false |
| age_sleeparm | 34 | 1920 | E30b | 14 |  | 128 | true | false |
| age_sleeparm | 34 | 2048 | M7b | 15 |  | 0 | true | false |
| age_sleeparm | 35 | 128 | E30a | 0 | 400 | 272 | false | false |
| age_sleeparm | 35 | 256 | C7a | 1 | 502 | 246 | false | false |
| age_sleeparm | 35 | 384 | M7a | 2 | 529 | 145 | false | false |
| age_sleeparm | 35 | 512 | C7b | 3 | 562 | 50 | false | true |
| age_sleeparm | 35 | 640 | C1b | 4 | 863 | 223 | false | false |
| age_sleeparm | 35 | 768 | E110b | 5 | 1002 | 234 | false | false |
| age_sleeparm | 35 | 896 | M37b | 6 | 950 | 54 | false | true |
| age_sleeparm | 35 | 1024 | M7a | 7 | 1233 | 209 | false | false |
| age_sleeparm | 35 | 1152 | C7a | 8 | 1294 | 142 | false | false |
| age_sleeparm | 35 | 1280 | C1b | 9 | 1403 | 123 | false | true |
| age_sleeparm | 35 | 1408 | M7b | 10 | 1529 | 121 | false | true |
| age_sleeparm | 35 | 1536 | Gb | 11 | 1651 | 115 | false | true |
| age_sleeparm | 35 | 1664 | E30a | 12 | 1778 | 114 | false | true |
| age_sleeparm | 35 | 1792 | C1b | 13 | 1993 | 201 | false | false |
| age_sleeparm | 35 | 1920 | E30b | 14 |  | 128 | true | false |
| age_sleeparm | 35 | 2048 | M37b | 15 |  | 0 | true | false |
| age_fixed_matched | 24 | 128 | E110b | 0 | 306 | 178 | false | false |
| age_fixed_matched | 24 | 256 | C7a | 1 | 267 | 11 | false | true |
| age_fixed_matched | 24 | 384 | C7b | 2 | 674 | 290 | false | false |
| age_fixed_matched | 24 | 512 | E110a | 3 | 650 | 138 | false | false |
| age_fixed_matched | 24 | 640 | E30b | 4 | 641 | 1 | false | true |
| age_fixed_matched | 24 | 768 | E30b | 5 | 881 | 113 | false | true |
| age_fixed_matched | 24 | 896 | C7b | 6 | 954 | 58 | false | true |
| age_fixed_matched | 24 | 1024 | E30a | 7 | 1222 | 198 | false | false |
| age_fixed_matched | 24 | 1152 | Gb | 8 | 1187 | 35 | false | true |
| age_fixed_matched | 24 | 1280 | C1a | 9 | 1565 | 285 | false | false |
| age_fixed_matched | 24 | 1408 | M37a | 10 | 1588 | 180 | false | false |
| age_fixed_matched | 24 | 1536 | E110b | 11 | 1598 | 62 | false | true |
| age_fixed_matched | 24 | 1664 | E110b | 12 | 1836 | 172 | false | false |
| age_fixed_matched | 24 | 1792 | M7a | 13 | 1797 | 5 | false | true |
| age_fixed_matched | 24 | 1920 | M7a | 14 | 2011 | 91 | false | true |
| age_fixed_matched | 24 | 2048 | C7b | 15 |  | 0 | true | false |
| age_fixed_matched | 25 | 128 | Ga | 0 | 235 | 107 | false | true |
| age_fixed_matched | 25 | 256 | Ga | 1 | 397 | 141 | false | false |
| age_fixed_matched | 25 | 384 | C1b | 2 | 567 | 183 | false | false |
| age_fixed_matched | 25 | 512 | E110a | 3 | 826 | 314 | false | false |
| age_fixed_matched | 25 | 640 | E30b | 4 | 756 | 116 | false | true |
| age_fixed_matched | 25 | 768 | Gb | 5 | 778 | 10 | false | true |
| age_fixed_matched | 25 | 896 | C7a | 6 | 1171 | 275 | false | false |
| age_fixed_matched | 25 | 1024 | M37a | 7 | 1209 | 185 | false | false |
| age_fixed_matched | 25 | 1152 | M37b | 8 | 1218 | 66 | false | true |
| age_fixed_matched | 25 | 1280 | C7b | 9 | 1401 | 121 | false | true |
| age_fixed_matched | 25 | 1408 | E30a | 10 | 1602 | 194 | false | false |
| age_fixed_matched | 25 | 1536 | M37a | 11 | 1823 | 287 | false | false |
| age_fixed_matched | 25 | 1664 | Ga | 12 | 1775 | 111 | false | true |
| age_fixed_matched | 25 | 1792 | M7b | 13 | 1798 | 6 | false | true |
| age_fixed_matched | 25 | 1920 | E30b | 14 |  | 128 | true | false |
| age_fixed_matched | 25 | 2048 | M7a | 15 |  | 0 | true | false |
| age_fixed_matched | 26 | 128 | Gb | 0 | 135 | 7 | false | true |
| age_fixed_matched | 26 | 256 | M7a | 1 | 277 | 21 | false | true |
| age_fixed_matched | 26 | 384 | C7a | 2 | 535 | 151 | false | false |
| age_fixed_matched | 26 | 512 | C1a | 3 | 785 | 273 | false | false |
| age_fixed_matched | 26 | 640 | C1b | 4 | 825 | 185 | false | false |
| age_fixed_matched | 26 | 768 | C7a | 5 | 1013 | 245 | false | false |
| age_fixed_matched | 26 | 896 | M7a | 6 | 947 | 51 | false | true |
| age_fixed_matched | 26 | 1024 | Gb | 7 | 1079 | 55 | false | true |
| age_fixed_matched | 26 | 1152 | E110a | 8 | 1348 | 196 | false | false |
| age_fixed_matched | 26 | 1280 | M37b | 9 | 1398 | 118 | false | true |
| age_fixed_matched | 26 | 1408 | C7a | 10 | 1579 | 171 | false | false |
| age_fixed_matched | 26 | 1536 | M37b | 11 | 1638 | 102 | false | true |
| age_fixed_matched | 26 | 1664 | Ga | 12 | 1698 | 34 | false | true |
| age_fixed_matched | 26 | 1792 | Gb | 13 | 2003 | 211 | false | false |
| age_fixed_matched | 26 | 1920 | C7a | 14 |  | 128 | true | false |
| age_fixed_matched | 26 | 2048 | E110b | 15 |  | 0 | true | false |
| age_fixed_matched | 27 | 128 | M7b | 0 | 356 | 228 | false | false |
| age_fixed_matched | 27 | 256 | C7a | 1 | 334 | 78 | false | true |
| age_fixed_matched | 27 | 384 | Gb | 2 | 401 | 17 | false | true |
| age_fixed_matched | 27 | 512 | E30a | 3 | 738 | 226 | false | false |
| age_fixed_matched | 27 | 640 | C7a | 4 | 641 | 1 | false | true |
| age_fixed_matched | 27 | 768 | C7a | 5 | 926 | 158 | false | false |
| age_fixed_matched | 27 | 896 | C7b | 6 | 998 | 102 | false | true |
| age_fixed_matched | 27 | 1024 | M7b | 7 | 1190 | 166 | false | false |
| age_fixed_matched | 27 | 1152 | C7a | 8 | 1212 | 60 | false | true |
| age_fixed_matched | 27 | 1280 | M37a | 9 | 1449 | 169 | false | false |
| age_fixed_matched | 27 | 1408 | E30b | 10 | 1630 | 222 | false | false |
| age_fixed_matched | 27 | 1536 | C7a | 11 | 1826 | 290 | false | false |
| age_fixed_matched | 27 | 1664 | E110b | 12 | 1851 | 187 | false | false |
| age_fixed_matched | 27 | 1792 | Ga | 13 | 1948 | 156 | false | false |
| age_fixed_matched | 27 | 1920 | C1a | 14 | 2016 | 96 | false | true |
| age_fixed_matched | 27 | 2048 | M37a | 15 |  | 0 | true | false |
| age_fixed_matched | 28 | 128 | E110b | 0 | 408 | 280 | false | false |
| age_fixed_matched | 28 | 256 | M37b | 1 | 257 | 1 | false | true |
| age_fixed_matched | 28 | 384 | Gb | 2 | 385 | 1 | false | true |
| age_fixed_matched | 28 | 512 | E110b | 3 | 712 | 200 | false | false |
| age_fixed_matched | 28 | 640 | E30a | 4 | 743 | 103 | false | true |
| age_fixed_matched | 28 | 768 | M37b | 5 | 908 | 140 | false | false |
| age_fixed_matched | 28 | 896 | C1b | 6 | 921 | 25 | false | true |
| age_fixed_matched | 28 | 1024 | C1b | 7 | 1263 | 239 | false | false |
| age_fixed_matched | 28 | 1152 | E110a | 8 | 1432 | 280 | false | false |
| age_fixed_matched | 28 | 1280 | E110b | 9 | 1290 | 10 | false | true |
| age_fixed_matched | 28 | 1408 | Ga | 10 | 1538 | 130 | false | false |
| age_fixed_matched | 28 | 1536 | Ga | 11 | 1538 | 2 | false | true |
| age_fixed_matched | 28 | 1664 | M37a | 12 | 1889 | 225 | false | false |
| age_fixed_matched | 28 | 1792 | E30a | 13 | 2011 | 219 | false | false |
| age_fixed_matched | 28 | 1920 | C7b | 14 | 2022 | 102 | false | true |
| age_fixed_matched | 28 | 2048 | C7b | 15 |  | 0 | true | false |
| age_fixed_matched | 29 | 128 | Ga | 0 | 135 | 7 | false | true |
| age_fixed_matched | 29 | 256 | M37a | 1 | 262 | 6 | false | true |
| age_fixed_matched | 29 | 384 | C1a | 2 | 677 | 293 | false | false |
| age_fixed_matched | 29 | 512 | C1b | 3 | 744 | 232 | false | false |
| age_fixed_matched | 29 | 640 | C7a | 4 | 864 | 224 | false | false |
| age_fixed_matched | 29 | 768 | C1b | 5 | 1048 | 280 | false | false |
| age_fixed_matched | 29 | 896 | Ga | 6 | 945 | 49 | false | true |
| age_fixed_matched | 29 | 1024 | E30a | 7 | 1162 | 138 | false | false |
| age_fixed_matched | 29 | 1152 | E30a | 8 | 1162 | 10 | false | true |
| age_fixed_matched | 29 | 1280 | C7a | 9 | 1388 | 108 | false | true |
| age_fixed_matched | 29 | 1408 | C7b | 10 | 1410 | 2 | false | true |
| age_fixed_matched | 29 | 1536 | E30b | 11 | 1586 | 50 | false | true |
| age_fixed_matched | 29 | 1664 | C7b | 12 | 1684 | 20 | false | true |
| age_fixed_matched | 29 | 1792 | E30b | 13 | 1856 | 64 | false | true |
| age_fixed_matched | 29 | 1920 | C7a | 14 |  | 128 | true | false |
| age_fixed_matched | 29 | 2048 | M37b | 15 |  | 0 | true | false |
| age_fixed_matched | 30 | 128 | E30a | 0 | 153 | 25 | false | true |
| age_fixed_matched | 30 | 256 | E30a | 1 | 441 | 185 | false | false |
| age_fixed_matched | 30 | 384 | E30a | 2 | 441 | 57 | false | true |
| age_fixed_matched | 30 | 512 | E110b | 3 | 611 | 99 | false | true |
| age_fixed_matched | 30 | 640 | E30b | 4 | 821 | 181 | false | false |
| age_fixed_matched | 30 | 768 | C1b | 5 | 987 | 219 | false | false |
| age_fixed_matched | 30 | 896 | C7a | 6 | 958 | 62 | false | true |
| age_fixed_matched | 30 | 1024 | E30a | 7 | 1105 | 81 | false | true |
| age_fixed_matched | 30 | 1152 | M7a | 8 | 1244 | 92 | false | true |
| age_fixed_matched | 30 | 1280 | M37a | 9 | 1281 | 1 | false | true |
| age_fixed_matched | 30 | 1408 | E30b | 10 | 1539 | 131 | false | false |
| age_fixed_matched | 30 | 1536 | E110b | 11 | 1599 | 63 | false | true |
| age_fixed_matched | 30 | 1664 | E30a | 12 | 1719 | 55 | false | true |
| age_fixed_matched | 30 | 1792 | E110b | 13 | 1913 | 121 | false | true |
| age_fixed_matched | 30 | 1920 | C1a | 14 |  | 128 | true | false |
| age_fixed_matched | 30 | 2048 | C1b | 15 |  | 0 | true | false |
| age_fixed_matched | 31 | 128 | E110a | 0 | 135 | 7 | false | true |
| age_fixed_matched | 31 | 256 | C7b | 1 | 356 | 100 | false | true |
| age_fixed_matched | 31 | 384 | M7b | 2 | 494 | 110 | false | true |
| age_fixed_matched | 31 | 512 | E30a | 3 | 583 | 71 | false | true |
| age_fixed_matched | 31 | 640 | C7b | 4 | 926 | 286 | false | false |
| age_fixed_matched | 31 | 768 | E30b | 5 | 973 | 205 | false | false |
| age_fixed_matched | 31 | 896 | M37b | 6 | 995 | 99 | false | true |
| age_fixed_matched | 31 | 1024 | C1a | 7 | 1197 | 173 | false | false |
| age_fixed_matched | 31 | 1152 | Gb | 8 | 1164 | 12 | false | true |
| age_fixed_matched | 31 | 1280 | E30a | 9 | 1327 | 47 | false | true |
| age_fixed_matched | 31 | 1408 | Ga | 10 | 1452 | 44 | false | true |
| age_fixed_matched | 31 | 1536 | M7b | 11 | 1656 | 120 | false | true |
| age_fixed_matched | 31 | 1664 | E110a | 12 | 1867 | 203 | false | false |
| age_fixed_matched | 31 | 1792 | C7a | 13 | 1801 | 9 | false | true |
| age_fixed_matched | 31 | 1920 | M7a | 14 | 1924 | 4 | false | true |
| age_fixed_matched | 31 | 2048 | M7a | 15 |  | 0 | true | false |
| age_fixed_matched | 32 | 128 | E110a | 0 | 173 | 45 | false | true |
| age_fixed_matched | 32 | 256 | M7b | 1 | 366 | 110 | false | true |
| age_fixed_matched | 32 | 384 | M7b | 2 | 662 | 278 | false | false |
| age_fixed_matched | 32 | 512 | Ga | 3 | 570 | 58 | false | true |
| age_fixed_matched | 32 | 640 | M37b | 4 | 677 | 37 | false | true |
| age_fixed_matched | 32 | 768 | C1a | 5 | 965 | 197 | false | false |
| age_fixed_matched | 32 | 896 | C7b | 6 | 1000 | 104 | false | true |
| age_fixed_matched | 32 | 1024 | E30a | 7 | 1288 | 264 | false | false |
| age_fixed_matched | 32 | 1152 | E30a | 8 | 1288 | 136 | false | false |
| age_fixed_matched | 32 | 1280 | E30a | 9 | 1288 | 8 | false | true |
| age_fixed_matched | 32 | 1408 | C7b | 10 | 1484 | 76 | false | true |
| age_fixed_matched | 32 | 1536 | E30a | 11 | 1576 | 40 | false | true |
| age_fixed_matched | 32 | 1664 | C7b | 12 | 1758 | 94 | false | true |
| age_fixed_matched | 32 | 1792 | M7a | 13 | 1823 | 31 | false | true |
| age_fixed_matched | 32 | 1920 | C7b | 14 | 2010 | 90 | false | true |
| age_fixed_matched | 32 | 2048 | M37b | 15 |  | 0 | true | false |
| age_fixed_matched | 33 | 128 | E30b | 0 | 328 | 200 | false | false |
| age_fixed_matched | 33 | 256 | M37a | 1 | 278 | 22 | false | true |
| age_fixed_matched | 33 | 384 | Gb | 2 | 560 | 176 | false | false |
| age_fixed_matched | 33 | 512 | E110b | 3 | 546 | 34 | false | true |
| age_fixed_matched | 33 | 640 | M7a | 4 | 675 | 35 | false | true |
| age_fixed_matched | 33 | 768 | E30a | 5 | 769 | 1 | false | true |
| age_fixed_matched | 33 | 896 | C7b | 6 | 1160 | 264 | false | false |
| age_fixed_matched | 33 | 1024 | C1a | 7 | 1283 | 259 | false | false |
| age_fixed_matched | 33 | 1152 | Ga | 8 | 1327 | 175 | false | false |
| age_fixed_matched | 33 | 1280 | Ga | 9 | 1327 | 47 | false | true |
| age_fixed_matched | 33 | 1408 | C1a | 10 | 1527 | 119 | false | true |
| age_fixed_matched | 33 | 1536 | Ga | 11 | 1571 | 35 | false | true |
| age_fixed_matched | 33 | 1664 | M7b | 12 | 1867 | 203 | false | false |
| age_fixed_matched | 33 | 1792 | C7b | 13 | 1968 | 176 | false | false |
| age_fixed_matched | 33 | 1920 | M37a | 14 | 1922 | 2 | false | true |
| age_fixed_matched | 33 | 2048 | Gb | 15 |  | 0 | true | false |
| age_fixed_matched | 34 | 128 | M37b | 0 | 326 | 198 | false | false |
| age_fixed_matched | 34 | 256 | C1a | 1 | 301 | 45 | false | true |
| age_fixed_matched | 34 | 384 | Ga | 2 | 391 | 7 | false | true |
| age_fixed_matched | 34 | 512 | M37a | 3 | 522 | 10 | false | true |
| age_fixed_matched | 34 | 640 | C1a | 4 | 895 | 255 | false | false |
| age_fixed_matched | 34 | 768 | E110a | 5 | 871 | 103 | false | true |
| age_fixed_matched | 34 | 896 | C7a | 6 | 947 | 51 | false | true |
| age_fixed_matched | 34 | 1024 | C1a | 7 | 1169 | 145 | false | false |
| age_fixed_matched | 34 | 1152 | Gb | 8 | 1153 | 1 | false | true |
| age_fixed_matched | 34 | 1280 | C7b | 9 | 1515 | 235 | false | false |
| age_fixed_matched | 34 | 1408 | C7b | 10 | 1515 | 107 | false | true |
| age_fixed_matched | 34 | 1536 | E30a | 11 | 1543 | 7 | false | true |
| age_fixed_matched | 34 | 1664 | E30b | 12 | 1686 | 22 | false | true |
| age_fixed_matched | 34 | 1792 | Ga | 13 | 1987 | 195 | false | false |
| age_fixed_matched | 34 | 1920 | E30b | 14 | 1948 | 28 | false | true |
| age_fixed_matched | 34 | 2048 | M7b | 15 |  | 0 | true | false |
| age_fixed_matched | 35 | 128 | E30a | 0 | 440 | 312 | false | false |
| age_fixed_matched | 35 | 256 | C7a | 1 | 404 | 148 | false | false |
| age_fixed_matched | 35 | 384 | M7a | 2 | 561 | 177 | false | false |
| age_fixed_matched | 35 | 512 | C7b | 3 | 594 | 82 | false | true |
| age_fixed_matched | 35 | 640 | C1b | 4 | 877 | 237 | false | false |
| age_fixed_matched | 35 | 768 | E110b | 5 | 836 | 68 | false | true |
| age_fixed_matched | 35 | 896 | M37b | 6 | 1070 | 174 | false | false |
| age_fixed_matched | 35 | 1024 | M7a | 7 | 1251 | 227 | false | false |
| age_fixed_matched | 35 | 1152 | C7a | 8 | 1308 | 156 | false | false |
| age_fixed_matched | 35 | 1280 | C1b | 9 | 1379 | 99 | false | true |
| age_fixed_matched | 35 | 1408 | M7b | 10 | 1665 | 257 | false | false |
| age_fixed_matched | 35 | 1536 | Gb | 11 | 1567 | 31 | false | true |
| age_fixed_matched | 35 | 1664 | E30a | 12 | 1718 | 54 | false | true |
| age_fixed_matched | 35 | 1792 | C1b | 13 | 1947 | 155 | false | false |
| age_fixed_matched | 35 | 1920 | E30b | 14 |  | 128 | true | false |
| age_fixed_matched | 35 | 2048 | M37b | 15 |  | 0 | true | false |

## Decision shares

Mean over seeds. A stay counts for the room. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.

| controller | Ga | C1a | C7a | E30a | E110a | M7a | M37a | Gb | C1b | C7b | E30b | E110b | M7b | M37b | N1 | N2 | N3 | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| age_nosleep | 0.0434 | 0.0306 | 0.0329 | 0.0339 | 0.0329 | 0.0403 | 0.0413 | 0.0495 | 0.0324 | 0.0335 | 0.0325 | 0.0322 | 0.0390 | 0.0420 | 0.0350 | 0.0367 | 0.0352 | 0.0284 | 0.3483 | 0.0000 |
| age_fixed20 | 0.0365 | 0.0257 | 0.0274 | 0.0273 | 0.0276 | 0.0277 | 0.0303 | 0.0413 | 0.0269 | 0.0267 | 0.0277 | 0.0254 | 0.0269 | 0.0319 | 0.0306 | 0.0309 | 0.0316 | 0.0230 | 0.2754 | 0.1992 |
| age_sleeparm | 0.0378 | 0.0319 | 0.0335 | 0.0316 | 0.0332 | 0.0339 | 0.0364 | 0.0439 | 0.0335 | 0.0329 | 0.0318 | 0.0322 | 0.0343 | 0.0371 | 0.0326 | 0.0345 | 0.0348 | 0.0293 | 0.3260 | 0.0588 |
| age_fixed_matched | 0.0436 | 0.0296 | 0.0316 | 0.0321 | 0.0320 | 0.0345 | 0.0350 | 0.0492 | 0.0309 | 0.0316 | 0.0321 | 0.0304 | 0.0340 | 0.0374 | 0.0335 | 0.0348 | 0.0345 | 0.0273 | 0.3269 | 0.0589 |

| controller | seed | Ga | C1a | C7a | E30a | E110a | M7a | M37a | Gb | C1b | C7b | E30b | E110b | M7b | M37b | N1 | N2 | N3 | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| age_nosleep | 24 | 0.0312 | 0.0273 | 0.0352 | 0.0312 | 0.0312 | 0.0508 | 0.0430 | 0.0586 | 0.0273 | 0.0312 | 0.0303 | 0.0312 | 0.0391 | 0.0547 | 0.0312 | 0.0430 | 0.0391 | 0.0234 | 0.3408 | 0.0000 |
| age_nosleep | 25 | 0.0391 | 0.0312 | 0.0352 | 0.0430 | 0.0273 | 0.0430 | 0.0352 | 0.0430 | 0.0352 | 0.0312 | 0.0430 | 0.0312 | 0.0312 | 0.0352 | 0.0352 | 0.0391 | 0.0352 | 0.0312 | 0.3555 | 0.0000 |
| age_nosleep | 26 | 0.0430 | 0.0273 | 0.0352 | 0.0312 | 0.0312 | 0.0352 | 0.0273 | 0.0391 | 0.0352 | 0.0430 | 0.0391 | 0.0352 | 0.0508 | 0.0430 | 0.0312 | 0.0391 | 0.0391 | 0.0273 | 0.3477 | 0.0000 |
| age_nosleep | 27 | 0.0366 | 0.0312 | 0.0312 | 0.0312 | 0.0352 | 0.0547 | 0.0469 | 0.0508 | 0.0273 | 0.0391 | 0.0273 | 0.0430 | 0.0312 | 0.0430 | 0.0352 | 0.0352 | 0.0312 | 0.0273 | 0.3423 | 0.0000 |
| age_nosleep | 28 | 0.0430 | 0.0273 | 0.0312 | 0.0430 | 0.0273 | 0.0469 | 0.0391 | 0.0586 | 0.0332 | 0.0391 | 0.0273 | 0.0273 | 0.0430 | 0.0430 | 0.0430 | 0.0391 | 0.0273 | 0.0234 | 0.3379 | 0.0000 |
| age_nosleep | 29 | 0.0352 | 0.0312 | 0.0312 | 0.0312 | 0.0391 | 0.0430 | 0.0430 | 0.0586 | 0.0312 | 0.0273 | 0.0273 | 0.0273 | 0.0391 | 0.0352 | 0.0391 | 0.0352 | 0.0391 | 0.0288 | 0.3579 | 0.0000 |
| age_nosleep | 30 | 0.0547 | 0.0273 | 0.0352 | 0.0273 | 0.0234 | 0.0273 | 0.0625 | 0.0391 | 0.0312 | 0.0312 | 0.0273 | 0.0352 | 0.0273 | 0.0586 | 0.0352 | 0.0312 | 0.0352 | 0.0273 | 0.3633 | 0.0000 |
| age_nosleep | 31 | 0.0508 | 0.0352 | 0.0312 | 0.0352 | 0.0352 | 0.0430 | 0.0391 | 0.0430 | 0.0352 | 0.0312 | 0.0352 | 0.0312 | 0.0308 | 0.0391 | 0.0352 | 0.0469 | 0.0352 | 0.0312 | 0.3364 | 0.0000 |
| age_nosleep | 32 | 0.0430 | 0.0352 | 0.0352 | 0.0352 | 0.0391 | 0.0352 | 0.0352 | 0.0586 | 0.0278 | 0.0352 | 0.0312 | 0.0312 | 0.0312 | 0.0391 | 0.0430 | 0.0312 | 0.0352 | 0.0273 | 0.3511 | 0.0000 |
| age_nosleep | 33 | 0.0391 | 0.0352 | 0.0312 | 0.0312 | 0.0312 | 0.0391 | 0.0391 | 0.0469 | 0.0352 | 0.0312 | 0.0352 | 0.0352 | 0.0547 | 0.0352 | 0.0288 | 0.0352 | 0.0312 | 0.0352 | 0.3501 | 0.0000 |
| age_nosleep | 34 | 0.0430 | 0.0312 | 0.0352 | 0.0391 | 0.0352 | 0.0312 | 0.0352 | 0.0508 | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0586 | 0.0352 | 0.0312 | 0.0347 | 0.0352 | 0.0312 | 0.3481 | 0.0000 |
| age_nosleep | 35 | 0.0625 | 0.0273 | 0.0273 | 0.0273 | 0.0391 | 0.0342 | 0.0508 | 0.0469 | 0.0391 | 0.0312 | 0.0352 | 0.0273 | 0.0312 | 0.0430 | 0.0312 | 0.0312 | 0.0391 | 0.0273 | 0.3486 | 0.0000 |
| age_fixed20 | 24 | 0.0273 | 0.0234 | 0.0273 | 0.0273 | 0.0234 | 0.0312 | 0.0312 | 0.0469 | 0.0234 | 0.0273 | 0.0273 | 0.0273 | 0.0298 | 0.0312 | 0.0273 | 0.0391 | 0.0352 | 0.0234 | 0.2710 | 0.1992 |
| age_fixed20 | 25 | 0.0312 | 0.0234 | 0.0312 | 0.0352 | 0.0234 | 0.0312 | 0.0312 | 0.0430 | 0.0312 | 0.0234 | 0.0312 | 0.0234 | 0.0234 | 0.0308 | 0.0312 | 0.0352 | 0.0312 | 0.0273 | 0.2622 | 0.1992 |
| age_fixed20 | 26 | 0.0430 | 0.0234 | 0.0273 | 0.0234 | 0.0269 | 0.0312 | 0.0234 | 0.0352 | 0.0273 | 0.0273 | 0.0352 | 0.0234 | 0.0273 | 0.0391 | 0.0273 | 0.0312 | 0.0352 | 0.0195 | 0.2739 | 0.1992 |
| age_fixed20 | 27 | 0.0312 | 0.0273 | 0.0234 | 0.0273 | 0.0273 | 0.0273 | 0.0391 | 0.0430 | 0.0264 | 0.0312 | 0.0234 | 0.0273 | 0.0273 | 0.0312 | 0.0312 | 0.0234 | 0.0312 | 0.0234 | 0.2783 | 0.1992 |
| age_fixed20 | 28 | 0.0391 | 0.0234 | 0.0273 | 0.0312 | 0.0234 | 0.0234 | 0.0273 | 0.0469 | 0.0273 | 0.0312 | 0.0273 | 0.0234 | 0.0234 | 0.0391 | 0.0312 | 0.0312 | 0.0273 | 0.0195 | 0.2773 | 0.1992 |
| age_fixed20 | 29 | 0.0312 | 0.0234 | 0.0273 | 0.0273 | 0.0391 | 0.0273 | 0.0312 | 0.0469 | 0.0234 | 0.0234 | 0.0273 | 0.0234 | 0.0234 | 0.0234 | 0.0312 | 0.0352 | 0.0352 | 0.0234 | 0.2773 | 0.1992 |
| age_fixed20 | 30 | 0.0391 | 0.0234 | 0.0312 | 0.0234 | 0.0195 | 0.0234 | 0.0352 | 0.0312 | 0.0273 | 0.0234 | 0.0273 | 0.0312 | 0.0195 | 0.0352 | 0.0352 | 0.0273 | 0.0312 | 0.0234 | 0.2930 | 0.1992 |
| age_fixed20 | 31 | 0.0352 | 0.0312 | 0.0244 | 0.0312 | 0.0273 | 0.0273 | 0.0273 | 0.0352 | 0.0273 | 0.0273 | 0.0273 | 0.0234 | 0.0234 | 0.0352 | 0.0312 | 0.0391 | 0.0312 | 0.0234 | 0.2725 | 0.1992 |
| age_fixed20 | 32 | 0.0430 | 0.0273 | 0.0273 | 0.0234 | 0.0312 | 0.0234 | 0.0273 | 0.0469 | 0.0195 | 0.0273 | 0.0273 | 0.0273 | 0.0234 | 0.0312 | 0.0391 | 0.0273 | 0.0312 | 0.0220 | 0.2749 | 0.1992 |
| age_fixed20 | 33 | 0.0352 | 0.0312 | 0.0273 | 0.0273 | 0.0273 | 0.0352 | 0.0312 | 0.0430 | 0.0273 | 0.0273 | 0.0312 | 0.0312 | 0.0391 | 0.0234 | 0.0273 | 0.0273 | 0.0273 | 0.0273 | 0.2539 | 0.1992 |
| age_fixed20 | 34 | 0.0391 | 0.0273 | 0.0273 | 0.0273 | 0.0312 | 0.0234 | 0.0273 | 0.0430 | 0.0312 | 0.0234 | 0.0234 | 0.0234 | 0.0352 | 0.0312 | 0.0273 | 0.0273 | 0.0312 | 0.0234 | 0.2773 | 0.1992 |
| age_fixed20 | 35 | 0.0430 | 0.0234 | 0.0273 | 0.0234 | 0.0312 | 0.0273 | 0.0312 | 0.0352 | 0.0312 | 0.0273 | 0.0234 | 0.0195 | 0.0273 | 0.0312 | 0.0273 | 0.0273 | 0.0312 | 0.0195 | 0.2930 | 0.1992 |
| age_sleeparm | 24 | 0.0312 | 0.0312 | 0.0391 | 0.0352 | 0.0273 | 0.0391 | 0.0352 | 0.0508 | 0.0273 | 0.0312 | 0.0312 | 0.0312 | 0.0352 | 0.0469 | 0.0312 | 0.0391 | 0.0352 | 0.0273 | 0.3359 | 0.0391 |
| age_sleeparm | 25 | 0.0352 | 0.0312 | 0.0391 | 0.0352 | 0.0312 | 0.0352 | 0.0352 | 0.0469 | 0.0352 | 0.0312 | 0.0352 | 0.0312 | 0.0273 | 0.0312 | 0.0352 | 0.0352 | 0.0312 | 0.0273 | 0.3516 | 0.0391 |
| age_sleeparm | 26 | 0.0430 | 0.0312 | 0.0352 | 0.0312 | 0.0312 | 0.0312 | 0.0352 | 0.0391 | 0.0352 | 0.0352 | 0.0352 | 0.0312 | 0.0352 | 0.0391 | 0.0312 | 0.0352 | 0.0352 | 0.0273 | 0.3223 | 0.0605 |
| age_sleeparm | 27 | 0.0352 | 0.0273 | 0.0312 | 0.0312 | 0.0312 | 0.0391 | 0.0352 | 0.0508 | 0.0312 | 0.0391 | 0.0273 | 0.0352 | 0.0312 | 0.0391 | 0.0312 | 0.0352 | 0.0391 | 0.0312 | 0.3359 | 0.0430 |
| age_sleeparm | 28 | 0.0391 | 0.0312 | 0.0312 | 0.0391 | 0.0352 | 0.0312 | 0.0352 | 0.0469 | 0.0312 | 0.0352 | 0.0298 | 0.0352 | 0.0391 | 0.0430 | 0.0352 | 0.0352 | 0.0312 | 0.0273 | 0.3179 | 0.0508 |
| age_sleeparm | 29 | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0352 | 0.0312 | 0.0347 | 0.0391 | 0.0352 | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0391 | 0.0352 | 0.0391 | 0.0352 | 0.0273 | 0.3130 | 0.0859 |
| age_sleeparm | 30 | 0.0391 | 0.0312 | 0.0312 | 0.0312 | 0.0352 | 0.0312 | 0.0352 | 0.0352 | 0.0352 | 0.0352 | 0.0312 | 0.0352 | 0.0273 | 0.0352 | 0.0352 | 0.0352 | 0.0352 | 0.0312 | 0.3203 | 0.0742 |
| age_sleeparm | 31 | 0.0391 | 0.0352 | 0.0352 | 0.0312 | 0.0312 | 0.0352 | 0.0352 | 0.0391 | 0.0312 | 0.0312 | 0.0430 | 0.0312 | 0.0352 | 0.0352 | 0.0312 | 0.0391 | 0.0352 | 0.0312 | 0.3047 | 0.0703 |
| age_sleeparm | 32 | 0.0430 | 0.0391 | 0.0273 | 0.0273 | 0.0430 | 0.0312 | 0.0312 | 0.0430 | 0.0312 | 0.0352 | 0.0312 | 0.0352 | 0.0288 | 0.0352 | 0.0430 | 0.0312 | 0.0352 | 0.0273 | 0.3267 | 0.0547 |
| age_sleeparm | 33 | 0.0312 | 0.0312 | 0.0312 | 0.0273 | 0.0312 | 0.0430 | 0.0352 | 0.0508 | 0.0391 | 0.0273 | 0.0273 | 0.0273 | 0.0508 | 0.0273 | 0.0273 | 0.0273 | 0.0312 | 0.0312 | 0.3281 | 0.0742 |
| age_sleeparm | 34 | 0.0430 | 0.0352 | 0.0391 | 0.0312 | 0.0352 | 0.0273 | 0.0352 | 0.0469 | 0.0312 | 0.0312 | 0.0273 | 0.0312 | 0.0391 | 0.0352 | 0.0273 | 0.0312 | 0.0391 | 0.0312 | 0.3281 | 0.0547 |
| age_sleeparm | 35 | 0.0430 | 0.0273 | 0.0312 | 0.0278 | 0.0312 | 0.0312 | 0.0547 | 0.0391 | 0.0391 | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0391 | 0.0273 | 0.0312 | 0.0352 | 0.0312 | 0.3276 | 0.0586 |
| age_fixed_matched | 24 | 0.0312 | 0.0312 | 0.0391 | 0.0352 | 0.0352 | 0.0469 | 0.0352 | 0.0469 | 0.0273 | 0.0312 | 0.0312 | 0.0312 | 0.0352 | 0.0391 | 0.0312 | 0.0391 | 0.0391 | 0.0273 | 0.3281 | 0.0391 |
| age_fixed_matched | 25 | 0.0430 | 0.0273 | 0.0312 | 0.0405 | 0.0273 | 0.0352 | 0.0312 | 0.0469 | 0.0352 | 0.0312 | 0.0352 | 0.0312 | 0.0312 | 0.0352 | 0.0352 | 0.0391 | 0.0352 | 0.0312 | 0.3384 | 0.0391 |
| age_fixed_matched | 26 | 0.0508 | 0.0234 | 0.0312 | 0.0273 | 0.0273 | 0.0430 | 0.0254 | 0.0430 | 0.0312 | 0.0352 | 0.0430 | 0.0273 | 0.0430 | 0.0391 | 0.0273 | 0.0352 | 0.0391 | 0.0234 | 0.3223 | 0.0625 |
| age_fixed_matched | 27 | 0.0352 | 0.0312 | 0.0273 | 0.0312 | 0.0312 | 0.0352 | 0.0430 | 0.0469 | 0.0273 | 0.0352 | 0.0273 | 0.0352 | 0.0312 | 0.0469 | 0.0312 | 0.0273 | 0.0312 | 0.0273 | 0.3555 | 0.0430 |
| age_fixed_matched | 28 | 0.0469 | 0.0273 | 0.0312 | 0.0430 | 0.0273 | 0.0273 | 0.0312 | 0.0547 | 0.0312 | 0.0391 | 0.0273 | 0.0288 | 0.0273 | 0.0430 | 0.0391 | 0.0352 | 0.0312 | 0.0234 | 0.3345 | 0.0508 |
| age_fixed_matched | 29 | 0.0391 | 0.0312 | 0.0312 | 0.0239 | 0.0430 | 0.0312 | 0.0312 | 0.0586 | 0.0273 | 0.0273 | 0.0273 | 0.0234 | 0.0273 | 0.0312 | 0.0352 | 0.0352 | 0.0352 | 0.0273 | 0.3276 | 0.0859 |
| age_fixed_matched | 30 | 0.0508 | 0.0234 | 0.0352 | 0.0312 | 0.0234 | 0.0273 | 0.0352 | 0.0430 | 0.0312 | 0.0273 | 0.0342 | 0.0391 | 0.0273 | 0.0391 | 0.0391 | 0.0312 | 0.0352 | 0.0273 | 0.3252 | 0.0742 |
| age_fixed_matched | 31 | 0.0469 | 0.0352 | 0.0312 | 0.0352 | 0.0312 | 0.0391 | 0.0312 | 0.0430 | 0.0312 | 0.0273 | 0.0391 | 0.0273 | 0.0273 | 0.0430 | 0.0352 | 0.0469 | 0.0312 | 0.0273 | 0.3008 | 0.0703 |
| age_fixed_matched | 32 | 0.0430 | 0.0352 | 0.0312 | 0.0312 | 0.0391 | 0.0312 | 0.0312 | 0.0547 | 0.0273 | 0.0352 | 0.0312 | 0.0391 | 0.0312 | 0.0352 | 0.0391 | 0.0312 | 0.0352 | 0.0273 | 0.3164 | 0.0547 |
| age_fixed_matched | 33 | 0.0352 | 0.0312 | 0.0273 | 0.0273 | 0.0283 | 0.0352 | 0.0352 | 0.0469 | 0.0312 | 0.0312 | 0.0352 | 0.0312 | 0.0469 | 0.0273 | 0.0273 | 0.0352 | 0.0312 | 0.0312 | 0.3311 | 0.0742 |
| age_fixed_matched | 34 | 0.0469 | 0.0312 | 0.0352 | 0.0352 | 0.0352 | 0.0273 | 0.0391 | 0.0586 | 0.0312 | 0.0273 | 0.0273 | 0.0273 | 0.0483 | 0.0352 | 0.0312 | 0.0312 | 0.0352 | 0.0273 | 0.3149 | 0.0547 |
| age_fixed_matched | 35 | 0.0547 | 0.0273 | 0.0273 | 0.0234 | 0.0352 | 0.0352 | 0.0508 | 0.0469 | 0.0391 | 0.0312 | 0.0273 | 0.0234 | 0.0312 | 0.0352 | 0.0312 | 0.0312 | 0.0352 | 0.0273 | 0.3281 | 0.0586 |

## M8 reference

M8-W5, period 128, seeds 12–23, L = 2048. Source: results/m8w5/results.json. prog_age_s is the agent age_fixed20 copies.

| controller | T14 mean | T14 std ddof=0 | W14 mean | W14 std ddof=0 |
|---|---:|---:|---:|---:|
| rand_dwell_s | 2.332399 | 0.221872 | 1.349514 | 0.257101 |
| oracle_fresh_s | 1.991094 | 0.144618 | 1.341186 | 0.305705 |
| prog_chan_s | 1.927012 | 0.189953 | 1.511300 | 0.278899 |
| prog_age_s | 1.745765 | 0.113682 | 1.240120 | 0.308144 |

| seed | prog_age_s T14 | prog_age_s W14 |
|---|---:|---:|
| 12 | 1.785188 | 1.759845 |
| 13 | 1.918485 | 1.282478 |
| 14 | 1.581165 | 0.865502 |
| 15 | 1.570576 | 0.850855 |
| 16 | 1.736055 | 1.361845 |
| 17 | 1.596326 | 0.985695 |
| 18 | 1.736144 | 0.915535 |
| 19 | 1.880330 | 1.469353 |
| 20 | 1.748226 | 0.996176 |
| 21 | 1.698552 | 1.183130 |
| 22 | 1.878288 | 1.670411 |
| 23 | 1.819841 | 1.540619 |

## Kernels

| kernel | seed | wall s | workers |
|---|---:|---:|---:|
| ezzzzzx/m9-w5-s24 | 24 | 496.3 | 3 |
| ezzzzzx/m9-w5-s25 | 25 | 291.5 | 3 |
| ezzzzzx/m9-w5-s26 | 26 | 447.2 | 3 |
| ezzzzzx/m9-w5-s27 | 27 | 505.1 | 3 |
| ezzzzzx/m9-w5-s28 | 28 | 453.5 | 3 |
| ezzzzzx/m9-w5-s29 | 29 | 503.0 | 3 |
| ezzzzzx/m9-w5-s30 | 30 | 477.7 | 3 |
| ezzzzzx/m9-w5-s31 | 31 | 459.9 | 3 |
| ezzzzzx/m9-w5-s32 | 32 | 513.5 | 3 |
| ezzzzzx/m9-w5-s33 | 33 | 291.1 | 3 |
| ezzzzzx/m9-w5-s34 | 34 | 413.5 | 3 |
| ezzzzzx/m9-w5-s35 | 35 | 477.0 | 3 |
| mean ± std ddof=0 |  | 444.1 ± 73.5 |  |

## Smoke test

```
torch=2.15.0+cpu threads=1 model_seed=0 n_ch=19 tau=256 period=128
dose L64 16/64 share 0.250000 checkpoints [8, 16, 24, 32, 40, 48, 56, 64]
dose L128 24/128 share 0.187500 checkpoints [16, 32, 48, 64, 80, 96, 112, 128]
dose L256 48/256 share 0.187500 checkpoints [32, 64, 96, 128, 160, 192, 224, 256]
dose L512 104/512 share 0.203125 checkpoints [64, 128, 192, 256, 320, 384, 448, 512]
dose L2048 408/2048 share 0.199219 checkpoints [256, 512, 768, 1024, 1280, 1536, 1792, 2048]
smoke check 1: age_fixed20 reproduces M8 prog_age_s seed 12
  L=64 age_fixed20 T14 13.124334 W14 11.472060 prog_age_s T14 13.124334 W14 11.472060 abs_dT14 0.000e+00 limit 1.0e-06 wall 21.19s m8_wall 21.59s
  L=256 age_fixed20 T14 6.237725 W14 3.457408 prog_age_s T14 6.237725 W14 3.457408 abs_dT14 0.000e+00 limit 1.0e-06 wall 47.53s m8_wall 46.64s
smoke check 2: age_sleeparm seed 24 L=256 stores and sleep state
  overlap 0 exam_drawn 0 sleep_decisions 8 sleep_checks 1 intact True param_delta 0.012209586799144745 replay_draws 8 wall 35.45s
smoke check 3: age_sleeparm seed 24 L=512 visit trace
visit 1-8 arm M7a decisions 8 reward 35.082319 sleep_aged none best_room none best_aged none
visit 9-17 arm M37a decisions 9 reward 81.444784 sleep_aged none best_room M7a best_aged 35.082319
visit 18-26 arm N2 decisions 9 reward 119.411499 sleep_aged none best_room M37a best_aged 81.444784
visit 27-35 arm E110b decisions 9 reward 38.586398 sleep_aged none best_room N2 best_aged 119.411499
visit 36-44 arm Gb decisions 9 reward 175.942352 sleep_aged none best_room N2 best_aged 119.411499
visit 45-53 arm C7b decisions 9 reward 9.378769 sleep_aged none best_room Gb best_aged 175.942352
visit 54-62 arm N3 decisions 9 reward 57.070074 sleep_aged none best_room Gb best_aged 175.942352
visit 63-71 arm M37b decisions 9 reward 98.397838 sleep_aged none best_room Gb best_aged 175.942352
visit 72-80 arm E30b decisions 9 reward 21.715052 sleep_aged none best_room Gb best_aged 175.942352
visit 81-89 arm Ga decisions 9 reward 17.353843 sleep_aged none best_room Gb best_aged 175.942352
visit 90-98 arm C7a decisions 9 reward 76.832102 sleep_aged none best_room Gb best_aged 175.942352
visit 99-107 arm E110a decisions 9 reward 34.679486 sleep_aged none best_room Gb best_aged 175.942352
visit 108-116 arm N1 decisions 9 reward 51.372573 sleep_aged none best_room Gb best_aged 175.942352
visit 117-125 arm E30a decisions 9 reward 29.349387 sleep_aged none best_room Gb best_aged 175.942352
visit 126-134 arm C1a decisions 9 reward 25.563085 sleep_aged none best_room Gb best_aged 175.942352
visit 135-143 arm Z decisions 9 reward 0.015212 sleep_aged none best_room Gb best_aged 175.942352
visit 144-152 arm C1b decisions 9 reward 6.057606 sleep_aged none best_room Gb best_aged 175.942352
visit 153-161 arm M7b decisions 9 reward 45.234804 sleep_aged none best_room Gb best_aged 175.942352
visit 162-169 arm sleep decisions 8 reward 51.957112 sleep_aged none best_room Gb best_aged 175.942352
visit 170-180 arm C1a decisions 11 reward 2.986374 sleep_aged 51.957112 best_room Gb best_aged 175.942352
visit 181-196 arm Gb decisions 16 reward 35.156682 sleep_aged 57.171767 best_room Gb best_aged 175.942352
visit 197-206 arm N2 decisions 10 reward 71.295472 sleep_aged 64.367713 best_room N2 best_aged 146.842762
visit 207-216 arm Gb decisions 10 reward 44.424233 sleep_aged 68.642070 best_room Gb best_aged 135.324675
visit 217-227 arm M37a decisions 11 reward 22.922424 sleep_aged 72.752680 best_room M37a best_aged 132.508882
visit 228-241 arm M37b decisions 14 reward 21.228937 sleep_aged 77.092700 best_room M37b best_aged 133.782131
visit 242-256 arm M7a decisions 15 reward 27.549883 sleep_aged 82.353383 best_room M7a best_aged 119.251635
visit 257-272 arm C7a decisions 16 reward 49.807572 sleep_aged 87.679548 best_room C7a best_aged 122.476383
visit 273-284 arm N3 decisions 12 reward 68.782549 sleep_aged 93.027121 best_room N3 best_aged 123.603596
visit 285-295 arm E110b decisions 11 reward 15.501292 sleep_aged 96.824086 best_room E110b best_aged 124.011165
visit 296-304 arm N2 decisions 9 reward 74.998286 sleep_aged 100.151695 best_room N2 best_aged 125.816350
visit 305-314 arm Gb decisions 10 reward 76.605768 sleep_aged 102.769917 best_room Gb best_aged 126.999541
visit 315-330 arm N1 decisions 16 reward 30.815870 sleep_aged 105.573109 best_room N1 best_aged 118.462842
visit 331-345 arm C7b decisions 15 reward 5.885456 sleep_aged 109.836566 best_room C7b best_aged 119.492884
visit 346-356 arm E30b decisions 11 reward 21.149024 sleep_aged 113.598658 best_room E30b best_aged 121.165306
visit 357-367 arm E110a decisions 11 reward 5.330485 sleep_aged 116.220751 best_room E110a best_aged 122.534050
visit 368-377 arm Ga decisions 10 reward 78.403153 sleep_aged 118.732563 best_room Ga best_aged 122.405226
visit 378-389 arm E30a decisions 12 reward 25.334315 sleep_aged 120.924235 best_room E30a best_aged 121.164563
visit 390-397 arm sleep decisions 8 reward -20.589018 sleep_aged 123.443698 best_room M7b best_aged 122.300082
visit 398-408 arm C7a decisions 11 reward 44.862012 sleep_aged 30.193273 best_room M7b best_aged 123.950481
visit 409-423 arm M7b decisions 15 reward 19.126128 sleep_aged 36.323286 best_room M7b best_aged 126.137191
visit 424-439 arm M37b decisions 16 reward 129.414745 sleep_aged 44.269032 best_room M37b best_aged 126.482539
visit 440-452 arm N2 decisions 13 reward -5.163999 sleep_aged 52.246715 best_room N2 best_aged 128.752799
visit 453-462 arm Gb decisions 10 reward 100.454366 sleep_aged 58.371311 best_room Gb best_aged 130.378116
visit 463-473 arm M37a decisions 11 reward 86.775375 sleep_aged 62.875387 best_room M37a best_aged 131.195790
visit 474-485 arm Z decisions 12 reward 0.216927 sleep_aged 67.630833 best_room Z best_aged 127.469328
visit 486-494 arm C1b decisions 9 reward 20.740094 sleep_aged 72.590779 best_room C1b best_aged 129.679514
visit 495-504 arm C1a decisions 10 reward 11.008184 sleep_aged 76.161105 best_room C1a best_aged 129.849735
visit 505-512 arm E30a decisions 8 reward 48.545067 sleep_aged 79.983665 best_room N3 best_aged 127.096474
  sleep share by quarter 0.000000 0.062500 0.000000 0.062500
  sleep share overall 0.031250 decisions 16 visits 2 T14 3.716398 W14 2.284949 wall 77.27s learn 30.66s score 35.03s eval 11.22s
  one L2048 age_sleeparm 274.0s age_fixed20 267.3s wave 274.0s kernel 547.9s guarded 1095.8s limit 5400s
smoke check 4: four controllers, seed 24, L=128, finite T14
  age_nosleep T14 8.781136 W14 4.438975 finite True sleep 0.0000 decisions 0 corridor 0.1094 swaps 1 wall 29.43s
  age_fixed20 T14 9.569980 W14 4.562542 finite True sleep 0.1875 decisions 24 corridor 0.0859 swaps 1 wall 29.15s
  age_sleeparm T14 8.781136 W14 4.438975 finite True sleep 0.0000 decisions 0 corridor 0.1094 swaps 1 wall 23.19s
  age_fixed_matched T14 8.781136 W14 4.438975 finite True sleep 0.0000 decisions 0 corridor 0.1094 swaps 1 wall 23.09s
```

Smoke wall 268.7s.

```
2026-11-14T04:59:47Z dataset create ezzzzzx/pdp-code-m9w5
2026-11-14T05:00:15Z dataset ready ezzzzzx/pdp-code-m9w5
2026-11-14T05:00:15Z push ezzzzzx/m9-w5-s24 try 1
2026-11-14T05:00:18Z pushed ezzzzzx/m9-w5-s24 try 1
2026-11-14T05:00:18Z push ezzzzzx/m9-w5-s25 try 1
2026-11-14T05:00:21Z pushed ezzzzzx/m9-w5-s25 try 1
2026-11-14T05:00:21Z push ezzzzzx/m9-w5-s26 try 1
2026-11-14T05:00:25Z pushed ezzzzzx/m9-w5-s26 try 1
2026-11-14T05:00:25Z push ezzzzzx/m9-w5-s27 try 1
2026-11-14T05:00:29Z pushed ezzzzzx/m9-w5-s27 try 1
2026-11-14T05:00:29Z push ezzzzzx/m9-w5-s28 try 1
2026-11-14T05:00:33Z pushed ezzzzzx/m9-w5-s28 try 1
2026-11-14T05:00:35Z status ezzzzzx/m9-w5-s24 pushed -> RUNNING
2026-11-14T05:00:37Z status ezzzzzx/m9-w5-s25 pushed -> RUNNING
2026-11-14T05:00:38Z status ezzzzzx/m9-w5-s26 pushed -> RUNNING
2026-11-14T05:00:40Z status ezzzzzx/m9-w5-s27 pushed -> RUNNING
2026-11-14T05:00:42Z status ezzzzzx/m9-w5-s28 pushed -> RUNNING
2026-11-14T05:05:54Z status ezzzzzx/m9-w5-s25 RUNNING -> COMPLETE
2026-11-14T05:05:57Z downloaded ezzzzzx/m9-w5-s25 -> results/m9w5/s25/m9_s25.json
2026-11-14T05:06:33Z push ezzzzzx/m9-w5-s29 try 1
2026-11-14T05:06:36Z pushed ezzzzzx/m9-w5-s29 try 1
2026-11-14T05:06:45Z status ezzzzzx/m9-w5-s29 pushed -> RUNNING
2026-11-14T05:08:39Z status ezzzzzx/m9-w5-s26 RUNNING -> COMPLETE
2026-11-14T05:08:42Z downloaded ezzzzzx/m9-w5-s26 -> results/m9w5/s26/m9_s26.json
2026-11-14T05:08:46Z status ezzzzzx/m9-w5-s28 RUNNING -> COMPLETE
2026-11-14T05:08:48Z downloaded ezzzzzx/m9-w5-s28 -> results/m9w5/s28/m9_s28.json
2026-11-14T05:09:20Z push ezzzzzx/m9-w5-s30 try 1
2026-11-14T05:09:23Z pushed ezzzzzx/m9-w5-s30 try 1
2026-11-14T05:09:23Z push ezzzzzx/m9-w5-s31 try 1
2026-11-14T05:09:26Z pushed ezzzzzx/m9-w5-s31 try 1
2026-11-14T05:09:29Z status ezzzzzx/m9-w5-s24 RUNNING -> COMPLETE
2026-11-14T05:09:31Z downloaded ezzzzzx/m9-w5-s24 -> results/m9w5/s24/m9_s24.json
2026-11-14T05:09:33Z status ezzzzzx/m9-w5-s27 RUNNING -> COMPLETE
2026-11-14T05:09:37Z downloaded ezzzzzx/m9-w5-s27 -> results/m9w5/s27/m9_s27.json
2026-11-14T05:09:40Z status ezzzzzx/m9-w5-s30 pushed -> RUNNING
2026-11-14T05:09:42Z status ezzzzzx/m9-w5-s31 pushed -> RUNNING
2026-11-14T05:10:12Z push ezzzzzx/m9-w5-s32 try 1
2026-11-14T05:10:15Z pushed ezzzzzx/m9-w5-s32 try 1
2026-11-14T05:10:15Z push ezzzzzx/m9-w5-s33 try 1
2026-11-14T05:10:19Z pushed ezzzzzx/m9-w5-s33 try 1
2026-11-14T05:10:26Z status ezzzzzx/m9-w5-s32 pushed -> RUNNING
2026-11-14T05:10:29Z status ezzzzzx/m9-w5-s33 pushed -> RUNNING
2026-11-14T05:15:39Z status ezzzzzx/m9-w5-s29 RUNNING -> COMPLETE
2026-11-14T05:15:42Z downloaded ezzzzzx/m9-w5-s29 -> results/m9w5/s29/m9_s29.json
2026-11-14T05:15:50Z status ezzzzzx/m9-w5-s33 RUNNING -> COMPLETE
2026-11-14T05:15:54Z downloaded ezzzzzx/m9-w5-s33 -> results/m9w5/s33/m9_s33.json
2026-11-14T05:16:24Z push ezzzzzx/m9-w5-s34 try 1
2026-11-14T05:16:27Z pushed ezzzzzx/m9-w5-s34 try 1
2026-11-14T05:16:27Z push ezzzzzx/m9-w5-s35 try 1
2026-11-14T05:16:31Z pushed ezzzzzx/m9-w5-s35 try 1
2026-11-14T05:16:39Z status ezzzzzx/m9-w5-s34 pushed -> RUNNING
2026-11-14T05:16:42Z status ezzzzzx/m9-w5-s35 pushed -> RUNNING
2026-11-14T05:17:53Z status ezzzzzx/m9-w5-s30 RUNNING -> COMPLETE
2026-11-14T05:17:56Z downloaded ezzzzzx/m9-w5-s30 -> results/m9w5/s30/m9_s30.json
2026-11-14T05:17:59Z status ezzzzzx/m9-w5-s31 RUNNING -> COMPLETE
2026-11-14T05:18:02Z downloaded ezzzzzx/m9-w5-s31 -> results/m9w5/s31/m9_s31.json
2026-11-14T05:19:15Z status ezzzzzx/m9-w5-s32 RUNNING -> COMPLETE
2026-11-14T05:19:17Z downloaded ezzzzzx/m9-w5-s32 -> results/m9w5/s32/m9_s32.json
2026-11-14T05:23:51Z status ezzzzzx/m9-w5-s34 RUNNING -> COMPLETE
2026-11-14T05:23:54Z downloaded ezzzzzx/m9-w5-s34 -> results/m9w5/s34/m9_s34.json
2026-11-14T05:25:01Z status ezzzzzx/m9-w5-s35 RUNNING -> COMPLETE
2026-11-14T05:25:03Z downloaded ezzzzzx/m9-w5-s35 -> results/m9w5/s35/m9_s35.json
2026-11-14T05:25:03Z all 12 kernels downloaded
```

## Protocol

- World5(world_seed, period=128). Room names, ring order, noise rooms and Z are W4's. After every positive multiple of 128 decisions, one of the 14 learnable rooms is redrawn with default_rng(world_seed + 80000), as in W5.
- The cortex, disjoint exam and replay reservoirs, coin, movement and dwell are M8's, which are W5's. World steps use default_rng(1000 + world_seed). Choices use default_rng(10000 + world_seed). A tie on the ring goes left. Equal distances among unvisited rooms break toward +index (right). Dwell is 8 stays.
- One sleep decision draws one replay glance and takes one trace-learning Adam step from a zero state. The live state and traces are saved and restored. The first sleep visit of a life checks that they are bit-identical afterwards. Sleep draws no exam glance.
- age_fixed20 calls the same room choice, sleep plan and sleep visit as M8 prog_age_s. The dose at L = 2048 is 408 decisions.
- age_sleeparm's optimistic first pass visits the nearest unvisited room, sleep last. Then it is eps-greedy over the 19 arms. The greedy rank uses the aged value. A room visit's reward is M8's per-room exam progress, travel included, divided by the visit's decisions. A sleep visit's reward is the sum, over its decisions, of whole-exam-store bits before minus bits after, divided by the number of decisions in the visit.
- age_fixed_matched rounds age_sleeparm's sleep decisions half up onto a multiple of 8 (floor(n/8 + 1/2) visits) and places those visits with m7.sleep_plan and w4.sleep_due, as age_fixed20 does.
- A checkpoint scores w4.evaluate_w14 after that decision's swap. T14 is the mean of the eight checkpoints. The tail mean drops the checkpoint at decision 256.
- Stale draws, sleep quarters, swap gaps and visit rewards are defined in the sections above.
- World seeds 24–35. L = 2048. Period 128. CPU. Private Kaggle kernels, one seed each. age_sleeparm runs first, then the other three in a pool of 3. No GPU, no internet. Nothing retuned.


## Verdict (orchestrator, 14 Nov 2026, UTC+8)

1. **Sleep pays in a drifting world:** age_fixed20 − age_nosleep = −0.319 on T14 (12 of 12). Without the first checkpoint it is −0.291 (12 of 12). **PASS.**
2. **Choosing when matters here:** age_sleeparm − age_fixed_matched = +0.016 (sleeparm lower on 7 of 12). **FAIL.**
Report only: age_sleeparm − age_fixed20 = +0.164 (1 of 12). The agent chooses to sleep 0.059 of its life, against the 0.20 that the fixed dose shows is
better. Sleep onsets do not track swaps: 61.8 decisions after the last swap, against 64.4 for room visits. A fifth of replay draws rehearse
rules already swapped out, yet sleep still pays.

Checks: age_fixed20 reproduces M8's prog_age_s exactly. The stores are disjoint, sleep drew no exam glance, and the live state and traces are restored bit
for bit.

**Mechanism.** At the same dose, timing adds nothing (+0.016), as in M7 on a stationary world. The new finding is the dose. Left to choose, the agent
sleeps 3.4× too little and loses 0.16. Its reward for sleep is the exam-store progress during the sleep visit itself. But most of what sleep buys
is retention, bits not lost later to interference, and progress measured on the spot cannot see that. Immediate compression progress
undervalues consolidation. The kill rule for timing applies: organic sleep timing is closed until a world makes rest costly at some times and cheap at others.
How much to sleep stays open. It needs a reward that credits loss prevented, not only gain made.
