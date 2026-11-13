# W2-1 headroom

Fixed policies on the locked-rooms world with a way back from room 8. Bits are prequential, on observation tokens only.

## PROBE at T

PROBE is the mean of probe_k over rooms 0..8. The probe is at step 60000.

| world_seed | PROBE rand8 | PROBE rand6 | PROBE oracle | PROBE oracle_deep | rand8 − oracle | rand6 − oracle | rand8 − rand6 | rand6 − oracle_deep |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.006292 | 0.004037 | 0.007386 | 0.005295 | 9.998906 | -0.003349 | 10.002255 | -0.001258 |
| 1 | 9.931172 | 0.015296 | 0.027083 | 0.005193 | 9.904089 | -0.011787 | 9.915876 | 0.010103 |
| 2 | 8.921240 | 0.024417 | 0.016091 | 0.004474 | 8.905149 | 0.008326 | 8.896823 | 0.019943 |
| mean |  |  |  |  | 9.602715 | -0.002270 | 9.604985 | 0.009596 |
| std ddof=0 |  |  |  |  | 0.494770 | 0.008246 | 0.501986 | 0.008663 |

## PROBE_cap at T

PROBE_cap is the mean over rooms 0..8 of min(probe_k, 4).

| world_seed | PROBE_cap rand8 | PROBE_cap rand6 | PROBE_cap oracle | PROBE_cap oracle_deep |
|---:|---:|---:|---:|---:|
| 0 | 3.122387 | 0.004037 | 0.007386 | 0.005295 |
| 1 | 3.125205 | 0.015296 | 0.027083 | 0.005193 |
| 2 | 3.117261 | 0.024417 | 0.016091 | 0.004474 |

## Per-room capped probe at T

Each cell is min(probe_k, 4) at step 60000.

| world_seed | policy | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | PROBE_cap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | rand8 | 0.002817 | 0.098665 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 3.122387 |
| 0 | rand6 | 0.004822 | 0.001749 | 0.002425 | 0.003844 | 0.001506 | 0.004744 | 0.002920 | 0.001620 | 0.012704 | 0.004037 |
| 0 | oracle | 0.002731 | 0.001728 | 0.003595 | 0.002597 | 0.004039 | 0.003844 | 0.003832 | 0.002159 | 0.041944 | 0.007386 |
| 0 | oracle_deep | 0.005978 | 0.002458 | 0.006084 | 0.010855 | 0.004972 | 0.007512 | 0.004750 | 0.003396 | 0.001652 | 0.005295 |
| 1 | rand8 | 0.001744 | 0.125100 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 3.125205 |
| 1 | rand6 | 0.090115 | 0.002561 | 0.003841 | 0.002640 | 0.002376 | 0.005272 | 0.014552 | 0.005317 | 0.010990 | 0.015296 |
| 1 | oracle | 0.001940 | 0.003656 | 0.003062 | 0.003803 | 0.005159 | 0.004494 | 0.003438 | 0.003219 | 0.214978 | 0.027083 |
| 1 | oracle_deep | 0.003696 | 0.002899 | 0.006151 | 0.007161 | 0.005974 | 0.008052 | 0.006070 | 0.004916 | 0.001818 | 0.005193 |
| 2 | rand8 | 0.001464 | 0.053889 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 3.117261 |
| 2 | rand6 | 0.170474 | 0.002997 | 0.005010 | 0.001725 | 0.003390 | 0.002089 | 0.002124 | 0.003631 | 0.028309 | 0.024417 |
| 2 | oracle | 0.002462 | 0.003831 | 0.005994 | 0.002515 | 0.003443 | 0.003413 | 0.002842 | 0.004488 | 0.115831 | 0.016091 |
| 2 | oracle_deep | 0.003668 | 0.003563 | 0.007243 | 0.002984 | 0.006138 | 0.004197 | 0.007737 | 0.002951 | 0.001780 | 0.004474 |

## Gaps on PROBE_cap

| world_seed | rand6 − oracle | rand6 − oracle_deep | rand8 − rand6 |
|---:|---:|---:|---:|
| 0 | -0.003349 | -0.001258 | 3.118350 |
| 1 | -0.011787 | 0.010103 | 3.109909 |
| 2 | 0.008326 | 0.019943 | 3.092845 |
| mean | -0.002270 | 0.009596 | 3.107034 |
| std ddof=0 | 0.008246 | 0.008663 | 0.010609 |

## rand6 cycles

A full cycle is a manipulation in room 8 that returns the agent to room 0. Reset is not a cycle.

| world_seed | full_cycles | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 25 | 5505 | 6799 | 9206 | 4332 | 9357 | 5471 | 4256 | 7499 | 7575 |
| 1 | 27 | 8583 | 6794 | 6129 | 7322 | 7740 | 5656 | 4636 | 5176 | 7964 |
| 2 | 28 | 8500 | 7851 | 5966 | 8271 | 5821 | 6512 | 4927 | 5279 | 6873 |

## W2-0 comparison

W2-0 is the locked-rooms world with no exit from room 8. Same seeds, rand6 and oracle. PROBE_cap is recomputed from that run as the mean of min(probe_k, 4).

| world_seed | W2-1 PROBE rand6 | W2-0 PROBE rand6 | W2-1 PROBE oracle | W2-0 PROBE oracle | W2-1 PROBE_cap rand6 | W2-0 PROBE_cap rand6 | W2-1 PROBE_cap oracle | W2-0 PROBE_cap oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.004037 | 1.782124 | 0.007386 | 0.005676 | 0.004037 | 1.573478 | 0.007386 | 0.005676 |
| 1 | 0.015296 | 0.501576 | 0.027083 | 0.004577 | 0.015296 | 0.501576 | 0.027083 | 0.004577 |
| 2 | 0.024417 | 1.772205 | 0.016091 | 0.005486 | 0.024417 | 0.762446 | 0.016091 | 0.005486 |

## Room 8 probe actions

Count of probe actions that differ from the W2-0 room-8 probe. The prefix is reset plus the keys into room 8. The scored part is the 96 stay actions.

| world_seed | n_actions | n_differ | prefix_differ | scored_differ |
|---:|---:|---:|---:|---:|
| 0 | 121 | 1 | 0 | 1 |
| 1 | 121 | 0 | 0 | 0 |
| 2 | 121 | 1 | 0 | 1 |

## Lives

### seed 0 rand8

Bits/observation 0.615821. Bits 36949.273. Wall seconds 88.551. Replay steps 1875. Finite True.

Full cycles 0. Deepest room 2. First step 9815.

Time in room, rooms 0..8: 58952 1019 29 0 0 0 0 0 0.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 13180.614 | 0.878708 |
| 1 | 15000 | 30000 | 15000 | 8093.478 | 0.539565 |
| 2 | 30000 | 45000 | 15000 | 7777.731 | 0.518515 |
| 3 | 45000 | 60000 | 15000 | 7897.451 | 0.526497 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.016140 | 0.602992 | 17.865785 | 12.357339 | 12.352929 | 18.700069 | 18.511328 | 15.726880 | 18.441442 | 12.730545 | 3.179904 | 4.065536 |
| 30000 | 30016 | 0.006673 | 0.275102 | 10.438973 | 12.750709 | 12.357718 | 13.002944 | 14.086063 | 12.341172 | 13.635241 | 9.877177 | 3.142419 | 7.843721 |
| 60000 | 60000 | 0.002817 | 0.098665 | 4.876034 | 13.150642 | 13.168119 | 14.375558 | 16.598016 | 12.768267 | 15.018510 | 10.006292 | 3.122387 | 5.099972 |

Token sha256 `6ce3ff7d4ef26e807cc5451ca4fce6dd66efa4c703322b6ab84f6e3f4bd86666`.

### seed 0 rand6

Bits/observation 0.479354. Bits 28761.244. Wall seconds 80.760. Replay steps 1875. Finite True.

Full cycles 25. Deepest room 8. First step 947.

Time in room, rooms 0..8: 5505 6799 9206 4332 9357 5471 4256 7499 7575.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 26350.830 | 1.756722 |
| 1 | 15000 | 30000 | 15000 | 2106.818 | 0.140455 |
| 2 | 30000 | 45000 | 15000 | 213.114 | 0.014208 |
| 3 | 45000 | 60000 | 15000 | 90.483 | 0.006032 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.187868 | 0.044618 | 0.119482 | 0.286191 | 0.045550 | 0.126055 | 0.175436 | 0.091823 | 1.519790 | 0.288535 | 0.288535 | 10.953415 |
| 30000 | 30016 | 0.029196 | 0.012919 | 0.018205 | 0.020559 | 0.013512 | 0.029706 | 0.026569 | 0.017886 | 0.059721 | 0.025364 | 0.025364 | 11.303158 |
| 60000 | 60000 | 0.004822 | 0.001749 | 0.002425 | 0.003844 | 0.001506 | 0.004744 | 0.002920 | 0.001620 | 0.012704 | 0.004037 | 0.004037 | 11.833159 |

Token sha256 `f105da9d1e089fc7ffa2cf38967ea6ba4dceb2349ad66f335aac077716617fdd`.

### seed 0 oracle

Bits/observation 0.639133. Bits 38347.993. Wall seconds 82.093. Replay steps 1875. Finite True.

Full cycles 11. Deepest room 8. First step 4120.

Time in room, rooms 0..8: 10295 6890 6812 6734 6656 6578 6500 6422 3113.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 35832.571 | 2.388838 |
| 1 | 15000 | 30000 | 15000 | 2006.059 | 0.133737 |
| 2 | 30000 | 45000 | 15000 | 383.601 | 0.025573 |
| 3 | 45000 | 60000 | 15000 | 125.763 | 0.008384 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.088747 | 0.052433 | 0.203847 | 0.107061 | 0.172099 | 0.153735 | 0.167271 | 0.122876 | 1.664368 | 0.303604 | 0.303604 | 9.428259 |
| 30000 | 30016 | 0.014201 | 0.009144 | 0.024182 | 0.013960 | 0.022366 | 0.026570 | 0.024542 | 0.014340 | 0.456485 | 0.067310 | 0.067310 | 10.090899 |
| 60000 | 60000 | 0.002731 | 0.001728 | 0.003595 | 0.002597 | 0.004039 | 0.003844 | 0.003832 | 0.002159 | 0.041944 | 0.007386 | 0.007386 | 11.023501 |

Token sha256 `2115ff842f35181f42626c6267549a1650d89721923a304b29fb7d28e1e901c5`.

### seed 0 oracle_deep

Bits/observation 0.667690. Bits 40061.399. Wall seconds 81.269. Replay steps 1875. Finite True.

Full cycles 0. Deepest room 8. First step 24.

Time in room, rooms 0..8: 4426 4378 3914 3782 3740 3698 3656 3614 28792.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 38818.156 | 2.587877 |
| 1 | 15000 | 30000 | 15000 | 988.258 | 0.065884 |
| 2 | 30000 | 45000 | 15000 | 185.305 | 0.012354 |
| 3 | 45000 | 60000 | 15000 | 69.680 | 0.004645 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.254350 | 0.133414 | 0.255386 | 0.184313 | 0.211978 | 0.270103 | 0.407518 | 1.007854 | 0.039430 | 0.307150 | 0.307150 | 9.215553 |
| 30000 | 30016 | 0.045639 | 0.017024 | 0.030046 | 0.029656 | 0.032087 | 0.053955 | 0.036328 | 0.024670 | 0.007521 | 0.030769 | 0.030769 | 9.652263 |
| 60000 | 60000 | 0.005978 | 0.002458 | 0.006084 | 0.010855 | 0.004972 | 0.007512 | 0.004750 | 0.003396 | 0.001652 | 0.005295 | 0.005295 | 10.222065 |

Token sha256 `ae358160bbcd9605f55334995bbea798ede82b155d46337903c11c1b7a3ae265`.

### seed 1 rand8

Bits/observation 0.613875. Bits 36832.515. Wall seconds 120.657. Replay steps 1875. Finite True.

Full cycles 0. Deepest room 2. First step 27012.

Time in room, rooms 0..8: 59004 964 32 0 0 0 0 0 0.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 13337.635 | 0.889176 |
| 1 | 15000 | 30000 | 15000 | 7926.305 | 0.528420 |
| 2 | 30000 | 45000 | 15000 | 8036.009 | 0.535734 |
| 3 | 45000 | 60000 | 15000 | 7532.566 | 0.502171 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.013367 | 0.864650 | 15.705412 | 15.701598 | 17.072610 | 16.596728 | 14.672080 | 13.562784 | 17.304892 | 12.388236 | 3.208669 | 4.119691 |
| 30000 | 30016 | 0.004276 | 0.372341 | 10.716378 | 13.703830 | 15.045978 | 14.183175 | 13.536373 | 12.914995 | 15.847739 | 10.702787 | 3.152957 | 4.136225 |
| 60000 | 60000 | 0.001744 | 0.125100 | 4.803557 | 14.231876 | 14.203723 | 14.878694 | 14.386463 | 13.629028 | 13.120363 | 9.931172 | 3.125205 | 4.067139 |

Token sha256 `9bc79f36d8aa7d73abf185d85532859fd78c67bb1b1da753639553c4ef020663`.

### seed 1 rand6

Bits/observation 0.566186. Bits 33971.186. Wall seconds 112.746. Replay steps 1875. Finite True.

Full cycles 27. Deepest room 8. First step 1476.

Time in room, rooms 0..8: 8583 6794 6129 7322 7740 5656 4636 5176 7964.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 31048.715 | 2.069914 |
| 1 | 15000 | 30000 | 15000 | 2524.611 | 0.168307 |
| 2 | 30000 | 45000 | 15000 | 282.079 | 0.018805 |
| 3 | 45000 | 60000 | 15000 | 115.781 | 0.007719 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.131429 | 0.129741 | 0.119585 | 0.123508 | 0.090402 | 0.551772 | 0.289202 | 0.120341 | 1.263068 | 0.313227 | 0.313227 | 6.660550 |
| 30000 | 30016 | 0.083754 | 0.017927 | 0.021395 | 0.016831 | 0.017729 | 0.051423 | 0.066545 | 0.027273 | 0.164723 | 0.051955 | 0.051955 | 7.818742 |
| 60000 | 60000 | 0.090115 | 0.002561 | 0.003841 | 0.002640 | 0.002376 | 0.005272 | 0.014552 | 0.005317 | 0.010990 | 0.015296 | 0.015296 | 8.581928 |

Token sha256 `ea431587a01092e906e713f096b471349b3391f24c0c7a4af448131c80a71071`.

### seed 1 oracle

Bits/observation 0.648856. Bits 38931.343. Wall seconds 112.859. Replay steps 1875. Finite True.

Full cycles 11. Deepest room 8. First step 4120.

Time in room, rooms 0..8: 10866 6890 6812 6734 6656 6578 6500 6422 2542.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 34266.971 | 2.284465 |
| 1 | 15000 | 30000 | 15000 | 3550.219 | 0.236681 |
| 2 | 30000 | 45000 | 15000 | 751.169 | 0.050078 |
| 3 | 45000 | 60000 | 15000 | 362.984 | 0.024199 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.024358 | 0.049371 | 0.115863 | 0.225834 | 0.220244 | 0.212775 | 0.213728 | 0.151076 | 5.095670 | 0.700991 | 0.579250 | 7.511297 |
| 30000 | 30016 | 0.007759 | 0.012105 | 0.015522 | 0.037705 | 0.036123 | 0.032199 | 0.020648 | 0.017574 | 1.288052 | 0.163076 | 0.163076 | 8.302843 |
| 60000 | 60000 | 0.001940 | 0.003656 | 0.003062 | 0.003803 | 0.005159 | 0.004494 | 0.003438 | 0.003219 | 0.214978 | 0.027083 | 0.027083 | 9.501735 |

Token sha256 `352a0bc5550c0623c381f17d72019085e721a3441426643029dd76e98bd98879`.

### seed 1 oracle_deep

Bits/observation 0.699768. Bits 41986.064. Wall seconds 112.617. Replay steps 1875. Finite True.

Full cycles 0. Deepest room 8. First step 24.

Time in room, rooms 0..8: 4426 4378 3914 3782 3740 3698 3656 3614 28792.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 40467.984 | 2.697866 |
| 1 | 15000 | 30000 | 15000 | 1190.872 | 0.079391 |
| 2 | 30000 | 45000 | 15000 | 240.179 | 0.016012 |
| 3 | 45000 | 60000 | 15000 | 87.029 | 0.005802 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.118730 | 0.186145 | 0.185148 | 0.257160 | 0.239549 | 0.610101 | 0.415849 | 0.765086 | 0.048447 | 0.314024 | 0.314024 | 6.815073 |
| 30000 | 30016 | 0.021437 | 0.022056 | 0.034938 | 0.055255 | 0.054651 | 0.073640 | 0.054566 | 0.035927 | 0.009909 | 0.040264 | 0.040264 | 7.592535 |
| 60000 | 60000 | 0.003696 | 0.002899 | 0.006151 | 0.007161 | 0.005974 | 0.008052 | 0.006070 | 0.004916 | 0.001818 | 0.005193 | 0.005193 | 8.746883 |

Token sha256 `4e14674b04bfc04c69469f89883d51026ce432bb6185d368ba8e13694d9020aa`.

### seed 2 rand8

Bits/observation 0.641083. Bits 38465.004. Wall seconds 120.886. Replay steps 1875. Finite True.

Full cycles 0. Deepest room 4. First step 29385.

Time in room, rooms 0..8: 58512 1450 18 16 4 0 0 0 0.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 14539.880 | 0.969325 |
| 1 | 15000 | 30000 | 15000 | 8539.635 | 0.569309 |
| 2 | 30000 | 45000 | 15000 | 7684.858 | 0.512324 |
| 3 | 45000 | 60000 | 15000 | 7700.630 | 0.513375 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.007424 | 0.687811 | 14.901891 | 13.725719 | 12.064781 | 12.563289 | 15.422305 | 11.597006 | 13.278875 | 10.472122 | 3.188359 | 4.109923 |
| 30000 | 30016 | 0.003272 | 0.362487 | 10.772615 | 10.588192 | 11.066559 | 11.677477 | 13.543830 | 11.141281 | 11.635126 | 8.976760 | 3.151751 | 4.035498 |
| 60000 | 60000 | 0.001464 | 0.053889 | 7.860346 | 9.310601 | 11.212041 | 12.557273 | 14.899076 | 11.965075 | 12.431396 | 8.921240 | 3.117261 | 4.042067 |

Token sha256 `81bc8883f4d73e8bc96aeb839b33daaea54a6b80b76fcc9c341f3d12721fa0b2`.

### seed 2 rand6

Bits/observation 0.561095. Bits 33665.723. Wall seconds 114.097. Replay steps 1875. Finite True.

Full cycles 28. Deepest room 8. First step 1836.

Time in room, rooms 0..8: 8500 7851 5966 8271 5821 6512 4927 5279 6873.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 31218.427 | 2.081228 |
| 1 | 15000 | 30000 | 15000 | 1947.020 | 0.129801 |
| 2 | 30000 | 45000 | 15000 | 370.372 | 0.024691 |
| 3 | 45000 | 60000 | 15000 | 129.903 | 0.008660 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.355740 | 0.190068 | 0.369937 | 0.042325 | 0.101650 | 0.192438 | 0.153534 | 0.324907 | 1.374461 | 0.345007 | 0.345007 | 9.044199 |
| 30000 | 30016 | 0.235265 | 0.023626 | 0.023346 | 0.006355 | 0.016757 | 0.027327 | 0.033015 | 0.043962 | 0.220048 | 0.069967 | 0.069967 | 10.361405 |
| 60000 | 60000 | 0.170474 | 0.002997 | 0.005010 | 0.001725 | 0.003390 | 0.002089 | 0.002124 | 0.003631 | 0.028309 | 0.024417 | 0.024417 | 12.360060 |

Token sha256 `7d583ae13916c0f8463e06f53ba0ad6a0f650c0b849357d0ff020923eb04da0a`.

### seed 2 oracle

Bits/observation 0.684715. Bits 41082.904. Wall seconds 115.456. Replay steps 1875. Finite True.

Full cycles 11. Deepest room 8. First step 4120.

Time in room, rooms 0..8: 10351 6890 6812 6734 6656 6578 6500 6422 3057.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 38249.845 | 2.549990 |
| 1 | 15000 | 30000 | 15000 | 1750.543 | 0.116703 |
| 2 | 30000 | 45000 | 15000 | 874.085 | 0.058272 |
| 3 | 45000 | 60000 | 15000 | 208.431 | 0.013895 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.049227 | 0.161900 | 0.212471 | 0.092485 | 0.254802 | 0.147511 | 0.155245 | 0.130261 | 2.261692 | 0.385066 | 0.385066 | 10.786448 |
| 30000 | 30016 | 0.009412 | 0.015687 | 0.030972 | 0.010196 | 0.020288 | 0.019460 | 0.032690 | 0.021480 | 1.192912 | 0.150344 | 0.150344 | 12.626859 |
| 60000 | 60000 | 0.002462 | 0.003831 | 0.005994 | 0.002515 | 0.003443 | 0.003413 | 0.002842 | 0.004488 | 0.115831 | 0.016091 | 0.016091 | 13.590581 |

Token sha256 `0228c55aa05aa4bce3e79162cdfc9bc24ac26f03ce795f35c94bd1813673908b`.

### seed 2 oracle_deep

Bits/observation 0.705565. Bits 42333.893. Wall seconds 114.505. Replay steps 1875. Finite True.

Full cycles 0. Deepest room 8. First step 24.

Time in room, rooms 0..8: 4426 4378 3914 3782 3740 3698 3656 3614 28792.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 41140.595 | 2.742706 |
| 1 | 15000 | 30000 | 15000 | 951.250 | 0.063417 |
| 2 | 30000 | 45000 | 15000 | 167.601 | 0.011173 |
| 3 | 45000 | 60000 | 15000 | 74.447 | 0.004963 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.148102 | 0.293937 | 0.254904 | 0.096150 | 0.208061 | 0.120197 | 0.280655 | 0.631618 | 0.048027 | 0.231295 | 0.231295 | 7.554017 |
| 30000 | 30016 | 0.023366 | 0.028302 | 0.051670 | 0.011360 | 0.024873 | 0.031788 | 0.050724 | 0.016192 | 0.008724 | 0.027444 | 0.027444 | 8.060074 |
| 60000 | 60000 | 0.003668 | 0.003563 | 0.007243 | 0.002984 | 0.006138 | 0.004197 | 0.007737 | 0.002951 | 0.001780 | 0.004474 | 0.004474 | 8.822656 |

Token sha256 `6ac61ca5732bf01c5aa6e9b29df0892c475371770cd24eb9de322e67ac898541`.

## Kernels

| Kernel | Version | Status | Elapsed seconds | Source | Script wall seconds |
|---|---:|---|---:|---|---:|
| ezzzzzx/w21-s0 | 1 | COMPLETE | 341.894 | log_clock | 332.683 |
| ezzzzzx/w21-s1 | 1 | COMPLETE | 475.501 | log_clock | 458.890 |
| ezzzzzx/w21-s2 | 1 | COMPLETE | 480.466 | log_clock | 464.957 |

## Excerpt

Oracle, world seed 0, first 200 tokens, as (action, room, y).

(7, 0, 0)
(4, 0, 12)
(3, 0, 0)
(0, 0, 6)
(5, 0, 11)
(2, 0, 1)
(2, 0, 8)
(1, 0, 13)
(1, 0, 13)
(1, 0, 13)
(1, 0, 13)
(4, 0, 12)
(1, 0, 13)
(1, 0, 13)
(4, 0, 12)
(0, 0, 6)
(2, 0, 8)
(5, 0, 11)
(4, 0, 0)
(4, 0, 12)
(0, 0, 6)
(1, 0, 13)
(1, 0, 13)
(4, 0, 12)
(0, 0, 6)
(4, 0, 12)
(1, 0, 13)
(3, 0, 0)
(0, 0, 6)
(2, 0, 8)
(1, 0, 13)
(1, 0, 13)
(3, 0, 0)
(1, 0, 13)
(0, 0, 6)
(0, 0, 6)
(1, 0, 13)
(2, 0, 8)
(0, 0, 6)
(5, 0, 11)
(1, 0, 2)
(0, 0, 6)
(5, 0, 11)
(1, 0, 2)
(4, 0, 12)
(5, 0, 11)
(3, 0, 13)
(2, 0, 7)
(5, 0, 11)
(0, 0, 13)
(1, 0, 13)
(1, 0, 13)
(2, 0, 8)
(3, 0, 0)
(2, 0, 8)
(4, 0, 12)
(3, 0, 0)
(3, 0, 0)
(0, 0, 6)
(5, 0, 11)
(0, 0, 13)
(0, 0, 6)
(4, 0, 12)
(2, 0, 8)
(5, 0, 11)
(4, 0, 0)
(0, 0, 6)
(4, 0, 12)
(0, 0, 6)
(5, 0, 11)
(0, 0, 13)
(5, 0, 11)
(0, 0, 13)
(4, 0, 12)
(0, 0, 6)
(1, 0, 13)
(0, 0, 6)
(5, 0, 11)
(3, 0, 13)
(4, 0, 6)
(2, 0, 8)
(2, 0, 8)
(1, 0, 13)
(1, 0, 13)
(0, 0, 6)
(4, 0, 12)
(3, 0, 0)
(1, 0, 13)
(2, 0, 8)
(5, 0, 11)
(3, 0, 13)
(2, 0, 7)
(5, 0, 11)
(4, 0, 0)
(1, 0, 13)
(4, 0, 12)
(4, 0, 12)
(5, 0, 11)
(3, 0, 13)
(1, 0, 4)

## Protocol

- Rooms 0..8. Actions 0..5 manipulate, 6 is the TV, 7 is reset. Vocabulary 152.
- Observation token is 8 + 16·room + y, room after the action.
- Keys, lock tables, room-8 out and next are drawn from default_rng(world_seed), in that order. key_8 is drawn from default_rng(world_seed + 50000), three integers in 0..5.
- Room 8 keeps y = out[q, a] and q = nxt[q, a]. s8 advances like a lock room. When s8 reaches 3, y is kept and room, s, q and s8 are set to 0.
- TV noise is default_rng(10000 + world_seed). Reset returns room 0, s 0, q 0, s8 0, y 0. The TV does not change s8.
- rand8 is uniform on 0..7 and rand6 is uniform on 0..5, both from default_rng(5000 + world_seed).
- Oracle blocks are 512 steps. Block b targets room b mod 9, presses reset, plays the keys to that room, then stays with default_rng(6000 + world_seed). It never presses the TV. At s = 2 it excludes key[2]. Its room-8 stay is the W2 stay, so key_8 can complete.
- oracle_deep uses the same block size, reset, navigation and stay rng. Block b targets room 8 when b is even and (b // 2) mod 8 when b is odd. In room 8 at s8 = 2 it excludes key_8[2], so key_8 does not complete.
- The last block is 96 steps because 60000 is not a multiple of 512. The oracle target is room 0. oracle_deep block 117 is odd, so its target is 2.
- A full cycle is a manipulation in room 8 that returns the agent to room 0. Reset is not a cycle.
- GRU embedding 32, hidden 128, 1 layer, readout 152. 86680 parameters. Adam lr 0.001, clip 1.0. Learner seed 0. CPU, 4 threads.
- Chunks are 64 tokens. Score the chunk, then one Adam step on the mean NLL of its observation tokens, state detached. Replay seed 30000. After each chunk, one step on 8 windows of 64, starts uniform on even positions that fit, zero state, observation loss only.
- Checkpoints T/4, T/2 and T are 15000, 30000 and 60000. A probe runs at the first chunk boundary at or after the nominal step. Those boundaries are 15008, 30016 and 60000. The probe at T is exact.
- A probe episode is a new World21(world_seed), so the noise rng restarts. Room probes reset, play the keys, then 96 manipulations from default_rng(999 + room). Lock rooms exclude key[2] at s = 2. The room-8 probe excludes key_8[2] at s8 = 2 and does not leave room 8. The path is teacher-forced and unscored. TV probe is 96 TV presses from room 0. PROBE is the unweighted mean of the nine room means. PROBE_cap is the mean of min(probe_k, 4).
- Room-8 probe differences are counted against the W2-0 room-8 probe. prefix_differ is the reset-and-keys prefix. scored_differ is the 96 stay actions.
- W2-0 numbers are read from the stored W2 headroom run, seeds 0, 1 and 2, policies rand6 and oracle. W2-0 PROBE_cap is the mean of min(probe_k, 4) on that run.
- Kernels ezzzzzx/w21-s0, s1 and s2. Private, CPU, no internet. Code dataset ezzzzzx/pdp-code-w21.


## Verdict (orchestrator, 13 Nov 2026, UTC+8)

**Bar 1 PASS, bar 2 FAIL. The kill rule fires: W2-1 has no headroom for curiosity.**
- Bar 1: rand6 completes 25, 27 and 28 full cycles (bar ≥ 3).
- Bar 2: rand6 − min(oracle, oracle_deep) on PROBE_cap = +0.001, +0.010 and +0.020 (bar ≥ 0.5 on 2 of 3 seeds).
  rand6's PROBE_cap is 0.004–0.024 bits: it knows every room almost perfectly by the end of life.

**Mechanism: all of W2's headroom was the trap.** In W2-0 rand6 scored 0.50–1.57 on PROBE_cap. With a way back from
room 8, the same random walk revisits every room about 26 times and the learner, with replay, saturates the whole
world. W2's gap between rand6 and the oracle came from rand6 never returning, not from where it spent its time. Nine
rooms of 18 table entries plus a 6-state machine are too little to learn for a 60,000-step life. When everything
gets learned, allocation does not matter, and curiosity has nothing to win.

**What a curiosity world needs (the lesson of W1c, W2 and W2-1 together).** More learnable content than the life
can cover, so time is scarce. Sources that differ in how learnable they are: easy, hard but learnable, and
unlearnable noise. And an allocation oracle that knows the learnability, whose gain over a random walk sets the
headroom. That was H23c's setting, where curiosity did win; the W-worlds since have lacked the scarcity. The next
world is specified with a headroom check first.
