# W2 headroom

Fixed policies on the locked-rooms world. Bits are prequential, on observation tokens only.

## Headroom at T

PROBE is the mean of probe_k over rooms 0..8. Headroom uses the probe at step 60000.

| world_seed | PROBE rand8 | PROBE rand6 | PROBE oracle | rand8 − oracle | rand6 − oracle | rand8 − rand6 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.953637 | 1.782124 | 0.005676 | 9.947961 | 1.776449 | 8.171513 |
| 1 | 9.931172 | 0.501576 | 0.004577 | 9.926595 | 0.496999 | 9.429596 |
| 2 | 8.922539 | 1.772205 | 0.005486 | 8.917053 | 1.766720 | 7.150333 |
| mean |  |  |  | 9.597203 | 1.346723 | 8.250481 |
| std ddof=0 |  |  |  | 0.481018 | 0.600858 | 0.932179 |

## Lives

### seed 0 rand8

Bits/observation 0.615821. Bits 36949.273. Wall seconds 119.677. Replay steps 1875. Finite True.

Deepest room 2. First step 9815.

Time in room, rooms 0..8: 58952 1019 29 0 0 0 0 0 0.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 13180.613 | 0.878708 |
| 1 | 15000 | 30000 | 15000 | 8093.478 | 0.539565 |
| 2 | 30000 | 45000 | 15000 | 7777.731 | 0.518515 |
| 3 | 45000 | 60000 | 15000 | 7897.451 | 0.526497 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.016140 | 0.602992 | 17.865784 | 12.357339 | 12.352929 | 18.700069 | 18.511328 | 15.726879 | 18.434654 | 12.729791 | 4.065536 |
| 30000 | 30016 | 0.006673 | 0.275102 | 10.438972 | 12.750709 | 12.357718 | 13.002943 | 14.086062 | 12.341172 | 13.707920 | 9.885252 | 7.843722 |
| 60000 | 60000 | 0.002817 | 0.098665 | 4.876036 | 13.150644 | 13.168121 | 14.375557 | 16.598017 | 12.768269 | 14.544607 | 9.953637 | 5.099974 |

Token sha256 `6ce3ff7d4ef26e807cc5451ca4fce6dd66efa4c703322b6ab84f6e3f4bd86666`.

### seed 0 rand6

Bits/observation 0.291737. Bits 17504.198. Wall seconds 110.095. Replay steps 1875. Finite True.

Deepest room 8. First step 947.

Time in room, rooms 0..8: 51 23 75 253 16 422 70 37 59053.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 17405.631 | 1.160375 |
| 1 | 15000 | 30000 | 15000 | 66.262 | 0.004417 |
| 2 | 30000 | 45000 | 15000 | 21.545 | 0.001436 |
| 3 | 45000 | 60000 | 15000 | 10.760 | 0.000717 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 4.613347 | 6.191772 | 2.407755 | 0.774325 | 6.121901 | 0.750857 | 2.613077 | 4.052284 | 0.098214 | 3.069281 | 8.352505 |
| 30000 | 30016 | 2.706301 | 4.297259 | 1.453691 | 0.632568 | 5.683789 | 0.431421 | 2.285191 | 2.831873 | 0.080576 | 2.266963 | 8.283345 |
| 60000 | 60000 | 1.830558 | 3.488226 | 1.013654 | 0.344698 | 5.877820 | 0.282358 | 1.319156 | 1.817883 | 0.064767 | 1.782124 | 8.564331 |

Token sha256 `cde21864ddb459cfcab91da30356dbc9fe0156b345be49134c15991e6bfeb909`.

### seed 0 oracle

Bits/observation 0.685994. Bits 41159.645. Wall seconds 111.334. Replay steps 1875. Finite True.

Deepest room 8. First step 4120.

Time in room, rooms 0..8: 7064 6890 6812 6734 6656 6578 6500 6422 6344.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 39037.211 | 2.602481 |
| 1 | 15000 | 30000 | 15000 | 1765.467 | 0.117698 |
| 2 | 30000 | 45000 | 15000 | 260.743 | 0.017383 |
| 3 | 45000 | 60000 | 15000 | 96.224 | 0.006415 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.189124 | 0.052569 | 0.207872 | 0.112106 | 0.184750 | 0.183197 | 0.177202 | 0.119389 | 1.188889 | 0.268344 | 9.642345 |
| 30000 | 30016 | 0.024285 | 0.009661 | 0.028818 | 0.014539 | 0.026424 | 0.028460 | 0.024635 | 0.016089 | 0.163373 | 0.037365 | 10.670156 |
| 60000 | 60000 | 0.004716 | 0.001820 | 0.003948 | 0.002452 | 0.003795 | 0.004016 | 0.004253 | 0.002389 | 0.023695 | 0.005676 | 11.392893 |

Token sha256 `6f9f84ebab04e1eee84aecaa3de7454bfe5ff4224149123e6d81ecc732493a7e`.

### seed 1 rand8

Bits/observation 0.613875. Bits 36832.515. Wall seconds 112.036. Replay steps 1875. Finite True.

Deepest room 2. First step 27012.

Time in room, rooms 0..8: 59004 964 32 0 0 0 0 0 0.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 13337.635 | 0.889176 |
| 1 | 15000 | 30000 | 15000 | 7926.305 | 0.528420 |
| 2 | 30000 | 45000 | 15000 | 8036.009 | 0.535734 |
| 3 | 45000 | 60000 | 15000 | 7532.566 | 0.502171 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.013367 | 0.864650 | 15.705412 | 15.701598 | 17.072610 | 16.596728 | 14.672080 | 13.562784 | 17.304892 | 12.388236 | 4.119691 |
| 30000 | 30016 | 0.004276 | 0.372341 | 10.716378 | 13.703830 | 15.045978 | 14.183175 | 13.536373 | 12.914995 | 15.847739 | 10.702787 | 4.136225 |
| 60000 | 60000 | 0.001744 | 0.125100 | 4.803557 | 14.231876 | 14.203723 | 14.878694 | 14.386463 | 13.629028 | 13.120363 | 9.931172 | 4.067139 |

Token sha256 `9bc79f36d8aa7d73abf185d85532859fd78c67bb1b1da753639553c4ef020663`.

### seed 1 rand6

Bits/observation 0.394290. Bits 23657.413. Wall seconds 102.607. Replay steps 1875. Finite True.

Deepest room 8. First step 1476.

Time in room, rooms 0..8: 137 146 266 213 333 114 144 123 58524.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 23500.173 | 1.566678 |
| 1 | 15000 | 30000 | 15000 | 107.836 | 0.007189 |
| 2 | 30000 | 45000 | 15000 | 32.861 | 0.002191 |
| 3 | 45000 | 60000 | 15000 | 16.543 | 0.001103 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 1.550032 | 0.860266 | 1.002051 | 0.939085 | 0.956586 | 2.659934 | 1.742741 | 1.734712 | 0.092908 | 1.282035 | 6.588542 |
| 30000 | 30016 | 1.133189 | 0.918640 | 0.734559 | 0.526429 | 0.446679 | 1.329393 | 0.696225 | 1.023313 | 0.076436 | 0.764985 | 6.813859 |
| 60000 | 60000 | 0.878178 | 0.268771 | 0.518233 | 0.234607 | 0.239014 | 1.120909 | 0.594108 | 0.611554 | 0.048812 | 0.501576 | 7.187133 |

Token sha256 `db3106e67a2800493eec33a6fb6cc845e7e5062fd1aa45df7d0b1b47bab04638`.

### seed 1 oracle

Bits/observation 0.713221. Bits 42793.230. Wall seconds 104.644. Replay steps 1875. Finite True.

Deepest room 8. First step 4120.

Time in room, rooms 0..8: 7064 6890 6812 6734 6656 6578 6500 6422 6344.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 39537.596 | 2.635840 |
| 1 | 15000 | 30000 | 15000 | 2616.155 | 0.174410 |
| 2 | 30000 | 45000 | 15000 | 500.838 | 0.033389 |
| 3 | 45000 | 60000 | 15000 | 138.641 | 0.009243 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.055573 | 0.057747 | 0.126171 | 0.239934 | 0.222968 | 0.256817 | 0.244944 | 0.168495 | 1.618317 | 0.332330 | 7.419458 |
| 30000 | 30016 | 0.012878 | 0.012734 | 0.017114 | 0.049242 | 0.035664 | 0.034138 | 0.022540 | 0.018888 | 0.452488 | 0.072854 | 7.980874 |
| 60000 | 60000 | 0.002430 | 0.002813 | 0.002871 | 0.002984 | 0.004758 | 0.004217 | 0.002985 | 0.002800 | 0.015331 | 0.004577 | 9.236977 |

Token sha256 `049daca495571172ebe126af48b5e38c1f58e0cbb685d45fed4ecdd0a84c7327`.

### seed 2 rand8

Bits/observation 0.641083. Bits 38465.004. Wall seconds 122.163. Replay steps 1875. Finite True.

Deepest room 4. First step 29385.

Time in room, rooms 0..8: 58512 1450 18 16 4 0 0 0 0.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 14539.880 | 0.969325 |
| 1 | 15000 | 30000 | 15000 | 8539.635 | 0.569309 |
| 2 | 30000 | 45000 | 15000 | 7684.858 | 0.512324 |
| 3 | 45000 | 60000 | 15000 | 7700.630 | 0.513375 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.007424 | 0.687811 | 14.901891 | 13.725719 | 12.064781 | 12.563289 | 15.422305 | 11.597006 | 13.324688 | 10.477213 | 4.109923 |
| 30000 | 30016 | 0.003272 | 0.362487 | 10.772615 | 10.588192 | 11.066559 | 11.677477 | 13.543830 | 11.141281 | 11.632723 | 8.976493 | 4.035498 |
| 60000 | 60000 | 0.001464 | 0.053889 | 7.860346 | 9.310601 | 11.212041 | 12.557273 | 14.899076 | 11.965075 | 12.443083 | 8.922539 | 4.042067 |

Token sha256 `81bc8883f4d73e8bc96aeb839b33daaea54a6b80b76fcc9c341f3d12721fa0b2`.

### seed 2 rand6

Bits/observation 0.402176. Bits 24130.584. Wall seconds 114.310. Replay steps 1875. Finite True.

Deepest room 8. First step 1836.

Time in room, rooms 0..8: 93 803 6 229 125 226 177 177 58164.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 23965.651 | 1.597710 |
| 1 | 15000 | 30000 | 15000 | 115.481 | 0.007699 |
| 2 | 30000 | 45000 | 15000 | 33.496 | 0.002233 |
| 3 | 45000 | 60000 | 15000 | 15.955 | 0.001064 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 2.071536 | 0.411837 | 15.582589 | 0.627309 | 2.502896 | 1.084195 | 2.159241 | 0.668016 | 0.125491 | 2.803679 | 6.486342 |
| 30000 | 30016 | 1.488666 | 0.365370 | 13.761858 | 0.301136 | 0.902632 | 0.407025 | 0.523453 | 0.535848 | 0.091057 | 2.041894 | 6.931044 |
| 60000 | 60000 | 1.204933 | 0.130736 | 13.087835 | 0.139634 | 0.714913 | 0.175717 | 0.207794 | 0.220309 | 0.067979 | 1.772205 | 7.378171 |

Token sha256 `4dbbc58adf635c04dc2eb31d2fb6fd96a6094e02b1a04cedc8a1b70be246bbf0`.

### seed 2 oracle

Bits/observation 0.706659. Bits 42399.552. Wall seconds 113.133. Replay steps 1875. Finite True.

Deepest room 8. First step 4120.

Time in room, rooms 0..8: 7064 6890 6812 6734 6656 6578 6500 6422 6344.

| Quarter | Start | End | Observations | Bits | Bits/observation |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 15000 | 15000 | 39711.161 | 2.647411 |
| 1 | 15000 | 30000 | 15000 | 2198.281 | 0.146552 |
| 2 | 30000 | 45000 | 15000 | 370.963 | 0.024731 |
| 3 | 45000 | 60000 | 15000 | 119.147 | 0.007943 |

| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15000 | 15008 | 0.089399 | 0.165858 | 0.212048 | 0.093649 | 0.252687 | 0.151335 | 0.159421 | 0.141927 | 1.759287 | 0.336179 | 10.716571 |
| 30000 | 30016 | 0.016371 | 0.016900 | 0.032511 | 0.010477 | 0.021129 | 0.023114 | 0.035507 | 0.024271 | 0.341285 | 0.057952 | 12.796994 |
| 60000 | 60000 | 0.003078 | 0.003445 | 0.005218 | 0.002195 | 0.003248 | 0.003454 | 0.002517 | 0.003918 | 0.022300 | 0.005486 | 13.615126 |

Token sha256 `8b09a27b5bc63fcf431244c44e435c5b742f646ce567c72088efa30cf5c7504c`.

## Kernels

| Kernel | Version | Status | Elapsed seconds | Source | Script wall seconds |
|---|---:|---|---:|---|---:|
| ezzzzzx/w2-headroom-s0 | 1 | COMPLETE | 356.139 | log_clock | 341.115 |
| ezzzzzx/w2-headroom-s1 | 1 | COMPLETE | 333.300 | log_clock | 319.297 |
| ezzzzzx/w2-headroom-s2 | 1 | COMPLETE | 365.252 | log_clock | 349.615 |

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
- Keys, lock tables, room-8 out and next are drawn from default_rng(world_seed), in that order.
- TV noise is default_rng(10000 + world_seed). Reset returns room 0, s 0, q 0, y 0.
- rand8 is uniform on 0..7 and rand6 is uniform on 0..5, both from default_rng(5000 + world_seed).
- Oracle blocks are 512 steps. Block b targets room b mod 9, presses reset, plays the keys to that room, then stays with default_rng(6000 + world_seed). It never presses the TV. At s = 2 it excludes key[2].
- The last block is 96 steps because 60000 is not a multiple of 512. Its target is room 0.
- GRU embedding 32, hidden 128, 1 layer, readout 152. 86680 parameters. Adam lr 0.001, clip 1.0. Learner seed 0. CPU, 4 threads.
- Chunks are 64 tokens. Score the chunk, then one Adam step on the mean NLL of its observation tokens, state detached. Replay seed 30000. After each chunk, one step on 8 windows of 64, starts uniform on even positions that fit, zero state, observation loss only.
- Checkpoints T/4, T/2 and T are 15000, 30000 and 60000. A probe runs at the first chunk boundary at or after the nominal step. Those boundaries are 15008, 30016 and 60000. The probe at T is exact.
- A probe episode is a new World(world_seed), so the noise rng restarts. Room probes reset, play the keys, then 96 manipulations from default_rng(999 + room), excluding key[2] at s = 2. The path is teacher-forced and unscored. TV probe is 96 TV presses from room 0. PROBE is the unweighted mean of the nine room means.
- Kernels ezzzzzx/w2-headroom-s0, s1 and s2. Private, CPU, no internet. Code dataset ezzzzzx/pdp-code-w2.


## Verdict (orchestrator, 11 Nov 2026, UTC+8)

**PASS. W2 qualifies for the curiosity run.** Headroom (rand8 − oracle) is 9.95 / 9.93 / 8.92 bits per observation on
world seeds 0 / 1 / 2. The bar was ≥ 0.5 on every seed.

**The raw number is inflated, and the pass survives the correction.** rand8 scores 12–16 bits in rooms it never
reached. That is worse than a uniform guess over all 144 observation tokens (7.17 bits), so it measures
overconfidence, not just missing knowledge. This is the W1c noise artefact again. I recomputed headroom with each
room's probe capped. A cap of 4 bits is the cost of knowing the room but nothing about y.

| cap per room | rand8 − oracle | rand6 − oracle | rand8 − rand6 |
|---|---|---|---|
| none (raw) | 9.95 / 9.93 / 8.92 | 1.78 / 0.50 / 1.77 | 8.17 / 9.43 / 7.15 |
| 7.17 bits (uniform over 144) | 5.33 / 5.32 / 5.58 | 1.78 / 0.50 / 1.11 | 3.55 / 4.83 / 4.47 |
| 4 bits (uniform over y) | 3.12 / 3.12 / 3.11 | 1.57 / 0.50 / 0.76 | 1.55 / 2.62 / 2.36 |

Reading:
- **Consequences are real.** rand8 presses reset one step in eight, so it never gets past room 2 (seed 2: room 4,
  for 4 steps). Its whole-life prequential bits (0.62) are *lower* than the oracle's (0.69): it stays where the world
  is easy. Prequential bits reward staying shallow. The probe, which measures world knowledge, is the right metric for
  curiosity.
- **Most of the gap comes from not pressing reset, not from planning.** A wrong press only resets the lock, not the
  room, so rand6 (no reset, no TV) walks down to room 8 within about 2,000 steps and then stays there, because room 8
  has no exit. Even so, it knows most rooms: rand6 − oracle is 0.50–1.78 raw and 0.50–1.57 capped. Seed 1 is just
  under 0.5.
- **Oracle never watched the TV, so it mis-predicts it at 9–14 bits.** rand8 scores 4.0–5.1. Same lesson as W1c.

**Consequences for M4 on W2.**
1. The baseline the curious agent must beat is **rand6**, not rand8. Beating rand8 only shows it learned not to press
   reset.
2. Score the probe both raw and capped at 4 bits per room, and judge on the capped value.
3. Room 8 has no exit, so it absorbs any agent that never presses reset. A curious agent has to learn when a reset
   pays off: when room 8 is learned and earlier rooms are stale. That is a real planning test.

Path: rand8 lives in world_seed 0..2 never leave rooms 0–2 (deepest_room 2, 2, 4). Probe checkpoints are at steps
15,008 / 30,016 / 60,000.
