# W3 headroom

Fixed policies on the ring of H23c channels. Bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. Lower is better.

## W

| policy | L | seed 0 | seed 1 | seed 2 | mean | std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|
| rand_walk | 512 | 3.720991 | 3.754106 | 3.491642 | 3.655580 | 0.116707 |
| rand_dwell | 512 | 2.961286 | 2.955954 | 2.772922 | 2.896721 | 0.087566 |
| oracle7 | 512 | 2.857721 | 2.747134 | 2.489710 | 2.698189 | 0.154175 |
| oracle_hard | 512 | 2.922988 | 3.004731 | 3.448089 | 3.125269 | 0.230694 |
| oracle7 | 2048 | 1.822230 | 1.840934 | 1.837338 | 1.833501 | 0.008104 |

## Per-channel bits at the end of life

| policy | L | seed | G | C1 | C7 | E30 | E110 | M7 | M37 | W |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_walk | 512 | 0 | 4.755073 | 4.579129 | 4.874188 | 1.539738 | 2.650493 | 3.052337 | 4.595980 | 3.720991 |
| rand_walk | 512 | 1 | 4.889194 | 4.454333 | 4.821951 | 1.930109 | 3.630711 | 1.688044 | 4.864402 | 3.754106 |
| rand_walk | 512 | 2 | 4.893923 | 4.897977 | 5.084943 | 2.181829 | 1.556112 | 1.724785 | 4.101924 | 3.491642 |
| rand_dwell | 512 | 0 | 3.942376 | 4.657989 | 4.078244 | 1.763127 | 1.303924 | 0.879811 | 4.103534 | 2.961286 |
| rand_dwell | 512 | 1 | 4.245959 | 4.659435 | 4.516253 | 1.484376 | 1.310974 | 1.086972 | 3.387712 | 2.955954 |
| rand_dwell | 512 | 2 | 3.545506 | 4.812635 | 4.662405 | 1.403068 | 1.532483 | 1.165384 | 2.288970 | 2.772922 |
| oracle7 | 512 | 0 | 3.702364 | 4.581745 | 3.949855 | 1.379132 | 1.318576 | 0.575396 | 4.496981 | 2.857721 |
| oracle7 | 512 | 1 | 3.518984 | 4.278431 | 4.437633 | 1.391696 | 1.713515 | 0.486997 | 3.402682 | 2.747134 |
| oracle7 | 512 | 2 | 3.454821 | 3.592609 | 4.416350 | 1.581886 | 1.438806 | 0.898594 | 2.044908 | 2.489710 |
| oracle_hard | 512 | 0 | 3.435639 | 4.252679 | 3.841821 | 2.177899 | 1.691933 | 1.688602 | 3.372341 | 2.922988 |
| oracle_hard | 512 | 1 | 3.165325 | 4.327734 | 4.437544 | 1.761935 | 2.106406 | 1.872160 | 3.362016 | 3.004731 |
| oracle_hard | 512 | 2 | 4.159479 | 3.535873 | 3.764170 | 1.486287 | 1.705735 | 5.358266 | 4.126811 | 3.448089 |
| oracle7 | 2048 | 0 | 2.114921 | 3.494469 | 3.578807 | 1.229491 | 1.173675 | 0.362113 | 0.802131 | 1.822230 |
| oracle7 | 2048 | 1 | 2.508508 | 3.272237 | 3.627534 | 1.239694 | 1.196238 | 0.337043 | 0.705284 | 1.840934 |
| oracle7 | 2048 | 2 | 1.995887 | 3.719691 | 3.663240 | 1.245474 | 1.222424 | 0.415800 | 0.598852 | 1.837338 |

## Room shares and corridor share

A stay counts for the room it observes. A move counts for the corridor. Shares sum to 1.

| policy | L | seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | corridor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_walk | 512 | 0 | 0.0273 | 0.0352 | 0.0488 | 0.0508 | 0.0371 | 0.0684 | 0.0371 | 0.0215 | 0.0195 | 0.6543 |
| rand_walk | 512 | 1 | 0.0527 | 0.0352 | 0.0273 | 0.0117 | 0.0469 | 0.0410 | 0.0352 | 0.0176 | 0.0391 | 0.6934 |
| rand_walk | 512 | 2 | 0.0273 | 0.0195 | 0.0117 | 0.0293 | 0.0312 | 0.0332 | 0.0488 | 0.0625 | 0.0586 | 0.6777 |
| rand_dwell | 512 | 0 | 0.0469 | 0.1719 | 0.0781 | 0.0625 | 0.1094 | 0.0469 | 0.0625 | 0.1309 | 0.0625 | 0.2285 |
| rand_dwell | 512 | 1 | 0.0508 | 0.1250 | 0.0781 | 0.1250 | 0.1094 | 0.0781 | 0.0469 | 0.0781 | 0.0781 | 0.2305 |
| rand_dwell | 512 | 2 | 0.0312 | 0.0625 | 0.0469 | 0.1719 | 0.1094 | 0.0469 | 0.1406 | 0.0840 | 0.0625 | 0.2441 |
| oracle7 | 512 | 0 | 0.0312 | 0.2031 | 0.0469 | 0.0000 | 0.1406 | 0.1250 | 0.0000 | 0.1250 | 0.0938 | 0.2344 |
| oracle7 | 512 | 1 | 0.0000 | 0.1406 | 0.1719 | 0.0469 | 0.0000 | 0.0938 | 0.1055 | 0.0781 | 0.1250 | 0.2383 |
| oracle7 | 512 | 2 | 0.1094 | 0.0938 | 0.0000 | 0.1875 | 0.0625 | 0.0625 | 0.1562 | 0.0000 | 0.0781 | 0.2500 |
| oracle_hard | 512 | 0 | 0.0781 | 0.2344 | 0.2188 | 0.0000 | 0.1719 | 0.0312 | 0.0000 | 0.0469 | 0.0352 | 0.1836 |
| oracle_hard | 512 | 1 | 0.0000 | 0.2812 | 0.2344 | 0.0312 | 0.0000 | 0.0938 | 0.0312 | 0.0781 | 0.0469 | 0.2031 |
| oracle_hard | 512 | 2 | 0.2344 | 0.3125 | 0.0000 | 0.0918 | 0.0781 | 0.0000 | 0.0469 | 0.0000 | 0.0469 | 0.1895 |
| oracle7 | 2048 | 0 | 0.0898 | 0.1406 | 0.1055 | 0.0000 | 0.1284 | 0.1172 | 0.0000 | 0.1016 | 0.0977 | 0.2192 |
| oracle7 | 2048 | 1 | 0.0000 | 0.1328 | 0.1001 | 0.1250 | 0.0000 | 0.1094 | 0.1211 | 0.0977 | 0.0977 | 0.2163 |
| oracle7 | 2048 | 2 | 0.1133 | 0.1016 | 0.0000 | 0.1484 | 0.0977 | 0.1211 | 0.1172 | 0.0000 | 0.0820 | 0.2188 |

Room order is the channel in ring rooms 0..8.

| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | M37 | C7 | C1 | N | G | E30 | Z | E110 | M7 |
| 1 | N | C1 | G | E110 | Z | C7 | M7 | M37 | E30 |
| 2 | C1 | C7 | N | G | E30 | M7 | M37 | Z | E110 |

## Gaps

| seed | rand_walk − oracle7 | rand_walk − oracle_hard | rand_dwell − oracle7 | oracle7 L512 − L2048 |
|---:|---:|---:|---:|---:|
| 0 | 0.863270 | 0.798003 | 0.103565 | 1.035492 |
| 1 | 1.006972 | 0.749375 | 0.208820 | 0.906200 |
| 2 | 1.001931 | 0.043553 | 0.283211 | 0.652372 |
| mean | 0.957391 | 0.530310 | 0.198532 | 0.864688 |
| std ddof=0 | 0.066586 | 0.344761 | 0.073700 | 0.159138 |

## Checkpoints

W after L/4, L/2, 3L/4 and L decisions.

| policy | L | seed | t1 | W1 | t2 | W2 | t3 | W3 | t4 | W4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_walk | 512 | 0 | 128 | 4.731883 | 256 | 4.171623 | 384 | 3.993615 | 512 | 3.720991 |
| rand_walk | 512 | 1 | 128 | 4.873458 | 256 | 4.356414 | 384 | 4.077162 | 512 | 3.754106 |
| rand_walk | 512 | 2 | 128 | 4.672226 | 256 | 4.205738 | 384 | 3.852968 | 512 | 3.491642 |
| rand_dwell | 512 | 0 | 128 | 4.284675 | 256 | 3.760255 | 384 | 3.178402 | 512 | 2.961286 |
| rand_dwell | 512 | 1 | 128 | 4.171662 | 256 | 3.385287 | 384 | 3.033750 | 512 | 2.955954 |
| rand_dwell | 512 | 2 | 128 | 3.976487 | 256 | 3.395617 | 384 | 3.071982 | 512 | 2.772922 |
| oracle7 | 512 | 0 | 128 | 4.284675 | 256 | 3.404562 | 384 | 3.027411 | 512 | 2.857721 |
| oracle7 | 512 | 1 | 128 | 4.205348 | 256 | 3.629309 | 384 | 2.957164 | 512 | 2.747134 |
| oracle7 | 512 | 2 | 128 | 3.976487 | 256 | 3.234958 | 384 | 2.787719 | 512 | 2.489710 |
| oracle_hard | 512 | 0 | 128 | 4.583621 | 256 | 3.698463 | 384 | 3.258831 | 512 | 2.922988 |
| oracle_hard | 512 | 1 | 128 | 4.818397 | 256 | 3.667600 | 384 | 3.513423 | 512 | 3.004731 |
| oracle_hard | 512 | 2 | 128 | 4.435684 | 256 | 3.850901 | 384 | 3.691055 | 512 | 3.448089 |
| oracle7 | 2048 | 0 | 512 | 2.857721 | 1024 | 2.174321 | 1536 | 1.889773 | 2048 | 1.822230 |
| oracle7 | 2048 | 1 | 512 | 2.747134 | 1024 | 2.273066 | 1536 | 2.078712 | 2048 | 1.840934 |
| oracle7 | 2048 | 2 | 512 | 2.489710 | 1024 | 2.110028 | 1536 | 1.936323 | 2048 | 1.837338 |

## H23c reference

H23c scored seeds 6–11, 512 glances, no travel. Published table in results/h23c/RESULTS.md.

| cond | W mean published | std published | W mean from the H23b baseline file | std ddof=0 |
|---|---:|---:|---:|---:|
| rand | 2.637 | 0.052 | 2.637333 | 0.051569 |
| rand7 | 2.481 | 0.046 | 2.481015 | 0.046022 |

oracle_hard draws its target with probability proportional to these rand7 held-out bits:

| channel | bits |
|---|---:|
| G | 3.693095 |
| C1 | 4.094935 |
| C7 | 4.254920 |
| E30 | 1.351967 |
| E110 | 1.291723 |
| M7 | 0.473991 |
| M37 | 2.206474 |

Source: mean held-out bits/symbol of rand7 on seeds 6-11 in results/h23b/score.json; H23c RESULTS.md states those runs are its baselines. results/h23c/score.json has prog only.

## Prefix check

oracle7 at L=512 and the t=512 checkpoint of oracle7 at L=2048 see the same decisions.

| seed | abs W |
|---:|---:|
| 0 | 0.000e+00 |
| 1 | 0.000e+00 |
| 2 | 0.000e+00 |

## Walls and kernel

Kernel `ezzzzzx/w3-headroom`. Script wall 25.843 s.

```
ezzzzzx/w3-headroom version 1
status KernelWorkerStatus.COMPLETE
private, CPU, enable_gpu false, enable_internet false
dataset ezzzzzx/pdp-code-w3
script wall_s 25.843
kernel log: results.json written at 27.4 s; nbconvert finished at 33.3 s
status poll UTC: 03:57:31 RUNNING, 03:57:52 RUNNING, 03:58:15 COMPLETE
```

| policy | L | seed | wall seconds |
|---|---:|---:|---:|
| rand_walk | 512 | 0 | 5.405 |
| rand_walk | 512 | 1 | 4.855 |
| rand_walk | 512 | 2 | 4.913 |
| rand_dwell | 512 | 0 | 5.314 |
| rand_dwell | 512 | 1 | 4.942 |
| rand_dwell | 512 | 2 | 4.907 |
| oracle7 | 512 | 0 | 5.254 |
| oracle7 | 512 | 1 | 5.008 |
| oracle7 | 512 | 2 | 4.887 |
| oracle_hard | 512 | 0 | 5.285 |
| oracle_hard | 512 | 1 | 5.055 |
| oracle_hard | 512 | 2 | 4.569 |
| oracle7 | 2048 | 0 | 12.166 |
| oracle7 | 2048 | 1 | 11.891 |
| oracle7 | 2048 | 2 | 5.698 |

## Smoke test

```
  64 rand s6 W 4.372 | G 5.23 C1 4.71 C7 5.16 E30 3.95 E110 2.99 M7 3.62 M37 4.94 | N 0.09 Z 0.11
equivalence teleport W 4.372328787590
equivalence curio.live W 4.372328787590 abs 0.000e+00
equivalence curio.py W 4.372328787590 abs 0.000e+00
equivalence per-channel abs G 0.000e+00 C1 0.000e+00 C7 0.000e+00 E30 0.000e+00 E110 0.000e+00 M7 0.000e+00 M37 0.000e+00
ring orders
  seed 0 perm [6, 2, 1, 7, 0, 3, 8, 4, 5] rooms ['M37', 'C7', 'C1', 'N', 'G', 'E30', 'Z', 'E110', 'M7']
  seed 1 perm [7, 1, 0, 4, 8, 2, 5, 6, 3] rooms ['N', 'C1', 'G', 'E110', 'Z', 'C7', 'M7', 'M37', 'E30']
  seed 2 perm [1, 2, 7, 0, 3, 5, 6, 8, 4] rooms ['C1', 'C7', 'N', 'G', 'E30', 'M7', 'M37', 'Z', 'E110']
rand_walk trace seed 0, 20 decisions (action, room_after, channel)
  (stay, 0, M37)
  (left, 8, corridor)
  (right, 0, corridor)
  (left, 8, corridor)
  (right, 0, corridor)
  (left, 8, corridor)
  (stay, 8, M7)
  (stay, 8, M7)
  (left, 7, corridor)
  (left, 6, corridor)
  (stay, 6, Z)
  (stay, 6, Z)
  (left, 5, corridor)
  (stay, 5, E30)
  (stay, 5, E30)
  (right, 6, corridor)
  (stay, 6, Z)
  (left, 5, corridor)
  (stay, 5, E30)
  (right, 6, corridor)
L=64 seed 0, four policies
  rand_walk W 5.086606 finite True wall 3.92s learn 0.40s eval 3.52s corridor 0.672
  rand_dwell W 4.490496 finite True wall 3.94s learn 0.42s eval 3.52s corridor 0.250
  oracle7 W 4.490496 finite True wall 4.04s learn 0.44s eval 3.60s corridor 0.250
  oracle_hard W 5.171749 finite True wall 3.99s learn 0.42s eval 3.58s corridor 0.125
estimate per decision 0.0069s per eval 0.8989s serial 139s pool4 wall 35s (0.6 min)
estimate is within 90 min; one kernel
smoke json /tmp/w3_smoke.json
```

## Protocol

- 9 rooms on a ring. Channel order is `default_rng(world_seed + 70000).permutation(9)` into `curio.CH`.
- Actions stay, left, right. A move shows 64 space symbols labelled corridor and does not observe the rooms it passes.
- Life length is the number of decisions. One decision is one glance. The agent starts in room 0.
- Compressor is `curio.SGDCompressor` with lr 0.004. Corridor uses the same update with channel index 9.
- Policy rng is `default_rng(1000 + world_seed)`, also used for N-channel noise. Each policy builds its own rng.
- rand_walk is uniform over the three actions. rand_dwell picks a room uniformly from 9, walks the shortest path, stays 8 decisions, and repeats.
- oracle7 is rand_dwell over the 7 learnable rooms. oracle_hard draws that target with probability proportional to the rand7 bits above.
- Shortest path is unique because 9 is odd.
- W is the mean over the 7 learnable channels of bits/symbol on EVAL=2048 fresh symbols. The world is copied before the continuation. Channels are scored in `curio.LEARN` order.
- Checkpoints are L/4, L/2, 3L/4 and L. The compressor keeps learning after a checkpoint.
- Seeds 0, 1 and 2. L=512 for every policy. oracle7 also runs at L=2048.
- PYTHONHASHSEED=0. CPU. Pool of 4.

## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (headroom over a random walk, ≥ 0.15 on all 3 seeds): PASS.** rand_walk − best oracle is 0.86, 1.01 and 1.00.
**Bar 2 (time stays scarce, oracle7 L = 512 − L = 2,048 ≥ 0.20 on all seeds): PASS.** The differences are 1.04, 0.91 and
0.65.

**What the headroom is made of.**
- Most of the gap to rand_walk is travel: two thirds of its decisions are moves, and moves show only the corridor.
  Dwelling alone (rand_dwell) recovers 0.76 of it.
- The part that needs knowing what is learnable is rand_dwell − oracle7: 0.10, 0.21 and 0.28 (mean 0.20). That is
  slightly more than H23c's rand − rand7 (0.156), so travel did not erase it.
- **oracle_hard loses to oracle7** (3.13 vs 2.70, and nearly as badly as rand_walk on seed 2). It spends its time in
  proportion to how hard a source is, so it goes mostly to C1 and C7. This compressor stays near 4 bits/symbol on
  those, and makes no progress. Hard is not the same as learnable. Error-seeking (1990) curiosity fails for the same
  reason, and that is why the reward has to be progress.

So W3 is the first embodied world here with real headroom for curiosity. The next run puts the H23c progress
controller in it, with its reward as a progress *rate* (progress per decision, travel included). It is scored on
held-out world seeds against rand_dwell, oracle7 and an error-seeking controller.
