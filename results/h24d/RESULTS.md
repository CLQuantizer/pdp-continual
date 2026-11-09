# H24d

| Seeds | Ticks / seed | Q | K | Replay / arrival | Pool | Wall seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 6, 7, 8 | 10000 | 0.2 | 200 | 8 | 2 | 3215.587 |

Baseline source: `results/h24c/results.json` (reused).

Margin: `FF_score(true) - max(FF_score(other))`, before learning, without softmax. Low confidence: margin <= linear 20th percentile of the preceding up-to-500 arrival margins; current margin excluded; no low-confidence write with empty history. Replay begins at tick 9, with replacement from the route's store after any current write, as in H24c. Store payload: 785 floats/exemplar; margin history: at most 500 extra scalars for gate_s2.

## Mean accuracy

| Route | Lifetime % | Normal % | Episode 1 % | Episode 2 % | Episode 3–5 % | Episode 6+ % |
| --- | --- | --- | --- | --- | --- | --- |
| ff | 65.993 | 76.775 | 9.500 | 12.833 | 16.246 | 31.855 |
| exem | 87.327 | 86.319 | 12.833 | 100.000 | 100.000 | 100.000 |
| gate | 79.633 | 76.775 | 9.500 | 100.000 | 100.000 | 100.000 |
| ffr | 78.237 | 83.894 | 10.833 | 20.333 | 44.744 | 77.966 |
| gate_r | 85.350 | 83.894 | 10.833 | 100.000 | 100.000 | 100.000 |
| ffr_s | 78.850 | 82.944 | 10.167 | 36.167 | 58.351 | 80.531 |
| gate_s | 84.473 | 82.944 | 10.167 | 96.833 | 99.605 | 99.838 |
| ffr_s2 | 80.223 | 82.680 | 11.167 | 36.333 | 68.140 | 90.178 |
| gate_s2 | 84.270 | 82.680 | 11.167 | 96.667 | 99.437 | 99.835 |

## Seed 6

| Route | Lifetime % | Normal % | Episode 1 % | Episode 2 % | Episode 3–5 % | Episode 6+ % |
| --- | --- | --- | --- | --- | --- | --- |
| ff | 65.110 | 76.203 | 6.500 | 8.500 | 12.648 | 29.746 |
| exem | 87.040 | 86.101 | 9.500 | 100.000 | 100.000 | 100.000 |
| gate | 79.040 | 76.203 | 6.500 | 100.000 | 100.000 | 100.000 |
| ffr | 77.910 | 83.919 | 8.500 | 16.500 | 41.821 | 77.259 |
| gate_r | 85.270 | 83.919 | 8.500 | 100.000 | 100.000 | 100.000 |
| ffr_s | 78.170 | 82.299 | 7.000 | 36.500 | 58.010 | 79.594 |
| gate_s | 83.840 | 82.299 | 7.000 | 96.500 | 99.494 | 100.000 |
| ffr_s2 | 80.260 | 82.685 | 8.500 | 38.000 | 69.140 | 90.355 |
| gate_s2 | 84.140 | 82.685 | 8.500 | 96.000 | 99.325 | 99.797 |

## Seed 7

| Route | Lifetime % | Normal % | Episode 1 % | Episode 2 % | Episode 3–5 % | Episode 6+ % |
| --- | --- | --- | --- | --- | --- | --- |
| ff | 67.040 | 77.669 | 11.000 | 17.000 | 17.785 | 35.094 |
| exem | 87.590 | 86.531 | 14.500 | 100.000 | 100.000 | 100.000 |
| gate | 80.480 | 77.669 | 11.000 | 100.000 | 100.000 | 100.000 |
| ffr | 78.060 | 83.673 | 11.000 | 20.000 | 46.309 | 77.453 |
| gate_r | 85.250 | 83.673 | 11.000 | 100.000 | 100.000 | 100.000 |
| ffr_s | 78.910 | 83.019 | 11.500 | 35.000 | 57.383 | 81.226 |
| gate_s | 84.660 | 83.019 | 11.500 | 97.000 | 100.000 | 99.811 |
| ffr_s2 | 79.830 | 82.565 | 11.500 | 31.500 | 64.094 | 90.189 |
| gate_s2 | 84.320 | 82.565 | 11.500 | 98.500 | 99.664 | 99.906 |

## Seed 8

| Route | Lifetime % | Normal % | Episode 1 % | Episode 2 % | Episode 3–5 % | Episode 6+ % |
| --- | --- | --- | --- | --- | --- | --- |
| ff | 65.830 | 76.453 | 11.000 | 13.000 | 18.305 | 30.723 |
| exem | 87.350 | 86.327 | 14.500 | 100.000 | 100.000 | 100.000 |
| gate | 79.380 | 76.453 | 11.000 | 100.000 | 100.000 | 100.000 |
| ffr | 78.740 | 84.089 | 13.000 | 24.500 | 46.102 | 79.187 |
| gate_r | 85.530 | 84.089 | 13.000 | 100.000 | 100.000 | 100.000 |
| ffr_s | 79.470 | 83.515 | 12.000 | 37.000 | 59.661 | 80.773 |
| gate_s | 84.920 | 83.515 | 12.000 | 97.000 | 99.322 | 99.703 |
| ffr_s2 | 80.580 | 82.790 | 13.500 | 39.500 | 71.186 | 89.990 |
| gate_s2 | 84.350 | 82.790 | 13.500 | 95.500 | 99.322 | 99.802 |

## Gate firing

| Seed | Route | Lifetime % | Normal % | Repeat episodes % |
| --- | --- | --- | --- | --- |
| 6 | gate | 17.780 | 0.000 | 100.000 |
| 6 | gate_r | 17.780 | 0.000 | 100.000 |
| 6 | gate_s | 17.360 | 0.000 | 97.638 |
| 6 | gate_s2 | 17.190 | 0.000 | 96.682 |
| 7 | gate | 18.570 | 0.013 | 100.000 |
| 7 | gate_r | 18.570 | 0.013 | 100.000 |
| 7 | gate_s | 16.810 | 0.000 | 90.571 |
| 7 | gate_s2 | 16.610 | 0.000 | 89.494 |
| 8 | gate | 18.000 | 0.012 | 100.000 |
| 8 | gate_r | 18.000 | 0.012 | 100.000 |
| 8 | gate_s | 16.990 | 0.000 | 94.441 |
| 8 | gate_s2 | 16.940 | 0.000 | 94.163 |

## Store payload

| Seed | Route | Peak exemplars | Final exemplars | Peak floats | Final floats |
| --- | --- | --- | --- | --- | --- |
| 6 | exem | 10000 | 10000 | 7850000 | 7850000 |
| 6 | gate_r | 10000 | 10000 | 7850000 | 7850000 |
| 6 | gate_s | 1616 | 1616 | 1268560 | 1268560 |
| 6 | gate_s2 | 2320 | 2320 | 1821200 | 1821200 |
| 7 | exem | 10000 | 10000 | 7850000 | 7850000 |
| 7 | gate_r | 10000 | 10000 | 7850000 | 7850000 |
| 7 | gate_s | 1534 | 1534 | 1204190 | 1204190 |
| 7 | gate_s2 | 2329 | 2329 | 1828265 | 1828265 |
| 8 | exem | 10000 | 10000 | 7850000 | 7850000 |
| 8 | gate_r | 10000 | 10000 | 7850000 | 7850000 |
| 8 | gate_s | 1508 | 1508 | 1183780 | 1183780 |
| 8 | gate_s2 | 2279 | 2279 | 1789015 | 1789015 |

## Mean store payload

| Route | Peak exemplars | Final exemplars | Peak floats | Final floats |
| --- | --- | --- | --- | --- |
| gate_r | 10000.000 | 10000.000 | 7850000.000 | 7850000.000 |
| gate_s | 1552.667 | 1552.667 | 1218843.333 | 1218843.333 |
| gate_s2 | 2309.333 | 2309.333 | 1812826.667 | 1812826.667 |

## Writes

| Seed | Route | Total | Episodes | Normal | Arrivals written % | Episode writes % | Normal writes % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | gate_s | 1616 | 196 | 1420 | 16.160 | 12.129 | 87.871 |
| 6 | gate_s2 | 2320 | 615 | 1705 | 23.200 | 26.509 | 73.491 |
| 7 | gate_s | 1534 | 185 | 1349 | 15.340 | 12.060 | 87.940 |
| 7 | gate_s2 | 2329 | 649 | 1680 | 23.290 | 27.866 | 72.134 |
| 8 | gate_s | 1508 | 189 | 1319 | 15.080 | 12.533 | 87.467 |
| 8 | gate_s2 | 2279 | 593 | 1686 | 22.790 | 26.020 | 73.980 |

## Store samples

| Seed | Tick | exem / gate_r | gate_s | gate_s2 |
| --- | --- | --- | --- | --- |
| 6 | 400 | 400 | 218 | 204 |
| 6 | 1000 | 1000 | 413 | 418 |
| 6 | 2000 | 2000 | 636 | 717 |
| 6 | 3000 | 3000 | 812 | 946 |
| 6 | 4000 | 4000 | 976 | 1147 |
| 6 | 5000 | 5000 | 1099 | 1336 |
| 6 | 6000 | 6000 | 1198 | 1530 |
| 6 | 7000 | 7000 | 1328 | 1736 |
| 6 | 8000 | 8000 | 1435 | 1922 |
| 6 | 9000 | 9000 | 1512 | 2120 |
| 6 | 10000 | 10000 | 1616 | 2320 |
| 7 | 400 | 400 | 214 | 223 |
| 7 | 1000 | 1000 | 395 | 434 |
| 7 | 2000 | 2000 | 619 | 726 |
| 7 | 3000 | 3000 | 805 | 966 |
| 7 | 4000 | 4000 | 937 | 1180 |
| 7 | 5000 | 5000 | 1072 | 1371 |
| 7 | 6000 | 6000 | 1175 | 1560 |
| 7 | 7000 | 7000 | 1268 | 1748 |
| 7 | 8000 | 8000 | 1352 | 1939 |
| 7 | 9000 | 9000 | 1444 | 2131 |
| 7 | 10000 | 10000 | 1534 | 2329 |
| 8 | 400 | 400 | 206 | 208 |
| 8 | 1000 | 1000 | 400 | 418 |
| 8 | 2000 | 2000 | 622 | 698 |
| 8 | 3000 | 3000 | 794 | 923 |
| 8 | 4000 | 4000 | 956 | 1125 |
| 8 | 5000 | 5000 | 1069 | 1312 |
| 8 | 6000 | 6000 | 1158 | 1510 |
| 8 | 7000 | 7000 | 1283 | 1716 |
| 8 | 8000 | 8000 | 1361 | 1913 |
| 8 | 9000 | 9000 | 1430 | 2095 |
| 8 | 10000 | 10000 | 1508 | 2279 |

## Cortex normal accuracy difference from ffr (percentage points)

| Seed | ffr_s − ffr | ffr_s2 − ffr |
| --- | --- | --- |
| 6 | -1.621 | -1.234 |
| 7 | -0.655 | -1.108 |
| 8 | -0.575 | -1.300 |

| Seed | Runtime seconds | Episodes |
| --- | --- | --- |
| 6 | 1669.030 | 1978 |
| 7 | 1682.653 | 2056 |
| 8 | 1546.345 | 1999 |

## Verdict (orchestrator, 10 Nov 2026 UTC+8): PASS

1. **M3-MNIST PASS:** gate_s lifetime 84.47 ≥ 84.35 (gate_r − 1.0) with a final store of 1 553 exemplars (15.5% of
   gate_r's 10 000; bar ≤ 5 000). gate_s2 also passes the size bar (2 309) but misses lifetime by 0.08.
2. **PASS:** episodes at exposure 6+ 99.84%.
3. Report: replaying only surprises costs the cortex 0.95 points on normal digits (82.94 vs 83.89) but consolidates
   episodes faster (exposures 3–5: 58.4 vs 44.7; 6+: 80.5 vs 78.0).

Mechanism: the history-compressor rule (write only what the system failed to predict) keeps ~15% of arrivals, about
12% of them episodes and 88% the normal digits the cortex still gets wrong. Exposure-2 episodes are 96.8%, not 100,
because a few were answered right by chance on first sight (1 in 10) and so never written. Replay from a store of
surprises is a curriculum of the cortex's own errors.
