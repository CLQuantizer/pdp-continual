Created [h24b_world.py](/root/pdp/h24b_world.py) and [RESULTS.md](/root/pdp/results/h24b_floors/RESULTS.md), including all per-process tables. No commits.

N = 524,288; seeds 6–8. Mean ± population std, bits/symbol:

| World | KT mixture | lzma -9e |
|---|---:|---:|
| slow | 1.8487 ± 0.0230 | 0.7396 ± 0.0728 |
| fast | 1.6934 ± 0.0232 | 0.9322 ± 0.0229 |

F value digits, pooled across lives:

| Prior appearances | KT bits/digit | Exact recall, unpriced |
|---|---:|---:|
| 0 | 4.7252 | 3.3219 |
| 1 | 3.8990 | 0 |
| 2–4 | 3.0643 | 0 |
| 5+ | 1.6657 | 0 |

Smoke passed: **22.79 s**. Full runs: **396.54 s**. Total: **419.33 s**.