I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is H24g: a utility-based store under a memory budget. I'll read the H24f results and the fine-store script before writing the new experiment.H24f kept surprising symbols and overshot the memory bar. H24g will keep the same fine-grained writes, credit each window by the bits it saves, and evict down to 0.15× full5’s peak memory.Smoke is inside the budget: replay-credit forwards add 12% wall time, and the next-chunk bit check matches to 1e-14. Production runs next, one seed at a time.H24g is measured. Both conditions ran on the slow world, N = 131072, seeds 6, 7 and 8. Nothing was committed.

The memory budget is B/N = 0.883848, which is 0.15 times full5's peak M. Retained memory stayed at or under B on every chunk. Replay credit stayed in: its extra wall fraction was 0.117 on the smoke run and 0.126, 0.125, 0.124 on the three `util` seeds, all below 0.30. The next-chunk bit check matched the later live GRU bits within 1e-12.

| Condition | P mean ± std | Peak M/N | F value bits/digit, ≥1 prior (mean / pooled) | Wall |
|---|---:|---:|---:|---:|
| full5 (carried) | 0.974201 ± 0.257310 | 5.892317 | 0.893400 / 0.751551 | historical |
| fine (carried) | 1.026776 ± 0.270682 | 2.413501 | 0.748335 / 0.742775 | historical |
| util | 1.007579 ± 0.268855 | 0.883838 | 0.629794 / 0.561753 | 2258 s |
| util_m | 0.997706 ± 0.259917 | 0.883846 | 0.640488 / 0.567852 | 2065 s |

Per seed, P and peak M/N:

| Condition | Seed 6 | Seed 7 | Seed 8 |
|---|---:|---:|---:|
| util P | 0.663219 | 1.319355 | 1.040163 |
| util peak M/N | 0.883822 | 0.883847 | 0.883845 |
| util final windows | 393 | 60 | 58 |
| util_m P | 0.660552 | 1.293094 | 1.039472 |
| util_m peak M/N | 0.883847 | 0.883846 | 0.883845 |
| util_m final windows | 443 | 80 | 54 |

F value digits at ≥1 prior appearance are 0.809395 and 0.450192 for `util` seeds 7 and 8, and 0.832218 and 0.448758 for `util_m`. Seed 6 has no F observations.

Evicted-window fractions, weighted by the true process labels inside each window:

| Condition | Seed | Largest shares |
|---|---:|---|
| util | 6 | E30 0.530, M7 0.241, C7 0.147, C1 0.082 |
| util | 7 | E110 0.332, G 0.236, M7 0.186, C1 0.127, C7 0.119 |
| util | 8 | E110 0.336, C7 0.278, M7 0.242, C1 0.129, F 0.016 |
| util_m | 6 | M37 0.345, E30 0.283, C7 0.198, C1 0.174 |
| util_m | 7 | E110 0.341, G 0.244, M7 0.158, C7 0.140, C1 0.117 |
| util_m | 8 | E110 0.414, C7 0.226, M7 0.224, C1 0.118, F 0.019 |

A zero in that table includes processes that never appear in the seed. Seed 6 has no G, E110, or F. Seed 7 has no E30 or M37. Seed 8 has no G or E30.

New-run wall time was 4323 s. gru, full, full5, fine, and fine3 are carried from H24f. The write-up is `results/h24g/RESULTS.md`.
