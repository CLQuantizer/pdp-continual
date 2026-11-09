# H21d measurements

Started: 2026-11-09 21:34:58 UTC+8. Updated: 2026-11-09 22:34:22 UTC+8.

Slow world; N = 131072; seeds = [6, 7, 8]; CPU threads = 2; population std (ddof=0).

| Learner / floor | Mean ± std bits/symbol | Seed 6 | Seed 7 | Seed 8 |
|---|---:|---:|---:|---:|
| gru_rb8x2 | 0.812894 ± 0.169738 | 0.720937 | 1.050900 | 0.666846 |
| gru_rb8x2+match | 0.772086 ± 0.157386 | 0.687302 | 0.992704 | 0.636254 |
| gru_rb8x4 | NA | NA | NA | NA |
| gru_rb8x4+match | NA | NA | NA | NA |
| gru_rb16x2 | NA | NA | NA | NA |
| gru_rb16x2+match | NA | NA | NA | NA |
| gru512_rb8x2 | 0.992951 ± 0.200443 | 0.835607 | 1.275825 | 0.867420 |
| gru512_rb8x2+match | 0.910381 ± 0.187050 | 0.757513 | 1.173778 | 0.799853 |
| kt_mixture | 1.735062 ± 0.376059 | 2.188957 | 1.268084 | 1.748144 |
| lzma_9e | 0.743896 ± 0.064964 | 0.758057 | 0.815430 | 0.658203 |

Per-process bits/symbol; arithmetic mean over seeds with observations.

| Learner | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru_rb8x2 | 1.050321 | 0.985559 | 0.717813 | 1.269215 | 1.169159 | 0.377766 | 0.462741 |
| gru_rb8x2+match | 0.997326 | 0.965341 | 0.709717 | 1.218045 | 1.141862 | 0.320175 | 0.327618 |
| gru_rb8x4 | NA | NA | NA | NA | NA | NA | NA |
| gru_rb8x4+match | NA | NA | NA | NA | NA | NA | NA |
| gru_rb16x2 | NA | NA | NA | NA | NA | NA | NA |
| gru_rb16x2+match | NA | NA | NA | NA | NA | NA | NA |
| gru512_rb8x2 | 1.354499 | 1.422716 | 0.733903 | 1.402192 | 1.322637 | 0.547014 | 0.579266 |
| gru512_rb8x2+match | 1.201567 | 1.332789 | 0.714878 | 1.333584 | 1.281996 | 0.396061 | 0.400866 |
| kt_mixture | 1.268344 | 3.168962 | 3.484201 | 1.189645 | 1.165874 | 0.511597 | 0.624158 |
| lzma_9e | NA | NA | NA | NA | NA | NA | NA |

| Seed | Learner | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | kt_mixture | 1.269379 | 3.253241 | 3.502203 | 1.201425 | 1.157696 | NA | 0.621232 |
| 6 | lzma_9e | NA | NA | NA | NA | NA | NA | NA |
| 7 | kt_mixture | 1.226282 | 3.082604 | NA | 1.177866 | 1.150181 | 0.511461 | 0.668232 |
| 7 | lzma_9e | NA | NA | NA | NA | NA | NA | NA |
| 8 | kt_mixture | 1.309371 | 3.171041 | 3.466199 | NA | 1.189745 | 0.511732 | 0.583011 |
| 8 | lzma_9e | NA | NA | NA | NA | NA | NA | NA |
| 6 | gru_rb8x2 | 1.254499 | 0.320983 | 0.676462 | 1.429011 | 1.091043 | NA | 0.353531 |
| 6 | gru_rb8x2+match | 1.168044 | 0.314251 | 0.660054 | 1.325919 | 1.091764 | NA | 0.309880 |
| 7 | gru_rb8x2 | 0.992089 | 1.998193 | NA | 1.109419 | 1.057152 | 0.379583 | 0.704197 |
| 7 | gru_rb8x2+match | 0.956783 | 1.953340 | NA | 1.110171 | 1.057856 | 0.335279 | 0.360174 |
| 8 | gru_rb8x2 | 0.904374 | 0.637503 | 0.759163 | NA | 1.359283 | 0.375948 | 0.330495 |
| 8 | gru_rb8x2+match | 0.867151 | 0.628432 | 0.759379 | NA | 1.275966 | 0.305071 | 0.312799 |
| 6 | gru512_rb8x2 | 1.917081 | 0.325238 | 0.610560 | 1.639422 | 1.187702 | NA | 0.354379 |
| 6 | gru512_rb8x2+match | 1.539290 | 0.311874 | 0.576364 | 1.501456 | 1.188422 | NA | 0.318952 |
| 7 | gru512_rb8x2 | 1.240277 | 2.828226 | NA | 1.164961 | 1.136764 | 0.511580 | 1.023064 |
| 7 | gru512_rb8x2+match | 1.184549 | 2.645925 | NA | 1.165711 | 1.134236 | 0.420402 | 0.539455 |
| 8 | gru512_rb8x2 | 0.906140 | 1.114686 | 0.857246 | NA | 1.643446 | 0.582447 | 0.360355 |
| 8 | gru512_rb8x2+match | 0.880862 | 1.040568 | 0.853392 | NA | 1.523331 | 0.371719 | 0.344190 |

| Seed | Learner | Wall seconds |
|---:|---|---:|
| 6 | kt_mixture | 20.467 |
| 6 | lzma_9e | 0.177 |
| 7 | kt_mixture | 15.253 |
| 7 | lzma_9e | 0.229 |
| 8 | kt_mixture | 20.823 |
| 8 | lzma_9e | 0.192 |
| 6 | gru_rb8x2 | 663.097 |
| 6 | gru_rb8x2+match | 667.697 |
| 7 | gru_rb8x2 | 668.958 |
| 7 | gru_rb8x2+match | 673.916 |
| 8 | gru_rb8x2 | 668.287 |
| 8 | gru_rb8x2+match | 673.434 |
| 6 | gru512_rb8x2 | 1219.632 |
| 6 | gru512_rb8x2+match | 1224.567 |
| 7 | gru512_rb8x2 | 1156.635 |
| 7 | gru512_rb8x2+match | 1162.471 |
| 8 | gru512_rb8x2 | 1160.939 |
| 8 | gru512_rb8x2+match | 1167.689 |

Reference scores and timings copied from H21c; no reference reruns. Match probabilities and timings reused from H21b.
Embedding 32, chunk 64, Adam lr 0.003, clip 1.0; H21b shifted inputs and detached carried live state.
gru_rb8x4: hidden 256, four replay steps, batch 8. gru_rb16x2: hidden 256, two replay steps, batch 16. gru512_rb8x2: hidden 512, two replay steps, batch 8.
Replay follows each scored chunk and its live Adam step. Uniform starts 0 through scored_prefix_length−64 inclusive, with replacement; zero replay hidden state; preceding-symbol input (zero embedding at global position 0). Live state is preserved during replay. Batched loss is the mean over all batch × 64 targets.
Mixtures reuse H21b initial weights (0.5, 0.5) and fixed share α = 0.001; wall seconds include cortex, reused match timing, and mixture update.
Scores are prequential observation bits; retained history is unpriced, as specified in H21b. lzma has no per-process attribution.
Full predictive distributions and code lengths are saved in learner-specific NPZ files.

Smoke projection (8192 symbols, seed 6; ×16 symbols ×3 seeds).

| Learner | Smoke wall seconds | Projected full wall seconds |
|---|---:|---:|
| gru512_rb8x2 | 77.912 | 3739.791 |
| gru_rb16x2 | 49.057 | 2354.754 |
| gru_rb8x4 | 72.126 | 3462.054 |

Projected retained cortex wall seconds: 3739.791.

Runtime note: gru_rb8x4 and gru_rb8x4+match omitted: smoke projection exceeds 4676.377-second planning allowance within 90-minute budget; prescribed drop order gru_rb8x4, then gru_rb16x2.
Runtime note: gru_rb16x2 and gru_rb16x2+match omitted: smoke projection exceeds 4676.377-second planning allowance within 90-minute budget; prescribed drop order gru_rb8x4, then gru_rb16x2.

Total full-run wall seconds: 3563.351.
Smoke plus full-run wall seconds: 3767.377.

## Verdict (orchestrator, 9 Nov 2026 UTC+8)

Only `gru512_rb8x2` ran. The spec's drop order removed both extra-replay arms first, which was the orchestrator's
mistake: the arm most likely to be slow should have been dropped first.

1. **M1 PASS (slow): FAIL.** gru512_rb8x2+match 0.910 (0.758 / 1.174 / 0.800) vs lzma 0.744.
2. **Lever: FAIL.** +0.138 *worse* than gru_rb8x2+match (0.772), on every seed.
3. CA: also worse (E30 1.33, E110 1.28).

Mechanism: online, at a fixed number of updates, a wider GRU learns more slowly per update, so capacity is not the
bottleneck. The CA rule needs exact, position-aligned access to the symbol 32 back, which is architectural. Next:
**H21e**, a small causal local-attention transformer (window 64) under the same batched replay and match mixture.
