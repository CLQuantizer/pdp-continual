# H24e measurements

Started: 2026-11-10 04:37:20 UTC+8. Updated: 2026-11-10 05:54:19 UTC+8.

Slow world; N = 131072; seeds = [6, 7, 8]; alphabet = 42; CPU threads = 2.

| Condition | Seed | P | Peak S bits | Peak S/N | T | Chunks written | Fraction written | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | 6 | 1.339368 | 0.000 | 0.000000 | 1.339368 | 0 | 0.000000 | 193.633 |
| gru | 7 | 2.093612 | 0.000 | 0.000000 | 2.093612 | 0 | 0.000000 | 190.770 |
| gru | 8 | 1.760025 | 0.000 | 0.000000 | 1.760025 | 0 | 0.000000 | 183.070 |
| full | 6 | 0.622545 | 772317.829 | 5.892317 | 6.514863 | 2048 | 1.000000 | 684.836 |
| full | 7 | 1.272562 | 772317.829 | 5.892317 | 7.164880 | 2048 | 1.000000 | 674.432 |
| full | 8 | 1.079035 | 772317.829 | 5.892317 | 6.971353 | 2048 | 1.000000 | 659.480 |
| surp | 6 | 0.597542 | 270386.662 | 2.062887 | 2.660429 | 717 | 0.350098 | 674.914 |
| surp | 7 | 1.328256 | 236069.805 | 1.801070 | 3.129325 | 626 | 0.305664 | 675.740 |
| surp | 8 | 1.690978 | 116903.578 | 0.891904 | 2.582881 | 310 | 0.151367 | 680.083 |

| Condition | P mean ± population std | Peak S/N mean ± std | T mean ± std | Fraction written mean ± std |
|---|---:|---:|---:|---:|
| gru | 1.731002 ± 0.308602 | 0.000000 ± 0.000000 | 1.731002 ± 0.308602 | 0.000000 ± 0.000000 |
| full | 0.991381 ± 0.272510 | 5.892317 ± 0.000000 | 6.883698 ± 0.272510 | 1.000000 ± 0.000000 |
| surp | 1.205592 ± 0.454742 | 1.585287 ± 0.501811 | 2.790878 ± 0.241403 | 0.269043 ± 0.085164 |
| surp90 | NA | NA | NA | NA |

| Condition | Seed | G | C1 | C7 | E30 | E110 | M7 | M37 | F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | 6 | NA | 1.570243 | 2.405219 | 1.238582 | NA | 0.654884 | 0.466952 | NA |
| gru | 7 | 1.751203 | 3.395918 | 3.144010 | NA | 1.379465 | 0.585471 | NA | 3.733900 |
| gru | 8 | NA | 0.852376 | 2.023701 | NA | 1.173049 | 0.445717 | NA | 3.194302 |
| full | 6 | NA | 0.366176 | 0.733541 | 1.243366 | NA | 0.345031 | 0.316530 | NA |
| full | 7 | 0.930272 | 1.416133 | 1.002985 | NA | 1.043214 | 0.353308 | NA | 3.348156 |
| full | 8 | NA | 0.674508 | 0.841705 | NA | 1.195194 | 0.331688 | NA | 1.776144 |
| surp | 6 | NA | 0.303475 | 0.757300 | 1.202825 | NA | 0.370806 | 0.336157 | NA |
| surp | 7 | 0.904001 | 1.444971 | 1.327435 | NA | 1.075650 | 0.412390 | NA | 3.310300 |
| surp | 8 | NA | 2.571098 | 1.550335 | NA | 1.235523 | 0.345864 | NA | 2.519572 |

| Condition | Seed | Prior key appearances | Value digits | Bits | Bits/digit |
|---|---:|---|---:|---:|---:|
| gru | 6 | 0 | 0 | 0.000000 | NA |
| gru | 6 | 1 | 0 | 0.000000 | NA |
| gru | 6 | 2–4 | 0 | 0.000000 | NA |
| gru | 6 | 5+ | 0 | 0.000000 | NA |
| gru | 7 | 0 | 1188 | 5358.192638 | 4.510263 |
| gru | 7 | 1 | 1184 | 4406.571892 | 3.721767 |
| gru | 7 | 2–4 | 2744 | 9502.616450 | 3.463053 |
| gru | 7 | 5+ | 1436 | 4665.245333 | 3.248778 |
| gru | 8 | 0 | 1200 | 4625.701945 | 3.854752 |
| gru | 8 | 1 | 1200 | 4226.804125 | 3.522337 |
| gru | 8 | 2–4 | 3568 | 12267.201016 | 3.438117 |
| gru | 8 | 5+ | 7139 | 20974.631387 | 2.938035 |
| full | 6 | 0 | 0 | 0.000000 | NA |
| full | 6 | 1 | 0 | 0.000000 | NA |
| full | 6 | 2–4 | 0 | 0.000000 | NA |
| full | 6 | 5+ | 0 | 0.000000 | NA |
| full | 7 | 0 | 1188 | 5759.676865 | 4.848213 |
| full | 7 | 1 | 1184 | 4347.093211 | 3.671531 |
| full | 7 | 2–4 | 2744 | 8083.955914 | 2.946048 |
| full | 7 | 5+ | 1436 | 2605.540787 | 1.814443 |
| full | 8 | 0 | 1200 | 4941.403096 | 4.117836 |
| full | 8 | 1 | 1200 | 3005.172778 | 2.504311 |
| full | 8 | 2–4 | 3568 | 2852.770056 | 0.799543 |
| full | 8 | 5+ | 7139 | 1790.364661 | 0.250786 |
| surp | 6 | 0 | 0 | 0.000000 | NA |
| surp | 6 | 1 | 0 | 0.000000 | NA |
| surp | 6 | 2–4 | 0 | 0.000000 | NA |
| surp | 6 | 5+ | 0 | 0.000000 | NA |
| surp | 7 | 0 | 1188 | 5901.146580 | 4.967295 |
| surp | 7 | 1 | 1184 | 4436.398480 | 3.746958 |
| surp | 7 | 2–4 | 2744 | 7766.212781 | 2.830252 |
| surp | 7 | 5+ | 1436 | 2200.750689 | 1.532556 |
| surp | 8 | 0 | 1200 | 5114.396845 | 4.261997 |
| surp | 8 | 1 | 1200 | 3941.201385 | 3.284334 |
| surp | 8 | 2–4 | 3568 | 7320.822162 | 2.051800 |
| surp | 8 | 5+ | 7139 | 6334.171180 | 0.887263 |

| Condition | Seed | F value bits/digit, ≥1 prior |
|---|---:|---:|
| gru | 6 | NA |
| gru | 7 | 3.462795 |
| gru | 8 | 3.146774 |
| full | 6 | NA |
| full | 7 | 2.803242 |
| full | 8 | 0.642337 |
| surp | 6 | NA |
| surp | 7 | 2.685191 |
| surp | 8 | 1.477803 |

P: prequential whole-stream observation bits/symbol. T = P + peak S/N. S = stored symbols × log2(42) + 32 bits per chunk.
GRU: embedding 32, hidden 256, chunk 64, Adam lr 0.003, clip 1.0; carried and detached live state. All conditions rerun.
Store admission: completed aligned 64-symbol chunks; system prequential chunk P strictly above the linear percentile of all preceding chunk scores; first 8 chunks written. No deletion.
Replay: two Adam steps of eight uniformly sampled stored chunks with replacement per arrival chunk, after live update and write; zero hidden state. Shifted preceding-symbol input comes from the preceding chunk only when stored, otherwise zero embedding.
Match: H21b preceding-8 context, latest stored target, extension up to 32 bounded by stored-chunk start; calibrated four length buckets. Index changes only on completed chunk writes. Confidence updates after each prediction. Fixed share 0.001, initial weights (0.5, 0.5).
F exposure counts use all observed records across visits, independent of store admission. NA denotes no observations.
Total wall seconds: 4618.817.

Runtime note: surp90 dropped first: smoke projection exceeds 95-minute planning allowance within 100-minute wall budget.

## Verdict (orchestrator, 10 Nov 2026 UTC+8): FAIL, and a metric flaw found

1. **Priced store pays: FAIL.** T(surp) 2.79 vs T(gru) 1.73: the store saves 0.53 bits of P but is charged 1.59 bits of S/N.
2. **F digits at ≥ 1 prior appearance ≤ 1.5: FAIL** (seed 7 ≈ 2.69, seed 8 ≈ 1.48, weighted; seed 6 has no F).
3. Report: T(full) 6.88 (P 0.99 + S/N 5.89).

Mechanisms:
- **The two-part T in docs/h20_world.md is the wrong price.** In a prequential code the decoder has already decoded the
  history, so keeping it adds nothing to the description length. That is why Schmidhuber's "store everything" is free
  in code length. What a store costs is *memory* (bits held), a hardware resource on a different axis. T charged it
  1:1 as description length. From now on, report P and memory footprint separately (a Pareto front); a single number
  needs an explicitly declared exchange rate, motivated by the hardware goal.
- **Granularity:** chunk-level writes keep 64 symbols to save a few surprising digits.
- **Match order:** the match model keys on the last 8 symbols, which include the previous record's value digits, so a
  fact's key never matches twice. Facts need a shorter order (~5). Even the full store gets 2.5 bits on a value seen
  once, where exact recall would give 0.

On the axis that matters, the surprise store is efficient: 27% of chunks recover 72% of the full store's P gain (P: no
store 1.731, surp 1.206, full 0.991). Next: **H24f**, symbol-level surprise writes, match order 5, Pareto reporting.
