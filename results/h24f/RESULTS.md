# H24f measurements

Started: 2026-11-10 08:53:47 UTC+8. Updated: 2026-11-10 11:46:11 UTC+8.

Slow world; N = 131072; seeds = [6, 7, 8]; alphabet = 42; CPU threads = 2.

| Condition | Seed | P | Peak M/N | Fraction stored | Stored symbols | Windows | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru | 6 | 1.339368 | 0.000000 | 0.000000 | 0 | 0 | 193.633 |
| gru | 7 | 2.093612 | 0.000000 | 0.000000 | 0 | 0 | 190.770 |
| gru | 8 | 1.760025 | 0.000000 | 0.000000 | 0 | 0 | 183.070 |
| full | 6 | 0.622545 | 5.892317 | 1.000000 | 131072 | 2048 | 684.836 |
| full | 7 | 1.272562 | 5.892317 | 1.000000 | 131072 | 2048 | 674.432 |
| full | 8 | 1.079035 | 5.892317 | 1.000000 | 131072 | 2048 | 659.480 |
| full5 | 6 | 0.620883 | 5.892317 | 1.000000 | 131072 | 2048 | 714.804 |
| full5 | 7 | 1.226276 | 5.892317 | 1.000000 | 131072 | 2048 | 702.664 |
| full5 | 8 | 1.075444 | 5.892317 | 1.000000 | 131072 | 2048 | 696.916 |
| fine | 6 | 0.663810 | 1.723501 | 0.270950 | 35514 | 1075 | 707.731 |
| fine | 7 | 1.313592 | 2.686861 | 0.451370 | 59162 | 1036 | 680.204 |
| fine | 8 | 1.102926 | 2.830140 | 0.485321 | 63612 | 873 | 671.410 |
| fine3 | 6 | 0.754674 | 2.383891 | 0.381828 | 50047 | 1331 | 671.812 |
| fine3 | 7 | 1.376289 | 3.409629 | 0.586403 | 76861 | 1014 | 662.185 |
| fine3 | 8 | 1.130408 | 3.357808 | 0.572220 | 75002 | 1115 | 685.580 |

| Condition | Seeds | P mean ± population std | Peak M/N mean ± std | Fraction stored mean ± std |
|---|---:|---:|---:|---:|
| gru | 3 | 1.731002 ± 0.308602 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| full | 3 | 0.991381 ± 0.272510 | 5.892317 ± 0.000000 | 1.000000 ± 0.000000 |
| full5 | 3 | 0.974201 ± 0.257310 | 5.892317 ± 0.000000 | 1.000000 ± 0.000000 |
| fine | 3 | 1.026776 ± 0.270682 | 2.413501 ± 0.491397 | 0.402547 ± 0.094080 |
| fine3 | 3 | 1.087124 ± 0.255612 | 3.050443 ± 0.471798 | 0.513484 ± 0.093274 |

| Condition | Seed | G | C1 | C7 | E30 | E110 | M7 | M37 | F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | 6 | NA | 1.570243 | 2.405219 | 1.238582 | NA | 0.654884 | 0.466952 | NA |
| gru | 7 | 1.751203 | 3.395918 | 3.144010 | NA | 1.379465 | 0.585471 | NA | 3.733900 |
| gru | 8 | NA | 0.852376 | 2.023701 | NA | 1.173049 | 0.445717 | NA | 3.194302 |
| full | 6 | NA | 0.366176 | 0.733541 | 1.243366 | NA | 0.345031 | 0.316530 | NA |
| full | 7 | 0.930272 | 1.416133 | 1.002985 | NA | 1.043214 | 0.353308 | NA | 3.348156 |
| full | 8 | NA | 0.674508 | 0.841705 | NA | 1.195194 | 0.331688 | NA | 1.776144 |
| full5 | 6 | NA | 0.366252 | 0.733517 | 1.241260 | NA | 0.337491 | 0.314780 | NA |
| full5 | 7 | 0.897395 | 1.416126 | 1.002803 | NA | 1.043406 | 0.350726 | NA | 3.012939 |
| full5 | 8 | NA | 0.671866 | 0.841867 | NA | 1.195484 | 0.329347 | NA | 1.763821 |
| fine | 6 | NA | 0.451478 | 0.836882 | 1.212071 | NA | 0.335805 | 0.359217 | NA |
| fine | 7 | 1.097534 | 1.762654 | 1.002268 | NA | 1.072507 | 0.370995 | NA | 3.057764 |
| fine | 8 | NA | 0.578198 | 0.862788 | NA | 1.159394 | 0.320902 | NA | 1.939971 |
| fine3 | 6 | NA | 0.644197 | 0.995423 | 1.207077 | NA | 0.335742 | 0.359485 | NA |
| fine3 | 7 | 1.178459 | 1.729402 | 1.537538 | NA | 1.064132 | 0.346664 | NA | 3.025853 |
| fine3 | 8 | NA | 0.595474 | 0.830572 | NA | 1.157559 | 0.319889 | NA | 2.075821 |

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
| full5 | 6 | 0 | 0 | 0.000000 | NA |
| full5 | 6 | 1 | 0 | 0.000000 | NA |
| full5 | 6 | 2–4 | 0 | 0.000000 | NA |
| full5 | 6 | 5+ | 0 | 0.000000 | NA |
| full5 | 7 | 0 | 1188 | 5886.563917 | 4.955020 |
| full5 | 7 | 1 | 1184 | 1723.478685 | 1.455641 |
| full5 | 7 | 2–4 | 2744 | 3251.164212 | 1.184827 |
| full5 | 7 | 5+ | 1436 | 1825.991016 | 1.271581 |
| full5 | 8 | 0 | 1200 | 4995.992299 | 4.163327 |
| full5 | 8 | 1 | 1200 | 1996.476825 | 1.663731 |
| full5 | 8 | 2–4 | 3568 | 2443.862264 | 0.684939 |
| full5 | 8 | 5+ | 7139 | 1739.061272 | 0.243600 |
| fine | 6 | 0 | 0 | 0.000000 | NA |
| fine | 6 | 1 | 0 | 0.000000 | NA |
| fine | 6 | 2–4 | 0 | 0.000000 | NA |
| fine | 6 | 5+ | 0 | 0.000000 | NA |
| fine | 7 | 0 | 1188 | 6118.462945 | 5.150221 |
| fine | 7 | 1 | 1184 | 983.672419 | 0.830804 |
| fine | 7 | 2–4 | 2744 | 1976.172441 | 0.720179 |
| fine | 7 | 5+ | 1436 | 1132.944483 | 0.788959 |
| fine | 8 | 0 | 1200 | 4973.291063 | 4.144409 |
| fine | 8 | 1 | 1200 | 1760.657106 | 1.467214 |
| fine | 8 | 2–4 | 3568 | 2836.279521 | 0.794921 |
| fine | 8 | 5+ | 7139 | 4138.735622 | 0.579736 |
| fine3 | 6 | 0 | 0 | 0.000000 | NA |
| fine3 | 6 | 1 | 0 | 0.000000 | NA |
| fine3 | 6 | 2–4 | 0 | 0.000000 | NA |
| fine3 | 6 | 5+ | 0 | 0.000000 | NA |
| fine3 | 7 | 0 | 1188 | 5640.754499 | 4.748110 |
| fine3 | 7 | 1 | 1184 | 1175.259219 | 0.992618 |
| fine3 | 7 | 2–4 | 2744 | 2247.762095 | 0.819155 |
| fine3 | 7 | 5+ | 1436 | 1221.884075 | 0.850894 |
| fine3 | 8 | 0 | 1200 | 4955.762964 | 4.129802 |
| fine3 | 8 | 1 | 1200 | 1729.857516 | 1.441548 |
| fine3 | 8 | 2–4 | 3568 | 2238.225346 | 0.627305 |
| fine3 | 8 | 5+ | 7139 | 4741.206926 | 0.664128 |

| Condition | Seed | F value bits/digit, ≥1 prior |
|---|---:|---:|
| gru | 6 | NA |
| gru | 7 | 3.462795 |
| gru | 8 | 3.146774 |
| full | 6 | NA |
| full | 7 | 2.803242 |
| full | 8 | 0.642337 |
| full5 | 6 | NA |
| full5 | 7 | 1.267829 |
| full5 | 8 | 0.518972 |
| fine | 6 | NA |
| fine | 7 | 0.763011 |
| fine | 8 | 0.733659 |
| fine3 | 6 | NA |
| fine3 | 7 | 0.865941 |
| fine3 | 8 | 0.731443 |

| Condition | F ≥1 prior mean over observed seeds | F ≥1 prior pooled bits/digit |
|---|---:|---:|
| gru | 3.304785 | 3.244923 |
| full | 1.722790 | 1.313468 |
| full5 | 0.893400 | 0.751551 |
| fine | 0.748335 | 0.742775 |
| fine3 | 0.798692 | 0.773215 |

Pareto: P against peak M/N. gru has M = 0. full is H24e, reused.

| Condition | Seed | P | Peak M/N |
|---|---:|---:|---:|
| gru | 6 | 1.339368 | 0.000000 |
| gru | 7 | 2.093612 | 0.000000 |
| gru | 8 | 1.760025 | 0.000000 |
| full | 6 | 0.622545 | 5.892317 |
| full | 7 | 1.272562 | 5.892317 |
| full | 8 | 1.079035 | 5.892317 |
| full5 | 6 | 0.620883 | 5.892317 |
| full5 | 7 | 1.226276 | 5.892317 |
| full5 | 8 | 1.075444 | 5.892317 |
| fine | 6 | 0.663810 | 1.723501 |
| fine | 7 | 1.313592 | 2.686861 |
| fine | 8 | 1.102926 | 2.830140 |
| fine3 | 6 | 0.754674 | 2.383891 |
| fine3 | 7 | 1.376289 | 3.409629 |
| fine3 | 8 | 1.130408 | 3.357808 |

| Condition | Seeds | P mean ± population std | Peak M/N mean ± population std |
|---|---:|---:|---:|
| gru | 3 | 1.731002 ± 0.308602 | 0.000000 ± 0.000000 |
| full | 3 | 0.991381 ± 0.272510 | 5.892317 ± 0.000000 |
| full5 | 3 | 0.974201 ± 0.257310 | 5.892317 ± 0.000000 |
| fine | 3 | 1.026776 ± 0.270682 | 2.413501 ± 0.491397 |
| fine3 | 3 | 1.087124 ± 0.255612 | 3.050443 ± 0.471798 |

P: prequential observation bits/symbol. M = stored symbols × log2(42) + 32 bits/window; peak tracked after each merged-window write; denominator is final N.
gru and full measurements reused from H24e; their wall seconds are historical. No reruns. Full5 retains H24e aligned 64-symbol windows and replay sampling.
Match order 5, latest target; backwards extension up to 32 within stored windows. Confidence buckets: 5–7, 8–11, 12–15, 16–23, 24–32. Fixed share 0.001, initial weights (0.5, 0.5).
Fine thresholds: strictly >4.0 and >3.0 final-mixture bits. Inclusive windows [i−16,i+4], clipped to life; overlapping windows merged, touching windows kept separate. A right tail past the chunk is admitted after that later chunk is scored and joined to its own head.
Fine replay: uniformly sample stored windows; uniform 64-symbol segment in long windows, concatenate sampled short windows to 64. Zero embedding at each sampled window boundary; zero initial replay hidden state.
Live GRU and replay schedule: H24e Cortex; two batch-8 Adam replay steps after each scored chunk, live update then replay. Scoring before any current-chunk admission or weight update.
M follows the specified payload-plus-window formula; neural parameters, optimizer, match index and transient live context are outside that formula. F exposure counts use all observations; NA denotes absent observations.
New-run wall seconds (sum of full5, fine and fine3 run times; gru and full excluded): 6193.305.

Continuation wall seconds: 3395.000.
Runtime note: Touching windows stay separate; only overlapping windows merge. fine seed 6 was recomputed with that rule before this continuation (10:26). fine seeds 7 and 8 were recomputed in this continuation. full5 was not rerun.
Runtime note: fine3 is the same corrected store with threshold 3.0, seeds 6 7 8, run in this continuation.
Runtime note: This continuation ran one job at a time, Torch threads 2. Sequential wall 10:47:54 to 11:44:29 UTC+8.

## Verdict (orchestrator, 10 Nov 2026 UTC+8)

Run history: codex wrote the script and the full5 runs plus the first fine runs, then was stopped at its usage cap. grok
reviewed the code, found and fixed a touching-window merge bug, and reran fine (6, 7, 8) and fine3 (6, 7, 8) one at a time.

1. **Facts recalled: PASS.** full5 F value digits at ≥ 1 prior appearance: 0.893 (mean of seeds 7 and 8; pooled 0.752)
   vs 1.723 with order-8 matching. Order 5 fixes recall.
2. **M3 (Pareto): FAIL.** Best fine store P 1.027 ≤ full5 + 0.10 = 1.074 (pass), but peak M/N 2.41 = 41% of full5's
   (bar ≤ 15%).
3. **PASS:** fine's F value digits 0.748.

Mechanism: surprise-gated writes keep storing *irreducible* surprise (grammar choices, CA noise flips, first sightings
of facts), symbols that memory can never make predictable. About 40% of symbols cross the 4-bit threshold, each pulling
in a 21-symbol window. Lowering the threshold (fine3) stores more *and* does worse (P 1.087), which confirms it. It is the
noisy TV again, in memory. Following the guideline, the store should keep what yields compression progress, not what is
surprising. Next: **H24g**, utility-based eviction (credit each window with the bits it saved; evict the least useful
to hold a memory budget). MNIST passed with surprise writes (H24d) because its episodes recur exactly.
