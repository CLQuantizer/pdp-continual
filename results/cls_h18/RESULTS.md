# H18: dense pairing under 2× cortex width

Pre-registered 08:58 UTC+8, 8 Nov 2026 (see `cls_sleep.py`). Runs finished on the box the same day
(`results/cls_h18_s49`, `results/cls_h18_bg002`, ~50 min/seed); the daily loop failed before write-up. Scored and
closed 09:00 UTC+8, 9 Nov 2026 from the on-disk JSONs. Both worlds, seeds 6-8, forgetful store (f2k), awake arrivals
only. Baseline `awu_w2_f2k` reused from H17 (current core = awu every-38, hidden=1000).

- awid_w2_f2k: H16 dense pairing (every arrival one step on [x] plus, with probability 32/38, one uniform store
  prototype; 38 steps per 38 arrivals, 0.84 replayed/arrival) with ForwardForward `hidden=1000` explicit.

Bar. KEEP as default awake replay only if, in BOTH worlds, ff_bits.all <= awu_w2_f2k − 0.02 AND lifetime >=
awu_w2_f2k − 0.05.

| World | Machine | ff_bits all (6 / 7 / 8) | Mean | Return bits | Lifetime | Returns | Swap |
|---|---|---|---|---|---|---|---|
| s49 | awu_w2_f2k | 0.4234 / 0.4196 / 0.4036 | 0.4155 | 1.407 | 95.82 | 69.9 | 67.0 |
| s49 | awid_w2_f2k | 0.3800 / 0.3890 / 0.3632 | 0.3774 | 1.054 | 95.69 | 75.7 | 65.1 |
| bg002 | awu_w2_f2k | 0.4171 / 0.4248 / 0.4068 | 0.4162 | 1.415 | 95.91 | 68.5 | 75.1 |
| bg002 | awid_w2_f2k | 0.3805 / 0.3844 / 0.3668 | 0.3772 | 1.061 | 95.73 | 74.5 | 71.6 |

**awid_w2_f2k: compression PASSES, lifetime guard FAILS → KILLED as default.** Whole-history bits fall 0.038 (s49) /
0.039 (bg002), lower in 6/6 seeds (~2× the −0.02 bar). Return bits −0.35 / −0.35; returns +5.8 / +6.0. But combined
lifetime drops 0.13 / 0.18 (guard −0.05), every seed — the same trade H16 saw at hidden=500 (−0.15 / −0.18).

Reading. Mirzadeh et al. (ICML 2022, "Wide Neural Networks Forget Less Catastrophically") argue wider nets forget less
mainly because gradients of new tasks stay more orthogonal to old ones. H17 already used that capacity lever and cleared
the bits bar under separate-step awu. H18 asks whether the same width also dissolves H16's pairing lifetime cost. It
does not: at hidden=1000 the bits win and the present-stream cost both reappear at nearly the H16 magnitudes. So the
pairing trade is not a capacity-starvation artifact; under Adam each step still moves ~lr regardless of batch, and
co-present old items still take about half of most update directions. Core unchanged: z5 + c3 + awu every-38 +
hidden=1000. Dense pairing stays a non-default trade-off curve.

Next. H19 cortex plasticity at the knee (Ash & Adams 2020 shrink-and-perturb, or slower wake LR) under standing awu
every-38 hidden=1000 — KEEP only if ff_bits.all <= awu_w2 − 0.02 AND lifetime >= awu_w2 − 0.05 in both worlds. Fallback
H9 store confidence for the router gap.
