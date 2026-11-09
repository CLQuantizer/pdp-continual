# H19: Ash & Adams shrink-and-perturb under standing awu every-38

Pre-registered 08:55 UTC+8, 9 Nov 2026 (see `cls_sleep.py`), scored ~09:30 UTC+8. Both worlds, seeds 6-8, forgetful
store (f2k), awake arrivals only, run on the box (`results/cls_h19_s49`, `results/cls_h19_bg002`). Baseline
`awu_w2_f2k` reused from H17 (current core = awu every-38, hidden=1000).

- awu_sp_f2k: standing awu every-38, hidden=1000, plus at the start of every day (every 4000 ticks) Ash & Adams 2020
  shrink-and-perturb: each cortex weight/bias ← 0.8 · W + N(0, 0.01²), then Adam moments reset.

Bar. KEEP only if, in BOTH worlds, ff_bits.all <= awu_w2_f2k − 0.02 AND lifetime >= awu_w2_f2k − 0.05.

| World | Machine | ff_bits all (6 / 7 / 8) | Mean | Return bits | Lifetime | Returns | Swap |
|---|---|---|---|---|---|---|---|
| s49 | awu_w2_f2k | 0.4234 / 0.4196 / 0.4036 | 0.4155 | 1.407 | 95.82 | 69.9 | 67.0 |
| s49 | awu_sp_f2k | 0.5361 / 0.5326 / 0.5121 | 0.5269 | 1.857 | 95.01 | 61.4 | 65.2 |
| bg002 | awu_w2_f2k | 0.4171 / 0.4248 / 0.4068 | 0.4162 | 1.415 | 95.91 | 68.5 | 75.1 |
| bg002 | awu_sp_f2k | 0.5322 / 0.5293 / 0.5190 | 0.5268 | 1.853 | 95.13 | 59.9 | 75.3 |

**awu_sp_f2k KILLED.** Whole-history bits rise +0.111 (s49) / +0.111 (bg002) — the wrong direction vs the −0.02 bar —
in 6/6 seeds; lifetime −0.81 / −0.78 (guard −0.05); returns −8.5 / −8.6. Daily shrink-and-perturb at λ=0.8, σ=0.01
does not buy plasticity for this dual; it undoes consolidation.

Reading. Ash & Adams (NeurIPS 2020) designed shrink-and-perturb for batch-online warm starts: shrink preserves ReLU
hypotheses while rebalancing gradients of old vs new data across a discrete training round. Here the "round" is a
circadian day on a stream that already interleaves awake replay every 38th arrival. Re-raising entropy once per day
after the cortex has been compressing under continuous SymBa+Adam erases more than it unlocks — consistent with H15's
reading that past the knee the lever is how each rehearsal lands, not resetting the learner. Core unchanged: z5 + c3 +
awu every-38 + hidden=1000.

Next. Ranked: (1) H19b slower wake LR (5e-4) under standing awu every-38 hidden=1000 — KEEP only if ff_bits.all <=
awu_w2 − 0.02 AND lifetime >= awu_w2 − 0.05 (a learning-rate plasticity lever without weight reset). (2) Gentler
shrink-and-perturb (λ≥0.95 or rarer than daily) only as a separate pre-registered mechanism if H19b fails. (3) H9
store confidence for the router gap. Park contagion/twins/vetoes.
