# H23: curiosity as compression progress on an active world: KILLED (bar 3), mechanism partly shown

Run 9 Nov 2026, UTC+8, on the VPS (`curio.py --seeds 6 7 8 9 10 11 --glances 512`, ~4 min). Pre-registration:
`docs/h23_curiosity.md`. 512 glances × 64 symbols per life; W = mean bits/symbol of the final frozen compressor on a
fresh 2 048-symbol continuation of each of the 7 learnable channels (lower is better).

| Cond | W (mean ± std, 6 seeds) | Share on N (noise) | Share on Z (dark) | Notable allocation |
|---|---|---|---|---|
| rand | 1.667 ± 0.019 | 0.11 | 0.11 | uniform |
| rand7 (oracle: no N/Z) | **1.636 ± 0.024** | 0.00 | 0.00 | uniform over learnables |
| err (prediction error) | 2.147 ± 0.057 | **0.89** | 0.02 | trapped by the noisy TV |
| prog (compression progress) | 1.693 ± 0.047 | **0.01** | 0.03 | C1 0.49 (0.70 in the last half) |
| cnt (count bonus) | 1.665 ± 0.017 | 0.11 | 0.11 | ≈ uniform |

prog − rand per seed: +0.005, +0.079, +0.017, +0.111, +0.017, −0.072 (1/6 at ≤ −0.05). prog − rand7: +0.026 mean.

prog's share by quarter of life: Q1 G .22 C1 .21 M37 .24 M7 .12 → Q2 G .27 C1 .33 → Q3 C1 .70 → Q4 C1 .71 C7 .21.

## Bars

1. Noisy-TV trap: err ≥ 60% on N. **PASS** (89%).
2. prog ≤ 15% on N and on Z. **PASS** (1%, 3%).
3. W(prog) ≤ W(rand) − 0.05 in ≥ 5/6 seeds. **FAIL** (1/6; mean +0.026, worse than rand).
4. Stretch, W(prog) ≤ W(rand7). **FAIL**.

## Why

- **The bar exceeded the headroom.** The oracle that skips noise and dark gains only 0.031 over rand. The KT
  compressor learns everything it *can* learn here within a few glances (motifs, grammar to its order-6 limit), so
  allocation hardly matters. The pilot should have measured oracle headroom; it is now a rule in CLAUDE.md.
- **The progress estimate was biased.** It is measured on a 256-symbol sliding window of the channel's own recent
  history under a memorising count model. On the step-1 counter, the recent prefixes ("12xx;") keep recurring, so each
  glance re-fits the window and looks like progress. That never transfers to fresh data (held C1 3.51, worse than
  rand's 3.36). Phase 1 behaves as the paper predicts: prog visits the grammar and long motif while they are
  learnable and leaves them once they are learnt. Phase 2 is the estimator artefact.

## What carries forward

- Progress-driven attention avoids both noise and boredom; error-driven attention does not. This is the qualitative
  core of Schmidhuber 1991 vs 1990, reproduced.
- **H23b:** a slow parametric compressor (so allocation has headroom; check the oracle first), and progress measured
  so memorising earns nothing: old vs new compressor on the *next* glance before it is learnt (held-out progress), or
  on the whole history with the model's description length charged.
