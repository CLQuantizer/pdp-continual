# H23c: compression progress on the whole history: PASS (all bars, including the stretch)

Run 9 Nov 2026, UTC+8, VPS. `CURIO_LR=0.004 curio.py --seeds 6 7 8 9 10 11 --glances 512 --conds prog --comp sgd --prog hist`.
Pre-registration: `docs/h23b_curiosity.md` (H23c section). Baselines are H23b's runs (same compressor, seeds, budget).

| Cond | W (6 seeds, lower = better) | N share | Z share |
|---|---|---|---|
| rand | 2.637 ± 0.052 | .11 | .11 |
| rand7 (oracle: knows N, Z are worthless) | 2.481 ± 0.046 | 0 | 0 |
| err (prediction error, 1990) | 4.355 ± 0.090 | .89 | .02 |
| cnt (count bonus) | 2.640 ± 0.043 | .11 | .11 |
| prog, held-out-next (H23b) | 2.998 ± 0.075 | .01 | .04 |
| **prog, whole-history eq. 5 (H23c)** | **2.373 ± 0.060** | **.01** | .03 |

prog − rand per seed: −0.232 −0.316 −0.263 −0.295 −0.218 −0.262 (6/6 ≤ −0.05). prog − rand7: −0.108 on the mean.
Held-out bits vs rand: M37 1.16 vs 2.72, G 3.21 vs 4.00, M7 0.61 vs 0.56, C1 4.35 vs 4.10, C7 4.37 vs 4.35, E30/E110 ≈ +0.1.

**A curriculum emerges.** Share by quarter of life:
- Q1: M7 .23, E30 .20, C1 .19, E110 .12
- Q2: M37 .39, C7 .15
- Q3: M37 .43, G .19
- Q4: **G .54**, M37 .21

It goes from easy to hard: short motif and CA first, then the long motif, then the grammar (the hardest source this
compressor can still improve on). Noise stays at 1% throughout.

## Bars

1 PASS (err 89% on N) · 2 PASS (1% N, 3% Z) · **3 PASS** (6/6; prog < err, prog < cnt) · **4 PASS** (beats the oracle).

## What it means

- The same compressor, bandit and budget fail with a sliding-window estimator (H23: memorisation) and a next-half
  estimator (H23b: tracking). They pass with eq. 5 measured the way the paper defines it: old vs new compressor on
  the *whole history*. The estimator is the mechanism.
- Computing that reward needs a uniform sample of everything seen so far: a store of the history. This is the
  hippocampus's job in this framework. It is what makes curiosity measurable, not a rival answerer
  (`docs/hippocampus_diagnosis.md`).

## Caveats

lr 0.004 makes learning curves long on purpose; the controller is an ε-greedy bandit with no state; one world;
256-pair reservoir (not swept). Counters and CA rows stay out of reach of a context model: that needs H21's substrate,
after which the same curiosity should find them interesting.
