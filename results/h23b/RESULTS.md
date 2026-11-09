# H23b: slow compressor + held-out progress: KILLED (prog worse than random on every seed)

Run 9 Nov 2026, UTC+8, VPS. `CURIO_LR=0.004 curio.py --seeds 6 7 8 9 10 11 --glances 512 --comp sgd --prog heldout`.
Pre-registration: `docs/h23b_curiosity.md`. Headroom present (pilot: rand − rand7 = +0.134).

| Cond | W (6 seeds) | N share | Z share | Allocation |
|---|---|---|---|---|
| rand | 2.637 ± 0.052 | .11 | .11 | uniform |
| rand7 (oracle) | **2.481 ± 0.046** | 0 | 0 | uniform over learnables |
| err | 4.355 ± 0.090 | **.89** | .02 | noisy TV |
| prog | 2.998 ± 0.075 | **.01** | .04 | C1 .59 (Q4: .84), G .01 |
| cnt | 2.640 ± 0.043 | .11 | .11 | ≈ uniform |

prog − rand: +0.44 +0.33 +0.43 +0.32 +0.36 +0.29 (0/6 pass). Bars 1, 2 pass; 3, 4 fail.
Held-out per channel, prog vs rand: C1 3.60 vs 4.10 (better); G 5.12 vs 4.00, M37 3.87 vs 2.72 (much worse).

## Why

1. **Held-out-next progress rewards tracking, not lasting compression.** A step-1 counter keeps entering new prefixes;
   learning the current ones predicts the next 32 symbols, so progress never runs out. It does not accumulate into
   knowledge of the source's history. Eq. 5 of the guideline scores old vs new on the **whole history**, where
   tracking that forgets old prefixes earns zero.
2. **Starvation by a greedy controller.** With a slow learner, one early glance of the grammar shows little progress;
   ε = 0.1 over 9 arms gives each loser ~1%, so G is never revisited (Oudeyer 2007 and Graves 2017 use
   proportional / bandit allocation over learning progress for this reason).

Both estimators so far (H23 window, H23b next-half) departed from the paper. Next, H23c: the faithful estimator.
