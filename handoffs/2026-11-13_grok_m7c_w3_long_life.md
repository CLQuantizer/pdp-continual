# M7c-W3: curiosity on the bank cortex with sleep, over longer lives and more seeds, headroom first (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/m7bw3/RESULTS.md` (including its Verdict) and `m7b_w3.py`.

## Context
In M7b, with a fixed sleep dose, per-room curiosity on the bank cortex beat random dwelling plus the same sleep by 0.067 on the
mean, but on only 3 of 6 seeds. The learnability oracle's own headroom was only 0.10, and the seed std was 0.16. In M5b,
curiosity's edge grew with life length. M7c checks headroom at a longer life and uses twice the seeds. Principles
served: reward compression progress (3), and act to maximise it (4).

## What to run
`m7b_w3.py` unchanged (import it). Controllers `rand_dwell_s`, `oracle7_s` and `prog_chan_s`, exactly as in M7b
(fixed sleep share 0.20). Lives L ∈ {2,048, 4,096}. **World seeds 42–53** (12 fresh seeds).

## Kernels
One private CPU kernel per seed (`ezzzzzx/m7c-w3-s42` … `-s53`), each running 3 controllers × 2 lives with a pool of up to
4, at most 90 minutes. Each has `"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. The code goes up as private dataset `ezzzzzx/pdp-code-m7cw3`.
Kaggle runs at most 5 CPU sessions at once; queue the rest.

## Smoke test (VPS CPU, before any kernel)
1. `rand_dwell_s` reproduces M7b's W for seed 36 at L = 64 to 1e-6. (If M7b has no L = 64 number on disk, rerun M7b's code at L = 64 and compare.)
2. All three controllers give a finite W on seed 42 at L = 64. Time a 512-decision `prog_chan_s` life and estimate a
   kernel's wall time. If it is over 90 minutes, split the two lives into separate kernels. Do not shrink anything.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/m7cw3/RESULTS.md` and `results/m7cw3/results.json`, with numbers only and no verdict:
- W and W9 per controller, L and seed, with mean ± std (ddof = 0);
- the gaps per seed and L: prog_chan_s − rand_dwell_s, oracle7_s − rand_dwell_s and prog_chan_s − oracle7_s, with the counts
  of seeds where the left term is lower;
- forgetting per controller and L; shares per channel, in corridors and in sleep;
- the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
**Precondition (headroom):** at L = 4,096, mean over seeds of (oracle7_s − rand_dwell_s) ≤ −0.15. If it is not met, the bars are
void, and the result is reported as "no headroom".
1. **Curiosity pays over a long life:** at L = 4,096, mean over seeds of (prog_chan_s − rand_dwell_s) ≤ −0.10, with prog_chan_s
   lower on at least 9 of 12 seeds.
2. **Its edge grows with life:** the mean of (prog_chan_s − rand_dwell_s) at L = 4,096 is lower than at L = 2,048.
Report only: prog_chan_s − oracle7_s, W9 and forgetting.
Kill rule: if the precondition holds and bar 1 fails, then per-room progress cannot find the allocation on a shared,
forgetting cortex that an oracle shows exists. The next step then is a progress estimate that accounts for interference,
or a modular cortex.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `kaggle/m7cw3/**`, `results/m7cw3/**`, and a small runner `m7c_w3.py` that only imports `m7b_w3.py`.
- Do not touch: `docs/loop.md`, `audit.md`, `m7b_w3.py`, `m7_w3.py` and all earlier experiment code, other `results/` or
  `kaggle/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
