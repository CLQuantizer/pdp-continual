# M7b-W3: curiosity on the bank cortex, with per-room progress and a fixed sleep dose (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/m7w3/RESULTS.md` (including its Verdict) and `m7_w3.py`.

## Context
In M7, sleep paid off on the bank cortex (−0.30, 6 of 6), and a fixed schedule at the same dose did as well as chosen sleep.
Curiosity broke, though. On shared weights, a room visit's whole-store progress nets its gain in that room against
interference everywhere else. After the first quarter it averaged negative, so the bandit chose rooms that hurt least. M7b
gives retention to sleep (a fixed dose for every controller) and gives curiosity the gain in the room visited (M5's
per-channel progress, which separates sources). Principles served: reward compression progress (3), act to maximise it
(4), and sleep (1, 2).

## What to build: `m7b_w3.py`
Import `m7_w3.py` and edit nothing. `BankCortex`, the stores, scoring, sleep decisions, the visit schedule, W, W9 and
forgetting are all exactly as in M7.

**Fixed sleep dose for every controller marked `_s`:** sleep visits (8 sleep decisions each) are forced at evenly spaced points so
that the sleep share of the life is 0.20, the mean of M7's prog_sleep share, rounded. Use the same placement rule as M7's
`fixed_sleep`: a forced sleep inside a visit starts right after that visit ends.

Controllers (choices from `default_rng(10_000 + seed)`):
- `rand_dwell`: as in M7, with no sleep. This is the reference for the sleep gain.
- `rand_dwell_s`: rand_dwell plus the fixed sleep dose.
- `oracle7_s`: oracle7 plus the fixed sleep dose.
- `prog_total_s`: M7's prog_total (whole-exam-store reward) plus the fixed sleep dose.
- `prog_chan_s`: the same bandit over the 9 rooms (optimistic first pass, EPS 0.1, BETA 0.3) plus the fixed sleep dose. Its reward
  for a room visit is Σ over that room's channel's exam glances of (bits before − bits after), summed over the visit's
  decisions and divided by the visit's decisions. Sleep visits update no bandit value.

## Seeds and lives
**World seeds 36–41** (fresh). L = 2,048.

## Kernels
As in M7: private CPU kernels, one per seed (`ezzzzzx/m7b-w3-s36` … `-s41`), pool of up to 4, at most 90 minutes each. Each has
`"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. The code goes up as private dataset `ezzzzzx/pdp-code-m7bw3`. Kaggle runs at most 5 CPU sessions at
once; queue the sixth.

## Smoke test (VPS CPU, before any kernel)
1. `rand_dwell` in `m7b_w3.py` reproduces M7's rand_dwell W for seed 30 at L = 64 to 1e-6.
2. For `rand_dwell_s` on seed 36 at L = 512, print the sleep share. It must be 0.20 ± 0.02.
3. For `prog_chan_s`, seed 36, L = 256: print a visit trace (room, decisions, per-channel reward, and the whole-store
   progress for comparison).
4. All five controllers give a finite W on seed 36 at L = 64. Estimate a kernel's wall time; if it is over 90 minutes, stop and report.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/m7bw3/RESULTS.md` and `results/m7bw3/results.json`, with numbers only and no verdict:
- W and W9 per controller and seed, with mean ± std (ddof = 0), and the per-channel held-out bits;
- the gaps per seed: prog_chan_s − rand_dwell_s, prog_chan_s − prog_total_s, prog_chan_s − oracle7_s and
  rand_dwell_s − rand_dwell;
- forgetting per controller, as in M7;
- shares per channel, in corridors and in sleep, per controller, and by quarter for prog_chan_s;
- the mean per-channel reward by quarter for prog_chan_s;
- for reference, M7's numbers;
- the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
1. **Curiosity pays on a cortex that forgets, once sleep handles retention:** mean over seeds of (prog_chan_s −
   rand_dwell_s) ≤ −0.10 on W, with prog_chan_s lower on at least 5 of 6 seeds.
2. **The reward is what was broken:** mean over seeds of (prog_chan_s − prog_total_s) ≤ −0.10 on W.
Report only: oracle7_s, rand_dwell_s − rand_dwell (the sleep gain, replicated), W9 and forgetting.
Kill rules: if bar 1 fails, then on this cortex, once sleep holds what was learned, where the agent spends its time adds
nothing beyond random dwelling, and curiosity needs a slower or modular cortex to matter. If bar 2 fails, the
per-room reward is not what fixes curiosity here.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `m7b_w3.py`, `kaggle/m7bw3/**` and `results/m7bw3/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `m7_w3.py`, and all earlier experiment code, other `results/` or `kaggle/`
  folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
