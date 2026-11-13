# M6-W3: the curious agent may choose to sleep. Does sleep earn its cost, and does choosing when matter? (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/m5bw3/RESULTS.md` (including its Verdict), `m5_w3.py`, `m5b_w3.py`,
`curio.py` (`SGDCompressor`, `score_pairs`, the reservoir code in `live`) and `w3_world.py`.

## Context
In M5b, a progress controller in W3 beat random dwelling and the learnability oracle over a long life. The north star
also asks the machine to choose when to sleep, and to make sleep earn its cost. In W3 the agent keeps a store: a
whole-history reservoir per room channel (`R_RES` = 256 pairs, as in curio). Here sleep is an action. For one
decision the agent does not look at the world. It replays 64 stored (context, symbol) pairs through its compressor's
learning step. Sleep costs a decision, the same as a glance or a move. Principles served: store everything (1),
improve the compressor (2), and act to maximise progress (4), including the act of sleeping.

## What to build: `m6_w3.py`
Import `m5_w3.py`, `curio.py` and `w3_world.py`, and edit none of them. World, learner (`curio.SGDCompressor`,
`CURIO_LR=0.004`), corridor glances, the visit schedule (travel, then 8 dwell decisions), per-channel reservoirs and the score W
(and W9, as in `m5b_w3.py`) are as in M5b.

- **One yardstick for every reward: total bits saved on the whole store.** For any decision, its progress is
  Σ_c [score_pairs(c, res[c]) before − after], summed over the 9 room channels, scored on the reservoirs as they stand
  before the decision. Progress is measured before the decision's own pairs enter the reservoir. A visit's reward is
  the sum of its decisions' progress divided by the decisions it took (travel plus dwell), as in M5.
- **Sleep:** one decision. Draw 64 pairs uniformly from the union of all channels' reservoirs (rng `default_rng(30_000 +
  seed)`), and apply the compressor's learning step to each pair, with the pair's channel as its key. Pairs enter no
  reservoir. A **sleep visit** is 8 sleep decisions in a row, taken where the agent stands, with no travel. Its reward is
  its summed progress divided by 8.

Controllers (choices from rng `default_rng(10_000 + seed)`):
- `prog`: M5's controller exactly (its own per-channel reward, no sleep). This is the reference.
- `prog_total`: M5's bandit over the 9 rooms, with the whole-store reward above, and no sleep.
- `prog_sleep`: the same bandit over 10 arms, the 9 rooms plus sleep, with the whole-store reward. Sleep joins the
  optimistic first pass as the last arm. EPS = 0.1 and BETA = 0.3, as before.
- `fixed_sleep`: `prog_total`'s controller, but it must take a sleep visit at evenly spaced points. The spacing is chosen so
  that its share of sleep decisions over the life equals `prog_sleep`'s share on the same seed and L. This is a
  matched dose. Run `prog_sleep` first for each seed and L. If a forced sleep falls inside a visit, it starts right
  after that visit ends.

## Seeds and lives
**World seeds 18–23** (fresh). L = 2,048 for every controller (scored). L = 512 for every controller, as a report.

## Where to run
Whole-store scoring costs about 9× M5's per-decision scoring. Use private Kaggle CPU kernels, `ezzzzzx/m6-w3-s<seed>`, or
fewer kernels if the timing allows. Each has `"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. The code goes up as private dataset
`ezzzzzx/pdp-code-m6w3`. Use a pool of 4 inside a kernel. At most 90 minutes per kernel.

## Smoke test (VPS CPU, before any kernel)
1. `prog` in `m6_w3.py` reproduces M5b's prog W for seed 12, L = 512, to 1e-9.
2. A sleep decision changes no reservoir and no world state. Check that the world's stream positions and the reservoir
   contents are identical before and after a sleep visit.
3. For `prog_sleep`, seed 18, L = 256: print the visit trace (arm, decisions and reward), and the sleep share.
4. All four controllers give a finite W on seed 18 at L = 128. Time it and estimate the kernels' wall time.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/m6w3/RESULTS.md` and `results/m6w3/results.json`, with numbers only and no verdict:
- W and W9 per controller, L and seed, with mean ± std (ddof = 0);
- the gaps per seed: prog_sleep − prog_total, prog_sleep − fixed_sleep and prog_total − prog;
- sleep share per seed and L, and sleep share by quarter of life for `prog_sleep`;
- the shares of decisions per channel, in corridors and in sleep, per controller;
- the mean reward of sleep visits versus room visits for `prog_sleep`, by quarter;
- the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
1. **Sleep earns its cost:** at L = 2,048, mean over seeds of (prog_sleep − prog_total) ≤ −0.05 on W, with prog_sleep
   lower on at least 5 of 6 seeds.
2. **Choosing when matters:** at L = 2,048, mean over seeds of (prog_sleep − fixed_sleep) ≤ −0.03 on W, with
   prog_sleep lower on at least 4 of 6 seeds.
Report only: L = 512, W9, prog_total − prog, and the sleep share and its time course.
Kill rules: if bar 1 fails, replaying a 256-pair store does not pay for the decisions it costs this learner, so sleep
needs a bigger store or a slower cortex to matter. If bar 2 fails, the dose matters but the timing does not, and
organic sleep reduces to a schedule.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `m6_w3.py`, `kaggle/m6w3/**` and `results/m6w3/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `m5_w3.py`, `m5b_w3.py`, `curio.py`, `w3_world.py`, `w3_headroom.py`, other
  `results/` or `kaggle/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
