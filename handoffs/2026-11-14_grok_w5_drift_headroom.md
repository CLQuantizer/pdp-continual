# W5: W4 with drifting rules. Headroom check for allocation when learnability changes over time (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then the Verdict in `results/w4/RESULTS.md`, and the code in `w4_world.py`, `w4_headroom.py`,
`m7_w3.py` and `h20_world.py`.

## Context
In W4 time is scarce (a 4× longer life halves the bits), but spreading time evenly is already near the best allocation.
The learnable rooms are stationary and pay at about the same rate, so the learnability oracle beats random dwelling plus sleep by only
0.008. What an allocation can gain is set by how unequal and how changing the rooms' learning rates are. W5 makes them change.
At fixed intervals one learnable room's rule is replaced by a fresh draw of the same process, so the place where there is
something new to learn moves. Headroom is measured before any controller runs. Principle served: act to maximise
compression progress (4), by checking that a world with changing learnability rewards knowing where the change is.

## What to build
1. **`w5_world.py`** (import `w4_world.py` and `h20_world.py`; edit neither). `World5(world_seed, period)` is `World4(world_seed)`
   plus drift:
   - Every `period` decisions (at decisions period, 2·period, …), one learnable room is chosen uniformly from the 14
     (rng `default_rng(world_seed + 80_000)`). Its source is replaced by the same process from a new
     `h20_world.World(world_seed + 2_000 + k)`, where k counts the swaps from 0, and the stream starts from that instance's
     beginning. Room names, ring order, noise rooms and Z are unchanged.
   - Log every swap: the decision, the room and k.
   - `copy()` must copy the drift state, so a held-out continuation follows the room's current rule.
2. **`w5_headroom.py`**: the cortex, sleep (fixed share 0.20, placed as in M7b), the stores and the policies' movement are exactly
   as in `w4_headroom.py`. Import it, and generalise by wrapping or subclassing only. Policies (rng `default_rng(1_000 +
   world_seed)`):
   - `rand_dwell_s` and `oracle14_s`, as in W4.
   - `oracle_fresh_s`: it knows the swap log. It keeps the set S of learnable rooms whose rule has changed since it last
     dwelt there, and at the start S holds all 14. Its target is uniform over S if S is not empty, otherwise uniform over the 14. It reaches the target by the shortest path and dwells
     8 decisions. Arriving at a room removes it from S.
   Score **T14** (tracking): at the 8 checkpoints decision = L/8, 2L/8, …, L, freeze a copy of the cortex. Score it as W4's
   W14 is scored (`evaluate_w14`: EVAL = 2,048 symbols per learnable room, from a copy of the world at that moment, from a zero state
   with the room's channel embedding). Then continue the live life unchanged. T14 is the mean over the 8 checkpoints.
   Also report the final W14 (the last checkpoint).

## Seeds and lives
World seeds 0–5 (the same as W4, so drift's effect can be compared seed by seed). L = 2,048. period ∈ {128, 512}. Every policy at
both periods. Also `rand_dwell_s` with no drift (period = ∞) as the W4 anchor, scored with T14.

## Kernels
One private CPU kernel per seed, `ezzzzzx/w5-s0` … `-s5`. Each has `"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. Use a pool of up to 4.
At most 90 minutes per kernel; split by period if the timing needs it. The code goes up as private dataset `ezzzzzx/pdp-code-w5`.
Kaggle runs at most 5 CPU sessions at once; queue the rest.

## Smoke test (VPS CPU, before any kernel)
1. With period = ∞, `World5` gives the same glances as `World4` for 200 decisions on seed 0, and `rand_dwell_s`'s
   final W14 at L = 64 equals W4's to 1e-6.
2. With period = 16, seed 0, L = 128: print the swap log. Show that a swapped room's next glance comes from the new
   instance: print the first 64 symbols before and after one swap. Show that `copy()` continues the current rule.
3. `oracle_fresh_s`, period = 16, L = 128: print its targets and S at each target choice.
4. All three policies give a finite T14 on seed 0, period = 128, at L = 256. Time a 512-decision life including
   the checkpoints, and estimate the kernels' wall times.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/w5/RESULTS.md` and `results/w5/results.json`, with numbers only and no verdict:
- T14 and the final W14 per policy, period and seed, with mean ± std (ddof = 0), and T14 per checkpoint;
- the gaps per seed and period: oracle_fresh_s − rand_dwell_s and oracle14_s − rand_dwell_s on T14, with the counts of seeds
  where the left term is lower;
- rand_dwell_s's T14 with drift minus without, per seed and period;
- per room: bits at each checkpoint, split by whether the room was swapped since the previous checkpoint;
- shares of decisions per room, in corridors and in sleep; the swap logs;
- W4's numbers for reference; the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
**Precondition (drift bites):** at period = 128, the mean over seeds of rand_dwell_s's T14 with drift minus without is ≥ +0.10. If it is
not met, the bars are void and the result is reported as "drift too weak".
1. **Knowing where the change is pays:** at period = 128, oracle_fresh_s − rand_dwell_s ≤ −0.15 on T14, on at least 5 of 6 seeds.
Report only: period = 512, oracle14_s, and the final W14.
Kill rule: if the precondition holds and bar 1 fails, then even knowing exactly where the rules changed does not beat spreading
time evenly on this cortex. A shared cortex re-learns a swapped room as fast through general skill as through targeted visits, and
curiosity needs a world where some sources are slow or impossible to learn, not just ones that change.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `w5_world.py`, `w5_headroom.py`, `kaggle/w5/**` and `results/w5/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `w4_world.py`, `w4_headroom.py`, `m7_w3.py`, `m7b_w3.py`, `curio.py`,
  `h20_world.py`, other experiment code, other `results/` or `kaggle/` folders, `.env`. Do not edit H33's files
  (`h33_bank_text.py`, `kaggle/h33/**`, `results/h33/**`), which another worker may be writing.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
