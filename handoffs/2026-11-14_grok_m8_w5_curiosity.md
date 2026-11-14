# M8-W5: curiosity on the bank cortex with sleep, in a world whose rules drift (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then the Verdicts in `results/w5/RESULTS.md` and `results/m7bw3/RESULTS.md`, and the code in
`w5_world.py`, `w5_headroom.py`, `w4_headroom.py`, `m7b_w3.py`, `m7_w3.py` and `curio.py`.

## Context
W5 has headroom. When one learnable room's rule is redrawn every 128 decisions, an oracle that knows which rooms changed beats random
dwelling plus sleep by 0.25 on tracking bits (5 of 6 seeds). Can the agent find that headroom from its own compression
progress, without being told where the change is? M7b's per-room progress bandit has one weakness here: a room it has stopped
visiting keeps its old, low value, so a swap there goes unseen until an ε-step lands on it. A second controller lets its
belief about a room grow stale. The longer since it looked, the more it might have changed. Principles served: reward compression progress (3) and act
to maximise it (4).

## What to build: `m8_w5.py`
Import `w5_headroom.py`, `w5_world.py` and the modules they use, and edit none of them. The world is `World5(world_seed, period = 128)`.
The cortex, sleep (fixed share 0.20, placed as in M7b), the stores, movement, dwell rules, T14 and the final W14 are all exactly as in W5. L = 2,048.

Controllers (choices from `default_rng(10_000 + world_seed)`):
- `rand_dwell_s` and `oracle_fresh_s`, as in W5.
- `prog_chan_s`: M7b's per-room progress bandit over the 18 rooms (optimistic first pass, EPS 0.1, BETA 0.3), with the per-room
  reward exactly as in M7b: Σ over the visited room's exam glances of (bits before − bits after), divided by the visit's decisions.
  Sleep visits update no bandit value.
- `prog_age_s`: the same, except that at each choice the greedy step ranks rooms by an aged value. With v_r the room's EMA value, age_r the decisions since
  its last visit, and v_max the largest single-visit reward seen so far in any room, the aged value is
  `v_r + (v_max − v_r) · (1 − exp(−age_r / τ))`, where τ = 256 decisions. This is fixed in advance; do not tune it. The ε-step and the first pass are unchanged.
  The stored EMA is not changed by aging.

## Seeds
**World seeds 12–23** (12 fresh seeds; W5's 0–5 were used for the headroom check). One life per controller and seed.

## Kernels
One private CPU kernel per seed, `ezzzzzx/m8-w5-s12` … `-s23`, each running the four controllers with a pool of up to 4. Each has
`"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. At most 90 minutes per kernel. The code goes up as private dataset `ezzzzzx/pdp-code-m8w5`.
Kaggle runs at most 5 CPU sessions at once; queue the rest.

## Smoke test (VPS CPU, before any kernel)
1. `rand_dwell_s` and `oracle_fresh_s` in `m8_w5.py` reproduce W5's T14 for seed 0, period 128, at L = 256 to 1e-6.
2. **The ageing is correct:** for `prog_age_s` with seed 12 and L = 512, at 5 choices print each room's v_r, age_r, aged value, v_max and the
   chosen room. Check by hand that a room with age 0 has its aged value equal to v_r.
3. For `prog_chan_s`, seed 12, L = 512: print the visit trace (room, decisions, reward) and the swap log side by side.
4. All four controllers give a finite T14 on seed 12 at L = 128. Time a 512-decision `prog_age_s` life and estimate a kernel's wall time.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/m8w5/RESULTS.md` and `results/m8w5/results.json`, with numbers only and no verdict:
- T14 and the final W14 per controller and seed, with mean ± std (ddof = 0), and T14 per checkpoint;
- the gaps per seed: prog_chan_s − rand_dwell_s, prog_age_s − rand_dwell_s, prog_age_s − prog_chan_s and oracle_fresh_s − rand_dwell_s,
  with the counts of seeds where the left term is lower;
- **swap-finding latency:** for each swap, the decisions until the controller next dwells in that room (censored at the end of life). Report the median and
  the share found within 128 decisions, per controller;
- shares of decisions per room, in corridors and in sleep; the swap logs;
- W5's numbers for reference; the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
**Precondition (headroom on these seeds):** mean over seeds 12–23 of (oracle_fresh_s − rand_dwell_s) ≤ −0.15 on T14. If it is not met,
the bars are void, and the result is reported as "no headroom".
1. **Curiosity finds the change:** mean of (prog_chan_s − rand_dwell_s) ≤ −0.10 on T14, with prog_chan_s lower on at least 9 of 12 seeds.
2. **Stale beliefs help:** mean of (prog_age_s − rand_dwell_s) ≤ −0.10 on T14, with prog_age_s lower on at least 9 of 12 seeds.
Report only: prog_age_s − prog_chan_s, the swap-finding latency, and the final W14.
Kill rules: if the precondition holds and both bars fail, progress felt only on visits cannot track a world whose learnability moves elsewhere.
The agent then needs a cheap signal of change from a distance, such as a glimpse through the door. If bar 2 passes and bar 1 fails, curiosity needs
a model of its own ignorance, not only of its progress.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `m8_w5.py`, `kaggle/m8w5/**` and `results/m8w5/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `w5_world.py`, `w5_headroom.py`, `w4_world.py`, `w4_headroom.py`, `m7_w3.py`, `m7b_w3.py`,
  `curio.py`, `h20_world.py`, other experiment code, other `results/` or `kaggle/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
