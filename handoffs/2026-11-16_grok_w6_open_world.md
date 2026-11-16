# W6: an open world. Rooms are born and die; headroom for allocation, and a first read of the ageing curious agent (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then the Verdicts in `results/w5/RESULTS.md`, `results/m8w5/RESULTS.md` and
`results/m9w5/RESULTS.md`, and the code in `w4_world.py`, `w5_world.py`, `w5_headroom.py`, `m8_w5.py` and `h20_world.py`.

## Context
Curiosity paid where the world outruns the learner (W5: one room's rule redrawn every 128 decisions). It did not pay in closed,
stationary worlds (W3 with the bank, W4). A living machine's world is open: new things appear, some learnable and some not,
and old things go away. W6 makes that the world's rule instead of building closed worlds and checking them for headroom.
Principle served: act to maximise compression progress (4), in a world where what is learnable keeps changing.

## What to build
1. **`w6_world.py`** (import `w4_world.py`, `w5_world.py` and `h20_world.py`; edit none). `World6(world_seed, birth_period)`
   starts as `World4(world_seed)` (same ring of 18 slots, same initial rooms). Then:
   - Every `birth_period` decisions (at birth_period, 2·birth_period, …), one slot is chosen uniformly from the 17 slots that
     are not the dark room Z (rng `default_rng(world_seed + 90_000)`). Its room **dies** and a **new room is born** in that slot.
   - The newborn's kind is drawn uniformly from the 8 kinds `h20_world.PROCS + ["N"]` (same rng). A learnable kind uses a
     fresh `h20_world.World(world_seed + 3_000 + k)` and starts from that instance's beginning, where k counts births from 0.
     Kind "N" is a uniform noise stream over the 41 symbols with its own rng `default_rng(world_seed + 95_000 + k)`.
   - A slot keeps its ring position and its channel index (the cortex's channel row is the slot's, so the agent is not told that a
     room was replaced). The dark room never changes.
   - Log every birth: decision, slot, dead room's kind, newborn's kind, k. `copy()` copies everything, so a held-out
     continuation follows the slot's current room.
   - With `birth_period = None`, `World6` must behave exactly as `World4`.
2. **`w6_headroom.py`**: cortex, fixed sleep share 0.20 and its placement, stores and movement exactly as in `w5_headroom.py`
   (import it; generalise by wrapping or subclassing only). Policies (rng `default_rng(1_000 + world_seed)`):
   - `rand_dwell_s`, as in W5.
   - `oracle_learn_s`: knows which slots currently hold a learnable room. Target uniform over the learnable slots; shortest path;
     dwell 8 decisions.
   - `oracle_new_s`: knows the birth log and the kinds. It keeps the set S of learnable slots whose room was born since it last
     dwelt there (at the start S holds every learnable slot). Target uniform over S if not empty, else uniform over the
     learnable slots; shortest path; dwell 8. Arriving removes the slot from S.
   - `prog_age_s`: M8's ageing progress bandit (τ = 256, ε 0.1, β 0.3, exactly as in `m8_w5.py`), run on World6 with arms =
     slots. It is told nothing about births. Report only.
   **Score T_open:** at the 8 checkpoints decision = L/8, 2L/8, …, L, freeze a copy of the cortex and score it as W5's W14 is scored
   (EVAL = 2,048 symbols per room, from a copy of the world at that moment, zero state, the slot's channel row), over the slots that
   hold a **learnable** room at that checkpoint. T_open is the mean over the 8 checkpoints of the mean over those slots. Then continue
   the live life unchanged. Also report the life's prequential bits on glances at learnable rooms, and the share of decisions
   spent in noise slots (initial N1–N3 and newborn N).

## Seeds and lives
World seeds 0–5. **L = 4,096.** birth_period ∈ {128, 512}. Every policy at both periods. Also `rand_dwell_s` with
birth_period = None (closed W4 world), scored with T_open, as the anchor.

## Kernels
One private CPU kernel per seed, `ezzzzzx/w6-s0` … `-s5`, each `"is_private": true`, `"enable_gpu": false`,
`"enable_internet": false`. Pool of up to 4 inside a kernel. Code as private dataset `ezzzzzx/pdp-code-w6`. At most 90 minutes per
kernel; split by period if the timing needs it. **Run at most 2 kernels at once** (another worker uses Kaggle today).

## Smoke test (VPS CPU, one process, before any kernel)
1. With birth_period = None, `World6` gives the same glances as `World4` for 200 decisions on seed 0, and `rand_dwell_s`'s final
   W14 at L = 64 equals W4's to 1e-6.
2. birth_period = 16, seed 0, L = 160: print the birth log; for one birth print 64 symbols from the slot before and after it; show
   that `copy()` continues the newborn; show at least one newborn of kind N over a longer log if none appears in 160 decisions.
3. `oracle_new_s`, birth_period = 16, L = 160: print its targets and S at each choice.
4. All four policies give a finite T_open on seed 0, birth_period = 128, L = 512. Time a 1,024-decision life with checkpoints and
   estimate kernel wall times.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/w6/RESULTS.md` and `results/w6/results.json`, numbers only, no verdict:
- T_open and the final checkpoint's score per policy, period and seed, mean ± std (ddof = 0), and per checkpoint;
- per seed and period: oracle_learn_s − rand_dwell_s, oracle_new_s − rand_dwell_s and prog_age_s − rand_dwell_s on T_open, with
  counts of seeds where the left term is lower;
- rand_dwell_s's T_open with births minus without, per seed and period;
- per checkpoint: bits on newborn rooms (born since the previous checkpoint) vs older rooms, per policy;
- birth logs; share of decisions per slot, in noise slots, in corridors and in sleep; latency from a birth to the first stay
  there, per policy;
- the smoke-test output, wall seconds, kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
**Precondition (openness bites):** at birth_period = 128, mean over seeds of rand_dwell_s's T_open with births minus without
is ≥ +0.10. If not met, the bars are void ("world too closed").
1. **Allocation headroom in the open world:** at birth_period = 128, min(oracle_learn_s, oracle_new_s) − rand_dwell_s ≤ −0.15 on
   T_open, on at least 5 of 6 seeds.
Report only: birth_period = 512, prog_age_s, the life's prequential bits.
Kill rule: if the precondition holds and bar 1 fails, knowing what is new and what is learnable does not beat spreading time
evenly in an open world on this cortex; the forward-looking controller then has nothing to win on W6 and W6 must make
learnability more unequal (slower kinds, more noise) before any controller runs.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `w6_world.py`, `w6_headroom.py`, `kaggle/w6/**`, `results/w6/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `w4_world.py`, `w4_headroom.py`, `w5_world.py`, `w5_headroom.py`, `m8_w5.py`,
  `m9_w5.py`, `h20_world.py`, other experiment code, other `results/` or `kaggle/` folders, `.env`. Another worker writes
  `h34a_local_depth.py`, `kaggle/h34a/**`, `results/h34a/**` today: do not edit them.
- Gates and bars are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.
