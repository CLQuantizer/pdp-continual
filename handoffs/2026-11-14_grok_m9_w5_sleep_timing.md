# M9-W5: does chosen sleep pay in a drifting world? Sleep as an arm of the ageing curious agent (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then the Verdicts in `results/m8w5/RESULTS.md`, `results/w5/RESULTS.md`, `results/m7w3/RESULTS.md` and
`results/m7bw3/RESULTS.md`, and the code in `m8_w5.py`, `w5_headroom.py`, `m7_w3.py` and `m7b_w3.py`.

## Context
In M7 (W3, stationary) sleep paid −0.30 on the bank cortex, and a fixed schedule at the same dose did as well as chosen sleep.
There, timing did not matter beyond the dose. W5's rules drift. The replay store holds glances of rules that may be gone, so
sleeping at the wrong time can rehearse the past. In M8, curiosity with ageing beliefs (`prog_age_s`) tracked the drift and beat an
oracle, with every controller at a fixed sleep share of 0.20. M9 asks two questions in this world. Does sleep pay at all? And does an agent that
chooses when to sleep, from the progress sleep brings, beat the same dose on a fixed clock? Principles served: store everything (1),
improve the compressor (2), and act to maximise progress (4).

## What to build: `m9_w5.py`
Import `m8_w5.py` and the modules it uses, and edit none of them. The world is `World5(world_seed, period = 128)`. The cortex, stores (disjoint
exam and replay reservoirs, as in M7), sleep decisions (one replay glance and one trace-learning Adam step per decision, with the live state
and traces saved and restored), movement, T14 and the final W14 are all as in M8. L = 2,048.

Controllers (choices from `default_rng(10_000 + world_seed)`; every one uses `prog_age_s`'s room choice: per-room progress reward, EPS 0.1,
BETA 0.3, ageing with τ = 256 and v_max as in M8):
- `age_nosleep`: `prog_age_s` with no sleep at all.
- `age_fixed20`: `prog_age_s` exactly as in M8 (fixed sleep share 0.20, placed as in M7b). This is the reference.
- `age_sleeparm`: the ageing bandit over 19 arms, the 18 rooms plus sleep. A sleep visit is 8 sleep decisions with no travel. Its reward is the
  whole exam-store progress over all channels (Σ over every exam glance of bits before − bits after) divided by 8. A room visit's reward is the
  per-room progress, as in M8. The sleep arm ages like a room: its age is the decisions since the last sleep visit. Sleep goes last in the optimistic first pass. There is no forced sleep.
- `age_fixed_matched`: `prog_age_s` with forced sleep visits, evenly spaced (M7b's placement rule), at the share `age_sleeparm` actually slept on the
  same seed, rounded to whole sleep visits. Run it after `age_sleeparm` for that seed.

## Seeds
**World seeds 24–35** (12 fresh seeds).

## Kernels
One private CPU kernel per seed, `ezzzzzx/m9-w5-s24` … `-s35`. Run `age_sleeparm` first, then the other three with a pool of up to 3. Each has
`"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. At most 90 minutes per kernel. The code goes up as private dataset `ezzzzzx/pdp-code-m9w5`.
Kaggle runs at most 5 CPU sessions at once; queue the rest.

## Smoke test (VPS CPU, before any kernel)
1. `age_fixed20` reproduces M8's `prog_age_s` T14 for seed 12 at L = 256 to 1e-6.
2. **Stores and sleep:** after a 256-decision `age_sleeparm` life on seed 24, no glance is in both stores, sleep drew 0 exam glances, and the live
   state and traces are bit-identical before and after a sleep visit.
3. For `age_sleeparm` on seed 24 at L = 512, print the visit trace (arm, decisions, reward, aged value of sleep and of the best room) and the
   sleep share by quarter.
4. All four controllers give a finite T14 on seed 24 at L = 128. Time a 512-decision `age_sleeparm` life and estimate a kernel's wall time.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/m9w5/RESULTS.md` and `results/m9w5/results.json`, with numbers only and no verdict:
- T14 and the final W14 per controller and seed, with mean ± std (ddof = 0), and T14 per checkpoint;
- the gaps per seed: age_fixed20 − age_nosleep, age_sleeparm − age_nosleep, age_sleeparm − age_fixed_matched and age_sleeparm − age_fixed20, with the
  counts of seeds where the left term is lower;
- the same gaps leaving out the first checkpoint (decision 256);
- for `age_sleeparm`: the sleep share overall and by quarter, sleep's position relative to swaps (the mean decisions since the last swap at sleep onsets
  vs at room-visit onsets), and the mean sleep reward vs room reward by quarter;
- the share of replay draws whose glance came from a rule that had already been swapped out at draw time, per controller that sleeps;
- swap-finding latency as in M8; shares of decisions per room, in corridors and in sleep;
- M8's numbers for reference; the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
1. **Sleep pays in a drifting world:** mean over seeds of (age_fixed20 − age_nosleep) ≤ −0.05 on T14, with age_fixed20 lower on at least 9 of 12 seeds.
2. **Choosing when matters here:** mean over seeds of (age_sleeparm − age_fixed_matched) ≤ −0.05 on T14, with age_sleeparm lower on at least 9 of 12 seeds.
Report only: age_sleeparm − age_fixed20, the gaps without the first checkpoint, the sleep share and its timing relative to swaps, and the stale-replay share.
Kill rules: if bar 1 fails, replaying a store of past rules does not pay its decisions when rules drift, and sleep needs a store that forgets
what is gone. If bar 1 passes and bar 2 fails, then even in a drifting world the dose is what matters, not the timing, and organic sleep
timing is closed until a world makes rest costly at some times and cheap at others.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `m9_w5.py`, `kaggle/m9w5/**` and `results/m9w5/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `m8_w5.py`, `w5_world.py`, `w5_headroom.py`, `w4_world.py`, `w4_headroom.py`, `m7_w3.py`, `m7b_w3.py`,
  `curio.py`, `h20_world.py`, other experiment code, other `results/` or `kaggle/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
