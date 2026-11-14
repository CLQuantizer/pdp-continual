# W4: a world sized to the bank cortex. Headroom check, with sleep, before any curious controller (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then the Verdicts in `results/m7cw3/RESULTS.md`, `results/m7bw3/RESULTS.md` and
`results/w3/RESULTS.md`, and the code in `m7_w3.py`, `m7b_w3.py`, `w3_world.py`, `w3_headroom.py`, `curio.py` and `h20_world.py`.

## Context
W3 had headroom for curiosity with a weak learner (M5b). With the trace-learned bank plus a fixed sleep dose, every
allocation learns W3's seven rooms equally well, and the learnability oracle beats random dwelling by only 0.02 (M7c).
Headroom belongs to a world and a learner together. W4 is sized to the bank. It has more learnable rooms than the life
can cover, rooms that share an alphabet but follow different rules (so a shared cortex must keep them apart), more noise,
and longer travel. As in every world check, headroom is measured before any controller runs. Principle served: act
to maximise compression progress (4), by checking that the world can reward it with this cortex.

## What to build
1. **`w4_world.py`** (import `curio.py` and `h20_world.py`; edit neither). `World4(world_seed)`:
   - **18 rooms on a ring:**
     - 14 learnable rooms: the 7 H20 processes from two independent `h20_world.World` instances, seeded
       `world_seed` and `world_seed + 1,000`. Their names are `<proc>a` and `<proc>b`, for example `C1a` and `C1b`.
       The two instances have different motifs, grammar words and rule draws, over the same 41 symbols.
     - 3 noise rooms `N1`, `N2` and `N3`, each uniform over the 41 symbols from its own rng
       (`default_rng(world_seed + 60_000 + k)`).
     - 1 dark room `Z`.
   - Order on the ring: `default_rng(world_seed + 70_000).permutation(18)`. Glances are 64 symbols, made with `curio.glance` on the right
     instance (noise rooms use their own rng). A room's stream resumes where it left off.
   - Actions, moves, corridor glances and the start in ring position 0 are all as in `w3_world.py`.
2. **`w4_headroom.py`.** The cortex is M7's `BankCortex` (roots init, trace learning, Adam lr 1e-3), with the channel embedding
   widened to 19 rows (18 rooms plus the corridor), initialised to zeros. Sleep, the stores, the fixed sleep plan and
   W's evaluation are as in `m7b_w3.py`. Generalise any hard-coded 9-room logic by subclassing or wrapping, without
   editing M7's files. Every policy sleeps at the **fixed share 0.20**, placed as in M7b.
   Policies (rng `default_rng(1_000 + world_seed)`):
   - `rand_dwell_s`: target uniform over the 18 rooms, shortest path there, dwell 8 decisions, then repeat.
   - `oracle14_s`: target uniform over the 14 learnable rooms.
   - `oracle7a_s`: target uniform over the 7 rooms of instance `a` only. It knows half the learnable world and gives that
     half all its time. This tests whether, for this cortex, depth beats breadth.
   Score **W14**: the frozen final bank on a fresh continuation (EVAL = 2,048 symbols) of each of the 14 learnable rooms,
   from a copy of the world, starting from a zero state with that room's channel embedding. Report the mean, and
   separately the mean over the `a` and the `b` rooms. Also report forgetting as in M7.

## Seeds and lives
World seeds 0–5. L = 2,048 for every policy. Also L = 8,192 for `oracle14_s`, for the scarcity check.

## Kernels
One private CPU kernel per seed, `ezzzzzx/w4-s0` … `-s5`. Each has `"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. Use a pool of up to 4.
At most 90 minutes per kernel; if the L = 8,192 life does not fit, put it in its own kernel per seed. The code goes up as private dataset
`ezzzzzx/pdp-code-w4`. Kaggle runs at most 5 CPU sessions at once; queue the rest.

## Smoke test (VPS CPU, before any kernel)
1. **The cortex is M7's:** with W4's code restricted to W3's 9 rooms (one instance, one noise room, no `b` rooms), `rand_dwell_s` reproduces M7b's
   W for seed 36 at L = 64 to 1e-6. If restricting the world is awkward, instead check that a `BankCortex` with a 19-row zero
   channel embedding gives the same bits as M7's on the same 64-glance sequence (channel ids < 10), to 1e-6.
2. **Instances differ:** for world seed 0, print the first 64 symbols of `M7a` and `M7b`, and of `Ga` and `Gb`. They must differ.
3. **Ring:** print the room order for seeds 0–2.
4. All three policies give a finite W14 on seed 0 at L = 64. Time a 512-decision `rand_dwell_s` life and estimate the
   kernels' wall times.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/w4/RESULTS.md` and `results/w4/results.json`, with numbers only and no verdict:
- W14 (and its `a` and `b` halves) per policy, seed and L, with mean ± std (ddof = 0), and per-room held-out bits;
- the gaps per seed: oracle14_s − rand_dwell_s, oracle7a_s − rand_dwell_s and W14(oracle14_s, 2,048) −
  W14(oracle14_s, 8,192);
- shares of decisions per room, in corridors and in sleep, per policy; forgetting per policy;
- M7c's W3 numbers, for reference;
- the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
1. **Headroom with this cortex:** min(oracle14_s, oracle7a_s) − rand_dwell_s ≤ −0.15 on W14, on at least 5 of 6 seeds.
2. **Time stays scarce:** W14(oracle14_s, L = 2,048) − W14(oracle14_s, L = 8,192) ≥ 0.20 on all 6 seeds.
Report only: which oracle wins (breadth or depth) and the `a`/`b` split.
Kill rules: if bar 1 fails, more rooms and noise still leave no allocation headroom for a bank with sleep. The next
world then needs sources whose learnability changes over time (drift), so that where to look keeps mattering. If bar 2 fails, the
life is too long for this world.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `w4_world.py`, `w4_headroom.py`, `kaggle/w4/**` and `results/w4/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `m7_w3.py`, `m7b_w3.py`, `m7c_w3.py`, `w3_world.py`, `w3_headroom.py`, `curio.py`,
  `h20_world.py`, other experiment code, other `results/` or `kaggle/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
