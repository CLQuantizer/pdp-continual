# W2-1: the locked-rooms world with a way back. Headroom check before any curiosity bar (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/w2_headroom/RESULTS.md` and `results/m4w2d/RESULTS.md` (both
including their Verdicts), `w2_world.py` and `w2_headroom.py`.

## Context
In W2, room 8 has no exit. M4-W2d showed what that does: once reset is never pressed at random, every agent ends up
trapped in room 8, and end-of-life knowledge measures forgetting, not exploration. W2-1 gives room 8 a way back: a
3-digit key typed in room 8 returns the agent to room 0. The world becomes a cycle of 9 rooms.

Before any curious controller runs here, we check that W2-1 has headroom: some allocation of time has to know the
world clearly better than a random walk. Otherwise there is nothing for curiosity to win. Principles served: act to
maximise compression progress (4), by checking that the world can reward it.

## What to build
1. **`w21_world.py`.** Import `w2_world.py` and do not edit it. Define `World21`, a subclass of `World`, that is
   identical except for room 8:
   - **Key for room 8:** `key_8 = default_rng(world_seed + 50_000).integers(0, 6, size=3)`. This is a separate rng, so
     every table drawn by `World` is unchanged.
   - **Room 8 lock state s8** follows the same rule as the lock rooms. On manipulation a: if a == key_8[s8], s8 += 1.
     Otherwise s8 = 1 if a == key_8[0], else 0.
   - **Room 8 output and machine are unchanged:** y = out[q, a], and q = nxt[q, a].
   - **When s8 reaches 3:** after emitting y, set room = 0, s = 0, q = 0 and s8 = 0.
   - Reset (action 7) also sets s8 = 0. The TV is unchanged.
   - Rollouts and policies: `rand6`, `rand8` and `oracle` as in `w2_world.py`. Add **`oracle_deep`**: the same block
     schedule as `oracle` (block 512, reset, then navigate with keys, then stay), but the target room for block b is
     8 when b is even and (b // 2) mod 8 when b is odd. Room 8 is the hardest room (a hidden 6-state machine), and this
     oracle gives it half the time. It uses the same stay rule as `oracle` (in lock rooms with s == 2, never press key[2]).
     In room 8 it must also never complete key_8; drop key_8[2] when s8 == 2.
2. **`w21_headroom.py`.** A copy of `w2_headroom.py` in which the world is `World21` and the policies are `rand6`,
   `rand8`, `oracle` and `oracle_deep`. Everything else is unchanged: the learner (GRU, the same sizes, lr and seed),
   tokens, chunks, replay, T = 60,000, block 512, the probe episodes and the probe schedule. Probes use `World21`. The
   room-8 probe uses the same stay rule as `oracle_deep` (drop key_8[2] when s8 == 2), so it never leaves room 8. That
   makes its action sequence differ slightly from W2-0's room-8 probe; report how many probe actions differ.

## Seeds and kernels
- World seeds 0, 1 and 2, as in W2-0, so the numbers can be compared. This is a world check, not a scored agent run.
- Use three private CPU kernels, `ezzzzzx/w21-s0`, `-s1` and `-s2`, one world seed each, all four policies. Each has
  `"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. The code goes up as private dataset
  `ezzzzzx/pdp-code-w21`.

## Smoke test (VPS CPU, before any kernel)
1. **Equivalence:** for world seed 0, `rand6` in `World21` and in `World` give identical tokens up to the step where
   the agent first types key_8 in room 8. Print that step, and check that the token sha256 of the prefix matches.
   Before that step the two worlds must agree exactly.
2. **Exit check:** from a fresh `World21(0)`, play keys 0–7, then key_8. Print the rooms after each key and check that
   the last one returns the agent to room 0 with s = q = s8 = 0.
3. **oracle_deep check:** over 4 blocks at block 512, print the target room per block and the time in each room.
   Room 8 must get blocks 0 and 2, and key_8 must never complete.
4. With T = 2,048, all four policies give finite bits and probes.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/w21/RESULTS.md` and `results/w21/results.json`, with numbers only and no verdict. Use the same fields as
`results/w2_headroom/`, plus:
- PROBE and PROBE_cap (the mean over rooms of min(probe_k, 4)) per policy and seed, with the per-room capped probe at T;
- for rand6: the number of full cycles (room 8 back to room 0) and the time in each room;
- the gaps per seed on PROBE_cap: rand6 − oracle, rand6 − oracle_deep and rand8 − rand6;
- the W2-0 numbers for rand6 and oracle on the same seeds, next to these, for comparison;
- wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
1. **The way back works:** on all 3 seeds, rand6 completes at least 3 full cycles in T = 60,000.
2. **Headroom for curiosity:** on at least 2 of 3 seeds, min(PROBE_cap of oracle, PROBE_cap of oracle_deep) is lower
   than rand6's PROBE_cap by ≥ 0.5.
Kill rule: if bar 2 fails, a random walk on a cycle already spreads its time well enough, and W2-1 cannot test
curiosity. The next world then needs rooms that differ in how learnable they are, so that where the agent spends its
time matters.

## Budget and rules
- CPU only. At most 90 minutes of kernel wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `w21_world.py`, `w21_headroom.py`, `kaggle/w21/**` and `results/w21/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `w2_world.py`, `w2_headroom.py`, any `m4w2*` file, other `kaggle/` folders,
  other `results/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
