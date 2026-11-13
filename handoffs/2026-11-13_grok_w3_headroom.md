# W3: H23c's sources as rooms on a ring, with travel. Headroom check before any curious controller (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/h23c/RESULTS.md`, `results/w21/RESULTS.md` (both including their
Verdicts), `curio.py` and `h20_world.py`.

## Context
W2 and W2-1 had nothing for curiosity to win. With the trap in place, nobody learned well. Once the trap had an exit, a random walk learned the whole
world, because every room was learnable and the life was long enough for all of them. H23c's channel bandit had what
those worlds lacked: sources of graded learnability (easy motifs, a hard grammar, noise, a dark channel) and a life
too short to learn them all. There, compression progress beat even an oracle that knew which channels were worthless.
But that agent chose channels for free. W3 puts the same sources in rooms on a ring. To look at a source the agent has
to *be* there, and moving costs time. Where it goes then decides what it can learn. Before any controller runs, we
check that W3 has headroom: some allocation of time must clearly beat a random walk, and time must still be scarce
at the end of a life. Principle served: act to maximise compression progress (4), by checking that the world can
reward it.

## What to build
1. **`w3_world.py`.** Import `curio.py` and `h20_world.py`. Do not edit them. `World3(world_seed)`:
   - **Rooms:** the 9 channels of `curio.CH` (the 7 H20 processes, N and Z), placed on a ring of 9 rooms in the order
     `default_rng(world_seed + 70_000).permutation(9)`. The channel streams come from one `h20_world.World` and
     `curio.glance` exactly as in `curio.py`, so a room's channel resumes where it left off.
   - **Actions** at each decision: `stay` (observe one glance, B = 64 symbols, of the current room's channel), `left`
     or `right` (move one room). A move costs one decision and shows one glance of the corridor: 64 copies of the
     space symbol, labelled with channel "corridor". The compressor learns on the corridor glance like any other.
   - **Life:** L decisions. One decision is one glance, whether the agent stays or moves.
   - The agent starts in room 0 of the ring.
2. **`w3_headroom.py`.** Learner: `curio.SGDCompressor` with `CURIO_LR=0.004`, exactly as in H23c. It is a local
   hashed-context model with no backprop through time. Contexts are keyed by the channel of the room the agent is in,
   or "corridor", as `curio.py` does. It learns online on every glance it sees (score, then update).
   Policies (rng `default_rng(1_000 + world_seed)` for each):
   - `rand_walk`: each decision is uniform over {stay, left, right}.
   - `rand_dwell`: pick a target room uniformly from all 9, take the shortest path there, stay 8 decisions, repeat.
   - `oracle7`: like `rand_dwell`, but the target is uniform over the 7 learnable rooms only. It knows N and Z are
     worthless, like H23c's rand7.
   - `oracle_hard`: like `oracle7`, but the target is drawn with probability ∝ the H23c rand7 end-of-life held-out
     bits of that process (from `results/h23c/`; if those per-process numbers are not on disk, use the "Held-out bits
     vs rand" line in its RESULTS.md and report which numbers you used). It spends more time on what is hard and
     learnable.
   Score **W** exactly as H23c does: the mean bits/symbol of the final frozen compressor on a fresh continuation
   (EVAL symbols) of each of the 7 learnable channels, from a copy of the world at the end of life. Also report bits
   per channel, the share of decisions spent in each room and in corridors, and W at 4 checkpoints across the life.

## Seeds and lives
- World seeds 0, 1 and 2 (this is a world check, on selection seeds).
- Life L = 512 decisions (H23c's glance budget) for every policy. Also run `oracle7` with L = 2,048 for the
  scarcity check.

## Kernels
One private CPU kernel, `ezzzzzx/w3-headroom`, running all seeds and policies (use a pool of up to 4 processes). It
has `"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. The code goes up as private dataset
`ezzzzzx/pdp-code-w3`. If the smoke timing says it will take more than 90 minutes, split by seed into `-s0`, `-s1` and `-s2`.

## Smoke test (VPS CPU, before any kernel)
1. **Equivalence with H23c:** a policy that teleports (no travel cost) to the channel `curio.py`'s `rand` controller
   picks, for seed 6 with 64 glances, gives the same W as `curio.py --seeds 6 --glances 64 --conds rand --comp sgd`
   (with `CURIO_LR=0.004`) to 1e-9. This check only; W3 itself has travel.
2. **Ring check:** print the room order for seeds 0, 1 and 2, and a 20-decision trace of `rand_walk` (action, room,
   channel).
3. With L = 64, all four policies give a finite W on seed 0. Time it and estimate the kernel's wall time.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/w3/RESULTS.md` and `results/w3/results.json`, with numbers only and no verdict:
- W per policy and seed, mean ± std (ddof = 0), and the per-channel bits;
- room shares and corridor share per policy and seed;
- the gaps per seed: rand_walk − oracle7, rand_walk − oracle_hard, rand_dwell − oracle7, and
  W(oracle7, L = 512) − W(oracle7, L = 2,048);
- W at the 4 checkpoints;
- H23c's rand and rand7 W on its seeds, next to these, for reference;
- wall seconds, kernel ids and the smoke-test output.

## Pre-registered bars (orchestrator scores; do not change)
1. **Headroom over a random walk:** on all 3 seeds, min(W of oracle7, W of oracle_hard) is lower than rand_walk's W
   by ≥ 0.15 bits/symbol.
2. **Time stays scarce:** on all 3 seeds, W(oracle7, L = 512) − W(oracle7, L = 2,048) ≥ 0.20 bits/symbol. A life of
   512 has not learned everything that can be learned.
Report only: rand_dwell, and the gaps against it.
Kill rules: if bar 1 fails, travel makes every allocation alike and W3 cannot test curiosity, so the travel cost or
the world size is wrong. If bar 2 fails, the life is long enough to learn everything, and W3 repeats W2-1's failure.

## Budget and rules
- CPU only. At most 90 minutes of kernel wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `w3_world.py`, `w3_headroom.py`, `kaggle/w3/**` and `results/w3/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `curio.py`, `h20_world.py`, any `w2*` or `w21*` file, other `kaggle/` or
  `results/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
