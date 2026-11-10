# Handoff M4a: every part in one life (worker: grok)

Read /root/pdp/CLAUDE.md, docs/roadmap.md (Status, M4), results/h24g/RESULTS.md, results/h25b/RESULTS.md and
results/h25d/RESULTS.md (Verdict sections), and the scripts `h24b_world.py`, `h24g_utility_store.py`,
`h25b_progress_fix.py`, `h25d_visit_slope.py`, `curio.py`. Create only `m4a_one_life.py` and `results/m4a/*` (import from
earlier scripts; do not edit them). Do not commit; do not touch docs/loop.md or audit.md. `.venv/bin/python`, CPU torch,
`torch.set_num_threads(2)`. **Run experiments in the foreground, one at a time.**

## World
Active, as curio/H25: channels = the 8 h24b processes (G C1 C7 E30 E110 M7 M37 F) + N (uniform noise over the 42
symbols) + Z (constant). Glance = 64 symbols of the chosen channel (processes resume where they left off). T = 512
glances, seeds 6 7 8.

## Machine (identical in every condition except the controller, and the store in `rand_nostore`)
- Cortex: GRU (embedding 32 + channel embedding 8, hidden 256, Adam lr 3e-3, clip 1.0), per-channel recurrent state,
  one Adam step per glance, then 2 batched replay steps (batch 8 segments of 64) sampled **from the store only**.
- Store: H24g's `util_m` rule (fine writes: mixture bits > 4.0 → window [i−16, i+4], merged with the touching-window fix;
  utility = match credit, decay 0.999 per glance, 32-glance protection; evict lowest utility whenever
  M > 0.15 × (observed symbols × log2 42)). Order-5 match model inside stored windows, keyed per channel; fixed-share
  mixture with the GRU (α = 0.001).
- Prequential: every symbol's probability is recorded before any update that uses it.

## Conditions
`rand`, `rand8` (oracle: uniform over the 8 learnable channels), `gmc_prop` (H25b's estimator and allocation),
`slope_prop` (H25d's), and `rand_nostore` (rand with no store: no replay, no match; the GRU alone).

## Score
W = the final frozen system (GRU + match model over the final store, mixture weights frozen) on a fresh 2048-symbol
continuation of each of the 8 learnable channels from a deep copy of the world, mean bits/symbol; also W_F (channel F
alone). Report per condition and seed: W, W_F, per-channel held-out bits, glance shares overall and by quarter, peak
memory / observed symbols, wall time. Smoke test T = 32, one seed. Budget 120 min wall: if over, drop `slope_prop` first,
and say so.

## Pre-registered bars (orchestrator scores; do not change)
1. **M4 PASS:** the better of gmc_prop / slope_prop (on the mean) has W ≤ W(rand) − 0.05 on all 3 seeds, with ≤ 15% of
   glances on N and ≤ 15% on Z.
2. **Store earns its place in the active life:** W(rand) ≤ W(rand_nostore) − 0.05 on the mean, and W_F(rand) ≤
   W_F(rand_nostore) − 0.5.
3. Report only: W relative to rand8 (the oracle).
