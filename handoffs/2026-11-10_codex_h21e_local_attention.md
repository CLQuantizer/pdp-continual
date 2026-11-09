# Handoff H21e: local-attention cortex vs GRU, under batched replay (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, results/h21c/RESULTS.md and results/h21d/RESULTS.md (both Verdict sections), and `h21c_replay_dose.py`.
Create only `h21e_local_attention.py` and `results/h21e/*` (import from earlier scripts; do not edit them). Do not commit;
do not touch docs/loop.md or audit.md. CPU torch in `.venv`, `torch.set_num_threads(2)`.

Data and rules as H21c: slow world, N = 131072, seeds 6 7 8; match model and fixed-share mixture (α = 0.001) as H21b.

## Learner `att_rb8x2` (and `att_rb8x2+match`)
- Causal transformer: symbol embedding d = 64 + learned absolute position embedding over a 64-symbol window; 2 layers,
  4 heads, MLP 4×, pre-LayerNorm, linear readout to 41. Parameter count printed.
- Online, prequential: process the stream in consecutive 64-symbol chunks. To predict chunk k, feed the window
  [previous 64 symbols | chunk k] (128 positions, causal mask, attention limited to the previous 64 positions per query)
  and record the bits of chunk k's 64 symbols *before* any update. Then take one Adam step (lr 1e-3, clip 1.0) on chunk
  k's loss, followed by 2 batched replay steps, each on a batch of 8 past chunks drawn uniformly (each with its own
  preceding 64 symbols as context, loss only on the chunk's 64 targets).
- Same replay sampling as H21c (uniform start over the already-scored past; never touches unscored symbols).
Reuse H21c's gru_rb8x2(+match) numbers and lzma; do not rerun them.

## Report: results/h21e/RESULTS.md (numbers only) + results.json
Main table (mean ± std, per seed) next to gru_rb8x2(+match), KT and lzma; per-process bits; wall seconds per stream;
parameter count. Smoke test N = 8192, one seed. Budget 90 min wall: if the projection exceeds it, use the 1-layer variant
(say so) rather than dropping seeds.

## Pre-registered bars (orchestrator scores; do not change)
1. **M1 PASS (slow):** att_rb8x2+match beats lzma on all 3 seeds, and its C1 and C7 bits ≤ 0.5× the KT mixture's.
2. **CA cracked:** att_rb8x2's E30 and E110 bits are each ≤ 0.6 (vs GRU ≈ 1.2, KT ≈ 1.2, generator entropy ≈ 0.04).
3. **Substrate:** att_rb8x2+match ≤ gru_rb8x2+match − 0.03 on the mean.
