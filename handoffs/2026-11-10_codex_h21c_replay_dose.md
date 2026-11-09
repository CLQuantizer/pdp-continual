# Handoff H21c: replay dose-response, cortex + store (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, results/h21b/RESULTS.md (with Verdict) and `h21b_cortex_store.py`. Create only `h21c_replay_dose.py`
and `results/h21c/*` (import from h21a/h21b scripts; do not edit them). Do not commit; do not touch docs/loop.md or
audit.md. CPU torch in `.venv`, `torch.set_num_threads(2)`.

Data: `h20_world.stream`, **slow world only**, N = 131072, seeds 6 7 8. GRU, match model, fixed-share mixture and
prequential rules exactly as H21b.

## Learners
1. `gru_r4`: as H21b's gru_r2 but 4 sequential replay Adam steps per new chunk (each on one uniformly drawn past
   64-symbol chunk, zero initial state).
2. `gru_rb8`: after each new chunk's Adam step, ONE extra Adam step on a **batch of 8** past chunks (each 64 symbols,
   uniformly drawn, zero initial state), forward/backward as one batch (loss = mean over the batch).
3. `gru_rb8x2`: two such batched steps per new chunk.
4. Mixtures `<each>+match` (fixed-share, α = 0.001, as H21b).
Reuse H21b's numbers for gru, gru_r2, gru_r2+match and lzma (same seeds and streams); do not rerun them.

## Report: results/h21c/RESULTS.md (numbers only) + results.json
Main table (slow, mean ± std and per seed) for the new rows next to H21b's gru / gru_r2 / gru_r2+match / KT / lzma.
Per-process bits (slow) for every new row. Wall seconds per stream per learner. Smoke test N = 8192, one seed.
Budget 90 min wall: if over, drop `gru_r4` first and say so.

## Pre-registered bars (orchestrator scores; do not change)
1. **M1 PASS (slow):** the best `<x>+match` beats lzma on all 3 slow seeds, and its C1 and C7 bits ≤ 0.5× the KT mixture's.
2. **Dose-response:** the best new row beats gru_r2 by ≥ 0.05 bits (mean).
3. **Cheap replay:** gru_rb8 ≤ gru_r2 on the mean at ≤ 1.5× gru_r2's wall time.
4. Report only: G (grammar) bits for every row vs plain gru's 1.177 (does more uniform replay crowd it out further?).
