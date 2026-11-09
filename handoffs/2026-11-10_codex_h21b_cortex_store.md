# Handoff H21b: cortex + store on the passive H20 world (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, docs/roadmap.md (M1), results/h21a/RESULTS.md (including the Verdict section) and `h21a_substrate.py`.
Create only `h21b_cortex_store.py` and `results/h21b/*` (import and reuse functions from h21a_substrate.py; do not edit
it). Do not commit; do not touch docs/loop.md or audit.md. CPU torch is in `.venv`; `torch.set_num_threads(2)`.

Same data as H21a: `h20_world.stream`, worlds slow and fast, N = 131072, seeds 6 7 8. Same GRU (embedding 32,
hidden 256, chunk 64, Adam lr 3e-3, clip 1.0, state carried and detached across chunks).

## Learners (all prequential: every symbol's probability is recorded before any update that uses that symbol)
1. `gru`: exactly H21a's GRU (rerun it, because the mixtures below need its per-symbol distributions).
2. `gru_r2`: after each new chunk's single Adam step, take 2 more Adam steps, each on a chunk of 64 consecutive symbols
   drawn uniformly from the already-observed history (start position uniform over the past, zero initial state for
   replayed chunks). Replay never touches symbols not yet scored.
3. `match`: a match model (exact recall). At time t, find the most recent earlier position whose preceding 8 symbols
   equal the last 8 symbols (hash table from the 8-gram to its last end position; update after scoring t). If found,
   extend the match length backwards up to 32. Predict the symbol that followed that occurrence with probability
   c[L] (L = match length bucketed into 8–11, 12–15, 16–23, 24–32), and spread the remaining (1 − c[L]) uniformly over
   the other 40 symbols. If no match, uniform over 41. c[L] = (hits[L] + 1) / (trials[L] + 2), updated after scoring.
4. `gru+match` and `gru_r2+match`: a fixed-share Bayesian mixture of the two predictive distributions, weights
   initialised to ½, updated after each symbol by the likelihood of the observed symbol, then
   w = (1 − 0.001)·w + 0.001/2.

## Report: results/h21b/RESULTS.md (numbers only, no verdict) + results.json
- Main table: rows = gru, gru_r2, match, gru+match, gru_r2+match, KT mixture, lzma -9e; columns slow / fast, mean ± std.
- Per-seed table of the same.
- Per-process bits (slow, mean over seeds with observations) for every row; also the fraction of symbols where the
  match model had a match, and its hit rate.
- Wall time per stream per learner. Smoke test: N = 8192, one seed. Budget 90 min wall. If over: drop the fast world for
  gru_r2 (and its mixture) first, and say so.

## Pre-registered bars (orchestrator scores; do not change)
1. **M1 PASS (with store):** the better of gru+match / gru_r2+match beats lzma in both worlds on every seed, AND its C1 and
   C7 bits are each ≤ 0.5 × the KT mixture's (slow world, mean).
2. **Replay helps the cortex:** gru_r2 ≤ gru − 0.05 bits on the mean in both worlds.
3. **Store helps the cortex:** gru+match ≤ min(gru, match) − 0.05 on the mean in both worlds.
4. Report only: CA (E30, E110) bits for every row against the KT mixture's.
