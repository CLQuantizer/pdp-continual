# Handoff H21a: substrate ceiling on the passive H20 world (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, docs/roadmap.md (milestone M1), docs/h20_world.md. You implement and run; the orchestrator scores the
bars. Create only new files (h21a_substrate.py, results/h21a/*); do not edit existing files; do not commit; do not touch docs/loop.md or audit.md.

## Setup
- CPU PyTorch is already installed in `.venv` (no network inside the sandbox). `torch.set_num_threads(2)`.
- Data: `from h20_world import stream, ALPHA, PROCS, WORLDS`. Worlds slow and fast, **N = 131072 symbols** (2¹⁷),
  seeds **6 7 8**. Note the season lengths are unchanged (16384 / 2048).
- Floors at this N (compute them in the same script for the same streams): KT mixture via `kt_floors`, and lzma -9e
  (bits = 8 · compressed bytes / N), per world and seed.

## Learners (all strictly online and prequential: score −log2 p(x_t | x_<t) BEFORE any update that uses x_t)
1. `gru`: embedding 32 → GRU hidden 256 → linear 41. Truncated BPTT: process the stream in consecutive chunks of 64
   symbols, carrying the hidden state across chunks (detached). For each chunk, first record the prequential bits of
   all 64 symbols with the current weights, then take ONE Adam step (lr 3e-3, grad-clip 1.0) on that chunk's loss.
2. `gru_big`: same with hidden 512 (skip if (1) takes > 25 min per stream; say so).
3. `lin_fw`: a fast-weight / linear-attention layer with the delta rule (Schlag et al. 2021): key/query/value from the
   embedding + previous output, d = 64, state W_t = W_{t-1} + β_t (v_t − W_{t-1} k_t) k_tᵀ, sigmoid β, followed by a
   linear readout; outer weights trained the same way as (1). Keep it simple and document the exact equations.
   If it fails to train (bits above KT mixture after the first 20% of the stream), report that rather than tuning.

## Report: results/h21a/RESULTS.md (numbers only, no verdict) + results/h21a/results.json
- Table: rows = learner + floors, columns = slow / fast mean ± std over seeds, prequential bits/symbol over all N.
- Per-process bits (mean over seeds, slow world) for every learner and the KT mixture: G C1 C7 E30 E110 M7 M37.
- Learning curve: bits in each eighth of the stream, per learner (slow world, mean over seeds).
- Wall time per stream per learner. Times UTC+8; no hostnames or paths.
- Smoke test first with N = 8192, one seed, to estimate runtime; total budget 90 min of wall time. Drop `gru_big`
  first, then the fast world, if over budget, and say so.

## Pre-registered bars (orchestrator scores these; do not change)
- **M1 PASS:** the best learner beats lzma at this N in both worlds on every seed, AND its counter bits (C1, C7) and CA
  bits (E30, E110) are each ≤ 0.5 × the KT mixture's on the same process.
- **Partial:** beats the KT mixture overall but not lzma → the world is fine and the learner needs scale/time.
- **Pivot:** cannot beat the KT mixture → shrink the problem (smaller alphabet / shorter records) before any curiosity
  work on this world.

Final message: the main table, wall times, and anything that looked wrong.
