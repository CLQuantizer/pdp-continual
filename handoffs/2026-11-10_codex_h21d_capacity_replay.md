# Handoff H21d: more batched replay and a wider cortex (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, results/h21c/RESULTS.md (with Verdict) and `h21c_replay_dose.py`. Create only `h21d_capacity_replay.py`
and `results/h21d/*` (import from the earlier scripts; do not edit them). Do not commit; do not touch docs/loop.md or
audit.md. CPU torch in `.venv`, `torch.set_num_threads(2)`.

Data and rules exactly as H21c: slow world, N = 131072, seeds 6 7 8; match model and fixed-share mixture as H21b;
prequential scoring before any update that uses a symbol; replayed chunks start from a zero state.

## Learners (each also with `+match`)
1. `gru_rb8x4`: hidden 256, four batched replay steps (batch of 8 past 64-symbol chunks) per new chunk.
2. `gru_rb16x2`: hidden 256, two batched replay steps with batch 16.
3. `gru512_rb8x2`: hidden 512, two batched replay steps with batch 8.
Reuse H21c's numbers for gru_rb8x2(+match) and lzma; do not rerun them.

## Report: results/h21d/RESULTS.md (numbers only) + results.json
Main table (mean ± std, per seed) next to gru_rb8x2(+match), KT and lzma; per-process bits; wall seconds per stream.
Smoke test N = 8192, one seed, to project time. Budget 90 min wall: if over, drop `gru_rb8x4` first, then `gru_rb16x2`,
and say so.

## Pre-registered bars (orchestrator scores; do not change)
1. **M1 PASS (slow):** the best `<x>+match` beats lzma on all 3 seeds, and its C1 and C7 bits ≤ 0.5× the KT mixture's.
2. **Lever:** the best new row beats gru_rb8x2+match by ≥ 0.03 bits on the mean.
3. Report only: E30 / E110 bits for every row (does capacity or replay start to crack the CA rule?).
