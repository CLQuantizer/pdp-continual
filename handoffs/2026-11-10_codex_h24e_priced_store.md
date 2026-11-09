# Handoff H24e: does a surprise-gated store pay for itself in bits? (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, docs/h20_world.md (two-part T), results/h24b_floors/RESULTS.md, results/h24d/RESULTS.md (Verdict),
results/h21c/RESULTS.md (Verdict), and `h24b_world.py`, `h21c_replay_dose.py`. Create only `h24e_priced_store.py` and
`results/h24e/*` (import from earlier scripts; do not edit them). Do not commit; do not touch docs/loop.md or audit.md.
CPU torch in `.venv`, `torch.set_num_threads(2)`.

Data: `h24b_world` stream (with process F), **slow world**, N = 131072, seeds 6 7 8. Alphabet of 42 symbols (embedding
and readout sized accordingly). GRU as H21c (embedding 32, hidden 256, chunk 64, Adam lr 3e-3, clip 1.0, state carried and
detached). Prequential scoring before any update that uses a symbol. The match model and fixed-share mixture as H21b,
but **the match model may only look up positions held in the store**.

## Store and pricing
- The store holds 64-symbol chunks of the stream (with their start position). Replay (2 batched steps of 8 chunks per
  new chunk, as H21c's rb8x2) samples only from the store; the match model only matches inside stored chunks.
- Price: S(t) = (stored symbols) × log2(42) + 32 bits per stored chunk. Report P (prequential bits/symbol), peak S, and
  **T = P + peak S / N**.

## Conditions
1. `gru`: no store, no replay, no match (T = P; reuse nothing, rerun on this stream).
2. `full`: every chunk written (the H21c learner with match, now priced).
3. `surp`: write a chunk only if its prequential bits/symbol exceeded the running 70th percentile of all chunks so far
   (the chunk the system found hard); the first 8 chunks are always written.
4. `surp90`: as surp with the 90th percentile.

## Report: results/h24e/RESULTS.md (numbers only) + results.json
Per condition and seed: P, peak S/N, T; per-process P (G C1 C7 E30 E110 M7 M37 F); F value-digit bits split by the
key's prior appearances (0, 1, 2–4, 5+); fraction of chunks written. Wall time. Smoke test N = 8192, one seed.
Budget 100 min wall: if over, drop `surp90` first and say so.

## Pre-registered bars (orchestrator scores; do not change)
1. **M3 in bits, PASS:** the best of surp / surp90 has T ≤ T(gru) − 0.05 on the mean, i.e. the store pays for itself
   after being priced.
2. That condition's F value digits at ≥ 1 prior appearance average ≤ 1.5 bits/digit (vs KT ≈ 3.9 → 1.7).
3. Report only: T(full) (expected to be far worse: the whole history costs ~5.4 bits/symbol).
