# Handoff H24f: fine-grained surprise store, recalls facts, Pareto reporting (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, results/h24e/RESULTS.md (with Verdict) and `h24e_priced_store.py`. Create only `h24f_fine_store.py` and
`results/h24f/*` (import from earlier scripts; do not edit them). Do not commit; do not touch docs/loop.md or audit.md.
CPU torch in `.venv`, `torch.set_num_threads(2)`.

Same data, GRU, replay schedule (2 × batch-8 per new chunk), prequential rules and seeds as H24e (h24b_world, slow,
N = 131072, seeds 6 7 8). Reuse H24e's gru and full numbers; do not rerun them.

## Changes
1. **Match order 5** (hash of the last 5 symbols; extend backwards up to 32; same confidence buckets with the lowest
   bucket 5–7). Run `full5` = H24e's full store with order-5 match, as the new reference.
2. **Symbol-level surprise writes** `fine`: after a chunk is scored, mark every symbol whose prequential bits (under the
   final mixture) exceed 4.0; store the window [i − 16, i + 4] around each marked symbol (merge overlapping windows).
   Replay samples 64-symbol segments only from the stored windows (pad short windows by sampling several), and the match
   model only matches inside stored windows. Run `fine3`: the same with threshold 3.0.
3. **Memory footprint** M = stored symbols × log2(42) + 32 bits per stored window; report peak M / N alongside P.
   No T.

## Report: results/h24f/RESULTS.md (numbers only) + results.json
Per condition and seed: P, peak M/N, fraction of symbols stored; per-process P; F value-digit bits by the key's prior
appearances (0, 1, 2–4, 5+). A Pareto table: P vs peak M/N for gru (M = 0), full (H24e), full5, fine, fine3. Wall time.
Smoke test N = 8192. Budget 100 min wall: if over, drop `fine3` first and say so.

## Pre-registered bars (orchestrator scores; do not change)
1. **Facts recalled:** full5's F value digits at ≥ 1 prior appearance average ≤ 1.0 bits/digit (order 5 fixes recall).
2. **M3 (Pareto) PASS:** the better of fine / fine3 has P ≤ P(full5) + 0.10 on the mean with peak M/N ≤ 0.15 × full5's.
3. That condition's F value digits at ≥ 1 prior appearance ≤ 1.5 bits/digit.
