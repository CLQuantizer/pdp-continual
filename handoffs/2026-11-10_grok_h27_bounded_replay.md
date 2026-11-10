# Handoff H27: does replay need the whole history, or a fair sample of it? (worker: grok, Kaggle)

Read /root/pdp/CLAUDE.md (incl. "Kaggle compute"), handoffs/2026-11-10_grok_m4c_kaggle.md, results/m4c/RESULTS.md
(Verdict), docs/compute-vs-data-movement.md, and `m4b_ablation.py`, `m4c_score.py`. Create only `h27_bounded_replay.py`,
`kaggle/h27/*` and `results/h27/*`; import, do not edit existing scripts. Do not commit; no docs/loop.md or audit.md.

## Why
M4c: replay from the whole raw history is the machine's biggest lever, but that log is unbounded memory. The question
(Schmidhuber's "store everything" vs a fixed memory budget): is it the *size* of the history or the *fairness* of the
sample that matters?

## Setup
World, GRU, live step and scoring exactly as M4c `hist_rep` (controller rand, no utility store, no match), T = 512,
**fresh seeds 15 16 17 18 19 20**. Only the replay source and dose change. A "segment" = 64 consecutive symbols of one
channel's observed stream, replayed from a zero state with that channel's embedding, as in M4b.
| name | replay source | steps per glance (batch 8) |
|---|---|---|
| `nostore` | none | 0 |
| `hist1` / `hist2` / `hist4` | whole history (= M4c hist_rep at 2) | 1 / 2 / 4 |
| `res77` | uniform reservoir (Algorithm R over glance-segments) with capacity 77 segments (= 15% of the final history) | 2 |
| `res16` | same, capacity 16 (≈ 3%) | 2 |
| `fifo77` | the most recent 77 segments | 2 |
Reservoir and FIFO contain whole 64-symbol glances; each replay segment is drawn uniformly from the current contents.

## Report (numbers only, no verdict): results/h27/RESULTS.md + results.json
Per condition and seed: W (mean held-out bits over the 8 learnable channels), per-channel bits, M = replay memory in
bits (segments × 64 × log2 42; for hist*, the whole log), **traffic = replayed symbols / observed symbols**, wall.
Smoke T = 32 one seed on Kaggle first. Private CPU kernels, dataset `ezzzzzx/pdp-code-h27` (scripts only, no secrets).
Budget 120 min.

## Pre-registered bars (orchestrator scores; do not change)
1. **A fair bounded sample suffices:** W(res77) ≤ W(hist2) + 0.03 on the mean.
2. **Fairness, not recency:** W(res77) ≤ W(fifo77) − 0.05 on the mean.
3. Report only: res16; the dose curve hist1/2/4 as bits saved vs nostore per unit traffic.

## Part 2 (papers; only R2 `pdp-papers` and D1 `pdp-papers-index`)
Register every reference in the References section of docs/compute-vs-data-movement.md in D1 (status `checked` only
if you confirm the citation; else `memory`). Fetch open-access copies from publisher/author/arXiv/PubMed Central only
(never shadow libraries) and sync with `python3 tools/sync_papers.py`; the rest get file_status `wanted`. No deletes.

## Output
Final message: means table and per-seed W, traffic and M per condition, kernel ids, the papers outcome.
