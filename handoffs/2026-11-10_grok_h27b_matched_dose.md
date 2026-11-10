# Handoff H27b: bounded memory at a matched replay dose (worker: grok, Kaggle)

Read /root/pdp/CLAUDE.md, handoffs/2026-11-10_grok_h27_bounded_replay.md, results/h27/RESULTS.md (Verdict) and
`h27_bounded_replay.py`. Create only `h27b_matched_dose.py`, `kaggle/h27b/*`, `results/h27b/*`; import, do not edit
existing scripts. Do not commit; no docs/loop.md or audit.md.

## Why
H27: a 77-segment buffer replayed at 2 steps per glance (each segment ~100 repeats) is worse than no replay; full
history at 1 step is best. Question: with repeats per stored symbol held low, does a bounded buffer help, and where
is the knee of the dose curve?

## Setup
Identical to H27 (controller rand, no utility store, no match, T = 512, scoring as H27) on **fresh seeds 21–26**.
"Every k" = one replay step (batch 8 × 64) after every k-th glance; traffic = replayed / observed symbols.
| name | source | dose | traffic |
|---|---|---|---:|
| `nostore` | none | — | 0 |
| `hist_e1` | whole history | every 1 (= H27 hist1) | 8 |
| `hist_e2` | whole history | every 2 | 4 |
| `hist_e4` | whole history | every 4 | 2 |
| `res77_e4` | uniform reservoir, 77 segments | every 4 | 2 |
| `res77_e8` | same | every 8 | 1 |
| `res308_e1` | uniform reservoir, 308 segments (60% of history) | every 1 | 8 |
| `res308_e4` | same | every 4 | 2 |
Report, per condition and seed: W, per-channel bits, M (bits held for replay), traffic, and **mean replays per stored
segment over the life**. Smoke T = 32 on Kaggle first. Private CPU kernels, dataset `ezzzzzx/pdp-code-h27b`. Budget 90 min.

## Pre-registered bars (orchestrator scores; do not change)
1. **Bounded memory can help:** min over res* of W ≤ W(nostore) − 0.05 on the mean.
2. **Bounded ≈ full at equal traffic:** W(res308_e4) ≤ W(hist_e4) + 0.05 on the mean.
3. Report only: the dose curve hist_e1/e2/e4 and the P / M / traffic table.

## Output
Final message: means table (W, M, traffic, replays per segment) and per-seed W, kernel ids.
