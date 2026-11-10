# Handoff H24g: keep what pays (utility-based store under a memory budget) (worker: grok)

Read /root/pdp/CLAUDE.md, results/h24f/RESULTS.md (with Verdict), and `h24f_fine_store.py` (including grok's
touching-window fix). Create only `h24g_utility_store.py` and `results/h24g/*` (import from earlier scripts; do not
edit them). Do not commit; do not touch docs/loop.md or audit.md. `.venv/bin/python`, CPU torch, `torch.set_num_threads(2)`.
**Run experiments in the foreground, one at a time** (two parallel jobs on 2 cores were ~9× slower in H24f).

Same data, GRU, replay schedule, order-5 match, mixture, prequential rules and seeds as H24f (h24b_world, slow,
N = 131072, seeds 6 7 8). Reuse H24f's gru / full5 / fine numbers; do not rerun them.

## Condition `util` (memory budget B = 0.15 × full5's peak M, i.e. M/N ≤ 0.884)
- Writes as `fine` (symbols with mixture bits > 4.0 → window [i−16, i+4], merged; keep the touching-window fix).
- Each stored window w has a utility u_w, initialised to 1.0 bit, decayed by ×0.999 per new chunk, and credited:
  (a) **match credit**: when the match model's prediction comes from inside w, add (bits of the symbol under the GRU alone
  − bits under the mixture), floored at 0; (b) **replay credit**: when a replay batch includes a segment from w, add
  max(0, bits of the next 64-symbol chunk before − after that replay step) / (number of segments in the batch).
  Document exactly how (b) is measured; if it costs more than 30% extra wall time, drop (b) and say so.
- Whenever M exceeds B, evict the lowest-utility windows until M ≤ B. Newly written windows are protected for 32 chunks.
Run `util_m` too: same with match credit only.
Report: P, peak M/N, final windows, per-process P, F value-digit bits by prior appearances, the fraction of evicted
windows by process (true process ids, for analysis only), and wall time. Smoke test N = 8192, one seed. Budget 100 min
wall: if over, drop `util_m` first.

## Report: results/h24g/RESULTS.md (numbers only, no verdict) + results.json, with the Pareto table extended.

## Pre-registered bars (orchestrator scores; do not change)
1. **M3 (Pareto) PASS:** the better of util / util_m has P ≤ P(full5) + 0.10 (≤ 1.074) with peak M/N ≤ 0.884 on every seed.
2. Its F value digits at ≥ 1 prior appearance ≤ 1.5 bits/digit.
3. Report only: which processes' windows are evicted most (prediction: noise-like content: CA flips, grammar choices).
