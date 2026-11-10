# Handoff M4b: why does the store hurt the active life? (worker: grok)

Read /root/pdp/CLAUDE.md, handoffs/2026-11-10_grok_m4a_one_life.md (the M4a spec), results/m4a/RESULTS.md (incl. the
Verdict section) and `m4a_one_life.py`. Create only `m4b_ablation.py` and `results/m4b/*`; import from `m4a_one_life.py`
and earlier scripts, do not edit them. Do not commit; do not touch docs/loop.md or audit.md. `.venv/bin/python`, CPU
torch, `torch.set_num_threads(2)`. **Run experiments in the foreground, one at a time.**

## Why
In M4a the store run (`rand`) was worse than the GRU alone (`rand_nostore`) by 0.08 bits, mostly on motifs and
counters. The store bundles three parts: replay (drawn only from the utility store), the match mixture, and the budget.
This run separates them. **Seeds 0 1 2 (selection seeds, not 6–8).** T = 512 glances. World identical to M4a.

## Conditions (controller = rand in all but the last; same glance sequence per seed in every rand condition)
| name | store + match | replay source (2 steps × batch 8 × 64 after each live step) |
|---|---|---|
| `nostore` | no | none (= M4a rand_nostore) |
| `hist_rep` | no | **whole history**: 64-symbol segments drawn uniformly over every symbol observed so far, any channel, from a zero state with that segment's channel embedding |
| `store_norep` | yes | none |
| `store_rep` | yes | utility store only (= M4a rand) |
| `store_hist` | yes | whole history (as hist_rep) |
| `gmc_hist` | yes | whole history, controller = M4a gmc_prop |

Store, match, mixture, writes, utility and decay exactly as M4a, with **one fix (all store conditions):** the budget is
hard. When M > 0.15 × observed symbols × log2 42 and no unprotected window is left, evict the oldest protected window,
repeating until M ≤ budget. Assert peak M/N ≤ budget + one window's bits / N. The whole-history replay source is the raw
observation log (memory not charged: it stands for "store everything"; report its size).

## Score (per condition and seed)
- **W_mix**: frozen mixture (GRU + match over the final store) on fresh 2048-symbol continuations of the 8 learnable
  channels, exactly as M4a. **W_gru**: the same continuations with the frozen GRU probability alone. For nostore /
  hist_rep, W_mix = W_gru.
- W_F (mix and gru), per-channel held-out bits (both), glance shares (overall, by quarter), peak and final M/N, replay
  steps, wall time.
- Smoke test T = 32, one seed, all conditions. Budget 120 min wall; if over, drop `gmc_hist` first and say so.

## Report: results/m4b/RESULTS.md (numbers only, no verdict) + results.json

## Pre-registered bars (orchestrator scores; do not change)
1. **Replay source is the culprit:** W_gru(store_hist) ≤ W_gru(store_rep) − 0.05 on the mean, and mean M7 and M37 bits
   (gru) in store_hist within 0.05 of nostore.
2. **Replay helps in the active world (H21c replicates):** W(hist_rep) ≤ W(nostore) − 0.05 on the mean.
3. **Match mixture earns its place:** in store_hist, W_mix ≤ W_gru − 0.02 on the mean and W_F_mix ≤ W_F_gru − 0.5.
4. Implementation check: every store run respects the hard budget (assert above).
5. Report only: gmc_hist vs store_hist (W_mix, W_F, N and Z shares).
Kill rule: if bar 1 fails, replay source is not the cause; the next suspect is the mixture-driven writes.

## Part 2, after the experiment (papers; touch only R2 `pdp-papers` and D1 `pdp-papers-index`)
Retry the four open-access papers whose downloads failed: french1991, krogh1995, doro2023, sun2025. Use only the
publisher's or authors' own open copy (eScholarship, NeurIPS proceedings PDF, OpenReview PDF, ICLR proceedings PDF, arXiv);
never shadow libraries. Save to docs/papers/, run `python3 tools/sync_papers.py`, and set file_status = stored for each
one that lands. No deletes. Report which landed.

## Output
Final message: the main table (W_mix, W_gru, W_F_mix, W_F_gru, M7, M37, C1 gru per condition, means and per seed),
wall time, anything dropped, and the papers outcome.
