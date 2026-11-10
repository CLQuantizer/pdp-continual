# Handoff: finish H24f (worker: grok). Codex started it and was stopped at its usage cap

Read /root/pdp/CLAUDE.md and the spec `handoffs/2026-11-10_codex_h24f_fine_store.md`: its bars and rules apply
unchanged. Codex wrote `h24f_fine_store.py` and partial results in `results/h24f/` (full5 for seeds 6 7 8; fine for seeds
6 7; see run.log, results.json, the *.npz files). Use `.venv/bin/python`; CPU torch is installed; 2 cores.

1. Read `h24f_fine_store.py` and check it against the spec: order-5 match model; symbol-level writes (bits > 4.0, window
   [i−16, i+4], merged); replay and matching only inside stored windows; prequential scoring before any update using a
   symbol; memory M = stored symbols × log2(42) + 32 bits per window. List every mismatch in your output. Fix a mismatch
   only if it would change the numbers, and then rerun what it affects.
2. Complete the missing runs: `fine` seed 8, then `fine3` (seeds 6 7 8) if the remaining budget allows (≤ 100 min wall
   in total for this handoff). Do not rerun completed runs unless step 1 requires it.
3. Write `results/h24f/RESULTS.md` (numbers only, no verdict) as the spec asks, including the Pareto table (P vs peak
   M/N for gru and full from H24e, plus full5, fine, fine3) and F value-digit bits by prior appearances.
4. Do not commit; do not edit docs/loop.md or audit.md. Final message: the main tables, wall time, and the step-1 check.

## Resume note (orchestrator, 10 Nov)
The first grok run fixed the touching-window merge (keep that fix) and completed the corrected `fine` seed 6
(results/h24f/fine_6.npz, 10:26). Its background jobs were killed when the headless run exited. Now: rerun the corrected
`fine` for seeds 7 and 8, then `fine3` for seeds 6 7 8 if the time allows. **Run the jobs in the foreground (blocking
commands, no background tasks)**; at most 2 in parallel. Then write RESULTS.md as specified, noting the fix and that
full5 was not rerun.
