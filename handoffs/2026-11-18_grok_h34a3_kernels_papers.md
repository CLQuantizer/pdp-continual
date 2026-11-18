# Run H34a3's kernels and register new papers (grok)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-18_codex_h34a3_cascade_and_h36_fix.md` (Job 1) and `results/h34a3/RESULTS.md`.

1. **H34a3:** codex passed every smoke check. Load `.env` (`set -a; . ./.env; set +a`) and run `.venv/bin/python kaggle/h34a3/dispatch.py` in the
   foreground until the 4 kernels finish (account running kernels ≤ 5). Then run `kaggle/h34a3/summarize.py`; check that `results/h34a3/RESULTS.md`
   starts with the scoring table (casc_4 − sum_1, casc_4 − plain_4, casc_4 − wide_1 per seed, full run and tail, counts). Fix only plumbing.
2. **Papers:** register in D1 `pdp-papers-index` as CLAUDE.md says (status `memory`, file_status `to_fetch` if open access else `wanted`), verifying
   title, year and id first: Mindermann et al. 2022, Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt (RHO-LOSS,
   arXiv 2206.07137); Xie et al. 2023, DoReMi (arXiv 2305.10429); Evans et al. 2024, Data curation via joint example selection (JEST, arXiv 2406.17711);
   Pagnoni et al. 2024, Byte Latent Transformer (arXiv 2412.09871); Hwang, Wang, Gu 2025, Dynamic Chunking for End-to-End Hierarchical Sequence
   Modeling (H-Net); Saffran, Aslin, Newport 1996, Statistical learning by 8-month-old infants (Science); Schmidhuber 1992, Learning complex, extended
   sequences using the principle of history compression (Neural Computation). Then `python3 tools/sync_papers.py --fetch` and `--wanted`. Report new
   `wanted` rows as "cite_key: title (url)".

Rules: files you may edit: `kaggle/h34a3/**`, `results/h34a3/**`. Do not touch `docs/`, `audit.md`, any `*.py` at the repo root, other kaggle or
results folders, `.env`. Never print secrets. Do not commit.
