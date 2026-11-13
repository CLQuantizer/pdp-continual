# Papers: register the persistence and space-time references (grok)

Read `/root/pdp/CLAUDE.md` first (section "Papers library"), then the last two sections of
`docs/compute-vs-data-movement.md`. Do it the same way as `handoffs/2026-11-11_grok_papers_brain_map.md`.

For each paper below:
1. Check the title, authors, year and venue against the publisher, the DOI record or arXiv. If a detail in our docs is
   wrong, report it. Do not edit the docs.
2. If the paper is not in D1 `pdp-papers-index` (table `papers`), add a row with status `checked`. Set `file_status` to
   `to_fetch` if an open copy exists (arXiv, PMC, or the author's or institution's repository), and `wanted`
   otherwise. Record the citing doc path: `docs/compute-vs-data-movement.md`.
3. If the row already exists, only add the citing path.
4. Run `python3 tools/sync_papers.py --fetch` to put the open PDFs in R2 `pdp-papers`.

Keys and papers:
- crick1984b: Crick, "Memory and molecular turnover", Nature 312:101, 1984. (The key crick1984 is already the
  searchlight paper.)
- lieb1972: Lieb & Robinson, "The finite group velocity of quantum spin systems", Commun. Math. Phys. 28:251, 1972.
- jeffress1948: Jeffress, "A place theory of sound localization", J. Comp. Physiol. Psychol. 41:35, 1948.
- okeefe1993: O'Keefe & Recce, "Phase relationship between hippocampal place units and the EEG theta rhythm",
  Hippocampus 3:317, 1993.
- izhikevich2006: Izhikevich, "Polychronization: computation with spikes", Neural Computation 18:245, 2006.
- fernando2003: Fernando & Sojakka, "Pattern recognition in a bucket", ECAL 2003 (LNCS 2801).

Rules: you may touch only D1 `pdp-papers-index`, R2 `pdp-papers` and the local files that `tools/sync_papers.py`
writes. No deletes. No public bucket access. No shadow libraries and no paywall bypass. Do not edit any doc or
`docs/loop.md`. Never print secrets. Do not commit. Final message: a table with cite_key, status, file_status and
source URL, plus any detail our docs got wrong.
