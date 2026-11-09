# Handoff: re-check the wanted list for legal free copies (worker: grok)

Read /root/pdp/CLAUDE.md ("Papers library") and handoffs/2026-11-09_grok_papers_registry.out.md. Same hard rules as
that spec (only R2 `pdp-papers` / D1 `pdp-papers-index`; no deletes; never print tokens or IDs; no commit; never use
shadow libraries: `tools/sync_papers.py` has the blocklist).

For every row with file_status = 'wanted' (`python3 tools/sync_papers.py --wanted`), look for a copy the authors or
publisher make freely available, in this order: the author's own homepage or lab page (e.g. people.idsia.ch/~juergen
for Schmidhuber), the conference's open proceedings (proceedings.neurips.cc, papers.nips.cc, eScholarship for CogSci,
PMLR), PubMed Central, institutional repositories, arXiv/bioRxiv/PsyArXiv versions. If found: put its URL in the row's
notes as `oa_pdf=<url>`, set file_status = 'to_fetch', and run `python3 tools/sync_papers.py --fetch` to store it. If
not: leave it 'wanted'. Do not change any other field.

Output `handoffs/2026-11-10_grok_wanted_recheck.out.md`: counts by file_status before / after, each newly stored
cite_key with its source URL, and the remaining wanted list (cite_key, title, year, url).
