# Handoff: every cited paper registered, open-access files fetched (worker: grok). Follow-up to 2026-11-09_grok_r2_papers

Read /root/pdp/CLAUDE.md (section "Papers library") and your previous output
handoffs/2026-11-09_grok_r2_papers.out.md. Same hard rules as that spec: touch only R2 `pdp-papers` and D1
`pdp-papers-index`; no public access; never print tokens or IDs; do not commit. Exception to "no deletes": after the
migration in step 1 is verified, you may drop the old `documents` table in `pdp-papers-index` only.

1. **Schema.** Create table `papers` (one row per paper, whether or not we have the file):
   `id INTEGER PRIMARY KEY, cite_key TEXT UNIQUE NOT NULL` (e.g. `schmidhuber2009`, `dohare2024`; lowercase first-author
   surname + year + optional letter), `title, authors, year INTEGER, venue, arxiv_id, doi, url TEXT,
   status TEXT CHECK(status IN ('read','checked','memory')),
   file_status TEXT NOT NULL CHECK(file_status IN ('stored','to_fetch','wanted','unavailable')),
   r2_key TEXT UNIQUE` (NULL unless stored), `filename, content_type, bytes INTEGER, sha256 TEXT,
   cited_in TEXT` (comma-separated repo paths that cite it), `notes, added_at, updated_at`. Indexes on file_status and
   year. Migrate the rows from `documents` (they are `stored`), verify the counts match, then drop `documents`.
2. **Register every paper cited** in: docs/papers/README.md (all tables), docs/literature_2026-11.md,
   handoffs/2026-11-09_grok_curiosity_refs.table.md, docs/curiosity.md, docs/hippocampus_diagnosis.md, docs/goal.md,
   docs/notes.md, docs/loop.md (the "Literature insight" sections), audit.md. Deduplicate by DOI, then arXiv id, then
   normalised title. Fill only fields you can confirm from the repo text or an arXiv/DOI landing page; leave others NULL.
3. **file_status.** arXiv or another open-access PDF you can download without login or payment → download it
   (into /tmp), upload it to R2 as `papers/<cite_key>.pdf`, set `stored` with sha256 and bytes. Everything else (paywalled
   journals, books, old conference proceedings with no free copy) → `wanted`, with `url` set to the publisher/DOI page.
   Never use shadow libraries or any source that bypasses a paywall.
4. **Update `tools/sync_papers.py`** to this schema: it uploads local files in docs/papers/ (matching a row by
   filename/cite_key, or creating one), and with `--fetch` downloads `to_fetch` rows. Add `--wanted` to print the
   wanted list (cite_key, title, authors, year, url) for Ez. Keep `--dry`.
5. **Verify** and put in `handoffs/2026-11-09_grok_papers_registry.out.md`: row counts by file_status; the full wanted
   list; 3 random stored rows with an sha256 check of the R2 object; a second sync run reporting everything unchanged.
