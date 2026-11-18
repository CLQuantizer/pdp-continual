# Verify the citations on the /gradient page and register them (grok)

Read `/root/pdp/CLAUDE.md` (Papers library) and `docs/credit_assignment.md`.

1. For **every** method row in `docs/credit_assignment.md`, check the cited authors and year (and the method's name) against the original paper
   (arXiv, DOI, publisher or DBLP). Where something is wrong, fix it in `docs/credit_assignment.md`, changing only the citation text (authors, year)
   or a method name that is plainly wrong. Do not change the descriptions, structure or the last section.
2. When every citation is checked, replace the line `*By Ez · 18 Nov 2026 · Years and authors are being checked against the original papers.*` with
   `*By Ez · 18 Nov 2026*`.
3. Register every cited paper in D1 `pdp-papers-index` as CLAUDE.md says, skipping ones already present (status `memory`, file_status `to_fetch` if
   open access, else `wanted`). Then run `python3 tools/sync_papers.py --fetch` and `python3 tools/sync_papers.py --wanted`.
4. Report: a table of every correction made (old → new, source link), the number of rows added to D1, and new `wanted` papers as
   "cite_key: title (url)".

Rules: the only repo file you may edit is `docs/credit_assignment.md`. Do not touch any other file, `.env`, or other docs. Never print secrets. Do not commit.
