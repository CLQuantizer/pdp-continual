# Papers: register and fetch one new reference (grok)

Read `/root/pdp/CLAUDE.md` (section "Papers library") first.

`docs/papers/README.md` now has a row for Zucchet, Meier, Schug, Mujika & Sacramento 2023, "Online learning of
long-range dependencies" (NeurIPS; arXiv 2305.15947). Make sure it has a row in D1 `pdp-papers-index` with
`status` = memory, and fetch the open-access PDF from arxiv.org into R2 `pdp-papers` (`file_status` = stored), the way
`tools/sync_papers.py` does it. If the arXiv number turns out to be wrong, report the correct one and do not guess.

Rules: touch only R2 `pdp-papers`, D1 `pdp-papers-index` and `docs/papers/README.md` (that row only). No deletes.
No public bucket access. No shadow libraries. Never print secrets. Do not commit. Do not edit `docs/loop.md` or `audit.md`.
