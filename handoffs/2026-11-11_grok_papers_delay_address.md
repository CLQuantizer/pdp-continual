# Papers: register the new references in docs/compute-vs-data-movement.md (grok)

Read `/root/pdp/CLAUDE.md` (section "Papers library"), then the last section of `docs/compute-vs-data-movement.md`
("Ez, 11 Nov 2026: moving is preserving").

Register each of these in D1 `pdp-papers-index`, table `papers`, unless a row already exists; if it does, only add
this doc to its citation path. Use `cite_key` as below, plus title, authors, year, DOI or URL, `status` = `checked`
(confirm the title, authors and year against the publisher, arXiv or DOI page), and `file_status`:
- `gu2020`: HiPPO: Recurrent Memory with Optimal Polynomial Projections (NeurIPS 2020; arXiv 2008.07669)
- `gu2022`: Efficiently Modeling Long Sequences with Structured State Spaces (ICLR 2022; arXiv 2111.00396)
- `voelker2019`: Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks (NeurIPS 2019)
- `orvieto2023`: Resurrecting Recurrent Neural Networks for Long Sequences (ICML 2023; arXiv 2303.06349)
- `ramsauer2021`: Hopfield Networks is All You Need (ICLR 2021; arXiv 2008.02217)
- `hopfield1982`: Neural networks and physical systems with emergent collective computational abilities (PNAS 1982)
- `teyler1986`: The hippocampal memory indexing theory (Behavioral Neuroscience 1986)
- `howard2002`: A distributed representation of temporal context (J. Math. Psychol. 2002)
- `dasgupta2017`: A neural algorithm for a fundamental computing problem (Science 2017)
- `landauer1961`: Irreversibility and heat generation in the computing process (IBM J. Res. Dev. 1961)
- `shin2017`: Continual Learning with Deep Generative Replay (NeurIPS 2017; arXiv 1705.08690)
- `grazzi2025`: Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues (ICLR 2025), if it is not
  already registered

File: store the PDF in R2 `pdp-papers` only from a legal open source (arXiv, the publisher's open-access copy, PMC,
the author's own page, NeurIPS/ICLR/PMLR proceedings). Mark `to_fetch` if open, `wanted` if not. Never use shadow
libraries or paywall bypasses. Then run `python3 tools/sync_papers.py` and `python3 tools/sync_papers.py --fetch`.

Touch only R2 `pdp-papers`, D1 `pdp-papers-index` and `docs/papers/` (and whatever files the sync tool writes). Delete
nothing. No public bucket access. Never print tokens. Do not commit. Do not edit `docs/loop.md`, `audit.md` or any
experiment code. Final message: a table of cite_key, status, file_status, source URL.
