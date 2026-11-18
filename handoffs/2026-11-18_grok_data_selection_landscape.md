# Landscape: learning-progress data selection and who sells data curation to AI labs (grok, research only; internal)

Read `/root/pdp/CLAUDE.md`, `docs/loop.md`'s entries from 17–18 Nov on H35, H35b and H35c, and `handoffs/2026-11-18_grok_h35c_web_curiosity.md`.

Write `internal/data_selection_landscape.md` (an **internal** document: not under `docs/`, never published). Use primary sources (papers, official
blogs, company pages), with a link for every claim; mark anything you could not verify.
1. **Methods:** every published method that selects or weights pretraining data by model-based signals: RHO-LOSS, JEST, DoReMi, ODM / online data mixing,
   ADO, data-mixing laws, learnability-based selection, influence-function and gradient-based selection (e.g. LESS), perplexity and classifier filters
   (CCNet, FineWeb-Edu, DCLM), dedup (MinHash, SemDeDup, D4), and anything else close to "select by learning progress". For each: signal, cost relative to
   training, scale tested, reported gains, and whether it needs a reference model or curated holdout.
2. **Closest prior art to ours:** which methods already use loss decrease over training time, or snapshot differences, as the score? Quote the exact
   formulation.
3. **Companies and teams** selling or building data curation for model developers (e.g. DatologyAI, Scale AI, Hugging Face's data teams, Ai2): what they
   offer, public claims, customers, funding where public.
4. **Open questions labs say they care about** in data selection (from papers' limitations sections and talks).
5. **A one-page summary** at the top: where a learning-progress selector could be new, where it is already covered, and which evidence would be needed to
   convince a frontier lab.
Also register every paper you cite in D1 `pdp-papers-index` (CLAUDE.md rules), then run `python3 tools/sync_papers.py --fetch` and `--wanted`, and list new
`wanted` papers as "cite_key: title (url)" in your final message.

Rules: the only repo file you may create is `internal/data_selection_landscape.md`. Do not touch any other file, `docs/`, `.env`. Never print secrets. Do not commit.
