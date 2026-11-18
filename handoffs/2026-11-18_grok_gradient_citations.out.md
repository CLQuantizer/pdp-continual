I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is a citation check of the gradient page, then a papers-index registration. I'll read the credit-assignment doc and how papers are registered.The credit map has about fifty method rows. I'll read how papers get into D1, then check each citation against the original source.The index already has 168 papers. I'll pull the cite keys, then check each gradient-page citation against the original record.OpenAlex missed most of the list, and one hit already disagrees with the page: SnAp is listed as 2020 but the index year is 2021. I'll query Crossref, arXiv, and DBLP directly.Auto-compacting conversation (84% full)...
Conversation compacted.
The citation checks are done. I’ll apply only the confirmed origin-column fixes, then register the papers and run the sync.The six origin strings match the file. I’ll apply those edits and fill the remaining paper metadata before inserting rows.The page now has the six citation fixes. Next I’ll pull author lists, identifiers, and open-access status, then insert only the missing rows.The journal issue for OSTL is November 2023, so that year is updated. I’ll build the D1 rows from the publisher and arXiv records, then insert only the missing keys.Citations on `docs/credit_assignment.md` are checked. Seven origin lines were corrected, the byline is now `*By Ez · 18 Nov 2026*`, and 56 papers were added to D1 `pdp-papers-index`. Nothing was committed. The only repo file edited is `docs/credit_assignment.md`.

| Old | New | Source |
|---|---|---|
| Tallec & Ollivier 2017 | Tallec & Ollivier 2018 | [ICLR 2018](https://openreview.net/forum?id=rJQDjk-0b) (arXiv 1702.05043 is the 2017 preprint) |
| Menick et al. 2020 | Menick et al. 2021 | [ICLR 2021 poster](https://iclr.cc/virtual/2021/poster/3203) |
| Bohnstingl et al. 2022 | Bohnstingl et al. 2023 | [IEEE TNNLS 34(11), Nov 2023](https://doi.org/10.1109/TNNLS.2022.3153985) (DOI minted in 2022) |
| Bengio et al. 2007 | Bengio et al. 2006 | [NeurIPS 2006](https://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks) |
| Journé et al. 2023 (SoftHebb) | Moraitis et al. 2022 (SoftHebb) | [Moraitis et al., arXiv 2107.05747](https://arxiv.org/abs/2107.05747), journal version 2022. Journé et al. 2023 is a later paper. |
| Williams 1992 | Jabri & Flower 1992; Widrow & Lehr 1990; Williams 1992 | [Weight perturbation](https://doi.org/10.1109/72.105429); [node perturbation](https://doi.org/10.1109/5.58323); [REINFORCE](https://doi.org/10.1007/BF00992696) |
| Rombouts et al. 2015 | Rombouts et al. 2012 | [NeurIPS 2012](https://papers.nips.cc/paper/4813-neurally-plausible-reinforcement-learning-of-working-memory-tasks), where AuGMEnT is named. AGREL stays Roelfsema & van Ooyen 2005. |

**Rows added:** 56 (`status` `memory`). `python3 tools/sync_papers.py --fetch` stored 41 open-access PDFs in R2. `bienenstock1982` is open access at [the Journal of Neuroscience](https://doi.org/10.1523/jneurosci.02-01-00032.1982), but the PDF request returned 403, so that row is still `to_fetch`.

Papers already in the index were left as they were, including `moraitis2022`. `journe2023` is no longer cited and was not changed.

**New wanted papers** (14):

- werbos1974: Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences (no DOI or public landing page)
- oja1982: Simplified neuron model as a principal component analyzer (https://doi.org/10.1007/bf00275687)
- rumelhart1986: Learning representations by back-propagating errors (https://doi.org/10.1038/323533a0)
- williams1989: A Learning Algorithm for Continually Running Fully Recurrent Neural Networks (https://doi.org/10.1162/neco.1989.1.2.270)
- werbos1990: Backpropagation through time: what it does and how to do it (https://doi.org/10.1109/5.58337)
- widrow1990: 30 years of adaptive neural networks: perceptron, Madaline, and backpropagation (https://doi.org/10.1109/5.58323)
- foldiak1991: Learning Invariance from Transformation Sequences (https://doi.org/10.1162/neco.1991.3.2.194)
- movellan1991: Contrastive Hebbian Learning in the Continuous Hopfield Model (https://doi.org/10.1016/B978-1-4832-1448-1.50007-X)
- jabri1992: Weight perturbation: an optimal architecture and learning technique for analog VLSI feedforward and recurrent multilayer networks (https://doi.org/10.1109/72.105429)
- spall1992: Multivariate stochastic approximation using a simultaneous perturbation gradient approximation (https://doi.org/10.1109/9.119632)
- kolen1994: Backpropagation without weight transport (https://doi.org/10.1109/icnn.1994.374486)
- oreilly1996: Biologically Plausible Error-Driven Learning Using Local Activation Differences: The Generalized Recirculation Algorithm (https://doi.org/10.1162/neco.1996.8.5.895)
- wiskott2002: Slow Feature Analysis: Unsupervised Learning of Invariances (https://doi.org/10.1162/089976602317318938)
- roelfsema2005: Attention-Gated Reinforcement Learning of Internal Representations for Classification (https://doi.org/10.1162/0899766054615699)
