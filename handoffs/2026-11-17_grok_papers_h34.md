# Papers: register the H34 and W6 references (grok)

Read `/root/pdp/CLAUDE.md` (Papers library) and `docs/papers/README.md`. Register each paper below in D1 `pdp-papers-index`
(table `papers`) with `status` = `memory` unless it is already there, and set `file_status` (`to_fetch` when open access, else
`wanted`), following how existing rows are written. Then run `python3 tools/sync_papers.py --fetch` and
`python3 tools/sync_papers.py --wanted`. Report the rows added and any `wanted` papers as "cite_key: title (url)".

- He, Zhang, Ren, Sun 2016, Deep Residual Learning for Image Recognition (arXiv 1512.03385)
- Gu, Goel, Ré 2022, Efficiently Modeling Long Sequences with Structured State Spaces (S4, arXiv 2111.00396)
- Gu, Dao 2023, Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv 2312.00752)
- Orvieto et al. 2023, Resurrecting Recurrent Neural Networks for Long Sequences (LRU, arXiv 2303.06349)
- Zucchet, Meier, Schug, Mujika, Sacramento 2023, Online learning of long-range dependencies (arXiv 2305.15947)
- Lillicrap et al. 2016, Random synaptic feedback weights support error backpropagation for deep learning (Nat Commun)
- Nøkland 2016, Direct Feedback Alignment Provides Learning in Deep Neural Networks (arXiv 1609.01596)
- Felleman, Van Essen 1991, Distributed hierarchical processing in the primate cerebral cortex (Cereb Cortex)
- Thorpe, Fize, Marlot 1996, Speed of processing in the human visual system (Nature)
- Kubilius et al. 2019, Brain-like object recognition with high-performing shallow recurrent ANNs (CORnet-S, arXiv 1909.06161)
- Douglas, Martin 2004, Neuronal circuits of the neocortex (Annu Rev Neurosci)
- Schmidhuber 1991, Curious model-building control systems (IJCNN) — if not already registered

Verify each title, year and arXiv id against the source before inserting; fix any that are wrong and say so.

Rules: only D1/R2 via `cf` and the sync tool; do not touch any code, `docs/`, `results/`, `.env`. Never print secrets. Do not commit.
