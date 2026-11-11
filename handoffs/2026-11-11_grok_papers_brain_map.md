# Papers: register the brain-map and nonlinearity references (grok)

Read `/root/pdp/CLAUDE.md` first (section "Papers library"), then `docs/brain_map.md` and the last section of
`docs/compute-vs-data-movement.md`. Do it the same way as `handoffs/2026-11-11_grok_papers_delay_address.md`.

For each paper below:
1. Check the title, authors, year and venue against the publisher, the DOI record or arXiv. If a detail in our docs is
   wrong, report it. Do not edit the docs.
2. If the paper is not in D1 `pdp-papers-index` (table `papers`), add a row with status `checked`. Set `file_status` to
   `to_fetch` if an open copy exists (arXiv, PMC, or the author's or institution's repository), and `wanted`
   otherwise. Record the citing doc path: `docs/brain_map.md` or `docs/compute-vs-data-movement.md`.
3. If the row already exists, only add the citing path.
4. Run `python3 tools/sync_papers.py --fetch` to put the open PDFs in R2 `pdp-papers`.

Keys and papers:
- crick1984: Crick, "Function of the thalamic reticular complex: the searchlight hypothesis", PNAS 1984.
- sherman2002: Sherman & Guillery, "The role of the thalamus in the flow of information to the cortex", Phil Trans R Soc B 2002.
- sherman2001: Sherman, "Tonic and burst firing: dual modes of thalamocortical relay", Trends Neurosci 2001.
- schmitt2017: Schmitt et al., "Thalamic amplification of cortical connectivity sustains attentional control", Nature 2017.
- staresina2015: Staresina et al., "Hierarchical nesting of slow oscillations, spindles and ripples in the human hippocampus during sleep", Nat Neurosci 2015.
- latchoumane2017: Latchoumane et al., "Thalamic spindles promote memory formation during sleep through triple phase-locking of cortical, thalamic, and hippocampal rhythms", Neuron 2017.
- schultz1997: Schultz, Dayan & Montague, "A neural substrate of prediction and reward", Science 1997.
- marr1969: Marr, "A theory of cerebellar cortex", J Physiol 1969.
- albus1971: Albus, "A theory of cerebellar function", Math Biosci 1971.
- wolpert1998: Wolpert, Miall & Kawato, "Internal models in the cerebellum", Trends Cogn Sci 1998.
- mcgaugh2004: McGaugh, "The amygdala modulates the consolidation of memories of emotionally arousing experiences", Annu Rev Neurosci 2004.
- yu2005: Yu & Dayan, "Uncertainty, neuromodulation, and attention", Neuron 2005.
- astonjones2005: Aston-Jones & Cohen, "An integrative theory of locus coeruleus-norepinephrine function", Annu Rev Neurosci 2005.
- hasselmo2006: Hasselmo, "The role of acetylcholine in learning and memory", Curr Opin Neurobiol 2006.
- doya2002: Doya, "Metalearning and neuromodulation", Neural Netw 2002.
- crick2005: Crick & Koch, "What is the function of the claustrum?", Phil Trans R Soc B 2005.
- saper2005: Saper, Scammell & Lu, "Hypothalamic regulation of sleep and circadian rhythms", Nature 2005.
- papez1937: Papez, "A proposed mechanism of emotion", Arch Neurol Psychiatry 1937.
- boyd1985: Boyd & Chua, "Fading memory and the problem of approximating nonlinear operators with Volterra series", IEEE Trans Circuits Syst 1985.
- koopman1931: Koopman, "Hamiltonian systems and transformation in Hilbert space", PNAS 1931.
- mezic2005: Mezić, "Spectral properties of dynamical systems, model reduction and decompositions", Nonlinear Dyn 2005.
- canolty2010: Canolty & Knight, "The functional role of cross-frequency coupling", Trends Cogn Sci 2010.
- lisman2013: Lisman & Jensen, "The theta-gamma neural code", Neuron 2013.

Rules: you may touch only D1 `pdp-papers-index`, R2 `pdp-papers` and the local files that `tools/sync_papers.py`
writes. No deletes. No public bucket access. No shadow libraries and no paywall bypass. Do not edit any doc or
`docs/loop.md`. Never print secrets. Do not commit. Final message: a table with cite_key, status, file_status and
source URL, plus any detail our docs got wrong.
