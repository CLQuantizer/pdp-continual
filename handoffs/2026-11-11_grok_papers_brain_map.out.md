I'll read the project instructions and the handoff, then follow that handoff exactly.The handoff is a papers-library registration. I'll read the cited docs, the earlier delay-address handoff, and how `sync_papers.py` writes rows.The catalogue and the sync script are clear. Next I’ll check which of these papers are already in D1 and how citation paths are stored.Three cite keys are already in the index. I’ll check every title, author list, year, and venue against the publisher record, and see which ones have an open PDF.The publisher records match the citations. Next I’ll confirm which open PDFs actually download, and look for author or institutional copies of the closed ones.The publisher records match. Open copies are in hand for most of the new papers. I’ll register the rows, store the PDFs the sync script cannot download itself, and then run the fetch.Schmitt and Staresina check out. Next I’ll write the index rows and store the open files.Registered all 23 papers in D1 `pdp-papers-index` and synced the open files into R2 `pdp-papers`. `yu2005`, `astonjones2005`, and `hasselmo2006` were already there, so only `docs/brain_map.md` was added to their citation paths. Titles, authors, years, and venues in the docs match the publisher pages and the DOI records.

| cite_key | status | file_status | source URL |
|---|---|---|---|
| crick1984 | checked | stored | https://pmc.ncbi.nlm.nih.gov/articles/PMC345636/pdf/pnas00615-0353.pdf |
| sherman2002 | checked | stored | https://pmc.ncbi.nlm.nih.gov/articles/PMC1693087/pdf/12626004.pdf |
| sherman2001 | checked | stored | https://voices.uchicago.edu/shermanlab/files/2020/01/144-2001-burst.pdf |
| schmitt2017 | checked | stored | https://pmc-oa-opendata.s3.amazonaws.com/PMC5570520.1/PMC5570520.1.pdf |
| staresina2015 | checked | stored | https://pmc-oa-opendata.s3.amazonaws.com/PMC4625581.1/PMC4625581.1.pdf |
| latchoumane2017 | checked | wanted | https://doi.org/10.1016/j.neuron.2017.06.025 |
| schultz1997 | checked | stored | https://www.gatsby.ucl.ac.uk/~dayan/papers/sdm97.pdf |
| marr1969 | checked | stored | https://pmc.ncbi.nlm.nih.gov/articles/PMC1351491/pdf/jphysiol01078-0177.pdf |
| albus1971 | checked | stored | https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=820146 |
| wolpert1998 | checked | wanted | https://doi.org/10.1016/S1364-6613(98)01221-2 |
| mcgaugh2004 | checked | wanted | https://doi.org/10.1146/annurev.neuro.27.070203.144157 |
| yu2005 | checked | stored | http://www.gatsby.ucl.ac.uk/~dayan/papers/yud2005.pdf |
| astonjones2005 | checked | wanted | https://doi.org/10.1146/annurev.neuro.28.061604.135709 |
| hasselmo2006 | checked | stored | https://www.bu.edu/hasselmo/HasselmoCurrOpinion2006.pdf |
| doya2002 | checked | wanted | https://doi.org/10.1016/S0893-6080(02)00044-8 |
| crick2005 | checked | stored | https://pmc.ncbi.nlm.nih.gov/articles/PMC1569501/pdf/rstb20051661.pdf |
| saper2005 | checked | wanted | https://doi.org/10.1038/nature04284 |
| papez1937 | checked | wanted | https://doi.org/10.1001/archneurpsyc.1937.02260220069003 |
| boyd1985 | checked | stored | https://web.stanford.edu/~boyd/papers/pdf/fading_volterra.pdf |
| koopman1931 | checked | stored | https://pmc.ncbi.nlm.nih.gov/articles/PMC1076052/pdf/pnas01721-0085.pdf |
| mezic2005 | checked | wanted | https://doi.org/10.1007/s11071-005-2824-x |
| canolty2010 | checked | stored | https://pmc.ncbi.nlm.nih.gov/articles/PMC3359652/pdf/nihms235872.pdf |
| lisman2013 | checked | stored | https://pmc.ncbi.nlm.nih.gov/articles/PMC3648857/pdf/nihms455900.pdf |

`schmitt2017`, `staresina2015`, `canolty2010`, and `lisman2013` are PMC author manuscripts. `schultz1997` is Dayan's copy, `sherman2001` is Sherman's Chicago copy, `boyd1985` is Boyd's Stanford copy, and `albus1971` is NIST's copy of the Mathematical Biosciences article. The seven `wanted` rows have no legal open PDF. Nothing was deleted or committed.
