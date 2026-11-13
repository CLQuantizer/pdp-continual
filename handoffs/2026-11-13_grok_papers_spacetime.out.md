I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is a papers-library registration. I'll read the cited sections and the previous papers handoff so I follow the same process.The previous papers handoff shows the registration pattern. I'll match that and check how citing paths are stored.The citations are in the docs. Next I’ll confirm each record against the publisher and see which ones are already in the index.The six papers are not in the index. One has an author PDF; the others are closed. I’ll add the rows, then fetch the open file.Registered all six papers in D1 `pdp-papers-index`. `python3 tools/sync_papers.py --fetch` stored the one open PDF in R2 `pdp-papers`. Titles, authors, years, and venues match the publisher and Crossref records, except the Crick title prefix below. Nothing was deleted or committed.

| cite_key | status | file_status | source URL |
|---|---|---|---|
| crick1984b | checked | wanted | https://doi.org/10.1038/312101a0 |
| lieb1972 | checked | wanted | https://doi.org/10.1007/BF01645779 |
| jeffress1948 | checked | wanted | https://doi.org/10.1037/h0061495 |
| okeefe1993 | checked | wanted | https://doi.org/10.1002/hipo.450030307 |
| izhikevich2006 | checked | stored | https://www.izhikevich.org/publications/spnet.pdf |
| fernando2003 | checked | wanted | https://doi.org/10.1007/978-3-540-39432-7_63 |

`izhikevich2006` is Izhikevich’s own copy of the Neural Computation article (38 pages, 245–282). The five `wanted` rows are paywalled: Unpaywall lists them closed, and they are not in PMC, arXiv, or a live author or institutional repository. Fernando’s old Sussex page (`users.sussex.ac.uk/~ctf20/.../bucket.pdf`) returns 404. `crick1984` (the searchlight paper) was left unchanged.

Nature and Crossref title Crick’s 1984 note **“Neurobiology: Memory and molecular turnover”** (Nature 312:101, 8 November 1984). The docs and PubMed use **“Memory and molecular turnover”** and drop the News & Views heading. The index row uses the Crossref title.
