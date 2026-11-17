I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is a papers-library registration. I'll read the papers README and the sync tool, then check how existing D1 rows are written.Several of these may already be indexed. I'll check the live rows and confirm each title, year, and arXiv id before writing anything.The arXiv records match the preprints already in the index. I'll confirm the five missing papers against Crossref and check which ones have a legal open PDF.Seven of the twelve papers are already indexed and match the sources. I'll add the five that are missing, using the publisher and arXiv records. PubMed's Felleman DOI points at a different article, so that row will use the Crossref DOI.The five new rows are in. Next I’ll fetch the open-access PDFs and print the wanted list.Five papers were added to D1 `pdp-papers-index`. `python3 tools/sync_papers.py --fetch` stored the two open PDFs in R2 `pdp-papers`. Seven of the twelve were already registered, so those rows were left as they were. Nothing was committed.

**Added**

| cite_key | status | file_status | source |
|---|---|---|---|
| he2016 | memory | stored | https://arxiv.org/abs/1512.03385 |
| kubilius2019 | memory | stored | https://arxiv.org/abs/1909.06161 |
| felleman1991 | memory | wanted | https://doi.org/10.1093/cercor/1.1.1 |
| thorpe1996 | memory | wanted | https://doi.org/10.1038/381520a0 |
| douglas2004 | memory | wanted | https://doi.org/10.1146/annurev.neuro.27.070203.144152 |

`he2016.pdf` is 819,383 bytes, sha256 `1e0651b6810ecba34a3dbc5b5b0209226f889004607c1f203540a48d64e5a93a`. `kubilius2019.pdf` is 2,700,424 bytes, sha256 `15a1a794a751414787c3251d8eef07b2a5bd031b6c7d83fe373bd5572f145e9e`. A second download of each arXiv PDF matched those hashes.

**Already there:** `gu2022`, `gu2023`, `orvieto2023`, `zucchet2023`, `lillicrap2016`, `nokland2016`, and `schmidhuber1991b` (the IJCNN paper *Curious model-building control systems*). Titles, years, and arXiv ids match the arXiv API and the stored rows.

**Corrections**

- `he2016` keeps year 2016. That is the CVPR year (DOI `10.1109/cvpr.2016.90`). arXiv `1512.03385` is only v1, posted 10 December 2015.
- `kubilius2019` uses the arXiv title *Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs* and the arXiv v2 author list. The id `1909.06161` is correct, and the comment says NeurIPS 2019 oral. On the NeurIPS page, Rishi Rajalingham is listed before Ha Hong and Najib Majaj.
- `felleman1991` uses DOI `10.1093/cercor/1.1.1` (Felleman and Van Essen, Cerebral Cortex 1:1–47). PubMed lists `10.1093/cercor/1.1.1-a` for this paper. Crossref assigns that DOI to *Preface: Cerebral Cortex Has Come of Age* by Goldman-Rakic and Rakic. The stored title is the Crossref title, in title case.

Thorpe 1996 and Douglas and Martin 2004 match Crossref on title, authors, and year. Crossref prints the Douglas title in all caps. The row uses *Neuronal circuits of the neocortex*.

**Wanted** (`python3 tools/sync_papers.py --wanted` printed 50 rows; these three are the new ones). Unpaywall lists them closed, and they are not in PMC. The ETH record for Douglas and Martin is metadata only.

- felleman1991: Distributed Hierarchical Processing in the Primate Cerebral Cortex (https://doi.org/10.1093/cercor/1.1.1)
- thorpe1996: Speed of processing in the human visual system (https://doi.org/10.1038/381520a0)
- douglas2004: Neuronal circuits of the neocortex (https://doi.org/10.1146/annurev.neuro.27.070203.144152)
