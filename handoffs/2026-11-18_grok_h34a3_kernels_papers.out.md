I'll follow the handoff: read the H34a3 spec and results, then run the kernels and register the papers.Smoke passed and the dispatcher lock is free. I'll start the four kernels and verify the seven papers before registering them.All 4 H34a3 kernels finished. `results/h34a3/RESULTS.md` starts with the scoring table. 0 of 2 seeds meet any bar.

| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds ≤ threshold | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|
| Full 0–300 KB | casc_4 − sum_1 | 0.508703414 | 0.493013238 | 0.500858326 | -0.10 | 0 | 2 |
| Full 0–300 KB | casc_4 − plain_4 | 0.463453103 | 0.462066014 | 0.462759559 | -0.05 | 0 | 2 |
| Full 0–300 KB | casc_4 − wide_1 | 0.535499710 | 0.512393083 | 0.523946396 | -0.03 | 0 | 2 |
| Tail 50–300 KB | casc_4 − sum_1 | 0.493904590 | 0.474969934 | 0.484437262 | -0.10 | 0 | 2 |
| Tail 50–300 KB | casc_4 − plain_4 | 0.682100041 | 0.684796372 | 0.683448206 | -0.05 | 0 | 2 |
| Tail 50–300 KB | casc_4 − wide_1 | 0.513878065 | 0.488572960 | 0.501225512 | -0.03 | 0 | 2 |

Scored kernels: 4/4. Full bits/byte: casc_2 3.614 / 3.635, casc_4 3.997 / 4.014. Each result is 300,000 bytes and matches the code fingerprint and enwik8 hash. `summarize.py` reproduces this file. Nothing was committed.

Kaggle returns 403 for a private dataset that is not there yet. The dispatcher treated that as fatal, so it now creates the dataset in that case. The run stayed at two H34a3 kernels at a time.

**Papers.** Five open arXiv PDFs are stored in R2. A second download of each matched the stored sha256. `schmidhuber1992` was already stored (`checked`, Neural Computation, DOI `10.1162/neco.1992.4.2.234`) and was left as it was.

| cite_key | status | file_status | id |
|---|---|---|---|
| mindermann2022 | memory | stored | 2206.07137 |
| xie2023 | memory | stored | 2305.10429 |
| evans2024 | memory | stored | 2406.17711 |
| pagnoni2024 | memory | stored | 2412.09871 |
| hwang2025 | memory | stored | 2507.07955 |
| saffran1996 | memory | wanted | 10.1126/science.274.5294.1926 |

The arXiv title for Evans et al. is *Data curation via joint example selection further accelerates multimodal learning* (JEST). Hwang, Wang, and Gu 2025 is arXiv `2507.07955`.

New wanted row:

- saffran1996: Statistical Learning by 8-Month-Old Infants (https://doi.org/10.1126/science.274.5294.1926)

That notice is the old progress check from while the kernels were still starting. It finished cleanly, and the H34a3 run and the paper registration were already complete.
