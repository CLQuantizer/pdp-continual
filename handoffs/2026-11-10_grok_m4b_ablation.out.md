I'll read the project instructions and the handoff, then follow that handoff exactly.The handoff is M4b: ablate why the store hurts the active life. I'll read the M4a spec, results, and code before writing anything.M4a shows the store run is worse than the GRU alone. I'll read the life script and the pieces the ablation has to reuse.M4a’s store run is worse than the GRU alone. The ablation splits replay source, the match mixture, and a hard memory budget. I’ll add that as a new script and leave the earlier code untouched.The eval path was importing the alphabet inline. I'll pull that import up to the top, then run the checks.Checks passed. Next is the smoke run: one seed, 32 glances, all six conditions.Smoke finished in 71 seconds. The GRU matches across paired conditions, and every store run stayed inside the budget. I'll confirm the shared glance sequences, then start the scored runs.The scored runs finished in 44 minutes. Nothing was dropped. I'll check the tables and budget asserts, then fetch the four papers.Auto-compacting conversation (81% full)...
Conversation compacted.
The experiment is already scored. I am finishing the papers follow-up: download the four open copies, sync only the ones that land, then report the M4b table.Three publisher PDFs are confirmed. I am saving those and still looking for a legal ICLR copy of doro2023.The three publisher PDFs match the papers. OpenReview still blocks doro2023, so I am storing only the three that downloaded.M4b is scored. Seeds 0, 1, and 2, T=512. Nothing was dropped. The worker does not score the bars.

Means (population standard deviation):

| Condition | W_mix | W_gru | W_F_mix | W_F_gru | M7 gru | M37 gru | C1 gru |
|---|---:|---:|---:|---:|---:|---:|---:|
| nostore | 1.765047 ± 0.091497 | 1.765047 ± 0.091497 | 3.608702 ± 0.181980 | 3.608702 ± 0.181980 | 0.316123 ± 0.023572 | 0.268810 ± 0.029977 | 2.913251 ± 0.578596 |
| hist_rep | 1.462890 ± 0.132849 | 1.462890 ± 0.132849 | 3.179524 ± 0.185061 | 3.179524 ± 0.185061 | 0.317916 ± 0.020029 | 0.342590 ± 0.064552 | 2.353705 ± 0.823016 |
| store_norep | 1.770227 ± 0.094986 | 1.765047 ± 0.091497 | 3.634758 ± 0.158029 | 3.608702 ± 0.181980 | 0.316123 ± 0.023572 | 0.268810 ± 0.029977 | 2.913251 ± 0.578596 |
| store_rep | 1.731051 ± 0.124676 | 1.751776 ± 0.129508 | 3.490902 ± 0.043272 | 3.529547 ± 0.024568 | 0.421479 ± 0.012189 | 0.406250 ± 0.103408 | 2.865887 ± 0.637765 |
| store_hist | 1.445580 ± 0.127978 | 1.462890 ± 0.132849 | 3.140363 ± 0.150469 | 3.179524 ± 0.185061 | 0.317916 ± 0.020029 | 0.342590 ± 0.064552 | 2.353705 ± 0.823016 |
| gmc_hist | 1.279665 ± 0.054844 | 1.290487 ± 0.054224 | 2.522386 ± 0.075812 | 2.527490 ± 0.077651 | 0.310184 ± 0.031406 | 0.341990 ± 0.028893 | 1.532189 ± 0.675628 |

Per seed (W_mix, W_gru, W_F_mix, W_F_gru, M7 gru, M37 gru, C1 gru):

| Seed | Condition | W_mix | W_gru | W_F_mix | W_F_gru | M7 gru | M37 gru | C1 gru |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | nostore | 1.653856 | 1.653856 | 3.525909 | 3.525909 | 0.338729 | 0.267625 | 2.371157 |
| 0 | hist_rep | 1.367324 | 1.367324 | 3.435661 | 3.435661 | 0.292896 | 0.428077 | 1.997661 |
| 0 | store_norep | 1.655698 | 1.653856 | 3.521917 | 3.525909 | 0.338729 | 0.267625 | 2.371157 |
| 0 | store_rep | 1.593878 | 1.610001 | 3.483106 | 3.539800 | 0.431896 | 0.419408 | 2.199180 |
| 0 | store_hist | 1.351512 | 1.367324 | 3.347522 | 3.435661 | 0.292896 | 0.428077 | 1.997661 |
| 0 | gmc_hist | 1.276129 | 1.294480 | 2.478977 | 2.482909 | 0.327526 | 0.330391 | 1.181408 |
| 1 | nostore | 1.877956 | 1.877956 | 3.439066 | 3.439066 | 0.326039 | 0.306103 | 3.715109 |
| 1 | hist_rep | 1.650758 | 1.650758 | 3.098001 | 3.098001 | 0.341925 | 0.327581 | 3.491392 |
| 1 | store_norep | 1.888284 | 1.877956 | 3.524117 | 3.439066 | 0.326039 | 0.306103 | 3.715109 |
| 1 | store_rep | 1.895574 | 1.923079 | 3.442234 | 3.495671 | 0.404376 | 0.525806 | 3.725305 |
| 1 | store_hist | 1.626521 | 1.650758 | 3.078918 | 3.098001 | 0.341925 | 0.327581 | 3.491392 |
| 1 | gmc_hist | 1.348534 | 1.354811 | 2.459190 | 2.462866 | 0.336924 | 0.313858 | 2.477271 |
| 2 | nostore | 1.763329 | 1.763329 | 3.861130 | 3.861130 | 0.283602 | 0.232703 | 2.653486 |
| 2 | hist_rep | 1.370588 | 1.370588 | 3.004910 | 3.004910 | 0.318928 | 0.272110 | 1.572062 |
| 2 | store_norep | 1.766699 | 1.763329 | 3.858241 | 3.861130 | 0.283602 | 0.232703 | 2.653486 |
| 2 | store_rep | 1.703700 | 1.722249 | 3.547366 | 3.553170 | 0.428166 | 0.273536 | 2.673176 |
| 2 | store_hist | 1.358706 | 1.370588 | 2.994649 | 3.004910 | 0.318928 | 0.272110 | 1.572062 |
| 2 | gmc_hist | 1.214333 | 1.222169 | 2.628990 | 2.636694 | 0.266101 | 0.381720 | 0.937889 |

Score wall time is 2631.504 s. Smoke was 71.436 s. Together, 2702.941 s. Dropped conditions: none. The full numbers are in `results/m4b/RESULTS.md` and `results/m4b/results.json`.

Papers: french1991, krogh1995, and sun2025 are stored. doro2023 did not land and remains wanted. The OpenReview PDF returned an HTML browser check, and there is no arXiv copy and no ICLR proceedings PDF.
