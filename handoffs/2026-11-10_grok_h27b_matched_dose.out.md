I'll read the project instructions and the handoff note, then follow that handoff exactly.The handoff is a matched-dose replay experiment on fresh seeds. I'll read the H27 spec, results, and script before writing anything new.H27 showed a small buffer replayed too often overfits. H27b keeps that machine and lowers the dose on fresh seeds. I'll add the new script and kernels without touching the existing ones.The local match passed: `hist_e1` reproduces H27 `hist1`, and the shared glance log is intact. The private dataset `ezzzzzx/pdp-code-h27b` is up, and the smoke kernel `ezzzzzx/h27b-smoke` (seed 21, T = 32) is running. Score kernels wait until that smoke checks out.

Smoke finished. I'll pull its output and check the eight runs before starting the score kernels.Smoke checks passed: one shared glance log, the stated traffic, and the two unfilled reservoirs match. Next I’ll start the three scoring kernels.Smoke on seed 21 at T = 32 passed the structural checks. The three scoring kernels are running, two seeds each, at T = 512.Scoring is in progress on `ezzzzzx/h27b-score-a`, `ezzzzzx/h27b-score-b`, and `ezzzzzx/h27b-score-c`. I'll assemble the means table when all three finish.

All three score kernels finished. I'll download the runs, check them, and assemble the results.The kernel times in the report are full floats. I'll round them to three decimals, matching H27, and rebuild the report.48 of 48 scored runs are in `results/h27b/RESULTS.md`. Seeds 21–26, T = 512. Bars are not scored. Replays per segment is total segment-draws divided by sampling units ever admitted: overlapping windows for `hist*`, Algorithm R admissions for `res*`.

| Condition | W | M | traffic | replays per segment |
|---|---:|---:|---:|---:|
| nostore | 1.814979 ± 0.085205 | 0.000 | 0 | 0 |
| hist_e1 | 1.457255 ± 0.171641 | 176695.457 | 8 | 0.127450 ± 0.000000 |
| hist_e2 | 1.507376 ± 0.069611 | 176695.457 | 4 | 0.063725 ± 0.000000 |
| hist_e4 | 1.641399 ± 0.134922 | 176695.457 | 2 | 0.031863 ± 0.000000 |
| res77_e4 | 1.652755 ± 0.094558 | 26573.340 | 2 | 4.579965 ± 0.088712 |
| res77_e8 | 1.740002 ± 0.066214 | 26573.340 | 1 | 2.289983 ± 0.044356 |
| res308_e1 | 1.427976 ± 0.121325 | 106293.361 | 8 | 8.822824 ± 0.117797 |
| res308_e4 | 1.710623 ± 0.149977 | 106293.361 | 2 | 2.205706 ± 0.029449 |

Per-seed W.

| Seed | nostore | hist_e1 | hist_e2 | hist_e4 | res77_e4 | res77_e8 | res308_e1 | res308_e4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | 1.809581 | 1.303092 | 1.470141 | 1.533133 | 1.595977 | 1.725806 | 1.275894 | 1.599363 |
| 22 | 1.905519 | 1.681969 | 1.552391 | 1.918919 | 1.851227 | 1.849063 | 1.605533 | 1.947553 |
| 23 | 1.731281 | 1.254398 | 1.441056 | 1.548732 | 1.583180 | 1.665959 | 1.301929 | 1.607828 |
| 24 | 1.800010 | 1.356197 | 1.478868 | 1.631214 | 1.626377 | 1.702755 | 1.404874 | 1.647041 |
| 25 | 1.940089 | 1.683274 | 1.642174 | 1.677444 | 1.677053 | 1.808238 | 1.559073 | 1.890672 |
| 26 | 1.703396 | 1.464599 | 1.459624 | 1.538953 | 1.582718 | 1.688194 | 1.420555 | 1.571278 |

Kernels, all private CPU, version 1, no reruns. Elapsed time is compute wall from each manifest. Dataset `ezzzzzx/pdp-code-h27b`.

| Id | Seeds | T | Elapsed seconds |
|---|---|---:|---:|
| ezzzzzx/h27b-smoke | 21 | 32 | 26.387 |
| ezzzzzx/h27b-score-a | 21, 22 | 512 | 301.935 |
| ezzzzzx/h27b-score-b | 23, 24 | 512 | 287.310 |
| ezzzzzx/h27b-score-c | 25, 26 | 512 | 294.219 |

Local match before the scored runs: at seed 21, T = 8, `hist_e1` reproduced H27 `hist1` and `nostore` reproduced H27 `nostore` (W difference 0). Nothing was committed.
