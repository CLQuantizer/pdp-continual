I'll read the project instructions and the handoff note, then follow the handoff exactly.The handoff is to dispatch the H34f Kaggle kernels, rerun once if they fail, then summarize. I'll read the runner notes and dispatch script before starting.The runner is a foreground Kaggle dispatch of ten kernels, then a summarizer. I'll check the dispatch state and the long-running-job rules before starting.Smoke passed and no H34f dispatch is in progress. I'll start the dispatcher in the foreground and watch it through the dataset upload and the first kernel pushes.Both `plain_4` kernels finished and were fetched. `zgate_4` is running in both seeds. I'll wait for that pair to finish.The dispatcher exited 0. I'll confirm all ten kernels, run the summarizer, and read the scoring table.All ten H34f kernels finished on the first attempt. The dispatcher exited 0, and the summarizer exited 0. No failures and no retry.

| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | zgate_4 − plain_4 | -0.357802051 | -0.368770043 | -0.363286047 | ≤ -0.10 | 2/2 | 2/2 |
| Tail | zgate_4 − plain_4 | 0.002698770 | 0.001605886 | 0.002152328 | ≤ 0.01 | 2/2 | 2/2 |
| Full | zgate_6 − plain_6 | -0.732934247 | -0.763597990 | -0.748266119 | report only | — | 2/2 |
| Tail | zgate_6 − plain_6 | -0.001682290 | -0.004743004 | -0.003212647 | report only | — | 2/2 |
| Full | zgate_6 − plain_1 | -0.158610236 | -0.178769234 | -0.168689735 | ≤ -0.05 | 2/2 | 2/2 |
| Tail | zgate_6 − plain_1 | -0.185959938 | -0.207809983 | -0.196884961 | report only | — | 2/2 |

10 of 10 kernels scored. Smoke is true. Tail is bytes 40,000–180,000.

| Kernel | Wall seconds |
|---|---:|
| ezzzzzx/h34f-plain-4-s0 | 2143.776 |
| ezzzzzx/h34f-plain-4-s1 | 2083.833 |
| ezzzzzx/h34f-zgate-4-s0 | 1947.506 |
| ezzzzzx/h34f-zgate-4-s1 | 2177.397 |
| ezzzzzx/h34f-zgate-6-s0 | 3255.960 |
| ezzzzzx/h34f-zgate-6-s1 | 3382.260 |
| ezzzzzx/h34f-plain-6-s0 | 3357.314 |
| ezzzzzx/h34f-plain-6-s1 | 3599.512 |
| ezzzzzx/h34f-plain-1-s0 | 312.436 |
| ezzzzzx/h34f-plain-1-s1 | 346.055 |

Each kernel used one attempt. There is no `results/h34f/failed/` directory. Plain-arm scores match H34b within 1e-6. The table is in `results/h34f/RESULTS.md`. No code, bars, or docs were edited, and nothing was committed.
