I'll start by reading the project instructions and the H34e handoff so I follow that task exactly.The handoff is to dispatch the H34e Kaggle kernels, retry once if needed, then summarize. I'll read the H34e README and dispatcher before starting the run.Smoke artifacts are already on disk. I'll check that the gate passed and that nothing else holds the dispatcher lock before starting the run.Smoke passed and the dispatch lock is free. I'm starting the H34e dispatcher in the foreground and will watch it through to the summary.Two kernels are running and the account cap is holding. I'll wait for the first pair to finish before the dispatcher submits the next ones.The dispatcher finished with exit code 0. I'll run the summarizer and read the scoring table, kernel ids, and any failures.H34e finished. The dispatcher exited 0 on the first run, and `kaggle/h34e/summarize.py` left `results/h34e/RESULTS.md` unchanged. All 8 kernels completed on attempt 1. There were no failures.

| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|
| Tail | distill_4 − plain_4 | 0.048667566 | 0.045259578 | 0.046963572 | ≤ -0.03 | 0/2 | 2/2 |
| Tail | distill_4_top − plain_4 | 0.033354078 | 0.037904497 | 0.035629287 | report only | — | 2/2 |
| Full | distill_4 − plain_4 | 0.902894845 | 0.982561465 | 0.942728155 | report only | — | 2/2 |
| Full | distill_4_top − plain_4 | 0.603798789 | 0.626472682 | 0.615135736 | report only | — | 2/2 |

8 of 8 kernels scored. Smoke is true. Tail is bytes 40000–180000.

Kernel ids:

- `ezzzzzx/h34e-plain-4-s0`
- `ezzzzzx/h34e-plain-4-s1`
- `ezzzzzx/h34e-distill-4-s0`
- `ezzzzzx/h34e-distill-4-s1`
- `ezzzzzzx/h34e-distill-4-top-s0`
- `ezzzzzx/h34e-distill-4-top-s1`
- `ezzzzzx/h34e-distill-1-s0`
- `ezzzzzx/h34e-distill-1-s1`

No retry was required. `plain_4` matched the H34b reference on both seeds (recorded tail and full differences 0, limit 0.000001).
