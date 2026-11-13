I'll read the project instructions and the handoff file, then follow the handoff exactly.The handoff is M7c-W3: longer lives and more seeds for curiosity on the bank cortex. I'll read the M7b results and code, then run the smoke test before any kernel.M7b scored curiosity on six seeds at L = 2,048. Next I’ll read the runner and the M7b kernel layout so the smoke test and kernels match that code.M7b’s L = 64 `rand_dwell_s` score is on disk, so the smoke test can compare against it. I’ll add a runner that only imports `m7b_w3`, then run that smoke test before any kernel.Smoke passed. The L = 64 score matches M7b within 1.9e-7, all three controllers are finite, and one kernel is about 13 minutes, so both lives stay in a single kernel. Next I’ll check the staged kernels and upload them.All 12 kernels finished, each under 12 minutes. I'll assemble the results and check the tables.M7c-W3 is in `results/m7cw3/RESULTS.md` and `results/m7cw3/results.json`. Numbers only. Twelve private CPU kernels, `ezzzzzx/m7c-w3-s42` through `ezzzzzx/m7c-w3-s53`, each ran `rand_dwell_s`, `oracle7_s`, and `prog_chan_s` at L = 2,048 and L = 4,096. No kernel was rerun.

The smoke test passed in 109.3 s. `rand_dwell_s` on seed 36 at L = 64 reproduced M7b's W of 13.654479 to 1.891e-7. All three controllers returned a finite W on seed 42 at L = 64. A 512-decision `prog_chan_s` life took 80.45 s, and the estimated kernel wall was 13.4 minutes, so both lives stayed in one kernel.

Kernel walls were 333.8 s to 716.5 s. Sleep share was 0.199219 on every scored run.

**L = 2,048.** Mean W ± std (ddof = 0): `rand_dwell_s` 0.663905 ± 0.089948, `oracle7_s` 0.635424 ± 0.119076, `prog_chan_s` 0.671285 ± 0.146721. Mean W9: 1.663756 ± 0.097236, 2.362181 ± 0.133079, 1.330574 ± 0.120970. Forgetting: 0.141227, 0.156371, 0.206073 (n = 84 each).

| gap | mean | std ddof=0 | left lower |
|---|---:|---:|---:|
| prog_chan_s − rand_dwell_s | 0.007381 | 0.202312 | 6 of 12 |
| oracle7_s − rand_dwell_s | -0.028481 | 0.158523 | 9 of 12 |
| prog_chan_s − oracle7_s | 0.035861 | 0.184263 | 6 of 12 |

**L = 4,096.** Mean W: `rand_dwell_s` 0.475578 ± 0.075499, `oracle7_s` 0.455629 ± 0.042635, `prog_chan_s` 0.492275 ± 0.089801. Mean W9: 1.361983 ± 0.092518, 2.266685 ± 0.084923, 1.132486 ± 0.082781. Forgetting: 0.138138, 0.081172, 0.146783 (n = 84 each).

| gap | mean | std ddof=0 | left lower |
|---|---:|---:|---:|
| prog_chan_s − rand_dwell_s | 0.016697 | 0.134388 | 6 of 12 |
| oracle7_s − rand_dwell_s | -0.019949 | 0.095110 | 6 of 12 |
| prog_chan_s − oracle7_s | 0.036646 | 0.079810 | 4 of 12 |

Code is `m7c_w3.py`, which imports `m7b_w3` and does not modify it. The private dataset is `ezzzzzx/pdp-code-m7cw3`. Nothing was committed.
