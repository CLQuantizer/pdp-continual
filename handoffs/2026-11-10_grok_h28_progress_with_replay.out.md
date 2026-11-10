I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is H28: test whether history replay stops the curiosity noise leak. I'll read the cited results and scripts, then implement and run only the allowed files.The experiment is the M4c history-replay machine plus H25a progress controllers, scored on fresh seeds. I'll read the imported scripts and the last Kaggle layout before writing H28.The machine is H27b’s one-replay history dose on the M4c cortex, with H25a’s progress reward measured after that replay. I’ll add the scorer and the private CPU kernels, then smoke-check before the scored seeds.All three score kernels finished. I'll fetch the runs, check them, and write the results.Auto-compacting conversation (80% full)...
Conversation compacted.
The score kernels finished earlier. I'll check whether the fetch and report assembly completed, then write the numbers if they did.Fetch and assembly finished with no check failures. I'll read the report and the handoff so the summary matches the written numbers.H28 is complete. `results/h28/RESULTS.md` and `results/h28/results.json` hold 30 of 30 score runs. Checks are empty. No kernel was rerun. Bars are not scored here.

T = 512, seeds 27–32. W is the mean held-out bits over the 8 learnable channels. Population standard deviation, ddof = 0.

| Condition | W | N share | Z share | Wall seconds (sum) |
|---|---:|---:|---:|---:|
| rand | 1.396674 ± 0.033640 | 0.106120 | 0.111979 | 329.218 |
| rand8 | 1.340445 ± 0.069688 | 0.001953 | 0.001953 | 323.392 |
| gmc_prop | 1.299085 ± 0.055464 | 0.165039 | 0.031901 | 348.608 |
| prog_prop | 1.371173 ± 0.061057 | 0.087891 | 0.073242 | 2924.143 |
| prog_cm_prop | 1.383063 ± 0.070701 | 0.095052 | 0.070638 | 2943.200 |

Per-seed W (seeds 27, 28, 29, 30, 31, 32):

| Condition | 27 | 28 | 29 | 30 | 31 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| rand | 1.449380 | 1.369281 | 1.436754 | 1.365979 | 1.386779 | 1.371870 |
| rand8 | 1.399797 | 1.274162 | 1.407454 | 1.219136 | 1.362645 | 1.379475 |
| gmc_prop | 1.366375 | 1.347949 | 1.272253 | 1.218202 | 1.340932 | 1.248799 |
| prog_prop | 1.425407 | 1.383711 | 1.392091 | 1.283301 | 1.296729 | 1.445796 |
| prog_cm_prop | 1.441087 | 1.328149 | 1.444768 | 1.311318 | 1.301001 | 1.472057 |

Mean gaps:

| Gap | Mean ± std |
|---|---:|
| prog_prop − rand | −0.025501 ± 0.056694 |
| prog_cm_prop − rand | −0.013611 ± 0.059297 |
| prog_prop − gmc_prop | 0.072088 ± 0.074100 |
| prog_cm_prop − gmc_prop | 0.083978 ± 0.094455 |
| prog_prop − rand8 | 0.030728 ± 0.057862 |
| prog_cm_prop − rand8 | 0.042618 ± 0.051612 |

Mean N and Z shares, prog_prop: 0.087891 and 0.073242. prog_cm_prop: 0.095052 and 0.070638. gmc_prop: 0.165039 and 0.031901. rand8 is 0.001953 on both (1/512). Replay steps are 512 and traffic is 8.0 on every score row.

Mean controller reward on visits, prog_prop: N 0.020918, Z 0.083098. prog_cm_prop: N 0.019595, Z 0.061924. delta_mean over every glance, prog_prop: N 0.005395, Z 0.007396. prog_cm_prop: N 0.004933, Z 0.005864.

Kernels, all version 1, private CPU, rerun false:

| Id | Elapsed seconds |
|---|---:|
| ezzzzzx/h28-smoke | 43.623 |
| ezzzzzx/h28-score-a | 1471.844 |
| ezzzzzx/h28-score-b | 1505.749 |
| ezzzzzx/h28-score-c | 1528.748 |

Dataset `ezzzzzx/pdp-code-h28`. Smoke seed 27, T = 32, five conditions, wall 43.623 s. Local match before the scored kernels: seed 21, T = 8, rand W − h27b hist_e1 W = 0.000e+00, glance checksum 796030179972200500 shared, replay steps 8.
