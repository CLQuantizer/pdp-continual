I'll read the project instructions and the handoff note, then follow that handoff exactly.The handoff is a Kaggle scoring run of the M4 machine on fresh seeds. I'll read the M4b spec, results, and code before writing anything.M4b is the base. Next I’ll read how its conditions are wired and how earlier Kaggle kernels were set up.The two new conditions have to ride on M4b’s existing run path. I’ll trace the import tree and the Kaggle CLI so the kernel only ships the scripts it needs.M4c reuses M4b’s run path and adds `rand8_full` and `gmc_nostore`. I’ll write that scorer and the private CPU kernels, then smoke-check before the scored runs.The local dispatch checks passed. The private dataset is up, and the smoke kernel `ezzzzzx/m4c-smoke` is running. I’ll check its numbers before starting the scored kernels.

The smoke kernel finished. I’ll fetch its output and check the six runs before launching the scored kernels.Smoke checks out: the six runs share the right logs, the budget holds, and the local and Kaggle `nostore` streams match. The W difference is 2.0×10⁻⁷. Starting the three scored kernels.Smoke on seed 9 at T = 32 is sane, so the three scored kernels are running.

The local and Kaggle `nostore` logs match. W differs by 2.0×10⁻⁷ (torch 2.15.0+cpu here, 2.11.0+cpu on Kaggle). Kernels: `ezzzzzx/m4c-score-a` (seeds 9–10), `ezzzzzx/m4c-score-b` (11–12), `ezzzzzx/m4c-score-c` (13–14). I’ll assemble the tables when they finish.

All three scored kernels finished. I’ll fetch the runs, check them, and assemble the results.M4c is scored. 36 of 36 runs at T = 512 on seeds 9–14 are in `results/m4c/RESULTS.md` and `results/m4c/results.json`. No verdict. Nothing was rerun.

Means (population standard deviation):

| Condition | W_mix | W_gru | W_F_mix | W_F_gru | N share | Z share |
|---|---:|---:|---:|---:|---:|---:|
| nostore | 1.779544 ± 0.079385 | 1.779544 ± 0.079385 | 3.655393 ± 0.094386 | 3.655393 ± 0.094386 | 0.104492 | 0.093750 |
| hist_rep | 1.538178 ± 0.054983 | 1.538178 ± 0.054983 | 3.161189 ± 0.206388 | 3.161189 ± 0.206388 | 0.104492 | 0.093750 |
| rand_full | 1.521271 ± 0.042844 | 1.538178 ± 0.054983 | 3.229005 ± 0.179013 | 3.161189 ± 0.206388 | 0.104492 | 0.093750 |
| rand8_full | 1.271508 ± 0.086590 | 1.319222 ± 0.073408 | 2.750972 ± 0.132108 | 2.849994 ± 0.181191 | 0.001953 | 0.001953 |
| gmc_nostore | 1.283505 ± 0.124192 | 1.283505 ± 0.124192 | 2.518037 ± 0.097203 | 2.518037 ± 0.097203 | 0.173177 | 0.024414 |
| gmc_full | 1.282556 ± 0.109499 | 1.283505 ± 0.124192 | 2.596109 ± 0.182728 | 2.518037 ± 0.097203 | 0.173177 | 0.024414 |

Per-seed W_mix:

| Seed | nostore | hist_rep | rand_full | rand8_full | gmc_nostore | gmc_full |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.646120 | 1.544490 | 1.520351 | 1.327357 | 1.325122 | 1.317984 |
| 10 | 1.805427 | 1.610089 | 1.581312 | 1.253544 | 1.175160 | 1.167289 |
| 11 | 1.905811 | 1.466074 | 1.519492 | 1.201645 | 1.264368 | 1.246128 |
| 12 | 1.772985 | 1.594558 | 1.564556 | 1.152186 | 1.134296 | 1.190767 |
| 13 | 1.732967 | 1.542964 | 1.486250 | 1.273065 | 1.281021 | 1.272559 |
| 14 | 1.813952 | 1.470892 | 1.455662 | 1.421254 | 1.521066 | 1.500608 |

N and Z shares. `nostore`, `hist_rep`, and `rand_full` share one log per seed. `gmc_nostore` matches `gmc_full`. `rand8_full` is 0.001953 / 0.001953 on every seed (the first pass only).

| Seed | rand N | rand Z | gmc N | gmc Z |
|---:|---:|---:|---:|---:|
| 9 | 0.125000 | 0.097656 | 0.203125 | 0.023438 |
| 10 | 0.099609 | 0.083984 | 0.169922 | 0.019531 |
| 11 | 0.091797 | 0.091797 | 0.166016 | 0.021484 |
| 12 | 0.105469 | 0.101562 | 0.158203 | 0.029297 |
| 13 | 0.093750 | 0.097656 | 0.173828 | 0.031250 |
| 14 | 0.111328 | 0.089844 | 0.167969 | 0.021484 |

Determinism, `nostore` seed 9 T = 32. Glance and observation checksums are equal (`2418577838273500094`, `4494073894073903178`). Kaggle minus local W_mix is 2.022×10⁻⁷, W_gru the same, W_F_mix 5.858×10⁻⁸. Local torch 2.15.0+cpu, numpy 2.5.4. Kaggle torch 2.11.0+cpu, numpy 2.1.3.

Kernels, all private, CPU, no internet, dataset `ezzzzzx/pdp-code-m4c`. Elapsed time is compute wall. No rerun.

| Id | Seeds | T | Elapsed seconds | Finished (UTC) |
|---|---|---:|---:|---|
| ezzzzzx/m4c-smoke | 9 | 32 | 22.4 | 08:13:10 |
| ezzzzzx/m4c-score-a | 9, 10 | 512 | 447.5 | 08:22:08 |
| ezzzzzx/m4c-score-b | 11, 12 | 512 | 376.1 | 08:21:03 |
| ezzzzzx/m4c-score-c | 13, 14 | 512 | 420.5 | 08:21:51 |

The score-kernel watcher exited cleanly after all three kernels had already been fetched. The M4c tables in `results/m4c/RESULTS.md` are unchanged.
