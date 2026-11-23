I'll read the project instructions and the handoff, then follow the handoff exactly.

The handoff is a Kaggle dispatch: load the token, run H34h, then summarize. I'll read the kernel README and dispatcher so I know the exact commands.

The dispatcher polls until the kernels finish, then the summarizer writes the table. I'll check the script and whether a dispatch is already in progress.

Smoke passed and nothing is dispatched yet. Estimated walls are about 47–73 minutes per kernel, with dense_8 split. I'll start the dispatcher in the foreground and keep the account under five running kernels.

The code dataset is up and the first kernel is running. The account scan is slow, so I’ll watch for a failed push while the next two kernels start.

H34h is dispatching. The code dataset `ezzzzzx/pdp-code-h34h` is ready, the account had no other active kernels, and `ezzzzzx/h34h-dense-8-s0` is running. The dispatcher is filling the other two slots under the cap of 3 H34h kernels and 5 account-wide.

Ten jobs are planned (dense_8 is split, so each seed has a part 2). Estimated segment walls are about 47–73 minutes. No failures so far. I will report the ablation table, bar gaps, kernel ids, and any failures after the run and the summarizer finish.

The dispatcher finished successfully. I’ll pull the kernel ids, confirm the summarizer ran, and read the ablation table.

H34h finished. The dispatcher exited 0 on the first run (about 3.3 h). All 10 kernels completed on the first attempt. The summarizer scored 8/8 runs. No failures.

**Ablation** (tail and full bits/byte; `plain_8` is NA and was not run):

| Depth | Arm | Tail s0 | Tail s1 | Tail mean | Full s0 | Full s1 | Full mean | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | plain_1 | 3.264204845 | 3.285754554 | 3.274979700 | 3.533424381 | 3.565700932 | 3.549562656 | h34f |
| 1 | zgate_1 | 3.264204845 | 3.285754554 | 3.274979700 | 3.533424381 | 3.565700932 | 3.549562656 | h34f |
| 1 | dense_1 | 3.264204845 | 3.285754554 | 3.274979700 | 3.533424381 | 3.565700932 | 3.549562656 | h34f |
| 4 | plain_4 | 3.091224982 | 3.091462939 | 3.091343961 | 3.745828744 | 3.769168530 | 3.757498637 | h34f |
| 4 | zgate_4 | 3.093923752 | 3.093068825 | 3.093496289 | 3.388026693 | 3.400398487 | 3.394212590 | h34f |
| 4 | dense_4 | 3.057473152 | 3.055238880 | 3.056356016 | 3.356227281 | 3.366784896 | 3.361506089 | h34h |
| 6 | plain_6 | 3.079927197 | 3.082687574 | 3.081307386 | 4.107748392 | 4.150529688 | 4.129139040 | h34f |
| 6 | zgate_6 | 3.078244907 | 3.077944571 | 3.078094739 | 3.374814145 | 3.386931698 | 3.380872921 | h34f |
| 6 | dense_6 | 3.016241787 | 3.016160666 | 3.016201226 | 3.320349905 | 3.330644108 | 3.325497006 | h34h |
| 8 | plain_8 | NA | NA | NA | NA | NA | NA | NA |
| 8 | zgate_8 | 3.069103730 | 3.068949378 | 3.069026554 | 3.366845428 | 3.378966923 | 3.372906175 | h34g |
| 8 | dense_8 | 2.989027699 | 2.991593491 | 2.990310595 | 3.295147634 | 3.308002105 | 3.301574869 | h34h |
| 6 | dense_6_ones | 3.010042782 | 3.011322334 | 3.010682558 | 3.997741283 | 4.019653114 | 4.008697198 | h34h |

**Bar gaps:**

| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|
| Tail | dense_8 − dense_6 | -0.027214088 | -0.024567175 | -0.025890631 | ≤ -0.02 | 2/2 | 2/2 |
| Full | dense_8 − dense_6 | -0.025202271 | -0.022642003 | -0.023922137 | report only | — | 2/2 |
| Tail | dense_6 − zgate_8 | -0.052861943 | -0.052788712 | -0.052825328 | ≤ -0.03 | 2/2 | 2/2 |
| Full | dense_6 − zgate_8 | -0.046495523 | -0.048322815 | -0.047409169 | report only | — | 2/2 |
| Tail | dense_6_ones − dense_6 | -0.006199005 | -0.004838332 | -0.005518669 | report only | — | 2/2 |
| Full | dense_6_ones − dense_6 | 0.677391378 | 0.689009006 | 0.683200192 | report only | — | 2/2 |
| Tail | dense_4 − zgate_4 | -0.036450600 | -0.037829945 | -0.037140273 | report only | — | 2/2 |
| Full | dense_4 − zgate_4 | -0.031799412 | -0.033613590 | -0.032706501 | report only | — | 2/2 |

Runs scored 8/8. Smoke true. Tail is bytes 40000–180000. `dense_6` matches H34g at 0.000000000 on full, tail, and first 40 KB for both seeds (limit 1e-6).

**Kernels** (all `complete`, attempts=1):

- `ezzzzzx/h34h-dense-8-s0`
- `ezzzzzx/h34h-dense-8-s0-part2`
- `ezzzzzx/h34h-dense-8-s1`
- `ezzzzzx/h34h-dense-8-s1-part2`
- `ezzzzzx/h34h-dense-4-s0`
- `ezzzzzx/h34h-dense-4-s1`
- `ezzzzzx/h34h-dense-6-ones-s0`
- `ezzzzzx/h34h-dense-6-ones-s1`
- `ezzzzzx/h34h-dense-6-s0`
- `ezzzzzx/h34h-dense-6-s1`

No failed kernels and no retry. The account stayed at or under 3 H34h kernels. No code, bars, or docs were edited, and nothing was committed.
