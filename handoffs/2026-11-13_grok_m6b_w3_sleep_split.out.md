I'll start by reading the project instructions and the handoff so I follow them exactly.The handoff is M6b-W3: split sleep replay and exam stores. I'll read the prior result and the experiment code before changing anything.M6-W3 scored sleep on the same reservoirs it replayed. I'll read the kernel layout and the rest of the runner so the split stores match that protocol.M6 scored sleep on the same reservoirs it replayed. I'll add disjoint exam and replay stores, then smoke-test before any kernel.Smoke passed, and the upper bound for one six-seed kernel is 46 minutes. I'll package the code and push the private CPU kernel.The kernel is running. I'll poll it until it finishes, with the 90-minute cap.M6b-W3 is in `results/m6bw3/RESULTS.md` and `results/m6bw3/results.json`. Replay and exam are disjoint reservoirs of 2,048 pairs per channel. Progress is scored on the exam stores. No verdict.

At L = 2,048 on W, mean over seeds 24–29:

| gap | mean | std ddof=0 | left lower |
|---|---:|---:|---:|
| prog_sleep − prog_total | +0.011602 | 0.019301 | 1 of 6 |
| prog_sleep − fixed_sleep | +0.011662 | 0.032154 | 3 of 6 |
| prog_total − prog | −0.067355 | 0.055231 | 5 of 6 |

W9 at L = 2,048: prog 1.981437, prog_total 1.929705, prog_sleep 1.938560, fixed_sleep 1.929472.

Sleep share for `prog_sleep` at L = 2,048 is 0.117513. By quarter: 0.111979, 0.085938, 0.072917, 0.199219. Sleep versus room reward by quarter, mean of per-seed means: sleep 9.334, 5.334, 5.105, 3.296; room 13.404, 18.175, 10.035, 3.434.

At L = 512, W9 is prog 2.740339, prog_total 3.008818, prog_sleep 2.995039, fixed_sleep 2.991444. prog_total − prog on W is +0.344053. Sleep share is 0.111979, by quarter 0.093750, 0.052083, 0.197917, 0.104167.

Smoke passed. `prog` on seed 12 at L = 512 matches the M5b W of 2.742358251650. After the 256-decision life, no stream index is in both stores, and sleep drew 0 exam pairs. The same draw count is 0 on every scored life. By L = 2,048 most channels sit at the 2,048 cap in both stores. N is the channel that stays under the cap.

Kernel `ezzzzzx/m6b-w3` (CPU, private, no GPU, no internet) finished in 1688 s. Smoke took 53.4 s. Code dataset is `ezzzzzx/pdp-code-m6bw3`. Nothing was committed.
