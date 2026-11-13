I'll read the project instructions and the handoff, then follow that handoff exactly.The handoff is M7b-W3: curiosity on the bank cortex with per-room progress and a fixed sleep dose. I'll read M7's results and code, then implement only what the spec allows.M7 scored whole-store progress, which went negative after the first quarter, so the bandit avoided rooms. M7b keeps M7's cortex and a fixed 0.20 sleep dose, and scores only the room just visited. I'll add that as a new script and leave `m7_w3.py` untouched.The smoke test is next. It has to pass on this machine before any Kaggle kernel is pushed.Smoke passed. `rand_dwell` matches M7 exactly, the L=512 sleep share is 0.203, and the kernel estimate is about 9 minutes. Next I’ll upload the private code dataset and start the CPU kernels.Four kernels are running. I’ll keep the pool at four and start seeds 40 and 41 as slots free up.All six kernels finished. I’ll download their outputs and write the results.M7b-W3 is in `results/m7bw3/RESULTS.md` and `results/m7bw3/results.json`. Numbers only. The bars are unscored. Nothing was committed.

Seeds 36–41, L = 2048. Every `_s` controller sleeps 408 decisions (share 0.199219), in visits of 8. `rand_dwell` does not sleep.

**W** (mean ± std, ddof = 0):

| controller | W | W9 |
|---|---:|---:|
| rand_dwell | 0.928802 ± 0.133795 | 1.802802 ± 0.091011 |
| rand_dwell_s | 0.677641 ± 0.076008 | 1.700954 ± 0.090345 |
| oracle7_s | 0.575707 ± 0.045063 | 2.334511 ± 0.067212 |
| prog_total_s | 0.701528 ± 0.133404 | 1.509084 ± 0.180536 |
| prog_chan_s | 0.610735 ± 0.097388 | 1.354338 ± 0.075993 |

**Gaps on W:**

| gap | mean ± std | left lower |
|---|---:|---:|
| prog_chan_s − rand_dwell_s | −0.066906 ± 0.162741 | 3 of 6 |
| prog_chan_s − prog_total_s | −0.090793 ± 0.145821 | 5 of 6 |
| prog_chan_s − oracle7_s | +0.035028 ± 0.102717 | 2 of 6 |
| rand_dwell_s − rand_dwell | −0.251161 ± 0.144957 | 6 of 6 |

**Forgetting** (exam bits/symbol, end minus the snapshot after the last dwell): rand_dwell 0.356478, rand_dwell_s 0.170689, oracle7_s 0.083750, prog_total_s 0.137259, prog_chan_s 0.189323. Each mean is over 42 seed-channel pairs.

`prog_chan_s` mean per-channel room reward by quarter: 58.908779, 28.849230, 21.864674, 23.580554.

Smoke on this machine passed in 126.2 s. `rand_dwell` at seed 30, L = 64, matches `m7_w3` with absolute difference 0 (W 9.92232653). `rand_dwell_s` at seed 36, L = 512, slept 104/512 = 0.203125. The kernel estimate was 514 s.

Private CPU kernels, no GPU, no internet, dataset `ezzzzzx/pdp-code-m7bw3`. Wall times: s36 230 s, s37 538 s, s38 376 s, s39 439 s, s40 396 s, s41 280 s.
