I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is a Kaggle run: summarize H34a, dispatch H34a2, and wait for M11. I'll read the three specs and the current result state before starting.H34a kernels are already fetched, M11 has outputs on disk, and H34a2 still needs a dispatch. I'll check the scripts and whether anything is already running.M11's state file marks all 24 jobs complete. I'll confirm the dispatcher is finished and the outputs are whole, then summarize H34a.M11's 24 kernels are fetched and the dispatcher has exited. I'll add the scoring tables, then start H34a2 with an account-wide cap of 5 running kernels.H34a and M11 are written up. H34a2 is on Kaggle: 3 kernels running, 17 still queued, with the account held at or below 5 running kernels.

**H34a** (`results/h34a/RESULTS.md`). Full run, seeds 0 and 1. Count is how many seeds are at or under the threshold.

| Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds ≤ threshold |
|---|---:|---:|---:|---:|---:|
| left_4 − left_1 | 6661.116857838 | 6759.856341726 | 6710.486599782 | −0.10 | 0 |
| left_4 − plain_4 | −0.314073164 | −0.318196691 | −0.316134927 | −0.05 | 2 |
| left_4 − wide_1 | 6661.254413021 | 6759.986404673 | 6710.620408847 | −0.03 | 0 |

Tail 50–300 KB, same three gaps:

| Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds ≤ threshold |
|---|---:|---:|---:|---:|---:|
| left_4 − left_1 | 0.033322393 | 0.016543053 | 0.024932723 | −0.10 | 0 |
| left_4 − plain_4 | −0.072575556 | −0.075496954 | −0.074036255 | −0.05 | 2 |
| left_4 − wide_1 | 0.159412864 | 0.134701132 | 0.147056998 | −0.03 | 0 |

**M11.** The detached dispatcher had already finished: 24/24 kernels fetched. Scoring is at the top of `results/m11w6/RESULTS.md`.

| Quantity | Mean | Threshold | Count |
|---|---:|---:|---:|
| age_sleep_cf − age_sleeparm | −0.008773506 | −0.05 | 4 of 12 seeds ≤ −0.05 |
| age_sleep_cf − age_fixed20 | 0.009248371 | +0.03 on the mean | 8 seeds ≤ +0.03 |
| age_fixed20 − age_nosleep | −0.061048327 | gap < 0 | 7 |

Sleep-share means: age_fixed20 0.199218750, age_sleeparm 0.058919271, age_sleep_cf 0.032226562, age_nosleep 0.

**H34a2.** The code dataset is up, and `ezzzzzx/h34a2-plain-1-s0`, `plain-1-s1`, and `plain-2-s0` are running. The dispatcher polls in the foreground and will call `summarize.py` when all 20 outputs are in. That write-up gets the same kind of scoring table (sum_4 − sum_1, sum_4 − sumown_4, sum_4 − wide_1, and plain_4 − plain_1).

The H34a2 dispatcher exited cleanly. I'll check that all 20 kernels were fetched and that the scoring table is in the write-up.H34a2 is finished. All 20 kernels completed on the first attempt, and `results/h34a2/RESULTS.md` is written.

Full run, 300,000 bytes. Count is how many of the two seeds are at or under the threshold.

| Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds ≤ threshold |
|---|---:|---:|---:|---:|---:|
| sum_4 − sum_1 | 0.389080231 | 0.356559125 | 0.372819678 | −0.10 | 0 |
| sum_4 − sumown_4 | −3.077721767 | −3.075131551 | −3.076426659 | −0.05 | 2 |
| sum_4 − wide_1 | 0.415876527 | 0.375938970 | 0.395907748 | −0.03 | 0 |
| plain_4 − plain_1 | 0.058547290 | 0.044877671 | 0.051712481 | −0.10 | 0 |

The same check matched each kernel file to the assembled numbers. The longest kernel ran 3,784 seconds.

H34a and M11 were already written up, with their scoring tables at the top of `results/h34a/RESULTS.md` and `results/m11w6/RESULTS.md`.
