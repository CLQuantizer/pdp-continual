I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is H29b: three enwik8 arms that separate replay dose from learning rate. I'll read the H29 result, script, and Kaggle layout before changing anything.The local E10M hash matches H29. I'll turn that script into `h29b_enwik8.py` with the dose and learning-rate flags, leaving the training step unchanged.The training step stays the H29 copy. I'll replace only the entry points so each arm is one learning rate and one replay batch.Smoke matched exactly: both paths scored 16784.078925371294 bits on 4096 bytes. Two kernels are on Tesla T4s; Kaggle allows two GPU sessions, so the third waits for a free slot. I'll poll until they finish, then start it.All three kernels finished. I'll read their outputs and logs, then write the results with no verdict.The no-replay slices match H29's measured prefix exactly. I'll record the log-clock times and assemble the report from those files.H29b is in `results/h29b/RESULTS.md` and `results/h29b/results.json`. Numbers only. Nothing was committed.

The CPU smoke on the first 4,096 bytes matched exactly. The H29 model path and `h29b_enwik8.py --lr 1e-3 --replay-batch 32` both scored **16784.078925371294** bits (absolute difference 0). The three arms on those 4,096 bytes were finite: `nr_1e3` 19611.760897, `r8_1e3` 17019.407864, `nr_3e4` 23836.498041.

E10M (10,000,000 bytes, sha256 `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`), live −log2 p only, Tesla T4:

| Arm | lr | Replay batch | Bits/byte | Replay steps | Traffic | Wall seconds | Minus xz -9e (2.176205) | Minus PPMd order 16 (1.837254) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nr_1e3 | 0.001 | 0 | 2.794439 | 0 | 0 | 1448.785 | 0.618234 | 0.957185 |
| r8_1e3 | 0.001 | 8 | 3.220097 | 78125 | 8 | 3849.497 | 1.043892 | 1.382843 |
| nr_3e4 | 0.0003 | 0 | 1.973638 | 0 | 0 | 1454.947 | −0.202567 | 0.136384 |

1 MB slice bits/byte:

- `nr_1e3`: 2.329297, 2.218160, 2.194801, 2.296691, 2.371822, 2.436663, 2.690608, 2.942345, 3.886367, 4.577638
- `r8_1e3`: 2.213050, 2.438259, 3.755486, 3.367998, 3.475320, 3.350599, 3.315432, 3.385663, 3.416307, 3.482853
- `nr_3e4`: 2.511647, 2.109053, 1.973250, 1.964439, 1.914245, 1.870552, 1.909113, 1.851397, 1.777629, 1.855059

Reported differences, not scored here: **r8_1e3 − nr_1e3 = 0.425657**. **nr_3e4 − nr_1e3 = −0.820801**. Lowest E10M bits/byte is `nr_3e4` at 1.973638. Its slice 0 is 2.511647 and its slice 9 is 1.855059 (slice 9 − slice 0 = −0.656588).

`nr_1e3` slice bits on the first 2 MB equal H29's no-replay slices: 2329297.4129243223 and 2218159.93700599. Both differences are 0.

Kernels, each version 1, each run once. Kaggle allows two GPU sessions, so `ezzzzzx/h29b-nr-3e4` was pushed after `ezzzzzx/h29b-nr-1e3` completed. Elapsed seconds are the session log clock. GPU hours are that clock divided by 3600.

| Kernel | Elapsed seconds | GPU hours |
|---|---:|---:|
| ezzzzzx/h29b-nr-1e3 | 1482.705 | 0.4119 |
| ezzzzzx/h29b-r8-1e3 | 3885.196 | 1.0792 |
| ezzzzzx/h29b-nr-3e4 | 1507.173 | 0.4187 |

Total GPU hours: **1.9097**.
