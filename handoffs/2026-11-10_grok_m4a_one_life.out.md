I'll read the project instructions and the handoff, then follow that handoff exactly.The handoff is M4a: one life that stores, replays, and allocates glances. I'll read the milestone notes and the scripts it builds on before writing anything.M4 joins the utility store with the H25 controllers. I'll read those implementations next so the new script can import them unchanged.The machine is the H25 controller plus the H24g utility store, on the ten-channel world. I'll write that as one script, check it on a short run, then start the scored lives one at a time.Smoke fits the budget: about a minute per condition at T=32, so the scored runs should land near an hour. Slope goes last so a late cut cannot drop the store or the oracle.All 15 scored lives finished in about 40 minutes. I'll check the tables against the run log before writing them up.M4a is in `results/m4a/RESULTS.md`. All 15 scored lives finished. Nothing was dropped. Score wall was 2389 s, 2463 s including the smoke run.

W is held-out bits/symbol of the frozen mixture on a fresh 2048-symbol continuation of each learnable channel. Means over seeds 6, 7 and 8:

| Condition | W | W_F | N share | Z share | Peak M/N |
|---|---:|---:|---:|---:|---:|
| rand | 1.873 | 3.498 | 9.0% | 9.5% | 0.902 |
| rand8 | 1.723 | 3.042 | 0.2% | 0.2% | 0.859 |
| gmc_prop | 1.717 | 2.848 | 14.4% | 3.0% | 1.621 |
| slope_prop | 1.906 | 3.313 | 9.3% | 8.2% | 0.885 |
| rand_nostore | 1.791 | 3.637 | 9.0% | 9.5% | 0 |

`gmc_prop` is the lower of the two controllers on the mean. Its W minus rand is −0.062, −0.162, −0.244. Its N shares are 13.1%, 13.7%, 16.4%; its Z shares are 3.1%, 2.5%, 3.3%.

W(rand) minus W(rand_nostore) is +0.051, −0.035, +0.228 (mean +0.081). W_F(rand) minus W_F(rand_nostore) is +0.045, −0.229, −0.233 (mean −0.139).

W(gmc_prop) minus W(rand8) is +0.056, +0.085, −0.159. The means are 1.717 and 1.723.

`rand` and `rand_nostore` drew the same glances. `rand8` touches N and Z only on the opening pass (1/512 each). Every store run replayed twice per glance. The 32-glance protection held memory above the budget on many glances, so peak M/N sits above 0.15 × log2(42) ≈ 0.809 on those runs.
