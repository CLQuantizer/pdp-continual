I'll read the project instructions and the handoff document first, then follow the handoff exactly.The handoff is a Kaggle kernel split: one seed is over the time cap, so I'll read the snapshot spec and current packaging before changing only the dispatcher.Smoke estimates put one seed at about 122 minutes. I'll read the dispatcher and kernel layout so the split stays inside packaging only.One seed is about 122 minutes, over the 90-minute cap. I'll split each seed into three kernels — `rand`+`oracle`+`rho`, then each `prog_snap_k` — and gate the longest one at 85 minutes.The baseline kernel estimates at 81.4 minutes, under the 85-minute cap. Eighteen kernels are packaged. Next I’ll run the dispatcher in the foreground until they finish. Other experiments are already using account slots, so this one will wait for a free slot.The dataset is up, but the first slot check failed because one kernel’s status never came back. I’ll inspect that response and then resume.The slot check died on a Kaggle 429, and the dispatch state is still all pending. I’ll see whether the kernel list already includes status, then make rate limits wait instead of aborting.The kernel list is 313 rows, and the slot check only read the first 100 before a rate limit aborted it. I’ll page through the full list, cache finished kernels, and wait out 429s.The dispatcher returned after the last push. I’ll check whether every kernel actually finished and whether the summary was written.All 18 split kernels finished and validated. The scoring table is at the top of `results/h35b2/RESULTS.md`.

The checked kernel `ezzzzzx/h35b2-s6-base` (`rand`+`oracle`+`rho`, plus the rho reference build) estimates at **81.4 minutes**, under the 85-minute cap. Each seed is three private CPU kernels: that baseline bundle, `prog_snap_5`, and `prog_snap_20`. Selection on seeds 6–8 chooses **k=20** (mean T_clean 4.205 versus 4.277 for k=5). On scoring seeds 9–11, both bars are **0 of 3**.

| Seed | Phase | k | Chosen | Oracle − rand | Prog_snap − rand | 0.5·(oracle − rand) | Prog_snap − rho | ≤ half | ≤ −0.02 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | selection | 5 | 0 | −0.238 | 0.782 | −0.119 | 0.897 | 0 | 0 |
| 6 | selection | 20 | 1 | −0.238 | 0.894 | −0.119 | 1.009 | 0 | 0 |
| 7 | selection | 5 | 0 | −0.159 | 0.267 | −0.079 | 0.351 | 0 | 0 |
| 7 | selection | 20 | 1 | −0.159 | 0.311 | −0.079 | 0.396 | 0 | 0 |
| 8 | selection | 5 | 0 | −0.163 | 0.833 | −0.081 | 0.893 | 0 | 0 |
| 8 | selection | 20 | 1 | −0.163 | 0.459 | −0.081 | 0.520 | 0 | 0 |
| 9 | scoring | 5 | 0 | −0.166 | 0.800 | −0.083 | 0.869 | 0 | 0 |
| 9 | scoring | 20 | 1 | −0.166 | 1.239 | −0.083 | 1.308 | 0 | 0 |
| 10 | scoring | 5 | 0 | −0.161 | 0.694 | −0.080 | 0.747 | 0 | 0 |
| 10 | scoring | 20 | 1 | −0.161 | 0.203 | −0.080 | 0.257 | 0 | 0 |
| 11 | scoring | 5 | 0 | −0.177 | 0.169 | −0.088 | 0.256 | 0 | 0 |
| 11 | scoring | 20 | 1 | −0.177 | 0.556 | −0.088 | 0.643 | 0 | 0 |

| Threshold (k=20, seeds 9–11) | Count | Evaluated | Required |
|---|---:|---:|---:|
| bar1 | 0 | 3 | 2/3 |
| bar2 | 0 | 3 | 2/3 |

The longest kernel that actually ran was `s8-base` at 25.0 minutes. The model, data, seeds, and bars were not changed. Nothing was committed.
