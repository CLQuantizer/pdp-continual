I'll read the project instructions and the handoff, then follow that handoff exactly.The handoff is a W2-1 headroom check: a cycle world, then a Kaggle CPU comparison before any curiosity bar. I'll read the W2 code and prior results first.W2 traps agents in room 8. I'll read the world and headroom code, plus the Kaggle recipe from the last run.W2 traps a no-reset agent in room 8. I'll add a room-8 key back to room 0, smoke-test that the worlds match until that key is typed, then run the headroom check.World checks passed: key 8 first completes at step 1311, the exit returns to room 0, and `oracle_deep` keeps room 8 on the even blocks. Next I’ll adapt the learner.The learner smoke path is in place. I'll extend assembly so the report includes capped probes, cycles, and the W2-0 comparison.The modules compile. I'll run the T=2048 smoke on all four policies before any kernel is pushed.Smoke passed. rand6 completes 25, 27 and 28 cycles locally, and `oracle_deep` never finishes the room-8 key. I'll upload the private dataset and start the three CPU kernels.W2-1 is in `results/w21/RESULTS.md` and `results/w21/results.json`. No verdict and no commit.

Room 8 now has a 3-digit key back to room 0. Smoke on world seed 0 passed: rand6 tokens match W2 until step 1311, when key_8 completes (prefix sha256 `6b7b6c19e33a35344df2af7462e67f783d8721292a860118031499bc211ce173`). Keys 0–7 then key_8 end in rooms 1 2 3 4 5 6 7 8 0, with s = q = s8 = 0. `oracle_deep` gives room 8 blocks 0 and 2 and does not complete key_8. At T = 2048, all four policies return finite bits and probes.

Kernels `ezzzzzx/w21-s0`, `w21-s1` and `w21-s2` are private CPU runs, version 1, status COMPLETE. Log-clock times are 341.9 s, 475.5 s and 480.5 s. Code dataset: `ezzzzzx/pdp-code-w21`.

rand6 full cycles at T = 60,000 are 25, 27 and 28. Time is spread across rooms 0–8.

PROBE_cap at step 60,000, mean of min(probe_k, 4):

| world_seed | rand6 | oracle | oracle_deep | rand6 − oracle | rand6 − oracle_deep | rand8 − rand6 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.004037 | 0.007386 | 0.005295 | −0.003349 | −0.001258 | 3.118350 |
| 1 | 0.015296 | 0.027083 | 0.005193 | −0.011787 | 0.010103 | 3.109909 |
| 2 | 0.024417 | 0.016091 | 0.004474 | 0.008326 | 0.019943 | 3.092845 |

W2-0 on the same seeds, rand6 PROBE was 1.782124, 0.501576, 1.772205 and oracle PROBE was 0.005676, 0.004577, 0.005486. W2-0 PROBE_cap for rand6 was 1.573478, 0.501576, 0.762446.

Room-8 probe actions differ from W2-0 in 1, 0 and 1 positions out of 121. The reset-and-keys prefix matches on every seed. The differences are in the 96 stay actions.
