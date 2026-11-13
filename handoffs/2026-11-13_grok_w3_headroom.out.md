I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is a headroom check: put H23c’s sources in rooms on a ring, then measure whether travel still leaves something for a curious controller to win. I’ll read the cited results and the world code before writing anything.H23c’s score file has only the `prog` condition. I’ll pull the rand7 per-channel bits it cites as its baseline, then match the existing Kaggle kernel layout.H23c’s own score file has only the `prog` condition. The rand7 per-channel bits it treats as the baseline are the H23b runs, so those are the weights `oracle_hard` will use. Next I’ll add the ring world and the headroom runner.Smoke checks passed, and the kernel estimate is about half a minute. I’ll confirm `oracle7` really avoids the worthless rooms before uploading.The kernel is running. I’ll poll it until it finishes.W3 headroom is in `results/w3/RESULTS.md` and `results/w3/results.json`. No verdict and no commit.

The ring uses one `h20_world.World` and `curio.glance`. A move costs one decision and shows 64 corridor spaces. The learner is `curio.SGDCompressor` at `CURIO_LR=0.004`. `oracle_hard` draws targets with probability proportional to H23b `rand7` held-out bits on seeds 6–11 (the baseline named in `results/h23c/RESULTS.md`). Those means are G 3.693095, C1 4.094935, C7 4.254920, E30 1.351967, E110 1.291723, M7 0.473991, M37 2.206474.

End-of-life W, mean ± std (ddof=0) over seeds 0–2:

| policy | L | mean | std |
|---|---:|---:|---:|
| rand_walk | 512 | 3.655580 | 0.116707 |
| rand_dwell | 512 | 2.896721 | 0.087566 |
| oracle7 | 512 | 2.698189 | 0.154175 |
| oracle_hard | 512 | 3.125269 | 0.230694 |
| oracle7 | 2048 | 1.833501 | 0.008104 |

Gaps:

| seed | rand_walk − oracle7 | rand_walk − oracle_hard | rand_dwell − oracle7 | oracle7 L512 − L2048 |
|---:|---:|---:|---:|---:|
| 0 | 0.863270 | 0.798003 | 0.103565 | 1.035492 |
| 1 | 1.006972 | 0.749375 | 0.208820 | 0.906200 |
| 2 | 1.001931 | 0.043553 | 0.283211 | 0.652372 |

H23c reference, seeds 6–11, no travel: rand 2.637 ± 0.052, rand7 2.481 ± 0.046.

Smoke: a teleport policy matched `curio.py --seeds 6 --glances 64 --conds rand --comp sgd` with absolute difference 0. L=64 on seed 0 was finite for all four policies. The estimated kernel wall was 0.6 min, so this was one kernel.

Kernel `ezzzzzx/w3-headroom` version 1, private CPU, no GPU, no internet, dataset `ezzzzzx/pdp-code-w3`. Status COMPLETE. Script wall 25.843 s. The L=512 `oracle7` run matches the t=512 checkpoint of the L=2048 run on every seed (absolute difference 0).
