I'll read the project instructions and the handoff so I can follow them exactly.The handoff is the scored W7 dotted-window run. I'll read the design and sketch next, then build and run it.The scored run is the integrator on the dotted window, with a progress-based curious policy. I'll build it from the sketch and check it against the sketch before the full seeds.The scored W7 run is in `results/w7/RESULTS.md` and `results/w7/results.json`. One process, 55.935 s.

Opacity 768, integrator viewpoint probe (bits per scored grey). Lower is better.

| Bar | Result | Required |
|---|---|---|
| 1. random − still ≤ −0.30 at L=25 | 20/20, mean −0.878 | ≥ 18/20 |
| 2. curious_prog − random ≤ −0.30 at L=25 | 17/20, mean −0.488 | ≥ 16/20 |
| 3. mean curious_prog − sweep at L=64 | +0.034 | ≤ +0.10 |

Seeds 20, 24, and 26 miss bar 2 (−0.214, −0.177, −0.250). At L=25 the sweep still leads curious_prog by 0.278 on the mean. At L=64 that gap is 0.034.

`curious_prog` and `curious_surp` took the same path on every seed. Unvisited regions outrank visited ones, and the index-order tour only reaches region 7 by step 64, so the progress value does not change a choice inside these lives. The visit order for seeds 10–12 is in the results file.

Glued shifts leave coverage at the clear fraction for every policy (0.250 at opacity 768) and the same probe (0.186 at L=25, 0.076 at L=64). The view table stays at 3.000 for standing still and between 2.927 and 2.975 when the agent moves.
