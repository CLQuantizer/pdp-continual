I'll read the project instructions and the handoff file first, then follow that handoff exactly.The handoff is W7b: a noisy patch in the window, with a smoke test against W7-0. I'll read the design, the world, the runner, and the W7 results next.The world always shows the full logo, so I need to see how a noisy region is supposed to sit in that geometry before I change the code.The noise is an 8×8 patch of the logo, and the curious policies only split if that patch changes region rewards. I'll measure that on one seed before writing the run.An 8×8 patch never lifts a full-frame glance above the unvisited prior, so region choice may not split. I'll add the noise option so W7-0 stays bit-identical, then score seeds 30–49.The runner has an unused path that would fake path equality. I'll remove that before the smoke test.The scoring table listed policies in a different order than the counts. I'll correct the header and rewrite the results from the saved numbers.W7b is in `results/w7b/RESULTS.md` and `results/w7b/results.json`. One process, 18.238 s. Noise off, seed 10 matches W7-0 with maximum absolute error 0.

Opacity 768, integrator viewpoint probe on the logo pixels outside the noisy 8×8, bits per scored grey. Lower is better. The noise region is `default_rng(9500 + seed).integers(1, 16)`.

| Bar at L=64 | Result | Required |
|---|---|---|
| curious_prog − curious_surp ≤ −0.20 | 0/20, mean 0.000000 | ≥ 16/20 |
| mean curious_prog share of glances in the noise region | 0.075781 | ≤ 0.093750 |

`curious_prog` and `curious_surp` took the same path on 20 of 20 seeds. At L=64, curious_prog − random = −0.427. Noise-pixel surprise stays at 3.232 bits while the other visible pixels fall to 0.775. Those noise pixels are 0.066 of each glance. Equation-5 progress per glance is 9.19 bits on the patch and 120.04 bits on the rest of the logo.
