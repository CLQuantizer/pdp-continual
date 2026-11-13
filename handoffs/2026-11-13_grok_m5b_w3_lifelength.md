# M5b-W3: does curiosity's advantage grow with the length of a life? (grok, VPS CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/m5w3/RESULTS.md` (including its Verdict), `m5_w3.py`, `w3_world.py` and
`w3_headroom.py`.

## Context
In M5-W3 the progress controller beat random dwelling by 0.15 on the mean, but on only 4 of 6 seeds. At L = 2,048 it
beat the learnability oracle on 5 of 6, following an easy-to-hard curriculum. The claim to test: what curiosity buys is
an order of study, so its advantage grows with life length. This uses fresh seeds; nothing is retuned. Principles served: reward
compression progress (3), and act to maximise it (4).

## What to run
`m5_w3.py` unchanged (import it, do not edit it). Controllers `rand_dwell`, `oracle7` and `prog`, exactly as in
M5-W3. Lives L ∈ {512, 1,024, 2,048}. **World seeds 12–17** (fresh). Pool of 2 on the VPS. If the smoke timing says
more than 30 minutes, use one private Kaggle CPU kernel `ezzzzzx/m5b-w3` (`"is_private": true`, `"enable_gpu":
false`, `"enable_internet": false`, code as private dataset `ezzzzzx/pdp-code-m5bw3`).

Score W exactly as in M5-W3. Also report **W9**: the same frozen-compressor score over all 9 room channels (the 7, plus N
and Z). It is a report only, to see how much of prog's time goes to content that W does not count.

## Smoke test
1. `m5_w3.py` reproduces M5-W3's prog W for seed 6, L = 512, to 1e-9.
2. All three controllers give a finite W and W9 on seed 12 at L = 64. Print the timing.
If any check fails, stop and report.

## Outputs
`results/m5bw3/RESULTS.md` and `results/m5bw3/results.json`, with numbers only and no verdict:
- W and W9 per controller, L and seed; mean ± std (ddof = 0);
- the gaps per seed and L: prog − rand_dwell and prog − oracle7;
- shares of decisions per channel and in corridors, per controller and L (mean over seeds), and prog's shares by
  quarter at each L;
- the smoke-test output and wall seconds.

## Pre-registered bars (orchestrator scores; do not change)
1. **Over a long life, progress wins:** at L = 2,048, mean over seeds of (prog − rand_dwell) ≤ −0.10, with prog lower
   on at least 5 of 6 seeds.
2. **The advantage grows with life length:** mean(prog − oracle7) at L = 2,048 is lower than at L = 512 by ≥ 0.10.
Report only: L = 1,024, W9, and the Z and corridor shares.
Kill rules: if bar 1 fails, a bandit on progress does not beat random dwelling in an embodied world even with time
to spare, and the next controller plans where progress is. If bar 2 fails, the curriculum effect in M5-W3 was seed
noise.

## Budget and rules
- At most 40 minutes of compute. No GPU.
- Run experiments in the foreground.
- Files you may create: `results/m5bw3/**` and, only if Kaggle is used, `kaggle/m5bw3/**`. A small runner script
  `m5b_w3.py` is allowed, if it only imports `m5_w3.py` and adds W9.
- Do not touch: `docs/loop.md`, `audit.md`, `m5_w3.py`, `curio.py`, `w3_world.py`, `w3_headroom.py`, other `results/`
  or `kaggle/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
