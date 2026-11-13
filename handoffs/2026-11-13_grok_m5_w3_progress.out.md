I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is the M5-W3 progress controller. I'll read the W3 and H23c results and the world, learner, and headroom code before writing anything.The smoke test has to match W3's dwell policies exactly. I'll check how expensive whole-history progress scoring is, then write the controller.W3's dwell baselines have to match the existing headroom lives. I'll implement the visit controller on top of that world and learner, then smoke-test before the scored run.Smoke matched W3 to 0 and the job fits on this machine. Next is the scored run: seeds 6–11, pool of 2.`m5_w3.py` is in place and the scored lives are in `results/m5w3/RESULTS.md` and `results/m5w3/results.json`. Smoke passed, then the run stayed on this machine (pool of 2, 346 s). No verdict.

Smoke, world seed 0: `rand_dwell` and `oracle7` match `w3_headroom.py` at L = 512 to 0, including the published W3 numbers. All four controllers are finite at L = 64. The first six `prog` visits walk rooms 0–5 (M37, C7, C1, N, G, E30); N’s summed progress is 0.000229.

**L = 512, seeds 6–11.** W is bits/symbol, lower better, std with ddof = 0.

| controller | mean ± std |
|---|---|
| rand_dwell | 2.917287 ± 0.198684 |
| oracle7 | 2.664069 ± 0.130945 |
| err | 3.576348 ± 0.192500 |
| prog | 2.767184 ± 0.047137 |

Gaps, prog minus the other controller: mean prog − rand_dwell −0.150103 (prog lower on 4 of 6), prog − oracle7 +0.103115 (2 of 6), prog − err −0.809163 (6 of 6).

Noise and dark, mean share of decisions at L = 512: prog N 0.0260, Z 0.1094; err N 0.5749, Z 0.0156; oracle7 N 0, Z 0.

**L = 2048:** prog 1.779311 ± 0.053183, oracle7 1.825863 ± 0.025261. Mean prog − oracle7 −0.046551 (prog lower on 5 of 6). prog’s N and Z shares are 0.0124 and 0.0599.

`rand_dwell` and `oracle7` use `w3_headroom.py`’s planner and `default_rng(1_000 + seed)`, which is what the 1e-9 check fixes. `err` and `prog` draw choices from `default_rng(10_000 + seed)`; their reservoir uses `default_rng(20_000 + seed)`.
