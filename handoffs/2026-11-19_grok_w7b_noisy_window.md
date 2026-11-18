# W7b: a noisy patch in the window. Does progress ignore it where surprise is trapped? (grok; VPS CPU, seconds)

Read `/root/pdp/CLAUDE.md`, `results/w7/RESULTS.md`, `internal/w7_design.md`, `w7_world.py`, `w7_run.py`.

## Why
W7-0 passed: a progress-driven agent discovers that moving reveals the logo. But on a deterministic logo, surprise and progress rank regions alike.
W7b adds a noisy TV inside the window: in one 8×8 region of the scene (the region index chosen per seed with `default_rng(9500 + seed)`, never the origin's
region), every glance shows fresh uniform-random greys (0–7), so its surprise never falls and nothing can be learned there.

## Do
Extend `w7_world.py` with a `noise_region` option (wrap or subclass; keep W7-0 reproducible bit for bit when it is off). Same logo, dots at 768/1024, L ∈ {25, 64},
**fresh seeds 30–49**. Policies: still, random, sweep, curious_prog, curious_surp (as in W7-0). The probe scores the logo's non-noise pixels only.
**Outputs:** `results/w7b/RESULTS.md` + `results.json`, scoring table at the top: per seed, curious_prog − curious_surp and curious_prog − random at L = 25
and 64, share of glances each policy spends in the noise region, with counts. Then all policies (mean ± std), coverage, total wall.
**Smoke:** with noise off, W7-0's numbers for seed 10 reproduce to 1e-9.
**Bars (orchestrator; do not change), L = 64:** (1) curious_prog − curious_surp ≤ −0.20 on ≥ 16 of 20; (2) curious_prog's share of glances in the noise region ≤ 1.5 ×
1/16 (its fair share), mean over seeds.

Rules: files you may create or edit: `w7_world.py` (add the option only), `w7b_run.py`, `results/w7b/**`. Do not touch `docs/`, other code, `.env`. Never print secrets. Do not commit.
