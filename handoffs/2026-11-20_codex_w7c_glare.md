# W7c: a noisy TV that depends on where you stand. Does progress leave it while surprise stays? (codex; VPS CPU, seconds)

Read `/root/pdp/CLAUDE.md`, `results/w7b/RESULTS.md`, `handoffs/2026-11-19_grok_w7b_noisy_window.md`, `w7_world.py`, `w7_run.py`.

## Why W7b told us nothing (orchestrator's diagnosis)
W7b put the noise in an 8×8 region of the **scene**. A glance is the whole frame, so that patch is in every glance from every viewpoint (6.6% of the pixels).
The region bandits choose among **viewpoint** regions, so the noise added the same amount to every region's reward and `curious_prog` and `curious_surp`
took identical paths on 20 of 20 seeds. The noise has to depend on the viewpoint, like glare from a lamp on the window that you see only from some
angles.

## World (extend `w7_world.py` with a `glare_region` option; W7-0 must stay bit for bit identical when it is off)
- One viewpoint region g of the 16 (8×8 viewpoint blocks, the same `region_id` the bandits use), chosen per seed with
  `default_rng(9600 + seed).integers(1, 16)` (never region 0, the origin's).
- When the viewpoint is in region g, **every clear cell of the frame shows a fresh uniform-random grey (0–7)** from `default_rng(9700 + seed)`, drawn
  per glance. Holes stay holes. Outside g the frame is as in W7-0.
- The integrator stores what it sees, so glare glances add noise to its counts. That is part of the cost of being fooled.
- Probe: as W7-0, the integrator viewpoint probe on all logo pixels, opacity 768/1024.

## Runs
Seeds 50–69 (fresh), L ∈ {25, 64}. Policies: still, random, sweep, curious_prog, curious_surp, unchanged from W7-0 (`w7_run.py`).

## Outputs
`results/w7c/RESULTS.md` + `results.json`. **At the top, the scoring table:** per seed, the glare region, curious_prog − curious_surp and
curious_prog − random on probe bits at L = 25 and 64, each policy's share of glances in the glare region, with counts against the bars. Then every
policy's mean ± std (ddof = 0), coverage, and the total wall time.

## Smoke
1. With glare off, W7-0's numbers for seed 10 reproduce to 1e-9.
2. With glare on, seed 50: print the glare region, then a frame from inside it and one from outside, as 8 rows of greys and holes.
3. Paths differ between curious_prog and curious_surp on at least one seed (just print the count; no bar). If they are identical on all 20, stop
   and report: the test is void again.

## Pre-registered bars (orchestrator; do not change), L = 64
1. **Surprise is trapped:** curious_surp's share of glances in the glare region ≥ 3/16, mean over seeds.
2. **Progress escapes:** curious_prog's share ≤ 1.5/16, mean over seeds.
3. **And sees more for it:** curious_prog − curious_surp ≤ −0.20 bits on ≥ 16 of 20 seeds.
If bar 1 fails, the region bandit's surprise reward already saturates, and the noisy-TV contrast does not exist in this world. Record that.

## Rules
About 60 minutes of your time. Files you may create or edit: `w7_world.py` (add the option only), `w7c_run.py`, `results/w7c/**`. Do not touch
`docs/`, `audit.md`, `w7_run.py`, `w7b_run.py`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
