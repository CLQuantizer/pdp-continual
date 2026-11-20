# W8-1b: the watching wall again, with curiosity choosing the panel and a sweep inside it (grok; VPS CPU)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-20_grok_w8b_watching_wall.md`, `results/w8b/RESULTS.md` and your `w8b_wall.py`.

## Why W8-1 failed (orchestrator's diagnosis: a spec error)
The 64-arm bandit (4 panels × 16 viewpoint regions) starts optimistic and must try each arm once. Each try costs about 16 decisions (walk plus dwell), so the
first pass needs about 1,000 decisions, the whole life. It finished on 0 of 20 lives, so curious_prog never used what it learned: its shares were exactly a
quarter, and ageing never mattered. The question, whether curiosity looks where things change, was never asked. In W4–W6 the bandit chose among rooms,
and inside a room the glance was given. Here, too, the panel is the room.

## Change (everything else exactly as W8-1: world, panels, images, lives 0–19, L = 1,024, eye, scoring, demo strips to `internal/w8_demo/wall_b/`)
- **Inside a panel, every policy uses the same mover:** W7's planned sweep, resumed from where that panel's sweep last stopped (each panel keeps its own
  sweep position). When a panel's image is replaced, its sweep simply continues.
- **The policies differ only in which panel they visit next.** A visit is 8 decisions in that panel.
  - `round_robin`: panels in turn.
  - `random`: a uniform panel each visit.
  - `curious_prog`: prog_age_s over the 4 panels, exactly as W5/W6 (ageing τ = 256 decisions, ε = 0.1, β = 0.3, optimistic start). The reward of a visit is that
    panel's integrator compression progress over the visit, W7's progress definition summed over the visit's glances.
  - `curious_surp`: the same bandit with W7's surprise reward.
  - `curious_prog_noage`: curious_prog without ageing (report only).

## Outputs and smoke
As W8-1, in `results/w8c/RESULTS.md` + `results.json`, scoring table at the top. Smoke: W8-1's smoke check 1 still passes; life 0 prints the first 32 panel choices
of curious_prog with their kinds; one life per policy is timed.

## Pre-registered bars (orchestrator; do not change), 20 lives
1. **Curiosity beats the best fixed plan in a changing world:** curious_prog − round_robin life accuracy ≥ +0.03, CI excluding 0.
2. **It ignores the TV:** curious_prog's share of decisions on `tv` ≤ 0.15, and curious_surp's ≥ 0.35.
3. **It watches what changes:** curious_prog's share on `fast` > `slow` > `static`, mean over lives.
Report only: random, curious_prog_noage.

## Rules
About 2 hours. Foreground, one VPS process, `torch.set_num_threads(2)` if `uptime` load < 1.5, else a Kaggle kernel. Files you may create: `w8c_panels.py`
(import from `w8b_wall.py`, `w8_camera.py`, `w7_world.py`, `w7_run.py`; do not edit them), `results/w8c/**`, `internal/w8_demo/wall_b/**`, `kaggle/w8c/**`.
Do not touch `docs/`, `audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
