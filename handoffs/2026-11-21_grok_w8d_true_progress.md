# W8-1c: the watching wall with the true progress reward, measured on the stored sights (grok; VPS CPU)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-20_grok_w8c_panel_bandit.md`, `results/w8c/RESULTS.md`, your `w8c_panels.py`, and in `w7_world.py` the classes
`GlanceReservoir` and `ObservedReservoir`.

## Why W8-1b failed (orchestrator's diagnosis)
The panel bandit now works (first pass at decision 32; it watches fast > slow > static, bar 3 PASS), but it spent 59% of its time on the TV, because the TV earned
the largest "progress" (4,119 bits on life 0's first pass, against about 2,100 elsewhere). The reward was `GlanceReservoir.progress`, which assumes every stored
sight of a pixel showed the **same grey as the current glance** (true on W7's still logo, false on a TV or a replaced picture). On the TV, each glance's random
grey is new, so its probability jumps when learned, and that jump is multiplied by every stored sight of the pixel. It measures progress on the glance itself,
the same failure as H35b, H35c and W9. `ObservedReservoir.progress` (equation 5) scores every stored sight at the grey it actually showed: learning a random grey
helps some stored sights and hurts others, so the TV's progress should be near zero.

## Change (everything else exactly as W8-1b: world, lives 0–19, L = 1,024, panel bandit, sweep inside panels, eye, scoring, bars)
- `curious_prog` and `curious_prog_noage` use `ObservedReservoir` (one per panel, cap 256, seeded as W8-1b) and its `progress` as the visit reward.
- Add a report-only arm `curious_prog_glance`: W8-1b's curious_prog unchanged (GlanceReservoir), to show the difference in one table.
- round_robin, random, curious_surp unchanged (their numbers must reproduce W8-1b exactly; check it).

## Smoke
1. On a still panel (life 0, the static panel, first 64 glances) the two progress functions agree to 1e-9.
2. Life 0: print the first-pass visit rewards per panel kind for curious_prog. TV should be the smallest. If it is not, stop and report.
3. round_robin and curious_surp reproduce W8-1b life 0 exactly.

## Outputs
`results/w8d/RESULTS.md` + `results.json`, scoring table at the top as W8-1b, plus per-life panel shares. Demo strips to `internal/w8_demo/wall_c/`.

## Pre-registered bars (orchestrator; do not change), 20 lives
1. curious_prog − round_robin life accuracy ≥ +0.03, CI excluding 0.
2. curious_prog TV share ≤ 0.15, and curious_surp ≥ 0.35.
3. curious_prog share fast > slow > static, mean over lives.

## Rules
About 90 minutes. **Codex runs a CPU smoke on the VPS today: start your full run only when `uptime` 1-minute load is < 1.0** (check every 5 minutes; you may run the smoke
before). One process, `torch.set_num_threads(2)`. Files you may create: `w8d_progress.py` (import from `w8c_panels.py` etc.; do not edit them), `results/w8d/**`,
`internal/w8_demo/wall_c/**`. Do not touch `docs/`, `audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.

## Papers (do this too)
Register in D1 `pdp-papers-index` if missing, as CLAUDE.md says: Zhang, Song, Gao, Chen, Bao, Ma 2019, "Be Your Own Teacher: Improve the Performance of Convolutional
Neural Networks via Self Distillation" (ICCV 2019, arXiv 1905.08094); Borbély 1982, "A two process model of sleep regulation" (Human Neurobiology 1(3):195–204).

## Amendment (orchestrator, 21 Nov, after the first-pass gate stopped: TV 2,281 bits, the largest)
`w7_run._reward` puts the current glance into the reservoir **before** scoring progress, so the stored sights include the glance being learned. Its own random greys
then gain when learned: progress on the item again, only diluted. **Change: score `ObservedReservoir.progress` on the sights stored before this glance, then include
the glance.** (On the first glance of a life the reward is 0.) Do this for `curious_prog` and `curious_prog_noage` only; `curious_prog_glance` stays W8-1b's arm unchanged.
Add a smoke check: on life 0's static panel, print the first 64 glances' rewards under both orders (no bar; just show them). Everything else, the first-pass gate
(TV smallest) and the bars, is unchanged. If the gate fails again, stop and report the four first-pass rewards and the TV's per-glance rewards over its first
3 visits.
