# W8-1d: the watching wall, with a pixel model that can notice change (grok; VPS CPU)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-21_grok_w8d_true_progress.md` (with its amendment), `results/w8d/RESULTS.md`, your `w8d_progress.py`, `w8c_panels.py`.

## What W8-1c showed (orchestrator's diagnosis)
Progress measured on the sights stored before a glance fixed the TV (7.6% of decisions, bar ≤ 15%, PASS). But the camera then spent **70% of its time on the static
picture** and only 10% on the fast one, and lost to round-robin by 0.19. The pixel model is a lifetime KT count per pixel, and the reservoir is a uniform sample of
the whole life. When a picture is replaced, the new greys contradict most stored sights (the old picture), so learning them makes the stored history **longer**:
negative progress, and the camera avoids change. On the static picture, counts keep sharpening and progress stays positive for a long time. The reward is right
for this learner; the learner cannot represent change. A compressor that expects the world to change weighs recent sights more.

## Change (everything else exactly as W8-1c: world, lives 0–19, L = 1,024, panel bandit with prog_age_s, sweep inside panels, progress scored on sights stored
before the glance, eye, scoring, demo strips to `internal/w8_demo/wall_d/`)
For `curious_prog`, `curious_prog_noage` and a new report-only arm `curious_surp_forget`:
- **Forgetting counts:** each time a pixel is seen, its 8 grey counts are multiplied by γ = 0.8 before the new sight is added (per panel). Probability is KT on these
  counts (α = 0.5) as before.
- **Recent reservoir:** each panel keeps its **last 32 glances** (first in, first out), not a uniform sample. Progress = codelength of those stored sights under the
  counts before learning the glance, minus after (equation 5, scored before the glance is stored).
- The eye and the reconstruction are unchanged (a cell shows the last block seen there).
- `curious_surp_forget`: the surprise reward under the forgetting counts (report only).
Keep `round_robin`, `random`, `curious_surp` and W8-1c's `curious_prog` (as `curious_prog_lifetime`, report only); they must reproduce `results/w8d` exactly.
γ and the reservoir size are fixed; do not tune.

## Smoke
1. Life 0, the static panel: after 6 visits, its visit reward is below 10% of its first visit's reward (progress dies once the picture is known).
2. Life 0, the fast panel: print the visit rewards around its first two replacements; the first visit after a replacement should be higher than the visit before it.
3. Life 0's first pass: the TV's reward is still the smallest. If check 3 fails, stop and report. Checks 1–2 print only.
4. Reproduction of the unchanged arms on life 0.

## Outputs
`results/w8e/RESULTS.md` + `results.json`, scoring table at the top as W8-1c (life accuracy, CIs, shares per panel kind, counts against the bars).

## Pre-registered bars (orchestrator; do not change), 20 lives
1. curious_prog − round_robin life accuracy ≥ +0.03, CI excluding 0.
2. curious_prog TV share ≤ 0.15, and curious_surp ≥ 0.35.
3. curious_prog share fast > slow > static, mean over lives.

## Rules
About 90 minutes. **Codex runs CPU smokes on the VPS today: start your full run only when the 1-minute load is < 1.0** (check every 5 minutes). One process,
`torch.set_num_threads(2)`. Files you may create: `w8e_forget.py` (import from earlier w8 files; do not edit them), `results/w8e/**`, `internal/w8_demo/wall_d/**`. Do not
touch `docs/`, `audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
