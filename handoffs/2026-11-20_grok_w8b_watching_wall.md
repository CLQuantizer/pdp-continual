# W8-1: the curious camera watches a wall of four pictures. Does it look where things change, ignore the flickering TV, and leave what it already knows? (grok; VPS or Kaggle)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-20_grok_w8_virtual_camera.md`, `results/w8/RESULTS.md`, your `w8_camera.py`, and `results/w7c/RESULTS.md`.

## Why
W8-0 passed: a camera driven by its own pixel compression progress reveals photographs behind a dotted window, and a frozen eye then names them
(curious_prog 0.918 against random 0.672 at L = 16). But a planned sweep did better (0.984), because the scene never changed and had no noise. A
sweep is the right plan for one still picture. A camera that lives faces a world where some things change, some never do, and some are noise. W8-1
puts the camera in front of such a world, where no fixed plan is right.

## World
- **A wall of 4 panels.** Each panel is one W8-0 window: a 224×224 Imagenette image behind its own dotted window (opacity 768/1024, dot seed as below),
  with W8-0's viewpoint, parallax and 32×32 cell grid. Each decision the camera chooses **a panel and an action** inside it: its state is (panel,
  px, py) for each panel separately (each panel remembers its own viewpoint). Switching panels costs nothing extra.
- **Panel kinds** (one of each, panel positions shuffled per life with `default_rng(9800 + life)`):
  - `static`: one image for the whole life.
  - `fast`: the image is replaced by a new one every 32 decisions (all decisions, not visits to that panel).
  - `slow`: replaced every 128 decisions.
  - `tv`: every clear cell shows fresh uniform-random RGB blocks each glance (glare, as W7c); nothing to learn. Its "image" for scoring is ignored.
- Images are drawn without replacement from the 500 W8-0 images, in an order fixed per life with `default_rng(9900 + life)`. The dot field of a panel's
  image is `make_field(index)` as in W8-0. When an image is replaced, the camera's integrator for that panel keeps its counts (it must notice by itself).
- **Life length** 1,024 decisions. 20 lives (life ids 0–19).

## Agent
- Pixel model: W8-0's grey-proxy integrator, one per panel.
- Policies:
  - `random`: uniform panel, random action.
  - `round_robin`: panels in turn, 8 decisions each, W7's sweep inside a panel (the best fixed plan from W8-0).
  - `curious_prog`: the W7 region bandit generalised to 4 panels × 16 viewpoint regions = 64 arms, with W7's progress reward (unchanged definition),
    **plus prog_age_s ageing**: an arm's belief relaxes toward the optimistic maximum with age, v + (v_max − v)(1 − e^(−age/τ)), τ = 256 decisions
    (as in W5/W6). Walk to the chosen arm and spend 8 decisions there.
  - `curious_surp`: the same with W7's surprise reward and the same ageing.
  - `curious_prog_noage`: curious_prog without ageing (report only; shows whether ageing is what finds the changes).
- The eye reads each panel's reconstruction as in W8-0. A cell seen since the image was replaced holds the current image's block; a cell last seen
  before the replacement holds the **old** image's block (stale), as a real camera's memory would.

## Score
- **Life accuracy:** every 16 decisions, the eye classifies the current reconstruction of the `static`, `fast` and `slow` panels against each panel's
  current label; mean accuracy over these checks and the three panels.
- Per-panel life accuracy, the share of decisions spent at each panel kind, and log-loss.
- Paired differences across the 20 lives, with 95% bootstrap CIs (10,000 resamples, rng 0).

## Smoke
1. With one `static` panel and no others, the single-panel life reproduces W8-0's per-image coverage for images 0–2 under `round_robin`'s sweep (exact match).
2. Life 0: print the panel kinds' positions and the curious_prog panel sequence for the first 64 decisions.
3. Time one life per policy; the full run must fit in 90 minutes of VPS CPU or one Kaggle kernel ≤ 1.5 GPU hours (else use 12 lives and say so).

## Outputs
`results/w8b/RESULTS.md` + `results.json`, numbers only. **At the top, the scoring table:** per life, life accuracy for each policy;
curious_prog − round_robin and curious_prog − curious_surp with CIs; each policy's share of decisions on `tv`, `static`, `fast`, `slow`; counts against the
bars. Then per-panel accuracy, log-loss, walls. Save demo strips for life 0 (curious_prog and round_robin, every 128 decisions, all four panels) to
`internal/w8_demo/wall/`.

## Pre-registered bars (orchestrator; do not change), 20 lives
1. **Curiosity beats the best fixed plan in a changing world:** curious_prog − round_robin life accuracy ≥ +0.03, CI excluding 0.
2. **It ignores the TV:** curious_prog's share of decisions on `tv` ≤ 0.15, and curious_surp's ≥ 0.35.
3. **It watches what changes:** curious_prog's share on `fast` > its share on `slow` > its share on `static`, in mean over lives.
Report only: random, curious_prog_noage.

## Rules
- About 3 hours of your time. Foreground runs. VPS: one process, `torch.set_num_threads(2)`, only if `uptime` load < 1.5 (codex may run a smoke today);
  else a Kaggle kernel. GPU quota guard as CLAUDE.md says.
- Files you may create: `w8b_wall.py` (import from `w8_camera.py`, `w7_world.py`, `w7_run.py`; do not edit them), `kaggle/w8b/**`, `results/w8b/**`,
  `internal/w8_demo/wall/**`. Data stays in `data/w8/` (gitignored).
- Do not touch `docs/`, `audit.md`, other code, `.env`. Gates and bars are the orchestrator's: if one looks wrong, stop and report. Never print secrets.
  Do not commit.
