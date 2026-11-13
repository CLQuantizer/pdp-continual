# M5-W3: a compression-progress controller that has to walk to what it wants to learn (grok, VPS CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/w3/RESULTS.md` and `results/h23c/RESULTS.md` (both including their
Verdicts), `curio.py` (the `live` function), `w3_world.py` and `w3_headroom.py`.

## Context
W3 puts H23c's sources in rooms on a ring, and moving costs time. It has headroom. Knowing which rooms are learnable
(oracle7) beats random dwelling by 0.20 bits/symbol. Seeking hard rooms (oracle_hard) loses, because hard is not
learnable. In H23c, a progress controller choosing channels for free beat even the oracle. Here it has to pay travel
to reach what it wants. This is the first time a curious agent here acts in a world with real headroom, and it is
the north star's core loop: act, learn and choose by compression progress. Principles served: reward compression
progress (3), and act to maximise it (4).

## What to build: `m5_w3.py`
It imports `curio.py`, `w3_world.py` and `w3_headroom.py`, and edits none of them. World, learner (`curio.SGDCompressor`,
`CURIO_LR=0.004`), corridor glances, life L and the score W are exactly as in `w3_headroom.py`.

The controller decides in **visits**. A visit is: choose a target room, take the shortest path there (each move is one
decision, with a corridor glance), then stay 8 decisions (8 glances of that room's channel). This is the same
schedule as `rand_dwell`. A visit cut short by the end of life ends there.

Rewards are computed per dwell glance, exactly as `curio.live` computes them with `prog_kind = "hist"`. For prog, that
is old vs new compressor scored on the reservoir sample of the room's channel history (`R_RES`, the same reservoir
update, rng `default_rng(20_000 + seed)`). For err, it is the glance's bits per symbol. The **visit reward** is
the sum over its dwell glances of the per-glance reward, divided by the number of decisions the visit took
(travel plus dwell). That makes it a rate per decision, so travel counts as a cost. Corridor glances earn nothing
and are never a target.

Controllers (rng `default_rng(10_000 + seed)` for choices):
- `rand_dwell`: target uniform over the 9 rooms (as in `w3_headroom.py`).
- `oracle7`: target uniform over the 7 learnable rooms (as in `w3_headroom.py`).
- `err`: bandit on the err visit reward.
- `prog`: bandit on the prog visit reward.
- Bandit, the same for both: an optimistic first pass visits every room once, always going to the nearest unvisited
  room (ties broken to the right). Then, with probability EPS = 0.1, a uniform room; otherwise the room with the
  highest EMA value (ties broken at random), where the EMA uses curio's BETA = 0.3 per visit to that room.

## Seeds and lives
World seeds 6–11 (held out; W3's headroom check used 0–2). L = 512 for every controller; also L = 2,048 for `prog` and
`oracle7`, as a report.

## Where to run
On the VPS, with a pool of 2. The W3 kernel ran all its lives in 26 s, so this whole job should take a few
minutes. If the smoke timing says the job will take more than 20 minutes, use one private Kaggle CPU kernel instead,
`ezzzzzx/m5-w3` (`"is_private": true`, `"enable_gpu": false`, `"enable_internet": false`, code as private dataset
`ezzzzzx/pdp-code-m5w3`).

## Smoke test (before the scored run)
1. `rand_dwell` and `oracle7` in `m5_w3.py` reproduce `w3_headroom.py`'s W for world seed 0, L = 512, to 1e-9.
2. A 6-visit trace of `prog` on seed 0: target, travel decisions, the dwell's summed progress and the visit reward.
3. All four controllers give a finite W on seed 0 with L = 64. Print the timing.
If any check fails, stop and report.

## Outputs
`results/m5w3/RESULTS.md` and `results/m5w3/results.json`, with numbers only and no verdict:
- W per controller and seed (L = 512), mean ± std (ddof = 0), and the per-channel held-out bits;
- shares of decisions per room channel and in corridors, per controller (mean over seeds), and by quarter of life
  for `prog`;
- the gaps per seed: prog − rand_dwell, prog − oracle7 and prog − err;
- L = 2,048: W for prog and oracle7;
- the W3 headroom numbers (seeds 0–2) next to these, for reference;
- the smoke-test output and wall seconds.

## Pre-registered bars (orchestrator scores; do not change)
1. **Progress pays for its walking:** mean over seeds of (prog − rand_dwell) ≤ −0.10, with prog lower on at least 5 of
   the 6 seeds.
2. **Progress, not error:** mean over seeds of (prog − err) ≤ −0.20.
Report only: prog − oracle7 (H23c's stretch result was beating the oracle), the noise and dark shares, and L = 2,048.
Kill rules: if bar 1 fails, the progress signal does not survive travel cost and delayed reward, and the controller
needs a model of where progress is (planning) rather than a bandit. If bar 2 fails, rooms with no learnable content
do not trap the error-seeker in this world, which would be a world finding, not a curiosity one.

## Budget and rules
- At most 30 minutes of compute. No GPU.
- Run experiments in the foreground.
- Files you may create: `m5_w3.py`, `results/m5w3/**` and, only if Kaggle is used, `kaggle/m5w3/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `curio.py`, `w3_world.py`, `w3_headroom.py`, `h20_world.py`, other `results/`
  or `kaggle/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
