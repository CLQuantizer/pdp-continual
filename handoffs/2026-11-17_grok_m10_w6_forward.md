# M10-W6: curiosity in the open world on fresh seeds, and a controller that values future progress (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/w6/RESULTS.md`, `results/m8w5/RESULTS.md`, `results/m9w5/RESULTS.md`, and the code
in `w6_world.py`, `w6_headroom.py` and `m8_w5.py`.

## Context
W6 (seeds 0–5): births cost random dwelling +0.39 bits on T_open, but neither oracle beat random dwelling (oracle_new_s −0.04,
oracle_learn_s +0.04). They know which rooms are learnable or new, not how fast a room pays. M8's ageing progress bandit
(prog_age_s), reported only, was −0.19 (4 of 6) at birth period 128 and −0.13 (5 of 6) at 512, better than both oracles.
Seeds 0–5 were selection seeds, so this is confirmed here on fresh seeds. A second controller replaces the hand-set ageing
(τ = 256) with a learned estimate of a room's **future** progress (Schmidhuber 1991: reward the predicted improvement, learn
the predictor), rewarded by the change in the whole exam store, interference included.
Principles served: reward compression progress (3) and act to maximise it (4).

## World and fixed parts
`World6` from `w6_world.py`, unchanged. Cortex, fixed sleep share 0.20 and its placement, stores, movement, dwell 8, exam store and
T_open scoring exactly as in `w6_headroom.py` (import it; wrap or subclass only). L = 4,096. birth_period ∈ {128, 512}.

## Controllers (rng `default_rng(1_000 + world_seed)`)
- `rand_dwell_s`, as in W6.
- `prog_age_s`, as in W6 (M8's code: τ = 256, ε 0.1, β 0.3).
- `prog_fwd_s` (new). Arms are the 18 slots. Same ε 0.1, same optimistic first pass over unvisited slots as prog_age_s.
  - **Visit reward** r: the decrease in the whole exam store's bits (summed over all slots' exam items, as M9's sleep reward
    uses) from the start to the end of the visit, divided by the visit's decisions (dwell 8, plus the travel decisions that
    led to it). This counts interference: learning one room while others get worse earns less.
  - **Features of a slot at choice time** (computed for every slot): x = [1, r_last, r_ema (β 0.3), log(1 + visits),
    1 − exp(−age/64), 1 − exp(−age/256), 1 − exp(−age/1024), travel decisions to reach it / 8]. r_last and r_ema are 0
    for an unvisited slot; age is decisions since the slot's last visit ended (or since decision 0).
  - **Value:** one linear model Q(x) = w·x shared by all slots, w initialised to 0.
  - **Learning (TD over visits):** when a visit to slot i ends, with features x_i taken at the choice that started it, the
    target is y = r + γ · max_j Q(x_j) with the features of all slots at that moment, γ = 0.5. Update
    w ← w + α (y − Q(x_i)) x_i, with α = 0.05 / (1 + ‖x_i‖²) (normalised LMS). Clip |y| at 10.
  - **Choice:** with probability ε uniform over slots, otherwise argmax_j Q(x_j) (ties broken by the rng).
  These constants are fixed in advance; do not tune them.

## Seeds and lives
**Fresh world seeds 12–23** (not used in W6). Every controller at both periods: 3 × 2 × 12 = 72 lives.

## Kernels
One private CPU kernel per seed, `ezzzzzx/m10-w6-s12` … `-s23`, `"is_private": true`, `"enable_gpu": false`,
`"enable_internet": false`, pool of up to 4 inside. Code as private dataset `ezzzzzx/pdp-code-m10w6`. At most 90 minutes per kernel.
**At most 2 kernels at once** (codex runs H34a kernels on Kaggle).

## Smoke test (VPS CPU, one process, before any kernel)
1. `rand_dwell_s` and `prog_age_s` on W6 seed 0, period 128, L = 512 reproduce W6's code bit for bit (T_open to 1e-6).
2. `prog_fwd_s`, seed 0, L = 512: print, for the first 12 visits, the slot, r, the target y, Q before and after, and w.
   Check by hand on one visit that the update matches the formula.
3. The whole-store visit reward equals the sum of per-slot exam-bit changes over the visit (to 1e-9), on 3 visits.
4. All controllers finite at L = 512; time a 1,024-decision life and estimate kernel wall times.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/m10w6/RESULTS.md` and `results/m10w6/results.json`, numbers only, no verdict:
- T_open and the final checkpoint per controller, period and seed; mean ± std (ddof = 0); per checkpoint;
- gaps per seed and period: prog_age_s − rand_dwell_s, prog_fwd_s − rand_dwell_s, prog_fwd_s − prog_age_s, with counts of seeds
  where the left term is lower;
- birth latency (decisions from a birth to the first stay there), and noise share, corridor share, per controller;
- prog_fwd_s's final w per seed, and the mean visit reward by kind of room (the seven processes, N, Z);
- the smoke-test output, wall seconds, kernel ids.

## Pre-registered bars (orchestrator scores at birth_period 128; do not change)
1. **Curiosity pays in the open world (confirmation):** prog_age_s − rand_dwell_s ≤ −0.10 on T_open, on at least 9 of 12 seeds.
2. **Valuing future progress beats hand-set ageing:** prog_fwd_s − prog_age_s ≤ −0.05 on T_open, on at least 8 of 12 seeds.
Report only: birth_period 512; prog_fwd_s − rand_dwell_s.
Kill rules: if bar 1 fails, W6's prog_age_s gain was selection noise and the open world needs more unequal kinds before
controllers are compared. If bar 1 passes and bar 2 fails, a learned linear value over visits does no better than one hand-set
ageing constant; record which features carried w.

## Budget and rules
- CPU only. At most 90 minutes per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `m10_w6.py`, `kaggle/m10w6/**`, `results/m10w6/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `w6_world.py`, `w6_headroom.py`, `m8_w5.py`, `m9_w5.py`, any other experiment code,
  other `results/` or `kaggle/` folders, `.env`. Codex is writing `h34a_local_depth.py`, `kaggle/h34a/**`, `results/h34a/**`:
  do not edit them.
- Gates and bars are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.
