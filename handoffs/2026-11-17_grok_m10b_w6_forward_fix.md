# M10b-W6: the forward-value controller, with a reward on the right scale (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/m10w6/RESULTS.md` and the code in `m10_w6.py`, `w6_headroom.py` and `m8_w5.py`.

## Context
M10-W6 confirmed curiosity in the open world: prog_age_s beats random dwelling by 0.289 on fresh seeds (11 of 12). The
forward-value controller prog_fwd_s failed (worse than prog_age_s on 12 of 12), and the orchestrator traced it to the spec, not
the idea. The whole-store visit reward was a raw sum of exam bits, −20 to −350 per decision. The TD target was clipped at ±10, so
every target saturated at −10 and the value learned only "go where you were recently". The store's change is also dominated by
births and background forgetting elsewhere, which no action controls. M10b separates the two questions: does a learned value of
future progress beat the hand-set ageing when its reward is the same as prog_age_s's? And does an interference term, with the
background drift removed, add anything?
Principles served: reward compression progress (3) and act to maximise it (4).

## Fixed parts
`World6`, the cortex, the fixed sleep share 0.20, stores, movement, dwell 8, the exam store and T_open exactly as in M10
(`m10_w6.py`; import it, wrap or subclass only). L = 4,096. birth_period ∈ {128, 512}.

## Controllers (rng `default_rng(1_000 + world_seed)`)
- `rand_dwell_s` and `prog_age_s`, as in M10.
- `fwd_own_s`: prog_fwd_s's features, shared linear value, ε 0.1, optimistic first pass, normalised-LMS TD (γ 0.5,
  α = 0.05 / (1 + ‖x‖²)), choice by argmax, **with the visit reward replaced by prog_age_s's own visit reward** (the
  per-room exam progress that `m8_w5.py` computes for the visited room, in the same units). r_last and r_ema are features of that
  reward. **No clip.**
- `fwd_inter_s`: as fwd_own_s, with reward r_own + r_inter. r_inter is the change in the other slots' exam bits over the visit,
  divided by the total number of exam symbols of those slots and by the visit's decisions (bits per exam symbol per decision,
  sign so that "other rooms got better" is positive), **minus** a baseline b: an EMA (decay 0.95 per visit) of r_inter over all
  previous visits of this life, initial 0. Births during a visit are excluded: a slot born during the visit does not enter r_inter
  for that visit. No clip.
Constants fixed in advance; do not tune.

## Seeds and lives
**Fresh world seeds 24–35.** 4 controllers × 2 periods × 12 seeds = 96 lives.

## Kernels
One private CPU kernel per seed, `ezzzzzx/m10b-w6-s24` … `-s35`, `"is_private": true`, `"enable_gpu": false`,
`"enable_internet": false`, pool of up to 4. Code as private dataset `ezzzzzx/pdp-code-m10b`. At most 90 minutes per kernel; split
by period if needed. **At most 2 kernels at once** (codex runs H34a on Kaggle).

## Smoke test (VPS CPU, one process, before any kernel)
1. `rand_dwell_s` and `prog_age_s` reproduce M10's code bit for bit on seed 12, period 128, L = 512 (T_open to 1e-6).
2. For `fwd_own_s`, seed 12, L = 512: for the first 12 visits print the slot, r, prog_age_s's reward computed independently for
   the same visit (must be equal to 1e-12), y, Q before and after, and w.
3. For `fwd_inter_s`: print r_own, r_inter, b for the first 12 visits; show one visit containing a birth and that the newborn is
   excluded; print the range of r over the whole life (must be on the scale of r_own, not hundreds).
4. All controllers finite at L = 512; time a 1,024-decision life and estimate kernel wall times.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/m10b/RESULTS.md` and `results/m10b/results.json`, numbers only, no verdict:
- T_open and final checkpoint per controller, period and seed, mean ± std (ddof = 0), per checkpoint;
- gaps per seed and period: prog_age_s − rand_dwell_s, fwd_own_s − prog_age_s, fwd_inter_s − fwd_own_s, fwd_own_s − rand_dwell_s,
  with counts of seeds where the left term is lower;
- the range and mean of each controller's reward; final w per seed; birth latency; noise and corridor shares;
- the smoke output, wall seconds, kernel ids.

## Pre-registered bars (orchestrator scores at birth_period 128; do not change)
1. **A learned value of future progress beats hand-set ageing:** fwd_own_s − prog_age_s ≤ −0.05 on T_open, on at least 8 of 12.
2. **Interference, net of background drift, adds information:** fwd_inter_s − fwd_own_s ≤ −0.05, on at least 8 of 12.
Report only: birth_period 512; prog_age_s − rand_dwell_s (a second fresh-seed replication).
Kill rules: if bar 1 fails, one hand-set ageing constant is as good as a learned linear value over these features, and the
forward controller is parked until a world makes the future value of a room differ by kind. If bar 1 passes and bar 2 fails,
interference is not worth crediting in W6.

## Budget and rules
- CPU only. At most 90 minutes per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `m10b_w6.py`, `kaggle/m10b/**`, `results/m10b/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `m10_w6.py`, `w6_world.py`, `w6_headroom.py`, `m8_w5.py`, any other experiment code,
  other `results/` or `kaggle/` folders, `.env`. Codex is writing `h34a_local_depth.py`, `kaggle/h34a/**`, `results/h34a/**`: do
  not edit them.
- Gates and bars are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.
