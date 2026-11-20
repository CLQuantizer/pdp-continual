# W9b: sleep when the input is poor, measured as the input's own codelength (codex: implement + smoke; grok runs kernels). Also a one-line H34d fix.

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-19_codex_w9_day_night.md`, `results/w9/RESULTS.md`, and your `w9_day_night.py`.

## What W9 showed (orchestrator's diagnosis)
Sleeping exactly at night (the clock oracle) beat sleeping at random times by 0.197 bits on T14 (8 of 12 seeds at ≤ −0.10; bar 9). Night is worth sleeping
through. But `pressure_opp` slept at night only **8%** of its sleep time, below chance (25%), and lost to fixed_random by +0.070. Its "poor input" signal
was the gain from learning one glance, measured on that same glance. A noise glance has high loss, and learning from it lowers the loss **on that
glance** a lot (it memorises the glance), so night looked like rich input. This is the same failure as H35b and H35c: progress measured on the item
itself rewards noise.

## Change (one new arm; everything else exactly as W9: World5, period 128, L = 2,048, night = (t mod 256) ≥ 192, p = 0.8, clean exams, prog_age_s)
- `pressure_bits`: S dynamics as `pressure_opp` (+1 per wake decision, −4 per sleep decision; an episode ends at S = 0, rounded up to the end of an
  eight-decision visit). The input-quality signal is **b_t = EMA (0.9) of the glance's bits per symbol under the frozen scorer before learning**
  (the `before` value W9 already computes, divided by the glance's symbol count). m_t = median of b over the last 256 wake decisions (including the
  current one). **Sleep when (S ≥ 32 and b_t > m_t) or S ≥ 128.** It is never told the clock.
- Arms: `fixed_random`, `fixed_night`, `pressure`, `pressure_opp`, `pressure_bits`. **Fresh seeds 72–83.**

## Smoke (one process)
1. With night off, fixed_random and nosleep match M9 exactly (as W9's check).
2. On seed 72, the first 1,024 decisions: print the mean b_t during day and during night wake decisions. Night should be clearly higher; if it is not,
   stop and report.
3. All arms finite on 512 decisions; kernel wall estimate (≤ 85 min each).

## Kernels
Stage 12 private CPU kernels (`ezzzzzx/w9b-s<seed>`, internet off, code dataset `ezzzzzx/pdp-code-w9b`), a resumable dispatcher and a summarizer as for
W9 (W9b ≤ 3 at once, account ≤ 5). Grok runs them; you do not have network.

## Outputs (summarize.py; numbers only, no verdict)
`results/w9b/RESULTS.md` + `results.json`. **At the top, the scoring table:** per seed, fixed_night − fixed_random, pressure_bits − fixed_night,
pressure_bits − fixed_random, with counts against the bars; and each arm's night_sleep_share and sleep_share. Then T14 per arm and seed, smoke output, walls.

## Pre-registered bars (orchestrator; do not change), T14
1. **Night is worth sleeping (replication):** fixed_night − fixed_random ≤ −0.10 on ≥ 8 of 12.
2. **Poor input finds the night:** pressure_bits night_sleep_share ≥ 0.60, mean over seeds.
3. **And earns it:** pressure_bits − fixed_random ≤ −0.07 on ≥ 8 of 12, and pressure_bits − fixed_night ≤ +0.05 on the mean.
Report only: pressure, pressure_opp.

## Also: H34d summarizer
`kaggle/h34d/summarize.py:50` asserts own_norm and dfa_norm agree to 1e-6 relative. In float32 they differ by up to 1.4e-6 on 2–6 chunks of 5,628. Change
the tolerance to 1e-5 relative (say so in RESULTS.md) and run the summarizer on the fetched outputs in `results/h34d/kernels/`. Change nothing else.

## Rules
About 90 minutes of your time. Files you may create or edit: `w9b_poor_input.py` (import from `w9_day_night.py`; do not edit it), `kaggle/w9b/**`,
`results/w9b/**`, `kaggle/h34d/summarize.py` (tolerance only), `results/h34d/RESULTS.md`, `results/h34d/results.json`. Do not touch `docs/`, `audit.md`,
other code, `.env`. Do not change bars. Never print secrets. Do not commit.
