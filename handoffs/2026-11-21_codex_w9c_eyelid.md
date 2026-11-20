# W9c: sleep pressure that matches the day, with an eyelid (codex: implement + smoke; grok runs kernels)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-20_codex_w9b_poor_input.md`, `results/w9b/RESULTS.md`, your `w9_day_night.py` and `w9b_poor_input.py`.

## What W9b showed (orchestrator's diagnosis)
The signal works: a glance's own codelength is 3.3 bits/symbol by day and 7.4 at night. Night is worth sleeping through (fixed_night − fixed_random −0.108; 8 of 12
at ≤ −0.10, replicated). But `pressure_bits` slept at night only 32% of its sleep (bar 60%). The pressure arithmetic, not the signal, is the cause: a sleep
starts at S ≥ 32 and ends at S = 0 with S −4 per sleep decision, so each night sleep lasts about 8 decisions, then 32 wake decisions must pass before the next.
The night is 64 decisions. And while asleep the agent sees nothing, so it cannot tell that it is still dark.

## Change: one new arm, `pressure_eyelid` (everything else exactly as W9b: World5, period 128, L = 2,048, night (t mod 256) ≥ 192, p = 0.8, clean exams, prog_age_s)
- **Pressure:** S += 1 per wake decision, S −= 3 per sleep decision, S ≥ 0. (A day of 192 wake decisions is repaid by a night of 64 sleep decisions; this is the
  homeostat's set point, not a clock: the agent is never told t.)
- **Eyelid:** during a sleep decision the agent still receives that decision's glance from the room it is in. It is scored for b only (bits per symbol under the
  frozen scorer, as W9b) and is **not learned and not counted in any score**. b_t = EMA (0.9) of this per-glance value over wake and sleep decisions alike; m_t = median of
  b over the last 256 **wake** decisions.
- **Rule:** fall asleep when S ≥ 16 and b_t > m_t. Stay asleep while b_t > m_t and S > 0. Forced sleep when S ≥ 256, until S ≤ 128 (then wake). Episodes end at the
  end of an eight-decision visit as before.
- Arms: `fixed_random`, `fixed_night`, `pressure_bits` (W9b's), `pressure_eyelid`. Fresh seeds **84–95**.

## Smoke (one process)
1. fixed_random reproduces W9b's code path (as W9b's checks).
2. On seed 84, 1,024 decisions of pressure_eyelid: print S, b_t, m_t and asleep at every decision for t = 180–260, and the share of its sleep that falls at night.
   If that share is < 0.5, stop and report.
3. The eyelid glance changes no learner state (checksum of cortex and controller before and after a sleep decision, excluding b_t).
4. All arms finite on 512 decisions; kernel wall estimate ≤ 85 min.

## Kernels
Stage 12 private CPU kernels (`ezzzzzx/w9c-s<seed>`, internet off, code dataset `ezzzzzx/pdp-code-w9c`), resumable dispatcher (W9c ≤ 3 at once, account ≤ 5) and
summarizer, as for W9b. Grok runs them; you have no network.

## Outputs (numbers only)
`results/w9c/RESULTS.md` + `results.json`. **At the top:** per seed, pressure_eyelid − fixed_random, pressure_eyelid − fixed_night, with counts; each arm's sleep_share
and night_sleep_share. Then T14 per arm and seed, smoke output, walls.

## Pre-registered bars (orchestrator; do not change), T14, 12 seeds
1. **It sleeps at night without a clock:** pressure_eyelid night_sleep_share ≥ 0.60, mean over seeds.
2. **Sleep earns its keep:** pressure_eyelid − fixed_random ≤ −0.07 on ≥ 8 of 12.
3. **Close to the clock:** pressure_eyelid − fixed_night ≤ +0.05, mean over seeds.

## Rules
About 90 minutes. Smoke in the foreground, one process. Files you may create: `w9c_eyelid.py` (import from `w9_day_night.py`, `w9b_poor_input.py`; do not edit them),
`kaggle/w9c/**`, `results/w9c/**`. Do not touch `docs/`, `audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
