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

## Amendment (orchestrator, 21 Nov, after the smoke gate stopped: night share 0.359)
The trace shows why. The agent falls asleep at S ≥ 16 whenever b_t > m_t, so S drains to 0 again and again and never builds a day's worth of pressure. At nightfall S is about
10, so each night sleep lasts about 4 decisions, then 16 wake decisions. A low onset threshold makes the agent nap, not sleep. In Borbély's model sleep starts when
pressure is high **and** the circadian signal allows it. **Change only the onset threshold: fall asleep when S ≥ 96 and b_t > m_t** (or S ≥ 256, forced, until S ≤ 128).
Staying asleep is unchanged (while b_t > m_t and S > 0). Everything else, including the smoke gate and the bars, is unchanged. Rerun the smoke; if the gate passes,
stage the 12 kernels as the spec says. If it fails again, stop and report with the same trace.

## Amendment 2 (orchestrator, 21 Nov afternoon, after the smoke stopped at 3.6% night sleep)
The S ≥ 96 onset made things worse. The trace shows S at only 28 at dusk on day 1: in the first day, while everything is still being learned, b is high everywhere,
so the agent slept mid-day whenever it could, and its pressure fell out of phase with the night. The root cause is the threshold "b_t > m_t" with m_t the median of
**wake** b: wake decisions are mostly day, so by construction half of all day decisions count as "poor input". Night input is not slightly worse than the median; it
is twice as costly (7.4 against 3.3 bits/symbol in W9b).
**Change (replaces amendment 1):**
- poor_t = **b_t > 1.5 · m_t** (m_t as before).
- Fall asleep when S ≥ 32 and poor_t. Stay asleep while poor_t and S > 0. Forced sleep when S ≥ 256, until S ≤ 128. S +1 awake, −3 asleep, as before.
- **Smoke gate change:** run seed 84 for the full 2,048 decisions, print for each of the 8 days its sleep decisions by day and by night (and how many were forced), and
  gate on the night share of sleep over **days 3–8 ≥ 0.50** (the first two days are a learning transient). Also print the trace for t = 180–260 and t = 1,716–1,796.
Bars unchanged. If the gate passes, stage the 12 kernels. If it fails, stop and report with the per-day table.
