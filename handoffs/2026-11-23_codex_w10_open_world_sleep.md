# W10: one life in the open world, with nights. Does the clockless sleep rule carry over? (codex: implement + smoke; grok runs kernels)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-16_grok_w6_open_world.md`, `w6_world.py`, `m10_w6.py` (prog_age_s on W6), `handoffs/2026-11-21_codex_w9d_retina.md`,
`w9d_retina.py`, `results/w9d_rep/RESULTS.md`.

## Why
The sleep rule passed on World5 (W9d, replicated): pressure +1 awake, −3 asleep; fall asleep when S ≥ 32 and the eye's own codelength (EMA) is above 1.5 × its waking
median; stay asleep while it stays poor and S > 0; forced at S ≥ 256 until S ≤ 128; an eyelid glance while asleep. World5 has drift but no births, deaths or noise
rooms. W6 does: rooms die and are born, and some rooms are noise streams **by day**. A rule tuned on one world must survive the next. W10 puts the W9d rule, unchanged,
into W6 with nights.

## World
W6 exactly as `w6_world.py` (rooms, births, deaths, noise kinds, seeds' world construction), plus W9's night: decision t is night when (t mod 256) ≥ 192, and at night each
glance symbol is replaced by a uniform symbol with p = 0.8 (rng world_seed + 60_000; exams clean), exactly as W9's `observe`. L as W6's life length (state it). Room
choice: prog_age_s as M10 on W6 (unchanged). Fresh seeds **120–131**.

## Arms
`nosleep`; `fixed_random` (share 0.25 at random times, as W9); `fixed_night` (the clock oracle); `pressure_retina` (W9d's rule, **every constant unchanged**). Sleep itself
(replay) is as W9: import it; do not change it.

## Smoke (one process)
1. With nights off and sleep off, nosleep reproduces M10's prog_age_s life on one W6 seed exactly.
2. pressure_retina on seed 120 for the full life: the per-day table (day sleep, night sleep, forced), and the share of **day** sleep that happens while the agent is in a
   noise room.
3. Eyelid checksum check as W9c/W9d. All arms finite. Kernel wall ≤ 85 min (split with exact checkpoints if not, as H34g did).

## Kernels
12 private CPU kernels `ezzzzzx/w10-s<seed>` (internet off; code dataset `ezzzzzx/pdp-code-w10`), resumable dispatcher (≤ 2 at once, account ≤ 5; H34h runs alongside),
summarizer. Grok runs them; you have no network.

## Outputs (numbers only)
`results/w10/RESULTS.md` + `results.json`. **At the top:** per seed, pressure_retina − nosleep, pressure_retina − fixed_random, pressure_retina − fixed_night, with counts;
each arm's sleep share and night-sleep share; pressure_retina's share of day sleep spent in noise rooms. Then W6's score (as M10 reports it) per arm and seed, walls.

## Pre-registered bars (orchestrator; do not change), 12 seeds, on W6's life score as M10 used it
1. **It still finds the night:** pressure_retina night_sleep_share ≥ 0.55, mean over seeds.
2. **Sleep earns its keep:** pressure_retina − fixed_random ≤ −0.05 on ≥ 8 of 12.
3. **Close to the clock:** pressure_retina − fixed_night ≤ +0.05, mean.
Report only: nosleep, the noise-room share of day sleep (sleeping in a noise room by day is not a fault: that input is worthless too).

## Rules
About 90 minutes. Files you may create: `w10_open_sleep.py` (import from `w6_world.py`, `m10_w6.py`, `w9_day_night.py`, `w9d_retina.py`; do not edit them), `kaggle/w10/**`,
`results/w10/**`. Do not touch `docs/`, `audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
