# W9: a day-night world. Does sleep pressure plus input quality find the night by itself? (codex: implement + smoke only; no network; grok runs kernels)

Read `/root/pdp/CLAUDE.md`, `results/m9w5/RESULTS.md`, the scoring table of `results/m11b/RESULTS.md`, and the code `w5_world.py`, `w5_headroom.py`,
`m9_w5.py`, `m11b_w5.py`.

## Why (from Ez, 19 Nov)
Sleep in animals is not chosen by reward. It follows the two-process model (Borbély 1982): pressure S builds while awake and drains in sleep, and a circadian
rhythm makes night cheap to sleep through, because input is poor then. Our worlds had no night: every hour's input was equally valuable, so timing could
not matter (M9) and reward-chosen sleep under-slept (M9, M11, M11b). W9 adds night. Sleeping then sacrifices little incoming data, and replaying what
mattered before is worth more. Principles: the machine sleeps by its own state; sleep must earn its cost.

## World
`World5`, period 128 (drift as in M9), L = 2,048, with a **day-night cycle**: decisions t with (t mod 256) ≥ 192 are night (a quarter of each cycle). At night,
each symbol of a glance is independently replaced, with probability 0.8, by a uniform random symbol from the 41-symbol alphabet (rng
`default_rng(world_seed + 60_000)`). The learner learns from what it observes (the corrupted glance). Exams and T14 use clean continuations, exactly as in W5.
Room choice for every arm: prog_age_s (M8: τ = 256, ε 0.1, β 0.3); sleep visits and replay exactly as in M9 (8 decisions per visit, no travel).

## Arms
- `nosleep`.
- `fixed_random`: sleep share 0.25, visits placed at uniformly random times (rng `default_rng(world_seed + 61_000)`), non-overlapping.
- `fixed_night`: sleeps exactly the night decisions (share 0.25), the clock oracle.
- `pressure`: S starts at 0; +1 per wake decision, −4 per sleep decision, floored at 0. Start sleeping when S ≥ 64 and sleep (in 8-decision visits) until
  S = 0. Blind to input quality.
- `pressure_opp`: same S dynamics. Let g_t be the prequential gain of the last glance (bits under the cortex before learning that glance minus after), and
  w_t its EMA with decay 0.9 over wake decisions. Let m_t be the median of w over the last 256 wake decisions. Start sleeping when (S ≥ 32 and w_t < m_t)
  or S ≥ 128; once asleep, continue until S = 0. **Never told the clock.**
All constants fixed in advance; do not tune.

## Seeds, kernels
**Fresh world seeds 60–71.** 5 arms × 12 = 60 lives. Write `w9_day_night.py` (import; wrap or subclass only; do not edit the imported files), kernels
`kaggle/w9/s<seed>/` (private, CPU, no internet, code dataset `ezzzzzx/pdp-code-w9`; split by arm if > 80 min), a resumable dispatcher (loads `.env`;
account kernels ≤ 5; ≤ 3 at once) and `summarize.py`.
**Smoke (VPS, one process):** (1) with night disabled, `nosleep` and a fixed 0.20 arm reproduce M9's code on its seed 24 at L = 256 (T14 to 1e-6); (2) print the
first 64 symbols of one glance by day and by night with the corruption visible; (3) for `pressure_opp`, print t, night flag, S, w_t, m_t and sleep state for
the first 512 decisions; (4) all arms finite at L = 256; timing.
**Outputs (summarize.py, numbers only):** at the top a **scoring table** per seed: fixed_night − fixed_random, pressure_opp − fixed_night,
pressure_opp − fixed_random, pressure − fixed_random, fixed_random − nosleep (T14), with means and counts meeting each threshold. Then T14 per arm, sleep share per
arm, **share of each arm's sleep that falls at night**, and smoke output.

## Pre-registered bars (orchestrator; do not change), T14
1. **Timing matters when nights are poor:** fixed_night − fixed_random ≤ −0.10 on ≥ 9 of 12.
2. **Pressure plus opportunity finds the night:** pressure_opp − fixed_night ≤ +0.03 (mean) and pressure_opp − fixed_random ≤ −0.07 on ≥ 8 of 12.
Report only: pressure (no opportunity term), sleep at night, fixed_random − nosleep.

Rules: about 90 minutes; files you may create: `w9_day_night.py`, `kaggle/w9/**`, `results/w9/**`. Do not touch `docs/`, `audit.md`, other code, `.env`.
Never print secrets. Do not commit.
