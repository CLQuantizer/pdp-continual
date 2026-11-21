# W9d: judge the light at the eye, not in the cortex (codex: implement + smoke; grok runs kernels)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-21_codex_w9c_eyelid.md` (with both amendments), `results/w9c/RESULTS.md`, `w9c_eyelid.py`, `w9b_poor_input.py`.

## What W9c showed (orchestrator's diagnosis)
With poor = b > 1.5 × median, the agent still slept 46–68 decisions in every day and only 33% of its sleep fell at night (days 3–8). b is the glance's codelength under
**the cortex**. A glance costs many bits for two reasons: it is noise (night), or it is new and learnable (a room whose rule was just redrawn, every 128 decisions in
World5). The cortex cannot tell them apart from bits alone, so every rule change by day looks like nightfall. Retinas do not ask the cortex whether it is dark.
A light level is a property of the input itself: night input is incompressible to **any** model, while a new room's rule is still structured.

## Change: one new arm, `pressure_retina` (everything else exactly as W9c amendment 2: S +1 awake, −3 asleep; fall asleep when S ≥ 32 and poor; stay asleep while
poor and S > 0; forced at S ≥ 256 until S ≤ 128; the eyelid glance during sleep is used only for this signal)
- **r_t** = the glance's **own** order-0 codelength in bits per symbol: an adaptive KT estimator (α = 0.5 over the full symbol alphabet) that starts empty for each glance and
  codes that glance's symbols in order. No state crosses glances. It never sees the cortex.
- r̄_t = EMA (0.9) of r over wake and sleep decisions; m_t = median of r̄ over the last 256 wake decisions; **poor_t = r̄_t > 1.5 · m_t**.
- Arms: `fixed_random`, `fixed_night`, `pressure_eyelid` (amendment 2), `pressure_retina`. Fresh seeds 84–95 (W9c was never scored).

## Smoke (one process, seed 84, the full 2,048 decisions)
1. Print the mean r of day glances and of night glances, and of day glances in the 32 decisions after each rule redraw. Night must be clearly the highest.
2. The per-day table (day sleep, night sleep, forced) for pressure_retina, and the trace for t = 180–260.
3. **Gate:** pressure_retina's night share of sleep over days 3–8 ≥ 0.50. If it fails, stop and report the table.
4. Eyelid and baseline checks as W9c. Kernel wall estimate ≤ 85 min.

## Kernels, outputs
As W9c: 12 private CPU kernels `ezzzzzx/w9d-s<seed>` (code dataset `ezzzzzx/pdp-code-w9d`), resumable dispatcher (≤ 3 at once, account ≤ 5), summarizer;
`results/w9d/RESULTS.md` + `results.json` with the scoring table at the top (per seed pressure_retina − fixed_random and − fixed_night, sleep and night-sleep shares, counts).

## Pre-registered bars (orchestrator; do not change), T14, 12 seeds
1. pressure_retina night_sleep_share ≥ 0.60, mean over seeds.
2. pressure_retina − fixed_random ≤ −0.07 on ≥ 8 of 12.
3. pressure_retina − fixed_night ≤ +0.05, mean over seeds.
Report only: pressure_eyelid.

## Rules
About 90 minutes. Files you may create: `w9d_retina.py` (import from earlier w9 files; do not edit them), `kaggle/w9d/**`, `results/w9d/**`. Do not touch `docs/`,
`audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
