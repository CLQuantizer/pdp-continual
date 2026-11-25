# W11: the living machine gets a deep local cortex. Do Own, Wait and See help a life, not only a byte stream? (codex: implement + smoke + stage; grok runs kernels)

Read `/root/pdp/CLAUDE.md`, `docs/depth.md`, `w10_open_sleep.py` (W6 open world + nights + the W9d sleep rule), `m7_w3.py` (`BankCortex`), `m10_w6.py`, and the stack code in
`h34h_dense.py` / `h34g_depth.py` / `h34f_zero_gate.py` / `h34b_plain_depth.py`.

## Why
The depth tricks were found on passive byte streams. The machine that lives has a one-layer cortex (`BankCortex`: one roots-of-unity bank, |λ| = 0.98, with a channel
embedding row per room) that learns while it chooses rooms and sleeps. W11 asks whether a deep local cortex built with the three tricks makes the life itself better:
every glance it predicts, every exam it sits, every replay in sleep.

## Cortices
- `bank_1`: `BankCortex` unchanged (W10's).
- `deep_4`: a stack of 4 banks of the same type and size as `BankCortex`'s bank. Layer 1 is exactly `BankCortex`'s bank (same init, same channel embedding input). Layer k ≥ 2 reads
  [the symbol and channel embedding as layer 1 does; **the mean of the per-step-normalised features of layers 1 … k−1** (See)]. **Each layer has its own readout and its own
  loss** on the next symbol (Own); a GeoMixer combines the layers' predictions with weights **initialised [1, 0, 0, 0]** (Wait), learning as in H34f. No gradient crosses a
  layer. Every place the life uses the cortex's prediction (glance bits, exams, progress, replay in sleep, the frozen scorer for the eye's signal is **not** the cortex and
  stays unchanged) uses the mixture. Replay trains every layer on its own loss, as wake learning does.
- `deep_4_open`: as deep_4, mixer initialised to ones (report only; does Wait matter in a life).
Learning rates, clip, chunking and everything else as `BankCortex`.

## World and arms
W10 exactly (W6 + nights, prog_age_s room choice, the `pressure_retina` sleep rule, constants unchanged), L as W10. Fresh seeds **132–143**. Each seed runs the three cortices.

## Smoke (one process)
1. `deep_4` with layers 2–4 removed reproduces `bank_1`'s W10 life on seed 120 exactly (bits per glance, choices, sleeps), first 512 decisions.
2. On the first glance, `deep_4`'s mixture equals layer 1's prediction exactly (Wait).
3. No gradient crosses a layer: after one learn step, layer 1's parameters are identical whether layers 2–4 exist or not.
4. Eyelid and prequential checks as W10. Wall estimate; split with exact checkpoints if over 85 minutes.

## Kernels
12 seeds × 3 cortices as private CPU kernels `ezzzzzx/w11-<cortex>-s<seed>` (+ parts; underscores → hyphens; internet off; code dataset `ezzzzzx/pdp-code-w11`), a resumable
dispatcher (≤ 3 at once, account ≤ 5; it loads the token from `.env` itself), a summarizer. Grok runs them; you have no network.

## Outputs (numbers only)
`results/w11/RESULTS.md` + `results.json`. **At the top:** per seed, deep_4 − bank_1 on W6's life score (as M10/W10 use it) and on T_open as W10 reports it, with counts; sleep and
night-sleep shares per cortex; deep_4_open − deep_4. Then per-arm scores, the mixer weights over the life (per 512 decisions), walls.

## Pre-registered bars (orchestrator; do not change), 12 seeds, on W10's primary score (T_open)
1. **A deeper cortex lives better:** deep_4 − bank_1 ≤ −0.05 on ≥ 8 of 12.
2. **Sleep still works with it:** deep_4's night_sleep_share ≥ 0.55 (mean).
Report only: deep_4_open − deep_4.

## Rules
About 2 hours. Files you may create: `w11_deep_cortex.py` (import from the files above; do not edit them), `kaggle/w11/**`, `results/w11/**`. Do not touch `docs/`, `audit.md`,
other code, `.env`. Do not change bars. Never print secrets. Do not commit.

## Amendment (orchestrator, 25 Nov afternoon, before any scored run): reduced scope
The staged plan is 312 parts, deep cortices about 16 h per life (12 chained parts), several days of Kaggle time. **Run only seeds 132–137 with `bank_1` and `deep_4`** (12 lives;
`deep_4_open` dropped). Bars rescaled, decided now, before any scored life: (1) deep_4 − bank_1 ≤ −0.05 on ≥ 5 of 6; (2) deep_4 night_sleep_share ≥ 0.55 (mean). Codex: add
`--seeds` and `--cortices` filters to `kaggle/w11/dispatch.py` and `summarize.py` (state, locks and resume must respect the filter; the summarizer scores only the filtered set
against the rescaled bars), re-run the offline layout check, and change nothing else. Staging only; no network.
