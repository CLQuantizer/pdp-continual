# H35b2: does the snapshot-progress score clean data for a neural learner? (codex: implement + smoke only; no network)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-18_codex_h36_start_h35b.md` (Job 2, H35b), the scoring table of `results/h35b/RESULTS.md`, and the
`prog_snap` arm in `handoffs/2026-11-18_grok_h35c_web_curiosity.md`; code `h35b_neural_diet.py`.

## Why
H35b (GRU learner): precondition met (oracle − rand −0.17), but `prog_peek` was **+0.06 worse than random** (0 of 6), reading 55% clean, near random's 50%.
RHO-LOSS read 63% clean (−0.08). Diagnosis: one Adam step on 256 bytes barely moves a GRU, and early in life any text teaches the shared weights about
equally, so a trial step's effect is small and not specific to the candidate. H35c's `prog_snap` measures progress differently: the candidate's loss
under the model from some steps ago minus under the current model, i.e. what the learner has **recently learned about this kind of text**. H35b2 tests
that score in H35b's exact setting, where junk labels are known.

## Arms (everything else exactly as H35b: corpus `h35_corpus.build(c)` first 1,600 docs, GRU 256, B = 200, K = 16, peeks 512 bytes, fresh seeds 6–11,
T_clean at 8 checkpoints)
- `rand`, `oracle`, `rho`: rerun as in H35b (for a same-code comparison).
- `prog_snap_k`, k ∈ {5, 20}: score = bits/byte on the candidate's first 512 bytes under the model snapshot from k documents read ago, minus under the
  current model (frozen forward passes). Before k documents have been read, the snapshot is the initial model.
Write `h35b2_snapshot.py` (import from `h35b_neural_diet.py`; do not edit it), kernels `kaggle/h35b2/s<c>/` (private, CPU, no internet, code dataset
`ezzzzzx/pdp-code-h35b2`), a resumable dispatcher (loads `.env`; account kernels ≤ 5; ≤ 2 at once) and `summarize.py` with a scoring table at the top
(per seed: oracle − rand, prog_snap_k − rand, 0.5·(oracle − rand), prog_snap_k − rho, counts) plus shares of documents read per kind.
**Smoke:** reproduce H35b's `rand` and `oracle` T_clean on seed 6 at B = 8 to 1e-9; the snapshot is a frozen copy (checksums); every arm runs at B = 8;
timing.
**Bars (orchestrator; do not change):** for the better of k ∈ {5, 20}, chosen by **mean over seeds 6–8** and then scored on **seeds 9–11** only:
(1) prog_snap − rand ≤ 0.5·(oracle − rand) on ≥ 2 of 3 scoring seeds; (2) prog_snap − rho ≤ −0.02 on ≥ 2 of 3.

Rules: about 60 minutes; files you may create: `h35b2_snapshot.py`, `kaggle/h35b2/**`, `results/h35b2/**`. Do not touch `docs/`, `audit.md`, other code,
`.env`. Never print secrets. Do not commit.
