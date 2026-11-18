# Two jobs (codex: implement + smoke only; no network): finish H36 with a startup rule, and build H35b (curious data diet with a neural learner)

Read `/root/pdp/CLAUDE.md`, then `handoffs/2026-11-17_codex_h36_curious_patching.md`, your
`handoffs/2026-11-18_codex_h34a3_cascade_and_h36_fix.out.md`, `handoffs/2026-11-17_grok_h35_data_diet.md` and the scoring table in
`results/h35/RESULTS.md`; code `h36_curious_patching.py`, `h35_corpus.py`, `h35_data_diet.py`.

## Job 1: H36 startup rule (orchestrator's decision)
For `gain` and `progress`, a bucket with fewer than 8 samples is **not scored**: such a byte is chosen with probability 0.25 (rng
`default_rng(3_000 + seed)`), and these bytes are excluded from the θ update and from the running std. θ starts at the first finite score; until
there is one, every byte follows the 0.25 rule. Nothing else changes. Rerun smoke checks 3–5 for all four rules (ρ within 0.25 ± 0.02 over 256 KB on
the stand-in and over 64 KB of `h35_corpus.build(0)`), refresh the kernel code snapshot, and say "H36 ready" at the end if all pass.

## Job 2: H35b, the curious data diet with a neural learner
H35 used a count learner, where each document's statistics are separate. A neural learner shares weights across everything it reads, which blurred
per-source progress before (H25). H35b repeats H35 with a byte GRU.
- **Corpus:** `h35_corpus.build(c)` for **fresh seeds c ∈ {6, …, 11}**, but use only the first **1,600** documents of the shuffled pool (about half
  junk) as the pool. Same clean test set and reference set.
- **Learner:** byte GRU, hidden 256, embedding 32, Adam lr 3e-4, clip 1.0, truncated BPTT 64, one Adam step per 128 bytes (as H36's expensive
  model, every byte). Reading a document = one prequential pass (score each 128-byte chunk, then step), hidden state reset at each document start.
- **Budget:** B = **200** documents read. Checkpoints every 25 documents (8); T_clean = mean of the clean-test bits/byte at the checkpoints (frozen model).
- **Policies:** as H35 (`rand`, `oracle`, `high_loss`, `rho`, `prog_peek`), with K = **16** candidates and peeks of 512 bytes:
  - `rho`'s reference: the same GRU trained once on the reference set (one pass) and frozen.
  - `prog_peek`: deep-copy the GRU and its optimiser, take one Adam step on bytes 0–255 of the candidate (score is not counted), then score = bits
    saved by that step on bytes 256–511 of the candidate plus on a 1,024-byte history reservoir (4 pieces of 256 bytes from documents read,
    reservoir sampling with rng `default_rng(2_000 + c)`), divided by 1,280. Discard the copy.
- **Files:** `h35b_neural_diet.py` (import `h35_corpus`; do not edit it), kernels `kaggle/h35b/s<c>/` (one per seed; split policies into two kernels per
  seed if the timing says > 80 min; ids `ezzzzzx/h35b-s<c>[-p]`; private; CPU; no internet; code dataset `ezzzzzx/pdp-code-h35b`), a resumable
  dispatcher `kaggle/h35b/dispatch.py` (loads `.env`; account running kernels ≤ 5; at most 2 H35b at once; one retry) and `kaggle/h35b/summarize.py`.
- **Smoke (VPS, `torch.set_num_threads(1)`):** (1) prequential prefix check on one document; (2) the trial copy leaves the live GRU and optimiser
  bit-identical (checksum); (3) on seed 6 with B = 8, every policy runs; print kinds read; (4) time B = 25 for `prog_peek` and estimate kernel walls.
- **Outputs (summarize.py; numbers only):** at the top a **scoring table** per seed: oracle − rand, prog_peek − rand, 0.5·(oracle − rand),
  prog_peek − high_loss, with counts meeting each threshold. Then T_clean and final per policy and seed, shares of documents read per kind, walls,
  kernel ids.
- **Bars (orchestrator; do not change):** precondition oracle − rand ≤ −0.10 on T_clean, 5 of 6; (1) prog_peek − rand ≤ 0.5·(oracle − rand), 5 of 6;
  (2) prog_peek − high_loss ≤ −0.05, 5 of 6.

## Rules
About 120 minutes in total; smoke in the foreground. Files you may create or edit: `h36_curious_patching.py`, `kaggle/h36/**`, `results/h36/**`,
`h35b_neural_diet.py`, `kaggle/h35b/**`, `results/h35b/**`. Do not touch `docs/`, `audit.md`, `h35_corpus.py`, `h35_data_diet.py`, `h34a3_*`,
any other code, `.env`. Bars are the orchestrator's. Never print secrets. Do not commit.
