# Two jobs (codex: implement + smoke only; no network): H34a3 cascade stack, and the H36 budget fix

Read `/root/pdp/CLAUDE.md`, then `handoffs/2026-11-17_codex_h34a2_logit_sum.md`, the scoring table at the top of `results/h34a2/RESULTS.md`,
and `handoffs/2026-11-17_codex_h36_curious_patching.md` with your own `handoffs/2026-11-17_codex_h36_curious_patching.out.md`.

## Job 1: H34a3, the cascade (partial-sum) stack
**Diagnosis of H34a2 (orchestrator):** the logit-sum stack got worse with depth (sum_1 3.505, sum_2 3.575, sum_4 3.878 bits/byte) because every
layer received the identical error p − y at the same moment: D layers took the same step on the same mistake, overshooting, with nothing to
make them specialise. Boosting fits stages to what the earlier stages leave. Meanwhile the plain stack (own loss + learned mixer) improved with
depth on the tail (plain_1 3.311 → plain_4 3.126) and beat wide_1 (3.308).
**Cascade:** S_k = Σ_{j≤k} z_j (partial sums of logits, layer 1 first). Layer k learns from g_k = softmax(S_k) − y: it corrects the prediction of
layers 1..k, so layer 1 learns on its own loss, layer 2 on what layer 1 leaves, and so on. The stack predicts softmax(S_D). Everything else
exactly as H34a2 (`h34a2_logit_sum.py`: data 300 KB, N = 128, stateless per-step input normalisation, zero-init readouts, f detached, no gradient
across layers, seeds 0–1). Arms: `casc_2`, `casc_4` (casc_1 ≡ sum_1, reuse H34a2's numbers; confirm by running casc_1 for 8,192 bytes against
sum_1 in the smoke test, identical to 1e-9). Write `h34a3_cascade.py` (import from h34a2; do not edit it), kernel folders `kaggle/h34a3/<arm>-s<seed>/`
(4 kernels, private, CPU, no internet, code dataset `ezzzzzx/pdp-code-h34a3`), a resumable dispatcher (loads `.env`; account running kernels ≤ 5;
at most 2 H34a3 at once) and `kaggle/h34a3/summarize.py`, whose RESULTS.md starts with a scoring table: per seed, casc_4 − sum_1, casc_4 − plain_4,
casc_4 − wide_1 (full run and tail 50–300 KB; H34a2 numbers read from `results/h34a2/results.json`), with counts of seeds meeting each threshold.
**Smoke:** (1) casc_1 ≡ sum_1 as above; (2) g_k equals autograd's gradient of cross_entropy(S_k) w.r.t. z_k for every k on 512 bytes (≤ 1e-6);
(3) identity at init; (4) prequential prefix check; (5) the first 2,048 bytes ≤ 9 bits/byte for casc_4; (6) wall estimate ≤ 85 min.
**Bars (orchestrator; do not change), full 300 KB, both seeds:** casc_4 − sum_1 ≤ −0.10; casc_4 − plain_4 ≤ −0.05; casc_4 − wide_1 ≤ −0.03.

## Job 2: H36, fix the budget and finish the smoke test
The entropy rule chose ρ = 0.339 (allowed 0.23–0.27), so smoke stopped correctly. Replace the percentile threshold for **all three adaptive rules**
(entropy, gain, progress) by a feedback-controlled threshold: θ ← θ + η (chosen_t − 0.25) after every byte, η = 0.01 × (running std of the score,
EMA 0.999), initial θ = the score's first value; a byte is chosen when score > θ (ties: not chosen). Keep everything else in the spec.
Refresh the code snapshot so the kernels import `h35_corpus.build(seed)` (it exists now; do not edit it). Rerun smoke checks 3–5 for all rules
on the stand-in (achieved ρ within 0.25 ± 0.02 over 256 KB) and one 64 KB check on `h35_corpus.build(0)`. The RESULTS.md written by
`kaggle/h36/summarize.py` must start with the scoring table the H36 spec describes.

## Rules (both jobs)
About 120 minutes in total; smoke in the foreground. Files you may create or edit: `h34a3_cascade.py`, `kaggle/h34a3/**`, `results/h34a3/**`,
`h36_curious_patching.py`, `kaggle/h36/**`, `results/h36/**`. Do not touch `docs/loop.md`, `audit.md`, `h34a2_logit_sum.py`, `h34a_local_depth.py`,
`h35_*`, `m11_w6.py`, `kaggle/m11w6/**`, any other code, `.env`. Gates and bars are the orchestrator's; if one looks wrong, stop and report.
Never print or upload secrets. Do not commit.
