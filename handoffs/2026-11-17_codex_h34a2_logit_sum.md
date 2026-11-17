# H34a2: the leftover stack done right. A residual in logit space, with stable inputs (codex: implement + smoke; orchestrator runs the dispatcher)

Read `/root/pdp/CLAUDE.md`, then `handoffs/2026-11-16_codex_h34a_local_depth.md` (with both amendments), and `h34a_local_depth.py`
and `kaggle/h34a/` (your own code from H34a).

## What H34a showed (orchestrator's diagnosis)
1. **Every D = 4 arm blew up in the first 50 KB** (slice 0 at ~40,000 bits/byte, both seeds, plain and leftover alike), then recovered
   (later slices ~3.3–3.9). Cause, a spec error: the running RMS starts at 1 with decay 0.999, while the features' real RMS is 5–600, so
   for the first ~1,000 steps deep layers receive unnormalised inputs and the cascade explodes.
2. **At D = 2 (clean), depth helps the plain stack** (tail 50–300 KB: plain_1 3.311, plain_2 3.171) **and the leftover signal hurts**
   (left_2 3.254). Cause, a design error: with a learned mixer, a layer whose weight w_k is small gets the error p_mix − y, which does not
   depend on its own logits, so nothing bounds them (left_2's layer 2 scores 38 bits alone at weight 0.01). That signal is not the
   gradient of anything the layer controls. The exact form of the idea is a plain **sum of logits with fixed unit weights**:
   p = softmax(Σ_k z_k). Then g = p − y is the exact gradient of the stack's loss on every layer's logits, it depends on each layer's own
   output (closed loop), and a zero-init layer adds nothing (identity). That is boosting, a residual in logit space.

## Changes from H34a (everything else as in H34a: data, the bank layer, N = 128, RANK 64, EMB 32, trace rule, Adam 1e-3, clip 1.0,
chunk 128, float32, zero-init readouts, f detached, no gradient across layers, seeds 0 and 1)
- **Data:** the first **300,000 bytes** of enwik8 for every arm (as H34a's fallback).
- **Input normalisation (all arms):** drop the running RMS. Layer k ≥ 2 gets `f_{k−1,t} / rms(f_{k−1,t})`, where rms is the root mean
  square **across that vector's features at that step** (+1e-6), computed on the detached features. Causal, stateless, no warm-up.
- **Arms:**
  - `plain_D`, D ∈ {1, 2, 4}: as H34a (learned GeoMixer, each layer on its own loss p_k − y).
  - `sum_D`, D ∈ {1, 2, 4}: **no mixer**. p = softmax(Σ_k z_k). Every layer's error is g = p − y (identical for all layers, exact).
    The score is p's bits.
  - `sumown_D`, D ∈ {2, 4}: the same combination p = softmax(Σ_k z_k) for scoring, but each layer learns from its own loss p_k − y. This
    isolates the learning signal at a fixed combiner.
  - `sum_4_noskip`: as `sum_4`, layers k ≥ 2 get only the normalised f (no byte embedding).
  - `wide_1`: D = 1, N = 512, as H34a (sum_1 and plain_1 differ only by the mixer at D = 1; use the sum form for wide_1).
  10 arms × 2 seeds = 20 lives.
- **Diagnostics per layer:** the layer's own bits, the mean and max |logit| per 50 KB slice, and (plain only) the mean mixer weight in
  the last 50 KB.

## Your job (codex): implement + smoke test only (your sandbox has no network)
1. **H34a write-up:** write `kaggle/h34a/summarize.py`, which builds `results/h34a/RESULTS.md` and `results/h34a/results.json` from the
   fetched kernel outputs in `results/h34a/kernels/*/results.json` (every output in H34a's spec; numbers only, no verdict; add a table of
   bits per 50 KB slice and the tail mean over 50–300 KB). Do not run it if some kernels are missing; the orchestrator runs it when the
   last H34a kernels are fetched.
2. **H34a2:** `h34a2_logit_sum.py` (import from `h34a_local_depth.py` where possible; do not edit it), kernel folders
   `kaggle/h34a2/<arm>-s<seed>/` (ids `ezzzzzx/h34a2-<arm>-s<seed>`, underscores → hyphens; private; CPU; no internet; code from
   private dataset `ezzzzzx/pdp-code-h34a2`; enwik8 dataset as H34a), a resumable foreground dispatcher `kaggle/h34a2/dispatch.py`
   (at most 3 kernels at once, one retry; it must load `KAGGLE_API_TOKEN` from `.env` itself), and `kaggle/h34a2/summarize.py` that writes
   `results/h34a2/RESULTS.md` + `results.json` with the outputs listed below.

## Smoke test (VPS CPU, `torch.set_num_threads(1)`)
1. **Exact gradient:** for `sum_2` on 512 bytes, g = p − y on each layer's logits equals autograd's gradient of
   cross_entropy(Σ z_k) with respect to each z_k (max abs ≤ 1e-6).
2. **Identity at init:** adding a zero-init layer to a sum stack leaves p unchanged to 1e-9 on the first chunk.
3. **No early blow-up:** run `sum_4` and `plain_4` on the first 16,384 bytes: report bits per 2,048 bytes and max |logit| per layer; the
   first 2,048 bytes must be ≤ 9 bits/byte for both.
4. **Prequential:** first 1,024 bytes' bits identical for 1,024 vs 2,048 bytes of stream, for `sum_4`.
5. Trace exactness for layer 2 as H34a's check 2 (≤ 1e-4), with the new normalisation.
6. All arms finite on 8,192 bytes; time 32,768 bytes of `sum_4` and `wide_1`; estimate kernel walls (must be ≤ 85 min each at 300 KB;
   if not, stop and report).
If any check fails, stop and report.

## Outputs (written by summarize.py; numbers only, no verdict)
bits/byte per arm and seed and the mean; per 50 KB slice; tail mean over 50–300 KB; per-layer diagnostics; the gaps (per seed and mean):
sum_4 − sum_1, sum_2 − sum_1, sum_4 − sumown_4, sum_2 − sumown_2, sum_4 − wide_1, plain_4 − plain_1, sum_4 − plain_4,
sum_4_noskip − sum_4; the compute table (as H34a); smoke output; walls; kernel ids.

## Pre-registered bars (orchestrator scores on the full 300 KB mean over seeds 0–1, both seeds the same sign; do not change)
1. **Depth pays in the logit-sum stack:** sum_4 − sum_1 ≤ −0.10 bits/byte.
2. **The shared exact error is what makes it pay:** sum_4 − sumown_4 ≤ −0.05.
3. **Depth beats width at matched compute:** sum_4 − wide_1 ≤ −0.03.
Report only: the plain series, sum_2, noskip.
Kill rules: if bar 1 fails while plain_4 − plain_1 ≤ −0.10, depth pays with own losses but not with the shared error: record it. If bar 1
fails for both, local stacking at N = 128 does not compose beyond two layers on text; next is learned cross-layer credit (DFA into input
maps). If bars 1–2 pass and 3 fails, the stack is only extra width.

## Budget and rules
- About 90 minutes of your time. Smoke in the foreground.
- Files you may create: `h34a2_logit_sum.py`, `kaggle/h34a2/**`, `results/h34a2/**`, `kaggle/h34a/summarize.py`.
- Do not touch: `docs/loop.md`, `audit.md`, `h34a_local_depth.py`, `kaggle/h34a/dispatch.py`, `results/h34a/**` (only summarize.py
  writes there, later), `m11_w6.py`, `kaggle/m11w6/**`, `results/m11w6/**` (M11 kernels are running), any other code, `.env`.
- Gates and bars are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.
