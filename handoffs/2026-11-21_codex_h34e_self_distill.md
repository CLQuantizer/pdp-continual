# H34e: the stack teaches its own layers. Deep supervision plus distillation from the stack's mixture (codex: implement + smoke; grok runs kernels)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-18_codex_h34b_plain_depth.md` (with amendments), `handoffs/2026-11-19_codex_h34d_dfa.md`, `results/h34d/RESULTS.md`,
and `h34b_plain_depth.py`.

## Why
Deep supervision is the only depth trick that has paid (H34b: 1 → 6 layers, −0.19 on the tail). Adding cross-layer credit by random feedback hurt (H34d: +0.28).
A credit signal that lives in **output space** needs no random matrices: every layer also learns to predict what the whole stack predicts. This is
self-distillation ("Be Your Own Teacher", Zhang et al. 2019) and deep mutual learning, done locally: the mixture's distribution p_mix is broadcast to every
layer, as a neuromodulator would be, and each layer's readout and bank learn from it with their own trace rule. No gradient crosses a layer.

## Arms (everything else as H34b's `plain_4`: 180 KB of enwik8, tail 40–180 KB, N = 128, RANK 64, EMB 32, trace rule, Adam 1e-3, clip 1.0, chunk 128, float32,
learned GeoMixer, per-step normalised inputs, seeds 0 and 1)
- `plain_4`: as H34b (rerun, must match H34b's numbers for that seed to 1e-6).
- `distill_4`: each layer k's error is (1 − λ)(p_k − y) + λ(p_k − p̄), with p̄ = the mixture's distribution at that step, **detached, computed before any
  layer learns from byte t**, and λ = 0.5. That is the gradient of (1 − λ)·CE(y, p_k) + λ·CE(p̄, p_k) on layer k's logits.
- `distill_4_top`: the same, but only layers 1–3 get the distillation term; layer 4 learns from y alone (report only).
- `distill_1`: the single layer with λ = 0.5 (it distils from a mixture of one, which is itself; checks that the term alone does nothing useful; report only).
λ is fixed; do not tune.

## Smoke (VPS CPU, `torch.set_num_threads(1)`)
1. With λ = 0, distill_4 equals plain_4 exactly (bits on 8,192 bytes).
2. The error formula equals autograd's gradient of the stated loss on layer k's logits (max abs ≤ 1e-6, float64, 512 bytes).
3. Prequential: first 1,024 bytes' bits identical for streams of 1,024 and 2,048 bytes.
4. All arms finite on 8,192 bytes; kernel wall estimate ≤ 85 min.

## Kernels
8 private CPU kernels `ezzzzzx/h34e-<arm>-s<seed>` (underscores → hyphens; internet off; code dataset `ezzzzzx/pdp-code-h34e`; enwik8 as H34b), resumable
dispatcher (≤ 2 at once, account ≤ 5), summarizer. Grok runs them.

## Outputs (numbers only)
`results/h34e/RESULTS.md` + `results.json`. **At the top:** tail distill_4 − plain_4 per seed and mean, with the count against the bar. Then tail and full bits per arm and
seed, each layer's own tail bits, the mean mixer weights, walls.

## Pre-registered bar (orchestrator; do not change)
**Distillation pays:** tail distill_4 − plain_4 ≤ −0.03 on both seeds. Report only: distill_4_top, distill_1, per-layer bits.

## Rules
About 90 minutes. Files you may create: `h34e_self_distill.py` (import from `h34b_plain_depth.py`; do not edit it), `kaggle/h34e/**`, `results/h34e/**`. Do not
touch `docs/`, `audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
