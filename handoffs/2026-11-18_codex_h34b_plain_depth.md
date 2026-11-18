# H34b: the stack that works. Depth, the gate and the skip in the plain local stack (codex: implement + smoke only; no network)

Read `/root/pdp/CLAUDE.md`, the scoring tables of `results/h34a2/RESULTS.md` and `results/h34a3/RESULTS.md`, and `h34a2_logit_sum.py`.

## Why (orchestrator)
Five local stacks so far, a 2×2 plus the cascade. Signal {own loss, shared error} × combiner {learned geometric mixer, fixed logit sum}:
own + mixer (`plain`) improves with depth on the tail (plain_1 3.311 → plain_2 3.178 → plain_4 3.126) and beats a 4×-wide single layer (3.308);
shared + mixer (`left`), shared + sum (`sum`), own + sum (`sumown`) and the cascade (`casc`) all get worse with depth. The working recipe:
**every layer is a complete predictor on its own loss, and a learned gate decides how much each counts.** It resembles deep supervision plus a
learned per-branch scale (ReZero, LayerScale) more than a residual decomposition. H34b tests it: does depth keep paying, is the gate what makes it
safe, and is the input skip needed?

## Setup
Exactly H34a2's `plain` arm (`h34a2_logit_sum.py`: N = 128 per layer, RANK 64, EMB 32, stateless per-step input normalisation, zero-init
readouts, f detached, no gradient across layers, own loss p_k − y, learned GeoMixer with a single weight set, Adam 1e-3 / mixer 0.003, chunk 128,
float32), on the first **250,000 bytes** of enwik8. Seeds 0 and 1.
**Score: tail bits/byte over bytes 50,000–250,000** (the long-run rate of a living learner; all arms start slowly, and the first 50 KB are reported
separately), plus the full-run mean.

## Arms
- `plain_D`, D ∈ {1, 2, 4, 6}.
- `plain_4_fixmix`: as plain_4, but the mixer weights are frozen at 1/D each (no learned gate).
- `plain_4_noskip`: as plain_4, layers k ≥ 2 get only the normalised f (no byte embedding).
- `wide_6`: D = 1 with N = 768 (θ_j = 2πj/768), the width control for D = 6 (report its multiply-adds against plain_6's; flag a difference > 20%).
7 arms × 2 seeds = 14 kernels.

## Your job
`h34b_plain_depth.py` (import from h34a2; do not edit it), kernel folders `kaggle/h34b/<arm>-s<seed>/` (ids `ezzzzzx/h34b-<arm>-s<seed>`, underscores →
hyphens; private; CPU; no internet; code dataset `ezzzzzx/pdp-code-h34b`; enwik8 dataset as H34a), a resumable dispatcher (loads `.env`; account running
kernels ≤ 5; at most 3 H34b at once; one retry) and `kaggle/h34b/summarize.py`, whose RESULTS.md starts with a **scoring table**: per seed and mean,
plain_6 − plain_4, plain_4_fixmix − plain_4, plain_6 − wide_6 (tail), with counts meeting each threshold; then per arm the tail and full bits, per-layer
own bits and mixer weights, slice table (25 KB), compute table.
**Smoke (VPS, one thread):** (1) plain_4 reproduces H34a2's plain_4 bits on its first 8,192 bytes to 1e-9; (2) fixmix's weights stay exactly 1/D;
(3) prequential prefix check for plain_6; (4) the first 2,048 bytes ≤ 9 bits/byte for plain_6 and wide_6; (5) wall estimates ≤ 85 min each (if plain_6
or wide_6 exceeds it, use 200,000 bytes for every arm with the tail 50,000–200,000 and say so).

## Pre-registered bars (orchestrator; do not change), tail bits, both seeds
1. **Depth keeps paying:** plain_6 − plain_4 ≤ −0.03.
2. **The gate makes depth safe:** plain_4_fixmix − plain_4 ≥ +0.05.
3. **Depth beats width:** plain_6 − wide_6 ≤ −0.10.
Report only: noskip, the full-run gaps.

## Rules
About 90 minutes; smoke in the foreground. Files you may create: `h34b_plain_depth.py`, `kaggle/h34b/**`, `results/h34b/**`. Do not touch `docs/`,
`audit.md`, `h34a*_*.py`, `h35*`, `h36*`, other kaggle/results folders, `.env`. Bars are the orchestrator's. Never print secrets. Do not commit.

## Amendment (orchestrator, 18 Nov 16:30, after smoke)
plain_6 scored 10.9 bits/byte on its first 2,048 bytes (bound 9) and needs 87 min at 200 KB (bound 85). The 2,048-byte bound guarded against H34a's blow-up
(~40,000 bits/byte). 10.9 is ordinary early over-confidence, so **the bound becomes ≤ 12 bits/byte.** **Data becomes the first 180,000 bytes for every
arm, with the tail scored on 40,000–180,000.** Everything else and all bars are unchanged.
