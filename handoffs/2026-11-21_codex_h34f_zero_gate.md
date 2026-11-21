# H34f: zero-init gates. Let the deep stack start as a one-layer stack (codex: implement + smoke; grok runs kernels)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-18_codex_h34b_plain_depth.md` (with amendments), `results/h34b/RESULTS.md`, `results/h34e/RESULTS.md` and `h34b_plain_depth.py`.

## Why
Deep supervision is the depth trick that pays (H34b: 1 → 6 layers, −0.19 on the tail). Two cross-layer credit signals have failed: random feedback (H34d, +0.28)
and self-distillation (H34e, +0.047 on the tail, +0.94 over the whole stream). A cost shows in every deep run: **deep stacks start slowly** (H34b, first 40 KB:
plain_4 6.04 bits/byte against plain_1 4.48). The GeoMixer starts every layer's weight at 1, so the stack's prediction is the sum of all layers' logits from the start,
while the upper layers are still untrained and their inputs are still moving. In backprop nets, the same problem is solved by starting each residual branch at
zero (zero-γ, Goyal et al. 2017; SkipInit, De & Smith 2020; ReZero, Bachlechner et al. 2020): the deep net starts as a shallow one and grows into its depth.
Here the equivalent is the mixer.

## Arms (everything else exactly as H34b: 180 KB of enwik8, tail 40–180 KB, N = 128, RANK 64, EMB 32, trace rule, Adam 1e-3, clip 1.0, chunk 128, float32,
per-step normalised inputs, own losses, seeds 0 and 1)
- `plain_4`: as H34b (must match H34b's numbers to 1e-6).
- `zgate_4`: the same, with the GeoMixer's weights initialised to **[1, 0, 0, 0]** (layer 1 only). The mixer learns as before, so upper layers enter when they help.
- `zgate_6`: the same at six layers, init [1, 0, 0, 0, 0, 0].
- `plain_6`: as H34b (rerun; must match H34b).
- `plain_1`: as H34b (rerun; must match H34b).
Nothing is tuned.

## Smoke (VPS CPU, `torch.set_num_threads(1)`)
1. On the first chunk, zgate_4's mixed prediction equals plain_1's exactly.
2. plain_4 and plain_1 reproduce H34b's bits on 8,192 bytes exactly.
3. Prequential: first 1,024 bytes' bits identical for streams of 1,024 and 2,048 bytes (zgate_4).
4. All arms finite; kernel wall estimate ≤ 85 min.

## Kernels
10 private CPU kernels `ezzzzzx/h34f-<arm>-s<seed>` (underscores → hyphens; internet off; code dataset `ezzzzzx/pdp-code-h34f`; enwik8 as H34b), resumable
dispatcher (≤ 2 at once, account ≤ 5), summarizer. Grok runs them; you have no network.

## Outputs (numbers only)
`results/h34f/RESULTS.md` + `results.json`. **At the top:** per seed and mean, full-stream and tail differences zgate_4 − plain_4, zgate_6 − plain_6, zgate_6 − plain_1,
with counts against the bars. Then bits per arm and seed for the first 40 KB, tail and full stream, the mean mixer weights over time (per 20 KB), walls.

## Pre-registered bars (orchestrator; do not change), both seeds
1. **The slow start goes away:** full-stream zgate_4 − plain_4 ≤ −0.10.
2. **At no cost later:** tail zgate_4 − plain_4 ≤ +0.01.
3. **Depth now pays from the first byte:** full-stream zgate_6 − plain_1 ≤ −0.05.
Report only: zgate_6 − plain_6 on the tail.

## Rules
About 90 minutes. Files you may create: `h34f_zero_gate.py` (import from `h34b_plain_depth.py`; do not edit it), `kaggle/h34f/**`, `results/h34f/**`. Do not touch
`docs/`, `audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
