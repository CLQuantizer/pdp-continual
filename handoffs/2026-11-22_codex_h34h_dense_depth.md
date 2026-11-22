# H34h: dense inputs at depth, and are the three tricks independent? (codex: implement + smoke; grok runs kernels)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-22_codex_h34g_depth_scaling.md`, `results/h34g/RESULTS.md` and `h34g_depth.py`.

## Why
H34g: with zero-init gates, depth saturates (tail, 6 → 8 layers −0.009; bar −0.02 FAIL), while a dense input (each layer sees the mean of all earlier layers' normalised
features) gains −0.062 at six layers (PASS). Two questions for a clean ablation table: does depth pay again when layers see everything below them, and does the dense
input need the zero-init gate (or does each trick work alone)?

## Arms (everything else exactly as H34g: 180 KB of enwik8, tail 40–180 KB, N = 128, RANK 64, EMB 32, trace rule, Adam 1e-3, clip 1.0, chunk 128, float32, own
losses, seeds 0 and 1)
- `dense_8`: eight layers, dense input, mixer init [1, 0, …, 0] (checkpointed split as zgate_8 if over 85 minutes).
- `dense_4`: four layers, dense input, zero-init gate.
- `dense_6_ones`: dense_6 with the mixer initialised to all ones (as H34b's plain), to test whether dense needs the zero gate.
- `dense_6`: rerun as H34g (must match H34g to 1e-6).
Existing results reused (no rerun): plain_1, plain_4, plain_6 (H34b/H34f), zgate_4, zgate_6, zgate_8 (H34f/H34g).

## Smoke
1. dense_6 reproduces H34g on 8,192 bytes exactly; dense_4 at layers 1–4 equals dense_6's first four layers' own predictions on the first 2,048 bytes (the lower layers do
   not depend on the upper ones).
2. dense_6_ones equals dense_6 except for the mixer initialisation (diff the configs; print).
3. Prequential first-1,024-bytes check for dense_8. Exact split-resume check if split.
4. Wall estimates.

## Kernels
Private CPU kernels `ezzzzzx/h34h-<arm>-s<seed>` (+ `-part2`; underscores → hyphens; internet off; code dataset `ezzzzzx/pdp-code-h34h`), resumable dispatcher (≤ 3 at once,
account ≤ 5), summarizer. Grok runs them.

## Outputs (numbers only)
`results/h34h/RESULTS.md` + `results.json`. **At the top:** the ablation table, tail and full means and per seed, for depth ∈ {1, 4, 6, 8} × {plain (ones), zgate, dense (zero gate)} plus
dense_6_ones, filled from this run and the reused results; then the bar gaps with counts. Then per-layer own tail bits, mixer weights per 20 KB, walls.

## Pre-registered bars (orchestrator; do not change), both seeds, tail
1. **Dense inputs let depth pay again:** dense_8 − dense_6 ≤ −0.02.
2. **Dense beats extra depth:** dense_6 − zgate_8 ≤ −0.03.
Report only: dense_6_ones − dense_6 on the whole stream and the tail (does dense need the zero gate), dense_4 − zgate_4.

## Rules
About 90 minutes. Files you may create: `h34h_dense.py` (import from `h34g_depth.py` etc.; do not edit them), `kaggle/h34h/**`, `results/h34h/**`. Do not touch `docs/`, `audit.md`,
other code, `.env`. Do not change bars. Never print secrets. Do not commit.
