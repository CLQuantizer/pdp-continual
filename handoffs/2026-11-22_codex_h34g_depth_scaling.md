# H34g: with the slow start gone, does depth keep paying? And does a dense input help? (codex: implement + smoke; grok runs kernels)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-21_codex_h34f_zero_gate.md`, `results/h34f/RESULTS.md`, `h34f_zero_gate.py` and `h34b_plain_depth.py`.

## Why
H34f passed: initialising the mixer at [1, 0, …, 0] lets a local stack start as one layer and grow into its depth. The whole-stream cost of depth vanished (zgate_4 − plain_4
−0.36, zgate_6 − plain_6 −0.75) at no cost on the tail (+0.002). On the tail, though, depth still has diminishing returns (H34b: 4 → 6 layers −0.010). Two questions:
does depth keep paying at eight layers, and is a deep layer starved because it sees only the layer directly below it? DenseNet (Huang et al. 2017) gives each layer all
earlier features. The local, size-preserving version: layer k's input is the byte embedding plus the **mean** of the per-step-normalised features of layers 1 … k−1.

## Arms (everything else exactly as H34f's zgate arms: 180 KB of enwik8, tail 40–180 KB, N = 128, RANK 64, EMB 32, trace rule, Adam 1e-3, clip 1.0, chunk 128,
float32, own losses, mixer init [1, 0, …, 0], seeds 0 and 1)
- `zgate_2`: two layers (new).
- `zgate_8`: eight layers (new).
- `dense_6`: six layers; layer k ≥ 2 gets [emb(x); mean over j < k of f_j / rms(f_j)] in place of [emb(x); f_{k−1} / rms(f_{k−1})]. Same sizes as zgate_6.
- `zgate_6`: rerun as H34f (must match H34f to 1e-6).
Results for plain_1 and zgate_4 come from H34f (no rerun).

## Smoke (VPS CPU, `torch.set_num_threads(1)`)
1. zgate_6 reproduces H34f's bits on 8,192 bytes exactly.
2. dense_6 at layer 2 equals zgate_6 at layer 2 exactly (the mean of one feature vector is itself) on the first 2,048 bytes.
3. Prequential: first 1,024 bytes' bits identical for streams of 1,024 and 2,048 bytes (dense_6, zgate_8).
4. All arms finite; kernel wall estimates. **zgate_8 at 180 KB may exceed 85 minutes; if so, split each zgate_8 run into two kernels with a saved checkpoint
   (state, optimiser, traces, mixer) at 90 KB, resumed exactly, and show on 8,192 bytes split at 4,096 that a split run equals an unsplit one exactly.**

## Kernels
Private CPU kernels `ezzzzzx/h34g-<arm>-s<seed>` (plus `-part2` if split; underscores → hyphens; internet off; code dataset `ezzzzzx/pdp-code-h34g`; enwik8 as H34b),
resumable dispatcher (≤ 2 H34g at once, account ≤ 5; W9d runs alongside), summarizer. Grok runs them; you have no network.

## Outputs (numbers only)
`results/h34g/RESULTS.md` + `results.json`. **At the top:** tail and full, per seed and mean: zgate_8 − zgate_6, dense_6 − zgate_6, with counts against the bars; and the tail depth
curve plain_1, zgate_2, zgate_4, zgate_6, zgate_8 (means). Then per-layer own tail bits, mixer weights per 20 KB, walls.

## Pre-registered bars (orchestrator; do not change), both seeds, tail
1. **Depth keeps paying:** zgate_8 − zgate_6 ≤ −0.02.
2. **Dense input helps:** dense_6 − zgate_6 ≤ −0.03.
Report only: the depth curve, full-stream gaps.

## Rules
About 90 minutes. Files you may create: `h34g_depth.py` (import from `h34f_zero_gate.py`, `h34b_plain_depth.py`; do not edit them), `kaggle/h34g/**`, `results/h34g/**`.
Do not touch `docs/`, `audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
