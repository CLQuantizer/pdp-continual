# H34i: do Own, Wait and See hold on other data? (codex: implement + smoke + stage; the orchestrator launches the dispatcher)

Read `/root/pdp/CLAUDE.md`, `docs/depth.md` (the essay), `handoffs/2026-11-22_codex_h34h_dense_depth.md`, `results/h34h/RESULTS.md`, `h34h_dense.py`, `h34g_depth.py`,
`h34f_zero_gate.py`, `h34b_plain_depth.py`.

## Why
The three local depth tricks were found on one stream: the first 180 KB of enwik8 (Wikipedia XML). A trick is general only if it holds on data it was not found on. H34i
repeats the core ablation, every constant unchanged, on three new byte streams of 180,000 bytes each:
- `c4`: the first 180,000 bytes of `data/h35c/c4_val.bin` (C4 English validation, clean web text; manifest in `data/h35c/c4_val_manifest.json`).
- `bible`: bytes [0, 180,000) of `data/pg10.txt` (Project Gutenberg King James Bible, including its header; record the sha256).
- `mnist`: the raw pixel bytes of MNIST training images 0, 1, 2, … in order, row-major, 784 bytes per image, from `data/train-images-idx3-ubyte.gz` (skip the 16-byte header),
  the first 180,000 bytes. A different modality: 2-D images read as a 1-D byte stream.
Tail = bytes 40,000–180,000; full = all 180,000 bytes, as H34b.

## Arms (each on each stream, seeds 0 and 1; everything else exactly as the H34h code: N = 128, RANK 64, EMB 32, trace rule, Adam 1e-3, clip 1.0, chunk 128, float32,
own losses, per-step normalised inputs)
`plain_1`, `plain_6` (mixer init ones), `zgate_6` (init [1, 0, …]), `dense_6` (zero gate + mean of earlier features), `dense_6_ones` (dense, mixer init ones).
5 arms × 3 streams × 2 seeds = 30 lives.

## Smoke (VPS CPU, `torch.set_num_threads(1)`, one process)
1. With the stream set to enwik8, each arm reproduces its H34b/H34f/H34g/H34h bits on 8,192 bytes exactly.
2. For each new stream: print its sha256, byte count, 200 printable bytes (or, for mnist, one image as a 28×28 grid of hex), and plain_1's bits on its first 8,192 bytes.
3. Prequential first-1,024-bytes check on one new stream.
4. Wall estimates; split runs with exact checkpoints (as H34g/H34h) if any run would exceed 85 minutes.

## Kernels
Private CPU kernels `ezzzzzx/h34i-<stream>-<arm>-s<seed>` (+ `-part2`; underscores → hyphens; internet off; code dataset `ezzzzzx/pdp-code-h34i`; a data dataset
`ezzzzzx/pdp-data-h34i` holding the three 180,000-byte streams as files, created by the dispatcher). A resumable dispatcher (≤ 5 H34i at once, account ≤ 5; it loads
`KAGGLE_API_TOKEN` from `.env` itself, never prints it, retries a failed kernel once, persists state) and a summarizer. You have no network: stage and check offline only
(`--prepare-only` / an offline check as before). The orchestrator runs the dispatcher.

## Outputs (numbers only)
`results/h34i/RESULTS.md` + `results.json`. **At the top, one table per stream and one summary table:** per stream and seed, the three bar gaps and pass counts. Then tail and full
bits per arm, stream and seed, first-40 KB bits, walls, kernel ids. Include enwik8's numbers from H34b/H34f/H34g/H34h as a reference column.

## Pre-registered bars (orchestrator; do not change), on each of c4, bible and mnist, both seeds
1. **Own:** tail plain_6 − plain_1 ≤ −0.05.
2. **Wait:** full zgate_6 − plain_6 ≤ −0.10.
3. **See:** tail dense_6 − zgate_6 ≤ −0.02.
A trick is called general if its bar passes on all three streams. Report only: dense_6_ones − dense_6 (tail and full).

## Rules
About 2 hours. Files you may create: `h34i_generality.py` (import from the H34 files; do not edit them), `kaggle/h34i/**`, `results/h34i/**`, `data/h34i/**` (gitignored).
Do not touch `docs/`, `audit.md`, other code, `.env`. Do not change bars. Never print secrets. Do not commit.
