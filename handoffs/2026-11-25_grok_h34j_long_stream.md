# H34j: do Own, Wait and See still pay on a stream five times longer? (grok: stage and run on Kaggle CPU)

Read `/root/pdp/CLAUDE.md`, `docs/depth.md`, `handoffs/2026-11-24_codex_h34i_generality.md`, `kaggle/h34i/README.md`, `h34i_generality.py`, `h34h_dense.py` (its checkpointed split runs).

## Why
Every depth result so far is on 180,000 bytes. A trick that only speeds up a short run is a warm-up trick. H34j runs the core arms on the first **1,000,000 bytes** of enwik8.

## Arms (every setting unchanged from H34h/H34i; seeds 0 and 1)
`plain_1`, `zgate_6`, `dense_6`, `dense_8`. Tail = bytes 200,000–1,000,000; full = all 1,000,000; also report bits per 100 KB slice.

## How
Use the existing code unchanged (import `h34i_generality.py` / `h34h_dense.py`; pass the stream and length; if a length or stream is hard-coded, write a thin wrapper
`h34j_long.py` that only sets them). Split each run into kernels of ≤ 85 minutes with the exact checkpoint/resume the H34g/H34h code already has (state, optimiser, traces,
mixer); before any scored kernel, show on 8,192 bytes split at 4,096 that a split run equals an unsplit one exactly. Stage private CPU kernels `ezzzzzx/h34j-<arm>-s<seed>-p<k>`
(internet off; code dataset `ezzzzzx/pdp-code-h34j`; enwik8 as H34b), chained so part k+1 starts from part k's checkpoint. Run them in the foreground with ≤ 4 at once
(account ≤ 5; codex may stage W11 later). One retry per failed part. Token from `.env` via `set -a; . ./.env; set +a`; never print it.

## Smoke
1. Each arm reproduces its H34h numbers on the first 8,192 bytes exactly.
2. The split-equals-unsplit check above.
3. Timing and the part plan (how many parts per arm).

## Outputs (numbers only)
`results/h34j/RESULTS.md` + `results.json`. **At the top:** tail and full, per seed and mean: zgate_6 − plain_1, dense_6 − zgate_6, dense_8 − dense_6, with counts against the bars;
then bits per 100 KB slice per arm and seed, kernel ids, walls.

## Pre-registered bars (orchestrator; do not change), tail 200 KB–1 MB, both seeds
1. **Depth still pays:** zgate_6 − plain_1 ≤ −0.10.
2. **See still pays:** dense_6 − zgate_6 ≤ −0.02.
3. **Depth with See still pays at eight:** dense_8 − dense_6 ≤ −0.01.

## Rules
Files you may create: `h34j_long.py`, `kaggle/h34j/**`, `results/h34j/**`. Do not edit other code, `docs/`, `audit.md`, `.env`. Do not change bars. Never print secrets. Do not commit.
Run in the foreground until every part has finished, then summarize.

## Reassigned to codex (orchestrator, 25 Nov): grok is out of balance
Codex: implement, smoke and **stage** only (you have no network). Write a resumable dispatcher `kaggle/h34j/dispatch.py` (loads `KAGGLE_API_TOKEN` from `.env` itself, never prints
it, chains parts, ≤ 4 H34j at once, account ≤ 5, one retry per part, persists state) and `kaggle/h34j/summarize.py`. The orchestrator launches the dispatcher.
