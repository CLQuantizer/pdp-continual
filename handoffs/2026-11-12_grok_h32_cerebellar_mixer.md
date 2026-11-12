# H32: the cerebellar mixer. A fast count model beside the slow cortex (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `docs/brain_map.md` (the Cerebellum row), `results/h26/RESULTS.md`,
`h26_cued.py` (the `uniform` arm is the cortex to reuse), `kaggle/h26/`, and `h29_enwik8.py` (the PPMd baseline
code).

## Context
Our GRU cortex with uniform replay ties PPMd order 16 on enwik8 E10M (1.836 vs 1.837). The best compressors (cmix,
nncp) win by *mixing* many predictors: fast count models that learn a context in one shot, and slow networks that
generalise. Brains have the same split. The cerebellum is a fast, error-trained predictor with a huge sparse
expansion, sitting beside the slower cortex. H32 adds count models of orders 1–4 beside the cortex and a small mixer
learned online. Principle served: improve the compressor (2).

## Kaggle GPU quota (checked 12 Nov 2026)
About 16.2 of 30 GPU h remain (refresh Sat 14 Nov 08:00 UTC+8). Keep 10 h in reserve. **Budget for this job: at most
1.0 GPU h.** Run `kaggle quota` before pushing and stop if fewer than 11 h remain.

## What to build
Create `h32_mixer.py`. Import the cortex from `h26_cued.py` with arm `uniform` (interleaved, uniform replay, 8 windows
per step, lr 3e-4, seed 0, hidden 1024), and do not edit it. Data: E3M, sha256
`72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac`.

**The cortex runs exactly as in H26 `uniform`.** Its own training, including the replay, uses only its own loss. The
mixer's gradient never reaches it. The cortex's bits therefore reproduce H26 `uniform` (1.958153), and that is a
check.

**Count models (the fast learner).** For each order k ∈ {1, 2, 3, 4}, the context is the previous k bytes. At
positions with fewer than k previous bytes, use the empty context.
- n_k(ctx, b) counts how often byte b followed ctx. N_k(ctx) = Σ_b n_k(ctx, b).
- The prediction is p_k(b) = (n_k(ctx, b) + 0.05) / (N_k(ctx) + 0.05 · 256).
- **Prequential:** a byte is predicted from counts of earlier bytes only. Add its count after it has been scored.
- Any exact sparse storage is fine, such as a dict per order keyed by the context bytes. Counts must be exact, with
  no hashing collisions, and run on the kernel's CPU.

**Mixer (geometric, learned online).** For each byte, the input vector is
x = [log p_gru, log p_1, log p_2, log p_3, log p_4], with each entry a 256-vector of log-probabilities. The mixed
prediction is p_mix = softmax(Σ_j w_{s,j} · x_j).
- The weight set s ∈ {0..4} is the highest order k whose context has N_k(ctx) ≥ 2, or 0 if there is none.
- W is 5 × 5 and initialised to [1, 0, 0, 0, 0] in every row, so the mixer starts as the cortex alone.
- Per 128-byte chunk: score all the chunk's bytes with the current W. Those bits are the score. Then take one Adam
  step on W, with lr 3e-3, using the mean −log2 p_mix over the chunk. Then add the chunk's counts.
- Use the cortex's log-probabilities as computed for scoring, detached.

**Baseline.** PPMd order 16 on E3M, computed with H29's PPMd code. Report its bits/byte.

## Arms (one kernel: `ezzzzzx/h32-mixer`; private, GPU, no internet; code dataset `ezzzzzx/pdp-code-h32`)
From the single run, report the bits/byte of:
- `gru`: the cortex alone;
- `count_k` for each k ∈ {1..4}: each order alone;
- `mix`: the mixer.

## Smoke test (VPS CPU, before any kernel)
1. On the first 8,192 bytes on CPU, `gru` bits equal H26 `uniform` on the same 8,192 bytes to 1e-6.
2. With W frozen at its initial value, `mix` bits equal `gru` bits to 1e-6.
3. **Count check:** on the 20-byte string "abracadabraabracadab", hand-check p_2 for the last byte, and print both
   values.
4. All arms are finite.
If any check fails, stop and report. Do not push the kernel.

## Outputs
`results/h32/RESULTS.md` and `results/h32/results.json`, with numbers only and no verdict:
- E3M bits/byte for `gru`, `count_1` … `count_4`, `mix` and PPMd order 16, plus each one's three 1 MB slices;
- the gaps mix − gru and mix − PPMd;
- the final W, and the share of bytes in each weight set s;
- wall seconds, kernel id, version and GPU hours;
- the smoke-test numbers.

## Pre-registered bars (orchestrator scores; do not change)
1. **The fast learner helps the cortex:** mix − gru ≤ −0.05 bits/byte on E3M.
2. **Beats PPMd clearly:** mix − PPMd(order 16, E3M) ≤ −0.05 bits/byte.
Report only: the best single count order.

## Budget and rules
- At most 1.0 GPU h. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `h32_mixer.py`, `kaggle/h32/**`, `results/h32/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h26_cued.py`, `h31_sleep.py`, any `h29*` file, other `kaggle/` folders,
  other `results/` folders, `.env`.
- Never print or upload secrets. Do not commit.
