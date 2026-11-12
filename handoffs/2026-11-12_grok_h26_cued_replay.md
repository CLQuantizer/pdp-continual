# H26: replay keyed by content, not position (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `docs/compute-vs-data-movement.md` (the section "Addresses: the brain is
content-addressed"), `docs/brain_map.md`, `results/h31/RESULTS.md` (including its Verdict), `h31_sleep.py` and
`kaggle/h31/`.

## Context
Our replay picks a uniform random *position* in the log. That is RAM: a brain has no positions to sample. Brains
replay by content. A cue, the current state, retrieves similar past episodes: the hippocampal index, dentate-gyrus
hashing, CA3 completion. Spontaneous replay roams the store more broadly. Three times, selective replay hurt us (H10,
M4a, H27). H31 showed that replay must stay interleaved and close to the input.

H26 asks whether content-keyed replay, alone or mixed with broad replay, can match uniform-position replay at the same
dose. Principles served: store everything (1) and improve the compressor (2).

## Kaggle GPU quota (checked 12 Nov 2026)
17.19 of 30 GPU h remain (refresh Sat 14 Nov 08:00 UTC+8). Keep 10 h in reserve. **Budget for this job: at most
1.5 GPU h.** Run `kaggle quota` before pushing and stop if fewer than 11.5 h remain.

## What to build
Create `h26_cued.py`, a copy of `h31_sleep.py` with `--sleep-block 1` fixed (interleaved). The data is E3M, the first
3,000,000 bytes of enwik8; check the sha256 against `72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac`.
Model, lr 3e-4, seed 0 and the score-before-step order are all unchanged. Each replay step is still one step on
8 windows (traffic 8). Only *which* windows are chosen changes.

- **Content keys (the index).** When live chunk c (bytes 128c … 128c + 127) is scored, record a key for it:
  k_c = P · h_c. Here h_c is the cortex's live hidden state at the *start* of chunk c, detached, and P is a fixed
  random Gaussian projection 1024 → 128 with entries N(0, 1/128), drawn from `torch.Generator().manual_seed(2626)`.
  Store k_c, L2-normalised, in a GPU tensor. Keys are written once and never refreshed, like an index laid down at
  encoding.
- **Cue.** At replay time, the cue q is P · h, where h is the live hidden state right after the current chunk,
  L2-normalised.
- **Cued window.** Score every eligible stored chunk by cosine similarity q · k_c. Eligible means c ≤ current chunk −
  16, which excludes the last 2 KB so that recency is not what gets retrieved. Take the top 32 chunks and sample one
  of them uniformly. The window starts at that chunk's first byte, with the same window length, zero initial state and
  loss as H31's replay windows. If fewer than 32 chunks are eligible, sample uniformly from the eligible ones, and if
  none are eligible, use the uniform sampler.
- **Uniform window:** H31's sampler, unchanged.

## Arms (each 3 MB, interleaved, the same total replay steps)
| Arm | windows per replay step |
|---|---|
| `uniform` | 8 uniform (= H31 `inter`; rerun, because GPU kernels are not bit-exact) |
| `cued8` | 8 cued |
| `mix44` | 4 cued + 4 uniform |

Kernels `ezzzzzx/h26-uniform`, `ezzzzzx/h26-cued8` and `ezzzzzx/h26-mix44`, all with `"is_private": true`,
`"enable_gpu": true` and `"enable_internet": false`. The code goes up as private dataset `ezzzzzx/pdp-code-h26`.

## Smoke test (VPS CPU, before any kernel)
1. **Equivalence:** `h26_cued.py --arm uniform` and `h31_sleep.py --sleep-block 1` on the first 8,192 bytes on CPU
   give the same total bits to 1e-6.
2. **Retrieval check:** on the first 65,536 bytes on CPU (`cued8`), log for 5 replay steps the cue chunk index, the 8
   retrieved chunk indices and their cosine scores. Every retrieved index must be ≤ current − 16. Print 40 bytes of
   text at the cue position and at the top retrieved chunk, for one step.
3. All arms give finite bits on the first 8,192 bytes on CPU.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/h26/RESULTS.md` and `results/h26/results.json`, with numbers only and no verdict:
- per arm: E3M bits/byte, the three 1 MB slice bits/byte, replay steps, wall seconds and GPU type;
- the gaps cued8 − uniform and mix44 − uniform;
- for the cued arms: the mean age of retrieved chunks (current chunk − retrieved chunk, in chunks) against the mean
  age under uniform sampling at the same times, and the fraction of retrievals whose chunk lies in the most recent
  10% of the eligible log;
- a coverage measure: over the whole run, the fraction of stored chunks retrieved at least once (cued arms), against
  the fraction under uniform;
- kernel ids, versions, elapsed seconds and total GPU hours;
- the smoke-test output, including the retrieval excerpt.

## Pre-registered bars (orchestrator scores; do not change)
For scale, on E3M: noreplay − inter = 0.240 bits/byte (H31). Seed-to-seed std on E10M was 0.004 (H29d).
1. **Content-keyed plus broad replay matches position replay:** mix44 − uniform ≤ 0.01 bits/byte.
2. **Prediction (scored as right or wrong):** cued8 − uniform > 0.01. Pure cued replay is selective and should hurt,
   as selective replay did three times before.
Report only: the age and coverage numbers, which give the mechanism.

## Budget and rules
- At most 1.5 GPU h in total. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll kernels with `kaggle kernels status`.
- Files you may create: `h26_cued.py`, `kaggle/h26/**`, `results/h26/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h31_sleep.py`, any `h29*` file, `kaggle/h31/`, `results/h31/`, any other
  experiment code, `.env`.
- Never print or upload secrets. Do not commit.
