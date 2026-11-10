# H29c: replay at the stable learning rate on enwik8 E10M (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `results/h29b/RESULTS.md` (including its Verdict), `h29b_enwik8.py`, `kaggle/h29b/`.

## Context
H29b: the byte GRU cortex with no replay at lr 3e-4 scores 1.973638 bits/byte on E10M (the first 10,000,000 bytes of
enwik8). That beats xz -9e (2.176205) and trails PPMd order 16 (1.837254). At lr 1e-3 the model diverges, and replay
made it worse, because each replay step is another step at an unstable learning rate. H29c asks whether
store-everything replay (principle 1) helps the compressor (principle 2) once the learner is stable.

## What to run
Reuse `h29b_enwik8.py` unchanged; it already has `--lr`, `--replay-batch`, `--arm`, `--n-bytes` and `--out`. If it
truly needs a change, copy it to `h29c_enwik8.py` instead of editing it. Each arm runs on the full E10M, with one
private GPU kernel per arm. Kaggle allows 2 GPU sessions at once, so queue the third.

| Arm | lr | replay batch | traffic |
|---|---:|---:|---:|
| `r1_3e4` | 3e-4 | 1 | 1 |
| `r8_3e4` | 3e-4 | 8 | 8 |
| `nr_1e4` | 1e-4 | 0 | 0 |

The baseline is H29b's `nr_3e4` (1.973638). Do not rerun it.

Kernels: `ezzzzzx/h29c-r1-3e4`, `ezzzzzx/h29c-r8-3e4`, `ezzzzzx/h29c-nr-1e4`, all with `"is_private": true`,
`"enable_gpu": true` and `"enable_internet": false`. Code goes up as private dataset `ezzzzzx/pdp-code-h29c`, and the
data comes from `ezzzzzx/enwik8`. Check the E10M sha256 in each kernel against
`5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.

## Smoke test (VPS CPU, before any kernel)
`h29b_enwik8.py --lr 3e-4 --replay-batch 0` on the first 4,096 bytes must reproduce H29b's `nr_3e4` smoke value
(23836.498041 bits) to 1e-6. Each new arm on 4,096 bytes must give finite bits. If either check fails, stop.

## Outputs
`results/h29c/RESULTS.md` and `results/h29c/results.json`, with numbers only and no verdict. Per arm: E10M bits/byte,
the ten 1 MB slice bits/byte, replay steps, traffic, wall seconds, GPU type, and the differences from nr_3e4
(1.973638), xz -9e (2.176205) and PPMd order 16 (1.837254). Also list the kernel ids, elapsed seconds and total GPU
hours.

## Pre-registered bars (orchestrator scores; do not change)
1. **Replay helps when stable:** the better replay arm's E10M ≤ nr_3e4 − 0.02, that is ≤ 1.953638.
2. **Beats PPMd:** the best H29c arm's E10M ≤ 1.837254.
3. Report only: nr_1e4 − nr_3e4, and the last-slice value of each arm.
Kill rule: if both replay arms are worse than nr_3e4, replay on real text at this dose structure is closed for now.

## Budget and rules
- At most 4 GPU hours in total. You may rerun a failed kernel once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `kaggle/h29c/**`, `results/h29c/**`, and `h29c_enwik8.py` only if it is needed.
- Do not touch: `docs/loop.md`, `audit.md`, `h29_enwik8.py`, `h29b_enwik8.py`, `kaggle/h29/`, `kaggle/h29b/`, `.env`.
- Never print or upload secrets. Do not commit.
