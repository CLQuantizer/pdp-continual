# H29b: enwik8 E10M, replay dose vs stability (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `results/h29/RESULTS.md` (including its Verdict), `h29_enwik8.py`, `kaggle/h29/`.

## Context
H29 ran the H27b cortex (byte GRU, embedding 64, hidden 1024, Adam, clip 1.0, chunks of 128, TBPTT 128) prequentially on
the first 10,000,000 bytes of enwik8 ("E10M"). It scored 3.644 bits/byte, against xz -9e at 2.176 and PPMd order 16 at
1.837. The model got worse as it read: 2.26 on the first MB, then 3.5–4.2. Its replay was one batch of 32 windows per
128-byte chunk, which is traffic 32. Our toy-world knee is traffic 8 (H27b), so that dose was 4× too high. H29b
separates the two possible causes: replay dose, and plain instability of online Adam at lr 1e-3. Principle served:
improve the compressor (principle 2), on real data, with store-everything replay (principle 1).

## What to build
Create `h29b_enwik8.py`, a copy of `h29_enwik8.py` with these exposed as command-line flags: `--lr`, `--replay-batch`
(0 = no replay), `--arm <name>`, `--n-bytes`, `--out <dir>`. Everything else stays identical to H29: model, chunking,
the score-before-step order, clipping, replay window sampling (uniform over all bytes seen so far, zero hidden state),
seeds, and bits = live −log2 p only. Do not edit `h29_enwik8.py`.

Arms, all on the full E10M (10,000,000 bytes), one private GPU kernel each, run in parallel:

| Arm | lr | replay batch | traffic |
|---|---:|---:|---:|
| `nr_1e3` | 1e-3 | 0 | 0 |
| `r8_1e3` | 1e-3 | 8 | 8 |
| `nr_3e4` | 3e-4 | 0 | 0 |

Kernels: `ezzzzzx/h29b-nr-1e3`, `ezzzzzx/h29b-r8-1e3`, `ezzzzzx/h29b-nr-3e4`, all with `"is_private": true`,
`"enable_gpu": true` and `"enable_internet": false`. Code goes up as private dataset `ezzzzzx/pdp-code-h29b`, and the
data comes from the existing private dataset `ezzzzzx/enwik8`. Check the E10M sha256 inside each kernel against
`5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.

## Smoke test (VPS CPU, before any kernel)
1. Equivalence: run `h29_enwik8.py`'s model path and `h29b_enwik8.py --lr 1e-3 --replay-batch 32` on CPU over the
   first 4,096 bytes with the same seed. Total bits must match to 1e-6. Report both numbers.
2. Each arm on the first 4,096 bytes on CPU gives finite bits.
If either check fails, stop and report. Do not push kernels.

## Outputs
`results/h29b/RESULTS.md` and `results/h29b/results.json`, with numbers only and no verdict:
- per arm: E10M bits/byte, the ten 1 MB slice bits/byte, replay steps, traffic, wall seconds, GPU type;
- per arm: the difference from xz -9e (2.176205) and from PPMd order 16 (1.837254), using H29's measured baselines; do
  not rerun the baselines;
- kernel ids, versions, elapsed seconds, and total GPU hours.

## Pre-registered bars (orchestrator scores; do not change)
1. **Beats xz:** the best arm's E10M bits/byte ≤ 2.176205.
2. **Learns as it reads:** in the best arm, the last 1 MB slice is below its first 1 MB slice.
3. Report only: r8_1e3 − nr_1e3 on E10M (the replay effect at the knee dose), and nr_3e4 − nr_1e3.
This is a calibration on E10M with no held-out split; no further tuning on these numbers is scored against the same bar.

## Budget and rules
- At most 4.5 GPU hours in total (H29 took about 70 min per E10M pass on a T4). If a kernel fails, you may rerun it
  once. Otherwise no reruns.
- Run experiments in the foreground. Poll kernels with `kaggle kernels status`.
- Files you may create: `h29b_enwik8.py`, `kaggle/h29b/**`, `results/h29b/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h29_enwik8.py`, `kaggle/h29/`, any other experiment code, `.env`.
- Never print or upload secrets. Do not commit.
