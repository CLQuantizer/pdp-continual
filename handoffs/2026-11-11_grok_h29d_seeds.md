# H29d: seed spread on enwik8 E10M (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `results/h29c/RESULTS.md` and `results/h29b/RESULTS.md` (each with its
Verdict), `h29b_enwik8.py`, `kaggle/h29c/`.

## Context
At lr 3e-4 on E10M (the first 10,000,000 bytes of enwik8), H29c's replay at traffic 8 (`r8_3e4`) scored 1.831624
bits/byte. H29b's no-replay arm (`nr_3e4`) scored 1.973638, and PPMd order 16 scores 1.837254. Each of those numbers
comes from one seed (SEED = 0). The PPMd margin (−0.006) and the replay gain (−0.142) both need a seed spread before
they count. Principles served: store everything (1) and improve the compressor (2).

## What to build
Copy `h29b_enwik8.py` to `h29d_enwik8.py` and add a single flag, `--seed <int>`. It sets `SEED`, so the torch init
seed is `SEED` and the replay sampler seed is `REPLAY_SEED + SEED`, exactly as the code already derives them. Nothing
else changes. Do not edit `h29b_enwik8.py`.

## Runs: 4 private GPU kernels on the full E10M
| Kernel | arm | lr | replay batch | seed |
|---|---|---:|---:|---:|
| `ezzzzzx/h29d-r8-s1` | r8_3e4 | 3e-4 | 8 | 1 |
| `ezzzzzx/h29d-r8-s2` | r8_3e4 | 3e-4 | 8 | 2 |
| `ezzzzzx/h29d-nr-s1` | nr_3e4 | 3e-4 | 0 | 1 |
| `ezzzzzx/h29d-nr-s2` | nr_3e4 | 3e-4 | 0 | 2 |

Every kernel has `"is_private": true`, `"enable_gpu": true` and `"enable_internet": false`. Code goes up as private
dataset `ezzzzzx/pdp-code-h29d`; the data source is `ezzzzzx/enwik8`. Check the E10M sha256 against
`5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`. Kaggle allows 2 GPU sessions at once: start both
r8 kernels first and queue the nr kernels as slots free up.

## Smoke test (VPS CPU, before any kernel)
1. `h29d_enwik8.py --seed 0 --lr 3e-4 --replay-batch 0` on the first 4,096 bytes must reproduce 23836.498041 bits to
   1e-6 (H29b/H29c smoke).
2. `--seed 1` with replay batch 8 on 4,096 bytes must give finite bits that differ from seed 0.
If either fails, stop and report.

## Outputs
`results/h29d/RESULTS.md` and `results/h29d/results.json`, with numbers only and no verdict:
- per kernel: E10M bits/byte, the ten 1 MB slices, replay steps, traffic, wall seconds, GPU;
- a table over seeds {0, 1, 2} per arm, where seed 0 is taken from H29c `r8_3e4` and H29b `nr_3e4`: mean and
  population std (ddof = 0);
- per seed, the paired difference r8 − nr, with its mean ± std;
- per arm, mean − PPMd order 16 (1.837254) and mean − xz -9e (2.176205);
- kernel ids, elapsed seconds, total GPU hours.

## Pre-registered bars (orchestrator scores; do not change)
1. **Replay gain holds:** r8 − nr < 0 on all 3 seeds, and the mean is ≤ −0.05.
2. **PPMd:** report the r8 mean ± std against 1.837254. It counts as "beats PPMd" only if mean + std < 1.837254, and as
   a tie if |mean − 1.837254| ≤ std.

## Budget and rules
- At most 3.5 GPU hours in total. A failed kernel may be rerun once.
- Run in the foreground and poll with `kaggle kernels status`.
- Files you may create: `h29d_enwik8.py`, `kaggle/h29d/**`, `results/h29d/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h29*_enwik8.py` other than your new file, `kaggle/h29*/` other than
  `kaggle/h29d/`, `.env`.
- Never print or upload secrets. Do not commit.
