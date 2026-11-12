# H32b: the cerebellar mixer at 10 MB, with a calibration control (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `results/h32/RESULTS.md` (including its Verdict), `h32_mixer.py`,
`kaggle/h32/`, `results/h29d/RESULTS.md` and `h26_cued.py`.

## Context
H32 put count models of orders 1–4 beside the GRU cortex, with an online geometric mixer. On E3M the mix scored
1.846 bits/byte, against 1.958 for the cortex alone and 1.924 for PPMd order 16. Both bars passed. Two questions
remain:
1. **Does the gain hold at 10 MB?** On E10M the cortex alone ties PPMd (1.836 vs 1.837, H29d).
2. **Is the gain from the counts, or from calibration?** The mixer's rows are not normalised, so it can learn a
   temperature on the cortex per weight set. A mixer that sees only the cortex measures how much of the gain is
   calibration alone.

H32b also asks whether higher orders (5 and 6) add more. Principle served: improve the compressor (2).

## Kaggle GPU quota (checked 12 Nov 2026, 13:16 UTC+8)
15.47 of 30 GPU h remain (refresh Sat 14 Nov 08:00 UTC+8). Keep 10 h in reserve. One H29d E10M kernel took 1.07 GPU h;
H32 on E3M took 0.37 GPU h. **Budget for this job: at most 2.2 GPU h.** Run `kaggle quota` before pushing and stop if
fewer than 12.5 h remain.

## What to build
Create `h32b_mixer.py`. Import from `h32_mixer.py` and `h26_cued.py`, and do not edit them. Data: **E10M**, the first
10,000,000 bytes of enwik8, sha256 `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.

**Cortex:** exactly H32's cortex: H26 arm `uniform`, lr 3e-4, seed 0, replay seed as in H32, hidden 1024, chunks of
128, 8 replay windows per step. The mixers' gradients never reach it. One cortex pass feeds every arm below.

**Count models:** exactly as in H32 (p_k(b) = (n_k(ctx, b) + 0.05) / (N_k(ctx) + 12.8), exact counts, prequential,
empty context when fewer than k previous bytes), now for orders k ∈ {1, …, 6}. One count store serves every arm.
Each mixer reads the counts from before the chunk, and the counts of a chunk are added after all mixers have scored it
and taken their Adam step.

**Mixers.** Each one is H32's geometric mixer: score the chunk with the current weights, then one Adam step (lr 3e-3)
on the mean −log2 p_mix of the chunk. Each is initialised so that it starts as the cortex alone.
| Arm | Inputs x | Weight set s | Weights |
|---|---|---|---|
| `mix4` | [log p_gru, log p_1 … log p_4] | highest k ≤ 4 with N_k(ctx) ≥ 2, else 0 | 5 × 5, rows [1, 0, 0, 0, 0] (= H32) |
| `temp4` | [log p_gru] only | the same s as `mix4` | 5 × 1, all 1 (a learned temperature per set) |
| `mix6` | [log p_gru, log p_1 … log p_6] | highest k ≤ 6 with N_k(ctx) ≥ 2, else 0 | 7 × 7, rows [1, 0, …, 0] |

Also report `gru` (the cortex alone), `count_k` for k = 1 … 6 alone, and **PPMd order 16 on E10M** with H29's PPMd
code (H29d gave 1.837254; recompute and report both).

Memory: Kaggle gives 32 GB. If the order-5 and order-6 dicts would exceed about 24 GB, stop and report the size at the
point you stopped. Do not switch to hashing.

## Smoke test (VPS CPU, before any kernel)
1. On the first 8,192 bytes on CPU, `gru` and `mix4` equal H32's `gru` and `mix` on the same bytes to 1e-6 (run H32's
   code for the reference).
2. With weights frozen at their initial values, `temp4` and `mix6` equal `gru` to 1e-6.
3. The count check from H32 for order 2, plus the same hand check for order 5 on a string you choose; print both.
4. All arms are finite.
5. Time 65,536 bytes on CPU, and from it estimate the kernel's GPU hours. If the estimate exceeds 2.2 h, stop and
   report.
If any check fails, stop and report. Do not push the kernel.

## Kernel
One kernel, `ezzzzzx/h32b-mixer`: private, GPU, no internet. The code goes up as private dataset
`ezzzzzx/pdp-code-h32b`. Data: `ezzzzzx/enwik8`.

## Outputs
`results/h32b/RESULTS.md` and `results/h32b/results.json`, with numbers only and no verdict:
- E10M bits/byte for `gru`, `count_1` … `count_6`, `mix4`, `temp4`, `mix6` and PPMd order 16, plus each one's ten 1 MB
  slices;
- the gaps mix4 − PPMd, mix4 − temp4, temp4 − gru, mix6 − mix4 and mix4 − gru;
- the final weights of each mixer, and the share of bytes in each weight set (for both s definitions);
- `gru` against H29d's r8 seed-0 run (if H29d has none for seed 0, the r8 mean), as a check, not a gate;
- peak resident memory, wall seconds, kernel id, version and GPU hours;
- the smoke-test numbers.

## Pre-registered bars (orchestrator scores; do not change)
For scale: seed-to-seed std of the cortex on E10M is 0.004 (H29d).
1. **The gain holds at 10 MB:** mix4 − PPMd(order 16, E10M) ≤ −0.05 bits/byte.
2. **The counts carry it, not calibration:** mix4 − temp4 ≤ −0.05 bits/byte.
Report only: mix6 − mix4 and temp4 − gru.

## Budget and rules
- At most 2.2 GPU h. A failed kernel may be rerun once, if the quota check still passes.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `h32b_mixer.py`, `kaggle/h32b/**`, `results/h32b/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h32_mixer.py`, `h26_cued.py`, `h31_sleep.py`, any `h29*` file, other
  `kaggle/` folders, other `results/` folders, `.env`.
- Never print or upload secrets. Do not commit.
