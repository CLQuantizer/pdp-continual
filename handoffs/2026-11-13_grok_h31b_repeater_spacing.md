# H31b: replay as a repeater. Does the cost of spacing replay out grow with the forgetting rate? (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `docs/compute-vs-data-movement.md` (section "persistence is just movement
through time"), `results/h31/RESULTS.md` (including its Verdict), `h31_sleep.py` and `kaggle/h31/`.

## Context
Persistence is movement through time. The cortex's weights carry the past with loss, because each update wears old
content down, and replay is the repeater that re-sends it before it fades. H31 measured the spacing: the same replay
dose delivered in sleep blocks of 1,024 chunks cost 0.133 bits/byte more than interleaved replay on E3M (lr 3e-4).
A lossy-line model predicts that the cost of wide spacing grows with the loss per step. A higher learning rate
forgets faster, so it should pay more for the same spacing; a lower one should pay less. Principles served: store
everything (1) and improve the compressor (2).

## Kaggle GPU quota (checked 13 Nov 2026, 08:41 UTC+8)
14.19 of 30 GPU h remain (refresh Sat 14 Nov 08:00 UTC+8). Keep 10 h in reserve. One H31 arm took about 0.3 GPU h.
**Budget for this job: at most 1.6 GPU h.** Run `kaggle quota` before pushing and stop if fewer than 11.8 h remain.

## What to run
Use `h31_sleep.py` unchanged, with its flags. E3M, replay batch 8, seed 0, and every other setting as in H31.
| Arm | lr | sleep block B |
|---|---|---:|
| `inter_lo` | 1.5e-4 | 1 |
| `sleep1k_lo` | 1.5e-4 | 1,024 |
| `inter_hi` | 6e-4 | 1 |
| `sleep1k_hi` | 6e-4 | 1,024 |
The lr 3e-4 pair is H31's `inter` (1.958153) and `sleep_1k` (2.090683). Do not rerun it.

Kernels `ezzzzzx/h31b-<arm>` (use `-` for `_`), each with `"is_private": true`, `"enable_gpu": true` and
`"enable_internet": false`. Two arms per kernel is fine if that fits the quota math better. The code goes up as private
dataset `ezzzzzx/pdp-code-h31b`, a copy of H31's code.

## Smoke test (VPS CPU, before any kernel)
1. On the first 8,192 bytes on CPU, `h31_sleep.py --lr 3e-4 --sleep-block 1` reproduces H31's smoke bits to 1e-6.
2. All four arms give finite bits on the first 8,192 bytes on CPU, and the replay-step count is the same for
   `--sleep-block 1` and `--sleep-block 1024` (the final sleep covers the remainder, as in H31).
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/h31b/RESULTS.md` and `results/h31b/results.json`, with numbers only and no verdict:
- per arm: E3M bits/byte, the three 1 MB slices, replay steps, wall seconds and GPU type;
- the gap sleep1k − inter at each lr, next to H31's gap at lr 3e-4 (0.132530);
- the morning-after effect for the sleep arms, as in H31;
- if an arm diverges (nonfinite or exploding bits), report the step where it did, and do not change the lr;
- kernel ids, versions, elapsed seconds and total GPU hours;
- the smoke-test numbers.

## Pre-registered bars (orchestrator scores; do not change)
Let gap(lr) = sleep1k − inter at that lr. H31 gives gap(3e-4) = 0.133.
1. **The cost of spacing grows with the forgetting rate:** gap(1.5e-4) < gap(3e-4) < gap(6e-4).
2. **By a clear margin:** gap(6e-4) − gap(1.5e-4) ≥ 0.05 bits/byte.
If the lr 6e-4 pair diverges, the bars are void; report that.
Kill rule: if bar 1 fails, the cost of sleep blocks is not set by the cortex's forgetting rate, and the repeater
picture does not explain H31.

## Budget and rules
- At most 1.6 GPU h in total. A failed kernel may be rerun once, if the quota check still passes.
- Run experiments in the foreground. Poll kernels with `kaggle kernels status`.
- Files you may create: `kaggle/h31b/**` and `results/h31b/**`. Do not create new experiment code; use `h31_sleep.py`
  as it is.
- Do not touch: `docs/loop.md`, `audit.md`, `h31_sleep.py`, any `h29*` file, other `kaggle/` folders, other `results/`
  folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
