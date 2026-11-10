# Handoff M4c: score M4 on fresh seeds, on Kaggle compute (worker: grok)

Read /root/pdp/CLAUDE.md (incl. "Kaggle compute"), handoffs/2026-11-10_grok_m4b_ablation.md, results/m4b/RESULTS.md
(Verdict) and `m4b_ablation.py`. Create only `m4c_score.py`, `kaggle/m4c/*` (kernel folders) and `results/m4c/*`;
import from existing scripts, do not edit them. Do not commit; do not touch docs/loop.md or audit.md.

## What
The same machine as M4b, scored on **fresh seeds 9 10 11 12 13 14** (never used before), T = 512, world as M4a/M4b.
Conditions (all with the M4b hard budget; "history replay" = M4b's whole-history source):
| name | controller | store + match | replay |
|---|---|---|---|
| `nostore` | rand | no | none |
| `hist_rep` | rand | no | history |
| `rand_full` | rand | yes | history (= M4b store_hist) |
| `rand8_full` | rand8 (oracle) | yes | history |
| `gmc_nostore` | gmc_prop | no | history |
| `gmc_full` | gmc_prop | yes | history (= M4b gmc_hist) |
`m4c_score.py` adds the two new combinations by reusing M4b's code paths; no other change. Score exactly as M4b
(W_mix, W_gru, W_F both, per-channel bits, glance shares overall and by quarter, peak/final M/N, wall).

## Where it runs: Kaggle, private, CPU only
- Upload the needed repo scripts as a **private** Kaggle dataset `ezzzzzx/pdp-code-m4c` (`kaggle datasets create`;
  scripts only, no `.env`, no data, no results, no secrets). Kernels: `"is_private": true`, `"enable_gpu": false`,
  `"enable_internet": false`, the dataset as `dataset_sources`; the kernel installs nothing (Kaggle has torch).
- Split the 36 runs over up to 3 kernels (e.g. 2 seeds each), each running 2 processes in parallel with
  `torch.set_num_threads(2)`. Each writes per-run JSON to its output; fetch all with `kaggle kernels output` into
  `results/m4c/raw/`, then assemble `results/m4c/RESULTS.md` (numbers only, no verdict) + `results.json` here.
- Smoke first: one kernel, T = 32, seed 9, all 6 conditions; check its numbers are sane, then launch the scored kernels.
- Determinism check: before Kaggle, run `nostore` seed 9 T = 32 locally and on Kaggle; report both W (CPU numerics may
  differ slightly; report the difference, do not fix it).
- Poll status every few minutes; budget 150 min wall in total. If a kernel fails, fix and rerun only that kernel, and say so.

## Pre-registered bars (orchestrator scores; do not change)
1. **Curiosity:** W_mix(gmc_full) ≤ W_mix(rand_full) − 0.05 on ≥ 5 of 6 seeds, with mean N share ≤ 15% and Z ≤ 15%.
2. **Whole machine beats its parts:** W_mix(gmc_full) ≤ W(hist_rep) − 0.05 and ≤ W(nostore) − 0.05 on the mean.
3. **Store in the machine (report + sign):** W_mix(gmc_full) vs W(gmc_nostore), and W_F of both.
4. Report only: gmc_full vs rand8_full.
**M4 PASS = bars 1 and 2.**

## Output
Final message: the means table and per-seed W_mix of every condition, N/Z shares, Kaggle kernel ids and run times,
the determinism check, anything rerun.
