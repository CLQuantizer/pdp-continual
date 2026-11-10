# Handoff H28: does history replay cure the curiosity noise leak? (worker: grok, Kaggle)

Read /root/pdp/CLAUDE.md, results/m4c/RESULTS.md, results/h27b/RESULTS.md (Verdicts), results/h25a/RESULTS.md and
results/h25c/RESULTS.md (Verdicts), and `m4c_score.py`, `h27b_matched_dose.py`, `h25a_curio_neural.py`. Create only
`h28_progress_replay.py`, `kaggle/h28/*`, `results/h28/*`; import, do not edit existing scripts. Do not commit; no
docs/loop.md or audit.md.

## Why
Curiosity in the machine (gmc_prop) spends 17% of glances on noise. GMC variants are a closed line (H25b–d). The
whole-history estimator that worked with per-channel learners (H23c) failed with a shared GRU (H25a: dark room),
plausibly through forget–relearn cycles: other channels' updates erase Z, a Z glance relearns it, and that shows as
progress forever. History replay now rehearses every channel, which should break those cycles.

## Machine (every condition)
M4c `hist_rep` machine with the H27b best dose: GRU, per-channel state, one live Adam step per glance, then **one**
replay step (batch 8 × 64) from the whole history (every glance; traffic 8). No utility store, no match. T = 512,
**fresh seeds 27–32**.

## Controllers
- `rand`, `rand8` (oracle), `gmc_prop` (as M4c).
- `prog_prop`: H25a's whole-history progress: per channel a reservoir of 128 (32-symbol context, target) samples over
  every observed position; reward = bits of the chosen channel's reservoir before the glance's updates minus after
  its live step **and** replay step, / 128; positions added after the reward. Allocation as H25a prog_prop (∝ max(EMA,0)
  + 0.01 after one optimistic pass; EMA β = 0.3).
- `prog_cm_prop`: same, but reward = Δbits(chosen channel's reservoir) − mean over the other 9 channels' Δbits
  (common-mode change removed: what the update did to *this* channel beyond what it did to all).
Batch the reservoir evaluations (all channels in one forward pass).

## Report (numbers only): results/h28/RESULTS.md + results.json
W (8 learnable channels), per-channel bits, glance shares overall and by quarter (N and Z called out), reward traces'
per-channel means, wall. Smoke T = 32 on Kaggle first. Private CPU kernels, dataset `ezzzzzx/pdp-code-h28`. Budget 120 min.

## Pre-registered bars (orchestrator scores; do not change)
1. **Noise leak cured:** the better of prog_prop / prog_cm_prop (on the mean W) has mean N share ≤ 15% and Z ≤ 15%.
2. **Curiosity pays:** that controller has W ≤ W(rand) − 0.05 on ≥ 5 of 6 seeds.
**M4 PASS (machine = cortex + history store with replay + curiosity) = bars 1 and 2**, since W(rand) here is the
cortex + replay without curiosity and M4c already showed cortex + replay beats the cortex alone.
3. Report only: vs gmc_prop and vs rand8.
