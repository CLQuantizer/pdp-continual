# Handoff H25d: visit-to-visit learning progress (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, results/h25b/RESULTS.md and results/h25c/RESULTS.md (Verdict sections), and `h25b_progress_fix.py`.
Create only `h25d_visit_slope.py` and `results/h25d/*` (import from earlier scripts; do not edit them). Do not commit;
do not touch docs/loop.md or audit.md. CPU torch in `.venv`, `torch.set_num_threads(2)`.

Everything (world, learner with batched replay, bandits, W score, T = 512, seeds 6 7 8) exactly as H25a/H25b except the
reward. Reuse rand / rand7 / err / gmc_prop numbers from H25a/H25b; do not rerun them.

## Reward `slope`
- Per channel, the same 128-sample whole-history reservoir as H25b's progc, scored with probabilities clipped to
  [2⁻⁸/40, 1 − 2⁻⁸] and renormalised.
- Each time channel c is chosen, **before** the glance is learnt, score c's reservoir: b_c(k) for its k-th visit. Keep
  c's last 4 scores.
- Reward credited to c at visit k (k ≥ 3): r = −(least-squares slope of b_c over its last up-to-4 visits, in bits per
  visit). It is positive when c's whole-history bits have been falling across visits, ~0 when flat (noise, or a learnt
  constant at the clipping floor), and negative when c is being forgotten. For k < 3, r = the optimistic value 1.0.
- The bandit value v_c is this r (no extra EMA). Conditions: `slope` (ε-greedy, ε = 0.1) and `slope_prop`
  (∝ max(v, 0) + 0.01).
- Note: this reward reflects all updates between c's visits, including other channels' learning and replay.

## Report: results/h25d/RESULTS.md (numbers only) + results.json
Same tables as H25b with rand / rand7 / err / gmc_prop alongside; per-quarter shares; wall time. Smoke test T = 32,
one seed. Budget 90 min wall.

## Pre-registered bars (orchestrator scores; do not change)
1. The better of slope / slope_prop (chosen on the mean W) spends ≤ 15% of glances on N and ≤ 15% on Z.
2. **M2 PASS:** that condition has W ≤ W(rand) − 0.05 on all 3 seeds and W < W(err).
3. Stretch: W ≤ W(rand7) on the mean.
