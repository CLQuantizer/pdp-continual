# Handoff H25c: per-channel gradient-momentum progress (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, results/h25b/RESULTS.md (with Verdict) and `h25b_progress_fix.py`. Create only `h25c_gmc_channel.py` and
`results/h25c/*` (import from h25a/h25b scripts; do not edit them). Do not commit; do not touch docs/loop.md or audit.md.
CPU torch in `.venv`, `torch.set_num_threads(2)`.

Everything (world, learner with batched replay, bandits, W score, T = 512, seeds 6 7 8) exactly as H25a/H25b except the
reward. Reuse the rand / rand7 / err / gmc_prop numbers from H25a/H25b; do not rerun them.

## Reward `gmcc` (per-channel signed coupling)
- For each channel c keep its own momentum vector m_c over all parameters (same shape as the parameters), initialised to
  zero, updated only on glances of c: m_c ← 0.9·m_c + 0.1·g, where g is the gradient of the glance's own loss (before
  replay; the same gradient the glance's own Adam step uses).
- Reward for the glance, computed **before** updating m_c: r = (g · m_c) / (‖g‖·‖m_c‖ + 1e-12) × (glance bits per symbol
  before learning), a signed cosine scaled by how much is left to learn. For a channel's first glance, r = 0.
  Learnable structure gives repeatable gradient directions (cosine > 0); noise gives random directions (cosine ≈ 0); a
  learnt deterministic channel has a tiny gradient but a cosine that may be high, hence the scale by remaining bits.
- Conditions: `gmcc` (ε-greedy) and `gmcc_prop` (proportional to max(v, 0) + 0.01).
- Memory: 9 extra parameter-sized vectors; print the parameter count and confirm it fits.

## Report: results/h25c/RESULTS.md (numbers only) + results.json
Same tables as H25b with the H25a/H25b rows for rand / rand7 / err / gmc_prop alongside; per-quarter shares; wall time.
Smoke test T = 32, one seed. Budget 90 min wall.

## Pre-registered bars (orchestrator scores; do not change)
1. The better of gmcc / gmcc_prop (chosen on the mean W) spends ≤ 15% of glances on N and ≤ 15% on Z.
2. **M2 PASS:** that condition has W ≤ W(rand) − 0.05 on all 3 seeds and W < W(err).
3. Stretch: W ≤ W(rand7) on the mean.
4. Report only: N share of gmcc vs H25b's gmc (73.5%): does the per-channel signed coupling remove the noisy-TV pull?
