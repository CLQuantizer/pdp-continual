# Handoff H25b: fixing the progress signal for a neural learner (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, results/h25a/RESULTS.md (with Verdict), docs/literature_2026-11.md §1, and `h25a_curio_neural.py`.
Create only `h25b_progress_fix.py` and `results/h25b/*` (import from h25a_curio_neural.py; do not edit it). Do not
commit; do not touch docs/loop.md or audit.md. CPU torch in `.venv`, `torch.set_num_threads(2)`.

Everything (world, learner with batched replay, bandit, W score, T = 512, seeds 6 7 8) exactly as H25a except the reward.
Reuse H25a's rand / rand7 / err numbers; do not rerun them.

## Conditions (each with ε-greedy as in H25a, and a `_prop` variant with proportional allocation as in H25a)
1. `progc` / `progc_prop`: whole-history reservoir progress, but (a) the "after" score is taken **after the glance's own
   Adam step and before the 2 replay steps**, and (b) every probability used in the reservoir score is clipped to
   [2⁻⁸/40, 1 − 2⁻⁸] and renormalised (a finite-precision coder), so a perfectly learnt symbol costs ≥ ~0.0056 bits and
   cannot keep "improving".
2. `gmc` / `gmc_prop`: gradient-momentum coupling (Blad et al. 2026): on the glance's own Adam step, reward =
   (1/√d) Σ_i |g_i · m̂_i / s_i| with s_i = max(√v̂_i, α·|θ_i|) using Adam's own m̂, v̂ (bias-corrected), α = 1e-3, summed
   over all parameters (d = parameter count), computed from the gradient of the glance's own loss, before replay.
   Document the exact formula you implemented.

## Report: results/h25b/RESULTS.md (numbers only) + results.json
Same tables as H25a (W per seed, per-channel held-out bits, glance shares overall and per quarter), with the H25a rows
for rand / rand7 / err / prog / prog_prop alongside. Wall time. Smoke test T = 32, one seed. Budget 120 min wall: if over,
drop the `_prop` variants first and say so.

## Pre-registered bars (orchestrator scores; do not change)
1. The best of the four new conditions (chosen on the mean W) spends ≤ 15% of glances on N and ≤ 15% on Z.
2. **M2 PASS:** that condition has W ≤ W(rand) − 0.05 on all 3 seeds, and W < W(err).
3. Stretch: W ≤ W(rand7) on the mean.
4. Report only: Z share for progc vs H25a's prog (does clipping kill the dark-room trap?).
