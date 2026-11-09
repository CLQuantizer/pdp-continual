# Handoff H25a: curiosity with a real neural compressor (worker: codex, --sandbox workspace-write)

**Orchestrator gate: dispatch only after the H21a verdict, and only if H21a passed or was "Partial".** The
orchestrator fills in LEARNER below before dispatch.

LEARNER = ________ (the best learner in results/h21a/RESULTS.md, with its exact hyperparameters)

Read CLAUDE.md, docs/roadmap.md (M2), docs/h23_curiosity.md, docs/h23b_curiosity.md (the H23c section), results/h23c/RESULTS.md,
and `curio.py` + `h21a_substrate.py`. Create `h25a_curio_neural.py` and `results/h25a/*` only; do not edit existing
files; do not commit; do not touch docs/loop.md or audit.md. CPU torch is installed in `.venv`; `torch.set_num_threads(2)`.

## World and controllers (identical to H23c unless stated)
- Channels: `curio.CH` (7 H20 processes + N noise + Z dark), glance = 64 symbols, `curio.glance()` for the symbols.
- Bandit: one optimistic pass over all 9 channels, then ε-greedy (ε = 0.1) on an EMA (β = 0.3) of reward.
- Conditions: `rand`, `rand7`, `err` (bits of the glance before learning, per symbol), `prog` (whole-history progress,
  below), and `prog_prop`: same reward as prog but the channel is sampled with probability ∝ max(v_c, 0) + 0.01 after
  the first pass (no ε).

## Compressor
- LEARNER, one shared network for all channels, with a learned channel embedding (8 dims) concatenated to every
  input symbol embedding. Keep a separate recurrent state per channel: a glance on channel c starts from c's last state
  and saves it back.
- Learning: after scoring a glance prequentially (all 64 symbols with the current weights, from c's state), take ONE
  Adam step on that glance (truncated BPTT over its 64 symbols), as in H21a. No other training: no replay, no extra
  steps.

## Whole-history progress (prog, prog_prop)
- Per channel, a reservoir of 128 samples drawn uniformly over every position ever observed on that channel. Each
  sample = (the 32 symbols before the position on that channel, the target symbol). Scored from a zero state with the
  32-symbol context.
- Reward = (bits of the reservoir under the weights before this glance's Adam step − bits after) / 128. The glance's
  own positions are added to the reservoir only after the reward is computed.

## Budget and seeds
- Headroom pilot first, seeds 0 1 2, `rand` and `rand7` at T ∈ {512, 1024} glances. Pick the smallest T with
  W(rand) − W(rand7) ≥ 0.1. If neither qualifies, stop and report the pilot table: that is a result.
- Scoring: seeds 6–11 at the chosen T, all 5 conditions.
- Score W exactly as curio.py: the final frozen compressor on a fresh 2048-symbol continuation of each of the 7
  learnable channels from a deep copy of the world (continuing from each channel's saved state), mean bits/symbol.
- Report per condition: W mean ± std, per-channel held-out bits, glance share per channel overall and per quarter of
  life, wall time. Smoke test: T = 32, one seed. Budget: 90 min wall time; if over, use seeds 6–8 only and say so.

## Report: results/h25a/RESULTS.md (numbers only, no verdict) + results.json

## Pre-registered bars (orchestrator scores; do not change)
1. err ≥ 60% of glances on N.
2. The better of prog / prog_prop spends ≤ 15% on N and ≤ 15% on Z.
3. **M2 PASS:** the better of prog / prog_prop (chosen on the mean) has W ≤ W(rand) − 0.05 in ≥ 5 of 6 seeds and
   W < W(err).
4. Stretch: W ≤ W(rand7) on the mean.
5. Report only (no bar): share of glances on C1, C7, E30, E110 in the last quarter vs the first. If the learner can now
   learn counters or CA rows, curiosity should visit them while it is learning them.
