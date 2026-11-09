# H23: curiosity as compression progress, on an active world (pre-registration)

Pre-registered 9 Nov 2026, UTC+8, after a budget pilot on seeds 0–2 and before any scoring seed ran. Guideline:
`docs/curiosity.md` (Schmidhuber 2009, arXiv 0812.4360, in `docs/papers/`). Code: `curio.py`.

## Why

Everything up to H20 is passive: the world decides what the machine sees. Schmidhuber's principle has four parts
(store everything, improve the compressor, reward compression progress, act to maximise it). The fourth needs
actions. H23 is the smallest world where the machine chooses what to look at, and where the classic failure of
prediction-error curiosity (the "noisy TV") is on offer.

## World

Nine channels. Each action is a glance: the next B = 64 symbols of one channel.
- The 7 H20 processes (`G`, `C1`, `C7`, `E30`, `E110`, `M7`, `M37`). They resume where they left off.
- `N`: uniform noise over the 41 symbols. Always 5.36 bits, never learnable: the noisy TV.
- `Z`: dark, a constant symbol. Learnable in a few symbols, then boring.

## Machine

The compressor is fixed across conditions, so only the controller differs. It is a fixed-share mixture of KT orders 0–6,
with contexts keyed by (channel, last k symbols on that channel). The channel is the agent's own action, so conditioning
on it is allowed. This compressor cannot learn counters or look one CA row back. That limit is part of the test:
progress-driven curiosity should treat a source as uninteresting once *this* compressor stops improving on it.

All controllers use the same bandit: one optimistic pass over all channels, then ε-greedy (ε = 0.1) on an EMA
(β = 0.3) of their reward.

| Cond | Reward / rule |
|---|---|
| `rand` | uniform over 9 channels |
| `rand7` | uniform over the 7 learnable channels (oracle control: knows N and Z are worthless) |
| `err` | bits of the glance before learning it (Schmidhuber 1990: prediction error) |
| `prog` | compression progress on stored history: C(p_old, h) − C(p_new, h) per symbol, where h = the last 256 symbols seen on that channel *before* this glance (2009 eq. 5, f(a, b) = a − b; the glance itself is excluded, so memorising it earns nothing) |
| `cnt` | count bonus 1/√n_c (novelty baseline) |

## Score

**W** is the headline: mean bits/symbol of the *final, frozen* compressor on a fresh 2 048-symbol continuation of each of
the 7 learnable channels. The continuation is taken from a copy of the world at the end of life. Lower is better. It
measures how much of the world the machine has compressed with a fixed observation budget. Also reported: time share
per channel (overall and by quarter of life), and lifetime prequential bits.

## Budget (from the pilot, seeds 0–2, `rand` only)

Rule declared before the pilot: pick the smallest T in {512, 1024, 2048} such that W_rand(T) − W_rand(4T) ≥ 0.1 (the
budget still binds). Pilot W_rand: 512 → 1.642, 1024 → 1.569, 2048 → 1.540, 4096 → 1.526, 8192 → 1.486.
512 − 2048 = 0.102 → **T = 512 glances** (32 768 symbols). Scoring seeds: **6–11**.

## Predictions and bars

1. **Noisy-TV trap (sanity):** `err` spends ≥ 60% of glances on N.
2. **Prog avoids it:** `prog` spends ≤ 15% of glances on N and ≤ 15% on Z.
3. **PASS (mechanism):** W(prog) ≤ W(rand) − 0.05 in ≥ 5 of 6 seeds, and W(prog) < W(err) and W(prog) < W(cnt) on the mean.
4. **Stretch (allocation, not just avoidance):** W(prog) ≤ W(rand7) on the mean. Passing this means progress beats an
   oracle that knows which channels are noise. It allocates *among* learnable sources (focus on what is still
   improving), not only away from noise.

**Kill:** if bar 3 fails, compression progress buys nothing with this compressor in this world. The next step is then
a slower, parametric compressor (where allocation must matter), not tuning ε or β.

## What H23 does not test

- A learned controller with state (the bandit is the degenerate RL case with no world state). That is H25.
- The store/cortex split: here the count tables are both store and compressor. H24 (see
  `docs/hippocampus_diagnosis.md`) puts a parametric cortex beside an explicit store.
