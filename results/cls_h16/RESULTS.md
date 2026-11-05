# H16: interleave the arriving item with replay at matched gradient steps

Pre-registered 08:55 UTC+8, 5 Nov 2026 (see `cls_sleep.py`), scored 09:05 UTC+8. Both worlds, seeds 6-8, forgetful store
(f2k), awake arrivals only, run on the box (`results/cls_h16_s49`, `results/cls_h16_bg002`). awu_f2k reused from H13;
determinism re-checked (s49 seed 6 rerun on the edited script: 0.4560, identical).

- awi_f2k: on each replay slot (every 38th wake arrival) the arrival is fitted inside the replay batch, one step on
  [x + 32 store prototypes], instead of alone and then a separate replay batch.
- awid_f2k: dense interleave. Every wake arrival is one step on [x] plus, with probability 32/38, one uniform store prototype.
- Both: 38 gradient steps per 38 arrivals (awu 39), 0.84 replayed items per arrival.

Bar. KEEP a variant only if, in BOTH worlds, ff_bits.all <= awu_f2k - 0.02 AND lifetime >= awu_f2k - 0.05.

| World | Machine | ff_bits all (6 / 7 / 8) | Mean | Return bits | Lifetime | Returns | Swap |
|---|---|---|---|---|---|---|---|
| s49 | awu_f2k | 0.4560 / 0.4650 / 0.4379 | 0.4530 | 1.458 | 95.82 | 68.6 | |
| s49 | awi_f2k | 0.4657 / 0.4657 / 0.4495 | 0.4603 | 1.463 | 95.81 | 69.1 | 62.3 |
| s49 | awid_f2k | 0.4225 / 0.4342 / 0.3889 | 0.4152 | 1.105 | 95.67 | 75.2 | 63.2 |
| bg002 | awu_f2k | 0.4565 / 0.4719 / 0.4498 | 0.4594 | 1.502 | 95.89 | 66.1 | |
| bg002 | awi_f2k | 0.4620 / 0.4764 / 0.4592 | 0.4659 | 1.503 | 95.90 | 66.9 | 72.1 |
| bg002 | awid_f2k | 0.4222 / 0.4317 / 0.3979 | 0.4173 | 1.118 | 95.71 | 74.0 | 69.4 |

**awi_f2k KILLED.** Folding the arrival into the big replay batch costs +0.007 bits in both worlds (it dilutes the
arrival to 1/33 of one step); lifetime unchanged.

**awid_f2k: compression PASSES, lifetime guard FAILS, so KILLED as default by the pre-registered rule.** Dense pairing
cuts whole-history bits by 0.038 (s49) / 0.042 (bg002), lower in 6/6 seeds, twice the bar and three to four times what
doubling the replay budget bought in H15 (0.009 / 0.014). Return bits fall by a quarter (-0.35 / -0.38) and returns rise
+6.6 / +7.9. But combined lifetime drops 0.15 / 0.18 points (bar -0.05), in every seed, and the cortex's own lifetime
drops the same (95.85 to 95.71/95.72).

Reading. Same replay items, same step count: putting one old item next to the new one in each gradient step moves
compression far more than doubling the replay budget did. That fits the CLS 1995 claim (interleaving constrains each
update) and the Chaudhry et al. 2019 experience-replay recipe (current + memory in one step). But under Adam each step
moves the weights by about lr whatever its batch, so dense pairing also hands old items roughly half of 84% of all update
directions, against 1 step in 39 for awu. That is the likely source of the lifetime cost: the cortex tracks the present
stream more slowly. H16b (pre-registered 09:06 UTC+8) asks whether a lower pairing rate (8/38, 16/38) keeps the
compression win without the lifetime cost.

## H16b: dense pairing rate (pre-registered 09:06 UTC+8, scored 09:20 UTC+8)

KEEP the lowest pairing rate that, in BOTH worlds, has ff_bits.all <= awu_f2k - 0.02 AND lifetime >= awu_f2k - 0.05.

| World | Pairing p (replayed per arrival) | ff_bits all (6 / 7 / 8) | Mean | Return bits | Lifetime | Returns |
|---|---|---|---|---|---|---|
| s49 | awu_f2k (0.84, separate step) | 0.4560 / 0.4650 / 0.4379 | 0.4530 | 1.458 | 95.82 | 68.6 |
| s49 | 8/38 (0.21) | 0.4701 / 0.4673 / 0.4427 | 0.4600 | 1.281 | 95.72 | 72.9 |
| s49 | 16/38 (0.42) | 0.4304 / 0.4497 / 0.4193 | 0.4331 | 1.150 | 95.73 | 74.2 |
| s49 | 32/38 (0.84) | 0.4225 / 0.4342 / 0.3889 | 0.4152 | 1.105 | 95.67 | 75.2 |
| bg002 | awu_f2k (0.84, separate step) | 0.4565 / 0.4719 / 0.4498 | 0.4594 | 1.502 | 95.89 | 66.1 |
| bg002 | 8/38 (0.21) | 0.4726 / 0.4655 / 0.4601 | 0.4661 | 1.336 | 95.82 | 70.5 |
| bg002 | 16/38 (0.42) | 0.4360 / 0.4469 / 0.4211 | 0.4347 | 1.167 | 95.77 | 72.8 |
| bg002 | 32/38 (0.84) | 0.4222 / 0.4317 / 0.3979 | 0.4173 | 1.118 | 95.71 | 74.0 |

**KILLED.** No rate passes both bars. Lifetime is below awu at every rate (-0.07 to -0.18), past the -0.05 guard every time;
the bits bar is reached only at 16/38 and up (s49 at 16/38 misses by 0.0001). Dense pairing sits on a
trade-off curve: each step of more pairing buys bits and returns and costs a little present-stream accuracy. Separate-
step uniform replay every 38th stays the default.

Open question for the group: the lifetime guard (-0.05) was set to protect the present stream. Dense pairing at 32/38
buys 0.04 bits per label (8% of the code length) for 0.15 points of lifetime. If the claim is whole-history compression,
this is the first lever since replay itself that clears the bits bar by 2x; it was killed by the accuracy guard, not by
compression. Next: H17 asks whether the trade is a capacity limit (a cortex with twice the hidden units).
