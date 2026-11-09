# H23b: compression progress with a slow compressor and held-out progress (pre-registration)

Pre-registered 9 Nov 2026, UTC+8, after headroom pilots on seeds 0–2 and before any scoring seed. Parent: H23
(`results/h23/RESULTS.md`, killed). World, channels, bandit, controllers and score W are unchanged from
`docs/h23_curiosity.md`. Code: `curio.py --comp sgd --prog heldout`, `CURIO_LR=0.004`.

## What changes, and why

1. **Compressor:** `SGDCompressor`, an online softmax regression on hashed features (channel bias + channel-local
   contexts of orders 1, 2, 3, 4, 6), one shared 65 536 × 41 table, lr 0.004. H23's KT model learned all it could in
   a few glances, so allocation had no headroom.
2. **Progress, held out:** learn the first 32 symbols of the glance, then reward = old vs new compressor on the *unseen*
   last 32 (per symbol), then learn those too. Memorising earns nothing, which removes H23's counter-prefix artefact.

## Headroom pilots (seeds 0–2, rand vs rand7 oracle)

| lr | T = 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| 0.02 | +0.093 | −0.025 | −0.063 | −0.033 |
| 0.004 | +0.213 | **+0.134** (per seed .20/.13/.07) | +0.102 | +0.065 |

At lr 0.02 the SGD model plateaus as fast as KT: no headroom, so nothing to test. At lr 0.004, W_rand still falls
0.78 bits from T = 512 to 2048. Budget rule carried over from H23 (smallest T in {512, 1024, 2048} with headroom
≥ 0.1): **T = 512**. Scoring seeds **6–11**.

Caveat, stated up front: lr 0.004 makes learning curves long *artificially*. H23b tests the controller mechanism. It
does not claim this compressor is good. The real fix for headroom is a compressor that can keep learning deeper
structure (H21).

## Bars (same as H23)

1. err ≥ 60% of glances on N.
2. prog ≤ 15% on N and ≤ 15% on Z.
3. **PASS:** W(prog) ≤ W(rand) − 0.05 in ≥ 5 of 6 seeds, and W(prog) < W(err), W(prog) < W(cnt) on the mean.
4. Stretch: W(prog) ≤ W(rand7) on the mean.

**Kill:** if 3 fails with headroom present, progress as estimated here does not allocate attention well, and the next
step is the controller (a learned predictor of progress, as in Schmidhuber 1991), not the compressor.
