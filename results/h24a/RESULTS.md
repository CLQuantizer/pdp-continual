# H24a: the hippocampus is useful once the world has episodes and the store keeps them separate

Run 9 Nov 2026, UTC+8, VPS (`h24_episodic.py --seeds 6 7 8 --ticks 10000`, ~75 min). Pre-registration:
`docs/hippocampus_diagnosis.md`. World: the MNIST stream plus, with p = 0.2 per tick, one of 200 specific images
carrying a fixed arbitrary label (an "episode", recurring about every 1 000 ticks). Accuracy %, prequential, mean
of 3 seeds (per-seed spread ≤ 2 points on every row).

| Route | Lifetime | Normal digits | Episodes, 1st exposure | 2nd | 3rd–5th | 6th+ |
|---|---|---|---|---|---|---|
| ff (cortex) | 66.0 | 76.8 | 9.5 | 12.8 | 16.2 | 31.9 |
| proto (core ProtoStore) | 70.9 | 81.9 | 11.3 | 25.5 | 26.5 | 31.3 |
| exem (verbatim exemplar store) | **87.3** | 86.3 | 12.8 | **100** | **100** | **100** |
| gate (exem on near-identical match, else ff) | 79.6 | 76.8 | 9.5 | 100 | 100 | 100 |
| ffr (ff + 8 exemplar replays per arrival) | 78.1 | 83.6 | 10.3 | 18.7 | 45.4 | **79.0** |

The gate fired on 0.01% of normal arrivals and on 100% of repeat episodes (median nearest-neighbour d² between
distinct MNIST images: 27.4; threshold 1).

## Predictions

- **P1 PASS:** exem 100% from the 2nd exposure.
- **P2 FAIL as written:** proto 31.3% at 6+ (bar ≤ 30). In substance the averaging store is no better than the
  cortex on episodes (31.3 vs 31.9), while a store that keeps traces separate gets 100.
- **P3 PASS:** ff 16.2% at exposures 3–5.
- **P4 PASS:** gate +13.6 lifetime over ff, normal accuracy identical, 0.01% false firing.
- **P5 PASS on the main bar:** ffr +47 over ff at 6+. Its guard ("normal within 1 point of ff") is exceeded in the
  good direction (+6.8).

## What it shows

1. **The idea of a hippocampus was never the problem.** With episodic content in the world and a pattern-separated
   store, the store answers what the cortex cannot (100% vs 13–32%), and an exact-match gate takes those answers
   with no cost elsewhere.
2. **The core's ProtoStore is a second cortex.** Averaging into 9 prototypes per class erases episodes; it only beats
   ff on normal digits early, while its first 9 images per class are still verbatim.
3. **Replay consolidates.** Cortex + exemplar replay learns the arbitrary episodes (79% at 6+), so a consolidated entry
   could leave the store: the history-compressor scheme, now with evidence.
4. **Caveat:** 10 000 ticks is a short life; the cortex is undertrained (76.8% on normal digits), which is why exem
   and replay also help normal digits here. In the long MNIST life the cortex wins normal digits, as before. The
   episodic results do not depend on that.

## Next

- Default store for episodic worlds: exemplar traces, surprise-gated writes, delete once consolidated, priced in bits.
- Gate over ffr (replaying cortex + exact-match store) as the combined route; then H24b in the symbol world (handoff
  spec ready), scored with H20's two-part T.
