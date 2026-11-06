# H17: 2× cortex hidden width under standing awu every-38

Pre-registered 08:55 UTC+8, 6 Nov 2026 (see `cls_sleep.py`), scored 09:20 UTC+8. Both worlds, seeds 6-8, forgetful store
(f2k), awake arrivals only, run on the box (`results/cls_h17_s49`, `results/cls_h17_bg002`). awu_f2k reused from H13/H16;
determinism re-checked (s49 seed 6 rerun on the edited script: 0.4560, identical).

- awu_w2_f2k: same as awu_f2k (uniform awake replay every 38th wake arrival, BS=32, forget=2000) but ForwardForward
  `hidden=1000` instead of the default 500. Same lr (1e-3), same SymBa+homeo, same store and router.

Bar. KEEP only if, in BOTH worlds, ff_bits.all <= awu_f2k - 0.02 AND lifetime >= awu_f2k - 0.05.

| World | Machine | ff_bits all (6 / 7 / 8) | Mean | Return bits | Lifetime | Returns | Swap |
|---|---|---|---|---|---|---|---|
| s49 | awu_f2k | 0.4560 / 0.4650 / 0.4379 | 0.4530 | 1.458 | 95.82 | 68.6 | |
| s49 | awu_w2_f2k | 0.4234 / 0.4196 / 0.4036 | 0.4155 | 1.407 | 95.82 | 69.9 | 67.0 |
| bg002 | awu_f2k | 0.4565 / 0.4719 / 0.4498 | 0.4594 | 1.502 | 95.89 | 66.1 | |
| bg002 | awu_w2_f2k | 0.4171 / 0.4248 / 0.4068 | 0.4162 | 1.415 | 95.91 | 68.5 | 75.1 |

**awu_w2_f2k KEEP.** Whole-history bits fall 0.037 (s49) / 0.043 (bg002), lower in 6/6 seeds, clearing the −0.02 bar
by ~2×; lifetime is flat (0.00 / +0.02 vs guard −0.05). Returns +1.3 / +2.4; return bits slightly down (−0.05 / −0.09).
Compression gain matches what dense pairing bought in H16 (−0.038 / −0.042) without that lifetime cost.

Reading. McClelland, McNaughton & O'Reilly (1995) and French (1991, cited there) treat catastrophic interference as a
consequence of overlapping distributed codes: more capacity (or sparser codes) lowers overlap and so the cost of each
update, at the risk of less shared structure. Here doubling hidden units under the standing separate-step awu schedule
buys ~0.04 bits/arrival of whole-history compression with no present-stream lifetime loss — so the H16 trade was not
"interleave or capacity" alone; capacity alone is a free lunch on this probe. Core updates: z5 + c3 + awu every-38 +
hidden=1000.

Next. H18: dense pairing (H16 awid) under the new width — does the lifetime guard now pass while keeping the bits win?
One mechanism (pairing on top of width). Fallback: shrink-and-perturb / slower wake LR plasticity.
