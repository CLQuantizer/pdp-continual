I'll read the project instructions and the full handoff, including the amendment that the earlier run stopped before.The amendment changes when progress is scored: measure stored sights before the current glance is added. I'll read the prior handoff, W8-1b results, and the reservoir code before changing anything.The first pass already recorded TV as the largest reward. I'll read the reservoir, the reward order, and the existing W8d script so the amendment only changes when the glance is stored.The first pass scored the glance after it was already stored, so the TV's own random greys counted as progress. I'll score the sights from before this glance, then store it, and keep the old order only for the comparison.The still-panel check agrees under both orders, and the first glance scores 0. Both papers are already in the index. Next I’ll run the life-0 gate, then the scored lives if the TV is the smallest reward.The first-pass gate passed. On life 0 the TV visit reward is now the smallest: TV −166.2 bits, fast 533.1, static 542.7, slow 585.4.

Progress is scored on the sights stored before the current glance, then the glance is stored. On the still panel the two progress functions still agree (max absolute difference about 10⁻¹⁴), and the first glance scores 0. The scored run of lives 0–19 is going. Both papers were already in the index.

The scored run finished. I'll read the results and check the three bars.The amendment is in, and the scored run finished. Scoring progress before the glance is stored makes the TV the smallest first-pass reward. Lives 0–19 are in `results/w8d/RESULTS.md`.

Life 0 first pass, one 8-glance visit per panel:

| policy | static | tv | fast | slow |
|---|---:|---:|---:|---:|
| curious_prog | 542.696 | −166.181 | 533.062 | 585.372 |
| curious_prog_glance | 2164.082 | 4119.198 | 2162.345 | 2095.402 |

`curious_prog` and `curious_prog_noage` score `ObservedReservoir.progress` on the sights already stored, then store the glance. The first glance into an empty reservoir is 0. `curious_prog_glance` is still W8-1b: include the glance, then score.

On life 0’s static panel the two progress functions agree under both orders (max absolute difference 2.842e-14 before, 8.527e-14 after). The first of those 64 glances is 0.000 under the new order and 323.337 under the old one. The full table is at the bottom of `RESULTS.md`. `random`, `round_robin`, `curious_surp`, and `curious_prog_glance` match `results/w8c` on all 20 lives.

Bars, 20 lives:

| bar | observed | result |
|---|---|---|
| 1 curious_prog − round_robin ≥ +0.03, CI excludes 0 | −0.192448, CI [−0.231771, −0.153385], 0/20 lives above 0 | FAIL |
| 2 TV share: prog ≤ 0.15 and surp ≥ 0.35 | prog 0.076172, surp 0.730469 | PASS |
| 3 prog share fast > slow > static | fast 0.098437, slow 0.125391, static 0.700000 | FAIL |

`curious_prog` mean shares are static 0.700, slow 0.125, fast 0.098, TV 0.076. `curious_prog_noage` is static 0.824 and TV 0.032. `curious_prog_glance` remains TV 0.591. `curious_prog` minus `curious_surp` accuracy is +0.035156, CI [0.005469, 0.063021], 13/20 lives above 0.

Wall time was 2956 s. The full run started at a 1-minute load of 0.25. Demo strips are in `internal/w8_demo/wall_c/`. `zhang2019` is already `checked` / `stored` (arXiv 1905.08094). `borbely1982` is already `checked` / `wanted`. Nothing was committed.
