# H13: does absence-selected replay earn its keep when the store forgets?

Pre-registered 18:20 UTC+8, 1 Nov 2026 (see `cls_sleep.py`), scored 18:50 UTC+8. Both worlds, seeds 6-8, awake arrivals
only, run on the box (`results/cls_h13_s49`, `results/cls_h13_bg002`). f2k = the store drops every digit unseen for
>= 2000 ticks, so the cortex has to own returns. awr_f2k = every 38th wake arrival, one batch of store prototypes from
digits absent >= 300 ticks; awu_f2k = same clock and budget, any digit.

Bar. KEEP absence (as the default whenever the store can forget) only if, in BOTH worlds, awr_f2k ff_bits.all <=
awu_f2k - 0.02 AND returns >= awu_f2k + 1.5 AND lifetime >= awu_f2k - 0.05. Sanity: awu_f2k <= none_f2k - 0.02 bits.

| World | Machine | ff_bits all (6 / 7 / 8) | Mean all | Return bits (6 / 7 / 8) | Lifetime | Returns | Swap first 300 | Novel first 300 | Replay ÷ wake |
|---|---|---|---|---|---|---|---|---|---|
| s49 | none_f2k | 0.5372 / 0.5156 / 0.4784 | 0.5104 | mean 1.937 | 95.73 | 59.1 | 63.1 | 74.2 | 0.00 |
| s49 | awr_f2k | 0.4585 / 0.4669 / 0.4407 | 0.4554 | 1.367 / 1.276 / 1.402 | 95.80 | 71.1 | 66.3 | 72.7 | 0.82 |
| s49 | awu_f2k | 0.4560 / 0.4650 / 0.4379 | 0.4530 | 1.497 / 1.346 / 1.531 | 95.82 | 68.6 | 63.8 | 71.9 | 0.84 |
| bg002 | none_f2k | 0.5184 / 0.5236 / 0.5015 | 0.5145 | mean 1.988 | 95.81 | 57.4 | 75.0 | 74.2 | 0.00 |
| bg002 | awr_f2k | 0.4487 / 0.4713 / 0.4545 | 0.4581 | 1.434 / 1.320 / 1.421 | 95.85 | 69.5 | 73.2 | 72.7 | 0.82 |
| bg002 | awu_f2k | 0.4565 / 0.4719 / 0.4498 | 0.4594 | 1.525 / 1.381 / 1.600 | 95.89 | 66.1 | 72.6 | 71.9 | 0.84 |

**KILLED by the pre-registered bar, on the compression term only.** awr_f2k - awu_f2k on ff_bits.all: +0.002 (s49),
-0.001 (bg002); bar <= -0.02. The accuracy terms pass in both worlds: returns +2.5 / +3.4 (bar +1.5), lifetime -0.02 /
-0.04 (bar -0.05). Sanity passes: uniform replay is 0.057 / 0.055 bits below none.

Reading. Under forgetting, absence selection buys +2.5 to +3.4 returns and lowers the cortex's code length on return
arrivals by about 0.11 bits in 6 of 6 seed pairs, yet the total code length over the whole history does not move. It
reallocates the cortex's capacity toward the absent digits rather than compressing the history better. By the group
rule as pre-registered (whole-history bits), the selector is demoted in both regimes. Whether the group's compression
criterion should be the whole history or the part the store can no longer answer is now the open question; the
pre-registered verdict stands until a new bar is written down before any new run.
