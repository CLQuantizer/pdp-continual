# H14: does replaying the most-interfered memories compress the history better than uniform replay?

Pre-registered 19:10 UTC+8, 1 Nov 2026 (see `cls_sleep.py`), scored 19:45 UTC+8. Both worlds, seeds 6-8, forgetful store
(f2k), awake arrivals only, run on the box (`results/cls_h14_s49`, `results/cls_h14_bg002`); awu_f2k / awr_f2k / none_f2k
reused from H13 (runs are deterministic). mir_f2k = Maximally Interfered Retrieval (Aljundi et al. 2019) in look-back form:
at every 38th wake arrival, re-score the 128 store prototypes sampled at the previous slot and replay the 32 whose cortex
bits rose most over the 38 wake updates since. Same slots and budget as awu (0.84 replayed per wake arrival).

Bar. KEEP as selective bus only if, in BOTH worlds, ff_bits.all <= awu_f2k - 0.02 AND lifetime >= awu_f2k - 0.05.
Diagnostic only: return bits vs awr_f2k.

| World | Machine | ff_bits all (6 / 7 / 8) | Mean all | Return bits | Lifetime | Returns | Swap first 300 | Novel first 300 |
|---|---|---|---|---|---|---|---|---|
| s49 | none_f2k | 0.5372 / 0.5156 / 0.4784 | 0.5104 | 1.937 | 95.73 | 59.1 | 63.1 | 74.2 |
| s49 | awu_f2k | 0.4560 / 0.4650 / 0.4379 | 0.4530 | 1.458 | 95.82 | 68.6 | 63.8 | 71.9 |
| s49 | awr_f2k | 0.4585 / 0.4669 / 0.4407 | 0.4554 | 1.348 | 95.80 | 71.1 | 66.3 | 72.7 |
| s49 | mir_f2k | 0.4558 / 0.4664 / 0.4423 | 0.4548 | 1.376 | 95.83 | 70.1 | 64.7 | 72.0 |
| bg002 | none_f2k | 0.5184 / 0.5236 / 0.5015 | 0.5145 | 1.988 | 95.81 | 57.4 | 75.0 | 74.2 |
| bg002 | awu_f2k | 0.4565 / 0.4719 / 0.4498 | 0.4594 | 1.502 | 95.89 | 66.1 | 72.6 | 71.9 |
| bg002 | awr_f2k | 0.4487 / 0.4713 / 0.4545 | 0.4581 | 1.392 | 95.85 | 69.5 | 73.2 | 72.7 |
| bg002 | mir_f2k | 0.4569 / 0.4687 / 0.4496 | 0.4584 | 1.413 | 95.95 | 68.1 | 74.0 | 72.0 |

**KILLED.** mir_f2k - awu_f2k on ff_bits.all: +0.002 (s49), -0.001 (bg002); bar <= -0.02. Lifetime passes (+0.01 / +0.06).

The selector did find real interference: the 32 chosen prototypes had risen 0.14 to 0.15 bits since sampling, while the
128 candidates as a whole rose about 0.00 (all 6 seeds). Replaying them still leaves the whole-history code length where
uniform replay puts it. As with absence (H13), the effect is a reallocation: return bits 0.08 to 0.09 below uniform,
0.02 to 0.03 above absence; returns +1.5 / +1.9 over uniform, below absence.

Reading. Three selectors now tie uniform on whole-history bits within 0.003: need (absence, H10/H13), a global gain gate
(H11) and per-memory interference (H14), with a permanent and a forgetful store. Replay itself is worth about 0.055 bits
per arrival; which memories go into the batch is worth nothing measurable on that axis. Selection closes here. The
remaining compression lever is how much and how replay changes the weights, not which items it carries.
