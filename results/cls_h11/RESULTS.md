# H11: does a gain gate (replay only when the cortex is surprised) beat absence-selected replay?

Pre-registered 17:25 UTC+8, 1 Nov 2026 (see `cls_sleep.py`), scored 18:20 UTC+8. Awake arrivals only.
bg002 seeds 6-8 run on the box (`results/cls_h11_bg002`); none, awr, awu reused from H10 (runs are deterministic).
s49 seeds 6-8 awg + awr still finishing on the Dubai VPS (`results/cls_h11_s49`); reported for context only, because the
bar needs BOTH worlds and bg002 already decides it.

Machines. awr = core (every 38th wake arrival, one batch of store prototypes from digits absent >= 300 ticks: need only).
awu = same clock, any digit (no selector). awg = gain only: one token per 38 awake arrivals (cap 8), spent on a uniform
batch only while the 38-arrival running mean of the cortex's code length exceeds its 500-arrival median.

| Machine (bg002) | ff_bits all (6 / 7 / 8) | Mean all | Mean returns | Lifetime | Returns | Swap first 300 | Novel first 300 | Replay ÷ wake |
|---|---|---|---|---|---|---|---|---|
| none | 0.5184 / 0.5236 / 0.5015 | 0.5145 | 1.988 | 95.86 | 75.2 | 70.6 | 74.2 | 0.00 |
| awr | 0.4465 / 0.4837 / 0.4508 | 0.4603 | 1.302 | 95.86 | 81.8 | 69.7 | 73.4 | 0.82 |
| awu | 0.4483 / 0.4730 / 0.4484 | 0.4566 | 1.360 | 95.92 | 81.9 | 67.8 | 71.4 | 0.84 |
| awg | 0.4570 / 0.4694 / 0.4607 | 0.4624 | 1.359 | 95.89 | 81.8 | 69.4 | 73.0 | 0.83 |

**H11 KILLED.** awg returns +0.05 over awr (bar +1.0) at the same replay budget (0.83 vs 0.82 per wake arrival), so neither
the accuracy route nor the half-budget route passes. As a selective bus it also fails: ff_bits 0.4624 vs awu 0.4566
(+0.006, bar <= -0.02).

**H11b (uniform clock replaces awr): fails on bg002 swap300**, 67.8 vs 69.7 (-1.9, bar -1.0); returns and lifetime pass.
s49 awu 6-8 for context: 95.87 / 82.5 / 61.9, ff_bits 0.4499.

Reading. Three selectors (need only, gain only, none) at one budget sit within 0.006 bits of each other and 0.1 points of
returns. With a permanent store, that the cortex rehearses matters (-0.054 bits vs none); what it rehearses does not. The
gain gate here decides *when* to replay from a global error signal; it never asks which memory the coming update would
damage. That is the version of gain the literature actually supports (see audit 18:20, Aljundi et al. 2019), and it is
only worth testing where selection can matter at all, which is the forgetful store (H13, running).
