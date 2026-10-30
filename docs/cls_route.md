# CLS complementary routing: verdict (2026-10-30 17:27 UTC+8)

Code: `cls_route.py` (Split-MNIST + 2<->3 swap, 8000 examples), `cls_life.py` (life world, never-sleep,
bg 0.2, 16 days). All routes are scored on the same network + store at every step (paired). 3 seeds.
Split runs use clock sleep (proto9_nonoise recipe, 0.8x replay); life runs never sleep (the 94.1 baseline).

Routes: ff = FF network (the FF+sleep / life machine); store = nearest prototype (store alone);
m{tau} = store answers if its relative margin >= tau or its class is new (<1000 seen), else FF;
rel = per-predicted-class stream reliability arbiter; relp = rel with an awake-probe false-alarm discount;
gate = FF trusted only about classes in the last 2000 labels (stream precision decides when both name
recent classes; FF recent vs store old -> FF only if probe false-alarm rate < phi); oracle = either right.

## Against the three targets (9 protos/class, the memory the targets were set at)

| Route | Life lifetime (>=94.1) | Split final (>=91.6) | Swap new mapping @8000 (>=85) | all 3? |
|---|---|---|---|---|
| ff (FF+sleep / never-sleep) | 94.03 | 89.7 | 81.7 | no |
| store alone | 84.73 | 91.6 | 53.2 | no |
| m0.1 (margin) | 89.45 | **92.6** | 66.3 | no (1/3) |
| m0.4 | 93.57 | 90.8 | 80.2 | no (0/3) |
| rel | **94.21** | 90.4 | **85.6** | no (2/3) |
| gate0.02 | **94.21** | 91.5 | **88.0** | no (2/3, Split misses by 0.1) |
| oracle (bound) | 96.93 | 95.6 | 89.4 | – |

40 protos/class: store 94.2, m0.05 94.6 final; gate 94.0 final / 83.2 swap; FF 90.8.

## Verdict
- **No router passes all three.** Best = gate0.02: life 94.21 (pass, but only +0.18 over FF),
  Split 91.5 (fail by 0.1), swap 88.0 (pass). Pre-registered kill criterion ("router must beat the better
  of its two routes by >= 0.5"): **killed in the life** (+0.18) and on Split for gate (−0.1 vs store);
  survives only for the margin router on Split (+1.0 over store at 9/class, +0.4 at 40/class), which then
  fails swap and life.
- The trade-off is structural: confidence (margin) routing helps class-IL because the store is
  confidently right on old classes, but the store is also confidently *wrong* after the rule change, so
  margin routing can't adapt. Reliability routing adapts but can't see FF's forgetting of absent classes.
- **Stream-reliability is biased in class-IL** (falsified path): any answer naming an absent class is
  wrong by construction, so per-class stream precision punishes the store's old classes and hands them
  to the forgetful FF (rel final 90.4 ≈ FF). The awake-probe discount (relp) doesn't fix it: FF classifies
  the stored prototypes correctly because sleep replays exactly them; the probe is contaminated by replay.
- Oracle headroom is large (95.6 Split, 96.9 life, +3–4 points): the two systems err on different items,
  so the problem is the arbiter, not the routes. The arbiter needs a signal of FF reliability on held-out
  old inputs that isn't the replayed prototypes (e.g. jittered/held-out store items, or agreement of FF
  with the store on fresh unlabelled inputs).
- Side finding: under a long swap FF itself drifts back (new mapping 95.9 @500 -> 71.9 @2000 -> 81.7
  @8000) because clock sleep replays the store's stale 2/3 prototypes with the old labels. Hypothesis 5
  (store adaptation / change detector) is now the blocking item for any CLS route.
