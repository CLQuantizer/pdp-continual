# Audit: progress and directions (2026-10-30)

A snapshot of where the pdp research stands. Detail lives in `README.md`
(experiments), `docs/notes.md` (ideas and ★ results) and `docs/goal.md`
(criteria and scoreboard). The previous audit (2026-10-29) ended at the
sandal check. Since then the work has moved from *what to replay* to
*when the network forgets*, and that changed how every earlier number
should be read.

## Goal

A network plus a training method that doesn't depend on GPU-era
assumptions: local learning (no backprop, no update locking), sparse
activity, memory next to compute, continual learning with controlled
forgetting, wake/sleep modes, and robustness to imprecise hardware.
Methods are scored on hardware-neutral costs (floats stored, replay
amount, sparsity), not only accuracy.

## Headline: the final scores are a post-sleep best case

Every class-IL number before 2026-10-30 is measured at example 60000.
That is a multiple of the sleep clock, so it is taken just after a sleep.
**Anytime accuracy** (class-IL measured every 700 wake examples, wherever
the clock is) tells a different story for the standard setup (9 noise-free
prototypes, sleep every 2000):

| | Final (after a sleep) | Anytime average | Anytime, tasks 3–5 | Anytime worst |
|---|---|---|---|---|
| Clock 2000 | 89.3 | **68.0** | 54.1 | 24.9 |

Just before each sleep, a median of about 45% of stored prototypes are
misclassified. **Between sleeps, plain FF wake learning erases old
classes almost as fast as with no sleep at all, and each sleep rebuilds
them.** Sleep has been repairing the damage, not preventing it. This is
now the central problem.

## Phase 1: baseline benchmark (`bench.py`, `learners.py`)

- On standard MNIST every method reaches 95–98.4%. Forward-forward (FF)
  with the SymBa loss and a homeostatic bias reaches **98.40%**, level
  with backprop.
- **Every gradient-trained method forgets completely on Split-MNIST
  class-incremental (≈20%).** That covers backprop, feedback alignment,
  direct feedback alignment and FF. Forgetting comes from shared weights,
  not from how errors travel.
- Random features + an accumulate-only ridge readout score 95.5%, but the
  readout needs a dense O(H³) solve.

## Phase 2: wake/sleep FF (`sleep.py`, `sleep_cost.py`)

- Online FF plus a sleep every 2000 examples that replays a bounded store
  takes class-IL from **20% to 90%** (final score; see the headline
  above). Replaying 0.8× as many examples as were seen awake is enough.

## Phase 3: dreams instead of stored images (`dreams.py`, `sandal.py`) — closed

- A local, solve-free top-down generator reaches 89.1% on Split-MNIST and
  74.6% on Split Fashion-MNIST. Over 10 seeds it is significantly
  *worse* than prototype replay (−0.5 to −0.7, p ≈ 0.04), and it uses
  twice (MNIST) to four times (Fashion) the memory of noise-free
  prototypes.
- The sandal check showed why decoders lose sandals. It isn't averaging
  of styles (every store covers 7–8 of 8 styles). It's **incoherent fine
  detail**: ridge dreams are ghosted superpositions, and local-decoder
  dreams are speckled. Random detail actively hurts.
- Verdict: the local decoder's only case is that it stores no images.
  Further polishing is worth fractions of a point.

## Phase 4: forgetting over time (`swap.py`, 2026-10-29 → 30)

### Label-swap adaptation test
After Split-MNIST, digits 2 and 3 swap labels for 8000 examples.
- Awake, every method takes in the swap almost at once (96% after 500
  examples).
- **Each sleep partly undoes it,** because the store still replays the
  old mapping. Stores turn over by counting rules, blind to change.
  Prototypes adapt fastest (85%), only because of their learning-rate
  floor.
- Conflicting wake data drops the other 8 digits from 90% to 30–43%
  within 500 examples. This led to the anytime metric and the headline
  above: the collapse happens in the ordinary stream too.
- The local generator has no advantage here.

### Surprise as a signal: mostly the wrong one
- **Surprise-gated store updates** (move the prototype faster when the
  prediction contradicts the label) adapt much faster on the swap (91 vs
  85). Over 10 seeds they are neutral on MNIST (+0.16, noise), and they
  cost **−1.7 on Fashion** on every seed. Surprise mixes "the world
  changed" with "the example is ambiguous" (Yu & Dayan's expected vs
  unexpected uncertainty). On a reservoir store it is worse: slots
  become the network's mistakes (−4.1).
- **Surprise-triggered sleep** adds nothing beyond more sleeps. A matched
  faster clock beats it, because surprise stops about 100 examples after
  each sleep while the damage keeps building.

### Interference-triggered sleep
- **Probe trigger** (re-classify 30 stored prototypes while awake; sleep
  when accuracy drops 50%): at matched replay it raises the worst point
  a lot (46.6 vs 29.1) and the average slightly (76.1 vs 74.7). It is
  the right signal, but it manages the symptom.
- **Weight-drift trigger** (Tononi–Cirelli sleep pressure) acts as a
  task-boundary detector under Adam. It gives the best final score (90.28,
  3 seeds, suggestive) and worse anytime accuracy.
- No trigger brings anytime accuracy near the final score without
  doubling replay.

### Making waking less destructive
| Approach | Final | Anytime avg | Recent-class share (true 19.8%) |
|---|---|---|---|
| Clock 2000 (baseline) | 89.3 | 68.0 | 19.8% |
| Faster clock (400 / 160 / 40) | 71–78 | 83–85 | 40–46% |
| Awake replay 0.8, no sleep | 79.1 | **91.2** | 38.5% |
| Awake replay 0.4 + clock 2000 | 83.2 | 89.8 | 34.2% |
| **Wake lr ×0.3 + clock 2000 (10 seeds)** | 89.5 | 74.7 | 19.9% |

- **Interleaving replay into every wake update prevents the collapse.**
  Anytime accuracy reaches 91%, with the worst point at 76% vs 25%. Even
  0.2 items per update gives 88%.
- **But it builds a recency bias:** the last task takes 38% of
  predictions, and the final score drops 10 points. Faster clocks show the
  same bias, worse.
- **Slow wake learning is a free win** (complementary learning systems:
  fast store, slow cortex). With the wake rate at ×0.3, anytime accuracy is
  **+7.2 on 10/10 seeds** and retention during the swap is +10.7. The final
  score is unchanged and the cost is slightly slower adaptation. **On
  Fashion, ×0.1 gives 78.48%, the best Fashion result so far** (previous
  best 77.29, all 3 seeds higher, anytime +12).
- **Recency-bias diagnosis:** the "real texture = current class" shortcut
  hypothesis was **refuted**. A store of real images is *more* biased (57%
  recent share with awake replay). The current hypothesis is
  **memorised replay vs fresh data**: each stored item is replayed about 100
  times per task, so old classes are memorised rather than generalised. The
  prediction is that more varied replay should reduce the bias.

## Running now: `results/varied`

`swap.py --out results/varied`, 10 conditions × 3 seeds, started 12:35 on
2026-10-30. The log shows BrokenPipe errors and duplicated lines, so the
run seems to have been restarted. Partial so far: pixel noise under
awake replay gives 83.0 / 85.3 / 83.5 final vs 83.0 without noise (a
small effect, not a fix). Jitter without awake replay gives 89.9 on
seed 0. It tests the memorisation hypothesis by varying
replay three ways:
- pixel noise on prototypes;
- **jitter**, shifting replays by up to 2 px;
- 40 prototypes per class instead of 9.

Each is run with and without awake replay 0.8. One combination also has
wake lr ×0.3. **What to look for:** the recent-class share under awake
replay falling from 34–38% toward 20%, with anytime accuracy staying near
90%. If that happens, awake replay + varied replay + slow waking could
give both a high anytime score *and* a high final score, which no
configuration has done yet.

## Scoreboard (best so far)

| Criterion | Best | Caveat |
|---|---|---|
| Local learning, iid | FF (SymBa + homeostasis) 98.40% | |
| Continual, final (MNIST) | ~89.5–90.3% (prototypes; drift trigger 90.28 at 3 seeds) | post-sleep best case |
| Continual, anytime (MNIST) | awake replay 0.8: 91.2% | final 79% (recency bias) |
| Continual, both | wake lr ×0.3 + clock: 89.5 final / 74.7 anytime | anytime still 15 points under final |
| Continual, Fashion | prototypes + wake lr ×0.1: **78.48%** | 3 seeds; one-pass ceiling 83.4 |
| No image store | local decoder 89.1% MNIST / 74.6% Fashion | 2–4× prototype memory |
| Adaptation (label swap) | surprise-gated prototypes 91% new mapping | hurts Fashion |
| Sparsity, asynchrony, robustness | **not measured** | |

## Directions not started

1. **Two-compartment FF neuron:** the label goes on an "apical" branch;
   label-gated branches need no task ID.
2. **Context-gated dendritic units** on Split-MNIST, context inferred
   from the input. Phase 4 makes this more urgent. Its own conclusion was
   that the fix is units "whose context gating keeps a two-class stream
   away from other classes' weights."
3. **Locally learned mixture of experts** with a homeostatic router.
4. **An unlabelled fast store** (Hebbian layer as hippocampus). Every
   store so far is supervised.
5. A **change detector that discounts expected errors** (confidently
   wrong, or a class's error rate above its own baseline), for adaptation
   without hurting Fashion.
6. **Pipelined, one thread per layer,** to measure update locking.
7. Criteria not measured: **sparsity, asynchrony, robustness to noise.**

## Against the north star (added 2026-10-30)

`docs/goal.md` now leads with the north star: **a machine that lives,
alternating between acting and learning like an animal alternates between
waking and sleeping.** It has no train/inference split, carries its own
data and algorithm, and sleeps when its own state says so, at a cost.
Measured against that, the project is about halfway there. Some parts of
the current setup are still offline training in disguise.

| Piece | Already alive | Still offline in disguise |
|---|---|---|
| Learning | FF learns one example at a time from the stream, locally | – |
| Data with the algorithm | the machine carries its own fast store (prototypes) and replays it itself | the store is a hand-built per-class table with given labels, not an episodic memory |
| Sleep timing | the probe trigger (sleep when its own memories start failing) is internal | the default is an experimenter's clock every 2000 |
| Sleep cost | – | **sleep is free**: the world pauses while it sleeps, so "sleep more" always looks good |
| Evaluation | anytime accuracy is a step toward it | the headline is a **frozen post-stream test**, the very train-then-deploy split we want to remove |
| The world | the label swap introduced change | Split-MNIST is one pass of 5 tasks that never return, so there's no life to live and nothing to judge relevance by |

Three consequences for how to read Phase 4:

1. **"Recency bias" may not be a bug.** Awake replay was judged a failure
   because its last-task share (38%) hurts a test where all classes are
   equally likely. In a real world, what you saw recently is what you'll
   probably see next, and animals are deliberately biased this way. Only a
   stream where classes recur can say whether awake replay's bias is
   harmful or adaptive. The uniform final test can't.
2. **Free sleep distorts the trigger results.** Faster clocks and
   lower probe thresholds "help" partly because sleeping costs nothing.
   With a price on sleep, the real trade-off appears: sleep must be worth
   the inputs it misses. That is what makes it *useful* rather than just
   present.
3. **Slow waking and awake replay are the right kind of result.** They
   change how the machine behaves while awake, which is where it lives.
   The Phase 3 decoder work optimised a post-hoc score and is rightly
   closed.

### Proposed next step: a life, not a benchmark (`life.py`)

A harness where the machine lives instead of being trained:

- **The world:** an endless stream in which classes come and go and
  **recur** in seasons. Rules sometimes change (label swaps), and there is
  a **day/night rhythm**: heavy traffic by day, sparse at night.
- **The loop, every step:** the input arrives, and the machine must answer
  (scored), then usually receives the label and may learn. It can choose
  to sleep for k steps; inputs during sleep go unanswered and count as
  misses.
- **Sleep pressure:** S builds from interference (the awake probe's
  failure rate) and C comes from how quiet the world currently is. The
  machine sleeps when S is high relative to C's price (Borbély's
  two-process model). Sleep length is set by the pressure too, and sleep
  discharges it.
- **Scores:** lifetime prequential accuracy (primary); forgetting curves
  by time since last seen (does it keep what recurs and let go of what
  doesn't?); adaptation lag after a rule change; time asleep; store size
  and ops.
- **Contenders:** never sleep; clock sleep; probe-pressure sleep;
  pressure + circadian; each with and without awake replay and slow
  waking.
- **Predictions to falsify:**
  - (a) pressure + circadian sleep beats any clock at equal time asleep;
  - (b) awake replay's recency bias scores *higher* on a recurring stream
    than on the uniform test;
  - (c) with sleep priced, the best machine sleeps mostly at night, in
    consolidated bouts, the way animals do.

Everything built so far plugs in unchanged: FF, prototype store, probe,
slow waking, awake replay. This turns the project's metric into the north
star and makes "is sleep organically useful?" a measurable question.

## Assessment

1. **The anytime metric is the most important finding since sleep
   itself.** It shows the system's real weakness: wake learning is
   destructive, and sleep only repairs it. Future comparisons should report
   final *and* anytime accuracy. The scoreboard in `docs/goal.md` should
   carry both.
2. **Slow waking is the one change that helped everywhere** (anytime,
   swap retention, Fashion final) at zero cost. It should become the
   default wake rate, ×0.3 for MNIST and ×0.1 to ×0.3 for Fashion, so later
   experiments build on it.
3. **Awake replay is the most promising route to closing the
   anytime/final gap.** Whether its recency bias can be removed is the open
   question that `results/varied` is answering. If varied replay doesn't
   fix it, other candidates are balancing the current class's share of each
   update, or a short sleep after each task.
4. **Every patch so far works at the level of the data stream**: what,
   when and how fast to replay. None changes the unit. Both the
   anytime problem and the recency bias are interference inside shared
   weights, which is exactly what the project's core hunch says richer,
   gated units should solve. Once the varied-replay question is settled, the
   dendritic/gated-unit experiments (1–2) are the natural next phase.
   Anytime accuracy is a much sharper test for them than the final score.
5. **Statistical hygiene.** Twice now, a 3-seed result was overturned at
   10 seeds (local decoder "level"; surprise gating +0.7). The drift-trigger
   final (90.28) and the Fashion slow-wake result (78.48) are still 3-seed
   numbers and need 10 seeds before they're treated as established.
6. README.md is about 1600 lines of chronological sections. This audit plus
   the goal.md scoreboard are now the quickest way in. A short "current
   best setup" block at the top of the README would help.

## Update 2026-10-30 15:10 UTC+8 (hourly loop; detail in `docs/loop.md`)

- **Store-alone baseline:** nearest-prototype readout of the same store, no network, scores
  91.6 (9/class) / 94.2 (40/class) on Split-MNIST and 77.0 / 80.1 on Fashion, with no
  between-sleep forgetting. FF + sleep (89.3 / 91.0; Fashion 78.48) is at or below it. Every
  continual result must now beat store-alone at equal memory (`results/hippo_route`).
- **Life (bg 0.2):** never sleeping wins (94.1); sleep's maximum possible gain is ≈2.5 points
  because off-season inputs are 13% and already 81.6% right. Pressure triggers never fire.
  Trigger tuning on this world is closed until the world rewards memory (returns score).
- **Slow waking costs in a life** (−1.0 lifetime, novel −13, swap −24 in the first 300):
  hold off on making ×0.3 the default.
- Next: dual-route answering (store answers when confident, FF otherwise).

## Update 2026-10-30 17:30 UTC+8 (hourly loop; detail in `docs/loop.md`)
- bg 0.02 life: night sleep lifts returns 53.5 → 64.5, but the store alone answers returns at 83.1.
  Answering from memory beats consolidating it for returning digits.
- Closed: per-class reliability arbitration (no lifetime gain, worse returns).
- Age gate (store answers stale classes) gives returns 67.5 with no sleep but costs 2.8 lifetime;
  next is an uncertainty-gated version (loop 17:00 hyp 1).

## Update 2026-10-30 18:50 UTC+8 (hourly loop; detail in `docs/loop.md`)

- Closed: age-gated store answering, including the uncertainty-gated variant (hyp 1 killed on two seed sets).
  Class age is anti-informative among FF/store disagreements (AUC 0.34–0.40); FF goodness margin is the signal (0.80–0.88).
- New best answering route for the life world (`gate_probe.py`, held-out seeds): margin gate without age,
  lifetime 96.1 vs FF 96.0, returns 65.1 vs 53.6, with no sleep. Costs 8 points on the swap.
- Next: gain-prioritised night replay (replay the store near low-margin disagreements).

## Update 2026-10-30 19:40 UTC+8 (hourly loop; detail in `docs/loop.md`)

- Closed: gain-prioritised night replay (`gain_replay.py`, 18:00 hyp 1 killed). Spending half or all of each
  night batch on store slots near low-margin disagreements gives returns 63.7 (unlabelled) / 62.2 (label-verified)
  / 58.2 (all-gain) vs uniform night 64.5, at the same 95.0 lifetime. The more replay is focused, the worse returns get.
- Reading: in this world the binding term of Mattar & Daw's gain × need is *need* (coverage of absent classes), not gain.
- Next: need-prioritised night replay (replay classes in proportion to time away), then the change-detector veto for the swap.

## Update 2026-10-30 20:25 UTC+8 (hourly loop; detail in `docs/loop.md`)

- Closed: need-prioritised night replay (`need_replay.py`, 19:40 hyp 1 killed). Weighting classes by time away
  (up to ~11:1) gives returns 64.2 vs uniform 64.5 at the same 95.0 lifetime. With gain replay (19:40) this closes
  night-replay composition as a lever in the bg 0.02 world.
- Reading: McClelland, McNaughton & Lampinen 2020 put protection at wake time, weighted by similarity to the new
  input; that is the next replay test (awake, similarity-weighted), after the mismatch veto for the swap.

## Update 2026-10-30 22:15 UTC+8 (hourly loop; detail in `docs/loop.md`)
- 21:10: an answer veto keyed on the store's claimed class fixes the swap but costs returns (65.1 to 56.1); closed.
- 22:15: rewriting stale store labels (per-prototype conflict remap, flip rule) is killed on returns; about half
  of all remaps happen before the swap and are false. A coordinated class-level rule misses its prediction.
  Remapping does lift the store alone from 85 to about 92 lifetime. Next: only remap prototypes whose class is
  currently present (latent-cause view, Gershman et al. 2017), then SWIL.

## Update 2026-10-30 23:05 UTC+8 (hourly loop; detail in `docs/loop.md`)
- 23:05: first remap rule to pass its pre-registered prediction. A recency presence gate (label seen in the last
  200 ticks, g3) is killed (27 false pre-swap remaps), but a *rate* presence gate (>= 5 arrivals of the class in the
  last 500 ticks, r3) gives 0 false pre-swap remaps on held-out seeds 3-5, swap 76.1, returns 65.8, lifetime 96.2
  under the margin gate, and lifts the store alone to 91.6 lifetime / 82.0 returns / 60.2 swap (plain store 85.0 /
  80.1 / 2.8). Caveat: 5-in-500 assumes bg 0.02; robustness at bg 0.2 is the next test.

## Update 2026-10-31 01:30 UTC+8 (hourly loops 00:10 and 01:05; detail in `docs/loop.md`)
- 00:10: fixed-count presence (r3) fails at bg 0.2 (32 false pre-swap remaps); the scale-free a25 rule (count >= 25%
  of the top label's count in the window) gives 0 / 1 and replaces it. Loosening the margin gate to mq30_s2 /
  mq50_s2 is affordable once the store remaps (+3 returns, ~-4 swap).
- 01:05: **current default = a25 store with W = 250 + mq30_s2.** Held-out bg 0.02: lifetime 96.1, returns 68.3,
  swap 71.2 (FF 96.0 / 53.6 / 78.4); bg 0.2 (all held out): lifetime 94.4 vs FF 94.1, swap 70.2 vs 74.2. W = 250
  keeps 0 / 1 pre-swap remaps in both worlds and gains 1-2 swap points over W = 500. Remaining cost: swap.
- 01:05: **error-driven store vetoes closed as a family.** The change-point veto (NE-style, Yu & Dayan 2005) fails
  like the absolute veto did: returns 68.3 -> 60.5 (absolute 57.4) for swap 71.2 -> 76.2. At bg 0.02 a class's
  return after a long absence produces the same burst of confident store errors as the swap, so any veto keyed on
  store error rate removes the answers that make returns good.

## Update 2026-10-31 02:35 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Remap latency is set by the evidence each prototype needs (H conflicting hits), not by how many prototypes must
  agree (K). Single-hit remap (H = 1) buys ~4 swap points but writes 16-67 false memories per run before the swap:
  killed. Weighting a conflict by its depth in the prototype's basin does not rescue it, because relabelled digits
  land at the basin rim like ordinary borderline digits (`close_remap.py`; c = 0.7 is a lead at swap 73.9, unconfirmed).
- Default store is now a25w250h2k2 (+1.0 swap at bg 0.02, +1.4 at bg 0.2, equal returns, <= 1 false remap).
- Next: one-shot remap only when the FF cortex confirms the new label (Lisman & Grace 2005 input-side mismatch).

## Update 2026-10-31 03:30 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: one-shot store remap gated by the FF cortex, both ways (`ff_remap.py`). Counting a conflict as full evidence
  when the cortex already agrees with the new label (f50k2) gives held-out swap 72.2 (= h2k2) with more false remaps;
  the cortex still says the old label right after the swap, so confirmation comes too late. Counting it when the
  cortex confidently predicts the old label (associative mismatch, Kumaran & Maguire 2007) gives 72.4. First remap
  latency (~50-75 ticks) is unchanged in every variant: the bottleneck is downstream of evidence (presence guard or
  K-vote), which is the next diagnostic.

## Update 2026-10-31 04:40 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Where the post-swap remap latency goes (`lat_diag.py`): 42-44% of swapped-digit arrivals land just outside the old
  prototype's radius (no evidence), 24-26% are in-radius but below K, 15-19% are vetoed by a25. The a25 vetoes come
  from a world property: the swap always opens the pair's season, so the 250-tick presence window still shows the
  previous season.
- Onset-aware presence (`onset_presence.py`, a25 on W = 250 OR on the last S ticks): S = 50 killed (8.5 false remaps
  at bg 0.2); S = 100 lead (held-out bg 0.02 swap 74.0 vs 72.2, latency -20 ticks, bg 0.2 +0.3 with 2.7 false
  remaps). Default stays a25w250h2k2 until s100 is confirmed on fresh seeds 6-8.

## Update 2026-10-31 05:30 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Onset presence s100 not confirmed on fresh seeds 6-8 (`onset_s678.py`): swap +0.1, latency -10 ticks. Closed; the
  04:40 gain was seed noise.
- The default store a25w250h2k2 is not safe on fresh seed 6: 34 pre-swap false remaps (seeds 0-5 had <= 4). All are
  ping-pong between co-present confusable classes (4<->9, 7<->9) in a season holding both, which the a25 presence
  guard cannot see because both classes are present.
- Rival-aware evidence (`rival_remap.py`): a conflict labelled y counts only if no y prototype is a close rival
  (squared distance <= 1.5 x that to the s prototype). Removes every false remap but costs 3.4 swap points, because
  the first true remap plants a y rival among the s prototypes. Radius-based rival test (rad) fails (seed 6 still 10).
- Rival test with directional unlock (`rival_unlock.py`, c1.5_u): after a remap s->y the pair (s, y) skips the rival
  test for 2000 ticks; (y, s) stays checked. 0 pre-swap remaps on all 15 runs (base 42 total), swap 71.7 vs 72.0
  (bg 0.02 seeds 3-8) and 71.5 vs 71.6 (bg 0.2), returns and lifetime equal. Lead for the new default; C and U were
  set after seeing seed 6, so confirm on fresh seeds 9-11 before adopting.

## Update 2026-10-31 06:10 UTC+8 (hourly loop; detail in `docs/loop.md`)
- The unlock window of the rival test is not a parameter (`rival_robust.py`): U = 500, 2000, 8000 and permanent give
  identical results, so a pair that has remapped once is simply never rival-tested again (c1.5_uinf).
- The rival ratio C is fragile: only 1.5 passes. C = 1.2 lets 4 false remaps through on seeds 4 and 6; C = 2.0 vetoes
  the first true remap (held-out swap -9.2); C = 3.0 never remaps. The margin between a confusable lure and a
  relabelled digit is narrow (~1.2-2x squared distance), so a fixed step threshold is unlikely to transfer. Next:
  graded (CA1-like) rival weighting, and a confusable swap pair (4<->9) as the decisive test. Fresh-seed (9-11) FF logs
  are running on the box.

## Update 2026-10-31 07:05 UTC+8 (hourly loop; detail in `docs/loop.md`)
- **New default store: a25w250h2k2_c1.5_uinf** (rival test at C = 1.5 with permanent directional pair unlock).
  Confirmed on fresh seeds 9-11 whose FF logs were made after C was chosen: 0 pre-swap false remaps (base 24, seed 10
  alone 20), swap 70.2 vs 71.1, returns and lifetime equal. Over all 18 runs: 0 false remaps vs 66 for the old default.
- Closed: graded (CA1-like) rival weighting (`rival_fresh.py`). It keeps 0 false remaps but delays the first true
  remap and loses 2-3 swap points (g1-2) or 11-16 (g1.2-3). The remap is an identity decision; a step per hit is
  better than a ramp. Next decisive test: swap a confusable pair (4<->9).

## Update 2026-10-31 08:05 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Confusable swap 4<->9 (`rival_s49.py A`, FF logs bg002s49 seeds 0-5): the fixed-ratio rival test (c1.5_uinf) loses
  3.4 swap points vs base (66.3 vs 69.7) and remaps 4x later, because a relabelled 4 sits next to the real 9
  prototypes. Closed as a general rule.
- **New default store: a25w250h2k2_c1.5_uinf_z5.** The rival veto is lifted when the prototype's last 5 history
  entries hold no same-label confirmation (extinction / latent-cause idea, Gershman, Blei & Niv 2010). 0 pre-swap
  false remaps on all 24 runs; swap equal to base on the 18 old runs and -1.6 on the confusable swap (c1.5_uinf
  -3.4). Next: replace the rival test by a confirmation trace alone, and a label-noise world as its kill test.

## Update 2026-10-31 09:05 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: confirmation-only remap veto (`conf_trace.py`, n3/n5/n8). 0 false remaps on all 24 runs, but swap falls 4-16
  points, monotone in the window, because vetoed evidence is discarded.
- Closed: deferred confirmation gate at N >= 5 (`conf_defer.py`). d3 matches base on the 18 old runs but loses 2.8 on
  the confusable swap 4<->9 (z5 loses 1.6). Default store stays a25w250h2k2_c1.5_uinf_z5. Likely cause: swapped 9s
  labelled 4 land inside 4 prototypes and count as confirmations. Next: core-only confirmation, label-noise world.

## Update 2026-10-31 10:05 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: core-only confirmation inside z5 (`core_conf.py`, q0.5 and q0.25). It closes the s49 swap gap (69.6 vs z5
  68.1) only by switching the rival test off, and false remaps return (seed 6: 20, seed 10: up to 28). The store
  radius is the mean confirmation distance, so median confirmation d/r is about 0.96 and fewer than 1% are in the core.
  Default stays a25w250h2k2_c1.5_uinf_z5. Next: confirmation weighted by relative similarity to the nearest rival.

## Update 2026-10-31 11:05 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: posterior-weighted (REMERGE-style) confirmation inside z5 (`post_conf.py`, p1.0 and p1.5). s49 swap 68.6 and
  68.5 vs z5 68.1, under the +0.8 line; false remaps stay <= 2 per run on all 24 runs.
- The 09:05 mechanism is now confirmed with origin tracking: after the swap, about half (54%) of the hits on
  not-yet-remapped 4/9 prototypes are impostor confirmations. But their relative-similarity weight (0.579) equals that
  of real members (0.589): in raw pixel space a member is only ~1.4x closer to its own prototype than to a rival, so no
  geometric confirmation rule can separate them. Closes all store-geometry confirmation rules.
- Default stays a25w250h2k2_c1.5_uinf_z5. Next: cortex-vetted confirmation (a same-label hit confirms only if the FF
  cortex also predicts that label), then a pattern-separated store space.

## Update 2026-10-31 12:05 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: cortex-vetted confirmation inside z5 (`cortex_conf.py`, rule cv). s49 swap 68.2 vs z5 68.1 (need +0.8);
  0 pre-swap false remaps on all 24 runs, old runs unchanged. Safe but useless.
- Why: the CLS premise "the slow cortex still knows a 9 is a 9" is false for the current never-sleeping FF at wake LR
  x1. On true 4s and 9s after the swap, FF already gives the new label 49-69% of the time in the first 300 ticks and
  82-90% in ticks 300-1500, so it agrees with the impostor and vetoes only 3.5% of post-swap confirmations.
  In this world the cortex is as fast as the store; pdp has no slow learner at the swap.
- Default stays a25w250h2k2_c1.5_uinf_z5. Running: FF logs with wake LR x0.1 on s49 seeds 0-5
  (`gate_probe.py --worlds bg002s49wl1`, results/gate_probe_s49wl1, launched 12:14 UTC+8 on the box) to retest cv
  with a genuinely slow cortex.

## Update 2026-10-31 13:00 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: cortex-vetted confirmation with a slow cortex (`cortex_conf.py C`, wake LR x0.1 logs). The cortex keeps the
  old name ~63% in ticks 0-300 and vetoes 13.5% of post-swap 4/9 confirmations, but s49 swap 49.0 vs z5 49.3 and first
  remap 76 vs 77 ticks. Impostor confirmations are not what delays remapping. The whole cortex-vetting line is closed.
- Closed: shorter extinction window (`z_window.py`). z3 +0.4, z2 +0.7 on s49 (bar +0.8), 0 false remaps on all 24 runs.
  Window explains ~2 of the 18 extra ticks vs base; the rival test on the first conflicts explains the rest.
- Finding: in the never-sleeping life world, wake LR x0.1 costs the combined system 2.5 lifetime (93.5 vs 96.0) and
  19 swap points. Slow waking helped earlier only with sleep. Default stays a25w250h2k2_c1.5_uinf_z5, wake LR x1.

## Update 2026-10-31 14:00 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: lifting the c1.5 rival veto by evidence accumulation inside z5: streak (`run_lift.py` r2/r3/r5), CUSUM
  (`cusum_lift.py`), and CUSUM with retroactive credit for vetoed conflicts (`retro_lift.py`). s49 swap 68.1-68.6 vs z5
  68.1 (bar +0.8), first post-swap remap 73-77 vs base 59; retro credit brings false remaps back on seed 1. Read with
  Lorden 1971: the ~18-tick latency is the cost of 0 false remaps for any detector on the same per-hit evidence. z5
  stays the default; next lever is per-hit evidence (pattern-separated store code).

## Update 2026-10-31 15:00 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: random-expansion pattern separation of the store (`kwta_store.py A`, 2000 units, k-WTA 5% and 2%). On s49
  first post-swap remap 139 / 1059 ticks vs z5 77, swap 67.2 vs 68.1, store-only lifetime -5 to -6, returns -14 in
  mq30_s2. A random projection without k-WTA is neutral (72 ticks, swap 67.8). Per Babadi & Sompolinsky 2014, random
  sparse expansion amplifies within-class spread as much as between-class distance. Default stays
  a25w250h2k2_c1.5_uinf_z5; next store-code test must use a learned (cortex) code.

## Update 2026-10-31 16:00 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed offline (`code_probe.py`, no life run spent): competitive-Hebbian expansion as store code (impostor rate on
  4/9 0.349 vs pixels 0.077) and prototypes stored in FF hidden-code space. FF layer codes separate 4 from 9 about 3x
  better than pixels (0.027-0.028), but frozen code prototypes go stale within 2000 swapped arrivals (layer 1 0.119,
  layer 2 0.619): FF learns the label inside its hidden code, so a store holding codes inherits the swap.
- Live: re-read index. Re-encoding stored images through the current cortex keeps separation (layer 1 0.045, 0.58x
  pixels). Next: in-loop life run with pixel prototypes and layer-1 distances; default stays a25w250h2k2_c1.5_uinf_z5.

## Update 2026-10-31 17:00 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: re-read index as a remap accelerator (`reread_store.py`, in-loop life run, s49 seeds 0-5). Pixel prototypes
  compared through the current FF layer 1 (rr1) or layers 1+2 (rr12): latency 71 / 65 vs z5 77, swap 67.7 / 68.3 vs
  68.1, and false remaps come back (2.7 / 2.3 per run vs 0). Layer 1 is not the better index. Default stays
  a25w250h2k2_c1.5_uinf_z5.
- Live: the cortex-read store is a better answerer (store-only lifetime 93.1-93.5 vs 91.4; mq30 returns 76.9 vs 72.0),
  but the route gate threshold must be re-set in code units (it fires 2.1% vs 1.1%, lifetime 95.5). Next: split store,
  remap in pixels, answer in code.

## Update 2026-10-31 18:00 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: cortex-read store as the answering route (`split_store.py`, pixel z5 remaps, answers read through FF layer 1
  or 1+2, gate threshold swept in code units). Held-out best 95.99 lifetime / 67.1 returns vs bar 96.0 / 74. At a
  matched firing rate code and pixel readers lie on the same lifetime/returns curve; the 17:00 +4.9 returns was the
  gate firing twice as often. Reading the store through the cortex is closed for remapping and answering.
- Default unchanged: a25w250h2k2_c1.5_uinf_z5, pixel store, mq30_s2. Next: measure complement accuracy (store right
  when FF is wrong) by time since return, and gate on that.

## Update 2026-10-31 19:00 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: complement accuracy as what varies after a return (`complement_probe.py`, s49 seeds 0-5): the store is
  right on FF errors 46-75% of the time in every bin and disagrees with a correct FF at a flat 6-10%; ratio 1.48 vs
  kill bar 1.5. What changes is the FF error rate (45.8% in the first 100 ticks after a return vs 3.4% after 3000).
- Lead: a causal recency gate (store also answers when it disagrees and its label or the FF label returned < 2000
  ticks ago, margin >= 0.2) gives held-out returns 73.8 vs 71.5 at 95.96 vs 95.97 lifetime, +1.6 returns above the
  mq frontier at matched firing. Not the default until confirmed on bg002 seeds 6-8 and bg 0.2.

## Update 2026-10-31 20:00 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Closed: recency as a frontier move (`confirm_recency.py`, s49 3-5, bg002 6-8, bg 0.2 0-5). The 19:00 lead held
  against the FF-margin (mq) frontier, but a recency-free control, c3 (store answers when it disagrees with relative
  margin >= 0.3, or on mq30_s2), beats mq30_s2 and rt2000_s2 on both lifetime and returns in every held-out group
  (s49 96.12 / 74.8, bg002 95.86 / 75.2, bg 0.2 lifetime 94.70 vs 94.38). Decaying and hyperbolic recency traces
  trade about 0.1 lifetime for 1.1-1.4 returns against c3 at matched firing: a slide, not an outward move.
- New default router candidate: c3. Swap first 300 is about 1 point lower than mq30_s2.

## Update 2026-10-31 21:00 UTC+8 (hourly loop; detail in `docs/loop.md`)
- Lead: calibrated arbitration (`arbiter.py`, offline exact join). Store answers on a disagreement when its running
  P(right) in a joint FF-margin x store-margin bin beats the FF's (Beta counts from arriving labels). Held-out
  j+0.0b: s49 3-5 96.18 / 77.8 vs c3 96.12 / 74.8 (+1.5 above the c frontier at matched firing), bg002 6-8
  95.90 / 78.2 vs 95.86 / 75.2 (+1.9), bg 0.2 lifetime 94.99 vs 94.70. Swap first 300 about 1 point lower.
- Router candidate moves from c3 to j+0.0b, pending a label-noise world and an in-loop run.
