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
