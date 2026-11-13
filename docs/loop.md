# Hourly research loop log

Append-only. Each entry: what was learned, what was killed, what to test next. Newest last.

## 2026-10-30 15:00 UTC+8

**Read first:** `results/hippo_route/RESULTS.md`, `results/life/RESULTS.md`.

### Findings
1. **Store-alone baseline (new, `hippo_probe.py`).** Nearest-prototype readout of the same
   ProtoStore, no network: MNIST 91.6 (9/class) / 94.2 (40/class) final class-IL, never
   forgets; Fashion 77.0 / 80.1. FF + sleep on the same store: 89.3 / 91.0 MNIST, 78.48 Fashion
   best. **The cortex is at or below its own hippocampus.** Store is hopeless at the label
   swap (3–14%) where FF adapts in 500 examples.
2. **Life headroom bound (from `results/life`, bg 0.2, 16 days).** Sleep can only improve
   off-season answers: max gain = off_share × (1 − off_season acc) = 0.134 × 0.185 ≈ 2.5
   points of lifetime accuracy. Clock sleep pays 8.7 in misses, so it loses by construction;
   circadian pays 1.0 and gained **0.0** off-season (81.6 vs 81.6). Pressure / two-process
   triggers never fired (0 sleeps). The 20% background is itself interleaved replay, done by the
   world. Lowering bg doesn't fix it: forgotten classes become rarer at the same rate they are
   forgotten (bg 0.02 smoke: never-sleep 92.6 vs circadian 92.1). Prequential lifetime accuracy
   weights memories by how often they're queried, and the same frequency rehearses them.
3. **Varied replay (`results/varied`, finals only; anytime/recent share lost to the Mac crash,
   VPS rerun in progress).** Under awake replay 0.8: 83.0 → 85.9 (jitter 2 px) → 88.7 (40
   protos). Without awake replay: jitter 90.2, 40 protos 91.0. Supports the memorisation
   hypothesis on the final score; anytime + recent share still needed.
4. **Slow waking is not free in a life.** Life: wake lr ×0.3 costs −1.0 lifetime, novel
   first-300 74 → 61, swap first-300 78 → 54 (two-process). Audit item 2 (make ×0.3 the
   default) should wait.

### Literature insight
CLS (McClelland, McNaughton & O'Reilly 1995) has the hippocampus **answer** recent items
directly while the neocortex learns slowly; replay is only half its job. Kumaran, Hassabis &
McClelland 2016 (TICS, "What learning systems do intelligent agents need?") add that replay
should be weighted by relevance, not uniform. pdp only uses the store to train the net, never to
respond, which is why slow waking looks costly in the life (finding 4) and why the net trails
its own store (finding 1). In our system the store is the *stable* side and FF the *plastic*
side, the reverse of the textbook assignment.

### Paths closed / demoted
- "FF + sleep reaches 90% class-IL" as evidence for the *network*: demoted. Must be reported
  against store-alone at equal memory from now on.
- Tuning sleep triggers on the current life world: closed until the world gives memory value
  (finding 2). Prediction (a) of the audit can't be decided there.

### Hypotheses, ranked
1. **Architecture (dual route), test next.** Answer with the store when its nearest-prototype
   margin is high or the class is novel; answer with FF otherwise; a per-class running
   agreement score arbitrates. Prediction: lifetime ≥ never-sleep 94.1 *and* Split-MNIST final
   ≥ 91.6 (store-alone), with swap adaptation ≥ 85. Kill: if the router can't beat the better
   of its two routes by ≥0.5.
2. **Data (world).** Give memory value: long absences (bg ≤ 0.02, seasons of 1–2 digits) plus a
   "returns" score on the first 50 inputs of a digit back after ≥ 10k ticks, reported
   separately from lifetime. Prediction: night sleep beats never-sleep on returns by ≥ 10.
   (life.py already has a RET_GAP returns counter; use it.)
3. **Training algorithm.** Replace replay-to-distil with replay of the store's *errors relative
   to the net*: replay only prototypes the net misclassifies (cheap, uses the awake probe).
   Prediction: same final at ≤ 0.3× replay. Kill: final drops > 1.
4. **Neuron.** Two-compartment FF unit whose apical input is the store's retrieved label
   (hippocampal context, not the true label). Prediction: anytime ≥ 85 with clock 2000.
   Blocked behind 1.
5. **Store adaptation.** Swap shows the store's 1/n rate freezes it. Let the cortex's confident
   disagreement (FF right, store wrong, repeatedly for a class) reset that slot's count. Ties to
   the open "change detector" direction.

## 2026-10-30 17:00 UTC+8

**Read first:** `results/dual_age_bg002/RESULTS.md` (box run, this hour), `results/dual_life_bg002`,
`results/dual_life`, `results/dual_age` (16:00 run, logged here for the first time). Split-MNIST
routing (`cls_route.py`) and `cls_life.py` (bg 0.2) were committed at 17:03 and are running separately
in `/workspace/pdp-run` on the box; not duplicated here.

### Findings
1. **Returns: the store answering beats sleep consolidating.** bg 0.02, first 20 arrivals of a digit
   back after ≥ 2000 ticks: FF never-sleep 53.5, FF night sleep 64.5 (+11, meets loop-15:00 hyp 2's
   ≥ 10 bar), **store alone 83.1**, oracle 88.0. Sleep recovers about a third of what the store
   already knows. Night sleep still loses lifetime (95.0 vs 95.9): returns are ~1% of arrivals.
2. **Reliability arbiter (per predicted class) is dead for returns.** Lifetime +0.0–0.2 at bg 0.2 and
   0.02, returns 51.6 (< FF 53.5). FF's off-season errors name *in-season* classes, where FF's
   running reliability is high, so the arbiter sides with the wrong route exactly when it matters.
3. **Age gate is the right signal, wrong gate.** Store answers on disagreement if its class has been
   away longer (≥ 500 ticks): returns 53.5 → 67.5 and off-season 42.8 → 60.8 with **no sleep at all**
   (beats night sleep on returns), but lifetime −2.8 (93.1) and novel first-300 74 → 64. The gate
   fires when FF names a just-arrived novel/current class and the store names a stale neighbour.
   At bg 0.2 it never fires (dual_age = FF), since every class recurs within 500 ticks.
4. Hyp 1 kill rule (router ≥ better route + 0.5 lifetime) not met by either arbiter. Oracle headroom
   is 1.9 lifetime points, 34.5 on returns.

### Literature insight
Daw, Niv & Dayan 2005 (Nat Neurosci 8:1704) arbitrate between two controllers by each one's
*current uncertainty*, not its past hit rate; Lengyel & Dayan 2007 (NIPS, "Hippocampal contributions
to control: the third way") show episodic (store-like) control wins when experience in a region is
sparse. Finding 2 is the failure Daw et al. predict for a hit-rate arbiter; finding 3 is Lengyel &
Dayan's regime (sparse recent experience = long absence). Their recipe implies the gate should
condition on how much *recent evidence backs FF's answer for this input*, not on class age alone.

### Paths closed / demoted
- Per-class reliability arbitration between FF and store: closed (two worlds, no gain, hurts returns).
- "Night sleep is how this system remembers returning digits": demoted. Answering from the store
  gets 83 where consolidation gets 64.5.

### Hypotheses, ranked
1. **Architecture, test next: uncertainty-gated store.** Age gate AND (FF goodness margin < its
   median on recent arrivals OR store relative margin ≥ 0.2), and never override FF on a class seen
   < 1000 labelled times (protects novel/swap). Prediction: returns ≥ 75, lifetime ≥ 95.9. Kill: lifetime
   < 95.5 at returns ≥ 70.
2. **Training algorithm: replay what the store gets right and FF gets wrong.** Use age-gate
   disagreements logged during the day as the night's replay set (not uniform prototypes).
   Prediction: night-sleep returns ≥ 70 (vs 64.5) at the same 80 batches. Kill: < 66.
3. **Data.** Returns are ~1% of arrivals, so lifetime can't reward memory. Add a world with bg 0 and
   seasons of 1 digit (returns ≈ 5% of arrivals). Prediction: store/age-gate wins lifetime there.
4. **Neuron.** Apical context = store's retrieved label only on gate-fired inputs (hyp 4 at 15:00,
   now scoped): FF learns from its own disagreements with memory. Blocked behind 1.

## 2026-10-30 17:27 UTC+8 — CLS complementary routing (hypothesis 1) tested

**Read:** `results/cls_route/VERDICT.md` (also `docs/cls_route.md`). Code: `cls_route.py`, `cls_life.py`.

- Targets: life ≥94.1, Split final ≥91.6, swap ≥85. **No router passes all three.**
  Best (gate0.02): life 94.21 ✓ (+0.18 over FF), Split 91.5 ✗ (−0.1), swap 88.0 ✓. Margin router
  m0.1: Split 92.6 ✓ but life 89.5 ✗, swap 66.3 ✗. Kill criterion (≥0.5 over the better route) met
  only by margin routing on Split. **Hypothesis 1 as stated: killed.**
- Falsified path: per-class stream reliability is biased in class-IL (answers naming absent classes are
  always wrong), and the awake prototype probe is contaminated by replay (FF fits exactly the replayed
  prototypes). Oracle headroom 3–4 points: the arbiter, not the routes, is the bottleneck.
- Side finding: FF's swap adaptation decays under clock sleep (95.9 @500 → 81.7 @8000) because sleep
  replays stale store labels. Promote hypothesis 5 (store change detector) to next.
## 2026-10-30 18:00 UTC+8

**Read first:** `results/gate_probe/RESULTS.md` (seeds 0–2, worlds bg 0.02 and ctx 2/bg 0) and
`results/gate_probe_s345/RESULTS.md` (held-out seeds 3–5, bg 0.02). Code: `gate_probe.py` (logs both
routes + their margins per arrival from one never-sleep life; every gate scored offline on the same
arrivals). Also see `docs/cls_route.md` (the 17:27 verdict on `cls_route`/`cls_life`): no router passes life + Split + swap at once; gate0.02 misses
Split by 0.1; stale store prototypes make FF drift back after the swap.

### Findings
1. **Hyp 1 (uncertainty-gated *age* store) killed.** Pre-registered gates at bg 0.02, seeds 0–2 / 3–5:
   ug_or 94.3 / 94.1 lifetime at returns 66.2 / 64.1; ug_and 95.8 / 95.8 at 62.3 / 61.0. Kill rule
   (lifetime < 95.5 at returns ≥ 70) met: no age-conditioned gate reaches 70 without losing lifetime.
2. **Why: class age is the wrong signal, not too weak a signal.** Of the age gate's firings (seed 0),
   1506/1762 are inputs FF already gets right, 1501 of them in season; the store is right on only 131.
   Among all disagreements, the AUC for "store right, FF wrong" is 0.80–0.88 for low FF goodness margin,
   0.68 for the store's relative margin, and **0.34–0.40 for store-class age** (worse than chance).
   The age gate fires because the store's nearest prototype often belongs to a stale class, which is
   exactly when it is wrong.
3. **Margin gate without age (post hoc, then held-out).** Store answers on disagreement when FF's margin
   is in its causal bottom 10% (last 500 arrivals) and the store's relative margin ≥ 0.2, FF's class
   protected below 1000 labels. Seeds 0–2: lifetime 96.0 (FF 95.9), returns 68.5 (FF 53.5). Held-out
   seeds 3–5: **lifetime 96.1 (FF 96.0), returns 65.1 (FF 53.6), off-season 57.9 (43.9), novel 77.6
   (76.3)**, store answers 0.8% of arrivals. First route that adds returns without losing lifetime, and
   with no sleep it matches night sleep's returns (64.5 at lifetime 95.0). The ≥ 0.1 variant gets
   returns 67.9 at 95.7. Neither reaches 70 on held-out seeds. Cost: swap first 300 drops 78.4 to 70.3,
   because the store is confidently wrong after a rule change (same failure as the Split verdict).
4. **Hyp 3 (returns-heavy world) failed as designed, not tested.** ctx 2 / bg 0 gave a *lower* return
   share (0.4% vs 0.9%): fewer, longer absences. Returns are counted per season change, so the knob that
   raises the return share is season length, not background rate.

### Literature insight
Mattar & Daw 2018 (Nat Neurosci 21:1609, "Prioritized memory access explains planning and hippocampal
replay") rank a memory for replay by gain × need: gain = how much replaying it would change the policy,
need = how soon that state is expected again. Finding 2 is the "need" half failing in our answering
gate (age is a poor proxy for whether the store is right), but for *replay* the decomposition maps
directly onto what we can now measure: gain = disagreements where FF's margin is low and the store's is
high (the mq10 events, ~1% of arrivals), need = the class's expected return time from the season
history. Uniform-prototype night replay ignores both, which is a candidate reason sleep recovers only
a third of what the store knows (64.5 vs 83).

### Paths closed / demoted
- Age-gated store answering, with or without uncertainty conditions: closed (hyp 1 killed, two seed sets).
- Class age as an arbitration feature: demoted (AUC < 0.5 among disagreements).
- "bg 0 / 2-digit seasons makes lifetime reward memory": closed as designed (return share falls).

### Hypotheses, ranked
1. **Training algorithm, test next: gain-prioritised night replay.** Log mq10-style events during the
   day (low FF margin, store confident, store ≠ FF) and replay the store prototypes nearest to them at
   night instead of uniform prototypes, same 80 batches. Prediction: night-sleep returns ≥ 70 (vs 64.5)
   and lifetime ≥ 95.5. Kill: returns < 66. (Supersedes 17:00 hyp 2, which used age-gate events.)
2. **Architecture: margin gate + change detector.** Repeated confident FF-vs-store disagreement *with
   the label siding with FF* resets that store slot's count (17:00 hyp 5) and vetoes the gate for that
   class. Prediction: swap first 300 back to ≥ 76 with returns ≥ 64 and lifetime ≥ 96.0. Kill: swap < 74.
3. **Data: short seasons.** season 1500 (vs 6000), bg 0.02. Prediction: return share ≥ 3% and the
   mq10_s2 gate beats FF on lifetime by ≥ 0.5. Kill: < 0.2.
4. **Neuron: margin as a dendritic plateau signal.** Use FF's low-margin events (not the label) to
   switch a unit's apical compartment to the store's retrieved label for that one update. Prediction:
   FF's own returns rise ≥ 5 with no change to answering. Blocked behind 1.
## 2026-10-30 19:40 UTC+8

**Read first:** `results/gain_replay/RESULTS.md` (box run, seeds 0–2, bg 0.02, 16 days). Code: `gain_replay.py`
(subclasses `life.Machine`; logs mq10_s2-style events by day, mixes their nearest store slots into night replay).
Control is the existing `night` row in `results/dual_life_bg002/RESULTS.md` (same seeds, same code path).

### Findings
1. **Hyp 1 (gain-prioritised night replay) killed.** Pre-registered kill was returns < 66 for every variant.
   | Night replay | Lifetime | Returns | Off season | Frozen final | Swap first 300 |
   |---|---|---|---|---|---|
   | uniform (control) | 95.0 | 64.5 | 52.1 | 89.1 | 76.4 |
   | 50% gain, unlabelled events | 95.0 | 63.7 ± 3.2 | 52.0 | 86.2 | 78.1 |
   | 50% gain, label-verified | 95.0 | 62.2 ± 0.3 | 50.7 | 87.5 | 78.6 |
   | 100% gain, label-verified | 94.9 | 58.2 ± 0.1 | 47.3 | 83.1 | 77.8 |
2. **Monotone in focus.** More replay on the gain set means worse returns, off-season and frozen accuracy; the
   swap moves up ~2 points but within noise. Gain events cluster on a few slots of recently confused classes,
   and replaying them displaces coverage of the classes that are away, which are the ones a return tests.
3. Lifetime is flat (95.0) under every replay mix: night replay cannot move lifetime in this world, only the
   ~1% of arrivals that are returns/off-season. The answering route (margin gate, 18:00) remains the better lever.

### Literature insight
Hasselmo 2006 (Curr Opin Neurobiol 16:710, "The role of acetylcholine in learning and memory"): high
acetylcholine during novelty/mismatch suppresses recurrent *retrieval* in hippocampus and cortex and boosts
*encoding*; low ACh (quiet waking, slow-wave sleep) lets retrieval and consolidation run. Our margin gate fails
the swap for exactly the reason this predicts: after a rule change the store keeps retrieving confidently while
labels contradict it. The ACh-style fix is a mismatch signal that turns *off* store answering for a class and
turns *up* store plasticity for its slots (fast overwrite), which is 18:00 hyp 2 with a concrete mechanism.
Paired with finding 2, Mattar & Daw's need term predicts that night replay should weight classes by time away.

### Paths closed / demoted
- Gain-prioritised night replay (any mix ≥ 50%): closed.
- Night-replay composition as a lifetime lever in the bg 0.02 world: demoted (lifetime flat across all mixes).

### Hypotheses, ranked
1. **Training algorithm, test next: need-prioritised night replay.** Sample replay classes with probability
   ∝ ticks since last seen (floor at 1 day), uniform within class. Prediction: returns ≥ 68 and off-season ≥ 56
   at lifetime ≥ 94.9. Kill: returns < 65.5.
2. **Neuron/architecture: ACh-style mismatch veto.** Count, per store class, label arrivals where the store's
   confident claim (rel. margin ≥ 0.2) is wrong; after 3 in the last 50, veto the margin gate for that class and
   set that class's prototype lr to 0.5 until 20 consecutive agreements. Prediction: swap first 300 ≥ 76 with
   returns ≥ 64 and lifetime ≥ 96.0 (offline on `gate_probe` logs plus a store re-run). Kill: swap < 74.
3. **Data: short seasons** (season 1500, bg 0.02): unchanged from 18:00; return share should rise ≥ 3%.
   Prediction: mq10_s2 beats FF on lifetime by ≥ 0.5. Kill: < 0.2.
4. **Architecture: margin gate + need replay together.** Only if 1 passes: predicted returns ≥ 70 at lifetime ≥ 96.

## 2026-10-30 20:25 UTC+8

**Read first:** `results/need_replay/RESULTS.md` (box run, seeds 0–2, bg 0.02, 16 days). Code: `need_replay.py`
(subclasses `life.Machine`; night replay draws classes ∝ max(ticks since last seen, floor)^p). Control is the
`night` row in `results/dual_life_bg002/RESULTS.md`. The first run (`results/need_replay_floor4000/`) used the
pre-registered 1-day floor and is degenerate: at bg 0.02 every digit is revisited by background arrivals about
every 500 ticks, so no class is ever 4000 ticks away at night and the weights were all equal. Rerun with floor 100.

### Findings
1. **Hyp 1 (need-prioritised night replay) killed.** Kill was returns < 65.5.
   | Night replay | Weight ratio max/min (median night) | Lifetime | Returns | Off season | Frozen final | Swap first 300 |
   |---|---|---|---|---|---|---|
   | uniform (control) | 1 | 95.0 | 64.5 | 52.1 | 89.1 | 76.4 |
   | need, p=1, floor 100 | ~11 | 95.0 | 64.2 ± 1.7 | 52.4 | 87.1 | 77.4 |
   | need, p=0.5, floor 100 | ~3 | 95.0 | 64.2 ± 2.4 | 52.1 | 89.8 | 77.1 |
2. Together with 19:40: focusing night replay on *gain* hurts returns, focusing it on *need* does nothing, and
   lifetime is 95.0 under all five mixes. **Night-replay composition is closed as a lever in this world.** What
   decides a return is what happens between the night and the return (wake updates on in-season digits), and
   the answering route, not which prototypes were rehearsed.
3. Design note: "time away" is a weak signal whenever background arrivals exist; returns (≥ 2000 ticks away) are
   ~1% of arrivals and come mostly from gaps between background visits, not from seasons.

### Literature insight
McClelland, McNaughton & Lampinen 2020 (Phil Trans R Soc B 375:20190637, "Integration of new information in
memory: new insights from a complementary learning systems perspective"): full interleaving is not needed to
avoid interference; interleaving *only old items similar to the new one, during the learning of the new one*
(similarity-weighted interleaved learning) gets nearly all of the protection at a fraction of the replay. Two
consequences for pdp: (a) protection belongs at wake time, when the interfering update happens, which is consistent
with finding 2 (offline composition cannot undo the day's interference); (b) the right weight is similarity to
the *current input*, which neither gain (confusion events) nor need (time away) measures.

### Paths closed / demoted
- Need-prioritised night replay (p = 0.5, 1): closed.
- Night-replay composition (uniform / gain / need) as a lever for returns or lifetime at bg 0.02: closed.

### Hypotheses, ranked
1. **Neuron/architecture, test next: ACh-style mismatch veto** (19:40 hyp 2, unchanged). Per store class, count
   arrivals where the store's confident claim (rel. margin ≥ 0.2) is wrong; after 3 in the last 50, veto the
   margin gate for that class and set its prototype lr to 0.5 until 20 consecutive agreements. Prediction: swap
   first 300 ≥ 76 with returns ≥ 64 and lifetime ≥ 96.0. Kill: swap < 74.
2. **Training algorithm: similarity-weighted awake interleaving (SWIL).** At each wake update, add one store
   sample drawn from the nearest-prototype class ≠ y (by distance to the input), versus a control of one uniform
   store sample; night sleep unchanged. Prediction: FF returns ≥ 68 and off season ≥ 56 at lifetime ≥ 95.0, and
   beating the uniform 1-sample control by ≥ 2 on returns. Kill: < 1 point over the uniform control.
3. **Data: short seasons** (season 1500, bg 0.02): unchanged. Prediction: mq10_s2 beats FF on lifetime by ≥ 0.5.
4. **Neuron: margin as a dendritic plateau signal:** unchanged, behind 1.

## 2026-10-30 21:10 UTC+8

**Read first:** `results/veto_probe/RESULTS.md` (offline on the logged arrivals of `results/gate_probe` seeds 0-2
and `results/gate_probe_s345` held-out seeds 3-5, bg 0.02). Code: `veto_probe.py`, `veto_sweep.py`.

### Findings
1. **Hyp 1 (ACh-style mismatch veto on the margin gate) fails as pre-registered.** Not killed on swap (swap
   recovers to FF level, 76.8 / 78.4), but returns drop from 65.1 to 56.1 on held-out seeds, below the 64 floor.
   Lifetime 96.2, i.e. the veto mostly returns the machine to plain FF.
2. Sweep of the trigger count (3 to 20 wrong of 50 confident claims) is a monotone trade-off; no setting passes
   both swap ≥ 76 and returns ≥ 64 on held-out seeds. The store's base confident-claim error is 1.3-1.4%, so
   this is not base-rate noise: the confident errors that trip the veto co-occur with the claims that win returns.
3. Design lesson: vetoing *answers* by claimed class throws away the store's value; the swap failure is a stale
   *label* on specific prototypes, which is a store-update problem, not a routing problem.

### Literature insight
Hasselmo 2006 (Curr Opin Neurobiol 16:710, "The role of acetylcholine in learning and memory"): high ACh does two
things at once, it suppresses recurrent recall (the half we tested, and it costs returns) and it boosts
afferent-driven encoding so the new association overwrites. The second half is what the swap needs, and it is
local to the mismatching memory rather than class-wide. Kumaran, Hassabis & McClelland 2016 (TiCS 20:512) make the
same point for CLS: the hippocampus should rapidly encode prediction-error-laden episodes, not stop being consulted.

### Paths closed / demoted
- Answer veto keyed on the store's claimed class (any trigger count 3-20 of 50): closed.

### Hypotheses, ranked
1. **Neuron/architecture, test next: label-conflict remap of store prototypes.** When a labelled arrival's
   nearest prototype has a different label and is within that prototype's typical radius, count a conflict on that
   prototype; after 3 conflicts in its last 10 hits, relabel it to the majority recent label (or reset it to the
   new input). Margin gate unchanged, no veto. Needs a store re-run (gate_probe with remap). Prediction: swap first
   300 ≥ 76 with returns ≥ 64 and lifetime ≥ 96.0 on held-out seeds 3-5. Kill: returns < 62 or swap < 74.
2. **Training algorithm: similarity-weighted awake interleaving (SWIL)** (20:25 hyp 2, unchanged).
3. **Data: short seasons** (season 1500, bg 0.02): unchanged.
4. **Neuron: margin as a dendritic plateau signal:** unchanged.

## 2026-10-30 22:15 UTC+8

**Read first:** `results/remap_probe/RESULTS.md`. Code: `remap_probe.py`. Method: with no sleep and no mix the store
never touches FF learning, so the world stream is replayed through the store alone (about 30 s per seed instead of
10 min) and joined with the FF answers logged by `gate_probe.py`. Sanity check passes: the plain store reproduces
the logged store answers exactly on all six seeds.

### Findings
1. **Hyp 1 (per-prototype label-conflict remap, K = 3 of last 10) killed by its own rule** on held-out seeds 3-5:
   mq10_s2 returns 61.9 (kill < 62), swap 76.1, lifetime 96.2. K = 2 gets swap 78.7 but returns 60.3; K = 5 gets
   returns 63.8 but swap 73.4. Same monotone trade-off the answer veto showed at 21:10.
2. **Where it goes wrong:** about half of all remaps happen *before* the swap (70 of 152 at K = 3), so they are all
   false, while the swap pair itself needs only about 17. A "flip" rule (conflicts and at most 1 agreement in the
   last 10) still makes 41 false pre-swap remaps; a coordinated class-level rule (M = 3 prototypes of one class hot
   toward the same label, pre-registered as p3) still makes 36 and lands at swap 75.7, returns 62.8: it misses the
   prediction (pre-swap remaps <= 10, swap >= 76, returns >= 64) without hitting the kill. Demoted.
3. **Side finding worth keeping:** any remap lifts the store *alone* from 85.0 to about 92 lifetime, so the stale
   swap labels were about 7 points of the store's lifetime deficit. Under the margin gate this barely matters
   (store answers 0.8% of arrivals), which is why routing metrics hardly move.
4. Probable mechanism for the false remaps (not yet tested): while a class is away, its border prototypes are hit
   only by the present classes' labels and look exactly like stale labels; remapping them steals the away class's
   memory, which is precisely what returns measure.

### Literature insight
Gershman, Monfils, Norman & Niv 2017 (eLife 6:e23763, "The computational nature of memory modification") and
Gershman, Blei & Niv 2010 (Psych Rev 117:197): whether an old memory is *modified* or a *new* memory is formed
depends on inferring whether the same latent cause is still active. Under that view a prototype should only be
rewritten when its own cause is evidently present and now disagrees, and an *absent* cause's memory must be
protected from modification. Our remap rules modify memories of absent causes, which is the failure in finding 4.

### Paths closed / demoted
- Per-prototype label-conflict remap (K = 2, 3, 5) and the flip variant: closed.
- Coordinated class-level remap (p2-p4): demoted, not killed.

### Hypotheses, ranked
1. **Architecture/neuron, test next: presence-gated remap.** Same p3 rule, but a prototype of class s may only be
   remapped when class s itself had a correctly-labelled arrival within the last 200 ticks (its cause is active).
   Prediction on held-out seeds: pre-swap remaps <= 10, swap >= 76, returns >= 64, lifetime >= 96.0.
   Kill: pre-swap remaps > 20 or returns < 63. Cheap (store-only replay, about 1 min).
2. **Training algorithm: SWIL** (similarity-weighted awake interleaving), unchanged from 20:25; needs FF re-runs
   (about 10 min per seed), so run it on the box in the background next hour.
3. **Data: short seasons** (season 1500, bg 0.02): unchanged.
4. **Neuron: margin as a dendritic plateau signal:** unchanged.

## 2026-10-30 23:05 UTC+8

**Read first:** `results/presence_remap/RESULTS.md`. Code: `remap_probe.py` (new rules `gM`, `rM`), run with
`python3 remap_probe.py --Ks none p3 g3 r2 r3 r4 --out results/presence_remap` (about 1 min on the box).

### Findings
1. **Hyp 1 as written (presence = label s seen in the last 200 ticks, g3) killed:** held-out pre-swap remaps 27
   (kill > 20), returns 63.0, swap 76.6. The gate only delays remaps: hot prototypes stay hot and fire the moment a
   single background arrival of s shows up (bg 0.02 gives each class about one arrival per 500 ticks).
2. **Diagnostic confirms the 22:15 mechanism:** for p3 on held-out seeds, 90 of 108 pre-swap remaps move a class
   that is *out of season* toward an in-season label (pairs 4/9, 7/9, 3/8, 5/8, 2/3). The away class's border
   memories are being stolen; the mechanism was right, the presence test was wrong.
3. **Rate presence works (pre-registered after finding 2, before running):** r3 = p3 plus "label s arrived >= 5 times
   in the last 500 ticks". Held-out seeds 3-5 under mq10_s2: pre-swap remaps 0, swap 76.1, returns 65.8, lifetime
   96.2. Prediction (pre-swap <= 10, swap >= 76, returns >= 64, lifetime >= 96.0) **passes**. Seen seeds 0-2: 3 false
   remaps, swap 73.2 vs 67.0 plain, returns 68.9 vs 68.5. r2 and r4 are within noise of r3 (not a knife edge in M).
4. **The store alone is now a much better memory:** held-out lifetime 91.6 (plain 85.0), returns 82.0 (80.1),
   swap 60.2 (2.8). Under the margin gate the store answers only 0.7% of arrivals, so routing gains are small;
   the bigger payoff is that the store can now be trusted more often (hyp 1 below).
5. Caveat: the 5-in-500 threshold uses knowledge of the bg 0.02 world (background gives ~1 per class per 500
   ticks). It must be made adaptive or shown robust at bg 0.2 before it counts as a mechanism.

### Literature insight
Anderson & Schooler 1991 (Psychological Science 2:396, "Reflections of the environment in memory"): the odds that
a memory is needed track how *often* it was encountered in a recent window, not just how recently it was last
seen. Our g3 vs r3 contrast is exactly that distinction: a single recent sighting is a poor estimate that a latent
cause is active, a recent *rate* is a good one. Combined with Gershman et al. 2017 (modify only memories whose
cause is active), the store's remap rule is now "rewrite a memory only when its cause is frequent now and
disagrees", which is the first remap rule to pass.

### Paths closed / demoted
- Recency presence gate (gM, W = 200): closed.
- Coordinated remap without presence (pM): superseded by rM.

### Hypotheses, ranked
1. **Architecture, test next (cheap, offline): loosen the margin gate with the r3 store.** With stale labels fixed
   the store may be trusted more: sweep the FF-margin quantile 10/20/30 and store-margin 0.1/0.2 with r3.
   Prediction: some setting reaches returns >= 68 at lifetime >= 96.0 on held-out seeds. Kill: no setting beats
   mq10_s2+r3 returns by >= 2 at lifetime >= 96.0.
2. **Data/neuron: robustness of rate presence at bg 0.2.** Background then gives about 10 arrivals per class per
   500 ticks, so a fixed 5-in-500 gate should fail. Adaptive version: presence = recent count >= 3x the class's
   long-run background rate (EMA). Prediction: adaptive rule keeps pre-swap remaps <= 10 at both bg 0.02 and 0.2;
   fixed r3 exceeds 20 at bg 0.2. Needs gate_probe logs for the bg 0.2 world (check `raw[3:]`).
3. **Training algorithm: SWIL** (unchanged, needs FF re-runs, run on the box in background).
4. **Data: short seasons** (season 1500, bg 0.02): unchanged; now also a stress test for rate presence.

## 2026-10-31 00:10-00:35 HKT (UTC+8) loop

### Experiments (both offline store replays joined with gate_probe FF logs)
1. **Loosened margin gate with the r3 store (`loose_gate.py`, results/loose_gate): prediction passes, modestly.**
   Selected on seeds 0-2 (best returns at lifetime >= 96.0: mq50_s2), tested on held-out 3-5: lifetime 96.1,
   returns 68.7 vs 65.8 for mq10_s2+r3 (+2.9, kill threshold was +2), swap 71.9 vs 76.1 (-4.2). mq30_s2 is the
   gentler point (returns 68.3, swap 73.2). The r3 store is what makes loosening affordable: with the plain store,
   mq50_s2 costs lifetime (95.1); with r3 it holds 96.1. The store-margin threshold s carries the lifetime; the FF
   quantile mainly trades returns for swap. s1 settings all break lifetime 96.0.
2. **Adaptive presence (`presence_bg.py`, results/presence_bg, new FF logs results/gate_probe_bg02 for bg 0.2,
   seeds 0-5): prediction passes.** Fixed r3 fails at bg 0.2 as predicted: 32 pre-swap remaps (bg 0.02: 2).
   Scale-free a25 (s present iff its count in the last 500 ticks >= 25% of the most frequent label's count)
   gives 0 / 1 pre-swap remaps at bg 0.02 / 0.2, returns 67.5 vs r3 67.4 at bg 0.02, and mq10_s2 swap 72.6 / 71.8
   vs plain store 68.7 / 66.2. a10 is too loose at bg 0.2 (17 false), a40 slightly over-blocks swap. Cost: a25 swap
   is ~2 below r3 at bg 0.02 (72.6 vs 74.7). Note bg 0.2 has no "returns" (every digit recurs within 2000 ticks),
   and the store adds only 0.1-0.3 lifetime there under mq10_s2.
   **Decision: a25 replaces r3 as the default remap gate** (no knowledge of the background rate).

### Literature insight
Gallistel & Gibbon 2000 (Psychological Review 107:289, "Time, rate, and conditioning"): conditioning tracks the
*ratio* of the cue's event rate to the background rate, which makes acquisition invariant to time scale. a25 is the
same move for latent-cause presence: the fixed count r3 broke as soon as the background rate changed tenfold, the
ratio to the dominant cause did not. Prediction from the theory: a25 should also survive changes in arrival rate
(night_rate, day length) and window length, which a fixed count would not.

### Paths closed / demoted
- Fixed-count rate presence (r3) as a mechanism: demoted (fails at bg 0.2); kept as the bg 0.02 reference.

### Hypotheses, ranked
1. **Architecture (cheap, offline): a25 store + loosened gate (mq30_s2, mq50_s2) in both worlds.** Prediction:
   held-out bg 0.02 returns >= 68 at lifetime >= 96.0 and bg 0.2 lifetime >= 94.5 (mq10_s2 is 94.4). Kill: bg 0.2
   lifetime falls below FF (94.1) at any setting that gains >= 2 returns at bg 0.02.
2. **Data: short seasons (season 1500, bg 0.02) and a halved window (250).** Stress test for a25's
   ratio rule; prediction from rate theory: pre-swap remaps stay <= 10 with W=250 as well as 500.
3. **Neuron: per-prototype presence.** Gate each hot prototype by its own recent agreeing-hit rate vs its class's
   rate (a dendrite-local version of a25). Prediction: recovers the ~2 swap points a25 loses vs r3 while keeping
   false remaps <= 10. Kill: swap does not rise above a25's 72.6 at bg 0.02.
4. **Training algorithm: SWIL** (unchanged, needs FF re-runs; box in background).

## 2026-10-31 01:05-01:30 HKT (UTC+8) loop

### Experiments (offline store replays joined with gate_probe FF logs; ~2 min total on the box)
1. **a25 store + loosened gate in both worlds (`a25_gate.py`, results/a25_gate): half passes, not killed.**
   Held-out bg 0.02 (seeds 3-5): a25 + mq30_s2 lifetime 96.1, returns 68.4, swap 69.3; mq50_s2 96.0 / 68.8 / 68.4
   (prediction >= 68 at >= 96.0: pass). bg 0.2 (seeds 0-5, never used for selection): lifetime 94.4 / 94.3, the same
   as mq10_s2 (94.4) and above FF (94.1), so the kill does not trigger, but the >= 94.5 half of the prediction fails:
   loosening is free at bg 0.2, not a gain (there are no long absences to cover).
   Window: a25w250 keeps pre-swap remaps 0 (bg 0.02 held out) / 1 (bg 0.2) and *raises* swap by 1-2 points
   (mq30_s2: 71.2 vs 69.3 at bg 0.02, 70.2 vs 69.3 at bg 0.2) at equal returns. Rate theory's window-invariance
   prediction holds. **Decision: default = a25w250 + mq30_s2.**
2. **Change-point veto (`cp_veto.py`, results/cp_veto): killed.** Per store class, veto when the last-10 confident
   error rate is >= 0.5 and >= 4x its long-run EMA. Held-out bg 0.02 mq30_s2: swap 71.2 -> 76.2 (pass) but returns
   68.3 -> 60.5 (kill < 67); the old absolute veto gives 78.2 / 57.4. Pre-swap vetoed answers: 3.6% at bg 0.02 vs
   0.9% at bg 0.2. The extra false vetoes in the world with long absences say the trigger is the class *returning*.

### Literature insight
Yu & Dayan 2005 (Neuron 46:681, "Uncertainty, neuromodulation, and attention"): ACh reports expected uncertainty,
NE reports unexpected uncertainty, i.e. a change of context the current model did not predict. The cp veto was the
NE half (the 21:10 veto was the ACh half). It failed for a reason the theory names: "unexpected" is relative to a
model, and the store has no model of its own absence. A class coming back after days produces the same error burst
as a relabelling, so a scalar surprise signal cannot tell "the world changed" from "I have been away". Whatever
detects the swap must use *what* the conflicting labels are (a consistent new label for a familiar input, which
the remap rule already uses) rather than *how often* the store is wrong. So the swap cost should be attacked
through remap speed, not answer vetoes.

### Paths closed / demoted
- All store-error-rate vetoes (absolute 21:10, change-point 01:05): closed.
- a25 at W = 500: superseded by W = 250.

### Hypotheses, ranked
1. **Training algorithm (cheap, offline, test next): faster remap under the a25 guard.** The swap cost is remap
   latency (2 in-radius conflicting hits per prototype, >= 3 hot prototypes). With a25 blocking absent-cause
   remaps, lower to 1 hit and K = 2. Prediction: held-out bg 0.02 swap >= 74 under mq30_s2 with returns >= 67.5 and
   pre-swap remaps <= 5 in both worlds. Kill: pre-swap remaps > 10 in either world or returns < 67.
2. **Data (diagnostic, cheap): returns vs swap signature.** Prediction: >= 70% of pre-swap cp-vetoed answers at bg
   0.02 fall within 300 ticks of the vetoed class's season start. Confirms the mechanism behind today's kill.
3. **Neuron: per-prototype presence** (from 00:10, unchanged): recover swap by gating each hot prototype on its own
   agreeing-hit rate. Kill: swap not above a25w250's 71.2.
4. **Data: short seasons (1500, bg 0.02)** as a stress test for a25w250; needs new FF logs (box, background).
5. **Training algorithm: SWIL** (unchanged; needs FF re-runs).

## 2026-10-31 02:10-02:35 HKT (UTC+8) loop

### Experiments (offline store replays joined with gate_probe FF logs; ~3.5 min on the box, VPS idle)
1. **Faster remap under the a25 guard (`fast_remap.py`, results/fast_remap): test h1k2 killed on false remaps.**
   hHkK = a prototype is hot after H in-radius conflicts toward y (of its last 10 hits), remap when >= K are hot.
   Held-out bg 0.02, mq30_s2: h2k3 (default) 96.1 / 68.3 / swap 71.2, 0 pre-swap remaps; **h1k2 96.0 / 68.0 / 75.8
   but 61 pre-swap remaps (bg 0.2: 67)**; h1k3 96.0 / 68.4 / 74.7 with 16 (bg 0.2: 7); h2k2 96.1 / 68.6 / 72.2 with
   1 (bg 0.2: 0, swap 71.6 vs 70.2). Swap latency is set by H (evidence per prototype), not K (prototypes needed).
   The false remaps under H = 1 cost almost nothing on lifetime/returns at this horizon (the FF route covers), but
   they are wrong memories, so the pre-registered kill stands. **Default moves to h2k2** (+1.0 / +1.4 swap, equal
   returns, 1 / 0 false remaps): small but free.
2. **Distance-weighted evidence (`close_remap.py`, results/close_remap): test e50k2 killed.** A conflict deep in a
   prototype's radius (d <= c r) counts double. c = 0.5 is *identical* to h2k2 (deep conflicts never occur: swapped
   inputs sit near the radius edge, they are border cases of the old prototype, not its centre). c = 0.7: held-out
   bg 0.02 swap 73.9 (just under the 74 bar), returns 68.5, 1 pre-swap remap; seeds 0-2 72.0 with 2; bg 0.2 71.9.
   e70k2 is a lead, not a result: it was read on the held-out seeds, so it needs fresh seeds (6-8) before adoption.

### Literature insight
Lisman & Grace 2005 (Neuron 46:703, "The hippocampal-VTA loop"): CA1 compares what CA3 recalls with what the
cortex delivers; a mismatch (novelty) signal drives dopamine release that gates LTP, so one strong mismatch can
write a memory in one shot. Our store tested exactly that: a single conflict suffices when it is "strong", with
strength = depth inside the prototype's basin. The test says depth is the wrong strength axis here: relabelled
digits land at the rim of the old prototype's basin, the same place confusable borderline digits land, so one
rim conflict is ambiguous no matter how it is weighted. In the Lisman-Grace loop the mismatch is between recall
and *input*; ours compares recall with the *label*. A one-shot signal needs an input-side mismatch too (the input
is familiar to the cortex, i.e. high FF goodness/margin, yet the label disagrees), which the FF margin already
measures. That is the next test.

### Paths closed / demoted
- Single-hit remap (H = 1), with or without distance weighting at c <= 0.5: closed (false remaps).
- Default store: a25w250h2k2 (was h2k3).

### Hypotheses, ranked
1. **Training algorithm (cheap, offline, test next): FF-confirmed one-shot remap.** Count a single conflict as
   full evidence (H = 1) only when the FF cortex is confident in the *new* label on that input (FF margin >= its
   rolling q50 and FF argmax == y); else need H = 2. Needs the FF row joined per tick (already in gate_probe logs).
   Prediction: held-out bg 0.02 swap >= 74, pre-swap remaps <= 5 in both worlds. Kill: > 10 false remaps or swap < 73.
   Caveat: right after the swap the FF cortex still says the old label, so this may never fire; that would kill it
   cleanly and say the swap needs the store to lead the cortex.
2. **Neuron: e70k2 on fresh seeds.** Needs FF logs for bg 0.02 seeds 6-8 (gate_probe, ~box background). Prediction:
   swap >= 73 with <= 3 false remaps. Kill: swap gain over h2k2 < 1.
3. **Data: returns-vs-swap signature diagnostic** (from 01:05, unchanged).
4. **Architecture: let the router see remap events.** For 300 ticks after a class's prototypes remap, route that
   class to the store regardless of margin. Prediction: swap +2 over h2k2 at no returns cost; kill if lifetime < 96.0.
5. **Training: SWIL** (unchanged; needs FF re-runs).

## 2026-10-31 03:10-03:30 UTC+8 loop

### Experiments (offline store replays joined per arrival with gate_probe FF rows; ~5 min each on the box, VPS idle)
1. **FF-confirmed one-shot remap (`ff_remap.py`, results/ff_remap): test f50k2 killed (swap < 73).** A conflict
   counts as full evidence when the FF cortex already says the *new* label (argmax == y, margin >= rolling q50).
   Held-out bg 0.02 mq30_s2: f50k2 96.1 / 68.6 / swap 72.2 with 5 pre-swap remaps (baseline h2k2 identical swap
   72.2 with 1); bg 0.2 swap 71.2 with 6. Without the margin condition (f0k2) false remaps jump to 21-28 per run.
   First post-swap remap latency is unchanged (59/73/52 ticks vs 64/73/52): right after the swap the cortex still
   says the old label, so the confirmation never arrives when it is needed. The pre-registered caveat came true:
   the store has to lead the cortex on a swap, it cannot wait for it.
2. **Associative-mismatch one-shot remap (follow-up, results/ff_remap_v): v50k2 killed.** Opposite signal: a conflict
   counts double when the cortex confidently predicts the store's *old* class (familiar item, new pairing; Kumaran &
   Maguire 2007). Held-out swap 72.4 (v70k2 72.3) with 4-5 pre-swap remaps; bg 0.2 71.7 with 2. Same latency.
   Neither cortex-side signal moves the first post-swap remap, so the remap latency floor (~50-75 ticks) is not set
   by per-prototype evidence alone.

### Literature insight
Kumaran & Maguire 2007 (J Neurosci 27:8517, "Match-mismatch processes underlie human hippocampal responses to
associative novelty") and Lisman & Grace 2005: the hippocampal novelty signal fires for familiar items in a new
configuration *only when a prediction was made and violated*. Both versions of that signal are available here (FF
agrees with the new label; FF confidently predicts the old one) and neither changes when the first remap happens.
That points the latency at the gating that sits *after* evidence: the a25 presence check on the old class and the
K-of-n hot-prototype vote, which in the brain would be the slower neuromodulatory permission (VTA/NE) rather than
the fast CA1 comparison.

### Paths closed
- One-shot remap keyed on cortex confirmation of the new label (f*), or on violated confident cortex prediction (v*):
  closed. Default store stays a25w250h2k2.

### Hypotheses, ranked
1. **Architecture / diagnostic (cheap, test next): where the ~60-tick remap latency goes.** Log, per post-swap
   conflict, whether it was in-radius, whether it made a prototype hot, and whether a25 blocked the remap.
   Prediction: >= 50% of the post-swap latency is spent with >= K hot prototypes blocked by a25 (old class's ratio
   drops as its label stops arriving). Kill: a25 blocks 0 post-swap remaps, which would put the floor on in-radius
   hits (swapped digits outside the old radius) instead.
2. **Architecture: let the router see remap events** (from 02:35, unchanged). 300 ticks of store routing for a class
   after its prototypes remap. Prediction: swap +2 over h2k2, lifetime >= 96.0.
3. **Neuron: e70k2 on fresh seeds 6-8** (needs new FF logs; box background run).
4. **Data: returns-vs-swap signature diagnostic** (from 01:05).
5. **Training: SWIL** (needs FF re-runs).

## 2026-10-31 04:10-04:40 UTC+8 loop

### Experiments (box; VPS idle)
1. **Latency diagnostic (`lat_diag.py`, results/lat_diag; store only, 20 s).** For every swapped-digit arrival between
   the swap and its old class's first remap (default a25w250h2k2, seeds 0-5, both worlds), what the store did:
   out-of-radius (nearest prototype is the old class, item just outside its radius, median d/r 1.16) **42-44%**;
   in-radius but fewer than K = 2 prototypes hot **24-26%**; hot enough but **vetoed by a25 15-19%**; third-class
   basin 8-10%; already in a new-label basin 1-2%. Prediction (>= 50% of latency spent a25-blocked) **killed**; the
   kill condition for a25 (0 blocks) also fails, so a25 is a real but minority cost. Mechanism of the blocks: the
   world always opens the swapped pair's season at the swap, so at the swap the 250-tick window still holds the
   previous season (label s count 0-4 vs top label ~90) and a25 calls the old class absent for its first ~50 ticks.
2. **Onset-aware presence (`onset_presence.py`, results/onset_presence; ~1 min).** Present if a25 passes on W = 250
   OR on the last S ticks. Test setting s50: held-out bg 0.02 swap **74.0** (s0 72.2), returns 68.4, latency 75 -> 55
   ticks, but bg 0.2 pre-swap remaps **8.5 > 5: killed** as pre-registered. Sweep member s100: held-out bg 0.02 swap
   74.0 / returns 68.5 / lifetime 96.1, latency -20 ticks; seeds 0-2 swap 71.3 (+0.3); bg 0.2 swap 71.9 (+0.3) with
   2.7 pre-swap remaps (s0 0.3). Passes every kill line but was chosen after seeing s50, so it is a lead, not a
   default: confirm on fresh seeds 6-8 (gate_probe FF logs for bg 0.02 seeds 6-8 launched on the box, ~10 min,
   results/gate_probe_s678).

### Literature insight
Jezek, Henriksen, Treves, Moser & Moser 2011 (Nature 478:246, "Theta-paced flickering between place-cell maps in the
hippocampus"): after an instant switch of the light cues ("teleportation"), CA3 flips to the other map within one
theta cycle, flickering between maps for a few seconds before settling. Context membership is read on the fastest
timescale available and only *stabilised* by slower integration. Our a25 guard is a context-membership test, and
reading it only on a 250-tick window makes it lag a context switch by exactly the latency we were chasing. The
dual-window rule is the flicker version: a fast window may declare presence, the slow one keeps it. The bg 0.2 false
remaps at S = 50 are the cost of flicker without settling: a short window lets a burst of background arrivals
pass for a context.

### Paths closed
- a25 as the main source of remap latency: closed (15-19% of the window). The bulk is evidence that never counts:
  swapped digits landing just outside the old prototype's radius.
- Onset presence with S = 50: closed (false remaps at bg 0.2). S = 100 is a lead.

### Hypotheses, ranked
1. **Training/store (test next, needs results/gate_probe_s678): confirm s100 on fresh seeds 6-8.** Prediction: swap
   >= +1 over s0 under mq30_s2, returns within 0.5, pre-swap remaps <= 3. Kill: gain < 0.5 on seeds 6-8.
2. **Neuron: soft radius for conflict evidence.** 42-44% of post-swap evidence is lost just outside the radius (median
   d/r 1.16). Count conflicts out to 1.25 r as half a hit (H stays 2). Prediction: latency -15 ticks, swap +1;
   kill: pre-swap remaps > 5 in either world (borderline digits also sit at 1.0-1.3 r, cf. close_remap).
3. **Data: is the a25 lag a world artefact?** Swaps always open the pair's season. Variant world with the swap
   mid-season (pair already in context for >= 500 ticks). Prediction: s0 and s100 within 0.5 swap there; if s100
   still gains, presence lag is not the mechanism.
4. **Architecture: router sees remap events** (from 02:35, unchanged).
5. **Training: SWIL** (needs FF re-runs).

## 2026-10-31 05:30 UTC+8

### Experiments
1. **s100 fresh-seed confirmation** (`onset_s678.py`, results/onset_s678, FF logs results/gate_probe_s678): bg 0.02
   seeds 6-8, mq30_s2 swap 71.8 (s0) vs 71.9 (s100), returns 71.1 vs 71.4, latency 59 vs 49. Not confirmed (gain < 1).
   Onset presence closed.
2. **Seed 6 breaks the default store.** a25w250h2k2 makes 34 pre-swap remaps on seed 6 (0 on 7 and 8). Diagnosis
   (store replay, seed 6): all ping-pong between co-present confusable classes, 4->9 14, 9->4 12, 7->9 4, 9->7 4, in
   seasons 3-5 where 4, 8, 9 are all in season (~1000 arrivals each). a25 only blocks remaps of absent classes.
3. **Rival-aware evidence** (`rival_remap.py`, results/rival_remap): c1.5 kills false remaps (0 everywhere) but swap
   68.6 vs 72.0 (killed); rad keeps 10 on seed 6 (prediction <= 5 failed, killed).
4. **Directional unlock** (`rival_unlock.py`, results/rival_unlock): c1.5_u has 0 pre-swap remaps on all 15 runs,
   swap within 0.3 of base in both worlds, returns equal, latency unchanged. rad_u does not help (seed 6 at 40).
   Lead, post hoc on seed 6: needs fresh seeds 9-11 (gate_probe FF logs first).

### Literature insight
Yassa & Stark 2011 (Trends Neurosci 34:515, "Pattern separation in the hippocampus"): the dentate gyrus/CA3 decides
whether an input is a *new* event or a degraded copy of a stored one by its distance to all stored neighbours, not
to one. Our remap rule asked only "is this inside s's prototype?", so a borderline 9 was read as "the 4 memory now
means 9". The mnemonic-discrimination question is "is there already a 9 memory nearly as close?"; if yes the input
is a lure for an existing memory, not evidence of a rule change. The cost appears when the separation test also
blocks a real change after its first step. Leutgeb et al. 2005 (Science 309:619) found CA3 remapping is coherent
across the population (global remapping) rather than decided cell by cell; the directional unlock is a crude version:
once part of the s->y remap has fired, the rest of it is not re-tested against the rival it just created.

### Paths closed
- Onset-aware presence (s100): not confirmed on fresh seeds.
- Rival test without unlock (c1.5) and radius rival test (rad, rad_u).

### Hypotheses, ranked
1. **Architecture/store (test next): confirm c1.5_u on fresh seeds 9-11** (launch gate_probe FF logs for bg 0.02
   seeds 9-11 first). Prediction: pre-swap remaps <= 1 per run, swap within 0.5 of base. Kill: any run > 5 or swap
   loss > 1.
2. **Data: confusable co-presence world.** Force 4/9 and 3/5/8 into the same season in every seed. Prediction: base
   a25w250h2k2 gives > 10 false remaps per run, c1.5_u <= 1. This turns seed 6 from luck into a stress test.
3. **Neuron: soft radius for conflict evidence** (from 04:40; 42% of post-swap evidence lands at 1.0-1.3 r). Now
   testable safely because the rival test filters borderline lures: count 1.0-1.25 r conflicts as half hits only
   when no rival. Prediction: latency -15 ticks, swap +1, false remaps stay 0.
4. **Training: does the unlock window U matter?** U in {500, 2000, inf}. Prediction: flat above 500 (one remap
   burst is ~2 events), so U is not a tuned knob.
5. Router sees remap events; SWIL (unchanged).

## 2026-10-31 06:10 UTC+8

### Experiments
1. **Robustness of c1.5_u** (`rival_robust.py`, results/rival_robust; ~4 min on the box, same 15 runs). Pre-registered:
   robust if every neighbour keeps pre-swap remaps <= 1 per run (seed 6 <= 5) and held-out bg 0.02 swap within 1.0
   of base (72.0); fragile if two or more neighbours break.
   - **U is not a knob (05:30 hyp 4 confirmed).** U = 500, 2000, 8000 and inf give identical swap (71.7 / 70.6 /
     71.5 in the three blocks) and 0 pre-swap remaps everywhere. Use permanent unlock: once s->y has remapped, the
     pair is never rival-tested again. One knob fewer.
   - **C is a narrow window (fragile, killed as pre-registered).** C = 1.2 leaks (seed 4: 4 and seed 6: 4 pre-swap
     remaps; swap 72.1); C = 2.0 vetoes the first true remap (held-out swap 62.8, -9.2, first-remap latency 67 -> 129
     ticks); C = 3.0 never remaps (swap 54.3, store-alone swap 2.7). Only 1.5 passes both lines. The separating margin
     between a confusable co-present lure and a relabelled digit is a factor of ~1.2-2 in squared distance to the
     nearest prototype of the new label.
   - Consequence: c1.5_u stays the lead, but a fixed ratio threshold is not a rule we can trust across datasets.
     The fresh-seed confirmation (seeds 9-11) now tests c1.5_uinf; FF logs launched on the box at 06:11 UTC+8
     (results/gate_probe_s91011, ~40 min).

### Literature insight
Lacy, Yassa, Stark, Muftuler & Stark 2011 (Learn Mem 18:15, "Distinct pattern separation related transfer functions
in human CA3/dentate and CA1 revealed using high-resolution fMRI and variable mnemonic similarity"): as lure
similarity is varied in steps, CA3/DG responds with a steep, nonlinear transfer (a small input change flips it to
"new"), while CA1 responds graded and roughly linearly. Our rival test is a pure CA3/DG step at one ratio C, and the
sweep shows exactly the cost of a step placed in a narrow gap: 1.2 leaks lures, 2.0 blocks real change. The CA1-like
alternative is to weight each conflict by how unambiguous it is (a graded function of the rival ratio) and let the
H-hit accumulator integrate, so no single threshold has to sit in the gap.

### Paths closed
- Unlock window U as a parameter (any U >= 500 is identical; use permanent pair unlock).
- Rival ratio C as a robust setting: only C = 1.5 passes; C = 1.2, 2.0, 3.0 each fail a pre-registered line.

### Hypotheses, ranked
1. **Store (running): c1.5_uinf on fresh seeds 9-11.** Prediction: pre-swap remaps <= 1 per run, swap within 0.5 of
   base. Kill: any run > 5 or swap loss > 1. Script: rival_robust-style replay joined with results/gate_probe_s91011.
2. **Data (most decisive next): confusable swap pair.** All swaps so far relabel pairs whose real prototypes are far
   apart. Swap a confusable pair (4<->9). Prediction if the rival idea is wrong: c1.5_u vetoes the true swap (swap
   loss > 3 vs base) because the new label's real prototypes are close rivals by construction. Needs new FF logs.
3. **Neuron: graded rival evidence (CA1-like).** A conflict counts w = clip((ratio - 1.2) / 0.8, 0, 1) of a hit
   instead of 0/1. Prediction: 0 pre-swap remaps and swap within 0.5 of base for ramp ends 1.0-1.4 / 1.8-2.4, i.e.
   a flat plateau where the step had a cliff. Kill: no ramp beats the step on both lines. ~4 min, test next hour.
4. **Data: confusable co-presence world** (from 05:30 hyp 2): force 4/9 and 3/5/8 into one season every seed.
5. Soft radius for conflict evidence; router sees remap events; SWIL (unchanged).

## Loop 2026-10-31 07:05 UTC+8

### Experiment (`rival_fresh.py`, results/rival_fresh; pre-registered in the docstring)
- **Part A, fresh seeds 9-11 (FF logs made after C was chosen): c1.5_uinf confirmed.** Pre-swap false remaps 0 on
  all three seeds (base: 4, 20, 0; seed 10 is a second ping-pong seed). Swap (mq30_s2) 70.2 vs 71.1 (-0.9, inside
  the 1.0 kill line, outside the 0.5 hope); lifetime 96.0 vs 95.9, returns 67.7 = 67.7. Across all 18 runs
  c1.5_uinf has 0 pre-swap remaps (base 66). **New default store: a25w250h2k2_c1.5_uinf.**
- **Part B, graded (CA1-like) rival weight: killed.** w = clip((q - a)/(b - a), 0, 1) per conflict, remap at summed
  weight >= 2. g1-2 keeps 0 false remaps but loses swap 2.8 (fresh) / 3.2 (seeds 3-8) / 2.2 (bg 0.2), first remap
  +15 ticks; g1.2-3 loses 11-16 swap points. Grading does not widen the safe band; it dilutes the true swap evidence,
  because many relabelled digits sit at ratio 1.5-2 where the step counts them fully. Closed.

### Literature insight
Wills, Lever, Cacucci, Burgess & O'Keefe 2005 (Science 308:873, "Attractor dynamics in the hippocampal
representation of the local environment"): as a box is morphed step by step between square and circle, place cells
do not drift gradually; the whole ensemble switches abruptly and coherently between the two maps (cf. Leutgeb et al.
2005, Neuron 48:345, where a different protocol gives graded change). The lesson for pdp: the remap is an identity
decision, and the hippocampus makes identity decisions all-or-none and collectively. Our result agrees: a per-hit
step beats a per-hit ramp. Robustness should then come from the collective part (how many prototypes, and how
their evidence is normalised), not from softening each hit.

### Paths closed
- Graded per-hit rival weighting (both scales fail the swap line).

### Hypotheses, ranked
1. **Data (decisive, next): confusable swap pair 4<->9.** Every swap so far relabels far-apart classes. With 4<->9 the
   new label's real prototypes are close rivals by construction. Prediction if the rival rule is wrong: c1.5_uinf
   vetoes the true swap (swap loss > 3 vs base). Needs new FF logs with WORLD swap=(4, 9) (~40 min on the box).
2. **Neuron/architecture: self-normalised rival threshold.** Replace fixed C by the pair's own statistics: a conflict
   labelled y on an s prototype counts only if its ratio q exceeds the 90th percentile of q over recent correctly
   labelled s arrivals (how close y normally gets to real s). Prediction: 0 pre-swap remaps and swap within 0.5 of
   c1.5_uinf, and no free constant to tune per dataset. Kill: worse than c1.5_uinf on either line.
3. **Training algorithm: sleep consolidates remaps.** Remapped prototypes are new memories in the store only; replay
   them in the next sleep with priority (Mattar & Daw need term) so the FF cortex flips sooner. Prediction: FF swap
   accuracy in the second 300 ticks after swap +5. Kill: < +2.
4. Confusable co-presence world; soft radius; router sees remap events; SWIL (unchanged).
- 07:13 UTC+8: FF logs for the confusable-swap world (gate_probe world bg002s49 = bg 0.02, swap (4, 9)) launched on
  the box for seeds 0-5 -> results/gate_probe_s49 (~50 min). Note for the replay: rival_remap.run builds the world as
  dict(LF.WORLD, bg=bg), so it must also pass swap=(4, 9) for these logs, or the assert on labels fails. Since 9 is
  also the novel digit (appears at 30%), read "novel first 300" with care in that world.

## Loop 2026-10-31 08:05 UTC+8

### Experiment (`rival_s49.py`, pre-registered in its docstring; results/rival_s49, results/rival_z5)
- **A. Confusable swap 4<->9 (bg002s49, seeds 0-5, FF logs made after the default was chosen).** The fixed-ratio
  rival test fails exactly where predicted: c1.5_uinf swap (mq30_s2) 66.3 vs base 69.7 (-3.4, kill line 2.0), first
  post-swap remap 227 ticks vs 59, 528 rival vetoes per run. A 4 relabelled 9 sits close to the real 9 prototypes, so
  the rival test reads the true swap as a lure. **Killed as a general rule.** (Base makes only 1.0 pre-swap false
  remap per run in this world.)
- **B. Extinction override z5** (rival veto lifted when the s prototype's last 5 history entries contain no
  same-label confirmation). Safety: 0 pre-swap false remaps on all 24 runs (18 old + 6 s49; base 66 + 6). Swap:
  held-out bg 0.02 seeds 3-11 71.6 vs base 71.7 (c1.5_uinf 71.2); bg 0.2 71.4 vs 71.6; s49 68.1 vs 69.7 (-1.6,
  latency 77 vs 59). It misses its pre-registered s49 line (loss <= 1.0), so "z5 fully fixes confusable swaps" is
  killed, but z5 dominates c1.5_uinf on every line. **New default store: a25w250h2k2_c1.5_uinf_z5.**

### Literature insight
Gershman, Blei & Niv 2010 (Psychological Review 117:197, "Context, learning, and extinction"): an animal infers a
new latent cause not because one observation is surprising, but because the old cause's predictions stop being
confirmed; scattered surprises inside a stream that still confirms the old cause are absorbed by it. For pdp, a lure
(a borderline 9 in a 4 prototype) arrives while the 4 prototype keeps getting real 4s; a true relabelling stops every
confirmation. So the remap test should ask "has this prototype stopped being confirmed?", which needs no distance
ratio between classes and does not care whether the pair is confusable. z5 is the crudest form of this; it already
beats the geometric rival test everywhere.

### Paths closed
- Fixed-ratio rival test as a stand-alone rule (fails the confusable swap).

### Hypotheses, ranked
1. **Neuron: confirmation trace replaces the rival test.** Drop the distance ratio entirely: count a conflict hit
   only when the prototype's leaky confirmation trace (tau ~ 5 entries) is below a threshold. Prediction: 0 pre-swap
   remaps on all 24 runs and s49 swap within 1.0 of base (z5 is -1.6 because the rival test still vetoes the first
   hits). Kill: any run > 5 pre-swap remaps.
2. **Data: label-noise world.** Permanently relabel 10% of 4s as 9 (no swap). A confirmation rule must never remap a
   4 prototype there, since real 4s keep confirming it. Kill for z-style rules: > 1 remap per run.
3. **Neuron: z sensitivity (z3, z8).** If z3 closes the s49 gap without false remaps, the window is a free choice;
   if z3 leaks, the rule is a tuned point. Kill: z3 and z8 both break a line.
4. **Training: store remap tells the cortex.** After a store remap s->y, raise the FF wake LR for arrivals labelled
   y near s for 300 ticks. Prediction: FF swap first 300 +5 on s49 (FF 67.2). Kill: < +2.
5. Self-normalised rival threshold (demoted: the confirmation idea makes the ratio unnecessary); sleep consolidates
   remaps; SWIL.

## Loop 2026-10-31 09:05 UTC+8

### Experiments (pre-registered in each docstring; results/conf_trace*, results/conf_defer*)
- **conf_trace.py nN (confirmation-only veto, no distance ratio).** An in-radius conflicting hit counts only if the
  prototype's last N history entries hold no same-label confirmation. Safety is perfect (0 pre-swap false remaps on
  all 24 runs for n3/n5/n8), but swap (mq30_s2) collapses, monotone in N: held-out bg 0.02 seeds 3-11 67.3/64.7/61.0
  vs base 71.7; bg 0.2 63.7/59.4/55.1 vs 71.6; s49 64.7/60.2/57.2 vs 69.7. First post-swap remap 88-539 ticks vs 59-68.
  **Killed (hypothesis 1 of 08:05, and z-window sensitivity: both n3 and n8 break the line).** Mechanism: a vetoed hit
  is logged as out-of-radius, so evidence seen while pre-swap confirmations are still in the window is thrown away.
- **conf_defer.py dN (defer, don't discard).** Every in-radius conflict is logged as evidence as in base; a prototype
  may only be remapped once its last N entries hold no confirmation. d5 killed by its pre-registered line (swap -2.7
  held-out, -4.8 bg 0.2, -4.9 s49). d3: 0 false remaps on all 24 runs, swap within 0.8 of base on the 18 old runs
  (70.9/71.1/71.4 vs 71.7/71.0/71.6), but s49 66.9 vs 69.7 (-2.8), worse than z5 (68.1). Deferring recovers the
  non-confusable worlds but not the confusable one.
- **Decision: z5 stays the default.** A confirmation gate on its own costs swap in every world; as a lifter on top of
  the rival test (z5) it only acts where the rival test fired.

### Why s49 resists confirmation rules (untested mechanism, hypothesis 1 below)
After a 4<->9 swap, real 9s now arrive labelled 4. Because 4 and 9 are confusable, some of them land inside 4
prototypes and count as same-label confirmations. So the old cause keeps being "confirmed" by the swapped partner and
any staleness rule waits. In non-confusable swaps (2<->3) the partner lands far from the s prototypes.

### Literature insight
Yu & Dayan 2005 (Neuron 46:681, "Uncertainty, neuromodulation, and attention"): expected uncertainty (known
unreliability inside a context, signalled by ACh) and unexpected uncertainty (evidence that the context itself
changed, signalled by NE) must be tracked separately, and a context switch should be inferred only when surprise
exceeds what the known unreliability predicts. For pdp, a prototype at the 4/9 border has a high expected
conflict rate (lures) and its rim confirmations are themselves unreliable. Both the remap evidence and the
confirmation evidence should be read against that prototype's own baseline, not counted as equal-weight hits.

### Paths closed
- Confirmation-only veto (nN) for any N; deferred confirmation gate at N >= 5; window-free confirmation rules as a
  replacement for z5.

### Hypotheses, ranked
1. **Neuron: core-only confirmation.** Count a same-label hit as confirmation only if d <= 0.5 x radius (rim hits
   by a confusable partner don't confirm). Use it inside z5 (z5c). Prediction: s49 swap within 1.0 of base, 0 false
   remaps on all 24 runs. Kill: s49 gain < +0.8 over z5, or any run > 5 false remaps. First check the mechanism:
   count post-swap 9-labelled-4 confirmations of 4 prototypes and their d/r.
2. **Data: label-noise world.** Permanently relabel 10% of 4s as 9 (no swap). z5 must never remap a 4 prototype
   there. Kill: > 1 remap per run.
3. **Neuron: per-prototype expected-conflict baseline (Yu & Dayan).** Remap only when the conflict rate in the last
   W entries exceeds the prototype's long-run conflict rate by a margin. Prediction: replaces both rival test and z5
   with one rule; must pass the label-noise world.
4. **Training: store remap tells the cortex** (raise FF wake LR near s for 300 ticks after a remap s->y; FF s49
   swap +5 or kill < +2).
5. Sleep consolidates remaps; SWIL.

## Loop 2026-10-31 10:05 UTC+8

### Experiment (pre-registered in the docstring; results/core_conf_s49, results/core_conf)
- **core_conf.py z5_qQ (core-only confirmation inside z5).** A same-label hit still updates the radius but counts as a
  confirmation only if d <= Q x r (squared distances); rim hits are logged as neutral. Q = 0.5 and 0.25.
  On s49 swap (mq30_s2) is 69.6 for both vs z5 68.1 and base 69.7, so the swap gap closes. But false remaps come back:
  pre-swap remaps per run on seed 6 are 20/20 (z5 0, base 34), seed 10 4/28 (z5 0, base 20), and seeds 1, 4, 9 match
  base. Rival vetoes fall from 269-555 to 20-32. **Killed by its pre-registered line (any run > 5 false remaps).**
- **Why: the radius is a mean, not a bound.** The radius is an EMA of confirmation distances, so the typical
  confirmation sits at the rim by construction: post-swap same-label hits on 4/9 prototypes have median d/r 0.95-0.96
  on all 6 s49 seeds, and only 0.2-0.6% fall at d <= 0.5r. A core-only rule therefore deletes almost every
  confirmation, z5 lifts the rival veto almost always, and the store degrades to base. The +1.5 on s49 is just the
  rival test being switched off.
- **Mechanism check was confounded.** The impostor count (true class differs from the prototype's label) cannot
  separate swapped 9s confirming 4 prototypes from correct hits on already remapped prototypes, because remaps
  happen within about 60 ticks and flip the prototype's label. The suspected mechanism of 09:05 stays untested; it
  needs per-prototype origin tracking.
- **Decision: z5 stays the default.** Any confirmation rule must be scaled to the prototype's own distance
  distribution (a quantile or a variance), never a fraction of the mean radius.

### Literature insight
Kumaran & McClelland 2012 (Psychological Review 119:573, REMERGE): in a hippocampal store, how strongly a probe
supports an item is set by its similarity relative to every other stored item, through recurrent competition, not by
its absolute distance to that item. For pdp this says a confirmation of a 4 prototype by an arrival that sits almost
as close to a 9 prototype should count as roughly half a confirmation. The current store counts confirmations by
absolute distance, and core_conf shows absolute distance cannot tell a lure from a real member, because both sit at
the mean radius.

### Paths closed
- Core-only confirmation by a fraction of the mean radius (q0.5, q0.25), in any rule that uses the radius as a scale.

### Hypotheses, ranked
1. **Neuron: posterior-weighted confirmation (REMERGE).** Each same-label hit adds w = d_y / (d_s + d_y) (squared
   distances to the s prototype and the nearest rival prototype of any other label) to a leaky confirmation trace;
   z5 lifts the veto when the trace over the last 5 entries is below 1.5. Prediction: s49 swap >= z5 + 0.8 and
   0 runs > 5 false remaps on all 24. Kill: either line broken.
2. **Data: label-noise world.** Permanently relabel 10% of 4s as 9 (no swap). z5 must not remap a 4 prototype. Kill:
   > 1 remap per run. Still needed before any confirmation rule is trusted.
3. **Diagnostic: per-prototype origin tracking.** Tag each prototype with the class it was created from, and recount
   post-swap confirmations of not-yet-remapped 4 prototypes by true 9s. Decides whether the 09:05 mechanism is real.
4. **Training: store remap tells the cortex** (raise FF wake LR near s for 300 ticks after a remap; FF s49 swap +5,
   kill < +2).
5. Per-prototype expected-conflict baseline (Yu & Dayan); sleep consolidates remaps; SWIL.

## Loop 2026-10-31 11:05 UTC+8

### Experiment (pre-registered in the docstring; results/post_conf_s49, results/post_conf)
- **post_conf.py z5_pP (posterior-weighted confirmation).** Each same-label hit carries w = d_r / (d_s + d_r), d_r the
  squared distance to the nearest prototype of another label; z5 lifts the rival veto when the last 5 weights sum
  below P. On s49 swap (mq30_s2): p1.0 68.6, p1.5 68.5, z5 68.1, base 69.7. Rival vetoes fall from 269 to 155/113.
  Pre-swap false remaps <= 2 per run on all 24 runs; old runs within 0.2 of z5. **Killed (gain < +0.8).**
- **Origin tracking settles the 09:05 mechanism.** Each prototype carries the label it was created with. After the
  swap, on 4/9 prototypes still carrying their original label, base logs 740 impostor confirmations against 626
  in-radius conflicts (54%, per seed 12-71%): the stale cause really is being "confirmed" by partner digits.
- **But geometry cannot see impostors.** Their mean w is 0.579, real pre-swap confirmations 0.589. A real member is
  only about 1.4x closer (squared) to its own prototype than to the nearest rival, so relative similarity in pixel space
  carries almost no label information on a confusable border.
- **Decision: z5 stays the default.** Store-geometry confirmation rules (absolute 10:05, relative 11:05) are closed.

### Literature insight
O'Reilly & McClelland 1994 (Hippocampus 4:661) and McClelland, McNaughton & O'Reilly 1995: the hippocampus can store
overlapping episodes without interference only because the dentate gyrus re-codes them sparsely (pattern separation)
before CA3 stores them; the neocortex keeps the overlapping, similarity-based code. The pdp store keeps prototypes in
raw pixel space, the neocortical-style code, so it inherits the cortex's confusions (w 0.59 for real members) instead
of separating them. CLS also says which system already knows a 9 is a 9 right after the swap: the slow cortex. So the
discriminating signal for an impostor confirmation is cortical disagreement, not store geometry.

### Paths closed
- Any confirmation rule built only on store distances (absolute or relative) in pixel space.

### Hypotheses, ranked
1. **Architecture: cortex-vetted confirmation.** A same-label hit on prototype s counts as a confirmation for z5 only
   if the FF cortex's prediction for that arrival is s (taken from the gate_probe FF logs). Right after a 4<->9 swap
   the cortex still calls a true 9 a 9, so impostors stop confirming the 4 prototype. Prediction: s49 swap >= z5 + 0.8
   and <= 1 false remap per run on all 24. Kill: either line broken.
2. **Neuron: pattern-separated store space.** Keep prototypes in a fixed sparse random expansion (k-WTA, ~2000 units,
   5% active, a dentate-like code). Prediction: real-member w rises from 0.59 to >= 0.7 and the impostor-vs-member w gap
   to >= 0.05. Kill: w gap < 0.03.
3. **Data: label-noise world** (10% of 4s permanently labelled 9, no swap): z5 and hyp 1 must not remap. Kill: > 1 per run.
4. **Training: store remap tells the cortex** (raise FF wake LR near s for 300 ticks after a remap; FF s49 swap +5,
   kill < +2).

## Loop 2026-10-31 12:05 UTC+8

### Experiment (pre-registered in the docstring; results/cortex_conf_s49, results/cortex_conf)
- **cortex_conf.py cv (cortex-vetted confirmation).** A same-label hit on prototype s confirms (history 0) only if
  the FF cortex's logged prediction for that arrival is s; otherwise it is a neutral entry. s49 swap (mq30_s2): cv
  68.2, z5 68.1, base 69.7. 0 pre-swap false remaps on all 24 runs; old-run swap identical to z5. **Killed (gain
  < +0.8).**
- **Diagnostic: the cortex is not slow.** On true 4s and 9s after the swap, FF says the new label 49-69% of the time in
  ticks 0-300, 82-90% in 300-1500, 93-94% after. It vetoes 2.5% of pre-swap confirmations and only 3.5% of post-swap
  confirmations on 4/9 prototypes: it agrees with the impostors, so it cannot reject them.
- **Decision: z5 stays the default.** The store's swap edge lives only in the first ~300 ticks, where the cortex is a
  coin flip and the store remaps after ~60-77 ticks.

### Literature insight
Lisman & Grace 2005 (Neuron 46:703) put a comparator in CA1: it matches CA3's recalled prediction against the direct
entorhinal (cortical) input and signals novelty when they disagree, which gates new encoding. That comparator only
works if the two inputs carry different time constants: CA3 recall reflects the stored past, the entorhinal input the
present. McClelland, McNaughton & O'Reilly 1995 make the same point at system level: complementarity needs a
learning-rate gap. In pdp the FF cortex at wake LR x1 relabels a swapped digit within a few hundred ticks, as fast
as the store remaps, so there is no "old" signal for a comparator to use. The earlier finding that slow waking (wake
LR x0.3 to x0.1) is free or nearly free on anytime accuracy is therefore also an architectural choice: it is what
makes the cortex a usable reference for the store.

### Paths closed
- Cortex-vetted confirmation with the current fast (wake LR x1) never-sleeping cortex.

### Hypotheses, ranked
1. **Training: slow cortex makes vetting work.** With wake LR x0.1 the FF should still call a swapped digit by its
   old name for >= 1000 ticks. Prediction (on gate_probe_s49wl1 logs): FF old-name share in ticks 0-300 >= 80%, and cv
   gives s49 swap >= z5 + 0.8 with <= 1 false remap per run. Kill: old-name share < 60% or cv gain < +0.8.
   Logs launched 12:14 UTC+8 on the box (~45-60 min); next loop runs `cortex_conf.py A` against them (point LOGS
   at results/gate_probe_s49wl1, out results/cortex_conf_s49wl1).
2. **Neuron: pattern-separated store space** (fixed sparse k-WTA expansion, ~2000 units, 5% active, dentate-like).
   Prediction: real-member relative weight rises from 0.59 to >= 0.7 and the impostor-vs-member gap to >= 0.05.
   Kill: gap < 0.03.
3. **Architecture: comparator remap.** Count an in-radius conflict as strong evidence (bypass the rival veto) only
   when the store's prototype label and the slow cortex agree with each other but disagree with the given label (the
   CA1 mismatch). Needs hyp 1's slow cortex. Prediction: s49 swap >= base, 0 false remaps. Kill: > 1 per run.
4. **Data: label-noise world** (10% of 4s permanently labelled 9, no swap): z5 must not remap. Kill: > 1 per run.

## Loop 2026-10-31 13:00 UTC+8

### Experiments
- `cortex_conf.py C` (slow-cortex logs results/gate_probe_s49wl1): **killed**. The x0.1 cortex is genuinely slow (old
  name 63% in ticks 0-300, 59% in 300-1500; x1 was 39% / 12%), vetoes 13.5% of post-swap confirmations, yet cv = z5
  (49.0 vs 49.3) and first post-swap remap 76 vs 77 ticks. Combined lifetime 93.5 vs 96.0 with the fast cortex.
- `z_window.py A/B`: z2/z3 vs z5. **z3 killed** (+0.4), z2 +0.7 (under +0.8). 0 false remaps on all 24 runs. Latency
  75/76/77 vs base 59: the window is ~2 ticks of the 18-tick cost; the c1.5 rival test is the rest.

### Literature insight
McClelland, McNaughton & O'Reilly 1995 do not claim a slow cortex is good on its own: slow interleaved learning works
because the hippocampus reinstates recent memories offline, so the cortex eventually gets the new information without
overwriting the old. Kumaran, Hassabis & McClelland 2016 (Trends Cogn Sci 20:512) add that cortex can learn fast when
new data fit existing structure, and the hippocampus carries what doesn't fit. pdp now shows both halves: with sleep,
wake LR x0.3 was +7 anytime for free; in the never-sleeping life world the same slowing just loses 2.5 lifetime points,
because nothing ever reinstates the store's knowledge into the cortex. A label swap is the schema-inconsistent case,
which is exactly where the store should answer and the cortex should be taught offline, not consulted online.

### Paths closed
- Cortex-vetted store confirmation at any wake LR (x1 and x0.1).
- Extinction-window length as the source of the z5 swap cost.

### Hypotheses, ranked
1. **Architecture/training: sleep in life.py with a slow cortex.** Add periodic sleep (store replay into FF, as in
   the proto9 runs) to the life world with wake LR x0.1 vs x1. Prediction: x0.1 + sleep combined lifetime >= 96.0 and
   returns >= 75 (above the fast never-sleeping 72). Kill: lifetime < 95.5 or x0.1 + sleep <= x1 + sleep.
2. **Training: run-length rival lift.** Lift the c1.5 rival veto when the same prototype gets >= 3 consecutive
   in-radius conflicts with one consistent new label. Prediction: s49 first remap <= 65 ticks, swap >= 69.3, <= 1 false
   remap per run on 24 runs. Kill: false remaps return on any run.
3. **Neuron: pattern-separated store space** (sparse k-WTA expansion, ~2000 units, 5% active). Prediction: real-member
   vs impostor weight gap >= 0.05. Kill: < 0.03. Lower priority now that impostors are not the latency cause.
4. **Data: label-noise world** (10% of 4s permanently labelled 9): z5 must not remap. Kill: > 1 false remap per run.

## Loop 2026-10-31 14:00 UTC+8

### Experiments (each pre-registered in its docstring; s49 = confusable 4<->9 swap, seeds 0-5; mq30_s2 route)
- `run_lift.py` (lift the c1.5 rival veto after a streak of L same-label in-radius conflicts on one prototype):
  s49 swap r2 68.5, r3 68.2, r5 68.1 vs z5 68.1 (base 69.7); first post-swap remap 73/77/77 ticks (base 59).
  0 false remaps on all 24 runs. **Killed (gain < +0.8).** Streaks are rare because a 4 prototype's post-swap hits
  interleave true 4s now labelled 9 with impostor 9s now labelled 4.
- `cusum_lift.py` (Page CUSUM per prototype: +1 per conflict, -P per confirmation, lift at H): cu0.5_h3 68.5,
  cu0.5_h5 68.6, cu0.25_h3 68.5; latency 76. 0 false remaps on s49. **Killed.**
- `retro_lift.py` (CUSUM lift plus retroactive credit: vetoed conflicts kept as latent evidence and counted once the
  lift fires): 68.4/68.6, latency 75-76, and false remaps return on seed 1 (2). **Killed.**
- Decision: z5 stays the default. The rival-veto timing/accounting family is closed for the s49 latency gap: no rule
  that reuses the same per-hit evidence moved first remap below 73 ticks.

### Literature insight
Lorden 1971 (Ann. Math. Stat. 42:1897) proves Page's CUSUM is minimax-optimal for detecting a change, with detection
delay about log(gamma)/I, where gamma is the mean time between false alarms and I is the KL divergence per
observation between the post- and pre-change distributions. The store's remap is a change detector. Base (no rival
test) has ~7 false remaps per run on seeds 3-11 and remaps after 59 ticks; z5 has 0 and remaps after 77. Under
Lorden's bound that ~18-tick gap is the price of pushing gamma from ~2000 ticks to beyond a 16-day life, and no
counting rule on the same hits can remove it; this hour's three kills are what the bound predicts. Only raising I,
making each hit more diagnostic of "this prototype now means y", can cut latency at a fixed false-alarm rate. In pdp
that points at pattern separation of the store code (a 4 and a 9 that look alike in pixels should land in different
prototypes), not at the confirmation or veto rule.

### Paths closed
- Rival-veto lifting by streak, CUSUM, or retroactive credit (all under +0.8 on s49; retro credit leaks false remaps).

### Hypotheses, ranked
1. **Neuron: pattern-separated store space raises per-hit evidence.** Keep prototypes in a fixed sparse random
   expansion (k-WTA, ~2000 units, 5% active, dentate-like). Lorden-based prediction: with the same z5 rule, s49 first
   post-swap remap <= 65 ticks and swap >= z5 + 0.8 with 0 false remaps on 24 runs; diagnostic: real-member vs
   impostor relative weight gap >= 0.05. Kill: latency > 70 or gap < 0.03.
2. **Architecture/training: sleep in life.py with a slow cortex** (carried from 13:00). Prediction: wake LR x0.1 +
   sleep lifetime >= 96.0 and returns >= 75. Kill: lifetime < 95.5 or x0.1 + sleep <= x1 + sleep.
3. **Data: label-noise world** (10% of 4s permanently labelled 9, no swap): z5 must not remap. Kill: > 1 per run.
4. **Training: store remap tells the cortex** (raise FF wake LR near s for 300 ticks after a remap; FF s49 swap +5,
   kill < +2).

## Loop 2026-10-31 15:00 UTC+8

### Experiment (pre-registered in its docstring; s49 = confusable 4<->9 swap, seeds 0-5; mq30_s2 route)
- `kwta_store.py A` (14:00 hypothesis 1): store prototypes, radii, rival test and answers moved into a fixed random
  expansion of the pixels (N = 2000, h = W(x - xbar)), k-WTA binary at k = 100 (5%) or k = 40 (2%); control `lin` =
  same projection without k-WTA. All inside z5.
  | Store | Swap (mq30_s2) | Lifetime (mq30_s2) | Returns (mq30_s2) | Store-only lifetime | First post-swap remap |
  |---|---|---|---|---|---|
  | base | 69.7 | 96.0 | 71.8 | 91.4 | 59 |
  | z5 | 68.1 | 96.0 | 72.0 | 91.4 | 77 |
  | z5 x2000 k100 | 67.2 | 95.8 | 58.4 | 86.5 | 139 |
  | z5 x2000 k40 | 67.2 | 95.8 | 57.3 | 85.1 | 1059 |
  | z5 x2000 lin | 67.8 | 95.9 | 72.0 | 91.1 | 72 |
  0 pre-swap false remaps for every z5 variant. **Killed hard** (latency > 70, swap below z5, store lifetime -5 to -6,
  returns -14). The linear control is within noise of pixels, so the damage is the sparsification itself. The mq30
  route also stops using the store (store answers 1.1% -> 0.1%): binary codes compress the relative-margin scale the
  gate thresholds were set on.
  Note: run with OMP_NUM_THREADS=1; with default BLAS threads 6 workers oversubscribed 8 cores (load ~49) and stalled.

### Literature insight
Babadi & Sompolinsky 2014 (Neuron 83:1213, "Sparseness and expansion in sensory representations") show that a
random expansion with sparse thresholding decorrelates inputs but also amplifies small input differences: within-class
variability grows with the between-class separation, so a fixed random code does not raise the signal-to-noise of a
discrimination; only learned (Hebbian-tuned) expansion weights do. That is what this hour measured: true 4s scatter as
much as impostor 9s, the EMA radius widens to follow, and per-hit evidence I falls (latency 77 -> 139). In CLS terms
the hippocampus does not separate raw input; it indexes cortical representations (Teyler & DiScenna 1986 index theory;
McClelland, McNaughton & O'Reilly 1995 put the hippocampal input after neocortical processing). The selective
separation pdp needs should come from the cortex code that already discriminates 4 from 9, not from a random layer.

### Paths closed
- Random-expansion pattern separation of the store code (k-WTA at 5% and 2% of 2000 units). Random projection without
  sparsification is neutral. Any further store-geometry work must use a learned code.

### Hypotheses, ranked
1. **Architecture: store indexes the cortex code.** Keep prototypes on the FF network's top hidden layer (as the
   hippocampal index of cortical features) instead of pixels. Needs a gate_probe rerun that logs per-arrival hidden
   codes (s49, seeds 0-5, ~45-60 min on the box, OMP_NUM_THREADS=1). Prediction: z5 first post-swap remap <= 65 and
   swap >= z5 + 0.8, 0 false remaps. Kill: latency > 70 or store-only lifetime < pixel store - 1.0. Risk: the code
   drifts as the cortex learns, so prototypes go stale (would need re-encoding during sleep).
2. **Neuron: Hebbian-tuned expansion** (Babadi & Sompolinsky's remedy): expansion weights updated by a competitive
   Hebbian rule on the stream, then k-WTA. Prediction: within-class spread / between-class (4 vs 9) distance ratio
   below pixels by >= 10%; kill if not, before any life run.
3. **Data: label-noise world** (carried): 10% of 4s permanently labelled 9, no swap; z5 must not remap. Kill: > 1/run.
4. **Training: sleep in life.py with a slow cortex** (carried): lifetime >= 96.0 and returns >= 75; kill < 95.5.

## Loop 2026-10-31 16:00 UTC+8

### Experiment (`code_probe.py`, results/code_probe)
Offline fork-decider before any life rerun: store-like geometry (9 k-means prototypes per class) in six code spaces,
scored on test 4s and 9s, MNIST, online FF cortex at life.py settings, 20k normal arrivals then 2000 with 4<->9 swapped,
seeds 0-2. Impostor rate (share of 4s/9s nearest to a rival prototype): pixels 0.077, random k-WTA 0.077 (ratio worse,
0.943 vs 0.860, matching the 15:00 kill), competitive-Hebbian k-WTA 0.349 (killed, H2), FF layer 1 0.028, layer 2 0.027.
After the swap, code-space prototypes frozen at 20k go stale: layer 1 0.119, layer 2 0.619 (the top layer has moved 9
images to where 4 images were). H1 as pre-registered (stale impostor <= 0.75x pixels) is killed. Re-encoding the same
stored images through the post-swap cortex restores separation: layer 1 0.045 (0.58x pixels), layer 2 0.060.

### Literature insight
Rule, O'Leary & Harvey 2019 (Curr Opin Neurobiol 58:141, "Causes and consequences of representational drift"): when
a population code drifts, any downstream readout fixed to old activity patterns degrades unless it is continually
re-aligned; stable behaviour lives in a readout that tracks the drift, not in frozen patterns. In pdp the drift is
not slow noise but the cortex learning the swap itself: FF trains its hidden layers with the label in the input, so
even a label-neutral read of the top layer encodes the mapping. Teyler & DiScenna's index stores pointers to cortical
activity, and re-instatement runs through the current cortex. The pdp translation: the store keeps raw exemplars (the
pointer) and computes similarity through the current layer-1 code at query time, never storing the code.

### Paths closed
- Competitive-Hebbian expansion as store code (crude rule; ratio 0.951, impostor 0.349).
- Store prototypes kept in FF hidden-code space (stale after the swap; top layer worst).

### Hypotheses, ranked
1. **Architecture: re-read index.** Store keeps pixel prototypes and observations as now, but distances, radii and
   the rival test are computed on ff1(x) and ff1(proto) under the current cortex at each arrival (90 extra layer-1
   forwards per tick). Needs an in-loop life run (offline store replay on logs lacks the cortex). Prediction: s49 z5
   first post-swap remap <= 65 ticks and swap >= z5 + 0.8, 0 false remaps. Kill: latency > 70 or store-only lifetime
   < pixel store - 1.0. Risk: the EMA radius is in pixel units and must be re-set in code units.
2. **Neuron: layer 1 versus top layer as the index.** Layer 1 drifts less on a swap (stale 0.119 vs 0.619) because
   its input is the image, while the label weight is a small part of its fan-in. Prediction: with a fresh read, the
   ff1 index gives a lower remap latency than ff12 on s49. Kill if ff12 is better by > 5 ticks.
3. **Data: label-noise world** (carried): 10% of 4s permanently labelled 9; z5 must not remap. Kill: > 1 per run.
4. **Training: sleep in life.py with a slow cortex** (carried): lifetime >= 96.0 and returns >= 75; kill < 95.5.

## Loop 2026-10-31 17:00 UTC+8

### Experiment (`reread_store.py`, results/reread_store; diagnostic `reread_diag.py`, results/reread_store/DIAG.md)
16:00 hypothesis 1, in-loop life run (s49, seeds 0-5, never-sleeping life.py cortex, FF answers asserted equal to the
gate_probe_s49 logs, 0 mismatches). Store keeps pixel prototypes; answers, nearest-prototype, radii and the c1.5 rival
test are computed on L2-normalised FF layer-1 codes (rr1) or layers 1+2 (rr12) of the CURRENT cortex at each arrival,
before the cortex learns it. Pixel controls reproduce rival_s49 exactly (z5 swap 68.1, latency 77).
| Store | Store-only lifetime | Store-only returns | Swap (mq30_s2) | Lifetime (mq30_s2) | Returns (mq30_s2) | Store answers (mq30_s2) | Pre-swap false remaps / run | First post-swap remap |
|---|---|---|---|---|---|---|---|---|
| base | 91.4 | 84.6 | 69.7 | 96.0 | 71.8 | 1.1 | 1.0 | 59 |
| z5 | 91.4 | 84.9 | 68.1 | 96.0 | 72.0 | 1.1 | 0.0 | 77 |
| z5 rr1 | 93.1 | 86.3 | 67.7 | 95.5 | 76.9 | 2.1 | 2.7 | 71 |
| z5 rr12 | 93.5 | 83.6 | 68.3 | 95.4 | 75.7 | 2.1 | 2.3 | 65 |
- H1 killed: rr1 latency 71 (> 70), swap -0.4, and false remaps return (4 per run on 4 of 6 seeds). rr12 meets the
  latency bar (65) but swap is only +0.2 (need +0.8) with 2.3 false remaps per run.
- H2 killed: rr12 beats rr1 by 6 ticks (bar 5). Layer 1 is not the better index.
- Side finding worth keeping: the store as a classifier is better through the cortex (+1.7 to +2.1 store-only
  lifetime) and mq30_s2 returns rise +4.9, but the route gate, whose relative-margin threshold (0.2) was set in pixel
  units, now fires twice as often (2.1% vs 1.1%) and costs 0.5 lifetime.
- Diagnostic (seeds 0-1, first 300 ticks after the swap): in-radius conflicts 0.23 / 0.18 (rr1) vs 0.16 / 0.17 (pixels);
  per-hit evidence rises only slightly, so remap latency barely moves. The code separates *test* 4s from 9s
  (code_probe), but the post-swap stream's 4s labelled 9 still mostly fall outside the prototype's EMA radius.
- Process note: the first run had a bug (a global prototype-id column grew to ~4.5M, adding a time-growing offset to
  every observe() distance, so every conflict fell out of radius and no remap ever fired). Kept as
  results/reread_store_buggy_idoffset, not evidence. Answers were unaffected.

### Literature insight
Norman & O'Reilly 2003 (Psych Rev 110:611, CLS model of recognition memory): the hippocampus discriminates a lure from
a studied item only to the degree its input (the cortical code) keeps them apart, and its recall signal is all-or-none
on a threshold, so better input codes help recognition only where items sit near that threshold. The store's remap is
the same kind of thresholded recall (radius test, then count), and the cortex code moves 4s and 9s apart on average
without moving the swapped 4s inside the 4 prototypes' radius. Better input geometry bought recognition (store-only
accuracy +2) but not change detection. Remap latency is set by the radius rule, not by the code.

### Paths closed
- Re-read index as a remap accelerator (rr1, rr12). Store-code geometry is now closed for the s49 latency gap after
  four tries (core, posterior weight, random k-WTA, cortex code); z5 stays default.
- Layer 1 as the preferred index (H2).

### Hypotheses, ranked
1. **Architecture/routing: re-read store as the answering route only.** Keep pixel z5 for remapping (0 false remaps),
   but answer from the cortex-read store, with the mq30 relative-margin threshold re-set in code units so store
   answers return to about 1.1% (choose s on seeds 0-2, confirm on 3-5). Prediction: mq30 lifetime >= 96.0 and returns
   >= 75 on s49 and on bg002 seeds 3-8. Kill: lifetime < 96.0 or returns < 74. Cheap: reuse reread_store with a split
   store (remap in pixels, read in code).
2. **Neuron: radius in code units should be a quantile, not an EMA mean.** A per-prototype 90th-percentile radius of
   confirmation distance (in code space) admits the swapped 4s as in-radius conflicts. Prediction: rr12 latency <= 60
   with <= 1 false remap per run. Kill: false remaps > 1 per run.
3. **Data: label-noise world** (carried): 10% of 4s permanently labelled 9; z5 must not remap. Kill: > 1 per run.
4. **Training: sleep in life.py with a slow cortex** (carried): lifetime >= 96.0 and returns >= 75; kill < 95.5.

## Loop 2026-10-31 18:00 UTC+8

### State
17:00 commit 867524c had not reached GitHub (run ended before push); pushed, tar-synced to the VPS and published this
hour. Pre-registered and ran 17:00 hypothesis 1 (`split_store.py`, `results/split_store`, s49 seeds 0-5, ~15 min on
the box): one pixel z5 store remaps as before and answers three ways off the same prototypes (pixel, current FF
layer-1 code, layers 1+2), with the gate's store-margin threshold swept in code units (q 0.2/0.3 x s 0.1-0.8),
selected on seeds 0-2 and confirmed on 3-5.

### Result
- Killed. Held-out best code-read routes: sp1 mq20_s5 95.99 lifetime / 67.1 returns, sp12 mq20_s6 95.96 / 65.1 (bar
  96.0 and >= 74). Pixel mq20_s2 96.01 / 71.0.
- At matched firing rate the two readers lie on the same frontier (~1% fired: 96.0/72.0 pixel vs 95.9/71.9 code;
  ~2% fired: 95.7/75.8 vs 95.6/75.8). The 17:00 "+4.9 returns" was the gate firing 2.1% instead of 1.1%.
- The lifetime/returns trade is a single curve in fire rate for every store tried: returns buy lifetime one-for-one
  through extra store answers on FF-correct items. Remapping unchanged (0 false remaps, 77 ticks).

### Literature insight
Krogh & Vedelsby 1995 (NIPS 7, ensemble ambiguity decomposition): a combined predictor gains over its members only
through their disagreement, weighted by who is right when they disagree. Reading the store through the cortex raised
the store's own accuracy by making it agree with the cortex, which shrinks exactly the disagreement the gate lives on.
This is the CLS 1995 argument in ensemble terms: the hippocampal route is useful because its representation is
separate from the cortex's, so a store indexed by cortical similarity is a worse complement even when it is a better
classifier. Metric to watch from now on: accuracy of store answers on FF-wrong items, not store-only lifetime.

### Paths closed
- Cortex-read store as the answering route (sp1, sp12), at any gate threshold. With the 17:00 result, reading the
  store through the cortex is closed for both remapping and answering. Default stays a25w250h2k2_c1.5_uinf_z5 with
  pixel mq30_s2.

### Hypotheses, ranked
1. **Training/data: the frontier is set by FF errors the store can fix.** Log, per route, store accuracy on FF-wrong
   arrivals (complement accuracy) and FF-wrong rate by tick since return. Prediction: on s49 the pixel store fixes
   >= 40% of FF errors in the first 300 ticks after a return and < 15% elsewhere, so a gate that keys on
   time-since-return (not margin) beats mq30_s2 by >= +2 returns at lifetime >= 96.0. Kill: complement accuracy flat
   in time since return (ratio < 1.5). Cheap: offline on gate_probe_s49 rows plus split_store answers (save S per tick).
2. **Neuron: radius as a quantile** (carried, pixel units now): a per-prototype 90th-percentile confirmation radius
   admits swapped 4s as in-radius conflicts. Prediction: latency <= 65 with <= 1 false remap per run. Kill: > 1.
3. **Data: label-noise world** (carried): 10% of 4s permanently labelled 9; z5 must not remap. Kill: > 1 per run.
4. **Architecture/training: sleep with a slow cortex in life.py** (carried): lifetime >= 96.0 and returns >= 75;
   kill < 95.5.

## Loop 2026-10-31 19:00 UTC+8

### State
18:00 commit fa8997a on GitHub, VPS and site. Pre-registered and ran 18:00 hypothesis 1 (`complement_probe.py`,
`results/complement_probe`, ~4 min on the box): rerun the s49 world with the default pixel z5 store (the store does not
depend on the cortex, so no FF is needed) and join its answers to the logged FF answers from `results/gate_probe_s49`,
seeds 0-5. Measure complement accuracy (store right when FF is wrong) by ticks since the true class's return start,
and try a causal recency gate: also let the store answer when it disagrees and its label s, or the FF label f, came
back from >= 2000 ticks away less than T ticks ago (T and margin s chosen on seeds 0-2, confirmed on 3-5).

### Result
- Diagnostic killed as stated (ratio 1.48, bar 1.5): complement accuracy 70.5% in the first 300 ticks after a return
  vs 47.8% later. The store's harm (disagreeing with a correct FF) is flat at 6-10% in every bin. What moves is the
  FF error rate: 45.8% in the first 100 ticks after a return, 18.6% at 100-300, 3.4% after 3000.
- Gate is a lead: selected rt2000_s2 gives held-out 95.96 / 73.8 vs mq30_s2 95.97 / 71.5 (+2.3 returns). Against the
  mq frontier at matched firing (1.17%) it is +1.6 returns; rt2000_s1 matches mq30_s1 returns at 0.76% less firing and
  +0.20 lifetime. This is the first change since 11:05 that moves the lifetime/returns frontier outward instead of
  sliding along it. Not the default yet: the literal bar was lifetime >= 96.0, missed by 0.04 with the baseline at 95.97.

### Literature insight
Yu & Dayan 2005 (Neuron 46:681, "Uncertainty, neuromodulation, and attention"): expected uncertainty, a known
unreliability of the current model in a known context, should down-weight top-down predictions in favour of other
evidence, and ACh is their proposed signal for it. A class coming back after a long absence is exactly that known
context: the cortex has drifted on it (FF wrong 46% of the time) while the store has not. The gate should take this
prior from the stream, not infer it from FF margin alone. Ties to Hasselmo 2006 (ACh sets encoding vs retrieval),
already in notes: recency of a return is a cheap, causal proxy for the ACh level.

### Paths closed
- Complement accuracy as the thing that varies with time since return. The store's reliability is flat; only the
  cortex's prior error changes. Gate designs should model P(FF wrong), not P(store right).

### Hypotheses, ranked
1. **Architecture/routing: confirm the recency gate off its selection world.** Rerun `complement_probe.py` on
   bg002 seeds 6-8 (logs in `results/gate_probe_s678`) and on the bg 0.2 world with rt2000_s2 frozen. Prediction:
   >= +1.0 returns above the mq frontier at matched firing, lifetime within 0.05 of mq30_s2. Kill: < +0.5.
2. **Neuron/gating: a decaying recency trace instead of a hard window.** Margin threshold s(t) = s_lo + (s_hi - s_lo)
   (1 - exp(-age/tau)), age since the label's return. Prediction: beats rt2000_s2 by >= +0.5 returns at equal
   lifetime. Kill: no gain at any tau in (300, 1000, 3000).
3. **Data: label-noise world** (carried): 10% of 4s permanently labelled 9; z5 must not remap, and the recency gate
   must not fire on noise. Kill: > 1 false remap per run.
4. **Training: return-triggered replay.** Since FF error is concentrated right after a return, a short targeted
   sleep (replay of that class's store prototypes) at return onset should cut the 45.8% FF error. Prediction: FF error
   in the first 100 ticks <= 30% at lifetime >= 95.9. Kill: >= 40%.

## Loop 2026-10-31 20:00 UTC+8

### State
19:00 commit 677f97d on GitHub, VPS and site. Box idle, VPS idle. Pre-registered 19:00 hypotheses 1 and 2 in
`confirm_recency.py` (`results/confirm_recency`, ~5 min on the box, store answers cached as Z_*.npy, not in git):
rerun the default pixel z5 store on s49 seeds 0-5, bg002 seeds 6-8 and bg 0.2 seeds 0-5, join to the logged FF answers
(`results/gate_probe_s49`, `_s678`, `_bg02`), evaluate rt2000_s2 frozen and a decaying-trace threshold selected on s49
0-2. After the first pass the selected trace also raised lifetime on bg 0.2, where nothing ever returns, so a
recency-free control c{s} (mq30_s2 or store disagrees with margin >= s) and p{s} (store margin alone) were added and
put into the frontier before reading the rerun.

### Result
| Held-out group | mq30_s2 | rt2000_s2 | c3 (control) | dk3000_1_4 |
|---|---|---|---|---|
| s49 3-5, lifetime / returns / fired | 95.97 / 71.5 / 1.09 | 95.96 / 73.8 / 1.16 | 96.12 / 74.8 / 1.48 | 96.01 / 76.2 / 1.43 |
| bg002 6-8 | 95.69 / 71.4 / 1.16 | 95.68 / 73.5 / 1.25 | 95.86 / 75.2 / 1.50 | 95.74 / 76.3 / 1.52 |
| bg 0.2 0-5 (no returns) | 94.38 / - / 1.09 | 94.38 / - / 1.09 | 94.70 / - / 1.56 | 94.58 / - / 1.32 |

- H1 passes as literally stated (+1.5 over the mq frontier on bg002) but the mq frontier was incomplete. c3 dominates
  mq30_s2 and rt2000_s2 on lifetime and returns everywhere; swap first 300 costs about 1 point.
- H2 killed: recency traces (exponential or hyperbolic) give +1.1-1.4 returns over c3 for -0.11 lifetime at matched
  firing. Recency is a knob along the frontier, not past it.
- c2 = p2 exactly: mq30_s2 is a strict subset of "store disagrees with margin >= 0.2". The FF-margin quantile gate
  (12:05-19:00 default) mainly discarded good store answers.

### Literature insight
Nassar, Wilson, Heasly & Gold 2010 (J Neurosci 30:12366, approximately Bayesian delta rule): in a change-point world
the weight on new evidence is set by run length since the last change, roughly 1/(run length + 1), plus change-point
probability computed from the size of the surprise. Their model has two terms and our test separated them: the
surprise term (here the store's margin when it disagrees with the cortex) carried the gain, while the run-length term
(recency of a return) did nothing that a stricter surprise threshold could not. Read with Yu & Dayan 2005 from 19:00:
expected uncertainty matters only when it is not already visible in the evidence; in this stream the store's margin
already shows it.

### Paths closed
- Recency gating of the store route (hard window, exponential and hyperbolic traces). 19:00's outward-move claim
  withdrawn.
- FF-margin quantile as a necessary part of the router.

### Hypotheses, ranked
1. **Architecture/routing: c3 as the in-loop default.** Run life.py (or the swap/remap harness) with c3 replacing
   mq30_s2 on s49 and bg002, seeds 0-8. Prediction: lifetime +0.1 or more and returns +3 or more over mq30_s2, swap
   first 300 within 1.5. Kill: lifetime gain < 0.05 or swap loss > 2.
2. **Neuron/gating: calibrated store margin.** Replace the fixed relative-margin threshold with a running estimate of
   P(store right | disagree, margin bin) learned from the delayed label (a per-bin counter, purely local). Prediction:
   beats the best c{s} by >= +0.5 returns at matched lifetime, because the threshold tracks store drift after swaps.
   Kill: within 0.3 of the best c{s}.
3. **Data: label-noise world** (carried): 10% of 4s permanently labelled 9. c3 lets the store answer more often, so
   the risk is the store voicing noise labels. Kill: lifetime below mq30_s2 on that world.
4. **Training: return-triggered replay** (carried from 19:00): targeted sleep on a class's store prototypes at return
   onset. Prediction: FF error in the first 100 ticks after a return <= 30% (from 45.8%) at lifetime >= 95.9.

## Loop 2026-10-31 21:00 UTC+8

### State
20:00 closed recency gating and made c3 (store answers when it disagrees with relative margin >= 0.3, or on
mq30_s2) the router candidate. Its hypothesis 2 (calibrated store margin) is offline-testable on the cached runs, so it
ran first. VPS idle; run on the box (~9 s, cached store/FF logs).

### Result (`arbiter.py`, results/arbiter)
- Pre-registered: an online Beta-count table of P(store right) and P(FF right) on disagreements, binned by FF margin x
  store margin (quintile edges from s49 0-2); store answers when p_store > p_ff + d. Variant and d chosen on s49 0-2;
  must beat the c{s} frontier (s 0.10-0.80 step 0.05) by >= +0.5 returns at matched firing with lifetime >= c3 - 0.02.
- Selected j+0.0b. Held-out s49 3-5: 96.18 / 77.8 vs c3 96.12 / 74.8, +1.5 above frontier. bg002 6-8: 95.90 / 78.2
  vs 95.86 / 75.2, +1.9. bg 0.2 lifetime 94.99 vs 94.70. **Passes.**
- Without the mq30_s2 base the arbiter gives the highest lifetime seen in every world (96.27 / 96.04 / 95.04), more
  than any c{s}. Store-margin-only calibration (s+0.0b) moves returns but not lifetime: the FF margin is useful as
  context for when to trust the store, not as a gate.
- Cost: swap first 300 about 1 point below c3; store answers 2.7% vs 1.5%.

### Literature insight
Daw, Niv & Dayan 2005 (Nat Neurosci 8:1704): two controllers (model-based prefrontal, habitual striatal) are
arbitrated by each one's running estimate of its own uncertainty, which depends on where in the task you are. Our
version is the same move with two memories: the store and cortex each have a reliability that depends on the joint
state (how sure the cortex is, how sure the store is), and the arbiter learns that table from the labels that arrive
anyway. Fixed thresholds on either margin (12:05-20:00) were each a slice of this table. CLS says the systems are
complementary; Daw et al. say how to choose between complementary systems moment by moment.

### Paths closed
- Fixed-threshold store-margin routing (c{s}) as the best router: dominated offline by the calibrated arbiter.

### Hypotheses, ranked
1. **Data: label-noise world (kill test for the arbiter).** 10% of 4s permanently labelled 9. A per-bin counter could
   learn to trust the store where it memorised noise. Prediction: j+0.0b lifetime >= c3 on that world. Kill: below
   mq30_s2.
2. **Architecture: arbiter in-loop** (replaces 20:00's c3 in-loop test). life.py with j+0.0b routing, s49 and bg002,
   seeds 0-8. Prediction: lifetime >= +0.15 and returns >= +5 over mq30_s2, swap first 300 within 1.5. Kill: lifetime
   gain < 0.05.
3. **Neuron/gating: forgetting counters.** Beta counts with exponential decay (half-life 2000 ticks) so reliability
   tracks swaps. Prediction: recovers the 1-point swap-first-300 cost at equal lifetime. Kill: swap gain < 0.5.
4. **Training: return-triggered replay** (carried): targeted sleep on a class's store prototypes at return onset.

## Loop 2026-10-31 22:00 UTC+8

### State
21:00 made the calibrated arbiter j+0.0b the router candidate; its cost was about 1 point of swap first 300. Its
hypothesis 3 (forgetting counters) is offline-testable on the cached runs, so it ran first. VPS idle; run on the box
(~10 s, cached store/FF logs).

### Result (`arbiter_decay.py`, results/arbiter_decay)
- Pre-registered: Beta counts decay with half-life HL (500-8000 ticks), and a class-keyed variant (table also keyed by
  the store's predicted class). Selected on s49 0-2 for best swap first 300 with lifetime >= j+0.0b - 0.02 and returns
  >= j+0.0b - 0.5. Prediction on held-out: swap >= +0.5 under the same bars.
- Selected d2000b. Held-out swap +1.3 (s49 3-5) and +0.9 (bg002 6-8) at equal lifetime, but returns -2.0 and -1.9
  (75.8 vs 77.8, 76.3 vs 78.2), consistent across seeds. **Fails.** Every half-life is on the same trade.
- Class-keyed table: lower lifetime everywhere (s49 96.11 vs 96.18, bg 0.2 94.84 vs 94.99), no swap gain. **Killed.**
- Side note: d2000b dominates c3 on lifetime, returns and swap in both held-out swap worlds, so it is a valid point
  if swap is weighted over returns. j+0.0b stays the candidate.

### Literature insight
Behrens, Woolrich, Walton & Rushworth 2007 (Nat Neurosci 10:1214): people set their learning rate from an estimate
of the environment's volatility, high when contingencies change and low when they hold. A fixed half-life is the
volatility-blind version, and our numbers show its cost precisely: the evidence that makes the store trustworthy at a
return is old evidence from a stable stretch, and a clock forgets it. The project already has a change detector (the
z5 extinction override that remaps store slots on a swap); reliability should forget when it fires, and only then.

### Paths closed
- Clock-decayed arbiter counts (any half-life 500-8000) as an outward move: a slide along returns vs swap.
- Class-keyed arbiter table: dilutes evidence, lowers lifetime.

### Hypotheses, ranked
1. **Neuron/gating: change-gated forgetting.** Reset (or decay by 0.5) the arbiter counts only when z5 remaps a store
   slot, and only in rows touched by that class's disagreements. Prediction: keeps j+0.0b returns (within 0.5) and
   gains >= +0.5 swap first 300 on both held-out swap worlds. Kill: either bar missed. Needs remap events from the
   cached runs (check whether Z logs carry remap ticks; else re-run gate_probe with logging).
2. **Data: label-noise world (kill test for the arbiter, carried).** 10% of 4s permanently labelled 9. Prediction:
   j+0.0b lifetime >= c3. Kill: below mq30_s2.
3. **Architecture: arbiter in-loop (carried).** life.py with j+0.0b routing, s49 and bg002, seeds 0-8. Prediction:
   lifetime >= +0.15 and returns >= +5 over mq30_s2, swap first 300 within 1.5. Kill: lifetime gain < 0.05.
4. **Training: return-triggered replay (carried).** Targeted sleep on a class's store prototypes at return onset.

## Loop 2026-10-31 23:00 UTC+8: change-gated forgetting for the arbiter (`change_arbiter.py`)

### Experiment (falsification)
- Pre-registered two causal change signals, neither changing what store or cortex learns. H4a: reset ('rG') or halve
  ('rH') the arbiter's joint Beta table whenever the z5 store remaps a slot (remap ticks recovered by a store-only
  replay that asserts every answer equals the cached Z; 15-70 remaps per s49 run, 0 before the swap). H4b: a
  class-local surprise veto; per store-predicted class, a streak of disagreements where the store was wrong and the FF
  right; at streak >= m (2, 3, 5) the arbiter may not hand that class to the store until the store is right on a
  disagreement ('v') or the class is remapped ('w'); 'B' variants also silence mq30_s2. Same bars as 22:00.
- No variant passed the selection bars on s49 0-2. **Both killed.** Remap-triggered reset does nothing (held-out s49
  swap -0.6, bg002 +0.2, returns within 0.8): the remap comes after the ~75 ticks in which the store is wrong, as the
  mechanism prior said.
- The surprise veto is the most interesting failure so far: v2b gives the highest lifetime yet (s49 3-5 96.33 vs
  96.18, bg002 96.04 vs 95.90, bg 0.2 equal) and +2.3 / +1.0 swap first 300, but costs 3.4-4.0 returns (74.4 vs 77.8,
  74.2 vs 78.2). The 'B' variants push swap to +3 to +5 but returns fall to ~64.
- Diagnostic (not a test, s49 0-5, v2): the veto blocks ~2700-3250 disagreements per run, ~55% of them before the
  swap, so it is mostly detecting stable classes the store gets wrong, not change. On returns 74-84% of blocked
  disagreements are ones the store had right: the streak went stale while the class was away.

### Literature insight
Behrens et al. 2007 set the learning rate from volatility; Gershman, Blei & Niv 2010 add that the cue for a new latent
cause is a context break, not error alone. Our veto counts errors but never notices that a class's absence and
return is itself a context break, so evidence collected before the class left is applied after it returns, which is
exactly where the store's old knowledge is right and the cortex's is not. Surprise should be scoped to the current
episode of a class, ended by its departure.

### Paths closed
- Store-remap-triggered reset or halving of arbiter counts (acts after the error window).
- Error-streak veto with release only on store-right disagreement or remap, as a returns-neutral swap fix.

### Hypotheses, ranked
1. **Neuron/gating: episode-scoped surprise veto.** As v2b, but a class's streak is cleared when its label returns
   after RET_GAP (and optionally when the store is right on any arrival of that class, agreements included).
   Prediction (select m on s49 0-2): held-out returns within 0.5 of j+0.0b, swap >= +0.5, lifetime >= j+0.0b. Kill:
   returns loss > 0.5 or swap < +0.5 on either held-out swap world.
2. **Data: label-noise world (carried).** 10% of 4s permanently labelled 9; the veto and arbiter both see persistent
   store errors there, which is where a stable-error veto should help and a change detector should not fire.
3. **Architecture: arbiter in-loop (carried).** life.py with j+0.0b (or the winner of 1), s49 and bg002, seeds 0-8.
4. **Training: return-triggered replay (carried).** Targeted sleep on a class's store prototypes at return onset.

## Loop 2026-11-01 00:00 UTC+8: episode-scoped and pair-keyed surprise vetoes (`episode_veto.py`)

### Experiment (falsification)
- Pre-registered H5 (23:00 hypothesis 1): the v{m}b error-streak veto plus an episode reset. 'e' clears class c's
  streak before answering when label c has not been seen for >= RET_GAP ticks (causal); 'a' also releases on a
  store-right agreement; 'ea' both; m in 2, 3, 5. Then H5b, pre-registered after H5 failed: streaks keyed by the
  (store answer, FF answer) pair, so only a run where the FF keeps naming the same rival label vetoes ('p', 'pe').
  Same bars as 23:00 (held-out returns within 0.5 of j+0.0b, swap first 300 >= +0.5, lifetime >= -0.02).
  Pure re-scoring of the cached confirm_recency logs, about 12 s on the box.
- **Both killed.** Selected e2b: held-out s49 3-5 96.33 / 75.3 / swap +2.2, bg002 6-8 96.04 / 75.3 / +1.0, so the
  episode reset buys back only about 1 of the 3.4-4.0 lost returns. Selected p2b: 96.31 / 75.7 / +1.8 and
  96.02 / 75.2 / +0.7. Pair keys plus reset at m = 5 (pe5b) reach returns -0.4 on s49 but the swap gain falls to
  +1.0 / +0.0. Every variant (v, e, a, ea, p, pe, all m) sits on one trade line of roughly 1.5 returns per swap point.
- Diagnostic: even with the reset, 84-96% of disagreements blocked during returns are ones the store had right. The
  streaks are not stale; they are rebuilt inside the return episode by the few cases where the FF is right. A hard
  "m strikes" switch fires on those because it ignores how rarely the store is wrong at a return.

### Literature insight
Adams & MacKay 2007 (Bayesian online changepoint detection, arXiv:0710.3742): a change detector should carry a
run-length posterior whose update weighs each surprise against the predictive probability under the current regime
and a hazard rate. A streak counter is that posterior with the predictive term thrown away: two store errors count the
same whether the store is usually right 95% of the time on that bin (a return) or 50% (a fresh swap). The arbiter's
joint table already holds that predictive probability, so the error evidence should enter it as a likelihood ratio,
not as a veto beside it.

### Paths closed
- Hard error-streak vetoes on the store, in every form tried: per class, per (store, FF) pair, with episode reset on
  return, with release on agreement. They trade about 1.5 returns per swap point and never pass.

### Hypotheses, ranked
1. **Architecture: arbiter in-loop (promoted).** life.py with j+0.0b as the answering route, s49 and bg002, seeds
   0-8; prediction: lifetime and returns within 0.1 / 0.5 of the post-hoc scores (the route does not change what
   either system learns, so any gap means the replay assumption is wrong). Kill: in-loop returns > 1 below post-hoc.
2. **Data: label-noise world (carried).** 10% of 4s permanently labelled 9. Prediction: j+0.0b keeps lifetime
   >= c3; a store that remaps on noise should lose. This is where the router could fail for a reason we have not seen.
3. **Neuron/gating: likelihood-ratio arbiter.** Multiply the joint-bin odds by the BOCPD run-length posterior for
   "store wrong on class c" with hazard 1/ (median gap between swaps); a soft version of the veto that discounts
   errors in bins where the store is usually right. Kill: returns loss > 0.5 or swap gain < 0.5 held-out.
4. **Training: return-triggered replay (carried).** Targeted sleep on a class's store prototypes at return onset.

## Loop 2026-11-01 01:00 UTC+8

### Experiment: soft surprise in the arbiter (`soft_arbiter.py`, results/soft_arbiter)
- Tests 00:00's hypothesis 3 in its cheapest form. Each store-predicted class keeps its own disagreement counts
  (store right, FF right), shrunk toward the j+0.0b joint-bin posterior with prior strength K in 2-20: the class
  evidence moves the bin's prediction in proportion to how surprising it is, instead of a hard veto. Variants: whole
  life (h), reset when the class returns (he), half-life 500 ticks (hd). Pre-registered bars as for the vetoes.
- **Killed.** Selected h2b fails every bar (held-out s49 3-5 95.84 / 74.4 / swap +2.4; bg002 6-8 95.56 / 74.3 / -0.3;
  bg 0.2 lifetime -0.8). The best on returns, hd20b, gives s49 96.36 / 76.5 / +1.4 and bg002 96.07 / 75.6 / +1.1:
  about 1 return per swap point, a slightly better slope than the vetoes but the same line.
- Side result: the short half-life class counts (hd) give the highest lifetimes yet (s49 3-5 96.42, bg002 96.07,
  bg 0.2 95.12 vs j+0.0b 96.18 / 95.90 / 94.99). Lifetime is now dominated by stable-period disagreements, which is
  why returns, not lifetime, must stay the gate for router changes.

### Literature insight
Courville, Daw & Touretzky 2006 (Bayesian theories of conditioning in a changing world, TICS 10:294): change is
inferred from outcomes that are surprising under the current model, and a learner can only infer it from evidence
that differs between "the world changed" and "I was wrong". At the arbiter a swap and a return produce the same
evidence: the two systems disagree on a class and the system that was right before is now wrong. Soft or hard,
class-local reliability cannot tell them apart in the first ~75 ticks, so every variant slides on one line. What
does differ is inside the store: on a swap the label contradicts a confident, old prototype; on a return the store's
prototype is confirmed while the FF's is stale. That signal lives in the store's remap, not in the router.

### Paths closed
- Class-local reliability in the arbiter, hard or soft (vetoes 23:00-00:00, partial pooling now). Swap speed has to
  come from the store's remap; the router should stay j+0.0b.

### Hypotheses, ranked
1. **Architecture: arbiter in-loop (carried, top).** life.py with j+0.0b answering; prediction: lifetime and returns
   within 0.1 / 0.5 of post-hoc. Kill: in-loop returns > 1 below post-hoc.
2. **Neuron/gating: confidence-weighted remap.** The store remaps a prototype after k contradicting labels; weight
   each contradiction by the prototype's own margin on that item (a confident prototype contradicted is strong swap
   evidence, a marginal one is not). Prediction: s49 first-remap latency < 60 ticks (now ~75) with zero false remaps
   on bg 0.2. Kill: latency drop < 10 ticks or any false remap.
3. **Data: label-noise world (carried).** 10% of 4s permanently labelled 9; j+0.0b should keep lifetime >= c3.
4. **Training: return-triggered replay (carried).** Targeted sleep on a class's store prototypes at return onset.

## Loop 2026-11-01 02:00 UTC+8

### Experiment: confidence-weighted store remap (`conf_remap.py`, `conf_remap_confirm.py`; results/conf_remap, results/conf_remap_sweep, results/conf_remap_confirm)
- Tests 01:00's hypothesis 2 (swap speed must come from the store's remap, not the router). In the default store
  (a25w250h2k2_c1.5_uinf_z5) a contradiction that passes the radius and rival tests counts twice toward the H=2 hot
  threshold when it is confident: 'm{q}' store relative margin on the item >= q, 'r{p}' distance <= p * radius.
  Every store is routed by j+0.0b (same bin edges) OR mq30_s2 on the cached FF logs; the z5 control reproduces the
  confirm_recency store answers exactly. Pre-registered bars vs z5: latency <= -10 ticks, swap first 300 >= +0.5,
  returns >= -0.5, lifetime >= -0.02, pre-swap false remaps <= +0.2 per run.
- **Killed as pre-registered, but a lead.** Selected m0.3 (s49 0-2: latency 61 vs 74, swap 71.0 vs 66.6, returns
  76.8 vs 76.8). Held-out s49 3-5: 96.18 / 77.7 / swap 68.8 vs 96.18 / 77.8 / 66.0, latency 63 vs 79; bg002 6-8:
  95.90 / 78.1 / 72.6 vs 95.90 / 78.2 / 70.1, latency 40 vs 59; bg 0.2: lifetime 95.06 vs 94.99, swap 73.7 vs 69.4.
  It fails only the false-remap bar: s49 seed 4 has one transient pair (5->8 at t 33550, 8->5 293 ticks later; 4
  remap events = 1.33 per run in the group). Radius-weighted (r) and m0.5 variants barely fire.
- Post-hoc neighbour sweep (not a test): m0.35 meets every H7 bar on the original groups (s49 3-5 +2.3 swap, -0.3
  returns, latency 63 vs 79; bg002 6-8 +1.7 / -0.1 / 48 vs 59; 0 false remaps). Fresh-seed confirmation, bg 0.02
  seeds 3-5 and 9-11 (FF logs never used to choose q), pre-registered: m0.35 96.28 / 74.3 / swap 71.9 vs z5
  96.26 / 74.4 / 69.4, latency 65 vs 75; m0.3 96.29 / 74.4 / 73.1, latency 66. Both fail only the false-remap bar
  again (seed 10: 4 events for both; m0.3 also seed 4).
- What this establishes: the first change since 18:00 that moves swap first 300 (+2.4 to +4.3 points across 4 groups,
  21 runs) with no return cost and no lifetime cost. Router-side changes all paid ~1-1.5 returns per swap point;
  this pays ~0. The remaining problem is a rare transient false remap (2 of 21 runs), which costs nothing measurable
  here but would be a real error in a label-noise world.

### Literature insight
Kumaran & Maguire 2006 (An unexpected sequence of events: mismatch detection in the human hippocampus, PLoS Biol
4:e424): hippocampal activity peaks not for novelty per se but when a prediction retrieved from past experience is
violated by the input, i.e. the mismatch signal is gated by the strength of the retrieved expectation (entorhinal
cortex, by contrast, responds to novelty generally). The store's remap counted every in-radius contradiction alike;
weighting a contradiction by how confidently the store predicted the old label is the comparator they describe, and
it is what bought swap speed here without touching returns.

### Paths closed
- Radius-depth weighting of contradictions (r0.5, r0.7): too rare or inconsistent to change latency; margin, not
  depth within one prototype's radius, is the informative quantity.

### Hypotheses, ranked
1. **Neuron/gating: mismatch credit needs an expectation AND no alternative.** Give the double credit only when the
   store's margin >= 0.3 and the contradicting label's nearest prototype is far (d_y > 2 d, stricter than the c1.5
   rival test). Prediction: keeps >= 80% of m0.3's swap gain on all 5 groups with 0 false remaps on 21 runs. Kill:
   any false remap or swap gain < +1.5 on two groups.
2. **Data: label-noise world (now urgent).** 10% of 4s permanently labelled 9. Confident contradictions are exactly
   what label noise produces on a well-learned class, so m0.3 should be the store most likely to remap wrongly.
   Prediction: m0.3 remaps on noise in <= 1 run of 6; kill: > 1 run or lifetime < z5 - 0.1.
3. **Architecture: arbiter in-loop (carried).** life.py with j+0.0b answering and the winner of 1 as the store;
   lifetime / returns within 0.1 / 0.5 of post-hoc. Kill: in-loop returns > 1 below post-hoc.
4. **Training: return-triggered replay (carried).** Targeted sleep on a class's store prototypes at return onset.

## Loop 2026-11-01 03:00 UTC+8

### Experiment: a count floor on confidence-weighted remap (`guard_remap.py`, `guard_remap_confirm.py`; results/guard_remap, results/guard_remap_confirm)
- Tests 02:00's open problem: m0.3 (a contradiction counts twice toward remap when the store's relative margin >= 0.3)
  gave +2.4 to +4.3 swap first 300 with returns flat, but showed transient false remaps in 3 of 21 cached runs.
  With H=2, K=2 a doubled contradiction makes a prototype hot by itself, so a remap can rest on 2 arrivals.
  H8 guard 'n{N}': a remap s->y is allowed only if >= N distinct in-radius, rival-passing contradictions s->y
  arrived in the last 250 ticks. Selection on the 21 cached runs (zero false remaps, lowest latency); test on fresh
  store-only seeds 12-17 in s49 (4<->9) and bg002 (2<->3) plus routed bars on the cached runs.
- **Killed.** Selected m0.3n3 (cached: 0 false remaps, latency 60 vs 73; swap +2.3 / +2.4 / +3.7 / +3.6 on s49 3-5,
  bg002 6-8, bg002a, bg002b, returns -0.3 to +0.2). Fresh seeds 12-17: 0.50 false remaps per run (seed 13, both
  worlds). Unguarded m0.3 had 3.33 per run there (seed 14: 12 events), so false remaps are far commoner than the
  21 cached runs suggested.
- Post-hoc m0.3n4 met every H8 bar on seeds 12-17 (0 false remaps, latency 60 vs 72). Confirmation on 24 more fresh
  seeds (18-29, both worlds), pre-registered: **killed**. Latency 52.5 vs 71.4 (-19), and 22 of 24 runs are clean,
  but seed 26 (the same pre-swap stream in both worlds) has 68 pre-swap remap events: 34 4->9 and 34 9->4 from
  t 19361 until the swap at 38400. m0.35n4 shows the same runaway (32 events).
- What this establishes: the failure is not too few observations but no hysteresis. Once a confident 4/9 prototype
  is moved, the moved prototype is itself confidently matched by the other class's items, its contradictions are
  doubled, and the pair ping-pongs indefinitely. A count floor only delays the first flip. The default z5 store has 0
  false remaps on all 57 runs (21 cached + 36 fresh).

### Literature insight
Wills, Lever, Cacucci, Burgess & O'Keefe 2005 (Attractor dynamics in the hippocampal representation of the local
environment, Science 308:873-876): as a box is morphed step by step between a square and a circle, CA1 place cells
switch between the two maps abruptly and coherently, and which map an intermediate shape gets depends on the path
taken (the representation shows hysteresis, as expected of attractor states). Our store's remap is a switch with
no hysteresis: the evidence needed to move a prototype back is the same as the evidence that moved it. Seed 26's
4<->9 ping-pong is what a hysteresis-free switch does on confusable classes, and making each contradiction count
more just makes the flips faster.

### Paths closed
- Count floors on weighted remap (n3, n4, any q in 0.3-0.35): they delay but cannot stop runaway oscillation.
- FF-agreement gating of the doubled credit (m0.3f, exploratory): 0 false remaps on cached runs but keeps only about
  a third of the swap gain (+0.3 to +1.8), because the cortex has not learned the new label in the first ~60 ticks.

### Hypotheses, ranked
1. **Architecture/gating: remap hysteresis.** A prototype moved within the last T ticks (T in 2000, 8000) gets no
   double credit and needs H+1 contradictions to move again, so a reverse remap costs more than the forward one.
   Prediction on fresh seeds 30-41 (both worlds) plus seed 26: m0.3 with hysteresis has <= z5 + 0.2 false remaps per
   run, no run with > 4 events, and latency <= z5 - 8. Kill: any runaway (> 4 events) or latency gain < 8.
2. **Neuron: credit only for one-sided confidence.** Double only if the contradicting label's own prototypes are far
   (d_y > 2 d), so 4/9 border items never get double credit. Prediction: seed 26 clean, keeps >= 60% of m0.3's swap
   gain. Kill: runaway on any of 36 fresh seeds.
3. **Data: label-noise world.** 10% of 4s permanently labelled 9: the case that hysteresis must survive. Run after 1.
4. **Training: arbiter in-loop (carried).** Blocked on choosing the store; keep z5 as the in-loop store until 1 passes.

## Loop 2026-11-01 04:00 UTC+8

### Experiment: remap hysteresis (`hyst_remap.py`; results/hyst_remap)
- Tests 03:00's hypothesis 1. H9 'h{T}': a prototype moved within the last T ticks (2000 or inf) gets no doubled
  credit and needs H+1=3 contradictions to become hot; 'b' applies this only to a move back to its old label.
  Variants m0.3h2000, m0.3hinf, m0.3binf, m0.3n4hinf; controls z5, m0.3, m0.3n4. Selection on dev fresh seeds 12-29
  (incl. seed 26), test on held-out fresh seeds 30-47, store-only, both s49f (4<->9) and bg002f (2<->3).
- **Passed as pre-registered, but the test did not discriminate.** Selected m0.3n4hinf. Held-out 30-47: 0 false
  remaps on 36 runs, first remap 65.7 vs z5 78.6 ticks (-12.9). But plain m0.3n4 does exactly the same on held-out
  (0 false remaps, 65.7), so hysteresis contributed nothing there; the held-out set simply has no seed-26-like stream.
- Caveat found while reading the logs: s49f and bg002f with the same seed share the whole pre-swap stream (identical
  pre-swap remap events), so "36 runs" is 18 independent pre-swap streams. Earlier "57 runs" counts for z5 overstate
  independence in the same way. Future false-remap bars should count streams, not runs.
- What hysteresis does on dev seed 26 (s49f, no mislabels of 4/9 in the window): m0.3n4 ping-pongs 34x 4->9 and
  34x 9->4 from t 19361 until the swap at 38400; m0.3n4hinf flips 12/10 times and settles by t 22862. Hysteresis
  damps and terminates the oscillation (3.5k vs 19k ticks) but does not prevent it: the first move is the error.
  On other dev streams hysteresis cuts m0.3's events (seed 14: 12 -> 4, seed 33: 14 -> 4) but never the first pair.
- Status: m0.3n4hinf is the store candidate (latency -13 to -16 ticks, 0 false remaps on 18 held-out streams), but
  not the default: one dev stream in 18 still has 22 false remap events. z5 stays default.

### Literature insight
Jezek, Henriksen, Treves, Moser & Moser 2011 (Theta-paced flickering between place-cell maps in the hippocampus,
Nature 478:246-249): after an instant switch of the light cues between two familiar rooms, CA3 does not jump
cleanly to the new map; for a while it flickers between the two maps on single theta cycles (~125 ms), then
settles. The flicker is in retrieval, not in the stored maps: neither map is rewritten by a flicker. Our store
conflates the two. Every remap is a structural rewrite of a prototype's label, so a retrieval-level ambiguity
between confusable 4s and 9s becomes storage-level ping-pong. Hysteresis bounds how long the rewriting goes on;
Jezek et al. suggest the fix sits earlier: let the answer flicker (both labels live), and rewrite storage only
once one label wins consistently.

### Paths closed
- Hysteresis as a cure for false remaps (any T, forward or back-only): it terminates runaway oscillation (68 -> 22
  events on seed 26, 73 -> 18 for unfloored m0.3) but cannot stop the first false move, and without the n4 floor
  m0.3 + hysteresis still false-remaps on 8 of 18 streams.

### Hypotheses, ranked
1. **Architecture: copy, then prune (flicker in retrieval, not storage).** When a remap would fire, add a copy of
   the hot prototypes under the new label and keep the originals; delete whichever copy goes 300 ticks without
   winning a nearest-prototype match labelled with its class. Prediction on seed-26 stream plus held-out streams
   48-65: false remap events (originals deleted before the swap) <= 4 total, latency to first correct new-label
   answer <= m0.3n4hinf. Kill: any runaway or latency worse than z5 - 8.
2. **Neuron: one-sided confidence (carried).** Double credit only if the contradicting label's own nearest prototype
   is far (d_y > 2d). Prediction: seed 26 clean while keeping >= 60% of m0.3's latency gain. Kill: runaway anywhere.
3. **Data: label-noise world.** 10% of 4s permanently labelled 9; m0.3n4hinf must stay at <= 1 false remap event
   per stream. This is what makes the candidate safe or not; run before promoting it.
4. **Training: routed scoring of m0.3n4hinf on the cached FF logs** (lifetime, returns, swap first 300 via j+0.0b)
   before it goes in-loop. Cheap; do with 3.

## Loop 2026-11-01 05:00 UTC+8

### Experiment: copy, then prune (`copy_prune.py`, `results/copy_prune/`)
- Pre-registered H10 (doc string): when a remap would fire, copy the hot prototypes under the new label and keep
  the originals as twins; a hit on a twin credits the twin whose label arrived; retrieval answers with the twin
  that won most recently; a twin with no win for P ticks is deleted. Selected m0.3n4c600 on dev seeds 12-29.
- **Killed on latency, but it solves false remaps.** Committed false remaps: dev seed 26 68 (m0.3n4) / 22
  (m0.3n4hinf) -> 1; held-out fresh seeds 48-65: 0 on 18 streams; pre-swap store accuracy unchanged (91.90).
  But the store answers the new label later than even z5: held-out ans_lat (3 correct new-label answers in a row)
  103.8/113.7 vs z5 92.3/100.4 (s49f/bg002f, +12.4 mean, bar was -8), swap first 300 45.1/46.7 vs z5 52.0/51.5 and
  m0.3n4 57.1/58.8.
- New metric note: on held-out, m0.3n4's store-only answer latency gain over z5 is 9-24 ticks (ans_lat), close to
  its first-remap gain; first-remap latency is a fair proxy for moves, not for copies.
- Diagnostic (seeds 48, 50, s49f): copies are made on time (t+61 to t+110, like moves) and win 88-90% of twin hits
  in the first 300 ticks, but the originals keep winning 10-16% (9-images now labelled 4 landing on old 4
  prototypes near the border), so they are never pruned and the last-win answer flips back to the old label after
  each of those hits. The loss is in the retrieval rule, not the storage rule.

### Literature insight
Gold & Shadlen 2007 (The neural basis of decision making, Annu Rev Neurosci 30:535-574): perceptual choices
integrate noisy evidence to a bound; a choice that follows only the last sample is the worst-case integrator.
Our twin retrieval is exactly that: with 90/10 support for the copy after a swap, answering by the most recent win
gives the old label on every query that follows an original win, while a majority over the last ~10 twin hits
would almost never err. Jezek et al.'s flicker settles toward the better supported map, it does not follow the
latest theta cycle.

### Paths closed
- Last-win twin retrieval (any P): safe (0 committed false remaps on held-out) but slower than z5 on the new label.
- Storage-level fixes for false remaps (floors, hysteresis) are superseded: deferring the rewrite removes them.

### Hypotheses, ranked
1. **Neuron/retrieval: evidence-integrated twin answer.** Answer with the twin holding the majority of the last 10
   twin hits (tie: original), pruning unchanged. Prediction on fresh held-out seeds 66-83 (dev 12-29 for P):
   committed false remaps <= 4 total, ans_lat <= z5 - 8, swap first 300 >= m0.3n4 - 2. Kill: any bar missed.
2. **Architecture: seed the copy from the evidence.** Place the copy at the mean of the contradicting items that
   made the prototype hot, not at the original's location, so twins separate by content as well as label.
   Prediction: fewer original wins after a swap (< 5%) and swap first 300 >= m0.3n4.
3. **Data: label-noise world.** 10% of 4s permanently labelled 9: a copy store should keep both twins alive
   (flicker as a standing border code) with <= 1 committed remap per stream; a move store should ping-pong.
4. **Training: routed scoring** of the winning copy store on the cached FF logs via the j+0.0b arbiter
   (lifetime, returns, swap first 300) before it goes in-loop.

## Loop 2026-11-01 06:00 UTC+8

### Experiment: majority-vote twin retrieval (`majority_twin.py`, `results/majority_twin/`)
- Pre-registered H11 (doc string): copy-then-prune storage unchanged; answer with the twin holding the majority of
  the last K labelled twin hits (tie: original). Variants m0.3n4c600v10, m0.3n4c600v5, m0.3n4c300v10; selected
  m0.3n4c600v10 on dev seeds 12-29. Held-out fresh seeds 66-83 (never used).
- **Killed.** Committed false remaps 0 on held-out (pass), dev seed 26 1 (pass), pre_acc unchanged, but ans_lat
  +15.2 ticks vs z5 (bar -8) and swap first 300 -12.2 vs m0.3n4 (bar -2): held-out s49f/bg002f swap300 45.6/47.5
  vs last-win 44.2/47.1, m0.3n4 58.5/59.1, z5 51.6/51.8. K=5 and K=10 are indistinguishable.
- Diagnostic `copy_cap_diag.py` (dev only): lifting the 2k class cap during copies changes nothing (identical
  numbers). First copy after the swap 57.6 ticks vs m0.3n4 first move 56.4 and z5 70.3. Copies are on time; the
  ~30-tick answer deficit arises after the copy.
- The 05:00 explanation (original wins flip the last-win answer back) is withdrawn as the main cause; it is worth
  at most ~1 point.

### Literature insight
Yassa & Stark 2011 (Pattern separation in the hippocampus, Trends Neurosci 34:515-525): DG/CA3 give a new
experience that overlaps an old one a decorrelated code; overlap that is kept gets completed to the old memory.
Our copy sits at exactly the original's coordinates, so the twins are separated by label only and every border
item is resolved by pair bookkeeping (masking, credit rules) instead of by content. A move store has no such
contest. Prediction: the deficit comes from items that hit a pair whose arriving label is neither twin's, or from
unpaired originals near the copies; a copy displaced toward the contradicting items should answer like a move.

### Paths closed
- Twin retrieval rules (last-win, majority of 5 or 10): neither fixes the copy store's latency.
- Class-cap eviction on copy: not involved.

### Hypotheses, ranked
1. **Diagnostic first (decides the fork): error attribution.** For swapped-class arrivals in the first 300 ticks,
   log which prototype answered when the copy store is wrong and the move store right: (a) unpaired original,
   (b) a twin pair whose copy is masked, (c) the copy itself. Kill "pairing bookkeeping" if (a) dominates (>60%).
2. **Neuron/architecture: content-seeded copy (pattern separation).** Place the copy at the mean of the
   contradicting items that made the prototype hot. Prediction on fresh held-out seeds 84-101: committed false
   remaps <= 4, ans_lat <= z5 - 8, swap300 >= m0.3n4 - 2. Kill: any bar missed.
3. **Algorithm: evidence carries across pairs.** Contradictions that hit a paired prototype currently add nothing
   to the (s, y) evidence counter, so a second hot prototype needs fresh evidence; count them. Prediction: more
   copies in the first 300 ticks and swap300 within 2 of m0.3n4.
4. **Data: label-noise world** (10% of 4s permanently labelled 9): a copy store should hold both twins with
   <= 1 committed remap per stream; a move store ping-pongs.
5. **Training: routed scoring** of any copy store that passes, via the j+0.0b arbiter on cached FF logs.

## Loop 2026-11-01 07:00 UTC+8

### Experiment: copy-first twin retrieval + error attribution (`seed_copy.py`, `results/seed_copy/`)
- Code reading found the latency cause before the run: a new pair starts tied (w_o = w_c = t) and ties go to the
  original, so every fresh copy is masked at birth and the original keeps answering the old label until the copy
  earns its own labelled hit. Pre-registered H12: start each pair with the copy winning (w_o = t-1); storage,
  pruning and last-win retrieval unchanged. Variants m0.3n4c600s, m0.3n4c300s, m0.3c600s; selected m0.3c600s on
  dev seeds 12-29. Held-out fresh seeds 84-101 (never used).
- **Killed on one bar, latency cause confirmed.** Held-out: committed false remaps 0 (pass), dev seed 26 1 (pass),
  pre_acc -0.01 (pass), ans_lat -16.7 vs z5 (pass; s49f/bg002f 90.4/84.8 vs m0.3n4 90.8/79.4, last-win copy
  112.9/114.4), but swap first 300 -6.4 vs m0.3n4 (bar -2): 49.5/53.3 vs 57.0/58.5.
- Error attribution (wrong answers on swapped-class arrivals, first 300 ticks, per run, held-out): m0.3n4 84.3,
  all from unmoved old-label prototypes. Last-win copy store 110.3 = 89.0 unpaired + 13.5 masked-at-birth originals
  + 3.6 originals that won back + 4.2 copies. Copy-first removes the masked-at-birth errors (13.5 -> 0.6) and
  nothing else: 98.0 = 89.0 unpaired + 3.6 originals won back + 4.9 copies. Residual gap to the move store
  ~13.7 errors/run: +4.7 unpaired old-label prototypes, +3.6 flicker of originals, +4.9 copies answering wrong.
- The 06:00 claim "the deficit sits after the copy" is now split: the first-answer delay was the masked-at-birth
  rule (fixed); the swap-300 deficit is spread over three sources, none above 5 errors/run.

### Literature insight
Bittner, Milstein, Grienberger, Romani & Magee 2017 (Behavioral time scale synaptic plasticity underlies CA1 place
fields, Science 357:1033-1036): a single dendritic plateau potential creates a new CA1 place field that is
expressed on the very next traversal, with no further evidence needed. Formation is the decision; expression does
not wait for confirmation. Our copy store had the opposite rule (a formed copy stayed silent until re-confirmed),
which cost ~20 ticks of latency. The H confirming contradictions that make a prototype hot are our plateau; once
they fire, the copy should be "on". What BTSP also does that our copy does not: the new field is placed where the
plateau happened (the items that drove it), not on top of the old field.

### Paths closed
- Masked-at-birth copies (tie to original): closed as a design error; any copy store keeps copy-first.

### Hypotheses, ranked
1. **Neuron/architecture: content-seeded copy (BTSP placement / pattern separation).** On copy-first m0.3c600s,
   place the copy at the mean of the in-radius contradicting items that made the prototype hot, not at the
   original's coordinates. Targets the +3.6 flicker and +4.9 copy errors. Prediction on fresh held-out seeds
   102-119: committed false remaps <= 4, ans_lat <= z5 - 8, swap300 >= m0.3n4 - 2. Kill: any bar missed.
2. **Algorithm: evidence carries across pairs** (contradictions hitting a paired prototype count toward the (s, y)
   counter). Targets the +4.7 unpaired errors: predicts more copies in the first 300 ticks (m0.3c600s 14.2 vs
   m0.3n4 15.7).
3. **Data: label-noise world** (10% of 4s permanently labelled 9) on m0.3c600s vs m0.3n4: copy store <= 1 committed
   remap per stream, move store ping-pongs. This is where a copy store should earn its keep even at -6 swap300.
4. **Training: routed scoring** of m0.3c600s via the j+0.0b arbiter on cached FF logs; the arbiter may absorb a
   6-point store-only swap deficit if lifetime and returns hold.

## Loop 2026-11-01 08:00 UTC+8

### Experiment: content-placed copies (`place_copy.py`, `results/place_copy/`)
- Pre-registered H13 (07:00 rank 1): each prototype keeps the inputs of its last 10 in-radius contradictions; a copy
  under label y is placed at their mean ('p') or halfway between original and mean ('h'), else identical to
  copy-first m0.3c600s. Selected m0.3c600sh on dev seeds 12-29 (highest swap300). Held-out fresh seeds 102-119.
- **Killed on three bars, monotone in placement.** Held-out swap first 300 (s49f/bg002f): at the original
  (m0.3c600s) 46.3/51.7, halfway 41.0/46.2, at the evidence mean 32.1/29.4; m0.3n4 54.2/57.5. ans_lat vs z5:
  -5.6 for h (bar -8); p is slower than z5 (127/148 vs 105/103). Committed false remaps 5 (bar 4). pre_acc unchanged.
- Error attribution shows where it goes: unpaired old-label prototypes answer 87.7 (s) -> 95.8 (h) -> 118.6 (p)
  wrong per run; copy errors barely move (6.6 -> 7.1 -> 7.2). Moving the copy off its original's location does
  not make it more accurate; it makes it lose the region it should have inherited to neighbouring old-label
  prototypes. Likely cause (untested): a mean of <= 10 items is a sharp exemplar, farther in squared distance from
  typical inputs than a long-run EMA prototype, so it wins fewer nearest-prototype contests.
- Side finding: on seeds 102-119 the copy store's false-remap advantage shrinks: m0.3c600s commits 4 false remaps on
  3 streams (seeds 102, 103 5->3, 110 7->2), the move store m0.3n4 2 on 1 stream (seed 102). The copy store still
  ends seed 26's 68-event ping-pong, so it removes the worst case but not rare single false moves.

### Literature insight
Bittner et al. 2017 (Science 357:1033) again, read for the window: the plateau potentiates inputs active over a
~4-second window around it, so a new CA1 field is broad, about as wide as existing fields, not a point at the
triggering location. Our 'p' copy was the opposite, a 10-item mean. In Mamba terms (Gu & Dao 2023), a selective
write B(x) only helps if the read C(x) addresses state in the same basis; writing a sharp exemplar into a store
read by distance to long-run means puts the new memory where the read cannot find it. Placement must match the
scale of the read, and at our scale the original's location is already the right address.

### Paths closed
- Content-placed copies (mean of contradicting items, or halfway): closed. Copies stay at the original.

### Hypotheses, ranked
1. **Architecture/algorithm: class contagion.** Unpaired old-label prototypes are the bulk of post-swap errors in
   every store (move store 87/run). Once one prototype of class s commits or copies to y, require only H-1
   contradictions (or K=1) for the rest of class s for 600 ticks. Prediction on fresh held-out seeds: swap300 +4
   over m0.3n4 at <= m0.3n4's false remaps; kill if swap300 gain < 2 or false remaps double.
2. **Data: label-noise world** (10% of 4s permanently labelled 9): now decisive, since the copy store's clean-world
   false-remap edge is not robust (4 vs 2 on seeds 102-119). If the copy store does not beat the move store there,
   close the copy family.
3. **Neuron: placement at matched scale** (copy at the n-weighted blend of original and evidence mean, weight
   n_contra/(n_orig + n_contra)): low priority; today's monotone trend predicts it lands at m0.3c600s.
4. **Training: routed scoring** of m0.3c600s and m0.3n4 via the j+0.0b arbiter on cached FF logs.

## Loop 2026-11-01 09:00 UTC+8

### Experiment: class contagion (`contagion.py`, `results/contagion/`)
- Pre-registered H14 (08:00 rank 1): after any committed move s->y, for 600 ticks a further prototype of class s that
  takes one in-radius, rival-passing contradiction labelled y moves at once ('x600'); 'x600a' drops the radius
  requirement. Base m0.3n4. Selected m0.3n4x600a on dev seeds 12-29. Held-out fresh seeds 120-137.
- **Passed every pre-registered bar.** Held-out swap first 300 (s49f/bg002f): x600a 63.8/67.9, x600 63.2/66.1, m0.3n4
  58.9/59.7, z5 53.2/53.2, copy store m0.3c600s 50.9/53.6. Pooled +6.6 over m0.3n4 (bar +2). ans_lat 66.8/66.7 vs
  m0.3n4 72.5/72.4 (-5.7). swap1000 75.3 vs 72.0. pre_acc 92.10 = z5. Committed false remaps 0 (m0.3n4 also 0).
  First gain on swap300 over the move store since 02:00, and it comes without any return-side change in the store.
- **Caveat, not covered by the bar:** contagion multiplies a false commit. On dev seed 26 (the one stream where m0.3n4
  ping-pongs 4<->9 pre-swap, 68 events) x600 makes 242 events and x600a 866. Held-out seeds had no false commit in
  the base, so they could not test this. Contagion is only as safe as the commit it spreads from.
- Error attribution: unpaired old-label errors fall 81.3 -> 68.1/run (held-out). Two thirds of them remain: those
  prototypes get no contradiction at all in the first 300 ticks, so no per-prototype rule reaches them (the null's
  half that held). x600a makes ~205 contagion moves per run in 300 ticks (x600: ~43), i.e. heavy churn between the
  two swapped classes; accuracy is still higher, but this is not a clean one-shot class relabel.

### Literature insight
McClelland, McNaughton & O'Reilly 1995 (Psych. Rev. 102:419) and Kumaran, Hassabis & McClelland 2016 (TICS 20:512):
the fast system stores the episode and the slow system generalises across items that share structure; a new fact
about a category should reach members that were not in the episode. Our store had the episode decision (a committed
s->y) but applied it one prototype at a time. In Mamba terms (Gu & Dao 2023) the committed remap is a large step
Delta on one channel: once the gate opens, the write should land on the whole channel, not on the single entry that
triggered it. The 300-tick swap number moves exactly as much as this predicts for prototypes that are visited; it
cannot move those that are not, which is what offline replay (sleep) is for in CLS.

### Paths decided
- Class contagion: kept as move-store candidate m0.3n4x600 (fewer churn moves than x600a, swap300 within 1.2). Not
  default yet: needs a guard on the commit it spreads from (seed 26 x3.6 / x12.7 amplification).

### Hypotheses, ranked
1. **Architecture: contagion on a sticky base.** m0.3n4hinfx600 (hysteresis base cut seed 26 68 -> 22) and copy-store
   contagion m0.3c600sx600 (copy store ends the ping-pong). Prediction: dev seed 26 <= 22 events and held-out
   swap300 >= m0.3n4 + 4. Kill if seed 26 > 68 or swap300 gain < 2.
2. **Training/sleep: contagion by replay.** At the next sleep after a committed s->y, relabel every class-s prototype
   whose nearest y-labelled episode lies within its radius, reaching the prototypes no waking arrival touches (the
   remaining ~68 unpaired errors/run). Predicts swap1000 gain larger than swap300 gain.
3. **Data: label-noise world** (10% of 4s permanently labelled 9): contagion is the worst case here, since one noisy
   commit spreads to the class. Prediction: x600 false relabels >= 3x m0.3n4. If confirmed, contagion must be gated
   by evidence across prototypes (>= 2 distinct prototypes committed) before it spreads.
4. **Neuron: contagion radius at matched scale** ('x600' in-radius vs 'a' any distance): a 1.5x radius variant should
   keep most of x600a's swap300 with x600's churn.

## Loop 2026-11-01 10:00 UTC+8

Direction change (user, 1 Nov): Gall's law. Park daytime contagion / copies / vetoes; build the minimal CLS dual
first: store writes by day, selective replay of store memories into the cortex by night, cortex learns from replay only
during sleep. Core stays a25w250h2k2_c1.5_uinf_z5 answering through router c3.

- Experiment: `cls_sleep.py` (results/cls_sleep, results/cls_sleep_bg002). Conditions none / uni8 / swr2 / swr8 /
  swr2_wl3; s49 seeds 3-5 and bg002 seeds 6-8. Pre-registered bar in the docstring.
- **Killed as pre-registered in both worlds.** Best is swr2: returns +0.9 / +1.4, swap300 +0.0 / +0.2, lifetime
  -0.27 / -0.23 (bar -0.10). The lifetime loss is exactly the arrivals missed while offline: lifetime on answered
  arrivals equals `none` within 0.06 for every sleep condition. Replay does no damage.
- Mechanism check passes inside the cortex only: FF-only returns 56.1 -> 60.7 (swr2) / 62.8 (uni8) on s49, 54.5 ->
  58.9 / 64.0 on bg002. The combined answer barely moves because the store already answers returns (store-only
  82-89) and never forgets a class. Slow cortex (wake LR x0.3) loses ~10 points of swap300: killed.
- Literature: Kumaran, Hassabis & McClelland 2016; McClelland et al. 1995. Consolidation earns its keep only because
  the hippocampal trace is temporary. Our store is permanent, so sleep has no job in the combined system.
- Running at commit time: `results/cls_forget` (H15 below), s49 seeds 3-5.

### Hypotheses, ranked
1. **H15 Architecture: temporary hippocampus.** The store drops any class unseen for >= 2000 ticks (= the return gap,
   checked every 100 ticks and after each night's replay). Conditions none_f2k / swr2_f2k / uni8_f2k.
   Pre-registered: PASS if a sleep condition beats none_f2k on combined returns by >= 3 with lifetime >= none_f2k -
   0.30 (offline cost allowed). Kill if not: then this cortex cannot take over what the store loses and the dual
   reduces to a permanent store plus a router. Secondary: how close swr2_f2k gets to the permanent-store `none`
   (74.8 returns) measures how much store memory sleep can replace.
2. **Training: sleep in place of wake learning at matched compute.** Score lifetime on answered arrivals only, or let
   the world pause at night, so offline time is not charged twice; decide whether the bar's lifetime clause is fair.
3. **Router: hand returns to the cortex when it has consolidated.** If H15 passes, the router should prefer FF on
   classes replayed in the last night; predicts combined returns >= max(store-only, FF-only).

## 2026-11-01 10:45 UTC+8

### Findings
1. **Minimal CLS dual (`cls_sleep.py`) killed.** Pre-registered PASS needed lifetime >= none - 0.10 and returns
   >= none + 1.0 (or swap +2). s49: none 96.12/74.8; swr2 95.85/75.7; uni8 95.22/76.7; swr8 95.18/75.6;
   swr2_wl3 95.34/73.2 (swap -9.9). bg002: none 95.86/75.2; swr2 95.63/76.6; uni8 94.78/77.9; swr8 94.81/78.2.
2. **Sleep cost = downtime.** Answered-only lifetime 96.11-96.18 vs 96.12 (s49), 95.80-95.87 vs 95.86 (bg002).
3. **Replay consolidates, but redundantly.** FF-only returns 56.1 -> 60.7-62.8 (s49), 54.5 -> 58.9-64.0 (bg002);
   store-only returns already 82-89. Uniform >= selective.
4. **H2 temporary hippocampus killed.** none_f2k 96.06/58.0; swr2_f2k 95.80/58.6; uni8_f2k 95.12/60.1 (bar +3).

### Literature insight
Diekelmann & Born 2010 / Wilson & McNaughton 1994: consolidation happens offline because the animal is not
otherwise using its hippocampus then. Our machine pays for sleep in lost answers; the brain keeps quiet-wake
responses during rest. Separating "cortex learns from replay" from "machine stops answering" is the next cut.

### Next
Night answering by store only (no learning from arrivals), uni8/swr8, same bars, s49 3-5 + bg002 6-8.

## Loop 2026-11-01 11:00 UTC+8

### Housekeeping
- Found a duplicate run of the already-scored H2 (`results/cls_forget`, launched 10:54, orphaned): killed to free CPU.

### Findings
1. **H3 night answering: PASS (both worlds).** swr8_na -0.04/+2.3 (s49), -0.09/+4.5 (bg002); swr2_na -0.01/+1.3,
   +0.01/+1.5, swap flat; uni8_na fails bg002 lifetime (-0.12). FF-only returns +5 to +11 now reach the answer.
2. **H4 awake interleaved remote replay: PASS (s49 3-5).** awr_f2k 96.21/69.8 vs none_f2k 96.06/58.0; awu_f2k
   96.21/67.4 (remote > uniform by 2.4); awr on the permanent store 96.22/80.9/68.8 vs none 96.12/74.8/67.2.
   Cost: novel first-300 -1.8 (awr). 15 min per run vs 37 for night runs.

### Literature insight
Karlsson & Frank 2009 (Nat Neurosci 12:913-918): during brief waking pauses, 44% of SWR candidate events replay a
*remote* environment, as often as the local one, and more than during long quiescence (36%). Design consequence:
consolidation does not require going offline, and the content should be what is *absent* now. Tested directly as
H4; the remote-over-uniform margin (+2.4 returns) and the night-vs-awake gap (+2.1 vs +11.8 under forgetting)
both support it. Under CLS 1995, the night version fails because full-rate day learning overwrites between nights;
interleaving removes the gap.

### Running at commit time
- H3c / H4c held-out confirmation: `results/cls_na_conf_{s49,bg002}` (none, swr2_na, swr8_na) and
  `results/cls_aw_conf_{s49,bg002}` (awr), seeds s49 140-142 / bg002 143-145. Score with `CS.summarise` against the
  same-seed `none` rows in `cls_na_conf_*`; bars in the `cls_sleep.py` docstring.

### Hypotheses, ranked
1. **Confirm awr (H4c).** If it holds, adopt z5 + awake remote replay as the default core and retire nights.
2. **Training: replay rate.** every=38 was matched to uni8, not tuned. Prediction: returns gain saturates by every=19
   and novel first-300 cost grows with rate; kill rates that lose > 1.0 novel. Pre-register every in {19, 76}.
3. **Architecture: who answers returns.** With awake replay the FF-only returns (74.0) approach store-only (~86).
   Hypothesis: once FF-only returns >= store-only on a class, the store's prototypes for it can be pruned without
   loss (systems consolidation proper, Kumaran 2016). Test: awr_f2k with F lowered to 1000; PASS if returns stay
   >= none (permanent store) - 1.0. That would make the store a temporary buffer, as CLS assumes.
4. **Data: novel cost.** Remote replay pulls FF toward old classes when a novel class arrives. Exclude classes
   younger than 300 ticks? They already are by the gap rule; check whether the cost is from replay crowding instead.

## 2026-11-01 11:45 UTC+8 (H3 checkpoint)

### Findings
1. **H3 night answering passes (pre-registered 10:48).** Store answers alone at night, no learning from night
   arrivals, cortex learns only from replay. s49 3-5: swr8_na 96.07/77.1, uni8_na 96.11/78.2, swr2_na 96.10/76.1 vs
   none 96.12/74.8. bg002 6-8: swr8_na 95.77/79.7, uni8_na 95.74/79.4 (lifetime -0.12, fails), swr2_na 95.87/76.6
   vs none 95.86/75.2. swr8_na and swr2_na pass in both worlds.
2. Per seed, returns are up on 17 of 18 (condition, seed) pairs; lifetime deltas -0.22 to +0.24.
3. Costs: novel first-300 -1.4 to -6.6 with 8 chunks; swap300 about -1.

### Literature insight
Wilson & McNaughton 1994 / Diekelmann & Born 2010: offline consolidation runs while the animal still responds to
salient input during quiet rest; separating "cortex learns only from replay" from "machine stops answering" removes
the entire cost. Karlsson & Frank 2009 (awake remote replay) motivates H4.

### Next
H3c / H4c held-out confirmation (running). If both confirm, default = higher returns at lifetime within 0.05.

## 2026-11-01 13:10 UTC+8 (H6 return-aware arbiter)

### Findings
1. **H6 KILLED (pre-registered 13:15, both worlds).** awr_ra s49 93.81/83.8/67.1/61.6 vs awr 96.22/80.9/68.8/76.2;
   bg002 93.06/84.9/68.3/59.9 vs 95.86/81.8/69.7/73.4 (lifetime/returns/swap300/novel300). awr_raq (FF-confidence
   escape at q70) only trims the loss (lifetime -2.1 / -2.4). Overrides ~1,300-1,700 per run; store right 3%, FF 84%.
2. The store's winner is an absent class mostly when the input is background or novel and lands near an old prototype.
   c3's relative-margin rule was already rejecting exactly these.

### Literature insight
Daw, Niv & Dayan 2005 (Nat Neurosci 8:1704-1711): arbitration by each controller's *posterior uncertainty about the
current choice*, not by a global staleness flag. The test here used staleness of a class label as a proxy and failed
because it is a property of the store's answer, not of the cortex's belief about this input. A faithful version needs
an input-specific uncertainty for the cortex, or input-specific evidence for the store (H6b uses temporal evidence).

### Hypotheses, ranked
1. **H6b burst-confirmed return gate.** Override only when the store named the same absent class on another arrival in
   the last 20 ticks. Prediction: overrides drop >10x, store-right share > 50%, returns +2 kept. Bar as H6.
2. **H7 awr + swr2_na.** Does night replay add on top of awake replay? PASS if returns >= awr + 1.0 at lifetime within 0.05.
3. **awr rate sweep** every 19 / 76.

## 2026-11-01 14:40 UTC+8 (H6b / H6c return gates)

**Read first:** `results/cls_rw/RESULTS.md`. Code: `cls_sleep.py` conds `awr_rb`, `awr_rw` (pre-registered 14:15).

### Findings
1. H6's gain was only ever first arrivals: labels refresh `prev_seen`, so after one labelled return arrival the class is no
   longer "absent". H6b as written selects repeated store errors; confirmed (2.4% store-right).
2. H6c label-confirmed return window lifts returns +6.1 / +5.1, above store-only, but loses 3.7 / 4.5 lifetime: background
   arrivals reopen windows constantly, and inside a window the store is still right on < 4% of disagreements.
3. Across H6, H6b, H6c the store is right on 2-4% of disagreements however the class is gated. Class state cannot separate
   the store's true returns from its errors.

### Literature insight
Howard & Kahana 2002 (J Math Psych 46:269-299, temporal context model): retrieval of an item reinstates its temporal context,
which favours its neighbours in time, so the cue for "this is a return" is a context reinstatement, not an item label. H6c
used the label as a context cue for the whole class, which is too coarse: a single background exemplar reinstates nothing
about a season. A TCM-faithful gate would need a drifting context vector (recent input statistics) matching the context
stored with the store's prototype, i.e. input-specific evidence again. Together with Daw, Niv & Dayan 2005 this closes
class-level arbitration.

### Paths closed
- Routing by class state (stale, burst-named, recently returned): closed. c3 stays.

### Hypotheses, ranked
1. **H7 awr + swr2_na.** PASS if returns >= awr + 1.0 at lifetime within 0.05 in both worlds.
2. **awr rate sweep** every 19 / 76 (is 38 on the knee?).
3. **H8 store-side context match** (TCM): store prototypes carry the drifting context at write time; the store answers a
   disagreement only if its prototype's stored context matches the current context better than chance. Only if 1-2 leave
   the gap.

## 2026-11-01 17:30 UTC+8 (H10 selective bus vs compression; H11 gain gate)

**Read first:** `results/cls_h10/RESULTS.md`; audit.md updates 15:10, 16:20, 17:30 (H7 nights on top of awake replay
killed, H8 replay rate sweep killed, every=38 stays). Code: `cls_sleep.py` conds `awu`, `awg` (H11 pre-registered 17:25).

### Findings
1. Replay compresses: the cortex's prequential code length (ff_bits) falls from 0.515 to 0.460 bits per awake arrival
   with awake replay, and on returns from 1.99 to 1.30 bits.
2. The absence selector does not: awr 0.4603 vs uniform awu 0.4566 (bar: awr <= awu - 0.02). KILLED. Same returns
   (81.8 vs 81.9) and lifetime (95.86 vs 95.92). Absence helps return bits (-0.06) and swap300 (+1.9), costs novel300 (-2.0).
3. By the group constraint, selectivity is not justified as a bus with a permanent store. Its only earlier win (H4,
   +2.4 returns) was under a forgetful store.

### Literature insight
Mattar & Daw 2018 (Nat Neurosci 21:1609-1617): the utility of replaying a memory is gain x need. Need is expected future
use (absence predicts a return); gain is how much the update would change behaviour (the cortex's error on it). awr is
need-only, and H10 shows need alone buys no compression. H11 tests gain alone as a timing gate; H12 the product.

### Paths closed
- Absence (need-only) selection as the compression bus with a permanent store.

### Hypotheses, ranked
1. **H11 awg** gain-gated uniform awake replay (running): replaces awr only if returns >= awr + 1.0 and lifetime >= awr - 0.05
   in both worlds (or same returns at half the replay); a selective bus only if also ff_bits <= awu - 0.02.
2. **H12 gain x need batch** (new, sharp): sample replay prototypes with probability proportional to the cortex's bits on
   each prototype times (1 + absence/300), on the awr clock. Prediction: ff_bits <= awu - 0.02 AND returns >= awr. Kill if
   ff_bits within 0.02 of awu.
3. **H13 forgetful store** with ff_bits: awr_f2k / awu_f2k / awg_f2k, the regime where selection decides what survives.
4. **H9 store confidence** vs second-nearest prototype for the store-only vs combined returns gap.

## 2026-11-01 18:20 UTC+8 (H11 gain gate killed; H13 selection under forgetting)

**Read first:** `results/cls_h11/RESULTS.md`, audit.md 18:20. Code: `cls_sleep.py` H11 result + H13 pre-registration,
and a fix for no-awake conditions crashing after the H11 edit.

### Findings
1. awg (gain-gated uniform replay) = awr on bg002: returns 81.8 vs 81.8, same replay budget. KILLED.
2. Code length: awg 0.4624, awr 0.4603, awu 0.4566 bits per awake arrival. Selection of any kind is a no-op with a
   permanent store; replay itself is not (none 0.5145).
3. H11b: uniform replay misses awr's swap300 by 1.9 on bg002, so by the pre-registered bar awr stays, on swap alone.

### Literature insight
Aljundi et al. 2019 (MIR, NeurIPS): retrieve the memories whose loss the coming update would raise most. Gain has to be
per memory and look-ahead; a global surprise clock (awg) only changes when replay happens, not what it protects.

### Paths closed
- Global error-gated timing of awake replay.

### Hypotheses, ranked
1. **H13** (running on the box): absence vs uniform under the forgetful store, with ff_bits, both worlds.
2. **H14** MIR-style per-prototype interference selection on f2k, only if H13 shows selection matters.
3. **H9** store confidence vs second-nearest prototype.

## 2026-11-01 18:50 UTC+8 (H13 scored: absence demoted)

H13 KILLED on the compression term: under a forgetful store absence beats uniform replay by +2.5 / +3.4 returns and
~0.11 return bits in 6/6 seed pairs, but whole-history code length is unchanged (+0.002 / -0.001 bits, bar -0.02).
Core becomes z5 store + c3 router + uniform awake replay (awu, every 38th wake arrival). Next: H14 MIR-style per-memory
interference selection on f2k, and a returns vs present-digit split of ff_bits.

## 2026-11-09 09:00 UTC+8 (H18 scored: dense pairing @ 2× width killed; Nov 8 recovered)

**Read first:** `results/cls_h18/RESULTS.md`; audit.md 2026-11-09. Code: `cls_sleep.py` H18 result + H19 pre-registration.
Nov 8 loop ran awid_w2_f2k to completion on the box then failed before write-up; closed today from on-disk JSONs.

### Findings
1. awid_w2_f2k vs awu_w2_f2k: ff_bits −0.038 / −0.039 (bar −0.02, 6/6 seeds); lifetime −0.13 / −0.18 (guard −0.05,
   every seed). Returns +5.8 / +6.0. KILLED as default.
2. Same trade as H16 at hidden=500. Width does not dissolve the pairing lifetime cost.
3. Core stays z5 + c3 + awu every-38 + hidden=1000.

### Literature insight
Mirzadeh et al. 2022 (ICML): wider nets forget less mainly because gradients of new tasks stay more orthogonal to old
ones. That mechanism explains H17 (capacity under separate-step awu) and predicts — wrongly, here — that H16's pairing
cost should shrink at 2× width. The residual cost matches the Adam step-share account from H16, not residual overlap.

### Paths closed
- Dense pairing (awid) as default awake replay, including under the H17-width cortex.

### Hypotheses, ranked
1. **H19** shrink-and-perturb (Ash & Adams 2020) or slower wake LR under standing awu every-38 hidden=1000. KEEP only
   if, in both worlds, ff_bits.all <= awu_w2 − 0.02 AND lifetime >= awu_w2 − 0.05.
2. **H9** store confidence for the router gap.
3. Parked: contagion / twins / vetoes / class-local arbiter (Gall minimal dual first).

## 2026-11-09 09:30 UTC+8 (H19 shrink-and-perturb killed)

**Read first:** `results/cls_h19/RESULTS.md`; audit.md 2026-11-09 09:30. Code: `cls_sleep.py` H19 result; `learners.py`
`ForwardForward.shrink_perturb` + `Adam.reset`.

### Findings
1. awu_sp_f2k vs awu_w2_f2k: ff_bits +0.111 / +0.111 (bar −0.02, wrong direction, 6/6 seeds); lifetime −0.81 / −0.78
   (guard −0.05); returns −8.5 / −8.6. KILLED.
2. Daily λ=0.8 σ=0.01 undoes consolidation under standing awake replay. Core unchanged.

### Literature insight
Ash & Adams 2020 (NeurIPS): shrink-and-perturb closes the warm-start generalization gap by balancing old/new gradients
across discrete training rounds while preserving ReLU hypotheses. On a continuous wake/replay dual the same daily reset
raises whole-history −log₂p by ~0.11 bits/arrival — so the H15 knee is not a warm-start problem of that form.

### Paths closed
- Daily shrink-and-perturb (λ=0.8, σ=0.01) as a cortex plasticity lever on the standing awu every-38 hidden=1000 core.

### Hypotheses, ranked
1. **H19b** slower wake LR (5e-4) under standing awu every-38 hidden=1000. KEEP only if ff_bits.all <= awu_w2 − 0.02
   AND lifetime >= awu_w2 − 0.05 in both worlds.
2. Gentler SP (λ≥0.95 or rarer than daily) only if separately pre-registered after H19b.
3. **H9** store confidence for the router gap.

## 2026-11-09 12:00 UTC+8 (H20 new world pre-registered; How it works refreshed)

**Read first:** `docs/h20_world.md`, `results/h20_floors/RESULTS.md`. Code: `h20_world.py` (generator + floors only;
`cls_sleep.py` untouched, H19b runs as pre-registered).

### Findings
1. External review, accepted: since H10 the loop tunes knobs on a world with nothing algorithmic in it; compression is
   a label code and the store is free. The next lever is the world and the objective, not the learner.
2. H20 world: label-free 41-symbol stream from 7 hidden processes (grammar, two counters, CA rules 30/110, two motifs)
   switching in seasons (slow 16 384 / fast 2 048), N = 524 288, seeds 6–8, deterministic.
3. Floors (bits/symbol): uniform 5.36; KT order-0–8 fixed-share mixture 1.48 / 1.54; gzip 1.85 / 2.08; bz2 0.95 / 1.09;
   **lzma 0.675 ± 0.011 / 0.791 ± 0.015**. Short contexts sit at 3.2 bits on counters (generator entropy 0) and 1.2 on
   CA rows (≈ 0.044).
4. How it works page and Fig 1 now show the standing core (uniform replay, hidden=1000, forgetful z5 store).

### Literature insight
Schmidhuber's history compressor (1991–92): a predictor passes upward only what it fails to predict, and what is
predicted can be forgotten. Our store + cortex is that pair, but the store keeps everything and pays nothing. H20's
two-part score T = P + store bits / N is the first place the store has to pay for itself.

### Hypotheses, ranked
1. **H19b** wake lr=5e-4, replay lr unchanged (1e-3), as corrected before any run (see below).
2. **H20** current core on next-symbol prediction (window 32 one-hot, FF real-vs-corrupted next step), then any
   learner: PASS only if P ≤ lzma − 0.10 in both worlds and every seed, T ≤ lzma − 0.05, no process worse than the KT
   mixture + 0.05, switch bits ≤ the mixture's. Add a PPM/CTW floor first.
3. **H21** input-dependent state (gated linear recurrence / fast weights), **H22** compression progress as the replay
   selector. Loop budget from now: one hypothesis in three changes world, objective or substrate.

## 2026-11-09 12:10 UTC+8 (H19b pre-registration corrected before any run)

The 09:30 H19b entry in `cls_sleep.py` set `lr=5e-4` for the whole learner, which halved replay as well as waking and
so changed two things at once. No H19b run had started (none on disk on either machine). Corrected: `awu_lr5_f2k` =
standing awu every-38 hidden=1000 with **wake lr=5e-4, replay lr unchanged (1e-3)**, via the existing wake multiplier
(wl=0.5). Wake and replay share one Adam: step sizes differ, moments and step count are shared. Bar unchanged: KEEP only
if ff_bits.all <= awu_w2 − 0.02 AND lifetime >= awu_w2 − 0.05 in both worlds. Core unchanged.

## 2026-11-09 14:30 UTC+8 (orchestrator: curiosity H23/H23b/H23c; hippocampus diagnosis H24a)

**Read first:** `CLAUDE.md` (new: roles, handoffs, guideline), `docs/curiosity.md`, `docs/hippocampus_diagnosis.md`,
`results/h23*/RESULTS.md`, `results/h24a/RESULTS.md`. Code: `curio.py`, `h24_episodic.py`. Ran on the VPS; nothing in
`cls_sleep.py` or the H19b/H20 queue was touched.

### Findings
1. **Guideline adopted:** Schmidhuber 2009 (arXiv 0812.4360, in `docs/papers/`): store everything, compress, reward
   compression progress, act on it. Every hypothesis names the principle it serves.
2. **H23 (active world, 9 channels incl. noisy TV and dark) KILLED:** prediction-error curiosity spends 89% on noise,
   progress 1%; but W(prog) 1.693 vs rand 1.667. The oracle headroom was only 0.031 (KT saturates), and sliding-window
   progress rewarded re-fitting a counter's prefixes.
3. **H23b (slow SGD compressor, held-out-next progress) KILLED:** worse than random on every seed (+0.36). Next-chunk
   progress rewards *tracking* a drifting counter (84% of late glances); the greedy bandit starved the grammar.
4. **H23c (progress = eq. 5 on a whole-history reservoir) PASS, all bars:** W 2.373 vs rand 2.637 (6/6 seeds,
   −0.22 to −0.32), beats the noise-aware oracle by 0.108. An easy → hard curriculum emerges (motifs and CA, then the
   37-motif, then the grammar at 54% in Q4), with 1% on noise.
5. **H24a (episodic MNIST) diagnosis stands:** with 200 arbitrary-label episodes in the stream, an exemplar store is
   100% from the 2nd exposure, while cortex and ProtoStore score 13–32%. Exact-match gate: +13.6 lifetime at 0.01% false
   firing. Exemplar replay teaches the cortex the episodes (79% vs 32% at 6+). P2 missed by 1.3 points in letter.

### Literature insight
Schmidhuber 1991 / 2009 §3.2 already warns that error-driven curiosity gets stuck on noise; H23–H23c add that *how*
progress is estimated decides everything: only old vs new on the whole history (eq. 5) separates lasting compression
from memorising (window) and tracking (next chunk). CLS (McClelland et al. 1995; Kumaran et al. 2016): the hippocampus
pays off on specific, arbitrary, one-shot content and as a replay source. MNIST seasons had neither, and the averaging
ProtoStore erased what episodes there could have been.

### Paths closed
- Prediction-error curiosity; sliding-window and next-chunk progress estimators.
- "The hippocampus does not help" as a conclusion about CLS. It was a conclusion about the MNIST world plus an averaging
  store, and it is withdrawn.

### Hypotheses, ranked
1. **H19b** as pre-registered (unchanged), then **H20** condition 4 (unchanged).
2. **H24b** episodic process `F` in the symbol world (handoff spec `handoffs/2026-11-09_worker_h24b_world.md`); score
   an exemplar, surprise-gated store under H20's two-part T (bits saved per stored bit).
3. **H23d** controller: proportional / UCB allocation over whole-history progress (Graves 2017-style) vs ε-greedy, on
   the H23c setup; then the same curiosity on H20's world with the H21 substrate, where counters and CA rows become
   learnable and should become interesting.
4. **H24c** combined route on episodic MNIST: exact-match gate over the replaying cortex, plus deletion of
   consolidated entries (store size priced).
5. Store default for every new world: exemplar traces, not running-mean prototypes.

## 2026-11-09 19:30 UTC+8 (orchestrator: H21a scored; papers library; literature scan; worker split)

**Read first:** `docs/roadmap.md`, `results/h21a/RESULTS.md` (Verdict), `docs/literature_2026-11.md`, CLAUDE.md
("Roles", "Papers library", "Cadence").

### Findings
1. **Work split changed.** The orchestrator sets direction, writes specs with pre-registered bars, reviews and scores.
   codex (sandboxed) and grok implement and run. H21a was the first experiment run entirely by a worker.
2. **H21a PARTIAL (codex):** online GRU 1.259 / 1.657 bits (slow / fast) vs KT mixture 1.735 / 1.712 vs lzma
   0.744 / 0.976. It beats KT on 5/6 seeds and never beats lzma. Counters partly learnt (C1 0.5–0.9× KT); CA rows not.
   Plain delta-rule FWP above KT everywhere.
3. **Mechanism:** about 2 000 updates per life (step-starved), and lzma's advantage is verbatim recall, which is
   a store. M1 becomes cortex + store; H21b (replay ×2, match model, fixed-share mixtures) dispatched to codex.
4. **Papers library (grok):** R2 `pdp-papers` + D1 `pdp-papers-index`, idempotent `tools/sync_papers.py`; registry of
   every cited paper (file_status stored / to_fetch / wanted / unavailable) running.
5. **Literature scan:** three families of progress estimators (ours, LPM, gradient-momentum coupling); modern linear
   RNNs are fast-weight programmers whose slow weights still need BPTT; e-prop as the local candidate; Titans'
   surprise-gated memory; loss of plasticity in long lives.

### Literature insight
Irie & Gershman (2025/26) plus our wake/sleep criterion suggest **local wake, global sleep**: only fast weights change
while awake (local, online); the slow weights train from replay while asleep, where global credit assignment is
allowed. H26, after M1.

### Paths closed
- Plain (ungated) delta-rule fast weights as the M1 substrate at this scale, untuned.

### Hypotheses, ranked
1. **H21b** cortex + store (running, codex).
2. **H24c** combined store route on episodic MNIST (spec ready).
3. **H25a** curiosity with the M1 learner (gated on M1).
4. **H25b** progress-estimator ablation; **H26** local wake / global sleep.

## 2026-11-10 02:40 UTC+8 (orchestrator: overnight run, H21b–H21e and H25a–H25d by codex)

**Read first:** `docs/roadmap.md` (Status), the Verdict sections of `results/h21b..h21e/RESULTS.md` and
`results/h25a..h25d/RESULTS.md`.

### Findings
1. **M1 (cortex), Partial; line stopped.** Replay is the lever: ×2 sequential −0.27 bits; batched 2 × 8 better and
   cheaper, and it cures the grammar crowd-out. GRU rb8x2 + match beats lzma on 2/3 slow seeds (mean 0.772 vs 0.744).
   A wider GRU is worse at fixed updates (H21d); local attention is worse overall but slightly better on CA (H21e). The CA
   rule is parked.
2. **M2 (curiosity with a neural compressor), pass by the letter only.** H25a: ε-greedy progress falls into a dark-room
   trap (67% on the constant channel), and replay contaminates attribution. H25b: clipping + own-step attribution
   reduce the trap; gmc_prop beats random and the noise-aware oracle (0.852 vs 0.902) but spends 22% on noise. H25c:
   per-channel signed GMC fails. H25d: visit-to-visit slope with proportional allocation passes bars 1–2 (seed 8 by
   0.002) with near-uniform allocation.
3. **Papers library:** 109 cited papers in D1; 81 stored in R2 (arXiv, Unpaywall, author pages, institutional
   repositories); 28 wanted, six of them high relevance.

### Literature insight
Oudeyer et al. (2007) measure learning progress per region over a time window, not per sample. That is what H25d did,
and it fixed noise robustness but not the attribution scrambling of a shared network. Blad et al. (2026) GMC assumes
per-group momentum; with shared Adam momentum and an absolute product it degenerates into an error signal (H25b).

### Paths closed
- Wider GRU and local attention as M1 levers at this budget; single-glance progress for shared neural compressors;
  GMC with shared momentum; signed per-channel GMC scaled by remaining bits.

### Hypotheses, ranked
1. **H24c** combined store route on episodic MNIST (M3; spec ready).
2. **H24b** episodic process F in the symbol world (M3).
3. Parked open problems: CA rule (GRU + local-attention hybrid); clean progress attribution in shared nets (per-channel
   progress heads); H26 local wake / global sleep after M3.

## 2026-11-10 08:45 UTC+8 (daily orchestration: M3 overnight results; metric change; H24f dispatched)

**Read first:** `docs/roadmap.md` (Status), the Verdict sections of `results/h24c/`, `results/h24d/`, `results/h24e/`
(`RESULTS.md`), and `results/h24b_floors/RESULTS.md`.

### Findings
1. **H24c (codex): an exact-match gate over a replaying cortex** gives +5.7 lifetime over either part (85.35%).
   Deletion after "right twice" killed: normal items never repeat, so the store stayed at 84%.
2. **H24d (codex) PASS, M3 on MNIST:** surprise-gated writes (store only what the system got wrong) keep 15.5% of
   arrivals, a 6.4× smaller store, for −0.9 lifetime; repeat episodes 99.8%; the cortex consolidates episodes faster.
3. **H24b world (codex):** facts process F (300 arbitrary key=value records) added to the symbol world; exact recall
   makes a value free after one appearance, where KT needs 5+ appearances to reach 1.7 bits/digit.
4. **H24e (codex): FAIL as scored, metric flaw found.** H20's two-part T charges stored history as description length,
   but a prequential decoder already has the history: storing it is free in code length, and its real cost is memory.
   From now on, P and memory footprint are reported separately (Pareto). On that axis the surprise store recovers 72%
   of the full store's P gain with 27% of the memory. The order-8 match model cannot recall facts (the preceding context
   differs on every occurrence).

### Literature insight
Schmidhuber 2009 §1.2 ("store everything … the data is holy") is consistent with prequential coding: the history costs
no extra description length, so a store must be judged on hardware cost (memory) and on what it buys in P. H20's T
mixed the two axes.

### Paths closed
- Two-part T as a single score; deletion by repeated correct answers; order-8 matching for episodic facts.

### Hypotheses, ranked
1. **H24f** (running, codex): symbol-level surprise writes, match order 5, P vs memory Pareto.
2. PPM floor (grok, after H24f; CPU).
3. M4 (all parts in one life) after M3 in bits; parked: CA rule, clean progress attribution, H26 local wake / global sleep.
Codex budget: 76% of the week, resets Thu 10:55. One codex job per day until then.

## 2026-11-10 16:00 UTC+8 (orchestrator: M3 in the symbol world; M4a fails; M4b finds why; Kaggle compute)

**Read first:** `docs/roadmap.md` (Status 10 Nov afternoon), Verdict sections of `results/h24g/`, `results/m4a/`,
`results/m4b/` (`RESULTS.md`).

### Findings
1. **H24f/H24g (grok), M3 PASS in the symbol world:** order-5 matching recalls facts; a utility store (keep windows by
   bits actually saved, evict the least useful under a budget) gets P within 0.024 bits of the full store at 15% of its
   memory, evicting CA noise and consolidated motifs/counters.
2. **M4a (grok), M4 not passed:** gmc_prop beats random on 3/3 seeds and matches the noise-skipping oracle, but the
   store run is 0.08 bits *worse* than the GRU alone (motifs and counters forgotten).
3. **M4b (grok), diagnosis confirmed:** replay drawn from the utility store starves the cortex of the content the store
   evicted. Replay from the whole history: −0.30 bits vs no replay; −0.29 vs store-only replay. The match mixture adds
   only −0.017 in the active life. Curiosity + history replay: W 1.280 vs 1.765 for the GRU alone (seeds 0–2), but 17%
   of glances on noise.
4. **Infrastructure:** Kaggle kernels work headlessly (4 CPU, 32 GB, T4, private); recipe in CLAUDE.md. Kaggle lookup
   picked a real-data world W1 (Bible, chess moves, SQuAD, BTC book, synthetic noise; enwik8 for calibration).
   Papers: 84 stored, 25 wanted.

### Literature insight
Schmidhuber 2009 §1.2 again: store the raw history and keep learning from it. Our utility store optimised the wrong
target for replay: it keeps what the cortex *cannot* predict, so replaying it teaches the surprising and lets the
consolidated decay. The CLS view agrees: hippocampal replay interleaves old and new, not only the new.

### Paths closed
- Store-only replay; slope_prop (visit-to-visit progress) as a controller.

### Hypotheses, ranked
1. **M4c** (grok, Kaggle): M4 rescored on fresh seeds 9–14 with history replay.
2. Curiosity's noise share (17%): clean per-channel progress attribution (parked open problem) becomes the next M2 item.
3. W1 real-data world, enwik8 cortex calibration, after M4c. Parked: CA rule, H26 local wake / global sleep.

## 2026-11-10 17:00 UTC+8 (orchestrator: M4c scored on Kaggle; H27 from the compute-vs-data-movement note)

**Read first:** Verdict in `results/m4c/RESULTS.md`; `docs/compute-vs-data-movement.md` (Ez's note).

### Findings
1. **M4c (grok, Kaggle, fresh seeds 9–14):** curiosity + GRU + whole-history replay W 1.283; GRU alone 1.780; GRU +
   replay 1.538; oracle 1.272. Curiosity beats random 5/6. **M4 fails only the noise guard (17.3% on N).** The utility
   store contributes −0.001 bits.
2. First Kaggle run: 36 lives in ~8 min wall across 3 private kernels; Kaggle vs local numerics agree to 2e-7.

### Literature insight
Ez's note frames the store as HBM, the cortex as compute and replay as the bus. Our data refines it: select hard for
what sits next to compute (the cache), sample fairly for what crosses the bus to train weights. And the bus should be
metered: traffic becomes the third reported axis next to P and memory.

### Paths closed
- The utility store as part of the active machine in this world (kept as a cache option; it is not where the value is).

### Hypotheses, ranked
1. **H27** (grok, Kaggle): bounded uniform reservoir vs full history vs recent-only replay; traffic dose curve.
2. Noise leak in curiosity: per-channel progress attribution (M2 open problem), now the main defect of the machine.
3. W1 real-data world + enwik8 calibration.

## 2026-11-10 19:00 UTC+8 (orchestrator: H27, H27b replay economics; H28 dispatched)

### Findings
1. **H27 (grok, Kaggle):** a fixed small replay buffer replayed hard is worse than no replay (res77 1.877 vs 1.808;
   ~100 replays per segment: the cortex memorises it). Full history best; 1 step per glance beats 2 and 4.
2. **H27b:** at low repeats per segment bounded memory works: a 60% reservoir matches/beats the full log (1.428 vs
   1.457, traffic 8); 15% recovers part (1.653 vs no replay 1.815). Knee at traffic ≈ 8.
3. Papers from Ez's compute-vs-data-movement note registered (2 stored, 9 wanted).

### Literature insight
The note's "fixed state instead of a growing log" fails for learning in its strong form: replay needs distinct data,
each piece replayed ≲ 10 times. Memory needed ∝ traffic × life / allowed repeats.

### Paths closed
- Small fixed replay buffers at high dose; replay doses above traffic 8.

### Hypotheses, ranked
1. **H28** (grok, Kaggle): whole-history progress (plain and common-mode-removed) with history replay, to cure the
   17% noise leak; M4 PASS if it does.
2. W1 real-data world + enwik8 calibration. Parked: CA rule (negative-eigenvalue linear RNNs), H26.

## 2026-11-10 20:30 UTC+8 (orchestrator: H28 scored; M4 paused; H29 dispatched)

### Findings
1. **H28 (grok, Kaggle):** whole-history progress + replay cures the noise leak (N 8.8%) but stops beating random
   (2/6 seeds, −0.026 mean). gmc_prop: best W (1.299), beats the oracle, 16.5% on noise.
2. With dose-1 replay the oracle's headroom over random is only 0.056 bits. My 0.05 bar used almost all of it
   (headroom rule not applied: orchestrator error).

### Paths closed
- Noise-leak fixes in this world (H25b–d, H28). M4 paused, not killed: the world stopped rewarding curiosity.

### Hypotheses, ranked
1. **H29** (grok, Kaggle GPU): our cortex on enwik8 vs xz / zstd / bz2 / PPMd on the same 10 MB.
2. W1 active real-data world (Bible / chess / SQuAD / BTC + noise): curiosity's next test, with headroom checked first.
3. Parked: CA rule (negative-eigenvalue linear RNNs), H26 local wake / global sleep, progress attribution.

## 2026-11-11 01:30 UTC+8 (orchestrator: H29 scored; architecture page)

### Findings
1. **H29 (grok, Kaggle T4, 1.49 GPU h):** the H27b cortex (GRU 1024, byte level) on enwik8's first 10 MB scores 3.644
   bits/byte. xz -9e scores 2.176 and PPMd order 16 scores 1.837 on the same bytes. **Both bars FAIL.**
2. The model degrades as it reads: 2.26 on the first MB, then 3.5–4.2. Replay cost +0.35 bits/byte on 2 MB (replay
   2.62, no replay 2.27).
3. Cause: the spec's replay dose was traffic 32 (batch 32 per chunk), four times the H27b knee. That means hundreds
   of repeats per early segment. It was an orchestrator spec error, made the same day the knee was recorded.
4. Site: an architecture page (orchestrator, workers, Kaggle, Cloudflare, Telegram) is live.

### Literature insight
An online learner at a fixed Adam learning rate has to stay stable over 10^7 steps. The toy worlds (10^4 glances)
never tested that; real data does.

### Paths closed
- Replay traffic 32 on real text (it overfits, as H27 predicted).

### Hypotheses, ranked
1. **H29b** (grok, Kaggle GPU): E10M with no replay, with replay at traffic 8, and with no replay at a lower
   learning rate. This separates dose from stability.
2. W1 active real-data world, with headroom checked first.
3. Parked: CA rule (negative-eigenvalue linear RNNs), H26, progress attribution.

## 2026-11-11 03:10 UTC+8 (orchestrator: H29b scored)

### Findings
1. **H29b (grok, Kaggle T4, 1.91 GPU h):** on enwik8's first 10 MB, no replay at lr 3e-4 scores **1.974 bits/byte**.
   That beats xz -9e (2.176) and trails PPMd order 16 (1.837) by 0.14. Slices fall from 2.51 to 1.86. **Both bars
   PASS.**
2. The same model at lr 1e-3 diverges after about 5 MB with no replay (2.19 → 4.58). H29's collapse was mostly
   optimiser instability, compounded by a 4× replay dose.
3. Replay at traffic 8 with lr 1e-3 costs +0.43, because each replay step is another step at an unstable learning
   rate. Replay at lr 3e-4 has not been run.

### Literature insight
On real data, the learner's lifetime is set by its step size: 10^7 online steps need a smaller learning rate than 10^4
did. The toy worlds could not show this, and the cortex's hyperparameters were tuned there.

### Paths closed
- lr 1e-3 for long lives (it diverges).

### Hypotheses, ranked
1. **H29c** (grok, Kaggle GPU): replay at the stable lr 3e-4, at traffic 1 and 8, plus no replay at lr 1e-4. Does
   store-everything replay help once the learner is stable? Bar: beat PPMd order 16.
2. W1 active real-data world, run at the stable learning rate, with headroom checked first.
3. Parked: CA rule (negative-eigenvalue linear RNNs), H26, progress attribution.

## 2026-11-11 04:40 UTC+8 (orchestrator: H29c scored; replay works on real text)

### Findings
1. **H29c (grok, Kaggle T4, 2.25 GPU h):** at the stable lr 3e-4, replay helps on enwik8 E10M. Traffic 8 scores
   **1.832 bits/byte**, traffic 1 scores 1.874, and no replay scores 1.974. **Both bars PASS.** That beats xz -9e
   (2.176) by 0.34 and edges PPMd order 16 (1.837) by 0.006 on one run, so call it a tie with PPMd.
2. lr 1e-4 is too slow (2.151). The last MB of the traffic-8 arm (1.762) is still falling.
3. Arc of the night: H29 3.644 → H29b 1.974 (stable step size) → H29c 1.832 (replay at the knee). The toy-world
   replay findings (store everything, traffic about 8) carried over to real text once stability was fixed.

### Paths closed
- None new. "Replay hurts on real text" (H29b's reading) is withdrawn: it was an artefact of the unstable learning rate.

### Hypotheses, ranked
1. **H29d:** the full 100 MB at lr 3e-4 with traffic 8, against the published PPMd J1 (1.711) and xz (1.976).
   Measure seed spread on E10M (2 more seeds) first, so the PPMd tie gets an error bar. GPU estimate: about 6–7 h
   for the full file; it may need splitting across sessions.
2. W1 active real-data world: curiosity at the stable learning rate with replay at traffic 8, headroom checked first.
3. Parked: CA rule (negative-eigenvalue linear RNNs), H26, progress attribution.

## 2026-11-11 08:50 UTC+8 (orchestrator: daily run; H29d and W1-0 dispatched)

### Findings
1. Nothing new to score: H29, H29b and H29c were scored overnight (entries above). Roadmap status updated: the cortex
   holds up on real text and ties PPMd on E10M on one seed.
2. Usage: Claude 29% weekly and codex 79%, so codex stays off. Kaggle GPU about 5.7 h used this week of about 30.

### Paths closed
- None.

### Hypotheses, ranked
1. **H29d** (grok, Kaggle GPU, running): seeds 1 and 2 for r8_3e4 and nr_3e4 on E10M. Bars: replay gain < 0 on all 3
   seeds with mean ≤ −0.05; a PPMd win only if mean + std < 1.837254.
2. **W1-0** (grok, Kaggle GPU, running): build the 4 real channels, then per-channel learning curves and an exhaustive
   oracle allocation. Bar: headroom ≥ 0.10 bits/byte (the synthetic world had 0.056).
3. M4 on W1 if W1-0 passes. H29e on the full 100 MB enwik8. Parked: CA rule (negative-eigenvalue linear RNNs), H26,
   progress attribution.

## 2026-11-11 11:05 UTC+8 (orchestrator: H29d and W1-0 scored; W1b dispatched)

### Findings
1. **H29d (grok, Kaggle T4, 2.94 GPU h):** over 3 seeds on enwik8 E10M, replay at traffic 8 scores 1.8362 ± 0.0036
   and no replay scores 1.9739 ± 0.0004. Paired gain −0.138 ± 0.003, negative on every seed. **Bar 1 PASS.** Against
   PPMd order 16 (1.8373) it is a **TIE**, |difference| 0.0011 < std. H29c's edge was seed noise.
2. **W1-0 (grok, Kaggle T4, 0.90 GPU h):** the four real channels are built: bible, chess moves, SQuAD text, BTC book.
   Per-channel learners (lr 3e-4, traffic 8) on 2 MB each give an allocation headroom of **0.039 bits/byte**, under the
   0.10 bar. **FAIL.** The oracle only shifts budget toward SQuAD.
3. Mechanism: rich real streams are all unsaturated at 2 MB, with similar log-linear learning curves. Marginal gains are
   nearly equal across them, so uniform is almost optimal. Curiosity needs channels that differ in kind: unlearnable,
   saturated and learnable.

### Literature insight
Schmidhuber's argument for curiosity is about avoiding the unlearnable and the already-learned. Among equally
learnable streams there is little to choose, and W1-0 measures that: 0.04 bits/byte of headroom.

### Paths closed
- W1 with only rich learnable channels, as a curiosity test.

### Hypotheses, ranked
1. **W1b** (grok, Kaggle GPU, running): add dna (quickly learnable), noise (unlearnable) and dark (saturating), then
   remeasure headroom for the full set and the learnable set. M4 on W1b only if the full set reaches ≥ 0.10.
2. H29e: the full 100 MB enwik8 (about 10 GPU h at traffic 8), against published PPMd J1 (1.711). Deferred until the
   weekly GPU quota allows; about 9.5 h used this week.
3. Parked: CA rule (negative-eigenvalue linear RNNs), H26, progress attribution.

## 2026-11-11 13:30 UTC+8 (orchestrator: W1b and W1c scored; breadth worlds closed; W2 and notes)

### Findings
1. **W1b (grok, Kaggle T4, 0.64 GPU h):** adding dna, noise and dark gives a separate-learner headroom of 0.081 for
   the full set and 0.040 for the learnable set. **FAIL** (< 0.10). The gain comes from skipping noise and spending
   little on dark.
2. **W1c (grok, Kaggle T4, 1.21 GPU h):** with one shared cortex, uniform reading over all 7 channels is best.
   headroom_shared is −1.89, **FAIL**. Most of that is one term: a cortex that never read noise mis-predicts it at
   21.6 bits/byte. On the learnable channels uniform is still best (2.219 vs 2.304 and 2.368), and reading noise did
   not damage them. My prediction of interference was wrong.
3. Knowing what is unpredictable is knowledge. A curious agent should sample the TV briefly, not avoid it.

### Literature insight
Breadth worlds give curiosity only second-order headroom: concave learning curves have a flat optimum, and noise costs
only opportunity. Curiosity's case rests on depth worlds, where actions decide what can be learned (W2).

### Paths closed
- W1, W1b and W1c as curiosity tests (breadth worlds).

### Notes
- Ez's note on delay lines, frequency domain and content addressing is in `docs/compute-vs-data-movement.md`, with
  review. Our uniform-position replay is RAM-style. H26 (cued wake plus broad sleep replay) is the address-free test
  and moves up. H30 is an oscillator-bank cortex (complex and negative eigenvalues). 11 papers handed to grok for the
  library.
- Kaggle GPU quota guard added to CLAUDE.md. About 12.3 of 30 GPU h used this week.

### Hypotheses, ranked
1. **W2-0** (grok, Kaggle CPU, running): locked-rooms depth world; headroom of random vs oracle; bar ≥ 0.5
   bits/observation.
2. M4 on W2 if W2-0 qualifies.
3. H26 address-free replay (cued wake + broad sleep) vs uniform-position replay, then H30 oscillator cortex. Both
   CPU-sized.
4. H29e full 100 MB enwik8 (needs Ez's OK, about 10 GPU h).

## 2026-11-11 13:35 UTC+8 (orchestrator: frequency domain and nonlinearity corrected; brain map with the thalamus)

### Findings
1. **Correction.** I had written that the frequency picture unifies storage only with *linear* computation. Ez
   disputed it, and Ez is right. Products in time are convolutions in frequency. Volterra series give nonlinear systems
   frequency kernels, and any fading-memory operator has a finite Volterra approximation (Boyd & Chua 1985). The
   Koopman operator makes nonlinear dynamics linear on observables. The refined statement: storage and moving are
   *diagonal* in frequency; computation is *off-diagonal* coupling between frequencies. The cost of nonlinearity is
   the size of the lift (on the order of Nⁿ for an order-n kernel), not that it cannot be expressed. This is in
   `docs/compute-vs-data-movement.md`.
2. **Brain map** (`docs/brain_map.md`, figure on the site). Every major brain part now has a job in the machine, what
   we have measured and a next test. Our machine has a cortex, a hippocampus and a basal ganglia, and nothing else.
   The thalamus had no place at all. It is the gate in front of the cortex (the TRN searchlight, which W1c already
   priced as small), a router between cortical areas (pulvinar, MD) and the sleep gate that times replay (spindles
   coupled to ripples).

### Literature insight
Thalamic spindles locked to slow oscillations and hippocampal ripples predict consolidation (Staresina et al. 2015).
Driving spindles in phase improves memory (Latchoumane et al. 2017). So *when* replay reaches the cortex is a variable
brains control. We have held it fixed: interleaved, one replay step per chunk.

### Hypotheses, ranked
1. **W2-0** (grok, Kaggle CPU, running), then M4 on W2.
2. H26: content-cued wake replay plus broad sleep replay vs uniform-position replay.
3. **H31, the thalamic sleep gate (new):** the same replay dose (traffic 8), delivered interleaved or in sleep blocks
   with input off. CPU.
4. H30: oscillator-bank cortex, now with an order-2 multiplicative readout (a learned Volterra term).
5. **H32, the cerebellar mixer (new):** a fast count model mixed with the GRU cortex by a learned mixer on E10M. The bar
   to beat is PPMd order 16 by a clear margin. About 1.2 GPU h.
6. H29e: full enwik8 (needs Ez's OK).

## 2026-11-11 14:00 UTC+8 (orchestrator: W2-0 scored; the depth world qualifies)

### Findings
1. **W2-0 (grok, 3 Kaggle CPU kernels, about 6 min each): PASS.** Headroom (rand8 − oracle) is 9.95 / 9.93 / 8.92
   bits per observation (bar ≥ 0.5 on every seed). The raw number is inflated by overconfidence in unvisited rooms
   (12–16 bits, worse than uniform). With each room's probe capped at 4 bits it is still 3.12 / 3.12 / 3.11.
2. rand8 never gets past room 2 (seed 2: room 4), because it presses reset one step in eight. Its whole-life
   prequential bits are lower than the oracle's (0.62 vs 0.69). Staying shallow is cheap. The probe, not prequential
   bits, is the curiosity metric.
3. Most of the gap is not pressing reset. rand6 (no reset, no TV) reaches room 8 and stays there. rand6 − oracle
   is 1.57 / 0.50 / 0.76 capped. The planning headroom is that smaller number.
4. The oracle never watched the TV and mis-predicts it at 9–14 bits, the W1c lesson again.

### Hypotheses, ranked
1. **M4-W2** (spec next): curious controllers on W2, judged against rand6, on held-out world seeds 3–8, probes capped
   at 4 bits per room. Progress reward against surprise and count-novelty rewards.
2. H26, H31, H30, H32 as listed at 13:35.

## 2026-11-11 14:30 UTC+8 (orchestrator: M4-W2 scored, FAIL; M4-W2b dispatched)

### Findings
1. **M4-W2 (grok, 3 Kaggle CPU kernels, 74 CPU-min): FAIL, and the kill rule fires.** On PROBE_cap (held-out world
   seeds 3–8), rand6 0.50, count_q 3.23, prog_q 3.38, err_q 3.48 and oracle 0.005. Every Q arm stayed in rooms 0–2.
   rand6 reached room 8 on every seed.
2. **The in-sample progress estimator rewards memorising noise.** Re-scoring the trained chunk gave the TV about
   0.035 bits per step, against about 0.003 for learned rooms. prog_q watched the TV 62% of the time. err_q watched
   it 90% (the predicted noisy-TV trap).
3. **ε over all 8 actions put a random reset every ~80 steps, against ~216 needed to open a lock.** That capped every
   arm at room 2. My spec error.
4. Room-only state cannot carry lock progress. Rewards in room 1 were large (0.7–2.5 bits) but rarely reached.

### Paths closed
- Progress measured by re-scoring the data just trained on, as a curiosity reward when noise is available.

### Hypotheses, ranked
1. **M4-W2b** (grok, Kaggle CPU, running): ε over manipulations only, plus a visit-to-visit learning-progress reward
   (prequential, K = 16), on fresh seeds 9–14. Bars: beats rand6 by ≥ 0.25, beats count by ≥ 0.10, TV ≤ 10%.
2. If M4-W2b fails: a controller with memory (state including the recent action and output).
3. H26, H31, H30, H32 as listed.

## 2026-11-11 15:13 UTC+8 (orchestrator: M4-W2b scored, FAIL; noise leak fixed; M4-W2c dispatched)

### Findings
1. **M4-W2b (grok, 3 Kaggle CPU kernels, 54 CPU-min): FAIL on bars 1 and 2. Bar 3 holds.** On PROBE_cap (fresh seeds
   9–14), rand6 0.97, count_q 2.83, prog_q 2.89, prog_lp 3.05 and err_q 3.42. prog_lp is worse than rand6 on all 6
   seeds.
2. **Visit-to-visit learning progress fixed the noise leak.** prog_lp's TV share is 2.2%. prog_q, with the same ε fix
   but the in-sample reward, watched the TV 32–54% of the time. The estimator was the cause.
3. **Room-only greedy control cannot open locks.** In a learned room, every Q value is about 0.001 and argmax repeats
   one action, which never types a 3-digit key (seed 9: 50,873 steps in room 1). rand6 opens locks because it is
   uniformly random.

### Paths closed
- Room-only tabular controllers in W2, whatever the reward.

### Hypotheses, ranked
1. **M4-W2c** (grok, Kaggle CPU, running): memory state (room, last two actions), seeds 15–20. Bars: beats rand6 by
   ≥ 0.25, beats count by ≥ 0.10, beats room-only by ≥ 1.0. Kill: if memory does not help, move to a model-based
   planner that uses the learner's own predictions.
2. H26, H31, H30, H32 as listed.

## 2026-11-11 15:48 UTC+8 (orchestrator: M4-W2c scored; the tabular W2 line stops)

### Findings
1. **M4-W2c (grok, 3 Kaggle CPU kernels, 42 CPU-min): bar 1 FAIL, bar 2 PASS, bar 3 FAIL.** On PROBE_cap (seeds
   15–20), rand6 1.13, lp_mem 2.82, lp_room 3.05 and count_mem 3.14. Learning progress beat count novelty by
   0.32 ± 0.31. Memory gained only 0.23.
2. **Greedy tie-breaks over all 8 actions press reset.** Fresh states have Q = 0 everywhere, so a uniform tie-break
   is rand8. With an ideal new-room reward and no learner, the memory controller still reached only room 1 in 20,000
   steps. Reset shares were 3.5–8%. This is my spec error, carried through M4-W2b and M4-W2c.
3. **Across M4-W2, M4-W2b and M4-W2c**, the reset action decides every result. The noise leak is solved (visit-to-visit
   progress, TV share 2–3%). Progress beats count novelty when both share a controller.

### Paths closed
- Tabular Q controllers on W2 (kill rule of M4-W2c).

### Hypotheses, ranked
1. **Needs Ez's direction:** the curiosity test in W2 with a controller that treats reset as a deliberate choice.
   Either (a) a cheap check, with reset excluded from tie-breaks and exploration for every arm, or (b) a model-based
   planner on the learner's predictions, the bigger build.
2. H26 and H31 (CPU, ready to spec), then H30, then H32.

## 2026-11-12 08:50 UTC+8 (orchestrator: daily run; H31 dispatched)

### Findings
No new worker output since the M4-W2c review. M4 on W2 is paused for Ez's choice between (a) never pressing reset at
random and (b) a model-based planner. The roadmap status was updated.

### Hypotheses, ranked
1. **H31, the thalamic sleep gate** (grok, Kaggle GPU, about 1.5 GPU h, dispatched): the same replay dose
   interleaved or in sleep blocks of 64, 1,024 or 16,384 chunks, on enwik8 E3M. Bars: replay still helps by
   ≥ 0.03; sleep_1k costs ≤ 0.01 bits/byte.
2. M4 on W2, option (a) or (b), on Ez's call.
3. H26 content-cued wake replay plus broad sleep replay (spec after H31; one job at a time).
4. H30 oscillator cortex; H32 cerebellar mixer; H29e full enwik8 (needs Ez's OK).

## 2026-11-12 10:01 UTC+8 (orchestrator: H31 scored; awake replay wins; H26 dispatched)

### Findings
1. **H31 (grok, 5 Kaggle T4 kernels, 1.44 GPU h): bar 1 PASS, bar 2 FAIL.** On E3M, with the same 23,438 replay
   steps in every arm: inter 1.958, sleep_64 2.002, sleep_1k 2.091, sleep_16k 2.192 and noreplay 2.198 bits/byte.
   Replay gains 0.240. Batching it into sleeps costs +0.04 (8 KB), +0.13 (128 KB) and +0.23 (2 MB).
2. **Mechanism: staleness.** While awake, the cortex drifts toward recent text, and every byte is scored by the
   drifting cortex. Each sleep repairs it: the 128 KB after a sleep is 0.08–0.15 bits/byte better than the 128 KB
   before. The repair comes after the cost is paid. Caveat: the last sleep's replay is never scored, which inflates
   the 16k arm.
3. For the brain map: a predictor that is scored online needs *awake* replay. That fits sharp-wave ripples in awake
   rest. The sleep gate is likely a second, broad pass, not the main consolidation route.

### Hypotheses, ranked
1. **H26** (grok, Kaggle GPU, about 1 GPU h, dispatched): replay keyed by content (a random projection of the hidden
   state as the index, the live state as the cue) vs uniform position, on E3M. Bars: mix44 within 0.01 of uniform;
   prediction that cued8 is worse by more than 0.01.
2. M4 on W2, (a) or (b), on Ez's call.
3. H30 oscillator cortex; H32 cerebellar mixer.

## 2026-11-12 11:33 UTC+8 (orchestrator: H26 scored; content cueing hurts consolidation; H32 dispatched)

### Findings
1. **H26 (grok, 3 Kaggle T4 kernels, 0.97 GPU h): bar 1 FAIL (narrow); prediction right.** On E3M at the same replay
   dose: uniform 1.958, mix44 1.973 (+0.015, bar ≤ 0.01) and cued8 2.021 (+0.063). Selective replay has now hurt four
   times.
2. **Mechanism: redundancy, not age.** Cued windows are as old as uniform ones and cover almost as much of the log
   (0.835 vs 0.886). But the cue is *now*, so they replay text like the current input, which the live step is already
   training on. Replay's job is to counter forgetting of what is *not* in the input.
3. **For the address question:** content addressing is right for retrieval and wrong as the consolidation sampler.
   That needs broad, cue-independent sampling: spontaneous replay in brains, uniform position in ours. An address-free
   broad sampler (a random cue) is parked.

### Hypotheses, ranked
1. **H32, the cerebellar mixer** (grok, Kaggle GPU, ≤ 1 GPU h, dispatched): count models of orders 1–4 plus a learned
   geometric mixer beside the cortex, on E3M. Bars: ≥ 0.05 better than the cortex alone, and ≥ 0.05 better than PPMd
   order 16.
2. M4 on W2, (a) or (b), on Ez's call.
3. H30 oscillator cortex. Random-cue replay (parked).

## 2026-11-12 13:20 UTC+8 (orchestrator: H32 scored; the cerebellar mixer beats PPMd; H32b dispatched)

### Findings
1. **H32 (grok, 1 Kaggle T4 kernel, v2 after a missing-package failure, 0.75 GPU h in total): both bars PASS.** On
   E3M: cortex alone 1.958, best count model alone (order 3) 2.810, PPMd order 16 1.924, mix 1.846 bits/byte. The mix
   is 0.112 better than the cortex and 0.077 better than PPMd. The cortex reproduced H26 `uniform` to the bit.
2. **Mechanism: fast and slow learners cover each other.** Neither part beats PPMd alone. The count models remember
   an exact context after one sighting; the cortex generalises. On 90% of bytes the mixer settles near 0.65 · cortex +
   0.25 · order 4. The gap to PPMd widens with data: −0.018, −0.094 and −0.120 over the three MB.
3. **Caveat:** the mixer's rows are not normalised, so part of the gain may be a learned temperature on the cortex.
   H32b tests that directly.
4. Brain map: the cerebellum and the entorhinal cortex are now marked tested (H32, H26).

### Hypotheses, ranked
1. **H32b** (grok, Kaggle GPU, ≤ 2.2 GPU h, dispatched): the same mixer on E10M, plus a cortex-only mixer (calibration
   control) and orders up to 6. Bars: mix4 ≥ 0.05 better than PPMd on E10M, and ≥ 0.05 better than the
   calibration-only mixer.
2. M4 on W2, (a) or (b), on Ez's call.
3. H30 oscillator cortex. Random-cue replay (parked).

## 2026-11-12 13:45 UTC+8 (orchestrator: north-star check; M4-W2d dispatched)

### Findings
1. **Drift.** Ez asked me to think about the north star. Today's four runs (H31, H26, H32, H32b) were all enwik8 with a
   backprop GRU. That is one world and one substrate, against our rule that one hypothesis in three changes the world,
   the objective or the substrate. Beating PPMd is a benchmark score, not the living loop.
2. **What carries over:** H32's count models are local, learn in one shot and need no backprop, and they carry real
   weight beside the cortex. From here the enwik8 line asks how much of the backprop cortex local parts can replace.
   H32b is its last pure-score run.
3. **Curiosity that acts is the core, and it stalled while waiting on a call that was mine to make.** I chose (a):
   M4-W2d. Reset and the TV are taken only when the learned Q says so, never at random. The roadmap now has a
   north-star check and a re-ranked queue.

### Hypotheses, ranked
1. **M4-W2d** (grok, Kaggle CPU, dispatched): seeds 21–26. Bars as in M4-W2c: lp_mem beats rand6 by ≥ 0.25 on
   PROBE_cap (4 of 6 seeds), and beats count novelty by ≥ 0.10. If the capability check stays at room 2 or below, the
   bars are void and the planner is next.
2. Model-based planner, if needed. 3. Organic sleep, chosen by the agent. 4. Local parts replace backprop.

## 2026-11-12 14:30 UTC+8 (orchestrator: M4-W2d scored; curiosity loses to a random walk in W2; tabular line closed)

### Findings
1. **M4-W2d (grok, 3 Kaggle CPU kernels, 41 min): bar 1 FAIL, bar 2 PASS (noisy).** With reset and the TV taken only
   when Q says so, a perfect reward now reaches room 8 (step 1,933; it reached room 1 before). On PROBE_cap over seeds
   21–26: rand6 0.87, lp_mem 1.17, count_mem 1.34, lp_room 1.12. rand6 − lp_mem = −0.30 ± 0.68 (lp_mem wins 2 of 6);
   count_mem − lp_mem = +0.17 ± 0.59.
2. **The behaviour looks right; the knowledge does not follow.** Curious agents take the rooms in order and stay
   thousands of steps in each, with no TV. But nobody ever presses reset, so everyone ends trapped in room 8. At the
   end of life the curious arms have more rooms where the model is confidently wrong (1.3–1.7 per life vs 0.8 for
   rand6). Whether that is forgetting after leaving or uneven coverage during long stays is open.
3. **Four W2 runs, one lesson:** a tabular controller cannot learn reset as a deliberate choice. Random reset keeps it
   shallow; no reset traps it. As pre-registered, recorded as the finding; the tabular line on W2 is closed.

### Hypotheses, ranked
1. **W2-1 headroom** (next cycle): give room 8 a way back to room 0, then check oracle vs rand6 headroom before any
   controller bar. If the headroom is real, rerun lp_mem against rand6 there.
2. Model-based planner, which can value going back to rooms where progress remains.
3. H32b (running). Organic sleep. Local parts replace backprop.

## 2026-11-12 15:10 UTC+8 (orchestrator: H32b scored; local count models carry real weight at 10 MB)

### Findings
1. **H32b (grok, 1 Kaggle T4 kernel, 1.28 GPU h): both bars PASS.** On E10M: cortex alone 1.832, PPMd order 16 1.837,
   cortex with a learned temperature 1.818, mix with orders 1–4 1.750, mix with orders 1–6 1.712 bits/byte. mix4 beats
   PPMd by 0.087 and the temperature-only control by 0.068. Orders 5–6 add 0.038 more.
2. **Mechanism:** calibration is 17% of the gain; the count models are the rest. Each count model alone is far worse
   than the cortex; the gain is a fast one-shot memory beside a slow generaliser.
3. **North star:** these are local, backprop-free learners carrying about 0.1 bits/byte. The enwik8 line stops racing
   compressors here. Its next question is substitution: a smaller backprop cortex plus counts, in bits and compute.
4. Process: the worker revised its own GPU-hour gate (2.49 → 2.14) before pushing. The run used 1.28, but gates
   belong to the spec. I'll say so in the next spec.

### Hypotheses, ranked
1. W2-1 headroom: room 8 gets a way back; oracle vs rand6 before any bar (next cycle).
2. Model-based planner in W2. 3. Substitution: cortex hidden 1024 / 256 / 64 plus orders 1–6, bits against compute.
4. Organic sleep, chosen by the agent.

## 2026-11-13 08:55 UTC+8 (orchestrator: daily; W2-1 headroom dispatched; H30w and H31b queued)

### Findings
1. Nothing new to score. Every handoff up to yesterday is reviewed (H31, H26, H32, H32b and M4-W2d all have verdicts).
2. Theory from yesterday afternoon, with Ez: persistence is movement through time, and moving through space and time
   is one carriage, with the medium's speed as the exchange rate. The cut that remains is carry (diagonal) versus
   combine (coupling), and local in spacetime versus addressed. Two tests follow: replay as a repeater (H31b), and
   propagation in a medium (H30w).

### Hypotheses, ranked
1. **W2-1 headroom** (grok, Kaggle CPU, dispatched): room 8 gets a 3-digit exit back to room 0. rand6, rand8, oracle
   and oracle_deep (half its time in room 8) on seeds 0–2. Bars: rand6 completes ≥ 3 cycles; an oracle beats rand6
   by ≥ 0.5 PROBE_cap on 2 of 3 seeds. Kill: no headroom means the next world needs rooms of unequal learnability.
2. **H30w** (grok, Kaggle CPU, queued until W2-1's smoke clears the VPS): a diagonal complex bank with an order-2
   readout, the same units on a propagating ring, and a GRU, on the H20 slow world and an echo stream. Bars: the ring
   beats the bank by ≥ 0.25 bits/symbol on echo copies at D = 128; bank or ring beats the GRU by ≥ 0.10 on the
   counters.
3. **H31b** (grok, Kaggle GPU ≤ 1.6 h, queued): sleep1k − inter at lr 1.5e-4 and 6e-4, against H31's 3e-4. Bars:
   the gap grows with lr, by ≥ 0.05 from lowest to highest.
4. Papers: six new references registered by grok (dispatched).
