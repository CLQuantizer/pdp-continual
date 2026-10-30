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
