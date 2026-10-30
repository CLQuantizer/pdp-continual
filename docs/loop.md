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
