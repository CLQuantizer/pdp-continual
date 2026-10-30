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
