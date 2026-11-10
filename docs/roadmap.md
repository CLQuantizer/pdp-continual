# Roadmap: from parts to one curious machine (orchestrator's view, 9 Nov 2026)

## Where we are

Three results matter. The rest is detail.

1. **Curiosity works when measured right.** Compression progress on the *whole history* (eq. 5) picks learnable
   sources, ignores noise and boredom, and builds an easy → hard curriculum (H23c). Window and next-chunk estimators
   fail (H23, H23b).
2. **The hippocampus works when the world has episodes and the store keeps them separate** (H24a). It has two jobs:
   one-shot answers plus replay that teaches the cortex, and holding the history sample that curiosity is computed on.
3. **The compressor is now the bottleneck.** Every learner we have is a context model or a stateless MLP. None can
   learn a counter's carry or look one CA row back (H20 floors: 3.2 bits on counters, where the generator's entropy is 0).
   Curiosity can only be as good as the compressor's ability to *keep* improving. With a weak compressor the
   interesting sources run out early, and H23c needed an artificially slowed learner to show anything.

## Status (9 Nov 2026, evening)

- **M1: partial (H21a).** An online GRU beats the KT mixture (5/6 seeds) but not lzma; a plain delta-rule fast-weight
  net fails. The GRU is step-starved (~2 000 updates per life) and lzma wins by verbatim recall, i.e. by being a store.
  M1 is therefore redefined as **cortex + store** (H21b, running): replay from stored history for more updates per
  observation, plus a match model mixed with the cortex. M1 and M3 now meet.

## Status (10 Nov 2026, early morning)

- **M1: Partial, line stopped.** Best cortex = GRU + batched replay (2 × 8) + match model: beats lzma on 2/3 slow seeds
  (0.772 vs 0.744 mean). Replay was the big lever (H21b/c); width (H21d) and local attention (H21e) failed. The CA rule
  (exact one-row-back access) is parked as an open problem.
- **M2: pass by the letter, mechanism not demonstrated (H25a–d).** Visit-to-visit progress with proportional allocation
  beats random on all 3 seeds but allocates near-uniformly. Gradient-momentum (gmc_prop) compresses best (beats the
  noise-aware oracle) but spends 22% on noise. Finding: a shared neural compressor scrambles per-channel progress, so
  curiosity needs clean credit assignment. Open problem, parked.
- **M3 on episodic MNIST: PASS (H24c + H24d).** Exact-match gate over a replaying cortex (+5.7 lifetime over either
  part), with surprise-gated writes: a 6.4× smaller store (1 553 vs 10 000 exemplars) for −0.9 lifetime, 99.8% on
  repeat episodes, and faster consolidation. Deletion after "right twice" was killed (normal items never repeat).
- **M3 in the symbol world: PASS (H24f + H24g).** Order-5 matching recalls facts. A utility store (keep windows by the
  bits they actually saved, evict the least useful under a budget) reaches P within 0.024 bits of the full store at 15%
  of its memory, recalls facts better than the full store, and evicts irreducible content (CA noise) and consolidated
  content (motifs, counters) while keeping facts. Surprise-gated writes alone kept irreducible surprise (41% memory).
  **Memory follows the same principle as curiosity: keep what yields compression progress.**
- **M3 in bits (H24e): FAIL as scored, and the score was wrong.** H20's two-part T charges stored history as
  description length, but a prequential decoder already has the history, so storing it is free in code length (hence
  "store everything"). A store costs *memory*, a hardware axis. **Metric change:** report prediction bits P and memory
  footprint separately as a Pareto front, and use one number only with an explicit exchange rate. On P vs memory, the
  surprise store recovers 72% of the full store's gain with 27% of the memory. Next: H24f (symbol-level writes, match
  order 5 so facts are recalled), then M4.

## The one decision that unblocks everything

**Separate "what architecture learns like Schmidhuber's agent" from "can it be trained with local rules".** The goal
asks for both, and solving both at once is why progress stalled. Order:

- **Stage A (now): substrate-agnostic.** Build the full machine with whatever compressor learns best at small scale,
  backprop included (small GRU / fast-weight net, truncated BPTT, CPU torch). That sets the *ceiling* and tests the
  architecture: compressor + store + curious controller in one life.
- **Stage B: local.** Replace the compressor with local-learning versions (FF-style, delta-rule fast weights,
  predictive coding) and report the gap to Stage A's ceiling in bits. The hardware goal becomes a measurable distance,
  not a filter on what we may try.

This is a change to `docs/goal.md`'s "local learning" criterion: it moves from an entry condition to a scored axis.
It is Ez's call; this roadmap assumes yes, and Stage B keeps the criterion honest.

## The target machine (one life, one world, one number)

- **World:** H20's processes, made active (H23 channels: the agent picks what to watch), plus noise and dark, plus
  the episodic process F (H24b). It has algorithmic structure, episodic facts and traps in one place.
- **Compressor (cortex):** a stateful learner: gated recurrence or fast weights.
- **Store (hippocampus):** exemplar traces, written when surprising, replayed to teach, deleted once consolidated, and
  holding the whole-history sample for the progress signal. Priced in bits.
- **Controller:** chooses glances and replays by expected whole-history compression progress.
- **Score:** lifetime bits per observed symbol + store bits (two-part T), with switch / return / per-process guards.
  The curiosity test: at a fixed budget, held-out W across all learnable processes vs random and the oracle.

## Milestones, each with a kill condition

| # | Milestone | Passes when | Kill / pivot when |
|---|---|---|---|
| M1 | **Substrate ceiling** (H21a): a stateful compressor on passive H20 | beats lzma's 0.675/0.791 bits and gets counters/CA well below the KT mixture | a small GRU can't beat lzma at N = 2¹⁷ on CPU → shrink the world's alphabet/timescales, not the ambition |
| M2 | **Curiosity with a real compressor** (H25a): H23c's controller on the active world with M1's learner, no artificial slowing | beats random and the noise-aware oracle on W; counters/CA get attention once learnable | progress signal too noisy with a slow neural learner → learn a progress predictor (Schmidhuber 1991) |
| M3 | **Store earns its bits** (H24b/H24c): exemplar store + gate + replay + deletion in the active world | two-part T below no-store; F's value bits near 0 after one exposure | store bits exceed savings → surprise-gated writes only |
| M4 | **One machine** (H25b): M1 + M2 + M3 together in one life | lower lifetime T than any subset | interference between parts → stage them (store first, curiosity second) |
| M5 | **Local learning** (Stage B) | gap to M4's ceiling, in bits, shrinking across hypotheses | — (this is the long game) |

## How the work is split

- **Orchestrator:** this roadmap, pre-registrations and bars inside each worker spec, review of every result, verdicts
  in `docs/loop.md`, re-ranking. Writes no experiment code unless a worker's result looks wrong and needs a check.
- **Workers (codex sandboxed, grok):** implement from the spec, run, write `results/<name>/RESULTS.md` with the numbers
  only (no verdict).
- **Cadence:** once a day, morning UTC+8: the orchestrator reviews yesterday's outputs, writes verdicts, writes the
  next 2–3 specs, dispatches them. Workers finish during the day; the orchestrator reviews and commits.

## From the November literature scan (`docs/literature_2026-11.md`)

- **H25b (M2):** progress-estimator ablation on the same world: whole-history reservoir (ours) vs gradient-momentum
  coupling (Blad et al. 2026, O(d)) vs a learned error model (LPM, Hou et al. 2025).
- **H26 (bridge to Stage B): local wake, global sleep.** Awake, only fast weights change (local delta rule, online, no
  backward pass); asleep, the slow weights are trained from the store's replay, where global credit assignment is
  allowed. Sleep then does what waking cannot, which is the goal's own test for sleep. Compare against all-online BPTT
  and against frozen slow weights; score lifetime bits including sleep's downtime charge.
- **Standing report line for long lives:** dead-unit fraction and a late-life plasticity probe (Dohare et al. 2024,
  Nature: gradient learners lose plasticity; selective unit re-initialisation keeps it).
- **H21 follow-up if the plain delta rule lags:** gated delta rule (decay gate), the strongest fast-weight variant.

## Next specs (in `handoffs/`)

1. H21a: substrate ceiling (M1). Highest priority: everything downstream depends on it.
2. H24c: combined store route on episodic MNIST (M3 groundwork; cheap, uses existing code).
3. H24b: episodic process F in the symbol world (M3 world).
Parked: H23d (controller tuning), PPM floor (useful, low priority), citation check.
