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

## Next specs (in `handoffs/`)

1. H21a: substrate ceiling (M1). Highest priority: everything downstream depends on it.
2. H24c: combined store route on episodic MNIST (M3 groundwork; cheap, uses existing code).
3. H24b: episodic process F in the symbol world (M3 world).
Parked: H23d (controller tuning), PPM floor (useful, low priority), citation check.
