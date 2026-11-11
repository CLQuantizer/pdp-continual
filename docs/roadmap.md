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

## Status (10 Nov 2026, afternoon)

- **M4a (one life, all parts): not passed.** Curiosity works inside the full machine: gmc_prop beats random on all 3
  seeds (−0.16 mean), matches the noise-skipping oracle on W and beats it on facts, with 14% on noise (one seed 16%).
  But the store makes the machine *worse* than the GRU alone (+0.08 bits), losing motifs and counters. Suspect: replay
  drawn only from the utility store, which evicts consolidated content, so the cortex stops rehearsing it and forgets.
  The history (principle 1) should feed replay; the utility store is a prediction cache. **M4b** ablates replay source,
  match mixture and a hard budget on selection seeds 0–2; M4 is rescored on fresh seeds afterwards.

## Status (10 Nov 2026, evening)

- **M4c (fresh seeds 9–14, Kaggle): M4 not passed by the letter, machine works.** Curiosity + GRU + whole-history
  replay: W 1.283 vs 1.780 for the GRU alone (−0.50) and 1.538 for GRU + replay (−0.26); ties the noise-skipping oracle
  (1.272); beats random on 5/6 seeds. Fails only the noise guard (17% of glances on N, bar 15%). The utility store adds
  nothing in the active life (−0.001 W, facts slightly worse).
- **Hippocampus, revised:** the part that matters is the *stored raw history, replayed*. Select hard for the cache,
  sample fairly for the bus: a store chosen by "buys compression" is right for prediction (H24g) and wrong as the
  replay source (M4a/b).
- **Metric, revised again (from `docs/compute-vs-data-movement.md`):** report three axes: prediction bits P, memory M,
  and **bus traffic** (replayed symbols per observed symbol). Full-history replay is currently unpriced memory.
- **Next:** H27 (can a bounded, uniform sample of history replace the full log?) and the noise leak (clean per-channel
  progress attribution), then the real-data world W1.

## Status (10 Nov 2026, night)

- **Replay economics (H27, H27b):** replay needs distinct data, each stored piece replayed ≲ 10 times; a 60% reservoir
  matches the full log; best dose is one batched step per glance (traffic 8). Memory ∝ traffic × life / repeats.
- **M4: not passed, line paused (H28).** Whole-history progress with replay cures the noise leak (N 8.8%) but no longer
  beats random (2/6 seeds); gmc_prop compresses best (beats the oracle) but stays at 16% noise. With good replay the
  oracle's headroom over random is only 0.056 bits: **this toy world no longer rewards curiosity enough to test it.**
  Curiosity's next test needs a world where allocation matters more: W1 (real data, very different learnability).
- **Next:** H29 enwik8 calibration of the cortex (ruler against xz / ppmd / cmix), then the W1 active world on Kaggle.

## Status (11 Nov 2026, morning)

- **The cortex holds up on real text (H29 → H29c, enwik8 first 10 MB, Kaggle T4).** H29 scored 3.644 bits/byte: an
  unstable learning rate (1e-3) plus a 4× replay dose. H29b scored 1.974 at lr 3e-4 with no replay, which beats xz -9e
  (2.176). H29c scored **1.832** with replay at traffic 8, tying PPMd order 16 (1.837) on one seed. The toy-world replay
  findings carried over once the step size suited a 10^7-step life. This is M1's question asked of real data, and the
  answer is yes against xz, with PPMd pending seeds.
- **Running:** H29d (2 more seeds per arm, for the replay gain and PPMd error bars) and W1-0 (the real-data world:
  Bible, chess, SQuAD, BTC, with oracle headroom measured before any curiosity run; bar ≥ 0.10 bits/byte).
- **Next:** if W1 has headroom, M4 resumes on W1 at lr 3e-4 with traffic-8 replay. Then the full 100 MB enwik8 run
  (H29e), against published PPMd J1 (1.711) and xz (1.976).

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

## Real-data world, W1 (chosen 10 Nov 2026 from `handoffs/2026-11-10_grok_kaggle_worlds.table.md`)

Byte-level channels, one per source, built and run on Kaggle compute after M4 is scored on the synthetic world:
- **easy:** King James Bible (Gutenberg #10, header/footer stripped). xz 1.85 bits/byte; Canterbury `bible.txt` refs.
- **hard-learnable:** Lichess SAN move stream (`datasnaek/chess`, CC0, moves column only). xz 2.32; headroom far below.
- **episodic:** SQuAD v1.1 dev as plain text (article, question, answer; CC-BY-SA-4.0): answers recur inside the article.
- **medium numeric:** Coinbase BTC order book (`martinsn/...`, CC0); the 1-second file (1.7 GB) fits only on Kaggle.
- **controls:** synthetic uniform noise and a dark channel stay (an exactly known noisy TV beats a "nearly random" real one).
- **calibration (passive, separate):** enwik8 against published numbers (xz 1.976, ppmd 1.711, cmix 1.170 bits/byte),
  the M1 cortex test on real text.
