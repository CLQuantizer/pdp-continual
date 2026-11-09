# H20: change the world, not the learner (pre-registration)

Pre-registered 12:00 UTC+8, 9 Nov 2026, before any learner runs on it. Queued **after** H19b (half wake LR), which
stays as pre-registered in `cls_sleep.py`. Generator and floors: `h20_world.py`; floors in `results/h20_floors/RESULTS.md`.

## Why

Since H10 the effects are 0.002–0.04 bits and ±0.1 lifetime points, and the hypotheses are knobs (replay every 38, 2×
width, shrink-and-perturb, wake LR). Three properties of the MNIST-seasons world cap what the loop can learn:

1. **Nothing algorithmic to find.** The only regularity is "10 clusters come and go". The full system sits at
   95.82 / 95.91 lifetime and the forgetful store alone at 90.54 / 90.62 (H17, two worlds); what is left to compress
   is classification log-loss.
2. **Compression is a label code, not a world code.** `ff_bits` is −log₂ p(label), prequential. That is the right
   *kind* of metric, on the wrong quantity: the machine is never asked to predict what it will see next.
3. **The store is free.** It costs nothing in the score, so memorisation always wins against generalisation, and the
   store/cortex trade (CLS's central question) is never priced.

H20 changes the **world** and the **objective**. It touches no learner knob and no running loop code.

## World: `h20_world.py`

A label-free symbol stream over a declared 41-symbol alphabet (a–z, 0–9, `.`, `#`, `;`, space, newline). Seven hidden
processes; their parameters are drawn once per seed and kept across absences, so a returning process rewards memory:

| Process | What generates it | Timescale of its structure |
|---|---|---|
| `G` | Stochastic grammar: S → NP VP "."; NP → D N (.60) \| D A N (.25) \| D N *that* VP (.15, depth ≤ 4); VP → V NP (.6) \| V (.4). Lexicon of 17 seeded random 2–5-letter words | words (2–5), clauses (~10–40) |
| `C1`, `C7` | 4-digit decimal counter, step 1 / 7, records `dddd;`, resumes where it left off | 5 (record), carry over 10–10⁴ records |
| `E30`, `E110` | Elementary CA rule 30 / 110, width 31 + newline, fresh random row each visit, cell-flip noise 0.005 | 32 (one row back), rule = 3-cell lookup |
| `M7`, `M37` | Seeded motif of length 7 / 37 repeated, 2% letter substitution | 7 / 37 |

Seasons: every `season` symbols the stream switches to a uniformly drawn *other* process (no background mixing).
Two worlds, as before: **slow** (season 16 384) and **fast** (season 2 048). Length N = 2¹⁹ = 524 288 symbols.
Seeds 6, 7, 8. Deterministic: (world, seed) → byte-identical stream (numpy `default_rng`). Labels: none. The process
id is logged for analysis only; a learner may never read it.

Oracle entropy rate (generator known, after warm-up): counters 0; CA ≈ 0.044 bits/symbol (noise only); motifs ≈ 0.23;
grammar not computed. Every floor below sits far above these on the algorithmic processes (counters ≈ 3.2 bits
vs 0), so the headroom is in structure, not noise.

## Objective

- **Headline P: prequential bits per symbol over the whole lifetime.** For every symbol, −log₂ p(xₜ | x<ₜ) under the
  machine's current state, scored *before* it learns from xₜ, averaged over all N. The current core has no nights, so
  every tick is a wake tick. A future condition that sleeps is charged log₂ 41 = 5.36 bits for every symbol that
  arrives while it sleeps: a compressor cannot skip symbols. (MNIST-world scoring skipped asleep arrivals; here that
  would make sleep free.)
- **Two-part T = P + S/N.** S is the peak description length of the store over the life: each entry costs its payload
  (symbols × log₂ 41) plus 16 bits per stored scalar (count, age, key). Cortex weights are not charged: the decoder
  rebuilds them from the past stream with the same seed, which is what makes P a code length. The store *is*
  charged, because it is the part that grows with history; it now has to buy its keep in bits.
- **Guards (the lifetime analog).** (a) Switch bits: mean P over the first 1 024 symbols after every switch
  (adaptation). (b) Per-process P (no process may regress to buy another). (c) Return bits: first 1 024 symbols of a
  process already seen once.

## Floors, before any learner (all prequential or full-length codes; seeds 6–8)

| Floor | slow | fast |
|---|---|---|
| Uniform, log₂ 41 | 5.358 | 5.358 |
| Adaptive order-0 KT | 4.92 | 4.68 |
| Best single KT order (k = 2, hindsight) | 1.535 | 1.576 |
| Fixed-share mixture of KT orders 0–8 (α = 10⁻³) | 1.480 ± 0.091 | 1.539 ± 0.048 |
| gzip -9 | 1.850 | 2.079 |
| bz2 -9 | 0.952 | 1.085 |
| **lzma -9e (best floor)** | **0.675 ± 0.011** | **0.791 ± 0.015** |

Mean ± std over seeds. Per-process (KT mixture, slow): G 0.82, C1 3.23, C7 3.32, E30 1.19, E110 1.16, M7 0.39,
M37 0.51 — short contexts cannot do carries or look one CA row back. The KT mixture is a weak context model (α = ½
over 41 symbols starves high orders); lzma, a real one-pass code, beats it 2×. A PPM/CTW floor is added before the
first learner is scored; the bar always uses the lowest floor. Its count tables, priced like a store,
give T ≈ 48 bits/symbol: the n-gram memorises far more than it compresses.

4. **Current core adapted to next-symbol prediction (specified, not yet run).** The FF cortex has no state, so it
   can only see a fixed window: one-hot of the last w = 32 symbols in, 41-way next-symbol readout, real-vs-corrupted
   next step as FF's positive/negative data; the ProtoStore keyed by next symbol (z5 + c3 + awu every-38 +
   hidden=1000 otherwise unchanged). w = 32 just reaches one CA row back and cannot see a counter's carry history;
   that is the honest limit of a stateless cortex, and the reason for H21. This run is H20's first condition.

## Pre-declared bars (for any learner on this world)

- **PASS** only if, in **both** worlds and **every** seed: P ≤ lzma − 0.10 bits/symbol, AND T ≤ lzma − 0.05 (the
  store does not eat the win), AND no process's P exceeds the KT mixture's on that process + 0.05, AND switch bits ≤ the
  KT mixture's switch bits (1.61 slow / 1.55 fast).
- **Kill a line** (a run of hypotheses varying one mechanism) when its best |ΔP| against its parent is below the
  seed-to-seed std of the parent's P in either world, or after two consecutive fails in that line.
- **Loop budget.** At least one hypothesis in every three must change the world, the objective, or the substrate, not
  a scalar parameter. No more than two parameter sweeps in a row.

## Roadmap (not part of H20's bar)

- **H21 substrate.** A cortex whose state update depends on the input: a gated linear recurrence with a forget gate and
  a learned write, still updated locally. Fast-weight programmers (Schmidhuber 1992) are formally equivalent to linear
  transformers (Schlag, Irie & Schmidhuber 2021), and state-space duality (Dao & Gu 2024) links linear attention to
  selective SSMs, so the choice is input-dependent state, not a family.
- **H22 compression progress as the replay selector.** Replay what most reduces bits per unit of replay (Δ bits on the
  stream after a replay, Schmidhuber 1991/2010), not absence, gain, similarity or loss. H10–H14 tested selection only
  in a world where what is replayed barely matters.
- **Store as the incompressible residual** (history compressor, Schmidhuber 1991–92): write only what the cortex
  failed to predict; delete an entry once the cortex predicts it. T prices exactly this.
