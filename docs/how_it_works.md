# How it works

One machine, three parts, one coupling. The figure below is the current default core (`z5` store + `c3` router +
uniform awake replay every 38th waking arrival + a 1000-unit cortex), drawn as a bench instrument. It runs on an
illustrative digit stream; every rule and number in it is the system's own.

<iframe class="fig" src="../figures/cls-bench.html" title="Fig 1: CLS dual bench, store, router, uniform awake replay"></iframe>

## Reading the bench

| Part on the bench | Part of the learner | What it does |
|---|---|---|
| Card file (left), one slot per digit | Fast episodic store: pixel ProtoStore `a25w250h2k2_c1.5_uinf_z5`, forgetful (`f2k`) | Writes every waking arrival in one shot; entries older than 2000 arrivals are dropped. A short-term buffer of patterns, not weights. A raised tab marks a digit unseen for ≥ 300 ticks; it is there for reading the bench, replay no longer uses it. |
| Two-pan balance (centre) | Router `c3` | Weighs store against cortex on *this* image. The store answers only when it disagrees at relative margin ≥ 0.3 (the fixed rider on the cortex arm); otherwise the cortex answers. The only measurement on the answer path. |
| Lap plate (right) | Cortex: forward-forward network, 1000 hidden units | The long-term compressor and the only part that changes connection strengths. Turns a little for each arrival it learns from, and 32× as much for a replay batch. |
| Overhead hoist | Uniform awake replay (`awu`) | Every 38th waking arrival, carries one batch of 32 store memories, drawn uniformly from the store, into the cortex as its own update step. The single store → cortex coupling. Interleaved with answering: no nights, no downtime. |
| Ruler and pointer (front) | Replay clock | One replay period: 38 waking arrivals. |

## Numbers behind it

Since H10 the score is whole-history code length: the cortex's −log₂ p(label) on each waking arrival, measured
before it learns from it (`ff_bits`), with lifetime accuracy as a guard (no drop larger than 0.05).

- **Replay pays; what is replayed does not.** Awake replay cuts code length by ≈ 0.06 bits per arrival against no
  replay (H10). Choosing *which* memories to replay, by absence (H10, H13), estimated gain (H11) or interference
  (H14), changes it by at most 0.003, against a 0.02 bar. Absence-based selection still lifts returns by 2–3 points
  under a forgetful store, but buys no compression, so it was demoted from the core at H13.
- **Rate has a knee at 38.** Every 76 → 38 arrivals saves ≈ 0.03 bits; 38 → 19 only ≈ 0.01 (H15).
- **Width is close to free.** Doubling the cortex from 500 to 1000 hidden units cut code length by 0.037 / 0.043 bits
  in the two worlds, lower in 6/6 seed runs, lifetime flat (H17, kept).

## What the bench leaves out, on purpose

- **No night cabinet.** H7 (night replay on top of awake replay) was killed: it added no returns over awake replay
  alone. Nights are out of the core.
- **No selective hoist.** Absence (H10, H13), gain gating (H11) and interference (H14) selection all tie uniform.
- **No dense pairing.** Mixing a stored memory into every arrival's update (H16, H16b, H18) buys ≈ 0.04 bits but
  costs 0.13–0.18 lifetime points, past the 0.05 guard, at both widths. Why is open: the shared optimizer step is a
  hypothesis, not an isolated cause.
- **No daily reset.** Shrink-and-perturb once a day (H19, λ = 0.8, σ = 0.01) cost +0.11 bits and 0.8 lifetime points.
  That kills this schedule, not resets in general. H19b (slower wake learning) is queued.
- **No store override by class state.** H6, H6b and H6c were killed: on store-vs-cortex disagreements the store is
  right only 2–4% of the time, however the digit is gated. Only per-image confidence (`c3`) separates them.
- Also killed: similarity-weighted replay, random sparse DG expansion, cortex-vetted confirmation without an LR gap.

## What comes next

The digit world has nothing algorithmic in it, and its compression is a label code. [H20](h20_world.md) is
pre-registered: a label-free stream from hidden processes (a grammar, counters, cellular automata, motifs) switching
in seasons, scored in prequential bits per symbol with the store's size charged. It changes the world, not a knob.

Figure source: `figures/cls-bench/` (built with the MIT [Anatomy](https://github.com/wheresryan22/anatomy) kit).
See the [loop log](loop.md) for each test and the [audit](../audit.md) for the evidence.
