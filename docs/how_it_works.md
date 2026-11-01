# How it works

One machine, three parts, one coupling. The figure below is the current default core (`z5` store + `c3` router +
awake remote replay), drawn as a bench instrument. It runs on an illustrative digit stream; every rule and number in
it is the system's own.

<iframe class="fig" src="../figures/cls-bench.html" title="Fig 1: CLS dual bench, store, router, awake replay"></iframe>

## Reading the bench

| Part on the bench | Part of the learner | What it does |
|---|---|---|
| Card file (left), one slot per digit | Fast episodic store: pixel ProtoStore `a25w250h2k2_c1.5_uinf_z5` | Writes every waking arrival in one shot. A short-term buffer of patterns, not weights. A raised tab marks a digit unseen for ≥ 300 ticks. |
| Two-pan balance (centre) | Router `c3` | Weighs store against cortex on *this* image. The store answers only when it disagrees at relative margin ≥ 0.3 (the fixed rider on the cortex arm); otherwise the cortex answers. The only measurement on the answer path. |
| Lap plate (right) | Cortex: forward-forward weight space | The long-term compressor and the only part that changes connection strengths. Turns a little for each arrival it learns from, and 32× as much for a replay batch. |
| Overhead hoist | Awake remote replay | Every 38th waking arrival, carries one batch of 32 store memories, drawn only from digits absent ≥ 300 ticks, into the cortex. The single store → cortex coupling. Interleaved with answering: no nights, no downtime. |
| Ruler and pointer (front) | Replay clock | One replay period: 38 waking arrivals. |

## Numbers behind it

- Awake remote replay, held-out fresh seeds, two worlds: returns **+4.4 / +5.4**, lifetime accuracy **+0.05 / +0.15**;
  cortex-only returns 58–63 → 76–77. Scored on waking arrivals only.
- Remote (absent-only) beats uniform replay by 2.4 return points; similarity-weighted replay lost 4.4.

## What the bench leaves out, on purpose

- **No night cabinet.** Night replay is not in the default; whether it adds anything on top of awake replay is H7, running.
- **No store override by class state.** H6, H6b and H6c (trust the store when its digit is stale, burst-named, or just
  returned) were killed: on store-vs-cortex disagreements the store is right only 2–4% of the time, however the digit is
  gated. Only per-image confidence (`c3`) separates them.
- Also killed: similarity-weighted replay, random sparse DG expansion, cortex-vetted confirmation without an LR gap.

Figure source: `figures/cls-bench/` (built with the MIT [Anatomy](https://github.com/wheresryan22/anatomy) kit).
See the [loop log](loop.md) for each hour's tests and the [audit](../audit.md) for the evidence.
