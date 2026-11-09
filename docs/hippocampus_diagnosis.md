# Why the hippocampus looks useless: a diagnosis (9 Nov 2026, UTC+8)

**Short answer:** the store isn't weak. Three of our choices together gave it nothing to do. Two are about the
**data**, one about the **store design**, and one about the **metric**.

## The evidence we already have

| Observation | Source |
|---|---|
| On store-vs-cortex disagreements the store is right only 2–4% of the time, however the digit is gated | H6/H6b/H6c, audit 2026-11-01 |
| The age gate fired 1 762 times; 1 506 of those were on inputs FF already got right, and the store was right on only 131 | loop 2026-10-30 |
| A perfect (oracle) router adds at most **1.9 lifetime points** over FF | loop 2026-10-30 17:00 |
| FF error is 45.8% in the first 100 ticks after a return, 3.4% after 3 000 ticks; returns are ~1% of arrivals | loop 2026-10-31 19:00 |
| "Prequential lifetime accuracy weights memories by how often they're queried, and the same frequency rehearses them" | loop 2026-10-30 15:00 |
| Replay does compress: ff_bits 0.515 → 0.460 with awake replay; returns +4–5 | H10, H3c/H4c |

## Diagnosis

### 1. Data: MNIST contains no episodes

Every "3" is interchangeable with every other "3". Nothing about image #48 213 needs remembering *as that image*. All
the regularity is class statistics, which is exactly what a cortex is for. In complementary learning systems theory
(McClelland et al. 1995; Kumaran et al. 2016) the hippocampus earns its keep on **specific, arbitrary,
one-shot** associations, the kind class structure cannot predict (where I parked, what this person is called), and by
re-presenting rare experience. Our world has neither. The store and the cortex were set to compete at the cortex's
own job, and the cortex wins 96–98% of disagreements.

### 2. Data plus metric: the query distribution equals the rehearsal distribution

Lifetime accuracy sums over arrivals. The store can only add value where the cortex is wrong and the store is right.
In this world that happens almost only right after a return (~1% of arrivals). Background arrivals (bg 0.2 or 0.02)
and recurring seasons mean **the world does the interleaved replay itself, for free**. So the ceiling on the store's
contribution is set by the world, at ≤ 1.9 points, and roughly 19 hypotheses (H1–H19) fought over that 1.9.

### 3. Store design: our "hippocampus" is a second cortex

`ProtoStore` (sleep.py) keeps **9 running-mean prototypes per class**. After the first 9 images of a class, every new
image is *averaged into* its nearest prototype (lr = 1/n). That is a small class-conditional mixture model: semantic,
statistical memory. It destroys exactly what makes episodic memory useful: pattern separation, one specific trace per
experience. "One-shot episodic writes" in the code comments is a misnomer. Even if the world had episodes, this store
would blur them.

### 4. Methodology: the store is tested as a rival answerer

The router asks "on this image, is the store or the cortex right?" That is a contest over the cortex's specialty. In CLS
the hippocampus has two main jobs:

- **Bind fast.** Answer the things only it can know (items seen once or twice).
- **Teach.** Replay, so the cortex absorbs regularities from experiences the world does not repeat.

In Schmidhuber's terms it is the **stored history** that the compressor keeps re-compressing; it is what makes
"compression progress" measurable at all (old and new compressor scored on the same data). Its value should be
measured as a counterfactual (bits or accuracy saved per stored bit), not as a share of router wins.

## What would make it useful (predictions)

The hippocampus should become clearly valuable when (a) the world has item-specific, arbitrary content that recurs
after gaps, (b) the store keeps separated traces instead of averages, and (c) the score counts what only memory can
supply.

## H24a: decisive diagnostic (pre-registered before the run)

`h24_episodic.py`. The MNIST stream (true labels) plus, with probability Q = 0.2 per tick, an *episode*: one of K = 200
specific test-split images, each with a fixed arbitrary label. Each episode recurs about every 1 000 ticks.
10 000 ticks, seeds 6–8, FF hidden 500 (life.py's cortex). Routes on the same arrivals: `ff` (cortex), `proto` (the
core's ProtoStore), `exem` (verbatim exemplar store, nearest neighbour), `gate` (exem answers only when its
nearest neighbour is near-identical, d² < 1, else ff), `ffr` (FF plus 8 exemplar replays per arrival).

Predictions (mean over seeds):
- **P1:** exem on repeat episodes (exposure ≥ 2) ≥ 95%.
- **P2:** proto on episodes at exposure 6+ ≤ 30% (averaging blurs episodes once prototypes are full).
- **P3:** ff on episodes at exposures 3–5 ≤ 50%.
- **P4:** gate lifetime ≥ ff + 8 points, with gate firing on ≤ 0.5% of normal arrivals and normal-arrival accuracy
  within 0.5 of ff.
- **P5 (consolidation):** ffr's own answer on episodes at exposure 6+ ≥ ff + 20, normal accuracy within 1 point of ff.

**What it decides:**
- If P1, P2 and P4 hold, the diagnosis stands. The loop's world and store were the problem, not the idea of a
  hippocampus. Future worlds must contain episodic content, and the default store must keep separated traces.
- If P1 or P4 fails, the diagnosis is wrong and I reopen it.
- P5 decides whether replay can move episodes into the cortex. If it can, store entries become deletable once
  consolidated: Schmidhuber's history compressor, and the store priced in bits (H20's T).

## Follow-ups (for the loop)

- **H24b (symbol world):** add an episodic process to the H20 world: a seeded dictionary of arbitrary `key=value;`
  records whose keys recur after long gaps. Score store value in bits saved per stored bit under H20's two-part T.
- **Default store:** exemplar (pattern-separated) traces with a surprise-gated write (store what the cortex failed to
  predict) and deletion after consolidation. Replaces running-mean prototypes for anything episodic.
