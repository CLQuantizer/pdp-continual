# H26 content-keyed replay on enwik8 E3M

E3M bytes: 3000000. sha256: `72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac`.
E10M sha256, checked and not scored: `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
GRU bits/byte = sum of live −log2 p / 3000000. Replay is not part of the code length.

Fixed settings: lr 0.0003, seed 0, replay seed 30000, embedding 64, hidden 1024,
chunk 128, clip 1.0, Adam, sleep block 1, replay batch 8.
Projection P is 128 x 1024, entries N(0, 1/128) with variance 1/128,
generator seed 2626. Keys and the cue are L2-normalised. Eligible chunks
satisfy c <= current chunk - 16. A cued window is uniform on the top 32 eligible
chunks, or on all eligible chunks when fewer than 32 are eligible.

## E3M

| Arm | Bits/byte | Replay steps | Wall seconds | GPU |
|---|---:|---:|---:|---|
| uniform | 1.958153 | 23438 | 1098.899 | Tesla T4 |
| cued8 | 2.020962 | 23438 | 1175.909 | Tesla T4 |
| mix44 | 1.973144 | 23438 | 1116.194 | Tesla T4 |

## 1 MB slices

### uniform

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2104744.492 | 2.104744 |
| 1 | 1000000 | 2000000 | 1923321.387 | 1.923321 |
| 2 | 2000000 | 3000000 | 1846393.123 | 1.846393 |

### cued8

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2155571.684 | 2.155572 |
| 1 | 1000000 | 2000000 | 1992259.715 | 1.992260 |
| 2 | 2000000 | 3000000 | 1915054.961 | 1.915055 |

### mix44

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2118743.415 | 2.118743 |
| 1 | 1000000 | 2000000 | 1938800.529 | 1.938801 |
| 2 | 2000000 | 3000000 | 1861888.489 | 1.861888 |

## Gaps

Pre-registered differences. Not scored in this file.

cued8 − uniform = 0.062809 (2.020962 − 1.958153).
mix44 − uniform = 0.014991 (1.973144 − 1.958153).

Bar 1, not scored here: mix44 − uniform ≤ 0.01 bits/byte.
Bar 2, not scored here: cued8 − uniform > 0.01 bits/byte.

## Age and coverage

Age = current chunk − (window start // 128), in chunks. A cued window starts
at the retrieved chunk. Mean cued age is over content-index windows only.
Uniform expected age is the expected age of one uniform start, averaged over
the same windows (one expectation per cued window, at that step).
Uniform realized age is the uniform arm, on steps with current chunk >= 16.
The recent 10% is chunk >= floor(0.9 * n_eligible), with eligible indices
0 .. n_eligible−1 and n_eligible = current chunk − 15.
Coverage is the fraction of scored chunks whose byte range contains at least
one replay window start. Stored chunks are the chunks scored in the run.

| Arm | Cued windows | Mean cued age | Uniform expected age at those windows | Fraction of cued windows in recent 10% | Chunks retrieved | Stored chunks | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 0 | na | na | na | 20775 | 23438 | 0.886381 |
| cued8 | 187376 | 6086.5527 | 5863.7461 | 0.096704 | 19574 | 23438 | 0.835140 |
| mix44 | 93688 | 6035.8982 | 5863.7461 | 0.095839 | 20581 | 23438 | 0.878104 |

Uniform arm realized mean age on steps with current chunk >= 16: 5875.0693 over 187376 windows.
Uniform arm expected mean age on those windows: 5863.7461.
Uniform arm fraction of those windows in the recent 10% of the eligible log: 0.100077.
Uniform arm coverage: 0.886381 (20775 / 23438).

## Kernels

Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that
clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.
Script wall is time.monotonic from kernel_main entry through the result write. Queue time
is not included. Total GPU hours is the sum across the three kernels.

| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours |
|---|---:|---|---:|---|---:|---:|
| ezzzzzx/h26-uniform | 1 | COMPLETE | 1133.401 | log_clock | 1110.756 | 0.3148 |
| ezzzzzx/h26-cued8 | 1 | COMPLETE | 1208.419 | log_clock | 1187.986 | 0.3357 |
| ezzzzzx/h26-mix44 | 1 | COMPLETE | 1148.051 | log_clock | 1127.762 | 0.3189 |

Total GPU hours: 0.9694.

Kaggle accepted two GPU sessions at a time. ezzzzzx/h26-uniform version 1 and ezzzzzx/h26-cued8 version 1 ran together. ezzzzzx/h26-mix44 version 1 was pushed after both of those reached COMPLETE. No kernel was rerun.

## Protocol

- One online pass per arm. Prequential: −log2 p of each byte is recorded before that chunk is trained.
- Sleep block 1: score the chunk, one Adam step, then one replay step. That is the H31 inter path.
- Each replay step trains 8 windows of 128 bytes from a zero hidden state. The live state is kept.
- uniform draws one rng.integers of size 8, the H31 sampler, seed 30000.
- cued8 draws 8 windows from the content index. mix44 draws 4 uniform starts, then 4 cued windows.
- Before 16 chunks are eligible, a cued window uses the uniform sampler (one integers draw of that width).
- Keys are written once, from the hidden state at the start of the chunk, and are not refreshed.
- The cue is the hidden state after the current chunk. Cosine is the dot product of L2-normalised vectors.
- E3M is bytes [0, 3000000). 1 MB slices are 1000000 bytes.
- Traffic = replay steps × 8 × 128 / bytes.
- Code is h26_cued.py. Private dataset ezzzzzx/pdp-code-h26. Data dataset ezzzzzx/enwik8.

## Smoke

CPU. Not a score.

Bytes: 8192.
h31 --sleep-block 1 bits: 42063.298164869797.
h26 uniform bits: 42063.298164869797.
Absolute difference: 0.000e+00.
Within 1e-6: True.

Finite bits on 8192 bytes:

- uniform: bits=42063.298164869797 finite=True replay_steps=64.
- cued8: bits=43091.052702231464 finite=True replay_steps=64.
- mix44: bits=42494.282295086632 finite=True replay_steps=64.

Retrieval check: arm cued8, bytes 65536.
Logged cued steps: 5.
Every logged index <= cue chunk - 16: True.

H26 retrieval i=1 cue_chunk=16 n_eligible=1 retrieved=[0, 0, 0, 0, 0, 0, 0, 0] cosines=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
H26 retrieval text i=1 cue_chunk=16 cue_byte=2048 cue=b'  <text xml:space="preserve">#REDIRECT [' top_chunk=0 top_byte=0 top=b'<mediawiki xmlns="http://www.mediawiki.o'
H26 retrieval i=2 cue_chunk=17 n_eligible=2 retrieved=[1, 0, 1, 0, 1, 0, 0, 1] cosines=[0.272199, 0.0, 0.272199, 0.0, 0.272199, 0.0, 0.0, 0.272199]
H26 retrieval i=3 cue_chunk=18 n_eligible=3 retrieved=[0, 1, 0, 0, 2, 1, 2, 0] cosines=[0.0, 0.307362, 0.0, 0.0, 0.301685, 0.307362, 0.301685, 0.0]
H26 retrieval i=4 cue_chunk=19 n_eligible=4 retrieved=[1, 3, 1, 1, 1, 3, 3, 2] cosines=[0.307255, 0.576104, 0.307255, 0.307255, 0.307255, 0.576104, 0.576104, 0.282157]
H26 retrieval i=5 cue_chunk=20 n_eligible=5 retrieved=[4, 2, 3, 2, 1, 3, 4, 4] cosines=[0.625052, 0.278931, 0.543275, 0.278931, 0.282983, 0.543275, 0.625052, 0.625052]

## Verdict (orchestrator, 12 Nov 2026, UTC+8)

**Bar 1 FAIL (narrowly). Prediction 2 right.**
- Bar 1: mix44 − uniform = +0.0150 bits/byte (bar ≤ 0.01). Half content-cued, half broad replay comes close to
  uniform-position replay, but not within the bar.
- Prediction 2: cued8 − uniform = +0.063 (> 0.01). Pure content-cued replay hurts, the fourth time selective replay
  has hurt (after H10, M4a and H27). It throws away about a quarter of replay's 0.240 gain.
- The uniform arm matched H31 `inter` exactly, so the T4 runs were deterministic here.

**Mechanism: redundancy, not age.** Cued windows are not more recent; they are slightly *older* than uniform draws
(6,087 vs 5,875 chunks), with the same share from the most recent tenth (9.7% vs 10.0%). Coverage drops only a
little (0.835 vs 0.886 of chunks ever replayed). What changes is *what* is replayed. The cue is the current state, so
the 8 windows come from the 32 chunks most similar to *now*. The cortex already predicts that kind of text, and the
live step is already training on it. The replay batch adds gradient that overlaps with the live step and with itself,
and less of the far-away text the cortex is forgetting. Replay's job in a living compressor is to counter forgetting
of what is *not* in the current input, and similarity-to-now selects against exactly that. The mix pays about half
of what a linear share would (0.015 vs 0.031), because its uniform half still covers the store.

**What this says about the brain-address question.** Content addressing is right for *retrieval*: using the past to
predict now. It is wrong as the *sampler for consolidation*. Consolidation needs broad, cue-independent sampling, and
the brain's candidates for that are spontaneous replay in rest and sleep (attractor wandering). Uniform position
sampling is our RAM stand-in for it. An address-free broad sampler worth testing later: a random cue each step (a
random direction in key space) in place of the current state.
