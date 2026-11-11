# W1c: allocation headroom for ONE shared cortex on W1b's 7 channels (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `results/w1b_headroom/RESULTS.md` (including its Verdict), `w1b_headroom.py`,
`kaggle/w1b/`.

## Context
W1-0 and W1b measured headroom with a separate learner per channel: 0.039 on the 4 rich channels and 0.081 on all 7.
The real machine (M4) is one cortex reading every channel. There, reading noise also spends capacity and replay and
interferes with learned channels, and per-channel learners cannot see that. W1c measures headroom on one shared cortex,
with fixed reading schedules. Principle served: act to maximise compression progress (4), by checking whether the
world rewards choosing.

## Channels and data
The same 7 streams and splits as W1b: bible, chess, squad, btc, dna, noise, dark. Each has a 2,000,000-byte training
prefix and the next 100,000 bytes held out. Build them with W1b's builders; noise uses rng 777, dark is zeros.

## Learner
One H29c cortex: the model from `h29b_enwik8.py`, lr 3e-4, chunks of 128, TBPTT 128, clip 1.0, SEED 0. Copy W1b's
code into `w1c_shared.py` and add the schedule; do not edit earlier files.
- **Reading:** total budget B = 3,500,000 bytes, read in 128-byte chunks. Before each chunk, pick the channel with the
  smallest (bytes read from c) / quota_c among channels with quota_c > 0. Ties go to the lowest channel index in the
  order above. Read that channel's next 128 bytes from its own position.
- **State:** keep a separate carried hidden state per channel (detached at chunk boundaries); the weights are shared.
- **Prequential:** score each chunk (live −log2 p) before its Adam step, as in H29.
- **Replay:** after each live chunk, one step of batch 8 windows of 128 bytes with zero initial state (traffic 8). For
  each window, pick a channel with probability proportional to the bytes read from it so far, then a uniform start
  within that channel's read bytes.
- **Held-out:** after 875,000, 1,750,000 and 3,500,000 total bytes, score every channel's held-out block with no
  gradient (zero initial state, chunks of 128, carried state). Report L_c in bits/byte, and the mean over all 7
  channels, "M7".

## Arms (fixed schedules, quotas in bytes, each summing to 3,500,000)
| Arm | bible | chess | squad | btc | dna | noise | dark |
|---|---:|---:|---:|---:|---:|---:|---:|
| `U7` uniform | 500,000 | 500,000 | 500,000 | 500,000 | 500,000 | 500,000 | 500,000 |
| `N0` skip noise | 583,333 | 583,333 | 583,334 | 583,333 | 583,333 | 0 | 583,334 |
| `OR` W1b oracle, rescaled | 1,000,000 | 1,000,000 | 1,000,000 | 250,000 | 125,000 | 0 | 125,000 |

## Headroom
headroom_shared = M7(U7) − min(M7(N0), M7(OR)), all at 3,500,000 bytes. Also report each arm's mean over the 5
learnable channels, and each arm's per-channel L_c table.

## Kernels
Three private GPU kernels: `ezzzzzx/w1c-u7`, `ezzzzzx/w1c-n0`, `ezzzzzx/w1c-or`. Each has `"is_private": true`,
`"enable_gpu": true` and `"enable_internet": false`; the Kaggle datasets go in `dataset_sources`, and the code goes up
as `ezzzzzx/pdp-code-w1c`. Kaggle allows 2 GPU sessions, so queue the third.

## Smoke test (VPS CPU, before any kernel)
Run U7 with B = 7 × 512 bytes and held-out blocks cut to 1,024 bytes. Check that each channel read exactly 512 bytes,
the bits are finite, and the replay windows came only from read bytes (assert it). Run N0 the same way: noise must read
0 bytes. If any check fails, stop.

## Outputs
`results/w1c_shared/RESULTS.md` and `results/w1c_shared/results.json`, with numbers only and no verdict: per arm,
bytes read per channel, prequential bits per channel, the L_c tables at the 3 checkpoints, M7 and the 5-channel mean;
headroom_shared and which arm gives the minimum; kernel ids, elapsed time, GPU hours.

## Pre-registered bar (orchestrator scores; do not change)
**W1 qualifies for M4 (curiosity on one shared cortex) if headroom_shared ≥ 0.10 bits/byte.** Report only:
M7(U7) − M7(N0), the noisy-TV cost in a shared cortex, to compare with the separate-learner value (W1b).

## Budget and rules
- At most 1.5 GPU hours in total, 90 minutes of wall clock. A failed kernel may be rerun once.
- Run in the foreground and poll with `kaggle kernels status`.
- Files you may create: `w1c_shared.py`, `kaggle/w1c/**`, `results/w1c_shared/**`, and `data/` (gitignored).
- Do not touch: `docs/loop.md`, `audit.md`, `w1_headroom.py`, `w1b_headroom.py`, `kaggle/w1/`, `kaggle/w1b/`, any
  `h29*`, `.env`.
- Never print or upload secrets. Do not commit.
