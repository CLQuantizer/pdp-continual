# W1b: add saturating and unlearnable channels to W1; remeasure headroom (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `results/w1_headroom/RESULTS.md` (including its Verdict), `w1_headroom.py`,
`kaggle/w1/`.

## Context
W1-0 measured allocation headroom on 4 real channels (bible, chess, squad, btc): 0.039 bits/byte. That fails the 0.10
bar. All 4 are learnable and unsaturated at 2 MB, so allocation hardly matters. Curiosity pays when channels differ in
kind. W1b adds three channels, one per missing kind, and recomputes headroom over the full set. Principle served: act
to maximise compression progress (4), by testing whether the world rewards it.

## New channels (the 4 W1-0 channels are reused, not rerun)
| Channel | Source | Stream | Kind |
|---|---|---|---|
| `dna` | Kaggle `nageshsingh/dna-sequence-dataset`, `human.txt` | The sequence column only: the bases (A, C, G, T, N), one sequence per line, in file order. Drop the header and class labels. | quickly learnable, small alphabet |
| `noise` | Synthetic: `numpy.random.default_rng(777).integers(0, 256, size=2_100_000, dtype=uint8)` | uniform random bytes | unlearnable (noisy TV) |
| `dark` | Synthetic: 2,100,000 zero bytes | constant | saturates almost at once |

Each channel gets the same split as W1-0: 2,000,000 training bytes and the next 100,000 held out. Use the same
learner (lr 3e-4, replay batch 8, SEED 0), the same checkpoint grid {0, 62,500, 125,000, 250,000, 500,000, 1,000,000,
2,000,000} and the same held-out scoring. Extend `w1_headroom.py` by copying it to `w1b_headroom.py` and adding the 3
builders. Do not edit `w1_headroom.py`. Take the W1-0 curves for bible, chess, squad and btc from
`results/w1_headroom/results.json`. Also report xz -9e and PPMd order 16 (using W1-0's piecewise method) for dna, noise
and dark.

## Headroom (compute exactly like this, for two channel sets)
For a channel set S with k channels: budget B = 500,000 × k bytes, with each n_c on the checkpoint grid and Σ n_c ≤ B.
- uniform = mean over S of L_c(500,000);
- oracle = the minimum, over all grid allocations with Σ n_c ≤ B, of the mean over S of L_c(n_c), found by exhaustive
  search (7^k tuples);
- headroom = uniform − oracle, plus the oracle's allocation.

Sets:
1. **full**: all 7 channels (bible, chess, squad, btc, dna, noise, dark).
2. **learnable**: bible, chess, squad, btc, dna (noise and dark excluded).

## Kernel
One private GPU kernel, `ezzzzzx/w1b-headroom`, running dna, noise and dark in sequence. It has `"is_private": true`,
`"enable_gpu": true` and `"enable_internet": false`. The DNA dataset is attached through `dataset_sources`, and the
code goes up as private dataset `ezzzzzx/pdp-code-w1b`.

## Smoke test (VPS CPU, before the kernel)
1. Build the noise and dark streams locally and run 4,096 training bytes with a 1,024-byte held-out evaluation at n = 0
   and n = 4,096. Bits must be finite. Dark held-out bits at n = 4,096 must be below those at n = 0.
2. `w1b_headroom.py` on 4,096 Bible bytes must reproduce W1-0's smoke value, 19362.707616529544 bits, to 1e-6.
If either check fails, stop.

## Outputs
`results/w1b_headroom/RESULTS.md` and `results/w1b_headroom/results.json`, with numbers only and no verdict: the new
streams' lengths, hashes and 300-byte excerpts (DNA only); the L_c(n) table for all 7 channels; xz and PPMd for the new
3; for both sets, uniform, oracle, headroom and the allocation; kernel id, elapsed time, GPU hours.

## Pre-registered bars (orchestrator scores; do not change)
1. **W1b qualifies for M4-on-W1b if full headroom ≥ 0.10 bits/byte.**
2. Report only: the learnable headroom. If it is below 0.05, record that most of the gain comes from avoiding noise
   and dark rather than from choosing between learnable streams.

## Budget and rules
- At most 1 GPU hour, 90 minutes of wall clock. A failed kernel may be rerun once.
- Run in the foreground and poll with `kaggle kernels status`.
- Files you may create: `w1b_headroom.py`, `kaggle/w1b/**`, `results/w1b_headroom/**`, and `data/` (gitignored).
- Do not touch: `docs/loop.md`, `audit.md`, `w1_headroom.py`, `kaggle/w1/`, any `h29*`, `.env`.
- Never print or upload secrets. Do not commit.
