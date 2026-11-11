# W1-0: build the real-data world and measure its curiosity headroom (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `docs/roadmap.md` (section "Real-data world, W1"),
`handoffs/2026-11-10_grok_kaggle_worlds.table.md`, `results/h29c/RESULTS.md`, `h29b_enwik8.py`.

## Context
M4 (one curious life) is paused. In the synthetic world, the oracle allocation beat uniform-random glancing by only
0.056 bits, so curiosity had nothing to win (H28). W1 replaces that world with real byte streams. Before any
curiosity run, we measure whether W1 has allocation headroom: does it matter which channel the learner reads? That
is the CLAUDE.md rule "check oracle headroom before setting a bar". Principle served: act to maximise compression
progress (4), by testing whether the world rewards it.

## Channels: byte streams; all are Kaggle datasets attached as kernel sources
| Channel | Kaggle source | Stream |
|---|---|---|
| `bible` | `tentotheminus9/religious-and-philosophical-texts`, file `pg10.txt` | Strip the Project Gutenberg header and footer (the text between the `*** START` and `*** END` lines). |
| `chess` | `datasnaek/chess`, `games.csv` | The `moves` column only, one game per line, in file order. Drop every other column. |
| `squad` | `stanfordu/stanford-question-answering-dataset`, `dev-v1.1.json` | Plain text in file order: for each paragraph, its context, then each question and its first answer text, newline-separated. |
| `btc` | `martinsn/high-frequency-crypto-limit-order-book-data`, `BTC_5min.csv` | The raw csv bytes, header included. |

Use the first 2,000,000 bytes of each stream as the training stream, and the next 100,000 bytes as that channel's
held-out block. If a stream is shorter than 2,100,000 bytes, report it and use 95% / 5% of it instead. Report the
length and sha256 of each stream.

## Measurement
Per channel, in its own fresh learner, use the H29c cortex at its best setting: `h29b_enwik8.py`'s model, lr 3e-4,
replay batch 8 (replay windows uniform over that channel's bytes seen so far), SEED 0. Do not change the model. Copy
the training loop into `w1_headroom.py` and add held-out evaluation. Run the training stream prequentially, scoring
each chunk before its step as H29 does, and record the prequential bits per 1 MB slice. At checkpoints n ∈ {0, 62,500,
125,000, 250,000, 500,000, 1,000,000, 2,000,000} bytes read, evaluate the held-out block with no gradient: zero
initial state, chunks of 128, carried state, live −log2 p. That gives L_c(n) in bits/byte.

Also report per channel: xz -9e and PPMd order 16 (same settings as H29) on the 2,000,000-byte training stream, in
bits/byte.

## Headroom (compute it exactly like this)
Budget B = 2,000,000 bytes in total across the 4 channels. Allocations are n_c taken from the checkpoint grid with
Σ n_c ≤ B.
- uniform = mean over c of L_c(500,000);
- oracle = the minimum, over all grid allocations with Σ n_c ≤ B, of the mean over c of L_c(n_c), found by exhaustive
  search over the 7^4 grid tuples;
- headroom = uniform − oracle, in bits/byte. Also report the oracle's allocation.
State in RESULTS.md that these are separate learners per channel. The real machine shares one cortex, so this is an
estimate.

## Kernels
Two private GPU kernels with two channels each: `ezzzzzx/w1-headroom-a` (bible, chess) and `ezzzzzx/w1-headroom-b`
(squad, btc). Both have `"is_private": true`, `"enable_gpu": true` and `"enable_internet": false`. The Kaggle datasets
above are attached via `dataset_sources`, and the code goes up as private dataset `ezzzzzx/pdp-code-w1`. H29d uses the
2 GPU slots first: push these kernels only when slots are free, and poll.

## Smoke test (VPS CPU, before any kernel)
On the VPS, download only `pg10.txt` (`kaggle datasets download ... -f pg10.txt` into `data/`), build the bible stream,
then run 4,096 training bytes with a held-out evaluation of 1,024 bytes at n = 0 and n = 4,096. The bits must be
finite, and the training bits must equal an H29c-equivalent run on the same bytes to 1e-6 (same model, seed, lr and
replay batch). If either fails, stop.

## Outputs
`results/w1_headroom/RESULTS.md` and `results/w1_headroom/results.json`, with numbers only and no verdict: stream
lengths and hashes; the L_c(n) table; prequential slice bits; the xz and PPMd numbers; uniform, oracle, headroom and
the oracle allocation; kernel ids, elapsed seconds and GPU hours. Put a 300-byte excerpt of each stream in the
RESULTS, with nothing personal: the chess ids are dropped.

## Pre-registered bar (orchestrator scores; do not change)
**W1 qualifies for the curiosity run (M4 on W1) if headroom ≥ 0.10 bits/byte.** The synthetic world had 0.056. If it
is below 0.10, the world needs a redesign before any curiosity spec.

## Budget and rules
- At most 2 GPU hours in total, with a 90-minute wall-clock target after the slots free up. A failed kernel may be
  rerun once.
- Run in the foreground and poll with `kaggle kernels status`.
- Files you may create: `w1_headroom.py`, `kaggle/w1/**`, `results/w1_headroom/**`, and `data/` (gitignored).
- Do not touch: `docs/loop.md`, `audit.md`, any `h29*` file or folder, `.env`.
- Never print or upload secrets. Do not commit.
