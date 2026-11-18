# H35c: curiosity against real data-cleaning pipelines, on raw web text, with transformers at two sizes, compute counted (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` (GPU quota guard), then `handoffs/2026-11-17_grok_h35_data_diet.md`, the scoring table in `results/h35/RESULTS.md`,
and your own `h35_data_diet.py`.

## The three questions a frontier lab would ask, answered by experiment
1. On **real raw web data**, does learning-progress selection match the hand-built filters labs use (heuristics, deduplication, perplexity
   filtering, a quality classifier)?
2. Is it worth its cost: **counting the selection compute**, does it beat random selection given the same total compute?
3. Does the gain **hold as the model grows**?

## Data (prepare once; upload as private Kaggle datasets; never upload secrets)
- **Pool:** 60,000 documents from C4 `en.noclean` (raw Common Crawl text with only language ID; Hugging Face `allenai/c4`, config `en.noclean`;
  stream the first shards, take documents in order, drop ones under 200 bytes, truncate to 4,096 bytes). Record the shard files and document
  ids.
- **Clean evaluation sets (never in the pool):** (a) 2 MB of C4 `en` validation (the filtered C4); (b) the WikiText-103 test set (raw). Report
  both. The primary score is (a).
- If `en.noclean` cannot be streamed, use another public raw Common Crawl text sample, document the choice and stop before kernels to report.

## Model and training (same for every arm)
Byte-level decoder-only transformer, context 512, AdamW lr 6e-4 with 200 warm-up steps and cosine decay to 10%, weight decay 0.1, batch 32
sequences of 512 bytes, bf16 or fp16 autocast on a T4, fixed training steps S.
- **small:** 4 layers, d_model 256, 4 heads (≈ 3 M params). S = 3,000 steps.
- **medium:** 8 layers, d_model 512, 8 heads (≈ 25 M params). S = 3,000 steps.
Training sequences come only from the documents the arm has selected, and each selected document is read once in order of selection (cut into
512-byte sequences). The arm keeps selecting until it has 3,000 × 32 sequences.

## Arms (selection rules), each picks documents from the pool
Baselines labs use:
- `rand`: uniform order.
- `heur`: C4/Gopher-style rules: keep lines ending in terminal punctuation, drop documents with < 3 sentences or < 50 words, mean word length
  outside [3, 10], symbol-to-word ratio > 0.1, > 30% lines starting with a bullet, the words "lorem ipsum" or "javascript", or < 2 of the stop words
  {the, be, to, of, and, that, have, with}. Then uniform among survivors.
- `dedup`: exact dedup on normalised text (lower case, whitespace collapsed) plus MinHash near-dedup (5-grams of words, 128 permutations, Jaccard
  ≥ 0.8, `datasketch`), then uniform.
- `ppl`: perplexity filter (CCNet-style): a reference byte model (the small architecture trained 1,000 steps on 5 MB of English Wikipedia, kept
  frozen) scores every pool document; keep the 50% with the lowest bits/byte; uniform among them.
- `cls`: the FineWeb-Edu quality classifier (`HuggingFaceFW/fineweb-edu-classifier`); keep documents with score ≥ 2 (or the top 50% if fewer than
  half pass); uniform among them.
- `pipeline`: heur → dedup → cls in that order (the strongest hand-built pipeline); uniform among survivors.
Learning-progress and loss-based selection (online: chosen in rounds while training):
- `high_loss`: every round of 32 sequences, draw 128 candidate documents uniformly from unread ones, score each by the current model's bits/byte
  on its first 512 bytes, take the top ones until 32 sequences are filled.
- `rho`: as high_loss, score = current bits/byte − the frozen `ppl` reference model's bits/byte (RHO-LOSS).
- `prog_snap` (curiosity): as high_loss, score = bits/byte on the peek under a **snapshot of the model from 200 steps ago** minus under the current
  model: how much the model has recently learned about this kind of text. Noise has a flat score (nothing learned), and repeated or already-learned
  text has a flat score too. Before step 200 the snapshot is the initial model. The snapshot is refreshed every 200 steps.
- `rand_plus`: random selection, trained for **S + Δ** steps, where Δ is the number of extra training steps whose compute equals `prog_snap`'s
  selection compute at that size (forward FLOPs of all scoring passes ÷ FLOPs of one training step, measured, rounded up). This answers question 2.
All constants are fixed in advance; do not tune.

## Compute accounting (required)
For every arm: training FLOPs, selection FLOPs (scoring passes, reference models, classifier inference on the pool, MinHash excluded but its wall
time reported), wall seconds of each. Report selection compute as a percentage of training compute.

## Seeds, kernels, GPU budget
Seeds 0 and 1 (seed changes model init, pool order for `rand`, candidate draws). 10 arms × 2 sizes × 2 seeds = 40 runs. **Before any GPU push run
`kaggle quota`**, keep 10 GPU hours of weekly reserve, no single kernel over 3 GPU hours, and plan the whole of H35c within **14 GPU hours**. If the
smoke timing says it will not fit, cut in this order and say so: (1) seeds to 1 for `small`; (2) S to 2,000 for both sizes; then stop and report.
Pool scoring for `ppl` and `cls` can run once per size-independent pass and be reused. Private kernels, `"enable_internet": false` (data and the
classifier come from your private datasets), GPU on.

## Smoke test (VPS CPU and one short GPU kernel, before the scored kernels)
1. Data: print pool size, byte count, 3 random pool documents (first 200 bytes), and each filter's survival rate on the pool.
2. Training: the small model's loss falls over 200 steps on random selection (one short GPU kernel; report its GPU minutes).
3. `prog_snap`: on that short run, print the score distribution of 128 candidates at step 400, and the 3 highest and 3 lowest candidates (first 200 bytes).
4. Prequential note: selection uses only the model's own past state; the evaluation sets are never scored during selection.
5. Timing per arm and size; the GPU-hour plan.
If any check fails, stop and report.

## Outputs
`results/h35c/RESULTS.md` + `results.json`, numbers only, no verdict. **At the top a scoring table** per size and seed: rand, pipeline, prog_snap,
rand_plus on C4-val bits/byte; the gains G_pipe = rand − pipeline and G_prog = rand − prog_snap; G_prog / G_pipe; prog_snap − rand_plus; and
G_prog(medium) / G_prog(small); with counts meeting each threshold. Then every arm's C4-val and WikiText bits/byte per size and seed, the compute
table, filter survival rates, how many selected documents each arm shares with `pipeline`'s survivors, 10 example documents picked first and 10
rejected most often by `prog_snap` (200 bytes each), smoke output, walls, GPU hours, kernel ids.

## Pre-registered bars (orchestrator scores; do not change), on C4-val bits/byte
**Precondition (cleaning matters at this scale):** G_pipe ≥ 0.02 bits/byte at both sizes (mean over seeds). Else the bars are void.
1. **Curiosity matches a lab pipeline:** G_prog ≥ 0.5 · G_pipe at both sizes, on both seeds.
2. **Worth its compute:** prog_snap − rand_plus ≤ −0.01 at both sizes, on both seeds.
3. **Holds with scale:** G_prog(medium) ≥ 0.8 · G_prog(small) (means over seeds).
Report only: rho, high_loss, heur, dedup, ppl, cls, WikiText numbers.

## Budget and rules
- Implementation and smoke within about 3 hours of your time; then kernels in the foreground until all finish (allow a long wait).
- Files you may create: `h35c_web.py` (and helpers `h35c_*.py`), `kaggle/h35c/**`, `results/h35c/**`, `data/h35c/**` (gitignored data only).
- Do not touch: `docs/`, `audit.md`, `h35_corpus.py`, `h35_data_diet.py`, other experiment code, other `kaggle/` or `results/` folders, `.env`. Kaggle
  CPU kernels from H35b/H36 are running: keep the account's running kernels ≤ 5.
- Gates and bars are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.

## Amendment (orchestrator, 18 Nov 16:30, after the first scored run)
`prog_snap`'s scoring cost 2.5× its training compute (128 candidates × 512-byte peeks × 2 forward passes per 32 training sequences), so Δ > S and
`rand_plus` ran out of pool. That cost is a design choice, not a property of the idea. **Changed for the online arms (`high_loss`, `rho`, `prog_snap`):
peeks are the first 128 bytes, and K = 96 candidates per round.** This puts selection near 0.5× training compute. **`rand_plus` may start a second epoch
over the pool if it runs out** (report how many sequences it re-read). Everything else, including the bars, is unchanged. The completed `rand` runs stay
valid (they do not use peeks). Set `results/h35c/plan.json` accordingly and continue with the scored kernels within the same GPU budget.
