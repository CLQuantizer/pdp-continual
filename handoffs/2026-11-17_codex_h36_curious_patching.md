# H36: a curious tokeniser. Where should an expensive model spend its steps: fixed patches, surprise, or where it helps? (codex: implement + smoke; kernels launched later)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-17_grok_h35_data_diet.md` (the polluted corpus; grok writes `h35_corpus.py` today), and
`h33_bank_text.py` (counts_mix6) and `h26_cued.py` (the GRU cortex).

## Context
A tokeniser decides the units an expensive model steps on. BPE fixes them by frequency before training. Byte-level models with learned
patches (Byte Latent Transformer, Meta 2024) start a new patch where a small byte model is surprised (high next-byte entropy). Surprise has
the noisy-TV problem: random bytes are maximally surprising, so entropy patching should spend the most expensive steps on junk. A
curiosity-style rule spends them where the expensive model actually helps or is still learning. H36 tests this as compute allocation: at a
fixed budget of expensive steps, which placement rule gives the best bits on clean text? Principles served: 2 and 3.

## Stream
The H35 polluted pool (`h35_corpus.build(seed)`), read **passively** in the pool's shuffled order, first **400 documents** (1,638,400 bytes,
about half junk). The H35 clean test set for scoring. Until grok's `h35_corpus.py` exists, develop and smoke-test on a stand-in you build
inside `h36_curious_patching.py`: 64 clean 4,096-byte documents from enwik8 bytes [0, 262,144) interleaved with 64 uniform-random-byte
documents (rng 0). Scored kernels must use `h35_corpus.build(seed)` (import it; if it is missing when the kernels are prepared, prepare the
kernels anyway and say so).

## Models
- **Cheap model:** counts_mix6 as H33/H35 (orders 1–6, context reset at document starts). It predicts **every** byte and learns from every byte.
- **Expensive model:** a byte GRU, hidden 256, embedding 32 (`h26_cued`'s GRU design scaled down), Adam lr 3e-4, clip 1.0. It takes a step
  **only at chosen bytes**. At a chosen byte t its input is [emb(x_{t−1}); mean of emb over the bytes since its previous step] and it outputs
  a next-byte distribution for byte t. Training: truncated BPTT over its own last 32 steps, one Adam step per 32 of its steps.
- **Combination:** at a chosen byte, the GeoMixer mixes the cheap and expensive distributions (weights init [1, 0], Adam 0.003, one step per
  128 chosen bytes); at other bytes the cheap model alone predicts. Every byte is scored before anything learns from it.
- **Budget:** the expensive model steps on a fraction ρ = 0.25 of bytes, on average.

## Placement rules (decided before x_t is seen; causal)
- `fixed`: every 4th byte (positions ≡ 3 mod 4 within the stream).
- `entropy`: byte t is chosen when the cheap model's predictive entropy at t is above the running 75th percentile of that entropy (a P²
  quantile estimator or a sliding window of the last 4,096 entropies), so ≈ 25% are chosen.
- `gain`: buckets b = (cheap entropy decile at t) × (log2 bucket of bytes since document start, 0–12): 130 buckets. For each bucket keep an
  EMA (decay 0.99 per chosen byte in that bucket) of the expensive model's realised gain, (cheap bits − mixed bits) at chosen bytes. Choose t
  when its bucket's EMA is above the running 75th percentile of the EMAs of recent bytes' buckets (same estimator); a bucket never chosen
  has EMA +∞ until it has 8 samples (optimistic start).
- `progress`: as gain, but the bucket score is the EMA of the **decrease** of the expensive model's bits in that bucket between its
  successive visits (learning progress), optimistic start as above.
Constants fixed in advance; do not tune.

## Scoring
- **Clean test bits/byte:** after the stream, freeze both models and run the H35 clean test set with the same placement rule (no learning;
  rules' running statistics frozen too).
- **Life bits** on clean documents of the stream, and on junk documents, separately.
- **Junk share of expensive steps:** the fraction of chosen bytes that fall in junk documents, per junk kind.
- Achieved ρ per rule (must be 0.25 ± 0.02).
Seeds 0–5 (corpus seed = model seed).

## Your job (codex): implement + smoke only (no network)
`h36_curious_patching.py` with a `--seed --rule` kernel entry point; kernel folders `kaggle/h36/<rule>-s<seed>/` (ids
`ezzzzzx/h36-<rule>-s<seed>`; private; CPU; no internet; code dataset `ezzzzzx/pdp-code-h36`; enwik8 dataset as H33); a resumable
foreground dispatcher `kaggle/h36/dispatch.py` (loads `KAGGLE_API_TOKEN` from `.env`; keeps the account's running kernels ≤ 5; at most 2
H36 at once; one retry); and `kaggle/h36/summarize.py` that writes `results/h36/RESULTS.md` and `results.json`.

## Smoke test (VPS CPU, `torch.set_num_threads(1)`, on the stand-in stream)
1. Prequential: the first 8,192 bytes' bits are identical for streams of 8,192 and 16,384 bytes, for every rule.
2. With ρ forced to 1 (every byte chosen), the cheap+expensive mix runs and the GRU's bits fall over 64 KB of clean text.
3. Achieved ρ within 0.25 ± 0.02 for each rule over 256 KB.
4. Print the junk share of chosen bytes for each rule on the stand-in (no bar here, just finite and plausible).
5. Time 256 KB per rule; estimate a 400-document kernel's wall (must be ≤ 85 min; if not, use 300 documents for every rule and say so).
If a check fails, stop and report.

## Outputs (summarize.py; numbers only, no verdict)
**At the top, a scoring table:** per seed, entropy junk share ÷ fixed junk share, gain − entropy and progress − entropy on clean test bits, with
pass counts. Then: per rule and seed, clean test bits/byte, life bits on clean and junk, junk share of expensive steps (per kind), achieved ρ,
mean ± std (ddof = 0); smoke output; walls; kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
1. **Surprise spends compute on noise:** junk share of expensive steps under `entropy` ≥ 1.2 × under `fixed`, on at least 5 of 6 seeds.
2. **Placing compute where it helps beats placing it by surprise:** gain − entropy ≤ −0.03 bits/byte on the clean test set, on at least 5 of 6.
Report only: progress − entropy; fixed vs entropy on clean bits.
Kill rule: if bar 1 fails, the cheap model's entropy already discounts junk (the counts learn that J1 is uniform and J3/J4 are predictable), and
the noisy-TV argument against entropy patching does not hold for this cheap model.

## Budget and rules
- About 90 minutes of your time; smoke in the foreground.
- Files you may create: `h36_curious_patching.py`, `kaggle/h36/**`, `results/h36/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h35_*` (grok writes them), `h34a2_*`/`kaggle/h34a2/**` (your other job), `h33_bank_text.py`,
  `h26_cued.py`, any other code, `.env`.
- Gates and bars are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.
