# H35: a curious data diet. Can compression progress do pretraining data cleaning by itself? (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `h33_bank_text.py` (the counts_mix6 arm: `OrderCount`, `GeoMixer`) and `h26_cued.py` (enwik8 loading).

## Context
LLM pretraining cleans data by hand: it drops junk (unlearnable: random strings, corrupted text) and repeats (already learned:
duplicates, boilerplate). Compression progress rejects exactly those two: noise gives no progress, the already-learned gives none.
Choosing by high loss picks junk (our gmc_prop spent 16–22% on noise). H35 asks whether a learner that chooses what to read next by
its own learning progress approaches a hand filter that knows which documents are junk. Principles served: 3 and 4 (choose the data
that yields compression progress), on real text.

## Corpus (build once, deterministic in the corpus seed c = run seed; write `h35_corpus.py` with a function `build(seed)` returning the
pool, labels and the test set, and save to `data/h35/` for reuse by H36)
Source: enwik8 (`h26_cued.load_enwik8`, E10M sha256 check as `h33_bank_text` does). Documents are 4,096 bytes.
- **Clean pool:** 1,600 documents, bytes [0, 6,553,600) cut into consecutive 4,096-byte documents.
- **Junk pool:** 1,600 documents, 320 of each kind (rng `default_rng(c + 500)`):
  - `J1_random`: uniform random bytes 0–255.
  - `J2_corrupt`: a clean 4,096-byte block from bytes [6,553,600, 8,900,000) (block start uniform, aligned to 4,096) with 30% of byte positions
    replaced by uniform random bytes.
  - `J3_dup`: 16 documents chosen uniformly from the clean pool, each copied 20 times (exact duplicates of pool documents).
  - `J4_boiler`: a 300-byte template (bytes [8,900,000, 8,900,300) of enwik8) repeated to fill 4,096 bytes; in each 300-byte repeat, bytes
    100–105 are replaced by a random 6-digit ASCII number.
  - `J5_shuffle`: a clean block from [6,553,600, 8,900,000) split on spaces, words shuffled uniformly, rejoined with spaces, cut or padded
    (with spaces) to 4,096 bytes.
- **Pool** = clean + junk = 3,200 documents, shuffled (rng `default_rng(c + 600)`). Labels are hidden from every policy except the oracle.
- **Reference set** (for `rho` only): bytes [8,900,300, 9,000,300).
- **Clean test set:** bytes [9,000,300, 9,262,444) (64 documents of 4,096 bytes), never read by any policy.

## Learner
counts_mix6 exactly as H33's `counts_mix6` arm (orders 1–6 in one count store, smoothing 0.05, GeoMixer with H32b's weight sets, Adam
0.003, one step per 128-byte chunk, counts of a chunk added after the mixer has scored that chunk and stepped). Reading a document = one
prequential pass over its 4,096 bytes (score, then learn), state continuing across documents. Order-k contexts do not cross document
starts (reset the context window at each document start).

## Policies (each picks the next unread document; budget B = 800 documents read, a quarter of the pool; rng `default_rng(1_000 + c)`)
- `rand`: uniform over unread documents.
- `oracle`: uniform over unread **clean** documents (the hand filter).
- `high_loss`: draw K = 32 unread candidates uniformly; for each, the current model's bits/byte on its first 512 bytes, frozen (a peek: no
  learning, no count updates); read the highest.
- `rho`: as high_loss, but the score is (current bits/byte on the peek) − (bits/byte on the peek of a reference counts_mix6 trained once on
  the reference set and then frozen). Read the highest (RHO-LOSS, Mindermann et al. 2022).
- `prog_peek` (curiosity): K = 32 candidates as above. For each, a trial update on **bytes 0–255** of the candidate, applied to a temporary
  overlay of the count store (no copy of the full store; the mixer is not updated in the trial), then the score is the bits saved by that
  trial update on (a) **bytes 256–511 of the same candidate** plus (b) a **history reservoir**: 16 pieces of 256 bytes from documents already
  read (reservoir sampling, rng `default_rng(2_000 + c)`), bits before minus after, divided by 512 + 4,096 bytes. Read the highest. Before
  16 documents have been read, (b) uses whatever is in the reservoir.
- K, peek lengths and the reservoir size are fixed in advance; do not tune.

## Seeds, scoring
Run seeds c ∈ {0, …, 5}. At every 100 documents read (8 checkpoints), score the frozen model on the clean test set (bits/byte; no learning
during the test). **T_clean** = mean over the 8 checkpoints; also report the final checkpoint (B = 800).

## Kernels
One private CPU kernel per seed, `ezzzzzx/h35-s<c>`, `"is_private": true`, `"enable_gpu": false`, `"enable_internet": false`; pool of up
to 4 policies at once inside. Code as private dataset `ezzzzzx/pdp-code-h35`; enwik8 from the Kaggle dataset H33 used. At most 90 minutes
per kernel (split policies across two kernels per seed if timing needs it). **Kaggle capacity:** other runs use Kaggle today; before each
push, check `kaggle kernels list --mine` and keep the total of running kernels on the account ≤ 5 (run at most 1 H35 kernel at once while 4
others are running). Run the dispatcher in the foreground until every kernel finishes.

## Smoke test (VPS CPU, one process, before any kernel)
1. Corpus: print counts per kind, 80 bytes of one document of each kind, and that duplicates are byte-identical to their originals.
2. Learner: counts_mix6 bits on the first 20,000 bytes of enwik8 match H33's counts_mix6 code to 1e-6 (with context resets disabled).
3. Prequential: a document's bits are the same whether or not later documents exist.
4. `prog_peek`'s overlay: the full count store is unchanged after scoring a candidate (checksum before/after); on one candidate, the overlay
   score equals the score from a real deep-copied store to 1e-9.
5. On seed 0 with B = 40: every policy runs; print the kinds each policy read. Time B = 100 for `prog_peek` and estimate kernel walls.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/h35/RESULTS.md` and `results/h35/results.json`, numbers only, no verdict. **At the top: a scoring table** with, per seed, the
quantities each bar needs (oracle − rand, prog_peek − rand, prog_peek − high_loss, on T_clean) and the pass counts. Then: T_clean and the final
checkpoint per policy and seed (mean ± std, ddof = 0), per checkpoint; the share of read documents per kind per policy; the smoke output;
walls; kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
**Precondition (cleaning matters):** oracle − rand ≤ −0.10 bits/byte on T_clean, on at least 5 of 6 seeds. If not met, the bars are void
("junk too cheap for this learner").
1. **Curiosity cleans:** prog_peek closes at least half of the oracle's gain: (prog_peek − rand) ≤ 0.5 · (oracle − rand) on T_clean, on at
   least 5 of 6 seeds.
2. **Progress, not loss:** prog_peek − high_loss ≤ −0.05 on T_clean, on at least 5 of 6 seeds.
Report only: rho; junk shares; the final checkpoint.
Kill rule: if the precondition holds and bar 1 fails, a peek-based progress estimate cannot tell learnable from junk for a count learner;
record which junk kinds it read.

## Budget and rules
- CPU only. At most 90 minutes per kernel. A failed kernel may be rerun once. Run everything in the foreground.
- Files you may create: `h35_corpus.py`, `h35_data_diet.py`, `kaggle/h35/**`, `results/h35/**`, `data/h35/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h33_bank_text.py`, `h26_cued.py`, `h32b_mixer.py`, any other experiment code, other `results/`
  or `kaggle/` folders, `.env`. Codex writes `h36_*`, `kaggle/h36/**`, `results/h36/**` and `h34a2_*` today: do not edit them.
- Gates and bars are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.
