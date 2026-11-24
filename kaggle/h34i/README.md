H34i is implemented and staged offline. The orchestrator launches the foreground dispatcher:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34i/dispatch.py --dispatch
```

The dispatcher loads `KAGGLE_API_TOKEN` from the environment or `.env` itself, without printing it. It creates or versions the private datasets `ezzzzzx/pdp-code-h34i` and `ezzzzzx/pdp-data-h34i`, waits for them to become ready, runs all 30 lives, fetches outputs, and calls the summarizer. Each kernel is private, CPU only, internet off, and runs PyTorch with one thread. No shell setup or token export is needed.

The exact streams live in `data/h34i/{c4,bible,mnist}.bin`, each 180,000 bytes, with original-source and prefix SHA256 hashes in `manifest.json`. C4's existing manifest is verified before taking the prefix. Bible includes its Gutenberg header. MNIST's IDX header is checked and skipped; pixels retain image, row and column order. Data preparation uses only local files. It is repeated and compared with the smoke manifest before upload.

Arms: `plain_1`, `plain_6` (ones), `zgate_6` ([1,0,…]), `dense_6` (zero gate, mean of individually normalized earlier features), and `dense_6_ones` (dense, ones). Seeds 0 and 1. H34g's exact chunk training/checkpoint loop is imported with isolated globals; the original H34b/f/g/h files remain unchanged. N=128, rank=64, embedding=32, trace learning, own losses, per-step normalization, float32, Adam 0.001, clip 1.0 and chunk=128 are unchanged.

Kernel IDs: `ezzzzzx/h34i-<stream>-<arm-with-hyphens>-s<seed>`, with `-part2` for a continuation. The smoke's slowest measured per-arm one-thread 8,192-byte rate is extrapolated to 180,000 bytes. An arm estimated above 5,100 seconds is split for every stream and seed. Split kernels stop at 90,000 bytes seen (89,984 committed, 16 pending), saving models, states, traces, Adam, mixer, diagnostics, scores and RNG. Part 2 attaches its parent kernel and verifies the checkpoint, code and input hashes. It completes the unchanged original 128-byte chunk. Both aligned and unaligned checkpoint boundaries are smoke tested for all five arms and both seeds.

The dispatcher persists `results/h34i/dispatch.json`, locks against concurrent H34i dispatchers, counts all running/queued/pending account kernels immediately before every push, and limits both H34i and the account to five active kernels. External H34i kernels count toward the cap. Listing pagination is exhausted, and unreadable or unknown statuses stop submission. Other experiment dispatchers must respect the same account cap; Kaggle has no atomic account reservation. Explicit kernel errors receive one retry. A failed second attempt stops dispatch. Restart the same command to resume; completed jobs are not pushed again. Source, orchestration, data or split-plan changes after dispatch starts are refused.

Preparation and checks, without network or credentials:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python h34i_generality.py --smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34i/preview_data.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34i/dispatch.py --prepare-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34i/verify_offline.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34i/verify_template.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34i/summarize.py
```

Run the numerical smoke and template checks sequentially. The smoke reproduces H34b/f/g/h byte scores on 8,192 enwik8 bytes for both seeds, prints each new stream's hash/count/preview and plain_1 score, checks prefix causality for all arms on C4, and tests exact checkpoint resume. Offline orchestration checks simulate dataset creation, active caps with unrelated jobs, dependencies, a retry, terminal failure and dispatcher restarts. Numeric fixtures remain in temporary directories. Real template checks exercise local mock mounts, all three loaders, split resume and wrong-data/parent rejection.

The summarizer writes `results/h34i/RESULTS.md` and `results.json`. The first three tables show each stream's per-seed gaps and pass counts; the fourth gives per-trick counts across streams and the enwik8 reference. Then come full/tail/first-40-KB bits for all 30 lives alongside checked enwik8 reference columns, the report-only dense gate gaps, wall times, kernel IDs, stream hashes and smoke measurements. Missing runs remain `NA`, count as unevaluated, and cannot pass. All fetched scores require matching source/data fingerprints and coherent slice totals; split results require the exact parent checkpoint hash and wall accounting. The existing H34b/f/g/h baseline hashes are recorded.

Bars, unchanged, on every new stream and both seeds: Own: tail plain_6 − plain_1 ≤ −0.05; Wait: full zgate_6 − plain_6 ≤ −0.10; See: tail dense_6 − zgate_6 ≤ −0.02. `dense_6_ones − dense_6` is report only, tail and full. The report contains counts and numbers, without a verdict.
