H34b implements the plain own-loss stack from H34a2. It serves the improve-the-compressor principle, scored prequentially on observed bytes.

Local smoke (one thread, foreground):

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u h34b_plain_depth.py --smoke
```

Prepare the private CPU kernel folders and code dataset without network:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34b/dispatch.py --prepare-only
.venv/bin/python kaggle/h34b/summarize.py
```

A network-enabled runner can subsequently dispatch in the foreground:

```sh
.venv/bin/python -u kaggle/h34b/dispatch.py --dispatch
```

The dispatcher loads KAGGLE_API_TOKEN from the environment or .env without printing it. It publishes only the enumerated source files to the private ezzzzzx/pdp-code-h34b dataset. Each of 14 private CPU kernels also attaches ezzzzzx/enwik8, has internet disabled, and uses one thread. No dispatch happens by default. Source hashes must match the successful smoke. If smoke selects 200,000 bytes, preparation applies that length to every arm.

Resume using the same --dispatch command. State and lock are in results/h34b. Each push consumes one of two attempts; known failed runs get one retry. An interrupted/ambiguous push remains reserved and needs remote status/fingerprint reconciliation; it is never blindly repeated. Immediately before every submission, all listed account kernels are checked, with a maximum of five active account kernels and three active H34b kernels (including other launchers). Unknown/unreadable status refuses submission. Concurrent dispatchers for other experiments must also respect the account limit; Kaggle offers no atomic account-wide reservation.

Summarize after fetching, or inspect pending tables locally:

```sh
.venv/bin/python kaggle/h34b/summarize.py
```

RESULTS.md begins with the three pre-registered tail scoring rows. Missing seeds are NA, with evaluated counts; they never count as passing. Noskip and full-run gaps are report-only. Fetched results must match kernel identity, source fingerprint, input checksum, byte count, and 25 KB slices.

The smoke runner fails fast. If plain_6 fails its first-2,048-byte stability check after producing the 16,384-byte timing record, finish the independent wide_6 and common-budget checks without changing the model:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u kaggle/h34b/finish_smoke.py
```

This follow-up requires unchanged source hashes and the expected failed-smoke evidence. It preserves every failure, records the 200 KB fallback if triggered, and exits nonzero if any check fails. Failed smoke keeps dispatch disabled. Then run --prepare-only and summarize.py to refresh the folders and report at the selected length.
