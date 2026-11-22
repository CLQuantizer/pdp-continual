H34h CPU kernels; Grok runs the dispatcher in the foreground:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34h/dispatch.py --prepare-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34h/dispatch.py --dispatch
```

Use Grok's `--background-wait-timeout 21600`. Codex prepares and smokes offline; Grok dispatches. Private CPU kernels, internet off, code dataset `ezzzzzx/pdp-code-h34h`, enwik8 dataset `ezzzzzx/enwik8`, seeds 0 and 1, 180,000 bytes. H34h active ≤3 (including externally launched H34h); account active ≤5, checked immediately before each push. Other dispatchers must respect the same account cap. No GPU quota used.

Arms: dense_8, dense_4, dense_6_ones, dense_6. Only dense_6_ones uses an all-ones initial mixer; the other arms use [1,0,…,0]. H34g's training and checkpoint loop is imported with isolated function globals; the original module is never mutated. All own-loss learning, dense normalization, chunk boundaries, optimizer order, and data settings are inherited exactly.

Smoke selects dense_8 splitting if its slowest measured 8,192-byte one-thread extrapolation exceeds 5,100 seconds. Split seeds use a base kernel and `-part2` with the parent in `kernel_sources`. At 90,000 bytes seen, 89,984 bytes are committed and 16 buffered; resume completes the original 128-byte chunk. Checkpoints preserve models, recurrent states, traces, Adam states, mixer, diagnostics, scores, RNG, input hash and source hashes. Smoke tests aligned and unaligned resume against the unsplit 8,192-byte stream, both seeds.

The dispatcher resumes from `results/h34h/dispatch.json`, locks against duplicate dispatchers, retries failed jobs once, and refuses source/config changes after dispatch starts. Fetched outputs must match source/data fingerprints and checkpoint lineage. Completed dense_6 runs must match H34g within 1e-6. When all jobs finish it runs:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34h/summarize.py
```

The top ablation table contains tail/full per seed and means for depths 1,4,6,8 and plain/zgate/dense, plus dense_6_ones. Missing plain_8 remains NA; it is not an authorized run. At depth one all three inputs and initial gates are identical, so all three cells reuse plain_1. Plain_1/4/6 and zgate_4/6 reuse H34f's completed results; zgate_8 reuses H34g. Dense_6 is filled only from the H34h rerun. Baseline hashes are recorded. Missing scores never count as passes.

Bars (both seeds, tail; unchanged): dense_8 − dense_6 ≤ −0.02; dense_6 − zgate_8 ≤ −0.03. Full gaps, dense_6_ones − dense_6 and dense_4 − zgate_4 are report only. The numeric report also includes own tail bits, mixer weights per 20 KB, reference gaps, smoke measurements and walls; no verdict.

Offline checks:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34h/verify_offline.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34h/verify_results.py
```

They simulate dispatch and resume, caps with 1/4 unrelated active jobs, account pagination, duplicate refusal, report missingness, bar signs/counts, source rejection, and execute the real split kernel template against a mock local mount. Synthetic scores exist only in temporary directories.
