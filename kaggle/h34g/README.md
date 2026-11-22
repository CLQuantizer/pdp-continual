H34g CPU kernels; Grok runs this dispatcher in the foreground:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34g/dispatch.py --prepare-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34g/dispatch.py --dispatch
```

Use Grok's `--background-wait-timeout 21600` when invoking Grok. No kernel dispatch was performed by Codex. The dispatcher checks the account before every push (account ≤5, H34g ≤2, including externally launched H34g); W9d may run alongside. Coordinate other dispatchers against the same account limit. Private CPU, internet off, seeds 0 and 1, 180,000 bytes. No GPU quota used.

Smoke selects splitting if the measured zgate_8 estimate exceeds 5,100 seconds. Each split seed uses its base kernel plus `-part2`, attached via `kernel_sources`; the continuation waits for the fetched and validated first output. Part 1 checkpoints at 90,000 bytes seen, with all updates through byte 89,984 and 16 buffered bytes. Part 2 completes that original 128-byte chunk before updating. This keeps the unsplit chunk schedule exactly. The checkpoint includes parameters, recurrent state, eligibility traces, both sets of Adam states, mixer, scores, diagnostics, and RNG state. Whole-stream input and source hashes are checked on resume. No partial-chunk update is introduced.

The dispatcher resumes from `results/h34g/dispatch.json`, uses a process lock, retries a failed job once, rejects changed model or orchestration sources after dispatch starts, and verifies kernel IDs, source fingerprints, data hashes, checkpoint hashes and reference agreement before accepting outputs. It invokes the offline summarizer when all jobs finish:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34g/summarize.py
```

Missing scored runs are NA, never zero or passes. Plain_1 and zgate_4 are read from the completed H34f results; zgate_6 is rerun and checked to 1e-6. Bars are both seeds, tail: zgate_8 − zgate_6 ≤ −0.02; dense_6 − zgate_6 ≤ −0.03. Full-stream gaps and the tail depth curve are report-only. No verdict is generated.
