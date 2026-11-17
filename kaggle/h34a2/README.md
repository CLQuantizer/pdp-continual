# H34a2 execution

Implementation and foreground CPU smoke only were requested from Codex. The orchestrator runs the dispatcher after reviewing `results/h34a2/smoke.json` (`ok: true`).

```sh
.venv/bin/python h34a2_logit_sum.py --smoke
.venv/bin/python kaggle/h34a2/dispatch.py
```

The dispatcher reads `KAGGLE_API_TOKEN` from the environment or `.env` itself. It creates the private code dataset `ezzzzzx/pdp-code-h34a2`, waits for readiness, runs at most three of its kernels concurrently, fetches results, and invokes the summarizer after all 20 lives finish. No GPU or internet is enabled in the kernels. All lives use exactly the first 300,000 bytes of the checked enwik8 stream.

Run the same foreground command to resume. State is in `results/h34a2/dispatch.json`; a file lock prevents duplicate dispatchers. Completed jobs are retained. A failed kernel is retried at most once; failed outputs are kept under `results/h34a2/failed/`. A status or fetch error stops the process with state preserved. Source hashes prevent changing the code bundle midway through a dispatch.

The dispatcher journals `submitting` before each push. If interrupted during a push, acceptance by Kaggle is ambiguous: it stops on resume rather than potentially duplicating a life. Inspect that kernel's remote version/status, then set that job's state to `active` if accepted, or `pending` if not accepted; its attempt count already includes the uncertain push. Restart the dispatcher. Do not reset attempts or a terminal failed state to bypass the one-retry limit.

`--prepare-only` regenerates the kernel folders, explicit code bundle, and SHA256 manifest locally, without loading credentials or contacting Kaggle. The bundle contains only the ten Python dependencies listed in `dispatch.py` and dataset metadata.

```sh
.venv/bin/python kaggle/h34a2/summarize.py
.venv/bin/python kaggle/h34a/summarize.py
```

Each summarizer refuses incomplete kernel sets. H34a's summarizer is for the orchestrator after fetching its final kernels; it has not been run by this worker. Both write numbers and diagnostics without scoring bars or producing verdicts.
