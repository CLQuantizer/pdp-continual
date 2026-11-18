H34a3 implements detached local partial-sum losses, with H34a2 layer mechanics.
The four kernels cover casc_2/casc_4, seeds 0–1. casc_1 reuses sum_1 results.

Local foreground smoke (no network):

    .venv/bin/python h34a3_cascade.py --smoke
    .venv/bin/python kaggle/h34a3/dispatch.py --prepare-only
    .venv/bin/python kaggle/h34a3/summarize.py

The dispatcher without --prepare-only is for the orchestrator's network runner.
It loads the Kaggle token from .env without logging it, checks all account
kernels before each push (account cap 5; H34a3 cap 2), allows one retry, and
persists state/fingerprints. Dataset uploads are private; kernels use CPU and
have internet disabled. A failed smoke prevents launching. Ambiguous pushes
require remote reconciliation before retrying. No launches occurred locally.

RESULTS.md starts with full-run and tail scoring tables, and marks missing
scored kernels explicitly. Thresholds are unchanged and apply to both seeds
on the full 300,000 bytes. The tail table is diagnostic.
