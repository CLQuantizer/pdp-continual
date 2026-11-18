H35b2 runs five arms on each seed 6–11, one private offline CPU kernel per seed.
No scored kernels were submitted by Codex. The code dataset contains only the
four Python files listed in `code-sha256.json` and dataset metadata.

Local smoke (one CPU thread, real corpus):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u h35b2_snapshot.py --smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h35b2/dispatch.py --prepare-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h35b2/summarize.py
```

Network-enabled runner, foreground (loads KAGGLE_API_TOKEN from .env without
printing it):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u kaggle/h35b2/dispatch.py
```

Re-run the same command to resume. It enforces account active ≤5 and H35b2
active ≤2, counting queued/pending jobs and kernels started elsewhere, with
one retry. It uses the existing `results/h36/account-push.lock` read-only
for coordination with H36/H35b; it fails closed if that lock is absent.
Other account dispatchers must use that same lock for atomic slot reservation.
Status cache and all other output belong to results/h35b2. Interrupted pushes
with ambiguous results stop for inspection; no blind duplicate submission.
Code/configuration changes after dispatch begins are rejected. Kernel wrappers
verify the staged source hashes before importing experiment code.

Snapshots use exact sliding lag in documents: before reading document t+1,
score under state max(0,t-k) minus state t. Retained copies have gradients
disabled, independent parameter storage, and no optimizer. At t=0 all scores
are exactly zero; numpy.argmax takes the first candidate (same tie rule as
H35b). Labels only reach the oracle and post-selection kind accounting.

Selection minimizes mean T_clean on seeds 6–8 (exact ties use k=5). No chosen
lag is reported until both candidates have all three selection results. Only
seeds 9–11 of that chosen lag contribute to threshold counts. Required counts
remain 2/3 for both bars. Smoke results are separate from scored results;
missing scored values remain blank.

Timing estimates extrapolate measured B=8 selection/training walls to B=200,
with eight measured clean-test costs and one reference setup. They do not
assume a Kaggle speedup and do not change any experimental constants.
