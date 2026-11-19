# H34d offline handoff

Implements the compressor improvement experiment in `h34d_dfa.py`, importing
H34b unchanged. Fixed setup: 180,000 bytes of enwik8, tail [40,000,180,000),
N=128, four layers, chunk 128, seeds 0 and 1. The six arms/seeds use one
training loop. Readouts receive their own loss; only the recurrence gradients
receive the DFA addition (or replacement at layers 2–4 for `dfa_only_4`).

The orchestrator's unchanged bar is **tail dfa_4 − plain_4 ≤ −0.03 on both
seeds**. DFA-only and full-run gaps are report only. Missing runs remain NA.

Layer indexing in the feedback RNG is one-based: layer k=1..4 uses
`default_rng(7000 + 100*seed + k)`, matrix shape `(256,256)`, standard deviation
`1/16`. Alpha uses chunk Frobenius norms: the own mean-loss state gradient
versus `B_k @ (p_mix-y)` (unaveraged final error). No epsilon or tuned scale is
introduced. Alpha is zero when the own signal is zero, including the first
zero-readout chunk. Every per-chunk alpha and both norms are saved.

Local validation (foreground, no network):

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u h34d_dfa.py --smoke > results/h34d/smoke.log 2>&1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34d/dispatch.py --prepare-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34d/check_offline.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python kaggle/h34d/check_float32.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34d/summarize.py
```

The 512-byte autograd check unrolls the layer's recurrence with frozen
parameters in double precision, detached lower features, and a detached
combined-error DFA direction. It checks nu, theta, both input matrices and
the embedding, and separately checks own-only readout gradients.
`check_float32.py` additionally checks the full own + DFA update in production
float32 against autograd on 512 bytes. Prefix
checks end at chunk boundaries: parameters are updated after each scored
chunk, exactly as in H34b. Wall estimates extrapolate measured one-thread
8,192-byte runs to 180,000 bytes; every real run also enforces 5,100 seconds.

For the network-enabled runner, after reviewing smoke and preparation:

```sh
.venv/bin/python kaggle/h34d/dispatch.py --dispatch
```

This command is provided for the orchestrator; it is not part of local smoke.
It loads only `KAGGLE_API_TOKEN` from the existing `.env` without printing it,
uploads/versions private code dataset `ezzzzzx/pdp-code-h34d`, then dispatches
private CPU kernels with internet disabled. There are at most three active
H34d jobs and at most five account jobs (running, queued, or pending), checked
immediately before every push. Unknown or unreadable statuses refuse a push.
One retry per failed kernel; state and fingerprints persist in
`results/h34d/dispatch.json`. Interrupted ambiguous submissions stop for
inspection rather than risking duplicate work. Source/config changes after
first dispatch are rejected. Run the same command to resume.

The dispatcher lock serializes H34d dispatchers. Other experiment dispatchers
must respect the account cap too; Kaggle provides no atomic account-wide
reservation. The offline checker mocks pagination, caps, and resume paths;
it never reads credentials or invokes Kaggle.

Results are fetched to `results/h34d/kernels/<arm>-s<seed>/results.json`.
Summarization validates data/source/config fingerprints, score reconstruction
(including the 40–50 KB tail edge), norm logs and completed run sizes before
writing `results/h34d/RESULTS.md` with the scoring table first. No remote runs
are substituted with smoke numbers.
