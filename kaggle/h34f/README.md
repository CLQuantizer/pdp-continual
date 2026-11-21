H34f implements the registered zero-init gate experiment to improve the living
compressor. Every layer retains H34b's own-loss local trace learning. Only the
initial GeoMixer weights change: plain arms start at all ones; zgate arms at
[1, 0, ...]. Gates remain unconstrained and learn with the inherited Adam 0.003.
Layer Adam remains 0.001, clipping 1.0, chunks 128, float32, N=128, rank=64,
embedding=32, detached per-step normalized features. Seeds are 0 and 1.

Codex prepares and smokes locally. Grok runs all ten scored kernels. No kernel
is submitted by preparation or smoke. Run from the repository root:

```sh
.venv/bin/python -B h34f_zero_gate.py --smoke
.venv/bin/python -B kaggle/h34f/dispatch.py --prepare-only
.venv/bin/python -B kaggle/h34f/check_tools.py
.venv/bin/python -B kaggle/h34f/dispatch.py --dispatch
```

Grok should run the dispatcher in the foreground with its session configured
for `--background-wait-timeout 21600`. Rerun the same dispatch command to resume.
The smoke fingerprint must match the source. Kernels attach the private code
dataset ezzzzzx/pdp-code-h34f and ezzzzzx/enwik8, run on one CPU thread, and have
internet and GPU disabled. All arms score exactly the first 180,000 bytes;
tail is bytes 40,000–180,000. No parameters or thresholds are tuned.

The dispatcher reads KAGGLE_API_TOKEN from the environment or .env without
printing it. It permits at most two H34f jobs and five account jobs active,
including remote queued/pending jobs, checks all account pages before each push,
uses a process lock, and persists state in results/h34f/dispatch.json. Unknown
account state stops submission. Other dispatchers must respect the same account
cap; Kaggle has no atomic account-wide slot reservation. Failed kernels get one
retry. Full, tail, and first-40-KB baseline scores are checked against H34b with
a 1e-6 tolerance in every plain kernel and again when summarizing.

```sh
.venv/bin/python -B kaggle/h34f/summarize.py
```

The dispatcher summarizes automatically after fetching all ten results. Before
completion, missing results are NA; a missing pair never counts as a passing
seed and its two-seed mean stays NA. RESULTS.md starts with six comparisons:
full and tail for zgate_4 - plain_4, zgate_6 - plain_6, zgate_6 - plain_1.
Only the three registered bars are scored: full zgate_4 - plain_4 <= -0.10,
tail zgate_4 - plain_4 <= +0.01, full zgate_6 - plain_1 <= -0.05, both seeds.
Other comparisons are report-only. There is no verdict in the output.

Mean mixer weights per 20,000 bytes are weighted by the number of predictions
using each gate vector, measured before the chunk update. Chunks straddling a
window edge contribute to both windows. Scoring and prefix checks retain the
H34b chunk protocol; the 1,024-byte prefix is chunk aligned. Source hashes,
data checksum, kernel IDs, slice reconstruction, finite diagnostics, initial
gates, and wall limits are validated before accepting scored results.

The smoke checks exact first-chunk logits/probabilities/losses against plain_1;
byte-for-byte H34b loss equality for plain_4 and plain_1 on 8,192 bytes for both
seeds; unchanged 1,024-byte prefixes under extension and changed future bytes
for zgate_4 on both seeds; all-arm finite scores; and the <=85 minute kernel
estimate. Tool checks use synthetic result fixtures and mocked network calls.
