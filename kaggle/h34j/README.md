H34j stages the first 1,000,000 bytes of enwik8 for four unchanged arms and seeds 0 and 1. It serves principle 2, improving the living compressor. The wrapper binds H34g's existing checkpointed loop to the 200,000-byte tail boundary; H34i and H34h supply the existing model constructors. All learning settings and chunk order are preserved. Reports use decimal 100 KB slices.

The orchestrator launches the foreground dispatcher from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34j/dispatch.py --dispatch
```

It loads `KAGGLE_API_TOKEN` from the environment or `.env` itself and redacts the token in CLI output. It creates/versions the private code dataset `ezzzzzx/pdp-code-h34j`; enwik8 is the existing `ezzzzzx/enwik8` dataset. Only enumerated sources and the successful smoke artifact are uploaded. Kernels are private CPU scripts, internet off, one PyTorch thread. IDs are `ezzzzzx/h34j-<arm-with-hyphens>-s<seed>-p<k>`.

Local smoke compares every arm and both seeds with the existing reference on 8,192 bytes, then compares an unsplit run with a persisted 4,096/4,096 split. Byte scores and the complete final checkpoint must match exactly, excluding wall times: parameters, recurrent states, traces, optimizer state, mixer, diagnostics, and RNG. No scored job is prepared without a successful smoke matching current sources.

The slowest measured rate for each arm is extrapolated to 1 MB. Parts target at most 4,800 seconds with a 25% runtime reserve. Intermediate boundaries are multiples of 128; the final short chunk is unchanged. The original loop enforces its 5,100-second training limit, and each kernel also checks its measured total wall against 5,100 seconds. A hardware slowdown that exceeds the reserve fails the part rather than emitting a valid score. `results/h34j/smoke.json` and `code-sha256.json` record the measured plan and boundaries.

Each continuation attaches its predecessor's kernel output and checks its identity, code/data fingerprints and checkpoint hash. Every part saves a checkpoint, including the final part. Fetched checkpoints are hashed, parent links and cumulative walls are checked, and full/tail/prefix means must agree with the 100 KB slices.

Dispatcher state is atomically persisted to `results/h34j/dispatch.json`; an exclusive process lock prevents two H34j dispatchers. Every push exhausts account listing pagination and checks active remote statuses, counting running/queued/pending jobs and locally submitted jobs. H34j is limited to four and the whole account to five. Other experiment dispatchers must obey the same account cap; the API offers no atomic account reservation. Explicit failed parts get one retry; independent lives finish even when another exhausts its retry. Rerun the same command after interruption. Unknown or ambiguous remote states stop submission for inspection. Source, orchestration or split-plan changes after dispatch starts are refused.

Offline preparation and checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python h34j_long.py --smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34j/dispatch.py --prepare-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34j/verify_offline.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34j/verify_template.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34j/verify_summary.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34j/summarize.py
```

`RESULTS.md` and `results.json` begin with tail and full gaps, per seed and mean, and counts against the unchanged bars (-0.10, -0.02, -0.01). The preregistered bars apply to the tail; full counts are descriptive. Missing results remain `NA` and unevaluated. Then come per-arm scores, slice bits and bits/byte, every kernel ID, segment/cumulative/kernel walls, and smoke measurements. No experiment verdict is written.

The interrupted smoke was completed with `kaggle/h34j/finish_smoke.py`, which checks the original source and stream hashes, retains the seven completed arm/seed pairs, and reuses the final persisted unsplit checkpoint. Its log is `results/h34j/finish-smoke.log`. The synthetic report checks validate gap arithmetic, missing-data counts, checkpoint hashes, parent links, score totals, and wall limits; fixtures are deleted and never enter the scored outputs.

Measured plan (both seeds use the same boundaries):

| Arm | Estimated full seconds | Parts per seed | Estimated seconds per part |
|---|---:|---:|---:|
| plain_1 | 2195.394 | 1 | 2195.394 |
| zgate_6 | 24653.742 | 7 | 3521.963 |
| dense_6 | 24946.502 | 7 | 3563.786 |
| dense_8 | 32903.273 | 9 | 3655.919 |

48 staged kernels, eight lives. The smoke has eight exact reference comparisons and eight exact 4096/4096 split checks. Staging is offline only; no scored kernels have been launched.
