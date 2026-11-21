H34e implements the registered self-distillation experiment. It serves the
compressor-improvement principle and scores observations in bits per byte.

Codex prepares and smokes; Grok runs the eight scored kernels. No kernels were
submitted during offline preparation. From the repository root, Grok runs the
dispatcher in the foreground:

```sh
.venv/bin/python -B kaggle/h34e/dispatch.py --dispatch
```

The dispatcher reads only KAGGLE_API_TOKEN from the environment or `.env`,
creates/versions the private code dataset, and submits private CPU kernels with
internet disabled. It allows at most two H34e jobs and five account jobs active
(including queued/pending remote jobs). It checks every account page, uses a
process lock, stores resumable state in `results/h34e/dispatch.json`, and permits
one retry per failed kernel. Unknown account state stops submission. Other
simultaneous dispatchers must also respect the account cap: Kaggle has no atomic
account-wide slot reservation. Rerun the same command to resume.

```sh
.venv/bin/python -B kaggle/h34e/summarize.py
```

The dispatcher calls the summarizer after all results are fetched. It can also
be run on incomplete results: missing pairs stay NA, their mean stays NA, and
they never count against the bar. `results/h34e/RESULTS.md` starts with the tail
`distill_4 - plain_4` gaps and the count at <= -0.03. The JSON includes all runs,
per-layer scores, mean mixer weights, walls, and smoke measurements. Baseline
full and tail means must reproduce H34b within 1e-6 for each seed. Fingerprints,
data checksum, dimensions, lambda masks, and slice totals are also validated.

The teacher is softmax of the inherited GeoMixer's weighted logit sum, detached
before any bank learns. All layers score the chunk before any update; banks
receive their own explicit logit error divided by chunk length. Mixer learning
remains its original true-label CE, independently of distillation. Layer inputs
and features remain detached. `distill_4_top` uses lambda 0.5 on layers 1-3 and
0 on layer 4. No lambda option is exposed to scored kernels.

The inherited GeoMixer weights are unconstrained, including at depth one. Thus
`distill_1` keeps a learned scalar temperature: its teacher is softmax(w*z),
which equals its own softmax(z) when w=1. It is not guaranteed to be an exact
self-target once the mixer learns. The smoke separately checks the zero term
when teacher and student distributions are equal. Also, with an exact self
target, lambda=0.5 halves the supervised error; it does not preserve the full
plain_1 update. No special-case gate freeze or rescaling is introduced.

Reproduce the offline checks with one CPU thread (configured by the script):

```sh
.venv/bin/python -B h34e_self_distill.py --smoke
.venv/bin/python -B kaggle/h34e/dispatch.py --prepare-only
.venv/bin/python -B kaggle/h34e/check_tools.py
```

The smoke checks both seeds for exact H34b and zero-lambda equality on 8192 bytes;
512-byte float64 logit gradients; unchanged prefixes with extended and modified
future streams for every arm; finite scores; and a per-kernel 85-minute estimate.
The prefix ends at the inherited 128-byte chunk boundary. The tool checks mock
all network and credential operations and use temporary local fixtures.
