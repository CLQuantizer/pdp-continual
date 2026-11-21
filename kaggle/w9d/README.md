# W9d handoff

Codex implements and runs one foreground smoke process. Grok runs the scored
kernels only after `results/w9d/smoke.json` has `ok: true` and the layout check
succeeds. Seeds 84–95, L=2048, World5 period 128, night corruption probability
.8 when `(t mod 256)>=192`, clean exams, prog_age_s room choice. Arms:
`fixed_random`, `fixed_night`, `pressure_eyelid` (W9c amendment 2),
`pressure_retina`. The first three delegate directly to W9c/W9b.

```bash
.venv/bin/python -B w9d_retina.py smoke
.venv/bin/python -B kaggle/w9d/dispatch.py --check-only
.venv/bin/python -B kaggle/w9d/summarize.py --pending
```

Smoke automatically stages only on success. It checks the fresh KT estimator
against its independent gamma-function code probability, then night-off
seed24 L256 nosleep/fixed20 compatibility against M9 and fixed_random against
W9 (T14 and every checkpoint, tolerance 1e-6). Seed84 pressure_retina runs all
2048 decisions, with raw r means over day, night, and daytime decisions in
`[redraw, redraw+32)` (the initial world at t=0 is not a redraw). It prints the
pre-decision S, r_bar, wake median, sleep and forced state for t=180–260, and
post-decision signal/controller records for every decision in the JSON output.
It prints day/night/forced sleep counts for each day. The gate is night sleep
share over days 3–8 >= .50. On failure it stops before finite checks or staging.

Retina r is computed by coding the current glance in order under an order-0
KT estimator, alpha=.5 over all 41 declared symbols. Counts start at zero on
every call. It receives no cortex, clock, room identity or previous glance.
The EMA is .9 old + .1 new across wake and sleep decisions, initialized with
the first r. Its median contains the latest 256 wake EMAs, including the
current EMA on wake. Poor means EMA > 1.5*median. S rises by 1 awake and falls
by 3 asleep, floored at zero. Sleep starts at S>=32 and poor, persists while
poor and S>0; S>=256 forces sleep until S<=128. The inherited eight-decision
sleep visits and boundary timing are unchanged. Sleep glances advance only
the external stream and signal; they are never learned, stored, scored in
T14 or credited as room progress. Replay follows the glance normally.

Every seed84 sleep glance is checked by the inherited learner/controller
checksum, excluding only the signal EMA, timers and external world. The
summarizer independently reconstructs all 2048 controller decisions. After
the gate, all four arms must be finite at L512. Kernel wall estimate is the
sum of L512 arm walls times 4 (length) times 2 (growth), limit 5100 seconds,
without credit for parallel kernel workers.

Grok runs from the repository root:

```bash
.venv/bin/python -B kaggle/w9d/dispatch.py
.venv/bin/python -B kaggle/w9d/summarize.py
```

The resumable dispatcher creates/versions the private code dataset
`ezzzzzx/pdp-code-w9d`, submits private CPU kernels `ezzzzzx/w9d-s84` through
`ezzzzzx/w9d-s95`, internet off, four single-threaded arm workers per kernel.
It enforces <=3 W9d kernels and <=5 active kernels across the account, saves
state before submission, allows one retry, validates output hashes, and uses
a lock against duplicate W9d dispatchers. Re-run to resume. Credentials are
loaded without printing them. `--dataset-ready` skips uploading only when
this staged dataset version is ready; `--assemble-only` assembles fetched
outputs. `--check-only` is local and reads no credentials.

Pre-registered bars, T14, 12 seeds (unchanged):

1. pressure_retina mean night_sleep_share >= .60.
2. pressure_retina - fixed_random <= -.07 on >=8/12 seeds.
3. pressure_retina - fixed_night <= +.05 on the mean.

pressure_eyelid is report-only. `results/w9d/RESULTS.md` starts with per-seed
retina gaps, counts and arm sleep/night-sleep shares, then T14, smoke and
walls. `results.json` contains validated runs, thresholds, smoke and missing
lives. `--pending` retains NA/null for unavailable scored data; final assembly
requires all 48 validated lives. No local smoke life is a scored result.

Seed84 smoke completed: days 3–8 night sleep 230/391 =
0.588235294. Raw mean r: day 3.328899861,
night 5.470240752, day-after-redraw
3.323510141. All 526 eyelid checksums match;
controller reconstruction matches 2048 decisions. All four arms finite at
L512. Conservative wall estimate 2569.753s (<5100s).
Layout check: 12 private CPU kernels, 18 matching code hashes. Outputs are
pending (0/48 scored lives). Full smoke-life output validation passed and an
altered retina signal was rejected. Ready for Grok's foreground dispatcher:

```bash
.venv/bin/python -B kaggle/w9d/dispatch.py
```
