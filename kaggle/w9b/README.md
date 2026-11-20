# W9b handoff

Codex implements and runs one-process smoke; Grok runs all scored kernels.
Seeds 72–83; five arms; World5 period 128; L=2048. No scored lives run locally.
No constants or bars tuned. The code imports W9's world, random schedule, and
pressure update; W9 and all underlying experiment sources remain unchanged.

From the repository root, using the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/w9b/dispatch.py --check-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/w9b/dispatch.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/w9b/summarize.py
```

The dispatcher loads `.env` without printing credentials, creates or versions
the private `ezzzzzx/pdp-code-w9b` code dataset, waits for readiness, submits
12 private CPU kernels `ezzzzzx/w9b-s72` through `ezzzzzx/w9b-s83`, with internet
and GPU disabled, and fetches source-hash-validated outputs. It persists state
before submission, allows one retry, limits W9b to 3 concurrent kernels and
the account to 5 active kernels, and locks against a second W9b dispatcher.
Run the same command to resume `results/w9b/dispatch.json`. Do not change
staged source after upload. `--dataset-ready` skips uploading only when the
staged version is already ready. `--assemble-only` assembles fetched outputs.
Each kernel runs the five arms in a spawn pool of at most four workers.

Repeat local smoke and stage in one process:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python w9b_poor_input.py smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/w9b/summarize.py --pending
```

Smoke stops without staging if the seed-72, first-1024-decision night mean of
wake EMA bits/symbol is not above its day mean, or the conservative five-arm
kernel wall estimate exceeds 5100 seconds. Both actual means are printed for
review. All five arms are checked at L=512. The wall bound scales the sum of
the five L512 walls by 4 for L2048 and 2 for reservoir growth; kernel parallelism
is not credited in that bound.

Baseline detail: W9's original M9 check used `nosleep` and `fixed20_compat`,
with M9's fixed 20% sleep schedule. W9's `fixed_random` instead uses a uniform
random 25% schedule and can interrupt a room visit, so it cannot equal M9's
unmodified fixed20 run. Smoke retains both exact M9 checks at seed 24, L=256,
night off, and additionally checks `fixed_random` against the original W9
with night off (T14 and every checkpoint, tolerance 1e-6).

Decision indices are zero-based. Night is t mod 256 >=192; wake observations
have independent uniform-symbol replacement probability .8 (seed+60000).
Exams and T14 continuations remain clean. Random sleep is W9's uniform
non-overlapping eight-decision schedule (seed+61000), with sleep share .25.
Room choice remains prog_age_s; sleep pauses and resumes its room plan.

`pressure_bits` uses the frozen zero-state scorer's total glance bits before
learning, divided by the number of observed symbols. The first value initializes
the EMA; subsequent wake values use .9 old + .1 new. The median includes the
current EMA and up to 255 preceding wake EMAs. The controller receives only
this signal: S increments by 1 on wake and drains by 4 on sleep, floored at zero.
It starts an episode when (S>=32 and b>median) or S>=128; the episode ends at
S=0 rounded up to an eight-decision visit boundary. The end of life may truncate
the final visit. No clock or night flag enters the controller. Other arms keep
W9 behavior. Saved wake diagnostics contain the raw bits/symbol, EMA, and median;
the summarizer reconstructs these plus pressure and every sleep decision.

Pre-registered T14 bars, unchanged:

1. fixed_night - fixed_random <= -.10 on at least 8/12 seeds.
2. pressure_bits night_sleep_share >= .60, mean over seeds.
3. pressure_bits - fixed_random <= -.07 on at least 8/12 seeds, and
   pressure_bits - fixed_night <= +.05 on the mean.

Pressure and pressure_opp are report-only. `night_sleep_share` is night sleep
count divided by all sleep; `sleep_share` is all sleep divided by L.
The summarizer writes numeric tables and counts without a verdict. `--pending`
writes both output files with NA/null for missing scored results and a complete
missing-life list. Final assembly requires all 60 validated lives. No absent
run contributes to a threshold count.

Completed local smoke (one process, 848.820 s): all five arms finite at L512;
night-off baseline T14 and checkpoint differences all 0.0. Seed72 L1024 wake
EMA means: day 3.277640111 (587 wake decisions), night 7.349776615 (189),
night minus day 4.072136504 bits/symbol. Conservative kernel wall estimate:
3363.800 s = 56.063 min, limit 5100 s. All 12 staged kernel folders and all
16 code hashes passed `dispatch.py --check-only`. Scored lives remain 0/60.
