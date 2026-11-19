# W9 handoff

Codex implements and runs the one-process smoke only. Grok runs the scored kernels.
No scoring lives have been run locally. Seeds 60–71, 5 arms, L=2048,
World5 period=128. All constants are fixed; no tuning.

From the repository root, using the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/w9/dispatch.py --check-only
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/w9/dispatch.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/w9/summarize.py
```

The dispatcher loads `.env` without printing it, creates or versions the private
`ezzzzzx/pdp-code-w9` code dataset, waits for readiness, submits private CPU
kernels with internet disabled, and fetches results validated against the staged source hashes. It persists jobs
before submission, allows one retry, limits W9 to 3 concurrent kernels and the
account to 5 active kernels, and locks against a second W9 dispatcher. Re-running
the same command resumes `results/w9/dispatch.json`. Do not change the staged
source while kernels are running. `--dataset-ready` skips uploading only when
the staged version is already available. `--assemble-only` assembles outputs.

To repeat local smoke and stage (one process):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python w9_day_night.py smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/w9/summarize.py --pending
```

Decision indices t are zero-based. Night is t mod 256 >=192. World swaps and
W5 checkpoints still run after decision t+1. Night corruption applies to every
wake glance, including corridor, noise, and dark rooms. Symbols are independently
replaced with probability .8, using seed+60000; uniform replacement may equal the
original symbol. The learner and both reservoirs receive observed symbols.
W5 checkpoint continuations remain clean.

Fixed random starts are uniform over all non-overlapping sets of 8-decision
visits on the integer decision line (seed+61000). This is not a fixed grid:
choose n=L/32 distinct integers from [0,L-7n), sort, then add 7*i to start i.
This is a bijection to all feasible ordered starts, giving precisely L/4 sleep.
Fixed night sleeps exactly t=192..255 within each cycle.

Sleep pauses and resumes the current room visit without travel, ending a replay
snapshot boundary as in M9. Room reward remains exam bits saved per wake
decision; age uses the decision clock including sleep. No sleep reward enters
the room bandit. Sleep uses M9's underlying Life.sleep_decision and live-state
checks, with one replay draw/update per decision. A pressure episode ends only
at an 8-decision visit boundary after S reaches zero (S is floored each step).
The end of the life can truncate its final visit, as in M9.

Pressure controllers receive no clock or night flag. Opportunity uses the
same frozen zero-state glance scorer used elsewhere in M7/M9, on identical
observed symbols/channel/previous-symbol on each side of the wake update.
It includes corridor glances. The EMA is initialized to the first gain; median
uses up to the last 256 wake EMAs, including the current value. Trace values
are recorded before each decision, after choosing its wake/sleep state.

`night_sleep_share` is sleep at night divided by all sleep. It is undefined
(`NA`/JSON null) for nosleep. `--pending` creates tables with `NA` for absent
scored results. Final assembly requires exactly 60 validated lives.

Pre-registered T14 bars, unchanged:

1. fixed_night - fixed_random <= -.10 on at least 9/12 seeds.
2. pressure_opp - fixed_night <= +.03 on the mean, and
   pressure_opp - fixed_random <= -.07 on at least 8/12 seeds.

Pressure without opportunity, sleep at night, and fixed_random - nosleep are
report-only. The summary writes numbers and counts without a verdict.
