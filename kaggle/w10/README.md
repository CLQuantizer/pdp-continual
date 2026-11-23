# W10: W9d sleep on World6

Codex implements and smokes offline; Grok runs the scored Kaggle kernels.
Principles 2 and 4: replay improves the compressor; progress chooses rooms.

- Seeds 120–131; four arms; L=4096; birth period 128 (W6's primary open condition).
- World6's construction, births, deaths and native noise streams are unchanged.
- W9's observe method is imported directly: night at t%256>=192, corruption .8,
  observation RNG seed+60000. W6 continuation exams remain clean.
- Room selection imports M10/W6's prog_age_s selector (tau256, epsilon.1, beta.3).
- pressure_retina imports W9d's retina, controller, eyelid and checksum directly.
  All homeostat constants, history and eight-decision sleep visits are unchanged.
- Sleep calls the inherited Life.sleep_decision method used by W9. Sleep pauses
  the current room plan; births and checkpoints continue on the decision clock.
  Only pressure_retina takes eyelid glances, exactly as W9d's baseline comparison.
- T_open is W6/M10's eight checkpoint scores averaged over the life, EVAL=2048
  per currently learnable slot, zero state and original slot channel.
- night_sleep_share = sleep decisions at night / all sleep decisions.
  day_noise_sleep_share = sleep decisions by day in a currently N room / all
  sleep decisions by day. Kind is recorded before that decision's birth.
  Undefined shares are null/NA, including nosleep's night sleep share.

Offline smoke, one process:

```
.venv/bin/python -B w10_open_sleep.py smoke
```

The baseline disables only sleep_decisions in memory in M10's delegated W6
runner. Both lives run the full 4096 decisions without nights or sleep. Compare
all checkpoints and prequential bits exactly, plus a checksum of the final
learner, stores, RNGs, score caches and live state. Source files are unchanged.
The full-life pressure run checks every eyelid, reconstructs controller
transitions independently, and prints all 16 day/night/forced counts.

After smoke passes, stage is automatic. It can also be repeated with:

```
.venv/bin/python -B w10_open_sleep.py stage
.venv/bin/python -B kaggle/w10/dispatch.py --check-only
```

Grok, foreground (reads credentials privately; never echo them):

```
.venv/bin/python -B kaggle/w10/dispatch.py
```

This creates/versions the private code dataset ezzzzzx/pdp-code-w10 and launches
12 private CPU kernels ezzzzzx/w10-s120 through ezzzzzx/w10-s131, internet off.
Each kernel runs four single-threaded arm workers. W10 concurrency <=2 and the
account cap <=5 include queued and cancelling kernels. H34h's cap is three.
Reinvoke the same command after interruption; dispatch state, downloaded
outputs and code hashes are checked. At most one retry per failed kernel.
Unknown account status blocks submission. No network is called by --check-only.

Collect/assemble only after every kernel finishes:

```
.venv/bin/python -B kaggle/w10/summarize.py
```

Before scored kernels, make explicitly pending numeric tables with:

```
.venv/bin/python -B kaggle/w10/summarize.py --pending
```

Scored data go under results/w10/s<seed>/. The summary writes RESULTS.md and
results.json, gaps first, then sleep shares, daytime noise sleep share, T_open,
final checkpoints and walls. It refuses missing lives without --pending.

Unchanged preregistered bars on 12 seeds: mean pressure_retina night_sleep_share
>=.55; pressure_retina-fixed_random<=-.05 on >=8 seeds; mean
pressure_retina-fixed_night<=+.05. nosleep and daytime noise sleep share are
report-only. No verdict is generated.

Timing uses four concurrent single-threaded arm workers: L512 times scale by
8, the two measured full lives bound nosleep/retina from below, and the kernel
estimate is 2 times the four-worker makespan plus 30 seconds. The completed
smoke's original serial estimate is retained in smoke_serial_timing.json;
smoke.json records the corrected estimate and the unchanged behavior AST
outside smoke/timing. Completed measurements were revalidated after this
timing-only correction. No learning or controller code changed.

Offline tooling checks (no network or learner runs):

```
.venv/bin/python -B kaggle/w10/offline_checks.py
```
