# W11 runner handoff

To continue the interrupted local smoke, preserving its hash-checked passing checks:

```bash
.venv/bin/python kaggle/w11/finish_smoke.py
```

The continuation saves completed smoke lives before proceeding, checks exact checkpoint restoration, measures the remaining walls, and stages the kernels. It never runs scored seeds. A fresh smoke remains available with `.venv/bin/python w11_deep_cortex.py smoke`.

The checkpoint smoke compares the full 512-decision behavioral output exactly. W10's pickle-byte learner checksum is used within each eyelid glance, where object sharing is unchanged; its bytes differ across a pickle round trip even for a fresh untrained deep cortex. The checkpoint check therefore also compares all saved values directly (world, generators, controller, stores, banks, traces and Adam state), then requires identical next wake and replay bits and updated cortex values. Cross-run pickle hash equality is reported separately. The resumed eyelid checks must occur at identical decisions and preserve their own before/after hashes. Only smoke assertions changed to make this distinction; `smoke.json` retains the prior source hashes and records the verified smoke-only change.

From `/root/pdp`, after the local smoke and staging finish:

```bash
.venv/bin/python kaggle/w11/dispatch.py --check-only --seeds 132 133 134 135 136 137 --cortices bank_1 deep_4
.venv/bin/python kaggle/w11/dispatch.py --seeds 132 133 134 135 136 137 --cortices bank_1 deep_4
```

The reduced run selects seeds 132–137 with `bank_1` and `deep_4`: 12 lives, 84 private CPU kernel parts, 30 code files. The original 36-life, 312-part manifest remains staged. `bank_1` uses 2 parts per seed; both deep cortices use 12. The conservative per-part bounds are 3607.344, 5008.771, and 5053.888 seconds respectively (limit 5100). The layout check and an isolated import of the code dataset passed. The reduced numeric reports currently contain 0/12 scored lives; the orchestrator launches the dispatcher above. This amendment was staged offline; no scored run was launched.

The dispatcher reads `.env` itself without printing credentials. It creates or versions the private `ezzzzzx/pdp-code-w11` dataset, runs private CPU kernels with internet disabled, fetches and validates their outputs, and invokes the summarizer. It runs in the foreground; rerun the same filtered command to resume. One shared `dispatch.lock` protects all selections. `dispatch.json` retains the full manifest and existing attempts/statuses; only selected jobs are validated, fetched, retried or submitted. Unselected records remain unchanged, and remote jobs outside the selection still count toward slot limits. The same filters are forwarded to the final summarizer and by `--assemble-only`. W11 occupies at most three kernel slots; the account cap is five, including other experiments. Each kernel has one worker and one numerical-library thread. Each job gets at most one retry. An uncertain push is reconciled before another submission.

The manifest contains seeds 132–143 crossed with `bank_1`, `deep_4`, and `deep_4_open`, all using `pressure_retina` and `prog_age_s`, L=4096 and birth period 128. Unsplit kernel IDs are `ezzzzzx/w11-<cortex-with-hyphens>-s<seed>`; split jobs append `-p<part>`. Part jobs attach their completed predecessor through `kernel_sources`. They restore an exact serialized decision boundary, including the world, all random generators, models, optimizer state, eligibility traces, wake state, stores, pending visit and sleep, controller, accumulated prequential bits, scoring checkpoints and mixer history. No boundary changes a glance update, birth or exam schedule. The 253-decision smoke split verifies this inside a visit.

The wall estimate measures learning, room scoring at 32 glances, full replay scoring at 576 glances, and an 18-room 2048-symbol exam. It bounds every decision by the larger of wake and sleep work, adds checkpoint exams, and applies a 2x margin plus 30 seconds. Parts are chosen so this bound is at most 5100 seconds per part. These are conservative local CPU estimates, not Kaggle measurements. Actual walls are retained per kernel and per life.

`bank_1` uses the original W4 wrapper around M7's unchanged BankCortex, as W10 does (19 channel rows). Layer 1 in either deep cortex uses the original M7 trace function. Upper banks retain 256 units, the roots initialization with magnitude .98, 32-dimensional symbol and channel embeddings, and the same order-2 readout. Their extra input is the mean of lower-layer features `[Re(h), Im(h), P1(h)*P2(h)]`, each normalized per step with H34a2's unchanged RMS normalization. That gives 576 feature columns and 608 total input columns. Lower features and mixture inputs are detached; each bank receives only its own next-symbol loss, with M7's Adam, lr .001, clip 1 and chunk 64. Upper bank seeds are model_seed+k; extra input weights use local generators seeded 70000+model_seed+k and variance 1/input_width. Layer 1's initialization is unaffected. GeoMixer is the imported H32b mixer, trained as H34f, with Adam lr .003. Its weights are unconstrained, and must not be normalized when reporting them.

The mixture supplies wake bits, replay bits, progress exams, snapshots, store rescoring, and W6 held-out exams. The eye retains W10's cortex-independent retina signal. The report's `preq_bps` is W6's lifetime prequential score: bits on learnable-room stay glances divided by symbols on those glances. `T_open` is W10's primary score: the mean of its eight clean, frozen W6 exam checkpoints. No new composite score is introduced. Mixer means sample the weights before each decision (wake or sleep); endpoint weights are recorded after each 512 decisions.

To assemble an incomplete numeric table, keeping missing scored lives as `NA`/null:

```bash
.venv/bin/python kaggle/w11/summarize.py --pending --seeds 132 133 134 135 136 137 --cortices bank_1 deep_4
```

To reassemble and require all 12 selected completed lives:

```bash
.venv/bin/python kaggle/w11/summarize.py --seeds 132 133 134 135 136 137 --cortices bank_1 deep_4
```

Outputs are `results/w11/RESULTS.md` and `results/w11/results.json`. The two fixed bars are deep_4 minus bank_1 on T_open <= -.05 on at least 5/6 seeds, and mean deep_4 night_sleep_share >= .55. `deep_4_open` is excluded from the reduced run. Filters restrict input validation, tables, counts and completeness checks; missing selected lives remain NA/null under `--pending`. Without filters the original full selection and 8/12 bar remain available. Other seed subsets receive descriptive counts without a newly invented pass bar. No scored seeds run locally. No results are inferred from the smoke.
