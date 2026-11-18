H35b: fresh corpus seeds 6–11, first 1,600 shuffled documents, budget 200,
checkpoint interval 25, K=16. CPU, one Torch thread per process, no internet.
Principles 3 and 4. Bars are unchanged and belong to the orchestrator.

Measured scheduling: the unsplit sequential estimate is 141.3 minutes.
Two kernels per seed are prepared: `s<c>` runs oracle/high_loss/rho (64.7
minutes); `s<c>-p` runs rand/prog_peek (76.6 minutes). Policies run sequentially
within each kernel. The split balances only measured wall time; the learner,
random seeds, scores, and policy parameters are unchanged. The original
four-policy/progress-only split projected 83.7/57.6 minutes; its timing record
is preserved as `results/h35b/smoke-before-split-plan.json`. The final schedule
is in `results/h35b/smoke.json` and `code-sha256.json`.

Local smoke (foreground):

```sh
.venv/bin/python h35b_neural_diet.py --smoke
.venv/bin/python kaggle/h35b/dispatch.py --prepare-only
.venv/bin/python kaggle/h35b/summarize.py
```

After review, on a network-enabled runner, the foreground dispatcher is:

```sh
.venv/bin/python kaggle/h35b/dispatch.py
```

It reads the Kaggle token from `.env` without displaying it, uploads only the
three files listed in `code-sha256.json`, creates/versions the private
`ezzzzzx/pdp-code-h35b` dataset, and dispatches the prepared private CPU kernels.
Account running/queued/pending kernels are capped at five, H35b at two, with
one retry per failed kernel. Dataset creation uses the CLI's private default.
No network operation is performed by `--prepare-only` or `summarize.py`.

Re-run the same dispatcher to resume. Its state is
`results/h35b/dispatch.json`; source/configuration changes after dispatch starts
are rejected. An interrupted submission whose remote acceptance cannot be
established stops for inspection rather than launching duplicate work. Output
validation checks source fingerprint, identities, budget, checkpoint means,
unique selected documents and document-kind shares before accepting a result.

The GRU predicts from the previous byte, with zero input at document starts.
Hidden state resets at each document; weights and Adam moments continue.
Within each 128-byte scored training chunk, gradients truncate every 64 bytes;
one Adam step follows the scored chunk. A progress trial deep-copies the GRU
and Adam, truncates every 64 bytes, and takes exactly one step on bytes 0–255.
Both versions score the candidate tail after replaying its prefix as frozen
context; reservoir pieces each start with reset hidden state. The reservoir
stores four first-256-byte pieces, one sampled per document, matching H35's
sampling convention. Progress always divides by 1,280, even during filling.

Scored results are not populated by smoke runs. `RESULTS.md` starts with the
registered scoring quantities and threshold counts, then includes checkpoints,
T_clean, final, kind shares, walls, kernel ids, and smoke numbers. Missing
scored seeds remain empty; smoke outcomes are not experimental verdicts.
