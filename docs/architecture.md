# Architecture

How the research runs. One Claude Code session on a small VPS is the orchestrator. It decides what to test, writes the
spec with its bars fixed in advance, and judges the result. Worker CLIs write and run the experiment code. Kaggle does
the heavy compute. Cloudflare serves this site and holds the papers library. A Telegram message reports each day.

<iframe class="fig" src="../figures/architecture.html" title="Architecture: orchestrator, workers, Kaggle, Cloudflare, Telegram"></iframe>

## One research cycle

1. **Spec.** Claude Code writes `handoffs/<date>_<worker>_<slug>.md`. It holds the hypothesis, the Schmidhuber principle
   it serves, seeds, the smoke test, a time budget and the bars, marked "do not change". The oracle headroom is checked
   before any bar is set.
2. **Dispatch.** A worker CLI gets the spec. grok is the default. codex runs sandboxed and is capped at 80% of its
   weekly plan, so it stays off unless Ez asks for it.
3. **Smoke test on the VPS.** The worker runs a tiny-T check locally. It also checks that the new script reproduces the
   previous experiment's numbers exactly.
4. **Score on Kaggle.** The code goes up as a private Kaggle dataset. Scored seeds are split across up to three private
   kernels, with internet off. The CPU is the default and the T4 GPU is used only when the model needs it. One sweep that
   takes about 2 h on the VPS finishes in about 8 min there. Kaggle and VPS numbers agree to 2e-7.
5. **Results.** The worker fetches the outputs and writes `results/<name>/RESULTS.md`, with numbers only. It never
   writes a verdict and never commits.
6. **Verdict.** Claude Code reviews the code and the numbers and scores the pre-registered bars. It writes the verdict
   and a [loop log](../loop/) entry, then commits with the worker credited and pushes to GitHub.
7. **Publish.** `tools/publish.sh` renders the docs to static HTML. A Cloudflare Tunnel serves them here.
8. **Report.** `usage_report.py` sends Ez one Telegram message: what ran, the verdicts, and the Claude, codex and grok
   usage bars.

A scheduled task wakes the orchestrator at 08:40 UTC+8 each day. Further cycles run the same day while the usage bars
allow. A new cycle starts only when there are results to review.

## The parts

| Part | Where | Role |
|---|---|---|
| Claude Code (orchestrator) | VPS | Direction ([roadmap](../roadmap/)), specs, bars, review, verdicts, commits. It writes no experiment code. |
| grok CLI | VPS | Default worker: implements the spec, runs the smoke test, drives Kaggle, writes RESULTS.md, registers papers. |
| codex CLI | VPS | Second worker, always run with `--sandbox workspace-write`. Its plan is capped at 80% a week. |
| Scheduled task | Claude Code | Daily 08:40 UTC+8 wake-up for review, verdicts and the next specs. |
| Kaggle kernels | Kaggle | Private script kernels: 4 CPUs, 32 GB RAM, a T4 when needed. The default runner for every scored experiment. |
| Kaggle datasets | Kaggle | Private code bundles for the kernels, and public data for real-data worlds (Bible text, chess moves, SQuAD, order books). |
| Colab CLI | Google | Optional overflow for occasional GPU jobs. |
| R2 `pdp-papers` | Cloudflare | PDFs of the papers we read or cite. A private bucket. |
| D1 `pdp-papers-index` | Cloudflare | One row per paper, with how well we know it (`read` / `checked` / `memory`) and where the file is (`stored` / `wanted` / …). |
| Tunnel | Cloudflare | Serves the static site from the VPS. |
| GitHub | GitHub | The repository: code, specs, RESULTS.md write-ups and the loop log. |
| Telegram bot | Telegram | The daily summary to Ez's phone. |

The VPS is deliberately small: 2 cores and 3 GB RAM. It does the thinking and the bookkeeping, and the computing goes
elsewhere. This mirrors the research question itself. The note on [compute vs data movement](docs/compute-vs-data-movement.md) asks what a
learner should keep close and what it should fetch.
