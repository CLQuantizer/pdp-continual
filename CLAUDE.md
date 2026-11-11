# pdp: instructions for agents

Research repo for **learning by Ez** (public log: learning.mywords.io). Goal: a machine that *lives* (learns and acts at
once, no train/inference split) and learns the way Schmidhuber describes: it compresses its history and is curious
about what it can still learn to compress.

Read first, in order: `docs/roadmap.md` (direction and milestones), `docs/curiosity.md` (the guiding principle), `docs/goal.md` (criteria), the newest entries of
`docs/loop.md`, `docs/hippocampus_diagnosis.md`.

## Guideline (binding)

Schmidhuber 2009, *Driven by Compression Progress* (arXiv 0812.4360), in `docs/papers/`, summarised as rules in
`docs/curiosity.md`. Every hypothesis states which of its four principles it serves (store everything; improve the
compressor; reward compression progress; act to maximise it). Score in **bits**, on **observations**, with the store
priced. A hypothesis that serves none of the four is polishing.

## Roles

- **Orchestrator (Claude):** direction (`docs/roadmap.md`), which hypotheses run, the pre-registration and bars
  written *inside* each worker spec, review of every result, verdicts in `docs/loop.md`, re-ranking, commits. Writes
  no experiment code; only spot-checks a worker's code or numbers when something looks wrong.
- **Workers (codex sandboxed, grok):** implement and run the experiments from the spec, and write
  `results/<name>/RESULTS.md` with numbers only. Workers never change a bar, never write a verdict, never commit,
  and never edit `docs/loop.md` or `audit.md`.
- **The loop (pdp-loop):** the scheduled research loop on the compute box. It follows this file and the hypothesis
  queue at the end of `docs/loop.md`.

### How to hand off

1. Write the spec to `handoffs/YYYY-MM-DD_<worker>_<slug>.md`: context, files it may create, files it must not touch
   (always: `docs/loop.md`, `audit.md`, running experiment code), exact outputs, a smoke test, a time budget, and "do not
   commit".
2. Run the worker on it. Its final message goes to `handoffs/<same name>.out.md`:
   - codex: `codex exec --cd /root/pdp --sandbox workspace-write --skip-git-repo-check -o handoffs/<name>.out.md "$(cat handoffs/<name>.md)"`
   - grok: `grok -p "$(cat handoffs/<name>.md)" --cwd /root/pdp --output-format plain --max-turns 40 > handoffs/<name>.out.md`
   Always pass `--sandbox workspace-write` to codex (its config defaults to full machine access); never drop it.
   If the sandbox fails at startup with "Can't mkdir", some repo path is owned by a foreign uid (the Mac copy was
   uid 501); fix with `find /root/pdp -uid 501 -exec chown root:root {} +`.
   Running a worker unattended with auto-approval (or in the background) needs the user's go-ahead in the session.
   The permission classifier blocks it otherwise; do not route around that.
3. The orchestrator reviews the diff and the output, reruns the smoke test, and only then commits, crediting the
   worker in the commit message.

Hand off: implementing and running experiments, baselines, plots, lookups. Keep: deciding what to test, bars,
verdicts, diagnosis. Specs must be complete enough that a literal-minded worker gets it right: exact data, seeds, N,
learners, what "prequential" means, outputs, a smoke test, a time budget, and the bars (marked "do not change").

## Papers library (rule from Ez, 9 Nov 2026)

Every paper we look at or cite gets a row in D1 **`pdp-papers-index`** (table `papers`), and its file goes in R2
**`pdp-papers`** when we can get it. Two separate columns:
- `status`, how well we know it: `read` / `checked` / `memory` (same terms as `docs/papers/README.md`).
- `file_status`, where the file is: `stored` (in R2) / `to_fetch` (open access, next sync downloads it) / `wanted` (hard
  to get: Ez obtains it, e.g. paying the author, and hands over the file) / `unavailable`.
Whenever a doc, spec or loop entry cites a new paper, register it in the same commit's work (sync:
`python3 tools/sync_papers.py`). When Ez provides a `wanted` file, ingest it and flip it to `stored`.

## Worker budget (Ez, 10 Nov 2026)

- **grok is the default worker** for experiments and lookups.
- **codex has a hard cap of 80% of its weekly plan.** Check `python3 /root/bin/usage_report.py --dry` before any codex
  dispatch; at ≥ 75%, do not dispatch codex at all. Use codex only when Ez says so.
- grok runs that edit files or call `cf`: `grok -p "..." --cwd /root/pdp --output-format plain --max-turns 150 --always-approve --background-wait-timeout 7200` (headless grok otherwise kills its background jobs after 600 s; tell it to run experiments in the foreground).

## Cadence (Ez, 9–10 Nov 2026)

The daily 08:40 UTC+8 orchestration is the baseline: review, verdicts, next specs, dispatch. With Kaggle as the runner,
further cycles the same day are fine while the usage bars allow (Claude 5h/week, grok); codex stays capped. Never run
continuously without results to review in between.

## Research protocol (unchanged from the loop; keep it)

- **Pre-register before running:** prediction, bar, kill rule, seeds. Selection on seeds 0–5 (or a pilot), scoring on
  held-out seeds; never retune on scoring seeds.
- **Check oracle headroom before setting a bar.** A bar larger than the oracle's gain is a guaranteed kill.
- **Kill fast, record why.** A killed hypothesis gets one line on *why* (mechanism), not only the number.
- **Loop budget:** at least one hypothesis in three changes the world, the objective or the substrate. No more than
  two scalar-parameter sweeps in a row. Stop a line when its effect is below seed-to-seed std.
- **Log:** every material run gets a `docs/loop.md` entry (Findings / Literature insight / Paths closed / Hypotheses,
  ranked) and a `results/<name>/RESULTS.md` write-up. `results/` data stays out of git; only RESULTS.md is published.

## Machines and git

- This VPS: 2 cores, 3 GB RAM. Use `.venv/bin/python` (system python has no numpy). Pool size ≤ 2. Heavy runs go to
  Kaggle (below).
- Kaggle (datasets for new worlds): `set -a; . ./.env; set +a; .venv/bin/kaggle datasets download <owner/name> -f <file>`
  (CLI reads `KAGGLE_API_TOKEN` from `.env`, which is gitignored; never print or commit it). Downloads go to `data/`.
- **Kaggle is the default runner for experiments (Ez, 10 Nov 2026).** Every scored run goes to private Kaggle kernels
  (split seeds across up to 3 kernels; M4c: 36 lives in ~8 min vs ~2 h here). This VPS does specs, smoke tests at tiny T,
  review and commits. Run locally only when the job is under ~10 min or needs something Kaggle lacks, and say why in the
  spec. Use the GPU only when the model is big enough to gain from it (the weekly GPU quota is finite).
- **Kaggle compute (tested 10 Nov 2026):** a headless kernel gets 4 CPUs, 32 GB RAM and a Tesla T4 (torch with CUDA
  preinstalled), free within Kaggle's weekly GPU quota (~30 h; sessions ~12 h max). Use it for anything bigger than this VPS
  allows: many seeds in parallel, larger cortices, real-data worlds. Recipe: a folder with the script +
  `kernel-metadata.json` (`"id": "ezzzzzx/<slug>"`, `"kernel_type": "script"`, **`"is_private": true`**, `"enable_gpu"`
  as needed, `"enable_internet": false`), then `kaggle kernels push -p <dir>`, poll `kaggle kernels status <id>`, fetch
  with `kaggle kernels output <id> -p <dir>`. Code and data go up as private Kaggle datasets (`kaggle datasets create`),
  never with secrets inside. Results still come back into `results/<name>/` and are reviewed here.
- **GPU quota guard (Ez, 11 Nov 2026):** check `kaggle quota` before every GPU dispatch and put the hours in the spec.
  Keep a reserve of 10 GPU h per week (refresh Sat 08:00 UTC+8). No single job over 3 GPU h without Ez's go-ahead.
  Prefer CPU kernels (no weekly quota) when the model fits: smaller cortex, fewer bytes.
- The loop commits as `pdp-loop` and pushes to `origin` (GitHub). Before committing: `git pull --rebase origin main`.
  Append to `docs/loop.md`; never rewrite earlier entries. Don't edit a file the loop is running (check
  `docs/loop.md` for "running").
- Publish after a material commit: `tools/publish.sh` (see `docs/publish.md`). Public text: byline **Ez**, times
  **UTC+8**, no hostnames, IPs, home paths, emails or tokens.
