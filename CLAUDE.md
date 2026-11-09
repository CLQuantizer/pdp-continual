# pdp: instructions for agents

Research repo for **learning by Ez** (public log: learning.mywords.io). Goal: a machine that *lives* (learns and acts at
once, no train/inference split) and learns the way Schmidhuber describes: it compresses its history and is curious
about what it can still learn to compress.

Read first, in order: `docs/curiosity.md` (the guiding principle), `docs/goal.md` (criteria), the newest entries of
`docs/loop.md`, `docs/hippocampus_diagnosis.md`.

## Guideline (binding)

Schmidhuber 2009, *Driven by Compression Progress* (arXiv 0812.4360), in `docs/papers/`, summarised as rules in
`docs/curiosity.md`. Every hypothesis states which of its four principles it serves (store everything; improve the
compressor; reward compression progress; act to maximise it). Score in **bits**, on **observations**, with the store
priced. A hypothesis that serves none of the four is polishing.

## Roles

- **Orchestrator (Claude):** picks the next hypotheses, writes pre-registrations and bars, writes the core research
  code, reviews every handoff, scores results against the bars, writes verdicts in `docs/loop.md`, commits.
- **Workers (codex, grok):** simple, well-specified tasks handed off by the orchestrator: baselines and floors,
  plotting, refactors with a test, citation checks, literature lookups, reproducing a run, site and publish fixes.
  Workers never pre-register, never change a bar, never write a verdict, and never edit `docs/loop.md` or `audit.md`.
- **The loop (pdp-loop):** the scheduled research loop on the compute box. It follows this file and the hypothesis
  queue at the end of `docs/loop.md`.

### How to hand off

1. Write the spec to `handoffs/YYYY-MM-DD_<worker>_<slug>.md`: context, files it may create, files it must not touch
   (always: `docs/loop.md`, `audit.md`, running experiment code), exact outputs, a smoke test, a time budget, and "do not
   commit".
2. Run the worker on it. Its final message goes to `handoffs/<same name>.out.md`:
   - codex: `codex exec --cd /root/pdp --skip-git-repo-check -o handoffs/<name>.out.md "$(cat handoffs/<name>.md)"`
   - grok: `grok -p "$(cat handoffs/<name>.md)" --cwd /root/pdp --output-format plain --max-turns 40 > handoffs/<name>.out.md`
   Running a worker unattended with auto-approval (or in the background) needs the user's go-ahead in the session.
   The permission classifier blocks it otherwise; do not route around that.
3. The orchestrator reviews the diff and the output, reruns the smoke test, and only then commits, crediting the
   worker in the commit message.

Hand off: anything with a clear spec and a checkable output. Keep: deciding what to test, bars, verdicts, diagnosis,
anything where being subtly wrong would mislead the research.

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

- This VPS: 2 cores, 3 GB RAM. Use `.venv/bin/python` (system python has no numpy). Pool size ≤ 2. FF costs about
  50 ms per arrival here; budget runs accordingly. Heavy sweeps go to the box.
- Kaggle (datasets for new worlds): `set -a; . ./.env; set +a; .venv/bin/kaggle datasets download <owner/name> -f <file>`
  (CLI reads `KAGGLE_API_TOKEN` from `.env`, which is gitignored; never print or commit it). Downloads go to `data/`.
- The loop commits as `pdp-loop` and pushes to `origin` (GitHub). Before committing: `git pull --rebase origin main`.
  Append to `docs/loop.md`; never rewrite earlier entries. Don't edit a file the loop is running (check
  `docs/loop.md` for "running").
- Publish after a material commit: `tools/publish.sh` (see `docs/publish.md`). Public text: byline **Ez**, times
  **UTC+8**, no hostnames, IPs, home paths, emails or tokens.
