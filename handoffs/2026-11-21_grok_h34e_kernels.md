# Run the H34e kernels (grok, Kaggle CPU)
Read `/root/pdp/CLAUDE.md` and `kaggle/h34e/README.md`. Codex implemented H34e and the smoke tests passed. Token from `.env` via `set -a; . ./.env; set +a` (never print it).
Run `.venv/bin/python -B kaggle/h34e/dispatch.py --dispatch` in the foreground (≤ 2 H34e at once, account ≤ 5 running), rerun once on failure (it resumes), then run
`kaggle/h34e/summarize.py`. Report the scoring table at the top of `results/h34e/RESULTS.md`, kernel ids and failures. Do not edit code, bars or docs. Never print
secrets. Do not commit.
