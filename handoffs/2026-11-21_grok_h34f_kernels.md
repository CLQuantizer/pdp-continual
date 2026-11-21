# Run the H34f kernels (grok, Kaggle CPU)
Read `/root/pdp/CLAUDE.md` and `kaggle/h34f/README.md`. Codex implemented H34f and the smoke tests passed. Token from `.env` via `set -a; . ./.env; set +a` (never print it).
Run `.venv/bin/python -B kaggle/h34f/dispatch.py --dispatch` in the foreground (≤ 2 H34f at once, account ≤ 5 running), rerun once on failure (it resumes), then run the
H34f summarizer. Report the scoring table at the top of `results/h34f/RESULTS.md`, kernel ids and failures. Do not edit code, bars or docs. Never print secrets.
Do not commit.
