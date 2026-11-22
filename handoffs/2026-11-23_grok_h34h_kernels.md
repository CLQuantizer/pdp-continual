# Run the H34h kernels (grok, Kaggle CPU)
Read `/root/pdp/CLAUDE.md` and `kaggle/h34h/README.md`. Codex implemented H34h and the smoke tests passed. Token from `.env` via `set -a; . ./.env; set +a` (never print it).
Run `.venv/bin/python -B kaggle/h34h/dispatch.py --dispatch` in the foreground (≤ 3 H34h at once, account ≤ 5 running; codex may stage W10 kernels later, so leave room
when you can), rerun once on failure (it resumes), then the H34h summarizer. Report the ablation table and bar gaps at the top of `results/h34h/RESULTS.md`, kernel ids
and failures. Do not edit code, bars or docs. Never print secrets. Do not commit.
