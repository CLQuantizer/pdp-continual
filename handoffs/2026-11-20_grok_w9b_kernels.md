# Run the W9b kernels (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` and `kaggle/w9b/README.md`. Codex implemented W9b and the smoke test passed. Run the staged dispatcher exactly as the README
says (token from `.env` via `set -a; . ./.env; set +a`; never print it), keep the account at ≤ 5 running kernels (W9b ≤ 3), rerun once on failure (it resumes),
then run the summarizer. Foreground until all 12 kernels finish. Report the scoring table at the top of `results/w9b/RESULTS.md`, kernel ids and any failures.
Do not edit code, bars or docs. Never print secrets. Do not commit.
