# Run the H34d and W9 kernels (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md`, then `kaggle/h34d/README.md` and `kaggle/w9/README.md`. Codex implemented both and passed the smoke tests. Your job is
only to run the staged kernels and the summarizers.

1. `kaggle kernels list --mine` (token from `.env` via `set -a; . ./.env; set +a`; never print it). Keep the account's running kernels ≤ 5 at all times.
2. H34d: `.venv/bin/python kaggle/h34d/dispatch.py --prepare-only`, then `--dispatch` (6 kernels, about 18 minutes each). When all are done,
   `.venv/bin/python kaggle/h34d/summarize.py`.
3. W9: `.venv/bin/python kaggle/w9/dispatch.py --check-only`, then `kaggle/w9/dispatch.py` (12 kernels). When done, `kaggle/w9/summarize.py`.
   Run the two dispatchers one after the other or side by side, whichever keeps the account at ≤ 5 running kernels (H34d ≤ 2 and W9 ≤ 3 at once).
4. If a dispatcher fails, rerun the same command once (both resume). If it fails again, stop and report the error (no secrets).
5. Papers: check that D1 `pdp-papers-index` contains He et al. 2016, "Deep residual learning for image recognition" (arXiv 1512.03385), and
   Howard 2019 Imagenette (https://github.com/fastai/imagenette, as a dataset entry if the index allows). Register any that are missing, as CLAUDE.md says.

Run everything in the foreground until all kernels finish. Report at the end: the scoring tables at the top of `results/h34d/RESULTS.md` and
`results/w9/RESULTS.md`, kernel ids, and failures. Do not edit code, bars or docs. Never print secrets. Do not commit.
