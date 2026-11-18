# Run H36's and H35b's kernels (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md`, the specs `handoffs/2026-11-17_codex_h36_curious_patching.md` (+ startup rule in
`handoffs/2026-11-18_codex_h36_start_h35b.md`) and `handoffs/2026-11-18_codex_h36_start_h35b.md` (Job 2, H35b), and `results/h36/RESULTS.md`,
`results/h35b/RESULTS.md`. Codex passed every smoke check; its sandbox has no network.

Load `.env` (`set -a; . ./.env; set +a`). Run both dispatchers in the foreground, in parallel, until every kernel finishes:
`.venv/bin/python kaggle/h36/dispatch.py` (24 kernels; allow **3** at once) and `.venv/bin/python kaggle/h35b/dispatch.py` (12 kernels; 2 at once).
Keep the account's running kernels ≤ 5. Then run `kaggle/h36/summarize.py` and `kaggle/h35b/summarize.py` and check that each RESULTS.md starts with
the scoring table its spec asks for. Fix only plumbing (paths, `.env`, CLI, concurrency); never change models, data, seeds or bars.

Rules: files you may edit: `kaggle/h36/**`, `kaggle/h35b/**`, `results/h36/**`, `results/h35b/**`. Do not touch `docs/`, `audit.md`, any `*.py` at the
repo root, other folders, `.env`. Never print secrets. Do not commit.
