# Run H35b2's kernels (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md`, `handoffs/2026-11-18_codex_h35b2_snapshot.md` and `results/h35b2/RESULTS.md`. Codex passed every smoke check, but one seed per
kernel needs ~122 minutes, over the 90-minute cap. **Split each seed into kernels by arm** (e.g. `rand`+`oracle`+`rho` in one, each `prog_snap_k` in its own),
changing only `kaggle/h35b2/` packaging and the dispatcher, never the model, data, seeds or bars. Check one split kernel's estimate ≤ 85 min. Load `.env`, run
the dispatcher in the foreground until all kernels finish (account running kernels ≤ 5; others are queued, so wait for slots), then `summarize.py`, with the
scoring table at the top (selection on seeds 6–8, scoring on 9–11, as the spec says).

Rules: files you may edit: `kaggle/h35b2/**`, `results/h35b2/**`. Do not touch `docs/`, `audit.md`, any `*.py` at the repo root, `.env`. Never print secrets. Do not commit.
