# H34a kernels: push, collect, write up (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md`, then `handoffs/2026-11-16_codex_h34a_local_depth.md` (the spec, with both amendments) and
`results/h34a/RESULTS.md`. Codex implemented `h34a_local_depth.py`, passed every smoke check and prepared the kernels in
`kaggle/h34a/` with the dispatcher `kaggle/h34a/dispatch.py`, but its sandbox has no network, so no kernel ran. The
300,000-byte fallback applies (pre-registered; the 500 KB estimate exceeded 90 minutes).

## Do
1. Read `kaggle/h34a/dispatch.py` and check that it pushes the 16 kernels as the spec says (private, CPU, no internet, at most
   3 at once, one retry). Fix only network/CLI plumbing if needed (e.g. `.env` loading for `KAGGLE_API_TOKEN`); do not change
   `h34a_local_depth.py`, arms, data, seeds or bars. If anything else looks wrong, stop and report.
2. Create the private code dataset `ezzzzzx/pdp-code-h34a` if it does not exist, then run the dispatcher in the foreground until
   all 16 kernels finish (poll `kaggle kernels status`). At most 3 H34a kernels at once.
3. Fetch the outputs into `results/h34a/` and write `results/h34a/RESULTS.md` and `results.json` with every output the spec's
   "Outputs" section lists (numbers only, no verdict), keeping the smoke and compute sections already there.

## Rules
- Files you may create or edit: `kaggle/h34a/**`, `results/h34a/**`. Do not touch `docs/loop.md`, `audit.md`, `h34a_local_depth.py`,
  any other code, `.env`. Do not edit `m11_w6.py`, `kaggle/m11w6/**`, `results/m11w6/**` (codex writes them today).
- Never print or upload secrets. Do not commit.
