# Run H34a2's kernels, write up H34a, H34a2 and M11 (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md`, then the specs `handoffs/2026-11-16_codex_h34a_local_depth.md`, `handoffs/2026-11-17_codex_h34a2_logit_sum.md`
and `handoffs/2026-11-17_codex_m11_w6_sleep_credit.md`. Codex wrote all the code and passed the smoke tests; its sandbox has no network.

## Do, in the foreground (load `.env` first: `set -a; . ./.env; set +a`)
1. **H34a:** all 16 kernel outputs are in `results/h34a/kernels/`. Run `.venv/bin/python kaggle/h34a/summarize.py`.
2. **H34a2:** run `.venv/bin/python kaggle/h34a2/dispatch.py` until all 20 kernels finish. Kaggle capacity: before each push keep the account's
   running kernels ≤ 5 (`kaggle kernels list --mine`); M11 (2 at a time) and H35 also run today. Then run `kaggle/h34a2/summarize.py`.
3. **M11:** its dispatcher already runs detached (`results/m11w6/dispatch.run.log`, state in `results/m11w6/dispatch.json`). Do not start a
   second one. Wait until all 24 kernels are complete and fetched, then write `results/m11w6/RESULTS.md` and `results.json` with the outputs
   the M11 spec lists.
4. **At the top of each of the three RESULTS.md files, a compact scoring table:** for each pre-registered bar in that spec, the quantity per
   seed, the mean, and the count of seeds that meet the bar's threshold (numbers only; no PASS/FAIL words, no verdict).
   - H34a bars: left_4 − left_1 ≤ −0.10; left_4 − plain_4 ≤ −0.05; left_4 − wide_1 ≤ −0.03 (full-run means, seeds 0–1). Add the same three on
     the tail (50–300 KB) as a second table.
   - H34a2 bars: sum_4 − sum_1 ≤ −0.10; sum_4 − sumown_4 ≤ −0.05; sum_4 − wide_1 ≤ −0.03. Also plain_4 − plain_1.
   - M11 bars: age_sleep_cf − age_sleeparm ≤ −0.05 on T_open (count of 12); mean of age_sleep_cf − age_fixed20 vs +0.03. Also sleep shares per
     arm and age_fixed20 − age_nosleep.
If a script fails, fix only plumbing (paths, `.env`, CLI); never change models, data, seeds or bars. If something else looks wrong, stop and report.

## Rules
Files you may edit or create: `kaggle/h34a/**`, `kaggle/h34a2/**`, `results/h34a/**`, `results/h34a2/**`, `results/m11w6/**`. Do not touch
`docs/loop.md`, `audit.md`, any `*.py` at the repo root, `kaggle/m11w6/**`, H35/H36 files, `.env`. Never print secrets. Do not commit.
