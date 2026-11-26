# W11 dispatcher: completed parts marked failed (codex; diagnose + fix; no network)

Read `/root/pdp/CLAUDE.md`, `kaggle/w11/dispatch.py`, `kaggle/w11/README.md`, and `results/w11/dispatch.json`.

The dispatcher stopped with "Kernel exhausted its single retry". `ezzzzzx/w11-bank-1-s132-p2` and `ezzzzzx/w11-bank-1-s133-p2` are marked failed after 2 attempts, but their fetched
outputs show a clean finish: `results/w11/bank-1-s132-p2/w11-bank-1-s132-p2.log` ends with `{"seed": 132, "cortex": "bank_1", "part": 2, "wall_s": 332.99, "T_open": 1.2392,
"decision": 4096}` and `bank_1-s132-p2.json` exists. Find why a finished final part is judged failed (output name, hash or field check, final-part handling, status parsing...),
report the cause in ≤ 5 lines, and fix it in the dispatcher only, so that re-running the same command resumes: accepts the two finished parts from their fetched outputs
(validate them with the same checks a good part gets), does not push them again, and continues with the remaining filtered jobs (seeds 132–137, bank_1 + deep_4). Do not change
the experiment code, the split plan or the bars. Add an offline test reproducing the case. Also note in ≤ 3 lines how measured walls compare with the staged estimates
(bank_1 parts ran about 333 s against an estimate of 3,607 s).

Files you may edit: `kaggle/w11/dispatch.py`, `kaggle/w11/README.md`, `results/w11/dispatch.json` (only to repair the two jobs' state if needed; keep a backup copy). No network.
Never print secrets. Do not commit. Report the exact relaunch command.
