Created [script](/root/pdp/h21a_substrate.py), [report](/root/pdp/results/h21a/RESULTS.md), and [JSON](/root/pdp/results/h21a/results.json). No commit or verdict.

Bits/symbol, mean ± population std:

| Learner / floor | Slow | Fast |
|---|---:|---:|
| gru | 1.259 ± 0.130 | 1.657 ± 0.239 |
| lin_fw | 2.272 ± 0.248 | 2.963 ± 0.095 |
| KT mixture | 1.735 ± 0.376 | 1.712 ± 0.038 |
| lzma -9e | 0.744 ± 0.065 | 0.976 ± 0.011 |

Wall seconds per stream:

| World / learner | Seed 6 | Seed 7 | Seed 8 |
|---|---:|---:|---:|
| Slow gru | 195.0 | 203.1 | 223.8 |
| Slow lin_fw | 286.1 | 263.2 | 227.4 |
| Fast gru | 199.9 | 193.2 | 212.2 |
| Fast lin_fw | 228.8 | 229.2 | 241.8 |

Full run: **47.0 minutes**, plus a 65-second smoke test. `gru_big` was dropped because the smoke projection for all learners exceeded 90 minutes.

`lin_fw` exceeded KT at every 20% checkpoint; no tuning followed. Slow streams omitted M7/C7/E30 on seeds 6/7/8 respectively; missing values are documented. Causality and aggregate checks passed.

An unrelated change to `tools/sync_papers.py` appeared during execution; I left it untouched.