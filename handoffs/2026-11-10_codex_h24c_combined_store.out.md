Completed [implementation](/root/pdp/h24c_combined.py), [report](/root/pdp/results/h24c/RESULTS.md), and [results.json](/root/pdp/results/h24c/results.json). Smoke test passed. Full runtime: **72.8 minutes**. No commit; protected files unchanged.

Mean accuracy (%), seeds 6–8:

| Route | Lifetime | Normal | Episode 1 | 2 | 3–5 | 6+ |
|---|---:|---:|---:|---:|---:|---:|
| ff | 65.99 | 76.78 | 9.50 | 12.83 | 16.25 | 31.86 |
| exem | 87.33 | 86.32 | 12.83 | 100 | 100 | 100 |
| gate | 79.63 | 76.78 | 9.50 | 100 | 100 | 100 |
| ffr | 78.24 | 83.89 | 10.83 | 20.33 | 44.74 | 77.97 |
| gate_r | 85.35 | 83.89 | 10.83 | 100 | 100 | 100 |
| gate_rd | 84.24 | 84.07 | 10.67 | 100 | 96.74 | 89.64 |

Mean gate firing (%):

| Route | Normal | Repeat episodes |
|---|---:|---:|
| gate / gate_r | 0.0084 | 100 |
| gate_rd | 0.0084 | 70.02 |

Mean store size; peak equals final:

| Route | Exemplars | Floats |
|---|---:|---:|
| exem / gate_r | 10,000 | 7,850,000 |
| gate_rd | 8,372.67 | 6,572,543.33 |

Oddity: `ffr` lifetime differs from archived H24a by **+0.59 / +0.10 / −0.27 points**. The other four baseline routes match exactly; the discrepancy is recorded in the report.