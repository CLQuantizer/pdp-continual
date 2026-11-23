Implemented; smoke passed with L=4,096.

- M10 baseline reproduced exactly.
- All four arms finite; eyelid checks: 1,043/1,043.
- Seed 120: sleep 25.46%; night share 71.05%; daytime noise-room share 43.71%.
- 12 kernels staged; estimated wall 38.73 minutes.

[RESULTS.md](/root/pdp/results/w10/RESULTS.md) · [results.json](/root/pdp/results/w10/results.json)

All 48 scored lives remain pending Grok. Run:

```bash
.venv/bin/python -B kaggle/w10/dispatch.py
```