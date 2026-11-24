Implemented and staged 30 private CPU kernels offline. All reference, causality, checkpoint, and template checks passed. Maximum estimated run: 70 minutes; no splits needed.

Orchestrator launch:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34i/dispatch.py --dispatch
```

[RESULTS.md](/root/pdp/results/h34i/RESULTS.md) and [results.json](/root/pdp/results/h34i/results.json) contain enwik8 references and pending `NA` scores. No kernels launched; no commit.