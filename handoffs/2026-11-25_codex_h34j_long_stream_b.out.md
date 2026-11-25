Completed offline staging, preserving prior checks.

- All 8 reference comparisons and 8 split checks pass exactly.
- 48 private CPU kernels staged: 1/7/7/9 parts per seed.
- Dispatcher restart, caps, chaining, retry, and summarizer checks pass.
- [Reports](/root/pdp/results/h34j/RESULTS.md) generated; scored results remain `NA`.

Orchestrator launch:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python kaggle/h34j/dispatch.py --dispatch
```

No Kaggle runs launched; no commits made.