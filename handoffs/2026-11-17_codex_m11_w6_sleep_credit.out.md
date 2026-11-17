Implemented [m11_w6.py](/root/pdp/m11_w6.py) and staged 24 private CPU kernels: two per seed.

Smoke passed: both baselines matched exactly, shadow error was zero, and all arms were finite. The L=1024 counterfactual life took 172.0 seconds; the conservative unsplit estimate was 91.8 minutes, triggering the split.

[Results and smoke output](/root/pdp/results/m11w6/RESULTS.md). Grok can run:

```sh
.venv/bin/python kaggle/m11w6/dispatch.py
```

No kernels pushed; no commits made.