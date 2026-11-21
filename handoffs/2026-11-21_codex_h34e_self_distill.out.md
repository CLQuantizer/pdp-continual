Implemented H34e and prepared all eight private CPU kernels.

All smokes passed: exact baseline/λ=0 equality on both seeds, gradient error `4.44e-16`, identical prefixes, finite scores. Estimated walls: 3.0–20.6 minutes.

[Results](/root/pdp/results/h34e/RESULTS.md) show `NA` pending Grok’s runs. Launch command:

```sh
.venv/bin/python -B kaggle/h34e/dispatch.py --dispatch
```

The inherited `distill_1` mixer learns a scalar temperature; that caveat is documented in the [README](/root/pdp/kaggle/h34e/README.md). No kernels launched or commits made.