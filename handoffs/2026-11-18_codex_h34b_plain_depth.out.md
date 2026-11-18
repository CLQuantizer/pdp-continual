Implemented the runner, 14 kernel folders, resumable dispatcher, and summarizer.

Smoke found two plain_6 failures:

- First 2,048 bytes: **10.924 bits/byte**, above 9.
- Estimated wall at the required **200 KB fallback**: **87.1 minutes**, above 85.

Equivalence, fixed weights, prefix causality, and wide_6 checks passed. Dispatch remains disabled. No network or commits.

[RESULTS.md](/root/pdp/results/h34b/RESULTS.md)