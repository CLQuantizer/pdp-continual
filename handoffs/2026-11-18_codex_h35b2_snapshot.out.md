Implemented [h35b2_snapshot.py](/root/pdp/h35b2_snapshot.py), six private offline CPU kernels, a resumable dispatcher, and the summary script.

Smoke passed:

- `rand` and `oracle` reproduce H35b exactly; both deltas are 0.
- Frozen-copy checksums and exact snapshot lags pass.
- All five arms run at B=8.
- Estimated wall: 122 minutes per seed kernel.

[Results and smoke details](/root/pdp/results/h35b2/RESULTS.md). No network calls, submissions, or commits; original experiment code is unchanged.