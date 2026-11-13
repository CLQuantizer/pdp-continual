I'll read the project instructions and the handoff note first, then follow that handoff exactly.The handoff is H30y: start the delay bank on the roots of unity and score it with the H30x trace rule. I'll read the H30x results and the two source files before writing anything.H30x scored the same bank with a random init. I'll read the trace learner, the ring bank, and the H30x kernel layout so the roots init is the only change.The delay check is the gate that can stop the run. I'll measure the roots-of-unity reconstruction on 600 symbols before writing the kernels.The full-stream reconstruction error is large because a previous cycle still leaks in at about 0.77. I'll confirm the freeze path, then run the spec's smoke checks in order.Adam still steps a frozen parameter if its gradient is set, so the frozen arm has to keep ν and θ out of the update. I'll write the experiment and run the smoke checks in the spec's order.Stopped before any kernel. The delay check does not show small errors for lags below 256 on the 600-symbol stream, so the kernels were not pushed. No commit.

Check 1 passed. `trace_rand` on the first 4,096 symbols of echo D = 32, seed 0, matches the H30x trace arm with max absolute bits difference 0. Both score 11.69228047.

Check 2 used the roots init, λ frozen, and B = 1 · cᵀ. Every mode received the same scalar. Relative L2 error of the inverse-DFT reconstruction of s_{t−d} over all t ≥ d:

| d | relative error | first period only |
|---:|---:|---:|
| 1 | 0.641 | 4.03e-6 |
| 32 | 0.619 | 4.05e-6 |
| 128 | 0.504 | 5.54e-6 |
| 255 | 0.380 | 1.39e-5 |
| 256 | 1.366 | — |

Lag 256 matches a scaled lag 0 to 1.28e-5. Max |λ| at this init is 0.99900006. The mode spread of the injection is 0.

The update h_t = λ ⊙ h_{t−1} + s_t keeps every earlier visit to the same lag. The reconstruction of lag d is s_{t−d} + 0.999^256 s_{t−d−256} + 0.999^512 s_{t−d−512} + …, and 0.999^256 = 0.774. Over 600 symbols that leftover is an O(1) error. On the first period, before a previous cycle exists, the error is about 1e-5.

Smoke log and the same numbers are in `results/h30y/RESULTS.md` and `results/h30y/results.json`. The training code is in `h30y_roots.py`. Scored echo and H20 runs were not started.
