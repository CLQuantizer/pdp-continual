I'll read the project instructions and the handoff, then update `h30y_roots.py` to the amended spec.The handoff changes the roots-of-unity init from `|λ| = 0.999` to `0.98`. I'll read the H30x result, the trace learner, and the existing v1 script before touching the init.v1 stopped because `|λ| = 0.999` leaves an alias weight of about 0.77. I'll set the roots init to `|λ| = 0.98` and align the delay-check bar with the amended limit of 0.02.The init is `|λ| = 0.98`. Next I’ll run the smoke checks on the VPS before any kernel.Smoke passed. The kernel estimate is 1421 seconds, under the 90-minute cap. I'll package the private CPU kernels next.All three kernels are running. I'll poll them until they finish.H30y is measured. The roots init is now `|λ| = 0.98`. Numbers are in `results/h30y/RESULTS.md` and `results/h30y/results.json`. Nothing was committed.

Smoke, before the kernels:

- `trace_rand` matched the H30x trace arm on the first 4,096 symbols of echo D = 32, seed 0. The maximum absolute bits difference was 0.
- With the roots init and λ frozen, the inverse-DFT relative error was 4.28e-3, 4.17e-3, 3.70e-3 and 2.86e-3 at lags 1, 32, 128 and 255. Lag 256 matched a scaled lag 0 to 4.91e-6. The maximum `|λ|` at init was 0.98000008. The alias weight `0.98^256` was 0.005674.
- All three arms were finite on 4,096 symbols of every stream. The kernel estimate was 1,421 seconds.

Kernels `ezzzzzx/h30y-s0`, `-s1` and `-s2` finished in 992 s, 1,009 s and 968 s. The code dataset is `ezzzzzx/pdp-code-h30y`.

Echo copy bits, population standard deviation, ddof = 0:

| D | roots_learn | roots_frozen | trace_rand | learn − trace | frozen − trace |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.969558 ± 0.026299 | 1.777251 ± 0.035015 | 1.589833 ± 0.144076 | −0.620275 ± 0.129594 | +0.187418 ± 0.117756 |
| 128 | 4.405203 ± 0.027960 | 4.405125 ± 0.034501 | 4.274956 ± 0.016089 | +0.130247 ± 0.013299 | +0.130169 ± 0.019568 |
| 512 | 3.626910 ± 0.004289 | 3.657269 ± 0.009736 | 3.783280 ± 0.035196 | −0.156370 ± 0.030990 | −0.126012 ± 0.028624 |

D = 128 copy gaps by seed, learn − trace: +0.144387, +0.133918, +0.112437. Frozen − trace: +0.148723, +0.138674, +0.103111.

H20 totals: trace_rand 0.631740 ± 0.027936, roots_learn 0.612351 ± 0.006774, roots_frozen 0.659464 ± 0.004123. C1+C7: trace_rand 0.263566 ± 0.075677, roots_learn 0.253394 ± 0.070698, roots_frozen 0.309563 ± 0.085485. roots_learn − trace_rand on C1+C7: −0.010171 ± 0.022236. roots_frozen − trace_rand: +0.045998 ± 0.019067.

For `roots_learn`, every end-of-run `|λ|` is in `[0.9, 1.0]`. The count with `|λ| > 0.99` is 0 on H20 and on echo D = 128 and D = 512, and 68, 59 and 70 on echo D = 32. The mean absolute change in θ from the roots init is between 0.035 and 0.047.
