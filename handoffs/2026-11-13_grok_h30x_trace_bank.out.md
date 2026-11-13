I'll read the project instructions and the handoff note first, then follow that handoff exactly.The handoff is H30x: train the same diagonal bank with eligibility traces and compare it to BPTT-64. I'll read the H30w results and model next, then implement only the files the spec allows.The gradient check is the part that can fail the smoke test, so I'll pin the complex trace math against full BPTT before writing the training script.The trace gradients match full BPTT in float64, including the embedding. Next I’ll write the training script and run the smoke checks before any kernel.The gradient check and the BPTT match passed. The H20 smoke log crashed because the first 4,096 symbols contain no C1 or C7, so that combined rate was empty. I'll print that case and rerun the smoke test.Smoke checks passed and the kernel estimate is about 12 minutes, under the 90-minute cap. The three private CPU kernels are running; I'll poll them until they finish.H30x is done. Measurements are in `results/h30x/RESULTS.md` and `results/h30x/results.json`. No verdict. Nothing was committed.

The bank is unchanged. The embedding was trained in both arms. `bptt64` is `h30w_ring.train`. The trace arm keeps complex eligibility traces across chunks and steps once per chunk of 64.

Smoke, before any kernel:

- Trace versus full BPTT on 300 echo symbols, float64: max relative error `9.3e-16` on ν, θ, B, and the readout. The embedding matched as well.
- `bptt64` versus H30w on the first 4,096 symbols of echo D = 32, seed 0: max absolute bits difference `0`.
- Both arms were finite on 4,096 symbols of every stream. The kernel estimate was 745 s, under the 90-minute cap.

Kernels `ezzzzzx/h30x-s0`, `-s1`, and `-s2` finished in 483 s, 464 s, and 484 s. Code dataset: `ezzzzzx/pdp-code-h30x`.

Echo copy bits, trace − bptt64 (population std, ddof = 0):

| D | trace copy | bptt64 copy | trace − bptt64 |
|---:|---:|---:|---:|
| 32 | 1.589833 ± 0.144076 | 1.792200 ± 0.110562 | −0.202367 ± 0.035690 |
| 128 | 4.274956 ± 0.016089 | 4.410850 ± 0.068103 | −0.135895 ± 0.060994 |
| 512 | 3.783280 ± 0.035196 | 3.804233 ± 0.026939 | −0.020953 ± 0.009720 |

D = 128 copy gaps by seed: −0.221525, −0.102082, −0.084078. Last-quarter copy bits at D = 128: trace 3.907076 ± 0.026213, bptt64 4.026283 ± 0.001272.

H20 totals: bptt64 0.625387 ± 0.016476, trace 0.631740 ± 0.027936. C1+C7: bptt64 0.262897 ± 0.069503, trace 0.263566 ± 0.075677. trace − bptt64 on C1+C7: +0.000669 ± 0.007329.

The bptt64 tables match the H30w bank at the printed precision. End-of-run |λ| histograms and the count of units with |λ| > 0.99 are in the results file.
