I'll start by reading the project instructions and the handoff so I follow the amended spec exactly.The handoff is a scored H30w comparison of a frequency bank, a propagating ring, and a GRU. I'll read the cited docs and the existing script before changing anything.The previous ring was unstable because `|a| + |b|` could exceed 1. I'll update the script to the amended stable ring and the GRU hidden size of 166, then run the smoke checks before any kernel.Stability holds: at every seed, max `|a| + |b|` is about 0.999. Next I’ll run the full VPS smoke test, including the 8,192-symbol timing.Smoke passed. One kernel is estimated at about 24 minutes, under the 90-minute limit. I'll package the code and push the three private CPU kernels.All three kernels finished. I'll download their outputs and check that each run actually completed.H30w is scored. Numbers are in `results/h30w/RESULTS.md` and `results/h30w/results.json`. No verdict and no commit.

The ring now uses ρ = exp(−exp(ν)), g = sigmoid(γ), a = ρ g exp(iθ), b = ρ (1−g) exp(iφ), so |a| + |b| = ρ < 1. The GRU hidden size is 166. At alphabet 41 the parameter counts are GRU 107,759, bank 107,360, and ring 107,872.

Smoke passed on the VPS. Every model produced finite bits on 4,096 symbols. The hand-set transport ring held the input from k steps earlier at k = 0, 1, 5, and 200. The prequential check was unchanged through the scored chunk. At init, max |a| + |b| was below 1 for model seeds 0, 1, and 2 (largest value 0.998919). The 8,192-symbol timing estimated one kernel at 1,458 seconds, under the 90-minute limit.

Three private CPU kernels finished: `ezzzzzx/h30w-s0`, `ezzzzzx/h30w-s1`, and `ezzzzzx/h30w-s2`. Wall times were 481 seconds, 453 seconds, and 458 seconds.

H20 slow, bits/symbol, mean ± std over seeds 6, 7, and 8:

| Model | Total | C1+C7 |
|---|---:|---:|
| gru | 1.176566 ± 0.078233 | 1.871659 ± 0.158646 |
| bank | 0.625387 ± 0.016476 | 0.262897 ± 0.069503 |
| ring | 0.709337 ± 0.045911 | 0.489531 ± 0.083619 |

On C1+C7, ring − bank is 0.226634 ± 0.042261. min(bank, ring) − gru is −1.608763 ± 0.222078. The bank is the lower of bank and ring on each seed. World seed 6 has no M7 symbols, so the M7 mean uses seeds 7 and 8.

Echo copy halves, bits/symbol:

| D | Model | Copy, mean ± std |
|---|---|---:|
| 32 | gru | 4.007769 ± 0.000150 |
| 32 | bank | 1.792200 ± 0.110562 |
| 32 | ring | 4.148535 ± 0.006195 |
| 128 | gru | 4.003786 ± 0.000210 |
| 128 | bank | 4.410850 ± 0.068103 |
| 128 | ring | 4.074336 ± 0.003428 |

On D = 128 copies, ring − bank is −0.336514 ± 0.066063, and the ring is lower than the bank on seeds 0, 1, and 2 (−0.425653, −0.316176, −0.267713). min(bank, ring) − gru on those copies is 0.070551 ± 0.003276. On D = 32 copies, ring − bank is 2.356335 ± 0.106218.

Learned ring |a| and |b| sit mostly in [0.2, 0.8). The 10-bin histograms are in `RESULTS.md`.
