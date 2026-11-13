# H30y: a delay line is a bank on the roots of unity. Bank initialised there, trace-learned (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/h30x/RESULTS.md` (including its Verdict), `h30x_trace.py` and
`h30w_ring.py`.

## Context
In H30x the trace gradient was exact, yet the bank did not learn echo D = 128. After learning, its mean |λ| was 0.93,
a memory of about 14 steps. A pure transport ring of N units is diagonalised by the discrete Fourier transform, with
eigenvalues e^(2πik/N): a delay line is a diagonal bank whose modes sit evenly on the unit circle. This follows from our own
picture (carriage is diagonal in frequency, `docs/compute-vs-data-movement.md`). The H30w/H30x init (|λ| ∈ [0.9, 0.999],
θ ∈ [0, π]) is far from that point. H30y starts the bank there. Prediction: lags below N = 256 become recallable
(D = 32 and 128), while D = 512 aliases with lag 256 and does not. Principle served: improve the compressor (2), with a
local substrate and a local learning rule.

## Arms (all use the trace rule from `h30x_trace.py`, unchanged; import it, do not edit it)
- `trace_rand`: H30x's trace arm as it is (same init). This is the reference.
- `roots_learn`: the same bank, but initialised with θ_k = 2πk/256 for k = 0, …, 255 and |λ_k| = 0.98
  (ν_k = log(−log 0.98)). λ is learned. (Amended 13 Nov: v1 used 0.999. A bank on the roots of unity holds lags mod
  256, so the inverse DFT returns s_{t−d} + |λ|^256 s_{t−d−256} + …. At 0.999 that alias weight is 0.774, which
  makes the memory useless. At 0.98 it is 0.006.)
- `roots_frozen`: the same init as `roots_learn`, but ν and θ are frozen. B, the readout and the embedding learn.
Everything else is as in H30x: N = 256, embedding 32, order-2 readout, Adam lr 1e-3, clip 1.0, chunks of 64, score
before step, float32. The only differences are the init of ν and θ and, for one arm, freezing them.

## Streams
As H30x: echo D ∈ {32, 128, 512}, n = 131,072, seeds 0, 1 and 2, and H20 slow with world seeds 6, 7 and 8, n = 262,144. Model seeds
are as in H30x.

## Smoke test (VPS CPU, before any kernel)
1. `trace_rand` on the first 4,096 symbols of echo D = 32 seed 0 matches H30x's trace arm to 1e-6.
2. **Delay check (this check only):** with the roots init and λ frozen, set B = 1 · cᵀ for a fixed c, so every mode
   receives the same scalar s_t = cᵀ e(x_t). Feed 600 random symbols and reconstruct s_{t−d} from h_t by the inverse
   DFT (weights ∝ λ_k^(−d)/N), for d = 1, 32, 128, 255 and 256, over all t ≥ d. Print the relative errors. Prediction
   (amended): relative error ≤ 0.02 for d ≤ 255 (the alias weight is 0.98^256 ≈ 0.006), and lag 256 aliases with
   lag 0. v1 stopped here, correctly, at |λ| = 0.999. This check only confirms the theory; nothing in the real
   runs is hand-set.
3. All three arms give finite bits on 4,096 symbols of every stream. For both roots arms, print max |λ| at init.
4. Time 8,192 symbols per arm, and estimate each kernel's wall time. If one would exceed 90 minutes, split it, or stop
   and report. Do not shrink anything.
If any check fails, stop and report. Do not push kernels.

## Kernels
Three private CPU kernels, `ezzzzzx/h30y-s0`, `-s1` and `-s2`. Kernel i runs echo seed i at all D and H20 world seed
6 + i, for all three arms. Each has `"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. The code goes up as private
dataset `ezzzzzx/pdp-code-h30y`.

## Outputs
`results/h30y/RESULTS.md` and `results/h30y/results.json`, with numbers only and no verdict:
- echo: fresh and copy bits (whole stream and last quarter) per arm, D and seed, with mean ± std (ddof = 0);
- H20: bits per arm, seed and process, and C1 + C7 combined;
- the gaps: roots_learn − trace_rand and roots_frozen − trace_rand, on echo copies at each D and on H20 C1 + C7;
- for roots_learn, the end-of-run |λ| histogram (10 bins on [0, 1]), n(|λ| > 0.99), and the mean absolute change in
  θ from init;
- the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
1. **The roots hold delays:** on echo D = 128, for the better of roots_learn and roots_frozen (chosen by mean copy
   bits), mean copy bits ≤ 3.0 (uniform is 4), and lower than trace_rand by ≥ 1.0 on all 3 seeds.
2. **Counters survive:** on H20 C1 + C7, mean over seeds of (roots_learn − trace_rand) ≤ +0.10.
Report only: D = 32, D = 512 (the aliasing prediction), the fresh halves, H20 totals and roots_frozen on H20.
Kill rules: if bar 1 fails, the delay-line point is not reachable from a learned readout and B, even when the bank
starts on it, and our Fourier picture of carriage does not carry over to a learned memory. If bar 2 fails, a delay
line and a counter need different banks, and a cortex needs both kinds of unit.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create or update: `h30y_roots.py` (it exists from v1; change only the init), `kaggle/h30y/**` and `results/h30y/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h30x_trace.py`, `h30w_ring.py`, `h20_world.py`, other experiment code,
  other `kaggle/` or `results/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
