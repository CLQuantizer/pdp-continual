# H30 / H30w: a frequency-native cortex, as a diagonal bank or a propagating ring, against the GRU (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `docs/compute-vs-data-movement.md` (the last three sections), `h20_world.py`
and `h21a_substrate.py` (the online prequential training loop and the per-process report).

## Context
Our theory says that carrying a signal, whether through space or time, is diagonal in frequency, and that computing is
the coupling between frequencies. Two cortices follow from that:
- **bank**: a diagonal bank of complex units (damped rotations; a negative real eigenvalue is a sign flip, which counts
  mod 2), plus an order-2 multiplicative readout that does the coupling;
- **ring**: the same units on a ring, each also fed by its neighbour, so a signal can *propagate*. A pure transport
  ring is an exact delay line.
Neither has run. The GRU could never learn a counter's carry or look one CA row back (H20, H21a). Principle served:
improve the compressor (2), with a substrate that is local and diagonal (design criteria 1, 3 and 5).

## Streams (all prequential, next-symbol prediction)
1. **H20 slow world.** `h20_world.stream("slow", seed, n)` (season 16,384; it returns the text and the per-symbol process labels), world seeds 6, 7 and 8,
   n = 262,144 symbols, 41-symbol alphabet. Report bits per process (G, C1, C7, E30, E110, M7, M37).
2. **Echo stream.** Alphabet 16. Repeat: draw D symbols uniformly (`default_rng(100 + seed)`), emit them, then emit
   the same D symbols again. D ∈ {32, 128}, n = 131,072 symbols, seeds 0, 1 and 2. The floor is 2 bits/symbol: 4 on
   the fresh halves and 0 on the copies. Report bits on fresh and copy halves separately.

## Models (embedding 32 for every model)
- **gru:** `torch.nn.GRU`, hidden 128, linear readout to the alphabet.
- **bank:** N = 256 complex units. h_t = λ ⊙ h_{t−1} + B x_t, with λ = exp(−exp(ν) + iθ) as in the LRU (Orvieto et al.
  2023). Initialise |λ| uniformly in [0.9, 0.999] and θ uniformly in [0, π], so negative real eigenvalues are
  reachable. B is complex, 256 × 32. Readout: z = [Re h, Im h] (512). Logits = W z + V((P₁ z) ⊙ (P₂ z)), where P₁ and
  P₂ map 512 → 64 and V maps 64 → alphabet. That is the learned order-2 term.
- **ring:** the same 256 complex units on a ring. h_{i,t} = a_i h_{i,t−1} + b_i h_{i−1,t−1} + (B x_t)_i, with unit 0
  reading unit 255. a_i is parametrised and initialised like λ. b_i = exp(−exp(μ_i) + iφ_i), initialised with |b|
  uniform in [0.3, 1.0) and φ uniform in [0, π]. The readout is identical to the bank's. Use no hand-set delay line:
  learning has to find the transport.
Report each model's parameter count. Do not tune anything per model. Every model uses Adam with lr 1e-3, gradient
clip 1.0 and float32.

## Online prequential protocol (identical for every model)
Process the stream in chunks of 64 symbols. For each chunk:
1. Score it with the current parameters, starting from the carried state. Those −log2 p are the bits.
2. Take one Adam step on the mean cross-entropy of the same chunk (BPTT within the chunk).
3. Carry the state forward, detached.
Use no replay and no store, so that only the substrate differs. Model init seeds are 0 (H20 world seed 6 uses model
seed 0, 7 uses 1 and 8 uses 2; echo seeds use the same model seed as their stream seed).

## Smoke test (VPS CPU, before any kernel)
1. On 4,096 symbols of each stream, every model gives finite bits. Print the parameter counts.
2. **Ring check:** with a hand-set transport (a_i = 0, b_i = 1, B injecting only at unit 0, real), feed a sequence and
   check that unit k holds the input from k steps earlier. Print it for k = 0, 1, 5 and 200. This is a check only;
   the real runs learn from the init above.
3. **Prequential check:** the bits of chunk c do not change if the stream after chunk c is altered (test this on one
   model).
4. Time 8,192 symbols per model, and estimate each kernel's wall time. If any kernel would exceed 90 minutes, stop
   and report. Do not shrink the models or the streams yourself.
If any check fails, stop and report. Do not push kernels.

## Kernels
Use three private CPU kernels, one per seed index i ∈ {0, 1, 2}: `ezzzzzx/h30w-s0`, `-s1` and `-s2`. Kernel i runs H20
world seed 6 + i and echo seed i at both D, for all three models. Each has `"is_private": true`, `"enable_gpu": false`
and `"enable_internet": false`. The code goes up as private dataset `ezzzzzx/pdp-code-h30w`.

## Outputs
`results/h30w/RESULTS.md` and `results/h30w/results.json`, with numbers only and no verdict:
- H20: bits/symbol per model, per seed and per process, with the mean ± std (ddof = 0) over seeds;
- echo: bits/symbol on the fresh halves and on the copy halves, per model, D and seed, with the mean ± std;
- the gaps ring − bank and min(bank, ring) − gru on copy halves (echo), and on C1 + C7 combined (H20);
- for the ring, the learned |b_i| and |a_i| (histograms as 10-bin counts) at the end of each run;
- parameter counts, wall seconds and kernel ids;
- the smoke-test output.

## Pre-registered bars (orchestrator scores; do not change)
1. **Propagation holds exact delays:** on echo D = 128, mean over seeds of (ring − bank) on copy halves ≤ −0.25
   bits/symbol, with the ring better on all 3 seeds.
2. **The frequency-native cortex counts:** on H20, mean over seeds of (min(bank, ring) − gru) on C1 + C7 combined
   ≤ −0.10 bits/symbol.
Report only: totals, the CA processes and echo D = 32.
Kill rules: if bar 1 fails, propagation adds nothing over a frequency bank, and H30w is parked. If bar 2 fails, the
H30 line on counters closes.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `h30w_ring.py`, `kaggle/h30w/**` and `results/h30w/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h20_world.py`, `h21a_substrate.py`, other experiment code, other `kaggle/`
  or `results/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
