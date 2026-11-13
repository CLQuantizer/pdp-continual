# H30x: the diagonal bank trained by exact eligibility traces (online, local in time) against BPTT-64 (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/h30w/RESULTS.md` (including its Verdict) and `h30w_ring.py`.

## Context
In H30w, a diagonal bank of damped complex rotations beat an equal-size GRU on H20's counters by 1.6 bits/symbol. But
no model learned echo D = 128. Training was BPTT inside 64-symbol chunks with the state carried forward detached, so a
dependency 128 steps back never gets a gradient. In a diagonal recurrence the exact gradient through all of time is
local: each unit keeps an eligibility trace, updated forward with its own λ (exact RTRL for diagonal recurrences;
Zucchet et al. 2023, "Online learning of long-range dependencies"). That needs no unrolling and no backward pass
through time. This is the machine the north star asks for: it learns while it acts, in one forward sweep. Principle
served: improve the compressor (2), with a local learning rule. This run changes the substrate's learning rule.

## Model (unchanged from H30w)
The `bank` from `h30w_ring.py`: N = 256 complex units, h_t = λ ⊙ h_{t−1} + B x_t, λ = exp(−exp(ν) + iθ), with the same
init, embedding 32 and order-2 readout (W, P₁, P₂, V). Import it; do not edit `h30w_ring.py`.

## Two learning rules (the only difference between the arms)
- **bptt64:** exactly H30w's protocol (score the 64-chunk, one Adam step on its mean cross-entropy, carry state detached).
- **trace:** same chunks, same Adam (lr 1e-3, clip 1.0), one step per chunk, same score-before-step order. The gradient
  for the step is built online, without backprop through time:
  - Readout parameters (W, P₁, P₂, V, embedding as it enters B x_t): their gradient at step t uses only h_t and x_t
    (autograd on the step-t readout alone is fine). Sum over the chunk, divide by the chunk length.
  - Recurrent parameters: keep complex traces e^λ_i (per unit) and E^B_ij (256 × 32), carried across chunks and
    **never reset or detached by chunk**:
    e^λ_{i,t} = λ_i e^λ_{i,t−1} + h_{i,t−1},  E^B_{ij,t} = λ_i E^B_{ij,t−1} + x_{j,t}.
    With δ_t = ∂L_t/∂h_t (from the step-t readout only), the gradient contribution is the real part of the
    appropriate Wirtinger product of δ_t with the trace, chained to ν and θ through λ. Get the complex calculus right;
    smoke check 2 below verifies it.
  - Traces use the current λ. After an Adam step they are not recomputed (standard online approximation).
  - The embedding's gradient through B also uses E^B (x_t is the embedding of the previous symbol); the trace for
    the embedding row is the corresponding E^B contribution. If this is awkward, freeze the embedding in **both**
    arms and report it. Do not freeze it in one arm only.

## Streams (all prequential, as in H30w)
1. **Echo** (H30w's generator): alphabet 16, D ∈ {32, 128, 512}, n = 131,072, seeds 0, 1, 2 (model seed = stream
   seed). Report fresh and copy halves separately. Uniform is 4 bits on both halves.
2. **H20 slow:** `h20_world.stream("slow", seed, n)`, world seeds 6, 7, 8 (model seeds 0, 1, 2), n = 262,144. Report
   bits per process and C1 + C7 combined.

## Smoke test (VPS CPU, before any kernel)
1. **bptt64 reproduces H30w:** the bank under bptt64 on the first 4,096 symbols of echo D = 32 seed 0 matches
   `h30w_ring.py`'s bank bits to 1e-6.
2. **Trace gradient is exact:** with parameters frozen (no Adam step), take a 300-symbol echo D = 128 sequence. Compute
   the gradient of the summed loss over all 300 steps by full BPTT (no detaching), and by the trace rule. For ν, θ, B and
   the readout, max relative error ≤ 1e-4 (float64 for this check). Print both norms and the error.
3. Both arms give finite bits on 4,096 symbols of every stream.
4. Time 8,192 symbols per arm, and estimate each kernel's wall time. If any kernel would exceed 90 minutes, stop and
   report. Do not shrink anything.
If any check fails, stop and report. Do not push kernels.

## Kernels
Use three private CPU kernels, `ezzzzzx/h30x-s0`, `-s1` and `-s2`. Kernel i runs echo seed i at all three D and H20
world seed 6 + i, for both arms. Each has `"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`.
The code goes up as private dataset `ezzzzzx/pdp-code-h30x`.

## Outputs
`results/h30x/RESULTS.md` and `results/h30x/results.json`, with numbers only and no verdict:
- echo: fresh and copy bits per arm, D and seed, with mean ± std (ddof = 0); copy bits over the last quarter of the
  stream as well;
- H20: bits per arm, seed and process; C1 + C7 combined; mean ± std;
- the gaps trace − bptt64 on echo copies (each D) and on H20 C1 + C7;
- the learned |λ| at the end (10-bin histogram over [0, 1]) and the count of units with |λ| > 0.99, per arm and run;
- whether the embedding was frozen; wall seconds and kernel ids; the smoke-test output.

## Pre-registered bars (orchestrator scores; do not change)
1. **Traces reach past the window:** on echo D = 128, the trace arm's mean copy bits ≤ 3.0 (uniform is 4), and trace
   is lower than bptt64 by ≥ 1.0 on all 3 seeds.
2. **Local learning keeps the counters:** on H20 C1 + C7, mean over seeds of (trace − bptt64) ≤ +0.10.
Report only: D = 32, D = 512, the fresh halves, H20 totals.
Kill rules: if bar 1 fails, traces in a diagonal bank do not carry credit over long delays in practice, which is
a sign that stale traces or the fresh-half cost dominate. If bar 2 fails, the local rule costs us what the bank gained over the GRU.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `h30x_trace.py`, `kaggle/h30x/**` and `results/h30x/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h30w_ring.py`, `h20_world.py`, other experiment code, other `kaggle/` or
  `results/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
