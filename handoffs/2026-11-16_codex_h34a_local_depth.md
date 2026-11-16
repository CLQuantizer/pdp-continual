# H34a: does depth pay under local learning? Plain local stacks vs the leftover-error stack (codex, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then the Verdict in `results/h33/RESULTS.md` and `results/h30x/RESULTS.md`, and the code in
`h33_bank_text.py`, `h30x_trace.py`, `h30y_roots.py`, `h30w_ring.py` (class `Bank`) and `h32b_mixer.py` (class `GeoMixer`).

## Context
H33: one trace-learned diagonal bank on text adds 0.016 bits/byte over count models, where the backprop GRU adds 0.26. The bank
has memory in time (exact, local eligibility traces) but no nonlinear depth. State-space models (S4, Mamba, LRU) get their
power from stacking linear diagonal recurrences with nonlinearity between layers, trained by backprop across layers.
Here we stack banks **with no gradient crossing a layer boundary** and test which local learning signal lets depth pay.

The idea under test ("leftover error"): every layer has its own readout, a geometric mixer combines the layers' predictions,
and every layer learns from the mixer's leftover error `p_mix − onehot(y)`, one vector broadcast to all layers. For a
geometric mixer this is, up to the layer's mixer weight, the exact gradient of the mix loss on that layer's logits, so each
layer learns only what the stack still gets wrong. The control ("plain") trains every layer on its own loss `p_k − onehot(y)`.
Readouts start at zero, so a new layer predicts uniform and changes nothing in a geometric mix (identity at init).
Principle served: improve the compressor (2) with a learner that can live (no backward pass, online).

## Data
The first **500,000 bytes** of enwik8 (`h26_cued.load_enwik8`; check the E10M sha256 on the first 10 M as `h33_bank_text`
does, then take bytes 0–499,999). Alphabet 256. Every byte is scored by every arm before anything learns from it
(prequential). Chunk 128: score the chunk, then take one learning step per layer and one mixer step.

## The layer (same for every arm)
Layer k is `h30y_roots`' roots_learn bank (as used in H33), learned by the `h30x_trace` rule, with these changes only:
- **N = 128 units**, θ_j = 2πj/128, |λ| = 0.98 at init. Order-2 readout W, P1, P2, V (RANK 64, no bias) as in `h30w_ring.Bank`.
- **Zero-init readout:** W and V start at exactly 0 (P1, P2 keep their usual init), so every layer starts uniform.
- **Input:** layer 1 gets its own learned byte embedding (EMB 32). Layer k ≥ 2 gets `u_k = [emb_k(x_t) ; f_{k−1,t}]`, where
  `emb_k` is layer k's own learned embedding (EMB 32) and `f_{k−1,t} = [z_{k−1,t} ; quad_{k−1,t}]` (z = [Re h ; Im h], 256
  dims; quad = P1 z ⊙ P2 z, 64 dims) of layer k−1 at the same time step, **detached**, and divided per feature by its
  running RMS (EMA of the square with decay 0.999, initial value 1, updated after the chunk is scored). The input map B_k
  is complex N × dim(u_k), learned by the trace rule exactly as the embedding injection is in `h30x_trace` (the trace e_b
  is N × dim(u_k)).
- **Learning signal:** the arm's error vector g_k at layer k's logits (below). Readout gradients are g_k times the readout's
  inputs (autograd inside the layer is fine). ∂L/∂h_{k,t} comes from back through layer k's own readout only, then the
  `h30x_trace` rule gives ν, θ, B_k and emb_k their gradients. Nothing reaches layer k−1. Adam lr 1e-3, clip 1.0, float32.
  State and traces carry across the whole stream.

## The mixer (same for every arm)
`h32b_mixer.GeoMixer` over the D layers' distributions with a **single weight set** (no context selection), init weight 1.0
on every layer, Adam lr 0.003, one step per chunk on its own loss −log p_mix(y). The arm's score is the mixer's bits.

## Arms (one prequential pass each; D = number of layers)
- `plain_D`, D ∈ {1, 2, 4}: g_k = p_k − onehot(y), each layer's own loss.
- `left_D`, D ∈ {1, 2, 4}: g_k = p_mix − onehot(y) (p_mix detached; **not** scaled by the mixer weight).
- `left_4_noskip`: as `left_4`, but layers k ≥ 2 get `u_k = f_{k−1,t}` only (no byte embedding).
- `wide_1`: D = 1 with **N = 512** (θ_j = 2πj/512), as the matched-compute width control for D = 4.
Model seeds s ∈ {0, 1}. 8 arms × 2 seeds = 16 lives.

## Compute accounting
For each arm: parameters, multiply-adds per byte (forward plus learning, counted analytically from the shapes, formula shown),
and trace floats held. Flag in RESULTS.md if `wide_1`'s multiply-adds differ from `left_4`'s by more than 20%.

## Kernels
One private CPU kernel per (arm, seed): `ezzzzzx/h34a-<arm>-s<seed>` (underscores → hyphens). Each has `"is_private": true`,
`"enable_gpu": false`, `"enable_internet": false`. enwik8 from the Kaggle dataset H33 used. Code as private dataset
`ezzzzzx/pdp-code-h34a`. At most 90 minutes per kernel. **Run at most 3 kernels at once** (another worker uses Kaggle
today; Kaggle allows 5 CPU sessions). If the smoke timing says the slowest arm will not fit in 90 minutes on 500,000
bytes, use the first **300,000** bytes for every arm instead (pre-registered fallback) and say so; do not change N or D.

## Smoke test (VPS CPU, `torch.set_num_threads(1)`, before any kernel)
1. **Layer 1 is H33's bank:** with N = 256, a non-zero readout init, the mixer bypassed and g = p − onehot, this code's layer
   reproduces `h33_bank_text`'s bank bits on the first 4,096 bytes to 1e-6.
2. **Exact traces still:** for a 2-layer stack on 512 bytes with layer 2's input including f_1, the trace gradients of layer 2's
   ν, θ, B_2 match BPTT gradients of the same per-layer loss with f_1 detached (as `h30x_trace.gradient_check` does), max
   relative error ≤ 1e-4.
3. **Identity at init:** a freshly added layer gives exactly uniform logits, and adding it leaves p_mix unchanged to 1e-9 on
   the first chunk.
4. **Plain ≡ leftover at D = 1:** with the mixer weight frozen at 1.0, `plain_1` and `left_1` give identical bits on 8,192 bytes.
5. **Prequential:** the first 1,024 bytes' bits are identical whether the stream is 1,024 or 2,048 bytes long, for `left_4`.
6. All arms finite over 8,192 bytes. Time 32,768 bytes for `left_4` and `wide_1` and estimate kernel wall times.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/h34a/RESULTS.md` and `results/h34a/results.json`, numbers only, no verdict:
- bits/byte per arm and seed, and the mean over seeds; per 50 KB slice per arm and seed;
- per layer: the layer's own bits (p_k alone) and its mean mixer weight over the last 50 KB;
- the gaps (per seed and mean): left_4 − left_1, plain_4 − plain_1, left_4 − plain_4, left_2 − plain_2, left_4 − wide_1,
  left_4_noskip − left_4;
- per layer: fraction of units with |λ| < 0.5 at the end, and the running RMS of f at the end (dead-feature check);
- H33's reference numbers (first 1 MB, not directly comparable): bank 3.097826, counts_mix6 2.177547, GRU 2.104744;
- the compute table; the smoke-test output; wall seconds; kernel ids.

## Pre-registered bars (orchestrator scores on the mean over seeds 0–1, both seeds the same sign; do not change)
1. **Depth pays under the leftover signal:** left_4 − left_1 ≤ −0.10 bits/byte.
2. **The leftover signal is what makes it pay:** left_4 − plain_4 ≤ −0.05 bits/byte.
3. **Depth beats width at matched compute:** left_4 − wide_1 ≤ −0.03 bits/byte.
Report only: plain_4 − plain_1 (does a plain local stack degrade with depth?), the no-skip arm, D = 2.
Kill rules: if bar 1 fails for both plain and leftover, stacking with no cross-layer credit does not compose on text at this
size; next is learned cross-layer credit (broadcast error into lower layers' input maps, DFA style). If bar 1 passes and bar
2 fails, depth pays with any local signal and the leftover error is not the trick. If bars 1–2 pass and bar 3 fails, the
stack helps only as extra width.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `h34a_local_depth.py`, `kaggle/h34a/**`, `results/h34a/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h33_bank_text.py`, `h30x_trace.py`, `h30y_roots.py`, `h30w_ring.py`,
  `h32b_mixer.py`, `h26_cued.py`, any other experiment code, other `results/` or `kaggle/` folders, `.env`. Another worker
  writes `w6_world.py`, `w6_headroom.py`, `kaggle/w6/**`, `results/w6/**` today: do not edit them.
- Gates and bars are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.

## Amendment (orchestrator, 16 Nov, after smoke check 1 stopped the first attempt)
The first attempt stopped correctly at smoke check 1 (max per-byte difference 8.7e-4 bits in float32). The orchestrator
reran it per chunk: the difference is 1.7e-5 before any learning, stays flat (2e-4 to 9e-4) without growing, and the mean bits
agree to 1e-6 (11.2334750 vs 11.2334740). That is float32 summation order, not a structural difference. **Check 1 is amended:**
run both the H33 reference and this code's layer in **float64** (model parameters, state and traces) for that check only; the
limit stays 1e-6 bits per byte. Scored runs stay float32. Everything else in the spec is unchanged. Continue from check 1 with
the existing `h34a_local_depth.py`, then checks 2–6, then the kernels.

## Amendment 2 (orchestrator, 17 Nov, after the float64 check stopped the second attempt)
The float64 difference (1.0e-4) comes from initialisation only: this code's B_re/B_im start one float32 rounding step
(6e-8) away from `h30y.build_arm`'s. The orchestrator loaded the reference's initial parameters into the layer and reran:
max per-byte difference **8.6e-14** in float64, so the learning rule is exact. **Check 1 is amended again:** copy
`h30y.build_arm('roots_learn', 256, 0)`'s initial parameters into this code's layer before the float64 comparison (limit 1e-6,
unchanged). Then run checks 2–6 and the kernels as specified. Nothing else changes.
