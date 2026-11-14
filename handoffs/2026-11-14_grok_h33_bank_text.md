# H33: substitution on text. Can the trace-learned bank replace the backprop GRU under H32b's count mixer? (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then the Verdicts in `results/h32b/RESULTS.md` and `results/h30y/RESULTS.md`, and the code in
`h32b_mixer.py`, `h32_mixer.py`, `h26_cued.py`, `h30y_roots.py` and `h30x_trace.py`.

## Context
On enwik8, H32b's best compressor is the GRU cortex (hidden 1,024, trained by backprop through time) mixed with byte count
models of orders 1–6 (mix6). Over the first 1 MB, mix6 scores 1.917 bits/byte and the GRU alone 2.105. The trace-learned
diagonal bank with the roots init (H30y) is our best local learner. It uses exact RTRL, with no unrolling and no backward
pass through time. If the bank under the same mixer comes close to the GRU, the local cortex can replace
backprop on real data. The counts already carry most of the mixer's gain, so we must also check that the bank adds anything
beyond them. Principle served: improve the compressor (2), with a learner that can live.

## Data
The first **1,000,000 bytes** of enwik8 (`h26_cued.load_enwik8`; check the E10M sha256 on the first 10 M as `h26_cued`
does, then take bytes 0–999,999). Alphabet 256. This is H32b's slice 0, scored from scratch in the same
prequential way, so it is directly comparable.

## Arms (one prequential pass each; every byte is scored before anything learns from it)
- `bank`: `h30y_roots.py`'s `roots_learn` bank, with N = 256, θ_k = 2πk/256, |λ| = 0.98, the order-2 readout, trace
  learning (the `h30x_trace.py` rule), Adam lr 1e-3, clip 1.0 and float32. Alphabet 256, embedding 32, model seed s. State and traces carry across the whole
  stream. Chunk 128: score the chunk, then take one Adam step.
- `counts_mix6`: H32b's mix6 machinery (`OrderCount`, orders 1–6 in one count store, smoothing 0.05, `GeoMixer`, Adam lr
  0.003, one step per chunk of 128, and the same weight sets and init), with **no cortex expert**. The expert list is orders 1–6
  only; the row init puts 1 on order 4 and 0 elsewhere. Counts for a chunk are added after the mixer has scored that chunk and stepped,
  as in H32b.
- `bank_mix6`: H32b's mix6 exactly, with the bank's probabilities in the cortex slot in place of the GRU's. Mixer gradients do not
  enter the bank.
- `count_1` … `count_6`: the count models alone, as in H32b. These come out of the same pass.
Model seeds s ∈ {0, 1} for the bank (the count arms do not depend on s; report them once).

## Compute accounting
For each cortex, report the parameters, the multiply-adds per byte (forward plus learning, counted analytically from the
shapes, with the formula shown), and the stored state per byte of history the learning rule needs: the bank's traces,
and for the GRU, its BPTT window from `h26_cued.py`'s settings. Use the GRU's settings in `h26_cued.py` (hidden 1,024, chunk
128); do not rerun the GRU.

## Kernels
Two private CPU kernels, `ezzzzzx/h33-bank-text-s0` and `-s1`, each running every arm for one seed in a single pass. Each has
`"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`. enwik8 comes from the Kaggle dataset `h26_cued.find_kaggle_enwik8` already
uses (attach it as H32b did). The code goes up as private dataset `ezzzzzx/pdp-code-h33`. At most 90 minutes per kernel. If
the smoke-test timing says one 1 MB pass will not fit, stop and report; do not shrink the data or the bank.

## Smoke test (VPS CPU, before any kernel)
1. **The bank is H30y's:** with alphabet 41 on H20 slow seed 6, this code's bank reproduces `h30y_roots.py`'s
   roots_learn bits on the first 4,096 symbols to 1e-6 (with H30y's chunking for this check).
2. **The counts are H32b's:** `count_1` … `count_4` over the first 20,000 bytes match H32b's code (run `h32_mixer`'s
   `CountBank` on the same bytes) to 1e-6 bits.
3. **Prequential order:** a byte's probability from each arm is the same whether or not the bytes after it exist. Check
   that, over 1,024 bytes vs 2,048 bytes, the first 1,024 bytes' bits are identical.
4. All arms give finite bits over the first 8,192 bytes. Time 32,768 bytes and estimate a 1 MB kernel's wall time.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/h33/RESULTS.md` and `results/h33/results.json`, with numbers only and no verdict:
- bits/byte per arm and seed over the 1 MB, and the mean over seeds for `bank` and `bank_mix6`;
- per 100 KB slice: bits/byte per arm and seed;
- H32b's slice-0 numbers for reference: gru 2.104744, mix6 1.917176, mix4 1.951636, count_3 2.961961 and ppmd_order_16
  1.969704;
- the gaps: bank_mix6 − 1.917176, bank_mix6 − counts_mix6 and bank − 2.104744, per seed and as a mean;
- the mixer's mean weights on each expert in the last 100 KB, for `bank_mix6` and `counts_mix6`;
- the compute accounting table;
- the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores on the mean over seeds 0 and 1; do not change)
1. **Substitution holds:** bank_mix6 − 1.917176 (H32b's GRU mix6 over the same 1 MB) ≤ +0.03 bits/byte.
2. **The bank earns its slot:** bank_mix6 − counts_mix6 ≤ −0.05 bits/byte.
Report only: bank − 2.104744 (the cortices alone) and the compute ratio.
Kill rules: if bar 2 fails, a local bank of this size adds nothing to text beyond what the counts already hold, and the next
step is a larger or deeper bank, not mixing. If bar 2 passes and bar 1 fails, the bank helps but the backprop GRU still holds
an edge on text; record the gap per multiply-add.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `h33_bank_text.py`, `kaggle/h33/**` and `results/h33/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `h32b_mixer.py`, `h32_mixer.py`, `h26_cued.py`, `h30y_roots.py`, `h30x_trace.py`,
  any other experiment code, other `results/` or `kaggle/` folders, `.env`. Do not edit W4's files
  (`w4_world.py`, `w4_headroom.py`), which another worker is writing now.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
