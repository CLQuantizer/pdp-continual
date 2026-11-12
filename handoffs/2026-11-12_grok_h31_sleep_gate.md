# H31: the thalamic sleep gate. Replay interleaved or in sleep blocks (grok, Kaggle GPU)

Read `/root/pdp/CLAUDE.md` first, then `docs/brain_map.md` (section "The thalamus"), `results/h29d/RESULTS.md`
(including its Verdict), `h29b_enwik8.py`, `h29d_enwik8.py` and `kaggle/h29d/`.

## Context
Our best replay (H29c/H29d) is interleaved: after every 128-byte live chunk, the cortex takes one replay step on 8
windows sampled uniformly from all bytes seen so far (traffic 8). On enwik8 E10M that gains 0.138 bits/byte. Brains
mostly do not interleave. They batch replay into sleep, when the thalamus closes the senses and spindles open the
cortex to hippocampal ripples. H31 asks what batching costs. Hold the replay dose fixed and deliver it in sleep
blocks. If blocks cost little, the brain's schedule is cheap. If they cost a lot, that explains why brains also replay
while awake. Principles served: store everything (1) and improve the compressor (2).

## Kaggle GPU quota (checked 12 Nov 2026)
18.63 of 30 GPU h remain this week (refresh Sat 14 Nov 08:00 UTC+8). Keep 10 h in reserve. **Budget for this job:
at most 2.5 GPU h in total.** Run `kaggle quota` before pushing and stop if fewer than 13 h remain.

## What to build
Create `h31_sleep.py`, a copy of `h29d_enwik8.py` (the model, data path, sha256 check, score-before-step order and
replay sampler unchanged) with one new flag, `--sleep-block B` (an integer ≥ 1, default 1):
- **Awake:** every live 128-byte chunk is scored prequentially (bits = live −log2 p before any update on it), then the
  cortex takes its live step. No replay step follows it while awake, except when B = 1 (below).
- **Sleep:** after every B live chunks, input stops. The cortex takes **B replay steps** back to back. Each is one
  step on 8 windows sampled uniformly from all bytes seen so far, using the same sampler, window length, zero initial
  hidden state and rng stream as H29d. No bytes are scored during sleep. Then the next awake period starts. The live
  hidden state carries across sleep unchanged.
- With **B = 1**, this is exactly H29d's interleaved replay. It must match H29d's code path bit for bit (smoke test 1).
- Replay steps in total are identical across arms, up to a remainder of at most B − 1 at the end of the data. Run a
  final sleep on the remainder chunks if any remain, so the totals match exactly. Report the count.

Data: the first **3,000,000 bytes** of enwik8 ("E3M"), from the private dataset `ezzzzzx/enwik8`. Report the sha256 of
E3M.
Fixed settings: lr 3e-4, replay batch 8, seed 0, embedding 64, hidden 1024, chunks of 128, TBPTT 128, clip 1.0,
Adam.

## Arms (one private GPU kernel each, or two arms per kernel if the quota math is better)
| Arm | sleep block B | awake bytes between sleeps |
|---|---:|---:|
| `inter` | 1 | 128 B (H29d interleaved) |
| `sleep_64` | 64 | 8 KB |
| `sleep_1k` | 1,024 | 128 KB |
| `sleep_16k` | 16,384 | 2 MB |
| `noreplay` | none: no replay at all (`--replay-batch 0`) | — |

Kernels: `ezzzzzx/h31-<arm>` (use `-` for `_`), all with `"is_private": true`, `"enable_gpu": true` and
`"enable_internet": false`. Code goes up as private dataset `ezzzzzx/pdp-code-h31`.

## Smoke test (VPS CPU, before any kernel)
1. **Equivalence:** `h31_sleep.py --sleep-block 1` and `h29d_enwik8.py` with the same flags (lr 3e-4, replay batch 8,
   seed 0) on CPU over the first 8,192 bytes give the same total bits to 1e-6. Report both.
2. On the first 8,192 bytes on CPU with `--sleep-block 16`: the replay-step count equals the `inter` arm's count, and
   bits are finite.
3. **Print the schedule** for B = 4 over the first 12 chunks: which steps are live (scored) and which are replay.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/h31/RESULTS.md` and `results/h31/results.json`, with numbers only and no verdict:
- per arm: E3M bits/byte, the three 1 MB slice bits/byte, replay-step count, wall seconds and GPU type;
- the gaps: each sleep arm − `inter`, and `noreplay` − `inter`;
- per arm, bits/byte over the 128 KB awake stretch right after each sleep vs the stretch right before it, averaged
  over sleeps, for `sleep_1k` and `sleep_16k` only. This is the "morning after" effect;
- kernel ids, versions, elapsed seconds and total GPU hours;
- the smoke-test numbers.

## Pre-registered bars (orchestrator scores; do not change)
1. **Replay still helps on E3M:** `noreplay` − `inter` ≥ 0.03 bits/byte. If this fails, the comparison below is
   void, because the dose is too small to matter at 3 MB.
2. **Sleep batching is cheap:** `sleep_1k` − `inter` ≤ 0.01 bits/byte.
Report only: `sleep_64` − `inter` and `sleep_16k` − `inter`, where the prediction is that the cost grows with B, and
the morning-after effect.

## Budget and rules
- At most 2.5 GPU h in total. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll kernels with `kaggle kernels status`.
- Files you may create: `h31_sleep.py`, `kaggle/h31/**`, `results/h31/**`.
- Do not touch: `docs/loop.md`, `audit.md`, any `h29*` file or folder, any other experiment code, `.env`.
- Never print or upload secrets. Do not commit.
