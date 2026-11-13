# M7-W3: the whole loop on a local cortex. Curious agent, honest stores and sleep, with the trace-learned bank (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then the Verdicts in `results/m5bw3/`, `results/m6bw3/` and `results/h30y/`, and the code in
`m6b_w3.py`, `m6_w3.py`, `m5_w3.py`, `h30y_roots.py`, `h30x_trace.py` and `w3_world.py`.

## Context
In W3 the curious controller works (M5b). But sleep earned nothing (M6b), because the learner, a hashed table of
per-context weights, hardly forgets, so a repeater has nothing to repair. The trace-learned diagonal bank with the
roots init (H30y) is our best local cortex. It learns online with no unrolling, and its weights are shared across
everything it sees, so learning one room can wear down another. M7 makes the bank W3's cortex and runs the whole
loop: act, learn, choose by progress, and sleep when sleep pays. Principles served: all four.

## Learner: `BankCortex` (in `m7_w3.py`)
- The `roots_learn` bank from `h30y_roots.py`: N = 256, θ_k = 2πk/256, |λ| = 0.98, embedding 32, order-2 readout,
  trace learning (the `h30x_trace.py` rule), Adam lr 1e-3, clip 1.0, float32. Alphabet: the 41 H20 symbols.
- **Room input:** a learned channel embedding (10 rows: the 9 room channels and the corridor), added to the symbol
  embedding at every step. Init it to zeros.
- **One life, one stream:** the bank's state and traces carry across glances, in the order the agent experiences
  them (rooms and corridors). Each glance (64 symbols) is one chunk: score it, then take one Adam step, as in H30x.

## Stores and progress (as in M6b, at glance level)
- After a room glance is scored and learned, it goes whole (its 64 symbols, its channel and the symbol before it) by
  coin flip (rng `default_rng(40_000 + seed)`) to the **exam** or the **replay** store for its channel. Each is a reservoir of **32 glances
  per channel** (rngs `default_rng(20_000 + seed)` for exam and `default_rng(50_000 + seed)` for replay). Corridor
  glances are stored nowhere.
- **Scoring a stored glance:** run the bank from a zero state over the glance, with its channel embedding, and sum the
  bits. Batch the glances.
- **Progress of a decision** = Σ over all exam glances of (bits before − bits after). "Before" may reuse the previous
  decision's "after" when nothing changed in between. A visit's reward is its summed progress divided by its decisions, as in M6.
- **Sleep decision:** draw one glance uniformly from the union of the replay stores (rng `default_rng(30_000 + seed)`).
  Run the bank over it from a zero state with its channel embedding, and take one trace-learning Adam step, as for a
  live glance. The live state and traces are saved before the sleep and restored after, so the waking stream continues
  where it left off. A sleep visit is 8 sleep decisions, with no travel.

## Controllers (W3 visit schedule as in M5; choices from `default_rng(10_000 + seed)`)
- `rand_dwell` and `oracle7` as in `w3_headroom.py`.
- `prog_total`: the bandit over 9 rooms with the whole-exam-store reward, and no sleep.
- `prog_sleep`: the same over 10 arms, with sleep last in the optimistic first pass.
- `fixed_sleep`: `prog_total` with forced sleep visits, evenly spaced, matching `prog_sleep`'s sleep share per seed (as in M6).
Score **W**: the frozen final bank on a fresh continuation (EVAL = 2,048 symbols) of each of the 7 learnable channels,
from a copy of the world, starting from a zero state, with that channel's embedding. Also report **W9**.
**Forgetting:** for each learnable channel, the exam-store bits right after the agent's last visit to it, and at the
end of life. Report their difference.

## Seeds and lives
**World seeds 30–35** (fresh). L = 2,048.

## Kernels
Private CPU kernels, one per seed (`ezzzzzx/m7-w3-s30` … `-s35`), each running the five controllers with a pool of
up to 4. Each has `"is_private": true`, `"enable_gpu": false` and `"enable_internet": false`, and at most 90 minutes. The code goes up as private dataset
`ezzzzzx/pdp-code-m7w3`.

## Smoke test (VPS CPU, before any kernel)
1. **The bank is H30y's:** with the channel embedding fixed at zero and frozen, a `BankCortex` fed H20 slow seed 6 in
   64-symbol glances reproduces `h30y_roots.py`'s roots_learn bits on the first 4,096 symbols to 1e-6.
2. **Stores:** after a 128-decision `prog_sleep` life on seed 30, no glance is in both stores, and sleep drew 0 exam
   glances.
3. **Sleep leaves the waking stream intact:** the live state and traces are bit-identical before and after a sleep
   visit. Only the parameters change.
4. All five controllers give a finite W on seed 30 at L = 64. Time a 256-decision `prog_sleep` life and estimate one
   seed's kernel wall time. If it exceeds 90 minutes, stop and report. Do not shrink anything.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/m7w3/RESULTS.md` and `results/m7w3/results.json`, with numbers only and no verdict:
- W and W9 per controller and seed, with mean ± std (ddof = 0), and the per-channel held-out bits;
- the gaps per seed: prog_total − rand_dwell, prog_total − oracle7, prog_sleep − prog_total and prog_sleep − fixed_sleep;
- the forgetting numbers per controller (mean over seeds and channels, and per channel);
- sleep share overall and by quarter, and the sleep vs room reward by quarter, for prog_sleep;
- shares per channel, in corridors and in sleep, per controller;
- for reference, M5b's and M6b's numbers for the hashed learner;
- the smoke-test output, wall seconds and kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
1. **Curiosity survives the substrate change:** mean over seeds of (prog_total − rand_dwell) ≤ −0.10 on W, with
   prog_total lower on at least 5 of 6 seeds.
2. **Sleep earns its cost in a cortex that forgets:** mean over seeds of (prog_sleep − prog_total) ≤ −0.05 on W, with
   prog_sleep lower on at least 5 of 6 seeds.
Report only: prog_sleep − fixed_sleep, oracle7, W9, forgetting, and the sleep share and its time course.
Kill rules: if bar 1 fails, progress measured on a shared-weight cortex is too entangled (learning one room moves the
exam bits of all) for a bandit to read. If bar 2 fails while forgetting is clearly positive, replay at this dose and
store size does not repair this cortex's loss. If forgetting is near zero, the bank does not forget either, and
sleep had nothing to repair.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `m7_w3.py`, `kaggle/m7w3/**` and `results/m7w3/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `m6b_w3.py`, `m6_w3.py`, `m5_w3.py`, `h30y_roots.py`, `h30x_trace.py`,
  `curio.py`, `w3_world.py`, `w3_headroom.py`, other `results/` or `kaggle/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
