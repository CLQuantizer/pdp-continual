# M6b-W3: sleep as an action, with replay and exam stores kept apart (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/m6w3/RESULTS.md` (including its Verdict), `m6_w3.py`, `m5_w3.py` and
`curio.py`.

## Context
M6-W3 failed because sleep replayed pairs from the same reservoirs that every progress reward was scored on. Sleep
memorised its own exam, the bandit chased that, and the agent slept 72% of its late life while getting worse on fresh
data. M6b fixes the yardstick: replay and exam come from disjoint stores. It also makes the store larger, so that
replay can hold more than the agent can cheaply re-observe. Principles served: store everything (1), improve the
compressor (2), and act to maximise progress (4).

## Changes from M6-W3 (everything else as in `m6_w3.py`)
Write `m6b_w3.py`. Import `m6_w3.py` and the modules it uses, and edit none of them.
- **Two disjoint stores per room channel.** Every (context, symbol) pair from a room glance, after it is scored and
  learned, goes to the **exam** store with probability ½ and otherwise to the **replay** store. The coin uses rng
  `default_rng(40_000 + seed)`. Each store is a reservoir of **R = 2,048** pairs per channel, with its own reservoir
  counter (rngs `default_rng(20_000 + seed)` for exam and `default_rng(50_000 + seed)` for replay).
- **Every progress reward** (room visits, sleep visits, for every controller that uses the whole-store reward) is
  scored on the **exam** stores only: Σ_c [score_pairs(c, exam[c]) before − after].
- **Sleep** draws its 64 pairs per decision uniformly from the union of the **replay** stores (rng
  `default_rng(30_000 + seed)`). It never touches an exam store.
- Controllers: `prog_total`, `prog_sleep` and `fixed_sleep`, defined as in M6-W3 with the changes above, and `prog`
  (M5's controller, its own per-channel reward, on its usual R_RES = 256 reservoir, no sleep) as a reference.
  `fixed_sleep` matches `prog_sleep`'s sleep share per seed and L, as in M6-W3.

## Seeds and lives
**World seeds 24–29** (fresh). L = 2,048 scored; L = 512 reported.

## Kernels
As M6-W3: one private CPU kernel `ezzzzzx/m6b-w3` (split it by seed if the timing needs it), `"is_private": true`,
`"enable_gpu": false`, `"enable_internet": false`, pool of 4, at most 90 minutes. The code goes up as private dataset `ezzzzzx/pdp-code-m6bw3`.

## Smoke test (VPS CPU, before any kernel)
1. `prog` reproduces M5b's prog W for seed 12, L = 512, to 1e-9.
2. **Disjointness:** after a 256-decision life of `prog_sleep` on seed 24, no stored pair occurrence is in both stores. Track
   by the stream index of each pair. Print the store sizes per channel.
3. **Sleep never sees the exam:** count the exam pairs drawn by sleep. It must be 0.
4. All four controllers give a finite W on seed 24 at L = 128. Time it and estimate the kernel's wall time.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/m6bw3/RESULTS.md` and `results/m6bw3/results.json`, with the same fields as `results/m6w3/`, plus the store sizes at the end of
life. Numbers only and no verdict.

## Pre-registered bars (orchestrator scores; do not change)
1. **Sleep earns its cost:** at L = 2,048, mean over seeds of (prog_sleep − prog_total) ≤ −0.05 on W, with prog_sleep
   lower on at least 5 of 6 seeds.
2. **Choosing when matters:** at L = 2,048, mean over seeds of (prog_sleep − fixed_sleep) ≤ −0.03 on W, with
   prog_sleep lower on at least 4 of 6 seeds.
Report only: L = 512, W9, prog_total − prog, the sleep share and its time course, and sleep vs room reward by quarter.
Kill rules: if bar 1 fails with an honest yardstick and a larger store, replay does not pay its decisions for this
fast, slot-wise learner. Sleep then needs a cortex that forgets, such as the GRU of H31 or a shared-weight model, before it can matter. If
bar 2 fails, timing does not matter beyond the dose.

## Budget and rules
- CPU only. At most 90 minutes of wall time per kernel. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `m6b_w3.py`, `kaggle/m6bw3/**` and `results/m6bw3/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `m6_w3.py`, `m5_w3.py`, `m5b_w3.py`, `curio.py`, `w3_world.py`, other
  `results/` or `kaggle/` folders, `.env`.
- Gates and bars in this spec are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and
  report.
- Never print or upload secrets. Do not commit.
