# Handoff: PPM floor for the H20 world (worker: codex or grok)

Context: /root/pdp is a research repo; read CLAUDE.md first. `h20_world.py` generates a label-free symbol stream
(`stream(world, seed, n)` returns (string, list of process ids)). `docs/h20_world.md` asks for "a PPM/CTW floor before
the first learner is scored".
Must not edit: h20_world.py, cls_sleep.py, docs/loop.md, audit.md, any existing file. Do not git commit.
Use `.venv/bin/python` (system python has no numpy). Machine: 2 cores, 3 GB RAM: Pool(2) at most.

Task: create `h20_ppm.py` (plain Python + numpy, terse style like h20_world.py).
1. Import `stream, ALPHA, PROCS, WORLDS` from h20_world.
2. Online PPM over the 41-symbol alphabet, escape method D (PPMD), full exclusion, max order K (CLI `--orders`,
   default `4 6 8 12`); order -1 = uniform. Prequential: −log2 p(x_t | x_<t) with counts BEFORE the update, then update
   all orders 0..K (no update exclusion; say so in the docstring). Assert the 41 probabilities sum to 1 (|sum−1| < 1e-9)
   on the first 2000 symbols of one run.
3. For world in slow, fast and seed in 6 7 8, n = 524288 (`--n`): mean bits/symbol per K, per-process mean bits, switch
   bits = mean over the first 1024 symbols after every switch (switches every WORLDS[world] symbols from 0, as run() in
   h20_world.py).
4. Write results/h20_ppm/results.json and results/h20_ppm/RESULTS.md: table rows = K, columns slow/fast mean ± std;
   per-process table for the best K; switch bits; one sentence against lzma 0.675±0.011 / 0.791±0.015 and KT mixture
   1.480 / 1.539. Times UTC+8, no hostnames or paths.
5. Smoke test with --n 20000 first. If the full run would take > 40 min, drop the largest orders and say so.
6. Final message: the table, runtime, anything surprising. Short.
