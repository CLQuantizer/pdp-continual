# Handoff: episodic process for the symbol world, H24b generator only (worker: codex or grok)

Read /root/pdp/CLAUDE.md, docs/h20_world.md, docs/hippocampus_diagnosis.md. Create only `h24b_world.py` and
`results/h24b_floors/RESULTS.md`. Do not edit h20_world.py or any other file. Do not commit. Use .venv/bin/python.

Spec:
1. `from h20_world import World, ALPHA, PROCS, WORLDS, kt_floors`. Add process `F` (facts): at World-seed time draw a
   dictionary of 300 keys (each 4 random lowercase letters, unique) to values (each 4 random digits), fixed for the
   life. Each chunk of F is 16 records `kkkk=vvvv;`, keys drawn uniformly with replacement. Value digits are only
   predictable by remembering that specific pair.
2. `stream_f(world, seed, n)`: same season logic as h20_world.stream but over PROCS + ["F"], so F competes for seasons
   like any process. Deterministic per (world, seed). Use a separate rng stream for F so the other processes'
   symbols are unchanged where possible (say whether that holds).
3. Floors on seeds 6 7 8, worlds slow and fast, n = 524288: KT mixture via kt_floors, lzma -9e, with per-process bits.
   Also report, for F alone, mean bits on value digits split by how many times that key had appeared before (0, 1, 2–4,
   5+). This shows how much a memoriser gains per exposure.
4. RESULTS.md: tables plus 3 lines on what they show. UTC+8, no paths or hostnames. Smoke test with n = 20000 first.
5. Final message: tables + runtime.
