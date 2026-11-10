# Handoff: find real datasets for the next world (worker: grok, lookup only)

Read /root/pdp/CLAUDE.md (Kaggle section) and docs/roadmap.md ("The target machine"). An experiment (M4b) is running on
this machine: **do no heavy compute**, run nothing in `.venv/bin/python` beyond tiny file inspection, keep downloads small.
Kaggle CLI: `set -a; . ./.env; set +a; .venv/bin/kaggle ...` (never print the token). Create only
`handoffs/2026-11-10_grok_kaggle_worlds.table.md` and small samples under `data/kaggle_samples/` (gitignored).
Do not commit; do not touch docs/loop.md or audit.md.

## What the machine needs
Our agent learns online by compressing symbol streams (bits/symbol, prequential) and chooses which stream ("channel")
to watch next by compression progress. The current world is synthetic (42 symbols, counters, CA, motifs, key=value
facts, noise). We want real streams that give it, in one world:
1. **Learnable structure at different difficulty**, so curiosity has a curriculum: easy (repetitive), medium, hard but
   learnable (long-range rules: games, code, music).
2. **Episodic facts that recur**, so a memory store pays: the same entity/fact appears again later (e.g. Q&A, records,
   repeated names).
3. **Near-irreducible channels** as the real noisy TV: e.g. high-frequency price ticks, hashes, sensor noise.
4. **Known reference numbers** where possible (published bits/char of gzip/xz/PPM/cmix or similar), so we can calibrate.
5. Fits a 2-core, 3 GB RAM CPU box: a usable slice of 1–20 MB per channel; text or easily symbolised (byte-level is fine).
6. Licence that allows research use; no scraped-personal-data sets.

Ideas to check (not a fixed list): enwik8/enwik9 or text8 (Hutter Prize), chess games (PGN/UCI move lists, Lichess),
MIDI/ABC music, source code, DNA/protein sequences, Jeopardy or trivia Q&A, Wikipedia infobox / knowledge triples,
event logs, weather/sensor series, minute-level crypto/FX prices, game replays as action sequences.

## Task
1. Search with `kaggle datasets list -s "<term>" --sort-by votes` (and `--file-type`, `--max-size` if useful); inspect
   candidates with `kaggle datasets files <ref>` and `kaggle datasets metadata <ref>`.
2. For the top ~8, download **one small file each** (≤ 50 MB; skip if no such file) to `data/kaggle_samples/<slug>/`,
   and record: format, alphabet size at byte level, a 3-line excerpt (no personal data), and `xz -9 -T1` bits/byte on a
   1 MB slice (`head -c 1000000 f | xz -9 -T1 | wc -c`, ×8/1e6). That xz number is our cheap difficulty probe.
3. Write the table: ref, what it is, size, licence, which of needs 1–4 it serves, usable slice, xz bits/byte, published
   compression references if you find any (with source), risks. Rank by fit, and flag the 3 you would combine into one
   multi-channel world, with a sentence each on the channel's role (easy / hard-learnable / episodic / noise).
Time budget: 45 min.

## Output
Final message: the ranked table and the 3-pick suggestion. Facts only; I choose.
