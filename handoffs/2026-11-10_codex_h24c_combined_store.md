# Handoff H24c: combined store route on episodic MNIST (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, docs/hippocampus_diagnosis.md, results/h24a/RESULTS.md, and `h24_episodic.py` (the H24a code). Create
`h24c_combined.py` (copy and extend; do not edit h24_episodic.py). Do not commit; do not touch docs/loop.md or audit.md.
Use `.venv/bin/python`, Pool(2).

## What to add (same world, same seeds 6 7 8, ticks 10000, Q 0.2, K 200)
Routes (all prequential, on the same arrivals as H24a's routes, which you keep for comparison):
1. `gate_r`: the exact-match gate (d² < 1) over the *replaying* cortex `ffr` instead of `ff`.
2. `gate_rd`: like gate_r, plus **deletion**: a stored exemplar is deleted once the replaying cortex has answered that
   exact item correctly on its last 2 exposures (checked when the item arrives, before learning). Deleted items are no
   longer replayed or matched (if the cortex later fails on it again, it is simply re-stored on that arrival).
   Normal (non-episode) arrivals are stored as in H24a but deleted by the same rule.
3. Store cost: report the exemplar count over time (peak and final) for exem/gate_r vs gate_rd, and floats stored
   (784 per exemplar + 1 label).

## Report: results/h24c/RESULTS.md (numbers only) + results.json
Tables as in results/h24a/RESULTS.md (lifetime, normal, episodes by exposure 1 / 2 / 3–5 / 6+) for ff, exem, gate,
ffr, gate_r, gate_rd; gate firing rates; store size peak and final. Smoke test with ticks 400 first. Budget 90 min.

## Pre-registered bars (orchestrator scores; do not change)
- **P1:** gate_r lifetime ≥ max(gate, ffr) + 1.0 (mean over seeds).
- **P2:** gate_rd lifetime ≥ gate_r − 1.0 AND gate_rd final store size ≤ 0.5 × gate_r's.
- **Kill for deletion:** P2 fails → deletion by "right twice" is too eager; next try is consolidation by a held-out
  probe rather than by recent answers.

Final message: the tables, runtime, anything odd.
