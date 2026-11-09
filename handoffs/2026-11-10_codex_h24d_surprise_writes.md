# Handoff H24d: surprise-gated writes to the store (worker: codex, --sandbox workspace-write)

Read CLAUDE.md, results/h24c/RESULTS.md (with Verdict) and `h24c_combined.py`. Create only `h24d_surprise.py` and
`results/h24d/*` (import from earlier scripts; do not edit them). Do not commit; do not touch docs/loop.md or audit.md.
Use `.venv/bin/python`, Pool(2).

Same world as H24a/H24c (seeds 6 7 8, ticks 10000, Q 0.2, K 200), same replaying cortex `ffr` (8 replays per arrival from
the store). Reuse H24c's numbers for ff, exem, gate, ffr, gate_r; do not rerun them.

## New routes (prequential, on the same arrivals)
1. `gate_s`: exact-match gate (d² < 1) over a replaying cortex `ffr_s`, where the store **writes an arrival only if the
   route's answer for it was wrong** (the gate's answer when it fired, else ffr_s's answer). `ffr_s` replays only from
   this surprise-gated store. Everything else as gate_r.
2. `gate_s2`: like gate_s, but write when the route was wrong **or** ffr_s's softmax-free FF margin for the true label is in
   the bottom 20% of its last 500 margins (low confidence counts as surprise). Document how the margin is computed.
Report store size over time (peak, final), floats stored, and the fraction of writes that were episodes vs normal digits.

## Report: results/h24d/RESULTS.md (numbers only) + results.json
Tables as in H24c for ff, exem, gate, ffr, gate_r (reused) and ffr_s, gate_s, ffr_s2, gate_s2; store sizes; write
fractions. Smoke test ticks 400 first. Budget 90 min.

## Pre-registered bars (orchestrator scores; do not change)
1. **M3-MNIST PASS:** the better of gate_s / gate_s2 has lifetime ≥ gate_r − 1.0 (i.e. ≥ 84.35) with a final store
   ≤ 0.5 × gate_r's (≤ 5 000 exemplars), mean over seeds.
2. Episodes at exposure 6+ ≥ 99% for that route (the store still holds every episode it needs).
3. Report only: replaying-cortex normal-digit accuracy with the surprise-gated store vs ffr (does replaying only
   surprises hurt or help the cortex?).
