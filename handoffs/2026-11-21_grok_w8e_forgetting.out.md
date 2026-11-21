I'll read the project instructions and the handoff, then follow that handoff exactly.The handoff is W8-1d: a forgetting pixel model and a recent reservoir. I'll read the prior spec, results, and the code it builds on.W8-1c scored progress on a lifetime count, so the camera avoided change. I'll read the reservoir and the panel bandit so the forgetting model can sit on the same world.Codex is using the CPU (load 1.26), so the scored run waits until the 1-minute load drops below 1. I'll add the forgetting model and the recent reservoir first, then check the formula before that wait.The first-pass gate passed: on life 0 the TV visit reward is the smallest. Static progress is still 43% of the first visit after six visits, and the first visit after a fast-panel replacement is lower than the one before it. Those two checks are print-only, so the 20-life run is continuing. Unchanged arms are being checked against `results/w8d`.The run has been going for about an hour without a stop or a finish. I'll check the log to see whether it is still scoring lives.Lives 0–19 are in `results/w8e/RESULTS.md` and `results.json`. The forgetting model still camps on the static picture.

| bar | observed | result |
|---|---|---|
| 1. curious_prog − round_robin ≥ +0.03, CI excludes 0 | −0.2776, CI [−0.3141, −0.2406], 0/20 lives ahead | FAIL |
| 2. curious_prog TV share ≤ 0.15 and curious_surp ≥ 0.35 | 0.056 and 0.730 | PASS |
| 3. curious_prog share fast > slow > static | fast 0.064, slow 0.095, static 0.785 | FAIL |

Mean life accuracy for `curious_prog` is 0.278 below round-robin. It spends 78% of its glances on the static panel and 5.6% on the TV. `curious_surp_forget` (report only) spends 89% on the TV. `curious_prog_lifetime` reproduces W8-1c (static share 0.700).

Smoke, life 0:

- The static stare does not die by visit 6. Visit rewards are 442, 724, 608, 420, 307, 191. Visit 6 is 43% of visit 1.
- The first visit after each fast-panel replacement is lower. At t=32 the reward goes from +419 to −3161. At t=64 it goes from +325 to −2738. The next visits of the new picture climb back through zero (visit 8 of the first new picture is +325).
- The first pass keeps the TV smallest: static 442, slow 467, fast 432, TV −353.
- `random`, `round_robin`, `curious_surp`, and `curious_prog_lifetime` match `results/w8d` on every life.

γ = 0.8 and the reservoir length of 32 were left fixed. Wall time was 4373 s. Demo strips are in `internal/w8_demo/wall_d/`. Nothing was committed.

The scored run exited cleanly after 4397 seconds. Lives 0–19, the three bar counts, and the life-0 probes are in `results/w8e/RESULTS.md`.
