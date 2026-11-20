I'll start by reading the project instructions and the handoff so I follow that task exactly.The handoff is W8-1b: curiosity picks a wall panel, and a shared sweep moves inside it. I'll read the W8-1 spec, results, and code before changing anything.W8-1 spent the whole life exploring 64 arms, so the bandit never used what it learned. I'll read the sweep, the bandit, and the wall runner so the new code only changes which panel is chosen.Load is 0.24, so this stays on the VPS. W8-1’s 64-arm bandit never finished its first pass; W8-1b will choose among the four panels and keep one sweep cursor per panel.Coverage matches W8-0 and the panel bandit finishes its first pass in four visits. Next I’ll time one life of each policy.Life 0’s progress bandit returns to the TV after the first pass. I’ll check the four visit rewards before the full run, to see whether that follows the progress sum.W8-1b is scored for all 20 lives. The panel bandit finishes its first pass at decision 32 on every curious life, then uses what it learned. Results are in `results/w8c/RESULTS.md` and `results/w8c/results.json`. Demo strips are in `internal/w8_demo/wall_b/`.

| bar | observed | required | result |
|---|---|---|---|
| 1 curiosity beats round robin | −0.191146, CI [−0.226562, −0.157292], 0/20 | ≥ +0.03 and CI excludes 0 | FAIL |
| 2 ignores the TV | prog 0.590625, surp 0.730469 | prog ≤ 0.15 and surp ≥ 0.35 | FAIL |
| 3 watches what changes | fast 0.255078 > slow 0.102344 > static 0.051953 | fast > slow > static | PASS |

`curious_prog − curious_surp` is +0.036458, CI [0.015365, 0.055990], 15/20.

Round-robin matches the W8-1 life accuracies and log-losses on all 20 lives. Life 0 kinds are panel 0 static, 1 TV, 2 slow, 3 fast. The first 32 `curious_prog` choices start `static, tv, slow, fast`, then return to the TV. On that first pass the visit rewards were 2164, 4119, 2095 and 2162 bits, so the greedy step selects the TV. Each curious life makes 128 panel choices: 4 optimistic, then on average 12.6 epsilon and 111.4 greedy.

One life of each policy took about 17–20 seconds. Life 0 took 96 seconds for all five. The continuation ran on this VPS with 2 torch threads, starting at load 0.57. Nothing was committed.
