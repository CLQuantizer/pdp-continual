I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is W8-1c: score visit reward with true stored-sight progress, keep the rest of W8-1b, and register two papers. I'll read the prior spec, results, and the reservoir code first.W8-1b paid the TV for glance progress, which treats every stored sight as the current grey. I'll score visits with `ObservedReservoir.progress` and keep the old arm only for comparison.The first pass did not make the TV the smallest reward, so the scored run stopped. I'll check that this is the stored-sight score and not a bug before reporting it.The scored run stopped. On life 0 the TV's first-pass visit reward is the largest, not the smallest.

`curious_prog` uses `ObservedReservoir.progress`. Life 0, first pass (one 8-glance visit per panel):

| kind | observed | glance |
|---|---:|---:|
| static | 2164.082432 | 2164.082432 |
| tv | 2281.431119 | 4119.197990 |
| fast | 2162.344540 | 2162.344540 |
| slow | 2095.401885 | 2095.401885 |

The smallest observed reward is slow. The static, fast, and slow panels match between the two progress functions. On the TV, stored-sight progress cuts the glance reward from 4119.198 to 2281.431, and the later glances stay near 250 bits even when most pixels are repeats. The still-panel check passed: life 0, static panel, first 64 glances, max absolute difference 8.527e-14.

Lives 0–19 were not scored. Bars were not evaluated. `round_robin` and `curious_surp` were not run. The report is `results/w8d/RESULTS.md` and `results/w8d/results.json`. Eight life-0 `curious_prog` strips are in `internal/w8_demo/wall_c/`.

Both papers are in D1 `pdp-papers-index`. `zhang2019` is `checked` / `stored` (`papers/zhang2019.pdf`, arXiv 1905.08094). `borbely1982` is `checked` / `wanted` (Human Neurobiology 1982, PMID 7185792, no DOI on PubMed). Nothing was committed.

The scoring process exited 2 on purpose. That is the first-pass stop: the TV reward was the largest, so lives 0–19 were not scored.

The numbers are in `results/w8d/RESULTS.md`. On life 0 the observed visit rewards were static 2164.082, TV 2281.431, fast 2162.345, and slow 2095.402.

The log watcher exited on the same stop. The TV first-pass reward was the largest, so the scored run did not start. The numbers are in `results/w8d/RESULTS.md`.
