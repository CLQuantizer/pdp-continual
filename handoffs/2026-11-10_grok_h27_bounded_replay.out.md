I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is a bounded-replay experiment plus a papers registration. I'll read the cited specs and the M4c code before writing anything.The experiment keeps M4c’s `hist_rep` machine and only changes the replay source and dose. I’ll implement that, check it against M4b on a short run, then start the papers lookup.Auto-compacting conversation (81% full)...
Conversation compacted.
The dataset create was still running. I'll check that it finished, then push the smoke kernel and continue the papers registration.Unpaywall marks several of these open. I'll confirm the PDFs are real publisher or PMC copies before registering them, and keep watching the smoke kernel.Smoke finished. I'll pull its output and check the seven runs before starting the score kernels.Smoke on seed 15 at T=32 passed the structural checks, and the three scoring kernels are running.

At T=32 the capacity-77 banks are not full yet, so `res77` and `fifo77` matched (W 2.631890, 32 segments, traffic 16) and differed from whole-history `hist2` (W 2.989634). `res16` held 16 segments. All seven conditions shared one glance log.

Scoring kernels, private CPU, T=512:

- `ezzzzzx/h27-score-a` seeds 15–16
- `ezzzzzx/h27-score-b` seeds 17–18
- `ezzzzzx/h27-score-c` seeds 19–20

Papers from the references in `docs/compute-vs-data-movement.md` are in the index. `howarth2012` and `levy2021` are stored and checked (PMC publisher PDFs). `mcclelland1995` was already stored and checked; the citation path was added. The other nine are checked and marked wanted: Aston-Jones 2005, Attwell 2001, Bullmore 2012, Fries 2005, Lennie 2003, Mongillo 2008, Rao 1999, Sterling 2015, Williams 2009. No open publisher, author, arXiv, or PMC file for those was retrievable. The sync reported no rows left to fetch.

The three score kernels finished. I'll fetch their outputs, check the runs, and assemble the report.H27 is complete. 42 of 42 scored runs are in `results/h27/RESULTS.md` and `results/h27/results.json`. Bars are not scored here.

Mean held-out W over seeds 15–20 (population standard deviation). Traffic and M are the same on every seed.

| Condition | W | traffic | M |
|---|---:|---:|---:|
| nostore | 1.807552 ± 0.068635 | 0 | 0 |
| hist1 | 1.436939 ± 0.106409 | 8 | 176695.457 |
| hist2 | 1.491167 ± 0.184286 | 16 | 176695.457 |
| hist4 | 1.635926 ± 0.215833 | 32 | 176695.457 |
| res77 | 1.876734 ± 0.169140 | 16 | 26573.340 |
| res16 | 2.251083 ± 0.319990 | 16 | 5521.733 |
| fifo77 | 1.790832 ± 0.188153 | 16 | 26573.340 |

Per-seed W. Traffic and M for each seed are the condition values above.

| Seed | nostore | hist1 | hist2 | hist4 | res77 | res16 | fifo77 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 1.799047 | 1.509955 | 1.594252 | 1.840193 | 1.916798 | 2.101884 | 1.805840 |
| 16 | 1.920899 | 1.413629 | 1.466006 | 1.581480 | 1.824000 | 2.149717 | 1.737446 |
| 17 | 1.791745 | 1.317255 | 1.378718 | 1.477188 | 1.737666 | 1.921342 | 1.694793 |
| 18 | 1.740514 | 1.398488 | 1.302417 | 1.475161 | 1.765358 | 2.856997 | 1.617460 |
| 19 | 1.868521 | 1.633246 | 1.847050 | 2.011778 | 2.232908 | 2.468102 | 2.192400 |
| 20 | 1.724587 | 1.349060 | 1.358561 | 1.429755 | 1.783675 | 2.008453 | 1.697052 |

Recorded differences, mean over seeds: res77 − hist2 = 0.385567 ± 0.046532. res77 − fifo77 = 0.085902 ± 0.037382. Bits saved per unit traffic: hist1 0.046327 ± 0.011997, hist2 0.019774 ± 0.009724, hist4 0.005363 ± 0.005943.

Kernels, all private CPU, version 1, no reruns. Elapsed time is compute wall from each manifest. Dataset `ezzzzzx/pdp-code-h27`.

| Kernel | Seeds | T | Seconds |
|---|---|---:|---:|
| ezzzzzx/h27-smoke | 15 | 32 | 31.517 |
| ezzzzzx/h27-score-a | 15, 16 | 512 | 540.730 |
| ezzzzzx/h27-score-b | 17, 18 | 512 | 517.035 |
| ezzzzzx/h27-score-c | 19, 20 | 512 | 577.914 |

Smoke at T=32 kept one shared glance log. `res77` and `fifo77` matched there (W 2.631890, 32 segments) because capacity 77 was not full. At T=512 both hold 77 segments and their W values differ. `res16` holds 16.

Papers from the References section of `docs/compute-vs-data-movement.md` are registered. Status is checked for every one. `howarth2012` and `levy2021` are stored (PMC publisher PDFs). `mcclelland1995` was already stored; its citation list now includes that doc and the file was left as stored. Wanted, with no legal PDF retrieved: `astonjones2005`, `attwell2001`, `bullmore2012`, `fries2005`, `lennie2003`, `mongillo2008`, `rao1999`, `sterling2015`, `williams2009`. `python3 tools/sync_papers.py` and `--fetch` ran; there were no rows left to fetch. Nothing was deleted.
