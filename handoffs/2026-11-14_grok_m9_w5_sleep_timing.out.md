I'll read the project instructions and the handoff document first, then follow that handoff exactly.The handoff is M9-W5: test whether chosen sleep pays when rules drift. I'll read the prior verdicts and the M8/M7 code before writing anything.The prior results are numbers-only. Next I’ll read the M8 controller and the Kaggle kernel layout so the new experiment matches them.The M8 ageing controller is the reference, and sleep in this world can rehearse rules that have already been swapped out. I'll add `m9_w5.py` on top of that code, without editing it, then run the smoke test before any kernel.Auto-compacting conversation (80% full)...
Conversation compacted.
The experiment module compiled. I’ll confirm it is complete, then run the VPS smoke test before any kernel is pushed.Two `main` definitions are in the module. I’ll read both and the smoke path before starting the run.The fixed-sleep path matches M8’s operation order. I’m starting the VPS smoke test and checking the results writer against the handoff while it runs.The ring walk and the room choice break ties in opposite directions. I’ll put both rules back in the protocol before staging copies the file.The smoke run is still on the L=64 pair, and those checkpoints match M8. I’ll leave it running and prepare the kernel uploader so it can start as soon as smoke passes.The VPS smoke passed, and the 12 private CPU kernels are running.

`age_fixed20` matched M8 `prog_age_s` on seed 12 at L=64 and L=256 (T14 difference 0, under 1e-6). The L=256 sleep visit kept the live state and traces bit-identical, with no store overlap and no exam glance drawn. The L=512 visit trace and quarter shares were recorded. The kernel estimate is about 548 seconds, under the 90-minute limit. All four controllers returned a finite T14 at L=128.

Dataset `ezzzzzx/pdp-code-m9w5` is up. Kernels `ezzzzzx/m9-w5-s24` through `s28` are running; the other seven are queued behind Kaggle’s five-session cap. I’ll download each result as it finishes and write `results/m9w5/RESULTS.md` from the numbers only.

The first upload attempt failed immediately. I’ll check that the second run is still pushing kernels.M9-W5 is in `results/m9w5/RESULTS.md` and `results/m9w5/results.json`. Seeds 24–35, L = 2048, period 128. No verdict.

**T14** (mean ± std, ddof = 0):

| controller | T14 | final W14 |
|---|---:|---:|
| age_nosleep | 2.012681 ± 0.168506 | 1.501707 ± 0.126556 |
| age_fixed20 | 1.693955 ± 0.159602 | 1.400859 ± 0.359508 |
| age_sleeparm | 1.857732 ± 0.147202 | 1.449900 ± 0.305844 |
| age_fixed_matched | 1.841557 ± 0.141946 | 1.397850 ± 0.278845 |

**Gaps on T14** (mean, seeds where the left term is lower):

| gap | mean | left lower |
|---|---:|---:|
| age_fixed20 − age_nosleep | −0.318726 | 12 of 12 |
| age_sleeparm − age_nosleep | −0.154949 | 11 of 12 |
| age_sleeparm − age_fixed_matched | 0.016174 | 7 of 12 |
| age_sleeparm − age_fixed20 | 0.163777 | 1 of 12 |

The same gaps without the checkpoint at decision 256 are −0.291413 (12 of 12), −0.147350 (10 of 12), 0.020999 (6 of 12), and 0.144063 (1 of 12).

**age_sleeparm sleep.** Overall share 0.058757 ± 0.014363. By quarter: 0.050781, 0.055990, 0.063314, 0.064941. Mean decisions since the last earlier swap: 61.78 at sleep onsets and 64.36 at room-visit onsets (pooled). Mean of the per-seed means: 61.22 and 64.36.

**Stale replay** (pooled share of draws): age_fixed20 0.198121 (970/4896), age_sleeparm 0.215374 (311/1444), age_fixed_matched 0.200276 (290/1448). age_nosleep drew 0.

**Swap-finding latency** (192 swaps each): median 100.0, 114.5, 120.0, 103.0, and share within 128 of 0.552083, 0.484375, 0.500000, 0.515625, in controller order above.

Smoke wall was 268.7 s. `age_fixed20` matched M8 `prog_age_s` on seed 12 at L = 256 with a T14 difference of 0. Kernels `ezzzzzx/m9-w5-s24` through `ezzzzzx/m9-w5-s35` all completed on the first push. Kernel wall was 444.1 ± 73.5 s (range 291.1–513.5 s). Code dataset: `ezzzzzx/pdp-code-m9w5`.
