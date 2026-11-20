I'll read the project instructions and the H34d W9 handoff first, then follow that handoff exactly.The handoff is a Kaggle run of the staged H34d and W9 kernels, plus a papers-index check. I'll read both READMEs and the long-run rules before dispatching.H34d is about six kernels and W9 is twelve. I'll check the dispatcher caps and the live Kaggle queue before starting either one.Prepare and the W9 layout check both passed, and the account has no fresh kernels. W9 is about 69 minutes per kernel, so I'll run H34d first, then W9, and check the papers index while H34d starts.H34d's six kernels finished, and the summarizer stopped on a norm check. The handoff says to rerun the same dispatcher once, so I'll do that after seeing which run failed.The H34d retry failed on the same norm check, so I will not run it again. The six kernels did finish. Next is the W9 dispatch.W9 scored all 60 lives. H34d’s six kernels finished, and the summarizer failed twice on a norm check, so `results/h34d/RESULTS.md` was not rewritten.

The dispatchers ran one after the other. The account stayed at 3 running kernels. No code was edited and nothing was committed.

## W9

`results/w9/RESULTS.md` (summarizer: `lives=60/60 smoke=1`):

| T14 gap | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | Mean | Threshold <= | Count <= | Required count | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_night - fixed_random | -0.342886720 | -0.088811197 | -0.248997046 | 0.000239780 | -0.051492392 | -0.192265792 | -0.226319026 | -0.092256758 | -0.310888153 | -0.155381811 | -0.309051563 | -0.344804470 | -0.196909596 | -0.100000000 | 8 | 9 | 12 |
| pressure_opp - fixed_night | 0.329314347 | 0.231816841 | 0.165628897 | 0.226648083 | 0.295879527 | 0.274018142 | 0.464942271 | 0.085036414 | 0.417462800 | 0.144878844 | 0.259219006 | 0.311211941 | 0.267171426 | 0.030000000 | 0 | NA | 12 |
| pressure_opp - fixed_random | -0.013572373 | 0.143005644 | -0.083368149 | 0.226887863 | 0.244387135 | 0.081752351 | 0.238623245 | -0.007220343 | 0.106574647 | -0.010502967 | -0.049832556 | -0.033592529 | 0.070261831 | -0.070000000 | 1 | 8 | 12 |
| pressure - fixed_random | -0.108108392 | 0.116450247 | -0.139874177 | 0.079691111 | 0.175012431 | -0.057136941 | 0.056776361 | 0.040687965 | -0.064557795 | -0.040607341 | -0.178964966 | -0.237006693 | -0.029803182 | NA | NA | NA | 12 |
| fixed_random - nosleep | 0.014380720 | -0.258838103 | -0.065525012 | -0.358722333 | -0.399543588 | -0.281254178 | -0.231490314 | -0.201158594 | -0.078370182 | -0.164239117 | 0.000644060 | -0.074954193 | -0.174922570 | NA | NA | NA | 12 |

| Mean threshold | Mean | Threshold <= | Meets (0/1) |
|---|---|---|---|
| pressure_opp - fixed_night | 0.267171426 | .030000000 | 0 |

Kernels, all `complete` on attempt 1: `ezzzzzx/w9-day-night-s60` through `ezzzzzx/w9-day-night-s71`. No kernel failures.

## H34d

All six kernels are `complete` on attempt 1:

- `ezzzzzx/h34d-plain-4-s0`
- `ezzzzzx/h34d-plain-4-s1`
- `ezzzzzx/h34d-dfa-4-s0`
- `ezzzzzx/h34d-dfa-4-s1`
- `ezzzzzx/h34d-dfa-only-4-s0`
- `ezzzzzx/h34d-dfa-only-4-s1`

`kaggle/h34d/dispatch.py --dispatch` then called `summarize.py`, which raised `AssertionError` at `kaggle/h34d/summarize.py:50`:

`abs(own_norm - dfa_norm) <= 1e-6 * max(own_norm, 1e-30)`

The same command was run once more and failed on that same line. `results/h34d/RESULTS.md` is still the earlier table (`0/6` scored, every gap `NA`). The fetched `results.json` files are in `results/h34d/kernels/`.

The check fails on 2–6 chunks out of 5628 per DFA run. The largest relative gap is `1.415e-6` (`dfa_only_4` seed 0, chunk start 154880, layer 4: own `0.027639705687761307`, dfa `0.02763974480330944`). The other maxima are `1.253e-6`, `1.155e-6`, and `1.294e-6`.

Gaps from those result files, in the summarizer’s layout:

| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |
|---|---|---:|---:|---:|---|---:|---:|
| Tail | dfa_4 − plain_4 | 0.279110572 | 0.281270516 | 0.280190544 | ≤ -0.03 | 0/2 | 2/2 |
| Tail | dfa_only_4 − plain_4 | 0.304159705 | 0.315416540 | 0.309788123 | report only | — | 2/2 |
| Full | dfa_4 − plain_4 | 0.332267477 | 0.353187871 | 0.342727674 | report only | — | 2/2 |
| Full | dfa_only_4 − plain_4 | 0.449543685 | 0.483944206 | 0.466743945 | report only | — | 2/2 |

Wall seconds: plain_4 2242.449 / 1819.689, dfa_4 2189.762 / 2108.723, dfa_only_4 810.289 / 2298.451.

## Papers

`he2016` was already in D1 `pdp-papers-index`: He, Zhang, Ren, and Sun, “Deep Residual Learning for Image Recognition”, CVPR 2016, arXiv `1512.03385v1`, status `memory`, file status `stored`.

`howard2019` was missing. It is now a dataset row: Jeremy Howard, “Imagenette”, 2019, venue `dataset`, url `https://github.com/fastai/imagenette`, status `checked`, file status `unavailable`. The papers table has no separate dataset type, and there is no paper PDF. The GitHub README names Jeremy Howard and notes the 70/30 split on 6 Dec 2019. The image tarballs are not in R2.
