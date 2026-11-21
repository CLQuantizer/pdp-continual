| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|
| Tail | distill_4 − plain_4 | 0.048667566 | 0.045259578 | 0.046963572 | ≤ -0.03 | 0/2 | 2/2 |
| Tail | distill_4_top − plain_4 | 0.033354078 | 0.037904497 | 0.035629287 | report only | — | 2/2 |
| Full | distill_4 − plain_4 | 0.902894845 | 0.982561465 | 0.942728155 | report only | — | 2/2 |
| Full | distill_4_top − plain_4 | 0.603798789 | 0.626472682 | 0.615135736 | report only | — | 2/2 |

| Kernels scored | Kernels expected | Smoke | Tail start | Tail end |
|---:|---:|---|---:|---:|
| 8 | 8 | True | 40000 | 180000 |

| Arm | Tail seed 0 | Tail seed 1 | Tail mean | Full seed 0 | Full seed 1 | Full mean |
|---|---:|---:|---:|---:|---:|---:|
| plain_4 | 3.091224982 | 3.091462939 | 3.091343961 | 3.745828744 | 3.769168530 | 3.757498637 |
| distill_4 | 3.139892548 | 3.136722517 | 3.138307533 | 4.648723589 | 4.751729995 | 4.700226792 |
| distill_4_top | 3.124579060 | 3.129367435 | 3.126973248 | 4.349627533 | 4.395641212 | 4.372634373 |
| distill_1 | 3.266619184 | 3.290567197 | 3.278593190 | 3.537297311 | 3.571738225 | 3.554517768 |

| Arm | Seed | Layer | Own full | Own tail | Mean mixer full | Mean mixer tail | Final mixer |
|---|---:|---:|---:|---:|---:|---:|---:|
| plain_4 | 0 | 1 | 3.549181220 | 3.280796641 | 0.439583981 | 0.374364861 | 0.388745010 |
| plain_4 | 0 | 2 | 3.666601602 | 3.366669939 | 0.321331861 | 0.233370700 | 0.209912866 |
| plain_4 | 0 | 3 | 3.790442055 | 3.483556018 | 0.272484619 | 0.167412588 | 0.126218781 |
| plain_4 | 0 | 4 | 3.810333783 | 3.498336266 | 0.303609962 | 0.204869110 | 0.197282821 |
| plain_4 | 1 | 1 | 3.581901120 | 3.302487011 | 0.412154758 | 0.342964631 | 0.364153206 |
| plain_4 | 1 | 2 | 3.676486741 | 3.362395466 | 0.336243693 | 0.245222240 | 0.223142147 |
| plain_4 | 1 | 3 | 3.771494657 | 3.452304197 | 0.301848554 | 0.196825623 | 0.163144082 |
| plain_4 | 1 | 4 | 3.810237813 | 3.500585461 | 0.301910713 | 0.199021742 | 0.183973208 |
| distill_4 | 0 | 1 | 3.651228013 | 3.280306457 | 0.364511184 | 0.296823002 | 0.329532921 |
| distill_4 | 0 | 2 | 3.692774564 | 3.274716200 | 0.318945366 | 0.245275530 | 0.255880654 |
| distill_4 | 0 | 3 | 3.783632224 | 3.355220330 | 0.299429410 | 0.216533237 | 0.188258231 |
| distill_4 | 0 | 4 | 3.810440203 | 3.377345714 | 0.315705130 | 0.235099281 | 0.216490522 |
| distill_4 | 1 | 1 | 3.693623343 | 3.300636196 | 0.337790375 | 0.263512111 | 0.299414426 |
| distill_4 | 1 | 2 | 3.711429854 | 3.263636937 | 0.342381243 | 0.267102832 | 0.274726331 |
| distill_4 | 1 | 3 | 3.793437312 | 3.343037617 | 0.322999504 | 0.238706699 | 0.207205147 |
| distill_4 | 1 | 4 | 3.811081901 | 3.370566684 | 0.311154031 | 0.227018546 | 0.206924483 |
| distill_4_top | 0 | 1 | 3.631053481 | 3.274071989 | 0.378927949 | 0.309506045 | 0.334870756 |
| distill_4_top | 0 | 2 | 3.669933952 | 3.267028756 | 0.336530450 | 0.261053707 | 0.260203272 |
| distill_4_top | 0 | 3 | 3.758051899 | 3.347041974 | 0.313677743 | 0.228955668 | 0.198199525 |
| distill_4_top | 0 | 4 | 3.778622530 | 3.473922695 | 0.294435904 | 0.193545372 | 0.187070474 |
| distill_4_top | 1 | 1 | 3.669891798 | 3.294952511 | 0.355644075 | 0.281760241 | 0.313106865 |
| distill_4_top | 1 | 2 | 3.691484378 | 3.264508856 | 0.353017789 | 0.276741608 | 0.276827037 |
| distill_4_top | 1 | 3 | 3.766821663 | 3.336112067 | 0.337611417 | 0.253764360 | 0.218780100 |
| distill_4_top | 1 | 4 | 3.792622974 | 3.485759789 | 0.289290878 | 0.183588958 | 0.160146475 |
| distill_1 | 0 | 1 | 3.539418071 | 3.268415398 | 0.951804853 | 0.950369362 | 0.953892112 |
| distill_1 | 1 | 1 | 3.573851688 | 3.292289935 | 0.950987225 | 0.949850476 | 0.963868856 |

| Baseline − H34b | Seed | Tail | Full | Max abs limit |
|---|---:|---:|---:|---:|
| plain_4 | 0 | 0.000000000 | 0.000000000 | 0.000001 |
| plain_4 | 1 | -0.000000000 | -0.000000000 | 0.000001 |

| Smoke check | Measurements |
|---|---|
| logit_gradient_autograd | {"dtype": "float64", "equal_distribution_distillation_max_abs": 0.0, "layer_cases": 12, "limit": 1e-06, "max_abs": 4.440892098500626e-16, "n_bytes": 512} |
| lambda_zero_and_h34b_equivalence | {"bits_per_byte": 8.664208657786467, "h34b_max_bits_abs": 0.0, "lambda_zero_max_bits_abs": 0.0, "limit": 0, "n_bytes": 8192, "seed": 0} |
| lambda_zero_and_h34b_equivalence | {"bits_per_byte": 8.844678844063042, "h34b_max_bits_abs": 0.0, "lambda_zero_max_bits_abs": 0.0, "limit": 0, "n_bytes": 8192, "seed": 1} |
| prequential_prefix | {"arm": "plain_4", "changed_suffix_max_bits_abs": 0.0, "extended_bytes": 2048, "max_bits_abs": 0.0, "prefix_bytes": 1024} |
| finite | {"all_finite": true, "arm": "distill_4", "bits_per_byte": 15.660154960363714, "n_bytes": 8192, "wall_seconds": 52.69984225463122} |
| prequential_prefix | {"arm": "distill_4", "changed_suffix_max_bits_abs": 0.0, "extended_bytes": 2048, "max_bits_abs": 0.0, "prefix_bytes": 1024} |
| finite | {"all_finite": true, "arm": "distill_4_top", "bits_per_byte": 13.815925192469546, "n_bytes": 8192, "wall_seconds": 52.7303271163255} |
| prequential_prefix | {"arm": "distill_4_top", "changed_suffix_max_bits_abs": 0.0, "extended_bytes": 2048, "max_bits_abs": 0.0, "prefix_bytes": 1024} |
| finite | {"all_finite": true, "arm": "distill_1", "bits_per_byte": 5.1544340671216755, "n_bytes": 8192, "wall_seconds": 8.293600019067526} |
| prequential_prefix | {"arm": "distill_1", "changed_suffix_max_bits_abs": 0.0, "extended_bytes": 2048, "max_bits_abs": 0.0, "prefix_bytes": 1024} |
| wall_estimates | {"estimated_kernel_seconds": {"distill_1": 182.2324222939642, "distill_4": 1157.9555182902368, "distill_4_top": 1158.625351677074, "plain_4": 1236.6352261847169}, "limit": 5100, "method": "8192-byte one-thread rates; linear extrapolation; no assumed Kaggle speedup", "n_bytes": 180000} |

| Kernel | Wall seconds |
|---|---:|
| ezzzzzx/h34e-plain-4-s0 | 2314.111 |
| ezzzzzx/h34e-plain-4-s1 | 1876.794 |
| ezzzzzx/h34e-distill-4-s0 | 1981.689 |
| ezzzzzx/h34e-distill-4-s1 | 1946.508 |
| ezzzzzx/h34e-distill-4-top-s0 | 2149.836 |
| ezzzzzx/h34e-distill-4-top-s1 | 2236.787 |
| ezzzzzx/h34e-distill-1-s0 | 320.682 |
| ezzzzzx/h34e-distill-1-s1 | 106.179 |
