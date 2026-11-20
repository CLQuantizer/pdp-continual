| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|
| Tail | dfa_4 − plain_4 | 0.279110572 | 0.281270516 | 0.280190544 | ≤ -0.03 | 0/2 | 2/2 |
| Tail | dfa_only_4 − plain_4 | 0.304159705 | 0.315416540 | 0.309788123 | report only | — | 2/2 |
| Full | dfa_4 − plain_4 | 0.332267477 | 0.353187871 | 0.342727674 | report only | — | 2/2 |
| Full | dfa_only_4 − plain_4 | 0.449543685 | 0.483944206 | 0.466743945 | report only | — | 2/2 |

H34d: 6/6 scored kernels. Tail bytes 40,000–180,000; full bytes 0–180,000. Bar requires both seeds.
Smoke passed: True.


| Arm | Tail seed 0 | Tail seed 1 | Tail mean | Full seed 0 | Full seed 1 | Full mean |
|---|---:|---:|---:|---:|---:|---:|
| plain_4 | 3.091224982 | 3.091462939 | 3.091343961 | 3.745828744 | 3.769168530 | 3.757498637 |
| dfa_4 | 3.370335554 | 3.372733455 | 3.371534504 | 4.078096221 | 4.122356401 | 4.100226311 |
| dfa_only_4 | 3.395384688 | 3.406879479 | 3.401132083 | 4.195372429 | 4.253112736 | 4.224242582 |

| Arm | Seed | Layer | Own full | Own tail | Mean gate tail | Final gate |
|---|---:|---:|---:|---:|---:|---:|
| plain_4 | 0 | 1 | 3.549181220 | 3.280796641 | 0.374364861 | 0.388745010 |
| plain_4 | 0 | 2 | 3.666601602 | 3.366669939 | 0.233370700 | 0.209912866 |
| plain_4 | 0 | 3 | 3.790442055 | 3.483556018 | 0.167412588 | 0.126218781 |
| plain_4 | 0 | 4 | 3.810333783 | 3.498336266 | 0.204869110 | 0.197282821 |
| plain_4 | 1 | 1 | 3.581901120 | 3.302487011 | 0.342964631 | 0.364153206 |
| plain_4 | 1 | 2 | 3.676486741 | 3.362395466 | 0.245222240 | 0.223142147 |
| plain_4 | 1 | 3 | 3.771494657 | 3.452304197 | 0.196825623 | 0.163144082 |
| plain_4 | 1 | 4 | 3.810237813 | 3.500585461 | 0.199021742 | 0.183973208 |
| dfa_4 | 0 | 1 | 3.752960659 | 3.509215461 | 0.555544021 | 0.487682760 |
| dfa_4 | 0 | 2 | 4.435333428 | 4.152976757 | 0.102856142 | 0.144889474 |
| dfa_4 | 0 | 3 | 4.307901572 | 4.049011905 | 0.187273750 | 0.164768517 |
| dfa_4 | 0 | 4 | 4.805679002 | 4.576420934 | 0.053923997 | 0.116906397 |
| dfa_4 | 1 | 1 | 3.766232880 | 3.512354987 | 0.523164225 | 0.467401624 |
| dfa_4 | 1 | 2 | 4.201544863 | 3.919613762 | 0.192646436 | 0.192824990 |
| dfa_4 | 1 | 3 | 4.627958129 | 4.310287884 | 0.081354271 | 0.117538936 |
| dfa_4 | 1 | 4 | 4.506929997 | 4.252302612 | 0.112881222 | 0.142236099 |
| dfa_only_4 | 0 | 1 | 3.717603741 | 3.467030787 | 0.713710663 | 0.726992011 |
| dfa_only_4 | 0 | 2 | 5.777253090 | 5.678558993 | 0.057312295 | 0.057988789 |
| dfa_only_4 | 0 | 3 | 6.004091201 | 5.709374199 | 0.054566506 | 0.036353238 |
| dfa_only_4 | 0 | 4 | 9.482762874 | 10.051880398 | 0.005907714 | -0.008905983 |
| dfa_only_4 | 1 | 1 | 3.735105790 | 3.475782510 | 0.691002739 | 0.721017540 |
| dfa_only_4 | 1 | 2 | 5.324611643 | 5.247761148 | 0.099478508 | 0.042898964 |
| dfa_only_4 | 1 | 3 | 6.590104673 | 6.323619772 | 0.017848083 | 0.025191514 |
| dfa_only_4 | 1 | 4 | 6.370143107 | 6.318647268 | 0.041407471 | 0.020327495 |

| Smoke check | Measurements |
|---|---|
| dfa_trace_autograd | {"limit": 0.0001, "max_relative_error": 7.898248949811026e-16, "n_bytes": 512, "norms": {"alpha": 0.002944000213408588, "dfa_norm": 0.08355070016752951, "only": true, "own_norm": 0.08355070016752945, "raw_dfa_norm": 28.37999120618057}, "readout_own_max_abs": 1.1102230246251565e-16} |
| alpha_zero_equivalence | {"dfa_max_bits_abs": 0.0, "h34b_max_bits_abs": 0.0, "limit": 1e-09, "n_bytes": 8192, "seed": 0} |
| alpha_zero_equivalence | {"dfa_max_bits_abs": 0.0, "h34b_max_bits_abs": 0.0, "limit": 1e-09, "n_bytes": 8192, "seed": 1} |
| equal_norm_per_chunk | {"all_finite": true, "arm": "dfa_4", "chunks": 64, "layer_chunks": 256, "limit": 1e-06, "max_relative_error": 9.135269127312308e-07, "n_bytes": 8192, "zero_own_count": 4} |
| prequential_prefix | {"arm": "dfa_4", "changed_suffix_max_bits_abs": 0.0, "extended_bytes": [2048, 8192], "max_bits_abs": 0.0, "prefix_bytes": 1024} |
| equal_norm_per_chunk | {"all_finite": true, "arm": "dfa_only_4", "chunks": 64, "layer_chunks": 256, "limit": 1e-06, "max_relative_error": 8.244146099789287e-07, "n_bytes": 8192, "zero_own_count": 4} |
| prequential_prefix | {"arm": "dfa_only_4", "changed_suffix_max_bits_abs": 0.0, "extended_bytes": [2048, 8192], "max_bits_abs": 0.0, "prefix_bytes": 1024} |
| wall_estimates | {"estimated_kernel_seconds": {"dfa_4": 1091.3575108793339, "dfa_only_4": 1074.1844748531548, "plain_4": 1089.3264785590873}, "limit": 5100, "method": "Each arm measured on 8192 bytes, one CPU thread; linear extrapolation; no assumed Kaggle speedup", "n_bytes": 180000} |
| float32 own + DFA vs autograd, 512 bytes | max relative error 8.73250486313e-07; limit 1e-4 |

Feedback RNG uses one-based layer k=1..4: default_rng(7000 + 100·seed + k), normal(0, 1/16), shape (256,256).
Alpha uses the Frobenius norm over each chunk of mean-loss hidden gradients. If the own norm is zero, alpha is zero. Readouts learn only the own loss.
dfa_only_4 replaces the state gradient by DFA for layers 2–4; layer 1 keeps own + DFA. All chunk norms are retained in each results.json.
Prefix smoke ends on a chunk boundary, following the inherited prequential chunk update protocol.

| Kernel | Wall seconds |
|---|---:|
| ezzzzzx/h34d-plain-4-s0 | 2242.449 |
| ezzzzzx/h34d-plain-4-s1 | 1819.689 |
| ezzzzzx/h34d-dfa-4-s0 | 2189.762 |
| ezzzzzx/h34d-dfa-4-s1 | 2108.723 |
| ezzzzzx/h34d-dfa-only-4-s0 | 810.289 |
| ezzzzzx/h34d-dfa-only-4-s1 | 2298.451 |

Validation: own_norm and dfa_norm relative tolerance 1e-5 (previously 1e-6), for float32 chunk norms.
