# H24b floors

2026-11-10 04:30 UTC+8

N = 524288 symbols per life; seeds 6, 7, 8; no fitted selection. KT is the prequential fixed-share mixture of orders 0–8 (α = 0.001). Alphabet: H20's 41 symbols plus `=` (42; uniform 5.3923 bits/symbol). lzma uses preset 9 with PRESET_EXTREME, including container overhead.

F has 300 unique four-letter keys with fixed four-digit values per seed, 16 uniformly sampled records per chunk. F uses an independent seeded RNG. Under an identical sequence of original-process calls, their symbols are unchanged (smoke-checked); whole H20 streams are not unchanged because eight-way seasons alter scheduling and subsequent shared-RNG consumption.

## Whole-stream floors (bits/symbol)

| World | Seed | KT mixture | lzma -9e | Seconds |
|---|---:|---:|---:|---:|
| slow | 6 | 1.8634 | 0.6473 | 65.31 |
| slow | 7 | 1.8163 | 0.7462 | 67.00 |
| slow | 8 | 1.8665 | 0.8253 | 65.53 |
| fast | 6 | 1.7261 | 0.9587 | 66.03 |
| fast | 7 | 1.6793 | 0.9351 | 66.43 |
| fast | 8 | 1.6748 | 0.9028 | 66.22 |
| slow | mean ± population std | 1.8487 ± 0.0230 | 0.7396 ± 0.0728 | — |
| fast | mean ± population std | 1.6934 ± 0.0232 | 0.9322 ± 0.0229 | — |

## Per-process floors (bits/symbol)

KT entries attribute whole-stream prequential bits by the true process. lzma entries independently compress that process's concatenated observations across visits; these are separate codes, not additive attributions of the whole-stream lzma code. — means the process was absent.

| World | Seed | Process | Symbols | KT mixture | lzma -9e (isolated) |
|---|---:|---|---:|---:|---:|
| slow | 6 | G | 49152 | 0.9783 | 1.1055 |
| slow | 6 | C1 | 114688 | 3.1971 | 0.3016 |
| slow | 6 | C7 | 81920 | 3.2947 | 0.8586 |
| slow | 6 | E30 | 81920 | 1.1756 | 0.9750 |
| slow | 6 | E110 | 16384 | 1.1649 | 1.0449 |
| slow | 6 | M7 | 49152 | 0.4129 | 0.3366 |
| slow | 6 | M37 | 98304 | 0.4649 | 0.3190 |
| slow | 6 | F | 32768 | 3.3844 | 1.5723 |
| slow | 7 | G | 32768 | 1.0059 | 1.1025 |
| slow | 7 | C1 | 65536 | 3.2966 | 0.3521 |
| slow | 7 | C7 | 81920 | 3.4378 | 0.8086 |
| slow | 7 | E30 | 32768 | 1.1962 | 1.0723 |
| slow | 7 | E110 | 65536 | 1.1316 | 0.8740 |
| slow | 7 | M7 | 98304 | 0.3756 | 0.3057 |
| slow | 7 | M37 | 49152 | 0.5811 | 0.3320 |
| slow | 7 | F | 98304 | 2.4696 | 1.3418 |
| slow | 8 | G | 32768 | 1.0885 | 1.1338 |
| slow | 8 | C1 | 65536 | 3.3241 | 0.1768 |
| slow | 8 | C7 | 81920 | 3.4081 | 0.8609 |
| slow | 8 | E30 | 114688 | 1.1742 | 0.9601 |
| slow | 8 | E110 | 49152 | 1.1511 | 0.9251 |
| slow | 8 | M7 | 81920 | 0.3711 | 0.3148 |
| slow | 8 | M37 | 16384 | 0.7534 | 0.4023 |
| slow | 8 | F | 81920 | 2.5864 | 1.3699 |
| fast | 6 | G | 59392 | 0.9355 | 1.0830 |
| fast | 6 | C1 | 67584 | 3.3370 | 0.5099 |
| fast | 6 | C7 | 61440 | 3.4462 | 1.1422 |
| fast | 6 | E30 | 69632 | 1.1925 | 0.9991 |
| fast | 6 | E110 | 51200 | 1.1583 | 0.9437 |
| fast | 6 | M7 | 59392 | 0.4220 | 0.3254 |
| fast | 6 | M37 | 77824 | 0.5197 | 0.3261 |
| fast | 6 | F | 77824 | 2.6249 | 1.3742 |
| fast | 7 | G | 71680 | 0.8861 | 1.0067 |
| fast | 7 | C1 | 61440 | 3.3201 | 0.3750 |
| fast | 7 | C7 | 65536 | 3.4165 | 1.0088 |
| fast | 7 | E30 | 55296 | 1.1926 | 1.0162 |
| fast | 7 | E110 | 75776 | 1.1435 | 0.9037 |
| fast | 7 | M7 | 57344 | 0.4137 | 0.3181 |
| fast | 7 | M37 | 75776 | 0.5433 | 0.3269 |
| fast | 7 | F | 61440 | 2.7919 | 1.4094 |
| fast | 8 | G | 67584 | 0.9049 | 1.0436 |
| fast | 8 | C1 | 57344 | 3.3041 | 0.1992 |
| fast | 8 | C7 | 71680 | 3.3749 | 0.9830 |
| fast | 8 | E30 | 59392 | 1.1989 | 1.0027 |
| fast | 8 | E110 | 57344 | 1.1538 | 0.9442 |
| fast | 8 | M7 | 69632 | 0.4040 | 0.3171 |
| fast | 8 | M37 | 77824 | 0.5320 | 0.3146 |
| fast | 8 | F | 63488 | 2.8138 | 1.4103 |

## F value digits by prior key appearances

Buckets count earlier observed records of the same key across all visits, before scoring each value. Means use only the four value positions, including observed digits of a final partial record. Exact recall is a value-only reference: uniform decimal digits on the first appearance, then probability 1 for the remembered digits. Its store is unpriced here; this is not two-part T. lzma does not expose per-digit code lengths.

| World | Seed | Prior appearances | Value digits | KT bits/digit | Exact-recall bits/digit |
|---|---:|---|---:|---:|---:|
| slow | 6 | 0 | 1200 | 5.2431 | 3.3219 |
| slow | 6 | 1 | 1200 | 3.9249 | 0.0000 |
| slow | 6 | 2–4 | 3564 | 3.0002 | 0.0000 |
| slow | 6 | 5+ | 7143 | 1.9255 | 0.0000 |
| slow | 7 | 0 | 1200 | 4.7442 | 3.3219 |
| slow | 7 | 1 | 1200 | 3.9164 | 0.0000 |
| slow | 7 | 2–4 | 3600 | 3.0490 | 0.0000 |
| slow | 7 | 5+ | 33320 | 1.5132 | 0.0000 |
| slow | 8 | 0 | 1200 | 4.2691 | 3.3219 |
| slow | 8 | 1 | 1200 | 3.8684 | 0.0000 |
| slow | 8 | 2–4 | 3600 | 3.1642 | 0.0000 |
| slow | 8 | 5+ | 26768 | 1.5929 | 0.0000 |
| fast | 6 | 0 | 1200 | 4.8626 | 3.3219 |
| fast | 6 | 1 | 1200 | 3.9052 | 0.0000 |
| fast | 6 | 2–4 | 3600 | 3.0555 | 0.0000 |
| fast | 6 | 5+ | 25128 | 1.6977 | 0.0000 |
| fast | 7 | 0 | 1200 | 4.7617 | 3.3219 |
| fast | 7 | 1 | 1200 | 3.8941 | 0.0000 |
| fast | 7 | 2–4 | 3600 | 3.0419 | 0.0000 |
| fast | 7 | 5+ | 18576 | 1.8062 | 0.0000 |
| fast | 8 | 0 | 1200 | 4.4703 | 3.3219 |
| fast | 8 | 1 | 1200 | 3.8850 | 0.0000 |
| fast | 8 | 2–4 | 3600 | 3.0741 | 0.0000 |
| fast | 8 | 5+ | 19395 | 1.7567 | 0.0000 |

Smoke: six lives at N = 20000 plus deterministic, format, RNG-isolation, visit-boundary and exposure-bucket assertions passed (22.79 seconds). Full-run wall time: 396.54 seconds.

F's first value exposure costs the exact-recall reference 3.3219 bits/digit; every repeat costs 0.
KT first-exposure value costs range from 4.2691 to 5.2431 bits/digit; at 5+ prior appearances they range from 1.5132 to 1.9255.
lzma uses fewer whole-stream bits than KT in 6 of 6 lives.
