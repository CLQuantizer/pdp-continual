| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|
| Full | zgate_4 − plain_4 | NA | NA | NA | ≤ -0.10 | 0/2 | 0/2 |
| Tail | zgate_4 − plain_4 | NA | NA | NA | ≤ 0.01 | 0/2 | 0/2 |
| Full | zgate_6 − plain_6 | NA | NA | NA | report only | — | 0/2 |
| Tail | zgate_6 − plain_6 | NA | NA | NA | report only | — | 0/2 |
| Full | zgate_6 − plain_1 | NA | NA | NA | ≤ -0.05 | 0/2 | 0/2 |
| Tail | zgate_6 − plain_1 | NA | NA | NA | report only | — | 0/2 |

| Kernels scored | Kernels expected | Smoke | Tail start | Tail end |
|---:|---:|---|---:|---:|
| 0 | 10 | True | 40000 | 180000 |

| Arm | Seed | First 40 KB | Tail | Full |
|---|---:|---:|---:|---:|
| plain_4 | 0 | NA | NA | NA |
| plain_4 | 1 | NA | NA | NA |
| plain_4 | Mean | NA | NA | NA |
| zgate_4 | 0 | NA | NA | NA |
| zgate_4 | 1 | NA | NA | NA |
| zgate_4 | Mean | NA | NA | NA |
| zgate_6 | 0 | NA | NA | NA |
| zgate_6 | 1 | NA | NA | NA |
| zgate_6 | Mean | NA | NA | NA |
| plain_6 | 0 | NA | NA | NA |
| plain_6 | 1 | NA | NA | NA |
| plain_6 | Mean | NA | NA | NA |
| plain_1 | 0 | NA | NA | NA |
| plain_1 | 1 | NA | NA | NA |
| plain_1 | Mean | NA | NA | NA |

| Arm | Seed | Start | End | Mean mixer weights |
|---|---:|---:|---:|---|

| Baseline − H34b | Seed | First 40 KB | Tail | Full | Max abs limit |
|---|---:|---:|---:|---:|---:|
| plain_1 | 0 | NA | NA | NA | 0.000001 |
| plain_1 | 1 | NA | NA | NA | 0.000001 |
| plain_4 | 0 | NA | NA | NA | 0.000001 |
| plain_4 | 1 | NA | NA | NA | 0.000001 |
| plain_6 | 0 | NA | NA | NA | 0.000001 |
| plain_6 | 1 | NA | NA | NA | 0.000001 |

| Smoke check | Measurements |
|---|---|
| first_chunk_prediction | {"max_bits_abs": 0.0, "max_logits_abs": 0.0, "max_probability_abs": 0.0, "n_bytes": 128, "seed": 0} |
| h34b_equivalence | {"arm": "plain_4", "bits_per_byte": 8.664208657786467, "limit": 0, "max_bits_abs": 0.0, "n_bytes": 8192, "reference_bits": 8.664208657786467, "seed": 0} |
| h34b_equivalence | {"arm": "plain_1", "bits_per_byte": 5.142095267713781, "limit": 0, "max_bits_abs": 0.0, "n_bytes": 8192, "reference_bits": 5.142095267713781, "seed": 0} |
| first_chunk_prediction | {"max_bits_abs": 0.0, "max_logits_abs": 0.0, "max_probability_abs": 0.0, "n_bytes": 128, "seed": 1} |
| h34b_equivalence | {"arm": "plain_4", "bits_per_byte": 8.844678844063042, "limit": 0, "max_bits_abs": 0.0, "n_bytes": 8192, "reference_bits": 8.844678844063042, "seed": 1} |
| h34b_equivalence | {"arm": "plain_1", "bits_per_byte": 5.295348220502141, "limit": 0, "max_bits_abs": 0.0, "n_bytes": 8192, "reference_bits": 5.295348220502141, "seed": 1} |
| prequential_prefix | {"arm": "zgate_4", "changed_suffix_max_bits_abs": 0.0, "extended_bytes": 2048, "max_bits_abs": 0.0, "prefix_bytes": 1024, "seed": 0} |
| prequential_prefix | {"arm": "zgate_4", "changed_suffix_max_bits_abs": 0.0, "extended_bytes": 2048, "max_bits_abs": 0.0, "prefix_bytes": 1024, "seed": 1} |
| finite | {"all_finite": true, "arm": "plain_4", "bits_per_byte": 8.664208657786467, "n_bytes": 8192, "wall_seconds": 117.27971909567714} |
| finite | {"all_finite": true, "arm": "zgate_4", "bits_per_byte": 5.133850940618147, "n_bytes": 8192, "wall_seconds": 116.97459700983018} |
| finite | {"all_finite": true, "arm": "zgate_6", "bits_per_byte": 5.136672077864404, "n_bytes": 8192, "wall_seconds": 184.7723593749106} |
| finite | {"all_finite": true, "arm": "plain_6", "bits_per_byte": 11.727748600175858, "n_bytes": 8192, "wall_seconds": 184.93756417836994} |
| finite | {"all_finite": true, "arm": "plain_1", "bits_per_byte": 5.142095267713781, "n_bytes": 8192, "wall_seconds": 14.279282432049513} |
| wall_estimates | {"estimated_kernel_seconds": {"plain_1": 313.75376437608793, "plain_4": 2576.9469527858746, "plain_6": 4063.5695254036364, "zgate_4": 2570.2426100792763, "zgate_6": 4059.9395370463753}, "limit": 5100, "method": "8192-byte one-thread rates; slowest measured rate per arm; linear extrapolation; no assumed Kaggle speedup", "n_bytes": 180000} |

| Kernel | Wall seconds |
|---|---:|
