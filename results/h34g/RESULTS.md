| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|
| Tail | zgate_8 − zgate_6 | NA | NA | NA | ≤ -0.02 | 0/2 | 0/2 |
| Full | zgate_8 − zgate_6 | NA | NA | NA | report only | — | 0/2 |
| Tail | dense_6 − zgate_6 | NA | NA | NA | ≤ -0.03 | 0/2 | 0/2 |
| Full | dense_6 − zgate_6 | NA | NA | NA | report only | — | 0/2 |

| Depth | Arm | Seed 0 tail | Seed 1 tail | Mean tail |
|---:|---|---:|---:|---:|
| 1 | plain_1 | 3.264204845 | 3.285754554 | 3.274979700 |
| 2 | zgate_2 | NA | NA | NA |
| 4 | zgate_4 | 3.093923752 | 3.093068825 | 3.093496289 |
| 6 | zgate_6 | NA | NA | NA |
| 8 | zgate_8 | NA | NA | NA |

| Runs scored | Runs expected | Smoke | Tail start | Tail end |
|---:|---:|---|---:|---:|
| 0 | 8 | True | 40000 | 180000 |

| Arm | Seed | First 40 KB | Tail | Full |
|---|---:|---:|---:|---:|
| zgate_2 | 0 | NA | NA | NA |
| zgate_2 | 1 | NA | NA | NA |
| zgate_2 | Mean | NA | NA | NA |
| zgate_8 | 0 | NA | NA | NA |
| zgate_8 | 1 | NA | NA | NA |
| zgate_8 | Mean | NA | NA | NA |
| dense_6 | 0 | NA | NA | NA |
| dense_6 | 1 | NA | NA | NA |
| dense_6 | Mean | NA | NA | NA |
| zgate_6 | 0 | NA | NA | NA |
| zgate_6 | 1 | NA | NA | NA |
| zgate_6 | Mean | NA | NA | NA |

| Arm | Seed | Layer | Own tail bits |
|---|---:|---:|---:|

| Arm | Seed | Start | End | Mean mixer weights |
|---|---:|---:|---:|---|

| zgate_6 − H34f | Seed | First 40 KB | Tail | Full | Max abs limit |
|---|---:|---:|---:|---:|---:|
| zgate_6 | 0 | NA | NA | NA | 0.000001 |
| zgate_6 | 1 | NA | NA | NA | 0.000001 |

| Smoke check | Measurements |
|---|---|
| h34f_equivalence | {"bits_per_byte": 5.136672077864404, "max_bits_abs": 0.0, "n_bytes": 8192, "reference_bits": 5.136672077864404, "seed": 0} |
| layer_2_equivalence | {"max_features_abs": 0.0, "max_input_abs": 0.0, "max_logits_abs": 0.0, "n_bytes": 2048, "seed": 0} |
| prequential_prefix | {"arm": "dense_6", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 0} |
| prequential_prefix | {"arm": "zgate_8", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 0} |
| h34f_equivalence | {"bits_per_byte": 5.276438589593237, "max_bits_abs": 0.0, "n_bytes": 8192, "reference_bits": 5.276438589593237, "seed": 1} |
| layer_2_equivalence | {"max_features_abs": 0.0, "max_input_abs": 0.0, "max_logits_abs": 0.0, "n_bytes": 2048, "seed": 1} |
| prequential_prefix | {"arm": "dense_6", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 1} |
| prequential_prefix | {"arm": "zgate_8", "changed_suffix_max_bits_abs": 0.0, "max_bits_abs": 0.0, "n_bytes": 1024, "seed": 1} |
| finite | {"all_finite": true, "arm": "zgate_2", "bits_per_byte": 5.137991005949128, "n_bytes": 8192, "wall_seconds": 44.49153963942081} |
| finite | {"all_finite": true, "arm": "zgate_8", "bits_per_byte": 5.139973583749612, "n_bytes": 8192, "wall_seconds": 247.4849468022585} |
| finite | {"all_finite": true, "arm": "dense_6", "bits_per_byte": 5.1429546675619315, "n_bytes": 8192, "wall_seconds": 196.5380563363433} |
| finite | {"all_finite": true, "arm": "zgate_6", "bits_per_byte": 5.136672077864404, "n_bytes": 8192, "wall_seconds": 181.47757722437382} |
| checkpoint_equivalence | {"arm": "zgate_8", "committed_bytes": 4096, "max_bits_abs": 0.0, "n_bytes": 8192, "pending_bytes": 0, "seed": 0, "split_at": 4096, "state_exact": true} |
| checkpoint_equivalence | {"arm": "zgate_8", "committed_bytes": 4096, "max_bits_abs": 0.0, "n_bytes": 8192, "pending_bytes": 16, "seed": 0, "split_at": 4112, "state_exact": true} |
| checkpoint_equivalence | {"arm": "zgate_8", "committed_bytes": 4096, "max_bits_abs": 0.0, "n_bytes": 8192, "pending_bytes": 0, "seed": 1, "split_at": 4096, "state_exact": true} |
| checkpoint_equivalence | {"arm": "zgate_8", "committed_bytes": 4096, "max_bits_abs": 0.0, "n_bytes": 8192, "pending_bytes": 16, "seed": 1, "split_at": 4112, "state_exact": true} |
| wall_estimates | {"estimated_kernel_seconds": {"dense_6": 4318.4631519216055, "zgate_2": 977.5973065302423, "zgate_6": 4271.879969917336, "zgate_8": 5437.901663135563}, "estimated_segment_seconds": {"dense_6": 4318.4631519216055, "zgate_2": 977.5973065302423, "zgate_6": 4271.879969917336, "zgate_8": 2718.9508315677813}, "limit": 5100, "n_bytes": 180000, "split_zgate_8": true} |

| Kernel | Segment wall seconds | Run wall seconds |
|---|---:|---:|
