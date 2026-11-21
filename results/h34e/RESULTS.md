| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |
|---|---|---:|---:|---:|---:|---:|---:|
| Tail | distill_4 − plain_4 | NA | NA | NA | ≤ -0.03 | 0/2 | 0/2 |
| Tail | distill_4_top − plain_4 | NA | NA | NA | report only | — | 0/2 |
| Full | distill_4 − plain_4 | NA | NA | NA | report only | — | 0/2 |
| Full | distill_4_top − plain_4 | NA | NA | NA | report only | — | 0/2 |

| Kernels scored | Kernels expected | Smoke | Tail start | Tail end |
|---:|---:|---|---:|---:|
| 0 | 8 | True | 40000 | 180000 |

| Arm | Tail seed 0 | Tail seed 1 | Tail mean | Full seed 0 | Full seed 1 | Full mean |
|---|---:|---:|---:|---:|---:|---:|
| plain_4 | NA | NA | NA | NA | NA | NA |
| distill_4 | NA | NA | NA | NA | NA | NA |
| distill_4_top | NA | NA | NA | NA | NA | NA |
| distill_1 | NA | NA | NA | NA | NA | NA |

| Arm | Seed | Layer | Own full | Own tail | Mean mixer full | Mean mixer tail | Final mixer |
|---|---:|---:|---:|---:|---:|---:|---:|

| Baseline − H34b | Seed | Tail | Full | Max abs limit |
|---|---:|---:|---:|---:|
| plain_4 | 0 | NA | NA | 0.000001 |
| plain_4 | 1 | NA | NA | 0.000001 |

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
