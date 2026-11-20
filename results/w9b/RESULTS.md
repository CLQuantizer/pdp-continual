| T14 gap | 72 | 73 | 74 | 75 | 76 | 77 | 78 | 79 | 80 | 81 | 82 | 83 | Mean | Threshold <= | Count <= | Required count | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_night - fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | -0.100000000 | NA | 8 | 0 |
| pressure_bits - fixed_night | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0.050000000 | NA | NA | 0 |
| pressure_bits - fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | -0.070000000 | NA | 8 | 0 |
| pressure - fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 |
| pressure_opp - fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 |

| Mean threshold | Mean | Comparison | Threshold | Meets (0/1) | N |
|---|---|---|---|---|---|
| pressure_bits - fixed_night | NA | <= | 0.050000000 | NA | 0 |
| pressure_bits night_sleep_share | NA | >= | 0.600000000 | NA | 0 |


## night_sleep_share

| Arm | 72 | 73 | 74 | 75 | 76 | 77 | 78 | 79 | 80 | 81 | 82 | 83 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| fixed_night | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_opp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_bits | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## sleep_share

| Arm | 72 | 73 | 74 | 75 | 76 | 77 | 78 | 79 | 80 | 81 | 82 | 83 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| fixed_night | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_opp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_bits | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## T14

| Arm | 72 | 73 | 74 | 75 | 76 | 77 | 78 | 79 | 80 | 81 | 82 | 83 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| fixed_night | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_opp | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_bits | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

| Scored lives | Expected | Smoke workers | Smoke ok | Smoke wall s |
|---|---|---|---|---|
| 0 | 60 | 1 | 1 | 848.819822545 |

## Smoke output

```
s24 age_nosleep p=128 checkpoint t=32/256 W14 11.8096
s24 age_nosleep p=128 checkpoint t=64/256 W14 8.6276
s24 age_nosleep p=128 checkpoint t=96/256 W14 7.0769
s24 age_nosleep p=128 checkpoint t=128/256 W14 4.4390
s24 age_nosleep p=128 checkpoint t=160/256 W14 3.2648
s24 age_nosleep p=128 checkpoint t=192/256 W14 3.3518
s24 age_nosleep p=128 checkpoint t=224/256 W14 2.9221
s24 age_nosleep p=128 checkpoint t=256/256 W14 3.5367
s24 age_nosleep p=128 t=256/256
BASELINE nosleep {"T14": 5.6285469390612395, "reference_T14": 5.6285469390612395, "reference": "M9", "abs": 0.0, "checkpoint_max_abs": 0.0, "sleep_share": 0.0, "wall_s": 43.9369205031544}
s24 age_fixed20 p=128 checkpoint t=32/256 W14 11.8096
s24 age_fixed20 p=128 checkpoint t=64/256 W14 10.6724
s24 age_fixed20 p=128 checkpoint t=96/256 W14 6.6363
s24 age_fixed20 p=128 checkpoint t=128/256 W14 4.4603
s24 age_fixed20 p=128 checkpoint t=160/256 W14 4.2375
s24 age_fixed20 p=128 checkpoint t=192/256 W14 3.8962
s24 age_fixed20 p=128 checkpoint t=224/256 W14 3.4717
s24 age_fixed20 p=128 checkpoint t=256/256 W14 2.3318
s24 age_fixed20 p=128 t=256/256
BASELINE fixed20_compat {"T14": 5.939474666464639, "reference_T14": 5.939474666464639, "reference": "M9", "abs": 0.0, "checkpoint_max_abs": 0.0, "sleep_share": 0.1875, "wall_s": 40.89976096339524}
BASELINE fixed_random {"T14": 5.943012759003068, "reference_T14": 5.943012759003068, "reference": "W9", "abs": 0.0, "checkpoint_max_abs": 0.0, "sleep_share": 0.25, "wall_s": 38.33353410940617}
INPUT_BITS {"day": {"mean_b_t": 3.277640110576488, "wake_decisions": 587}, "night": {"mean_b_t": 7.349776614923628, "wake_decisions": 189}, "night_minus_day": 4.0721365043471405}
FINITE fixed_random {"T14": 3.154089828153281, "sleep_share": 0.25, "night_sleep_share": 0.359375, "wall_s": 75.86817475222051, "finite": true}
FINITE fixed_night {"T14": 3.5284190412627128, "sleep_share": 0.25, "night_sleep_share": 1.0, "wall_s": 77.68010063283145, "finite": true}
FINITE pressure {"T14": 3.4184926070000623, "sleep_share": 0.1875, "night_sleep_share": 0.3333333333333333, "wall_s": 76.06646045669913, "finite": true}
FINITE pressure_opp {"T14": 3.470530973502097, "sleep_share": 0.1875, "night_sleep_share": 0.17708333333333334, "wall_s": 100.54174679983407, "finite": true}
FINITE pressure_bits {"T14": 3.416588824641649, "sleep_share": 0.21875, "night_sleep_share": 0.24107142857142858, "wall_s": 90.31854634732008, "finite": true}
WALL_ESTIMATE {"kernel_wall_s": 3363.800231911242, "limit_s": 5100, "within_limit": true, "factor": 8, "pressure_bits_L1024_wall_s": 186.4697876572609}
SMOKE_OK
```

## Kernel walls

| Seed | Kernel | Workers | Wall s |
|---|---|---|---|

| Estimate s | Limit s | Within limit (0/1) |
|---|---|---|
| 3363.800231911 | 5100 | 1 |
