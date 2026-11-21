| T14 gap | 84 | 85 | 86 | 87 | 88 | 89 | 90 | 91 | 92 | 93 | 94 | 95 | Mean | Threshold <= | Count <= | Required count | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pressure_eyelid - fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | -.07 | NA | 8 | 0 |
| pressure_eyelid - fixed_night | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | .05 | NA | NA | 0 |

| Mean threshold | Mean | Comparison | Threshold | Meets (0/1) | N |
|---|---|---|---|---|---|
| pressure_eyelid night_sleep_share | NA | >= | .60 | NA | 0 |
| pressure_eyelid - fixed_night | NA | <= | .05 | NA | 0 |

## sleep_share

| Arm | 84 | 85 | 86 | 87 | 88 | 89 | 90 | 91 | 92 | 93 | 94 | 95 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| fixed_night | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_bits | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_eyelid | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## night_sleep_share

| Arm | 84 | 85 | 86 | 87 | 88 | 89 | 90 | 91 | 92 | 93 | 94 | 95 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| fixed_night | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_bits | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_eyelid | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## T14

| Arm | 84 | 85 | 86 | 87 | 88 | 89 | 90 | 91 | 92 | 93 | 94 | 95 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed_random | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| fixed_night | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_bits | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |
| pressure_eyelid | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

| Scored lives | Expected | Smoke workers | Smoke ok | Smoke wall s | Staged kernels |
|---|---|---|---|---|---|
| 0 | 48 | 1 | 0 | 501.272098037 | 0 |

## Smoke L1024

| Seed | Arm | L | Sleep | Night sleep | sleep_share | night_sleep_share | Stop threshold >= | Meets (0/1) | Wall s |
|---|---|---|---|---|---|---|---|---|---|
| 84 | pressure_eyelid | 1024 | 248 | 9 | 0.242187500 | 0.036290323 | .50 | 0 | 224.425248590 |

| Onset S >= | Eyelid checksum decisions | Intact | Mismatches |
|---|---|---|---|
| 96 | 248 | 248 | 0 |

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
BASELINE nosleep {"T14": 5.6285469390612395, "reference_T14": 5.6285469390612395, "reference": "M9", "abs": 0.0, "checkpoint_max_abs": 0.0, "sleep_share": 0.0, "wall_s": 72.01759328413755}
s24 age_fixed20 p=128 checkpoint t=32/256 W14 11.8096
s24 age_fixed20 p=128 checkpoint t=64/256 W14 10.6724
s24 age_fixed20 p=128 checkpoint t=96/256 W14 6.6363
s24 age_fixed20 p=128 checkpoint t=128/256 W14 4.4603
s24 age_fixed20 p=128 checkpoint t=160/256 W14 4.2375
s24 age_fixed20 p=128 checkpoint t=192/256 W14 3.8962
s24 age_fixed20 p=128 checkpoint t=224/256 W14 3.4717
s24 age_fixed20 p=128 checkpoint t=256/256 W14 2.3318
s24 age_fixed20 p=128 t=256/256
BASELINE fixed20_compat {"T14": 5.939474666464639, "reference_T14": 5.939474666464639, "reference": "M9", "abs": 0.0, "checkpoint_max_abs": 0.0, "sleep_share": 0.1875, "wall_s": 40.385598291642964}
BASELINE fixed_random {"T14": 5.943012759003068, "reference_T14": 5.943012759003068, "reference": "W9", "abs": 0.0, "checkpoint_max_abs": 0.0, "sleep_share": 0.25, "wall_s": 40.70611908007413}
TRACE {"t": 180, "night": 0, "S": 28, "b_t": 6.858693203417928, "m_t": 6.858305658822369, "asleep": 0}
TRACE {"t": 181, "night": 0, "S": 29, "b_t": 7.300595594644804, "m_t": 6.858693203417928, "asleep": 0}
TRACE {"t": 182, "night": 0, "S": 30, "b_t": 7.573381984127553, "m_t": 6.8675491542959595, "asleep": 0}
TRACE {"t": 183, "night": 0, "S": 31, "b_t": 7.676776260399274, "m_t": 6.8764051051739905, "asleep": 0}
TRACE {"t": 184, "night": 0, "S": 32, "b_t": 6.919539658598841, "m_t": 6.897972381886415, "asleep": 0}
TRACE {"t": 185, "night": 0, "S": 33, "b_t": 6.499877058439435, "m_t": 6.8764051051739905, "asleep": 0}
TRACE {"t": 186, "night": 0, "S": 34, "b_t": 6.027394210008993, "m_t": 6.8675491542959595, "asleep": 0}
TRACE {"t": 187, "night": 0, "S": 35, "b_t": 5.613500083064054, "m_t": 6.858693203417928, "asleep": 0}
TRACE {"t": 188, "night": 0, "S": 36, "b_t": 5.200168309634018, "m_t": 6.858305658822369, "asleep": 0}
TRACE {"t": 189, "night": 0, "S": 37, "b_t": 4.8526046279163095, "m_t": 6.85791811422681, "asleep": 0}
TRACE {"t": 190, "night": 0, "S": 38, "b_t": 4.562441710080589, "m_t": 6.857612433391271, "asleep": 0}
TRACE {"t": 191, "night": 0, "S": 39, "b_t": 4.2723456961030175, "m_t": 6.857306752555733, "asleep": 0}
TRACE {"t": 192, "night": 1, "S": 40, "b_t": 3.9975591987027728, "m_t": 6.8402832577722545, "asleep": 0}
TRACE {"t": 193, "night": 1, "S": 41, "b_t": 4.802371116144043, "m_t": 6.823259762988776, "asleep": 0}
TRACE {"t": 194, "night": 1, "S": 42, "b_t": 5.322338610133013, "m_t": 6.769355311830655, "asleep": 0}
TRACE {"t": 195, "night": 1, "S": 43, "b_t": 5.71106754534366, "m_t": 6.715450860672534, "asleep": 0}
TRACE {"t": 196, "night": 1, "S": 44, "b_t": 6.175635016164589, "m_t": 6.713459629356409, "asleep": 0}
TRACE {"t": 197, "night": 1, "S": 45, "b_t": 6.574548449792046, "m_t": 6.711468398040284, "asleep": 0}
TRACE {"t": 198, "night": 1, "S": 46, "b_t": 6.846178787174751, "m_t": 6.713459629356409, "asleep": 0}
TRACE {"t": 199, "night": 1, "S": 47, "b_t": 7.152463873167434, "m_t": 6.715450860672534, "asleep": 0}
TRACE {"t": 200, "night": 1, "S": 48, "b_t": 7.443251498894238, "m_t": 6.769355311830655, "asleep": 0}
TRACE {"t": 201, "night": 1, "S": 49, "b_t": 7.700910919577485, "m_t": 6.823259762988776, "asleep": 0}
TRACE {"t": 202, "night": 1, "S": 50, "b_t": 7.9518662889586675, "m_t": 6.834719275081763, "asleep": 0}
TRACE {"t": 203, "night": 1, "S": 51, "b_t": 8.326495753898792, "m_t": 6.846178787174751, "asleep": 0}
TRACE {"t": 204, "night": 1, "S": 52, "b_t": 8.510019749809647, "m_t": 6.8517427698652416, "asleep": 0}
TRACE {"t": 205, "night": 1, "S": 53, "b_t": 8.597582099244775, "m_t": 6.857306752555733, "asleep": 0}
TRACE {"t": 206, "night": 1, "S": 54, "b_t": 8.718313564107026, "m_t": 6.857612433391271, "asleep": 0}
TRACE {"t": 207, "night": 1, "S": 55, "b_t": 8.861810799731947, "m_t": 6.85791811422681, "asleep": 0}
TRACE {"t": 208, "night": 1, "S": 56, "b_t": 8.938148265626705, "m_t": 6.858305658822369, "asleep": 0}
TRACE {"t": 209, "night": 1, "S": 57, "b_t": 8.960254787841675, "m_t": 6.858693203417928, "asleep": 0}
TRACE {"t": 210, "night": 1, "S": 58, "b_t": 8.99048995745894, "m_t": 6.8675491542959595, "asleep": 0}
TRACE {"t": 211, "night": 1, "S": 59, "b_t": 9.104699155117295, "m_t": 6.8764051051739905, "asleep": 0}
TRACE {"t": 212, "night": 1, "S": 60, "b_t": 9.204641611128347, "m_t": 6.897972381886415, "asleep": 0}
TRACE {"t": 213, "night": 1, "S": 61, "b_t": 9.130648546350793, "m_t": 6.919539658598841, "asleep": 0}
TRACE {"t": 214, "night": 1, "S": 62, "b_t": 9.246200611948804, "m_t": 6.9452008940728565, "asleep": 0}
TRACE {"t": 215, "night": 1, "S": 63, "b_t": 9.379403041194072, "m_t": 6.970862129546872, "asleep": 0}
TRACE {"t": 216, "night": 1, "S": 64, "b_t": 9.279137523211062, "m_t": 6.973626218386428, "asleep": 0}
TRACE {"t": 217, "night": 1, "S": 65, "b_t": 9.251615255522927, "m_t": 6.976390307225984, "asleep": 0}
TRACE {"t": 218, "night": 1, "S": 66, "b_t": 9.258854446142479, "m_t": 6.997329249477653, "asleep": 0}
TRACE {"t": 219, "night": 1, "S": 67, "b_t": 9.210342467838446, "m_t": 7.018268191729323, "asleep": 0}
TRACE {"t": 220, "night": 1, "S": 68, "b_t": 9.208562514198348, "m_t": 7.03163141911672, "asleep": 0}
TRACE {"t": 221, "night": 1, "S": 69, "b_t": 9.20765195879497, "m_t": 7.044994646504117, "asleep": 0}
TRACE {"t": 222, "night": 1, "S": 70, "b_t": 9.162095949900676, "m_t": 7.053932701411517, "asleep": 0}
TRACE {"t": 223, "night": 1, "S": 71, "b_t": 9.203916616331862, "m_t": 7.062870756318917, "asleep": 0}
TRACE {"t": 224, "night": 1, "S": 72, "b_t": 9.143392831121567, "m_t": 7.107667314743176, "asleep": 0}
TRACE {"t": 225, "night": 1, "S": 73, "b_t": 9.177733017306625, "m_t": 7.152463873167434, "asleep": 0}
TRACE {"t": 226, "night": 1, "S": 74, "b_t": 9.202539926574595, "m_t": 7.182576737609471, "asleep": 0}
TRACE {"t": 227, "night": 1, "S": 75, "b_t": 9.197356544647704, "m_t": 7.212689602051509, "asleep": 0}
TRACE {"t": 228, "night": 1, "S": 76, "b_t": 9.22946686266451, "m_t": 7.214659964153303, "asleep": 0}
TRACE {"t": 229, "night": 1, "S": 77, "b_t": 9.264773726455505, "m_t": 7.216630326255097, "asleep": 0}
TRACE {"t": 230, "night": 1, "S": 78, "b_t": 9.267073770682073, "m_t": 7.25861296044995, "asleep": 0}
TRACE {"t": 231, "night": 1, "S": 79, "b_t": 9.254367286565563, "m_t": 7.300595594644804, "asleep": 0}
TRACE {"t": 232, "night": 1, "S": 80, "b_t": 9.137484073714827, "m_t": 7.326678475454171, "asleep": 0}
TRACE {"t": 233, "night": 1, "S": 81, "b_t": 9.087134249898593, "m_t": 7.352761356263539, "asleep": 0}
TRACE {"t": 234, "night": 1, "S": 82, "b_t": 9.06445402077423, "m_t": 7.386570078036398, "asleep": 0}
TRACE {"t": 235, "night": 1, "S": 83, "b_t": 9.053487832317055, "m_t": 7.420378799809256, "asleep": 0}
TRACE {"t": 236, "night": 1, "S": 84, "b_t": 9.094555483865017, "m_t": 7.423014370643477, "asleep": 0}
TRACE {"t": 237, "night": 1, "S": 85, "b_t": 8.98595591037861, "m_t": 7.425649941477698, "asleep": 0}
TRACE {"t": 238, "night": 1, "S": 86, "b_t": 8.98588498664781, "m_t": 7.434450720185968, "asleep": 0}
TRACE {"t": 239, "night": 1, "S": 87, "b_t": 9.026848459637122, "m_t": 7.443251498894238, "asleep": 0}
TRACE {"t": 240, "night": 1, "S": 88, "b_t": 8.962533326328233, "m_t": 7.453263240767238, "asleep": 0}
TRACE {"t": 241, "night": 1, "S": 89, "b_t": 8.910867414878007, "m_t": 7.463274982640238, "asleep": 0}
TRACE {"t": 242, "night": 1, "S": 90, "b_t": 8.863921494190247, "m_t": 7.489737492399181, "asleep": 0}
TRACE {"t": 243, "night": 1, "S": 91, "b_t": 8.783828064969004, "m_t": 7.516200002158125, "asleep": 0}
TRACE {"t": 244, "night": 1, "S": 92, "b_t": 8.709467757802463, "m_t": 7.544790993142839, "asleep": 0}
TRACE {"t": 245, "night": 1, "S": 93, "b_t": 8.710411311565576, "m_t": 7.573381984127553, "asleep": 0}
TRACE {"t": 246, "night": 1, "S": 94, "b_t": 8.737052258354801, "m_t": 7.5914988203384635, "asleep": 0}
TRACE {"t": 247, "night": 1, "S": 95, "b_t": 8.617163153051903, "m_t": 7.609615656549375, "asleep": 0}
TRACE {"t": 248, "night": 1, "S": 96, "b_t": 8.458653526005492, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 249, "night": 1, "S": 93, "b_t": 8.46631670240503, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 250, "night": 1, "S": 90, "b_t": 8.462440903650231, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 251, "night": 1, "S": 87, "b_t": 8.513335829004241, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 252, "night": 1, "S": 84, "b_t": 8.415899410672617, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 253, "night": 1, "S": 81, "b_t": 8.418517412278078, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 254, "night": 1, "S": 78, "b_t": 8.402626321378449, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 255, "night": 1, "S": 75, "b_t": 8.367509061938037, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 256, "night": 0, "S": 72, "b_t": 8.228108949006357, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 257, "night": 0, "S": 69, "b_t": 7.507816928088858, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 258, "night": 0, "S": 66, "b_t": 6.894223290003143, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 259, "night": 0, "S": 63, "b_t": 6.241344254897555, "m_t": 7.613389218451347, "asleep": 1}
TRACE {"t": 260, "night": 0, "S": 60, "b_t": 5.659873294982759, "m_t": 7.613389218451347, "asleep": 1}
EYELID {"sleep_share": 0.2421875, "night_sleep_share": 0.036290322580645164, "sleep_decisions": 248, "night_sleep_decisions": 9, "wall_s": 224.425248590298}
```

## Kernel walls

| Kernels | Finite L512 arms | Estimate s | Limit s |
|---|---|---|---|
| 0 | 0 | NA | 5100 |
