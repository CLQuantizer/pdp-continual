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
| 0 | 48 | 1 | 0 | 452.440215275 | 0 |

## Smoke L1024

| Seed | Arm | L | Sleep | Night sleep | sleep_share | night_sleep_share | Stop threshold >= | Meets (0/1) | Wall s |
|---|---|---|---|---|---|---|---|---|---|
| 84 | pressure_eyelid | 1024 | 304 | 109 | 0.296875000 | 0.358552632 | .50 | 0 | 210.379080690 |

| Eyelid checksum decisions | Intact | Mismatches |
|---|---|---|
| 304 | 304 | 0 |

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
BASELINE nosleep {"T14": 5.6285469390612395, "reference_T14": 5.6285469390612395, "reference": "M9", "abs": 0.0, "checkpoint_max_abs": 0.0, "sleep_share": 0.0, "wall_s": 44.38853231817484}
s24 age_fixed20 p=128 checkpoint t=32/256 W14 11.8096
s24 age_fixed20 p=128 checkpoint t=64/256 W14 10.6724
s24 age_fixed20 p=128 checkpoint t=96/256 W14 6.6363
s24 age_fixed20 p=128 checkpoint t=128/256 W14 4.4603
s24 age_fixed20 p=128 checkpoint t=160/256 W14 4.2375
s24 age_fixed20 p=128 checkpoint t=192/256 W14 3.8962
s24 age_fixed20 p=128 checkpoint t=224/256 W14 3.4717
s24 age_fixed20 p=128 checkpoint t=256/256 W14 2.3318
s24 age_fixed20 p=128 t=256/256
BASELINE fixed20_compat {"T14": 5.939474666464639, "reference_T14": 5.939474666464639, "reference": "M9", "abs": 0.0, "checkpoint_max_abs": 0.0, "sleep_share": 0.1875, "wall_s": 39.94194914959371}
BASELINE fixed_random {"T14": 5.943012759003068, "reference_T14": 5.943012759003068, "reference": "W9", "abs": 0.0, "checkpoint_max_abs": 0.0, "sleep_share": 0.25, "wall_s": 39.51636368222535}
TRACE {"t": 180, "night": 0, "S": 1, "b_t": 5.43527735222058, "m_t": 6.85791811422681, "asleep": 1}
TRACE {"t": 181, "night": 0, "S": 0, "b_t": 4.894038367428155, "m_t": 6.85791811422681, "asleep": 1}
TRACE {"t": 182, "night": 0, "S": 0, "b_t": 4.406955631135742, "m_t": 6.85791811422681, "asleep": 1}
TRACE {"t": 183, "night": 0, "S": 0, "b_t": 3.9686066486681777, "m_t": 6.85791811422681, "asleep": 0}
TRACE {"t": 184, "night": 0, "S": 1, "b_t": 3.5741273153937168, "m_t": 6.840588938607793, "asleep": 0}
TRACE {"t": 185, "night": 0, "S": 2, "b_t": 3.2191273406764145, "m_t": 6.823259762988776, "asleep": 0}
TRACE {"t": 186, "night": 0, "S": 3, "b_t": 2.8996558526131473, "m_t": 6.769355311830655, "asleep": 0}
TRACE {"t": 187, "night": 0, "S": 4, "b_t": 2.612157246676075, "m_t": 6.715450860672534, "asleep": 0}
TRACE {"t": 188, "night": 0, "S": 5, "b_t": 2.3534316868530474, "m_t": 6.713459629356409, "asleep": 0}
TRACE {"t": 189, "night": 0, "S": 6, "b_t": 2.1205994898606084, "m_t": 6.711468398040284, "asleep": 0}
TRACE {"t": 190, "night": 0, "S": 7, "b_t": 1.911069105252438, "m_t": 6.671327997605182, "asleep": 0}
TRACE {"t": 191, "night": 0, "S": 8, "b_t": 1.7225107866254539, "m_t": 6.63118759717008, "asleep": 0}
TRACE {"t": 192, "night": 1, "S": 9, "b_t": 2.7830646460116917, "m_t": 6.606234574814348, "asleep": 0}
TRACE {"t": 193, "night": 1, "S": 10, "b_t": 3.613385840655825, "m_t": 6.581281552458616, "asleep": 0}
TRACE {"t": 194, "night": 1, "S": 11, "b_t": 4.338118106353533, "m_t": 6.546625211607761, "asleep": 0}
TRACE {"t": 195, "night": 1, "S": 12, "b_t": 5.072923508782053, "m_t": 6.511968870756905, "asleep": 0}
TRACE {"t": 196, "night": 1, "S": 13, "b_t": 5.717886297009442, "m_t": 6.399379369392209, "asleep": 0}
TRACE {"t": 197, "night": 1, "S": 14, "b_t": 6.211961875900781, "m_t": 6.286789868027514, "asleep": 0}
TRACE {"t": 198, "night": 1, "S": 15, "b_t": 6.7472951189730574, "m_t": 6.399379369392209, "asleep": 0}
TRACE {"t": 199, "night": 1, "S": 16, "b_t": 7.200148838490592, "m_t": 6.511968870756905, "asleep": 1}
TRACE {"t": 200, "night": 1, "S": 13, "b_t": 7.491161890062452, "m_t": 6.511968870756905, "asleep": 1}
TRACE {"t": 201, "night": 1, "S": 10, "b_t": 7.898021079649265, "m_t": 6.511968870756905, "asleep": 1}
TRACE {"t": 202, "night": 1, "S": 7, "b_t": 8.249637370759547, "m_t": 6.511968870756905, "asleep": 1}
TRACE {"t": 203, "night": 1, "S": 4, "b_t": 8.632852898873283, "m_t": 6.511968870756905, "asleep": 1}
TRACE {"t": 204, "night": 1, "S": 1, "b_t": 8.851488626975105, "m_t": 6.511968870756905, "asleep": 1}
TRACE {"t": 205, "night": 1, "S": 0, "b_t": 9.126130444626309, "m_t": 6.511968870756905, "asleep": 1}
TRACE {"t": 206, "night": 1, "S": 0, "b_t": 9.369472380539452, "m_t": 6.511968870756905, "asleep": 1}
TRACE {"t": 207, "night": 1, "S": 0, "b_t": 9.627443582997, "m_t": 6.511968870756905, "asleep": 0}
TRACE {"t": 208, "night": 1, "S": 1, "b_t": 9.855584359235817, "m_t": 6.546625211607761, "asleep": 0}
TRACE {"t": 209, "night": 1, "S": 2, "b_t": 9.83874923722274, "m_t": 6.581281552458616, "asleep": 0}
TRACE {"t": 210, "night": 1, "S": 3, "b_t": 9.780915738501417, "m_t": 6.606234574814348, "asleep": 0}
TRACE {"t": 211, "night": 1, "S": 4, "b_t": 9.778568793492699, "m_t": 6.63118759717008, "asleep": 0}
TRACE {"t": 212, "night": 1, "S": 5, "b_t": 9.880122366227114, "m_t": 6.671327997605182, "asleep": 0}
TRACE {"t": 213, "night": 1, "S": 6, "b_t": 9.802798318196674, "m_t": 6.711468398040284, "asleep": 0}
TRACE {"t": 214, "night": 1, "S": 7, "b_t": 9.868987195423827, "m_t": 6.713459629356409, "asleep": 0}
TRACE {"t": 215, "night": 1, "S": 8, "b_t": 9.951560765229802, "m_t": 6.715450860672534, "asleep": 0}
TRACE {"t": 216, "night": 1, "S": 9, "b_t": 9.892141723329406, "m_t": 6.731372989822796, "asleep": 0}
TRACE {"t": 217, "night": 1, "S": 10, "b_t": 10.073941611000961, "m_t": 6.7472951189730574, "asleep": 0}
TRACE {"t": 218, "night": 1, "S": 11, "b_t": 10.16979202440753, "m_t": 6.7852774409809165, "asleep": 0}
TRACE {"t": 219, "night": 1, "S": 12, "b_t": 10.177853257232718, "m_t": 6.823259762988776, "asleep": 0}
TRACE {"t": 220, "night": 1, "S": 13, "b_t": 10.195154894693635, "m_t": 6.840588938607793, "asleep": 0}
TRACE {"t": 221, "night": 1, "S": 14, "b_t": 10.221885080115477, "m_t": 6.85791811422681, "asleep": 0}
TRACE {"t": 222, "night": 1, "S": 15, "b_t": 10.189921278731578, "m_t": 6.8671616097004, "asleep": 0}
TRACE {"t": 223, "night": 1, "S": 16, "b_t": 10.215178459804724, "m_t": 6.8764051051739905, "asleep": 1}
TRACE {"t": 224, "night": 1, "S": 13, "b_t": 10.17478460053271, "m_t": 6.8764051051739905, "asleep": 1}
TRACE {"t": 225, "night": 1, "S": 10, "b_t": 10.153141943223766, "m_t": 6.8764051051739905, "asleep": 1}
TRACE {"t": 226, "night": 1, "S": 7, "b_t": 10.139879821229584, "m_t": 6.8764051051739905, "asleep": 1}
TRACE {"t": 227, "night": 1, "S": 4, "b_t": 10.0914367253888, "m_t": 6.8764051051739905, "asleep": 1}
TRACE {"t": 228, "night": 1, "S": 1, "b_t": 10.037969043561969, "m_t": 6.8764051051739905, "asleep": 1}
TRACE {"t": 229, "night": 1, "S": 0, "b_t": 10.040947670885384, "m_t": 6.8764051051739905, "asleep": 1}
TRACE {"t": 230, "night": 1, "S": 0, "b_t": 9.964835839809712, "m_t": 6.8764051051739905, "asleep": 1}
TRACE {"t": 231, "night": 1, "S": 0, "b_t": 9.894838803715004, "m_t": 6.8764051051739905, "asleep": 0}
TRACE {"t": 232, "night": 1, "S": 1, "b_t": 9.84525663868126, "m_t": 6.923633617360432, "asleep": 0}
TRACE {"t": 233, "night": 1, "S": 2, "b_t": 9.82134768587358, "m_t": 6.970862129546872, "asleep": 0}
TRACE {"t": 234, "night": 1, "S": 3, "b_t": 9.846023778317141, "m_t": 6.973626218386428, "asleep": 0}
TRACE {"t": 235, "night": 1, "S": 4, "b_t": 9.883324953689849, "m_t": 6.976390307225984, "asleep": 0}
TRACE {"t": 236, "night": 1, "S": 5, "b_t": 9.853137265830773, "m_t": 6.997329249477653, "asleep": 0}
TRACE {"t": 237, "night": 1, "S": 6, "b_t": 9.846343389849562, "m_t": 7.018268191729323, "asleep": 0}
TRACE {"t": 238, "night": 1, "S": 7, "b_t": 9.785741800069275, "m_t": 7.03163141911672, "asleep": 0}
TRACE {"t": 239, "night": 1, "S": 8, "b_t": 9.759549320919868, "m_t": 7.044994646504117, "asleep": 0}
TRACE {"t": 240, "night": 1, "S": 9, "b_t": 9.720966407424024, "m_t": 7.122571742497355, "asleep": 0}
TRACE {"t": 241, "night": 1, "S": 10, "b_t": 9.718341376602174, "m_t": 7.200148838490592, "asleep": 0}
TRACE {"t": 242, "night": 1, "S": 11, "b_t": 9.657243905748567, "m_t": 7.206419220271051, "asleep": 0}
TRACE {"t": 243, "night": 1, "S": 12, "b_t": 9.596281333487672, "m_t": 7.212689602051509, "asleep": 0}
TRACE {"t": 244, "night": 1, "S": 13, "b_t": 9.580873167982357, "m_t": 7.214659964153303, "asleep": 0}
TRACE {"t": 245, "night": 1, "S": 14, "b_t": 9.662184648108624, "m_t": 7.216630326255097, "asleep": 0}
TRACE {"t": 246, "night": 1, "S": 15, "b_t": 9.622962008325954, "m_t": 7.2846958412593175, "asleep": 0}
TRACE {"t": 247, "night": 1, "S": 16, "b_t": 9.505046475567017, "m_t": 7.352761356263539, "asleep": 1}
TRACE {"t": 248, "night": 1, "S": 13, "b_t": 9.336975200747709, "m_t": 7.352761356263539, "asleep": 1}
TRACE {"t": 249, "night": 1, "S": 10, "b_t": 9.318820390601068, "m_t": 7.352761356263539, "asleep": 1}
TRACE {"t": 250, "night": 1, "S": 7, "b_t": 9.195272590609006, "m_t": 7.352761356263539, "asleep": 1}
TRACE {"t": 251, "night": 1, "S": 4, "b_t": 9.350975074838955, "m_t": 7.352761356263539, "asleep": 1}
TRACE {"t": 252, "night": 1, "S": 1, "b_t": 9.286099514364325, "m_t": 7.352761356263539, "asleep": 1}
TRACE {"t": 253, "night": 1, "S": 0, "b_t": 9.280283088790446, "m_t": 7.352761356263539, "asleep": 1}
TRACE {"t": 254, "night": 1, "S": 0, "b_t": 9.215436715258635, "m_t": 7.352761356263539, "asleep": 1}
TRACE {"t": 255, "night": 1, "S": 0, "b_t": 9.139790698483232, "m_t": 7.352761356263539, "asleep": 0}
TRACE {"t": 256, "night": 0, "S": 1, "b_t": 9.03295811371289, "m_t": 7.389205648870618, "asleep": 0}
TRACE {"t": 257, "night": 0, "S": 2, "b_t": 8.307713615587865, "m_t": 7.425649941477698, "asleep": 0}
TRACE {"t": 258, "night": 0, "S": 3, "b_t": 7.67403510352013, "m_t": 7.4444624620589686, "asleep": 0}
TRACE {"t": 259, "night": 0, "S": 4, "b_t": 7.084327887822113, "m_t": 7.425649941477698, "asleep": 0}
TRACE {"t": 260, "night": 0, "S": 5, "b_t": 6.519988564706308, "m_t": 7.389205648870618, "asleep": 0}
EYELID {"sleep_share": 0.296875, "night_sleep_share": 0.35855263157894735, "sleep_decisions": 304, "night_sleep_decisions": 109, "wall_s": 210.37908069044352}
```

## Kernel walls

| Kernels | Finite L512 arms | Estimate s | Limit s |
|---|---|---|---|
| 0 | 0 | NA | 5100 |
