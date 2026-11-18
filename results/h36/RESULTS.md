| Seed | Entropy / fixed junk share | Gain − entropy clean bits/byte | Progress − entropy clean bits/byte | Fixed − entropy clean bits/byte |
|---|---:|---:|---:|---:|
| 0 | 1.009637384 | 0.032227463 | 0.024396295 | 0.030025891 |
| 1 | 1.025807756 | 0.027503849 | 0.027424837 | 0.019453903 |
| 2 | 1.014423696 | -0.023591935 | -0.003037770 | -0.025297548 |
| 3 | 1.009006347 | 0.140246281 | -0.006470832 | 0.121790791 |
| 4 | 1.018260836 | 0.007399467 | 0.016582752 | 0.015312770 |
| 5 | 1.011167523 | 0.026771662 | 0.024244031 | 0.033358801 |

Bar 1 pass count: 0/6; required 5/6. Bar 2 pass count: 0/6; required 5/6.

Scored kernels: 24/24.

| Smoke rule | Bytes | ρ | Junk share | Estimated 400-doc wall minutes |
|---|---:|---:|---:|---:|
| fixed | 262144 | 0.250000000 | 0.500000000 | 48.526876312 |
| entropy | 262144 | 0.250423431 | 0.580757689 | 58.273503080 |
| gain | 262144 | 0.250907898 | 0.481162769 | 52.752371786 |
| progress | 262144 | 0.243862152 | 0.544621209 | 44.170006260 |

H35 build(0), fixed, 65536 bytes: ρ = 0.250000000; bits/byte = 4.176429416; junk share = 0.375000000.

H35 build(0), entropy, 65536 bytes: ρ = 0.245513916; bits/byte = 4.146310579; junk share = 0.394903667.

H35 build(0), gain, 65536 bytes: ρ = 0.254974365; bits/byte = 4.161201241; junk share = 0.367265111.

H35 build(0), progress, 65536 bytes: ρ = 0.240539551; bits/byte = 4.172676368; junk share = 0.394252728.

| Rule | Seed | Clean test bits/byte | Life clean bits | Life junk bits | Life clean bits/byte | Life junk bits/byte | Junk step share | ρ | Test ρ | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0 | 2.465328693 | 2073214.709830189 | 3255480.237409935 | 2.481156544 | 4.055076428 | 0.490000000 | 0.250000000 | 0.250000000 | 1177.250996120 |
| fixed | 1 | 2.473248472 | 1978176.896569486 | 3426096.965227811 | 2.414766719 | 4.182247272 | 0.500000000 | 0.250000000 | 0.250000000 | 1078.862699910 |
| fixed | 2 | 2.709551301 | 1845371.709810019 | 3385867.071443515 | 2.502945570 | 3.757398650 | 0.550000000 | 0.250000000 | 0.250000000 | 664.750546532 |
| fixed | 3 | 2.661869491 | 1811441.117292176 | 3067530.914322169 | 2.403512862 | 3.467170901 | 0.540000000 | 0.250000000 | 0.250000000 | 930.801460905 |
| fixed | 4 | 2.436757837 | 2058866.577402055 | 2913363.865945649 | 2.393585586 | 3.743528816 | 0.475000000 | 0.250000000 | 0.250000000 | 1003.091440105 |
| fixed | 5 | 2.460791146 | 2006808.131317656 | 3041323.316447285 | 2.425462334 | 3.750053411 | 0.495000000 | 0.250000000 | 0.250000000 | 1060.295862573 |
| entropy | 0 | 2.435302802 | 2052534.729173061 | 3221344.158407350 | 2.456407410 | 4.012556001 | 0.494722318 | 0.250377197 | 0.211215973 | 1134.171795324 |
| entropy | 1 | 2.453794569 | 1956407.743628033 | 3395233.183290901 | 2.388193046 | 4.144571757 | 0.512903878 | 0.250618286 | 0.168670654 | 983.810700994 |
| entropy | 2 | 2.734848849 | 1825266.085250747 | 3349547.417113893 | 2.475675571 | 3.717093636 | 0.557933033 | 0.250475464 | 0.018199921 | 691.199538365 |
| entropy | 3 | 2.540078700 | 1794056.603265762 | 3035970.448683828 | 2.380446198 | 3.431498717 | 0.544863427 | 0.250625000 | 0.972705841 | 1271.332383230 |
| entropy | 4 | 2.421445067 | 2040908.130787414 | 2878569.462929786 | 2.372707555 | 3.698819725 | 0.483673897 | 0.250367432 | 0.281669617 | 979.239048345 |
| entropy | 5 | 2.427432345 | 1986576.280012659 | 3011771.054089198 | 2.401009775 | 3.713614482 | 0.500527924 | 0.250881348 | 0.241886139 | 701.349767613 |
| gain | 0 | 2.467530265 | 2070438.544558966 | 3247386.250146021 | 2.477834119 | 4.044994432 | 0.492522686 | 0.252227783 | 0.275131226 | 955.477814724 |
| gain | 1 | 2.481298418 | 1979025.962670016 | 3422704.647336354 | 2.415803177 | 4.178106259 | 0.503160725 | 0.251036377 | 0.201816559 | 978.476952551 |
| gain | 2 | 2.711256914 | 1841702.510005065 | 3376352.601994409 | 2.497968899 | 3.746840157 | 0.552624102 | 0.251353149 | 0.243606567 | 1078.037429799 |
| gain | 3 | 2.680324982 | 1810571.975668267 | 3066235.178458133 | 2.402359640 | 3.465706356 | 0.545189575 | 0.251915894 | 0.103237152 | 1048.192009581 |
| gain | 4 | 2.428844534 | 2053563.498765537 | 2905370.347520283 | 2.387420362 | 3.733257539 | 0.479285794 | 0.251693115 | 0.214775085 | 950.787794065 |
| gain | 5 | 2.454204007 | 2001182.467724414 | 3034535.434529277 | 2.418663061 | 3.741683725 | 0.496727051 | 0.251752930 | 0.347278595 | 1147.856447442 |
| progress | 0 | 2.459699097 | 2072556.451710614 | 3255143.534409137 | 2.480368762 | 4.054657025 | 0.496840937 | 0.248560181 | 0.305553436 | 1004.216636705 |
| progress | 1 | 2.481219406 | 1975870.468475190 | 3423499.177433324 | 2.411951255 | 4.179076144 | 0.504471894 | 0.248200073 | 0.275424957 | 966.836880791 |
| progress | 2 | 2.731811079 | 1843193.209837375 | 3381780.346068874 | 2.499990790 | 3.752863488 | 0.556163556 | 0.248401489 | 0.031784058 | 632.710360178 |
| progress | 3 | 2.533607868 | 1809928.309778598 | 3067402.374952167 | 2.401505591 | 3.467025615 | 0.546003635 | 0.248486328 | 0.858955383 | 1078.907149444 |
| progress | 4 | 2.438027819 | 2057645.884526955 | 2910906.470606910 | 2.392166439 | 3.740371184 | 0.480146725 | 0.248757935 | 0.247257233 | 690.287101195 |
| progress | 5 | 2.451676376 | 2005585.187067353 | 3039266.892659407 | 2.423984263 | 3.747517771 | 0.498504112 | 0.248483887 | 0.227127075 | 991.103521467 |

| Rule | Seed | Junk kind | Share of all expensive steps |
|---|---:|---|---:|
| fixed | 0 | J1_random | 0.102500000 |
| fixed | 0 | J4_boiler | 0.077500000 |
| fixed | 0 | J3_dup | 0.092500000 |
| fixed | 0 | J5_shuffle | 0.107500000 |
| fixed | 0 | J2_corrupt | 0.110000000 |
| fixed | 1 | J1_random | 0.112500000 |
| fixed | 1 | J3_dup | 0.112500000 |
| fixed | 1 | J5_shuffle | 0.077500000 |
| fixed | 1 | J4_boiler | 0.077500000 |
| fixed | 1 | J2_corrupt | 0.120000000 |
| fixed | 2 | J3_dup | 0.122500000 |
| fixed | 2 | J1_random | 0.110000000 |
| fixed | 2 | J4_boiler | 0.117500000 |
| fixed | 2 | J2_corrupt | 0.107500000 |
| fixed | 2 | J5_shuffle | 0.092500000 |
| fixed | 3 | J3_dup | 0.125000000 |
| fixed | 3 | J2_corrupt | 0.085000000 |
| fixed | 3 | J5_shuffle | 0.117500000 |
| fixed | 3 | J1_random | 0.095000000 |
| fixed | 3 | J4_boiler | 0.117500000 |
| fixed | 4 | J4_boiler | 0.102500000 |
| fixed | 4 | J5_shuffle | 0.115000000 |
| fixed | 4 | J2_corrupt | 0.092500000 |
| fixed | 4 | J1_random | 0.085000000 |
| fixed | 4 | J3_dup | 0.080000000 |
| fixed | 5 | J1_random | 0.095000000 |
| fixed | 5 | J5_shuffle | 0.100000000 |
| fixed | 5 | J3_dup | 0.117500000 |
| fixed | 5 | J2_corrupt | 0.095000000 |
| fixed | 5 | J4_boiler | 0.087500000 |
| entropy | 0 | J1_random | 0.118800248 |
| entropy | 0 | J4_boiler | 0.063181040 |
| entropy | 0 | J3_dup | 0.081239731 |
| entropy | 0 | J5_shuffle | 0.105236728 |
| entropy | 0 | J2_corrupt | 0.126264572 |
| entropy | 1 | J1_random | 0.130071381 |
| entropy | 1 | J3_dup | 0.104950403 |
| entropy | 1 | J5_shuffle | 0.076816857 |
| entropy | 1 | J4_boiler | 0.062199687 |
| entropy | 1 | J2_corrupt | 0.138865550 |
| entropy | 2 | J3_dup | 0.117140497 |
| entropy | 2 | J1_random | 0.128707853 |
| entropy | 2 | J4_boiler | 0.094471208 |
| entropy | 2 | J2_corrupt | 0.124043872 |
| entropy | 2 | J5_shuffle | 0.093569603 |
| entropy | 3 | J3_dup | 0.117840165 |
| entropy | 3 | J2_corrupt | 0.098420940 |
| entropy | 3 | J5_shuffle | 0.121254481 |
| entropy | 3 | J1_random | 0.112058720 |
| entropy | 3 | J4_boiler | 0.095289121 |
| entropy | 4 | J4_boiler | 0.082634897 |
| entropy | 4 | J5_shuffle | 0.120374840 |
| entropy | 4 | J2_corrupt | 0.105343221 |
| entropy | 4 | J1_random | 0.100123841 |
| entropy | 4 | J3_dup | 0.075197098 |
| entropy | 5 | J1_random | 0.110584268 |
| entropy | 5 | J5_shuffle | 0.100412608 |
| entropy | 5 | J3_dup | 0.108545557 |
| entropy | 5 | J2_corrupt | 0.109723047 |
| entropy | 5 | J4_boiler | 0.071262444 |
| gain | 0 | J1_random | 0.103402299 |
| gain | 0 | J4_boiler | 0.076101633 |
| gain | 0 | J3_dup | 0.090753781 |
| gain | 0 | J5_shuffle | 0.107349062 |
| gain | 0 | J2_corrupt | 0.114915910 |
| gain | 1 | J1_random | 0.113246357 |
| gain | 1 | J3_dup | 0.111892107 |
| gain | 1 | J5_shuffle | 0.076766724 |
| gain | 1 | J4_boiler | 0.077632276 |
| gain | 1 | J2_corrupt | 0.123623261 |
| gain | 2 | J3_dup | 0.119026169 |
| gain | 2 | J1_random | 0.109650646 |
| gain | 2 | J4_boiler | 0.117447798 |
| gain | 2 | J2_corrupt | 0.113676706 |
| gain | 2 | J5_shuffle | 0.092822783 |
| gain | 3 | J3_dup | 0.125033011 |
| gain | 3 | J2_corrupt | 0.089012669 |
| gain | 3 | J5_shuffle | 0.117621548 |
| gain | 3 | J1_random | 0.095675475 |
| gain | 3 | J4_boiler | 0.117846872 |
| gain | 4 | J4_boiler | 0.103566180 |
| gain | 4 | J5_shuffle | 0.115637746 |
| gain | 4 | J2_corrupt | 0.094348819 |
| gain | 4 | J1_random | 0.087267868 |
| gain | 4 | J3_dup | 0.078465180 |
| gain | 5 | J1_random | 0.096418666 |
| gain | 5 | J5_shuffle | 0.099858900 |
| gain | 5 | J3_dup | 0.116194554 |
| gain | 5 | J2_corrupt | 0.097041739 |
| gain | 5 | J4_boiler | 0.087213193 |
| progress | 0 | J1_random | 0.111393499 |
| progress | 0 | J4_boiler | 0.079201259 |
| progress | 0 | J3_dup | 0.091994667 |
| progress | 0 | J5_shuffle | 0.107273089 |
| progress | 0 | J2_corrupt | 0.106978423 |
| progress | 1 | J1_random | 0.120506282 |
| progress | 1 | J3_dup | 0.110266543 |
| progress | 1 | J5_shuffle | 0.076232445 |
| progress | 1 | J4_boiler | 0.080373588 |
| progress | 1 | J2_corrupt | 0.117093036 |
| progress | 2 | J3_dup | 0.121858269 |
| progress | 2 | J1_random | 0.119327438 |
| progress | 2 | J4_boiler | 0.120059659 |
| progress | 2 | J2_corrupt | 0.104051540 |
| progress | 2 | J5_shuffle | 0.090866650 |
| progress | 3 | J3_dup | 0.124813323 |
| progress | 3 | J2_corrupt | 0.083154844 |
| progress | 3 | J5_shuffle | 0.114693456 |
| progress | 3 | J1_random | 0.102669974 |
| progress | 3 | J4_boiler | 0.120672038 |
| progress | 4 | J4_boiler | 0.106152393 |
| progress | 4 | J5_shuffle | 0.113069081 |
| progress | 4 | J2_corrupt | 0.088192068 |
| progress | 4 | J1_random | 0.092954498 |
| progress | 4 | J3_dup | 0.079778686 |
| progress | 5 | J1_random | 0.103891765 |
| progress | 5 | J5_shuffle | 0.099028287 |
| progress | 5 | J3_dup | 0.114279468 |
| progress | 5 | J2_corrupt | 0.091755175 |
| progress | 5 | J4_boiler | 0.089549416 |

| Rule | Metric | n | Mean | Std (ddof=0) |
|---|---|---:|---:|---:|
| fixed | clean_test_bpb | 6 | 2.534591157 | 0.108311926 |
| fixed | life_bits_clean | 6 | 1962313.190370264 | 100242.084326411 |
| fixed | life_bits_junk | 6 | 3181610.395132727 | 187795.109619262 |
| fixed | life_bpb_clean | 6 | 2.436904936 | 0.040686199 |
| fixed | life_bpb_junk | 6 | 3.825912580 | 0.232888361 |
| fixed | junk_share | 6 | 0.508333333 | 0.027182511 |
| fixed | rho | 6 | 0.250000000 | 0.000000000 |
| fixed | test_rho | 6 | 0.250000000 | 0.000000000 |
| fixed | wall_seconds | 6 | 985.842167691 | 161.895211828 |
| fixed | junk_share_by_kind.J1_random | 6 | 0.100000000 | 0.009464847 |
| fixed | junk_share_by_kind.J2_corrupt | 6 | 0.101666667 | 0.011873172 |
| fixed | junk_share_by_kind.J3_dup | 6 | 0.108333333 | 0.016499158 |
| fixed | junk_share_by_kind.J4_boiler | 6 | 0.096666667 | 0.016935335 |
| fixed | junk_share_by_kind.J5_shuffle | 6 | 0.101666667 | 0.013743685 |
| entropy | clean_test_bpb | 6 | 2.502150389 | 0.111422731 |
| entropy | life_bits_clean | 6 | 1942624.928686280 | 99755.605339761 |
| entropy | life_bits_junk | 6 | 3148739.287419159 | 187449.306454039 |
| entropy | life_bpb_clean | 6 | 2.412406593 | 0.039269792 |
| entropy | life_bpb_junk | 6 | 3.786359053 | 0.232077524 |
| entropy | junk_share | 6 | 0.515770746 | 0.026888192 |
| entropy | rho | 6 | 0.250557454 | 0.000177244 |
| entropy | test_rho | 6 | 0.315724691 | 0.305315155 |
| entropy | wall_seconds | 6 | 960.183872312 | 211.090101470 |
| entropy | junk_share_by_kind.J1_random | 6 | 0.116724385 | 0.010496845 |
| entropy | junk_share_by_kind.J2_corrupt | 6 | 0.117110200 | 0.013829280 |
| entropy | junk_share_by_kind.J3_dup | 6 | 0.100818909 | 0.016696852 |
| entropy | junk_share_by_kind.J4_boiler | 6 | 0.078173066 | 0.013577942 |
| entropy | junk_share_by_kind.J5_shuffle | 6 | 0.102944186 | 0.015390903 |
| gain | clean_test_bpb | 6 | 2.537243186 | 0.113569846 |
| gain | life_bits_clean | 6 | 1959414.159898711 | 99439.893630931 |
| gain | life_bits_junk | 6 | 3175430.743330745 | 187705.153839514 |
| gain | life_bpb_clean | 6 | 2.433341543 | 0.040307092 |
| gain | life_bpb_junk | 6 | 3.818431411 | 0.232157226 |
| gain | junk_share | 6 | 0.511584989 | 0.027422050 |
| gain | rho | 6 | 0.251663208 | 0.000383100 |
| gain | test_rho | 6 | 0.230974197 | 0.074235509 |
| gain | wall_seconds | 6 | 1026.471408027 | 71.806144579 |
| gain | junk_share_by_kind.J1_random | 6 | 0.100943552 | 0.008836189 |
| gain | junk_share_by_kind.J2_corrupt | 6 | 0.105436517 | 0.012594108 |
| gain | junk_share_by_kind.J3_dup | 6 | 0.106894134 | 0.016615250 |
| gain | junk_share_by_kind.J4_boiler | 6 | 0.096634659 | 0.017333255 |
| gain | junk_share_by_kind.J5_shuffle | 6 | 0.101676127 | 0.014044737 |
| progress | clean_test_bpb | 6 | 2.516006941 | 0.101226750 |
| progress | life_bits_clean | 6 | 1960796.585232681 | 100577.570215008 |
| progress | life_bits_junk | 6 | 3179666.466021637 | 187326.987285944 |
| progress | life_bpb_clean | 6 | 2.434994517 | 0.040602356 |
| progress | life_bpb_junk | 6 | 3.823585205 | 0.232599485 |
| progress | junk_share | 6 | 0.513688476 | 0.027606765 |
| progress | rho | 6 | 0.248481649 | 0.000167451 |
| progress | test_rho | 6 | 0.324350357 | 0.254800677 |
| progress | wall_seconds | 6 | 894.010274963 | 168.742351131 |
| progress | junk_share_by_kind.J1_random | 6 | 0.108457243 | 0.009717966 |
| progress | junk_share_by_kind.J2_corrupt | 6 | 0.098537514 | 0.011801241 |
| progress | junk_share_by_kind.J3_dup | 6 | 0.107165159 | 0.016164467 |
| progress | junk_share_by_kind.J4_boiler | 6 | 0.099334725 | 0.017281579 |
| progress | junk_share_by_kind.J5_shuffle | 6 | 0.100193835 | 0.013469904 |

Smoke output:

```json
{
  "ok": true,
  "sha256_e10m": "5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97",
  "threads": 1,
  "controller": "feedback .01 * EMA(.999) score std; strict ties; unscored<8 rng3000 .25",
  "allocation": {
    "fixed": {
      "n_bytes": 262144,
      "chosen": 65536,
      "rho": 0.25,
      "bits_per_byte": 5.362340923466673,
      "life_bits_clean": 350596.9250639621,
      "life_bits_junk": 1055108.5739772853,
      "life_bpb_clean": 2.6748422627560586,
      "life_bpb_junk": 8.049839584177287,
      "bits_by_kind": {
        "clean": 350596.9250639621,
        "J1_random": 1055108.5739772853
      },
      "bytes_by_kind": {
        "clean": 131072,
        "J1_random": 131072
      },
      "steps_by_kind": {
        "clean": 32768,
        "J1_random": 32768
      },
      "junk_share": 0.5,
      "junk_share_by_kind": {
        "J1_random": 0.5
      },
      "gru_bits_per_step": 6.38470447232183,
      "wall_seconds": 401.60173499863595,
      "estimated_400_document_wall_minutes": 48.526876312335176,
      "estimated_300_document_wall_minutes": 38.068497796745696
    },
    "entropy": {
      "n_bytes": 262144,
      "chosen": 65647,
      "rho": 0.2504234313964844,
      "bits_per_byte": 5.346386453625104,
      "life_bits_clean": 345975.2690629478,
      "life_bits_junk": 1055547.8614361514,
      "life_bpb_clean": 2.639581825736601,
      "life_bpb_junk": 8.053191081513607,
      "bits_by_kind": {
        "clean": 345975.2690629478,
        "J1_random": 1055547.8614361514
      },
      "bytes_by_kind": {
        "clean": 131072,
        "J1_random": 131072
      },
      "steps_by_kind": {
        "clean": 27522,
        "J1_random": 38125
      },
      "junk_share": 0.5807576888509757,
      "junk_share_by_kind": {
        "J1_random": 0.5807576888509757
      },
      "gru_bits_per_step": 6.99271297961501,
      "wall_seconds": 482.2634737640619,
      "estimated_400_document_wall_minutes": 58.27350307982415,
      "estimated_300_document_wall_minutes": 45.714558450551706
    },
    "gain": {
      "n_bytes": 262144,
      "chosen": 65774,
      "rho": 0.25090789794921875,
      "bits_per_byte": 5.350425517836985,
      "life_bits_clean": 347210.2936700223,
      "life_bits_junk": 1055371.6532778365,
      "life_bpb_clean": 2.6490043157197745,
      "life_bpb_junk": 8.051846719954197,
      "bits_by_kind": {
        "clean": 347210.2936700223,
        "J1_random": 1055371.6532778365
      },
      "bytes_by_kind": {
        "clean": 131072,
        "J1_random": 131072
      },
      "steps_by_kind": {
        "clean": 34126,
        "J1_random": 31648
      },
      "junk_share": 0.4811627694833825,
      "junk_share_by_kind": {
        "J1_random": 0.4811627694833825
      },
      "gru_bits_per_step": 6.559510409292175,
      "wall_seconds": 436.5713527081534,
      "estimated_400_document_wall_minutes": 52.75237178556854,
      "estimated_300_document_wall_minutes": 41.383326142127046
    },
    "progress": {
      "n_bytes": 262144,
      "chosen": 63927,
      "rho": 0.24386215209960938,
      "bits_per_byte": 5.359097377366792,
      "life_bits_clean": 349538.4822639002,
      "life_bits_junk": 1055316.74062854,
      "life_bpb_clean": 2.6667669850456255,
      "life_bpb_junk": 8.051427769687958,
      "bits_by_kind": {
        "clean": 349538.4822639002,
        "J1_random": 1055316.74062854
      },
      "bytes_by_kind": {
        "clean": 131072,
        "J1_random": 131072
      },
      "steps_by_kind": {
        "clean": 29111,
        "J1_random": 34816
      },
      "junk_share": 0.5446212085660206,
      "junk_share_by_kind": {
        "J1_random": 0.5446212085660206
      },
      "gru_bits_per_step": 6.681382154793387,
      "wall_seconds": 365.5448793908581,
      "estimated_400_document_wall_minutes": 44.17000625972869,
      "estimated_300_document_wall_minutes": 34.65060835892509
    }
  },
  "h35_checks": {
    "fixed": {
      "rule": "fixed",
      "seed": 0,
      "corpus": "h35_corpus.build(0)",
      "n_bytes": 65536,
      "chosen": 16384,
      "rho": 0.25,
      "bits_per_byte": 4.176429416139341,
      "life_bits_clean": 151777.48181830885,
      "life_bits_junk": 121928.99639779904,
      "life_bpb_clean": 3.7055049272048057,
      "life_bpb_junk": 4.9613035643635675,
      "bits_by_kind": {
        "clean": 151777.48181830885,
        "J1_random": 65865.73510034557,
        "J4_boiler": 3065.1064652960404,
        "J3_dup": 12286.477395899932,
        "J5_shuffle": 14216.485597982379,
        "J2_corrupt": 26495.191838275114
      },
      "bytes_by_kind": {
        "clean": 40960,
        "J1_random": 8192,
        "J4_boiler": 4096,
        "J3_dup": 4096,
        "J5_shuffle": 4096,
        "J2_corrupt": 4096
      },
      "steps_by_kind": {
        "clean": 10240,
        "J1_random": 2048,
        "J4_boiler": 1024,
        "J3_dup": 1024,
        "J5_shuffle": 1024,
        "J2_corrupt": 1024
      },
      "junk_share": 0.375,
      "junk_share_by_kind": {
        "J1_random": 0.125,
        "J4_boiler": 0.0625,
        "J3_dup": 0.0625,
        "J5_shuffle": 0.0625,
        "J2_corrupt": 0.0625
      },
      "gru_bits_per_step": 5.864067041721921,
      "wall_seconds": 81.26517284940928
    },
    "entropy": {
      "rule": "entropy",
      "seed": 0,
      "corpus": "h35_corpus.build(0)",
      "n_bytes": 65536,
      "chosen": 16090,
      "rho": 0.245513916015625,
      "bits_per_byte": 4.14631057894267,
      "life_bits_clean": 150624.0114366451,
      "life_bits_junk": 121108.59866494176,
      "life_bpb_clean": 3.677344029214968,
      "life_bpb_junk": 4.927921495155508,
      "bits_by_kind": {
        "clean": 150624.0114366451,
        "J1_random": 65869.36833359498,
        "J4_boiler": 3067.515191530357,
        "J3_dup": 12009.591586244489,
        "J5_shuffle": 14018.197172428818,
        "J2_corrupt": 26143.926381143116
      },
      "bytes_by_kind": {
        "clean": 40960,
        "J1_random": 8192,
        "J4_boiler": 4096,
        "J3_dup": 4096,
        "J5_shuffle": 4096,
        "J2_corrupt": 4096
      },
      "steps_by_kind": {
        "clean": 9736,
        "J1_random": 2332,
        "J4_boiler": 799,
        "J3_dup": 1030,
        "J5_shuffle": 1002,
        "J2_corrupt": 1191
      },
      "junk_share": 0.39490366687383466,
      "junk_share_by_kind": {
        "J1_random": 0.1449347420758235,
        "J4_boiler": 0.04965817277812306,
        "J3_dup": 0.06401491609695463,
        "J5_shuffle": 0.06227470478558111,
        "J2_corrupt": 0.0740211311373524
      },
      "gru_bits_per_step": 6.3210966629438206,
      "wall_seconds": 81.43412043713033
    },
    "gain": {
      "rule": "gain",
      "seed": 0,
      "corpus": "h35_corpus.build(0)",
      "n_bytes": 65536,
      "chosen": 16710,
      "rho": 0.254974365234375,
      "bits_per_byte": 4.161201241329245,
      "life_bits_clean": 151069.3625518657,
      "life_bits_junk": 121639.1219998877,
      "life_bpb_clean": 3.6882168591764084,
      "life_bpb_junk": 4.949508544917306,
      "bits_by_kind": {
        "clean": 151069.3625518657,
        "J1_random": 65873.59557428259,
        "J4_boiler": 3063.0834732739113,
        "J3_dup": 12212.15109468994,
        "J5_shuffle": 14124.099545048504,
        "J2_corrupt": 26366.192312592757
      },
      "bytes_by_kind": {
        "clean": 40960,
        "J1_random": 8192,
        "J4_boiler": 4096,
        "J3_dup": 4096,
        "J5_shuffle": 4096,
        "J2_corrupt": 4096
      },
      "steps_by_kind": {
        "clean": 10573,
        "J1_random": 2023,
        "J4_boiler": 973,
        "J3_dup": 1032,
        "J5_shuffle": 995,
        "J2_corrupt": 1114
      },
      "junk_share": 0.3672651107121484,
      "junk_share_by_kind": {
        "J1_random": 0.12106523040095751,
        "J4_boiler": 0.058228605625374026,
        "J3_dup": 0.06175942549371634,
        "J5_shuffle": 0.05954518252543387,
        "J2_corrupt": 0.06666666666666667
      },
      "gru_bits_per_step": 6.083199654822544,
      "wall_seconds": 82.54998962488025
    },
    "progress": {
      "rule": "progress",
      "seed": 0,
      "corpus": "h35_corpus.build(0)",
      "n_bytes": 65536,
      "chosen": 15764,
      "rho": 0.24053955078125,
      "bits_per_byte": 4.172676368264505,
      "life_bits_clean": 151629.75848010374,
      "life_bits_junk": 121830.75999047885,
      "life_bpb_clean": 3.7018984003931577,
      "life_bpb_junk": 4.95730631471675,
      "bits_by_kind": {
        "clean": 151629.75848010374,
        "J1_random": 65862.49880615006,
        "J4_boiler": 3084.7744970860344,
        "J3_dup": 12253.516829044836,
        "J5_shuffle": 14244.702860839927,
        "J2_corrupt": 26385.266997357987
      },
      "bytes_by_kind": {
        "clean": 40960,
        "J1_random": 8192,
        "J4_boiler": 4096,
        "J3_dup": 4096,
        "J5_shuffle": 4096,
        "J2_corrupt": 4096
      },
      "steps_by_kind": {
        "clean": 9549,
        "J1_random": 2170,
        "J4_boiler": 1051,
        "J3_dup": 994,
        "J5_shuffle": 1021,
        "J2_corrupt": 979
      },
      "junk_share": 0.39425272773407766,
      "junk_share_by_kind": {
        "J1_random": 0.13765541740674955,
        "J4_boiler": 0.0666708957117483,
        "J3_dup": 0.0630550621669627,
        "J5_shuffle": 0.06476782542501903,
        "J2_corrupt": 0.062103527023598074
      },
      "gru_bits_per_step": 5.896260817791376,
      "wall_seconds": 78.99270930793136
    }
  },
  "documents": 400
}
```

| Kernel id | Wall seconds |
|---|---:|
| ezzzzzx/h36-fixed-s0 | 1177.250996120 |
| ezzzzzx/h36-fixed-s1 | 1078.862699910 |
| ezzzzzx/h36-fixed-s2 | 664.750546532 |
| ezzzzzx/h36-fixed-s3 | 930.801460905 |
| ezzzzzx/h36-fixed-s4 | 1003.091440105 |
| ezzzzzx/h36-fixed-s5 | 1060.295862573 |
| ezzzzzx/h36-entropy-s0 | 1134.171795324 |
| ezzzzzx/h36-entropy-s1 | 983.810700994 |
| ezzzzzx/h36-entropy-s2 | 691.199538365 |
| ezzzzzx/h36-entropy-s3 | 1271.332383230 |
| ezzzzzx/h36-entropy-s4 | 979.239048345 |
| ezzzzzx/h36-entropy-s5 | 701.349767613 |
| ezzzzzx/h36-gain-s0 | 955.477814724 |
| ezzzzzx/h36-gain-s1 | 978.476952551 |
| ezzzzzx/h36-gain-s2 | 1078.037429799 |
| ezzzzzx/h36-gain-s3 | 1048.192009581 |
| ezzzzzx/h36-gain-s4 | 950.787794065 |
| ezzzzzx/h36-gain-s5 | 1147.856447442 |
| ezzzzzx/h36-progress-s0 | 1004.216636705 |
| ezzzzzx/h36-progress-s1 | 966.836880791 |
| ezzzzzx/h36-progress-s2 | 632.710360178 |
| ezzzzzx/h36-progress-s3 | 1078.907149444 |
| ezzzzzx/h36-progress-s4 | 690.287101195 |
| ezzzzzx/h36-progress-s5 | 991.103521467 |
