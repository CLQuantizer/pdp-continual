n_lives 20 decisions 1024 checks_every 16 panels 4 opacity 768 tau 256 dwell 8 eps 0.1 beta 0.3

| life | random | round_robin | curious_prog | curious_surp | curious_prog_noage | curious_surp_forget | curious_prog_lifetime | prog−robin | prog−surp |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.796875 | 0.760417 | 0.619792 | 0.609375 | 0.588542 | 0.562500 | 0.703125 | -0.140625 | 0.010417 |
| 1 | 0.786458 | 0.833333 | 0.572917 | 0.604167 | 0.567708 | 0.484375 | 0.619792 | -0.260417 | -0.031250 |
| 2 | 0.786458 | 0.880208 | 0.583333 | 0.536458 | 0.500000 | 0.494792 | 0.604167 | -0.296875 | 0.046875 |
| 3 | 0.822917 | 0.932292 | 0.593750 | 0.661458 | 0.520833 | 0.546875 | 0.593750 | -0.338542 | -0.067708 |
| 4 | 0.786458 | 0.895833 | 0.578125 | 0.645833 | 0.541667 | 0.515625 | 0.630208 | -0.317708 | -0.067708 |
| 5 | 0.802083 | 0.968750 | 0.640625 | 0.677083 | 0.458333 | 0.473958 | 0.677083 | -0.328125 | -0.036458 |
| 6 | 0.848958 | 0.958333 | 0.640625 | 0.692708 | 0.609375 | 0.604167 | 0.713542 | -0.317708 | -0.052083 |
| 7 | 0.760417 | 0.932292 | 0.541667 | 0.614583 | 0.479167 | 0.500000 | 0.614583 | -0.390625 | -0.072917 |
| 8 | 0.859375 | 0.833333 | 0.552083 | 0.718750 | 0.531250 | 0.515625 | 0.666667 | -0.281250 | -0.166667 |
| 9 | 0.859375 | 0.786458 | 0.598958 | 0.687500 | 0.536458 | 0.510417 | 0.578125 | -0.187500 | -0.088542 |
| 10 | 0.822917 | 0.770833 | 0.614583 | 0.635417 | 0.609375 | 0.505208 | 0.692708 | -0.156250 | -0.020833 |
| 11 | 0.713542 | 0.807292 | 0.583333 | 0.692708 | 0.572917 | 0.578125 | 0.697917 | -0.223958 | -0.109375 |
| 12 | 0.791667 | 0.921875 | 0.536458 | 0.656250 | 0.479167 | 0.458333 | 0.750000 | -0.385417 | -0.119792 |
| 13 | 0.843750 | 0.932292 | 0.572917 | 0.625000 | 0.484375 | 0.473958 | 0.718750 | -0.359375 | -0.052083 |
| 14 | 0.739583 | 0.786458 | 0.583333 | 0.598958 | 0.567708 | 0.520833 | 0.713542 | -0.203125 | -0.015625 |
| 15 | 0.776042 | 0.812500 | 0.473958 | 0.567708 | 0.421875 | 0.463542 | 0.661458 | -0.338542 | -0.093750 |
| 16 | 0.802083 | 0.838542 | 0.630208 | 0.687500 | 0.619792 | 0.609375 | 0.671875 | -0.208333 | -0.057292 |
| 17 | 0.729167 | 0.833333 | 0.598958 | 0.593750 | 0.557292 | 0.531250 | 0.718750 | -0.234375 | 0.005208 |
| 18 | 0.796875 | 0.786458 | 0.635417 | 0.598958 | 0.598958 | 0.578125 | 0.723958 | -0.151042 | 0.036458 |
| 19 | 0.776042 | 0.921875 | 0.489583 | 0.536458 | 0.453125 | 0.437500 | 0.593750 | -0.432292 | -0.046875 |

| contrast | accuracy | ci95_lo | ci95_hi | logloss_nats | ci95_lo | ci95_hi | n_acc_gt_0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| curious_prog-round_robin | -0.277604 | -0.314063 | -0.240625 | 3.162293 | 2.707610 | 3.667728 | 0/20 |
| curious_prog-curious_surp | -0.050000 | -0.073177 | -0.027604 | 0.217629 | -0.131291 | 0.548981 | 4/20 |

| policy | tv | static | fast | slow |
|---|---:|---:|---:|---:|
| random | 0.251563 | 0.255859 | 0.243750 | 0.248828 |
| round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| curious_prog | 0.056250 | 0.784766 | 0.064062 | 0.094922 |
| curious_surp | 0.730469 | 0.042188 | 0.149609 | 0.077734 |
| curious_prog_noage | 0.032031 | 0.844141 | 0.043750 | 0.080078 |
| curious_surp_forget | 0.889453 | 0.033984 | 0.036719 | 0.039844 |
| curious_prog_lifetime | 0.076172 | 0.700000 | 0.098437 | 0.125391 |

| bar | observed | required | result |
|---|---|---|---|
| 1 curiosity beats round robin | -0.277604 ci_excludes_0 true | >= 0.03 and CI excludes 0 | FAIL |
| 2 ignores the tv | prog 0.056250 surp 0.730469 | prog <= 0.15 and surp >= 0.35 | PASS |
| 3 watches what changes | fast 0.064062 slow 0.094922 static 0.784766 | fast > slow > static | FAIL |

per-life share

| life | policy | tv | static | fast | slow |
|---:|---|---:|---:|---:|---:|
| 0 | random | 0.226562 | 0.289062 | 0.250000 | 0.234375 |
| 0 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 0 | curious_prog | 0.046875 | 0.773438 | 0.078125 | 0.101562 |
| 0 | curious_surp | 0.742188 | 0.031250 | 0.148438 | 0.078125 |
| 0 | curious_prog_noage | 0.023438 | 0.835938 | 0.046875 | 0.093750 |
| 0 | curious_surp_forget | 0.890625 | 0.023438 | 0.039062 | 0.046875 |
| 0 | curious_prog_lifetime | 0.078125 | 0.687500 | 0.093750 | 0.140625 |
| 1 | random | 0.289062 | 0.289062 | 0.218750 | 0.203125 |
| 1 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 1 | curious_prog | 0.046875 | 0.781250 | 0.054688 | 0.117188 |
| 1 | curious_surp | 0.750000 | 0.039062 | 0.117188 | 0.093750 |
| 1 | curious_prog_noage | 0.015625 | 0.828125 | 0.046875 | 0.109375 |
| 1 | curious_surp_forget | 0.875000 | 0.031250 | 0.031250 | 0.062500 |
| 1 | curious_prog_lifetime | 0.062500 | 0.679688 | 0.093750 | 0.164062 |
| 2 | random | 0.226562 | 0.257812 | 0.273438 | 0.242188 |
| 2 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 2 | curious_prog | 0.046875 | 0.796875 | 0.062500 | 0.093750 |
| 2 | curious_surp | 0.804688 | 0.031250 | 0.093750 | 0.070312 |
| 2 | curious_prog_noage | 0.015625 | 0.859375 | 0.039062 | 0.085938 |
| 2 | curious_surp_forget | 0.898438 | 0.023438 | 0.039062 | 0.039062 |
| 2 | curious_prog_lifetime | 0.062500 | 0.710938 | 0.093750 | 0.132812 |
| 3 | random | 0.281250 | 0.226562 | 0.304688 | 0.187500 |
| 3 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 3 | curious_prog | 0.054688 | 0.781250 | 0.062500 | 0.101562 |
| 3 | curious_surp | 0.750000 | 0.039062 | 0.132812 | 0.078125 |
| 3 | curious_prog_noage | 0.023438 | 0.851562 | 0.039062 | 0.085938 |
| 3 | curious_surp_forget | 0.906250 | 0.023438 | 0.031250 | 0.039062 |
| 3 | curious_prog_lifetime | 0.070312 | 0.710938 | 0.093750 | 0.125000 |
| 4 | random | 0.289062 | 0.203125 | 0.289062 | 0.218750 |
| 4 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 4 | curious_prog | 0.062500 | 0.757812 | 0.062500 | 0.117188 |
| 4 | curious_surp | 0.648438 | 0.054688 | 0.218750 | 0.078125 |
| 4 | curious_prog_noage | 0.039062 | 0.812500 | 0.046875 | 0.101562 |
| 4 | curious_surp_forget | 0.859375 | 0.039062 | 0.046875 | 0.054688 |
| 4 | curious_prog_lifetime | 0.078125 | 0.695312 | 0.093750 | 0.132812 |
| 5 | random | 0.203125 | 0.265625 | 0.187500 | 0.343750 |
| 5 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 5 | curious_prog | 0.101562 | 0.750000 | 0.062500 | 0.085938 |
| 5 | curious_surp | 0.734375 | 0.039062 | 0.132812 | 0.093750 |
| 5 | curious_prog_noage | 0.078125 | 0.820312 | 0.031250 | 0.070312 |
| 5 | curious_surp_forget | 0.898438 | 0.031250 | 0.031250 | 0.039062 |
| 5 | curious_prog_lifetime | 0.109375 | 0.679688 | 0.085938 | 0.125000 |
| 6 | random | 0.234375 | 0.273438 | 0.187500 | 0.304688 |
| 6 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 6 | curious_prog | 0.046875 | 0.789062 | 0.062500 | 0.101562 |
| 6 | curious_surp | 0.742188 | 0.046875 | 0.109375 | 0.101562 |
| 6 | curious_prog_noage | 0.015625 | 0.843750 | 0.046875 | 0.093750 |
| 6 | curious_surp_forget | 0.882812 | 0.039062 | 0.031250 | 0.046875 |
| 6 | curious_prog_lifetime | 0.070312 | 0.703125 | 0.101562 | 0.125000 |
| 7 | random | 0.257812 | 0.257812 | 0.218750 | 0.265625 |
| 7 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 7 | curious_prog | 0.054688 | 0.789062 | 0.070312 | 0.085938 |
| 7 | curious_surp | 0.781250 | 0.054688 | 0.093750 | 0.070312 |
| 7 | curious_prog_noage | 0.031250 | 0.851562 | 0.046875 | 0.070312 |
| 7 | curious_surp_forget | 0.875000 | 0.046875 | 0.039062 | 0.039062 |
| 7 | curious_prog_lifetime | 0.070312 | 0.703125 | 0.101562 | 0.125000 |
| 8 | random | 0.242188 | 0.257812 | 0.281250 | 0.218750 |
| 8 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 8 | curious_prog | 0.054688 | 0.796875 | 0.062500 | 0.085938 |
| 8 | curious_surp | 0.609375 | 0.031250 | 0.281250 | 0.078125 |
| 8 | curious_prog_noage | 0.046875 | 0.843750 | 0.039062 | 0.070312 |
| 8 | curious_surp_forget | 0.914062 | 0.023438 | 0.031250 | 0.031250 |
| 8 | curious_prog_lifetime | 0.085938 | 0.703125 | 0.093750 | 0.117188 |
| 9 | random | 0.210938 | 0.226562 | 0.296875 | 0.265625 |
| 9 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 9 | curious_prog | 0.046875 | 0.820312 | 0.054688 | 0.078125 |
| 9 | curious_surp | 0.664062 | 0.039062 | 0.242188 | 0.054688 |
| 9 | curious_prog_noage | 0.015625 | 0.898438 | 0.031250 | 0.054688 |
| 9 | curious_surp_forget | 0.914062 | 0.031250 | 0.031250 | 0.023438 |
| 9 | curious_prog_lifetime | 0.062500 | 0.734375 | 0.085938 | 0.117188 |
| 10 | random | 0.226562 | 0.289062 | 0.265625 | 0.218750 |
| 10 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 10 | curious_prog | 0.062500 | 0.757812 | 0.070312 | 0.109375 |
| 10 | curious_surp | 0.609375 | 0.054688 | 0.257812 | 0.078125 |
| 10 | curious_prog_noage | 0.046875 | 0.796875 | 0.062500 | 0.093750 |
| 10 | curious_surp_forget | 0.851562 | 0.046875 | 0.054688 | 0.046875 |
| 10 | curious_prog_lifetime | 0.078125 | 0.687500 | 0.109375 | 0.125000 |
| 11 | random | 0.273438 | 0.257812 | 0.242188 | 0.226562 |
| 11 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 11 | curious_prog | 0.046875 | 0.804688 | 0.054688 | 0.093750 |
| 11 | curious_surp | 0.757812 | 0.039062 | 0.125000 | 0.078125 |
| 11 | curious_prog_noage | 0.015625 | 0.867188 | 0.031250 | 0.085938 |
| 11 | curious_surp_forget | 0.906250 | 0.031250 | 0.023438 | 0.039062 |
| 11 | curious_prog_lifetime | 0.070312 | 0.710938 | 0.101562 | 0.117188 |
| 12 | random | 0.226562 | 0.304688 | 0.179688 | 0.289062 |
| 12 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 12 | curious_prog | 0.054688 | 0.804688 | 0.054688 | 0.085938 |
| 12 | curious_surp | 0.562500 | 0.031250 | 0.343750 | 0.062500 |
| 12 | curious_prog_noage | 0.031250 | 0.882812 | 0.031250 | 0.054688 |
| 12 | curious_surp_forget | 0.906250 | 0.023438 | 0.039062 | 0.031250 |
| 12 | curious_prog_lifetime | 0.078125 | 0.710938 | 0.093750 | 0.117188 |
| 13 | random | 0.234375 | 0.195312 | 0.312500 | 0.257812 |
| 13 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 13 | curious_prog | 0.062500 | 0.796875 | 0.054688 | 0.085938 |
| 13 | curious_surp | 0.781250 | 0.039062 | 0.093750 | 0.085938 |
| 13 | curious_prog_noage | 0.031250 | 0.859375 | 0.039062 | 0.070312 |
| 13 | curious_surp_forget | 0.914062 | 0.023438 | 0.031250 | 0.031250 |
| 13 | curious_prog_lifetime | 0.078125 | 0.718750 | 0.093750 | 0.109375 |
| 14 | random | 0.343750 | 0.218750 | 0.203125 | 0.234375 |
| 14 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 14 | curious_prog | 0.046875 | 0.804688 | 0.062500 | 0.085938 |
| 14 | curious_surp | 0.820312 | 0.046875 | 0.078125 | 0.054688 |
| 14 | curious_prog_noage | 0.031250 | 0.859375 | 0.031250 | 0.078125 |
| 14 | curious_surp_forget | 0.890625 | 0.046875 | 0.031250 | 0.031250 |
| 14 | curious_prog_lifetime | 0.070312 | 0.703125 | 0.093750 | 0.132812 |
| 15 | random | 0.273438 | 0.218750 | 0.171875 | 0.335938 |
| 15 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 15 | curious_prog | 0.054688 | 0.796875 | 0.062500 | 0.085938 |
| 15 | curious_surp | 0.789062 | 0.039062 | 0.093750 | 0.078125 |
| 15 | curious_prog_noage | 0.039062 | 0.843750 | 0.039062 | 0.078125 |
| 15 | curious_surp_forget | 0.898438 | 0.031250 | 0.031250 | 0.039062 |
| 15 | curious_prog_lifetime | 0.078125 | 0.695312 | 0.101562 | 0.125000 |
| 16 | random | 0.250000 | 0.289062 | 0.250000 | 0.210938 |
| 16 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 16 | curious_prog | 0.062500 | 0.765625 | 0.070312 | 0.101562 |
| 16 | curious_surp | 0.796875 | 0.046875 | 0.101562 | 0.054688 |
| 16 | curious_prog_noage | 0.039062 | 0.828125 | 0.054688 | 0.078125 |
| 16 | curious_surp_forget | 0.875000 | 0.046875 | 0.039062 | 0.039062 |
| 16 | curious_prog_lifetime | 0.085938 | 0.695312 | 0.101562 | 0.117188 |
| 17 | random | 0.289062 | 0.273438 | 0.234375 | 0.203125 |
| 17 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 17 | curious_prog | 0.046875 | 0.773438 | 0.085938 | 0.093750 |
| 17 | curious_surp | 0.765625 | 0.046875 | 0.140625 | 0.046875 |
| 17 | curious_prog_noage | 0.023438 | 0.835938 | 0.070312 | 0.070312 |
| 17 | curious_surp_forget | 0.875000 | 0.039062 | 0.062500 | 0.023438 |
| 17 | curious_prog_lifetime | 0.070312 | 0.695312 | 0.117188 | 0.117188 |
| 18 | random | 0.234375 | 0.289062 | 0.242188 | 0.234375 |
| 18 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 18 | curious_prog | 0.054688 | 0.789062 | 0.070312 | 0.085938 |
| 18 | curious_surp | 0.750000 | 0.054688 | 0.085938 | 0.109375 |
| 18 | curious_prog_noage | 0.031250 | 0.835938 | 0.054688 | 0.078125 |
| 18 | curious_surp_forget | 0.867188 | 0.046875 | 0.039062 | 0.046875 |
| 18 | curious_prog_lifetime | 0.078125 | 0.695312 | 0.109375 | 0.117188 |
| 19 | random | 0.218750 | 0.234375 | 0.265625 | 0.281250 |
| 19 | round_robin | 0.250000 | 0.250000 | 0.250000 | 0.250000 |
| 19 | curious_prog | 0.070312 | 0.765625 | 0.062500 | 0.101562 |
| 19 | curious_surp | 0.750000 | 0.039062 | 0.101562 | 0.109375 |
| 19 | curious_prog_noage | 0.046875 | 0.828125 | 0.046875 | 0.078125 |
| 19 | curious_surp_forget | 0.890625 | 0.031250 | 0.031250 | 0.046875 |
| 19 | curious_prog_lifetime | 0.085938 | 0.679688 | 0.109375 | 0.125000 |

per-panel life accuracy, mean over lives

| policy | static | fast | slow |
|---|---:|---:|---:|
| random | 0.979688 | 0.557813 | 0.847656 |
| round_robin | 0.994531 | 0.706250 | 0.878125 |
| curious_prog | 0.994531 | 0.232813 | 0.518750 |
| curious_surp | 0.994531 | 0.343750 | 0.557813 |
| curious_prog_noage | 0.994531 | 0.193750 | 0.416406 |
| curious_surp_forget | 0.994531 | 0.185156 | 0.375000 |
| curious_prog_lifetime | 0.994531 | 0.335156 | 0.671875 |

per-panel life logloss nats, mean over lives

| policy | static | fast | slow | life |
|---|---:|---:|---:|---:|
| random | 0.086891 | 4.933805 | 1.654783 | 2.225160 |
| round_robin | 0.028043 | 2.693825 | 1.201863 | 1.307910 |
| curious_prog | 0.028736 | 7.987587 | 5.394289 | 4.470204 |
| curious_surp | 0.032267 | 7.325435 | 5.400022 | 4.252575 |
| curious_prog_noage | 0.028765 | 8.008276 | 6.739490 | 4.925510 |
| curious_surp_forget | 0.032945 | 8.139350 | 7.279746 | 5.150680 |
| curious_prog_lifetime | 0.029111 | 6.858516 | 3.507126 | 3.464918 |

per-life per-panel accuracy

| life | policy | static | fast | slow | logloss |
|---:|---|---:|---:|---:|---:|
| 0 | random | 1.000000 | 0.531250 | 0.859375 | 2.250365 |
| 0 | round_robin | 1.000000 | 0.437500 | 0.843750 | 2.262645 |
| 0 | curious_prog | 1.000000 | 0.265625 | 0.593750 | 4.906375 |
| 0 | curious_surp | 1.000000 | 0.359375 | 0.468750 | 4.649235 |
| 0 | curious_prog_noage | 1.000000 | 0.218750 | 0.546875 | 4.449888 |
| 0 | curious_surp_forget | 1.000000 | 0.265625 | 0.421875 | 5.673217 |
| 0 | curious_prog_lifetime | 1.000000 | 0.296875 | 0.812500 | 3.754098 |
| 1 | random | 0.984375 | 0.531250 | 0.843750 | 2.415074 |
| 1 | round_robin | 0.984375 | 0.546875 | 0.968750 | 1.747776 |
| 1 | curious_prog | 0.984375 | 0.234375 | 0.500000 | 5.188176 |
| 1 | curious_surp | 0.984375 | 0.312500 | 0.515625 | 5.779247 |
| 1 | curious_prog_noage | 0.984375 | 0.234375 | 0.484375 | 5.304058 |
| 1 | curious_surp_forget | 0.984375 | 0.109375 | 0.359375 | 6.693168 |
| 1 | curious_prog_lifetime | 0.984375 | 0.281250 | 0.593750 | 3.568238 |
| 2 | random | 1.000000 | 0.609375 | 0.750000 | 2.081081 |
| 2 | round_robin | 1.000000 | 0.875000 | 0.765625 | 0.768613 |
| 2 | curious_prog | 1.000000 | 0.281250 | 0.468750 | 4.175539 |
| 2 | curious_surp | 1.000000 | 0.187500 | 0.421875 | 5.131901 |
| 2 | curious_prog_noage | 1.000000 | 0.109375 | 0.390625 | 4.316390 |
| 2 | curious_surp_forget | 1.000000 | 0.156250 | 0.328125 | 3.561700 |
| 2 | curious_prog_lifetime | 1.000000 | 0.421875 | 0.390625 | 3.147218 |
| 3 | random | 1.000000 | 0.609375 | 0.859375 | 2.370777 |
| 3 | round_robin | 0.984375 | 0.937500 | 0.875000 | 0.630547 |
| 3 | curious_prog | 0.984375 | 0.250000 | 0.546875 | 3.833425 |
| 3 | curious_surp | 0.984375 | 0.312500 | 0.687500 | 5.271259 |
| 3 | curious_prog_noage | 0.984375 | 0.156250 | 0.421875 | 5.686687 |
| 3 | curious_surp_forget | 0.984375 | 0.203125 | 0.453125 | 5.679206 |
| 3 | curious_prog_lifetime | 0.984375 | 0.312500 | 0.484375 | 4.362632 |
| 4 | random | 0.953125 | 0.562500 | 0.843750 | 1.740472 |
| 4 | round_robin | 1.000000 | 0.937500 | 0.750000 | 0.707272 |
| 4 | curious_prog | 1.000000 | 0.265625 | 0.468750 | 3.530943 |
| 4 | curious_surp | 1.000000 | 0.468750 | 0.468750 | 3.440521 |
| 4 | curious_prog_noage | 1.000000 | 0.203125 | 0.421875 | 4.216465 |
| 4 | curious_surp_forget | 1.000000 | 0.250000 | 0.296875 | 4.201831 |
| 4 | curious_prog_lifetime | 1.000000 | 0.390625 | 0.500000 | 3.428291 |
| 5 | random | 1.000000 | 0.468750 | 0.937500 | 2.737336 |
| 5 | round_robin | 0.984375 | 1.000000 | 0.921875 | 0.509122 |
| 5 | curious_prog | 0.984375 | 0.218750 | 0.718750 | 6.411012 |
| 5 | curious_surp | 0.984375 | 0.359375 | 0.687500 | 5.022094 |
| 5 | curious_prog_noage | 0.984375 | 0.140625 | 0.250000 | 8.987960 |
| 5 | curious_surp_forget | 0.984375 | 0.187500 | 0.250000 | 7.751537 |
| 5 | curious_prog_lifetime | 0.984375 | 0.406250 | 0.640625 | 3.603674 |
| 6 | random | 1.000000 | 0.593750 | 0.953125 | 1.316820 |
| 6 | round_robin | 1.000000 | 0.968750 | 0.906250 | 0.576678 |
| 6 | curious_prog | 1.000000 | 0.343750 | 0.578125 | 5.068375 |
| 6 | curious_surp | 1.000000 | 0.453125 | 0.625000 | 3.986301 |
| 6 | curious_prog_noage | 1.000000 | 0.250000 | 0.578125 | 5.336206 |
| 6 | curious_surp_forget | 1.000000 | 0.234375 | 0.578125 | 5.336536 |
| 6 | curious_prog_lifetime | 1.000000 | 0.406250 | 0.734375 | 3.029173 |
| 7 | random | 1.000000 | 0.406250 | 0.875000 | 2.752059 |
| 7 | round_robin | 0.984375 | 0.937500 | 0.875000 | 0.671749 |
| 7 | curious_prog | 0.984375 | 0.218750 | 0.421875 | 4.829772 |
| 7 | curious_surp | 0.984375 | 0.218750 | 0.640625 | 3.824007 |
| 7 | curious_prog_noage | 0.984375 | 0.234375 | 0.218750 | 6.016351 |
| 7 | curious_surp_forget | 0.984375 | 0.156250 | 0.359375 | 4.868059 |
| 7 | curious_prog_lifetime | 0.984375 | 0.328125 | 0.531250 | 4.244670 |
| 8 | random | 1.000000 | 0.656250 | 0.921875 | 1.497658 |
| 8 | round_robin | 1.000000 | 0.500000 | 1.000000 | 1.572349 |
| 8 | curious_prog | 1.000000 | 0.171875 | 0.484375 | 4.502180 |
| 8 | curious_surp | 1.000000 | 0.468750 | 0.687500 | 3.363102 |
| 8 | curious_prog_noage | 1.000000 | 0.265625 | 0.328125 | 4.597522 |
| 8 | curious_surp_forget | 1.000000 | 0.234375 | 0.312500 | 4.790821 |
| 8 | curious_prog_lifetime | 1.000000 | 0.312500 | 0.687500 | 4.034532 |
| 9 | random | 0.953125 | 0.734375 | 0.890625 | 1.628012 |
| 9 | round_robin | 1.000000 | 0.500000 | 0.859375 | 2.406475 |
| 9 | curious_prog | 1.000000 | 0.140625 | 0.656250 | 4.577019 |
| 9 | curious_surp | 1.000000 | 0.390625 | 0.671875 | 3.748654 |
| 9 | curious_prog_noage | 1.000000 | 0.125000 | 0.484375 | 3.888591 |
| 9 | curious_surp_forget | 1.000000 | 0.203125 | 0.328125 | 6.144728 |
| 9 | curious_prog_lifetime | 1.000000 | 0.187500 | 0.546875 | 4.836412 |
| 10 | random | 0.968750 | 0.578125 | 0.921875 | 1.989117 |
| 10 | round_robin | 1.000000 | 0.531250 | 0.781250 | 2.040717 |
| 10 | curious_prog | 1.000000 | 0.296875 | 0.546875 | 4.087945 |
| 10 | curious_surp | 1.000000 | 0.468750 | 0.437500 | 3.933453 |
| 10 | curious_prog_noage | 1.000000 | 0.265625 | 0.562500 | 3.018456 |
| 10 | curious_surp_forget | 1.000000 | 0.234375 | 0.281250 | 5.418544 |
| 10 | curious_prog_lifetime | 1.000000 | 0.359375 | 0.718750 | 3.357131 |
| 11 | random | 0.937500 | 0.546875 | 0.656250 | 2.743695 |
| 11 | round_robin | 1.000000 | 0.484375 | 0.937500 | 1.874734 |
| 11 | curious_prog | 1.000000 | 0.218750 | 0.531250 | 3.917394 |
| 11 | curious_surp | 1.000000 | 0.375000 | 0.703125 | 2.940771 |
| 11 | curious_prog_noage | 1.000000 | 0.218750 | 0.500000 | 2.802081 |
| 11 | curious_surp_forget | 1.000000 | 0.250000 | 0.484375 | 3.493775 |
| 11 | curious_prog_lifetime | 1.000000 | 0.343750 | 0.750000 | 2.409990 |
| 12 | random | 0.984375 | 0.562500 | 0.828125 | 3.006096 |
| 12 | round_robin | 0.984375 | 0.906250 | 0.875000 | 0.678418 |
| 12 | curious_prog | 0.984375 | 0.156250 | 0.468750 | 3.045273 |
| 12 | curious_surp | 0.984375 | 0.531250 | 0.453125 | 2.924949 |
| 12 | curious_prog_noage | 0.984375 | 0.093750 | 0.359375 | 4.131205 |
| 12 | curious_surp_forget | 0.984375 | 0.109375 | 0.281250 | 3.338819 |
| 12 | curious_prog_lifetime | 0.984375 | 0.375000 | 0.890625 | 2.836265 |
| 13 | random | 1.000000 | 0.640625 | 0.890625 | 1.788728 |
| 13 | round_robin | 0.984375 | 0.968750 | 0.843750 | 0.692473 |
| 13 | curious_prog | 0.984375 | 0.234375 | 0.500000 | 4.887667 |
| 13 | curious_surp | 0.984375 | 0.281250 | 0.609375 | 4.657506 |
| 13 | curious_prog_noage | 0.984375 | 0.109375 | 0.359375 | 6.135114 |
| 13 | curious_surp_forget | 0.984375 | 0.093750 | 0.343750 | 6.030086 |
| 13 | curious_prog_lifetime | 0.984375 | 0.296875 | 0.875000 | 2.968228 |
| 14 | random | 0.921875 | 0.406250 | 0.890625 | 2.010598 |
| 14 | round_robin | 1.000000 | 0.468750 | 0.890625 | 1.784447 |
| 14 | curious_prog | 1.000000 | 0.218750 | 0.531250 | 4.201620 |
| 14 | curious_surp | 1.000000 | 0.265625 | 0.531250 | 3.382733 |
| 14 | curious_prog_noage | 1.000000 | 0.203125 | 0.500000 | 4.322894 |
| 14 | curious_surp_forget | 1.000000 | 0.187500 | 0.375000 | 3.990250 |
| 14 | curious_prog_lifetime | 1.000000 | 0.375000 | 0.765625 | 2.976929 |
| 15 | random | 0.984375 | 0.453125 | 0.890625 | 2.396287 |
| 15 | round_robin | 0.984375 | 0.515625 | 0.937500 | 1.497110 |
| 15 | curious_prog | 0.984375 | 0.187500 | 0.250000 | 3.716693 |
| 15 | curious_surp | 0.984375 | 0.250000 | 0.468750 | 4.600348 |
| 15 | curious_prog_noage | 0.984375 | 0.109375 | 0.171875 | 4.097628 |
| 15 | curious_surp_forget | 0.984375 | 0.078125 | 0.328125 | 4.614731 |
| 15 | curious_prog_lifetime | 0.984375 | 0.296875 | 0.703125 | 3.020704 |
| 16 | random | 0.984375 | 0.562500 | 0.859375 | 2.280691 |
| 16 | round_robin | 1.000000 | 0.515625 | 1.000000 | 1.308826 |
| 16 | curious_prog | 1.000000 | 0.234375 | 0.656250 | 3.477676 |
| 16 | curious_surp | 1.000000 | 0.375000 | 0.687500 | 3.504682 |
| 16 | curious_prog_noage | 1.000000 | 0.234375 | 0.625000 | 5.875330 |
| 16 | curious_surp_forget | 1.000000 | 0.171875 | 0.656250 | 4.166300 |
| 16 | curious_prog_lifetime | 1.000000 | 0.359375 | 0.656250 | 3.282998 |
| 17 | random | 0.921875 | 0.531250 | 0.734375 | 3.645045 |
| 17 | round_robin | 1.000000 | 0.500000 | 1.000000 | 1.760868 |
| 17 | curious_prog | 1.000000 | 0.203125 | 0.593750 | 5.540173 |
| 17 | curious_surp | 1.000000 | 0.343750 | 0.437500 | 6.196887 |
| 17 | curious_prog_noage | 1.000000 | 0.250000 | 0.421875 | 6.565747 |
| 17 | curious_surp_forget | 1.000000 | 0.218750 | 0.375000 | 7.213741 |
| 17 | curious_prog_lifetime | 1.000000 | 0.281250 | 0.875000 | 3.273281 |
| 18 | random | 1.000000 | 0.640625 | 0.750000 | 1.657699 |
| 18 | round_robin | 1.000000 | 0.593750 | 0.765625 | 2.084887 |
| 18 | curious_prog | 1.000000 | 0.296875 | 0.609375 | 3.939952 |
| 18 | curious_surp | 1.000000 | 0.218750 | 0.578125 | 3.876788 |
| 18 | curious_prog_noage | 1.000000 | 0.265625 | 0.531250 | 3.337437 |
| 18 | curious_surp_forget | 1.000000 | 0.203125 | 0.531250 | 4.301596 |
| 18 | curious_prog_lifetime | 1.000000 | 0.390625 | 0.781250 | 2.866693 |
| 19 | random | 1.000000 | 0.531250 | 0.796875 | 2.195585 |
| 19 | round_robin | 1.000000 | 1.000000 | 0.765625 | 0.582499 |
| 19 | curious_prog | 1.000000 | 0.218750 | 0.250000 | 5.566865 |
| 19 | curious_surp | 1.000000 | 0.234375 | 0.375000 | 4.817056 |
| 19 | curious_prog_noage | 1.000000 | 0.187500 | 0.171875 | 5.424196 |
| 19 | curious_surp_forget | 1.000000 | 0.156250 | 0.156250 | 5.744964 |
| 19 | curious_prog_lifetime | 1.000000 | 0.281250 | 0.500000 | 4.297193 |

life 0 curious_prog panel choices, first 32

| choice | t | panel | kind | mode |
|---:|---:|---:|---|---|
| 0 | 0 | 0 | static | first |
| 1 | 8 | 1 | tv | first |
| 2 | 16 | 2 | slow | first |
| 3 | 24 | 3 | fast | first |
| 4 | 32 | 2 | slow | greedy |
| 5 | 40 | 2 | slow | greedy |
| 6 | 48 | 2 | slow | greedy |
| 7 | 56 | 2 | slow | greedy |
| 8 | 64 | 2 | slow | eps |
| 9 | 72 | 0 | static | greedy |
| 10 | 80 | 0 | static | greedy |
| 11 | 88 | 0 | static | greedy |
| 12 | 96 | 0 | static | greedy |
| 13 | 104 | 3 | fast | greedy |
| 14 | 112 | 0 | static | greedy |
| 15 | 120 | 2 | slow | greedy |
| 16 | 128 | 0 | static | greedy |
| 17 | 136 | 2 | slow | greedy |
| 18 | 144 | 0 | static | greedy |
| 19 | 152 | 0 | static | greedy |
| 20 | 160 | 3 | fast | greedy |
| 21 | 168 | 3 | fast | eps |
| 22 | 176 | 0 | static | greedy |
| 23 | 184 | 1 | tv | greedy |
| 24 | 192 | 0 | static | greedy |
| 25 | 200 | 0 | static | greedy |
| 26 | 208 | 0 | static | greedy |
| 27 | 216 | 0 | static | greedy |
| 28 | 224 | 0 | static | greedy |
| 29 | 232 | 0 | static | greedy |
| 30 | 240 | 0 | static | greedy |
| 31 | 248 | 0 | static | greedy |

0=static 1=tv 2=slow 3=fast 2=slow 2=slow 2=slow 2=slow 2=slow 0=static 0=static 0=static 0=static 3=fast 0=static 2=slow 0=static 2=slow 0=static 0=static 3=fast 3=fast 0=static 1=tv 0=static 0=static 0=static 0=static 0=static 0=static 0=static 0=static

life 0 panel kinds 0=static 1=tv 2=slow 3=fast

mechanism

| life | policy | n_choices | n_walk | n_dwell | n_first | n_eps | n_greedy | first_pass_t | wall_s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 30.270957 |
| 0 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 30.073242 |
| 0 | curious_prog | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 22.008970 |
| 0 | curious_surp | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 30.649374 |
| 0 | curious_prog_noage | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 34.100735 |
| 0 | curious_surp_forget | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 27.827952 |
| 0 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 36.227715 |
| 1 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.378909 |
| 1 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 16.984384 |
| 1 | curious_prog | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 19.759233 |
| 1 | curious_surp | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 19.191562 |
| 1 | curious_prog_noage | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 19.810700 |
| 1 | curious_surp_forget | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 17.608603 |
| 1 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 26.669193 |
| 2 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.552857 |
| 2 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.468346 |
| 2 | curious_prog | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 19.167398 |
| 2 | curious_surp | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 17.321755 |
| 2 | curious_prog_noage | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 19.650353 |
| 2 | curious_surp_forget | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 17.450214 |
| 2 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 25.327374 |
| 3 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.564741 |
| 3 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.061371 |
| 3 | curious_prog | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 19.761443 |
| 3 | curious_surp | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 16.876479 |
| 3 | curious_prog_noage | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 19.360215 |
| 3 | curious_surp_forget | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 16.908336 |
| 3 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 26.242962 |
| 4 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.587684 |
| 4 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.931266 |
| 4 | curious_prog | 128 | 0 | 1024 | 4 | 18 | 106 | 32 | 20.538119 |
| 4 | curious_surp | 128 | 0 | 1024 | 4 | 18 | 106 | 32 | 17.104642 |
| 4 | curious_prog_noage | 128 | 0 | 1024 | 4 | 18 | 106 | 32 | 20.012737 |
| 4 | curious_surp_forget | 128 | 0 | 1024 | 4 | 18 | 106 | 32 | 16.831257 |
| 4 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 18 | 106 | 32 | 25.099683 |
| 5 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 19.573486 |
| 5 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.732556 |
| 5 | curious_prog | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 19.341480 |
| 5 | curious_surp | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 17.972221 |
| 5 | curious_prog_noage | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 19.999649 |
| 5 | curious_surp_forget | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 17.042839 |
| 5 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 25.296645 |
| 6 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.256768 |
| 6 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.108107 |
| 6 | curious_prog | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 19.572524 |
| 6 | curious_surp | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 17.348204 |
| 6 | curious_prog_noage | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 19.461791 |
| 6 | curious_surp_forget | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 16.711103 |
| 6 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 25.599271 |
| 7 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.250843 |
| 7 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.275294 |
| 7 | curious_prog | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 19.558059 |
| 7 | curious_surp | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 17.146047 |
| 7 | curious_prog_noage | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 19.972644 |
| 7 | curious_surp_forget | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 17.252197 |
| 7 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 26.367613 |
| 8 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.371588 |
| 8 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.137067 |
| 8 | curious_prog | 128 | 0 | 1024 | 4 | 11 | 113 | 32 | 20.454077 |
| 8 | curious_surp | 128 | 0 | 1024 | 4 | 11 | 113 | 32 | 18.225971 |
| 8 | curious_prog_noage | 128 | 0 | 1024 | 4 | 11 | 113 | 32 | 21.968847 |
| 8 | curious_surp_forget | 128 | 0 | 1024 | 4 | 11 | 113 | 32 | 17.855118 |
| 8 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 11 | 113 | 32 | 27.875572 |
| 9 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.042015 |
| 9 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.492697 |
| 9 | curious_prog | 128 | 0 | 1024 | 4 | 5 | 119 | 32 | 20.682096 |
| 9 | curious_surp | 128 | 0 | 1024 | 4 | 5 | 119 | 32 | 19.665776 |
| 9 | curious_prog_noage | 128 | 0 | 1024 | 4 | 5 | 119 | 32 | 20.587548 |
| 9 | curious_surp_forget | 128 | 0 | 1024 | 4 | 5 | 119 | 32 | 17.899320 |
| 9 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 5 | 119 | 32 | 27.452868 |
| 10 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.919148 |
| 10 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.218878 |
| 10 | curious_prog | 128 | 0 | 1024 | 4 | 21 | 103 | 32 | 21.074125 |
| 10 | curious_surp | 128 | 0 | 1024 | 4 | 21 | 103 | 32 | 17.960418 |
| 10 | curious_prog_noage | 128 | 0 | 1024 | 4 | 21 | 103 | 32 | 20.811527 |
| 10 | curious_surp_forget | 128 | 0 | 1024 | 4 | 21 | 103 | 32 | 18.500771 |
| 10 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 21 | 103 | 32 | 27.026696 |
| 11 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.852098 |
| 11 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.415401 |
| 11 | curious_prog | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 20.272507 |
| 11 | curious_surp | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 18.420419 |
| 11 | curious_prog_noage | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 20.187113 |
| 11 | curious_surp_forget | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 17.430409 |
| 11 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 8 | 116 | 32 | 26.873211 |
| 12 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.709422 |
| 12 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.155318 |
| 12 | curious_prog | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 21.192793 |
| 12 | curious_surp | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 17.857899 |
| 12 | curious_prog_noage | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 21.361234 |
| 12 | curious_surp_forget | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 18.241430 |
| 12 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 27.202181 |
| 13 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.330499 |
| 13 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.566457 |
| 13 | curious_prog | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 21.151320 |
| 13 | curious_surp | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 18.860557 |
| 13 | curious_prog_noage | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 24.041708 |
| 13 | curious_surp_forget | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 18.692701 |
| 13 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 9 | 115 | 32 | 27.774641 |
| 14 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.341412 |
| 14 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.640049 |
| 14 | curious_prog | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 21.679830 |
| 14 | curious_surp | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 18.393735 |
| 14 | curious_prog_noage | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 20.552642 |
| 14 | curious_surp_forget | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 18.217180 |
| 14 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 12 | 112 | 32 | 27.873729 |
| 15 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 19.547540 |
| 15 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.447282 |
| 15 | curious_prog | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 21.202489 |
| 15 | curious_surp | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 18.826620 |
| 15 | curious_prog_noage | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 21.357716 |
| 15 | curious_surp_forget | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 18.198245 |
| 15 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 13 | 111 | 32 | 27.349232 |
| 16 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.483802 |
| 16 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.500306 |
| 16 | curious_prog | 128 | 0 | 1024 | 4 | 16 | 108 | 32 | 21.049850 |
| 16 | curious_surp | 128 | 0 | 1024 | 4 | 16 | 108 | 32 | 17.598831 |
| 16 | curious_prog_noage | 128 | 0 | 1024 | 4 | 16 | 108 | 32 | 20.940535 |
| 16 | curious_surp_forget | 128 | 0 | 1024 | 4 | 16 | 108 | 32 | 16.733283 |
| 16 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 16 | 108 | 32 | 26.095696 |
| 17 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.794966 |
| 17 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.421174 |
| 17 | curious_prog | 128 | 0 | 1024 | 4 | 15 | 109 | 32 | 21.417465 |
| 17 | curious_surp | 128 | 0 | 1024 | 4 | 15 | 109 | 32 | 17.545800 |
| 17 | curious_prog_noage | 128 | 0 | 1024 | 4 | 15 | 109 | 32 | 20.614526 |
| 17 | curious_surp_forget | 128 | 0 | 1024 | 4 | 15 | 109 | 32 | 19.876519 |
| 17 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 15 | 109 | 32 | 26.793974 |
| 18 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.185868 |
| 18 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 17.588453 |
| 18 | curious_prog | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 21.108543 |
| 18 | curious_surp | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 17.795905 |
| 18 | curious_prog_noage | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 21.147848 |
| 18 | curious_surp_forget | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 18.033849 |
| 18 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 17 | 107 | 32 | 26.526097 |
| 19 | random | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.597733 |
| 19 | round_robin | 0 | 0 | 1024 | 0 | 0 | 0 |  | 18.401255 |
| 19 | curious_prog | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 20.237131 |
| 19 | curious_surp | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 17.189971 |
| 19 | curious_prog_noage | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 20.453133 |
| 19 | curious_surp_forget | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 17.930376 |
| 19 | curious_prog_lifetime | 128 | 0 | 1024 | 4 | 14 | 110 | 32 | 27.459758 |

smoke

single-panel round_robin coverage matches results/w8 sweep for images 0-2 at L=4,8,16,32,64, exact float
image 0 L 4 coverage 0.726562
image 0 L 8 coverage 0.942383
image 0 L 16 coverage 0.999023
image 0 L 32 coverage 1.000000
image 0 L 64 coverage 1.000000
image 1 L 4 coverage 0.751953
image 1 L 8 coverage 0.941406
image 1 L 16 coverage 0.997070
image 1 L 32 coverage 1.000000
image 1 L 64 coverage 1.000000
image 2 L 4 coverage 0.706055
image 2 L 8 coverage 0.934570
image 2 L 16 coverage 1.000000
image 2 L 32 coverage 1.000000
image 2 L 64 coverage 1.000000

walls

life0_s 211.159
estimate_20_s 4223.179
wall_s 4373.436
pack_load_s 1.435
pack_prep_s 2.846
load_average_start 0.74
load_average_end 2.20
target_lives 20
first_pass_complete_curious_prog 20
mean_first_pass_t_curious_prog 32.000000
mean_n_choices_curious_prog 128.000000
mean_n_walk_curious_prog 0.000000
mean_n_first_curious_prog 4.000000
mean_n_eps_curious_prog 12.600000
mean_n_greedy_curious_prog 111.400000

source

dataset imagenette2-160
split val
pool_per_class 50
opacity 768
scene_shift 1
dot_shift 2
eye torchvision resnet18 IMAGENET1K_V1 frozen eval
softmax 10 logits, synset table order
logloss nats
unseen imagenet mean colour
stale cell keeps the rgb block from the generation that was seen
tv uniform random rgb block per clear cell, default_rng(9700+life) per policy
kinds_rng default_rng(9800+life).permutation
images_rng default_rng(9900+life).permutation, consumed in panel order
random_rng default_rng(8000+life), one uniform panel per visit
choice_rng default_rng(10000+life), prog_age_s
gamma 0.8
reservoir last 32 glances FIFO per panel for curious_prog and curious_prog_noage; none for curious_surp_forget; curious_prog_lifetime uses W8-1c ObservedReservoir cap 256
counts on each sight, the pixel's 8 grey counts are multiplied by gamma 0.8 and then the sight is added; KT alpha 0.5
mover W7 path_sweep, one cursor per panel, not reset when the image changes
visit 8 decisions, every policy
bandit 4 panels, W5/W6 prog_age_s
ema_beta 0.3
epsilon 0.1
tau 256
dwell 8
optimistic nearest unvisited on a 4-ring, start panel 0, values start at 0
reward curious_prog and curious_prog_noage sum equation 5 on the last 32 stored glances under the counts before the forgetting update, then learn and store the glance; curious_surp_forget sums mean KT surprise under the forgetting counts; curious_prog_lifetime is W8-1c curious_prog; curious_surp sums per-glance surprise; EMA once per visit
age at the choice, completed decisions minus last_end; last_end is the completed count when the visit ended; greedy step only; stored EMA is not aged
v_max largest visit reward so far
noage curious_prog_noage ranks the raw EMA on the greedy step
lifetime_arm curious_prog_lifetime is W8-1c curious_prog
forget_surp_arm curious_surp_forget is report only
bootstrap 10000 paired percentile, numpy quantile linear, default_rng(0) per contrast
runner vps
device cpu
torch_threads 2
torch 2.15.0+cpu torchvision 0.29.2+cpu
numpy 2.5.4
code_sha256 67ca974a8ef063381ce4491eab5283cdb1106a873a8bb3b0cd00fd625a1c1e59
n_lives_registered 20
n_lives_run 20
short_run no

demos

life0_curious_prog_t0128.png
life0_curious_prog_t0256.png
life0_curious_prog_t0384.png
life0_curious_prog_t0512.png
life0_curious_prog_t0640.png
life0_curious_prog_t0768.png
life0_curious_prog_t0896.png
life0_curious_prog_t1024.png
life0_round_robin_t0128.png
life0_round_robin_t0256.png
life0_round_robin_t0384.png
life0_round_robin_t0512.png
life0_round_robin_t0640.png
life0_round_robin_t0768.png
life0_round_robin_t0896.png
life0_round_robin_t1024.png

forgetting probes, life 0, curious_prog learner, print only

static stare 6 visits visit1 441.839352 visit6 190.844818 ratio 0.431933 below_10pct false

| visit | t0 | t1 | image | reward |
|---:|---:|---:|---:|---:|
| 1 | 0 | 7 | 215 | 441.839352 |
| 2 | 8 | 15 | 215 | 723.893270 |
| 3 | 16 | 23 | 215 | 607.697331 |
| 4 | 24 | 31 | 215 | 420.415863 |
| 5 | 32 | 39 | 215 | 307.417277 |
| 6 | 40 | 47 | 215 | 190.844818 |

fast stare, first two replacements

| visit | t0 | t1 | image | reward |
|---:|---:|---:|---:|---:|
| 1 | 0 | 7 | 77 | 431.505971 |
| 2 | 8 | 15 | 77 | 735.732995 |
| 3 | 16 | 23 | 77 | 607.896162 |
| 4 | 24 | 31 | 77 | 418.821029 |
| 5 | 32 | 39 | 171 | -3160.917080 |
| 6 | 40 | 47 | 171 | -1358.720200 |
| 7 | 48 | 55 | 171 | -389.914519 |
| 8 | 56 | 63 | 171 | 325.463073 |
| 9 | 64 | 71 | 59 | -2738.184764 |
| 10 | 72 | 79 | 59 | -1304.691020 |

| replacement_t | before_visit | before | after_visit | after | higher |
|---:|---:|---:|---:|---:|---|
| 32 | 4 | 418.821029 | 5 | -3160.917080 | false |
| 64 | 8 | 325.463073 | 9 | -2738.184764 | false |

life 0 random round_robin curious_surp match results/w8d exactly; curious_prog_lifetime matches results/w8d curious_prog exactly

life 0 first-pass visit rewards, curious_prog

| kind | reward |
|---|---:|
| static | 441.839352 |
| tv | -353.226795 |
| fast | 431.505971 |
| slow | 466.769588 |

tv_smallest true
