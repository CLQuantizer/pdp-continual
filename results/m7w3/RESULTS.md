# M7-W3 bank cortex

The W3 loop on the trace-learned roots bank. W is bits/symbol of the frozen bank on a fresh continuation of the 7 learnable channels, from a zero state, with that channel's embedding. W9 is that mean over all 9 room channels. Lower is better. Forgetting is end-of-life exam bits/symbol minus the exam bits/symbol just after the last visit to that channel. Positive forgetting is a worse code. Numbers only.

## W

| controller | seed 30 | seed 31 | seed 32 | seed 33 | seed 34 | seed 35 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 0.804654 | 0.704805 | 0.671865 | 0.649118 | 0.710645 | 1.268567 | 0.801609 ± 0.214395 |
| oracle7 | 0.868493 | 1.222725 | 1.124020 | 0.827258 | 0.905792 | 0.802698 | 0.958498 ± 0.157898 |
| prog_total | 0.894627 | 0.931975 | 1.278002 | 1.408650 | 0.814759 | 1.058003 | 1.064336 ± 0.213251 |
| prog_sleep | 0.495599 | 0.533178 | 1.145291 | 1.200333 | 0.666642 | 0.539888 | 0.763488 ± 0.294616 |
| fixed_sleep | 0.897702 | 0.530045 | 0.807006 | 0.745396 | 0.605543 | 0.811784 | 0.732913 ± 0.126755 |

## W9

| controller | seed 30 | seed 31 | seed 32 | seed 33 | seed 34 | seed 35 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 1.683813 | 1.592825 | 1.608719 | 1.645277 | 1.649447 | 1.928649 | 1.684788 ± 0.112933 |
| oracle7 | 2.464598 | 2.710732 | 2.645289 | 2.459652 | 2.481604 | 2.337826 | 2.516617 ± 0.124753 |
| prog_total | 2.222954 | 1.597342 | 1.804555 | 2.014114 | 1.717394 | 1.708770 | 1.844188 ± 0.211840 |
| prog_sleep | 1.506560 | 1.807790 | 2.444152 | 1.917340 | 1.613687 | 1.436921 | 1.787741 ± 0.336775 |
| fixed_sleep | 1.472984 | 1.608655 | 1.743396 | 1.894097 | 1.319173 | 1.438382 | 1.579448 ± 0.193956 |

## Gaps

On W.

| seed | prog_total − rand_dwell | prog_total − oracle7 | prog_sleep − prog_total | prog_sleep − fixed_sleep |
|---:|---:|---:|---:|---:|
| 30 | 0.089973 | 0.026134 | -0.399028 | -0.402103 |
| 31 | 0.227170 | -0.290750 | -0.398797 | 0.003133 |
| 32 | 0.606138 | 0.153982 | -0.132711 | 0.338285 |
| 33 | 0.759532 | 0.581392 | -0.208317 | 0.454936 |
| 34 | 0.104113 | -0.091033 | -0.148117 | 0.061099 |
| 35 | -0.210564 | 0.255305 | -0.518115 | -0.271895 |
| mean | 0.262727 | 0.105838 | -0.300848 | 0.030576 |
| std ddof=0 | 0.327876 | 0.274834 | 0.145257 | 0.304179 |
| left lower (count) | 1 of 6 | 2 of 6 | 6 of 6 | 2 of 6 |

## Forgetting

Exam-store bits/symbol just after the last visit that dwelled in the channel, minus nothing: the difference is end minus that snapshot. A channel with no exam glance at that visit is left out of the mean. n is the number of seed-channel pairs in the mean.

| controller | mean delta bps | std ddof=0 | n |
|---|---:|---:|---:|
| rand_dwell | 0.284698 | 0.468117 | 42 |
| oracle7 | 0.381058 | 0.547088 | 42 |
| prog_total | 0.611238 | 0.746440 | 42 |
| prog_sleep | 0.234550 | 0.637066 | 42 |
| fixed_sleep | 0.239063 | 0.372335 | 42 |

| controller | channel | n | after bps | end bps | delta bps | std ddof=0 |
|---|---|---:|---:|---:|---:|---:|
| rand_dwell | G | 6 | 1.203177 | 1.558168 | 0.354991 | 0.285186 |
| rand_dwell | C1 | 6 | 0.485803 | 0.579085 | 0.093282 | 0.128367 |
| rand_dwell | C7 | 6 | 0.596580 | 0.749896 | 0.153316 | 0.072963 |
| rand_dwell | E30 | 6 | 0.975382 | 1.459732 | 0.484349 | 0.579241 |
| rand_dwell | E110 | 6 | 0.998120 | 1.617843 | 0.619723 | 0.889933 |
| rand_dwell | M7 | 6 | 0.369157 | 0.390263 | 0.021105 | 0.028134 |
| rand_dwell | M37 | 6 | 0.474642 | 0.740761 | 0.266119 | 0.147445 |
| oracle7 | G | 6 | 1.116705 | 1.233950 | 0.117245 | 0.110801 |
| oracle7 | C1 | 6 | 0.459261 | 0.534113 | 0.074852 | 0.121575 |
| oracle7 | C7 | 6 | 0.658083 | 0.707274 | 0.049191 | 0.032864 |
| oracle7 | E30 | 6 | 0.901562 | 1.519165 | 0.617603 | 0.419986 |
| oracle7 | E110 | 6 | 1.084078 | 1.781342 | 0.697264 | 0.732862 |
| oracle7 | M7 | 6 | 0.403039 | 1.004860 | 0.601821 | 0.863999 |
| oracle7 | M37 | 6 | 0.496521 | 1.005948 | 0.509427 | 0.335235 |
| prog_total | G | 6 | 0.954674 | 2.300495 | 1.345821 | 1.118756 |
| prog_total | C1 | 6 | 0.657812 | 0.769648 | 0.111836 | 0.269700 |
| prog_total | C7 | 6 | 0.560550 | 0.746473 | 0.185923 | 0.154155 |
| prog_total | E30 | 6 | 1.069312 | 2.208078 | 1.138766 | 0.601496 |
| prog_total | E110 | 6 | 1.458801 | 1.704371 | 0.245570 | 0.364631 |
| prog_total | M7 | 6 | 0.335774 | 1.338111 | 1.002338 | 0.516686 |
| prog_total | M37 | 6 | 0.468609 | 0.717024 | 0.248415 | 0.351945 |
| prog_sleep | G | 6 | 1.120859 | 1.970747 | 0.849888 | 1.363262 |
| prog_sleep | C1 | 6 | 0.546933 | 0.647181 | 0.100247 | 0.144858 |
| prog_sleep | C7 | 6 | 0.551233 | 0.583826 | 0.032593 | 0.042017 |
| prog_sleep | E30 | 6 | 0.935698 | 1.374020 | 0.438322 | 0.559656 |
| prog_sleep | E110 | 6 | 1.049685 | 1.111920 | 0.062235 | 0.235868 |
| prog_sleep | M7 | 6 | 0.321109 | 0.360270 | 0.039160 | 0.079646 |
| prog_sleep | M37 | 6 | 0.476099 | 0.595505 | 0.119406 | 0.156322 |
| fixed_sleep | G | 6 | 1.037197 | 1.759115 | 0.721917 | 0.664067 |
| fixed_sleep | C1 | 6 | 0.421369 | 0.540621 | 0.119252 | 0.141804 |
| fixed_sleep | C7 | 6 | 0.572165 | 0.727770 | 0.155605 | 0.167051 |
| fixed_sleep | E30 | 6 | 0.796058 | 0.932680 | 0.136622 | 0.097307 |
| fixed_sleep | E110 | 6 | 0.993783 | 1.219193 | 0.225410 | 0.261336 |
| fixed_sleep | M7 | 6 | 0.385642 | 0.541777 | 0.156135 | 0.171618 |
| fixed_sleep | M37 | 6 | 0.375752 | 0.534254 | 0.158502 | 0.309352 |

## Sleep share

Sleep decisions divided by L.

| controller | seed 30 | seed 31 | seed 32 | seed 33 | seed 34 | seed 35 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| oracle7 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_total | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 ± 0.000000 |
| prog_sleep | 0.102051 | 0.406250 | 0.269531 | 0.105469 | 0.234375 | 0.148438 | 0.211019 ± 0.107250 |
| fixed_sleep | 0.102051 | 0.406250 | 0.269531 | 0.105469 | 0.234375 | 0.148438 | 0.211019 ± 0.107250 |

## prog_sleep sleep share by quarter

Quarter i is decisions [(i−1)L/4, iL/4). The share is sleep decisions in the quarter over L/4.

| seed | quarter 1 | quarter 2 | quarter 3 | quarter 4 |
|---:|---:|---:|---:|---:|
| 30 | 0.015625 | 0.078125 | 0.000000 | 0.314453 |
| 31 | 0.328125 | 0.312500 | 0.746094 | 0.238281 |
| 32 | 0.208984 | 0.582031 | 0.287109 | 0.000000 |
| 33 | 0.179688 | 0.242188 | 0.000000 | 0.000000 |
| 34 | 0.218750 | 0.390625 | 0.328125 | 0.000000 |
| 35 | 0.197266 | 0.177734 | 0.000000 | 0.218750 |
| mean | 0.191406 | 0.297201 | 0.226888 | 0.128581 |

## prog_sleep visit rewards by quarter

Exam-store bits saved per decision, mean over the visits whose first decision falls in the quarter. A quarter with no visits of that kind is blank. The mean row is the mean of the per-seed means, over seeds that have at least one such visit.

| seed | quarter | sleep visits | sleep mean reward | room visits | room mean reward |
|---:|---:|---:|---:|---:|---:|
| 30 | 1 | 1 | -0.966035 | 52 | 22.550484 |
| 30 | 2 | 5 | 41.022768 | 55 | 14.619802 |
| 30 | 3 | 0 | — | 58 | 8.199001 |
| 30 | 4 | 21 | 35.523319 | 37 | -2.055523 |
| 31 | 1 | 21 | 77.061313 | 37 | 24.380988 |
| 31 | 2 | 20 | 96.573064 | 37 | -32.822647 |
| 31 | 3 | 48 | 5.617026 | 14 | -55.908274 |
| 31 | 4 | 15 | 5.466888 | 46 | 3.361956 |
| 32 | 1 | 14 | 98.301657 | 41 | 22.965616 |
| 32 | 2 | 37 | 30.259774 | 21 | -44.326487 |
| 32 | 3 | 18 | -7.173733 | 43 | -13.828111 |
| 32 | 4 | 0 | — | 62 | 2.366103 |
| 33 | 1 | 12 | 84.000404 | 44 | 20.297597 |
| 33 | 2 | 15 | 5.995665 | 44 | 4.416078 |
| 33 | 3 | 0 | — | 58 | 7.656403 |
| 33 | 4 | 0 | — | 62 | -6.628185 |
| 34 | 1 | 14 | 87.251098 | 44 | 36.244385 |
| 34 | 2 | 25 | 23.355069 | 34 | -18.751672 |
| 34 | 3 | 21 | 18.007656 | 38 | -1.458850 |
| 34 | 4 | 0 | — | 59 | -0.071373 |
| 35 | 1 | 13 | 86.816411 | 44 | 43.992916 |
| 35 | 2 | 11 | 1.659695 | 45 | 3.623778 |
| 35 | 3 | 0 | — | 57 | -4.589678 |
| 35 | 4 | 14 | 72.307816 | 47 | -9.606453 |
| mean | 1 | 75 | 72.077475 | 262 | 28.405331 |
| mean | 2 | 113 | 33.144339 | 236 | -12.206858 |
| mean | 3 | 87 | 5.483649 | 268 | -9.988251 |
| mean | 4 | 50 | 37.766008 | 313 | -2.105579 |

## Decision shares

Mean over seeds. A stay counts for the room channel. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.

| controller | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 0.0820 | 0.0951 | 0.0775 | 0.0807 | 0.0938 | 0.0854 | 0.0921 | 0.0827 | 0.0872 | 0.2235 | 0.0000 |
| oracle7 | 0.0970 | 0.1090 | 0.1074 | 0.1243 | 0.1126 | 0.1230 | 0.1133 | 0.0000 | 0.0000 | 0.2132 | 0.0000 |
| prog_total | 0.1204 | 0.0723 | 0.1267 | 0.0840 | 0.0983 | 0.1074 | 0.1417 | 0.1204 | 0.0208 | 0.1079 | 0.0000 |
| prog_sleep | 0.0963 | 0.0320 | 0.1042 | 0.0742 | 0.0365 | 0.1471 | 0.1019 | 0.0924 | 0.0169 | 0.0876 | 0.2110 |
| fixed_sleep | 0.0876 | 0.0352 | 0.1120 | 0.0991 | 0.0527 | 0.0306 | 0.1385 | 0.1270 | 0.0208 | 0.0854 | 0.2110 |

## Mean held-out bits

Mean over seeds. W is the mean of the 7 learnable channels. W9 is the mean of all 9.

| controller | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z | W | W9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell | 1.842158 | 0.147609 | 0.234593 | 1.024339 | 1.417863 | 0.278650 | 0.666052 | 9.550675 | 0.001156 | 0.801609 | 1.684788 |
| oracle7 | 1.668240 | 0.144706 | 0.180501 | 1.032789 | 1.564012 | 1.086845 | 1.032391 | 15.903721 | 0.036346 | 0.958498 | 2.516617 |
| prog_total | 2.508290 | 0.288795 | 0.204750 | 1.952854 | 0.922942 | 1.088824 | 0.483897 | 9.138509 | 0.008834 | 1.064336 | 1.844188 |
| prog_sleep | 2.276102 | 0.274080 | 0.197333 | 1.063128 | 0.800535 | 0.299954 | 0.433286 | 10.738275 | 0.006978 | 0.763488 | 1.787741 |
| fixed_sleep | 2.014206 | 0.254035 | 0.255749 | 0.509368 | 1.139080 | 0.457117 | 0.500833 | 8.999183 | 0.085459 | 0.732913 | 1.579448 |

## Per-seed held-out bits

### rand_dwell

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 2.040443 | 0.097934 | 0.134067 | 1.105250 | 0.998015 | 0.284066 | 0.972802 | 9.519911 | 0.001831 |
| 31 | 2.597037 | 0.127156 | 0.278283 | 0.488615 | 0.909750 | 0.248637 | 0.284155 | 9.401522 | 0.000266 |
| 32 | 1.570584 | 0.159579 | 0.303304 | 0.576311 | 1.320885 | 0.263670 | 0.508720 | 9.774368 | 0.001048 |
| 33 | 1.859600 | 0.072954 | 0.129196 | 0.710292 | 0.770723 | 0.234598 | 0.766464 | 10.262476 | 0.001187 |
| 34 | 1.484194 | 0.050629 | 0.150945 | 0.937180 | 1.237145 | 0.363191 | 0.751235 | 9.869551 | 0.000951 |
| 35 | 1.501089 | 0.377400 | 0.411764 | 2.328383 | 3.270661 | 0.277736 | 0.712934 | 8.476221 | 0.001654 |

### oracle7

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 2.227175 | 0.123006 | 0.133242 | 1.142797 | 0.874740 | 0.340711 | 1.237778 | 16.045258 | 0.056675 |
| 31 | 1.616539 | 0.057428 | 0.122498 | 0.757985 | 0.762820 | 3.440177 | 1.801632 | 15.824987 | 0.012521 |
| 32 | 1.519940 | 0.085019 | 0.237482 | 1.255645 | 3.483317 | 0.440734 | 0.846006 | 15.899030 | 0.040431 |
| 33 | 1.444990 | 0.084659 | 0.203259 | 1.463830 | 1.360207 | 0.336556 | 0.897304 | 16.267324 | 0.078735 |
| 34 | 1.585501 | 0.494424 | 0.318508 | 1.050649 | 1.462605 | 0.408281 | 1.020575 | 15.983837 | 0.010057 |
| 35 | 1.615295 | 0.023701 | 0.068015 | 0.525831 | 1.440383 | 1.554609 | 0.391050 | 15.401890 | 0.019658 |

### prog_total

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1.663691 | 0.316825 | 0.327241 | 0.637587 | 2.526624 | 0.389072 | 0.401349 | 13.740534 | 0.003667 |
| 31 | 1.432127 | 0.683627 | 0.254421 | 1.481780 | 0.455582 | 1.134639 | 1.081651 | 7.817349 | 0.034908 |
| 32 | 4.014011 | 0.082277 | 0.062688 | 2.943136 | 0.604703 | 0.777978 | 0.461224 | 7.290409 | 0.004571 |
| 33 | 4.124639 | 0.117955 | 0.265629 | 3.330271 | 0.608298 | 1.041186 | 0.372571 | 8.263211 | 0.003268 |
| 34 | 1.300651 | 0.481682 | 0.054286 | 1.238224 | 0.760554 | 1.556516 | 0.311397 | 9.750723 | 0.002510 |
| 35 | 2.514621 | 0.050405 | 0.264238 | 2.086123 | 0.581892 | 1.633553 | 0.275189 | 7.968829 | 0.004079 |

### prog_sleep

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 1.244113 | 0.202399 | 0.113314 | 0.741233 | 0.481376 | 0.343332 | 0.343427 | 10.087468 | 0.002373 |
| 31 | 1.242092 | 0.159242 | 0.535649 | 0.685405 | 0.503282 | 0.220601 | 0.385974 | 12.535854 | 0.002009 |
| 32 | 3.888284 | 0.449345 | 0.148401 | 2.197508 | 0.596098 | 0.395913 | 0.341489 | 13.976448 | 0.003881 |
| 33 | 4.736970 | 0.426299 | 0.167944 | 0.742781 | 1.480989 | 0.275249 | 0.572096 | 8.826892 | 0.026840 |
| 34 | 1.048588 | 0.272600 | 0.044121 | 1.283518 | 1.122535 | 0.269434 | 0.625694 | 9.852636 | 0.004054 |
| 35 | 1.496568 | 0.134595 | 0.174571 | 0.728322 | 0.618932 | 0.295196 | 0.331035 | 9.150353 | 0.002713 |

### fixed_sleep

| seed | G | C1 | C7 | E30 | E110 | M7 | M37 | N | Z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 2.995356 | 0.362481 | 0.514872 | 0.806182 | 0.944048 | 0.299012 | 0.361963 | 6.970300 | 0.002642 |
| 31 | 1.189091 | 0.258385 | 0.183834 | 0.778911 | 0.550213 | 0.350273 | 0.399605 | 10.764019 | 0.003566 |
| 32 | 1.707391 | 0.419504 | 0.371215 | 0.234808 | 1.236628 | 0.325980 | 1.353519 | 10.038765 | 0.002752 |
| 33 | 2.292018 | 0.144112 | 0.064138 | 0.695186 | 0.814491 | 0.917738 | 0.290092 | 11.348187 | 0.480910 |
| 34 | 1.636277 | 0.108199 | 0.159824 | 0.188493 | 1.442067 | 0.392231 | 0.311709 | 7.631871 | 0.001883 |
| 35 | 2.265104 | 0.231531 | 0.240613 | 0.352630 | 1.847032 | 0.457469 | 0.288107 | 7.241955 | 0.021000 |

## Hashed learner reference

M5b and M6b at L = 2048, the hashed table. Different seeds from this run.

### M5b L = 2048

| controller | seed 12 | seed 13 | seed 14 | seed 15 | seed 16 | seed 17 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell W | 1.979503 | 1.939061 | 1.992578 | 1.944813 | 2.100946 | 1.961694 | 1.986432 ± 0.054448 |
| rand_dwell W9 | 2.135669 | 2.104623 | 2.145593 | 2.108565 | 2.230599 | 2.121881 | 2.141155 ± 0.042462 |
| oracle7 W | 1.824375 | 1.925945 | 1.878755 | 1.914727 | 1.959996 | 1.831212 | 1.889168 ± 0.049477 |
| oracle7 W9 | 2.614618 | 2.688707 | 2.652051 | 2.680026 | 2.717141 | 2.615158 | 2.661284 ± 0.037890 |
| prog W | 1.737457 | 1.753690 | 1.734712 | 1.851563 | 1.761203 | 1.726191 | 1.760803 ± 0.042253 |
| prog W9 | 1.947638 | 1.960192 | 1.945573 | 2.036360 | 1.966125 | 1.939081 | 1.965828 ± 0.032821 |

### M6b L = 2048

| controller | seed 24 | seed 25 | seed 26 | seed 27 | seed 28 | seed 29 | mean ± std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prog W | 1.777857 | 1.912109 | 1.740865 | 1.748879 | 1.728790 | 1.776491 | 1.780832 ± 0.061357 |
| prog W9 | 1.979149 | 2.083224 | 1.950666 | 1.956580 | 1.940894 | 1.978105 | 1.981437 ± 0.047584 |
| prog_total W | 1.795986 | 1.741791 | 1.692497 | 1.685354 | 1.657242 | 1.707991 | 1.713477 ± 0.044783 |
| prog_total W9 | 1.993968 | 1.951238 | 1.913721 | 1.907883 | 1.885898 | 1.925526 | 1.929705 ± 0.034789 |
| prog_sleep W | 1.768572 | 1.751245 | 1.711638 | 1.699932 | 1.693468 | 1.725618 | 1.725079 ± 0.027046 |
| prog_sleep W9 | 1.972269 | 1.958721 | 1.928338 | 1.919030 | 1.913932 | 1.939067 | 1.938560 ± 0.020958 |
| fixed_sleep W | 1.691541 | 1.728918 | 1.705069 | 1.718443 | 1.697241 | 1.739289 | 1.713417 ± 0.017053 |
| fixed_sleep W9 | 1.912410 | 1.941174 | 1.923379 | 1.933301 | 1.916845 | 1.949727 | 1.929472 ± 0.013226 |

## Ring order

Channel in rooms 0..8. One order per seed; controllers share it.

| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---:|---|---|---|---|---|---|---|---|---|
| 30 | M37 | M7 | N | C1 | Z | C7 | G | E30 | E110 |
| 31 | C1 | M37 | C7 | G | M7 | Z | E30 | N | E110 |
| 32 | N | M7 | C1 | E110 | G | M37 | Z | C7 | E30 |
| 33 | N | M37 | M7 | Z | C1 | G | E110 | E30 | C7 |
| 34 | C7 | C1 | M37 | E110 | Z | E30 | G | M7 | N |
| 35 | M7 | E110 | N | G | Z | C7 | C1 | E30 | M37 |

## Walls

| kernel | seed | wall seconds |
|---|---:|---:|
| ezzzzzx/m7-w3-s30 | 30 | 848.359 |
| ezzzzzx/m7-w3-s31 | 31 | 624.221 |
| ezzzzzx/m7-w3-s32 | 32 | 625.325 |
| ezzzzzx/m7-w3-s33 | 33 | 687.344 |
| ezzzzzx/m7-w3-s34 | 34 | 419.991 |
| ezzzzzx/m7-w3-s35 | 35 | 822.623 |

| controller | seed | wall seconds | learn seconds | score seconds |
|---|---:|---:|---:|---:|
| rand_dwell | 30 | 191.180 | 158.764 | 19.536 |
| rand_dwell | 31 | 147.017 | 121.952 | 15.076 |
| rand_dwell | 32 | 147.486 | 122.747 | 14.771 |
| rand_dwell | 33 | 151.741 | 126.344 | 15.042 |
| rand_dwell | 34 | 82.191 | 64.353 | 10.814 |
| rand_dwell | 35 | 190.358 | 156.818 | 20.510 |
| oracle7 | 30 | 192.801 | 162.247 | 17.821 |
| oracle7 | 31 | 149.658 | 125.847 | 14.167 |
| oracle7 | 32 | 148.166 | 124.202 | 13.914 |
| oracle7 | 33 | 152.987 | 128.337 | 14.196 |
| oracle7 | 34 | 79.778 | 62.165 | 10.216 |
| oracle7 | 35 | 193.446 | 161.883 | 18.853 |
| prog_total | 30 | 469.750 | 158.309 | 298.841 |
| prog_total | 31 | 364.918 | 110.639 | 244.548 |
| prog_total | 32 | 354.294 | 93.242 | 251.420 |
| prog_total | 33 | 370.178 | 110.250 | 249.847 |
| prog_total | 34 | 231.693 | 39.102 | 185.504 |
| prog_total | 35 | 471.418 | 125.066 | 333.488 |
| prog_sleep | 30 | 453.957 | 128.740 | 312.600 |
| prog_sleep | 31 | 330.389 | 94.066 | 226.667 |
| prog_sleep | 32 | 347.306 | 100.643 | 237.029 |
| prog_sleep | 33 | 360.445 | 109.079 | 241.223 |
| prog_sleep | 34 | 219.008 | 45.447 | 166.503 |
| prog_sleep | 35 | 457.603 | 131.211 | 313.778 |
| fixed_sleep | 30 | 375.365 | 114.309 | 259.525 |
| fixed_sleep | 31 | 256.570 | 66.609 | 188.811 |
| fixed_sleep | 32 | 268.323 | 78.285 | 188.932 |
| fixed_sleep | 33 | 314.470 | 92.756 | 220.528 |
| fixed_sleep | 34 | 186.281 | 30.093 | 155.292 |
| fixed_sleep | 35 | 347.578 | 95.822 | 250.322 |

## Smoke test

```
torch=2.15.0+cpu threads=1 model_seed=0
smoke check 1: frozen zero channel vs h30y roots_learn, H20 slow seed 6, n=4096
  ref_bits 1.19039465 bank_bits 1.19039465 max_abs_delta 0.000e+00 limit 1.0e-06 ok True ref_s 3.17 bank_s 3.17
smoke check 1b: batched zero-state score vs trace loss score 873.503480 trace 873.503471 abs 8.771e-06 ok True
smoke check 2: prog_sleep seed 30 L=128, disjoint stores, sleep draws no exam glance
  G exam 6 (offered 6) replay 4 (offered 4)
  C1 exam 12 (offered 12) replay 4 (offered 4)
  C7 exam 6 (offered 6) replay 2 (offered 2)
  E30 exam 4 (offered 4) replay 4 (offered 4)
  E110 exam 7 (offered 7) replay 9 (offered 9)
  M7 exam 6 (offered 6) replay 2 (offered 2)
  M37 exam 5 (offered 5) replay 3 (offered 3)
  N exam 4 (offered 4) replay 4 (offered 4)
  Z exam 7 (offered 7) replay 9 (offered 9)
  overlap 0 exam_glances_drawn 0 sleep_decisions 8 sleep_state_intact True sleep_param_delta 9.737e-03 cache_delta 2.254e-05 wall 16.15s
smoke check 3: first sleep visit kept live state and traces bit-identical; parameter max abs delta 9.737e-03
smoke check 4: five controllers, seed 30, L=64, W and W9 finite
  rand_dwell W 9.922327 W9 9.474917 finite True sleep 0.0000 corridor 0.2031 wall 5.70s learn 3.27s score 1.12s
  oracle7 W 5.542174 W9 6.234600 finite True sleep 0.0000 corridor 0.2500 wall 5.31s learn 2.92s score 1.05s
  prog_total W 8.209481 W9 8.133197 finite True sleep 0.0000 corridor 0.1094 wall 8.28s learn 2.88s score 4.01s
  prog_sleep W 8.209481 W9 8.133197 finite True sleep 0.0000 corridor 0.1094 wall 7.91s learn 2.74s score 3.80s
  fixed_sleep W 8.209481 W9 8.133197 finite True sleep 0.0000 target 0 wall 8.09s
smoke timing: prog_sleep seed 30 L=256
  L256 wall 36.17s learn 15.53s score 20.42s learn_per_decision 0.0607s full_rescore_288 0.2514s
  pessimistic one slow controller 639.1s (10.65 min), two-wave kernel 1278.1s (21.30 min), limit 5400s
  estimate charges every decision a full exam store and runs fixed_sleep after prog_sleep; the other three controllers share prog_sleep's wave
```

## Protocol

- `BankCortex` is `h30y_roots.build_arm('roots_learn')` with model seed 0: N = 256, θ_k = 2πk/256, |λ| = 0.98, embedding 32, order-2 readout, Adam lr 1e-3, clip 1.0, float32. The trace step is the H30x rule. A 10-row channel embedding, initialised at zero, is added to the symbol embedding at every step. Rows follow `curio.CH` and then the corridor.
- One life is one stream. State and traces carry across glances. Each glance is 64 symbols: score it, then one Adam step. Sleep and stored glances run from a zero state, including zero traces, and use the stored previous symbol as the first input. A missing previous symbol, only at the start of life, uses a zero symbol embedding. Eval continuations also start from a zero symbol embedding.
- After a room glance is learned, one coin flip, `default_rng(40_000 + seed).integers(2) == 0`, sends the whole glance to the exam reservoir, and otherwise to the replay reservoir. Each holds 32 glances per channel. Exam replacement uses `default_rng(20_000 + seed)`. Replay replacement uses `default_rng(50_000 + seed)`. The replacement rng is drawn only when that channel's reservoir is full. Corridor glances are learned and stored nowhere.
- A decision's progress is the sum of exam-glance bits before the Adam step minus the sum after it, on the glances already stored. The new glance is inserted after that score. The next decision reuses this after-score, plus the inserted glance if it went to the exam store. A visit reward is the sum of its decision progresses divided by its decisions. Travel counts.
- A sleep decision draws one glance uniformly from the union of the replay stores, `default_rng(30_000 + seed)`, and takes one trace Adam step from a zero state. An empty union learns nothing. The live state and traces are not written. A sleep visit is 8 such decisions and does not move. N's eval stream is a fresh `default_rng(30_000 + seed)`, not the sleep rng.
- `rand_dwell` and `oracle7` use `w3_headroom.next_action` and `default_rng(1_000 + seed)` for both the plan and N-channel glances. `prog_total` is the 9-room bandit. `prog_sleep` adds sleep as the last arm of the optimistic first pass, then eps-greedy with EPS=0.1 and BETA=0.3 over 10 arms. Choice rng is `default_rng(10_000 + seed)`. `fixed_sleep` is `prog_total` plus planned sleep visits whose lengths sum to that seed's `prog_sleep` sleep count.
- Forgetting snapshots the exam glances of a channel at the end of the last visit that dwelled there, and scores those same glances again at the end of life. W and W9 use EVAL=2048 symbols from a copy of the H20 world, learning off.
- World seeds 30, 31, 32, 33, 34, 35. L = 2048. CPU. Private Kaggle kernels, one seed each, pool of up to 4, no GPU, no internet. Nothing retuned.

## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (prog_total − rand_dwell ≤ −0.10, 5 of 6): FAIL.** The gap is +0.263, and prog_total is lower on 1 of 6 seeds. The kill
rule fires: progress on the whole store, read from a shared-weight cortex, is too entangled for the bandit.
**Bar 2 (prog_sleep − prog_total ≤ −0.05, 5 of 6): PASS.** The gap is −0.301, and prog_sleep is lower on 6 of 6.

**The repeater theory, confirmed by contrast.** On the hashed learner, which hardly forgets, sleep bought nothing
(M6b, +0.012). On the bank, whose weights are shared, end-of-life forgetting without sleep is 0.61 bits/symbol in
the exam stores. Sleep cuts it to 0.23 and gains 0.30 on fresh data. Replay pays exactly where carriage is lossy. A
fixed schedule at the same dose does as well (fixed_sleep 0.733 vs prog_sleep 0.763; prog_sleep lower on only 2 of 6). So the
dose matters and the timing, so far, does not.

**Why curiosity broke.** The bank learns far better than the hashed model: rand_dwell scores W = 0.80 here against 1.99 in M5b. But
on a shared cortex, a room visit's whole-store progress is its gain in that room minus what it wears down elsewhere.
After the first quarter, room visits average *negative* progress (−12, −10, −2 bits per decision). The bandit then
picks the rooms that hurt least, not the ones that teach most. It narrows, and forgets the rest: prog_total has the
highest forgetting (0.61). The learnability oracle also loses to random dwelling (0.96 vs 0.80). The two noise rooms
it skips cost nothing to visit, while the extra time it spends on learnable rooms speeds up interference among them.
On this cortex, how well the agent retains what it learned matters more than where it goes.

**Next (M7b, fresh seeds).** Sleep at a fixed dose for every controller, since the dose is what pays. The curiosity
reward becomes per-channel progress (M5's: the gain in the room visited), with consolidation left to sleep. The
comparison is against random dwelling with the same sleep.
