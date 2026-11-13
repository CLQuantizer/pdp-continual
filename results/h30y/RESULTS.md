# H30y measurements

H20 slow n = 262144. Echo n = 131072, D in [32, 128, 512]. Chunk = 64. Adam lr = 0.001. Clip = 1.0. Embedding = 32. Units = 256. Order-2 maps 512 to 64. Population std, ddof = 0. Embedding frozen = false.

Protocol: the trace rule is h30x_trace.train_trace, unchanged. The bank is h30w_ring.Bank, N = 256, h_t = λ ⊙ h_{t−1} + B x_t, λ = exp(−exp(ν) + iθ). Order-2 readout with no bias: W, P1, P2, V. Input is the previous symbol. The first input embedding is zero. Float32. Chunks of 64, score before the Adam step, lr 1e-3, clip 1.0. trace_rand uses the H30x init: |λ| uniform on [0.9, 0.999], θ uniform on [0, π]. roots_learn and roots_frozen use θ_k = 2πk/256 and |λ_k| = 0.98, ν_k = log(−log 0.98). roots_learn steps ν and θ. roots_frozen does not step ν or θ; B, the readout and the embedding do. H20 world seeds 6, 7, 8 use model seeds 0, 1, 2. Echo uses the stream seed as the model seed. Echo uniform rate is 4 bits on both halves. Last quarter is symbols from index 3n/4 to the end.

## Parameter counts

| Alphabet | Parameters |
|---:|---:|
| 16 | 92160 |
| 41 | 107360 |

## H20 slow, bits/symbol

| Arm | mean ± std | seed 6 | seed 7 | seed 8 |
|---|---:|---:|---:|---:|
| trace_rand | 0.631740 ± 0.027936 | 0.661936 | 0.594579 | 0.638705 |
| roots_learn | 0.612351 ± 0.006774 | 0.621809 | 0.608934 | 0.606309 |
| roots_frozen | 0.659464 ± 0.004123 | 0.664716 | 0.654644 | 0.659033 |

## H20 per process, bits/symbol, mean ± std over seeds

| Arm | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trace_rand | 1.490360 ± 0.137844 | 0.256014 ± 0.082286 | 0.280484 ± 0.067779 | 0.706410 ± 0.233857 | 0.554923 ± 0.410775 | 0.366202 ± 0.030538 | 0.423524 ± 0.022624 |
| roots_learn | 1.472460 ± 0.088931 | 0.241871 ± 0.053825 | 0.289043 ± 0.120940 | 0.587694 ± 0.120482 | 0.498744 ± 0.298119 | 0.361283 ± 0.031628 | 0.448399 ± 0.039696 |
| roots_frozen | 1.564071 ± 0.119193 | 0.308435 ± 0.072613 | 0.323239 ± 0.123544 | 0.623435 ± 0.185059 | 0.561903 ± 0.410281 | 0.391672 ± 0.023104 | 0.475768 ± 0.043921 |

## H20 per seed per process, bits/symbol

| Arm | Seed | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_rand | 6 | 1.677373 | 0.162496 | 0.232563 | 1.036617 | 0.284940 | NA | 0.437466 |
| trace_rand | 7 | 1.444512 | 0.362750 | 0.376338 | 0.557307 | 0.244453 | 0.335664 | 0.441493 |
| trace_rand | 8 | 1.349196 | 0.242795 | 0.232552 | 0.525305 | 1.135377 | 0.396740 | 0.391614 |
| roots_learn | 6 | 1.518120 | 0.200394 | 0.211576 | 0.753359 | 0.317582 | NA | 0.472818 |
| roots_learn | 7 | 1.551118 | 0.317886 | 0.459832 | 0.470363 | 0.259631 | 0.392911 | 0.479966 |
| roots_learn | 8 | 1.348143 | 0.207334 | 0.195720 | 0.539360 | 0.919019 | 0.329655 | 0.392412 |
| roots_frozen | 6 | 1.623111 | 0.235751 | 0.249701 | 0.879639 | 0.270163 | NA | 0.505237 |
| roots_frozen | 7 | 1.671284 | 0.407600 | 0.497263 | 0.449071 | 0.273421 | 0.414775 | 0.508387 |
| roots_frozen | 8 | 1.397817 | 0.281952 | 0.222754 | 0.541595 | 1.142124 | 0.368568 | 0.413681 |

## H20 process counts

| Seed | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 65536 | 65536 | 32768 | 16384 | 49152 | 0 | 32768 |
| 7 | 49152 | 49152 | 16384 | 49152 | 32768 | 32768 | 32768 |
| 8 | 65536 | 49152 | 32768 | 16384 | 16384 | 32768 | 49152 |

## H20 C1+C7 combined, bits/symbol

| Arm | mean ± std | seed 6 | seed 7 | seed 8 | count 6 | count 7 | count 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| trace_rand | 0.263566 ± 0.075677 | 0.185852 | 0.366147 | 0.238698 | 98304 | 65536 | 81920 |
| roots_learn | 0.253394 ± 0.070698 | 0.204122 | 0.353372 | 0.202688 | 98304 | 65536 | 81920 |
| roots_frozen | 0.309563 ± 0.085485 | 0.240401 | 0.430016 | 0.258272 | 98304 | 65536 | 81920 |

## Gaps on H20 C1+C7 combined, roots − trace_rand

| Seed | roots_learn | roots_frozen | trace_rand | roots_learn − trace_rand | roots_frozen − trace_rand |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.204122 | 0.240401 | 0.185852 | 0.018270 | 0.054549 |
| 7 | 0.353372 | 0.430016 | 0.366147 | -0.012775 | 0.063869 |
| 8 | 0.202688 | 0.258272 | 0.238698 | -0.036009 | 0.019575 |
| mean ± std |  |  |  | -0.010171 ± 0.022236 | 0.045998 ± 0.019067 |

## Echo D = 32, bits/symbol

| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|---:|
| trace_rand | fresh | 5.092031 ± 0.087205 | 5.212680 | 5.053839 | 5.009575 |
| trace_rand | copy | 1.589833 ± 0.144076 | 1.783928 | 1.546469 | 1.439101 |
| roots_learn | fresh | 5.108781 ± 0.043116 | 5.139345 | 5.139193 | 5.047806 |
| roots_learn | copy | 0.969558 ± 0.026299 | 0.982391 | 0.993372 | 0.932910 |
| roots_frozen | fresh | 5.394091 ± 0.048273 | 5.432416 | 5.423856 | 5.326002 |
| roots_frozen | copy | 1.777251 ± 0.035015 | 1.806254 | 1.797508 | 1.727990 |

## Echo D = 32, last quarter, bits/symbol

| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 | count |
|---|---|---:|---:|---:|---:|---:|
| trace_rand | fresh | 4.770873 ± 0.017060 | 4.790369 | 4.748817 | 4.773433 | 16384 |
| trace_rand | copy | 0.533419 ± 0.030662 | 0.576148 | 0.518452 | 0.505656 | 16384 |
| roots_learn | fresh | 4.606163 ± 0.011264 | 4.617179 | 4.610619 | 4.590689 | 16384 |
| roots_learn | copy | 0.189162 ± 0.001231 | 0.188382 | 0.190900 | 0.188204 | 16384 |
| roots_frozen | fresh | 4.956247 ± 0.009580 | 4.969406 | 4.946874 | 4.952461 | 16384 |
| roots_frozen | copy | 1.113719 ± 0.006444 | 1.122767 | 1.110143 | 1.108247 | 16384 |

## Gaps on echo D = 32 copy halves, roots − trace_rand

| Seed | roots_learn | roots_frozen | trace_rand | roots_learn − trace_rand | roots_frozen − trace_rand |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.982391 | 1.806254 | 1.783928 | -0.801537 | 0.022326 |
| 1 | 0.993372 | 1.797508 | 1.546469 | -0.553098 | 0.251039 |
| 2 | 0.932910 | 1.727990 | 1.439101 | -0.506191 | 0.288889 |
| mean ± std |  |  |  | -0.620275 ± 0.129594 | 0.187418 ± 0.117756 |

## Echo D = 128, bits/symbol

| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|---:|
| trace_rand | fresh | 4.672466 ± 0.055485 | 4.746026 | 4.659344 | 4.612029 |
| trace_rand | copy | 4.274956 ± 0.016089 | 4.279956 | 4.291679 | 4.253232 |
| roots_learn | fresh | 4.750640 ± 0.046319 | 4.783403 | 4.783380 | 4.685135 |
| roots_learn | copy | 4.405203 ± 0.027960 | 4.424343 | 4.425597 | 4.365668 |
| roots_frozen | fresh | 4.807086 ± 0.049797 | 4.840470 | 4.844093 | 4.736693 |
| roots_frozen | copy | 4.405125 ± 0.034501 | 4.428679 | 4.430354 | 4.356342 |

## Echo D = 128, last quarter, bits/symbol

| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 | count |
|---|---|---:|---:|---:|---:|---:|
| trace_rand | fresh | 4.171303 ± 0.018012 | 4.196510 | 4.155522 | 4.161877 | 16384 |
| trace_rand | copy | 3.907076 ± 0.026213 | 3.870013 | 3.924928 | 3.926286 | 16384 |
| roots_learn | fresh | 4.167154 ± 0.001025 | 4.167018 | 4.168471 | 4.165972 | 16384 |
| roots_learn | copy | 4.036256 ± 0.003334 | 4.040690 | 4.032652 | 4.035426 | 16384 |
| roots_frozen | fresh | 4.207713 ± 0.002485 | 4.207017 | 4.211045 | 4.205077 | 16384 |
| roots_frozen | copy | 4.002110 ± 0.003506 | 4.006964 | 4.000556 | 3.998808 | 16384 |

## Gaps on echo D = 128 copy halves, roots − trace_rand

| Seed | roots_learn | roots_frozen | trace_rand | roots_learn − trace_rand | roots_frozen − trace_rand |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.424343 | 4.428679 | 4.279956 | 0.144387 | 0.148723 |
| 1 | 4.425597 | 4.430354 | 4.291679 | 0.133918 | 0.138674 |
| 2 | 4.365668 | 4.356342 | 4.253232 | 0.112437 | 0.103111 |
| mean ± std |  |  |  | 0.130247 ± 0.013299 | 0.130169 ± 0.019568 |

## Echo D = 512, bits/symbol

| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|---:|
| trace_rand | fresh | 4.913085 ± 0.097748 | 5.036582 | 4.905127 | 4.797547 |
| trace_rand | copy | 3.783280 ± 0.035196 | 3.825286 | 3.785402 | 3.739153 |
| roots_learn | fresh | 5.113481 ± 0.060319 | 5.152956 | 5.159234 | 5.028254 |
| roots_learn | copy | 3.626910 ± 0.004289 | 3.631343 | 3.628279 | 3.621108 |
| roots_frozen | fresh | 5.144076 ± 0.061903 | 5.184865 | 5.190765 | 5.056599 |
| roots_frozen | copy | 3.657269 ± 0.009736 | 3.661103 | 3.666804 | 3.643898 |

## Echo D = 512, last quarter, bits/symbol

| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 | count |
|---|---|---:|---:|---:|---:|---:|
| trace_rand | fresh | 4.290373 ± 0.015394 | 4.311504 | 4.284345 | 4.275270 | 16384 |
| trace_rand | copy | 3.738984 ± 0.006442 | 3.747973 | 3.733200 | 3.735780 | 16384 |
| roots_learn | fresh | 4.366017 ± 0.007021 | 4.371830 | 4.370081 | 4.356139 | 16384 |
| roots_learn | copy | 3.636968 ± 0.012068 | 3.645564 | 3.619901 | 3.645438 | 16384 |
| roots_frozen | fresh | 4.402126 ± 0.008165 | 4.398046 | 4.413522 | 4.394811 | 16384 |
| roots_frozen | copy | 3.619866 ± 0.010949 | 3.622117 | 3.605474 | 3.632008 | 16384 |

## Gaps on echo D = 512 copy halves, roots − trace_rand

| Seed | roots_learn | roots_frozen | trace_rand | roots_learn − trace_rand | roots_frozen − trace_rand |
|---:|---:|---:|---:|---:|---:|
| 0 | 3.631343 | 3.661103 | 3.825286 | -0.193943 | -0.164183 |
| 1 | 3.628279 | 3.666804 | 3.785402 | -0.157124 | -0.118598 |
| 2 | 3.621108 | 3.643898 | 3.739153 | -0.118045 | -0.095255 |
| mean ± std |  |  |  | -0.156370 ± 0.030990 | -0.126012 ± 0.028624 |

## roots_learn |λ|

Ten bins on [0, 1]. Edges 0.0, 0.1, ..., 1.0. Counts are |λ| at the end of the run. n(|λ| > 0.99) is a separate count, not an extra bin. mean |Δθ| is the mean absolute change in θ from the roots init.

| Run | [0.0,0.1) | [0.1,0.2) | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) | [0.8,0.9) | [0.9,1.0] | n(|λ|>0.99) | min | max | mean | mean |Δθ| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h20 seed 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 0 | 0.965537 | 0.976440 | 0.971311 | 0.035237 |
| h20 seed 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 0 | 0.965621 | 0.977269 | 0.972215 | 0.036053 |
| h20 seed 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 0 | 0.960833 | 0.975849 | 0.970351 | 0.041785 |
| echo D=32 seed 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 68 | 0.971757 | 0.991897 | 0.985001 | 0.042316 |
| echo D=32 seed 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 59 | 0.972201 | 0.992127 | 0.985064 | 0.042913 |
| echo D=32 seed 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 70 | 0.971863 | 0.992197 | 0.985288 | 0.042604 |
| echo D=128 seed 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 0 | 0.968746 | 0.977023 | 0.972073 | 0.037584 |
| echo D=128 seed 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 0 | 0.968301 | 0.975724 | 0.972164 | 0.036576 |
| echo D=128 seed 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 0 | 0.965421 | 0.975297 | 0.972156 | 0.035659 |
| echo D=512 seed 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 0 | 0.969901 | 0.977201 | 0.973232 | 0.047153 |
| echo D=512 seed 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 0 | 0.969630 | 0.976274 | 0.973310 | 0.045101 |
| echo D=512 seed 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 0 | 0.969512 | 0.976659 | 0.973519 | 0.046901 |

## Wall seconds

| Kernel | Stream | Seed | Arm | Seconds |
|---|---|---:|---|---:|
| ezzzzzx/h30y-s0 | h20_slow | 6 | trace_rand | 190.586 |
| ezzzzzx/h30y-s0 | h20_slow | 6 | roots_learn | 188.950 |
| ezzzzzx/h30y-s0 | h20_slow | 6 | roots_frozen | 188.336 |
| ezzzzzx/h30y-s1 | h20_slow | 7 | trace_rand | 185.829 |
| ezzzzzx/h30y-s1 | h20_slow | 7 | roots_learn | 182.326 |
| ezzzzzx/h30y-s1 | h20_slow | 7 | roots_frozen | 186.965 |
| ezzzzzx/h30y-s2 | h20_slow | 8 | trace_rand | 182.457 |
| ezzzzzx/h30y-s2 | h20_slow | 8 | roots_learn | 183.098 |
| ezzzzzx/h30y-s2 | h20_slow | 8 | roots_frozen | 179.378 |
| ezzzzzx/h30y-s0 | echo D=32 | 0 | trace_rand | 47.311 |
| ezzzzzx/h30y-s0 | echo D=32 | 0 | roots_learn | 45.860 |
| ezzzzzx/h30y-s0 | echo D=32 | 0 | roots_frozen | 45.917 |
| ezzzzzx/h30y-s1 | echo D=32 | 1 | trace_rand | 50.277 |
| ezzzzzx/h30y-s1 | echo D=32 | 1 | roots_learn | 49.847 |
| ezzzzzx/h30y-s1 | echo D=32 | 1 | roots_frozen | 49.282 |
| ezzzzzx/h30y-s2 | echo D=32 | 2 | trace_rand | 45.964 |
| ezzzzzx/h30y-s2 | echo D=32 | 2 | roots_learn | 46.170 |
| ezzzzzx/h30y-s2 | echo D=32 | 2 | roots_frozen | 46.158 |
| ezzzzzx/h30y-s0 | echo D=128 | 0 | trace_rand | 46.063 |
| ezzzzzx/h30y-s0 | echo D=128 | 0 | roots_learn | 45.881 |
| ezzzzzx/h30y-s0 | echo D=128 | 0 | roots_frozen | 46.959 |
| ezzzzzx/h30y-s1 | echo D=128 | 1 | trace_rand | 51.214 |
| ezzzzzx/h30y-s1 | echo D=128 | 1 | roots_learn | 49.922 |
| ezzzzzx/h30y-s1 | echo D=128 | 1 | roots_frozen | 49.693 |
| ezzzzzx/h30y-s2 | echo D=128 | 2 | trace_rand | 46.481 |
| ezzzzzx/h30y-s2 | echo D=128 | 2 | roots_learn | 45.921 |
| ezzzzzx/h30y-s2 | echo D=128 | 2 | roots_frozen | 45.753 |
| ezzzzzx/h30y-s0 | echo D=512 | 0 | trace_rand | 45.938 |
| ezzzzzx/h30y-s0 | echo D=512 | 0 | roots_learn | 46.382 |
| ezzzzzx/h30y-s0 | echo D=512 | 0 | roots_frozen | 47.119 |
| ezzzzzx/h30y-s1 | echo D=512 | 1 | trace_rand | 49.963 |
| ezzzzzx/h30y-s1 | echo D=512 | 1 | roots_learn | 49.536 |
| ezzzzzx/h30y-s1 | echo D=512 | 1 | roots_frozen | 47.190 |
| ezzzzzx/h30y-s2 | echo D=512 | 2 | trace_rand | 46.326 |
| ezzzzzx/h30y-s2 | echo D=512 | 2 | roots_learn | 46.917 |
| ezzzzzx/h30y-s2 | echo D=512 | 2 | roots_frozen | 46.352 |

| Kernel | Total seconds |
|---|---:|
| ezzzzzx/h30y-s0 | 992.034 |
| ezzzzzx/h30y-s1 | 1008.758 |
| ezzzzzx/h30y-s2 | 968.094 |

## Kernel ids

- ezzzzzx/h30y-s0
- ezzzzzx/h30y-s1
- ezzzzzx/h30y-s2

## Smoke test

Embedding frozen = false. Train dtype = float32. Trace = h30x_trace.train_trace.

```
torch=2.15.0+cpu threads=2 train_dtype=float32 trace_impl=h30x_trace.train_trace
embedding_frozen=False roots_abs_lam=0.98 roots_theta=2*pi*k/256
trace_rand_match stream=echo D=32 seed=0 n=4096 h30x_bits=11.69228047 trace_rand_bits=11.69228047 max_abs_delta=0.000e+00 limit=1.0e-06 ok=True
delay_setup n=600 rho=0.98 rho_N=0.005674 abs_lam_min=0.97999996 abs_lam_max=0.98000008 inject_mode_spread=0.000e+00 inject_imag_rel=0.000e+00
delay d=1 rel=4.280916e-03 first_period_rel=6.982e-07 imag_rel=7.101e-07
delay d=32 rel=4.168811e-03 first_period_rel=7.779e-07 imag_rel=7.474e-07
delay d=128 rel=3.697603e-03 first_period_rel=2.366e-05 imag_rel=2.097e-05
delay d=255 rel=2.860457e-03 first_period_rel=8.105e-04 imag_rel=6.455e-04
delay d=256 rel=1.789732e+02 first_period_rel=NA imag_rel=6.402e-04
delay_alias d=256 alias_with_lag0_rel=4.908376e-06 scale_spread=6.720e-05 alias_with_lag0=True
delay_summary small_limit=2.0e-02 full_stream_small=True alias_with_lag0=True ok=True
smoke_finite stream=echo D=32 seed=0 arm=trace_rand n=4096 parameters=92160 bits=11.692280 fresh=11.821994 copy=11.562567 wall_seconds=2.898 finite=True
roots_init arm=roots_learn abs_lam_min=0.97999996 abs_lam_max=0.98000008
smoke_finite stream=echo D=32 seed=0 arm=roots_learn n=4096 parameters=92160 bits=10.647492 fresh=11.163580 copy=10.131403 wall_seconds=2.891 finite=True
roots_init arm=roots_frozen abs_lam_min=0.97999996 abs_lam_max=0.98000008
smoke_finite stream=echo D=32 seed=0 arm=roots_frozen n=4096 parameters=92160 bits=11.180803 fresh=11.615536 copy=10.746070 wall_seconds=2.725 finite=True
smoke_finite stream=echo D=128 seed=0 arm=trace_rand n=4096 parameters=92160 bits=11.702442 fresh=12.385836 copy=11.019047 wall_seconds=2.667 finite=True
roots_init arm=roots_learn stream=echo D=128 abs_lam_max=0.98000008
smoke_finite stream=echo D=128 seed=0 arm=roots_learn n=4096 parameters=92160 bits=10.599719 fresh=11.995217 copy=9.204221 wall_seconds=2.914 finite=True
roots_init arm=roots_frozen stream=echo D=128 abs_lam_max=0.98000008
smoke_finite stream=echo D=128 seed=0 arm=roots_frozen n=4096 parameters=92160 bits=10.769992 fresh=12.129956 copy=9.410028 wall_seconds=2.821 finite=True
smoke_finite stream=echo D=512 seed=0 arm=trace_rand n=4096 parameters=92160 bits=10.559650 fresh=12.821472 copy=8.297828 wall_seconds=2.758 finite=True
roots_init arm=roots_learn stream=echo D=512 abs_lam_max=0.98000008
smoke_finite stream=echo D=512 seed=0 arm=roots_learn n=4096 parameters=92160 bits=8.494239 fresh=12.537465 copy=4.451013 wall_seconds=2.631 finite=True
roots_init arm=roots_frozen stream=echo D=512 abs_lam_max=0.98000008
smoke_finite stream=echo D=512 seed=0 arm=roots_frozen n=4096 parameters=92160 bits=8.586004 fresh=12.578158 copy=4.593850 wall_seconds=2.684 finite=True
h20_repro seed=6 n=4096 match=True
smoke_finite stream=h20_slow seed=6 arm=trace_rand n=4096 parameters=107360 bits=0.934869 c1_c7=NA c1_c7_count=0 wall_seconds=2.899 finite=True
roots_init arm=roots_learn stream=h20_slow abs_lam_max=0.98000008
smoke_finite stream=h20_slow seed=6 arm=roots_learn n=4096 parameters=107360 bits=1.190395 c1_c7=NA c1_c7_count=0 wall_seconds=2.881 finite=True
roots_init arm=roots_frozen stream=h20_slow abs_lam_max=0.98000008
smoke_finite stream=h20_slow seed=6 arm=roots_frozen n=4096 parameters=107360 bits=1.263165 c1_c7=NA c1_c7_count=0 wall_seconds=2.775 finite=True
time_8192 arm=trace_rand stream=h20_slow seed=6 seconds=6.006 bits=0.666092 threads=2
time_8192 arm=roots_learn stream=h20_slow seed=6 seconds=6.222 bits=0.806712 threads=2
time_8192 arm=roots_frozen stream=h20_slow seed=6 seconds=5.538 bits=0.857593 threads=2
kernel_estimate symbols_per_arm=655360 scale=80.000000 trace_rand_seconds=480.5 roots_learn_seconds=497.8 roots_frozen_seconds=443.0 total_seconds=1421.3 limit_seconds=5400
```


## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (echo D = 128: better roots arm ≤ 3.0 on copies and ≥ 1.0 below trace_rand on all seeds): FAIL.** Both roots arms
score 4.41, worse than trace_rand (4.27) and above uniform. The kill rule fires: starting on the delay-line point does
not give a learned 128-step memory.

**Bar 2 (H20 C1 + C7, roots_learn − trace_rand ≤ +0.10): PASS.** −0.010. Totals also improve, from 0.632 to 0.612, with a
quarter of the seed spread. With λ frozen, the bank loses a little (+0.046 on the counters), so the bank wants to move λ.

**Mechanism.** The memory is there: frozen at this init, the bank holds every lag below 256 to 0.4% (smoke check 2).
What is missing is the code. The inverse-DFT readout works when every mode receives the *same* scalar projection of the
symbol (B = 1·cᵀ). A learned, generic B spreads a 32-dimensional embedding differently across the modes. Then 512 real
numbers of state must hold 128 lags × 32 dimensions, and the lags interfere. At D = 32 the window fits. There roots_learn
reaches 0.97 bits on copies, the best of any model here (trace_rand 1.59, H30w's GRU 4.01), and it does so with learned
λ, not frozen λ (1.78). At D = 128 it does not fit. The gradient does not find the rank-one, circle-coded B that would make
the window fit. On the fresh half it would cost, and the fresh half is half the stream.

**What this closes.** The search for a verbatim delay line in a learned diagonal cortex. A bank of N modes carries
statistics over a window of a few dozen steps, and it learns the counters online and locally. Holding 128 arbitrary
symbols verbatim is a store's job, not a cortex's. That fits the brain map: verbatim recall at a distance is the
hippocampus, and the cortex holds regularities. The echo stream at D = 128 is a store test, and a count model with a long
context, or an episodic store, should recall it after one symbol.

**Keep:** the trace-learned bank with the roots init (θ evenly spaced, |λ| = 0.98) is the best local cortex so far: H20
total 0.612 against the GRU's 1.177, with no unrolling.
