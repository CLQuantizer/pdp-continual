# H30w measurements

H20 slow n = 262144. Echo n = 131072. Chunk = 64. Adam lr = 0.001. Clip = 1.0. Embedding = 32. Bank and ring units = 256. GRU hidden = 166. Order-2 maps 512 to 64. Population std, ddof = 0.

Protocol: score each chunk under current parameters (−log2 p), one Adam step on the mean cross-entropy of that chunk, carry the state forward detached. No replay and no store. Input is the previous symbol. First input embedding is zero. Float32. Bank: λ = exp(−exp(ν) + iθ), |λ| initial uniform on [0.9, 0.999], θ uniform on [0, π]. Ring: ρ = exp(−exp(ν)), g = sigmoid(γ), a = ρ g exp(iθ), b = ρ (1−g) exp(iφ), so |a| + |b| = ρ < 1. ρ initial uniform on [0.9, 0.999], g uniform on [0.2, 0.8], θ and φ uniform on [0, π]. B real and imag entries initial Normal(0, 1/sqrt(32)). Bank and ring readouts have no bias. GRU is torch.nn.GRU hidden 166 plus a linear readout, pytorch default initialisation, biases on. H20 world seed 6, 7, 8 uses model seed 0, 1, 2. Echo uses the stream seed as the model seed.

## Parameter counts

| Model | Alphabet | Parameters |
|---|---:|---:|
| gru | 16 | 102784 |
| gru | 41 | 107759 |
| bank | 16 | 92160 |
| bank | 41 | 107360 |
| ring | 16 | 92672 |
| ring | 41 | 107872 |

## H20 slow, bits/symbol

| Model | mean ± std | seed 6 | seed 7 | seed 8 |
|---|---:|---:|---:|---:|
| gru | 1.176566 ± 0.078233 | 1.284826 | 1.102674 | 1.142198 |
| bank | 0.625387 ± 0.016476 | 0.646460 | 0.606240 | 0.623463 |
| ring | 0.709337 ± 0.045911 | 0.713585 | 0.763321 | 0.651104 |

## H20 per process, bits/symbol, mean ± std over seeds

| Model | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru | 0.870220 ± 0.037270 | 1.726694 ± 0.185728 | 2.200077 ± 0.056290 | 1.213462 ± 0.104190 | 1.094897 ± 0.083704 | 0.348015 ± 0.010582 | 0.688393 ± 0.155947 |
| bank | 1.422037 ± 0.102278 | 0.246414 ± 0.055393 | 0.310848 ± 0.122348 | 0.753763 ± 0.244991 | 0.588696 ± 0.424590 | 0.346684 ± 0.006252 | 0.431079 ± 0.025412 |
| ring | 0.864694 ± 0.024929 | 0.493468 ± 0.087850 | 0.504788 ± 0.137535 | 1.263373 ± 0.054777 | 1.251495 ± 0.166346 | 0.335416 ± 0.004601 | 0.382256 ± 0.007277 |

## H20 per seed per process, bits/symbol

| Model | Seed | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | 6 | 0.817562 | 1.831240 | 2.122334 | 1.319594 | 1.039687 | NA | 0.639341 |
| gru | 7 | 0.898530 | 1.465746 | 2.224120 | 1.248914 | 1.031820 | 0.358597 | 0.899130 |
| gru | 8 | 0.894568 | 1.883094 | 2.253776 | 1.071878 | 1.213185 | 0.337433 | 0.526708 |
| bank | 6 | 1.553817 | 0.181317 | 0.223287 | 1.093957 | 0.325780 | NA | 0.442474 |
| bank | 7 | 1.407787 | 0.316703 | 0.483870 | 0.640517 | 0.252638 | 0.352936 | 0.454899 |
| bank | 8 | 1.304507 | 0.241221 | 0.225387 | 0.526814 | 1.187670 | 0.340432 | 0.395864 |
| ring | 6 | 0.890345 | 0.371462 | 0.385106 | 1.338331 | 1.161009 | NA | 0.389283 |
| ring | 7 | 0.872814 | 0.534166 | 0.697411 | 1.208962 | 1.108680 | 0.340018 | 0.385254 |
| ring | 8 | 0.830924 | 0.574777 | 0.431846 | 1.242826 | 1.484796 | 0.330815 | 0.372230 |

## H20 process counts

| Seed | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 65536 | 65536 | 32768 | 16384 | 49152 | 0 | 32768 |
| 7 | 49152 | 49152 | 16384 | 49152 | 32768 | 32768 | 32768 |
| 8 | 65536 | 49152 | 32768 | 16384 | 16384 | 32768 | 49152 |

## H20 C1+C7 combined, bits/symbol

| Model | mean ± std | seed 6 | seed 7 | seed 8 | count 6 | count 7 | count 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru | 1.871659 ± 0.158646 | 1.928271 | 1.655340 | 2.031367 | 98304 | 65536 | 81920 |
| bank | 0.262897 ± 0.069503 | 0.195307 | 0.358495 | 0.234887 | 98304 | 65536 | 81920 |
| ring | 0.489531 ± 0.083619 | 0.376010 | 0.574977 | 0.517605 | 98304 | 65536 | 81920 |

## Gaps on H20 C1+C7 combined

| Seed | ring | bank | gru | ring − bank | min(bank, ring) − gru |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.376010 | 0.195307 | 1.928271 | 0.180703 | -1.732964 |
| 7 | 0.574977 | 0.358495 | 1.655340 | 0.216482 | -1.296845 |
| 8 | 0.517605 | 0.234887 | 2.031367 | 0.282717 | -1.796479 |
| mean ± std |  |  |  | 0.226634 ± 0.042261 | -1.608763 ± 0.222078 |

## Echo D = 32, bits/symbol

| Model | half | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|---:|
| gru | fresh | 4.007846 ± 0.000163 | 4.008030 | 4.007874 | 4.007633 |
| gru | copy | 4.007769 ± 0.000150 | 4.007870 | 4.007881 | 4.007558 |
| bank | fresh | 5.340563 ± 0.066160 | 5.432083 | 5.311653 | 5.277954 |
| bank | copy | 1.792200 ± 0.110562 | 1.935822 | 1.773919 | 1.666858 |
| ring | fresh | 4.148908 ± 0.005943 | 4.152523 | 4.153673 | 4.140529 |
| ring | copy | 4.148535 ± 0.006195 | 4.151873 | 4.153882 | 4.139851 |

## Echo D = 32, totals, bits/symbol

| Model | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|
| gru | 4.007808 ± 0.000153 | 4.007950 | 4.007877 | 4.007595 |
| bank | 3.566382 ± 0.087960 | 3.683953 | 3.542786 | 3.472406 |
| ring | 4.148722 ± 0.006067 | 4.152198 | 4.153777 | 4.140190 |

## Gaps on echo D = 32 copy halves

| Seed | ring | bank | gru | ring − bank | min(bank, ring) − gru |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.151873 | 1.935822 | 4.007870 | 2.216051 | -2.072048 |
| 1 | 4.153882 | 1.773919 | 4.007881 | 2.379963 | -2.233961 |
| 2 | 4.139851 | 1.666858 | 4.007558 | 2.472993 | -2.340700 |
| mean ± std |  |  |  | 2.356335 ± 0.106218 | -2.215570 ± 0.110445 |

## Echo D = 128, bits/symbol

| Model | half | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|---:|
| gru | fresh | 4.009443 ± 0.000101 | 4.009464 | 4.009555 | 4.009311 |
| gru | copy | 4.003786 ± 0.000210 | 4.004044 | 4.003783 | 4.003530 |
| bank | fresh | 4.724031 ± 0.089199 | 4.844803 | 4.695194 | 4.632096 |
| bank | copy | 4.410850 ± 0.068103 | 4.501481 | 4.393761 | 4.337310 |
| ring | fresh | 4.160987 ± 0.006667 | 4.164895 | 4.166464 | 4.151602 |
| ring | copy | 4.074336 ± 0.003428 | 4.075828 | 4.077585 | 4.069596 |

## Echo D = 128, totals, bits/symbol

| Model | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|
| gru | 4.006615 ± 0.000141 | 4.006754 | 4.006669 | 4.006420 |
| bank | 4.567441 ± 0.078625 | 4.673142 | 4.544478 | 4.484703 |
| ring | 4.117662 ± 0.005040 | 4.120362 | 4.122025 | 4.110599 |

## Gaps on echo D = 128 copy halves

| Seed | ring | bank | gru | ring − bank | min(bank, ring) − gru |
|---:|---:|---:|---:|---:|---:|
| 0 | 4.075828 | 4.501481 | 4.004044 | -0.425653 | 0.071784 |
| 1 | 4.077585 | 4.393761 | 4.003783 | -0.316176 | 0.073802 |
| 2 | 4.069596 | 4.337310 | 4.003530 | -0.267713 | 0.066066 |
| mean ± std |  |  |  | -0.336514 ± 0.066063 | 0.070551 ± 0.003276 |

## Ring |a| and |b| histograms

Ten bins on [0, 1]. Edges 0.0, 0.1, ..., 1.0. Counts are the learned magnitudes at the end of the run.

| Run | quantity | [0.0,0.1) | [0.1,0.2) | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) | [0.8,0.9) | [0.9,1.0] | min | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h20 seed 6 | |a| | 0 | 1 | 52 | 44 | 41 | 42 | 44 | 32 | 0 | 0 | 0.196956 | 0.776480 | 0.474236 |
| h20 seed 6 | |b| | 0 | 7 | 50 | 42 | 39 | 41 | 48 | 29 | 0 | 0 | 0.176766 | 0.771553 | 0.475912 |
| echo D=32 seed 0 | |a| | 0 | 2 | 44 | 48 | 57 | 39 | 54 | 12 | 0 | 0 | 0.193908 | 0.773897 | 0.464885 |
| echo D=32 seed 0 | |b| | 0 | 1 | 52 | 37 | 48 | 55 | 49 | 14 | 0 | 0 | 0.197123 | 0.769147 | 0.470902 |
| echo D=128 seed 0 | |a| | 0 | 1 | 43 | 47 | 61 | 46 | 49 | 9 | 0 | 0 | 0.194656 | 0.759112 | 0.463874 |
| echo D=128 seed 0 | |b| | 0 | 0 | 49 | 39 | 57 | 56 | 45 | 10 | 0 | 0 | 0.200543 | 0.763622 | 0.467197 |
| h20 seed 7 | |a| | 0 | 3 | 46 | 43 | 48 | 51 | 35 | 28 | 2 | 0 | 0.187843 | 0.889802 | 0.473904 |
| h20 seed 7 | |b| | 1 | 4 | 44 | 41 | 48 | 51 | 43 | 24 | 0 | 0 | 0.087149 | 0.765885 | 0.472678 |
| echo D=32 seed 1 | |a| | 0 | 1 | 43 | 44 | 52 | 55 | 48 | 13 | 0 | 0 | 0.194094 | 0.750486 | 0.471354 |
| echo D=32 seed 1 | |b| | 0 | 0 | 45 | 58 | 44 | 49 | 47 | 13 | 0 | 0 | 0.204672 | 0.748145 | 0.463499 |
| echo D=128 seed 1 | |a| | 0 | 0 | 37 | 50 | 52 | 63 | 43 | 11 | 0 | 0 | 0.201894 | 0.753857 | 0.469571 |
| echo D=128 seed 1 | |b| | 0 | 0 | 40 | 62 | 47 | 51 | 47 | 9 | 0 | 0 | 0.202077 | 0.747314 | 0.460645 |
| h20 seed 8 | |a| | 0 | 1 | 45 | 47 | 47 | 53 | 35 | 27 | 1 | 0 | 0.198693 | 0.803792 | 0.478154 |
| h20 seed 8 | |b| | 0 | 7 | 41 | 41 | 55 | 49 | 44 | 19 | 0 | 0 | 0.164250 | 0.784935 | 0.472425 |
| echo D=32 seed 2 | |a| | 0 | 1 | 55 | 42 | 50 | 53 | 44 | 11 | 0 | 0 | 0.189970 | 0.743796 | 0.457214 |
| echo D=32 seed 2 | |b| | 0 | 0 | 36 | 49 | 58 | 44 | 48 | 21 | 0 | 0 | 0.202303 | 0.764240 | 0.479188 |
| echo D=128 seed 2 | |a| | 0 | 1 | 54 | 41 | 57 | 49 | 44 | 10 | 0 | 0 | 0.195277 | 0.739944 | 0.455677 |
| echo D=128 seed 2 | |b| | 0 | 0 | 34 | 50 | 63 | 42 | 51 | 16 | 0 | 0 | 0.207531 | 0.756670 | 0.476455 |

## Wall seconds

| Kernel | Stream | Seed | Model | Seconds |
|---|---|---:|---|---:|
| ezzzzzx/h30w-s0 | h20_slow | 6 | gru | 115.454 |
| ezzzzzx/h30w-s0 | h20_slow | 6 | bank | 48.500 |
| ezzzzzx/h30w-s0 | h20_slow | 6 | ring | 74.663 |
| ezzzzzx/h30w-s0 | echo D=32 | 0 | gru | 57.863 |
| ezzzzzx/h30w-s0 | echo D=32 | 0 | bank | 23.717 |
| ezzzzzx/h30w-s0 | echo D=32 | 0 | ring | 36.008 |
| ezzzzzx/h30w-s0 | echo D=128 | 0 | gru | 57.512 |
| ezzzzzx/h30w-s0 | echo D=128 | 0 | bank | 23.720 |
| ezzzzzx/h30w-s0 | echo D=128 | 0 | ring | 36.395 |
| ezzzzzx/h30w-s1 | h20_slow | 7 | gru | 112.637 |
| ezzzzzx/h30w-s1 | h20_slow | 7 | bank | 43.536 |
| ezzzzzx/h30w-s1 | h20_slow | 7 | ring | 68.272 |
| ezzzzzx/h30w-s1 | echo D=32 | 1 | gru | 56.708 |
| ezzzzzx/h30w-s1 | echo D=32 | 1 | bank | 22.069 |
| ezzzzzx/h30w-s1 | echo D=32 | 1 | ring | 33.797 |
| ezzzzzx/h30w-s1 | echo D=128 | 1 | gru | 54.838 |
| ezzzzzx/h30w-s1 | echo D=128 | 1 | bank | 21.551 |
| ezzzzzx/h30w-s1 | echo D=128 | 1 | ring | 33.574 |
| ezzzzzx/h30w-s2 | h20_slow | 8 | gru | 109.725 |
| ezzzzzx/h30w-s2 | h20_slow | 8 | bank | 46.906 |
| ezzzzzx/h30w-s2 | h20_slow | 8 | ring | 70.280 |
| ezzzzzx/h30w-s2 | echo D=32 | 2 | gru | 54.179 |
| ezzzzzx/h30w-s2 | echo D=32 | 2 | bank | 22.994 |
| ezzzzzx/h30w-s2 | echo D=32 | 2 | ring | 34.668 |
| ezzzzzx/h30w-s2 | echo D=128 | 2 | gru | 53.756 |
| ezzzzzx/h30w-s2 | echo D=128 | 2 | bank | 23.084 |
| ezzzzzx/h30w-s2 | echo D=128 | 2 | ring | 35.216 |

| Kernel | Total seconds |
|---|---:|
| ezzzzzx/h30w-s0 | 481.260 |
| ezzzzzx/h30w-s1 | 453.336 |
| ezzzzzx/h30w-s2 | 457.698 |

## Kernel ids

- ezzzzzx/h30w-s0
- ezzzzzx/h30w-s1
- ezzzzzx/h30w-s2

## Smoke test

```
torch=2.15.0+cpu threads=2 dtype=float32
parameters model=gru alphabet=41 stream=h20 count=107759
parameters model=bank alphabet=41 stream=h20 count=107360
init_bank abs_eig_min=0.900043 abs_eig_max=0.998912 theta_min=0.010735 theta_max=3.134428
parameters model=ring alphabet=41 stream=h20 count=107872
init_ring abs_eig_min=0.900043 abs_eig_max=0.998912 theta_min=0.010735 theta_max=3.134428
init_ring g_min=0.203283 g_max=0.798551 phi_min=0.007434 phi_max=3.141531 max_abs_a_plus_abs_b=0.99891186
parameters model=gru alphabet=16 stream=echo count=102784
parameters model=bank alphabet=16 stream=echo count=92160
parameters model=ring alphabet=16 stream=echo count=92672
echo_check D=32 n=131072 fresh_count=65536 copy_matches=True aligned=True ok=True
echo_check D=128 n=131072 fresh_count=65536 copy_matches=True aligned=True ok=True
h20_repro seed=6 n=4096 match=True
smoke_finite stream=h20_slow seed=6 model=gru n=4096 parameters=107759 bits=2.511359 wall_seconds=5.756 finite=True
smoke_finite stream=h20_slow seed=6 model=bank n=4096 parameters=107360 bits=0.952492 wall_seconds=1.514 finite=True
smoke_finite stream=h20_slow seed=6 model=ring n=4096 parameters=107872 bits=0.649013 wall_seconds=2.324 finite=True
smoke_ring_recurrence_max_abs_delta=2.93607712e-02
smoke_finite stream=echo D=32 seed=0 model=gru n=4096 parameters=102784 bits=4.012468 fresh=4.012856 copy=4.012079 wall_seconds=4.413 finite=True
smoke_finite stream=echo D=32 seed=0 model=bank n=4096 parameters=92160 bits=11.276620 fresh=11.545076 copy=11.008164 wall_seconds=1.348 finite=True
smoke_finite stream=echo D=32 seed=0 model=ring n=4096 parameters=92672 bits=4.507526 fresh=4.509305 copy=4.505747 wall_seconds=2.152 finite=True
smoke_finite stream=echo D=128 seed=0 model=gru n=4096 parameters=102784 bits=4.009299 fresh=4.017891 copy=4.000706 wall_seconds=4.273 finite=True
smoke_finite stream=echo D=128 seed=0 model=bank n=4096 parameters=92160 bits=12.429276 fresh=13.317328 copy=11.541224 wall_seconds=1.394 finite=True
smoke_finite stream=echo D=128 seed=0 model=ring n=4096 parameters=92672 bits=4.447774 fresh=4.703295 copy=4.192253 wall_seconds=2.121 finite=True
ring_check handset a=0 b=1 injection=unit0_real steps=256
ring_check k=0 held_real=256.000000 held_imag=0.000000 input_k_steps_earlier=256.000000 match=True
ring_check k=1 held_real=255.000000 held_imag=0.000000 input_k_steps_earlier=255.000000 match=True
ring_check k=5 held_real=251.000000 held_imag=0.000000 input_k_steps_earlier=251.000000 match=True
ring_check k=200 held_real=56.000000 held_imag=0.000000 input_k_steps_earlier=56.000000 match=True
prequential_check model=bank chunk=2 end=192 max_abs_delta_through_chunk=0.00000000e+00 max_abs_delta_after=3.91480765e+01 unchanged=True
time_8192 model=gru seconds=14.708 bits=1.418268 threads=2
time_8192 model=bank seconds=3.423 bits=0.676713 threads=2
time_8192 model=ring seconds=4.647 bits=0.482004 threads=2
kernel_estimate symbols_per_model=524288 scale=64.000000 gru_seconds=941.3 bank_seconds=219.1 ring_seconds=297.4 total_seconds=1457.8 limit_seconds=5400
stability alphabet=41 model_seed=0 max_abs_a_plus_abs_b=0.99891186 min_abs_a_plus_abs_b=0.90004265 rho_min=0.900043 rho_max=0.998912 g_min=0.203283 g_max=0.798551 theta_min=0.010735 theta_max=3.134428 phi_min=0.007434 phi_max=3.141531 stable=true
stability alphabet=41 model_seed=1 max_abs_a_plus_abs_b=0.99847603 min_abs_a_plus_abs_b=0.90017307 rho_min=0.900173 rho_max=0.998476 g_min=0.200817 g_max=0.799040 theta_min=0.000648 theta_max=3.136208 phi_min=0.020374 phi_max=3.104039 stable=true
stability alphabet=41 model_seed=2 max_abs_a_plus_abs_b=0.99891919 min_abs_a_plus_abs_b=0.90014315 rho_min=0.900143 rho_max=0.998919 g_min=0.210205 g_max=0.797923 theta_min=0.006025 theta_max=3.128793 phi_min=0.044245 phi_max=3.112986 stable=true
stability alphabet=16 model_seed=0 max_abs_a_plus_abs_b=0.99891531 min_abs_a_plus_abs_b=0.90120375 rho_min=0.901204 rho_max=0.998915 g_min=0.202375 g_max=0.799505 theta_min=0.000250 theta_max=3.138986 phi_min=0.010101 phi_max=3.137024 stable=true
stability alphabet=16 model_seed=1 max_abs_a_plus_abs_b=0.99816096 min_abs_a_plus_abs_b=0.90036213 rho_min=0.900362 rho_max=0.998161 g_min=0.207281 g_max=0.797447 theta_min=0.000271 theta_max=3.140451 phi_min=0.003079 phi_max=3.122753 stable=true
stability alphabet=16 model_seed=2 max_abs_a_plus_abs_b=0.99887168 min_abs_a_plus_abs_b=0.90033948 rho_min=0.900339 rho_max=0.998872 g_min=0.201380 g_max=0.799588 theta_min=0.004229 theta_max=3.133848 phi_min=0.005087 phi_max=3.131140 stable=true
```


## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (echo D = 128: ring − bank on copies ≤ −0.25, ring better on all 3 seeds): PASS by the letter, hollow.**
The ring − bank gap is −0.337 and the ring is lower on every seed. But the ring's copy bits are 4.074, above the uniform 4.000. The GRU sits at 4.004. No model holds a 128-step delay. The ring "wins" only because the bank does worse than uniform at D = 128 (4.41 on copies, 4.72 on fresh). At D = 32 the order is reversed: the bank learns the copies (1.79 bits; floor 0) and the ring learns nothing (4.15). So propagation is not shown, and the ring is parked as if the kill rule had fired. The bar was my error: it compared two models without anchoring either one to the uniform rate. Future delay bars will also require copy bits < 4 − margin.

**Mechanism for D = 128.** Training uses BPTT inside 64-symbol chunks, and state is carried forward detached. A 128-step dependency never gets a gradient, so no model can learn it, whatever its substrate. The D = 32 result shows the bank can store and read a delay when credit reaches it. The bank's fresh-half cost (5.34 at D = 32) is the price of predicting "copy" with confidence when it shouldn't. It has no cue for which half it is in, beyond the phase it has learned.

**Bar 2 (H20: min(bank, ring) − gru on C1 + C7 ≤ −0.10): PASS, by a wide margin.** −1.609 ± 0.222. The bank is best on every seed: C1 + C7 0.263 vs the GRU's 1.872, at equal size (107k parameters). The bank also beats the GRU on totals (0.625 vs 1.177), and on E30, E110 and M37. It is worse on the grammar G (1.42 vs 0.87). This is the first online substrate here that learns a counter's carry. The H20/H21a failure was the GRU's, not online learning's. Damped rotations count, and the order-2 readout reads the count out.

**Next.** The bank is diagonal, so its exact gradient through all time is local: each unit keeps an eligibility trace e_t = λ e_{t−1} + ∂h_t/∂λ (exact RTRL for diagonal recurrences, Zucchet et al. 2023). That removes the truncation and the backward pass. Learning becomes online and local in time: one forward sweep, no unrolling. This is the north-star direction. H30x tests it: a bank with trace learning should learn echo D = 128 where BPTT-64 cannot.
