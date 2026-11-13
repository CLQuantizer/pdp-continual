# H30x measurements

H20 slow n = 262144. Echo n = 131072, D in [32, 128, 512]. Chunk = 64. Adam lr = 0.001. Clip = 1.0. Embedding = 32. Units = 256. Order-2 maps 512 to 64. Population std, ddof = 0. Embedding frozen = false.

Protocol: one bank, two learning rules. The bank is h30w_ring.Bank, N = 256, h_t = λ ⊙ h_{t−1} + B x_t, λ = exp(−exp(ν) + iθ). |λ| initial uniform on [0.9, 0.999], θ uniform on [0, π]. B real and imag entries initial Normal(0, 1/sqrt(32)). Readout is order-2 with no bias: W, P1, P2, V. Input is the previous symbol. The first input embedding is zero. Float32. bptt64 is h30w_ring.train: score a chunk of 64 under the current parameters, one Adam step on the mean cross-entropy, carry the state forward detached. trace uses the same chunks, the same Adam, and the same score-before-step order. Readout gradients are autograd of the chunk losses through W, P1, P2, V with h detached. Recurrent gradients use complex eligibility traces carried across chunks and never reset: e^λ_t = λ e^λ_{t−1} + h_{t−1}, E^B_t = λ E^B_{t−1} + x_t, and z_{a,t} = λ z_{a,t−1} + 1{symbol_{t−1} = a} (no addition on the first step). δ_t = ∂L_t/∂h_t from the step-t readout. The real parameter gradient is the real part of conj(δ) times the sensitivity, chained from λ to ν and θ. The embedding gradient uses z and B. Traces use the current λ and are not recomputed after the Adam step. Summed chunk gradients are divided by the chunk length. H20 world seeds 6, 7, 8 use model seeds 0, 1, 2. Echo uses the stream seed as the model seed. Echo uniform rate is 4 bits on both halves. Copy last quarter is the mean over copy symbols in the last quarter of the stream.

## Parameter counts

| Alphabet | Parameters |
|---:|---:|
| 16 | 92160 |
| 41 | 107360 |

## H20 slow, bits/symbol

| Arm | mean ± std | seed 6 | seed 7 | seed 8 |
|---|---:|---:|---:|---:|
| bptt64 | 0.625387 ± 0.016476 | 0.646460 | 0.606240 | 0.623463 |
| trace | 0.631740 ± 0.027936 | 0.661936 | 0.594579 | 0.638705 |

## H20 per process, bits/symbol, mean ± std over seeds

| Arm | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bptt64 | 1.422037 ± 0.102278 | 0.246414 ± 0.055393 | 0.310848 ± 0.122348 | 0.753763 ± 0.244991 | 0.588696 ± 0.424590 | 0.346684 ± 0.006252 | 0.431079 ± 0.025412 |
| trace | 1.490360 ± 0.137844 | 0.256014 ± 0.082286 | 0.280484 ± 0.067779 | 0.706410 ± 0.233857 | 0.554923 ± 0.410775 | 0.366202 ± 0.030538 | 0.423524 ± 0.022624 |

## H20 per seed per process, bits/symbol

| Arm | Seed | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bptt64 | 6 | 1.553817 | 0.181317 | 0.223287 | 1.093957 | 0.325780 | NA | 0.442474 |
| bptt64 | 7 | 1.407787 | 0.316703 | 0.483870 | 0.640517 | 0.252638 | 0.352936 | 0.454899 |
| bptt64 | 8 | 1.304507 | 0.241221 | 0.225387 | 0.526814 | 1.187670 | 0.340432 | 0.395864 |
| trace | 6 | 1.677373 | 0.162496 | 0.232563 | 1.036617 | 0.284940 | NA | 0.437466 |
| trace | 7 | 1.444512 | 0.362750 | 0.376338 | 0.557307 | 0.244453 | 0.335664 | 0.441493 |
| trace | 8 | 1.349196 | 0.242795 | 0.232552 | 0.525305 | 1.135377 | 0.396740 | 0.391614 |

## H20 process counts

| Seed | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 65536 | 65536 | 32768 | 16384 | 49152 | 0 | 32768 |
| 7 | 49152 | 49152 | 16384 | 49152 | 32768 | 32768 | 32768 |
| 8 | 65536 | 49152 | 32768 | 16384 | 16384 | 32768 | 49152 |

## H20 C1+C7 combined, bits/symbol

| Arm | mean ± std | seed 6 | seed 7 | seed 8 | count 6 | count 7 | count 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bptt64 | 0.262897 ± 0.069503 | 0.195307 | 0.358495 | 0.234887 | 98304 | 65536 | 81920 |
| trace | 0.263566 ± 0.075677 | 0.185852 | 0.366147 | 0.238698 | 98304 | 65536 | 81920 |

## Gaps on H20 C1+C7 combined, trace − bptt64

| Seed | trace | bptt64 | trace − bptt64 |
|---:|---:|---:|---:|
| 6 | 0.185852 | 0.195307 | -0.009455 |
| 7 | 0.366147 | 0.358495 | 0.007652 |
| 8 | 0.238698 | 0.234887 | 0.003810 |
| mean ± std |  |  | 0.000669 ± 0.007329 |

## Echo D = 32, bits/symbol

| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|---:|
| bptt64 | fresh | 5.340563 ± 0.066160 | 5.432083 | 5.311653 | 5.277954 |
| bptt64 | copy | 1.792200 ± 0.110562 | 1.935822 | 1.773919 | 1.666858 |
| trace | fresh | 5.092031 ± 0.087205 | 5.212680 | 5.053839 | 5.009575 |
| trace | copy | 1.589833 ± 0.144076 | 1.783928 | 1.546469 | 1.439101 |

## Echo D = 32, copy bits in the last quarter of the stream

| Arm | mean ± std | seed 0 | seed 1 | seed 2 | count |
|---|---:|---:|---:|---:|---:|
| bptt64 | 0.784669 ± 0.029647 | 0.821884 | 0.782785 | 0.749336 | 16384 |
| trace | 0.533419 ± 0.030662 | 0.576148 | 0.518452 | 0.505656 | 16384 |

## Echo D = 32, totals, bits/symbol

| Arm | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|
| bptt64 | 3.566382 ± 0.087960 | 3.683953 | 3.542786 | 3.472406 |
| trace | 3.340932 ± 0.115503 | 3.498304 | 3.300154 | 3.224338 |

## Gaps on echo D = 32 copy halves, trace − bptt64

| Seed | trace | bptt64 | trace − bptt64 |
|---:|---:|---:|---:|
| 0 | 1.783928 | 1.935822 | -0.151895 |
| 1 | 1.546469 | 1.773919 | -0.227450 |
| 2 | 1.439101 | 1.666858 | -0.227758 |
| mean ± std |  |  | -0.202367 ± 0.035690 |

## Echo D = 128, bits/symbol

| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|---:|
| bptt64 | fresh | 4.724031 ± 0.089199 | 4.844803 | 4.695194 | 4.632096 |
| bptt64 | copy | 4.410850 ± 0.068103 | 4.501481 | 4.393761 | 4.337310 |
| trace | fresh | 4.672466 ± 0.055485 | 4.746026 | 4.659344 | 4.612029 |
| trace | copy | 4.274956 ± 0.016089 | 4.279956 | 4.291679 | 4.253232 |

## Echo D = 128, copy bits in the last quarter of the stream

| Arm | mean ± std | seed 0 | seed 1 | seed 2 | count |
|---|---:|---:|---:|---:|---:|
| bptt64 | 4.026283 ± 0.001272 | 4.024508 | 4.027426 | 4.026915 | 16384 |
| trace | 3.907076 ± 0.026213 | 3.870013 | 3.924928 | 3.926286 | 16384 |

## Echo D = 128, totals, bits/symbol

| Arm | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|
| bptt64 | 4.567441 ± 0.078625 | 4.673142 | 4.544478 | 4.484703 |
| trace | 4.473711 ± 0.032832 | 4.512991 | 4.475511 | 4.432631 |

## Gaps on echo D = 128 copy halves, trace − bptt64

| Seed | trace | bptt64 | trace − bptt64 |
|---:|---:|---:|---:|
| 0 | 4.279956 | 4.501481 | -0.221525 |
| 1 | 4.291679 | 4.393761 | -0.102082 |
| 2 | 4.253232 | 4.337310 | -0.084078 |
| mean ± std |  |  | -0.135895 ± 0.060994 |

## Echo D = 512, bits/symbol

| Arm | half | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---|---:|---:|---:|---:|
| bptt64 | fresh | 4.939496 ± 0.101558 | 5.062224 | 4.942743 | 4.813521 |
| bptt64 | copy | 3.804233 ± 0.026939 | 3.839330 | 3.799518 | 3.773851 |
| trace | fresh | 4.913085 ± 0.097748 | 5.036582 | 4.905127 | 4.797547 |
| trace | copy | 3.783280 ± 0.035196 | 3.825286 | 3.785402 | 3.739153 |

## Echo D = 512, copy bits in the last quarter of the stream

| Arm | mean ± std | seed 0 | seed 1 | seed 2 | count |
|---|---:|---:|---:|---:|---:|
| bptt64 | 3.730790 ± 0.014950 | 3.715552 | 3.725716 | 3.751102 | 16384 |
| trace | 3.738984 ± 0.006442 | 3.747973 | 3.733200 | 3.735780 | 16384 |

## Echo D = 512, totals, bits/symbol

| Arm | mean ± std | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|---:|
| bptt64 | 4.371865 ± 0.064134 | 4.450777 | 4.371131 | 4.293686 |
| trace | 4.348183 ± 0.066407 | 4.430934 | 4.345265 | 4.268350 |

## Gaps on echo D = 512 copy halves, trace − bptt64

| Seed | trace | bptt64 | trace − bptt64 |
|---:|---:|---:|---:|
| 0 | 3.825286 | 3.839330 | -0.014044 |
| 1 | 3.785402 | 3.799518 | -0.014116 |
| 2 | 3.739153 | 3.773851 | -0.034699 |
| mean ± std |  |  | -0.020953 ± 0.009720 |

## Learned |λ|

Ten bins on [0, 1]. Edges 0.0, 0.1, ..., 1.0. Counts are |λ| at the end of the run. n(|λ| > 0.99) is a separate count, not an extra bin.

| Run | [0.0,0.1) | [0.1,0.2) | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,0.6) | [0.6,0.7) | [0.7,0.8) | [0.8,0.9) | [0.9,1.0] | n(|λ|>0.99) | min | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| h20 seed 6 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 49 | 207 | 13 | 0.867775 | 0.997130 | 0.937119 |
| h20 seed 6 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 54 | 202 | 16 | 0.862854 | 0.997562 | 0.937506 |
| h20 seed 7 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 51 | 205 | 11 | 0.854359 | 0.995628 | 0.936091 |
| h20 seed 7 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 49 | 207 | 12 | 0.844905 | 0.996601 | 0.935811 |
| h20 seed 8 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 54 | 202 | 15 | 0.863733 | 0.997061 | 0.934678 |
| h20 seed 8 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 58 | 198 | 19 | 0.868945 | 0.997627 | 0.934644 |
| echo D=32 seed 0 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 244 | 35 | 0.885835 | 0.998853 | 0.956631 |
| echo D=32 seed 0 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 230 | 56 | 0.871579 | 0.998232 | 0.955955 |
| echo D=32 seed 1 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 243 | 29 | 0.890859 | 0.998646 | 0.955541 |
| echo D=32 seed 1 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 234 | 51 | 0.882515 | 0.997792 | 0.954605 |
| echo D=32 seed 2 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 242 | 26 | 0.883816 | 0.998493 | 0.955732 |
| echo D=32 seed 2 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | 232 | 51 | 0.878027 | 0.998163 | 0.955673 |
| echo D=128 seed 0 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 73 | 183 | 20 | 0.857881 | 0.997843 | 0.929759 |
| echo D=128 seed 0 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 65 | 191 | 34 | 0.855172 | 0.998481 | 0.932544 |
| echo D=128 seed 1 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 74 | 182 | 14 | 0.855207 | 0.996678 | 0.928939 |
| echo D=128 seed 1 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 72 | 184 | 18 | 0.857854 | 0.998194 | 0.930385 |
| echo D=128 seed 2 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 72 | 184 | 12 | 0.854139 | 0.997659 | 0.930095 |
| echo D=128 seed 2 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 69 | 187 | 19 | 0.862073 | 0.998529 | 0.932425 |
| echo D=512 seed 0 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 54 | 202 | 20 | 0.873133 | 0.997581 | 0.935100 |
| echo D=512 seed 0 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 58 | 198 | 20 | 0.864905 | 0.997839 | 0.934330 |
| echo D=512 seed 1 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 54 | 202 | 14 | 0.866265 | 0.996443 | 0.934062 |
| echo D=512 seed 1 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 57 | 199 | 16 | 0.863993 | 0.996664 | 0.933777 |
| echo D=512 seed 2 bptt64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 58 | 198 | 13 | 0.868759 | 0.997374 | 0.935681 |
| echo D=512 seed 2 trace | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 58 | 198 | 13 | 0.864091 | 0.997945 | 0.935117 |

## Wall seconds

| Kernel | Stream | Seed | Arm | Seconds |
|---|---|---:|---|---:|
| ezzzzzx/h30x-s0 | h20_slow | 6 | bptt64 | 52.473 |
| ezzzzzx/h30x-s0 | h20_slow | 6 | trace | 198.117 |
| ezzzzzx/h30x-s1 | h20_slow | 7 | bptt64 | 49.802 |
| ezzzzzx/h30x-s1 | h20_slow | 7 | trace | 186.169 |
| ezzzzzx/h30x-s2 | h20_slow | 8 | bptt64 | 48.618 |
| ezzzzzx/h30x-s2 | h20_slow | 8 | trace | 205.928 |
| ezzzzzx/h30x-s0 | echo D=32 | 0 | bptt64 | 23.177 |
| ezzzzzx/h30x-s0 | echo D=32 | 0 | trace | 48.956 |
| ezzzzzx/h30x-s1 | echo D=32 | 1 | bptt64 | 24.192 |
| ezzzzzx/h30x-s1 | echo D=32 | 1 | trace | 50.030 |
| ezzzzzx/h30x-s2 | echo D=32 | 2 | bptt64 | 22.742 |
| ezzzzzx/h30x-s2 | echo D=32 | 2 | trace | 52.848 |
| ezzzzzx/h30x-s0 | echo D=128 | 0 | bptt64 | 23.872 |
| ezzzzzx/h30x-s0 | echo D=128 | 0 | trace | 53.529 |
| ezzzzzx/h30x-s1 | echo D=128 | 1 | bptt64 | 23.983 |
| ezzzzzx/h30x-s1 | echo D=128 | 1 | trace | 49.279 |
| ezzzzzx/h30x-s2 | echo D=128 | 2 | bptt64 | 22.052 |
| ezzzzzx/h30x-s2 | echo D=128 | 2 | trace | 51.339 |
| ezzzzzx/h30x-s0 | echo D=512 | 0 | bptt64 | 26.406 |
| ezzzzzx/h30x-s0 | echo D=512 | 0 | trace | 48.262 |
| ezzzzzx/h30x-s1 | echo D=512 | 1 | bptt64 | 24.910 |
| ezzzzzx/h30x-s1 | echo D=512 | 1 | trace | 48.713 |
| ezzzzzx/h30x-s2 | echo D=512 | 2 | bptt64 | 23.209 |
| ezzzzzx/h30x-s2 | echo D=512 | 2 | trace | 49.764 |

| Kernel | Total seconds |
|---|---:|
| ezzzzzx/h30x-s0 | 482.978 |
| ezzzzzx/h30x-s1 | 464.205 |
| ezzzzzx/h30x-s2 | 483.595 |

## Kernel ids

- ezzzzzx/h30x-s0
- ezzzzzx/h30x-s1
- ezzzzzx/h30x-s2

## Smoke test

Embedding frozen = false. Train dtype = float32. Gradient check dtype = float64.

```
torch=2.15.0+cpu threads=2 train_dtype=float32 grad_check_dtype=float64
embedding_frozen=False bptt64_impl=h30w_ring.train
parameters arm=bank alphabet=41 stream=h20 count=107360 abs_lam_min=0.900043 abs_lam_max=0.998912
parameters arm=bank alphabet=16 stream=echo count=92160 abs_lam_min=0.901204 abs_lam_max=0.998915
echo_check D=32 n=131072 fresh_count=65536 copy_matches=True aligned=True last_quarter_copy_count=16384 ok=True
echo_check D=128 n=131072 fresh_count=65536 copy_matches=True aligned=True last_quarter_copy_count=16384 ok=True
echo_check D=512 n=131072 fresh_count=65536 copy_matches=True aligned=True last_quarter_copy_count=16384 ok=True
trace_grad_check D=128 seed=0 n=300 dtype=float64 chunk=64 loss_bptt=2999.39041270 loss_trace=2999.39041270 loss_abs=1.364e-12
trace_grad name=nu bptt_norm=3.54432684e+02 trace_norm=3.54432684e+02 max_abs=1.13686838e-13 max_rel=3.77691633e-16 max_elem_rel=4.42294249e-14
trace_grad name=theta bptt_norm=2.40173339e+04 trace_norm=2.40173339e+04 max_abs=1.31876732e-11 max_rel=9.28694020e-16 max_elem_rel=9.04808097e-14
trace_grad name=B_re bptt_norm=3.50079003e+03 trace_norm=3.50079003e+03 max_abs=1.13686838e-12 max_rel=7.53893795e-16 max_elem_rel=1.23278992e-12
trace_grad name=B_im bptt_norm=1.42914362e+03 trace_norm=1.42914362e+03 max_abs=3.12638804e-13 max_rel=8.61144759e-16 max_elem_rel=2.10456737e-12
trace_grad name=W.weight bptt_norm=4.49918057e+03 trace_norm=4.49918057e+03 max_abs=1.36424205e-12 max_rel=5.77797478e-16 max_elem_rel=2.77710786e-11
trace_grad name=P1.weight bptt_norm=6.65157609e+03 trace_norm=6.65157609e+03 max_abs=1.13686838e-12 max_rel=8.14095480e-16 max_elem_rel=4.64869356e-12
trace_grad name=P2.weight bptt_norm=6.96953223e+03 trace_norm=6.96953223e+03 max_abs=1.36424205e-12 max_rel=7.82264855e-16 max_elem_rel=6.23667918e-12
trace_grad name=V.weight bptt_norm=2.22295761e+03 trace_norm=2.22295761e+03 max_abs=2.27373675e-13 max_rel=4.39054558e-16 max_elem_rel=3.82389515e-13
trace_grad name=embedding.weight bptt_norm=7.05676391e+02 trace_norm=7.05676391e+02 max_abs=7.81597009e-14 max_rel=8.49177689e-16 max_elem_rel=6.91861726e-14
trace_grad_summary max_rel_spec=9.28694020e-16 max_elem_rel_spec=2.77710786e-11 embedding_max_rel=8.49177689e-16 limit=1.0e-04 ok=True
bptt64_match stream=echo D=32 seed=0 n=4096 h30w_bits=11.27662024 bptt64_bits=11.27662024 max_abs_delta=0.000e+00 mean_abs_delta=0.000e+00 limit=1.0e-06 ok=True
smoke_finite stream=echo D=32 seed=0 arm=bptt64 n=4096 parameters=92160 bits=11.276620 fresh=11.545076 copy=11.008164 wall_seconds=1.331 finite=True
smoke_finite stream=echo D=32 seed=0 arm=trace n=4096 parameters=92160 bits=11.692280 fresh=11.821994 copy=11.562567 wall_seconds=2.211 finite=True
smoke_finite stream=echo D=128 seed=0 arm=bptt64 n=4096 parameters=92160 bits=12.429276 fresh=13.317328 copy=11.541224 wall_seconds=1.406 finite=True
smoke_finite stream=echo D=128 seed=0 arm=trace n=4096 parameters=92160 bits=11.702442 fresh=12.385836 copy=11.019047 wall_seconds=2.440 finite=True
smoke_finite stream=echo D=512 seed=0 arm=bptt64 n=4096 parameters=92160 bits=9.752232 fresh=13.033112 copy=6.471353 wall_seconds=1.441 finite=True
smoke_finite stream=echo D=512 seed=0 arm=trace n=4096 parameters=92160 bits=10.559650 fresh=12.821472 copy=8.297828 wall_seconds=2.288 finite=True
h20_repro seed=6 n=4096 match=True
smoke_finite stream=h20_slow seed=6 arm=bptt64 n=4096 parameters=107360 bits=0.952492 c1_c7=NA c1_c7_count=0 wall_seconds=1.386 finite=True
smoke_finite stream=h20_slow seed=6 arm=trace n=4096 parameters=107360 bits=0.934869 c1_c7=NA c1_c7_count=0 wall_seconds=2.841 finite=True
time_8192 arm=bptt64 stream=h20_slow seed=6 seconds=2.924 bits=0.676713 threads=2
time_8192 arm=trace stream=h20_slow seed=6 seconds=6.392 bits=0.666092 threads=2
kernel_estimate symbols_per_arm=655360 scale=80.000000 bptt64_seconds=233.9 trace_seconds=511.4 total_seconds=745.3 limit_seconds=5400
```


## Verdict (orchestrator, 13 Nov 2026)

**Bar 1 (echo D = 128: trace copy bits ≤ 3.0, and ≥ 1.0 below bptt64 on all 3 seeds): FAIL.** Trace copies score 4.27
against bptt64's 4.41, a gap of only −0.14, and both are above uniform. The kill rule fires as written, but the
diagnosis differs from the one it named.

**Bar 2 (H20 C1 + C7: trace − bptt64 ≤ +0.10): PASS.** The gap is +0.0007 ± 0.007. The local rule loses nothing on
the counters, and on totals the two are equal within noise (0.632 vs 0.625).

**Mechanism.** The trace gradient is exact: it matches full BPTT to 9e-16. So credit reaches 128 steps back, and the
failure is not stale traces. It is memory. After learning, the mean |λ| is 0.93, a memory of about 14 steps, and
0.93^128 ≈ 1e-4. Only 12–34 of the 256 units have |λ| > 0.99, and each of those holds a superposition of about 100
random symbols that the readout cannot separate. Learning moves |λ| down from its init, because on the fresh half,
long memory is noise, and fresh symbols are half the stream. The exact gradient follows the local slope, and that
slope points away from a delay line.

**Theory link.** A pure transport ring of N units is diagonalised by the discrete Fourier transform, and its eigenvalues
are the N-th roots of unity, e^(2πik/N). A delay line *is* a diagonal bank whose modes sit on the unit circle at evenly
spaced phases. Carriage is diagonal in frequency, so this follows from our own picture. The learned bank holds the
right form, but its init (|λ| ∈ [0.9, 0.999], θ ∈ [0, π]) is far from that point, and gradient never takes it there.
H30y tests this directly: a bank initialised on the roots of unity, with λ learned or frozen.

**What H30x does establish.** The bank can be learned online, local in time, with one forward sweep, at no cost to
compression. On the substrate line this is the first learner here with no unrolling, and it is the one the north star asks for.
