# W4 headroom

Fixed policies on an 18-room ring. W14 is bits/symbol of the frozen bank on a fresh continuation of the 14 learnable rooms, from a zero state, with that room's channel embedding. Wa is the mean over instance a and Wb over instance b. Lower is better. Forgetting is end-of-life exam bits/symbol minus the exam bits/symbol just after the last visit to that room. Positive forgetting is a worse code.

Every policy sleeps for the fixed share 0.20, in visits of 8, placed as in M7b. At L = 2048 that is 408 decisions, share 0.199219. At L = 8192 that is 1640 decisions, share 0.200195.

## Oracle comparison

Breadth is oracle14_s. Depth is oracle7a_s. The lower mean W14 at L = 2048 is oracle14_s (breadth).

| policy | W14 | Wa | Wb |
|---|---:|---:|---:|
| oracle14_s | 0.897082 | 0.877516 | 0.916649 |
| oracle7a_s | 3.125105 | 0.599146 | 5.651064 |

Seeds where oracle14_s has the lower W14: 6 of 6. Seeds where oracle7a_s has the lower W14: 0 of 6.

## W14

| policy | L | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 2048 | 0.914549 | 0.896978 | 0.907788 | 0.872090 | 0.796528 | 1.044226 | 0.905360 | 0.073480 |
| oracle14_s | 2048 | 0.850834 | 0.976609 | 0.689678 | 1.021504 | 0.890195 | 0.953675 | 0.897082 | 0.108175 |
| oracle7a_s | 2048 | 3.025500 | 3.106380 | 3.318303 | 3.134773 | 3.091033 | 3.074642 | 3.125105 | 0.092555 |
| oracle14_s | 8192 | 0.448728 | 0.447008 | 0.400537 | 0.479213 | 0.548423 | 0.404245 | 0.454692 | 0.049932 |

## Wa

| policy | L | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 2048 | 0.805527 | 0.954908 | 0.783893 | 0.931166 | 0.784588 | 0.998673 | 0.876459 | 0.087677 |
| oracle14_s | 2048 | 0.862659 | 1.009381 | 0.733217 | 0.818240 | 0.792695 | 1.048905 | 0.877516 | 0.114392 |
| oracle7a_s | 2048 | 0.586649 | 0.537451 | 0.638139 | 0.601994 | 0.652565 | 0.578080 | 0.599146 | 0.038272 |
| oracle14_s | 8192 | 0.488231 | 0.399602 | 0.406547 | 0.423875 | 0.488724 | 0.365450 | 0.428738 | 0.045658 |

## Wb

| policy | L | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | std ddof=0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 2048 | 1.023571 | 0.839048 | 1.031682 | 0.813014 | 0.808468 | 1.089779 | 0.934260 | 0.116364 |
| oracle14_s | 2048 | 0.839009 | 0.943838 | 0.646139 | 1.224768 | 0.987694 | 0.858444 | 0.916649 | 0.174793 |
| oracle7a_s | 2048 | 5.464350 | 5.675309 | 5.998467 | 5.667552 | 5.529501 | 5.571204 | 5.651064 | 0.172030 |
| oracle14_s | 8192 | 0.409226 | 0.494413 | 0.394527 | 0.534552 | 0.608122 | 0.443041 | 0.480647 | 0.074473 |

## Gaps

On W14. A negative gap means the first policy is lower. The L = 2,048 gap uses both policies at L = 2,048. The scarcity gap is oracle14_s at L = 2,048 minus oracle14_s at L = 8,192.

| seed | oracle14_s − rand_dwell_s | oracle7a_s − rand_dwell_s | oracle14_s L2048 − L8192 |
|---:|---:|---:|---:|
| 0 | -0.063715 | 2.110951 | 0.402106 |
| 1 | 0.079632 | 2.209403 | 0.529602 |
| 2 | -0.218110 | 2.410515 | 0.289141 |
| 3 | 0.149414 | 2.262683 | 0.542291 |
| 4 | 0.093666 | 2.294505 | 0.341772 |
| 5 | -0.090551 | 2.030416 | 0.549429 |
| mean | -0.008277 | 2.219745 | 0.442390 |
| std ddof=0 | 0.127056 | 0.123568 | 0.103502 |
| left lower (count) | 3 of 6 | 0 of 6 | 0 of 6 |

## Per-room held-out bits

| policy | L | seed | Ga | C1a | C7a | E30a | E110a | M7a | M37a | Gb | C1b | C7b | E30b | E110b | M7b | M37b | W14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 2048 | 0 | 2.919810 | 0.100316 | 0.139393 | 0.515902 | 0.692493 | 0.825851 | 0.444924 | 4.303330 | 0.135803 | 0.123057 | 0.501595 | 0.680325 | 0.955683 | 0.465202 | 0.914549 |
| rand_dwell_s | 2048 | 1 | 3.995668 | 0.178962 | 0.332050 | 0.902037 | 0.213991 | 0.404723 | 0.656921 | 2.521729 | 0.082824 | 0.165057 | 0.727272 | 0.413090 | 0.811413 | 1.151949 | 0.896978 |
| rand_dwell_s | 2048 | 2 | 2.711255 | 0.071462 | 0.209556 | 0.843891 | 1.063773 | 0.316049 | 0.271265 | 2.056506 | 0.105986 | 0.188068 | 0.933078 | 0.977001 | 0.364728 | 2.596408 | 0.907788 |
| rand_dwell_s | 2048 | 3 | 2.992974 | 0.154315 | 0.319544 | 0.717998 | 0.403086 | 1.646876 | 0.283369 | 3.215545 | 0.088352 | 0.379008 | 0.590333 | 0.515896 | 0.316376 | 0.585588 | 0.872090 |
| rand_dwell_s | 2048 | 4 | 2.404964 | 0.141835 | 0.134939 | 0.701365 | 0.839937 | 0.784859 | 0.484220 | 2.995828 | 0.132259 | 0.120487 | 0.726166 | 0.828221 | 0.241745 | 0.614573 | 0.796528 |
| rand_dwell_s | 2048 | 5 | 4.058748 | 0.173807 | 0.212980 | 1.232102 | 0.482718 | 0.416469 | 0.413887 | 3.169643 | 0.137547 | 0.284185 | 1.208730 | 0.404332 | 0.275711 | 2.148304 | 1.044226 |
| oracle14_s | 2048 | 0 | 3.180267 | 0.085413 | 0.097504 | 0.824984 | 1.033536 | 0.323726 | 0.493186 | 2.754053 | 0.143114 | 0.148785 | 0.858259 | 1.190847 | 0.368386 | 0.409622 | 0.850834 |
| oracle14_s | 2048 | 1 | 3.254473 | 0.322045 | 0.235156 | 1.291068 | 0.744212 | 0.415514 | 0.803196 | 3.294179 | 0.347205 | 0.184377 | 1.262457 | 0.748253 | 0.303939 | 0.466459 | 0.976609 |
| oracle14_s | 2048 | 2 | 3.034934 | 0.072832 | 0.122383 | 0.730706 | 0.320798 | 0.298454 | 0.552410 | 2.182206 | 0.141906 | 0.103888 | 0.751931 | 0.450404 | 0.343097 | 0.549541 | 0.689678 |
| oracle14_s | 2048 | 3 | 3.226519 | 0.024297 | 0.061430 | 0.845300 | 0.338623 | 0.325311 | 0.906196 | 4.121191 | 0.068719 | 0.069787 | 0.821451 | 0.635064 | 0.413515 | 2.443650 | 1.021504 |
| oracle14_s | 2048 | 4 | 2.747880 | 0.067809 | 0.086020 | 0.873892 | 0.856697 | 0.558728 | 0.357842 | 3.416500 | 0.110163 | 0.053778 | 0.850817 | 0.858144 | 0.323020 | 1.301437 | 0.890195 |
| oracle14_s | 2048 | 5 | 3.289985 | 0.080688 | 0.119654 | 0.959388 | 0.582666 | 1.264055 | 1.045899 | 2.648048 | 0.133333 | 0.106775 | 1.147282 | 0.404789 | 0.396826 | 1.172054 | 0.953675 |
| oracle7a_s | 2048 | 0 | 1.715771 | 0.087902 | 0.085951 | 0.624239 | 0.566916 | 0.286108 | 0.739658 | 12.429256 | 0.057131 | 0.066578 | 0.559529 | 1.042117 | 10.029563 | 14.066276 | 3.025500 |
| oracle7a_s | 2048 | 1 | 1.659146 | 0.068277 | 0.117567 | 0.742009 | 0.393434 | 0.403212 | 0.378515 | 12.068187 | 0.044363 | 0.168317 | 0.786053 | 0.345186 | 12.359563 | 13.955496 | 3.106380 |
| oracle7a_s | 2048 | 2 | 1.663517 | 0.811610 | 0.117647 | 0.692802 | 0.354705 | 0.382881 | 0.443808 | 14.867384 | 0.465627 | 0.269525 | 0.722001 | 0.448614 | 9.553095 | 15.663024 | 3.318303 |
| oracle7a_s | 2048 | 3 | 1.529471 | 0.108667 | 0.106791 | 0.680011 | 0.699903 | 0.620420 | 0.468694 | 12.673615 | 0.264946 | 0.209748 | 1.024786 | 0.464329 | 11.271795 | 13.763643 | 3.134773 |
| oracle7a_s | 2048 | 4 | 1.913586 | 0.218295 | 0.186917 | 0.317281 | 1.210814 | 0.399733 | 0.321332 | 13.124774 | 0.184061 | 0.175445 | 0.392597 | 1.634254 | 10.532503 | 12.662874 | 3.091033 |
| oracle7a_s | 2048 | 5 | 1.626895 | 0.270727 | 0.098120 | 0.555266 | 0.515464 | 0.289742 | 0.690346 | 13.272690 | 0.187682 | 0.121724 | 0.677336 | 0.519866 | 11.104141 | 13.114992 | 3.074642 |
| oracle14_s | 8192 | 0 | 1.332069 | 0.044692 | 0.059330 | 0.437793 | 0.249596 | 0.409959 | 0.884179 | 1.262360 | 0.028294 | 0.053548 | 0.488923 | 0.259594 | 0.508078 | 0.263781 | 0.448728 |
| oracle14_s | 8192 | 1 | 1.307931 | 0.019577 | 0.026797 | 0.449833 | 0.333676 | 0.294947 | 0.364454 | 1.833481 | 0.020419 | 0.030452 | 0.380004 | 0.332871 | 0.291077 | 0.572590 | 0.447008 |
| oracle14_s | 8192 | 2 | 1.330579 | 0.014073 | 0.046570 | 0.290900 | 0.413246 | 0.283545 | 0.466915 | 1.270884 | 0.018335 | 0.079727 | 0.329116 | 0.352217 | 0.351889 | 0.359519 | 0.400537 |
| oracle14_s | 8192 | 3 | 1.327324 | 0.026988 | 0.050039 | 0.265298 | 0.458981 | 0.220435 | 0.618058 | 1.607869 | 0.026130 | 0.072131 | 0.285765 | 0.565229 | 0.373274 | 0.811462 | 0.479213 |
| oracle14_s | 8192 | 4 | 1.285398 | 0.017224 | 0.054867 | 0.246656 | 1.215390 | 0.254201 | 0.347328 | 1.467479 | 0.039218 | 0.064528 | 0.245434 | 1.014706 | 0.552370 | 0.873119 | 0.548423 |
| oracle14_s | 8192 | 5 | 1.297701 | 0.069851 | 0.066781 | 0.238137 | 0.320011 | 0.270394 | 0.295273 | 1.714617 | 0.060909 | 0.051021 | 0.231884 | 0.253780 | 0.304956 | 0.484119 | 0.404245 |

## Forgetting

Exam-store bits/symbol just after the last visit that dwelled in the room, subtracted from the end-of-life bits/symbol. A room with no exam glance at that visit is left out. n is the number of seed-room pairs in the mean.

| policy | L | mean delta bps | std ddof=0 | n |
|---|---:|---:|---:|---:|
| rand_dwell_s | 2048 | 0.212649 | 0.345963 | 84 |
| oracle14_s | 2048 | 0.194578 | 0.294331 | 84 |
| oracle7a_s | 2048 | 0.084065 | 0.108008 | 42 |
| oracle14_s | 8192 | 0.064422 | 0.114237 | 84 |

| policy | L | room | n | after bps | end bps | delta bps | std ddof=0 |
|---|---:|---|---:|---:|---:|---:|---:|
| rand_dwell_s | 2048 | Ga | 6 | 2.023752 | 2.538966 | 0.515213 | 0.509903 |
| rand_dwell_s | 2048 | C1a | 6 | 0.379995 | 0.464992 | 0.084997 | 0.081747 |
| rand_dwell_s | 2048 | C7a | 6 | 0.514639 | 0.612492 | 0.097853 | 0.052306 |
| rand_dwell_s | 2048 | E30a | 6 | 0.896818 | 1.128883 | 0.232064 | 0.292683 |
| rand_dwell_s | 2048 | E110a | 6 | 0.908146 | 1.043729 | 0.135583 | 0.082139 |
| rand_dwell_s | 2048 | M7a | 6 | 0.343718 | 0.719181 | 0.375463 | 0.472466 |
| rand_dwell_s | 2048 | M37a | 6 | 0.583153 | 0.712728 | 0.129574 | 0.155103 |
| rand_dwell_s | 2048 | Gb | 6 | 2.276021 | 2.358395 | 0.082374 | 0.520231 |
| rand_dwell_s | 2048 | C1b | 6 | 0.389225 | 0.479698 | 0.090472 | 0.037750 |
| rand_dwell_s | 2048 | C7b | 6 | 0.549323 | 0.634673 | 0.085350 | 0.099628 |
| rand_dwell_s | 2048 | E30b | 6 | 0.894082 | 1.112581 | 0.218499 | 0.260424 |
| rand_dwell_s | 2048 | E110b | 6 | 0.961007 | 1.029196 | 0.068189 | 0.105435 |
| rand_dwell_s | 2048 | M7b | 6 | 0.356940 | 0.556740 | 0.199801 | 0.187880 |
| rand_dwell_s | 2048 | M37b | 6 | 0.527763 | 1.189413 | 0.661651 | 0.488683 |
| oracle14_s | 2048 | Ga | 6 | 1.734309 | 2.222500 | 0.488192 | 0.277558 |
| oracle14_s | 2048 | C1a | 6 | 0.451792 | 0.492494 | 0.040702 | 0.080724 |
| oracle14_s | 2048 | C7a | 6 | 0.590450 | 0.623946 | 0.033495 | 0.062739 |
| oracle14_s | 2048 | E30a | 6 | 0.893687 | 1.150370 | 0.256683 | 0.426267 |
| oracle14_s | 2048 | E110a | 6 | 0.789602 | 0.975313 | 0.185711 | 0.237806 |
| oracle14_s | 2048 | M7a | 6 | 0.355811 | 0.532630 | 0.176819 | 0.258648 |
| oracle14_s | 2048 | M37a | 6 | 0.611674 | 0.855674 | 0.244000 | 0.265239 |
| oracle14_s | 2048 | Gb | 6 | 1.757452 | 1.996384 | 0.238932 | 0.169797 |
| oracle14_s | 2048 | C1b | 6 | 0.410319 | 0.449174 | 0.038855 | 0.084233 |
| oracle14_s | 2048 | C7b | 6 | 0.582462 | 0.616676 | 0.034214 | 0.067591 |
| oracle14_s | 2048 | E30b | 6 | 0.889501 | 1.187441 | 0.297939 | 0.341990 |
| oracle14_s | 2048 | E110b | 6 | 0.845821 | 1.018085 | 0.172264 | 0.140060 |
| oracle14_s | 2048 | M7b | 6 | 0.363067 | 0.388139 | 0.025073 | 0.027232 |
| oracle14_s | 2048 | M37b | 6 | 0.522164 | 1.013382 | 0.491218 | 0.499653 |
| oracle7a_s | 2048 | Ga | 6 | 1.071994 | 1.190480 | 0.118487 | 0.098857 |
| oracle7a_s | 2048 | C1a | 6 | 0.486253 | 0.522644 | 0.036390 | 0.028122 |
| oracle7a_s | 2048 | C7a | 6 | 0.646872 | 0.695321 | 0.048448 | 0.065828 |
| oracle7a_s | 2048 | E30a | 6 | 0.844041 | 0.940459 | 0.096418 | 0.162738 |
| oracle7a_s | 2048 | E110a | 6 | 0.855453 | 0.947001 | 0.091549 | 0.107866 |
| oracle7a_s | 2048 | M7a | 6 | 0.298182 | 0.411760 | 0.113578 | 0.098159 |
| oracle7a_s | 2048 | M37a | 6 | 0.509024 | 0.592612 | 0.083588 | 0.114903 |
| oracle7a_s | 2048 | Gb | 0 | — | — | — | — |
| oracle7a_s | 2048 | C1b | 0 | — | — | — | — |
| oracle7a_s | 2048 | C7b | 0 | — | — | — | — |
| oracle7a_s | 2048 | E30b | 0 | — | — | — | — |
| oracle7a_s | 2048 | E110b | 0 | — | — | — | — |
| oracle7a_s | 2048 | M7b | 0 | — | — | — | — |
| oracle7a_s | 2048 | M37b | 0 | — | — | — | — |
| oracle14_s | 8192 | Ga | 6 | 1.087916 | 1.204342 | 0.116426 | 0.071821 |
| oracle14_s | 8192 | C1a | 6 | 0.410413 | 0.418138 | 0.007725 | 0.018712 |
| oracle14_s | 8192 | C7a | 6 | 0.454117 | 0.482137 | 0.028020 | 0.035638 |
| oracle14_s | 8192 | E30a | 6 | 0.737758 | 0.731592 | -0.006165 | 0.067690 |
| oracle14_s | 8192 | E110a | 6 | 0.747416 | 0.818731 | 0.071316 | 0.118317 |
| oracle14_s | 8192 | M7a | 6 | 0.269009 | 0.310704 | 0.041695 | 0.030697 |
| oracle14_s | 8192 | M37a | 6 | 0.423004 | 0.561543 | 0.138540 | 0.174701 |
| oracle14_s | 8192 | Gb | 6 | 1.187484 | 1.364990 | 0.177505 | 0.118445 |
| oracle14_s | 8192 | C1b | 6 | 0.419517 | 0.419225 | -0.000291 | 0.030911 |
| oracle14_s | 8192 | C7b | 6 | 0.466479 | 0.504348 | 0.037869 | 0.038967 |
| oracle14_s | 8192 | E30b | 6 | 0.675506 | 0.707377 | 0.031871 | 0.048558 |
| oracle14_s | 8192 | E110b | 6 | 0.802073 | 0.804869 | 0.002796 | 0.145846 |
| oracle14_s | 8192 | M7b | 6 | 0.282657 | 0.360958 | 0.078301 | 0.084414 |
| oracle14_s | 8192 | M37b | 6 | 0.394134 | 0.570430 | 0.176296 | 0.158401 |

## Decision shares

Mean over seeds. A stay counts for the room. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.

| policy | L | Ga | C1a | C7a | E30a | E110a | M7a | M37a | Gb | C1b | C7b | E30b | E110b | M7b | M37b | N1 | N2 | N3 | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 2048 | 0.0273 | 0.0280 | 0.0293 | 0.0339 | 0.0299 | 0.0319 | 0.0273 | 0.0286 | 0.0260 | 0.0254 | 0.0312 | 0.0299 | 0.0260 | 0.0208 | 0.0333 | 0.0282 | 0.0273 | 0.0280 | 0.2881 | 0.1992 |
| oracle14_s | 2048 | 0.0365 | 0.0306 | 0.0435 | 0.0360 | 0.0442 | 0.0397 | 0.0397 | 0.0326 | 0.0378 | 0.0391 | 0.0332 | 0.0319 | 0.0374 | 0.0339 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2848 | 0.1992 |
| oracle7a_s | 2048 | 0.0856 | 0.0840 | 0.0651 | 0.0828 | 0.0672 | 0.0775 | 0.0662 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2724 | 0.1992 |
| oracle14_s | 8192 | 0.0331 | 0.0359 | 0.0393 | 0.0410 | 0.0409 | 0.0356 | 0.0374 | 0.0357 | 0.0397 | 0.0400 | 0.0340 | 0.0365 | 0.0347 | 0.0326 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2834 | 0.2002 |

Per seed.

| policy | L | seed | Ga | C1a | C7a | E30a | E110a | M7a | M37a | Gb | C1b | C7b | E30b | E110b | M7b | M37b | N1 | N2 | N3 | Z | corridor | sleep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 2048 | 0 | 0.0312 | 0.0352 | 0.0352 | 0.0352 | 0.0195 | 0.0508 | 0.0156 | 0.0078 | 0.0117 | 0.0312 | 0.0273 | 0.0352 | 0.0234 | 0.0234 | 0.0469 | 0.0273 | 0.0312 | 0.0312 | 0.2812 | 0.1992 |
| rand_dwell_s | 2048 | 1 | 0.0273 | 0.0156 | 0.0273 | 0.0117 | 0.0508 | 0.0195 | 0.0586 | 0.0312 | 0.0312 | 0.0195 | 0.0469 | 0.0195 | 0.0312 | 0.0352 | 0.0469 | 0.0078 | 0.0078 | 0.0195 | 0.2930 | 0.1992 |
| rand_dwell_s | 2048 | 2 | 0.0273 | 0.0234 | 0.0234 | 0.0273 | 0.0117 | 0.0430 | 0.0156 | 0.0547 | 0.0234 | 0.0312 | 0.0273 | 0.0273 | 0.0195 | 0.0195 | 0.0352 | 0.0288 | 0.0352 | 0.0312 | 0.2954 | 0.1992 |
| rand_dwell_s | 2048 | 3 | 0.0156 | 0.0234 | 0.0195 | 0.0352 | 0.0391 | 0.0195 | 0.0352 | 0.0312 | 0.0312 | 0.0195 | 0.0352 | 0.0391 | 0.0352 | 0.0234 | 0.0122 | 0.0469 | 0.0273 | 0.0273 | 0.2847 | 0.1992 |
| rand_dwell_s | 2048 | 4 | 0.0391 | 0.0391 | 0.0469 | 0.0469 | 0.0156 | 0.0430 | 0.0195 | 0.0234 | 0.0234 | 0.0195 | 0.0234 | 0.0195 | 0.0273 | 0.0156 | 0.0273 | 0.0195 | 0.0430 | 0.0273 | 0.2812 | 0.1992 |
| rand_dwell_s | 2048 | 5 | 0.0234 | 0.0312 | 0.0234 | 0.0469 | 0.0430 | 0.0156 | 0.0195 | 0.0234 | 0.0352 | 0.0312 | 0.0273 | 0.0391 | 0.0195 | 0.0078 | 0.0312 | 0.0391 | 0.0195 | 0.0312 | 0.2930 | 0.1992 |
| oracle14_s | 2048 | 0 | 0.0352 | 0.0391 | 0.0430 | 0.0244 | 0.0312 | 0.0703 | 0.0391 | 0.0273 | 0.0430 | 0.0234 | 0.0234 | 0.0273 | 0.0352 | 0.0586 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2803 | 0.1992 |
| oracle14_s | 2048 | 1 | 0.0352 | 0.0234 | 0.0391 | 0.0469 | 0.0703 | 0.0273 | 0.0781 | 0.0273 | 0.0156 | 0.0273 | 0.0352 | 0.0273 | 0.0215 | 0.0352 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2910 | 0.1992 |
| oracle14_s | 2048 | 2 | 0.0352 | 0.0352 | 0.0352 | 0.0234 | 0.0298 | 0.0430 | 0.0273 | 0.0508 | 0.0312 | 0.0469 | 0.0430 | 0.0391 | 0.0430 | 0.0312 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2866 | 0.1992 |
| oracle14_s | 2048 | 3 | 0.0312 | 0.0352 | 0.0352 | 0.0352 | 0.0400 | 0.0352 | 0.0508 | 0.0234 | 0.0508 | 0.0625 | 0.0234 | 0.0312 | 0.0430 | 0.0234 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2803 | 0.1992 |
| oracle14_s | 2048 | 4 | 0.0391 | 0.0391 | 0.0703 | 0.0430 | 0.0352 | 0.0508 | 0.0195 | 0.0312 | 0.0352 | 0.0352 | 0.0430 | 0.0156 | 0.0430 | 0.0195 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2812 | 0.1992 |
| oracle14_s | 2048 | 5 | 0.0430 | 0.0117 | 0.0386 | 0.0430 | 0.0586 | 0.0117 | 0.0234 | 0.0352 | 0.0508 | 0.0391 | 0.0312 | 0.0508 | 0.0391 | 0.0352 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2896 | 0.1992 |
| oracle7a_s | 2048 | 0 | 0.0684 | 0.0859 | 0.0938 | 0.0703 | 0.0781 | 0.1094 | 0.0625 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2324 | 0.1992 |
| oracle7a_s | 2048 | 1 | 0.0898 | 0.1016 | 0.0547 | 0.0703 | 0.0674 | 0.0664 | 0.0703 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2803 | 0.1992 |
| oracle7a_s | 2048 | 2 | 0.0977 | 0.0781 | 0.0586 | 0.0708 | 0.0625 | 0.0859 | 0.0625 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2847 | 0.1992 |
| oracle7a_s | 2048 | 3 | 0.0859 | 0.0820 | 0.0586 | 0.0781 | 0.0781 | 0.0547 | 0.0859 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2773 | 0.1992 |
| oracle7a_s | 2048 | 4 | 0.0820 | 0.0781 | 0.0625 | 0.1133 | 0.0703 | 0.0664 | 0.0537 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2744 | 0.1992 |
| oracle7a_s | 2048 | 5 | 0.0898 | 0.0781 | 0.0625 | 0.0938 | 0.0469 | 0.0820 | 0.0625 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2852 | 0.1992 |
| oracle14_s | 8192 | 0 | 0.0312 | 0.0430 | 0.0352 | 0.0352 | 0.0315 | 0.0430 | 0.0479 | 0.0371 | 0.0332 | 0.0342 | 0.0361 | 0.0342 | 0.0312 | 0.0381 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2888 | 0.2002 |
| oracle14_s | 8192 | 1 | 0.0303 | 0.0332 | 0.0439 | 0.0381 | 0.0547 | 0.0273 | 0.0430 | 0.0256 | 0.0371 | 0.0400 | 0.0322 | 0.0361 | 0.0332 | 0.0400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2849 | 0.2002 |
| oracle14_s | 8192 | 2 | 0.0283 | 0.0322 | 0.0469 | 0.0468 | 0.0361 | 0.0322 | 0.0322 | 0.0439 | 0.0391 | 0.0439 | 0.0322 | 0.0361 | 0.0332 | 0.0352 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2814 | 0.2002 |
| oracle14_s | 8192 | 3 | 0.0333 | 0.0400 | 0.0332 | 0.0322 | 0.0420 | 0.0439 | 0.0439 | 0.0371 | 0.0459 | 0.0420 | 0.0283 | 0.0391 | 0.0332 | 0.0273 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2782 | 0.2002 |
| oracle14_s | 8192 | 4 | 0.0410 | 0.0349 | 0.0381 | 0.0430 | 0.0361 | 0.0410 | 0.0273 | 0.0381 | 0.0371 | 0.0371 | 0.0420 | 0.0312 | 0.0469 | 0.0254 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2805 | 0.2002 |
| oracle14_s | 8192 | 5 | 0.0342 | 0.0322 | 0.0387 | 0.0508 | 0.0449 | 0.0264 | 0.0303 | 0.0322 | 0.0459 | 0.0430 | 0.0332 | 0.0420 | 0.0303 | 0.0293 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2865 | 0.2002 |

## Ring order

Room name in ring positions 0..17. One order per seed. Policies share it.

| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | r9 | r10 | r11 | r12 | r13 | r14 | r15 | r16 | r17 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | Ga | Gb | N1 | M7a | E110a | C7a | E30a | C1b | C1a | N2 | E30b | M37a | C7b | M37b | E110b | Z | N3 | M7b |
| 1 | C1a | M37a | M37b | Ga | C1b | N2 | E110a | E30b | C7b | M7a | Z | C7a | Gb | E110b | M7b | N3 | E30a | N1 |
| 2 | N3 | E30b | M7a | M7b | Z | C1a | Gb | C7b | Ga | M37a | C7a | E30a | C1b | E110b | M37b | N2 | N1 | E110a |
| 3 | Z | C7a | E30b | M37a | C1a | C1b | E110b | N1 | E30a | M7b | E110a | M7a | Gb | Ga | M37b | C7b | N3 | N2 |
| 4 | C1a | C7b | M37a | Gb | C1b | Z | E110a | Ga | M7b | N1 | N3 | E30b | M37b | C7a | E30a | E110b | N2 | M7a |
| 5 | M37a | C7a | C1b | C1a | M37b | N3 | E30a | E110a | Z | E30b | Gb | M7a | M7b | Ga | C7b | E110b | N1 | N2 |

## M7c reference

M7c-W3, seeds 42–53, the same bank and the same fixed sleep share, on W3's 9 rooms. Source: results/m7cw3/results.json.

| L | controller | W mean | std ddof=0 |
|---:|---|---:|---:|
| 2048 | rand_dwell_s | 0.663905 | 0.089948 |
| 2048 | oracle7_s | 0.635424 | 0.119076 |
| 4096 | rand_dwell_s | 0.475578 | 0.075499 |
| 4096 | oracle7_s | 0.455629 | 0.042635 |

| L | oracle7_s − rand_dwell_s mean | std ddof=0 | left lower |
|---:|---:|---:|---:|
| 2048 | -0.028481 | 0.158523 | 9 of 12 |
| 4096 | -0.019949 | 0.095110 | 6 of 12 |

## Walls and kernels

| kernel | seed | lives | wall seconds | workers |
|---|---:|---|---:|---:|
| ezzzzzx/w4-s0 | 0 | 2048,8192 | 790.572 | 4 |
| ezzzzzx/w4-s1 | 1 | 2048,8192 | 863.983 | 4 |
| ezzzzzx/w4-s2 | 2 | 2048,8192 | 745.331 | 4 |
| ezzzzzx/w4-s3 | 3 | 2048,8192 | 725.491 | 4 |
| ezzzzzx/w4-s4 | 4 | 2048,8192 | 882.513 | 4 |
| ezzzzzx/w4-s5 | 5 | 2048,8192 | 777.492 | 4 |

| policy | L | seed | wall seconds | learn seconds | score seconds | eval seconds |
|---|---:|---:|---:|---:|---:|---:|
| rand_dwell_s | 2048 | 0 | 217.843 | 110.681 | 96.296 | 0.841 |
| rand_dwell_s | 2048 | 1 | 231.177 | 124.362 | 94.531 | 1.534 |
| rand_dwell_s | 2048 | 2 | 205.822 | 106.009 | 89.063 | 0.831 |
| rand_dwell_s | 2048 | 3 | 201.208 | 107.456 | 84.114 | 0.804 |
| rand_dwell_s | 2048 | 4 | 227.023 | 111.811 | 103.691 | 1.496 |
| rand_dwell_s | 2048 | 5 | 215.529 | 111.644 | 92.768 | 1.456 |
| oracle14_s | 2048 | 0 | 215.384 | 116.136 | 88.287 | 0.896 |
| oracle14_s | 2048 | 1 | 232.674 | 130.585 | 90.302 | 0.952 |
| oracle14_s | 2048 | 2 | 212.739 | 113.694 | 88.305 | 0.798 |
| oracle14_s | 2048 | 3 | 199.688 | 111.155 | 78.849 | 0.850 |
| oracle14_s | 2048 | 4 | 227.983 | 118.989 | 98.004 | 0.915 |
| oracle14_s | 2048 | 5 | 217.739 | 118.900 | 88.235 | 0.864 |
| oracle7a_s | 2048 | 0 | 198.878 | 110.146 | 77.339 | 1.320 |
| oracle7a_s | 2048 | 1 | 208.650 | 120.458 | 75.972 | 1.317 |
| oracle7a_s | 2048 | 2 | 191.856 | 110.695 | 69.963 | 1.244 |
| oracle7a_s | 2048 | 3 | 180.169 | 102.684 | 67.365 | 1.250 |
| oracle7a_s | 2048 | 4 | 199.776 | 108.191 | 80.153 | 1.355 |
| oracle7a_s | 2048 | 5 | 197.442 | 116.631 | 69.800 | 1.277 |
| oracle14_s | 8192 | 0 | 787.227 | 392.888 | 382.196 | 0.859 |
| oracle14_s | 8192 | 1 | 860.694 | 439.057 | 408.435 | 1.004 |
| oracle14_s | 8192 | 2 | 742.324 | 370.718 | 359.729 | 0.806 |
| oracle14_s | 8192 | 3 | 722.641 | 374.300 | 337.499 | 0.889 |
| oracle14_s | 8192 | 4 | 879.426 | 413.364 | 453.957 | 0.926 |
| oracle14_s | 8192 | 5 | 774.368 | 397.743 | 364.881 | 0.867 |

```
2026-11-14T09:10:40+08:00 ezzzzzx/w4-s0 NONE -> RUNNING
2026-11-14T09:10:42+08:00 ezzzzzx/w4-s1 NONE -> RUNNING
2026-11-14T09:10:45+08:00 ezzzzzx/w4-s2 NONE -> RUNNING
2026-11-14T09:10:47+08:00 ezzzzzx/w4-s3 NONE -> RUNNING
2026-11-14T09:20:34+08:00 ezzzzzx/w4-s3 RUNNING -> COMPLETE
2026-11-14T09:20:38+08:00 downloaded ezzzzzx/w4-s3
2026-11-14T09:20:41+08:00 pushed ezzzzzx/w4-s4
2026-11-14T09:21:28+08:00 ezzzzzx/w4-s0 RUNNING -> COMPLETE
2026-11-14T09:21:32+08:00 downloaded ezzzzzx/w4-s0
2026-11-14T09:21:36+08:00 ezzzzzx/w4-s2 RUNNING -> COMPLETE
2026-11-14T09:21:39+08:00 downloaded ezzzzzx/w4-s2
2026-11-14T09:21:41+08:00 ezzzzzx/w4-s4 NONE -> RUNNING
2026-11-14T09:21:45+08:00 pushed ezzzzzx/w4-s5
2026-11-14T09:22:36+08:00 ezzzzzx/w4-s5 NONE -> RUNNING
2026-11-14T09:23:24+08:00 ezzzzzx/w4-s1 RUNNING -> COMPLETE
2026-11-14T09:23:26+08:00 downloaded ezzzzzx/w4-s1
2026-11-14T09:35:48+08:00 ezzzzzx/w4-s4 RUNNING -> COMPLETE
2026-11-14T09:35:51+08:00 downloaded ezzzzzx/w4-s4
2026-11-14T09:35:53+08:00 ezzzzzx/w4-s5 RUNNING -> COMPLETE
2026-11-14T09:35:56+08:00 downloaded ezzzzzx/w4-s5
```

## Smoke test

```
torch=2.15.0+cpu threads=1 model_seed=0 n_ch=19
dose L64 16/64 share 0.250000
dose L512 104/512 share 0.203125
dose L2048 408/2048 share 0.199219
dose L8192 1640/8192 share 0.200195
sleep_due on a 9-ring matches m7.sleep_due
ring shortest paths, including the distance-9 tie, go left on a tie
smoke check 1: 19-row bank matches M7 on 64 glances, channel ids < 10
  bit abs 0.000e+00 param abs 0.000e+00 channel-row abs 0.000e+00 limit 1.0e-06
smoke check 2: instances differ, seed 0
  M7a 'xbynjroxbynjroxbynjroxbynjroxbykjroxbynjroxbynjroxbygjroxbynjrox'
  M7b 'paflootpaflootpaflootpaflootpaflootpaflootpaflootpaflootpaflootp'
  Ga 'qnhib ad swecw rskwd qnhib gqtj swecw. ae woa that zuz ae gqtj q'
  Gb 'nhem jtidy gueq sd vmff gte. vmff awh awxu vxdv. nhem jtidy gte '
  streams resume, noise rngs differ, a move shows the corridor
ring orders
  seed 0 perm [0, 7, 14, 5, 4, 2, 3, 8, 1, 15, 10, 6, 9, 13, 11, 17, 16, 12] rooms ['Ga', 'Gb', 'N1', 'M7a', 'E110a', 'C7a', 'E30a', 'C1b', 'C1a', 'N2', 'E30b', 'M37a', 'C7b', 'M37b', 'E110b', 'Z', 'N3', 'M7b']
  seed 1 perm [1, 6, 13, 0, 8, 15, 4, 10, 9, 5, 17, 2, 7, 11, 12, 16, 3, 14] rooms ['C1a', 'M37a', 'M37b', 'Ga', 'C1b', 'N2', 'E110a', 'E30b', 'C7b', 'M7a', 'Z', 'C7a', 'Gb', 'E110b', 'M7b', 'N3', 'E30a', 'N1']
  seed 2 perm [16, 10, 5, 12, 17, 1, 7, 9, 0, 6, 2, 3, 8, 11, 13, 15, 14, 4] rooms ['N3', 'E30b', 'M7a', 'M7b', 'Z', 'C1a', 'Gb', 'C7b', 'Ga', 'M37a', 'C7a', 'E30a', 'C1b', 'E110b', 'M37b', 'N2', 'N1', 'E110a']
smoke check 3: W4 restricted to W3's 9 rooms, rand_dwell_s seed 36 L=64
  m7b W 13.65447919 W9 12.78679691 live W 13.65447919 W9 12.78679691 abs_dW 0.000e+00 abs_dW9 0.000e+00 limit 1.0e-06
smoke check 4: three policies, seed 0, L=64, finite W14
  rand_dwell_s W14 14.116672 Wa 13.289834 Wb 14.943510 finite True sleep 0.2500 decisions 16 corridor 0.3125 wall 5.20s
  oracle14_s W14 11.027399 Wa 11.346558 Wb 10.708240 finite True sleep 0.2500 decisions 16 corridor 0.2500 wall 5.46s
  oracle7a_s W14 10.846188 Wa 9.716936 Wb 11.975440 finite True sleep 0.2500 decisions 16 corridor 0.1875 wall 5.49s
smoke check 5: time rand_dwell_s seed 0 L=512
  upper bound within 90 min; one kernel per seed runs every life
  rand_dwell_s L=512 W14 2.792055 wall 39.93s learn 27.07s score 10.95s eval 1.74s sleep 104
  learn_per_decision 0.0529s full_rescore_576 0.4777s
  linear L2048 153.8s L8192 610.0s pessimistic L2048 304.9s L8192 1218.4s guarded L8192 1218.3s kernel 1218.4s limit 5400s split False
```

## Protocol

- 18 rooms on a ring. Identity order is the 7 H20 processes with suffix a, then suffix b, then N1, N2, N3, then Z. Ring order is `default_rng(world_seed + 70000).permutation(18)`.
- Instance a is `h20_world.World(world_seed)`. Instance b is `h20_world.World(world_seed + 1000)`. Glances are `curio.glance` on that instance. Noise room Nk uses `default_rng(world_seed + 60000 + k)` with k = 0, 1, 2. Z is 64 spaces. A room's stream resumes.
- Actions stay, left, right. A move shows 64 space symbols labelled corridor and does not observe the rooms it passes. The agent starts at ring position 0.
- Shortest path uses the same comparison as `w3_headroom.shortest`. On a tie, which an 18-ring allows at distance 9, the walk goes left.
- The cortex is `m7_w3.BankCortex`: roots init, trace learning, Adam lr 1e-3, model seed 0. The channel embedding has 19 rows (18 rooms plus the corridor), initialised at zero. Corridor is row 18. Room rows follow the identity order.
- Stores, coin and replacement seeds are M7's. Exam coin `default_rng(40000 + seed)`, exam replacement `default_rng(20000 + seed)`, replay replacement `default_rng(50000 + seed)`, sleep draw `default_rng(30000 + seed)`. Reservoirs hold 32 glances per room. A sleep glance is learned from a zero state and does not write the live stream.
- The dose is 8 times the integer nearest to 0.20 L / 8. Placement is `m7_w3.sleep_plan` and the same due rule, with the longest visit set to half the ring plus 8. On a 9-ring that longest visit is 12, matching `m7_w3.sleep_due`. A sleep visit starts only when the current plan is empty.
- Policy rng is `default_rng(1000 + world_seed)`. rand_dwell_s targets the 18 rooms. oracle14_s targets the 14 learnable rooms. oracle7a_s targets the 7 rooms of instance a. A target is drawn uniformly, then the shortest path and 8 stays.
- W14 is the mean over the 14 learnable rooms of bits/symbol on EVAL=2048 fresh symbols from a copy of the world. The frozen bank starts from a zero state and uses that room's channel row. Wa and Wb are the means of the two instances.
- Forgetting is M7's: end-of-life exam bits/symbol minus the snapshot taken at the end of the last dwell in that room, before a sleep that follows it. The mean is over learnable rooms that had an exam glance in that snapshot.
- World seeds 0, 1, 2, 3, 4 and 5. L = 2048 for every policy. oracle14_s also runs at L = 8192. CPU. Private Kaggle kernels, one seed each, pool of 4, no GPU, no internet. Nothing retuned.

## Verdict (orchestrator, 14 Nov 2026, UTC+8)

1. **Headroom with this cortex:** the better oracle is oracle14_s. Its gap to random dwelling plus sleep is −0.008 on
   the mean, and it is ≤ −0.15 on only 1 of 6 seeds (seed 2, −0.218); the bar needed 5. **FAIL.**
2. **Time stays scarce:** oracle14_s at L = 2,048 minus at L = 8,192 is +0.29 to +0.55 (mean +0.44), on 6 of 6 seeds. **PASS.**
Report only: breadth beats depth by a wide margin. oracle7a_s is +2.22 behind random dwelling. It learns its half better
(Wa 0.60 vs 0.88), but leaves the other half at 5.65 bits, worse than uniform over 41 symbols (5.36). The cortex's
shared weights are tuned away from the rooms it never visits.

Checks: the 19-row bank matches M7 bit for bit, and W4 restricted to W3's rooms reproduces M7b's W exactly. The dose is
0.199 at L = 2,048 and 0.200 at L = 8,192.

**Mechanism.** The life is short for this world: a 4× longer life halves the bits. But in a stationary world whose
learnable rooms all pay at about the same rate, spreading time evenly is already near the best allocation. Random
dwelling over 18 rooms wastes only the 4 noise and dark rooms (about 22% of room time). With a cortex whose gains
come steadily and evenly, that waste costs less than the seed-to-seed spread (std 0.07–0.11). What an allocation can gain
is set by how unequal and how changing the rooms' learning rates are, not by how many rooms there are. More rooms of the
same kind add scarcity, not headroom. As the kill rule says, the next world needs learnability that changes over time.
