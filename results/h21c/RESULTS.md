# H21c measurements

Started: 2026-11-09 20:24:48 UTC+8. Updated: 2026-11-09 21:26:40 UTC+8.

Slow world; N = 131072; seeds = [6, 7, 8]; CPU threads = 2; population std (ddof=0).

| Learner / floor | Mean ± std bits/symbol | Seed 6 | Seed 7 | Seed 8 |
|---|---:|---:|---:|---:|
| gru | 1.259044 ± 0.130395 | 1.431414 | 1.229612 | 1.116107 |
| gru_r2 | 0.988791 ± 0.135392 | 1.048390 | 1.116576 | 0.801409 |
| gru_r2+match | 0.901485 ± 0.114063 | 0.894554 | 1.044519 | 0.765382 |
| gru_r4 | NA | NA | NA | NA |
| gru_r4+match | NA | NA | NA | NA |
| gru_rb8 | 0.852642 ± 0.145386 | 0.772752 | 1.056656 | 0.728518 |
| gru_rb8+match | 0.808877 ± 0.128151 | 0.740492 | 0.988420 | 0.697719 |
| gru_rb8x2 | 0.812894 ± 0.169738 | 0.720937 | 1.050900 | 0.666846 |
| gru_rb8x2+match | 0.772086 ± 0.157386 | 0.687302 | 0.992704 | 0.636254 |
| kt_mixture | 1.735062 ± 0.376059 | 2.188957 | 1.268084 | 1.748144 |
| lzma_9e | 0.743896 ± 0.064964 | 0.758057 | 0.815430 | 0.658203 |

Per-process bits/symbol; arithmetic mean over seeds with observations.

| Learner | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru | 1.176889 | 1.853313 | 2.127385 | 1.331514 | 1.175189 | 0.445674 | 0.694593 |
| gru_r2 | 1.534191 | 1.212134 | 1.160798 | 1.310950 | 1.223179 | 0.382472 | 0.510376 |
| gru_r2+match | 1.165851 | 1.176660 | 1.157308 | 1.242084 | 1.187001 | 0.319807 | 0.355675 |
| gru_r4 | NA | NA | NA | NA | NA | NA | NA |
| gru_r4+match | NA | NA | NA | NA | NA | NA | NA |
| gru_rb8 | 1.020384 | 1.068212 | 0.851709 | 1.253129 | 1.165054 | 0.376013 | 0.500352 |
| gru_rb8+match | 0.976095 | 1.053610 | 0.844138 | 1.194060 | 1.136936 | 0.312849 | 0.338829 |
| gru_rb8x2 | 1.050321 | 0.985559 | 0.717813 | 1.269215 | 1.169159 | 0.377766 | 0.462741 |
| gru_rb8x2+match | 0.997326 | 0.965341 | 0.709717 | 1.218045 | 1.141862 | 0.320175 | 0.327618 |
| kt_mixture | 1.268344 | 3.168962 | 3.484201 | 1.189645 | 1.165874 | 0.511597 | 0.624158 |
| lzma_9e | NA | NA | NA | NA | NA | NA | NA |

| Seed | Learner | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | gru_rb8 | 1.140665 | 0.464639 | 0.793763 | 1.388930 | 1.090653 | NA | 0.374089 |
| 6 | gru_rb8+match | 1.095401 | 0.458382 | 0.778541 | 1.270040 | 1.091374 | NA | 0.313434 |
| 7 | gru_rb8 | 0.992257 | 1.936218 | NA | 1.117328 | 1.051589 | 0.397706 | 0.789234 |
| 7 | gru_rb8+match | 0.939799 | 1.903486 | NA | 1.118080 | 1.052308 | 0.328673 | 0.394625 |
| 8 | gru_rb8 | 0.928230 | 0.803778 | 0.909656 | NA | 1.352920 | 0.354319 | 0.337733 |
| 8 | gru_rb8+match | 0.893084 | 0.798963 | 0.909735 | NA | 1.267128 | 0.297025 | 0.308426 |
| 6 | gru_rb8x2 | 1.254499 | 0.320983 | 0.676462 | 1.429011 | 1.091043 | NA | 0.353531 |
| 6 | gru_rb8x2+match | 1.168044 | 0.314251 | 0.660054 | 1.325919 | 1.091764 | NA | 0.309880 |
| 7 | gru_rb8x2 | 0.992089 | 1.998193 | NA | 1.109419 | 1.057152 | 0.379583 | 0.704197 |
| 7 | gru_rb8x2+match | 0.956783 | 1.953340 | NA | 1.110171 | 1.057856 | 0.335279 | 0.360174 |
| 8 | gru_rb8x2 | 0.904374 | 0.637503 | 0.759163 | NA | 1.359283 | 0.375948 | 0.330495 |
| 8 | gru_rb8x2+match | 0.867151 | 0.628432 | 0.759379 | NA | 1.275966 | 0.305071 | 0.312799 |

| Seed | Learner | Wall seconds |
|---:|---|---:|
| 6 | kt_mixture | 20.467 |
| 6 | lzma_9e | 0.177 |
| 6 | gru | 303.413 |
| 6 | gru_r2 | 756.621 |
| 6 | gru_r2+match | 761.120 |
| 7 | kt_mixture | 15.253 |
| 7 | lzma_9e | 0.229 |
| 7 | gru | 453.409 |
| 7 | gru_r2 | 880.424 |
| 7 | gru_r2+match | 885.637 |
| 8 | kt_mixture | 20.823 |
| 8 | lzma_9e | 0.192 |
| 8 | gru | 315.016 |
| 8 | gru_r2 | 976.028 |
| 8 | gru_r2+match | 981.723 |
| 6 | gru_rb8 | 712.414 |
| 6 | gru_rb8+match | 716.929 |
| 7 | gru_rb8 | 524.722 |
| 7 | gru_rb8+match | 530.607 |
| 8 | gru_rb8 | 428.953 |
| 8 | gru_rb8+match | 434.618 |
| 6 | gru_rb8x2 | 663.097 |
| 6 | gru_rb8x2+match | 667.697 |
| 7 | gru_rb8x2 | 668.958 |
| 7 | gru_rb8x2+match | 673.916 |
| 8 | gru_rb8x2 | 668.287 |
| 8 | gru_rb8x2+match | 673.434 |

Baseline scores and timings copied from H21b; no baseline reruns. Match probabilities and timings reused from H21b.
GRU(256), embedding 32, chunk 64, Adam lr 0.003, clip 1.0; H21b shifted inputs and detached carried live state.
Replay follows each scored chunk and its live Adam step. Uniform starts 0 through scored_prefix_length−64 inclusive, with replacement; zero replay hidden state; preceding-symbol input (zero embedding at global position 0). Live state is preserved during replay.
gru_r4: four sequential single-chunk Adam steps. gru_rb8: one Adam step on eight chunks. gru_rb8x2: two such steps. Batched loss is the mean over all 8 × 64 targets.
Mixtures reuse H21b fixed share α = 0.001; wall seconds include cortex, reused match timing, and mixture update.
Scores are prequential observation bits; retained history is unpriced, as specified in H21b. lzma has no per-process attribution.
Grammar G for every row is in the per-process table; plain GRU reference = 1.176889 bits/symbol (H21b).
Full predictive distributions and code lengths are saved in learner-specific NPZ files.

Runtime note: gru_r4 and gru_r4+match omitted first: smoke projection exceeds 81-minute planning allowance within 90-minute budget.

Total run wall seconds: 3711.950.

## Verdict (orchestrator, 9 Nov 2026 UTC+8)

1. **M1 PASS (slow): FAIL, 2 of 3 seeds.** gru_rb8x2+match 0.687 / 0.993 / 0.636 vs lzma 0.758 / 0.815 / 0.658: beats lzma
   on seeds 6 and 8, loses seed 7. Mean 0.772 vs 0.744. Counters pass (C1 0.30×, C7 0.20× KT).
2. **Dose-response: PASS.** gru_rb8x2 0.813 vs gru_r2 0.989 (−0.18).
3. **Cheap replay: PASS.** gru_rb8 0.853 ≤ gru_r2 0.989 at 0.64× gru_r2's wall time (mean 555 s vs 871 s per stream).
4. Grammar: batched replay *improves* G (1.02 / 1.05 vs plain 1.18), where sequential ×2 hurt it (1.53).

Mechanism:
- A batch of 8 diverse past chunks makes each update less biased toward whatever the last draw was. That cures the
  crowd-out and costs less on a 2-core CPU than sequential steps.
- **The CA rows are the remaining obstacle.** Seed 7 (the loss) holds 2× more CA than the other seeds. CA bits stay ≈1.2
  against a generator entropy of ≈0.04, because the rule needs exact access to the symbol 32 back (one row up). That is
  the algorithmic part a GRU cannot pick up on ~6 000 updates. Next: H21d (more batched replay, a wider GRU).
