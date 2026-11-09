# H21e measurements

Started: 2026-11-09 22:38:23 UTC+8. Updated: 2026-11-09 22:53:37 UTC+8.

Slow world; N = 131072; seeds = [6, 7, 8]; CPU threads = 2; population std (ddof=0).

| Learner / floor | Mean ± std bits/symbol | Seed 6 | Seed 7 | Seed 8 |
|---|---:|---:|---:|---:|
| att_rb8x2 | 1.312966 ± 0.019504 | 1.338130 | 1.290601 | 1.310169 |
| att_rb8x2+match | 1.232075 ± 0.036177 | 1.281367 | 1.195560 | 1.219296 |
| gru_rb8x2 | 0.812894 ± 0.169738 | 0.720937 | 1.050900 | 0.666846 |
| gru_rb8x2+match | 0.772086 ± 0.157386 | 0.687302 | 0.992704 | 0.636254 |
| kt_mixture | 1.735062 ± 0.376059 | 2.188957 | 1.268084 | 1.748144 |
| lzma_9e | 0.743896 ± 0.064964 | 0.758057 | 0.815430 | 0.658203 |

Per-process bits/symbol; arithmetic mean over seeds with observations.

| Learner | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|
| att_rb8x2 | 2.306775 | 1.781183 | 2.244199 | 1.074090 | 0.989432 | 0.480043 | 0.823859 |
| att_rb8x2+match | 2.109527 | 1.745446 | 2.244856 | 1.042561 | 0.967966 | 0.382000 | 0.578293 |
| gru_rb8x2 | 1.050321 | 0.985559 | 0.717813 | 1.269215 | 1.169159 | 0.377766 | 0.462741 |
| gru_rb8x2+match | 0.997326 | 0.965341 | 0.709717 | 1.218045 | 1.141862 | 0.320175 | 0.327618 |
| kt_mixture | 1.268344 | 3.168962 | 3.484201 | 1.189645 | 1.165874 | 0.511597 | 0.624158 |
| lzma_9e | NA | NA | NA | NA | NA | NA | NA |

| Seed | Learner | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | kt_mixture | 1.269379 | 3.253241 | 3.502203 | 1.201425 | 1.157696 | NA | 0.621232 |
| 6 | lzma_9e | NA | NA | NA | NA | NA | NA | NA |
| 7 | kt_mixture | 1.226282 | 3.082604 | NA | 1.177866 | 1.150181 | 0.511461 | 0.668232 |
| 7 | lzma_9e | NA | NA | NA | NA | NA | NA | NA |
| 8 | kt_mixture | 1.309371 | 3.171041 | 3.466199 | NA | 1.189745 | 0.511732 | 0.583011 |
| 8 | lzma_9e | NA | NA | NA | NA | NA | NA | NA |
| 6 | gru_rb8x2 | 1.254499 | 0.320983 | 0.676462 | 1.429011 | 1.091043 | NA | 0.353531 |
| 6 | gru_rb8x2+match | 1.168044 | 0.314251 | 0.660054 | 1.325919 | 1.091764 | NA | 0.309880 |
| 7 | gru_rb8x2 | 0.992089 | 1.998193 | NA | 1.109419 | 1.057152 | 0.379583 | 0.704197 |
| 7 | gru_rb8x2+match | 0.956783 | 1.953340 | NA | 1.110171 | 1.057856 | 0.335279 | 0.360174 |
| 8 | gru_rb8x2 | 0.904374 | 0.637503 | 0.759163 | NA | 1.359283 | 0.375948 | 0.330495 |
| 8 | gru_rb8x2+match | 0.867151 | 0.628432 | 0.759379 | NA | 1.275966 | 0.305071 | 0.312799 |
| 6 | att_rb8x2 | 2.339267 | 1.090336 | 2.281205 | 1.137280 | 0.957175 | NA | 0.719100 |
| 6 | att_rb8x2+match | 2.178555 | 1.074258 | 2.281798 | 1.073600 | 0.957895 | NA | 0.536316 |
| 7 | att_rb8x2 | 2.262747 | 2.671840 | NA | 1.010900 | 0.856810 | 0.483349 | 1.171450 |
| 7 | att_rb8x2+match | 2.077170 | 2.615997 | NA | 1.011522 | 0.856672 | 0.400828 | 0.734097 |
| 8 | att_rb8x2 | 2.318310 | 1.581373 | 2.207194 | NA | 1.154311 | 0.476737 | 0.581027 |
| 8 | att_rb8x2+match | 2.072856 | 1.546082 | 2.207915 | NA | 1.089329 | 0.363173 | 0.464467 |

| Seed | Learner | Wall seconds | Parameters |
|---:|---|---:|---:|
| 6 | kt_mixture | 20.467 | NA |
| 6 | lzma_9e | 0.177 | NA |
| 7 | kt_mixture | 15.253 | NA |
| 7 | lzma_9e | 0.229 | NA |
| 8 | kt_mixture | 20.823 | NA |
| 8 | lzma_9e | 0.192 | NA |
| 6 | gru_rb8x2 | 663.097 | 234569 |
| 6 | gru_rb8x2+match | 667.697 | NA |
| 7 | gru_rb8x2 | 668.958 | 234569 |
| 7 | gru_rb8x2+match | 673.916 | NA |
| 8 | gru_rb8x2 | 668.287 | 234569 |
| 8 | gru_rb8x2+match | 673.434 | NA |
| 6 | att_rb8x2 | 329.363 | 109481 |
| 6 | att_rb8x2+match | 333.958 | 109481 |
| 7 | att_rb8x2 | 278.438 | 109481 |
| 7 | att_rb8x2+match | 283.334 | 109481 |
| 8 | att_rb8x2 | 282.565 | 109481 |
| 8 | att_rb8x2+match | 287.782 | 109481 |

Attention layers: 2; d=64; heads=4; MLP=256; pre-LayerNorm; final LayerNorm; dropout=0; readout=41.
Learned absolute position slots 0–63 repeat across the context and target halves of the 128-position input.
Inputs shifted by one symbol; zero symbol embedding before global position 1. Causal keys i−63 through i (64 preceding observed symbols); negative context positions masked. Loss only on the final 64 targets.
Each chunk scored before its live Adam step; lr=0.001; clip=1.0; then two replay Adam steps, batch 8. Uniform starts 0 through scored_prefix_length−64 inclusive with replacement, each with preceding 64-symbol context.
GRU, KT and lzma scores/timings copied from H21c; no reruns. Match probabilities/timings reused from H21b. Fixed-share mixture initial weights (0.5,0.5), alpha=0.001.
Mixture wall seconds include cortex, reused match timing and mixture update. Parameters for mixture rows count the cortex only.
Prequential observation bits; retained history unpriced as in H21b. lzma has no per-process attribution.
Full probabilities and code lengths saved in learner-specific NPZ files.
Causality check: passed. Probability normalization, finite bits, and bit/probability agreement checked for each new stream.

Two-layer smoke wall seconds: 20.430. Projected three-seed cortex seconds: 980.647.

total_wall_seconds: 914.081.
smoke_plus_full_wall_seconds: 937.971.

## Verdict (orchestrator, 9 Nov 2026 UTC+8)

1. **M1 PASS (slow): FAIL.** att_rb8x2+match 1.232 vs lzma 0.744.
2. **CA cracked (≤ 0.6): FAIL.** E30 1.04, E110 0.97: better than the GRU (1.22 / 1.14), as predicted, but far from 0.6.
3. **Substrate: FAIL.** +0.460 worse than gru_rb8x2+match; grammar 2.11 vs 1.00, counters 1.75–2.24 vs 0.71–0.97.

Mechanism: position-aligned attention helps the CA rows a little, but a transformer from scratch on ~6 000 online updates
learns everything else much more slowly than a GRU.

**Line decision:** the substrate line has two consecutive fails (H21d, H21e), so it stops under the kill rule. M1 stays
**Partial**, with best learner gru_rb8x2+match (beats lzma on 2/3 slow seeds, mean 0.772 vs 0.744, counters learnt).
The CA rule is parked as an open problem (candidate later: a GRU + local-attention hybrid). M2 (H25a) proceeds with
the GRU + batched replay learner, as its gate allows for Partial.
