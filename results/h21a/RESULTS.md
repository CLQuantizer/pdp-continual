# H21a measurements

Started: 2026-11-09 18:05:47 UTC+8. Updated: 2026-11-09 18:53:20 UTC+8.

N = 131072; seeds = [6, 7, 8]; CPU threads = 2; population std (ddof=0).

| Learner / floor | slow bits/symbol | fast bits/symbol |
|---|---:|---:|
| gru | 1.259044 ± 0.130395 | 1.656944 ± 0.238808 |
| gru_big | NA | NA |
| lin_fw | 2.272146 ± 0.247650 | 2.963325 ± 0.094583 |
| kt_mixture | 1.735062 ± 0.376059 | 1.712076 ± 0.038289 |
| lzma_9e | 0.743896 ± 0.064964 | 0.976156 ± 0.011462 |

Per-process bits/symbol, slow world; arithmetic mean over seeds with observations (counts below).

| Learner | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru | 1.176889 | 1.853313 | 2.127385 | 1.331514 | 1.175189 | 0.445674 | 0.694593 |
| gru_big | NA | NA | NA | NA | NA | NA | NA |
| lin_fw | 2.373249 | 3.180613 | 3.522539 | 1.696457 | 1.248321 | 0.945315 | 2.531060 |
| kt_mixture | 1.268344 | 3.168962 | 3.484201 | 1.189645 | 1.165874 | 0.511597 | 0.624158 |

Slow-world learning curve, bits/symbol (mean over seeds).

| Learner | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gru | 0.689877 | 1.535245 | 2.016395 | 1.439278 | 1.110973 | 0.923198 | 0.953378 | 1.404012 |
| gru_big | NA | NA | NA | NA | NA | NA | NA | NA |
| lin_fw | 1.558257 | 3.084868 | 2.998031 | 2.456633 | 2.181288 | 1.511071 | 1.982645 | 2.404375 |

| World | Seed | Learner / floor | Wall seconds |
|---|---:|---|---:|
| slow | 6 | kt_mixture | 20.323 |
| slow | 6 | lzma_9e | 0.091 |
| slow | 6 | gru | 195.014 |
| slow | 6 | lin_fw | 286.053 |
| slow | 7 | kt_mixture | 14.681 |
| slow | 7 | lzma_9e | 0.194 |
| slow | 7 | gru | 203.054 |
| slow | 7 | lin_fw | 263.193 |
| slow | 8 | kt_mixture | 17.471 |
| slow | 8 | lzma_9e | 0.213 |
| slow | 8 | gru | 223.820 |
| slow | 8 | lin_fw | 227.354 |
| fast | 6 | kt_mixture | 17.776 |
| fast | 6 | lzma_9e | 0.218 |
| fast | 6 | gru | 199.918 |
| fast | 6 | lin_fw | 228.814 |
| fast | 7 | kt_mixture | 16.843 |
| fast | 7 | lzma_9e | 0.222 |
| fast | 7 | gru | 193.228 |
| fast | 7 | lin_fw | 229.189 |
| fast | 8 | kt_mixture | 17.331 |
| fast | 8 | lzma_9e | 0.227 |
| fast | 8 | gru | 212.174 |
| fast | 8 | lin_fw | 241.805 |

| World | Seed | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| slow | 6 | 16384 | 49152 | 16384 | 16384 | 16384 | 0 | 16384 |
| slow | 7 | 16384 | 16384 | 0 | 32768 | 32768 | 16384 | 16384 |
| slow | 8 | 16384 | 32768 | 16384 | 0 | 16384 | 16384 | 32768 |
| fast | 6 | 12288 | 20480 | 20480 | 22528 | 22528 | 12288 | 20480 |
| fast | 7 | 20480 | 24576 | 18432 | 16384 | 24576 | 12288 | 14336 |
| fast | 8 | 20480 | 12288 | 26624 | 18432 | 16384 | 10240 | 26624 |

Fast-weight 20% diagnostics (same-prefix KT comparison; no tuning or early termination).

| World | Seed | N scored | lin_fw bits | KT bits | Above KT |
|---|---:|---:|---:|---:|---|
| slow | 6 | 26240 | 2.329744 | 1.519507 | True |
| slow | 7 | 26240 | 2.587949 | 1.024295 | True |
| slow | 8 | 26240 | 2.114850 | 0.987815 | True |
| fast | 6 | 26240 | 4.187431 | 1.761078 | True |
| fast | 7 | 26240 | 3.889810 | 1.801604 | True |
| fast | 8 | 26240 | 4.175715 | 1.770985 | True |

Protocol: embedding 32; GRU hidden 256 / 512; readout 41; chunks 64; Adam lr 0.003; clip 1.0.
Initial hidden state zero; first input embedding zero; input is shifted by one symbol; state carried and detached after every chunk. No process IDs are passed to learners.
lin_fw: z_t=[Emb(x_(t-1)); y_(t-1)], k_t=softmax(Kz_t), q_t=softmax(Qz_t), v_t=tanh(Vz_t), beta_t=sigmoid(bz_t), W_t=W_(t-1)+beta_t(v_t-W_(t-1)k_t)k_t^T, y_t=W_t q_t, logits_t=R y_t. Dimension 64; W_0=0, y_0=0; projections/readout affine. Both W and y detached between chunks.
First 20% diagnostic uses the first complete 64-symbol chunk at or beyond 20%. All scores are recorded before the chunk optimizer step.
lzma uses preset 9 | PRESET_EXTREME including container overhead; KT uses h20_world.kt_floors on identical streams.
Scope: compressor-only measurement; no auxiliary exemplar store; no two-part T or recurrent-state charge added to the specified prequential score.

Runtime note: gru_big omitted: smoke projection for six streams with all three learners = 96.58 minutes; planning allowance = 81 minutes within the 90-minute total budget. Both worlds retained.

## Verdict (orchestrator, 9 Nov 2026 UTC+8): PARTIAL

Code reviewed: bits are recorded from the forward pass before backward/step (h21a_substrate.py lines 102–108); no
process ids reach the learners; lin_fw equations as documented.

- **M1 PASS: no.** The GRU never beats lzma (slow 1.12–1.43 vs 0.66–0.82; fast 1.40–1.98 vs 0.96–0.99).
- **Partial: yes.** The GRU beats the KT mixture on the mean in both worlds (1.259 vs 1.735 slow, 1.657 vs 1.712 fast),
  on 5 of 6 seeds (fast seed 7: 1.975 vs 1.759).
- **lin_fw (plain delta-rule FWP): fails**, above KT on every seed at 20% and at the end; untuned, per spec.

Mechanism:
1. **Step-starved, not capacity-limited.** One online pass with one Adam step per 64 symbols is about 2 000 updates per
   life. The GRU starts on counters (C1 0.50–0.89×, C7 0.57–1.21× the KT mixture's bits), but CA rows stay at KT level
   (0.94–1.46×). It has not had the updates to learn the one-row-back rule.
2. **lzma wins by verbatim recall.** LZ coding points back to exact earlier repeats (motif copies, CA rows, counter
   digits). That is episodic memory. The cortex is being beaten by a store, which is CLS's argument in compression
   form.

Next: **H21b**, cortex + store. Replay from the stored history gives more updates per observed symbol; a match model
(exact recall) is mixed with the GRU's predictions.
