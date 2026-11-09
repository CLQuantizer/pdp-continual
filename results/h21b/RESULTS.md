# H21b measurements

Started: 2026-11-09 18:59:38 UTC+8. Updated: 2026-11-09 20:18:11 UTC+8.

N = 131072; seeds = [6, 7, 8]; CPU threads = 2; population std (ddof=0).

| Learner / floor | slow bits/symbol | fast bits/symbol |
|---|---:|---:|
| gru | 1.259044 ± 0.130395 | 1.656944 ± 0.238808 |
| gru_r2 | 0.988791 ± 0.135392 | NA |
| match | 3.384170 ± 0.456640 | 3.392577 ± 0.095232 |
| gru+match | 1.158131 ± 0.144050 | 1.507010 ± 0.147992 |
| gru_r2+match | 0.901485 ± 0.114063 | NA |
| kt_mixture | 1.735062 ± 0.376059 | 1.712076 ± 0.038289 |
| lzma_9e | 0.743896 ± 0.064964 | 0.976156 ± 0.011462 |

| World | Seed | gru | gru_r2 | match | gru+match | gru_r2+match | kt_mixture | lzma_9e |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| slow | 6 | 1.431414 | 1.048390 | 4.028478 | 1.360578 | 0.894554 | 2.188957 | 0.758057 |
| slow | 7 | 1.229612 | 1.116576 | 3.024204 | 1.076575 | 1.044519 | 1.268084 | 0.815430 |
| slow | 8 | 1.116107 | 0.801409 | 3.099827 | 1.037239 | 0.765382 | 1.748144 | 0.658203 |
| fast | 6 | 1.596882 | NA | 3.451577 | 1.514312 | NA | 1.712171 | 0.984863 |
| fast | 7 | 1.974791 | NA | 3.467925 | 1.684501 | NA | 1.758922 | 0.959961 |
| fast | 8 | 1.399159 | NA | 3.258230 | 1.322217 | NA | 1.665135 | 0.983643 |

Per-process bits/symbol, slow world; arithmetic mean over seeds with observations.

| Learner | G | C1 | C7 | E30 | E110 | M7 | M37 |
|---|---:|---:|---:|---:|---:|---:|---:|
| gru | 1.176889 | 1.853313 | 2.127385 | 1.331514 | 1.175189 | 0.445674 | 0.694593 |
| gru_r2 | 1.534191 | 1.212134 | 1.160798 | 1.310950 | 1.223179 | 0.382472 | 0.510376 |
| match | 2.396121 | 5.344787 | 5.368773 | 3.884858 | 3.469096 | 0.864290 | 1.048116 |
| gru+match | 1.028806 | 1.824066 | 2.126055 | 1.199350 | 1.141241 | 0.309366 | 0.395665 |
| gru_r2+match | 1.165851 | 1.176660 | 1.157308 | 1.242084 | 1.187001 | 0.319807 | 0.355675 |
| kt_mixture | 1.268344 | 3.168962 | 3.484201 | 1.189645 | 1.165874 | 0.511597 | 0.624158 |
| lzma_9e | NA | NA | NA | NA | NA | NA | NA |

lzma per-process bits: NA; the whole-stream file code has no per-symbol or process attribution.

Slow-world match statistics: arithmetic mean over seeds with observations (hit rate excludes seeds with no matches).

| Process | Seeds observed | Mean match fraction | Mean match hit rate |
|---|---:|---:|---:|
| all | 3 | 0.638908 | 0.716759 |
| G | 3 | 0.905457 | 0.773456 |
| C1 | 3 | 0.035645 | 0.198859 |
| C7 | 2 | 0.086823 | 0.298773 |
| E30 | 2 | 0.946121 | 0.479310 |
| E110 | 3 | 0.976308 | 0.550517 |
| M7 | 2 | 0.922028 | 0.963029 |
| M37 | 3 | 0.883016 | 0.963727 |

| World | Seed | Process | Symbols | Match fraction | Match hit rate |
|---|---:|---|---:|---:|---:|
| slow | 6 | all | 131072 | 0.495338 | 0.656419 |
| slow | 6 | G | 16384 | 0.895752 | 0.761311 |
| slow | 6 | C1 | 49152 | 0.061829 | 0.297795 |
| slow | 6 | C7 | 16384 | 0.076599 | 0.298805 |
| slow | 6 | E30 | 16384 | 0.929199 | 0.478061 |
| slow | 6 | E110 | 16384 | 0.995544 | 0.550855 |
| slow | 6 | M7 | 0 | NA | NA |
| slow | 6 | M37 | 16384 | 0.880127 | 0.964078 |
| slow | 7 | all | 131072 | 0.829582 | 0.674291 |
| slow | 7 | G | 16384 | 0.914856 | 0.782440 |
| slow | 7 | C1 | 16384 | 0.000061 | 0.000000 |
| slow | 7 | C7 | 0 | NA | NA |
| slow | 7 | E30 | 32768 | 0.963043 | 0.480559 |
| slow | 7 | E110 | 32768 | 0.998627 | 0.551691 |
| slow | 7 | M7 | 16384 | 0.921692 | 0.961724 |
| slow | 7 | M37 | 16384 | 0.876709 | 0.964216 |
| slow | 8 | all | 131072 | 0.591805 | 0.819567 |
| slow | 8 | G | 16384 | 0.905762 | 0.776617 |
| slow | 8 | C1 | 32768 | 0.045044 | 0.298780 |
| slow | 8 | C7 | 16384 | 0.097046 | 0.298742 |
| slow | 8 | E30 | 0 | NA | NA |
| slow | 8 | E110 | 16384 | 0.934753 | 0.549004 |
| slow | 8 | M7 | 16384 | 0.922363 | 0.964333 |
| slow | 8 | M37 | 32768 | 0.892212 | 0.962888 |
| fast | 6 | all | 131072 | 0.654701 | 0.686271 |
| fast | 6 | G | 12288 | 0.863688 | 0.758504 |
| fast | 6 | C1 | 20480 | 0.053711 | 0.297273 |
| fast | 6 | C7 | 20480 | 0.061475 | 0.296267 |
| fast | 6 | E30 | 22528 | 0.955611 | 0.470550 |
| fast | 6 | E110 | 22528 | 0.984331 | 0.540564 |
| fast | 6 | M7 | 12288 | 0.913086 | 0.960873 |
| fast | 6 | M37 | 20480 | 0.874902 | 0.962328 |
| fast | 7 | all | 131072 | 0.645050 | 0.693606 |
| fast | 7 | G | 20480 | 0.925586 | 0.781072 |
| fast | 7 | C1 | 24576 | 0.061401 | 0.296885 |
| fast | 7 | C7 | 18432 | 0.051324 | 0.297040 |
| fast | 7 | E30 | 16384 | 0.949097 | 0.479035 |
| fast | 7 | E110 | 24576 | 0.979004 | 0.541978 |
| fast | 7 | M7 | 12288 | 0.909180 | 0.962943 |
| fast | 7 | M37 | 14336 | 0.861816 | 0.960016 |
| fast | 8 | all | 131072 | 0.661026 | 0.728515 |
| fast | 8 | G | 20480 | 0.920410 | 0.772308 |
| fast | 8 | C1 | 12288 | 0.040527 | 0.299197 |
| fast | 8 | C7 | 26624 | 0.041091 | 0.296161 |
| fast | 8 | E30 | 18432 | 0.955458 | 0.465561 |
| fast | 8 | E110 | 16384 | 0.970398 | 0.530851 |
| fast | 8 | M7 | 10240 | 0.899902 | 0.960933 |
| fast | 8 | M37 | 26624 | 0.881723 | 0.962513 |

| World | Seed | Learner | Wall seconds |
|---|---:|---|---:|
| slow | 6 | kt_mixture | 20.467 |
| slow | 6 | lzma_9e | 0.177 |
| slow | 6 | match | 1.681 |
| slow | 6 | gru | 303.413 |
| slow | 6 | gru+match | 308.607 |
| slow | 6 | gru_r2 | 756.621 |
| slow | 6 | gru_r2+match | 761.120 |
| slow | 7 | kt_mixture | 15.253 |
| slow | 7 | lzma_9e | 0.229 |
| slow | 7 | match | 2.392 |
| slow | 7 | gru | 453.409 |
| slow | 7 | gru+match | 459.054 |
| slow | 7 | gru_r2 | 880.424 |
| slow | 7 | gru_r2+match | 885.637 |
| slow | 8 | kt_mixture | 20.823 |
| slow | 8 | lzma_9e | 0.192 |
| slow | 8 | match | 2.442 |
| slow | 8 | gru | 315.016 |
| slow | 8 | gru+match | 320.292 |
| slow | 8 | gru_r2 | 976.028 |
| slow | 8 | gru_r2+match | 981.723 |
| fast | 6 | kt_mixture | 17.495 |
| fast | 6 | lzma_9e | 0.208 |
| fast | 6 | match | 2.270 |
| fast | 6 | gru | 231.399 |
| fast | 6 | gru+match | 236.930 |
| fast | 7 | kt_mixture | 16.747 |
| fast | 7 | lzma_9e | 0.234 |
| fast | 7 | match | 1.865 |
| fast | 7 | gru | 217.310 |
| fast | 7 | gru+match | 222.353 |
| fast | 8 | kt_mixture | 16.951 |
| fast | 8 | lzma_9e | 0.247 |
| fast | 8 | match | 2.096 |
| fast | 8 | gru | 270.042 |
| fast | 8 | gru+match | 274.833 |

Mixture wall time includes its cortex, match model, and mixture update; components are reused, not rerun.
GRU: H21a imported GRU(256), embedding 32, chunk 64, Adam lr 0.003, clip 1.0; shifted inputs, first embedding zero, carried state detached per chunk.
Replay: 2 separate Adam steps after every scored chunk; independent uniform starts 0 through scored_prefix_length−64 inclusive; replay hidden state zero, preceding symbol input (zero embedding at position 0). Live hidden state is preserved across replay.
Match: preceding 8-symbol key, most recent earlier target, backward extension to 32; bucket confidence (hits+1)/(trials+2); remaining probability uniform over 40 symbols. Updates follow scoring.
Mixtures: initial weights (0.5,0.5); likelihood update then fixed share 0.001 after each scored symbol.
Scores are prequential observation bits; raw history and match table are retained. No store charge is added to the specified score; this is not two-part T.
lzma: preset 9 | PRESET_EXTREME, whole-stream bytes including container overhead. KT: imported h20_world.kt_floors.
Full float64 predictive distributions and symbol code lengths are saved in stream-specific NPZ files.

Runtime note: Fast gru_r2 and gru_r2+match omitted: smoke projection exceeds 81-minute planning allowance within 90-minute budget.

Total run wall seconds: 4592.801.

## Verdict (orchestrator, 9 Nov 2026 UTC+8)

Checks: the six plain-GRU runs reproduce H21a exactly (deterministic pipeline); fast-world replay was dropped under the
spec's budget rule.

1. **M1 PASS (with store): FAIL.** Best mixture gru_r2+match 0.895 / 1.045 / 0.765 vs lzma 0.758 / 0.815 / 0.658 (slow;
   gap 0.16 bits on the mean). The counter half passes: C1 0.37×, C7 0.33× the KT mixture's bits.
2. **Replay helps: PASS in slow** (−0.270 bits, 1.259 → 0.989); fast not run, so not scored there.
3. **Store helps: PASS** (gru+match vs gru: −0.101 slow, −0.150 fast).

Mechanism:
- Replay is the biggest lever so far: two extra updates per chunk recover 0.27 bits, more than half the remaining gap
  to lzma. Counters are learnt now; CA rows are not (E30 1.24 vs KT 1.19).
- Uniform replay hurts the grammar (G 1.18 → 1.53): replaying uniformly from history crowds some processes out.
  *What* to replay matters, which is where whole-history progress (H23c) can select replay (H22).
- The match model alone is weak (3.38: uniform when it has no match, crude confidence), but it adds 0.10–0.15 bits in
  mixture, mostly on motifs (M37 0.69 → 0.40).

Next: **H21c**, a replay dose-response (sequential ×4; batched ×8 at about the cost of ×2) on the slow world.
