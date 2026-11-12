# H31 sleep gate on enwik8 E3M

E3M bytes: 3000000. sha256: `72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac`.
E10M sha256, checked and not scored: `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
GRU bits/byte = sum of live −log2 p / 3000000. Replay is not part of the code length.
No bytes are scored during sleep.

Fixed settings: lr 0.0003, seed 0, replay seed 30000, embedding 64, hidden 1024,
chunk 128, clip 1.0, Adam. Replay batch 8, except noreplay batch 0.

## E3M

| Arm | Sleep block | Replay batch | Bits/byte | Replay steps | Wall seconds | GPU |
|---|---:|---:|---:|---:|---:|---|
| inter | 1 | 8 | 1.958153 | 23438 | 1087.922 | Tesla T4 |
| sleep_64 | 64 | 8 | 2.001622 | 23438 | 1141.211 | Tesla T4 |
| sleep_1k | 1024 | 8 | 2.090683 | 23438 | 1144.753 | Tesla T4 |
| sleep_16k | 16384 | 8 | 2.192371 | 23438 | 1172.649 | Tesla T4 |
| noreplay | none | 0 | 2.197983 | 0 | 446.571 | Tesla T4 |

## 1 MB slices

### inter

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2104744.492 | 2.104744 |
| 1 | 1000000 | 2000000 | 1923321.387 | 1.923321 |
| 2 | 2000000 | 3000000 | 1846393.123 | 1.846393 |

### sleep_64

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2161188.033 | 2.161188 |
| 1 | 1000000 | 2000000 | 1965735.152 | 1.965735 |
| 2 | 2000000 | 3000000 | 1877942.248 | 1.877942 |

### sleep_1k

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2310924.083 | 2.310924 |
| 1 | 1000000 | 2000000 | 2030953.121 | 2.030953 |
| 2 | 2000000 | 3000000 | 1930172.802 | 1.930173 |

### sleep_16k

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2511646.944 | 2.511647 |
| 1 | 1000000 | 2000000 | 2109053.234 | 2.109053 |
| 2 | 2000000 | 3000000 | 1956413.384 | 1.956413 |

### noreplay

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2511646.944 | 2.511647 |
| 1 | 1000000 | 2000000 | 2109053.234 | 2.109053 |
| 2 | 2000000 | 3000000 | 1973250.047 | 1.973250 |

## Gaps

Pre-registered differences. Not scored in this file.

sleep_64 − inter = 0.043469 (2.001622 − 1.958153).
sleep_1k − inter = 0.132530 (2.090683 − 1.958153).
sleep_16k − inter = 0.234218 (2.192371 − 1.958153).
noreplay − inter = 0.239830 (2.197983 − 1.958153).

## Morning after

128 KB = 131072 bytes = 1024 chunks. For each sleep that has a full 128 KB awake
stretch immediately before it and a full 128 KB awake stretch immediately after it,
bits/byte of those stretches. The mean is over those sleeps. Awake bits only.

### sleep_1k

Sleeps in the mean: 21.
Mean bits/byte before: 2.103714.
Mean bits/byte after: 2.027268.
Mean after − before: -0.076445.

| Sleep | Bytes seen | Bits/byte before | Bits/byte after | After − before |
|---:|---:|---:|---:|---:|
| 0 | 131072 | 3.636092 | 2.261903 | -1.374188 |
| 1 | 262144 | 2.261903 | 2.188859 | -0.073044 |
| 2 | 393216 | 2.188859 | 2.128630 | -0.060229 |
| 3 | 524288 | 2.128630 | 1.881514 | -0.247116 |
| 4 | 655360 | 1.881514 | 2.141407 | 0.259893 |
| 5 | 786432 | 2.141407 | 2.079799 | -0.061607 |
| 6 | 917504 | 2.079799 | 2.044012 | -0.035787 |
| 7 | 1048576 | 2.044012 | 2.039447 | -0.004565 |
| 8 | 1179648 | 2.039447 | 2.039877 | 0.000430 |
| 9 | 1310720 | 2.039877 | 2.009533 | -0.030344 |
| 10 | 1441792 | 2.009533 | 2.050558 | 0.041025 |
| 11 | 1572864 | 2.050558 | 1.977503 | -0.073055 |
| 12 | 1703936 | 1.977503 | 2.022446 | 0.044943 |
| 13 | 1835008 | 2.022446 | 2.054601 | 0.032155 |
| 14 | 1966080 | 2.054601 | 2.018657 | -0.035943 |
| 15 | 2097152 | 2.018657 | 1.905399 | -0.113258 |
| 16 | 2228224 | 1.905399 | 1.888180 | -0.017219 |
| 17 | 2359296 | 1.888180 | 1.933696 | 0.045516 |
| 18 | 2490368 | 1.933696 | 1.878379 | -0.055316 |
| 19 | 2621440 | 1.878379 | 1.997497 | 0.119117 |
| 20 | 2752512 | 1.997497 | 2.030739 | 0.033242 |

### sleep_16k

Sleeps in the mean: 1.
Mean bits/byte before: 2.088853.
Mean bits/byte after: 1.933889.
Mean after − before: -0.154965.

| Sleep | Bytes seen | Bits/byte before | Bits/byte after | After − before |
|---:|---:|---:|---:|---:|
| 0 | 2097152 | 2.088853 | 1.933889 | -0.154965 |

## Kernels

Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that
clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.
Script wall is time.monotonic from kernel_main entry through the result write. Queue time
is not included. Total GPU hours is the sum across the five kernels.

| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours |
|---|---:|---|---:|---|---:|---:|
| ezzzzzx/h31-inter | 1 | COMPLETE | 1133.053 | log_clock | 1102.688 | 0.3147 |
| ezzzzzx/h31-sleep-64 | 1 | COMPLETE | 1178.563 | log_clock | 1155.195 | 0.3274 |
| ezzzzzx/h31-sleep-1k | 1 | COMPLETE | 1179.833 | log_clock | 1158.948 | 0.3277 |
| ezzzzzx/h31-sleep-16k | 1 | COMPLETE | 1208.045 | log_clock | 1185.612 | 0.3356 |
| ezzzzzx/h31-noreplay | 1 | COMPLETE | 483.633 | log_clock | 461.459 | 0.1343 |

Total GPU hours: 1.4398.

Kaggle accepted two GPU sessions at a time. ezzzzzx/h31-inter version 1 and ezzzzzx/h31-sleep-64 version 1 ran together. ezzzzzx/h31-sleep-1k version 1 and ezzzzzx/h31-sleep-16k version 1 were pushed after both of those reached COMPLETE. ezzzzzx/h31-noreplay version 1 was pushed after those two reached COMPLETE. No kernel was rerun.

## Protocol

- One online pass per arm. Prequential: −log2 p of each byte is recorded before that chunk is trained.
- Awake: score the 128-byte chunk, then one Adam step on it. No replay step while awake, except sleep block 1.
- Sleep: after every B live chunks, B replay steps back to back. Each step is batch 8, window 128, hidden state 0.
- Replay starts are uniform on bytes seen so far, the same sampler as H29d. One rng draw per replay step.
- Sleep block 1 replays once after every live chunk. That is the H29d path.
- A final sleep covers a remainder shorter than B. Replay steps match across the replay arms.
- The live hidden state is detached after the live score and is not replaced by replay.
- E3M is bytes [0, 3000000). 1 MB slices are 1000000 bytes.
- Traffic = replay steps × replay batch × 128 / bytes.
- Code is h31_sleep.py. Private dataset ezzzzzx/pdp-code-h31. Data dataset ezzzzzx/enwik8.

## Smoke

CPU. Not a score.

Bytes: 8192.
h29d bits: 42063.298164869797.
h31 --sleep-block 1 bits: 42063.298164869797.
Absolute difference: 0.000e+00.
Within 1e-6: True.

sleep block 16 replay steps: 64.
inter replay steps: 64.
Counts equal: True.
sleep block 16 bits finite: True.
sleep block 16 bits: 44293.816701635013.

Schedule, B = 4, first 12 chunks. live = scored. replay = not scored.

B=4 chunks=12
0 live chunk=0 scored bytes=[0,128)
1 live chunk=1 scored bytes=[128,256)
2 live chunk=2 scored bytes=[256,384)
3 live chunk=3 scored bytes=[384,512)
4 replay sleep_step=1/4 n_seen=512
5 replay sleep_step=2/4 n_seen=512
6 replay sleep_step=3/4 n_seen=512
7 replay sleep_step=4/4 n_seen=512
8 live chunk=4 scored bytes=[512,640)
9 live chunk=5 scored bytes=[640,768)
10 live chunk=6 scored bytes=[768,896)
11 live chunk=7 scored bytes=[896,1024)
12 replay sleep_step=1/4 n_seen=1024
13 replay sleep_step=2/4 n_seen=1024
14 replay sleep_step=3/4 n_seen=1024
15 replay sleep_step=4/4 n_seen=1024
16 live chunk=8 scored bytes=[1024,1152)
17 live chunk=9 scored bytes=[1152,1280)
18 live chunk=10 scored bytes=[1280,1408)
19 live chunk=11 scored bytes=[1408,1536)
20 replay sleep_step=1/4 n_seen=1536
21 replay sleep_step=2/4 n_seen=1536
22 replay sleep_step=3/4 n_seen=1536
23 replay sleep_step=4/4 n_seen=1536

Schedule kinds: LLLLRRRRLLLLRRRRLLLLRRRR.
Run step kinds: LLLLRRRRLLLLRRRRLLLLRRRR.
Schedule matches the run: True.

Remainder 200 bytes, sleep block 4 replay steps: 2.
Sleep block 1 replay steps: 2.
Counts equal: True. Both finite: True.

## Verdict (orchestrator, 12 Nov 2026, UTC+8)

**Bar 1 PASS, bar 2 FAIL.**
- Bar 1: noreplay − inter = 0.240 bits/byte on E3M (bar ≥ 0.03). Replay matters at 3 MB, so the comparison counts.
- Bar 2: sleep_1k − inter = 0.133 (bar ≤ 0.01). Batching replay into 128 KB sleeps gives up more than half the replay
  gain.
- The cost grows with block size: +0.043 (8 KB), +0.133 (128 KB), +0.234 (2 MB). At 2 MB it is almost all of the
  replay gain (noreplay +0.240).

**Mechanism: staleness, not dose.** Every arm took the same 23,438 replay steps, and sleep costs no scored time. The
difference is *when* the replay lands. In an awake stretch the cortex trains only on the live stream and drifts toward
recent text, and every byte in that stretch is scored by the drifting cortex. A sleep repairs it: the 128 KB right
after a sleep scores 0.076 (sleep_1k) and 0.155 (sleep_16k) bits/byte better than the 128 KB before it. But the
repair arrives after the cost has been paid.

Caveat, which does not change the verdict: the last sleep runs on the remainder chunks at the end of the data, and
its replay is never scored. That wastes about 4% of sleep_1k's dose and most of sleep_16k's (the 16k arm has one full
sleep at 2 MB, then the remaining 1 MB is awake). The 16k number therefore overstates batching's cost. The 64 and 1k
numbers barely change.

**For the brain map.** An online predictor that is scored on every input needs replay *while awake*, close to the
input. Sleep-only consolidation is costly in exactly the regime our machine lives in. This fits the fact that
hippocampal sharp-wave ripples also happen in awake rest pauses, not only in sleep. The thalamic sleep gate's job is
then probably not the bulk of consolidation for a predictor that lives online. It is a second, broad pass (H26 tests
that), or protection of the store while input is off.
