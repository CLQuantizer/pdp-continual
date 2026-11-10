# H29 enwik8

File bytes: 100000000. sha256: `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`.
E10M bytes: 10000000. sha256: `5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97`.
Prefix starts: `<mediawiki xmlns="http://www.mediawiki.o`.

## E10M bits/byte

| Method | Compressed bytes | Bits/byte | Wall seconds |
|---|---:|---:|---:|
| xz -9e -T1 | 2720256 | 2.176205 | 13.452 |
| zstd --ultra -22 --long=27 | 2793335 | 2.234668 | 15.091 |
| bz2 level 9 | 2916026 | 2.332821 | 1.947 |
| lzma preset 9|EXTREME | 2720256 | 2.176205 | 12.532 |
| ppmd H order 6 mem 1073741824 | 2368372 | 1.894698 | 9.356 |
| ppmd H order 16 mem 1073741824 | 2296568 | 1.837254 | 15.978 |
| GRU replay lr 0.001 | prequential | 3.644034 | 4230.479 |

Baseline bits/byte = 8 × compressed bytes / 10000000, container included, round-trip checked.
GRU bits/byte = sum of live −log2 p / 10000000. Replay is not part of the code length.

## GRU learning curve, E10M, 1 MB slices

| Slice | Start | End | Bits | Bits/byte |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1000000 | 2259171.224 | 2.259171 |
| 1 | 1000000 | 2000000 | 2988231.974 | 2.988232 |
| 2 | 2000000 | 3000000 | 4170414.854 | 4.170415 |
| 3 | 3000000 | 4000000 | 3583871.429 | 3.583871 |
| 4 | 4000000 | 5000000 | 3912805.136 | 3.912805 |
| 5 | 5000000 | 6000000 | 3602210.995 | 3.602211 |
| 6 | 6000000 | 7000000 | 4161338.492 | 4.161338 |
| 7 | 7000000 | 8000000 | 4157890.413 | 4.157890 |
| 8 | 8000000 | 9000000 | 4111762.461 | 4.111762 |
| 9 | 9000000 | 10000000 | 3492647.483 | 3.492647 |

Last 1 MB (bytes 9000000:10000000): 3.492647 bits/byte.

## Learning-rate probe, first 1 MB, replay on

| lr | Bits/byte | Wall seconds | Replay steps |
|---:|---:|---:|---:|
| 0.001 | 2.259171 | 431.857 | 7813 |
| 0.003 | 3.838527 | 422.735 | 7813 |
Chosen lr: 0.001.

## Replay versus no-replay, first 2 MB, chosen lr

| Variant | Bits/byte | Bits | Wall seconds | Replay steps |
|---|---:|---:|---:|---:|
| replay (prefix of the E10M run) | 2.623702 | 5247403.198 | (inside the E10M pass) | 15625 |
| no-replay | 2.273729 | 4547457.350 | 283.804 | 0 |

Difference (replay − no-replay) bits/byte: 0.349973.

## Published full-enwik8 references (Mahoney LTCB)

Not measured here. Full-file numbers; an E10M number is on less data.

| Compressor | Bits/byte |
|---|---:|
| xz | 1.976 |
| ppmd J1 | 1.711 |
| cmix v21 | 1.170 |

GRU last 1 MB of E10M: 3.492647.

## Bar inputs

Pre-registered. Not scored in this file.

1. GRU E10M − xz -9e E10M = 1.467830 (3.644034 − 2.176205).
2. GRU E10M − better PPMd (ppmd H order 16 mem 1073741824) = 1.806780 (3.644034 − 1.837254).
3. Last 1 MB 3.492647. Replay 2 MB 2.623702. No-replay 2 MB 2.273729.

## Compute

| Item | Value |
|---|---|
| Kernel | ezzzzzx/h29-enwik8 |
| Kernel version | 1 |
| GPU | Tesla T4 |
| Session wall seconds | 5368.876 |
| GPU hours (session wall / 3600) | 1.4914 |
| E10M wall seconds | 4230.479 |
| Probe walls seconds | 431.857, 422.735 |
| No-replay wall seconds | 283.804 |
| Stretch | E10M wall 4230.5s is not under 3600s |
| Parameters | 3627264 |
| Torch | 2.11.0+cu130 |

## Smoke, first 100000 bytes, CPU

| Bits/byte | Wall seconds | Replay steps | Finite | Device |
|---:|---:|---:|---|---|
| 2.664724 | 2771.968 | 782 | True | cpu |

## Protocol

- One online pass. Prequential: −log2 p of each byte is recorded before that chunk is trained.
- GRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam, clip 1.0. Seed 0. Replay RNG 30000.
- Live state is carried and detached every 128 bytes. Replay state is zero. Replay batch 32.
- Replay starts are uniform on 0 .. L−128 inclusive, with replacement.
- Chosen learning rate 0.001 from the two 1 MB probes. Scored model is fresh.
- E10M is bytes [0, 10000000). 1 MB slices are 1000000 bytes.
- No-replay is a fresh model at the chosen learning rate on bytes [0, 2000000).
- PPMd is pyppmd variant H (Ppmd7), orders 6 and 16, mem_size 1073741824, raw stream plus flush.
- Smoke is not a score.

## Verdict (orchestrator, 11 Nov 2026, 01:30 UTC+8)

**Bar 1 (beat xz on E10M): FAIL**, +1.468 bits/byte. **Bar 2 (beat PPMd): FAIL**, +1.807. Within the 3 GPU-hour budget (1.49 h used).

Mechanism. The model is not short of capacity; it gets worse as it reads. The first MB costs 2.26 bits/byte, about
what xz achieves on the whole 10 MB. Later slices drift up to 3.5–4.2, close to a byte-unigram code. Replay made the
first 2 MB worse by +0.35 bits/byte. The replay dose was the cause, and the spec set it. One replay batch of 32 × 128
bytes per 128-byte live chunk is traffic 32, four times the knee H27b found (8). Early in the stream that is hundreds
of repeats per stored segment, far past the ≲ 10 repeats where replay stops helping. The orchestrator wrote that dose
the same evening it closed "replay doses above traffic 8". That is an orchestrator error, not a worker one. The code
matches the spec: prequential order, scoring before each step, clipping, carried state.

What this does not show: whether the cortex can beat xz once the dose is right. The no-replay run (2.27 at 2 MB) is
also not yet better than xz on the same 2 MB. H29b separates the dose from the stability question.
