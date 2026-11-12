I'll read the project instructions and the handoff, then follow that handoff exactly.The handoff is H31: hold the replay dose fixed and deliver it in sleep blocks. I'll read the thalamus notes, H29d results, and the existing experiment code before changing anything.H29d replays after every 128-byte chunk. H31 keeps that dose and that sampler, and only changes when the replay steps run. I'll check the E3M hash and the GPU quota, then add the sleep-block path.The live path matches H29d when the sleep block is 1. Next I’ll add the report writer and the schedule check, then run the CPU smoke tests.The smoke matched H29d exactly. I'll refresh the dataset copy so it includes the report edit, then check the quota and upload.H31 is in `results/h31/RESULTS.md` and `results/h31/results.json`. Replay dose is fixed at 23,438 steps on E3M. Total GPU time is 1.4398 hours. No kernel was rerun.

E3M sha256 is `72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac`.

Smoke, first 8,192 bytes, CPU, lr 3e-4, replay batch 8, seed 0: `h29d_enwik8.py` and `h31_sleep.py --sleep-block 1` both scored 42063.298164869797 bits. The absolute difference is 0. With `--sleep-block 16`, replay steps are 64, the same as the interleaved run, and the bits are finite (44293.816701635013). For B = 4 over 12 chunks the run is `LLLLRRRRLLLLRRRRLLLLRRRR`: four scored live chunks, then four replay steps on the bytes seen so far, three times.

| Arm | Bits/byte | Replay steps | Wall seconds | GPU |
|---|---:|---:|---:|---|
| inter | 1.958153 | 23438 | 1087.922 | Tesla T4 |
| sleep_64 | 2.001622 | 23438 | 1141.211 | Tesla T4 |
| sleep_1k | 2.090683 | 23438 | 1144.753 | Tesla T4 |
| sleep_16k | 2.192371 | 23438 | 1172.649 | Tesla T4 |
| noreplay | 2.197983 | 0 | 446.571 | Tesla T4 |

Gaps against `inter`: `sleep_64` 0.043469, `sleep_1k` 0.132530, `sleep_16k` 0.234218, `noreplay` 0.239830.

Morning after, 128 KB awake stretches, mean bits/byte after a sleep minus the stretch before it: `sleep_1k` −0.076445 over 21 sleeps, `sleep_16k` −0.154965 over 1 sleep.

The bytes before the first sleep match `noreplay` exactly: 64 chunks for `sleep_64`, 1,024 for `sleep_1k`, and 16,384 for `sleep_16k`.

Kernels, all version 1, private GPU, no internet: `ezzzzzx/h31-inter`, `ezzzzzx/h31-sleep-64`, `ezzzzzx/h31-sleep-1k`, `ezzzzzx/h31-sleep-16k`, `ezzzzzx/h31-noreplay`. Code dataset `ezzzzzx/pdp-code-h31`. Quota before the push was 18.63 GPU hours remaining. It is 17.19 now.
