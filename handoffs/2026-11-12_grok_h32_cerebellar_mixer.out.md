I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is a cerebellar mixer experiment. I'll read the cited sources and the H26/H29 code before writing anything.The cortex has to stay the H26 uniform run. I'll check that a frozen mixer matches its bits, then implement the experiment.The cortex stays the H26 uniform run. Count models and the mixer sit beside it, and the mixer does not train the cortex. I'll write that and smoke-test it on CPU before any kernel.The cortex matches H26 exactly on 8,192 bytes. The frozen mixer is off by float32 renormalization, so I'll score the initial weights with the exact identity `softmax(log p) = p` and keep the real softmax gradient for learning.Smoke passed: the cortex matches H26 exactly, and a frozen mixer matches the cortex exactly. Next I’ll check the GPU quota and push the private kernel.Auto-compacting conversation (80% full)...
Conversation compacted.
The smoke checks passed and the GPU quota is still above the stop line. I will confirm the code copies, then create the private dataset and push the kernel.The kernel is running as version 1. Logs are still empty while the session starts. I am watching it until it finishes or fails.The kernel is running. I will check that the results assembler matches the kernel output so the write-up is ready when the job finishes.The H32 kernel is running on Kaggle. I will write the results once it finishes.

- Private code dataset: `ezzzzzx/pdp-code-h32`
- Kernel: `ezzzzzx/h32-mixer`, version 1, GPU, internet off, timeout 3600 seconds
- Quota before the push: 16.22 GPU hours remaining (refresh 2026-11-14 00:00 UTC)
- Status: running
- CPU smoke already passed: cortex and frozen mixer both match the H26 uniform 8192-byte bits exactly (difference 0)

The kernel ended in error. I am reading the log to find the failure before deciding on the one allowed rerun.The cortex finished and matched H26 (1.958153 bits/byte). The run then died because the GPU image has no `pyppmd`. I am checking how H29 got that package and how to ship it offline for the one allowed rerun.The scored stream finished and the cortex matched H26. The image is Python 3.13 and has no `pyppmd`, so the run died after that. I will ship the 1.3.1 wheel in the dataset and fail fast if it does not load, then use the one allowed rerun.The first kernel run finished the scored stream, then stopped because the GPU image has no `pyppmd`. Version 2 is the one allowed rerun, and it is running now.

Version 1 used 0.38 GPU hours. The cortex reached 1.958153 bits/byte on E3M, the same figure as the H26 uniform run, and the mixer reached the end of the 3,000,000 bytes. The traceback is `ModuleNotFoundError: No module named 'pyppmd'` on Python 3.13. The code dataset now includes the `pyppmd` 1.3.1 wheel, and the kernel loads it and checks a 3-byte roundtrip before the scored stream. Quota before this push was 15.84 GPU hours remaining. Kernel `ezzzzzx/h32-mixer` version 2 is running with the same 3600-second cap. I will write the results when it finishes.

The rerun finished. I am downloading the kernel output and checking the cortex bits before writing the results.The scored H32 run is in `results/h32/RESULTS.md` and `results/h32/results.json`. Kernel `ezzzzzx/h32-mixer` version 2 completed on a Tesla T4. The cortex bits match the H26 uniform E3M figure exactly.

| Arm | Bits/byte | Bits |
|---|---:|---:|
| gru | 1.958153 | 5874459.001 |
| count_1 | 3.876779 | 11630336.268 |
| count_2 | 3.151852 | 9455557.022 |
| count_3 | 2.810473 | 8431418.347 |
| count_4 | 2.956180 | 8868540.958 |
| mix | 1.846412 | 5539237.045 |
| PPMd order 16 | 1.923685 | 5771056.000 |

mix − gru = −0.111741 bits/byte. mix − PPMd = −0.077273 bits/byte. Those gaps are recorded and not scored in the write-up. The lowest count order is 3, at 2.810473 bits/byte. The GRU minus the published H26 uniform bits is 0. E3M sha256 is `72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac`.

Version 2 log clock is 1344.241 seconds (0.3734 GPU hours). Replay steps are 23438. PPMd is variant H, order 16, 721382 compressed bytes, and the incremental slices match the one-shot length. The CPU smoke matched H26 on 8192 bytes with both absolute gaps at 0.

Version 1 reached the same cortex figure and then stopped because the Python 3.13 image has no `pyppmd`. Version 2 loads `pyppmd` 1.3.1 from the private code dataset before the stream. Both runs together used 0.75 GPU hours. 15.47 GPU hours remain.
