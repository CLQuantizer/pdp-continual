I'll start by reading the project instructions and the handoff so I follow them exactly.The handoff is W8-0: curiosity on real photos through a dotted window, scored by a frozen ResNet-18. I'll read the W7 design and code before writing anything.Load is under 1.5, so this can run on the VPS. Torch is installed, torchvision is not. I'll match a torchvision install to the existing torch and then build the camera.ResNet-18 on this machine is about 13 images a second, so the eye pass fits the budget. Synset indices match Imagenette. Next I’ll download the images and write the camera script.The luminance weights summed to just under 1 in floating point, so a flat grey of 32 fell into the wrong bin. I’ll quantize with integer BT.601 weights instead.W7 seed 10 matches exactly. Load is 0.50, so the full run stays on this machine.W8-0 is scored on all 500 Imagenette validation images. The run stayed on this machine: load 0.36, one CPU process, two threads, 1244 seconds. No kernel. Nothing was committed.

At L=16 the 10-way accuracies are:

| policy | accuracy | log-loss (nats) | coverage |
|---|---:|---:|---:|
| full | 0.978 | 0.057 | 1.000 |
| none | 0.100 | 3.384 | 0.000 |
| still | 0.228 | 3.236 | 0.250 |
| random | 0.672 | 1.028 | 0.610 |
| sweep | 0.984 | 0.053 | 0.997 |
| curious_prog | 0.918 | 0.243 | 0.805 |
| curious_surp | 0.918 | 0.243 | 0.805 |

Paired differences, 95% bootstrap CI, 10,000 resamples, rng 0:

| contrast | accuracy | CI | log-loss | CI |
|---|---:|---|---:|---|
| random − still | +0.444 | [0.398, 0.492] | −2.208 | [−2.394, −2.025] |
| curious_prog − random | +0.246 | [0.204, 0.288] | −0.785 | [−0.908, −0.665] |

full − still accuracy is 0.750. The same quantities at L=4, 8, 32 and 64 are in `results/w8/RESULTS.md`. curious_prog and curious_surp match on every image at every L.

Smoke: unoccluded 10-way accuracy is 0.978 (1000-way 0.666). W7 seed 10, 60 rows, max absolute error 0. The 20-image timing projected 1264 seconds, inside the 90-minute budget, so the set stayed at 50 images per class.

Files: `w8_camera.py`, `results/w8/RESULTS.md`, `results/w8/results.json`, and nine strips in `internal/w8_demo/`. Image 0's curious strip is the fish, with the mean-colour holes closing from L=1 to L=64.
