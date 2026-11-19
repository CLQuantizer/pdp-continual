# W8-0: the curious camera, on real photographs. Does task-free curiosity help a frozen eye recognise what is behind a dotted window? (grok; VPS or Kaggle)

Read `/root/pdp/CLAUDE.md`, `internal/w7_design.md`, `w7_world.py`, `w7_run.py` and `results/w7/RESULTS.md`.

## Why
W7-0 showed that, on a synthetic 32×32 logo, an agent driven by its own compression progress learns to move and so reveals a scene that standing
still hides. W8-0 moves the same world onto real photographs, and asks a question an outsider can see: after L glances, can a frozen, pretrained
image classifier name what is behind the dotted window? The curiosity signal never sees the label or the classifier. If it still helps recognition,
curiosity about pixels is useful to an eye.

## Data and eye
- Images: **Imagenette** (fastai; 10 easy ImageNet classes), `imagenette2-160` validation split. Take the first 50 images per class in sorted
  filename order (500 images). Resize the short side to 224 and centre-crop 224×224, RGB. Record the file list.
- Eye: torchvision **ResNet-18** with ImageNet weights (`IMAGENET1K_V1`), frozen, eval mode. Predict among the 10 Imagenette classes only:
  take the 10 logits of their ImageNet synsets (list them in RESULTS.md) and softmax over those 10.
- Source both from the VPS (it has network) or from Kaggle public datasets and models (internet off). Your choice; record which.

## World: W7's geometry on a photograph
- Cut the 224×224 image into a 32×32 grid of 7×7-pixel cells. The cell grid is the W7 raster: scene cell (sy, sx) is a 7×7 RGB block.
- Dots: W7's soft dot field (`make_field`, `occluder` from `w7_world.py`) with **seed = image index** (0–499), opacity **768/1024**.
- Viewpoint, actions and parallax exactly as W7 (scene shift 1, dot shift 2, torus 32). A glance shows, for each viewport cell, the scene block if
  that dot cell is clear, else a hole.
- The **pixel model** the curious agent learns is W7's integrator on a grey proxy: each cell's mean luminance quantised to 8 levels (0–7, equal
  bins over 0–255). Use the W7 code unchanged on that grey raster, so `curious_prog` and `curious_surp` compute their rewards as in W7-0.
- The **reconstruction** the eye reads after L glances: every scene cell seen at least once holds its true RGB block; unseen cells hold the
  ImageNet mean colour. Apply the usual ImageNet normalisation, then the eye.

## Policies (from `w7_run.py`, unchanged): still, random, sweep, curious_prog, curious_surp. Seeds: the policy rng seed = image index.
Also report `full` (the eye on the unoccluded image) and `none` (nothing seen: all mean colour).

## Scores (per image, policy and L ∈ {4, 8, 16, 32, 64})
- Top-1 accuracy over the 10 classes, and log-loss of the true class (nats).
- Coverage (fraction of scene cells seen).
- Paired differences over the 500 images with a 95% bootstrap CI (10,000 resamples, rng 0).

## Demo output (for a later presentation; internal only)
For images 0, 1 and 2, save to `internal/w8_demo/` a PNG strip per policy (still, random, curious_prog) at L = 1, 4, 16, 64: the reconstruction,
with the eye's top class and its probability in the filename. Never put these under `docs/` or `site/`.

## Smoke (before the full run)
1. The ResNet-18 accuracy on the 500 unoccluded images (10-way) is printed. It should be about 0.9 or higher; if it is below 0.80, stop and report.
2. With the grey proxy replaced by W7's logo, W7-0's seed-10 numbers reproduce to 1e-9 (shows the world code is unchanged).
3. Image 0: print its grey raster at L = 0 and the curious_prog path's first 10 viewpoints.
4. Time 20 images × all policies; estimate the full run. The full run must fit in 90 minutes of CPU or one Kaggle GPU kernel of ≤ 1.5 hours;
   if not, use the first 30 images per class (300) and say so.

## Outputs
`results/w8/RESULTS.md` + `results.json`, numbers only, no verdict. **At the top, the scoring table:** at L = 16, accuracy for full, none, still,
random, sweep, curious_prog, curious_surp; random − still and curious_prog − random (accuracy and log-loss), each with its 95% CI. Then the same
at every L, coverage per policy, the synset list, the image list's source, walls, kernel ids if any.

## Pre-registered bars (orchestrator; do not change), at L = 16, 500 images
- **Precondition:** full accuracy ≥ 0.85 and still accuracy ≤ full − 0.20 (the window hides enough to matter). Else the bars are void.
1. **Moving reveals:** random − still accuracy ≥ +0.10, CI excluding 0.
2. **Curiosity helps the eye:** curious_prog − random accuracy ≥ +0.03 **and** log-loss difference < 0, both CIs excluding 0.
Report only: sweep, curious_surp, the other L.

## Rules
- About 3 hours of your time. Foreground runs. If on the VPS: one process, `torch.set_num_threads(2)`; check no other CPU-heavy job is running
  (`uptime` load < 1.5), else use a Kaggle kernel. GPU quota guard as CLAUDE.md says (`kaggle quota` first; ≤ 1.5 GPU hours here).
- Files you may create: `w8_camera.py` (import from `w7_world.py`/`w7_run.py`; do not edit them), `kaggle/w8/**`, `results/w8/**`, `data/w8/**`
  (gitignore data and weights), `internal/w8_demo/**`.
- Do not touch `docs/`, `audit.md`, other code, `.env`. Gates and bars are the orchestrator's: if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.
