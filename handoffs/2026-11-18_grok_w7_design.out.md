I'll read the project instructions and the handoff, then follow that handoff exactly.The handoff is a W7 design plus a short CPU headroom check. I'll read the curiosity rules, the roadmap's world sections, and the existing world and headroom code before writing anything.The world series scores a symbol stream the cortex can read. I'll look at how W6 and the headroom checks define that stream so the sketch matches it.The world has to make motion matter: one view must leave the logo unread, and a generic learner must not be handed the two layers. I'll sketch that and measure still, random walk, and sweep.The shift maps were shaped (32, 1) and (1, 32), so indexing the logo with the hole mask failed. I'll broadcast them to full frames.The sketch shows a real gap for a learner that can register the logo, and none for a view table. I'll write that up with the measured numbers.W7 has headroom if the learner registers the logo, and none if it only memorizes views. The draft is `internal/w7_design.md`. The numbers are from `w7_sketch.py` (one process, 6.6 s).

The world is a fixed 32×32 logo behind a soft dot field. Each step slides the logo by 1 pixel and the dots by 2. A glance is the whole frame, as a tape of 78 symbols: greys 0–7, a hole symbol, the action, and the viewpoint, with the viewpoint repeated on every row. The probe asks for the unoccluded pixels at 32 held-out viewpoints. Holes are not scored.

Integrator probe, bits per grey (seeds 0–2). Lower means the logo was read.

**Life 25**

| Opaque | Still | Random walk | Sweep |
|---|---|---|---|
| 512/1024 | 1.602 ± 0.010 | 0.791 ± 0.053 | 0.349 ± 0.001 |
| 768/1024 | 2.305 ± 0.009 | 1.533 ± 0.080 | 0.628 ± 0.004 |
| 922/1024 | 2.734 ± 0.026 | 2.255 ± 0.090 | 1.238 ± 0.046 |

At 768/1024 the walk beats standing still by 0.77 bits and the sweep beats the walk by 0.91. That is the operating point the draft recommends. At life 64 and 512/1024 the walk is already within 0.17 of the sweep.

A view table keyed by viewpoint and raster cell stays at the prior, 3.000 bits, for standing still, and between 2.92 and 2.97 for every other policy. A curious policy that follows the integrator's surprise lands between the walk and the sweep at life 25 and ties the sweep at life 64. Its view-table score stays at the prior too.

Two checks are in the same run. If the dots are glued to the logo, every policy scores 0.186 on the probe and still leaves three quarters of the logo unread (logo bits 2.296). A one-integer shift search on the sweep recovers the true shifts on all three seeds.

Nothing was committed. `docs/` and `.env` were not touched.
