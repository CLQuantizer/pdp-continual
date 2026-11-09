# H20 floors: code length of the new world before any learner

Run 12:00 UTC+8, 9 Nov 2026 on the box, `python3 h20_world.py` (~8 s, 6 jobs). Pre-registration: `docs/h20_world.md`.
N = 524 288 symbols, 41-symbol alphabet (uniform 5.358 bits). Bits per symbol, prequential except the byte compressors
(full compressed size × 8 / N). Mix = fixed-share (α = 10⁻³) mixture of KT orders 0–8; switch = first 1 024 symbols
after each process switch. Per process = mix bits on G / C1 / C7 / E30 / E110 / M7 / M37.

| World | Seed | KT0 | KT2 | Mix | Mix switch | gzip | bz2 | lzma | Per process (mix) |
|---|---|---|---|---|---|---|---|---|---|
| slow | 6 | 4.830 | 1.614 | 1.577 | 1.724 | 1.941 | 1.032 | 0.678 | 0.83 / 3.24 / 3.31 / 1.19 / 1.13 / 0.40 / 0.50 |
| slow | 7 | 4.865 | 1.437 | 1.395 | 1.493 | 1.765 | 0.981 | 0.685 | 0.84 / 3.24 / 3.31 / 1.18 / 1.15 / 0.39 / 0.50 |
| slow | 8 | 5.064 | 1.553 | 1.468 | 1.603 | 1.842 | 0.844 | 0.663 | 0.78 / 3.20 / 3.34 / 1.20 / 1.19 / 0.38 / 0.54 |
| fast | 6 | 4.650 | 1.609 | 1.582 | 1.596 | 2.129 | 1.100 | 0.774 | 0.90 / 3.24 / 3.32 / 1.19 / 1.15 / 0.39 / 0.50 |
| fast | 7 | 4.682 | 1.522 | 1.487 | 1.502 | 2.007 | 1.071 | 0.800 | 0.88 / 3.25 / 3.32 / 1.19 / 1.14 / 0.38 / 0.54 |
| fast | 8 | 4.716 | 1.598 | 1.549 | 1.564 | 2.101 | 1.083 | 0.799 | 0.86 / 3.22 / 3.33 / 1.19 / 1.15 / 0.42 / 0.54 |

**Best floor = lzma: 0.675 ± 0.011 (slow), 0.791 ± 0.015 (fast).** A learner on this world must beat it by 0.10 in
every seed (see the bar). Short-context models are stuck on the algorithmic processes (counters 3.2–3.3 bits, CA
1.15–1.2) where the generator's entropy is 0 and ≈ 0.044. The KT mixture is a deliberately simple context model;
a PPM/CTW floor is stronger and is added before the first learner is scored; the bar uses the lowest floor.
