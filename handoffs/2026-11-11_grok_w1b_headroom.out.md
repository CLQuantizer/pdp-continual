I'll read the project instructions and the handoff, then follow the handoff exactly.The handoff is W1b: add three channels and remeasure headroom. I'll read the W1-0 results, the learner, and the Kaggle setup before changing anything.W1-0 is a four-channel headroom run. I'll inspect the DNA sample and the Kaggle push pattern, then add the three new channels.W1-0's four curves stay as they are. I'll add DNA, noise, and dark, smoke-test the copied learner on CPU, then run one private GPU kernel.The DNA file matches the Kaggle copy. Next I’ll copy the W1 learner and add the three channel builders.The copied search matches W1-0 (headroom 0.039409). Next is the CPU smoke test, before any kernel.Smoke matched W1-0 exactly, and dark’s held-out bits fell. I’ll upload the code and start the private GPU kernel.The kernel log stream ended. I’ll confirm it finished cleanly and pull the outputs.W1b headroom is **0.080936 bits/byte** on all 7 channels and **0.039979 bits/byte** on the 5 learnable channels. Numbers are in `results/w1b_headroom/RESULTS.md` and `results/w1b_headroom/results.json`. No verdict. Nothing was committed.

Bible, chess, squad, and btc are the W1-0 curves and were not rerun. DNA, noise, and dark each use a fresh learner (lr 3e-4, replay batch 8, seed 0), a 2,000,000-byte training prefix, and the next 100,000 bytes held out.

Full set, budget 3,500,000 bytes. Uniform mean L_c(500000) is 2.647814. The oracle is 2.566878. One allocation hits that minimum:

| bible | chess | squad | btc | dna | noise | dark |
|---:|---:|---:|---:|---:|---:|---:|
| 1000000 | 1000000 | 1000000 | 250000 | 125000 | 0 | 125000 |

Learnable set (noise and dark left out), budget 2,500,000 bytes. Uniform is 2.077059. The oracle is 2.037080. Allocation: bible 500000, chess 500000, squad 1000000, btc 250000, dna 250000.

Learnable headroom is below 0.05. Most of the gain comes from avoiding noise and dark rather than from choosing between learnable streams.

Held-out bits/byte at 500,000 bytes: dna 1.892069, noise 8.149331, dark 0.000076. On the training bytes, xz -9e and PPMd order 16 are 0.653744 and 0.622920 for dna, 8.000640 and 8.172264 for noise, and 0.001664 and 0.002968 for dark.

Smoke on 4,096 Bible bytes matched W1-0 exactly: 19362.707616529544 bits. Dark held-out bits fell from 8064.478 to 9.046. Noise and dark bits were finite.

`ezzzzzx/w1b-headroom` version 1 ran once on a Tesla T4 and completed. Elapsed 2316.191 seconds on the session log clock, 0.6434 GPU hours.
