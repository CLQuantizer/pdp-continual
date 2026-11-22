I'll start by reading the project instructions and the handoff so I follow them exactly.The handoff is a Kaggle run of W9d plus paper registration. I'll read the kernel README and how papers get registered before I start anything.W9d is a scored Kaggle run plus five paper registrations. I'll check the smoke gate, the dispatcher, and how papers are written into D1 before starting.The layout check passed and there is no dispatch state yet. I'll confirm the lock is free, then start the kernels and look up the five papers.The lock is free and Codex is still on the H34g smoke, so W9d can take three of the five kernel slots. I'll start the dispatcher and check whether those five papers are already in the index.Three CPU kernels are running and two account slots are still free. The five papers are missing from the index, so I'll register them and store the open PDFs.The five rows are in D1 and the PDFs are in R2. Two files hid their titles in compressed streams, so I'll extract the text and confirm each one is the right paper.W9d scored all 48 lives. Every kernel finished on the first attempt, and the summarizer reports `lives=48/48`. Nothing was committed.

| T14 gap | 84 | 85 | 86 | 87 | 88 | 89 | 90 | 91 | 92 | 93 | 94 | 95 | Mean | Threshold <= | Count <= | Required count | N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pressure_retina − fixed_random | −0.157683737 | −0.266394831 | −0.219905892 | −0.154909404 | −0.048387751 | 0.067053846 | 0.023431553 | −0.009372135 | −0.057801169 | −0.413476455 | −0.215898636 | 0.081886240 | −0.114288198 | −0.070000000 | 6 | 8 | 12 |
| pressure_retina − fixed_night | −0.071086175 | 0.037117514 | 0.046014008 | 0.050119701 | 0.017946530 | 0.089264838 | 0.159881479 | 0.027301771 | −0.050854614 | −0.026470115 | 0.087552784 | −0.059687326 | 0.025591700 | 0.050000000 | 8 | NA | 12 |

| Mean threshold | Mean | Comparison | Threshold | Meets (0/1) | N |
|---|---|---|---|---|---|
| pressure_retina − fixed_night | 0.025591700 | <= | 0.050000000 | 1 | 12 |
| pressure_retina night_sleep_share | 0.620284397 | >= | 0.600000000 | 1 | 12 |

Kernels `ezzzzzx/w9d-s84` through `ezzzzzx/w9d-s95` are complete, one attempt each. No push failed and no kernel was retried. `results/w9d/results.json` has `missing: []`. Wall times were 230–402 seconds.

Five papers were missing from D1 `pdp-papers-index`. Each is now `checked` / `stored`, and an R2 read-back matched the uploaded sha256:

| cite_key | record | sha256 |
|---|---|---|
| `goyal2017` | arXiv 1706.02677v2 | `761b87763b59e48c090d0ace419613ae18040acbcbccf1973a25ec5b744b6f33` |
| `de2020` | NeurIPS 2020, arXiv 2002.10444v3 | `88e051adc4e07a0543134f0e359f6cba5a121597c4888336a733f033c315eb83` |
| `bachlechner2020` | arXiv 2003.04887v2 | `54eea1741fad2e7a20e52f55c64b34977f1206c5e3c8e14804ea5385c299c070` |
| `huang2017` | CVPR 2017, doi 10.1109/cvpr.2017.243, arXiv 1608.06993v5 | `b55aa1adbdf07f731daa84b94d23103d1eb22d1821a556b80212debee69b096d` |
| `baranes2013` | Robotics and Autonomous Systems 61(1), doi 10.1016/j.robot.2012.05.008, arXiv 1301.4862v1 | `98eb51d10e5821362fcb195b93c0ccfce93f405acd31e3d97b92179ac58595fc` |
