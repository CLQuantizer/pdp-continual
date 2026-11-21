# Run the W9d kernels (grok, Kaggle CPU), and register papers

Read `/root/pdp/CLAUDE.md` and `kaggle/w9d/README.md`. Codex implemented W9d and the smoke gate passed. Token from `.env` via `set -a; . ./.env; set +a` (never print it).
Run the dispatcher exactly as the README says in the foreground (≤ 3 W9d at once, account ≤ 5 running; codex will stage H34g kernels later today, so leave room),
rerun once on failure (it resumes), then run the summarizer. Report the scoring table at the top of `results/w9d/RESULTS.md`, kernel ids and failures.

Papers: register in D1 `pdp-papers-index` if missing, as CLAUDE.md says:
- Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (arXiv 1706.02677)
- De & Smith 2020, "Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks" (SkipInit; arXiv 2002.10444)
- Bachlechner et al. 2020, "ReZero is All You Need: Fast Convergence at Large Depth" (arXiv 2003.04887)
- Huang et al. 2017, "Densely Connected Convolutional Networks" (arXiv 1608.06993)
- Baranes & Oudeyer 2013, "Active learning of inverse models with intrinsically motivated goal exploration in robots" (Robotics and Autonomous Systems 61(1))

Do not edit code, bars or docs. Never print secrets. Do not commit.
