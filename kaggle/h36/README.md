H36 uses per-byte threshold feedback for entropy/gain/progress: target .25,
step .01 times the EMA(.999) score standard deviation, strict score > theta,
theta initialized to the first finite score. Entropy deciles retain the causal
4096-observation window. Running statistics stay frozen on clean tests.

Gain/progress buckets with fewer than eight chosen samples are unscored.
Each such byte is chosen with probability .25 using default_rng(3000 + seed),
and is excluded from theta and running score-standard-deviation updates.
Until a finite score occurs, every byte uses that startup rule. The entropy
window still observes every byte. These are the orchestrator's startup rules;
the feedback controller and model updates are unchanged.

Foreground checks 3–5 and 64 KB checks on H35 build(0) for all four rules:

    .venv/bin/python h36_curious_patching.py --budget-smoke

A subset can be checked with --smoke-rules fixed entropy; this never sets
smoke ok=true. --smoke retains the original complete smoke sequence.

Offline code/kernel snapshot refresh and summary:

    .venv/bin/python kaggle/h36/dispatch.py --prepare-only
    .venv/bin/python kaggle/h36/summarize.py

The snapshot includes h35_corpus.py; kernels call h35_corpus.build(seed).
The dispatcher without --prepare-only is for the orchestrator's network runner.
No network calls, kernel launches, or commits occurred in this job.
