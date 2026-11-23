# Run the W10 kernels, then draft the depth essay (grok)

Read `/root/pdp/CLAUDE.md` first.

## 1. W10 kernels (Kaggle CPU)
Read `kaggle/w10/README.md`. Token from `.env` via `set -a; . ./.env; set +a` (never print it). Run `.venv/bin/python -B kaggle/w10/dispatch.py` in the foreground (≤ 3 at once,
account ≤ 5), rerun once on failure (it resumes), then the W10 summarizer. Report the scoring table at the top of `results/w10/RESULTS.md`, kernel ids, failures.
While the kernels run, do part 2.

## 2. Draft an essay: "Own. Wait. See." (internal draft only)
Write `internal/essay_depth_draft.md`. It goes on the public site only after the orchestrator edits it; do not put it under `docs/` and do not publish.

**Model the form on** https://mywords.io/transformer-syntax/ ("Split. Stack. Residual."): fetch and read it first. Note its length, how it opens with a question, how each
idea gets one short section with one picture-in-words, and how plain the sentences are. Match that, do not copy it.

**Subject:** how to make depth pay in a network that learns without backpropagation. The story, in order:
1. The question (15 Nov): backprop nets got deep with residual connections. What does a network get when every layer learns only from local signals?
2. The setting in one paragraph: a stack of small recurrent layers reading text byte by byte (enwik8), each learning online with an exact local trace rule, nothing crossing
   a layer. Lower is better: bits per byte. Every result is two seeds, pre-registered bars.
3. What failed, briefly, with numbers: summing layers' outputs and giving each layer the stack's leftover error (H34a, H34a2, H34a3: worse with depth); growing the stack
   one layer at a time (H34c: +0.14); random feedback of the stack's error (H34d: +0.28); distilling the stack's prediction into each layer (H34e: +0.05). The pattern:
   every signal that makes a layer answer for the whole stack's error hurt.
4. **Own.** Every layer is a complete predictor on its own loss; a gate mixes them (H34b: 1 → 6 layers, 3.275 → 3.081 on the tail; depth beats a six-times-wider single
   layer; a learned gate barely beats fixed equal weights).
5. **Wait.** Start the gate on the first layer only, so the stack starts as one layer and the upper layers get a say when they are useful (H34f: whole stream −0.36 at 4
   layers, −0.75 at 6; tail unchanged). The same idea as zero-γ, SkipInit and ReZero, found again without a backward pass.
6. **See.** Give each layer the mean of all earlier layers' features, not only the layer below (H34g/H34h: −0.062 at 6 layers; with it, depth pays again at 8: tail curve
   4, 6, 8 layers = 3.056, 3.016, 2.990, against 3.093, 3.078, 3.069 without). DenseNet's idea, made size-preserving by averaging.
7. The ablation that separates them (H34h): dense without the zero gate has the same tail (−0.006) but pays +0.68 over the whole stream. **Wait fixes the start; See fixes
   where it ends.** Own makes depth possible at all.
8. One closing paragraph: what this suggests about local learning (responsibility per layer, admission when useful, access to everything below), and what is not known yet
   (longer streams, other data, real cross-layer credit). No hype; no claims beyond the numbers.

Take every number from `results/h34b/RESULTS.md`, `results/h34c/`, `results/h34d/`, `results/h34e/`, `results/h34f/`, `results/h34g/`, `results/h34h/` and `docs/loop.md`; check each
against the files and list them in a table at the end of the draft ("numbers used", with the file each comes from). Byline "*By Ez*". Times UTC+8. No hostnames, paths
or worker names in the essay body. About 1,200–1,800 words. Cite the backprop-era papers by author and year (He et al. 2016; Lee et al. 2015 deeply supervised nets;
Goyal et al. 2017; De & Smith 2020; Bachlechner et al. 2020; Huang et al. 2017).

Files you may create: `internal/essay_depth_draft.md`, `kaggle/w10/**` run outputs, `results/w10/**`. Do not edit code, bars, `docs/`. Never print secrets. Do not commit.
