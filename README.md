# pdp: can MNIST be learned without backprop?

Plain NumPy. Backprop as the baseline, against five learners that never send an
exact error backward through the network.

| Key | Method | What crosses layers backward |
|---|---|---|
| `bp` | Backprop MLP (784-256-256-10) | exact gradient, through the transposed forward weights |
| `fa` | Feedback alignment (Lillicrap 2016) | error, through **fixed random** matrices |
| `dfa` | Direct feedback alignment (Nøkland 2016) | output error, sent **straight to each layer** through fixed random matrices |
| `ff` | Forward-forward (Hinton 2022), 784-500-500, SymBa loss + homeostatic bias (`ff_orig`: original threshold loss) | **nothing**. Each layer has its own good-vs-bad objective |
| `hebb` | Competitive Hebbian layer (2000 units, soft winner-take-all + homeostasis) + ridge readout | **nothing**. The layer never sees labels |
| `rf` | Random fixed features (2000) + ridge readout | **nothing**. Only the readout learns |

The two ridge readouts keep only running sums (`Σhhᵀ`, `Σhyᵀ`). That makes them
online learners whose answer doesn't depend on the order of the data, which
matters for Split-MNIST.

## Protocols

- **standard**: a few epochs of shuffled minibatches, then test accuracy.
- **iid**: one pass, one example at a time, shuffled. The online control.
- **split**: one pass, one example at a time, digits arrive as 0/1 → 2/3 → 4/5 → 6/7 → 8/9.
  After every task, test on every task. Class-incremental (one 10-way output, no task
  identity at test time), with task-incremental accuracy also reported.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install numpy
.venv/bin/python bench.py                 # all methods, all protocols, seed 0
.venv/bin/python bench.py --seeds 0 1 2   # mean ± std
.venv/bin/python bench.py --tune          # batch-1 learning-rate sweep (validation split)
```

MNIST is downloaded once into `data/`. Results go to `results/RESULTS.md` and
`results/results.json`.

## Findings (seeds 0, 1, 2; mean ± std; full tables in `results/RESULTS.md`)

| Method | Standard (multi-epoch) | Single pass, batch 1 | Split-MNIST class-IL | Split-MNIST task-IL |
|---|---|---|---|---|
| Backprop | 98.43 ± 0.04 | 96.80 ± 0.20 | 19.65 ± 0.04 | 91.05 ± 0.93 |
| Feedback alignment | 97.75 ± 0.04 | 93.38 ± 0.82 | 19.38 ± 0.24 | 88.89 ± 3.58 |
| Direct feedback alignment | 97.79 ± 0.05 | 94.23 ± 0.98 | 19.10 ± 0.54 | 88.52 ± 4.59 |
| Forward-forward, threshold loss (original) | 97.23 ± 0.10 | 95.33 ± 0.25 | 12.22 ± 1.66 | 48.23 ± 4.20 |
| Forward-forward, SymBa + homeostasis | 98.40 ± 0.06 | 96.25 ± 0.10 | 20.09 ± 0.36 | 85.14 ± 3.14 |
| Competitive Hebbian + ridge | 96.39 ± 0.03 | 96.19 ± 0.08 | 60.91 ± 1.29 | 98.93 ± 0.05 |
| Random features + ridge | 95.50 ± 0.21 | 95.50 ± 0.21 | **95.50 ± 0.21** | **99.28 ± 0.07** |

Std is over three seeds, so treat it as a rough spread. Each seed changes
the initialisation and the data order.

1. On ordinary MNIST every method reaches 95–98.5%. Backprop is best, but by
   only ~1 point over FA/DFA and ~1.3 over forward-forward.
2. Every gradient-trained method (BP, FA, DFA, FF) fails Split-MNIST
   class-incremental in the same way: after each task it scores ≈0% on all
   earlier ones. Removing the backward pass did not remove forgetting.
   Forgetting comes from shared weights being overwritten by a loss that
   only sees the current classes.
3. What helps is *where* learning happens, not how errors travel. The ridge
   readout keeps running sums, so its answer is independent of data order.
   With fixed random features it ends Split-MNIST exactly where iid training
   does (95.3%). The Hebbian layer keeps learning during the stream, so its
   features drift away from the statistics the readout stored. It forgets
   partly (61%).
4. The original forward-forward loss kills its own first layer. It pushes
   goodness for negatives below a fixed threshold, but a negative differs
   from its positive in only the 10 label pixels. So the push lands on the
   shared image weights and biases, and ReLU units fall silent and never
   recover (no gradient while off). On Split-MNIST 84% of layer-1 units
   were dead by the end, leaving task-IL at chance (48 ± 4%). Two local
   fixes, together:
   - **SymBa loss** (Lee & Song 2023): `softplus(-α(g_pos − g_neg))` per
     image. Only the gap between positive and negative counts, so nothing
     pushes goodness down overall.
   - **Homeostatic bias**: each unit's bias drifts toward firing on 10% of
     inputs. It changes the bias directly, not through the gradient, so it
     can revive a silent unit.

   With both, forward-forward reaches 98.40% on standard MNIST (level with
   backprop, up from 97.23%), and layer-1 activity holds at 11–15%. On
   Split-MNIST it now behaves like backprop: it learns each new task (99%)
   and forgets the old ones (class-IL 20%). The fix removed an instability,
   not forgetting. That matches point 2.
5. Caveats: three seeds, small networks, light tuning. Hebbian here is a basic
   soft-WTA layer (no Krotov–Hopfield power rule, no trained nonlinear top
   layer), so it sits below the ~98.5% reported in that paper.

## Wake/sleep forward-forward (`sleep.py`)

While awake, FF learns from the stream (batch 1) and a **bounded fast
store** records it. Every 2000 examples it sleeps, on a timer and without
knowing task boundaries. During sleep input is off, it trains on
class-balanced replay from the store (positives), the wrong label it finds
most plausible serves as the negative, weak weights are pruned
(downscaling), and homeostasis runs on replayed data. Full tables are in
`results/sleep/RESULTS.md`.

Split-MNIST class-IL, 3 seeds:

| Condition | Store (floats) | Class-IL | Task-IL | Weight sparsity |
|---|---|---|---|---|
| No sleep | 0 | 20.09 ± 0.36 | 85.14 ± 3.14 | 0% |
| 1 prototype/class | 15,680 | 80.19 ± 1.39 | 97.99 ± 0.17 | 38% |
| 10 prototypes/class | 156,800 | 89.95 ± 0.42 | 98.77 ± 0.08 | 28% |
| 10 prototypes/class, no downscaling | 156,800 | 90.42 ± 0.18 | 98.76 ± 0.04 | 0% |
| 20 raw images/class (same memory) | 156,800 | 88.60 ± 0.40 | 98.69 ± 0.12 | 29% |
| 50 prototypes/class | 784,000 | 91.23 ± 0.07 | 98.95 ± 0.15 | 23% |
| 100 raw images/class (same memory) | 784,000 | 93.21 ± 0.41 | 99.10 ± 0.06 | 27% |

1. **Sleep takes FF from 20% to 90% class-IL.** The learning rule and
   network are unchanged; the only difference is an offline replay phase.
   For reference, the original threshold-loss FF scored 12%, and random
   features + ridge (which fails the goal's criteria) 95.5%.
2. **Compressed memory works.** One mean + variance per class (15,680
   floats, the size of 20 images in total) already gives 80%. At small
   budgets, prototypes beat raw images of the same size (90.0 vs 88.6). At
   larger budgets raw images win (93.2 vs 91.2).
3. **Downscaling doesn't help memory, but it makes the weights sparse.**
   Accuracy is about the same either way (−0.5 points, within ~1 std),
   and 23–38% of weights end up exactly 0. So its role here is sparsity,
   not retention.
4. **Wake still damages old tasks; sleep repairs them.** Accuracy on 0/1
   dips to ~65–70% just after 2/3 is learned, then recovers to 97–99%
   after later sleeps. Evaluation happens at a task's end, up to 2000 wake
   steps after the last sleep, so these numbers include some unrepaired
   damage.
5. **Costs.** Sleep replays 3.2× as many examples as were seen awake (0.8× is enough: see below)
   (30 sleeps × 6400). The store is labelled by class, so the "hippocampus"
   is supervised. Fewer, shorter sleeps and an unsupervised store are not
   tested yet.

### How much sleep is needed (`sleep_cost.py`)

10 prototypes/class, 3 seeds, pruning off (pruning runs once per sleep,
so with it on, frequent sleeps also prune more: 52–58% zero weights at
every-500 against 15% at every-6000). Replay ratio = examples replayed
asleep per example seen awake. *Final* is measured right after the last
sleep. *Task-end avg* is class-IL over the digits seen so far at the end
of tasks 2–5, which fall anywhere in the sleep cycle. Full tables, with
pruning on and off, are in `results/sleep_cost*/RESULTS.md`.

| Replay ratio | Sleep every 500 (final / task-end) | Every 2000 | Every 6000 |
|---|---|---|---|
| 3.2× | 82.0 / 87.7 | 90.4 / 88.4 | 90.6 / 87.1 |
| 0.8× | 71.8 / 84.5 | **90.0 / 89.4** | 89.6 / 86.0 |
| 0.2× | 55.7 / 77.7 | 82.3 / 85.5 | 86.4 / 83.6 |
| 0.05× | 44.3 / 64.1 (0.064×) | 45.9 / 70.6 | 76.0 / 77.5 |

1. **Replay can drop 4× for free.** At 0.8× replay, sleeping every 2000
   keeps 90.0% final and 89.4% task-end, the best task-end score in the
   grid. The original 3.2× was overkill.
2. **Below ~0.2× it falls off.** At 0.2× it's 82–86% final; at 0.05× only
   rare, long sleeps (every 6000) hold up at ~76%.
3. **Frequent short sleeps are worst at every budget.** Sleeping every 500
   is lowest on both measures except task-end at 3.2×, where all schedules
   tie (87–88%). At the same total replay, fewer and longer sleeps are
   better. Why isn't established. One guess: a short sleep is too small a
   batch of all classes to undo the wake updates biased toward the
   current task.
4. **Part of the long-sleep advantage is timing.** On *final*,
   every-6000 benefits from a large sleep just before the test. On
   *task-end*, 2000 and 6000 are close, and at high budgets every schedule
   ties.

Recommended setting: sleep every 2000 with 1600 replays (0.8×).

### Dreams and Hinton's split (`dreams.py`)

Two changes to the 0.8× setting above, 3 seeds. Full tables are in
`results/dreams/RESULTS.md`.

- **Dreams**: no store. At each sleep the network imagines its own
  positives. Starting from unlabelled pixel noise (a running mean and
  variance of all pixels, 1,568 floats), it climbs log p(label | x) by
  gradient ascent on the pixels, for every label seen so far. The dreams
  are discarded after each sleep. The gradient runs through the frozen
  network, so dreaming uses a backward pass, but no weight learns from it.
- **Hinton's split**: wake trains positives only and sleep trains
  negatives only, using the original threshold loss since there's no
  pair to compare. The negative is either a replayed or dreamed image
  with the most plausible wrong label, or a dream with its own label
  (Hinton's version: the network's generations are the negatives).

| Condition | Sleep every | Store (floats) | Class-IL final | Task-end avg | Task-IL |
|---|---|---|---|---|---|
| No sleep | – | 0 | 20.09 ± 0.36 | – | 85.14 ± 3.14 |
| Prototype replay (reference) | 2000 | 156,800 | **89.99 ± 0.56** | **89.41 ± 0.57** | 98.82 ± 0.12 |
| Dreams as positives | 2000 | 1,568 | 26.94 ± 3.83 | 37.53 ± 1.36 | 90.23 ± 3.23 |
| Split, prototypes + wrong label | 2000 | 156,800 | 20.55 ± 3.74 | 22.37 ± 1.68 | 56.46 ± 5.50 |
| Split, dreams + wrong label | 2000 | 1,568 | 22.81 ± 9.32 | 20.77 ± 2.61 | 67.08 ± 2.41 |
| Split, dreams + own label (Hinton) | 2000 | 1,568 | 12.75 ± 2.84 | 15.58 ± 2.77 | 63.14 ± 5.02 |
| Split, prototypes + wrong label | **100** | 156,800 | 47.61 ± 4.59 | 69.47 ± 1.91 | 90.98 ± 2.91 |

1. **The network's own dreams barely help** (20% → 27% final, 38%
   task-end), and stay far from prototype replay (90%). The dreams are
   speckle with faint class structure (`results/dreams/dreams_*.png`):
   an average dream correlates 0.45 with the real average digit of its
   class, and 0.39 with the best *other* class. They're near-adversarial
   inputs that score high for a class without looking like it, so
   training on them protects the network's response to those inputs and
   not to real old digits. They're also already confident (p = 0.82), so
   they produce little gradient. On two-task smoke tests, fewer steps,
   starting from recent inputs, blurred steps and dreaming every 200
   examples all left 0–21% of 0/1. **Inverting a discriminative network
   doesn't give it a memory.** Dreams need a generative pathway (learned
   top-down weights) or something to start from (the prototype store).
2. **Hinton's split fails at every 2000.** Without the store, or with
   dreams as negatives, it can't even learn the current task well
   (task-IL 56–67%), and up to 53% of units die. Pushing positives up for
   2000 steps and then negatives down mostly moves the shared image
   weights back and forth. What separates labels is the image × label
   interaction, which a paired loss isolates within each batch and a
   time-split loss doesn't.
3. **The split needs fast alternation.** At the same ~0.8–1× replay,
   sleeping every 100 brings it to 69% task-end and 91% task-IL, but
   that's still well below paired replay, and 40% of units die (the
   negative-only phase silences units and homeostasis only acts awake).
   Hinton noted the phases can be separated only if the learning rate is
   small or they alternate quickly; this matches. It reverses the earlier
   finding that fewer, longer sleeps are better: here sleep is the only
   place discrimination is learned, so long wake stretches undo it.

### Top-down generative dreams (`dreams.py --set topdown`)

Helmholtz-style: each FF layer gets top-down weights `G` that predict the
layer below (the image, for layer 1) from its own activity. They are learned
**awake, locally**: each `G` only sees the two layers it connects, and no
gradient touches the recognition weights. The top layer keeps a small
class-conditional prior over its own codes (k code prototypes per class,
mean + variance). To dream a class: sample a code from its prior and run
it down through the `G`s. No image is ever stored. Every 2000, 0.8× replay,
3 seeds; dreams replace prototype + noise as sleep positives.

| Condition | Code prior (floats) | Generator memory (floats) | Class-IL final | Task-end avg | Forgetting |
|---|---|---|---|---|---|
| 10 prototypes/class (reference) | – | 156,800 (store) | **89.99 ± 0.56** | **89.41 ± 0.57** | 8 |
| 1 prototype/class | – | 15,680 (store) | 82.31 ± 0.92 | 81.18 ± 1.64 | 20 |
| Top-down, delta rule (fixed step), 10 codes/class | 100,000 | 392,000 | 31.63 ± 4.10 | 51.36 ± 3.31 | 84 |
| Top-down, count rule (step 1/activity), 10 codes/class | 100,000 | 392,000 | 34.57 ± 2.11 | 46.37 ± 1.76 | 81 |
| **Top-down, ridge rule, 10 codes/class** | 100,000 | 392,000 + 644,000 sums | **86.75 ± 1.36** | **85.81 ± 1.91** | 13 |
| Top-down from layer 2, ridge rule, 10 codes/class | 100,000 | 642,000 + 1,145,000 sums | 85.17 ± 0.41 | 84.62 ± 0.26 | 14 |
| Top-down, ridge rule, 1 code/class | 10,000 | 392,000 + 644,000 sums | 82.82 ± 0.85 | 84.00 ± 0.61 | 18 |

1. **A generative path makes real dreams.** With the ridge rule the final
   network dreams every digit recognisably, including 0/1 seen 48,000
   examples earlier (`results/topdown/dreams_gen1_k10_ridge_every2000.png`).
   An average dream correlates 0.97 with the real average digit of its
   class (inverting the FF net gave 0.45). Class-IL goes from 27% (inverted
   dreams) to **86.8%**, 3 points short of prototype replay.
2. **The generator forgets too, unless it only accumulates.** With a
   fixed-step delta rule the top-down weights are overwritten by each new
   task: old dreams turn into blends with the current digits, and class-IL
   is 32%. A per-unit 1/activity step (a running average, still local)
   didn't fix it (35%). The ridge rule keeps running sums of `h hᵀ` and
   `h xᵀ` awake and solves for `G` at the start of each sleep. Its answer
   doesn't depend on data order, so old codes keep decoding to old digits.
   It's the same lesson as the ridge readout in the main benchmark:
   **memory that only accumulates can't be overwritten.**
3. **A second level doesn't help** (85.2 vs 86.8). Layer 1 → image is
   enough on MNIST.
4. **It isn't cheaper than prototypes.** The ridge sums (644,000 floats
   for one level) plus `G` outweigh the 10-prototype store (156,800), and
   the solve is a dense O(H³) operation, though only in sleep. With 1 code
   per class it ties 1 prototype/class (82.8 vs 82.3 final, 84.0 vs 81.2
   task-end). What's gained is that nothing image-shaped is stored: the
   memory is a fixed-size statistic of the network's own activity,
   independent of how many classes or examples arrive, and the dreams are
   the network's own.

### Local generator rules without the solve (`dreams.py --set local`)

Can a local, accumulate-only rule replace the ridge solve and its 644k
sums? Two candidates, layer 1 → image, 10 codes/class, every 2000, 0.8×,
3 seeds:

- **Outstar** (Grossberg): unit j's top-down weights are the running
  activity-weighted mean of the image whenever j fires,
  `G_j += (h_j / N_j)(x − G_j)`. Each unit uses its own error, so units
  can't overwrite each other. Decoding is the activity-weighted average
  `Σ a_j G_j / Σ a_j`, with `a = h` or `h²`. Extra memory: one count per
  unit (500). No solve.
- **Contrastive wake-sleep**: awake, each synapse accumulates its own
  Hebbian sum `C_ij += h_i x_j`. Asleep, codes are sampled from the class
  prior and decoded, `x̂ = hG`, and each synapse takes local steps
  `G_ij += η (C_ij / n − mean(h_i x̂_j))`: Hebbian on memories of real data,
  anti-Hebbian on its own dreams. Its fixed point is the least-squares
  decoder, but under the prior's code statistics instead of the data's.
  No `hhᵀ`, no solve; extra memory `C`, the same size as `G` (393k).

| Generator rule | Extra memory (floats) | Solve | Class-IL final | Task-end avg | Forgetting |
|---|---|---|---|---|---|
| 10 prototypes/class (reference, no generator) | 156,800 store | – | **89.99 ± 0.56** | **89.41 ± 0.57** | 8 |
| Ridge (previous section) | 643,785 | O(H³) per sleep | 86.75 ± 1.36 | 85.81 ± 1.91 | 13 |
| **Contrastive wake-sleep, 100 steps/sleep** | 392,784 | none | **84.23 ± 1.06** | **81.63 ± 0.90** | 16 |
| Contrastive wake-sleep, 300 steps/sleep | 392,784 | none | 78.58 ± 2.60 | 74.94 ± 3.06 | 25 |
| Outstar, weight h² | 500 | none | 75.68 ± 1.53 | 80.48 ± 0.82 | 25 |
| Outstar, weight h | 500 | none | 72.05 ± 3.61 | 79.14 ± 1.14 | 31 |
| Outstar, weight h, 1 code/class | 500 | none | 69.75 ± 1.87 | 76.60 ± 1.95 | 34 |
| Delta rule (previous section, for scale) | 0 | none | 31.63 ± 4.10 | 51.36 ± 3.31 | 84 |

1. **The contrastive wake-sleep rule gets within 2.5 points of ridge**
   (84.2 vs 86.8 final; 81.6 vs 85.8 task-end) with only local updates and
   no solve. It drops the 251k `hhᵀ` matrix but still needs a per-synapse
   accumulator as big as the weights, so memory falls from 644k to 393k
   extra, not to near zero.
2. **More sleep iterations make it worse** (300 steps: 78.6). The rule
   converges to the least-squares decoder *for codes drawn from the prior*
   (a few prototypes per class plus diagonal noise), not for the codes real
   data produced. Some early stopping of that drift helps. Matching the
   prior to the real code statistics is the obvious next fix.
3. **Outstar meets the memory goal but loses 11–17 points.** With 500
   extra floats it can't be overwritten (each unit only averages), but the
   dreams are blurry class averages with almost no variety
   (`results/local/dreams_gen1_k10_outstar2_every2000.png`). A pure
   average gives up the cancellation between units that least squares uses
   to sharpen. Outstar is still 2.3× better than a delta rule, and on
   two-task smoke tests it looked as good as ridge (68% vs 64% of 0/1
   kept), which didn't hold over five tasks.
4. **Ranking by memory**: outstar (500) < prototypes (157k) <
   contrastive (393k) < ridge (644k); by accuracy the order nearly
   reverses. No local rule here beats a plain prototype store on both.

### Matching the code prior to real codes (`dreams.py --set prior`)

The contrastive rule's fixed point is the least-squares decoder *under the
prior's code statistics*. With a prototype + diagonal-noise prior those
differ from real codes, which is why more sleep steps hurt (84.2 → 78.6).
Two priors that match real codes better, every 2000, 0.8×, 3 seeds:

- **Reservoir of real codes**: k codes/class kept by reservoir sampling;
  dreams and the negative phase both sample them.
- **PPCA**: class mean + a rank-20 principal subspace learned online with
  Sanger's (local, Hebbian) rule, its variances and a residual variance.

| Condition | Prior (floats) | Generator extra (floats) | Class-IL final | Task-end avg | Forgetting |
|---|---|---|---|---|---|
| 10 prototypes/class (image store, reference) | 156,800 | – | **89.99 ± 0.56** | **89.41 ± 0.57** | 8 |
| 20 raw images/class (image store) | 156,800 | – | 87.69 ± 0.66 | 87.35 ± 1.93 | 12 |
| Ridge, 20 real codes/class | 100,000 | 643,785 | **89.60 ± 0.46** | **88.94 ± 1.08** | 10 |
| **Contrastive, 20 real codes/class, 100 steps** | 100,000 | 392,784 | **87.47 ± 0.82** | 86.42 ± 1.81 | 10 |
| Contrastive, 20 real codes/class, 300 steps | 100,000 | 392,784 | 87.53 ± 1.10 | 86.70 ± 0.74 | 11 |
| Contrastive, 10 real codes/class, 100 steps | 50,000 | 392,784 | 86.63 ± 0.33 | 86.77 ± 1.24 | 12 |
| Contrastive, PPCA rank 20, 100 steps | 105,210 | 392,784 | 85.14 ± 0.58 | 83.37 ± 1.15 | 14 |
| Contrastive, PPCA rank 20, 300 steps | 105,210 | 392,784 | 81.31 ± 1.53 | 78.73 ± 4.00 | 21 |
| Contrastive, dreams from prototypes, negatives from 10 real codes | 150,000 | 392,784 | 78.58 ± 3.77 | 77.55 ± 4.92 | 21 |
| Contrastive, prototype prior (previous section) | 100,000 | 392,784 | 84.23 ± 1.06 | 81.63 ± 0.90 | 16 |

1. **Real codes fix the fixed point.** With a reservoir of real codes the
   contrastive rule goes from 84.2 to **87.5%**, and running 3× more sleep
   steps no longer hurts (87.5 either way). The diagnosis was right: the
   damage came from the prior, not from the rule.
2. **It reaches ridge's earlier level, not ridge itself.** Contrastive
   with real codes (87.5) beats ridge with the old prior (86.8), but ridge
   with real codes rises too, to 89.6, statistically level with prototype
   replay (90.0). The local rule stays ~2 points under the solve on the
   same prior.
3. **Memory.** 10 real codes/class (50,000 floats) still gives 86.6%. A
   code is 500 floats against an image's 784, and code dreams beat raw
   images at less memory (87.5 at 100k vs 87.7 at 157k; 86.6 at 50k). But
   the generator's accumulator (393k) still has to be counted, so in total
   it isn't smaller than the 10-prototype image store.
4. **PPCA only half-fixes it** (85.1, and worse with more steps). Matching
   mean and top-20 covariance isn't enough: sampled codes are clipped at 0
   to stay ReLU-like, which changes their statistics, and real codes are
   sparse, not Gaussian.
5. **Dreams and negatives must come from the same prior.** Dreaming from
   prototypes while the negative phase uses real codes is worst (78.6):
   the generator is then fitted for codes it's never asked to decode.

### One number per synapse (`dreams.py --set fold`)

The contrastive rule keeps a Hebbian sum C next to every weight. Here C is
folded into G: awake, each synapse adds its Hebbian term straight into its
weight, `G += r_t h xᵀ` with `r_t = κ / (t·E|h|²)` (a 1/t step, so every
example counts about equally, as in a running mean). Asleep, the
anti-Hebbian term owed for that wake (total weight Σ r_t) is paid back on
decoded prior codes, `G -= η·mean(z x̂ᵀ)`. The fixed point when the data is
stationary is the same as the contrastive rule's. Extra memory: two scalars.
Prior = 20 real codes/class, every 2000 / 0.8×, 3 seeds.

| Condition | Generator extra (floats) | Class-IL final | Task-end avg | Recon MSE | Dream corr same / other |
|---|---|---|---|---|---|
| Contrastive with accumulator C (reference) | 392,784 | **87.47 ± 0.82** | **86.42 ± 1.81** | 0.047 | 0.98 / 0.82 |
| Folded, κ = 0.25 | 2 | 47.15 ± 4.42 | 61.72 ± 5.39 | 0.097 | 0.72 / 0.76 |
| Folded, κ = 1 | 2 | 51.85 ± 2.82 | 66.50 ± 2.40 | 0.090 | 0.75 / 0.75 |
| **Folded, κ = 4** | 2 | **65.43 ± 1.50** | 72.62 ± 2.88 | 0.140 | 0.79 / 0.78 |
| Folded, κ = 16 | 2 | 50.73 ± 2.10 | 61.50 ± 2.26 | 0.217 | 0.75 / 0.74 |

1. **Folding C in costs 22 points** at the best κ (87.5 → 65.4), and the
   dreams lose their class: each class's mean dream correlates with the
   wrong digit as well as the right one.
2. **No κ works, for a predictable reason.** With a 1/t step, the weight
   an old example keeps along a code direction with variance λ scales like
   (s/t)^(κλ/E|h|²). Weak directions (small κλ) barely move from where the
   first examples put them, so they never converge. Strong directions
   (large κλ) weight recent data, i.e. they forget. Small κ fails the
   first way, large κ the second, and κ = 4 is only the least-bad
   compromise. Reconstruction error grows steadily with κ.
3. **What C actually buys.** Keeping C lets each sleep re-solve against
   the mean of *all* past data, with as many iterations as needed and from
   any starting point. Folded into G, the past survives only as the
   current solution, so every new wake's Hebbian term is mixed into a
   half-converged decoder. Being accumulate-only is what protects the old
   tasks, and it needs its own storage: a weight that is also being
   corrected can't also be the running sum.

### A cheap accumulator (`dreams.py --set cheapc`)

Keep C, but make it small. Rewrite C/n = diag(N/n)·F, where N_i is unit
i's total activity and F_ij is the activity-weighted mean of pixel j
whenever unit i fired (Grossberg's outstar weights,
`F_i += (h_i/N_i)(x − F_i)`). This is exact in float. F lies in [0, 1]
however many examples have been seen, so it can be quantised without a
scale. Two ways to shrink it:
- **b bits per entry** with stochastic rounding, which stays unbiased even
  when the step is far below one level;
- **top-k per row:** each unit keeps only its k largest pixels (16-bit
  index each), and a new pixel enters only by beating the row's smallest.

Memory counts F plus N (501 floats), in 32-bit-float equivalents. The prior
is 20 real codes/class (100,000 floats), every 2000 / 0.8×, 3 seeds.

| C | C memory (floats) | Total with prior | Class-IL final | Task-end avg | Recon MSE | Dream corr same / other |
|---|---|---|---|---|---|---|
| float (reference) | 392,784 | 492,784 | **87.47 ± 0.82** | 86.42 ± 1.81 | 0.047 | 0.98 / 0.82 |
| **8-bit** | **98,697** | 198,697 | **87.20 ± 1.10** | **86.27 ± 0.87** | 0.054 | 0.98 / 0.82 |
| 4-bit | 49,599 | 149,599 | 83.18 ± 2.43 | 84.78 ± 0.94 | 0.083 | 0.94 / 0.81 |
| 2-bit | 25,050 | 125,050 | 61.77 ± 4.94 | 74.97 ± 3.47 | 0.128 | 0.83 / 0.76 |
| top-200/row, float | 150,801 | 250,801 | 78.10 ± 2.66 | 79.77 ± 3.48 | 0.101 | 0.88 / 0.79 |
| top-100/row, float | 75,651 | 175,651 | 56.60 ± 1.50 | 68.15 ± 2.08 | 0.111 | 0.79 / 0.77 |
| top-50/row, float | 38,076 | 138,076 | 34.69 ± 2.44 | 55.16 ± 0.44 | 0.113 | 0.71 / 0.71 |
| top-100/row, 8-bit | 38,076 | 138,076 | 57.36 ± 2.72 | 69.86 ± 0.48 | 0.116 | 0.80 / 0.76 |
| top-50/row, 4-bit | 16,157 | 116,157 | 38.04 ± 2.36 | 57.11 ± 0.68 | 0.109 | 0.75 / 0.72 |
| *10 prototypes/class (image store)* | – | *156,800* | *89.99 ± 0.56* | *89.41 ± 0.57* | | |

1. **8 bits are free.** C shrinks 4× (393k → 99k floats) for 0.3 points,
   well within seed noise, and the dreams are just as good. Stochastic
   rounding matters here: the late steps h/N are far below one 8-bit level
   and would round to zero deterministically.
2. **4 bits cost 4 points, 2 bits cost 26.** Rounding noise doesn't decay
   with n. It settles at roughly √(level·|x−F|), and the sleep solve
   amplifies it along weak code directions.
3. **Sparsity is the wrong way to shrink C.** Even top-200 (a quarter of
   the pixels) loses 9 points, and top-50 collapses to 35%. A unit's
   template isn't sparse: the many dim pixels together carry the shape
   that tells digits apart. The support also freezes early, because a new
   pixel's value (h/N)·x can rarely beat the row's minimum once N is
   large. With a 16-bit index per entry, sparse entries cost 3× a dense
   8-bit one. Quantising wins at every memory budget.
4. **Total memory.** 8-bit C plus the 20-code prior is 199k floats for
   87.2%, still above the prototype image store (157k for 90.0). 4-bit C
   gets under it (150k) but loses 4 points. 10 codes/class with 8-bit C
   (≈149k) is tested in the next section.

### Under the prototype store's budget (`dreams.py --set budget`)

8-bit C with fewer real codes per class. Every 2000 / 0.8×, 3 seeds. The
top-down weights G (392,000) are part of the network and not counted, as
in every section above.

| Condition | Prior | 8-bit C | Total (floats) | Class-IL final | Task-end avg |
|---|---|---|---|---|---|
| 10 prototypes/class (image store) | 156,800 | – | 156,800 | **89.99 ± 0.56** | **89.41 ± 0.57** |
| 1 prototype/class (image store) | 7,840 | – | 7,840 | 82.31 | 81.18 |
| Contrastive, 8-bit C, 20 codes/class | 100,000 | 98,697 | 198,697 | 87.20 ± 1.10 | 86.27 ± 0.87 |
| **Contrastive, 8-bit C, 10 codes/class** | 50,000 | 98,697 | **148,697** | **86.64 ± 1.17** | 85.11 ± 0.53 |
| Contrastive, 8-bit C, 5 codes/class | 25,000 | 98,697 | 123,697 | 84.59 ± 1.60 | 84.31 ± 1.58 |
| Contrastive, float C, 10 codes/class | 50,000 | (392,784 float) | 442,784 | 86.63 ± 0.33 | 86.77 ± 1.24 |

1. **Under budget, not at par.** 10 codes/class with 8-bit C fits in 149k
   floats (5% under the prototype store) but reaches 86.6%, 3.4 points
   below prototypes. 8 bits cost nothing here either (86.64 vs 86.63 with
   float C).
2. **The prior degrades slowly.** Halving it again to 5 codes/class
   (124k) costs only 2 more points (84.6), and dreams stay as clean
   (correlation 0.98). The accumulator (99k) is now most of the memory,
   so shrinking the prior further buys little.
3. **Where the gap comes from** is tested in the next section. (An
   earlier version of this point blamed the decoder, comparing with ridge
   on 20 codes/class. On 10 codes/class ridge does no better.)

### Where the gap to prototypes comes from (`dreams.py --set gap`)

Same 10-code prior, every 2000 / 0.8×, 3 seeds.

| Decoder | Local? | Class-IL final | Task-end avg | Recon MSE |
|---|---|---|---|---|
| Ridge (sums over all real codes, O(H³) solve) | no | 86.50 ± 0.04 | 86.62 ± 1.04 | 0.036 |
| **Contrastive, 8-bit C, 100 steps** (previous section) | yes | **86.64 ± 1.17** | 85.11 ± 0.53 | 0.055 |
| Contrastive, 8-bit C, 1000 steps | yes | 85.97 ± 0.48 | 85.99 ± 0.91 | 0.144 |
| Contrastive's fixed point solved exactly | no | 81.00 ± 1.93 | 82.50 ± 1.72 | 0.309 |

1. **There is no decoder gap on this prior.** Ridge with 10 codes/class
   reaches 86.5%, level with the local contrastive rule (86.6). The 3
   points that ridge gained before came with the 20-code prior (89.6),
   and the prior only changes ridge's *dreams*, not its decoder (its sums
   cover every real code). So the gap to prototypes is **dream
   diversity**: 10 stored codes per class aren't enough samples, however
   well they're decoded.
2. **Early stopping is the contrastive rule's regulariser.** Its exact
   fixed point (least squares under 100 stored codes, rank ≤ 101 in 501
   dimensions) overfits badly: reconstruction error 6× worse and 81.0%.
   1000 steps drift toward it (recon 0.144) without changing accuracy.
   100 steps stay near the well-conditioned directions, which is why
   preconditioning or whitening, which would speed the weak directions up,
   would push it the wrong way.
3. **Better decoding doesn't buy accuracy past a point.** Ridge decodes
   real codes 35% better than contrastive (0.036 vs 0.055) for the same
   accuracy.

So the lever is the prior, not the decoder: more codes per class in the
same memory (e.g. codes stored at 8 bits: 4× as many for the same floats).
The next section tests that, and the answer is only partly yes.

### More codes per class: the prior at 8 bits (`dreams.py --set codes8`)

Each real code is stored at 8 bits, scaled by its own max (one float), with
stochastic rounding: 126 floats per code instead of 500. Accumulator C is
8-bit as well (98,697). Every 2000 / 0.8×, 3 seeds.

| Condition | Prior (floats) | Total (floats) | Class-IL final | Task-end avg |
|---|---|---|---|---|
| **9 prototypes/class (image store)** | 141,120 | **141,120** | **89.88 ± 0.55** | **88.52 ± 0.45** |
| 10 prototypes/class (image store) | 156,800 | 156,800 | 89.99 ± 0.56 | 89.41 ± 0.57 |
| Contrastive, 10 float codes/class (budget section) | 50,000 | 148,697 | 86.64 ± 1.17 | 85.11 ± 0.53 |
| Contrastive, 20 codes/class at 8 bits | 25,200 | 123,897 | 86.71 ± 0.74 | 86.42 ± 1.02 |
| **Contrastive, 40 codes/class at 8 bits** | 50,400 | **149,097** | **87.16 ± 0.77** | **87.61 ± 0.76** |
| Contrastive, 80 codes/class at 8 bits | 100,800 | 199,497 | 88.22 ± 0.91 | 87.10 ± 1.16 |
| Contrastive, 20 float codes/class (cheapc section) | 100,000 | 198,697 | 87.20 ± 1.10 | 86.27 ± 0.87 |
| Ridge, 20 float codes/class (not local) | 100,000 | 743,785 | 89.60 ± 0.46 | 88.94 ± 1.08 |

1. **8-bit codes are free:** 20 codes at 8 bits (86.7) ≈ 20 float codes
   (87.2), at a quarter of the memory.
2. **Quadrupling the codes barely helps.** At the prototype budget, 40
   codes/class instead of 10 gives +0.5 points (87.2, task-end +2.5);
   80 codes gives 88.2 at 200k. Prototypes at 141k still lead by 2.7
   points.
3. **This revises the previous section.** On 10 codes, dreams were the
   limit for both rules. With more codes, ridge climbs fast (86.5 → 89.6
   from 10 to 20 codes) and contrastive doesn't (86.6 → 87.2). So past
   ~10 codes/class the early-stopped contrastive decoder *is* the limit:
   it blurs (recon 0.055 vs ridge's 0.036), and the extra codes' detail
   doesn't survive decoding. And its exact fixed point overfits (81%).
   Closing the gap needs a decoder between the two: regularised but
   converged.
4. **Where this leaves the local, image-free path:** 87.2% at 149k floats
   against 89.9% for prototypes at 141k. Under budget, but 2.7 points
   short.

### A regularised, converged contrastive decoder (`dreams.py --set decay`)

Sleep steps with per-synapse weight decay,
`G += η (C/n − mean(z x̂ᵀ) − λ·ē·G)`, where ē = E|z|²/(H+1) is the mean
eigenvalue of the code second moment (so λ is relative, like ridge's), and
a larger step (η·E|z|² = 1.0 instead of 0.2) so it converges. "Noisy
negatives" adds Gaussian noise of variance λ·ē to each negative code
instead, which gives the same decay in expectation. Offline, on snapshots
of one run, λ ≥ 1 converges within ~300 steps and λ = 0.3 needs ~1000+.
The unregularised exact fixed point overfits badly. Prior = 40 codes/class
at 8 bits, C 8-bit (149k floats in total), every 2000 / 0.8×, 3 seeds.

| Decoder | Local? | Class-IL final | Task-end avg | Wake recon MSE |
|---|---|---|---|---|
| *9 prototypes/class (image store, 141k)* | | *89.88 ± 0.55* | *88.52 ± 0.45* | |
| Ridge (644k floats of sums, O(H³) solve) | no | **90.33 ± 0.81** | **90.80 ± 0.47** | 0.038 |
| Contrastive, no decay, 100 steps (previous section) | yes | 87.16 ± 0.77 | 87.61 ± 0.76 | 0.055 |
| Decay 3, 300 steps | yes | 86.42 ± 0.20 | 85.37 ± 0.97 | 0.046 |
| Decay 1, 300 steps | yes | 86.99 ± 0.51 | 86.83 ± 0.83 | 0.048 |
| Noisy negatives (≈ decay 1), 300 steps | yes | 87.72 ± 0.36 | 87.26 ± 0.09 | 0.046 |
| **Decay 0.3, 1000 steps** | yes | **88.25 ± 0.42** | **88.45 ± 0.75** | 0.059 |

1. **Ridge on this prior matches prototypes** (90.3 vs 89.9), so the
   prior is good enough and the decoder is the whole remaining gap.
2. **Weaker decay with more steps helps; strong decay hurts.** λ = 0.3
   with 1000 steps gains 1.1 points (88.3), cutting the gap to prototypes
   from 2.7 to 1.6 points. λ = 1 gains nothing and λ = 3 loses. The
   trend points toward ridge's own setting (λ = 0.01), which a local rule
   can't reach in reasonable time: weak directions converge at rate ~λ.
3. **Noise beats decay at the same strength** (87.7 vs 87.0). Jittered
   negatives are also the more physical mechanism: no explicit decay
   term, just noisy dreams.
4. **Wake reconstruction error doesn't predict accuracy.** The best
   contrastive run has the worst reconstruction (0.059). So the earlier
   "it blurs" explanation was too simple: what matters is how stored
   codes from older versions of the network decode, not how well current
   codes do.
5. Cost: 1000 steps of 64 codes per sleep is ~10× the previous sleep
   compute for the generator, still small next to the network's replay.

### Pushing it: weaker noise, 3000 steps (`dreams.py --set noise`)

Same setup (40 8-bit codes/class, 8-bit C, step 1.0), 3000 sleep steps.

| Regulariser | Steps | Class-IL final | Task-end avg | Forgetting | Wake recon MSE |
|---|---|---|---|---|---|
| *9 prototypes/class (141k)* | | *89.88 ± 0.55* | *88.52 ± 0.45* | | |
| Decay 0.3 (previous section) | 1000 | 88.25 ± 0.42 | 88.45 ± 0.75 | | 0.059 |
| **Decay 0.3** | 3000 | **88.68 ± 0.63** | **88.75 ± 0.45** | 10.7 | 0.063 |
| Noisy negatives 0.3 | 3000 | 87.78 ± 0.51 | 85.28 ± 0.42 | 11.5 | 0.062 |
| Noisy negatives 0.1 | 3000 | 85.97 ± 0.99 | 86.30 ± 0.66 | 15.0 | 0.101 |

1. **The trend stops at λ ≈ 0.3.** 3000 steps add 0.4 points at decay
   0.3 (88.7, now 1.2 under prototypes, level on task-end average), but
   λ = 0.1 is worse (86.0): reconstruction error nearly doubles, the
   signature of a fit to too few codes (400 codes in 501 dimensions).
   That matches the offline check, where the exact solution at λ = 0.1
   was already poor. With this prior, ~0.3 is the right amount of
   regularisation; ridge gets away with 0.01 only because its sums cover
   every real code.
2. **Noise loses to decay once the sleep is long.** At 300 steps noise
   beat decay (87.7 vs 87.0); at 3000 steps it's behind (87.8 vs 88.7).
   Math: Gaussian noise gives the decay term only *in expectation*. Each
   64-code batch estimates it with large variance, and a constant step
   never averages that out, so the weights keep jittering around the
   fixed point. Early, the jitter acts as extra regularisation; late, it's
   just error. Fix if wanted: shrink the step during the sleep, or average
   the weights over the sleep (Polyak averaging).
3. **Cost:** 3000 steps triples the generator's sleep compute for 0.4
   points. A cheaper route to the same point is to converge faster
   (per-unit step sizes) rather than for longer. (The next section shows
   that this premise was wrong.)

### Per-unit sleep steps and dropout negatives (`dreams.py --set jacobi`)

Same setup: 40 8-bit codes/class, 8-bit C, step 1.0, every 2000 / 0.8×,
3 seeds.
- **Centred:** each presynaptic unit uses z_i − (its own mean wake
  activity), the covariance rule, and the bias is not decayed.
- **Jacobi:** a per-presynaptic-unit step 1/E[u_i² + λ], times one global
  gain (1 / largest eigenvalue of the normalised code statistics,
  estimated once per sleep, a single broadcast scalar).
- **Dropout negatives:** each unit of a negative code is kept with
  probability p and rescaled by 1/p (unreliable synapses), with no
  explicit decay. In expectation this is a ridge penalty
  diag(E[z_i²])·(1−p)/p, i.e. stronger for busier units.
- **Polyak:** the weights are averaged over the second half of each sleep
  (a transient copy of G during sleep only).

| Sleep rule | Steps | Class-IL final | Task-end avg | Forgetting | Dream corr same / other |
|---|---|---|---|---|---|
| *9 prototypes/class (141k)* | | *89.88 ± 0.55* | *88.52 ± 0.45* | | |
| Decay 0.3 | 300 | 88.50 ± 0.25 | 87.43 ± 1.24 | 10 | 0.97 / 0.81 |
| Decay 0.3 (previous sections) | 1000 / 3000 | 88.25 / 88.68 | 88.45 / 88.75 | | |
| Centred, decay 0.3 | 300 | 53.68 ± 3.66 | 71.69 ± 3.32 | 57 | 0.83 / 0.64 |
| Centred + Jacobi, decay 0.3 | 300 | 88.35 ± 0.75 | 88.40 ± 0.44 | 11 | 0.94 / 0.80 |
| Dropout keep 0.8 | 1000 | 88.93 ± 0.77 | **89.12 ± 1.32** | 10 | 0.97 / 0.82 |
| **Dropout keep 0.8 + Polyak** | 1000 | **88.99 ± 0.34** | 88.70 ± 1.08 | 10 | 0.97 / 0.82 |
| Dropout keep 0.5 + Polyak | 1000 | 88.57 ± 0.19 | 87.39 ± 0.42 | 9 | 0.99 / 0.83 |

1. **Convergence wasn't the bottleneck.** Plain steps with decay 0.3 reach
   88.5 in 300 steps, the same as 1000 or 3000 within seed noise. The
   offline check measured one sleep from a warm start, but in a real run
   the decoder carries over between sleeps, so 30 sleeps × 300 steps
   converge anyway. The gains credited to "more steps" in the last two
   sections came from the larger step (1.0) and the decay. So Jacobi had
   nothing left to speed up (88.4).
2. **Centring alone breaks the rule (53.7%, 57 points forgotten).**
   Offline, centring made the code statistics 5–10× worse conditioned
   (the mean direction, which carried most of the variance and was easy,
   is gone, and what's left spans 4 orders of magnitude). With a plain
   step, the slow directions, which carry the class differences, never
   converge, and dreams lose their class (same/other correlation 0.83 /
   0.64). Jacobi steps undo exactly that. The lesson: the mean direction
   was quietly doing the conditioning; don't remove it without per-unit
   steps.
3. **Dropout negatives are the best local rule so far: 89.0%,** 0.9
   under prototypes, and the task-end average without averaging (89.1)
   is level with them (88.5). The edge over decay 0.3 (88.5–88.7) is
   within seed noise, so call it a tie that needs no decay term at all.
   It's also the most biological of the mechanisms tried: no weight
   decay, no Gaussian noise generator, just synapses that fail 20% of
   the time during dreams.
4. **Polyak averaging helps less than expected.** It halves the spread
   across seeds (±0.34 vs ±0.77) but doesn't move the mean.
   Multiplicative dropout noise is smaller where activity is small, so it
   jitters the weights less than additive Gaussian noise did.
5. **Keep 0.5 is too much** (88.6; dreams sharper, 0.99, but less varied).

### Split Fashion-MNIST (`dreams.py --set fashion --data fashion`)

The best setups, unchanged (same hyperparameters as MNIST), on Split
Fashion-MNIST: 5 tasks of 2 classes (T-shirt/trouser, pullover/dress,
coat/sandal, shirt/sneaker, bag/boot). Every 2000 / 0.8×, 3 seeds.
"Ceiling" = the same FF trained in one shuffled pass, batch 1, with no
continual problem (`bench.py` iid protocol): 83.44 ± 1.49 on Fashion vs
96.25 ± 0.10 on MNIST. Backprop on the same protocol: 85.00 / 96.80.

| Condition | Memory (floats) | Fashion class-IL | % of ceiling | Task-end avg | Forgetting | MNIST class-IL |
|---|---|---|---|---|---|---|
| No sleep | 0 | 19.94 ± 0.01 | 24% | 31.84 | 99 | 20.09 |
| 9 prototypes/class | 141,120 | 75.42 ± 0.41 | 90% | 80.65 | 11 | 89.88 |
| 10 prototypes/class | 156,800 | 75.59 ± 0.61 | 91% | 80.93 | 12 | 89.99 |
| Ridge, 40 8-bit codes (not local) | 50,400 + 643,785 | **75.75 ± 0.15** | 91% | 80.33 | 14 | 90.33 |
| Contrastive, decay 0.3, 300 steps | 149,097 | 72.06 ± 0.44 | 86% | 78.90 | 17 | 88.50 |
| **Contrastive, dropout keep 0.8, Polyak** | 149,097 | **72.83 ± 1.18** | 87% | 79.52 | 16 | 88.99 |

Final accuracy per task (mean of 3 seeds):

| Condition | T-shirt / trouser | Pullover / dress | Coat / sandal | Shirt / sneaker | Bag / boot |
|---|---|---|---|---|---|
| 10 prototypes/class | 87.0 | 69.0 | 79.3 | 54.8 | 88.0 |
| Ridge, 40 8-bit codes | 89.2 | 71.9 | 70.4 | 55.5 | 91.8 |
| Contrastive, dropout | 81.8 | 73.5 | 65.2 | 54.7 | 88.9 |

1. **Sleep replay holds up on a harder dataset.** Prototype replay keeps
   90% of the no-forgetting ceiling (93% on MNIST); without sleep, class-IL
   is 20% on both. That isn't the Bartunov et al. collapse: the
   method transfers without retuning.
2. **Image-free dreams still match images when the decoder is ridge.**
   Ridge on 40 8-bit codes/class (75.8) ties prototypes (75.6), as on
   MNIST. So top-down dreams from stored codes are a full replacement for
   stored images on this data too.
3. **The local decoder's gap triples:** dropout contrastive is 2.8 points
   under prototypes (0.9 on MNIST); decay 3.5 (1.5). Dropout still beats
   decay. Its dreams decode worse (wake recon 0.130 vs ridge's 0.070;
   Fashion images are twice as dense in ink, mean pixel 0.29 vs 0.13).
   The loss sits on the coat/sandal task (65 vs 79). Probably coat, the
   class most like pullover and shirt, where a blurrier dream protects
   least; per-class numbers would confirm it. Finer detail matters more
   where classes differ by texture, not outline.
4. **The shirt/sneaker task is hard for every method** (≈55% in every
   row), most likely shirt, the classic Fashion confusion with T-shirt,
   pullover and coat. It shares features with three old classes, so
   interference is worst there.
5. Next for the local path: whatever makes the decoder sharper on dense
   images, e.g. a two-level generator (decode to layer 1, then to pixels)
   or more steps per sleep on this data, since 300 steps sufficed only
   on MNIST.

### A sharper local decoder: two levels, more sleep steps (`dreams.py --set deep --data fashion`)

Split Fashion-MNIST, every 2000, 0.8×, 3 seeds, all with 40 8-bit codes
per class. Two levels: the prior stores layer-2 codes, decoded layer 2 →
layer 1 → image. Each level keeps its own 8-bit Hebbian accumulator awake
(layer 1 scaled by its running max activity). In sleep, the top level
learns on the stored codes and the bottom level on the layer-1 codes the
top level now dreams (wake-sleep: each decoder learns under the statistics
it will decode). Dropout negatives keep 0.8 and Polyak averaging, as
before. References from the `fashion` set above.

| Condition | Generator memory beyond G | Class-IL final | Task-end avg | Forgetting |
|---|---|---|---|---|
| 10 prototypes/class (images) | – | 75.59 ± 0.61 | 80.93 | 12.2 |
| Ridge, one level (not local) | 644k | 75.75 ± 0.15 | 80.33 | 14.4 |
| Contrastive, one level, 1000 steps | 99k | 72.83 ± 1.18 | 79.52 | 16.2 |
| **Contrastive, one level, 3000 steps** | 99k | **74.57 ± 0.41** | 80.12 | 15.3 |
| Ridge, two levels (not local) | 1,145k | 73.17 ± 0.61 | 79.16 | 16.3 |
| Contrastive, two levels, 1000 steps | 162k | 72.17 ± 2.53 | 78.97 | 16.9 |
| Contrastive, two levels, 3000 steps | 162k | 72.00 ± 0.61 | 78.73 | 19.0 |

Per task, class-IL at the end:

| Condition | T-shirt / trouser | Pullover / dress | Coat / sandal | Shirt / sneaker | Bag / boot |
|---|---|---|---|---|---|
| 10 prototypes/class | 87.0 | 69.0 | 79.3 | 54.8 | 88.0 |
| Contrastive, one level, 1000 steps | 81.8 | 73.5 | 65.2 | 54.7 | 88.9 |
| Contrastive, one level, 3000 steps | 86.2 | 72.3 | 62.8 | 59.4 | 92.2 |
| Ridge, two levels | 88.0 | 68.4 | 63.6 | 51.4 | 94.4 |
| Contrastive, two levels, 3000 steps | 83.4 | 71.2 | 61.9 | 51.3 | 92.2 |

1. **More sleep is what the dense images needed.** 3000 steps instead of
   1000 takes the local rule from 72.8 to **74.6**, 1.0 under prototypes
   and 1.2 under ridge, about the gap it had on MNIST (0.9). Every seed
   improves (+1.1, +3.1, +0.8) and the spread shrinks. Memory is
   unchanged (149k floats); only sleep gets longer. On MNIST 300 steps
   were enough because warm starts converged across sleeps. Denser images
   put more weight in the slow, low-variance code directions, which take
   longer to converge.
2. **The second level hurts, and the local rule isn't why.** Ridge drops
   from 75.8 to 73.2 when it decodes from layer 2, so the loss comes from
   the design, not the learning rule. The local two-level rule is level
   with that (72.0–72.2) and gains nothing from more steps. As on MNIST
   (85.2 vs 86.8), the top code is the worse thing to store. Layer 2 is
   further from the image and carries more of the label, so detail lost
   there can't be recovered, and errors compound through two decoders.
3. **A wake-sleep mismatch shows in the bottom decoder.** It learns on
   dreamed layer-1 codes, so on real layer-1 codes it reconstructs badly
   (wake MSE 0.22–0.24, vs 0.066 for two-level ridge, which trains on
   real codes). It still dreams recognisable classes (same-class
   correlation 0.97). This is the classic wake-sleep bias: each level
   fits the distribution the level above dreams, not the real one.
4. **Reconstruction still doesn't predict accuracy.** The 3000-step
   one-level decoder has worse wake MSE than the 1000-step one (0.152 vs
   0.130) and higher accuracy. The extra steps change the slow directions
   that separate classes, not the pixel average.
5. **The coat/sandal task is still the gap** (63–65 for every
   code-decoded method, 79 for prototypes; one-level ridge 70). The other
   tasks are now at or above prototypes. What's missing is probably the
   fine texture that separates coat from pullover and shirt, which any
   linear decoder of a 500-unit code blurs.

### Longer sleeps on Split MNIST (`dreams.py --set long`)

The 3000-step setting that helped on Fashion, back on MNIST: one level,
40 8-bit codes/class, 8-bit C, dropout negatives keep 0.8, Polyak, every
2000, 0.8×, 3 seeds. References from earlier sets.

| Condition | Memory (floats) | Class-IL final | Seeds 0 / 1 / 2 | Task-end avg | Forgetting |
|---|---|---|---|---|---|
| 9 prototypes/class (images) | 141k | 89.88 ± 0.55 | 89.3 / 89.7 / 90.6 | 88.52 | 8.4 |
| Ridge, same prior (not local) | 149k + sums | 90.33 ± 0.82 | 89.8 / 89.8 / 91.5 | 90.80 | 8.5 |
| Local, decay 0.3, 300 steps | 149k | 88.50 ± 0.25 | 88.4 / 88.2 / 88.8 | 87.43 | 10.2 |
| Local, dropout 0.8, 1000 steps | 149k | 88.99 ± 0.34 | 88.6 / 89.0 / 89.4 | 88.70 | 10.2 |
| **Local, dropout 0.8, 3000 steps** | 149k | **89.36 ± 0.52** | 89.1 / 90.1 / 88.9 | **88.97** | 9.9 |

1. **About half the remaining gap closes, not all of it.** 3000 steps
   gives 89.4, 0.5 under prototypes (was 0.9) and 1.0 under ridge. Seed
   by seed against 1000 steps: +0.5, +1.1, −0.5. So the gain (+0.4) is
   within the noise of 3 seeds. On Fashion it was +1.7 and every seed
   improved.
2. **On task-end average the local rule now beats prototypes** (89.0 vs
   88.5), as the 1000-step run nearly did. The final-accuracy gap is
   mostly seed 2, where prototypes had an unusually good run (90.6).
3. **That's the pattern the slow-direction story predicts.** Sparse MNIST
   codes put little weight where convergence is slow, so 300 steps
   already got most of the way and 3000 adds a little. Dense Fashion
   codes put much more there, so the extra steps mattered. Recon again
   gets worse as accuracy rises (0.080 vs 0.073 at 1000 steps).
4. **Practical setting: 3000 steps on both datasets.** It never hurts,
   costs only sleep time, and on MNIST the local, image-free decoder is
   now statistically level with stored prototype images at about the
   same memory (149k vs 141k floats). Separating a 0.5-point difference
   would need about 10 seeds per arm.
   **Correction (10 seeds, `--set tenseed`):** it isn't level. Over 10
   seeds the local decoder is 89.13 vs 89.88 for prototypes, a gap of
   −0.74, significant at paired t = −2.37.

### A nonlinear local decoder: code + sparse random features (`dreams.py --set expand --data fashion`)

Can a nonlinear image decoder fix the coat/sandal gap? The layer-1 code
fans out through **fixed random weights** to 1000 or 2000 extra units.
Only the 100 most active fire (k-winners-take-all), and their total
activity is scaled to match the code's. The decoder reads [code,
features] and is learned with the same local contrastive rule: 8-bit C,
dropout negatives keep 0.8, Polyak, 3000 steps. This is the cerebellar
granule-cell layer (Marr–Albus): a fixed, sparse expansion makes a
linear readout nonlinear in the input, and only the readout learns. No
credit has to cross a hidden layer, so the rule stays local. Split
Fashion-MNIST, every 2000, 0.8×, 3 seeds, 40 8-bit codes/class.
References from earlier sets.

| Condition | Generator memory beyond G | Class-IL final | Seeds 0 / 1 / 2 | Task-end avg |
|---|---|---|---|---|
| 10 prototypes/class (images) | – | 75.59 ± 0.61 | 76.2 / 74.8 / 75.8 | 80.93 |
| Ridge, code only (not local) | 644k | 75.75 ± 0.15 | 75.6 / 75.7 / 76.0 | 80.33 |
| **Ridge, code + 1000 features (not local)** | 3,430k | **76.48 ± 0.58** | 77.2 / 76.5 / 75.8 | 80.99 |
| Local, code only, 3000 steps | 99k | 74.57 ± 0.41 | 75.2 / 74.3 / 74.2 | 80.12 |
| Local, code + 1000 features | 296k | 75.26 ± 0.25 | 75.3 / 74.9 / 75.6 | 80.77 |
| **Local, code + 2000 features** | 493k | **75.58 ± 0.68** | 76.1 / 76.0 / 74.6 | 80.80 |

Per task, class-IL at the end:

| Condition | T-shirt / trouser | Pullover / dress | Coat / sandal | Shirt / sneaker | Bag / boot |
|---|---|---|---|---|---|
| 10 prototypes/class | 87.0 | 69.0 | **79.3** | 54.8 | 88.0 |
| Ridge, code only | 89.2 | 71.9 | 70.4 | 55.5 | 91.8 |
| Ridge, code + 1000 features | 86.9 | 72.3 | 69.7 | 62.3 | 91.2 |
| Local, code only, 3000 steps | 86.2 | 72.3 | 62.8 | 59.4 | 92.2 |
| Local, code + 1000 features | 85.6 | 75.0 | 60.5 | 61.5 | 93.8 |
| Local, code + 2000 features | 88.8 | 67.0 | 65.3 | 63.5 | 93.3 |

1. **The nonlinear decoder helps overall.** The local rule goes from 74.6
   to 75.3 with 1000 features and **75.6 with 2000, level with prototype
   images (75.6)**. Each seed improves on code-only (+0.2, +0.6, +1.3
   and +1.0, +1.6, +0.4). Ridge on the same features reaches **76.5**,
   the first image-free decoder clearly above the image store (+0.9;
   above prototypes on 2 of 3 seeds).
2. **It missed its target.** The coat/sandal task doesn't move (60–65
   local, 70 ridge, 79 prototypes). The gain is on shirt/sneaker (+2 to
   +7 points: 61–64 vs 55–59) and bag/boot. So the coat/sandal gap
   isn't decoder sharpness: ridge with features reconstructs coats 16%
   better in a one-task test, and the task still sits at 70. Prototypes
   do something there that decoded codes don't. Average dreams can't
   show what: dreamed coat correlates 0.96 with pullover under every
   method. The next step is per-class accuracy, which the logs don't
   keep yet.
3. **The local rule converges to a different decoder than ridge**, but
   gains in step with it (+1.0 vs +0.7). Its wake reconstruction is poor
   (0.27 vs ridge's 0.079), and pixel error again doesn't predict
   accuracy. One cause: the sleep statistics of a 1500–2500-dimensional
   feature vector are estimated from at most 400 stored codes, so they
   are rank-deficient, and dropout's diagonal does most of the
   regularising. Ridge sees every wake code.
4. **It costs memory.** The accumulator grows with the feature count, so
   the local decoder level with prototypes uses 543k floats in total
   (code store 50k + C 493k) against prototypes' 157k. The fixed random
   weights cost nothing to store: one seed regenerates them. This is the
   first local, image-free method level with stored images on Fashion,
   but not yet under their memory.

### Per-class accuracy: what prototypes do on coat/sandal (`dreams.py --set perclass --data fashion`)

Runs now record the test confusion matrix after every task, and how the
final network sees its own replays: their confusion, pixel spread and
correlation with the real class mean. Rerun on Split Fashion-MNIST, every
2000, 0.8×, 3 seeds. The three reference conditions reproduce their
earlier numbers exactly. The new arm is **prototypes replayed without
their per-pixel noise**, which tests whether replay variety is what
protects coat. Tasks: T-shirt/trouser, pullover/dress, coat/sandal,
shirt/sneaker, bag/boot.

Final per-class accuracy (confusions summed over seeds):

| Condition | Class-IL | T-shirt | Trouser | Pullover | Dress | **Coat** | **Sandal** | Shirt | Sneaker | Bag | Boot |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 prototypes + noise | 75.59 ± 0.61 | 80 | 94 | 55 | 83 | **84** | 75 | 19 | 90 | 88 | 88 |
| **10 prototypes, no noise** | **77.29 ± 0.40** | 82 | 94 | 58 | 79 | 82 | **83** | 28 | 85 | 91 | 91 |
| Ridge, 40 8-bit codes | 75.75 ± 0.15 | 84 | 94 | 61 | 83 | 72 | 69 | 28 | 83 | 88 | 95 |
| Local, code + 2000 features | 75.58 ± 0.67 | 83 | 94 | 56 | 78 | 73 | **58** | **44** | 83 | 91 | 95 |

Sums over the classes that compete with each other:

| Condition | Coat + shirt | Pullover + coat + shirt | Sandal + sneaker + boot | Sandal → boot / sneaker |
|---|---|---|---|---|
| 10 prototypes + noise | 102.5 | 157.6 | 253.7 | 11 / 14 % |
| 10 prototypes, no noise | 110.0 | 167.9 | **259.2** | 8 / 7 % |
| Ridge | 99.9 | 160.5 | 247.2 | 14 / 16 % |
| Local, 2000 features | **116.8** | **173.3** | 236.0 | **28** / 14 % |

1. **The "coat/sandal gap" is two different things.** The task-level
   number mixed them.
2. **Coat isn't a memory failure but a trade with shirt.** Shirt
   arrives in the next task and is the most confusable class. Prototype
   replay holds coat at 84% by pushing the coat/shirt boundary into
   shirt: 36% of shirts are called coat, and shirt reaches only 19%. The
   local decoder replays coat less forcefully. Coat drops to 73%
   (15% go to shirt), and shirt rises to **44%**. Over coat + shirt the
   local decoder is the best method by 7–14 points, and over pullover +
   coat + shirt too. Ridge sits between them. Its coat collapses when
   shirt is learned (33% after task 4, 32% called shirt) and recovers
   only partly.
3. **Sandal is the real loss.** Sandals fall to boot (28% for the local
   decoder vs 8–11% for prototypes) and to sneaker. Both are learned
   later. Over the three footwear classes, the local decoder is 18
   points below prototypes and ridge 7. A sandal is thin straps around
   empty space. A decoder that blurs fills that space in, so the
   network learns "sandal" from replays that look like closed shoes,
   and real sandals then fall to the shoes learned live. Its replayed
   sandals have the lowest correlation with the real class (0.98) and
   less spread than real sandals (0.161 vs 0.192). Most classes' replay
   spread matches the real data; only sandal and shirt fall short.
4. **The same mechanism shows in the prototypes, and fixing it gives the
   new best.** Per-pixel noise does the same thing to sandals as blur:
   it fills the gaps. Removing it lifts sandal from 75 to **83%**
   (sandal → sneaker 14 → 7%). Class-IL rises to **77.3%** on every seed
   (+1.5, +2.0, +1.6), the best of any method so far. So **replay
   variety isn't what protects old classes here.** Noise-free prototype
   replays are half as varied as real images (0.10–0.18 vs 0.13–0.27)
   and do better. Ridge's replays are even less varied and do fine
   wherever no thin structure is involved.
5. **The network recognises every replay as its own class.** Its final
   confusion on its own replays is 87–100% for coat and sandal under
   every method. Replays aren't mislabelled. They're sound but simpler
   than the real class, so they defend it only against the near
   neighbours that look like the replay.

What this means for the local decoder: on Fashion it isn't blurrier
everywhere, it's blurrier where shape lives in empty space. Next steps:
- a decoder that can produce sharp holes, so empty space can be dark
  inside a bright outline (e.g. a sparse code over oriented edges);
- check noise-free prototypes on MNIST, where noise was the default.

### Prototypes without pixel noise on Split MNIST (`dreams.py --set nonoise`)

On Fashion, dropping the prototypes' per-pixel replay noise gained 1.7
points, mostly on sandal. Does it carry over to MNIST? Every 2000, 0.8×,
3 seeds; the noisy references were rerun (identical numbers) so that
MNIST also has per-class confusions. Without noise a prototype is just
its mean, so the variance array isn't needed and **memory halves**
(earlier tables count both).

| Condition | Memory (floats) | Class-IL final | Seeds 0 / 1 / 2 | Task-end avg | Forgetting |
|---|---|---|---|---|---|
| 9 prototypes + noise | 141,120 | 89.88 ± 0.55 | 89.3 / 89.7 / 90.6 | 88.52 | 8.4 |
| 9 prototypes, no noise | **70,560** | 89.34 ± 0.41 | 89.4 / 89.8 / 88.8 | 89.75 | 7.9 |
| 10 prototypes + noise | 156,800 | 89.99 ± 0.56 | 89.3 / 90.6 / 90.1 | 89.41 | 8.3 |
| 10 prototypes, no noise | **78,400** | 89.71 ± 1.14 | 88.1 / 90.3 / 90.7 | 89.54 | 7.3 |
| Local decoder, 3000 steps (`--set long`) | 149,097 | 89.36 ± 0.52 | 89.1 / 90.1 / 88.9 | 88.97 | 9.9 |

1. **On MNIST the noise neither helps nor hurts.** Final accuracy moves
   by −0.5 and −0.3 (within seed noise), task-end average by +1.2 and
   +0.1, and forgetting drops by about 0.5–1. Per class it reshuffles
   (no-noise better on 2 and 4, worse on 9 and 3), with no net effect.
   On Fashion it was +1.7 on every seed. That fits the Fashion
   diagnosis: noise hurts where a class is defined by thin structure
   around empty space (sandal). Digits are strokes on an empty
   background, with no class whose identity is the gaps.
2. **Noise-free replay is less varied and isn't worse.** Replay spread is
   0.08 vs 0.13 with noise (real 0.16), confirming on a second dataset
   that variety isn't what protects old classes.
3. **This moves the memory comparison against the local decoder.** A
   noise-free 9-prototype store reaches 89.3% at **70.6k floats**,
   matching the local image-free decoder (89.4%) at under half its
   memory (149k). The earlier claim that the local decoder is "level
   with prototypes at about the same memory" holds only against the
   noisy store, which was paying for a variance it didn't need. The
   local path's remaining case is not memory but that nothing
   image-shaped is stored and the dreams are the network's own.

### Sharpened dreams: centre-surround lateral inhibition (`dreams.py --set sharpen --data fashion`)

The per-class diagnosis suggested decoded sandals lose their thin straps
to blur. The retina and LGN restore edge contrast with centre-surround
inhibition: each pixel minus the mean of its neighbourhood, added back
with a gain α, x' = clip(x + α(x − blur₃ₓ₃(x))). This is an unsharp mask
applied to every dream before replay, with no learning and no memory. On
3×3-blurred real sandals, α = 1 recovers about a third of the lost pixel
spread and lowers the error to the real image. Split Fashion-MNIST, every
2000, 0.8×, 3 seeds, 40 8-bit codes/class; references from earlier sets.

| Condition | Class-IL final | Seeds 0 / 1 / 2 | Task-end | Coat | Sandal | Shirt | Footwear sum | Sandal → boot | Replay spread (real 0.19) |
|---|---|---|---|---|---|---|---|---|---|
| 10 prototypes, no noise | 77.29 ± 0.40 | 77.7 / 76.8 / 77.4 | 81.55 | 82 | 83 | 28 | 259 | 8% | 0.126 |
| Ridge | 75.75 ± 0.15 | 75.6 / 75.7 / 76.0 | 80.33 | 72 | 69 | 28 | 247 | 14% | 0.100 |
| **Ridge, sharpened α = 1** | **76.62 ± 0.14** | 76.7 / 76.4 / 76.7 | 81.27 | 75 | 71 | **50** | 251 | 12% | 0.110 |
| Local, 3000 steps | 74.57 ± 0.41 | 75.1 / 74.3 / 74.2 | 80.12 | – | – | – | – | – | – |
| Local, sharpened α = 1 | 73.30 ± 0.31 | 73.0 / 73.2 / 73.7 | 79.62 | 72 | 58 | 36 | 233 | 27% | 0.160 |
| Local, sharpened α = 2 | 72.63 ± 1.14 | 74.2 / 71.6 / 72.1 | 78.65 | 83 | 53 | 38 | 227 | 29% | 0.192 |

1. **Sharpening helps ridge.** It gains +0.9, on every seed (+1.1, +0.7,
   +0.7). Ridge on 40 8-bit codes plus a fixed retina-like filter is now
   **76.6%**, above noisy prototypes (75.6) and 0.7 under noise-free
   ones. The gain isn't on sandal (69 → 71) but on shirt (28 → 50) and
   coat (72 → 75), so the coat/shirt trade improves on both sides.
2. **It hurts the local decoder.** It loses 1.3 at α = 1 and 1.9 at
   α = 2, on every seed. An unsharp mask amplifies whatever fine detail
   a dream already has. Ridge dreams are smooth (spread 0.10, real
   0.19), so it adds edge contrast. The local decoder's dreams carry
   pixel noise from an unconverged decoder (wake error 0.15 vs ridge's
   0.07), so it adds noise. That's the same harm the per-pixel noise
   did to prototypes. Replay spread rises toward and to the real level
   (0.16, 0.19), and accuracy falls as it does. More spread is not
   better replay, for the third time.
3. **Sandal isn't fixed by local contrast, so the earlier explanation
   was too simple.** For neither decoder does sharpening bring sandal
   back (ridge 69 → 71, local 58 → 58 / 53; sandal → boot stays
   12–29%). Noise-free prototypes still hold sandal at 83. So what
   decoded sandals lack isn't 3×3-scale edge contrast. More likely
   it's shape: a linear decoder averages the many sandal styles into a
   boot-like silhouette, while each prototype is one real style's
   average. That's a guess this experiment can't test; a per-code
   nearest-neighbour check of decoded sandals against real ones would.

### Noise-free prototype memory curve on Fashion (`dreams.py --set frontier --data fashion`)

How cheap can the image store be once it needs no variance? Split
Fashion-MNIST, every 2000, 0.8×, 3 seeds. Memory counts only the means
(the `perclass` run's table still shows 156,800 for 10 noise-free
prototypes; the right figure is 78,400).

| Condition | Memory (floats) | Class-IL final | Seeds 0 / 1 / 2 | Task-end | Coat | Sandal | Shirt |
|---|---|---|---|---|---|---|---|
| 3 prototypes, no noise | 23,520 | 72.73 ± 1.11 | 73.7 / 71.2 / 73.3 | 78.62 | 70 | 69 | 28 |
| 5 prototypes, no noise | 39,200 | 75.16 ± 0.61 | 75.9 / 74.4 / 75.1 | 79.94 | 78 | 76 | 22 |
| **10 prototypes, no noise** | **78,400** | **77.29 ± 0.40** | 77.7 / 76.8 / 77.4 | 81.55 | 82 | 83 | 28 |
| 10 prototypes + noise | 156,800 | 75.59 ± 0.61 | 76.2 / 74.8 / 75.8 | 80.93 | 83 | 75 | 19 |
| Ridge, 40 8-bit codes (+ 644k sums) | 50,400 | 75.75 ± 0.15 | 75.6 / 75.7 / 76.0 | 80.33 | 72 | 69 | 28 |
| Ridge + sharpening | 50,400 | 76.62 ± 0.14 | 76.7 / 76.4 / 76.7 | 81.27 | 75 | 71 | 50 |
| Local decoder, 3000 steps | 149,097 | 74.57 ± 0.41 | 75.1 / 74.3 / 74.2 | 80.12 | – | – | – |

1. **Noise-free prototypes are the most memory-efficient replay on
   Fashion.** At 39k floats, 5 per class beat the local decoder at 149k
   (75.2 vs 74.6). At 78k, 10 per class are the best result on Fashion
   (77.3). Accuracy rises steadily with the number per class: 72.7, 75.2,
   77.3.
2. **Sandal tracks the number of prototypes** (69, 76, 83). That fits
   the shape-averaging guess: more prototypes cover more distinct sandal
   styles, while a decoder of codes averages them. With 3 per class,
   prototypes are no better on sandal than ridge (69).

### Local decoder vs prototypes on Split MNIST, 10 seeds (`dreams.py --set tenseed --seeds 3 4 5 6 7 8 9`)

The 3-seed comparison (`--set long`) called the local decoder
statistically level with prototypes (89.4 vs 89.9). Seeds 3–9 added
for three conditions (seeds 0–2 from `long` and `nonoise`). Every 2000,
0.8×. Differences are paired by seed.

| Condition | Memory (floats) | Class-IL, 10 seeds | SEM | Task-end |
|---|---|---|---|---|
| Local decoder, 3000 steps | 149,097 | 89.13 ± 0.55 | 0.17 | 86.60 |
| 9 prototypes + noise | 141,120 | **89.88 ± 0.67** | 0.21 | 86.64 |
| 9 prototypes, no noise | **70,560** | 89.65 ± 0.44 | 0.14 | **88.02** |

| Paired difference | Mean | Paired t (9 df) | Local wins |
|---|---|---|---|
| Local − prototypes + noise | −0.74 | −2.37 (p ≈ 0.04) | 2 / 10 |
| Local − prototypes, no noise | −0.51 | −2.35 (p ≈ 0.04) | 3 / 10 |
| Task-end: local − no-noise prototypes | −1.41 | −3.33 (p ≈ 0.01) | – |

1. **The gap is real, and the earlier claim is withdrawn.** With 10
   seeds the local decoder is 0.5–0.7 points under prototype replay.
   The difference is small but consistent, and both paired tests reject
   equality at about p = 0.04. The 3-seed estimate (89.36) was lucky;
   over 10 seeds it is 89.13. The `long` section's "statistically level
   with stored prototype images" is wrong.
2. **Noise on MNIST: nothing on the final number, something during
   learning.** Without noise, prototypes lose 0.2 on final accuracy (not
   significant) but have a higher task-end average (88.0 vs 86.6) and
   need half the memory.
3. **Where this leaves the local decoder.** It is a working,
   image-free, solve-free generator within about 0.5 points of image
   replay on MNIST and 2.7 on Fashion (vs noise-free prototypes). It
   costs about twice the memory of the best image store on MNIST and
   four times on Fashion. Its case rests on what it doesn't store (no
   images, dreams are the network's own), not on accuracy or memory.

### Sandal shape check (`sandal.py`)

Why are decoded sandals lost to sneaker and boot? The guess was **shape
averaging**: a decoder of codes blends many sandal styles into one
boot-like silhouette, while each prototype keeps one style. `sandal.py`
retrains each condition (Split Fashion-MNIST, every 2000, 0.8×, 3
seeds, identical numbers to the earlier runs), keeps the final network
and store, and measures the store's sandal items against the 1000 real
test sandals:
- **openness:** the empty fraction of the shoe's bounding box. Real
  sandals 0.52, sneakers 0.35, boots 0.40.
- **nearest class:** whether each item's nearest real image among
  sandal, sneaker and boot is a sandal.
- **styles hit:** of 8 k-means sandal styles fitted on the training set.
- **recon:** for the decoders, real test sandals decoded from the final
  network's own codes. Good recon with bad dreams would mean stale
  stored codes.

Picture: `results/sandal/sandals.png`. Rows top to bottom: 10
prototypes without noise, the same with noise, 3 prototypes, ridge,
local decoder, style centroids.

| Condition | Sandal acc. | Items | Openness | Nearest is sandal / sneaker / boot | Styles hit | Fine-detail realism (seed 0) |
|---|---|---|---|---|---|---|
| Real sandals (held-out half) | – | 500 | 0.517 | 1.00 / 0.00 / 0.00 | 8 | – |
| 10 prototypes, no noise | **83** | 10 | 0.271 | 0.47 / 0.30 / 0.23 | 7.7 | **0.33** |
| 10 prototypes + noise (same items) | 75 | 10 | 0.271 | 0.47 / 0.30 / 0.23 | 7.7 | 0.33 (+ noise in replay) |
| 3 prototypes, no noise | 69 | 3 | 0.189 | 0.33 / 0.56 / 0.11 | 3.0 | 0.22 |
| Ridge, 40 codes | 69 | 40 | 0.218 | 0.42 / 0.48 / 0.11 | 7.7 | 0.21 |
| ↳ recon of real sandals | | 1000 | 0.309 | 0.30 / 0.55 / 0.15 | 8.0 | – |
| Local decoder, 3000 steps | 62 | 40 | 0.237 | 0.30 / 0.53 / 0.17 | 7.7 | **0.08** |
| ↳ recon of real sandals | | 1000 | 0.466* | 0.20 / 0.49 / 0.30 | 7.7 | – |

Fine-detail realism: correlation between an item's high-pass detail
(the image minus its 3×3 blur) and that of its nearest real sandal.
For scale, a real held-out sandal against its nearest other real sandal
scores 0.29, so prototypes are at real-image level.
Starred openness is inflated by speckle (bright isolated pixels move the
ink box). The mean of all real sandals has openness 0.07, and its
nearest real image is a sneaker: averaging sandals makes a closed shoe.

1. **Style coverage isn't the problem, and the hypothesis is mostly
   wrong.** Every 10- or 40-item store hits 7–8 of the 8 sandal styles.
   **Prototypes are averages too:** openness 0.27 (real 0.52), and
   only 47% have a real sandal as their nearest neighbour. On openness,
   style coverage and nearest class, prototypes differ from ridge by
   little (0.27 vs 0.22, 8 vs 8, 0.47 vs 0.42) and are ranked by none of
   them consistently: 3 prototypes are the least sandal-like store
   (0.33) and tie ridge on sandal accuracy (69).
2. **What tracks sandal accuracy is how real the fine detail is.**
   The ranking is prototypes 0.33 → 83%, 3 prototypes 0.22 → 69%,
   ridge 0.21 → 69%, local 0.08 → 62%. The picture shows why:
   - prototypes are blurred but **coherent single shapes** (one is a
     sharp gladiator sandal);
   - ridge dreams are **ghosted superpositions**, such as a heel laid
     over a flat sandal: one code decoded linearly mixes two shapes;
   - the local decoder's dreams are **covered in salt-and-pepper
     speckle**. They have twice the prototypes' high-frequency energy,
     almost none of it real structure.
3. **The prototype noise result is the controlled version of the same
   effect.** Noisy and noise-free prototypes store identical means
   (the store learns from images, not from the network), and adding
   random pixel detail at replay costs sandal 8 points (83 → 75). Random
   fine detail actively hurts; real fine detail helps. Sandal is where
   this shows, because its identity is in thin structure.
4. **It isn't stale codes.** Decoding the final network's codes of
   real test sandals gives no more sandal-like images than the dreams
   from stored codes (ridge 0.30 vs 0.42 nearest-is-sandal; local 0.20
   vs 0.30). The loss is in code → image, not in code drift.
5. **This revises the Fashion diagnosis.** "Blur fills the straps" was
   wrong for prototypes (they're blurry too) and incomplete for the
   decoders. The defect is incoherent detail: ghosting for ridge,
   speckle for the local rule. It also explains why sharpening helped
   ridge (it amplified real edges) and hurt the local decoder (it
   amplified speckle).

Next for the local decoder: remove the speckle rather than add
capacity. Speckle is pixel-independent weight noise in rarely used code
directions (8-bit C, unconverged slow modes). Three local options: a
smoothness prior on each unit's projective field (neighbouring output
pixels share a weight, as in a retinotopic map); a weight decay that
acts only on the high-pass part; or simply blurring the dreams with
3×3 lateral excitation, the opposite of the sharpening test.

### Label-swap adaptation test (`swap.py`)

Does a method forget when it *should*? Split-MNIST as usual (every 2000,
0.8×, 3 seeds). Then digits 2 and 3 swap labels, and the stream shows
only those two digits, relabelled, for 8000 more examples on the same
sleep clock. Every 500 examples the test set is probed:
- **new:** 2s and 3s predicted with the swapped label (adaptation);
- **old:** still predicted with the old label (perseveration);
- **others:** class-IL on the other 8 digits (retention);
- **store new:** the fraction of the store's replays labelled 2 or 3
  that look like the other digit (nearest real class mean).

Sleeps fall at t = 2000, 4000, 6000, 8000, so those probes are taken
just after a sleep; t = 500, 1000, 3000 are mid-wake.

| Condition | New, mid-wake (t=500 / 3000) | New, after sleeps (t=2000 / 4000 / 8000) | Others, mid-wake (t=500 / 1000) | Others after sleep (t=8000) | Store new (t=2000 / 8000) |
|---|---|---|---|---|---|
| No sleep | 96.4 / 98.3 | 98.1 / 98.6 / 99.2 | 3.0 / 1.8 | 1.4 | – |
| 9 prototypes, no noise | 95.6 / 98.0 | 76.6 / 78.7 / **85.4** | 42.6 / 30.4 | 89.7 | 31.7 / **67.6** |
| Ridge, 40 codes | 96.1 / 98.0 | 74.0 / 72.5 / 77.7 | 39.4 / 31.2 | 90.4 | 11.1 / 37.6 |
| Local decoder, 3000 steps | 96.0 / 98.1 | 70.4 / 74.7 / 82.8 | 41.2 / 32.5 | 89.3 | 16.6 / 40.2 |

Final new-mapping accuracy by seed: prototypes 85 / 90 / 81, ridge
75 / 87 / 71, local 82 / 84 / 83.

1. **Awake, every method takes in the swap almost at once** (96% after
   500 examples). FF learns the new labels as fast as plain online
   learning.
2. **Each sleep partly undoes it,** because the store still holds the
   old mapping and replays it. Right after a sleep, new-mapping accuracy
   falls back to 70–79%, and it recovers only as fast as the store
   turns over. After 8000 swapped examples (four sleeps): prototypes
   85%, local decoder 83%, ridge 78%.
3. **Neither store notices that the world changed, so turnover follows
   its counting rule.** Prototypes have a learning-rate floor (0.01), so
   the nearest prototype drifts toward the new data: 68% of their 2/3
   replays show the new mapping after 8000 examples. The reservoir of
   codes that both decoders share replaces items with probability
   40 / (number seen), so only 38–40% have turned over. Prototypes
   adapt fastest for a mundane reason: their recency floor, not
   anything about images.
4. **Conflicting wake data wrecks everything else between sleeps.** Within
   500 swapped examples the other 8 digits drop from 90% to 30–43%
   (worst seed 13%) under every sleep method, and each sleep restores
   them to about 90%. In the ordinary task stream this doesn't happen
   (new classes don't contradict old knowledge). *(Corrected later: it
   does happen in the ordinary stream too. Anytime class-IL is 68%, and
   a median 45% of stored prototypes fail just before each sleep; see
   "Interference-triggered sleep" below.)* A label that
   contradicts the network's confident prediction drives large FF
   updates that disrupt shared features. **Retention under conflict
   exists only right after a sleep.** On a clock of 2000 the network
   spends most of its time broken.
5. **The swapped digits that aren't relabelled aren't kept either**:
   after the last sleep only 2–8% keep the old label, and 9–19% go to
   some third class. Old and new replays fight, and the boundary lands
   elsewhere.
6. **No advantage for the local generator.** It adapts like ridge on
   the same reservoir (its 83% vs ridge's 78% is within seed spread),
   and slower than noise-free prototypes. The decoder is label-free and
   adapts through the prior, which is a reservoir.

The two problems point to two biological mechanisms, tested next:
- **Store turnover should be driven by surprise, not counts.** A
  hippocampus-like store should overwrite an item fast when the network's
  prediction contradicts the label (novelty or mismatch; dopamine and
  acetylcholine raise hippocampal plasticity).
- **Sleep should be triggered by conflict, not a clock.** Under conflict
  the network needs replay sooner.

### Surprise-gated store turnover (`swap.py --conds proto9_nn_sg p40_3k_sg ...`)

The label-swap test showed that both stores turn over by counting rules,
blind to change. Biology gates hippocampal plasticity by novelty and
mismatch (dopamine, acetylcholine). Here the gate is **surprise**: the
network's prediction for the incoming example, before it learns from it,
contradicts the label. A surprising example changes the store faster:
- **prototypes:** the nearest prototype moves with step
  max(usual, 0.2) instead of max(1/n, 0.01);
- **reservoir** (the local decoder's code prior): with probability 0.5
  it overwrites a random slot of its class, otherwise reservoir sampling
  as usual.

Surprise costs one extra scoring pass per example (0.3 ms vs 1.55 ms
for learning). Same swap protocol (2 ↔ 3 after Split-MNIST, 8000
examples), 3 seeds; baselines from `swap.py`.

| Condition | Split-MNIST class-IL before the swap | New mapping after 1 / 2 / 4 sleeps (t=2000 / 4000 / 8000) | Final new, by seed | Others at t=8000 | Store new (t=2000 / 8000) |
|---|---|---|---|---|---|
| 9 prototypes, no noise | 89.34 (89.4 / 89.8 / 88.8) | 76.6 / 78.7 / 85.4 | 85 / 90 / 81 | 89.7 | 31.7 / 67.6 |
| **+ surprise-gated update** | **90.06** (89.3 / 90.0 / 90.9) | **82.3 / 89.3 / 91.2** | 88 / 95 / 91 | 89.5 | 42.6 / 62.1 |
| Local decoder, 3000 steps | 89.36 (89.1 / 90.1 / 88.9) | 70.4 / 74.7 / 82.8 | 82 / 84 / 83 | 89.3 | 16.6 / 40.2 |
| + surprise-gated prior | **85.26** (86.3 / 84.2 / 85.2) | 83.7 / 86.4 / 84.6 | 87 / 85 / 81 | 88.5 | 30.8 / 44.8 |

1. **For prototypes, surprise-gating is a free win.** It adapts faster
   after every sleep (+5.7, +10.6, +5.8 points) and better on every seed
   (+3, +5, +10 at the end), with retention unchanged. On the ordinary
   benchmark before the swap it is if anything slightly better (90.06 vs
   89.34). A surprised example only nudges the nearest *mean* by 0.2, so
   hard examples shade the prototype without taking it over.
2. **For the reservoir, the same idea hurts.** The local decoder adapts
   faster after the first sleep (83.7 vs 70.4), but its ordinary
   class-IL falls by 4.1 points (85.3 vs 89.4, every seed). A reservoir
   slot *becomes* the surprising example, so the prior fills up with
   the network's mistakes: before the swap, 23% of its 2/3 replays
   already look like the other digit (7% without gating). Replaying
   hard examples as if they were typical shifts every boundary.
   **Averaging stores tolerate surprise; sampling stores are corrupted
   by it.**
3. **Surprise is brief, because FF adapts fast awake.** Over the
   60,000-example stream the network was surprised only about 1100 times
   (1.8%), mostly in the first ~100 examples of each new class. After
   the swap it is surprised for roughly 100 examples, then it predicts
   the new labels and the gate closes. Then a sleep reverts it, it is
   surprised again, and the store moves again. That loop is why gating
   keeps helping across sleeps.
4. **Surprise-triggered sleep at a threshold of 200 never fired.**
   Between two clock sleeps there were always fewer than 200 surprises,
   so those runs equal the baseline. Rerun at 50 and 25 below.

### Surprise-gated prototypes on the ordinary benchmark (`dreams.py --set sgmnist / sgfashion`)

Surprise-gating made prototypes adapt faster in the swap test, and their
ordinary 3-seed class-IL looked slightly better (90.06 vs 89.34). Checked
with 10 seeds on MNIST (0–2 from the swap runs, 3–9 new) and 3 seeds on
Fashion, every 2000, 0.8×.

| Data | Condition | Class-IL final | By seed | Paired diff vs no gating |
|---|---|---|---|---|
| MNIST (10 seeds) | 9 prototypes, no noise | 89.65 ± 0.44 | – | – |
| MNIST (10 seeds) | + surprise-gated update | 89.81 ± 0.54 | 89.3 / 90.0 / 90.9 / 90.0 / 90.3 / 89.6 / 89.8 / 89.1 / 89.9 / 89.2 | +0.16 (t = 0.73, better on 5/10) |
| Fashion (3 seeds) | 10 prototypes, no noise | 77.29 ± 0.40 | 77.7 / 76.8 / 77.4 | – |
| Fashion (3 seeds) | + surprise-gated update | 75.63 ± 0.38 | 75.9 / 75.1 / 75.9 | **−1.67** (−1.8 / −1.7 / −1.5) |

Fashion per class (gated vs not): pullover 50 vs 58, shirt 23 vs 28,
sandal 80 vs 83, boot 87 vs 91; sneaker 91 vs 85 and bag 92 vs 90 gain.

1. **On MNIST it makes no difference.** The 3-seed +0.7 was noise.
   Gating speeds adaptation (swap test) without costing ordinary
   accuracy there.
2. **On Fashion it costs 1.7 points on every seed.** "The network was
   wrong" mixes up two different things: **the world changed**
   (swap) and **the example is ambiguous** (shirt vs pullover vs coat,
   where the network is wrong about 20% of the time even with no
   change). On Fashion most surprises are ambiguity, so the gate keeps
   dragging prototypes toward borderline images. That blurs exactly the
   confusable classes (pullover −8, shirt −5).
3. **This is Yu & Dayan's (2005) distinction between expected and
   unexpected uncertainty.** In their account acetylcholine signals
   known noise (don't update much) and noradrenaline signals a change of
   context (update fast). A gate that should fire only on change needs
   to discount expected errors. For example, it could fire only when the
   network was **confidently** wrong (large margin for the old label),
   or when a class's error rate jumps above its own running baseline.

### Surprise-triggered sleep (`swap.py --conds proto9_nn_st50 ...`, control `--every 1000`)

In the swap test the other 8 digits collapse between clock sleeps. Can
the network sleep when it needs to? An extra sleep fires once the
number of surprises since the last sleep reaches a threshold (the clock
still runs). A threshold of 200 never fired (see above), so 50 and 25
were tested, and both with surprise-gated prototypes. Control: a plain
clock every 1000 (same 0.8× replay ratio, 25 batches per sleep), which
gives the same 8 sleeps during the swap as threshold 50. All with 9
noise-free prototypes, 3 seeds. Averages are over all probes in the
swap phase, which is fair to methods whose sleeps don't line up with
the probes.

| Condition | Ordinary class-IL before the swap | Sleeps (whole run / swap phase) | New mapping, avg / late (t ≥ 6000) | Others, avg | Others, worst |
|---|---|---|---|---|---|
| Clock 2000 | 89.34 | 34 / 4 | 93.6 / 92.9 | 54.7 | 24.5 |
| Clock 2000 + sleep after 50 surprises | 89.76 | 47 / 8 | 95.7 / 94.6 | 64.5 | 37.7 |
| Clock 2000 + sleep after 25 surprises | 88.43 (one seed 84.3) | 73 / 16 | 96.7 / 97.3 | **72.3** | **48.2** |
| + surprise-gated update, sleep after 50 | 89.81 | 48 / 8–10 | 96.6 / 96.8 | 64.1 | 40.4 |
| **Clock 1000 (control)** | **88.00** | 68 / 8 | 96.2 / **97.5** | 70.4 | 42.9 |

1. **More sleep during conflict helps a lot.** Average retention rises
   from 55% to 64–72% and the worst point from 25% to 38–48%. Adaptation
   improves too (late new mapping 93 → 95–98%).
2. **Timing sleep by surprise adds nothing beyond the extra sleeps.**
   With the same 8 sleeps in the swap phase, a plain faster clock beats
   the surprise trigger (others 70.4 vs 64.5, new 97.5 vs 94.6 late).
   Surprise fires in the first ~100 examples after each sleep. The
   damage to the other digits builds up later, while the network
   predicts the swapped digits correctly and so isn't surprised. It
   keeps training on two classes only, and that shrinks the others. So
   surprise signals the wrong thing.
3. **But a faster clock has a cost when nothing is wrong.** Sleeping
   every 1000 lowers the ordinary benchmark by 1.3 points (88.0 vs
   89.3, every seed), because the same replay split into shorter sleeps
   consolidates less. The surprise trigger only adds sleeps when
   something happens (39 vs 30 before the swap) and doesn't hurt ordinary
   accuracy (89.8). Threshold 25 sleeps too often and becomes unstable
   (one seed 84.3).
4. **What the trigger should measure is interference, not surprise.**
   Tononi and Cirelli's synaptic-homeostasis view gives a candidate:
   sleep pressure builds with the amount of synaptic change since the
   last sleep. Here that would be the accumulated |ΔW| (or the drop in
   goodness of a few stored prototypes, probed while awake). It rises
   steadily during the damaging stretch of wake, unlike surprise.

### Interference-triggered sleep and anytime accuracy (`swap.py --conds proto9_nn_dr20 proto9_nn_pr50 ... --anytime 700`)

Surprise was the wrong trigger for extra sleep: it stops ~100 examples
after each sleep, while the damage keeps building. Two signals that
should grow *with* the damage, both checked every 100 wake examples and
reset by each sleep (the clock every 2000 still runs; at least 200 wake
examples between sleeps):

- **drift** (Tononi & Cirelli's sleep pressure): relative weight change
  since the last sleep, Σ‖W − W_sleep‖² / Σ‖W_sleep‖².
- **probe** (awake replay): at each sleep, 30 stored prototypes are drawn
  and the network's accuracy on them is noted. Awake, the network
  re-classifies them; sleep when accuracy has dropped by the threshold.
  Cost: 0.27 probe items (10 label passes each) per wake example.

A new metric, **anytime accuracy**: class-IL on the classes seen so far,
measured every 700 wake examples wherever the clock is (fixed 2000-image
test subset). All 9 noise-free prototypes, Split-MNIST then the 2↔3 swap,
3 seeds. Extra sleeps are full sleeps (50 batches), so the triggered runs
replay more; the last two rows are clocks with matched replay.

| Sleep trigger | Final | Anytime avg / tasks 3–5 / worst | Sleeps (run / swap) | Replay ÷ wake | Swap: others avg / worst |
|---|---|---|---|---|---|
| Clock 2000 | 89.34 | 68.0 / 54.1 / 24.9 | 30 / 4 | 0.80× | 54.7 / 24.5 |
| Clock 1000, 25 batches | 88.00 | 76.0 / 65.2 / 30.3 | 60 / 8 | 0.80× | 70.4 / 42.9 |
| + drift ≥ 0.02 | **90.28** | 72.0 / 57.9 / 27.7 | 62 / 5 | 1.65× | 59.6 / 31.3 |
| + drift ≥ 0.03 | 89.70 | 70.8 / 57.1 / 28.9 | 51 / 5 | 1.36× | 58.5 / 35.1 |
| + probe drop ≥ 30% | 85.00 (82.6–87.0) | **81.0 / 72.6 / 53.5** | 75 / 13 | 2.01× | **75.4 / 53.1** |
| + probe drop ≥ 50% | 89.11 | 76.1 / 66.7 / 46.6 | 49 / 8 | 1.31× | 64.6 / 42.3 |
| Clock 1250, 50 batches (replay of probe 50%) | 88.93 | 74.7 / 63.3 / 29.1 | 48 / 6 | 1.28× | 59.6 / 25.3 |
| Clock 1000, 50 batches (replay of drift 0.02) | 88.34 | 76.8 / 66.7 / 31.9 | 60 / 8 | 1.60× | 72.1 / 31.4 |

1. **The headline numbers are a best case.** Every final class-IL in this
   README is measured after example 60000, which is a multiple of the
   clock, so it falls right after a sleep. Measured at any time, clock
   2000 averages **68%**, **54% in tasks 3–5**, and drops as low as 25%. The
   calibration trace shows why. Just before each clock sleep in tasks 2–5,
   13–100% of the probed prototypes are misclassified (median ≈ 45%).
   **Between sleeps the network forgets old classes almost as fast as
   with no sleep at all, and each sleep rebuilds them.** The label swap
   only made this visible. The earlier claim that ordinary new tasks
   don't cause this collapse is wrong.
2. **The probe is the right signal, and it mostly raises the floor.**
   Compared with a clock that replays the same amount (1250 vs probe
   50%), it gives slightly higher average anytime accuracy (76.1 vs
   74.7) and a much higher worst point (46.6 vs 29.1; swap worst 42.3 vs
   25.3), with no loss of final accuracy. Sleep comes when a memory
   actually fails, which is mostly in the first few hundred examples of a
   new task, and that is where a clock is blind. A lower threshold (30%)
   raises anytime accuracy to 81% but sleeps too often: final accuracy
   falls and varies more between seeds (85.0, 82.6–87.0), the same
   instability as the surprise threshold of 25.
3. **Weight drift is a task-boundary detector, not an interference
   meter.** Under Adam every step has roughly the same size, so drift
   mainly counts steps that point the same way. That is large when a new
   task starts (0.04–0.18) and small later (0.01–0.03), even though the
   forgetting late in a task is just as bad. At matched replay it loses
   to the clock on anytime accuracy (72.0 vs 76.8) and on swap retention
   (59.6 vs 72.1). It does give the best *final* score (90.28, every seed
   ≥ 89.7). Extra sleeps right at task boundaries help consolidation,
   though with 3 seeds this is suggestive, not established.
4. **What this means.** Sleep triggers only manage the symptom. The
   cause is wake learning: plain FF on a stream of one or two classes
   overwrites the other classes within a few hundred examples. No trigger
   brings anytime accuracy near the final score without doubling replay.
   The fix has to make awake learning less destructive: interleaving a
   little replay into every wake step (Hinton's fast alternation; awake
   sharp-wave replay in the hippocampus), or units whose context gating
   keeps a two-class stream away from other classes' weights (the
   dendritic directions).

### Fast alternation, awake replay and slow wake learning (`swap.py --every 400 / --conds proto9_nn_mix8 / proto9_nn_wl3 ... --anytime 700`)

Between clock sleeps, wake learning erases old classes: anytime class-IL
is 68% against a post-sleep final of 89%. Three ways to make waking less
destructive were tested, all with 9 noise-free prototypes/class, Split-MNIST
then the 2↔3 label swap, 3 seeds:

- **Faster clock** (Hinton's fast alternation): sleep every 400 / 160 / 40
  examples, with fewer batches so replay stays at 0.8×.
- **Awake replay** (hippocampal sharp-wave replay while awake): every wake
  update also includes, on average, `mix` items drawn from the store
  (class-balanced over the classes seen so far), in the same batch. No sleep,
  or together with the clock (hybrid).
- **Slow wake learning** (complementary learning systems, McClelland,
  McNaughton & O'Reilly 1995): the cortex learns slowly while awake, with the
  learning rate multiplied by 0.3 or 0.1. Sleep runs at the full rate.

"Recent share" is the fraction of final test predictions that are 8 or 9,
the last task's classes (true share 19.8%).

| Condition | Final (by seed) | Anytime avg / late / worst | Replay ÷ wake | Recent share | Swap: new avg / others avg / others worst |
|---|---|---|---|---|---|
| Clock 2000, 50 batches | 89.34 (89.4 / 89.8 / 88.8) | 68.0 / 54.1 / 24.9 | 0.80× | 19.8% | 93.6 / 54.7 / 24.5 |
| Clock 400, 10 batches | 77.66 (76.5 / 77.1 / 79.4) | 83.1 / 74.6 / 36.9 | 0.80× | 39.9% | 97.4 / 71.1 / 60.7 |
| Clock 160, 4 batches | 71.53 (74.7 / 63.9 / 75.9) | 85.2 / 77.5 / 55.7 | 0.80× | 46.2% | 97.7 / 69.6 / 55.2 |
| Clock 40, 1 batch | 71.29 (71.7 / 65.0 / 77.2) | 84.4 / 76.4 / 52.7 | 0.80× | 46.4% | 97.8 / 67.5 / 51.7 |
| Awake replay 0.2, no sleep | 75.19 (74.6 / 74.7 / 76.3) | 88.3 / 82.1 / 66.4 | 0.20× | 42.5% | 97.2 / 76.8 / 66.5 |
| Awake replay 0.8, no sleep | 79.05 (79.5 / 76.8 / 80.9) | 91.2 / 86.4 / 75.9 | 0.80× | 38.5% | 96.2 / 81.6 / 76.1 |
| Awake replay 2, no sleep | 78.68 (81.6 / 73.4 / 81.0) | 91.4 / 86.7 / 78.2 | 2.00× | 38.1% | 94.7 / 82.6 / 78.8 |
| Awake replay 0.4 + clock 2000, 25 batches | 83.23 (82.4 / 86.4 / 80.8) | 89.8 / 84.5 / 72.8 | 0.80× | 34.2% | 96.9 / 80.2 / 70.7 |
| Awake replay 0.8 + clock 2000, 50 batches | 82.98 (83.1 / 83.0 / 82.8) | 90.7 / 85.7 / 75.1 | 1.60× | 33.8% | 96.1 / 81.0 / 74.4 |
| Wake lr ×0.3 + clock 2000 | 89.73 (89.8 / 89.6 / 89.8) | 75.3 / 64.2 / 32.0 | 0.80× | 19.9% | 91.0 / 65.8 / 40.0 |
| Awake replay 0.8, wake lr ×0.3 + clock 2000 | 77.00 (75.6 / 77.5 / 77.9) | 87.0 / 79.8 / 65.5 | 1.60× | 39.2% | 93.3 / 78.5 / 73.6 |
| Awake replay 0.8, wake lr ×0.1 + clock 2000 | 78.79 (77.4 / 79.2 / 79.7) | 86.5 / 79.3 / 64.7 | 1.60× | 36.7% | 85.3 / 80.9 / 77.0 |

1. **Interleaving replay into waking prevents the collapse.** Awake replay
   of 0.8 items per update, with no sleep at all, gives anytime **91.2%**
   (clock 2000: 68.0), 86.4% in tasks 3–5 (54.1), and a worst point of 76%
   (25%). During the swap the other digits stay at 82% on average (55%) and
   never fall below 76% (25%). The amount of replay barely matters: 0.2 items
   already gives 88%, and 2 items gives 91.4%. What matters is that old
   classes are in **every** update.
2. **But the final score falls 10 points (79%), and the cause is recency
   bias.** The last task's two classes take 38% of all predictions. Faster
   clocks have the same problem, and worse: a sleep every 40–400 examples
   gives final scores of 71–78% with 40–46% recent share. Adding the
   2000-clock sleeps on top (hybrid) only partly helps: final 83%,
   recent share 34%. A sleep that makes clock-only waking balanced again
   cannot undo the bias that interleaved waking builds up. **So "sleep
   every 2000" isn't working because it is rare, but because it is the only
   time the network trains on a balanced set of one kind of input.**
3. **Slow wake learning alone is a free improvement.** With the wake rate at
   0.3×: final **89.73** (every seed 89.6–89.8, against 88.8–89.8), anytime
   75.3 (68.0), 64.2 in tasks 3–5 (54.1), and swap retention 65.8 (54.7), at
   the same 0.8× replay, with no bias (19.9%). The cost is slightly slower
   adaptation to the swap (new mapping 91.0 vs 93.6 on average). It is the
   complementary-learning-systems prescription: the store learns fast and the
   cortex slowly, so less is erased between sleeps.
4. **Slow waking does not fix awake replay's bias** (77–79% final,
   37–39% recent share with the rate at 0.3× or 0.1×). The bias is not about
   step size.
5. **Hypothesis for the bias: a shortcut cue.** During awake replay, the
   current class always arrives as **real images**, while old classes arrive
   as **smooth prototype averages**. Over 2000 updates a task, "real texture"
   predicts "current class", and a real test image of an old class then looks
   current. The clock-2000 network never sees this pairing in sleep, because
   sleep replays only prototypes. Tested below with a store of real images.

### Slow wake learning: 10 seeds and Fashion (`swap.py --conds proto9_nonoise proto9_nn_wl3 --seeds 3 … 9 --anytime 700`; `--data fashion`)

Paired over 10 MNIST seeds (seeds 0–2 from the runs above), 9 noise-free
prototypes/class, clock 2000, 0.8× replay; and 3 Fashion seeds with
10 noise-free prototypes/class (`results/slow_mnist10`, `slow_wl1`,
`slow_fashion`).

| Data | Wake lr | Final | Anytime avg / tasks 3–5 / worst | Swap: new avg / others avg / others worst |
|---|---|---|---|---|
| MNIST (10 seeds) | ×1 | 89.65 ± 0.41 | 67.5 / 53.6 / 25.1 | 93.8 / 54.6 / 25.1 |
| MNIST (10 seeds) | ×0.3 | 89.52 ± 0.40 | **74.7 / 63.7 / 31.5** | 91.2 / **65.3 / 39.5** |
| | paired diff | −0.12 (t = −0.68, 3/10 better) | **+7.2 (t = 20.7) / +10.1 / +6.4, 10/10** | −2.6 / **+10.7 / +14.4** |
| MNIST (3 seeds) | ×0.1 | 87.48 | 79.9 / 71.2 / 45.6 | 85.7 / 75.8 / 62.1 |
| Fashion (3 seeds) | ×1 | 77.29 (77.7 / 76.8 / 77.4) | 56.4 / 42.5 / 26.5 | 89.3 / 52.1 / 35.3 |
| Fashion (3 seeds) | ×0.3 | 78.15 (78.7 / 77.6 / 78.1) | 62.5 / 50.2 / 34.4 | 88.4 / 57.8 / 42.7 |
| Fashion (3 seeds) | ×0.1 | **78.48** (78.9 / 78.2 / 78.4) | **68.5 / 57.9 / 39.8** | 86.9 / 63.4 / 51.5 |

1. **The anytime gain is robust:** +7.2 points on every one of 10 seeds,
   +10 in tasks 3–5, and +10.7 retention of the other digits during the
   swap. The extra 0.4 in the final score at 3 seeds was noise: over 10
   seeds the final score is unchanged.
2. **There is a dial between stability and plasticity.** At 0.1× the
   anytime score rises to 80% on MNIST, but the final score drops by 2.2
   points and adaptation to the swap slows (new mapping 86%). The awake
   network learns too little that the prototypes don't already hold.
   MNIST's best setting is about 0.3×.
3. **On Fashion, slower waking also raises the final score:** 78.48 at 0.1×,
   higher on every seed, **the best Fashion result so far** (previous best
   77.29), with anytime +12. Denser images and more confusable classes
   reward a slower cortex: each awake step on a shirt disturbs more shared
   features.
4. In complementary-learning-systems terms, the store is the fast
   hippocampus and the network the slow cortex. Sleep was already
   consolidating; slowing the awake rate makes the split real, and costs
   nothing in memory or replay.

### Diagnostic: is awake replay's recency bias a shortcut cue? (`swap.py --conds res9 res9_mix8 --anytime 700`)

If the bias came from "real images = current class, smooth prototypes =
old classes", a store of **real** images (9 per class, reservoir sampling)
should remove it. 3 seeds, clock 2000 (50 batches), with and without awake
replay 0.8.

| Store | Final | Anytime avg / tasks 3–5 / worst | Recent share (true 19.8%) |
|---|---|---|---|
| 9 prototypes, clock 2000 | 89.34 | 68.0 / 54.1 / 24.9 | 19.8% |
| 9 prototypes, awake replay 0.8 + clock | 82.98 | 90.7 / 85.7 / 75.1 | 33.8% |
| 9 real images, clock 2000 | 78.82 | 66.5 / 51.9 / 27.1 | 35.3% |
| 9 real images, awake replay 0.8 + clock | 60.45 | 82.2 / 72.8 / 55.7 | **57.4%** |

1. **Refuted.** With real images the bias is *worse*, both with and without
   awake replay. The shortcut-cue explanation is withdrawn.
2. **New hypothesis: memorised replay against fresh data.** The current
   classes are learnt from thousands of new images and generalise; old
   classes are learnt from a few replay items seen over and over. With awake
   replay each stored item is replayed about 100 times per task, so the
   network memorises those items rather than the class, and real test
   digits of old classes fall back to the well-generalised current classes.
   Prototypes are class averages, which generalise better than 9 raw
   images, so they show the effect less (and score 10 points higher with the
   clock alone). Clock-2000 sleeps escape it because waking erases the
   memorised fit each time, so sleep refits the old classes from scratch
   in one balanced burst. Prediction: anything that makes replay more
   varied (prototype + noise, augmentation, more prototypes) should reduce
   awake replay's recency bias. Untested.

## Files

- `data.py`: MNIST download + cache
- `learners.py`: the six learners, one interface (`fit_batch`, `scores`)
- `bench.py`: protocols, metrics, report
- `sleep.py`: wake/sleep forward-forward with bounded fast stores
- `sleep_cost.py`: sweep of sleep frequency × replay amount
- `swap.py`: label-swap adaptation test (digits a and b swap labels after the five tasks)
- `sandal.py`: shape check of each store's sandal replays (openness, nearest class, style coverage, recon)
- `dreams.py`: network-generated dreams, Hinton's wake/sleep split (`--set dreams`) top-down generative dreams (`--set topdown`) local generator rules (`--set local`) matched code priors (`--set prior`) the one-number-per-synapse fold (`--set fold`) a low-bit / sparse accumulator (`--set cheapc`) smaller priors with 8-bit C (`--set budget`) the source of the remaining gap (`--set gap`) 8-bit codes (`--set codes8`) a regularised contrastive decoder (`--set decay`) longer sleeps with weaker noise (`--set noise`) per-unit steps / dropout negatives (`--set jacobi`) Split Fashion-MNIST (`--set fashion --data fashion`) a two-level generator / longer sleeps there (`--set deep --data fashion`) 3000-step sleeps on MNIST (`--set long`) a nonlinear decoder from sparse random features (`--set expand --data fashion`) per-class confusions (`--set perclass --data fashion`) noise-free prototypes on MNIST (`--set nonoise`) sharpened dreams (`--set sharpen --data fashion`), the noise-free prototype memory curve (`--set frontier --data fashion`) and 10-seed MNIST comparisons (`--set tenseed`)
