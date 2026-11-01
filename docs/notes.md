# Notes

Ideas from discussion, with who raised them and what the benchmark says.
References are in [`papers/README.md`](papers/README.md). Numbers are
from `results/RESULTS.md`, 3 seeds.

---

## The core hunch

### Deep nets are complex to make up for a neuron that's too simple *(you)*
> Modern deep nets pile on architecture to compensate for an
> over-simplified single neuron.

- Support: one cortical pyramidal neuron needs a 5–8 layer net to emulate
  (Beniaguev 2021). A neuron behaves like a 2-layer net (Poirazi 2003).
  Dendrites can compute XOR, which a single artificial unit can't
  (Gidon 2020).
- Counter: richer units don't add expressive power, since a small
  sub-network can stand in for them. Any win has to come from **learning**
  or **efficiency**. GPUs also favour simple units in dense matmuls (the
  "bitter lesson").

### The Elixir/BEAM analogy *(you)*
> One Elixir process is more complex than a JVM thread, but the whole
> system has much less overhead when scaling.

| | JVM threads | BEAM processes |
|---|---|---|
| State | shared | private, isolated |
| Coordination | locks | async messages |
| Scaling | contention grows | ~linear |

| | Backprop | Local learning |
|---|---|---|
| State | one global loss everyone depends on | each layer/unit has its own objective |
| Coordination | update locking, weight transport | send activity forward, learn locally |
| Scaling | waiting and stored activations grow with depth | layers run independently |

- Backprop's update locking is **literally a lock on shared state**.
  Local learning is the actor model for neurons.
- Refinement *(me)*: a BEAM process is *lighter* than a JVM thread, not
  heavier. What's richer is its **semantics** (mailbox, isolation, local
  failure), not its weight.

### ★ Give each unit enough local machinery that it doesn't need global coordination to learn *(me, from your hunch)*
This is the corrected form of the hunch. The aim isn't heavier neurons,
it's neurons that can learn on their own.
- The neuron's complexity may be **mainly what makes local learning
  work**, not extra computing power. The brain may move backprop's global
  credit assignment *into each cell*: segregated dendrites (Guerguiev
  2017), burstprop (Payeur 2021), active dendrites (Iyer 2022).
- Complex parts, simple coordination. That's the BEAM point.

### ★ Backprop inside the cell, none between cells *(me)*
- Forward-forward's only rule is that **no learning signal crosses a layer
  boundary**. It says nothing about how complex a unit is *inside* that
  boundary.
- So a dendritic neuron can be trained with FF: the goodness gradient
  flows from the cell body back into its own branches.
- Biologically real: **backpropagating action potentials** travel from the
  cell body back into the cell's own dendrites and drive plasticity there.

---

## What the benchmark showed

### ★ A smarter learning rule doesn't fix forgetting
| Rule | Split-MNIST class-IL |
|---|---|
| Backprop | 19.7% |
| Feedback alignment | 19.4% |
| Direct feedback alignment | 19.1% |
| Forward-forward (fixed) | 20.1% |

Four very different rules, from exact global gradients to purely local,
all remember only the last task. The rule changes *how* a weight updates,
but the same weights are still overwritten by every task. **Removing the
backward pass doesn't remove forgetting.**

### ★ What fixed it: memory that only accumulates
Random features + least-squares readout: **95.5%**, the same as iid
training.
- The features never change, so nothing gets overwritten.
- The readout keeps only running sums (Σhhᵀ, Σhyᵀ), so the result doesn't
  depend on data order.
- BEAM version: **state that only accumulates can't be clobbered.**
- The Hebbian layer (61%) shows the other side: its features drift with
  each task, so the stored sums go stale.

### ★ Homeostasis keeps coming up
The same failure and the same fix, three times:
1. **Hebbian layer**: 1 unit of 2000 captured every input. Fixed with a
   per-unit threshold that rises when the unit wins too often
   (DeSieno's "conscience").
2. **Forward-forward**: 84% of layer-1 units died during Split-MNIST.
   Fixed with SymBa loss plus a bias nudged toward a 10% firing rate.
   Result: 98.40% on standard MNIST (level with backprop's 98.43%).
3. **LLM mixture-of-experts**: *expert collapse*. DeepSeek-V3 fixes it
   with a per-expert bias nudged by load, **outside the gradient**. That's
   the same rule as (2).

Large-scale engineering and neuroscience (intrinsic plasticity) landed on
the same mechanism. It looks fundamental.

Why FF died: negatives differ from positives only in the 10 label pixels,
so "push negative goodness down" lands on the shared image weights. ReLU
units that go silent get no gradient and never come back. The fix acts on
the bias directly, so it can revive them.

---

## Update locking

*Time-locking of layer updates* (Journé et al.; term from Jaderberg 2017):
under backprop a layer can update only after the signal has gone all the
way forward and the error all the way back.

| | needs before updating | consequence |
|---|---|---|
| **Unlocked** (FF, Hebbian) | its own input and output, available immediately | every layer busy every tick; throughput independent of depth |
| **Locked** (BP, FA) | an error from the top that arrives ~2L steps later | either wait (slow, mostly idle) or keep going and apply the error late (stale weights; activations stored until the error arrives) |

**Unlocked layers learn from what they just saw. Locked layers learn from
a verdict that arrives later from the top.** The price of unlocking: no
direct signal about what would have helped the final answer, which is
part of why these methods trail backprop on hard data.

Practical note *(you: "ditch numpy")*: a faster library doesn't show this.
Layers have to run **concurrently** (one thread per layer). NumPy releases
the GIL in matmuls, so threads with NumPy can do it. GPUs are *worse* for
batch-1 online learning.

---

## MoE is half of the idea *(you)*
> Modern LLMs went for MoE, which is a kind of complex local neuron.

- Matches: conditional computation (context picks which machinery
  runs), **top-k routing = winner-take-all**, and load balancing =
  homeostasis.
- Doesn't match: still trained end to end with global backprop. Experts
  are coarse (millions of params), but the trend is finer: DeepSeekMoE,
  and *Mixture of a Million Experts* with single-neuron experts.
- **LLMs converged on sparse, gated, competing specialists kept balanced
  by homeostasis. They haven't touched each specialist learning locally.**
  That's the gap pdp sits in.

---

## Forgetting is good, if it's the right kind *(you)*
> Forgetting is said to be important, for preventing overfitting.

| | Catastrophic | Adaptive |
|---|---|---|
| Cause | new learning **overwrites** unrelated old knowledge | unused/noisy/outdated information **fades** |
| Effect | harmful | regularisation, room for new learning, keeping up with change |

- Forgetting as a regulariser: Richards & Frankland 2017.
- Our ridge readout never forgets, so it would adapt ever more slowly if
  the data changed. Fix: a forgetting factor (old statistics decay).
- Loss of plasticity (Dohare et al., Nature 2024): long-trained nets lose
  the ability to learn, partly via dying units, the same thing we saw in
  FF.
- ★ **Forget by relevance, never by interference.** Test both
  retention (Split-MNIST) and adaptation (a rule change mid-stream).

## Sleep as a mode with different rules *(you)*
> Implement sleep as a mode that can be turned on and off. Only certain
> things can be processed during sleep.

| | Wake | Sleep |
|---|---|---|
| Input | external stream | off; runs on its own activity |
| Real-time answers | required | not required |
| Learning | fast, local, per example | slow, may reorganise |
| May disrupt behaviour | no | yes |

What only sleep can do:
1. **Replay/consolidation**: bounded fast store → slow weights, mixed with
   old memories (Complementary Learning Systems: McClelland, McNaughton &
   O'Reilly 1995; generative replay: van de Ven 2020).
2. **Synaptic downscaling**: everything shrinks a little and weak synapses
   drop out (Tononi & Cirelli). Controlled forgetting at the right time.
3. ★ **FF's negative phase as dreaming** (Hinton's own suggestion): wake
   learns from real data as positives, sleep learns from the network's
   own generated data as negatives.
4. **Maintenance**: revive dead units, rebalance experts.
5. **Limited coordination**: no deadline, so somewhat global operations
   (e.g. a readout solve) may be acceptable in sleep only.

Fairness: the fast store must be **bounded** (prototypes, statistics or a
generator, not every raw image), and its size and sleep cost are reported.

### ★ Result: sleep works *(built Oct 2026, `sleep.py`)*
- FF + clock-triggered sleep: Split-MNIST class-IL **20% → 90%** (10
  prototypes/class, 3 seeds). Same rule, same net; only the offline phase
  is new.
- **One mean + variance per class (15,680 floats) → 80%.** A tiny
  compressed hippocampus is enough for most of the effect.
- At equal small memory, prototypes beat raw images (90.0 vs 88.6).
  Compression isn't just cheaper, it's better here.
- Downscaling didn't help retention, but it gave 23–38% exact-zero
  weights at no real cost, so it serves sparsity.
- Wake still damages old tasks, and sleep repairs them (0/1 drops to
  ~65%, then back to 99%).
- Open: the store uses labels; "dreams" are prototype + noise, not
  generated by the network itself.

### ★ Result: how much sleep *(`sleep_cost.py`)*
- **0.8× replay is enough** (sleep every 2000, 1600 replays): 90.0% final,
  89.4% task-end. 4× cheaper than the first setting, no loss.
- Below 0.2× it falls off; at 0.05× only rare, long sleeps hold (~76%).
- ★ **Fewer, longer sleeps beat many short ones** at the same total replay.
  Sleeping every 500 is worst at every budget. That parallels real sleep:
  consolidation in long blocks, not micro-naps between every experience.
  Mechanism not established.
- Watch out for measurement timing: *final* is right after a sleep, which
  favours big last sleeps. Task-end accuracy (anywhere in the cycle) is
  the fairer measure.
- Confound caught: pruning ran once per sleep, so frequent sleeps
  over-pruned (58% zero weights, 49% dead units). Pruning should be
  scaled by time awake, not number of sleeps.

### ★ Result: the network's own dreams don't work, the split needs fast alternation *(`dreams.py`)*
- Dreams by **inverting the FF net** (gradient ascent on log p(label | x)
  from pixel noise, no store): 20% → **27%** (task-end 38%), against 90%
  for prototype replay. Dreams are speckle; they score high for a class
  without looking like it (correlation with the real class mean 0.45,
  with the best other class 0.39). ★ **Inverting a discriminative network
  doesn't give it a memory.** Generative replay needs a generative path:
  top-down weights learned locally (Helmholtz-machine wake-sleep, Hinton
  1995), or the store as a starting point.
- **Hinton's split** (wake positives only, sleep negatives only) at every
  2000: fails, 13–23%, task-IL 56–67%, up to 53% dead units. Every 100:
  69% task-end, 91% task-IL. Discrimination lives in the image × label
  interaction; a paired loss isolates it within a batch, a time-split loss
  only after the phases average out. So phases must alternate fast.
- ★ That reverses "fewer, longer sleeps": what's right depends on what
  sleep does. If sleep only *consolidates*, it can be rare and long. If
  sleep is where a whole half of the learning signal lives (negatives),
  it has to be frequent, i.e. more like a theta cycle than a night.
- Negative-only phases kill units; homeostasis only ran awake. Running it
  in sleep too is the obvious fix to try.

### ★ Result: top-down dreams work, if the generator only accumulates *(`dreams.py --set topdown`)*
- Helmholtz-style top-down weights G (layer → layer below), learned awake
  and locally, plus a class prior over top codes. Dreams = sample a code,
  run it down. No images stored.
- **Ridge rule** (running sums Σhhᵀ, Σhxᵀ; solve G in sleep): **86.8%**
  class-IL, dreams correlate 0.97 with real digits, all ten classes
  recognisable at the end. Prototype replay: 90.0%.
- **Delta rule: 32%. Per-unit 1/activity rule: 35%.** The generator
  forgets just like the classifier: old dreams become blends of the new
  digits.
- ★ Third time the same lesson: **order-independent, accumulate-only
  statistics are what survive a non-stationary stream** (ridge readout,
  ridge generator, and prototypes are running means too). Gradient-style
  updates with any fixed or decaying step don't.
- Cost: the sums are 644k floats and the solve is O(H³), so it's no cheaper
  than prototypes. Open: a *local*, accumulate-only rule that reaches the
  ridge solution (e.g. per-unit sums with lateral decorrelation), or
  sleep-time consolidation of a delta-rule generator on its own
  pre-drift dreams.

### ★ Result: a local generator rule *(`dreams.py --set local`)*
- **Contrastive wake-sleep** (Hebbian sums of real data while awake,
  anti-Hebbian on its own decoded dreams while asleep, per synapse):
  **84.2%**, 2.5 points under ridge, no `hhᵀ`, no solve. It's the
  Boltzmann/wake-sleep positive and negative phase, used for the
  generator, and it finally gives sleep's negative phase a job that works.
  Extra memory is still an accumulator the size of G (393k).
  More sleep steps hurt (300: 78.6%): the fixed point is least squares
  under the *prior's* code statistics, not the data's.
- **Outstar** (each unit averages what it fired for, 500 extra floats):
  72–76%. Can't be overwritten, but the dreams are blurry class means, no
  variety. Averaging gives up the cancellation between units that least
  squares uses to sharpen.
- Smoke tests on two tasks over-predicted outstar (68% vs ridge 64%).
  ★ Two-task smoke tests rank methods unreliably; confirm on all five.
- Open: make the prior match the real code statistics (e.g. a few
  stored real codes per class, or full code covariance per prototype);
  fold `C` into `G` so a synapse holds one number, not two.

### ★ Result: matching the prior fixes the contrastive rule *(`dreams.py --set prior`)*
- Prior = a reservoir of real codes (20/class): contrastive **87.5%**
  (from 84.2), and no longer hurt by more sleep steps. The fixed point was
  the problem, as diagnosed.
- Ridge with the same real codes: **89.6%**, level with prototype replay
  (90.0). The local rule stays ~2 points under the solve.
- 10 real codes/class (50k floats): 86.6%. Codes (500 floats) are a
  better memory than raw images (784): 87.5 at 100k vs 87.7 at 157k.
- PPCA prior (Sanger, rank 20): 85.1%, worse with more steps. ReLU-clipped
  Gaussian samples aren't sparse like real codes. Second moments alone
  aren't enough.
- ★ **Dreams and the negative phase must sample the same prior** (mixing
  them: 78.6%). The generator has to be fitted for exactly the codes it
  will be asked to decode.
- Total memory is still no lower than the 10-prototype image store,
  because of the 393k accumulator C. Next: fold C into G.

### ★ Result: folding C into G fails *(`dreams.py --set fold`)*
- One number per synapse (Hebbian into G while awake with a 1/t step,
  anti-Hebbian paid back in sleep): best **65.4%** (κ = 4), against 87.5%
  with C. Dreams lose class identity.
- Why: the 1/t stochastic approximation keeps old data with weight
  (s/t)^(κλ/E|h|²). Weak code directions never converge, strong ones
  forget. No single κ fits both; the spread of code variances is the
  problem.
- ★ **The accumulator is the memory.** An accumulate-only sum and a
  weight being corrected can't share one number. That matches the
  recurring lesson: order-independent sums are what survive the stream.
- Possible ways round it: per-unit (not global) step sizes, normalising
  each direction by its own variance (whitened codes), or low-precision or
  sparse C rather than none at all.

### ★ Result: an 8-bit accumulator is free *(`dreams.py --set cheapc`)*
- Rewrite C/n = diag(N/n)·F with F the outstar weights (activity-weighted
  mean input per unit, in [0, 1]). F quantises without a scale.
- 8-bit F with stochastic rounding: **87.2%** vs 87.5% float, at a quarter
  of the memory (99k floats). 4-bit: 83.2%. 2-bit: 61.8%.
- Top-k per row is much worse: top-200 78.1%, top-100 56.6%, top-50 34.7%.
  Templates aren't sparse (dim pixels carry shape), the support freezes
  early, and indices cost more than the values. **Quantise, don't
  sparsify.**
- ★ Link to the outstar result: outstar weights alone decode badly (72–76%),
  but as the *statistics* a contrastive sleep solves against they're
  enough. Same numbers, different use.
- Fits criterion 6 (robust to imprecision): the accumulator tolerates
  8-bit storage, as long as rounding is stochastic.
- Total is still 199k floats (with the 20-code prior) vs 157k for the
  prototype store. Next: 10 codes/class + 8-bit C (≈149k).

### ★ Result: under the prototype budget, 3 points short *(`dreams.py --set budget`)*
- 10 codes/class + 8-bit C: **86.6%** at 149k floats, vs prototypes 90.0%
  at 157k. 5 codes/class: 84.6% at 124k.
- The accumulator (99k) is now most of the memory; the prior shrinks
  gracefully. (Blamed the decoder here at first; wrong, see below.)

### ★ Result: the gap is dream diversity, not the decoder *(`dreams.py --set gap`)*
- Ridge on 10 codes/class: **86.5%**, level with local contrastive (86.6).
  Ridge's 89.6 needed 20 codes/class; the prior only feeds its dreams.
- Contrastive's exact fixed point (non-local solve): 81.0%, recon 6× worse.
  100 codes in 501 dims is rank-deficient; **early stopping is the
  regulariser**. 1000 steps: 86.0%. Preconditioning would hurt.
- ★ Lever: more distinct codes per class in the same memory (8-bit codes).

### ★ Result: 8-bit codes are free, but more codes barely help *(`dreams.py --set codes8`)*
- 40 codes/class at 8 bits + 8-bit C = 149k: **87.2%** (task-end 87.6).
  9 prototypes/class = 141k: **89.9%**. 80 codes: 88.2% at 200k.
- Revises the gap result: ridge gains 3 points from 10→20 codes, the
  contrastive rule 0.6. Past ~10 codes/class the early-stopped decoder is
  the limit after all; its exact fixed point overfits. Need a decoder
  that is regularised *and* converged: e.g. explicit weight decay with
  many more steps, or negatives with real-code statistics beyond the
  stored ones.

### ★ Result: weak decay + more steps closes part of the gap *(`dreams.py --set decay`)*
- Ridge on 40 8-bit codes: **90.3%**, matching prototypes (89.9). The
  decoder is the whole remaining gap.
- Contrastive with per-synapse decay 0.3, 1000 sleep steps: **88.3%**
  (from 87.2). Decay 1: 87.0; decay 3: 86.4. Noisy negatives ≈ decay 1:
  87.7, better than explicit decay.
- Less regularisation is better, but weak directions converge at rate ~λ,
  so the local rule can't reach ridge's λ = 0.01. Next: decay 0.1 with
  noise instead of decay, ~3000 steps, or a better-conditioned negative
  distribution (more codes).
- Wake recon error doesn't track accuracy (best run, worst recon).

### ★ Result: λ ≈ 0.3 is the sweet spot; noise loses on long sleeps *(`dreams.py --set noise`)*
- Decay 0.3, 3000 steps: **88.7%** (task-end 88.8, level with prototypes'
  88.5; final 1.2 under).
- Noise 0.1: 86.0 (overfits 400 codes). Noise 0.3: 87.8, below decay 0.3.
- Math: Gaussian noise = decay only in expectation; the 64-code batch
  estimate is noisy and a constant step never averages it out. Good early
  (extra regularisation), bad late. Polyak averaging or a decaying step
  would fix it.
- Biology angle to test next: unreliable synapses = multiplicative
  (dropout) noise = per-unit ridge scaled by each unit's activity (Wager
  et al. 2013), and per-presynaptic-unit learning rates (Jacobi
  preconditioning) to converge in 300 steps instead of 3000.

### ★ Result: dropout dreams, no decay needed *(`dreams.py --set jacobi`)*
- **Dropout negatives keep 0.8 (+ Polyak): 89.0%**, 0.9 under prototypes;
  task-end 89.1 without averaging. Ties decay 0.3 within noise, with no
  decay term: synapses fail 20% of the time during dreams.
- 300 plain steps are already enough (88.5 = 3000 steps): the decoder
  converges across sleeps via warm starts. The earlier "more steps" gains
  were really step size + decay. Jacobi had nothing to fix (88.4).
- ★ Centring alone collapses (53.7%): removing the mean direction leaves
  a far worse-conditioned problem; class differences live in the slow
  directions. Jacobi restores it. The mean was doing the conditioning.
- Math ↔ biology agreement: dropout = activity-scaled ridge = unreliable
  synapses. The best number came from the most biological mechanism.

### ★ Result: Split Fashion-MNIST *(`dreams.py --set fashion --data fashion`)*
- Ceiling (FF, one shuffled pass): 83.4% (MNIST 96.3). No sleep: 19.9%.
- Prototypes 75.6 (91% of ceiling), ridge on 40 8-bit codes 75.8: image-free
  dreams still match images with a good decoder. Sleep transfers without
  retuning; no Bartunov collapse.
- Local contrastive with dropout: 72.8 (gap 2.8, was 0.9 on MNIST); decay
  72.1. Loss is on the coat/sandal task (65 vs 79), probably coat vs
  pullover/shirt. Recon 0.130 vs ridge 0.070 (denser images).
- Shirt/sneaker task ≈ 55% everywhere (probably shirt): worst
  interference.
- Next: a sharper local decoder for dense images (two-level generator, or
  more sleep steps here).

### ★ Result: a sharper local decoder on Fashion *(`dreams.py --set deep --data fashion`)*
- One level, dropout 0.8 + Polyak, **3000 steps: 74.6** (1000: 72.8).
  Gap to prototypes back to 1.0, as on MNIST. Every seed up; memory
  unchanged. Dense images need the slow directions converged.
- Two levels (prior on layer 2): ridge 73.2 (vs 75.8 one level), so the
  architecture costs 2.6 before any local rule. Local two-level 72.0–72.2,
  no gain from more steps. Same as MNIST (85.2 vs 86.8): layer 2 is the
  wrong code to store.
- Wake-sleep bias seen directly: the bottom decoder trained on dreamed
  layer-1 codes decodes real ones badly (MSE 0.23 vs 0.066 for ridge).
- Recon got worse with 3000 steps (0.152 vs 0.130) while accuracy rose.
- Remaining gap: coat/sandal task (63 vs 79 for prototypes).
- Next: check 3000 steps on MNIST; or put the second level *below* the
  code instead (a nonlinear decoder from layer-1 codes: hidden units
  between code and pixels).

### ★ Result: 3000 sleep steps on MNIST *(`dreams.py --set long`)*
- Dropout 0.8 + Polyak, 3000 steps: **89.36 ± 0.52** (1000: 88.99; seeds
  +0.5, +1.1, −0.5: within noise). Prototypes 89.88, ridge 90.33.
- Gap to prototypes 0.9 → 0.5; task-end 89.0 > prototypes' 88.5.
- Small gain on sparse MNIST, large on dense Fashion (+1.7): fits the
  slow-direction reading. Use 3000 steps as the default.
- Statistically level with prototypes at the same memory; a 0.5 gap
  needs ~10 seeds/arm to resolve.
- **Withdrawn** by the 10-seed run: gap −0.74, t = −2.37 (see below).

### ★ Result: nonlinear local decoder on Fashion *(`dreams.py --set expand --data fashion`)*
- Decoder reads [code, k-WTA sparse random features of the code] (fixed
  fan-out, 100 active, divisively normalised: granule cells, Marr–Albus).
  Only the readout learns, with the same contrastive rule.
- Local: 1000 features 75.3, **2000 features 75.6 = prototypes** (code
  only 74.6); every seed up. Ridge + 1000 features **76.5**: first
  image-free decoder above the image store.
- Target missed: coat/sandal unchanged (60–65 local, 70 ridge, 79
  prototypes). Gains on shirt/sneaker (+2 to +7) and bag/boot. So the
  coat/sandal gap isn't decoder sharpness.
- Memory: 543k total for the local version (C grows with features) vs
  157k prototypes. Random weights cost a seed.
- Sleep statistics in 2500-d from ≤ 400 stored codes are rank-deficient;
  dropout carries the regularisation.
- Next: log per-class accuracy to find what prototypes do on coat/sandal;
  cheaper C (4-bit features, or features only in C, sparse).

### ★ Result: per-class confusions on Fashion *(`dreams.py --set perclass --data fashion`)*
- Runs now log test confusion after each task + replay confusion / spread.
- Coat (84 proto vs 72–73 decoded) is a **trade with shirt**, the next
  task's confusable class: prototypes keep coat by stealing shirt (shirt
  19%; 36% of shirts called coat). Local decoder: coat 73, shirt 44. Over
  coat + shirt the local decoder is best (117 vs 103).
- Sandal is a **real loss**: local 58% (28% → boot), ridge 69, proto 75.
  Footwear sum: local −18 vs prototypes. Blur fills the gaps between the
  straps, so replayed sandals look like closed shoes.
- Same mechanism in prototypes: pixel noise fills those gaps. **No-noise
  prototypes: 77.29 ± 0.40** (+1.7, every seed), sandal 83. Replay
  variety doesn't help; half the real spread is fine.
- The network judges replays as their own class (87–100%) under every
  method: replays are right but simpler than the real class.
- Next: no-noise prototypes on MNIST; a decoder that can draw holes.

### ★ Result: noise-free prototypes on MNIST *(`dreams.py --set nonoise`)*
- No effect on MNIST: 9 protos 89.34 vs 89.88 noisy, 10 protos 89.71 vs
  89.99 (within noise; task-end +1.2 / +0.1). Fashion's +1.7 is a
  thin-structure (sandal) effect.
- Without noise the variance isn't stored: **9 protos = 70.6k floats**
  at 89.3%, level with the local decoder (89.4%) at under half its 149k.
  The memory argument for the local decoder is gone; its case is that it
  stores nothing image-shaped.
- ProtoStore.floats() now counts the variance only when noise is used.

### ★ Result: sharpened dreams on Fashion *(`dreams.py --set sharpen --data fashion`)*
- Centre-surround unsharp mask on each dream (lateral inhibition, no
  learning). Ridge **76.62 ± 0.14** (+0.9, every seed): gain on shirt
  (28 → 50) and coat, not sandal (69 → 71).
- Local decoder: **hurt** (74.6 → 73.3 at α 1, 72.6 at α 2, every seed).
  It amplifies the unconverged decoder's pixel noise, like prototype
  noise did. Replay spread up to real level, accuracy down.
- Sandal unchanged by sharpening for both → the deficit isn't edge
  contrast; probably shape averaging across sandal styles (guess).
- Next: nearest-neighbour check of decoded sandals; denoise local dreams
  (more steps / Polyak window) before sharpening.

### ★ Result: noise-free prototype memory curve on Fashion *(`dreams.py --set frontier --data fashion`)*
- 3 / 5 / 10 protos, no noise: 72.7 / 75.2 / 77.3 at 23.5k / 39.2k / 78.4k
  floats. 5 protos (39k) beat the local decoder (74.6 at 149k).
- Sandal tracks protos (69 / 76 / 83): supports shape averaging across
  styles as the sandal problem.

### ★ Result: 10 seeds on MNIST *(`dreams.py --set tenseed --seeds 3 … 9`)*
- Local 3000 steps **89.13 ± 0.55**; protos 9 + noise 89.88; protos 9
  no noise 89.65 (70.6k floats).
- Paired: local −0.74 (t −2.37), −0.51 (t −2.35): **the gap is real**.
  The 3-seed "level" claim (`long`) is withdrawn.
- Local decoder: within ~0.5 on MNIST, 2.7 on Fashion, at 2–4× the
  memory of the best image store. Its case is image-free, own dreams.

### ★ Result: sandal shape check *(`sandal.py`)*
- Shape-averaging hypothesis mostly **wrong**: all stores hit 7–8/8
  sandal styles; prototypes are averages too (openness 0.27 vs real
  0.52, 47% nearest-real is a sandal). No shape metric ranks the methods.
- What tracks sandal accuracy is **fine-detail realism** (high-pass corr
  with nearest real sandal): protos 0.33 → 83, 3 protos 0.22 → 69, ridge
  0.21 → 69, local 0.08 → 62 (real vs real: 0.29).
- Picture: prototypes = coherent blurred single shapes; ridge = ghosted
  superpositions (heel over flat); local = salt-and-pepper speckle.
- Noise on identical prototypes costs sandal 8 → random detail hurts.
- Not stale codes: recon of real sandals from current codes is no more
  sandal-like than the dreams.
- Next: de-speckle the local decoder (smoothness prior on projective
  fields / high-pass decay / 3×3 lateral excitation on dreams).

### ★ Result: label-swap adaptation *(`swap.py`)*
- 2 ↔ 3 swapped after the 5 tasks, 8000 examples of only those two.
- Awake: all methods adapt within 500 examples (96%). Each sleep reverts
  to 70–79% because the store replays the old mapping. After 4 sleeps:
  prototypes 85, local 83, ridge 78; no sleep 99 (others 1%).
- Store turnover follows the counting rule: prototypes (lr floor 0.01)
  68% new, reservoir (40/n) 38–40%.
- **Between sleeps, conflict wrecks the other 8 digits (90 → 30–43%)**;
  sleep restores them. Ordinary tasks don't do this; contradicting
  labels drive large FF updates.
- Next: surprise-gated store turnover; conflict-triggered sleep.

### ★ Result: surprise-gated store turnover *(swap test)*
- Surprise = prediction before learning ≠ label (1.8% of the stream).
- Prototypes + surprise step 0.2: **adapt +6 to +11 points after each
  sleep** (final 91.2 vs 85.4, every seed up), retention same, ordinary
  class-IL 90.06 vs 89.34. Free win.
- Reservoir + surprise overwrite (p 0.5): **ordinary class-IL −4.1**
  (85.3): the prior fills with mistakes. Averaging stores tolerate
  surprise, sampling stores are corrupted by it.
- Surprise-triggered sleep at 200 never fired (fewer surprises per
  interval); rerunning at 50 / 25.

### ★ Result: surprise-gated prototypes, ordinary benchmark *(`--set sgmnist / sgfashion`)*
- MNIST 10 seeds: 89.81 vs 89.65 (+0.16, t 0.73): no effect.
- Fashion: **75.63 vs 77.29 (−1.7, every seed)**; pullover −8, shirt −5.
- "Wrong" mixes up change with ambiguity; on Fashion it's mostly
  ambiguity. Yu & Dayan 2005: expected (ACh) vs unexpected (NE)
  uncertainty. Next: gate on *confident* errors, or on a class's error
  rate jumping above its running baseline.

### ★ Result: surprise-triggered sleep *(swap test; control clock 1000)*
- Extra sleep after 50 / 25 surprises: avg retention during conflict
  55 → 64.5 / 72.3, worst 25 → 38 / 48; adaptation 93 → 95–97.
- **Matched control (clock 1000, same 8 swap-phase sleeps) does better**
  (others 70.4, new 97.5 late): surprise timing adds nothing. Surprise
  ends ~100 examples after each sleep; the damage builds afterwards.
- Clock 1000 costs 1.3 on the ordinary benchmark (shorter sleeps); the
  surprise trigger doesn't (89.8, 39 vs 30 sleeps before the swap).
- Next: trigger sleep on **interference**: accumulated |ΔW| (Tononi's
  sleep pressure) or the awake goodness drop of a few stored prototypes.
  Gate store turnover on *confident* errors (Yu & Dayan).

### ★ Result: interference-triggered sleep; anytime accuracy *(swap test + `--anytime 700`)*
- **Final class-IL is a best case**: 60000 is a multiple of the clock, so it's
  measured right after a sleep. Anytime accuracy (every 700 examples) at clock
  2000: **68% avg, 54% in tasks 3–5, worst 25%**. Between sleeps a median 45%
  of probed prototypes fail on *ordinary* tasks too; sleep rebuilds them.
- Probe trigger (30 stored prototypes re-classified awake; sleep at a 50% drop):
  at matched replay vs clock 1250, anytime 76.1 vs 74.7, **worst 46.6 vs 29.1**,
  final unchanged (89.1). Threshold 30%: anytime 81 but final 85 (oversleeping).
- Drift trigger (‖ΔW‖² since sleep): fires at task starts only (Adam makes steps
  equal size); loses to matched clock on anytime/swap, best final (90.3, 3 seeds).
- Triggers treat the symptom. The cause: FF wake learning on a 1–2-class stream
  erases the other classes in a few hundred examples. Next: interleave replay into
  wake (fast alternation / awake replay), or context-gated units.

### ★ Result: fast alternation and awake replay *(`--every 400/160/40`, `proto9_nn_mix*`)*
- Awake replay (0.8 replayed items in every wake update, no sleep): anytime
  **91.2%** (clock 2000: 68), worst 76 (25), swap retention 82 (55). Even 0.2
  items gives 88%: old classes need to be in *every* update, and the amount
  matters little.
- But final drops to 79% (hybrid with clock sleeps 83%): **recency bias**, with
  the last task taking 34–46% of predictions. Faster clocks (40–400) are worse
  (final 71–78). Clock 2000 works because sleep is the only balanced fit.
- Not a shortcut cue: a store of real images makes the bias *worse* (57%).
  Hypothesis: replay items get memorised (about 100 replays each per task) while
  current classes generalise from fresh data. Test: more varied replay.

### ★ Result: slow wake learning = complementary learning systems *(`proto9_nn_wl3`, `proto10_nn_wl1`)*
- Wake lr ×0.3, sleep at full rate. MNIST, 10 seeds: anytime **+7.2 on every
  seed** (t = 20.7), tasks 3–5 +10, swap retention +10.7, final unchanged
  (−0.12), adaptation −2.6. ×0.1: anytime 80 but final −2.2.
- Fashion: ×0.1 → **78.48% final (new best, every seed up)**, anytime +12.
- Free in memory and replay. Fast store (hippocampus) and slow cortex, as
  McClelland, McNaughton & O'Reilly (1995) prescribe.

## Open experiments

1. **Two-compartment FF neuron.** Image on the "basal" input, candidate
   label on a separate "apical" branch, positive = they agree. This
   replaces Hinton's label-in-pixels hack. FF already tries all 10 labels
   at test time, so **label-gated branches need no task identity**. Watch
   the sharing trade-off: gate too hard and it's 10 separate networks
   whose goodness scales don't compare.
2. **Context-gated dendritic units on Split-MNIST**, with context
   *inferred from the input* (no task label). Does a smarter unit fix what
   a smarter rule didn't?
3. **Locally learned MoE.** Experts trained with FF or Hebbian rules, a
   router with homeostatic bias. Does routing alone protect old digits?
4. **Pipelined, one thread per layer.** Measure throughput and
   wait-before-update for BP vs FF vs Hebbian at 2/4/8 layers, with wide
   layers (2000–4000) so threads have enough work.
5. Read SFFA (FF for continual learning) and DeeperForward (dead neurons
   in FF). Both hit our exact problems.
6. ~~Sleep on Split-MNIST.~~ Done: 20% → 90%. ~~Less sleep~~ done: 0.8×
   is enough. Next steps:
   - an **unlabelled** store (e.g. the Hebbian layer as hippocampus);
   - ~~true dreams~~ done: inverting the FF net fails (27%).
     ~~Generative top-down path~~ done: 86.8% with a ridge-rule generator,
     ~33% with delta rules. ~~Local rule~~ done: contrastive
     wake-sleep 84.2% (no solve, still 393k accumulator), outstar 72–76%
     (500 floats). ~~Matched prior~~ done: real-code reservoir 87.5%
     (ridge on it 89.6%). ~~Fold C into G~~ done: 65.4%, C is needed.
     ~~Shrink C~~ done: 8-bit free (87.2%), 4-bit 83.2%, top-k bad.
     ~~Under the prototype budget~~ done: 86.6% at 149k (prototypes 90.0
     at 157k). ~~Decoder gap~~: none, ridge on 10 codes is 86.5%; the
     gap is dream diversity. ~~8-bit codes~~ done: 40 codes/class 87.2% at
     149k vs 9 prototypes 89.9% at 141k. ~~Regularised decoder~~ done:
     decay 0.3 + 1000 steps 88.3% (ridge on that prior 90.3). ~~Weaker
     noise~~: λ 0.3 is optimal, decay 0.3 + 3000 steps 88.7%. ~~Jacobi,
     dropout~~ done: dropout keep 0.8 89.0%, 300 steps suffice. Next:
     dropout at 300 steps. ~~Fashion-MNIST~~ done: prototypes 75.6, ridge
     75.8, local 72.8 (ceiling 83.4). ~~Sharper decoder~~ done: 3000
     steps 74.6; two levels hurt (ridge 73.2). ~~3000 steps on MNIST~~:
     89.4 (gap 0.5, level within noise). ~~Nonlinear decoder~~: random
     expansion, local 75.6 = prototypes on Fashion, ridge 76.5; coat/sandal
     unchanged. ~~Per-class~~: coat = trade with shirt, sandal = blur
     fills straps; no-noise prototypes 77.3. ~~No-noise on MNIST~~: no effect, but half
     the memory (70.6k at 89.3). ~~Sharpening~~: ridge +0.9, local −1.3,
     sandal unchanged. ~~Frontier~~: 5 no-noise protos (39k) beat local.
     ~~10 seeds~~: local −0.5 to −0.7, significant. ~~Sandal shape~~: not averaging; incoherent detail (ghosts,
     speckle). Next: de-speckle the local decoder;
   - ~~Hinton's split~~ done: fails at every 2000, 69% at every 100. Try
     homeostasis during sleep, and faster alternation (every 10–50).
     ~~Fast alternation~~ done: faster clocks and awake replay fix anytime
     accuracy (91%) but add recency bias (final 71–83); slow waking (×0.3)
     is the free win (anytime +7, final same; Fashion 78.5). Next: more
     varied replay to remove awake replay's bias; slow waking + probe trigger.
7. ~~**Adaptation test.**~~ done (`swap.py`): stores turn over by counts, conflict wrecks retention between sleeps. Swap two digits' labels mid-stream and measure
   how fast each method drops the old rule.

- 2026-10-31 12:05 UTC+8: the never-sleeping FF cortex at wake LR x1 is not a slow learner on a label swap (new label
  49-69% within 300 ticks, ~85% by 1500), so cortex-vetted store confirmation is useless at that rate. A CLS-style
  check needs the learning-rate gap; testing wake LR x0.1 (results/gate_probe_s49wl1).
- 2026-10-31 13:00 UTC+8: with a genuinely slow cortex (wake LR x0.1, old name kept ~60% for 1500 ticks)
  cortex-vetted confirmation still does nothing (s49 swap 49.0 vs 49.3, remap latency 76 vs 77), and the slow cortex
  costs the never-sleeping system 2.5 lifetime and 19 swap points. Impostor confirmations are closed as the cause of
  remap delay; z-window length explains ~2 ticks (z2/z3 under the +0.8 bar). The remaining ~16-tick delay is the
  c1.5 rival test on the first conflicts. A learning-rate gap only pays with consolidation (sleep), as CLS says.
- 2026-10-31 14:00 UTC+8: three ways to lift the rival veto sooner (streak, CUSUM, CUSUM with retroactive credit) all
  leave s49 first remap at 73-77 ticks. Treat the remap as a change detector: Lorden's CUSUM bound says delay scales
  with log(false-alarm interval) over per-hit information, so the remaining swap gap is the price of zero false remaps
  unless the store code makes each hit more informative (pattern separation).
- 2026-10-31 16:00 UTC+8: the FF cortex's hidden code separates 4 from 9 about 3x better than pixels, but it is not a
  stable index: FF trains with the label in the input, so after a 4<->9 swap its top layer moves 9 images to where 4s
  were (stale-prototype impostor rate 0.62). The store should keep pixels and measure similarity through the current
  layer-1 code at query time (0.045 vs pixels 0.077), not store codes. Random and Hebbian expansions are both closed.
- 2026-10-31 18:00 UTC+8: a store read through the cortex is a better classifier but not a better complement:
  at the same gate firing rate it gives exactly the pixel store's lifetime and returns. What a store adds is its
  accuracy where the cortex is wrong, so that is the number to optimise next, not store-only accuracy.
- 2026-10-31 19:00 UTC+8: after a return the cortex is wrong far more often (46% vs 3%), but the store is not more
  reliable; a gate that keys on how recently a label came back moves the lifetime/returns frontier outward for the
  first time (+1.6 returns at matched firing). Expected uncertainty (Yu & Dayan 2005) belongs in the router.
- 2026-10-31 20:00 UTC+8: the store's own relative margin is the router's signal. A plain "store disagrees with
  margin >= 0.3" route beats every FF-margin and recency gate tried since 18:00 on both lifetime and returns; the
  recency gate only looked like an outward move because the baseline frontier left this route out. Always include
  the simplest store-only route in a frontier.
- 2026-10-31 21:00 UTC+8: arbitrate, don't threshold. An online table of how often the store vs the cortex is right
  on disagreements, binned by both systems' margins (Daw, Niv & Dayan 2005), beats every fixed store-margin route on
  held-out worlds: +1.5 to +1.9 returns at matched firing with higher lifetime, and the highest lifetime seen yet
  without the old FF-margin route. Next: label-noise kill test, then in-loop.
- 2026-10-31 22:00 UTC+8: don't forget on a clock. Decaying the arbiter's counts buys back swap only by throwing
  away the stable-period evidence that makes the store trustworthy at a return (about -2 returns for +1 swap).
  Volatility should set the forgetting rate (Behrens et al. 2007): forget when the change detector fires.
- 2026-10-31 23:00 UTC+8: forgetting on the store's remap is too late (the store is wrong for ~75 ticks before it
  remaps), and an error-streak veto gives the highest lifetime yet but loses ~3.5 returns because streaks go stale
  while a class is away. Surprise belongs to the current episode of a class; a return should clear it.
- 2026-11-01 00:00 UTC+8: scoping the error-streak veto to a class's current episode, or to a consistent (store, FF)
  label pair, does not fix it: all variants trade about 1.5 returns per swap point. The streaks on returns are fresh,
  not stale; the veto fails because it ignores how rarely the store is wrong there. Error evidence must be weighed
  against the arbiter's own predictive probability (Adams & MacKay 2007), not used as a hard switch. Veto family closed.
- 2026-11-01 01:00 UTC+8: softening the veto does not help. Shrinking class-local error counts toward the arbiter's
  own bin prediction (partial pooling, any strength, with or without reset or decay) stays on the same trade line,
  about 1 return lost per swap point gained. At the router a swap and a return look identical: one system was right
  and now is wrong. Class-local reliability in the arbiter is closed; swap speed must come from the store's remap.
- 2026-11-01 02:00 UTC+8: swap speed was in the store after all. Counting a contradiction twice when the store was
  confident about the old label (margin >= 0.3) cuts first-remap latency by 10-19 ticks and adds 2.4-4.3 points of
  swap first 300 in every world, with returns and lifetime unchanged, the first swap gain since 18:00 that costs no
  returns. It fails its pre-registered bar only on rare transient false remaps (2 of 21 runs). A mismatch counts in
  proportion to the expectation it violates (Kumaran & Maguire 2006). Next: a guard against false remaps, then label noise.
- 2026-11-01 03:00 UTC+8: the confident-remap gain is real (latency -19 ticks on 24 fresh seeds) but unsafe:
  one fresh stream in 12 ping-pongs 4<->9 for 19000 ticks (68 remap events), and requiring 3 or 4 contradictions
  first only delays the first flip. The default store had no false remaps on 57 runs. The store needs hysteresis
  (Wills et al. 2005): moving a prototype back must cost more than moving it in the first place. z5 stays default.
- 2026-11-01 04:00 UTC+8: hysteresis on the confident remap (a recently moved prototype gets no double credit and
  needs one more contradiction) passes its held-out bar with the n4 floor (0 false remaps on 18 fresh streams,
  latency -13 ticks) but only because plain n4 also passes there. On the bad stream it ends the 4<->9 ping-pong
  after 22 events instead of 68 but cannot stop the first false move. Same-seed s49f/bg002f runs share their
  pre-swap stream, so count streams, not runs. Flicker belongs in retrieval, not storage (Jezek et al. 2011).

- 2026-11-01 05:00 UTC+8: copy, then prune works as storage and fails as retrieval. Keeping the old prototype and
  adding a relabelled copy, deleting whichever stops winning, takes seed 26 from 68 false remap events to 1 and
  leaves 0 on 18 held-out streams. But answering with whichever twin won last makes the new label slower than the
  plain store, because the old twin still wins 10-16% of hits after a swap. Retrieval must integrate evidence
  (Gold & Shadlen 2007), not follow the last sample.

- 2026-11-01 06:00 UTC+8: the copy store's slowness is not in the retrieval rule. Answering by a majority of the
  last ten twin hits instead of the last one changes almost nothing (swap first 300 up about a point, new-label
  latency still 15 ticks worse than the plain store). Lifting the class cap changes nothing at all. Copies are made
  as early as moves, so the delay sits after the copy. A copy at exactly the original's location is pattern
  completion, not separation (Yassa & Stark 2011): the twins differ only by label, so every border item is decided
  by bookkeeping rather than content. Next: find which prototype answers wrong after a swap.

- 2026-11-01 07:00 UTC+8: a new memory should be on the moment it forms. The copy store was slow because a fresh
  copy started tied with its original and ties went to the original, so the old label kept answering until the copy
  was confirmed again. Letting the copy win at birth gives move-store latency (17 ticks faster than the plain store)
  with no false remaps on 18 held-out streams, but swap accuracy in the first 300 ticks is still 6 points behind the
  move store, spread over unmoved prototypes, originals winning back and copies answering wrong. In CA1 a single
  plateau both creates and places a new field (Bittner et al. 2017); next, place the copy where its evidence is.


- 2026-11-01 08:00 UTC+8: a new memory belongs where the read will look for it, not where its evidence sits.
  Placing the copy at the mean of the items that contradicted the old label (or halfway there) makes swap accuracy
  worse step by step (first 300 ticks: 46/52 at the original, 41/46 halfway, 32/29 at the evidence), because the
  moved copy loses its region to neighbouring old-label prototypes. A new CA1 field is as broad as the window that
  made it (Bittner et al. 2017), not a point. Copies stay at the original; the bulk of post-swap errors in every
  store is prototypes that never change label, so next comes spreading a class's remap to its other prototypes.

- 2026-11-01 09:00 UTC+8: once a class's label change is decided, spread it to the rest of the class. After one
  committed remap, other prototypes of that class now move on a single contradiction for 600 ticks. On 18 unseen
  streams swap accuracy in the first 300 ticks rises from 59/60 to 64/68 with no false remaps and answers 6 ticks
  sooner. The catch: on the one stream where the plain store already flips back and forth before any swap, spreading
  makes that 4 to 13 times worse. In complementary learning systems terms the fast store decided the fact and now
  generalises it; next comes a sticky base so only a sound decision spreads.

- 2026-11-01 10:00 UTC+8: the simplest wake/sleep loop runs, and sleep does nothing useful yet. Each night the
  machine goes offline briefly and the cortex is trained only on replayed store memories, either uniformly or on the
  day's mistakes. The cortex does consolidate (its own accuracy on returning digits rises 4 to 10 points), but the
  combined system gains at most 1.4 points on returns and loses exactly the arrivals it missed while asleep. The
  reason is that the store never forgets, so it already answers returning digits. In complementary learning systems
  the hippocampus is temporary (Kumaran, Hassabis & McClelland 2016); next, the store forgets digits it has not seen
  for a while, and sleep has to carry them.

- 2026-11-01 10:45 UTC+8: the simplest wake/sleep dual works mechanically but does not pay. By day the store
  writes and the cortex learns; at night the machine goes offline and the cortex learns only from the store's
  replay, either uniform or focused on what the cortex got wrong. Every version loses a little lifetime accuracy,
  and all of that loss is the time spent asleep: on what it does answer, a sleeping machine is exactly as accurate.
  Replay does teach the cortex returning digits (4 to 10 points better alone), but the store already answers them,
  so the system gains only 1 to 3. Making the store forget absent digits, as the hippocampus is supposed to, hurts
  returns by 17 points and sleep wins back only 2. Next: keep the store answering while the cortex sleeps.

- 2026-11-01 11:40 UTC+8: the store-and-cortex pair finally pays, and it does not need to sleep. Letting the store
  keep answering at night removes the whole cost of sleep, and replay then lifts returning-digit accuracy by 1 to 5
  points with no lifetime loss in both worlds. Better still is replay with no night at all: every 38th arrival the
  cortex rehearses a batch of store memories of digits it has not seen lately, the way rats replay a distant maze
  during brief pauses (Karlsson & Frank 2009). That gains 6 points on returns, 1.6 on the swap and a little lifetime,
  and when the store is made forgetful it recovers 12 of the 17 points the forgetting cost, where night replay
  recovered 2. This is McClelland's interleaving argument showing up directly. Fresh-seed confirmation is running
  before anything becomes the default.

- 2026-11-01 11:45 UTC+8: once the store keeps answering while the cortex sleeps, sleep pays. Nothing learns from
  what arrives at night; the cortex only replays the store's memories. Against a machine that never sleeps, lifetime
  accuracy is flat (within 0.1) and returning digits are 1.3 to 4.5 points better in both test worlds, which was
  the bar set before the run. The earlier loss really was just the time spent offline. Not the default yet: three
  runs per world and one margin of 0.01; fresh-seed confirmation is running, alongside a version with no nights at
  all that slips short replays into the waking day.

- 2026-11-01 12:35 UTC+8: the first working store-and-cortex pair is now the default, and it does not sleep. On six
  streams never used for development, both versions held up: letting the store answer at night while the cortex
  replays still gains 2 to 3 points on returning digits, and slipping a short replay of absent digits into the waking
  day (every 38th arrival) gains 4 to 5, with every stream better and lifetime accuracy slightly up. By the rule set
  before the run, the awake version wins. The cortex's own memory of returning digits rises from about 60 to 77
  percent. What is left: the store alone answers returns at 88 to 89, so the part that decides who answers still
  gives away about 5 points. Next test, already running: replay the stored digits that look most like what is
  arriving now, rather than the ones that have been away (McClelland, McNaughton & Lampinen 2020).

- 2026-11-01 12:45 UTC+8: replay what is missing, not what looks like the present. McClelland and colleagues showed
  in 2020 that a network only needs to rehearse old items similar to a new one, because new learning mostly disturbs
  similar knowledge. Here that loses: rehearsing the stored digits nearest each arrival recovers 4 points fewer
  returning digits than rehearsing digits that have been away, and does worse even than picking at random. This
  cortex does not mainly suffer overlap; it forgets whatever it has not been shown lately. Next: when a digit
  returns, let the store, which still remembers it, answer more often.
