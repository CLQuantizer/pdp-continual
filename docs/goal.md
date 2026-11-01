# Goal

> Come up with a new neural-network architecture, coupled with training
> methods, that transcends the NVIDIA-hardware-bound, CUDA-ninja paradigm.

## North star: a machine that lives, not a model that is trained

The training/inference split is the paradigm's deepest assumption, and
the one to break. Today a model is trained offline on a dataset, frozen,
then deployed for inference. Its data and its learning algorithm never
coexist with its use.

We want a **genuine machine that alternates between acting and learning
the way animals alternate between waking and sleeping**:

- **Always on.** Every input is first something to respond to
  (inference) and then, if feedback comes, something to learn from.
  There is no separate training phase and no frozen deployment.
- **Data and algorithm live together.** The machine carries its own
  memory of experience (a hippocampus-like fast store) and its own
  learning rules. Nothing is retrained from an external dataset.
- **Sleep is organic and useful.** The machine decides when to sleep,
  from its own internal state (sleep pressure), not on an experimenter's
  clock. Sleep **costs** something: while asleep it can't respond. It
  has to earn that cost by doing what waking can't: consolidating,
  rebalancing, forgetting what is no longer relevant.
- **Judged over its whole life,** on the stream it actually lives in, not
  on a held-out test set after the stream ends.

Everything below (criteria, metrics) serves this. An experiment that
improves a post-hoc test score without moving the machine toward this
loop is polishing, not progress.

## What the current paradigm assumes

Today's deep learning works because architecture, training and hardware
were co-designed around one another:

| Assumption | Why GPUs like it |
|---|---|
| Dense matrix multiplies | thousands of identical multiply-adds in lockstep |
| Big batches | amortise memory traffic; keep cores busy |
| Global backprop | one loss, exact gradients, and every activation stored for the backward pass |
| Synchronous layers | the whole network advances step by step (update locking) |
| Train once, then freeze | offline training on shuffled data; continual learning is an afterthought |
| Precise digital arithmetic | gradients assume exact, repeatable maths |

"Transcending" it means an architecture **and** training method that
don't need these assumptions to work well, so they could run on
substrates where these assumptions are expensive or impossible:
neuromorphic chips, analog/in-memory compute, many small asynchronous
cores, physical systems.

## Design criteria

A candidate should aim for all of these:

1. **Local learning.** Each unit or layer updates from information it
   already has. No global error, no weight transport, **no update
   locking**.
   *"Give each unit enough local machinery that it doesn't need global
   coordination to learn."*
2. **Sparse, conditional activity.** Most units idle for any given input;
   cost scales with *activity*, not size (gating, winner-take-all,
   routing).
3. **Memory where compute is.** Weights change where they live. No
   activations stored for a later backward pass.
4. **Online and continual, with controlled forgetting.** Learns from a
   stream, one example at a time. Forgets by **relevance** (age, disuse,
   noise), never by **interference** (new learning overwriting unrelated
   old knowledge). Never forgetting is a failure too: it overfits and
   can't adapt when the world changes (loss of plasticity).
5. **Asynchronous and scalable by addition.** Grows by adding units that
   talk through messages, like BEAM processes, not by synchronising more
   of them.
6. **Robust to imprecision.** Tolerates noise, low precision and device
   variation, as analog hardware requires.
7. **Self-stabilising.** Homeostasis built in, so no unit, expert or
   layer collapses or dies. This came up three times already (see
   `notes.md`).
8. **Wake/sleep modes.** A switchable sleep mode with different rules:
   input off, no real-time deadline, and permission to disrupt. Sleep
   hosts what wake can't: replay and consolidation from a **bounded** fast
   store, global synaptic downscaling (controlled forgetting), FF's
   negative phase (dreams as negatives), and maintenance (revive dead
   units, rebalance experts). Sleep may afford limited coordination
   because it has no deadline. **Sleep is self-initiated and has a
   price:** pressure builds from the machine's own state (like adenosine,
   Borbély's Process S), the world's quiet periods make it cheap (Process
   C, circadian), and time spent asleep is time not responding.
9. **No train/inference split.** The same machine answers and learns on
   every step. Performance is what it scores while living, not what a
   frozen copy scores afterwards.

## How we measure it

Accuracy alone rewards whatever runs well on GPUs. Each candidate is also
scored on hardware-neutral costs, counted in the simulation:

| Metric | Measures |
|---|---|
| **Lifetime prequential accuracy** (predict each input before learning from it, over a long recurring, drifting stream; scored on the inputs that arrive while it is awake; since 2026-11-01 14:05 UTC+8 inputs that arrive while asleep are not evaluated, and the time spent asleep is reported separately as downtime) | the north star: how well it does while living (9) |
| Forgetting curve: accuracy on a class vs time since it was last seen | forgetting by relevance, not interference (4) |
| Anytime class-IL (held-out test every 700 wake examples) | retention between sleeps; interim stand-in for the above |
| Accuracy (standard, online, Split-MNIST class-IL, after the last sleep) | legacy; a post-sleep best case, not the north star |
| Adaptation after a rule change (e.g. two digits' labels swapped mid-stream) | can it forget when it should |
| Fast-store size, sleep cost (ops per sleep) and sleep frequency | sleep isn't cheating by storing everything |
| Active-unit fraction / synaptic ops per example | sparsity (criterion 2) |
| Bytes stored per example for learning | memory locality (3) |
| Update latency (steps from input to weight change) and concurrency | locking (1, 5) |
| Accuracy under weight noise / low precision | robustness (6) |
| Dead or dominant unit fraction | stability (7) |

Then scale up: MNIST → Fashion-MNIST → CIFAR-10. Bartunov et al. 2018
showed most bio-plausible methods hold up on MNIST and collapse beyond it,
so MNIST alone proves little.

## Where we stand

| Criterion | Best so far | Gap |
|---|---|---|
| Local learning | FF (SymBa + homeostasis): 98.40%, level with backprop | still forgets |
| Continual | random features + ridge: 95.5% Split-MNIST | its readout is a dense O(H³) linear solve, which is very GPU-paradigm |
| Continual, local | **FF + sleep, 10 prototypes/class: 90.0%** (50/class: 91.2%) | replay 0.8× wake examples is enough (sleep every 2000); store needs labels |
| Continual, no image store | FF + top-down generative dreams (ridge rule): **86.8%** (inverted-net dreams: 27%) | generator needs accumulate-only sums (644k floats) and an O(H³) solve in sleep; delta-rule generators forget (32%) |
| Continual, no image store, local | FF + top-down dreams, contrastive wake-sleep generator rule with dropout dreams, 8-bit accumulator, prior = 40 real codes/class at 8 bits: **89.0% at 149k floats** (ridge on the same prior: 90.3%) | no solve, but a per-synapse accumulator as big as G (393k); total memory not below a prototype image store. Folding it into G (one number per synapse) drops to 65.4%; an **8-bit** C with stochastic rounding is free (87.2%, 99k floats), with 10 codes/class the total is **149k, under the prototype store (157k), at 86.6%** vs 90.0%: storing codes at 8 bits fits 40/class in 149k; with dropout on the dreamed negative codes (keep 0.8, no weight decay) the local rule reaches **89.0%**, and **89.1% with 3000 sleep steps over 10 seeds**, vs **89.9%** for 9 prototypes/class (141k): a small but significant gap (−0.74, paired t −2.37); ridge on the same prior 90.3%. **But** prototypes need no replay noise on MNIST (89.3%), and without it their variance isn't stored: **70.6k floats**, under half the local decoder's memory at the same accuracy |
| Scale-up | Split Fashion-MNIST: prototypes 75.6%, ridge dreams 75.8%, local dropout dreams **74.6%** with 3000 sleep steps (72.8% at 1000; one-pass ceiling 83.4%) | sleep transfers unchanged; denser images need longer sleeps to keep the local gap at ~1 point; a two-level generator (prior on layer 2) loses 2.6 points even with ridge. A nonlinear decoder (code + 2000 fixed sparse random features, local readout) reaches **75.6%, level with prototypes**, but at 543k floats vs 157k; ridge on the features 76.5%. Per class: the coat gap is a trade with shirt (local decoder best on coat + shirt); the real loss is sandal (blur fills the straps). Prototypes **without pixel noise: 77.3%**; with the wake learning rate ×0.1 (slow cortex, fast store): **78.5%, the new best**. Ridge dreams + a fixed centre-surround sharpening filter: 76.6% (local decoder hurt by it: 73.3%). Noise-free prototypes need only the means: 5/class = 39k floats → 75.2%, above the local decoder at 149k. Sandal check: the decoders' weakness is incoherent fine detail (ridge ghosting, local speckle), not style averaging |
| Adaptation (forget when it should) | Label swap 2↔3 after Split-MNIST: FF + sleep adapts awake in 500 examples (96%), but each sleep replays the old mapping; after 8000 examples: prototypes 85%, local decoder 83%, ridge 78% (no sleep 99% but others 1%) | stores turn over by counts, not surprise; conflicting wake data drops the other 8 digits to 30–43% between sleeps. Surprise-gated prototype updates: **91.2%** after the swap (helps adaptation; neutral on MNIST, −1.7 on Fashion: they confuse ambiguity with change). Extra sleep on surprise lifts retention during conflict (55 → 64–72%), but a matched faster clock does as well or better: surprise is the wrong trigger. Sleep when an awake probe of 30 stored prototypes drops 50%: at matched replay, worst-case retention 42 vs 25 (anytime worst 47 vs 29), final unchanged; weight-drift pressure only detects task starts |
| Anytime retention (accuracy between sleeps, not just after one) | FF + 9 prototypes with **wake lr ×0.3**: 74.7% avg (10 seeds, +7.2 on every seed), final 89.5%; with a 50% probe-drop sleep trigger: 76.1%, worst 46.6% (final 89.1%). Awake replay (0.8 items per update): 91.2% avg, worst 76%, but final 79% (recency bias) | **All final scores are measured right after a sleep.** At clock 2000 anytime class-IL is 68% (54% in tasks 3–5, worst 25%): wake learning erases old classes between sleeps |
| Stability | homeostatic bias (Hebbian, FF) | |
| Sparsity, asynchrony, robustness | not measured yet | |

No single method meets all the criteria yet.

## Candidate directions (from `notes.md`)

- Two-compartment FF neurons: label/context on an "apical" branch,
  label-gated branches at test time.
- Locally learned mixture of experts: FF/Hebbian experts, homeostatic
  router.
- Wake/sleep: FF positives while awake, replay and dreamed negatives in
  sleep, with a bounded fast store and synaptic downscaling.
- Accumulate-only memory (sufficient statistics) inside each unit,
  replacing the global ridge solve with something local.
- Pipelined, one-thread-per-layer execution, to measure unlocking for
  real.

## Non-goals

- Beating backprop on GPU benchmarks *on GPUs*.
- Biological fidelity for its own sake. Biology is a source of
  mechanisms, not a requirement.

## Risk to keep in mind

The "hardware lottery" (Hooker 2020): ideas win partly because they suit
the hardware of their day. Simulating a non-GPU method on a CPU will make
it look slow. That's why the metrics above count operations, memory and
latency rather than wall-clock time.
