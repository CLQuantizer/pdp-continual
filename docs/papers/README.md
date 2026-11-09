# Papers

**Status** says how well we know each one:
- **read**: we have the PDF and read it.
- **checked**: authors, venue and claim confirmed by a web search (Oct 2026).
- **memory**: cited from memory in discussion. Verify before quoting.

## In this folder

| File | Paper | Status |
|---|---|---|
| `journe2022_hebbian_deep_learning_without_feedback.pdf` | Journé, Garcia Rodriguez, Guo, Moraitis (Huawei). *Hebbian Deep Learning Without Feedback.* ICLR 2023 (arXiv 2209.11883). | read |
| `schmidhuber2009_driven_by_compression_progress.pdf` / `.html` | Schmidhuber. *Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes.* arXiv 0812.4360v2 (2009). **Standing guideline**, see `docs/curiosity.md`. | read |

Key points from Schmidhuber 2009 (the guideline):
- Four principles: store the entire raw history; improve an adaptive compressor of it; give intrinsic reward equal to
  compression progress, C(p_old, h) − C(p_new, h) on the same history (eq. 5); let an RL controller maximise it.
- Prediction-error curiosity (1990) gets stuck on noise; reward the predictor's *improvement* instead (1991).
  White noise and an unchanging input are both boring: neither allows further compression progress.
- Performance measures: l(p) (eq. 3), and l(p) + log τ (eq. 4, one bit is worth halving the runtime).

Key points from Journé et al.:
- Multilayer SoftHebb trains deep nets with **no feedback, target or error
  signal**. It reports 99.4% MNIST, 80.3% CIFAR-10, 76.2% STL-10 and
  27.3% ImageNet (up to 5 hidden layers plus a linear classifier).
- Lists five problems with backprop: weight transport, non-local
  plasticity, **update locking** (time-locking of layer updates), iterative
  equilibria, and global (self-)supervised losses.
- **Update locking** (Section 1): a synapse's error exists only after the
  signal has gone fully forward and back, so layer updates are
  "time-locked to these delays". Examples can only arrive as fast as one
  full round trip.
- Table 1 compares bio-plausible methods on weight-transport-free /
  local / update-unlocked / unsupervised. SoftHebb is the only one with
  all four.

## Methods in the benchmark

| Paper | Used for | Status |
|---|---|---|
| Hinton 2022. *The Forward-Forward Algorithm: Some Preliminary Investigations.* (sole author) | `ff`, `ff_orig` | memory |
| Lee & Song 2023. *SymBa: Symmetric Backpropagation-Free Contrastive Learning with Forward-Forward Algorithm for Optimizing Convergence.* arXiv 2303.08418 (Kakao Enterprise) | the loss that fixed `ff` | checked |
| Lillicrap, Cownden, Tweed, Akerman 2016. *Random synaptic feedback weights support error backpropagation for deep learning.* Nature Communications | `fa` | memory |
| Nøkland 2016. *Direct Feedback Alignment Provides Learning in Deep Neural Networks.* NeurIPS. arXiv 1609.01596 | `dfa` | checked |
| Moraitis et al. 2021/22. *SoftHebb: Bayesian Inference in Unsupervised Hebbian Soft Winner-Take-All Networks.* Neuromorphic Computing and Engineering. arXiv 2107.05747 | `hebb` update rule | checked |
| Krotov & Hopfield 2019. *Unsupervised learning by competing hidden units.* PNAS | Hebbian background (~98.5% MNIST) | memory |
| Huang et al. *Extreme learning machine* (random features + least squares) | `rf` | memory |
| DeSieno 1988. *Adding a conscience to competitive learning.* | homeostatic thresholds in `hebb` | memory |
| Jaderberg et al. 2017. *Decoupled Neural Interfaces using Synthetic Gradients.* | coined forward / update / backward locking | memory |

## Follow-ups surfaced in searches (Oct 2026)

| Paper | Why it matters here | Status |
|---|---|---|
| Terres-Escudero et al. 2025. *A Contrastive Symmetric Forward-Forward Algorithm (SFFA) for Continual Learning Tasks.* PMLR v274 | FF on continual learning, the same question as our Split-MNIST | checked (not read) |
| DeeperForward, ICLR 2025 | new goodness + layer norm to fix **dead neurons** in deep FF, the same failure we hit | checked (not read) |
| Wu & Ren, ICLR 2026 | FF-style local learning for reinforcement learning | checked (not read) |
| *Training CNNs with the Forward-Forward Algorithm.* Scientific Reports 2025 | FF in convolutional networks | checked (not read) |
| *Energy-Efficient Deep Learning Without Backpropagation.* arXiv 2511.01061 | benchmark of forward-only methods, up to 41% less energy | checked (not read) |
| Shafiei Kafraj, Krotov, Latham. *A Biologically Plausible Dense Associative Memory with Exponential Capacity.* ICLR 2026, arXiv 2601.00984 | Krotov's current direction | checked (not read) |
| *Long Sequence Hopfield Memory*, Krotov, Hoover et al., ICML 2025 | same | checked (not read) |
| *Accelerated Predictive Coding Networks via Direct Kolen-Pollack Feedback Alignment.* arXiv 2602.15571 | carries DFA into predictive coding | checked (not read) |
| Lillicrap, Santoro, Marris, Akerman, Hinton 2020. *Backpropagation and the brain.* Nature Reviews Neuroscience | review: the brain may approximate backprop with activity differences | checked |

## Neuron complexity and dendrites

| Paper | Claim | Status |
|---|---|---|
| Beniaguev, Segev, London 2021. *Single cortical neurons as deep artificial neural networks.* Neuron | emulating one detailed pyramidal neuron needs a 5–8 layer TCN; with NMDA removed, a much shallower net suffices | memory |
| Poirazi, Brannon, Mel 2003. Neuron | a pyramidal neuron acts like a two-layer network (branches are nonlinear subunits) | memory |
| Gidon et al. 2020. Science | dendrites in human cortex compute XOR | memory |
| Guerguiev, Lillicrap, Richards 2017. *Towards deep learning with segregated dendrites.* eLife | feedback lands on a separate apical compartment | memory |
| Payeur et al. 2021. *Burstprop.* Nature Neuroscience | bursts vs single spikes multiplex signal and learning signal | memory (also in Journé Table 1) |
| Iyer et al. 2022 (Numenta). *Avoiding catastrophe: active dendrites enable multi-task learning in dynamic environments.* | context-gated dendritic segments reduce forgetting | memory |

## Mixture of experts

| Paper | Claim | Status |
|---|---|---|
| Fedus et al. *Switch Transformer* | top-1 routing plus a load-balancing loss against expert collapse | memory |
| DeepSeek-V3 technical report (2024) | auxiliary-loss-free balancing: per-expert bias nudged up/down by load, outside the gradient | memory |
| DeepSeekMoE | many fine-grained experts plus shared experts | memory |
| He 2024 (DeepMind). *Mixture of A Million Experts* (PEER) | single-neuron experts selected by product-key retrieval | memory |

## Scaling caveat

| Paper | Claim | Status |
|---|---|---|
| Bartunov et al. 2018. *Assessing the scalability of biologically-motivated deep learning algorithms and architectures.* NeurIPS | methods close to backprop on MNIST fall well behind on CIFAR-10 and ImageNet | memory |

## Curiosity, fast weights, plasticity (scan 9 Nov 2026; see `docs/literature_2026-11.md`)

| Paper | Claim | Status |
|---|---|---|
| Hou, An, Du 2025. *Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring.* arXiv 2509.25438 | reward = change in a learned error model's prediction; robust to noisy TV | checked (abstract) |
| Blad, Längkvist, Loutfi 2026. *Gradient-Momentum Coupling: A Parameter-Space Proxy for Learning Progress.* arXiv 2605.05856 | \|grad · momentum\| as O(d) learning progress; easy-first, noise-robust | checked (summary) |
| Irie & Gershman 2025/26. *Fast weight programming and linear transformers: from machine learning to neurobiology.* arXiv 2508.08435 | modern linear RNNs are FWPs; Gated DeltaNet strongest; slow weights still need BPTT | checked (summary) |
| Bellec et al. 2020 (Nat. Commun.), e-prop; *Generalising E-prop to Deep Networks*, arXiv 2512.24506 | online eligibility-trace approximation to RTRL, BPTT-level cost | checked (abstract) |
| Behrouz, Zhong, Mirrokni 2025. *Titans: Learning to Memorize at Test Time.* arXiv 2501.00663 | memory updated by surprise (gradient), momentum + forgetting gate | checked (summary) |
| Dohare et al. 2024. *Loss of plasticity in deep continual learning.* Nature | gradient learners lose plasticity; continual backprop (re-init low-utility units) keeps it | checked |
| Elmoznino et al. 2025. *In-context learning and Occam's razor.* ICML (arXiv 2410.14086) | minimising prequential code length = training error + complexity | checked (abstract) |
| Zhang et al. 2026. *Fast Weight Attention for Continual Learning.* arXiv 2608.27763 | fast/slow weight split for continual learning; slow weights by backprop | checked (summary, low confidence) |

