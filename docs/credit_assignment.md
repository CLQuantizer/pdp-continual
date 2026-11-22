# How deep networks get credit: a map of gradient and gradient-free learning rules

*By Ez · 18 Nov 2026*

A deep network learns only if every layer, and every moment in a sequence, gets some signal about how it contributed to the outcome. Backpropagation sends the exact gradient backwards. Every other rule here is a way to get some or all of that signal more cheaply, more locally, or more like a brain. The rules are grouped by **how credit reaches a deep layer**.

## 1. Exact gradients (the baselines)

| Method | Origin | How credit moves | Notes |
|---|---|---|---|
| Backpropagation (reverse-mode AD) | Linnainmaa 1970; Werbos 1974; Rumelhart, Hinton, Williams 1986 | errors flow back through transposed weights | Needs the backward weights to mirror the forward ones, and stored activations |
| Backpropagation through time (BPTT) | Werbos 1990 | unroll the sequence, then backpropagate | Stores the whole history |
| Real-time recurrent learning (RTRL) | Williams & Zipser 1989 | carries sensitivities forward in time | Exact and online, but O(n⁴) |
| Forward-mode AD, forward gradient | Baydin et al. 2022 | a random direction times its directional derivative, in one forward pass | Unbiased, but noise grows with size |
| Implicit and adjoint gradients | Chen et al. 2018 (Neural ODE); Bai et al. 2019 (DEQ) | solve an adjoint equation | Still global |

## 2. Online credit through time (approximate RTRL)

| Method | Origin | Idea |
|---|---|---|
| UORO | Tallec & Ollivier 2018 | rank-one unbiased RTRL |
| KF-RTRL | Mujika et al. 2018 | Kronecker-factored RTRL |
| SnAp | Menick et al. 2021 | sparse n-step approximation |
| e-prop | Bellec et al. 2020 | per-synapse eligibility trace times a broadcast learning signal |
| OSTL | Bohnstingl et al. 2023 | online spatio-temporal learning |
| Diagonal linear-recurrence RTRL | Zucchet et al. 2023 | exact and cheap when each unit only feeds itself in time |
| Three-factor rules | review: Gerstner et al. 2018 | eligibility trace times a neuromodulator such as dopamine |

## 3. Feedback without weight transport

| Method | Origin | Idea |
|---|---|---|
| Feedback alignment | Lillicrap et al. 2016 | fixed random backward weights; the forward weights come to align with them |
| Direct feedback alignment | Nøkland 2016 | the output error goes straight to every layer through random matrices |
| Direct random target projection | Frenkel et al. 2021 | project the target, not the error |
| Sign symmetry | Liao, Leibo, Poggio 2016 | backward weights share only the sign of the forward ones |
| Kolen-Pollack, weight mirrors | Kolen & Pollack 1994; Akrout et al. 2019 | learn backward weights that match the forward ones |

Bartunov et al. (2018) found that feedback alignment and target propagation fall well short of backpropagation on ImageNet-scale problems. Direct feedback alignment has been pushed further, including to transformers (Launay et al. 2020), with a remaining gap.

## 4. Targets instead of errors

| Method | Origin |
|---|---|
| Target propagation, difference target propagation | Bengio 2014; Lee et al. 2015 |
| Improved difference target propagation | Meulemans et al. 2020; Ernoult et al. 2022 |
| Deep feedback control, least-control principle | Meulemans et al. 2021, 2022 |
| Auxiliary coordinates, ADMM training | Carreira-Perpiñán & Wang 2014; Taylor et al. 2016 |

## 5. Settle, then compare (energy-based and equilibrium methods)

| Method | Origin | Idea |
|---|---|---|
| Boltzmann machines | Ackley, Hinton, Sejnowski 1985 | compare a free phase with a clamped phase |
| Contrastive Hebbian learning | Movellan 1991 | the same for deterministic networks |
| GeneRec | O'Reilly 1996 | a biological version of contrastive Hebbian learning |
| Wake-sleep | Hinton et al. 1995 | recognition learns by day, generation by night |
| Equilibrium propagation | Scellier & Bengio 2017 | nudge the output; the change in the settled state gives the gradient, with a local rule |
| Predictive coding | Rao & Ballard 1999; Whittington & Bogacz 2017 | prediction errors at every level; approximates backpropagation at equilibrium |
| Dendritic microcircuits, burstprop | Sacramento et al. 2018; Payeur et al. 2021 | errors carried in apical dendrites, multiplexed as bursts |
| Latent equilibrium | Haider et al. 2021 | fast settling for networks that run in real time |

## 6. Layer-local objectives (nothing crosses a layer)

| Method | Origin | Idea |
|---|---|---|
| Greedy layer-wise pretraining | Hinton et al. 2006; Bengio et al. 2006 | train one layer, freeze it, stack the next |
| Deeply supervised nets | Lee et al. 2015 | an auxiliary prediction at every layer |
| Local error signals | Nøkland & Eidnes 2019 | per-layer prediction and similarity losses |
| Greedy and decoupled greedy learning | Belilovsky et al. 2019, 2020 | local heads per block, scaled to ImageNet |
| InfoPro | Wang et al. 2021 | local losses that keep information |
| Synthetic gradients | Jaderberg et al. 2017 | each layer predicts its own future gradient |
| Greedy InfoMax, CLAPP | Löwe et al. 2019; Illing et al. 2021 | local contrastive prediction over time |
| HSIC bottleneck | Ma et al. 2020 | per-layer dependence objectives |
| Forward-Forward | Hinton 2022 | each layer separates real data from negative data |
| PEPITA | Dellaferrera & Kreiman 2022 | two forward passes; the error modulates the input |
| Hebbian family | Oja 1982; Bienenstock, Cooper, Munro 1982; Moraitis et al. 2022 (SoftHebb) | unsupervised local rules |
| Slowness, trace rule | Földiák 1991; Wiskott & Sejnowski 2002 | learn what stays stable while the input changes |

## 7. Perturb and measure (zeroth order)

| Method | Origin |
|---|---|
| Weight and node perturbation, REINFORCE | Jabri & Flower 1992; Widrow & Lehr 1990; Williams 1992 |
| SPSA | Spall 1992 |
| Evolution strategies | Salimans et al. 2017 |
| MeZO, fine-tuning language models with forward passes only | Malladi et al. 2023 |
| Attention-gated reinforcement learning (AGREL, AuGMEnT) | Roelfsema & van Ooyen 2005; Rombouts et al. 2012 |

## Where this research stands on the map

- **Tested:** feedback alignment, direct feedback alignment, Forward-Forward and Hebbian rules on continual Split-MNIST. All four forgot the earlier tasks as backpropagation does: a different update rule does not stop the same weights being overwritten.
- **Tested since:** diagonal trace RTRL, which learns a recurrent bank exactly and online. Then layer-local stacks of those banks on text. The only stacks that improve with depth are those in which **every layer is a complete predictor on its own loss** (deep supervision). Going from one to six layers helps, with diminishing returns, and beats one wide layer of the same size. Whether a learned gate or fixed equal weights combine the layers barely matters. Stacks that add the layers' outputs together get worse with depth, whatever error each layer learns from, and growing the stack one layer at a time is worse than training all layers from the start. A second trick removes the cost deep stacks paid at the start: begin the gate that weighs the layers with all its weight on the first layer, so the stack starts as one layer and the upper layers enter as they become useful. This is the local twin of zero-initialised residual branches (zero-γ, SkipInit, ReZero). It saves 0.36 bits/byte at four layers and 0.75 at six over the whole stream, at no cost later.
- **Tested and failed:** cross-layer credit added to each layer's own loss, by direct feedback alignment (+0.28 bits/byte) and by distillation from the stack's combined prediction (+0.05).
- **Tested since:** with zero-initialised gates, depth saturates by eight layers (6 → 8 layers gains 0.009 bits/byte). A third trick does pay: a layer that sees the mean of all earlier layers' features, as in DenseNet, gains 0.062 at six layers.
- **Next:** dense inputs at eight layers, and whether each of the three tricks works without the others.
