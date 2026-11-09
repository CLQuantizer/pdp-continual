# Literature scan, November 2026: what it changes for the roadmap

Orchestrator's scan, 9 Nov 2026, UTC+8. Status of every paper: **checked** (authors, title and venue confirmed via
search; abstract or summary read, not the full paper) unless marked otherwise. Each finding names the milestone it
touches (`docs/roadmap.md`).

## 1. Learning-progress estimators: ours is one of three families (M2)

| Family | Representative | How progress is measured | Cost |
|---|---|---|---|
| Old vs new model on the same data | Schmidhuber 2009 eq. 5; **our H23c** (whole-history reservoir) | bits of a history sample before vs after the update | two extra evaluations per step |
| Learned model of the error | **LPM**, Hou, An, Du 2025, *Beyond Noisy-TVs: Noise-Robust Exploration via Learning Progress Monitoring* (arXiv 2509.25438) | an error model predicts the dynamics model's error; reward = change in that predicted error between iterations | a second network |
| Parameter space | **GMC**, Blad, Längkvist, Loutfi 2026, *Gradient-Momentum Coupling: A Parameter-Space Proxy for Learning Progress* (arXiv 2605.05856) | normalised \|gradient · Adam momentum\|: learnable structure pushes consistently and builds momentum; noise cancels | O(d), same order as the backward pass |

- GMC reports what H23c showed: easier groups first, slow ones deferred, near-uniform across noise levels.
- LPM is Schmidhuber 1991's "predictor of the predictor's changes", modernised.
- **Our H23b result (next-chunk progress rewards tracking a drifting source) is not in these papers.** Worth a short
  write-up later: it is a third failure mode next to noise and memorisation.
- **For M2:** if H25a's reservoir scoring is too noisy or slow with a neural learner, GMC is the cheap drop-in. Queue
  it as an estimator ablation (H25b): reservoir vs GMC vs LPM-style, same world.

## 2. Substrate: "fast weights" is now the family name for modern linear RNNs (M1)

- Irie & Gershman 2025/26, *Fast weight programming and linear transformers: from machine learning to neurobiology*
  (arXiv 2508.08435): Mamba2, GLA, RetNet and DeltaNet are all fast-weight programmers that differ in the
  update/decay rule. **Gated DeltaNet** (decay + delta rule) is reported strongest on language modelling. The fast
  weights learn locally and online; **the slow weights are still trained by BPTT**. Nobody has removed that.
- Schlag, Irie, Schmidhuber 2021 (ICML), *Linear Transformers Are Secretly Fast Weight Programmers*: the origin of the
  delta-rule FWP. Already our H21a `lin_fw`.
- **For M1:** H21a tests a plain delta-rule FWP. If it underperforms the GRU, the next substrate variant should be
  gated delta (a decay gate α_t on the fast weights), not a different family.

## 3. Credit assignment without BPTT exists, but is approximate (Stage B)

- **E-prop** (Bellec et al. 2020, Nature Communications; "Generalising E-prop to Deep Networks", arXiv 2512.24506,
  Dec 2025): an RTRL approximation with a per-synapse eligibility trace. Online, forward-only, BPTT-level cost; ignores
  indirect cross-time interactions. The 2025 paper extends it to arbitrary depth / DAGs.
- **For Stage B:** e-prop on the H21 winner is the first local candidate: a measurable gap in bits to its BPTT twin.

## 4. A synthesis this suggests: local wake, global sleep

Sections 2 and 3, together with our own wake/sleep criterion (`docs/goal.md`: "sleep may afford limited coordination
because it has no deadline"), point to one architecture:

- **Awake:** only fast weights change, by a local delta rule, online, with no backward pass: in-context learning as
  compression within the life. The store writes what was surprising.
- **Asleep:** the slow weights (which program the fast-weight updates) are trained from the store's replay, where
  global credit assignment (BPTT, or e-prop when we want it local) is allowed because nothing has to be answered.

This gives sleep a job that waking cannot do, which is the goal's own test for sleep, and it is the first version of
the local-learning criterion that does not cap what the machine can learn. **Proposed as H26** after M1 picks the
substrate: compare (a) everything trained online by BPTT (Stage A), (b) fast weights online + slow weights only in
sleep from replay, (c) fast weights only, slow weights frozen after a short childhood. Score: lifetime bits including
the downtime charge of sleep.

## 5. Memory written by surprise, forgotten by decay (M3)

- **Titans**, Behrouz, Zhong, Mirrokni 2025 (arXiv 2501.00663, Google): a neural long-term memory updated at test time
  by gradient steps whose size is the input's *surprise* (gradient of an associative loss), with momentum ("past
  surprise") and a forgetting gate. A predicted token causes no update.
- This is our "store as the residual of the history compressor" (write what the cortex failed to predict, delete what it
  predicts), as a parametric memory. **For M3:** H24c's deletion-after-consolidation is the explicit version; a
  surprise-gated write threshold is the next store knob, measured in bits saved per stored bit.

## 6. A lifelong learner loses plasticity (all milestones; long lives)

- Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton 2024, *Loss of plasticity in deep continual learning*
  (Nature): over long continual training, gradient methods lose the ability to learn (up to 90% dead units).
  **Continual backpropagation**, which re-initialises a small fraction of low-utility units, keeps plasticity
  indefinitely; gradient descent alone does not.
- Our H19 (shrink-and-perturb of the *whole* network) failed; selective re-initialisation of low-utility units is
  different and untested.
- **For all milestones:** every long-life run should report the dead-unit fraction and a plasticity probe (learning
  speed on a fresh process late in life). Fix only if the probe drops.

## 7. Support for the metric

- Elmoznino et al. 2025 (ICML), *In-context learning and Occam's razor* (arXiv 2410.14086): minimising prequential
  code length jointly minimises training error and model complexity; in-context learners act as prequential
  compressors. Supports scoring the machine by prequential bits (H20 onward) rather than accuracy.

## Changes made to the plan

- Roadmap: H25b (estimator ablation), H26 (local wake / global sleep), plasticity probe as a standing report line.
- H21 follow-up: gated delta rule as the next fast-weight variant, if needed.
