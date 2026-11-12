# Brain map

Ez, 11 Nov 2026: "Think about all the parts in the brain and have them each a place in our theory and
experimentation." This page does that. Each major part of the brain gets three things: the job it does in the
machine we are building, what we have measured so far, and the next test. The theory is still Schmidhuber's four
principles: store everything, improve the compressor, reward compression progress, and act to maximise it. The map
shows which principle each part serves, where our machine already has the part, and where it has nothing yet.

<iframe class="fig" src="../figures/brain-map.html" title="Brain map: each brain part, its role in our machine, and its test status"></iframe>

Colours give status. **Tested** means we have scored results. **Queued** means a test is specified or next in line.
**Open** means the part has a role in the theory but no experiment yet. **Absent** means our machine leaves the part out
on purpose. The figure is a wiring diagram of loops, not an anatomical drawing.

## The thalamus, which we had left out

Almost every input to the cortex passes through the thalamus. Smell is the main exception. It is also the hub of
nearly every loop the cortex is part of: cortex to basal ganglia to thalamus to cortex, cortex to cerebellum to
thalamus to cortex, and hippocampus to anterior thalamus to cingulate cortex (the Papez circuit). Our machine has had
no such part. The stream went straight into the cortex. The thalamus does five jobs that matter here.

1. **Gate before the compressor.** The thalamic reticular nucleus (TRN) is a sheet of inhibitory cells around the
   thalamus. It decides which relay channels pass to the cortex, which made Crick (1984) call it the attentional
   "searchlight". In our breadth worlds, the controller that picks which channel to read *is* this searchlight. W1c
   measured its value: with one shared cortex, uniform reading was best. So a searchlight used to allocate learning
   time gains little when every channel is reachable anyway. If the thalamus earns its place, it is through the other
   jobs.
2. **A router between cortical areas.** Sherman & Guillery (2002) separate *driver* inputs, which carry the message,
   from *modulator* inputs, which set gain. Higher-order nuclei such as the pulvinar and the mediodorsal nucleus (MD)
   take driver input from one cortical area and relay it to another. Cortex-to-cortex traffic therefore partly runs
   through the thalamus. It is the nearest thing the brain has to a switchboard, but it routes by content and
   context, not by address (see [compute vs data movement](docs/compute-vs-data-movement.md)). For us that means a
   cortex made of modules, with a learned router between them, instead of one GRU.
3. **Keeping working memory alive.** MD thalamus sustains and sharpens the prefrontal activity that holds a task rule
   (Schmitt et al. 2017). W2 is the first world where the agent has to hold a plan across steps: which room it is in,
   and how far into the lock it is.
4. **Two firing modes, linear and nonlinear.** In tonic mode a thalamic relay cell passes its input on roughly in
   proportion. In burst mode, after a hyperpolarisation, T-type calcium channels make it fire an all-or-none burst
   that flags a change, a "wake-up call" (Sherman 2001). The same cell switches between a linear relay and a novelty
   detector. That links to Ez's frequency-domain question: whether the cell mixes frequencies is a state the brain
   sets.
5. **The sleep gate for replay.** In sleep the thalamus shuts out the senses. The TRN paces sleep spindles
   (11–16 Hz). Spindles lock to the cortical slow oscillation and to hippocampal sharp-wave ripples, and that
   three-way coupling predicts how much is consolidated (Staresina et al. 2015). Driving thalamic spindles in phase
   with slow oscillations improved memory in mice (Latchoumane et al. 2017). So the thalamus sets *when* replay
   reaches the cortex. We have never varied that. Our replay is interleaved, one replay step after every live chunk.

**Thalamus tests.**
- **H31, the sleep gate.** Keep the replay dose fixed at our best traffic of 8. Deliver it either interleaved (as
  now) or in sleep blocks, where input stops and the same replay steps run back to back. Score prequential bits.
  Brains batch replay into sleep. If blocks cost little, the batching is cheap. If they cost a lot, there is a reason
  replay also runs while awake, which brains do. Principles 1 and 2. CPU-sized.
- **Thalamic router (open).** Modular cortex plus learned router, against one cortex of the same size, on the
  multi-channel W1 stream. Interference was small in W1c, so the expected gain is small. This waits until a world
  shows interference.

## Every part, its place

| Part | Job in our machine | Principle | What we have | Next |
|---|---|---|---|---|
| **Neocortex** | The compressor: predicts the next symbol and learns online | 2 | GRU cortex on enwik8 E10M: 1.836 ± 0.004 bits/byte, ties PPMd and beats xz (H29d) | H30: oscillator-bank cortex (diagonal complex/negative eigenvalues plus an order-2 mixing readout) |
| **Thalamus** | Gate, router and sleep gate (above) | 1, 4 | As a searchlight: uniform reading best (W1c). As a sleep gate: batching replay into sleep costs 0.04–0.23 bits/byte against interleaving (H31) | Router open |
| **Hippocampus** | Store everything and replay it | 1 | Uniform replay at traffic 8 helps on real text: −0.138 ± 0.003 bits/byte (H29d) | H26: content-cued replay, below |
| · dentate gyrus | Content hash: a sparse expansion that separates similar episodes, like the fly's LSH (Dasgupta et al. 2017) | 1 | None: we index by *position*, which is RAM | H26 keys replay by content |
| · CA3 | Pattern completion: a partial cue recalls the whole episode (Hopfield-style attractor) | 1 | None | H26 cued (wake) replay |
| · CA1 | Compares what CA3 recalls with what arrives: a novelty signal | 3 | Our per-symbol surprise plays this part | Feeds the controller |
| · sharp-wave ripples | Replay in compressed time | 1 | The replay step itself (H27b, H29c, H29d) | H31 sets when ripples run |
| **Entorhinal cortex** | Grid and time cells; a slowly drifting context that keys memories (Howard & Kahana 2002) | 1 | None | Context vector as the cue in H26 |
| **Basal ganglia** | Action selection: picks the next action from learned values | 4 | The controller in M4 and H25a (breadth worlds: small headroom) | M4 on W2, if W2-0 qualifies |
| **VTA and SNc (dopamine)** | Teaching signal. Classic dopamine reports reward prediction error (Schultz et al. 1997). For us the reward is compression progress | 3 | Progress controllers (H25a, H28) did not beat random reading. Noise leaks into the progress signal | Retest on W2, where progress needs action |
| **Cerebellum** | A fast forward model trained by error, with a huge sparse granule-cell expansion (Marr 1969; Albus 1971; Wolpert et al. 1998) | 2 | None | H32: a fast count model (PPM-like) mixed with the cortex by a learned mixer, as cmix and nncp do |
| **Prefrontal cortex** | Goals and plans held across steps; sets the task | 4 | The GRU state only; no separate part | Needed in W2; a planner arm after M4 on W2 |
| **Amygdala** | Tags events as important, which boosts their storage and replay (McGaugh 2004) | 1 | Tag-driven (selective) replay hurt three times (H10, M4a, H27) | A mismatch: our worlds have had no events worth tagging |
| **Locus coeruleus (noradrenaline)** | Unexpected uncertainty: a surprise raises gain and learning rate; switches exploit and explore (Yu & Dayan 2005; Aston-Jones & Cohen 2005) | 2, 4 | A fixed lr. At 1e-3 the cortex diverges on enwik8; 3e-4 is stable (H29b, H29c) | Surprise-gated learning rate in a world that changes |
| **Basal forebrain (acetylcholine)** | Expected uncertainty; sets encode mode (high ACh, awake) against consolidate mode (low ACh, sleep) (Hasselmo 2006) | 1, 2 | The fixed live/replay alternation | Its switch is H31's interleaved-or-blocks choice |
| **Raphe (serotonin)** | Patience: how far ahead to value outcomes (Doya 2002) | 4 | None. Breadth worlds needed no horizon | W2: the reward is 3·k actions away |
| **Hypothalamus** | Homeostatic drives: hunger, temperature, the extrinsic rewards | — | **Absent on purpose**: the machine has only intrinsic reward | — |
| **Claustrum** | Thin sheet linked to almost all of cortex; proposed to bind areas in sync (Crick & Koch 2005) | — | No role yet | Open |
| **Sleep switch (VLPO, SCN)** | Decides when the senses close and replay runs (Saper et al. 2005) | 1 | H31: the same dose in sleep blocks costs more as blocks grow (8 KB +0.04, 128 KB +0.13). An online predictor needs awake replay | H26: sleep as a second, broad pass |
| **Brainstem, spinal cord, senses** | The interface: tokens in, actions out, reflexes | — | The token stream; fixed policies (random, oracle) are the reflexes | — |
| **Neuromodulators as a class** | One signal broadcast to whole areas, with no addresses | 3 | One scalar reward to the controller | The same broadcast pattern as content addressing |

## What the map says

- **The machine has a cortex, a hippocampus and a basal ganglia, and that is all.** Every scored result so far tests
  one of those three. The thalamus, the cerebellum and the neuromodulators are where the untested ideas are.
- **Three parts are about *when* and *how much* to learn, not *what*.** These are the locus coeruleus (learning
  rate), the basal forebrain (encode or consolidate) and the thalamus (sleep gating). Our machine fixes all three by
  hand. H29b already showed that the learning rate alone decides whether the cortex works on real text.
- **The cerebellum is the likeliest quick compression gain.** The best compressors (cmix, nncp) mix many predictors,
  and a fast count model next to a slow network is exactly a cerebellum next to a cortex. H32 tests whether that
  beats PPMd clearly instead of tying it.
- **Curiosity's parts are the basal ganglia, dopamine, the prefrontal cortex and serotonin.** All four need a world
  where actions matter, and W2 is that world. They are tested there or not at all.

## Order of tests

1. W2-0 (running), then M4 on W2: basal ganglia and dopamine where actions matter.
2. H26: cued and broad replay (dentate gyrus, CA3, entorhinal cortex).
3. H31: sleep gate (thalamus, basal forebrain, sleep switch). Done 12 Nov: interleaved wins; batching costs bits.
4. H30: oscillator-bank cortex (neocortex, frequency domain). CPU first.
5. H32: cerebellar mixer (cerebellum). One E10M pass on the GPU, about 1.2 GPU h, checked against the quota guard.
6. Surprise-gated learning rate (locus coeruleus), once a world changes over time.

References added on this page: Crick 1984; Sherman & Guillery 2002; Sherman 2001; Schmitt et al. 2017; Staresina et
al. 2015; Latchoumane et al. 2017; Schultz, Dayan & Montague 1997; Marr 1969; Albus 1971; Wolpert, Miall & Kawato
1998; McGaugh 2004; Yu & Dayan 2005; Aston-Jones & Cohen 2005; Hasselmo 2006; Doya 2002; Crick & Koch 2005; Saper,
Scammell & Lu 2005; Papez 1937. Already registered: Howard & Kahana 2002; Dasgupta et al. 2017; Hopfield 1982.
