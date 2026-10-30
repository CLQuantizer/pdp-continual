i always have this hunch that the overall architecture is too complicated in modrern deep neural net to compensate for the over simplification of the single cell/neuron. it is like one elixir agent is complex compared to a jvm thread, but over all you get much less over head when scaling

Your hunch has real support, and your analogy fits better than you might expect. One part of it needs adjusting, though.

### Evidence that real neurons are far richer than the units in deep nets
- **A single neuron behaves like a small deep network.** Beniaguev, Segev & London (*Neuron*, 2021) trained neural networks to copy a detailed model of one cortical pyramidal neuron. It took a temporal convolutional network about **5–8 layers deep** to match it. With one ion-channel type (NMDA) switched off, a much shallower network was enough. So a large share of the computation sits *inside the cell*, in its dendrites.
- **Neurons as two-layer networks.** Poirazi, Brannon & Mel (2003) showed that a pyramidal neuron acts like a two-layer network: each dendritic branch is a nonlinear subunit, and the cell body combines them.
- **One neuron can compute XOR.** Gidon et al. (*Science*, 2020) found dendrites in human cortex that compute XOR. That is the classic problem a single artificial neuron *cannot* solve.

So the standard unit (a weighted sum passed through one nonlinearity) is a drastic simplification, and deep nets do make up for it by stacking very many of them.

### Where your analogy fits well
The BEAM-vs-JVM point is really about **shared state and coordination**, and that maps closely onto the last two answers:

| | JVM threads | BEAM processes |
|---|---|---|
| State | shared memory | private, isolated |
| Coordination | locks, global synchronisation | asynchronous messages |
| Scaling cost | contention grows with thread count | close to linear |

| | Backprop network | Local-learning network |
|---|---|---|
| State | a global loss everyone depends on | each unit or layer has its own objective |
| Coordination | update locking, weight transport | send activity forward, learn locally |
| Scaling cost | waiting and stored activations grow with depth | each layer runs independently |

Backprop's update locking is *literally* a lock on shared state. Local learning is the actor model applied to neurons.

### The adjustment
A BEAM process isn't *heavier* than a JVM thread. It's much **lighter** (a few hundred words of memory against roughly 1 MB of stack). What's richer is its **semantics**: a mailbox, isolation, and failure that stays local. That distinction matters for your hunch. The useful version probably isn't "make neurons computationally heavier." It's **"give each unit enough local machinery that it doesn't need global coordination to learn."**

That leads to the most interesting idea in this area: **the neuron's complexity may be mainly what makes local learning work**, not extra computing power. A few models from this line of research do exactly that:
- **Segregated dendrites** (Guerguiev, Lillicrap & Richards, *eLife* 2017) feed feedback into a separate dendritic compartment, so the cell can tell "what I'm computing" apart from "how I should change."
- **Burstprop** (Payeur et al., *Nature Neuroscience* 2021, which is in Table 1 of the SoftHebb paper you just read) uses spike bursts versus single spikes as two channels at once, one for the signal and one for learning.
- **Active dendrites for continual learning** (Iyer et al., Numenta, 2022) let dendritic segments gate which units fire in each context. This targets **catastrophic forgetting on tasks like Split-MNIST**, the exact failure your benchmark showed in every gradient-based method.

In this view, the brain moves backprop's global credit-assignment machinery *into each cell*. The parts get more complex so that the whole needs less coordination, which is your Elixir point.

### The case against
- **Expressive power isn't the bottleneck.** A complex neuron can be replaced by a small sub-network, as Beniaguev's result shows. So richer units don't let you compute anything new. Any advantage has to come from *learning* or *efficiency*.
- **Hardware favours simple units.** GPUs run dense matrix multiplies, and the "bitter lesson" is that simple methods that scale have repeatedly beaten clever ones. Rich neurons are expensive on digital hardware. They make most sense on neuromorphic or analog chips, where the physics does the work for free.
- **The brain's wiring isn't simple either:** six-layered cortex, loops through the thalamus, neuromodulators. It's complex at both levels.

### A test that fits pdp
Your benchmark has already isolated the problem: every unit shares the same weights across all tasks, so new learning overwrites old. A unit with **several dendritic branches**, each gated by context and learning locally, is a small, testable step toward "richer neuron, simpler network." It plugs straight into the Split-MNIST harness, so you'd see directly whether a smarter unit fixes what a smarter learning rule did not.