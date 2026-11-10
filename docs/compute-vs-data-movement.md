# Compute versus moving data

The two scarce resources that set kernel time on a GPU are arithmetic
throughput and the movement of bytes through the memory hierarchy,
especially between HBM and on-chip SRAM. The same cost hierarchy shows
up as CPU-bound versus I/O-bound work in classical software, and as
compute versus storage in data systems. Moving state usually costs more
than computing on it locally, at every hop: SRAM, DRAM, disk, then the
network.

People mash three different cuts into one slogan. They are related, not
identical.

## GPU kernels: the roofline

The roofline model (Williams, Waterman, Patterson 2009) makes the GPU
cut quantitative. Compare a kernel's arithmetic intensity (FLOPs per
byte moved) to the machine's ridge point (peak FLOPs divided by peak
bandwidth). Below the ridge the kernel is memory-bound; above it, it is
compute-bound.

This split has become sharper, not weaker, because peak FLOPs have grown
much faster than DRAM bandwidth — the memory wall. That is why
FlashAttention exists (do not write the N×N attention matrix out to HBM)
and why the selective SSM scan in Mamba is IO-aware. A naive
implementation materializes an expanded state of shape `(B, L, D, N)` in
HBM, which is a bandwidth problem, not a FLOP problem. The fused kernel
loads the parameters into SRAM, discretizes, runs the parallel scan
there, and writes back only outputs of shape `(B, L, D)`. The arithmetic
barely changes; the HBM traffic drops by a factor of `N`.

It is a pair of bounds, not a personality type. A kernel sits somewhere
on that roof, and near the ridge both constraints bite. "Moving data" is
also not one thing: registers, shared memory, L2, and HBM have very
different bandwidths and capacities, and the expensive hop is usually
HBM. A third constraint is latency — not enough independent work to hide
memory or instruction delay — which is what a sequential recurrent scan
hits, or decode at batch 1. Bank conflicts, warp divergence, and atomics
can dominate a poorly written kernel, but those mean you are not on the
roofline yet; they do not replace the two-resource picture.

In SSMs this split shows up as convolution versus scan. A global
convolution can raise intensity and ride well-tuned FFT or GEMM
(compute-side). A naive recurrence is latency-bound. The SRAM parallel
scan turns that into a bandwidth-sensitive but parallel kernel whose HBM
traffic no longer grows with state size.

## Classical software: CPU versus I/O

On a GPU the roofline cut is bandwidth: FLOPs versus bytes over the HBM
bus. Classical "CPU-bound versus I/O-bound" is often latency: the
process is stalled on a disk or network round-trip, sometimes without
even saturating the pipe. "Compute versus storage" is a third cut: where
the durable state lives, whether it fits, and who is allowed to update
it.

MySQL, PlanetScale, D1, and Durable Objects are four answers to where
that bus sits.

**MySQL** puts the query engine and InnoDB in one process. The buffer
pool is the SRAM analogue — a cache in front of disk — so a hot query is
CPU-bound and a cold scan is I/O-bound, inside one machine.

**PlanetScale** (Vitess) keeps that pairing per shard and scales by
cutting the data so each query can run next to the rows it needs. You
move a small query to the large state. A cross-shard join is the
equivalent of spilling out of SRAM: the data has to move again, and that
is where it hurts.

**D1** is SQLite as a service for Workers. Compute and storage are still
together inside SQLite, but the Worker is one hop away — closer than an
app server talking to RDS across a VPC, still a bus.

**Durable Objects** fuse them per actor. The object *is* the state; code
runs against it with a single thread and no concurrent writers. The
network hop is "address this object," not "scan a table." That refuses
to distribute one piece of state, the way an SSM keeps a fixed-size
hidden state in fast memory and updates it in place instead of rereading
a growing log (attention's KV cache, or `SELECT` over history). You
scale by having many objects, not by slicing one object's storage.

The useful refinement is whether you are short on bandwidth, stalled on
latency, or paying to coordinate a working set that no longer fits next
to its compute. Those four systems are mostly fighting the third
problem, with Durable Objects taking the most radical position: never
separate the state from the code that owns it.

## The PDP dual

The store-versus-cortex split is the same move. The store is addressable
storage, the cortex is compute that has absorbed history, and replay is
the bus. The H15 knee at every-38 was the point where moving more data
into the compute unit stopped paying.

## SRAM and HBM are two throttles

SRAM and HBM both throttle what a GPU can do, and they are not the same
constraint. Mixing them up hides the pivot.

SRAM is the working-set cap: how much state a kernel can keep next to
the ALUs. That is what sizes a FlashAttention tile, and why the
selective scan refuses to materialize a state of shape `(B, L, D, N)`.
If it does not fit in SRAM, you spill to HBM and you have already lost.

HBM is the big store and the bus to it — weights, KV cache, anything
written out. Capacity and bandwidth both bind. They also bind
production: you cannot ship those FLOPs without stacking and packaging
that memory, and SRAM itself eats die area and scales worse than logic.
Peak math has been growing faster than both, which is why buying more
tensor cores keeps landing on a memory wall.

The pivot is to stop paying that hop. Compress history into a fixed
state that stays in fast memory (an SSM, a HiPPO coefficient vector)
instead of a growing log you reread every step — attention's KV cache is
the HBM problem that gets worse with context. Colocate memory with
compute so you are not fetching weights into an ALU: that is a synapse,
a Durable Object, and cortical weights. Then meter the remaining bus.
In this project the store is the HBM analogue, replay is the traffic,
and H15 already found the knee at every-38.

Selectivity is a ticket onto SRAM, not a slogan: a piece of the past
earns a place in the fast state only if it buys compression. The
interesting local compute — including whether state eigenvalues may go
negative — only matters after you have refused to spill.

## Does the same split exist in the brain?

Yes, but not as a von Neumann split of CPU versus RAM. Synapses are both
the memory and the computational parameters, so the brain does not
stream weights from a distant RAM into an ALU the way a GPU does. That
is the problem it evolved to avoid, by colocating storage with compute —
more like a Durable Object or an SSM hidden state than like
MySQL-over-the-network.

What remains expensive is moving signals: sending a spike down an axon,
releasing transmitter, and pumping ions back after the membrane voltage
changes. Attwell and Laughlin (2001) built the first quantitative energy
budget for grey-matter signaling. In rodent cortex they put about 47% of
signaling energy on action potentials and 34% on postsynaptic glutamate
effects, and argued this forces distributed codes with at most about 15%
of neurons active at once. After it turned out that mammalian spikes are
cheaper than the squid-axon figure they had used, Howarth, Gleeson, and
Attwell (2012) revised the cortical split: about 50% of signaling energy
on postsynaptic glutamate receptors, 21% on action potentials, 20% on
resting potentials. Receiving the message, not sending it, became the
largest slice in grey matter.

Levy and Calvert (2021) then cut the pie the way an engineer would:
computation as a neuron integrating synaptic charge to decide when to
fire, versus communication as action potentials, transmitter release,
and white-matter costs including myelin. On that partition, human
cortical computation is about 0.1 W of ATP and long-distance
communication about 3.5 W — a 35-to-1 ratio, or 2% versus 71% of their
4.94 ATP-watt cortical budget. They also note that the oft-quoted 20 W
is whole-brain glucose, not this cortical ATP slice. The 35-to-1 figure
is real, but it depends on calling dendritic integration "compute" and
spikes plus white matter "communication." It does not mean synapses are
cheap: Howarth's budget says the opposite about postsynaptic currents.

Lennie (2003) used the same kind of budget to estimate that human
cortical neurons can sustain only about 0.16 spikes per second on
average, and that fewer than 1% can be substantially active at once.
Even pushing every assumption toward cheaper spikes, he stayed under
about 1 spike per second per neuron. That is a bandwidth cap: you cannot
keep the whole cortex "on the bus."

The spatial version of the same fact is wiring economy, already flagged
by Cajal. Bullmore and Sporns (2012) review the trade-off: short local
connections are cheap and numerous; long-range links buy integration and
cost volume, delay, and ATP. Sterling and Laughlin's *Principles of
Neural Design* (2015) is the book-length version. The brain's answers
look like the systems tricks above: colocate memory with compute at the
synapse; keep related work nearby (cortical maps, like cache locality);
invest in the interconnect (myelin); send sparsely; send only what the
receiver cannot already predict (Rao and Ballard, 1999); hold working
memory in silent synaptic traces instead of persistent spike traffic
(Mongillo, Barak, and Tsodyks, 2008). Complementary Learning Systems
(McClelland, McNaughton, and O'Reilly, 1995) is the architectural
version: a fast hippocampal store, a slow cortex, and replay as the bus
— including sharp-wave ripples as a compressed, offline burst of that
traffic, more like batched DMA than a continuous HBM stream.

People have studied this under the names energy budget, wiring cost, and
systems consolidation, not "roofline." The GPU dichotomy is what you get
when you separate an ALU from a memory chip. The brain refuses that
separation, then rediscovers the cost as spikes and synapses.

## Is there a bus in the brain?

There is no von Neumann bus: no single shared wire that every detailed
message must take turns on. That is the architecture PDP was a reaction
against. What exists are three different things people lump together as
a "bus."

White-matter tracts and the corpus callosum are cables: huge bundles of
dedicated parallel axons. That is the opposite of a bus. Many
conversations happen at once, each on its own wire.

The thalamus is a hub, not a bus. Almost all senses except smell relay
there, but as many parallel nuclei (vision through the LGN, hearing
through the MGN, and so on), plus higher-order nuclei that sit in
cortico-thalamo-cortical loops. Traffic passes through a place; it does
not serialize onto one medium.

The closest thing to a bus is broadcast control, not a data path. A few
thousand locus coeruleus neurons send norepinephrine across most of
cortex; the raphe, ventral tegmental area, and basal forebrain do the
same for serotonin, dopamine, and acetylcholine (Aston-Jones and Cohen,
2005, on LC as an adaptive-gain signal). Those channels shout a shared
state — arousal, surprise, reward — not an image or a sentence.
Oscillations can play a related role as a schedule: Fries's
communication-through-coherence is a clock for who may speak, not a pipe
that carries the content. Hippocampal replay is a fourth case: a
temporary protocol that bursts compressed traffic from a fast store into
cortex (McClelland, McNaughton, and O'Reilly, 1995). Anatomically it
uses existing pathways; temporally it behaves like a scheduled bus.

### The spinal cord is not a bus

The spinal cord is a poor example of a bus, even though it gets called
an information highway. It is a trunk of parallel tracts — dorsal
columns and spinothalamic fibers going up, corticospinal and other
fibers going down — plus a computer at every segment. Withdrawal
reflexes, reciprocal inhibition, and central pattern generators for
locomotion (Grillner's work, and Sherrington long before) run in the
cord itself. A spinal animal is not a disconnected disk; it can still
produce patterned movement. If the cord were only a bus to a brain-CPU,
cutting it would leave inert cable below the cut, and every reflex would
have to go to cortex and back. Neither is true. Fine voluntary control
in primates does lean more on descending tracts, especially
corticospinal, but that is one important pathway among many, not "the
bus."

### Is a bus against PDP?

A bus is against PDP only if it is a serialized data bus that all
content must traverse, and if knowledge lives in the thing you fetch
rather than in the connections. That is the von Neumann machine: one
ALU, one memory, one hop between them. Rumelhart and McClelland's point
was that cortex does not work that way — many units compute at once, and
the knowledge is in the distributed weights.

Structured pathways do not violate that. Neither does a broadcast gain
signal. Neither does complementary replay. McClelland himself added the
hippocampal store and the replay path without abandoning PDP: the cortex
remains the parallel distributed learner; the store and its bus exist so
that cortex can change slowly without being overwritten. The constraint
already adopted here is the right one: a selective bus is not a
substitute for distributed weight change.

The violation would be if replay *became* the computer — retrieve from
the store, never change cortex, and you have built RAM plus a CPU. The
H10–H14 kills on selection, and the H15 knee at every-38, are that
principle showing up as data: more bus did not replace cortical
learning, and past a point more traffic stopped paying. Having a bus is
not anti-PDP. Having only a bus is.

## References

- Aston-Jones, G., & Cohen, J. D. (2005). An integrative theory of
  locus coeruleus-norepinephrine function: Adaptive gain and optimal
  performance. *Annual Review of Neuroscience*.
- Attwell, D., & Laughlin, S. B. (2001). An energy budget for signaling
  in the grey matter of the brain. *Journal of Cerebral Blood Flow &
  Metabolism*.
- Bullmore, E., & Sporns, O. (2012). The economy of brain network
  organization. *Nature Reviews Neuroscience*.
- Fries, P. (2005). A mechanism for cognitive dynamics: Neuronal
  communication through neuronal coherence. *Trends in Cognitive
  Sciences*.
- Howarth, C., Gleeson, P., & Attwell, D. (2012). Updated energy
  budgets for neural computation in the neocortex and cerebellum.
  *Journal of Cerebral Blood Flow & Metabolism*.
- Lennie, P. (2003). The cost of cortical computation. *Current
  Biology*.
- Levy, W. B., & Calvert, V. G. (2021). Communication consumes 35 times
  more energy than computation in the human cortex, but both costs are
  needed to predict synapse number. *PNAS*.
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why
  there are complementary learning systems in the hippocampus and
  neocortex. *Psychological Review*.
- Mongillo, G., Barak, O., & Tsodyks, M. (2008). Synaptic theory of
  working memory. *Science*.
- Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the
  visual cortex. *Nature Neuroscience*.
- Sterling, P., & Laughlin, S. (2015). *Principles of Neural Design*.
  MIT Press.
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An
  insightful visual performance model for multicore architectures.
  *Communications of the ACM*.

## Orchestrator review (10 Nov 2026, UTC+8): what holds, what our data contradicts

**Holds, and adopted:**
- **The three cuts are different:** bandwidth, latency, and where state lives. Keeping them apart is right.
- **"Having a bus is not anti-PDP; having only a bus is."** M4b/M4c confirm it. The store's recall alone adds ≤ 0.02 bits;
  replay that *changes the cortex* adds 0.26–0.30 bits.
- **"Meter the remaining bus."** Adopted: results now report bus traffic (replayed symbols per observed symbol) next
  to prediction bits and memory (roadmap, Status 10 Nov evening). Until now traffic and the replayed log were free.
- **Brain as colocated memory + compute, with replay as a scheduled burst.** This matches CLS and gives the right test for
  the machine: the store exists so the cortex can change slowly without being overwritten.
- **The aside on negative state eigenvalues.** It points at our parked counter/CA problem. Linear RNNs with eigenvalues
  restricted to [0, 1] cannot track parity or mod counts; allowing negative ones can (Grazzi et al., *Unlocking
  state-tracking in linear RNNs through negative eigenvalues*, ICLR 2025). A concrete candidate for the counters.

**Contradicted or too strong:**
- **"Selectivity is a ticket onto SRAM: a piece of the past earns a place only if it buys compression."** It is true for
  the *cache* that sits next to compute for prediction (H24g: 15% of memory, P within 0.024 bits). It is false for the
  *training bus*. Replay drawn from a store selected by compression value made the machine worse than no store (M4a,
  +0.08 bits); replay from the unselected history was 0.29 bits better (M4b). The rule: **select hard for the cache,
  sample fairly for the bus.** Our own H10 (1 Nov: the absence selector does not beat uniform replay) had already shown
  this, and M4a repeated the mistake.
- **"Compress history into a fixed state instead of rereading a growing log."** That is right for *inference*, and the
  cortex is that fixed state. For *learning*, the best machine so far rereads the raw log (M4c), so the log has not been
  made unnecessary. Whether a fixed-size *fair sample* of it suffices is H27, running.
- **"The store is the HBM analogue."** Only if "store" means the raw history. The selective store we built is closer
  to a prediction cache; in the active world it contributes nothing (M4c: −0.001 bits).
- **Attribution to check:** every-38 replay was set by the H8 rate sweep (loop, 1 Nov). H15 was the temporary-hippocampus
  test (killed), though later loop entries call the replay knee "the H15 knee".
- **Energy is not bits.** The brain numbers (Attwell & Laughlin; Howarth et al.; Levy & Calvert; Lennie) are energy
  budgets. Our score is code length. They become comparable only through the traffic axis and an explicit exchange rate;
  until we fix one, they motivate the metric but do not enter it.
