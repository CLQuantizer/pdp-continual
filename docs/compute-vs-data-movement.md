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

## Ez, 11 Nov 2026: moving is preserving; is the dichotomy false?

> Mercury can store information because waves travel so slowly in it. Perhaps move vs compute is a false dichotomy:
> moving itself can be preservation, and compute is another form of moving. Perhaps we can project moving and
> computing into a frequency domain, so the dichotomy resolves into consistency. And memory addressing: if the brain
> has no RAM, how does it work? It is content-addressable for sure. The difference from LLMs is that the brain
> fundamentally does not need address-based memory, the kind GPUs, and so LLM infrastructure, are built on.

### Orchestrator review (11 Nov 2026, UTC+8)

**Delay lines: storage is data in flight. Holds.**
- Mercury delay-line memories (EDSAC, UNIVAC) kept bits as acoustic pulses. Sound in mercury travels at about
  1.45 km/s, so a 1.5 m tube held about a millisecond of pulses. Each pulse was read at the far end, regenerated and fed
  back in.
- Capacity is bandwidth × delay, the same "bits in the pipe" as a network link. Moving can be preserving.

**Frequency domain: exact for the linear part.**
- A delay τ multiplies every frequency by e^(−iωτ): gain 1, phase only.
- Any linear time-invariant operation is also one multiplier per frequency, H(ω). So storage is an all-pass
  (|H| = 1), forgetting is |H| < 1, and linear computation is shaping. They are one kind of object, and for this part
  the dichotomy does dissolve.
- Diagonal linear recurrences (state-space layers: HiPPO, S4, LRU, the Legendre Memory Unit) work this way. Each
  unit is a damped rotation: the angle is a frequency and the modulus is a retention time. A bank of them is a set of
  delay lines at many frequencies. HiPPO and the LMU are derived as optimal projections of the history onto a basis,
  which is "project into a frequency domain" made literal.
- A negative real eigenvalue is the Nyquist frequency, a sign flip every step: counting mod 2. That ties this idea to
  the parked counter/CA problem (Grazzi et al. 2025, above).

**Where the line survives, in other forms.**
- **Nonlinearity.** This first said "the frequency picture unifies storage only with *linear* computation". That
  was wrong. Ez pushed back, and the correction is the next section. Nonlinearity is not outside the frequency domain. It is what moves energy
  *between* frequencies.
- **Access.** A delay line is sequential: you wait for your bit to come round. Random access was invented to avoid
  that wait, and it pays in wires, which is the movement cost this note started from. The surviving cut is
  sequential versus addressed access.
- **Energy.** By Landauer, moving and storing can in principle be reversible. Only erasing has a floor of kT ln 2 per
  bit. The mercury tube's energy went into regenerating against loss, not into moving. Physics draws its line between
  preserving and destroying information.

**Replay is the regeneration stage.** The cortex loses a little with each update (|H| < 1 per step), and replay
re-injects stored history before it fades. Our measured dose (traffic about 8, each segment ≲ 10 times; H27b, H29c)
is the regeneration rate that balances that loss. Testable: the optimal dose should track the cortex's forgetting
rate.

**Addresses: the brain is content-addressed in both algorithm and substrate.**
- **LLMs are content-addressed in the algorithm.** Attention is a soft lookup by similarity: query · key, then a
  weighted read, which is a modern Hopfield update (Ramsauer et al. 2020). The substrate is address-based, though. The
  KV cache sits at addresses in HBM, and each new token streams every key to the compute units to compare. Content
  lookup is emulated by a brute-force scan of addressed memory: O(context) bytes moved per token. That is where the
  memory wall bites.
- **The brain moves the query, not the memories.** A cue is broadcast along axons. Every synapse computes its match
  where it sits, in parallel, and recall is the network settling to the completed pattern (attractor dynamics,
  Hopfield 1982; CA3 autoassociation). Storage and comparison happen in the same place, as in a hardware CAM.
  Nothing stored has to travel to be compared.
- **The brain does have "addresses", but they are computed from content.** Hippocampal indexing theory (Teyler &
  DiScenna 1986) has the hippocampus store a sparse index that reinstates a distributed cortical pattern. The dentate
  gyrus makes that index by pattern separation (Yassa & Stark 2011): a sparse expansion that acts like
  locality-sensitive hashing, as in the fly's olfactory circuit (Dasgupta et al. 2017; Babadi & Sompolinsky 2014).
  It is a hash table with no table: the hash is a set of neurons. Time and place also act as content keys, through a
  slowly drifting context vector (Howard & Kahana 2002). Order comes back through sequences replayed in compressed
  time, a delay-line readout.
- **The price.** Recall is approximate. Stored patterns interfere, since they share synapses (Hopfield capacity is
  about 0.14 N patterns for N units), and that interference is why there are two systems and replay (McClelland et
  al. 1995). Arbitrary pointers and exact variable binding are hard for brains and easy for RAM. Each side buys
  something.

**What this means for our machine: our replay is RAM.**
- We replay by choosing a uniform random *position* in the log. A brain cannot do that, because it has no positions
  to sample. Its replay is either cued (the current state retrieves similar past content) or spontaneous (attractor
  dynamics wander the store; sleep replay reaches remote experiences, Karlsson & Frank 2009).
- Cued replay is selective by construction, and we have found three times that selective replay hurts (H10, M4a,
  H27). The prediction: content-cued replay alone underperforms uniform replay, and cued (wake) plus spontaneous,
  broad (sleep) replay recovers it.
- That is exactly the parked H26, "local wake, global sleep". H26 is the address-free version of our best replay
  result, so it moves up the queue, after W2-0.
- A second test: do the delay-line/oscillator layers above (complex and negative eigenvalues) replace the GRU as the
  cortex (H30)?

References added for this section: Gu et al. 2020 (HiPPO); Gu, Goel & Ré 2022 (S4); Voelker, Kajić & Eliasmith 2019
(LMU); Orvieto et al. 2023 (LRU); Ramsauer et al. 2021 (*Hopfield networks is all you need*); Hopfield 1982; Teyler &
DiScenna 1986; Howard & Kahana 2002; Dasgupta, Stevens & Navlakha 2017; Landauer 1961; Shin et al. 2017 (deep
generative replay). Already registered: Yassa & Stark 2011, Babadi & Sompolinsky 2014, Karlsson & Frank 2009,
McClelland et al. 1995, Fries 2005, Mongillo et al. 2008.

## Ez, 11 Nov 2026: can the frequency domain not handle nonlinearity?

> Can the frequency domain not handle non linearity? I don't think so.

### Orchestrator review (11 Nov 2026, UTC+8): Ez is right; the cut is diagonal versus off-diagonal

**It can, and has for a century.**
- **Products become convolutions.** Multiplying two signals in time convolves their spectra. Squaring a tone at ω
  gives 0 and 2ω. Multiplying tones at ω₁ and ω₂ gives ω₁ ± ω₂. Every radio mixer depends on this: a superheterodyne
  receiver is a deliberate nonlinearity that moves a station to a fixed frequency. AND, comparison and thresholds are
  steep nonlinearities, and their output is a predictable spray of harmonics and intermodulation products.
- **Volterra series** are the frequency-domain language for nonlinear systems with memory. The output is a sum of
  terms of order n, each with a kernel h_n(τ₁…τ_n). In frequency that kernel becomes H_n(ω₁…ω_n), which sends inputs
  at ω₁…ω_n to output at ω₁+…+ω_n. Boyd & Chua (1985) showed that any time-invariant operator with *fading memory*
  can be approximated arbitrarily well by a finite Volterra series. A learner that forgets (|H| < 1, as above) is one
  of those operators.
- **Koopman operator.** A nonlinear dynamical system acts *linearly* on the space of functions of its state (Koopman
  1931; Mezić 2005). In that lifted space it has eigenvalues, so it has frequencies and decay rates exactly like the
  oscillator bank above. Dynamic mode decomposition estimates them from data.
- **Brains show it directly.** Cross-frequency coupling, where the phase of theta (4–8 Hz) modulates the amplitude of
  gamma (30–100 Hz) in hippocampus and cortex, is a nonlinear interaction measured as a frequency-domain object
  (Canolty & Knight 2010). The proposed theta-gamma code (Lisman & Jensen 2013) packs a short ordered list of items
  into gamma cycles nested inside one theta cycle. That is storage, ordering and computation on one frequency
  ladder.

**So the dichotomy does resolve, into one statement.** In the frequency basis every operation is a matrix acting on
the spectrum:
- **Storage and moving are diagonal.** Each frequency keeps its own channel. All-pass is perfect storage, |H| < 1
  is forgetting, and linear filtering reshapes channels without mixing them.
- **Computation is off-diagonal.** Nonlinearity couples channels, sending energy from ω₁, ω₂ to ω₁ ± ω₂, 2ω and so on.
- Move and compute are the same kind of object, an operator on spectra. They differ in *structure*: diagonal versus
  coupling. That is Ez's "consistency", made precise.

**What it costs, which is where the old cut went.**
- A diagonal operator on N frequencies needs N numbers. A full order-n Volterra kernel needs on the order of Nⁿ, and
  a Koopman lift can be infinite-dimensional; chaotic systems have a continuous spectrum. The price of nonlinearity is
  the dimension of the lift, not being expressible.
- Architectures already pay it this way. A modern state-space layer (LRU, S4) is a diagonal linear recurrence, which
  stores and moves, plus a nonlinear, position-wise mixing layer between recurrences, which computes. Orvieto et al.
  (2023) argue that this stack is still a universal sequence model. The design question becomes how much
  off-diagonal coupling to buy, and where.
- For us, that refines H30. An oscillator-bank cortex (a diagonal bank of complex and negative eigenvalues) plus a
  small multiplicative readout, a learned order-2 Volterra term, is the frequency-native compressor. The test is still
  bits on E10M against the GRU.

References added for this section: Boyd & Chua 1985 (*Fading memory and the problem of approximating nonlinear
operators with Volterra series*); Koopman 1931; Mezić 2005; Canolty & Knight 2010; Lisman & Jensen 2013. Orvieto et
al. 2023 is already listed above.


## Status after H26, H31, H32 and H32b (12 Nov 2026, UTC+8): what the data say

Ez asked whether the move-versus-compute question has been resolved. In theory, yes: moving and storing are diagonal
operators on the spectrum, computing is the coupling between frequencies, and the old cut survives as a cost (the
dimension of the lift), as sequential versus addressed access, and as erasure (the only energy floor). In data, there
are four partial answers and one open test.

- **Content addressing has two jobs, and the data split them.** As the sampler that feeds consolidation, content
  lost to position: cued replay was 0.063 bits/byte worse than uniform, and cued plus uniform was 0.015 worse (H26).
  As a store read for prediction, content won: the count models are exact lookup tables keyed by the last k bytes,
  which is content-addressed memory with no positions. Beside the cortex they gain 0.068 bits/byte beyond calibration
  on E10M, and 0.119 with orders up to 6 (H32b). The rule from 10 Nov holds with a sharper reason: **select by
  content for recall, sample broadly for the bus.** What we have not built is an address-free broad sampler.
  Random-cue replay is that test, and it stays parked.
- **Regeneration has to keep pace with loss.** If replay is the regeneration stage of a delay line, it must arrive
  while the signal is still there. H31 measured it: the same replay dose delivered in sleep blocks costs +0.04 (8 KB
  blocks) to +0.23 (2 MB blocks) bits/byte, because the cortex drifts between sleeps. That fits the delay-line
  picture. The sharper prediction is still untested: the best replay dose should track the cortex's forgetting rate.
- **Memory where compute is: the first scored case.** The count models update in place, from one sighting, with no
  backprop and no stored activations. Nothing travels to a separate compute unit. They now carry about 0.1 bits/byte
  that a backprop cortex on a GPU does not get.
- **Movement is metered but not priced.** Results report traffic (replayed symbols per observed symbol), but there is
  still no exchange rate between bits and moved or computed bytes. The substitution test (a smaller cortex plus
  counts) is the first place a bits-against-compute front can be drawn.
- **Open: the frequency-native compressor.** H30 is the direct test of the diagonal-plus-coupling view: an
  oscillator bank of complex and negative eigenvalues (storage and moving) plus a learned order-2 readout (coupling),
  scored in bits against the GRU. It has not run. It moves up the queue, behind W2-1, and runs on a W world with a
  counter (the parked counter and CA problem). There a negative eigenvalue should do what the GRU could not.

## Ez, 12 Nov 2026: persistence is just movement through time

> And note that persistence is just movement thru time.

### Orchestrator review (12 Nov 2026, UTC+8)

**It holds, and it closes the loop on the mercury tube.** A stored bit is a bit carried from t to t + τ. A wire carries
it from x to x + d. Both are translations, one in time and one in space. In the frequency domain each is a pure phase:
e^(−iωτ) for time, e^(−ikd) for space. Both are diagonal. So "move" and "store" are not two things. They are the same
operation along different axes: carry the signal unchanged across a displacement. Compute is whatever does not carry
it unchanged, the coupling between channels.

**What follows for our machine.**
- **Forgetting is loss in transit, and replay is a repeater.** The log carries the past through time losslessly (it
  is digital). The cortex's weights carry it lossily: each update attenuates old content a little. Replay re-injects
  the signal before it falls below the noise, as a repeater does on a long line. H31 now reads as repeater spacing.
  Interleaved replay is a repeater every 128 bytes. Sleep blocks of 8 KB, 128 KB and 2 MB space the repeaters
  further apart, and the cost grows with the spacing (+0.04, +0.13, +0.23 bits/byte), as a lossy line predicts.
- **The prediction becomes concrete.** The best repeater spacing (replay dose) should scale with the loss per unit
  time (the cortex's forgetting rate). A faster learning rate means more loss per step, so it needs denser replay.
  This is testable by crossing lr with sleep block on E3M.
- **Lossless carriage is not free of cost either.** The count models carry every context forever, with no decay. In
  a stationary text that is ideal. In a world that drifts, carrying stale counts through time is the loss of
  plasticity that `docs/goal.md` names as a failure. Prediction: in a drifting stream, count models with a decay
  (forgetting by age or disuse) beat exact ones.
- **Brains persist by regenerating.** Synaptic proteins turn over in days, yet memories last decades. Crick (1984)
  pointed out that memory must therefore be held by active renewal, not by stable molecules. The mercury tube's
  regeneration stage again: persistence through time is movement that is paid for by repeated rewriting.

So the dichotomy reduces to one line: carry (in space or time, diagonal) versus combine (coupling, off-diagonal).
Everything we call memory, communication and storage is carriage. Its costs are loss in transit and the repeaters that
fight it.

Reference added: Crick 1984 (*Memory and molecular turnover*, Nature 312:101).

## Ez, 12 Nov 2026: moving through time and space should be unified

> Moving thru time space should be unified.

### Orchestrator review (12 Nov 2026, UTC+8)

**They are one operation, and a medium's velocity is the exchange rate.** A wave in a medium is e^(i(kx − ωt)).
Space and time enter together, and the dispersion relation ω(k) ties them. Carrying a signal a distance d at speed v
*is* holding it for a time τ = d / v. The mercury tube is the proof. It stored bits *because* it moved them slowly
through space, which was Ez's first point on 11 Nov. So memory and wire are not two kinds of carriage. They are one
spacetime translation, and the medium's speed and loss decide which one a device looks like. A slow medium works as a
memory and a fast one as a wire. Information in it can only spread inside a cone set by the speed of propagation. In
lattices this is exact: the Lieb–Robinson bound (1972) gives a maximum speed even for quantum spin chains.

**The cut that survives is now precise: local in spacetime versus not.** A medium carries a signal only to its
neighbours, one step at a time, inside the cone. Random access does not: an address reaches anywhere in one step. Of
the three cuts that survived on 11 Nov, two now fold into this one. Sequential versus addressed access is local versus
nonlocal. The movement cost is what it takes to fake nonlocality with wires: HBM, buses, a KV cache scanned for every
token. The third cut, erasure, stays separate.

**The brain is local in spacetime, and it trades space for time openly.**
- **Time into space.** In sound localisation, axons of graded length act as delay lines. A coincidence detector fires
  at the place where the two ears' delays cancel, so a time difference becomes a place (Jeffress 1948; confirmed in
  the barn owl).
- **Space into time.** A rat's place cells fire at earlier phases of theta as it crosses a field (phase precession,
  O'Keefe & Recce 1993). Each theta cycle replays a stretch of path as a short time sequence. Replay in ripples runs
  whole spatial paths through time at 10–20 times the speed.
- **Delays as memory.** With realistic axonal delays, spiking networks store patterns as precise time-locked firing
  chains (polychronous groups, Izhikevich 2006). The storage lives in the delays: wire as memory.
- **Physical media compute.** Waves on water in a tank, read out linearly, separated speech classes. A medium can
  carry, hold and mix (Fernando & Sojakka 2003, a physical reservoir).

**What it means for our machine.** We split the two cleanly: space is GPU memory traffic, time is the log and the
weights. A unified cortex would be a medium: units on a lattice, each passing its state to its neighbours with a
delay, plus a local nonlinearity at each node. Carriage through space and time is then the same linear propagation
(diagonal in (k, ω)). Compute is the node coupling. Everything is local, which is design criteria 1, 3 and 5 at once.

**Test (H30w, a variant of H30).** At the same parameter count, compare:
- (a) the diagonal oscillator bank (time only);
- (b) the same units on a ring with nearest-neighbour linear coupling, a discretised transport or wave equation,
  which gives exact delays by propagation;
- (c) the GRU.
Score in bits on a W world with counters and on a copy-with-delay stream. Prediction: (b) holds exact long delays that
(a) can only approximate with decaying modes, and so wins where the task needs an exact past symbol (the parked CA
"one row back" problem). Kill rule: if (b) does not beat (a) by more than seed std on the delay stream, propagation
adds nothing over a frequency bank.

References added: Lieb & Robinson 1972; Jeffress 1948; O'Keefe & Recce 1993; Izhikevich 2006; Fernando & Sojakka 2003.
