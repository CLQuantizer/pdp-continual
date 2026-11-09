# Guideline: a learning system driven by compression progress

Source: J. Schmidhuber, *Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective
Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes*, arXiv
0812.4360v2 (2009). Copies: `docs/papers/schmidhuber2009_driven_by_compression_progress.{pdf,html}`. Status: read.

This is a standing guideline for pdp. Every hypothesis should say which of the four principles below it serves.
An experiment that serves none of them is polishing, not progress (same rule as `docs/goal.md`).

## The four principles (Section 1.2 of the paper), and what each means here

1. **Store everything.** "The data is holy as it is the only basis of all that can be known about the world." Keep
   the raw history of observations and actions.
   *Here:* the store (hippocampus) is the history. Its jobs are to let the compressor make more passes over
   experience than the world grants, and to make progress measurable. It is not a second classifier. Where storage
   must be bounded, keep what the compressor cannot yet predict (Schmidhuber's 1991–92 history compressor) and drop
   what it already predicts.
2. **Improve subjective compressibility.** Spend computation on an adaptive compressor (a predictor) that reduces the
   bits needed to encode the whole history.
   *Here:* the cortex is that compressor. The score is code length in bits (prequential −log₂ p of each observation,
   before learning it), on observations, not labels. Store size counts toward the code (H20's two-part T).
3. **Reward compression progress, not compression and not error.** Intrinsic reward
   r_int(t+1) = C(p(t), h(≤t+1)) − C(p(t+1), h(≤t+1)) (eq. 5): the *old and new* compressor scored on the *same*
   history.
   *Here:* prediction error is the wrong reward (1990 version). It gets trapped by noise, which is always surprising
   and never learnable. Pure novelty and pure predictability are both boring. Only data with still-learnable regularity
   is interesting.
4. **Act to maximise future compression progress.** An RL controller that can observe the compressor's state picks
   actions that steepen the compressor's learning curve: attention, experiments, curiosity.
   *Here:* the machine must have actions that change what it sees: which channel to watch, what to replay, when to
   sleep, later which experiment to run. A world without actions cannot test this principle.

## Practical rules that follow (and that H23 already taught us)

- **Measure progress on the whole history, as eq. 5 says (confirmed by H23 → H23c).** Score old and new compressor
  on a uniform sample of *everything seen so far* (a reservoir in the store), excluding the glance just learnt. The two
  shortcuts both fail, in opposite ways:
  - a sliding window of recent history rewards memorising (H23);
  - the next unseen chunk rewards *tracking* a drifting source, not lasting compression (H23b: locked onto a counter,
    worse than random on every seed).
  With the whole-history sample, the same compressor, bandit and budget beat random by 0.26 bits and a noise-aware
  oracle by 0.11, and a curriculum emerged: motifs, then the long motif, then the grammar (H23c).
- **So the store is what makes curiosity computable.** Old-vs-new on the same history needs the history. This is
  the hippocampus's first job here.
- **Check headroom before setting a bar.** Run an oracle allocation first; a bar larger than the oracle's gain is a
  guaranteed kill (H23).
- **Noise and dark controls in every active world.** A uniform-noise channel (noisy TV) and a constant channel are
  the minimum sanity checks: the curious agent must leave both.
- **The compressor's limits define what is interesting.** A source the current compressor cannot improve on is
  boring *to it*. Progress-driven curiosity should move on; when the substrate improves (H21 state, fast weights), the
  same source can become interesting again.
- **Time counts too (C_lτ, eq. 4).** One bit of compression is worth halving the runtime. Report compute next to
  bits.
- **External reward is just more history.** Labels, when present, are observations the compressor predicts like any
  other symbol. They are not a privileged target.

## Ladder of milestones

1. Passive world, bits as the score, store priced (H20).
2. Active world, compression progress as reward, beats random and error-seeking controllers and an oracle that knows
   what is noise (H23 failed on headroom and estimator; H23b).
3. A substrate that can represent processes, not only categories (H21: gated recurrence / fast weights), so the
   compressor can learn algorithms (counters, CA rules, grammars).
4. The store as the residual of the history compressor: written by surprise, deleted after consolidation, selected for
   replay by expected progress (H22, H24).
5. A learned controller with state (proper RL on intrinsic reward) choosing glances, replays and sleeps (H25).
6. Experiments as actions: the agent generates inputs to test its predictions (PowerPlay-style self-invented tasks).
