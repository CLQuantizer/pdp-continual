# W2-0: the locked-rooms depth world, built and headroom measured (grok, Kaggle CPU)

Read `/root/pdp/CLAUDE.md` first, then `results/w1b_headroom/RESULTS.md` (including its Verdict), `h29b_enwik8.py`
(the model and training loop style), `kaggle/w1b/` (the kernel layout).

## Context
Every world so far was a *breadth* world: the agent chooses which existing stream to watch. There, allocation headroom
is second-order (0.04–0.08 bits/byte), because concave learning curves have a flat optimum. Curiosity's large wins
should come in *depth* worlds. In a depth world the agent's actions change the world's state and decide what data can
exist, and learnable structure is reachable only through chains of the right actions. A random policy reaches depth d
with exponentially small probability.

W2 is such a world. This handoff builds it and measures its headroom with **fixed** policies: no curiosity yet. The
rule is "check oracle headroom before setting a bar". Principle served: act to maximise compression progress (4), by
testing whether the world rewards acting.

## The world (implement exactly; deterministic given `world_seed`)
- **Rooms 0..8** (D = 8; room 8 is the deepest). The agent starts in room 0 with lock state s = 0.
- **Actions** a ∈ {0..7}: 0..5 are manipulations, 6 is the TV, 7 is reset.
- **Rooms k < 8 each have a combination lock.** The key is key_k ∈ {0..5}^3, and s ∈ {0, 1, 2} is the lock state.
  There is also an output table table_k ∈ {0..15}^{3×6}. On a manipulation a:
  - the output is y = table_k[s][a], using s before the update;
  - if a == key_k[s], then s ← s + 1; otherwise s ← (1 if a == key_k[0] else 0);
  - if s reaches 3, the agent moves to room k + 1 and s ← 0.
- **Room 8 has a 6-state machine** with state q, starting at q = 0 on entry. Its tables are out ∈ {0..15}^{6×6} and
  next ∈ {0..5}^{6×6}. On a manipulation a: y = out[q][a], then q ← next[q][a]. There is no exit except reset.
- **TV (a = 6):** y is uniform in {0..15} from a world-noise rng (`numpy.random.default_rng(10_000 + world_seed)`).
  Room, s and q are unchanged.
- **Reset (a = 7):** the agent goes to room 0 with s = 0 and q = 0, and y = 0.
- **Hidden tables:** keys, table_k, out and next are drawn from `numpy.random.default_rng(world_seed)`, in this order:
  key_0..key_7 (integers in [0, 6)), table_0..table_7 (integers in [0, 16)), out, next.
- **Observation after each action:** o = (room after the action, y).

## Token stream and learner
- **Tokens:** each step appends the action token a (ids 0..7), then the observation token 8 + 16·room + y (ids
  8..151). The vocabulary is 152.
- **Model:** a GRU with embedding 32 and hidden 128, 1 layer, with a linear readout to 152. Use torch on CPU with
  `torch.set_num_threads(4)` and SEED 0 for the learner.
- **Scoring:** bits are counted **only on observation tokens**, as live −log2 p of the observation token given all
  previous tokens, including the current action. Actions are the agent's own choices, not data.
- **Training:** Adam with lr 1e-3 and clip 1.0. Chunks are 64 tokens (32 steps), with TBPTT 64 and the hidden state
  carried across chunks (detached at boundaries). **Prequential:** score each chunk's observation tokens before that
  chunk's step. The step loss is the mean NLL over the chunk's observation tokens.
- **Replay** (our best setting): after each live chunk, one step on 8 windows of 64 tokens, with starts uniform over
  all *even* token positions seen so far (so every window starts on an action token), zero initial state, and loss
  only on observation tokens.
- **Life length:** T = 60,000 steps (120,000 tokens).

## Fixed policies (no learning in the policy)
- **`rand8`:** uniform over {0..7}, rng `default_rng(5_000 + world_seed)`.
- **`rand6`:** uniform over {0..5}, so it never uses the TV or reset (same rng seed rule). It decomposes the gap: how
  much comes from avoiding reset/TV versus from planning.
- **`oracle`:** knows the world. Time is split into blocks of 512 steps, and block b targets room b mod 9. At the
  start of a block it presses reset, then plays key_0, …, key_{target−1} (3 actions each). For the rest of the block
  it takes uniform actions over {0..5}, drawn with rng `default_rng(6_000 + world_seed)`. When s = 2, it excludes
  key_k[2] so that it never leaves the target room. It never uses the TV.

## Probe: path-independent world knowledge
At checkpoints T/4, T/2 and T, freeze the learner (no gradient) and score these episodes, each from a fresh world state
with zero hidden state and teacher forcing:
- **Room probes:** for each room k ∈ {0..8}, reset, then play the keys to reach room k. Those tokens are fed in but not
  scored. Then play 96 manipulations from `default_rng(999 + k)` over {0..5}. In rooms k < 8, exclude key_k[2] when
  s = 2, so the probe stays in room k. Score the bits on those 96 observation tokens.
- **probe_k** = mean bits per observation in room k. **PROBE** = mean of probe_k over k = 0..8.
- **TV probe** (reported separately, not part of PROBE): from room 0, 96 TV presses.

## Runs
World seeds {0, 1, 2} × policies {rand8, rand6, oracle} gives 9 lives. Use three private **CPU** kernels, one per world
seed, each running its 3 policies in sequence: `ezzzzzx/w2-headroom-s0`, `-s1`, `-s2`. Each has `"is_private": true`,
**`"enable_gpu": false`** and `"enable_internet": false`. The code goes up as private dataset `ezzzzzx/pdp-code-w2`.
Do not use the GPU.

## Smoke test (VPS CPU, before any kernel)
On world seed 0, with T = 600 and oracle block size 32 (smoke only):
1. Determinism: two runs of the world under rand8 give the same token checksum.
2. The oracle visits all 9 rooms. Print time-in-room counts per policy.
3. A key check: playing key_0..key_7 from reset reaches room 8 in exactly 24 steps.
4. Bits and all probes are finite.
If any check fails, stop and report. Do not push kernels.

## Outputs
`results/w2_headroom/RESULTS.md` and `results/w2_headroom/results.json`, with numbers only and no verdict:
- per (world seed, policy): prequential bits per observation (whole life, and per quarter); time-in-room histogram;
  the deepest room reached and the first step it was reached; probe_k for all 9 rooms, PROBE and the TV probe, at the
  3 checkpoints; wall seconds;
- **headroom** = PROBE(rand8) − PROBE(oracle) at T, per seed and mean ± std (ddof = 0); the same for rand6 − oracle
  and rand8 − rand6;
- kernel ids and elapsed seconds;
- a 200-token excerpt of the oracle's stream for world seed 0, decoded as (action, room, y).

## Pre-registered bar (orchestrator scores; do not change)
**W2 qualifies for the curiosity run (M4 on W2) if headroom (rand8 − oracle) ≥ 0.5 bits per observation on every
world seed.** For scale: an observation costs at most log2(144) ≈ 7.2 bits, and a never-visited room should cost
about 4 bits or more. Report only: rand6 − oracle (the planning part of the gap) and rand8 − rand6 (the part from
avoiding reset and TV).

## Budget and rules
- CPU only. 90 minutes of wall clock for the kernels. A failed kernel may be rerun once.
- Run experiments in the foreground. Poll with `kaggle kernels status`.
- Files you may create: `w2_world.py` (world + policies + probe), `w2_headroom.py` (learner + runs), `kaggle/w2/**`,
  `results/w2_headroom/**`.
- Do not touch: `docs/loop.md`, `audit.md`, any `w1*` or `h29*` file or folder, `.env`.
- Never print or upload secrets. Do not commit.
