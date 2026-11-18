# W7-0: the dotted window, scored. Does a curious agent discover that moving reveals the logo? (grok; VPS CPU, runs take seconds)

Read `/root/pdp/CLAUDE.md`, your design `internal/w7_design.md` and `w7_sketch.py`. Build the scored version from the sketch, with the changes below.
The runs are seconds long, so they run on the VPS (CLAUDE.md allows local runs under ~10 minutes; say the total wall time). Use one process.

## World, learner, probe
Exactly as in `internal/w7_design.md`: the 32×32 logo (fixed), the dot field at opacity 768/1024 (primary) and 512 and 922/1024 (report), parallax as
designed, the integrator learner (with the agent's own motion as input), and the viewpoint probe (bits per scored grey at held-out viewpoints).
Lives L ∈ {25, 64}. Write `w7_world.py` (world + learner + probe) and `w7_run.py` (policies, seeds, outputs).

## Policies
- `still`, `random` (as in the sketch), `sweep` (the oracle).
- `curious_prog` (new, principle 3): the 16 regions of the sketch. A region's value is the **compression progress** of the last visit: bits of a stored-glance
  reservoir (all glances so far, **including the current glance**, capped at 256 by uniform reservoir sampling, rng `default_rng(9000 + seed)`) under the
  integrator before the visit's glance was learned minus after. EMA β 0.5, unvisited regions valued at +∞ (visited once each in index order first),
  ties by fewest visits then lowest index; walk the shortest path to the chosen region's centre.
- `curious_surp`: the sketch's surprise policy (report only).
Controls (report only): `glued` (the dot field moves with the logo; every policy) and `viewtable` (a learner that memorises views without motion input).

## Seeds
**Fresh seeds 10–29** for the dot field and the random walk (20 seeds). The logo is fixed.

## Outputs
`results/w7/RESULTS.md` + `results.json`, numbers only. **Scoring table at the top** (opacity 768, per life): per seed, curious_prog − random, curious_prog − sweep,
random − still; means; counts meeting each threshold. Then every policy × opacity × life (mean ± std), coverage, logo bits, the controls, the curious agent's
visit order for 3 seeds, total wall time.

## Pre-registered bars (orchestrator; do not change), opacity 768/1024
1. **Moving reveals the logo:** random − still ≤ −0.30 at L = 25 on ≥ 18 of 20 seeds.
2. **Curiosity finds the motion that pays:** curious_prog − random ≤ −0.30 at L = 25 on ≥ 16 of 20 seeds.
3. **Close to a planned sweep in a longer life:** curious_prog − sweep ≤ +0.10 at L = 64 (mean over seeds).
Report only: other opacities, curious_surp, the controls.

Rules: files you may create: `w7_world.py`, `w7_run.py`, `results/w7/**`. Do not touch `docs/`, `internal/` (read only), other code, `.env`. Never print secrets. Do not commit.
