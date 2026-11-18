# W7, "the dotted window": a design draft and a headroom estimate (grok; design + small CPU checks; no Kaggle)

Read `/root/pdp/CLAUDE.md`, `docs/curiosity.md`, `docs/roadmap.md` (the open-world and W-series sections), and `w6_world.py`, `w4_headroom.py` for how worlds
and headroom checks are built here.

## The idea (from Ez)
A window covered in dots hides a logo behind it. Standing still, every view shows the same fragments, so nothing more can be learned. Moving the head shifts
the near dots more than the far logo (parallax), each view uncovers other pieces, and a learner that knows its own motion can integrate them into the logo, the
thing that stays the same while its own motion changes everything else. A curious agent should discover by itself that moving pays.

## Deliverable
`internal/w7_design.md` (internal; the orchestrator will turn it into a spec) and a small script `w7_sketch.py` for the checks below. Propose concretely:
1. **World:** a hidden scene (e.g. a 32×32 image of a few glyphs, 4–16 grey levels), a near occluder (a fixed dot mask blocking a set fraction of pixels),
   viewpoint actions (move in x/y, stay), and an observation model with parallax (the occluder shifts by a different amount than the scene per unit of
   viewer motion). Observations as a symbol stream our cortices can read (state the alphabet and ordering).
2. **Learner:** the simplest generic learner that *could* integrate views if given the agent's own motion (efference copy) as an input, without hard-coding the
   two layers. Say exactly what it sees.
3. **Probe for "reading the logo":** a held-out score of the hidden scene that does not reward staring at one view (e.g. predict unoccluded pixels at random
   viewpoints).
4. **Policies for the headroom check:** stand still, random walk, an oracle sweep that visits viewpoints covering the scene, and the curious agent (progress per
   viewpoint region).
5. **A first headroom estimate:** with `w7_sketch.py` on the VPS (one process, ≤ 10 minutes total), the probe score for stand-still vs random walk vs sweep at 2–3
   occluder densities. Report the numbers.
6. Failure modes you foresee (e.g. the learner memorises views without integrating; motion that is too easy so random already reads the logo).

Rules: files you may create: `internal/w7_design.md`, `w7_sketch.py`. Do not touch `docs/`, other code, `.env`. Never print secrets. Do not commit.
