# M11-W6: credit sleep with the loss it prevents, measured against a shadow cortex that did not sleep (codex: implement + smoke; grok pushes kernels)

Read `/root/pdp/CLAUDE.md` first, then `results/m9w5/RESULTS.md`, `results/m10w6/RESULTS.md`, and the code in `m9_w5.py`, `m10_w6.py`,
`w6_headroom.py`, `w5_headroom.py` and `w4_headroom.py`.

## Context
M9-W5: a fixed sleep share of 0.20 beats no sleep by 0.32 (12 of 12), but an agent that chooses sleep as a 19th bandit arm sleeps only
0.059 of its life and loses 0.16 to the fixed dose. Its sleep reward was the progress during the sleep visit. Sleep's value is mostly
retention, bits **not lost later**, which immediate progress cannot see. M11 credits a sleep visit with the loss it prevents, measured
honestly: a shadow copy of the cortex that skipped the sleep sees exactly the same later glances and learns from them in the same way,
and the difference between the two on the exam store after a fixed horizon is the sleep's reward. If the agent then sleeps about as
much as it should, sleep becomes the machine's own choice, which is the north star's test for sleep.
Principles served: reward compression progress (3), act to maximise it (4); sleep as an organic action.

## World and fixed parts
`World6` (`w6_world.py`), birth_period = 128, L = 4,096. Cortex, stores, movement, dwell 8, exam store and T_open as in `w6_headroom.py`.
Room choice for every arm below is `prog_age_s` (M8's ageing bandit, τ = 256, ε 0.1, β 0.3) over the 18 slots. Import; wrap or
subclass only.

## Arms
- `age_fixed20`: prog_age_s with W6's fixed sleep share 0.20 and its placement (the W6 baseline).
- `age_sleeparm`: M9's design on W6: sleep is a 19th arm of the same bandit, a sleep visit is 8 sleep decisions with no travel, its
  reward is the whole exam store's bits saved over the visit divided by its decisions (as `m9_w5.py`). No forced sleep.
- `age_sleep_cf` (new): as `age_sleeparm`, but the sleep arm's reward is **counterfactual**:
  - At the start of a sleep visit (decision t0) take a deep copy S of the cortex (weights, optimiser state, recurrent state, traces,
    stores: everything the sleep visit changes).
  - The live cortex sleeps 8 decisions as usual. S does not sleep: it is held unchanged during those 8 decisions.
  - For the next H = 64 wake decisions after the visit, every glance the live cortex observes is also fed to S, which learns from it
    with exactly the same rule and the same rng draws (if the learning step uses rng, give S its own copy of that rng state at t0).
    S never chooses actions; the live agent's actions drive the world. If another sleep visit starts within H, S also skips it
    (it stays awake and is fed nothing during that visit), and that visit gets its own shadow.
  - At t0 + 8 + H, score both on the whole exam store (all slots' current exam items, frozen forward pass as the exam uses). The
    reward is (bits(S) − bits(live)) / (number of exam symbols) / 8, i.e. bits per exam symbol saved per sleep decision. It is
    delivered to the sleep arm at that moment (delayed). Room arms' rewards are unchanged (prog_age_s's).
  - Life end: a visit whose horizon is cut off by L is scored at L with the horizon it had.
  - The bandit compares arms on one scale, so convert the room arms' reward into the same units for the comparison only: room
    reward / (that room's exam symbols) per decision. State the conversion and print both scales in the smoke test.
Constants fixed in advance (H = 64, the rest from M8/M9); do not tune.

## Seeds and lives
**Fresh world seeds 36–47.** 3 arms × 12 seeds = 36 lives. Also report `age_nosleep` (prog_age_s, never sleeps) on the same seeds as
the anchor: 48 lives.

## Your job (codex): implement + smoke test only
Your sandbox has no network, so you do not push kernels. Write `m11_w6.py` (with a `--seed` kernel entry point that runs every arm for
one seed with a pool of 4 and writes JSON), prepare one kernel folder per seed in `kaggle/m11w6/s<seed>/` (`kernel-metadata.json`
with `"id": "ezzzzzx/m11-w6-s<seed>"`, `"is_private": true`, `"enable_gpu": false`, `"enable_internet": false`, code from private dataset
`ezzzzzx/pdp-code-m11w6`, prepared in `kaggle/m11w6/code/`), and a foreground dispatcher `kaggle/m11w6/dispatch.py` (at most 2 kernels
at once, one retry, then fetch outputs into `results/m11w6/` and write `results/m11w6/RESULTS.md`). A later grok job runs the dispatcher.

## Smoke test (VPS CPU, one process, before you finish)
1. `age_fixed20` and `age_nosleep` reproduce W6/M10's code bit for bit on seed 12, period 128, L = 512 (T_open to 1e-6).
2. **Shadow is exact:** with the sleep visit replaced by 8 decisions that change nothing, the shadow and the live cortex give identical
   exam bits at t0 + 8 + H (to 1e-9). So any reward is caused by the sleep alone.
3. For `age_sleep_cf`, seed 12, L = 1,024: print every sleep visit (t0, the reward when delivered, the arm values, the sleep share so far)
   and both reward scales for the first 12 room visits.
4. All arms finite at L = 512; time a 1,024-decision life of `age_sleep_cf` and estimate one kernel's wall time. If it exceeds 80
   minutes, split each seed into two kernels by arm (say so in RESULTS.md).
If a check fails, stop and report.

## Outputs (written by the dispatcher after the kernels; numbers only, no verdict)
`results/m11w6/RESULTS.md` and `results.json`: T_open and final checkpoint per arm and seed, mean ± std (ddof = 0), per checkpoint;
sleep share per arm and seed; gaps per seed with counts of seeds where the left term is lower: age_sleep_cf − age_sleeparm,
age_sleep_cf − age_fixed20, age_fixed20 − age_nosleep; the counterfactual rewards (mean, range, by decision); the smoke output; walls;
kernel ids.

## Pre-registered bars (orchestrator scores; do not change)
1. **Counterfactual credit beats immediate credit:** age_sleep_cf − age_sleeparm ≤ −0.05 on T_open, on at least 8 of 12.
2. **The agent's own sleep is as good as the hand-set dose:** age_sleep_cf − age_fixed20 ≤ +0.03 on T_open (mean over seeds).
Report only: sleep shares; age_fixed20 − age_nosleep (does sleep pay in W6?).
Kill rule: if bar 1 fails while age_fixed20 beats age_nosleep, a 64-decision counterfactual does not capture what sleep is worth; record
the counterfactual rewards' size against the room rewards.

## Budget and rules
- Implement and smoke test within about 90 minutes of your time. Run the smoke in the foreground.
- Files you may create: `m11_w6.py`, `kaggle/m11w6/**`, `results/m11w6/**`.
- Do not touch: `docs/loop.md`, `audit.md`, `w6_world.py`, `w6_headroom.py`, `m10_w6.py`, `m10b_w6.py`, `m9_w5.py`, `m8_w5.py`, any
  other experiment code, other `results/` or `kaggle/` folders, `.env`. Grok runs H34a's kernels today (`kaggle/h34a/**`,
  `results/h34a/**`): do not edit them.
- Gates and bars are the orchestrator's. Do not change or reinterpret them; if one looks wrong, stop and report.
- Never print or upload secrets. Do not commit.
