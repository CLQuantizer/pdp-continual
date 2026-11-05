"""Minimal CLS wake/sleep dual (loop 09:40 UTC+8 1 Nov; Gall's law: one working sleep path on the working default).

Hippocampus = the default store (a25w250h2k2_c1.5_uinf_z5): one-shot episodic writes and its own remaps, by day only.
Neocortex  = the FF network. By day it answers and learns from the world at wake LR x`wl`.
Answer     = route c3 (store answers when it disagrees with FF and its relative margin >= 0.3, or the mq30_s2 gate):
             the combined system, scored in-loop. FF-only and store-only routes are scored on the same arrivals.
Sleep      = circadian: at the start of each night the machine stops answering for `chunks` chunks (10 batches of 32,
             40 ticks each; arrivals then are misses and nothing is learned). The cortex learns ONLY from replay:
  uni      uniform replay of store prototypes (life.py's `night`, control)
  swr      sharp-wave-ripple selective replay: by day the hippocampus tags episodes the cortex got wrong (prediction
           error; FIFO cap `cap`, cleared after each sleep). At night each tagged episode is replayed as the store's
           CURRENT nearest prototype with the store's label (so committed remaps are what the cortex is taught),
           interleaved 50/50 with uniform store prototypes (CLS interleaving protects old schemas).
Conditions (s49 world: bg 0.02, 4<->9 swap; seeds 3-5, held out for the c3 router; nothing tuned here):
  none        never sleeps, wake LR x1 (baseline; must reproduce the offline c3 numbers 96.12 / 74.8)
  uni8        uniform night replay, 8 chunks, wake LR x1
  swr2        selective, 2 chunks, wake LR x1
  swr8        selective, 8 chunks, wake LR x1
  swr2_wl3    selective, 2 chunks, wake LR x0.3 (slow cortex + sleep: the textbook CLS split)
Pre-registered (combined c3 route, vs `none`, mean of seeds 3-5):
  PASS if some sleep condition has lifetime >= none - 0.10 AND (returns >= none + 1.0 OR swap300 >= none + 2.0 with
  returns >= none - 0.5).  KILL if every sleep condition misses that.  Secondary (CLS mechanism check, not a gate): swr
  should lift FF-only returns over `none` by >= 3 (the cortex actually consolidates what the store holds).
Replication (reported, same bars): bg002 world (bg 0.02, 2<->3 swap), seeds 6-8.
Result 09:40 run (both worlds): KILL. No sleep condition met the bar (see results/cls_sleep/RESULTS.md).
Follow-up H2, pre-registered 10:25 UTC+8 1 Nov before running ("temporary hippocampus"; s49 seeds 3-5):
  Why sleep can't win above: the store never forgets, so returns are already answered by the store and sleep has
  nothing to hand over. CLS assumes the hippocampus is temporary. `*_f2k` drops every class unseen for >= 2000 ticks
  (= RET_GAP), so on a return only the cortex can know the class.
  PASS if swr2_f2k or uni8_f2k has combined returns >= none_f2k + 3.0 AND lifetime >= none_f2k - 0.10.
  KILL otherwise (sleep does not even rescue a forgetful hippocampus -> the FF cortex can't consolidate usefully).
  Reported, not a gate: gap of the best sleep_f2k to `none` (permanent store, never sleeps).
Result H2: KILL (results/cls_sleep/RESULTS.md).
H3 night answering, pre-registered 10:48 UTC+8 1 Nov before running (s49 seeds 3-5 AND bg002 seeds 6-8):
  The whole lifetime cost of sleep was downtime. `*_na`: through each night the store answers alone (FF inference
  only scored as its own route), nothing learns from night arrivals; the cortex learns only from replay as before.
  PASS if uni8_na or swr8_na, in BOTH worlds, has combined lifetime >= none - 0.10 AND (returns >= none + 1.0 OR
  swap300 >= none + 2.0 with returns >= none - 0.5). swr2_na reported with the same bars. KILL otherwise.
  `none` rows reused from the 09:40 runs (never-sleeping code path unchanged, deterministic per seed).
Result H3 (11:37 UTC+8): PASS in both worlds for swr8_na (s49 -0.04 / +2.3 returns; bg002 -0.09 / +4.5) and swr2_na
  (s49 -0.01 / +1.3; bg002 +0.01 / +1.5); uni8_na passes s49, fails bg002 lifetime (-0.12).
H3c held-out confirmation, pre-registered 11:38 UTC+8 1 Nov before running: fresh seeds s49 140-142 and bg002 143-145,
  conditions none, swr2_na, swr8_na, same bar as H3 against same-seed `none`. Confirmed if swr2_na or swr8_na passes in
  BOTH worlds; then it becomes the default core (never-sleeping z5 -> z5 + store-answered nights + replay). Else H3 was noise.
H4 awake interleaved remote replay, pre-registered 11:22 UTC+8 1 Nov before running (s49 seeds 3-5):
  No nights. The forgetful store (f2k) is the case where the cortex has a job (returns 74.8 -> 58.0). H2 showed night replay
  is overwritten by full-rate day learning; CLS's own prescription is interleaving. Every 38th wake arrival, one batch of
  store prototypes from classes unseen >= 300 ticks (awr_f2k) or any class (awu_f2k, control) goes into the cortex.
  PASS if awr_f2k returns >= none_f2k + 3.0 AND lifetime >= none_f2k - 0.10. 'remote' selection counts only if awr_f2k
  beats awu_f2k on returns by >= 1.5. Regression check on the permanent store: awr lifetime >= none - 0.10.
  KILL otherwise. none_f2k / none reused from the 10:25 / 09:40 runs.
Result H4 (11:39 UTC+8): PASS. awr_f2k +0.15 lifetime / +11.8 returns vs none_f2k; awu_f2k +0.15 / +9.3 (remote beats
  uniform by 2.4 >= 1.5); permanent store awr +0.10 lifetime / +6.1 returns / +1.6 swap vs none (all 3 seeds up on returns),
  better than every night condition. No sleep needed.
H4c held-out confirmation, pre-registered 11:40 UTC+8 1 Nov before running: awr on the H3c fresh seeds (s49 140-142,
  bg002 143-145) against same-seed `none` from H3c. Confirmed if, in BOTH worlds, lifetime >= none - 0.10 and returns >=
  none + 1.0 (same bar as H3). Head-to-head reported: awr vs swr2_na / swr8_na. If both awr and a *_na confirm, the
  default becomes the one with higher mean returns at lifetime within 0.05; combining them is the next test, not a default.
Result H3c/H4c (12:35 UTC+8): both CONFIRMED in both worlds. swr2_na +0.07/+2.2 (s49), -0.04/+2.0 (bg002); swr8_na
  -0.04/+2.0, -0.01/+3.0; awr +0.05/+4.4, +0.15/+5.4 (all six seeds up). Tie-break -> awr is the default core.
H5 similarity-weighted awake interleaving, pre-registered 12:15 UTC+8 1 Nov before running (s49 seeds 3-5):
  McClelland, McNaughton & Lampinen 2020 (Phil Trans R Soc B 375:20190637) and Saxena et al. 2022 (PNAS): new learning
  interferes mostly with *similar* old knowledge, so interleaving only old items similar to the new one protects as much as
  full interleaving at a fraction of the cost. awr picks replay by absence (classes unseen >= 300 ticks). aws keeps the
  same rate (every 38th arrival, one batch) but picks the store prototypes of other classes nearest the current arrival.
  PASS (similarity is the better selector) if aws_f2k returns >= awr_f2k + 1.5 AND lifetime >= awr_f2k - 0.05, or
  swap first 300 >= awr_f2k + 2.0 with returns >= awr_f2k - 1.0. Permanent-store check: aws lifetime >= awr - 0.05.
  KILL otherwise: absence stays the selector. awr_f2k / awr reused from results/cls_awake (same seeds).
Result H5 (12:44 UTC+8): KILLED. aws_f2k 96.15/65.4/66.6 vs awr_f2k 96.21/69.8/66.2 (returns -4.4, below uniform 67.4);
  aws 96.22/80.1/67.0 vs awr 96.22/80.9/68.8. The cortex forgets what it is not shown; absence stays the selector.
H6 return-aware arbiter, pre-registered 13:15 UTC+8 1 Nov before running (s49 seeds 3-5 AND bg002 seeds 6-8):
  Store-only returns 88-89 vs combined 84 under awr: c3 gives away store-right returns. Daw, Niv & Dayan 2005: control
  goes to the system whose estimate is less uncertain; the cortex's estimate of a class it has not been trained on for a
  while is stale. awr_ra: when store and FF disagree and the store's class was last seen >= 300 ticks ago, the store
  answers. awr_raq: same unless FF's margin is in the top 30% of the last 500 arrivals.
  PASS if awr_ra or awr_raq, in BOTH worlds, has combined returns >= awr + 2.0 AND lifetime >= awr - 0.10 AND swap300
  >= awr - 1.0 AND novel300 >= awr - 1.0. KILL otherwise (c3 stays). awr s49 3-5 reused from results/cls_awake;
  awr bg002 6-8 run alongside. Reported: ra_used per run and the share of overrides where store / FF was right.
Result H6 (13:30 UTC+8): KILLED. Returns +2.9 (s49) / +3.1 (bg002) but lifetime -2.4 / -2.8 and novel300 -14.6 / -13.4.
  ~1,500 overrides per run, store right on 3%, FF on 84%: an absent store class is an error signal, not a return signal.
H6b/H6c, pre-registered 14:15 UTC+8 1 Nov before running (s49 seeds 3-5 AND bg002 seeds 6-8; awr reused):
  Audit of H6: `prev_seen` is updated by every label, so the absent-class override could only ever touch the FIRST arrival
  of a return (ra_right ~41-60 per run = the +3 returns); every later override was a store error onto a class still away.
  H6b as written (store names the same absent class twice within 20 ticks) therefore selects repeated *errors*: predicted
  store-right share < 10% (logic check, s49 seeds 3-4 only). H6c awr_rw instead uses the label as the evidence: after a
  class reappears (label) following >= 300 ticks away, for its next 20 labelled arrivals the store answers any
  disagreement naming that class. PASS (same bar as H6) if awr_rw, in BOTH worlds, has returns >= awr + 2.0 AND
  lifetime >= awr - 0.10 AND swap300 >= awr - 1.0 AND novel300 >= awr - 1.0. KILL otherwise.
SCORING CHANGE (1 Nov 14:05 UTC+8, Ez: "don't evaluate when it is sleeping"). Arrivals during a sleep chunk are no
  longer scored, whether they were missed or answered by the store (`*_na`). `route` = awake arrivals only (primary);
  `route_match` = the same mask for every machine (first 8 chunks of every night dropped, sleepers or not);
  `route_all` = the old score (asleep misses count wrong). Swap / novel = first 300 evaluated arrivals.
  Under `route`, `swr8` and `swr8_na` must be identical (answering at night changes nothing that is learned).
H7 (pre-registered 14:10 UTC+8, before running): rerun s49 3-5 and bg002 6-8 under the new scoring with none, awr,
  uni8, swr2, swr8, swr8_na, and night replay on top of awake replay (swr8_awr, uni8_awr). Sleep re-enters the core
  only if some sleep condition, in BOTH worlds, has awake-scored lifetime >= awr - 0.05 AND returns >= awr + 1.0
  (or swap300 >= awr + 2.0 with returns >= awr - 0.5). Otherwise awr stays the default and nights stay optional.
Result H6b/H6c (14:40 UTC+8): both KILLED. awr_rb s49 95.47/81.0, store right 2.4% (as predicted). awr_rw s49 92.49/87.0,
  bg002 91.33/86.9: returns +6.1/+5.1 but lifetime -3.7/-4.5, novel300 -11/-10; ~2,500 overrides/run, store right < 4%.
  Background arrivals reopen windows constantly. Routing by class state is closed; c3 stays.
H7 awr + swr2_na, pre-registered 14:45 UTC+8 1 Nov before running (s49 seeds 3-5 AND bg002 seeds 6-8; awr reused):
  Does night replay (store answers at night, cortex learns only from selective replay) add on top of awake remote replay?
  PASS if awr_swr2_na, in BOTH worlds, has returns >= awr + 1.0 AND lifetime >= awr - 0.05 AND novel300 >= awr - 1.0.
  KILL otherwise (nights stay out of the core).
Result H7 (15:10 UTC+8): KILLED. awr_swr2_na s49 96.18/80.3/66.9 (novel 76.3) vs awr 96.22/80.9/68.8 (returns -0.6,
  swap -1.9); bg002 95.87/82.5/69.8 vs awr 95.86/81.8/69.7 (returns +0.7 < +1.0). Awake scoring, awr reused from the
  14:10 rerun (results/cls_wake_*, same seeds). Nights add nothing on top of awake remote replay; they stay out of the core.
H8 awake replay rate, pre-registered 15:10 UTC+8 1 Nov before running (s49 seeds 3-5 AND bg002 seeds 6-8; awr reused):
  awr19 (replay every 19th wake arrival, 2x the cost) and awr76 (every 76th, half the cost), otherwise identical to awr.
  awr19 replaces awr only if, in BOTH worlds, returns >= awr + 1.0 AND lifetime >= awr - 0.05 AND novel300 >= awr - 1.0.
  awr76 replaces awr (Gall: same result for half the replay) if, in BOTH worlds, returns >= awr - 1.0 AND
  lifetime >= awr - 0.05 AND swap300 >= awr - 1.0. If both pass, awr76 wins. KILL otherwise: every=38 stays.
Result H8 (16:20 UTC+8): KILLED, every=38 stays. s49 awr19 96.24/81.8/67.0 (returns +0.94 < +1.0), awr76 96.27/80.5/67.1
  (swap -1.7 < -1.0); bg002 both pass (awr19 returns +1.5; awr76 95.85/81.8/69.3). 4x the budget moves returns ~1.4 points.
H10 selective bus vs compression progress, pre-registered 16:10 UTC+8 1 Nov before scoring (bg002 seeds 6-8: none, awr, awu):
  Group constraint: keep the absence selector only if it buys compression progress, not just return accuracy. Measure the
  cortex's prequential code length on the growing history: bits = -log2 softmax(FF scores)[label], taken on every awake
  arrival before the cortex fits it (ff_bits; lower = better). awu = uniform awake replay at the same rate (every 38th, any class).
  KEEP selectivity if awr ff_bits.all <= awu - 0.02 AND awr ff_bits.all <= none. If awr's code length is not below awu's
  while its returns are, the selector is buying return accuracy without compressing the history better: flag it to the group
  and test a mixed selector next. awr staying ahead of none on ff_bits is the minimal sanity check of replay itself.
Result H10 (17:20 UTC+8): KILLED. ff_bits.all none 0.5145 / awr 0.4603 / awu 0.4566 (awr - awu = +0.004, bar <= -0.02);
  returns 75.2 / 81.8 / 81.9, lifetime 95.86 / 95.86 / 95.92. Replay compresses (-0.054 bits vs none); the absence selector
  does not (it helps only return bits -0.06 and swap300 +1.9, loses novel300 -2.0). Selectivity is not justified as a bus here.
H11 gain-gated awake replay, pre-registered 17:25 UTC+8 1 Nov before running (s49 seeds 6-8 AND bg002 seeds 6-8; awr
  bg002 6-8 reused from H10, runs are deterministic). Mattar & Daw 2018: replay priority = gain x need. awr is need-only
  (absence); H10 asks whether that need term compresses better than uniform. awg drops the absence selector and keeps only
  a gain gate: one replay token per 38 awake arrivals (cap 8); spend one on a uniform store batch only while the running
  mean of ff_bits over the last 38 awake arrivals exceeds the median of that running mean over the last 500 arrivals.
  awg replaces awr only if, in BOTH worlds, returns >= awr + 1.0 AND lifetime >= awr - 0.05, or returns >= awr - 0.5 AND
  lifetime >= awr - 0.05 at <= half awr's replay per wake. It counts as a selective bus (group constraint) only if, in
  BOTH worlds, also ff_bits.all <= awu - 0.02. H11b (Gall, awu reference run alongside on s49 6-8): awu (uniform clock,
  no selector at all) replaces awr if, in BOTH worlds, returns >= awr - 0.5 AND lifetime >= awr - 0.05 AND swap300 >= awr - 1.0.
Result H11 (18:20 UTC+8): KILLED on bg002 (bar needs BOTH worlds; s49 awr 6-8 still finishing on Dubai, reported only).
  awg 95.89/81.8/69.4 vs awr 95.86/81.8/69.7 (returns +0.05 < +1.0, same replay 0.83 vs 0.82 per wake); ff_bits.all
  awg 0.4624 vs awu 0.4566 (worse, bar <= awu - 0.02): the gain gate is not a selective bus. H11b: awu fails bg002 on
  swap300 67.8 vs awr 69.7 (-1.9 < -1.0); returns 81.9 / lifetime 95.92 pass. Need-only, gain-only and uniform selection
  sit within 0.006 bits of each other at a matched budget: with a permanent store, that replay happens matters, not what.
H13 selection under a forgetful store, pre-registered 18:20 UTC+8 1 Nov before running (s49 seeds 6-8 AND bg002 seeds 6-8:
  none_f2k, awr_f2k, awu_f2k; all scored awake-only with ff_bits). H4 found absence beats uniform by +2.5 returns when the
  store forgets (the cortex must own returns); H10/H11 found no selector compresses better with a permanent store.
  KEEP the absence selector (as the default whenever the store can forget) only if, in BOTH worlds, awr_f2k
  ff_bits.all <= awu_f2k - 0.02 AND returns >= awu_f2k + 1.5 AND lifetime >= awu_f2k - 0.05. Sanity: awu_f2k
  ff_bits.all <= none_f2k - 0.02. KILL otherwise: absence is demoted in both regimes (kept only as a swap300 note) and
  the default awake replay is the uniform clock; H12 (gain x need) is then parked, since neither factor compresses alone.
Result H13 (18:50 UTC+8): KILLED on compression only. awr_f2k - awu_f2k ff_bits.all +0.002 (s49) / -0.001 (bg002),
  bar <= -0.02. Accuracy passes both worlds: returns 71.1 vs 68.6 (+2.5) and 69.5 vs 66.1 (+3.4), lifetime -0.02 / -0.04.
  Return-arrival bits are lower under absence in 6/6 seed pairs (~ -0.11): a reallocation toward absent digits, not
  better compression of the history. Absence is demoted in both regimes; the default awake replay is the uniform clock.
H14 per-memory interference (MIR, Aljundi et al. 2019, look-back form), pre-registered 19:10 UTC+8 1 Nov before running
  (s49 AND bg002 seeds 6-8, forgetful store f2k; awu_f2k / awr_f2k reused from H13, deterministic). mir_f2k replays, at the
  same slots as awu (every 38th wake arrival), the 32 of 128 previously sampled store prototypes whose cortex bits rose most
  since they were sampled. KEEP as selective bus only if, in BOTH worlds, ff_bits.all <= awu_f2k - 0.02 AND lifetime >=
  awu_f2k - 0.05. Diagnostic only (cannot rescue a kill): ff_bits.returns vs awr_f2k (protector vs reallocation). KILL
  otherwise: core stays z5 + c3 + awu and per-memory selection is closed for the permanent and f2k stores.
H15 replay budget (Fedus et al. 2020 replay ratio; Chaudhry et al. 2019 tiny memories), pre-registered 08:55 UTC+8
  4 Nov 2026 before running (s49 AND bg002 seeds 6-8, forgetful store f2k; awu_f2k / none_f2k reused from H13/H14,
  deterministic). awu19_f2k / awu76_f2k = uniform awake replay of one 32-prototype batch every 19th / 76th wake arrival
  (1.68 / 0.42 replayed per wake arrival; awu_f2k = 0.84). KEEP "budget-limited" only if, in BOTH worlds, each doubling
  buys >= 0.02 bits/arrival on ff_bits.all (awu76 - awu38 >= 0.02 AND awu38 - awu19 >= 0.02) and lifetime >= awu_f2k - 0.05
  at every budget. KILL (flat) if awu38 - awu19 < 0.02 in either world: the core is not budget-limited above every 38,
  every=38 stays, and the next lever is how replay changes the weights (H16 interleave / learning rule), not how much.
Result H15 (09:20 UTC+8 4 Nov): KILLED, knee at every 38. ff_bits.all none/76/38/19: s49 0.5104/0.4818/0.4530/0.4438,
  bg002 0.5145/0.4860/0.4594/0.4457. 76->38 buys 0.029/0.027 (pass); 38->19 buys 0.009/0.014 (fail, though lower in 6/6
  seeds). Lifetime passes everywhere. every=38 stays; next lever is how replay changes the weights (H16), not how much.
H16 interleave at matched gradient steps (McClelland, McNaughton & O'Reilly 1995 interleaved learning; Chaudhry et al.
  2019 experience replay = one SGD step on current + memory batch), pre-registered 08:55 UTC+8 5 Nov 2026 before scoring
  (s49 AND bg002 seeds 6-8, forgetful store f2k; awu_f2k reused from H13, determinism re-checked on s49 seed 6).
  awi_f2k = on each replay slot the arrival is fitted inside the replay batch ([x + 32 store prototypes], one step) instead
  of alone then a separate batch. awid_f2k = dense: every arrival is one step on [x] plus, with probability 32/38, one
  uniform store prototype. Both: 38 steps per 38 arrivals (awu 39), 0.84 replayed per arrival. KEEP a variant only if, in
  BOTH worlds, ff_bits.all <= awu_f2k - 0.02 AND lifetime >= awu_f2k - 0.05. KILL otherwise: separate-step replay stays,
  and the remaining weight-side lever is cortex capacity / plasticity (H17).
Result H16 (09:05 UTC+8 5 Nov): awi_f2k KILLED (ff_bits +0.007 both worlds). awid_f2k compression PASSES (ff_bits.all
  0.4152 / 0.4173 vs awu 0.4530 / 0.4594, -0.038 / -0.042, 6/6 seeds) but lifetime FAILS (95.67 / 95.71 vs 95.82 / 95.89,
  -0.15 / -0.18 vs bar -0.05, every seed): KILLED as default; core unchanged. Returns +6.6 / +7.9, return bits -0.35 / -0.38.
H16b dense pairing rate, pre-registered 09:06 UTC+8 5 Nov before running (s49 AND bg002 seeds 6-8, f2k). awid8_f2k /
  awid16_f2k = awid at pairing probability 8/38 / 16/38 (0.21 / 0.42 replayed per arrival, a quarter / half of awu's
  items, still 38 steps per 38 arrivals). KEEP the lowest-rate variant that, in BOTH worlds, has ff_bits.all <=
  awu_f2k - 0.02 AND lifetime >= awu_f2k - 0.05: it becomes the default awake replay. KILL if none does: dense pairing
  trades present-stream tracking for compression one-for-one, and separate-step awu stays.
Result H16b (09:20 UTC+8 5 Nov): KILLED, no pairing rate passes both. ff_bits.all p=8/16/32 of 38: s49 0.4600 / 0.4331 /
  0.4152, bg002 0.4661 / 0.4347 / 0.4173 (awu 0.4530 / 0.4594); lifetime s49 95.72 / 95.73 / 95.67, bg002 95.82 / 95.77 /
  95.71 (awu 95.82 / 95.89). p=16 reaches the bits bar in bg002 (-0.025) and misses it by 0.0001 in s49; lifetime fails at
  every rate. Dense pairing trades present-stream accuracy (~0.1 to 0.2 points) for compression; separate-step awu stays.
Usage: OMP_NUM_THREADS=1 python3 cls_sleep.py [--world s49|bg002] [--seeds 3 4 5] [--conds ...] [--days 16]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, time
from collections import deque
from multiprocessing import Pool
import numpy as np
import life as LF
import learners as L
import remap_probe as RP
import rival_s49 as RS
from data import load_mnist

BASEK = "a25w250h2k2_c1.5_uinf_z5"
WORLDS = {"s49": dict(bg=0.02, swap=(4, 9)), "bg002": dict(bg=0.02)}
WORLD = WORLDS["s49"]
CONDS = {
    "none": dict(sleep=None, wl=1.0),
    "uni8": dict(sleep="uni", chunks=8, wl=1.0),
    "swr2": dict(sleep="swr", chunks=2, wl=1.0),
    "swr8": dict(sleep="swr", chunks=8, wl=1.0),
    "swr2_wl3": dict(sleep="swr", chunks=2, wl=0.3),
    # 10:00 loop: temporary hippocampus. Every 100 ticks (and after each night's replay) the store drops every class not observed
    # for >= F ticks (F = RET_GAP = 2000, so every return finds its class gone from the store).
    "none_f2k": dict(sleep=None, wl=1.0, forget=2000),
    "swr2_f2k": dict(sleep="swr", chunks=2, wl=1.0, forget=2000),
    "uni8_f2k": dict(sleep="uni", chunks=8, wl=1.0, forget=2000),
    # 10:48 loop: night answering. During night chunks the store keeps answering (store-only, no learning from the
    # night arrivals: no FF fit, no store observe/add); the cortex still learns only from replay.
    "uni8_na": dict(sleep="uni", chunks=8, wl=1.0, night_answer=True),
    "swr8_na": dict(sleep="swr", chunks=8, wl=1.0, night_answer=True),
    "swr2_na": dict(sleep="swr", chunks=2, wl=1.0, night_answer=True),
    # 11:22 loop: awake interleaved replay (Karlsson & Frank 2009; McClelland et al. 1995). No nights. Every 38th waking
    # arrival the cortex fits one batch (BS=32) of store prototypes: 'remote' = only classes unseen for >= 300 ticks,
    # 'uni' = any class. 32/38 = 0.84 replayed items per wake arrival, matched to uni8 (0.83).
    "awr_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="remote", every=38, gap=300),
    "awu_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="uni", every=38, gap=300),
    # H15 (4 Nov 08:55 UTC+8): replay budget, uniform, forgetful store
    "awu19_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="uni", every=19, gap=300),
    "awu76_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="uni", every=76, gap=300),
    # H16 (5 Nov 08:55 UTC+8): interleave the arriving item with replay at matched gradient steps, forgetful store.
    # awi_f2k: on each replay slot (every 38th wake arrival) the arrival is NOT fitted alone; one step on [x + 32 uniform
    #   store prototypes] instead (38 steps per 38 arrivals vs awu's 39; replay 0.84/arrival, same as awu).
    # awid_f2k: dense interleave (McClelland et al. 1995): every wake arrival is one step on [x] plus, with probability
    #   32/38, one uniform store prototype (38 steps per 38 arrivals; replay 0.84/arrival in expectation).
    "awi_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="uni", every=38, gap=300, inter="slot"),
    "awid_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="dense", every=38, gap=300, inter="dense"),
    # H16b (5 Nov 09:06 UTC+8): dense pairing at lower rates. Under Adam each step moves ~lr whatever its batch, so dense
    # pairing at p=32/38 gives old items ~half of 84% of update directions (vs 1 of 39 steps for awu). Is there a rate
    # that keeps the compression win without the lifetime cost? p = 8/38 and 16/38 (0.21 / 0.42 replayed per arrival).
    "awid8_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="dense", every=38, gap=300, inter="dense", pn=8),
    "awid16_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="dense", every=38, gap=300, inter="dense", pn=16),
    "awr": dict(sleep=None, wl=1.0, awake="remote", every=38, gap=300),
    # 12:10 loop: similarity-weighted interleaving (McClelland, McNaughton & Lampinen 2020; Saxena et al. 2022). Same rate;
    # the batch is the LF.BS store prototypes of *other* classes nearest the current arrival, not the absent classes.
    "aws_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="sim", every=38, gap=300),
    "aws": dict(sleep=None, wl=1.0, awake="sim", every=38, gap=300),
    # 13:10 loop H6: return-aware arbiter on the awr core (Daw, Niv & Dayan 2005 uncertainty arbitration). When the store
    # disagrees with FF and the store's class has been absent >= ra ticks before this arrival, the store answers
    # (ra: always; raq: unless FF's margin is in the top 30% of the last 500).
    "awr_ra": dict(sleep=None, wl=1.0, awake="remote", every=38, gap=300, ra=300),
    "awr_raq": dict(sleep=None, wl=1.0, awake="remote", every=38, gap=300, ra=300, raq=0.7),
    # H7 (14:10 UTC+8): night replay on top of the awake-replay default, scored awake-only
    "swr8_awr": dict(sleep="swr", chunks=8, wl=1.0, awake="remote", every=38, gap=300),
    "uni8_awr": dict(sleep="uni", chunks=8, wl=1.0, awake="remote", every=38, gap=300),
    # 14:15 loop H6b (literal, logic check): override as awr_ra only if the store also named that same absent class on
    # another disagreement within the last rb ticks.
    "awr_rb": dict(sleep=None, wl=1.0, awake="remote", every=38, gap=300, ra=300, rb=20),
    # 14:15 loop H6c label-confirmed return window: once a class's *label* reappears after >= rw_gap ticks away, for its
    # next rw_n labelled arrivals the store answers any disagreement in which the store names that class.
    # 14:45 loop H7: awake remote replay plus two-chunk selective nights with night answering.
    "awr19": dict(sleep=None, wl=1.0, awake="remote", every=19, gap=300),
    "awr76": dict(sleep=None, wl=1.0, awake="remote", every=76, gap=300),
    "awr_swr2_na": dict(sleep="swr", chunks=2, wl=1.0, night_answer=True, awake="remote", every=38, gap=300),
    # 16:00 loop H10 (selective bus vs compression progress): uniform awake replay at the awr rate, no forgetting.
    "awu": dict(sleep=None, wl=1.0, awake="uni", every=38, gap=300),
    "awr_rw": dict(sleep=None, wl=1.0, awake="remote", every=38, gap=300, rw_gap=300, rw_n=20),
    # 17:00 loop H11 (Mattar & Daw 2018: replay priority = gain x need). Gain-gated awake replay with no absence selector:
    # one token per 38 awake arrivals (cap 8); spend a token on one uniform store batch only while the running mean of
    # ff_bits over the last 38 awake arrivals exceeds the median of that running mean over the last 500 arrivals
    # (before 500 arrivals: spend whenever a token is there). Budget <= awr's 0.82-0.84 per wake arrival.
    "awg": dict(sleep=None, wl=1.0, awake="uni", every=38, gap=300, gate=38, cap=8),
    # 19:00 loop H14 MIR (Aljundi et al. 2019, look-back form): at each awake replay slot, re-score the 128 store
    # prototypes sampled at the previous slot; replay the 32 whose cortex bits rose most over the 38 wake updates since
    # (actual interference, not a virtual step). Then sample 128 fresh candidates and record their bits. Same rate as awu.
    "mir_f2k": dict(sleep=None, wl=1.0, forget=2000, awake="mir", every=38, gap=300, ncand=128),
    "mir": dict(sleep=None, wl=1.0, awake="mir", every=38, gap=300, ncand=128),
}
CAP, CHUNK_TICKS = 128, 40


def protos(st):
    mu = np.concatenate([st.mu[c] for c in sorted(st.mu)])
    ys = np.concatenate([np.full(len(st.mu[c]), c) for c in sorted(st.mu)])
    return mu, ys


def live(args):
    cond, seed, days, wk = args
    cfg = CONDS[cond]
    w = dict(LF.WORLD, **WORLDS[wk])
    xtr, ytr, _, _ = load_mnist("mnist")
    ticks = days * w["day"]
    world = LF.World(xtr, ytr, ticks, seed, w)
    lr = 1e-3
    ff = L.ForwardForward(lr=lr, loss="symba", homeo=0.001, seed=seed)
    st = RS.ZStore(9, np.random.default_rng(seed + 1000), noise=0.0, K=BASEK)
    rng = np.random.default_rng(seed + 3000)
    tags = deque(maxlen=CAP)
    cnt = np.zeros(10); ffms = deque(maxlen=500)
    last_seen, returning, rows = {}, {}, []
    n_sleep = replayed = tagged_replayed = 0
    t0, t = time.time(), 0
    forgot = 0
    night_answered = 0
    n_wake = 0
    ffbits = []          # H10: cortex prequential code length, -log2 softmax(FF scores)[y] before fitting each awake arrival
    ra_used = ra_right = ra_ffright = 0
    rb_ev = deque()      # (tick, store class) for store-names-absent-class disagreements (awr_rb)
    rw_left = {}         # class -> labelled arrivals left in its return window (awr_rw)
    tok, fired = 0.0, 0  # H11 awg: replay tokens and batches fired
    b38 = deque(maxlen=cfg.get("gate") or 1); rmh = deque(maxlen=500)
    mir_c = None         # H14: (x, y, bits) of the candidates sampled at the previous replay slot
    mir_inc = []         # H14: mean bit rise of the chosen batch vs all candidates, per slot

    def bits(xs, ys_):
        sc_ = ff.scores(xs.astype(np.float32)); z_ = sc_ - sc_.max(1, keepdims=True)
        return (np.log(np.exp(z_).sum(1)) - z_[np.arange(len(ys_)), ys_]) / np.log(2)

    def forget():
        nonlocal forgot
        F = cfg.get("forget")
        if not F:
            return
        for c in [c for c in list(st.mu) if t - last_seen.get(c, -10 ** 9) >= F]:
            if len(st.mu) <= 1:
                break
            for D in (st.mu, st.var, st.n, st.hist, st.rad):
                D.pop(c, None)
            forgot += 1

    def sleep_chunk():
        nonlocal replayed, tagged_replayed
        mu, ys = protos(st)
        for _ in range(LF.CHUNK):
            if cfg["sleep"] == "swr" and tags:
                k = LF.BS // 2
                xs = np.concatenate([tags[i] for i in rng.choice(len(tags), k)])
                j = ((mu[None] - xs[:, None]) ** 2).sum(2).argmin(1)   # the store's current memory of each episode
                u = rng.choice(len(mu), LF.BS - k)
                bx, by = np.concatenate([mu[j], mu[u]]), np.concatenate([ys[j], ys[u]])
                tagged_replayed += k
            else:
                u = rng.choice(len(mu), LF.BS)
                bx, by = mu[u], ys[u]
            ff.fit_batch(bx.astype(np.float32), by.astype(int))
            replayed += LF.BS

    while t < ticks:
        if cfg["sleep"] and st.mu and t % w["day"] == w["day"] - w["night"]:
            n_sleep += 1
            ff.opt.lr = lr
            for _ in range(cfg["chunks"]):
                sleep_chunk()
                for _ in range(CHUNK_TICKS):
                    if t >= ticks:
                        break
                    a = world.arrive(t)
                    if a is not None:
                        x, c, y, ins = a
                        age = t - last_seen.get(c, -10 ** 9)
                        if c in last_seen and age >= LF.RET_GAP:
                            returning[c] = LF.RET_N
                        ret = int(returning.get(c, 0) > 0); returning[c] = returning.get(c, 0) - 1; last_seen[c] = t
                        if cfg.get("night_answer"):
                            s_, _ = RP.answer(st, x); f_ = int(ff.scores(x)[0].argmax())
                            rows.append((t, c, y, ret, 1, f_, s_, s_ if s_ >= 0 else f_, 1))
                            night_answered += 1
                        else:
                            rows.append((t, c, y, ret, 0, -1, -1, -1, 1))
                    t += 1
            tags.clear()
            forget()
            continue
        if t % 100 == 0:
            forget()
        a = world.arrive(t)
        if a is not None:
            x, c, y, ins = a
            age = t - last_seen.get(c, -10 ** 9)
            prev_seen = dict(last_seen)
            if c in last_seen and age >= LF.RET_GAP:
                returning[c] = LF.RET_N
            ret = int(returning.get(c, 0) > 0); returning[c] = returning.get(c, 0) - 1; last_seen[c] = t
            sc = ff.scores(x)[0]; o = np.sort(sc); f = int(sc.argmax()); ffm = float(o[-1] - o[-2])
            z = sc - sc.max(); ffbits.append((float((np.log(np.exp(z).sum()) - z[y]) / np.log(2)), ret))
            s, sm = RP.answer(st, x)
            q30 = np.quantile(ffms, 0.3) if len(ffms) == 500 else -np.inf
            dis = s >= 0 and s != f
            use = dis and ((cnt[f] >= 1000 and ffm < q30 and sm >= 0.2) or sm >= 0.3)
            if dis and not use and cfg.get("rw_n") and rw_left.get(s, 0) > 0:
                use = True; ra_used += 1; ra_right += int(s == y); ra_ffright += int(f == y)
            if dis and not use and cfg.get("rb") and t - prev_seen.get(s, -10 ** 9) >= cfg["ra"]:
                while rb_ev and t - rb_ev[0][0] > cfg["rb"]:
                    rb_ev.popleft()
                if any(k == s for _, k in rb_ev):
                    use = True; ra_used += 1; ra_right += int(s == y); ra_ffright += int(f == y)
                rb_ev.append((t, s))
            elif dis and not use and cfg.get("ra") and t - prev_seen.get(s, -10 ** 9) >= cfg["ra"]:
                qh = np.quantile(ffms, cfg["raq"]) if cfg.get("raq") and len(ffms) == 500 else np.inf
                if ffm < qh:
                    use = True; ra_used += 1; ra_right += int(s == y); ra_ffright += int(f == y)
            d = s if use else f
            if cfg.get("rw_n"):   # the label is seen only after answering
                if c in prev_seen and age >= cfg["rw_gap"]:
                    rw_left[c] = cfg["rw_n"]
                elif rw_left.get(c, 0) > 0:
                    rw_left[c] -= 1
            rows.append((t, c, y, ret, 1, f, s, d, 0))
            ffms.append(ffm); cnt[y] += 1
            if f != y:
                tags.append(x.copy())
            st.t = t; st.observe(x, y); st.add(x, y)
            inter = cfg.get("inter")
            hold = inter == "slot" and (n_wake + 1) % cfg["every"] == 0 and bool(st.mu)   # x joins this slot's replay batch
            if inter == "dense" and st.mu and rng.random() < cfg.get("pn", LF.BS) / cfg["every"]:
                mu, ys = protos(st); u = rng.integers(len(mu))
                ff.fit_batch(np.concatenate([x, mu[u:u + 1]]).astype(np.float32), np.array([y, int(ys[u])]))
                replayed += 1
            elif not hold:
                ff.opt.lr = lr * cfg["wl"]
                ff.fit_batch(x, np.array([y]))
                ff.opt.lr = lr
            n_wake += 1
            if cfg.get("awake") == "sim" and n_wake % cfg["every"] == 0 and st.mu:
                mu, ys = protos(st)
                keep = ys != y
                if keep.sum() >= 1:
                    mu, ys = mu[keep], ys[keep]
                    j = np.argsort(((mu - x.reshape(1, -1)) ** 2).sum(1))[:LF.BS]
                    j = j[rng.choice(len(j), LF.BS)]
                    ff.fit_batch(mu[j].astype(np.float32), ys[j].astype(int))
                    replayed += LF.BS
            fire = bool(cfg.get("every")) and n_wake % cfg["every"] == 0  # 18:20 fix: no-awake conds (none*) crashed after H11
            if cfg.get("gate"):
                b38.append(ffbits[-1][0]); rm = sum(b38) / len(b38)
                tok = min(tok + 1.0 / cfg["every"], cfg["cap"])
                fire = tok >= 1.0 and (len(rmh) < 500 or rm > np.median(rmh))
                rmh.append(rm)
                if fire:
                    tok -= 1.0; fired += 1
            if cfg.get("awake") == "sim":
                pass
            elif cfg.get("awake") == "mir" and fire and st.mu:
                mu, ys = protos(st)
                if mir_c is not None:
                    cx, cy, cb = mir_c
                    alive = np.isin(cy, list(st.mu))   # never replay what the store has already dropped
                    if alive.sum() >= 1:
                        cx, cy, cb = cx[alive], cy[alive], cb[alive]
                        inc = bits(cx, cy) - cb
                        j = np.argsort(-inc)[:LF.BS]
                        j = j[rng.choice(len(j), LF.BS)]
                        mir_inc.append((float(inc[np.argsort(-inc)[:LF.BS]].mean()), float(inc.mean())))
                        ff.fit_batch(cx[j].astype(np.float32), cy[j].astype(int))
                        replayed += LF.BS
                else:
                    u = rng.choice(len(mu), LF.BS)
                    ff.fit_batch(mu[u].astype(np.float32), ys[u].astype(int))
                    replayed += LF.BS
                u = rng.choice(len(mu), cfg["ncand"])
                mir_c = (mu[u].copy(), ys[u].astype(int), bits(mu[u], ys[u].astype(int)))
            elif cfg.get("awake") in ("uni", "remote") and fire and st.mu:
                pool = [k for k in st.mu if cfg["awake"] == "uni" or t - last_seen.get(k, -10 ** 9) >= cfg["gap"]]
                if pool:
                    mu = np.concatenate([st.mu[k] for k in pool])
                    ys = np.concatenate([np.full(len(st.mu[k]), k) for k in pool])
                    u = rng.choice(len(mu), LF.BS)
                    bx, by = mu[u], ys[u].astype(int)
                    if hold:
                        bx, by = np.concatenate([x, bx]), np.concatenate([[y], by])
                    ff.fit_batch(bx.astype(np.float32), by)
                    replayed += LF.BS
        t += 1

    R = np.array(rows)
    T, C, Y, RET, ANS, F, S, D, SLP = R.T
    sw = world.swap_at
    if os.environ.get("CLS_SAVE_ROWS"):
        os.makedirs(os.environ["CLS_SAVE_ROWS"], exist_ok=True)
        np.savez_compressed(os.path.join(os.environ["CLS_SAVE_ROWS"], f"{wk}_{cond}_{seed}.npz"), rows=R.astype(np.int32),
                            swap_at=sw, novel_at=world.novel_at)
    out = dict(condition=cond, world=wk, seed=seed, days=days, arrivals=len(R), missed=float(1 - ANS.mean()), sleeps=n_sleep,
               replay_per_wake=replayed / max(ANS.sum(), 1), tagged_share=tagged_replayed / max(replayed, 1),
               pre_swap_remaps=sum(tt < sw for tt, _, _ in st.log), classes_forgotten=forgot, night_answered=night_answered, ra_used=ra_used, ra_right=ra_right, ra_ffright=ra_ffright, gate_fired=fired,
               mir_rise=[float(np.mean([a for a, _ in mir_inc])), float(np.mean([b for _, b in mir_inc]))] if mir_inc else None, seconds=time.time() - t0)
    out.update(score(R, sw, world.novel_at, w))
    if ffbits:
        B = np.array(ffbits); h = len(B) // 2
        out["ff_bits"] = dict(all=float(B[:, 0].mean()), returns=float(B[B[:, 1] == 1, 0].mean()) if (B[:, 1] == 1).any() else float("nan"),
                              first_half=float(B[:h, 0].mean()), second_half=float(B[h:, 0].mean()))
    print(f"{wk} {cond:9s} seed={seed} " + " ".join(f"{k}={v['lifetime']*100:.2f}/{v['returns']*100:.1f}/{v['swap']*100:.1f}"
                                            for k, v in out["route"].items()) + f" missed={out['missed']*100:.2f} ({out['seconds']:.0f}s)", flush=True)
    return out


def score(R, sw, novel_at, w):
    """Route metrics under three scorings (rule from 1 Nov 2026 14:00 UTC+8: do not evaluate while asleep).
    route        (primary) awake arrivals only: arrivals during a sleep chunk are not scored, whether or not the
                 store answered them (`*_na`). Swap / novel = first 300 awake arrivals of those digits.
    route_all    the old score: every arrival, misses while asleep count as wrong, night answers count.
    route_match  the same tick mask for every machine, sleeper or not: drop the first 8 chunks (the longest night
                 tested) of every night, so all machines are scored on identical arrivals."""
    T, C, Y, RET, ANS, F, S, D, SLP = R.T
    ph = T % w["day"] - (w["day"] - w["night"])
    masks = dict(route=SLP == 0, route_all=np.ones(len(T), bool),
                 route_match=~((ph >= 0) & (ph < 8 * CHUNK_TICKS)))
    out = {}
    for key, E in masks.items():
        idx = np.flatnonzero(np.isin(C, list(w["swap"])) & (T >= sw) & E)[:300]
        nidx = np.flatnonzero(np.isin(C, list(w["novel"])) & (T >= novel_at) & E)[:300]
        out[key] = {}
        for name, P in (("combined", D), ("ff", F), ("store", S)):
            ok = (P == Y) & (ANS == 1)
            out[key][name] = dict(lifetime=float(ok[E].mean()), returns=float(ok[E & (RET == 1)].mean()),
                                  swap=float(ok[idx].mean()), novel=float(ok[nidx].mean()))
    out["asleep_share"] = float(SLP.mean())
    return out


def summarise(rows, conds, key="route"):
    M = lambda rs, r, k: 100 * np.mean([x[key][r][k] for x in rs])
    base = [x for x in rows if x["condition"] == "none"]
    out = ["| Machine | Lifetime | Returns | Swap first 300 | Novel first 300 | FF-only lifetime / returns / swap | Store-only lifetime | Asleep % | Replay ÷ wake | Δ lifetime | Δ returns | Δ swap |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for c in conds:
        rs = [x for x in rows if x["condition"] == c]
        if not rs:
            continue
        dl = M(rs, "combined", "lifetime") - M(base, "combined", "lifetime") if base else float("nan")
        dr = M(rs, "combined", "returns") - M(base, "combined", "returns") if base else float("nan")
        ds = M(rs, "combined", "swap") - M(base, "combined", "swap") if base else float("nan")
        out.append(f"| {c} | {M(rs,'combined','lifetime'):.2f} | {M(rs,'combined','returns'):.1f} | {M(rs,'combined','swap'):.1f} | "
                   f"{M(rs,'combined','novel'):.1f} | {M(rs,'ff','lifetime'):.2f} / {M(rs,'ff','returns'):.1f} / {M(rs,'ff','swap'):.1f} | "
                   f"{M(rs,'store','lifetime'):.2f} | {100*np.mean([x.get('asleep_share', x['missed']) for x in rs]):.2f} | {np.mean([x['replay_per_wake'] for x in rs]):.2f} | "
                   f"{dl:+.2f} | {dr:+.1f} | {ds:+.1f} |")
        if c != "none":
            verdict[c] = bool(dl >= -0.10 and (dr >= 1.0 or (ds >= 2.0 and dr >= -0.5)))
    return "\n".join(out), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--days", type=int, default=16)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="results/cls_sleep")
    ap.add_argument("--world", default="s49")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load_mnist("mnist")
    jobs = [(c, s, a.days, a.world) for c in a.conds for s in a.seeds]
    rows = []
    with Pool(a.jobs) as p:
        for r in p.imap_unordered(live, jobs):
            rows.append(r)
            json.dump(dict(world=WORLDS[a.world], conds=CONDS, rows=rows), open(os.path.join(a.out, "results.json"), "w"))
    verdict = {}
    for key in ("route", "route_match", "route_all"):
        md, verdict[key] = summarise(rows, a.conds, key)
        print(f"## {key}"); print(md); print(verdict[key])
    json.dump(dict(world=WORLDS[a.world], conds=CONDS, rows=rows, verdict=verdict["route"], verdict_by_scoring=verdict),
              open(os.path.join(a.out, "results.json"), "w"))


if __name__ == "__main__":
    main()
