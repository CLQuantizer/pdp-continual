"""M10-W6: curiosity on fresh World6 seeds, and a value of future progress.

Numbers only. No verdict. rand_dwell_s and prog_age_s are w6_headroom.run_life.
prog_fwd_s learns one shared linear value of the next whole-store visit reward.
"""
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CURIO_LR", "0.004")
    if os.environ.get("_M10_MAIN") != "1":
        os.environ["_M10_MAIN"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import math
import shutil
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

import curio
import h20_world
import m7_w3 as m7
import w4_headroom as w4
import w4_world
import w5_headroom as w5
import w6_headroom as w6
import w6_world
from w4_world import DARK, IDENTITIES, N_RING
from w5_world import finite_period
from w6_world import KINDS, World6, expected_birth_log

SEEDS = tuple(range(12, 24))
POLICIES = ("rand_dwell_s", "prog_age_s", "prog_fwd_s")
W6_POLICIES = ("rand_dwell_s", "prog_age_s")
PERIODS = (128, 512)
L_SCORE = 4096
N_CHECKPOINTS = 8
SLEEP_SHARE = 0.20
GAMMA = 0.5
ALPHA_NUM = 0.05
Y_CLIP = 10.0
N_FEAT = 8
FEAT_NAMES = (
    "bias",
    "r_last",
    "r_ema",
    "log_visits",
    "age64",
    "age256",
    "age1024",
    "travel",
)
AGE_SCALES = (64.0, 256.0, 1024.0)
REWARD_KINDS = tuple(list(h20_world.PROCS) + ["N", "Z"])
KERNEL_WALL_LIMIT_S = 90 * 60
KERNEL_WORKERS = 4
ROOT = Path(__file__).resolve().parent
CODE_FILES = w6.CODE_FILES + ("m10_w6.py",)
DATASET_ID = "ezzzzzx/pdp-code-m10w6"
GAPS = (
    "prog_age_s_minus_rand_dwell_s",
    "prog_fwd_s_minus_rand_dwell_s",
    "prog_fwd_s_minus_prog_age_s",
)
SMOKE_SEED = 0
SMOKE_PERIOD = 128
BITS_LIMIT = 1e-6
STORE_LIMIT = 1e-9
UPDATE_LIMIT = 1e-12

if curio.EPS != 0.1 or curio.BETA != 0.3 or m7.DWELL != 8 or w6.TAU != 256:
    raise RuntimeError("fixed controller constants drifted")
if w6.L_SCORE != L_SCORE or w6.SLEEP_SHARE != SLEEP_SHARE or tuple(w6.PERIODS) != PERIODS:
    raise RuntimeError("W6 score constants drifted")
if w4.N_CH != N_RING + 1 or N_RING != 18 or m7.DWELL != 8:
    raise RuntimeError("ring or dwell drifted")
if list(KINDS) != list(h20_world.PROCS) + ["N"] or "Z" in KINDS:
    raise RuntimeError("newborn kinds drifted")
if len(FEAT_NAMES) != N_FEAT or len(REWARD_KINDS) != 9:
    raise RuntimeError("feature or kind list drifted")
if "m9_w5.py" in CODE_FILES or ".env" in CODE_FILES:
    raise RuntimeError("dataset list holds a file it must not")


def period_name(period):
    return w6.period_name(period)


def checkpoints_of(L):
    return w6.checkpoints_of(L)


def _threads():
    w4._threads()


def expected_jobs():
    return [(policy, period) for period in PERIODS for policy in POLICIES]


def mean_std(values):
    return m7.mean_std(values)


def _median(values):
    if not values:
        return None
    return float(np.median(np.asarray(list(values), dtype=np.float64)))


def makespan(durations, workers):
    free = [0.0] * int(workers)
    for dur in durations:
        i = min(range(len(free)), key=lambda k: free[k])
        free[i] += float(dur)
    return max(free) if free else 0.0


def whole_exam_bits(life):
    """Sum of bits of every exam glance. Empty store is 0. Does not use cached_sum."""
    glances = life.exam.flat()
    if not glances:
        return 0.0
    t0 = time.perf_counter()
    values = m7.score_glances(life.cortex.model, life.cortex.channel_emb, glances)
    life.score_s += time.perf_counter() - t0
    return float(values.sum())


def partition_exam_bits(life):
    """Bits of each exam slot, in reservoir order, scored separately from the flat sum."""
    parts = []
    names = []
    t0 = time.perf_counter()
    for name in life.exam.order:
        names.append(name)
        glances = list(life.exam.slots[name])
        if not glances:
            parts.append(0.0)
            continue
        values = m7.score_glances(life.cortex.model, life.cortex.channel_emb, glances)
        parts.append(float(values.sum()))
    life.score_s += time.perf_counter() - t0
    flat_ids = [glance.gid for glance in life.exam.flat()]
    part_ids = [glance.gid for name in life.exam.order for glance in life.exam.slots[name]]
    if flat_ids != part_ids:
        raise RuntimeError("exam flat() is not the concatenation of the slots")
    return names, parts


def slot_features(room, r_last, r_ema, counts, last_end, t):
    """Features of all 18 slots at this decision. log is natural. Age uses last_end."""
    room = int(room)
    t = int(t)
    xs = np.zeros((N_RING, N_FEAT), dtype=np.float64)
    for j in range(N_RING):
        age = float(t - int(last_end[j]))
        if age < 0.0:
            raise RuntimeError(f"slot {j} age is negative")
        visits = int(counts[j])
        travel = len(w4_world.shortest(room, j, N_RING)) / float(m7.DWELL)
        xs[j, 0] = 1.0
        if visits == 0:
            xs[j, 1] = 0.0
            xs[j, 2] = 0.0
        else:
            xs[j, 1] = float(r_last[j])
            xs[j, 2] = float(r_ema[j])
        xs[j, 3] = math.log(1.0 + visits)
        xs[j, 4] = 1.0 - math.exp(-age / AGE_SCALES[0])
        xs[j, 5] = 1.0 - math.exp(-age / AGE_SCALES[1])
        xs[j, 6] = 1.0 - math.exp(-age / AGE_SCALES[2])
        xs[j, 7] = float(travel)
    return xs


def choose_slot(room, unvisited, w, r_last, r_ema, counts, last_end, t, rng):
    """Optimistic first pass, then epsilon, then argmax. First pass draws no coin.

    The greedy tie break draws one rng.random() per arm, in slot order, then max.
    """
    xs = slot_features(room, r_last, r_ema, counts, last_end, t)
    if unvisited:
        target = int(m7.nearest_unvisited(int(room), unvisited, n=N_RING))
        unvisited.remove(target)
        mode = "first"
    elif float(rng.random()) < float(curio.EPS):
        target = int(rng.integers(N_RING))
        mode = "eps"
    else:
        q = xs @ w
        keys = [(float(q[k]), float(rng.random())) for k in range(N_RING)]
        target = max(range(N_RING), key=lambda k: keys[k])
        mode = "greedy"
    if int(target) < 0 or int(target) >= N_RING:
        raise RuntimeError("choice slot is off the ring")
    return int(target), mode, xs[int(target)].copy()


def apply_td(w, x_i, r, max_q):
    """One normalised LMS step. Clips the target y, not the reward and not w."""
    x_i = np.asarray(x_i, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x_i.shape != (N_FEAT,) or w.shape != (N_FEAT,):
        raise RuntimeError("weight or feature shape drifted")
    q_before = float(np.dot(w, x_i))
    y_raw = float(r) + GAMMA * float(max_q)
    y = float(np.clip(y_raw, -Y_CLIP, Y_CLIP))
    alpha = ALPHA_NUM / (1.0 + float(np.dot(x_i, x_i)))
    step = alpha * (y - q_before)
    w_new = w + step * x_i
    for i in range(N_FEAT):
        piece = float(w[i]) + step * float(x_i[i])
        if abs(piece - float(w_new[i])) > UPDATE_LIMIT:
            raise RuntimeError("TD update does not match the formula")
    q_after = float(np.dot(w_new, x_i))
    if not math.isfinite(q_after) or not np.all(np.isfinite(w_new)):
        raise RuntimeError("TD update is not finite")
    return w_new, {
        "r": float(r),
        "max_q": float(max_q),
        "y_raw": float(y_raw),
        "y": float(y),
        "q_before": float(q_before),
        "q_after": float(q_after),
        "alpha": float(alpha),
        "x_i": [float(v) for v in x_i],
        "w_before": [float(v) for v in w],
        "w_after": [float(v) for v in w_new],
    }


def verify_hand(rec):
    """Recompute one logged visit from its numbers. Returns the max abs weight error."""
    x_i = np.asarray(rec["x_i"], dtype=np.float64)
    w0 = np.asarray(rec["w_before"], dtype=np.float64)
    w1 = np.asarray(rec["w_after"], dtype=np.float64)
    q_before = float(np.dot(w0, x_i))
    if abs(q_before - float(rec["q_before"])) > UPDATE_LIMIT:
        raise RuntimeError("Q before is not w·x at choice time")
    y_raw = float(rec["r"]) + GAMMA * float(rec["max_q"])
    y = float(np.clip(y_raw, -Y_CLIP, Y_CLIP))
    if abs(y_raw - float(rec["y_raw"])) > UPDATE_LIMIT or abs(y - float(rec["y"])) > UPDATE_LIMIT:
        raise RuntimeError("y is not clip(r + gamma max Q, -10, 10)")
    alpha = ALPHA_NUM / (1.0 + float(np.dot(x_i, x_i)))
    if abs(alpha - float(rec["alpha"])) > UPDATE_LIMIT:
        raise RuntimeError("alpha is not 0.05 / (1 + ||x||^2)")
    expect = w0 + alpha * (y - q_before) * x_i
    err = float(np.max(np.abs(expect - w1)))
    if err > UPDATE_LIMIT:
        raise RuntimeError(f"w update error {err}")
    q_after = float(np.dot(w1, x_i))
    if abs(q_after - float(rec["q_after"])) > UPDATE_LIMIT:
        raise RuntimeError("Q after is not the updated w·x")
    return err


def _freeze_band(band):
    return (
        tuple(float(v) for v in band["r_last"]),
        tuple(float(v) for v in band["r_ema"]),
        tuple(int(v) for v in band["counts"]),
        tuple(int(v) for v in band["last_end"]),
        tuple(float(v) for v in band["w"]),
        frozenset(int(v) for v in band["unvisited"]),
        int(band["n_choices"]),
        int(band["n_updates"]),
    )


def run_fwd(seed, L, period, trace=False, audit_store=False):
    """prog_fwd_s. Same life as W6, with a linear TD value on the whole exam store."""
    L = int(L)
    period = finite_period(period)
    seed = int(seed)
    if period is None:
        raise ValueError("prog_fwd_s requires a birth period")
    n_sleep = w4.sleep_decisions(L)
    plan = m7.sleep_plan(L, n_sleep)
    if any(length != m7.DWELL for _, length in plan):
        raise RuntimeError("a fixed sleep visit is not 8 decisions")
    points = checkpoints_of(L)
    point_set = set(points)
    world = World6(seed, period)
    initial_rooms = list(world.rooms)
    names = list(IDENTITIES)
    index = {name: i for i, name in enumerate(names)}
    t0 = time.perf_counter()
    life = w6.OpenLife(seed, world, names, index, N_RING, w4.N_CH, list(w4_world.LEARN14))
    rng = np.random.default_rng(1_000 + seed)
    state = {"room": 0, "plan": [], "target": None, "arrived": True, "visit": None}
    band = {
        "r_last": [0.0] * N_RING,
        "r_ema": [0.0] * N_RING,
        "counts": [0] * N_RING,
        "last_end": [0] * N_RING,
        "unvisited": set(range(N_RING)),
        "w": np.zeros(N_FEAT, dtype=np.float64),
        "n_choices": 0,
        "n_first": 0,
        "n_eps": 0,
        "n_greedy": 0,
        "n_updates": 0,
        "trace": [],
        "audits": [],
    }
    reward_sum = {kind: 0.0 for kind in REWARD_KINDS}
    reward_n = {kind: 0 for kind in REWARD_KINDS}
    records = []
    clock = {"eval": 0.0}
    prev_cp = {"t": 0}
    visit_pending = None
    sleep_marks = []
    slot_counts = [0] * N_RING
    noise_count = 0
    dark_count = 0
    corridor_count = 0
    learnable_stays = 0
    stays = []
    si = 0
    t = 0
    checked = False

    def on_decision(done):
        world.maybe_birth(done)
        if done in point_set:
            rec = w6.take_checkpoint(life, done, prev_cp["t"], clock)
            prev_cp["t"] = done
            records.append(rec)
            print(
                f"s{seed} prog_fwd_s p={period_name(period)} checkpoint t={done}/{L} "
                f"open {rec['open']:.4f} n {rec['n']}",
                flush=True,
            )

    def start_visit():
        if state["room"] != world.room:
            raise RuntimeError("state room and world room diverged")
        if state["plan"]:
            raise RuntimeError("a visit started while a plan was still running")
        stored_counts = list(band["counts"])
        target, mode, x_i = choose_slot(
            state["room"], band["unvisited"], band["w"], band["r_last"], band["r_ema"],
            band["counts"], band["last_end"], t, rng,
        )
        if band["counts"] != stored_counts:
            raise RuntimeError("the choice changed visit counts")
        if abs(float(x_i[3]) - math.log(1.0 + stored_counts[target])) > UPDATE_LIMIT:
            raise RuntimeError("log visits was not the count at choice time")
        if stored_counts[target] == 0 and (float(x_i[1]) != 0.0 or float(x_i[2]) != 0.0):
            raise RuntimeError("an unvisited slot has a stored reward feature")
        band["n_choices"] += 1
        band[{"first": "n_first", "eps": "n_eps", "greedy": "n_greedy"}[mode]] += 1
        state["plan"] = w4_world.shortest(state["room"], target, N_RING) + ["stay"] * m7.DWELL
        state["target"] = int(target)
        state["arrived"] = False
        bits_start = whole_exam_bits(life)
        visit = {
            "open": True,
            "start_t": int(t),
            "travel": 0,
            "dwell": 0,
            "dwell_name": None,
            "target": int(target),
            "mode": mode,
            "x_i": x_i,
            "bits_start": float(bits_start),
            "visits_before": int(stored_counts[target]),
        }
        if audit_store and len(band["audits"]) < 3:
            _names, parts = partition_exam_bits(life)
            gap = abs(float(sum(parts)) - float(bits_start))
            if gap > STORE_LIMIT:
                raise RuntimeError(f"visit-start partition gap {gap}")
            visit["slot_start"] = parts
        state["visit"] = visit

    def finish_visit():
        info = state["visit"]
        if info is None or not info["open"]:
            return
        n_dec = int(info["travel"] + info["dwell"])
        if n_dec <= 0:
            raise RuntimeError("empty room visit")
        if n_dec != int(t) - int(info["start_t"]):
            raise RuntimeError("visit length drifted from the clock")
        if info["dwell"] and info["dwell_name"] != world.rooms[info["target"]]:
            raise RuntimeError("dwell channel is not the chosen slot")
        if int(band["counts"][info["target"]]) != int(info["visits_before"]):
            raise RuntimeError("visit count changed during the visit")
        bits_end = whole_exam_bits(life)
        decrease = float(info["bits_start"] - bits_end)
        reward = decrease / float(n_dec)
        if not math.isfinite(reward):
            raise RuntimeError("nonfinite room reward")
        if "slot_start" in info:
            _names, parts_end = partition_exam_bits(life)
            slot_start_sum = float(sum(info["slot_start"]))
            slot_end_sum = float(sum(parts_end))
            gap_end = abs(slot_end_sum - float(bits_end))
            gap_dec = abs((slot_start_sum - slot_end_sum) - decrease)
            audit = {
                "visit": int(band["n_updates"] + 1),
                "n_dec": int(n_dec),
                "whole_start": float(info["bits_start"]),
                "whole_end": float(bits_end),
                "slot_start": slot_start_sum,
                "slot_end": slot_end_sum,
                "whole_decrease": float(decrease),
                "slot_decrease": float(slot_start_sum - slot_end_sum),
                "abs_gap": float(gap_dec),
                "abs_end_gap": float(gap_end),
            }
            print(
                f"AUDIT visit {audit['visit']} n_dec {n_dec} "
                f"whole_decrease {decrease:.12g} slot_decrease {audit['slot_decrease']:.12g} "
                f"abs_gap {gap_dec:.3e} abs_end_gap {gap_end:.3e}",
                flush=True,
            )
            if gap_end > STORE_LIMIT or gap_dec > STORE_LIMIT:
                raise RuntimeError(
                    f"whole-store decrease does not match the per-slot sum "
                    f"(gap {gap_dec}, end gap {gap_end})"
                )
            band["audits"].append(audit)
        target = int(info["target"])
        band["r_last"][target] = float(reward)
        m7.ema(band["r_ema"], band["counts"], target, float(reward))
        band["last_end"][target] = int(t)
        if int(t) - int(band["last_end"][target]) != 0:
            raise RuntimeError("just-visited slot is not age 0")
        if state["room"] != world.room:
            raise RuntimeError("state room and world room diverged")
        xs = slot_features(
            state["room"], band["r_last"], band["r_ema"], band["counts"], band["last_end"], t,
        )
        if xs[target, 4] != 0.0 or xs[target, 5] != 0.0 or xs[target, 6] != 0.0:
            raise RuntimeError("age features of the finished slot are not 0")
        if abs(float(xs[target, 3]) - math.log(1.0 + int(band["counts"][target]))) > UPDATE_LIMIT:
            raise RuntimeError("bootstrap log visits missed the visit that just finished")
        q = xs @ band["w"]
        max_q = float(np.max(q))
        w_new, rec = apply_td(band["w"], info["x_i"], reward, max_q)
        band["w"] = w_new
        band["n_updates"] += 1
        kind = world.kind[target]
        if kind not in reward_n:
            raise RuntimeError(f"visit kind {kind}")
        reward_sum[kind] += float(reward)
        reward_n[kind] += 1
        info["open"] = False
        if trace and band["n_updates"] <= 12:
            rec["visit"] = int(band["n_updates"])
            rec["slot"] = int(target)
            rec["mode"] = info["mode"]
            rec["n_dec"] = int(n_dec)
            rec["decrease"] = float(decrease)
            rec["kind"] = kind
            band["trace"].append(rec)
            w_txt = " ".join(f"{v:.8f}" for v in rec["w_after"])
            print(
                f"FWD visit {rec['visit']} slot {rec['slot']} mode {rec['mode']} "
                f"r {rec['r']:.10f} y {rec['y']:.10f} "
                f"Q_before {rec['q_before']:.10f} Q_after {rec['q_after']:.10f} w {w_txt}",
                flush=True,
            )

    while t < L:
        if not state["plan"] and w4.sleep_due(t, L, plan, si, N_RING):
            if state["visit"] is not None and state["visit"]["open"]:
                raise RuntimeError("sleep started inside a room visit")
            visit_pending = w4._close_visit(life, visit_pending)
            frozen = _freeze_band(band)
            _n, t, checked = w5.sleep_visit(
                life, seed, "prog_fwd_s", period, sleep_marks, plan[si][1], t, L,
                False, checked, on_decision,
            )
            if _freeze_band(band) != frozen:
                raise RuntimeError("a sleep visit changed the bandit state")
            si += 1
            continue
        if not state["plan"]:
            start_visit()
        action = state["plan"].pop(0)
        if action == "stay" and not state["arrived"]:
            if state["room"] != state["target"]:
                raise RuntimeError(
                    f"arrival slot {state['room']} is not target {state['target']}"
                )
            state["arrived"] = True
        learnable_stay = action == "stay" and world.is_learnable(state["room"])
        room, channel, symbols = world.step(action, rng)
        state["room"] = room
        life.count_preq = learnable_stay
        try:
            if channel == "corridor":
                life.corridor_decision(symbols)
                state["visit"]["travel"] += 1
            else:
                life.room_decision(symbols, channel)
                state["visit"]["dwell"] += 1
                state["visit"]["dwell_name"] = channel
                visit_pending = channel
        finally:
            life.count_preq = False
        if channel == "corridor":
            if learnable_stay:
                raise RuntimeError("a corridor counted as a learnable stay")
            corridor_count += 1
        else:
            if channel != world.rooms[room]:
                raise RuntimeError("stay channel is not the slot's initial room")
            slot_counts[room] += 1
            kind_now = world.kind[room]
            if kind_now == "N":
                noise_count += 1
            elif kind_now == "Z":
                dark_count += 1
            elif learnable_stay:
                pass
            else:
                raise RuntimeError(f"stay kind {kind_now} was not counted")
            if learnable_stay and kind_now not in w4_world.PROC_SET:
                raise RuntimeError("learnable stay flag disagrees with the kind")
            if not learnable_stay and kind_now in w4_world.PROC_SET:
                raise RuntimeError("a learnable stay was not marked")
            stays.append((t + 1, int(room)))
            if learnable_stay:
                learnable_stays += 1
        t += 1
        on_decision(t)
        if t % 256 == 0:
            print(f"s{seed} prog_fwd_s p={period_name(period)} t={t}/{L}", flush=True)
        if not state["plan"]:
            visit_pending = w4._close_visit(life, visit_pending)
            finish_visit()
    finish_visit()
    if visit_pending is not None:
        life.snapshot(visit_pending)
    if si != len(plan):
        raise RuntimeError(f"prog_fwd_s took {si} of {len(plan)} planned sleep visits")
    if len(sleep_marks) != n_sleep:
        raise RuntimeError("sleep decisions do not match the dose")
    if life.exam_drawn != 0:
        raise RuntimeError(f"sleep drew {life.exam_drawn} exam glances")
    if world.rooms != initial_rooms:
        raise RuntimeError("a birth changed the channel name")
    z_slot = world.rooms.index(DARK)
    if world.kind[z_slot] != "Z" or world.source[z_slot] is not None:
        raise RuntimeError("the dark room changed")
    if world.birth_log != expected_birth_log(seed, period, L):
        raise RuntimeError("birth log does not match the independent draw")
    decisions_taken = sum(slot_counts) + corridor_count + n_sleep
    if decisions_taken != L:
        raise RuntimeError(f"prog_fwd_s ran {decisions_taken} decisions")
    if sum(slot_counts) != len(stays):
        raise RuntimeError("stay log length drifted")
    kind_sum = noise_count + dark_count + learnable_stays
    if kind_sum != sum(slot_counts):
        raise RuntimeError("slot stays were not partitioned into noise, dark and learnable")
    if life.preq_symbols != learnable_stays * curio.B:
        raise RuntimeError(
            f"prequential symbols {life.preq_symbols} != {learnable_stays} glances"
        )
    if band["n_updates"] != band["n_choices"]:
        raise RuntimeError("TD updates do not match room choices")
    if band["n_first"] + band["n_eps"] + band["n_greedy"] != band["n_choices"]:
        raise RuntimeError("choice modes do not add up")
    if band["n_first"] != min(band["n_choices"], N_RING):
        raise RuntimeError("the optimistic first pass did not cover the unvisited slots")
    if len(band["unvisited"]) != N_RING - band["n_first"]:
        raise RuntimeError("unvisited set size drifted")
    slot_share = [count / L for count in slot_counts]
    noise_share = noise_count / L
    corridor_share = corridor_count / L
    sleep_share = n_sleep / L
    if abs((sum(slot_share) + corridor_share + sleep_share) - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    if [rec["decision"] for rec in records] != points:
        raise RuntimeError("checkpoints are not L/8, 2L/8, …, L")
    t_open = float(np.mean([rec["open"] for rec in records]))
    final = float(records[-1]["open"])
    preq_bps = None
    if life.preq_symbols:
        preq_bps = float(life.preq_bits / life.preq_symbols)
    finite = bool(
        math.isfinite(t_open) and math.isfinite(final)
        and all(math.isfinite(rec["open"]) for rec in records)
        and all(math.isfinite(room["bits"]) for rec in records for room in rec["rooms"])
        and (preq_bps is None or math.isfinite(preq_bps))
        and bool(np.all(np.isfinite(band["w"])))
    )
    by_kind = {}
    for kind in REWARD_KINDS:
        n = int(reward_n[kind])
        total = float(reward_sum[kind])
        by_kind[kind] = {
            "n": n,
            "sum": total,
            "mean": None if n == 0 else float(total / n),
        }
    wall = time.perf_counter() - t0
    return {
        "seed": seed,
        "policy": "prog_fwd_s",
        "period": int(period),
        "L": int(L),
        "T_open": float(t_open),
        "final": float(final),
        "checkpoints": records,
        "birth_log": [
            {
                "decision": int(event["decision"]),
                "slot": int(event["slot"]),
                "dead_kind": event["dead_kind"],
                "newborn_kind": event["newborn_kind"],
                "k": int(event["k"]),
            }
            for event in world.birth_log
        ],
        "latency": w6.latencies_of(stays, world.birth_log, L),
        "slot_share": [float(v) for v in slot_share],
        "noise_share": float(noise_share),
        "dark_share": float(dark_count / L),
        "corridor_share": float(corridor_share),
        "sleep_share": float(sleep_share),
        "sleep_decisions": int(n_sleep),
        "room_order": list(world.rooms),
        "preq_bits": float(life.preq_bits),
        "preq_symbols": int(life.preq_symbols),
        "preq_bps": None if preq_bps is None else float(preq_bps),
        "learnable_stays": int(learnable_stays),
        "overlap": int(len(life.exam.ids & life.replay.ids)),
        "exam_glances_drawn": int(life.exam_drawn),
        "wall_s": float(wall),
        "eval_s": float(clock["eval"]),
        "learn_s": float(life.learn_s),
        "score_s": float(life.score_s),
        "finite": bool(finite),
        "n_ch": int(w4.N_CH),
        "tau": None,
        "n_choices": int(band["n_choices"]),
        "n_first": int(band["n_first"]),
        "n_eps": int(band["n_eps"]),
        "n_greedy": int(band["n_greedy"]),
        "ema_updates": int(band["n_updates"]),
        "n_updates": int(band["n_updates"]),
        "w": [float(v) for v in band["w"]],
        "w_names": list(FEAT_NAMES),
        "gamma": float(GAMMA),
        "alpha_num": float(ALPHA_NUM),
        "y_clip": float(Y_CLIP),
        "choice_seed": int(1_000 + seed),
        "visit_reward": by_kind,
        "trace": list(band["trace"]),
        "audits": list(band["audits"]),
    }


def run_life(seed, policy, L, period, score=True, trace=False, audit_store=False):
    """Worker entry. The two W6 controllers are w6_headroom.run_life, unchanged."""
    if not score:
        raise ValueError("m10 lives are scored")
    if policy in W6_POLICIES:
        if trace or audit_store:
            raise ValueError("trace and the store audit are only for prog_fwd_s")
        return w6.run_life(seed, policy, L, period, score=True)
    if policy == "prog_fwd_s":
        return run_fwd(seed, L, period, trace=trace, audit_store=audit_store)
    raise ValueError(policy)


def _check_row(row, seed, policy, period, L):
    period = finite_period(period)
    if row["seed"] != int(seed) or row["policy"] != policy or row["L"] != int(L):
        raise RuntimeError("row does not match its job")
    if finite_period(row["period"]) != period:
        raise RuntimeError("row period does not match its job")
    if not row["finite"]:
        raise RuntimeError(f"{policy} seed {seed} period {period_name(period)} is not finite")
    if row["sleep_decisions"] != w4.sleep_decisions(L):
        raise RuntimeError("sleep dose drifted")
    if row["overlap"] != 0 or row["exam_glances_drawn"] != 0:
        raise RuntimeError("stores overlapped")
    if row["n_ch"] != w4.N_CH:
        raise RuntimeError("channel width drifted")
    if abs(row["sleep_share"] - (w4.sleep_decisions(L) / float(L))) > 1e-12:
        raise RuntimeError("sleep share drifted")
    if len(row["slot_share"]) != N_RING:
        raise RuntimeError("slot shares are not 18")
    total = sum(row["slot_share"]) + row["corridor_share"] + row["sleep_share"]
    if abs(total - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    if row["preq_symbols"] != row["learnable_stays"] * curio.B:
        raise RuntimeError("prequential symbols drifted")
    if row["preq_symbols"] and abs(row["preq_bps"] - row["preq_bits"] / row["preq_symbols"]) > 1e-9:
        raise RuntimeError("prequential bits/symbol drifted")
    points = checkpoints_of(L)
    if [rec["decision"] for rec in row["checkpoints"]] != points:
        raise RuntimeError("checkpoint decisions drifted")
    if abs(row["final"] - row["checkpoints"][-1]["open"]) > 1e-12:
        raise RuntimeError("final score is not the last checkpoint")
    if abs(row["T_open"] - float(np.mean([rec["open"] for rec in row["checkpoints"]]))) > 1e-12:
        raise RuntimeError("T_open is not the mean of the checkpoints")
    for rec in row["checkpoints"]:
        if rec["n"] != len(rec["rooms"]) or rec["n"] < 1:
            raise RuntimeError("checkpoint room count drifted")
        if abs(rec["open"] - float(np.mean([room["bits"] for room in rec["rooms"]]))) > 1e-12:
            raise RuntimeError("checkpoint score is not the mean of its learnable slots")
        for room in rec["rooms"]:
            if room["kind"] not in h20_world.PROCS:
                raise RuntimeError("a scored slot is not learnable")
    if row["birth_log"] != expected_birth_log(seed, period, L):
        raise RuntimeError("birth log does not match the independent draw")
    if len(row["latency"]) != len(row["birth_log"]):
        raise RuntimeError("latency rows do not match births")
    if policy == "prog_age_s":
        if row["tau"] != w6.TAU or row["n_choices"] != row["ema_updates"]:
            raise RuntimeError("prog_age_s bookkeeping drifted")
    elif policy == "prog_fwd_s":
        if row["tau"] is not None:
            raise RuntimeError("prog_fwd_s stored an ageing tau")
        if list(row["w_names"]) != list(FEAT_NAMES) or len(row["w"]) != N_FEAT:
            raise RuntimeError("prog_fwd_s weight layout drifted")
        if any(not math.isfinite(float(v)) for v in row["w"]):
            raise RuntimeError("prog_fwd_s weight is not finite")
        if row["n_choices"] != row["n_updates"] or row["ema_updates"] != row["n_updates"]:
            raise RuntimeError("prog_fwd_s updates do not match choices")
        if row["n_first"] + row["n_eps"] + row["n_greedy"] != row["n_choices"]:
            raise RuntimeError("prog_fwd_s choice modes do not add up")
        if row["gamma"] != GAMMA or row["alpha_num"] != ALPHA_NUM or row["y_clip"] != Y_CLIP:
            raise RuntimeError("prog_fwd_s learning constants drifted")
        if int(row["choice_seed"]) != 1_000 + int(seed):
            raise RuntimeError("prog_fwd_s choice seed drifted")
        kinds = list(row["visit_reward"])
        if kinds != list(REWARD_KINDS):
            raise RuntimeError("visit reward kinds drifted")
        n_visits = 0
        for kind in REWARD_KINDS:
            item = row["visit_reward"][kind]
            n_visits += int(item["n"])
            if item["n"] == 0 and item["mean"] is not None:
                raise RuntimeError("empty kind has a mean reward")
            if item["n"] and abs(item["mean"] - item["sum"] / item["n"]) > 1e-9:
                raise RuntimeError("kind mean is not sum/n")
        if n_visits != int(row["n_choices"]):
            raise RuntimeError("visit rewards do not match choices")
    elif policy == "rand_dwell_s":
        if row["tau"] is not None or row["n_choices"] is not None:
            raise RuntimeError("rand_dwell_s stored bandit state")
    else:
        raise RuntimeError(f"unknown policy {policy}")


def _score_delta(left, right):
    deltas = [
        abs(float(left["T_open"]) - float(right["T_open"])),
        abs(float(left["final"]) - float(right["final"])),
    ]
    if len(left["checkpoints"]) != len(right["checkpoints"]):
        raise RuntimeError("checkpoint counts differ")
    for a, b in zip(left["checkpoints"], right["checkpoints"]):
        if a["decision"] != b["decision"]:
            raise RuntimeError("checkpoint decisions differ")
        deltas.append(abs(float(a["open"]) - float(b["open"])))
    return float(max(deltas))


def _job(args):
    seed, policy, period, L = args
    _threads()
    print(
        f"s{seed} {policy} p={period_name(period)} L={L} start dose {w4.sleep_decisions(L)}",
        flush=True,
    )
    row = run_life(seed, policy, L, period, score=True)
    _check_row(row, seed, policy, period, L)
    # The kernel file does not need the smoke trace.
    row.pop("trace", None)
    row.pop("audits", None)
    preq = "none" if row["preq_bps"] is None else f"{row['preq_bps']:.4f}"
    print(
        f"s{seed} {policy} p={period_name(period)} L={L} T_open {row['T_open']:.4f} "
        f"final {row['final']:.4f} noise {row['noise_share']:.4f} "
        f"preq {preq} sleep {row['sleep_share']:.4f} "
        f"corridor {row['corridor_share']:.3f} wall {row['wall_s']:.1f}s "
        f"learn {row['learn_s']:.1f}s score {row['score_s']:.1f}s eval {row['eval_s']:.1f}s",
        flush=True,
    )
    return row


def kernel_main(seed, out, workers=4, kernel_id=None, jobs=None):
    if __name__ != "m10_w6":
        raise SystemExit("import m10_w6 and call kernel_main so the pool can spawn workers")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 12-23")
    if int(workers) < 1 or int(workers) > KERNEL_WORKERS:
        raise SystemExit("kernel pool is at most 4")
    if jobs is None:
        jobs = expected_jobs()
    jobs = tuple((str(policy), finite_period(period)) for policy, period in jobs)
    allowed = set(expected_jobs())
    if any(job not in allowed for job in jobs) or len(jobs) != len(set(jobs)) or len(jobs) != len(allowed):
        raise SystemExit(f"jobs {jobs} are not an M10 kernel")
    _threads()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"kernel {kernel_id} seed {seed} workers {workers} jobs {len(jobs)}",
        flush=True,
    )
    t0 = time.perf_counter()
    ctx = get_context("spawn")
    with ctx.Pool(int(workers)) as pool:
        asyncs = [
            pool.apply_async(_job, ((seed, policy, period, L_SCORE),))
            for policy, period in jobs
        ]
        rows = [item.get() for item in asyncs]
    wall = time.perf_counter() - t0
    if len(rows) != len(jobs):
        raise SystemExit(f"expected {len(jobs)} rows, got {len(rows)}")
    for row, (policy, period) in zip(rows, jobs):
        _check_row(row, seed, policy, period, L_SCORE)
    by_key = {(row["policy"], finite_period(row["period"])): row for row in rows}
    if len(by_key) != len(jobs):
        raise SystemExit("duplicate policy rows")
    ordered = [by_key[job] for job in jobs]
    payload = {
        "kernel_id": kernel_id,
        "seed": seed,
        "jobs": [
            {"policy": policy, "period": int(period)}
            for policy, period in jobs
        ],
        "workers": int(workers),
        "wall_s": wall,
        "torch": torch.__version__,
        "model_seed": m7.MODEL_SEED,
        "n_ch": w4.N_CH,
        "L": L_SCORE,
        "sleep_share_target": SLEEP_SHARE,
        "room_order": ordered[0]["room_order"],
        "runs": ordered,
    }
    path = out / f"m10w6_s{seed}.json"
    m7.dump(path, payload)
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    return payload


def _linear_scale(row, src_L):
    scale = float(L_SCORE) / float(src_L)
    wall = float(row["wall_s"])
    eval_s = float(row["eval_s"])
    learn_s = float(row["learn_s"])
    score_s = float(row["score_s"])
    other = max(0.0, wall - eval_s - learn_s - score_s)
    linear = wall * scale
    # Eight checkpoints at every L, so the measured eval does not grow with L.
    component = eval_s + (learn_s + score_s + other) * scale
    return {
        "src_L": int(src_L),
        "wall_s": wall,
        "eval_s": eval_s,
        "learn_s": learn_s,
        "score_s": score_s,
        "linear_s": float(linear),
        "component_s": float(component),
    }


def smoke(out_dir=None):
    """VPS checks, one process. A failed check does not stage kernels."""
    _threads()
    out_dir = Path(out_dir or (ROOT / "results" / "m10w6"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    t0 = time.perf_counter()
    payload = {
        "torch": torch.__version__,
        "model_seed": m7.MODEL_SEED,
        "n_ch": w4.N_CH,
        "ok": False,
    }
    try:
        log(
            "smoke check 1: rand_dwell_s and prog_age_s, seed 0, period 128, L=512, "
            "against w6_headroom.run_life"
        )
        walls_512 = {}
        for policy in W6_POLICIES:
            direct = w6.run_life(SMOKE_SEED, policy, 512, SMOKE_PERIOD, score=True)
            wrapped = run_life(SMOKE_SEED, policy, 512, SMOKE_PERIOD, score=True)
            _check_row(direct, SMOKE_SEED, policy, SMOKE_PERIOD, 512)
            _check_row(wrapped, SMOKE_SEED, policy, SMOKE_PERIOD, 512)
            delta = _score_delta(direct, wrapped)
            same_birth = direct["birth_log"] == wrapped["birth_log"]
            same_counts = (
                direct["learnable_stays"] == wrapped["learnable_stays"]
                and direct["sleep_decisions"] == wrapped["sleep_decisions"]
                and direct["preq_symbols"] == wrapped["preq_symbols"]
            )
            preq_gap = abs(float(direct["preq_bits"]) - float(wrapped["preq_bits"]))
            log(
                f"  {policy} direct T_open {direct['T_open']:.10f} "
                f"wrapper T_open {wrapped['T_open']:.10f} "
                f"max_abs {delta:.3e} preq_gap {preq_gap:.3e} "
                f"birth_match {int(same_birth)} counts_match {int(same_counts)} "
                f"finite {wrapped['finite']} wall_wrapper {wrapped['wall_s']:.2f}s "
                f"learn {wrapped['learn_s']:.2f}s score {wrapped['score_s']:.2f}s "
                f"eval {wrapped['eval_s']:.2f}s"
            )
            if delta > BITS_LIMIT or preq_gap > BITS_LIMIT or not same_birth or not same_counts:
                raise RuntimeError(f"{policy} does not reproduce W6 to {BITS_LIMIT}")
            if not wrapped["finite"] or not direct["finite"]:
                raise RuntimeError(f"{policy} L=512 is not finite")
            walls_512[policy] = wrapped

        log("smoke check 2: prog_fwd_s seed 0 period 128 L=512, first 12 visits")
        fwd = run_life(
            SMOKE_SEED, "prog_fwd_s", 512, SMOKE_PERIOD,
            score=True, trace=True, audit_store=True,
        )
        _check_row(fwd, SMOKE_SEED, "prog_fwd_s", SMOKE_PERIOD, 512)
        walls_512["prog_fwd_s"] = fwd
        trace = fwd["trace"]
        if len(trace) != 12:
            raise RuntimeError(f"expected 12 traced visits, got {len(trace)}")
        errs = []
        for rec in trace:
            if rec["mode"] != "first":
                raise RuntimeError(f"visit {rec['visit']} mode {rec['mode']} is inside the first pass")
            err = verify_hand(rec)
            errs.append(err)
            log(
                f"  FWD visit {rec['visit']} slot {rec['slot']} mode {rec['mode']} "
                f"r {rec['r']:.10f} y {rec['y']:.10f} "
                f"Q_before {rec['q_before']:.10f} Q_after {rec['q_after']:.10f} "
                f"w {' '.join(format(v, '.8f') for v in rec['w_after'])}"
            )
        hand = trace[11]
        err = verify_hand(hand)
        log(f"HAND visit {hand['visit']} slot {hand['slot']} mode {hand['mode']}")
        log(f"  x_i {[repr(float(v)) for v in hand['x_i']]}")
        log(f"  r {repr(float(hand['r']))}")
        log(f"  max_q {repr(float(hand['max_q']))}")
        log(f"  y_raw {repr(float(hand['y_raw']))}")
        log(f"  y {repr(float(hand['y']))}")
        log(f"  Q_before {repr(float(hand['q_before']))}")
        log(f"  alpha {repr(float(hand['alpha']))}")
        log(f"  w_before {[repr(float(v)) for v in hand['w_before']]}")
        log(f"  w_after {[repr(float(v)) for v in hand['w_after']]}")
        log(f"  formula_max_abs_err {err:.3e}")
        log(f"  hand recompute max abs err {max(errs):.3e} on {len(errs)} visits")
        if max(errs) > UPDATE_LIMIT:
            raise RuntimeError("hand recompute failed")
        if not fwd["finite"]:
            raise RuntimeError("prog_fwd_s L=512 is not finite")

        log("smoke check 3: whole-store decrease equals the sum of per-slot decreases, 3 visits")
        audits = fwd["audits"]
        if len(audits) != 3:
            raise RuntimeError(f"expected 3 store audits, got {len(audits)}")
        for audit in audits:
            log(
                f"  AUDIT visit {audit['visit']} n_dec {audit['n_dec']} "
                f"whole_start {audit['whole_start']:.12g} whole_end {audit['whole_end']:.12g} "
                f"slot_start {audit['slot_start']:.12g} slot_end {audit['slot_end']:.12g} "
                f"whole_decrease {audit['whole_decrease']:.12g} "
                f"slot_decrease {audit['slot_decrease']:.12g} "
                f"abs_gap {audit['abs_gap']:.3e}"
            )
            if float(audit["abs_gap"]) > STORE_LIMIT or float(audit["abs_end_gap"]) > STORE_LIMIT:
                raise RuntimeError("store partition failed")
            ratio = float(audit["whole_decrease"]) / float(audit["n_dec"])
            # The traced visit's r is the whole decrease over the visit's decisions.
            traced = trace[audit["visit"] - 1]
            if abs(ratio - float(traced["r"])) > UPDATE_LIMIT:
                raise RuntimeError("visit reward is not the whole-store decrease per decision")

        log("smoke check 4: finite at L=512, and L=1024 wall times")
        for policy in POLICIES:
            row = walls_512[policy]
            log(
                f"  {policy} L=512 finite {row['finite']} T_open {row['T_open']:.6f} "
                f"final {row['final']:.6f} wall {row['wall_s']:.2f}s "
                f"learn {row['learn_s']:.2f}s score {row['score_s']:.2f}s "
                f"eval {row['eval_s']:.2f}s"
            )
            if not row["finite"]:
                raise RuntimeError(f"{policy} is not finite at L=512")
        timed = {}
        for policy in ("rand_dwell_s", "prog_fwd_s"):
            row = run_life(SMOKE_SEED, policy, 1024, SMOKE_PERIOD, score=True)
            _check_row(row, SMOKE_SEED, policy, SMOKE_PERIOD, 1024)
            timed[policy] = row
            log(
                f"  {policy} L=1024 finite {row['finite']} T_open {row['T_open']:.6f} "
                f"final {row['final']:.6f} wall {row['wall_s']:.2f}s "
                f"learn {row['learn_s']:.2f}s score {row['score_s']:.2f}s "
                f"eval {row['eval_s']:.2f}s sleep {row['sleep_decisions']} "
                f"births {len(row['birth_log'])}"
            )
            if not row["finite"]:
                raise RuntimeError(f"{policy} L=1024 is not finite")

        per_policy = {}
        durations = []
        for policy in POLICIES:
            from_512 = _linear_scale(walls_512[policy], 512)
            if policy in timed:
                from_1024 = _linear_scale(timed[policy], 1024)
                linear = max(from_512["linear_s"], from_1024["linear_s"])
                component = max(from_512["component_s"], from_1024["component_s"])
            else:
                from_1024 = None
                linear = from_512["linear_s"]
                component = from_512["component_s"]
            per_policy[policy] = {
                "from_512": from_512,
                "from_1024": from_1024,
                "linear_s": float(linear),
                "component_s": float(component),
            }
            durations.append(linear)
            durations.append(linear)
            log(
                f"  estimate {policy} linear {linear:.1f}s component {component:.1f}s "
                f"L512_wall {from_512['wall_s']:.1f}s"
                + (
                    "" if from_1024 is None
                    else f" L1024_wall {from_1024['wall_s']:.1f}s"
                )
            )
        kernel_s = makespan(durations, KERNEL_WORKERS)
        guarded = kernel_s * 1.5
        slow = max(item["linear_s"] for item in per_policy.values())
        component_kernel = makespan(
            [item["component_s"] for item in per_policy.values() for _ in range(2)],
            KERNEL_WORKERS,
        )
        log(
            f"  kernel linear {kernel_s:.1f}s guarded_x1.5 {guarded:.1f}s "
            f"component_kernel {component_kernel:.1f}s slow_life {slow:.1f}s "
            f"jobs {len(durations)} workers {KERNEL_WORKERS} limit {KERNEL_WALL_LIMIT_S:.0f}s"
        )
        payload["walls_512"] = {
            policy: {
                "T_open": float(walls_512[policy]["T_open"]),
                "final": float(walls_512[policy]["final"]),
                "wall_s": float(walls_512[policy]["wall_s"]),
                "learn_s": float(walls_512[policy]["learn_s"]),
                "score_s": float(walls_512[policy]["score_s"]),
                "eval_s": float(walls_512[policy]["eval_s"]),
                "finite": bool(walls_512[policy]["finite"]),
            }
            for policy in POLICIES
        }
        payload["timed_1024"] = {
            policy: {
                "T_open": float(timed[policy]["T_open"]),
                "final": float(timed[policy]["final"]),
                "wall_s": float(timed[policy]["wall_s"]),
                "learn_s": float(timed[policy]["learn_s"]),
                "score_s": float(timed[policy]["score_s"]),
                "eval_s": float(timed[policy]["eval_s"]),
                "finite": bool(timed[policy]["finite"]),
            }
            for policy in timed
        }
        payload["trace"] = trace
        payload["audits"] = audits
        payload["hand_visit"] = 12
        payload["hand_max_abs_err"] = float(max(errs))
        payload["estimate"] = {
            "per_policy": per_policy,
            "kernel_s": float(kernel_s),
            "guarded_s": float(guarded),
            "component_kernel_s": float(component_kernel),
            "slow_s": float(slow),
            "jobs": len(durations),
            "workers": KERNEL_WORKERS,
            "limit_s": KERNEL_WALL_LIMIT_S,
            "over": bool(guarded > KERNEL_WALL_LIMIT_S or slow > KERNEL_WALL_LIMIT_S),
        }
        if payload["estimate"]["over"]:
            raise RuntimeError(
                f"kernel estimate {guarded:.0f}s or slow life {slow:.0f}s "
                f"exceeds {KERNEL_WALL_LIMIT_S:.0f}s"
            )
        payload["ok"] = True
    except Exception as exc:
        payload["ok"] = False
        payload["reason"] = f"{type(exc).__name__}: {exc}"
        log(f"SMOKE_FAIL {payload['reason']}")
    payload["wall_s"] = time.perf_counter() - t0
    payload["log"] = lines
    m7.dump(out_dir / "smoke.json", payload)
    (out_dir / "smoke.log").write_text("\n".join(lines) + "\n")
    if not payload["ok"]:
        print(f"SMOKE_FAIL {payload['reason']}. Kernels not pushed.", flush=True)
        raise SystemExit(2)
    print(f"smoke_ok wall {payload['wall_s']:.1f}s", flush=True)
    return payload


def _write_kernel(directory, seed):
    directory.mkdir(parents=True, exist_ok=True)
    slug = f"m10-w6-s{seed}"
    kernel_id = f"ezzzzzx/{slug}"
    meta = {
        "id": kernel_id,
        "title": slug,
        "code_file": "m10_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (directory / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    jobs = expected_jobs()
    parts = ", ".join(f'("{policy}", {int(period)})' for policy, period in jobs)
    text = f'''"""M10-W6 seed {seed}. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_M10_KERNEL") != "1":
    os.environ["_M10_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("m10_w6.py"))
    if not hits:
        raise SystemExit("m10_w6.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import m10_w6
    m10_w6.kernel_main(
        seed={seed},
        out="/kaggle/working",
        workers=4,
        kernel_id="{kernel_id}",
        jobs=({parts},),
    )


if __name__ == "__main__":
    main()
'''
    (directory / "m10_kernel.py").write_text(text)
    return kernel_id


def stage(smoke_payload, kaggle_dir=None):
    """Write the private dataset folder and one kernel folder per seed. Does not push."""
    if not smoke_payload.get("ok"):
        raise SystemExit("smoke did not pass; kernels not staged")
    if smoke_payload.get("estimate", {}).get("over"):
        raise SystemExit("kernel estimate exceeds 90 minutes; kernels not staged")
    kaggle_dir = Path(kaggle_dir or (ROOT / "kaggle" / "m10w6"))
    if kaggle_dir.exists():
        shutil.rmtree(kaggle_dir)
    code = kaggle_dir / "code"
    code.mkdir(parents=True)
    for name in CODE_FILES:
        src = ROOT / name
        if not src.is_file():
            raise RuntimeError(f"missing code file {name}")
        shutil.copy2(src, code / name)
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "pdp code m10w6",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    ids = []
    for seed in SEEDS:
        ids.append(_write_kernel(kaggle_dir / f"s{seed}", seed))
    print(f"staged {len(ids)} kernels", flush=True)
    for kernel_id in ids:
        print(kernel_id, flush=True)
    return ids


def _fmt(value, digits=6):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _pick(runs, policy, period, seed):
    period = finite_period(period)
    hit = [
        row for row in runs
        if row["policy"] == policy and finite_period(row["period"]) == period and int(row["seed"]) == int(seed)
    ]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {policy} p={period_name(period)} seed {seed}, got {len(hit)}")
    return hit[0]


def _latency_summary(rows):
    items = [item for row in rows for item in row["latency"]]
    values = [item["latency"] for item in items]
    return {
        "n": len(items),
        "found": int(sum(not item["censored"] for item in items)),
        "censored": int(sum(item["censored"] for item in items)),
        "median": _median(values),
        "mean": None if not values else float(np.mean(values)),
    }


def summarize(runs):
    by = {}
    for policy in POLICIES:
        for period in PERIODS:
            rows = [_pick(runs, policy, period, seed) for seed in SEEDS]
            t_mu, t_sd = mean_std([row["T_open"] for row in rows])
            f_mu, f_sd = mean_std([row["final"] for row in rows])
            n_mu, n_sd = mean_std([row["noise_share"] for row in rows])
            c_mu, c_sd = mean_std([row["corridor_share"] for row in rows])
            s_mu, s_sd = mean_std([row["sleep_share"] for row in rows])
            cps = []
            for index, decision in enumerate(checkpoints_of(L_SCORE)):
                mu, sd = mean_std([row["checkpoints"][index]["open"] for row in rows])
                for row in rows:
                    if row["checkpoints"][index]["decision"] != decision:
                        raise RuntimeError("checkpoint order drifted")
                cps.append({"decision": int(decision), "mean": mu, "std_ddof0": sd})
            latency_seeds = []
            for row in rows:
                latency_seeds.append({"seed": int(row["seed"]), **_latency_summary([row])})
            key = f"{policy}|{period}"
            by[key] = {
                "policy": policy,
                "period": int(period),
                "T_open_mean": t_mu,
                "T_open_std_ddof0": t_sd,
                "final_mean": f_mu,
                "final_std_ddof0": f_sd,
                "noise_mean": n_mu,
                "noise_std_ddof0": n_sd,
                "corridor_mean": c_mu,
                "corridor_std_ddof0": c_sd,
                "sleep_mean": s_mu,
                "sleep_std_ddof0": s_sd,
                "checkpoints": cps,
                "latency_by_seed": latency_seeds,
                "latency": _latency_summary(rows),
            }
    gaps = []
    gap_summary = {}
    for period in PERIODS:
        rows = []
        for seed in SEEDS:
            rand = _pick(runs, "rand_dwell_s", period, seed)
            age = _pick(runs, "prog_age_s", period, seed)
            fwd = _pick(runs, "prog_fwd_s", period, seed)
            item = {
                "period": int(period),
                "seed": int(seed),
                "prog_age_s_minus_rand_dwell_s": float(age["T_open"] - rand["T_open"]),
                "prog_fwd_s_minus_rand_dwell_s": float(fwd["T_open"] - rand["T_open"]),
                "prog_fwd_s_minus_prog_age_s": float(fwd["T_open"] - age["T_open"]),
            }
            gaps.append(item)
            rows.append(item)
        block = {}
        for key in GAPS:
            values = [item[key] for item in rows]
            mu, sd = mean_std(values)
            block[key] = {
                "mean": mu,
                "std_ddof0": sd,
                "left_lower": int(sum(value < 0.0 for value in values)),
                "n": len(values),
            }
        gap_summary[str(period)] = block
    weights = {}
    rewards = {}
    for period in PERIODS:
        w_rows = []
        reward_rows = []
        for seed in SEEDS:
            row = _pick(runs, "prog_fwd_s", period, seed)
            w_rows.append({"seed": int(seed), "w": [float(v) for v in row["w"]]})
            reward_rows.append({"seed": int(seed), "visit_reward": row["visit_reward"]})
        w_mean = []
        w_std = []
        for index in range(N_FEAT):
            mu, sd = mean_std([item["w"][index] for item in w_rows])
            w_mean.append(mu)
            w_std.append(sd)
        kind_pool = {}
        for kind in REWARD_KINDS:
            total_n = 0
            total_sum = 0.0
            for item in reward_rows:
                block = item["visit_reward"][kind]
                total_n += int(block["n"])
                total_sum += float(block["sum"])
            kind_pool[kind] = {
                "n": int(total_n),
                "sum": float(total_sum),
                "mean": None if total_n == 0 else float(total_sum / total_n),
            }
        weights[str(period)] = {
            "names": list(FEAT_NAMES),
            "by_seed": w_rows,
            "mean": w_mean,
            "std_ddof0": w_std,
        }
        rewards[str(period)] = {
            "by_seed": reward_rows,
            "pooled": kind_pool,
        }
    return {
        "by_policy": by,
        "gaps": gaps,
        "gap_summary": gap_summary,
        "weights": weights,
        "visit_reward": rewards,
    }


def render_markdown(result):
    runs = result["runs"]
    summary = result["summary"]
    lines = []
    lines.append("# M10-W6 forward value on World6")
    lines.append("")
    lines.append(
        "World6, the cortex, fixed sleep share 0.20, stores, movement, dwell 8, "
        "the exam store and T_open are W6's. L = 4096. birth_period is 128 or 512. "
        "Seeds are 12 through 23. Lower is better."
    )
    lines.append("")
    lines.append("## T_open")
    lines.append("")
    header = "| policy | period | " + " | ".join(f"seed {seed}" for seed in SEEDS) + " | mean | std ddof=0 |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * len(SEEDS) + "---:|---:|")
    for policy in POLICIES:
        for period in PERIODS:
            cells = []
            for seed in SEEDS:
                cells.append(_fmt(_pick(runs, policy, period, seed)["T_open"]))
            block = summary["by_policy"][f"{policy}|{period}"]
            lines.append(
                f"| {policy} | {period} | " + " | ".join(cells)
                + f" | {_fmt(block['T_open_mean'])} | {_fmt(block['T_open_std_ddof0'])} |"
            )
    lines.append("")
    lines.append("## Final checkpoint")
    lines.append("")
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * len(SEEDS) + "---:|---:|")
    for policy in POLICIES:
        for period in PERIODS:
            cells = []
            for seed in SEEDS:
                cells.append(_fmt(_pick(runs, policy, period, seed)["final"]))
            block = summary["by_policy"][f"{policy}|{period}"]
            lines.append(
                f"| {policy} | {period} | " + " | ".join(cells)
                + f" | {_fmt(block['final_mean'])} | {_fmt(block['final_std_ddof0'])} |"
            )
    lines.append("")
    lines.append("## T_open per checkpoint")
    lines.append("")
    points = checkpoints_of(L_SCORE)
    lines.append(
        "| policy | period | seed | "
        + " | ".join(str(point) for point in points)
        + " | T_open |"
    )
    lines.append("|---|---:|---:|" + "---:|" * len(points) + "---:|")
    for policy in POLICIES:
        for period in PERIODS:
            for seed in SEEDS:
                row = _pick(runs, policy, period, seed)
                cells = [_fmt(rec["open"]) for rec in row["checkpoints"]]
                lines.append(
                    f"| {policy} | {period} | {seed} | " + " | ".join(cells)
                    + f" | {_fmt(row['T_open'])} |"
                )
    lines.append("")
    lines.append("| policy | period | checkpoint | mean | std ddof=0 |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy in POLICIES:
        for period in PERIODS:
            block = summary["by_policy"][f"{policy}|{period}"]
            for rec in block["checkpoints"]:
                lines.append(
                    f"| {policy} | {period} | {rec['decision']} | "
                    f"{_fmt(rec['mean'])} | {_fmt(rec['std_ddof0'])} |"
                )
    lines.append("")
    lines.append("## Gaps")
    lines.append("")
    lines.append(
        "On T_open. A negative gap means the first policy is lower. "
        "left lower counts seeds where the gap is negative."
    )
    lines.append("")
    lines.append(
        "| period | seed | prog_age_s − rand_dwell_s | "
        "prog_fwd_s − rand_dwell_s | prog_fwd_s − prog_age_s |"
    )
    lines.append("|---:|---:|---:|---:|---:|")
    for gap in summary["gaps"]:
        lines.append(
            f"| {gap['period']} | {gap['seed']} | "
            f"{_fmt(gap['prog_age_s_minus_rand_dwell_s'])} | "
            f"{_fmt(gap['prog_fwd_s_minus_rand_dwell_s'])} | "
            f"{_fmt(gap['prog_fwd_s_minus_prog_age_s'])} |"
        )
    for period in PERIODS:
        block = summary["gap_summary"][str(period)]
        lines.append(
            f"| {period} | mean | "
            + " | ".join(_fmt(block[key]["mean"]) for key in GAPS)
            + " |"
        )
        lines.append(
            f"| {period} | std ddof=0 | "
            + " | ".join(_fmt(block[key]["std_ddof0"]) for key in GAPS)
            + " |"
        )
        lines.append(
            f"| {period} | left lower (count) | "
            + " | ".join(
                f"{block[key]['left_lower']} of {block[key]['n']}" for key in GAPS
            )
            + " |"
        )
    lines.append("")
    lines.append("## Birth latency")
    lines.append("")
    lines.append(
        "Decisions from a birth until the next stay in that slot. The stay on the birth "
        "decision observed the dead room. Censored latency is L minus the birth decision "
        "when no later stay falls in the slot. The median includes censored values."
    )
    lines.append("")
    lines.append("| policy | period | seed | n | found | censored | median | mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for policy in POLICIES:
        for period in PERIODS:
            block = summary["by_policy"][f"{policy}|{period}"]
            for item in block["latency_by_seed"]:
                lines.append(
                    f"| {policy} | {period} | {item['seed']} | {item['n']} | {item['found']} | "
                    f"{item['censored']} | {_fmt(item['median'])} | {_fmt(item['mean'])} |"
                )
            pooled = block["latency"]
            lines.append(
                f"| {policy} | {period} | all | {pooled['n']} | {pooled['found']} | "
                f"{pooled['censored']} | {_fmt(pooled['median'])} | {_fmt(pooled['mean'])} |"
            )
    lines.append("")
    lines.append(
        "| policy | period | seed | decision | slot | dead kind | newborn kind | "
        "k | next stay | latency | censored |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|")
    for policy in POLICIES:
        for period in PERIODS:
            for seed in SEEDS:
                row = _pick(runs, policy, period, seed)
                for item in row["latency"]:
                    nxt = "—" if item["next_stay"] is None else str(item["next_stay"])
                    lines.append(
                        f"| {policy} | {period} | {seed} | {item['decision']} | {item['slot']} | "
                        f"{item['dead_kind']} | {item['newborn_kind']} | {item['k']} | {nxt} | "
                        f"{item['latency']} | {int(item['censored'])} |"
                    )
    lines.append("")
    lines.append("## Noise share and corridor share")
    lines.append("")
    lines.append(
        "Noise share is stays whose kind at the stay, before that decision's birth, is N, "
        "divided by L. Corridor share is corridor decisions divided by L."
    )
    lines.append("")
    lines.append("| policy | period | seed | noise share | corridor share | sleep share |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for policy in POLICIES:
        for period in PERIODS:
            for seed in SEEDS:
                row = _pick(runs, policy, period, seed)
                lines.append(
                    f"| {policy} | {period} | {seed} | {_fmt(row['noise_share'])} | "
                    f"{_fmt(row['corridor_share'])} | {_fmt(row['sleep_share'])} |"
                )
            block = summary["by_policy"][f"{policy}|{period}"]
            lines.append(
                f"| {policy} | {period} | mean | {_fmt(block['noise_mean'])} | "
                f"{_fmt(block['corridor_mean'])} | {_fmt(block['sleep_mean'])} |"
            )
            lines.append(
                f"| {policy} | {period} | std ddof=0 | {_fmt(block['noise_std_ddof0'])} | "
                f"{_fmt(block['corridor_std_ddof0'])} | {_fmt(block['sleep_std_ddof0'])} |"
            )
    lines.append("")
    lines.append("## prog_fwd_s final weights")
    lines.append("")
    lines.append(
        "One shared linear value, features in this order: "
        + ", ".join(FEAT_NAMES)
        + ". w starts at 0."
    )
    lines.append("")
    lines.append("| period | seed | " + " | ".join(FEAT_NAMES) + " |")
    lines.append("|---:|---:|" + "---:|" * N_FEAT)
    for period in PERIODS:
        block = summary["weights"][str(period)]
        for item in block["by_seed"]:
            cells = " | ".join(_fmt(v, 8) for v in item["w"])
            lines.append(f"| {period} | {item['seed']} | {cells} |")
        lines.append(
            f"| {period} | mean | " + " | ".join(_fmt(v, 8) for v in block["mean"]) + " |"
        )
        lines.append(
            f"| {period} | std ddof=0 | "
            + " | ".join(_fmt(v, 8) for v in block["std_ddof0"])
            + " |"
        )
    lines.append("")
    lines.append("## Mean visit reward by kind")
    lines.append("")
    lines.append(
        "Visit reward is the decrease in whole-exam-store bits from the start of the visit "
        "to the end, divided by the visit's decisions. The kind is the target slot's kind "
        "after the visit's last decision, so a birth on that decision counts. A visit cut "
        "by the end of the life is included. The pooled mean is the sum of visit rewards "
        "divided by the number of visits."
    )
    lines.append("")
    lines.append("| period | seed | kind | n | mean |")
    lines.append("|---:|---:|---|---:|---:|")
    for period in PERIODS:
        block = summary["visit_reward"][str(period)]
        for item in block["by_seed"]:
            for kind in REWARD_KINDS:
                cell = item["visit_reward"][kind]
                lines.append(
                    f"| {period} | {item['seed']} | {kind} | {cell['n']} | {_fmt(cell['mean'])} |"
                )
        for kind in REWARD_KINDS:
            cell = block["pooled"][kind]
            lines.append(
                f"| {period} | all | {kind} | {cell['n']} | {_fmt(cell['mean'])} |"
            )
    lines.append("")
    lines.append("## Walls and kernels")
    lines.append("")
    lines.append("| kernel | seed | wall seconds | workers |")
    lines.append("|---|---:|---:|---:|")
    for kernel in result["kernels"]:
        lines.append(
            f"| {kernel['kernel_id']} | {kernel['seed']} | "
            f"{kernel['wall_s']:.3f} | {kernel['workers']} |"
        )
    lines.append("")
    lines.append(
        "| policy | period | seed | wall seconds | learn seconds | score seconds | eval seconds |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for policy in POLICIES:
        for period in PERIODS:
            for seed in SEEDS:
                row = _pick(runs, policy, period, seed)
                lines.append(
                    f"| {policy} | {period} | {seed} | {row['wall_s']:.3f} | "
                    f"{row['learn_s']:.3f} | {row['score_s']:.3f} | {row['eval_s']:.3f} |"
                )
    lines.append("")
    lines.append("## Smoke test")
    lines.append("")
    lines.append(f"Smoke wall seconds: {result['smoke_wall_s']:.3f}")
    lines.append("")
    lines.append("```")
    lines.append("\n".join(result["smoke_log"]).rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "- World6(world_seed, birth_period). After every positive multiple of birth_period "
        "up to L, one of the 17 slots that are not Z dies and a room is born there. "
        "The draw is default_rng(world_seed + 90000): slot uniform on those 17, then kind "
        "uniform on " + ", ".join(KINDS) + ". Z is never born and never dies. "
        "The slot keeps its ring position and its channel row."
    )
    lines.append(
        "- T_open is the mean of the checkpoint scores at L/8, 2L/8, …, L. Each score is "
        "the mean bits/symbol of a frozen bank on an EVAL=2048 continuation of each slot "
        "that holds a learnable room then, from a zero state, on that slot's channel. "
        "The final score is the checkpoint at L."
    )
    lines.append(
        "- Sleep is the fixed share 0.20, visits of 8, placed as in W5. At L = 4096 that is "
        f"{w4.sleep_decisions(L_SCORE)} decisions, share "
        f"{w4.sleep_decisions(L_SCORE) / float(L_SCORE):.6f}. "
        "Sleep is not a bandit update. It advances the decision clock, so ages grow, and "
        "it does not change stored rewards or w."
    )
    lines.append(
        "- Dwell is 8. Movement is the ring shortest path. An equal split goes left. "
        "rand_dwell_s and prog_age_s are w6_headroom.run_life. "
        "Policy rng for rand_dwell_s is default_rng(1000 + world_seed). "
        "prog_age_s choice rng is default_rng(10000 + world_seed), tau 256, epsilon 0.1, "
        "beta 0.3. prog_fwd_s choice rng is default_rng(1000 + world_seed) and is not the "
        "10000 stream. World, coin, exam, replay, sleep and birth generators are unchanged."
    )
    lines.append(
        "- prog_fwd_s arms are the 18 slots. The first visit to each slot is the nearest "
        "unvisited slot, ties toward +index, and that choice does not draw epsilon. "
        "After that, epsilon 0.1 draws a uniform slot, otherwise argmax of Q. Ties draw "
        "one uniform number per slot and take the max of (Q, draw)."
    )
    lines.append(
        "- The visit reward is the decrease in the sum of bits of every exam glance, all "
        "slots, from before the visit's first decision to after its last decision's learning, "
        "divided by the decisions taken (travel plus dwell, or fewer if the life ends during "
        "the visit). New glances and interference are in the end sum. The start sum is "
        "measured again after any sleep between visits. r_ema uses m7.ema: the first "
        "completed visit sets it, later updates use beta 0.3."
    )
    lines.append(
        "- Features at a choice, for every slot: "
        + ", ".join(FEAT_NAMES)
        + ". r_last and r_ema are 0 on an unvisited slot. log is natural log, and the visit "
        "count is the count before the visit being chosen. Age is decisions since that "
        "slot's last visit ended, or since decision 0 when it has not ended one. "
        "Travel is the number of shortest-path decisions divided by 8."
    )
    lines.append(
        "- Q(x) = w·x, w initialised at 0, one vector shared by every slot. When the visit "
        "ends, r_last, r_ema, the visit count and last_end are written first, so the "
        "finished slot has age 0, then y = r + 0.5 · max_j Q(x_j) with features at that "
        "moment. y is clipped to [-10, 10]. "
        "w ← w + alpha (y − Q(x_i)) x_i with x_i taken at the choice that started the visit "
        "and with alpha = 0.05 / (1 + ||x_i||^2). The first pass updates w too."
    )
    lines.append(
        "- The kind attached to a visit reward is the target slot's kind after the last "
        "decision of the visit. Seeds 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 and 23. "
        "CPU. Private kernels, pool of 4, no GPU, no internet. Nothing retuned."
    )
    lines.append("")
    return "\n".join(lines)


def _load_seed(out_dir, seed):
    out_dir = Path(out_dir)
    names = [
        out_dir / f"s{seed}" / f"m10w6_s{seed}.json",
        out_dir / f"m10w6_s{seed}.json",
    ]
    names += sorted(out_dir.glob(f"s{seed}/**/m10w6_s{seed}.json"))
    found = []
    seen = set()
    for path in names:
        if path.is_file():
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                found.append(path)
    if len(found) != 1:
        raise RuntimeError(f"expected one results file for seed {seed}, found {len(found)}")
    payload = json.loads(found[0].read_text())
    if int(payload["seed"]) != int(seed):
        raise RuntimeError(f"seed file {found[0].name} holds seed {payload['seed']}")
    if payload.get("kernel_id") != f"ezzzzzx/m10-w6-s{seed}":
        raise RuntimeError(f"kernel id drifted for seed {seed}")
    return payload


def assemble(out_dir=None):
    out_dir = Path(out_dir or (ROOT / "results" / "m10w6"))
    smoke_path = out_dir / "smoke.json"
    if not smoke_path.is_file():
        raise RuntimeError("smoke.json is missing")
    smoke_payload = json.loads(smoke_path.read_text())
    if not smoke_payload.get("ok"):
        raise RuntimeError("smoke did not pass")
    runs = []
    kernels = []
    for seed in SEEDS:
        payload = _load_seed(out_dir, seed)
        if len(payload["runs"]) != len(expected_jobs()):
            raise RuntimeError(f"seed {seed} does not have 6 runs")
        for row in payload["runs"]:
            _check_row(row, seed, row["policy"], row["period"], L_SCORE)
            runs.append(row)
        kernels.append({
            "kernel_id": payload["kernel_id"],
            "seed": int(seed),
            "wall_s": float(payload["wall_s"]),
            "workers": int(payload["workers"]),
        })
    if len(runs) != len(SEEDS) * len(expected_jobs()):
        raise RuntimeError("run count is not 72")
    summary = summarize(runs)
    result = {
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "periods": list(PERIODS),
        "L": L_SCORE,
        "feature_names": list(FEAT_NAMES),
        "reward_kinds": list(REWARD_KINDS),
        "gamma": GAMMA,
        "alpha_num": ALPHA_NUM,
        "y_clip": Y_CLIP,
        "runs": runs,
        "summary": summary,
        "kernels": kernels,
        "smoke_wall_s": float(smoke_payload["wall_s"]),
        "smoke_log": list(smoke_payload["log"]),
        "estimate": smoke_payload.get("estimate"),
        "torch": smoke_payload.get("torch"),
        "model_seed": smoke_payload.get("model_seed"),
    }
    text = render_markdown(result)
    m7.dump(out_dir / "results.json", result)
    (out_dir / "RESULTS.md").write_text(text)
    print(f"wrote {out_dir / 'RESULTS.md'} and results.json", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    sub.add_parser("stage")
    assembled = sub.add_parser("assemble")
    assembled.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if args.cmd == "smoke":
        payload = smoke()
        stage(payload)
        return
    if args.cmd == "stage":
        path = ROOT / "results" / "m10w6" / "smoke.json"
        payload = json.loads(path.read_text())
        if not payload.get("ok"):
            raise SystemExit("smoke did not pass")
        stage(payload)
        return
    if args.cmd == "assemble":
        assemble(args.out or None)
        return
    raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
