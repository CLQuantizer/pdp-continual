"""M10b-W6: a forward value of prog_age_s's own visit reward, and interference net of drift.

Numbers only. No verdict. rand_dwell_s and prog_age_s are w6_headroom.run_life.
fwd_own_s and fwd_inter_s use prog_fwd_s's features and normalised LMS, with no target clip.
The visit reward is the per-room exam progress m8_w5 computes. fwd_inter_s adds the other
slots' exam-bit change per symbol per decision, minus an EMA of that rate.
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
    if os.environ.get("_M10B_MAIN") != "1":
        os.environ["_M10B_MAIN"] = "1"
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
import m8_w5 as m8
import m10_w6 as m10
import w4_headroom as w4
import w4_world
import w5_headroom as w5
import w6_headroom as w6
import w6_world
from w4_world import DARK, IDENTITIES, N_RING, PROC_SET
from w5_world import finite_period
from w6_world import World6, expected_birth_log

SEEDS = tuple(range(24, 36))
POLICIES = ("rand_dwell_s", "prog_age_s", "fwd_own_s", "fwd_inter_s")
W6_POLICIES = ("rand_dwell_s", "prog_age_s")
FWD_POLICIES = ("fwd_own_s", "fwd_inter_s")
PERIODS = (128, 512)
L_SCORE = 4096
N_CHECKPOINTS = 8
SLEEP_SHARE = 0.20
GAMMA = 0.5
ALPHA_NUM = 0.05
BASELINE_DECAY = 0.95
N_FEAT = 8
KERNEL_WALL_LIMIT_S = 90 * 60
KERNEL_WORKERS = 4
ROOT = Path(__file__).resolve().parent
CODE_FILES = w6.CODE_FILES + ("m10_w6.py", "m10b_w6.py")
DATASET_ID = "ezzzzzx/pdp-code-m10b"
GAPS = (
    "prog_age_s_minus_rand_dwell_s",
    "fwd_own_s_minus_prog_age_s",
    "fwd_inter_s_minus_fwd_own_s",
    "fwd_own_s_minus_rand_dwell_s",
)
SMOKE_SEED = 12
SMOKE_PERIOD = 128
BITS_LIMIT = 1e-6
UPDATE_LIMIT = 1e-12
REWARD_SCALE_LIMIT = 100.0

if curio.EPS != 0.1 or curio.BETA != 0.3 or m7.DWELL != 8 or w6.TAU != 256:
    raise RuntimeError("fixed controller constants drifted")
if m10.GAMMA != GAMMA or m10.ALPHA_NUM != ALPHA_NUM or m10.N_FEAT != N_FEAT:
    raise RuntimeError("M10 TD constants drifted")
if list(m10.FEAT_NAMES) != ["bias", "r_last", "r_ema", "log_visits", "age64", "age256", "age1024", "travel"]:
    raise RuntimeError("M10 feature order drifted")
if w6.L_SCORE != L_SCORE or w6.SLEEP_SHARE != SLEEP_SHARE or tuple(w6.PERIODS) != PERIODS:
    raise RuntimeError("W6 score constants drifted")
if w4.N_CH != N_RING + 1 or N_RING != 18:
    raise RuntimeError("ring or dwell drifted")
if curio.B != 64 or m7.R != 32:
    raise RuntimeError("glance width or exam cap drifted")
if ".env" in CODE_FILES or "h34a_local_depth.py" in CODE_FILES:
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


def _reward_block(values):
    if not values:
        return {"n": 0, "sum": 0.0, "mean": None, "min": None, "max": None}
    arr = np.asarray(list(values), dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("a visit reward is not finite")
    return {
        "n": int(arr.size),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def apply_td(w, x_i, r, max_q):
    """One normalised LMS step. The target y is not clipped."""
    x_i = np.asarray(x_i, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x_i.shape != (N_FEAT,) or w.shape != (N_FEAT,):
        raise RuntimeError("weight or feature shape drifted")
    q_before = float(np.dot(w, x_i))
    y_raw = float(r) + GAMMA * float(max_q)
    y = float(y_raw)
    alpha = ALPHA_NUM / (1.0 + float(np.dot(x_i, x_i)))
    step = alpha * (y - q_before)
    w_new = w + step * x_i
    for i in range(N_FEAT):
        piece = float(w[i]) + step * float(x_i[i])
        if abs(piece - float(w_new[i])) > UPDATE_LIMIT:
            raise RuntimeError("TD update does not match the formula")
    q_after = float(np.dot(w_new, x_i))
    if not math.isfinite(q_after) or not np.all(np.isfinite(w_new)) or not math.isfinite(y):
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


def verify_td(rec):
    """Recompute one logged visit. Returns the max abs weight error. No clip."""
    x_i = np.asarray(rec["x_i"], dtype=np.float64)
    w0 = np.asarray(rec["w_before"], dtype=np.float64)
    w1 = np.asarray(rec["w_after"], dtype=np.float64)
    q_before = float(np.dot(w0, x_i))
    if abs(q_before - float(rec["q_before"])) > UPDATE_LIMIT:
        raise RuntimeError("Q before is not w·x at choice time")
    y = float(rec["r"]) + GAMMA * float(rec["max_q"])
    if abs(y - float(rec["y_raw"])) > UPDATE_LIMIT or abs(y - float(rec["y"])) > UPDATE_LIMIT:
        raise RuntimeError("y is not r + gamma max Q")
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


def _score_listed(life, glances):
    if not glances:
        return 0.0
    t0 = time.perf_counter()
    values = m7.score_glances(life.cortex.model, life.cortex.channel_emb, glances)
    life.score_s += time.perf_counter() - t0
    return float(values.sum())


def _own_progress(life, symbols, observed, dest_name, check):
    """prog_age_s's per-decision exam progress. check rescores the same glances around it."""
    indep = None
    if check:
        glances = list(life.exam.slots[dest_name])
        index = life.index[dest_name]
        before = m8._exam_bits(life, glances, index)
        progress = float(m8._progress_decision(life, symbols, observed, dest_name))
        after = m8._exam_bits(life, glances, index)
        indep = float(before - after)
        if abs(indep - progress) > UPDATE_LIMIT:
            raise RuntimeError(
                f"independent progress {indep:.16g} != m8 progress {progress:.16g}"
            )
        return progress, indep
    return float(m8._progress_decision(life, symbols, observed, dest_name)), None


def _snapshot_others(life, world, target):
    """Exam glances of every ring slot except the one being visited, scored in one batch."""
    rows = []
    flat = []
    for slot in range(N_RING):
        if int(slot) == int(target):
            continue
        name = world.rooms[int(slot)]
        glances = list(life.exam.slots[name])
        n_sym = int(sum(int(g.symbols.shape[0]) for g in glances))
        rows.append({
            "slot": int(slot),
            "name": name,
            "glances": glances,
            "n_sym": n_sym,
            "start": len(flat),
            "end": len(flat) + len(glances),
        })
        flat.extend(glances)
    if flat:
        t0 = time.perf_counter()
        values = m7.score_glances(life.cortex.model, life.cortex.channel_emb, flat)
        life.score_s += time.perf_counter() - t0
        bits = [float(values[row["start"]:row["end"]].sum()) for row in rows]
    else:
        bits = [0.0 for _ in rows]
    for row, bit in zip(rows, bits):
        row["bits"] = float(bit)
        if row["n_sym"] == 0 and row["bits"] != 0.0:
            raise RuntimeError("exam bits with no symbols")
    return rows


def _rescore_rows(life, rows):
    flat = []
    spans = []
    for row in rows:
        spans.append((len(flat), len(flat) + len(row["glances"])))
        flat.extend(row["glances"])
    if not flat:
        return [0.0 for _ in rows]
    t0 = time.perf_counter()
    values = m7.score_glances(life.cortex.model, life.cortex.channel_emb, flat)
    life.score_s += time.perf_counter() - t0
    return [float(values[a:b].sum()) for a, b in spans]


def _interference(rows, births, n_dec, b):
    """r_inter = u - b. u is other slots' bit decrease per exam symbol per decision.

    b is the EMA of u over previous visits (decay 0.95, initial 0). A slot born during
    the visit is left out of both the bit change and the symbol count. The baseline
    subtracted here is the value before this visit updates it.
    """
    born = {int(slot) for slot in births}
    used = []
    excluded = []
    for row in rows:
        live = list(life_gids(row))
        snap = [int(g.gid) for g in row["glances"]]
        if int(row["slot"]) in born:
            excluded.append(row)
            continue
        if live != snap:
            raise RuntimeError(
                f"slot {row['slot']} exam changed during the visit and was not a birth"
            )
        end_sym = int(sum(int(g.symbols.shape[0]) for g in row["glances"]))
        if end_sym != int(row["n_sym"]):
            raise RuntimeError("included slot symbol count changed")
        used.append(row)
    n_sym = int(sum(int(row["n_sym"]) for row in used))
    bits_start = float(sum(float(row["bits"]) for row in used))
    bits_end = float(sum(float(row["bits_end"]) for row in used))
    if n_sym == 0:
        if bits_start != 0.0 or bits_end != 0.0:
            raise RuntimeError("interference bits with no symbols")
        u = 0.0
    else:
        u = (bits_start - bits_end) / float(n_sym) / float(n_dec)
    if not math.isfinite(u):
        raise RuntimeError("interference rate is not finite")
    r_inter = float(u - b)
    b_next = BASELINE_DECAY * float(b) + (1.0 - BASELINE_DECAY) * float(u)
    excluded_syms = int(sum(int(row["n_sym"]) for row in excluded))
    return {
        "u": float(u),
        "b": float(b),
        "b_next": float(b_next),
        "r_inter": float(r_inter),
        "n_sym": n_sym,
        "n_sym_excluded": excluded_syms,
        "included": [int(row["slot"]) for row in used],
        "excluded": [int(row["slot"]) for row in excluded],
        "bits_start": bits_start,
        "bits_end": bits_end,
        "excluded_bits_start": float(sum(float(row["bits"]) for row in excluded)),
        "excluded_detail": [
            {
                "slot": int(row["slot"]),
                "n_sym": int(row["n_sym"]),
                "bits_start": float(row["bits"]),
            }
            for row in excluded
        ],
    }


def life_gids(row):
    return [int(g.gid) for g in row["live_glances"]]


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
        float(band["b"]),
    )


def run_w6(seed, policy, L, period):
    """W6's life. The EMA hook records prog_age_s visit rewards and does not change them."""
    captured = []
    original = m7.ema

    def wrapped(values, counts, arm, reward):
        captured.append(float(reward))
        return original(values, counts, arm, reward)

    m7.ema = wrapped
    try:
        row = w6.run_life(int(seed), policy, int(L), period, score=True)
    finally:
        m7.ema = original
    row = dict(row)
    block = _reward_block(captured)
    if policy == "rand_dwell_s" and block["n"] != 0:
        raise RuntimeError("rand_dwell_s updated a visit reward")
    if policy == "prog_age_s" and block["n"] != int(row["n_choices"]):
        raise RuntimeError("prog_age_s reward count does not match choices")
    row["reward"] = block
    row["reward_own"] = None if policy != "prog_age_s" else dict(block)
    row["reward_inter"] = None
    row["w"] = None
    row["y_clip"] = None
    row["baseline_decay"] = None
    return row


def run_fwd(seed, L, period, policy, trace=False):
    """Shared linear TD. Reward is r_own, or r_own + r_inter. No clip."""
    if policy not in FWD_POLICIES:
        raise ValueError(policy)
    use_inter = policy == "fwd_inter_s"
    L = int(L)
    period = finite_period(period)
    seed = int(seed)
    if period is None:
        raise ValueError(f"{policy} requires a birth period")
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
        "b": 0.0,
        "n_choices": 0,
        "n_first": 0,
        "n_eps": 0,
        "n_greedy": 0,
        "n_updates": 0,
        "trace": [],
        "birth_example": None,
    }
    rewards = []
    owns = []
    inters = []
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
        event = world.maybe_birth(done)
        visit = state["visit"]
        if event is not None and visit is not None and visit["open"]:
            if int(event["decision"]) != int(done):
                raise RuntimeError("birth decision does not match the clock")
            visit["births"].append({
                "decision": int(event["decision"]),
                "slot": int(event["slot"]),
                "dead_kind": event["dead_kind"],
                "newborn_kind": event["newborn_kind"],
                "k": int(event["k"]),
            })
        if done in point_set:
            rec = w6.take_checkpoint(life, done, prev_cp["t"], clock)
            prev_cp["t"] = done
            records.append(rec)
            print(
                f"s{seed} {policy} p={period_name(period)} checkpoint t={done}/{L} "
                f"open {rec['open']:.4f} n {rec['n']}",
                flush=True,
            )

    def start_visit():
        if state["room"] != world.room:
            raise RuntimeError("state room and world room diverged")
        if state["plan"]:
            raise RuntimeError("a visit started while a plan was still running")
        stored_counts = list(band["counts"])
        target, mode, x_i = m10.choose_slot(
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
        check = bool(trace and len(band["trace"]) < 12)
        state["visit"] = {
            "open": True,
            "start_t": int(t),
            "travel": 0,
            "dwell": 0,
            "dwell_name": None,
            "target": int(target),
            "mode": mode,
            "x_i": x_i,
            "visits_before": int(stored_counts[target]),
            "own": 0.0,
            "own_indep": 0.0 if check else None,
            "check": check,
            "births": [],
            "others": _snapshot_others(life, world, target) if use_inter else None,
        }

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
        r_own = float(info["own"]) / float(n_dec)
        if info["check"]:
            r_prog = float(info["own_indep"]) / float(n_dec)
            if abs(r_own - r_prog) > UPDATE_LIMIT:
                raise RuntimeError(
                    f"visit reward {r_own:.16g} != independent prog_age reward {r_prog:.16g}"
                )
        else:
            r_prog = None
        if not math.isfinite(r_own):
            raise RuntimeError("nonfinite room reward")
        inter = None
        if use_inter:
            born_slots = [int(event["slot"]) for event in info["births"]]
            if len(born_slots) != len(set(born_slots)):
                raise RuntimeError("one visit recorded the same birth twice")
            for event in info["births"]:
                if not (int(info["start_t"]) < int(event["decision"]) <= int(t)):
                    raise RuntimeError("a birth outside the visit was charged to it")
            rows = info["others"]
            born = set(born_slots)
            end_rows = [row for row in rows if int(row["slot"]) not in born]
            # Rescore the snapshot, not the live slot, so a birth cannot enter the sum.
            # Included slots must still hold those glances.
            for row in rows:
                row["live_glances"] = list(life.exam.slots[row["name"]])
            end_bits = _rescore_rows(life, end_rows)
            by_slot = {int(row["slot"]): bit for row, bit in zip(end_rows, end_bits)}
            for row in rows:
                row["bits_end"] = float(by_slot.get(int(row["slot"]), 0.0))
            inter = _interference(rows, born_slots, n_dec, float(band["b"]))
            for slot in born_slots:
                if slot in inter["included"]:
                    raise RuntimeError(f"newborn slot {slot} entered r_inter")
                if slot != int(info["target"]) and slot not in inter["excluded"]:
                    raise RuntimeError(f"newborn slot {slot} was not recorded as excluded")
            if int(info["target"]) in inter["included"]:
                raise RuntimeError("the visited slot entered r_inter")
            r_inter = float(inter["r_inter"])
            band["b"] = float(inter["b_next"])
        else:
            r_inter = 0.0
        reward = float(r_own + r_inter) if use_inter else float(r_own)
        if not math.isfinite(reward):
            raise RuntimeError("nonfinite TD reward")
        target = int(info["target"])
        band["r_last"][target] = float(reward)
        m7.ema(band["r_ema"], band["counts"], target, float(reward))
        band["last_end"][target] = int(t)
        if int(t) - int(band["last_end"][target]) != 0:
            raise RuntimeError("just-visited slot is not age 0")
        xs = m10.slot_features(
            state["room"], band["r_last"], band["r_ema"], band["counts"], band["last_end"], t,
        )
        if xs[target, 4] != 0.0 or xs[target, 5] != 0.0 or xs[target, 6] != 0.0:
            raise RuntimeError("age features of the finished slot are not 0")
        q = xs @ band["w"]
        max_q = float(np.max(q))
        w_new, rec = apply_td(band["w"], info["x_i"], reward, max_q)
        if abs(float(rec["y"]) - float(rec["y_raw"])) > 0.0:
            raise RuntimeError("the target was clipped")
        band["w"] = w_new
        band["n_updates"] += 1
        rewards.append(float(reward))
        owns.append(float(r_own))
        if use_inter:
            inters.append(float(r_inter))
        info["open"] = False
        rec["visit"] = int(band["n_updates"])
        rec["slot"] = int(target)
        rec["mode"] = info["mode"]
        rec["n_dec"] = int(n_dec)
        rec["r_own"] = float(r_own)
        rec["r_prog"] = None if r_prog is None else float(r_prog)
        rec["r_inter"] = None if not use_inter else float(r_inter)
        rec["u"] = None if inter is None else float(inter["u"])
        rec["b"] = None if inter is None else float(inter["b"])
        if trace and band["n_updates"] <= 12:
            band["trace"].append(rec)
            w_txt = " ".join(f"{v:.8f}" for v in rec["w_after"])
            if use_inter:
                print(
                    f"INTER visit {rec['visit']} slot {rec['slot']} "
                    f"r_own {rec['r_own']:.12g} r_inter {rec['r_inter']:.12g} "
                    f"b {rec['b']:.12g} u {rec['u']:.12g} r {rec['r']:.12g} "
                    f"y {rec['y']:.12g}",
                    flush=True,
                )
            else:
                print(
                    f"OWN visit {rec['visit']} slot {rec['slot']} "
                    f"r {rec['r']:.12g} r_prog {rec['r_prog']:.12g} "
                    f"abs {abs(rec['r'] - rec['r_prog']):.3e} "
                    f"y {rec['y']:.12g} Q_before {rec['q_before']:.12g} "
                    f"Q_after {rec['q_after']:.12g} w {w_txt}",
                    flush=True,
                )
        if use_inter and info["births"] and band["birth_example"] is None:
            newborn = info["births"][0]
            band["birth_example"] = {
                "visit": int(rec["visit"]),
                "target": int(target),
                "decision": int(newborn["decision"]),
                "slot": int(newborn["slot"]),
                "dead_kind": newborn["dead_kind"],
                "newborn_kind": newborn["newborn_kind"],
                "k": int(newborn["k"]),
                "n_births": len(info["births"]),
                "birth_slots": [int(event["slot"]) for event in info["births"]],
                "included": list(inter["included"]),
                "excluded": list(inter["excluded"]),
                "n_sym": int(inter["n_sym"]),
                "n_sym_excluded": int(inter["n_sym_excluded"]),
                "excluded_detail": list(inter["excluded_detail"]),
                "r_own": float(r_own),
                "r_inter": float(r_inter),
                "b": float(inter["b"]),
                "u": float(inter["u"]),
                "r": float(reward),
            }
            ex = band["birth_example"]
            print(
                f"BIRTH visit {ex['visit']} decision {ex['decision']} "
                f"newborn_slot {ex['slot']} target {ex['target']} "
                f"excluded {ex['slot'] in ex['excluded'] or ex['slot'] == ex['target']} "
                f"in_included {ex['slot'] in ex['included']} "
                f"n_sym_used {ex['n_sym']} n_sym_excluded {ex['n_sym_excluded']} "
                f"detail {ex['excluded_detail']}",
                flush=True,
            )

    while t < L:
        if not state["plan"] and w4.sleep_due(t, L, plan, si, N_RING):
            if state["visit"] is not None and state["visit"]["open"]:
                raise RuntimeError("sleep started inside a room visit")
            visit_pending = w4._close_visit(life, visit_pending)
            frozen = _freeze_band(band)
            _n, t, checked = w5.sleep_visit(
                life, seed, policy, period, sleep_marks, plan[si][1], t, L,
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
            dest = world.rooms[state["target"]]
            progress, indep = _own_progress(
                life, symbols, channel, dest, state["visit"]["check"],
            )
            state["visit"]["own"] += float(progress)
            if indep is not None:
                state["visit"]["own_indep"] += float(indep)
            if channel == "corridor":
                state["visit"]["travel"] += 1
            else:
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
            if learnable_stay and kind_now not in PROC_SET:
                raise RuntimeError("learnable stay flag disagrees with the kind")
            if not learnable_stay and kind_now in PROC_SET:
                raise RuntimeError("a learnable stay was not marked")
            stays.append((t + 1, int(room)))
            if learnable_stay:
                learnable_stays += 1
        t += 1
        on_decision(t)
        if t % 256 == 0:
            print(f"s{seed} {policy} p={period_name(period)} t={t}/{L}", flush=True)
        if not state["plan"]:
            visit_pending = w4._close_visit(life, visit_pending)
            finish_visit()
    finish_visit()
    if visit_pending is not None:
        life.snapshot(visit_pending)
    if si != len(plan):
        raise RuntimeError(f"{policy} took {si} of {len(plan)} planned sleep visits")
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
        raise RuntimeError(f"{policy} ran {decisions_taken} decisions")
    if sum(slot_counts) != len(stays):
        raise RuntimeError("stay log length drifted")
    if noise_count + dark_count + learnable_stays != sum(slot_counts):
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
    if len(rewards) != band["n_choices"]:
        raise RuntimeError("reward log length drifted")
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
        and math.isfinite(float(band["b"]))
    )
    wall = time.perf_counter() - t0
    return {
        "seed": seed,
        "policy": policy,
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
        "w_names": list(m10.FEAT_NAMES),
        "gamma": float(GAMMA),
        "alpha_num": float(ALPHA_NUM),
        "y_clip": None,
        "baseline_decay": None if not use_inter else float(BASELINE_DECAY),
        "b_final": None if not use_inter else float(band["b"]),
        "choice_seed": int(1_000 + seed),
        "reward": _reward_block(rewards),
        "reward_own": _reward_block(owns),
        "reward_inter": None if not use_inter else _reward_block(inters),
        "trace": list(band["trace"]),
        "birth_example": band["birth_example"],
    }


def run_life(seed, policy, L, period, score=True, trace=False):
    if not score:
        raise ValueError("m10b lives are scored")
    if policy in W6_POLICIES:
        if trace:
            raise ValueError("the visit trace is only for the forward controllers")
        return run_w6(seed, policy, L, period)
    if policy in FWD_POLICIES:
        return run_fwd(seed, L, period, policy, trace=trace)
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
    reward = row["reward"]
    if policy == "prog_age_s":
        if row["tau"] != w6.TAU or row["n_choices"] != row["ema_updates"]:
            raise RuntimeError("prog_age_s bookkeeping drifted")
        if reward["n"] != int(row["n_choices"]):
            raise RuntimeError("prog_age_s rewards do not match choices")
    elif policy == "rand_dwell_s":
        if row["tau"] is not None or row["n_choices"] is not None:
            raise RuntimeError("rand_dwell_s stored bandit state")
        if reward["n"] != 0:
            raise RuntimeError("rand_dwell_s stored a visit reward")
    elif policy in FWD_POLICIES:
        if row["tau"] is not None:
            raise RuntimeError(f"{policy} stored an ageing tau")
        if row["y_clip"] is not None:
            raise RuntimeError(f"{policy} stored a clip")
        if len(row["w"]) != N_FEAT or row["w_names"] != list(m10.FEAT_NAMES):
            raise RuntimeError(f"{policy} weight layout drifted")
        if not all(math.isfinite(float(v)) for v in row["w"]):
            raise RuntimeError(f"{policy} weight is not finite")
        if row["n_updates"] != row["n_choices"] or row["ema_updates"] != row["n_choices"]:
            raise RuntimeError(f"{policy} updates do not match choices")
        if row["n_first"] + row["n_eps"] + row["n_greedy"] != row["n_choices"]:
            raise RuntimeError(f"{policy} choice modes do not add up")
        if row["gamma"] != GAMMA or row["alpha_num"] != ALPHA_NUM:
            raise RuntimeError(f"{policy} learning constants drifted")
        if row["choice_seed"] != 1_000 + int(seed):
            raise RuntimeError(f"{policy} choice seed drifted")
        if reward["n"] != int(row["n_choices"]) or row["reward_own"]["n"] != int(row["n_choices"]):
            raise RuntimeError(f"{policy} rewards do not match choices")
        if policy == "fwd_own_s":
            if row["baseline_decay"] is not None or row["reward_inter"] is not None:
                raise RuntimeError("fwd_own_s stored an interference term")
        else:
            if row["baseline_decay"] != BASELINE_DECAY:
                raise RuntimeError("fwd_inter_s baseline decay drifted")
            if row["reward_inter"]["n"] != int(row["n_choices"]):
                raise RuntimeError("fwd_inter_s interference rewards do not match choices")
            if not math.isfinite(float(row["b_final"])):
                raise RuntimeError("baseline is not finite")
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
    row = run_life(seed, policy, L, period, score=True, trace=False)
    _check_row(row, seed, policy, period, L)
    row.pop("trace", None)
    preq = "none" if row["preq_bps"] is None else f"{row['preq_bps']:.4f}"
    reward = row["reward"]
    reward_txt = "none" if reward["n"] == 0 else (
        f"mean {reward['mean']:.6f} min {reward['min']:.6f} max {reward['max']:.6f}"
    )
    print(
        f"s{seed} {policy} p={period_name(period)} L={L} T_open {row['T_open']:.4f} "
        f"final {row['final']:.4f} noise {row['noise_share']:.4f} "
        f"preq {preq} sleep {row['sleep_share']:.4f} "
        f"corridor {row['corridor_share']:.3f} wall {row['wall_s']:.1f}s "
        f"learn {row['learn_s']:.1f}s score {row['score_s']:.1f}s eval {row['eval_s']:.1f}s "
        f"reward {reward_txt}",
        flush=True,
    )
    return row


def kernel_main(seed, out, workers=4, kernel_id=None, jobs=None):
    if __name__ != "m10b_w6":
        raise SystemExit("import m10b_w6 and call kernel_main so the pool can spawn workers")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 24-35")
    if int(workers) < 1 or int(workers) > KERNEL_WORKERS:
        raise SystemExit("kernel pool is at most 4")
    if jobs is None:
        jobs = expected_jobs()
    jobs = tuple((str(policy), finite_period(period)) for policy, period in jobs)
    allowed = set(expected_jobs())
    if not jobs or any(job not in allowed for job in jobs) or len(jobs) != len(set(jobs)):
        raise SystemExit(f"jobs {jobs} are not an M10b kernel")
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
    periods = sorted({int(period) for _, period in jobs})
    payload = {
        "kernel_id": kernel_id,
        "seed": seed,
        "periods": periods,
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
    if periods == list(PERIODS):
        name = f"m10b_s{seed}.json"
    elif len(periods) == 1:
        name = f"m10b_s{seed}_p{periods[0]}.json"
    else:
        raise SystemExit(f"unexpected periods {periods}")
    path = out / name
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
    return {
        "src_L": int(src_L),
        "wall_s": wall,
        "eval_s": eval_s,
        "learn_s": learn_s,
        "score_s": score_s,
        "linear_s": float(wall * scale),
        "component_s": float(eval_s + (learn_s + score_s + other) * scale),
    }


def _span(block):
    if block is None or block["n"] == 0:
        return 0.0
    return max(abs(float(block["min"])), abs(float(block["max"])))


def smoke(out_dir=None):
    """VPS checks, one process. A failed check does not stage kernels."""
    _threads()
    out_dir = Path(out_dir or (ROOT / "results" / "m10b"))
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
        log("smoke check 0: TD target is not clipped, baseline step is 0.95")
        w0 = np.zeros(N_FEAT, dtype=np.float64)
        x0 = np.zeros(N_FEAT, dtype=np.float64)
        x0[0] = 1.0
        _w1, rec = apply_td(w0, x0, 1000.0, 0.0)
        if abs(rec["y"] - 1000.0) > UPDATE_LIMIT or abs(rec["y"] - rec["y_raw"]) > 0.0:
            raise RuntimeError("a target of 1000 was clipped")
        if abs(verify_td(rec)) > UPDATE_LIMIT:
            raise RuntimeError("hand TD check failed")
        b0 = 0.0
        u0 = 2.0
        r_inter0 = u0 - b0
        b1 = BASELINE_DECAY * b0 + (1.0 - BASELINE_DECAY) * u0
        if r_inter0 != 2.0 or abs(b1 - 0.1) > UPDATE_LIMIT:
            raise RuntimeError("baseline EMA is not decay 0.95")
        log(f"  y {rec['y']:.1f} y_raw {rec['y_raw']:.1f} b_after {b1:.3f}")

        log(
            "smoke check 1: rand_dwell_s and prog_age_s, seed 12, period 128, L=512, "
            "against w6_headroom.run_life, T_open to 1e-6"
        )
        walls_512 = {}
        for policy in W6_POLICIES:
            direct = w6.run_life(SMOKE_SEED, policy, 512, SMOKE_PERIOD, score=True)
            wrapped = run_life(SMOKE_SEED, policy, 512, SMOKE_PERIOD, score=True)
            _check_row(wrapped, SMOKE_SEED, policy, SMOKE_PERIOD, 512)
            delta = _score_delta(direct, wrapped)
            same_birth = direct["birth_log"] == wrapped["birth_log"]
            preq_gap = abs(float(direct["preq_bits"]) - float(wrapped["preq_bits"]))
            log(
                f"  {policy} direct T_open {direct['T_open']:.10f} "
                f"wrapper T_open {wrapped['T_open']:.10f} "
                f"max_abs {delta:.3e} preq_gap {preq_gap:.3e} "
                f"birth_match {int(same_birth)} finite {wrapped['finite']} "
                f"wall_wrapper {wrapped['wall_s']:.2f}s "
                f"learn {wrapped['learn_s']:.2f}s score {wrapped['score_s']:.2f}s "
                f"eval {wrapped['eval_s']:.2f}s"
            )
            if delta > BITS_LIMIT or preq_gap > BITS_LIMIT or not same_birth:
                raise RuntimeError(f"{policy} does not reproduce M10 to {BITS_LIMIT}")
            if not wrapped["finite"] or not direct["finite"]:
                raise RuntimeError(f"{policy} L=512 is not finite")
            walls_512[policy] = wrapped

        log("smoke check 2: fwd_own_s seed 12 period 128 L=512, first 12 visits")
        own = run_life(SMOKE_SEED, "fwd_own_s", 512, SMOKE_PERIOD, score=True, trace=True)
        _check_row(own, SMOKE_SEED, "fwd_own_s", SMOKE_PERIOD, 512)
        walls_512["fwd_own_s"] = own
        trace = own["trace"]
        if len(trace) != 12:
            raise RuntimeError(f"expected 12 traced visits, got {len(trace)}")
        errs = []
        for rec in trace:
            if rec["mode"] != "first":
                raise RuntimeError(f"visit {rec['visit']} mode {rec['mode']} is inside the first pass")
            if rec["r_prog"] is None or abs(float(rec["r"]) - float(rec["r_prog"])) > UPDATE_LIMIT:
                raise RuntimeError("r is not prog_age_s's reward to 1e-12")
            if abs(float(rec["y"]) - float(rec["y_raw"])) > 0.0:
                raise RuntimeError("y was clipped")
            errs.append(verify_td(rec))
        if max(errs) > UPDATE_LIMIT:
            raise RuntimeError("hand recompute failed")
        hand = trace[11]
        log(
            f"HAND visit {hand['visit']} slot {hand['slot']} mode {hand['mode']} "
            f"r {hand['r']!r} r_prog {hand['r_prog']!r} y {hand['y']!r} "
            f"Q_before {hand['q_before']!r} Q_after {hand['q_after']!r}"
        )
        log(f"  x_i {[repr(float(v)) for v in hand['x_i']]}")
        log(f"  max_q {hand['max_q']!r} alpha {hand['alpha']!r}")
        log(f"  w_before {[repr(float(v)) for v in hand['w_before']]}")
        log(f"  w_after {[repr(float(v)) for v in hand['w_after']]}")
        log(f"  hand recompute max abs err {max(errs):.3e} on {len(errs)} visits")
        if not own["finite"]:
            raise RuntimeError("fwd_own_s L=512 is not finite")

        log("smoke check 3: fwd_inter_s seed 12 period 128 L=512")
        inter = run_life(SMOKE_SEED, "fwd_inter_s", 512, SMOKE_PERIOD, score=True, trace=True)
        _check_row(inter, SMOKE_SEED, "fwd_inter_s", SMOKE_PERIOD, 512)
        walls_512["fwd_inter_s"] = inter
        itrace = inter["trace"]
        if len(itrace) != 12:
            raise RuntimeError(f"expected 12 interference visits, got {len(itrace)}")
        for rec in itrace:
            if abs(float(rec["r"]) - (float(rec["r_own"]) + float(rec["r_inter"]))) > UPDATE_LIMIT:
                raise RuntimeError("r is not r_own + r_inter")
            if abs(float(rec["r_inter"]) - (float(rec["u"]) - float(rec["b"]))) > UPDATE_LIMIT:
                raise RuntimeError("r_inter is not u - b")
            verify_td(rec)
        example = inter["birth_example"]
        if example is None:
            births = ", ".join(
                f"d{event['decision']} slot {event['slot']}" for event in inter["birth_log"]
            )
            raise RuntimeError(f"no room visit contained a birth; births: {births}")
        if example["slot"] in example["included"]:
            raise RuntimeError("the newborn entered r_inter")
        newborn_excluded = (
            example["slot"] in example["excluded"] or example["slot"] == example["target"]
        )
        if not newborn_excluded:
            raise RuntimeError("the newborn was neither excluded nor the visited slot")
        log(
            f"  BIRTH visit {example['visit']} decision {example['decision']} "
            f"newborn_slot {example['slot']} target {example['target']} "
            f"in_included {example['slot'] in example['included']} "
            f"in_excluded {example['slot'] in example['excluded']} "
            f"n_sym_used {example['n_sym']} n_sym_excluded {example['n_sym_excluded']} "
            f"detail {example['excluded_detail']}"
        )
        r_span = _span(inter["reward"])
        own_span = _span(inter["reward_own"])
        log(
            f"  r min {inter['reward']['min']:.8g} max {inter['reward']['max']:.8g} "
            f"mean {inter['reward']['mean']:.8g} n {inter['reward']['n']}"
        )
        log(
            f"  r_own min {inter['reward_own']['min']:.8g} max {inter['reward_own']['max']:.8g} "
            f"mean {inter['reward_own']['mean']:.8g}"
        )
        log(
            f"  r_inter min {inter['reward_inter']['min']:.8g} "
            f"max {inter['reward_inter']['max']:.8g} "
            f"mean {inter['reward_inter']['mean']:.8g}"
        )
        inter_span = _span(inter["reward_inter"])
        mean_gap = abs(
            float(inter["reward"]["mean"])
            - float(inter["reward_own"]["mean"])
            - float(inter["reward_inter"]["mean"])
        )
        log(
            f"  scale r span {r_span:.6g} r_own span {own_span:.6g} "
            f"r_inter span {inter_span:.6g} mean_gap {mean_gap:.3e}"
        )
        # r_own is prog_age_s's per-decision exam progress. On a fresh cortex
        # that quantity itself reaches the low hundreds, and r must track it.
        # The M10 failure mode is an r_inter that is a raw whole-store bit sum.
        if mean_gap > 1e-6:
            raise RuntimeError(
                f"mean r is not mean r_own + mean r_inter (gap {mean_gap:.3e})"
            )
        if inter_span >= REWARD_SCALE_LIMIT:
            raise RuntimeError(
                f"r_inter span {inter_span:.6g} is a raw bit-sum "
                f"(r span {r_span:.6g}, r_own span {own_span:.6g})"
            )
        if r_span > max(own_span * 5.0, 1.0):
            raise RuntimeError(
                f"r span {r_span:.6g} is not on the scale of r_own span {own_span:.6g}"
            )
        if not inter["finite"]:
            raise RuntimeError("fwd_inter_s L=512 is not finite")

        log("smoke check 4: all controllers finite at L=512; time a 1024-decision life")
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
        timed = run_life(SMOKE_SEED, "fwd_inter_s", 1024, SMOKE_PERIOD, score=True, trace=False)
        _check_row(timed, SMOKE_SEED, "fwd_inter_s", SMOKE_PERIOD, 1024)
        log(
            f"  fwd_inter_s L=1024 finite {timed['finite']} T_open {timed['T_open']:.6f} "
            f"final {timed['final']:.6f} wall {timed['wall_s']:.2f}s "
            f"learn {timed['learn_s']:.2f}s score {timed['score_s']:.2f}s "
            f"eval {timed['eval_s']:.2f}s sleep {timed['sleep_decisions']} "
            f"births {len(timed['birth_log'])}"
        )
        if not timed["finite"]:
            raise RuntimeError("timed life is not finite")

        per_policy = {}
        for policy in POLICIES:
            from_512 = _linear_scale(walls_512[policy], 512)
            if policy == "fwd_inter_s":
                from_1024 = _linear_scale(timed, 1024)
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
            log(
                f"  estimate {policy} linear {linear:.1f}s component {component:.1f}s "
                f"L512_wall {from_512['wall_s']:.1f}s"
                + ("" if from_1024 is None else f" L1024_wall {from_1024['wall_s']:.1f}s")
            )
        full_durations = [item["linear_s"] for item in per_policy.values() for _ in range(2)]
        full_s = makespan(full_durations, KERNEL_WORKERS)
        full_guard = full_s * 1.5
        slow = max(item["linear_s"] for item in per_policy.values())
        period_durations = [item["linear_s"] for item in per_policy.values()]
        period_s = makespan(period_durations, KERNEL_WORKERS)
        period_guard = period_s * 1.5
        split = bool(full_guard > KERNEL_WALL_LIMIT_S or slow > KERNEL_WALL_LIMIT_S)
        over = bool(split and (period_guard > KERNEL_WALL_LIMIT_S or slow > KERNEL_WALL_LIMIT_S))
        log(
            f"  full kernel linear {full_s:.1f}s guarded_x1.5 {full_guard:.1f}s "
            f"one-period linear {period_s:.1f}s guarded_x1.5 {period_guard:.1f}s "
            f"slow_life {slow:.1f}s split {int(split)} limit {KERNEL_WALL_LIMIT_S:.0f}s"
        )
        if over:
            raise RuntimeError(
                f"even one period is over 90 min "
                f"(guarded {period_guard:.0f}s, slow life {slow:.0f}s)"
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
                "reward": walls_512[policy]["reward"],
            }
            for policy in POLICIES
        }
        payload["timed_1024"] = {
            "policy": "fwd_inter_s",
            "T_open": float(timed["T_open"]),
            "final": float(timed["final"]),
            "wall_s": float(timed["wall_s"]),
            "learn_s": float(timed["learn_s"]),
            "score_s": float(timed["score_s"]),
            "eval_s": float(timed["eval_s"]),
            "finite": bool(timed["finite"]),
        }
        payload["own_trace"] = trace
        payload["inter_trace"] = [
            {
                "visit": rec["visit"],
                "slot": rec["slot"],
                "r_own": rec["r_own"],
                "r_inter": rec["r_inter"],
                "b": rec["b"],
                "u": rec["u"],
                "r": rec["r"],
                "y": rec["y"],
            }
            for rec in itrace
        ]
        payload["birth_example"] = example
        payload["hand_max_abs_err"] = float(max(errs))
        payload["estimate"] = {
            "per_policy": per_policy,
            "full_kernel_s": float(full_s),
            "full_guarded_s": float(full_guard),
            "period_kernel_s": float(period_s),
            "period_guarded_s": float(period_guard),
            "slow_s": float(slow),
            "split_by_period": bool(split),
            "workers": KERNEL_WORKERS,
            "limit_s": KERNEL_WALL_LIMIT_S,
            "over": False,
        }
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


def _write_kernel(directory, seed, jobs):
    directory.mkdir(parents=True, exist_ok=True)
    periods = sorted({int(period) for _, period in jobs})
    # One kernel per seed keeps ezzzzzx/m10b-w6-sN. A period split uses that id for
    # period 128 and ezzzzzx/m10b-w6-sN-p512 for period 512.
    if periods == list(PERIODS) or periods == [128]:
        slug = f"m10b-w6-s{seed}"
    elif periods == [512]:
        slug = f"m10b-w6-s{seed}-p512"
    else:
        raise RuntimeError(f"cannot name jobs {jobs}")
    kernel_id = f"ezzzzzx/{slug}"
    meta = {
        "id": kernel_id,
        "title": slug,
        "code_file": "m10b_kernel.py",
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
    parts = ", ".join(f'("{policy}", {int(period)})' for policy, period in jobs)
    text = f'''"""M10b-W6 seed {seed}. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_M10B_KERNEL") != "1":
    os.environ["_M10B_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("m10b_w6.py"))
    if not hits:
        raise SystemExit("m10b_w6.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import m10b_w6
    m10b_w6.kernel_main(
        seed={seed},
        out="/kaggle/working",
        workers=4,
        kernel_id="{kernel_id}",
        jobs=({parts},),
    )


if __name__ == "__main__":
    main()
'''
    (directory / "m10b_kernel.py").write_text(text)
    return kernel_id


def stage(smoke_payload, kaggle_dir=None):
    """Write the private dataset folder and one kernel folder per seed. Does not push."""
    if not smoke_payload.get("ok"):
        raise SystemExit("smoke did not pass; kernels not staged")
    if smoke_payload.get("estimate", {}).get("over"):
        raise SystemExit("kernel estimate exceeds 90 minutes; kernels not staged")
    kaggle_dir = Path(kaggle_dir or (ROOT / "kaggle" / "m10b"))
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
        "title": "pdp code m10b",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    split = bool(smoke_payload.get("estimate", {}).get("split_by_period"))
    ids = []
    for seed in SEEDS:
        if split:
            for period in PERIODS:
                jobs = [(policy, period) for policy in POLICIES]
                folder = kaggle_dir / f"s{seed}p{period}"
                ids.append(_write_kernel(folder, seed, jobs))
        else:
            ids.append(_write_kernel(kaggle_dir / f"s{seed}", seed, expected_jobs()))
    print(f"staged {len(ids)} kernels split {int(split)}", flush=True)
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


def _pool_reward(rows, key):
    blocks = [row[key] for row in rows]
    if any(block is None for block in blocks):
        return {"n": 0, "sum": 0.0, "mean": None, "min": None, "max": None}
    n = int(sum(int(block["n"]) for block in blocks))
    total = float(sum(float(block["sum"]) for block in blocks))
    mins = [float(block["min"]) for block in blocks if block["n"]]
    maxs = [float(block["max"]) for block in blocks if block["n"]]
    return {
        "n": n,
        "sum": total,
        "mean": None if n == 0 else float(total / n),
        "min": None if not mins else float(min(mins)),
        "max": None if not maxs else float(max(maxs)),
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
            by[f"{policy}|{period}"] = {
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
                "checkpoints": cps,
                "latency_by_seed": latency_seeds,
                "latency": _latency_summary(rows),
                "reward": _pool_reward(rows, "reward"),
                "reward_own": _pool_reward(rows, "reward_own"),
                "reward_inter": _pool_reward(rows, "reward_inter"),
            }
    gaps = []
    gap_summary = {}
    for period in PERIODS:
        rows = []
        for seed in SEEDS:
            rand = _pick(runs, "rand_dwell_s", period, seed)
            age = _pick(runs, "prog_age_s", period, seed)
            own = _pick(runs, "fwd_own_s", period, seed)
            inter = _pick(runs, "fwd_inter_s", period, seed)
            item = {
                "period": int(period),
                "seed": int(seed),
                "prog_age_s_minus_rand_dwell_s": float(age["T_open"] - rand["T_open"]),
                "fwd_own_s_minus_prog_age_s": float(own["T_open"] - age["T_open"]),
                "fwd_inter_s_minus_fwd_own_s": float(inter["T_open"] - own["T_open"]),
                "fwd_own_s_minus_rand_dwell_s": float(own["T_open"] - rand["T_open"]),
            }
            rows.append(item)
            gaps.append(item)
        summary = {"period": int(period)}
        for name in GAPS:
            values = [float(item[name]) for item in rows]
            mu, sd = mean_std(values)
            summary[name] = {
                "mean": mu,
                "std_ddof0": sd,
                "left_lower": int(sum(value < 0.0 for value in values)),
                "n": len(values),
            }
        gap_summary[str(period)] = summary
    return {"by_policy": by, "gaps": gaps, "gap_summary": gap_summary}


def _header_seeds():
    return " | ".join(str(seed) for seed in SEEDS)


def render_markdown(result):
    lines = []
    lines.append("# M10b-W6 forward value on the prog_age_s reward")
    lines.append("")
    lines.append(
        "World6, the cortex, fixed sleep share 0.20, stores, movement, dwell 8, the exam "
        "store and T_open are M10's. L = 4096. birth_period is 128 or 512. Seeds are 24 "
        "through 35. Lower is better. fwd_own_s learns one shared linear value of "
        "prog_age_s's per-room exam progress. fwd_inter_s adds the other slots' exam-bit "
        "change per exam symbol per decision, minus an EMA (decay 0.95, initial 0) of that "
        "rate over previous visits. The TD target is not clipped. No verdict."
    )
    lines.append("")
    lines.append("## T_open")
    lines.append("")
    lines.append(
        f"| policy | period | {_header_seeds()} | mean | std ddof=0 |"
    )
    lines.append("|---|---:|" + "---:|" * (len(SEEDS) + 2))
    for period in PERIODS:
        for policy in POLICIES:
            cells = []
            for seed in SEEDS:
                row = _pick(result["runs"], policy, period, seed)
                cells.append(_fmt(row["T_open"]))
            block = result["summary"]["by_policy"][f"{policy}|{period}"]
            lines.append(
                f"| {policy} | {period} | " + " | ".join(cells)
                + f" | {_fmt(block['T_open_mean'])} | {_fmt(block['T_open_std_ddof0'])} |"
            )
    lines.append("")
    lines.append("## Final checkpoint")
    lines.append("")
    lines.append(f"| policy | period | {_header_seeds()} | mean | std ddof=0 |")
    lines.append("|---|---:|" + "---:|" * (len(SEEDS) + 2))
    for period in PERIODS:
        for policy in POLICIES:
            cells = []
            for seed in SEEDS:
                row = _pick(result["runs"], policy, period, seed)
                cells.append(_fmt(row["final"]))
            block = result["summary"]["by_policy"][f"{policy}|{period}"]
            lines.append(
                f"| {policy} | {period} | " + " | ".join(cells)
                + f" | {_fmt(block['final_mean'])} | {_fmt(block['final_std_ddof0'])} |"
            )
    lines.append("")
    lines.append("## T_open per checkpoint")
    lines.append("")
    lines.append(
        "| policy | period | seed | "
        + " | ".join(str(d) for d in checkpoints_of(L_SCORE))
        + " | T_open |"
    )
    lines.append("|---|---:|---:|" + "---:|" * (N_CHECKPOINTS + 1))
    for period in PERIODS:
        for policy in POLICIES:
            for seed in SEEDS:
                row = _pick(result["runs"], policy, period, seed)
                cells = " | ".join(_fmt(rec["open"]) for rec in row["checkpoints"])
                lines.append(
                    f"| {policy} | {period} | {seed} | {cells} | {_fmt(row['T_open'])} |"
                )
    lines.append("")
    lines.append("| policy | period | checkpoint | mean | std ddof=0 |")
    lines.append("|---|---:|---:|---:|---:|")
    for period in PERIODS:
        for policy in POLICIES:
            block = result["summary"]["by_policy"][f"{policy}|{period}"]
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
        "| period | seed | prog_age_s − rand_dwell_s | fwd_own_s − prog_age_s | "
        "fwd_inter_s − fwd_own_s | fwd_own_s − rand_dwell_s |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for gap in result["summary"]["gaps"]:
        lines.append(
            f"| {gap['period']} | {gap['seed']} | "
            f"{_fmt(gap['prog_age_s_minus_rand_dwell_s'])} | "
            f"{_fmt(gap['fwd_own_s_minus_prog_age_s'])} | "
            f"{_fmt(gap['fwd_inter_s_minus_fwd_own_s'])} | "
            f"{_fmt(gap['fwd_own_s_minus_rand_dwell_s'])} |"
        )
    for period in PERIODS:
        block = result["summary"]["gap_summary"][str(period)]
        lines.append(
            f"| {period} | mean | "
            + " | ".join(_fmt(block[name]["mean"]) for name in GAPS)
            + " |"
        )
        lines.append(
            f"| {period} | std ddof=0 | "
            + " | ".join(_fmt(block[name]["std_ddof0"]) for name in GAPS)
            + " |"
        )
        lines.append(
            f"| {period} | left lower (count) | "
            + " | ".join(
                f"{block[name]['left_lower']} of {block[name]['n']}" for name in GAPS
            )
            + " |"
        )
    lines.append("")
    lines.append("## Reward")
    lines.append("")
    lines.append(
        "r is the visit reward the controller stores. rand_dwell_s has none. "
        "prog_age_s and fwd_own_s store the per-room exam progress "
        "(sum over the visit of m8._progress_decision, divided by the visit's decisions). "
        "fwd_inter_s stores r_own + r_inter. r_inter = u − b, where u is the decrease in "
        "the other slots' exam bits over the visit, divided by those slots' exam symbols "
        "and by the visit's decisions. A slot born during the visit is not in u. "
        "b starts at 0 and updates after the visit as 0.95·b + 0.05·u. "
        "The pooled mean is the sum of visit rewards divided by the number of visits."
    )
    lines.append("")
    lines.append("| policy | period | n | mean | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for period in PERIODS:
        for policy in POLICIES:
            block = result["summary"]["by_policy"][f"{policy}|{period}"]["reward"]
            lines.append(
                f"| {policy} | {period} | {block['n']} | {_fmt(block['mean'])} | "
                f"{_fmt(block['min'])} | {_fmt(block['max'])} |"
            )
    lines.append("")
    lines.append("| policy | period | seed | n | mean | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for period in PERIODS:
        for policy in POLICIES:
            for seed in SEEDS:
                row = _pick(result["runs"], policy, period, seed)
                block = row["reward"]
                lines.append(
                    f"| {policy} | {period} | {seed} | {block['n']} | "
                    f"{_fmt(block['mean'])} | {_fmt(block['min'])} | {_fmt(block['max'])} |"
                )
    lines.append("")
    lines.append("## Final weights")
    lines.append("")
    lines.append(
        "w at the end of the life, in feature order "
        + ", ".join(m10.FEAT_NAMES)
        + ". rand_dwell_s and prog_age_s have no linear value."
    )
    lines.append("")
    lines.append("| policy | period | seed | " + " | ".join(m10.FEAT_NAMES) + " |")
    lines.append("|---|---:|---:|" + "---:|" * N_FEAT)
    for period in PERIODS:
        for policy in FWD_POLICIES:
            for seed in SEEDS:
                row = _pick(result["runs"], policy, period, seed)
                cells = " | ".join(_fmt(v, 8) for v in row["w"])
                lines.append(f"| {policy} | {period} | {seed} | {cells} |")
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
    for period in PERIODS:
        for policy in POLICIES:
            block = result["summary"]["by_policy"][f"{policy}|{period}"]
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
    lines.append("## Noise and corridor shares")
    lines.append("")
    lines.append("| policy | period | noise mean | noise std ddof=0 | corridor mean | corridor std ddof=0 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for period in PERIODS:
        for policy in POLICIES:
            block = result["summary"]["by_policy"][f"{policy}|{period}"]
            lines.append(
                f"| {policy} | {period} | {_fmt(block['noise_mean'])} | "
                f"{_fmt(block['noise_std_ddof0'])} | {_fmt(block['corridor_mean'])} | "
                f"{_fmt(block['corridor_std_ddof0'])} |"
            )
    lines.append("")
    lines.append("| policy | period | seed | noise | corridor |")
    lines.append("|---|---:|---:|---:|---:|")
    for period in PERIODS:
        for policy in POLICIES:
            for seed in SEEDS:
                row = _pick(result["runs"], policy, period, seed)
                lines.append(
                    f"| {policy} | {period} | {seed} | {_fmt(row['noise_share'])} | "
                    f"{_fmt(row['corridor_share'])} |"
                )
    lines.append("")
    lines.append("## Kernels")
    lines.append("")
    lines.append("| kernel | seed | wall_s | workers |")
    lines.append("|---|---:|---:|---:|")
    for item in result["kernels"]:
        lines.append(
            f"| {item['kernel_id']} | {item['seed']} | {_fmt(item['wall_s'], 1)} | "
            f"{item['workers']} |"
        )
    lines.append("")
    lines.append(f"Smoke wall seconds: {_fmt(result['smoke_wall_s'], 1)}.")
    lines.append("")
    lines.append("## Smoke")
    lines.append("")
    lines.append("```")
    lines.extend(result["smoke_log"])
    lines.append("```")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append(
        f"- Seeds {', '.join(str(s) for s in SEEDS)}. Periods 128 and 512. L = {L_SCORE}. "
        "CPU kernels, private, no GPU, no internet. At most two kernels at a time."
    )
    lines.append(
        "- rand_dwell_s and prog_age_s are w6_headroom.run_life. "
        "Choice rng for the forward controllers is default_rng(1000 + world_seed), "
        "shared with world.step, as in prog_fwd_s. prog_age_s keeps its own choice rng."
    )
    lines.append(
        "- Features, optimistic first pass, epsilon 0.1, argmax with a uniform tie draw, "
        "gamma 0.5 and alpha = 0.05 / (1 + ||x||^2) are prog_fwd_s's. "
        "y = r + 0.5 · max Q, with no clip. r_last and r_ema are features of r."
    )
    lines.append(
        "- r_own sums m8._progress_decision on the destination room over the visit and "
        "divides by the visit's decisions. The new glance is not in that sum."
    )
    lines.append(
        "- u = (other slots' exam bits at the start − the same glances at the end) "
        "/ (their exam symbols) / (visit decisions), or 0 when they hold no symbols. "
        "Positive u means those rooms got better. r_inter = u − b. "
        "b ← 0.95·b + 0.05·u after the visit, an EMA of u, not of the residual. "
        "An EMA of the residual would leave half of a constant u. b starts at 0. "
        "The reward is not rescaled and the TD target is not clipped."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _load_payloads(out_dir):
    out_dir = Path(out_dir)
    found = []
    seen = set()
    for path in sorted(out_dir.glob("s*/**/*.json")) + sorted(out_dir.glob("m10b_s*.json")):
        if not path.is_file():
            continue
        key = str(path.resolve())
        if key in seen or path.name == "smoke.json":
            continue
        seen.add(key)
        payload = json.loads(path.read_text())
        if "runs" not in payload or "kernel_id" not in payload:
            continue
        found.append(payload)
    return found


def assemble(out_dir=None):
    out_dir = Path(out_dir or (ROOT / "results" / "m10b"))
    smoke_path = out_dir / "smoke.json"
    if not smoke_path.is_file():
        raise RuntimeError("smoke.json is missing")
    smoke_payload = json.loads(smoke_path.read_text())
    if not smoke_payload.get("ok"):
        raise RuntimeError("smoke did not pass")
    payloads = _load_payloads(out_dir)
    runs = []
    kernels = []
    seen = {}
    for payload in payloads:
        seed = int(payload["seed"])
        if seed not in SEEDS:
            raise RuntimeError(f"unexpected seed {seed}")
        kid = str(payload["kernel_id"])
        if not kid.startswith(f"ezzzzzx/m10b-w6-s{seed}"):
            raise RuntimeError(f"kernel id drifted for seed {seed}: {kid}")
        for row in payload["runs"]:
            _check_row(row, seed, row["policy"], row["period"], L_SCORE)
            key = (row["policy"], finite_period(row["period"]), int(seed))
            if key in seen:
                raise RuntimeError(f"duplicate run {key}")
            seen[key] = kid
            runs.append(row)
        kernels.append({
            "kernel_id": kid,
            "seed": seed,
            "wall_s": float(payload["wall_s"]),
            "workers": int(payload["workers"]),
            "jobs": list(payload["jobs"]),
        })
    expected = {(policy, period, seed) for seed in SEEDS for policy, period in expected_jobs()}
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        raise RuntimeError(f"missing runs {missing[:8]} count {len(missing)}")
    summary = summarize(runs)
    result = {
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "periods": list(PERIODS),
        "L": L_SCORE,
        "feature_names": list(m10.FEAT_NAMES),
        "gamma": GAMMA,
        "alpha_num": ALPHA_NUM,
        "y_clip": None,
        "baseline_decay": BASELINE_DECAY,
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
        path = ROOT / "results" / "m10b" / "smoke.json"
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
