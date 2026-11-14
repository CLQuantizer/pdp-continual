"""M9-W5: does chosen sleep pay on W5's drifting ring?

Numbers only. No verdict. Imports m8_w5 and does not modify it.
The world, cortex, stores, sleep step, movement, T14 and the final W14 are M8's.
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
    if os.environ.get("_M9_MAIN") != "1":
        os.environ["_M9_MAIN"] = "1"
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
import m7_w3 as m7
import m8_w5 as m8
import w4_headroom as w4
import w4_world
import w5_headroom as w5
from w4_world import IDENTITIES, LEARN14, N_RING
from w5_world import World5

SEEDS = tuple(range(24, 36))
CONTROLLERS = ("age_nosleep", "age_fixed20", "age_sleeparm", "age_fixed_matched")
SLEEPERS = ("age_fixed20", "age_sleeparm", "age_fixed_matched")
PERIOD = 128
L_SCORE = 2048
TAU = m8.TAU
N_CHECKPOINTS = 8
SLEEP_SHARE = 0.20
SLEEP_ARM = N_RING
N_ARMS = N_RING + 1
DROP_DECISION = 256
KERNEL_WALL_LIMIT_S = 90 * 60
BITS_LIMIT = 1e-6
KERNEL_WORKERS = 3
ROOT = Path(__file__).resolve().parent
CODE_FILES = (
    "m9_w5.py",
    "m8_w5.py",
    "w5_world.py",
    "w5_headroom.py",
    "w4_world.py",
    "w4_headroom.py",
    "m7_w3.py",
    "w3_world.py",
    "w3_headroom.py",
    "curio.py",
    "h20_world.py",
    "h30w_ring.py",
    "h30x_trace.py",
    "h30y_roots.py",
)
DATASET_ID = "ezzzzzx/pdp-code-m9w5"
GAP_KEYS = (
    "age_fixed20_minus_age_nosleep",
    "age_sleeparm_minus_age_nosleep",
    "age_sleeparm_minus_age_fixed_matched",
    "age_sleeparm_minus_age_fixed20",
)

if curio.EPS != 0.1 or curio.BETA != 0.3:
    raise RuntimeError(f"bandit constants drifted: EPS {curio.EPS} BETA {curio.BETA}")
if m7.DWELL != 8 or N_RING != 18 or SLEEP_ARM != 18:
    raise RuntimeError("M9 ring or dwell drifted")
if TAU != 256 or m8.PERIOD != PERIOD or m8.L_SCORE != L_SCORE:
    raise RuntimeError("M8 constants drifted")


def _threads():
    m8._threads()


def matched_dose(slept, L):
    """Round a realized sleep count half up to a whole number of 8-decision visits."""
    slept = int(slept)
    L = int(L)
    if slept < 0 or slept > L:
        raise ValueError(f"sleep count {slept} does not fit L={L}")
    visits = int(math.floor(slept / float(m7.DWELL) + 0.5))
    n = int(visits * m7.DWELL)
    if n > L:
        n = (L // m7.DWELL) * m7.DWELL
    return n


def planned_sleep(controller, L, matched_slept):
    if controller == "age_nosleep":
        return 0
    if controller == "age_fixed20":
        return w4.sleep_decisions(L)
    if controller == "age_fixed_matched":
        if matched_slept is None:
            raise ValueError("age_fixed_matched needs age_sleeparm's sleep count")
        return matched_dose(matched_slept, L)
    if controller == "age_sleeparm":
        return None
    raise ValueError(controller)


def _rule(world, name):
    """Swap index of this room's current rule, or -1 when it still has the original draw."""
    k = -1
    for event in world.swap_log:
        if event["room"] == name:
            k = int(event["k"])
    return k


def _remember(life, tags, gid, name):
    found = None
    for store in (life.exam, life.replay):
        for glance in store.slots[name]:
            if glance.gid == gid:
                if found is not None:
                    raise RuntimeError("a glance is in both stores")
                found = glance
    if found is None:
        return
    tags[gid] = {"room": name, "rule": _rule(life.world, name)}


def _install_stores(life, tags, stale):
    """Tag each stored glance with the rule that produced it, and count stale replay draws.

    The wrappers do not draw the coin, step Adam, or rescore. room_decision runs before
    this decision's swap, so the tag is the rule the glance was read from.
    """
    orig_room = life.room_decision

    def room_decision(symbols, name):
        gid = life.next_gid
        orig_room(symbols, name)
        if life.next_gid != gid + 1:
            raise RuntimeError("room decision did not take one glance id")
        _remember(life, tags, gid, name)

    life.room_decision = room_decision
    orig_draw = life.replay.draw

    def draw(rng):
        glance = orig_draw(rng)
        if glance is None:
            stale["empty"] += 1
            return None
        info = tags.get(glance.gid)
        if info is None:
            raise RuntimeError(f"untagged replay glance {glance.gid}")
        now = _rule(life.world, info["room"])
        stale["draws"] += 1
        if now != info["rule"]:
            stale["stale"] += 1
        return glance

    life.replay.draw = draw


def whole_exam_bits(life):
    glances = life.exam.flat()
    if not glances:
        return 0.0
    t0 = time.perf_counter()
    values = m7.score_glances(life.cortex.model, life.cortex.channel_emb, glances)
    life.score_s += time.perf_counter() - t0
    return float(values.sum())


def arm_rows(values, counts, last_end, v_max, t, rooms, with_sleep):
    rows = m8._room_rows(
        rooms,
        list(values[:N_RING]),
        list(counts[:N_RING]),
        list(last_end[:N_RING]),
        v_max,
        t,
    )
    if not with_sleep:
        return rows
    age = None if last_end[SLEEP_ARM] is None else int(t - last_end[SLEEP_ARM])
    aged = None
    if age is not None and v_max is not None:
        aged = m8.aged_value(values[SLEEP_ARM], age, v_max)
        if age == 0 and aged != float(values[SLEEP_ARM]):
            raise RuntimeError(f"age 0 changed sleep: v {values[SLEEP_ARM]} aged {aged}")
    rows.append({
        "index": int(SLEEP_ARM),
        "room": "sleep",
        "v": float(values[SLEEP_ARM]),
        "age": age,
        "aged": None if aged is None else float(aged),
        "count": int(counts[SLEEP_ARM]),
    })
    return rows


def _best_room(rows):
    best = None
    for row in rows:
        if row["room"] == "sleep" or row["aged"] is None:
            continue
        if best is None or (row["aged"], -row["index"]) > (best["aged"], -best["index"]):
            best = row
    return best


def _sleep_aged(rows):
    for row in rows:
        if row["room"] == "sleep":
            return row["aged"]
    return None


def choose_sleeparm(room, unvisited, first_sleep, values, counts, last_end, v_max, t, rooms, rng):
    """18 rooms plus sleep. Sleep is the last arm of the optimistic first pass.

    Greedy ranks by the aged value. The stored EMA is not written.
    Returns (arm, mode, rows, first_sleep).
    """
    if len(values) != N_ARMS or int(room) < 0 or int(room) >= N_RING:
        raise RuntimeError("sleeparm choice is off the ring")
    stored = [float(v) for v in values]
    stored_counts = list(counts)
    rows = arm_rows(values, counts, last_end, v_max, t, rooms, True)
    if unvisited:
        target = m7.nearest_unvisited(int(room), unvisited, n=N_RING)
        unvisited.remove(target)
        mode = "first"
        flag = first_sleep
    elif first_sleep:
        target = SLEEP_ARM
        mode = "first"
        flag = False
    elif rng.random() < curio.EPS:
        target = int(rng.integers(N_ARMS))
        mode = "eps"
        flag = False
    else:
        if v_max is None:
            raise RuntimeError("greedy age with no visit reward")
        if any(last_end[i] is None for i in range(N_ARMS)):
            raise RuntimeError("greedy age before the first pass finished")
        if any(rows[k]["aged"] is None for k in range(N_ARMS)):
            raise RuntimeError("greedy age missing an aged value")
        target = max(range(N_ARMS), key=lambda k: (rows[k]["aged"], rng.random()))
        mode = "greedy"
        flag = False
    if [float(v) for v in values] != stored or list(counts) != stored_counts:
        raise RuntimeError("the choice changed the stored EMA")
    return int(target), mode, rows, bool(flag)


def _choice_snapshot(rows):
    best = _best_room(rows)
    return {
        "sleep_aged": None if _sleep_aged(rows) is None else float(_sleep_aged(rows)),
        "best_room": None if best is None else best["room"],
        "best_aged": None if best is None else float(best["aged"]),
    }


def sleep_quarters(channels):
    n = len(channels)
    if n % 4 != 0:
        raise RuntimeError("life length is not divisible by 4")
    q = n // 4
    out = []
    for i in range(4):
        chunk = channels[i * q:(i + 1) * q]
        out.append(float(sum(name == "sleep" for name in chunk) / q))
    return out


def _since_swap(start, swap_log):
    last = 0
    for event in swap_log:
        if int(event["decision"]) < int(start):
            last = int(event["decision"])
        else:
            break
    return int(start) - last


def _mean_or_none(values):
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def visit_summary(visits, swap_log, L):
    q = int(L) // 4
    slots = [{"sleep_n": 0, "sleep_sum": 0.0, "room_n": 0, "room_sum": 0.0} for _ in range(4)]
    sleep_gaps = []
    room_gaps = []
    for visit in visits:
        slot = slots[min((int(visit["start"]) - 1) // q, 3)]
        if visit["kind"] == "sleep":
            slot["sleep_n"] += 1
            slot["sleep_sum"] += float(visit["reward"])
            sleep_gaps.append(_since_swap(visit["start"], swap_log))
        elif visit["kind"] == "room":
            slot["room_n"] += 1
            slot["room_sum"] += float(visit["reward"])
            room_gaps.append(_since_swap(visit["start"], swap_log))
        else:
            raise RuntimeError(f"visit kind {visit['kind']}")
    quarters = []
    for slot in slots:
        quarters.append({
            "sleep_n": int(slot["sleep_n"]),
            "sleep_mean": (float(slot["sleep_sum"] / slot["sleep_n"]) if slot["sleep_n"] else None),
            "room_n": int(slot["room_n"]),
            "room_mean": (float(slot["room_sum"] / slot["room_n"]) if slot["room_n"] else None),
        })
    return {
        "reward_quarters": quarters,
        "since_swap_sleep": sleep_gaps,
        "since_swap_room": room_gaps,
        "since_swap_sleep_mean": _mean_or_none(sleep_gaps),
        "since_swap_room_mean": _mean_or_none(room_gaps),
        "n_sleep_visits": int(len(sleep_gaps)),
        "n_room_visits": int(len(room_gaps)),
    }


def format_visit(visit):
    reward = "none" if visit["reward"] is None else f"{visit['reward']:.6f}"
    sleep_aged = "none" if visit["sleep_aged"] is None else f"{visit['sleep_aged']:.6f}"
    best_room = "none" if visit["best_room"] is None else visit["best_room"]
    best_aged = "none" if visit["best_aged"] is None else f"{visit['best_aged']:.6f}"
    return (
        f"visit {visit['start']}-{visit['end']} arm {visit['arm']} "
        f"decisions {visit['decisions']} reward {reward} "
        f"sleep_aged {sleep_aged} best_room {best_room} best_aged {best_aged}"
    )


def run_life(seed, controller, L, score=True, check_sleep=False, trace_visits=False, matched_slept=None):
    """One controller on World5(seed, period=128).

    age_fixed20 is prog_age_s. age_nosleep is that agent with an empty sleep plan.
    age_fixed_matched uses the same placement at the rounded dose of matched_slept.
    age_sleeparm adds sleep as arm 18.
    """
    if controller not in CONTROLLERS:
        raise ValueError(controller)
    L = int(L)
    seed = int(seed)
    n_plan = planned_sleep(controller, L, matched_slept)
    if controller == "age_sleeparm":
        plan = []
    else:
        plan = m7.sleep_plan(L, n_plan)
        if n_plan and any(length != m7.DWELL for _, length in plan):
            raise RuntimeError("a fixed sleep visit is not 8 decisions")
    points = w5.checkpoints_of(L)
    point_set = set(points)
    world = World5(seed, PERIOD)
    names = list(IDENTITIES)
    index = {name: i for i, name in enumerate(names)}
    t0 = time.perf_counter()
    life = w4.Life(seed, world, names, index, N_RING, w4.N_CH, list(LEARN14))
    tags = {}
    stale = {"draws": 0, "stale": 0, "empty": 0}
    _install_stores(life, tags, stale)
    rng = np.random.default_rng(1_000 + seed)
    choice_rng = np.random.default_rng(10_000 + seed)
    n_arms = N_ARMS if controller == "age_sleeparm" else N_RING
    band = {
        "values": [0.0] * n_arms,
        "counts": [0] * n_arms,
        "last_end": [None] * n_arms,
        "v_max": None,
        "unvisited": set(range(N_RING)),
        "first_sleep": controller == "age_sleeparm",
        "n_choices": 0,
        "n_first": 0,
        "n_eps": 0,
        "n_greedy": 0,
        "ema_updates": 0,
    }
    state = {"room": 0, "plan": [], "target": None, "arrived": True, "visit": None}
    channels = []
    records = []
    visits = []
    clock = {"eval": 0.0}
    prev_cp = {"t": 0}
    visit_pending = None
    si = 0
    t = 0
    checked = False

    def on_decision(done):
        world.maybe_swap(done)
        if score and done in point_set:
            rec = w5.take_checkpoint(life, done, prev_cp["t"], clock)
            prev_cp["t"] = done
            records.append(rec)
            print(
                f"s{seed} {controller} p={PERIOD} checkpoint t={done}/{L} W14 {rec['W14']:.4f}",
                flush=True,
            )

    def note_choice(mode):
        band["n_choices"] += 1
        band[{"first": "n_first", "eps": "n_eps", "greedy": "n_greedy"}[mode]] += 1

    def start_room(target, mode, rows):
        if state["room"] != world.room:
            raise RuntimeError("state room and world room diverged")
        note_choice(mode)
        state["plan"] = w4_world.shortest(state["room"], target, N_RING) + ["stay"] * m7.DWELL
        state["target"] = int(target)
        state["arrived"] = False
        snap = _choice_snapshot(rows)
        state["visit"] = {
            "open": True,
            "start_t": int(t),
            "total": 0.0,
            "travel": 0,
            "dwell": 0,
            "dwell_name": None,
            "target": int(target),
            "mode": mode,
            "sleep_aged": snap["sleep_aged"],
            "best_room": snap["best_room"],
            "best_aged": snap["best_aged"],
        }

    def finish_room():
        info = state["visit"]
        if info is None or not info["open"]:
            return
        n_dec = int(info["travel"] + info["dwell"])
        if n_dec <= 0:
            raise RuntimeError("empty room visit")
        if info["dwell"] and info["dwell_name"] != world.rooms[info["target"]]:
            raise RuntimeError("dwell channel is not the chosen room")
        reward = float(info["total"] / n_dec)
        if not math.isfinite(reward):
            raise RuntimeError("nonfinite room reward")
        target = int(info["target"])
        m7.ema(band["values"], band["counts"], target, reward)
        if band["v_max"] is None or reward > band["v_max"]:
            band["v_max"] = float(reward)
        band["last_end"][target] = int(t)
        band["ema_updates"] += 1
        info["open"] = False
        if controller == "age_sleeparm":
            visit = {
                "kind": "room",
                "arm": world.rooms[target],
                "start": int(info["start_t"] + 1),
                "end": int(t),
                "decisions": n_dec,
                "reward": float(reward),
                "mode": info["mode"],
                "sleep_aged": info["sleep_aged"],
                "best_room": info["best_room"],
                "best_aged": info["best_aged"],
            }
            visits.append(visit)
            if trace_visits:
                print(format_visit(visit), flush=True)

    def chosen_sleep(mode, rows):
        nonlocal t, checked, visit_pending
        length = min(m7.DWELL, L - t)
        if length <= 0:
            raise RuntimeError("empty sleep visit")
        if state["visit"] is not None and state["visit"]["open"]:
            raise RuntimeError("sleep started inside a room visit")
        note_choice(mode)
        snap_choice = _choice_snapshot(rows)
        visit_pending = w4._close_visit(life, visit_pending)
        values_before = list(band["values"])
        start_t = int(t)
        snap = life.begin_sleep_visit() if (check_sleep and not checked) else None
        total = 0.0
        n = 0
        learned = 0
        while n < length and t < L:
            before = whole_exam_bits(life)
            did = bool(life.sleep_decision())
            if did:
                step = float(before - life.cached_sum)
            else:
                step = 0.0
            if not math.isfinite(step):
                raise RuntimeError("nonfinite sleep progress")
            total += step
            learned += int(did)
            channels.append("sleep")
            n += 1
            t += 1
            on_decision(t)
            if t % 256 == 0:
                print(f"s{seed} {controller} p={PERIOD} t={t}/{L} sleep", flush=True)
        if snap is not None:
            delta = life.end_sleep_visit(snap)
            checked = True
            if learned and delta <= 0.0:
                raise RuntimeError("sleep visit learned and left the parameters unchanged")
        if n != length and t < L:
            raise RuntimeError("sleep visit ended early")
        # A full visit divides by 8. A visit cut off by the end of the life divides by
        # the decisions it actually took, which is the same per-decision reward.
        reward = float(total / n)
        if not math.isfinite(reward):
            raise RuntimeError("nonfinite sleep reward")
        m7.ema(band["values"], band["counts"], SLEEP_ARM, reward)
        if band["v_max"] is None or reward > band["v_max"]:
            band["v_max"] = float(reward)
        band["last_end"][SLEEP_ARM] = int(t)
        band["ema_updates"] += 1
        if band["values"][:N_RING] != values_before[:N_RING]:
            raise RuntimeError("a sleep visit changed a room EMA")
        visit = {
            "kind": "sleep",
            "arm": "sleep",
            "start": int(start_t + 1),
            "end": int(t),
            "decisions": int(n),
            "reward": float(reward),
            "mode": mode,
            "sleep_aged": snap_choice["sleep_aged"],
            "best_room": snap_choice["best_room"],
            "best_aged": snap_choice["best_aged"],
        }
        visits.append(visit)
        if trace_visits:
            print(format_visit(visit), flush=True)

    while t < L:
        if not state["plan"]:
            if controller != "age_sleeparm" and w4.sleep_due(t, L, plan, si, N_RING):
                if state["visit"] is not None and state["visit"]["open"]:
                    raise RuntimeError("sleep started inside a room visit")
                visit_pending = w4._close_visit(life, visit_pending)
                values_before = list(band["values"])
                _n, t, checked = w5.sleep_visit(
                    life, seed, controller, PERIOD, channels, plan[si][1], t, L,
                    check_sleep, checked, on_decision,
                )
                if band["values"] != values_before:
                    raise RuntimeError("a sleep visit changed the stored EMA")
                si += 1
                continue
            if controller == "age_sleeparm":
                arm, mode, rows, flag = choose_sleeparm(
                    world.room, band["unvisited"], band["first_sleep"], band["values"],
                    band["counts"], band["last_end"], band["v_max"], t, world.rooms, choice_rng,
                )
                band["first_sleep"] = flag
                if arm == SLEEP_ARM:
                    chosen_sleep(mode, rows)
                    continue
                start_room(arm, mode, rows)
            else:
                stored_counts = list(band["counts"])
                target, mode, rows = m8.choose_room(
                    world.room, band["unvisited"], band["values"], band["counts"],
                    band["last_end"], band["v_max"], t, world.rooms, choice_rng, True,
                )
                if band["counts"] != stored_counts:
                    raise RuntimeError("the choice changed visit counts")
                start_room(target, mode, rows)
        action = state["plan"].pop(0)
        if action == "stay" and not state["arrived"]:
            name = world.rooms[state["room"]]
            if name != world.rooms[state["target"]]:
                raise RuntimeError(
                    f"arrival room {name} is not target {world.rooms[state['target']]}"
                )
            state["arrived"] = True
        room, channel, symbols = world.step(action, rng)
        state["room"] = room
        dest_name = world.rooms[state["target"]]
        progress = m8._progress_decision(life, symbols, channel, dest_name)
        state["visit"]["total"] += progress
        if channel == "corridor":
            state["visit"]["travel"] += 1
        else:
            state["visit"]["dwell"] += 1
            state["visit"]["dwell_name"] = channel
            visit_pending = channel
        channels.append(channel)
        t += 1
        on_decision(t)
        if t % 256 == 0:
            print(f"s{seed} {controller} p={PERIOD} t={t}/{L}", flush=True)
        if not state["plan"]:
            visit_pending = w4._close_visit(life, visit_pending)
            finish_room()
    finish_room()
    if visit_pending is not None:
        life.snapshot(visit_pending)
    if len(channels) != L:
        raise RuntimeError(f"{controller} ran {len(channels)} decisions")
    if controller != "age_sleeparm":
        if si != len(plan):
            raise RuntimeError(f"{controller} took {si} of {len(plan)} planned sleep visits")
        if sum(name == "sleep" for name in channels) != int(n_plan):
            raise RuntimeError("sleep decisions do not match the dose")
    if life.exam_drawn != 0:
        raise RuntimeError(f"sleep drew {life.exam_drawn} exam glances")
    if life.exam.ids & life.replay.ids:
        raise RuntimeError("a glance is in both stores")
    if controller == "age_nosleep" and stale["draws"] != 0:
        raise RuntimeError("age_nosleep drew a replay glance")
    if [event["decision"] for event in world.swap_log] != list(range(PERIOD, L + 1, PERIOD)):
        raise RuntimeError("swap decisions are not period, 2·period, …")
    if world.swap_log != w5.expected_swap_log(seed, PERIOD, L):
        raise RuntimeError("swap log does not match the independent draw")
    share, corridor_share, sleep_share = w4.shares_of(channels, life.names)
    if abs((sum(share.values()) + corridor_share + sleep_share) - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    n_sleep = int(sum(name == "sleep" for name in channels))
    if abs(sleep_share - (n_sleep / float(L))) > 1e-12:
        raise RuntimeError("sleep share does not match the sleep decisions")
    if band["ema_updates"] != band["n_choices"]:
        raise RuntimeError("EMA updates do not match choices")
    if controller == "age_sleeparm":
        if sum(v["decisions"] for v in visits if v["kind"] == "sleep") != n_sleep:
            raise RuntimeError("sleep visits do not add up to the sleep decisions")
        if any(v["reward"] is None or not math.isfinite(v["reward"]) for v in visits):
            raise RuntimeError("sleeparm visit reward is missing or nonfinite")
        for visit in visits:
            if int(visit["end"]) - int(visit["start"]) + 1 != int(visit["decisions"]):
                raise RuntimeError("visit span does not match its decisions")
    if score:
        if [rec["decision"] for rec in records] != points:
            raise RuntimeError("checkpoints are not L/8, 2L/8, …, L")
        t14 = float(np.mean([rec["W14"] for rec in records]))
        w14 = float(records[-1]["W14"])
        finite = bool(
            math.isfinite(t14) and math.isfinite(w14)
            and all(math.isfinite(rec["W14"]) for rec in records)
            and all(math.isfinite(v) for rec in records for v in rec["held"].values())
        )
    else:
        t14 = None
        w14 = None
        finite = False
    intact = None
    param_delta = None
    if life.sleep_checks:
        intact = all(bool(item["intact"]) for item in life.sleep_checks)
        param_delta = float(max(item["param_delta"] for item in life.sleep_checks))
        if not intact:
            raise RuntimeError("sleep changed the live state or traces")
    summary = None
    if controller == "age_sleeparm":
        summary = visit_summary(visits, world.swap_log, L)
    wall = time.perf_counter() - t0
    stale_share = None if stale["draws"] == 0 else float(stale["stale"] / stale["draws"])
    row = {
        "seed": int(seed),
        "controller": controller,
        "period": int(PERIOD),
        "L": int(L),
        "T14": None if t14 is None else float(t14),
        "W14": None if w14 is None else float(w14),
        "checkpoints": records,
        "swap_log": [
            {"decision": int(event["decision"]), "room": event["room"], "k": int(event["k"])}
            for event in world.swap_log
        ],
        "latency": m8.latencies_of(channels, world.swap_log, L),
        "share": {name: float(share[name]) for name in life.names},
        "corridor_share": float(corridor_share),
        "sleep_share": float(sleep_share),
        "sleep_quarters": sleep_quarters(channels),
        "sleep_decisions": int(n_sleep),
        "planned_sleep": None if n_plan is None else int(n_plan),
        "matched_slept": None if matched_slept is None else int(matched_slept),
        "room_order": list(world.rooms),
        "overlap": int(len(life.exam.ids & life.replay.ids)),
        "exam_glances_drawn": int(life.exam_drawn),
        "replay_draws": int(stale["draws"]),
        "stale_draws": int(stale["stale"]),
        "stale_share": stale_share,
        "sleep_state_intact": intact,
        "sleep_param_delta": param_delta,
        "sleep_checks": int(len(life.sleep_checks)),
        "wall_s": float(wall),
        "eval_s": float(clock["eval"]),
        "learn_s": float(life.learn_s),
        "score_s": float(life.score_s),
        "finite": bool(finite),
        "n_ch": int(w4.N_CH),
        "tau": int(TAU),
        "n_choices": int(band["n_choices"]),
        "n_first": int(band["n_first"]),
        "n_eps": int(band["n_eps"]),
        "n_greedy": int(band["n_greedy"]),
        "ema_updates": int(band["ema_updates"]),
        "v_max": None if band["v_max"] is None else float(band["v_max"]),
    }
    if summary is not None:
        row["reward_quarters"] = summary["reward_quarters"]
        row["since_swap_sleep_mean"] = summary["since_swap_sleep_mean"]
        row["since_swap_room_mean"] = summary["since_swap_room_mean"]
        row["n_sleep_visits"] = summary["n_sleep_visits"]
        row["n_room_visits"] = summary["n_room_visits"]
        row["since_swap_sleep"] = summary["since_swap_sleep"]
        row["since_swap_room"] = summary["since_swap_room"]
    if controller == "age_sleeparm":
        row["visits"] = visits
    return row


def _job(args):
    seed, controller, L, score, trace_visits, matched_slept = args
    _threads()
    dose = planned_sleep(controller, L, matched_slept)
    shown = "chosen" if dose is None else str(dose)
    print(f"s{seed} {controller} p={PERIOD} L={L} start dose {shown}", flush=True)
    row = run_life(
        seed, controller, L, score=score, check_sleep=True,
        trace_visits=trace_visits, matched_slept=matched_slept,
    )
    t14 = "none" if row["T14"] is None else f"{row['T14']:.4f}"
    print(
        f"s{seed} {controller} p={PERIOD} L={L} T14 {t14} "
        f"sleep {row['sleep_share']:.4f} corridor {row['corridor_share']:.3f} "
        f"wall {row['wall_s']:.1f}s learn {row['learn_s']:.1f}s "
        f"score {row['score_s']:.1f}s eval {row['eval_s']:.1f}s",
        flush=True,
    )
    return row


def _m8_job(args):
    seed, L = args
    _threads()
    print(f"s{seed} prog_age_s p={PERIOD} L={L} start dose {w4.sleep_decisions(L)}", flush=True)
    return m8.run_life(seed, "prog_age_s", L, score=True, check_sleep=True)


def _within_bits(a, b, limit=BITS_LIMIT):
    return abs(float(a) - float(b)) <= limit


def _same_score(row, m8_row, limit=BITS_LIMIT):
    if not _within_bits(row["T14"], m8_row["T14"], limit):
        return f"T14 {row['T14']} vs {m8_row['T14']}"
    if not _within_bits(row["W14"], m8_row["W14"], limit):
        return f"W14 {row['W14']} vs {m8_row['W14']}"
    if [c["decision"] for c in row["checkpoints"]] != [c["decision"] for c in m8_row["checkpoints"]]:
        return "checkpoint decisions differ"
    for left, right in zip(row["checkpoints"], m8_row["checkpoints"]):
        if not _within_bits(left["W14"], right["W14"], limit):
            return f"checkpoint {left['decision']} W14 {left['W14']} vs {right['W14']}"
        for name in LEARN14:
            if not _within_bits(left["held"][name], right["held"][name], limit):
                return (
                    f"checkpoint {left['decision']} {name} "
                    f"{left['held'][name]} vs {right['held'][name]}"
                )
    if row["swap_log"] != m8_row["swap_log"]:
        return "swap log differs"
    if row["sleep_decisions"] != m8_row["sleep_decisions"]:
        return f"sleep {row['sleep_decisions']} vs {m8_row['sleep_decisions']}"
    return None


def _pool_map(jobs, fn, workers):
    ctx = get_context("spawn")
    with ctx.Pool(int(workers)) as pool:
        return [item.get() for item in [pool.apply_async(fn, (job,)) for job in jobs]]


def _scale_one(row, L):
    scale = float(L_SCORE) / float(row["L"])
    return float(row["learn_s"] * scale + row["score_s"] * scale + row["eval_s"])


def smoke(out_dir=None, workers=2):
    """VPS checks. A failed check does not stage kernels."""
    _threads()
    workers = int(workers)
    if workers < 1 or workers > 2:
        raise RuntimeError("this VPS pool is at most 2")
    out_dir = Path(out_dir or (ROOT / "results" / "m9w5"))
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
        "tau": TAU,
        "period": PERIOD,
        "ok": False,
    }
    try:
        log(
            f"torch={torch.__version__} threads=1 model_seed={m7.MODEL_SEED} "
            f"n_ch={w4.N_CH} tau={TAU} period={PERIOD}"
        )
        for life_L in (64, 128, 256, 512, L_SCORE):
            dose = w4.sleep_decisions(life_L)
            log(
                f"dose L{life_L} {dose}/{life_L} share {dose / life_L:.6f} "
                f"checkpoints {w5.checkpoints_of(life_L)}"
            )
        if matched_dose(0, L_SCORE) != 0 or matched_dose(4, L_SCORE) != 8:
            raise RuntimeError("matched dose rounding drifted")
        if matched_dose(w4.sleep_decisions(L_SCORE), L_SCORE) != w4.sleep_decisions(L_SCORE):
            raise RuntimeError("the M8 dose is not a whole number of sleep visits")

        log("smoke check 1: age_fixed20 reproduces M8 prog_age_s seed 12")
        check1 = {}
        for life_L in (64, 256):
            pair = smoke_check1_pair(workers, life_L)
            row = pair["m9"]
            m8_row = pair["m8"]
            reason = _same_score(row, m8_row)
            diff = abs(row["T14"] - m8_row["T14"])
            log(
                f"  L={life_L} age_fixed20 T14 {row['T14']:.6f} W14 {row['W14']:.6f} "
                f"prog_age_s T14 {m8_row['T14']:.6f} W14 {m8_row['W14']:.6f} "
                f"abs_dT14 {diff:.3e} limit {BITS_LIMIT:.1e} "
                f"wall {row['wall_s']:.2f}s m8_wall {m8_row['wall_s']:.2f}s"
            )
            if reason is not None or diff > BITS_LIMIT:
                raise RuntimeError(f"age_fixed20 L={life_L} does not reproduce prog_age_s: {reason}")
            check1[str(life_L)] = {
                "T14": row["T14"],
                "W14": row["W14"],
                "m8_T14": m8_row["T14"],
                "m8_W14": m8_row["W14"],
                "abs_dT14": diff,
                "wall_s": row["wall_s"],
                "learn_s": row["learn_s"],
                "score_s": row["score_s"],
                "eval_s": row["eval_s"],
            }
        payload["check1"] = check1

        log("smoke check 2: age_sleeparm seed 24 L=256 stores and sleep state")
        stored = _job((24, "age_sleeparm", 256, False, False, None))
        log(
            f"  overlap {stored['overlap']} exam_drawn {stored['exam_glances_drawn']} "
            f"sleep_decisions {stored['sleep_decisions']} sleep_checks {stored['sleep_checks']} "
            f"intact {stored['sleep_state_intact']} param_delta {stored['sleep_param_delta']} "
            f"replay_draws {stored['replay_draws']} wall {stored['wall_s']:.2f}s"
        )
        if stored["overlap"] != 0:
            raise RuntimeError("a glance is in both stores")
        if stored["exam_glances_drawn"] != 0:
            raise RuntimeError("sleep drew an exam glance")
        if stored["sleep_decisions"] <= 0 or stored["sleep_checks"] < 1:
            raise RuntimeError("the L=256 life did not check a sleep visit")
        if stored["sleep_state_intact"] is not True:
            raise RuntimeError("live state and traces were not bit-identical across a sleep visit")
        payload["check2"] = {
            "overlap": stored["overlap"],
            "exam_glances_drawn": stored["exam_glances_drawn"],
            "sleep_decisions": stored["sleep_decisions"],
            "sleep_checks": stored["sleep_checks"],
            "sleep_state_intact": stored["sleep_state_intact"],
            "sleep_param_delta": stored["sleep_param_delta"],
            "replay_draws": stored["replay_draws"],
        }

        log("smoke check 3: age_sleeparm seed 24 L=512 visit trace")
        traced = _job((24, "age_sleeparm", 512, True, True, None))
        if not traced["finite"]:
            raise RuntimeError("timed age_sleeparm T14 is not finite")
        for visit in traced["visits"]:
            log(format_visit(visit))
        quarters = traced["sleep_quarters"]
        log(
            "  sleep share by quarter "
            + " ".join(f"{q:.6f}" for q in quarters)
        )
        log(
            f"  sleep share overall {traced['sleep_share']:.6f} "
            f"decisions {traced['sleep_decisions']} visits {traced['n_sleep_visits']} "
            f"T14 {traced['T14']:.6f} W14 {traced['W14']:.6f} "
            f"wall {traced['wall_s']:.2f}s learn {traced['learn_s']:.2f}s "
            f"score {traced['score_s']:.2f}s eval {traced['eval_s']:.2f}s"
        )
        fixed_row = {
            "L": 256,
            "learn_s": check1["256"]["learn_s"],
            "score_s": check1["256"]["score_s"],
            "eval_s": check1["256"]["eval_s"],
        }
        est_sleeparm = _scale_one(traced, L_SCORE)
        est_fixed = _scale_one(fixed_row, L_SCORE)
        share = float(traced["sleep_share"])
        wave = max(est_sleeparm, est_fixed) * max(1.0, share / SLEEP_SHARE)
        kernel_s = est_sleeparm + wave
        guarded = kernel_s * 2.0
        log(
            f"  one L2048 age_sleeparm {est_sleeparm:.1f}s age_fixed20 {est_fixed:.1f}s "
            f"wave {wave:.1f}s kernel {kernel_s:.1f}s guarded {guarded:.1f}s "
            f"limit {KERNEL_WALL_LIMIT_S:.0f}s"
        )
        if kernel_s > KERNEL_WALL_LIMIT_S:
            raise RuntimeError(
                f"kernel estimate {kernel_s:.0f}s exceeds {KERNEL_WALL_LIMIT_S:.0f}s"
            )
        payload["check3"] = {
            "T14": traced["T14"],
            "W14": traced["W14"],
            "sleep_share": traced["sleep_share"],
            "sleep_quarters": quarters,
            "sleep_decisions": traced["sleep_decisions"],
            "n_sleep_visits": traced["n_sleep_visits"],
            "wall_s": traced["wall_s"],
            "learn_s": traced["learn_s"],
            "score_s": traced["score_s"],
            "eval_s": traced["eval_s"],
            "visits": traced["visits"],
        }
        payload["estimate"] = {
            "sleeparm_s": float(est_sleeparm),
            "fixed20_s": float(est_fixed),
            "wave_s": float(wave),
            "kernel_s": float(kernel_s),
            "guarded_s": float(guarded),
            "limit_s": KERNEL_WALL_LIMIT_S,
            "workers": KERNEL_WORKERS,
        }

        log("smoke check 4: four controllers, seed 24, L=128, finite T14")
        small_arm = _job((24, "age_sleeparm", 128, True, False, None))
        small_jobs = [
            (24, "age_nosleep", 128, True, False, None),
            (24, "age_fixed20", 128, True, False, None),
            (24, "age_fixed_matched", 128, True, False, small_arm["sleep_decisions"]),
        ]
        small_rest = _pool_map(small_jobs, _job, workers)
        small = {
            "age_sleeparm": small_arm,
            "age_nosleep": small_rest[0],
            "age_fixed20": small_rest[1],
            "age_fixed_matched": small_rest[2],
        }
        check4 = {}
        for controller in CONTROLLERS:
            row = small[controller]
            log(
                f"  {controller} T14 {row['T14']:.6f} W14 {row['W14']:.6f} "
                f"finite {row['finite']} sleep {row['sleep_share']:.4f} "
                f"decisions {row['sleep_decisions']} corridor {row['corridor_share']:.4f} "
                f"swaps {len(row['swap_log'])} wall {row['wall_s']:.2f}s"
            )
            if row["controller"] != controller or not row["finite"]:
                raise RuntimeError(f"{controller} T14 is not finite")
            if len(row["checkpoints"]) != N_CHECKPOINTS:
                raise RuntimeError(f"{controller} did not score 8 checkpoints")
            if row["swap_log"] != small_arm["swap_log"]:
                raise RuntimeError("controllers disagree on the L=128 swap log")
            if row["overlap"] != 0 or row["exam_glances_drawn"] != 0:
                raise RuntimeError(f"{controller} stores overlapped")
            check4[controller] = {"T14": row["T14"], "W14": row["W14"], "finite": True}
        expect = matched_dose(small_arm["sleep_decisions"], 128)
        if small["age_fixed_matched"]["sleep_decisions"] != expect:
            raise RuntimeError("age_fixed_matched did not take the rounded dose")
        if small["age_nosleep"]["sleep_decisions"] != 0:
            raise RuntimeError("age_nosleep slept")
        if small["age_fixed20"]["sleep_decisions"] != w4.sleep_decisions(128):
            raise RuntimeError("age_fixed20 dose drifted")
        payload["check4"] = check4
        payload["ok"] = True
    except Exception as exc:
        payload["ok"] = False
        payload["reason"] = str(exc)
        log(f"SMOKE_FAIL {exc}")
    payload["wall_s"] = time.perf_counter() - t0
    payload["log"] = lines
    m7.dump(out_dir / "smoke.json", payload)
    (out_dir / "smoke.log").write_text("\n".join(lines) + "\n")
    if not payload["ok"]:
        print(f"SMOKE_FAIL {payload['reason']}. Kernels not pushed.", flush=True)
        raise SystemExit(2)
    print(f"smoke_ok wall {payload['wall_s']:.1f}s", flush=True)
    return payload


def smoke_check1_pair(workers, life_L):
    jobs_out = _pool_map(
        [
            ("m9", 12, "age_fixed20", life_L, True, False, None),
            ("m8", 12, life_L),
        ],
        _either_job,
        workers,
    )
    by = {}
    for kind, row in jobs_out:
        by[kind] = row
    return by


def _either_job(args):
    kind = args[0]
    _threads()
    if kind == "m8":
        _tag, seed, L = args
        print(f"s{seed} prog_age_s p={PERIOD} L={L} start dose {w4.sleep_decisions(L)}", flush=True)
        row = m8.run_life(seed, "prog_age_s", L, score=True, check_sleep=True)
        return "m8", row
    if kind == "m9":
        row = _job(args[1:])
        return "m9", row
    raise ValueError(kind)


def _write_kernel(directory, seed):
    directory.mkdir(parents=True, exist_ok=True)
    slug = f"m9-w5-s{seed}"
    kernel_id = f"ezzzzzx/{slug}"
    meta = {
        "id": kernel_id,
        "title": slug,
        "code_file": "m9_kernel.py",
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
    text = f'''"""M9-W5 seed {seed}. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_M9_KERNEL") != "1":
    os.environ["_M9_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("m9_w5.py"))
    if not hits:
        raise SystemExit("m9_w5.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import m9_w5
    m9_w5.kernel_main(
        seed={seed},
        out="/kaggle/working",
        workers=3,
        kernel_id="{kernel_id}",
    )


if __name__ == "__main__":
    main()
'''
    (directory / "m9_kernel.py").write_text(text)
    return kernel_id


def stage(smoke_payload, kaggle_dir=None):
    """Write the private dataset folder and one kernel folder per seed. Does not push."""
    if not smoke_payload.get("ok"):
        raise SystemExit("smoke did not pass; kernels not staged")
    kaggle_dir = Path(kaggle_dir or (ROOT / "kaggle" / "m9w5"))
    if kaggle_dir.exists():
        shutil.rmtree(kaggle_dir)
    code = kaggle_dir / "code"
    code.mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "pdp code m9w5",
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


def kernel_main(seed, out, workers=3, kernel_id=None):
    if __name__ != "m9_w5":
        raise SystemExit("import m9_w5 and call kernel_main so the pool can spawn workers")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 24-35")
    workers = int(workers)
    if workers < 1 or workers > KERNEL_WORKERS:
        raise SystemExit("kernel pool is at most 3")
    _threads()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"kernel {kernel_id} seed {seed} workers {workers} controllers {len(CONTROLLERS)}",
        flush=True,
    )
    t0 = time.perf_counter()
    sleeparm = _job((seed, "age_sleeparm", L_SCORE, True, False, None))
    slept = int(sleeparm["sleep_decisions"])
    print(f"s{seed} age_sleeparm slept {slept} matched dose {matched_dose(slept, L_SCORE)}", flush=True)
    jobs = [
        (seed, "age_nosleep", L_SCORE, True, False, None),
        (seed, "age_fixed20", L_SCORE, True, False, None),
        (seed, "age_fixed_matched", L_SCORE, True, False, slept),
    ]
    if workers == 1:
        others = [_job(job) for job in jobs]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(workers) as pool:
            others = [item.get() for item in [pool.apply_async(_job, (job,)) for job in jobs]]
    wall = time.perf_counter() - t0
    rows = [sleeparm, *others]
    if len(rows) != len(CONTROLLERS):
        raise SystemExit(f"expected {len(CONTROLLERS)} rows, got {len(rows)}")
    for row in rows:
        _check_row(row, seed, row["controller"], L_SCORE)
    by_name = {row["controller"]: row for row in rows}
    if len(by_name) != len(CONTROLLERS):
        raise SystemExit("duplicate controller rows")
    ordered = [by_name[name] for name in CONTROLLERS]
    if by_name["age_sleeparm"]["sleep_decisions"] != slept:
        raise SystemExit("sleeparm sleep count changed")
    if by_name["age_fixed_matched"]["matched_slept"] != slept:
        raise SystemExit("matched life did not see this seed's sleep count")
    if by_name["age_fixed_matched"]["sleep_decisions"] != matched_dose(slept, L_SCORE):
        raise SystemExit("matched dose drifted")
    logs = [row["swap_log"] for row in ordered]
    if any(log != logs[0] for log in logs):
        raise SystemExit("controllers disagree on the swap log")
    orders = [row["room_order"] for row in ordered]
    if any(order != orders[0] for order in orders):
        raise SystemExit("controllers disagree on the ring order")
    payload = {
        "kernel_id": kernel_id,
        "seed": seed,
        "controllers": list(CONTROLLERS),
        "workers": workers,
        "wall_s": wall,
        "torch": torch.__version__,
        "model_seed": m7.MODEL_SEED,
        "n_ch": w4.N_CH,
        "L": L_SCORE,
        "period": PERIOD,
        "tau": TAU,
        "sleep_share_target": SLEEP_SHARE,
        "sleeparm_sleep_decisions": slept,
        "matched_dose": matched_dose(slept, L_SCORE),
        "room_order": ordered[0]["room_order"],
        "runs": ordered,
    }
    path = out / f"m9_s{seed}.json"
    m7.dump(path, payload)
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    return payload


def _check_row(row, seed, controller, L):
    if row["seed"] != int(seed) or row["controller"] != controller or row["L"] != int(L):
        raise RuntimeError("row does not match its job")
    if row["period"] != PERIOD or row["tau"] != TAU:
        raise RuntimeError("row period or tau drifted")
    if not row["finite"]:
        raise RuntimeError(f"{controller} seed {seed} is not finite")
    if row["overlap"] != 0 or row["exam_glances_drawn"] != 0:
        raise RuntimeError("stores overlapped")
    if row["n_ch"] != w4.N_CH:
        raise RuntimeError("channel width drifted")
    if [rec["decision"] for rec in row["checkpoints"]] != w5.checkpoints_of(L):
        raise RuntimeError("checkpoints drifted")
    if row["swap_log"] != w5.expected_swap_log(seed, PERIOD, L):
        raise RuntimeError("swap log drifted")
    if len(row["latency"]) != len(row["swap_log"]):
        raise RuntimeError("latency rows do not match the swap log")
    if abs((sum(row["share"].values()) + row["corridor_share"] + row["sleep_share"]) - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    if abs(row["sleep_share"] - row["sleep_decisions"] / float(L)) > 1e-12:
        raise RuntimeError("sleep share drifted")
    if len(row["sleep_quarters"]) != 4:
        raise RuntimeError("sleep quarters drifted")
    if abs(sum(row["sleep_quarters"]) / 4.0 - row["sleep_share"]) > 1e-9:
        raise RuntimeError("quarter shares do not average to the life share")
    if controller == "age_nosleep":
        if row["sleep_decisions"] != 0 or row["replay_draws"] != 0:
            raise RuntimeError("age_nosleep slept or drew replay")
    elif controller == "age_fixed20":
        if row["sleep_decisions"] != w4.sleep_decisions(L):
            raise RuntimeError("age_fixed20 dose drifted")
    elif controller == "age_fixed_matched":
        if row["matched_slept"] is None:
            raise RuntimeError("age_fixed_matched has no source count")
        if row["sleep_decisions"] != matched_dose(row["matched_slept"], L):
            raise RuntimeError("age_fixed_matched dose drifted")
    elif controller == "age_sleeparm":
        if row["n_sleep_visits"] < 0:
            raise RuntimeError("sleeparm visit count")
        if len(row["reward_quarters"]) != 4:
            raise RuntimeError("sleeparm reward quarters drifted")
    else:
        raise RuntimeError(controller)
    if row["sleep_decisions"] > 0 and row["sleep_state_intact"] is not True:
        raise RuntimeError("a sleeping life did not keep the live state")


def mean_std(values):
    return m7.mean_std(values)


def _fmt(value, digits=6):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _pm(mu, sd, digits=6):
    return f"{_fmt(mu, digits)} ± {_fmt(sd, digits)}"


def _pick(runs, controller, seed):
    hit = [row for row in runs if row["controller"] == controller and row["seed"] == seed]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {controller} seed {seed}, got {len(hit)}")
    return hit[0]


def tail_t14(row):
    vals = [float(rec["W14"]) for rec in row["checkpoints"] if int(rec["decision"]) != DROP_DECISION]
    if not any(int(rec["decision"]) == DROP_DECISION for rec in row["checkpoints"]):
        raise RuntimeError("checkpoint 256 is missing")
    if len(vals) != N_CHECKPOINTS - 1:
        raise RuntimeError("tail is not the seven checkpoints after 256")
    return float(np.mean(vals))


def _latency_summary(rows):
    if not rows:
        raise RuntimeError("no swaps")
    values = [int(row["latency"]) for row in rows]
    within = int(sum(bool(row["within_128"]) for row in rows))
    censored = int(sum(bool(row["censored"]) for row in rows))
    return {
        "n": len(rows),
        "median": float(np.median(np.asarray(values, dtype=np.float64))),
        "within_128": within,
        "share_within_128": float(within / len(rows)),
        "censored": censored,
        "found": int(len(rows) - censored),
    }


def _gap_block(values):
    mu, sd = mean_std(values)
    return {
        "mean": mu,
        "std_ddof0": sd,
        "left_lower": int(sum(value < 0.0 for value in values)),
        "n": len(values),
    }


def _reward_quarter_mean(rows):
    out = []
    for q in range(4):
        sleep_means = []
        room_means = []
        sleep_n = 0
        room_n = 0
        for row in rows:
            slot = row["reward_quarters"][q]
            sleep_n += int(slot["sleep_n"])
            room_n += int(slot["room_n"])
            if slot["sleep_mean"] is not None:
                sleep_means.append(float(slot["sleep_mean"]))
            if slot["room_mean"] is not None:
                room_means.append(float(slot["room_mean"]))
        out.append({
            "sleep_n": int(sleep_n),
            "sleep_mean": _mean_or_none(sleep_means),
            "room_n": int(room_n),
            "room_mean": _mean_or_none(room_means),
        })
    return out


def m8_reference():
    path = ROOT / "results" / "m8w5" / "results.json"
    data = json.loads(path.read_text())
    by = {}
    for name in m8.CONTROLLERS:
        block = data["summary"]["by_controller"][name]
        by[name] = {
            "T14_mean": float(block["T14_mean"]),
            "T14_std_ddof0": float(block["T14_std_ddof0"]),
            "W14_mean": float(block["W14_mean"]),
            "W14_std_ddof0": float(block["W14_std_ddof0"]),
        }
    per = []
    for row in data["runs"]:
        if row["controller"] == "prog_age_s":
            per.append({
                "seed": int(row["seed"]),
                "T14": float(row["T14"]),
                "W14": float(row["W14"]),
            })
    per.sort(key=lambda item: item["seed"])
    return {
        "source": "results/m8w5/results.json",
        "seeds": list(m8.SEEDS),
        "L": int(data["L"]),
        "by_controller": by,
        "prog_age_s": per,
    }


def summarize(runs):
    by = {}
    for controller in CONTROLLERS:
        rows = [_pick(runs, controller, seed) for seed in SEEDS]
        t14_mu, t14_sd = mean_std([row["T14"] for row in rows])
        w14_mu, w14_sd = mean_std([row["W14"] for row in rows])
        tail = [tail_t14(row) for row in rows]
        tail_mu, tail_sd = mean_std(tail)
        share_mean = {}
        share_std = {}
        for name in IDENTITIES:
            mu, sd = mean_std([row["share"][name] for row in rows])
            share_mean[name] = mu
            share_std[name] = sd
        corridor_mu, corridor_sd = mean_std([row["corridor_share"] for row in rows])
        sleep_mu, sleep_sd = mean_std([row["sleep_share"] for row in rows])
        points = w5.checkpoints_of(L_SCORE)
        checkpoints = []
        for decision in points:
            vals = []
            for row in rows:
                hit = [rec for rec in row["checkpoints"] if rec["decision"] == decision]
                if len(hit) != 1:
                    raise RuntimeError(f"checkpoint {decision} missing")
                vals.append(hit[0]["W14"])
            mu, sd = mean_std(vals)
            checkpoints.append({"decision": int(decision), "W14_mean": mu, "W14_std_ddof0": sd})
        quarter_mean = []
        quarter_std = []
        for q in range(4):
            mu, sd = mean_std([row["sleep_quarters"][q] for row in rows])
            quarter_mean.append(mu)
            quarter_std.append(sd)
        lat_rows = [item for row in rows for item in row["latency"]]
        block = {
            "controller": controller,
            "n": len(rows),
            "T14_mean": t14_mu,
            "T14_std_ddof0": t14_sd,
            "W14_mean": w14_mu,
            "W14_std_ddof0": w14_sd,
            "tail_T14_mean": tail_mu,
            "tail_T14_std_ddof0": tail_sd,
            "share_mean": share_mean,
            "share_std_ddof0": share_std,
            "corridor_mean": corridor_mu,
            "corridor_std_ddof0": corridor_sd,
            "sleep_mean": sleep_mu,
            "sleep_std_ddof0": sleep_sd,
            "sleep_quarter_mean": quarter_mean,
            "sleep_quarter_std_ddof0": quarter_std,
            "checkpoints": checkpoints,
            "latency": _latency_summary(lat_rows),
        }
        if controller in SLEEPERS:
            draws = [row["replay_draws"] for row in rows]
            stale = [row["stale_draws"] for row in rows]
            shares = [row["stale_share"] for row in rows]
            share_mu, share_sd = mean_std(shares)
            block["replay_draws"] = int(sum(draws))
            block["stale_draws"] = int(sum(stale))
            block["stale_share_pooled"] = float(sum(stale) / sum(draws)) if sum(draws) else None
            block["stale_share_mean"] = share_mu
            block["stale_share_std_ddof0"] = share_sd
        if controller == "age_sleeparm":
            block["reward_quarters"] = _reward_quarter_mean(rows)
            sleep_gaps = [gap for row in rows for gap in row["since_swap_sleep"]]
            room_gaps = [gap for row in rows for gap in row["since_swap_room"]]
            block["since_swap_sleep_mean"] = _mean_or_none(sleep_gaps)
            block["since_swap_room_mean"] = _mean_or_none(room_gaps)
            block["since_swap_sleep_n"] = len(sleep_gaps)
            block["since_swap_room_n"] = len(room_gaps)
            seed_sleep = [row["since_swap_sleep_mean"] for row in rows if row["since_swap_sleep_mean"] is not None]
            seed_room = [row["since_swap_room_mean"] for row in rows if row["since_swap_room_mean"] is not None]
            mu_s, sd_s = mean_std(seed_sleep)
            mu_r, sd_r = mean_std(seed_room)
            block["since_swap_sleep_mean_of_seeds"] = mu_s
            block["since_swap_sleep_std_of_seeds"] = sd_s
            block["since_swap_room_mean_of_seeds"] = mu_r
            block["since_swap_room_std_of_seeds"] = sd_r
        by[controller] = block
    gaps = []
    gaps_tail = []
    for seed in SEEDS:
        nosleep = _pick(runs, "age_nosleep", seed)
        fixed = _pick(runs, "age_fixed20", seed)
        arm = _pick(runs, "age_sleeparm", seed)
        matched = _pick(runs, "age_fixed_matched", seed)
        gaps.append({
            "seed": int(seed),
            "age_fixed20_minus_age_nosleep": float(fixed["T14"] - nosleep["T14"]),
            "age_sleeparm_minus_age_nosleep": float(arm["T14"] - nosleep["T14"]),
            "age_sleeparm_minus_age_fixed_matched": float(arm["T14"] - matched["T14"]),
            "age_sleeparm_minus_age_fixed20": float(arm["T14"] - fixed["T14"]),
        })
        gaps_tail.append({
            "seed": int(seed),
            "age_fixed20_minus_age_nosleep": float(tail_t14(fixed) - tail_t14(nosleep)),
            "age_sleeparm_minus_age_nosleep": float(tail_t14(arm) - tail_t14(nosleep)),
            "age_sleeparm_minus_age_fixed_matched": float(tail_t14(arm) - tail_t14(matched)),
            "age_sleeparm_minus_age_fixed20": float(tail_t14(arm) - tail_t14(fixed)),
        })
    gap_summary = {key: _gap_block([row[key] for row in gaps]) for key in GAP_KEYS}
    gap_tail_summary = {key: _gap_block([row[key] for row in gaps_tail]) for key in GAP_KEYS}
    latency_by_seed = {}
    for controller in CONTROLLERS:
        latency_by_seed[controller] = {}
        for seed in SEEDS:
            latency_by_seed[controller][str(seed)] = _latency_summary(
                _pick(runs, controller, seed)["latency"]
            )
    return {
        "by_controller": by,
        "gaps": gaps,
        "gap_summary": gap_summary,
        "gaps_without_decision_256": gaps_tail,
        "gap_summary_without_decision_256": gap_tail_summary,
        "latency_by_seed": latency_by_seed,
    }


def _header(labels):
    return "| " + " | ".join(labels) + " |"


def _sep(n):
    return "|" + "|".join(["---:" if i else "---" for i in range(n)]) + "|"


def _gap_table(lines, gaps, summary):
    lines.append(_header([
        "seed",
        "age_fixed20 − age_nosleep",
        "age_sleeparm − age_nosleep",
        "age_sleeparm − age_fixed_matched",
        "age_sleeparm − age_fixed20",
    ]))
    lines.append(_sep(5))
    for gap in gaps:
        lines.append(
            f"| {gap['seed']} | {_fmt(gap['age_fixed20_minus_age_nosleep'])} | "
            f"{_fmt(gap['age_sleeparm_minus_age_nosleep'])} | "
            f"{_fmt(gap['age_sleeparm_minus_age_fixed_matched'])} | "
            f"{_fmt(gap['age_sleeparm_minus_age_fixed20'])} |"
        )
    for label, key in (("mean", "mean"), ("std ddof=0", "std_ddof0")):
        lines.append(
            f"| {label} | {_fmt(summary['age_fixed20_minus_age_nosleep'][key])} | "
            f"{_fmt(summary['age_sleeparm_minus_age_nosleep'][key])} | "
            f"{_fmt(summary['age_sleeparm_minus_age_fixed_matched'][key])} | "
            f"{_fmt(summary['age_sleeparm_minus_age_fixed20'][key])} |"
        )
    lines.append(
        "| left lower (count) | "
        f"{summary['age_fixed20_minus_age_nosleep']['left_lower']} of {summary['age_fixed20_minus_age_nosleep']['n']} | "
        f"{summary['age_sleeparm_minus_age_nosleep']['left_lower']} of {summary['age_sleeparm_minus_age_nosleep']['n']} | "
        f"{summary['age_sleeparm_minus_age_fixed_matched']['left_lower']} of {summary['age_sleeparm_minus_age_fixed_matched']['n']} | "
        f"{summary['age_sleeparm_minus_age_fixed20']['left_lower']} of {summary['age_sleeparm_minus_age_fixed20']['n']} |"
    )


def render_markdown(result):
    summary = result["summary"]
    seeds = result["seeds"]
    runs = result["runs"]
    points = [item["decision"] for item in summary["by_controller"]["age_nosleep"]["checkpoints"]]
    dose = w4.sleep_decisions(L_SCORE)
    lines = [
        "# M9-W5 sleep timing on the drifting ring",
        "",
        "W5's 18-room ring with period 128. One learnable room's rule is redrawn every 128 decisions. "
        "T14 is the mean of W14 at decisions L/8, 2L/8, …, L. W14 is bits/symbol of the frozen bank "
        "on a fresh EVAL=2048 continuation of the 14 learnable rooms. The final W14 is the checkpoint at L. "
        "Lower is better. The tail leaves out the checkpoint at decision 256 and averages the other seven.",
        "",
        "Every controller uses prog_age_s's room choice: per-room exam progress, EPS 0.1, BETA 0.3, "
        f"and on the greedy step v + (v_max − v) · (1 − exp(−age / {TAU})). Age is the decisions since "
        "that arm's last visit ended. v_max is the largest visit reward so far. For age_sleeparm the visits "
        "include sleep, so v_max includes sleep rewards. The stored EMA is not aged. "
        f"`age_fixed20` is prog_age_s: {dose} sleep decisions, share {dose / L_SCORE:.6f}, placed as in M8. "
        "`age_nosleep` uses the same room choice and never sleeps. `age_sleeparm` is the same bandit over "
        "19 arms, the 18 rooms plus sleep. Sleep is last in the optimistic first pass. A sleep visit is 8 "
        "decisions and no travel. Its reward is the whole exam store's bits saved over those decisions, "
        "divided by the decisions in the visit (8, unless the life ends first). There is no forced sleep. "
        "`age_fixed_matched` is prog_age_s at the share age_sleeparm slept on that seed, rounded half up "
        "to a whole number of 8-decision visits, placed with the same plan and due rule as age_fixed20.",
        "",
        "A replay draw is stale when the room's rule at the draw is not the rule that produced the glance. "
        "The rule id is the latest swap index of that room, or −1 before its first swap. "
        "Decisions since the last swap use the latest swap whose decision is strictly earlier than the "
        "visit's first decision. Before the first swap the origin is decision 0. "
        "A swap's latency is the decisions until the next stay in that room. If the life ends first, the "
        "latency is L minus the swap decision and the swap is censored. within 128 counts a stay whose "
        "latency is at most 128. The median includes censored latencies.",
        "",
        "## T14",
        "",
        _header(["controller", *[f"seed {seed}" for seed in seeds], "mean ± std ddof=0"]),
        _sep(len(seeds) + 2),
    ]
    for controller in CONTROLLERS:
        block = summary["by_controller"][controller]
        cells = [_fmt(_pick(runs, controller, seed)["T14"]) for seed in seeds]
        lines.append(
            f"| {controller} | " + " | ".join(cells)
            + f" | {_pm(block['T14_mean'], block['T14_std_ddof0'])} |"
        )
    lines.extend([
        "",
        "## Final W14",
        "",
        _header(["controller", *[f"seed {seed}" for seed in seeds], "mean ± std ddof=0"]),
        _sep(len(seeds) + 2),
    ])
    for controller in CONTROLLERS:
        block = summary["by_controller"][controller]
        cells = [_fmt(_pick(runs, controller, seed)["W14"]) for seed in seeds]
        lines.append(
            f"| {controller} | " + " | ".join(cells)
            + f" | {_pm(block['W14_mean'], block['W14_std_ddof0'])} |"
        )
    lines.extend([
        "",
        "## T14 per checkpoint",
        "",
        _header(["controller", "seed", *[str(point) for point in points], "T14"]),
        _sep(len(points) + 3),
    ])
    for controller in CONTROLLERS:
        for seed in seeds:
            row = _pick(runs, controller, seed)
            cells = [_fmt(rec["W14"]) for rec in row["checkpoints"]]
            lines.append(
                f"| {controller} | {seed} | " + " | ".join(cells) + f" | {_fmt(row['T14'])} |"
            )
    lines.extend([
        "",
        "Mean over seeds at each checkpoint.",
        "",
        _header(["controller", *[str(point) for point in points], "T14"]),
        _sep(len(points) + 2),
    ])
    for controller in CONTROLLERS:
        block = summary["by_controller"][controller]
        cells = [_fmt(item["W14_mean"]) for item in block["checkpoints"]]
        lines.append(f"| {controller} | " + " | ".join(cells) + f" | {_fmt(block['T14_mean'])} |")
    lines.extend([
        "",
        "## T14 without decision 256",
        "",
        "Mean of the checkpoints at 512, 768, 1024, 1280, 1536, 1792 and 2048.",
        "",
        _header(["controller", *[f"seed {seed}" for seed in seeds], "mean ± std ddof=0"]),
        _sep(len(seeds) + 2),
    ])
    for controller in CONTROLLERS:
        block = summary["by_controller"][controller]
        cells = [_fmt(tail_t14(_pick(runs, controller, seed))) for seed in seeds]
        lines.append(
            f"| {controller} | " + " | ".join(cells)
            + f" | {_pm(block['tail_T14_mean'], block['tail_T14_std_ddof0'])} |"
        )
    lines.extend([
        "",
        "## Gaps",
        "",
        "On T14. A negative gap means the first controller is lower. left lower counts seeds where the gap is negative.",
        "",
    ])
    _gap_table(lines, summary["gaps"], summary["gap_summary"])
    lines.extend([
        "",
        "## Gaps without decision 256",
        "",
        "Same gaps on the tail mean. A negative gap means the first controller is lower.",
        "",
    ])
    _gap_table(lines, summary["gaps_without_decision_256"], summary["gap_summary_without_decision_256"])
    arm = summary["by_controller"]["age_sleeparm"]
    lines.extend([
        "",
        "## age_sleeparm sleep share",
        "",
        "Sleep decisions in the quarter divided by L/4. Quarter i is decisions [(i−1)L/4, iL/4).",
        "",
        _header(["seed", "overall", "quarter 1", "quarter 2", "quarter 3", "quarter 4", "sleep decisions", "matched dose"]),
        _sep(8),
    ])
    for seed in seeds:
        row = _pick(runs, "age_sleeparm", seed)
        matched = _pick(runs, "age_fixed_matched", seed)
        cells = [_fmt(row["sleep_share"]), *[_fmt(q) for q in row["sleep_quarters"]]]
        lines.append(
            f"| {seed} | " + " | ".join(cells)
            + f" | {row['sleep_decisions']} | {matched['sleep_decisions']} |"
        )
    lines.append(
        f"| mean ± std ddof=0 | {_pm(arm['sleep_mean'], arm['sleep_std_ddof0'])} | "
        + " | ".join(
            _pm(arm["sleep_quarter_mean"][q], arm["sleep_quarter_std_ddof0"][q]) for q in range(4)
        )
        + " |  |  |"
    )
    lines.extend([
        "",
        "## Sleep position relative to swaps",
        "",
        "Mean decisions from the last earlier swap to the visit onset. The mean column pools every onset. "
        "The mean-of-seeds row averages the per-seed means.",
        "",
        _header(["seed", "sleep onsets", "sleep mean", "room onsets", "room mean"]),
        _sep(5),
    ])
    for seed in seeds:
        row = _pick(runs, "age_sleeparm", seed)
        lines.append(
            f"| {seed} | {row['n_sleep_visits']} | {_fmt(row['since_swap_sleep_mean'])} | "
            f"{row['n_room_visits']} | {_fmt(row['since_swap_room_mean'])} |"
        )
    lines.append(
        f"| pooled | {arm['since_swap_sleep_n']} | {_fmt(arm['since_swap_sleep_mean'])} | "
        f"{arm['since_swap_room_n']} | {_fmt(arm['since_swap_room_mean'])} |"
    )
    lines.append(
        f"| mean of seeds |  | {_pm(arm['since_swap_sleep_mean_of_seeds'], arm['since_swap_sleep_std_of_seeds'])} | "
        f" | {_pm(arm['since_swap_room_mean_of_seeds'], arm['since_swap_room_std_of_seeds'])} |"
    )
    lines.extend([
        "",
        "## age_sleeparm visit rewards by quarter",
        "",
        "Exam bits saved per decision, mean over the visits whose first decision falls in the quarter. "
        "Sleep reward is the whole exam store. Room reward is that room's exam glances, travel included. "
        "A quarter with no visits of that kind is blank. The mean row's visit count is the sum across seeds. "
        "The mean reward is the mean of the per-seed means, over seeds that have at least one such visit.",
        "",
        _header(["seed", "quarter", "sleep visits", "sleep mean reward", "room visits", "room mean reward"]),
        _sep(6),
    ])
    for seed in seeds:
        row = _pick(runs, "age_sleeparm", seed)
        for q, slot in enumerate(row["reward_quarters"], start=1):
            lines.append(
                f"| {seed} | {q} | {slot['sleep_n']} | {_fmt(slot['sleep_mean'])} | "
                f"{slot['room_n']} | {_fmt(slot['room_mean'])} |"
            )
    for q, slot in enumerate(arm["reward_quarters"], start=1):
        lines.append(
            f"| mean | {q} | {slot['sleep_n']} | {_fmt(slot['sleep_mean'])} | "
            f"{slot['room_n']} | {_fmt(slot['room_mean'])} |"
        )
    lines.extend([
        "",
        "## Stale replay",
        "",
        "Share of replay draws whose glance came from a rule that had already been swapped out. "
        "age_nosleep does not draw.",
        "",
        _header(["controller", "seed", "draws", "stale", "share"]),
        _sep(5),
    ])
    for controller in SLEEPERS:
        for seed in seeds:
            row = _pick(runs, controller, seed)
            lines.append(
                f"| {controller} | {seed} | {row['replay_draws']} | {row['stale_draws']} | "
                f"{_fmt(row['stale_share'])} |"
            )
        block = summary["by_controller"][controller]
        lines.append(
            f"| {controller} | pooled | {block['replay_draws']} | {block['stale_draws']} | "
            f"{_fmt(block['stale_share_pooled'])} |"
        )
        lines.append(
            f"| {controller} | mean ± std ddof=0 |  |  | "
            f"{_pm(block['stale_share_mean'], block['stale_share_std_ddof0'])} |"
        )
    lines.extend([
        "",
        "## Swap-finding latency",
        "",
        "Median over swaps. Censored swaps use L minus the swap decision. Share within 128 is the fraction of swaps with a later stay at most 128 decisions after the swap.",
        "",
        _header(["controller", "n", "found", "censored", "median", "share within 128"]),
        _sep(6),
    ])
    for controller in CONTROLLERS:
        lat = summary["by_controller"][controller]["latency"]
        lines.append(
            f"| {controller} | {lat['n']} | {lat['found']} | {lat['censored']} | "
            f"{_fmt(lat['median'])} | {_fmt(lat['share_within_128'])} |"
        )
    lines.extend([
        "",
        _header(["controller", "seed", "n", "found", "censored", "median", "share within 128"]),
        _sep(7),
    ])
    for controller in CONTROLLERS:
        for seed in seeds:
            lat = summary["latency_by_seed"][controller][str(seed)]
            lines.append(
                f"| {controller} | {seed} | {lat['n']} | {lat['found']} | {lat['censored']} | "
                f"{_fmt(lat['median'])} | {_fmt(lat['share_within_128'])} |"
            )
    lines.extend([
        "",
        _header(["controller", "seed", "decision", "room", "k", "next dwell", "latency", "censored", "within 128"]),
        _sep(9),
    ])
    for controller in CONTROLLERS:
        for seed in seeds:
            for item in _pick(runs, controller, seed)["latency"]:
                nxt = "" if item["next_dwell"] is None else str(item["next_dwell"])
                lines.append(
                    f"| {controller} | {seed} | {item['decision']} | {item['room']} | {item['k']} | "
                    f"{nxt} | {item['latency']} | {str(bool(item['censored'])).lower()} | "
                    f"{str(bool(item['within_128'])).lower()} |"
                )
    share_names = list(IDENTITIES) + ["corridor", "sleep"]
    lines.extend([
        "",
        "## Decision shares",
        "",
        "Mean over seeds. A stay counts for the room. A move counts for the corridor. A sleep decision counts for sleep. Shares sum to 1.",
        "",
        _header(["controller", *share_names]),
        _sep(len(share_names) + 1),
    ])
    for controller in CONTROLLERS:
        block = summary["by_controller"][controller]
        cells = [_fmt(block["share_mean"][name], 4) for name in IDENTITIES]
        cells.append(_fmt(block["corridor_mean"], 4))
        cells.append(_fmt(block["sleep_mean"], 4))
        lines.append(f"| {controller} | " + " | ".join(cells) + " |")
    lines.extend(["", _header(["controller", "seed", *share_names]), _sep(len(share_names) + 2)])
    for controller in CONTROLLERS:
        for seed in seeds:
            row = _pick(runs, controller, seed)
            cells = [_fmt(row["share"][name], 4) for name in IDENTITIES]
            cells.append(_fmt(row["corridor_share"], 4))
            cells.append(_fmt(row["sleep_share"], 4))
            lines.append(f"| {controller} | {seed} | " + " | ".join(cells) + " |")
    ref = result["m8_reference"]
    lines.extend([
        "",
        "## M8 reference",
        "",
        f"M8-W5, period 128, seeds {ref['seeds'][0]}–{ref['seeds'][-1]}, L = {ref['L']}. "
        f"Source: {ref['source']}. prog_age_s is the agent age_fixed20 copies.",
        "",
        _header(["controller", "T14 mean", "T14 std ddof=0", "W14 mean", "W14 std ddof=0"]),
        _sep(5),
    ])
    for name in m8.CONTROLLERS:
        block = ref["by_controller"][name]
        lines.append(
            f"| {name} | {_fmt(block['T14_mean'])} | {_fmt(block['T14_std_ddof0'])} | "
            f"{_fmt(block['W14_mean'])} | {_fmt(block['W14_std_ddof0'])} |"
        )
    lines.extend([
        "",
        _header(["seed", "prog_age_s T14", "prog_age_s W14"]),
        _sep(3),
    ])
    for item in ref["prog_age_s"]:
        lines.append(f"| {item['seed']} | {_fmt(item['T14'])} | {_fmt(item['W14'])} |")
    lines.extend([
        "",
        "## Kernels",
        "",
        _header(["kernel", "seed", "wall s", "workers"]),
        _sep(4),
    ])
    for item in result["kernels"]:
        lines.append(
            f"| {item['kernel_id']} | {item['seed']} | {_fmt(item['wall_s'], 1)} | {item['workers']} |"
        )
    walls = [item["wall_s"] for item in result["kernels"]]
    if walls:
        mu, sd = mean_std(walls)
        lines.append(f"| mean ± std ddof=0 |  | {_pm(mu, sd, 1)} |  |")
    lines.extend([
        "",
        "## Smoke test",
        "",
        "```",
        *result["smoke_log"],
        "```",
        "",
    ])
    smoke_wall = result.get("smoke", {}).get("wall_s")
    if smoke_wall is not None:
        lines.append(f"Smoke wall {float(smoke_wall):.1f}s.")
        lines.append("")
    if result.get("kernel_status"):
        lines.extend(["```", result["kernel_status"].rstrip(), "```", ""])
    lines.extend([
        "## Protocol",
        "",
        "- World5(world_seed, period=128). Room names, ring order, noise rooms and Z are W4's. "
        "After every positive multiple of 128 decisions, one of the 14 learnable rooms is redrawn "
        "with default_rng(world_seed + 80000), as in W5.",
        "- The cortex, disjoint exam and replay reservoirs, coin, movement and dwell are M8's, which are W5's. "
        "World steps use default_rng(1000 + world_seed). Choices use default_rng(10000 + world_seed). "
        "A tie on the ring goes left. Equal distances among unvisited rooms break toward +index (right). Dwell is 8 stays.",
        "- One sleep decision draws one replay glance and takes one trace-learning Adam step from a zero state. "
        "The live state and traces are saved and restored. The first sleep visit of a life checks that they "
        "are bit-identical afterwards. Sleep draws no exam glance.",
        "- age_fixed20 calls the same room choice, sleep plan and sleep visit as M8 prog_age_s. "
        f"The dose at L = {L_SCORE} is {dose} decisions.",
        "- age_sleeparm's optimistic first pass visits the nearest unvisited room, sleep last. "
        "Then it is eps-greedy over the 19 arms. The greedy rank uses the aged value. "
        "A room visit's reward is M8's per-room exam progress, travel included, divided by the visit's decisions. "
        "A sleep visit's reward is the sum, over its decisions, of whole-exam-store bits before minus bits after, "
        "divided by the number of decisions in the visit.",
        "- age_fixed_matched rounds age_sleeparm's sleep decisions half up onto a multiple of 8 "
        "(floor(n/8 + 1/2) visits) and places those visits with m7.sleep_plan and w4.sleep_due, as age_fixed20 does.",
        "- A checkpoint scores w4.evaluate_w14 after that decision's swap. T14 is the mean of the eight checkpoints. "
        "The tail mean drops the checkpoint at decision 256.",
        "- Stale draws, sleep quarters, swap gaps and visit rewards are defined in the sections above.",
        f"- World seeds {seeds[0]}–{seeds[-1]}. L = {L_SCORE}. Period 128. CPU. "
        "Private Kaggle kernels, one seed each. age_sleeparm runs first, then the other three in a pool of 3. "
        "No GPU, no internet. Nothing retuned.",
        "",
    ])
    return "\n".join(lines) + "\n"


def _load_kernel_payloads(out_dir):
    out_dir = Path(out_dir)
    found = {}
    for seed in SEEDS:
        hits = sorted(out_dir.glob(f"s{seed}/m9_s{seed}.json"))
        if not hits:
            raise RuntimeError(f"no results file for seed {seed}")
        if len(hits) != 1:
            raise RuntimeError(f"more than one results file for seed {seed}")
        found[seed] = hits[0]
    return found


def assemble(out_dir, smoke_payload=None):
    out_dir = Path(out_dir)
    if smoke_payload is None:
        smoke_payload = json.loads((out_dir / "smoke.json").read_text())
    if not smoke_payload.get("ok"):
        raise RuntimeError("smoke did not pass")
    runs = []
    kernels = []
    for seed, path in _load_kernel_payloads(out_dir).items():
        payload = json.loads(path.read_text())
        if payload["seed"] != seed:
            raise RuntimeError(f"{path} seed {payload['seed']} != {seed}")
        if payload["controllers"] != list(CONTROLLERS):
            raise RuntimeError(f"{path} controllers drifted")
        for row in payload["runs"]:
            _check_row(row, seed, row["controller"], row["L"])
            runs.append(row)
        kernels.append({
            "kernel_id": payload["kernel_id"],
            "seed": seed,
            "wall_s": payload["wall_s"],
            "workers": payload["workers"],
            "path": str(path.resolve().relative_to(ROOT)),
        })
        logs = [row["swap_log"] for row in payload["runs"]]
        if any(log != logs[0] for log in logs):
            raise RuntimeError(f"controllers disagree on the swap log, seed {seed}")
        slept = int(payload["sleeparm_sleep_decisions"])
        matched = [row for row in payload["runs"] if row["controller"] == "age_fixed_matched"][0]
        if matched["sleep_decisions"] != int(payload["matched_dose"]):
            raise RuntimeError(f"seed {seed} matched dose does not match the payload")
        if matched_dose(slept, L_SCORE) != int(payload["matched_dose"]):
            raise RuntimeError(f"seed {seed} matched dose does not match the sleep count")
    if len(runs) != len(SEEDS) * len(CONTROLLERS):
        raise RuntimeError(f"expected {len(SEEDS) * len(CONTROLLERS)} runs, got {len(runs)}")
    summary = summarize(runs)
    kernels.sort(key=lambda item: item["seed"])
    status_path = out_dir / "kernel_status.txt"
    smoke_keep = {}
    for key, value in smoke_payload.items():
        if key == "log":
            continue
        if key == "check3" and isinstance(value, dict):
            smoke_keep[key] = {k: v for k, v in value.items() if k != "visits"}
        else:
            smoke_keep[key] = value
    result = {
        "seeds": list(SEEDS),
        "controllers": list(CONTROLLERS),
        "period": PERIOD,
        "L": L_SCORE,
        "tau": TAU,
        "model_seed": m7.MODEL_SEED,
        "n_ch": w4.N_CH,
        "sleep_share_target": SLEEP_SHARE,
        "sleep_decisions_fixed20": w4.sleep_decisions(L_SCORE),
        "dataset_id": DATASET_ID,
        "kernels": kernels,
        "summary": summary,
        "m8_reference": m8_reference(),
        "smoke": smoke_keep,
        "smoke_log": smoke_payload["log"],
        "kernel_status": status_path.read_text() if status_path.exists() else "",
        "runs": runs,
    }
    text = render_markdown(result)
    m7.dump(out_dir / "results.json", result)
    (out_dir / "RESULTS.md").write_text(text)
    print(f"wrote {out_dir / 'RESULTS.md'} and results.json", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "stage", "assemble"))
    parser.add_argument("--out", default=str(ROOT / "results" / "m9w5"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "smoke":
        payload = smoke(args.out, workers=args.workers)
        stage(payload)
        return
    smoke_payload = json.loads((Path(args.out) / "smoke.json").read_text())
    if args.command == "stage":
        stage(smoke_payload)
        return
    assemble(args.out, smoke_payload)


if __name__ == "__main__":
    main()

