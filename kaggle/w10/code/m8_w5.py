"""M8-W5: per-room curiosity on W5's drifting ring, with M7b's fixed sleep.

Numbers only. No verdict. Imports w5_headroom and does not modify it.
The cortex, sleep, stores, movement, dwell, T14 and the final W14 are W5's.
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
    if os.environ.get("_M8_MAIN") != "1":
        os.environ["_M8_MAIN"] = "1"
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
import w4_headroom as w4
import w4_world
import w5_headroom as w5
from w4_world import IDENTITIES, LEARN14, N_RING
from w5_world import World5

SEEDS = tuple(range(12, 24))
CONTROLLERS = ("rand_dwell_s", "oracle_fresh_s", "prog_chan_s", "prog_age_s")
FIXED = ("rand_dwell_s", "oracle_fresh_s")
BANDITS = ("prog_chan_s", "prog_age_s")
PERIOD = 128
L_SCORE = 2048
TAU = 256
N_CHECKPOINTS = 8
SLEEP_SHARE = 0.20
KERNEL_WALL_LIMIT_S = 90 * 60
BITS_LIMIT = 1e-6
KERNEL_WORKERS = 4
ROOT = Path(__file__).resolve().parent
CODE_FILES = (
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
DATASET_ID = "ezzzzzx/pdp-code-m8w5"
PUBLISHED_L256 = {
    "rand_dwell_s": {"T14": 10.444404, "W14": 6.268811},
    "oracle_fresh_s": {"T14": 7.876722, "W14": 3.486887},
}

if curio.EPS != 0.1 or curio.BETA != 0.3:
    raise RuntimeError(f"bandit constants drifted: EPS {curio.EPS} BETA {curio.BETA}")
if m7.DWELL != 8 or curio.B != 64 or curio.EVAL != 2048:
    raise RuntimeError("M8 inherited constants drifted")
if w4.N_CH != N_RING + 1 or N_RING != 18:
    raise RuntimeError("W4 ring width drifted")


def _threads():
    w4._threads()


def aged_value(v, age, v_max, tau=TAU):
    """v + (v_max - v) * (1 - exp(-age / tau)). Age 0 returns v."""
    gap = float(v_max) - float(v)
    return float(v) + gap * (1.0 - math.exp(-float(age) / float(tau)))


def _exam_bits(life, glances, channel_index):
    """Total bits of these already-stored glances whose channel is channel_index."""
    selected = [g for g in glances if int(g.channel) == int(channel_index)]
    if not selected:
        return 0.0
    t0 = time.perf_counter()
    values = m7.score_glances(life.cortex.model, life.cortex.channel_emb, selected)
    life.score_s += time.perf_counter() - t0
    return float(values.sum())


def _progress_decision(life, symbols, observed, dest_name):
    """Bits saved on dest_name's exam glances by this decision. The new glance is not in the sum.

    Learning, the coin and the reservoir offer are Life.room_decision / corridor_decision.
    """
    glances = list(life.exam.slots[dest_name])
    index = life.index[dest_name]
    before = _exam_bits(life, glances, index)
    if observed == "corridor":
        life.corridor_decision(symbols)
    else:
        life.room_decision(symbols, observed)
    after = _exam_bits(life, glances, index)
    return float(before - after)


def _room_rows(rooms, values, counts, last_end, v_max, t):
    rows = []
    for i, name in enumerate(rooms):
        age = None if last_end[i] is None else int(t - last_end[i])
        aged = None
        if age is not None and v_max is not None:
            aged = aged_value(values[i], age, v_max)
            if age == 0 and aged != float(values[i]):
                raise RuntimeError(
                    f"age 0 changed room {name}: v {values[i]} aged {aged}"
                )
        rows.append({
            "index": int(i),
            "room": name,
            "v": float(values[i]),
            "age": age,
            "aged": None if aged is None else float(aged),
            "count": int(counts[i]),
        })
    return rows


def choose_room(room, unvisited, values, counts, last_end, v_max, t, rooms, rng, use_age):
    """M7b's arm choice over 18 rooms. Aging ranks only the greedy step.

    The stored values are not written. Returns (arm, mode, rows).
    """
    n_arms = len(values)
    if int(room) < 0 or int(room) >= n_arms:
        raise RuntimeError("choice room is off the ring")
    stored = [float(v) for v in values]
    rows = _room_rows(rooms, values, counts, last_end, v_max, t)
    if unvisited:
        target = m7.nearest_unvisited(int(room), unvisited, n=n_arms)
        unvisited.remove(target)
        mode = "first"
    elif rng.random() < curio.EPS:
        target = int(rng.integers(n_arms))
        mode = "eps"
    elif use_age:
        if v_max is None:
            raise RuntimeError("greedy age with no visit reward")
        if any(last_end[i] is None for i in range(n_arms)):
            raise RuntimeError("greedy age before the first pass finished")
        target = max(range(n_arms), key=lambda k: (rows[k]["aged"], rng.random()))
        mode = "greedy"
    else:
        target = max(range(n_arms), key=lambda k: (values[k], rng.random()))
        mode = "greedy"
    if [float(v) for v in values] != stored or list(counts) != [row["count"] for row in rows]:
        raise RuntimeError("the choice changed the stored EMA")
    return int(target), mode, rows


def latencies_of(channels, swap_log, L):
    """Decisions from each swap until the next stay in that room. Censored at L."""
    stays = [
        (i + 1, name)
        for i, name in enumerate(channels)
        if name not in ("corridor", "sleep")
    ]
    out = []
    for event in swap_log:
        decision = int(event["decision"])
        room = event["room"]
        nxt = None
        for dec, name in stays:
            if dec > decision and name == room:
                nxt = dec
                break
        if nxt is None:
            latency = int(L - decision)
            censored = True
        else:
            latency = int(nxt - decision)
            censored = False
        out.append({
            "decision": decision,
            "room": room,
            "k": int(event["k"]),
            "next_dwell": None if nxt is None else int(nxt),
            "latency": int(latency),
            "censored": bool(censored),
            "within_128": bool((not censored) and latency <= 128),
        })
    return out


def _fmt_age(value):
    if value is None:
        return "none"
    return str(int(value))


def _fmt_num(value):
    if value is None:
        return "none"
    return f"{float(value):.6f}"


def format_choice(record):
    lines = [
        f"choice {record['choice']} decision {record['decision']} t {record['t']} "
        f"mode {record['mode']} chosen {record['chosen']} index {record['chosen_index']} "
        f"v_max {_fmt_num(record['v_max'])}"
    ]
    for row in record["rooms"]:
        lines.append(
            f"  {row['room']} index {row['index']} v {_fmt_num(row['v'])} "
            f"age {_fmt_age(row['age'])} aged {_fmt_num(row['aged'])} v_max {_fmt_num(record['v_max'])}"
        )
    return lines


def run_life(seed, controller, L, score=True, check_sleep=False, trace_visits=False, trace_choices=0):
    """One controller on World5(seed, period=128). Fixed policies take W5's actions."""
    if controller not in CONTROLLERS:
        raise ValueError(controller)
    L = int(L)
    seed = int(seed)
    n_sleep = w4.sleep_decisions(L)
    plan = m7.sleep_plan(L, n_sleep)
    if any(length != m7.DWELL for _, length in plan):
        raise RuntimeError("a fixed sleep visit is not 8 decisions")
    points = w5.checkpoints_of(L)
    point_set = set(points)
    world = World5(seed, PERIOD)
    names = list(IDENTITIES)
    index = {name: i for i, name in enumerate(names)}
    t0 = time.perf_counter()
    life = w4.Life(seed, world, names, index, N_RING, w4.N_CH, list(LEARN14))
    rng = np.random.default_rng(1_000 + seed)
    state = {"kind": controller, "room": 0, "plan": [], "target": None, "arrived": True, "visit": None}
    pending_rooms = set(LEARN14) if controller == "oracle_fresh_s" else None
    band = None
    choice_rng = None
    if controller in BANDITS:
        choice_rng = np.random.default_rng(10_000 + seed)
        band = {
            "values": [0.0] * N_RING,
            "counts": [0] * N_RING,
            "last_end": [None] * N_RING,
            "v_max": None,
            "unvisited": set(range(N_RING)),
            "n_choices": 0,
            "n_first": 0,
            "n_eps": 0,
            "n_greedy": 0,
            "ema_updates": 0,
        }
    channels = []
    records = []
    visits = []
    choices = []
    clock = {"eval": 0.0}
    prev_cp = {"t": 0}
    visit_pending = None
    si = 0
    t = 0
    checked = False
    use_age = controller == "prog_age_s"

    def on_decision(done):
        event = world.maybe_swap(done)
        if event is not None and pending_rooms is not None:
            pending_rooms.add(event["room"])
        if score and done in point_set:
            rec = w5.take_checkpoint(life, done, prev_cp["t"], clock)
            prev_cp["t"] = done
            records.append(rec)
            print(
                f"s{seed} {controller} p={PERIOD} checkpoint t={done}/{L} W14 {rec['W14']:.4f}",
                flush=True,
            )

    def start_bandit_visit():
        if state["room"] != world.room:
            raise RuntimeError("state room and world room diverged")
        stored_counts = list(band["counts"])
        target, mode, rows = choose_room(
            world.room, band["unvisited"], band["values"], band["counts"],
            band["last_end"], band["v_max"], t, world.rooms, choice_rng, use_age,
        )
        if band["counts"] != stored_counts:
            raise RuntimeError("the choice changed visit counts")
        band["n_choices"] += 1
        band[{"first": "n_first", "eps": "n_eps", "greedy": "n_greedy"}[mode]] += 1
        state["plan"] = w4_world.shortest(state["room"], target, N_RING) + ["stay"] * m7.DWELL
        state["target"] = target
        state["arrived"] = False
        state["visit"] = {
            "open": True,
            "start_t": int(t),
            "total": 0.0,
            "travel": 0,
            "dwell": 0,
            "dwell_name": None,
            "target": int(target),
            "mode": mode,
        }
        record = {
            "choice": int(band["n_choices"]),
            "decision": int(t + 1),
            "t": int(t),
            "mode": mode,
            "chosen": world.rooms[target],
            "chosen_index": int(target),
            "v_max": None if band["v_max"] is None else float(band["v_max"]),
            "rooms": rows,
        }
        if trace_choices and band["n_choices"] <= int(trace_choices):
            choices.append(record)
            for line in format_choice(record):
                print(line, flush=True)

    def finish_bandit_visit():
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
        if trace_visits:
            visits.append({
                "kind": "room",
                "room": world.rooms[target],
                "start": int(info["start_t"] + 1),
                "end": int(t),
                "decisions": n_dec,
                "reward": float(reward),
                "travel": int(info["travel"]),
                "dwell": int(info["dwell"]),
                "mode": info["mode"],
            })

    while t < L:
        if not state["plan"] and w4.sleep_due(t, L, plan, si, N_RING):
            if state["visit"] is not None and state["visit"]["open"]:
                raise RuntimeError("sleep started inside a room visit")
            visit_pending = w4._close_visit(life, visit_pending)
            values_before = None if band is None else list(band["values"])
            _n, t, checked = w5.sleep_visit(
                life, seed, controller, PERIOD, channels, plan[si][1], t, L,
                check_sleep, checked, on_decision,
            )
            if values_before is not None and band["values"] != values_before:
                raise RuntimeError("a sleep visit changed the stored EMA")
            if trace_visits:
                visits.append({
                    "kind": "sleep",
                    "room": "sleep",
                    "start": int(t - plan[si][1] + 1),
                    "end": int(t),
                    "decisions": int(plan[si][1]),
                    "reward": None,
                    "travel": 0,
                    "dwell": 0,
                    "mode": "sleep",
                })
            si += 1
            continue
        if controller in FIXED:
            action = w5.take_action(state, world, controller, rng, pending_rooms, None, t)
        else:
            if not state["plan"]:
                start_bandit_visit()
            action = state["plan"].pop(0)
        if action == "stay" and not state["arrived"]:
            name = world.rooms[state["room"]]
            if name != world.rooms[state["target"]]:
                raise RuntimeError(
                    f"arrival room {name} is not target {world.rooms[state['target']]}"
                )
            if pending_rooms is not None:
                pending_rooms.discard(name)
            state["arrived"] = True
        room, channel, symbols = world.step(action, rng)
        state["room"] = room
        if controller in BANDITS:
            dest_name = world.rooms[state["target"]]
            progress = _progress_decision(life, symbols, channel, dest_name)
            state["visit"]["total"] += progress
            if channel == "corridor":
                state["visit"]["travel"] += 1
            else:
                state["visit"]["dwell"] += 1
                state["visit"]["dwell_name"] = channel
                visit_pending = channel
        elif channel == "corridor":
            life.corridor_decision(symbols)
        else:
            life.room_decision(symbols, channel)
            visit_pending = channel
        channels.append(channel)
        t += 1
        on_decision(t)
        if t % 256 == 0:
            print(f"s{seed} {controller} p={PERIOD} t={t}/{L}", flush=True)
        if not state["plan"]:
            visit_pending = w4._close_visit(life, visit_pending)
            if controller in BANDITS:
                finish_bandit_visit()
    if controller in BANDITS:
        finish_bandit_visit()
    if visit_pending is not None:
        life.snapshot(visit_pending)
    if len(channels) != L:
        raise RuntimeError(f"{controller} ran {len(channels)} decisions")
    if si != len(plan):
        raise RuntimeError(f"{controller} took {si} of {len(plan)} planned sleep visits")
    if life.exam_drawn != 0:
        raise RuntimeError(f"sleep drew {life.exam_drawn} exam glances")
    if PERIOD is None:
        raise RuntimeError("M8 period is 128")
    if [event["decision"] for event in world.swap_log] != list(range(PERIOD, L + 1, PERIOD)):
        raise RuntimeError("swap decisions are not period, 2·period, …")
    if world.swap_log != w5.expected_swap_log(seed, PERIOD, L):
        raise RuntimeError("swap log does not match the independent draw")
    share, corridor_share, sleep_share = w4.shares_of(channels, life.names)
    if abs((sum(share.values()) + corridor_share + sleep_share) - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    if sum(name == "sleep" for name in channels) != n_sleep:
        raise RuntimeError("sleep decisions do not match the dose")
    if band is not None and band["ema_updates"] != band["n_choices"]:
        raise RuntimeError("EMA updates do not match room choices")
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
    wall = time.perf_counter() - t0
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
        "latency": latencies_of(channels, world.swap_log, L),
        "share": {name: float(share[name]) for name in life.names},
        "corridor_share": float(corridor_share),
        "sleep_share": float(sleep_share),
        "sleep_decisions": int(n_sleep),
        "room_order": list(world.rooms),
        "overlap": int(len(life.exam.ids & life.replay.ids)),
        "exam_glances_drawn": int(life.exam_drawn),
        "wall_s": float(wall),
        "eval_s": float(clock["eval"]),
        "learn_s": float(life.learn_s),
        "score_s": float(life.score_s),
        "finite": bool(finite),
        "n_ch": int(w4.N_CH),
        "tau": None if controller != "prog_age_s" else int(TAU),
        "n_choices": None if band is None else int(band["n_choices"]),
        "n_first": None if band is None else int(band["n_first"]),
        "n_eps": None if band is None else int(band["n_eps"]),
        "n_greedy": None if band is None else int(band["n_greedy"]),
        "ema_updates": None if band is None else int(band["ema_updates"]),
    }
    if trace_visits:
        row["visits"] = visits
    if trace_choices:
        row["choices"] = choices
    return row


def _job(args):
    seed, controller, L, score, trace_visits, trace_choices = args
    _threads()
    print(
        f"s{seed} {controller} p={PERIOD} L={L} start dose {w4.sleep_decisions(L)}",
        flush=True,
    )
    row = run_life(
        seed, controller, L, score=score, check_sleep=True,
        trace_visits=trace_visits, trace_choices=trace_choices,
    )
    shown = "none" if row["T14"] is None else f"{row['T14']:.4f}"
    print(
        f"s{seed} {controller} p={PERIOD} L={L} T14 {shown} "
        f"sleep {row['sleep_share']:.4f} corridor {row['corridor_share']:.3f} "
        f"wall {row['wall_s']:.1f}s learn {row['learn_s']:.1f}s "
        f"score {row['score_s']:.1f}s eval {row['eval_s']:.1f}s",
        flush=True,
    )
    return row


def _w5_job(args):
    seed, policy, L = args
    _threads()
    return w5.run_life(seed, policy, L, PERIOD, score=True, check_sleep=True)


def _within_bits(a, b, limit=BITS_LIMIT):
    return abs(float(a) - float(b)) <= limit


def _same_score(m8_row, w5_row, limit=BITS_LIMIT):
    if not _within_bits(m8_row["T14"], w5_row["T14"], limit):
        return f"T14 {m8_row['T14']} vs {w5_row['T14']}"
    if not _within_bits(m8_row["W14"], w5_row["W14"], limit):
        return f"W14 {m8_row['W14']} vs {w5_row['W14']}"
    if [c["decision"] for c in m8_row["checkpoints"]] != [c["decision"] for c in w5_row["checkpoints"]]:
        return "checkpoint decisions differ"
    for left, right in zip(m8_row["checkpoints"], w5_row["checkpoints"]):
        if not _within_bits(left["W14"], right["W14"], limit):
            return f"checkpoint {left['decision']} W14 {left['W14']} vs {right['W14']}"
        for name in LEARN14:
            if not _within_bits(left["held"][name], right["held"][name], limit):
                return f"checkpoint {left['decision']} {name} {left['held'][name]} vs {right['held'][name]}"
    if m8_row["swap_log"] != w5_row["swap_log"]:
        return "swap log differs"
    return None


def _check_age(choices):
    if len(choices) < 5:
        raise RuntimeError(f"expected 5 choices, got {len(choices)}")
    saw_age0 = False
    for record in choices[:5]:
        for row in record["rooms"]:
            if row["age"] is None:
                if row["aged"] is not None:
                    raise RuntimeError("an unvisited room has an aged value")
                continue
            if record["v_max"] is None:
                raise RuntimeError("a visited room has no v_max")
            expect = aged_value(row["v"], row["age"], record["v_max"])
            if row["aged"] != expect and abs(row["aged"] - expect) > 0.0:
                raise RuntimeError(
                    f"aged value {row['aged']} != formula {expect} for {row['room']}"
                )
            if row["age"] == 0:
                saw_age0 = True
                if row["aged"] != row["v"]:
                    raise RuntimeError(
                        f"age 0 room {row['room']} aged {row['aged']} != v {row['v']}"
                    )
    if not saw_age0:
        raise RuntimeError("no room with age 0 in the first 5 choices")
    first = choices[0]
    second = choices[1]
    if first["mode"] != "first" or first["chosen_index"] != 0:
        raise RuntimeError(
            f"first choice mode {first['mode']} index {first['chosen_index']}, expected first pass at room 0"
        )
    followed = second["rooms"][first["chosen_index"]]
    if followed["age"] != 0 or followed["aged"] != followed["v"]:
        raise RuntimeError(
            f"room chosen first has age {followed['age']} aged {followed['aged']} v {followed['v']} at the next choice"
        )
    if second["v_max"] is None or followed["v"] != second["v_max"]:
        raise RuntimeError("the only visit reward is not v_max at choice 2")


def _check_formula():
    if aged_value(1.25, 0, 9.0) != 1.25:
        raise RuntimeError("age 0 did not return v")
    got = aged_value(0.0, TAU, 1.0)
    expect = 1.0 - math.exp(-1.0)
    if abs(got - expect) > 1e-15:
        raise RuntimeError(f"tau step {got} != {expect}")
    raw = [0.0, 0.5]
    ages = [TAU, 0]
    v_max = 1.0
    aged = [aged_value(raw[0], ages[0], v_max), aged_value(raw[1], ages[1], v_max)]
    if max(range(2), key=lambda k: aged[k]) != 0:
        raise RuntimeError("aged rank did not prefer the stale room")
    if max(range(2), key=lambda k: raw[k]) != 1:
        raise RuntimeError("raw rank did not prefer the higher EMA")
    if aged[1] != 0.5:
        raise RuntimeError("age 0 on the second room changed v")


def _pool_map(jobs, fn, workers):
    ctx = get_context("spawn")
    with ctx.Pool(int(workers)) as pool:
        return [item.get() for item in [pool.apply_async(fn, (job,)) for job in jobs]]


def smoke(out_dir=None, workers=2):
    """VPS checks. A failed check does not stage kernels."""
    _threads()
    workers = int(workers)
    if workers < 1 or workers > 2:
        raise RuntimeError("this VPS pool is at most 2")
    out_dir = Path(out_dir or (ROOT / "results" / "m8w5"))
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
        for L in (64, 128, 256, 512, L_SCORE):
            dose = w4.sleep_decisions(L)
            log(f"dose L{L} {dose}/{L} share {dose / L:.6f} checkpoints {w5.checkpoints_of(L)}")
        _check_formula()
        log("formula: age 0 returns v; tau=256 moves 0 toward 1 by 1-exp(-1); aged rank differs from raw rank")

        log("smoke check 1: rand_dwell_s and oracle_fresh_s seed 0 period 128 L=256 vs W5")
        m8_jobs = [(0, policy, 256, True, False, 0) for policy in FIXED]
        w5_jobs = [(0, policy, 256) for policy in FIXED]
        m8_rows = _pool_map(m8_jobs, _job, workers)
        w5_rows = _pool_map(w5_jobs, _w5_job, workers)
        check1 = {}
        for policy, m8_row, w5_row in zip(FIXED, m8_rows, w5_rows):
            if m8_row["controller"] != policy or w5_row["policy"] != policy:
                raise RuntimeError("check 1 row is not the requested policy")
            reason = _same_score(m8_row, w5_row)
            published = PUBLISHED_L256[policy]
            pub_t = abs(m8_row["T14"] - published["T14"])
            pub_w = abs(m8_row["W14"] - published["W14"])
            log(
                f"  {policy} m8 T14 {m8_row['T14']:.6f} W14 {m8_row['W14']:.6f} "
                f"w5 T14 {w5_row['T14']:.6f} W14 {w5_row['W14']:.6f} "
                f"abs_dT14 {abs(m8_row['T14'] - w5_row['T14']):.3e} "
                f"published_dT14 {pub_t:.3e} published_dW14 {pub_w:.3e} "
                f"limit {BITS_LIMIT:.1e}"
            )
            if reason is not None:
                raise RuntimeError(f"{policy} does not reproduce W5: {reason}")
            if pub_t > BITS_LIMIT or pub_w > BITS_LIMIT:
                raise RuntimeError(
                    f"{policy} differs from W5's published L=256 line by T14 {pub_t:.3e} W14 {pub_w:.3e}"
                )
            check1[policy] = {
                "T14": m8_row["T14"],
                "W14": m8_row["W14"],
                "w5_T14": w5_row["T14"],
                "w5_W14": w5_row["W14"],
                "abs_dT14": abs(m8_row["T14"] - w5_row["T14"]),
                "published_dT14": pub_t,
            }
        payload["check1"] = check1

        log("smoke check 2: prog_age_s seed 12 L=512, first 5 choices, with checkpoints")
        aged = _job((12, "prog_age_s", 512, True, False, 5))
        _check_age(aged["choices"])
        for record in aged["choices"]:
            for line in format_choice(record):
                log(line)
        age0 = [
            (record["choice"], row["room"], row["v"], row["aged"])
            for record in aged["choices"]
            for row in record["rooms"]
            if row["age"] == 0
        ]
        log(
            "  age 0 rooms: "
            + ", ".join(f"choice {c} {name} v {v:.6f} aged {a:.6f}" for c, name, v, a in age0)
        )
        if not aged["finite"]:
            raise RuntimeError("timed prog_age_s T14 is not finite")
        log(
            f"  prog_age_s seed 12 L=512 T14 {aged['T14']:.6f} W14 {aged['W14']:.6f} "
            f"wall {aged['wall_s']:.2f}s learn {aged['learn_s']:.2f}s "
            f"score {aged['score_s']:.2f}s eval {aged['eval_s']:.2f}s "
            f"sleep {aged['sleep_decisions']} swaps {len(aged['swap_log'])} "
            f"choices {aged['n_choices']} first {aged['n_first']} eps {aged['n_eps']} "
            f"greedy {aged['n_greedy']}"
        )
        scale_l = L_SCORE / 512.0
        one = aged["learn_s"] * scale_l + aged["score_s"] * scale_l + aged["eval_s"]
        waves = math.ceil(len(CONTROLLERS) / KERNEL_WORKERS)
        kernel_s = one * waves
        guarded = kernel_s * 2.0
        log(
            f"  one L2048 prog_age_s {one:.1f}s waves {waves} jobs {len(CONTROLLERS)} "
            f"workers {KERNEL_WORKERS} kernel {kernel_s:.1f}s guarded {guarded:.1f}s "
            f"limit {KERNEL_WALL_LIMIT_S:.0f}s"
        )
        if kernel_s > KERNEL_WALL_LIMIT_S:
            raise RuntimeError(
                f"kernel estimate {kernel_s:.0f}s exceeds {KERNEL_WALL_LIMIT_S:.0f}s"
            )
        payload["timed_512"] = {
            "seed": 12,
            "controller": "prog_age_s",
            "L": 512,
            "T14": aged["T14"],
            "W14": aged["W14"],
            "wall_s": aged["wall_s"],
            "learn_s": aged["learn_s"],
            "score_s": aged["score_s"],
            "eval_s": aged["eval_s"],
            "sleep_decisions": aged["sleep_decisions"],
            "n_choices": aged["n_choices"],
            "n_first": aged["n_first"],
            "n_eps": aged["n_eps"],
            "n_greedy": aged["n_greedy"],
        }
        payload["estimate"] = {
            "one_s": float(one),
            "waves": int(waves),
            "jobs": len(CONTROLLERS),
            "workers": KERNEL_WORKERS,
            "kernel_s": float(kernel_s),
            "guarded_s": float(guarded),
            "limit_s": KERNEL_WALL_LIMIT_S,
        }
        payload["age_choices"] = aged["choices"]

        log("smoke check 3: prog_chan_s seed 12 L=512 visit trace and swap log")
        traced = _job((12, "prog_chan_s", 512, False, True, 0))
        if traced["swap_log"] != w5.expected_swap_log(12, PERIOD, 512):
            raise RuntimeError("prog_chan_s swap log does not match the independent draw")
        room_visits = [v for v in traced["visits"] if v["kind"] == "room"]
        if not room_visits:
            raise RuntimeError("prog_chan_s trace has no room visit")
        swaps = list(traced["swap_log"])
        cursor = 0
        for visit in room_visits:
            while cursor < len(swaps) and swaps[cursor]["decision"] < visit["start"]:
                event = swaps[cursor]
                log(
                    f"  swap decision {event['decision']} room {event['room']} k {event['k']}"
                )
                cursor += 1
            log(
                f"  visit {visit['start']}-{visit['end']} room {visit['room']} "
                f"decisions {visit['decisions']} reward {visit['reward']:.6f}"
            )
            while cursor < len(swaps) and swaps[cursor]["decision"] <= visit["end"]:
                event = swaps[cursor]
                log(
                    f"  swap decision {event['decision']} room {event['room']} k {event['k']} "
                    f"inside visit {visit['start']}-{visit['end']}"
                )
                cursor += 1
        while cursor < len(swaps):
            event = swaps[cursor]
            log(f"  swap decision {event['decision']} room {event['room']} k {event['k']}")
            cursor += 1
        if any(v["reward"] is None or not math.isfinite(v["reward"]) for v in room_visits):
            raise RuntimeError("room visit reward is missing or nonfinite")
        if traced["ema_updates"] != len(room_visits):
            raise RuntimeError("sleep or a missing visit updated the bandit")
        log(
            f"  visits {len(room_visits)} sleep {traced['sleep_decisions']} "
            f"swaps {len(traced['swap_log'])} wall {traced['wall_s']:.2f}s "
            f"choices {traced['n_choices']} first {traced['n_first']} "
            f"eps {traced['n_eps']} greedy {traced['n_greedy']}"
        )
        payload["chan_trace"] = {
            "n_visits": len(room_visits),
            "sleep_decisions": traced["sleep_decisions"],
            "swap_log": traced["swap_log"],
            "wall_s": traced["wall_s"],
        }

        log("smoke check 4: four controllers, seed 12, L=128, finite T14")
        small_jobs = [(12, controller, 128, True, False, 0) for controller in CONTROLLERS]
        small = _pool_map(small_jobs, _job, workers)
        check4 = {}
        for controller, row in zip(CONTROLLERS, small):
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
            if row["swap_log"] != small[0]["swap_log"]:
                raise RuntimeError("controllers disagree on the L=128 swap log")
            check4[controller] = {"T14": row["T14"], "W14": row["W14"], "finite": True}
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


def _write_kernel(directory, seed):
    directory.mkdir(parents=True, exist_ok=True)
    slug = f"m8-w5-s{seed}"
    kernel_id = f"ezzzzzx/{slug}"
    meta = {
        "id": kernel_id,
        "title": slug,
        "code_file": "m8_kernel.py",
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
    text = f'''"""M8-W5 seed {seed}. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_M8_KERNEL") != "1":
    os.environ["_M8_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("m8_w5.py"))
    if not hits:
        raise SystemExit("m8_w5.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import m8_w5
    m8_w5.kernel_main(
        seed={seed},
        out="/kaggle/working",
        workers=4,
        kernel_id="{kernel_id}",
    )


if __name__ == "__main__":
    main()
'''
    (directory / "m8_kernel.py").write_text(text)
    return kernel_id


def stage(smoke_payload, kaggle_dir=None):
    """Write the private dataset folder and one kernel folder per seed. Does not push."""
    if not smoke_payload.get("ok"):
        raise SystemExit("smoke did not pass; kernels not staged")
    kaggle_dir = Path(kaggle_dir or (ROOT / "kaggle" / "m8w5"))
    if kaggle_dir.exists():
        shutil.rmtree(kaggle_dir)
    code = kaggle_dir / "code"
    code.mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "pdp code m8w5",
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


def kernel_main(seed, out, workers=4, kernel_id=None):
    if __name__ != "m8_w5":
        raise SystemExit("import m8_w5 and call kernel_main so the pool can spawn workers")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 12-23")
    if int(workers) < 1 or int(workers) > KERNEL_WORKERS:
        raise SystemExit("kernel pool is at most 4")
    _threads()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"kernel {kernel_id} seed {seed} workers {workers} controllers {len(CONTROLLERS)}",
        flush=True,
    )
    t0 = time.perf_counter()
    jobs = [(seed, controller, L_SCORE, True, False, 0) for controller in CONTROLLERS]
    ctx = get_context("spawn")
    with ctx.Pool(int(workers)) as pool:
        asyncs = [pool.apply_async(_job, (job,)) for job in jobs]
        rows = [item.get() for item in asyncs]
    wall = time.perf_counter() - t0
    if len(rows) != len(CONTROLLERS):
        raise SystemExit(f"expected {len(CONTROLLERS)} rows, got {len(rows)}")
    for row, controller in zip(rows, CONTROLLERS):
        _check_row(row, seed, controller, L_SCORE)
    by_name = {row["controller"]: row for row in rows}
    if len(by_name) != len(CONTROLLERS):
        raise SystemExit("duplicate controller rows")
    ordered = [by_name[name] for name in CONTROLLERS]
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
        "workers": int(workers),
        "wall_s": wall,
        "torch": torch.__version__,
        "model_seed": m7.MODEL_SEED,
        "n_ch": w4.N_CH,
        "L": L_SCORE,
        "period": PERIOD,
        "tau": TAU,
        "sleep_share_target": SLEEP_SHARE,
        "room_order": ordered[0]["room_order"],
        "runs": ordered,
    }
    path = out / f"m8_s{seed}.json"
    m7.dump(path, payload)
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    return payload


def _check_row(row, seed, controller, L):
    if row["seed"] != int(seed) or row["controller"] != controller or row["L"] != int(L):
        raise RuntimeError("row does not match its job")
    if row["period"] != PERIOD:
        raise RuntimeError("row period is not 128")
    if not row["finite"]:
        raise RuntimeError(f"{controller} seed {seed} is not finite")
    if row["sleep_decisions"] != w4.sleep_decisions(L):
        raise RuntimeError("sleep dose drifted")
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
    if controller == "prog_age_s" and row["tau"] != TAU:
        raise RuntimeError("tau drifted")
    if controller in FIXED and row["tau"] is not None:
        raise RuntimeError("a fixed controller stored tau")


def mean_std(values):
    return m7.mean_std(values)


def _fmt(value, digits=6):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _pm(mu, sd, digits=6):
    return f"{_fmt(mu, digits)} ± {_fmt(sd, digits)}"


def _pick(runs, controller, seed):
    hit = [row for row in runs if row["controller"] == controller and row["seed"] == seed]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {controller} seed {seed}, got {len(hit)}")
    return hit[0]


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


def w5_reference():
    path = ROOT / "results" / "w5" / "results.json"
    data = json.loads(path.read_text())
    summary = data["summary"]
    by = {}
    for policy in ("rand_dwell_s", "oracle14_s", "oracle_fresh_s"):
        block = summary["by_policy"][f"{policy}/p128"]
        by[policy] = {
            "T14_mean": float(block["T14_mean"]),
            "T14_std_ddof0": float(block["T14_std_ddof0"]),
            "W14_mean": float(block["W14_mean"]),
            "W14_std_ddof0": float(block["W14_std_ddof0"]),
        }
    per = []
    for row in data["runs"]:
        if row.get("period") == 128 and row["policy"] in by:
            per.append({
                "policy": row["policy"],
                "seed": int(row["seed"]),
                "T14": float(row["T14"]),
                "W14": float(row["W14"]),
            })
    gap = summary["gap_summary"]["128"]["oracle_fresh_s_minus_rand_dwell_s"]
    return {
        "source": "results/w5/results.json",
        "period": 128,
        "seeds": [0, 1, 2, 3, 4, 5],
        "L": int(data["L"]),
        "by_policy": by,
        "per_seed": per,
        "oracle_fresh_s_minus_rand_dwell_s": {
            "mean": float(gap["mean"]),
            "std_ddof0": float(gap["std_ddof0"]),
            "left_lower": int(gap["left_lower"]),
            "n": int(gap["n"]),
        },
    }


def summarize(runs):
    by = {}
    for controller in CONTROLLERS:
        rows = [_pick(runs, controller, seed) for seed in SEEDS]
        t14_mu, t14_sd = mean_std([row["T14"] for row in rows])
        w14_mu, w14_sd = mean_std([row["W14"] for row in rows])
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
            checkpoints.append({"decision": int(decision), "T14_mean": mu, "T14_std_ddof0": sd})
        lat_rows = [item for row in rows for item in row["latency"]]
        by[controller] = {
            "controller": controller,
            "n": len(rows),
            "T14_mean": t14_mu,
            "T14_std_ddof0": t14_sd,
            "W14_mean": w14_mu,
            "W14_std_ddof0": w14_sd,
            "share_mean": share_mean,
            "share_std_ddof0": share_std,
            "corridor_mean": corridor_mu,
            "corridor_std_ddof0": corridor_sd,
            "sleep_mean": sleep_mu,
            "sleep_std_ddof0": sleep_sd,
            "checkpoints": checkpoints,
            "latency": _latency_summary(lat_rows),
        }
    gaps = []
    for seed in SEEDS:
        rand = _pick(runs, "rand_dwell_s", seed)
        chan = _pick(runs, "prog_chan_s", seed)
        age = _pick(runs, "prog_age_s", seed)
        fresh = _pick(runs, "oracle_fresh_s", seed)
        gaps.append({
            "seed": int(seed),
            "prog_chan_s_minus_rand_dwell_s": float(chan["T14"] - rand["T14"]),
            "prog_age_s_minus_rand_dwell_s": float(age["T14"] - rand["T14"]),
            "prog_age_s_minus_prog_chan_s": float(age["T14"] - chan["T14"]),
            "oracle_fresh_s_minus_rand_dwell_s": float(fresh["T14"] - rand["T14"]),
        })
    gap_summary = {}
    for key in (
        "prog_chan_s_minus_rand_dwell_s",
        "prog_age_s_minus_rand_dwell_s",
        "prog_age_s_minus_prog_chan_s",
        "oracle_fresh_s_minus_rand_dwell_s",
    ):
        values = [row[key] for row in gaps]
        mu, sd = mean_std(values)
        gap_summary[key] = {
            "mean": mu,
            "std_ddof0": sd,
            "left_lower": int(sum(value < 0.0 for value in values)),
            "n": len(values),
        }
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
        "latency_by_seed": latency_by_seed,
    }


def _header(labels):
    return "| " + " | ".join(labels) + " |"


def _sep(n):
    return "|" + "|".join(["---:" if i else "---" for i in range(n)]) + "|"


def render_markdown(result):
    summary = result["summary"]
    seeds = result["seeds"]
    runs = result["runs"]
    points = [item["decision"] for item in summary["by_controller"]["rand_dwell_s"]["checkpoints"]]
    lines = [
        "# M8-W5 curiosity on the drifting ring",
        "",
        "W5's 18-room ring with period 128. One learnable room's rule is redrawn every 128 decisions. "
        "T14 is the mean of W14 at decisions L/8, 2L/8, …, L. W14 is bits/symbol of the frozen bank "
        "on a fresh EVAL=2048 continuation of the 14 learnable rooms. The final W14 is the checkpoint at L. "
        "Lower is better. Every controller sleeps for the fixed share 0.20, in visits of 8, placed as in W5. "
        f"At L = {L_SCORE} that is {w4.sleep_decisions(L_SCORE)} decisions, "
        f"share {w4.sleep_decisions(L_SCORE) / L_SCORE:.6f}.",
        "",
        "`prog_chan_s` ranks 18 rooms by the stored EMA. `prog_age_s` uses the same EMA and, on the greedy "
        f"step only, ranks by v + (v_max − v) · (1 − exp(−age / {TAU})). Age is the decisions since that "
        "room's last visit ended. v_max is the largest single-visit reward so far. The stored EMA is not aged. "
        "The room reward is the sum, over the visited room's exam glances, of bits before minus bits after, "
        "summed over the visit's decisions and divided by those decisions. Travel counts. Sleep visits are not "
        "bandit updates. A swap's latency is the decisions until the next stay in that room. If the life ends "
        "first, the latency is L minus the swap decision and the swap is censored. within 128 counts a stay "
        "whose latency is at most 128. The median includes censored latencies.",
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
    lines.extend(["", "## Final W14", "", _header(["controller", *[f"seed {seed}" for seed in seeds], "mean ± std ddof=0"]), _sep(len(seeds) + 2)])
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
        cells = [_fmt(item["T14_mean"]) for item in block["checkpoints"]]
        lines.append(f"| {controller} | " + " | ".join(cells) + f" | {_fmt(block['T14_mean'])} |")
    lines.extend([
        "",
        "## Gaps",
        "",
        "On T14. A negative gap means the first controller is lower. left lower counts seeds where the gap is negative.",
        "",
        _header([
            "seed",
            "prog_chan_s − rand_dwell_s",
            "prog_age_s − rand_dwell_s",
            "prog_age_s − prog_chan_s",
            "oracle_fresh_s − rand_dwell_s",
        ]),
        _sep(5),
    ])
    for gap in summary["gaps"]:
        lines.append(
            f"| {gap['seed']} | {_fmt(gap['prog_chan_s_minus_rand_dwell_s'])} | "
            f"{_fmt(gap['prog_age_s_minus_rand_dwell_s'])} | "
            f"{_fmt(gap['prog_age_s_minus_prog_chan_s'])} | "
            f"{_fmt(gap['oracle_fresh_s_minus_rand_dwell_s'])} |"
        )
    for label, key in (
        ("mean", "mean"),
        ("std ddof=0", "std_ddof0"),
    ):
        block = summary["gap_summary"]
        lines.append(
            f"| {label} | {_fmt(block['prog_chan_s_minus_rand_dwell_s'][key])} | "
            f"{_fmt(block['prog_age_s_minus_rand_dwell_s'][key])} | "
            f"{_fmt(block['prog_age_s_minus_prog_chan_s'][key])} | "
            f"{_fmt(block['oracle_fresh_s_minus_rand_dwell_s'][key])} |"
        )
    block = summary["gap_summary"]
    lines.append(
        "| left lower (count) | "
        f"{block['prog_chan_s_minus_rand_dwell_s']['left_lower']} of {block['prog_chan_s_minus_rand_dwell_s']['n']} | "
        f"{block['prog_age_s_minus_rand_dwell_s']['left_lower']} of {block['prog_age_s_minus_rand_dwell_s']['n']} | "
        f"{block['prog_age_s_minus_prog_chan_s']['left_lower']} of {block['prog_age_s_minus_prog_chan_s']['n']} | "
        f"{block['oracle_fresh_s_minus_rand_dwell_s']['left_lower']} of {block['oracle_fresh_s_minus_rand_dwell_s']['n']} |"
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
    lines.extend([
        "",
        "## Bandit choices",
        "",
        _header(["controller", "seed", "choices", "first", "eps", "greedy", "ema updates"]),
        _sep(7),
    ])
    for controller in BANDITS:
        for seed in seeds:
            row = _pick(runs, controller, seed)
            lines.append(
                f"| {controller} | {seed} | {row['n_choices']} | {row['n_first']} | "
                f"{row['n_eps']} | {row['n_greedy']} | {row['ema_updates']} |"
            )
    lines.extend([
        "",
        "## Swap log",
        "",
        "One log per seed. The same log was checked against every controller. k counts swaps from 0.",
        "",
        _header(["seed", "decision", "room", "k"]),
        _sep(4),
    ])
    for seed in seeds:
        for event in _pick(runs, "rand_dwell_s", seed)["swap_log"]:
            lines.append(f"| {seed} | {event['decision']} | {event['room']} | {event['k']} |")
    lines.extend([
        "",
        "## Ring order",
        "",
        "Room name in ring positions 0..17. One order per seed. Controllers share it.",
        "",
        _header(["seed", *[f"r{i}" for i in range(N_RING)]]),
        _sep(N_RING + 1),
    ])
    for seed in seeds:
        order = _pick(runs, "rand_dwell_s", seed)["room_order"]
        lines.append(f"| {seed} | " + " | ".join(order) + " |")
    ref = result["w5_reference"]
    lines.extend([
        "",
        "## W5 reference",
        "",
        f"Period 128, seeds 0–5, L = {ref['L']}. Source: {ref['source']}.",
        "",
        _header(["policy", "T14 mean", "T14 std ddof=0", "W14 mean", "W14 std ddof=0"]),
        _sep(5),
    ])
    for policy in ("rand_dwell_s", "oracle14_s", "oracle_fresh_s"):
        block = ref["by_policy"][policy]
        lines.append(
            f"| {policy} | {_fmt(block['T14_mean'])} | {_fmt(block['T14_std_ddof0'])} | "
            f"{_fmt(block['W14_mean'])} | {_fmt(block['W14_std_ddof0'])} |"
        )
    gap = ref["oracle_fresh_s_minus_rand_dwell_s"]
    lines.append("")
    lines.append(
        f"oracle_fresh_s − rand_dwell_s on T14: mean {_fmt(gap['mean'])}, "
        f"std ddof=0 {_fmt(gap['std_ddof0'])}, left lower {gap['left_lower']} of {gap['n']}."
    )
    lines.extend([
        "",
        _header(["policy", "seed", "T14", "W14"]),
        _sep(4),
    ])
    for row in ref["per_seed"]:
        lines.append(f"| {row['policy']} | {row['seed']} | {_fmt(row['T14'])} | {_fmt(row['W14'])} |")
    lines.extend([
        "",
        "## Walls and kernels",
        "",
        _header(["kernel", "seed", "wall seconds", "workers"]),
        _sep(4),
    ])
    for kernel in result["kernels"]:
        lines.append(
            f"| {kernel['kernel_id']} | {kernel['seed']} | {_fmt(kernel['wall_s'], 3)} | {kernel['workers']} |"
        )
    lines.extend([
        "",
        _header(["controller", "seed", "wall seconds", "learn seconds", "score seconds", "eval seconds"]),
        _sep(6),
    ])
    for controller in CONTROLLERS:
        for seed in seeds:
            row = _pick(runs, controller, seed)
            lines.append(
                f"| {controller} | {seed} | {_fmt(row['wall_s'], 3)} | {_fmt(row['learn_s'], 3)} | "
                f"{_fmt(row['score_s'], 3)} | {_fmt(row['eval_s'], 3)} |"
            )
    lines.extend(["", "## Smoke test", "", "```"])
    lines.extend(result["smoke_log"])
    lines.extend(["```", ""])
    if result.get("kernel_status"):
        lines.extend(["```", result["kernel_status"].rstrip(), "```", ""])
    lines.extend([
        "## Protocol",
        "",
        "- World5(world_seed, period=128). Room names, ring order, noise rooms and Z are W4's. "
        "After every positive multiple of 128 decisions, one of the 14 learnable rooms is redrawn "
        "with default_rng(world_seed + 80000), as in W5.",
        "- The cortex, stores, coin, sleep dose and sleep placement are W5's, which are W4's. "
        "Policy movement for rand_dwell_s and oracle_fresh_s is w5.take_action with "
        "default_rng(1000 + world_seed). A tie on the ring goes left. Dwell is 8 stays.",
        "- prog_chan_s and prog_age_s choose with default_rng(10000 + world_seed). The optimistic "
        "first pass is m7.nearest_unvisited over the 18 ring positions, with n=18. Then eps-greedy "
        "with EPS 0.1. The EMA uses BETA 0.3 and m7.ema. Sleep is not an arm.",
        f"- prog_age_s ranks the greedy step by v_r + (v_max − v_r) · (1 − exp(−age_r / {TAU})). "
        "age_r is the decisions since that room's last visit ended, measured at the choice. "
        "v_max is the largest room-visit reward seen so far. Tau is 256. The stored EMA is not changed by aging.",
        "- The room reward sums bits saved on the destination room's exam glances over the visit, "
        "including corridor steps, and divides by the visit's decisions. The glance inserted by the "
        "decision is not part of that decision's sum. Sleep does not call the EMA.",
        "- A checkpoint scores w4.evaluate_w14 after that decision's swap. T14 is the mean of the eight checkpoints.",
        "- Swap latency uses the next stay in the swapped room after the swap decision. A missing stay is censored at L − decision.",
        f"- World seeds {', '.join(str(seed) for seed in seeds)}. L = {L_SCORE}. Period 128. CPU. "
        "Private Kaggle kernels, one seed each, pool of 4, no GPU, no internet. Nothing retuned.",
        "",
    ])
    return "\n".join(lines)


def _load_kernel_payloads(out_dir):
    out_dir = Path(out_dir)
    found = {}
    for seed in SEEDS:
        hits = sorted(out_dir.glob(f"s{seed}/m8_s{seed}.json"))
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
    if len(runs) != len(SEEDS) * len(CONTROLLERS):
        raise RuntimeError(f"expected {len(SEEDS) * len(CONTROLLERS)} runs, got {len(runs)}")
    summary = summarize(runs)
    kernels.sort(key=lambda item: item["seed"])
    status_path = out_dir / "kernel_status.txt"
    result = {
        "seeds": list(SEEDS),
        "controllers": list(CONTROLLERS),
        "period": PERIOD,
        "L": L_SCORE,
        "tau": TAU,
        "model_seed": m7.MODEL_SEED,
        "n_ch": w4.N_CH,
        "sleep_share_target": SLEEP_SHARE,
        "sleep_decisions": w4.sleep_decisions(L_SCORE),
        "dataset_id": DATASET_ID,
        "kernels": kernels,
        "summary": summary,
        "w5_reference": w5_reference(),
        "smoke": {key: smoke_payload[key] for key in smoke_payload if key not in ("log", "age_choices")},
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
    parser.add_argument("--out", default=str(ROOT / "results" / "m8w5"))
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
