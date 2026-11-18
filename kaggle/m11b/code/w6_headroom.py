"""W6 headroom: W5's bank and fixed sleep, on a ring where rooms are born and die.

Numbers only. No verdict. The cortex, sleep, stores and movement are w5_headroom's.
T_open is the mean of eight checkpoint scores. Each score is the mean bits/symbol
of the slots that hold a learnable room then.
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
    if os.environ.get("_W6_MAIN") != "1":
        os.environ["_W6_MAIN"] = "1"
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
import w4_headroom as w4
import w4_world
import w5_headroom as w5
import w6_world
from w4_world import DARK, IDENTITIES, N_RING
from w5_world import finite_period
from w6_world import KINDS, World6, expected_birth_log

SEEDS = (0, 1, 2, 3, 4, 5)
POLICIES = ("rand_dwell_s", "oracle_learn_s", "oracle_new_s", "prog_age_s")
FIXED = ("rand_dwell_s", "oracle_learn_s", "oracle_new_s")
PERIODS = (128, 512)
L_SCORE = 4096
N_CHECKPOINTS = 8
SLEEP_SHARE = 0.20
TAU = 256
KERNEL_WALL_LIMIT_S = 90 * 60
BITS_LIMIT = 1e-6
KERNEL_WORKERS = 4
ROOT = Path(__file__).resolve().parent
CODE_FILES = (
    "w6_world.py",
    "w6_headroom.py",
    "w5_world.py",
    "w5_headroom.py",
    "w4_world.py",
    "w4_headroom.py",
    "m8_w5.py",
    "m7_w3.py",
    "w3_world.py",
    "w3_headroom.py",
    "curio.py",
    "h20_world.py",
    "h30w_ring.py",
    "h30x_trace.py",
    "h30y_roots.py",
)
DATASET_ID = "ezzzzzx/pdp-code-w6"
GAPS = (
    "oracle_learn_s_minus_rand_dwell_s",
    "oracle_new_s_minus_rand_dwell_s",
    "prog_age_s_minus_rand_dwell_s",
)

if curio.EPS != 0.1 or curio.BETA != 0.3 or m8.TAU != TAU:
    raise RuntimeError(f"bandit constants drifted: EPS {curio.EPS} BETA {curio.BETA} TAU {m8.TAU}")
if m7.DWELL != 8 or curio.B != 64 or curio.EVAL != 2048:
    raise RuntimeError("W6 inherited constants drifted")
if w4.N_CH != N_RING + 1 or N_RING != 18:
    raise RuntimeError("W4 ring width drifted")
if list(KINDS) != list(h20_world.PROCS) + ["N"]:
    raise RuntimeError("newborn kinds drifted")


def period_name(period):
    return w5.period_name(period)


def checkpoints_of(L):
    return w5.checkpoints_of(L)


def job_list(split):
    """Lives in one kernel, or split into a period-128 kernel and the rest."""
    if not split:
        return [("all", expected_jobs())]
    return [
        ("p128", [(policy, 128) for policy in POLICIES]),
        ("p512", [(policy, 512) for policy in POLICIES] + [("rand_dwell_s", None)]),
    ]


def expected_jobs():
    jobs = [(policy, period) for period in PERIODS for policy in POLICIES]
    jobs.append(("rand_dwell_s", None))
    return jobs


def _threads():
    w4._threads()


def as_text(symbols):
    return "".join(h20_world.ALPHA[int(s)] for s in symbols)


class OpenLife(w4.Life):
    """W4's life, plus the prequential bits of stay-glances the caller marks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preq_bits = 0.0
        self.preq_symbols = 0
        self.count_preq = False

    def _learn(self, symbols, channel, prev, state=None, commit=True):
        bits = super()._learn(symbols, channel, prev, state=state, commit=commit)
        if self.count_preq:
            arr = np.asarray(bits, dtype=np.float64)
            self.preq_bits += float(arr.sum())
            self.preq_symbols += int(arr.shape[0])
        return bits


def stream_order(world, slots):
    """Native rooms in LEARN14 order, then newborns by slot.

    Processes on one side share one h20 generator, so the order of take() is
    part of the stream. LEARN14 is the order evaluate_w14 uses. A closed world
    still has every native room, and this order yields W4's symbols. A newborn
    has its own world, so reading it does not move the native generator.
    """
    native = []
    born = []
    for slot in slots:
        slot = int(slot)
        if world.source[slot] is None:
            native.append(slot)
        else:
            born.append(slot)
    native.sort(key=lambda slot: w4_world.LEARN14.index(world.rooms[slot]))
    born.sort()
    return native + born


def evaluate_open(cortex, world, index, prev, decision):
    """Frozen forward pass. A copy of the world supplies EVAL symbols per learnable slot.

    The channel row is the slot's initial room, which a birth does not change.
    """
    wc = world.copy()
    slots = stream_order(wc, wc.learnable_slots())
    if not slots:
        raise RuntimeError(f"no learnable slot at decision {decision}")
    sequences = [wc.continuation_slot(slot, curio.EVAL) for slot in slots]
    symbols = torch.from_numpy(np.stack(sequences))
    prev_sym = torch.full((len(slots),), -1, dtype=torch.long)
    channels = torch.tensor([index[wc.rooms[slot]] for slot in slots], dtype=torch.long)
    totals = m7.score_sequences(cortex.model, cortex.channel_emb, symbols, prev_sym, channels)
    rooms = []
    bits = []
    for j, slot in enumerate(slots):
        value = float(totals[j]) / curio.EVAL
        born = int(wc.born_at[slot])
        rooms.append({
            "slot": int(slot),
            "channel": wc.rooms[slot],
            "kind": wc.kind[slot],
            "born": born,
            "bits": value,
            "newborn": int(int(prev) < born <= int(decision)),
        })
        bits.append(value)
    return float(np.mean(bits)), rooms


def take_checkpoint(life, decision, prev, clock):
    params = life.cortex.clone_params()
    live = life.cortex.clone_live()
    t0 = time.perf_counter()
    score, rooms = evaluate_open(life.cortex, life.world, life.index, prev, decision)
    clock["eval"] += time.perf_counter() - t0
    if life.cortex.param_delta(params) > 0.0:
        raise RuntimeError("checkpoint scoring changed parameters")
    if not life.cortex.live_matches(live):
        raise RuntimeError("checkpoint scoring changed the live state")
    return {
        "decision": int(decision),
        "open": float(score),
        "n": len(rooms),
        "rooms": rooms,
    }


def draw_plan(state, world, policy, rng, pending, trace, t):
    """One new visit for a fixed policy. The caller has an empty plan."""
    if state["plan"]:
        raise RuntimeError("a plan was drawn while one was still running")
    if policy == "rand_dwell_s":
        targets = list(range(N_RING))
        shown = None
    elif policy == "oracle_learn_s":
        targets = world.learnable_slots()
        shown = None
    elif policy == "oracle_new_s":
        if pending:
            targets = sorted(pending)
            for slot in targets:
                if not world.is_learnable(slot):
                    raise RuntimeError("S holds a slot that is not learnable")
        else:
            targets = world.learnable_slots()
        shown = sorted(pending)
    else:
        raise ValueError(policy)
    if not targets:
        raise RuntimeError(f"{policy} has no target slots")
    room_before = state["room"]
    action = w4.next_w4(state, N_RING, targets, rng)
    target = w5.target_after(room_before, [action] + state["plan"], N_RING)
    if target not in targets:
        raise RuntimeError("target was drawn outside the candidate slots")
    state["target"] = int(target)
    state["arrived"] = False
    if trace is not None:
        trace.append({
            "decision": int(t + 1),
            "target": int(target),
            "kind": world.kind[target],
            "targets": [int(i) for i in targets],
            "S": [] if shown is None else [int(i) for i in shown],
        })
    return action


def latencies_of(stays, birth_log, L):
    """Decisions from each birth until the next stay in that slot. Censored at L.

    The stay on the birth decision observed the dead room. The next stay is the
    first one on the newborn.
    """
    out = []
    for event in birth_log:
        decision = int(event["decision"])
        slot = int(event["slot"])
        nxt = None
        for dec, stayed in stays:
            if dec > decision and stayed == slot:
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
            "slot": slot,
            "k": int(event["k"]),
            "dead_kind": event["dead_kind"],
            "newborn_kind": event["newborn_kind"],
            "next_stay": None if nxt is None else int(nxt),
            "latency": int(latency),
            "censored": bool(censored),
        })
    return out


def run_life(seed, policy, L, period, score=True, check_sleep=False, trace=None):
    """One policy. Births and checkpoints sit on the decision clock, including sleep."""
    L = int(L)
    period = finite_period(period)
    seed = int(seed)
    if policy not in POLICIES:
        raise ValueError(policy)
    if period is None and policy != "rand_dwell_s":
        raise ValueError("the closed-world anchor is rand_dwell_s")
    n_sleep = w4.sleep_decisions(L)
    plan = m7.sleep_plan(L, n_sleep)
    if any(length != m7.DWELL for _, length in plan):
        raise RuntimeError("a fixed sleep visit is not 8 decisions")
    if score:
        points = checkpoints_of(L)
    else:
        points = []
    point_set = set(points)
    world = World6(seed, period)
    initial_rooms = list(world.rooms)
    names = list(IDENTITIES)
    index = {name: i for i, name in enumerate(names)}
    t0 = time.perf_counter()
    life = OpenLife(seed, world, names, index, N_RING, w4.N_CH, list(w4_world.LEARN14))
    rng = np.random.default_rng(1_000 + seed)
    state = {"room": 0, "plan": [], "target": None, "arrived": True, "visit": None}
    # S is the learnable slots whose current room was born since the last arrival.
    # It starts as every learnable slot. Arrival removes the slot. A learnable
    # birth puts the slot back. A noise birth removes it.
    pending = set(world.learnable_slots()) if policy == "oracle_new_s" else None
    band = None
    choice_rng = None
    if policy == "prog_age_s":
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
        if event is not None and pending is not None:
            if world.is_learnable(event["slot"]):
                pending.add(event["slot"])
            else:
                pending.discard(event["slot"])
        if score and done in point_set:
            rec = take_checkpoint(life, done, prev_cp["t"], clock)
            prev_cp["t"] = done
            records.append(rec)
            print(
                f"s{seed} {policy} p={period_name(period)} checkpoint t={done}/{L} "
                f"open {rec['open']:.4f} n {rec['n']}",
                flush=True,
            )

    def start_bandit_visit():
        if state["room"] != world.room:
            raise RuntimeError("state room and world room diverged")
        stored_counts = list(band["counts"])
        target, mode, _rows = m8.choose_room(
            world.room, band["unvisited"], band["values"], band["counts"],
            band["last_end"], band["v_max"], t, world.rooms, choice_rng, True,
        )
        if band["counts"] != stored_counts:
            raise RuntimeError("the choice changed visit counts")
        band["n_choices"] += 1
        band[{"first": "n_first", "eps": "n_eps", "greedy": "n_greedy"}[mode]] += 1
        state["plan"] = w4_world.shortest(state["room"], target, N_RING) + ["stay"] * m7.DWELL
        state["target"] = int(target)
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

    def finish_bandit_visit():
        info = state["visit"]
        if info is None or not info["open"]:
            return
        n_dec = int(info["travel"] + info["dwell"])
        if n_dec <= 0:
            raise RuntimeError("empty room visit")
        if info["dwell"] and info["dwell_name"] != world.rooms[info["target"]]:
            raise RuntimeError("dwell channel is not the chosen slot")
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

    while t < L:
        if not state["plan"] and w4.sleep_due(t, L, plan, si, N_RING):
            if band is not None and state["visit"] is not None and state["visit"]["open"]:
                raise RuntimeError("sleep started inside a room visit")
            visit_pending = w4._close_visit(life, visit_pending)
            values_before = None if band is None else list(band["values"])
            _n, t, checked = w5.sleep_visit(
                life, seed, policy, period, sleep_marks, plan[si][1], t, L,
                check_sleep, checked, on_decision,
            )
            if values_before is not None and band["values"] != values_before:
                raise RuntimeError("a sleep visit changed the stored EMA")
            si += 1
            continue
        if policy == "prog_age_s":
            if not state["plan"]:
                start_bandit_visit()
            action = state["plan"].pop(0)
        elif not state["plan"]:
            action = draw_plan(state, world, policy, rng, pending, trace, t)
        else:
            action = state["plan"].pop(0)
        if action == "stay" and not state["arrived"]:
            if state["room"] != state["target"]:
                raise RuntimeError(
                    f"arrival slot {state['room']} is not target {state['target']}"
                )
            if pending is not None:
                pending.discard(state["room"])
            state["arrived"] = True
        learnable_stay = action == "stay" and world.is_learnable(state["room"])
        room, channel, symbols = world.step(action, rng)
        state["room"] = room
        life.count_preq = learnable_stay
        try:
            if policy == "prog_age_s":
                dest = world.rooms[state["target"]]
                progress = m8._progress_decision(life, symbols, channel, dest)
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
            print(f"s{seed} {policy} p={period_name(period)} t={t}/{L}", flush=True)
        if not state["plan"]:
            visit_pending = w4._close_visit(life, visit_pending)
            if policy == "prog_age_s":
                finish_bandit_visit()
    if policy == "prog_age_s":
        finish_bandit_visit()
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
    kind_sum = noise_count + dark_count + learnable_stays
    if kind_sum != sum(slot_counts):
        raise RuntimeError("slot stays were not partitioned into noise, dark and learnable")
    if life.preq_symbols != learnable_stays * curio.B:
        raise RuntimeError(
            f"prequential symbols {life.preq_symbols} != {learnable_stays} glances"
        )
    if band is not None and band["ema_updates"] != band["n_choices"]:
        raise RuntimeError("EMA updates do not match room choices")
    slot_share = [count / L for count in slot_counts]
    noise_share = noise_count / L
    corridor_share = corridor_count / L
    sleep_share = n_sleep / L
    if abs((sum(slot_share) + corridor_share + sleep_share) - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    if score:
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
        )
    else:
        t_open = None
        final = None
        preq_bps = None
        if life.preq_symbols:
            preq_bps = float(life.preq_bits / life.preq_symbols)
        finite = False
    wall = time.perf_counter() - t0
    return {
        "seed": seed,
        "policy": policy,
        "period": None if period is None else int(period),
        "L": int(L),
        "T_open": None if t_open is None else float(t_open),
        "final": None if final is None else float(final),
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
        "latency": latencies_of(stays, world.birth_log, L),
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
        "tau": None if policy != "prog_age_s" else int(TAU),
        "n_choices": None if band is None else int(band["n_choices"]),
        "n_first": None if band is None else int(band["n_first"]),
        "n_eps": None if band is None else int(band["n_eps"]),
        "n_greedy": None if band is None else int(band["n_greedy"]),
        "ema_updates": None if band is None else int(band["ema_updates"]),
    }


def _job(args):
    seed, policy, period, L = args
    _threads()
    print(
        f"s{seed} {policy} p={period_name(period)} L={L} start dose {w4.sleep_decisions(L)}",
        flush=True,
    )
    row = run_life(seed, policy, L, period, score=True, check_sleep=True)
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


def mean_std(values):
    return w4.mean_std(values)


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


def smoke(out_dir=None):
    """VPS checks, one process. A failed check does not stage kernels."""
    _threads()
    out_dir = Path(out_dir or (ROOT / "results" / "w6"))
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
        log(f"torch={torch.__version__} threads=1 model_seed={m7.MODEL_SEED} n_ch={w4.N_CH}")
        for life_L in (64, 160, 512, 1024, L_SCORE):
            dose = w4.sleep_decisions(life_L)
            points = checkpoints_of(life_L) if life_L % N_CHECKPOINTS == 0 else None
            log(f"dose L{life_L} {dose}/{life_L} share {dose / life_L:.6f} checkpoints {points}")
        if World6(0, None).birth_period is not None:
            raise RuntimeError("birth_period None was stored as a finite period")

        log("smoke check 1: birth_period None matches World4 glances, seed 0, 200 decisions")
        base = w4_world.World4(0)
        live = World6(0, None)
        if base.rooms != live.rooms or base.perm != live.perm:
            raise RuntimeError("birth_period None ring does not match World4")
        fresh_a = w4_world.World4(0)
        fresh_b = World6(0, None)
        for name in list(w4_world.LEARN14) + list(w4_world.NOISE) + [DARK]:
            if list(fresh_a.glance_room(name)) != list(fresh_b.glance_room(name)):
                raise RuntimeError(f"birth_period None first glance differs for {name}")
        rng = np.random.default_rng(1_000)
        state = {"room": 0, "plan": []}
        targets = w4.targets_for(base.rooms, "rand_dwell_s")
        for decision in range(1, 201):
            action = w4.next_w4(state, N_RING, targets, rng)
            r1, c1, s1 = base.step(action, None)
            r2, c2, s2 = live.step(action, None)
            state["room"] = r1
            if live.maybe_birth(decision) is not None:
                raise RuntimeError("birth_period None birthed")
            if r1 != r2 or c1 != c2 or list(s1) != list(s2):
                raise RuntimeError(f"glance mismatch at decision {decision}")
        log("  200 decisions match World4, including stays, moves and the corridor")

        log("smoke check 1b: rand_dwell_s birth_period None L=64 final score equals W4 W14")
        ref = w4.run_life(0, "rand_dwell_s", 64, score=True, check_sleep=True)
        matched = run_life(0, "rand_dwell_s", 64, None, score=True, check_sleep=True)
        gap = abs(matched["final"] - ref["W14"])
        log(
            f"  W4 W14 {ref['W14']:.8f} W6 final {matched['final']:.8f} "
            f"abs {gap:.3e} limit {BITS_LIMIT:.1e} T_open {matched['T_open']:.6f}"
        )
        if gap > BITS_LIMIT:
            raise RuntimeError(f"birth_period None final score differs by {gap:.3e}")
        if not matched["finite"] or matched["birth_log"]:
            raise RuntimeError("birth_period None L=64 life is not a finite anchor")
        payload["anchor_L64"] = {
            "W4_W14": float(ref["W14"]),
            "W6_final": float(matched["final"]),
            "abs": gap,
            "T_open": float(matched["T_open"]),
        }

        log("smoke check 2: birth_period 16, seed 0, L=160 birth log, new stream, copy")
        world = World6(0, 16)
        saved = None
        for decision in range(1, 161):
            world.step("left", None)
            pre = world.copy() if saved is None and decision % 16 == 0 else None
            event = world.maybe_birth(decision)
            if event is not None and saved is None:
                saved = (pre, world.copy(), dict(event))
        expect = expected_birth_log(0, 16, 160)
        if world.birth_log != expect:
            raise RuntimeError("birth log does not match an independent draw")
        pre, post, event = saved
        if world.rooms != pre.rooms or world.rooms != post.rooms:
            raise RuntimeError("a birth changed the ring")
        for item in world.birth_log:
            log(
                f"  birth decision {item['decision']} slot {item['slot']} "
                f"dead {item['dead_kind']} newborn {item['newborn_kind']} k {item['k']}"
            )
        slot = int(event["slot"])
        kind = event["newborn_kind"]
        k = int(event["k"])
        other = next(i for i in range(N_RING) if i != slot)
        other_same = list(pre.copy().glance_slot(other)) == list(post.copy().glance_slot(other))
        old = list(pre.copy().glance_slot(slot))
        if kind == "N":
            fresh_rng = np.random.default_rng(0 + 95_000 + k)
            expect1 = list(curio.glance(None, "N", fresh_rng))
            expect2 = list(curio.glance(None, "N", fresh_rng))
        else:
            fresh = h20_world.World(0 + 3_000 + k)
            expect1 = list(curio.glance(fresh, kind, None))
            expect2 = list(curio.glance(fresh, kind, None))
        copy_at_start = post.copy()
        first_live = list(post.glance_slot(slot))
        first_copy = list(copy_at_start.glance_slot(slot))
        copy_after = post.copy()
        second_live = list(post.glance_slot(slot))
        second_later = list(copy_after.glance_slot(slot))
        second_start = list(copy_at_start.glance_slot(slot))
        channel_same = post.rooms[slot] == pre.rooms[slot]
        log(
            f"  slot {slot} channel {post.rooms[slot]} kind {kind} "
            f"dead {event['dead_kind']} k {k}"
        )
        log(f"  before {as_text(old)!r}")
        log(f"  after  {as_text(first_live)!r}")
        log(
            f"  new-instance match {first_live == expect1} "
            f"differs-from-old {old != first_live} other-slot-unchanged {other_same} "
            f"channel-unchanged {channel_same}"
        )
        log(
            f"  copy at birth matches {first_copy == first_live == expect1} "
            f"copy after one glance matches {second_later == second_live == second_start == expect2}"
        )
        if (
            first_live != expect1 or first_copy != expect1 or old == first_live
            or not other_same or not channel_same
        ):
            raise RuntimeError("newborn glance is not the new instance")
        if second_live != expect2 or second_later != expect2 or second_start != expect2:
            raise RuntimeError("copy() did not continue the newborn")
        noise_event = next((item for item in world.birth_log if item["newborn_kind"] == "N"), None)
        if noise_event is None:
            longer = World6(0, 16)
            limit = 16 * 200
            for decision in range(1, limit + 1):
                longer.step("stay", None)
                item = longer.maybe_birth(decision)
                if item is not None and item["newborn_kind"] == "N":
                    noise_event = item
                    break
            if noise_event is None:
                raise RuntimeError(f"no newborn of kind N in {limit} decisions")
            log(
                f"  no kind N in 160 decisions; longer log birth decision {noise_event['decision']} "
                f"slot {noise_event['slot']} dead {noise_event['dead_kind']} "
                f"newborn {noise_event['newborn_kind']} k {noise_event['k']}"
            )
        else:
            log(
                f"  kind N inside 160: decision {noise_event['decision']} "
                f"slot {noise_event['slot']} dead {noise_event['dead_kind']} "
                f"k {noise_event['k']}"
            )
        payload["birth_L160"] = {
            "n": len(world.birth_log),
            "birth_log": world.birth_log,
            "shown_slot": slot,
            "shown_kind": kind,
            "noise": {
                "decision": int(noise_event["decision"]),
                "slot": int(noise_event["slot"]),
                "k": int(noise_event["k"]),
            },
        }

        log("smoke check 3: oracle_new_s birth_period 16 seed 0 L=160 targets and S")
        trace = []
        traced = run_life(0, "oracle_new_s", 160, 16, score=False, check_sleep=True, trace=trace)
        if traced["birth_log"] != expected_birth_log(0, 16, 160):
            raise RuntimeError("oracle_new_s birth log does not match the independent draw")
        initial_s = World6(0, 16).learnable_slots()
        if not trace or trace[0]["S"] != initial_s:
            raise RuntimeError("S did not start as every learnable slot")
        for choice in trace:
            log(
                f"  choice decision {choice['decision']} target {choice['target']} "
                f"kind {choice['kind']} targets={choice['targets']} S={choice['S']}"
            )
        log(
            f"  choices {len(trace)} sleep {traced['sleep_decisions']} "
            f"births {len(traced['birth_log'])} wall {traced['wall_s']:.2f}s"
        )
        payload["oracle_new_trace"] = {
            "n_choices": len(trace),
            "sleep_decisions": traced["sleep_decisions"],
            "birth_log": traced["birth_log"],
        }

        log("smoke check 4: four policies, seed 0, birth_period 128, L=512, finite T_open")
        walls_512 = {}
        for policy in POLICIES:
            row = run_life(policy=policy, seed=0, L=512, period=128, score=True, check_sleep=True)
            walls_512[policy] = float(row["wall_s"])
            log(
                f"  {policy} T_open {row['T_open']:.6f} final {row['final']:.6f} "
                f"finite {row['finite']} noise {row['noise_share']:.4f} "
                f"preq_bps {row['preq_bps']:.6f} sleep {row['sleep_share']:.4f} "
                f"corridor {row['corridor_share']:.4f} births {len(row['birth_log'])} "
                f"wall {row['wall_s']:.2f}s learn {row['learn_s']:.2f}s "
                f"score {row['score_s']:.2f}s eval {row['eval_s']:.2f}s"
            )
            if not row["finite"]:
                raise RuntimeError(f"{policy} T_open is not finite")
            if len(row["checkpoints"]) != N_CHECKPOINTS:
                raise RuntimeError(f"{policy} did not score 8 checkpoints")
            if row["birth_log"] != expected_birth_log(0, 128, 512):
                raise RuntimeError(f"{policy} birth log drifted")

        log("smoke check 5: time rand_dwell_s seed 0 birth_period 128 L=1024, with checkpoints")
        timed = run_life(0, "rand_dwell_s", 1024, 128, score=True, check_sleep=True)
        if not timed["finite"]:
            raise RuntimeError("timed life T_open is not finite")
        log(
            f"  rand_dwell_s p=128 L=1024 T_open {timed['T_open']:.6f} final {timed['final']:.6f} "
            f"wall {timed['wall_s']:.2f}s learn {timed['learn_s']:.2f}s "
            f"score {timed['score_s']:.2f}s eval {timed['eval_s']:.2f}s "
            f"sleep {timed['sleep_decisions']} births {len(timed['birth_log'])}"
        )
        durations = []
        per_policy = {}
        for policy in POLICIES:
            from_512 = walls_512[policy] * (L_SCORE / 512.0)
            if policy == "rand_dwell_s":
                from_1024 = timed["wall_s"] * (L_SCORE / 1024.0)
                est = max(from_512, from_1024)
            else:
                from_1024 = None
                est = from_512
            per_policy[policy] = {
                "L512_wall_s": walls_512[policy],
                "from_512_s": from_512,
                "from_1024_s": from_1024,
                "L4096_s": est,
            }
            durations.append(est)
            durations.append(est)
        anchor_est = per_policy["rand_dwell_s"]["L4096_s"]
        durations.append(anchor_est)
        kernel_s = makespan(durations, KERNEL_WORKERS)
        guarded = kernel_s * 2.0
        slow = max(item["L4096_s"] for item in per_policy.values())
        one_guarded = slow * 2.0
        if one_guarded > KERNEL_WALL_LIMIT_S:
            raise RuntimeError(
                f"one L=4096 life estimate {one_guarded:.0f}s exceeds {KERNEL_WALL_LIMIT_S:.0f}s"
            )
        split = guarded > KERNEL_WALL_LIMIT_S
        if split:
            log("  guarded kernel wall exceeds 90 min; splitting by period")
        else:
            log("  guarded kernel wall fits in 90 min; one kernel per seed runs every life")
        p128_s = makespan([per_policy[policy]["L4096_s"] for policy in POLICIES], KERNEL_WORKERS)
        p512_s = makespan(
            [per_policy[policy]["L4096_s"] for policy in POLICIES] + [anchor_est],
            KERNEL_WORKERS,
        )
        log(
            f"  one L4096 slow {slow:.1f}s kernel {kernel_s:.1f}s guarded {guarded:.1f}s "
            f"p128 {p128_s:.1f}s p512 {p512_s:.1f}s waves jobs {len(durations)} "
            f"workers {KERNEL_WORKERS} limit {KERNEL_WALL_LIMIT_S:.0f}s split {split}"
        )
        for policy in POLICIES:
            item = per_policy[policy]
            log(
                f"  estimate {policy} L512 {item['L512_wall_s']:.1f}s "
                f"L4096 {item['L4096_s']:.1f}s"
            )
        payload["timed_1024"] = {
            "seed": 0,
            "policy": "rand_dwell_s",
            "period": 128,
            "L": 1024,
            "T_open": timed["T_open"],
            "final": timed["final"],
            "wall_s": timed["wall_s"],
            "learn_s": timed["learn_s"],
            "score_s": timed["score_s"],
            "eval_s": timed["eval_s"],
            "sleep_decisions": timed["sleep_decisions"],
        }
        payload["walls_512"] = walls_512
        payload["estimate"] = {
            "per_policy": per_policy,
            "kernel_s": kernel_s,
            "guarded_s": guarded,
            "p128_s": p128_s,
            "p512_s": p512_s,
            "slow_s": slow,
            "jobs": len(durations),
            "workers": KERNEL_WORKERS,
            "split": split,
        }
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


def _write_kernel(directory, seed, jobs, slug):
    directory.mkdir(parents=True, exist_ok=True)
    kernel_id = f"ezzzzzx/{slug}"
    meta = {
        "id": kernel_id,
        "title": slug,
        "code_file": "w6_kernel.py",
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
    parts = []
    for policy, period in jobs:
        period_src = "None" if period is None else str(int(period))
        parts.append(f'("{policy}", {period_src})')
    jobs_src = ", ".join(parts)
    text = f'''"""W6 seed {seed}. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_W6_KERNEL") != "1":
    os.environ["_W6_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("w6_headroom.py"))
    if not hits:
        raise SystemExit("w6_headroom.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import w6_headroom
    w6_headroom.kernel_main(
        seed={seed},
        out="/kaggle/working",
        workers=4,
        kernel_id="{kernel_id}",
        jobs=({jobs_src},),
    )


if __name__ == "__main__":
    main()
'''
    (directory / "w6_kernel.py").write_text(text)
    return kernel_id


def stage(smoke_payload, kaggle_dir=None):
    """Write the private dataset folder and the kernel folders. Does not push."""
    if not smoke_payload.get("ok"):
        raise SystemExit("smoke did not pass; kernels not staged")
    split = bool(smoke_payload["estimate"]["split"])
    kaggle_dir = Path(kaggle_dir or (ROOT / "kaggle" / "w6"))
    if kaggle_dir.exists():
        shutil.rmtree(kaggle_dir)
    code = kaggle_dir / "code"
    code.mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "pdp code w6",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    ids = []
    groups = job_list(split)
    for seed in SEEDS:
        for label, jobs in groups:
            if not split:
                slug = f"w6-s{seed}"
                folder = f"s{seed}"
            elif label == "p128":
                slug = f"w6-s{seed}-p128"
                folder = f"s{seed}-p128"
            elif label == "p512":
                slug = f"w6-s{seed}-p512"
                folder = f"s{seed}-p512"
            else:
                raise RuntimeError(f"unexpected split label {label}")
            ids.append(_write_kernel(kaggle_dir / folder, seed, jobs, slug))
    print(f"staged {len(ids)} kernels split {split}", flush=True)
    for kernel_id in ids:
        print(kernel_id, flush=True)
    return ids


def kernel_main(seed, out, workers=4, kernel_id=None, jobs=None):
    if __name__ != "w6_headroom":
        raise SystemExit("import w6_headroom and call kernel_main so the pool can spawn workers")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 0-5")
    if int(workers) < 1 or int(workers) > KERNEL_WORKERS:
        raise SystemExit("kernel pool is at most 4")
    if jobs is None:
        jobs = expected_jobs()
    jobs = tuple((str(policy), finite_period(period)) for policy, period in jobs)
    allowed = set(expected_jobs())
    if any(job not in allowed for job in jobs) or len(jobs) != len(set(jobs)):
        raise SystemExit(f"jobs {jobs} are not a W6 kernel")
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
            {"policy": policy, "period": None if period is None else int(period)}
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
    path = out / f"w6_s{seed}.json"
    m7.dump(path, payload)
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    return payload


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
        if row["tau"] != TAU or row["n_choices"] != row["ema_updates"]:
            raise RuntimeError("prog_age_s bookkeeping drifted")
    else:
        if row["tau"] is not None:
            raise RuntimeError("a fixed policy stored a bandit tau")


def _fmt(value, digits=6):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _pick(runs, policy, period, seed):
    period = finite_period(period)
    hit = [
        row for row in runs
        if row["policy"] == policy and finite_period(row["period"]) == period and row["seed"] == seed
    ]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {policy} p={period_name(period)} seed {seed}, got {len(hit)}")
    return hit[0]


def _policy_keys():
    keys = [(policy, period) for period in PERIODS for policy in POLICIES]
    keys.append(("rand_dwell_s", None))
    return keys


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
    for policy, period in _policy_keys():
        rows = [_pick(runs, policy, period, seed) for seed in SEEDS]
        t_mu, t_sd = mean_std([row["T_open"] for row in rows])
        f_mu, f_sd = mean_std([row["final"] for row in rows])
        p_mu, p_sd = mean_std([row["preq_bps"] for row in rows])
        slot_mu = []
        for index in range(N_RING):
            mu, sd = mean_std([row["slot_share"][index] for row in rows])
            slot_mu.append({"mean": mu, "std_ddof0": sd})
        n_mu, n_sd = mean_std([row["noise_share"] for row in rows])
        c_mu, c_sd = mean_std([row["corridor_share"] for row in rows])
        s_mu, s_sd = mean_std([row["sleep_share"] for row in rows])
        cps = []
        for index, decision in enumerate(checkpoints_of(L_SCORE)):
            mu, sd = mean_std([row["checkpoints"][index]["open"] for row in rows])
            newborn_bits = []
            older_bits = []
            for row in rows:
                rec = row["checkpoints"][index]
                if rec["decision"] != decision:
                    raise RuntimeError("checkpoint order drifted")
                for room in rec["rooms"]:
                    if room["newborn"]:
                        newborn_bits.append(room["bits"])
                    else:
                        older_bits.append(room["bits"])
            nb_mu, nb_sd = mean_std(newborn_bits)
            old_mu, old_sd = mean_std(older_bits)
            cps.append({
                "decision": decision,
                "open_mean": mu,
                "open_std_ddof0": sd,
                "newborn_mean": nb_mu,
                "newborn_std_ddof0": nb_sd,
                "newborn_n": len(newborn_bits),
                "older_mean": old_mu,
                "older_std_ddof0": old_sd,
                "older_n": len(older_bits),
            })
        by[f"{policy}/p{period_name(period)}"] = {
            "policy": policy,
            "period": None if period is None else int(period),
            "n": len(rows),
            "T_open_mean": t_mu,
            "T_open_std_ddof0": t_sd,
            "final_mean": f_mu,
            "final_std_ddof0": f_sd,
            "preq_bps_mean": p_mu,
            "preq_bps_std_ddof0": p_sd,
            "slot_share_mean": slot_mu,
            "noise_mean": n_mu,
            "noise_std_ddof0": n_sd,
            "corridor_mean": c_mu,
            "corridor_std_ddof0": c_sd,
            "sleep_mean": s_mu,
            "sleep_std_ddof0": s_sd,
            "latency": _latency_summary(rows),
            "checkpoints": cps,
        }
    gaps = {128: [], 512: []}
    for period in PERIODS:
        for seed in SEEDS:
            rand = _pick(runs, "rand_dwell_s", period, seed)
            learn = _pick(runs, "oracle_learn_s", period, seed)
            new = _pick(runs, "oracle_new_s", period, seed)
            age = _pick(runs, "prog_age_s", period, seed)
            gaps[period].append({
                "seed": int(seed),
                "period": int(period),
                "oracle_learn_s_minus_rand_dwell_s": float(learn["T_open"] - rand["T_open"]),
                "oracle_new_s_minus_rand_dwell_s": float(new["T_open"] - rand["T_open"]),
                "prog_age_s_minus_rand_dwell_s": float(age["T_open"] - rand["T_open"]),
                "T_open": {
                    "rand_dwell_s": rand["T_open"],
                    "oracle_learn_s": learn["T_open"],
                    "oracle_new_s": new["T_open"],
                    "prog_age_s": age["T_open"],
                },
            })
    gap_summary = {}
    for period in PERIODS:
        block = {}
        for key in GAPS:
            mu, sd = mean_std([row[key] for row in gaps[period]])
            block[key] = {
                "mean": mu,
                "std_ddof0": sd,
                "left_lower": int(sum(row[key] < 0.0 for row in gaps[period])),
                "n": len(gaps[period]),
            }
        gap_summary[str(period)] = block
    births = {}
    anchor_rows = {seed: _pick(runs, "rand_dwell_s", None, seed) for seed in SEEDS}
    for period in PERIODS:
        rows = []
        for seed in SEEDS:
            rand = _pick(runs, "rand_dwell_s", period, seed)
            rows.append({
                "seed": int(seed),
                "period": int(period),
                "with_births": rand["T_open"],
                "without_births": anchor_rows[seed]["T_open"],
                "births_minus_anchor": float(rand["T_open"] - anchor_rows[seed]["T_open"]),
            })
        mu, sd = mean_std([row["births_minus_anchor"] for row in rows])
        births[str(period)] = {
            "mean": mu,
            "std_ddof0": sd,
            "n": len(rows),
            "rows": rows,
        }
    return {
        "by_policy": by,
        "gaps": gaps[128] + gaps[512],
        "gap_summary": gap_summary,
        "births_minus_anchor": births,
    }


def render_markdown(result):
    summary = result["summary"]
    seeds = list(result["seeds"])
    lines = []
    lines.append("# W6 open world")
    lines.append("")
    lines.append(
        "World6 starts as World4's 18-slot ring. Every `birth_period` decisions, one of the "
        "17 slots that are not Z dies and a new room is born there. The newborn's kind is "
        "uniform on the 7 H20 processes plus N. T_open is the mean of the checkpoint score "
        "at decisions L/8, 2L/8, …, L. The checkpoint score is bits/symbol of the frozen bank "
        "on a fresh EVAL=2048 continuation of each slot that holds a learnable room then, "
        "from a zero state, on that slot's channel row, from a copy of the world at that "
        "checkpoint. The final score is the checkpoint at L. Lower is better. The anchor is "
        "rand_dwell_s with birth_period None, the closed World4 life, scored with the eight checkpoints."
    )
    lines.append("")
    dose = w4.sleep_decisions(L_SCORE)
    lines.append(
        f"Every policy sleeps for the fixed share 0.20, in visits of 8, placed as in W5. "
        f"At L = {L_SCORE} that is {dose} decisions, share {dose / L_SCORE:.6f}. "
        f"prog_age_s is M8's ageing bandit with tau {TAU}, EPS {curio.EPS}, BETA {curio.BETA}, "
        "arms = slots. It is not told about births."
    )
    lines.append("")

    def block_table(title, field):
        lines.append(f"## {title}")
        lines.append("")
        header = (
            "| policy | period | "
            + " | ".join(f"seed {seed}" for seed in seeds)
            + " | mean | std ddof=0 |"
        )
        lines.append(header)
        lines.append("|---|---:|" + "---:|" * len(seeds) + "---:|---:|")
        for policy, period in _policy_keys():
            block = summary["by_policy"][f"{policy}/p{period_name(period)}"]
            cells = []
            for seed in seeds:
                row = _pick(result["runs"], policy, period, seed)
                cells.append(_fmt(row[field]))
            lines.append(
                f"| {policy} | {period_name(period)} | " + " | ".join(cells)
                + f" | {_fmt(block[field + '_mean'])} | {_fmt(block[field + '_std_ddof0'])} |"
            )
        lines.append("")

    block_table("T_open", "T_open")
    block_table("Final checkpoint", "final")

    lines.append("## Prequential bits/symbol on learnable-room glances")
    lines.append("")
    lines.append(
        "Sum of the bank's prequential glance bits on stays in a slot whose room was "
        "learnable at that decision, divided by the symbols in those glances."
    )
    lines.append("")
    header = (
        "| policy | period | "
        + " | ".join(f"seed {seed}" for seed in seeds)
        + " | mean | std ddof=0 |"
    )
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * len(seeds) + "---:|---:|")
    for policy, period in _policy_keys():
        block = summary["by_policy"][f"{policy}/p{period_name(period)}"]
        cells = []
        for seed in seeds:
            row = _pick(result["runs"], policy, period, seed)
            cells.append(_fmt(row["preq_bps"]))
        lines.append(
            f"| {policy} | {period_name(period)} | " + " | ".join(cells)
            + f" | {_fmt(block['preq_bps_mean'])} | {_fmt(block['preq_bps_std_ddof0'])} |"
        )
    lines.append("")
    lines.append("| policy | period | seed | preq bits | preq symbols | preq bits/symbol | learnable stays |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    runs = sorted(result["runs"], key=lambda row: (
        _policy_keys().index((row["policy"], finite_period(row["period"]))), row["seed"],
    ))
    for row in runs:
        lines.append(
            f"| {row['policy']} | {period_name(row['period'])} | {row['seed']} | "
            f"{_fmt(row['preq_bits'])} | {row['preq_symbols']} | {_fmt(row['preq_bps'])} | "
            f"{row['learnable_stays']} |"
        )
    lines.append("")

    lines.append("## T_open per checkpoint")
    lines.append("")
    decisions = checkpoints_of(L_SCORE)
    lines.append(
        "| policy | period | seed | "
        + " | ".join(str(decision) for decision in decisions)
        + " | T_open |"
    )
    lines.append("|---|---:|---:|" + "---:|" * len(decisions) + "---:|")
    for row in runs:
        cells = [_fmt(rec["open"]) for rec in row["checkpoints"]]
        lines.append(
            f"| {row['policy']} | {period_name(row['period'])} | {row['seed']} | "
            + " | ".join(cells)
            + f" | {_fmt(row['T_open'])} |"
        )
    lines.append("")
    lines.append("Mean over seeds at each checkpoint.")
    lines.append("")
    lines.append(
        "| policy | period | "
        + " | ".join(str(decision) for decision in decisions)
        + " | T_open |"
    )
    lines.append("|---|---:|" + "---:|" * len(decisions) + "---:|")
    for policy, period in _policy_keys():
        block = summary["by_policy"][f"{policy}/p{period_name(period)}"]
        cells = [_fmt(item["open_mean"]) for item in block["checkpoints"]]
        lines.append(
            f"| {policy} | {period_name(period)} | " + " | ".join(cells)
            + f" | {_fmt(block['T_open_mean'])} |"
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
        "| period | seed | oracle_learn_s − rand_dwell_s | "
        "oracle_new_s − rand_dwell_s | prog_age_s − rand_dwell_s |"
    )
    lines.append("|---:|---:|---:|---:|---:|")
    for gap in summary["gaps"]:
        lines.append(
            f"| {gap['period']} | {gap['seed']} | "
            f"{_fmt(gap['oracle_learn_s_minus_rand_dwell_s'])} | "
            f"{_fmt(gap['oracle_new_s_minus_rand_dwell_s'])} | "
            f"{_fmt(gap['prog_age_s_minus_rand_dwell_s'])} |"
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

    lines.append("## Births minus anchor")
    lines.append("")
    lines.append(
        "rand_dwell_s T_open with births minus rand_dwell_s T_open with birth_period None, "
        "on the same seed."
    )
    lines.append("")
    lines.append("| period | seed | with births | without births | births − anchor |")
    lines.append("|---:|---:|---:|---:|---:|")
    for period in PERIODS:
        for row in summary["births_minus_anchor"][str(period)]["rows"]:
            lines.append(
                f"| {period} | {row['seed']} | {_fmt(row['with_births'])} | "
                f"{_fmt(row['without_births'])} | {_fmt(row['births_minus_anchor'])} |"
            )
        block = summary["births_minus_anchor"][str(period)]
        lines.append(f"| {period} | mean |  |  | {_fmt(block['mean'])} |")
        lines.append(f"| {period} | std ddof=0 |  |  | {_fmt(block['std_ddof0'])} |")
    lines.append("")
    lines.append(
        "Mean over seeds of rand_dwell_s T_open at birth_period 128 minus T_open with no births: "
        f"{_fmt(summary['births_minus_anchor']['128']['mean'])}."
    )
    lines.append("")

    lines.append("## Newborn and older rooms")
    lines.append("")
    lines.append(
        "Bits/symbol at that checkpoint, learnable slots only. newborn is 1 when the slot's "
        "current room was born after the previous checkpoint and at or before this one. "
        "The previous checkpoint of the first one is decision 0. Initial rooms have born = 0. "
        "Means pool seeds and learnable slots."
    )
    lines.append("")
    lines.append(
        "| policy | period | checkpoint | newborn mean | newborn std ddof=0 | newborn n | "
        "older mean | older std ddof=0 | older n |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy, period in _policy_keys():
        block = summary["by_policy"][f"{policy}/p{period_name(period)}"]
        for item in block["checkpoints"]:
            lines.append(
                f"| {policy} | {period_name(period)} | {item['decision']} | "
                f"{_fmt(item['newborn_mean'])} | {_fmt(item['newborn_std_ddof0'])} | "
                f"{item['newborn_n']} | {_fmt(item['older_mean'])} | "
                f"{_fmt(item['older_std_ddof0'])} | {item['older_n']} |"
            )
    lines.append("")
    lines.append(
        "| policy | period | seed | checkpoint | slot | channel | kind | born | bits | newborn |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---:|---:|")
    for row in runs:
        for rec in row["checkpoints"]:
            for room in rec["rooms"]:
                lines.append(
                    f"| {row['policy']} | {period_name(row['period'])} | {row['seed']} | "
                    f"{rec['decision']} | {room['slot']} | {room['channel']} | {room['kind']} | "
                    f"{room['born']} | {_fmt(room['bits'])} | {room['newborn']} |"
                )
    lines.append("")

    lines.append("## Birth logs")
    lines.append("")
    lines.append(
        "One log per seed and period. The same log was checked against every policy, and "
        "against an independent draw from default_rng(world_seed + 90000): slot uniform on "
        "the 17 non-Z ring positions, then kind uniform on "
        + ", ".join(KINDS)
        + ". k starts at 0. A learnable newborn is h20_world.World(world_seed + 3000 + k). "
        "Kind N uses default_rng(world_seed + 95000 + k). birth_period None has no births. "
        "The channel column is the slot's initial room, which does not change."
    )
    lines.append("")
    lines.append("| period | seed | decision | slot | channel | dead kind | newborn kind | k |")
    lines.append("|---:|---:|---:|---:|---|---|---|---:|")
    orders = {}
    for row in result["runs"]:
        orders[row["seed"]] = row["room_order"]
    for period in PERIODS:
        for seed in seeds:
            row = _pick(result["runs"], "rand_dwell_s", period, seed)
            for event in row["birth_log"]:
                lines.append(
                    f"| {period} | {seed} | {event['decision']} | {event['slot']} | "
                    f"{orders[seed][event['slot']]} | {event['dead_kind']} | "
                    f"{event['newborn_kind']} | {event['k']} |"
                )
    lines.append("")

    lines.append("## Decision shares")
    lines.append("")
    lines.append(
        "A stay counts for the slot. Noise is the share of decisions that were stays in a "
        "slot whose room was N at that decision, initial N1–N3 or a newborn N. A move counts "
        "for the corridor. A sleep decision counts for sleep. Slot shares, corridor and sleep "
        "sum to 1. Noise is part of the slot shares."
    )
    lines.append("")
    slot_names = [f"s{i}" for i in range(N_RING)]
    lines.append(
        "| policy | period | " + " | ".join(slot_names)
        + " | noise | corridor | sleep |"
    )
    lines.append("|---|---:|" + "---:|" * (N_RING + 3))
    for policy, period in _policy_keys():
        block = summary["by_policy"][f"{policy}/p{period_name(period)}"]
        cells = [_fmt(item["mean"], 4) for item in block["slot_share_mean"]]
        cells.append(_fmt(block["noise_mean"], 4))
        cells.append(_fmt(block["corridor_mean"], 4))
        cells.append(_fmt(block["sleep_mean"], 4))
        lines.append(f"| {policy} | {period_name(period)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Per seed.")
    lines.append("")
    lines.append(
        "| policy | period | seed | " + " | ".join(slot_names)
        + " | noise | corridor | sleep |"
    )
    lines.append("|---|---:|---:|" + "---:|" * (N_RING + 3))
    for row in runs:
        cells = [f"{value:.4f}" for value in row["slot_share"]]
        cells.append(f"{row['noise_share']:.4f}")
        cells.append(f"{row['corridor_share']:.4f}")
        cells.append(f"{row['sleep_share']:.4f}")
        lines.append(
            f"| {row['policy']} | {period_name(row['period'])} | {row['seed']} | "
            + " | ".join(cells) + " |"
        )
    lines.append("")

    lines.append("## Birth latency")
    lines.append("")
    lines.append(
        "Decisions from a birth until the next stay in that slot. If the life ends first, "
        "the latency is L minus the birth decision and the birth is censored. The median "
        "and the mean include censored latencies."
    )
    lines.append("")
    lines.append("| policy | period | n | found | censored | median | mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for policy, period in _policy_keys():
        block = summary["by_policy"][f"{policy}/p{period_name(period)}"]["latency"]
        lines.append(
            f"| {policy} | {period_name(period)} | {block['n']} | {block['found']} | "
            f"{block['censored']} | {_fmt(block['median'])} | {_fmt(block['mean'])} |"
        )
    lines.append("")
    lines.append("| policy | period | seed | n | found | censored | median | mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for policy, period in _policy_keys():
        for seed in seeds:
            row = _pick(result["runs"], policy, period, seed)
            block = _latency_summary([row])
            lines.append(
                f"| {policy} | {period_name(period)} | {seed} | {block['n']} | {block['found']} | "
                f"{block['censored']} | {_fmt(block['median'])} | {_fmt(block['mean'])} |"
            )
    lines.append("")
    lines.append(
        "| policy | period | seed | decision | slot | dead kind | newborn kind | k | "
        "next stay | latency | censored |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|")
    for row in runs:
        for item in row["latency"]:
            nxt = "—" if item["next_stay"] is None else str(item["next_stay"])
            lines.append(
                f"| {row['policy']} | {period_name(row['period'])} | {row['seed']} | "
                f"{item['decision']} | {item['slot']} | {item['dead_kind']} | "
                f"{item['newborn_kind']} | {item['k']} | {nxt} | {item['latency']} | "
                f"{int(item['censored'])} |"
            )
    lines.append("")

    lines.append("## Ring order")
    lines.append("")
    lines.append(
        "Initial room name in ring positions 0..17. This is the slot's channel. "
        "One order per seed. Policies share it. Births do not change it."
    )
    lines.append("")
    lines.append("| seed | " + " | ".join(f"s{i}" for i in range(N_RING)) + " |")
    lines.append("|---:|" + "---|" * N_RING)
    for seed in seeds:
        lines.append(f"| {seed} | " + " | ".join(orders[seed]) + " |")
    lines.append("")

    lines.append("## Walls and kernels")
    lines.append("")
    lines.append("| kernel | seed | periods | wall seconds | workers |")
    lines.append("|---|---:|---|---:|---:|")
    for kernel in result["kernels"]:
        periods = ",".join(period_name(period) for period in kernel["periods"])
        lines.append(
            f"| {kernel['kernel_id']} | {kernel['seed']} | {periods} | "
            f"{kernel['wall_s']:.3f} | {kernel['workers']} |"
        )
    lines.append("")
    lines.append(
        "| policy | period | seed | wall seconds | learn seconds | score seconds | eval seconds |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in runs:
        lines.append(
            f"| {row['policy']} | {period_name(row['period'])} | {row['seed']} | "
            f"{row['wall_s']:.3f} | {row['learn_s']:.3f} | {row['score_s']:.3f} | "
            f"{row['eval_s']:.3f} |"
        )
    lines.append("")
    if result.get("kernel_status"):
        lines.append("```")
        lines.append(result["kernel_status"].rstrip())
        lines.append("```")
        lines.append("")

    lines.append("## Smoke test")
    lines.append("")
    lines.append("```")
    lines.append("\n".join(result["smoke_log"]).rstrip())
    lines.append("```")
    lines.append("")

    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "- World6(world_seed, birth_period) is World4(world_seed). birth_period None does not birth."
    )
    lines.append(
        "- After decision t, for every positive multiple t = birth_period, 2·birth_period, … "
        "that is at most L, one slot is drawn uniformly from the 17 that are not Z, in ring "
        "order, with default_rng(world_seed + 90000). The kind is then drawn uniformly from "
        + ", ".join(KINDS)
        + ". A learnable kind reads h20_world.World(world_seed + 3000 + k) from its beginning. "
        "Kind N is uniform on 41 symbols with default_rng(world_seed + 95000 + k). k counts "
        "births from 0. The slot keeps its ring position and its channel row, the initial "
        "room's row. Sleep and corridor decisions are on the same clock."
    )
    lines.append(
        "- A checkpoint at the same decision scores the world after that birth. The score is "
        "the mean over slots whose current room is learnable. EVAL=2048 symbols come from a "
        "copy of that room, from a zero state, on the slot's channel row. The live life then continues."
    )
    lines.append(
        "- The cortex, stores, coin, sleep dose and sleep placement are W5's, which are W4's. "
        "Policy rng is default_rng(1000 + world_seed). prog_age_s's choice rng is "
        "default_rng(10000 + world_seed), as in M8. Dwell is 8. A tie on the ring goes left."
    )
    lines.append(
        "- oracle_learn_s targets a learnable slot, uniform, at each choice. oracle_new_s keeps "
        "S, the learnable slots whose current room was born since it last arrived there. S starts "
        "as every learnable slot. The target is uniform over S, or over the learnable slots when "
        "S is empty. The first stay of the visit removes that slot from S. A later learnable "
        "birth puts it back."
    )
    lines.append(
        "- prog_age_s ranks slots by v + (v_max − v) · (1 − exp(−age / 256)) on the greedy step. "
        "The room reward is bits saved on the slot's exam glances over the visit, divided by the "
        "visit's decisions. Travel counts. Sleep is not a bandit update. The stored EMA is not aged."
    )
    lines.append(
        "- World seeds 0, 1, 2, 3, 4 and 5. L = 4096. birth_period 128 and 512 for every policy, "
        "plus rand_dwell_s with birth_period None. CPU. Private Kaggle kernels, pool of 4, no GPU, "
        "no internet. Nothing retuned."
    )
    lines.append("")
    return "\n".join(lines)


def _load_kernel_payloads(out_dir):
    out_dir = Path(out_dir)
    found = {}
    for seed in SEEDS:
        hits = sorted(out_dir.glob(f"s{seed}/w6_s{seed}.json"))
        hits += sorted(out_dir.glob(f"s{seed}-p128/w6_s{seed}.json"))
        hits += sorted(out_dir.glob(f"s{seed}-p512/w6_s{seed}.json"))
        if not hits:
            raise RuntimeError(f"no results file for seed {seed}")
        found[seed] = hits
    return found


def assemble(out_dir, smoke_payload=None):
    out_dir = Path(out_dir)
    if smoke_payload is None:
        smoke_payload = json.loads((out_dir / "smoke.json").read_text())
    if not smoke_payload.get("ok"):
        raise RuntimeError("smoke did not pass")
    runs = []
    kernels = []
    for seed, hits in _load_kernel_payloads(out_dir).items():
        seen = []
        for path in hits:
            payload = json.loads(path.read_text())
            if payload["seed"] != seed:
                raise RuntimeError(f"{path} seed {payload['seed']} != {seed}")
            for row in payload["runs"]:
                _check_row(row, seed, row["policy"], row["period"], row["L"])
                runs.append(row)
                seen.append((row["policy"], finite_period(row["period"])))
            kernels.append({
                "kernel_id": payload["kernel_id"],
                "seed": seed,
                "periods": [finite_period(job["period"]) for job in payload["jobs"]],
                "wall_s": payload["wall_s"],
                "workers": payload["workers"],
                "path": str(path.resolve().relative_to(ROOT)),
            })
        if sorted(seen, key=lambda item: (period_name(item[1]), item[0])) != sorted(
            expected_jobs(), key=lambda item: (period_name(item[1]), item[0])
        ):
            raise RuntimeError(f"seed {seed} jobs {seen} != {expected_jobs()}")
    if len(runs) != len(SEEDS) * len(expected_jobs()):
        raise RuntimeError(f"expected {len(SEEDS) * len(expected_jobs())} runs, got {len(runs)}")
    for period in PERIODS:
        for seed in SEEDS:
            logs = [
                _pick(runs, policy, period, seed)["birth_log"]
                for policy in POLICIES
            ]
            if any(log != logs[0] for log in logs):
                raise RuntimeError(f"policies disagree on the birth log, seed {seed} period {period}")
            orders = [
                _pick(runs, policy, period, seed)["room_order"]
                for policy in POLICIES
            ]
            anchor_order = _pick(runs, "rand_dwell_s", None, seed)["room_order"]
            if any(order != orders[0] for order in orders) or orders[0] != anchor_order:
                raise RuntimeError(f"ring order drifted, seed {seed}")
    summary = summarize(runs)
    kernels.sort(key=lambda item: (item["seed"], [period_name(p) for p in item["periods"]]))
    status_path = out_dir / "kernel_status.txt"
    result = {
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "periods": [128, 512, None],
        "L": L_SCORE,
        "model_seed": m7.MODEL_SEED,
        "n_ch": w4.N_CH,
        "sleep_share_target": SLEEP_SHARE,
        "sleep_decisions": w4.sleep_decisions(L_SCORE),
        "dataset_id": DATASET_ID,
        "kernels": kernels,
        "summary": summary,
        "smoke": {key: smoke_payload[key] for key in smoke_payload if key != "log"},
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
    parser.add_argument("--out", default=str(ROOT / "results" / "w6"))
    args = parser.parse_args()
    if args.command == "smoke":
        payload = smoke(args.out)
        stage(payload)
        return
    smoke_payload = json.loads((Path(args.out) / "smoke.json").read_text())
    if args.command == "stage":
        stage(smoke_payload)
        return
    assemble(args.out, smoke_payload)


if __name__ == "__main__":
    main()
