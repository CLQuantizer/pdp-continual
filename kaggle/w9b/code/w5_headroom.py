"""W5 headroom: W4's bank and fixed policies, on a ring whose learnable rules drift.

Numbers only. No verdict. The cortex, sleep, stores and movement are w4_headroom's.
T14 is the mean of W14 at eight checkpoints. The final W14 is the last checkpoint.
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
    if os.environ.get("_W5_MAIN") != "1":
        os.environ["_W5_MAIN"] = "1"
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
from h20_world import ALPHA
from w4_world import DARK, IDENTITIES, LEARN14, N_RING, NOISE
from w5_world import World5, finite_period

SEEDS = (0, 1, 2, 3, 4, 5)
POLICIES = ("rand_dwell_s", "oracle14_s", "oracle_fresh_s")
PERIODS = (128, 512)
L_SCORE = 2048
N_CHECKPOINTS = 8
SLEEP_SHARE = 0.20
KERNEL_WALL_LIMIT_S = 90 * 60
BITS_LIMIT = 1e-6
KERNEL_WORKERS = 4
ROOT = Path(__file__).resolve().parent
CODE_FILES = (
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
DATASET_ID = "ezzzzzx/pdp-code-w5"

if m7.DWELL != 8 or curio.B != 64 or curio.EVAL != 2048:
    raise RuntimeError("W5 inherited constants drifted")
if w4.N_CH != N_RING + 1:
    raise RuntimeError("W4 channel width drifted")


def period_name(period):
    period = finite_period(period)
    if period is None:
        return "inf"
    return str(int(period))


def checkpoints_of(L):
    L = int(L)
    if L % N_CHECKPOINTS != 0:
        raise ValueError(f"L={L} is not divisible by {N_CHECKPOINTS}")
    step = L // N_CHECKPOINTS
    return [step * i for i in range(1, N_CHECKPOINTS + 1)]


def job_list(split):
    """Lives in one kernel, or split into a period-128 kernel and the rest."""
    drifting = [(policy, period) for period in PERIODS for policy in POLICIES]
    anchor = [("rand_dwell_s", None)]
    if not split:
        return [("all", drifting + anchor)]
    return [
        ("p128", [(policy, 128) for policy in POLICIES]),
        ("p512", [(policy, 512) for policy in POLICIES] + anchor),
    ]


def expected_jobs():
    return [(policy, period) for period in PERIODS for policy in POLICIES] + [
        ("rand_dwell_s", None)
    ]


def fresh_targets(rooms, pending):
    """Ring positions. Uniform over S when S is non-empty, else over the 14."""
    if pending:
        targets = [i for i, name in enumerate(rooms) if name in pending]
    else:
        targets = [
            i for i, name in enumerate(rooms) if w4.split_learnable(name) is not None
        ]
    if not targets:
        raise RuntimeError("oracle_fresh_s has no target rooms")
    return targets


def target_after(room, actions, n_ring):
    room = int(room)
    for action in actions:
        if action == "left":
            room = (room - 1) % n_ring
        elif action == "right":
            room = (room + 1) % n_ring
        elif action != "stay":
            raise ValueError(action)
    return room


def s_names(rooms, pending):
    return [name for name in rooms if name in pending]


def take_action(state, world, policy, rng, pending, trace, t):
    """One action. A new plan is drawn only when the current plan is empty."""
    n_ring = N_RING
    if not state["plan"]:
        if policy == "oracle_fresh_s":
            targets = fresh_targets(world.rooms, pending)
            shown = s_names(world.rooms, pending)
        else:
            targets = w4.targets_for(world.rooms, policy)
            shown = None
        room_before = state["room"]
        action = w4.next_w4(state, n_ring, targets, rng)
        target = target_after(room_before, [action] + state["plan"], n_ring)
        state["target"] = target
        state["arrived"] = False
        if shown is not None:
            target_name = world.rooms[target]
            if shown and target_name not in shown:
                raise RuntimeError("target was drawn outside S")
            if not shown and w4.split_learnable(target_name) is None:
                raise RuntimeError("fallback target is not learnable")
            if trace is not None:
                trace.append({
                    "decision": int(t + 1),
                    "target": target_name,
                    "S": list(shown),
                })
        return action
    return w4.next_w4(state, n_ring, [0], rng)


def swapped_since(swap_log, prev, decision):
    names = []
    seen = set()
    for event in swap_log:
        if prev < int(event["decision"]) <= int(decision) and event["room"] not in seen:
            seen.add(event["room"])
            names.append(event["room"])
    return names


def take_checkpoint(life, decision, prev, clock):
    """W14 of a frozen forward pass. The live cortex and the live world stay put."""
    params = life.cortex.clone_params()
    live = life.cortex.clone_live()
    t0 = time.perf_counter()
    w14, wa, wb, held = w4.evaluate_w14(life.cortex, life.world, life.index)
    clock["eval"] += time.perf_counter() - t0
    if life.cortex.param_delta(params) > 0.0:
        raise RuntimeError("checkpoint scoring changed parameters")
    if not life.cortex.live_matches(live):
        raise RuntimeError("checkpoint scoring changed the live state")
    swapped = swapped_since(life.world.swap_log, prev, decision)
    return {
        "decision": int(decision),
        "W14": float(w14),
        "Wa": float(wa),
        "Wb": float(wb),
        "held": {name: float(held[name]) for name in LEARN14},
        "swapped": list(swapped),
    }


def sleep_visit(life, seed, policy, period, channels, length, t, L, check_sleep, checked, on_decision):
    """w4's sleep visit, with a hook after every decision."""
    if length <= 0:
        raise RuntimeError("empty sleep visit")
    snap = life.begin_sleep_visit() if (check_sleep and not checked) else None
    n = 0
    learned = 0
    while n < length and t < L:
        learned += int(life.sleep_decision())
        channels.append("sleep")
        n += 1
        t += 1
        on_decision(t)
        if t % 256 == 0:
            print(f"s{seed} {policy} p={period_name(period)} t={t}/{L} sleep", flush=True)
    if snap is not None:
        delta = life.end_sleep_visit(snap)
        checked = True
        if learned and delta <= 0.0:
            raise RuntimeError("sleep visit learned and left the parameters unchanged")
    if n != length and t < L:
        raise RuntimeError("sleep visit ended early")
    return n, t, checked


def run_life(seed, policy, L, period, score=True, check_sleep=False, trace=None):
    """One fixed policy. Swaps and checkpoints sit on the decision clock, including sleep."""
    L = int(L)
    period = finite_period(period)
    if policy not in POLICIES:
        raise ValueError(policy)
    if period is None and policy != "rand_dwell_s":
        raise ValueError("the no-drift anchor is rand_dwell_s")
    n_sleep = w4.sleep_decisions(L)
    plan = m7.sleep_plan(L, n_sleep)
    if any(length != m7.DWELL for _, length in plan):
        raise RuntimeError("a fixed sleep visit is not 8 decisions")
    points = checkpoints_of(L)
    point_set = set(points)
    world = World5(seed, period)
    names = list(IDENTITIES)
    index = {name: i for i, name in enumerate(names)}
    t0 = time.perf_counter()
    life = w4.Life(seed, world, names, index, N_RING, w4.N_CH, list(LEARN14))
    rng = np.random.default_rng(1_000 + int(seed))
    state = {"kind": policy, "room": 0, "plan": [], "target": None, "arrived": True}
    # S is the learnable rooms whose rule changed since the agent last arrived there.
    # It starts as all 14. Arrival removes the room. A later swap puts it back.
    pending_rooms = set(LEARN14) if policy == "oracle_fresh_s" else None
    channels = []
    records = []
    clock = {"eval": 0.0}
    prev_cp = {"t": 0}
    visit = {"pending": None}
    si = 0
    t = 0
    checked = False

    def on_decision(done):
        event = world.maybe_swap(done)
        if event is not None and pending_rooms is not None:
            pending_rooms.add(event["room"])
        if score and done in point_set:
            rec = take_checkpoint(life, done, prev_cp["t"], clock)
            prev_cp["t"] = done
            records.append(rec)
            print(
                f"s{seed} {policy} p={period_name(period)} checkpoint t={done}/{L} "
                f"W14 {rec['W14']:.4f}",
                flush=True,
            )

    while t < L:
        if not state["plan"] and w4.sleep_due(t, L, plan, si, N_RING):
            visit["pending"] = w4._close_visit(life, visit["pending"])
            _n, t, checked = sleep_visit(
                life, seed, policy, period, channels, plan[si][1], t, L,
                check_sleep, checked, on_decision,
            )
            si += 1
            continue
        action = take_action(state, world, policy, rng, pending_rooms, trace, t)
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
        if channel == "corridor":
            life.corridor_decision(symbols)
        else:
            life.room_decision(symbols, channel)
            visit["pending"] = channel
        channels.append(channel)
        t += 1
        on_decision(t)
        if t % 256 == 0:
            print(f"s{seed} {policy} p={period_name(period)} t={t}/{L}", flush=True)
        if not state["plan"]:
            visit["pending"] = w4._close_visit(life, visit["pending"])
    if visit["pending"] is not None:
        life.snapshot(visit["pending"])
    if len(channels) != L:
        raise RuntimeError(f"{policy} ran {len(channels)} decisions")
    if si != len(plan):
        raise RuntimeError(f"{policy} took {si} of {len(plan)} planned sleep visits")
    if life.exam_drawn != 0:
        raise RuntimeError(f"sleep drew {life.exam_drawn} exam glances")
    if period is None:
        if world.swap_log:
            raise RuntimeError("a life with no drift swapped a room")
    elif [event["decision"] for event in world.swap_log] != list(range(period, L + 1, period)):
        raise RuntimeError("swap decisions are not period, 2·period, …")
    share, corridor_share, sleep_share = w4.shares_of(channels, life.names)
    if abs((sum(share.values()) + corridor_share + sleep_share) - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    if sum(name == "sleep" for name in channels) != n_sleep:
        raise RuntimeError("sleep decisions do not match the dose")
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
    return {
        "seed": int(seed),
        "policy": policy,
        "period": None if period is None else int(period),
        "L": int(L),
        "T14": None if t14 is None else float(t14),
        "W14": None if w14 is None else float(w14),
        "checkpoints": records,
        "swap_log": [
            {"decision": int(event["decision"]), "room": event["room"], "k": int(event["k"])}
            for event in world.swap_log
        ],
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
    }


def _job(args):
    seed, policy, period, L = args
    w4._threads()
    print(
        f"s{seed} {policy} p={period_name(period)} L={L} start dose {w4.sleep_decisions(L)}",
        flush=True,
    )
    row = run_life(seed, policy, L, period, score=True, check_sleep=True)
    print(
        f"s{seed} {policy} p={period_name(period)} L={L} T14 {row['T14']:.4f} "
        f"W14 {row['W14']:.4f} sleep {row['sleep_share']:.4f} "
        f"corridor {row['corridor_share']:.3f} wall {row['wall_s']:.1f}s "
        f"learn {row['learn_s']:.1f}s score {row['score_s']:.1f}s eval {row['eval_s']:.1f}s",
        flush=True,
    )
    return row


def mean_std(values):
    return w4.mean_std(values)


def _threads():
    w4._threads()


def as_text(symbols):
    return "".join(ALPHA[int(s)] for s in symbols)


def expected_swap_log(world_seed, period, L):
    """Independent of World5, so a clock bug cannot confirm itself."""
    rng = np.random.default_rng(int(world_seed) + 80_000)
    rooms = list(LEARN14)
    out = []
    for k, decision in enumerate(range(int(period), int(L) + 1, int(period))):
        name = rooms[int(rng.integers(len(rooms)))]
        out.append({"decision": int(decision), "room": name, "k": int(k)})
    return out


def smoke(out_dir=None):
    """VPS checks. A failed check does not stage kernels."""
    _threads()
    out_dir = Path(out_dir or (ROOT / "results" / "w5"))
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
        for L in (64, 128, 256, 512, L_SCORE):
            dose = w4.sleep_decisions(L)
            log(f"dose L{L} {dose}/{L} share {dose / L:.6f} checkpoints {checkpoints_of(L)}")
        if World5(0, math.inf).period is not None or World5(0, "inf").period is not None:
            raise RuntimeError("period inf was stored as a finite period")

        log("smoke check 1a: period inf matches World4 glances, seed 0, 200 decisions")
        base = w4_world.World4(0)
        live = World5(0, math.inf)
        if base.rooms != live.rooms or base.perm != live.perm:
            raise RuntimeError("period inf ring does not match World4")
        fresh_a = w4_world.World4(0)
        fresh_b = World5(0, math.inf)
        for name in list(LEARN14) + list(NOISE) + [DARK]:
            if list(fresh_a.glance_room(name)) != list(fresh_b.glance_room(name)):
                raise RuntimeError(f"period inf first glance differs for {name}")
        rng = np.random.default_rng(1_000)
        state = {"room": 0, "plan": []}
        targets = w4.targets_for(base.rooms, "rand_dwell_s")
        for decision in range(1, 201):
            action = w4.next_w4(state, N_RING, targets, rng)
            r1, c1, s1 = base.step(action, None)
            r2, c2, s2 = live.step(action, None)
            state["room"] = r1
            if live.maybe_swap(decision) is not None:
                raise RuntimeError("period inf swapped")
            if r1 != r2 or c1 != c2 or list(s1) != list(s2):
                raise RuntimeError(f"glance mismatch at decision {decision}")
        log("  200 decisions match World4, including stays, moves and the corridor")

        log("smoke check 2: period 16, seed 0, L=128 swap log, new stream, copy")
        world = World5(0, 16)
        saved = None
        for decision in range(1, 129):
            world.step("left", None)
            pre = world.copy() if saved is None and decision % 16 == 0 else None
            event = world.maybe_swap(decision)
            if event is not None and saved is None:
                saved = (pre, world.copy(), dict(event))
        expect = expected_swap_log(0, 16, 128)
        if world.swap_log != expect:
            raise RuntimeError("swap log does not match an independent draw")
        for event in world.swap_log:
            log(f"  swap decision {event['decision']} room {event['room']} k {event['k']}")
        pre, post, event = saved
        name = event["room"]
        k = int(event["k"])
        proc, _side = w4.split_learnable(name)
        other = next(room for room in LEARN14 if room != name)
        # Peek from copies. A glance of the swapped room on the shared h20 world
        # would advance that world's rng and make a later peek of another room lie.
        other_same = list(pre.copy().glance_room(other)) == list(post.copy().glance_room(other))
        old = list(pre.copy().glance_room(name))
        fresh = h20_world.World(0 + 2_000 + k)
        expect1 = list(curio.glance(fresh, proc, None))
        expect2 = list(curio.glance(fresh, proc, None))
        copy_at_start = post.copy()
        first_live = list(post.glance_room(name))
        first_copy = list(copy_at_start.glance_room(name))
        copy_after = post.copy()
        second_live = list(post.glance_room(name))
        second_later = list(copy_after.glance_room(name))
        second_start = list(copy_at_start.glance_room(name))
        log(f"  room {name} proc {proc} new World({2000 + k})")
        log(f"  before {as_text(old)!r}")
        log(f"  after  {as_text(first_live)!r}")
        log(
            f"  new-instance match {first_live == expect1} "
            f"differs-from-old {old != first_live} other-room-unchanged {other_same}"
        )
        log(
            f"  copy at swap matches {first_copy == first_live == expect1} "
            f"copy after one glance matches {second_later == second_live == second_start == expect2}"
        )
        if first_live != expect1 or first_copy != expect1 or old == first_live or not other_same:
            raise RuntimeError("swapped glance is not the new instance")
        if second_live != expect2 or second_later != expect2 or second_start != expect2:
            raise RuntimeError("copy() did not continue the current rule")
        if pre.rooms != post.rooms or pre.rooms != world.rooms:
            raise RuntimeError("a swap changed the ring")

        log("smoke check 1b: rand_dwell_s period inf L=64 final W14 equals W4")
        ref = w4.run_life(0, "rand_dwell_s", 64, score=True, check_sleep=True)
        matched = run_life(0, "rand_dwell_s", 64, math.inf, score=True, check_sleep=True)
        gap = abs(matched["W14"] - ref["W14"])
        log(
            f"  W4 W14 {ref['W14']:.8f} W5 W14 {matched['W14']:.8f} "
            f"abs {gap:.3e} limit {BITS_LIMIT:.1e} T14 {matched['T14']:.6f}"
        )
        if gap > BITS_LIMIT:
            raise RuntimeError(f"period inf final W14 differs by {gap:.3e}")
        if not matched["finite"] or matched["swap_log"]:
            raise RuntimeError("period inf L=64 life is not a finite anchor")
        payload["anchor_L64"] = {
            "W4_W14": float(ref["W14"]),
            "W5_W14": float(matched["W14"]),
            "abs": gap,
            "T14": float(matched["T14"]),
        }

        log("smoke check 3: oracle_fresh_s period 16 seed 0 L=128 targets and S")
        trace = []
        traced = run_life(
            0, "oracle_fresh_s", 128, 16, score=False, check_sleep=True, trace=trace,
        )
        if traced["swap_log"] != expected_swap_log(0, 16, 128):
            raise RuntimeError("oracle_fresh_s swap log does not match the independent draw")
        if not trace or len(trace[0]["S"]) != 14:
            raise RuntimeError("S did not start as all 14")
        for choice in trace:
            log(
                f"  choice decision {choice['decision']} target {choice['target']} "
                f"S={choice['S']}"
            )
        log(
            f"  choices {len(trace)} sleep {traced['sleep_decisions']} "
            f"swaps {len(traced['swap_log'])} wall {traced['wall_s']:.2f}s"
        )
        payload["oracle_fresh_trace"] = {
            "n_choices": len(trace),
            "sleep_decisions": traced["sleep_decisions"],
            "swap_log": traced["swap_log"],
        }

        log("smoke check 4: three policies, seed 0, period 128, L=256, finite T14")
        for policy in POLICIES:
            row = run_life(policy=policy, seed=0, L=256, period=128, score=True, check_sleep=True)
            log(
                f"  {policy} T14 {row['T14']:.6f} W14 {row['W14']:.6f} "
                f"finite {row['finite']} sleep {row['sleep_share']:.4f} "
                f"decisions {row['sleep_decisions']} corridor {row['corridor_share']:.4f} "
                f"swaps {len(row['swap_log'])} wall {row['wall_s']:.2f}s"
            )
            if not row["finite"]:
                raise RuntimeError(f"{policy} T14 is not finite")
            if len(row["checkpoints"]) != N_CHECKPOINTS:
                raise RuntimeError(f"{policy} did not score 8 checkpoints")
            if policy != "rand_dwell_s":
                for noise in list(NOISE) + [DARK]:
                    if row["share"][noise] != 0.0:
                        raise RuntimeError(f"{policy} stayed in {noise}")

        log("smoke check 5: time rand_dwell_s seed 0 period 128 L=512, with checkpoints")
        timed = run_life(0, "rand_dwell_s", 512, 128, score=True, check_sleep=True)
        if not timed["finite"]:
            raise RuntimeError("timed life T14 is not finite")
        scale_l = L_SCORE / 512.0
        scale_score = w4.sleep_decisions(L_SCORE) / timed["sleep_decisions"]
        one_linear = (timed["learn_s"] + timed["score_s"]) * scale_l + timed["eval_s"]
        one_scaled = timed["learn_s"] * scale_l + timed["score_s"] * scale_score + timed["eval_s"]
        one = max(one_linear, one_scaled)
        n_jobs = len(expected_jobs())
        waves = math.ceil(n_jobs / KERNEL_WORKERS)
        kernel_s = one * waves
        guarded = kernel_s * 2.0
        one_guarded = one * 2.0
        if one_guarded > KERNEL_WALL_LIMIT_S:
            raise RuntimeError(
                f"one L=2048 life estimate {one_guarded:.0f}s exceeds {KERNEL_WALL_LIMIT_S:.0f}s"
            )
        split = guarded > KERNEL_WALL_LIMIT_S
        if split:
            log("  guarded kernel wall exceeds 90 min; splitting by period")
        else:
            log("  guarded kernel wall fits in 90 min; one kernel per seed runs every life")
        log(
            f"  rand_dwell_s p=128 L=512 T14 {timed['T14']:.6f} W14 {timed['W14']:.6f} "
            f"wall {timed['wall_s']:.2f}s learn {timed['learn_s']:.2f}s "
            f"score {timed['score_s']:.2f}s eval {timed['eval_s']:.2f}s "
            f"sleep {timed['sleep_decisions']} swaps {len(timed['swap_log'])}"
        )
        log(
            f"  one L2048 linear {one_linear:.1f}s scaled {one_scaled:.1f}s "
            f"waves {waves} jobs {n_jobs} kernel {kernel_s:.1f}s guarded {guarded:.1f}s "
            f"limit {KERNEL_WALL_LIMIT_S:.0f}s split {split}"
        )
        payload["timed_512"] = {
            "seed": 0,
            "policy": "rand_dwell_s",
            "period": 128,
            "L": 512,
            "T14": timed["T14"],
            "W14": timed["W14"],
            "wall_s": timed["wall_s"],
            "learn_s": timed["learn_s"],
            "score_s": timed["score_s"],
            "eval_s": timed["eval_s"],
            "sleep_decisions": timed["sleep_decisions"],
        }
        payload["estimate"] = {
            "one_linear_s": one_linear,
            "one_scaled_s": one_scaled,
            "one_s": one,
            "waves": waves,
            "jobs": n_jobs,
            "workers": KERNEL_WORKERS,
            "kernel_s": kernel_s,
            "guarded_s": guarded,
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
        "code_file": "w5_kernel.py",
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
    text = f'''"""W5 seed {seed}. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_W5_KERNEL") != "1":
    os.environ["_W5_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("w5_headroom.py"))
    if not hits:
        raise SystemExit("w5_headroom.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import w5_headroom
    w5_headroom.kernel_main(
        seed={seed},
        out="/kaggle/working",
        workers=4,
        kernel_id="{kernel_id}",
        jobs=({jobs_src},),
    )


if __name__ == "__main__":
    main()
'''
    (directory / "w5_kernel.py").write_text(text)
    return kernel_id


def stage(smoke_payload, kaggle_dir=None):
    """Write the private dataset folder and the kernel folders. Does not push."""
    if not smoke_payload.get("ok"):
        raise SystemExit("smoke did not pass; kernels not staged")
    split = bool(smoke_payload["estimate"]["split"])
    kaggle_dir = Path(kaggle_dir or (ROOT / "kaggle" / "w5"))
    if kaggle_dir.exists():
        shutil.rmtree(kaggle_dir)
    code = kaggle_dir / "code"
    code.mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "pdp code w5",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    ids = []
    groups = job_list(split)
    for seed in SEEDS:
        for label, jobs in groups:
            if not split:
                slug = f"w5-s{seed}"
                folder = f"s{seed}"
            elif label == "p128":
                slug = f"w5-s{seed}-p128"
                folder = f"s{seed}-p128"
            elif label == "p512":
                slug = f"w5-s{seed}-p512"
                folder = f"s{seed}-p512"
            else:
                raise RuntimeError(f"unexpected split label {label}")
            ids.append(_write_kernel(kaggle_dir / folder, seed, jobs, slug))
    print(f"staged {len(ids)} kernels split {split}", flush=True)
    for kernel_id in ids:
        print(kernel_id, flush=True)
    return ids


def kernel_main(seed, out, workers=4, kernel_id=None, jobs=None):
    if __name__ != "w5_headroom":
        raise SystemExit("import w5_headroom and call kernel_main so the pool can spawn workers")
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
        raise SystemExit(f"jobs {jobs} are not a W5 kernel")
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
    path = out / f"w5_s{seed}.json"
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
    if set(row["share"]) != set(IDENTITIES):
        raise RuntimeError("share keys are not the 18 rooms")
    total = sum(row["share"].values()) + row["corridor_share"] + row["sleep_share"]
    if abs(total - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    if policy in ("oracle14_s", "oracle_fresh_s"):
        for name in list(NOISE) + [DARK]:
            if row["share"][name] != 0.0:
                raise RuntimeError(f"{policy} stayed in {name}")
    points = checkpoints_of(L)
    if [rec["decision"] for rec in row["checkpoints"]] != points:
        raise RuntimeError("checkpoint decisions drifted")
    if abs(row["W14"] - row["checkpoints"][-1]["W14"]) > 1e-12:
        raise RuntimeError("final W14 is not the last checkpoint")
    if abs(row["T14"] - float(np.mean([rec["W14"] for rec in row["checkpoints"]]))) > 1e-12:
        raise RuntimeError("T14 is not the mean of the checkpoints")
    for rec in row["checkpoints"]:
        if set(rec["held"]) != set(LEARN14):
            raise RuntimeError("checkpoint rooms are not the 14")
        if any(name not in LEARN14 for name in rec["swapped"]):
            raise RuntimeError("a swapped name is not learnable")
    decisions = [event["decision"] for event in row["swap_log"]]
    ks = [event["k"] for event in row["swap_log"]]
    if period is None:
        if row["swap_log"]:
            raise RuntimeError("the anchor swapped")
    else:
        if decisions != list(range(period, int(L) + 1, period)):
            raise RuntimeError("swap decisions drifted")
        if ks != list(range(len(row["swap_log"]))):
            raise RuntimeError("swap k drifted")
        if row["swap_log"] != expected_swap_log(seed, period, L):
            raise RuntimeError("swap log does not match the independent draw")


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


def summarize(runs):
    by = {}
    for policy, period in _policy_keys():
        rows = [_pick(runs, policy, period, seed) for seed in SEEDS]
        t_mu, t_sd = mean_std([row["T14"] for row in rows])
        w_mu, w_sd = mean_std([row["W14"] for row in rows])
        share_mu = {}
        for name in IDENTITIES:
            mu, sd = mean_std([row["share"][name] for row in rows])
            share_mu[name] = mu
        c_mu, c_sd = mean_std([row["corridor_share"] for row in rows])
        s_mu, s_sd = mean_std([row["sleep_share"] for row in rows])
        cps = []
        for index, decision in enumerate(checkpoints_of(L_SCORE)):
            mu, sd = mean_std([row["checkpoints"][index]["W14"] for row in rows])
            swapped_bits = []
            kept_bits = []
            for row in rows:
                rec = row["checkpoints"][index]
                if rec["decision"] != decision:
                    raise RuntimeError("checkpoint order drifted")
                swapped = set(rec["swapped"])
                for name in LEARN14:
                    if name in swapped:
                        swapped_bits.append(rec["held"][name])
                    else:
                        kept_bits.append(rec["held"][name])
            s_bits, s_bits_sd = mean_std(swapped_bits)
            k_bits, k_bits_sd = mean_std(kept_bits)
            cps.append({
                "decision": decision,
                "T14_mean": mu,
                "T14_std_ddof0": sd,
                "swapped_mean": s_bits,
                "swapped_std_ddof0": s_bits_sd,
                "swapped_n": len(swapped_bits),
                "unswapped_mean": k_bits,
                "unswapped_std_ddof0": k_bits_sd,
                "unswapped_n": len(kept_bits),
            })
        by[f"{policy}/p{period_name(period)}"] = {
            "policy": policy,
            "period": None if period is None else int(period),
            "n": len(rows),
            "T14_mean": t_mu,
            "T14_std_ddof0": t_sd,
            "W14_mean": w_mu,
            "W14_std_ddof0": w_sd,
            "share_mean": share_mu,
            "corridor_mean": c_mu,
            "corridor_std_ddof0": c_sd,
            "sleep_mean": s_mu,
            "sleep_std_ddof0": s_sd,
            "checkpoints": cps,
        }
    gaps = {128: [], 512: []}
    for period in PERIODS:
        for seed in SEEDS:
            rand = _pick(runs, "rand_dwell_s", period, seed)
            broad = _pick(runs, "oracle14_s", period, seed)
            fresh = _pick(runs, "oracle_fresh_s", period, seed)
            gaps[period].append({
                "seed": int(seed),
                "period": int(period),
                "oracle_fresh_s_minus_rand_dwell_s": float(fresh["T14"] - rand["T14"]),
                "oracle14_s_minus_rand_dwell_s": float(broad["T14"] - rand["T14"]),
                "T14": {
                    "rand_dwell_s": rand["T14"],
                    "oracle14_s": broad["T14"],
                    "oracle_fresh_s": fresh["T14"],
                },
            })
    gap_summary = {}
    for period in PERIODS:
        block = {}
        for key in ("oracle_fresh_s_minus_rand_dwell_s", "oracle14_s_minus_rand_dwell_s"):
            mu, sd = mean_std([row[key] for row in gaps[period]])
            block[key] = {
                "mean": mu,
                "std_ddof0": sd,
                "left_lower": int(sum(row[key] < 0.0 for row in gaps[period])),
                "n": len(gaps[period]),
            }
        gap_summary[str(period)] = block
    drift = {}
    anchor_rows = {seed: _pick(runs, "rand_dwell_s", None, seed) for seed in SEEDS}
    for period in PERIODS:
        rows = []
        for seed in SEEDS:
            rand = _pick(runs, "rand_dwell_s", period, seed)
            rows.append({
                "seed": int(seed),
                "period": int(period),
                "with_drift": rand["T14"],
                "without_drift": anchor_rows[seed]["T14"],
                "drift_minus_anchor": float(rand["T14"] - anchor_rows[seed]["T14"]),
            })
        mu, sd = mean_std([row["drift_minus_anchor"] for row in rows])
        drift[str(period)] = {
            "mean": mu,
            "std_ddof0": sd,
            "n": len(rows),
            "rows": rows,
        }
    precondition_mean = drift["128"]["mean"]
    return {
        "by_policy": by,
        "gaps": gaps[128] + gaps[512],
        "gap_summary": gap_summary,
        "drift_minus_anchor": drift,
        "precondition": {
            "period": 128,
            "policy": "rand_dwell_s",
            "mean_drift_minus_anchor": precondition_mean,
            "std_ddof0": drift["128"]["std_ddof0"],
            "label": None if precondition_mean >= 0.10 else "drift too weak",
        },
    }


def w4_reference():
    path = ROOT / "results" / "w4" / "results.json"
    data = json.loads(path.read_text())
    rows = []
    for row in data["runs"]:
        if int(row["L"]) != L_SCORE or row["policy"] not in ("rand_dwell_s", "oracle14_s"):
            continue
        rows.append({
            "policy": row["policy"],
            "seed": int(row["seed"]),
            "L": L_SCORE,
            "W14": float(row["W14"]),
            "Wa": float(row["Wa"]),
            "Wb": float(row["Wb"]),
        })
    out = {"source": "results/w4/results.json", "L": L_SCORE, "rows": rows}
    for policy in ("rand_dwell_s", "oracle14_s"):
        chosen = [row for row in rows if row["policy"] == policy]
        if len(chosen) != len(SEEDS):
            raise RuntimeError(f"W4 reference {policy} has {len(chosen)} seeds")
        mu, sd = mean_std([row["W14"] for row in chosen])
        out[policy] = {"W14_mean": mu, "W14_std_ddof0": sd, "n": len(chosen)}
    return out


def render_markdown(result):
    summary = result["summary"]
    seeds = list(result["seeds"])
    lines = []
    lines.append("# W5 headroom")
    lines.append("")
    lines.append(
        "Fixed policies on W4's 18-room ring. Every `period` decisions, one of the 14 "
        "learnable rooms is given a fresh draw of its own process. T14 is the mean of W14 "
        "at decisions L/8, 2L/8, …, L. W14 is bits/symbol of the frozen bank on a fresh "
        "EVAL=2048 continuation of the 14 learnable rooms, from a zero state, with that "
        "room's channel embedding, on a copy of the world at that checkpoint. The final W14 "
        "is the checkpoint at L. Lower is better. The anchor is rand_dwell_s with period inf, "
        "the same life as W4, scored with the eight checkpoints."
    )
    lines.append("")
    dose = w4.sleep_decisions(L_SCORE)
    lines.append(
        f"Every policy sleeps for the fixed share 0.20, in visits of 8, placed as in M7b. "
        f"At L = {L_SCORE} that is {dose} decisions, share {dose / L_SCORE:.6f}."
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

    block_table("T14", "T14")
    block_table("Final W14", "W14")

    lines.append("## T14 per checkpoint")
    lines.append("")
    decisions = checkpoints_of(L_SCORE)
    lines.append(
        "| policy | period | seed | "
        + " | ".join(str(decision) for decision in decisions)
        + " | T14 |"
    )
    lines.append("|---|---:|---:|" + "---:|" * len(decisions) + "---:|")
    runs = sorted(result["runs"], key=lambda row: (
        _policy_keys().index((row["policy"], finite_period(row["period"]))), row["seed"],
    ))
    for row in runs:
        cells = [_fmt(rec["W14"]) for rec in row["checkpoints"]]
        lines.append(
            f"| {row['policy']} | {period_name(row['period'])} | {row['seed']} | "
            + " | ".join(cells)
            + f" | {_fmt(row['T14'])} |"
        )
    lines.append("")
    lines.append("Mean over seeds at each checkpoint.")
    lines.append("")
    lines.append(
        "| policy | period | "
        + " | ".join(str(decision) for decision in decisions)
        + " | T14 |"
    )
    lines.append("|---|---:|" + "---:|" * len(decisions) + "---:|")
    for policy, period in _policy_keys():
        block = summary["by_policy"][f"{policy}/p{period_name(period)}"]
        cells = [_fmt(item["T14_mean"]) for item in block["checkpoints"]]
        lines.append(
            f"| {policy} | {period_name(period)} | " + " | ".join(cells)
            + f" | {_fmt(block['T14_mean'])} |"
        )
    lines.append("")

    lines.append("## Gaps")
    lines.append("")
    lines.append(
        "On T14. A negative gap means the first policy is lower. "
        "left lower counts seeds where the gap is negative."
    )
    lines.append("")
    lines.append(
        "| period | seed | oracle_fresh_s − rand_dwell_s | oracle14_s − rand_dwell_s |"
    )
    lines.append("|---:|---:|---:|---:|")
    for gap in summary["gaps"]:
        lines.append(
            f"| {gap['period']} | {gap['seed']} | "
            f"{_fmt(gap['oracle_fresh_s_minus_rand_dwell_s'])} | "
            f"{_fmt(gap['oracle14_s_minus_rand_dwell_s'])} |"
        )
    for period in PERIODS:
        block = summary["gap_summary"][str(period)]
        lines.append(
            f"| {period} | mean | "
            f"{_fmt(block['oracle_fresh_s_minus_rand_dwell_s']['mean'])} | "
            f"{_fmt(block['oracle14_s_minus_rand_dwell_s']['mean'])} |"
        )
        lines.append(
            f"| {period} | std ddof=0 | "
            f"{_fmt(block['oracle_fresh_s_minus_rand_dwell_s']['std_ddof0'])} | "
            f"{_fmt(block['oracle14_s_minus_rand_dwell_s']['std_ddof0'])} |"
        )
        lines.append(
            f"| {period} | left lower (count) | "
            f"{block['oracle_fresh_s_minus_rand_dwell_s']['left_lower']} of "
            f"{block['oracle_fresh_s_minus_rand_dwell_s']['n']} | "
            f"{block['oracle14_s_minus_rand_dwell_s']['left_lower']} of "
            f"{block['oracle14_s_minus_rand_dwell_s']['n']} |"
        )
    lines.append("")

    lines.append("## Drift minus anchor")
    lines.append("")
    lines.append(
        "rand_dwell_s T14 with drift minus rand_dwell_s T14 with period inf, on the same seed."
    )
    lines.append("")
    lines.append("| period | seed | with drift | without drift | drift − anchor |")
    lines.append("|---:|---:|---:|---:|---:|")
    for period in PERIODS:
        for row in summary["drift_minus_anchor"][str(period)]["rows"]:
            lines.append(
                f"| {period} | {row['seed']} | {_fmt(row['with_drift'])} | "
                f"{_fmt(row['without_drift'])} | {_fmt(row['drift_minus_anchor'])} |"
            )
        block = summary["drift_minus_anchor"][str(period)]
        lines.append(
            f"| {period} | mean |  |  | {_fmt(block['mean'])} |"
        )
        lines.append(
            f"| {period} | std ddof=0 |  |  | {_fmt(block['std_ddof0'])} |"
        )
    lines.append("")
    pre = summary["precondition"]
    lines.append(
        "Mean over seeds of rand_dwell_s T14 at period 128 minus T14 with no drift: "
        f"{_fmt(pre['mean_drift_minus_anchor'])}."
    )
    lines.append("")
    if pre["label"]:
        lines.append(pre["label"])
        lines.append("")

    lines.append("## Per-room checkpoint bits")
    lines.append("")
    lines.append(
        "Bits/symbol at that checkpoint. swapped is 1 when the room's rule was replaced "
        "after the previous checkpoint and at or before this one. The previous checkpoint "
        "of the first one is decision 0. Means pool seeds and rooms."
    )
    lines.append("")
    lines.append(
        "| policy | period | checkpoint | swapped mean | swapped std ddof=0 | swapped n | "
        "unswapped mean | unswapped std ddof=0 | unswapped n |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for policy, period in _policy_keys():
        block = summary["by_policy"][f"{policy}/p{period_name(period)}"]
        for item in block["checkpoints"]:
            lines.append(
                f"| {policy} | {period_name(period)} | {item['decision']} | "
                f"{_fmt(item['swapped_mean'])} | {_fmt(item['swapped_std_ddof0'])} | "
                f"{item['swapped_n']} | {_fmt(item['unswapped_mean'])} | "
                f"{_fmt(item['unswapped_std_ddof0'])} | {item['unswapped_n']} |"
            )
    lines.append("")
    lines.append("| policy | period | seed | checkpoint | room | bits | swapped |")
    lines.append("|---|---:|---:|---:|---|---:|---:|")
    for row in runs:
        for rec in row["checkpoints"]:
            swapped = set(rec["swapped"])
            for name in LEARN14:
                flag = 1 if name in swapped else 0
                lines.append(
                    f"| {row['policy']} | {period_name(row['period'])} | {row['seed']} | "
                    f"{rec['decision']} | {name} | {_fmt(rec['held'][name])} | {flag} |"
                )
    lines.append("")

    lines.append("## Decision shares")
    lines.append("")
    lines.append(
        "Mean over seeds. A stay counts for the room. A move counts for the corridor. "
        "A sleep decision counts for sleep. Shares sum to 1."
    )
    lines.append("")
    share_names = list(IDENTITIES) + ["corridor", "sleep"]
    lines.append("| policy | period | " + " | ".join(share_names) + " |")
    lines.append("|---|---:|" + "---:|" * len(share_names))
    for policy, period in _policy_keys():
        block = summary["by_policy"][f"{policy}/p{period_name(period)}"]
        cells = [_fmt(block["share_mean"][name], 4) for name in IDENTITIES]
        cells.append(_fmt(block["corridor_mean"], 4))
        cells.append(_fmt(block["sleep_mean"], 4))
        lines.append(
            f"| {policy} | {period_name(period)} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    lines.append("Per seed.")
    lines.append("")
    lines.append("| policy | period | seed | " + " | ".join(share_names) + " |")
    lines.append("|---|---:|---:|" + "---:|" * len(share_names))
    for row in runs:
        cells = [f"{row['share'][name]:.4f}" for name in IDENTITIES]
        cells.append(f"{row['corridor_share']:.4f}")
        cells.append(f"{row['sleep_share']:.4f}")
        lines.append(
            f"| {row['policy']} | {period_name(row['period'])} | {row['seed']} | "
            + " | ".join(cells) + " |"
        )
    lines.append("")

    lines.append("## Swap logs")
    lines.append("")
    lines.append(
        "One log per seed and period. The same log was checked against every policy, "
        "and against an independent draw from default_rng(world_seed + 80000) over the "
        "14 learnable rooms in identity order. k starts at 0. The new source is "
        "h20_world.World(world_seed + 2000 + k). Period inf has no swaps."
    )
    lines.append("")
    lines.append("| period | seed | decision | room | k |")
    lines.append("|---:|---:|---:|---|---:|")
    for period in PERIODS:
        for seed in seeds:
            row = _pick(result["runs"], "rand_dwell_s", period, seed)
            for event in row["swap_log"]:
                lines.append(
                    f"| {period} | {seed} | {event['decision']} | {event['room']} | {event['k']} |"
                )
    lines.append("")

    lines.append("## Ring order")
    lines.append("")
    lines.append("Room name in ring positions 0..17. One order per seed. Policies share it.")
    lines.append("")
    lines.append("| seed | " + " | ".join(f"r{i}" for i in range(N_RING)) + " |")
    lines.append("|---:|" + "---|" * N_RING)
    orders = {}
    for row in result["runs"]:
        orders[row["seed"]] = row["room_order"]
    for seed in seeds:
        lines.append(f"| {seed} | " + " | ".join(orders[seed]) + " |")
    lines.append("")

    lines.append("## W4 reference")
    lines.append("")
    ref = result["w4_reference"]
    lines.append(
        f"W4 final W14 at L = {L_SCORE}, one exam at the end of life, not T14. "
        f"Source: {ref['source']}."
    )
    lines.append("")
    lines.append("| policy | seed | W14 |")
    lines.append("|---|---:|---:|")
    for policy in ("rand_dwell_s", "oracle14_s"):
        for seed in seeds:
            hit = [row for row in ref["rows"] if row["policy"] == policy and row["seed"] == seed]
            if len(hit) != 1:
                raise RuntimeError(f"W4 reference missing {policy} seed {seed}")
            lines.append(f"| {policy} | {seed} | {_fmt(hit[0]['W14'])} |")
        lines.append(
            f"| {policy} | mean | {_fmt(ref[policy]['W14_mean'])} |"
        )
        lines.append(
            f"| {policy} | std ddof=0 | {_fmt(ref[policy]['W14_std_ddof0'])} |"
        )
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
        "- World5(world_seed, period) is World4(world_seed). Room names, ring order, noise "
        "rooms and Z are unchanged. period inf does not swap."
    )
    lines.append(
        "- After decision t, for every positive multiple t = period, 2·period, … that is "
        "at most L, one learnable room is drawn uniformly from the 14 in identity order "
        "with default_rng(world_seed + 80000). Its source becomes the same process in "
        "h20_world.World(world_seed + 2000 + k), from that world's beginning. k counts "
        "swaps from 0. Sleep and corridor decisions are on the same clock. The swap is "
        "logged as the decision, the room and k."
    )
    lines.append(
        "- A checkpoint at the same decision scores the world after that swap. The score "
        "is w4.evaluate_w14: a copy of the cortex is not stepped, and a copy of the world "
        "supplies EVAL=2048 symbols per learnable room. The live life then continues."
    )
    lines.append(
        "- The cortex, stores, coin, sleep dose and sleep placement are W4's, which are "
        "M7's bank with 19 channel rows and M7b's fixed sleep share 0.20. Policy rng is "
        "default_rng(1000 + world_seed). rand_dwell_s and oracle14_s move as in W4."
    )
    lines.append(
        "- oracle_fresh_s keeps S, the learnable rooms whose rule has changed since it "
        "last arrived there. S starts as all 14. The target is uniform over S in ring "
        "order, or over the 14 when S is empty, then the shortest path and 8 stays. "
        "The first stay of the visit removes that room from S. A tie on the ring goes left."
    )
    lines.append(
        "- World seeds 0, 1, 2, 3, 4 and 5. L = 2048. Periods 128 and 512 for every "
        "policy, plus rand_dwell_s at period inf. CPU. Private Kaggle kernels, one seed "
        "each, pool of 4, no GPU, no internet. Nothing retuned."
    )
    lines.append("")
    return "\n".join(lines)


def _load_kernel_payloads(out_dir):
    out_dir = Path(out_dir)
    found = {}
    for seed in SEEDS:
        hits = sorted(out_dir.glob(f"s{seed}/w5_s{seed}.json"))
        hits += sorted(out_dir.glob(f"s{seed}-p128/w5_s{seed}.json"))
        hits += sorted(out_dir.glob(f"s{seed}-p512/w5_s{seed}.json"))
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
                _pick(runs, policy, period, seed)["swap_log"]
                for policy in POLICIES
            ]
            if any(log != logs[0] for log in logs):
                raise RuntimeError(f"policies disagree on the swap log, seed {seed} period {period}")
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
        "w4_reference": w4_reference(),
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
    parser.add_argument("--out", default=str(ROOT / "results" / "w5"))
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
