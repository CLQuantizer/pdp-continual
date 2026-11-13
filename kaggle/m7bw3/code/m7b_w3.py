"""M7b-W3: per-channel curiosity on the bank, with one fixed sleep dose.

Imports m7_w3 and does not modify it. Numbers only. No verdict.
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
    if os.environ.get("_M7B_MAIN") != "1":
        os.environ["_M7B_MAIN"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import json
import math
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

import curio
import m7_w3 as m7
from w3_headroom import next_action

SEEDS = (36, 37, 38, 39, 40, 41)
CONTROLLERS = ("rand_dwell", "rand_dwell_s", "oracle7_s", "prog_total_s", "prog_chan_s")
SLEPT = ("rand_dwell_s", "oracle7_s", "prog_total_s", "prog_chan_s")
SCORED_L = 2048
SLEEP_SHARE = 0.20
KERNEL_WALL_LIMIT_S = 90 * 60
BITS_LIMIT = 1e-6
ROOT = Path(__file__).resolve().parent

if curio.EPS != 0.1 or curio.BETA != 0.3:
    raise RuntimeError(f"bandit constants drifted: EPS {curio.EPS} BETA {curio.BETA}")
if m7.DWELL != 8 or m7.SCORED_L != 2048:
    raise RuntimeError("M7 dwell or scored length drifted")


def sleep_decisions(L):
    """Nearest multiple of 8 to 20% of the life. Every sleep visit is 8 decisions."""
    L = int(L)
    if L % 4 != 0:
        raise ValueError(f"L={L} is not divisible by 4")
    dwell = m7.DWELL
    numer = 20 * L
    denom = 100 * dwell
    visits = (numer + denom // 2) // denom
    n = int(visits * dwell)
    if n <= 0 or n > L or n % dwell != 0:
        raise ValueError(f"sleep dose {n} does not fit L={L}")
    return n


def _threads():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _sum_selected(life, glances, channel_index):
    """Bits of these already-stored glances. Returns (channel sum, whole sum)."""
    if not glances:
        return 0.0, 0.0
    t0 = time.perf_counter()
    values = m7.score_glances(life.cortex.model, life.cortex.channel_emb, glances)
    life.score_s += time.perf_counter() - t0
    whole = float(values.sum())
    selected = 0.0
    for glance, value in zip(glances, values.tolist()):
        if int(glance.channel) == int(channel_index):
            selected += float(value)
    return selected, whole


def _learn_scored(life, symbols, observed, score_name, want_whole):
    """One waking glance. Progress is that room's exam bits saved, before any insert.

    The learn, coin flip and reservoir offer follow Life.room_decision / corridor_decision.
    want_whole also returns the whole-exam-store progress, for the smoke trace.
    """
    index = m7.ROOM_INDEX[score_name]
    glances = life.exam.flat() if want_whole else list(life.exam.slots[score_name])
    before_c, before_w = _sum_selected(life, glances, index)
    prev = life.cortex.prev_symbol
    if observed == "corridor":
        life._learn(symbols, m7.CORRIDOR, prev, commit=True)
        after_c, after_w = _sum_selected(life, glances, index)
        return float(before_c - after_c), (None if not want_whole else float(before_w - after_w))
    life._learn(symbols, m7.ROOM_INDEX[observed], prev, commit=True)
    after_c, after_w = _sum_selected(life, glances, index)
    glance = m7.Glance(life.next_gid, m7.ROOM_INDEX[observed], symbols, prev)
    life.next_gid += 1
    if int(life.coin_rng.integers(2)) == 0:
        life._insert_exam(glance)
    else:
        life.replay.offer(observed, glance)
    if life.exam.ids & life.replay.ids:
        raise RuntimeError("a glance is in both stores")
    return float(before_c - after_c), (None if not want_whole else float(before_w - after_w))


def _sleep_visit(life, seed, kind, channels, length, t, L, check_sleep, checked):
    """Eight (or fewer, at the end) M7 sleep decisions. Returns start, n, reward, learned, t, checked."""
    if length <= 0:
        raise RuntimeError("empty sleep visit")
    start = t
    # Waking inserts do not refresh the whole-store cache. One rescore makes the
    # first sleep_decision's before-score the exam store as it is.
    if life.score_progress:
        life._rescore()
    snap = life.begin_sleep_visit() if (check_sleep and not checked) else None
    total = 0.0
    n = 0
    learned = 0
    while n < length and t < L:
        progress, did_learn = life.sleep_decision()
        total += progress
        learned += int(did_learn)
        channels.append("sleep")
        n += 1
        t += 1
        if t % 256 == 0:
            print(
                f"s{seed} {kind} t={t}/{L} sleep exam {sum(life.exam.sizes().values())} "
                f"replay {sum(life.replay.sizes().values())}",
                flush=True,
            )
    if snap is not None:
        delta = life.end_sleep_visit(snap)
        checked = True
        if learned and delta <= 0.0:
            raise RuntimeError("sleep visit learned and left the parameters unchanged")
    reward = total / n if n else 0.0
    if not math.isfinite(reward):
        raise RuntimeError("nonfinite sleep reward")
    return start, n, reward, learned, t, checked


def _close_visit(life, pending):
    if pending is not None:
        life.snapshot(pending)
    return None


def run_fixed_sleep(seed, policy, L, score=True, check_sleep=False):
    """rand_dwell or oracle7, plus the fixed sleep dose at visit boundaries."""
    if policy not in ("rand_dwell", "oracle7"):
        raise ValueError(policy)
    n_sleep = sleep_decisions(L)
    plan = m7.sleep_plan(L, n_sleep)
    if any(length != m7.DWELL for _, length in plan):
        raise RuntimeError("a fixed sleep visit is not 8 decisions")
    t0 = time.perf_counter()
    life = m7.Life(seed, score_progress=False)
    rng = np.random.default_rng(1_000 + seed)
    state = {"kind": policy, "room": 0, "plan": []}
    channels = []
    pending = None
    si = 0
    t = 0
    checked = False
    kind = f"{policy}_s"
    while t < L:
        if not state["plan"] and m7.sleep_due(t, L, plan, si):
            pending = _close_visit(life, pending)
            _start, _n, _reward, _learned, t, checked = _sleep_visit(
                life, seed, kind, channels, plan[si][1], t, L, check_sleep, checked,
            )
            si += 1
            continue
        action = next_action(state, life.world.rooms, rng)
        room, channel, symbols = life.world.step(action, rng)
        state["room"] = room
        if channel == "corridor":
            life.corridor_decision(symbols)
        else:
            life.room_decision(symbols, channel)
            pending = channel
        channels.append(channel)
        t += 1
        if t % 256 == 0:
            print(f"s{seed} {kind} t={t}/{L}", flush=True)
        if not state["plan"]:
            pending = _close_visit(life, pending)
    if pending is not None:
        life.snapshot(pending)
    if len(channels) != L:
        raise RuntimeError(f"{kind} ran {len(channels)} decisions")
    if si != len(plan):
        raise RuntimeError(f"{kind} took {si} of {len(plan)} planned sleep visits")
    if life.exam_drawn != 0:
        raise RuntimeError(f"sleep drew {life.exam_drawn} exam glances")
    row = m7.finish_row(seed, kind, L, life, channels, t0, score, None, n_sleep)
    if row["sleep_decisions"] != n_sleep:
        raise RuntimeError(f"{kind} sleep decisions {row['sleep_decisions']} != dose {n_sleep}")
    if check_sleep and row["sleep_decisions"] > 0 and row["sleep_state_intact"] is not True:
        raise RuntimeError("sleep visit did not keep the waking stream")
    return row


def run_prog_total_s(seed, L, score=True, check_sleep=False):
    """M7 prog_total (whole-store reward, 9 rooms) plus the fixed sleep dose."""
    n_sleep = sleep_decisions(L)
    row = m7.run_bandit(
        seed, "fixed_sleep", L, score=score, n_sleep_target=n_sleep, check_sleep=check_sleep,
    )
    row["controller"] = "prog_total_s"
    if row["sleep_decisions"] != n_sleep:
        raise RuntimeError(f"prog_total_s sleep decisions {row['sleep_decisions']} != dose {n_sleep}")
    return row


def run_prog_chan(seed, L, score=True, check_sleep=False, trace_visits=False):
    """Nine-room bandit. The reward is the visited room's exam progress. Sleep is not an arm."""
    n_sleep = sleep_decisions(L)
    plan = m7.sleep_plan(L, n_sleep)
    if any(length != m7.DWELL for _, length in plan):
        raise RuntimeError("a fixed sleep visit is not 8 decisions")
    if L % 4 != 0:
        raise ValueError(f"L={L} is not divisible by 4")
    t0 = time.perf_counter()
    life = m7.Life(seed, score_progress=True)
    choice_rng = np.random.default_rng(10_000 + seed)
    world_rng = np.random.default_rng(1_000 + seed)
    values = [0.0] * 9
    counts = [0] * 9
    unvisited = set(range(9))
    channels = []
    reward_q = m7.empty_reward_quarters()
    visits = []
    si = 0
    t = 0
    checked = False
    kind = "prog_chan_s"
    while t < L:
        if m7.sleep_due(t, L, plan, si):
            start, n, reward, _learned, t, checked = _sleep_visit(
                life, seed, kind, channels, plan[si][1], t, L, check_sleep, checked,
            )
            m7.note_visit(reward_q, "sleep", start, L, reward)
            if trace_visits:
                visits.append({
                    "kind": "sleep",
                    "room": "sleep",
                    "decisions": int(n),
                    "per_channel_reward": None,
                    "whole_store_progress": float(reward),
                    "travel": 0,
                    "dwell": 0,
                })
            si += 1
            continue
        arm, _flag = m7.choose_arm(life.world, unvisited, False, values, 9, choice_rng)
        if arm == m7.SLEEP_ARM:
            raise RuntimeError("prog_chan_s has no sleep arm")
        target = int(arm)
        dest = life.world.rooms[target]
        start = t
        path = m7.shortest(life.world.room, target)
        chan_total = 0.0
        whole_total = 0.0
        travel = 0
        while path and t < L:
            action = path.pop(0)
            _room, channel, symbols = life.world.step(action, world_rng)
            if channel != "corridor":
                raise RuntimeError("a move did not observe the corridor")
            chan_p, whole_p = _learn_scored(life, symbols, "corridor", dest, trace_visits)
            chan_total += chan_p
            if trace_visits:
                whole_total += whole_p
            channels.append(channel)
            travel += 1
            t += 1
        dwell = 0
        dwell_name = None
        while dwell < m7.DWELL and t < L:
            _room, channel, symbols = life.world.step("stay", world_rng)
            if channel == "corridor":
                raise RuntimeError("a stay observed the corridor")
            chan_p, whole_p = _learn_scored(life, symbols, channel, dest, trace_visits)
            chan_total += chan_p
            if trace_visits:
                whole_total += whole_p
            channels.append(channel)
            dwell_name = channel
            dwell += 1
            t += 1
            if t % 256 == 0:
                print(
                    f"s{seed} {kind} t={t}/{L} exam {sum(life.exam.sizes().values())} "
                    f"replay {sum(life.replay.sizes().values())}",
                    flush=True,
                )
        if dwell and dwell_name != dest:
            raise RuntimeError("dwell channel is not the chosen room")
        if dwell:
            life.snapshot(dest)
        n_dec = travel + dwell
        if n_dec <= 0:
            raise RuntimeError("empty room visit")
        reward = chan_total / n_dec
        whole_reward = (whole_total / n_dec) if trace_visits else None
        if not math.isfinite(reward) or (whole_reward is not None and not math.isfinite(whole_reward)):
            raise RuntimeError("nonfinite room reward")
        m7.ema(values, counts, target, reward)
        m7.note_visit(reward_q, "room", start, L, reward)
        if trace_visits:
            visits.append({
                "kind": "room",
                "room": dest,
                "decisions": int(n_dec),
                "per_channel_reward": float(reward),
                "whole_store_progress": float(whole_reward),
                "travel": int(travel),
                "dwell": int(dwell),
            })
    if len(channels) != L:
        raise RuntimeError(f"prog_chan_s seed {seed} ran {len(channels)} decisions")
    if si != len(plan):
        raise RuntimeError(f"prog_chan_s took {si} of {len(plan)} planned sleep visits")
    if life.exam_drawn != 0:
        raise RuntimeError(f"sleep drew {life.exam_drawn} exam glances")
    row = m7.finish_row(seed, kind, L, life, channels, t0, score, reward_q, n_sleep)
    if row["sleep_decisions"] != n_sleep:
        raise RuntimeError(f"prog_chan_s sleep decisions {row['sleep_decisions']} != dose {n_sleep}")
    if check_sleep and row["sleep_decisions"] > 0 and row["sleep_state_intact"] is not True:
        raise RuntimeError("sleep visit did not keep the waking stream")
    if trace_visits:
        row["visits"] = visits
    return row


def run_controller(seed, controller, L, score=True, check_sleep=False, trace_visits=False):
    if controller == "rand_dwell":
        if trace_visits:
            raise ValueError("rand_dwell has no visit reward")
        return m7.run_controller(seed, "rand_dwell", L, score=score)
    if controller in ("rand_dwell_s", "oracle7_s"):
        policy = "rand_dwell" if controller == "rand_dwell_s" else "oracle7"
        return run_fixed_sleep(seed, policy, L, score=score, check_sleep=check_sleep)
    if controller == "prog_total_s":
        return run_prog_total_s(seed, L, score=score, check_sleep=check_sleep)
    if controller == "prog_chan_s":
        return run_prog_chan(
            seed, L, score=score, check_sleep=check_sleep, trace_visits=trace_visits,
        )
    raise ValueError(controller)


def _job(args):
    seed, controller, L = args
    _threads()
    dose = sleep_decisions(L) if controller in SLEPT else 0
    print(f"s{seed} {controller} L={L} start dose {dose}", flush=True)
    row = run_controller(seed, controller, L, score=True, check_sleep=True)
    row.pop("visits", None)
    print(
        f's{seed} {controller} L={L} W {row["W"]:.4f} W9 {row["W9"]:.4f} '
        f'sleep {row["sleep_share"]:.3f} corridor {row["corridor_share"]:.3f} '
        f'wall {row["wall_s"]:.1f}s learn {row["learn_s"]:.1f}s score {row["score_s"]:.1f}s',
        flush=True,
    )
    return row


def _time_channel_rescore(model, channel_emb):
    from h20_world import ALPHA
    rng = np.random.default_rng(2)
    glances = []
    for i in range(m7.R):
        glances.append(m7.Glance(
            i, 0,
            rng.integers(0, len(ALPHA), size=m7.CHUNK, dtype=np.int64),
            int(rng.integers(0, len(ALPHA))),
        ))
    m7.score_glances(model, channel_emb, glances)
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        m7.score_glances(model, channel_emb, glances)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def kernel_estimate(learn_per, full_s, channel_s):
    """Four workers, five jobs. The two scored controllers start in the first wave.

    prog_total_s rescores the whole exam store on every decision. prog_chan_s
    rescores one channel twice on a waking decision and the whole store on a
    sleep decision, plus one extra whole rescore at the start of each sleep visit.
    Stores are charged at capacity, so this is above the run it estimates.
    """
    L = SCORED_L
    n_sleep = sleep_decisions(L)
    frac = n_sleep / L
    visits = n_sleep / m7.DWELL
    total = L * (full_s + learn_per)
    chan = (
        (L - n_sleep) * (2.0 * channel_s + learn_per)
        + n_sleep * (full_s + learn_per)
        + visits * full_s
    )
    slept = (L - n_sleep) * learn_per + n_sleep * (full_s + learn_per)
    plain = L * learn_per
    wave = (total, chan, slept, slept)
    fifth = plain
    kernel = max(max(wave), min(wave) + fifth)
    return {
        "prog_total_s_s": float(total),
        "prog_chan_s_s": float(chan),
        "rand_dwell_s_s": float(slept),
        "oracle7_s_s": float(slept),
        "rand_dwell_s_queued_s": float(fifth),
        "kernel_s": float(kernel),
        "sleep_decisions": int(n_sleep),
        "sleep_share": float(frac),
    }


def _within(share, target, tol):
    return abs(float(share) - float(target)) <= float(tol)


def smoke(out_dir=None):
    _threads()
    out_dir = Path(out_dir or (ROOT / "results" / "m7bw3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    t0 = time.perf_counter()
    payload = {
        "ok": False,
        "reason": None,
        "torch": torch.__version__,
        "threads": int(torch.get_num_threads()),
        "model_seed": m7.MODEL_SEED,
        "kernels_pushed": False,
        "sleep_share_target": SLEEP_SHARE,
    }
    log(
        f"torch={torch.__version__} threads={torch.get_num_threads()} "
        f"model_seed={m7.MODEL_SEED}"
    )
    try:
        dose_2048 = sleep_decisions(SCORED_L)
        dose_512 = sleep_decisions(512)
        dose_256 = sleep_decisions(256)
        dose_64 = sleep_decisions(64)
        log(
            f"dose L2048 {dose_2048}/{SCORED_L} share {dose_2048 / SCORED_L:.6f}; "
            f"L512 {dose_512}/512 share {dose_512 / 512:.6f}; "
            f"L256 {dose_256}/256 share {dose_256 / 256:.6f}; "
            f"L64 {dose_64}/64 share {dose_64 / 64:.6f}"
        )
        for L, dose in ((SCORED_L, dose_2048), (512, dose_512), (256, dose_256), (64, dose_64)):
            plan = m7.sleep_plan(L, dose)
            if any(length != m7.DWELL for _, length in plan):
                raise RuntimeError(f"L={L} sleep plan has a visit that is not {m7.DWELL}")
            if sum(length for _, length in plan) != dose:
                raise RuntimeError(f"L={L} sleep plan does not sum to the dose")

        log("smoke check 1: rand_dwell seed 30 L=64 versus m7_w3.run_controller")
        ref = m7.run_controller(30, "rand_dwell", 64, score=True)
        got = run_controller(30, "rand_dwell", 64, score=True)
        delta_w = abs(got["W"] - ref["W"])
        delta_w9 = abs(got["W9"] - ref["W9"])
        ok = delta_w <= BITS_LIMIT and delta_w9 <= BITS_LIMIT
        log(
            f"  m7 W {ref['W']:.8f} W9 {ref['W9']:.8f} m7b W {got['W']:.8f} W9 {got['W9']:.8f} "
            f"abs_dW {delta_w:.3e} abs_dW9 {delta_w9:.3e} limit {BITS_LIMIT:.1e} ok {ok}"
        )
        log(
            f"  published M7 smoke W 9.922327 W9 9.474917; "
            f"live m7 minus published W {ref['W'] - 9.922327:.3e}"
        )
        if not ok:
            raise RuntimeError("rand_dwell does not reproduce M7 at seed 30 L=64")
        if got["sleep_decisions"] != 0 or ref["sleep_decisions"] != 0:
            raise RuntimeError("rand_dwell slept")

        log("smoke check 2: rand_dwell_s seed 36 L=512 sleep share")
        slept = run_controller(36, "rand_dwell_s", 512, score=False, check_sleep=True)
        share = slept["sleep_share"]
        log(
            f"  sleep_decisions {slept['sleep_decisions']} target {dose_512} "
            f"share {share:.6f} target {SLEEP_SHARE:.2f} tol 0.02 "
            f"intact {slept['sleep_state_intact']} wall {slept['wall_s']:.2f}s"
        )
        if slept["sleep_decisions"] != dose_512 or not _within(share, SLEEP_SHARE, 0.02):
            raise RuntimeError(f"rand_dwell_s sleep share {share:.6f} is outside 0.20 ± 0.02")
        if slept["sleep_state_intact"] is not True:
            raise RuntimeError("rand_dwell_s sleep did not keep the waking stream")

        log("smoke check 3: prog_chan_s seed 36 L=256 visit trace")
        traced = run_controller(36, "prog_chan_s", 256, score=False, check_sleep=True, trace_visits=True)
        log(
            f"  visits {len(traced['visits'])} sleep_decisions {traced['sleep_decisions']} "
            f"share {traced['sleep_share']:.6f} intact {traced['sleep_state_intact']} "
            f"wall {traced['wall_s']:.2f}s"
        )
        if traced["sleep_decisions"] != dose_256 or not _within(traced["sleep_share"], SLEEP_SHARE, 0.02):
            raise RuntimeError(
                f"prog_chan_s sleep share {traced['sleep_share']:.6f} is outside 0.20 ± 0.02"
            )
        n_room = 0
        for i, visit in enumerate(traced["visits"]):
            if visit["kind"] == "room":
                n_room += 1
                log(
                    f"  visit {i} room {visit['room']} decisions {visit['decisions']} "
                    f"travel {visit['travel']} dwell {visit['dwell']} "
                    f"per_channel_reward {visit['per_channel_reward']:.6f} "
                    f"whole_store_progress {visit['whole_store_progress']:.6f}"
                )
            else:
                log(
                    f"  visit {i} room sleep decisions {visit['decisions']} "
                    f"per_channel_reward none "
                    f"whole_store_progress {visit['whole_store_progress']:.6f}"
                )
        if n_room <= 0:
            raise RuntimeError("prog_chan_s trace has no room visit")
        traced.pop("visits", None)

        log("smoke check 4: five controllers, seed 36, L=64, W and W9 finite")
        timed = []
        for controller in CONTROLLERS:
            row = run_controller(36, controller, 64, score=True, check_sleep=False)
            log(
                f"  {controller} W {row['W']:.6f} W9 {row['W9']:.6f} finite {row['finite']} "
                f"sleep {row['sleep_share']:.4f} decisions {row['sleep_decisions']} "
                f"corridor {row['corridor_share']:.4f} wall {row['wall_s']:.2f}s "
                f"learn {row['learn_s']:.2f}s score {row['score_s']:.2f}s"
            )
            if not row["finite"]:
                raise RuntimeError(f"{controller} W is not finite")
            expect = 0 if controller == "rand_dwell" else dose_64
            if row["sleep_decisions"] != expect:
                raise RuntimeError(f"{controller} sleep decisions {row['sleep_decisions']} != {expect}")
            timed.append(row)
        learn_per = max(row["learn_s"] / 64.0 for row in timed)
        probe = m7.BankCortex(model_seed=m7.MODEL_SEED, freeze_channel=False)
        full_s = m7.time_full_rescore(probe.model, probe.channel_emb)
        channel_s = _time_channel_rescore(probe.model, probe.channel_emb)
        est = kernel_estimate(learn_per, full_s, channel_s)
        log(
            f"  learn_per_decision {learn_per:.4f}s full_rescore_288 {full_s:.4f}s "
            f"channel_rescore_32 {channel_s:.4f}s"
        )
        log(
            f"  pessimistic prog_total_s {est['prog_total_s_s']:.1f}s "
            f"prog_chan_s {est['prog_chan_s_s']:.1f}s "
            f"rand_dwell_s {est['rand_dwell_s_s']:.1f}s "
            f"queued rand_dwell {est['rand_dwell_s_queued_s']:.1f}s "
            f"kernel {est['kernel_s']:.1f}s ({est['kernel_s'] / 60:.2f} min) "
            f"limit {KERNEL_WALL_LIMIT_S:.0f}s"
        )
        log(
            "  estimate runs prog_total_s and prog_chan_s in a pool of 4 and charges a full exam "
            "store on every whole-store rescore; rand_dwell starts when the first wave frees a worker"
        )
        payload.update(estimate=est, learn_per_decision_s=learn_per, full_rescore_s=full_s, channel_rescore_s=channel_s)
        if est["kernel_s"] > KERNEL_WALL_LIMIT_S:
            raise RuntimeError(
                f"kernel estimate {est['kernel_s']:.0f}s exceeds {KERNEL_WALL_LIMIT_S:.0f}s"
            )
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


def kernel_main(seed, out, workers=4, kernel_id=None):
    if __name__ != "m7b_w3":
        raise SystemExit("import m7b_w3 and call kernel_main so the pool can spawn workers")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 36-41")
    if int(workers) < 1 or int(workers) > 4:
        raise SystemExit("kernel pool is at most 4")
    _threads()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    order = ("prog_total_s", "prog_chan_s", "rand_dwell_s", "oracle7_s", "rand_dwell")
    jobs = [(seed, controller, SCORED_L) for controller in order]
    print(
        f"kernel {kernel_id} seed {seed} workers {workers} jobs {len(jobs)} "
        f"dose {sleep_decisions(SCORED_L)}",
        flush=True,
    )
    ctx = get_context("spawn")
    with ctx.Pool(int(workers)) as pool:
        asyncs = [pool.apply_async(_job, (job,)) for job in jobs]
        rows = [item.get() for item in asyncs]
    wall = time.perf_counter() - t0
    if len(rows) != len(CONTROLLERS):
        raise SystemExit(f"expected {len(CONTROLLERS)} rows, got {len(rows)}")
    dose = sleep_decisions(SCORED_L)
    by_name = {}
    for row in rows:
        if row["seed"] != seed or row["L"] != SCORED_L:
            raise SystemExit("row seed or L mismatch")
        if not row["finite"]:
            raise SystemExit(f'{row["controller"]} is not finite')
        if row["overlap"] != 0 or row["exam_glances_drawn"] != 0:
            raise SystemExit(f'{row["controller"]} store check failed')
        expect = 0 if row["controller"] == "rand_dwell" else dose
        if row["sleep_decisions"] != expect:
            raise SystemExit(f'{row["controller"]} sleep dose mismatch')
        by_name[row["controller"]] = row
    if set(by_name) != set(CONTROLLERS):
        raise SystemExit(f"controllers {sorted(by_name)} != {CONTROLLERS}")
    rows = [by_name[name] for name in CONTROLLERS]
    payload = {
        "kernel_id": kernel_id,
        "seed": seed,
        "workers": int(workers),
        "wall_s": wall,
        "torch": torch.__version__,
        "model_seed": m7.MODEL_SEED,
        "sleep_decisions": dose,
        "sleep_share_target": SLEEP_SHARE,
        "runs": rows,
    }
    path = out / f"m7b_s{seed}.json"
    m7.dump(path, payload)
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    return payload


def _fmt(value, digits=6):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _pm(mu, sd, digits=6):
    if mu is None:
        return "—"
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"


def _pick(runs, controller, seed):
    hit = [row for row in runs if row["controller"] == controller and row["seed"] == seed]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {controller} seed {seed}, got {len(hit)}")
    return hit[0]


def _m7_reference():
    path = ROOT / "results" / "m7w3" / "results.json"
    payload = json.loads(path.read_text())
    summary = payload["summary"]
    out = {"seeds": list(payload["seeds"]), "L": payload["L"], "by_controller": {}, "gap_summary": summary["gap_summary"]}
    for controller, block in summary["by_controller"].items():
        out["by_controller"][controller] = {
            "W_mean": block["W_mean"],
            "W_std_ddof0": block["W_std_ddof0"],
            "W9_mean": block["W9_mean"],
            "W9_std_ddof0": block["W9_std_ddof0"],
            "sleep_mean": block["sleep_mean"],
            "sleep_std_ddof0": block["sleep_std_ddof0"],
            "forgetting_mean": block["forgetting_mean"],
            "forgetting_std_ddof0": block["forgetting_std_ddof0"],
            "forgetting_n": block["forgetting_n"],
        }
    per = {controller: [] for controller in payload["controllers"]}
    for row in payload["runs"]:
        if row["L"] != 2048:
            continue
        per[row["controller"]].append({
            "seed": row["seed"],
            "W": row["W"],
            "W9": row["W9"],
            "sleep_share": row["sleep_share"],
        })
    for controller in per:
        per[controller].sort(key=lambda item: item["seed"])
    out["per_seed"] = per
    return out


def summarize(runs):
    by = {}
    for controller in CONTROLLERS:
        rows = sorted((row for row in runs if row["controller"] == controller), key=lambda row: row["seed"])
        if len(rows) != len(SEEDS):
            raise RuntimeError(f"{controller}: {len(rows)} rows, expected {len(SEEDS)}")
        w_mu, w_sd = m7.mean_std(row["W"] for row in rows)
        w9_mu, w9_sd = m7.mean_std(row["W9"] for row in rows)
        held_mu = {name: m7.mean_std(row["held"][name] for row in rows)[0] for name in curio.LEARN}
        held9_mu = {name: m7.mean_std(row["held9"][name] for row in rows)[0] for name in curio.CH}
        share_mu = {name: m7.mean_std(row["share"][name] for row in rows)[0] for name in curio.CH}
        corr_mu = m7.mean_std(row["corridor_share"] for row in rows)[0]
        sleep_mu, sleep_sd = m7.mean_std(row["sleep_share"] for row in rows)
        quarter = []
        for i in range(4):
            quarter.append({
                "share": {name: m7.mean_std(row["quarter"][i]["share"][name] for row in rows)[0] for name in curio.CH},
                "corridor": m7.mean_std(row["quarter"][i]["corridor"] for row in rows)[0],
                "sleep": m7.mean_std(row["quarter"][i]["sleep"] for row in rows)[0],
            })
        forgetting = {}
        for name in curio.LEARN:
            deltas = [
                row["forgetting"][name]["delta_bps"]
                for row in rows
                if row["forgetting"][name]["delta_bps"] is not None
            ]
            mu, sd = m7.mean_std(deltas)
            forgetting[name] = {
                "n": len(deltas),
                "delta_bps_mean": mu,
                "delta_bps_std_ddof0": sd,
                "after_bps_mean": m7.mean_std(
                    row["forgetting"][name]["after_bps"]
                    for row in rows
                    if row["forgetting"][name]["after_bps"] is not None
                )[0],
                "end_bps_mean": m7.mean_std(
                    row["forgetting"][name]["end_bps"]
                    for row in rows
                    if row["forgetting"][name]["end_bps"] is not None
                )[0],
            }
        defined = [
            row["forgetting"][name]["delta_bps"]
            for row in rows
            for name in curio.LEARN
            if row["forgetting"][name]["delta_bps"] is not None
        ]
        mu, sd = m7.mean_std(defined)
        by[controller] = {
            "n": len(rows),
            "W_mean": w_mu,
            "W_std_ddof0": w_sd,
            "W9_mean": w9_mu,
            "W9_std_ddof0": w9_sd,
            "held_mean": held_mu,
            "held9_mean": held9_mu,
            "share_mean": share_mu,
            "corridor_mean": corr_mu,
            "sleep_mean": sleep_mu,
            "sleep_std_ddof0": sleep_sd,
            "quarter_mean": quarter,
            "forgetting_channel": forgetting,
            "forgetting_mean": mu,
            "forgetting_std_ddof0": sd,
            "forgetting_n": len(defined),
        }
    gaps = []
    for seed in SEEDS:
        chan = _pick(runs, "prog_chan_s", seed)
        rand_s = _pick(runs, "rand_dwell_s", seed)
        total = _pick(runs, "prog_total_s", seed)
        oracle = _pick(runs, "oracle7_s", seed)
        rand = _pick(runs, "rand_dwell", seed)
        gaps.append({
            "seed": seed,
            "prog_chan_s_minus_rand_dwell_s": chan["W"] - rand_s["W"],
            "prog_chan_s_minus_prog_total_s": chan["W"] - total["W"],
            "prog_chan_s_minus_oracle7_s": chan["W"] - oracle["W"],
            "rand_dwell_s_minus_rand_dwell": rand_s["W"] - rand["W"],
        })
    gap_summary = {}
    for key in (
        "prog_chan_s_minus_rand_dwell_s",
        "prog_chan_s_minus_prog_total_s",
        "prog_chan_s_minus_oracle7_s",
        "rand_dwell_s_minus_rand_dwell",
    ):
        values = [row[key] for row in gaps]
        mu, sd = m7.mean_std(values)
        gap_summary[key] = {
            "mean": mu,
            "std_ddof0": sd,
            "left_lower": int(sum(value < 0.0 for value in values)),
            "n": len(values),
        }
    rooms = {}
    for seed in SEEDS:
        order = _pick(runs, "rand_dwell", seed)["room_channel"]
        for controller in CONTROLLERS:
            if _pick(runs, controller, seed)["room_channel"] != order:
                raise RuntimeError(f"seed {seed} ring order differs across controllers")
        rooms[seed] = order
    return {"by_controller": by, "gaps": gaps, "gap_summary": gap_summary, "rooms": rooms}


def _header(seeds):
    return "| controller | " + " | ".join(f"seed {seed}" for seed in seeds) + " | mean ± std ddof=0 |"


def _sep(n):
    return "|---|" + "---:|" * n + "---:|"


def render_markdown(result):
    seeds = result["seeds"]
    runs = result["runs"]
    summary = result["summary"]
    dose = sleep_decisions(SCORED_L)
    lines = [
        "# M7b-W3 per-channel curiosity, fixed sleep",
        "",
        "The W3 loop on the trace-learned roots bank. W is bits/symbol of the frozen bank on a fresh "
        "continuation of the 7 learnable channels, from a zero state, with that channel's embedding. "
        "W9 is that mean over all 9 room channels. Lower is better. Forgetting is end-of-life exam "
        "bits/symbol minus the exam bits/symbol just after the last visit to that channel. "
        "Positive forgetting is a worse code. Numbers only.",
        "",
        f"Sleep dose for every controller whose name ends in `_s`: {dose} decisions, "
        f"share {dose / SCORED_L:.6f}, in visits of {m7.DWELL}. "
        f"`rand_dwell` does not sleep. `prog_chan_s` room reward is the sum, over that room's exam "
        "glances, of bits before minus bits after, summed over the visit's decisions and divided by "
        "those decisions. Travel counts. Sleep visits are not bandit updates.",
        "",
        "## W",
        "",
        _header(seeds),
        _sep(len(seeds)),
    ]
    for controller in CONTROLLERS:
        cells = [_fmt(_pick(runs, controller, seed)["W"]) for seed in seeds]
        block = summary["by_controller"][controller]
        lines.append(
            f"| {controller} | " + " | ".join(cells) + f" | {_pm(block['W_mean'], block['W_std_ddof0'])} |"
        )
    lines += ["", "## W9", "", _header(seeds), _sep(len(seeds))]
    for controller in CONTROLLERS:
        cells = [_fmt(_pick(runs, controller, seed)["W9"]) for seed in seeds]
        block = summary["by_controller"][controller]
        lines.append(
            f"| {controller} | " + " | ".join(cells) + f" | {_pm(block['W9_mean'], block['W9_std_ddof0'])} |"
        )
    lines += [
        "",
        "## Gaps",
        "",
        "On W. A negative gap means the first controller is lower.",
        "",
        "| seed | prog_chan_s − rand_dwell_s | prog_chan_s − prog_total_s | prog_chan_s − oracle7_s | rand_dwell_s − rand_dwell |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["gaps"]:
        lines.append(
            f"| {row['seed']} | {_fmt(row['prog_chan_s_minus_rand_dwell_s'])} | "
            f"{_fmt(row['prog_chan_s_minus_prog_total_s'])} | "
            f"{_fmt(row['prog_chan_s_minus_oracle7_s'])} | "
            f"{_fmt(row['rand_dwell_s_minus_rand_dwell'])} |"
        )
    labels = (
        "prog_chan_s_minus_rand_dwell_s",
        "prog_chan_s_minus_prog_total_s",
        "prog_chan_s_minus_oracle7_s",
        "rand_dwell_s_minus_rand_dwell",
    )
    lines.append("| mean | " + " | ".join(_fmt(summary["gap_summary"][key]["mean"]) for key in labels) + " |")
    lines.append(
        "| std ddof=0 | " + " | ".join(_fmt(summary["gap_summary"][key]["std_ddof0"]) for key in labels) + " |"
    )
    lines.append(
        "| left lower (count) | "
        + " | ".join(
            f"{summary['gap_summary'][key]['left_lower']} of {summary['gap_summary'][key]['n']}"
            for key in labels
        )
        + " |"
    )
    lines += [
        "",
        "## Forgetting",
        "",
        "Exam-store bits/symbol just after the last visit that dwelled in the channel, minus nothing: "
        "the difference is end minus that snapshot. A channel with no exam glance at that visit is left out "
        "of the mean. n is the number of seed-channel pairs in the mean.",
        "",
        "| controller | mean delta bps | std ddof=0 | n |",
        "|---|---:|---:|---:|",
    ]
    for controller in CONTROLLERS:
        block = summary["by_controller"][controller]
        lines.append(
            f"| {controller} | {_fmt(block['forgetting_mean'])} | {_fmt(block['forgetting_std_ddof0'])} | "
            f"{block['forgetting_n']} |"
        )
    lines += [
        "",
        "| controller | channel | n | after bps | end bps | delta bps | std ddof=0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for controller in CONTROLLERS:
        forget = summary["by_controller"][controller]["forgetting_channel"]
        for name in curio.LEARN:
            item = forget[name]
            lines.append(
                f"| {controller} | {name} | {item['n']} | {_fmt(item['after_bps_mean'])} | "
                f"{_fmt(item['end_bps_mean'])} | {_fmt(item['delta_bps_mean'])} | "
                f"{_fmt(item['delta_bps_std_ddof0'])} |"
            )
    lines += [
        "",
        "## Sleep share",
        "",
        "Sleep decisions divided by L.",
        "",
        _header(seeds),
        _sep(len(seeds)),
    ]
    for controller in CONTROLLERS:
        cells = [_fmt(_pick(runs, controller, seed)["sleep_share"]) for seed in seeds]
        block = summary["by_controller"][controller]
        lines.append(
            f"| {controller} | " + " | ".join(cells)
            + f" | {_pm(block['sleep_mean'], block['sleep_std_ddof0'])} |"
        )
    lines += [
        "",
        "## Decision shares",
        "",
        "Mean over seeds. A stay counts for the room channel. A move counts for the corridor. "
        "A sleep decision counts for sleep. Shares sum to 1.",
        "",
        "| controller | " + " | ".join(curio.CH) + " | corridor | sleep |",
        "|---|" + "---:|" * (len(curio.CH) + 2),
    ]
    for controller in CONTROLLERS:
        block = summary["by_controller"][controller]
        cells = [_fmt(block["share_mean"][name], 4) for name in curio.CH]
        lines.append(
            f"| {controller} | " + " | ".join(cells)
            + f" | {_fmt(block['corridor_mean'], 4)} | {_fmt(block['sleep_mean'], 4)} |"
        )
    chan_rows = [_pick(runs, "prog_chan_s", seed) for seed in seeds]
    lines += [
        "",
        "## prog_chan_s shares by quarter",
        "",
        "Quarter i is decisions [(i−1)L/4, iL/4). A stay counts for the room channel. "
        "A move counts for the corridor. A sleep decision counts for sleep.",
        "",
    ]
    head = "| seed | quarter | " + " | ".join(curio.CH) + " | corridor | sleep |"
    lines += [head, "|---:|---:|" + "---:|" * (len(curio.CH) + 2)]
    for row in chan_rows:
        for i, block in enumerate(row["quarter"]):
            cells = [_fmt(block["share"][name], 4) for name in curio.CH]
            lines.append(
                f"| {row['seed']} | {i + 1} | " + " | ".join(cells)
                + f" | {_fmt(block['corridor'], 4)} | {_fmt(block['sleep'], 4)} |"
            )
    for i in range(4):
        block = summary["by_controller"]["prog_chan_s"]["quarter_mean"][i]
        cells = [_fmt(block["share"][name], 4) for name in curio.CH]
        lines.append(
            f"| mean | {i + 1} | " + " | ".join(cells)
            + f" | {_fmt(block['corridor'], 4)} | {_fmt(block['sleep'], 4)} |"
        )
    lines += [
        "",
        "## prog_chan_s per-channel reward by quarter",
        "",
        "Room reward is per-channel exam bits saved per decision, mean over the room visits whose "
        "first decision falls in the quarter. Sleep reward is whole-store exam bits saved per decision "
        "on sleep visits. Sleep visits do not update the bandit. A quarter with no visits of that kind "
        "is blank. The mean row's visit count is the sum across seeds. The mean reward is the mean of "
        "the per-seed means, over seeds that have at least one such visit.",
        "",
        "| seed | quarter | sleep visits | sleep mean reward | room visits | room mean per-channel reward |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in chan_rows:
        for i, slot in enumerate(row["reward_quarter"]):
            lines.append(
                f"| {row['seed']} | {i + 1} | {slot['sleep_n']} | {_fmt(slot['sleep_mean'])} | "
                f"{slot['room_n']} | {_fmt(slot['room_mean'])} |"
            )
    for i in range(4):
        sleep_means = [
            row["reward_quarter"][i]["sleep_mean"]
            for row in chan_rows
            if row["reward_quarter"][i]["sleep_n"]
        ]
        room_means = [
            row["reward_quarter"][i]["room_mean"]
            for row in chan_rows
            if row["reward_quarter"][i]["room_n"]
        ]
        sleep_n = sum(row["reward_quarter"][i]["sleep_n"] for row in chan_rows)
        room_n = sum(row["reward_quarter"][i]["room_n"] for row in chan_rows)
        lines.append(
            f"| mean | {i + 1} | {sleep_n} | {_fmt(m7.mean_std(sleep_means)[0])} | {room_n} | "
            f"{_fmt(m7.mean_std(room_means)[0])} |"
        )
    lines += [
        "",
        "## Mean held-out bits",
        "",
        "Mean over seeds. W is the mean of the 7 learnable channels. W9 is the mean of all 9.",
        "",
        "| controller | " + " | ".join(curio.CH) + " | W | W9 |",
        "|---|" + "---:|" * (len(curio.CH) + 2),
    ]
    for controller in CONTROLLERS:
        block = summary["by_controller"][controller]
        cells = [_fmt(block["held9_mean"][name]) for name in curio.CH]
        lines.append(
            f"| {controller} | " + " | ".join(cells)
            + f" | {_fmt(block['W_mean'])} | {_fmt(block['W9_mean'])} |"
        )
    lines += ["", "## Per-seed held-out bits", ""]
    for controller in CONTROLLERS:
        lines += [
            f"### {controller}",
            "",
            "| seed | " + " | ".join(curio.CH) + " |",
            "|---:|" + "---:|" * len(curio.CH),
        ]
        for seed in seeds:
            row = _pick(runs, controller, seed)
            cells = [_fmt(row["held9"][name]) for name in curio.CH]
            lines.append(f"| {seed} | " + " | ".join(cells) + " |")
        lines.append("")
    ref = result["m7"]
    lines += [
        "## M7 reference",
        "",
        "M7-W3, seeds 30–35, L = 2048, same cortex. `prog_sleep` chose its sleep. "
        "`fixed_sleep` used that seed's prog_sleep count. This run's dose is 0.20 for every `_s` controller.",
        "",
        "### M7 W",
        "",
        _header(ref["seeds"]),
        _sep(len(ref["seeds"])),
    ]
    for controller in ref["per_seed"]:
        cells = [_fmt(item["W"]) for item in ref["per_seed"][controller]]
        block = ref["by_controller"][controller]
        lines.append(
            f"| {controller} | " + " | ".join(cells) + f" | {_pm(block['W_mean'], block['W_std_ddof0'])} |"
        )
    lines += ["", "### M7 W9", "", _header(ref["seeds"]), _sep(len(ref["seeds"]))]
    for controller in ref["per_seed"]:
        cells = [_fmt(item["W9"]) for item in ref["per_seed"][controller]]
        block = ref["by_controller"][controller]
        lines.append(
            f"| {controller} | " + " | ".join(cells) + f" | {_pm(block['W9_mean'], block['W9_std_ddof0'])} |"
        )
    lines += [
        "",
        "### M7 gaps, forgetting, sleep share",
        "",
        "| gap | mean | std ddof=0 | left lower |",
        "|---|---:|---:|---:|",
    ]
    for key, block in ref["gap_summary"].items():
        lines.append(
            f"| {key} | {_fmt(block['mean'])} | {_fmt(block['std_ddof0'])} | "
            f"{block['left_lower']} of {block['n']} |"
        )
    lines += [
        "",
        "| controller | forgetting mean | forgetting std | n | sleep share |",
        "|---|---:|---:|---:|---:|",
    ]
    for controller, block in ref["by_controller"].items():
        lines.append(
            f"| {controller} | {_fmt(block['forgetting_mean'])} | {_fmt(block['forgetting_std_ddof0'])} | "
            f"{block['forgetting_n']} | {_pm(block['sleep_mean'], block['sleep_std_ddof0'])} |"
        )
    lines += [
        "",
        "## Ring order",
        "",
        "Channel in rooms 0..8. One order per seed; controllers share it.",
        "",
        "| seed | " + " | ".join(f"r{i}" for i in range(9)) + " |",
        "|---:|" + "---|" * 9,
    ]
    for seed in seeds:
        lines.append(f"| {seed} | " + " | ".join(summary["rooms"][seed]) + " |")
    lines += ["", "## Walls", "", "| kernel | seed | wall seconds |", "|---|---:|---:|"]
    for item in result["kernels"]:
        lines.append(f"| {item['kernel_id']} | {item['seed']} | {_fmt(item['wall_s'], 3)} |")
    lines += [
        "",
        "| controller | seed | wall seconds | learn seconds | score seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for controller in CONTROLLERS:
        for seed in seeds:
            row = _pick(runs, controller, seed)
            lines.append(
                f"| {controller} | {seed} | {_fmt(row['wall_s'], 3)} | {_fmt(row['learn_s'], 3)} | "
                f"{_fmt(row['score_s'], 3)} |"
            )
    lines += ["", "## Smoke test", "", "```"]
    lines.extend(result["smoke_log"])
    lines += ["```", "", "## Protocol", ""]
    lines.append(
        "- `BankCortex`, the stores, the sleep decision, W, W9 and forgetting are `m7_w3`'s. "
        "Model seed 0. Glance length 64. Reservoirs hold 32 glances per channel. "
        "Exam coin `default_rng(40_000 + seed)`, exam replacement `default_rng(20_000 + seed)`, "
        "replay replacement `default_rng(50_000 + seed)`, sleep draw `default_rng(30_000 + seed)`."
    )
    lines.append(
        "- `rand_dwell` is `m7_w3.run_controller(seed, 'rand_dwell', L)`. "
        "`rand_dwell_s` and `oracle7_s` use `w3_headroom.next_action` and `default_rng(1_000 + seed)`. "
        "A sleep visit starts only when the current plan is empty, which is the end of a visit."
    )
    lines.append(
        "- `prog_total_s` is `m7_w3.run_bandit` kind `fixed_sleep` with this dose, then renamed. "
        "Its room reward is the whole exam store. `prog_chan_s` uses the same 9-room bandit "
        "(`default_rng(10_000 + seed)`, optimistic first pass, EPS 0.1, BETA 0.3) and the same dose. "
        "Its room reward sums bits saved on the destination room's exam glances over the visit, "
        "including corridor steps, and divides by the visit's decisions. Sleep does not call the EMA."
    )
    lines.append(
        f"- The dose is {m7.DWELL} times the integer nearest to 0.20 L / {m7.DWELL}. "
        f"At L = {SCORED_L} that is {dose} decisions, share {dose / SCORED_L:.6f}. "
        "Placement is `m7_w3.sleep_plan` and `m7_w3.sleep_due`."
    )
    lines.append(
        "- A forgetting snapshot is taken at the end of a dwell visit, before a sleep that follows it. "
        "World seeds "
        + ", ".join(str(seed) for seed in seeds)
        + f". L = {SCORED_L}. CPU. Private Kaggle kernels, one seed each, pool of 4, no GPU, no internet. "
        "Nothing retuned."
    )
    lines.append("")
    return "\n".join(lines)


def assemble(out_dir, smoke_payload):
    out_dir = Path(out_dir)
    runs = []
    kernels = []
    for seed in SEEDS:
        hits = sorted(out_dir.glob(f"s{seed}/m7b_s{seed}.json")) + sorted(out_dir.glob(f"m7b_s{seed}.json"))
        if len(hits) != 1:
            raise RuntimeError(f"expected one results file for seed {seed}, found {hits}")
        payload = json.loads(hits[0].read_text())
        if payload["seed"] != seed:
            raise RuntimeError(f"payload seed {payload['seed']} != {seed}")
        runs.extend(payload["runs"])
        kernels.append({
            "kernel_id": payload["kernel_id"],
            "seed": seed,
            "wall_s": payload["wall_s"],
            "workers": payload["workers"],
        })
    if len(runs) != len(SEEDS) * len(CONTROLLERS):
        raise RuntimeError(f"expected {len(SEEDS) * len(CONTROLLERS)} runs, got {len(runs)}")
    for row in runs:
        row.pop("visits", None)
    summary = summarize(runs)
    # JSON object keys are strings. summarize stored int keys; render uses ints.
    rooms_json = {str(seed): rooms for seed, rooms in summary["rooms"].items()}
    result = {
        "seeds": list(SEEDS),
        "L": SCORED_L,
        "controllers": list(CONTROLLERS),
        "model_seed": m7.MODEL_SEED,
        "sleep_decisions": sleep_decisions(SCORED_L),
        "sleep_share_target": SLEEP_SHARE,
        "kernels": kernels,
        "summary": {**summary, "rooms": rooms_json},
        "m7": _m7_reference(),
        "smoke": {key: smoke_payload[key] for key in smoke_payload if key != "log"},
        "smoke_log": smoke_payload["log"],
        "runs": runs,
    }
    # render wants int room keys
    result_for_md = {
        **result,
        "summary": summary,
    }
    text = render_markdown(result_for_md)
    m7.dump(out_dir / "results.json", result)
    (out_dir / "RESULTS.md").write_text(text)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "assemble"))
    parser.add_argument("--out", default=str(ROOT / "results" / "m7bw3"))
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out)
        return
    smoke_payload = json.loads((Path(args.out) / "smoke.json").read_text())
    assemble(args.out, smoke_payload)


if __name__ == "__main__":
    main()
