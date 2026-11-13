"""M6-W3: sleep is an action on the W3 ring. Whole-store progress is the reward.

Imports m5_w3, m5b_w3, curio and w3_world and does not edit them. No verdict.
"""
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if os.environ.get("PYTHONHASHSEED") != "0" or os.environ.get("CURIO_LR") != "0.004":
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["CURIO_LR"] = "0.004"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import copy
import json
import math
import time
from multiprocessing import get_context

import numpy as np

import curio
import m5_w3
import m5b_w3
from w3_headroom import LR, W3Compressor, evaluate, shortest
from w3_world import World3

SEEDS = (18, 19, 20, 21, 22, 23)
CONTROLLERS = ("prog", "prog_total", "prog_sleep", "fixed_sleep")
LENGTHS = (512, 2048)
SCORED_L = 2048
REPORT_L = 512
DWELL = m5_w3.DWELL
SLEEP_ARM = 9
REPLAY_N = 64
# Ring of 9: the shortest path is at most 4 moves, then 8 dwells.
MAX_ROOM_VISIT = 4 + DWELL
KERNEL_WALL_LIMIT_S = 90 * 60
SMOKE_SEED = 18
REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "m5bw3", "results.json")
NFEAT = 1 + len(curio.ORDERS)


def _bits_saved(before, after):
    return float(sum(before[c] - after[c] for c in curio.CH))


def _pack(comp, channel, pairs):
    """Feature rows for score_pairs, one row per stored pair, in pair order."""
    ci = curio.CH.index(channel)
    n = len(pairs)
    idx = np.zeros((n, NFEAT), np.int64)
    mask = np.zeros((n, NFEAT), np.bool_)
    xs = np.empty(n, np.int64)
    for i, (ctx, x) in enumerate(pairs):
        feats = comp.feats(ci, list(ctx))
        k = len(feats)
        idx[i, :k] = feats
        mask[i, :k] = True
        xs[i] = int(x)
    return idx, mask, xs


def _score_pack(comp, pack):
    """Same reduction as SGDCompressor.score_pairs. Padded rows are zeros."""
    idx, mask, xs = pack
    gathered = np.where(mask[:, :, None], comp.W[idx], np.float32(0))
    z = gathered.sum(axis=1)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    tot = 0.0
    for i, x in enumerate(xs):
        tot -= math.log2(max(float(p[i, int(x)]), 1e-12))
    return float(tot)


class Store:
    """Per-channel reservoirs. Scoring matches curio.SGDCompressor.score_pairs."""

    def __init__(self, comp):
        self.comp = comp
        self.res = {c: [] for c in curio.CH}
        self.seen = {c: 0 for c in curio.CH}
        self.pack = {c: None for c in curio.CH}

    def score_channel(self, channel):
        pairs = self.res[channel]
        if not pairs:
            return 0.0
        if self.pack[channel] is None:
            self.pack[channel] = _pack(self.comp, channel, pairs)
        return _score_pack(self.comp, self.pack[channel])

    def score_all(self):
        return {c: self.score_channel(c) for c in curio.CH}

    def invalidate(self, channel):
        self.pack[channel] = None


def _learn_pair(comp, channel, pair):
    ctx, x = pair
    comp.bits(channel, list(ctx), [int(x)], learn=True)


def _replay(comp, store, rng):
    """64 independent uniform draws from the union of the room reservoirs.

    With replacement. An empty store learns nothing and draws nothing.
    Drawn pairs are not written back into any reservoir.
    """
    pool = [(c, pair) for c in curio.CH for pair in store.res[c]]
    if not pool:
        return 0
    n = len(pool)
    for _ in range(REPLAY_N):
        channel, pair = pool[int(rng.integers(n))]
        _learn_pair(comp, channel, pair)
    return REPLAY_N


def _room_decision(comp, hist, store, res_rng, channel, symbols):
    """Learn one room glance. Score the store before its pairs are inserted."""
    before = store.score_all()
    comp.bits(channel, hist[channel], symbols, learn=True)
    after = store.score_all()
    full = hist[channel] + symbols
    max_k = max(curio.ORDERS)
    changed = False
    for i in range(len(hist[channel]), len(full)):
        pair = (tuple(full[max(0, i - max_k):i]), full[i])
        store.seen[channel] += 1
        if len(store.res[channel]) < curio.R_RES:
            store.res[channel].append(pair)
            changed = True
        else:
            k = int(res_rng.integers(store.seen[channel]))
            if k < curio.R_RES:
                store.res[channel][k] = pair
                changed = True
    if changed:
        store.invalidate(channel)
    hist[channel] += symbols
    return _bits_saved(before, after)


def _corridor_decision(comp, hist, store, symbols):
    before = store.score_all()
    comp.bits("corridor", hist["corridor"], symbols, learn=True)
    after = store.score_all()
    hist["corridor"] += symbols
    return _bits_saved(before, after)


def _sleep_decision(comp, store, rng):
    before = store.score_all()
    _replay(comp, store, rng)
    after = store.score_all()
    return _bits_saved(before, after)


def _split_even(total, n):
    """n non-negative ints that sum to total and differ by at most 1.

    The extra ones go to the earlier parts.
    """
    if n <= 0:
        return []
    base, rem = divmod(int(total), int(n))
    return [base + (1 if i < rem else 0) for i in range(n)]


def sleep_plan(L, n_sleep):
    """Evenly spaced sleep visits whose lengths sum to n_sleep.

    Visits are 8 decisions, and the last is shorter when n_sleep is not a
    multiple of 8. Awake decisions fill the gaps before, between and after the
    visits, and those gaps differ by at most 1. Each entry is (start, length).
    """
    n_sleep = int(n_sleep)
    if n_sleep <= 0:
        return []
    if n_sleep > L:
        raise ValueError(f"sleep decisions {n_sleep} exceed L={L}")
    lengths = [DWELL] * (n_sleep // DWELL)
    rem = n_sleep % DWELL
    if rem:
        lengths.append(rem)
    gaps = _split_even(L - n_sleep, len(lengths) + 1)
    plan = []
    t = 0
    for i, length in enumerate(lengths):
        t += gaps[i]
        plan.append((int(t), int(length)))
        t += length
    if t + gaps[-1] != L:
        raise RuntimeError(f"sleep plan covers {t + gaps[-1]} decisions, L={L}")
    return plan


def _owed(plan, si):
    return int(sum(length for _, length in plan[si:]))


def _sleep_due(t, L, plan, si):
    """True at a visit boundary when the next planned sleep should start.

    A room visit is also refused when the decisions still left could not cover
    the sleeps still owed plus one maximum room visit. That keeps a delay from
    pushing owed sleeps off the end of the life.
    """
    if si >= len(plan):
        return False
    owed = _owed(plan, si)
    if t >= plan[si][0]:
        return True
    if (L - t) <= owed:
        return True
    if (L - t) < owed + MAX_ROOM_VISIT:
        return True
    return False


def _shares(channels):
    n = len(channels)
    counts = {c: 0 for c in curio.CH}
    corridor = 0
    sleep = 0
    for ch in channels:
        if ch == "corridor":
            corridor += 1
        elif ch == "sleep":
            sleep += 1
        else:
            counts[ch] += 1
    share = {c: counts[c] / n for c in curio.CH}
    return share, corridor / n, sleep / n


def _quarters(channels):
    q = len(channels) // 4
    out = []
    for i in range(4):
        share, corridor, sleep = _shares(channels[i * q:(i + 1) * q])
        out.append({"share": share, "corridor": corridor, "sleep": sleep})
    return out


def _empty_reward_quarters():
    return [{"sleep_n": 0, "sleep_sum": 0.0, "room_n": 0, "room_sum": 0.0} for _ in range(4)]


def _note_visit(quarters, kind, start, L, reward):
    q = L // 4
    i = min(int(start) // q, 3)
    slot = quarters[i]
    if kind == "sleep":
        slot["sleep_n"] += 1
        slot["sleep_sum"] += float(reward)
    else:
        slot["room_n"] += 1
        slot["room_sum"] += float(reward)


def _reward_quarters(raw):
    out = []
    for slot in raw:
        out.append({
            "sleep_n": int(slot["sleep_n"]),
            "sleep_mean": (float(slot["sleep_sum"] / slot["sleep_n"]) if slot["sleep_n"] else None),
            "room_n": int(slot["room_n"]),
            "room_mean": (float(slot["room_sum"] / slot["room_n"]) if slot["room_n"] else None),
        })
    return out


def _finish_row(seed, controller, L, comp, world, hist, channels, t0, score, reward_q, sleep_target):
    eval_s = 0.0
    if score:
        e0 = time.perf_counter()
        W, held = evaluate(comp, world, hist)
        extra = m5b_w3.score_nz(comp, hist, seed)
        eval_s = time.perf_counter() - e0
        held9 = {c: float(held[c]) for c in curio.LEARN}
        held9["N"] = float(extra["N"])
        held9["Z"] = float(extra["Z"])
        W9 = float(np.mean([held9[c] for c in curio.CH]))
        finite = bool(
            math.isfinite(W) and math.isfinite(W9)
            and all(math.isfinite(held9[c]) for c in curio.CH)
        )
    else:
        W, W9 = float("nan"), float("nan")
        held, held9 = {}, {}
        finite = False
    share, corridor, sleep = _shares(channels)
    quarter = []
    for block in _quarters(channels):
        quarter.append({
            "share": {c: float(block["share"][c]) for c in curio.CH},
            "corridor": float(block["corridor"]),
            "sleep": float(block["sleep"]),
        })
    wall = time.perf_counter() - t0
    return {
        "seed": int(seed),
        "controller": controller,
        "L": int(L),
        "W": float(W),
        "W9": float(W9),
        "held": {c: float(held[c]) for c in held},
        "held9": {c: float(held9[c]) for c in held9},
        "share": {c: float(share[c]) for c in curio.CH},
        "corridor_share": float(corridor),
        "sleep_share": float(sleep),
        "sleep_decisions": int(sum(ch == "sleep" for ch in channels)),
        "sleep_target": (None if sleep_target is None else int(sleep_target)),
        "quarter": quarter,
        "reward_quarter": _reward_quarters(reward_q),
        "room_channel": list(world.rooms),
        "wall_s": float(wall),
        "eval_s": float(eval_s),
        "learn_s": float(wall - eval_s),
        "finite": finite,
    }


def _choose_arm(world, unvisited, first_sleep, v, n_arms, rng):
    """M5's bandit. Sleep is the last arm of the optimistic first pass."""
    if unvisited:
        target = m5_w3.nearest_unvisited(world.room, unvisited)
        unvisited.remove(target)
        return target, first_sleep
    if first_sleep:
        return SLEEP_ARM, False
    if rng.random() < curio.EPS:
        return int(rng.integers(n_arms)), first_sleep
    return max(range(n_arms), key=lambda k: (v[k], rng.random())), first_sleep


def run_whole(seed, kind, L, score=True, trace_visits=0, n_sleep_target=None):
    """prog_total, prog_sleep, or fixed_sleep. Whole-store reward. Life length L."""
    if kind not in ("prog_total", "prog_sleep", "fixed_sleep"):
        raise ValueError(kind)
    if L % 4 != 0:
        raise ValueError(f"L={L} is not divisible by 4")
    if kind == "fixed_sleep":
        if n_sleep_target is None:
            raise ValueError("fixed_sleep needs prog_sleep's sleep-decision count")
        plan = sleep_plan(L, int(n_sleep_target))
    else:
        plan = []
        if n_sleep_target not in (None, 0):
            raise ValueError("n_sleep_target is only for fixed_sleep")
    t0 = time.perf_counter()
    world = World3(seed)
    choice_rng = np.random.default_rng(10_000 + seed)
    world_rng = np.random.default_rng(1_000 + seed)
    res_rng = np.random.default_rng(20_000 + seed)
    sleep_rng = np.random.default_rng(30_000 + seed)
    comp = W3Compressor(lr=LR)
    hist = m5_w3._empty_hist()
    store = Store(comp)
    n_arms = 10 if kind == "prog_sleep" else 9
    v = [0.0] * n_arms
    nvis = [0] * n_arms
    unvisited = set(range(9))
    first_sleep = kind == "prog_sleep"
    channels = []
    visits = []
    reward_q = _empty_reward_quarters()
    si = 0
    t = 0

    def emit(arm, kind_visit, channel, start, decisions, reward):
        _note_visit(reward_q, kind_visit, start, L, reward)
        if trace_visits and len(visits) < trace_visits:
            visits.append({
                "arm": arm,
                "channel": channel,
                "start": int(start),
                "decisions": int(decisions),
                "reward": float(reward),
            })

    def one_sleep_visit(length):
        nonlocal t
        start = t
        sum_r = 0.0
        n = 0
        while n < length and t < L:
            sum_r += _sleep_decision(comp, store, sleep_rng)
            channels.append("sleep")
            n += 1
            t += 1
        reward = sum_r / n if n else 0.0
        return start, n, reward

    while t < L:
        if kind == "fixed_sleep" and _sleep_due(t, L, plan, si):
            _start, n, reward = one_sleep_visit(plan[si][1])
            emit("sleep", "sleep", "sleep", _start, n, reward)
            si += 1
            continue
        arm, first_sleep = _choose_arm(world, unvisited, first_sleep, v, n_arms, choice_rng)
        if arm == SLEEP_ARM:
            _start, n, reward = one_sleep_visit(min(DWELL, L - t))
            m5_w3._ema(v, nvis, SLEEP_ARM, reward)
            emit("sleep", "sleep", "sleep", _start, n, reward)
            continue
        target = int(arm)
        start = t
        path = shortest(world.room, target)
        sum_r = 0.0
        travel = 0
        while path and t < L:
            action = path.pop(0)
            _room, channel, symbols = world.step(action, world_rng)
            sum_r += _corridor_decision(comp, hist, store, symbols)
            channels.append(channel)
            travel += 1
            t += 1
        dwell = 0
        while dwell < DWELL and t < L:
            _room, channel, symbols = world.step("stay", world_rng)
            sum_r += _room_decision(comp, hist, store, res_rng, channel, symbols)
            channels.append(channel)
            dwell += 1
            t += 1
        n_dec = travel + dwell
        reward = sum_r / n_dec if n_dec else 0.0
        m5_w3._ema(v, nvis, target, reward)
        emit(int(target), "room", world.rooms[target], start, n_dec, reward)
    if len(channels) != L:
        raise RuntimeError(f"{kind} seed {seed} L={L} ran {len(channels)} decisions")
    if kind == "fixed_sleep" and si != len(plan):
        raise RuntimeError(
            f"fixed_sleep seed {seed} L={L} took {si} of {len(plan)} planned sleep visits"
        )
    row = _finish_row(
        seed, kind, L, comp, world, hist, channels, t0, score, reward_q,
        (int(n_sleep_target) if kind == "fixed_sleep" else None),
    )
    if kind == "fixed_sleep" and row["sleep_decisions"] != int(n_sleep_target):
        raise RuntimeError(
            f"fixed_sleep seed {seed} L={L} sleep decisions {row['sleep_decisions']} "
            f"!= target {int(n_sleep_target)}"
        )
    row["visits"] = visits if trace_visits else None
    return row


def run_prog(seed, L, score=True):
    """M5's prog controller, plus W9 from m5b. No sleep."""
    m5b_w3.install()
    row = m5_w3.run_controller(seed, "prog", L, score=score, trace_visits=0)
    for block in row["quarter"]:
        block["sleep"] = 0.0
    row["sleep_share"] = 0.0
    row["sleep_decisions"] = 0
    row["sleep_target"] = None
    row["reward_quarter"] = None
    if score and "W9" not in row:
        raise RuntimeError("prog row has no W9; m5b hook did not run")
    return row


def run_controller(seed, controller, L, score=True, trace_visits=0, n_sleep_target=None):
    if controller == "prog":
        return run_prog(seed, L, score=score)
    return run_whole(
        seed, controller, L, score=score, trace_visits=trace_visits, n_sleep_target=n_sleep_target,
    )


def run_pair(seed, L, score=True, trace_visits=0):
    """prog_sleep, then fixed_sleep at that life length's sleep dose."""
    slept = run_whole(seed, "prog_sleep", L, score=score, trace_visits=trace_visits)
    fixed = run_whole(
        seed, "fixed_sleep", L, score=score, trace_visits=0,
        n_sleep_target=slept["sleep_decisions"],
    )
    return slept, fixed


def _world_snap(world, world_rng):
    eca = {}
    for key, val in world.h20.eca.items():
        eca[key] = None if val is None else np.array(val, copy=True)
    return {
        "room": int(world.room),
        "buf": {k: list(v) for k, v in world.h20.buf.items()},
        "count": {k: int(v) for k, v in world.h20.count.items()},
        "eca": eca,
        "h20": copy.deepcopy(world.h20.r.bit_generator.state),
        "wrng": copy.deepcopy(world_rng.bit_generator.state),
    }


def _snap_equal(a, b):
    if a["room"] != b["room"] or a["buf"] != b["buf"] or a["count"] != b["count"]:
        return False
    if a["h20"] != b["h20"] or a["wrng"] != b["wrng"]:
        return False
    if set(a["eca"]) != set(b["eca"]):
        return False
    for key, val in a["eca"].items():
        other = b["eca"][key]
        if val is None or other is None:
            if val is not None or other is not None:
                return False
        elif not np.array_equal(val, other):
            return False
    return True


def _self_check_score():
    rng = np.random.default_rng(0)
    comp = W3Compressor(lr=LR)
    comp.W += rng.normal(0, 0.05, comp.W.shape).astype(np.float32)
    worst = 0.0
    for trial in range(20):
        pairs = []
        n_pairs = int(rng.integers(0, curio.R_RES + 1))
        for _ in range(n_pairs):
            nctx = int(rng.integers(0, 7))
            ctx = tuple(int(x) for x in rng.integers(0, curio.A, nctx))
            pairs.append((ctx, int(rng.integers(0, curio.A))))
        channel = curio.CH[int(rng.integers(len(curio.CH)))]
        ref = comp.score_pairs(channel, pairs) if pairs else 0.0
        got = _score_pack(comp, _pack(comp, channel, pairs)) if pairs else 0.0
        worst = max(worst, abs(got - ref))
        if pairs:
            ctx, x = pairs[0]
            comp.bits(channel, list(ctx), [int(x)], learn=True)
            ref2 = comp.score_pairs(channel, pairs)
            got2 = _score_pack(comp, _pack(comp, channel, pairs))
            worst = max(worst, abs(got2 - ref2))
        if worst > 0.0:
            raise SystemExit(f"score pack diverges from score_pairs by {worst:.3e} at trial {trial}")
    return worst


def _check_row(row, tag):
    if row["controller"] == "prog":
        if "held9" not in row or set(row["held9"]) != set(curio.CH):
            raise SystemExit(f"{tag}: prog held9 missing")
    else:
        if set(row["held9"]) != set(curio.CH):
            raise SystemExit(f"{tag}: held9 channels are {sorted(row['held9'])}")
    if abs(float(np.mean([row["held"][c] for c in curio.LEARN])) - row["W"]) > 1e-12:
        raise SystemExit(f"{tag}: W is not the mean of the 7 learnable channels")
    if abs(float(np.mean([row["held9"][c] for c in curio.CH])) - row["W9"]) > 1e-12:
        raise SystemExit(f"{tag}: W9 is not the mean of the 9 channels")
    for c in curio.LEARN:
        if abs(row["held9"][c] - row["held"][c]) > 1e-12:
            raise SystemExit(f"{tag}: held9[{c}] drifted from held")
    total = sum(row["share"].values()) + row["corridor_share"] + row["sleep_share"]
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(f"{tag}: shares sum to {total}")
    qsum = 0.0
    for block in row["quarter"]:
        part = sum(block["share"].values()) + block["corridor"] + block["sleep"]
        if abs(part - 1.0) > 1e-9:
            raise SystemExit(f"{tag}: a quarter's shares sum to {part}")
        qsum += block["sleep"]
    if not row["finite"]:
        raise SystemExit(f"{tag}: W or W9 is not finite")


def _ref_prog_w():
    payload = json.load(open(REF_PATH))
    hit = [r for r in payload["runs"] if r["controller"] == "prog" and r["seed"] == 12 and r["L"] == 512]
    if len(hit) != 1:
        raise SystemExit(f"expected one M5b prog seed 12 L=512 row, got {len(hit)}")
    return float(hit[0]["W"])


def _pool_wall(durations, workers):
    loads = [0.0] * workers
    for dur in sorted(durations, reverse=True):
        loads[0] += dur
        loads.sort()
    return max(loads)


def _estimate(timed):
    """Kernel wall for one seed from the L=128 lives. Linear in L. Pool of 4.

    fixed_sleep waits on prog_sleep, so those two are one chain job. The L=128
    lives include reservoir fill-up, so scaling them up overestimates a long life.
    """
    def at(controller, L):
        return timed[controller] * (L / 128.0)

    jobs = []
    for L in LENGTHS:
        jobs.append(at("prog", L))
        jobs.append(at("prog_total", L))
        jobs.append(at("prog_sleep", L) + at("fixed_sleep", L))
    wall = _pool_wall(jobs, 4)
    return {
        "L128": {c: timed[c] for c in CONTROLLERS},
        "kernel_wall_s": wall,
        "kernel_wall_min": wall / 60.0,
        "limit_s": KERNEL_WALL_LIMIT_S,
        "over_limit": wall > KERNEL_WALL_LIMIT_S,
        "jobs_s": jobs,
    }


def _check_plan():
    for L, n_sleep in ((128, 0), (128, 1), (128, 8), (128, 64), (2048, 200), (512, 512)):
        plan = sleep_plan(L, n_sleep)
        got = sum(length for _, length in plan)
        if got != n_sleep:
            raise SystemExit(f"sleep plan L={L} n={n_sleep} sums to {got}")
        prev = -1
        for start, length in plan:
            if start < prev or start + length > L:
                raise SystemExit(f"sleep plan overlap or overflow L={L} n={n_sleep}")
            prev = start + length
        if n_sleep == 0 and plan:
            raise SystemExit("empty dose produced a sleep visit")


def _sleep_invariance():
    """One sleep visit leaves the world stream and the reservoirs unchanged."""
    seed = SMOKE_SEED
    world = World3(seed)
    world_rng = np.random.default_rng(1_000 + seed)
    res_rng = np.random.default_rng(20_000 + seed)
    sleep_rng = np.random.default_rng(30_000 + seed)
    comp = W3Compressor(lr=LR)
    hist = m5_w3._empty_hist()
    store = Store(comp)
    # Two dwells in room 0, so the store is non-empty before the sleep.
    for _ in range(2):
        _room, channel, symbols = world.step("stay", world_rng)
        _room_decision(comp, hist, store, res_rng, channel, symbols)
    before_res = {c: list(store.res[c]) for c in curio.CH}
    before_seen = dict(store.seen)
    before_world = _world_snap(world, world_rng)
    weights = comp.W.copy()
    scored = {}
    for c in curio.CH:
        ref = comp.score_pairs(c, store.res[c]) if store.res[c] else 0.0
        got = store.score_channel(c)
        if got != ref and abs(got - ref) > 0.0:
            raise SystemExit(f"sleep setup score diverges on {c} by {abs(got - ref):.3e}")
        scored[c] = got
    for _ in range(DWELL):
        _sleep_decision(comp, store, sleep_rng)
    after_res = {c: list(store.res[c]) for c in curio.CH}
    after_world = _world_snap(world, world_rng)
    if before_res != after_res or before_seen != store.seen:
        raise SystemExit("sleep visit changed a reservoir")
    if not _snap_equal(before_world, after_world):
        raise SystemExit("sleep visit changed the world stream or the room")
    if np.array_equal(weights, comp.W):
        raise SystemExit("sleep visit did not change the compressor")
    for c in curio.CH:
        ref = comp.score_pairs(c, store.res[c]) if store.res[c] else 0.0
        got = store.score_channel(c)
        if abs(got - ref) > 0.0:
            raise SystemExit(f"post-sleep score diverges on {c} by {abs(got - ref):.3e}")
    return {
        "reservoir_unchanged": True,
        "world_unchanged": True,
        "weights_changed": True,
        "score_matches": True,
        "pre_sleep_bits": scored[world.rooms[0]],
    }


def smoke():
    """Four checks from the spec. Exit before any kernel if one fails."""
    m5b_w3.install()
    lines = []

    def say(msg):
        print(msg, flush=True)
        lines.append(msg)

    say("smoke check 0: packed score matches score_pairs, and the sleep plan fits")
    _check_plan()
    worst = _self_check_score()
    say(f"  packed score abs {worst:.3e} on 20 random reservoirs")

    say("smoke check 1: prog seed 12 L=512 vs results/m5bw3/results.json")
    ref = _ref_prog_w()
    repro = run_prog(12, 512, score=True)
    dW = abs(repro["W"] - ref)
    say(
        f'  prog W {repro["W"]:.12f} ref {ref:.12f} abs {dW:.3e} '
        f'W9 {repro["W9"]:.6f} finite {repro["finite"]} wall {repro["wall_s"]:.2f}s'
    )
    _check_row(repro, "smoke check 1")
    if dW > 1e-9:
        raise SystemExit(f"smoke check 1 failed: abs W {dW:.3e} > 1e-9")
    if repro["sleep_decisions"] != 0 or repro["sleep_share"] != 0.0:
        raise SystemExit("smoke check 1 failed: prog slept")

    say("smoke check 2: a sleep visit changes no reservoir and no world stream")
    inv = _sleep_invariance()
    say(
        "  reservoir identical True world identical True "
        f'weights changed {inv["weights_changed"]} score matches {inv["score_matches"]}'
    )

    say("smoke check 3: prog_sleep seed 18 L=256 visit trace")
    traced = run_whole(SMOKE_SEED, "prog_sleep", 256, score=True, trace_visits=10 ** 9)
    _check_row(traced, "smoke check 3")
    if not traced["visits"]:
        raise SystemExit("smoke check 3 failed: no visits")
    for i, visit in enumerate(traced["visits"], 1):
        say(
            f'  visit {i} arm {visit["arm"]} {visit["channel"]} '
            f'decisions {visit["decisions"]} reward {visit["reward"]:.6f}'
        )
    say(
        f'  sleep_share {traced["sleep_share"]:.6f} '
        f'sleep_decisions {traced["sleep_decisions"]} / {traced["L"]} '
        f'W {traced["W"]:.6f} W9 {traced["W9"]:.6f} wall {traced["wall_s"]:.2f}s'
    )
    if traced["visits"][9]["arm"] != "sleep":
        raise SystemExit(
            f'smoke check 3 failed: visit 10 arm is {traced["visits"][9]["arm"]}, expected sleep last'
        )
    for visit in traced["visits"][:9]:
        if visit["arm"] == "sleep":
            raise SystemExit("smoke check 3 failed: sleep is not last in the first pass")

    say("smoke check 4: seed 18 L=128, four controllers, W and W9 finite")
    timed = {}
    rows = []
    order = ("prog", "prog_total", "prog_sleep", "fixed_sleep")
    slept_n = None
    for controller in order:
        if controller == "fixed_sleep":
            row = run_whole(
                SMOKE_SEED, "fixed_sleep", 128, score=True, n_sleep_target=slept_n,
            )
        elif controller == "prog":
            row = run_prog(SMOKE_SEED, 128, score=True)
        else:
            row = run_whole(SMOKE_SEED, controller, 128, score=True)
            if controller == "prog_sleep":
                slept_n = row["sleep_decisions"]
        say(
            f'  {controller} W {row["W"]:.6f} W9 {row["W9"]:.6f} finite {row["finite"]} '
            f'sleep {row["sleep_share"]:.4f} corridor {row["corridor_share"]:.4f} '
            f'wall {row["wall_s"]:.2f}s learn {row["learn_s"]:.2f}s eval {row["eval_s"]:.2f}s'
        )
        _check_row(row, f"smoke check 4 {controller}")
        timed[controller] = row["wall_s"]
        rows.append(row)
    fixed = rows[-1]
    if fixed["sleep_decisions"] != slept_n:
        raise SystemExit(
            f"smoke check 4 failed: fixed_sleep decisions {fixed['sleep_decisions']} != prog_sleep {slept_n}"
        )
    if abs(fixed["sleep_share"] - rows[-2]["sleep_share"]) > 1e-12:
        raise SystemExit("smoke check 4 failed: sleep shares differ")

    est = _estimate(timed)
    say(
        f'estimate one-seed kernel wall {est["kernel_wall_s"]:.0f}s '
        f'({est["kernel_wall_min"]:.1f} min) pool 4 limit {KERNEL_WALL_LIMIT_S / 60:.0f} min'
    )
    if est["over_limit"]:
        say("estimate exceeds 90 min; do not push kernels")
    else:
        say("estimate is within 90 min per kernel")
    return {
        "lines": lines,
        "text": "\n".join(lines),
        "score_abs": worst,
        "invariance": inv,
        "reproduce": {
            "seed": 12,
            "L": 512,
            "controller": "prog",
            "W": repro["W"],
            "W_ref": ref,
            "abs_W": dW,
            "W9": repro["W9"],
            "wall_s": repro["wall_s"],
            "finite": repro["finite"],
        },
        "trace": {
            "seed": SMOKE_SEED,
            "L": 256,
            "controller": "prog_sleep",
            "sleep_share": traced["sleep_share"],
            "sleep_decisions": traced["sleep_decisions"],
            "W": traced["W"],
            "W9": traced["W9"],
            "wall_s": traced["wall_s"],
            "visits": traced["visits"],
        },
        "L128": [{
            "controller": r["controller"],
            "seed": r["seed"],
            "L": r["L"],
            "W": r["W"],
            "W9": r["W9"],
            "finite": r["finite"],
            "sleep_share": r["sleep_share"],
            "sleep_decisions": r["sleep_decisions"],
            "corridor_share": r["corridor_share"],
            "wall_s": r["wall_s"],
            "learn_s": r["learn_s"],
            "eval_s": r["eval_s"],
        } for r in rows],
        "estimate": est,
        "over_limit": est["over_limit"],
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
    }


def _job(args):
    seed, kind, L = args
    m5b_w3.install()
    if kind == "prog":
        row = run_prog(seed, L, score=True)
        rows = [row]
    elif kind == "prog_total":
        row = run_whole(seed, "prog_total", L, score=True)
        rows = [row]
    elif kind == "chain":
        rows = list(run_pair(seed, L, score=True))
    else:
        raise ValueError(kind)
    for row in rows:
        row.pop("visits", None)
        print(
            f's{seed} {row["controller"]} L={L} W {row["W"]:.4f} W9 {row["W9"]:.4f} '
            f'sleep {row["sleep_share"]:.3f} corridor {row["corridor_share"]:.3f} '
            f'wall {row["wall_s"]:.1f}s',
            flush=True,
        )
        _check_row(row, f's{seed} {row["controller"]} L={L}')
    return rows


def kernel_jobs(seed):
    """Long whole-store lives first. fixed_sleep is inside the chain, after prog_sleep."""
    jobs = []
    for L in (SCORED_L, REPORT_L):
        jobs.append((seed, "chain", L))
        jobs.append((seed, "prog_total", L))
    for L in (SCORED_L, REPORT_L):
        jobs.append((seed, "prog", L))
    return jobs


def kernel_main(seed, out, workers=4, kernel_id=None):
    if int(workers) != 4:
        raise SystemExit("kernel pool is 4")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 18-23")
    m5b_w3.install()
    t0 = time.perf_counter()
    jobs = kernel_jobs(seed)
    print(f"kernel {kernel_id} seed {seed} jobs {len(jobs)} workers {workers}", flush=True)
    ctx = get_context("fork")
    with ctx.Pool(workers) as pool:
        nested = list(pool.imap_unordered(_job, jobs, chunksize=1))
    runs = [row for batch in nested for row in batch]
    wall = time.perf_counter() - t0
    if len(runs) != len(CONTROLLERS) * len(LENGTHS):
        raise SystemExit(f"expected {len(CONTROLLERS) * len(LENGTHS)} runs, got {len(runs)}")
    for row in runs:
        if not row["finite"]:
            raise SystemExit(f's{seed} {row["controller"]} L={row["L"]} is not finite')
        if row["controller"] == "fixed_sleep":
            match = [
                r for r in runs
                if r["controller"] == "prog_sleep" and r["L"] == row["L"] and r["seed"] == seed
            ]
            if len(match) != 1 or row["sleep_decisions"] != match[0]["sleep_decisions"]:
                raise SystemExit(f"fixed_sleep dose mismatch at L={row['L']}")
    payload = {
        "kernel_id": kernel_id,
        "seed": seed,
        "workers": workers,
        "wall_s": wall,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
        "runs": runs,
    }
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"m6_s{seed}.json")
    json.dump(payload, open(path, "w"), indent=1)
    open(path, "a").write("\n")
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    return payload


def _pick(runs, controller, L, seed):
    hit = [r for r in runs if r["controller"] == controller and r["L"] == L and r["seed"] == seed]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {controller} L={L} seed {seed}, got {len(hit)}")
    return hit[0]


def _mean_std(xs):
    return m5_w3.mean_std(xs)


def summarize(runs):
    by = {}
    for L in LENGTHS:
        for controller in CONTROLLERS:
            rows = sorted(
                (r for r in runs if r["controller"] == controller and r["L"] == L),
                key=lambda r: r["seed"],
            )
            if len(rows) != len(SEEDS):
                raise RuntimeError(f"{controller} L={L}: {len(rows)} rows, expected {len(SEEDS)}")
            w_mu, w_sd = _mean_std(r["W"] for r in rows)
            w9_mu, w9_sd = _mean_std(r["W9"] for r in rows)
            held_mu = {c: _mean_std(r["held"][c] for r in rows)[0] for c in curio.LEARN}
            held9_mu = {c: _mean_std(r["held9"][c] for r in rows)[0] for c in curio.CH}
            share_mu = {c: _mean_std(r["share"][c] for r in rows)[0] for c in curio.CH}
            corr_mu = _mean_std(r["corridor_share"] for r in rows)[0]
            sleep_mu, sleep_sd = _mean_std(r["sleep_share"] for r in rows)
            quarter = []
            for i in range(4):
                q_share = {c: _mean_std(r["quarter"][i]["share"][c] for r in rows)[0] for c in curio.CH}
                q_corr = _mean_std(r["quarter"][i]["corridor"] for r in rows)[0]
                q_sleep = _mean_std(r["quarter"][i]["sleep"] for r in rows)[0]
                quarter.append({"share": q_share, "corridor": q_corr, "sleep": q_sleep})
            by[f"{controller}/L{L}"] = {
                "controller": controller,
                "L": L,
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
                "per_seed": [{
                    "seed": r["seed"],
                    "W": r["W"],
                    "W9": r["W9"],
                    "held": r["held"],
                    "held9": r["held9"],
                    "share": r["share"],
                    "corridor_share": r["corridor_share"],
                    "sleep_share": r["sleep_share"],
                    "sleep_decisions": r["sleep_decisions"],
                    "quarter": r["quarter"],
                    "reward_quarter": r["reward_quarter"],
                    "room_channel": r["room_channel"],
                    "wall_s": r["wall_s"],
                } for r in rows],
            }
    gaps = []
    for L in LENGTHS:
        for s in SEEDS:
            prog = _pick(runs, "prog", L, s)
            total = _pick(runs, "prog_total", L, s)
            slept = _pick(runs, "prog_sleep", L, s)
            fixed = _pick(runs, "fixed_sleep", L, s)
            gaps.append({
                "L": L,
                "seed": s,
                "prog_sleep": slept["W"],
                "prog_total": total["W"],
                "fixed_sleep": fixed["W"],
                "prog": prog["W"],
                "prog_sleep_minus_prog_total": slept["W"] - total["W"],
                "prog_sleep_minus_fixed_sleep": slept["W"] - fixed["W"],
                "prog_total_minus_prog": total["W"] - prog["W"],
            })
    return {"by_controller": by, "gaps": gaps}


def _fmt(x):
    return f"{x:.6f}"


def _pm(mu, sd):
    return f"{_fmt(mu)} ± {_fmt(sd)}"


def _seed_header():
    return " | ".join(f"seed {s}" for s in SEEDS)


def _cell_mean(value):
    if value is None:
        return "—"
    return _fmt(value)


def render_markdown(payload):
    by = payload["summary"]["by_controller"]
    lines = []
    lines.append("# M6-W3 sleep")
    lines.append("")
    lines.append(
        "Sleep is an action on the W3 ring. "
        "W is bits/symbol of the frozen compressor on a fresh continuation of the 7 learnable channels. "
        "W9 is that mean over all 9 room channels. Lower is better. Numbers only."
    )
    lines.append("")
    for title, key_mu, key_sd, field in (
        ("W", "W_mean", "W_std_ddof0", "W"),
        ("W9", "W9_mean", "W9_std_ddof0", "W9"),
    ):
        lines.append(f"## {title}")
        lines.append("")
        for L in LENGTHS:
            lines.append(f"### L = {L}")
            lines.append("")
            lines.append(f"| controller | {_seed_header()} | mean ± std ddof=0 |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for controller in CONTROLLERS:
                block = by[f"{controller}/L{L}"]
                cells = []
                for s in SEEDS:
                    hit = [p for p in block["per_seed"] if p["seed"] == s][0]
                    cells.append(_fmt(hit[field]))
                lines.append(
                    f"| {controller} | " + " | ".join(cells)
                    + f' | {_pm(block[key_mu], block[key_sd])} |'
                )
            lines.append("")
    lines.append("## Gaps")
    lines.append("")
    lines.append("On W. prog_sleep minus prog_total, prog_sleep minus fixed_sleep, prog_total minus prog.")
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append("| seed | prog_sleep − prog_total | prog_sleep − fixed_sleep | prog_total − prog |")
        lines.append("|---:|---:|---:|---:|")
        gaps = [g for g in payload["summary"]["gaps"] if g["L"] == L]
        for g in gaps:
            lines.append(
                f'| {g["seed"]} | {_fmt(g["prog_sleep_minus_prog_total"])} | '
                f'{_fmt(g["prog_sleep_minus_fixed_sleep"])} | {_fmt(g["prog_total_minus_prog"])} |'
            )
        mu1, sd1 = _mean_std(g["prog_sleep_minus_prog_total"] for g in gaps)
        mu2, sd2 = _mean_std(g["prog_sleep_minus_fixed_sleep"] for g in gaps)
        mu3, sd3 = _mean_std(g["prog_total_minus_prog"] for g in gaps)
        n1 = sum(g["prog_sleep_minus_prog_total"] < 0 for g in gaps)
        n2 = sum(g["prog_sleep_minus_fixed_sleep"] < 0 for g in gaps)
        n3 = sum(g["prog_total_minus_prog"] < 0 for g in gaps)
        lines.append(f"| mean | {_fmt(mu1)} | {_fmt(mu2)} | {_fmt(mu3)} |")
        lines.append(f"| std ddof=0 | {_fmt(sd1)} | {_fmt(sd2)} | {_fmt(sd3)} |")
        lines.append(f"| left lower (count) | {n1} of 6 | {n2} of 6 | {n3} of 6 |")
        lines.append("")
    lines.append("## Sleep share")
    lines.append("")
    lines.append("Sleep decisions divided by L. fixed_sleep uses prog_sleep's count on the same seed and L.")
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append(f"| controller | {_seed_header()} | mean ± std ddof=0 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for controller in CONTROLLERS:
            block = by[f"{controller}/L{L}"]
            cells = []
            for s in SEEDS:
                hit = [p for p in block["per_seed"] if p["seed"] == s][0]
                cells.append(_fmt(hit["sleep_share"]))
            lines.append(
                f"| {controller} | " + " | ".join(cells)
                + f' | {_pm(block["sleep_mean"], block["sleep_std_ddof0"])} |'
            )
        lines.append("")
    lines.append("## prog_sleep sleep share by quarter")
    lines.append("")
    lines.append("Quarter i is decisions [(i−1)L/4, iL/4). The share is sleep decisions in the quarter over L/4.")
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append("| seed | quarter 1 | quarter 2 | quarter 3 | quarter 4 |")
        lines.append("|---:|---:|---:|---:|---:|")
        block = by[f"prog_sleep/L{L}"]
        for s in SEEDS:
            hit = [p for p in block["per_seed"] if p["seed"] == s][0]
            cells = [_fmt(hit["quarter"][i]["sleep"]) for i in range(4)]
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        means = [_fmt(block["quarter_mean"][i]["sleep"]) for i in range(4)]
        lines.append("| mean | " + " | ".join(means) + " |")
        lines.append("")
    lines.append("## Decision shares")
    lines.append("")
    lines.append(
        "Mean over seeds. A stay counts for the room channel. A move counts for the corridor. "
        "A sleep decision counts for sleep. Shares sum to 1."
    )
    lines.append("")
    header = "| controller | L | " + " | ".join(list(curio.CH) + ["corridor", "sleep"]) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * 11)
    for L in LENGTHS:
        for controller in CONTROLLERS:
            block = by[f"{controller}/L{L}"]
            cells = [f'{block["share_mean"][c]:.4f}' for c in curio.CH]
            cells.append(f'{block["corridor_mean"]:.4f}')
            cells.append(f'{block["sleep_mean"]:.4f}')
            lines.append(f"| {controller} | {L} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## prog_sleep visit rewards by quarter")
    lines.append("")
    lines.append(
        "Whole-store bits saved per decision, mean over the visits whose first decision falls in the quarter. "
        "A quarter with no visits of that kind is blank. The mean row is the mean of the per-seed means, "
        "over seeds that have at least one such visit."
    )
    lines.append("")
    for L in LENGTHS:
        lines.append(f"### L = {L}")
        lines.append("")
        lines.append("| seed | quarter | sleep visits | sleep mean reward | room visits | room mean reward |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        block = by[f"prog_sleep/L{L}"]
        for s in SEEDS:
            hit = [p for p in block["per_seed"] if p["seed"] == s][0]
            for i, slot in enumerate(hit["reward_quarter"], 1):
                lines.append(
                    f"| {s} | {i} | {slot['sleep_n']} | {_cell_mean(slot['sleep_mean'])} | "
                    f"{slot['room_n']} | {_cell_mean(slot['room_mean'])} |"
                )
        for i in range(4):
            sleep_vals = []
            room_vals = []
            sleep_n = 0
            room_n = 0
            for s in SEEDS:
                hit = [p for p in block["per_seed"] if p["seed"] == s][0]
                slot = hit["reward_quarter"][i]
                sleep_n += slot["sleep_n"]
                room_n += slot["room_n"]
                if slot["sleep_mean"] is not None:
                    sleep_vals.append(slot["sleep_mean"])
                if slot["room_mean"] is not None:
                    room_vals.append(slot["room_mean"])
            sleep_mu = _fmt(_mean_std(sleep_vals)[0]) if sleep_vals else "—"
            room_mu = _fmt(_mean_std(room_vals)[0]) if room_vals else "—"
            lines.append(
                f"| mean | {i + 1} | {sleep_n} | {sleep_mu} | {room_n} | {room_mu} |"
            )
        lines.append("")
    lines.append("## Mean held-out bits")
    lines.append("")
    lines.append("Mean over seeds. W is the mean of the 7 learnable channels. W9 is the mean of all 9.")
    lines.append("")
    lines.append("| controller | L | " + " | ".join(list(curio.CH)) + " | W | W9 |")
    lines.append("|---|---:|" + "---:|" * 11)
    for L in LENGTHS:
        for controller in CONTROLLERS:
            block = by[f"{controller}/L{L}"]
            cells = [_fmt(block["held9_mean"][c]) for c in curio.CH]
            lines.append(
                f"| {controller} | {L} | " + " | ".join(cells)
                + f' | {_fmt(block["W_mean"])} | {_fmt(block["W9_mean"])} |'
            )
    lines.append("")
    lines.append("## Ring order")
    lines.append("")
    lines.append("Channel in rooms 0..8, seeds 18–23. One order per seed; controllers share it.")
    lines.append("")
    lines.append("| seed | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |")
    lines.append("|---:|---|---|---|---|---|---|---|---|---|")
    for s in SEEDS:
        rooms = _pick(payload["runs"], "prog", SCORED_L, s)["room_channel"]
        lines.append(f"| {s} | " + " | ".join(rooms) + " |")
    lines.append("")
    lines.append("## Walls")
    lines.append("")
    kernel_wall = payload.get("kernel_wall_s")
    if kernel_wall is None:
        lines.append(f'Smoke wall {payload["smoke_wall_s"]:.3f} s.')
    else:
        lines.append(
            f'Smoke wall {payload["smoke_wall_s"]:.3f} s. '
            f'Kernel wall {kernel_wall:.3f} s.'
        )
    lines.append("")
    lines.append("| kernel | seed | wall seconds |")
    lines.append("|---|---:|---:|")
    for item in payload["kernels"]:
        lines.append(f'| {item["kernel_id"]} | {item["seed"]} | {item["wall_s"]:.3f} |')
    lines.append("")
    lines.append("| controller | L | seed | wall seconds |")
    lines.append("|---|---:|---:|---:|")
    order = [(c, L) for L in LENGTHS for c in CONTROLLERS]
    runs_sorted = sorted(payload["runs"], key=lambda r: (order.index((r["controller"], r["L"])), r["seed"]))
    for r in runs_sorted:
        lines.append(f'| {r["controller"]} | {r["L"]} | {r["seed"]} | {r["wall_s"]:.3f} |')
    lines.append("")
    lines.append("## Smoke test")
    lines.append("")
    lines.append("```")
    lines.append(payload["smoke_text"].rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        "- `prog` is `m5_w3.run_controller` with the M5 per-channel reward and no sleep. "
        "W9 is `m5b_w3.score_nz` on that frozen compressor. `m5_w3.py`, `m5b_w3.py`, `curio.py` and "
        "`w3_world.py` are imported and not edited."
    )
    lines.append(
        "- `prog_total` is M5's 9-room bandit. `prog_sleep` is the same bandit over 10 arms. "
        "Sleep is the last arm of the optimistic first pass, then eps-greedy with EPS=0.1 and BETA=0.3. "
        "Choice rng is `default_rng(10_000 + seed)`. N-channel noise and corridor steps use "
        "`default_rng(1_000 + seed)`. The reservoir uses `default_rng(20_000 + seed)`."
    )
    lines.append(
        "- A decision's progress is the sum over the 9 room channels of "
        "`score_pairs` before the learning step minus `score_pairs` after it, on the reservoirs as they "
        "stand before the decision's own pairs enter. The visit reward is that sum over the visit's "
        "decisions, divided by the decisions the visit took. Travel counts. Corridor glances are learned "
        "and do not enter a room reservoir."
    )
    lines.append(
        "- The packed score used for that difference was checked against `SGDCompressor.score_pairs`. "
        "The smoke check reports the absolute gap."
    )
    lines.append(
        "- A sleep decision draws 64 pairs uniformly with replacement from the union of the room "
        f"reservoirs, rng `default_rng(30_000 + seed)`, and applies the compressor's learning step to each "
        "pair on its own channel. Pairs enter no reservoir. An empty union learns nothing. A sleep visit "
        "is 8 such decisions where the agent stands, with no travel. Its reward is the summed progress "
        "divided by the decisions it took."
    )
    lines.append(
        "- `fixed_sleep` is the `prog_total` controller plus planned sleep visits. The visit lengths are 8, "
        "with a shorter last visit when the count is not a multiple of 8, and the lengths sum to "
        "`prog_sleep`'s sleep decisions on that seed and L. Awake gaps before, between and after the visits "
        "differ by at most one decision. A due sleep that falls inside a room visit starts on the next "
        "decision after that visit. A room visit is not started when the decisions left would not cover "
        "the sleeps still owed plus one maximum room visit (4 moves + 8 dwells)."
    )
    lines.append(
        "- W is `w3_headroom.evaluate`. W9 adds N and Z on the same frozen compressor. "
        f"N is EVAL uniform symbols from `default_rng({m5b_w3.N_EVAL_BASE} + seed)`, a fresh generator, "
        "not the sleep rng. Z is EVAL space symbols. Corridor is not in W or W9."
    )
    lines.append(
        "- Quarter i is decisions [(i−1)L/4, iL/4). A visit reward is counted in the quarter of its first "
        "decision. Seeds 18–23. L = 2048 is the scored life. L = 512 is the report life. Nothing retuned."
    )
    lines.append(
        "- PYTHONHASHSEED=0. CURIO_LR=0.004. CPU. Private Kaggle kernels, pool of 4, no GPU, no internet."
    )
    lines.append("")
    return "\n".join(lines)


def assemble(parts, smoke_payload, smoke_wall_s, out_dir, kernel_wall_s=None):
    runs = []
    kernels = []
    for part in parts:
        kernels.append({
            "kernel_id": part["kernel_id"],
            "seed": part["seed"],
            "wall_s": part["wall_s"],
            "workers": part["workers"],
        })
        runs.extend(part["runs"])
    if len(runs) != len(SEEDS) * len(CONTROLLERS) * len(LENGTHS):
        raise SystemExit(f"expected {len(SEEDS) * len(CONTROLLERS) * len(LENGTHS)} runs, got {len(runs)}")
    for row in runs:
        _check_row(row, f's{row["seed"]} {row["controller"]} L={row["L"]}')
    summary = summarize(runs)
    payload = {
        "seeds": list(SEEDS),
        "lengths": list(LENGTHS),
        "controllers": list(CONTROLLERS),
        "kernels": kernels,
        "kernel_wall_s": kernel_wall_s,
        "smoke_wall_s": smoke_wall_s,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "curio_lr": os.environ.get("CURIO_LR"),
        "runs": runs,
        "summary": summary,
        "smoke": {
            k: smoke_payload[k]
            for k in (
                "reproduce", "trace", "L128", "estimate", "over_limit",
                "score_abs", "invariance", "pythonhashseed", "curio_lr",
            )
        },
        "smoke_text": smoke_payload["text"],
    }
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "results.json")
    md_path = os.path.join(out_dir, "RESULTS.md")
    json.dump(payload, open(json_path, "w"), indent=1)
    open(json_path, "a").write("\n")
    open(md_path, "w").write(render_markdown(payload))
    print(f"wrote {md_path} and {json_path}", flush=True)
    return payload


def _load_smoke(path):
    payload = json.load(open(path))
    if "text" not in payload or "reproduce" not in payload:
        raise SystemExit(f"smoke file {path} is missing fields")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--kernel", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kernel-id", default=None)
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--kernel-summary", default=None)
    ap.add_argument("--smoke-json", default="results/m6w3/smoke.json")
    ap.add_argument("--out-dir", default="results/m6w3")
    a = ap.parse_args()
    if a.kernel:
        kernel_main(a.seed, a.out_dir, workers=a.workers, kernel_id=a.kernel_id)
        return
    if a.assemble:
        smoke_payload = _load_smoke(a.smoke_json)
        parts = [json.load(open(path)) for path in a.parts]
        kernel_wall_s = None
        if a.kernel_summary:
            kernel_wall_s = float(json.load(open(a.kernel_summary))["wall_s"])
        assemble(
            parts, smoke_payload, smoke_payload.get("smoke_wall_s", float("nan")),
            a.out_dir, kernel_wall_s=kernel_wall_s,
        )
        return
    t_smoke = time.perf_counter()
    result = smoke()
    smoke_wall = time.perf_counter() - t_smoke
    print(f"smoke wall {smoke_wall:.3f}s", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    out = {**result, "smoke_wall_s": smoke_wall}
    json.dump(out, open(os.path.join(a.out_dir, "smoke.json"), "w"), indent=1)
    open(os.path.join(a.out_dir, "smoke.json"), "a").write("\n")
    if result["over_limit"]:
        raise SystemExit("smoke estimate exceeds 90 min; kernels were not pushed")


if __name__ == "__main__":
    main()
