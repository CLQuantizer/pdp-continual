"""W4 headroom: fixed policies on the 18-room ring, with M7's bank and a fixed sleep dose.

Numbers only. No verdict. The cortex is M7's BankCortex with 19 channel rows.
Sleep, the stores, the sleep plan and the frozen-bank evaluation follow m7b_w3.
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
    if os.environ.get("_W4_MAIN") != "1":
        os.environ["_W4_MAIN"] = "1"
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
from h20_world import ALPHA
from w3_headroom import next_action as w3_next_action
from w3_world import World3
from w4_world import (
    DARK,
    IDENTITIES,
    LEARN14,
    LEARN_A,
    LEARN_B,
    N_RING,
    NOISE,
    World4,
    shortest,
    split_learnable,
)

SEEDS = (0, 1, 2, 3, 4, 5)
POLICIES = ("rand_dwell_s", "oracle14_s", "oracle7a_s")
L_SHORT = 2048
L_LONG = 8192
LONG_POLICY = "oracle14_s"
SLEEP_SHARE = 0.20
N_CH = N_RING + 1
KERNEL_WALL_LIMIT_S = 90 * 60
BITS_LIMIT = 1e-6
ROOT = Path(__file__).resolve().parent
CODE_FILES = (
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
DATASET_ID = "ezzzzzx/pdp-code-w4"

if m7.DWELL != 8 or curio.B != 64 or curio.EVAL != 2048 or len(ALPHA) != 41:
    raise RuntimeError("W4 inherited constants drifted")
if m7.N_CH != 10 or m7.CORRIDOR != 9:
    raise RuntimeError("M7 channel width drifted")


def sleep_decisions(L):
    """Nearest multiple of 8 to 20% of the life. Every sleep visit is 8 decisions.

    Same arithmetic as m7b_w3.sleep_decisions.
    """
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


def sleep_due(t, L, plan, si, n_ring):
    """m7.sleep_due with the ring's longest visit in place of the 9-room constant.

    For n_ring=9 the longest visit is 12, which is m7.MAX_ROOM_VISIT.
    """
    if si >= len(plan):
        return False
    still = m7.owed(plan, si)
    if t >= plan[si][0]:
        return True
    if (L - t) <= still:
        return True
    max_visit = (int(n_ring) // 2) + m7.DWELL
    if (L - t) < still + max_visit:
        return True
    return False


def targets_for(rooms, policy):
    if policy == "rand_dwell_s":
        return list(range(len(rooms)))
    if policy == "oracle14_s":
        return [i for i, name in enumerate(rooms) if split_learnable(name) is not None]
    if policy == "oracle7a_s":
        return [
            i for i, name in enumerate(rooms)
            if (split_learnable(name) or (None, None))[1] == "a"
        ]
    raise ValueError(policy)


def next_w4(state, n_ring, targets, rng):
    """Draw the next target only when the current plan is empty. Dwell is 8 stays."""
    if not state["plan"]:
        if not targets:
            raise RuntimeError("policy has no target rooms")
        target = int(targets[int(rng.integers(len(targets)))])
        state["plan"] = shortest(state["room"], target, n_ring) + ["stay"] * m7.DWELL
    return state["plan"].pop(0)


def _threads():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class Glance:
    __slots__ = ("gid", "channel", "symbols", "prev")

    def __init__(self, gid, channel, symbols, prev, n_rooms):
        self.gid = int(gid)
        self.channel = int(channel)
        self.symbols = np.asarray(symbols, dtype=np.int64)
        self.prev = int(prev)
        if self.symbols.shape != (m7.CHUNK,):
            raise RuntimeError(f"glance length {self.symbols.shape}")
        if self.channel < 0 or self.channel >= int(n_rooms):
            raise RuntimeError("room channel index")


class Reservoir:
    """One reservoir per room channel. The replacement rng is drawn only when full."""

    def __init__(self, cap, rng, order):
        self.cap = int(cap)
        self.rng = rng
        self.order = list(order)
        self.slots = {name: [] for name in self.order}
        self.seen = {name: 0 for name in self.order}
        self.ids = set()

    def offer(self, name, glance):
        self.seen[name] += 1
        slots = self.slots[name]
        if len(slots) < self.cap:
            slots.append(glance)
            self.ids.add(glance.gid)
            return "append", None
        k = int(self.rng.integers(self.seen[name]))
        if k < self.cap:
            old = slots[k]
            self.ids.remove(old.gid)
            slots[k] = glance
            self.ids.add(glance.gid)
            return "replace", old
        return "reject", None

    def flat(self):
        return [glance for name in self.order for glance in self.slots[name]]

    def draw(self, rng):
        pool = self.flat()
        if not pool:
            return None
        return pool[int(rng.integers(len(pool)))]

    def sizes(self):
        return {name: len(self.slots[name]) for name in self.order}


class WideCortex(m7.BankCortex):
    """M7 BankCortex with a caller-chosen number of channel rows, initialised at zero.

    trace_glance reads m7.N_CH, so each call widens that name and puts it back.
    Rows the life never reads stay zero and do not change the rows it does read.
    """

    def __init__(self, n_ch, model_seed=m7.MODEL_SEED):
        n_ch = int(n_ch)
        if n_ch < m7.N_CH:
            raise ValueError(f"n_ch {n_ch} is narrower than M7")
        saved = m7.N_CH
        m7.N_CH = n_ch
        try:
            super().__init__(model_seed=model_seed, freeze_channel=False)
        finally:
            m7.N_CH = saved
        if int(m7.N_CH) != saved:
            raise RuntimeError("N_CH was not restored")
        if tuple(self.channel_emb.shape) != (n_ch, m7.EMB):
            raise RuntimeError(f"channel embedding {tuple(self.channel_emb.shape)}")
        if float(self.channel_emb.detach().abs().sum()) != 0.0:
            raise RuntimeError("channel embedding did not start at zero")
        self.n_ch = n_ch

    def learn_glance(self, symbols, channel, prev, state=None, commit=True):
        use = self.state if state is None else state
        if int(use[-1].shape[0]) != self.n_ch:
            raise RuntimeError("trace state width does not match the channel table")
        saved = m7.N_CH
        m7.N_CH = self.n_ch
        try:
            return super().learn_glance(symbols, channel, prev, state=state, commit=commit)
        finally:
            m7.N_CH = saved

    def zero_state(self):
        saved = m7.N_CH
        m7.N_CH = self.n_ch
        try:
            state = m7.fresh_state(self.model)
        finally:
            m7.N_CH = saved
        if int(state[-1].shape[0]) != self.n_ch:
            raise RuntimeError("zero state has the wrong width")
        return state


class Life:
    """One agent, one wide bank, two glance reservoirs. Same store rules as M7."""

    def __init__(self, seed, world, names, index, corridor, n_ch, learnable):
        self.seed = int(seed)
        self.world = world
        self.names = list(names)
        self.index = dict(index)
        self.corridor = int(corridor)
        self.n_rooms = len(self.names)
        self.learnable = list(learnable)
        self.cortex = WideCortex(n_ch, model_seed=m7.MODEL_SEED)
        self.coin_rng = np.random.default_rng(40_000 + self.seed)
        self.exam = Reservoir(m7.R, np.random.default_rng(20_000 + self.seed), self.names)
        self.replay = Reservoir(m7.R, np.random.default_rng(50_000 + self.seed), self.names)
        self.sleep_rng = np.random.default_rng(30_000 + self.seed)
        self.cached_sum = 0.0
        self.bit_by_id = {}
        self.next_gid = 0
        self.exam_drawn = 0
        self.snapshots = {}
        self.learn_s = 0.0
        self.score_s = 0.0
        self.sleep_checks = []

    def _rescore(self):
        t0 = time.perf_counter()
        glances = self.exam.flat()
        if not glances:
            self.bit_by_id = {}
            self.cached_sum = 0.0
            self.score_s += time.perf_counter() - t0
            return 0.0
        bits = m7.score_glances(self.cortex.model, self.cortex.channel_emb, glances)
        self.bit_by_id = {g.gid: float(bits[i]) for i, g in enumerate(glances)}
        self.cached_sum = float(bits.sum())
        self.score_s += time.perf_counter() - t0
        return self.cached_sum

    def _score_one(self, glance):
        t0 = time.perf_counter()
        bits = m7.score_glances(self.cortex.model, self.cortex.channel_emb, [glance])
        self.score_s += time.perf_counter() - t0
        return float(bits[0])

    def _insert_exam(self, name, glance):
        action, old = self.exam.offer(name, glance)
        if action == "reject":
            return
        if action == "replace":
            self.cached_sum -= self.bit_by_id.pop(old.gid)
        value = self._score_one(glance)
        self.bit_by_id[glance.gid] = value
        self.cached_sum += value

    def _learn(self, symbols, channel, prev, state=None, commit=True):
        t0 = time.perf_counter()
        bits = self.cortex.learn_glance(symbols, channel, prev, state=state, commit=commit)
        self.learn_s += time.perf_counter() - t0
        return bits

    def room_decision(self, symbols, name):
        prev = self.cortex.prev_symbol
        self._learn(symbols, self.index[name], prev, commit=True)
        glance = Glance(self.next_gid, self.index[name], symbols, prev, self.n_rooms)
        self.next_gid += 1
        if int(self.coin_rng.integers(2)) == 0:
            self._insert_exam(name, glance)
        else:
            self.replay.offer(name, glance)
        if self.exam.ids & self.replay.ids:
            raise RuntimeError("a glance is in both stores")

    def corridor_decision(self, symbols):
        prev = self.cortex.prev_symbol
        self._learn(symbols, self.corridor, prev, commit=True)

    def sleep_decision(self):
        """One replay glance from a zero state. The live stream is not written."""
        glance = self.replay.draw(self.sleep_rng)
        if glance is None:
            return False
        if glance.gid in self.exam.ids:
            self.exam_drawn += 1
        self._learn(
            glance.symbols, glance.channel, glance.prev,
            state=self.cortex.zero_state(), commit=False,
        )
        self._rescore()
        return True

    def begin_sleep_visit(self):
        return self.cortex.clone_live(), self.cortex.clone_params()

    def end_sleep_visit(self, snap):
        live, params = snap
        intact = self.cortex.live_matches(live)
        delta = self.cortex.param_delta(params)
        self.sleep_checks.append({"intact": bool(intact), "param_delta": float(delta)})
        if not intact:
            raise RuntimeError("sleep changed the live state or traces")
        return float(delta)

    def snapshot(self, name):
        glances = list(self.exam.slots[name])
        if not glances:
            self.snapshots[name] = {"n": 0, "bits": None, "bps": None, "ids": []}
            return
        t0 = time.perf_counter()
        total = float(m7.score_glances(self.cortex.model, self.cortex.channel_emb, glances).sum())
        self.score_s += time.perf_counter() - t0
        n = len(glances)
        self.snapshots[name] = {
            "n": n,
            "bits": total,
            "bps": total / (n * m7.CHUNK),
            "ids": [g.gid for g in glances],
        }

    def forgetting(self):
        out = {}
        for name in self.learnable:
            snap = self.snapshots.get(name)
            current = list(self.exam.slots[name])
            if snap is None or snap["n"] == 0:
                out[name] = {
                    "visited": snap is not None,
                    "n": 0 if snap is None else int(snap["n"]),
                    "after_bps": None,
                    "end_bps": None,
                    "delta_bps": None,
                }
                continue
            if [g.gid for g in current] != snap["ids"]:
                raise RuntimeError(f"{name} exam set changed after its last visit")
            end_bits = float(m7.score_glances(
                self.cortex.model, self.cortex.channel_emb, current,
            ).sum())
            end_bps = end_bits / (snap["n"] * m7.CHUNK)
            out[name] = {
                "visited": True,
                "n": int(snap["n"]),
                "after_bps": float(snap["bps"]),
                "end_bps": float(end_bps),
                "delta_bps": float(end_bps - snap["bps"]),
            }
        return out


def evaluate_w14(cortex, world, index):
    """Frozen bank, zero state, fresh EVAL continuation, that room's channel row."""
    wc = world.copy()
    names = list(LEARN14)
    sequences = [wc.continuation(name, curio.EVAL) for name in names]
    symbols = torch.from_numpy(np.stack(sequences))
    prev = torch.full((len(names),), -1, dtype=torch.long)
    channels = torch.tensor([index[name] for name in names], dtype=torch.long)
    totals = m7.score_sequences(cortex.model, cortex.channel_emb, symbols, prev, channels)
    held = {name: float(totals[i]) / curio.EVAL for i, name in enumerate(names)}
    w14 = float(np.mean([held[name] for name in names]))
    wa = float(np.mean([held[name] for name in LEARN_A]))
    wb = float(np.mean([held[name] for name in LEARN_B]))
    return w14, wa, wb, held


def shares_of(channels, names):
    n = len(channels)
    counts = {name: 0 for name in names}
    corridor = 0
    sleep = 0
    for name in channels:
        if name == "corridor":
            corridor += 1
        elif name == "sleep":
            sleep += 1
        else:
            counts[name] += 1
    share = {name: counts[name] / n for name in names}
    return share, corridor / n, sleep / n


def _close_visit(life, pending):
    if pending is not None:
        life.snapshot(pending)
    return None


def _sleep_visit(life, seed, policy, channels, length, t, L, check_sleep, checked):
    if length <= 0:
        raise RuntimeError("empty sleep visit")
    start = t
    snap = life.begin_sleep_visit() if (check_sleep and not checked) else None
    n = 0
    learned = 0
    while n < length and t < L:
        did_learn = life.sleep_decision()
        learned += int(did_learn)
        channels.append("sleep")
        n += 1
        t += 1
        if t % 256 == 0:
            print(f"s{seed} {policy} t={t}/{L} sleep", flush=True)
    if snap is not None:
        delta = life.end_sleep_visit(snap)
        checked = True
        if learned and delta <= 0.0:
            raise RuntimeError("sleep visit learned and left the parameters unchanged")
    if n != length and t < L:
        raise RuntimeError("sleep visit ended early")
    return start, n, learned, t, checked


def run_life(seed, policy, L, restricted=False, score=True, check_sleep=False):
    """One fixed policy with the fixed sleep dose, taken only between visits."""
    L = int(L)
    n_sleep = sleep_decisions(L)
    plan = m7.sleep_plan(L, n_sleep)
    if any(length != m7.DWELL for _, length in plan):
        raise RuntimeError("a fixed sleep visit is not 8 decisions")
    if restricted:
        if policy != "rand_dwell_s":
            raise ValueError("the W3 restriction is rand_dwell_s only")
        world = World3(seed)
        names = list(curio.CH)
        index = {name: m7.ROOM_INDEX[name] for name in names}
        if [index[name] for name in names] != list(range(len(names))):
            raise RuntimeError("W3 channel order is not curio.CH")
        n_ring = 9
        corridor = m7.CORRIDOR
        n_ch = N_CH
        learnable = list(curio.LEARN)
    else:
        if policy not in POLICIES:
            raise ValueError(policy)
        world = World4(seed)
        names = list(IDENTITIES)
        index = {name: i for i, name in enumerate(names)}
        n_ring = N_RING
        corridor = N_RING
        n_ch = N_CH
        learnable = list(LEARN14)
    t0 = time.perf_counter()
    life = Life(seed, world, names, index, corridor, n_ch, learnable)
    rng = np.random.default_rng(1_000 + int(seed))
    if restricted:
        state = {"kind": "rand_dwell", "room": 0, "plan": []}
    else:
        state = {"kind": policy, "room": 0, "plan": []}
        targets = targets_for(world.rooms, policy)
    channels = []
    pending = None
    si = 0
    t = 0
    checked = False
    while t < L:
        if not state["plan"] and sleep_due(t, L, plan, si, n_ring):
            pending = _close_visit(life, pending)
            _start, _n, _learned, t, checked = _sleep_visit(
                life, seed, policy, channels, plan[si][1], t, L, check_sleep, checked,
            )
            si += 1
            continue
        if restricted:
            action = w3_next_action(state, life.world.rooms, rng)
        else:
            action = next_w4(state, n_ring, targets, rng)
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
            print(f"s{seed} {policy} t={t}/{L}", flush=True)
        if not state["plan"]:
            pending = _close_visit(life, pending)
    if pending is not None:
        life.snapshot(pending)
    if len(channels) != L:
        raise RuntimeError(f"{policy} ran {len(channels)} decisions")
    if si != len(plan):
        raise RuntimeError(f"{policy} took {si} of {len(plan)} planned sleep visits")
    if life.exam_drawn != 0:
        raise RuntimeError(f"sleep drew {life.exam_drawn} exam glances")
    eval_s = 0.0
    if score and restricted:
        e0 = time.perf_counter()
        W, W9, held9 = m7.evaluate_bank(life.cortex, life.world, seed)
        forgetting = life.forgetting()
        eval_s = time.perf_counter() - e0
        held = {name: float(held9[name]) for name in curio.LEARN}
        w14, wa, wb = float(W), None, None
        finite = bool(math.isfinite(W) and math.isfinite(W9) and all(math.isfinite(v) for v in held.values()))
    elif score:
        e0 = time.perf_counter()
        w14, wa, wb, held = evaluate_w14(life.cortex, life.world, life.index)
        forgetting = life.forgetting()
        eval_s = time.perf_counter() - e0
        W9 = None
        finite = bool(
            math.isfinite(w14) and math.isfinite(wa) and math.isfinite(wb)
            and all(math.isfinite(v) for v in held.values())
        )
    else:
        w14 = wa = wb = W9 = None
        held = {}
        forgetting = {}
        finite = False
    share, corridor_share, sleep_share = shares_of(channels, life.names)
    if abs((sum(share.values()) + corridor_share + sleep_share) - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    if sum(name == "sleep" for name in channels) != n_sleep:
        raise RuntimeError("sleep decisions do not match the dose")
    wall = time.perf_counter() - t0
    checks = life.sleep_checks
    return {
        "seed": int(seed),
        "policy": policy,
        "L": int(L),
        "restricted": bool(restricted),
        "W14": None if w14 is None else float(w14),
        "Wa": None if wa is None else float(wa),
        "Wb": None if wb is None else float(wb),
        "W9": None if W9 is None else float(W9),
        "held": {name: float(held[name]) for name in held},
        "forgetting": forgetting,
        "share": {name: float(share[name]) for name in life.names},
        "corridor_share": float(corridor_share),
        "sleep_share": float(sleep_share),
        "sleep_decisions": int(n_sleep),
        "room_order": list(life.world.rooms),
        "exam_sizes": {name: int(v) for name, v in life.exam.sizes().items()},
        "replay_sizes": {name: int(v) for name, v in life.replay.sizes().items()},
        "exam_glances_drawn": int(life.exam_drawn),
        "overlap": int(len(life.exam.ids & life.replay.ids)),
        "sleep_state_intact": (None if not checks else all(item["intact"] for item in checks)),
        "wall_s": float(wall),
        "eval_s": float(eval_s),
        "learn_s": float(life.learn_s),
        "score_s": float(life.score_s),
        "finite": bool(finite),
        "n_ch": int(n_ch),
    }


def _job(args):
    seed, policy, L = args
    _threads()
    print(f"s{seed} {policy} L={L} start dose {sleep_decisions(L)}", flush=True)
    row = run_life(seed, policy, L, restricted=False, score=True, check_sleep=True)
    print(
        f's{seed} {policy} L={L} W14 {row["W14"]:.4f} Wa {row["Wa"]:.4f} Wb {row["Wb"]:.4f} '
        f'sleep {row["sleep_share"]:.4f} corridor {row["corridor_share"]:.3f} '
        f'wall {row["wall_s"]:.1f}s learn {row["learn_s"]:.1f}s score {row["score_s"]:.1f}s',
        flush=True,
    )
    return row


def mean_std(values):
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return None, None
    return float(data.mean()), float(data.std(ddof=0))


def _pick(runs, policy, seed, L):
    hit = [
        row for row in runs
        if row["policy"] == policy and row["seed"] == seed and row["L"] == L and not row.get("restricted")
    ]
    if len(hit) != 1:
        raise RuntimeError(f"expected one {policy} L={L} seed {seed}, got {len(hit)}")
    return hit[0]


def summarize(runs):
    runs = [row for row in runs if not row.get("restricted")]
    keys = []
    for policy in POLICIES:
        keys.append((policy, L_SHORT))
    keys.append((LONG_POLICY, L_LONG))
    by = {}
    for policy, L in keys:
        rows = [row for row in runs if row["policy"] == policy and row["L"] == L]
        rows = sorted(rows, key=lambda row: row["seed"])
        if len(rows) != len(SEEDS):
            raise RuntimeError(f"{policy} L={L} has {len(rows)} seeds")
        w_mu, w_sd = mean_std([row["W14"] for row in rows])
        wa_mu, wa_sd = mean_std([row["Wa"] for row in rows])
        wb_mu, wb_sd = mean_std([row["Wb"] for row in rows])
        held_mu, held_sd = {}, {}
        for name in LEARN14:
            mu, sd = mean_std([row["held"][name] for row in rows])
            held_mu[name] = mu
            held_sd[name] = sd
        share_mu = {}
        for name in IDENTITIES:
            mu, sd = mean_std([row["share"][name] for row in rows])
            share_mu[name] = mu
        c_mu, c_sd = mean_std([row["corridor_share"] for row in rows])
        s_mu, s_sd = mean_std([row["sleep_share"] for row in rows])
        deltas = []
        per_room = {name: [] for name in LEARN14}
        for row in rows:
            for name in LEARN14:
                item = row["forgetting"][name]
                if item["delta_bps"] is None:
                    continue
                deltas.append(item["delta_bps"])
                per_room[name].append(item)
        d_mu, d_sd = mean_std(deltas)
        forget_room = {}
        for name in LEARN14:
            items = per_room[name]
            mu, sd = mean_std([item["delta_bps"] for item in items])
            after_mu, _after_sd = mean_std([item["after_bps"] for item in items])
            end_mu, _end_sd = mean_std([item["end_bps"] for item in items])
            forget_room[name] = {
                "n": len(items),
                "delta_bps_mean": mu,
                "delta_bps_std_ddof0": sd,
                "after_bps_mean": after_mu,
                "end_bps_mean": end_mu,
            }
        by[f"{policy}/L{L}"] = {
            "policy": policy,
            "L": L,
            "n": len(rows),
            "W14_mean": w_mu,
            "W14_std_ddof0": w_sd,
            "Wa_mean": wa_mu,
            "Wa_std_ddof0": wa_sd,
            "Wb_mean": wb_mu,
            "Wb_std_ddof0": wb_sd,
            "held_mean": held_mu,
            "held_std_ddof0": held_sd,
            "share_mean": share_mu,
            "corridor_mean": c_mu,
            "corridor_std_ddof0": c_sd,
            "sleep_mean": s_mu,
            "sleep_std_ddof0": s_sd,
            "forgetting_mean": d_mu,
            "forgetting_std_ddof0": d_sd,
            "forgetting_n": len(deltas),
            "forgetting_room": forget_room,
        }
    gaps = []
    for seed in SEEDS:
        rand = _pick(runs, "rand_dwell_s", seed, L_SHORT)
        broad = _pick(runs, "oracle14_s", seed, L_SHORT)
        depth = _pick(runs, "oracle7a_s", seed, L_SHORT)
        long = _pick(runs, "oracle14_s", seed, L_LONG)
        gaps.append({
            "seed": int(seed),
            "oracle14_s_minus_rand_dwell_s": float(broad["W14"] - rand["W14"]),
            "oracle7a_s_minus_rand_dwell_s": float(depth["W14"] - rand["W14"]),
            "oracle14_s_L2048_minus_L8192": float(broad["W14"] - long["W14"]),
            "W14": {
                "rand_dwell_s": rand["W14"],
                "oracle14_s": broad["W14"],
                "oracle7a_s": depth["W14"],
                "oracle14_s_L8192": long["W14"],
            },
        })
    gap_summary = {}
    for key in (
        "oracle14_s_minus_rand_dwell_s",
        "oracle7a_s_minus_rand_dwell_s",
        "oracle14_s_L2048_minus_L8192",
    ):
        mu, sd = mean_std([row[key] for row in gaps])
        gap_summary[key] = {
            "mean": mu,
            "std_ddof0": sd,
            "left_lower": int(sum(row[key] < 0.0 for row in gaps)),
            "n": len(gaps),
        }
    broad = by[f"{LONG_POLICY}/L{L_SHORT}"]
    depth = by[f"oracle7a_s/L{L_SHORT}"]
    if broad["W14_mean"] < depth["W14_mean"]:
        winner, kind = "oracle14_s", "breadth"
    elif depth["W14_mean"] < broad["W14_mean"]:
        winner, kind = "oracle7a_s", "depth"
    else:
        winner, kind = "tie", "tie"
    return {
        "by_policy": by,
        "gaps": gaps,
        "gap_summary": gap_summary,
        "oracle_comparison": {
            "L": L_SHORT,
            "lower_mean_W14": winner,
            "kind": kind,
            "oracle14_s_W14_mean": broad["W14_mean"],
            "oracle7a_s_W14_mean": depth["W14_mean"],
            "oracle14_s_Wa_mean": broad["Wa_mean"],
            "oracle14_s_Wb_mean": broad["Wb_mean"],
            "oracle7a_s_Wa_mean": depth["Wa_mean"],
            "oracle7a_s_Wb_mean": depth["Wb_mean"],
            "seeds_oracle14_lower": int(sum(
                row["oracle14_s_minus_rand_dwell_s"] < row["oracle7a_s_minus_rand_dwell_s"]
                for row in gaps
            )),
            "seeds_oracle7a_lower": int(sum(
                row["oracle7a_s_minus_rand_dwell_s"] < row["oracle14_s_minus_rand_dwell_s"]
                for row in gaps
            )),
        },
    }


def _replay_bits(cortex, glances):
    prev = -1
    out = []
    for symbols, channel in glances:
        bits = np.asarray(cortex.learn_glance(symbols, channel, prev, commit=True), dtype=np.float64)
        out.append(bits)
        prev = int(symbols[-1])
    return out


def _time_full_rescore(n_glances, n_rooms):
    cortex = WideCortex(N_CH)
    rng = np.random.default_rng(2)
    glances = []
    for i in range(int(n_glances)):
        glances.append(Glance(
            i, int(i % n_rooms),
            rng.integers(0, len(ALPHA), size=m7.CHUNK, dtype=np.int64),
            int(rng.integers(0, len(ALPHA))),
            n_rooms,
        ))
    m7.score_glances(cortex.model, cortex.channel_emb, glances)
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        m7.score_glances(cortex.model, cortex.channel_emb, glances)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def smoke(out_dir=None):
    """VPS checks. A failed check does not stage kernels."""
    _threads()
    out_dir = Path(out_dir or (ROOT / "results" / "w4"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    t0 = time.perf_counter()
    payload = {
        "torch": torch.__version__,
        "model_seed": m7.MODEL_SEED,
        "n_ch": N_CH,
        "ok": False,
    }
    try:
        import m7b_w3 as m7b

        log(f"torch={torch.__version__} threads=1 model_seed={m7.MODEL_SEED} n_ch={N_CH}")
        if int(m7.N_CH) != 10:
            raise RuntimeError("M7 N_CH changed before the checks")
        for L in (64, 512, L_SHORT, L_LONG):
            dose = sleep_decisions(L)
            ref = m7b.sleep_decisions(L)
            if dose != ref:
                raise RuntimeError(f"sleep dose L={L} {dose} != m7b {ref}")
            log(f"dose L{L} {dose}/{L} share {dose / L:.6f}")
        plan = m7.sleep_plan(L_SHORT, sleep_decisions(L_SHORT))
        for t in range(0, L_SHORT, 17):
            for si in range(len(plan) + 1):
                if sleep_due(t, L_SHORT, plan, si, 9) != m7.sleep_due(t, L_SHORT, plan, si):
                    raise RuntimeError(f"sleep_due diverged at t={t} si={si}")
        log("sleep_due on a 9-ring matches m7.sleep_due")
        for src in range(N_RING):
            for dst in range(N_RING):
                path = shortest(src, dst, N_RING)
                room = src
                for action in path:
                    room = (room - 1) % N_RING if action == "left" else (room + 1) % N_RING
                dist = min((dst - src) % N_RING, (src - dst) % N_RING)
                if room != dst or len(path) != dist:
                    raise RuntimeError(f"shortest path failed {src}->{dst}")
                if dist and (dst - src) % N_RING == (src - dst) % N_RING and path[0] != "left":
                    raise RuntimeError("an equal split did not go left")
        log("ring shortest paths, including the distance-9 tie, go left on a tie")

        log("smoke check 1: 19-row bank matches M7 on 64 glances, channel ids < 10")
        rng = np.random.default_rng(0)
        glances = []
        for _ in range(64):
            symbols = rng.integers(0, len(ALPHA), size=m7.CHUNK, dtype=np.int64)
            channel = int(rng.integers(0, 10))
            glances.append((symbols, channel))
        base = m7.BankCortex(model_seed=m7.MODEL_SEED, freeze_channel=False)
        wide = WideCortex(N_CH, model_seed=m7.MODEL_SEED)
        base_bits = _replay_bits(base, glances)
        wide_bits = _replay_bits(wide, glances)
        bit_abs = max(float(np.max(np.abs(a - b))) for a, b in zip(base_bits, wide_bits))
        with torch.no_grad():
            param_abs = max(
                float((a - b).abs().max())
                for a, b in zip(base.model.parameters(), wide.model.parameters())
            )
            row_abs = float((base.channel_emb.detach() - wide.channel_emb.detach()[:m7.N_CH]).abs().max())
        log(f"  bit abs {bit_abs:.3e} param abs {param_abs:.3e} channel-row abs {row_abs:.3e} limit {BITS_LIMIT:.1e}")
        if bit_abs > BITS_LIMIT:
            raise RuntimeError(f"19-row bank bits differ by {bit_abs:.3e}")
        if int(m7.N_CH) != 10:
            raise RuntimeError("N_CH was left widened")
        payload["bit_check"] = {"abs": bit_abs, "param_abs": param_abs, "channel_row_abs": row_abs}

        log("smoke check 2: instances differ, seed 0")
        shown = World4(0)
        texts = {}
        for name in ("M7a", "M7b", "Ga", "Gb"):
            symbols = list(shown.glance_room(name))
            texts[name] = "".join(ALPHA[i] for i in symbols)
            log(f"  {name} {texts[name]!r}")
        if texts["M7a"] == texts["M7b"] or texts["Ga"] == texts["Gb"]:
            raise RuntimeError("the two instances produced the same first glance")
        fresh = World4(0)
        first = list(fresh.glance_room("C1a"))
        copied = fresh.copy()
        second = list(fresh.glance_room("C1a"))
        if list(copied.glance_room("C1a")) != second:
            raise RuntimeError("C1a did not resume on a copy")
        if list(World4(0).glance_room("C1a")) != first:
            raise RuntimeError("C1a did not restart from the world seed")
        noise = World4(0)
        n1 = list(noise.glance_room("N1"))
        n2 = list(noise.glance_room("N2"))
        if n1 == n2:
            raise RuntimeError("N1 and N2 drew the same first glance")
        noise_copy = noise.copy()
        n1_next = list(noise.glance_room("N1"))
        if list(noise_copy.glance_room("N1")) != n1_next:
            raise RuntimeError("N1 did not resume")
        parked = World4(1)
        parked_name = parked.rooms[0]
        peek = list(parked.copy().glance_room(parked_name))
        room, channel, symbols = parked.step("left", None)
        if channel != "corridor" or room != N_RING - 1 or symbols != [ALPHA.index(" ")] * curio.B:
            raise RuntimeError("a left move did not show the corridor")
        if list(parked.glance_room(parked_name)) != peek:
            raise RuntimeError("a move consumed the room it left")
        log("  streams resume, noise rngs differ, a move shows the corridor")

        log("ring orders")
        orders = {}
        for seed in (0, 1, 2):
            world = World4(seed)
            orders[str(seed)] = list(world.rooms)
            log(f"  seed {seed} perm {world.perm} rooms {world.rooms}")
        payload["room_orders"] = orders

        log("smoke check 3: W4 restricted to W3's 9 rooms, rand_dwell_s seed 36 L=64")
        ref = m7b.run_controller(36, "rand_dwell_s", 64, score=True, check_sleep=False)
        live = run_life(36, "rand_dwell_s", 64, restricted=True, score=True, check_sleep=False)
        dW = abs(live["W14"] - ref["W"])
        dW9 = abs(live["W9"] - ref["W9"])
        log(
            f"  m7b W {ref['W']:.8f} W9 {ref['W9']:.8f} "
            f"live W {live['W14']:.8f} W9 {live['W9']:.8f} "
            f"abs_dW {dW:.3e} abs_dW9 {dW9:.3e} limit {BITS_LIMIT:.1e}"
        )
        if dW > BITS_LIMIT or dW9 > BITS_LIMIT:
            raise RuntimeError(f"restricted rand_dwell_s differs by dW {dW:.3e} dW9 {dW9:.3e}")
        if live["sleep_decisions"] != 16 or ref["sleep_decisions"] != 16:
            raise RuntimeError("L=64 sleep dose is not 16")
        payload["restricted"] = {
            "seed": 36,
            "L": 64,
            "W_m7b": float(ref["W"]),
            "W9_m7b": float(ref["W9"]),
            "W_live": float(live["W14"]),
            "W9_live": float(live["W9"]),
            "abs_dW": dW,
            "abs_dW9": dW9,
        }

        log("smoke check 4: three policies, seed 0, L=64, finite W14")
        for policy in POLICIES:
            row = run_life(0, policy, 64, score=True, check_sleep=True)
            log(
                f"  {policy} W14 {row['W14']:.6f} Wa {row['Wa']:.6f} Wb {row['Wb']:.6f} "
                f"finite {row['finite']} sleep {row['sleep_share']:.4f} "
                f"decisions {row['sleep_decisions']} corridor {row['corridor_share']:.4f} "
                f"wall {row['wall_s']:.2f}s"
            )
            if not row["finite"]:
                raise RuntimeError(f"{policy} W14 is not finite")
            if row["sleep_decisions"] != 16:
                raise RuntimeError(f"{policy} sleep decisions {row['sleep_decisions']}")
            if policy == "oracle7a_s":
                for name, share in row["share"].items():
                    parsed = split_learnable(name)
                    if (parsed is None or parsed[1] != "a") and share != 0.0:
                        raise RuntimeError(f"oracle7a_s stayed in {name}")
            if policy == "oracle14_s":
                for name in NOISE + [DARK]:
                    if row["share"][name] != 0.0:
                        raise RuntimeError(f"oracle14_s stayed in {name}")

        log("smoke check 5: time rand_dwell_s seed 0 L=512")
        timed = run_life(0, "rand_dwell_s", 512, score=True, check_sleep=True)
        if not timed["finite"]:
            raise RuntimeError("timed life W14 is not finite")
        full_s = _time_full_rescore(N_RING * m7.R, N_RING)
        learn_per = timed["learn_s"] / 512.0
        linear = {}
        pessimistic = {}
        for L in (L_SHORT, L_LONG):
            linear[str(L)] = (timed["learn_s"] + timed["score_s"]) * (L / 512.0) + timed["eval_s"]
            pessimistic[str(L)] = learn_per * L + full_s * sleep_decisions(L) + timed["eval_s"]
        # Four jobs, four workers: the kernel wall is the longer life.
        linear_wall = linear[str(L_LONG)]
        pessimistic_wall = pessimistic[str(L_LONG)]
        guarded = (timed["learn_s"] + timed["score_s"]) * (L_LONG / 512.0) * 2.0 + timed["eval_s"]
        split = False
        if pessimistic_wall <= KERNEL_WALL_LIMIT_S:
            log("  upper bound within 90 min; one kernel per seed runs every life")
        elif guarded <= KERNEL_WALL_LIMIT_S:
            log("  full-store bound exceeds 90 min; linear x2 still fits; one kernel per seed")
        else:
            split = True
            if guarded > KERNEL_WALL_LIMIT_S:
                raise RuntimeError(
                    f"L=8192 estimate {guarded:.0f}s exceeds {KERNEL_WALL_LIMIT_S:.0f}s on its own"
                )
            log("  splitting the L=8192 life into its own kernel per seed")
        log(
            f"  rand_dwell_s L=512 W14 {timed['W14']:.6f} wall {timed['wall_s']:.2f}s "
            f"learn {timed['learn_s']:.2f}s score {timed['score_s']:.2f}s eval {timed['eval_s']:.2f}s "
            f"sleep {timed['sleep_decisions']}"
        )
        log(
            f"  learn_per_decision {learn_per:.4f}s full_rescore_{N_RING * m7.R} {full_s:.4f}s"
        )
        log(
            f"  linear L2048 {linear['2048']:.1f}s L8192 {linear['8192']:.1f}s "
            f"pessimistic L2048 {pessimistic['2048']:.1f}s L8192 {pessimistic['8192']:.1f}s "
            f"guarded L8192 {guarded:.1f}s kernel {max(pessimistic_wall, linear_wall):.1f}s "
            f"limit {KERNEL_WALL_LIMIT_S:.0f}s split {split}"
        )
        payload["timed_512"] = {
            "seed": 0,
            "policy": "rand_dwell_s",
            "L": 512,
            "W14": timed["W14"],
            "Wa": timed["Wa"],
            "Wb": timed["Wb"],
            "wall_s": timed["wall_s"],
            "learn_s": timed["learn_s"],
            "score_s": timed["score_s"],
            "eval_s": timed["eval_s"],
            "sleep_decisions": timed["sleep_decisions"],
        }
        payload["estimate"] = {
            "learn_per_decision_s": learn_per,
            "full_rescore_s": full_s,
            "full_rescore_glances": N_RING * m7.R,
            "linear_s": linear,
            "pessimistic_s": pessimistic,
            "guarded_L8192_s": guarded,
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


def _jobs_for(split):
    short = [(policy, L_SHORT) for policy in POLICIES]
    if split:
        return short, [(LONG_POLICY, L_LONG)]
    return short + [(LONG_POLICY, L_LONG)], None


def _write_kernel(directory, seed, jobs):
    directory.mkdir(parents=True, exist_ok=True)
    lives = sorted({int(L) for _policy, L in jobs})
    if lives == [L_SHORT, L_LONG]:
        slug = f"w4-s{seed}"
    elif lives == [L_SHORT]:
        slug = f"w4-s{seed}"
    elif lives == [L_LONG]:
        slug = f"w4-s{seed}-l8192"
    else:
        raise RuntimeError(f"unexpected lives {lives}")
    kernel_id = f"ezzzzzx/{slug}"
    meta = {
        "id": kernel_id,
        "title": slug,
        "code_file": "w4_kernel.py",
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
    jobs_src = ", ".join(f'("{policy}", {L})' for policy, L in jobs)
    text = f'''"""W4 seed {seed}. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "0"
os.environ["CURIO_LR"] = "0.004"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if os.environ.get("_W4_KERNEL") != "1":
    os.environ["_W4_KERNEL"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    hits = sorted(Path("/kaggle/input").rglob("w4_headroom.py"))
    if not hits:
        raise SystemExit("w4_headroom.py not found under /kaggle/input")
    code = str(hits[0].parent)
    prev = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print("code_dir", code, flush=True)
    import w4_headroom
    w4_headroom.kernel_main(
        seed={seed},
        out="/kaggle/working",
        workers=4,
        kernel_id="{kernel_id}",
        jobs=({jobs_src},),
    )


if __name__ == "__main__":
    main()
'''
    (directory / "w4_kernel.py").write_text(text)
    return kernel_id


def stage(smoke_payload, kaggle_dir=None):
    """Write the private dataset folder and the kernel folders. Does not push."""
    if not smoke_payload.get("ok"):
        raise SystemExit("smoke did not pass; kernels not staged")
    split = bool(smoke_payload["estimate"]["split"])
    kaggle_dir = Path(kaggle_dir or (ROOT / "kaggle" / "w4"))
    if kaggle_dir.exists():
        shutil.rmtree(kaggle_dir)
    code = kaggle_dir / "code"
    code.mkdir(parents=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
    (code / "dataset-metadata.json").write_text(json.dumps({
        "title": "pdp code w4",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    ids = []
    for seed in SEEDS:
        together, long_only = _jobs_for(split)
        ids.append(_write_kernel(kaggle_dir / f"s{seed}", seed, together))
        if long_only is not None:
            ids.append(_write_kernel(kaggle_dir / f"s{seed}-l8192", seed, long_only))
    print(f"staged {len(ids)} kernels split {split}", flush=True)
    for kernel_id in ids:
        print(kernel_id, flush=True)
    return ids


def kernel_main(seed, out, workers=4, kernel_id=None, jobs=None):
    if __name__ != "w4_headroom":
        raise SystemExit("import w4_headroom and call kernel_main so the pool can spawn workers")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 0-5")
    if int(workers) < 1 or int(workers) > 4:
        raise SystemExit("kernel pool is at most 4")
    if jobs is None:
        jobs = [(policy, L_SHORT) for policy in POLICIES] + [(LONG_POLICY, L_LONG)]
    jobs = tuple((str(policy), int(L)) for policy, L in jobs)
    allowed = {(policy, L_SHORT) for policy in POLICIES} | {(LONG_POLICY, L_LONG)}
    if any(job not in allowed for job in jobs):
        raise SystemExit(f"jobs {jobs} are not a W4 kernel")
    _threads()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    order = sorted(jobs, key=lambda item: (-item[1], item[0]))
    print(
        f"kernel {kernel_id} seed {seed} workers {workers} jobs {len(order)}",
        flush=True,
    )
    t0 = time.perf_counter()
    ctx = get_context("spawn")
    with ctx.Pool(int(workers)) as pool:
        asyncs = [pool.apply_async(_job, ((seed, policy, L),)) for policy, L in order]
        rows = [item.get() for item in asyncs]
    wall = time.perf_counter() - t0
    if len(rows) != len(order):
        raise SystemExit(f"expected {len(order)} rows, got {len(rows)}")
    for row in rows:
        _check_row(row, seed, row["policy"], row["L"])
    by_key = {(row["policy"], row["L"]): row for row in rows}
    if len(by_key) != len(order):
        raise SystemExit("duplicate policy rows")
    ordered = [by_key[job] for job in jobs]
    payload = {
        "kernel_id": kernel_id,
        "seed": seed,
        "jobs": [{"policy": policy, "L": L} for policy, L in jobs],
        "workers": int(workers),
        "wall_s": wall,
        "torch": torch.__version__,
        "model_seed": m7.MODEL_SEED,
        "n_ch": N_CH,
        "sleep_share_target": SLEEP_SHARE,
        "room_order": ordered[0]["room_order"],
        "runs": ordered,
    }
    path = out / f"w4_s{seed}.json"
    m7.dump(path, payload)
    print(f"wrote {path} wall {wall:.1f}s", flush=True)
    return payload


def _check_row(row, seed, policy, L):
    if row["seed"] != int(seed) or row["policy"] != policy or row["L"] != int(L):
        raise RuntimeError("row does not match its job")
    if row.get("restricted"):
        raise RuntimeError("a scored row is the W3 restriction")
    if not row["finite"]:
        raise RuntimeError(f"{policy} seed {seed} L={L} is not finite")
    if row["sleep_decisions"] != sleep_decisions(L):
        raise RuntimeError("sleep dose drifted")
    if row["overlap"] != 0 or row["exam_glances_drawn"] != 0:
        raise RuntimeError("stores overlapped")
    if row["n_ch"] != N_CH:
        raise RuntimeError("channel width drifted")
    if abs(row["sleep_share"] - (sleep_decisions(L) / float(L))) > 1e-12:
        raise RuntimeError("sleep share drifted")
    names = set(row["share"])
    if names != set(IDENTITIES):
        raise RuntimeError("share keys are not the 18 rooms")
    total = sum(row["share"].values()) + row["corridor_share"] + row["sleep_share"]
    if abs(total - 1.0) > 1e-9:
        raise RuntimeError("shares do not sum to 1")
    if set(row["held"]) != set(LEARN14):
        raise RuntimeError("held-out keys are not the 14 learnable rooms")
    if policy == "oracle7a_s":
        for name, share in row["share"].items():
            parsed = split_learnable(name)
            if (parsed is None or parsed[1] != "a") and share != 0.0:
                raise RuntimeError(f"oracle7a_s stayed in {name}")
    if policy == "oracle14_s":
        for name in NOISE + [DARK]:
            if row["share"][name] != 0.0:
                raise RuntimeError(f"oracle14_s stayed in {name}")


def _fmt(value, digits=6):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _pm(mu, sd, digits=6):
    if mu is None:
        return "—"
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"


def _policy_keys():
    return [(policy, L_SHORT) for policy in POLICIES] + [(LONG_POLICY, L_LONG)]


def m7c_reference():
    path = ROOT / "results" / "m7cw3" / "results.json"
    data = json.loads(path.read_text())
    out = {"source": "results/m7cw3/results.json", "seeds": "42-53"}
    for label in ("2048", "4096"):
        block = data["summary"][label]
        out[label] = {}
        for name in ("rand_dwell_s", "oracle7_s"):
            row = block["by_controller"][name]
            out[label][name] = {
                "W_mean": row["W_mean"],
                "W_std_ddof0": row["W_std_ddof0"],
            }
        gap = block["gap_summary"]["oracle7_s_minus_rand_dwell_s"]
        out[label]["oracle7_s_minus_rand_dwell_s"] = {
            "mean": gap["mean"],
            "std_ddof0": gap["std_ddof0"],
            "left_lower": gap["left_lower"],
            "n": gap["n"],
        }
    return out


def render_markdown(result):
    summary = result["summary"]
    seeds = list(result["seeds"])
    lines = []
    lines.append("# W4 headroom")
    lines.append("")
    lines.append(
        "Fixed policies on an 18-room ring. W14 is bits/symbol of the frozen bank on a fresh "
        "continuation of the 14 learnable rooms, from a zero state, with that room's channel "
        "embedding. Wa is the mean over instance a and Wb over instance b. Lower is better. "
        "Forgetting is end-of-life exam bits/symbol minus the exam bits/symbol just after the "
        "last visit to that room. Positive forgetting is a worse code."
    )
    lines.append("")
    lines.append(
        "Every policy sleeps for the fixed share 0.20, in visits of 8, placed as in M7b. "
        f"At L = {L_SHORT} that is {sleep_decisions(L_SHORT)} decisions, share "
        f"{sleep_decisions(L_SHORT) / L_SHORT:.6f}. At L = {L_LONG} that is "
        f"{sleep_decisions(L_LONG)} decisions, share {sleep_decisions(L_LONG) / L_LONG:.6f}."
    )
    lines.append("")
    comp = summary["oracle_comparison"]
    lines.append("## Oracle comparison")
    lines.append("")
    lines.append(
        "Breadth is oracle14_s. Depth is oracle7a_s. The lower mean W14 at L = 2048 is "
        f'{comp["lower_mean_W14"]} ({comp["kind"]}).'
    )
    lines.append("")
    lines.append("| policy | W14 | Wa | Wb |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f'| oracle14_s | {_fmt(comp["oracle14_s_W14_mean"])} | {_fmt(comp["oracle14_s_Wa_mean"])} | '
        f'{_fmt(comp["oracle14_s_Wb_mean"])} |'
    )
    lines.append(
        f'| oracle7a_s | {_fmt(comp["oracle7a_s_W14_mean"])} | {_fmt(comp["oracle7a_s_Wa_mean"])} | '
        f'{_fmt(comp["oracle7a_s_Wb_mean"])} |'
    )
    lines.append("")
    lines.append(
        f'Seeds where oracle14_s has the lower W14: {comp["seeds_oracle14_lower"]} of {len(seeds)}. '
        f'Seeds where oracle7a_s has the lower W14: {comp["seeds_oracle7a_lower"]} of {len(seeds)}.'
    )
    lines.append("")

    def block_table(title, field):
        lines.append(f"## {title}")
        lines.append("")
        header = "| policy | L | " + " | ".join(f"seed {seed}" for seed in seeds) + " | mean | std ddof=0 |"
        lines.append(header)
        lines.append("|---|---:|" + "---:|" * len(seeds) + "---:|---:|")
        for policy, L in _policy_keys():
            block = summary["by_policy"][f"{policy}/L{L}"]
            cells = []
            for seed in seeds:
                row = _pick(result["runs"], policy, seed, L)
                cells.append(_fmt(row[field]))
            lines.append(
                f'| {policy} | {L} | ' + " | ".join(cells)
                + f' | {_fmt(block[field + "_mean"])} | {_fmt(block[field + "_std_ddof0"])} |'
            )
        lines.append("")

    block_table("W14", "W14")
    block_table("Wa", "Wa")
    block_table("Wb", "Wb")

    lines.append("## Gaps")
    lines.append("")
    lines.append(
        "On W14. A negative gap means the first policy is lower. The L = 2,048 gap uses both "
        "policies at L = 2,048. The scarcity gap is oracle14_s at L = 2,048 minus oracle14_s at L = 8,192."
    )
    lines.append("")
    lines.append(
        "| seed | oracle14_s − rand_dwell_s | oracle7a_s − rand_dwell_s | "
        "oracle14_s L2048 − L8192 |"
    )
    lines.append("|---:|---:|---:|---:|")
    for gap in summary["gaps"]:
        lines.append(
            f'| {gap["seed"]} | {_fmt(gap["oracle14_s_minus_rand_dwell_s"])} | '
            f'{_fmt(gap["oracle7a_s_minus_rand_dwell_s"])} | '
            f'{_fmt(gap["oracle14_s_L2048_minus_L8192"])} |'
        )
    gsum = summary["gap_summary"]
    lines.append(
        "| mean | "
        + " | ".join(_fmt(gsum[key]["mean"]) for key in (
            "oracle14_s_minus_rand_dwell_s",
            "oracle7a_s_minus_rand_dwell_s",
            "oracle14_s_L2048_minus_L8192",
        ))
        + " |"
    )
    lines.append(
        "| std ddof=0 | "
        + " | ".join(_fmt(gsum[key]["std_ddof0"]) for key in (
            "oracle14_s_minus_rand_dwell_s",
            "oracle7a_s_minus_rand_dwell_s",
            "oracle14_s_L2048_minus_L8192",
        ))
        + " |"
    )
    lines.append(
        "| left lower (count) | "
        + " | ".join(
            f'{gsum[key]["left_lower"]} of {gsum[key]["n"]}' for key in (
                "oracle14_s_minus_rand_dwell_s",
                "oracle7a_s_minus_rand_dwell_s",
                "oracle14_s_L2048_minus_L8192",
            )
        )
        + " |"
    )
    lines.append("")

    lines.append("## Per-room held-out bits")
    lines.append("")
    header = "| policy | L | seed | " + " | ".join(LEARN14) + " | W14 |"
    lines.append(header)
    lines.append("|---|---:|---:|" + "---:|" * len(LEARN14) + "---:|")
    runs = sorted(result["runs"], key=lambda row: (
        _policy_keys().index((row["policy"], row["L"])), row["seed"],
    ))
    for row in runs:
        lines.append(
            f'| {row["policy"]} | {row["L"]} | {row["seed"]} | '
            + " | ".join(_fmt(row["held"][name]) for name in LEARN14)
            + f' | {_fmt(row["W14"])} |'
        )
    lines.append("")

    lines.append("## Forgetting")
    lines.append("")
    lines.append(
        "Exam-store bits/symbol just after the last visit that dwelled in the room, subtracted "
        "from the end-of-life bits/symbol. A room with no exam glance at that visit is left out. "
        "n is the number of seed-room pairs in the mean."
    )
    lines.append("")
    lines.append("| policy | L | mean delta bps | std ddof=0 | n |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy, L in _policy_keys():
        block = summary["by_policy"][f"{policy}/L{L}"]
        lines.append(
            f'| {policy} | {L} | {_fmt(block["forgetting_mean"])} | '
            f'{_fmt(block["forgetting_std_ddof0"])} | {block["forgetting_n"]} |'
        )
    lines.append("")
    lines.append("| policy | L | room | n | after bps | end bps | delta bps | std ddof=0 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for policy, L in _policy_keys():
        block = summary["by_policy"][f"{policy}/L{L}"]
        for name in LEARN14:
            item = block["forgetting_room"][name]
            lines.append(
                f'| {policy} | {L} | {name} | {item["n"]} | {_fmt(item["after_bps_mean"])} | '
                f'{_fmt(item["end_bps_mean"])} | {_fmt(item["delta_bps_mean"])} | '
                f'{_fmt(item["delta_bps_std_ddof0"])} |'
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
    lines.append("| policy | L | " + " | ".join(share_names) + " |")
    lines.append("|---|---:|" + "---:|" * len(share_names))
    for policy, L in _policy_keys():
        block = summary["by_policy"][f"{policy}/L{L}"]
        cells = [_fmt(block["share_mean"][name], 4) for name in IDENTITIES]
        cells.append(_fmt(block["corridor_mean"], 4))
        cells.append(_fmt(block["sleep_mean"], 4))
        lines.append(f'| {policy} | {L} | ' + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Per seed.")
    lines.append("")
    lines.append("| policy | L | seed | " + " | ".join(share_names) + " |")
    lines.append("|---|---:|---:|" + "---:|" * len(share_names))
    for row in runs:
        cells = [f'{row["share"][name]:.4f}' for name in IDENTITIES]
        cells.append(f'{row["corridor_share"]:.4f}')
        cells.append(f'{row["sleep_share"]:.4f}')
        lines.append(
            f'| {row["policy"]} | {row["L"]} | {row["seed"]} | ' + " | ".join(cells) + " |"
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
        lines.append(f'| {seed} | ' + " | ".join(orders[seed]) + " |")
    lines.append("")

    lines.append("## M7c reference")
    lines.append("")
    ref = result["m7c_reference"]
    lines.append(
        "M7c-W3, seeds 42–53, the same bank and the same fixed sleep share, on W3's 9 rooms. "
        f'Source: {ref["source"]}.'
    )
    lines.append("")
    lines.append("| L | controller | W mean | std ddof=0 |")
    lines.append("|---:|---|---:|---:|")
    for label in ("2048", "4096"):
        for name in ("rand_dwell_s", "oracle7_s"):
            row = ref[label][name]
            lines.append(
                f'| {label} | {name} | {_fmt(row["W_mean"])} | {_fmt(row["W_std_ddof0"])} |'
            )
    lines.append("")
    lines.append("| L | oracle7_s − rand_dwell_s mean | std ddof=0 | left lower |")
    lines.append("|---:|---:|---:|---:|")
    for label in ("2048", "4096"):
        gap = ref[label]["oracle7_s_minus_rand_dwell_s"]
        lines.append(
            f'| {label} | {_fmt(gap["mean"])} | {_fmt(gap["std_ddof0"])} | '
            f'{gap["left_lower"]} of {gap["n"]} |'
        )
    lines.append("")

    lines.append("## Walls and kernels")
    lines.append("")
    lines.append("| kernel | seed | lives | wall seconds | workers |")
    lines.append("|---|---:|---|---:|---:|")
    for kernel in result["kernels"]:
        lives = ",".join(str(L) for L in kernel["lives"])
        lines.append(
            f'| {kernel["kernel_id"]} | {kernel["seed"]} | {lives} | '
            f'{kernel["wall_s"]:.3f} | {kernel["workers"]} |'
        )
    lines.append("")
    lines.append("| policy | L | seed | wall seconds | learn seconds | score seconds | eval seconds |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in runs:
        lines.append(
            f'| {row["policy"]} | {row["L"]} | {row["seed"]} | {row["wall_s"]:.3f} | '
            f'{row["learn_s"]:.3f} | {row["score_s"]:.3f} | {row["eval_s"]:.3f} |'
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
        "- 18 rooms on a ring. Identity order is the 7 H20 processes with suffix a, then suffix b, "
        "then N1, N2, N3, then Z. Ring order is `default_rng(world_seed + 70000).permutation(18)`."
    )
    lines.append(
        "- Instance a is `h20_world.World(world_seed)`. Instance b is `h20_world.World(world_seed + 1000)`. "
        "Glances are `curio.glance` on that instance. Noise room Nk uses `default_rng(world_seed + 60000 + k)` "
        "with k = 0, 1, 2. Z is 64 spaces. A room's stream resumes."
    )
    lines.append(
        "- Actions stay, left, right. A move shows 64 space symbols labelled corridor and does not "
        "observe the rooms it passes. The agent starts at ring position 0."
    )
    lines.append(
        "- Shortest path uses the same comparison as `w3_headroom.shortest`. On a tie, which an "
        "18-ring allows at distance 9, the walk goes left."
    )
    lines.append(
        "- The cortex is `m7_w3.BankCortex`: roots init, trace learning, Adam lr 1e-3, model seed 0. "
        "The channel embedding has 19 rows (18 rooms plus the corridor), initialised at zero. "
        "Corridor is row 18. Room rows follow the identity order."
    )
    lines.append(
        "- Stores, coin and replacement seeds are M7's. Exam coin `default_rng(40000 + seed)`, "
        "exam replacement `default_rng(20000 + seed)`, replay replacement `default_rng(50000 + seed)`, "
        "sleep draw `default_rng(30000 + seed)`. Reservoirs hold 32 glances per room. "
        "A sleep glance is learned from a zero state and does not write the live stream."
    )
    lines.append(
        "- The dose is 8 times the integer nearest to 0.20 L / 8. Placement is `m7_w3.sleep_plan` "
        "and the same due rule, with the longest visit set to half the ring plus 8. On a 9-ring "
        "that longest visit is 12, matching `m7_w3.sleep_due`. A sleep visit starts only when the "
        "current plan is empty."
    )
    lines.append(
        "- Policy rng is `default_rng(1000 + world_seed)`. rand_dwell_s targets the 18 rooms. "
        "oracle14_s targets the 14 learnable rooms. oracle7a_s targets the 7 rooms of instance a. "
        "A target is drawn uniformly, then the shortest path and 8 stays."
    )
    lines.append(
        "- W14 is the mean over the 14 learnable rooms of bits/symbol on EVAL=2048 fresh symbols "
        "from a copy of the world. The frozen bank starts from a zero state and uses that room's "
        "channel row. Wa and Wb are the means of the two instances."
    )
    lines.append(
        "- Forgetting is M7's: end-of-life exam bits/symbol minus the snapshot taken at the end of "
        "the last dwell in that room, before a sleep that follows it. The mean is over learnable "
        "rooms that had an exam glance in that snapshot."
    )
    lines.append(
        "- World seeds 0, 1, 2, 3, 4 and 5. L = 2048 for every policy. oracle14_s also runs at "
        "L = 8192. CPU. Private Kaggle kernels, one seed each, pool of 4, no GPU, no internet. "
        "Nothing retuned."
    )
    lines.append("")
    return "\n".join(lines)


def _load_kernel_payloads(out_dir):
    out_dir = Path(out_dir)
    found = {seed: [] for seed in SEEDS}
    for seed in SEEDS:
        hits = sorted(out_dir.glob(f"s{seed}/w4_s{seed}.json"))
        hits += sorted(out_dir.glob(f"s{seed}-l8192/w4_s{seed}.json"))
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
                _check_row(row, seed, row["policy"], row["L"])
                runs.append(row)
                seen.append((row["policy"], row["L"]))
            kernels.append({
                "kernel_id": payload["kernel_id"],
                "seed": seed,
                "lives": sorted({int(job["L"]) for job in payload["jobs"]}),
                "wall_s": payload["wall_s"],
                "workers": payload["workers"],
                "path": str(path.resolve().relative_to(ROOT)),
            })
        expect = [(policy, L_SHORT) for policy in POLICIES] + [(LONG_POLICY, L_LONG)]
        if sorted(seen) != sorted(expect):
            raise RuntimeError(f"seed {seed} jobs {seen} != {expect}")
    if len(runs) != len(SEEDS) * 4:
        raise RuntimeError(f"expected {len(SEEDS) * 4} runs, got {len(runs)}")
    summary = summarize(runs)
    kernels.sort(key=lambda item: (item["seed"], item["lives"]))
    status_path = out_dir / "kernel_status.txt"
    result = {
        "seeds": list(SEEDS),
        "policies": list(POLICIES),
        "L_short": L_SHORT,
        "L_long": L_LONG,
        "model_seed": m7.MODEL_SEED,
        "n_ch": N_CH,
        "sleep_share_target": SLEEP_SHARE,
        "sleep_decisions": {str(L): sleep_decisions(L) for L in (L_SHORT, L_LONG)},
        "dataset_id": DATASET_ID,
        "kernels": kernels,
        "summary": summary,
        "m7c_reference": m7c_reference(),
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
    parser.add_argument("--out", default=str(ROOT / "results" / "w4"))
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
