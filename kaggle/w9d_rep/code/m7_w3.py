"""M7-W3: the W3 loop on the trace-learned roots bank. Numbers only. No verdict.

The cortex is H30y's roots_learn bank. A channel embedding is added to the symbol
embedding. Room glances are stored whole, exam and replay disjoint. Sleep replays
one store and is scored on the other. The live state is restored after a sleep visit.
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
    if os.environ.get("_M7_MAIN") != "1":
        os.environ["_M7_MAIN"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import copy
import json
import math
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import curio
import h20_world
from h20_world import ALPHA
from h30w_ring import CHUNK, CLIP, EMB, LR, UNITS
from h30x_trace import init_state
from h30y_roots import build_arm
from w3_headroom import next_action, shortest
from w3_world import World3

SEEDS = (30, 31, 32, 33, 34, 35)
CONTROLLERS = ("rand_dwell", "oracle7", "prog_total", "prog_sleep", "fixed_sleep")
SCORED_L = 2048
MODEL_SEED = 0
R = 32
DWELL = 8
SLEEP_ARM = 9
MAX_ROOM_VISIT = 4 + DWELL
N_CH = 10
KERNEL_WALL_LIMIT_S = 90 * 60
LOG2 = math.log(2.0)
BITS_LIMIT = 1e-6
SMOKE_SEED = 30

if UNITS != 256 or EMB != 32 or CHUNK != 64 or CLIP != 1.0 or abs(LR - 1e-3) > 0.0:
    raise RuntimeError("H30 bank constants are not the M7 spec")
if len(ALPHA) != 41 or curio.B != 64 or curio.EVAL != 2048 or len(curio.CH) != 9:
    raise RuntimeError("W3 world constants are not the M7 spec")

ROOM_INDEX = {name: i for i, name in enumerate(curio.CH)}
CORRIDOR = 9
ROOT = Path(__file__).resolve().parent


def py(value):
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    if isinstance(value, np.ndarray):
        return py(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(py(obj), indent=2, allow_nan=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def mean_std(values):
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return None, None
    return float(data.mean()), float(data.std(ddof=0))


def fresh_state(model):
    h, e_lam, e_b, z = init_state(model)
    e_ch = torch.zeros(N_CH, UNITS, dtype=h.dtype)
    return (h, e_lam, e_b, z, e_ch)


def clone_state(state):
    return tuple(tensor.detach().clone() for tensor in state)


def states_equal(left, right):
    return all(torch.equal(a, b) for a, b in zip(left, right))


class Glance:
    __slots__ = ("gid", "channel", "symbols", "prev")

    def __init__(self, gid, channel, symbols, prev):
        self.gid = int(gid)
        self.channel = int(channel)
        self.symbols = np.asarray(symbols, dtype=np.int64)
        self.prev = int(prev)
        if self.symbols.shape != (CHUNK,):
            raise RuntimeError(f"glance length {self.symbols.shape}")
        if self.channel < 0 or self.channel >= len(curio.CH):
            raise RuntimeError("room channel index")


class Reservoir:
    """One reservoir per room channel. The replacement rng is drawn only when full."""

    def __init__(self, cap, rng):
        self.cap = int(cap)
        self.rng = rng
        self.slots = {name: [] for name in curio.CH}
        self.seen = {name: 0 for name in curio.CH}
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
        return [glance for name in curio.CH for glance in self.slots[name]]

    def draw(self, rng):
        pool = self.flat()
        if not pool:
            return None
        return pool[int(rng.integers(len(pool)))]

    def sizes(self):
        return {name: len(self.slots[name]) for name in curio.CH}


def score_sequences(model, channel_emb, symbols, prev, channels):
    """Total bits of each sequence, from a zero state. symbols is (B, T) long."""
    if symbols.numel() == 0:
        return torch.zeros(0, dtype=torch.float64)
    batch, length = symbols.shape
    token_ids = torch.empty(batch, length, dtype=torch.long)
    token_ids[:, 0] = prev.clamp(min=0)
    if length > 1:
        token_ids[:, 1:] = symbols[:, :-1]
    with torch.no_grad():
        emb = model.embedding(token_ids)
        missing = prev < 0
        if bool(missing.any()):
            emb = emb.clone()
            emb[missing, 0] = 0
        emb = emb + channel_emb[channels][:, None, :]
        hidden = torch.zeros(batch, UNITS, dtype=torch.complex64)
        lam = model.lam()
        injected = torch.complex(emb @ model.B_re.T, emb @ model.B_im.T)
        total = torch.zeros(batch, dtype=torch.float64)
        for t in range(length):
            hidden = lam * hidden + injected[:, t]
            logits = model.logits_from(hidden)
            loss = F.cross_entropy(logits, symbols[:, t], reduction="none")
            total = total + loss.double()
    return total / LOG2


def score_glances(model, channel_emb, glances):
    if not glances:
        return torch.zeros(0, dtype=torch.float64)
    symbols = torch.from_numpy(np.stack([g.symbols for g in glances]))
    prev = torch.tensor([g.prev for g in glances], dtype=torch.long)
    channels = torch.tensor([g.channel for g in glances], dtype=torch.long)
    return score_sequences(model, channel_emb, symbols, prev, channels)


def trace_glance(model, channel_emb, symbols, prev, channel, state, mean=True):
    """One glance of the H30x trace rule, with a channel embedding added to x_t.

    symbols is a 1-D long tensor. prev is -1 only at the start of a stream, where
    the symbol embedding is zero. Channel eligibility carries in state and decays
    on every step, whether or not that channel is the one being read.
    """
    hidden, e_lam, e_b, z, e_ch = state
    length = int(symbols.shape[0])
    if length != CHUNK:
        raise RuntimeError(f"glance length {length}")
    channel = int(channel)
    prev_id = int(prev)
    if channel < 0 or channel >= N_CH:
        raise RuntimeError("channel index")
    token_ids = torch.empty(length, dtype=torch.long)
    token_ids[0] = 0 if prev_id < 0 else prev_id
    if length > 1:
        token_ids[1:] = symbols[:-1]
    emb = model.embedding(token_ids).detach()
    if prev_id < 0:
        emb = emb.clone()
        emb[0] = 0
    emb = emb + channel_emb[channel].detach()
    cdtype = hidden.dtype
    with torch.no_grad():
        lam = model.lam().detach()
        injection_b = torch.complex(model.B_re.detach(), model.B_im.detach())
        injected = model.inject(emb)
        dlam_dnu = -torch.exp(model.nu.detach()) * lam
        dlam_dth = 1j * lam
        h_buf = torch.empty(length, UNITS, dtype=cdtype)
        e_buf = torch.empty(length, UNITS, dtype=cdtype)
        b_buf = torch.empty(length, UNITS, EMB, dtype=cdtype)
        z_buf = torch.empty(length, UNITS, z.shape[1], dtype=cdtype)
        e_ch_buf = torch.empty(length, N_CH, UNITS, dtype=cdtype)
        lam_col = lam[:, None]
        for t in range(length):
            e_lam.mul_(lam).add_(hidden)
            e_b.mul_(lam_col).add_(emb[t])
            z.mul_(lam_col)
            if t == 0:
                if prev_id >= 0:
                    z[:, prev_id] += 1
            else:
                z[:, int(symbols[t - 1])] += 1
            e_ch.mul_(lam)
            e_ch[channel] += 1
            hidden.mul_(lam).add_(injected[t])
            h_buf[t].copy_(hidden)
            e_buf[t].copy_(e_lam)
            b_buf[t].copy_(e_b)
            z_buf[t].copy_(z)
            e_ch_buf[t].copy_(e_ch)
    hr = h_buf.real.detach().requires_grad_(True)
    hi = h_buf.imag.detach().requires_grad_(True)
    logits = model.logits_from(torch.complex(hr, hi))
    losses = F.cross_entropy(logits, symbols, reduction="none")
    model.zero_grad(set_to_none=True)
    channel_emb.grad = None
    losses.sum().backward()
    if hr.grad is None or hi.grad is None:
        raise RuntimeError("readout did not produce a hidden-state gradient")
    delta = torch.complex(hr.grad.detach(), hi.grad.detach())
    with torch.no_grad():
        conj = torch.conj(delta)
        model.nu.grad = (conj * (e_buf * dlam_dnu)).real.sum(dim=0)
        model.theta.grad = (conj * (e_buf * dlam_dth)).real.sum(dim=0)
        cc = conj[:, :, None]
        model.B_re.grad = (cc * b_buf).real.sum(dim=0)
        model.B_im.grad = (cc * (1j * b_buf)).real.sum(dim=0)
        coeff = conj[:, :, None] * z_buf
        model.embedding.weight.grad = torch.einsum("tia,ie->ae", coeff, injection_b).real
        weighted = conj[:, None, :] * e_ch_buf
        channel_emb.grad = torch.einsum("tci,ie->ce", weighted, injection_b).real
        for name, param in model.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise FloatingPointError(f"{name} gradient is missing or nonfinite")
        if channel_emb.grad is None or not torch.isfinite(channel_emb.grad).all():
            raise FloatingPointError("channel gradient is missing or nonfinite")
        if mean:
            scale = 1.0 / float(length)
            for param in model.parameters():
                param.grad.mul_(scale)
            channel_emb.grad.mul_(scale)
    for tensor in (hidden, e_lam, e_b, z, e_ch):
        if tensor.requires_grad or not torch.isfinite(tensor.real).all() or not torch.isfinite(tensor.imag).all():
            raise FloatingPointError("trace state is nonfinite or still on the graph")
    return losses.detach(), (hidden, e_lam, e_b, z, e_ch)


class BankCortex:
    """roots_learn bank plus one embedding row per room and the corridor.

    Model seed 0 is the H30y seed for H20 world 6. The channel rows start at zero.
    Freeze them, and leave them out of Adam, to recover that bank on a channel-free stream.
    """

    def __init__(self, model_seed=MODEL_SEED, freeze_channel=False):
        self.model = build_arm("roots_learn", len(ALPHA), int(model_seed))
        if self.model.nu.dtype != torch.float32:
            raise RuntimeError(f"bank dtype {self.model.nu.dtype}")
        with torch.no_grad():
            mag = self.model.lam().abs()
            if abs(float(mag.min()) - 0.98) > 1e-5 or abs(float(mag.max()) - 0.98) > 1e-5:
                raise RuntimeError(f"|λ| is not 0.98: {float(mag.min())} {float(mag.max())}")
        self.channel_emb = nn.Parameter(torch.zeros(N_CH, EMB))
        self.freeze_channel = bool(freeze_channel)
        if self.freeze_channel:
            self.channel_emb.requires_grad_(False)
        params = list(self.model.parameters())
        if not self.freeze_channel:
            params.append(self.channel_emb)
        self.opt = torch.optim.Adam(params, lr=LR)
        self.params = params
        self.state = fresh_state(self.model)
        self.prev_symbol = -1

    def learn_glance(self, symbols, channel, prev, state=None, commit=True):
        tokens = torch.as_tensor(symbols, dtype=torch.long)
        if state is None:
            state = self.state
        losses, new_state = trace_glance(
            self.model, self.channel_emb, tokens, prev, channel, state, mean=True,
        )
        bits = losses.double().numpy() / LOG2
        if not np.all(np.isfinite(bits)):
            raise FloatingPointError("nonfinite glance bits")
        nn.utils.clip_grad_norm_(self.params, CLIP)
        self.opt.step()
        if commit:
            self.state = new_state
            self.prev_symbol = int(tokens[-1])
        return bits

    def clone_live(self):
        return clone_state(self.state), int(self.prev_symbol)

    def live_matches(self, snap):
        state, prev = snap
        return states_equal(self.state, state) and self.prev_symbol == int(prev)

    def clone_params(self):
        tensors = [param.detach().clone() for param in self.model.parameters()]
        tensors.append(self.channel_emb.detach().clone())
        return tensors

    def param_delta(self, snap):
        current = self.clone_params()
        return max(float((a - b).abs().max()) for a, b in zip(current, snap))


def nearest_unvisited(src, unvisited, n=9):
    """Nearest unvisited room. Equal distances break toward +index (right)."""
    best, best_key = None, None
    for dst in unvisited:
        right = (dst - src) % n
        left = (src - dst) % n
        dist = right if right <= left else left
        key = (dist, 0 if right <= left else 1, dst)
        if best_key is None or key < best_key:
            best, best_key = dst, key
    return best


def ema(values, counts, arm, reward):
    if counts[arm] == 0:
        values[arm] = reward
    else:
        values[arm] = (1.0 - curio.BETA) * values[arm] + curio.BETA * reward
    counts[arm] += 1


def choose_arm(world, unvisited, first_sleep, values, n_arms, rng):
    """M5 bandit. Sleep is the last arm of the optimistic first pass."""
    if unvisited:
        target = nearest_unvisited(world.room, unvisited)
        unvisited.remove(target)
        return target, first_sleep
    if first_sleep:
        return SLEEP_ARM, False
    if rng.random() < curio.EPS:
        return int(rng.integers(n_arms)), first_sleep
    return max(range(n_arms), key=lambda k: (values[k], rng.random())), first_sleep


def split_even(total, n):
    if n <= 0:
        return []
    base, rem = divmod(int(total), int(n))
    return [base + (1 if i < rem else 0) for i in range(n)]


def sleep_plan(L, n_sleep):
    """Evenly spaced sleep visits whose lengths sum to n_sleep. Same rule as M6."""
    n_sleep = int(n_sleep)
    if n_sleep <= 0:
        return []
    if n_sleep > L:
        raise ValueError(f"sleep decisions {n_sleep} exceed L={L}")
    lengths = [DWELL] * (n_sleep // DWELL)
    rem = n_sleep % DWELL
    if rem:
        lengths.append(rem)
    gaps = split_even(L - n_sleep, len(lengths) + 1)
    plan = []
    t = 0
    for i, length in enumerate(lengths):
        t += gaps[i]
        plan.append((int(t), int(length)))
        t += length
    if t + gaps[-1] != L:
        raise RuntimeError(f"sleep plan covers {t + gaps[-1]} decisions, L={L}")
    return plan


def owed(plan, si):
    return int(sum(length for _, length in plan[si:]))


def sleep_due(t, L, plan, si):
    if si >= len(plan):
        return False
    still = owed(plan, si)
    if t >= plan[si][0]:
        return True
    if (L - t) <= still:
        return True
    if (L - t) < still + MAX_ROOM_VISIT:
        return True
    return False


def shares_of(channels):
    n = len(channels)
    counts = {name: 0 for name in curio.CH}
    corridor = 0
    sleep = 0
    for name in channels:
        if name == "corridor":
            corridor += 1
        elif name == "sleep":
            sleep += 1
        else:
            counts[name] += 1
    share = {name: counts[name] / n for name in curio.CH}
    return share, corridor / n, sleep / n


def quarters_of(channels):
    q = len(channels) // 4
    out = []
    for i in range(4):
        share, corridor, sleep = shares_of(channels[i * q:(i + 1) * q])
        out.append({"share": share, "corridor": corridor, "sleep": sleep})
    return out


def empty_reward_quarters():
    return [{"sleep_n": 0, "sleep_sum": 0.0, "room_n": 0, "room_sum": 0.0} for _ in range(4)]


def note_visit(quarters, kind, start, L, reward):
    q = L // 4
    slot = quarters[min(int(start) // q, 3)]
    if kind == "sleep":
        slot["sleep_n"] += 1
        slot["sleep_sum"] += float(reward)
    else:
        slot["room_n"] += 1
        slot["room_sum"] += float(reward)


def reward_quarters(raw):
    out = []
    for slot in raw:
        out.append({
            "sleep_n": int(slot["sleep_n"]),
            "sleep_mean": (float(slot["sleep_sum"] / slot["sleep_n"]) if slot["sleep_n"] else None),
            "room_n": int(slot["room_n"]),
            "room_mean": (float(slot["room_sum"] / slot["room_n"]) if slot["room_n"] else None),
        })
    return out


class Life:
    """One agent, one bank, two glance reservoirs. Progress reads the exam store."""

    def __init__(self, seed, score_progress):
        self.seed = int(seed)
        self.score_progress = bool(score_progress)
        self.world = World3(self.seed)
        self.cortex = BankCortex(model_seed=MODEL_SEED, freeze_channel=False)
        self.coin_rng = np.random.default_rng(40_000 + self.seed)
        self.exam = Reservoir(R, np.random.default_rng(20_000 + self.seed))
        self.replay = Reservoir(R, np.random.default_rng(50_000 + self.seed))
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
        bits = score_glances(self.cortex.model, self.cortex.channel_emb, glances)
        self.bit_by_id = {g.gid: float(bits[i]) for i, g in enumerate(glances)}
        self.cached_sum = float(bits.sum())
        self.score_s += time.perf_counter() - t0
        return self.cached_sum

    def _score_one(self, glance):
        t0 = time.perf_counter()
        bits = score_glances(self.cortex.model, self.cortex.channel_emb, [glance])
        self.score_s += time.perf_counter() - t0
        return float(bits[0])

    def _insert_exam(self, glance):
        action, old = self.exam.offer(curio.CH[glance.channel], glance)
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
        """Learn the live glance, then score the exam store it has not yet entered."""
        before = self.cached_sum
        prev = self.cortex.prev_symbol
        self._learn(symbols, ROOM_INDEX[name], prev, commit=True)
        if self.score_progress:
            after = self._rescore()
            progress = before - after
        else:
            progress = 0.0
        glance = Glance(self.next_gid, ROOM_INDEX[name], symbols, prev)
        self.next_gid += 1
        if int(self.coin_rng.integers(2)) == 0:
            self._insert_exam(glance)
        else:
            self.replay.offer(name, glance)
        if self.exam.ids & self.replay.ids:
            raise RuntimeError("a glance is in both stores")
        return float(progress)

    def corridor_decision(self, symbols):
        before = self.cached_sum
        prev = self.cortex.prev_symbol
        self._learn(symbols, CORRIDOR, prev, commit=True)
        if not self.score_progress:
            return 0.0
        after = self._rescore()
        return float(before - after)

    def sleep_decision(self):
        """One replay glance from a zero state. The live stream is not written.

        Returns (progress, learned). An empty replay store learns nothing.
        """
        before = self.cached_sum
        glance = self.replay.draw(self.sleep_rng)
        if glance is None:
            return 0.0, False
        if glance.gid in self.exam.ids:
            self.exam_drawn += 1
        self._learn(
            glance.symbols, glance.channel, glance.prev,
            state=fresh_state(self.cortex.model), commit=False,
        )
        after = self._rescore()
        return float(before - after), True

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
        """Exam-store bits of one channel under the weights just after a visit to it."""
        glances = list(self.exam.slots[name])
        if not glances:
            self.snapshots[name] = {"n": 0, "bits": None, "bps": None, "ids": []}
            return
        t0 = time.perf_counter()
        total = float(score_glances(self.cortex.model, self.cortex.channel_emb, glances).sum())
        self.score_s += time.perf_counter() - t0
        n = len(glances)
        self.snapshots[name] = {
            "n": n,
            "bits": total,
            "bps": total / (n * CHUNK),
            "ids": [g.gid for g in glances],
        }

    def forgetting(self):
        out = {}
        for name in curio.LEARN:
            snap = self.snapshots.get(name)
            current = list(self.exam.slots[name])
            if snap is None or snap["n"] == 0:
                out[name] = {
                    "visited": snap is not None,
                    "n": 0 if snap is None else int(snap["n"]),
                    "after_bits": None,
                    "end_bits": None,
                    "after_bps": None,
                    "end_bps": None,
                    "delta_bps": None,
                    "delta_bits": None,
                }
                continue
            if [g.gid for g in current] != snap["ids"]:
                raise RuntimeError(f"{name} exam set changed after its last visit")
            end_bits = float(score_glances(self.cortex.model, self.cortex.channel_emb, current).sum())
            end_bps = end_bits / (snap["n"] * CHUNK)
            out[name] = {
                "visited": True,
                "n": int(snap["n"]),
                "after_bits": float(snap["bits"]),
                "end_bits": end_bits,
                "after_bps": float(snap["bps"]),
                "end_bps": float(end_bps),
                "delta_bps": float(end_bps - snap["bps"]),
                "delta_bits": float(end_bits - snap["bits"]),
            }
        return out

    def audit_cache(self):
        if not self.score_progress:
            return 0.0
        glances = self.exam.flat()
        if not glances:
            if abs(self.cached_sum) > 1e-6:
                raise RuntimeError("empty exam cache is not zero")
            return 0.0
        fresh = float(score_glances(self.cortex.model, self.cortex.channel_emb, glances).sum())
        delta = abs(fresh - self.cached_sum)
        if delta > 1.0:
            raise RuntimeError(f"exam cache drifted by {delta} bits")
        return delta


def evaluate_bank(cortex, world, seed):
    """Frozen bank, zero state, fresh EVAL continuation, that channel's embedding."""
    wc = copy.deepcopy(world.h20)
    sequences = []
    for name in curio.CH:
        if name == "N":
            rng = np.random.default_rng(30_000 + int(seed))
            seq = rng.integers(0, curio.A, size=curio.EVAL, dtype=np.int64)
        elif name == "Z":
            seq = np.full(curio.EVAL, ALPHA.index(" "), dtype=np.int64)
        else:
            seq = np.asarray([ALPHA.index(s) for s in wc.take(name, curio.EVAL)], dtype=np.int64)
        if seq.shape != (curio.EVAL,):
            raise RuntimeError(f"{name} continuation length {seq.shape}")
        sequences.append(seq)
    symbols = torch.from_numpy(np.stack(sequences))
    prev = torch.full((len(curio.CH),), -1, dtype=torch.long)
    channels = torch.arange(len(curio.CH), dtype=torch.long)
    totals = score_sequences(cortex.model, cortex.channel_emb, symbols, prev, channels)
    held = {name: float(totals[i]) / curio.EVAL for i, name in enumerate(curio.CH)}
    W = float(np.mean([held[name] for name in curio.LEARN]))
    W9 = float(np.mean([held[name] for name in curio.CH]))
    return W, W9, held


def finish_row(seed, controller, L, life, channels, t0, score, reward_q, sleep_target):
    eval_s = 0.0
    if score:
        e0 = time.perf_counter()
        W, W9, held9 = evaluate_bank(life.cortex, life.world, seed)
        forgetting = life.forgetting()
        eval_s = time.perf_counter() - e0
        finite = bool(
            math.isfinite(W) and math.isfinite(W9)
            and all(math.isfinite(held9[name]) for name in curio.CH)
        )
    else:
        W, W9 = float("nan"), float("nan")
        held9 = {}
        forgetting = {}
        finite = False
    share, corridor, sleep = shares_of(channels)
    quarter = []
    for block in quarters_of(channels):
        quarter.append({
            "share": {name: float(block["share"][name]) for name in curio.CH},
            "corridor": float(block["corridor"]),
            "sleep": float(block["sleep"]),
        })
    wall = time.perf_counter() - t0
    checks = life.sleep_checks
    return {
        "seed": int(seed),
        "controller": controller,
        "L": int(L),
        "W": float(W),
        "W9": float(W9),
        "held": {name: float(held9[name]) for name in curio.LEARN} if held9 else {},
        "held9": {name: float(held9[name]) for name in curio.CH} if held9 else {},
        "forgetting": forgetting,
        "share": {name: float(share[name]) for name in curio.CH},
        "corridor_share": float(corridor),
        "sleep_share": float(sleep),
        "sleep_decisions": int(sum(name == "sleep" for name in channels)),
        "sleep_target": (None if sleep_target is None else int(sleep_target)),
        "quarter": quarter,
        "reward_quarter": reward_quarters(reward_q) if reward_q is not None else None,
        "room_channel": list(life.world.rooms),
        "exam_sizes": life.exam.sizes(),
        "replay_sizes": life.replay.sizes(),
        "exam_seen": {name: int(life.exam.seen[name]) for name in curio.CH},
        "replay_seen": {name: int(life.replay.seen[name]) for name in curio.CH},
        "exam_glances_drawn": int(life.exam_drawn),
        "overlap": int(len(life.exam.ids & life.replay.ids)),
        "sleep_state_intact": (None if not checks else all(item["intact"] for item in checks)),
        "sleep_param_delta": (None if not checks else float(checks[0]["param_delta"])),
        "sleep_visits_checked": int(len(checks)),
        "cache_delta": None,
        "wall_s": float(wall),
        "eval_s": float(eval_s),
        "learn_s": float(life.learn_s),
        "score_s": float(life.score_s),
        "finite": finite,
    }


def run_fixed(seed, policy, L, score=True):
    """rand_dwell or oracle7. Same planner and rng stream as w3_headroom.run_life."""
    if L % 4 != 0:
        raise ValueError(f"L={L} is not divisible by 4")
    t0 = time.perf_counter()
    life = Life(seed, score_progress=False)
    rng = np.random.default_rng(1_000 + seed)
    state = {"kind": policy, "room": 0, "plan": []}
    channels = []
    pending = None
    for _ in range(L):
        action = next_action(state, life.world.rooms, rng)
        if pending is not None:
            staying = action == "stay" and life.world.rooms[life.world.room] == pending
            if not staying:
                life.snapshot(pending)
                pending = None
        room, channel, symbols = life.world.step(action, rng)
        state["room"] = room
        if channel == "corridor":
            life.corridor_decision(symbols)
        else:
            life.room_decision(symbols, channel)
            pending = channel
        channels.append(channel)
    if pending is not None:
        life.snapshot(pending)
    if len(channels) != L:
        raise RuntimeError(f"{policy} ran {len(channels)} decisions")
    return finish_row(seed, policy, L, life, channels, t0, score, None, None)


def run_bandit(seed, kind, L, score=True, n_sleep_target=None, check_sleep=False):
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
    life = Life(seed, score_progress=True)
    choice_rng = np.random.default_rng(10_000 + seed)
    world_rng = np.random.default_rng(1_000 + seed)
    n_arms = 10 if kind == "prog_sleep" else 9
    values = [0.0] * n_arms
    counts = [0] * n_arms
    unvisited = set(range(9))
    first_sleep = kind == "prog_sleep"
    channels = []
    reward_q = empty_reward_quarters()
    si = 0
    t = 0
    checked = False

    def one_sleep_visit(length):
        nonlocal t, checked
        start = t
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
        return start, n, reward

    while t < L:
        if kind == "fixed_sleep" and sleep_due(t, L, plan, si):
            start, n, reward = one_sleep_visit(plan[si][1])
            note_visit(reward_q, "sleep", start, L, reward)
            si += 1
            continue
        arm, first_sleep = choose_arm(life.world, unvisited, first_sleep, values, n_arms, choice_rng)
        if arm == SLEEP_ARM:
            start, n, reward = one_sleep_visit(min(DWELL, L - t))
            ema(values, counts, SLEEP_ARM, reward)
            note_visit(reward_q, "sleep", start, L, reward)
            continue
        target = int(arm)
        start = t
        path = shortest(life.world.room, target)
        total = 0.0
        travel = 0
        while path and t < L:
            action = path.pop(0)
            _room, channel, symbols = life.world.step(action, world_rng)
            if channel != "corridor":
                raise RuntimeError("a move did not observe the corridor")
            total += life.corridor_decision(symbols)
            channels.append(channel)
            travel += 1
            t += 1
        dwell = 0
        dwell_name = None
        while dwell < DWELL and t < L:
            _room, channel, symbols = life.world.step("stay", world_rng)
            if channel == "corridor":
                raise RuntimeError("a stay observed the corridor")
            total += life.room_decision(symbols, channel)
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
        if dwell and dwell_name != life.world.rooms[target]:
            raise RuntimeError("dwell channel is not the chosen room")
        if dwell:
            life.snapshot(life.world.rooms[target])
        n_dec = travel + dwell
        reward = total / n_dec if n_dec else 0.0
        ema(values, counts, target, reward)
        note_visit(reward_q, "room", start, L, reward)
    if len(channels) != L:
        raise RuntimeError(f"{kind} seed {seed} ran {len(channels)} decisions")
    if kind == "fixed_sleep" and si != len(plan):
        raise RuntimeError(f"fixed_sleep took {si} of {len(plan)} planned sleep visits")
    if life.exam_drawn != 0:
        raise RuntimeError(f"sleep drew {life.exam_drawn} exam glances")
    cache_delta = life.audit_cache()
    row = finish_row(
        seed, kind, L, life, channels, t0, score, reward_q,
        (int(n_sleep_target) if kind == "fixed_sleep" else None),
    )
    row["cache_delta"] = float(cache_delta)
    if kind == "fixed_sleep" and row["sleep_decisions"] != int(n_sleep_target):
        raise RuntimeError(
            f"fixed_sleep sleep decisions {row['sleep_decisions']} != target {int(n_sleep_target)}"
        )
    if check_sleep and kind in ("prog_sleep", "fixed_sleep") and row["sleep_decisions"] > 0:
        if row["sleep_state_intact"] is not True:
            raise RuntimeError("sleep visit did not keep the waking stream")
    return row


def run_controller(seed, controller, L, score=True, n_sleep_target=None, check_sleep=False):
    if controller in ("rand_dwell", "oracle7"):
        if n_sleep_target not in (None, 0):
            raise ValueError("fixed policies do not take a sleep target")
        return run_fixed(seed, controller, L, score=score)
    return run_bandit(
        seed, controller, L, score=score,
        n_sleep_target=n_sleep_target, check_sleep=check_sleep,
    )


def _job(args):
    seed, controller, L, n_sleep_target = args
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if controller == "fixed_sleep":
        raise RuntimeError("fixed_sleep is started by the kernel after prog_sleep")
    print(f"s{seed} {controller} L={L} start", flush=True)
    row = run_controller(seed, controller, L, score=True, check_sleep=True)
    print(
        f's{seed} {controller} L={L} W {row["W"]:.4f} W9 {row["W9"]:.4f} '
        f'sleep {row["sleep_share"]:.3f} corridor {row["corridor_share"]:.3f} '
        f'wall {row["wall_s"]:.1f}s learn {row["learn_s"]:.1f}s score {row["score_s"]:.1f}s',
        flush=True,
    )
    return row


def _job_fixed(args):
    seed, L, n_sleep_target = args
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    print(f"s{seed} fixed_sleep L={L} start target {n_sleep_target}", flush=True)
    row = run_controller(
        seed, "fixed_sleep", L, score=True, n_sleep_target=n_sleep_target, check_sleep=True,
    )
    print(
        f's{seed} fixed_sleep L={L} W {row["W"]:.4f} W9 {row["W9"]:.4f} '
        f'sleep {row["sleep_share"]:.3f} wall {row["wall_s"]:.1f}s',
        flush=True,
    )
    return row


def check_h20(log):
    """Frozen zero channel embedding versus h30y roots_learn on H20 slow seed 6."""
    from h30w_ring import encode_text
    from h30x_trace import train_trace
    from h20_world import stream

    text, _labels = stream("slow", 6, 4096)
    tokens = encode_text(text)
    if len(tokens) != 4096:
        raise RuntimeError("H20 smoke length")
    ref = build_arm("roots_learn", len(ALPHA), MODEL_SEED)
    ref_bits, ref_s = train_trace(ref, tokens, "h30y-ref", progress_every=10 ** 9)
    cortex = BankCortex(model_seed=MODEL_SEED, freeze_channel=True)
    got = []
    t0 = time.perf_counter()
    for start in range(0, 4096, CHUNK):
        prev = -1 if start == 0 else int(tokens[start - 1])
        got.append(cortex.learn_glance(tokens[start:start + CHUNK], 0, prev))
    got_bits = np.concatenate(got)
    delta = float(np.max(np.abs(got_bits - ref_bits)))
    ok = delta <= BITS_LIMIT
    log(
        f"smoke check 1: frozen zero channel vs h30y roots_learn, H20 slow seed 6, n=4096"
    )
    log(
        f"  ref_bits {float(ref_bits.mean()):.8f} bank_bits {float(got_bits.mean()):.8f} "
        f"max_abs_delta {delta:.3e} limit {BITS_LIMIT:.1e} ok {ok} "
        f"ref_s {ref_s:.2f} bank_s {time.perf_counter() - t0:.2f}"
    )
    if not ok:
        raise RuntimeError(f"H30y bit match failed, max abs delta {delta}")
    return delta


def check_scorer(log):
    """Zero-state batched bits agree with the trace loss before the Adam step."""
    cortex = BankCortex(model_seed=MODEL_SEED, freeze_channel=False)
    with torch.no_grad():
        cortex.channel_emb[2, 0] = 0.25
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, len(ALPHA), size=CHUNK, dtype=np.int64)
    prev = 5
    glance = Glance(0, 2, symbols, prev)
    scored = score_glances(cortex.model, cortex.channel_emb, [glance])
    state = fresh_state(cortex.model)
    losses, _state = trace_glance(
        cortex.model, cortex.channel_emb,
        torch.as_tensor(symbols, dtype=torch.long), prev, 2, state, mean=False,
    )
    traced = float(losses.double().sum() / LOG2)
    delta = abs(float(scored[0]) - traced)
    ok = delta <= 1e-4
    log(
        f"smoke check 1b: batched zero-state score vs trace loss "
        f"score {float(scored[0]):.6f} trace {traced:.6f} abs {delta:.3e} ok {ok}"
    )
    if not ok:
        raise RuntimeError(f"scorer disagrees with the trace loss by {delta}")
    return delta


def _store_line(row, name):
    return (
        f"  {name} exam {row['exam_sizes'][name]} (offered {row['exam_seen'][name]}) "
        f"replay {row['replay_sizes'][name]} (offered {row['replay_seen'][name]})"
    )


def check_stores(log):
    log("smoke check 2: prog_sleep seed 30 L=128, disjoint stores, sleep draws no exam glance")
    row = run_controller(SMOKE_SEED, "prog_sleep", 128, score=False, check_sleep=True)
    for name in curio.CH:
        log(_store_line(row, name))
    log(
        f"  overlap {row['overlap']} exam_glances_drawn {row['exam_glances_drawn']} "
        f"sleep_decisions {row['sleep_decisions']} sleep_state_intact {row['sleep_state_intact']} "
        f"sleep_param_delta {row['sleep_param_delta']:.3e} cache_delta {row['cache_delta']:.3e} "
        f"wall {row['wall_s']:.2f}s"
    )
    if row["overlap"] != 0 or row["exam_glances_drawn"] != 0:
        raise RuntimeError("stores overlap or sleep drew an exam glance")
    if row["sleep_decisions"] <= 0:
        raise RuntimeError("L=128 prog_sleep took no sleep, so the waking-stream check did not run")
    if not row["sleep_state_intact"] or not (row["sleep_param_delta"] > 0.0):
        raise RuntimeError("sleep did not preserve the waking stream")
    log(
        "smoke check 3: first sleep visit kept live state and traces bit-identical; "
        f"parameter max abs delta {row['sleep_param_delta']:.3e}"
    )
    return row


def check_finite(log):
    log("smoke check 4: five controllers, seed 30, L=64, W and W9 finite")
    rows = []
    for controller in ("rand_dwell", "oracle7", "prog_total", "prog_sleep"):
        row = run_controller(SMOKE_SEED, controller, 64, score=True, check_sleep=False)
        rows.append(row)
        log(
            f"  {controller} W {row['W']:.6f} W9 {row['W9']:.6f} finite {row['finite']} "
            f"sleep {row['sleep_share']:.4f} corridor {row['corridor_share']:.4f} "
            f"wall {row['wall_s']:.2f}s learn {row['learn_s']:.2f}s score {row['score_s']:.2f}s"
        )
        if not row["finite"]:
            raise RuntimeError(f"{controller} W is not finite")
    slept = rows[-1]["sleep_decisions"]
    fixed = run_controller(
        SMOKE_SEED, "fixed_sleep", 64, score=True, n_sleep_target=slept, check_sleep=False,
    )
    rows.append(fixed)
    log(
        f"  fixed_sleep W {fixed['W']:.6f} W9 {fixed['W9']:.6f} finite {fixed['finite']} "
        f"sleep {fixed['sleep_share']:.4f} target {slept} wall {fixed['wall_s']:.2f}s"
    )
    if not fixed["finite"]:
        raise RuntimeError("fixed_sleep W is not finite")
    if fixed["sleep_decisions"] != slept:
        raise RuntimeError("fixed_sleep dose does not match prog_sleep at L=64")
    return rows


def time_full_rescore(model, channel_emb):
    rng = np.random.default_rng(1)
    glances = []
    for i in range(R * len(curio.CH)):
        glances.append(Glance(
            i,
            i % len(curio.CH),
            rng.integers(0, len(ALPHA), size=CHUNK, dtype=np.int64),
            int(rng.integers(0, len(ALPHA))),
        ))
    score_glances(model, channel_emb, glances)
    samples = []
    for _ in range(3):
        t0 = time.perf_counter()
        score_glances(model, channel_emb, glances)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def check_time(log):
    log("smoke timing: prog_sleep seed 30 L=256")
    row = run_controller(SMOKE_SEED, "prog_sleep", 256, score=False, check_sleep=False)
    learn_per = row["learn_s"] / 256.0
    full = time_full_rescore(row and BankCortex(model_seed=MODEL_SEED).model, BankCortex(model_seed=MODEL_SEED).channel_emb)
    # The line above builds two extra cortices. Time the rescore on one cortex.
    return row, learn_per, full


def kernel_estimate(learn_per, full_rescore_s):
    """Two waves: prog_sleep beside the other three, then fixed_sleep.

    Every decision is charged a full 288-glance exam rescore plus one learn step.
    Stores start empty, so this is above the run it estimates.
    """
    one = SCORED_L * (full_rescore_s + learn_per)
    return 2.0 * one, one


def smoke(out_dir=None):
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    out_dir = Path(out_dir or (ROOT / "results" / "m7w3"))
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
        "model_seed": MODEL_SEED,
        "kernels_pushed": False,
    }
    log(f"torch={torch.__version__} threads={torch.get_num_threads()} model_seed={MODEL_SEED}")
    try:
        check_h20(log)
        check_scorer(log)
        check_stores(log)
        check_finite(log)
        log("smoke timing: prog_sleep seed 30 L=256")
        timed = run_controller(SMOKE_SEED, "prog_sleep", 256, score=False, check_sleep=False)
        learn_per = timed["learn_s"] / 256.0
        probe = BankCortex(model_seed=MODEL_SEED, freeze_channel=False)
        full = time_full_rescore(probe.model, probe.channel_emb)
        kernel_s, one_s = kernel_estimate(learn_per, full)
        log(
            f"  L256 wall {timed['wall_s']:.2f}s learn {timed['learn_s']:.2f}s "
            f"score {timed['score_s']:.2f}s learn_per_decision {learn_per:.4f}s "
            f"full_rescore_288 {full:.4f}s"
        )
        log(
            f"  pessimistic one slow controller {one_s:.1f}s ({one_s / 60:.2f} min), "
            f"two-wave kernel {kernel_s:.1f}s ({kernel_s / 60:.2f} min), limit {KERNEL_WALL_LIMIT_S:.0f}s"
        )
        log(
            "  estimate charges every decision a full exam store and runs fixed_sleep after prog_sleep; "
            "the other three controllers share prog_sleep's wave"
        )
        payload.update(
            timed_L=256,
            timed_wall_s=timed["wall_s"],
            timed_learn_s=timed["learn_s"],
            timed_score_s=timed["score_s"],
            learn_per_decision_s=learn_per,
            full_rescore_s=full,
            pessimistic_controller_s=one_s,
            kernel_estimate_s=kernel_s,
            kernel_limit_s=KERNEL_WALL_LIMIT_S,
        )
        if kernel_s > KERNEL_WALL_LIMIT_S:
            raise RuntimeError(
                f"kernel estimate {kernel_s:.0f}s exceeds {KERNEL_WALL_LIMIT_S:.0f}s"
            )
        payload["ok"] = True
    except Exception as exc:
        payload["ok"] = False
        payload["reason"] = str(exc)
        log(f"SMOKE_FAIL {exc}")
    payload["wall_s"] = time.perf_counter() - t0
    payload["log"] = lines
    dump(out_dir / "smoke.json", payload)
    (out_dir / "smoke.log").write_text("\n".join(lines) + "\n")
    if not payload["ok"]:
        print(f"SMOKE_FAIL {payload['reason']}. Kernels not pushed.", flush=True)
        raise SystemExit(2)
    print(f"smoke_ok wall {payload['wall_s']:.1f}s", flush=True)
    return payload


def kernel_main(seed, out, workers=4, kernel_id=None):
    if __name__ != "m7_w3":
        raise SystemExit("import m7_w3 and call kernel_main so the pool can spawn workers")
    seed = int(seed)
    if seed not in SEEDS:
        raise SystemExit(f"seed {seed} is outside 30-35")
    if int(workers) < 1 or int(workers) > 4:
        raise SystemExit("kernel pool is at most 4")
    torch.set_num_threads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    jobs = [(seed, controller, SCORED_L, None) for controller in ("rand_dwell", "oracle7", "prog_total", "prog_sleep")]
    print(f"kernel {kernel_id} seed {seed} workers {workers} jobs {len(jobs) + 1}", flush=True)
    ctx = get_context("spawn")
    with ctx.Pool(int(workers)) as pool:
        asyncs = [pool.apply_async(_job, (job,)) for job in jobs]
        rows = [item.get() for item in asyncs]
        slept = [row for row in rows if row["controller"] == "prog_sleep"]
        if len(slept) != 1:
            raise SystemExit("prog_sleep did not return one row")
        fixed = pool.apply(_job_fixed, ((seed, SCORED_L, slept[0]["sleep_decisions"]),))
        rows.append(fixed)
    wall = time.perf_counter() - t0
    if len(rows) != len(CONTROLLERS):
        raise SystemExit(f"expected {len(CONTROLLERS)} rows, got {len(rows)}")
    for row in rows:
        if row["seed"] != seed or row["L"] != SCORED_L:
            raise SystemExit("row seed or L mismatch")
        if not row["finite"]:
            raise SystemExit(f'{row["controller"]} is not finite')
        if row["overlap"] != 0 or row["exam_glances_drawn"] != 0:
            raise SystemExit(f'{row["controller"]} store check failed')
    if fixed["sleep_decisions"] != slept[0]["sleep_decisions"]:
        raise SystemExit("fixed_sleep dose mismatch")
    payload = {
        "kernel_id": kernel_id,
        "seed": seed,
        "workers": int(workers),
        "wall_s": wall,
        "torch": torch.__version__,
        "model_seed": MODEL_SEED,
        "runs": rows,
    }
    path = out / f"m7_s{seed}.json"
    dump(path, payload)
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


def _reference_block(path, L, controllers):
    payload = json.loads(Path(path).read_text())
    runs = [row for row in payload["runs"] if row["L"] == L and row["controller"] in controllers]
    out = {}
    for controller in controllers:
        rows = sorted((row for row in runs if row["controller"] == controller), key=lambda row: row["seed"])
        w_mu, w_sd = mean_std(row["W"] for row in rows)
        w9_mu, w9_sd = mean_std(row["W9"] for row in rows)
        out[controller] = {
            "seeds": [row["seed"] for row in rows],
            "W": [row["W"] for row in rows],
            "W9": [row["W9"] for row in rows],
            "W_mean": w_mu,
            "W_std_ddof0": w_sd,
            "W9_mean": w9_mu,
            "W9_std_ddof0": w9_sd,
        }
    return out


def summarize(runs):
    by = {}
    for controller in CONTROLLERS:
        rows = sorted((row for row in runs if row["controller"] == controller), key=lambda row: row["seed"])
        if len(rows) != len(SEEDS):
            raise RuntimeError(f"{controller}: {len(rows)} rows, expected {len(SEEDS)}")
        w_mu, w_sd = mean_std(row["W"] for row in rows)
        w9_mu, w9_sd = mean_std(row["W9"] for row in rows)
        held_mu = {name: mean_std(row["held"][name] for row in rows)[0] for name in curio.LEARN}
        held9_mu = {name: mean_std(row["held9"][name] for row in rows)[0] for name in curio.CH}
        share_mu = {name: mean_std(row["share"][name] for row in rows)[0] for name in curio.CH}
        corr_mu = mean_std(row["corridor_share"] for row in rows)[0]
        sleep_mu, sleep_sd = mean_std(row["sleep_share"] for row in rows)
        quarter = []
        for i in range(4):
            quarter.append({
                "share": {name: mean_std(row["quarter"][i]["share"][name] for row in rows)[0] for name in curio.CH},
                "corridor": mean_std(row["quarter"][i]["corridor"] for row in rows)[0],
                "sleep": mean_std(row["quarter"][i]["sleep"] for row in rows)[0],
            })
        forgetting = {}
        for name in curio.LEARN:
            deltas = [
                row["forgetting"][name]["delta_bps"]
                for row in rows
                if row["forgetting"][name]["delta_bps"] is not None
            ]
            mu, sd = mean_std(deltas)
            forgetting[name] = {
                "n": len(deltas),
                "delta_bps_mean": mu,
                "delta_bps_std_ddof0": sd,
                "after_bps_mean": mean_std(
                    row["forgetting"][name]["after_bps"]
                    for row in rows
                    if row["forgetting"][name]["after_bps"] is not None
                )[0],
                "end_bps_mean": mean_std(
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
        mu, sd = mean_std(defined)
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
        total = _pick(runs, "prog_total", seed)
        rand = _pick(runs, "rand_dwell", seed)
        oracle = _pick(runs, "oracle7", seed)
        slept = _pick(runs, "prog_sleep", seed)
        fixed = _pick(runs, "fixed_sleep", seed)
        gaps.append({
            "seed": seed,
            "prog_total_minus_rand_dwell": total["W"] - rand["W"],
            "prog_total_minus_oracle7": total["W"] - oracle["W"],
            "prog_sleep_minus_prog_total": slept["W"] - total["W"],
            "prog_sleep_minus_fixed_sleep": slept["W"] - fixed["W"],
        })
    gap_summary = {}
    for key in (
        "prog_total_minus_rand_dwell",
        "prog_total_minus_oracle7",
        "prog_sleep_minus_prog_total",
        "prog_sleep_minus_fixed_sleep",
    ):
        values = [row[key] for row in gaps]
        mu, sd = mean_std(values)
        gap_summary[key] = {
            "mean": mu,
            "std_ddof0": sd,
            "left_lower": int(sum(value < 0.0 for value in values)),
            "n": len(values),
        }
    return {"by_controller": by, "gaps": gaps, "gap_summary": gap_summary}


def _header(seeds):
    return "| controller | " + " | ".join(f"seed {seed}" for seed in seeds) + " | mean ± std ddof=0 |"


def _sep(n):
    return "|---|" + "---:|" * n + "---:|"


def render_markdown(result):
    seeds = result["seeds"]
    runs = result["runs"]
    summary = result["summary"]
    lines = [
        "# M7-W3 bank cortex",
        "",
        "The W3 loop on the trace-learned roots bank. W is bits/symbol of the frozen bank on a fresh "
        "continuation of the 7 learnable channels, from a zero state, with that channel's embedding. "
        "W9 is that mean over all 9 room channels. Lower is better. Forgetting is end-of-life exam "
        "bits/symbol minus the exam bits/symbol just after the last visit to that channel. "
        "Positive forgetting is a worse code. Numbers only.",
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
        "On W.",
        "",
        "| seed | prog_total − rand_dwell | prog_total − oracle7 | prog_sleep − prog_total | prog_sleep − fixed_sleep |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["gaps"]:
        lines.append(
            f"| {row['seed']} | {_fmt(row['prog_total_minus_rand_dwell'])} | "
            f"{_fmt(row['prog_total_minus_oracle7'])} | {_fmt(row['prog_sleep_minus_prog_total'])} | "
            f"{_fmt(row['prog_sleep_minus_fixed_sleep'])} |"
        )
    labels = (
        ("prog_total_minus_rand_dwell", "prog_total − rand_dwell"),
        ("prog_total_minus_oracle7", "prog_total − oracle7"),
        ("prog_sleep_minus_prog_total", "prog_sleep − prog_total"),
        ("prog_sleep_minus_fixed_sleep", "prog_sleep − fixed_sleep"),
    )
    lines.append("| mean | " + " | ".join(_fmt(summary["gap_summary"][key]["mean"]) for key, _name in labels) + " |")
    lines.append("| std ddof=0 | " + " | ".join(_fmt(summary["gap_summary"][key]["std_ddof0"]) for key, _name in labels) + " |")
    lines.append(
        "| left lower (count) | "
        + " | ".join(
            f"{summary['gap_summary'][key]['left_lower']} of {summary['gap_summary'][key]['n']}"
            for key, _name in labels
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
    lines += ["", "## Sleep share", "", "Sleep decisions divided by L.", "", _header(seeds), _sep(len(seeds))]
    for controller in CONTROLLERS:
        cells = [_fmt(_pick(runs, controller, seed)["sleep_share"]) for seed in seeds]
        block = summary["by_controller"][controller]
        lines.append(
            f"| {controller} | " + " | ".join(cells)
            + f" | {_pm(block['sleep_mean'], block['sleep_std_ddof0'])} |"
        )
    lines += [
        "",
        "## prog_sleep sleep share by quarter",
        "",
        "Quarter i is decisions [(i−1)L/4, iL/4). The share is sleep decisions in the quarter over L/4.",
        "",
        "| seed | quarter 1 | quarter 2 | quarter 3 | quarter 4 |",
        "|---:|---:|---:|---:|---:|",
    ]
    sleep_rows = [_pick(runs, "prog_sleep", seed) for seed in seeds]
    quarter_means = []
    for i in range(4):
        quarter_means.append(mean_std(row["quarter"][i]["sleep"] for row in sleep_rows)[0])
    for row in sleep_rows:
        cells = [_fmt(row["quarter"][i]["sleep"]) for i in range(4)]
        lines.append(f"| {row['seed']} | " + " | ".join(cells) + " |")
    lines.append("| mean | " + " | ".join(_fmt(value) for value in quarter_means) + " |")
    lines += [
        "",
        "## prog_sleep visit rewards by quarter",
        "",
        "Exam-store bits saved per decision, mean over the visits whose first decision falls in the quarter. "
        "A quarter with no visits of that kind is blank. The mean row is the mean of the per-seed means, "
        "over seeds that have at least one such visit.",
        "",
        "| seed | quarter | sleep visits | sleep mean reward | room visits | room mean reward |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sleep_rows:
        for i, slot in enumerate(row["reward_quarter"]):
            lines.append(
                f"| {row['seed']} | {i + 1} | {slot['sleep_n']} | {_fmt(slot['sleep_mean'])} | "
                f"{slot['room_n']} | {_fmt(slot['room_mean'])} |"
            )
    for i in range(4):
        sleep_means = [row["reward_quarter"][i]["sleep_mean"] for row in sleep_rows if row["reward_quarter"][i]["sleep_n"]]
        room_means = [row["reward_quarter"][i]["room_mean"] for row in sleep_rows if row["reward_quarter"][i]["room_n"]]
        sleep_n = sum(row["reward_quarter"][i]["sleep_n"] for row in sleep_rows)
        room_n = sum(row["reward_quarter"][i]["room_n"] for row in sleep_rows)
        lines.append(
            f"| mean | {i + 1} | {sleep_n} | {_fmt(mean_std(sleep_means)[0])} | {room_n} | "
            f"{_fmt(mean_std(room_means)[0])} |"
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
    ref = result["reference"]
    lines += [
        "## Hashed learner reference",
        "",
        "M5b and M6b at L = 2048, the hashed table. Different seeds from this run.",
        "",
        "### M5b L = 2048",
        "",
    ]
    m5_seeds = ref["m5b"]["rand_dwell"]["seeds"]
    lines += [_header(m5_seeds), _sep(len(m5_seeds))]
    for controller, block in ref["m5b"].items():
        cells = [_fmt(value) for value in block["W"]]
        lines.append(f"| {controller} W | " + " | ".join(cells) + f" | {_pm(block['W_mean'], block['W_std_ddof0'])} |")
        cells = [_fmt(value) for value in block["W9"]]
        lines.append(f"| {controller} W9 | " + " | ".join(cells) + f" | {_pm(block['W9_mean'], block['W9_std_ddof0'])} |")
    lines += ["", "### M6b L = 2048", ""]
    m6_seeds = ref["m6b"]["prog_total"]["seeds"]
    lines += [_header(m6_seeds), _sep(len(m6_seeds))]
    for controller, block in ref["m6b"].items():
        cells = [_fmt(value) for value in block["W"]]
        lines.append(f"| {controller} W | " + " | ".join(cells) + f" | {_pm(block['W_mean'], block['W_std_ddof0'])} |")
        cells = [_fmt(value) for value in block["W9"]]
        lines.append(f"| {controller} W9 | " + " | ".join(cells) + f" | {_pm(block['W9_mean'], block['W9_std_ddof0'])} |")
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
        rooms = _pick(runs, "prog_total", seed)["room_channel"]
        lines.append(f"| {seed} | " + " | ".join(rooms) + " |")
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
        "- `BankCortex` is `h30y_roots.build_arm('roots_learn')` with model seed 0: N = 256, "
        "θ_k = 2πk/256, |λ| = 0.98, embedding 32, order-2 readout, Adam lr 1e-3, clip 1.0, float32. "
        "The trace step is the H30x rule. A 10-row channel embedding, initialised at zero, is added to "
        "the symbol embedding at every step. Rows follow `curio.CH` and then the corridor."
    )
    lines.append(
        "- One life is one stream. State and traces carry across glances. Each glance is 64 symbols: "
        "score it, then one Adam step. Sleep and stored glances run from a zero state, including zero traces, "
        "and use the stored previous symbol as the first input. A missing previous symbol, only at the start "
        "of life, uses a zero symbol embedding. Eval continuations also start from a zero symbol embedding."
    )
    lines.append(
        "- After a room glance is learned, one coin flip, `default_rng(40_000 + seed).integers(2) == 0`, "
        "sends the whole glance to the exam reservoir, and otherwise to the replay reservoir. Each holds "
        f"{R} glances per channel. Exam replacement uses `default_rng(20_000 + seed)`. Replay replacement "
        "uses `default_rng(50_000 + seed)`. The replacement rng is drawn only when that channel's reservoir "
        "is full. Corridor glances are learned and stored nowhere."
    )
    lines.append(
        "- A decision's progress is the sum of exam-glance bits before the Adam step minus the sum after it, "
        "on the glances already stored. The new glance is inserted after that score. The next decision reuses "
        "this after-score, plus the inserted glance if it went to the exam store. A visit reward is the sum "
        "of its decision progresses divided by its decisions. Travel counts."
    )
    lines.append(
        "- A sleep decision draws one glance uniformly from the union of the replay stores, "
        "`default_rng(30_000 + seed)`, and takes one trace Adam step from a zero state. An empty union "
        "learns nothing. The live state and traces are not written. A sleep visit is 8 such decisions and "
        "does not move. N's eval stream is a fresh `default_rng(30_000 + seed)`, not the sleep rng."
    )
    lines.append(
        "- `rand_dwell` and `oracle7` use `w3_headroom.next_action` and `default_rng(1_000 + seed)` for "
        "both the plan and N-channel glances. `prog_total` is the 9-room bandit. `prog_sleep` adds sleep "
        "as the last arm of the optimistic first pass, then eps-greedy with EPS=0.1 and BETA=0.3 over 10 arms. "
        "Choice rng is `default_rng(10_000 + seed)`. `fixed_sleep` is `prog_total` plus planned sleep visits "
        "whose lengths sum to that seed's `prog_sleep` sleep count."
    )
    lines.append(
        "- Forgetting snapshots the exam glances of a channel at the end of the last visit that dwelled there, "
        "and scores those same glances again at the end of life. W and W9 use EVAL=2048 symbols from a copy "
        "of the H20 world, learning off."
    )
    lines.append(
        f"- World seeds {', '.join(str(seed) for seed in seeds)}. L = {SCORED_L}. "
        "CPU. Private Kaggle kernels, one seed each, pool of up to 4, no GPU, no internet. Nothing retuned."
    )
    lines.append("")
    return "\n".join(lines)


def assemble(out_dir, smoke_payload):
    out_dir = Path(out_dir)
    runs = []
    kernels = []
    for seed in SEEDS:
        hits = sorted(out_dir.glob(f"s{seed}/m7_s{seed}.json")) + sorted(out_dir.glob(f"m7_s{seed}.json"))
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
    summary = summarize(runs)
    reference = {
        "m5b": _reference_block(ROOT / "results" / "m5bw3" / "results.json", 2048, ("rand_dwell", "oracle7", "prog")),
        "m6b": _reference_block(
            ROOT / "results" / "m6bw3" / "results.json", 2048,
            ("prog", "prog_total", "prog_sleep", "fixed_sleep"),
        ),
    }
    result = {
        "seeds": list(SEEDS),
        "L": SCORED_L,
        "controllers": list(CONTROLLERS),
        "model_seed": MODEL_SEED,
        "R": R,
        "kernels": kernels,
        "summary": summary,
        "reference": reference,
        "smoke": {key: smoke_payload[key] for key in smoke_payload if key != "log"},
        "smoke_log": smoke_payload["log"],
        "runs": runs,
    }
    text = render_markdown(result)
    dump(out_dir / "results.json", result)
    (out_dir / "RESULTS.md").write_text(text)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("smoke", "assemble"))
    parser.add_argument("--out", default=str(ROOT / "results" / "m7w3"))
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out)
        return
    smoke_payload = json.loads((Path(args.out) / "smoke.json").read_text())
    assemble(args.out, smoke_payload)


if __name__ == "__main__":
    main()
