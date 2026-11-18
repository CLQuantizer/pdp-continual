"""W6: W4's ring, where a room can die and a new room can be born in its slot.

World4, World5 and h20_world are unchanged. birth_period None never births, and
the glances are World4's. A birth keeps the slot's ring position and the cortex
channel (the initial room name). The dark room Z is never chosen.
"""
import copy

import numpy as np

import curio
import h20_world
import w4_world
import w5_world
from h20_world import ALPHA, PROCS
from w4_world import DARK, IDENTITIES, N_RING, NOISE, PROC_SET, World4, split_learnable
from w5_world import finite_period

KINDS = list(PROCS) + ["N"]

if len(KINDS) != 8 or len(set(KINDS)) != 8:
    raise RuntimeError("a newborn kind is not one of 8")
if len(ALPHA) != 41:
    raise RuntimeError("noise alphabet is not 41 symbols")


def kind_of(name):
    """Process name, N for a noise room, Z for the dark room."""
    parsed = split_learnable(name)
    if parsed is not None:
        return parsed[0]
    if name in NOISE:
        return "N"
    if name == DARK:
        return "Z"
    raise ValueError(name)


def expected_birth_log(world_seed, birth_period, L):
    """Independent of World6. A clock bug cannot confirm itself.

    The ring is World4's. The draw is default_rng(world_seed + 90000): a slot
    uniform on the 17 non-Z positions in ring order, then a kind uniform on
    PROCS + ["N"].
    """
    period = finite_period(birth_period)
    if period is None:
        return []
    rooms = list(World4(int(world_seed)).rooms)
    eligible = [i for i in range(N_RING) if rooms[i] != DARK]
    if len(eligible) != 17:
        raise RuntimeError("expected 17 slots that are not Z")
    rng = np.random.default_rng(int(world_seed) + 90_000)
    current = [kind_of(name) for name in rooms]
    out = []
    for k, decision in enumerate(range(int(period), int(L) + 1, int(period))):
        slot = eligible[int(rng.integers(len(eligible)))]
        kind = KINDS[int(rng.integers(len(KINDS)))]
        out.append({
            "decision": int(decision),
            "slot": int(slot),
            "dead_kind": current[slot],
            "newborn_kind": kind,
            "k": int(k),
        })
        current[slot] = kind
    return out


class World6(World4):
    def __init__(self, world_seed, birth_period):
        super().__init__(world_seed)
        self.birth_period = finite_period(birth_period)
        self.birth_k = 0
        self.birth_log = []
        # Drawn only when a birth happens. Period None never reads it.
        self.birth_rng = np.random.default_rng(self.world_seed + 90_000)
        self.kind = [kind_of(name) for name in self.rooms]
        self.born_at = [0] * N_RING
        # None means this slot still reads World4's stream. A birth replaces it.
        self.source = [None] * N_RING
        self.eligible = [i for i in range(N_RING) if self.rooms[i] != DARK]
        if len(self.eligible) != 17:
            raise RuntimeError("expected 17 slots that are not Z")
        if self.kind.count("Z") != 1:
            raise RuntimeError("expected one dark room")
        if list(IDENTITIES) != list(w4_world.IDENTITIES):
            raise RuntimeError("room identities drifted")

    def is_learnable(self, slot):
        return self.kind[int(slot)] in PROC_SET

    def learnable_slots(self):
        return [i for i in range(N_RING) if self.is_learnable(i)]

    def maybe_birth(self, decision):
        """After decision `decision` (1-based). Births only on positive multiples of the period.

        k counts births from 0. A learnable newborn is h20_world.World(world_seed + 3000 + k)
        from its beginning. Kind N is uniform on 41 symbols, rng world_seed + 95000 + k.
        The slot's ring position and rooms[] entry (the cortex channel) stay put.
        """
        if self.birth_period is None:
            return None
        decision = int(decision)
        if decision <= 0 or decision % self.birth_period != 0:
            return None
        if self.birth_log and decision == self.birth_log[-1]["decision"]:
            raise RuntimeError(f"second birth at decision {decision}")
        slot = self.eligible[int(self.birth_rng.integers(len(self.eligible)))]
        kind = KINDS[int(self.birth_rng.integers(len(KINDS)))]
        k = int(self.birth_k)
        if self.rooms[slot] == DARK or self.kind[slot] == "Z":
            raise RuntimeError("a birth chose the dark room")
        dead = self.kind[slot]
        if kind == "N":
            self.source[slot] = {
                "kind": "N",
                "rng": np.random.default_rng(self.world_seed + 95_000 + k),
            }
        elif kind in PROC_SET:
            self.source[slot] = {
                "kind": kind,
                "world": h20_world.World(self.world_seed + 3_000 + k),
            }
        else:
            raise RuntimeError(f"newborn kind {kind}")
        self.kind[slot] = kind
        self.born_at[slot] = decision
        event = {
            "decision": decision,
            "slot": int(slot),
            "dead_kind": dead,
            "newborn_kind": kind,
            "k": k,
        }
        self.birth_log.append(event)
        self.birth_k = k + 1
        return event

    def glance_slot(self, slot):
        slot = int(slot)
        src = self.source[slot]
        if src is None:
            return super().glance_room(self.rooms[slot])
        if src["kind"] == "N":
            return curio.glance(None, "N", src["rng"])
        return curio.glance(src["world"], src["kind"], None)

    def glance_room(self, name):
        return self.glance_slot(self.rooms.index(name))

    def continuation_slot(self, slot, n):
        """n fresh symbols of the slot's current learnable room. Advances this stream."""
        slot = int(slot)
        if not self.is_learnable(slot):
            raise ValueError(f"slot {slot} kind {self.kind[slot]} is not learnable")
        src = self.source[slot]
        n = int(n)
        if src is None:
            return super().continuation(self.rooms[slot], n)
        text = src["world"].take(src["kind"], n)
        if len(text) != n:
            raise RuntimeError(f"slot {slot} continuation length {len(text)}")
        return np.asarray([ALPHA.index(s) for s in text], dtype=np.int64)

    def continuation(self, name, n):
        return self.continuation_slot(self.rooms.index(name), n)

    def step(self, action, rng):
        """Stay observes the slot's current room on the slot's original channel name."""
        if action != "stay":
            return super().step(action, rng)
        del rng
        channel = self.rooms[self.room]
        return self.room, channel, self.glance_slot(self.room)

    def copy(self):
        return copy.deepcopy(self)
