"""W4: eighteen rooms on a ring, sized to the bank cortex. Moving costs a corridor glance.

Fourteen learnable rooms are the seven H20 processes from two independent worlds.
Three rooms are noise and one is dark. A room's stream resumes where it left off.
The agent starts in room 0.
"""
import copy

import numpy as np

import curio
import h20_world
from h20_world import ALPHA, PROCS

N_RING = 18
LEARN_A = [p + "a" for p in PROCS]
LEARN_B = [p + "b" for p in PROCS]
NOISE = ["N1", "N2", "N3"]
DARK = "Z"
IDENTITIES = LEARN_A + LEARN_B + NOISE + [DARK]
LEARN14 = LEARN_A + LEARN_B
PROC_SET = set(PROCS)
SPACE = ALPHA.index(" ")

if len(IDENTITIES) != N_RING or len(set(IDENTITIES)) != N_RING:
    raise RuntimeError("W4 does not have 18 distinct rooms")
if len(LEARN14) != 14:
    raise RuntimeError("W4 does not have 14 learnable rooms")


def split_learnable(name):
    """Return (process, side) for a learnable room, or None."""
    if len(name) >= 2 and name[-1] in ("a", "b") and name[:-1] in PROC_SET:
        return name[:-1], name[-1]
    return None


def shortest(src, dst, n):
    """Shortest walk on a ring. An equal split, which n=18 allows, goes left.

    Same comparison as w3_headroom.shortest. For n=9 the shortest walk is unique.
    """
    src, dst, n = int(src), int(dst), int(n)
    right = (dst - src) % n
    left = (src - dst) % n
    if right == 0:
        return []
    if right < left:
        return ["right"] * right
    return ["left"] * left


class World4:
    def __init__(self, world_seed):
        self.world_seed = int(world_seed)
        self.h20a = h20_world.World(self.world_seed)
        self.h20b = h20_world.World(self.world_seed + 1_000)
        # k = 0, 1, 2 for N1, N2, N3. Each stream has its own rng and resumes.
        self.noise_rng = [
            np.random.default_rng(self.world_seed + 60_000 + k) for k in range(3)
        ]
        perm = np.random.default_rng(self.world_seed + 70_000).permutation(N_RING)
        self.perm = [int(i) for i in perm]
        self.rooms = [IDENTITIES[i] for i in self.perm]
        self.room = 0
        if len(set(self.rooms)) != N_RING:
            raise RuntimeError("ring permutation dropped a room")

    def _world_for(self, side):
        if side == "a":
            return self.h20a
        if side == "b":
            return self.h20b
        raise ValueError(side)

    def glance_room(self, name):
        """One 64-symbol glance. Learnable rooms use curio.glance on their world."""
        parsed = split_learnable(name)
        if parsed is not None:
            proc, side = parsed
            return curio.glance(self._world_for(side), proc, None)
        if name in NOISE:
            k = NOISE.index(name)
            return curio.glance(None, "N", self.noise_rng[k])
        if name == DARK:
            return curio.glance(None, "Z", None)
        raise ValueError(name)

    def continuation(self, name, n):
        """n fresh symbols of a learnable room, continuing this world's stream."""
        parsed = split_learnable(name)
        if parsed is None:
            raise ValueError(name)
        proc, side = parsed
        text = self._world_for(side).take(proc, int(n))
        if len(text) != int(n):
            raise RuntimeError(f"{name} continuation length {len(text)}")
        return np.asarray([ALPHA.index(s) for s in text], dtype=np.int64)

    def step(self, action, rng):
        """One decision. Returns (room_after, glance_channel, symbols).

        stay observes the current room. left/right move one room and observe the
        corridor. rng is unused: noise rooms have their own generators, and the
        corridor is 64 spaces. The argument is kept so the call matches World3.
        """
        del rng
        if action == "stay":
            channel = self.rooms[self.room]
            return self.room, channel, self.glance_room(channel)
        if action == "left":
            self.room = (self.room - 1) % N_RING
        elif action == "right":
            self.room = (self.room + 1) % N_RING
        else:
            raise ValueError(action)
        return self.room, "corridor", [SPACE] * curio.B

    def copy(self):
        return copy.deepcopy(self)
