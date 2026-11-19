"""W5: W4's ring, with a learnable room's rule replaced on a fixed schedule.

World4 is unchanged. A swap picks one of the 14 learnable rooms and points that
room at the same process in a new h20 world, read from the start of that world.
Noise rooms, Z, names and the ring order stay put. period None never swaps.
"""
import copy
import math

import numpy as np

import curio
import h20_world
import w4_world
from h20_world import ALPHA
from w4_world import LEARN14, World4, split_learnable


def finite_period(period):
    """None means no drift. Any other value is a positive decision count."""
    if period is None:
        return None
    if isinstance(period, str):
        if period.lower() in {"inf", "infinity", "∞"}:
            return None
        period = float(period)
    if isinstance(period, float) and math.isinf(period):
        return None
    period = int(period)
    if period <= 0:
        raise ValueError(f"period {period} is not positive")
    return period


class World5(World4):
    def __init__(self, world_seed, period):
        super().__init__(world_seed)
        self.period = finite_period(period)
        self.swap_k = 0
        self.swap_log = []
        self.swap_rng = np.random.default_rng(self.world_seed + 80_000)
        # Room name -> h20 World that now supplies that room's process.
        self.override = {}

    def maybe_swap(self, decision):
        """After decision `decision` (1-based). Swaps only when it is a positive multiple of period.

        The draw happens after that decision's observation. k counts swaps from 0.
        The new stream is h20_world.World(world_seed + 2000 + k), from its beginning.
        """
        if self.period is None:
            return None
        decision = int(decision)
        if decision <= 0 or decision % self.period != 0:
            return None
        if self.swap_log and decision == self.swap_log[-1]["decision"]:
            raise RuntimeError(f"second swap at decision {decision}")
        name = LEARN14[int(self.swap_rng.integers(len(LEARN14)))]
        k = int(self.swap_k)
        parsed = split_learnable(name)
        if parsed is None:
            raise RuntimeError(f"swap picked a room that is not learnable: {name}")
        self.override[name] = h20_world.World(self.world_seed + 2_000 + k)
        event = {"decision": decision, "room": name, "k": k}
        self.swap_log.append(event)
        self.swap_k = k + 1
        return event

    def glance_room(self, name):
        world = self.override.get(name)
        if world is None:
            return super().glance_room(name)
        proc, _side = split_learnable(name)
        return curio.glance(world, proc, None)

    def continuation(self, name, n):
        world = self.override.get(name)
        if world is None:
            return super().continuation(name, n)
        proc, _side = split_learnable(name)
        text = world.take(proc, int(n))
        if len(text) != int(n):
            raise RuntimeError(f"{name} continuation length {len(text)}")
        return np.asarray([ALPHA.index(s) for s in text], dtype=np.int64)

    def copy(self):
        return copy.deepcopy(self)
