"""W3: the nine H23c channels as rooms on a ring. Moving costs a corridor glance.

A room's channel is one of curio.CH. Streams come from one h20_world.World and curio.glance,
so a channel resumes where it left off. The agent starts in room 0.
"""
import numpy as np

import curio
import h20_world


class World3:
    def __init__(self, world_seed):
        self.world_seed = int(world_seed)
        self.h20 = h20_world.World(self.world_seed)
        perm = np.random.default_rng(self.world_seed + 70_000).permutation(9)
        self.perm = [int(i) for i in perm]
        self.rooms = [curio.CH[i] for i in self.perm]   # ring index -> channel
        self.room = 0

    def glance(self, channel, rng):
        """One glance. Corridor is 64 spaces and does not draw from rng. Other channels use curio.glance."""
        if channel == "corridor":
            return [h20_world.ALPHA.index(" ")] * curio.B
        return curio.glance(self.h20, channel, rng)

    def step(self, action, rng):
        """One decision. Returns (room_after, glance_channel, symbols).

        stay observes the current room. left/right move one room and observe the corridor.
        """
        if action == "stay":
            channel = self.rooms[self.room]
            return self.room, channel, self.glance(channel, rng)
        if action == "left":
            self.room = (self.room - 1) % 9
        elif action == "right":
            self.room = (self.room + 1) % 9
        else:
            raise ValueError(action)
        return self.room, "corridor", self.glance("corridor", rng)
