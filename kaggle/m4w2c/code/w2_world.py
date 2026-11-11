"""W2 locked-rooms world, fixed policies, and probe episodes. Numbers only.

Deterministic given world_seed. No learner and no verdict. Does not commit.
"""
import hashlib

import numpy as np

N_ROOMS = 9
DEPTH = 8
N_MANIP = 6
N_ACTIONS = 8
TV = 6
RESET = 7
Y_MAX = 16
VOCAB = 152
OBS_BASE = 8
KEY_LEN = 3
ROOM8_STATES = 6
POLICY_SEED = 5_000
ORACLE_SEED = 6_000
NOISE_SEED = 10_000
PROBE_SEED = 999
PROBE_MANIPS = 96
POLICIES = ('rand8', 'rand6', 'oracle')


def obs_token(room, y):
    return OBS_BASE + Y_MAX * int(room) + int(y)


def decode_obs(token):
    value = int(token) - OBS_BASE
    return value // Y_MAX, value % Y_MAX


def token_sha256(tokens):
    raw = np.ascontiguousarray(tokens, dtype=np.uint16)
    return hashlib.sha256(raw.tobytes()).hexdigest()


class World:
    """Rooms 0..8. Lock rooms 0..7, then a 6-state machine in room 8.

    Tables are drawn from default_rng(world_seed) in the spec order:
    key_0..key_7, table_0..table_7, out, next. The noise rng is separate.
    """

    def __init__(self, world_seed):
        self.world_seed = int(world_seed)
        rng = np.random.default_rng(self.world_seed)
        self.keys = np.zeros((DEPTH, KEY_LEN), dtype=np.int64)
        for k in range(DEPTH):
            for j in range(KEY_LEN):
                self.keys[k, j] = int(rng.integers(0, N_MANIP))
        self.tables = np.zeros((DEPTH, KEY_LEN, N_MANIP), dtype=np.int64)
        for k in range(DEPTH):
            for s in range(KEY_LEN):
                for a in range(N_MANIP):
                    self.tables[k, s, a] = int(rng.integers(0, Y_MAX))
        self.out = np.zeros((ROOM8_STATES, N_MANIP), dtype=np.int64)
        self.nxt = np.zeros((ROOM8_STATES, N_MANIP), dtype=np.int64)
        for q in range(ROOM8_STATES):
            for a in range(N_MANIP):
                self.out[q, a] = int(rng.integers(0, Y_MAX))
        for q in range(ROOM8_STATES):
            for a in range(N_MANIP):
                self.nxt[q, a] = int(rng.integers(0, N_MANIP))
        self.reset_agent()

    def reset_agent(self):
        self.room = 0
        self.s = 0
        self.q = 0
        self.noise = np.random.default_rng(NOISE_SEED + self.world_seed)

    def step(self, action):
        a = int(action)
        if a == TV:
            y = int(self.noise.integers(0, Y_MAX))
        elif a == RESET:
            self.room = 0
            self.s = 0
            self.q = 0
            y = 0
        elif 0 <= a < N_MANIP:
            y = self._manipulate(a)
        else:
            raise ValueError(f'action {a} is outside 0..7')
        if not (0 <= y < Y_MAX) or not (0 <= self.room < N_ROOMS):
            raise AssertionError(f'bad observation room={self.room} y={y}')
        return self.room, y

    def _manipulate(self, a):
        if self.room < DEPTH:
            k = self.room
            s = self.s
            if not (0 <= s < KEY_LEN):
                raise AssertionError(f'lock state {s} in room {k}')
            y = int(self.tables[k, s, a])
            if a == int(self.keys[k, s]):
                new_s = s + 1
            else:
                new_s = 1 if a == int(self.keys[k, 0]) else 0
            if new_s == KEY_LEN:
                self.room = k + 1
                self.s = 0
                if self.room == DEPTH:
                    self.q = 0
            else:
                self.s = new_s
            return y
        q = self.q
        y = int(self.out[q, a])
        self.q = int(self.nxt[q, a])
        return y


def _stay_action(world, rng):
    """Uniform over manipulations. In a lock room with s == 2, drop key[2]."""
    if world.room < DEPTH and world.s == 2:
        ban = int(world.keys[world.room, 2])
        draw = int(rng.integers(0, N_MANIP - 1))
        return draw if draw < ban else draw + 1
    return int(rng.integers(0, N_MANIP))


def _oracle_action(world, t, block_size, explore_rng):
    target = (t // block_size) % N_ROOMS
    pos = t % block_size
    nav = 1 + KEY_LEN * target
    if nav > block_size:
        raise AssertionError(f'oracle nav {nav} exceeds block {block_size}')
    if pos == 0:
        return RESET
    if pos < nav:
        room_i = (pos - 1) // KEY_LEN
        offset = (pos - 1) % KEY_LEN
        return int(world.keys[room_i, offset])
    return _stay_action(world, explore_rng)


def rollout(world_seed, policy, steps, block_size):
    """One fixed-policy life. Returns tokens (action, observation) and counts."""
    if policy not in POLICIES:
        raise ValueError(policy)
    world = World(world_seed)
    actions = np.empty(steps, dtype=np.int64)
    rooms = np.empty(steps, dtype=np.int64)
    ys = np.empty(steps, dtype=np.int64)
    tokens = np.empty(steps * 2, dtype=np.int64)
    if policy == 'rand8':
        actions[:] = np.random.default_rng(POLICY_SEED + world_seed).integers(0, N_ACTIONS, size=steps)
    elif policy == 'rand6':
        actions[:] = np.random.default_rng(POLICY_SEED + world_seed).integers(0, N_MANIP, size=steps)
    explore_rng = None
    if policy == 'oracle':
        explore_rng = np.random.default_rng(ORACLE_SEED + world_seed)
    counts = np.zeros(N_ROOMS, dtype=np.int64)
    deepest = -1
    first_step = -1
    for t in range(steps):
        if policy == 'oracle':
            actions[t] = _oracle_action(world, t, block_size, explore_rng)
        room, y = world.step(actions[t])
        rooms[t] = room
        ys[t] = y
        tokens[2 * t] = int(actions[t])
        tokens[2 * t + 1] = obs_token(room, y)
        counts[room] += 1
        if room > deepest:
            deepest = int(room)
            first_step = int(t)
    return {
        'world_seed': int(world_seed),
        'policy': policy,
        'steps': int(steps),
        'block_size': int(block_size),
        'actions': actions,
        'rooms': rooms,
        'ys': ys,
        'tokens': tokens,
        'time_in_room': [int(v) for v in counts],
        'deepest_room': int(deepest),
        'deepest_first_step': int(first_step),
        'token_sha256': token_sha256(tokens),
    }


def _append_step(world, action, actions, rooms, ys):
    room, y = world.step(action)
    actions.append(int(action))
    rooms.append(int(room))
    ys.append(int(y))
    return room, y


def room_probe_episode(world_seed, room_k):
    """Reset, walk the keys to room_k (unscored), then 96 staying manipulations."""
    world = World(world_seed)
    actions = []
    rooms = []
    ys = []
    _append_step(world, RESET, actions, rooms, ys)
    for k in range(room_k):
        for j in range(KEY_LEN):
            _append_step(world, int(world.keys[k, j]), actions, rooms, ys)
    if world.room != room_k or world.s != 0:
        raise AssertionError(
            f'probe path seed={world_seed} room={world.room} s={world.s} target={room_k}')
    if room_k == DEPTH and world.q != 0:
        raise AssertionError('room 8 probe did not enter at q=0')
    prefix = len(actions)
    rng = np.random.default_rng(PROBE_SEED + room_k)
    for _ in range(PROBE_MANIPS):
        action = _stay_action(world, rng)
        room, _y = _append_step(world, action, actions, rooms, ys)
        if room != room_k:
            raise AssertionError(f'probe left room {room_k} for {room}')
    tokens = np.empty(len(actions) * 2, dtype=np.int64)
    for i, (action, room, y) in enumerate(zip(actions, rooms, ys)):
        tokens[2 * i] = action
        tokens[2 * i + 1] = obs_token(room, y)
    scored = np.arange(prefix, len(actions), dtype=np.int64)
    if len(scored) != PROBE_MANIPS:
        raise AssertionError(f'probe scored {len(scored)} steps')
    return {
        'room': int(room_k),
        'tokens': tokens,
        'prefix_steps': int(prefix),
        'scored_steps': scored,
    }


def tv_probe_episode(world_seed):
    """96 TV presses from the initial room-0 state. No reset prefix."""
    world = World(world_seed)
    tokens = np.empty(PROBE_MANIPS * 2, dtype=np.int64)
    for i in range(PROBE_MANIPS):
        room, y = world.step(TV)
        if room != 0 or world.s != 0 or world.q != 0:
            raise AssertionError('TV probe moved the world')
        tokens[2 * i] = TV
        tokens[2 * i + 1] = obs_token(room, y)
    return {
        'tokens': tokens,
        'prefix_steps': 0,
        'scored_steps': np.arange(PROBE_MANIPS, dtype=np.int64),
    }


def build_probe_episodes(world_seed):
    return {
        'rooms': [room_probe_episode(world_seed, k) for k in range(N_ROOMS)],
        'tv': tv_probe_episode(world_seed),
    }


def _play_keys(world_seed):
    world = World(world_seed)
    for k in range(DEPTH):
        for j in range(KEY_LEN):
            world.step(int(world.keys[k, j]))
        if world.room != k + 1 or world.s != 0:
            raise AssertionError(f'after key_{k} room={world.room} s={world.s}')
        if world.room == DEPTH and world.q != 0:
            raise AssertionError('entry to room 8 did not set q=0')
    return world


def smoke_world(steps, block_size, world_seed=0):
    """World checks for one seed at the smoke length, plus the 24-step key path."""
    checks = {}
    first = rollout(world_seed, 'rand8', steps, block_size)
    second = rollout(world_seed, 'rand8', steps, block_size)
    checks['rand8_checksum_match'] = first['token_sha256'] == second['token_sha256']
    if not checks['rand8_checksum_match']:
        raise AssertionError('rand8 token checksums differ')
    checks['rand8_token_sha256'] = first['token_sha256']

    rows = {}
    for policy in POLICIES:
        life = first if policy == 'rand8' else rollout(world_seed, policy, steps, block_size)
        if sum(life['time_in_room']) != steps:
            raise AssertionError(f'{policy} time-in-room does not sum to T')
        if np.any(life['tokens'][0::2] != life['actions']):
            raise AssertionError(f'{policy} action tokens drifted')
        obs = life['tokens'][1::2]
        if np.any((obs < OBS_BASE) | (obs >= VOCAB)):
            raise AssertionError(f'{policy} observation token out of range')
        decoded = [decode_obs(tok) for tok in obs]
        if decoded[0] != (int(life['rooms'][0]), int(life['ys'][0])):
            raise AssertionError('observation decode mismatch')
        print(
            f"smoke policy={policy} time_in_room={life['time_in_room']} "
            f"deepest={life['deepest_room']} first_step={life['deepest_first_step']}",
            flush=True)
        rows[policy] = {
            'time_in_room': life['time_in_room'],
            'deepest_room': life['deepest_room'],
            'deepest_first_step': life['deepest_first_step'],
            'token_sha256': life['token_sha256'],
        }
    oracle = rows['oracle']
    checks['oracle_visits_all_rooms'] = all(v > 0 for v in oracle['time_in_room'])
    if not checks['oracle_visits_all_rooms']:
        raise AssertionError(f"oracle missed a room: {oracle['time_in_room']}")
    oracle_life = rollout(world_seed, 'oracle', steps, block_size)
    if np.any(oracle_life['actions'] == TV):
        raise AssertionError('oracle used the TV')
    _assert_oracle_stays(oracle_life)
    room8_at = oracle_room8_step(block_size)
    if steps > room8_at:
        if oracle['deepest_room'] != DEPTH or oracle['deepest_first_step'] != room8_at:
            raise AssertionError(
                f"oracle reached room {oracle['deepest_room']} "
                f"at step {oracle['deepest_first_step']}, expected room 8 at {room8_at}")
        checks['oracle_room8_first_step'] = room8_at
    rand6 = rollout(world_seed, 'rand6', steps, block_size)
    if np.any(rand6['actions'] >= N_MANIP):
        raise AssertionError('rand6 used TV or reset')

    world = _play_keys(world_seed)
    checks['keys_reach_room_8_in_24'] = world.room == DEPTH
    if world.room != DEPTH:
        raise AssertionError(f'24 key steps ended in room {world.room}')
    # The entry step is the lock's output. The next manipulation is the machine.
    q_before = world.q
    action0 = 0
    expect_y = int(world.out[q_before, action0])
    expect_q = int(world.nxt[q_before, action0])
    room, y = world.step(action0)
    if room != DEPTH or y != expect_y or world.q != expect_q:
        raise AssertionError('room 8 machine did not use q at entry')
    _entry_clears_q(world_seed)
    _lock_and_tv_checks(world_seed)
    checks['world_seed'] = int(world_seed)
    checks['steps'] = int(steps)
    checks['block_size'] = int(block_size)
    checks['policies'] = rows
    print(
        f"smoke world ok sha256={checks['rand8_token_sha256']} "
        f"oracle_rooms={oracle['time_in_room']}",
        flush=True)
    return checks


def _assert_oracle_stays(life):
    steps = life['steps']
    block = life['block_size']
    rooms = life['rooms']
    for start in range(0, steps, block):
        target = (start // block) % N_ROOMS
        end = min(start + block, steps)
        nav = 1 + KEY_LEN * target
        arrived = start + nav - 1
        if arrived >= end:
            raise AssertionError(f'block at {start} is shorter than navigation')
        if int(rooms[arrived]) != target:
            raise AssertionError(
                f'oracle block {start} arrived in {int(rooms[arrived])}, target {target}')
        for i in range(arrived, end):
            if int(rooms[i]) != target:
                raise AssertionError(f'oracle left room {target} at step {i}')


def _entry_clears_q(world_seed):
    """A dirty q must still be 0 on the step that enters room 8."""
    world = World(world_seed)
    for k in range(DEPTH - 1):
        for j in range(KEY_LEN):
            world.step(int(world.keys[k, j]))
    if world.room != DEPTH - 1 or world.s != 0:
        raise AssertionError(f'expected room 7 before the last key, got {world.room}')
    world.q = 4
    for j in range(KEY_LEN):
        world.step(int(world.keys[DEPTH - 1, j]))
    if world.room != DEPTH or world.q != 0:
        raise AssertionError(f'entry left room={world.room} q={world.q}')


def _lock_and_tv_checks(world_seed):
    world = World(world_seed)
    key0 = int(world.keys[0, 0])
    wrong = (key0 + 1) % N_MANIP
    room, y = world.step(wrong)
    if room != 0 or world.s != 0 or y != int(world.tables[0, 0, wrong]):
        raise AssertionError('wrong first action did not hold the lock at 0')
    room, y = world.step(key0)
    if room != 0 or world.s != 1 or y != int(world.tables[0, 0, key0]):
        raise AssertionError('key[0] did not advance s to 1')
    key1 = int(world.keys[0, 1])
    other = None
    for action in range(N_MANIP):
        if action != key0 and action != key1:
            other = action
            break
    if other is not None:
        room, _y = world.step(other)
        if room != 0 or world.s != 0:
            raise AssertionError('non-key action did not return s to 0')
        world.step(key0)
    if key0 != key1:
        room, _y = world.step(key0)
        if room != 0 or world.s != 1:
            raise AssertionError('repeating key[0] at s=1 did not stay at s=1')
    room_before, s_before, q_before = world.room, world.s, world.q
    room, y = world.step(TV)
    if room != room_before or world.s != s_before or world.q != q_before:
        raise AssertionError('TV changed the lock state')
    if not (0 <= y < Y_MAX):
        raise AssertionError('TV y out of range')
    room, y = world.step(RESET)
    if (room, world.s, world.q, y) != (0, 0, 0, 0):
        raise AssertionError('reset did not clear the agent')


def oracle_room8_step(block_size):
    """First step index at which the oracle's post-action room is 8."""
    return DEPTH * block_size + KEY_LEN * DEPTH
