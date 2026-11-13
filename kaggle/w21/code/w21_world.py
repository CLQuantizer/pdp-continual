"""W2-1 locked rooms with a way back from room 8. Numbers only.

World21 is World with one extra lock: key_8, drawn from its own rng, returns
the agent to room 0. w2_world.py is imported and not modified. No learner
and no verdict. Does not commit.
"""
import numpy as np

import w2_world as w2

N_ROOMS = w2.N_ROOMS
DEPTH = w2.DEPTH
N_MANIP = w2.N_MANIP
N_ACTIONS = w2.N_ACTIONS
TV = w2.TV
RESET = w2.RESET
Y_MAX = w2.Y_MAX
VOCAB = w2.VOCAB
OBS_BASE = w2.OBS_BASE
KEY_LEN = w2.KEY_LEN
ROOM8_STATES = w2.ROOM8_STATES
POLICY_SEED = w2.POLICY_SEED
ORACLE_SEED = w2.ORACLE_SEED
NOISE_SEED = w2.NOISE_SEED
PROBE_SEED = w2.PROBE_SEED
PROBE_MANIPS = w2.PROBE_MANIPS
KEY8_SEED = 50_000
POLICIES = ('rand8', 'rand6', 'oracle', 'oracle_deep')

obs_token = w2.obs_token
decode_obs = w2.decode_obs
token_sha256 = w2.token_sha256


class World21(w2.World):
    """Same tables as World. Room 8 has a 3-digit key back to room 0."""

    def __init__(self, world_seed):
        super().__init__(world_seed)
        # Separate rng, so every table World drew is unchanged.
        rng = np.random.default_rng(self.world_seed + KEY8_SEED)
        self.key_8 = np.asarray(rng.integers(0, 6, size=3), dtype=np.int64)

    def reset_agent(self):
        super().reset_agent()
        self.s8 = 0

    def step(self, action):
        room, y = super().step(action)
        if int(action) == RESET:
            self.s8 = 0
        return room, y

    def _manipulate(self, a):
        if self.room < DEPTH:
            return super()._manipulate(a)
        s8 = int(self.s8)
        if not 0 <= s8 < KEY_LEN:
            raise AssertionError(f'room 8 lock state {s8}')
        q = self.q
        y = int(self.out[q, a])
        self.q = int(self.nxt[q, a])
        if a == int(self.key_8[s8]):
            new_s8 = s8 + 1
        else:
            new_s8 = 1 if a == int(self.key_8[0]) else 0
        if new_s8 == KEY_LEN:
            # y is already the machine output. The observation room is 0.
            self.room = 0
            self.s = 0
            self.q = 0
            self.s8 = 0
        else:
            self.s8 = new_s8
        return y


def _stay_room8(world, rng):
    """Oracle stay rule, plus drop key_8[2] when s8 == 2."""
    if world.room < DEPTH and world.s == 2:
        ban = int(world.keys[world.room, 2])
        draw = int(rng.integers(0, N_MANIP - 1))
        return draw if draw < ban else draw + 1
    if world.room == DEPTH and world.s8 == 2:
        ban = int(world.key_8[2])
        draw = int(rng.integers(0, N_MANIP - 1))
        return draw if draw < ban else draw + 1
    return int(rng.integers(0, N_MANIP))


def oracle_deep_target(block_index):
    """Even blocks visit room 8. Odd block b visits (b // 2) mod 8."""
    block_index = int(block_index)
    if block_index % 2 == 0:
        return DEPTH
    return (block_index // 2) % DEPTH


def _oracle_deep_action(world, t, block_size, explore_rng):
    target = oracle_deep_target(t // block_size)
    pos = t % block_size
    nav = 1 + KEY_LEN * target
    if nav > block_size:
        raise AssertionError(f'oracle_deep nav {nav} exceeds block {block_size}')
    if pos == 0:
        return RESET
    if pos < nav:
        room_i = (pos - 1) // KEY_LEN
        offset = (pos - 1) % KEY_LEN
        return int(world.keys[room_i, offset])
    return _stay_room8(world, explore_rng)


def rollout(world_seed, policy, steps, block_size):
    """One fixed-policy life on World21. Returns tokens and room counts."""
    if policy not in POLICIES:
        raise ValueError(policy)
    world = World21(world_seed)
    actions = np.empty(steps, dtype=np.int64)
    rooms = np.empty(steps, dtype=np.int64)
    ys = np.empty(steps, dtype=np.int64)
    tokens = np.empty(steps * 2, dtype=np.int64)
    if policy == 'rand8':
        actions[:] = np.random.default_rng(POLICY_SEED + world_seed).integers(
            0, N_ACTIONS, size=steps)
    elif policy == 'rand6':
        actions[:] = np.random.default_rng(POLICY_SEED + world_seed).integers(
            0, N_MANIP, size=steps)
    explore_rng = None
    if policy in ('oracle', 'oracle_deep'):
        explore_rng = np.random.default_rng(ORACLE_SEED + world_seed)
    counts = np.zeros(N_ROOMS, dtype=np.int64)
    deepest = -1
    first_step = -1
    full_cycles = 0
    for t in range(steps):
        if policy == 'oracle':
            actions[t] = w2._oracle_action(world, t, block_size, explore_rng)
        elif policy == 'oracle_deep':
            actions[t] = _oracle_deep_action(world, t, block_size, explore_rng)
        before = int(world.room)
        room, y = world.step(int(actions[t]))
        if before == DEPTH and int(room) == 0 and int(actions[t]) != RESET:
            full_cycles += 1
        rooms[t] = room
        ys[t] = y
        tokens[2 * t] = int(actions[t])
        tokens[2 * t + 1] = obs_token(room, y)
        counts[room] += 1
        if room > deepest:
            deepest = int(room)
            first_step = int(t)
    if policy == 'oracle_deep' and full_cycles != 0:
        raise AssertionError(f'oracle_deep completed key_8 {full_cycles} times')
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
        'full_cycles': int(full_cycles),
        'token_sha256': token_sha256(tokens),
    }


def _append_step(world, action, actions, rooms, ys):
    room, y = world.step(action)
    actions.append(int(action))
    rooms.append(int(room))
    ys.append(int(y))
    return room, y


def room_probe_episode(world_seed, room_k):
    """Reset, walk the keys to room_k, then 96 staying manipulations.

    Room 8 uses the oracle_deep stay rule so the probe never leaves.
    """
    world = World21(world_seed)
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
    if room_k == DEPTH and (world.q != 0 or world.s8 != 0):
        raise AssertionError('room 8 probe did not enter at q=0 s8=0')
    prefix = len(actions)
    rng = np.random.default_rng(PROBE_SEED + room_k)
    for _ in range(PROBE_MANIPS):
        if room_k == DEPTH:
            action = _stay_room8(world, rng)
        else:
            action = w2._stay_action(world, rng)
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
    world = World21(world_seed)
    tokens = np.empty(PROBE_MANIPS * 2, dtype=np.int64)
    for i in range(PROBE_MANIPS):
        room, y = world.step(TV)
        if room != 0 or world.s != 0 or world.q != 0 or world.s8 != 0:
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


def room8_probe_diff(world_seed):
    """How many room-8 probe actions differ from W2-0. Prefix should match."""
    old = w2.room_probe_episode(world_seed, DEPTH)
    new = room_probe_episode(world_seed, DEPTH)
    old_actions = old['tokens'][0::2]
    new_actions = new['tokens'][0::2]
    if old_actions.shape != new_actions.shape:
        raise AssertionError(
            f'room 8 probe length {old_actions.shape} != {new_actions.shape}')
    prefix = int(old['prefix_steps'])
    if int(new['prefix_steps']) != prefix:
        raise AssertionError('room 8 probe prefix length changed')
    differ = np.flatnonzero(old_actions != new_actions)
    prefix_differ = int(np.sum(differ < prefix))
    scored_differ = int(np.sum(differ >= prefix))
    return {
        'world_seed': int(world_seed),
        'n_actions': int(old_actions.shape[0]),
        'prefix_steps': prefix,
        'n_differ': int(differ.shape[0]),
        'prefix_differ': prefix_differ,
        'scored_differ': scored_differ,
    }


def _tables_match(world_seed):
    new = World21(world_seed)
    old = w2.World(world_seed)
    if not (
        np.array_equal(new.keys, old.keys)
        and np.array_equal(new.tables, old.tables)
        and np.array_equal(new.out, old.out)
        and np.array_equal(new.nxt, old.nxt)
    ):
        raise AssertionError(f'world seed {world_seed} tables differ from World')
    if new.key_8.shape != (KEY_LEN,) or np.any((new.key_8 < 0) | (new.key_8 > 5)):
        raise AssertionError(f'key_8 {new.key_8} is not 3 digits in 0..5')
    return new.key_8


def equivalence_check(world_seed=0, steps=60_000):
    """rand6 tokens match World until the step that completes key_8."""
    _tables_match(world_seed)
    actions = np.random.default_rng(POLICY_SEED + world_seed).integers(
        0, N_MANIP, size=steps)
    agent = World21(world_seed)
    base = w2.World(world_seed)
    prefix = []
    completed = None
    for t in range(steps):
        if (agent.room, agent.s, agent.q) != (base.room, base.s, base.q):
            raise AssertionError(f'state diverged before step {t}')
        act = int(actions[t])
        completes = (
            agent.room == DEPTH
            and int(agent.s8) == KEY_LEN - 1
            and act == int(agent.key_8[agent.s8])
        )
        room_a, y_a = agent.step(act)
        room_b, y_b = base.step(act)
        if completes:
            if (room_a, agent.s, agent.q, agent.s8) != (0, 0, 0, 0):
                raise AssertionError(
                    f'key_8 completion left room={room_a} s={agent.s} '
                    f'q={agent.q} s8={agent.s8}')
            if room_b != DEPTH or y_a != y_b:
                raise AssertionError(
                    f'World did not stay in room 8 with the same y at step {t}')
            completed = t
            break
        if (room_a, y_a, agent.s, agent.q) != (room_b, y_b, base.s, base.q):
            raise AssertionError(
                f'worlds diverged at step {t} before key_8 completed '
                f'W21=({room_a},{y_a},s={agent.s},q={agent.q}) '
                f'W2=({room_b},{y_b},s={base.s},q={base.q})')
        prefix.append(act)
        prefix.append(obs_token(room_a, y_a))
    if completed is None:
        raise AssertionError(f'rand6 never completed key_8 in {steps} steps')
    prefix_tokens = np.asarray(prefix, dtype=np.int64)
    if prefix_tokens.shape[0] != completed * 2:
        raise AssertionError('prefix token length')
    life21 = rollout(world_seed, 'rand6', steps, 512)
    life2 = w2.rollout(world_seed, 'rand6', steps, 512)
    if not np.array_equal(life21['actions'], actions):
        raise AssertionError('rollout rand6 actions differ from the check stream')
    if not np.array_equal(life21['tokens'][:completed * 2], prefix_tokens):
        raise AssertionError('World21 rollout prefix differs from the stepped prefix')
    if not np.array_equal(life2['tokens'][:completed * 2], prefix_tokens):
        raise AssertionError('World rollout prefix differs from the stepped prefix')
    digest = token_sha256(prefix_tokens)
    if token_sha256(life2['tokens'][:completed * 2]) != digest:
        raise AssertionError('prefix sha256 mismatch')
    print(
        f'equivalence world_seed={world_seed} first_key8_step={completed} '
        f'prefix_sha256={digest}',
        flush=True)
    return {
        'world_seed': int(world_seed),
        'first_key8_step': int(completed),
        'prefix_sha256': digest,
        'prefix_steps': int(completed),
    }


def exit_check(world_seed=0):
    """Play keys 0..7, then key_8. The last key returns to room 0."""
    world = World21(world_seed)
    rooms_after = []
    for k in range(DEPTH):
        for j in range(KEY_LEN):
            world.step(int(world.keys[k, j]))
        rooms_after.append(int(world.room))
        if world.room != k + 1 or world.s != 0:
            raise AssertionError(f'after key_{k} room={world.room} s={world.s}')
    if world.room != DEPTH or world.q != 0 or world.s8 != 0:
        raise AssertionError(
            f'entry to room 8 left room={world.room} q={world.q} s8={world.s8}')
    for j in range(KEY_LEN):
        world.step(int(world.key_8[j]))
    rooms_after.append(int(world.room))
    expected = list(range(1, DEPTH + 1)) + [0]
    if rooms_after != expected:
        raise AssertionError(f'rooms after keys {rooms_after} != {expected}')
    if (world.room, world.s, world.q, world.s8) != (0, 0, 0, 0):
        raise AssertionError(
            f'key_8 left room={world.room} s={world.s} q={world.q} s8={world.s8}')
    print(
        'exit rooms_after_keys=' + ' '.join(str(r) for r in rooms_after)
        + f' s={world.s} q={world.q} s8={world.s8}',
        flush=True)
    return {
        'rooms_after_keys': rooms_after,
        's': int(world.s),
        'q': int(world.q),
        's8': int(world.s8),
        'key_8': [int(v) for v in world.key_8],
    }


def oracle_deep_check(world_seed=0, n_blocks=4, block_size=512):
    """Four blocks. Room 8 gets the even blocks and key_8 never completes."""
    steps = n_blocks * block_size
    life = rollout(world_seed, 'oracle_deep', steps, block_size)
    rows = []
    for b in range(n_blocks):
        target = oracle_deep_target(b)
        start = b * block_size
        end = start + block_size
        counts = np.bincount(life['rooms'][start:end], minlength=N_ROOMS)
        nav = 1 + KEY_LEN * target
        arrived = start + nav - 1
        if int(life['rooms'][arrived]) != target:
            raise AssertionError(
                f'block {b} arrived in {int(life["rooms"][arrived])}, target {target}')
        if np.any(life['rooms'][arrived:end] != target):
            raise AssertionError(f'oracle_deep left room {target} in block {b}')
        print(
            f'oracle_deep block={b} target={target} time='
            + ' '.join(str(int(v)) for v in counts),
            flush=True)
        rows.append({
            'block': b,
            'target': int(target),
            'time_in_room': [int(v) for v in counts],
        })
    if rows[0]['target'] != DEPTH or rows[2]['target'] != DEPTH:
        raise AssertionError('room 8 did not get blocks 0 and 2')
    if life['full_cycles'] != 0:
        raise AssertionError(f"key_8 completed {life['full_cycles']} times")
    if np.any(life['actions'] == TV):
        raise AssertionError('oracle_deep used the TV')
    print(
        'oracle_deep total_time=' + ' '.join(str(v) for v in life['time_in_room'])
        + f" full_cycles={life['full_cycles']}",
        flush=True)
    return {
        'blocks': rows,
        'time_in_room': life['time_in_room'],
        'full_cycles': life['full_cycles'],
    }


def smoke_checks(world_seed=0):
    """Spec checks 1–3. Raises on failure. No learner and no kernel."""
    equivalence = equivalence_check(world_seed)
    exit_row = exit_check(world_seed)
    oracle_row = oracle_deep_check(world_seed)
    diff = room8_probe_diff(world_seed)
    if diff['prefix_differ'] != 0:
        raise AssertionError(
            f"room 8 probe prefix differs in {diff['prefix_differ']} actions")
    print(
        f"room8_probe_diff world_seed={world_seed} n_differ={diff['n_differ']} "
        f"scored_differ={diff['scored_differ']} prefix_differ={diff['prefix_differ']} "
        f"n_actions={diff['n_actions']}",
        flush=True)
    print('SMOKE_WORLD_OK', flush=True)
    return {
        'equivalence': equivalence,
        'exit': exit_row,
        'oracle_deep': oracle_row,
        'room8_probe_diff': diff,
    }


if __name__ == '__main__':
    smoke_checks()
