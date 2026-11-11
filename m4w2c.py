"""M4-W2c: a curious controller with memory on the locked-rooms world.

Imports the world, the learner, and M4-W2b's visit-to-visit reward. The memory
arms use state (room, p1, p2): the room and the two previous actions, with 8
meaning no action. A step that changes the room sets both to 8. rand6 and
lp_room are M4-W2b's rand6 and prog_lp. Numbers only. No verdict. Does not commit.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import m4w2
import m4w2b
import w2_headroom as head
import w2_world as world

ARMS = ('rand6', 'lp_mem', 'count_mem', 'lp_room')
MEM_ARMS = ('lp_mem', 'count_mem')
Q_ARMS = ('lp_mem', 'count_mem', 'lp_room')
SEEDS = (15, 16, 17, 18, 19, 20)
KERNELS = (
    ('ezzzzzx/m4w2c-a', (15, 16)),
    ('ezzzzzx/m4w2c-b', (17, 18)),
    ('ezzzzzx/m4w2c-c', (19, 20)),
)
SMOKE_STEPS = 640
SMOKE_BLOCK = 32
SMOKE_SEED = 0
CAPABILITY_STEPS = 20_000
NONE = 8
N_MEM = 9
N_STATES = 9 * 9 * 9
OUT_DIR = Path('results/m4w2c')
GAP_NAMES = (
    'rand6_minus_lp_mem',
    'count_mem_minus_lp_mem',
    'lp_room_minus_lp_mem',
)
PARENT_ARM = {'rand6': 'rand6', 'lp_room': 'prog_lp'}


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def _fmt(value, digits=6):
    if value is None:
        return ''
    return f'{value:.{digits}f}'


def _require_constants():
    m4w2b._require_parent_constants()
    if m4w2b.K_LP != 16 or NONE != 8 or N_MEM != 9 or N_STATES != 729:
        raise AssertionError('memory layout')
    if world.N_ROOMS * N_MEM * N_MEM != N_STATES:
        raise AssertionError('state count')


def state_index(room, p1, p2):
    room = int(room)
    p1 = int(p1)
    p2 = int(p2)
    if not (0 <= room < world.N_ROOMS and 0 <= p1 < N_MEM and 0 <= p2 < N_MEM):
        raise AssertionError(f'state {(room, p1, p2)}')
    return (room * N_MEM + p1) * N_MEM + p2


def decode_state(index):
    index = int(index)
    if not (0 <= index < N_STATES):
        raise AssertionError(f'state index {index}')
    p2 = index % N_MEM
    index //= N_MEM
    p1 = index % N_MEM
    room = index // N_MEM
    return int(room), int(p1), int(p2)


def advance_memory(room, p1, p2, action, room_after):
    """Next (room, p1, p2). A room change drops both previous actions."""
    if int(room_after) != int(room):
        return int(room_after), NONE, NONE
    return int(room_after), int(action), int(p1)


def _check_state_index():
    seen = set()
    for room in range(world.N_ROOMS):
        for p1 in range(N_MEM):
            for p2 in range(N_MEM):
                index = state_index(room, p1, p2)
                if index in seen or decode_state(index) != (room, p1, p2):
                    raise AssertionError('state index is not one-to-one')
                seen.add(index)
    if seen != set(range(N_STATES)):
        raise AssertionError('state index does not cover 0..728')


def _eps_greedy_state(q_table, s_idx, rng):
    """ε over actions 0..5. Otherwise argmax over all 8, ties uniform."""
    if float(rng.random()) < m4w2.EPS:
        return int(rng.integers(0, world.N_MANIP)), 'eps'
    row = q_table[int(s_idx)]
    best = row.max()
    cands = np.flatnonzero(row == best)
    if int(cands.size) == 1:
        return int(cands[0]), 'greedy'
    return int(rng.choice(cands)), 'greedy'


def _choose_memory(q_table, s_idx, rng):
    return _eps_greedy_state(q_table, s_idx, rng)


def _rooms_by_quarter(world_seed, actions):
    steps = int(len(actions))
    if steps % 4 != 0:
        raise AssertionError('T is not divisible by 4')
    env = world.World(int(world_seed))
    counts = np.zeros((4, world.N_ROOMS), dtype=np.int64)
    total = np.zeros(world.N_ROOMS, dtype=np.int64)
    for t, action in enumerate(actions):
        room_after, _y = env.step(int(action))
        counts[t * 4 // steps, int(room_after)] += 1
        total[int(room_after)] += 1
    return counts, total


def _quarter_lists(counts):
    return [[int(v) for v in row] for row in counts]


def _run_parent(world_seed, arm, steps, block_size, device, log_every, keep_trace):
    """rand6 and lp_room are M4-W2b lives, relabelled."""
    parent = PARENT_ARM[arm]
    row = m4w2b.run_closed(
        world_seed, parent, steps, block_size, device,
        log_every=log_every, keep_trace=True)
    actions = row.pop('actions')
    counts, total = _rooms_by_quarter(world_seed, actions)
    if [int(v) for v in total] != [int(v) for v in row['time_in_room']]:
        raise AssertionError(f'{arm} replay time-in-room drifted')
    row['arm'] = arm
    row['time_in_room_by_quarter'] = _quarter_lists(counts)
    row['n_states'] = int(world.N_ROOMS) if arm == 'lp_room' else None
    row['state_kind'] = 'room' if arm == 'lp_room' else 'none'
    if keep_trace:
        row['actions'] = [int(v) for v in actions]
    return row


def _run_memory(world_seed, arm, steps, block_size, device, log_every=200,
                keep_trace=False):
    """One closed-loop life whose Q state is (room, p1, p2)."""
    _require_constants()
    if arm not in MEM_ARMS:
        raise ValueError(arm)
    if steps % 4 != 0:
        raise AssertionError(f'T={steps} is not divisible by 4')
    t0 = time.monotonic()
    env = world.World(world_seed)
    episodes = world.build_probe_episodes(world_seed)
    torch.manual_seed(head.SEED)
    model = head.TokenGRU().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != head.expected_params():
        raise AssertionError(f'parameter count {n_params} != {head.expected_params()}')
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=head.LR)
    replay_rng = np.random.default_rng(head.REPLAY_SEED + head.SEED)
    ctrl = np.random.default_rng(m4w2.CTRL_SEED + world_seed)
    q_table = np.zeros((N_STATES, world.N_ACTIONS), dtype=np.float64)
    visits = np.zeros((N_STATES, world.N_ACTIONS), dtype=np.int64)
    reward_sum = np.zeros((N_STATES, world.N_ACTIONS), dtype=np.float64)
    reward_n = np.zeros((N_STATES, world.N_ACTIONS), dtype=np.int64)
    histories = None
    if arm == 'lp_mem':
        histories = [[] for _ in range(world.N_ROOMS * world.N_ACTIONS)]
    tokens = np.empty(steps * 2, dtype=np.int64)
    actions = np.empty(steps, dtype=np.int64)
    mem_room = int(env.room)
    mem_p1 = NONE
    mem_p2 = NONE
    state = None
    total_bits = 0.0
    n_obs = 0
    quarter_bits = np.zeros(4, dtype=np.float64)
    quarter_count = np.zeros(4, dtype=np.int64)
    nominal = (steps // 4, steps // 2, steps)
    probes = []
    next_probe = 0
    replay_steps = 0
    chunk_count = 0
    room_counts = np.zeros(world.N_ROOMS, dtype=np.int64)
    room_quarter = np.zeros((4, world.N_ROOMS), dtype=np.int64)
    deepest = -1
    first_step = -1
    last_quarter = 3 * steps // 4
    pos = 0
    n_tokens = steps * 2
    while pos < n_tokens:
        end = min(pos + head.CHUNK, n_tokens)
        n_local = (end - pos) // 2
        step0 = pos // 2
        rooms_before = np.empty(n_local, dtype=np.int64)
        s_before = np.empty(n_local, dtype=np.int64)
        s_after = np.empty(n_local, dtype=np.int64)
        for i in range(n_local):
            t = step0 + i
            if int(env.room) != mem_room:
                raise AssertionError('memory room drifted from the world')
            rooms_before[i] = mem_room
            s_before[i] = state_index(mem_room, mem_p1, mem_p2)
            action, _branch = _choose_memory(q_table, int(s_before[i]), ctrl)
            room_after, y = env.step(action)
            actions[t] = action
            tokens[2 * t] = int(action)
            tokens[2 * t + 1] = world.obs_token(room_after, y)
            room_counts[int(room_after)] += 1
            room_quarter[t * 4 // steps, int(room_after)] += 1
            if int(room_after) > deepest:
                deepest = int(room_after)
                first_step = int(t)
            mem_room, mem_p1, mem_p2 = advance_memory(
                int(rooms_before[i]), mem_p1, mem_p2, action, room_after)
            s_after[i] = state_index(mem_room, mem_p1, mem_p2)
        prev, tgt = head.live_batch(tokens, pos, end)
        prev_t, tgt_t = head._as_batch(prev, tgt, device)
        losses, new_state = model.nll(prev_t, tgt_t, state)
        bits_before = head._obs_bits(losses)
        if bits_before.shape[0] != n_local:
            raise AssertionError('scored a token that is not an observation')
        carried = new_state.detach()
        step_ids = step0 + np.arange(n_local, dtype=np.int64)
        quarters = np.minimum(3, step_ids * 4 // steps)
        for q in range(4):
            mask = quarters == q
            if np.any(mask):
                quarter_bits[q] += float(bits_before[mask].sum())
                quarter_count[q] += int(mask.sum())
        total_bits += float(bits_before.sum())
        n_obs += int(n_local)
        state = carried
        head._step(opt, model, losses[1::2])
        del losses
        if end >= head.CHUNK:
            r_prev, r_tgt = head.sample_replay(tokens, end, replay_rng)
            r_prev_t, r_tgt_t = head._as_batch(r_prev, r_tgt, device)
            r_losses, _r_state = model.nll(r_prev_t, r_tgt_t, None)
            head._step(opt, model, r_losses[1::2])
            del r_losses
            replay_steps += 1
        for i in range(n_local):
            room_b = int(rooms_before[i])
            action = int(actions[step0 + i])
            s_b = int(s_before[i])
            s_n = int(s_after[i])
            if arm == 'lp_mem':
                hist = histories[room_b * world.N_ACTIONS + action]
                n_visit = len(hist) + 1
                reward = m4w2b.prog_lp_reward(hist, n_visit)
                hist.append(float(bits_before[i]))
            else:
                reward = m4w2._reward('count_q', 0.0, 0.0, s_b, action, visits)
            if not math.isfinite(reward):
                raise FloatingPointError(f'{arm} reward at step {step0 + i}')
            m4w2._q_update(q_table, s_b, action, reward, s_n)
            if step0 + i >= last_quarter:
                reward_sum[s_b, action] += reward
                reward_n[s_b, action] += 1
        del bits_before
        chunk_count += 1
        pos = end
        steps_done = pos // 2
        while next_probe < len(nominal) and steps_done >= nominal[next_probe]:
            row = head.probe_model(model, episodes, device)
            row['nominal_step'] = int(nominal[next_probe])
            row['at_step'] = int(steps_done)
            row['PROBE_cap'] = m4w2.probe_cap(row['probe_k'])
            probes.append(row)
            print(
                f'M4W2C seed={world_seed} arm={arm} probe nominal={row["nominal_step"]} '
                f'at_step={row["at_step"]} PROBE={row["PROBE"]:.4f} '
                f'cap={row["PROBE_cap"]:.4f} tv={row["tv_probe"]:.4f} '
                f'wall_s={time.monotonic() - t0:.1f}',
                flush=True)
            next_probe += 1
        if chunk_count % log_every == 0 or pos == n_tokens:
            print(
                f'M4W2C seed={world_seed} arm={arm} steps={steps_done}/{steps} '
                f'bpo={total_bits / max(n_obs, 1):.4f} replay={replay_steps} '
                f'wall_s={time.monotonic() - t0:.1f}',
                flush=True)
    if n_obs != steps or next_probe != len(nominal):
        raise AssertionError(f'scored {n_obs} observations, probes {next_probe}')
    if abs(float(quarter_bits.sum()) - total_bits) > 1e-4:
        raise AssertionError('quarter bits do not sum to the life')
    if int(quarter_count.sum()) != steps:
        raise AssertionError('quarter counts do not sum to T')
    if int(room_counts.sum()) != steps or int(room_quarter.sum()) != steps:
        raise AssertionError('time-in-room does not sum to T')
    if [int(v) for v in room_quarter.sum(axis=0)] != [int(v) for v in room_counts]:
        raise AssertionError('quarter rooms do not sum to the life')
    if not np.isfinite(q_table).all():
        raise FloatingPointError(f'{arm} Q is not finite')
    sha = world.token_sha256(tokens)
    bounds = head._quarter_bounds(steps)
    quarters = []
    width = steps // 4
    resets_per_1000 = []
    for i, (start, stop) in enumerate(bounds):
        count = int(quarter_count[i])
        bits_q = float(quarter_bits[i])
        sl = actions[start:stop]
        resets_per_1000.append(float(np.sum(sl == world.RESET) * 1000.0 / width))
        quarters.append({
            'index': i,
            'start_step': int(start),
            'end_step': int(stop),
            'observations': count,
            'bits': bits_q,
            'bits_per_observation': bits_q / count,
        })
    tv_by_quarter, reset_by_quarter = m4w2b._quarter_shares(actions, steps)
    for i in range(4):
        if abs(reset_by_quarter[i] * 1000.0 - resets_per_1000[i]) > 1e-6:
            raise AssertionError('reset share does not match resets per 1,000')
        if int(room_quarter[i].sum()) != width:
            raise AssertionError('quarter time-in-room width')
    counts = np.bincount(actions, minlength=world.N_ACTIONS)
    if int(counts.sum()) != steps:
        raise AssertionError('action counts do not sum to T')
    share = counts.astype(np.float64) / steps
    if abs(float(share.sum()) - 1.0) > 1e-9 or np.any(share < 0):
        raise AssertionError('action share is not a distribution')
    if abs(float(np.mean(tv_by_quarter)) - float(share[world.TV])) > 1e-12:
        raise AssertionError('quarter TV shares')
    if abs(float(np.mean(reset_by_quarter)) - float(share[world.RESET])) > 1e-12:
        raise AssertionError('quarter reset shares')
    action_share = {
        'manipulation': [float(share[i]) for i in range(world.N_MANIP)],
        'tv': float(share[world.TV]),
        'reset': float(share[world.RESET]),
    }
    finite = all(row['finite'] for row in probes) and math.isfinite(total_bits)
    mean_reward = []
    for s_idx in range(N_STATES):
        row = []
        for a in range(world.N_ACTIONS):
            n = int(reward_n[s_idx, a])
            row.append(None if n == 0 else float(reward_sum[s_idx, a] / n))
        mean_reward.append(row)
    wall = time.monotonic() - t0
    result = {
        'world_seed': int(world_seed),
        'arm': arm,
        'steps': int(steps),
        'block_size': int(block_size),
        'token_sha256': sha,
        'time_in_room': [int(v) for v in room_counts],
        'time_in_room_by_quarter': _quarter_lists(room_quarter),
        'deepest_room': int(deepest),
        'deepest_first_step': int(first_step),
        'bits': float(total_bits),
        'bits_per_observation': float(total_bits / steps),
        'quarters': quarters,
        'probes': probes,
        'PROBE_T': float(probes[-1]['PROBE']),
        'PROBE_cap_T': float(probes[-1]['PROBE_cap']),
        'tv_probe_T': float(probes[-1]['tv_probe']),
        'action_counts': [int(v) for v in counts],
        'action_share': action_share,
        'tv_share_by_quarter': tv_by_quarter,
        'reset_share_by_quarter': reset_by_quarter,
        'resets_per_1000': resets_per_1000,
        'Q': [[float(v) for v in row] for row in q_table],
        'mean_reward_last_quarter': mean_reward,
        'reward_count_last_quarter': [[int(v) for v in row] for row in reward_n],
        'controller': {
            'eps': m4w2.EPS,
            'alpha': m4w2.ALPHA,
            'gamma': m4w2.GAMMA,
            'seed': int(m4w2.CTRL_SEED + world_seed),
        },
        'wall_seconds': wall,
        'replay_steps': int(replay_steps),
        'chunk_count': int(chunk_count),
        'finite': bool(finite),
        'learner_seed': head.SEED,
        'replay_seed': head.REPLAY_SEED + head.SEED,
        'lr': head.LR,
        'clip': head.CLIP,
        'replay_batch': head.REPLAY_BATCH,
        'n_params': int(n_params),
        'freeze_q': False,
        'n_states': N_STATES,
        'state_kind': 'memory',
    }
    if keep_trace:
        result['actions'] = [int(v) for v in actions]
    final = probes[-1]
    print(
        f'M4W2C done seed={world_seed} arm={arm} bpo={result["bits_per_observation"]:.4f} '
        f'PROBE={final["PROBE"]:.4f} cap={final["PROBE_cap"]:.4f} '
        f'tv_share={action_share["tv"]:.4f} reset_share={action_share["reset"]:.4f} '
        f'deepest={deepest} room8={int(room_counts[8])} wall_s={wall:.1f}',
        flush=True)
    return result


def run_closed(world_seed, arm, steps, block_size, device, log_every=200,
               keep_trace=False):
    if arm in PARENT_ARM:
        return _run_parent(
            world_seed, arm, steps, block_size, device, log_every, keep_trace)
    if arm in MEM_ARMS:
        return _run_memory(
            world_seed, arm, steps, block_size, device, log_every, keep_trace)
    raise ValueError(arm)


def _state_check(world_seed=SMOKE_SEED):
    """Scripted key, one manipulation, TV, reset. Returns the state list."""
    _check_state_index()
    env = world.World(world_seed)
    k0 = int(env.keys[0, 0])
    k1 = int(env.keys[0, 1])
    k2 = int(env.keys[0, 2])
    manip = 0
    actions = [k0, k1, k2, manip, world.TV, world.RESET]
    expected = [
        (0, NONE, NONE),
        (0, k0, NONE),
        (0, k1, k0),
        (1, NONE, NONE),
        (1, manip, NONE),
        (1, world.TV, manip),
        (0, NONE, NONE),
    ]
    room, p1, p2 = 0, NONE, NONE
    if int(env.room) != 0:
        raise AssertionError('world did not start in room 0')
    states = [(room, p1, p2)]
    for action in actions:
        room_before = int(env.room)
        if room_before != room:
            raise AssertionError('script room drifted')
        room_after, _y = env.step(action)
        room, p1, p2 = advance_memory(room_before, p1, p2, action, room_after)
        if room != int(env.room):
            raise AssertionError('memory room does not match the world')
        states.append((room, p1, p2))
    text = ' -> '.join(f'({r},{a},{b})' for r, a, b in states)
    print(
        f'SMOKE state key0={k0} key1={k1} key2={k2} manip={manip} tv={world.TV} '
        f'reset={world.RESET}',
        flush=True)
    print(f'SMOKE state {text}', flush=True)
    match = states == expected
    if not match:
        exp = ' -> '.join(f'({r},{a},{b})' for r, a, b in expected)
        print(f'SMOKE state expected {exp}', flush=True)
        raise AssertionError('state check failed')
    return {
        'world_seed': int(world_seed),
        'key_0': [k0, k1, k2],
        'manipulation': manip,
        'actions': [int(v) for v in actions],
        'states': [[int(v) for v in row] for row in states],
        'expected': [[int(v) for v in row] for row in expected],
        'match': True,
    }


def capability_check(world_seed=SMOKE_SEED, steps=CAPABILITY_STEPS):
    """No learner. +1 the first time a step enters a room, else 0.

    Room 0 is already entered at the start. Q updates on every step.
    """
    _require_constants()
    env = world.World(world_seed)
    rng = np.random.default_rng(m4w2.CTRL_SEED + world_seed)
    q_table = np.zeros((N_STATES, world.N_ACTIONS), dtype=np.float64)
    seen = np.zeros(world.N_ROOMS, dtype=bool)
    seen[int(env.room)] = True
    room, p1, p2 = int(env.room), NONE, NONE
    deepest = -1
    first_step = -1
    first_entry = [None] * world.N_ROOMS
    for t in range(int(steps)):
        if int(env.room) != room:
            raise AssertionError('capability room drifted')
        s_idx = state_index(room, p1, p2)
        action, _branch = _choose_memory(q_table, s_idx, rng)
        room_after, _y = env.step(action)
        reward = 0.0
        if not bool(seen[int(room_after)]):
            reward = 1.0
            seen[int(room_after)] = True
            first_entry[int(room_after)] = int(t)
        room, p1, p2 = advance_memory(room, p1, p2, action, room_after)
        m4w2._q_update(q_table, s_idx, action, reward, state_index(room, p1, p2))
        if int(room_after) > deepest:
            deepest = int(room_after)
            first_step = int(t)
    if not np.isfinite(q_table).all():
        raise FloatingPointError('capability Q is not finite')
    print(
        f'SMOKE capability seed={world_seed} steps={steps} '
        f'deepest={deepest} first_step={first_step}',
        flush=True)
    return {
        'world_seed': int(world_seed),
        'steps': int(steps),
        'learner': False,
        'q_update': 'every_step',
        'reward': 'first_entry',
        'initial_room_already_entered': True,
        'deepest_room': int(deepest),
        'deepest_first_step': int(first_step),
        'first_entry_step': first_entry,
        'Q_finite': True,
    }


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _compare_parent(arm, closed, ref):
    diff = abs(float(closed['bits']) - float(ref['bits']))
    sha_match = closed['token_sha256'] == ref['token_sha256']
    probe_diff = []
    for c_row, r_row in zip(closed['probes'], ref['probes']):
        probe_diff.append(_max_abs(
            c_row['probe_k'] + [c_row['tv_probe']],
            r_row['probe_k'] + [r_row['tv_probe']]))
    probe_max = float(max(probe_diff))
    q_max = None
    if arm == 'lp_room':
        q_max = _max_abs(closed['Q'], ref['Q'])
    row = {
        'arm': arm,
        'parent_arm': PARENT_ARM[arm],
        'token_sha256': closed['token_sha256'],
        'ref_token_sha256': ref['token_sha256'],
        'sha_match': bool(sha_match),
        'bits': float(closed['bits']),
        'ref_bits': float(ref['bits']),
        'abs_diff': float(diff),
        'probe_max_abs': probe_max,
        'Q_max_abs': q_max,
    }
    print(
        f'SMOKE equivalence arm={arm} parent={PARENT_ARM[arm]} sha_match={sha_match} '
        f'bits={row["bits"]:.6f} ref_bits={row["ref_bits"]:.6f} '
        f'abs_diff={diff:.3e} probe_max_abs={probe_max:.3e}',
        flush=True)
    print(f'SMOKE sha {arm} {closed["token_sha256"]}', flush=True)
    print(f'SMOKE sha_ref {arm} {ref["token_sha256"]}', flush=True)
    if not sha_match or diff > 1e-6 or probe_max > 1e-6:
        raise AssertionError(f'{arm} does not match M4-W2b {PARENT_ARM[arm]}')
    if q_max is not None and q_max > 1e-9:
        raise AssertionError(f'{arm} Q does not match M4-W2b')
    return row


def smoke():
    _require_constants()
    head.configure()
    t0 = time.monotonic()
    device = torch.device('cpu')
    state_row = _state_check(SMOKE_SEED)
    capability = capability_check(SMOKE_SEED, CAPABILITY_STEPS)
    refs = {}
    for arm, parent in PARENT_ARM.items():
        refs[arm] = m4w2b.run_closed(
            SMOKE_SEED, parent, SMOKE_STEPS, SMOKE_BLOCK, device, log_every=50)
    lives = {}
    equivalence = []
    arm_rows = []
    for arm in ARMS:
        row_life = run_closed(
            SMOKE_SEED, arm, SMOKE_STEPS, SMOKE_BLOCK, device,
            log_every=50, keep_trace=True)
        if arm in PARENT_ARM:
            equivalence.append(_compare_parent(arm, row_life, refs[arm]))
        q_finite = None
        if row_life['Q'] is not None:
            q_table = np.asarray(row_life['Q'], dtype=np.float64)
            expect = (N_STATES, world.N_ACTIONS) if arm in MEM_ARMS else (world.N_ROOMS, world.N_ACTIONS)
            q_finite = bool(np.isfinite(q_table).all() and q_table.shape == expect)
        bits_ok = bool(row_life['finite']) and math.isfinite(float(row_life['bits']))
        share = row_life['action_share']
        manip = share['manipulation']
        total = float(sum(manip) + share['tv'] + share['reset'])
        ok = bits_ok and abs(total - 1.0) <= 1e-9 and min(manip) >= 0
        if arm in Q_ARMS:
            ok = ok and q_finite
        print(
            f'SMOKE arm={arm} bits={row_life["bits"]:.6f} finite={row_life["finite"]} '
            f'Q_finite={q_finite} tv={share["tv"]:.6f} reset={share["reset"]:.6f}',
            flush=True)
        if not ok:
            raise AssertionError(f'{arm} smoke is not finite')
        lives[arm] = row_life
        arm_rows.append({
            'arm': arm,
            'bits': float(row_life['bits']),
            'bits_per_observation': float(row_life['bits_per_observation']),
            'finite': bool(row_life['finite']),
            'Q_finite': q_finite,
            'Q_shape': None if row_life['Q'] is None else [len(row_life['Q']), len(row_life['Q'][0])],
            'action_share': share,
            'PROBE_cap_T': row_life['PROBE_cap_T'],
            'PROBE_T': row_life['PROBE_T'],
            'wall_seconds': row_life['wall_seconds'],
        })
    first = {arm: lives[arm]['actions'][:head.CHUNK // 2] for arm in Q_ARMS}
    same_first = first['lp_mem'] == first['count_mem'] == first['lp_room']
    if not same_first:
        raise AssertionError('Q arms diverged inside the first frozen chunk')
    payload = {
        'world_seed': SMOKE_SEED,
        'steps': SMOKE_STEPS,
        'block_size': SMOKE_BLOCK,
        'equivalence': equivalence,
        'state_check': state_row,
        'capability': capability,
        'arms': arm_rows,
        'first_chunk_actions_match': True,
        'device': 'cpu',
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'wall_seconds': time.monotonic() - t0,
    }
    dump(OUT_DIR / 'smoke.json', payload)
    print(f'SMOKE_OK wall_s={payload["wall_seconds"]:.1f}', flush=True)
    return payload


def kernel_main(out, world_seeds, kernel_id, steps=head.T_FULL, block_size=head.BLOCK_FULL):
    _require_constants()
    head.configure()
    t0 = time.monotonic()
    out = Path(out)
    device = torch.device('cpu')
    seeds = tuple(int(s) for s in world_seeds)
    print(
        f'M4W2C kernel={kernel_id} seeds={list(seeds)} device={device} '
        f'cuda={torch.cuda.is_available()} torch={torch.__version__} '
        f'numpy={np.__version__} threads={torch.get_num_threads()}',
        flush=True)
    if device.type != 'cpu':
        raise SystemExit('M4-W2c is a CPU run')
    lives = []
    expect_chunks = math.ceil((steps * 2) / head.CHUNK)
    for world_seed in seeds:
        for arm in ARMS:
            print(f'M4W2C start seed={world_seed} arm={arm}', flush=True)
            row = run_closed(world_seed, arm, int(steps), int(block_size), device)
            if not row['finite']:
                raise FloatingPointError(f'{arm} seed {world_seed} is not finite')
            if row['chunk_count'] != expect_chunks or row['replay_steps'] != expect_chunks:
                raise AssertionError(
                    f'{arm} chunks {row["chunk_count"]} replay {row["replay_steps"]} '
                    f'!= {expect_chunks}')
            if row['probes'][-1]['at_step'] != int(steps):
                raise AssertionError('final probe is not at T')
            if row['freeze_q']:
                raise AssertionError('scored life froze Q')
            if arm in Q_ARMS:
                if row['controller']['seed'] != m4w2.CTRL_SEED + int(world_seed):
                    raise AssertionError('controller seed')
                if abs(row['controller']['eps'] - m4w2.EPS) > 0:
                    raise AssertionError('controller epsilon')
            if 'actions' in row:
                raise AssertionError('scored life kept the action trace')
            lives.append(row)
            dump(out / 'result.json', {
                'kernel_id': kernel_id,
                'world_seeds': list(seeds),
                'lives': lives,
            })
    result = {
        'kernel_id': kernel_id,
        'world_seeds': list(seeds),
        'steps': int(steps),
        'block_size': int(block_size),
        'device': 'cpu',
        'cuda_available': bool(torch.cuda.is_available()),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'n_params': head.expected_params(),
        'learner_seed': head.SEED,
        'replay_seed': head.REPLAY_SEED + head.SEED,
        'lr': head.LR,
        'clip': head.CLIP,
        'replay_batch': head.REPLAY_BATCH,
        'eps': m4w2.EPS,
        'alpha': m4w2.ALPHA,
        'gamma': m4w2.GAMMA,
        'controller_seed_base': m4w2.CTRL_SEED,
        'k_lp': m4w2b.K_LP,
        'explore': 'manipulations',
        'probe_cap': m4w2.CAP,
        'n_states': N_STATES,
        'memory_none': NONE,
        'arms': list(ARMS),
        'lives': lives,
        'session_wall_seconds': time.monotonic() - t0,
    }
    dump(out / 'result.json', result)
    summary = {
        'kernel_id': kernel_id,
        'world_seeds': list(seeds),
        'session_wall_seconds': result['session_wall_seconds'],
        'PROBE_cap_T': {f'{row["arm"]}_s{row["world_seed"]}': row['PROBE_cap_T'] for row in lives},
        'finite': all(row['finite'] for row in lives),
    }
    dump(out / 'kernel.json', summary)
    print('KERNEL_DONE ' + json.dumps(summary), flush=True)
    return result


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _log_elapsed(log_path):
    return m4w2b._log_elapsed(log_path)


def _status_name(status):
    return m4w2b._status_name(status)


def _require_life(row, world_seed):
    if int(row['world_seed']) != int(world_seed):
        raise SystemExit(f'life seed {row["world_seed"]} != {world_seed}')
    if row['arm'] not in ARMS:
        raise SystemExit(f'unknown arm {row["arm"]}')
    if int(row['steps']) != head.T_FULL or int(row['block_size']) != head.BLOCK_FULL:
        raise SystemExit(f'life length {row["steps"]} block {row["block_size"]}')
    if abs(float(row['lr']) - head.LR) > 1e-15 or int(row['replay_batch']) != head.REPLAY_BATCH:
        raise SystemExit('life was not the registered learner')
    if int(row['learner_seed']) != head.SEED or not row['finite']:
        raise SystemExit('life is not a finite seed-0 learner run')
    if row['probes'][-1]['at_step'] != head.T_FULL or row['probes'][-1]['nominal_step'] != head.T_FULL:
        raise SystemExit('final probe is not at T')
    if len(row['probes']) != 3 or len(row['quarters']) != 4:
        raise SystemExit('life is missing quarters or probes')
    if sum(row['time_in_room']) != head.T_FULL:
        raise SystemExit('time-in-room does not sum to T')
    quarters = row['time_in_room_by_quarter']
    if len(quarters) != 4:
        raise SystemExit('time-in-room quarters')
    flat = np.asarray(quarters, dtype=np.int64)
    if flat.shape != (4, world.N_ROOMS):
        raise SystemExit('time-in-room quarter shape')
    if [int(v) for v in flat.sum(axis=0)] != [int(v) for v in row['time_in_room']]:
        raise SystemExit('time-in-room quarters do not sum to the life')
    if [int(v) for v in flat.sum(axis=1)] != [head.T_FULL // 4] * 4:
        raise SystemExit('time-in-room quarter width')
    if len(row['tv_share_by_quarter']) != 4 or len(row['reset_share_by_quarter']) != 4:
        raise SystemExit('quarter shares')
    if len(row['resets_per_1000']) != 4:
        raise SystemExit('resets per 1,000')
    share = row['action_share']
    total = float(sum(share['manipulation']) + share['tv'] + share['reset'])
    if abs(total - 1.0) > 1e-8:
        raise SystemExit(f'{row["arm"]} action share sums to {total}')
    if len(share['manipulation']) != world.N_MANIP:
        raise SystemExit('manipulation share length')
    if abs(float(np.mean(row['tv_share_by_quarter'])) - float(share['tv'])) > 1e-8:
        raise SystemExit('TV quarter shares do not match the life')
    if abs(float(np.mean(row['reset_share_by_quarter'])) - float(share['reset'])) > 1e-8:
        raise SystemExit('reset quarter shares do not match the life')
    for i in range(4):
        if abs(float(row['reset_share_by_quarter'][i]) * 1000.0 - float(row['resets_per_1000'][i])) > 1e-6:
            raise SystemExit('resets per 1,000 do not match the reset share')
    cap = m4w2.probe_cap(row['probes'][-1]['probe_k'])
    if abs(cap - float(row['PROBE_cap_T'])) > 1e-9:
        raise SystemExit('PROBE_cap does not match probe_k')
    if row['arm'] in Q_ARMS:
        q_table = np.asarray(row['Q'], dtype=np.float64)
        expect = (N_STATES, world.N_ACTIONS) if row['arm'] in MEM_ARMS else (world.N_ROOMS, world.N_ACTIONS)
        if q_table.shape != expect or not np.isfinite(q_table).all():
            raise SystemExit(f'{row["arm"]} Q is missing or not finite')
        if row['arm'] in MEM_ARMS and row.get('state_kind') != 'memory':
            raise SystemExit(f'{row["arm"]} state kind')
        if row['arm'] == 'lp_room' and row.get('state_kind') != 'room':
            raise SystemExit('lp_room state kind')
        ctrl = row['controller']
        if int(ctrl['seed']) != m4w2.CTRL_SEED + int(world_seed):
            raise SystemExit('controller seed')
        if abs(float(ctrl['eps']) - m4w2.EPS) > 0 or abs(float(ctrl['alpha']) - m4w2.ALPHA) > 0:
            raise SystemExit('controller hyperparameters')
        if abs(float(ctrl['gamma']) - m4w2.GAMMA) > 0:
            raise SystemExit('controller gamma')
    elif row['Q'] is not None:
        raise SystemExit(f'{row["arm"]} stored a Q table')
    if row.get('freeze_q'):
        raise SystemExit('scored life froze Q')


def _arm_table(loaded, key):
    per_arm = {}
    for arm in ARMS:
        values = [float(loaded[seed][arm][key]) for seed in SEEDS]
        mean, std = _mean_std(values)
        per_arm[arm] = {
            'per_seed': [{'world_seed': int(seed), 'value': float(values[i])} for i, seed in enumerate(SEEDS)],
            'mean': mean,
            'std_ddof0': std,
        }
    return per_arm


def _gap_table(loaded, key):
    rows = []
    for seed in SEEDS:
        base = float(loaded[seed]['lp_mem'][key])
        rows.append({
            'world_seed': int(seed),
            'rand6_minus_lp_mem': float(loaded[seed]['rand6'][key]) - base,
            'count_mem_minus_lp_mem': float(loaded[seed]['count_mem'][key]) - base,
            'lp_room_minus_lp_mem': float(loaded[seed]['lp_room'][key]) - base,
        })
    summary = {}
    for name in GAP_NAMES:
        mean, std = _mean_std([row[name] for row in rows])
        summary[name] = {'mean': mean, 'std_ddof0': std}
    return {'per_seed': rows, 'mean_std_ddof0': summary}


def _quarter_share_table(loaded):
    out = {}
    for arm in ARMS:
        per_seed = []
        tv_cols = [[] for _ in range(4)]
        reset_cols = [[] for _ in range(4)]
        reset_1000_cols = [[] for _ in range(4)]
        for seed in SEEDS:
            row = loaded[seed][arm]
            tv = [float(v) for v in row['tv_share_by_quarter']]
            reset = [float(v) for v in row['reset_share_by_quarter']]
            per_1000 = [float(v) for v in row['resets_per_1000']]
            per_seed.append({
                'world_seed': int(seed),
                'tv': tv,
                'reset': reset,
                'resets_per_1000': per_1000,
            })
            for i in range(4):
                tv_cols[i].append(tv[i])
                reset_cols[i].append(reset[i])
                reset_1000_cols[i].append(per_1000[i])
        tv_mean, tv_std, reset_mean, reset_std = [], [], [], []
        per_1000_mean, per_1000_std = [], []
        for i in range(4):
            mean, std = _mean_std(tv_cols[i])
            tv_mean.append(mean)
            tv_std.append(std)
            mean, std = _mean_std(reset_cols[i])
            reset_mean.append(mean)
            reset_std.append(std)
            mean, std = _mean_std(reset_1000_cols[i])
            per_1000_mean.append(mean)
            per_1000_std.append(std)
        out[arm] = {
            'per_seed': per_seed,
            'tv_mean': tv_mean,
            'tv_std_ddof0': tv_std,
            'reset_mean': reset_mean,
            'reset_std_ddof0': reset_std,
            'resets_per_1000_mean': per_1000_mean,
            'resets_per_1000_std_ddof0': per_1000_std,
        }
    return out


def _room_quarter_table(loaded):
    out = {}
    for arm in ARMS:
        per_seed = []
        sums = np.zeros((4, world.N_ROOMS), dtype=np.float64)
        for seed in SEEDS:
            quarters = loaded[seed][arm]['time_in_room_by_quarter']
            per_seed.append({'world_seed': int(seed), 'counts': quarters})
            sums += np.asarray(quarters, dtype=np.float64)
        out[arm] = {
            'per_seed': per_seed,
            'mean': [[float(v) for v in row] for row in (sums / len(SEEDS))],
        }
    return out


def assemble(out_dir=OUT_DIR, kernel_meta=None):
    out_dir = Path(out_dir)
    if kernel_meta is None:
        meta_path = out_dir / 'kernel_meta.json'
        kernel_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    smoke_payload = json.loads((out_dir / 'smoke.json').read_text())
    loaded = {seed: {} for seed in SEEDS}
    raw_kernels = []
    for kernel_id, seeds in KERNELS:
        slug = kernel_id.rsplit('-', 1)[-1]
        path = out_dir / 'raw' / slug / 'result.json'
        payload = json.loads(path.read_text())
        if payload.get('kernel_id') != kernel_id:
            raise SystemExit(f'{path} kernel id {payload.get("kernel_id")}')
        if tuple(payload['world_seeds']) != tuple(seeds):
            raise SystemExit(f'{path} seeds {payload["world_seeds"]}')
        if payload.get('device') != 'cpu' or payload.get('cuda_available') is True:
            raise SystemExit(f'{kernel_id} did not stay on CPU')
        if int(payload['steps']) != head.T_FULL or int(payload['block_size']) != head.BLOCK_FULL:
            raise SystemExit(f'{kernel_id} length')
        if int(payload.get('k_lp', -1)) != m4w2b.K_LP or payload.get('explore') != 'manipulations':
            raise SystemExit(f'{kernel_id} is not the M4-W2c controller')
        if int(payload.get('n_states', -1)) != N_STATES or int(payload.get('memory_none', -1)) != NONE:
            raise SystemExit(f'{kernel_id} memory layout')
        by_key = {}
        for row in payload['lives']:
            _require_life(row, row['world_seed'])
            if int(row['world_seed']) not in seeds:
                raise SystemExit(f'{kernel_id} unexpected seed {row["world_seed"]}')
            if row['arm'] == 'rand6':
                local = world.rollout(int(row['world_seed']), 'rand6', head.T_FULL, head.BLOCK_FULL)
                if local['token_sha256'] != row['token_sha256']:
                    raise SystemExit(
                        f'seed {row["world_seed"]} rand6 token sha local '
                        f'{local["token_sha256"]} != kernel {row["token_sha256"]}')
                if local['time_in_room'] != row['time_in_room']:
                    raise SystemExit(f'seed {row["world_seed"]} rand6 time-in-room mismatch')
                if local['deepest_first_step'] != row['deepest_first_step']:
                    raise SystemExit(f'seed {row["world_seed"]} rand6 deepest step mismatch')
            by_key[(int(row['world_seed']), row['arm'])] = row
        if len(by_key) != len(seeds) * len(ARMS):
            raise SystemExit(f'{kernel_id} lives {len(by_key)}')
        for seed in seeds:
            for arm in ARMS:
                loaded[seed][arm] = by_key[(seed, arm)]
        raw_kernels.append(payload)

    kernels = []
    for payload in raw_kernels:
        kernel_id = payload['kernel_id']
        info = kernel_meta.get(kernel_id, {})
        slug = kernel_id.rsplit('-', 1)[-1]
        logs = sorted((out_dir / 'raw' / slug).glob('*.log'))
        log_s = _log_elapsed(logs[0]) if logs else None
        script_s = float(payload['session_wall_seconds'])
        if log_s is None:
            elapsed = script_s
            source = 'script_wall'
        else:
            elapsed = float(log_s)
            source = 'log_clock'
        kernels.append({
            'kernel_id': kernel_id,
            'world_seeds': list(payload['world_seeds']),
            'version': info.get('version'),
            'status': _status_name(info.get('status', '')),
            'elapsed_seconds': elapsed,
            'elapsed_source': source,
            'session_wall_seconds': script_s,
            'log_elapsed_seconds': log_s,
            'torch_version': payload.get('torch_version'),
            'numpy_version': payload.get('numpy_version'),
            'cuda_available': payload.get('cuda_available'),
        })

    lives_out = []
    for seed in SEEDS:
        for arm in ARMS:
            lives_out.append(loaded[seed][arm])
    share_mean = {}
    depth = {}
    for arm in ARMS:
        tv = [float(loaded[seed][arm]['action_share']['tv']) for seed in SEEDS]
        reset = [float(loaded[seed][arm]['action_share']['reset']) for seed in SEEDS]
        manip = [
            [float(loaded[seed][arm]['action_share']['manipulation'][i]) for seed in SEEDS]
            for i in range(world.N_MANIP)
        ]
        tv_mean, tv_std = _mean_std(tv)
        reset_mean, reset_std = _mean_std(reset)
        share_mean[arm] = {
            'tv_mean': tv_mean,
            'tv_std_ddof0': tv_std,
            'reset_mean': reset_mean,
            'reset_std_ddof0': reset_std,
            'manipulation_mean': [_mean_std(col)[0] for col in manip],
            'tv_per_seed': [{'world_seed': int(seed), 'tv': float(tv[i])} for i, seed in enumerate(SEEDS)],
        }
        per_seed = []
        for seed in SEEDS:
            row = loaded[seed][arm]
            per_seed.append({
                'world_seed': int(seed),
                'deepest_room': int(row['deepest_room']),
                'deepest_first_step': int(row['deepest_first_step']),
                'time_in_room_8': int(row['time_in_room'][8]),
            })
        depth[arm] = {
            'per_seed': per_seed,
            'deepest_mean': float(np.mean([row['deepest_room'] for row in per_seed])),
            'time_in_room_8_mean': float(np.mean([row['time_in_room_8'] for row in per_seed])),
        }

    result = {
        'steps': head.T_FULL,
        'block_size': head.BLOCK_FULL,
        'vocab': head.VOCAB,
        'emb': head.EMB,
        'hidden': head.HIDDEN,
        'chunk_tokens': head.CHUNK,
        'lr': head.LR,
        'clip': head.CLIP,
        'replay_batch': head.REPLAY_BATCH,
        'learner_seed': head.SEED,
        'replay_seed': head.REPLAY_SEED + head.SEED,
        'n_params': head.expected_params(),
        'eps': m4w2.EPS,
        'alpha': m4w2.ALPHA,
        'gamma': m4w2.GAMMA,
        'controller_seed_base': m4w2.CTRL_SEED,
        'k_lp': m4w2b.K_LP,
        'explore': 'manipulations',
        'probe_cap': m4w2.CAP,
        'n_states': N_STATES,
        'memory_none': NONE,
        'arms': list(ARMS),
        'world_seeds': list(SEEDS),
        'lives': lives_out,
        'probe_cap_table': _arm_table(loaded, 'PROBE_cap_T'),
        'probe_raw_table': _arm_table(loaded, 'PROBE_T'),
        'gaps_probe_cap': _gap_table(loaded, 'PROBE_cap_T'),
        'gaps_probe_raw': _gap_table(loaded, 'PROBE_T'),
        'action_share_mean': share_mean,
        'quarter_shares': _quarter_share_table(loaded),
        'time_in_room_by_quarter': _room_quarter_table(loaded),
        'depth': depth,
        'capability': smoke_payload['capability'],
        'kernels': kernels,
        'kernel_elapsed_seconds_sum': float(sum(row['elapsed_seconds'] for row in kernels)),
        'smoke': smoke_payload,
    }
    dump(out_dir / 'results.json', result)
    lines = _markdown(result)
    (out_dir / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    tv = result['action_share_mean']['lp_mem']
    print(
        'M4W2C assemble '
        + ' '.join(
            f'{name}={result["gaps_probe_cap"]["mean_std_ddof0"][name]["mean"]:.4f}'
            for name in GAP_NAMES)
        + f' lp_mem_tv={tv["tv_mean"]:.4f}',
        flush=True)
    return result


def _matrix_lines(matrix, digits=4):
    lines = [
        '| room | a0 | a1 | a2 | a3 | a4 | a5 | tv | reset |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r, row in enumerate(matrix):
        cells = [_fmt(v, digits) if v is not None else '' for v in row]
        lines.append(f'| {r} | {" | ".join(cells)} |')
    return lines


def _memory_matrix_lines(matrix, digits=4, drop_zeros=False):
    lines = [
        '| room | p1 | p2 | a0 | a1 | a2 | a3 | a4 | a5 | tv | reset |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    omitted = 0
    for index, row in enumerate(matrix):
        values = [None if v is None else float(v) for v in row]
        if drop_zeros and all(v is None or v == 0.0 for v in values):
            omitted += 1
            continue
        room, p1, p2 = decode_state(index)
        cells = [_fmt(v, digits) if v is not None else '' for v in values]
        lines.append(f'| {room} | {p1} | {p2} | {" | ".join(cells)} |')
    lines.append(f'Rows omitted {omitted} of {len(matrix)}.')
    return lines


def _gap_lines(title, block):
    lines = [
        '',
        title,
        '',
        '| world_seed | rand6 − lp_mem | count_mem − lp_mem | lp_room − lp_mem |',
        '|---:|---:|---:|---:|',
    ]
    for row in block['per_seed']:
        lines.append(
            f'| {row["world_seed"]} | {_fmt(row["rand6_minus_lp_mem"])} | '
            f'{_fmt(row["count_mem_minus_lp_mem"])} | {_fmt(row["lp_room_minus_lp_mem"])} |')
    gaps = block['mean_std_ddof0']
    lines.append(
        f'| mean | {_fmt(gaps["rand6_minus_lp_mem"]["mean"])} | '
        f'{_fmt(gaps["count_mem_minus_lp_mem"]["mean"])} | '
        f'{_fmt(gaps["lp_room_minus_lp_mem"]["mean"])} |')
    lines.append(
        f'| std ddof=0 | {_fmt(gaps["rand6_minus_lp_mem"]["std_ddof0"])} | '
        f'{_fmt(gaps["count_mem_minus_lp_mem"]["std_ddof0"])} | '
        f'{_fmt(gaps["lp_room_minus_lp_mem"]["std_ddof0"])} |')
    return lines


def _markdown(result):
    seed_header = ' | '.join(str(seed) for seed in SEEDS)
    col_sep = '|---|---:|---:|---:|---:|---:|---:|---:|---:|'
    lines = [
        '# M4-W2c curious controller with memory',
        '',
        'Closed loop on the locked-rooms world. Bits are prequential, on observation tokens only.',
        'PROBE_cap is the mean over rooms 0..8 of min(probe_k, 4). Seeds 15–20.',
        '',
        '## PROBE_cap at T',
        '',
        f'| arm | {seed_header} | mean | std ddof=0 |',
        col_sep,
    ]
    for arm in ARMS:
        block = result['probe_cap_table'][arm]
        cells = [_fmt(row['value']) for row in block['per_seed']]
        lines.append(
            f'| {arm} | {" | ".join(cells)} | {_fmt(block["mean"])} | {_fmt(block["std_ddof0"])} |')
    lines += _gap_lines('## Gaps on PROBE_cap', result['gaps_probe_cap'])
    lines += [
        '',
        '## Raw PROBE at T',
        '',
        f'| arm | {seed_header} | mean | std ddof=0 |',
        col_sep,
    ]
    for arm in ARMS:
        block = result['probe_raw_table'][arm]
        cells = [_fmt(row['value']) for row in block['per_seed']]
        lines.append(
            f'| {arm} | {" | ".join(cells)} | {_fmt(block["mean"])} | {_fmt(block["std_ddof0"])} |')
    lines += _gap_lines('## Gaps on raw PROBE', result['gaps_probe_raw'])
    tv = result['action_share_mean']['lp_mem']
    lines += [
        '',
        '## Action shares',
        '',
        'Mean over seeds 15–20. TV is action 6. Reset is action 7.',
        '',
        f'lp_mem TV share mean {_fmt(tv["tv_mean"])}. std ddof=0 {_fmt(tv["tv_std_ddof0"])}.',
        '',
        '| arm | TV share | reset share | a0 | a1 | a2 | a3 | a4 | a5 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        block = result['action_share_mean'][arm]
        manip = ' | '.join(_fmt(v) for v in block['manipulation_mean'])
        lines.append(
            f'| {arm} | {_fmt(block["tv_mean"])} | {_fmt(block["reset_mean"])} | {manip} |')
    lines += [
        '',
        '## TV and reset shares by quarter',
        '',
        'Mean over seeds 15–20. Quarter width is T/4.',
        '',
        '| arm | quarter | TV share mean | TV std ddof=0 | reset share mean | reset std ddof=0 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        block = result['quarter_shares'][arm]
        for q in range(4):
            lines.append(
                f'| {arm} | {q} | {_fmt(block["tv_mean"][q])} | {_fmt(block["tv_std_ddof0"][q])} | '
                f'{_fmt(block["reset_mean"][q])} | {_fmt(block["reset_std_ddof0"][q])} |')
    lines += [
        '',
        '## Resets per 1,000 steps by quarter',
        '',
        'Mean over seeds 15–20.',
        '',
        '| arm | quarter | mean | std ddof=0 |',
        '|---|---:|---:|---:|',
    ]
    for arm in ARMS:
        block = result['quarter_shares'][arm]
        for q in range(4):
            lines.append(
                f'| {arm} | {q} | {_fmt(block["resets_per_1000_mean"][q], 3)} | '
                f'{_fmt(block["resets_per_1000_std_ddof0"][q], 3)} |')
    lines += [
        '',
        '## Depth',
        '',
        '| arm | seed | deepest room | first step | time in room 8 |',
        '|---|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        for row in result['depth'][arm]['per_seed']:
            lines.append(
                f'| {arm} | {row["world_seed"]} | {row["deepest_room"]} | '
                f'{row["deepest_first_step"]} | {row["time_in_room_8"]} |')
    lines += [
        '',
        '| arm | deepest mean | time in room 8 mean |',
        '|---|---:|---:|',
    ]
    for arm in ARMS:
        block = result['depth'][arm]
        lines.append(
            f'| {arm} | {_fmt(block["deepest_mean"])} | {_fmt(block["time_in_room_8_mean"], 1)} |')
    lines += [
        '',
        '## Time in room by quarter',
        '',
        'Mean over seeds 15–20. Rooms 0..8. Quarter width is T/4.',
        '',
    ]
    for arm in ARMS:
        lines.append(f'### {arm}')
        lines.append('')
        lines.append('| quarter | r0 | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 |')
        lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for q, row in enumerate(result['time_in_room_by_quarter'][arm]['mean']):
            cells = ' | '.join(_fmt(v, 1) for v in row)
            lines.append(f'| {q} | {cells} |')
        lines.append('')
    cap = result['capability']
    lines += [
        '## Capability check',
        '',
        f'World seed {cap["world_seed"]}, T = {cap["steps"]}, no learner. '
        'Reward 1 on the first step that ends in a room not entered before, else 0. '
        'Room 0 counts as entered at the start. Q updates every step.',
        '',
        f'Deepest room {cap["deepest_room"]}. First step {cap["deepest_first_step"]}.',
        '',
        'First entry step by room. A hyphen means the room was not entered by a step. '
        'Room 0 is a hyphen because it is entered at the start: '
        + ' '.join('-' if v is None else str(v) for v in cap['first_entry_step']) + '.',
        '',
        '## Lives',
        '',
    ]
    for row in result['lives']:
        share = row['action_share']
        lines += [
            f'### seed {row["world_seed"]} {row["arm"]}',
            '',
            f'Bits/observation {_fmt(row["bits_per_observation"])}. '
            f'Bits {_fmt(row["bits"], 3)}. Wall seconds {_fmt(row["wall_seconds"], 3)}. '
            f'Replay steps {row["replay_steps"]}. Finite {row["finite"]}.',
            '',
            f'PROBE_cap {_fmt(row["PROBE_cap_T"])}. PROBE {_fmt(row["PROBE_T"])}. '
            f'TV probe {_fmt(row["tv_probe_T"])}.',
            '',
            f'Deepest room {row["deepest_room"]}. First step {row["deepest_first_step"]}.',
            '',
            'Time in room, rooms 0..8: ' + ' '.join(str(v) for v in row['time_in_room']) + '.',
            '',
            'Time in room by quarter, rooms 0..8:',
            '',
        ]
        for q, counts in enumerate(row['time_in_room_by_quarter']):
            lines.append(f'{q}: ' + ' '.join(str(v) for v in counts))
        lines += [
            '',
            f'TV share {_fmt(share["tv"])}. Reset share {_fmt(share["reset"])}. '
            'Manipulation shares ' + ' '.join(_fmt(v) for v in share['manipulation']) + '.',
            '',
            'TV share by quarter: ' + ' '.join(_fmt(v) for v in row['tv_share_by_quarter']) + '.',
            '',
            'Reset share by quarter: '
            + ' '.join(_fmt(v) for v in row['reset_share_by_quarter']) + '.',
            '',
            'Resets per 1,000 steps by quarter: '
            + ' '.join(_fmt(v, 3) for v in row['resets_per_1000']) + '.',
            '',
            '| Quarter | Start | End | Observations | Bits | Bits/observation |',
            '|---:|---:|---:|---:|---:|---:|',
        ]
        for quarter in row['quarters']:
            lines.append(
                f'| {quarter["index"]} | {quarter["start_step"]} | {quarter["end_step"]} | '
                f'{quarter["observations"]} | {_fmt(quarter["bits"], 3)} | '
                f'{_fmt(quarter["bits_per_observation"])} |')
        lines += [
            '',
            '| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | PROBE_cap | TV |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for probe in row['probes']:
            cells = [_fmt(v) for v in probe['probe_k']]
            cap_bits = probe.get('PROBE_cap')
            if cap_bits is None:
                cap_bits = m4w2.probe_cap(probe['probe_k'])
            lines.append(
                f'| {probe["nominal_step"]} | {probe["at_step"]} | {" | ".join(cells)} | '
                f'{_fmt(probe["PROBE"])} | {_fmt(cap_bits)} | {_fmt(probe["tv_probe"])} |')
        lines.append('')
        lines.append(f'Token sha256 `{row["token_sha256"]}`.')
        lines.append('')
        if row['Q'] is not None:
            lines.append('Q at T. Columns are actions 0..5, TV, reset.')
            lines.append('')
            if row['arm'] in MEM_ARMS:
                lines.append(
                    'Memory rows are (room, p1, p2). p1 is the previous action and p2 the one before. '
                    '8 means none. All-zero rows are omitted here. results.json has every row.')
                lines.append('')
                lines += _memory_matrix_lines(row['Q'], drop_zeros=True)
                lines.append('')
                lines.append('Mean reward per (room, p1, p2, action) over the last quarter.')
                lines.append('')
                lines.append('Rows with no visit in the last quarter are omitted.')
                lines.append('')
                lines += _memory_matrix_lines(row['mean_reward_last_quarter'], drop_zeros=True)
            else:
                lines += _matrix_lines(row['Q'])
                lines.append('')
                lines.append('Mean reward per (room, action) over the last quarter.')
                lines.append('')
                lines += _matrix_lines(row['mean_reward_last_quarter'])
            lines.append('')
    lines += ['## Kernels', '']
    lines.append('| Kernel | Seeds | Version | Status | Elapsed seconds | Source | Script wall seconds |')
    lines.append('|---|---|---:|---|---:|---|---:|')
    for row in result['kernels']:
        version = '' if row['version'] is None else str(row['version'])
        status = '' if row['status'] is None else str(row['status'])
        seeds = ' '.join(str(s) for s in row['world_seeds'])
        lines.append(
            f'| {row["kernel_id"]} | {seeds} | {version} | {status} | '
            f'{_fmt(row["elapsed_seconds"], 3)} | {row["elapsed_source"]} | '
            f'{_fmt(row["session_wall_seconds"], 3)} |')
    lines.append('')
    lines.append(f'Elapsed seconds sum {_fmt(result["kernel_elapsed_seconds_sum"], 3)}.')
    lines += ['', '## Smoke', '']
    smoke_payload = result['smoke']
    lines.append(
        f'World seed {smoke_payload["world_seed"]}, T = {smoke_payload["steps"]}, '
        f'block {smoke_payload["block_size"]}. '
        f'Wall seconds {_fmt(smoke_payload["wall_seconds"], 3)}.')
    lines += [
        '',
        '| arm | parent | sha256 | ref sha256 | bits | ref bits | abs diff | probe max abs |',
        '|---|---|---|---|---:|---:|---:|---:|',
    ]
    for row in smoke_payload['equivalence']:
        lines.append(
            f'| {row["arm"]} | {row["parent_arm"]} | `{row["token_sha256"]}` | '
            f'`{row["ref_token_sha256"]}` | {_fmt(row["bits"])} | {_fmt(row["ref_bits"])} | '
            f'{row["abs_diff"]:.3e} | {row["probe_max_abs"]:.3e} |')
    state_row = smoke_payload['state_check']
    state_text = ' -> '.join(f'({r},{a},{b})' for r, a, b in state_row['states'])
    cap_row = smoke_payload['capability']
    lines += [
        '',
        f'State check, world seed {state_row["world_seed"]}. '
        f'key_0 {state_row["key_0"][0]} {state_row["key_0"][1]} {state_row["key_0"][2]}. '
        f'Manipulation {state_row["manipulation"]}. Match {state_row["match"]}.',
        '',
        state_text + '.',
        '',
        f'Capability check, world seed {cap_row["world_seed"]}, T = {cap_row["steps"]}, '
        'no learner. Deepest room '
        f'{cap_row["deepest_room"]}. First step {cap_row["deepest_first_step"]}.',
        '',
        '| arm | bits | Q finite | TV share | reset share | a0 | a1 | a2 | a3 | a4 | a5 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in smoke_payload['arms']:
        share = row['action_share']
        manip = ' | '.join(_fmt(v) for v in share['manipulation'])
        q_cell = '' if row['Q_finite'] is None else str(row['Q_finite'])
        lines.append(
            f'| {row["arm"]} | {_fmt(row["bits"])} | {q_cell} | '
            f'{_fmt(share["tv"])} | {_fmt(share["reset"])} | {manip} |')
    lines += [
        '',
        '## Protocol',
        '',
        '- Rooms 0..8. Actions 0..5 manipulate, 6 is the TV, 7 is reset. Vocabulary 152.',
        '- Closed loop, chunks of 32 actions. Q is frozen inside a chunk. Score, live Adam step, replay Adam step, then Q updates in order.',
        '- lp_mem and lp_room use prog_lp. Context is (room before the action, action). K = 16. Visit n ≤ 32 rewards 1. Visit n > 32 rewards mean(bits at visits n−32..n−17) minus mean(bits at visits n−16..n−1). The current visit is not in the window. Its bits_before is appended after the reward.',
        '- count_mem reward is 1/sqrt(n) on the memory state-action pair, n including the current visit.',
        '- lp_mem and count_mem: state is (room, p1, p2). p1 is the previous action, p2 the one before. Each is 0..8, and 8 means none. A step that changes the room, including a reset that leaves the room, sets p1 and p2 to 8. A step that stays in the room sets p1 to that action and p2 to the old p1. A TV press stays in the room. Q is 729 by 8, zeros at the start.',
        '- lp_room is M4-W2b prog_lp with room-only state. Q is 9 by 8. rand6 is M4-W2b rand6.',
        '- ε = 0.1 over actions 0..5 only. Greedy argmax is over all 8 actions. Ties are uniform. α = 0.1, γ = 0.9. Controller rng is default_rng(7000 + world_seed).',
        '- rand6 uses the W2-0 policy and rng. Learner seed 0. Replay seed 30000. Adam lr 0.001, clip 1.0, replay batch 8.',
        '- GRU embedding 32, hidden 128, 1 layer, readout 152. 86680 parameters. CPU, 4 threads.',
        '- Checkpoints T/4, T/2 and T. Probe boundaries are 15008, 30016 and 60000.',
        '- PROBE_cap caps each room probe at 4 bits. Seeds 15, 16, 17, 18, 19 and 20.',
        '- Kernels ezzzzzx/m4w2c-a, m4w2c-b and m4w2c-c. Private, CPU, no internet. Code dataset ezzzzzx/pdp-code-m4w2c.',
        '',
    ]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--assemble', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.assemble:
        assemble()
    else:
        parser.print_help()
        raise SystemExit(2)


if __name__ == '__main__':
    main()
