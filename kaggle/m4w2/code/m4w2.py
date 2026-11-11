"""M4-W2: closed-loop controllers on the locked-rooms world. Numbers only.

World, tokens, learner, replay and probe are imported from w2_world and
w2_headroom. A chunk is 32 actions (64 tokens). Inside a chunk the controller
acts with the values it had at the start of the chunk. The learner then scores,
takes the live step and the replay step, and the same chunk is scored again
under no_grad from the same initial hidden state. Q updates run after that,
one transition at a time. No verdict. Does not commit.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import w2_headroom as head
import w2_world as world

ARMS = ('rand6', 'rand8', 'oracle', 'prog_q', 'err_q', 'count_q')
FIXED = ('rand6', 'rand8', 'oracle')
Q_ARMS = ('prog_q', 'err_q', 'count_q')
SEEDS = (3, 4, 5, 6, 7, 8)
KERNELS = (
    ('ezzzzzx/m4w2-a', (3, 4)),
    ('ezzzzzx/m4w2-b', (5, 6)),
    ('ezzzzzx/m4w2-c', (7, 8)),
)
SMOKE_STEPS = 640
SMOKE_BLOCK = 32
SMOKE_SEED = 0
CAP = 4.0
EPS = 0.1
ALPHA = 0.1
GAMMA = 0.9
CTRL_SEED = 7_000
OUT_DIR = Path('results/m4w2')
CHUNK_STEPS = head.CHUNK // 2


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def _fmt(value, digits=6):
    if value is None:
        return ''
    return f'{value:.{digits}f}'


def probe_cap(probe_k, cap=CAP):
    return float(np.mean([min(float(v), cap) for v in probe_k]))


def _eps_greedy(q_table, room, rng):
    """ε-greedy. Ties on the max are uniform. Frozen Q is the caller's job."""
    if float(rng.random()) < EPS:
        return int(rng.integers(0, world.N_ACTIONS))
    row = q_table[int(room)]
    best = row.max()
    cands = np.flatnonzero(row == best)
    if int(cands.size) == 1:
        return int(cands[0])
    return int(rng.choice(cands))


class _Chooser:
    def __init__(self, arm, world_seed, steps):
        self.arm = arm
        self.actions = None
        self.explore = None
        self.ctrl = None
        if arm == 'rand8':
            self.actions = np.random.default_rng(world.POLICY_SEED + world_seed).integers(
                0, world.N_ACTIONS, size=steps)
        elif arm == 'rand6':
            self.actions = np.random.default_rng(world.POLICY_SEED + world_seed).integers(
                0, world.N_MANIP, size=steps)
        elif arm == 'oracle':
            self.explore = np.random.default_rng(world.ORACLE_SEED + world_seed)
        elif arm in Q_ARMS:
            self.ctrl = np.random.default_rng(CTRL_SEED + world_seed)
        else:
            raise ValueError(arm)

    def act(self, env, t, block_size, q_table):
        if self.actions is not None:
            return int(self.actions[t])
        if self.arm == 'oracle':
            return int(world._oracle_action(env, t, block_size, self.explore))
        return _eps_greedy(q_table, env.room, self.ctrl)


def _reward(arm, bits_before, bits_after, room, action, visits):
    if arm == 'prog_q':
        return float(bits_before - bits_after)
    if arm == 'err_q':
        return float(bits_before)
    if arm == 'count_q':
        visits[room, action] += 1
        n = int(visits[room, action])
        return 1.0 / math.sqrt(n)
    raise ValueError(arm)


def _q_update(q_table, room, action, reward, room_next):
    current = float(q_table[room, action])
    target = float(reward) + GAMMA * float(q_table[room_next].max())
    q_table[room, action] = current + ALPHA * (target - current)


def run_closed(world_seed, arm, steps, block_size, device, log_every=200, keep_trace=False):
    """One closed-loop life. Fixed arms use the W2-0 policies and rngs."""
    if arm not in ARMS:
        raise ValueError(arm)
    if steps % 4 != 0:
        raise AssertionError(f'T={steps} is not divisible by 4')
    if CHUNK_STEPS * 2 != head.CHUNK:
        raise AssertionError('chunk is not 32 steps')
    t0 = time.monotonic()
    q_arm = arm in Q_ARMS
    env = world.World(world_seed)
    chooser = _Chooser(arm, world_seed, steps)
    episodes = world.build_probe_episodes(world_seed)
    torch.manual_seed(head.SEED)
    model = head.TokenGRU().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != head.expected_params():
        raise AssertionError(f'parameter count {n_params} != {head.expected_params()}')
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=head.LR)
    replay_rng = np.random.default_rng(head.REPLAY_SEED + head.SEED)
    q_table = np.zeros((world.N_ROOMS, world.N_ACTIONS), dtype=np.float64)
    visits = np.zeros((world.N_ROOMS, world.N_ACTIONS), dtype=np.int64)
    reward_sum = np.zeros((world.N_ROOMS, world.N_ACTIONS), dtype=np.float64)
    reward_n = np.zeros((world.N_ROOMS, world.N_ACTIONS), dtype=np.int64)
    tokens = np.empty(steps * 2, dtype=np.int64)
    actions = np.empty(steps, dtype=np.int64)
    rooms_after = np.empty(steps, dtype=np.int64)
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
        # Act with the Q table from the start of the chunk. It is not written here.
        for i in range(n_local):
            t = step0 + i
            rooms_before[i] = int(env.room)
            action = chooser.act(env, t, block_size, q_table)
            room_after, y = env.step(action)
            actions[t] = action
            rooms_after[t] = int(room_after)
            tokens[2 * t] = int(action)
            tokens[2 * t + 1] = world.obs_token(room_after, y)
            room_counts[int(room_after)] += 1
            if int(room_after) > deepest:
                deepest = int(room_after)
                first_step = int(t)
        state_for_rescore = None if state is None else state.detach().clone()
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
        with torch.no_grad():
            losses_after, _after_state = model.nll(prev_t, tgt_t, state_for_rescore)
            bits_after = head._obs_bits(losses_after)
            del losses_after
        if bits_after.shape != bits_before.shape:
            raise AssertionError('re-score length')
        if q_arm:
            for i in range(n_local):
                room_b = int(rooms_before[i])
                action = int(actions[step0 + i])
                room_n = int(rooms_after[step0 + i])
                reward = _reward(arm, bits_before[i], bits_after[i], room_b, action, visits)
                if not math.isfinite(reward):
                    raise FloatingPointError(f'{arm} reward at step {step0 + i}')
                _q_update(q_table, room_b, action, reward, room_n)
                if step0 + i >= last_quarter:
                    reward_sum[room_b, action] += reward
                    reward_n[room_b, action] += 1
        del bits_before, bits_after
        chunk_count += 1
        pos = end
        steps_done = pos // 2
        while next_probe < len(nominal) and steps_done >= nominal[next_probe]:
            row = head.probe_model(model, episodes, device)
            row['nominal_step'] = int(nominal[next_probe])
            row['at_step'] = int(steps_done)
            row['PROBE_cap'] = probe_cap(row['probe_k'])
            probes.append(row)
            print(
                f'M4W2 seed={world_seed} arm={arm} probe nominal={row["nominal_step"]} '
                f'at_step={row["at_step"]} PROBE={row["PROBE"]:.4f} '
                f'cap={row["PROBE_cap"]:.4f} tv={row["tv_probe"]:.4f} '
                f'wall_s={time.monotonic() - t0:.1f}',
                flush=True)
            next_probe += 1
        if chunk_count % log_every == 0 or pos == n_tokens:
            print(
                f'M4W2 seed={world_seed} arm={arm} steps={steps_done}/{steps} '
                f'bpo={total_bits / max(n_obs, 1):.4f} replay={replay_steps} '
                f'wall_s={time.monotonic() - t0:.1f}',
                flush=True)
    if n_obs != steps or next_probe != len(nominal):
        raise AssertionError(f'scored {n_obs} observations, probes {next_probe}')
    if abs(float(quarter_bits.sum()) - total_bits) > 1e-4:
        raise AssertionError('quarter bits do not sum to the life')
    if int(quarter_count.sum()) != steps:
        raise AssertionError('quarter counts do not sum to T')
    if int(room_counts.sum()) != steps:
        raise AssertionError('time-in-room does not sum to T')
    if q_arm and not np.isfinite(q_table).all():
        raise FloatingPointError(f'{arm} Q is not finite')
    sha = world.token_sha256(tokens)
    if arm in FIXED:
        ref = world.rollout(world_seed, arm, steps, block_size)
        if ref['token_sha256'] != sha:
            raise AssertionError(f'{arm} token sha drifted from the fixed policy')
        if ref['time_in_room'] != [int(v) for v in room_counts]:
            raise AssertionError(f'{arm} time-in-room drifted from the fixed policy')
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
    counts = np.bincount(actions, minlength=world.N_ACTIONS)
    if int(counts.sum()) != steps:
        raise AssertionError('action counts do not sum to T')
    share = counts.astype(np.float64) / steps
    if abs(float(share.sum()) - 1.0) > 1e-9 or np.any(share < 0):
        raise AssertionError('action share is not a distribution')
    action_share = {
        'manipulation': [float(share[i]) for i in range(world.N_MANIP)],
        'tv': float(share[world.TV]),
        'reset': float(share[world.RESET]),
    }
    finite = all(row['finite'] for row in probes) and math.isfinite(total_bits)
    if q_arm:
        mean_reward = []
        for r in range(world.N_ROOMS):
            row = []
            for a in range(world.N_ACTIONS):
                n = int(reward_n[r, a])
                row.append(None if n == 0 else float(reward_sum[r, a] / n))
            mean_reward.append(row)
        q_out = [[float(v) for v in row] for row in q_table]
        reward_count = [[int(v) for v in row] for row in reward_n]
        controller = {
            'eps': EPS,
            'alpha': ALPHA,
            'gamma': GAMMA,
            'seed': int(CTRL_SEED + world_seed),
        }
    else:
        mean_reward = None
        q_out = None
        reward_count = None
        controller = None
    wall = time.monotonic() - t0
    result = {
        'world_seed': int(world_seed),
        'arm': arm,
        'steps': int(steps),
        'block_size': int(block_size),
        'token_sha256': sha,
        'time_in_room': [int(v) for v in room_counts],
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
        'resets_per_1000': resets_per_1000,
        'Q': q_out,
        'mean_reward_last_quarter': mean_reward,
        'reward_count_last_quarter': reward_count,
        'controller': controller,
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
    }
    if keep_trace:
        result['actions'] = [int(v) for v in actions]
    final = probes[-1]
    print(
        f'M4W2 done seed={world_seed} arm={arm} bpo={result["bits_per_observation"]:.4f} '
        f'PROBE={final["PROBE"]:.4f} cap={final["PROBE_cap"]:.4f} '
        f'tv_share={action_share["tv"]:.4f} reset_share={action_share["reset"]:.4f} '
        f'deepest={deepest} room8={int(room_counts[8])} wall_s={wall:.1f}',
        flush=True)
    return result


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def rescore_check(device):
    """Frozen learner: bits_after equals bits_before on one chunk, to 1e-6.

    Chunk 0 starts from a zero state. Chunk 1 starts from the carried state
    after chunk 0 has been updated. The checked forwards themselves do not update.
    """
    steps = CHUNK_STEPS * 2
    life = world.rollout(SMOKE_SEED, 'rand6', steps, SMOKE_BLOCK)
    tokens = life['tokens']
    torch.manual_seed(head.SEED)
    model = head.TokenGRU().to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=head.LR)
    state = None
    reported = []
    for chunk_i, pos in enumerate((0, head.CHUNK)):
        end = pos + head.CHUNK
        prev, tgt = head.live_batch(tokens, pos, end)
        prev_t, tgt_t = head._as_batch(prev, tgt, device)
        held = None if state is None else state.detach().clone()
        losses, new_state = model.nll(prev_t, tgt_t, state)
        bits_before = head._obs_bits(losses)
        with torch.no_grad():
            losses_after, _ = model.nll(prev_t, tgt_t, held)
            bits_after = head._obs_bits(losses_after)
            del losses_after
        err = _max_abs(bits_before, bits_after)
        reported.append({'chunk': chunk_i, 'max_abs': err, 'steps': int(bits_before.shape[0])})
        if err > 1e-6:
            raise AssertionError(f're-score chunk {chunk_i} max abs {err}')
        if chunk_i == 0:
            carried = new_state.detach()
            head._step(opt, model, losses[1::2])
            state = carried
        del losses
    print(
        'SMOKE rescore ' + ' '.join(f'c{row["chunk"]}={row["max_abs"]:.3e}' for row in reported),
        flush=True)
    return {'chunks': reported, 'max_abs': max(row['max_abs'] for row in reported)}


def smoke():
    head.configure()
    t0 = time.monotonic()
    device = torch.device('cpu')
    rescore = rescore_check(device)
    equivalence = []
    for arm in FIXED:
        ref = head.run_life(SMOKE_SEED, arm, SMOKE_STEPS, SMOKE_BLOCK, device, log_every=50)
        closed = run_closed(
            SMOKE_SEED, arm, SMOKE_STEPS, SMOKE_BLOCK, device, log_every=50)
        diff = abs(float(closed['bits']) - float(ref['bits']))
        sha_match = closed['token_sha256'] == ref['token_sha256']
        probe_diff = []
        for c_row, r_row in zip(closed['probes'], ref['probes']):
            probe_diff.append(_max_abs(c_row['probe_k'] + [c_row['tv_probe']],
                                        r_row['probe_k'] + [r_row['tv_probe']]))
        row = {
            'arm': arm,
            'token_sha256': closed['token_sha256'],
            'ref_token_sha256': ref['token_sha256'],
            'sha_match': bool(sha_match),
            'bits': float(closed['bits']),
            'ref_bits': float(ref['bits']),
            'abs_diff': float(diff),
            'probe_max_abs': float(max(probe_diff)),
        }
        equivalence.append(row)
        print(
            f'SMOKE equivalence arm={arm} sha_match={sha_match} '
            f'bits={row["bits"]:.6f} ref_bits={row["ref_bits"]:.6f} '
            f'abs_diff={diff:.3e} probe_max_abs={row["probe_max_abs"]:.3e}',
            flush=True)
        print(f'SMOKE sha {arm} {closed["token_sha256"]}', flush=True)
        print(f'SMOKE sha_ref {arm} {ref["token_sha256"]}', flush=True)
        if not sha_match or diff > 1e-6:
            raise AssertionError(f'{arm} closed loop does not match W2-0')
        if row['probe_max_abs'] > 1e-6:
            raise AssertionError(f'{arm} probe drifted from W2-0')
    q_rows = []
    traces = {}
    for arm in Q_ARMS:
        closed = run_closed(
            SMOKE_SEED, arm, SMOKE_STEPS, SMOKE_BLOCK, device, log_every=50, keep_trace=True)
        q_table = np.asarray(closed['Q'], dtype=np.float64)
        share = closed['action_share']
        manip = share['manipulation']
        total = float(sum(manip) + share['tv'] + share['reset'])
        ok = (
            closed['finite']
            and np.isfinite(q_table).all()
            and q_table.shape == (world.N_ROOMS, world.N_ACTIONS)
            and abs(total - 1.0) <= 1e-9
            and min(manip) >= 0
            and share['tv'] >= 0
            and share['reset'] >= 0
        )
        print(
            f'SMOKE q arm={arm} bits={closed["bits"]:.6f} finite={closed["finite"]} '
            f'Q_finite={bool(np.isfinite(q_table).all())} '
            f'tv={share["tv"]:.6f} reset={share["reset"]:.6f} '
            f'manip={" ".join(f"{v:.6f}" for v in manip)}',
            flush=True)
        if not ok:
            raise AssertionError(f'{arm} smoke is not finite or the action share is invalid')
        traces[arm] = closed['actions']
        q_rows.append({
            'arm': arm,
            'bits': float(closed['bits']),
            'bits_per_observation': float(closed['bits_per_observation']),
            'finite': bool(closed['finite']),
            'Q_finite': True,
            'action_share': share,
            'PROBE_cap_T': closed['PROBE_cap_T'],
            'PROBE_T': closed['PROBE_T'],
            'wall_seconds': closed['wall_seconds'],
        })
    first = {arm: traces[arm][:CHUNK_STEPS] for arm in Q_ARMS}
    same_first = first['prog_q'] == first['err_q'] == first['count_q']
    if not same_first:
        raise AssertionError('Q arms diverged inside the first frozen chunk')
    payload = {
        'world_seed': SMOKE_SEED,
        'steps': SMOKE_STEPS,
        'block_size': SMOKE_BLOCK,
        'equivalence': equivalence,
        'q_arms': q_rows,
        'rescore': rescore,
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
    head.configure()
    t0 = time.monotonic()
    out = Path(out)
    device = torch.device('cpu')
    seeds = tuple(int(s) for s in world_seeds)
    print(
        f'M4W2 kernel={kernel_id} seeds={list(seeds)} device={device} '
        f'cuda={torch.cuda.is_available()} torch={torch.__version__} '
        f'numpy={np.__version__} threads={torch.get_num_threads()}',
        flush=True)
    if device.type != 'cpu':
        raise SystemExit('M4-W2 is a CPU run')
    lives = []
    expect_chunks = math.ceil((steps * 2) / head.CHUNK)
    for world_seed in seeds:
        for arm in ARMS:
            print(f'M4W2 start seed={world_seed} arm={arm}', flush=True)
            row = run_closed(world_seed, arm, int(steps), int(block_size), device)
            if not row['finite']:
                raise FloatingPointError(f'{arm} seed {world_seed} is not finite')
            if row['chunk_count'] != expect_chunks or row['replay_steps'] != expect_chunks:
                raise AssertionError(
                    f'{arm} chunks {row["chunk_count"]} replay {row["replay_steps"]} '
                    f'!= {expect_chunks}')
            if row['probes'][-1]['at_step'] != int(steps):
                raise AssertionError('final probe is not at T')
            if arm in Q_ARMS:
                if row['controller']['seed'] != CTRL_SEED + int(world_seed):
                    raise AssertionError('controller seed')
                if abs(row['controller']['eps'] - EPS) > 0 or abs(row['controller']['alpha'] - ALPHA) > 0:
                    raise AssertionError('controller step size')
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
        'eps': EPS,
        'alpha': ALPHA,
        'gamma': GAMMA,
        'controller_seed_base': CTRL_SEED,
        'probe_cap': CAP,
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
    text = Path(log_path).read_text(errors='replace')
    times = []
    key = '"time":'
    start = 0
    while True:
        at = text.find(key, start)
        if at < 0:
            break
        at += len(key)
        end = at
        while end < len(text) and text[end] in ' 0123456789.':
            end += 1
        try:
            times.append(float(text[at:end]))
        except ValueError:
            pass
        start = end
    if not times:
        return None
    return max(times)


def _status_name(status):
    text = str(status).strip().strip('"')
    if '.' in text:
        text = text.rsplit('.', 1)[-1]
    return text


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
    share = row['action_share']
    total = float(sum(share['manipulation']) + share['tv'] + share['reset'])
    if abs(total - 1.0) > 1e-8:
        raise SystemExit(f'{row["arm"]} action share sums to {total}')
    if len(share['manipulation']) != world.N_MANIP:
        raise SystemExit('manipulation share length')
    cap = probe_cap(row['probes'][-1]['probe_k'])
    if abs(cap - float(row['PROBE_cap_T'])) > 1e-9:
        raise SystemExit('PROBE_cap does not match probe_k')
    if row['arm'] in Q_ARMS:
        q_table = np.asarray(row['Q'], dtype=np.float64)
        if q_table.shape != (world.N_ROOMS, world.N_ACTIONS) or not np.isfinite(q_table).all():
            raise SystemExit(f'{row["arm"]} Q is missing or not finite')
        ctrl = row['controller']
        if int(ctrl['seed']) != CTRL_SEED + int(world_seed):
            raise SystemExit('controller seed')
        if abs(float(ctrl['eps']) - EPS) > 0 or abs(float(ctrl['alpha']) - ALPHA) > 0:
            raise SystemExit('controller hyperparameters')
        if abs(float(ctrl['gamma']) - GAMMA) > 0:
            raise SystemExit('controller gamma')
    elif row['Q'] is not None:
        raise SystemExit(f'{row["arm"]} stored a Q table')


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
        prog = float(loaded[seed]['prog_q'][key])
        rows.append({
            'world_seed': int(seed),
            'rand6_minus_prog_q': float(loaded[seed]['rand6'][key]) - prog,
            'count_q_minus_prog_q': float(loaded[seed]['count_q'][key]) - prog,
            'err_q_minus_prog_q': float(loaded[seed]['err_q'][key]) - prog,
            'prog_q_minus_oracle': prog - float(loaded[seed]['oracle'][key]),
        })
    summary = {}
    for name in ('rand6_minus_prog_q', 'count_q_minus_prog_q', 'err_q_minus_prog_q', 'prog_q_minus_oracle'):
        mean, std = _mean_std([row[name] for row in rows])
        summary[name] = {'mean': mean, 'std_ddof0': std}
    return {'per_seed': rows, 'mean_std_ddof0': summary}


def assemble(out_dir=OUT_DIR, kernel_meta=None):
    out_dir = Path(out_dir)
    if kernel_meta is None:
        meta_path = out_dir / 'kernel_meta.json'
        kernel_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    smoke_payload = json.loads((out_dir / 'smoke.json').read_text())
    loaded = {seed: {} for seed in SEEDS}
    raw_kernels = []
    for kernel_id, seeds in KERNELS:
        slug = kernel_id.rsplit('/', 1)[-1]
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
        by_key = {}
        for row in payload['lives']:
            _require_life(row, row['world_seed'])
            if int(row['world_seed']) not in seeds:
                raise SystemExit(f'{kernel_id} unexpected seed {row["world_seed"]}')
            if row['arm'] in FIXED:
                local = world.rollout(int(row['world_seed']), row['arm'], head.T_FULL, head.BLOCK_FULL)
                if local['token_sha256'] != row['token_sha256']:
                    raise SystemExit(
                        f'seed {row["world_seed"]} {row["arm"]} token sha local '
                        f'{local["token_sha256"]} != kernel {row["token_sha256"]}')
                if local['time_in_room'] != row['time_in_room']:
                    raise SystemExit(f'seed {row["world_seed"]} {row["arm"]} time-in-room mismatch')
                if local['deepest_first_step'] != row['deepest_first_step']:
                    raise SystemExit(f'seed {row["world_seed"]} {row["arm"]} deepest step mismatch')
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
        slug = kernel_id.rsplit('/', 1)[-1]
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
        'eps': EPS,
        'alpha': ALPHA,
        'gamma': GAMMA,
        'controller_seed_base': CTRL_SEED,
        'probe_cap': CAP,
        'arms': list(ARMS),
        'world_seeds': list(SEEDS),
        'lives': lives_out,
        'probe_cap_table': _arm_table(loaded, 'PROBE_cap_T'),
        'probe_raw_table': _arm_table(loaded, 'PROBE_T'),
        'gaps_probe_cap': _gap_table(loaded, 'PROBE_cap_T'),
        'gaps_probe_raw': _gap_table(loaded, 'PROBE_T'),
        'action_share_mean': share_mean,
        'depth': depth,
        'kernels': kernels,
        'kernel_elapsed_seconds_sum': float(sum(row['elapsed_seconds'] for row in kernels)),
        'smoke': smoke_payload,
    }
    dump(out_dir / 'results.json', result)
    lines = _markdown(result)
    (out_dir / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(
        'M4W2 assemble '
        + ' '.join(
            f'{name}={result["gaps_probe_cap"]["mean_std_ddof0"][name]["mean"]:.4f}'
            for name in result['gaps_probe_cap']['mean_std_ddof0']),
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


def _markdown(result):
    lines = [
        '# M4-W2 curious controllers',
        '',
        'Closed loop on the locked-rooms world. Bits are prequential, on observation tokens only.',
        'PROBE_cap is the mean over rooms 0..8 of min(probe_k, 4). Seeds 3–8.',
        '',
        '## PROBE_cap at T',
        '',
        '| arm | 3 | 4 | 5 | 6 | 7 | 8 | mean | std ddof=0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        block = result['probe_cap_table'][arm]
        cells = [_fmt(row['value']) for row in block['per_seed']]
        lines.append(
            f'| {arm} | {" | ".join(cells)} | {_fmt(block["mean"])} | {_fmt(block["std_ddof0"])} |')
    lines += [
        '',
        '## Gaps on PROBE_cap',
        '',
        '| world_seed | rand6 − prog_q | count_q − prog_q | err_q − prog_q | prog_q − oracle |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in result['gaps_probe_cap']['per_seed']:
        lines.append(
            f'| {row["world_seed"]} | {_fmt(row["rand6_minus_prog_q"])} | '
            f'{_fmt(row["count_q_minus_prog_q"])} | {_fmt(row["err_q_minus_prog_q"])} | '
            f'{_fmt(row["prog_q_minus_oracle"])} |')
    gaps = result['gaps_probe_cap']['mean_std_ddof0']
    lines.append(
        f'| mean | {_fmt(gaps["rand6_minus_prog_q"]["mean"])} | '
        f'{_fmt(gaps["count_q_minus_prog_q"]["mean"])} | '
        f'{_fmt(gaps["err_q_minus_prog_q"]["mean"])} | '
        f'{_fmt(gaps["prog_q_minus_oracle"]["mean"])} |')
    lines.append(
        f'| std ddof=0 | {_fmt(gaps["rand6_minus_prog_q"]["std_ddof0"])} | '
        f'{_fmt(gaps["count_q_minus_prog_q"]["std_ddof0"])} | '
        f'{_fmt(gaps["err_q_minus_prog_q"]["std_ddof0"])} | '
        f'{_fmt(gaps["prog_q_minus_oracle"]["std_ddof0"])} |')
    lines += [
        '',
        '## Raw PROBE at T',
        '',
        '| arm | 3 | 4 | 5 | 6 | 7 | 8 | mean | std ddof=0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        block = result['probe_raw_table'][arm]
        cells = [_fmt(row['value']) for row in block['per_seed']]
        lines.append(
            f'| {arm} | {" | ".join(cells)} | {_fmt(block["mean"])} | {_fmt(block["std_ddof0"])} |')
    lines += [
        '',
        '## Gaps on raw PROBE',
        '',
        '| world_seed | rand6 − prog_q | count_q − prog_q | err_q − prog_q | prog_q − oracle |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in result['gaps_probe_raw']['per_seed']:
        lines.append(
            f'| {row["world_seed"]} | {_fmt(row["rand6_minus_prog_q"])} | '
            f'{_fmt(row["count_q_minus_prog_q"])} | {_fmt(row["err_q_minus_prog_q"])} | '
            f'{_fmt(row["prog_q_minus_oracle"])} |')
    gaps = result['gaps_probe_raw']['mean_std_ddof0']
    lines.append(
        f'| mean | {_fmt(gaps["rand6_minus_prog_q"]["mean"])} | '
        f'{_fmt(gaps["count_q_minus_prog_q"]["mean"])} | '
        f'{_fmt(gaps["err_q_minus_prog_q"]["mean"])} | '
        f'{_fmt(gaps["prog_q_minus_oracle"]["mean"])} |')
    lines.append(
        f'| std ddof=0 | {_fmt(gaps["rand6_minus_prog_q"]["std_ddof0"])} | '
        f'{_fmt(gaps["count_q_minus_prog_q"]["std_ddof0"])} | '
        f'{_fmt(gaps["err_q_minus_prog_q"]["std_ddof0"])} | '
        f'{_fmt(gaps["prog_q_minus_oracle"]["std_ddof0"])} |')
    lines += [
        '',
        '## Action shares',
        '',
        'Mean over seeds 3–8. TV is action 6. Reset is action 7.',
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
    lines += ['', '## Lives', '']
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
            f'TV share {_fmt(share["tv"])}. Reset share {_fmt(share["reset"])}. '
            'Manipulation shares ' + ' '.join(_fmt(v) for v in share['manipulation']) + '.',
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
            cap = probe.get('PROBE_cap')
            if cap is None:
                cap = probe_cap(probe['probe_k'])
            lines.append(
                f'| {probe["nominal_step"]} | {probe["at_step"]} | {" | ".join(cells)} | '
                f'{_fmt(probe["PROBE"])} | {_fmt(cap)} | {_fmt(probe["tv_probe"])} |')
        lines.append('')
        lines.append(f'Token sha256 `{row["token_sha256"]}`.')
        lines.append('')
        if row['Q'] is not None:
            lines.append('Q at T. Columns are actions 0..5, TV, reset.')
            lines.append('')
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
        f'oracle block {smoke_payload["block_size"]}. '
        f'Wall seconds {_fmt(smoke_payload["wall_seconds"], 3)}.')
    lines += [
        '',
        '| arm | sha256 | ref sha256 | bits | ref bits | abs diff | probe max abs |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for row in smoke_payload['equivalence']:
        lines.append(
            f'| {row["arm"]} | `{row["token_sha256"]}` | `{row["ref_token_sha256"]}` | '
            f'{_fmt(row["bits"])} | {_fmt(row["ref_bits"])} | {row["abs_diff"]:.3e} | '
            f'{row["probe_max_abs"]:.3e} |')
    lines += [
        '',
        '| arm | bits | Q finite | TV share | reset share | a0 | a1 | a2 | a3 | a4 | a5 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in smoke_payload['q_arms']:
        share = row['action_share']
        manip = ' | '.join(_fmt(v) for v in share['manipulation'])
        lines.append(
            f'| {row["arm"]} | {_fmt(row["bits"])} | {row["Q_finite"]} | '
            f'{_fmt(share["tv"])} | {_fmt(share["reset"])} | {manip} |')
    rescore = smoke_payload['rescore']
    lines += [
        '',
        f'Re-score max abs diff {rescore["max_abs"]:.3e}.',
        '',
        '## Protocol',
        '',
        '- Rooms 0..8. Actions 0..5 manipulate, 6 is the TV, 7 is reset. Vocabulary 152.',
        '- Closed loop, chunks of 32 actions. Q is frozen inside a chunk. Score, live Adam step, replay Adam step, re-score under no_grad from the same initial hidden state, then Q updates in order.',
        '- prog_q reward is bits_before minus bits_after, unclipped. err_q reward is bits_before. count_q reward is 1/sqrt(n) with n including the current visit.',
        '- Q is 9 by 8, zeros at the start. ε = 0.1, α = 0.1, γ = 0.9. Ties on the greedy action are uniform. Controller rng is default_rng(7000 + world_seed).',
        '- rand8, rand6 and the oracle use the W2-0 policies and rngs. Oracle block size is 512. Learner seed 0. Replay seed 30000. Adam lr 0.001, clip 1.0, replay batch 8.',
        '- GRU embedding 32, hidden 128, 1 layer, readout 152. 86680 parameters. CPU, 4 threads.',
        '- Checkpoints T/4, T/2 and T. Probe boundaries are 15008, 30016 and 60000.',
        '- PROBE_cap caps each room probe at 4 bits. Seeds 3, 4, 5, 6, 7 and 8.',
        '- Kernels ezzzzzx/m4w2-a, m4w2-b and m4w2-c. Private, CPU, no internet. Code dataset ezzzzzx/pdp-code-m4w2.',
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
