"""M4-W2b: visit-to-visit learning progress on the locked-rooms world.

Imports the world, the learner and M4-W2's Q update. ε-exploration draws from
manipulations 0..5 only. prog_lp rewards the drop in a context's prequential
loss between an older window and a recent window. Numbers only. No verdict.
Does not commit.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import m4w2
import w2_headroom as head
import w2_world as world

ARMS = ('rand6', 'prog_lp', 'prog_q', 'count_q', 'err_q')
Q_ARMS = ('prog_lp', 'prog_q', 'count_q', 'err_q')
SEEDS = (9, 10, 11, 12, 13, 14)
KERNELS = (
    ('ezzzzzx/m4w2b-a', (9, 10)),
    ('ezzzzzx/m4w2b-b', (11, 12)),
    ('ezzzzzx/m4w2b-c', (13, 14)),
)
SMOKE_STEPS = 640
SMOKE_BLOCK = 32
SMOKE_SEED = 0
EPS_CHECK_STEPS = 2_000
K_LP = 16
OUT_DIR = Path('results/m4w2b')
CHUNK_STEPS = head.CHUNK // 2
GAP_NAMES = (
    'rand6_minus_prog_lp',
    'count_q_minus_prog_lp',
    'prog_q_minus_prog_lp',
    'err_q_minus_prog_lp',
)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def _fmt(value, digits=6):
    if value is None:
        return ''
    return f'{value:.{digits}f}'


def _require_parent_constants():
    if m4w2.EPS != 0.1 or m4w2.ALPHA != 0.1 or abs(m4w2.GAMMA - 0.9) > 0:
        raise AssertionError('parent controller constants')
    if m4w2.CTRL_SEED != 7_000 or m4w2.CAP != 4.0:
        raise AssertionError('parent seed or cap')
    if world.N_MANIP != 6 or world.N_ACTIONS != 8 or world.TV != 6 or world.RESET != 7:
        raise AssertionError('action layout')
    if world.N_ROOMS != 9 or head.CHUNK != 64 or head.T_FULL != 60_000:
        raise AssertionError('world or horizon')
    if CHUNK_STEPS != 32:
        raise AssertionError('chunk is not 32 steps')


def prog_lp_reward(prev_bits, n):
    """Reward at 1-based visit n. prev_bits is visits 1..n-1, not visit n.

    For n ≤ 2K the reward is 1. For n > 2K it is the mean of visits
    n−2K..n−K−1 minus the mean of visits n−K..n−1.
    """
    n = int(n)
    if n < 1:
        raise AssertionError('visit index')
    if len(prev_bits) != n - 1:
        raise AssertionError(f'history length {len(prev_bits)} != {n - 1}')
    if n <= 2 * K_LP:
        return 1.0
    older = np.asarray(prev_bits[n - 2 * K_LP - 1:n - K_LP - 1], dtype=np.float64)
    recent = np.asarray(prev_bits[n - K_LP - 1:n - 1], dtype=np.float64)
    if int(older.shape[0]) != K_LP or int(recent.shape[0]) != K_LP:
        raise AssertionError('window length')
    return float(older.mean() - recent.mean())


def _eps_greedy_manip(q_table, room, rng):
    """ε over actions 0..5. Otherwise argmax over all 8, ties uniform."""
    if float(rng.random()) < m4w2.EPS:
        return int(rng.integers(0, world.N_MANIP)), 'eps'
    row = q_table[int(room)]
    best = row.max()
    cands = np.flatnonzero(row == best)
    if int(cands.size) == 1:
        return int(cands[0]), 'greedy'
    return int(rng.choice(cands)), 'greedy'


class _Chooser:
    def __init__(self, arm, world_seed, steps):
        self.arm = arm
        self.actions = None
        self.ctrl = None
        if arm == 'rand6':
            self.actions = np.random.default_rng(world.POLICY_SEED + world_seed).integers(
                0, world.N_MANIP, size=steps)
        elif arm in Q_ARMS:
            self.ctrl = np.random.default_rng(m4w2.CTRL_SEED + world_seed)
        else:
            raise ValueError(arm)

    def act(self, env, t, q_table):
        if self.actions is not None:
            return int(self.actions[t]), 'fixed'
        return _eps_greedy_manip(q_table, env.room, self.ctrl)


def _quarter_shares(actions, steps):
    tv = []
    reset = []
    for start, stop in head._quarter_bounds(steps):
        sl = actions[start:stop]
        width = float(stop - start)
        tv.append(float(np.sum(sl == world.TV) / width))
        reset.append(float(np.sum(sl == world.RESET) / width))
    return tv, reset


def run_closed(world_seed, arm, steps, block_size, device, log_every=200,
               keep_trace=False, freeze_q=False):
    """One closed-loop life. rand6 uses the W2-0 policy and rng."""
    _require_parent_constants()
    if arm not in ARMS:
        raise ValueError(arm)
    if steps % 4 != 0:
        raise AssertionError(f'T={steps} is not divisible by 4')
    if freeze_q and arm not in Q_ARMS:
        raise AssertionError('freeze_q is for a Q arm')
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
    histories = None
    if arm == 'prog_lp' and not freeze_q:
        histories = [[] for _ in range(world.N_ROOMS * world.N_ACTIONS)]
    tokens = np.empty(steps * 2, dtype=np.int64)
    actions = np.empty(steps, dtype=np.int64)
    rooms_after = np.empty(steps, dtype=np.int64)
    branch_counts = None
    if freeze_q:
        branch_counts = {
            'eps': np.zeros(world.N_ACTIONS, dtype=np.int64),
            'greedy': np.zeros(world.N_ACTIONS, dtype=np.int64),
        }
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
        for i in range(n_local):
            t = step0 + i
            rooms_before[i] = int(env.room)
            action, branch = chooser.act(env, t, q_table)
            if branch_counts is not None:
                branch_counts[branch][action] += 1
            room_after, y = env.step(action)
            actions[t] = action
            rooms_after[t] = int(room_after)
            tokens[2 * t] = int(action)
            tokens[2 * t + 1] = world.obs_token(room_after, y)
            room_counts[int(room_after)] += 1
            if int(room_after) > deepest:
                deepest = int(room_after)
                first_step = int(t)
        do_rescore = arm == 'prog_q' and not freeze_q
        state_for_rescore = None
        if do_rescore and state is not None:
            state_for_rescore = state.detach().clone()
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
        bits_after = None
        if do_rescore:
            with torch.no_grad():
                losses_after, _after_state = model.nll(prev_t, tgt_t, state_for_rescore)
                bits_after = head._obs_bits(losses_after)
                del losses_after
            if bits_after.shape != bits_before.shape:
                raise AssertionError('re-score length')
        if q_arm and not freeze_q:
            for i in range(n_local):
                room_b = int(rooms_before[i])
                action = int(actions[step0 + i])
                room_n = int(rooms_after[step0 + i])
                if arm == 'prog_lp':
                    hist = histories[room_b * world.N_ACTIONS + action]
                    n_visit = len(hist) + 1
                    reward = prog_lp_reward(hist, n_visit)
                    hist.append(float(bits_before[i]))
                else:
                    after = float(bits_after[i]) if arm == 'prog_q' else 0.0
                    reward = m4w2._reward(
                        arm, float(bits_before[i]), after, room_b, action, visits)
                if not math.isfinite(reward):
                    raise FloatingPointError(f'{arm} reward at step {step0 + i}')
                m4w2._q_update(q_table, room_b, action, reward, room_n)
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
            row['PROBE_cap'] = m4w2.probe_cap(row['probe_k'])
            probes.append(row)
            print(
                f'M4W2B seed={world_seed} arm={arm} probe nominal={row["nominal_step"]} '
                f'at_step={row["at_step"]} PROBE={row["PROBE"]:.4f} '
                f'cap={row["PROBE_cap"]:.4f} tv={row["tv_probe"]:.4f} '
                f'wall_s={time.monotonic() - t0:.1f}',
                flush=True)
            next_probe += 1
        if chunk_count % log_every == 0 or pos == n_tokens:
            print(
                f'M4W2B seed={world_seed} arm={arm} steps={steps_done}/{steps} '
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
    if freeze_q and float(np.abs(q_table).max()) != 0.0:
        raise AssertionError('frozen Q changed')
    sha = world.token_sha256(tokens)
    if arm == 'rand6':
        ref = world.rollout(world_seed, arm, steps, block_size)
        if ref['token_sha256'] != sha:
            raise AssertionError('rand6 token sha drifted from the fixed policy')
        if ref['time_in_room'] != [int(v) for v in room_counts]:
            raise AssertionError('rand6 time-in-room drifted from the fixed policy')
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
    tv_by_quarter, reset_by_quarter = _quarter_shares(actions, steps)
    for i in range(4):
        if abs(reset_by_quarter[i] * 1000.0 - resets_per_1000[i]) > 1e-6:
            raise AssertionError('reset share does not match resets per 1,000')
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
            'eps': m4w2.EPS,
            'alpha': m4w2.ALPHA,
            'gamma': m4w2.GAMMA,
            'seed': int(m4w2.CTRL_SEED + world_seed),
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
        'tv_share_by_quarter': tv_by_quarter,
        'reset_share_by_quarter': reset_by_quarter,
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
        'freeze_q': bool(freeze_q),
    }
    if keep_trace:
        result['actions'] = [int(v) for v in actions]
    if branch_counts is not None:
        result['branch_counts'] = {
            name: [int(v) for v in counts_row]
            for name, counts_row in branch_counts.items()
        }
    final = probes[-1]
    print(
        f'M4W2B done seed={world_seed} arm={arm} bpo={result["bits_per_observation"]:.4f} '
        f'PROBE={final["PROBE"]:.4f} cap={final["PROBE_cap"]:.4f} '
        f'tv_share={action_share["tv"]:.4f} reset_share={action_share["reset"]:.4f} '
        f'deepest={deepest} room8={int(room_counts[8])} wall_s={wall:.1f}',
        flush=True)
    return result


def _hand_linear_visit_40():
    """mean(b(8)..b(23)) - mean(b(24)..b(39)), b(v) = 4 * (64 - v) / 63."""
    def b(v):
        return 4.0 * (64.0 - float(v)) / 63.0
    older = np.array([b(v) for v in range(8, 24)], dtype=np.float64)
    recent = np.array([b(v) for v in range(24, 40)], dtype=np.float64)
    return float(older.mean() - recent.mean())


def synthetic_prog_lp():
    """Constant 4-bit context after 2K visits, and a linear drop at visit 40."""
    constant_n = 2 * K_LP + 1
    constant = prog_lp_reward([4.0] * (2 * K_LP), constant_n)
    optimistic = prog_lp_reward([4.0] * (2 * K_LP - 1), 2 * K_LP)
    bits = [4.0 * (64.0 - float(v)) / 63.0 for v in range(1, 65)]
    linear = prog_lp_reward(bits[:39], 40)
    hand = _hand_linear_visit_40()
    closed = 64.0 / 63.0
    print(
        f'SMOKE prog_lp constant visits_recorded={2 * K_LP} '
        f'reward_at_visit={constant_n} reward={constant:.16f}',
        flush=True)
    print(
        f'SMOKE prog_lp linear visit=40 reward={linear:.16f} '
        f'hand={hand:.16f} closed_form={closed:.16f}',
        flush=True)
    if optimistic != 1.0:
        raise AssertionError(f'optimistic reward at visit 2K is {optimistic}')
    if constant != 0.0:
        raise AssertionError(f'constant reward after 2K visits is {constant}')
    if not (linear > 0.0):
        raise AssertionError(f'linear reward at visit 40 is {linear}')
    if linear != hand:
        raise AssertionError(f'linear reward {linear} != hand {hand}')
    if abs(hand - closed) > 1e-12:
        raise AssertionError(f'hand {hand} != 64/63')
    return {
        'K': K_LP,
        'constant_bits': 4.0,
        'constant_visits_recorded': 2 * K_LP,
        'constant_reward_at_visit': constant_n,
        'constant_reward': constant,
        'linear_from': 4.0,
        'linear_to': 0.0,
        'linear_visits': 64,
        'linear_visit': 40,
        'linear_reward': linear,
        'linear_hand': hand,
        'linear_closed_form': closed,
        'optimistic_reward_at_visit_2K': optimistic,
    }


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def smoke():
    _require_parent_constants()
    head.configure()
    t0 = time.monotonic()
    device = torch.device('cpu')
    synthetic = synthetic_prog_lp()
    eps_life = run_closed(
        SMOKE_SEED, 'count_q', EPS_CHECK_STEPS, SMOKE_BLOCK, device,
        log_every=10_000, freeze_q=True)
    branches = eps_life['branch_counts']
    eps_counts = branches['eps']
    greedy_counts = branches['greedy']
    tv_from_eps = int(eps_counts[world.TV])
    reset_from_eps = int(eps_counts[world.RESET])
    tv_from_greedy = int(greedy_counts[world.TV])
    reset_from_greedy = int(greedy_counts[world.RESET])
    print(
        'SMOKE eps branch counts '
        + ' '.join(f'a{i}={eps_counts[i]}' for i in range(world.N_ACTIONS)),
        flush=True)
    print(
        'SMOKE greedy branch counts '
        + ' '.join(f'a{i}={greedy_counts[i]}' for i in range(world.N_ACTIONS)),
        flush=True)
    print(
        f'SMOKE eps tv_from_eps={tv_from_eps} reset_from_eps={reset_from_eps} '
        f'tv_from_greedy={tv_from_greedy} reset_from_greedy={reset_from_greedy}',
        flush=True)
    if tv_from_eps != 0 or reset_from_eps != 0:
        raise AssertionError('epsilon branch pressed the TV or reset')
    if sum(eps_counts) + sum(greedy_counts) != EPS_CHECK_STEPS:
        raise AssertionError('branch counts do not sum to the run')
    if tv_from_greedy <= 0 or reset_from_greedy <= 0:
        raise AssertionError('greedy tie-break produced no TV or reset')
    if float(np.abs(np.asarray(eps_life['Q'], dtype=np.float64)).max()) != 0.0:
        raise AssertionError('epsilon-check Q is not zero')
    ref = m4w2.run_closed(
        SMOKE_SEED, 'rand6', SMOKE_STEPS, SMOKE_BLOCK, device, log_every=50)
    closed = run_closed(
        SMOKE_SEED, 'rand6', SMOKE_STEPS, SMOKE_BLOCK, device, log_every=50)
    diff = abs(float(closed['bits']) - float(ref['bits']))
    sha_match = closed['token_sha256'] == ref['token_sha256']
    probe_diff = []
    for c_row, r_row in zip(closed['probes'], ref['probes']):
        probe_diff.append(_max_abs(
            c_row['probe_k'] + [c_row['tv_probe']],
            r_row['probe_k'] + [r_row['tv_probe']]))
    equivalence = {
        'arm': 'rand6',
        'token_sha256': closed['token_sha256'],
        'ref_token_sha256': ref['token_sha256'],
        'sha_match': bool(sha_match),
        'bits': float(closed['bits']),
        'ref_bits': float(ref['bits']),
        'abs_diff': float(diff),
        'probe_max_abs': float(max(probe_diff)),
    }
    print(
        f'SMOKE equivalence arm=rand6 sha_match={sha_match} '
        f'bits={equivalence["bits"]:.6f} ref_bits={equivalence["ref_bits"]:.6f} '
        f'abs_diff={diff:.3e} probe_max_abs={equivalence["probe_max_abs"]:.3e}',
        flush=True)
    print(f'SMOKE sha rand6 {closed["token_sha256"]}', flush=True)
    print(f'SMOKE sha_ref rand6 {ref["token_sha256"]}', flush=True)
    if not sha_match or diff > 1e-6:
        raise AssertionError('rand6 does not match M4-W2')
    arm_rows = []
    traces = {}
    for arm in ARMS:
        if arm == 'rand6':
            row_life = closed
        else:
            row_life = run_closed(
                SMOKE_SEED, arm, SMOKE_STEPS, SMOKE_BLOCK, device,
                log_every=50, keep_trace=True)
        q_finite = None
        if row_life['Q'] is not None:
            q_table = np.asarray(row_life['Q'], dtype=np.float64)
            q_finite = bool(
                np.isfinite(q_table).all()
                and q_table.shape == (world.N_ROOMS, world.N_ACTIONS))
        bits_ok = bool(row_life['finite']) and math.isfinite(float(row_life['bits']))
        share = row_life['action_share']
        manip = share['manipulation']
        total = float(sum(manip) + share['tv'] + share['reset'])
        ok = bits_ok and abs(total - 1.0) <= 1e-9 and min(manip) >= 0
        if arm in Q_ARMS:
            ok = ok and q_finite
            traces[arm] = row_life['actions']
        print(
            f'SMOKE arm={arm} bits={row_life["bits"]:.6f} finite={row_life["finite"]} '
            f'Q_finite={q_finite} tv={share["tv"]:.6f} reset={share["reset"]:.6f}',
            flush=True)
        if not ok:
            raise AssertionError(f'{arm} smoke is not finite')
        arm_rows.append({
            'arm': arm,
            'bits': float(row_life['bits']),
            'bits_per_observation': float(row_life['bits_per_observation']),
            'finite': bool(row_life['finite']),
            'Q_finite': q_finite,
            'action_share': share,
            'PROBE_cap_T': row_life['PROBE_cap_T'],
            'PROBE_T': row_life['PROBE_T'],
            'wall_seconds': row_life['wall_seconds'],
        })
    first = {arm: traces[arm][:CHUNK_STEPS] for arm in Q_ARMS}
    same_first = first['prog_lp'] == first['prog_q'] == first['err_q'] == first['count_q']
    if not same_first:
        raise AssertionError('Q arms diverged inside the first frozen chunk')
    payload = {
        'world_seed': SMOKE_SEED,
        'steps': SMOKE_STEPS,
        'block_size': SMOKE_BLOCK,
        'equivalence': [equivalence],
        'eps_check': {
            'arm': 'count_q',
            'world_seed': SMOKE_SEED,
            'steps': EPS_CHECK_STEPS,
            'freeze_q': True,
            'eps_counts': eps_counts,
            'greedy_counts': greedy_counts,
            'tv_from_eps': tv_from_eps,
            'reset_from_eps': reset_from_eps,
            'tv_from_greedy': tv_from_greedy,
            'reset_from_greedy': reset_from_greedy,
            'Q_max_abs': 0.0,
        },
        'prog_lp_synthetic': synthetic,
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
    _require_parent_constants()
    head.configure()
    t0 = time.monotonic()
    out = Path(out)
    device = torch.device('cpu')
    seeds = tuple(int(s) for s in world_seeds)
    print(
        f'M4W2B kernel={kernel_id} seeds={list(seeds)} device={device} '
        f'cuda={torch.cuda.is_available()} torch={torch.__version__} '
        f'numpy={np.__version__} threads={torch.get_num_threads()}',
        flush=True)
    if device.type != 'cpu':
        raise SystemExit('M4-W2b is a CPU run')
    lives = []
    expect_chunks = math.ceil((steps * 2) / head.CHUNK)
    for world_seed in seeds:
        for arm in ARMS:
            print(f'M4W2B start seed={world_seed} arm={arm}', flush=True)
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
                if abs(row['controller']['alpha'] - m4w2.ALPHA) > 0:
                    raise AssertionError('controller step size')
            lives.append(row)
            dump(out / 'result.json', {
                'kernel_id': kernel_id,
                'world_seeds': list(seeds),
                'lives': [{k: v for k, v in life.items() if k != 'actions'} for life in lives],
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
        'k_lp': K_LP,
        'explore': 'manipulations',
        'probe_cap': m4w2.CAP,
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
    if len(row['tv_share_by_quarter']) != 4 or len(row['reset_share_by_quarter']) != 4:
        raise SystemExit('quarter shares')
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
    cap = m4w2.probe_cap(row['probes'][-1]['probe_k'])
    if abs(cap - float(row['PROBE_cap_T'])) > 1e-9:
        raise SystemExit('PROBE_cap does not match probe_k')
    if row['arm'] in Q_ARMS:
        q_table = np.asarray(row['Q'], dtype=np.float64)
        if q_table.shape != (world.N_ROOMS, world.N_ACTIONS) or not np.isfinite(q_table).all():
            raise SystemExit(f'{row["arm"]} Q is missing or not finite')
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
        base = float(loaded[seed]['prog_lp'][key])
        rows.append({
            'world_seed': int(seed),
            'rand6_minus_prog_lp': float(loaded[seed]['rand6'][key]) - base,
            'count_q_minus_prog_lp': float(loaded[seed]['count_q'][key]) - base,
            'prog_q_minus_prog_lp': float(loaded[seed]['prog_q'][key]) - base,
            'err_q_minus_prog_lp': float(loaded[seed]['err_q'][key]) - base,
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
        for seed in SEEDS:
            row = loaded[seed][arm]
            tv = [float(v) for v in row['tv_share_by_quarter']]
            reset = [float(v) for v in row['reset_share_by_quarter']]
            per_seed.append({'world_seed': int(seed), 'tv': tv, 'reset': reset})
            for i in range(4):
                tv_cols[i].append(tv[i])
                reset_cols[i].append(reset[i])
        tv_mean, tv_std, reset_mean, reset_std = [], [], [], []
        for i in range(4):
            mean, std = _mean_std(tv_cols[i])
            tv_mean.append(mean)
            tv_std.append(std)
            mean, std = _mean_std(reset_cols[i])
            reset_mean.append(mean)
            reset_std.append(std)
        out[arm] = {
            'per_seed': per_seed,
            'tv_mean': tv_mean,
            'tv_std_ddof0': tv_std,
            'reset_mean': reset_mean,
            'reset_std_ddof0': reset_std,
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
        if int(payload.get('k_lp', -1)) != K_LP or payload.get('explore') != 'manipulations':
            raise SystemExit(f'{kernel_id} is not the M4-W2b controller')
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
        'k_lp': K_LP,
        'explore': 'manipulations',
        'probe_cap': m4w2.CAP,
        'arms': list(ARMS),
        'world_seeds': list(SEEDS),
        'lives': lives_out,
        'probe_cap_table': _arm_table(loaded, 'PROBE_cap_T'),
        'probe_raw_table': _arm_table(loaded, 'PROBE_T'),
        'gaps_probe_cap': _gap_table(loaded, 'PROBE_cap_T'),
        'gaps_probe_raw': _gap_table(loaded, 'PROBE_T'),
        'action_share_mean': share_mean,
        'quarter_shares': _quarter_share_table(loaded),
        'depth': depth,
        'kernels': kernels,
        'kernel_elapsed_seconds_sum': float(sum(row['elapsed_seconds'] for row in kernels)),
        'smoke': smoke_payload,
    }
    dump(out_dir / 'results.json', result)
    lines = _markdown(result)
    (out_dir / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(
        'M4W2B assemble '
        + ' '.join(
            f'{name}={result["gaps_probe_cap"]["mean_std_ddof0"][name]["mean"]:.4f}'
            for name in GAP_NAMES),
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


def _gap_lines(title, block):
    lines = [
        '',
        title,
        '',
        '| world_seed | rand6 − prog_lp | count_q − prog_lp | prog_q − prog_lp | err_q − prog_lp |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in block['per_seed']:
        lines.append(
            f'| {row["world_seed"]} | {_fmt(row["rand6_minus_prog_lp"])} | '
            f'{_fmt(row["count_q_minus_prog_lp"])} | {_fmt(row["prog_q_minus_prog_lp"])} | '
            f'{_fmt(row["err_q_minus_prog_lp"])} |')
    gaps = block['mean_std_ddof0']
    lines.append(
        f'| mean | {_fmt(gaps["rand6_minus_prog_lp"]["mean"])} | '
        f'{_fmt(gaps["count_q_minus_prog_lp"]["mean"])} | '
        f'{_fmt(gaps["prog_q_minus_prog_lp"]["mean"])} | '
        f'{_fmt(gaps["err_q_minus_prog_lp"]["mean"])} |')
    lines.append(
        f'| std ddof=0 | {_fmt(gaps["rand6_minus_prog_lp"]["std_ddof0"])} | '
        f'{_fmt(gaps["count_q_minus_prog_lp"]["std_ddof0"])} | '
        f'{_fmt(gaps["prog_q_minus_prog_lp"]["std_ddof0"])} | '
        f'{_fmt(gaps["err_q_minus_prog_lp"]["std_ddof0"])} |')
    return lines


def _markdown(result):
    seed_header = ' | '.join(str(seed) for seed in SEEDS)
    lines = [
        '# M4-W2b visit-to-visit learning progress',
        '',
        'Closed loop on the locked-rooms world. Bits are prequential, on observation tokens only.',
        'PROBE_cap is the mean over rooms 0..8 of min(probe_k, 4). Seeds 9–14.',
        '',
        '## PROBE_cap at T',
        '',
        f'| arm | {seed_header} | mean | std ddof=0 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
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
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for arm in ARMS:
        block = result['probe_raw_table'][arm]
        cells = [_fmt(row['value']) for row in block['per_seed']]
        lines.append(
            f'| {arm} | {" | ".join(cells)} | {_fmt(block["mean"])} | {_fmt(block["std_ddof0"])} |')
    lines += _gap_lines('## Gaps on raw PROBE', result['gaps_probe_raw'])
    lines += [
        '',
        '## Action shares',
        '',
        'Mean over seeds 9–14. TV is action 6. Reset is action 7.',
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
        'Mean over seeds 9–14. Quarter width is T/4.',
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
            cap = probe.get('PROBE_cap')
            if cap is None:
                cap = m4w2.probe_cap(probe['probe_k'])
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
        f'block {smoke_payload["block_size"]}. '
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
    eps = smoke_payload['eps_check']
    lines += [
        '',
        f'ε check: count_q, world seed {eps["world_seed"]}, T = {eps["steps"]}, '
        'Q held at zero.',
        '',
        '| branch | a0 | a1 | a2 | a3 | a4 | a5 | tv | reset |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| eps | ' + ' | '.join(str(v) for v in eps['eps_counts']) + ' |',
        '| greedy | ' + ' | '.join(str(v) for v in eps['greedy_counts']) + ' |',
        '',
        f'TV from eps {eps["tv_from_eps"]}. Reset from eps {eps["reset_from_eps"]}. '
        f'TV from greedy {eps["tv_from_greedy"]}. Reset from greedy {eps["reset_from_greedy"]}.',
        '',
    ]
    syn = smoke_payload['prog_lp_synthetic']
    lines += [
        f'prog_lp synthetic, K = {syn["K"]}. Constant bits {syn["constant_bits"]:.1f}, '
        f'reward at visit {syn["constant_reward_at_visit"]} after '
        f'{syn["constant_visits_recorded"]} visits: {syn["constant_reward"]:.16f}.',
        f'Linear bits from {syn["linear_from"]:.1f} to {syn["linear_to"]:.1f} over visits '
        f'1..{syn["linear_visits"]}. Reward at visit {syn["linear_visit"]}: '
        f'{syn["linear_reward"]:.16f}. Hand: {syn["linear_hand"]:.16f}.',
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
        '- Re-score under no_grad from the same initial hidden state runs only for prog_q. prog_lp stores bits_before and does not re-score.',
        '- prog_lp context is (room before the action, action). K = 16. Visit n ≤ 32 rewards 1. Visit n > 32 rewards mean(bits at visits n−32..n−17) minus mean(bits at visits n−16..n−1). The current visit is not in the window. Its bits_before is appended after the reward, in order inside the chunk.',
        '- prog_q reward is bits_before minus bits_after, unclipped. err_q reward is bits_before. count_q reward is 1/sqrt(n) with n including the current visit.',
        '- Q is 9 by 8, zeros at the start. ε = 0.1 over actions 0..5 only. Greedy argmax is over all 8 actions. Ties are uniform. α = 0.1, γ = 0.9. Controller rng is default_rng(7000 + world_seed).',
        '- rand6 uses the W2-0 policy and rng. Learner seed 0. Replay seed 30000. Adam lr 0.001, clip 1.0, replay batch 8.',
        '- GRU embedding 32, hidden 128, 1 layer, readout 152. 86680 parameters. CPU, 4 threads.',
        '- Checkpoints T/4, T/2 and T. Probe boundaries are 15008, 30016 and 60000.',
        '- PROBE_cap caps each room probe at 4 bits. Seeds 9, 10, 11, 12, 13 and 14.',
        '- Kernels ezzzzzx/m4w2b-a, m4w2b-b and m4w2b-c. Private, CPU, no internet. Code dataset ezzzzzx/pdp-code-m4w2b.',
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
