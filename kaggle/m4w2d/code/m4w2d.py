"""M4-W2d: curiosity in W2, with reset and the TV taken only on purpose.

Imports M4-W2c and replaces the action rule. Everything else is M4-W2c:
state, step size, discount, rewards, learner, chunks and scoring. ε = 0.1
draws from manipulations 0..5. A greedy step takes action 6 or 7 only when
that Q is strictly the highest. Numbers only. No verdict. Does not commit.
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
import m4w2c
import w2_headroom as head
import w2_world as world

ARMS = m4w2c.ARMS
MEM_ARMS = m4w2c.MEM_ARMS
Q_ARMS = m4w2c.Q_ARMS
SEEDS = (21, 22, 23, 24, 25, 26)
KERNELS = (
    ('ezzzzzx/m4w2d-a', (21, 22)),
    ('ezzzzzx/m4w2d-b', (23, 24)),
    ('ezzzzzx/m4w2d-c', (25, 26)),
)
SMOKE_STEPS = m4w2c.SMOKE_STEPS
SMOKE_BLOCK = m4w2c.SMOKE_BLOCK
SMOKE_SEED = m4w2c.SMOKE_SEED
CAPABILITY_STEPS = m4w2c.CAPABILITY_STEPS
OUT_DIR = Path('results/m4w2d')
ACTION_RULE = 'strict_tv_reset'
EXPLORE = 'manipulations'
M4W2C_CAPABILITY = {
    'deepest_room': 1,
    'deepest_first_step': 496,
    'source': 'results/m4w2c/RESULTS.md',
}
RULE_DRAWS = 1_000
RULE_SEED = 7_000


def dump(path, obj):
    m4w2c.dump(path, obj)


def _fmt(value, digits=6):
    return m4w2c._fmt(value, digits)


def choose_action(q_table, index, rng):
    """ε over actions 0..5. TV and reset only when strictly best.

    m = max Q over all 8 actions. If action 6 or 7 equals m and no
    manipulation does, take that action, the lower id if both do.
    Otherwise draw uniformly from the manipulations whose Q equals m.
    """
    if float(rng.random()) < m4w2.EPS:
        return int(rng.integers(0, world.N_MANIP)), 'eps'
    row = q_table[int(index)]
    m = row.max()
    manip = np.flatnonzero(row[:world.N_MANIP] == m)
    special = [a for a in (int(world.TV), int(world.RESET)) if row[a] == m]
    if special and int(manip.size) == 0:
        return int(min(special)), 'greedy'
    if int(manip.size) == 0:
        raise AssertionError('max is not a manipulation or a special')
    if int(manip.size) == 1:
        return int(manip[0]), 'greedy'
    return int(rng.choice(manip)), 'greedy'


def install_rule():
    """Point both learning controllers at choose_action. rand6 never calls them."""
    m4w2c._choose_memory = choose_action
    m4w2b._eps_greedy_manip = choose_action


install_rule()


class _GreedyProbe:
    """One greedy draw. choice() rejects a candidate set that contains 6 or 7."""

    def __init__(self):
        self.choice_calls = 0

    def random(self):
        return 1.0

    def integers(self, low, high):
        raise AssertionError(f'greedy probe drew epsilon integers({low}, {high})')

    def choice(self, cands):
        self.choice_calls += 1
        arr = np.asarray(cands, dtype=np.int64)
        if np.any(arr >= world.N_MANIP) or np.any(arr < 0):
            raise AssertionError(f'greedy candidates {arr.tolist()}')
        return int(arr[0])


class _EpsProbe:
    def __init__(self):
        self.low = None
        self.high = None

    def random(self):
        return 0.0

    def integers(self, low, high):
        self.low = int(low)
        self.high = int(high)
        return 3

    def choice(self, cands):
        raise AssertionError('epsilon probe called choice')


def _row_table(values):
    table = np.zeros((1, world.N_ACTIONS), dtype=np.float64)
    table[0] = np.asarray(values, dtype=np.float64)
    return table


def _draw_row(values, rng, n):
    table = _row_table(values)
    actions = []
    branches = []
    for _ in range(int(n)):
        action, branch = choose_action(table, 0, rng)
        actions.append(int(action))
        branches.append(branch)
    return actions, branches


def _count_actions(actions):
    counts = np.bincount(np.asarray(actions, dtype=np.int64), minlength=world.N_ACTIONS)
    if int(counts.shape[0]) != world.N_ACTIONS:
        raise AssertionError('action outside 0..7')
    return [int(v) for v in counts]


def rule_check(n=RULE_DRAWS, seed=RULE_SEED):
    """Hand-made Q rows, n draws each. Raises if the action rule is wrong."""
    m4w2c._require_constants()
    if world.TV != 6 or world.RESET != 7 or world.N_MANIP != 6:
        raise AssertionError('action ids')
    rows = {
        'zeros': [0.0] * 8,
        'reset_only': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        'reset_tie_manip': [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
        'tv_and_reset': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
    }
    probes = {}
    for name, values in rows.items():
        greedy = _GreedyProbe()
        action, branch = choose_action(_row_table(values), 0, greedy)
        eps = _EpsProbe()
        eps_action, eps_branch = choose_action(_row_table(values), 0, eps)
        if eps_branch != 'eps' or eps_action != 3 or eps.low != 0 or eps.high != world.N_MANIP:
            raise AssertionError(f'{name} epsilon probe {(eps_action, eps_branch, eps.low, eps.high)}')
        probes[name] = {
            'greedy_action': int(action),
            'greedy_branch': branch,
            'greedy_choice_calls': int(greedy.choice_calls),
        }
        print(
            f'SMOKE rule probe row={name} greedy_action={action} branch={branch} '
            f'choice_calls={greedy.choice_calls}',
            flush=True)
    if probes['zeros']['greedy_branch'] != 'greedy' or probes['zeros']['greedy_choice_calls'] != 1:
        raise AssertionError('zeros did not tie-break over manipulations')
    if probes['zeros']['greedy_action'] not in range(world.N_MANIP):
        raise AssertionError('zeros greedy left 0..5')
    if probes['reset_only']['greedy_action'] != world.RESET or probes['reset_only']['greedy_choice_calls'] != 0:
        raise AssertionError('reset_only was not a strict reset')
    if probes['reset_tie_manip']['greedy_action'] != 2 or probes['reset_tie_manip']['greedy_choice_calls'] != 0:
        raise AssertionError('reset tied with action 2 did not take 2')
    if probes['tv_and_reset']['greedy_action'] != world.TV or probes['tv_and_reset']['greedy_choice_calls'] != 0:
        raise AssertionError('TV and reset tied and the lower id was not taken')

    rng = np.random.default_rng(int(seed))
    drawn = {}
    for name, values in rows.items():
        actions, branches = _draw_row(values, rng, n)
        counts = _count_actions(actions)
        greedy_actions = [a for a, b in zip(actions, branches) if b == 'greedy']
        eps_actions = [a for a, b in zip(actions, branches) if b == 'eps']
        if len(greedy_actions) + len(eps_actions) != int(n):
            raise AssertionError(f'{name} branch labels')
        if not greedy_actions or not eps_actions:
            raise AssertionError(f'{name} missing a branch in {n} draws')
        if any(a >= world.N_MANIP for a in eps_actions):
            raise AssertionError(f'{name} epsilon pressed the TV or reset')
        print(
            f'SMOKE rule draws row={name} n={n} '
            + ' '.join(f'a{i}={counts[i]}' for i in range(world.N_ACTIONS))
            + f' greedy={len(greedy_actions)} eps={len(eps_actions)}',
            flush=True)
        print(
            f'SMOKE rule sequence row={name} ' + ' '.join(str(a) for a in actions),
            flush=True)
        drawn[name] = {
            'n': int(n),
            'seed': int(seed),
            'counts': counts,
            'greedy_n': len(greedy_actions),
            'eps_n': len(eps_actions),
            'actions': actions,
            'branches': branches,
        }
        if name == 'zeros':
            if counts[world.TV] or counts[world.RESET]:
                raise AssertionError(f'zeros pressed a special {counts}')
            if min(counts[:world.N_MANIP]) < 80 or max(counts[:world.N_MANIP]) > 260:
                raise AssertionError(f'zeros is not about uniform {counts}')
        elif name == 'reset_only':
            if any(a != world.RESET for a in greedy_actions) or counts[world.TV]:
                raise AssertionError(f'reset_only greedy was not 7 {counts}')
        elif name == 'reset_tie_manip':
            if any(a != 2 for a in greedy_actions) or counts[world.TV] or counts[world.RESET]:
                raise AssertionError(f'reset tie took a special {counts}')
        elif name == 'tv_and_reset':
            if any(a != world.TV for a in greedy_actions) or counts[world.RESET]:
                raise AssertionError(f'TV and reset tie took reset {counts}')
        else:
            raise AssertionError(name)
    return {'probe': probes, 'draws': {k: {kk: vv for kk, vv in row.items() if kk != 'actions' and kk != 'branches'}
                                        for k, row in drawn.items()}}


def _max_abs(a, b):
    return m4w2c._max_abs(a, b)


def smoke():
    m4w2c._require_constants()
    install_rule()
    if m4w2c._choose_memory is not choose_action or m4w2b._eps_greedy_manip is not choose_action:
        raise AssertionError('action rule was not installed')
    head.configure()
    t0 = time.monotonic()
    device = torch.device('cpu')
    ref = m4w2c.run_closed(
        SMOKE_SEED, 'rand6', SMOKE_STEPS, SMOKE_BLOCK, device, log_every=50)
    closed = m4w2c.run_closed(
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
        'parent_arm': 'rand6',
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
        raise AssertionError('rand6 does not match M4-W2c')
    rules = rule_check()
    capability = m4w2c.capability_check(SMOKE_SEED, CAPABILITY_STEPS)
    print(
        f'SMOKE capability compare m4w2d_deepest={capability["deepest_room"]} '
        f'm4w2d_first={capability["deepest_first_step"]} '
        f'm4w2c_deepest={M4W2C_CAPABILITY["deepest_room"]} '
        f'm4w2c_first={M4W2C_CAPABILITY["deepest_first_step"]}',
        flush=True)
    lives = {'rand6': closed}
    arm_rows = []
    for arm in ARMS:
        if arm == 'rand6':
            row_life = closed
        else:
            row_life = m4w2c.run_closed(
                SMOKE_SEED, arm, SMOKE_STEPS, SMOKE_BLOCK, device,
                log_every=50, keep_trace=True)
        q_finite = None
        if row_life['Q'] is not None:
            q_table = np.asarray(row_life['Q'], dtype=np.float64)
            expect = (
                (m4w2c.N_STATES, world.N_ACTIONS) if arm in MEM_ARMS
                else (world.N_ROOMS, world.N_ACTIONS))
            q_finite = bool(np.isfinite(q_table).all() and tuple(q_table.shape) == expect)
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
        'action_rule': ACTION_RULE,
        'equivalence': [equivalence],
        'rule_check': rules,
        'capability': capability,
        'capability_m4w2c': M4W2C_CAPABILITY,
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
    install_rule()
    if m4w2c._choose_memory is not choose_action or m4w2b._eps_greedy_manip is not choose_action:
        raise AssertionError('action rule was not installed')
    result = m4w2c.kernel_main(out, world_seeds, kernel_id, steps=steps, block_size=block_size)
    result['action_rule'] = ACTION_RULE
    if result.get('explore') != EXPLORE:
        raise AssertionError('parent explore field changed')
    out = Path(out)
    dump(out / 'result.json', result)
    summary_path = out / 'kernel.json'
    summary = json.loads(summary_path.read_text())
    summary['action_rule'] = ACTION_RULE
    dump(summary_path, summary)
    print(f'M4W2D action_rule={ACTION_RULE} kernel={kernel_id}', flush=True)
    return result


def _mean_std(values):
    return m4w2c._mean_std(values)


def _with_seeds(fn, *args):
    old = m4w2c.SEEDS
    m4w2c.SEEDS = SEEDS
    try:
        return fn(*args)
    finally:
        m4w2c.SEEDS = old


def _bar_block(loaded, capability_room):
    gaps = _with_seeds(m4w2c._gap_table, loaded, 'PROBE_cap_T')
    summary = gaps['mean_std_ddof0']
    beats = sum(1 for row in gaps['per_seed'] if row['rand6_minus_lp_mem'] > 0.0)
    tv = [float(loaded[seed]['lp_mem']['action_share']['tv']) for seed in SEEDS]
    tv_mean, tv_std = _mean_std(tv)
    deepest = {}
    for arm in ARMS:
        deepest[arm] = [int(loaded[seed][arm]['deepest_room']) for seed in SEEDS]
    kill = int(capability_room) <= 2
    return {
        'kill_rule': 'capability_deepest_room_le_2',
        'kill_rule_applies': bool(kill),
        'bar_1': 'void' if kill else None,
        'bar_2': 'void' if kill else None,
        'rand6_minus_lp_mem_mean': summary['rand6_minus_lp_mem']['mean'],
        'rand6_minus_lp_mem_std_ddof0': summary['rand6_minus_lp_mem']['std_ddof0'],
        'n_seeds_lp_mem_beats_rand6': int(beats),
        'count_mem_minus_lp_mem_mean': summary['count_mem_minus_lp_mem']['mean'],
        'count_mem_minus_lp_mem_std_ddof0': summary['count_mem_minus_lp_mem']['std_ddof0'],
        'lp_room_minus_lp_mem_mean': summary['lp_room_minus_lp_mem']['mean'],
        'lp_room_minus_lp_mem_std_ddof0': summary['lp_room_minus_lp_mem']['std_ddof0'],
        'lp_mem_tv_share_mean': tv_mean,
        'lp_mem_tv_share_std_ddof0': tv_std,
        'deepest_room_per_arm': deepest,
    }


def assemble(out_dir=OUT_DIR, kernel_meta=None):
    out_dir = Path(out_dir)
    if kernel_meta is None:
        meta_path = out_dir / 'kernel_meta.json'
        kernel_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    smoke_payload = json.loads((out_dir / 'smoke.json').read_text())
    if smoke_payload.get('action_rule') != ACTION_RULE:
        raise SystemExit('smoke is not the strict TV/reset rule')
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
        if int(payload.get('k_lp', -1)) != m4w2b.K_LP or payload.get('explore') != EXPLORE:
            raise SystemExit(f'{kernel_id} is not the M4-W2d controller')
        if payload.get('action_rule') != ACTION_RULE:
            raise SystemExit(f'{kernel_id} action rule {payload.get("action_rule")}')
        if int(payload.get('n_states', -1)) != m4w2c.N_STATES or int(payload.get('memory_none', -1)) != m4w2c.NONE:
            raise SystemExit(f'{kernel_id} memory layout')
        by_key = {}
        for row in payload['lives']:
            m4w2c._require_life(row, row['world_seed'])
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
        log_s = m4w2c._log_elapsed(logs[0]) if logs else None
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
            'status': m4w2c._status_name(info.get('status', '')),
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

    capability = smoke_payload['capability']
    comparison = {
        'm4w2d_deepest_room': int(capability['deepest_room']),
        'm4w2d_deepest_first_step': int(capability['deepest_first_step']),
        'm4w2c_deepest_room': int(M4W2C_CAPABILITY['deepest_room']),
        'm4w2c_deepest_first_step': int(M4W2C_CAPABILITY['deepest_first_step']),
        'm4w2c_source': M4W2C_CAPABILITY['source'],
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
        'explore': EXPLORE,
        'action_rule': ACTION_RULE,
        'probe_cap': m4w2.CAP,
        'n_states': m4w2c.N_STATES,
        'memory_none': m4w2c.NONE,
        'arms': list(ARMS),
        'world_seeds': list(SEEDS),
        'lives': lives_out,
        'probe_cap_table': _with_seeds(m4w2c._arm_table, loaded, 'PROBE_cap_T'),
        'probe_raw_table': _with_seeds(m4w2c._arm_table, loaded, 'PROBE_T'),
        'gaps_probe_cap': _with_seeds(m4w2c._gap_table, loaded, 'PROBE_cap_T'),
        'gaps_probe_raw': _with_seeds(m4w2c._gap_table, loaded, 'PROBE_T'),
        'action_share_mean': share_mean,
        'quarter_shares': _with_seeds(m4w2c._quarter_share_table, loaded),
        'time_in_room_by_quarter': _with_seeds(m4w2c._room_quarter_table, loaded),
        'depth': depth,
        'capability': capability,
        'capability_comparison': comparison,
        'bars': _bar_block(loaded, capability['deepest_room']),
        'kernels': kernels,
        'kernel_elapsed_seconds_sum': float(sum(row['elapsed_seconds'] for row in kernels)),
        'smoke': smoke_payload,
    }
    dump(out_dir / 'results.json', result)
    lines = _markdown(result)
    (out_dir / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    bars = result['bars']
    print(
        'M4W2D assemble '
        f'bar1={bars["bar_1"]} bar2={bars["bar_2"]} '
        f'rand6_minus_lp_mem={bars["rand6_minus_lp_mem_mean"]:.4f} '
        f'count_minus_lp={bars["count_mem_minus_lp_mem_mean"]:.4f} '
        f'lp_room_minus_lp_mem={bars["lp_room_minus_lp_mem_mean"]:.4f} '
        f'lp_mem_tv={bars["lp_mem_tv_share_mean"]:.4f} '
        f'capability={comparison["m4w2d_deepest_room"]}',
        flush=True)
    return result


def _markdown(result):
    seed_header = ' | '.join(str(seed) for seed in SEEDS)
    col_sep = '|---|' + '---:|' * (len(SEEDS) + 2)
    lines = [
        '# M4-W2d curiosity with reset taken only on purpose',
        '',
        'Closed loop on the locked-rooms world. Bits are prequential, on observation tokens only.',
        'PROBE_cap is the mean over rooms 0..8 of min(probe_k, 4). Seeds 21–26.',
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
    lines += m4w2c._gap_lines('## Gaps on PROBE_cap', result['gaps_probe_cap'])
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
    lines += m4w2c._gap_lines('## Gaps on raw PROBE', result['gaps_probe_raw'])
    tv = result['action_share_mean']['lp_mem']
    lines += [
        '',
        '## Action shares',
        '',
        'Mean over seeds 21–26. TV is action 6. Reset is action 7.',
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
        'Mean over seeds 21–26. Quarter width is T/4.',
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
        'Mean over seeds 21–26.',
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
        'Mean over seeds 21–26. Rooms 0..8. Quarter width is T/4.',
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
    comp = result['capability_comparison']
    bars = result['bars']
    lines += [
        '## Capability check',
        '',
        f'World seed {cap["world_seed"]}, T = {cap["steps"]}, no learner. '
        'Reward 1 on the first step that ends in a room not entered before, else 0. '
        'Room 0 counts as entered at the start. Q updates every step. '
        'The action rule is the strict TV and reset rule.',
        '',
        f'M4-W2d deepest room {comp["m4w2d_deepest_room"]}. '
        f'First step {comp["m4w2d_deepest_first_step"]}.',
        '',
        f'M4-W2c deepest room {comp["m4w2c_deepest_room"]}. '
        f'First step {comp["m4w2c_deepest_first_step"]}.',
        '',
        'First entry step by room. A hyphen means the room was not entered by a step. '
        'Room 0 is a hyphen because it is entered at the start: '
        + ' '.join('-' if v is None else str(v) for v in cap['first_entry_step']) + '.',
        '',
        '## Bars',
        '',
        f'rand6 − lp_mem mean {_fmt(bars["rand6_minus_lp_mem_mean"])}. '
        f'std ddof=0 {_fmt(bars["rand6_minus_lp_mem_std_ddof0"])}. '
        f'Seeds where lp_mem beats rand6: {bars["n_seeds_lp_mem_beats_rand6"]} of {len(SEEDS)}.',
        '',
        f'count_mem − lp_mem mean {_fmt(bars["count_mem_minus_lp_mem_mean"])}. '
        f'std ddof=0 {_fmt(bars["count_mem_minus_lp_mem_std_ddof0"])}.',
        '',
        f'lp_room − lp_mem mean {_fmt(bars["lp_room_minus_lp_mem_mean"])}. '
        f'std ddof=0 {_fmt(bars["lp_room_minus_lp_mem_std_ddof0"])}.',
        '',
        f'lp_mem TV share mean {_fmt(bars["lp_mem_tv_share_mean"])}. '
        f'std ddof=0 {_fmt(bars["lp_mem_tv_share_std_ddof0"])}.',
        '',
        'Deepest room by seed: '
        + ' '.join(
            arm + ' ' + ' '.join(str(v) for v in bars['deepest_room_per_arm'][arm])
            for arm in ARMS) + '.',
        '',
    ]
    if bars['kill_rule_applies']:
        lines += [
            f'Capability deepest room {comp["m4w2d_deepest_room"]}. Bars 1 and 2: void.',
            '',
        ]
    lines += ['## Lives', '']
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
                lines += m4w2c._memory_matrix_lines(row['Q'], drop_zeros=True)
                lines.append('')
                lines.append('Mean reward per (room, p1, p2, action) over the last quarter.')
                lines.append('')
                lines.append('Rows with no visit in the last quarter are omitted.')
                lines.append('')
                lines += m4w2c._memory_matrix_lines(row['mean_reward_last_quarter'], drop_zeros=True)
            else:
                lines += m4w2c._matrix_lines(row['Q'])
                lines.append('')
                lines.append('Mean reward per (room, action) over the last quarter.')
                lines.append('')
                lines += m4w2c._matrix_lines(row['mean_reward_last_quarter'])
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
    draws = smoke_payload['rule_check']['draws']
    lines += [
        '',
        f'Rule check, {RULE_DRAWS} draws, rng default_rng({RULE_SEED}). Counts are actions 0..7.',
        '',
    ]
    for name in ('zeros', 'reset_only', 'reset_tie_manip', 'tv_and_reset'):
        counts = draws[name]['counts']
        lines.append(f'{name}: ' + ' '.join(str(v) for v in counts) + '.')
    cap_row = smoke_payload['capability']
    lines += [
        '',
        f'Capability check, world seed {cap_row["world_seed"]}, T = {cap_row["steps"]}, '
        'no learner. M4-W2d deepest room '
        f'{cap_row["deepest_room"]}. First step {cap_row["deepest_first_step"]}. '
        f'M4-W2c deepest room {M4W2C_CAPABILITY["deepest_room"]}. '
        f'First step {M4W2C_CAPABILITY["deepest_first_step"]}.',
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
        '- lp_room is M4-W2b prog_lp with room-only state, under this action rule. Q is 9 by 8. rand6 is M4-W2b rand6.',
        '- ε = 0.1 over actions 0..5 only. Otherwise m = max Q over all 8 actions. If action 6 or 7 has Q = m and no manipulation in 0..5 has Q = m, take that action, the lower id if both do. Otherwise uniform among manipulations in 0..5 with Q = m. Reset and the TV are never chosen at random. α = 0.1, γ = 0.9. Controller rng is default_rng(7000 + world_seed).',
        '- rand6 uses the W2-0 policy and rng. Learner seed 0. Replay seed 30000. Adam lr 0.001, clip 1.0, replay batch 8.',
        '- GRU embedding 32, hidden 128, 1 layer, readout 152. 86680 parameters. CPU, 4 threads.',
        '- Checkpoints T/4, T/2 and T. Probe boundaries are 15008, 30016 and 60000.',
        '- PROBE_cap caps each room probe at 4 bits. Seeds 21, 22, 23, 24, 25 and 26.',
        '- Kernels ezzzzzx/m4w2d-a, m4w2d-b and m4w2d-c. Private, CPU, no internet. Code dataset ezzzzzx/pdp-code-m4w2d.',
        '',
    ]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--rule-check', action='store_true')
    args = parser.parse_args()
    if args.rule_check:
        rule_check()
    elif args.smoke:
        smoke()
    elif args.assemble:
        assemble()
    else:
        parser.print_help()
        raise SystemExit(2)


if __name__ == '__main__':
    main()
