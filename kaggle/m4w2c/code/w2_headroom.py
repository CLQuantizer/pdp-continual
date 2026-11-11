"""W2 headroom: one GRU cortex on the locked-rooms world. Numbers only.

Fixed policies, prequential bits on observation tokens, replay batch 8.
No verdict. Does not commit.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import w2_world as world

VOCAB = world.VOCAB
EMB = 32
HIDDEN = 128
CHUNK = 64
REPLAY_BATCH = 8
CLIP = 1.0
LR = 1e-3
SEED = 0
REPLAY_SEED = 30_000
T_FULL = 60_000
BLOCK_FULL = 512
T_SMOKE = 600
BLOCK_SMOKE = 32
SMOKE_SEED = 0
EXCERPT_STEPS = 100
LOG2 = math.log(2)
OFFSETS = np.arange(CHUNK, dtype=np.int64)[:, None]
PARAMS = (
    VOCAB * EMB
    + 3 * (HIDDEN * EMB + HIDDEN * HIDDEN + 2 * HIDDEN)
    + HIDDEN * VOCAB
    + VOCAB
)
POLICIES = world.POLICIES
KERNEL_IDS = {
    0: 'ezzzzzx/w2-headroom-s0',
    1: 'ezzzzzx/w2-headroom-s1',
    2: 'ezzzzzx/w2-headroom-s2',
}
OUT_DIR = Path('results/w2_headroom')


def configure():
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def _fmt(value, digits=6):
    return f'{value:.{digits}f}'


class TokenGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, EMB)
        self.gru = nn.GRU(EMB, HIDDEN)
        self.readout = nn.Linear(HIDDEN, VOCAB)

    def nll(self, prev, targets, state):
        # -1 is an absent predecessor. It is not a 153rd symbol.
        emb = self.embedding(prev.clamp_min(0))
        emb = emb.masked_fill((prev < 0).unsqueeze(-1), 0)
        y, state = self.gru(emb, state)
        logits = self.readout(y)
        losses = F.cross_entropy(
            logits.reshape(-1, VOCAB), targets.reshape(-1), reduction='none')
        return losses.reshape(targets.shape), state


def expected_params():
    return PARAMS


def live_batch(tokens, start, end):
    if start % 2 != 0 or (end - start) % 2 != 0:
        raise AssertionError(f'chunk {start}:{end} is not action-aligned')
    tgt = np.ascontiguousarray(tokens[start:end], dtype=np.int64)
    prev = np.empty(end - start, dtype=np.int64)
    if start == 0:
        prev[0] = -1
        if end > 1:
            prev[1:] = tokens[:end - 1]
    else:
        prev[:] = tokens[start - 1:end - 1]
    return prev, tgt


def sample_replay(tokens, n_seen, rng):
    """Eight windows. Starts are uniform on even positions that fit a full chunk."""
    if n_seen < CHUNK:
        raise AssertionError(f'replay needs {CHUNK} tokens, saw {n_seen}')
    n_starts = (n_seen - CHUNK) // 2 + 1
    pick = rng.integers(0, n_starts, size=REPLAY_BATCH, dtype=np.int64)
    starts = pick * 2
    idx = starts[None, :] + OFFSETS
    if int(idx.max()) >= n_seen:
        raise AssertionError('replay read past the live prefix')
    targets = np.ascontiguousarray(tokens[idx], dtype=np.int64)
    prev = np.empty(idx.shape, dtype=np.int64)
    prev[1:] = targets[:-1]
    at_zero = starts == 0
    prev[0, at_zero] = -1
    if not np.all(at_zero):
        prev[0, ~at_zero] = tokens[starts[~at_zero] - 1]
    return prev, targets


def _as_batch(prev, tgt, device):
    prev_t = torch.tensor(prev, dtype=torch.long, device=device)
    tgt_t = torch.tensor(tgt, dtype=torch.long, device=device)
    if prev_t.ndim == 1:
        prev_t = prev_t.unsqueeze(1)
        tgt_t = tgt_t.unsqueeze(1)
    return prev_t, tgt_t


def _step(opt, model, losses):
    opt.zero_grad(set_to_none=True)
    losses.mean().backward()
    nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()


def _obs_bits(losses):
    # Action tokens sit on even local indexes because every chunk starts on an action.
    obs = losses[1::2]
    bits = (obs.detach().double() / LOG2).cpu().numpy().reshape(-1)
    if not np.isfinite(bits).all():
        raise FloatingPointError('nonfinite observation bits')
    return bits


@torch.no_grad()
def score_episode(model, episode, device):
    tokens = episode['tokens']
    n = int(tokens.shape[0])
    prev = np.empty(n, dtype=np.int64)
    prev[0] = -1
    if n > 1:
        prev[1:] = tokens[:-1]
    prev_t, tgt_t = _as_batch(prev, np.ascontiguousarray(tokens, dtype=np.int64), device)
    losses, _state = model.nll(prev_t, tgt_t, None)
    all_bits = (losses.double().reshape(-1) / LOG2).cpu().numpy()
    obs_pos = episode['scored_steps'] * 2 + 1
    bits = all_bits[obs_pos]
    if bits.shape != (world.PROBE_MANIPS,) or not np.isfinite(bits).all():
        raise FloatingPointError('probe bits are not 96 finite values')
    return bits


def probe_model(model, episodes, device):
    was_training = model.training
    model.eval()
    probe_k = []
    for episode in episodes['rooms']:
        bits = score_episode(model, episode, device)
        probe_k.append(float(bits.mean()))
    tv_bits = score_episode(model, episodes['tv'], device)
    if was_training:
        model.train()
    return {
        'probe_k': probe_k,
        'PROBE': float(np.mean(probe_k)),
        'tv_probe': float(tv_bits.mean()),
        'finite': True,
    }


def _quarter_bounds(steps):
    if steps % 4 != 0:
        raise AssertionError(f'T={steps} is not divisible by 4')
    width = steps // 4
    bounds = []
    for i in range(4):
        bounds.append((i * width, (i + 1) * width))
    return bounds


def run_life(world_seed, policy, steps, block_size, device, log_every=200):
    t0 = time.monotonic()
    life = world.rollout(world_seed, policy, steps, block_size)
    tokens = life['tokens']
    if len(tokens) != steps * 2:
        raise AssertionError('token stream length')
    episodes = world.build_probe_episodes(world_seed)
    torch.manual_seed(SEED)
    model = TokenGRU().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != expected_params():
        raise AssertionError(f'parameter count {n_params} != {expected_params()}')
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    replay_rng = np.random.default_rng(REPLAY_SEED + SEED)
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
    pos = 0
    n_tokens = len(tokens)
    while pos < n_tokens:
        end = min(pos + CHUNK, n_tokens)
        prev, tgt = live_batch(tokens, pos, end)
        prev_t, tgt_t = _as_batch(prev, tgt, device)
        losses, new_state = model.nll(prev_t, tgt_t, state)
        bits = _obs_bits(losses)
        if bits.shape[0] != (end - pos) // 2:
            raise AssertionError('scored a token that is not an observation')
        step0 = pos // 2
        step_ids = step0 + np.arange(bits.shape[0], dtype=np.int64)
        quarters = np.minimum(3, step_ids * 4 // steps)
        for q in range(4):
            mask = quarters == q
            if np.any(mask):
                quarter_bits[q] += float(bits[mask].sum())
                quarter_count[q] += int(mask.sum())
        total_bits += float(bits.sum())
        n_obs += int(bits.shape[0])
        state = new_state.detach()
        _step(opt, model, losses[1::2])
        del losses
        if end >= CHUNK:
            r_prev, r_tgt = sample_replay(tokens, end, replay_rng)
            r_prev_t, r_tgt_t = _as_batch(r_prev, r_tgt, device)
            r_losses, _r_state = model.nll(r_prev_t, r_tgt_t, None)
            _step(opt, model, r_losses[1::2])
            del r_losses
            replay_steps += 1
        chunk_count += 1
        pos = end
        steps_done = pos // 2
        while next_probe < len(nominal) and steps_done >= nominal[next_probe]:
            row = probe_model(model, episodes, device)
            row['nominal_step'] = int(nominal[next_probe])
            row['at_step'] = int(steps_done)
            probes.append(row)
            print(
                f'W2 seed={world_seed} policy={policy} probe nominal={row["nominal_step"]} '
                f'at_step={row["at_step"]} PROBE={row["PROBE"]:.4f} '
                f'tv={row["tv_probe"]:.4f} wall_s={time.monotonic() - t0:.1f}',
                flush=True)
            next_probe += 1
        if chunk_count % log_every == 0 or pos == n_tokens:
            print(
                f'W2 seed={world_seed} policy={policy} steps={steps_done}/{steps} '
                f'bpo={total_bits / max(n_obs, 1):.4f} replay={replay_steps} '
                f'wall_s={time.monotonic() - t0:.1f}',
                flush=True)
    if n_obs != steps or next_probe != len(nominal):
        raise AssertionError(f'scored {n_obs} observations, probes {next_probe}')
    if abs(float(quarter_bits.sum()) - total_bits) > 1e-4:
        raise AssertionError('quarter bits do not sum to the life')
    if int(quarter_count.sum()) != steps:
        raise AssertionError('quarter counts do not sum to T')
    bounds = _quarter_bounds(steps)
    quarters = []
    for i, (start, end) in enumerate(bounds):
        count = int(quarter_count[i])
        bits_q = float(quarter_bits[i])
        quarters.append({
            'index': i,
            'start_step': int(start),
            'end_step': int(end),
            'observations': count,
            'bits': bits_q,
            'bits_per_observation': bits_q / count,
        })
    finite = all(row['finite'] for row in probes) and np.isfinite(total_bits)
    excerpt = None
    if int(world_seed) == 0 and policy == 'oracle':
        excerpt = []
        n_ex = min(EXCERPT_STEPS, steps)
        for i in range(n_ex):
            excerpt.append({
                'action': int(life['actions'][i]),
                'room': int(life['rooms'][i]),
                'y': int(life['ys'][i]),
            })
    wall = time.monotonic() - t0
    return {
        'world_seed': int(world_seed),
        'policy': policy,
        'steps': int(steps),
        'block_size': int(block_size),
        'token_sha256': life['token_sha256'],
        'time_in_room': life['time_in_room'],
        'deepest_room': life['deepest_room'],
        'deepest_first_step': life['deepest_first_step'],
        'bits': float(total_bits),
        'bits_per_observation': float(total_bits / steps),
        'quarters': quarters,
        'probes': probes,
        'wall_seconds': wall,
        'replay_steps': int(replay_steps),
        'chunk_count': int(chunk_count),
        'finite': bool(finite),
        'learner_seed': SEED,
        'replay_seed': REPLAY_SEED + SEED,
        'lr': LR,
        'clip': CLIP,
        'replay_batch': REPLAY_BATCH,
        'n_params': int(n_params),
        'excerpt': excerpt,
    }


def _check_replay_sampler():
    tokens = np.arange(128, dtype=np.int64)
    rng = np.random.default_rng(0)
    prev, tgt = sample_replay(tokens, 64, rng)
    if tgt.shape != (CHUNK, REPLAY_BATCH) or not np.all(tgt[0] == 0):
        raise AssertionError('the only window in a 64-token prefix is the prefix')
    if not np.all(prev[0] == -1):
        raise AssertionError('a window at token 0 must have an empty predecessor')
    prev, tgt = sample_replay(tokens, 128, np.random.default_rng(1))
    starts = tgt[0]
    if np.any(starts % 2 != 0) or np.any(starts > 64):
        raise AssertionError(f'replay start left the even feasible set: {starts}')
    for b in range(REPLAY_BATCH):
        if int(starts[b]) == 0:
            if int(prev[0, b]) != -1:
                raise AssertionError('zero start predecessor')
        elif int(prev[0, b]) != int(starts[b]) - 1:
            raise AssertionError('replay predecessor is not the previous token')


def smoke():
    configure()
    t0 = time.monotonic()
    _check_replay_sampler()
    world_checks = world.smoke_world(T_SMOKE, BLOCK_SMOKE, SMOKE_SEED)
    device = torch.device('cpu')
    lives = []
    for policy in POLICIES:
        row = run_life(SMOKE_SEED, policy, T_SMOKE, BLOCK_SMOKE, device, log_every=50)
        if not row['finite']:
            raise AssertionError(f'{policy} smoke bits are not finite')
        if len(row['probes']) != 3:
            raise AssertionError(f'{policy} smoke probe count {len(row["probes"])}')
        for probe in row['probes']:
            if len(probe['probe_k']) != world.N_ROOMS:
                raise AssertionError('smoke probe_k length')
            if not np.isfinite(probe['PROBE']) or not np.isfinite(probe['tv_probe']):
                raise AssertionError(f'{policy} smoke probe is not finite')
            if not all(np.isfinite(v) for v in probe['probe_k']):
                raise AssertionError(f'{policy} smoke probe_k is not finite')
        if row['replay_steps'] != row['chunk_count']:
            raise AssertionError(
                f"{policy} replay_steps {row['replay_steps']} != chunks {row['chunk_count']}")
        lives.append({
            'policy': policy,
            'bits_per_observation': row['bits_per_observation'],
            'deepest_room': row['deepest_room'],
            'deepest_first_step': row['deepest_first_step'],
            'time_in_room': row['time_in_room'],
            'probes': row['probes'],
            'wall_seconds': row['wall_seconds'],
            'replay_steps': row['replay_steps'],
            'finite': row['finite'],
            'token_sha256': row['token_sha256'],
        })
        print(
            f"smoke learner policy={policy} bpo={row['bits_per_observation']:.4f} "
            f"PROBE_T={row['probes'][-1]['PROBE']:.4f} wall_s={row['wall_seconds']:.1f}",
            flush=True)
    payload = {
        'world': world_checks,
        'lives': lives,
        'n_params': expected_params(),
        'device': 'cpu',
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'wall_seconds': time.monotonic() - t0,
        'finite': True,
    }
    dump(OUT_DIR / 'smoke.json', payload)
    print(f'SMOKE_OK wall_s={payload["wall_seconds"]:.1f} params={payload["n_params"]}', flush=True)
    return payload


def kernel_main(out, world_seed, kernel_id, steps=T_FULL, block_size=BLOCK_FULL):
    configure()
    t0 = time.monotonic()
    out = Path(out)
    device = torch.device('cpu')
    print(
        f'W2 kernel={kernel_id} seed={world_seed} device={device} '
        f'cuda={torch.cuda.is_available()} torch={torch.__version__} '
        f'numpy={np.__version__} threads={torch.get_num_threads()}',
        flush=True)
    if device.type != 'cpu':
        raise SystemExit('W2 is a CPU run')
    lives = []
    excerpt = None
    for policy in POLICIES:
        print(f'W2 start seed={world_seed} policy={policy}', flush=True)
        row = run_life(int(world_seed), policy, int(steps), int(block_size), device)
        if not row['finite']:
            raise FloatingPointError(f'{policy} is not finite')
        expect_chunks = math.ceil((steps * 2) / CHUNK)
        if row['chunk_count'] != expect_chunks or row['replay_steps'] != expect_chunks:
            raise AssertionError(
                f"{policy} chunks {row['chunk_count']} replay {row['replay_steps']} "
                f"!= {expect_chunks}")
        if row['probes'][-1]['at_step'] != int(steps):
            raise AssertionError('final probe is not at T')
        if policy == 'oracle' and row['excerpt'] is not None:
            excerpt = row['excerpt']
        stored = dict(row)
        stored['excerpt'] = None
        lives.append(stored)
        partial = {
            'kernel_id': kernel_id,
            'world_seed': int(world_seed),
            'lives': lives,
        }
        dump(out / 'result.json', partial)
    result = {
        'kernel_id': kernel_id,
        'world_seed': int(world_seed),
        'steps': int(steps),
        'block_size': int(block_size),
        'device': 'cpu',
        'cuda_available': bool(torch.cuda.is_available()),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'n_params': expected_params(),
        'learner_seed': SEED,
        'replay_seed': REPLAY_SEED + SEED,
        'lr': LR,
        'clip': CLIP,
        'replay_batch': REPLAY_BATCH,
        'lives': lives,
        'excerpt': excerpt,
        'session_wall_seconds': time.monotonic() - t0,
    }
    dump(out / 'result.json', result)
    summary = {
        'kernel_id': kernel_id,
        'world_seed': int(world_seed),
        'session_wall_seconds': result['session_wall_seconds'],
        'bits_per_observation': {row['policy']: row['bits_per_observation'] for row in lives},
        'PROBE_T': {row['policy']: row['probes'][-1]['PROBE'] for row in lives},
        'finite': all(row['finite'] for row in lives),
    }
    dump(out / 'kernel.json', summary)
    print('KERNEL_DONE ' + json.dumps(summary), flush=True)
    return result


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _require_life(row, world_seed):
    if int(row['world_seed']) != int(world_seed):
        raise SystemExit(f"life seed {row['world_seed']} != {world_seed}")
    if int(row['steps']) != T_FULL or int(row['block_size']) != BLOCK_FULL:
        raise SystemExit(f"life length {row['steps']} block {row['block_size']}")
    if abs(float(row['lr']) - LR) > 1e-15 or int(row['replay_batch']) != REPLAY_BATCH:
        raise SystemExit('life was not the registered learner')
    if int(row['learner_seed']) != SEED or not row['finite']:
        raise SystemExit('life is not a finite seed-0 run')
    if row['probes'][-1]['at_step'] != T_FULL or row['probes'][-1]['nominal_step'] != T_FULL:
        raise SystemExit('final probe is not at T')
    if len(row['probes']) != 3 or len(row['quarters']) != 4:
        raise SystemExit('life is missing quarters or probes')
    if sum(row['time_in_room']) != T_FULL:
        raise SystemExit('time-in-room does not sum to T')


def assemble(out_dir=OUT_DIR, kernel_meta=None):
    out_dir = Path(out_dir)
    loaded = {}
    raw_kernels = []
    for world_seed, kernel_id in KERNEL_IDS.items():
        path = out_dir / 'raw' / f's{world_seed}' / 'result.json'
        payload = json.loads(path.read_text())
        if payload.get('kernel_id') != kernel_id:
            raise SystemExit(f'{path} kernel id {payload.get("kernel_id")}')
        if int(payload['world_seed']) != world_seed:
            raise SystemExit(f'{path} seed {payload["world_seed"]}')
        if payload.get('device') != 'cpu' or payload.get('cuda_available') is True:
            raise SystemExit(f'{kernel_id} did not stay on CPU')
        by_policy = {}
        for row in payload['lives']:
            _require_life(row, world_seed)
            local = world.rollout(world_seed, row['policy'], T_FULL, BLOCK_FULL)
            if local['token_sha256'] != row['token_sha256']:
                raise SystemExit(
                    f"seed {world_seed} {row['policy']} token sha local "
                    f"{local['token_sha256']} != kernel {row['token_sha256']}")
            if local['time_in_room'] != row['time_in_room']:
                raise SystemExit(f"seed {world_seed} {row['policy']} time-in-room mismatch")
            if local['deepest_first_step'] != row['deepest_first_step']:
                raise SystemExit(f"seed {world_seed} {row['policy']} deepest step mismatch")
            by_policy[row['policy']] = row
        if tuple(by_policy) != POLICIES and set(by_policy) != set(POLICIES):
            raise SystemExit(f'{kernel_id} policies {tuple(by_policy)}')
        loaded[world_seed] = by_policy
        raw_kernels.append(payload)
    excerpt_src = loaded[0]['oracle']['excerpt']
    if excerpt_src is None:
        excerpt_payload = json.loads((out_dir / 'raw' / 's0' / 'result.json').read_text())
        excerpt_src = excerpt_payload.get('excerpt')
    if excerpt_src is None or len(excerpt_src) != EXCERPT_STEPS:
        raise SystemExit('oracle excerpt for world seed 0 is missing')
    local_oracle = world.rollout(0, 'oracle', T_FULL, BLOCK_FULL)
    for i, step in enumerate(excerpt_src):
        if (int(step['action']), int(step['room']), int(step['y'])) != (
            int(local_oracle['actions'][i]),
            int(local_oracle['rooms'][i]),
            int(local_oracle['ys'][i]),
        ):
            raise SystemExit(f'excerpt step {i} does not match the local oracle stream')

    per_seed = []
    for world_seed in KERNEL_IDS:
        rand8 = loaded[world_seed]['rand8']['probes'][-1]['PROBE']
        rand6 = loaded[world_seed]['rand6']['probes'][-1]['PROBE']
        oracle = loaded[world_seed]['oracle']['probes'][-1]['PROBE']
        per_seed.append({
            'world_seed': int(world_seed),
            'PROBE_rand8': rand8,
            'PROBE_rand6': rand6,
            'PROBE_oracle': oracle,
            'rand8_minus_oracle': rand8 - oracle,
            'rand6_minus_oracle': rand6 - oracle,
            'rand8_minus_rand6': rand8 - rand6,
        })
    gaps = {}
    for key in ('rand8_minus_oracle', 'rand6_minus_oracle', 'rand8_minus_rand6'):
        mean, std = _mean_std([row[key] for row in per_seed])
        gaps[key] = {'mean': mean, 'std_ddof0': std}

    kernels = []
    meta = kernel_meta or {}
    for payload in raw_kernels:
        kernel_id = payload['kernel_id']
        info = meta.get(kernel_id, {})
        script_s = float(payload['session_wall_seconds'])
        log_s = info.get('log_elapsed_seconds')
        if log_s is None:
            elapsed = script_s
            source = 'script_wall'
        else:
            elapsed = float(log_s)
            source = 'log_clock'
        kernels.append({
            'kernel_id': kernel_id,
            'world_seed': int(payload['world_seed']),
            'version': info.get('version'),
            'status': info.get('status'),
            'elapsed_seconds': elapsed,
            'elapsed_source': source,
            'session_wall_seconds': script_s,
            'log_elapsed_seconds': log_s,
            'torch_version': payload.get('torch_version'),
            'numpy_version': payload.get('numpy_version'),
            'cuda_available': payload.get('cuda_available'),
        })

    lives_out = []
    for world_seed in KERNEL_IDS:
        for policy in POLICIES:
            row = dict(loaded[world_seed][policy])
            row['excerpt'] = excerpt_src if world_seed == 0 and policy == 'oracle' else None
            lives_out.append(row)

    result = {
        'steps': T_FULL,
        'block_size': BLOCK_FULL,
        'vocab': VOCAB,
        'emb': EMB,
        'hidden': HIDDEN,
        'chunk_tokens': CHUNK,
        'lr': LR,
        'clip': CLIP,
        'replay_batch': REPLAY_BATCH,
        'learner_seed': SEED,
        'replay_seed': REPLAY_SEED + SEED,
        'n_params': expected_params(),
        'policies': list(POLICIES),
        'world_seeds': list(KERNEL_IDS),
        'lives': lives_out,
        'headroom': {
            'at_step': T_FULL,
            'per_seed': per_seed,
            'mean_std_ddof0': gaps,
        },
        'kernels': kernels,
        'excerpt': {
            'world_seed': 0,
            'policy': 'oracle',
            'n_tokens': EXCERPT_STEPS * 2,
            'steps': excerpt_src,
        },
    }
    dump(out_dir / 'results.json', result)
    lines = _markdown(result)
    (out_dir / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(
        'W2 headroom '
        + ' '.join(
            f"{key}={gaps[key]['mean']:.4f}±{gaps[key]['std_ddof0']:.4f}"
            for key in gaps),
        flush=True)
    return result


def _markdown(result):
    gaps = result['headroom']['mean_std_ddof0']
    lines = [
        '# W2 headroom',
        '',
        'Fixed policies on the locked-rooms world. Bits are prequential, on observation tokens only.',
        '',
        '## Headroom at T',
        '',
        'PROBE is the mean of probe_k over rooms 0..8. Headroom uses the probe at step 60000.',
        '',
        '| world_seed | PROBE rand8 | PROBE rand6 | PROBE oracle | rand8 − oracle | rand6 − oracle | rand8 − rand6 |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in result['headroom']['per_seed']:
        lines.append(
            f"| {row['world_seed']} | {_fmt(row['PROBE_rand8'])} | {_fmt(row['PROBE_rand6'])} | "
            f"{_fmt(row['PROBE_oracle'])} | {_fmt(row['rand8_minus_oracle'])} | "
            f"{_fmt(row['rand6_minus_oracle'])} | {_fmt(row['rand8_minus_rand6'])} |")
    lines.append(
        f"| mean |  |  |  | {_fmt(gaps['rand8_minus_oracle']['mean'])} | "
        f"{_fmt(gaps['rand6_minus_oracle']['mean'])} | {_fmt(gaps['rand8_minus_rand6']['mean'])} |")
    lines.append(
        f"| std ddof=0 |  |  |  | {_fmt(gaps['rand8_minus_oracle']['std_ddof0'])} | "
        f"{_fmt(gaps['rand6_minus_oracle']['std_ddof0'])} | {_fmt(gaps['rand8_minus_rand6']['std_ddof0'])} |")
    lines += ['', '## Lives', '']
    for row in result['lives']:
        lines += [
            f"### seed {row['world_seed']} {row['policy']}",
            '',
            f"Bits/observation { _fmt(row['bits_per_observation']) }. "
            f"Bits { _fmt(row['bits'], 3) }. Wall seconds { _fmt(row['wall_seconds'], 3) }. "
            f"Replay steps { row['replay_steps'] }. Finite { row['finite'] }.",
            '',
            f"Deepest room { row['deepest_room'] }. First step { row['deepest_first_step'] }.",
            '',
            'Time in room, rooms 0..8: ' + ' '.join(str(v) for v in row['time_in_room']) + '.',
            '',
            '| Quarter | Start | End | Observations | Bits | Bits/observation |',
            '|---:|---:|---:|---:|---:|---:|',
        ]
        for quarter in row['quarters']:
            lines.append(
                f"| {quarter['index']} | {quarter['start_step']} | {quarter['end_step']} | "
                f"{quarter['observations']} | {_fmt(quarter['bits'], 3)} | "
                f"{_fmt(quarter['bits_per_observation'])} |")
        lines += [
            '',
            '| Nominal step | At step | probe_0 | probe_1 | probe_2 | probe_3 | probe_4 | probe_5 | probe_6 | probe_7 | probe_8 | PROBE | TV |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for probe in row['probes']:
            cells = [_fmt(v) for v in probe['probe_k']]
            lines.append(
                f"| {probe['nominal_step']} | {probe['at_step']} | {' | '.join(cells)} | "
                f"{_fmt(probe['PROBE'])} | {_fmt(probe['tv_probe'])} |")
        lines.append('')
        lines.append(f"Token sha256 `{row['token_sha256']}`.")
        lines.append('')
    lines += ['## Kernels', '']
    lines.append('| Kernel | Version | Status | Elapsed seconds | Source | Script wall seconds |')
    lines.append('|---|---:|---|---:|---|---:|')
    for row in result['kernels']:
        version = '' if row['version'] is None else str(row['version'])
        status = '' if row['status'] is None else str(row['status'])
        lines.append(
            f"| {row['kernel_id']} | {version} | {status} | {_fmt(row['elapsed_seconds'], 3)} | "
            f"{row['elapsed_source']} | {_fmt(row['session_wall_seconds'], 3)} |")
    lines += [
        '',
        '## Excerpt',
        '',
        'Oracle, world seed 0, first 200 tokens, as (action, room, y).',
        '',
    ]
    for step in result['excerpt']['steps']:
        lines.append(f"({step['action']}, {step['room']}, {step['y']})")
    lines += [
        '',
        '## Protocol',
        '',
        '- Rooms 0..8. Actions 0..5 manipulate, 6 is the TV, 7 is reset. Vocabulary 152.',
        '- Observation token is 8 + 16·room + y, room after the action.',
        '- Keys, lock tables, room-8 out and next are drawn from default_rng(world_seed), in that order.',
        '- TV noise is default_rng(10000 + world_seed). Reset returns room 0, s 0, q 0, y 0.',
        '- rand8 is uniform on 0..7 and rand6 is uniform on 0..5, both from default_rng(5000 + world_seed).',
        '- Oracle blocks are 512 steps. Block b targets room b mod 9, presses reset, plays the keys to that room, then stays with default_rng(6000 + world_seed). It never presses the TV. At s = 2 it excludes key[2].',
        '- The last block is 96 steps because 60000 is not a multiple of 512. Its target is room 0.',
        '- GRU embedding 32, hidden 128, 1 layer, readout 152. 86680 parameters. Adam lr 0.001, clip 1.0. Learner seed 0. CPU, 4 threads.',
        '- Chunks are 64 tokens. Score the chunk, then one Adam step on the mean NLL of its observation tokens, state detached. Replay seed 30000. After each chunk, one step on 8 windows of 64, starts uniform on even positions that fit, zero state, observation loss only.',
        '- Checkpoints T/4, T/2 and T are 15000, 30000 and 60000. A probe runs at the first chunk boundary at or after the nominal step. Those boundaries are 15008, 30016 and 60000. The probe at T is exact.',
        '- A probe episode is a new World(world_seed), so the noise rng restarts. Room probes reset, play the keys, then 96 manipulations from default_rng(999 + room), excluding key[2] at s = 2. The path is teacher-forced and unscored. TV probe is 96 TV presses from room 0. PROBE is the unweighted mean of the nine room means.',
        '- Kernels ezzzzzx/w2-headroom-s0, s1 and s2. Private, CPU, no internet. Code dataset ezzzzzx/pdp-code-w2.',
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
