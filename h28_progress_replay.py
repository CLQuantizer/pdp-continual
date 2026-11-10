"""H28: does one whole-history replay step cure the curiosity noise leak?

Protocol: handoffs/2026-11-10_grok_h28_progress_with_replay.md. Numbers only; no verdict.
Creates only this file, kaggle/h28/* and results/h28/*. Imports earlier scripts;
does not edit them.

Readings of the spec, fixed before the scored runs:
- The machine is M4c's hist_rep cortex on FactWorld (10 channels, glance 64,
  alphabet 42, per-channel GRU state, Adam lr 3e-3, clip 1.0) with H27b's best
  dose: one live step, then one replay step (batch 8 x 64) from the whole
  history, every glance. Traffic is 8. M4c's hist_rep took two replay steps;
  this run takes one. No utility store and no match, so W is the frozen GRU.
- T = 512. Fresh seeds 27 28 29 30 31 32. Smoke is seed 27, T = 32.
- First pass: glance g visits channel g for g < 10, then the controller.
- rand is uniform on all 10 channels. rand8 is uniform on the 8 learnable
  channels. gmc_prop is M4c's estimator: GMC of the live step only, before
  replay, EMA beta 0.3, allocation proportional to max(value, 0) + 0.01.
- prog_prop is H25a's whole-history progress, not H25b's clipped coder.
  Cortex.reservoir_bits is that clipped coder and is not called. Each channel
  keeps an Algorithm R reservoir of 128 positions. A sample is up to 32
  preceding symbols plus the target, from a zero state, absent symbol only
  before the context. Reward = (bits before the glance's updates minus bits
  after the live step and the replay step) / 128. The current glance's
  positions are added after the reward.
- prog_cm_prop uses the same deltas. Reward = delta of the chosen channel
  minus the mean of the other 9 channels' deltas.
- Reservoir bits for every channel are one forward pass. Sequences are
  left-aligned; steps past the real context are not read, so the tail padding
  does not change the score.
- Choice and observations use 10_000 + seed. Replay draws use 30_000 + seed.
  Algorithm R uses 20_000 + seed. torch.manual_seed(seed). FactWorld(seed).
- The reward trace mean is the mean of the controller reward on glances that
  chose that channel. delta_mean, progress conditions only, is the mean over
  every glance of (before - after) / 128.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import torch
from torch.nn import functional as F

import m4b_ablation as m4b
from h21a_substrate import mean_std, now
from h25a_curio_neural import CONTEXT, RES
from h25b_progress_fix import gmc_reward

CONDS = ['rand', 'rand8', 'gmc_prop', 'prog_prop', 'prog_cm_prop']
PROGRESS = ('prog_prop', 'prog_cm_prop')
SEEDS = [27, 28, 29, 30, 31, 32]
SCORE_T = 512
SMOKE_T = 32
SMOKE_SEED = 27
BATCH = 8
assert RES == 128 and CONTEXT == 32
assert BATCH == 8 and m4b.B == 64 and m4b.A == 42
assert len(m4b.LEARN) == 8 and len(m4b.CH) == 10
assert abs(m4b.BETA - 0.3) < 1e-15
assert list(m4b.CH)[:8] == list(m4b.LEARN)
assert m4b.CH[8:] == ['N', 'Z']


def choose(cond, g, values, rng):
    if g < len(m4b.CH):
        return g
    if cond == 'rand':
        return int(rng.integers(len(m4b.CH)))
    if cond == 'rand8':
        return int(rng.integers(len(m4b.LEARN)))
    weights = np.maximum(values, 0.0) + 0.01
    return int(rng.choice(len(m4b.CH), p=weights / weights.sum()))


def progress_reward(cond, delta, ci):
    """delta is per-channel (bits before - bits after) / 128."""
    if cond == 'prog_prop':
        return float(delta[ci])
    if cond == 'prog_cm_prop':
        others = (float(delta.sum()) - float(delta[ci])) / (len(delta) - 1)
        return float(delta[ci]) - others
    raise ValueError(cond)


@torch.no_grad()
def reservoir_bits_all(model, reservoirs, hist):
    """H25a cross-entropy bits, every channel, one forward.

    inputs[0] is the absent symbol and inputs[1:c+1] are the c preceding
    symbols. The logit at index c is the one H25a reads. Later steps are
    padding and are not read.
    """
    chunks_pos, chunks_ci = [], []
    for ci, positions in enumerate(reservoirs):
        if not positions:
            continue
        chunks_pos.append(torch.tensor(positions, dtype=torch.long))
        chunks_ci.append(torch.full((len(positions),), ci, dtype=torch.long))
    totals = np.zeros(len(reservoirs), dtype=np.float64)
    if not chunks_pos:
        return totals
    pos = torch.cat(chunks_pos)
    cis = torch.cat(chunks_ci)
    length = torch.minimum(pos, torch.tensor(CONTEXT, dtype=torch.long))
    n = int(pos.shape[0])
    inputs = torch.full((CONTEXT + 1, n), -1, dtype=torch.long)
    for L in torch.unique(length).tolist():
        if L == 0:
            continue
        idx = (length == L).nonzero(as_tuple=False).flatten()
        offs = torch.arange(L)
        gathered = hist[cis[idx].unsqueeze(1), pos[idx].unsqueeze(1) - L + offs.unsqueeze(0)]
        inputs[1:L + 1, idx] = gathered.transpose(0, 1)
    targets = hist[cis, pos]
    logits, _ = model(inputs, cis)
    chosen = logits[length, torch.arange(n)]
    bits = F.cross_entropy(chosen, targets, reduction='none') / math.log(2)
    np.add.at(totals, cis.numpy(), bits.double().numpy())
    return totals


def _reference_bits(model, ci, positions, history):
    """H25a's per-length reservoir score, for the unit check."""
    if not positions:
        return 0.0
    positions = np.asarray(positions)
    total = 0.0
    with torch.no_grad():
        for L in np.unique(np.minimum(positions, CONTEXT)):
            pos = torch.tensor(positions[np.minimum(positions, CONTEXT) == L])
            inputs = torch.full((int(L) + 1, len(pos)), -1, dtype=torch.long)
            if L:
                inputs[1:] = history[pos[None, :] - int(L) + torch.arange(int(L))[:, None]]
            logits, _ = model(inputs, torch.full((len(pos),), ci, dtype=torch.long))
            total += float(F.cross_entropy(logits[-1], history[pos], reduction='sum')) / math.log(2)
    return total


def _admit(reservoir, oldlen, rrng):
    for position in range(oldlen, oldlen + m4b.B):
        if len(reservoir) < RES:
            reservoir.append(position)
        else:
            k = int(rrng.integers(position + 1))
            if k < RES:
                reservoir[k] = position


def run(cond, seed, glances, phase):
    if cond not in CONDS:
        raise ValueError(cond)
    progress = cond in PROGRESS
    started = time.monotonic()
    torch.manual_seed(seed)
    model = m4b.Cortex()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    world = m4b.FactWorld(seed)
    rng = np.random.default_rng(10_000 + seed)
    replay_rng = np.random.default_rng(30_000 + seed)
    rrng = np.random.default_rng(20_000 + seed)
    hist = torch.zeros((len(m4b.CH), glances * m4b.B), dtype=torch.long)
    lengths = np.zeros(len(m4b.CH), dtype=np.int64)
    states = [None] * len(m4b.CH)
    reservoirs = [[] for _ in m4b.CH]
    values = np.zeros(len(m4b.CH))
    counts = np.zeros(len(m4b.CH), dtype=int)
    reward_sum = np.zeros(len(m4b.CH), dtype=np.float64)
    delta_sum = np.zeros(len(m4b.CH), dtype=np.float64)
    log = []
    life = []
    replay_steps = 0

    def step(loss, coupling=False):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        floors = [1e-3 * p.detach().abs() for p in model.parameters()] if coupling else None
        opt.step()
        return gmc_reward(model, opt, floors) if coupling else 0.0

    for g in range(glances):
        ci = choose(cond, g, values, rng)
        channel = m4b.CH[ci]
        oldlen = int(lengths[ci])
        x = torch.tensor(m4b.observe(world, channel, rng), dtype=torch.long)
        before = reservoir_bits_all(model, reservoirs, hist) if progress else None
        if oldlen:
            previous = torch.cat((hist[ci, oldlen - 1:oldlen], x[:-1]))
        else:
            previous = torch.cat((torch.tensor([-1]), x[:-1]))
        logits, state = model(previous[:, None], torch.tensor([ci]), states[ci])
        losses = F.cross_entropy(logits[:, 0], x, reduction='none')
        states[ci] = state.detach()
        if cond == 'gmc_prop':
            reward = step(losses.mean(), coupling=True)
        else:
            step(losses.mean())
            reward = 0.0
        hist[ci, oldlen:oldlen + m4b.B] = x
        lengths[ci] = oldlen + m4b.B
        batch = m4b.sample_history(hist, lengths, replay_rng)
        if batch is None:
            raise AssertionError(f'{cond} empty history at glance {g}')
        previous_r, targets, cis = batch
        m4b.replay_batch(model, opt, previous_r, targets, cis)
        replay_steps += 1
        if progress:
            after = reservoir_bits_all(model, reservoirs, hist)
            delta = (before - after) / RES
            if len(reservoirs[ci]) == 0 and abs(float(delta[ci])) > 1e-8:
                raise AssertionError(f'{cond} scored an empty reservoir')
            reward = progress_reward(cond, delta, ci)
            if cond == 'prog_prop' and counts[ci] == 0 and abs(reward) > 1e-8:
                raise AssertionError('prog_prop first visit was not zero')
            delta_sum += delta
            _admit(reservoirs[ci], oldlen, rrng)
        if counts[ci] == 0:
            values[ci] = reward
        else:
            values[ci] = (1.0 - m4b.BETA) * values[ci] + m4b.BETA * reward
        reward_sum[ci] += reward
        counts[ci] += 1
        log.append(ci)
        life.append(float(losses.detach().double().mean() / math.log(2)))
        if (g + 1) % 64 == 0 or g + 1 == glances:
            print(f'{now()} {phase} {cond} s{seed} T={g + 1}/{glances} '
                  f'bits={np.mean(life):.4f} replay={replay_steps} '
                  f'wall={time.monotonic() - started:.1f}s',
                  flush=True)

    held = {}
    wc = copy.deepcopy(world)
    model.eval()
    with torch.no_grad():
        for lci, lch in enumerate(m4b.LEARN):
            seq = torch.tensor(
                [m4b.F_ALPHA.index(s) for s in wc.take(lch, m4b.EVAL)], dtype=torch.long)
            if lengths[lci]:
                last = hist[lci, lengths[lci] - 1:lengths[lci]]
            else:
                last = torch.tensor([-1])
            previous = torch.cat((last, seq[:-1]))
            state = states[lci]
            total = 0.0
            for start in range(0, m4b.EVAL, m4b.B):
                logits, state = model(
                    previous[start:start + m4b.B, None], torch.tensor([lci]), state)
                gp = logits[:, 0].double().softmax(-1).numpy()
                for offset, symbol in enumerate(seq[start:start + m4b.B].tolist()):
                    total += -math.log2(float(gp[offset, symbol]))
            held[lch] = total / m4b.EVAL
    choices = np.asarray(log, dtype=np.int64)
    n_obs = glances * m4b.B
    assert counts.sum() == glances and np.all(lengths == counts * m4b.B)
    assert replay_steps == glances
    assert choices.shape == (glances,)
    if glances >= len(m4b.CH):
        assert choices[:len(m4b.CH)].tolist() == list(range(len(m4b.CH)))
    if cond == 'rand8' and glances > len(m4b.CH):
        assert np.all(choices[len(m4b.CH):] < len(m4b.LEARN))
    if progress:
        for i, reservoir in enumerate(reservoirs):
            assert len(reservoir) == min(RES, int(lengths[i]))
            assert len(set(reservoir)) == len(reservoir)
    else:
        assert all(len(reservoir) == 0 for reservoir in reservoirs)
        assert np.all(delta_sum == 0.0)
    assert np.isfinite(life).all() and np.isfinite(list(held.values())).all()
    assert np.isfinite(reward_sum).all()
    result = dict(
        phase=phase, cond=cond, seed=int(seed), glances=int(glances),
        W=float(np.mean([held[c] for c in m4b.LEARN])),
        P=float(np.mean(life)),
        held={c: float(held[c]) for c in m4b.LEARN},
        share={c: float(counts[i] / glances) for i, c in enumerate(m4b.CH)},
        share_by_quarter=[
            {c: float(np.mean(q == i)) for i, c in enumerate(m4b.CH)}
            for q in np.array_split(choices, 4)],
        reward_mean={
            c: float(reward_sum[i] / counts[i]) if counts[i] else 0.0
            for i, c in enumerate(m4b.CH)},
        delta_mean=(None if not progress else {
            c: float(delta_sum[i] / glances) for i, c in enumerate(m4b.CH)}),
        reservoir_len=[len(r) for r in reservoirs],
        replay_steps=int(replay_steps),
        traffic=float(replay_steps * BATCH * m4b.B / n_obs),
        history_symbols=int(n_obs),
        history_bits_unpriced=float(n_obs * m4b.LOG2A),
        controller=cond,
        wall_seconds=time.monotonic() - started,
        parameters=int(sum(p.numel() for p in model.parameters())),
        glance_checksum=int(m4b.checksum(choices.tolist())),
        observation_checksum=int(m4b.observation_checksum(hist, lengths)),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
    )
    print(f'{now()} done {phase} {cond} s{seed} T={glances} W={result["W"]:.6f} '
          f'N={result["share"]["N"]:.4f} Z={result["share"]["Z"]:.4f} '
          f'wall={result["wall_seconds"]:.1f}s', flush=True)
    return result


def _write_run(result, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'run_{result["phase"]}_{result["cond"]}_{result["seed"]}_{result["glances"]}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return path


def configure_threads():
    m4b.configure_threads()


def _worker(job):
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    configure_threads()
    result = run(job['cond'], job['seed'], job['glances'], job['phase'])
    path = _write_run(result, job['out'])
    print(f'WROTE {path.name} W={result["W"]:.6f} wall={result["wall_seconds"]:.1f}', flush=True)
    return {'cond': result['cond'], 'seed': result['seed'], 'W': result['W'],
            'N': result['share']['N'], 'Z': result['share']['Z'],
            'wall_seconds': result['wall_seconds'],
            'glance_checksum': result['glance_checksum'],
            'observation_checksum': result['observation_checksum']}


def kernel_main(seeds, glances, phase, out, processes=2):
    """Kaggle entry. Two processes, two torch threads each. One JSON per run."""
    import multiprocessing as mp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    jobs = [{'cond': cond, 'seed': int(seed), 'glances': int(glances), 'phase': phase,
             'out': str(out)} for seed in seeds for cond in CONDS]
    try:
        mem = open('/proc/meminfo').readline().strip()
    except OSError:
        mem = ''
    print(f'kernel phase={phase} seeds={list(seeds)} T={glances} jobs={len(jobs)} '
          f'cpus={os.cpu_count()} torch={torch.__version__} {mem}', flush=True)
    started = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes) as pool:
        rows = pool.map(_worker, jobs)
    manifest = {
        'phase': phase, 'seeds': [int(s) for s in seeds], 'glances': int(glances),
        'processes': processes, 'cpus': os.cpu_count(), 'meminfo': mem,
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'started_unix': started, 'finished_unix': time.time(),
        'wall_seconds': time.time() - started, 'rows': rows,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('KERNEL_DONE ' + json.dumps(manifest), flush=True)
    return manifest


def _cell(value):
    return f'{value:.6f}'


def _pick(rows, seed, cond, key):
    for row in rows:
        if row['seed'] == seed and row['cond'] == cond:
            return row[key]
    raise KeyError((seed, cond, key))


def _problems(runs):
    problems = []
    by = {}
    for row in runs:
        by.setdefault(row['seed'], []).append(row)
    for seed, rows in sorted(by.items()):
        found = {r['cond'] for r in rows}
        if found != set(CONDS):
            problems.append(f'seed {seed} conditions {sorted(found)}')
        glances = rows[0]['glances']
        for row in rows:
            if row['glances'] != glances:
                problems.append(f'{row["cond"]} seed {seed} glances={row["glances"]}')
            if row['replay_steps'] != glances:
                problems.append(f'{row["cond"]} seed {seed} replay={row["replay_steps"]}')
            if abs(row['traffic'] - 8.0) > 1e-9:
                problems.append(f'{row["cond"]} seed {seed} traffic={row["traffic"]}')
            if row['history_symbols'] != glances * m4b.B:
                problems.append(f'{row["cond"]} seed {seed} history symbols')
            share_sum = sum(row['share'].values())
            if abs(share_sum - 1.0) > 1e-9:
                problems.append(f'{row["cond"]} seed {seed} shares sum to {share_sum}')
            if not math.isfinite(row['W']) or not (0.0 < row['W'] < m4b.LOG2A + 0.5):
                problems.append(f'{row["cond"]} seed {seed} W={row["W"]}')
            if not math.isfinite(row['P']) or not (0.0 < row['P'] < 8.0):
                problems.append(f'{row["cond"]} seed {seed} P={row["P"]}')
            for channel, value in row['reward_mean'].items():
                if not math.isfinite(value):
                    problems.append(f'{row["cond"]} seed {seed} reward {channel}')
            if row['cond'] in PROGRESS:
                if not row['delta_mean']:
                    problems.append(f'{row["cond"]} seed {seed} missing delta_mean')
                elif any(not math.isfinite(v) for v in row['delta_mean'].values()):
                    problems.append(f'{row["cond"]} seed {seed} delta not finite')
                expect = [min(RES, int(round(row['share'][c] * glances * m4b.B)))
                          for c in m4b.CH]
                if row['reservoir_len'] != expect:
                    problems.append(f'{row["cond"]} seed {seed} reservoir_len')
            elif row['reservoir_len'] != [0] * len(m4b.CH):
                problems.append(f'{row["cond"]} seed {seed} reservoir not empty')
            if glances >= len(m4b.CH) and row['cond'] == 'rand8':
                expect_nz = 1.0 / glances
                if abs(row['share']['N'] - expect_nz) > 1e-12 or abs(row['share']['Z'] - expect_nz) > 1e-12:
                    problems.append(
                        f'rand8 seed {seed} N={row["share"]["N"]} Z={row["share"]["Z"]}')
            for quarter in row['share_by_quarter']:
                if abs(sum(quarter.values()) - 1.0) > 1e-9:
                    problems.append(f'{row["cond"]} seed {seed} quarter shares')
    return problems


def _share_line(rows, key):
    return ' | '.join(f'{np.mean([r[key][c] for r in rows]):.6f}' for c in m4b.CH)


def write_report(score, smoke, kernels, out, started, notes, match_info):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        '# H28 measurements', '',
        f'Started: {started}. Updated: {now()}.', '',
        'Kaggle CPU. 2 processes, torch.set_num_threads(2) each. '
        'Standard deviation: population (ddof=0). No verdict.', '',
        f'Chosen T: {SCORE_T}. Scoring seeds: {SEEDS}.',
        'Conditions: rand; rand8 (uniform on the 8 learnable channels); '
        'gmc_prop (M4c live-step GMC); prog_prop (H25a whole-history progress, '
        'one replay step included); prog_cm_prop (same, common mode of the other '
        '9 channels removed).',
        'W is the mean held-out bits over the 8 learnable channels. '
        'P is the mean live bits/symbol over every glance, taken before that glance\'s replay. '
        'One replay step of batch 8 after every glance. Traffic is 8. '
        'No utility store and no match.',
        'The reward mean is the mean of the controller reward on glances that chose that channel. '
        'delta_mean is the mean over every glance of (reservoir bits before − after) / 128.',
        '',
    ]
    header = ('| Condition | W | N share | Z share | ' + ' | '.join(m4b.LEARN)
              + ' | Wall seconds (sum) |')
    align = '|---|---:|---:|---:|' + '---:|' * len(m4b.LEARN) + '---:|'
    lines += [header, align]
    for cond in CONDS:
        rs = [r for r in score if r['cond'] == cond]
        if not rs:
            continue
        chans = ' | '.join(mean_std([r['held'][c] for r in rs]) for c in m4b.LEARN)
        lines.append(
            f'| {cond} | {mean_std([r["W"] for r in rs])} | '
            f'{np.mean([r["share"]["N"] for r in rs]):.6f} | '
            f'{np.mean([r["share"]["Z"] for r in rs]):.6f} | {chans} | '
            f'{sum(r["wall_seconds"] for r in rs):.3f} |')
    lines += ['', 'Per seed. W, P, N share, Z share, replay steps.', '',
              '| Seed | Condition | W | P | N share | Z share | replay steps | traffic | Wall seconds |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in score:
        lines.append(
            f'| {row["seed"]} | {row["cond"]} | {_cell(row["W"])} | {_cell(row["P"])} | '
            f'{row["share"]["N"]:.6f} | {row["share"]["Z"]:.6f} | {row["replay_steps"]} | '
            f'{row["traffic"]:.1f} | {row["wall_seconds"]:.3f} |')
    lines += ['', 'Per seed, held-out bits per learnable channel.', '',
              '| Seed | Condition | ' + ' | '.join(m4b.LEARN) + ' |',
              '|---:|---|' + '---:|' * len(m4b.LEARN)]
    for row in score:
        lines.append(
            f'| {row["seed"]} | {row["cond"]} | '
            + ' | '.join(_cell(row['held'][c]) for c in m4b.LEARN) + ' |')
    if score:
        seeds = sorted({r['seed'] for r in score})
        lines += ['', 'Measurements for the pre-registered comparisons. Bars are not scored here.', '',
                  '| Seed | W rand | W rand8 | W gmc_prop | W prog_prop | W prog_cm_prop |',
                  '|---:|---:|---:|---:|---:|---:|']
        for seed in seeds:
            lines.append(
                f'| {seed} | '
                + ' | '.join(_cell(_pick(score, seed, cond, 'W')) for cond in CONDS) + ' |')
        lines += ['', '| Seed | prog_prop − rand | prog_cm_prop − rand | prog_prop − gmc_prop | '
                  'prog_cm_prop − gmc_prop | prog_prop − rand8 | prog_cm_prop − rand8 |',
                  '|---:|---:|---:|---:|---:|---:|---:|']
        gaps = {name: [] for name in ('pp_rand', 'cm_rand', 'pp_gmc', 'cm_gmc', 'pp_r8', 'cm_r8')}
        for seed in seeds:
            w = {cond: _pick(score, seed, cond, 'W') for cond in CONDS}
            row_gaps = [
                w['prog_prop'] - w['rand'], w['prog_cm_prop'] - w['rand'],
                w['prog_prop'] - w['gmc_prop'], w['prog_cm_prop'] - w['gmc_prop'],
                w['prog_prop'] - w['rand8'], w['prog_cm_prop'] - w['rand8'],
            ]
            for key, gap in zip(gaps, row_gaps):
                gaps[key].append(gap)
            lines.append(f'| {seed} | ' + ' | '.join(_cell(g) for g in row_gaps) + ' |')
        lines.append('| mean | ' + ' | '.join(
            f'{np.mean(v):.6f} ± {np.std(v, ddof=0):.6f}' for v in gaps.values()) + ' |')
        lines += ['', '| Seed | Condition | N share | Z share |',
                  '|---:|---|---:|---:|']
        for seed in seeds:
            for cond in CONDS:
                share = _pick(score, seed, cond, 'share')
                lines.append(f'| {seed} | {cond} | {share["N"]:.6f} | {share["Z"]:.6f} |')
    lines += ['', 'Mean overall glance shares.', '',
              '| Condition | ' + ' | '.join(m4b.CH) + ' |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cond in CONDS:
        rs = [r for r in score if r['cond'] == cond]
        if rs:
            lines.append(f'| {cond} | {_share_line(rs, "share")} |')
    lines += ['', 'Mean glance shares by quarter.', '',
              '| Condition | Quarter | ' + ' | '.join(m4b.CH) + ' |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cond in CONDS:
        rs = [r for r in score if r['cond'] == cond]
        if not rs:
            continue
        for q in range(4):
            cells = ' | '.join(
                f'{np.mean([r["share_by_quarter"][q][c] for r in rs]):.6f}' for c in m4b.CH)
            lines.append(f'| {cond} | {q + 1} | {cells} |')
    lines += ['', 'Glance shares per seed, overall and by quarter.', '',
              '| Seed | Condition | Period | ' + ' | '.join(m4b.CH) + ' |',
              '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in score:
        periods = [('all', row['share'])] + [
            (f'Q{i + 1}', q) for i, q in enumerate(row['share_by_quarter'])]
        for label, shares in periods:
            cells = ' | '.join(f'{shares[c]:.6f}' for c in m4b.CH)
            lines.append(f'| {row["seed"]} | {row["cond"]} | {label} | {cells} |')
    lines += ['', 'Reward trace means, per channel. Mean of the controller reward on visits.', '',
              '| Condition | ' + ' | '.join(m4b.CH) + ' |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cond in CONDS:
        rs = [r for r in score if r['cond'] == cond]
        if rs:
            lines.append(f'| {cond} | {_share_line(rs, "reward_mean")} |')
    lines += ['', 'Reward trace means per seed.', '',
              '| Seed | Condition | ' + ' | '.join(m4b.CH) + ' |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in score:
        cells = ' | '.join(_cell(row['reward_mean'][c]) for c in m4b.CH)
        lines.append(f'| {row["seed"]} | {row["cond"]} | {cells} |')
    lines += ['', 'delta_mean per channel, progress conditions. '
              'Mean over every glance of (before − after) / 128.', '',
              '| Condition | ' + ' | '.join(m4b.CH) + ' |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cond in PROGRESS:
        rs = [r for r in score if r['cond'] == cond and r['delta_mean']]
        if rs:
            lines.append(f'| {cond} | {_share_line(rs, "delta_mean")} |')
    lines += ['', 'delta_mean per seed.', '',
              '| Seed | Condition | ' + ' | '.join(m4b.CH) + ' |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in score:
        if not row['delta_mean']:
            continue
        cells = ' | '.join(_cell(row['delta_mean'][c]) for c in m4b.CH)
        lines.append(f'| {row["seed"]} | {row["cond"]} | {cells} |')
    if smoke:
        lines += ['', f'Smoke, Kaggle, seed {SMOKE_SEED}, T = {SMOKE_T}, all 5 conditions.', '',
                  '| Condition | W | P | N share | Z share | replay steps | Wall seconds | Glance checksum |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in smoke:
            lines.append(
                f'| {row["cond"]} | {_cell(row["W"])} | {_cell(row["P"])} | '
                f'{row["share"]["N"]:.6f} | {row["share"]["Z"]:.6f} | {row["replay_steps"]} | '
                f'{row["wall_seconds"]:.3f} | {row["glance_checksum"]} |')
    if match_info:
        lines += ['', 'Local match, before the scored kernels. Not a score.',
                  f'- Seed {match_info["seed"]}, T = {match_info["glances"]}. '
                  f'rand W − h27b hist_e1 W = {match_info["W_delta"]:.3e}. '
                  f'Glance checksum {match_info["glance_checksum"]} shared. '
                  f'Replay steps {match_info["replay_steps"]}.']
    lines += ['', 'Kaggle kernels.', '',
              '| Id | Phase | Seeds | T | Status | Elapsed seconds | Rerun | Note |',
              '|---|---|---|---:|---|---:|---|---|']
    for ker in kernels:
        lines.append(
            f'| {ker["id"]} | {ker["phase"]} | {ker["seeds"]} | {ker["T"]} | {ker["status"]} | '
            f'{ker["elapsed"]} | {ker["rerun"]} | {ker["note"]} |')
    lines += ['', 'Protocol notes:',
              '- World: FactWorld, channels G C1 C7 E30 E110 M7 M37 F N Z. Glance = 64 symbols.',
              '- N: uniform over 42 symbols. Z: the space symbol repeated. Seeds 27 28 29 30 31 32.',
              '- GRU: symbol embedding 32, channel embedding 8, hidden 256, Adam lr 3e-3, clip 1.0.',
              '  Per-channel state. One live Adam step per glance, then one replay step.',
              '- Replay is m4b.sample_history: uniform over starts 0 through L−64, batch 8, from a',
              '  zero state. The glance is written before the replay draw, so it can be sampled.',
              '- Traffic = 8 × 64 / 64 = 8. The history log is not part of W.',
              '  history_bits_unpriced = observed symbols × log2(42).',
              '- W: frozen final GRU, fresh 2048-symbol continuation of each learnable channel,',
              '  from a deep-copied world, starting from the saved live channel state.',
              '- prog_prop / prog_cm_prop score H25a unclamped cross-entropy bits. The H25b clipped',
              '  coder on Cortex.reservoir_bits is not used. Divide by 128, including short reservoirs.',
              '- Current positions enter the reservoir after the reward. Algorithm R, one stream,',
              '  numpy seed 20_000 + seed. Reservoir eval is one batched forward and does not',
              '  change live state or weights.',
              '- prog_cm_prop reward = Δ[chosen] − mean of the other 9 Δ. Empty reservoirs contribute',
              '  Δ = 0 and stay in that mean.',
              '- gmc_prop reward is the live-step GMC only, computed before replay, as M4c.',
              '- Choice and observations use 10_000 + seed. Replay uses 30_000 + seed.',
              '  torch.manual_seed(seed). FactWorld(seed).',
              '- First pass: glance g visits channel g for g < 10, in every condition, then the controller.',
              '- EMA is (1 − 0.3) × value + 0.3 × reward. Proportional weights are max(value, 0) + 0.01.',
              '']
    if notes:
        lines += ['Checks:', ''] + [f'- {item}' for item in notes] + ['']
    else:
        lines += ['Checks: none.', '']
    n_score = len(score)
    lines += [f'Score runs: {n_score} of {len(SEEDS) * len(CONDS)}.',
              f'Status: {"complete" if n_score == len(SEEDS) * len(CONDS) and not notes else "incomplete"}.',
              '']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    payload = {
        'started': started, 'updated': now(), 'score_T': SCORE_T, 'seeds': SEEDS,
        'conditions': CONDS, 'runs': score, 'smoke': smoke, 'kernels': kernels,
        'match': match_info, 'notes': notes,
        'status': 'complete' if n_score == len(SEEDS) * len(CONDS) and not notes else 'incomplete',
    }
    (out / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return payload['status']


def load_runs(folder):
    rows = []
    folder = Path(folder)
    if not folder.exists():
        return rows
    for path in sorted(folder.rglob('run_*.json')):
        rows.append(json.loads(path.read_text()))
    return rows


def assemble(root):
    root = Path(root)
    score, smoke = [], []
    for row in load_runs(root / 'raw'):
        if row.get('phase') == 'score' and row.get('glances') == SCORE_T and row.get('seed') in SEEDS:
            score.append(row)
        elif (row.get('phase') == 'smoke' and row.get('glances') == SMOKE_T
              and row.get('seed') == SMOKE_SEED):
            smoke.append(row)
    score.sort(key=lambda r: (r['seed'], CONDS.index(r['cond'])))
    smoke.sort(key=lambda r: CONDS.index(r['cond']))
    ker_path = root / 'kernels.json'
    kernels = json.loads(ker_path.read_text()).get('kernels', []) if ker_path.exists() else []
    meta_path = root / 'meta.json'
    started = json.loads(meta_path.read_text())['started'] if meta_path.exists() else now()
    match_path = root / 'match.json'
    match_info = json.loads(match_path.read_text()) if match_path.exists() else None
    notes = []
    if score:
        notes.extend(_problems(score))
    if smoke:
        notes.extend('smoke: ' + item for item in _problems(smoke))
    return write_report(score, smoke, kernels, root, started, notes, match_info)


def _check_reservoir():
    torch.manual_seed(0)
    model = m4b.Cortex()
    hist = torch.randint(0, m4b.A, (len(m4b.CH), 400))
    reservoirs = [
        [],
        [0],
        [0, 1, 5, 40],
        list(range(0, 200, 3)),
        list(range(128)),
        [7, 31, 32, 33, 100],
        list(range(50, 178)),
        [2, 8, 16],
        list(range(300, 364)),
        [0, 32, 63],
    ]
    got = reservoir_bits_all(model, reservoirs, hist)
    for ci, positions in enumerate(reservoirs):
        ref = _reference_bits(model, ci, positions, hist[ci])
        if abs(got[ci] - ref) > 1e-4:
            raise AssertionError(f'reservoir bits ci={ci} got={got[ci]} ref={ref}')
    empty = reservoir_bits_all(model, [[] for _ in m4b.CH], hist)
    if np.any(empty != 0.0):
        raise AssertionError('empty reservoir scored bits')
    delta = np.arange(10, dtype=np.float64)
    assert abs(progress_reward('prog_prop', delta, 3) - 3.0) < 1e-12
    others = (delta.sum() - delta[3]) / 9.0
    assert abs(progress_reward('prog_cm_prop', delta, 3) - (3.0 - others)) < 1e-12
    flat = np.full(10, 1.5)
    assert abs(progress_reward('prog_cm_prop', flat, 4)) < 1e-12
    only = np.zeros(10)
    only[2] = 1.0
    assert abs(progress_reward('prog_cm_prop', only, 2) - 1.0) < 1e-12


def checks():
    _check_reservoir()
    rng = np.random.default_rng(0)
    values = np.zeros(len(m4b.CH))
    assert [choose('rand', g, values, rng) for g in range(len(m4b.CH))] == list(range(len(m4b.CH)))
    assert choose('rand8', len(m4b.CH), values, rng) < len(m4b.LEARN)
    print(f'{now()} h28 reservoir checks passed', flush=True)


def match(glances=8, seed=21, out=None):
    """rand reproduces H27b hist_e1: same cortex, one history step per glance."""
    import h27b_matched_dose as h27b
    configure_threads()
    mine = run('rand', seed, glances, 'match')
    theirs = h27b.run('hist_e1', seed, glances, 'match')
    if mine['glance_checksum'] != theirs['glance_checksum']:
        raise SystemExit('rand glance checksum diverged from h27b hist_e1')
    if mine['observation_checksum'] != theirs['observation_checksum']:
        raise SystemExit('rand observation checksum diverged from h27b hist_e1')
    if mine['replay_steps'] != theirs['replay_steps']:
        raise SystemExit(
            f'replay_steps {mine["replay_steps"]} != {theirs["replay_steps"]}')
    if abs(mine['W'] - theirs['W']) > 1e-8:
        raise SystemExit(f'W {mine["W"]:.8f} != hist_e1 {theirs["W"]:.8f}')
    for channel in m4b.LEARN:
        if abs(mine['held'][channel] - theirs['held'][channel]) > 1e-8:
            raise SystemExit(f'{channel} held diverged')
    info = {
        'seed': int(seed), 'glances': int(glances),
        'W_delta': float(mine['W'] - theirs['W']),
        'glance_checksum': int(mine['glance_checksum']),
        'replay_steps': int(mine['replay_steps']),
        'W': float(mine['W']),
    }
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'match.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'{now()} h28 match passed W={mine["W"]:.6f} delta={info["W_delta"]:.3e}', flush=True)
    return info


def _short_life(glances=16, seed=27):
    """Every controller runs, and the two progress rewards differ once history exists."""
    configure_threads()
    ran = {}
    for cond in CONDS:
        ran[cond] = run(cond, seed, glances, 'check')
    problems = _problems(list(ran.values()))
    if problems:
        raise SystemExit(f'short life: {problems}')
    if abs(ran['prog_prop']['W'] - ran['prog_cm_prop']['W']) < 1e-12:
        # Identical W is possible but the rewards should still differ: common mode
        # is not zero once another channel has a reservoir.
        same = all(
            abs(ran['prog_prop']['reward_mean'][c] - ran['prog_cm_prop']['reward_mean'][c]) < 1e-12
            for c in m4b.CH)
        if same:
            raise SystemExit('prog_prop and prog_cm_prop rewards matched')
    print(f'{now()} h28 short life passed', flush=True)
    return {cond: {'W': row['W'], 'N': row['share']['N'], 'Z': row['share']['Z'],
                   'wall_seconds': row['wall_seconds']} for cond, row in ran.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--match', action='store_true')
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--out', default='results/h28')
    args = parser.parse_args()
    if args.checks:
        checks()
    elif args.match:
        match(out=args.out)
    elif args.short:
        _short_life()
    elif args.assemble:
        status = assemble(args.out)
        print(status, flush=True)
    else:
        parser.error('give --checks, --match, --short or --assemble')


if __name__ == '__main__':
    main()
