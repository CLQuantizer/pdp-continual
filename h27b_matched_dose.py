"""H27b: bounded memory at a matched replay dose.

Protocol: handoffs/2026-11-10_grok_h27b_matched_dose.md. Numbers only; no verdict.
Creates only this file, kaggle/h27b/* and results/h27b/*. Imports earlier scripts;
does not edit them.

Readings of the spec, fixed before the scored runs:
- World, GRU, live step and held-out W are H27 / M4c hist_rep: controller rand
  (m4b.choose('hist_rep')), no utility store, no match. T = 512. Fresh seeds
  21 22 23 24 25 26. Glance = 64. Replay batch = 8.
- "Every k" is one replay step after glances k, 2k, 3k, ... (1-based). The
  glance just observed is inserted into a bounded bank before that decision.
  Traffic = replayed symbols / observed symbols = 8/k when k divides T.
  T = 512 and the smoke T = 32 are divisible by 1, 2, 4 and 8.
- hist_e1 is H27 hist1: m4b.sample_history, one step every glance. hist_e2 and
  hist_e4 are that source with the step taken every 2nd or 4th glance.
- res* is H27's Algorithm R over glance-aligned segments. Capacity 77 is H27's
  15% of the T = 512 log. Capacity 308 is the specified 60% figure
  (308/512 = 0.6015625), not round(0.6*512).
- M is the H27 charge: retained segments × 64 × log2(42). hist* charges every
  observed glance. res* charges final occupancy. nostore charges 0. The raw
  log stays so a bank can index glances; M does not charge it for res*.
- Replays per segment, over the life, is total segment-draws divided by the
  sampling units ever admitted. A history unit is one valid overlapping
  window (what sample_history draws). A reservoir unit is one glance Algorithm
  R admitted, including a glance later replaced. nostore is 0.
- Draws per M-segment is total segment-draws divided by the segment count M
  charges. That is the H27 "~100 repeats" ratio.
- P is the mean live bits/symbol over every glance, scored before that
  glance's replay. It is not W and it is not a bar.
- Choice and observations use 10_000 + seed. Replay draws use 30_000 + seed.
  Algorithm R uses 40_000 + seed. torch.manual_seed(seed). FactWorld(seed).
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

import h27_bounded_replay as h27
import m4b_ablation as m4b
from h21a_substrate import mean_std, now
from h27_bounded_replay import Reservoir, memory_bits, sample_bank

CONDS = [
    'nostore', 'hist_e1', 'hist_e2', 'hist_e4',
    'res77_e4', 'res77_e8', 'res308_e1', 'res308_e4',
]
SEEDS = [21, 22, 23, 24, 25, 26]
SCORE_T = 512
SMOKE_T = 32
BATCH = h27.BATCH
RES_NAMES = ['res77_e4', 'res77_e8', 'res308_e1', 'res308_e4']
HIST_DOSE = ['hist_e1', 'hist_e2', 'hist_e4']
SPEC = {
    'nostore': {'source': 'none', 'every': None, 'capacity': 0},
    'hist_e1': {'source': 'history', 'every': 1, 'capacity': None},
    'hist_e2': {'source': 'history', 'every': 2, 'capacity': None},
    'hist_e4': {'source': 'history', 'every': 4, 'capacity': None},
    'res77_e4': {'source': 'reservoir', 'every': 4, 'capacity': 77},
    'res77_e8': {'source': 'reservoir', 'every': 8, 'capacity': 77},
    'res308_e1': {'source': 'reservoir', 'every': 1, 'capacity': 308},
    'res308_e4': {'source': 'reservoir', 'every': 4, 'capacity': 308},
}
TRAFFIC = {
    'nostore': 0, 'hist_e1': 8, 'hist_e2': 4, 'hist_e4': 2,
    'res77_e4': 2, 'res77_e8': 1, 'res308_e1': 8, 'res308_e4': 2,
}
assert list(SPEC) == CONDS and set(TRAFFIC) == set(CONDS)
assert BATCH == 8 and m4b.B == 64 and m4b.A == 42
assert len(m4b.LEARN) == 8 and len(m4b.CH) == 10
assert SPEC['res77_e4']['capacity'] == 77 and SPEC['res77_e8']['capacity'] == 77
assert SPEC['res308_e1']['capacity'] == 308 and SPEC['res308_e4']['capacity'] == 308
assert abs(308 / 512 - 0.6015625) < 1e-12
assert abs(0.15 * 512 - 76.8) < 1e-9
assert SPEC['hist_e1']['every'] == 1 and SPEC['hist_e1']['source'] == 'history'
assert h27.SPEC['hist1']['dose'] == 1 and h27.SPEC['hist1']['source'] == 'history'
for name, spec in SPEC.items():
    every = spec['every']
    if every is None:
        assert TRAFFIC[name] == 0 and spec['source'] == 'none'
        continue
    assert SCORE_T % every == 0 and SMOKE_T % every == 0
    assert TRAFFIC[name] * every == BATCH
    assert spec['source'] in ('history', 'reservoir')


class TrackedReservoir(Reservoir):
    """H27 Algorithm R, plus a count of glances that occupied a slot.

    The RNG calls match Reservoir.add. A miss does not increment admitted.
    """

    def __init__(self, capacity, rng):
        super().__init__(capacity, rng)
        self.admitted = 0

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            self.admitted += 1
            return
        j = int(self.rng.integers(0, self.seen))
        if j < self.capacity:
            self.items[j] = item
            self.admitted += 1


def valid_windows(lengths):
    return int(np.maximum(np.asarray(lengths) - m4b.B + 1, 0).sum())


def expect_windows(glances):
    """Every channel the first pass touches has at least one glance."""
    covered = min(int(glances), len(m4b.CH))
    return int(m4b.B * glances - (m4b.B - 1) * covered)


def _ratio(num, den):
    return 0.0 if den == 0 else float(num) / float(den)


def _check_divisible(glances):
    for name, spec in SPEC.items():
        every = spec['every']
        if every and int(glances) % every:
            raise SystemExit(f'{name}: T={glances} is not divisible by every={every}')


def run(cond, seed, glances, phase):
    """One rand life. Replay source and the every-k schedule are the only knobs."""
    if cond not in SPEC:
        raise ValueError(cond)
    _check_divisible(glances)
    spec = SPEC[cond]
    source, every, capacity = spec['source'], spec['every'], spec['capacity']
    started = time.monotonic()
    torch.manual_seed(seed)
    model = m4b.Cortex()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    world = m4b.FactWorld(seed)
    rng = np.random.default_rng(10_000 + seed)
    replay_rng = np.random.default_rng(30_000 + seed)
    hist = torch.zeros((len(m4b.CH), glances * m4b.B), dtype=torch.long)
    lengths = np.zeros(len(m4b.CH), dtype=np.int64)
    states = [None] * len(m4b.CH)
    counts = np.zeros(len(m4b.CH), dtype=int)
    values = np.zeros(len(m4b.CH))
    bank = None
    if source == 'reservoir':
        bank = TrackedReservoir(capacity, np.random.default_rng(40_000 + seed))
    log = []
    life = []
    replay_steps = 0

    def step(loss):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for g in range(glances):
        ci = m4b.choose('hist_rep', g, values, rng)
        channel = m4b.CH[ci]
        oldlen = int(lengths[ci])
        x = torch.tensor(m4b.observe(world, channel, rng), dtype=torch.long)
        hist[ci, oldlen:oldlen + m4b.B] = x
        lengths[ci] = oldlen + m4b.B
        if oldlen:
            previous = torch.cat((hist[ci, oldlen - 1:oldlen], x[:-1]))
        else:
            previous = torch.cat((torch.tensor([-1]), x[:-1]))
        logits, state = model(previous[:, None], torch.tensor([ci]), states[ci])
        losses = F.cross_entropy(logits[:, 0], x, reduction='none')
        states[ci] = state.detach()
        step(losses.mean())
        if bank is not None:
            bank.add((ci, oldlen))
        if every is not None and (g + 1) % every == 0:
            if source == 'history':
                batch = m4b.sample_history(hist, lengths, replay_rng)
            else:
                batch = sample_bank(hist, bank.items, replay_rng)
            if batch is None:
                raise AssertionError(f'{cond} replay with an empty source at glance {g}')
            previous_r, targets, cis = batch
            m4b.replay_batch(model, opt, previous_r, targets, cis)
            replay_steps += 1
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
    choices = np.array(log)
    n_obs = glances * m4b.B
    windows = valid_windows(lengths)
    if source == 'none':
        segments = 0
        admitted = 0
    elif source == 'history':
        segments = int(glances)
        admitted = int(windows)
    else:
        segments = len(bank.items)
        admitted = int(bank.admitted)
    draws = int(replay_steps * BATCH)
    replayed = int(draws * m4b.B)
    assert counts.sum() == glances and np.all(lengths == counts * m4b.B)
    assert windows == expect_windows(glances)
    if source == 'none':
        assert replay_steps == 0 and segments == 0 and admitted == 0
    else:
        assert replay_steps == glances // every
        if source == 'history':
            assert segments == glances and admitted == windows
        else:
            assert bank.seen == glances
            assert segments == min(capacity, glances)
            assert admitted >= segments
            assert admitted <= glances
    result = dict(
        phase=phase, cond=cond, seed=int(seed), glances=int(glances),
        W=float(np.mean([held[c] for c in m4b.LEARN])),
        P=float(np.mean(life)),
        held={c: float(held[c]) for c in m4b.LEARN},
        M_bits=memory_bits(segments),
        segments=int(segments),
        segments_admitted=int(admitted),
        valid_windows=int(windows),
        capacity=None if capacity is None else int(capacity),
        traffic=float(replayed / n_obs),
        replays_per_segment=_ratio(draws, admitted),
        draws_per_m_segment=_ratio(draws, segments),
        segment_draws=int(draws),
        replayed_symbols=replayed,
        observed_symbols=int(n_obs),
        replay_steps=int(replay_steps),
        every=None if every is None else int(every),
        replay_source=source,
        controller='rand',
        wall_seconds=time.monotonic() - started,
        parameters=int(sum(p.numel() for p in model.parameters())),
        glance_checksum=int(m4b.checksum(choices.tolist())),
        observation_checksum=int(m4b.observation_checksum(hist, lengths)),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
    )
    print(f'{now()} done {phase} {cond} s{seed} T={glances} W={result["W"]:.6f} '
          f'P={result["P"]:.6f} M={result["M_bits"]:.3f} traffic={result["traffic"]:.4f} '
          f'replays={result["replays_per_segment"]:.4f} wall={result["wall_seconds"]:.1f}s',
          flush=True)
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
            'P': result['P'], 'M_bits': result['M_bits'], 'traffic': result['traffic'],
            'replays_per_segment': result['replays_per_segment'],
            'wall_seconds': result['wall_seconds'],
            'glance_checksum': result['glance_checksum'],
            'observation_checksum': result['observation_checksum']}


def kernel_main(seeds, glances, phase, out, processes=2):
    """Kaggle entry. Two processes, two torch threads each. One JSON per run."""
    import multiprocessing as mp
    _check_divisible(glances)
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
    print('spec ' + json.dumps(SPEC), flush=True)
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


def _close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def _problems(runs):
    problems = []
    by = {}
    for row in runs:
        by.setdefault(row['seed'], []).append(row)
    for seed, rows in sorted(by.items()):
        if len({r['glance_checksum'] for r in rows}) != 1:
            problems.append(f'seed {seed} glance logs diverged')
        if len({r['observation_checksum'] for r in rows}) != 1:
            problems.append(f'seed {seed} observation logs diverged')
        if len({r['valid_windows'] for r in rows}) != 1:
            problems.append(f'seed {seed} valid windows diverged')
        found = {r['cond'] for r in rows}
        if found != set(CONDS):
            problems.append(f'seed {seed} conditions {sorted(found)}')
        glances = rows[0]['glances']
        for row in rows:
            spec = SPEC[row['cond']]
            if row['glances'] != glances:
                problems.append(f'{row["cond"]} seed {seed} glances={row["glances"]}')
            if row['valid_windows'] != expect_windows(row['glances']):
                problems.append(
                    f'{row["cond"]} seed {seed} windows={row["valid_windows"]}')
            every = spec['every']
            if spec['source'] == 'none':
                expect_steps = 0
                expect_seg = 0
                expect_admitted = 0
                expect_traffic = 0.0
            else:
                expect_steps = row['glances'] // every
                expect_traffic = expect_steps * BATCH / row['glances']
                if spec['source'] == 'history':
                    expect_seg = row['glances']
                    expect_admitted = row['valid_windows']
                else:
                    expect_seg = min(spec['capacity'], row['glances'])
                    probe = TrackedReservoir(
                        spec['capacity'], np.random.default_rng(40_000 + seed))
                    for i in range(row['glances']):
                        probe.add(i)
                    expect_admitted = probe.admitted
                    if probe.seen != row['glances'] or len(probe.items) != expect_seg:
                        problems.append(f'{row["cond"]} seed {seed} probe diverged')
            draws = expect_steps * BATCH
            if row['replay_steps'] != expect_steps:
                problems.append(f'{row["cond"]} seed {seed} replay_steps={row["replay_steps"]}')
            if row['segments'] != expect_seg:
                problems.append(f'{row["cond"]} seed {seed} segments={row["segments"]}')
            if row['segments_admitted'] != expect_admitted:
                problems.append(
                    f'{row["cond"]} seed {seed} admitted={row["segments_admitted"]}')
            if not _close(row['M_bits'], memory_bits(expect_seg), 1e-6):
                problems.append(f'{row["cond"]} seed {seed} M={row["M_bits"]}')
            if not _close(row['traffic'], expect_traffic):
                problems.append(f'{row["cond"]} seed {seed} traffic={row["traffic"]}')
            if row['segment_draws'] != draws:
                problems.append(f'{row["cond"]} seed {seed} draws={row["segment_draws"]}')
            if not _close(row['replays_per_segment'], _ratio(draws, expect_admitted)):
                problems.append(
                    f'{row["cond"]} seed {seed} replays={row["replays_per_segment"]}')
            if not _close(row['draws_per_m_segment'], _ratio(draws, expect_seg)):
                problems.append(
                    f'{row["cond"]} seed {seed} draws/M={row["draws_per_m_segment"]}')
            if spec['source'] != 'none' and not _close(row['traffic'], TRAFFIC[row['cond']]):
                problems.append(f'{row["cond"]} seed {seed} traffic label')
            if not math.isfinite(row['W']) or not (0.0 < row['W'] < m4b.LOG2A + 0.5):
                problems.append(f'{row["cond"]} seed {seed} W={row["W"]}')
            if not math.isfinite(row['P']) or not (0.0 < row['P'] < 8.0):
                problems.append(f'{row["cond"]} seed {seed} P={row["P"]}')
        if glances <= 77 and {'res77_e4', 'res308_e4'} <= found:
            a = next(r for r in rows if r['cond'] == 'res77_e4')
            b = next(r for r in rows if r['cond'] == 'res308_e4')
            if abs(a['W'] - b['W']) > 1e-8 or a['replay_steps'] != b['replay_steps']:
                problems.append(
                    f'seed {seed} T={glances} res77_e4 and res308_e4 diverged before fill')
            if a['glance_checksum'] != b['glance_checksum']:
                problems.append(f'seed {seed} bounded pair broke the glance log')
    return problems


def _pick(rows, seed, cond, key):
    for row in rows:
        if row['seed'] == seed and row['cond'] == cond:
            return row[key]
    raise KeyError((seed, cond, key))


def write_report(score, smoke, kernels, out, started, notes, match_info):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        '# H27b measurements', '',
        f'Started: {started}. Updated: {now()}.', '',
        'Kaggle CPU. 2 processes, torch.set_num_threads(2) each. '
        'Standard deviation: population (ddof=0). No verdict.', '',
        f'Chosen T: {SCORE_T}. Scoring seeds: {SEEDS}.',
        'Conditions: nostore (no replay); hist_e1/hist_e2/hist_e4 (whole history, '
        'one batch-8 step after every 1/2/4 glances); res77_e4/res77_e8 (Algorithm R, '
        '77 segments, every 4/8); res308_e1/res308_e4 (Algorithm R, 308 segments, every 1/4).',
        'W is the mean held-out bits over the 8 learnable channels. '
        'P is the mean live bits/symbol over every glance, taken before that glance\'s replay. '
        'M is retained replay segments × 64 × log2(42); hist* charges every observed glance; '
        'res* charges final occupancy; nostore charges 0. '
        'Traffic is replayed symbols / observed symbols.',
        'Replays per segment is total segment-draws / sampling units ever admitted, over the life. '
        'A history unit is a valid overlapping 64-symbol window. A reservoir unit is a glance '
        'Algorithm R admitted, including one later replaced. '
        'Draws per M-segment is total segment-draws / the segment count M charges.',
        '',
    ]
    header = ('| Condition | W | P | M | traffic | replays per segment | '
              'draws per M-segment | segments | ' + ' | '.join(m4b.LEARN)
              + ' | Wall seconds (sum) |')
    align = '|---|---:|---:|---:|---:|---:|---:|---:|' + '---:|' * len(m4b.LEARN) + '---:|'
    lines += [header, align]
    for cond in CONDS:
        rs = [r for r in score if r['cond'] == cond]
        if not rs:
            continue
        chans = ' | '.join(mean_std([r['held'][c] for r in rs]) for c in m4b.LEARN)
        lines.append(
            f'| {cond} | {mean_std([r["W"] for r in rs])} | {mean_std([r["P"] for r in rs])} | '
            f'{rs[0]["M_bits"]:.3f} | {rs[0]["traffic"]:.6f} | '
            f'{mean_std([r["replays_per_segment"] for r in rs])} | '
            f'{rs[0]["draws_per_m_segment"]:.6f} | {rs[0]["segments"]} | {chans} | '
            f'{sum(r["wall_seconds"] for r in rs):.3f} |')
    lines += ['', 'Per seed. W, P, traffic, M, replays per segment.', '',
              '| Seed | Condition | W | P | traffic | M | replays per segment | '
              'draws per M-segment | segments | admitted | windows | replay steps | Wall seconds |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in score:
        lines.append(
            f'| {row["seed"]} | {row["cond"]} | {_cell(row["W"])} | {_cell(row["P"])} | '
            f'{row["traffic"]:.6f} | {row["M_bits"]:.3f} | {row["replays_per_segment"]:.6f} | '
            f'{row["draws_per_m_segment"]:.6f} | {row["segments"]} | {row["segments_admitted"]} | '
            f'{row["valid_windows"]} | {row["replay_steps"]} | {row["wall_seconds"]:.3f} |')
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
                  '| Seed | W nostore | W res77_e4 | W res77_e8 | W res308_e1 | W res308_e4 | '
                  'min res | min res − nostore |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|']
        gaps = []
        for seed in seeds:
            nostore = _pick(score, seed, 'nostore', 'W')
            res_w = [_pick(score, seed, name, 'W') for name in RES_NAMES]
            gap = min(res_w) - nostore
            gaps.append(gap)
            lines.append(
                f'| {seed} | {_cell(nostore)} | '
                + ' | '.join(_cell(v) for v in res_w)
                + f' | {_cell(min(res_w))} | {_cell(gap)} |')
        lines.append(
            f'| mean | {mean_std([_pick(score, s, "nostore", "W") for s in seeds])} | '
            + ' | '.join(mean_std([_pick(score, s, name, 'W') for s in seeds]) for name in RES_NAMES)
            + f' | {mean_std([min(_pick(score, s, name, "W") for name in RES_NAMES) for s in seeds])} | '
            f'{mean_std(gaps)} |')
        lines += ['', '| Seed | W res308_e4 | W hist_e4 | res308_e4 − hist_e4 |',
                  '|---:|---:|---:|---:|']
        gaps2 = []
        for seed in seeds:
            bounded = _pick(score, seed, 'res308_e4', 'W')
            full = _pick(score, seed, 'hist_e4', 'W')
            gaps2.append(bounded - full)
            lines.append(
                f'| {seed} | {_cell(bounded)} | {_cell(full)} | {_cell(bounded - full)} |')
        lines.append(
            f'| mean | {mean_std([_pick(score, s, "res308_e4", "W") for s in seeds])} | '
            f'{mean_std([_pick(score, s, "hist_e4", "W") for s in seeds])} | '
            f'{mean_std(gaps2)} |')
        lines += ['', 'Dose curve for hist_e1, hist_e2, hist_e4. '
                  'Bits saved = W(nostore) − W(hist). Per unit traffic divides that saving by traffic. '
                  'Report only.', '',
                  '| Seed | traffic hist_e1 | saved hist_e1 | saved/traffic hist_e1 | '
                  'traffic hist_e2 | saved hist_e2 | saved/traffic hist_e2 | '
                  'traffic hist_e4 | saved hist_e4 | saved/traffic hist_e4 |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        per = {name: [] for name in HIST_DOSE}
        for seed in seeds:
            nostore = _pick(score, seed, 'nostore', 'W')
            cells = []
            for name in HIST_DOSE:
                saved = nostore - _pick(score, seed, name, 'W')
                traffic = _pick(score, seed, name, 'traffic')
                per[name].append(saved / traffic)
                cells += [f'{traffic:.6f}', _cell(saved), _cell(saved / traffic)]
            lines.append(f'| {seed} | ' + ' | '.join(cells) + ' |')
        lines.append(
            '| mean | '
            + ' | '.join(
                f'{_pick(score, seeds[0], name, "traffic"):.6f} | '
                f'{mean_std([_pick(score, s, "nostore", "W") - _pick(score, s, name, "W") for s in seeds])} | '
                f'{mean_std(per[name])}'
                for name in HIST_DOSE)
            + ' |')
        lines += ['', 'P / M / traffic. Report only. P is the live bits/symbol defined above.', '',
                  '| Condition | P | M | traffic |',
                  '|---|---:|---:|---:|']
        for cond in CONDS:
            rs = [r for r in score if r['cond'] == cond]
            if not rs:
                continue
            lines.append(
                f'| {cond} | {mean_std([r["P"] for r in rs])} | {rs[0]["M_bits"]:.3f} | '
                f'{rs[0]["traffic"]:.6f} |')
    lines += ['', 'Smoke.', '']
    if smoke:
        lines += ['| Condition | W | P | M | traffic | replays per segment | segments | '
                  'admitted | replay steps | Wall seconds | Glance checksum |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in smoke:
            lines.append(
                f'| {row["cond"]} | {_cell(row["W"])} | {_cell(row["P"])} | {row["M_bits"]:.3f} | '
                f'{row["traffic"]:.6f} | {row["replays_per_segment"]:.6f} | {row["segments"]} | '
                f'{row["segments_admitted"]} | {row["replay_steps"]} | {row["wall_seconds"]:.3f} | '
                f'{row["glance_checksum"]} |')
    else:
        lines.append('No smoke rows.')
    lines += ['', 'Kaggle kernels.', '',
              '| Id | Phase | Seeds | T | Status | Elapsed seconds | Rerun | Note |',
              '|---|---|---|---:|---|---:|---|---|']
    for ker in kernels:
        lines.append(
            f'| {ker["id"]} | {ker["phase"]} | {ker["seeds"]} | {ker["glances"]} | {ker["status"]} | '
            f'{ker["elapsed_seconds"]} | {ker["rerun"]} | {ker["note"]} |')
    lines += ['', 'Protocol notes:',
              '- World: FactWorld, channels G C1 C7 E30 E110 M7 M37 F N Z. Glance = 64 symbols.',
              '- N: uniform over 42 symbols. Z: the space symbol repeated. Seeds 21 22 23 24 25 26.',
              '- GRU: symbol embedding 32, channel embedding 8, hidden 256, Adam lr 3e-3, clip 1.0.',
              '  Per-channel state. One live Adam step per glance, then one replay step every k glances.',
              '- Controller rand. No utility store and no match, so W is the frozen GRU on a fresh',
              '  2048-symbol continuation of each learnable channel.',
              '- hist* calls m4b.sample_history: uniform over starts 0 through L−64. hist_e1 is that',
              '  machine at one step per glance, the same schedule as H27 hist1.',
              '- res* is Algorithm R over the glance segments seen so far. 77 is H27\'s 15% of the',
              '  T = 512 log. 308 is the specified 60% capacity (308/512 = 0.6015625).',
              '- Each replay step draws 8 segments with replacement. Traffic = 8/k.',
              '- M = segments × 64 × log2(42). hist* counts every observed glance. nostore counts 0.',
              '  res* counts the final occupancy, not every admission.',
              '- Replays per segment = (replay steps × 8) / units ever admitted. History\'s denominator',
              '  is the final overlapping-window count, 64×T − 630 at T ≥ 10. The reservoir\'s',
              '  denominator counts every admitted glance, including those later replaced.',
              '- Draws per M-segment = (replay steps × 8) / segments charged by M.',
              '- P is the mean live bits/symbol on all 10 channels, before that glance\'s replay.',
              '- Reservoir replacement uses numpy seed 40_000 + seed. Replay draws use 30_000 + seed.',
              '  Choice and observations use 10_000 + seed, shared by every condition.',
              '- First pass: glance g visits channel g for g < 10, then the controller.',
              '- The raw log stays in the run so glances can be indexed. For res* it is not the',
              '  replay source; M does not charge it.']
    if match_info:
        lines += ['', 'Local match, before the scored kernels. Not a score.',
                  f'- Seed {match_info["seed"]}, T = {match_info["glances"]}. '
                  f'hist_e1 W − h27 hist1 W = {match_info["hist_delta"]:.3e}. '
                  f'nostore W − h27 nostore W = {match_info["nostore_delta"]:.3e}. '
                  f'res77_e4 W − res308_e4 W = {match_info["bounded_delta"]:.3e}.',
                  f'- Glance checksum {match_info["glance_checksum"]} shared by every condition '
                  f'and by the H27 nostore and hist1 runs.']
    if notes:
        lines += ['', 'Notes:'] + [f'- {note}' for note in notes]
    complete = len(score) == len(SEEDS) * len(CONDS) and not notes
    lines += ['', f'Score runs: {len(score)} of {len(SEEDS) * len(CONDS)}.',
              'Status: ' + ('complete' if complete else 'incomplete') + '.', '']
    text = '\n'.join(lines)
    (out / 'RESULTS.md').write_text(text)
    payload = {
        'started': started, 'updated': now(), 'glances': SCORE_T, 'seeds': SEEDS,
        'conditions': CONDS, 'spec': SPEC, 'traffic': TRAFFIC,
        'runs': score, 'smoke_runs': smoke, 'kernels': kernels,
        'match': match_info, 'notes': notes,
        'definitions': {
            'W': 'mean held-out bits over the 8 learnable channels',
            'P': 'mean live bits/symbol over every glance, before that glance\'s replay',
            'M': 'retained segments * 64 * log2(42); hist* is the glance log; res* is final occupancy',
            'traffic': 'replayed symbols / observed symbols',
            'replays_per_segment': 'segment draws / sampling units ever admitted',
            'draws_per_m_segment': 'segment draws / segments charged by M',
        },
    }
    (out / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return 'complete' if complete else 'incomplete'


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
        elif row.get('phase') == 'smoke' and row.get('glances') == SMOKE_T:
            smoke.append(row)
    score.sort(key=lambda r: (r['seed'], CONDS.index(r['cond'])))
    smoke.sort(key=lambda r: (r['seed'], CONDS.index(r['cond'])))
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


def _same_bank(left, right):
    return left.seen == right.seen and [(a, b) for a, b in left.items] == [
        (a, b) for a, b in right.items]


def checks():
    for glances, every, expect in ((512, 1, 512), (512, 2, 256), (512, 4, 128), (512, 8, 64),
                                   (32, 4, 8), (32, 8, 4)):
        got = sum(1 for g in range(glances) if (g + 1) % every == 0)
        assert got == expect == glances // every
    assert expect_windows(8) == 8
    assert expect_windows(10) == 10
    assert expect_windows(32) == 64 * 32 - 630
    assert expect_windows(512) == 64 * 512 - 630
    uneven = np.array([23, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert int(uneven.sum()) == 32
    assert valid_windows(uneven * m4b.B) == expect_windows(32)
    spread = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    assert valid_windows(spread * m4b.B) == expect_windows(8)
    for seed in (0, 1, 21):
        for capacity in (4, 77, 308):
            host = Reservoir(capacity, np.random.default_rng(seed))
            mine = TrackedReservoir(capacity, np.random.default_rng(seed))
            for item in range(600):
                host.add(('c', item))
                mine.add(('c', item))
            if not _same_bank(host, mine):
                raise AssertionError(f'reservoir diverged seed={seed} k={capacity}')
            if capacity <= 600 and mine.admitted < capacity:
                raise AssertionError('admission count below occupancy')
            if len(mine.items) != min(capacity, 600):
                raise AssertionError('occupancy')
    # The subclass detector must agree with the direct counter. A missed RNG
    # step would desynchronise the two banks above; this checks the counter.
    direct = TrackedReservoir(5, np.random.default_rng(3))
    for item in range(40):
        direct.add(item)
    assert len(direct.items) == 5 and direct.seen == 40 and direct.admitted >= 5
    assert memory_bits(0) == 0.0
    assert memory_bits(512) == 512 * m4b.B * m4b.LOG2A
    assert memory_bits(77) == 77 * m4b.B * m4b.LOG2A
    assert memory_bits(308) == 308 * m4b.B * m4b.LOG2A
    hist = torch.zeros((len(m4b.CH), 128), dtype=torch.long)
    hist[1, :64] = torch.arange(64)
    previous, targets, cis = sample_bank(hist, [(1, 0)], np.random.default_rng(0))
    assert previous.shape == (m4b.B, BATCH) and int(previous[0, 0]) == -1
    assert torch.equal(targets[:, 0], hist[1, :64])
    print(f'{now()} h27b checks passed', flush=True)


def _compare_pair(mine, theirs, label):
    if mine['glance_checksum'] != theirs['glance_checksum']:
        raise SystemExit(f'{label} glance checksum diverged')
    if mine['observation_checksum'] != theirs['observation_checksum']:
        raise SystemExit(f'{label} observation checksum diverged')
    if mine['replay_steps'] != theirs['replay_steps']:
        raise SystemExit(
            f'{label} replay_steps {mine["replay_steps"]} != {theirs["replay_steps"]}')
    if abs(mine['W'] - theirs['W']) > 1e-8:
        raise SystemExit(f'{label} W {mine["W"]:.8f} != {theirs["W"]:.8f}')
    for channel in m4b.LEARN:
        if abs(mine['held'][channel] - theirs['held'][channel]) > 1e-8:
            raise SystemExit(
                f'{label} {channel} {mine["held"][channel]} != {theirs["held"][channel]}')


def match(glances=8, seed=21, out=None):
    """hist_e1 reproduces H27 hist1. Bounded banks share the log."""
    configure_threads()
    _check_divisible(glances)
    pairs = [('nostore', 'nostore'), ('hist_e1', 'hist1')]
    deltas = {}
    ran = {}
    base_checksum = None
    for cond, host in pairs:
        mine = run(cond, seed, glances, 'match')
        theirs = h27.run(host, seed, glances, 'match')
        _compare_pair(mine, theirs, f'{cond} vs h27 {host}')
        deltas[cond] = mine['W'] - theirs['W']
        ran[cond] = mine
        base_checksum = mine['glance_checksum']
        print(f'{now()} match {cond} == h27 {host} W={mine["W"]:.6f}', flush=True)
    base = ran['nostore']
    for cond in CONDS:
        if cond in ran:
            other = ran[cond]
        else:
            other = run(cond, seed, glances, 'match')
            ran[cond] = other
        if other['glance_checksum'] != base['glance_checksum']:
            raise SystemExit(f'{cond} broke the shared glance log')
        if other['observation_checksum'] != base['observation_checksum']:
            raise SystemExit(f'{cond} broke the shared observation log')
        if other['valid_windows'] != expect_windows(glances):
            raise SystemExit(f'{cond} windows {other["valid_windows"]}')
        spec = SPEC[cond]
        if spec['source'] == 'none':
            expect_steps = 0
        else:
            expect_steps = glances // spec['every']
        if other['replay_steps'] != expect_steps:
            raise SystemExit(f'{cond} replay_steps {other["replay_steps"]}')
        if not _close(other['traffic'], TRAFFIC[cond]):
            raise SystemExit(f'{cond} traffic {other["traffic"]}')
        print(f'{now()} match {cond} log shared W={other["W"]:.6f} '
              f'replays={other["replays_per_segment"]:.4f} traffic={other["traffic"]:.4f}',
              flush=True)
    if glances <= 77:
        delta = ran['res77_e4']['W'] - ran['res308_e4']['W']
        if abs(delta) > 1e-8:
            raise SystemExit(f'res77_e4 and res308_e4 diverged by {delta}')
        if ran['res77_e4']['replay_steps'] != ran['res308_e4']['replay_steps']:
            raise SystemExit('bounded pair replay_steps diverged')
    else:
        delta = float('nan')
    info = {
        'seed': int(seed), 'glances': int(glances),
        'hist_delta': float(deltas['hist_e1']),
        'nostore_delta': float(deltas['nostore']),
        'bounded_delta': float(delta),
        'glance_checksum': int(base_checksum),
    }
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'match.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'{now()} h27b match passed', flush=True)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--match', action='store_true')
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--out', default='results/h27b')
    parser.add_argument('--match-glances', type=int, default=8)
    parser.add_argument('--match-seed', type=int, default=21)
    args = parser.parse_args()
    if args.checks:
        checks()
    if args.match:
        match(args.match_glances, args.match_seed, args.out)
    if args.assemble:
        status = assemble(args.out)
        print(f'{now()} assemble {status}', flush=True)


if __name__ == '__main__':
    main()
