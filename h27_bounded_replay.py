"""H27: does replay need the whole history, or a fair sample of it?

Protocol: handoffs/2026-11-10_grok_h27_bounded_replay.md. Numbers only; no verdict.
Creates only this file, kaggle/h27/* and results/h27/*. Imports earlier scripts;
does not edit them.

Readings of the spec, fixed before the scored runs:
- World, GRU, live step and held-out W are M4c hist_rep: controller rand, no
  utility store, no match. T = 512. Seeds 15 16 17 18 19 20. Glance = 64.
- A replay segment is 64 symbols of one channel, from a zero hidden state, with
  that channel's embedding. hist* draws those segments exactly as M4b
  sample_history (every valid start in the raw log). Dose is Adam steps per
  glance, batch 8. hist2 is M4b hist_rep.
- res* is Vitter's Algorithm R over glance-aligned segments (one segment per
  glance, not every overlapping start). fifo77 keeps the most recent glances.
  A replay step draws 8 segments uniformly with replacement from the current
  contents. The glance just observed is inserted before that glance's replay.
- M is retained segments × 64 × log2(42). hist* charges the whole log. nostore
  charges nothing. The run keeps the raw log for the live step, for checksums,
  and so a bounded bank can index its glances; that log is not the replay
  source for res* or fifo77.
- The first replay input is the true preceding symbol, or a zero embedding at
  channel position 0, the same rule as m4b.sample_history.
- Choice and observations use 10_000 + seed. Replay draws use 30_000 + seed.
  Algorithm R's replacement index uses 40_000 + seed. torch.manual_seed(seed)
  initialises the GRU. FactWorld(seed).
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

CONDS = ['nostore', 'hist1', 'hist2', 'hist4', 'res77', 'res16', 'fifo77']
SEEDS = [15, 16, 17, 18, 19, 20]
SCORE_T = 512
BATCH = 8
SPEC = {
    'nostore': {'source': 'none', 'dose': 0, 'capacity': 0},
    'hist1': {'source': 'history', 'dose': 1, 'capacity': None},
    'hist2': {'source': 'history', 'dose': 2, 'capacity': None},
    'hist4': {'source': 'history', 'dose': 4, 'capacity': None},
    'res77': {'source': 'reservoir', 'dose': 2, 'capacity': 77},
    'res16': {'source': 'reservoir', 'dose': 2, 'capacity': 16},
    'fifo77': {'source': 'fifo', 'dose': 2, 'capacity': 77},
}
assert list(SPEC) == CONDS
assert abs(0.15 * 512 - 76.8) < 1e-9
assert SPEC['res77']['capacity'] == 77 and SPEC['res16']['capacity'] == 16
assert SPEC['fifo77']['capacity'] == 77 and SPEC['hist2']['dose'] == 2
assert m4b.B == 64 and m4b.A == 42 and len(m4b.LEARN) == 8


class Reservoir:
    """Algorithm R. Capacity k; the n-th glance replaces a slot with probability k/n."""

    def __init__(self, capacity, rng):
        self.capacity = int(capacity)
        self.rng = rng
        self.items = []
        self.seen = 0

    def add(self, item):
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        j = int(self.rng.integers(0, self.seen))
        if j < self.capacity:
            self.items[j] = item


class Fifo:
    """The most recent `capacity` glance-segments, in arrival order."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.items = []

    def add(self, item):
        self.items.append(item)
        if len(self.items) > self.capacity:
            del self.items[0]


def sample_bank(hist, items, rng):
    """Eight uniform draws from the bank. Previous-symbol rule matches sample_history."""
    n = len(items)
    if n <= 0:
        return None
    picks = rng.integers(0, n, size=BATCH)
    channels = np.array([items[int(i)][0] for i in picks], dtype=np.int64)
    starts = np.array([items[int(i)][1] for i in picks], dtype=np.int64)
    starts_t = torch.as_tensor(starts, dtype=torch.long)
    cis = torch.as_tensor(np.ascontiguousarray(channels), dtype=torch.long)
    indices = starts_t[None, :] + torch.arange(m4b.B, dtype=torch.long)[:, None]
    targets = hist[cis[None, :], indices]
    previous = hist[cis[None, :], (indices - 1).clamp_min(0)].clone()
    previous[indices == 0] = -1
    return previous, targets, cis


def retained_segments(source, bank, observed_symbols):
    if source == 'none':
        return 0
    if source == 'history':
        return int(observed_symbols // m4b.B)
    return len(bank.items)


def memory_bits(segments):
    return float(segments) * m4b.B * m4b.LOG2A


def run(cond, seed, glances, phase):
    """One rand life. Replay source and dose are the only knobs."""
    if cond not in SPEC:
        raise ValueError(cond)
    spec = SPEC[cond]
    source, dose, capacity = spec['source'], spec['dose'], spec['capacity']
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
        bank = Reservoir(capacity, np.random.default_rng(40_000 + seed))
    elif source == 'fifo':
        bank = Fifo(capacity)
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
        for _ in range(dose):
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
            segs = retained_segments(source, bank, (g + 1) * m4b.B)
            print(f'{now()} {phase} {cond} s{seed} T={g + 1}/{glances} '
                  f'bits={np.mean(life):.4f} segments={segs} '
                  f'replay={replay_steps} wall={time.monotonic() - started:.1f}s',
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
    segments = retained_segments(source, bank, n_obs)
    replayed = int(replay_steps * BATCH * m4b.B)
    assert counts.sum() == glances and np.all(lengths == counts * m4b.B)
    assert replay_steps == dose * glances
    assert np.isfinite(life).all() and np.isfinite(list(held.values())).all()
    if source == 'none':
        assert segments == 0 and replay_steps == 0
    elif source == 'history':
        assert segments == glances
    else:
        assert segments == min(capacity, glances)
        assert bank.seen == glances if source == 'reservoir' else len(bank.items) == segments
    result = dict(
        phase=phase, cond=cond, seed=int(seed), glances=int(glances),
        W=float(np.mean([held[c] for c in m4b.LEARN])),
        held={c: float(held[c]) for c in m4b.LEARN},
        M_bits=memory_bits(segments),
        segments=int(segments),
        capacity=None if capacity is None else int(capacity),
        traffic=float(replayed / n_obs),
        replayed_symbols=replayed,
        observed_symbols=int(n_obs),
        replay_steps=int(replay_steps),
        dose=int(dose),
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
          f'M={result["M_bits"]:.3f} traffic={result["traffic"]:.4f} '
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
            'M_bits': result['M_bits'], 'traffic': result['traffic'],
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
        found = {r['cond'] for r in rows}
        if found != set(CONDS):
            problems.append(f'seed {seed} conditions {sorted(found)}')
        for row in rows:
            spec = SPEC[row['cond']]
            glances = row['glances']
            if row['replay_steps'] != spec['dose'] * glances:
                problems.append(f'{row["cond"]} seed {seed} replay_steps={row["replay_steps"]}')
            if spec['source'] == 'none':
                expect_seg = 0
            elif spec['source'] == 'history':
                expect_seg = glances
            else:
                expect_seg = min(spec['capacity'], glances)
            if row['segments'] != expect_seg:
                problems.append(f'{row["cond"]} seed {seed} segments={row["segments"]}')
            if abs(row['M_bits'] - memory_bits(expect_seg)) > 1e-6:
                problems.append(f'{row["cond"]} seed {seed} M={row["M_bits"]}')
            expect_traffic = spec['dose'] * BATCH
            if abs(row['traffic'] - expect_traffic) > 1e-9:
                problems.append(f'{row["cond"]} seed {seed} traffic={row["traffic"]}')
            if not math.isfinite(row['W']) or not (0.0 < row['W'] < m4b.LOG2A + 0.5):
                problems.append(f'{row["cond"]} seed {seed} W={row["W"]}')
    return problems


def _pick(rows, seed, cond, key):
    for row in rows:
        if row['seed'] == seed and row['cond'] == cond:
            return row[key]
    raise KeyError((seed, cond, key))


def write_report(score, smoke, kernels, out, started, notes):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        '# H27 measurements', '',
        f'Started: {started}. Updated: {now()}.', '',
        'Kaggle CPU. 2 processes, torch.set_num_threads(2) each. '
        'Standard deviation: population (ddof=0). No verdict.', '',
        f'Chosen T: {SCORE_T}. Scoring seeds: {SEEDS}.',
        'Conditions: nostore (no replay); hist1/hist2/hist4 (whole-history replay, '
        '1/2/4 steps per glance, batch 8); res77 and res16 (Algorithm R, capacities '
        '77 and 16, 2 steps); fifo77 (most recent 77 glances, 2 steps).',
        'W is the mean held-out bits over the 8 learnable channels. '
        'M is retained replay segments × 64 × log2(42); hist* is the whole log. '
        'Traffic is replayed symbols / observed symbols.', '',
    ]
    header = ('| Condition | W | M | traffic | segments | ' + ' | '.join(m4b.LEARN)
              + ' | Wall seconds (sum) |')
    align = '|---|---:|---:|---:|---:|' + '---:|' * len(m4b.LEARN) + '---:|'
    lines += [header, align]
    for cond in CONDS:
        rs = [r for r in score if r['cond'] == cond]
        if not rs:
            continue
        chans = ' | '.join(mean_std([r['held'][c] for r in rs]) for c in m4b.LEARN)
        lines.append(
            f'| {cond} | {mean_std([r["W"] for r in rs])} | {rs[0]["M_bits"]:.3f} | '
            f'{rs[0]["traffic"]:.6f} | {rs[0]["segments"]} | {chans} | '
            f'{sum(r["wall_seconds"] for r in rs):.3f} |')
    lines += ['', 'Per seed. W, traffic and M.', '',
              '| Seed | Condition | W | traffic | M | segments | replay steps | Wall seconds |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in score:
        lines.append(
            f'| {row["seed"]} | {row["cond"]} | {_cell(row["W"])} | {row["traffic"]:.6f} | '
            f'{row["M_bits"]:.3f} | {row["segments"]} | {row["replay_steps"]} | '
            f'{row["wall_seconds"]:.3f} |')
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
                  '| Seed | W res77 | W hist2 | res77 − hist2 | W fifo77 | res77 − fifo77 | W res16 | W nostore |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|']
        diffs_h, diffs_f = [], []
        for seed in seeds:
            res77 = _pick(score, seed, 'res77', 'W')
            hist2 = _pick(score, seed, 'hist2', 'W')
            fifo = _pick(score, seed, 'fifo77', 'W')
            res16 = _pick(score, seed, 'res16', 'W')
            nostore = _pick(score, seed, 'nostore', 'W')
            diffs_h.append(res77 - hist2)
            diffs_f.append(res77 - fifo)
            lines.append(
                f'| {seed} | {_cell(res77)} | {_cell(hist2)} | {_cell(res77 - hist2)} | '
                f'{_cell(fifo)} | {_cell(res77 - fifo)} | {_cell(res16)} | {_cell(nostore)} |')
        lines.append(
            f'| mean | {mean_std([_pick(score, s, "res77", "W") for s in seeds])} | '
            f'{mean_std([_pick(score, s, "hist2", "W") for s in seeds])} | '
            f'{mean_std(diffs_h)} | '
            f'{mean_std([_pick(score, s, "fifo77", "W") for s in seeds])} | '
            f'{mean_std(diffs_f)} | '
            f'{mean_std([_pick(score, s, "res16", "W") for s in seeds])} | '
            f'{mean_std([_pick(score, s, "nostore", "W") for s in seeds])} |')
        lines += ['', 'Dose curve. Bits saved = W(nostore) − W(hist). '
                  'Per unit traffic divides that saving by traffic. Report only.', '',
                  '| Seed | traffic hist1 | saved hist1 | saved/traffic hist1 | '
                  'traffic hist2 | saved hist2 | saved/traffic hist2 | '
                  'traffic hist4 | saved hist4 | saved/traffic hist4 |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        per = {name: [] for name in ('hist1', 'hist2', 'hist4')}
        for seed in seeds:
            nostore = _pick(score, seed, 'nostore', 'W')
            cells = []
            for name in ('hist1', 'hist2', 'hist4'):
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
                for name in ('hist1', 'hist2', 'hist4'))
            + ' |')
    lines += ['', 'Smoke.', '']
    if smoke:
        lines += ['| Condition | W | M | traffic | segments | replay steps | Wall seconds | Glance checksum |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in smoke:
            lines.append(
                f'| {row["cond"]} | {_cell(row["W"])} | {row["M_bits"]:.3f} | {row["traffic"]:.6f} | '
                f'{row["segments"]} | {row["replay_steps"]} | {row["wall_seconds"]:.3f} | '
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
              '- N: uniform over 42 symbols. Z: the space symbol repeated. Seeds 15 16 17 18 19 20.',
              '- GRU: symbol embedding 32, channel embedding 8, hidden 256, Adam lr 3e-3, clip 1.0.',
              '  Per-channel state. One live Adam step per glance, then `dose` replay steps.',
              '- Controller rand. No utility store and no match, so W is the frozen GRU on a fresh',
              '  2048-symbol continuation of each learnable channel.',
              '- hist* calls m4b.sample_history: uniform over starts 0 through L−64 on every channel',
              '  with L ≥ 64. hist2 is that machine at dose 2, the same dose as M4c hist_rep.',
              '- res* is Algorithm R over the glance segments seen so far, capacity fixed at 77 or 16',
              '  (77 = 15% of the T = 512 log; 16 ≈ 3%). fifo77 drops the oldest glance past 77.',
              '- Each replay step draws 8 segments with replacement. Traffic = dose × 8.',
              '- M = segments × 64 × log2(42). hist* counts every observed glance. nostore counts 0.',
              '- Reservoir replacement uses numpy seed 40_000 + seed. Replay draws use 30_000 + seed.',
              '  Choice and observations use 10_000 + seed, shared by every condition.',
              '- First pass: glance g visits channel g for g < 10, then the controller.',
              '- The raw log stays in the run so glances can be indexed. For res* and fifo77 it is',
              '  not the replay source; M does not charge it.']
    if notes:
        lines += ['', 'Notes:'] + [f'- {note}' for note in notes]
    lines += ['', f'Score runs: {len(score)} of {len(SEEDS) * len(CONDS)}.',
              'Status: ' + ('complete' if len(score) == len(SEEDS) * len(CONDS) and not notes else 'incomplete') + '.',
              '']
    text = '\n'.join(lines)
    (out / 'RESULTS.md').write_text(text)
    payload = {
        'started': started, 'updated': now(), 'glances': SCORE_T, 'seeds': SEEDS,
        'conditions': CONDS, 'spec': SPEC, 'runs': score, 'smoke_runs': smoke,
        'kernels': kernels, 'notes': notes,
    }
    (out / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return 'complete' if len(score) == len(SEEDS) * len(CONDS) and not notes else 'incomplete'


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
        elif row.get('phase') == 'smoke':
            smoke.append(row)
    score.sort(key=lambda r: (r['seed'], CONDS.index(r['cond'])))
    smoke.sort(key=lambda r: CONDS.index(r['cond']))
    ker_path = root / 'kernels.json'
    kernels = json.loads(ker_path.read_text()).get('kernels', []) if ker_path.exists() else []
    meta_path = root / 'meta.json'
    started = json.loads(meta_path.read_text())['started'] if meta_path.exists() else now()
    notes = []
    if score:
        notes.extend(_problems(score))
    if smoke:
        notes.extend('smoke: ' + item for item in _problems(smoke))
    return write_report(score, smoke, kernels, root, started, notes)


def checks():
    rng = np.random.default_rng(0)
    fifo = Fifo(3)
    for item in range(5):
        fifo.add(item)
    assert fifo.items == [2, 3, 4], fifo.items
    # Algorithm R keeps k distinct items and, once full, stays full.
    bank = Reservoir(4, np.random.default_rng(1))
    for item in range(20):
        bank.add(('c', item))
    assert len(bank.items) == 4 and bank.seen == 20
    assert len({item[1] for item in bank.items}) == 4
    # Inclusion frequencies track k/n. Algorithm R on a stream of n distinct items.
    k, n, trials = 10, 40, 2000
    hits = np.zeros(n)
    for trial in range(trials):
        trial_bank = Reservoir(k, np.random.default_rng(10_000 + trial))
        for item in range(n):
            trial_bank.add(item)
        for item in trial_bank.items:
            hits[item] += 1
    freq = hits / trials
    expect = k / n
    if np.max(np.abs(freq - expect)) > 0.04:
        raise AssertionError(f'reservoir frequencies off: max {np.max(np.abs(freq - expect)):.4f}')
    hist = torch.zeros((len(m4b.CH), 128), dtype=torch.long)
    hist[1, :64] = torch.arange(64)
    hist[2, :96] = torch.arange(100, 196)
    for item, start0 in (((1, 0), True), ((2, 32), False)):
        previous, targets, cis = sample_bank(hist, [item], np.random.default_rng(0))
        assert previous.shape == (m4b.B, BATCH) and targets.shape == (m4b.B, BATCH)
        assert torch.equal(cis, torch.full((BATCH,), item[0]))
        assert torch.equal(targets[:, 0], hist[item[0], item[1]:item[1] + m4b.B])
        if start0:
            assert int(previous[0, 0]) == -1
        else:
            assert int(previous[0, 0]) == int(hist[item[0], item[1] - 1])
    lengths = np.array([64] * len(m4b.CH))
    wide = torch.arange(len(m4b.CH) * 64).reshape(len(m4b.CH), 64)
    prev, tgt, cis = m4b.sample_history(wide, lengths, np.random.default_rng(0))
    assert prev.shape == (64, 8) and tgt.shape == (64, 8)
    assert memory_bits(512) == 512 * 64 * m4b.LOG2A
    assert memory_bits(0) == 0.0
    print(f'{now()} h27 checks passed', flush=True)


def match(glances=12, seed=15):
    """hist2 and nostore must reproduce m4b.run. Bounded banks must share the log."""
    configure_threads()
    pairs = [('nostore', 'nostore'), ('hist2', 'hist_rep')]
    for cond, host in pairs:
        mine = run(cond, seed, glances, 'match')
        theirs, _, _ = m4b.run(host, seed, glances, 'match')
        if mine['glance_checksum'] != theirs['glance_checksum']:
            raise SystemExit(f'{cond} glance checksum diverged from m4b {host}')
        if mine['observation_checksum'] != theirs['observation_checksum']:
            raise SystemExit(f'{cond} observation checksum diverged from m4b {host}')
        if mine['replay_steps'] != theirs['replay_steps']:
            raise SystemExit(
                f'{cond} replay_steps {mine["replay_steps"]} != m4b {theirs["replay_steps"]}')
        if abs(mine['W'] - theirs['W_mix']) > 1e-8:
            raise SystemExit(f'{cond} W {mine["W"]:.8f} != m4b W_mix {theirs["W_mix"]:.8f}')
        for channel in m4b.LEARN:
            if abs(mine['held'][channel] - theirs['held_gru'][channel]) > 1e-8:
                raise SystemExit(
                    f'{cond} {channel} {mine["held"][channel]} != {theirs["held_gru"][channel]}')
        print(f'{now()} match {cond} == m4b {host} W={mine["W"]:.6f}', flush=True)
    base = run('nostore', seed, glances, 'match')
    for cond in ('res16', 'fifo77', 'hist1', 'hist4'):
        other = run(cond, seed, glances, 'match')
        if other['glance_checksum'] != base['glance_checksum']:
            raise SystemExit(f'{cond} broke the shared glance log')
        if other['observation_checksum'] != base['observation_checksum']:
            raise SystemExit(f'{cond} broke the shared observation log')
        spec = SPEC[cond]
        expect = 0 if spec['source'] == 'none' else (
            glances if spec['source'] == 'history' else min(spec['capacity'], glances))
        if other['segments'] != expect or other['replay_steps'] != spec['dose'] * glances:
            raise SystemExit(f'{cond} segments={other["segments"]} replay={other["replay_steps"]}')
        print(f'{now()} match {cond} log shared W={other["W"]:.6f} '
              f'segments={other["segments"]} traffic={other["traffic"]:.4f}', flush=True)
    print(f'{now()} h27 match passed', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--match', action='store_true')
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--out', default='results/h27')
    args = parser.parse_args()
    if args.checks:
        checks()
    if args.match:
        match()
    if args.assemble:
        status = assemble(args.out)
        print(f'{now()} assemble {status}', flush=True)


if __name__ == '__main__':
    main()
