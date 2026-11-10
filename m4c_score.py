"""M4c: score the M4b machine on fresh seeds, on Kaggle.

Protocol: handoffs/2026-11-10_grok_m4c_kaggle.md. Numbers only; no verdict.
Creates only this file, kaggle/m4c/* and results/m4c/*. Imports earlier
scripts; does not edit them.

Readings of the spec, fixed before the scored runs:
- The machine is M4b's, including the hard budget and whole-history replay.
  T = 512. Seeds 9 10 11 12 13 14. World, cortex, mixture and W are M4b's.
- Four conditions are M4b runs renamed: nostore, hist_rep, rand_full
  (store_hist), gmc_full (gmc_hist).
- Two new combinations only change the controller and whether the store
  exists. rand8_full is store_hist with M4a's rand8 choice (uniform on the
  8 learnable channels after the first pass). gmc_nostore is gmc_hist's
  controller and history replay with the store and the match turned off,
  so W_mix = W_gru.
- Both go through m4b.run. A host name makes m4b's existing branches fire;
  has_store or choose is swapped only for that one call.
- Score fields match M4b. Bars are not scored here.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import torch

import m4b_ablation as m4b
from h21a_substrate import mean_std, now

CONDS = ['nostore', 'hist_rep', 'rand_full', 'rand8_full', 'gmc_nostore', 'gmc_full']
STORE_CONDS = ['rand_full', 'rand8_full', 'gmc_full']
RAND_CONDS = ['nostore', 'hist_rep', 'rand_full']
DIRECT = {
    'nostore': 'nostore',
    'hist_rep': 'hist_rep',
    'rand_full': 'store_hist',
    'gmc_full': 'gmc_hist',
}
CONTROLLER = {
    'nostore': 'rand',
    'hist_rep': 'rand',
    'rand_full': 'rand',
    'rand8_full': 'rand8',
    'gmc_nostore': 'gmc_prop',
    'gmc_full': 'gmc_prop',
}
SEEDS = [9, 10, 11, 12, 13, 14]
SCORE_T = 512


class Hosted(str):
    """M4c name that also compares equal to one M4b condition.

    m4b.run branches on `cond == 'gmc_hist'` and `cond in CONDS`. Equality
    with the host reuses those branches. str(self) stays the M4c name.
    Equality is not transitive (the host string does not equal the M4c
    name); do not use Hosted as a dict key.
    """

    def __new__(cls, name, host):
        obj = str.__new__(cls, name)
        obj.host = host
        return obj

    def __eq__(self, other):
        if isinstance(other, str):
            return str.__eq__(self, other) or other == self.host
        return False

    def __hash__(self):
        return hash(str(self))


class _patch:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn
        self.old = None

    def __enter__(self):
        self.old = getattr(m4b, self.name)
        setattr(m4b, self.name, self.fn)
        return self

    def __exit__(self, *args):
        setattr(m4b, self.name, self.old)


_ORIG_HAS_STORE = m4b.has_store
_ORIG_CHOOSE = m4b.choose


def _has_store(cond):
    if str(cond) == 'gmc_nostore':
        return False
    return _ORIG_HAS_STORE(cond)


def _choose(cond, g, values, rng):
    if str(cond) == 'rand8_full':
        if g < len(m4b.CH):
            return g
        return int(rng.integers(len(m4b.LEARN)))
    return _ORIG_CHOOSE(cond, g, values, rng)


def configure_threads():
    m4b.configure_threads()


def run_one(cond, seed, glances, phase):
    """One M4b life. The two new conditions patch one dispatch each."""
    if cond not in CONDS:
        raise ValueError(cond)
    if cond in DIRECT:
        result, choices, bits = m4b.run(DIRECT[cond], seed, glances, phase)
        host = DIRECT[cond]
    elif cond == 'gmc_nostore':
        with _patch('has_store', _has_store):
            result, choices, bits = m4b.run(Hosted(cond, 'gmc_hist'), seed, glances, phase)
        host = 'gmc_hist'
    elif cond == 'rand8_full':
        with _patch('choose', _choose):
            result, choices, bits = m4b.run(Hosted(cond, 'store_hist'), seed, glances, phase)
        host = 'store_hist'
    else:
        raise ValueError(cond)
    result = dict(result)
    result['cond'] = cond
    result['m4b_host'] = host
    result['controller'] = CONTROLLER[cond]
    result['torch_version'] = torch.__version__
    result['numpy_version'] = np.__version__
    expect_store = cond in STORE_CONDS
    if result['replay_source'] != ('none' if cond == 'nostore' else 'history'):
        raise AssertionError(f'{cond} replay_source={result["replay_source"]}')
    if expect_store:
        if not result['hard_budget_ok']:
            raise AssertionError(f'{cond} seed {seed} broke the hard budget')
    else:
        if result['peak_M_bits'] != 0.0 or result['W_mix'] != result['W_gru']:
            raise AssertionError(f'{cond} seed {seed} stored or mixed without a store')
    expect_replay = 0 if cond == 'nostore' else 2 * glances
    if result['replay_steps'] != expect_replay:
        raise AssertionError(f'{cond} replay_steps={result["replay_steps"]} expected {expect_replay}')
    return result, choices, bits


def _worker(job):
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    configure_threads()
    result, _, _ = run_one(job['cond'], job['seed'], job['glances'], job['phase'])
    path = Path(job['out']) / (
        f'run_{job["phase"]}_{job["cond"]}_{job["seed"]}_{job["glances"]}.json')
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'WROTE {path.name} W_mix={result["W_mix"]:.6f} '
          f'W_gru={result["W_gru"]:.6f} wall={result["wall_seconds"]:.1f}', flush=True)
    return {'cond': result['cond'], 'seed': result['seed'], 'W_mix': result['W_mix'],
            'W_gru': result['W_gru'], 'wall_seconds': result['wall_seconds'],
            'glance_checksum': result['glance_checksum'],
            'observation_checksum': result['observation_checksum']}


def kernel_main(seeds, glances, phase, out, processes=2):
    """Kaggle entry. Two processes, two torch threads each. Writes one JSON per run."""
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


def family_ok(runs):
    """Shared rand logs match. rand8 and gmc diverge once the controller acts."""
    problems = []
    by = {}
    for row in runs:
        by.setdefault(row['seed'], []).append(row)
    for seed, rows in sorted(by.items()):
        rand = [r for r in rows if r['cond'] in RAND_CONDS]
        if len(rand) >= 2:
            glance = {r['glance_checksum'] for r in rand}
            observed = {r['observation_checksum'] for r in rand}
            if len(glance) != 1 or len(observed) != 1:
                problems.append(f'seed {seed} rand logs diverged')
        if rand and rand[0]['glances'] > len(m4b.CH):
            base = rand[0]['glance_checksum']
            for cond in ('rand8_full', 'gmc_nostore', 'gmc_full'):
                hit = [r for r in rows if r['cond'] == cond]
                if hit and hit[0]['glance_checksum'] == base:
                    problems.append(f'seed {seed} {cond} matched the rand glance log')
        for row in rows:
            if row['cond'] not in STORE_CONDS and row['W_mix'] != row['W_gru']:
                problems.append(f'seed {seed} {row["cond"]} mixed without a store')
            if not row.get('hard_budget_ok', False):
                problems.append(f'seed {seed} {row["cond"]} hard budget')
            for key in ('W_mix', 'W_gru', 'W_F_mix', 'W_F_gru'):
                if not np.isfinite(row[key]) or not (0.0 < row[key] < 20.0):
                    problems.append(f'seed {seed} {row["cond"]} {key}={row[key]}')
    return problems


def load_runs(folder):
    runs = []
    for path in sorted(Path(folder).rglob('run_*.json')):
        runs.append(json.loads(path.read_text()))
    return runs


def _cell(value):
    return m4b.cell(value)


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def write_report(score, smoke, det, kernels, out, started, notes):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shown = [c for c in CONDS if any(r['cond'] == c for r in score)]
    lines = ['# M4c measurements', '',
             f'Started: {started}. Updated: {now()}.', '',
             'Kaggle CPU. 2 processes, torch.set_num_threads(2) each. '
             'Standard deviation: population (ddof=0). No verdict.', '',
             f'Chosen T: {SCORE_T}. Scoring seeds: {SEEDS}.',
             'Conditions: nostore (rand, no store, no replay); '
             'hist_rep (rand, no store, history); '
             'rand_full (rand, store, history = M4b store_hist); '
             'rand8_full (rand8, store, history); '
             'gmc_nostore (gmc_prop, no store, history); '
             'gmc_full (gmc_prop, store, history = M4b gmc_hist).', '']
    lines += ['| Condition | W_mix | W_gru | W_F_mix | W_F_gru | M7 gru | M37 gru | C1 gru | '
              'N share | Z share | Peak M/N | Final M/N | Replay steps | History symbols | '
              'History bits unpriced | Wall seconds (sum) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cond in shown:
        rs = [r for r in score if r['cond'] == cond]
        lines.append(
            '| ' + cond
            + f' | {mean_std([r["W_mix"] for r in rs])} | {mean_std([r["W_gru"] for r in rs])} '
            + f'| {mean_std([r["W_F_mix"] for r in rs])} | {mean_std([r["W_F_gru"] for r in rs])} '
            + f'| {mean_std([r["held_gru"]["M7"] for r in rs])} '
            + f'| {mean_std([r["held_gru"]["M37"] for r in rs])} '
            + f'| {mean_std([r["held_gru"]["C1"] for r in rs])} '
            + f'| {np.mean([r["share"]["N"] for r in rs]):.6f} '
            + f'| {np.mean([r["share"]["Z"] for r in rs]):.6f} '
            + f'| {np.mean([r["peak_M_over_N"] for r in rs]):.6f} '
            + f'| {np.mean([r["final_M_over_N"] for r in rs]):.6f} '
            + f'| {np.mean([r["replay_steps"] for r in rs]):.1f} '
            + f'| {np.mean([r["history_symbols"] for r in rs]):.0f} '
            + f'| {np.mean([r["history_bits_unpriced"] for r in rs]):.3f} '
            + f'| {sum(r["wall_seconds"] for r in rs):.3f} |')
    lines += ['', 'Per-seed W_mix.', '',
              '| Seed | ' + ' | '.join(CONDS) + ' |',
              '|---:|' + '---:|' * len(CONDS)]
    for seed in SEEDS:
        cells = []
        for cond in CONDS:
            hit = [r for r in score if r['seed'] == seed and r['cond'] == cond]
            cells.append(_cell(hit[0]['W_mix']) if hit else 'NA')
        lines.append(f'| {seed} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Per-seed N share and Z share.', '',
              '| Seed | Condition | N share | Z share |',
              '|---:|---|---:|---:|']
    for seed in SEEDS:
        for cond in CONDS:
            hit = [r for r in score if r['seed'] == seed and r['cond'] == cond]
            if not hit:
                continue
            lines.append(f'| {seed} | {cond} | {_cell(hit[0]["share"]["N"])} | '
                         f'{_cell(hit[0]["share"]["Z"])} |')
    lines += ['', 'score: per run.', '',
              '| T | Seed | Condition | W_mix | W_gru | W_F_mix | W_F_gru | M7 gru | M37 gru | C1 gru | '
              'Peak M/N | Final M/N | Peak ratio M/N | Windows | Evicted | Protected evictions | '
              'Replay steps | History symbols | History bits unpriced | Wall seconds |',
              '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in score:
        lines.append(
            f'| {r["glances"]} | {r["seed"]} | {r["cond"]} | {_cell(r["W_mix"])} | {_cell(r["W_gru"])} | '
            f'{_cell(r["W_F_mix"])} | {_cell(r["W_F_gru"])} | {_cell(r["held_gru"]["M7"])} | '
            f'{_cell(r["held_gru"]["M37"])} | {_cell(r["held_gru"]["C1"])} | '
            f'{_cell(r["peak_M_over_N"])} | {_cell(r["final_M_over_N"])} | {_cell(r["peak_ratio_M_over_N"])} | '
            f'{r["final_windows"]} | {r["evicted_windows"]} | {r["protected_evictions"]} | '
            f'{r["replay_steps"]} | {r["history_symbols"]} | {r["history_bits_unpriced"]:.3f} | '
            f'{r["wall_seconds"]:.3f} |')
    lines += ['', 'score: held-out mixture bits per channel.', '',
              '| T | Seed | Condition | ' + ' | '.join(m4b.LEARN) + ' |',
              '|---:|---:|---|---:|' + '---:|' * (len(m4b.LEARN) - 1)]
    for r in score:
        lines.append(f'| {r["glances"]} | {r["seed"]} | {r["cond"]} | '
                     + ' | '.join(_cell(r['held_mix'][c]) for c in m4b.LEARN) + ' |')
    lines += ['', 'score: held-out GRU bits per channel.', '',
              '| T | Seed | Condition | ' + ' | '.join(m4b.LEARN) + ' |',
              '|---:|---:|---|---:|' + '---:|' * (len(m4b.LEARN) - 1)]
    for r in score:
        lines.append(f'| {r["glances"]} | {r["seed"]} | {r["cond"]} | '
                     + ' | '.join(_cell(r['held_gru'][c]) for c in m4b.LEARN) + ' |')
    if score:
        lines += ['', 'Mean overall glance shares.', '',
                  '| Condition | ' + ' | '.join(m4b.CH) + ' |',
                  '|---|' + '---:|' * len(m4b.CH)]
        for cond in shown:
            rs = [r for r in score if r['cond'] == cond]
            lines.append('| ' + cond + ' | ' + ' | '.join(
                f'{np.mean([r["share"][c] for r in rs]):.6f}' for c in m4b.CH) + ' |')
        lines += ['', 'Mean glance shares by quarter.', '',
                  '| Condition | Quarter | ' + ' | '.join(m4b.CH) + ' |',
                  '|---|---:|' + '---:|' * len(m4b.CH)]
        for cond in shown:
            rs = [r for r in score if r['cond'] == cond]
            for q in range(4):
                lines.append(f'| {cond} | {q + 1} | ' + ' | '.join(
                    f'{np.mean([r["share_by_quarter"][q][c] for r in rs]):.6f}' for c in m4b.CH) + ' |')
        lines += ['', 'Glance shares per seed, overall and by quarter.', '',
                  '| Seed | Condition | Period | ' + ' | '.join(m4b.CH) + ' |',
                  '|---:|---|---|' + '---:|' * len(m4b.CH)]
        for r in score:
            for label, shares in [('all', r['share'])] + [
                    (f'Q{i + 1}', q) for i, q in enumerate(r['share_by_quarter'])]:
                lines.append(f'| {r["seed"]} | {r["cond"]} | {label} | '
                             + ' | '.join(f'{shares[c]:.6f}' for c in m4b.CH) + ' |')
        lines += ['', 'Observation alignment. nostore, hist_rep and rand_full share one log per seed.', '',
                  '| Seed | Condition | Glance checksum | Observation checksum |',
                  '|---:|---|---:|---:|']
        for r in score:
            lines.append(f'| {r["seed"]} | {r["cond"]} | {r["glance_checksum"]} | '
                         f'{r["observation_checksum"]} |')
        lines += ['', 'Hard budget. Allowance is 0.15 × log2(42) + (21 × log2(42) + 32) / N. '
                  'Peak ratio is the max of M/N after each glance.', '',
                  '| Seed | Condition | Peak M/N | Peak ratio M/N | Budget ratio | Allowance | '
                  'Over-budget glances | Protected evictions | Final windows |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in score:
            if r['cond'] not in STORE_CONDS:
                continue
            allowance = r['budget_ratio'] + r['one_window_bits'] / r['observed_symbols']
            lines.append(f'| {r["seed"]} | {r["cond"]} | {_cell(r["peak_M_over_N"])} | '
                         f'{_cell(r["peak_ratio_M_over_N"])} | {_cell(r["budget_ratio"])} | '
                         f'{allowance:.6f} | {r["glances_over_budget"]} | {r["protected_evictions"]} | '
                         f'{r["final_windows"]} |')
        lines += ['', 'Measurements for the pre-registered comparisons. Bars are not scored here.', '',
                  '| Seed | W_mix gmc_full | W_mix rand_full | gmc_full − rand_full | N gmc_full | Z gmc_full | '
                  'W_mix hist_rep | W_mix nostore | gmc_full − hist_rep | gmc_full − nostore | '
                  'W_mix gmc_nostore | gmc_full − gmc_nostore | W_F_mix gmc_full | W_F_mix gmc_nostore | '
                  'W_F gmc_full − gmc_nostore | W_mix rand8_full | gmc_full − rand8_full |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        by = {(r['seed'], r['cond']): r for r in score}

        def pick(seed, cond, key):
            row = by.get((seed, cond))
            if row is None:
                return None
            return row[key]

        acc = []
        for seed in SEEDS:
            gmc = by.get((seed, 'gmc_full'))
            if gmc is None:
                continue
            vals = {
                'gmc': gmc['W_mix'],
                'rand': pick(seed, 'rand_full', 'W_mix'),
                'hist': pick(seed, 'hist_rep', 'W_mix'),
                'none': pick(seed, 'nostore', 'W_mix'),
                'gn': pick(seed, 'gmc_nostore', 'W_mix'),
                'wf': gmc['W_F_mix'],
                'wfn': pick(seed, 'gmc_nostore', 'W_F_mix'),
                'r8': pick(seed, 'rand8_full', 'W_mix'),
            }
            def sub(a, b):
                if a is None or b is None:
                    return None
                return a - b
            row = [vals['gmc'], vals['rand'], sub(vals['gmc'], vals['rand']),
                   gmc['share']['N'], gmc['share']['Z'],
                   vals['hist'], vals['none'], sub(vals['gmc'], vals['hist']),
                   sub(vals['gmc'], vals['none']), vals['gn'], sub(vals['gmc'], vals['gn']),
                   vals['wf'], vals['wfn'], sub(vals['wf'], vals['wfn']),
                   vals['r8'], sub(vals['gmc'], vals['r8'])]
            acc.append(row)
            lines.append('| ' + str(seed) + ' | ' + ' | '.join(
                'NA' if v is None else _cell(v) for v in row) + ' |')
        if acc:
            means = []
            for i in range(len(acc[0])):
                col = [row[i] for row in acc if row[i] is not None]
                means.append(float(np.mean(col)) if col else None)
            lines.append('| mean | ' + ' | '.join(
                'NA' if v is None else _cell(v) for v in means) + ' |')
    if smoke:
        lines += ['', 'Smoke, Kaggle, seed 9, T = 32, all 6 conditions.', '',
                  '| Condition | W_mix | W_gru | W_F_mix | W_F_gru | N share | Z share | '
                  'Peak M/N | Replay steps | Wall seconds | Glance checksum |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for cond in CONDS:
            hit = [r for r in smoke if r['cond'] == cond]
            if not hit:
                continue
            r = hit[0]
            lines.append(f'| {cond} | {_cell(r["W_mix"])} | {_cell(r["W_gru"])} | {_cell(r["W_F_mix"])} | '
                         f'{_cell(r["W_F_gru"])} | {_cell(r["share"]["N"])} | {_cell(r["share"]["Z"])} | '
                         f'{_cell(r["peak_M_over_N"])} | {r["replay_steps"]} | {r["wall_seconds"]:.3f} | '
                         f'{r["glance_checksum"]} |')
    lines += ['', 'Determinism. nostore, seed 9, T = 32. CPU numerics may differ. Difference is Kaggle minus local.', '']
    if det and smoke:
        remote = [r for r in smoke if r['cond'] == 'nostore' and r['seed'] == 9 and r['glances'] == 32]
        if remote:
            k = remote[0]
            lines += ['| Source | W_mix | W_gru | W_F_mix | Glance checksum | Observation checksum | '
                      'torch | numpy | Wall seconds |',
                      '|---|---:|---:|---:|---:|---:|---|---|---:|',
                      f'| local | {_cell(det["W_mix"])} | {_cell(det["W_gru"])} | {_cell(det["W_F_mix"])} | '
                      f'{det["glance_checksum"]} | {det["observation_checksum"]} | {det.get("torch_version", "")} | '
                      f'{det.get("numpy_version", "")} | {det["wall_seconds"]:.3f} |',
                      f'| kaggle | {_cell(k["W_mix"])} | {_cell(k["W_gru"])} | {_cell(k["W_F_mix"])} | '
                      f'{k["glance_checksum"]} | {k["observation_checksum"]} | {k.get("torch_version", "")} | '
                      f'{k.get("numpy_version", "")} | {k["wall_seconds"]:.3f} |',
                      f'| kaggle − local | {k["W_mix"] - det["W_mix"]:.9e} | {k["W_gru"] - det["W_gru"]:.9e} | '
                      f'{k["W_F_mix"] - det["W_F_mix"]:.9e} | | | | | |']
        else:
            lines.append('Kaggle nostore smoke row missing.')
    elif det:
        lines.append(f'Local only. W_mix={_cell(det["W_mix"])} W_gru={_cell(det["W_gru"])} '
                     f'torch={det.get("torch_version", "")}.')
    else:
        lines.append('Local determinism row missing.')
    lines += ['', 'Kaggle kernels.', '',
              '| Id | Phase | Seeds | T | Status | Elapsed seconds | Rerun | Note |',
              '|---|---|---|---:|---|---:|---|---|']
    for ker in kernels:
        lines.append(f'| {ker.get("id", "")} | {ker.get("phase", "")} | {ker.get("seeds", "")} | '
                     f'{ker.get("T", "")} | {ker.get("status", "")} | {ker.get("elapsed_seconds", "")} | '
                     f'{ker.get("rerun", False)} | {ker.get("note", "")} |')
    lines += ['', 'Protocol notes:',
              '- World: FactWorld, channels G C1 C7 E30 E110 M7 M37 F N Z. Glance = 64 symbols. Processes resume.',
              '- N: uniform over 42 symbols. Z: the space symbol repeated. Seeds 9 10 11 12 13 14.',
              '- GRU: symbol embedding 32, channel embedding 8, hidden 256, Adam lr 3e-3, clip 1.0.',
              '  Per-channel state. One live Adam step per glance, then zero or two replay steps.',
              '- Same hard budget, history replay, match, writes and W as M4b. Code paths are m4b.run.',
              '- nostore, hist_rep, rand_full and gmc_full call m4b nostore, hist_rep, store_hist and gmc_hist.',
              '- rand8_full: store_hist path, choice uniform on the 8 learnable channels after the first pass.',
              '- gmc_nostore: gmc_hist path with the store and the match off. History replay stays on.',
              '- Rand conditions share the choice/observation stream (10_000 + seed). Replay uses 30_000 + seed.',
              '- W_mix and W_gru are M4b\'s frozen 2048-symbol continuations. No store means W_mix = W_gru.',
              '- First pass: glance g visits channel g for g < 10, in every condition, then the controller.',
              '- History log is not charged to M. history_bits_unpriced = observed symbols × log2(42).',
              '']
    lines += [f'Note: {n}' for n in notes]
    n_score = len(score)
    status = 'complete' if n_score == len(SEEDS) * len(CONDS) else 'incomplete'
    lines += ['', f'Score runs: {n_score} of {len(SEEDS) * len(CONDS)}.',
              f'Status: {status}.']
    text = '\n'.join(lines) + '\n'
    (out / 'RESULTS.md').write_text(text)
    payload = {
        'started': started, 'updated': now(), 'status': status, 'seeds': SEEDS, 'T': SCORE_T,
        'conditions': CONDS, 'runs': score, 'smoke_runs': smoke, 'determinism_local': det,
        'kernels': kernels, 'notes': notes,
    }
    (out / 'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    return status


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
    det_path = root / 'det_local.json'
    det = json.loads(det_path.read_text()) if det_path.exists() else None
    ker_path = root / 'kernels.json'
    kernels = json.loads(ker_path.read_text()).get('kernels', []) if ker_path.exists() else []
    meta_path = root / 'meta.json'
    started = json.loads(meta_path.read_text())['started'] if meta_path.exists() else now()
    notes = []
    problems = family_ok(score) if score else []
    if problems:
        notes.extend(problems)
    return write_report(score, smoke, det, kernels, root, started, notes)


def checks():
    m4b.checks()
    hosted = Hosted('gmc_nostore', 'gmc_hist')
    assert hosted == 'gmc_hist' and hosted == 'gmc_nostore' and hosted != 'store_hist'
    assert hosted in m4b.CONDS and str(hosted) == 'gmc_nostore'
    rand8 = Hosted('rand8_full', 'store_hist')
    assert rand8 == 'store_hist' and rand8 != 'gmc_hist' and rand8 in m4b.CONDS
    assert m4b.has_store('gmc_hist') is True and m4b.has_store('nostore') is False
    with _patch('has_store', _has_store):
        assert m4b.has_store(hosted) is False
        assert m4b.has_store('store_hist') is True
        assert m4b.has_store('gmc_hist') is True
    assert m4b.has_store('gmc_hist') is True
    rng = np.random.default_rng(0)
    with _patch('choose', _choose):
        assert m4b.choose(rand8, 0, np.zeros(len(m4b.CH)), rng) == 0
        picks = [m4b.choose(rand8, 10, np.zeros(len(m4b.CH)), rng) for _ in range(30)]
        assert all(0 <= p < len(m4b.LEARN) for p in picks)
        rng_wide = np.random.default_rng(1)
        wide = [m4b.choose('nostore', 10, np.zeros(len(m4b.CH)), rng_wide) for _ in range(40)]
        assert any(p >= len(m4b.LEARN) for p in wide)
    print(f'{now()} m4c checks passed', flush=True)


def _write_run(result, out, phase):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'run_{phase}_{result["cond"]}_{result["seed"]}_{result["glances"]}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--det', action='store_true')
    parser.add_argument('--assemble', action='store_true')
    parser.add_argument('--cond')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--T', type=int, default=32)
    parser.add_argument('--phase', default='local')
    parser.add_argument('--out', default='results/m4c')
    args = parser.parse_args()
    configure_threads()
    print(f'{now()} torch threads {torch.get_num_threads()} torch {torch.__version__}', flush=True)
    if args.checks:
        checks()
    if args.preflight:
        out = Path('results/m4c/preflight')
        runs = []
        for cond in CONDS:
            result, _, _ = run_one(cond, 9, 12, 'preflight')
            _write_run(result, out, 'preflight')
            runs.append(result)
        problems = family_ok(runs)
        if problems:
            raise SystemExit('preflight: ' + '; '.join(problems))
        print(f'{now()} preflight ok', flush=True)
    if args.det:
        root = Path('results/m4c')
        root.mkdir(parents=True, exist_ok=True)
        meta = root / 'meta.json'
        if not meta.exists():
            meta.write_text(json.dumps({'started': now()}) + '\n')
        result, _, _ = run_one('nostore', 9, 32, 'det')
        (root / 'det_local.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(f'{now()} det W_mix={result["W_mix"]:.6f} W_gru={result["W_gru"]:.6f} '
              f'glance={result["glance_checksum"]} obs={result["observation_checksum"]}', flush=True)
    if args.cond:
        result, _, _ = run_one(args.cond, args.seed, args.T, args.phase)
        path = _write_run(result, args.out, args.phase)
        print(f'{now()} wrote {path}', flush=True)
    if args.assemble:
        status = assemble(args.out)
        print(f'{now()} assemble {status}', flush=True)


if __name__ == '__main__':
    main()
