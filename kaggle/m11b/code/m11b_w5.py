"""M11b: M11's four sleep arms on World5, scored as T14.

The arm bodies stay in m11_w6.run_life. This file only swaps the world those
bodies already call: World5 under World6's names, M9's life and exam store,
and W14 written into the checkpoint field M11 averages.
"""
import os
import sys

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CURIO_LR'] = '0.004'
    os.environ['PYTHONHASHSEED'] = '0'
    if os.environ.get('_M11B_MAIN') != '1':
        os.environ['_M11B_MAIN'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import contextlib
import hashlib
import json
import math
import shutil
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import m7_w3 as m7
import m9_w5 as m9
import m11_w6 as m11
import w4_headroom as w4
import w4_world
import w5_headroom as w5
from w4_world import split_learnable
from w5_world import World5

ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(48, 60))
ARMS = m11.ARMS
H = m11.H
L_SCORE = 2048
PERIOD = 128
SMOKE_SEED = 24
DATASET_ID = 'ezzzzzx/pdp-code-m11b'
KERNEL_LIMIT_S = 80 * 60
GAPS = (
    ('age_sleep_cf', 'age_sleeparm', -0.05),
    ('age_sleep_cf', 'age_fixed20', 0.03),
    ('age_fixed20', 'age_nosleep', -0.10),
)
CODE_FILES = (
    'm11b_w5.py',
    'm11_w6.py',
    'm9_w5.py',
    'm8_w5.py',
    'm7_w3.py',
    'w6_world.py',
    'w6_headroom.py',
    'w5_world.py',
    'w5_headroom.py',
    'w4_world.py',
    'w4_headroom.py',
    'w3_world.py',
    'w3_headroom.py',
    'curio.py',
    'h20_world.py',
    'h30w_ring.py',
    'h30x_trace.py',
    'h30y_roots.py',
)
assert m11.H == 64 and m11.PERIOD == PERIOD
assert set(ARMS) == {'age_fixed20', 'age_sleeparm', 'age_sleep_cf', 'age_nosleep'}


class World5As6(World5):
    """World5, with the method and log names m11_w6.run_life already calls."""

    def maybe_birth(self, decision):
        return self.maybe_swap(decision)

    def is_learnable(self, slot):
        return split_learnable(self.rooms[int(slot)]) is not None

    @property
    def birth_log(self):
        return self.swap_log


def take_checkpoint(life, decision, prev, clock):
    """W5's frozen W14. M11 averages checkpoint['open'], so that field holds W14."""
    rec = w5.take_checkpoint(life, decision, prev, clock)
    rec['open'] = rec['W14']
    return rec


def use_world5():
    m11.World6 = World5As6
    m11.expected_birth_log = w5.expected_swap_log
    m11.w6.OpenLife = w4.Life
    m11.w6.take_checkpoint = take_checkpoint
    m11.w6.checkpoints_of = w5.checkpoints_of
    m11.w6._threads = w5._threads


def run_life(seed, arm, L=L_SCORE, trace=False, noop_sleep=False):
    use_world5()
    row = m11.run_life(seed, arm, L=L, trace=trace, noop_sleep=noop_sleep)
    scores = [cp['W14'] for cp in row['checkpoints']]
    row['T14'] = float(np.mean(scores))
    row['W14'] = float(scores[-1])
    row['world'] = 'World5'
    if abs(row['T14'] - row['T_open']) > 1e-12:
        raise RuntimeError('T14 and the averaged checkpoint diverged')
    return row


def _job(job):
    seed, arm, L = job
    row = run_life(seed, arm, L)
    print(
        f's{seed} {arm} T14={row["T14"]:.9f} sleep={row["sleep_share"]:.6f} wall={row["wall_s"]:.3f}',
        flush=True,
    )
    return row


def kernel_main(seed, out, workers=4, kernel_id=None, arms=None, label=''):
    if seed not in SEEDS or workers != 4:
        raise ValueError('kernels require fresh seed 48..59 and a pool of 4')
    arms = tuple(arms or ARMS)
    start = time.perf_counter()
    with get_context('spawn').Pool(4) as pool:
        runs = pool.map(_job, [(seed, arm, L_SCORE) for arm in arms])
    payload = dict(
        seed=seed, workers=4, kernel_id=kernel_id or f'ezzzzzx/m11b-w5-s{seed}',
        runs=runs, wall_s=time.perf_counter() - start,
    )
    m7.dump(Path(out) / f'm11b_s{seed}{label}.json', payload)
    return payload


def _checkpoint_gap(row, ref):
    return max(abs(a['W14'] - b['W14']) for a, b in zip(row['checkpoints'], ref['checkpoints']))


def smoke(out):
    """One process. A failed check leaves its log and does not stage kernels."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(ok=False, seed=SMOKE_SEED, period=PERIOD, L_score=L_SCORE, H=H, workers=1, log=[])

    class Tee:
        def write(self, text):
            sys.__stdout__.write(text)
            log_file.write(text)

        def flush(self):
            sys.__stdout__.flush()
            log_file.flush()

    start = time.perf_counter()
    try:
        with (out / 'smoke.txt').open('w') as log_file, contextlib.redirect_stdout(Tee()):
            w5._threads()
            use_world5()
            print('threads=1 process=1 World5 period=128 H=64 wake decisions', flush=True)
            print(
                'Arms are m11_w6.run_life. World5 stands in for World6; the life is w4.Life; '
                'checkpoint open is W14, so the averaged score is T14.',
                flush=True,
            )
            print(
                'CF room comparison stays M11: bits/decision divided by destination exam symbols. '
                'age_sleeparm keeps M9 bits/decision.',
                flush=True,
            )
            baselines = {}
            for arm in ('age_fixed20', 'age_sleeparm'):
                row = run_life(SMOKE_SEED, arm, 256)
                ref = m9.run_life(SMOKE_SEED, arm, 256)
                gap = abs(row['T14'] - ref['T14'])
                cp_gap = _checkpoint_gap(row, ref)
                print(
                    f'BASELINE {arm} T14={row["T14"]:.12f} M9={ref["T14"]:.12f} '
                    f'abs={gap:.3e} checkpoint_max_abs={cp_gap:.3e} '
                    f'sleep_share={row["sleep_share"]:.9f} M9_sleep_share={ref["sleep_share"]:.9f}',
                    flush=True,
                )
                if gap > 1e-6 or cp_gap > 1e-6:
                    raise RuntimeError(f'{arm} does not reproduce M9 on seed {SMOKE_SEED} L=256')
                baselines[arm] = dict(
                    T14=row['T14'], reference=ref['T14'], abs=gap,
                    checkpoint_max_abs=cp_gap, sleep_share=row['sleep_share'],
                    reference_sleep_share=ref['sleep_share'],
                )
            payload['baselines'] = baselines
            shadow_L = 512
            exact = run_life(SMOKE_SEED, 'age_sleep_cf', shadow_L, noop_sleep=True)
            sleeps = [v for v in exact['visits'] if v['kind'] == 'sleep']
            if not sleeps or not any(v.get('wake_horizon') == H for v in sleeps):
                shadow_L = 1024
                exact = run_life(SMOKE_SEED, 'age_sleep_cf', shadow_L, noop_sleep=True)
                sleeps = [v for v in exact['visits'] if v['kind'] == 'sleep']
            if not sleeps or not any(v.get('wake_horizon') == H for v in sleeps):
                raise RuntimeError('shadow-exact life had no full H=64 wake horizon')
            err = max(abs(v['shadow_bits'] - v['live_bits']) for v in sleeps)
            if err > 1e-9:
                raise RuntimeError(f'no-op shadow mismatch {err}')
            full = sum(v['wake_horizon'] == H for v in sleeps)
            print(
                f'SHADOW_EXACT L={shadow_L} no-op sleep visits={len(sleeps)} '
                f'max_abs_bits={err:.12g} full_horizons={full}',
                flush=True,
            )
            payload['shadow_exact'] = dict(
                L=shadow_L, max_abs_bits=err, visits=len(sleeps), full_horizons=full,
            )
            finite = {}
            for arm in ARMS:
                row = run_life(SMOKE_SEED, arm, 256)
                finite[arm] = dict(
                    T14=row['T14'], W14=row['W14'], sleep_share=row['sleep_share'], wall_s=row['wall_s'],
                )
                print('FINITE ' + json.dumps(dict(arm=arm, **finite[arm])), flush=True)
                if not row['finite'] or not math.isfinite(row['T14']):
                    raise RuntimeError(f'{arm} nonfinite at L=256')
            payload['finite_L256'] = finite
            timed = run_life(SMOKE_SEED, 'age_sleep_cf', 1024, trace=True)
            payload['cf_L1024_wall_s'] = timed['wall_s']
            payload['cf_L1024_T14'] = timed['T14']
            # Same allowance M11 used: decision ratio, 2x store growth, 4 arms, no pool speedup.
            factor = (L_SCORE / 1024) * 2 * 4
            estimate = timed['wall_s'] * factor
            payload['estimate'] = dict(
                cf_L1024_wall_s=timed['wall_s'], kernel_wall_s=estimate, split=estimate > KERNEL_LIMIT_S,
                factor=factor, L_score=L_SCORE,
                method='(2048/1024) decisions * 2x store allowance * 4 arms; no parallel speedup assumed',
            )
            print('WALL_ESTIMATE ' + json.dumps(payload['estimate']), flush=True)
            payload['ok'] = True
            print('SMOKE_OK', flush=True)
    except Exception as exc:
        payload['error'] = repr(exc)
        raise
    finally:
        payload['wall_s'] = time.perf_counter() - start
        if (out / 'smoke.txt').exists():
            payload['log'] = (out / 'smoke.txt').read_text().splitlines()
        m7.dump(out / 'smoke.json', payload)
    return payload


def stage(smoke_payload):
    if not smoke_payload.get('ok'):
        raise RuntimeError('refusing to stage after a failed smoke')
    base = ROOT / 'kaggle/m11b'
    code = base / 'code'
    code.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
        hashes[name] = hashlib.sha256((code / name).read_bytes()).hexdigest()
    m7.dump(code / 'dataset-metadata.json', dict(
        title='pdp code m11b', id=DATASET_ID, licenses=[{'name': 'other'}],
    ))
    m7.dump(base / 'code-sha256.json', hashes)
    split = bool(smoke_payload['estimate']['split'])
    groups = [('a', ARMS[:2]), ('b', ARMS[2:])] if split else [('', ARMS)]
    manifest = []
    for seed in SEEDS:
        for label, arms in groups:
            suffix = f'-{label}' if label else ''
            folder = base / f's{seed}{suffix}'
            folder.mkdir(parents=True, exist_ok=True)
            kid = f'ezzzzzx/m11b-w5-s{seed}{suffix}'
            meta = dict(
                id=kid, title=kid.split('/')[1], code_file='kernel.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=False, enable_tpu=False,
                enable_internet=False, dataset_sources=[DATASET_ID], competition_sources=[],
                kernel_sources=[], model_sources=[],
            )
            m7.dump(folder / 'kernel-metadata.json', meta)
            src = f'''import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CURIO_LR'] = '0.004'
os.environ['PYTHONHASHSEED'] = '0'
if os.environ.get('_M11B_KERNEL') != '1':
    os.environ['_M11B_KERNEL'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)

def main():
    hits = sorted(Path('/kaggle/input').rglob('m11b_w5.py'))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one code dataset, found {{len(hits)}}')
    code = str(hits[0].parent)
    sys.path.insert(0, code)
    os.environ['PYTHONPATH'] = code
    import m11b_w5
    m11b_w5.kernel_main({seed}, '/kaggle/working', workers=4, kernel_id={kid!r}, arms={tuple(arms)!r}, label={suffix!r})

if __name__ == '__main__':
    main()
'''
            (folder / 'kernel.py').write_text(src)
            manifest.append(dict(
                seed=seed, folder=folder.name, kernel_id=kid, arms=list(arms),
                output=f'm11b_s{seed}{suffix}.json',
            ))
    m7.dump(base / 'manifest.json', manifest)
    text = (
        '# M11b-W5\n\n'
        f'{len(manifest)} scored kernels pending. Seeds 48–59; L = 2048; period = 128; H = 64 wake decisions.\n\n'
        'Wall estimate: ' + json.dumps(smoke_payload['estimate']) + '\n\n'
        'Kernels: ' + ', '.join(item['kernel_id'] for item in manifest) + '\n\n'
        'Smoke output:\n\n```\n' + '\n'.join(smoke_payload['log']) + '\n```\n'
    )
    results = ROOT / 'results/m11b'
    results.mkdir(parents=True, exist_ok=True)
    (results / 'RESULTS.md').write_text(text)
    print(f'STAGED {len(manifest)} private CPU kernels; split={split}', flush=True)
    return manifest


def stats(values):
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return dict(mean=None, std_ddof0=None, min=None, max=None, n=0)
    return dict(
        mean=float(array.mean()), std_ddof0=float(array.std(ddof=0)),
        min=float(array.min()), max=float(array.max()), n=int(array.size),
    )


def _cell(value):
    return f'{value:.9f}'


def assemble(out):
    out = Path(out)
    smoke_payload = json.loads((out / 'smoke.json').read_text())
    if not smoke_payload.get('ok'):
        raise RuntimeError('smoke did not pass')
    manifest = json.loads((ROOT / 'kaggle/m11b/manifest.json').read_text())
    runs, kernels = [], []
    for item in manifest:
        payload = json.loads((out / item['folder'] / item['output']).read_text())
        if payload['seed'] != item['seed'] or payload['kernel_id'] != item['kernel_id']:
            raise RuntimeError(f'payload identity mismatch for {item["kernel_id"]}')
        if sorted(row['policy'] for row in payload['runs']) != sorted(item['arms']):
            raise RuntimeError(f'arms mismatch for {item["kernel_id"]}')
        for row in payload['runs']:
            if row['seed'] != item['seed'] or row['L'] != L_SCORE or row['period'] != PERIOD or not row['finite']:
                raise RuntimeError(f'bad row {item["kernel_id"]} {row.get("policy")}')
            decisions = [cp['decision'] for cp in row['checkpoints']]
            if decisions != w5.checkpoints_of(L_SCORE):
                raise RuntimeError(f'checkpoints are not L/8 .. L for seed {row["seed"]}')
            w14 = [cp['W14'] for cp in row['checkpoints']]
            if not math.isclose(row['T14'], float(np.mean(w14)), abs_tol=1e-12):
                raise RuntimeError(f'T14 is not the mean of W14 for seed {row["seed"]}')
        runs.extend(payload['runs'])
        kernels.append({key: payload[key] for key in ('kernel_id', 'seed', 'workers', 'wall_s')})
    if len(runs) != 48 or len({(row['seed'], row['policy']) for row in runs}) != 48:
        raise RuntimeError(f'expected 48 lives, found {len(runs)}')
    by = {(row['seed'], row['policy']): row for row in runs}
    summary = {}
    for arm in ARMS:
        rows = [by[seed, arm] for seed in SEEDS]
        summary[arm] = {key: stats([row[key] for row in rows]) for key in ('T14', 'W14', 'sleep_share', 'wall_s')}
    gaps = {}
    for left, right, threshold in GAPS:
        vals = [by[seed, left]['T14'] - by[seed, right]['T14'] for seed in SEEDS]
        name = f'{left}_minus_{right}'
        gaps[name] = dict(
            per_seed={str(seed): value for seed, value in zip(SEEDS, vals)},
            threshold=threshold,
            count=int(sum(value <= threshold for value in vals)),
            **stats(vals),
        )
    cf_rewards = [
        dict(seed=row['seed'], **visit)
        for row in runs if row['policy'] == 'age_sleep_cf'
        for visit in row['visits'] if visit['kind'] == 'sleep'
    ]
    reward_values = [visit['reward'] for visit in cf_rewards]
    reward_summary = dict(counterfactual=stats(reward_values))
    result = dict(
        seeds=list(SEEDS), arms=list(ARMS), L=L_SCORE, period=PERIOD, H=H,
        world='World5', score='T14', runs=runs, kernels=kernels, summary=summary,
        gaps=gaps, counterfactual_rewards=cf_rewards, reward_summary=reward_summary,
        smoke=smoke_payload,
    )
    m7.dump(out / 'results.json', result)
    lines = [
        '# M11b-W5',
        '',
        'World5; seeds 48–59; L=2048; period=128; H=64 wake decisions; std ddof=0.',
        'T14 is the mean of W14 at L/8, 2L/8, …, L. Count is the number of seeds whose gap is ≤ the threshold.',
        '',
        '## Scoring',
        '',
        '| Quantity | ' + ' | '.join(str(seed) for seed in SEEDS) + ' | Mean | Threshold | Count |',
        '|---|' + '---:|' * 15,
    ]
    for left, right, threshold in GAPS:
        name = f'{left}_minus_{right}'
        gap = gaps[name]
        cells = ' | '.join(_cell(gap['per_seed'][str(seed)]) for seed in SEEDS)
        lines.append(
            f'| {left} - {right} | {cells} | {_cell(gap["mean"])} | {threshold:.2f} | {gap["count"]} |'
        )
    lines += [
        '',
        '## Sleep share',
        '',
        '| Arm | ' + ' | '.join(str(seed) for seed in SEEDS) + ' | Mean |',
        '|---|' + '---:|' * 13,
    ]
    for arm in ARMS:
        cells = ' | '.join(_cell(by[seed, arm]['sleep_share']) for seed in SEEDS)
        lines.append(f'| {arm} | {cells} | {_cell(summary[arm]["sleep_share"]["mean"])} |')
    lines += [
        '',
        '## T14',
        '',
        '| Arm | ' + ' | '.join(str(seed) for seed in SEEDS) + ' | Mean | Std |',
        '|---|' + '---:|' * 14,
    ]
    for arm in ARMS:
        cells = ' | '.join(_cell(by[seed, arm]['T14']) for seed in SEEDS)
        arm_stats = summary[arm]['T14']
        lines.append(f'| {arm} | {cells} | {_cell(arm_stats["mean"])} | {_cell(arm_stats["std_ddof0"])} |')
    cf = reward_summary['counterfactual']
    lines += [
        '',
        '## Counterfactual rewards',
        '',
        f'Mean {cf["mean"]:.12g}. Min {cf["min"]:.12g}. Max {cf["max"]:.12g}. N {cf["n"]}.',
        '',
        '| seed | t0 | delivered | wake | reward | bits/decision | sleep share |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for visit in cf_rewards:
        lines.append(
            f'| {visit["seed"]} | {visit["t0"]} | {visit["delivered"]} | {visit["wake_horizon"]} | '
            f'{visit["reward"]:.12g} | {visit["bits_per_decision"]:.12g} | {visit["sleep_share_delivered"]:.6f} |'
        )
    lines += [
        '',
        '## Walls',
        '',
        '| Arm | ' + ' | '.join(str(seed) for seed in SEEDS) + ' | Mean |',
        '|---|' + '---:|' * 13,
    ]
    for arm in ARMS:
        cells = ' | '.join(f'{by[seed, arm]["wall_s"]:.3f}' for seed in SEEDS)
        lines.append(f'| {arm} | {cells} | {summary[arm]["wall_s"]["mean"]:.3f} |')
    lines += [
        '',
        '## Kernel ids',
        '',
        '```json',
        json.dumps(kernels, indent=2),
        '```',
        '',
        '## Smoke output',
        '',
        '```',
        '\n'.join(smoke_payload['log']),
        '```',
        '',
    ]
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', choices=('smoke', 'stage', 'assemble'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--out', default=str(ROOT / 'results/m11b'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.seed is not None:
        kernel_main(args.seed, args.out, args.workers)
    elif args.command == 'smoke':
        payload = smoke(args.out)
        stage(payload)
    elif args.command == 'stage':
        stage(json.loads((Path(args.out) / 'smoke.json').read_text()))
    elif args.command == 'assemble':
        assemble(args.out)
    else:
        parser.error('supply --seed or smoke/stage/assemble')


if __name__ == '__main__':
    main()
