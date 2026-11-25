"""W9b: sleep pressure plus pre-learning input codelength.

Principles 2/4: replay improves the compressor; room choice uses prog_age_s.
Imports W9's world and pressure dynamics. The decision loop preserves W9's
learning, visit, replay and scoring operations; only pressure_bits adds a
frozen pre-learning score and feeds its bits/symbol to BitsPressure. Its EMA
starts at the first wake observation; the 256-wake median includes the current
EMA. The controller has no clock input. All constants are preregistered.
"""
import os
import sys
sys.dont_write_bytecode = True
if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CURIO_LR'] = '0.004'
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if os.environ.get('_W9B_MAIN') != '1':
        os.environ['_W9B_MAIN'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
from collections import deque
import contextlib
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
import m9_w5 as m9
from w5_world import World5
from h20_world import ALPHA

m7, m8, w4, w5 = m9.m7, m9.m8, m9.w4, m9.w5
ROOT = Path(__file__).resolve().parent
import w9_day_night as w9
SEEDS = tuple(range(72, 84))
ARMS = ('fixed_random', 'fixed_night', 'pressure', 'pressure_opp', 'pressure_bits')
L_SCORE = 2048
DATASET = 'ezzzzzx/pdp-code-w9b'
CODE_FILES = ('w9b_poor_input.py',) + w9.CODE_FILES
dump, night = w9.dump, w9.night
DayNightWorld, Pressure = w9.DayNightWorld, w9.Pressure
random_sleep_mask = w9.random_sleep_mask


class BitsPressure(Pressure):
    """Clock-blind: receives only pre-learning bits/symbol on wake decisions.

    Inherited update initializes the EMA to the first observation, includes
    the current EMA in the 256-wake median, and drains S during sleep.
    w is the inherited storage slot for b, never a learning-gain signal.
    """
    def __init__(self):
        super().__init__(True)

    def wants_sleep(self):
        if self.asleep:
            self.asleep = self.S > 0
        if not self.asleep:
            self.asleep = (self.S >= 128 or
                           (self.S >= 32 and self.w is not None and self.w > self.m))
        return self.asleep


def run_life(seed, arm, L=L_SCORE, night_enabled=True, trace=False):
    if arm not in ARMS + ('nosleep', 'fixed20_compat'):
        raise ValueError(arm)
    m9._threads()
    start = time.perf_counter()
    world = DayNightWorld(seed, night_enabled)
    names = list(m9.IDENTITIES)
    index = {name: i for i, name in enumerate(names)}
    life = w4.Life(seed, world, names, index, m9.N_RING, w4.N_CH, list(m9.LEARN14))
    rng = np.random.default_rng(1_000 + seed)
    choice_rng = np.random.default_rng(10_000 + seed)
    values, counts, last_end = [0.] * 18, [0] * 18, [None] * 18
    v_max = None
    unvisited = set(range(18))
    plan, target, total, n_dec = [], None, 0., 0
    visit_pending = None
    channels, records, traces, sleep_visits = [], [], [], []
    points = set(w5.checkpoints_of(L))
    clock = {'eval': 0.}
    prev_cp = 0
    controller = BitsPressure() if arm == 'pressure_bits' else Pressure(arm == 'pressure_opp')
    wake_bits = []
    random_mask, random_starts = random_sleep_mask(seed, L)
    compat_plan = m7.sleep_plan(L, w4.sleep_decisions(L)) if arm == 'fixed20_compat' else []
    si, sleep_left, t = 0, 0, 0
    snap, checked = None, False
    sleep_learned = 0

    def finish_room():
        nonlocal v_max, visit_pending
        if n_dec:
            reward = total / n_dec
            m7.ema(values, counts, target, reward)
            v_max = reward if v_max is None else max(v_max, reward)
            last_end[target] = t
            visit_pending = w4._close_visit(life, visit_pending)

    while t < L:
        if sleep_left == 0:
            if arm == 'fixed20_compat':
                want = not plan and w4.sleep_due(t, L, compat_plan, si, 18)
            elif arm == 'fixed_random':
                want = t in random_starts
            elif arm == 'fixed_night':
                want = night(t)
            elif arm.startswith('pressure'):
                want = controller.wants_sleep()
            else:
                want = False
            if want:
                sleep_left = min(8, L - t)
                sleep_visits.append({'t': t, 'length': sleep_left})
                if arm == 'fixed20_compat':
                    si += 1
                visit_pending = w4._close_visit(life, visit_pending)
                if not checked:
                    snap = life.begin_sleep_visit()
                    sleep_learned = 0
        sleeping = sleep_left > 0
        # Trace reports controller state BEFORE this decision and AFTER selection.
        if trace:
            rec = dict(t=t, night=int(night(t)), S=controller.S, w_t=controller.w,
                       m_t=controller.m, sleep=int(sleeping))
            traces.append(rec)
            print('TRACE ' + json.dumps(rec), flush=True)
        if sleeping:
            sleep_learned += int(life.sleep_decision())
            channels.append('sleep')
            controller.update(True)
            sleep_left -= 1
            if sleep_left == 0 and snap is not None:
                delta = life.end_sleep_visit(snap)
                if sleep_learned and delta <= 0:
                    raise RuntimeError('sleep learned without changing parameters')
                snap, checked = None, True
        else:
            if not plan:
                target, _, _ = m8.choose_room(world.room, unvisited, values, counts,
                                             last_end, v_max, t, world.rooms, choice_rng, True)
                plan = m9.w4_world.shortest(world.room, target, 18) + ['stay'] * 8
                total, n_dec = 0., 0
            action = plan.pop(0)
            world.t = t
            _, channel, symbols = world.step(action, rng)
            # M7 frozen scorer on the same glance on both sides of learning.
            glance = SimpleNamespace(channel=life.corridor if channel == 'corridor' else index[channel],
                                     symbols=np.asarray(symbols, dtype=np.int64), prev=life.cortex.prev_symbol)
            before = life._score_one(glance) if arm in ('pressure_opp', 'pressure_bits') else None
            total += m8._progress_decision(life, symbols, channel, world.rooms[target])
            n_dec += 1
            signal = (before / len(glance.symbols) if arm == 'pressure_bits' else
                      before - life._score_one(glance) if before is not None else None)
            controller.update(False, signal)
            if arm == 'pressure_bits':
                wake_bits.append(dict(t=t, bits_per_symbol=signal, b_t=controller.w, m_t=controller.m))
            if channel != 'corridor':
                visit_pending = channel
            channels.append(channel)
        t += 1
        world.maybe_swap(t)
        if t in points:
            records.append(w5.take_checkpoint(life, t, prev_cp, clock))
            prev_cp = t
        if not sleeping and not plan:
            finish_room()
            n_dec = 0
    if n_dec:
        finish_room()
    if visit_pending is not None:
        life.snapshot(visit_pending)
    scores = [r['W14'] for r in records]
    slept = sum(c == 'sleep' for c in channels)
    night_slept = sum(c == 'sleep' and night(i) for i, c in enumerate(channels))
    assert world.swap_log == w5.expected_swap_log(seed, 128, L)
    assert not life.exam.ids & life.replay.ids and life.exam_drawn == 0
    assert len(channels) == L and len(records) == 8
    if arm == 'fixed_night':
        assert all((c == 'sleep') == night(i) for i, c in enumerate(channels))
    if arm == 'fixed_random':
        assert np.array_equal(random_mask, np.array(channels) == 'sleep')
    if arm == 'fixed20_compat':
        assert slept == w4.sleep_decisions(L) and si == len(compat_plan)
    row = dict(seed=seed, arm=arm, L=L, period=128, night_enabled=night_enabled,
               T14=float(np.mean(scores)), W14=scores[-1], checkpoints=records,
               sleep_share=slept / L, sleep_decisions=slept, night_sleep_decisions=night_slept,
               night_sleep_share=night_slept / slept if slept else None,
               sleep_visits=sleep_visits, random_starts=random_starts if arm == 'fixed_random' else [],
               swap_log=world.swap_log, sleep_checks=life.sleep_checks,
               finite=all(math.isfinite(v) for cp in records for v in cp['held'].values()),
               wall_s=time.perf_counter() - start)
    if arm == 'pressure_bits':
        row['wake_bits'] = wake_bits
    if trace:
        row['trace'] = traces
    if not row['finite'] or not math.isfinite(row['T14']):
        raise RuntimeError('nonfinite score')
    return row


def smoke(out):
    """One process only; scored L2048 lives are reserved for Kaggle."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(ok=False, workers=1, log=[])
    started = time.perf_counter()
    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            logfile.write(s)
        def flush(self):
            sys.__stdout__.flush()
            logfile.flush()
    try:
        with (out / 'smoke.txt').open('w') as logfile, contextlib.redirect_stdout(Tee()):
            payload['baselines'] = {}
            # W9's M9 check uses the inherited fixed20 schedule. Random
            # placement differs from M9 by definition and is checked vs W9.
            for arm, refarm in (('nosleep', 'age_nosleep'), ('fixed20_compat', 'age_fixed20'),
                                ('fixed_random', 'fixed_random')):
                row = run_life(24, arm, 256, night_enabled=False)
                ref = (w9.run_life(24, refarm, 256, night_enabled=False)
                       if arm == 'fixed_random' else m9.run_life(24, refarm, 256))
                gap = abs(row['T14'] - ref['T14'])
                cp_gap = max(abs(a['W14'] - b['W14'])
                             for a, b in zip(row['checkpoints'], ref['checkpoints']))
                info = dict(T14=row['T14'], reference_T14=ref['T14'],
                            reference='W9' if arm == 'fixed_random' else 'M9',
                            abs=gap, checkpoint_max_abs=cp_gap,
                            sleep_share=row['sleep_share'], wall_s=row['wall_s'])
                print('BASELINE ' + arm + ' ' + json.dumps(info), flush=True)
                assert gap <= 1e-6 and cp_gap <= 1e-6
                payload['baselines'][arm] = info
            timed = run_life(72, 'pressure_bits', 1024)
            dump(out / 'pressure_bits_L1024.json', timed)
            signal = {}
            for label, flag in (('day', False), ('night', True)):
                vals = [r['b_t'] for r in timed['wake_bits'] if night(r['t']) == flag]
                signal[label] = dict(mean_b_t=float(np.mean(vals)), wake_decisions=len(vals))
            signal['night_minus_day'] = signal['night']['mean_b_t'] - signal['day']['mean_b_t']
            payload['signal_L1024'] = signal
            print('INPUT_BITS ' + json.dumps(signal), flush=True)
            # Stop before staging if the preregistered diagnostic fails.
            assert signal['night_minus_day'] > 0, 'night EMA bits/symbol not higher'
            payload['finite_L512'] = {}
            for arm in ARMS:
                row = run_life(72, arm, 512)
                info = {k: row[k] for k in ('T14', 'sleep_share', 'night_sleep_share', 'wall_s', 'finite')}
                payload['finite_L512'][arm] = info
                print('FINITE ' + arm + ' ' + json.dumps(info), flush=True)
            # Sequential five-arm bound with 2x allowance for reservoir growth;
            # staged kernels actually use up to four concurrent arm workers.
            estimate = sum(v['wall_s'] for v in payload['finite_L512'].values()) * 4 * 2
            payload['estimate'] = dict(kernel_wall_s=estimate, limit_s=85 * 60,
                                       within_limit=estimate <= 85 * 60, factor=8,
                                       pressure_bits_L1024_wall_s=timed['wall_s'])
            print('WALL_ESTIMATE ' + json.dumps(payload['estimate']), flush=True)
            assert estimate <= 85 * 60, 'kernel estimate exceeds 85 minutes'
            payload['ok'] = True
            print('SMOKE_OK', flush=True)
    finally:
        payload['wall_s'] = time.perf_counter() - started
        payload['log'] = (out / 'smoke.txt').read_text().splitlines()
        dump(out / 'smoke.json', payload)
    return payload


def job(args):
    row = run_life(*args)
    print(json.dumps({k: row[k] for k in ('seed', 'arm', 'T14', 'sleep_share', 'wall_s')}), flush=True)
    return row


def kernel_main(seed, out, arms, kernel_id, suffix=''):
    if seed not in SEEDS or not set(arms) <= set(ARMS):
        raise ValueError('kernel jobs')
    start = time.perf_counter()
    with get_context('spawn').Pool(min(4, len(arms))) as pool:
        runs = pool.map(job, [(seed, arm, L_SCORE) for arm in arms])
    dump(Path(out) / f'w9b_s{seed}{suffix}.json', dict(seed=seed, kernel_id=kernel_id,
         workers=min(4, len(arms)), runs=runs, wall_s=time.perf_counter() - start,
         code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES}))


def stage(payload):
    assert payload['ok']
    base = ROOT / 'kaggle/w9b'
    code = base / 'code'
    code.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
        hashes[name] = hashlib.sha256((code / name).read_bytes()).hexdigest()
    dump(code / 'dataset-metadata.json', dict(title='pdp code w9b', id=DATASET, licenses=[{'name': 'other'}]))
    dump(base / 'code-sha256.json', hashes)
    assert payload['estimate']['within_limit']
    groups = [('', ARMS)]
    manifest = []
    for seed in SEEDS:
        for suffix, arms in groups:
            folder = base / f's{seed}' / (suffix[1:] if suffix else '')
            folder.mkdir(parents=True, exist_ok=True)
            kid = f'ezzzzzx/w9b-s{seed}{suffix.replace("_", "-")}'
            dump(folder / 'kernel-metadata.json', dict(id=kid, title=kid.split('/')[1], code_file='kernel.py',
                 language='python', kernel_type='script', is_private=True, enable_gpu=False, enable_tpu=False,
                 enable_internet=False, dataset_sources=[DATASET], competition_sources=[], kernel_sources=[], model_sources=[]))
            src = f'''import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CURIO_LR'] = '0.004'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONHASHSEED'] = '0'
if os.environ.get('_W9B_KERNEL') != '1':
    os.environ['_W9B_KERNEL'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)
if __name__ == '__main__':
    hits = sorted(Path('/kaggle/input').rglob('w9b_poor_input.py'))
    if len(hits) != 1:
        raise RuntimeError('expected exactly one W9b dataset')
    code = str(hits[0].parent)
    sys.path.insert(0, code)
    os.environ['PYTHONPATH'] = code
    import w9b_poor_input as w9
    w9.kernel_main({seed}, '/kaggle/working', {arms!r}, {kid!r}, {suffix!r})
'''
            (folder / 'kernel.py').write_text(src)
            manifest.append(dict(seed=seed, folder=str(folder.relative_to(base)), arms=list(arms),
                                 kernel_id=kid, output=f'w9b_s{seed}{suffix}.json'))
    dump(base / 'manifest.json', manifest)
    print(f'STAGED {len(manifest)} private CPU kernels', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('smoke', 'stage'))
    p.add_argument('--out', type=Path, default=ROOT / 'results/w9b')
    a = p.parse_args()
    payload = smoke(a.out) if a.command == 'smoke' else json.loads((a.out / 'smoke.json').read_text())
    stage(payload)


if __name__ == '__main__':
    main()
