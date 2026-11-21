"""W9c: clock-blind homeostasis with a frozen, unlearned eyelid glance.

Principles 2/4: replay improves the compressor; room choice uses prog_age_s.
W9b supplies all old arms unchanged. New-arm sleep receives a stationary room
observation before replay; only its EMA consumes that glance. Median history
contains wake EMAs only. Transitions occur at eight-decision sleep boundaries;
traces show pre-decision state, input_bits show post-decision state.
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
    if os.environ.get('_W9C_MAIN') != '1':
        os.environ['_W9C_MAIN'] = '1'
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
import w9b_poor_input as w9b
import io
import pickle
import torch
SEEDS = tuple(range(84, 96))
ARMS = ('fixed_random', 'fixed_night', 'pressure_bits', 'pressure_eyelid')
L_SCORE = 2048
DATASET = 'ezzzzzx/pdp-code-w9c'
CODE_FILES = ('w9c_eyelid.py',) + w9b.CODE_FILES
dump, night = w9.dump, w9.night
DayNightWorld, Pressure = w9.DayNightWorld, w9.Pressure
random_sleep_mask = w9.random_sleep_mask


class EyelidPressure:
    """Clock blind; observe updates only b, update accounts for the decision."""
    def __init__(self):
        self.S = 0
        self.w = None
        self.m = None
        self.history = deque(maxlen=256)
        self.asleep = False
        self.forced = False

    def wants_sleep(self):
        # Called only at visit boundaries. Forced episodes release at <=128.
        if self.forced:
            self.forced = self.S > 128
            self.asleep = self.forced
            return self.asleep
        if self.S >= 256:
            self.forced = self.asleep = True
        elif self.asleep:
            self.asleep = self.S > 0 and self.w > 1.5 * self.m
        else:
            self.asleep = self.S >= 32 and self.w is not None and self.w > 1.5 * self.m
        return self.asleep

    def observe(self, bits):
        self.w = bits if self.w is None else .9 * self.w + .1 * bits

    def update(self, sleeping, gain=None):
        self.S = max(0, self.S - 3) if sleeping else self.S + 1
        if not sleeping:
            self.observe(gain)
            self.history.append(self.w)
            self.m = float(np.median(self.history))


class StatePickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor):
            a = obj.detach().cpu().numpy()
            return ('tensor', str(a.dtype), a.shape, a.tobytes())
        if isinstance(obj, np.ndarray):
            return ('array', str(obj.dtype), obj.shape, obj.tobytes())
        return None


def eyelid_checksum(life, controller, room_controller):
    # Timers are instrumentation. World streams advance when a glance arrives.
    # Cortex, stores, RNGs, score caches and all control state except b are hashed.
    state = {k: v for k, v in vars(life).items() if k not in ('world', 'score_s', 'learn_s')}
    control = {k: v for k, v in vars(controller).items() if k != 'w'}
    buf = io.BytesIO()
    StatePickler(buf, protocol=5).dump((state, control, room_controller))
    return hashlib.sha256(buf.getvalue()).hexdigest()


def eyelid_glance(world, life, controller, index, rng):
    _, channel, symbols = world.step('stay', rng)
    glance = SimpleNamespace(channel=index[channel], symbols=np.asarray(symbols, dtype=np.int64),
                             prev=life.cortex.prev_symbol)
    bits = life._score_one(glance) / len(glance.symbols)
    controller.observe(bits)
    return bits


def run_life(seed, arm, L=L_SCORE, night_enabled=True, trace=False, check_eyelid=False):
    if arm != 'pressure_eyelid':
        return w9b.run_life(seed, arm, L, night_enabled, trace)
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
    controller = EyelidPressure()
    eyelid_checks, input_bits = [], []
    sleep_left, t = 0, 0
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
            want = controller.wants_sleep()
            if want:
                sleep_left = min(8, L - t)
                sleep_visits.append({'t': t, 'length': sleep_left})
                visit_pending = w4._close_visit(life, visit_pending)
                if not checked:
                    snap = life.begin_sleep_visit()
                    sleep_learned = 0
        sleeping = sleep_left > 0
        # Trace reports controller state BEFORE this decision and AFTER selection.
        if trace and (180 <= t <= 260 or 1716 <= t <= 1796):
            rec = dict(t=t, night=int(night(t)), S=controller.S, b_t=controller.w,
                       m_t=controller.m, asleep=int(sleeping), forced=int(controller.forced))
            traces.append(rec)
            print('TRACE ' + json.dumps(rec), flush=True)
        if sleeping:
            world.t = t
            control_state = (values, counts, last_end, v_max, unvisited, plan, target,
                             total, n_dec, visit_pending, choice_rng)
            before_hash = eyelid_checksum(life, controller, control_state) if check_eyelid else None
            signal = eyelid_glance(world, life, controller, index, rng)
            if check_eyelid:
                after_hash = eyelid_checksum(life, controller, control_state)
                assert before_hash == after_hash, 'eyelid changed learner/controller state'
                eyelid_checks.append(dict(t=t, before=before_hash, after=after_hash, intact=True))
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
            before = life._score_one(glance)
            total += m8._progress_decision(life, symbols, channel, world.rooms[target])
            n_dec += 1
            signal = before / len(glance.symbols)
            controller.update(False, signal)
            if channel != 'corridor':
                visit_pending = channel
            channels.append(channel)
        input_bits.append(dict(t=t, bits_per_symbol=signal, b_t=controller.w,
                               m_t=controller.m, S=controller.S, asleep=int(sleeping),
                               forced=int(controller.forced)))
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
    row = dict(seed=seed, arm=arm, L=L, period=128, night_enabled=night_enabled,
               T14=float(np.mean(scores)), W14=scores[-1], checkpoints=records,
               sleep_share=slept / L, sleep_decisions=slept, night_sleep_decisions=night_slept,
               night_sleep_share=night_slept / slept if slept else None,
               sleep_visits=sleep_visits, random_starts=[],
               swap_log=world.swap_log, sleep_checks=life.sleep_checks,
               finite=all(math.isfinite(v) for cp in records for v in cp['held'].values()),
               wall_s=time.perf_counter() - start)
    row['input_bits'] = input_bits
    row['eyelid_checks'] = eyelid_checks
    if trace:
        row['trace'] = traces
    if not row['finite'] or not math.isfinite(row['T14']):
        raise RuntimeError('nonfinite score')
    return row


def smoke(out):
    """One foreground process; full scored lives are reserved for Kaggle."""
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
            for arm, refarm in (('nosleep', 'age_nosleep'), ('fixed20_compat', 'age_fixed20'),
                                ('fixed_random', 'fixed_random')):
                row = run_life(24, arm, 256, night_enabled=False)
                ref = (w9.run_life(24, refarm, 256, night_enabled=False)
                       if arm == 'fixed_random' else m9.run_life(24, refarm, 256))
                gap = abs(row['T14'] - ref['T14'])
                cp_gap = max(abs(a['W14'] - b['W14']) for a, b in zip(row['checkpoints'], ref['checkpoints']))
                info = dict(T14=row['T14'], reference_T14=ref['T14'],
                            reference='W9' if arm == 'fixed_random' else 'M9',
                            abs=gap, checkpoint_max_abs=cp_gap,
                            sleep_share=row['sleep_share'], wall_s=row['wall_s'])
                print('BASELINE ' + arm + ' ' + json.dumps(info), flush=True)
                assert gap <= 1e-6 and cp_gap <= 1e-6
                payload['baselines'][arm] = info
            timed = run_life(84, 'pressure_eyelid', 2048, trace=True, check_eyelid=True)
            dump(out / 'pressure_eyelid_L2048.json', timed)
            payload['eyelid_L2048'] = {k: timed[k] for k in
                ('sleep_share', 'night_sleep_share', 'sleep_decisions', 'night_sleep_decisions', 'wall_s')}
            print('EYELID ' + json.dumps(payload['eyelid_L2048']), flush=True)
            days = []
            for day in range(8):
                sleeps = [r for r in timed['input_bits'] if r['t'] // 256 == day and r['asleep']]
                info = dict(day=day + 1,
                            day_sleep=sum(not night(r['t']) for r in sleeps),
                            night_sleep=sum(night(r['t']) for r in sleeps),
                            forced_day=sum(r['forced'] and not night(r['t']) for r in sleeps),
                            forced_night=sum(r['forced'] and night(r['t']) for r in sleeps),
                            forced=sum(r['forced'] for r in sleeps))
                days.append(info)
                print('DAY ' + json.dumps(info), flush=True)
            payload['per_day'] = days
            late_sleep = sum(r['day_sleep'] + r['night_sleep'] for r in days[2:])
            late_night = sum(r['night_sleep'] for r in days[2:])
            share = late_night / late_sleep if late_sleep else None
            payload['gate'] = dict(days=[3, 8], sleep_decisions=late_sleep,
                                   night_sleep_decisions=late_night, night_sleep_share=share,
                                   threshold=.5, meets=share is not None and share >= .5)
            print('GATE ' + json.dumps(payload['gate']), flush=True)
            # Preregistered stop: do not proceed to finite checks or staging.
            assert payload['gate']['meets'], 'days 3-8 night sleep share < 0.5'
            checks = timed['eyelid_checks']
            assert checks and all(c['intact'] and c['before'] == c['after'] for c in checks)
            payload['eyelid_checksum'] = dict(decisions=len(checks), intact=sum(c['intact'] for c in checks),
                                              first=checks[0], last=checks[-1])
            print('EYELID_CHECKSUM ' + json.dumps(payload['eyelid_checksum']), flush=True)
            payload['finite_L512'] = {}
            for arm in ARMS:
                row = run_life(84, arm, 512)
                info = {k: row[k] for k in ('T14', 'sleep_share', 'night_sleep_share', 'wall_s', 'finite')}
                payload['finite_L512'][arm] = info
                print('FINITE ' + arm + ' ' + json.dumps(info), flush=True)
            # Sequential four-arm bound, 2x allowance for reservoir growth.
            estimate = sum(v['wall_s'] for v in payload['finite_L512'].values()) * 4 * 2
            payload['estimate'] = dict(kernel_wall_s=estimate, limit_s=85 * 60,
                                       within_limit=estimate <= 85 * 60, factor=8,
                                       pressure_eyelid_L2048_wall_s=timed['wall_s'])
            print('WALL_ESTIMATE ' + json.dumps(payload['estimate']), flush=True)
            assert estimate <= 85 * 60, 'kernel estimate exceeds 85 minutes'
            payload['ok'] = True
            print('SMOKE_OK', flush=True)
    except Exception as exc:
        payload['error'] = str(exc)
        raise
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
    dump(Path(out) / f'w9c_s{seed}{suffix}.json', dict(seed=seed, kernel_id=kernel_id,
         workers=min(4, len(arms)), runs=runs, wall_s=time.perf_counter() - start,
         code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES}))


def stage(payload):
    assert payload['ok']
    base = ROOT / 'kaggle/w9c'
    code = base / 'code'
    code.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
        hashes[name] = hashlib.sha256((code / name).read_bytes()).hexdigest()
    dump(code / 'dataset-metadata.json', dict(title='pdp code w9c', id=DATASET, licenses=[{'name': 'other'}]))
    dump(base / 'code-sha256.json', hashes)
    assert payload['estimate']['within_limit']
    groups = [('', ARMS)]
    manifest = []
    for seed in SEEDS:
        for suffix, arms in groups:
            folder = base / f's{seed}' / (suffix[1:] if suffix else '')
            folder.mkdir(parents=True, exist_ok=True)
            kid = f'ezzzzzx/w9c-s{seed}{suffix.replace("_", "-")}'
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
if os.environ.get('_W9C_KERNEL') != '1':
    os.environ['_W9C_KERNEL'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)
if __name__ == '__main__':
    hits = sorted(Path('/kaggle/input').rglob('w9c_eyelid.py'))
    if len(hits) != 1:
        raise RuntimeError('expected exactly one W9c dataset')
    code = str(hits[0].parent)
    sys.path.insert(0, code)
    os.environ['PYTHONPATH'] = code
    import w9c_eyelid as w9
    w9.kernel_main({seed}, '/kaggle/working', {arms!r}, {kid!r}, {suffix!r})
'''
            (folder / 'kernel.py').write_text(src)
            manifest.append(dict(seed=seed, folder=str(folder.relative_to(base)), arms=list(arms),
                                 kernel_id=kid, output=f'w9c_s{seed}{suffix}.json'))
    dump(base / 'manifest.json', manifest)
    print(f'STAGED {len(manifest)} private CPU kernels', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('smoke', 'stage'))
    p.add_argument('--out', type=Path, default=ROOT / 'results/w9c')
    a = p.parse_args()
    payload = smoke(a.out) if a.command == 'smoke' else json.loads((a.out / 'smoke.json').read_text())
    stage(payload)


if __name__ == '__main__':
    main()
