"""W9: state-triggered sleep in World5 with poor night observations.

Principles 2/4: replay improves the compressor; wake room choice rewards exam
compression progress. No scoring-seed tuning. Decision t is zero-based; swaps
and checkpoints remain W5's after-decision (one-based) events. Glance gain uses
M7's frozen zero-state scorer, with identical observed symbols/channel/prev
before and after the wake update. EMA starts at the first gain; its median
includes the current EMA and up to 255 previous wake EMAs. Sleep pauses a room
plan; its reward denominator counts wake decisions as in M9. A pressure episode
ends at S=0, rounded up to the end of an eight-decision visit.
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
    if os.environ.get('_W9_MAIN') != '1':
        os.environ['_W9_MAIN'] = '1'
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
SEEDS = tuple(range(60, 72))
ARMS = ('nosleep', 'fixed_random', 'fixed_night', 'pressure', 'pressure_opp')
L_SCORE = 2048
DATASET = 'ezzzzzx/pdp-code-w9'
CODE_FILES = ('w9_day_night.py',) + m9.CODE_FILES


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def night(t):
    return t % 256 >= 192


class DayNightWorld(World5):
    """Only step observations are corrupted; continuation remains clean."""
    def __init__(self, seed, enabled=True):
        super().__init__(seed, 128)
        self.night_enabled = enabled
        self.observation_rng = np.random.default_rng(seed + 60_000)
        self.t = 0

    def observe(self, symbols, t):
        out = np.asarray(symbols, dtype=np.int64).copy()
        if self.night_enabled and night(t):
            mask = self.observation_rng.random(len(out)) < 0.8
            out[mask] = self.observation_rng.integers(len(ALPHA), size=int(mask.sum()))
        return out

    def step(self, action, rng):
        room, channel, clean = super().step(action, rng)
        return room, channel, self.observe(clean, self.t)


class Pressure:
    """No clock input: S and wake gain history are its only inputs."""
    def __init__(self, opportunity):
        self.opportunity = opportunity
        self.S = 0
        self.w = None
        self.m = None
        self.history = deque(maxlen=256)
        self.asleep = False

    def wants_sleep(self):
        if self.asleep:
            self.asleep = self.S > 0
        if not self.asleep:
            if self.opportunity:
                self.asleep = (self.S >= 128 or
                               (self.S >= 32 and self.w is not None and self.w < self.m))
            else:
                self.asleep = self.S >= 64
        return self.asleep

    def update(self, sleeping, gain=None):
        if sleeping:
            self.S = max(0, self.S - 4)
        else:
            self.S += 1
            if gain is not None:
                self.w = gain if self.w is None else .9 * self.w + .1 * gain
                self.history.append(self.w)
                self.m = float(np.median(self.history))


def random_sleep_mask(seed, L):
    """Uniform non-overlapping integer starts via a uniform weak composition.

    Removing the mandatory seven positions after each ordered start gives a
    bijection to subsets of L-7*n positions; every feasible set is equiprobable.
    """
    n = L // 32
    rng = np.random.default_rng(seed + 61_000)
    starts = np.sort(rng.choice(L - 7 * n, n, replace=False)) + 7 * np.arange(n)
    mask = np.zeros(L, dtype=bool)
    for start in starts:
        mask[start:start + 8] = True
    assert int(mask.sum()) == n * 8
    return mask, [int(s) for s in starts]


def run_life(seed, arm, L=L_SCORE, night_enabled=True, trace=False):
    if arm not in ARMS + ('fixed20_compat',):
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
    controller = Pressure(arm == 'pressure_opp')
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
            before = life._score_one(glance) if arm == 'pressure_opp' else None
            total += m8._progress_decision(life, symbols, channel, world.rooms[target])
            n_dec += 1
            gain = before - life._score_one(glance) if before is not None else None
            controller.update(False, gain)
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
    if trace:
        row['trace'] = traces
    if not row['finite'] or not math.isfinite(row['T14']):
        raise RuntimeError('nonfinite score')
    return row


def smoke(out):
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
            for arm, refarm in (('nosleep', 'age_nosleep'), ('fixed20_compat', 'age_fixed20')):
                row = run_life(24, arm, 256, night_enabled=False)
                ref = m9.run_life(24, refarm, 256)
                gap = abs(row['T14'] - ref['T14'])
                cp_gap = max(abs(a['W14'] - b['W14']) for a, b in zip(row['checkpoints'], ref['checkpoints']))
                info = dict(T14=row['T14'], M9=ref['T14'], abs=gap, checkpoint_max_abs=cp_gap,
                            sleep_share=row['sleep_share'], wall_s=row['wall_s'])
                print('BASELINE ' + arm + ' ' + json.dumps(info), flush=True)
                assert gap <= 1e-6 and cp_gap <= 1e-6
                payload['baselines'][arm] = info
            world = DayNightWorld(24)
            clean = world.glance_room(world.rooms[0])
            day, dark = world.observe(clean, 0), world.observe(clean, 192)
            # Numeric symbol IDs retain spaces and make all 64 positions visible.
            print('GLANCE_DAY ' + json.dumps(day[:64].tolist()), flush=True)
            print('GLANCE_NIGHT ' + json.dumps(dark[:64].tolist()), flush=True)
            changed = int(np.sum(day != dark))
            print('GLANCE_CHANGED ' + str(changed), flush=True)
            assert changed > 0 and np.array_equal(day, clean)
            payload['glance_changed'] = changed
            timed = run_life(24, 'pressure_opp', 512, trace=True)
            dump(out / 'pressure_opp_L512.json', timed)
            payload['finite_L256'] = {}
            for arm in ARMS:
                row = run_life(24, arm, 256)
                info = {k: row[k] for k in ('T14', 'sleep_share', 'night_sleep_share', 'wall_s', 'finite')}
                payload['finite_L256'][arm] = info
                print('FINITE ' + arm + ' ' + json.dumps(info), flush=True)
            # Conservative sequential five-arm bound, 2x allowance for reservoir growth.
            estimate = timed['wall_s'] * (2048 / 512) * 2 * 5
            payload['estimate'] = dict(kernel_wall_s=estimate, split=estimate > 80 * 60,
                                       pressure_opp_L512_wall_s=timed['wall_s'], factor=40)
            print('WALL_ESTIMATE ' + json.dumps(payload['estimate']), flush=True)
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
    dump(Path(out) / f'w9_s{seed}{suffix}.json', dict(seed=seed, kernel_id=kernel_id,
         workers=min(4, len(arms)), runs=runs, wall_s=time.perf_counter() - start,
         code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES}))


def stage(payload):
    assert payload['ok']
    base = ROOT / 'kaggle/w9'
    code = base / 'code'
    code.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)
        hashes[name] = hashlib.sha256((code / name).read_bytes()).hexdigest()
    dump(code / 'dataset-metadata.json', dict(title='pdp code w9', id=DATASET, licenses=[{'name': 'other'}]))
    dump(base / 'code-sha256.json', hashes)
    groups = [(f'-{arm}', (arm,)) for arm in ARMS] if payload['estimate']['split'] else [('', ARMS)]
    manifest = []
    for seed in SEEDS:
        for suffix, arms in groups:
            folder = base / f's{seed}' / (suffix[1:] if suffix else '')
            folder.mkdir(parents=True, exist_ok=True)
            kid = f'ezzzzzx/w9-day-night-s{seed}{suffix.replace("_", "-")}'
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
if os.environ.get('_W9_KERNEL') != '1':
    os.environ['_W9_KERNEL'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)
if __name__ == '__main__':
    hits = sorted(Path('/kaggle/input').rglob('w9_day_night.py'))
    if len(hits) != 1:
        raise RuntimeError('expected exactly one W9 dataset')
    code = str(hits[0].parent)
    sys.path.insert(0, code)
    os.environ['PYTHONPATH'] = code
    import w9_day_night as w9
    w9.kernel_main({seed}, '/kaggle/working', {arms!r}, {kid!r}, {suffix!r})
'''
            (folder / 'kernel.py').write_text(src)
            manifest.append(dict(seed=seed, folder=str(folder.relative_to(base)), arms=list(arms),
                                 kernel_id=kid, output=f'w9_s{seed}{suffix}.json'))
    dump(base / 'manifest.json', manifest)
    print(f'STAGED {len(manifest)} private CPU kernels', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('smoke', 'stage'))
    p.add_argument('--out', type=Path, default=ROOT / 'results/w9')
    a = p.parse_args()
    payload = smoke(a.out) if a.command == 'smoke' else json.loads((a.out / 'smoke.json').read_text())
    stage(payload)


if __name__ == '__main__':
    main()
