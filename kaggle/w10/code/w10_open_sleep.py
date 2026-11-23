"""W10: unchanged W9d clockless sleep in World6 (principles 2 and 4).
L=4096; birth_period=128; clean W6 T_open checkpoints. No retuning.
"""
import os
import sys
sys.dont_write_bytecode = True
if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CURIO_LR'] = '0.004'
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if os.environ.get('_W10_MAIN') != '1':
        os.environ['_W10_MAIN'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)
import argparse
import contextlib
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import m10_w6 as m10
import w6_world
import w9_day_night as w9
import w9d_retina as retina
m7, m8, w4, w6 = m10.m7, m10.w6.m8, m10.w4, m10.w6
ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(120,132))
ARMS = ('nosleep','fixed_random','fixed_night','pressure_retina')
L_SCORE = m10.L_SCORE
PERIOD = 128
DATASET = 'ezzzzzx/pdp-code-w10'
CODE_FILES = tuple(dict.fromkeys(('w10_open_sleep.py',) + m10.CODE_FILES + retina.CODE_FILES))
dump, night = w9.dump, w9.night
RetinaPressure = retina.RetinaPressure
eyelid_checksum, eyelid_glance = retina.eyelid_checksum, retina.eyelid_glance
random_sleep_mask = w9.random_sleep_mask

class DayNightWorld(w6_world.World6):
    observe = w9.DayNightWorld.observe
    def __init__(self, seed, enabled=True):
        super().__init__(seed, PERIOD)
        self.night_enabled = enabled
        self.observation_rng = np.random.default_rng(seed + 60_000)
        self.t = 0
    def step(self, action, rng):
        room, channel, clean = super().step(action, rng)
        return room, channel, self.observe(clean, self.t)

def run_life(seed, arm, L=L_SCORE, night_enabled=True, trace=False, check_eyelid=False):
    if arm not in ARMS:
        raise ValueError(arm)
    m10._threads()
    start = time.perf_counter()
    world = DayNightWorld(seed, night_enabled)
    names = list(m10.IDENTITIES)
    index = {name: i for i, name in enumerate(names)}
    life = w6.OpenLife(seed, world, names, index, m10.N_RING, w4.N_CH, list(m10.w4_world.LEARN14))
    rng = np.random.default_rng(1_000 + seed)
    choice_rng = np.random.default_rng(10_000 + seed)
    values, counts, last_end = [0.] * 18, [0] * 18, [None] * 18
    v_max = None
    unvisited = set(range(18))
    plan, target, total, n_dec = [], None, 0., 0
    visit_pending = None
    channels, records, traces, sleep_visits = [], [], [], []
    points = set(w6.checkpoints_of(L))
    clock = {'eval': 0.}
    prev_cp = 0
    controller = RetinaPressure()
    eyelid_checks, input_bits = [], []
    random_mask, random_starts = random_sleep_mask(seed, L)
    input_rooms, choices = [], []
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
            want = (controller.wants_sleep() if arm == 'pressure_retina' else
                    t in random_starts if arm == 'fixed_random' else
                    night(t) if arm == 'fixed_night' else False)
            if want:
                sleep_left = min(8, L - t)
                sleep_visits.append({'t': t, 'length': sleep_left})
                visit_pending = w4._close_visit(life, visit_pending)
                if not checked:
                    snap = life.begin_sleep_visit()
                    sleep_learned = 0
        sleeping = sleep_left > 0
        # Trace reports controller state BEFORE this decision and AFTER selection.
        if trace and 180 <= t <= 260:
            rec = dict(t=t, night=int(night(t)), S=controller.S, r_bar=controller.w,
                       m_t=controller.m, asleep=int(sleeping), forced=int(controller.forced))
            traces.append(rec)
            print('TRACE ' + json.dumps(rec), flush=True)
        input_rooms.append(dict(t=t, slot=world.room, kind=world.kind[world.room], asleep=int(sleeping)))
        if sleeping:
            world.t = t
            control_state = (values, counts, last_end, v_max, unvisited, plan, target,
                             total, n_dec, visit_pending, choice_rng)
            before_hash = eyelid_checksum(life, controller, control_state) if check_eyelid else None
            signal = eyelid_glance(world, life, controller, index, rng) if arm == 'pressure_retina' else None
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
                choices.append(dict(t=t, target=target))
                plan = m10.w4_world.shortest(world.room, target, 18) + ['stay'] * 8
                total, n_dec = 0., 0
            action = plan.pop(0)
            world.t = t
            learnable_stay = action == 'stay' and world.is_learnable(world.room)
            _, channel, symbols = world.step(action, rng)
            signal = retina.retina_bits(symbols)
            life.count_preq = learnable_stay
            total += m8._progress_decision(life, symbols, channel, world.rooms[target])
            life.count_preq = False
            n_dec += 1
            controller.update(False, signal)
            if channel != 'corridor':
                visit_pending = channel
            channels.append(channel)
        input_bits.append(dict(t=t, r=signal, r_bar=controller.w,
                               m_t=controller.m, S=controller.S, asleep=int(sleeping),
                               forced=int(controller.forced)))
        t += 1
        world.maybe_birth(t)
        if t in points:
            records.append(w6.take_checkpoint(life, t, prev_cp, clock))
            prev_cp = t
        if not sleeping and not plan:
            finish_room()
            n_dec = 0
    if n_dec:
        finish_room()
    if visit_pending is not None:
        life.snapshot(visit_pending)
    scores = [r['open'] for r in records]
    slept = sum(c == 'sleep' for c in channels)
    night_slept = sum(c == 'sleep' and night(i) for i, c in enumerate(channels))
    assert world.birth_log == w6_world.expected_birth_log(seed, PERIOD, L)
    assert not life.exam.ids & life.replay.ids and life.exam_drawn == 0
    assert len(channels) == L and len(records) == 8
    row = dict(seed=seed, arm=arm, L=L, period=128, night_enabled=night_enabled,
               T_open=float(np.mean(scores)), final=scores[-1], checkpoints=records,
               sleep_share=slept / L, sleep_decisions=slept, night_sleep_decisions=night_slept,
               night_sleep_share=night_slept / slept if slept else None,
               sleep_visits=sleep_visits, random_starts=[],
               birth_log=world.birth_log, sleep_checks=life.sleep_checks,
               finite=all(math.isfinite(v) for cp in records for v in [cp['open']] + [r['bits'] for r in cp['rooms']]),
               wall_s=time.perf_counter() - start)
    if arm == 'fixed_random':
        assert np.array_equal(random_mask, np.array(channels) == 'sleep')
    if arm == 'fixed_night':
        assert all((c == 'sleep') == night(i) for i,c in enumerate(channels))
    row['input_retina'] = input_bits if arm == 'pressure_retina' else []
    row['input_rooms'] = input_rooms
    row['choices'] = choices
    row['preq_bits'], row['preq_symbols'] = life.preq_bits, life.preq_symbols
    row['learner_checksum'] = eyelid_checksum(life, SimpleNamespace(), ())
    day_sleeps = [r for r in input_rooms if r['asleep'] and not night(r['t'])]
    row['day_sleep_decisions'] = len(day_sleeps)
    row['day_noise_sleep_decisions'] = sum(r['kind'] == 'N' for r in day_sleeps)
    row['day_noise_sleep_share'] = row['day_noise_sleep_decisions'] / len(day_sleeps) if day_sleeps else None
    row['random_starts'] = random_starts if arm == 'fixed_random' else []
    row['eyelid_checks'] = eyelid_checks
    if trace:
        row['trace'] = traces
    if not row['finite'] or not math.isfinite(row['T_open']):
        raise RuntimeError('nonfinite score')
    return row

def estimate_kernel(payload):
    """Four one-threaded arms on four Kaggle CPUs; 2x timing allowance.

    L512 times are scaled to L4096. Full measured lives bound nosleep and
    pressure_retina from below. Each arm has its own worker, so makespan is
    the longest arm, with a 2x allowance plus 30 seconds for pool startup.
    """
    durations = {arm: 8 * payload['finite_L512'][arm]['wall_s'] for arm in ARMS}
    durations['nosleep'] = max(durations['nosleep'], payload['baseline']['wall_s'],
                               payload['baseline']['reference_wall_s'])
    durations['pressure_retina'] = max(durations['pressure_retina'],
                                       payload['pressure_retina']['wall_s'])
    parallel = m10.makespan(list(durations.values()), 4)
    wall = 30. + 2 * parallel
    return dict(kernel_wall_s=wall, limit_s=5100, within_limit=wall <= 5100,
                workers=4, margin=2., startup_s=30.,
                full_life_estimates_s=durations, parallel_life_s=parallel,
                sequential_life_s=sum(durations.values()))


def smoke(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(ok=False, workers=1, L=L_SCORE, period=PERIOD)
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
            # M10 delegates this policy directly to W6. Disable only its sleep dose.
            captured = []
            original_life = w6.OpenLife
            def capture(*args, **kwargs):
                obj = original_life(*args, **kwargs)
                captured.append(obj)
                return obj
            with patch.object(w4, 'sleep_decisions', return_value=0), patch.object(w6, 'OpenLife', capture):
                ref = m10.run_life(120, 'prog_age_s', L_SCORE, PERIOD)
            row = run_life(120, 'nosleep', L_SCORE, night_enabled=False)
            ref_hash = eyelid_checksum(captured[0], SimpleNamespace(), ())
            exact = row['checkpoints'] == ref['checkpoints']
            payload['baseline'] = dict(seed=120, L=L_SCORE, T_open=row['T_open'],
                reference_T_open=ref['T_open'], checkpoints_exact=exact,
                preq_bits_exact=row['preq_bits'] == ref['preq_bits'],
                learner_exact=row['learner_checksum'] == ref_hash,
                learner_checksum=row['learner_checksum'], reference_checksum=ref_hash,
                reference_wall_s=ref['wall_s'], wall_s=row['wall_s'])
            print('BASELINE ' + json.dumps(payload['baseline']), flush=True)
            assert exact and row['preq_bits'] == ref['preq_bits'] and row['learner_checksum'] == ref_hash
            dump(out / 'baseline_nosleep_L4096.json', row)
            timed = run_life(120, 'pressure_retina', L_SCORE, check_eyelid=True)
            dump(out / 'pressure_retina_L4096.json', timed)
            payload['per_day'] = []
            for day in range(L_SCORE // 256):
                sleeps = [r for r in timed['input_retina'] if r['t']//256 == day and r['asleep']]
                info = dict(day=day+1, day_sleep=sum(not night(r['t']) for r in sleeps),
                    night_sleep=sum(night(r['t']) for r in sleeps), forced=sum(r['forced'] for r in sleeps))
                payload['per_day'].append(info)
                print('DAY ' + json.dumps(info), flush=True)
            payload['pressure_retina'] = {k:timed[k] for k in ('T_open','sleep_share','night_sleep_share',
                'day_sleep_decisions','day_noise_sleep_decisions','day_noise_sleep_share','wall_s')}
            print('RETINA ' + json.dumps(payload['pressure_retina']), flush=True)
            checks = timed['eyelid_checks']
            assert checks and all(c['intact'] and c['before']==c['after'] for c in checks)
            payload['eyelid_checksum'] = dict(decisions=len(checks), intact=len(checks))
            print('EYELID ' + json.dumps(payload['eyelid_checksum']), flush=True)
            # Independent reconstruction, using the W9d checker unchanged.
            import importlib.util
            spec = importlib.util.spec_from_file_location('w10_summary', ROOT/'kaggle/w10/summarize.py')
            summary = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(summary)
            summary.validate_retina(timed, {r['t'] for r in timed['input_rooms'] if r['asleep']})
            payload['controller_reconstruction'] = True
            payload['finite_L512'] = {}
            for arm in ARMS:
                row = run_life(120, arm, 512)
                payload['finite_L512'][arm] = {k:row[k] for k in ('T_open','finite','wall_s','sleep_share','night_sleep_share')}
                print('FINITE ' + arm + ' ' + json.dumps(payload['finite_L512'][arm]), flush=True)
            # Four independent arm workers run concurrently on four CPUs.
            payload['estimate'] = estimate_kernel(payload)
            estimate = payload['estimate']['kernel_wall_s']
            print('WALL_ESTIMATE ' + json.dumps(payload['estimate']), flush=True)
            assert estimate<=5100, 'split with exact checkpoints required before staging'
            payload['ok'] = True
            print('SMOKE_OK', flush=True)
    except Exception as exc:
        payload['error'] = str(exc)
        raise
    finally:
        payload['wall_s'] = time.perf_counter()-started
        dump(out/'smoke.json',payload)
    return payload


def job(args):
    row = run_life(*args)
    print(json.dumps({k:row[k] for k in ('seed','arm','T_open','sleep_share','wall_s')}),flush=True)
    return row


def kernel_main(seed, out, arms, kernel_id):
    if seed not in SEEDS or tuple(arms)!=ARMS or kernel_id!=f'ezzzzzx/w10-s{seed}':
        raise ValueError('kernel jobs')
    start=time.perf_counter()
    with get_context('spawn').Pool(4) as pool:
        runs=pool.map(job,[(seed,arm,L_SCORE) for arm in arms])
    dump(Path(out)/f'w10_s{seed}.json',dict(seed=seed,kernel_id=kernel_id,workers=4,
        runs=runs,wall_s=time.perf_counter()-start,
        code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in CODE_FILES}))


def stage(payload):
    assert payload['ok'] and payload['estimate']['within_limit']
    base=ROOT/'kaggle/w10'
    code=base/'code'
    code.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name in CODE_FILES:
        shutil.copy2(ROOT/name,code/name)
        hashes[name]=hashlib.sha256((code/name).read_bytes()).hexdigest()
    dump(code/'dataset-metadata.json',dict(title='pdp code w10',id=DATASET,licenses=[{'name':'other'}]))
    dump(base/'code-sha256.json',hashes)
    manifest=[]
    for seed in SEEDS:
        folder=base/f's{seed}'
        folder.mkdir(parents=True,exist_ok=True)
        kid=f'ezzzzzx/w10-s{seed}'
        dump(folder/'kernel-metadata.json',dict(id=kid,title=kid.split('/')[1],code_file='kernel.py',
            language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,
            enable_internet=False,dataset_sources=[DATASET],competition_sources=[],kernel_sources=[],model_sources=[]))
        src=f'''import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CURIO_LR']='0.004'
os.environ['PYTHONHASHSEED']='0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
if os.environ.get('_W10_KERNEL')!='1':
    os.environ['_W10_KERNEL']='1'
    os.execv(sys.executable,[sys.executable]+sys.argv)
if __name__=='__main__':
    hits=sorted(Path('/kaggle/input').rglob('w10_open_sleep.py'))
    if len(hits)!=1:
        raise RuntimeError('expected one W10 code dataset')
    code=str(hits[0].parent)
    sys.path.insert(0,code)
    os.environ['PYTHONPATH']=code
    import w10_open_sleep as w10
    w10.kernel_main({seed},'/kaggle/working',{ARMS!r},{kid!r})
'''
        (folder/'kernel.py').write_text(src)
        manifest.append(dict(seed=seed,folder=f's{seed}',kernel_id=kid,arms=list(ARMS),output=f'w10_s{seed}.json'))
    dump(base/'manifest.json',manifest)
    print(f'STAGED {len(manifest)} private CPU kernels',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('smoke','stage'))
    p.add_argument('--out',type=Path,default=ROOT/'results/w10')
    args=p.parse_args()
    payload=smoke(args.out) if args.command=='smoke' else json.loads((args.out/'smoke.json').read_text())
    stage(payload)

if __name__=='__main__':
    main()
