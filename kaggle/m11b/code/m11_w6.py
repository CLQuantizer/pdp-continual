"""M11-W6: delayed counterfactual sleep credit. Fixed preregistered constants.

CF horizons count wake decisions (travel included); subsequent sleep pauses them.
CF room comparison reward = M8 visit reward / room exam symbols at visit end.
Immediate-credit control keeps M9's original bits/decision scale.
Learning in the inherited trace/Adam step is deterministic (no RNG draws).
"""
import os
import sys
if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CURIO_LR'] = '0.004'
    os.environ['PYTHONHASHSEED'] = '0'
    if os.environ.get('_M11_MAIN') != '1':
        os.environ['_M11_MAIN'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)
import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from multiprocessing import get_context
import numpy as np
import torch
import curio
import m7_w3 as m7
import m8_w5 as m8
import m9_w5 as m9
import w4_headroom as w4
import w4_world
import w6_headroom as w6
from w6_world import World6, expected_birth_log

ROOT = Path(__file__).resolve().parent
SEEDS = tuple(range(36, 48))
ARMS = ('age_fixed20', 'age_sleeparm', 'age_sleep_cf', 'age_nosleep')
H = 64
L_SCORE = 4096
PERIOD = 128
DATASET_ID = 'ezzzzzx/pdp-code-m11w6'
CODE_FILES = tuple(dict.fromkeys(w6.CODE_FILES + ('m9_w5.py', 'm11_w6.py')))
GAPS = (('age_sleep_cf', 'age_sleeparm'), ('age_sleep_cf', 'age_fixed20'), ('age_fixed20', 'age_nosleep'))
assert curio.EPS == .1 and curio.BETA == .3 and m8.TAU == 256 and m7.DWELL == 8


def exam_bits(cortex, glances):
    return float(m7.score_glances(cortex.model, cortex.channel_emb, glances).sum())


class Shadow:
    def __init__(self, life, t0, visit):
        # Full life copy includes optimizer aliases, traces, reservoirs and all RNGs.
        # The copied world never runs: live observations alone drive shadow learning.
        self.life = copy.deepcopy(life)
        self.t0 = t0
        self.visit = visit
        self.wake = 0
        self.ready = False
        self.torch_rng = torch.get_rng_state().clone()

    def observe(self, symbols, channel):
        own = self.life
        # The inherited learning rule has no randomness. Restore/capture explicitly
        # so adding a stochastic operation cannot contaminate the live RNG stream.
        live_rng = torch.get_rng_state()
        torch.set_rng_state(self.torch_rng)
        try:
            if channel == 'corridor':
                own.corridor_decision(symbols)
            else:
                own.room_decision(symbols, channel)
        finally:
            self.torch_rng = torch.get_rng_state().clone()
            torch.set_rng_state(live_rng)
        self.wake += 1


def run_life(seed, arm, L=L_SCORE, trace=False, noop_sleep=False):
    if arm not in ARMS:
        raise ValueError(arm)
    w6._threads()
    start = time.perf_counter()
    world = World6(seed, PERIOD)
    names = list(w4_world.IDENTITIES)
    life = w6.OpenLife(seed, world, names, {n: i for i, n in enumerate(names)}, 18, w4.N_CH, list(w4_world.LEARN14))
    rng = np.random.default_rng(1000 + seed)
    choices = np.random.default_rng(10000 + seed)
    learned_sleep = arm in ('age_sleeparm', 'age_sleep_cf')
    cf = arm == 'age_sleep_cf'
    n = 19 if learned_sleep else 18
    values, counts, last_end = [0.] * n, [0] * n, [None] * n
    vmax = None
    unvisited = set(range(18))
    first_sleep = learned_sleep
    fixed_plan = m7.sleep_plan(L, w4.sleep_decisions(L) if arm == 'age_fixed20' else 0)
    si = 0
    t = 0
    slept = 0
    checkpoints, visits, pending = [], [], []
    points = set(w6.checkpoints_of(L))
    clock = {'eval': 0.}
    prev_cp = 0
    visit_pending = None
    max_pending = 0

    def update(target, reward):
        nonlocal vmax
        if not math.isfinite(reward):
            raise RuntimeError('nonfinite reward')
        m7.ema(values, counts, target, reward)
        vmax = reward if vmax is None else max(vmax, reward)

    def deliver(shadow):
        gs = life.exam.flat()  # SAME current live exam items for both cortices
        symbols = sum(len(g.symbols) for g in gs)
        live = exam_bits(life.cortex, gs)
        other = exam_bits(shadow.life.cortex, gs)
        reward = (other - live) / symbols / 8 if symbols else 0.
        update(18, reward)
        rec = shadow.visit
        rec.update(delivered=t, wake_horizon=shadow.wake, exam_symbols=symbols,
                   live_bits=live, shadow_bits=other, reward=reward,
                   bits_per_decision=(other-live)/8, values_delivered=list(values),
                   sleep_share_delivered=slept/t)
        if noop_sleep and abs(other - live) > 1e-9:
            raise RuntimeError(f'no-op shadow mismatch {other-live}')
        if trace:
            print('SLEEP_DELIVER ' + json.dumps(rec), flush=True)

    def on_decision(wake):
        nonlocal prev_cp
        world.maybe_birth(t)
        for shadow in list(pending):
            if shadow.ready and ((wake and shadow.wake >= H) or t == L):
                deliver(shadow)
                pending.remove(shadow)
        if t in points:
            checkpoints.append(w6.take_checkpoint(life, t, prev_cp, clock))
            prev_cp = t

    while t < L:
        fixed_due = arm == 'age_fixed20' and w4.sleep_due(t, L, fixed_plan, si, 18)
        if fixed_due:
            target, mode = 18, 'fixed'
        elif learned_sleep:
            target, mode, rows, first_sleep = m9.choose_sleeparm(
                world.room, unvisited, first_sleep, values, counts, last_end, vmax, t, world.rooms, choices)
        else:
            target, mode, rows = m8.choose_room(world.room, unvisited, values, counts, last_end, vmax, t, world.rooms, choices, True)
        t0 = t
        if target == 18:
            visit_pending = w4._close_visit(life, visit_pending)
            rec = dict(kind='sleep', t0=t0, mode=mode, values_start=list(values),
                       sleep_share_start=slept/t if t else 0.)
            visits.append(rec)
            shadow = Shadow(life, t0, rec) if cf else None
            if shadow:
                pending.append(shadow)
                max_pending = max(max_pending, len(pending))
            total = 0.
            snap = life.begin_sleep_visit()
            for _ in range(min(8, L-t)):
                before = m9.whole_exam_bits(life) if arm == 'age_sleeparm' else 0.
                if not noop_sleep:
                    life.sleep_decision()
                if arm == 'age_sleeparm':
                    total += before - m9.whole_exam_bits(life)
                t += 1
                slept += 1
                on_decision(False)
            life.end_sleep_visit(snap)
            rec['end'] = t
            if learned_sleep:
                last_end[18] = t
            if shadow:
                shadow.ready = True
                if t == L:
                    deliver(shadow)
                    pending.remove(shadow)
            elif learned_sleep:
                reward = total / (t-t0)
                update(18, reward)
                rec.update(reward=reward, delivered=t, values_delivered=list(values), sleep_share_delivered=slept/t)
            else:
                si += 1
            if trace:
                print('SLEEP_VISIT ' + json.dumps(rec), flush=True)
            continue
        plan = w4_world.shortest(world.room, target, 18) + ['stay'] * 8
        total = 0.
        for action in plan:
            if t >= L:
                break
            learnable = action == 'stay' and world.is_learnable(world.room)
            room, channel, symbols = world.step(action, rng)
            life.count_preq = learnable
            try:
                total += m8._progress_decision(life, symbols, channel, world.rooms[target])
            finally:
                life.count_preq = False
            for shadow in pending:
                if shadow.ready:
                    shadow.observe(symbols, channel)
            if channel != 'corridor':
                visit_pending = channel
            t += 1
            on_decision(True)
        visit_pending = w4._close_visit(life, visit_pending)
        raw = total / (t-t0)
        exam_symbols = sum(len(g.symbols) for g in life.exam.slots[world.rooms[target]])
        normalized = raw / exam_symbols if exam_symbols else 0.
        update(target, normalized if cf else raw)
        last_end[target] = t
        rec = dict(kind='room', t0=t0, end=t, target=target, mode=mode,
                   bits_per_decision=raw, exam_symbols=exam_symbols,
                   bps_per_decision=normalized, reward=normalized if cf else raw)
        visits.append(rec)
        if trace and sum(v['kind'] == 'room' for v in visits) <= 12:
            print('ROOM_SCALES ' + json.dumps(rec), flush=True)
    for shadow in list(pending):
        deliver(shadow)
        pending.remove(shadow)
    assert world.birth_log == expected_birth_log(seed, PERIOD, L)
    assert not life.exam.ids & life.replay.ids and life.exam_drawn == 0
    if arm == 'age_fixed20':
        assert slept == w4.sleep_decisions(L) and si == len(fixed_plan)
    scores = [cp['open'] for cp in checkpoints]
    row = dict(seed=seed, policy=arm, L=L, period=PERIOD, T_open=float(np.mean(scores)),
               final=scores[-1], checkpoints=checkpoints, sleep_share=slept/L,
               sleep_decisions=slept, visits=visits, max_pending=max_pending,
               wall_s=time.perf_counter()-start, eval_s=clock['eval'],
               learn_s=life.learn_s, score_s=life.score_s,
               finite=all(math.isfinite(s) for s in scores))
    assert row['finite']
    return row


def smoke(out):
    """Sequential foreground smoke; failure writes its evidence and stops."""
    import contextlib
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(ok=False, seed=12, period=128, workers=1, log=[])
    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            log_file.write(s)
        def flush(self):
            sys.__stdout__.flush()
            log_file.flush()
    start = time.perf_counter()
    try:
        with (out/'smoke.txt').open('w') as log_file, contextlib.redirect_stdout(Tee()):
            w6._threads()
            print('threads=1 process=1 H=64 wake decisions; later sleep pauses horizon', flush=True)
            print('CF comparison: room bits/decision divided by destination exam symbols at visit end; immediate arm retains M9 scale.', flush=True)
            baselines = {}
            for arm in ('age_fixed20', 'age_nosleep'):
                row = run_life(12, arm, 512)
                dose_fn = w4.sleep_decisions
                try:
                    if arm == 'age_nosleep':
                        w4.sleep_decisions = lambda L: 0
                    ref = w6.run_life(12, 'prog_age_s', 512, 128, score=True, check_sleep=True)
                finally:
                    w4.sleep_decisions = dose_fn
                gap = abs(row['T_open']-ref['T_open'])
                cp_gap = max(abs(a['open']-b['open']) for a,b in zip(row['checkpoints'],ref['checkpoints']))
                print(f'BASELINE {arm} T_open={row["T_open"]:.12f} W6={ref["T_open"]:.12f} abs={gap:.3e} checkpoint_max_abs={cp_gap:.3e} sleep_share={row["sleep_share"]:.9f}', flush=True)
                assert gap <= 1e-6 and cp_gap <= 1e-6
                baselines[arm] = dict(T_open=row['T_open'], reference=ref['T_open'], abs=gap, checkpoint_max_abs=cp_gap)
            payload['baselines'] = baselines
            exact = run_life(12, 'age_sleep_cf', 512, noop_sleep=True)
            sleeps = [v for v in exact['visits'] if v['kind']=='sleep']
            assert sleeps and any(v['wake_horizon'] == H for v in sleeps)
            err = max(abs(v['shadow_bits']-v['live_bits']) for v in sleeps)
            assert err <= 1e-9
            print(f'SHADOW_EXACT no-op sleep visits={len(sleeps)} max_abs_bits={err:.12g} full_horizons={sum(v["wake_horizon"]==H for v in sleeps)}', flush=True)
            payload['shadow_exact'] = dict(max_abs_bits=err, visits=sleeps)
            finite = {}
            for arm in ARMS:
                row = run_life(12, arm, 512)
                finite[arm] = dict(T_open=row['T_open'], final=row['final'], sleep_share=row['sleep_share'], wall_s=row['wall_s'])
                print('FINITE ' + json.dumps(dict(arm=arm, **finite[arm])), flush=True)
            payload['finite_L512'] = finite
            timed = run_life(12, 'age_sleep_cf', 1024, trace=True)
            payload['cf_L1024'] = timed
            # Full-store work grows with store occupancy. Conservative 8x life
            # scaling, then 4x contention for a four-process kernel (no speedup).
            estimate = timed['wall_s'] * (L_SCORE/1024) * 2 * 4
            payload['estimate'] = dict(cf_L1024_wall_s=timed['wall_s'], kernel_wall_s=estimate,
                                       split=estimate>80*60, factor=32,
                                       method='4x decisions * 2x store allowance * 4x pool contention; no parallel speedup assumed')
            print('WALL_ESTIMATE ' + json.dumps(payload['estimate']), flush=True)
            payload['ok'] = True
            print('SMOKE_OK', flush=True)
    except Exception as exc:
        payload['error'] = repr(exc)
        raise
    finally:
        payload['wall_s'] = time.perf_counter()-start
        payload['log'] = (out/'smoke.txt').read_text().splitlines()
        m7.dump(out/'smoke.json', payload)
    return payload


def _job(job):
    seed, arm = job
    row = run_life(seed, arm)
    print(f's{seed} {arm} T_open={row["T_open"]:.9f} sleep={row["sleep_share"]:.6f} wall={row["wall_s"]:.3f}', flush=True)
    return row


def kernel_main(seed, out, workers=4, kernel_id=None, arms=None, label=''):
    if seed not in SEEDS or workers != 4:
        raise ValueError('kernels require fresh seed 36..47 and pool of 4')
    arms = tuple(arms or ARMS)
    start = time.perf_counter()
    with get_context('spawn').Pool(4) as pool:
        runs = pool.map(_job, [(seed, arm) for arm in arms])
    payload = dict(seed=seed, workers=4, kernel_id=kernel_id or f'ezzzzzx/m11-w6-s{seed}',
                   runs=runs, wall_s=time.perf_counter()-start)
    m7.dump(Path(out)/f'm11_s{seed}{label}.json', payload)
    return payload


def stage(smoke_payload):
    assert smoke_payload['ok']
    base = ROOT/'kaggle/m11w6'
    code = base/'code'
    code.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in CODE_FILES:
        shutil.copy2(ROOT/name, code/name)
        hashes[name] = hashlib.sha256((code/name).read_bytes()).hexdigest()
    m7.dump(code/'dataset-metadata.json', dict(title='pdp code m11w6', id=DATASET_ID, licenses=[{'name':'other'}]))
    m7.dump(base/'code-sha256.json', hashes)
    split = smoke_payload['estimate']['split']
    groups = [('a', ARMS[:2]), ('b', ARMS[2:])] if split else [('', ARMS)]
    manifest = []
    for seed in SEEDS:
        for label, arms in groups:
            suffix = f'-{label}' if label else ''
            folder = base/f's{seed}{suffix}'
            folder.mkdir(parents=True, exist_ok=True)
            kid = f'ezzzzzx/m11-w6-s{seed}{suffix}'
            meta = dict(id=kid, title=kid.split('/')[1], code_file='kernel.py', language='python', kernel_type='script',
                        is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False,
                        dataset_sources=[DATASET_ID], competition_sources=[], kernel_sources=[], model_sources=[])
            m7.dump(folder/'kernel-metadata.json', meta)
            src = f'''import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CURIO_LR'] = '0.004'
os.environ['PYTHONHASHSEED'] = '0'
if os.environ.get('_M11_KERNEL') != '1':
    os.environ['_M11_KERNEL'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)

def main():
    hits = sorted(Path('/kaggle/input').rglob('m11_w6.py'))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one code dataset, found {{len(hits)}}')
    code = str(hits[0].parent)
    sys.path.insert(0, code)
    os.environ['PYTHONPATH'] = code
    import m11_w6
    m11_w6.kernel_main({seed}, '/kaggle/working', workers=4, kernel_id={kid!r}, arms={tuple(arms)!r}, label={suffix!r})

if __name__ == '__main__':
    main()
'''
            (folder/'kernel.py').write_text(src)
            manifest.append(dict(seed=seed, folder=folder.name, kernel_id=kid, arms=list(arms), output=f'm11_s{seed}{suffix}.json'))
    m7.dump(base/'manifest.json', manifest)
    write_pending(smoke_payload, manifest)
    print(f'STAGED {len(manifest)} private CPU kernels; split={split}', flush=True)


def write_pending(smoke_payload, manifest):
    out = ROOT/'results/m11w6'
    text = '# M11-W6\n\n48 scored lives pending. Seeds 36–47; L = 4096; birth_period = 128; H = 64 wake decisions.\n\n'
    text += 'CF room comparison = original bits saved per decision / destination exam symbols at visit end (0 for an empty exam slot). Immediate-credit control retains M9 units. Later sleep pauses a shadow’s wake horizon. Sleep EMA changes only at delivery; its age is measured from the latest sleep visit end. Pending sleep starts with value 0; no placeholder EMA update.\n\n'
    text += 'Wall estimate: ' + json.dumps(smoke_payload['estimate']) + '\n\n'
    text += 'Kernels: ' + ', '.join(m['kernel_id'] for m in manifest) + '\n\nSmoke output:\n\n```\n' + '\n'.join(smoke_payload['log']) + '\n```\n'
    (out/'RESULTS.md').write_text(text)


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), std_ddof0=float(a.std(ddof=0)), min=float(a.min()), max=float(a.max()), n=len(a)) if len(a) else dict(mean=None,std_ddof0=None,min=None,max=None,n=0)


def assemble(out):
    out = Path(out)
    smoke_payload = json.loads((out/'smoke.json').read_text())
    assert smoke_payload['ok']
    manifest = json.loads((ROOT/'kaggle/m11w6/manifest.json').read_text())
    runs, kernels = [], []
    for item in manifest:
        payload = json.loads((out/item['folder']/item['output']).read_text())
        assert payload['seed'] == item['seed'] and payload['kernel_id'] == item['kernel_id']
        assert sorted(r['policy'] for r in payload['runs']) == sorted(item['arms'])
        for row in payload['runs']:
            assert row['seed'] == item['seed'] and row['L'] == L_SCORE and row['period'] == PERIOD and row['finite']
            assert [cp['decision'] for cp in row['checkpoints']] == w6.checkpoints_of(L_SCORE)
            assert math.isclose(row['T_open'], np.mean([cp['open'] for cp in row['checkpoints']]), abs_tol=1e-12)
        runs += payload['runs']
        kernels.append({k: payload[k] for k in ('kernel_id','seed','workers','wall_s')})
    assert len(runs) == 48 and len({(r['seed'],r['policy']) for r in runs}) == 48
    by = {(r['seed'],r['policy']):r for r in runs}
    summary = {}
    for arm in ARMS:
        rows = [by[s,arm] for s in SEEDS]
        summary[arm] = {key:stats([r[key] for r in rows]) for key in ('T_open','final','sleep_share','wall_s')}
        summary[arm]['checkpoints'] = [dict(decision=d, **stats([r['checkpoints'][i]['open'] for r in rows])) for i,d in enumerate(w6.checkpoints_of(L_SCORE))]
    gaps = {}
    for left,right in GAPS:
        vals = [by[s,left]['T_open']-by[s,right]['T_open'] for s in SEEDS]
        gaps[left+'_minus_'+right] = dict(per_seed=dict(zip(SEEDS,vals)),left_lower=sum(v<0 for v in vals),**stats(vals))
    cf_rewards = [dict(seed=r['seed'],**v) for r in runs if r['policy']=='age_sleep_cf' for v in r['visits'] if v['kind']=='sleep']
    cf_rooms = [v for r in runs if r['policy']=='age_sleep_cf' for v in r['visits'] if v['kind']=='room']
    reward_summary = dict(counterfactual=stats([v['reward'] for v in cf_rewards]),
                          room_bps_per_decision=stats([v['bps_per_decision'] for v in cf_rooms]),
                          room_bits_per_decision=stats([v['bits_per_decision'] for v in cf_rooms]))
    result = dict(seeds=SEEDS, arms=ARMS, L=L_SCORE, period=PERIOD,H=H, runs=runs, kernels=kernels,
                  summary=summary,gaps=gaps,counterfactual_rewards=cf_rewards,reward_summary=reward_summary,smoke=smoke_payload)
    m7.dump(out/'results.json',result)
    lines = ['# M11-W6','', 'World6; seeds 36–47; L=4096; period=128; H=64 wake decisions; std ddof=0.','',
             'CF room comparison: bits saved/decision divided by destination exam symbols at visit end. Immediate control uses M9 bits/decision. Later sleep pauses wake horizons; life-end rewards use the available horizon and divisor 8.','']
    for field in ('T_open','final','sleep_share','wall_s'):
        lines += [field,'','| arm | '+' | '.join(map(str,SEEDS))+' | mean ± std |','|---|'+'---:|'*13]
        for arm in ARMS:
            st=summary[arm][field]
            lines.append('| '+arm+' | '+' | '.join(f'{by[s,arm][field]:.9f}' for s in SEEDS)+f' | {st["mean"]:.9f} ± {st["std_ddof0"]:.9f} |')
        lines.append('')
    lines += ['Checkpoints','','| arm | seed | decision | open |','|---|---:|---:|---:|']
    for r in runs:
        lines += [f'| {r["policy"]} | {r["seed"]} | {cp["decision"]} | {cp["open"]:.9f} |' for cp in r['checkpoints']]
    lines += ['', '| arm | decision | mean ± std |','|---|---:|---:|']
    for arm in ARMS:
        lines += [f'| {arm} | {cp["decision"]} | {cp["mean"]:.9f} ± {cp["std_ddof0"]:.9f} |' for cp in summary[arm]['checkpoints']]
    lines += ['', 'Gaps','','| gap | '+' | '.join(map(str,SEEDS))+' | mean ± std | left lower |','|---|'+'---:|'*14]
    for name,g in gaps.items():
        lines.append('| '+name+' | '+' | '.join(f'{g["per_seed"][s]:.9f}' for s in SEEDS)+f' | {g["mean"]:.9f} ± {g["std_ddof0"]:.9f} | {g["left_lower"]}/12 |')
    lines += ['', 'Reward scales','','```json',json.dumps(reward_summary,indent=2),'```','', 'Counterfactual rewards by decision','','| seed | t0 | delivered | wake | reward | bits/decision | sleep share |','|---:|---:|---:|---:|---:|---:|---:|']
    for v in cf_rewards:
        lines.append(f'| {v["seed"]} | {v["t0"]} | {v["delivered"]} | {v["wake_horizon"]} | {v["reward"]:.12g} | {v["bits_per_decision"]:.12g} | {v["sleep_share_delivered"]:.6f} |')
    lines += ['', 'Kernel walls','','```json',json.dumps(kernels,indent=2),'```','','Smoke output','','```','\n'.join(smoke_payload['log']),'```','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command',nargs='?',choices=('smoke','stage','assemble'))
    p.add_argument('--seed',type=int)
    p.add_argument('--out',default=str(ROOT/'results/m11w6'))
    p.add_argument('--workers',type=int,default=4)
    args = p.parse_args()
    if args.seed is not None:
        kernel_main(args.seed,args.out,args.workers)
    elif args.command == 'smoke':
        payload = smoke(args.out)
        stage(payload)
    elif args.command == 'stage':
        stage(json.loads((Path(args.out)/'smoke.json').read_text()))
    elif args.command == 'assemble':
        assemble(args.out)
    else:
        p.error('supply --seed or smoke/stage/assemble')

if __name__ == '__main__':
    main()
