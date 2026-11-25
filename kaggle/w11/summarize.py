"""W11 numeric tables. Missing scored lives stay null; bars are never retuned."""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'kaggle/w11'
SEEDS = tuple(range(132,144))
CORTICES = ('bank_1','deep_4','deep_4_open')
REDUCED_SEEDS = tuple(range(132,138))
REDUCED_CORTICES = ('bank_1','deep_4')


def add_filters(parser):
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--cortices', nargs='+', choices=CORTICES, default=CORTICES)


def selection(seeds, cortices):
    return (tuple(s for s in SEEDS if s in seeds),
            tuple(c for c in CORTICES if c in cortices))


def filter_manifest(manifest, seeds, cortices):
    return [j for j in manifest if j['seed'] in seeds and j['cortex'] in cortices]


def w10_validator():
    spec = importlib.util.spec_from_file_location('w10_summary', ROOT / 'kaggle/w10/summarize.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate(job, dest):
    dest = Path(dest)
    payload = json.loads((dest / job['output']).read_text())
    hashes = json.loads((BASE / 'code-sha256.json').read_text())
    assert payload['code_sha256'] == hashes
    for k in ('seed','cortex','part','parts','kernel_id'):
        assert payload[k] == job[k]
    assert payload['workers'] == 1 and len(payload['runs']) == 1
    row = payload['runs'][0]
    assert row['seed'] == job['seed'] and row['cortex'] == job['cortex']
    final = job['part'] == job['parts']
    assert row['complete'] == final
    assert math.isfinite(payload['wall_s']) and payload['wall_s'] > 0
    if not final:
        assert row['decision'] == 4096 * job['part'] // job['parts']
        cp = dest / payload['checkpoint_file']
        assert cp.name == f"{job['cortex']}-s{job['seed']}-p{job['part']}.pkl"
        assert hashlib.sha256(cp.read_bytes()).hexdigest() == payload['checkpoint_sha256']
        return payload
    assert row['L'] == 4096 and row['period'] == 128 and row['arm'] == 'pressure_retina'
    assert row['night_enabled'] is True and row['finite'] is True
    cps = row['checkpoints']
    assert [cp['decision'] for cp in cps] == list(range(512,4097,512))
    assert all(math.isfinite(cp['open']) and cp['n'] == len(cp['rooms']) and
               all(math.isfinite(r['bits']) for r in cp['rooms']) for cp in cps)
    assert abs(sum(cp['open'] for cp in cps)/8 - row['T_open']) <= 1e-12
    assert row['final'] == cps[-1]['open']
    visits = row['sleep_visits']
    assert all(0 <= v['t'] < 4096 and 1 <= v['length'] <= 8 and v['t']+v['length'] <= 4096
               and (v['length'] == 8 or v['t']+v['length'] == 4096) for v in visits)
    assert all(b['t'] >= a['t']+a['length'] for a,b in zip(visits,visits[1:]))
    asleep = {t for v in visits for t in range(v['t'],v['t']+v['length'])}
    nights = sum(t%256 >= 192 for t in asleep)
    assert row['sleep_decisions'] == len(asleep) and row['sleep_share'] == len(asleep)/4096
    assert row['night_sleep_decisions'] == nights
    assert row['night_sleep_share'] == (nights / len(asleep) if asleep else None)
    assert [r['t'] for r in row['input_rooms']] == list(range(4096))
    assert all(bool(r['asleep']) == (r['t'] in asleep) for r in row['input_rooms'])
    assert all(s['intact'] and s['param_delta'] > 0 for s in row['sleep_checks'])
    w10_validator().validate_retina(row, asleep)
    sys.path.insert(0,str(ROOT))
    import w6_world
    assert row['birth_log'] == w6_world.expected_birth_log(job['seed'],128,4096)
    world = w6_world.World6(job['seed'],128)
    births = {e['decision']:e for e in row['birth_log']}
    for r in row['input_rooms']:
        if r['t'] in births:
            e = births[r['t']]
            world.kind[e['slot']] = e['newborn_kind']
        assert r['kind'] == world.kind[r['slot']]
    day = [r for r in row['input_rooms'] if r['asleep'] and r['t']%256 < 192]
    noise = sum(r['kind']=='N' for r in day)
    assert row['day_sleep_decisions'] == len(day) and row['day_noise_sleep_decisions'] == noise
    assert row['day_noise_sleep_share'] == (noise/len(day) if day else None)
    logs = row['glance_log']
    assert sum(g['wake'] for g in logs) == 4096-len(asleep)
    assert all(g['n']==64 and math.isfinite(g['bits']) and len(g['sha256'])==64 for g in logs)
    assert row['preq_symbols'] == sum(g['n'] for g in logs if g['preq'])
    assert row['preq_bits'] == sum(g['bits'] for g in logs if g['preq'])
    assert row['preq_bps'] == row['preq_bits'] / row['preq_symbols']
    depth = 1 if job['cortex'] == 'bank_1' else 4
    initial = [1.] if depth==1 else [1.]*4 if job['cortex']=='deep_4_open' else [1.,0.,0.,0.]
    assert row['initial_mixer_weights'] == initial
    assert [r['decision'] for r in row['mixer_history']] == list(range(512,4097,512))
    assert all(len(r[k])==depth and all(math.isfinite(v) for v in r[k])
               for r in row['mixer_history'] for k in ('weights','mean_weights'))
    return payload


def cell(v):
    return 'NA' if v is None else f'{v:.9f}'


def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)] + [
        '| '+' | '.join(str(v) for v in row)+' |' for row in rows]


def summarize(out, pending=False, seeds=SEEDS, cortices=CORTICES):
    seeds, cortices = selection(seeds, cortices)
    reduced = seeds == REDUCED_SEEDS and cortices == REDUCED_CORTICES
    registered = reduced or seeds == SEEDS
    required_count = (5 if reduced else 8) if registered else None
    expected_lives = len(seeds) * len(cortices)
    out = Path(out)
    smoke = json.loads((out / 'smoke.json').read_text())
    assert smoke['ok']
    manifest = filter_manifest(json.loads((BASE / 'manifest.json').read_text()), seeds, cortices)
    by, kernels = {}, []
    for job in manifest:
        path = out / job['folder'] / job['output']
        if pending and not path.exists():
            continue
        payload = validate(job, path.parent)
        kernels.append({k:payload[k] for k in ('seed','cortex','part','parts','kernel_id','wall_s')})
        if job['part'] == job['parts']:
            key = (job['seed'],job['cortex'])
            assert key not in by
            by[key] = payload['runs'][0]
    if not pending:
        assert set(by) == {(s,c) for s in seeds for c in cortices}
    lines, gaps = [], {}
    comparisons = [(lhs,rhs) for lhs,rhs in (('deep_4','bank_1'),('deep_4_open','deep_4'))
                   if lhs in cortices and rhs in cortices]
    for lhs,rhs in comparisons:
        for field in ('preq_bps','T_open'):
            vals = [by[s,lhs][field]-by[s,rhs][field] if (s,lhs) in by and (s,rhs) in by else None for s in seeds]
            have = [v for v in vals if v is not None]
            threshold = -.05 if (lhs,rhs,field)==('deep_4','bank_1','T_open') else None
            data = dict(per_seed=dict(zip(seeds,vals)), n=len(have),
                mean=sum(have)/len(have) if have else None, lower_count=sum(v<0 for v in have),
                threshold=threshold, count=sum(v<=threshold for v in have) if threshold is not None else None,
                required_count=required_count if threshold is not None else None)
            gaps[f'{lhs}-{rhs}/{field}'] = data
    # Compact primary and lifetime prequential gaps first, then sleep, then open-gate ablation.
    def gap_tables(lhs,rhs):
        for field in ('preq_bps','T_open'):
            d = gaps[f'{lhs}-{rhs}/{field}']
            lines.extend(table([f'{lhs} - {rhs}: {field}', *map(str,seeds),'Mean','Count < 0','Threshold <=','Count <=','Required','N'],[
                [field,*map(cell,d['per_seed'].values()),cell(d['mean']),d['lower_count'],cell(d['threshold']),
                 d['count'] if d['count'] is not None else 'NA',d['required_count'] or 'NA',d['n']]]))
            lines.append('')
    if ('deep_4','bank_1') in comparisons:
        gap_tables('deep_4','bank_1')
    metrics = {}
    def metric_table(field):
        rows=[]
        metrics[field]={}
        for c in cortices:
            vals=[by[s,c][field] if (s,c) in by else None for s in seeds]
            have=[v for v in vals if v is not None]
            d=dict(per_seed=dict(zip(seeds,vals)),mean=sum(have)/len(have) if have else None,n=len(have))
            metrics[field][c]=d
            rows.append([c,*map(cell,vals),cell(d['mean']),d['n']])
        lines.extend(table([field,*map(str,seeds),'Mean','N'],rows)); lines.append('')
    for field in ('sleep_share','night_sleep_share'):
        metric_table(field)
    if 'deep_4' in cortices:
        night = metrics['night_sleep_share']['deep_4']
        lines.extend(table(['cortex','night_sleep_share mean','Threshold >=','N'],[['deep_4',cell(night['mean']),'.55',night['n']]])); lines.append('')
    if ('deep_4_open','deep_4') in comparisons:
        gap_tables('deep_4_open','deep_4')
    for field in ('preq_bps','T_open','final','preq_bits','preq_symbols','sleep_decisions','night_sleep_decisions','wall_s'):
        metric_table(field)
    lines.extend(table(['seed','cortex',*map(str,range(512,4097,512)),'T_open'],[
        [seed,cortex,*[cell(cp['open']) for cp in by[seed,cortex]['checkpoints']],cell(by[seed,cortex]['T_open'])]
        for seed in seeds for cortex in cortices if (seed,cortex) in by])); lines.append('')
    mixer=[]
    for s in seeds:
        for c in cortices:
            if (s,c) in by:
                for r in by[s,c]['mixer_history']:
                    mixer.append(dict(seed=s,cortex=c,**r))
    lines.extend(table(['seed','cortex','decision','mean weights','weights'],[
        [r['seed'],r['cortex'],r['decision'],', '.join(map(cell,r['mean_weights'])),', '.join(map(cell,r['weights']))] for r in mixer])); lines.append('')
    lines.extend(table(['seed','cortex','part','parts','wall s'],[[r['seed'],r['cortex'],r['part'],r['parts'],cell(r['wall_s'])] for r in kernels])); lines.append('')
    estimate = smoke['estimate']
    lines.extend(table(['cortex','Estimated life wall s','Parts','Estimated part wall s','Limit s'],[
        [c,cell(estimate['kernel_wall_s'][c]),estimate['parts'][c],
         cell(estimate['segment_wall_s'][c]),estimate['limit_s']] for c in cortices])); lines.append('')
    wall_label = 'Smoke measured segments s' if 'continuation_wall_s' in smoke else 'Smoke wall s'
    lines.extend(table(['Scored lives','Expected','Smoke workers','Smoke ok',wall_label],[[len(by),expected_lives,smoke['workers'],int(smoke['ok']),cell(smoke['wall_s'])]]))
    result = dict(seeds=seeds,cortices=cortices,L=4096,period=128,scored_lives=len(by),expected_lives=expected_lives,
        gaps=gaps,metrics=metrics,bars=([dict(quantity='deep_4-bank_1/T_open',threshold=-.05,required_count=required_count,n=len(seeds))]
              if registered and ('deep_4','bank_1') in comparisons else []) +
             ([dict(quantity='deep_4/night_sleep_share',threshold=.55,comparison='>=',n=len(seeds))]
              if registered and 'deep_4' in cortices else []),
        mixer_history=mixer,kernels=kernels,runs=list(by.values()),smoke=smoke)
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (out / 'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'{len(by)}/{expected_lives} scored lives',flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'results/w11')
    p.add_argument('--pending',action='store_true')
    add_filters(p)
    args=p.parse_args()
    summarize(args.out,args.pending,args.seeds,args.cortices)
