"""W10 numeric summary; strict scored-output validation, optional pending tables."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
SEEDS=tuple(range(120,132))
ARMS=('nosleep','fixed_random','fixed_night','pressure_retina')

def validate_retina(row, asleep):
    """Independently reconstruct the controller from all glance codelengths."""
    from statistics import median
    observations = row['input_retina']
    assert [r['t'] for r in observations] == list(range(row['L']))
    history, ema, S, active, forced, left = [], None, 0, False, False, 0
    for t, r in enumerate(observations):
        if left == 0:
            if forced:
                forced = S > 128
                active = forced
            elif S >= 256:
                active = forced = True
            elif active:
                active = S > 0 and ema > 1.5 * median(history)
            else:
                active = S >= 32 and ema is not None and ema > 1.5 * median(history)
            if active:
                left = min(8, row['L'] - t)
        sleeping = left > 0
        assert sleeping == (t in asleep) == bool(r['asleep']), 'eyelid schedule'
        x = r['r']
        assert math.isfinite(x) and x >= 0
        ema = x if ema is None else .9 * ema + .1 * x
        if sleeping:
            S = max(0, S - 3)
            left -= 1
        else:
            S += 1
            history = (history + [ema])[-256:]
        assert S == r['S'] and forced == bool(r['forced']), 'eyelid pressure'
        assert math.isclose(ema, r['r_bar'], rel_tol=0, abs_tol=1e-12), 'eyelid EMA'
        assert math.isclose(median(history), r['m_t'], rel_tol=0, abs_tol=1e-12), 'wake median'
    for check in row['eyelid_checks']:
        assert check['t'] in asleep and check['intact'] and check['before'] == check['after']


def validate(job, dest):
    payload=json.loads((Path(dest)/job['output']).read_text())
    hashes=json.loads((ROOT/'kaggle/w10/code-sha256.json').read_text())
    assert payload['code_sha256']==hashes
    assert payload['seed']==job['seed'] and payload['kernel_id']==job['kernel_id']
    assert payload['workers']==4
    assert sorted(r['arm'] for r in payload['runs'])==sorted(ARMS)==sorted(job['arms'])
    for row in payload['runs']:
        assert row['seed']==job['seed'] and row['L']==4096 and row['period']==128
        assert row['night_enabled'] is True and row['finite'] is True
        cps=row['checkpoints']
        assert [r['decision'] for r in cps]==list(range(512,4097,512))
        assert all(math.isfinite(cp['open']) and all(math.isfinite(r['bits']) for r in cp['rooms']) for cp in cps)
        assert math.isclose(sum(cp['open'] for cp in cps)/8,row['T_open'],rel_tol=0,abs_tol=1e-12)
        assert row['final']==cps[-1]['open']
        visits=row['sleep_visits']
        assert all(0<=v['t']<4096 and 1<=v['length']<=8 and v['t']+v['length']<=4096
            and (v['length']==8 or v['t']+v['length']==4096) for v in visits)
        assert all(b['t']>=a['t']+a['length'] for a,b in zip(visits,visits[1:]))
        asleep={t for v in visits for t in range(v['t'],v['t']+v['length'])}
        night_n=sum(t%256>=192 for t in asleep)
        assert len(asleep)==row['sleep_decisions']
        assert row['sleep_share']==len(asleep)/4096
        assert row['night_sleep_decisions']==night_n
        assert row['night_sleep_share']==(night_n/len(asleep) if asleep else None)
        assert [r['t'] for r in row['input_rooms']]==list(range(4096))
        assert all(bool(r['asleep'])==(r['t'] in asleep) for r in row['input_rooms'])
        # Reconstruct each occupied slot's kind from the birth log, including births in sleep.
        import sys
        sys.path.insert(0,str(ROOT))
        import w6_world
        assert row['birth_log']==w6_world.expected_birth_log(job['seed'],128,4096)
        world=w6_world.World6(job['seed'],128)
        births={e['decision']:e for e in row['birth_log']}
        for r in row['input_rooms']:
            if r['t'] in births:
                e=births[r['t']]
                world.kind[e['slot']]=e['newborn_kind']
            assert r['kind']==world.kind[r['slot']]
        day=[r for r in row['input_rooms'] if r['asleep'] and r['t']%256<192]
        noise=sum(r['kind']=='N' for r in day)
        assert row['day_sleep_decisions']==len(day) and row['day_noise_sleep_decisions']==noise
        assert row['day_noise_sleep_share']==(noise/len(day) if day else None)
        assert all(c['intact'] for c in row['sleep_checks'])
        if row['arm']=='fixed_random':
            from w9_day_night import random_sleep_mask
            mask,starts=random_sleep_mask(job['seed'],4096)
            assert asleep==set(mask.nonzero()[0]) and row['random_starts']==starts and len(asleep)==1024
        elif row['arm']=='fixed_night':
            assert asleep=={t for t in range(4096) if t%256>=192}
        elif row['arm']=='nosleep':
            assert not asleep
        else:
            validate_retina(row,asleep)
    return payload


def cell(v):
    return 'NA' if v is None else f'{v:.9f}'

def table(headers, rows):
    return ['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+[
        '| '+' | '.join(str(v) for v in row)+' |' for row in rows]


def summarize(out, pending=False):
    out=Path(out)
    smoke=json.loads((out/'smoke.json').read_text())
    assert pending or smoke['ok']
    manifest_path=ROOT/'kaggle/w10/manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    by,kernels={},[]
    for job in manifest:
        path=out/job['folder']/job['output']
        if pending and not path.exists():
            continue
        payload=validate(job,path.parent)
        for row in payload['runs']:
            key=(row['seed'],row['arm'])
            assert key not in by
            by[key]=row
        kernels.append({k:payload[k] for k in ('seed','kernel_id','workers','wall_s')})
    if not pending:
        assert set(by)=={(s,a) for s in SEEDS for a in ARMS}
    gaps,lines={},[]
    rows=[]
    for right,threshold,required in [('nosleep',None,None),('fixed_random',-.05,8),('fixed_night',.05,None)]:
        vals=[by[s,'pressure_retina']['T_open']-by[s,right]['T_open']
            if (s,'pressure_retina') in by and (s,right) in by else None for s in SEEDS]
        have=[v for v in vals if v is not None]
        mean=math.fsum(have)/len(have) if have else None
        lower=sum(v<0 for v in have) if have else None
        count=sum(v<=threshold for v in have) if have and threshold is not None else None
        gaps[right]=dict(per_seed=dict(zip(SEEDS,vals)),mean=mean,lower_count=lower,
            threshold=threshold,count=count,required_count=required,n=len(have))
        rows.append(['pressure_retina - '+right,*map(cell,vals),cell(mean),lower if lower is not None else 'NA',
            cell(threshold),count if count is not None else 'NA',required if required is not None else 'NA',len(have)])
    lines+=table(['T_open gap',*map(str,SEEDS),'Mean','Count < 0','Threshold <=','Count <=','Required count','N'],rows)
    metrics={}
    for field in ('sleep_share','night_sleep_share','day_noise_sleep_share','T_open','final','wall_s'):
        metrics[field]={}
        rows=[]
        for arm in (('pressure_retina',) if field=='day_noise_sleep_share' else ARMS):
            vals=[by[s,arm][field] if (s,arm) in by else None for s in SEEDS]
            have=[v for v in vals if v is not None]
            mean=math.fsum(have)/len(have) if have else None
            metrics[field][arm]=dict(per_seed=dict(zip(SEEDS,vals)),mean=mean,n=len(have))
            rows.append([arm,*map(cell,vals),cell(mean),len(have)])
        lines+=['',*table([field,*map(str,SEEDS),'Mean','N'],rows)]
    night_mean=metrics['night_sleep_share']['pressure_retina']['mean']
    mean_gap=gaps['fixed_night']['mean']
    bars=[dict(quantity='pressure_retina night_sleep_share',mean=night_mean,comparison='>=',threshold=.55),
          dict(quantity='pressure_retina - fixed_night',mean=mean_gap,comparison='<=',threshold=.05)]
    lines+=['',*table(['Mean threshold','Mean','Comparison','Threshold'],[
        [b['quantity'],cell(b['mean']),b['comparison'],cell(b['threshold'])] for b in bars])]
    lines+=['',*table(['L','birth_period','Scored lives','Expected','Smoke workers','Smoke ok','Smoke wall s'],[
        [4096,128,len(by),48,smoke['workers'],int(smoke['ok']),cell(smoke['wall_s'])]])]
    if 'baseline' in smoke:
        b=smoke['baseline']
        lines+=['',*table(['Baseline seed','L','T_open','Reference T_open','Checkpoints exact','Prequential exact','Learner exact'],[
            [b['seed'],b['L'],cell(b['T_open']),cell(b['reference_T_open']),
             int(b['checkpoints_exact']),int(b['preq_bits_exact']),int(b['learner_exact'])]])]
    if 'per_day' in smoke:
        lines+=['',*table(['Day','Day sleep','Night sleep','Forced'],[
            [r[k] for k in ('day','day_sleep','night_sleep','forced')] for r in smoke['per_day']])]
    if 'pressure_retina' in smoke:
        r=smoke['pressure_retina']
        lines+=['',*table(['Smoke seed','Day sleep','Day noise sleep','Day noise sleep share','Sleep share','Night sleep share'],[
            [120,r['day_sleep_decisions'],r['day_noise_sleep_decisions'],cell(r['day_noise_sleep_share']),
             cell(r['sleep_share']),cell(r['night_sleep_share'])]])]
    if 'eyelid_checksum' in smoke:
        c=smoke['eyelid_checksum']
        lines+=['',*table(['Eyelid decisions','Matching checksums','Controller reconstruction'],[
            [c['decisions'],c['intact'],int(smoke.get('controller_reconstruction',False))]])]
    lines+=['',*table(['Seed','Kernel','Workers','Wall s'],[
        [r['seed'],r['kernel_id'],r['workers'],cell(r['wall_s'])] for r in kernels])]
    if 'estimate' in smoke:
        e=smoke['estimate']
        lines+=['',*table(['Estimate s','Limit s','Within limit'],[
            [cell(e['kernel_wall_s']),e['limit_s'],int(e['within_limit'])]])]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    missing=[dict(seed=s,arm=a) for s in SEEDS for a in ARMS if (s,a) not in by]
    (out/'results.json').write_text(json.dumps(dict(L=4096,period=128,seeds=SEEDS,arms=ARMS,
        gaps=gaps,mean_bars=bars,arm_metrics=metrics,runs=list(by.values()),kernels=kernels,
        smoke=smoke,missing=missing,pending=bool(missing)),indent=2,allow_nan=False)+'\n')
    print(f'lives={len(by)}/48 smoke={int(smoke["ok"])}')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'results/w10')
    p.add_argument('--pending',action='store_true')
    a=p.parse_args()
    summarize(a.out,a.pending)
