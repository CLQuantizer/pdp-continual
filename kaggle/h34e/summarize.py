"""Offline H34e scoring. Missing pairs are NA and never count as passes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h34e_self_distill import ARMS, N_DATA, TAIL_START, H34B_REFERENCE, dump, source_hashes, layer_lambda


def fmt(x):
    return 'NA' if x is None else f'{x:.9f}'


def average(xs):
    return sum(xs)/len(xs) if all(x is not None for x in xs) else None


def reconstruct(r,start):
    boundary=((start+24999)//25000)*25000
    total=r['tail_edge_bits_per_byte']*(boundary-start) if boundary>start else 0.
    count=boundary-start
    for s in r['slices']:
        if s['start']>=boundary:
            total+=s['bits_per_byte']*(s['end']-s['start']); count+=s['end']-s['start']
    assert count==r['n_bytes']-start
    return total/count


def validate(r,arm,seed,smoke,manifest,fingerprint):
    assert smoke['ok'] and manifest['hashes']==smoke['source_hashes']==source_hashes()
    assert manifest['documents']==N_DATA==smoke['n_bytes']
    assert (r['arm'],r['seed'],r['n_bytes'],r['tail_start'])==(arm,seed,N_DATA,TAIL_START)
    assert r['kernel_id']==f"ezzzzzx/h34e-{arm.replace('_','-')}-s{seed}"
    assert r['sha256_e10m']==smoke['sha256_e10m'] and r['code_fingerprint']==fingerprint
    depth=1 if arm=='distill_1' else 4
    assert len(r['layers'])==depth and 0<r['wall_seconds']<=5100
    assert r['distill_lambda']==(0 if arm=='plain_4' else .5)
    assert r['layer_lambdas']==[layer_lambda(arm,k,depth) for k in range(depth)]
    assert [(s['start'],s['end']) for s in r['slices']]==[(i,min(i+25000,N_DATA)) for i in range(0,N_DATA,25000)]
    for key,start in (('bits_per_byte',0),('tail_bits_per_byte',TAIL_START)):
        assert math.isfinite(r[key]) and abs(reconstruct(r,start)-r[key])<=1e-9
    for d in r['layers']:
        assert all(math.isfinite(d[k]) for k in ('own_bits_per_byte','tail_own_bits_per_byte',
                   'mean_mixer_weight_tail','mean_mixer_weight_full','final_mixer_weight'))
        full=sum(s['own_bits_per_byte']*(s['end']-s['start']) for s in d['slices'])/N_DATA
        assert abs(full-d['own_bits_per_byte'])<=1e-9
    if arm=='plain_4':
        assert all(abs(r[k]-v)<=1e-6 for k,v in H34B_REFERENCE[seed].items()), 'H34b baseline mismatch'


def assemble(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    path=out/'smoke.json'; smoke=json.loads(path.read_text()) if path.exists() else {}
    path=ROOT/'kaggle/h34e/code-sha256.json'; manifest=json.loads(path.read_text()) if path.exists() else None
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest() if manifest else None
    runs={}; missing=[]
    for arm in ARMS:
        for seed in (0,1):
            path=out/'kernels'/f'{arm}-s{seed}'/'results.json'
            if not path.exists(): missing.append(f'{arm}-s{seed}'); continue
            r=json.loads(path.read_text()); validate(r,arm,seed,smoke,manifest,fingerprint)
            runs[arm,seed]=r
    lines=['| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    scoring=[]
    for key,label in (('tail_bits_per_byte','Tail'),('bits_per_byte','Full')):
        # distill_1 is report-only on its own; a four-layer difference is not its control.
        for arm in ('distill_4','distill_4_top'):
            gaps=[runs[arm,s][key]-runs['plain_4',s][key]
                  if (arm,s) in runs and ('plain_4',s) in runs else None for s in (0,1)]
            evaluated=sum(x is not None for x in gaps)
            bar=-.03 if (label,arm)==('Tail','distill_4') else None
            count=sum(x<=bar for x in gaps if x is not None) if bar is not None else None
            scoring.append(dict(range=label,quantity=arm+' - plain_4',per_seed=gaps,mean=average(gaps),
                                threshold=bar,seeds_meeting=count,evaluated=evaluated))
            lines.append(f'| {label} | {arm} − plain_4 | {fmt(gaps[0])} | {fmt(gaps[1])} | {fmt(average(gaps))} | '+
                         ('≤ -0.03' if bar is not None else 'report only')+f' | {str(count)+"/2" if count is not None else "—"} | {evaluated}/2 |')
    lines+=['','| Kernels scored | Kernels expected | Smoke | Tail start | Tail end |',
            '|---:|---:|---|---:|---:|',f"| {len(runs)} | 8 | {smoke.get('ok',False)} | {TAIL_START} | {N_DATA} |",
            '', '| Arm | Tail seed 0 | Tail seed 1 | Tail mean | Full seed 0 | Full seed 1 | Full mean |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        vals=[[runs.get((arm,s),{}).get(key) for s in (0,1)] for key in ('tail_bits_per_byte','bits_per_byte')]
        lines.append('| '+arm+' | '+' | '.join(fmt(x) for xs in vals for x in (*xs,average(xs)))+' |')
    lines+=['','| Arm | Seed | Layer | Own full | Own tail | Mean mixer full | Mean mixer tail | Final mixer |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for (arm,seed),r in runs.items():
        for k,d in enumerate(r['layers'],1):
            lines.append(f'| {arm} | {seed} | {k} | '+' | '.join(fmt(d[key]) for key in
                ('own_bits_per_byte','tail_own_bits_per_byte','mean_mixer_weight_full','mean_mixer_weight_tail','final_mixer_weight'))+' |')
    lines+=['','| Baseline − H34b | Seed | Tail | Full | Max abs limit |','|---|---:|---:|---:|---:|']
    for seed in (0,1):
        r=runs.get(('plain_4',seed),{})
        gaps=[r[k]-H34B_REFERENCE[seed][k] if k in r else None for k in ('tail_bits_per_byte','bits_per_byte')]
        lines.append(f'| plain_4 | {seed} | {fmt(gaps[0])} | {fmt(gaps[1])} | 0.000001 |')
    lines+=['','| Smoke check | Measurements |','|---|---|']
    for row in smoke.get('log',[]):
        lines.append('| '+row['check']+' | '+json.dumps({k:v for k,v in row.items() if k!='check'},sort_keys=True)+' |')
    lines+=['','| Kernel | Wall seconds |','|---|---:|']
    lines += [f"| {r['kernel_id']} | {r['wall_seconds']:.3f} |" for r in runs.values()]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'results.json',dict(n_bytes=N_DATA,tail_start=TAIL_START,scoring=scoring,
                               runs=list(runs.values()),missing=missing,smoke=smoke))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'results/h34e'))
    assemble(p.parse_args().out)
