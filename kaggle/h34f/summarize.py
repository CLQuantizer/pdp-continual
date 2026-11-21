"""Offline H34f scoring; missing pairs remain NA and never count as passes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h34f_zero_gate import ARMS, N_DATA, TAIL_START, H34B_REFERENCE, initial_gate, dump, source_hashes


def fmt(x):
    return 'NA' if x is None else f'{x:.9f}'


def average(xs):
    return sum(xs)/len(xs) if all(x is not None for x in xs) else None


def validate(r,arm,seed,smoke,manifest,fingerprint):
    assert smoke['ok'] and manifest['hashes']==smoke['source_hashes']==source_hashes()
    assert manifest['documents']==N_DATA==smoke['n_bytes']
    assert (r['arm'],r['seed'],r['n_bytes'],r['tail_start'])==(arm,seed,N_DATA,TAIL_START)
    assert r['kernel_id']==f"ezzzzzx/h34f-{arm.replace('_','-')}-s{seed}"
    assert r['sha256_e10m']==smoke['sha256_e10m'] and r['code_fingerprint']==fingerprint
    depth=int(arm.split('_')[1])
    assert len(r['layers'])==depth and 0<r['wall_seconds']<=5100
    assert r['initial_mixer_weights']==initial_gate(arm)[0].tolist()
    expected=[(i,min(i+20000,N_DATA)) for i in range(0,N_DATA,20000)]
    assert [(s['start'],s['end']) for s in r['slices']]==expected
    assert [(s['start'],s['end']) for s in r['mixer_slices']]==expected
    for key,lo,hi in (('bits_per_byte',0,N_DATA),('tail_bits_per_byte',TAIL_START,N_DATA),
                      ('first_40kb_bits_per_byte',0,TAIL_START)):
        total=sum(s['bits_per_byte']*(s['end']-s['start']) for s in r['slices'] if lo<=s['start']<hi)
        assert math.isfinite(r[key]) and abs(total/(hi-lo)-r[key])<=1e-9
    for s in r['mixer_slices']:
        assert len(s['mean_weights'])==depth and all(math.isfinite(x) for x in s['mean_weights'])
    for d in r['layers']:
        assert all(math.isfinite(d[k]) for k in ('own_bits_per_byte','tail_own_bits_per_byte',
                   'mean_mixer_weight_tail','mean_mixer_weight_last_50kb','final_mixer_weight'))
        assert [(s['start'],s['end']) for s in d['slices']]==expected
        assert abs(sum(s['own_bits_per_byte']*(s['end']-s['start']) for s in d['slices'])/N_DATA-d['own_bits_per_byte'])<=1e-9
    if arm in H34B_REFERENCE:
        assert all(abs(r[k]-v)<=1e-6 for k,v in H34B_REFERENCE[arm][seed].items()), 'H34b baseline mismatch'


def assemble(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    path=out/'smoke.json'; smoke=json.loads(path.read_text()) if path.exists() else {}
    path=ROOT/'kaggle/h34f/code-sha256.json'; manifest=json.loads(path.read_text()) if path.exists() else None
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
    for arm,control in (('zgate_4','plain_4'),('zgate_6','plain_6'),('zgate_6','plain_1')):
        for key,label in (('bits_per_byte','Full'),('tail_bits_per_byte','Tail')):
            gaps=[runs[arm,s][key]-runs[control,s][key]
                  if (arm,s) in runs and (control,s) in runs else None for s in (0,1)]
            evaluated=sum(x is not None for x in gaps)
            bar={('zgate_4','plain_4','Full'):-.10,('zgate_4','plain_4','Tail'):.01,
                 ('zgate_6','plain_1','Full'):-.05}.get((arm,control,label))
            count=sum(x<=bar for x in gaps if x is not None) if bar is not None else None
            scoring.append(dict(range=label,quantity=arm+' - '+control,per_seed=gaps,mean=average(gaps),
                                threshold=bar,seeds_meeting=count,evaluated=evaluated))
            lines.append(f'| {label} | {arm} − {control} | {fmt(gaps[0])} | {fmt(gaps[1])} | {fmt(average(gaps))} | '+
                         (f'≤ {bar:.2f}' if bar is not None else 'report only')+f' | {str(count)+"/2" if count is not None else "—"} | {evaluated}/2 |')
    lines+=['','| Kernels scored | Kernels expected | Smoke | Tail start | Tail end |',
            '|---:|---:|---|---:|---:|',f"| {len(runs)} | 10 | {smoke.get('ok',False)} | {TAIL_START} | {N_DATA} |",
            '', '| Arm | Seed | First 40 KB | Tail | Full |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for seed in (0,1):
            r=runs.get((arm,seed),{})
            lines.append(f'| {arm} | {seed} | '+' | '.join(fmt(r.get(k)) for k in
                ('first_40kb_bits_per_byte','tail_bits_per_byte','bits_per_byte'))+' |')
        means=[average([runs.get((arm,s),{}).get(k) for s in (0,1)]) for k in
               ('first_40kb_bits_per_byte','tail_bits_per_byte','bits_per_byte')]
        lines.append('| '+arm+' | Mean | '+' | '.join(fmt(x) for x in means)+' |')
    lines+=['','| Arm | Seed | Start | End | Mean mixer weights |','|---|---:|---:|---:|---|']
    for arm in ARMS:
        for seed in (0,1):
            for sl in runs.get((arm,seed),{}).get('mixer_slices',[]):
                lines.append(f"| {arm} | {seed} | {sl['start']} | {sl['end']} | "+', '.join(fmt(x) for x in sl['mean_weights'])+' |')
        if all((arm,s) in runs for s in (0,1)):
            for a,b in zip(runs[arm,0]['mixer_slices'],runs[arm,1]['mixer_slices']):
                lines.append(f"| {arm} | Mean | {a['start']} | {a['end']} | "+', '.join(fmt((x+y)/2) for x,y in zip(a['mean_weights'],b['mean_weights']))+' |')
    lines+=['','| Baseline − H34b | Seed | First 40 KB | Tail | Full | Max abs limit |','|---|---:|---:|---:|---:|---:|']
    for arm in H34B_REFERENCE:
        for seed in (0,1):
            r=runs.get((arm,seed),{})
            gaps=[r[k]-H34B_REFERENCE[arm][seed][k] if k in r else None for k in
                  ('first_40kb_bits_per_byte','tail_bits_per_byte','bits_per_byte')]
            lines.append(f'| {arm} | {seed} | '+' | '.join(fmt(x) for x in gaps)+' | 0.000001 |')
    lines+=['','| Smoke check | Measurements |','|---|---|']
    for row in smoke.get('log',[]):
        lines.append('| '+row['check']+' | '+json.dumps({k:v for k,v in row.items() if k!='check'},sort_keys=True)+' |')
    lines+=['','| Kernel | Wall seconds |','|---|---:|']
    lines += [f"| {r['kernel_id']} | {r['wall_seconds']:.3f} |" for r in runs.values()]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'results.json',dict(n_bytes=N_DATA,tail_start=TAIL_START,scoring=scoring,
                               runs=list(runs.values()),missing=missing,smoke=smoke))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'results/h34f'))
    assemble(p.parse_args().out)
