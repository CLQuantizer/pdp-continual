"""Offline H34g numeric report. Missing seeds are NA and never count as passes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h34g_depth import ARMS, N_DATA, TAIL_START, initial_gate, dump, source_hashes, check_reference


def fmt(x):
    return 'NA' if x is None else f'{x:.9f}'


def average(xs):
    return sum(xs)/len(xs) if xs and all(x is not None for x in xs) else None


def validate(r,arm,seed,smoke,manifest,fingerprint):
    assert smoke['ok'] and manifest['hashes']==smoke['source_hashes']==source_hashes()
    assert manifest['documents']==N_DATA==smoke['n_bytes']
    assert (r['arm'],r['seed'],r['n_bytes'],r['tail_start'])==(arm,seed,N_DATA,TAIL_START)
    split=arm=='zgate_8' and manifest['split_zgate_8']
    assert r['kernel_id']==f"ezzzzzx/h34g-{arm.replace('_','-')}-s{seed}"+('-part2' if split else '')
    assert r['sha256_e10m']==smoke['sha256_e10m'] and r['code_fingerprint']==fingerprint
    depth=int(arm.split('_')[1])
    assert len(r['layers'])==depth and 0<r['segment_wall_seconds']<=5100
    assert r['segment_wall_seconds']<=r['wall_seconds']<=5100*(2 if split else 1)
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
        for key,lo,n in (('own_bits_per_byte',0,N_DATA),('tail_own_bits_per_byte',TAIL_START,N_DATA-TAIL_START)):
            assert abs(sum(s['own_bits_per_byte']*(s['end']-s['start']) for s in d['slices'] if s['start']>=lo)/n-d[key])<=1e-9
    check_reference(r)


def assemble(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    path=out/'smoke.json'; smoke=json.loads(path.read_text()) if path.exists() else {}
    path=ROOT/'kaggle/h34g/code-sha256.json'; manifest=json.loads(path.read_text()) if path.exists() else None
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest() if manifest else None
    baseline_path=ROOT/'results/h34f/results.json'
    baseline=json.loads(baseline_path.read_text())
    assert baseline['smoke']['ok'] and not baseline['missing']
    assert baseline['smoke']['source_hashes']=={k:v for k,v in source_hashes().items() if k!='h34g_depth.py'}
    reference={(r['arm'],r['seed']):r for r in baseline['runs']}
    runs={}; missing=[]; parts=[]
    for arm in ARMS:
        for seed in (0,1):
            tag=f'{arm}-s{seed}'
            split=bool(manifest and manifest['split_zgate_8'] and arm=='zgate_8')
            path=out/'kernels'/(tag+'-part2' if split else tag)/'results.json'
            if not path.exists(): missing.append(tag); continue
            r=json.loads(path.read_text()); validate(r,arm,seed,smoke,manifest,fingerprint)
            if split:
                parent_path=out/'kernels'/tag/'results.json'
                parent=json.loads(parent_path.read_text())
                assert parent['kernel_id']==f"ezzzzzx/h34g-zgate-8-s{seed}"
                assert parent['code_fingerprint']==fingerprint and parent['sha256_e10m']==smoke['sha256_e10m']
                assert (parent['arm'],parent['seed'],parent['n_bytes'])==(arm,seed,N_DATA)
                assert parent['bytes_seen']==90000 and parent['offset']==89984 and parent['pending_bytes']==16
                assert 0<parent['segment_wall_seconds']<=5100
                assert r['resumed_checkpoint_sha256']==parent['checkpoint_sha256']
                assert hashlib.sha256((parent_path.parent/'checkpoint.pt').read_bytes()).hexdigest()==parent['checkpoint_sha256']
                assert abs(r['wall_seconds']-r['segment_wall_seconds']-parent['wall_seconds'])<=1e-6
                parts.append(parent)
            if arm=='zgate_6':
                assert all(abs(r[k]-reference[arm,seed][k])<=1e-6 for k in
                    ('bits_per_byte','tail_bits_per_byte','first_40kb_bits_per_byte'))
            runs[arm,seed]=r
    lines=['| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    scoring=[]
    for arm,bar in (('zgate_8',-.02),('dense_6',-.03)):
        for key,label in (('tail_bits_per_byte','Tail'),('bits_per_byte','Full')):
            gaps=[runs[arm,s][key]-runs['zgate_6',s][key] if (arm,s) in runs and ('zgate_6',s) in runs else None for s in (0,1)]
            threshold=bar if label=='Tail' else None
            count=sum(x<=threshold for x in gaps if x is not None) if threshold is not None else None
            evaluated=sum(x is not None for x in gaps)
            scoring.append(dict(range=label,quantity=arm+' - zgate_6',per_seed=gaps,mean=average(gaps),
                                threshold=threshold,seeds_meeting=count,evaluated=evaluated))
            lines.append(f'| {label} | {arm} − zgate_6 | {fmt(gaps[0])} | {fmt(gaps[1])} | {fmt(average(gaps))} | '+
                         (f'≤ {threshold:.2f}' if threshold is not None else 'report only')+f' | {str(count)+"/2" if count is not None else "—"} | {evaluated}/2 |')
    curve=[]
    lines+=['','| Depth | Arm | Seed 0 tail | Seed 1 tail | Mean tail |','|---:|---|---:|---:|---:|']
    for arm in ('plain_1','zgate_2','zgate_4','zgate_6','zgate_8'):
        source=reference if arm in ('plain_1','zgate_4') else runs
        values=[source.get((arm,s),{}).get('tail_bits_per_byte') for s in (0,1)]
        curve.append(dict(arm=arm,per_seed=values,mean=average(values),source='h34f' if source is reference else 'h34g'))
        lines.append(f'| {arm.split("_")[1]} | {arm} | '+ ' | '.join(fmt(x) for x in (*values,average(values)))+' |')
    lines+=['','| Runs scored | Runs expected | Smoke | Tail start | Tail end |',
            '|---:|---:|---|---:|---:|',f"| {len(runs)} | 8 | {smoke.get('ok',False)} | {TAIL_START} | {N_DATA} |",
            '', '| Arm | Seed | First 40 KB | Tail | Full |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for seed in (0,1):
            r=runs.get((arm,seed),{})
            lines.append(f'| {arm} | {seed} | '+' | '.join(fmt(r.get(k)) for k in
                ('first_40kb_bits_per_byte','tail_bits_per_byte','bits_per_byte'))+' |')
        means=[average([runs.get((arm,s),{}).get(k) for s in (0,1)]) for k in
               ('first_40kb_bits_per_byte','tail_bits_per_byte','bits_per_byte')]
        lines.append('| '+arm+' | Mean | '+' | '.join(fmt(x) for x in means)+' |')
    lines+=['','| Arm | Seed | Layer | Own tail bits |','|---|---:|---:|---:|']
    for arm in ARMS:
        for seed in (0,1):
            for k,d in enumerate(runs.get((arm,seed),{}).get('layers',[]),1):
                lines.append(f'| {arm} | {seed} | {k} | {fmt(d["tail_own_bits_per_byte"])} |')
        if all((arm,s) in runs for s in (0,1)):
            for k,(a,b) in enumerate(zip(runs[arm,0]['layers'],runs[arm,1]['layers']),1):
                lines.append(f'| {arm} | Mean | {k} | {fmt((a["tail_own_bits_per_byte"]+b["tail_own_bits_per_byte"])/2)} |')
    lines+=['','| Arm | Seed | Start | End | Mean mixer weights |','|---|---:|---:|---:|---|']
    for arm in ARMS:
        for seed in (0,1):
            for sl in runs.get((arm,seed),{}).get('mixer_slices',[]):
                lines.append(f"| {arm} | {seed} | {sl['start']} | {sl['end']} | "+', '.join(fmt(x) for x in sl['mean_weights'])+' |')
        if all((arm,s) in runs for s in (0,1)):
            for a,b in zip(runs[arm,0]['mixer_slices'],runs[arm,1]['mixer_slices']):
                lines.append(f"| {arm} | Mean | {a['start']} | {a['end']} | "+', '.join(fmt((x+y)/2) for x,y in zip(a['mean_weights'],b['mean_weights']))+' |')
    lines+=['','| zgate_6 − H34f | Seed | First 40 KB | Tail | Full | Max abs limit |','|---|---:|---:|---:|---:|---:|']
    for seed in (0,1):
        r=runs.get(('zgate_6',seed),{})
        gaps=[r[k]-reference['zgate_6',seed][k] if k in r else None for k in
              ('first_40kb_bits_per_byte','tail_bits_per_byte','bits_per_byte')]
        lines.append(f'| zgate_6 | {seed} | '+' | '.join(fmt(x) for x in gaps)+' | 0.000001 |')
    lines+=['','| Smoke check | Measurements |','|---|---|']
    for row in smoke.get('log',[]):
        lines.append('| '+row['check']+' | '+json.dumps({k:v for k,v in row.items() if k!='check'},sort_keys=True)+' |')
    lines+=['','| Kernel | Segment wall seconds | Run wall seconds |','|---|---:|---:|']
    lines += [f"| {r['kernel_id']} | {r['segment_wall_seconds']:.3f} | {r['wall_seconds']:.3f} |" for r in parts+list(runs.values())]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'results.json',dict(n_bytes=N_DATA,tail_start=TAIL_START,scoring=scoring,depth_curve=curve,
        runs=list(runs.values()),parts=parts,missing=missing,smoke=smoke,
        baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest()))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'results/h34g'))
    assemble(p.parse_args().out)
