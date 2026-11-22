"""Offline H34h numeric report. Missing seeds are NA and never count as passes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h34h_dense import ARMS, N_DATA, TAIL_START, initial_gate, dump, source_hashes, check_reference


def fmt(x):
    return 'NA' if x is None else f'{x:.9f}'


def average(xs):
    return sum(xs)/len(xs) if xs and all(x is not None for x in xs) else None


def validate(r,arm,seed,smoke,manifest,fingerprint):
    assert smoke['ok'] and manifest['hashes']==smoke['source_hashes']==source_hashes()
    assert manifest['documents']==N_DATA==smoke['n_bytes']
    assert (r['arm'],r['seed'],r['n_bytes'],r['tail_start'])==(arm,seed,N_DATA,TAIL_START)
    split=arm=='dense_8' and manifest['split_dense_8']
    assert r['kernel_id']==f"ezzzzzx/h34h-{arm.replace('_','-')}-s{seed}"+('-part2' if split else '')
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
    path=ROOT/'kaggle/h34h/code-sha256.json'; manifest=json.loads(path.read_text()) if path.exists() else None
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest() if manifest else None
    reference={}; baseline_hashes={}
    for name, arms in (('h34f',('plain_1','plain_4','plain_6','zgate_4','zgate_6')),('h34g',('zgate_8',))):
        path=ROOT/f'results/{name}/results.json'; baseline=json.loads(path.read_text())
        assert baseline['smoke']['ok'] and not baseline['missing']
        assert all(source_hashes()[k]==v for k,v in baseline['smoke']['source_hashes'].items())
        baseline_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for r in baseline['runs']:
            if r['arm'] in arms:
                assert r['n_bytes']==N_DATA and r['tail_start']==TAIL_START
                reference[r['arm'],r['seed']]=r
    prior_dense={r['seed']:r for r in json.loads((ROOT/'results/h34g/results.json').read_text())['runs'] if r['arm']=='dense_6'}
    runs={}; missing=[]; parts=[]
    for arm in ARMS:
        for seed in (0,1):
            tag=f'{arm}-s{seed}'; split=bool(manifest and manifest['split_dense_8'] and arm=='dense_8')
            path=out/'kernels'/(tag+'-part2' if split else tag)/'results.json'
            if not path.exists(): missing.append(tag); continue
            r=json.loads(path.read_text()); validate(r,arm,seed,smoke,manifest,fingerprint)
            if split:
                parent_path=out/'kernels'/tag/'results.json'; parent=json.loads(parent_path.read_text())
                assert parent['kernel_id']==f'ezzzzzx/h34h-dense-8-s{seed}'
                assert parent['code_fingerprint']==fingerprint and parent['sha256_e10m']==smoke['sha256_e10m']
                assert (parent['arm'],parent['seed'],parent['n_bytes'])==(arm,seed,N_DATA)
                assert (parent['bytes_seen'],parent['offset'],parent['pending_bytes'])==(90000,89984,16)
                assert 0<parent['segment_wall_seconds']<=5100
                assert r['resumed_checkpoint_sha256']==parent['checkpoint_sha256']
                assert hashlib.sha256((parent_path.parent/'checkpoint.pt').read_bytes()).hexdigest()==parent['checkpoint_sha256']
                assert abs(r['wall_seconds']-r['segment_wall_seconds']-parent['wall_seconds'])<=1e-6
                parts.append(parent)
            if arm=='dense_6':
                assert all(abs(r[k]-prior_dense[seed][k])<=1e-6 for k in ('bits_per_byte','tail_bits_per_byte','first_40kb_bits_per_byte'))
            runs[arm,seed]=r
    combined={**reference,**runs}
    lines=['| Depth | Arm | Tail s0 | Tail s1 | Tail mean | Full s0 | Full s1 | Full mean | Source |',
           '|---:|---|---:|---:|---:|---:|---:|---:|---|']
    ablation=[]
    for depth in (1,4,6,8):
        for kind in ('plain','zgate','dense'):
            arm=f'{kind}_{depth}'; source_arm='plain_1' if depth==1 else arm
            row=dict(arm=arm,depth=depth,source_arm=source_arm,
                     source='h34h' if source_arm in ARMS else ('h34g' if source_arm=='zgate_8' else 'h34f'))
            vals=[]
            for key,label in (('tail_bits_per_byte','tail'),('bits_per_byte','full')):
                v=[combined.get((source_arm,s),{}).get(key) for s in (0,1)]
                row[label]=dict(per_seed=v,mean=average(v)); vals.extend([*v,average(v)])
            if all(x is None for x in vals): row['source']='NA'
            ablation.append(row)
            lines.append(f'| {depth} | {arm} | '+' | '.join(fmt(x) for x in vals)+f" | {row['source']} |")
    arm='dense_6_ones'; vals=[]; row=dict(arm=arm,depth=6,source_arm=arm,source='h34h')
    for key,label in (('tail_bits_per_byte','tail'),('bits_per_byte','full')):
        v=[runs.get((arm,s),{}).get(key) for s in (0,1)]
        row[label]=dict(per_seed=v,mean=average(v)); vals.extend([*v,average(v)])
    ablation.append(row); lines.append('| 6 | dense_6_ones | '+' | '.join(fmt(x) for x in vals)+' | h34h |')
    lines+=['','| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    scoring=[]
    for left,right,bar in (('dense_8','dense_6',-.02),('dense_6','zgate_8',-.03),
                            ('dense_6_ones','dense_6',None),('dense_4','zgate_4',None)):
        for key,label in (('tail_bits_per_byte','Tail'),('bits_per_byte','Full')):
            gaps=[combined[left,s][key]-combined[right,s][key] if (left,s) in combined and (right,s) in combined else None for s in (0,1)]
            threshold=bar if label=='Tail' else None
            count=sum(x<=threshold for x in gaps if x is not None) if threshold is not None else None
            evaluated=sum(x is not None for x in gaps)
            scoring.append(dict(range=label,quantity=left+' - '+right,per_seed=gaps,mean=average(gaps),threshold=threshold,seeds_meeting=count,evaluated=evaluated))
            lines.append(f'| {label} | {left} − {right} | '+ ' | '.join(fmt(x) for x in (*gaps,average(gaps)))+' | '+
                         (f'≤ {threshold:.2f}' if threshold is not None else 'report only')+f' | {str(count)+"/2" if count is not None else "—"} | {evaluated}/2 |')
    lines+=['','| Runs scored | Runs expected | Smoke | Tail start | Tail end |',
            '|---:|---:|---|---:|---:|',f"| {len(runs)} | 8 | {smoke.get('ok',False)} | {TAIL_START} | {N_DATA} |"]
    lines+=['','| Arm | Seed | Layer | Own tail bits |','|---|---:|---:|---:|']
    for (arm,seed),r in combined.items():
        for k,d in enumerate(r['layers'],1):
            lines.append(f'| {arm} | {seed} | {k} | {fmt(d["tail_own_bits_per_byte"])} |')
    lines+=['','| Arm | Seed | Start | End | Mean mixer weights |','|---|---:|---:|---:|---|']
    for (arm,seed),r in combined.items():
        for sl in r['mixer_slices']:
            lines.append(f"| {arm} | {seed} | {sl['start']} | {sl['end']} | "+', '.join(fmt(x) for x in sl['mean_weights'])+' |')
    lines+=['','| dense_6 − H34g | Seed | Full | Tail | First 40 KB | Max abs limit |','|---|---:|---:|---:|---:|---:|']
    for seed in (0,1):
        r=runs.get(('dense_6',seed),{})
        v=[r[k]-prior_dense[seed][k] if k in r else None for k in ('bits_per_byte','tail_bits_per_byte','first_40kb_bits_per_byte')]
        lines.append(f'| dense_6 | {seed} | '+' | '.join(fmt(x) for x in v)+' | 0.000001 |')
    lines+=['','| Smoke check | Measurements |','|---|---|']
    for row in smoke.get('log',[]):
        lines.append('| '+row['check']+' | '+json.dumps({k:v for k,v in row.items() if k!='check'},sort_keys=True)+' |')
    lines+=['','| Kernel | Segment wall seconds | Run wall seconds |','|---|---:|---:|']
    for r in parts+list(combined.values()):
        lines.append(f"| {r['kernel_id']} | {r.get('segment_wall_seconds',r['wall_seconds']):.3f} | {r['wall_seconds']:.3f} |")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'results.json',dict(n_bytes=N_DATA,tail_start=TAIL_START,ablation=ablation,scoring=scoring,runs=list(runs.values()),
         reused_runs=list(reference.values()),parts=parts,missing=missing,smoke=smoke,baseline_sha256=baseline_hashes))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'results/h34h'))
    assemble(p.parse_args().out)
