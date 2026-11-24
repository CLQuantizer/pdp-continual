"""H34i numeric tables. Missing runs are NA; never passes or scored fixtures."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h34i_generality import ARMS, STREAMS, N_DATA, TAIL_START, initial_gate, dump, source_hashes
BASE=ROOT/'kaggle/h34i'
BARS=(('Own','plain_6','plain_1','tail_bits_per_byte',-.05),
      ('Wait','zgate_6','plain_6','bits_per_byte',-.10),
      ('See','dense_6','zgate_6','tail_bits_per_byte',-.02))


def fmt(x):
    return 'NA' if x is None else f'{x:.9f}'


def difference(runs,stream,left,right,seed,key):
    a,b=runs.get((stream,left,seed)),runs.get((stream,right,seed))
    return a[key]-b[key] if a is not None and b is not None else None


def validate(r,stream,arm,seed,smoke,manifest,fingerprint):
    assert smoke['ok'] and manifest['hashes']==smoke['source_hashes']==source_hashes()
    assert manifest['documents']==N_DATA==smoke['n_bytes']
    assert (r['arm'],r['seed'],r['n_bytes'],r['tail_start'])==(arm,seed,N_DATA,TAIL_START)
    split=arm in manifest['split_arms']
    assert r['kernel_id']==f"ezzzzzx/h34i-{stream}-{arm.replace('_','-')}-s{seed}"+('-part2' if split else '')
    assert r['stream']==stream and r['raw_sha256']==smoke['streams'][stream]['sha256']
    assert manifest['streams']==smoke['streams'] and r['code_fingerprint']==fingerprint
    assert r['part']==(2 if split else 0)
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


def references():
    sources={}; hashes={}; refs={}
    for name in ('h34b','h34f','h34g','h34h'):
        path=ROOT/f'results/{name}/results.json'
        baseline=json.loads(path.read_text())
        assert baseline['smoke']['ok'] and not baseline['missing']
        assert all(source_hashes()[k]==v for k,v in baseline['smoke']['source_hashes'].items())
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        sources[name]={(r['arm'],r['seed']):r for r in baseline['runs']}
    for arm in ARMS:
        name='h34h' if arm.startswith('dense') else 'h34f'
        for seed in (0,1):
            r=copy.deepcopy(sources[name][arm,seed])
            assert r['n_bytes']==N_DATA and r['tail_start']==TAIL_START
            r.update(stream='enwik8',source=name)
            refs['enwik8',arm,seed]=r
            prior='h34b' if arm.startswith('plain') else ('h34g' if arm=='dense_6' else None)
            if prior:
                old=sources[prior][arm,seed]
                for key in ('bits_per_byte','tail_bits_per_byte'):
                    assert abs(r[key]-old[key])<=1e-6
    return refs,hashes


def assemble(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    smoke_path=out/'smoke.json'
    smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    manifest_path=BASE/'code-sha256.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest() if manifest else None
    reference,baseline_hashes=references()
    runs={}; missing=[]; parts=[]
    for stream in STREAMS:
        for arm in ARMS:
            for seed in (0,1):
                tag=f'{stream}-{arm}-s{seed}'
                split=bool(manifest and arm in manifest['split_arms'])
                path=out/'kernels'/(tag+'-part2' if split else tag)/'results.json'
                if not path.exists():
                    missing.append(tag); continue
                r=json.loads(path.read_text())
                validate(r,stream,arm,seed,smoke,manifest,fingerprint)
                if split:
                    parent_path=out/'kernels'/tag/'results.json'
                    parent=json.loads(parent_path.read_text())
                    assert parent['kernel_id']==r['kernel_id'].removesuffix('-part2')
                    assert parent['code_fingerprint']==fingerprint and parent['raw_sha256']==r['raw_sha256']
                    assert (parent['stream'],parent['arm'],parent['seed'],parent['n_bytes'],parent['part'])==(stream,arm,seed,N_DATA,1)
                    assert (parent['bytes_seen'],parent['offset'],parent['pending_bytes'])==(90000,89984,16)
                    assert 0<parent['segment_wall_seconds']<=5100 and parent['wall_seconds']==parent['segment_wall_seconds']
                    assert r['resumed_checkpoint_sha256']==parent['checkpoint_sha256']
                    assert hashlib.sha256((parent_path.parent/'checkpoint.pt').read_bytes()).hexdigest()==parent['checkpoint_sha256']
                    assert abs(r['wall_seconds']-r['segment_wall_seconds']-parent['wall_seconds'])<=1e-6
                    parts.append(parent)
                runs[stream,arm,seed]=r
    combined={**reference,**runs}
    scoring=[]; lines=[]
    for stream in STREAMS:
        lines += [f'| {stream} seed | Own tail gap | Wait full gap | See tail gap | Bars meeting | Bars evaluated |',
                  '|---:|---:|---:|---:|---:|---:|']
        for seed in (0,1):
            values=[]; passes=0; evaluated=0
            for trick,left,right,key,bar in BARS:
                gap=difference(runs,stream,left,right,seed,key)
                values.append(gap); evaluated+=gap is not None; passes+=gap is not None and gap<=bar
                scoring.append(dict(stream=stream,seed=seed,trick=trick,quantity=left+' - '+right,
                                    range='full' if key=='bits_per_byte' else 'tail',gap=gap,threshold=bar,
                                    passes=None if gap is None else gap<=bar))
            lines.append(f'| {seed} | '+' | '.join(fmt(x) for x in values)+f' | {passes}/3 | {evaluated}/3 |')
        lines.append('')
    lines += ['| Trick | Bar | c4 seeds meeting | bible seeds meeting | mnist seeds meeting | Total seeds meeting | Seeds evaluated | Streams with 2/2 | enwik8 s0 gap | enwik8 s1 gap | enwik8 seeds meeting |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary=[]
    for trick,left,right,key,bar in BARS:
        rows=[r for r in scoring if r['trick']==trick]
        counts={stream:sum(r['passes'] is True for r in rows if r['stream']==stream) for stream in STREAMS}
        evaluated=sum(r['gap'] is not None for r in rows)
        gaps=[difference(reference,'enwik8',left,right,seed,key) for seed in (0,1)]
        row=dict(trick=trick,threshold=bar,per_stream_passes=counts,seeds_meeting=sum(counts.values()),
                 evaluated=evaluated,streams_with_both_seeds=sum(n==2 for n in counts.values()),
                 enwik8_gaps=gaps,enwik8_seeds_meeting=sum(x<=bar for x in gaps))
        summary.append(row)
        lines.append(f'| {trick} | {bar:.2f} | '+' | '.join(f'{counts[s]}/2' for s in STREAMS)+
                     f" | {row['seeds_meeting']}/6 | {evaluated}/6 | {row['streams_with_both_seeds']}/3 | "+
                     ' | '.join(fmt(x) for x in gaps)+f" | {row['enwik8_seeds_meeting']}/2 |")
    lines += ['', '| Stream | Arm | Seed | Tail bits | Full bits | First 40 KB bits | enwik8 tail | enwik8 full | enwik8 first 40 KB | Reference source |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    ablation=[]
    for stream in STREAMS:
        for arm in ARMS:
            for seed in (0,1):
                r=runs.get((stream,arm,seed),{}); ref=reference['enwik8',arm,seed]
                keys=('tail_bits_per_byte','bits_per_byte','first_40kb_bits_per_byte')
                vals=[r.get(k) for k in keys]+[ref[k] for k in keys]
                ablation.append(dict(stream=stream,arm=arm,seed=seed,**{k:r.get(k) for k in keys},
                                     enwik8={k:ref[k] for k in keys},reference_source=ref['source']))
                lines.append(f'| {stream} | {arm} | {seed} | '+' | '.join(fmt(x) for x in vals)+f" | {ref['source']} |")
    lines += ['', '| Stream | Seed | dense_6_ones − dense_6 tail | dense_6_ones − dense_6 full |',
              '|---|---:|---:|---:|']
    report_only=[]
    for stream in (*STREAMS,'enwik8'):
        for seed in (0,1):
            tail=difference(combined,stream,'dense_6_ones','dense_6',seed,'tail_bits_per_byte')
            full=difference(combined,stream,'dense_6_ones','dense_6',seed,'bits_per_byte')
            report_only.append(dict(stream=stream,seed=seed,tail_gap=tail,full_gap=full))
            lines.append(f'| {stream} | {seed} | {fmt(tail)} | {fmt(full)} |')
    lines += ['', '| Runs scored | Runs expected | Smoke | Tail start | Tail end |',
              '|---:|---:|---|---:|---:|',f"| {len(runs)} | 30 | {smoke.get('ok',False)} | {TAIL_START} | {N_DATA} |",
              '', '| Stream | Arm | Seed | Segment wall seconds | Run wall seconds | Kernel |',
              '|---|---|---:|---:|---:|---|']
    for stream in (*STREAMS,'enwik8'):
        for arm in ARMS:
            for seed in (0,1):
                r=combined.get((stream,arm,seed),{})
                lines.append(f'| {stream} | {arm} | {seed} | '+fmt(r.get('segment_wall_seconds',r.get('wall_seconds')))+
                             ' | '+fmt(r.get('wall_seconds'))+' | '+r.get('kernel_id','NA')+' |')
    for r in parts:
        lines.append(f"| {r['stream']} | {r['arm']} | {r['seed']} | {fmt(r['segment_wall_seconds'])} | {fmt(r['wall_seconds'])} | {r['kernel_id']} |")
    lines += ['', '| Stream | Bytes | SHA256 | Source SHA256 |', '|---|---:|---|---|']
    for stream,row in smoke.get('streams',{}).items():
        lines.append(f"| {stream} | {row['n_bytes']} | {row['sha256']} | {row['source_sha256']} |")
    lines += ['', '| Smoke check | Measurements |', '|---|---|']
    for row in smoke.get('log',[]):
        values=json.dumps({k:v for k,v in row.items() if k!='check'},sort_keys=True).replace('|','&#124;')
        lines.append('| '+row['check']+' | '+values+' |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'results.json',dict(n_bytes=N_DATA,tail_start=TAIL_START,scoring=scoring,summary=summary,ablation=ablation,
         report_only=report_only,runs=list(runs.values()),reference_runs=list(reference.values()),parts=parts,
         missing=missing,smoke=smoke,baseline_sha256=baseline_hashes,code_fingerprint=fingerprint))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'results/h34i'))
    assemble(p.parse_args().out)


