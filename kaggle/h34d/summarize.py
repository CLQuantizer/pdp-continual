"""Offline H34d scoring; incomplete pairs are NA, never counted as passes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h34d_dfa import ARMS, CHUNK, N_DATA, TAIL_START, dump, source_hashes


def fmt(x):
    return 'NA' if x is None else f'{x:.9f}'


def average(xs):
    return sum(xs)/len(xs) if all(x is not None for x in xs) else None


def reconstruct(r, start):
    end=r['n_bytes']; boundary=((start+24999)//25000)*25000
    total=(r['tail_edge_bits_per_byte']*(boundary-start)) if boundary>start else 0.
    count=boundary-start
    for s in r['slices']:
        if s['start']>=boundary:
            total+=s['bits_per_byte']*(s['end']-s['start']); count+=s['end']-s['start']
    assert count==end-start
    return total/count


def validate(r,arm,seed,smoke,manifest,fingerprint):
    assert smoke['ok'] and manifest['hashes']==smoke['source_hashes']==source_hashes()
    assert manifest['documents']==N_DATA==smoke['n_bytes']
    assert (r['arm'],r['seed'],r['n_bytes'],r['tail_start'])==(arm,seed,N_DATA,TAIL_START)
    assert r['kernel_id']==f"ezzzzzx/h34d-{arm.replace('_','-')}-s{seed}"
    assert r['sha256_e10m']==smoke['sha256_e10m'] and r['code_fingerprint']==fingerprint
    assert len(r['layers'])==4 and r['wall_seconds']<=5100
    assert [(s['start'],s['end']) for s in r['slices']]==[(i,min(i+25000,N_DATA)) for i in range(0,N_DATA,25000)]
    for key,start in (('bits_per_byte',0),('tail_bits_per_byte',TAIL_START)):
        assert math.isfinite(r[key]) and abs(reconstruct(r,start)-r[key])<=1e-9
    rows=r['dfa_norms']
    if arm=='plain_4':
        assert rows==[]
    else:
        expected=[(start,min(start+CHUNK,N_DATA),k) for start in range(0,N_DATA,CHUNK) for k in range(1,5)]
        assert [(v['start'],v['end'],v['layer']) for v in rows]==expected
        for v in rows:
            assert all(math.isfinite(v[key]) for key in ('own_norm','raw_dfa_norm','alpha','dfa_norm'))
            assert abs(v['own_norm']-v['dfa_norm'])<=1e-6*max(v['own_norm'],1e-30)
            assert v['only']==(arm=='dfa_only_4' and v['layer']>=2)


def assemble(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    path=out/'smoke.json'; smoke=json.loads(path.read_text()) if path.exists() else {}
    path=ROOT/'kaggle/h34d/code-sha256.json'; manifest=json.loads(path.read_text()) if path.exists() else None
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
        for arm in ('dfa_4','dfa_only_4'):
            gaps=[runs[arm,s][key]-runs['plain_4',s][key]
                  if (arm,s) in runs and ('plain_4',s) in runs else None for s in (0,1)]
            evaluated=sum(x is not None for x in gaps)
            bar=-.03 if (label,arm)==('Tail','dfa_4') else None
            count=sum(x<=bar for x in gaps if x is not None) if bar is not None else None
            scoring.append(dict(range=label,quantity=arm+' - plain_4',per_seed=gaps,mean=average(gaps),
                                threshold=bar,seeds_meeting=count,evaluated=evaluated))
            lines.append(f'| {label} | {arm} − plain_4 | {fmt(gaps[0])} | {fmt(gaps[1])} | {fmt(average(gaps))} | '+
                         ('≤ -0.03' if bar is not None else 'report only')+f' | {str(count)+"/2" if count is not None else "—"} | {evaluated}/2 |')
    lines+=['',f'H34d: {len(runs)}/6 scored kernels. Tail bytes 40,000–180,000; full bytes 0–180,000. Bar requires both seeds.',
            f"Smoke passed: {smoke.get('ok',False)}.", 'Missing: '+', '.join(missing)+'.' if missing else '',
            '', '| Arm | Tail seed 0 | Tail seed 1 | Tail mean | Full seed 0 | Full seed 1 | Full mean |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        vals=[[runs.get((arm,s),{}).get(key) for s in (0,1)] for key in ('tail_bits_per_byte','bits_per_byte')]
        lines.append('| '+arm+' | '+' | '.join(fmt(x) for xs in vals for x in (*xs,average(xs)))+' |')
    lines+=['','| Arm | Seed | Layer | Own full | Own tail | Mean gate tail | Final gate |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for (arm,seed),r in runs.items():
        for k,d in enumerate(r['layers'],1):
            lines.append(f'| {arm} | {seed} | {k} | '+' | '.join(fmt(d[key]) for key in
                ('own_bits_per_byte','tail_own_bits_per_byte','mean_mixer_weight_tail','final_mixer_weight'))+' |')
    lines+=['','| Smoke check | Measurements |','|---|---|']
    for row in smoke.get('log',[]):
        small={k:v for k,v in row.items() if k not in ('check','gradients')}
        lines.append('| '+row['check']+' | '+json.dumps(small,sort_keys=True)+' |')
    supplement=out/'float32.json'
    if supplement.exists():
        f32=json.loads(supplement.read_text())
        assert f32['ok'] and f32['source_hashes']==source_hashes()
        lines.append(f"| float32 own + DFA vs autograd, 512 bytes | max relative error {f32['max_relative_error']:.12g}; limit 1e-4 |")
    lines+=['','Feedback RNG uses one-based layer k=1..4: default_rng(7000 + 100·seed + k), normal(0, 1/16), shape (256,256).',
            'Alpha uses the Frobenius norm over each chunk of mean-loss hidden gradients. If the own norm is zero, alpha is zero. Readouts learn only the own loss.',
            'dfa_only_4 replaces the state gradient by DFA for layers 2–4; layer 1 keeps own + DFA. All chunk norms are retained in each results.json.',
            'Prefix smoke ends on a chunk boundary, following the inherited prequential chunk update protocol.',
            '', '| Kernel | Wall seconds |','|---|---:|']
    lines += [f"| {r['kernel_id']} | {r['wall_seconds']:.3f} |" for r in runs.values()]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'results.json',dict(n_bytes=N_DATA,tail_start=TAIL_START,scoring=scoring,
                               runs=list(runs.values()),missing=missing,smoke=smoke))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'results/h34d'))
    assemble(p.parse_args().out)
