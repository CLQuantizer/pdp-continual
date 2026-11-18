"""Local H34b scoring; incomplete runs remain NA and never count as passes."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h34b_plain_depth import ARMS, EARLY_BITS_LIMIT, N_DATA, TAIL_START, compute_account, dump
BARS=(('plain_6','plain_4',-.03,'≤'),
      ('plain_4_fixmix','plain_4',.05,'≥'),
      ('plain_6','wide_6',-.10,'≤'))

def fmt(x):
    return 'NA' if x is None else f'{x:.9f}'

def bits_from_slices(slices, start, end, edge):
    """Mean of [start, end) from 25 KB slice means.

    A tail that begins at 40,000 splits the slice [25,000, 50,000). `edge` is
    the mean on [start, next slice boundary). The full run starts on a boundary
    and does not use `edge`.
    """
    slice_size=25_000
    if end<=start:
        raise AssertionError('empty window')
    aligned=((start+slice_size-1)//slice_size)*slice_size
    if aligned>end:
        raise AssertionError('window ends before the next slice boundary')
    total=0.0
    if aligned>start:
        if edge is None:
            raise AssertionError('tail edge mean missing')
        total+=edge*(aligned-start)
    got=aligned-start
    for s in slices:
        if s['end']<=aligned or s['start']>=end:
            continue
        if s['start']<aligned or s['end']>end:
            raise AssertionError('slice crosses the reconstructed window')
        total+=s['bits_per_byte']*(s['end']-s['start'])
        got+=s['end']-s['start']
    if got!=end-start:
        raise AssertionError(f'reconstructed {got} bytes of {end-start}')
    return total/(end-start)

def average(values):
    return float(np.mean(values)) if all(x is not None for x in values) else None

def assemble(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    smoke_path=out/'smoke.json'
    smoke=json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    n=smoke.get('n_bytes',N_DATA)
    manifest_path=ROOT/'kaggle/h34b/code-sha256.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    fingerprint=__import__('hashlib').sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest() if manifest else None
    runs=[]; missing=[]
    for arm in ARMS:
        for seed in (0,1):
            p=out/'kernels'/f'{arm}-s{seed}'/'results.json'
            if not p.exists(): missing.append(f'{arm}-s{seed}'); continue
            r=json.loads(p.read_text())
            assert smoke.get('ok') and manifest and manifest['hashes']==smoke['source_hashes']
            assert manifest['documents']==n
            assert (r['arm'],r['seed'],r['n_bytes'])==(arm,seed,n)
            assert r['kernel_id']==f"ezzzzzx/h34b-{arm.replace('_','-')}-s{seed}"
            assert r['sha256_e10m']==smoke['sha256_e10m'] and r['code_fingerprint']==fingerprint
            assert len(r['layers'])==(1 if arm=='wide_6' else int(arm.split('_')[1]))
            assert [(x['start'],x['end']) for x in r['slices']]==[(i,min(i+25000,n)) for i in range(0,n,25000)]
            for key in ('bits_per_byte','tail_bits_per_byte','wall_seconds'):
                assert math.isfinite(r[key])
            for key,start in (('bits_per_byte',0),('tail_bits_per_byte',TAIL_START)):
                edge=None if start==0 else r.get('tail_edge_bits_per_byte')
                reconstructed=bits_from_slices(r['slices'],start,n,edge)
                assert abs(reconstructed-r[key])<1e-9
            if arm.endswith('fixmix'):
                assert r['max_frozen_weight_change']==0
                assert all(x['final_mixer_weight']==.25 for x in r['layers'])
            runs.append(r)
    lookup={(r['arm'],r['seed']):r for r in runs}
    def value(a,s,key): return lookup.get((a,s),{}).get(key)
    def gap(a,b,s,key):
        x,y=value(a,s,key),value(b,s,key)
        return x-y if x is not None and y is not None else None
    lines=['| Tail quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |',
           '|---|---:|---:|---:|---:|---:|---:|']
    scores=[]
    for a,b,bar,op in BARS:
        vals=[gap(a,b,s,'tail_bits_per_byte') for s in (0,1)]
        count=sum(x<=bar if op=='≤' else x>=bar for x in vals if x is not None)
        evaluated=sum(x is not None for x in vals)
        scores.append(dict(quantity=f'{a} - {b}',per_seed=vals,mean=average(vals),threshold=bar,operator=op,count=count,evaluated=evaluated))
        lines.append(f'| {a} − {b} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(average(vals))} | {op} {bar:+.2f} | {count}/2 | {evaluated} |')
    lines+=['',f'H34b. Tail bytes {TAIL_START:,}–{n:,}; full bytes 0–{n:,}. Bars require both seeds. Scored kernels: {len(runs)}/14.',
            f"Smoke passed: {smoke.get('ok',False)}. No thresholds apply to noskip or full-run gaps."]
    if n==200000: lines+=['All arms reduced to 200,000 bytes by the pre-registered 85-minute timing rule.']
    if missing: lines+=['Missing: '+', '.join(missing)+'.']
    lines+=['','| Arm | Tail seed 0 | Tail seed 1 | Tail mean | Full seed 0 | Full seed 1 | Full mean | First 40 KB seed 0 | First 40 KB seed 1 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        ts=[value(a,s,'tail_bits_per_byte') for s in (0,1)]
        fs=[value(a,s,'bits_per_byte') for s in (0,1)]
        early=[(f*n-t*(n-TAIL_START))/TAIL_START if f is not None and t is not None else None for f,t in zip(fs,ts)]
        lines.append('| '+a+' | '+' | '.join(fmt(x) for x in (*ts,average(ts),*fs,average(fs),*early))+' |')
    lines+=['','| Range | Gap (report only) | Seed 0 | Seed 1 | Mean |','|---|---|---:|---:|---:|']
    gaps=[]
    for key,label,pairs in (('tail_bits_per_byte','Tail',[('plain_4_noskip','plain_4')]),
                            ('bits_per_byte','Full',[(a,b) for a,b,_,_ in BARS]+[('plain_4_noskip','plain_4')])):
        for a,b in pairs:
            vals=[gap(a,b,s,key) for s in (0,1)]
            gaps.append(dict(range=label,quantity=f'{a} - {b}',per_seed=vals,mean=average(vals)))
            lines.append(f'| {label} | {a} − {b} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(average(vals))} |')
    lines+=['','| Arm | Seed | Layer | Own full bits | Own tail bits | Mean gate tail | Mean gate last 50 KB | Final gate |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        for k,d in enumerate(r['layers'],1):
            vals=[d[x] for x in ('own_bits_per_byte','tail_own_bits_per_byte','mean_mixer_weight_tail','mean_mixer_weight_last_50kb','final_mixer_weight')]
            lines.append(f"| {r['arm']} | {r['seed']} | {k} | "+' | '.join(fmt(x) for x in vals)+' |')
    lines+=['','| Arm | Start byte | End byte | Seed 0 bits/byte | Seed 1 | Mean |','|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        for j,i in enumerate(range(0,n,25000)):
            vals=[lookup[a,s]['slices'][j]['bits_per_byte'] if (a,s) in lookup else None for s in (0,1)]
            lines.append(f'| {a} | {i} | {min(i+25000,n)} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(average(vals))} |')
    compute={a:compute_account(a) for a in ARMS}
    ratio=compute['wide_6']['macs_per_byte']/compute['plain_6']['macs_per_byte']
    lines+=['','| Smoke quantity | Value | Limit |','|---|---:|---:|']
    for row in smoke.get('log',[]):
        check=row['check']
        if check=='h34a2_equivalence':
            lines.append(f"| plain_4 equivalence, seed {row['seed']}, maximum bits difference | {row['max_bits_abs']:.12g} | ≤ 1e-9 |")
        elif check=='fixed_mixer':
            lines.append(f"| Fixed gate maximum change | {row['max_change']:.12g} | = 0 |")
        elif check=='prequential_prefix':
            lines.append(f"| plain_6 extended prefix maximum bits difference | {row['max_bits_abs']:.12g} | = 0 |")
            lines.append(f"| plain_6 changed suffix maximum prefix bits difference | {row['changed_suffix_max_bits_abs']:.12g} | = 0 |")
        elif check=='stability_and_timing':
            lines.append(f"| {row['arm']} first 2,048 bytes, bits/byte | {row['first_2048_bits']:.9f} | ≤ {EARLY_BITS_LIMIT} |")
        elif check=='wall_estimates':
            for arm,seconds in row['estimated_kernel_seconds'].items():
                lines.append(f'| {arm} estimated wall minutes at {n:,} bytes | {seconds/60:.6f} | ≤ 85 |')
    if smoke.get('failures'):
        lines+=['','Smoke failures: '+ '; '.join(smoke['failures'])+'.']
    lines+=['','| Arm | Trainable parameters | MACs/byte forward + learning | Trace floats | State floats | Chunk trace-cache floats |',
            '|---|---:|---:|---:|---:|---:|']
    for a,c in compute.items():
        lines.append(f"| {a} | {c['parameters']} | {c['macs_per_byte']:.3f} | {c['trace_floats']} | {c['state_floats']} | {c['chunk_trace_cache_floats']} |")
    lines+=['',f'wide_6 / plain_6 MACs = {ratio:.9f}; difference >20% = {abs(ratio-1)>.2}.',
            '', 'MAC convention: H34a2 real-operation accounting, forward plus learning. Real GEMM entry = 1 MAC; complex multiply/add = 4 real MACs; Adam = 6P/128. Transcendentals, comparisons, reductions, copies and scoring diagnostics excluded. Per-layer terms are in results.json compute.layers[].terms.',
            '', 'Stateless normalization: detached f / (sqrt(mean(f²)) + 1e-6). Learned gates start at 1 per layer and are unconstrained (H34a2); fixed gates stay at 1/4. One weight set; Adam mixer lr 0.003, layer lr 0.001; chunk 128; float32.',
            '', '| Kernel id | Wall seconds |','|---|---:|']
    lines += [f"| {r['kernel_id']} | {r['wall_seconds']:.3f} |" for r in runs]
    lines+=['','Smoke evidence: smoke.json and smoke.log. Scored runs pending remote dispatch.' if missing else '','']
    dump(out/'results.json',dict(n_bytes=n,runs=runs,missing=missing,scoring=scores,gaps=gaps,compute=compute,
                               wide_over_plain6_macs=ratio,wide_compute_diff_gt_20pct=bool(abs(ratio-1)>.2),smoke=smoke))
    (out/'RESULTS.md').write_text('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'results/h34b'))
    assemble(p.parse_args().out)
