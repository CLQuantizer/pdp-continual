"""Assemble all 20 fetched H34a2 lives; no scoring verdicts."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h34a2_logit_sum import ARMS, compute_account, dump
PAIRS=(('sum_4','sum_1'),('sum_2','sum_1'),('sum_4','sumown_4'),('sum_2','sumown_2'),
       ('sum_4','wide_1'),('plain_4','plain_1'),('sum_4','plain_4'),('sum_4_noskip','sum_4'))


def assemble(out):
    out=Path(out)
    paths=[out/'kernels'/f'{a}-s{s}'/'results.json' for a in ARMS for s in (0,1)]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing: raise FileNotFoundError('Missing kernels: '+', '.join(missing))
    smoke=json.loads((out/'smoke.json').read_text())
    assert smoke['ok'] and smoke['n_bytes']==300_000
    runs=[json.loads(p.read_text()) for p in paths]
    for r,p in zip(runs,paths):
        assert r['n_bytes']==300_000 and p.parent.name==f"{r['arm']}-s{r['seed']}"
        assert r['kernel_id']==f"ezzzzzx/h34a2-{r['arm'].replace('_','-')}-s{r['seed']}"
        assert [(s['start'],s['end']) for s in r['slices']]==[(i,i+50_000) for i in range(0,300_000,50_000)]
        assert r['sha256_e10m']==smoke['sha256_e10m']
        assert np.isfinite(r['bits_per_byte'])
    lookup={(r['arm'],r['seed']):r for r in runs}
    means={a:float(np.mean([lookup[a,s]['bits_per_byte'] for s in (0,1)])) for a in ARMS}
    tails={a:float(np.mean([lookup[a,s]['tail_bits_per_byte'] for s in (0,1)])) for a in ARMS}
    slices={a:[dict(start=i*50_000,end=(i+1)*50_000,bits_per_byte=float(np.mean([
        lookup[a,s]['slices'][i]['bits_per_byte'] for s in (0,1)]))) for i in range(6)] for a in ARMS}
    gaps=[]
    for a,b in PAIRS:
        values=[lookup[a,s]['bits_per_byte']-lookup[b,s]['bits_per_byte'] for s in (0,1)]
        gaps.append(dict(gap=f'{a} - {b}',per_seed=values,mean=float(np.mean(values))))
    compute={a:compute_account(a) for a in ARMS}
    ratio=compute['wide_1']['macs_per_byte']/compute['sum_4']['macs_per_byte']
    formulas=dict(convention='Real GEMM entry = 1 MAC; complex multiply/add = 4 real MACs; Adam = 6P/128. Transcendentals, comparisons, reductions, copies and scoring-only diagnostics excluded.',
                  parameters='AE + 2N + 2NU + 2NA + 4NR + AR per layer; plus D weights for plain only. A=256, R=64, E=32 iff byte embedding, U=E+Fprev.',
                  macs='Per layer: 2NU + 8N + 4NU + 4NA I + 3(2NA+4NR+AR) + 3R + 16N + 8NU + 4NA I + 4NAE + 4N/128 + 3A + (2Fprev+2) Iprev + 6P/128. I=1 iff embedding, Iprev=1 iff a lower layer exists. Normalization counts one square and one division per feature plus mean scaling and epsilon addition; the reduction and square root are excluded. Plain combiner: 2DA+3A+6D/128; fixed sum: (D-1)A+3A.',
                  traces='2N(1+U+AI) real floats per layer; state 2N extra; chunk caches 128 times trace floats.',
                  normalization='f / (sqrt(mean(f^2 across features at this step)) + 1e-6), detached; no running statistics.')
    result=dict(n_bytes=300_000,runs=runs,means=means,tail_means=tails,slice_means=slices,gaps=gaps,
                compute=compute,compute_formulas=formulas,wide_over_sum4_macs=ratio,
                wide_compute_diff_gt_20pct=bool(abs(ratio-1)>.2),smoke=smoke,kernel_ids=[r['kernel_id'] for r in runs])
    lines=['# H34a2','','Bytes per life: 300,000. CPU; float32; chunks 128; seeds 0, 1.','',
           '| Arm | Seed 0 bits/byte | Seed 1 | Mean | Tail 50–300 KB seed 0 | Tail seed 1 | Tail mean |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        x,y=lookup[a,0],lookup[a,1]
        lines.append(f"| {a} | {x['bits_per_byte']:.9f} | {y['bits_per_byte']:.9f} | {means[a]:.9f} | {x['tail_bits_per_byte']:.9f} | {y['tail_bits_per_byte']:.9f} | {tails[a]:.9f} |")
    lines+=['','| Arm | Start | End | Seed 0 bits/byte | Seed 1 | Mean |','|---|---:|---:|---:|---:|---:|']
    for a in ARMS:
        for i,s in enumerate(slices[a]):
            lines.append(f"| {a} | {s['start']} | {s['end']} | {lookup[a,0]['slices'][i]['bits_per_byte']:.9f} | {lookup[a,1]['slices'][i]['bits_per_byte']:.9f} | {s['bits_per_byte']:.9f} |")
    lines+=['','| Gap | Seed 0 | Seed 1 | Mean |','|---|---:|---:|---:|']
    for g in gaps: lines.append(f"| {g['gap']} | {g['per_seed'][0]:.9f} | {g['per_seed'][1]:.9f} | {g['mean']:.9f} |")
    lines+=['','| Arm | Seed | Layer | Own bits/byte | Mean mixer weight last 50 KB (plain) | Fraction abs(lambda)<0.5 |','|---|---:|---:|---:|---:|---:|']
    for r in runs:
        for k,d in enumerate(r['layers'],1):
            w=d.get('mean_mixer_weight_last_50kb','')
            lines.append(f"| {r['arm']} | {r['seed']} | {k} | {d['own_bits_per_byte']:.9f} | {w} | {d['fraction_lambda_lt_half']:.9f} |")
    lines+=['','| Arm | Seed | Layer | Start | End | Own bits/byte | Mean abs(logit) | Max abs(logit) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        for k,d in enumerate(r['layers'],1):
            for s in d['slices']:
                lines.append(f"| {r['arm']} | {r['seed']} | {k} | {s['start']} | {s['end']} | {s['own_bits_per_byte']:.9f} | {s['mean_abs_logit']:.9f} | {s['max_abs_logit']:.9f} |")
    lines+=['','| Arm | Parameters | MACs/byte forward + learning | Trace floats | State floats | Chunk trace-cache floats |','|---|---:|---:|---:|---:|---:|']
    for a,c in compute.items():
        lines.append(f"| {a} | {c['parameters']} | {c['macs_per_byte']:.3f} | {c['trace_floats']} | {c['state_floats']} | {c['chunk_trace_cache_floats']} |")
    lines+=['',*(f'{k}: {v}' for k,v in formulas.items()),'',f'wide_1 / sum_4 MACs = {ratio:.9f}; difference exceeds 20% = {abs(ratio-1)>.2}.','',
            '| Kernel id | Wall seconds |','|---|---:|']
    lines += [f"| {r['kernel_id']} | {r['wall_seconds']:.3f} |" for r in runs]
    lines += ['','Smoke output:','','```json',json.dumps(smoke,indent=2),'```','']
    dump(out/'results.json',result)
    (out/'RESULTS.md').write_text('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'results/h34a2'))
    assemble(p.parse_args().out)
