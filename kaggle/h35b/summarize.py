"""H35b numeric summaries, with missing scored seeds left empty."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h35b_neural_diet import POLICIES, SEEDS, dump
from dispatch import validate_payload


def number(value):
    return '—' if value is None else f'{value:.9f}'


def summarize(out=ROOT/'results/h35b', base=ROOT/'kaggle/h35b'):
    out, base = Path(out), Path(base)
    manifest_path = base/'code-sha256.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest() if manifest else None
    jobs = manifest['jobs'] if manifest else []
    runs, kernels, missing = [], [], []
    for job in jobs:
        path = out/'kernels'/job['tag']/'results.json'
        if not path.exists():
            missing.append(job['id'])
            kernels.append(dict(id=job['id'],wall_seconds=None))
            continue
        payload = json.loads(path.read_text())
        validate_payload(payload,job,fingerprint)
        runs.extend(payload['runs'])
        kernels.append(dict(id=job['id'],wall_seconds=payload['wall_seconds']))
        for row in payload['runs']:
            row['kernel_id'] = job['id']
    lookup = {(r['seed'],r['policy']):r for r in runs}
    assert len(lookup) == len(runs)
    scores = []
    counts = dict(precondition=0,bar1=0,bar2=0)
    evaluated = dict(precondition=0,bar1=0,bar2=0)
    for seed in SEEDS:
        r,o,p,h = [lookup.get((seed,policy)) for policy in ('rand','oracle','prog_peek','high_loss')]
        oracle_rand = o['T_clean']-r['T_clean'] if o and r else None
        prog_rand = p['T_clean']-r['T_clean'] if p and r else None
        half = .5*oracle_rand if oracle_rand is not None else None
        prog_loss = p['T_clean']-h['T_clean'] if p and h else None
        tests = dict(precondition=oracle_rand <= -.10 if oracle_rand is not None else None,
                     bar1=prog_rand <= half if prog_rand is not None and half is not None else None,
                     bar2=prog_loss <= -.05 if prog_loss is not None else None)
        for key,value in tests.items():
            if value is not None:
                evaluated[key] += 1
                counts[key] += int(value)
        scores.append(dict(seed=seed,oracle_minus_rand=oracle_rand,prog_peek_minus_rand=prog_rand,
                           half_oracle_minus_rand=half,prog_peek_minus_high_loss=prog_loss,
                           **{key:int(value) if value is not None else None for key,value in tests.items()}))
    lines = ['| Seed | Oracle − rand | Prog_peek − rand | 0.5·(oracle − rand) | Prog_peek − high_loss | ≤ −0.10 | ≤ half | ≤ −0.05 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in scores:
        keys = ('oracle_minus_rand','prog_peek_minus_rand','half_oracle_minus_rand','prog_peek_minus_high_loss')
        lines.append(f"| {row['seed']} | "+' | '.join(number(row[k]) for k in keys)+' | '+
                     ' | '.join(str(row[k]) if row[k] is not None else '—' for k in counts)+' |')
    lines += ['', '| Threshold | Count | Evaluated | Required |', '|---|---:|---:|---:|']
    for key in counts:
        lines.append(f'| {key} | {counts[key]} | {evaluated[key]} | 5/6 |')
    lines += ['', '| Policy | Metric | s6 | s7 | s8 | s9 | s10 | s11 | Mean | Std (ddof=0) |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    aggregates = {}
    metrics = ['T_clean','final'] + [f'cp{n}' for n in range(25,201,25)] + ['share.'+k for k in ('clean','J1_random','J2_corrupt','J3_dup','J4_boiler','J5_shuffle')] + ['wall_seconds']
    for policy in POLICIES:
        aggregates[policy] = {}
        for metric in metrics:
            values = []
            for seed in SEEDS:
                row = lookup.get((seed,policy))
                if row is None:
                    values.append(None)
                elif metric.startswith('share.'):
                    values.append(row['shares'][metric.split('.',1)[1]])
                elif metric.startswith('cp'):
                    n = int(metric[2:]); values.append(next(c['clean_bpb'] for c in row['checkpoints'] if c['documents']==n))
                else:
                    values.append(row[metric])
            finite = [v for v in values if v is not None]
            mean,std = (float(np.mean(finite)),float(np.std(finite,ddof=0))) if finite else (None,None)
            aggregates[policy][metric] = dict(n=len(finite),mean=mean,std=std)
            lines.append(f'| {policy} | {metric} | '+' | '.join(number(v) for v in values+[mean,std])+' |')
    lines += ['', '| Kernel id | Wall seconds |','|---|---:|']
    for row in kernels:
        lines.append(f"| {row['id']} | {number(row['wall_seconds'])} |")
    smoke_path = out/'smoke.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    lines += ['', '```json', json.dumps(smoke,indent=2), '```','']
    dump(out/'results.json',dict(scoring=scores,threshold_counts=counts,evaluated=evaluated,
                               runs=runs,aggregates=aggregates,kernels=kernels,missing=missing,smoke=smoke))
    (out/'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',default=str(ROOT/'results/h35b'))
    p.add_argument('--base',default=str(ROOT/'kaggle/h35b'))
    args = p.parse_args()
    summarize(args.out,args.base)
