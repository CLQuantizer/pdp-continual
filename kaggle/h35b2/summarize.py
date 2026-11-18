"""Numeric report; select lag on s6–8 only, count bars on s9–11 only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h35b2_snapshot import POLICIES, SEEDS, dump
from dispatch import validate_payload


def number(value):
    return '—' if value is None else f'{value:.9f}'


def select_lag(lookup):
    means = {}
    for lag in (5,20):
        rows = [lookup.get((s,f'prog_snap_{lag}')) for s in (6,7,8)]
        means[lag] = float(np.mean([r['T_clean'] for r in rows])) if all(rows) else None
    chosen = min((5,20), key=lambda k:(means[k],k)) if all(v is not None for v in means.values()) else None
    return chosen, means


def summarize(out=ROOT/'results/h35b2', base=ROOT/'kaggle/h35b2'):
    out, base = Path(out), Path(base)
    manifest_path = base/'code-sha256.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    fingerprint = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest() if manifest else None
    runs, kernels, missing = [], [], []
    for job in manifest['jobs'] if manifest else []:
        path = out/'kernels'/job['tag']/'results.json'
        if not path.exists():
            missing.append(job['id'])
            kernels.append(dict(id=job['id'],wall_seconds=None))
            continue
        payload = json.loads(path.read_text())
        validate_payload(payload,job,fingerprint,smoke=json.loads((out/'smoke.json').read_text()))
        runs.extend(dict(r,kernel_id=job['id']) for r in payload['runs'])
        kernels.append(dict(id=job['id'],wall_seconds=payload['wall_seconds']))
    lookup = {(r['seed'],r['policy']):r for r in runs}
    assert len(lookup) == len(runs)
    chosen, means = select_lag(lookup)
    scores, counts, evaluated = [], dict(bar1=0,bar2=0), dict(bar1=0,bar2=0)
    lines = ['| Seed | Phase | k | Chosen k | Oracle − rand | Prog_snap_k − rand | 0.5·(oracle − rand) | Prog_snap_k − rho | ≤ half | ≤ −0.02 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for lag in (5,20):
            r,o,p,h = [lookup.get((seed,policy)) for policy in ('rand','oracle',f'prog_snap_{lag}','rho')]
            oracle_rand = o['T_clean']-r['T_clean'] if o and r else None
            prog_rand = p['T_clean']-r['T_clean'] if p and r else None
            half = .5*oracle_rand if oracle_rand is not None else None
            prog_rho = p['T_clean']-h['T_clean'] if p and h else None
            tests = dict(bar1=prog_rand <= half if prog_rand is not None and half is not None else None,
                         bar2=prog_rho <= -.02 if prog_rho is not None else None)
            if seed in (9,10,11) and chosen == lag:
                for key,value in tests.items():
                    if value is not None:
                        evaluated[key] += 1
                        counts[key] += int(value)
            row = dict(seed=seed,phase='selection' if seed < 9 else 'scoring',lag=lag,
                       chosen=None if chosen is None else int(lag==chosen),
                       oracle_minus_rand=oracle_rand,prog_snap_minus_rand=prog_rand,
                       half_oracle_minus_rand=half,prog_snap_minus_rho=prog_rho,
                       **{key:int(value) if value is not None else None for key,value in tests.items()})
            scores.append(row)
            lines.append(f'| {seed} | {row["phase"]} | {lag} | '+('—' if chosen is None else str(int(lag==chosen)))+' | '+
                ' | '.join(number(v) for v in (oracle_rand,prog_rand,half,prog_rho))+' | '+
                ' | '.join('—' if v is None else str(int(v)) for v in tests.values())+' |')
    lines += ['', '| Threshold (chosen k, seeds 9–11) | Count | Evaluated | Required |','|---|---:|---:|---:|']
    for key in counts:
        lines.append(f'| {key} | {counts[key]} | {evaluated[key]} | 2/3 |')
    lines += ['', '| k | Mean T_clean, seeds 6–8 | Chosen |','|---:|---:|---:|']
    for lag in (5,20):
        lines.append(f'| {lag} | {number(means[lag])} | '+('—' if chosen is None else str(int(chosen==lag)))+' |')
    lines += ['', 'Selection: lower mean T_clean on seeds 6–8; exact ties choose k=5. Missing selection runs leave k unselected.',
              '', '| Policy | Metric | s6 | s7 | s8 | s9 | s10 | s11 | Mean | Std (ddof=0) |',
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
                    values.append(next(c['clean_bpb'] for c in row['checkpoints'] if c['documents']==int(metric[2:])))
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
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'results.json',dict(scoring=scores,chosen_lag=chosen,selection_means=means,
        threshold_counts=counts,evaluated=evaluated,runs=runs,aggregates=aggregates,
        kernels=kernels,missing=missing,smoke=smoke))
    (out/'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',default=str(ROOT/'results/h35b2'))
    p.add_argument('--base',default=str(ROOT/'kaggle/h35b2'))
    args = p.parse_args()
    summarize(args.out,args.base)
