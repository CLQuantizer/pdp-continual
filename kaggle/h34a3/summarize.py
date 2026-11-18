"""H34a3 measurements and unchanged full-run thresholds; no verdicts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h34a3_cascade import ARMS, compute_account, dump
BARS = (('sum_1', -.10), ('plain_4', -.05), ('wide_1', -.03))


def number(value):
    return '—' if value is None else f'{value:.9f}'


def summarize(out, baseline):
    out = Path(out)
    old = json.loads(Path(baseline).read_text())
    refs = {(r['arm'], r['seed']): r for r in old['runs']}
    runs, missing = [], []
    manifest = json.loads((ROOT/'kaggle/h34a3/code-sha256.json').read_text())
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    smoke_path = out/'smoke.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    for arm in ARMS:
        for seed in (0, 1):
            path = out/'kernels'/f'{arm}-s{seed}'/'results.json'
            if not path.exists():
                missing.append(f'{arm}-s{seed}')
                continue
            row = json.loads(path.read_text())
            assert (row['arm'], row['seed'], row['n_bytes']) == (arm, seed, 300_000)
            assert row['kernel_id'] == f'ezzzzzx/h34a3-{arm.replace("_", "-")}-s{seed}'
            assert row['code_fingerprint'] == fingerprint
            assert smoke and smoke['ok'] and row['sha256_e10m'] == smoke['sha256_e10m']
            assert [(s['start'], s['end']) for s in row['slices']] == [(i,i+50_000) for i in range(0,300_000,50_000)]
            assert np.isfinite([row['bits_per_byte'], row['tail_bits_per_byte']]).all()
            runs.append(row)
    lookup = {(r['arm'],r['seed']):r for r in runs}
    scoring = []
    lines = ['| Range | Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds ≤ threshold | Evaluated |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for scope, metric in (('Full 0–300 KB','bits_per_byte'),('Tail 50–300 KB','tail_bits_per_byte')):
        for ref, threshold in BARS:
            values = []
            for seed in (0,1):
                row = lookup.get(('casc_4',seed))
                base = refs[ref,seed]
                assert base['n_bytes'] == 300_000
                if row:
                    assert row['sha256_e10m'] == base['sha256_e10m']
                values.append(row[metric]-base[metric] if row else None)
            present = [v for v in values if v is not None]
            mean = float(np.mean(present)) if present else None
            count = sum(v <= threshold for v in present)
            scoring.append(dict(range=scope,gap=f'casc_4 - {ref}',per_seed=values,mean=mean,
                                threshold=threshold,seeds_meeting=count,evaluated=len(present)))
            lines.append(f'| {scope} | casc_4 − {ref} | {number(values[0])} | {number(values[1])} | {number(mean)} | {threshold:.2f} | {count} | {len(present)} |')
    lines += ['', 'Bars apply to the full 300,000-byte run, both seeds. Tail counts use the same thresholds for reporting.',
              '', f'Scored kernels: {len(runs)}/4. Missing: {", ".join(missing) or "none"}.',
              'casc_1 reuses H34a2 sum_1; baseline data: '+str(Path(baseline).relative_to(ROOT) if Path(baseline).is_relative_to(ROOT) else baseline)+'.',
              '', '| Arm | Seed | Full bits/byte | Tail 50–300 KB | Wall seconds |', '|---|---:|---:|---:|---:|']
    for arm in (*ARMS,'casc_1'):
        for seed in (0,1):
            row = refs['sum_1',seed] if arm == 'casc_1' else lookup.get((arm,seed))
            lines.append(f'| {arm} | {seed} | {number(row["bits_per_byte"] if row else None)} | {number(row["tail_bits_per_byte"] if row else None)} | {number(row["wall_seconds"] if row else None)} |')
    means, tail_means = {}, {}
    lines += ['', '| Arm | n | Mean full bits/byte | Mean tail 50–300 KB |', '|---|---:|---:|---:|']
    for arm in ARMS:
        group = [r for r in runs if r['arm'] == arm]
        means[arm] = float(np.mean([r['bits_per_byte'] for r in group])) if group else None
        tail_means[arm] = float(np.mean([r['tail_bits_per_byte'] for r in group])) if group else None
        lines.append(f'| {arm} | {len(group)} | {number(means[arm])} | {number(tail_means[arm])} |')
    lines += ['', '| Arm | Seed | Start | End | Bits/byte |','|---|---:|---:|---:|---:|']
    for row in runs:
        for s in row['slices']:
            lines.append(f'| {row["arm"]} | {row["seed"]} | {s["start"]} | {s["end"]} | {number(s["bits_per_byte"])} |')
    lines += ['', '| Arm | Seed | Layer | Own bits/byte | Fraction abs(lambda)<0.5 |', '|---|---:|---:|---:|---:|']
    for row in runs:
        for k, layer in enumerate(row['layers'],1):
            lines.append(f'| {row["arm"]} | {row["seed"]} | {k} | {number(layer["own_bits_per_byte"])} | {number(layer["fraction_lambda_lt_half"])} |')
    lines += ['', '| Arm | Seed | Layer | Start | End | Own bits/byte | Mean abs(logit) | Max abs(logit) |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in runs:
        for k, layer in enumerate(row['layers'],1):
            for s in layer['slices']:
                lines.append(f'| {row["arm"]} | {row["seed"]} | {k} | {s["start"]} | {s["end"]} | {number(s["own_bits_per_byte"])} | {number(s["mean_abs_logit"])} | {number(s["max_abs_logit"])} |')
    compute = {arm:compute_account(arm) for arm in ARMS}
    lines += ['', '| Arm | Parameters | MACs/byte | Trace floats | State floats | Chunk trace-cache floats |', '|---|---:|---:|---:|---:|---:|']
    for arm,c in compute.items():
        lines.append(f'| {arm} | {c["parameters"]} | {c["macs_per_byte"]:.3f} | {c["trace_floats"]} | {c["state_floats"]} | {c["chunk_trace_cache_floats"]} |')
    lines += ['', 'H34a2 compute convention plus 3A MACs per additional partial-sum softmax and (D-1)A additions recomputed by the local-error helper. A=256.', '', 'Smoke output:', '', '```json', json.dumps(smoke,indent=2), '```', '', '| Kernel id | Wall seconds |', '|---|---:|']
    for arm in ARMS:
        for seed in (0,1):
            row = lookup.get((arm,seed))
            lines.append(f'| ezzzzzx/h34a3-{arm.replace("_","-")}-s{seed} | {number(row["wall_seconds"] if row else None)} |')
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'results.json',dict(scoring=scoring,runs=runs,means=means,tail_means=tail_means,missing=missing,compute=compute,smoke=smoke,
                                reused_casc1=[refs['sum_1',s] for s in (0,1)],baseline=str(baseline)))
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',default=str(ROOT/'results/h34a3'))
    p.add_argument('--baseline',default=str(ROOT/'results/h34a2/results.json'))
    args = p.parse_args()
    summarize(args.out,args.baseline)
