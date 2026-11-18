"""H36 numbers only: incomplete smoke/scored output is explicitly missing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h36_curious_patching import RULES, dump


def number(x):
    return '—' if x is None else f'{x:.9f}'


def summarize(out):
    out = Path(out)
    runs, missing = [], []
    for rule in RULES:
        for seed in range(6):
            path = out/'kernels'/f'{rule}-s{seed}'/'results.json'
            if not path.exists():
                missing.append(f'{rule}-s{seed}')
                continue
            r = json.loads(path.read_text())
            assert r['rule'] == rule and r['seed'] == seed
            assert r['kernel_id'] == f'ezzzzzx/h36-{rule}-s{seed}'
            assert r['corpus'] == 'h35_corpus.build' and r['documents'] in (300,400)
            assert np.isfinite(r['clean_test_bpb']) and abs(r['life']['rho']-.25) <= .02
            runs.append(r)
    if runs:
        assert len({r['documents'] for r in runs}) == 1
        assert len({r['code_fingerprint'] for r in runs}) == 1
    lookup = {(r['rule'],r['seed']):r for r in runs}
    scores = []
    for seed in range(6):
        rows = {rule:lookup.get((rule,seed)) for rule in RULES}
        f,e,g,p = (rows[rule] for rule in RULES)
        ratio = e['life']['junk_share']/f['life']['junk_share'] if e and f and f['life']['junk_share'] else None
        gain = g['clean_test_bpb']-e['clean_test_bpb'] if g and e else None
        progress = p['clean_test_bpb']-e['clean_test_bpb'] if p and e else None
        fixed = f['clean_test_bpb']-e['clean_test_bpb'] if f and e else None
        scores.append(dict(seed=seed,entropy_over_fixed_junk_share=ratio,gain_minus_entropy=gain,
                           progress_minus_entropy=progress,fixed_minus_entropy=fixed))
    evaluated1 = sum(r['entropy_over_fixed_junk_share'] is not None for r in scores)
    evaluated2 = sum(r['gain_minus_entropy'] is not None for r in scores)
    passes1 = sum(r['entropy_over_fixed_junk_share'] is not None and r['entropy_over_fixed_junk_share'] >= 1.2 for r in scores)
    passes2 = sum(r['gain_minus_entropy'] is not None and r['gain_minus_entropy'] <= -.03 for r in scores)
    smoke_path = out/'smoke.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    result = dict(scoring=scores,pass_counts=dict(bar1=passes1,bar2=passes2,evaluated1=evaluated1,evaluated2=evaluated2),
                  runs=runs,missing=missing,smoke=smoke,kernel_ids=[f'ezzzzzx/h36-{rule}-s{seed}' for rule in RULES for seed in range(6)])
    lines = ['| Seed | Entropy / fixed junk share | Gain − entropy clean bits/byte | Progress − entropy clean bits/byte | Fixed − entropy clean bits/byte |',
             '|---|---:|---:|---:|---:|']
    for row in scores:
        lines.append(f"| {row['seed']} | {number(row['entropy_over_fixed_junk_share'])} | {number(row['gain_minus_entropy'])} | {number(row['progress_minus_entropy'])} | {number(row['fixed_minus_entropy'])} |")
    lines += ['',f'Bar 1 pass count: {passes1}/{evaluated1}; required 5/6. Bar 2 pass count: {passes2}/{evaluated2}; required 5/6.',
              '',f'Scored kernels: {len(runs)}/24.','',
              '| Rule | Seed | Clean test bits/byte | Life clean bits | Life junk bits | Life clean bits/byte | Life junk bits/byte | Junk step share | ρ | Test ρ | Wall seconds |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    # Keep the revised smoke quantities visible immediately after scoring.
    marker = lines.index('| Rule | Seed | Clean test bits/byte | Life clean bits | Life junk bits | Life clean bits/byte | Life junk bits/byte | Junk step share | ρ | Test ρ | Wall seconds |')
    smoke_lines = ['| Smoke rule | Bytes | ρ | Junk share | Estimated 400-doc wall minutes |',
                   '|---|---:|---:|---:|---:|']
    for rule in RULES:
        row = (smoke or {}).get('allocation',{}).get(rule,{})
        smoke_lines.append(f"| {rule} | {row.get('n_bytes','—')} | {number(row.get('rho'))} | {number(row.get('junk_share'))} | {number(row.get('estimated_400_document_wall_minutes'))} |")
    for check in (smoke or {}).get('h35_checks',{}).values():
        smoke_lines += ['', f"H35 build(0), {check['rule']}, {check['n_bytes']} bytes: ρ = {number(check['rho'])}; bits/byte = {number(check['bits_per_byte'])}; junk share = {number(check['junk_share'])}."]
    if (smoke or {}).get('failure'):
        smoke_lines += ['', 'Smoke exception: '+smoke['failure']+'.']
    smoke_lines += ['']
    lines[marker:marker] = smoke_lines
    for r in runs:
        life = r['life']
        vals = [r['clean_test_bpb'],life['life_bits_clean'],life['life_bits_junk'],life['life_bpb_clean'],life['life_bpb_junk'],life['junk_share'],life['rho'],r['test_rho'],r['wall_seconds']]
        lines.append(f"| {r['rule']} | {r['seed']} | "+' | '.join(number(v) for v in vals)+' |')
    lines += ['','| Rule | Seed | Junk kind | Share of all expensive steps |','|---|---:|---|---:|']
    for r in runs:
        for kind,share in r['life']['junk_share_by_kind'].items():
            lines.append(f"| {r['rule']} | {r['seed']} | {kind} | {number(share)} |")
    metrics = ['clean_test_bpb','life_bits_clean','life_bits_junk','life_bpb_clean','life_bpb_junk','junk_share','rho','test_rho','wall_seconds']
    result['aggregates'] = {}
    lines += ['','| Rule | Metric | n | Mean | Std (ddof=0) |','|---|---|---:|---:|---:|']
    for rule in RULES:
        group = [r for r in runs if r['rule'] == rule]
        result['aggregates'][rule] = {}
        kinds = sorted({k for r in group for k in r['life']['junk_share_by_kind']})
        for metric in metrics + ['junk_share_by_kind.'+k for k in kinds]:
            vals = []
            for r in group:
                if metric.startswith('junk_share_by_kind.'):
                    val = r['life']['junk_share_by_kind'].get(metric.split('.',1)[1],0)
                else:
                    val = r.get(metric,r['life'].get(metric))
                if val is not None:
                    vals.append(val)
            mean,std = (float(np.mean(vals)),float(np.std(vals,ddof=0))) if vals else (None,None)
            result['aggregates'][rule][metric] = dict(n=len(vals),mean=mean,std=std)
            lines.append(f'| {rule} | {metric} | {len(vals)} | {number(mean)} | {number(std)} |')
    lines += ['','Smoke output:','','```json',json.dumps(smoke,indent=2),'```','',
              '| Kernel id | Wall seconds |','|---|---:|']
    for rule in RULES:
        for seed in range(6):
            row = lookup.get((rule,seed))
            lines.append(f"| ezzzzzx/h36-{rule}-s{seed} | {number(row['wall_seconds'] if row else None)} |")
    lines += ['']
    dump(out/'results.json',result)
    (out/'RESULTS.md').write_text('\n'.join(lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',default=str(ROOT/'results/h36'))
    summarize(parser.parse_args().out)
