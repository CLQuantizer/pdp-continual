"""H34j numerical results; absent lives remain NA and unevaluated."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import h34j_long as h
BASE = ROOT/'kaggle/h34j'
BARS = (('zgate_6','plain_1',-.10), ('dense_6','zgate_6',-.02), ('dense_8','dense_6',-.01))


def fmt(x):
    return 'NA' if x is None else f'{x:.9f}'


def validate_part(r, job, manifest, fingerprint, folder):
    assert manifest['hashes'] == h.source_hashes()
    assert r['code_fingerprint'] == fingerprint
    assert (r['kernel_id'],r['arm'],r['seed'],r['n_bytes'],r['part'],r['start'],r['end']) == (
        job['id'],job['arm'],job['seed'],h.N_DATA,job['part'],job['start'],job['end'])
    assert r['raw_sha256'] == manifest['raw_sha256'] and r['sha256_e10m'] == manifest['sha256_e10m']
    assert 0 < r['segment_wall_seconds'] <= r['kernel_wall_seconds'] <= 5100
    assert r['segment_wall_seconds'] <= r['wall_seconds'] <= job['part']*5100
    assert h.sha((folder/'checkpoint.pt').read_bytes()) == r['checkpoint_sha256']
    if job['part'] == 1:
        assert r['resumed_checkpoint_sha256'] is None
        assert r['wall_seconds'] == r['segment_wall_seconds']
    if job['end'] < h.N_DATA:
        assert (r['bytes_seen'],r['offset'],r['pending_bytes']) == (job['end'],job['end'],0)
        return
    assert r['tail_start'] == h.TAIL_START
    assert r['initial_mixer_weights'] == h.initial_gate(job['arm'])[0].tolist()
    edges = [(x,x+h.SLICE) for x in range(0,h.N_DATA,h.SLICE)]
    assert [(s['start'],s['end']) for s in r['slices']] == edges
    assert [(s['start'],s['end']) for s in r['mixer_slices']] == edges
    for key,lo,hi in (('bits_per_byte',0,h.N_DATA),('tail_bits_per_byte',h.TAIL_START,h.N_DATA),
                      ('first_200kb_bits_per_byte',0,h.TAIL_START)):
        total = sum(s['bits_per_byte']*(s['end']-s['start']) for s in r['slices'] if lo<=s['start']<hi)
        assert math.isfinite(r[key]) and abs(total/(hi-lo)-r[key]) < 1e-9
    depth = int(job['arm'].split('_')[1]); assert len(r['layers']) == depth
    for s in r['mixer_slices']:
        assert len(s['mean_weights']) == depth and all(math.isfinite(x) for x in s['mean_weights'])
    for layer in r['layers']:
        assert [(s['start'],s['end']) for s in layer['slices']] == edges
        for key, lo, n in (('own_bits_per_byte',0,h.N_DATA), ('tail_own_bits_per_byte',h.TAIL_START,h.N_DATA-h.TAIL_START)):
            value = sum(s['own_bits_per_byte']*h.SLICE for s in layer['slices'] if s['start']>=lo)/n
            assert math.isfinite(layer[key]) and abs(value-layer[key]) < 1e-9


def assemble(out):
    from dispatch import job_plan
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((BASE/'code-sha256.json').read_text())
    smoke = json.loads((out/'smoke.json').read_text())
    assert smoke['ok'] and smoke['source_hashes'] == manifest['hashes'] == h.source_hashes()
    fingerprint = h.sha(json.dumps(manifest,sort_keys=True).encode())
    runs, parts, missing, validated = {}, [], [], {}
    for job in job_plan(manifest):
        folder = out/'kernels'/job['tag']; path = folder/'results.json'
        if not path.exists():
            missing.append(job['tag']); continue
        r = json.loads(path.read_text()); validate_part(r,job,manifest,fingerprint,folder)
        if job['parent_tag']:
            assert job['parent_tag'] in validated, 'Missing parent output'
            parent = validated[job['parent_tag']]
            assert r['resumed_checkpoint_sha256'] == parent['checkpoint_sha256']
            assert abs(r['wall_seconds']-r['segment_wall_seconds']-parent['wall_seconds']) < 1e-6
        validated[job['tag']] = r; parts.append(r)
        if job['end'] == h.N_DATA: runs[job['arm'],job['seed']] = r
    lines, scoring = [], []
    for label,key in (('Tail 200000–1000000','tail_bits_per_byte'),('Full 0–1000000','bits_per_byte')):
        lines += [f'| {label} gap | Seed 0 | Seed 1 | Mean | Bar | Seeds meeting | Seeds evaluated |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for left,right,bar in BARS:
            gaps = [runs[left,s][key]-runs[right,s][key] if (left,s) in runs and (right,s) in runs else None for s in (0,1)]
            mean = sum(gaps)/2 if all(x is not None for x in gaps) else None
            passes = sum(x is not None and x<=bar for x in gaps); evaluated = sum(x is not None for x in gaps)
            scoring.append(dict(range='tail' if key.startswith('tail') else 'full',left=left,right=right,
                seed_gaps=gaps,mean_gap=mean,threshold=bar,seeds_meeting=passes,seeds_evaluated=evaluated,
                preregistered=key.startswith('tail')))
            lines.append(f'| {left} − {right} | {fmt(gaps[0])} | {fmt(gaps[1])} | {fmt(mean)} | {bar:.2f} | {passes}/2 | {evaluated}/2 |')
        lines.append('')
    lines += ['| Arm | Seed | Tail bits/byte | Full bits/byte | First 200 KB bits/byte |', '|---|---:|---:|---:|---:|']
    for arm in h.ARMS:
        for seed in (0,1):
            r = runs.get((arm,seed),{})
            lines.append(f'| {arm} | {seed} | '+' | '.join(fmt(r.get(k)) for k in ('tail_bits_per_byte','bits_per_byte','first_200kb_bits_per_byte'))+' |')
    lines += ['', '| Arm | Seed | Slice start | Slice end | Bits/byte | Bits |', '|---|---:|---:|---:|---:|---:|']
    for arm in h.ARMS:
        for seed in (0,1):
            slices = runs.get((arm,seed),{}).get('slices',[])
            for idx,start in enumerate(range(0,h.N_DATA,h.SLICE)):
                value = slices[idx]['bits_per_byte'] if slices else None
                lines.append(f'| {arm} | {seed} | {start} | {start+h.SLICE} | {fmt(value)} | {fmt(value*h.SLICE if value is not None else None)} |')
    lines += ['', '| Kernel | Start | End | Segment wall s | Cumulative run wall s | Kernel wall s |', '|---|---:|---:|---:|---:|---:|']
    for job in job_plan(manifest):
        r = validated.get(job['tag'],{})
        lines.append(f"| {job['id']} | {job['start']} | {job['end']} | "+' | '.join(fmt(r.get(k)) for k in ('segment_wall_seconds','wall_seconds','kernel_wall_seconds'))+' |')
    lines += ['', '| Lives scored | Lives expected | Parts fetched | Parts expected | Tail start | End |',
              '|---:|---:|---:|---:|---:|---:|',f'| {len(runs)} | 8 | {len(parts)} | {len(job_plan(manifest))} | {h.TAIL_START} | {h.N_DATA} |',
              '', '| Smoke check | Measurements |', '|---|---|']
    for row in smoke['log']:
        lines.append('| '+row['check']+' | '+json.dumps({k:v for k,v in row.items() if k!='check'},sort_keys=True).replace('|','&#124;')+' |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    h.dump(out/'results.json',dict(n_bytes=h.N_DATA,tail_start=h.TAIL_START,scoring=scoring,
        runs=list(runs.values()),parts=parts,missing=missing,smoke=smoke,code_fingerprint=fingerprint))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'results/h34j'))
    assemble(p.parse_args().out)
