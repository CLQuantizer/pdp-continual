"""Assemble H34a only after every kernel output has been fetched."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h34a_local_depth import ARMS, compute_account, dump

BARS = (('left_4 - left_1', 'left_4', 'left_1', -0.10),
        ('left_4 - plain_4', 'left_4', 'plain_4', -0.05),
        ('left_4 - wide_1', 'left_4', 'wide_1', -0.03))


def scoring_lines(lookup, key):
    lines = ['| Quantity | Seed 0 | Seed 1 | Mean | Threshold | Seeds ≤ threshold |',
             '|---|---:|---:|---:|---:|---:|']
    for name, left, right, thr in BARS:
        vals = [lookup[left, s][key] - lookup[right, s][key] for s in (0, 1)]
        mean = float(np.mean(vals))
        n = int(sum(v <= thr for v in vals))
        lines.append(f"| {name} | {vals[0]:.9f} | {vals[1]:.9f} | {mean:.9f} | {thr:.2f} | {n} |")
    return lines


def assemble(out):
    out = Path(out)
    smoke_result = json.loads((out/'smoke.json').read_text())
    paths = [out/'kernels'/f'{arm}-s{seed}'/'results.json' for arm in ARMS for seed in (0, 1)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing kernels: '+', '.join(missing))
    runs = [json.loads(path.read_text()) for path in paths]
    for r, path in zip(runs, paths):
        assert path.parent.name == f"{r['arm']}-s{r['seed']}"
        assert r['kernel_id'] == f"ezzzzzx/h34a-{r['arm'].replace('_','-')}-s{r['seed']}"
        selected = [s for s in r['slices'] if s['start'] >= 50_000 and s['end'] <= 300_000]
        assert sum(s['end']-s['start'] for s in selected) == 250_000
        r['tail_bits_per_byte'] = sum(s['bits_per_byte']*(s['end']-s['start']) for s in selected)/250_000
    assert smoke_result['ok'] and len(runs) == 16
    assert all(r['n_bytes'] == smoke_result['n_bytes'] for r in runs)
    lookup = {(r['arm'], r['seed']): r for r in runs}
    means = {arm: float(np.mean([lookup[arm, s]['bits_per_byte'] for s in (0, 1)])) for arm in ARMS}
    slice_means = {a:[dict(start=s['start'],end=s['end'],bits_per_byte=float(np.mean([lookup[a,k]['slices'][i]['bits_per_byte'] for k in (0,1)]))) for i,s in enumerate(lookup[a,0]['slices'])] for a in ARMS}
    pairs = [('left_4', 'left_1'), ('plain_4', 'plain_1'), ('left_4', 'plain_4'),
             ('left_2', 'plain_2'), ('left_4', 'wide_1'), ('left_4_noskip', 'left_4')]
    gaps = []
    for a, b in pairs:
        values = [lookup[a, s]['bits_per_byte']-lookup[b, s]['bits_per_byte'] for s in (0, 1)]
        gaps.append(dict(gap=f'{a} - {b}', per_seed=values, mean=float(np.mean(values))))
    compute = {arm: compute_account(arm) for arm in ARMS}
    ratio = compute['wide_1']['macs_per_byte']/compute['left_4']['macs_per_byte']
    formulas = dict(
        convention='Real GEMM entry = 1 MAC; complex multiply or multiply-add = 4 real MACs. Adam = 6 P/chunk. Transcendentals, comparisons, reductions, memory copies and scoring-only diagnostics excluded.',
        parameters='A E + 2N + 2N U + 2N A + 4N R + A R, plus D mixer weights; E=32 with embedding, otherwise 0.',
        macs='2NU + 8N + 4NU + 4NA I + 3(2NA+4NR+AR) + 3R + 16N + 8NU + 4NA I + 4NAE + 4N/128 + 3A + 3(2N+R) + Fprev + 6P/128 per layer; mixer 2DA+3A+6D/128. I=1 iff embedding present.',
        traces='2N(1+U+A I) real floats per layer: complex lambda, input-map and embedding traces; hidden state 2N extra; chunk caches 128 times persistent trace floats.')
    result = dict(n_bytes=smoke_result['n_bytes'], runs=runs, means=means, gaps=gaps, slice_means=slice_means,
                  tail_means={a:float(np.mean([lookup[a,s]['tail_bits_per_byte'] for s in (0,1)])) for a in ARMS},
                  compute=compute, compute_formulas=formulas, wide_over_left4_macs=ratio,
                  wide_compute_diff_gt_20pct=bool(abs(ratio-1) > .2), smoke=smoke_result,
                  h33_reference_first_1mb=dict(bank=3.097826, counts_mix6=2.177547, GRU=2.104744),
                  kernel_ids=[r['kernel_id'] for r in runs])
    dump(out/'results.json', result)
    lines = ['# H34a', '', '## Scoring', '',
             'Full run. Each row is the gap per seed, the mean, and how many of the 2 seeds are ≤ the threshold.', '',
             *scoring_lines(lookup, 'bits_per_byte'), '',
             'Tail 50–300 KB. Same three gaps.', '',
             *scoring_lines(lookup, 'tail_bits_per_byte'), '',
             f"Bytes per life: {result['n_bytes']:,}. CPU; float32; chunks 128; seeds 0, 1.", '',
             '| Arm | Seed 0 bits/byte | Seed 1 bits/byte | Mean |', '|---|---:|---:|---:|']
    for arm in ARMS:
        lines.append(f"| {arm} | {lookup[arm,0]['bits_per_byte']:.9f} | {lookup[arm,1]['bits_per_byte']:.9f} | {means[arm]:.9f} |")
    lines += ['', '| Arm | Tail 50–300 KB seed 0 | Seed 1 | Mean |', '|---|---:|---:|---:|']
    for arm in ARMS:
        lines.append(f"| {arm} | {lookup[arm,0]['tail_bits_per_byte']:.9f} | {lookup[arm,1]['tail_bits_per_byte']:.9f} | {result['tail_means'][arm]:.9f} |")
    lines += ['', '| Gap | Seed 0 | Seed 1 | Mean |', '|---|---:|---:|---:|']
    for g in gaps:
        lines.append(f"| {g['gap']} | {g['per_seed'][0]:.9f} | {g['per_seed'][1]:.9f} | {g['mean']:.9f} |")
    lines += ['', '| Arm | Seed | Slice start | Slice end | Bits/byte |', '|---|---:|---:|---:|---:|']
    for run_result in runs:
        for s in run_result['slices']:
            lines.append(f"| {run_result['arm']} | {run_result['seed']} | {s['start']} | {s['end']} | {s['bits_per_byte']:.9f} |")
    lines += ['', '| Arm | Slice start | Slice end | Mean bits/byte over seeds |', '|---|---:|---:|---:|']
    for arm in ARMS:
        for s in slice_means[arm]:
            lines.append(f"| {arm} | {s['start']} | {s['end']} | {s['bits_per_byte']:.9f} |")
    lines += ['', '| Arm | Seed | Layer | Own bits/byte | Mean weight, last 50 KB | Fraction abs(lambda)<0.5 | RMS min | RMS mean | RMS max |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        for k, layer in enumerate(r['layers'], 1):
            rms = np.asarray(layer['feature_rms'])
            lines.append(f"| {r['arm']} | {r['seed']} | {k} | {layer['own_bits_per_byte']:.9f} | {layer['mean_mixer_weight_last_50kb']:.9f} | {layer['fraction_lambda_lt_half']:.9f} | {rms.min():.9g} | {rms.mean():.9g} | {rms.max():.9g} |")
    lines += ['', 'Full per-feature final running RMS vectors: `results.json`, `runs[].layers[].feature_rms` (z then quadratic features).', '',
              '| Arm | Parameters incl. mixer | MACs/byte, forward + learning | Trace floats | State floats | Chunk trace-cache floats |',
              '|---|---:|---:|---:|---:|---:|']
    for arm, c in compute.items():
        lines.append(f"| {arm} | {c['parameters']} | {c['macs_per_byte']:.3f} | {c['trace_floats']} | {c['state_floats']} | {c['chunk_trace_cache_floats']} |")
    lines += ['', *(f'{key}: {value}' for key, value in formulas.items()), '',
              f"wide_1 / left_4 MACs = {ratio:.9f}; difference exceeds 20% = {result['wide_compute_diff_gt_20pct']}.", '',
              '| H33 reference, first 1 MB (not directly comparable) | Bits/byte |', '|---|---:|',
              '| bank | 3.097826 |', '| counts_mix6 | 2.177547 |', '| GRU | 2.104744 |', '',
              '| Kernel id | Wall seconds |', '|---|---:|']
    lines += [f"| {r['kernel_id']} | {r['wall_seconds']:.3f} |" for r in runs]
    lines += ['', 'Smoke-test output (1 CPU thread; check 1 float64, copied reference initialization; all scored runs float32):',
              '', '```json', json.dumps(smoke_result, indent=2), '```', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default=str(ROOT/'results/h34a'))
    assemble(p.parse_args().out)
