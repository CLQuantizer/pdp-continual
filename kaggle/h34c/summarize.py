"""Score H34c. Missing kernels stay NA and do not count as meeting a bar."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h34c_grow import ARMS, N_BYTES, SCHEDULE, SETTLE_REL, SLICE, TAIL_END, TAIL_START, WINDOW, dump
SEEDS = (0, 1)
GAPS = ('grow_4_sched', 'grow_4_settle', 'grow_4_freeze')


def fmt(x):
    return 'NA' if x is None else f'{x:.9f}'


def average(values):
    return float(np.mean(values)) if values and all(x is not None for x in values) else None


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _check_run(row, smoke, fingerprint, arm, seed):
    if (row['arm'], row['seed'], row['n_bytes']) != (arm, seed, N_BYTES):
        raise AssertionError(f'{arm} s{seed} identity')
    if row['kernel_id'] != f'ezzzzzx/h34c-{arm.replace("_", "-")}-s{seed}':
        raise AssertionError(row['kernel_id'])
    if row['sha256_e10m'] != smoke['sha256_e10m'] or row['code_fingerprint'] != fingerprint:
        raise AssertionError(f'{arm} s{seed} fingerprint')
    if not (_finite(row['bits_per_byte']) and _finite(row['tail_bits_per_byte']) and _finite(row['wall_seconds'])):
        raise AssertionError(f'{arm} s{seed} bits')
    slices = row['slices']
    expect = [(i, min(i + SLICE, N_BYTES)) for i in range(0, N_BYTES, SLICE)]
    if [(s['start'], s['end']) for s in slices] != expect:
        raise AssertionError(f'{arm} s{seed} slices')
    full = sum(s['bits_per_byte'] * (s['end'] - s['start']) for s in slices) / N_BYTES
    tail_slices = [s for s in slices if s['start'] >= TAIL_START]
    tail_n = TAIL_END - TAIL_START
    tail = sum(s['bits_per_byte'] * (s['end'] - s['start']) for s in tail_slices) / tail_n
    if abs(full - row['bits_per_byte']) > 1e-9 or abs(tail - row['tail_bits_per_byte']) > 1e-9:
        raise AssertionError(f'{arm} s{seed} slice reconstruction')
    added = row['added_at']
    if arm == 'plain_4':
        if added != [0, 0, 0, 0] or row['n_short_chunks'] != 1 or row['identity_bytes'] or row['settle_checks']:
            raise AssertionError('plain_4 schedule')
        if row['frozen_parameters_identical'] is not None or row['identity_max_probability_abs'] is not None:
            raise AssertionError('plain_4 grew or froze')
    elif arm in ('grow_4_sched', 'grow_4_freeze'):
        if added != [0, *SCHEDULE[arm]] or row['identity_bytes'] != list(SCHEDULE[arm]) or row['n_short_chunks'] != 4:
            raise AssertionError(f'{arm} schedule {added} {row["identity_bytes"]} short={row["n_short_chunks"]}')
        for key in ('identity_max_probability_abs', 'identity_max_bits_abs', 'identity_max_added_logit_abs'):
            if not _finite(row[key]) or row[key] > 1e-9:
                raise AssertionError(f'{arm} identity {key}')
        if row['identity_max_new_weight_abs_from_1'] != 0.0:
            raise AssertionError(f'{arm} new mixer weight')
        if arm == 'grow_4_freeze':
            if row['frozen_parameters_identical'] is not True or row['frozen_parameter_max_abs'] != 0.0:
                raise AssertionError('freeze did not hold parameters')
        elif row['frozen_parameters_identical'] is not None:
            raise AssertionError('sched froze a layer')
    else:
        if added[0] != 0 or len(added) > 4 or row['n_short_chunks'] != 1:
            raise AssertionError(f'settle shape {added}')
        for a, b in zip(added, added[1:]):
            if b < a + 2 * WINDOW or b % WINDOW != 0 or b >= N_BYTES:
                raise AssertionError(f'settle add {a} -> {b}')
        checks = row['settle_checks']
        triggered = [c['byte'] for c in checks if c['added']]
        observed = [c['byte'] for c in checks]
        if observed != list(range(2 * WINDOW, (observed[-1] if observed else 2 * WINDOW) + 1, WINDOW)):
            raise AssertionError(f'settle checks are not contiguous: {observed[:4]}...{observed[-4:]}')
        if added != [0, *triggered]:
            raise AssertionError('settle log does not match added_at')
        if row['identity_bytes'] != triggered:
            raise AssertionError('settle identity bytes')
        for c in checks:
            if c['byte'] % WINDOW != 0 or c['byte'] < 2 * WINDOW:
                raise AssertionError('settle check cadence')
            if c['ready']:
                if not (_finite(c['prev_bits']) and _finite(c['last_bits']) and _finite(c['relative_improvement'])):
                    raise AssertionError('settle window')
                if (c['relative_improvement'] < SETTLE_REL) != bool(c['added']):
                    raise AssertionError('settle threshold')
            elif c['added']:
                raise AssertionError('settle added before two windows')
        if len(added) == 4 and any(c['byte'] > added[-1] for c in checks):
            raise AssertionError('settle checked after depth 4')
        if triggered and (not _finite(row['identity_max_probability_abs']) or row['identity_max_probability_abs'] > 1e-9):
            raise AssertionError('settle identity')
    if len(row['layers']) != len(added):
        raise AssertionError('layer rows')
    for k, layer in enumerate(row['layers']):
        if layer['layer'] != k + 1 or layer['added_at'] != added[k]:
            raise AssertionError('layer index')
        if not _finite(layer['own_bits_per_byte']) or not _finite(layer['final_mixer_weight']):
            raise AssertionError('layer bits')
        if added[k] < TAIL_END and not _finite(layer['tail_own_bits_per_byte']):
            raise AssertionError('tail own bits')
        if added[k] < TAIL_END and not _finite(layer['mean_mixer_weight_tail']):
            raise AssertionError('tail mixer weight')


def assemble(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    smoke_path = out / 'smoke.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    manifest_path = ROOT / 'kaggle/h34c/code-sha256.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() if manifest else None
    runs, missing = [], []
    for arm in ARMS:
        for seed in SEEDS:
            path = out / 'kernels' / f'{arm}-s{seed}' / 'results.json'
            if not path.exists():
                missing.append(f'{arm}-s{seed}')
                continue
            row = json.loads(path.read_text())
            if not smoke.get('ok') or not manifest or manifest['hashes'] != smoke.get('source_hashes'):
                raise AssertionError('Refusing to score without a matching passed smoke')
            _check_run(row, smoke, fingerprint, arm, seed)
            runs.append(row)
    lookup = {(r['arm'], r['seed']): r for r in runs}

    def value(arm, seed, key):
        return lookup.get((arm, seed), {}).get(key)

    def gap(arm, seed):
        x, y = value(arm, seed, 'tail_bits_per_byte'), value('plain_4', seed, 'tail_bits_per_byte')
        return x - y if x is not None and y is not None else None

    lines = [
        '| Bar (tail bits/byte) | Seed 0 | Seed 1 | Mean | Threshold | Seeds meeting | Evaluated |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    bar_vals = []
    for seed in SEEDS:
        opts = [value(a, seed, 'tail_bits_per_byte') for a in ('grow_4_sched', 'grow_4_settle')]
        base = value('plain_4', seed, 'tail_bits_per_byte')
        bar_vals.append(min(opts) - base if base is not None and all(x is not None for x in opts) else None)
    bar_count = sum(x <= -0.03 for x in bar_vals if x is not None)
    bar_eval = sum(x is not None for x in bar_vals)
    lines.append('| min(grow_4_sched, grow_4_settle) − plain_4 | '
                 + ' | '.join(fmt(x) for x in (*bar_vals, average(bar_vals)))
                 + f' | ≤ -0.03 | {bar_count}/2 | {bar_eval} |')
    lines += ['',
              '| Tail quantity | Seed 0 | Seed 1 | Mean | Seeds meeting −0.03 | Evaluated |',
              '|---|---:|---:|---:|---:|---:|']
    scores = []
    for arm in GAPS:
        vals = [gap(arm, seed) for seed in SEEDS]
        count = sum(x <= -0.03 for x in vals if x is not None)
        evaluated = sum(x is not None for x in vals)
        label = f'{arm} − plain_4' + (' (report only)' if arm == 'grow_4_freeze' else '')
        scores.append(dict(quantity=f'{arm} - plain_4', per_seed=vals, mean=average(vals),
                           threshold=-0.03, count=count, evaluated=evaluated, report_only=arm == 'grow_4_freeze'))
        lines.append(f'| {label} | ' + ' | '.join(fmt(x) for x in (*vals, average(vals))) + f' | {count}/2 | {evaluated} |')
    lines += ['',
              f'H34c. Tail bytes {TAIL_START:,}–{TAIL_END:,}; full bytes 0–{N_BYTES:,}. '
              f'Bar requires both seeds. grow_4_freeze is report only. Scored kernels: {len(runs)}/8.',
              f'Smoke passed: {smoke.get("ok", False)}.',
              'Settle adds layer k+1 when (previous window − last window) / previous window < 0.005 '
              'on layer k own bits/byte. Windows are 4,096 bytes, checked every 4,096, depth at most 4.',
              'Scheduled adds are bytes 20,000, 40,000 and 60,000. A new mixer weight starts at 1. '
              'A new layer starts with a zero readout and a zero recurrent state.']
    if missing:
        lines.append('Missing: ' + ', '.join(missing) + '.')
    lines += ['',
              '| Arm | Tail seed 0 | Tail seed 1 | Tail mean | Full seed 0 | Full seed 1 | Full mean |',
              '|---|---:|---:|---:|---:|---:|---:|']
    arm_rows = []
    for arm in ARMS:
        tails = [value(arm, seed, 'tail_bits_per_byte') for seed in SEEDS]
        fulls = [value(arm, seed, 'bits_per_byte') for seed in SEEDS]
        arm_rows.append(dict(arm=arm, tail=tails, tail_mean=average(tails), full=fulls, full_mean=average(fulls)))
        lines.append('| ' + arm + ' | ' + ' | '.join(fmt(x) for x in (*tails, average(tails), *fulls, average(fulls))) + ' |')
    lines += ['',
              '| Arm | Seed | Layer | Added byte | Own full bits | Own tail bits | Tail mixer weight | Final mixer weight |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in runs:
        for layer in row['layers']:
            vals = [layer['added_at'], layer['own_bits_per_byte'], layer['tail_own_bits_per_byte'],
                    layer['mean_mixer_weight_tail'], layer['final_mixer_weight']]
            shown = [str(vals[0])] + [fmt(x) for x in vals[1:]]
            lines.append(f"| {row['arm']} | {row['seed']} | {layer['layer']} | " + ' | '.join(shown) + ' |')
    lines += ['',
              '| Arm | Seed | Check byte | Top layer | Ready | Previous window bits | Last window bits | Relative improvement | Added |',
              '|---|---:|---:|---:|---|---:|---:|---:|---|']
    settle_rows = 0
    for row in runs:
        if row['arm'] != 'grow_4_settle':
            continue
        for c in row['settle_checks']:
            settle_rows += 1
            lines.append('| grow_4_settle | {seed} | {byte} | {top} | {ready} | {prev} | {last} | {rel} | {added} |'.format(
                seed=row['seed'], byte=c['byte'], top=c['top_layer'], ready=int(bool(c['ready'])),
                prev=fmt(c['prev_bits']), last=fmt(c['last_bits']), rel=fmt(c['relative_improvement']),
                added=int(bool(c['added']))))
    if settle_rows == 0:
        lines.append('| grow_4_settle | NA | NA | NA | NA | NA | NA | NA | NA |')
    lines += ['',
              '| Arm | Start byte | End byte | Seed 0 bits/byte | Seed 1 | Mean |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for j, start in enumerate(range(0, N_BYTES, SLICE)):
            vals = []
            for seed in SEEDS:
                row = lookup.get((arm, seed))
                vals.append(row['slices'][j]['bits_per_byte'] if row else None)
            lines.append(f'| {arm} | {start} | {min(start + SLICE, N_BYTES)} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(average(vals))} |')
    lines += ['', '| Smoke check | Quantity | Value | Limit |', '|---|---|---:|---:|']
    for row in smoke.get('log', []):
        check = row['check']
        if check == 'h34b_equivalence':
            lines.append(f"| h34b equivalence | plain_4 seed {row['seed']} max bits difference, 8,192 bytes | {row['max_bits_abs']:.12g} | ≤ 1e-9 |")
        elif check == 'add_identity':
            lines.append(f"| add identity | grow_4_sched probability change at byte 20,000 | {row['max_probability_abs']:.12g} | ≤ 1e-9 |")
            lines.append(f"| add identity | grow_4_sched bits change at the add chunk | {row['max_bits_abs']:.12g} | ≤ 1e-9 |")
        elif check == 'freeze_parameters':
            lines.append(f"| freeze | parameter max abs change after byte 20,000 | {row['max_abs']:.12g} | = 0 |")
        elif check == 'prequential_prefix':
            lines.append(f"| prequential | {row['arm']} extended prefix max bits difference | {row['max_bits_abs']:.12g} | = 0 |")
            lines.append(f"| prequential | {row['arm']} changed suffix max prefix bits difference | {row['changed_suffix_max_bits_abs']:.12g} | = 0 |")
        elif check == 'wall_estimates':
            for name, seconds in row['estimated_kernel_seconds'].items():
                lines.append(f'| wall | {name} estimated minutes at 180,000 bytes | {seconds / 60:.6f} | ≤ 90 |')
    lines += ['', '| Kernel id | Wall seconds |', '|---|---:|']
    lines += [f"| {r['kernel_id']} | {r['wall_seconds']:.3f} |" for r in runs]
    lines.append('')
    dump(out / 'results.json', dict(
        n_bytes=N_BYTES, tail_start=TAIL_START, tail_end=TAIL_END, slice_size=SLICE,
        runs=runs, missing=missing,
        bar=dict(quantity='min(grow_4_sched, grow_4_settle) - plain_4', per_seed=bar_vals,
                 mean=average(bar_vals), threshold=-0.03, count=bar_count, evaluated=bar_eval),
        scoring=scores, arms=arm_rows, smoke=smoke))
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'results/h34c'))
    assemble(parser.parse_args().out)
