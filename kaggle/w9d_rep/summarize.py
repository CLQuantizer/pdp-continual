"""W9d numeric tables. Refuses partial/malformed scored results; --pending retains NA for missing scored lives."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(96, 120))
ARMS = ('fixed_random', 'fixed_night', 'pressure_eyelid', 'pressure_retina')
# Orchestrator bars for 24 seeds. Thresholds unchanged. Bar 2 requires 16 of 24.
# fixed_night - fixed_random is report-only.
GAPS = (
    ('pressure_retina', 'fixed_random', -.07, 16),
    ('pressure_retina', 'fixed_night', .05, None),
    ('fixed_night', 'fixed_random', None, None),
)


def validate(job, dest):
    payload = json.loads((Path(dest) / job['output']).read_text())
    hashes = json.loads((ROOT / 'kaggle/w9d_rep/code-sha256.json').read_text())
    if payload.get('code_sha256') != hashes:
        raise AssertionError('source hashes')
    if payload['seed'] != job['seed'] or payload['kernel_id'] != job['kernel_id']:
        raise AssertionError('kernel identity')
    if payload['workers'] != min(4, len(job['arms'])):
        raise AssertionError('kernel workers')
    if sorted(r['arm'] for r in payload['runs']) != sorted(job['arms']):
        raise AssertionError('kernel arms')
    for row in payload['runs']:
        if (row['seed'] != job['seed'] or row['L'] != 2048 or row['period'] != 128
                or row['night_enabled'] is not True or row['finite'] is not True):
            raise AssertionError('row protocol')
        if [cp['decision'] for cp in row['checkpoints']] != list(range(256, 2049, 256)):
            raise AssertionError('checkpoint decisions')
        scores = [cp['W14'] for cp in row['checkpoints']]
        if not all(math.isfinite(v) for v in scores):
            raise AssertionError('nonfinite W14')
        if not math.isclose(sum(scores) / 8, row['T14'], rel_tol=0, abs_tol=1e-12):
            raise AssertionError('T14 average')
        if not math.isclose(row['W14'], scores[-1], rel_tol=0, abs_tol=1e-12):
            raise AssertionError('final W14')
        if not math.isclose(row['sleep_share'], row['sleep_decisions'] / 2048, rel_tol=0, abs_tol=1e-12):
            raise AssertionError('sleep share')
        visits = row['sleep_visits']
        if any(not 1 <= v['length'] <= 8 or v['t'] < 0 or v['t'] + v['length'] > 2048
               or (v['length'] < 8 and v['t'] + v['length'] != 2048) for v in visits):
            raise AssertionError('sleep visits')
        if any(b['t'] < a['t'] + a['length'] for a, b in zip(visits, visits[1:])):
            raise AssertionError('overlapping sleep')
        asleep = {t for v in visits for t in range(v['t'], v['t'] + v['length'])}
        night_n = sum(t % 256 >= 192 for t in asleep)
        if len(asleep) != row['sleep_decisions'] or night_n != row['night_sleep_decisions']:
            raise AssertionError('sleep counts')
        expected = night_n / len(asleep) if asleep else None
        if expected != row['night_sleep_share']:
            raise AssertionError('night sleep share')
        if row['arm'] in ('fixed_random', 'fixed_night') and len(asleep) != 512:
            raise AssertionError('fixed dose')
        if row['arm'] == 'fixed_night' and asleep != {t for t in range(2048) if t % 256 >= 192}:
            raise AssertionError('fixed night clock')
        if row['arm'] == 'nosleep' and asleep:
            raise AssertionError('nosleep')
        if any(not x['intact'] for x in row['sleep_checks']):
            raise AssertionError('live sleep state')
        if row['arm'] == 'pressure_eyelid':
            validate_eyelid(row, asleep)
        if row['arm'] == 'pressure_retina':
            validate_retina(row, asleep)
    return payload


def validate_eyelid(row, asleep):
    """Independently reconstruct the controller from all glance codelengths."""
    from statistics import median
    observations = row['input_bits']
    assert [r['t'] for r in observations] == list(range(row['L']))
    history, ema, S, active, forced, left = [], None, 0, False, False, 0
    for t, r in enumerate(observations):
        if left == 0:
            if forced:
                forced = S > 128
                active = forced
            elif S >= 256:
                active = forced = True
            elif active:
                active = S > 0 and ema > 1.5 * median(history)
            else:
                active = S >= 32 and ema is not None and ema > 1.5 * median(history)
            if active:
                left = min(8, row['L'] - t)
        sleeping = left > 0
        assert sleeping == (t in asleep) == bool(r['asleep']), 'eyelid schedule'
        x = r['bits_per_symbol']
        assert math.isfinite(x) and x >= 0
        ema = x if ema is None else .9 * ema + .1 * x
        if sleeping:
            S = max(0, S - 3)
            left -= 1
        else:
            S += 1
            history = (history + [ema])[-256:]
        assert S == r['S'] and forced == bool(r['forced']), 'eyelid pressure'
        assert math.isclose(ema, r['b_t'], rel_tol=0, abs_tol=1e-12), 'eyelid EMA'
        assert math.isclose(median(history), r['m_t'], rel_tol=0, abs_tol=1e-12), 'wake median'
    for check in row['eyelid_checks']:
        assert check['t'] in asleep and check['intact'] and check['before'] == check['after']


def validate_retina(row, asleep):
    """Independently reconstruct the controller from all glance codelengths."""
    from statistics import median
    observations = row['input_retina']
    assert [r['t'] for r in observations] == list(range(row['L']))
    history, ema, S, active, forced, left = [], None, 0, False, False, 0
    for t, r in enumerate(observations):
        if left == 0:
            if forced:
                forced = S > 128
                active = forced
            elif S >= 256:
                active = forced = True
            elif active:
                active = S > 0 and ema > 1.5 * median(history)
            else:
                active = S >= 32 and ema is not None and ema > 1.5 * median(history)
            if active:
                left = min(8, row['L'] - t)
        sleeping = left > 0
        assert sleeping == (t in asleep) == bool(r['asleep']), 'eyelid schedule'
        x = r['r']
        assert math.isfinite(x) and x >= 0
        ema = x if ema is None else .9 * ema + .1 * x
        if sleeping:
            S = max(0, S - 3)
            left -= 1
        else:
            S += 1
            history = (history + [ema])[-256:]
        assert S == r['S'] and forced == bool(r['forced']), 'eyelid pressure'
        assert math.isclose(ema, r['r_bar'], rel_tol=0, abs_tol=1e-12), 'eyelid EMA'
        assert math.isclose(median(history), r['m_t'], rel_tol=0, abs_tol=1e-12), 'wake median'
    for check in row['eyelid_checks']:
        assert check['t'] in asleep and check['intact'] and check['before'] == check['after']


def cell(v):
    return 'NA' if v is None else f'{v:.9f}'


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)] + [
        '| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]


def summarize(out, pending=False):
    out = Path(out)
    smoke = json.loads((out / 'smoke.json').read_text())
    if not smoke['ok'] and not pending:
        raise RuntimeError('smoke did not pass')
    manifest_path = ROOT / 'kaggle/w9d_rep/manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    by, kernels = {}, []
    for job in manifest:
        path = out / job['folder'] / job['output']
        if not path.exists() and pending:
            continue
        payload = validate(job, path.parent)
        for row in payload['runs']:
            key = (row['seed'], row['arm'])
            if key in by:
                raise RuntimeError('duplicate life')
            by[key] = row
        kernels.append({k: payload[k] for k in ('seed', 'kernel_id', 'workers', 'wall_s')})
    expected = {(s, a) for s in SEEDS for a in ARMS}
    n_lives = len(expected)
    if not pending and set(by) != expected:
        raise RuntimeError(f'expected {n_lives} scored lives')
    for seed in SEEDS:
        logs = [row['swap_log'] for (s, _), row in by.items() if s == seed]
        if any(log != logs[0] for log in logs):
            raise RuntimeError('swap logs disagree')
    lines = []
    gap_rows, gaps = [], {}
    for left, right, threshold, required in GAPS:
        values = [by[s, left]['T14'] - by[s, right]['T14']
                  if (s, left) in by and (s, right) in by else None for s in SEEDS]
        have = [v for v in values if v is not None]
        mean = math.fsum(have) / len(have) if have else None
        # pressure_retina-fixed_night's bar applies to the mean, not a seed count.
        count = sum(v <= threshold for v in have) if threshold is not None and have else None
        gaps[f'{left}-{right}'] = dict(per_seed=dict(zip(SEEDS, values)), mean=mean,
            threshold=threshold, count=count, required_count=required, n=len(have))
        gap_rows.append([f'{left} - {right}', *map(cell, values), cell(mean),
                         cell(threshold), 'NA' if count is None else count,
                         'NA' if required is None else required, len(have)])
    lines += table(['T14 gap', *map(str, SEEDS), 'Mean', 'Threshold <=', 'Count <=', 'Required count', 'N'], gap_rows)
    mean_gap = gaps['pressure_retina-fixed_night']['mean']
    night_vals = [by[s, 'pressure_retina']['night_sleep_share']
                  for s in SEEDS if (s, 'pressure_retina') in by]
    night_mean = math.fsum(night_vals) / len(night_vals) if night_vals else None
    mean_bars = [
        dict(quantity='pressure_retina - fixed_night', mean=mean_gap, comparison='<=', threshold=.05,
             meets=None if mean_gap is None else int(mean_gap <= .05), n=gaps['pressure_retina-fixed_night']['n']),
        dict(quantity='pressure_retina night_sleep_share', mean=night_mean, comparison='>=', threshold=.60,
             meets=None if night_mean is None else int(night_mean >= .60), n=len(night_vals)),
    ]
    lines += ['', *table(['Mean threshold', 'Mean', 'Comparison', 'Threshold', 'Meets (0/1)', 'N'], [
        [r['quantity'], cell(r['mean']), r['comparison'], cell(r['threshold']),
         'NA' if r['meets'] is None else r['meets'], r['n']] for r in mean_bars]), '']
    arm_metrics = {}
    for field in ('night_sleep_share', 'sleep_share', 'sleep_decisions', 'night_sleep_decisions', 'T14'):
        arm_metrics[field] = {}
        lines += ['', '## ' + field, '']
        rows = []
        for arm in ARMS:
            vals = [by[s, arm][field] if (s, arm) in by else None for s in SEEDS]
            have = [v for v in vals if v is not None]
            arm_metrics[field][arm] = dict(per_seed=dict(zip(SEEDS, vals)),
                mean=math.fsum(have) / len(have) if have else None, n=len(have))
            rows.append([arm, *map(cell, vals), cell(math.fsum(have) / len(have) if have else None)])
        lines += table(['Arm', *map(str, SEEDS), 'Mean'], rows)
    lines += ['', *table(['Scored lives', 'Expected', 'Smoke workers', 'Smoke ok', 'Smoke wall s'], [
        [len(by), n_lives, smoke['workers'], int(smoke['ok']), cell(smoke['wall_s'])]])]
    if 'eyelid_checksum' in smoke:
        c = smoke['eyelid_checksum']
        lines += ['', *table(['Eyelid decisions', 'Matching checksums', 'Controller decisions', 'Controller matches (0/1)'], [[
            c['decisions'], c['intact'], smoke.get('controller_reconstruction', {}).get('decisions', 'NA'),
            int(smoke['controller_reconstruction']['intact']) if 'controller_reconstruction' in smoke else 'NA']]), '']
        lines += table(['Eyelid t', 'Before SHA256', 'After SHA256', 'Intact (0/1)'], [
            [c[k]['t'], c[k]['before'], c[k]['after'], int(c[k]['intact'])] for k in ('first', 'last')]) + ['']
    if 'per_day' in smoke:
        lines += ['', *table(['Day', 'Day sleep', 'Night sleep', 'Forced day', 'Forced night', 'Forced'], [
            [r[k] for k in ('day', 'day_sleep', 'night_sleep', 'forced_day', 'forced_night', 'forced')]
            for r in smoke['per_day']]), '']
    if 'gate' in smoke:
        g = smoke['gate']
        lines += table(['First day', 'Last day', 'Sleep', 'Night sleep', 'Night share', 'Threshold >=', 'Meets (0/1)'], [[
            *g['days'], g['sleep_decisions'], g['night_sleep_decisions'], cell(g['night_sleep_share']),
            cell(g['threshold']), int(g['meets'])]]) + ['']
    lines += ['', '## Smoke output', '', '```', *smoke['log'], '```', '']
    lines += ['## Kernel walls', '', *table(['Seed', 'Kernel', 'Workers', 'Wall s'], [
        [r['seed'], r['kernel_id'], r['workers'], cell(r['wall_s'])] for r in kernels]), '']
    if 'estimate' in smoke:
        lines += table(['Estimate s', 'Limit s', 'Within limit (0/1)'], [[
            cell(smoke['estimate']['kernel_wall_s']), smoke['estimate']['limit_s'],
            int(smoke['estimate']['within_limit'])]]) + ['']
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    missing = [dict(seed=s, arm=a) for s in SEEDS for a in ARMS if (s, a) not in by]
    (out / 'results.json').write_text(json.dumps(dict(gaps=gaps, mean_bars=mean_bars, arm_metrics=arm_metrics,
        runs=list(by.values()), kernels=kernels, smoke=smoke, missing=missing, pending=bool(missing)),
        indent=2, allow_nan=False) + '\n')
    print(f'lives={len(by)}/{n_lives} smoke={int(smoke["ok"])}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'results/w9d_rep')
    p.add_argument('--pending', action='store_true')
    a = p.parse_args()
    summarize(a.out, a.pending)
