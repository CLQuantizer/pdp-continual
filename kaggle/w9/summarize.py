"""W9 numeric tables. Refuses partial/malformed scored results; --pending shows smoke only."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(60, 72))
ARMS = ('nosleep', 'fixed_random', 'fixed_night', 'pressure', 'pressure_opp')
GAPS = (
    ('fixed_night', 'fixed_random', -.10, 9),
    ('pressure_opp', 'fixed_night', .03, None),
    ('pressure_opp', 'fixed_random', -.07, 8),
    ('pressure', 'fixed_random', None, None),
    ('fixed_random', 'nosleep', None, None),
)


def validate(job, dest):
    payload = json.loads((Path(dest) / job['output']).read_text())
    hashes = json.loads((ROOT / 'kaggle/w9/code-sha256.json').read_text())
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
    return payload


def cell(v):
    return 'NA' if v is None else f'{v:.9f}'


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)] + [
        '| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]


def summarize(out, pending=False):
    out = Path(out)
    smoke = json.loads((out / 'smoke.json').read_text())
    if not smoke['ok']:
        raise RuntimeError('smoke did not pass')
    manifest = json.loads((ROOT / 'kaggle/w9/manifest.json').read_text())
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
    if not pending and set(by) != expected:
        raise RuntimeError('expected 60 scored lives')
    for seed in SEEDS:
        logs = [row['swap_log'] for (s, _), row in by.items() if s == seed]
        if any(log != logs[0] for log in logs):
            raise RuntimeError('swap logs disagree')
    lines = ['# W9', '']
    gap_rows, gaps = [], {}
    for left, right, threshold, required in GAPS:
        values = [by[s, left]['T14'] - by[s, right]['T14']
                  if (s, left) in by and (s, right) in by else None for s in SEEDS]
        have = [v for v in values if v is not None]
        mean = sum(have) / len(have) if have else None
        # pressure_opp-fixed_night's bar applies to the mean, not a seed count.
        count = sum(v <= threshold for v in have) if threshold is not None and have else None
        gaps[f'{left}-{right}'] = dict(per_seed=dict(zip(SEEDS, values)), mean=mean,
            threshold=threshold, count=count, required_count=required, n=len(have))
        gap_rows.append([f'{left} - {right}', *map(cell, values), cell(mean),
                         cell(threshold), 'NA' if count is None else count,
                         'NA' if required is None else required, len(have)])
    lines += table(['T14 gap', *map(str, SEEDS), 'Mean', 'Threshold <=', 'Count <=', 'Required count', 'N'], gap_rows)
    mean_gap = gaps['pressure_opp-fixed_night']['mean']
    lines += ['', *table(['Mean threshold', 'Mean', 'Threshold <=', 'Meets (0/1)'], [
        ['pressure_opp - fixed_night', cell(mean_gap), '.030000000',
         'NA' if mean_gap is None else int(mean_gap <= .03)]]), '']
    lines += table(['Scored lives', 'Expected', 'Smoke workers', 'Smoke ok', 'Smoke wall s'], [
        [len(by), 60, smoke['workers'], int(smoke['ok']), cell(smoke['wall_s'])]])
    for field in ('T14', 'sleep_share', 'night_sleep_share'):
        lines += ['', '## ' + field, '']
        rows = []
        for arm in ARMS:
            vals = [by[s, arm][field] if (s, arm) in by else None for s in SEEDS]
            have = [v for v in vals if v is not None]
            rows.append([arm, *map(cell, vals), cell(sum(have) / len(have) if have else None)])
        lines += table(['Arm', *map(str, SEEDS), 'Mean'], rows)
    lines += ['', '## Smoke output', '', '```', *smoke['log'], '```', '']
    (out / 'RESULTS.md').write_text('\n'.join(lines))
    if not pending:
        (out / 'results.json').write_text(json.dumps(dict(runs=list(by.values()), kernels=kernels,
            gaps=gaps, smoke=smoke), indent=2, allow_nan=False) + '\n')
    print(f'lives={len(by)}/60 smoke={int(smoke["ok"])}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'results/w9')
    p.add_argument('--pending', action='store_true')
    a = p.parse_args()
    summarize(a.out, a.pending)
