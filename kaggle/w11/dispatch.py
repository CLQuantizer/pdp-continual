"""Foreground W11 scheduler.

At most one retry per kernel. The account's running kernels, including other
experiments, stay at or below 5; W11 stays at or below 3 concurrent kernels. Creates or versions the private code dataset,
then fetches outputs into results/w11/ and assembles the scoring table.

Run from the repository root with Kaggle credentials in the environment or in
.env. Does not print credentials.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'kaggle/w11'
OUT = ROOT / 'results/w11'
STATE = OUT / 'dispatch.json'
CACHE = OUT / 'kernel-status-cache.json'
CLI = ROOT / '.venv/bin/kaggle'
DATASET = 'ezzzzzx/pdp-code-w11'
MAX_RUNNING = 5
# CPU sessions end at 12 hours. lastRunTime is the latest session stamp, so an
# older kernel cannot still be occupying a slot.
WINDOW_S = 18 * 3600
ACTIVE_STATES = {'queued', 'pending', 'running', 'cancel_requested', 'cancel_acknowledged'}


def load_env():
    path = ROOT / '.env'
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[len('export '):]
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cli(*args, check=True, timeout=180):
    proc = subprocess.run(
        [str(CLI), *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )
    if check and proc.returncode:
        raise RuntimeError(f'Kaggle {args[0]} {args[1]} exit {proc.returncode}')
    return proc.returncode, proc.stdout


def save(state):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATE)


import summarize as summary
from summarize import add_filters, selection, filter_manifest, SEEDS, CORTICES
_scoring_validate = summary.validate


def validate(job, dest):
    """Use the scoring validator, allowing only prequential summation roundoff.

    Life adds chunk sums sequentially; Python 3.12's sum of the glance log
    uses improved float summation. They can differ by a few ulps. Validate a temporary view
    with that one total and its derived ratio normalized, preserving the fetched
    artifact and every other scoring check. Return the original measurements.
    """
    dest = Path(dest)
    if job['part'] != job['parts']:
        return _scoring_validate(job, dest)
    payload = json.loads((dest / job['output']).read_text())
    row = payload['runs'][0]
    total = sum(g['bits'] for g in row['glance_log'] if g['preq'])
    if row['preq_bits'] == total:
        return _scoring_validate(job, dest)
    assert math.isclose(row['preq_bits'], total, rel_tol=1e-14, abs_tol=1e-9), 'preq_bits mismatch'
    assert row['preq_bps'] == row['preq_bits'] / row['preq_symbols'], 'preq_bps mismatch'
    original_bits, original_bps = row['preq_bits'], row['preq_bps']
    row['preq_bits'] = total
    row['preq_bps'] = total / row['preq_symbols']
    with tempfile.TemporaryDirectory(prefix='w11-validate-') as tmp:
        (Path(tmp) / job['output']).write_text(json.dumps(payload, allow_nan=False))
        _scoring_validate(job, tmp)
    row['preq_bits'], row['preq_bps'] = original_bits, original_bps
    return payload


VALIDATION_ERRORS = (OSError, ValueError, KeyError, AssertionError, TypeError, IndexError, ZeroDivisionError)


def reconcile_outputs(jobs):
    """Recover selected fetched outputs before considering any retry or push."""
    for job in jobs:
        dest = OUT / job['folder']
        if job['status'] != 'complete' and not (dest / job['output']).is_file():
            continue
        try:
            validate(job, dest)
        except VALIDATION_ERRORS as exc:
            print(job['kernel_id'], 'local validation failed:', type(exc).__name__, str(exc), flush=True)
            if job['status'] == 'complete':
                job['status'] = 'active'  # refetch completed remote output
        else:
            if job['status'] != 'complete':
                print(job['kernel_id'], 'complete (validated fetched output)', flush=True)
            job['status'] = 'complete'


def assemble(seeds, cortices):
    # Keep dispatch --assemble-only and final assembly on the same validator.
    original = summary.validate
    try:
        summary.validate = validate
        return summary.summarize(OUT, seeds=seeds, cortices=cortices)
    finally:
        summary.validate = original


def load_cache():
    if not CACHE.is_file():
        return {}
    try:
        data = json.loads(CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(cache):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix('.tmp')
    tmp.write_text(json.dumps(cache, indent=2) + '\n')
    tmp.replace(CACHE)


def within_window(stamp, now):
    if stamp is None or stamp == '':
        return True
    if isinstance(stamp, datetime):
        dt = stamp
    else:
        try:
            dt = datetime.fromisoformat(str(stamp))
        except ValueError:
            return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt).total_seconds() <= WINDOW_S


def stamp_key(stamp):
    if stamp is None:
        return None
    if isinstance(stamp, datetime):
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.isoformat()
    return str(stamp)


def redact(text):
    token = os.environ.get('KAGGLE_API_TOKEN', '')
    if token and token in text:
        text = text.replace(token, '[redacted]')
    return text


_API = None


def kaggle_api():
    global _API
    if _API is None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        _API = api
    return _API


def listed_kernels():
    for _ in range(3):
        try:
            return _listed_kernels()
        except Exception:
            time.sleep(5)
    raise RuntimeError('Kaggle kernels list failed')


def _listed_kernels():
    api = kaggle_api()
    now = datetime.now(timezone.utc)
    rows = []
    for page in range(1, 6):
        batch = api.kernels_list(mine=True, page=page, page_size=100, sort_by='dateRun') or []
        if not batch:
            break
        oldest = None
        for row in batch:
            ref = getattr(row, 'ref', None)
            stamp = getattr(row, 'last_run_time', None)
            if ref:
                rows.append((ref, stamp))
                oldest = stamp
        if oldest is not None and not within_window(oldest, now):
            break
    if not rows:
        raise RuntimeError('kernel list returned no rows')
    return rows


def other_active_ids():
    found = []
    results = ROOT / 'results'
    if not results.is_dir():
        return found
    for path in results.glob('*/dispatch.json'):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        jobs = data.get('jobs') if isinstance(data, dict) else None
        if not isinstance(jobs, list):
            continue
        for job in jobs:
            if job.get('status') not in ('active', 'submitting'):
                continue
            kid = job.get('kernel_id') or job.get('id')
            if isinstance(kid, str):
                found.append(kid)
    return found


def status_name(ref):
    """KernelWorkerStatus name, or None when this account cannot read the kernel."""
    from requests.exceptions import HTTPError
    for attempt in range(4):
        try:
            response = kaggle_api().kernels_status(ref)
            return response.status.name.lower()
        except HTTPError as exc:
            response = getattr(exc, 'response', None)
            code = getattr(response, 'status_code', None)
            if code in (401, 403, 404):
                return None
            time.sleep(5 * (attempt + 1))
        except Exception:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError('Refusing to push while a kernel status is unknown: %s' % ref)


def running_ids():
    """Refs that still occupy a slot: running, queued, or cancelling.

    `kaggle kernels list` has no status column. Sessions inside the CPU window,
    plus kernels another local dispatcher has marked active, are checked with
    the status API. Terminal results are cached by lastRunTime.
    """
    cache = load_cache()
    now = datetime.now(timezone.utc)
    rows = listed_kernels()
    watch = {ref for ref in (cache.get('_watch') or []) if isinstance(ref, str)}
    candidates = []
    chosen = set()
    for ref, stamp in rows:
        key = stamp_key(stamp)
        if not within_window(stamp, now) and ref not in watch:
            continue
        cached = cache.get(ref)
        if (
            isinstance(cached, dict)
            and cached.get('lastRunTime') == key
            and cached.get('kind') in ('complete', 'failed', 'idle')
            and key is not None
            and ref not in watch
        ):
            continue
        candidates.append((ref, key))
        chosen.add(ref)
    for ref in list(other_active_ids()) + list(watch):
        if ref not in chosen:
            candidates.append((ref, None))
            chosen.add(ref)
    running = set()
    new_watch = []
    for ref, key in candidates:
        name = status_name(ref)
        if name is None:
            raise RuntimeError('Refusing to submit: unreadable account kernel ' + ref)
        if name in ACTIVE_STATES:
            kind = 'active'
        elif name == 'complete':
            kind = 'complete'
        elif name == 'error':
            kind = 'failed'
        elif name == 'new_script':
            kind = 'idle'
        else:
            raise RuntimeError('Unknown remote status; refusing to submit more work: %s %s' % (ref, name))
        if key is not None and kind in ('complete', 'failed', 'idle'):
            cache[ref] = {'kind': kind, 'lastRunTime': key}
        if kind == 'active':
            running.add(ref)
            new_watch.append(ref)
    cache['_watch'] = new_watch
    save_cache(cache)
    print('account slot check: %s statuses, %s running' % (len(candidates), len(running)), flush=True)
    return running


def check_layout(manifest, hashes, seeds=SEEDS, cortices=CORTICES):
    expected = {(s,c) for s in seeds for c in cortices}
    assert {(j['seed'],j['cortex']) for j in manifest} == expected
    smoke = json.loads((OUT / 'smoke.json').read_text())
    assert smoke['source_hashes'] == hashes
    ids = {j['kernel_id'] for j in manifest}
    assert len(ids) == len(manifest)
    for seed, cortex in expected:
        jobs = [j for j in manifest if (j['seed'], j['cortex']) == (seed, cortex)]
        parts = smoke['estimate']['parts'][cortex]
        assert [j['part'] for j in jobs] == list(range(1, parts+1))
        prev = None
        for j in jobs:
            slug = f"w11-{cortex.replace('_','-')}-s{seed}" + (f"-p{j['part']}" if parts>1 else '')
            assert j['kernel_id'] == 'ezzzzzx/' + slug and j['depends_on'] == prev
            assert j['parts'] == parts and j['arms'] == ['pressure_retina']
            prev = j['kernel_id']
    expected_files = set(hashes) | {'dataset-metadata.json'}
    if {p.name for p in (BASE / 'code').iterdir()} != expected_files:
        raise RuntimeError('unexpected file in code dataset directory')
    dataset_meta = json.loads((BASE / 'code' / 'dataset-metadata.json').read_text())
    if dataset_meta['id'] != DATASET or any(not n.endswith('.py') for n in hashes):
        raise RuntimeError('code dataset whitelist')
    for name, digest in hashes.items():
        if hashlib.sha256((BASE / 'code' / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'staged hash mismatch for {name}')
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'repo hash mismatch for {name}')
    for job in manifest:
        meta = json.loads((BASE / job['folder'] / 'kernel-metadata.json').read_text())
        if meta['id'] != job['kernel_id'] or meta['is_private'] is not True:
            raise RuntimeError(f'kernel metadata for {job["kernel_id"]}')
        if meta['enable_gpu'] is not False or meta['enable_tpu'] is not False or meta['enable_internet'] is not False:
            raise RuntimeError(f'kernel resources for {job["kernel_id"]}')
        assert meta['kernel_sources'] == ([job['depends_on']] if job['depends_on'] else [])
        if meta['dataset_sources'] != [DATASET]:
            raise RuntimeError(f'kernel dataset for {job["kernel_id"]}')


def dataset_kind(report, code):
    """Classify `kaggle datasets status` without treating a 403 body as failure.

    A private dataset that does not exist yet, or is not visible yet, returns
    403 Client Error. That substring contains "error" and is not a processing failure.
    """
    lower = report.lower()
    if code == 0 and 'ready' in lower:
        return 'ready'
    if any(token in lower for token in ('403', '404', 'forbidden', "don't have access", 'not found')):
        return 'missing'
    if code == 0 and any(token in lower for token in ('pending', 'processing')):
        return 'processing'
    if any(token in lower for token in ('error', 'fail')):
        return 'failed'
    return 'processing'


def dataset_call(op, *args):
    code, report = cli('datasets', op, *args, check=False, timeout=300)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'dataset-{op}.log').write_text(redact(report))
    return code


def dataset_ready(state, dataset_ready_flag):
    if dataset_ready_flag:
        state['dataset_created'] = True
        save(state)
        return
    if not state['dataset_created']:
        code, report = cli('datasets', 'status', DATASET, check=False)
        kind = dataset_kind(report, code)
        if kind == 'failed':
            raise RuntimeError('Code dataset processing failed')
        if kind == 'missing':
            created = dataset_call('create', '-p', str(BASE / 'code'))
            if created:
                if dataset_call('version', '-p', str(BASE / 'code'), '-m', 'w11 deep cortex code'):
                    raise RuntimeError('Kaggle datasets create/version failed')
                print('code dataset version', flush=True)
            else:
                print('code dataset create', flush=True)
        else:
            if dataset_call('version', '-p', str(BASE / 'code'), '-m', 'w11 deep cortex code'):
                raise RuntimeError('Kaggle datasets version failed')
            print('code dataset version', flush=True)
        state['dataset_created'] = True
        save(state)
    while True:
        code, report = cli('datasets', 'status', DATASET, check=False)
        kind = dataset_kind(report, code)
        if kind == 'ready':
            return
        if kind == 'failed':
            raise RuntimeError('Code dataset processing failed')
        print('code dataset processing', flush=True)
        time.sleep(30)


def offline_test():
    """Regression against fetched final parts; no credentials or network calls."""
    import copy
    import shutil
    import unittest
    from contextlib import ExitStack
    from unittest.mock import patch

    module = sys.modules[__name__]
    fixtures = [j for j in json.loads((BASE / 'manifest.json').read_text())
                if j['seed'] in (132, 133) and j['cortex'] == 'bank_1' and j['part'] == 2]

    class Regression(unittest.TestCase):
        def test_final_roundoff_and_corruption(self):
            for job in fixtures:
                dest = OUT / job['folder']
                raw = (dest / job['output']).read_bytes()
                with self.assertRaises(AssertionError):
                    _scoring_validate(job, dest)  # reproduce the original failure
                payload = validate(job, dest)
                self.assertEqual((dest / job['output']).read_bytes(), raw)
                self.assertEqual(payload, json.loads(raw))
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / job['output']
                    # A material total error, a wrong ratio, hash, identity or
                    # incomplete final must still fail the full validator.
                    for field, value in [('preq_bits', payload['runs'][0]['preq_bits'] + .001),
                                         ('preq_bps', 0), ('complete', False), ('finite', False)]:
                        bad = copy.deepcopy(payload)
                        bad['runs'][0][field] = value
                        path.write_text(json.dumps(bad))
                        with self.assertRaises(AssertionError):
                            validate(job, tmp)
                    for field, value in [('code_sha256', {}), ('seed', -1), ('part', 1)]:
                        bad = copy.deepcopy(payload)
                        bad[field] = value
                        path.write_text(json.dumps(bad))
                        with self.assertRaises(AssertionError):
                            validate(job, tmp)

        def test_resume_without_repush(self):
            source_out = OUT
            state = json.loads(STATE.read_text())
            for job in state['jobs']:
                if job['kernel_id'] in {j['kernel_id'] for j in fixtures}:
                    job.update(status='failed', attempts=2)
            before = copy.deepcopy(state)
            class NextPush(BaseException):
                pass
            pushed = []
            def fake_cli(*args, **kwargs):
                if args[:2] == ('kernels', 'status'):
                    return 0, 'has status "running"'
                if args[:2] == ('kernels', 'push'):
                    pushed.append(Path(args[3]).name)
                    raise NextPush
                self.fail('unexpected CLI call: ' + repr(args[:2]))
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                out = Path(tmp)
                (out / 'dispatch.json').write_text(json.dumps(state))
                shutil.copyfile(source_out / 'smoke.json', out / 'smoke.json')
                for job in state['jobs']:
                    dest = source_out / job['folder']
                    if dest.is_dir():
                        (out / job['folder']).symlink_to(dest, target_is_directory=True)
                for name, value in [('OUT', out), ('STATE', out / 'dispatch.json'),
                                    ('load_env', lambda: None), ('dataset_ready', lambda *a: None),
                                    ('running_ids', lambda: set()), ('cli', fake_cli)]:
                    stack.enter_context(patch.object(module, name, value))
                stack.enter_context(patch.object(sys, 'argv', [__file__, '--seeds',
                    '132', '133', '134', '135', '136', '137', '--cortices', 'bank_1', 'deep_4']))
                with self.assertRaises(NextPush):
                    main()
                resumed = json.loads((out / 'dispatch.json').read_text())
                for old, new in zip(before['jobs'], resumed['jobs']):
                    if old['kernel_id'] in {j['kernel_id'] for j in fixtures}:
                        self.assertEqual(new['status'], 'complete')
                        self.assertEqual(new['attempts'], 2)
                    if old['seed'] not in range(132, 138) or old['cortex'] not in ('bank_1', 'deep_4'):
                        self.assertEqual(new, old)
                self.assertTrue(pushed)
                self.assertNotIn(pushed[0], {j['folder'] for j in fixtures})
                self.assertIn(pushed[0], {j['folder'] for j in before['jobs']
                    if j['status'] == 'pending' and j['seed'] in range(132, 138)
                    and j['cortex'] in ('bank_1', 'deep_4')})
                # Assembly must use the same checks and preserve measured totals.
                result = assemble((132, 133), ('bank_1',))
                self.assertEqual(result['scored_lives'], 2)
                for row in result['runs']:
                    original = json.loads((source_out / f"bank-1-s{row['seed']}-p2" /
                                           f"bank_1-s{row['seed']}-p2.json").read_text())['runs'][0]
                    self.assertEqual(row, original)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Regression))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--assemble-only', action='store_true')
    parser.add_argument('--dataset-ready', action='store_true')
    add_filters(parser)
    args = parser.parse_args()
    seeds, cortices = selection(args.seeds, args.cortices)
    smoke = json.loads((OUT / 'smoke.json').read_text())
    if not smoke.get('ok'):
        raise RuntimeError('smoke did not pass')
    all_manifest = json.loads((BASE / 'manifest.json').read_text())
    manifest = filter_manifest(all_manifest, seeds, cortices)
    hashes = json.loads((BASE / 'code-sha256.json').read_text())
    check_layout(manifest, hashes, seeds, cortices)
    if args.check_only:
        print(f'checked {len(manifest)} kernel folders and {len(hashes)} code hashes', flush=True)
        return
    if args.assemble_only:
        assemble(seeds, cortices)
        return
    load_env()
    if STATE.exists():
        state = json.loads(STATE.read_text())
    else:
        state = dict(dataset_created=False, code_sha256=hashes, jobs=[dict(**job, status='pending', attempts=0) for job in all_manifest])
    if state.get('code_sha256') != hashes:
        raise RuntimeError('dispatch state code hashes changed; refusing to mix code versions')
    if [(job['kernel_id'], job['arms']) for job in state['jobs']] != [(job['kernel_id'], job['arms']) for job in all_manifest]:
        raise RuntimeError('dispatch state does not match the manifest')
    jobs = filter_manifest(state['jobs'], seeds, cortices)
    reconcile_outputs(jobs)
    save(state)
    dataset_ready(state, args.dataset_ready)
    while True:
        for job in jobs:
            if job['status'] not in ('active', 'submitting'):
                continue
            code, report = cli('kernels', 'status', job['kernel_id'], check=False)
            status = report.lower()
            if code:
                if '404' in status or 'not found' in status:
                    job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
                    print(job['kernel_id'], 'not found', job['status'], flush=True)
                else:
                    print(job['kernel_id'], 'status unavailable', flush=True)
                save(state)
                continue
            line = status.split('has status', 1)[-1].splitlines()[0] if 'has status' in status else status
            if 'complete' in line and 'running' not in line and 'pending' not in line:
                dest = OUT / job['folder']
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    cli('kernels', 'output', job['kernel_id'], '-p', str(dest), '--force', timeout=300)
                    validate(job, dest)
                except (RuntimeError, *VALIDATION_ERRORS) as exc:
                    print(job['kernel_id'], 'fetch/validation failed:', type(exc).__name__, str(exc), flush=True)
                    job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
                else:
                    job['status'] = 'complete'
                    print(job['kernel_id'], 'complete', flush=True)
            elif any(word in line for word in ('error', 'cancel', 'fail', 'new_script', 'newscript')):
                dest = OUT / 'failed' / f"{job['folder']}-attempt{job['attempts']}"
                dest.mkdir(parents=True, exist_ok=True)
                cli('kernels', 'output', job['kernel_id'], '-p', str(dest), check=False)
                job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
                print(job['kernel_id'], job['status'], flush=True)
            else:
                print(job['kernel_id'], 'active', flush=True)
            save(state)
        if any(job['status'] == 'failed' for job in jobs):
            raise RuntimeError('Kernel exhausted its single retry; state saved')
        active_ids = {job['kernel_id'] for job in jobs if job['status'] in ('active', 'submitting')}
        listed = running_ids()
        manifest_ids = {j['kernel_id'] for j in all_manifest}
        active_ids |= listed & manifest_ids
        occupied = len(listed | active_ids)
        if any(job['status'] == 'pending' for job in jobs) and occupied >= MAX_RUNNING:
            print('account running %s; holding pushes (cap %s)' % (occupied, MAX_RUNNING), flush=True)
        for job in jobs:
            if occupied >= MAX_RUNNING or len(active_ids) >= 3:
                break
            if job['status'] != 'pending' or job['kernel_id'] in active_ids:
                continue
            if job['depends_on'] and not any(j['kernel_id'] == job['depends_on'] and j['status'] == 'complete' for j in jobs):
                continue
            # Recheck immediately before each push, including external dispatchers.
            current = running_ids() | active_ids
            if len(current) >= MAX_RUNNING or len(current & manifest_ids) >= 3:
                break
            job.update(status='submitting', attempts=job['attempts'] + 1)
            save(state)
            try:
                cli('kernels', 'push', '-p', str(BASE / job['folder']), timeout=300)
            except (RuntimeError, subprocess.TimeoutExpired):
                print(job['kernel_id'], 'push outcome uncertain; will reconcile', flush=True)
            job['status'] = 'active'
            save(state)
            occupied += 1
            active_ids.add(job['kernel_id'])
            print(job['kernel_id'], f"attempt {job['attempts']}", 'running', occupied, flush=True)
        if all(job['status'] == 'complete' for job in jobs):
            break
        time.sleep(30)
    assemble(seeds, cortices)


if __name__ == '__main__':
    if sys.argv[1:] == ['--offline-test']:
        offline_test()
        raise SystemExit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'dispatch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
