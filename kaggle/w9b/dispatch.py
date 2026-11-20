"""Foreground W9b scheduler.

At most one retry per kernel. The account's running kernels, including other
experiments, stay at or below 5; W9b stays at or below 3 concurrent kernels. Creates or versions the private code dataset,
then fetches outputs into results/w9b/ and assembles the scoring table.

Run from the repository root with Kaggle credentials in the environment or in
.env. Does not print credentials.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'kaggle/w9b'
OUT = ROOT / 'results/w9b'
STATE = OUT / 'dispatch.json'
CACHE = OUT / 'kernel-status-cache.json'
CLI = ROOT / '.venv/bin/kaggle'
DATASET = 'ezzzzzx/pdp-code-w9b'
MAX_RUNNING = 5
# CPU sessions end at 12 hours. lastRunTime is the latest session stamp, so an
# older kernel cannot still be occupying a slot.
WINDOW_S = 18 * 3600
ACTIVE_STATES = {'queued', 'running', 'cancel_requested', 'cancel_acknowledged'}


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


from summarize import validate


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
        if path.parent == OUT:
            continue
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


def check_layout(manifest, hashes):
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
                if dataset_call('version', '-p', str(BASE / 'code'), '-m', 'w9b day night code'):
                    raise RuntimeError('Kaggle datasets create/version failed')
                print('code dataset version', flush=True)
            else:
                print('code dataset create', flush=True)
        else:
            if dataset_call('version', '-p', str(BASE / 'code'), '-m', 'w9b day night code'):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--assemble-only', action='store_true')
    parser.add_argument('--dataset-ready', action='store_true')
    args = parser.parse_args()
    smoke = json.loads((OUT / 'smoke.json').read_text())
    if not smoke.get('ok'):
        raise RuntimeError('smoke did not pass')
    manifest = json.loads((BASE / 'manifest.json').read_text())
    hashes = json.loads((BASE / 'code-sha256.json').read_text())
    check_layout(manifest, hashes)
    if args.check_only:
        print(f'checked {len(manifest)} kernel folders and {len(hashes)} code hashes', flush=True)
        return
    if args.assemble_only:
        subprocess.run([sys.executable, str(BASE / 'summarize.py')], cwd=ROOT, check=True)
        return
    load_env()
    if STATE.exists():
        state = json.loads(STATE.read_text())
    else:
        state = dict(dataset_created=False, jobs=[dict(**job, status='pending', attempts=0) for job in manifest])
    if [(job['kernel_id'], job['arms']) for job in state['jobs']] != [(job['kernel_id'], job['arms']) for job in manifest]:
        raise RuntimeError('dispatch state does not match the manifest')
    for job in state['jobs']:
        if job['status'] == 'complete':
            try:
                validate(job, OUT / job['folder'])
            except (OSError, ValueError, KeyError, AssertionError):
                job['status'] = 'active'  # refetch completed remote output
    save(state)
    dataset_ready(state, args.dataset_ready)
    while True:
        for job in state['jobs']:
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
                except (RuntimeError, AssertionError, OSError, ValueError, KeyError, TypeError):
                    job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
                else:
                    job['status'] = 'complete'
                    print(job['kernel_id'], 'complete', flush=True)
            elif any(word in line for word in ('error', 'cancel', 'fail')):
                dest = OUT / 'failed' / f"{job['folder']}-attempt{job['attempts']}"
                dest.mkdir(parents=True, exist_ok=True)
                cli('kernels', 'output', job['kernel_id'], '-p', str(dest), check=False)
                job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
                print(job['kernel_id'], job['status'], flush=True)
            else:
                print(job['kernel_id'], 'active', flush=True)
            save(state)
        if any(job['status'] == 'failed' for job in state['jobs']):
            raise RuntimeError('Kernel exhausted its single retry; state saved')
        active_ids = {job['kernel_id'] for job in state['jobs'] if job['status'] in ('active', 'submitting')}
        listed = running_ids()
        manifest_ids = {j['kernel_id'] for j in state['jobs']}
        active_ids |= listed & manifest_ids
        occupied = len(listed | active_ids)
        if any(job['status'] == 'pending' for job in state['jobs']) and occupied >= MAX_RUNNING:
            print('account running %s; holding pushes (cap %s)' % (occupied, MAX_RUNNING), flush=True)
        for job in state['jobs']:
            if occupied >= MAX_RUNNING or len(active_ids) >= 3:
                break
            if job['status'] != 'pending' or job['kernel_id'] in active_ids:
                continue
            job.update(status='submitting', attempts=job['attempts'] + 1)
            save(state)
            try:
                cli('kernels', 'push', '-p', str(BASE / job['folder']), timeout=300)
            except RuntimeError:
                print(job['kernel_id'], 'push failed; will reconcile', flush=True)
            job['status'] = 'active'
            save(state)
            occupied += 1
            active_ids.add(job['kernel_id'])
            print(job['kernel_id'], f"attempt {job['attempts']}", 'running', occupied, flush=True)
        if all(job['status'] == 'complete' for job in state['jobs']):
            break
        time.sleep(30)
    subprocess.run([sys.executable, str(BASE / 'summarize.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'dispatch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
