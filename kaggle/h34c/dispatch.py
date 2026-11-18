"""Resumable H34c dispatcher. Account-wide running kernels stay <= 5.

One retry. A process lock serializes H34c dispatchers. Other experiments share
the account cap; the API has no atomic reservation, so each push rechecks.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h34c_grow import ARMS, N_BYTES, dump, source_hashes
BASE, OUT = ROOT / 'kaggle/h34c', ROOT / 'results/h34c'
STATE = OUT / 'dispatch.json'
CLI = str(ROOT / '.venv/bin/kaggle')


class RateLimited(RuntimeError):
    """Kaggle returned HTTP 429. The dispatcher waits and retries; it does not exit."""
FILES = ('h34c_grow.py', 'h34b_plain_depth.py', 'h34a2_logit_sum.py', 'h34a_local_depth.py',
         'h26_cued.py', 'h32_mixer.py', 'h32b_mixer.py', 'h33_bank_text.py', 'h20_world.py',
         'h30w_ring.py', 'h30x_trace.py', 'h30y_roots.py')


def load_token():
    if os.environ.get('KAGGLE_API_TOKEN'):
        return
    for raw in (ROOT / '.env').read_text().splitlines():
        line = raw.strip()
        if line.startswith('export '):
            line = line[7:].strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        if key.strip() != 'KAGGLE_API_TOKEN':
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        if not value:
            raise RuntimeError('KAGGLE_API_TOKEN is empty')
        os.environ['KAGGLE_API_TOKEN'] = value
        return
    raise RuntimeError('KAGGLE_API_TOKEN is unset')


def cli(*args, allow_failure=False, timeout=300, retries=6):
    last = ''
    for attempt in range(retries):
        proc = subprocess.run([CLI, *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout)
        text = proc.stdout.replace(os.environ.get('KAGGLE_API_TOKEN') or '\0', '[REDACTED]')
        last = text
        if proc.returncode and not allow_failure:
            if '429' in text or 'Too Many' in text:
                if attempt == 0:
                    print(f'kaggle 429 {args[0]} {args[1]} backoff 90s', flush=True)
                    time.sleep(90)
                    continue
                raise RateLimited(f'Kaggle {args[0]} {args[1]} rate limited')
            raise RuntimeError(f'Kaggle {args[:2]} failed: {text}')
        return proc.returncode, text
    raise RuntimeError(f'Kaggle {args[:2]} failed: {last}')


def prepare():
    smoke_path = OUT / 'smoke.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    if smoke.get('n_bytes') not in (None, N_BYTES):
        raise RuntimeError(f"smoke n_bytes {smoke.get('n_bytes')} != {N_BYTES}")
    if smoke.get('ok') and smoke.get('source_hashes') != source_hashes():
        raise RuntimeError('Smoke does not match current source')
    code = BASE / 'code'
    code.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES if (ROOT / name).is_file()}
    manifest = dict(hashes=hashes, missing=missing, documents=N_BYTES)
    manifest_path = BASE / 'code-sha256.json'
    if STATE.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Source changed after dispatch began; refusing mixed versions')
    for name in hashes:
        shutil.copyfile(ROOT / name, code / name)
    dump(code / 'dataset-metadata.json', dict(id='ezzzzzx/pdp-code-h34c', title='pdp code h34c', licenses=[dict(name='other')]))
    dump(manifest_path, manifest)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    for arm in ARMS:
        for seed in (0, 1):
            tag = f'{arm}-s{seed}'
            folder = BASE / tag
            folder.mkdir(exist_ok=True)
            slug = f'h34c-{arm.replace("_", "-")}-s{seed}'
            dump(folder / 'kernel-metadata.json', dict(
                id=f'ezzzzzx/{slug}', title=slug, code_file='kernel.py', language='python',
                kernel_type='script', is_private=True, enable_gpu=False, enable_tpu=False,
                enable_internet=False, dataset_sources=['ezzzzzx/enwik8', 'ezzzzzx/pdp-code-h34c'],
                competition_sources=[], kernel_sources=[], model_sources=[]))
            (folder / 'kernel.py').write_text(f'''"""H34c private CPU, {arm}, seed {seed}. Growing plain stack."""
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
code = sorted(Path('/kaggle/input').rglob('h34c_grow.py'))
assert len(code) == 1, code
sys.path.insert(0, str(code[0].parent))
import h34c_grow as h
h.h33.configure(1)
assert h.N_BYTES == {N_BYTES}
assert h.source_hashes() == {hashes!r}, "Attached code differs from smoke"
raw_paths = [p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size == 100_000_000]
assert len(raw_paths) == 1, raw_paths
raw, digest = h.h33.load_checked(str(raw_paths[0]))
r = h.run({arm!r}, {seed}, raw[:h.N_BYTES])
assert r['n_bytes'] == h.N_BYTES
r.update(kernel_id='ezzzzzx/{slug}', sha256_e10m=digest, code_fingerprint={fingerprint!r})
h.dump('/kaggle/working/results.json', r)
''')
    print(json.dumps(dict(prepared=len(ARMS) * 2, documents=N_BYTES, missing=missing, code_fingerprint=fingerprint)), flush=True)
    return manifest, fingerprint


def status_kind(text):
    status = text.lower().split('has status', 1)[-1].strip().splitlines()[0]
    if 'complete' in status:
        return 'complete'
    if any(s in status for s in ('error', 'cancel', 'fail')):
        return 'failed'
    if any(s in status for s in ('running', 'queued', 'pending')):
        return 'active'
    raise RuntimeError('Unknown kernel status; refusing submission: ' + text)


# A rerun of an old kernel may keep its previous lastRunTime until it finishes, so idle
# cache hits are rechecked on a full sweep. New kernels and anything still active are
# always rechecked. Status calls are serial: a thread pool gets HTTP 429.
ACTIVE_STATES = {'QUEUED', 'RUNNING', 'CANCEL_REQUESTED', 'CANCEL_ACKNOWLEDGED', 'PENDING'}
IDLE_STATES = {'COMPLETE', 'ERROR', 'NEW_SCRIPT', 'NOT_FOUND'}


def _kaggle_api():
    load_token()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _list_kernels(api):
    rows = []
    for page in range(1, 50):
        batch = api.kernels_list(mine=True, page=page, page_size=100, sort_by='dateRun') or []
        rows.extend(batch)
        if len(batch) < 100:
            break
    if not rows:
        raise RuntimeError('Account kernel list is empty')
    return rows


def _status_name(api, ref):
    for attempt in range(2):
        try:
            response = api.kernels_status(ref)
        except Exception as exc:
            text = str(exc)
            if '429' in text or 'Too Many' in text:
                if attempt == 0:
                    print(f'account status 429 {ref} backoff 180s', flush=True)
                    time.sleep(180)
                    continue
                raise RateLimited(ref)
            # A listed kernel with no session is not a running worker.
            if '404' in text or 'Not Found' in text:
                return 'NOT_FOUND'
            low = text.lower()
            if any(s in low for s in ('403', 'forbidden', "don't have access", 'denied')):
                raise RuntimeError('Cannot establish account slots: unreadable ' + ref)
            raise RuntimeError('Refusing to push while a kernel status is unknown: ' + ref)
        status = response.status
        name = status.name if hasattr(status, 'name') else str(status).split('.')[-1]
        return name.upper()
    raise RateLimited(ref)


def account_active():
    api = _kaggle_api()
    kernels = _list_kernels(api)
    cache_path = OUT / 'status-cache.json'
    try:
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    except json.JSONDecodeError:
        cache = {}
    now = time.time()
    # A full sweep of every idle kernel is every 2 hours. Between sweeps, only the
    # newest 40 and anything still marked active are statused. lastRunTime can stay
    # stale across a rerun, so the periodic sweep is what catches that case.
    full = now - float(cache.get('_full_at') or 0) > 7200
    active, checked = [], 0
    newest = {k.ref for k in kernels[:40] if k is not None and k.ref}
    for k in kernels:
        if k is None or not k.ref:
            print('account status skip row with no ref', flush=True)
            continue
        ref = k.ref
        last = k.last_run_time.isoformat() if hasattr(k.last_run_time, 'isoformat') else str(k.last_run_time)
        cached = cache.get(ref)
        recent = isinstance(cached, dict) and now - float(cached.get('at') or 0) < 7200
        if (isinstance(cached, dict) and cached.get('last') == last and cached.get('kind') == 'idle'
                and ref not in newest and (recent or not full)):
            continue
        try:
            name = _status_name(api, ref)
        except RateLimited:
            dump(cache_path, cache)
            print(f'account slot check paused after {checked} status calls', flush=True)
            return None
        time.sleep(1.2)
        checked += 1
        if checked == 1 or checked % 25 == 0:
            print(f'account status {checked} latest={ref} {name}', flush=True)
        if name in ACTIVE_STATES:
            cache[ref] = dict(last=last, kind='active', name=name, at=now)
            active.append(ref)
        elif name in IDLE_STATES:
            cache[ref] = dict(last=last, kind='idle', name=name, at=now)
        else:
            raise RuntimeError('Unknown kernel status; refusing submission: ' + ref + ' ' + name)
        if checked % 25 == 0:
            dump(cache_path, cache)
    if full or '_full_at' not in cache:
        cache['_full_at'] = now
    dump(cache_path, {key: value for key, value in cache.items()})
    print(f'account slot check: {len(kernels)} listed, {checked} status calls, {len(active)} active', flush=True)
    dump(OUT / 'account-active.json', dict(at=now, active=active, total=len(active), listed=len(kernels), checked=checked))
    return active


def fetch(job, fingerprint):
    dest = OUT / 'kernels' / job['tag']
    dest.mkdir(parents=True, exist_ok=True)
    cli('kernels', 'output', job['id'], '-p', str(dest))
    row = json.loads((dest / 'results.json').read_text())
    if row['kernel_id'] != job['id'] or row['code_fingerprint'] != fingerprint or row['n_bytes'] != N_BYTES:
        raise RuntimeError('Fetched output does not match this dispatch: ' + job['id'])
    if row['sha256_e10m'] != json.loads((OUT / 'smoke.json').read_text())['sha256_e10m']:
        raise RuntimeError('Fetched enwik8 hash mismatch: ' + job['id'])


def dispatch(manifest, fingerprint):
    smoke = json.loads((OUT / 'smoke.json').read_text())
    if not smoke['ok']:
        raise RuntimeError('Smoke gate failed; no kernels may launch')
    if manifest['missing']:
        raise RuntimeError('Missing source files: ' + ', '.join(manifest['missing']))
    if smoke['n_bytes'] != N_BYTES or manifest['documents'] != N_BYTES:
        raise RuntimeError('Document count is not 180000')
    if smoke['source_hashes'] != manifest['hashes']:
        raise RuntimeError('Smoke hashes differ from the dataset manifest')
    load_token()
    jobs = [dict(tag=f'{arm}-s{seed}', id=f'ezzzzzx/h34c-{arm.replace("_", "-")}-s{seed}', status='pending', attempts=0)
            for arm in ARMS for seed in (0, 1)]
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(code_fingerprint=fingerprint, dataset_ready=False, jobs=jobs)
    if state['code_fingerprint'] != fingerprint:
        raise RuntimeError('Dispatch state is for a different source fingerprint')
    if [(j['tag'], j['id']) for j in state['jobs']] != [(j['tag'], j['id']) for j in jobs]:
        raise RuntimeError('Dispatch state job list does not match H34c')
    if not all(j['status'] in ('pending', 'submitting', 'active', 'complete', 'failed') and 0 <= j['attempts'] <= 2 for j in state['jobs']):
        raise RuntimeError('Dispatch state has an illegal job')
    dump(STATE, state)
    if not state['dataset_ready']:
        _, text = cli('datasets', 'status', 'ezzzzzx/pdp-code-h34c', allow_failure=True)
        low = text.lower()
        missing = any(s in low for s in ('403', '404', 'forbidden', "don't have access", 'not found', 'does not exist'))
        exists = (not missing) and any(s in low for s in ('ready', 'pending', 'processing'))
        if not missing and not exists:
            raise RuntimeError('Cannot establish dataset existence: ' + text)
        if exists:
            cli('datasets', 'version', '-p', str(BASE / 'code'), '-m', 'H34c source ' + fingerprint[:16], timeout=600)
        else:
            code, text = cli('datasets', 'create', '-p', str(BASE / 'code'), allow_failure=True, timeout=600)
            print(text.strip(), flush=True)
            if code:
                created = text.lower()
                if any(s in created for s in ('409', 'already exists', 'conflict')):
                    cli('datasets', 'version', '-p', str(BASE / 'code'), '-m', 'H34c source ' + fingerprint[:16], timeout=600)
                else:
                    raise RuntimeError('Dataset create failed: ' + text)
        while True:
            code, text = cli('datasets', 'status', 'ezzzzzx/pdp-code-h34c', allow_failure=True)
            low = text.lower()
            print(text.strip(), flush=True)
            if code == 0 and 'ready' in low:
                break
            if any(s in low for s in ('403', '404', 'forbidden', "don't have access", 'not found')):
                time.sleep(30)
                continue
            if any(s in low for s in ('error', 'fail')):
                raise RuntimeError(text)
            time.sleep(30)
        state['dataset_ready'] = True
        dump(STATE, state)
    while True:
      try:
        for job in state['jobs']:
            if job['status'] not in ('active', 'submitting'):
                continue
            _, text = cli('kernels', 'status', job['id'])
            time.sleep(2)
            kind = status_kind(text)
            if job['status'] == 'submitting' and kind != 'active':
                if kind != 'complete':
                    raise RuntimeError('Ambiguous interrupted push: ' + job['id'])
            if kind == 'complete':
                fetch(job, fingerprint)
                job['status'] = 'complete'
            elif kind == 'failed':
                failed = OUT / 'failed' / f"{job['tag']}-attempt{job['attempts']}"
                failed.mkdir(parents=True, exist_ok=True)
                cli('kernels', 'output', job['id'], '-p', str(failed), allow_failure=True)
                job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
            else:
                job['status'] = 'active'
            dump(STATE, state)
        if any(j['status'] == 'failed' for j in state['jobs']):
            subprocess.run([sys.executable, str(BASE / 'summarize.py')], cwd=ROOT, check=False)
            raise RuntimeError('A kernel exhausted its one retry')
        if all(j['status'] == 'complete' for j in state['jobs']):
            break
        launched = False
        for job in state['jobs']:
            if job['status'] != 'pending':
                continue
            found = account_active()
            if found is None:
                print('wait slots reason=rate_limit', flush=True)
                break
            active = set(found)
            local = {j['id'] for j in state['jobs'] if j['status'] == 'active'}
            active |= local
            if len(active) >= 5:
                print('wait slots active=' + str(len(active)) + ' ' + ','.join(sorted(active)), flush=True)
                break
            if job['attempts'] >= 2:
                raise RuntimeError('Retry count exceeded before push: ' + job['id'])
            job.update(status='submitting', attempts=job['attempts'] + 1)
            dump(STATE, state)
            try:
                _, text = cli('kernels', 'push', '-p', str(BASE / job['tag']))
            except RateLimited:
                job.update(status='pending', attempts=job['attempts'] - 1)
                dump(STATE, state)
                raise
            print(text.strip(), flush=True)
            job['status'] = 'active'
            dump(STATE, state)
            launched = True
        if not launched:
            time.sleep(60)
        else:
            time.sleep(30)
      except RateLimited as exc:
        print(f'wait slots reason=rate_limit {exc}', flush=True)
        time.sleep(120)
        continue
    subprocess.run([sys.executable, str(BASE / 'summarize.py')], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--dispatch', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'dispatch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, fingerprint = prepare()
        if args.dispatch and not args.prepare_only:
            dispatch(manifest, fingerprint)


if __name__ == '__main__':
    main()
