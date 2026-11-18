"""Prepare offline, or resume a foreground dispatcher. Never launches on import.

Account-wide <=5 active (running/queued/pending); H36 <=3, including jobs
launched outside this dispatcher. At most two push attempts per job. A
process lock serializes H36 dispatchers. The account push lock and status
cache under results/h36 are shared with the H35b dispatcher: the API has no
atomic account-wide reservation.
"""
import argparse
import csv
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h36_curious_patching import RULES, dump
BASE, OUT = ROOT/'kaggle/h36', ROOT/'results/h36'
STATE = OUT/'dispatch.json'
CLI = str(ROOT/'.venv/bin/kaggle')
ACCOUNT_LOCK = ROOT/'results'/'h36'/'account-push.lock'
STATUS_CACHE = ROOT/'results'/'h36'/'kernel-status-cache.json'
ACCOUNT_CAP = 5
FAMILY_CAP = 3
FAMILY_PREFIX = 'ezzzzzx/h36-'
DATASET_ID = 'ezzzzzx/pdp-code-h36'
FILES = ('h36_curious_patching.py','h35_corpus.py','h26_cued.py','h32_mixer.py',
         'h32b_mixer.py','h33_bank_text.py','h20_world.py','h30w_ring.py',
         'h30x_trace.py','h30y_roots.py')


def load_token():
    if os.environ.get('KAGGLE_API_TOKEN'):
        return
    for raw in (ROOT/'.env').read_text().splitlines():
        line = raw.strip()
        if line.startswith('export '):
            line = line[7:].strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=',1)
        if key.strip() != 'KAGGLE_API_TOKEN':
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value:
            raise RuntimeError('KAGGLE_API_TOKEN is empty')
        os.environ['KAGGLE_API_TOKEN'] = value
        return
    raise RuntimeError('KAGGLE_API_TOKEN is unset')


def cli(*args, allow_failure=False):
    p = subprocess.run([CLI,*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT,timeout=180)
    text = p.stdout.replace(os.environ.get('KAGGLE_API_TOKEN') or '\0','[REDACTED]')
    if p.returncode and not allow_failure:
        raise RuntimeError(f'Kaggle {args[:2]} failed: {text}')
    return p.returncode, text


def prepare():
    smoke_path = OUT/'smoke.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    documents = smoke.get('documents') or 400
    code = BASE/'code'
    code.mkdir(parents=True,exist_ok=True)
    missing = [name for name in FILES if not (ROOT/name).is_file()]
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in FILES if (ROOT/name).is_file()}
    manifest = dict(hashes=hashes, missing=missing, documents=documents)
    manifest_path = BASE/'code-sha256.json'
    if STATE.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Source/configuration changed after dispatch began; refusing mixed versions')
    for name in hashes:
        shutil.copyfile(ROOT/name,code/name)
    dump(code/'dataset-metadata.json',dict(id='ezzzzzx/pdp-code-h36',title='pdp code h36',licenses=[dict(name='other')]))
    dump(manifest_path,manifest)
    fingerprint = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    for rule in RULES:
        for seed in range(6):
            tag = f'{rule}-s{seed}'
            folder = BASE/tag
            folder.mkdir(exist_ok=True)
            slug = f'h36-{tag}'
            dump(folder/'kernel-metadata.json',dict(id=f'ezzzzzx/{slug}',title=slug,code_file='kernel.py',
                language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,
                enable_internet=False,dataset_sources=['ezzzzzx/enwik8','ezzzzzx/pdp-code-h36'],
                competition_sources=[],kernel_sources=[],model_sources=[]))
            (folder/'kernel.py').write_text(f'''"""H36 private CPU, {rule}, seed {seed}. Requires H35 corpus."""
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h36_curious_patching.py'))
assert len(code)==1,code
sys.path.insert(0,str(code[0].parent))
assert (code[0].parent/'h35_corpus.py').is_file(),'H35 corpus missing from code dataset'
import h36_curious_patching as h
# H35 locates enwik8 through its own loader; also offer the conventional
# local path for implementations with build(seed)'s default data argument.
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
h.configure()
raw=h.h26.load_enwik8(str(raw_paths[0]))
digest=h.h26.sha256_arr(raw[:h.h26.E10M])
assert digest==h.h26.E10M_SHA256
local=Path('/kaggle/working/data/enwik8/enwik8')
local.parent.mkdir(parents=True,exist_ok=True)
if not local.exists(): local.symlink_to(raw_paths[0])
os.chdir('/kaggle/working')
r=h.run({rule!r},{seed},{documents})
r.update(kernel_id='ezzzzzx/{slug}',sha256_e10m=digest,code_fingerprint={fingerprint!r})
h.dump('/kaggle/working/results.json',r)
''')
    print(json.dumps(dict(prepared=24,documents=documents,missing=missing,code_fingerprint=fingerprint)),flush=True)
    return manifest,fingerprint


def status_kind(text):
    status = text.lower().split('has status',1)[-1].strip().splitlines()[0]
    if 'complete' in status:
        return 'complete'
    if any(s in status for s in ('error','cancel','fail')):
        return 'failed'
    if any(s in status for s in ('running','queued','pending')):
        return 'active'
    raise RuntimeError('Unknown kernel status; refusing submission: '+text)


def load_status_cache():
    if not STATUS_CACHE.is_file():
        return {}
    try:
        data = json.loads(STATUS_CACHE.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save_status_cache(cache):
    dump(STATUS_CACHE, cache)


@contextmanager
def account_push_lock():
    """One reservation for H36 and H35b. Released when the holder exits."""
    ACCOUNT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with ACCOUNT_LOCK.open('a') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def account_active():
    """Status recent kernels, skipping ones already known complete or failed.

    The list call returns at most 100 rows and no further page, so sort by
    dateRun: a running kernel is among the most recently run. Caller holds
    account_push_lock so both dispatchers share this cache.
    """
    stamps = {}
    token = None
    while True:
        args = ['kernels','list','--mine','--page-size','100','--csv','--sort-by','dateRun']
        if token:
            args += ['--page-token', token]
        text = ''
        for attempt in range(4):
            code, text = cli(*args, allow_failure=True)
            if code == 0 and ('ref,' in text or 'Not found' in text or text.startswith('Next Page Token')):
                break
            print('kernel list retry', flush=True)
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError('Kernel list failed')
        lines = text.splitlines()
        tokens = [s.split('=',1)[1].strip() for s in lines if s.startswith('Next Page Token =')]
        body = '\n'.join(s for s in lines if not s.startswith('Next Page Token ='))
        if body.strip() != 'Not found':
            rows = list(csv.DictReader(io.StringIO(body)))
            if rows and 'ref' not in rows[0]:
                raise RuntimeError('Unrecognized account listing')
            for row in rows:
                stamps[row['ref']] = row.get('lastRunTime') or ''
        if not tokens:
            break
        if token == tokens[0]:
            raise RuntimeError('Repeated account page token')
        token = tokens[0]
    cache = load_status_cache()
    active = []
    checked = 0
    changed = False
    for ref in sorted(stamps):
        stamp = stamps[ref]
        cached = cache.get(ref)
        if isinstance(cached, dict) and cached.get('lastRunTime') == stamp and cached.get('kind') in ('complete', 'failed'):
            continue
        checked += 1
        if checked == 1 or checked % 25 == 0:
            print(f'account status {checked}', flush=True)
        report = ''
        for attempt in range(4):
            _, report = cli('kernels', 'status', ref, allow_failure=True)
            if 'has status' in report.lower():
                break
            print('status retry ' + ref, flush=True)
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError('Refusing to push while a kernel status is unknown: ' + ref)
        kind = status_kind(report)
        if kind in ('complete', 'failed'):
            cache[ref] = dict(lastRunTime=stamp, kind=kind)
            changed = True
        elif isinstance(cached, dict):
            cache.pop(ref, None)
            changed = True
        if kind == 'active':
            active.append(ref)
    if changed:
        save_status_cache(cache)
    print(f'account slot check: listed {len(stamps)} statused {checked} active {len(active)}', flush=True)
    dump(OUT/'account-active.json', dict(at=time.time(), active=active, total=len(active)))
    return active


def kernel_report(ref):
    report = ''
    for attempt in range(4):
        _, report = cli('kernels', 'status', ref, allow_failure=True)
        if 'has status' in report.lower():
            return report
        print('status retry ' + ref, flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError('Refusing to continue while a kernel status is unknown: ' + ref)


def dataset_missing(text):
    low = text.lower()
    return any(s in low for s in ('403', '404', 'forbidden', "don't have access", 'not found', 'does not exist'))


def ensure_dataset(fingerprint, message):
    """Create the private code dataset. A missing private slug returns 403, not 404."""
    code, text = cli('datasets', 'status', DATASET_ID, allow_failure=True)
    low = text.lower()
    missing = dataset_missing(text)
    exists = (not missing) and any(s in low for s in ('ready', 'pending', 'processing'))
    if not missing and not exists:
        raise RuntimeError('Cannot establish dataset existence: ' + text)
    if exists:
        cli('datasets', 'version', '-p', str(BASE / 'code'), '-m', message + fingerprint[:16])
    else:
        code, text = cli('datasets', 'create', '-p', str(BASE / 'code'), allow_failure=True)  # Private by default.
        print(text.strip(), flush=True)
        if code:
            created = text.lower()
            if any(s in created for s in ('409', 'already exists', 'conflict')):
                cli('datasets', 'version', '-p', str(BASE / 'code'), '-m', message + fingerprint[:16])
            else:
                raise RuntimeError('Dataset create failed: ' + text)
    while True:
        code, text = cli('datasets', 'status', DATASET_ID, allow_failure=True)
        low = text.lower()
        print(text.strip(), flush=True)
        if code == 0 and 'ready' in low:
            return
        if dataset_missing(text):
            time.sleep(30)
            continue
        if any(s in low for s in ('error', 'fail')):
            raise RuntimeError(text)
        time.sleep(30)


def fetch(job, fingerprint, documents):
    dest = OUT/'kernels'/job['tag']
    dest.mkdir(parents=True,exist_ok=True)
    cli('kernels','output',job['id'],'-p',str(dest))
    row = json.loads((dest/'results.json').read_text())
    assert row['kernel_id'] == job['id']
    assert row['code_fingerprint'] == fingerprint
    assert row['documents'] == documents and row['corpus'] == 'h35_corpus.build'
    assert abs(row['life']['rho']-.25) <= .02


def dispatch(manifest, fingerprint):
    smoke = json.loads((OUT/'smoke.json').read_text())
    if not smoke['ok']:
        raise RuntimeError('Smoke gate failed; no kernels may launch')
    if manifest['missing']:
        raise RuntimeError('Missing source files: '+', '.join(manifest['missing']))
    assert smoke['documents'] == manifest['documents']
    load_token()
    jobs = [dict(tag=f'{r}-s{s}',id=f'ezzzzzx/h36-{r}-s{s}',status='pending',attempts=0)
            for r in RULES for s in range(6)]
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(code_fingerprint=fingerprint,dataset_ready=False,jobs=jobs)
    assert state['code_fingerprint'] == fingerprint
    assert [(j['tag'],j['id']) for j in state['jobs']] == [(j['tag'],j['id']) for j in jobs]
    assert all(j['status'] in ('pending','submitting','active','complete','failed') and 0 <= j['attempts'] <= 2 for j in state['jobs'])
    dump(STATE,state)
    if not state['dataset_ready']:
        ensure_dataset(fingerprint, 'H36 source ')
        state['dataset_ready'] = True
        dump(STATE,state)
    while True:
        for job in state['jobs']:
            if job['status'] not in ('active','submitting'):
                continue
            text = kernel_report(job['id'])
            kind = status_kind(text)
            if job['status'] == 'submitting' and kind != 'active':
                # A completed run's fingerprint proves whether an interrupted
                # push succeeded; failed/older ambiguous pushes need inspection.
                if kind != 'complete':
                    raise RuntimeError('Ambiguous interrupted push: '+job['id'])
            if kind == 'complete':
                fetch(job,fingerprint,manifest['documents'])
                job['status'] = 'complete'
            elif kind == 'failed':
                failed = OUT/'failed'/f"{job['tag']}-attempt{job['attempts']}"
                failed.mkdir(parents=True,exist_ok=True)
                cli('kernels','output',job['id'],'-p',str(failed),allow_failure=True)
                job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
            else:
                job['status'] = 'active'
            dump(STATE,state)
        if any(j['status'] == 'failed' for j in state['jobs']):
            raise RuntimeError('A kernel exhausted its one retry')
        if all(j['status'] == 'complete' for j in state['jobs']):
            break
        for job in state['jobs']:
            if job['status'] != 'pending':
                continue
            # Recheck the entire account immediately before EACH push.
            # The lock is shared with H35b so the two cannot both pass the cap.
            with account_push_lock():
                active = set(account_active())
                local = {j['id'] for j in state['jobs'] if j['status'] == 'active'}
                active |= local  # Protect against listing latency.
                family = sum(ref.startswith(FAMILY_PREFIX) for ref in active)
                print(f'account slots {len(active)} family {family}/{FAMILY_CAP}', flush=True)
                if len(active) >= ACCOUNT_CAP or family >= FAMILY_CAP:
                    break
                assert job['attempts'] < 2
                job.update(status='submitting',attempts=job['attempts']+1)
                dump(STATE,state)
                _, text = cli('kernels','push','-p',str(BASE/job['tag']))
                print(text.strip(),flush=True)
                job['status'] = 'active'
                dump(STATE,state)
        time.sleep(30)
    subprocess.run([sys.executable,str(BASE/'summarize.py')],cwd=ROOT,check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only',action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,fingerprint = prepare()
        if not args.prepare_only:
            dispatch(manifest,fingerprint)

if __name__ == '__main__':
    main()
