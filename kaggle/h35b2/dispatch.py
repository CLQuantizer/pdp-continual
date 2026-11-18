"""Prepare offline, or resume a foreground dispatcher. Never launches on import.

Each seed is three private CPU kernels: rand+oracle+rho, then prog_snap_5,
then prog_snap_20. Smoke extrapolation for every split kernel must be <= 85
minutes (reference build counted on the kernel that runs rho). Account-wide
<=5 active (running/queued/pending); H35b2 <=2, including jobs launched
outside this dispatcher. At most two push attempts per job. A process lock
serializes H35b2 dispatchers. The existing account push lock under results/h36
is shared with the H36 and H35b dispatchers; status cache is local to H35b2:
the API has no atomic account-wide reservation.
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
from h35b2_snapshot import POLICIES, SEEDS, dump
BASE, OUT = ROOT/'kaggle/h35b2', ROOT/'results/h35b2'
STATE = OUT/'dispatch.json'
CLI = str(ROOT/'.venv/bin/kaggle')
ACCOUNT_LOCK = ROOT/'results'/'h36'/'account-push.lock'
STATUS_CACHE = OUT/'kernel-status-cache.json'
ACCOUNT_CAP = 5
FAMILY_CAP = 2
FAMILY_PREFIX = 'ezzzzzx/h35b2-'
DATASET_ID = 'ezzzzzx/pdp-code-h35b2'
FILES = ('h35b2_snapshot.py','h35b_neural_diet.py','h35_corpus.py','h26_cued.py')
SPLIT_LIMIT_MIN = 85
# One kernel for the three baselines; each snapshot lag on its own.
# Order matches POLICIES so a seed's arms stay disjoint and complete.
POLICY_GROUPS = (
    ('base', ('rand', 'oracle', 'rho')),
    ('k5', ('prog_snap_5',)),
    ('k20', ('prog_snap_20',)),
)


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


def job_specs():
    covered = tuple(p for _, policies in POLICY_GROUPS for p in policies)
    if covered != tuple(POLICIES):
        raise RuntimeError('split groups must cover each arm once, in POLICIES order')
    jobs = []
    for seed in SEEDS:
        for suffix, policies in POLICY_GROUPS:
            tag = f's{seed}-{suffix}'
            jobs.append(dict(tag=tag, id=f'ezzzzzx/h35b2-{tag}', seed=seed, policies=list(policies)))
    return jobs


def split_kernel_minutes(smoke):
    """Same extrapolation as the smoke report, applied to one kernel's arms.

    reference_seconds is the rho reference build. It is charged only to the
    kernel that runs rho. Data-load time is not in the smoke extrapolation.
    """
    minutes = smoke['estimated_policy_minutes']
    reference = float(smoke['reference_seconds']) / 60.0
    rows = []
    for job in job_specs():
        estimate = sum(float(minutes[p]) for p in job['policies'])
        if 'rho' in job['policies']:
            estimate += reference
        rows.append(dict(tag=job['tag'], id=job['id'], seed=job['seed'],
                         policies=list(job['policies']), estimated_minutes=estimate))
    return rows


def prepare():
    smoke_path = OUT/'smoke.json'
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
    if not smoke.get('ok'):
        raise RuntimeError('H35b2 smoke and timing must pass before kernel preparation')
    estimates = split_kernel_minutes(smoke)
    checked = next(row for row in estimates if row['tag'] == 's6-base')
    longest = max(estimates, key=lambda row: row['estimated_minutes'])
    if checked['estimated_minutes'] > SPLIT_LIMIT_MIN or longest['estimated_minutes'] > SPLIT_LIMIT_MIN:
        raise RuntimeError(
            f'split kernel estimate exceeds {SPLIT_LIMIT_MIN} min: '
            f'{checked["tag"]}={checked["estimated_minutes"]:.3f} '
            f'longest={longest["tag"]}={longest["estimated_minutes"]:.3f}')
    dump(OUT/'split-estimates.json', dict(limit_minutes=SPLIT_LIMIT_MIN, checked=checked,
                                          longest=longest, kernels=estimates))
    print(json.dumps(dict(checked_kernel=checked['id'],
                          estimated_minutes=checked['estimated_minutes'],
                          limit_minutes=SPLIT_LIMIT_MIN,
                          longest_kernel=longest['id'],
                          longest_minutes=longest['estimated_minutes'])), flush=True)
    code = BASE/'code'
    code.mkdir(parents=True,exist_ok=True)
    hashes = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    manifest = dict(hashes=hashes, jobs=job_specs())
    manifest_path = BASE/'code-sha256.json'
    if STATE.exists() and (not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest):
        raise RuntimeError('Source/configuration changed after dispatch began; refusing mixed versions')
    for name in hashes:
        shutil.copyfile(ROOT/name,code/name)
    dump(code/'dataset-metadata.json',dict(id='ezzzzzx/pdp-code-h35b2',title='pdp code h35b2',licenses=[dict(name='other')]))
    dump(manifest_path,manifest)
    fingerprint = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    for job in manifest['jobs']:
        folder = BASE/job['tag']
        folder.mkdir(exist_ok=True)
        dump(folder/'kernel-metadata.json',dict(id=job['id'],title=job['id'].split('/')[1],code_file='kernel.py',
            language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,
            enable_internet=False,dataset_sources=['ezzzzzx/enwik8','ezzzzzx/pdp-code-h35b2'],
            competition_sources=[],kernel_sources=[],model_sources=[]))
        (folder/'kernel.py').write_text(kernel_source(job,fingerprint,hashes))
    live = {job['tag'] for job in manifest['jobs']}
    for child in BASE.iterdir():
        if child.is_dir() and child.name not in live and child.name != 'code' and (child/'kernel-metadata.json').is_file():
            shutil.rmtree(child)
    print(json.dumps(dict(prepared=len(manifest['jobs']),code_fingerprint=fingerprint)),flush=True)
    return manifest,fingerprint


def kernel_source(job, fingerprint, hashes):
    return f'''"""H35b2 private CPU, seed {job['seed']}, {job['policies']}."""
import os
import hashlib
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h35b2_snapshot.py'))
assert len(code)==1,code
code_root=code[0].parent
expected_hashes={hashes!r}
for name, expected in expected_hashes.items():
    assert hashlib.sha256((code_root/name).read_bytes()).hexdigest()==expected,name
sys.path.insert(0,str(code_root))
import h35b2_snapshot as h
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
if __name__ == '__main__':
    h.kernel_main({job['seed']},'/kaggle/working',{job['id']!r},policies={job['policies']!r},
                  data=str(raw_paths[0]),fingerprint={fingerprint!r})
'''


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
    """Use the existing account push lock shared by H36 and H35b."""
    # Reuse the other dispatchers' existing lock without writing outside H35b2.
    # If unavailable, fail closed; the runner must establish the shared lock.
    with ACCOUNT_LOCK.open('r') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _rate_limited(text):
    low = text.lower()
    return '429' in low or 'too many' in low or 'rate limit' in low


def status_text(ref):
    """Return one kernel's status body. HTTP 429 waits; any other body returns."""
    pause = 20
    while True:
        for attempt in range(4):
            _, text = cli('kernels', 'status', ref, allow_failure=True)
            if _rate_limited(text):
                print('status 429 ' + ref, flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            return text
        print('status paused ' + ref, flush=True)
        time.sleep(pause)
        pause = min(180, pause * 2)


def list_kernel_stamps():
    """Every owned kernel, newest run first. --page, not page-token: the token is empty."""
    stamps = {}
    order = []
    for page in range(1, 51):
        args = ['kernels', 'list', '--mine', '--page-size', '100', '--page', str(page),
                '--csv', '--sort-by', 'dateRun']
        pause = 20
        hard = 0
        while True:
            code, text = cli(*args, allow_failure=True)
            if _rate_limited(text):
                print('kernel list 429', flush=True)
                time.sleep(pause)
                pause = min(180, pause * 2)
                continue
            if code == 0 and ('ref,' in text or 'Not found' in text):
                break
            hard += 1
            print('kernel list retry', flush=True)
            if hard >= 4:
                raise RuntimeError('Kernel list failed')
            time.sleep(5 * hard)
        lines = text.splitlines()
        body = '\n'.join(s for s in lines if not s.startswith('Next Page Token ='))
        if body.strip() == 'Not found' or not body.strip():
            break
        rows = list(csv.DictReader(io.StringIO(body)))
        if rows and 'ref' not in rows[0]:
            raise RuntimeError('Unrecognized account listing')
        for row in rows:
            ref = row['ref']
            if ref not in stamps:
                order.append(ref)
                stamps[ref] = row.get('lastRunTime') or ''
        if len(rows) < 100:
            break
    else:
        raise RuntimeError('Account kernel list did not end within 50 pages')
    if not stamps:
        raise RuntimeError('Account kernel list is empty')
    return stamps, order


def account_active():
    """Status owned kernels. Terminal rows are cached by lastRunTime.

    The newest 40 are always rechecked: a rerun can keep its previous
    lastRunTime until it finishes. Active rows are not cached, so the next
    check sees them again. Caller holds account_push_lock.
    """
    stamps, order = list_kernel_stamps()
    cache = load_status_cache()
    newest = set(order[:40])
    active = []
    checked = 0
    now = time.time()
    for ref in order:
        stamp = stamps[ref]
        cached = cache.get(ref)
        # A missing `at` is a row just written by the previous sweep.
        fresh = (isinstance(cached, dict) and cached.get('lastRunTime') == stamp
                 and cached.get('kind') in ('complete', 'failed', 'absent'))
        recent = isinstance(cached, dict) and now - float(cached.get('at') or now) < 600
        if fresh and (ref not in newest or recent):
            continue
        checked += 1
        if checked == 1 or checked % 25 == 0:
            print(f'account status {checked} {ref}', flush=True)
        report = status_text(ref)
        if 'has status' not in report.lower():
            low = report.lower()
            # A listed kernel with no session is not a running worker.
            if '404' in low or 'not found' in low:
                cache[ref] = dict(lastRunTime=stamp, kind='absent', at=now)
                save_status_cache(cache)
                time.sleep(0.8)
                continue
            if any(s in low for s in ('403', 'forbidden', "don't have access")):
                raise RuntimeError('Cannot establish account slots: unreadable ' + ref)
            raise RuntimeError('Refusing to push while a kernel status is unknown: ' + ref)
        kind = status_kind(report)
        if kind in ('complete', 'failed'):
            cache[ref] = dict(lastRunTime=stamp, kind=kind, at=now)
            save_status_cache(cache)
        elif isinstance(cached, dict):
            cache.pop(ref, None)
            save_status_cache(cache)
        if kind == 'active':
            active.append(ref)
        time.sleep(0.8)
    print(f'account slot check: listed {len(stamps)} statused {checked} active {len(active)}', flush=True)
    dump(OUT/'account-active.json', dict(at=time.time(), active=active, total=len(active), listed=len(stamps)))
    return active


def validate_payload(row, job, fingerprint, smoke=None):
    import math
    assert row['kernel_id'] == job['id'] and row['seed'] == job['seed']
    assert row['code_fingerprint'] == fingerprint
    assert row['policies'] == job['policies']
    assert [r['policy'] for r in row['runs']] == job['policies']
    smoke = smoke if smoke is not None else json.loads((OUT/'smoke.json').read_text())
    assert row['sha256_e10m'] == smoke['sha256_e10m']
    for run in row['runs']:
        assert run['seed'] == job['seed'] and run['budget'] == 200 and run['pool_documents'] == 1600
        assert run['checkpoint_every'] == 25
        assert [c['documents'] for c in run['checkpoints']] == list(range(25,201,25))
        assert all(math.isfinite(c['clean_bpb']) for c in run['checkpoints'])
        assert abs(run['T_clean']-sum(c['clean_bpb'] for c in run['checkpoints'])/8) < 1e-10
        assert run['final'] == run['checkpoints'][-1]['clean_bpb']
        assert len(run['selected']) == len(set(run['selected'])) == 200
        assert all(0 <= idx < 1600 for idx in run['selected'])
        assert sum(run['kinds_read'].values()) == 200
        assert abs(sum(run['shares'].values())-1) < 1e-10
        assert all(abs(run['shares'][k]-v/200) < 1e-10 for k,v in run['kinds_read'].items())
        if run['policy'].startswith('prog_snap_'):
            lag = int(run['policy'].rsplit('_',1)[1])
            assert run['snapshot_lag'] == lag
            assert run['peek_bytes'] == 512 and run['candidates'] == 16
            assert run['snapshot_documents'] == [max(0,t-lag) for t in range(200)]
        if run['policy'] == 'oracle':
            assert run['kinds_read']['clean'] == 200


def kernel_report(ref):
    report = status_text(ref)
    if 'has status' not in report.lower():
        raise RuntimeError('Refusing to continue while a kernel status is unknown: ' + ref)
    return report


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


def fetch(job, fingerprint):
    dest = OUT/'kernels'/job['tag']
    dest.mkdir(parents=True,exist_ok=True)
    cli('kernels','output',job['id'],'-p',str(dest),'-o')
    row = json.loads((dest/'results.json').read_text())
    validate_payload(row,job,fingerprint)


def dispatch(manifest, fingerprint):
    smoke = json.loads((OUT/'smoke.json').read_text())
    if not smoke['ok']:
        raise RuntimeError('Smoke gate failed; no kernels may launch')
    load_token()
    jobs = [dict(**job,status='pending',attempts=0) for job in manifest['jobs']]
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(code_fingerprint=fingerprint,dataset_ready=False,jobs=jobs)
    assert state['code_fingerprint'] == fingerprint
    assert [(j['tag'],j['id'],j['seed'],j['policies']) for j in state['jobs']] == [(j['tag'],j['id'],j['seed'],j['policies']) for j in jobs]
    assert all(j['status'] in ('pending','submitting','active','complete','failed') and 0 <= j['attempts'] <= 2 for j in state['jobs'])
    for job in state['jobs']:
        if job['status'] == 'complete':
            path = OUT/'kernels'/job['tag']/'results.json'
            if path.exists():
                validate_payload(json.loads(path.read_text()),job,fingerprint)
            else:
                fetch(job,fingerprint)
    dump(STATE,state)
    if not state['dataset_ready']:
        ensure_dataset(fingerprint, 'H35b2 source ')
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
                fetch(job,fingerprint)
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
            # The lock is shared with H36 so the two cannot both pass the cap.
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
