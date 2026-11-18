"""Prepare offline, or resume a foreground dispatcher. Never launches on import.

Account-wide <=5 active (running/queued/pending); H36 <=2, including jobs
launched outside this dispatcher. At most two push attempts per job. A
process lock serializes H36 dispatchers. Other experiment dispatchers must
also respect the account cap: the API has no atomic account-wide reservation.
"""
import argparse
import csv
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


def account_active():
    """Exhaust account pagination; inspect every remote kernel's live status."""
    refs = set()
    token = None
    while True:
        args = ['kernels','list','--mine','--page-size','100','--csv']
        if token:
            args += ['--page-token',token]
        _, text = cli(*args)
        lines = text.splitlines()
        tokens = [s.split('=',1)[1].strip() for s in lines if s.startswith('Next Page Token =')]
        body = '\n'.join(s for s in lines if not s.startswith('Next Page Token ='))
        if body.strip() != 'Not found':
            rows = list(csv.DictReader(io.StringIO(body)))
            if rows and 'ref' not in rows[0]:
                raise RuntimeError('Unrecognized account listing')
            refs.update(r['ref'] for r in rows)
        if not tokens:
            break
        if token == tokens[0]:
            raise RuntimeError('Repeated account page token')
        token = tokens[0]
    active = []
    for ref in sorted(refs):
        _, text = cli('kernels','status',ref)
        if status_kind(text) == 'active':
            active.append(ref)
    dump(OUT/'account-active.json',dict(at=time.time(),active=active,total=len(active)))
    return active


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
        code, text = cli('datasets','status','ezzzzzx/pdp-code-h36',allow_failure=True)
        if code:
            if not any(s in text.lower() for s in ('not found','404','does not exist')):
                raise RuntimeError('Cannot establish dataset existence: '+text)
            cli('datasets','create','-p',str(BASE/'code'))  # Private by default.
        else:
            cli('datasets','version','-p',str(BASE/'code'),'-m','H36 source '+fingerprint[:16])
        while True:
            _, text = cli('datasets','status','ezzzzzx/pdp-code-h36')
            if 'ready' in text.lower():
                break
            if any(s in text.lower() for s in ('error','fail')):
                raise RuntimeError(text)
            time.sleep(30)
        state['dataset_ready'] = True
        dump(STATE,state)
    while True:
        for job in state['jobs']:
            if job['status'] not in ('active','submitting'):
                continue
            _, text = cli('kernels','status',job['id'])
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
            active = account_active()
            local = {j['id'] for j in state['jobs'] if j['status'] == 'active'}
            active = set(active) | local  # Protect against listing latency.
            if len(active) >= 5 or sum(ref.startswith('ezzzzzx/h36-') for ref in active) >= 2:
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
