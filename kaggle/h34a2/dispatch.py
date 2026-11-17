"""Resumable foreground H34a2 dispatcher: <=3 active kernels, one retry."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from h34a2_logit_sum import ARMS
BASE=ROOT/'kaggle/h34a2'
OUT=ROOT/'results/h34a2'
STATE=OUT/'dispatch.json'
CLI=str(ROOT/'.venv/bin/kaggle')
FILES=('h20_world.py','h26_cued.py','h30w_ring.py','h30x_trace.py','h30y_roots.py',
       'h32_mixer.py','h32b_mixer.py','h33_bank_text.py','h34a_local_depth.py','h34a2_logit_sum.py')


def load_token():
    if os.environ.get('KAGGLE_API_TOKEN'): return
    for raw in (ROOT/'.env').read_text().splitlines():
        line=raw.strip()
        if line.startswith('export '): line=line[7:].strip()
        if not line or line.startswith('#') or '=' not in line: continue
        key,val=line.split('=',1)
        if key.strip()!='KAGGLE_API_TOKEN': continue
        val=val.strip()
        if len(val)>=2 and val[0]==val[-1] and val[0] in "\"'": val=val[1:-1]
        if not val: raise RuntimeError('KAGGLE_API_TOKEN is empty')
        os.environ['KAGGLE_API_TOKEN']=val
        return
    raise RuntimeError('KAGGLE_API_TOKEN is unset')


def cli(*args,allow_failure=False):
    p=subprocess.run([CLI,*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
    output=p.stdout.replace(os.environ.get('KAGGLE_API_TOKEN') or '\0','[REDACTED]')
    if p.returncode and not allow_failure:
        raise RuntimeError(f'Kaggle command failed: {args[:2]}\n{output}')
    return output


def save(state):
    tmp=STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2)+'\n'); tmp.replace(STATE)


def prepare():
    code=BASE/'code'; code.mkdir(parents=True,exist_ok=True)
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    manifest=BASE/'code-sha256.json'
    if STATE.exists() and not manifest.is_file():
        raise RuntimeError('Dispatch state exists without its code manifest')
    if STATE.exists() and json.loads(manifest.read_text())!=hashes:
        raise RuntimeError('Source changed since dispatch preparation; refusing mixed code versions')
    for name in FILES: shutil.copyfile(ROOT/name,code/name)
    (code/'dataset-metadata.json').write_text(json.dumps(dict(title='pdp code h34a2',id='ezzzzzx/pdp-code-h34a2',licenses=[dict(name='other')]),indent=2)+'\n')
    manifest.write_text(json.dumps(hashes,indent=2)+'\n')
    for arm in ARMS:
        for seed in (0,1):
            folder=BASE/f'{arm}-s{seed}'; folder.mkdir(exist_ok=True)
            slug=f'h34a2-{arm.replace("_","-")}-s{seed}'
            meta=dict(id=f'ezzzzzx/{slug}',title=slug,code_file='kernel.py',language='python',kernel_type='script',
                      is_private=True,enable_gpu=False,enable_tpu=False,enable_internet=False,
                      dataset_sources=['ezzzzzx/enwik8','ezzzzzx/pdp-code-h34a2'],competition_sources=[],kernel_sources=[],model_sources=[])
            (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
            (folder/'kernel.py').write_text(f'''"""Private CPU H34a2: {arm}, seed {seed}."""
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h34a2_logit_sum.py'))
assert len(code)==1, code
sys.path.insert(0,str(code[0].parent))
import h34a2_logit_sum as h
h.h33.configure(1)
data=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(data)==1, data
raw,digest=h.h33.load_checked(str(data[0]))
r=h.run({arm!r},{seed},raw[:300_000])
r.update(sha256_e10m=digest,kernel_id='ezzzzzx/{slug}')
h.dump('/kaggle/working/results.json',r)
''')


def status_kind(report):
    low=report.lower().split('has status',1)[-1].splitlines()[0]
    if 'complete' in low: return 'complete'
    if any(x in low for x in ('error','cancel','fail')): return 'failed'
    if any(x in low for x in ('running','queued','pending')): return 'active'
    raise RuntimeError('Unknown remote status; refusing to submit more work: '+report)


def validate_result(job,smoke):
    dest=OUT/'kernels'/job['tag']; dest.mkdir(parents=True,exist_ok=True)
    print(cli('kernels','output',job['id'],'-p',str(dest)),flush=True)
    r=json.loads((dest/'results.json').read_text())
    assert r['kernel_id']==job['id'] and r['n_bytes']==300_000
    assert r['sha256_e10m']==smoke['sha256_e10m']


def dispatch():
    load_token()
    smoke=json.loads((OUT/'smoke.json').read_text())
    assert smoke['ok'] and smoke['n_bytes']==300_000
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(dataset_created=False,jobs=[
        dict(tag=f'{a}-s{s}',id=f'ezzzzzx/h34a2-{a.replace("_","-")}-s{s}',status='pending',attempts=0)
        for a in ARMS for s in (0,1)])
    expected={f'{a}-s{s}':f'ezzzzzx/h34a2-{a.replace("_","-")}-s{s}' for a in ARMS for s in (0,1)}
    assert {j['tag']:j['id'] for j in state['jobs']}==expected and len(state['jobs'])==20
    assert sum(j['status'] in ('active','submitting') for j in state['jobs'])<=3
    assert all(j['status'] in ('pending','submitting','active','complete','failed') and 0<=j['attempts']<=2 for j in state['jobs'])
    save(state)
    if not state['dataset_created']:
        report=cli('datasets','status','ezzzzzx/pdp-code-h34a2',allow_failure=True)
        if 'failed' in report.lower():
            raise RuntimeError('Existing code dataset processing failed')
        if not any(x in report.lower() for x in ('ready','pending','processing')):
            # Kaggle datasets create defaults to private; never pass --public.
            print(cli('datasets','create','-p',str(BASE/'code')),flush=True)
        state['dataset_created']=True; save(state)
    while True:
        report=cli('datasets','status','ezzzzzx/pdp-code-h34a2')
        print(report.strip(),flush=True)
        if 'ready' in report.lower(): break
        if any(x in report.lower() for x in ('error','fail')): raise RuntimeError('Code dataset processing failed')
        time.sleep(30)
    while True:
        for job in state['jobs']:
            if job['status']=='submitting':
                # A push may have succeeded just before interruption. Reconcile
                # remotely; never automatically repeat an ambiguous submission.
                raise RuntimeError(f"Ambiguous push for {job['id']}: inspect remote version/status, then set dispatch.json status to active if accepted or pending if not; attempts already includes this push")
            if job['status']!='active': continue
            report=cli('kernels','status',job['id']); print(report.strip(),flush=True)
            (OUT/f"{job['tag']}.status.txt").write_text(report)
            kind=status_kind(report)
            if kind=='complete':
                validate_result(job,smoke); job['status']='complete'
            elif kind=='failed':
                dest=OUT/'failed'/f"{job['tag']}-attempt{job['attempts']}"; dest.mkdir(parents=True,exist_ok=True)
                print(cli('kernels','output',job['id'],'-p',str(dest),allow_failure=True),flush=True)
                job['status']='pending' if job['attempts']<2 else 'failed'
            save(state)
        if any(j['status']=='failed' for j in state['jobs']): raise RuntimeError('Kernel exhausted its single retry')
        active=sum(j['status']=='active' for j in state['jobs'])
        for job in state['jobs']:
            if active>=3: break
            if job['status']!='pending': continue
            if job['attempts']>=2: raise RuntimeError('Retry limit reached')
            job.update(status='submitting',attempts=job['attempts']+1); save(state)
            print(cli('kernels','push','-p',str(BASE/job['tag'])).strip(),flush=True)
            job['status']='active'; save(state); active+=1
        if all(j['status']=='complete' for j in state['jobs']): break
        time.sleep(30)
    subprocess.run([sys.executable,str(BASE/'summarize.py')],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare-only',action='store_true'); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare()
        if not args.prepare_only: dispatch()

if __name__=='__main__': main()
