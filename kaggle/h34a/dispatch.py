"""Foreground Kaggle scheduler. Three H34a kernels maximum; one retry maximum."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = str(ROOT/'.venv/bin/kaggle')
BASE = ROOT/'kaggle/h34a'
OUT = ROOT/'results/h34a'
STATE = OUT/'dispatch.json'
ARMS = ('plain_1','plain_2','plain_4','left_1','left_2','left_4','left_4_noskip','wide_1')


def load_token():
    # The Kaggle CLI reads KAGGLE_API_TOKEN from the environment only.
    if os.environ.get('KAGGLE_API_TOKEN'):
        return
    path = ROOT/'.env'
    if not path.is_file():
        raise RuntimeError('KAGGLE_API_TOKEN is unset and .env is missing')
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith('export '):
            line = line[len('export '):].strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        if key.strip() != 'KAGGLE_API_TOKEN':
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in '"\'':
            val = val[1:-1]
        if not val:
            raise RuntimeError('KAGGLE_API_TOKEN is empty')
        os.environ['KAGGLE_API_TOKEN'] = val
        return
    raise RuntimeError('KAGGLE_API_TOKEN is unset')


def cli(*args):
    proc = subprocess.run([CLI, *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120)
    if proc.returncode:
        raise RuntimeError(f'Kaggle command failed: {args[0:2]}\n{proc.stdout}')
    return proc.stdout


def save(state):
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(STATE)


def prepare(n):
    for arm in ARMS:
        for seed in (0,1):
            tag=f'{arm}-s{seed}'
            slug=f'h34a-{arm.replace("_","-")}-s{seed}'
            folder=BASE/tag
            folder.mkdir(parents=True,exist_ok=True)
            metadata=dict(id=f'ezzzzzx/{slug}',title=slug,code_file='kernel.py',language='python',
                          kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,
                          enable_internet=False,dataset_sources=['ezzzzzx/enwik8','ezzzzzx/pdp-code-h34a'],
                          competition_sources=[],kernel_sources=[],model_sources=[])
            (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            script=f'''"""Private CPU H34a life: {arm}, seed {seed}."""
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h34a_local_depth.py'))
assert len(code)==1, code
sys.path.insert(0,str(code[0].parent))
import h34a_local_depth as h
h.h33.configure(1)
data=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(data)==1, data
raw,digest=h.h33.load_checked(str(data[0]))
r=h.run({arm!r},{seed},raw[:{n}])
r.update(sha256_e10m=digest,kernel_id='ezzzzzx/{slug}')
h.dump('/kaggle/working/results.json',r)
'''
            (folder/'kernel.py').write_text(script)


def main():
    load_token()
    smoke=json.loads((OUT/'smoke.json').read_text())
    assert smoke['ok']
    prepare(smoke['n_bytes'])
    if '--prepare-only' in sys.argv:
        return
    state=json.loads(STATE.read_text()) if STATE.exists() else dict(dataset_created=False,jobs=[])
    if not state['jobs']:
        state['jobs']=[dict(tag=f'{a}-s{s}',id=f'ezzzzzx/h34a-{a.replace("_","-")}-s{s}',status='pending',attempts=0) for a in ARMS for s in (0,1)]
    save(state)
    if not state['dataset_created']:
        print(cli('datasets','create','-p',str(BASE/'code')),flush=True)
        state['dataset_created']=True
        save(state)
    while True:
        report=cli('datasets','status','ezzzzzx/pdp-code-h34a')
        print(report.strip(),flush=True)
        low = report.lower()
        if 'ready' in low:
            break
        # dataset_status prints the enum name, so a failed bundle is "failed".
        if 'error' in low or 'fail' in low:
            raise RuntimeError('Code dataset processing failed')
        time.sleep(30)
    while True:
        for job in state['jobs']:
            if job['status'] != 'active':
                continue
            report=cli('kernels','status',job['id'])
            print(report.strip(),flush=True)
            (OUT/f"{job['tag']}.status.txt").write_text(report)
            status=report.lower()
            if 'complete' in status:
                dest=OUT/'kernels'/job['tag']
                dest.mkdir(parents=True,exist_ok=True)
                print(cli('kernels','output',job['id'],'-p',str(dest)),flush=True)
                result=json.loads((dest/'results.json').read_text())
                assert result['kernel_id']==job['id'] and result['n_bytes']==smoke['n_bytes']
                job['status']='complete'
            elif 'error' in status or 'cancel' in status or 'fail' in status:
                dest=OUT/'failed'/f"{job['tag']}-attempt{job['attempts']}"
                dest.mkdir(parents=True,exist_ok=True)
                print(cli('kernels','output',job['id'],'-p',str(dest)),flush=True)
                job['status']='pending' if job['attempts'] < 2 else 'failed'
            save(state)
        if any(j['status']=='failed' for j in state['jobs']):
            raise RuntimeError('Kernel exhausted its single retry')
        active=sum(j['status']=='active' for j in state['jobs'])
        for job in state['jobs']:
            if active>=3:
                break
            if job['status']=='pending':
                response=cli('kernels','push','-p',str(BASE/job['tag']))
                print(response.strip(),flush=True)
                job.update(status='active',attempts=job['attempts']+1)
                active+=1
                save(state)
        if all(j['status']=='complete' for j in state['jobs']):
            break
        time.sleep(30)
    subprocess.run([sys.executable,str(ROOT/'h34a_local_depth.py'),'--assemble'],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
