"""Offline staging and resumable foreground H34j dispatcher.

Serializes this dispatcher; every push rechecks <=4 H34j and <=5 account
slots. Other dispatchers must also obey the account cap (no API reservation).
Interrupted ambiguous pushes fail closed; rerun after inspecting remote state.
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
import h34j_long as h
BASE, OUT = ROOT/'kaggle/h34j', ROOT/'results/h34j'
STATE = OUT/'dispatch.json'
CLI = str(ROOT/'.venv/bin/kaggle')

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n'); tmp.replace(path)

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


def status_kind(text):
    if 'has status' not in text.lower():
        raise RuntimeError('Unknown kernel status; refusing submission: '+text)
    status = text.lower().split('has status',1)[-1].strip().splitlines()[0]
    if 'complete' in status:
        return 'complete'
    if any(s in status for s in ('error','cancel','fail')):
        return 'failed'
    if any(s in status for s in ('running','queued','pending')):
        return 'active'
    raise RuntimeError('Unknown kernel status; refusing submission: '+text)


def kernel_status_text(ref):
    """Return a real status body. A rate limit pauses and retries."""
    pause = 15
    while True:
        for attempt in range(4):
            _, text = cli('kernels','status',ref,allow_failure=True)
            low = text.lower()
            if '429' in low or 'too many' in low or 'rate limit' in low:
                time.sleep(2 + attempt * 3)
                continue
            return text
        print(f'account status paused {ref}', flush=True)
        time.sleep(pause)
        pause = min(60, pause * 2)


def account_active():
    """Exhaust account pagination; inspect every remote kernel's live status.

    dateRun puts kernels that are running now on the first page. The default
    hotness sort can omit them, and the account cap would not see them.
    """
    refs = set()
    token = None
    while True:
        args = ['kernels','list','--mine','--page-size','100','--csv','--sort-by','dateRun']
        if token:
            args += ['--page-token',token]
        _, text = cli(*args)
        lines = text.splitlines()
        tokens = [s.split('=',1)[1].strip() for s in lines if s.startswith('Next Page Token =')]
        body = '\n'.join(s for s in lines if not s.startswith('Next Page Token ='))
        if body.strip() != 'Not found':
            reader = csv.DictReader(io.StringIO(body))
            rows = list(reader)
            if not reader.fieldnames or 'ref' not in reader.fieldnames:
                raise RuntimeError('Unrecognized account listing')
            refs.update(r['ref'] for r in rows)
        if not tokens:
            break
        if token == tokens[0]:
            raise RuntimeError('Repeated account page token')
        token = tokens[0]
    active = []
    ordered = sorted(refs)
    for i, ref in enumerate(ordered, 1):
        if i == 1 or i % 25 == 0 or i == len(ordered):
            print(f'account status {i}/{len(ordered)}', flush=True)
        text = kernel_status_text(ref)
        low = text.lower()
        if 'has status' not in low:
            # A private kernel this token cannot read is not a free slot and not proof of one.
            if any(s in low for s in ('403','404','forbidden',"don't have access",'not found')):
                raise RuntimeError('Cannot establish account slots: unreadable '+ref)
            raise RuntimeError('Refusing to push while a kernel status is unknown: '+ref)
        if status_kind(text) == 'active':
            active.append(ref)
    print(f'account slot check: {len(refs)} listed, {len(active)} active', flush=True)
    dump(OUT/'account-active.json',dict(at=time.time(),active=active,total=len(active)))
    return active


def job_plan(manifest):
    jobs = []
    for arm in h.ARMS:
        for seed in (0, 1):
            start = 0
            for part, end in enumerate(manifest['boundaries'][arm], 1):
                tag = f'{arm}-s{seed}-p{part}'
                parent = f'{arm}-s{seed}-p{part-1}' if part > 1 else None
                jobs.append(dict(tag=tag, id='ezzzzzx/h34j-'+tag.replace('_','-'), arm=arm,
                    seed=seed, part=part, start=start, end=end, parent_tag=parent,
                    parent_id='ezzzzzx/h34j-'+parent.replace('_','-') if parent else None,
                    attempts=0, status='pending'))
                start = end
    return jobs


def prepare():
    smoke = json.loads((OUT/'smoke.json').read_text())
    assert smoke['ok'] and smoke['source_hashes'] == h.source_hashes()
    assert (smoke['n_bytes'], smoke['tail_start']) == (h.N_DATA, h.TAIL_START)
    manifest = dict(hashes=h.source_hashes(), documents=h.N_DATA, tail_start=h.TAIL_START,
        boundaries=smoke['boundaries'], raw_sha256=smoke['raw_sha256'], sha256_e10m=smoke['sha256_e10m'],
        orchestration_hashes={n:h.sha((BASE/n).read_bytes()) for n in
                              ('dispatch.py','kernel_template.py','summarize.py')})
    for arm in h.ARMS:
        edges = [0]+manifest['boundaries'][arm]
        assert edges[-1] == h.N_DATA and all(a < b for a,b in zip(edges,edges[1:]))
        assert all(x % 128 == 0 for x in edges[:-1])
    mp = BASE/'code-sha256.json'
    if STATE.exists():
        assert json.loads(mp.read_text()) == manifest, 'Changed source/config after dispatch'
    code = BASE/'code'; code.mkdir(parents=True, exist_ok=True)
    for name in manifest['hashes']:
        shutil.copyfile(ROOT/name, code/name)
    assert {p.name for p in code.iterdir()} <= set(manifest['hashes']) | {'dataset-metadata.json','smoke.json'}
    dump(code/'dataset-metadata.json', dict(id='ezzzzzx/pdp-code-h34j', title='pdp code h34j', licenses=[dict(name='other')]))
    dump(code/'smoke.json', smoke)
    fingerprint = h.sha(json.dumps(manifest,sort_keys=True).encode())
    for job in job_plan(manifest):
        folder = BASE/job['tag']; folder.mkdir(exist_ok=True)
        dump(folder/'kernel-metadata.json', dict(id=job['id'], title=job['id'].split('/')[-1],
            code_file='kernel.py', language='python', kernel_type='script', is_private=True,
            enable_gpu=False, enable_tpu=False, enable_internet=False,
            dataset_sources=['ezzzzzx/enwik8','ezzzzzx/pdp-code-h34j'], competition_sources=[],
            kernel_sources=[job['parent_id']] if job['parent_id'] else [], model_sources=[]))
        cfg = dict(job, hashes=manifest['hashes'], fingerprint=fingerprint,
                   kernel_id=job['id'], raw_sha256=manifest['raw_sha256'], sha256_e10m=manifest['sha256_e10m'])
        (folder/'kernel.py').write_text('CONFIG = '+repr(cfg)+'\n'+(BASE/'kernel_template.py').read_text())
    dump(mp,manifest)
    print(json.dumps(dict(prepared=len(job_plan(manifest)), lives=8, boundaries=manifest['boundaries'], fingerprint=fingerprint)), flush=True)
    return manifest, fingerprint


def fetch(job, manifest, fingerprint):
    from summarize import validate_part
    dest = OUT/'kernels'/job['tag']; dest.mkdir(parents=True, exist_ok=True)
    # Clear only this output directory so a failed download cannot reuse stale files.
    for name in ('results.json','checkpoint.pt'):
        (dest/name).unlink(missing_ok=True)
    cli('kernels','output',job['id'],'-p',str(dest))
    row = json.loads((dest/'results.json').read_text())
    validate_part(row, job, manifest, fingerprint, dest)
    if job['parent_tag']:
        parent = json.loads((OUT/'kernels'/job['parent_tag']/'results.json').read_text())
        assert row['resumed_checkpoint_sha256'] == parent['checkpoint_sha256']
        assert abs(row['wall_seconds']-row['segment_wall_seconds']-parent['wall_seconds']) < 1e-6


def dispatch(manifest, fingerprint):
    load_token()
    jobs = job_plan(manifest)
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(code_fingerprint=fingerprint,dataset_ready=False,jobs=jobs)
    assert state['code_fingerprint'] == fingerprint
    assert [j['id'] for j in state['jobs']] == [j['id'] for j in jobs]
    assert all(j['status'] in ('pending','submitting','active','complete','failed') and 0<=j['attempts']<=2 for j in state['jobs'])
    dump(STATE,state)
    if not state['dataset_ready']:
        dataset = 'ezzzzzx/pdp-code-h34j'
        rc, text = cli('datasets','status',dataset,allow_failure=True)
        missing = any(s in text.lower() for s in ('403','404','forbidden',"don't have access",'not found','does not exist'))
        if missing:
            cli('datasets','create','-p',str(BASE/'code'))  # private by default
        elif rc == 0 and any(s in text.lower() for s in ('ready','pending','processing')):
            cli('datasets','version','-p',str(BASE/'code'),'-m','H34j '+fingerprint[:16])
        else:
            raise RuntimeError('Cannot establish code dataset status: '+text)
        while True:
            rc, text = cli('datasets','status',dataset,allow_failure=True)
            if rc == 0 and 'ready' in text.lower(): break
            if any(s in text.lower() for s in ('403','404','forbidden',"don't have access",'not found')):
                time.sleep(30)
                continue
            if any(s in text.lower() for s in ('error','fail')): raise RuntimeError(text)
            time.sleep(30)
        state['dataset_ready'] = True; dump(STATE,state)
    while True:
        for job in state['jobs']:
            if job['status'] == 'complete':
                # Completed local artifacts were validated when fetched.
                continue
            if job['status'] not in ('active','submitting'): continue
            text = kernel_status_text(job['id'])
            kind = status_kind(text)
            if kind == 'complete':
                fetch(job,manifest,fingerprint); job['status'] = 'complete'
            elif kind == 'failed':
                failed = OUT/'failed'/f"{job['tag']}-attempt{job['attempts']}"
                failed.mkdir(parents=True,exist_ok=True)
                cli('kernels','output',job['id'],'-p',str(failed),allow_failure=True)
                job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
            else:
                job['status'] = 'active'
            dump(STATE,state)
        if all(j['status'] == 'complete' for j in state['jobs']): break
        for job in state['jobs']:
            if job['status'] != 'pending': continue
            if job['parent_tag'] and not any(j['tag']==job['parent_tag'] and j['status']=='complete' for j in state['jobs']): continue
            active = set(account_active()) | {j['id'] for j in state['jobs'] if j['status'] in ('active','submitting')}
            if job['id'] in active: raise RuntimeError('Already active outside dispatcher: '+job['id'])
            if len(active)>=5 or sum(ref.startswith('ezzzzzx/h34j-') for ref in active)>=4: break
            assert job['attempts'] < 2
            job.update(status='submitting', attempts=job['attempts']+1); dump(STATE,state)
            _, text = cli('kernels','push','-p',str(BASE/job['tag']))
            print(text.strip(),flush=True); job['status']='active'; dump(STATE,state)
        # Drain independent lives before reporting an exhausted retry.
        if not any(j['status'] in ('active','submitting') for j in state['jobs']):
            runnable = any(j['status']=='pending' and (not j['parent_tag'] or any(p['tag']==j['parent_tag'] and p['status']=='complete' for p in state['jobs'])) for j in state['jobs'])
            if not runnable:
                subprocess.run([sys.executable,str(BASE/'summarize.py')],cwd=ROOT,check=True)
                raise RuntimeError('Parts exhausted one retry; dependent parts remain unrun')
        time.sleep(30)
    subprocess.run([sys.executable,str(BASE/'summarize.py')],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--dispatch',action='store_true'); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,fingerprint=prepare()
        if args.dispatch and not args.prepare_only: dispatch(manifest,fingerprint)

if __name__ == '__main__': main()
