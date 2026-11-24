"""Prepare offline, or resume a foreground dispatcher. Never launches on import.

Account-wide <=5 active (running/queued/pending); H34i <=5, including jobs
launched outside this dispatcher. At most two push attempts per job. A
process lock serializes H34i dispatchers. Other experiment dispatchers must
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
from h34i_generality import ARMS, STREAMS, N_DATA, dump, source_hashes, prepare_data
BASE, OUT = ROOT/'kaggle/h34i', ROOT/'results/h34i'
STATE = OUT/'dispatch.json'
CLI = str(ROOT/'.venv/bin/kaggle')
FILES = ('h34i_generality.py','h34h_dense.py','h34g_depth.py','h34f_zero_gate.py','h34b_plain_depth.py','h34a2_logit_sum.py','h34a_local_depth.py','h26_cued.py','h32_mixer.py',
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
    if not smoke.get('ok') or smoke.get('source_hashes') != source_hashes():
        raise RuntimeError('Successful current-source smoke required')
    streams = prepare_data()
    assert streams == smoke['streams']
    code = BASE/'code'; code.mkdir(parents=True,exist_ok=True)
    hashes = source_hashes()
    assert set(hashes) == set(FILES)
    manifest = dict(hashes=hashes, missing=[], documents=N_DATA,
        split_arms=smoke['split_arms'], streams=streams,
        orchestration_hashes={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest()
                              for n in ('dispatch.py','kernel_template.py','summarize.py')})
    manifest_path = BASE/'code-sha256.json'
    if STATE.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Source/configuration changed after dispatch began; refusing mixed versions')
    for name in hashes:
        shutil.copyfile(ROOT/name,code/name)
    dump(code/'dataset-metadata.json',dict(id='ezzzzzx/pdp-code-h34i',title='pdp code h34i',licenses=[dict(name='other')]))
    dump(ROOT/'data/h34i/dataset-metadata.json',dict(id='ezzzzzx/pdp-data-h34i',title='pdp data h34i',licenses=[dict(name='other')]))
    dump(manifest_path,manifest)
    fingerprint = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    jobs = job_plan(manifest)
    planned = {j['tag'] for j in jobs}
    for folder in BASE.glob('*-s*-part2'):
        if folder.name not in planned:
            shutil.rmtree(folder)
    for job in jobs:
        folder=BASE/job['tag']; folder.mkdir(exist_ok=True)
        dump(folder/'kernel-metadata.json',dict(id=job['id'],title=job['id'].split('/')[-1],code_file='kernel.py',
            language='python',kernel_type='script',is_private=True,enable_gpu=False,enable_tpu=False,
            enable_internet=False,dataset_sources=['ezzzzzx/pdp-data-h34i','ezzzzzx/pdp-code-h34i'],
            competition_sources=[],kernel_sources=[job['parent_id']] if job['part']==2 else [],model_sources=[]))
        config=dict(arm=job['arm'],stream=job['stream'],seed=job['seed'],part=job['part'],documents=N_DATA,
                    hashes=hashes,streams=streams,fingerprint=fingerprint,kernel_id=job['id'],parent_id=job.get('parent_id'))
        (folder/'kernel.py').write_text('CONFIG = '+repr(config)+'\n'+(BASE/'kernel_template.py').read_text())
    print(json.dumps(dict(prepared=len(jobs),lives=30,documents=N_DATA,split_arms=manifest['split_arms'],
                         code_fingerprint=fingerprint)),flush=True)
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


def fetch(job, fingerprint, documents):
    dest = OUT/'kernels'/job['tag']; dest.mkdir(parents=True,exist_ok=True)
    cli('kernels','output',job['id'],'-p',str(dest))
    row = json.loads((dest/'results.json').read_text())
    smoke = json.loads((OUT/'smoke.json').read_text())
    assert row['kernel_id'] == job['id'] and row['code_fingerprint'] == fingerprint
    assert (row['stream'],row['arm'],row['seed'],row['n_bytes']) == (job['stream'],job['arm'],job['seed'],documents)
    assert row['raw_sha256'] == smoke['streams'][job['stream']]['sha256']
    if job['part']==1:
        assert (row['bytes_seen'],row['offset'],row['pending_bytes']) == (90000,89984,16)
        assert 0 < row['segment_wall_seconds'] <= 5100
        assert hashlib.sha256((dest/'checkpoint.pt').read_bytes()).hexdigest()==row['checkpoint_sha256']
    else:
        from summarize import validate
        manifest=json.loads((BASE/'code-sha256.json').read_text())
        validate(row,job['stream'],job['arm'],job['seed'],smoke,manifest,fingerprint)
        if job['part']==2:
            parent=json.loads((OUT/'kernels'/job['parent_tag']/'results.json').read_text())
            assert row['resumed_checkpoint_sha256']==parent['checkpoint_sha256']


def job_plan(manifest):
    jobs=[]
    for stream in STREAMS:
        for arm in ARMS:
            for seed in (0,1):
                tag=f'{stream}-{arm}-s{seed}'; ref=f'ezzzzzx/h34i-{stream}-{arm.replace("_","-")}-s{seed}'
                split=arm in manifest['split_arms']
                jobs.append(dict(tag=tag,id=ref,stream=stream,arm=arm,seed=seed,part=1 if split else 0,status='pending',attempts=0))
                if split:
                    jobs.append(dict(tag=tag+'-part2',id=ref+'-part2',stream=stream,arm=arm,seed=seed,part=2,
                                     parent_tag=tag,parent_id=ref,status='pending',attempts=0))
    return jobs


def dispatch(manifest, fingerprint):
    smoke = json.loads((OUT/'smoke.json').read_text())
    if not smoke['ok']:
        raise RuntimeError('Smoke gate failed; no kernels may launch')
    if manifest['missing']:
        raise RuntimeError('Missing source files: '+', '.join(manifest['missing']))
    assert smoke['n_bytes'] == manifest['documents']
    assert smoke['source_hashes'] == manifest['hashes']
    load_token()
    jobs = job_plan(manifest)
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(code_fingerprint=fingerprint,datasets_ready=[],jobs=jobs)
    assert state['code_fingerprint'] == fingerprint
    assert [(j['tag'],j['id']) for j in state['jobs']] == [(j['tag'],j['id']) for j in jobs]
    assert all(j['status'] in ('pending','submitting','active','complete','failed') and 0 <= j['attempts'] <= 2 for j in state['jobs'])
    dump(STATE,state)
    for dataset, folder in (('ezzzzzx/pdp-code-h34i', BASE/'code'), ('ezzzzzx/pdp-data-h34i', ROOT/'data/h34i')):
        if dataset in state['datasets_ready']:
            continue
        code, text = cli('datasets','status',dataset,allow_failure=True)
        low = text.lower()
        # A private dataset that does not exist yet returns 403 Client Error, not 404.
        # That body contains "error" and is not a processing failure.
        missing = any(s in low for s in ('403','404','forbidden',"don't have access",'not found','does not exist'))
        exists = (not missing) and any(s in low for s in ('ready','pending','processing'))
        if not missing and not exists:
            raise RuntimeError('Cannot establish dataset existence: '+text)
        if exists:
            cli('datasets','version','-p',str(folder),'-m','H34i source '+fingerprint[:16])
        else:
            code, text = cli('datasets','create','-p',str(folder),allow_failure=True)  # Private by default.
            print(text.strip(), flush=True)
            if code:
                created = text.lower()
                if any(s in created for s in ('409','already exists','conflict')):
                    cli('datasets','version','-p',str(folder),'-m','H34i source '+fingerprint[:16])
                else:
                    raise RuntimeError('Dataset create failed: '+text)
        while True:
            code, text = cli('datasets','status',dataset,allow_failure=True)
            low = text.lower()
            print(text.strip(), flush=True)
            if code == 0 and 'ready' in low:
                break
            if any(s in low for s in ('403','404','forbidden',"don't have access",'not found')):
                time.sleep(30)
                continue
            if any(s in low for s in ('error','fail')):
                raise RuntimeError(text)
            time.sleep(30)
        state['datasets_ready'].append(dataset)
        dump(STATE,state)
    while True:
        for job in state['jobs']:
            if job['status'] not in ('active','submitting'):
                continue
            text = kernel_status_text(job['id'])
            kind = status_kind(text)
            if job['status'] == 'submitting' and kind != 'active':
                # A completed run's fingerprint proves whether an interrupted
                # push succeeded; failed/older ambiguous pushes need inspection.
                if kind not in ('complete','failed'):
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
        # Already at the H34i cap. Polling those kernels above is enough;
        # another full account scan would only add status calls.
        local_h34i = sum(j['status'] == 'active' and j['id'].startswith('ezzzzzx/h34i-') for j in state['jobs'])
        if local_h34i >= 5:
            time.sleep(30)
            continue
        for job in state['jobs']:
            if job['status'] != 'pending':
                continue
            if job['part']==2 and not any(j['tag']==job['parent_tag'] and j['status']=='complete' for j in state['jobs']):
                continue
            # Recheck the entire account immediately before EACH push.
            active = account_active()
            local = {j['id'] for j in state['jobs'] if j['status'] == 'active'}
            active = set(active) | local  # Protect against listing latency.
            if job['id'] in active:
                raise RuntimeError('Kernel already active outside dispatcher state: '+job['id'])
            if len(active) >= 5 or sum(ref.startswith('ezzzzzx/h34i-') for ref in active) >= 5:
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
    p.add_argument('--dispatch',action='store_true',help='Enable network and foreground dispatch after offline smoke')
    args = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,fingerprint = prepare()
        if args.dispatch and not args.prepare_only:
            dispatch(manifest,fingerprint)

if __name__ == '__main__':
    main()
