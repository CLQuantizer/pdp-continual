"""Foreground M11 scheduler: at most two active kernels, one retry per job.

Run with the repository venv after supplying Kaggle credentials in the environment.
Creates the code dataset privately; resumes from results/m11w6/dispatch.json.
Does not start detached processes or touch other experiments.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'kaggle/m11w6'
OUT = ROOT/'results/m11w6'
STATE = OUT/'dispatch.json'
CLI = ROOT/'.venv/bin/kaggle'


def cli(*args, check=True):
    p = subprocess.run([str(CLI), *args], cwd=ROOT, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    if check and p.returncode:
        # Do not include API responses or credential-bearing environment in errors.
        raise RuntimeError(f'Kaggle {args[0]} {args[1]} exit {p.returncode}')
    return p.returncode, p.stdout


def save(state):
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2)+'\n')
    tmp.replace(STATE)


def validate(job, dest):
    data = json.loads((dest/job['output']).read_text())
    assert data['seed'] == job['seed'] and data['kernel_id'] == job['kernel_id']
    assert data['workers'] == 4
    assert sorted(r['policy'] for r in data['runs']) == sorted(job['arms'])
    for row in data['runs']:
        assert row['seed'] == job['seed'] and row['L'] == 4096 and row['finite']
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--assemble-only',action='store_true')
    parser.add_argument('--dataset-ready',action='store_true',help='use an already uploaded matching private code dataset')
    args = parser.parse_args()
    smoke = json.loads((OUT/'smoke.json').read_text())
    assert smoke['ok']
    manifest = json.loads((BASE/'manifest.json').read_text())
    hashes = json.loads((BASE/'code-sha256.json').read_text())
    for name,digest in hashes.items():
        assert hashlib.sha256((BASE/'code'/name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    for job in manifest:
        meta = json.loads((BASE/job['folder']/'kernel-metadata.json').read_text())
        assert meta['id'] == job['kernel_id'] and meta['is_private'] is True
        assert meta['enable_gpu'] is False and meta['enable_internet'] is False
        assert meta['dataset_sources'] == ['ezzzzzx/pdp-code-m11w6']
    if args.check_only:
        print(f'checked {len(manifest)} kernel folders and {len(hashes)} code hashes',flush=True)
        return
    if args.assemble_only:
        subprocess.run([sys.executable,str(ROOT/'m11_w6.py'),'assemble'],cwd=ROOT,check=True)
        return
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(dataset_created=False,jobs=[dict(**j,status='pending',attempts=0) for j in manifest])
    assert [(j['kernel_id'],j['arms']) for j in state['jobs']] == [(j['kernel_id'],j['arms']) for j in manifest]
    if args.dataset_ready:
        state['dataset_created'] = True
    save(state)
    if not state['dataset_created']:
        # Kaggle datasets create defaults to PRIVATE (never pass --public).
        cli('datasets','create','-p',str(BASE/'code'))
        state['dataset_created'] = True
        save(state)
    while True:
        _, report = cli('datasets','status','ezzzzzx/pdp-code-m11w6')
        if 'ready' in report.lower():
            break
        if 'error' in report.lower() or 'fail' in report.lower():
            raise RuntimeError('Code dataset processing failed')
        print('code dataset processing',flush=True)
        time.sleep(30)
    while True:
        for job in state['jobs']:
            if job['status'] != 'active':
                continue
            _, report = cli('kernels','status',job['kernel_id'])
            status = report.lower()
            if 'complete' in status:
                dest = OUT/job['folder']
                dest.mkdir(parents=True,exist_ok=True)
                try:
                    cli('kernels','output',job['kernel_id'],'-p',str(dest),'--force')
                    validate(job,dest)
                except (RuntimeError,AssertionError,FileNotFoundError,json.JSONDecodeError):
                    job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
                else:
                    job['status'] = 'complete'
            elif any(s in status for s in ('error','cancel','fail')):
                dest = OUT/'failed'/f"{job['folder']}-attempt{job['attempts']}"
                dest.mkdir(parents=True,exist_ok=True)
                cli('kernels','output',job['kernel_id'],'-p',str(dest),check=False)
                job['status'] = 'pending' if job['attempts'] < 2 else 'failed'
            else:
                print(job['kernel_id'],'active',flush=True)
            save(state)
        if any(j['status']=='failed' for j in state['jobs']):
            raise RuntimeError('Kernel exhausted its single retry; state saved')
        active = sum(j['status']=='active' for j in state['jobs'])
        for job in state['jobs']:
            if active >= 2:
                break
            if job['status'] != 'pending':
                continue
            # Persist before push: on a network ambiguity, a resumed scheduler
            # polls this id instead of submitting an uncounted duplicate.
            job.update(status='active',attempts=job['attempts']+1)
            save(state)
            cli('kernels','push','-p',str(BASE/job['folder']))
            active += 1
            print(job['kernel_id'],f"attempt {job['attempts']}",flush=True)
        if all(j['status']=='complete' for j in state['jobs']):
            break
        time.sleep(30)
    subprocess.run([sys.executable,str(ROOT/'m11_w6.py'),'assemble'],cwd=ROOT,check=True)

if __name__ == '__main__':
    main()
