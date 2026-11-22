"""Offline orchestration checks; no Kaggle calls, credentials or scored outputs."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import os
import sys

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location('h34h_'+name,BASE/(name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    d=load('dispatch'); s=load('summarize'); checks=[]
    for split,n in ((False,8),(True,10)):
        jobs=d.job_plan(dict(split_dense_8=split))
        assert len(jobs)==n and len({j['id'] for j in jobs})==n
        assert sum(j['part']==2 for j in jobs)==(2 if split else 0)
        for j in jobs:
            if j['part']==2:
                assert any(p['id']==j['parent_id'] and p['tag']==j['parent_tag'] and p['part']==1 for p in jobs)
    checks.append(dict(check='job_plans',unsplit=8,split=10,continuation_dependencies=2))
    for name in ('dispatch.py','kernel_template.py','summarize.py'):
        ast.parse((BASE/name).read_text())
    for p in BASE.glob('*/kernel.py'):
        ast.parse(p.read_text())
        m=json.loads((p.parent/'kernel-metadata.json').read_text())
        assert m['is_private'] and not m['enable_gpu'] and not m['enable_internet']
        assert m['dataset_sources']==['ezzzzzx/enwik8','ezzzzzx/pdp-code-h34h']
        assert len(m['kernel_sources'])==(1 if m['id'].endswith('-part2') else 0)
    checks.append(dict(check='generated_kernel_syntax_and_metadata',kernels=len(list(BASE.glob('*/kernel.py')))))
    assert d.status_kind('Kernel x has status "COMPLETE"')=='complete'
    assert d.status_kind('Kernel x has status "RUNNING"')=='active'
    assert d.status_kind('Kernel x has status "ERROR"')=='failed'
    try: d.status_kind('ambiguous')
    except RuntimeError: pass
    else: raise AssertionError('Unknown status accepted')
    checks.append(dict(check='status_classification',cases=4))
    # Exhaustive account pagination and unknown-status refusal.
    with tempfile.TemporaryDirectory(prefix='h34h-') as tmp:
        tmp=Path(tmp); d.OUT=tmp
        pages=[]
        def cli(*args,**kwargs):
            if args[:2]==('kernels','list'):
                pages.append(args)
                return (0,'ref\nezzzzzx/w9d-test\nNext Page Token = second') if '--page-token' not in args else (0,'ref\nezzzzzx/h34h-external\n')
            raise AssertionError(args)
        d.cli=cli
        d.kernel_status_text=lambda ref:'Kernel '+ref+' has status "RUNNING"'
        assert set(d.account_active())=={'ezzzzzx/w9d-test','ezzzzzx/h34h-external'}
        assert len(pages)==2 and all('--sort-by' in a and 'dateRun' in a for a in pages)
        d.kernel_status_text=lambda ref:'403 forbidden'
        try: d.account_active()
        except RuntimeError: pass
        else: raise AssertionError('Unreadable kernel accepted')
    checks.append(dict(check='account_pagination_and_fail_closed',pages=2,active=2))
    # A pre-existing run of the same ID must never be overwritten by a push.
    d=load('dispatch')
    with tempfile.TemporaryDirectory(prefix='h34h-duplicate-') as tmp:
        tmp=Path(tmp); d.OUT=tmp; d.STATE=tmp/'state.json'
        manifest=dict(documents=180000,missing=[],hashes=d.source_hashes(),split_dense_8=True)
        d.dump(tmp/'smoke.json',dict(ok=True,n_bytes=180000,source_hashes=manifest['hashes']))
        jobs=d.job_plan(manifest)
        d.dump(d.STATE,dict(code_fingerprint='offline',dataset_ready=True,jobs=jobs))
        d.load_token=lambda:None
        d.account_active=lambda:[jobs[0]['id']]
        def forbidden_push(*args,**kwargs): raise AssertionError('Duplicate push attempted')
        d.cli=forbidden_push
        try: d.dispatch(manifest,'offline')
        except RuntimeError as exc: assert 'already active outside' in str(exc)
        else: raise AssertionError('External duplicate ID accepted')
    checks.append(dict(check='external_duplicate_kernel_refused',pushes=0))
    # Drive the real dispatcher loop with an offline remote and one unrelated W9d.
    for external in (1,4):
        d=load('dispatch')
        with tempfile.TemporaryDirectory(prefix='h34h-') as tmp:
            tmp=Path(tmp); d.OUT=tmp; d.STATE=tmp/'state.json'
            manifest=dict(documents=180000,missing=[],hashes=d.source_hashes(),split_dense_8=True)
            d.dump(tmp/'smoke.json',dict(ok=True,n_bytes=180000,source_hashes=manifest['hashes']))
            jobs=d.job_plan(manifest)
            d.dump(d.STATE,dict(code_fingerprint='offline',dataset_ready=True,jobs=jobs))
            d.load_token=lambda:None
            remote=set(); completed=set(); pushes=[]; max_active=0; max_h34h=0
            def account(): return ['ezzzzzx/w9d-'+str(i) for i in range(external)]+list(remote)
            d.account_active=account
            def cli(*args,**kwargs):
                nonlocal max_active,max_h34h
                assert args[:2]==('kernels','push')
                tag=Path(args[-1]).name
                job=next(j for j in jobs if j['tag']==tag)
                if job['part']==2: assert job['parent_id'] in completed
                remote.add(job['id']); pushes.append(job['id'])
                max_h34h=max(max_h34h,len(remote)); max_active=max(max_active,len(remote)+external)
                assert len(remote)<=3 and len(remote)+external<=5
                return 0,'offline pushed'
            d.cli=cli
            def status(ref):
                remote.discard(ref); completed.add(ref)
                return 'Kernel '+ref+' has status "COMPLETE"'
            d.kernel_status_text=status
            d.fetch=lambda *args:None
            old_sleep=d.time.sleep; old_run=d.subprocess.run
            d.time.sleep=lambda seconds:None
            d.subprocess.run=lambda *a,**kw:None
            try: d.dispatch(manifest,'offline')
            finally: d.time.sleep=old_sleep; d.subprocess.run=old_run
            assert len(pushes)==10 and len(set(pushes))==10
            state=json.loads(d.STATE.read_text()); assert all(j['status']=='complete' for j in state['jobs'])
            # A second invocation must not repush anything.
            old_run=d.subprocess.run; d.subprocess.run=lambda *a,**kw:None
            try: d.dispatch(manifest,'offline')
            finally: d.subprocess.run=old_run
            assert len(pushes)==10
            checks.append(dict(check='dispatcher_caps_dependencies_resume',unrelated_active=external,
                               pushes=len(pushes),max_account_active=max_active,max_h34h_active=max_h34h))
    d.dump(ROOT/'results/h34h/tool-checks.json',dict(ok=True,checks=checks))
    print(json.dumps(dict(ok=True,checks=checks)))

if __name__=='__main__': main()
