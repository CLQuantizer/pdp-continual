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
    spec=importlib.util.spec_from_file_location('h34g_'+name,BASE/(name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    d=load('dispatch'); s=load('summarize'); checks=[]
    for split,n in ((False,8),(True,10)):
        jobs=d.job_plan(dict(split_zgate_8=split))
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
        assert m['dataset_sources']==['ezzzzzx/enwik8','ezzzzzx/pdp-code-h34g']
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
    with tempfile.TemporaryDirectory(prefix='h34g-') as tmp:
        tmp=Path(tmp); d.OUT=tmp
        pages=[]
        def cli(*args,**kwargs):
            if args[:2]==('kernels','list'):
                pages.append(args)
                return (0,'ref\nezzzzzx/w9d-test\nNext Page Token = second') if '--page-token' not in args else (0,'ref\nezzzzzx/h34g-external\n')
            raise AssertionError(args)
        d.cli=cli
        d.kernel_status_text=lambda ref:'Kernel '+ref+' has status "RUNNING"'
        assert set(d.account_active())=={'ezzzzzx/w9d-test','ezzzzzx/h34g-external'}
        assert len(pages)==2 and all('--sort-by' in a and 'dateRun' in a for a in pages)
        d.kernel_status_text=lambda ref:'403 forbidden'
        try: d.account_active()
        except RuntimeError: pass
        else: raise AssertionError('Unreadable kernel accepted')
    checks.append(dict(check='account_pagination_and_fail_closed',pages=2,active=2))
    # A pre-existing run of the same ID must never be overwritten by a push.
    d=load('dispatch')
    with tempfile.TemporaryDirectory(prefix='h34g-duplicate-') as tmp:
        tmp=Path(tmp); d.OUT=tmp; d.STATE=tmp/'state.json'
        manifest=dict(documents=180000,missing=[],hashes=d.source_hashes(),split_zgate_8=True)
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
        with tempfile.TemporaryDirectory(prefix='h34g-') as tmp:
            tmp=Path(tmp); d.OUT=tmp; d.STATE=tmp/'state.json'
            manifest=dict(documents=180000,missing=[],hashes=d.source_hashes(),split_zgate_8=True)
            d.dump(tmp/'smoke.json',dict(ok=True,n_bytes=180000,source_hashes=manifest['hashes']))
            jobs=d.job_plan(manifest)
            d.dump(d.STATE,dict(code_fingerprint='offline',dataset_ready=True,jobs=jobs))
            d.load_token=lambda:None
            remote=set(); completed=set(); pushes=[]; max_active=0; max_h34g=0
            def account(): return ['ezzzzzx/w9d-'+str(i) for i in range(external)]+list(remote)
            d.account_active=account
            def cli(*args,**kwargs):
                nonlocal max_active,max_h34g
                assert args[:2]==('kernels','push')
                tag=Path(args[-1]).name
                job=next(j for j in jobs if j['tag']==tag)
                if job['part']==2: assert job['parent_id'] in completed
                remote.add(job['id']); pushes.append(job['id'])
                max_h34g=max(max_h34g,len(remote)); max_active=max(max_active,len(remote)+external)
                assert len(remote)<=2 and len(remote)+external<=5
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
                               pushes=len(pushes),max_account_active=max_active,max_h34g_active=max_h34g))
    # Missing runs must remain unevaluated; baseline curve values come from H34f.
    with tempfile.TemporaryDirectory(prefix='h34g-') as tmp:
        s.assemble(tmp)
        report=json.loads((Path(tmp)/'results.json').read_text())
        assert len(report['missing'])==8 and not report['runs']
        assert all(r['evaluated']==0 and r['mean'] is None for r in report['scoring'])
        assert all(r['mean'] is None for r in report['depth_curve'] if r['arm'] in ('zgate_2','zgate_6','zgate_8'))
        assert all(r['mean'] is not None for r in report['depth_curve'] if r['arm'] in ('plain_1','zgate_4'))
    checks.append(dict(check='missing_runs_no_fabricated_scores',missing=8,evaluated=0))
    # Synthetic fixtures live only in a temporary directory; check score signs and seed counts.
    manifest=json.loads((BASE/'code-sha256.json').read_text())
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    baseline=json.loads((ROOT/'results/h34f/results.json').read_text())
    refs={(r['arm'],r['seed']):r for r in baseline['runs']}
    with tempfile.TemporaryDirectory(prefix='h34g-scoring-fixture-') as tmp:
        tmp=Path(tmp)
        d.dump(tmp/'smoke.json',dict(ok=True,n_bytes=180000,source_hashes=d.source_hashes(),
                                   sha256_e10m=baseline['smoke']['sha256_e10m']))
        for arm in s.ARMS:
            for seed in (0,1):
                r=copy.deepcopy(refs['zgate_6',seed]); depth=int(arm.split('_')[1])
                split=arm=='zgate_8' and manifest['split_zgate_8']
                tag=f'{arm}-s{seed}'; ref=f'ezzzzzx/h34g-{arm.replace("_","-")}-s{seed}'
                delta={'zgate_8':(-.021,-.019),'dense_6':(-.031,-.029),'zgate_2':(.04,.05),'zgate_6':(0.,0.)}[arm][seed]
                for key in ('bits_per_byte','tail_bits_per_byte','first_40kb_bits_per_byte'): r[key]+=delta
                for row in r['slices']: row['bits_per_byte']+=delta
                r['layers']=(r['layers']+[copy.deepcopy(r['layers'][-1]) for _ in range(2)])[:depth]
                for row in r['mixer_slices']: row['mean_weights']=(row['mean_weights']+[0.,0.])[:depth]
                r.update(arm=arm,initial_mixer_weights=s.initial_gate(arm)[0].tolist(),
                    kernel_id=ref+('-part2' if split else ''),code_fingerprint=fingerprint,
                    segment_wall_seconds=1.,wall_seconds=2. if split else 1.)
                if split:
                    parent=tmp/'kernels'/tag; parent.mkdir(parents=True)
                    checkpoint=parent/'checkpoint.pt'; checkpoint.write_bytes(b'SYNTHETIC OFFLINE TEST ONLY')
                    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    d.dump(parent/'results.json',dict(arm=arm,seed=seed,n_bytes=180000,kernel_id=ref,
                        code_fingerprint=fingerprint,sha256_e10m=r['sha256_e10m'],bytes_seen=90000,
                        offset=89984,pending_bytes=16,segment_wall_seconds=1.,wall_seconds=1.,checkpoint_sha256=digest))
                    r['resumed_checkpoint_sha256']=digest
                dest=tmp/'kernels'/(tag+'-part2' if split else tag); dest.mkdir(parents=True,exist_ok=True)
                d.dump(dest/'results.json',r)
        s.assemble(tmp)
        report=json.loads((tmp/'results.json').read_text())
        assert not report['missing'] and len(report['runs'])==8
        tails=[r for r in report['scoring'] if r['range']=='Tail']
        assert [r['threshold'] for r in tails]==[-.02,-.03]
        assert all(r['seeds_meeting']==1 and r['evaluated']==2 for r in tails)
        assert all(r['threshold'] is None for r in report['scoring'] if r['range']=='Full')
        path=tmp/'kernels/zgate_6-s0/results.json'; bad=json.loads(path.read_text())
        bad['code_fingerprint']='wrong'; d.dump(path,bad)
        try: s.assemble(tmp)
        except AssertionError: pass
        else: raise AssertionError('Mixed source fingerprint accepted')
    checks.append(dict(check='synthetic_scoring_and_fingerprint_rejection',runs=8,tail_counts=[1,1],thresholds=[-.02,-.03]))
    # Execute the real kernel template against a local mock Kaggle mount.
    with tempfile.TemporaryDirectory(prefix='h34g-kernel-fixture-') as tmp:
        tmp=Path(tmp); inp=tmp/'input'; work=tmp/'working'; inp.mkdir(); work.mkdir()
        code=inp/'code'; code.mkdir()
        for name in manifest['hashes']: (code/name).symlink_to(ROOT/name)
        (inp/'enwik8').symlink_to(ROOT/'data/enwik8/enwik8')
        template=(BASE/'kernel_template.py').read_text().replace('/kaggle/input',str(inp)).replace('/kaggle/working',str(work))
        template=template.replace('stop_at=90000','stop_at=128')
        config=dict(arm='zgate_8',seed=0,part=1,documents=256,hashes=d.source_hashes(),
                    fingerprint='offline-template',kernel_id='ezzzzzx/h34g-zgate-8-s0',parent_id=None)
        script=tmp/'kernel.py'
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        script.write_text('CONFIG = '+repr(config)+'\n'+template)
        subprocess.run([sys.executable,str(script)],check=True,env=env)
        first=json.loads((work/'results.json').read_text())
        assert first['bytes_seen']==first['offset']==128
        parent=inp/'parent'; work.rename(parent); work.mkdir()
        config.update(part=2,kernel_id='ezzzzzx/h34g-zgate-8-s0-part2',parent_id='ezzzzzx/h34g-zgate-8-s0')
        script.write_text('CONFIG = '+repr(config)+'\n'+template)
        subprocess.run([sys.executable,str(script)],check=True,env=env)
        resumed=json.loads((work/'results.json').read_text())
        assert resumed['resumed_checkpoint_sha256']==first['checkpoint_sha256']
        import h34g_depth as h
        h.h33.configure(1)
        raw,_=h.h33.load_checked(str(ROOT/'data/enwik8/enwik8'))
        whole=h.run('zgate_8',0,raw[:256])
        assert whole['bits_per_byte']==resumed['bits_per_byte']
        assert whole['layers']==resumed['layers'] and whole['mixer_slices']==resumed['mixer_slices']
    checks.append(dict(check='kernel_template_execution',n_bytes=256,split_at=128,max_bits_abs=0.))
    d.dump(ROOT/'results/h34g/tool-checks.json',dict(ok=True,checks=checks))
    print(json.dumps(dict(ok=True,checks=checks)))


if __name__=='__main__': main()
