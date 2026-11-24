"""Exercise orchestration offline. All simulated scores/state stay in /tmp."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT))
import h34i_generality as h


def load(name):
    spec=importlib.util.spec_from_file_location('h34i_test_'+name,BASE/(name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture_manifest(split_arms=()):
    return dict(documents=h.N_DATA,missing=[],hashes=h.source_hashes(),split_arms=list(split_arms),
                streams=json.loads((ROOT/'data/h34i/manifest.json').read_text()))


def check_dispatch(external, split_arms, failures=0, interrupted=False, datasets_ready=True, existing_code=False):
    d=load('dispatch'); manifest=fixture_manifest(split_arms)
    with tempfile.TemporaryDirectory(prefix='h34i-dispatch-') as folder:
        tmp=Path(folder); d.OUT=tmp; d.STATE=tmp/'state.json'
        smoke=dict(ok=True,n_bytes=h.N_DATA,source_hashes=manifest['hashes'])
        h.dump(tmp/'smoke.json',smoke)
        jobs=d.job_plan(manifest)
        state=dict(code_fingerprint='offline',datasets_ready=['ezzzzzx/pdp-code-h34i','ezzzzzx/pdp-data-h34i'] if datasets_ready else [],jobs=jobs)
        remote=set(); completed=set(); pushes=[]; attempts={}; maxima=[0,0]; dataset_calls=[]
        if interrupted:
            jobs[0].update(status='submitting',attempts=1)
            remote.add(jobs[0]['id']); attempts[jobs[0]['id']]=1
        h.dump(d.STATE,state)
        d.load_token=lambda:None
        d.account_active=lambda:['ezzzzzx/other-'+str(i) for i in range(external)]+list(remote)
        def cli(*args,**kwargs):
            if args[:2]==('datasets','status'):
                dataset_calls.append(args)
                exists=existing_code and args[2]=='ezzzzzx/pdp-code-h34i'
                return (0,'ready') if exists or any(c[:2] in (('datasets','create'),('datasets','version')) and args[2] in c for c in dataset_calls) else (1,'404 not found')
            if args[:2] in (('datasets','create'),('datasets','version')):
                assert '--public' not in args
                dataset='ezzzzzx/pdp-code-h34i' if Path(args[3]).name=='code' else 'ezzzzzx/pdp-data-h34i'
                dataset_calls.append((*args,dataset)); return 0,'private dataset ready'
            if args[:2]==('kernels','output'):
                return 0,'offline failed log'
            assert args[:2]==('kernels','push'),args
            tag=Path(args[-1]).name; job=next(j for j in jobs if j['tag']==tag)
            if job['part']==2:
                assert job['parent_id'] in completed
            remote.add(job['id']); pushes.append(job['id'])
            attempts[job['id']]=attempts.get(job['id'],0)+1
            maxima[0]=max(maxima[0],len(remote)+external); maxima[1]=max(maxima[1],len(remote))
            assert len(remote)<=5 and len(remote)+external<=5
            return 0,'offline pushed'
        d.cli=cli
        def status(ref):
            if ref==jobs[0]['id'] and attempts[ref]<=failures:
                remote.discard(ref)
                return 'Kernel '+ref+' has status "ERROR"'
            remote.discard(ref); completed.add(ref)
            return 'Kernel '+ref+' has status "COMPLETE"'
        d.kernel_status_text=status
        d.fetch=lambda *args:None
        with patch.object(d.time,'sleep',lambda seconds:None),patch.object(d.subprocess,'run',lambda *a,**kw:None):
            if failures==2:
                try:
                    d.dispatch(manifest,'offline')
                except RuntimeError as exc:
                    assert 'exhausted its one retry' in str(exc)
                else:
                    raise AssertionError('Third attempt allowed')
                final=json.loads(d.STATE.read_text())
                assert final['jobs'][0]['status']=='failed' and final['jobs'][0]['attempts']==2
            else:
                d.dispatch(manifest,'offline')
                final=json.loads(d.STATE.read_text())
                assert all(j['status']=='complete' for j in final['jobs'])
                before=len(pushes)
                d.dispatch(manifest,'offline')
                assert before==len(pushes)
                assert len(pushes)==len(jobs)+failures-int(interrupted)
        assert max(attempts.values())<=2
        if not datasets_ready:
            assert len([c for c in dataset_calls if c[:2]==('datasets','create')])==2-int(existing_code)
            assert len([c for c in dataset_calls if c[:2]==('datasets','version')])==int(existing_code)
        return dict(check='dispatch_caps_retry_resume',unrelated_active=external,split_arms=list(split_arms),
                    failures=failures,interrupted=interrupted,pushes=len(pushes),max_account_active=maxima[0],
                    max_h34i_active=maxima[1],datasets_created=0 if datasets_ready else 2-int(existing_code),
                    datasets_versioned=int(existing_code))


def check_report():
    s=load('summarize'); manifest=fixture_manifest(('dense_6',)); checks=[]
    with tempfile.TemporaryDirectory(prefix='h34i-report-') as folder:
        tmp=Path(folder); s.BASE=tmp/'base'; s.BASE.mkdir()
        h.dump(s.BASE/'code-sha256.json',manifest)
        fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        s.assemble(tmp)
        report=json.loads((tmp/'results.json').read_text())
        assert len(report['missing'])==30 and len(report['runs'])==0
        assert all(r['seeds_meeting']==r['evaluated']==r['streams_with_both_seeds']==0 for r in report['summary'])
        checks.append(dict(check='missing_report',missing=30,passes=0,evaluated=0))
        smoke=dict(ok=True,n_bytes=h.N_DATA,source_hashes=h.source_hashes(),streams=manifest['streams'])
        h.dump(tmp/'smoke.json',smoke)
        baseline=json.loads((ROOT/'results/h34g/results.json').read_text())
        ref=next(r for r in baseline['runs'] if r['arm']=='dense_6' and r['seed']==0)
        for stream in h.STREAMS:
            for arm in h.ARMS:
                for seed in (0,1):
                    r=copy.deepcopy(ref); depth=h.config(arm)['depth']
                    # Known signs: Own s0 passes, s1 misses; Wait both pass; See only s1 passes.
                    delta={'plain_1':0.,'plain_6':(-.06,-.04)[seed],
                           'zgate_6':(-.17,-.15)[seed],'dense_6':(-.18,-.18)[seed],
                           'dense_6_ones':(.02,.02)[seed]}[arm]
                    for key in ('bits_per_byte','tail_bits_per_byte','first_40kb_bits_per_byte'):
                        r[key]+=delta
                    for sl in r['slices']:
                        sl['bits_per_byte']+=delta
                    r['layers']=r['layers'][:depth]
                    for sl in r['mixer_slices']:
                        sl['mean_weights']=sl['mean_weights'][:depth]
                    split=arm in manifest['split_arms']; tag=f'{stream}-{arm}-s{seed}'
                    kernel=f'ezzzzzx/h34i-{stream}-{arm.replace("_","-")}-s{seed}'
                    r.update(stream=stream,arm=arm,seed=seed,raw_sha256=manifest['streams'][stream]['sha256'],
                             initial_mixer_weights=h.initial_gate(arm)[0].tolist(),kernel_id=kernel+('-part2' if split else ''),
                             code_fingerprint=fingerprint,segment_wall_seconds=1.,wall_seconds=2. if split else 1.,part=2 if split else 0)
                    if split:
                        parent=tmp/'kernels'/tag; parent.mkdir(parents=True)
                        checkpoint=parent/'checkpoint.pt'; checkpoint.write_bytes(b'SYNTHETIC CHECKPOINT FOR OFFLINE TEST')
                        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                        p={k:r[k] for k in ('stream','arm','seed','n_bytes','code_fingerprint','raw_sha256')}
                        p.update(kernel_id=kernel,part=1,bytes_seen=90000,offset=89984,pending_bytes=16,
                                 segment_wall_seconds=1.,wall_seconds=1.,checkpoint_sha256=digest)
                        h.dump(parent/'results.json',p); r['resumed_checkpoint_sha256']=digest
                    dest=tmp/'kernels'/(tag+'-part2' if split else tag); dest.mkdir(parents=True,exist_ok=True)
                    h.dump(dest/'results.json',r)
        s.assemble(tmp); report=json.loads((tmp/'results.json').read_text())
        assert not report['missing'] and len(report['runs'])==30
        assert [r['threshold'] for r in report['summary']]==[-.05,-.10,-.02]
        assert [r['seeds_meeting'] for r in report['summary']]==[3,6,3]
        assert [r['streams_with_both_seeds'] for r in report['summary']]==[0,3,0]
        assert all(r['threshold']==s.BARS[i%3][4] for i,r in enumerate(report['scoring']))
        assert len(report['report_only'])==8
        assert all(r['tail_gap'] is not None and 'threshold' not in r for r in report['report_only'])
        first=tmp/'kernels/c4-plain_1-s0/results.json'; good=json.loads(first.read_text())
        for key,value in (('code_fingerprint','bad'),('raw_sha256','bad'),('stream','bible'),('seed',1)):
            bad=copy.deepcopy(good); bad[key]=value; h.dump(first,bad)
            try:
                s.assemble(tmp)
            except AssertionError:
                pass
            else:
                raise AssertionError('Invalid run accepted: '+key)
        h.dump(first,good)
        checks.append(dict(check='report_signs_counts_identity_and_lineage',lives=30,counts=[3,6,3],streams_with_both=[0,3,0],rejected=4))
    return checks


def check_prepare():
    d=load('dispatch'); manifest=fixture_manifest(('plain_6','dense_6'))
    with tempfile.TemporaryDirectory(prefix='h34i-prepare-') as folder:
        tmp=Path(folder); d.ROOT=tmp; d.BASE=tmp/'kaggle/h34i'; d.OUT=tmp/'results/h34i'; d.STATE=d.OUT/'dispatch.json'
        d.BASE.mkdir(parents=True); d.OUT.mkdir(parents=True)
        for name in d.FILES:
            shutil.copyfile(ROOT/name,tmp/name)
        for name in ('dispatch.py','kernel_template.py','summarize.py'):
            shutil.copyfile(BASE/name,d.BASE/name)
        d.prepare_data=lambda:manifest['streams']
        smoke=dict(ok=True,n_bytes=h.N_DATA,source_hashes=manifest['hashes'],streams=manifest['streams'],split_arms=manifest['split_arms'])
        h.dump(d.OUT/'smoke.json',smoke)
        prepared,fingerprint=d.prepare()
        assert prepared['streams']==manifest['streams']
        jobs=d.job_plan(prepared); assert len(jobs)==42
        for job in jobs:
            code=ast.parse((d.BASE/job['tag']/'kernel.py').read_text())
            config=ast.literal_eval(code.body[0].value)
            meta=json.loads((d.BASE/job['tag']/'kernel-metadata.json').read_text())
            assert config['fingerprint']==fingerprint and config['hashes']==manifest['hashes']
            assert (config['stream'],config['arm'],config['seed'],config['part'])==(job['stream'],job['arm'],job['seed'],job['part'])
            assert config['streams']==manifest['streams'] and meta['id']==job['id']
            assert meta['is_private'] and not meta['enable_gpu'] and not meta['enable_internet']
            assert meta['kernel_sources']==([job['parent_id']] if job['part']==2 else [])
        # Once dispatch state exists, even orchestration-only changes cannot mix versions.
        h.dump(d.STATE,dict(code_fingerprint=fingerprint))
        script=d.BASE/'kernel_template.py'; script.write_text(script.read_text()+'\n# changed\n')
        try:
            d.prepare()
        except RuntimeError as exc:
            assert 'changed after dispatch began' in str(exc)
        else:
            raise AssertionError('Changed source plan accepted after dispatch')
        smoke['source_hashes']={}; h.dump(d.OUT/'smoke.json',smoke)
        try:
            d.prepare()
        except RuntimeError as exc:
            assert 'Successful current-source smoke' in str(exc)
        else:
            raise AssertionError('Stale smoke accepted')
    return dict(check='isolated_prepare_metadata_and_version_guards',kernels=42,guards=2)


def main():
    checks=[]; d=load('dispatch')
    for split,n in (([],30),(list(h.ARMS[1:]),54),(list(h.ARMS),60)):
        jobs=d.job_plan(fixture_manifest(split))
        assert len(jobs)==n and len({j['id'] for j in jobs})==n
        assert sum(j['part']==2 for j in jobs)==n-30
        for j in jobs:
            if j['part']==2:
                assert any(p['id']==j['parent_id'] and p['tag']==j['parent_tag'] and p['part']==1 for p in jobs)
    checks.append(dict(check='plans',lives=30,segment_cases=[30,54,60]))
    for name in ('dispatch.py','kernel_template.py','summarize.py'):
        ast.parse((BASE/name).read_text())
    for path in BASE.glob('*/kernel.py'):
        ast.parse(path.read_text())
        meta=json.loads((path.parent/'kernel-metadata.json').read_text())
        assert meta['is_private'] and not meta['enable_gpu'] and not meta['enable_internet']
        assert meta['dataset_sources']==['ezzzzzx/pdp-data-h34i','ezzzzzx/pdp-code-h34i']
        assert len(meta['kernel_sources'])==(1 if meta['id'].endswith('-part2') else 0)
    checks.append(dict(check='syntax_and_private_cpu_metadata',kernels=len(list(BASE.glob('*/kernel.py')))))
    for status,expected in (('COMPLETE','complete'),('RUNNING','active'),('QUEUED','active'),('PENDING','active'),('ERROR','failed'),('CANCELLED','failed')):
        assert d.status_kind('Kernel x has status "'+status+'"')==expected
    try:
        d.status_kind('ambiguous')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Unknown status accepted')
    with tempfile.TemporaryDirectory(prefix='h34i-pages-') as folder:
        d.OUT=Path(folder); pages=[]
        def cli(*args,**kwargs):
            assert args[:2]==('kernels','list')
            pages.append(args)
            return (0,'ref\nezzzzzx/other-test\nNext Page Token = second') if '--page-token' not in args else (0,'ref\nezzzzzx/h34i-external\n')
        d.cli=cli
        d.kernel_status_text=lambda ref:'Kernel '+ref+' has status "RUNNING"'
        assert set(d.account_active())=={'ezzzzzx/other-test','ezzzzzx/h34i-external'}
        assert len(pages)==2 and all('dateRun' in a for a in pages)
        d.kernel_status_text=lambda ref:'403 forbidden'
        try:
            d.account_active()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unreadable active count accepted')
    checks.append(dict(check='pagination_and_unknown_refusal',pages=2))
    d=load('dispatch')
    with tempfile.TemporaryDirectory(prefix='h34i-duplicate-') as folder:
        tmp=Path(folder); d.OUT=tmp; d.STATE=tmp/'state.json'
        manifest=fixture_manifest(); jobs=d.job_plan(manifest)
        h.dump(tmp/'smoke.json',dict(ok=True,n_bytes=h.N_DATA,source_hashes=manifest['hashes']))
        h.dump(d.STATE,dict(code_fingerprint='offline',datasets_ready=['ezzzzzx/pdp-code-h34i','ezzzzzx/pdp-data-h34i'],jobs=jobs))
        d.load_token=lambda:None
        d.account_active=lambda:[jobs[0]['id']]
        def forbidden(*args,**kwargs):
            raise AssertionError('External duplicate push attempted')
        d.cli=forbidden
        try:
            d.dispatch(manifest,'offline')
        except RuntimeError as exc:
            assert 'already active outside' in str(exc)
        else:
            raise AssertionError('External duplicate accepted')
    checks.append(dict(check='external_duplicate_refused',pushes=0))
    for external,split,failures,interrupted,ready in ((0,[],0,False,False),(1,['dense_6'],1,False,True),
              (4,list(h.ARMS[1:]),0,False,True),(1,[],0,True,True),(0,[],2,False,True)):
        checks.append(check_dispatch(external,split,failures,interrupted,ready))
    checks.append(check_dispatch(0,[],datasets_ready=False,existing_code=True))
    checks.append(check_prepare())
    checks.extend(check_report())
    h.dump(ROOT/'results/h34i/tool-checks.json',dict(ok=True,checks=checks))
    print(json.dumps(dict(ok=True,checks=checks)))


if __name__=='__main__':
    main()
