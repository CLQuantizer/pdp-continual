"""Network-free dispatcher lifecycle/cap/retry and output validation checks."""
import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
from unittest.mock import patch
import dispatch as d
import summarize as s


def check_dispatch(manifest, fingerprint, fail_twice=False):
    with tempfile.TemporaryDirectory(dir=d.OUT, prefix='offline-') as temp:
        out=Path(temp); remote={}; pushes={}; maximum=0; max_account=0
        jobs=d.job_plan(manifest)
        def cli(*args, **kwargs):
            nonlocal maximum,max_account
            if args[:2] == ('kernels','push'):
                tag=Path(args[-1]).name; job=next(j for j in jobs if j['tag']==tag)
                state=json.loads((out/'dispatch.json').read_text())
                if job['parent_tag']:
                    assert next(j for j in state['jobs'] if j['tag']==job['parent_tag'])['status']=='complete'
                assert len(remote)<4
                pushes[tag]=pushes.get(tag,0)+1; assert pushes[tag]<=2
                remote[job['id']]=job
                maximum=max(maximum,len(remote)); max_account=max(max_account,len(remote)+1)
                return 0,'pushed'
            if args[:2] == ('kernels','output'): return 0,'output'
            raise AssertionError(args)
        target=jobs[0]['tag']
        class SimulatedInterruption(Exception): pass
        interrupted=False
        def sleep(*args):
            nonlocal interrupted
            if not interrupted:
                interrupted=True
                raise SimulatedInterruption()
        def status(ref):
            job=remote.pop(ref)
            failing=job['tag']==target and (fail_twice or pushes[job['tag']]==1)
            return ref+' has status '+('error' if failing else 'complete')
        def account():
            return ['ezzzzzx/unrelated-running']+list(remote)
        initial=dict(code_fingerprint=fingerprint,dataset_ready=True,jobs=copy.deepcopy(jobs))
        d.dump(out/'dispatch.json',initial)
        with patch.object(d,'OUT',out), patch.object(d,'STATE',out/'dispatch.json'), \
             patch.object(d,'cli',cli), patch.object(d,'load_token',lambda:None), \
             patch.object(d,'account_active',account), patch.object(d,'kernel_status_text',status), \
             patch.object(d,'fetch',lambda *a:None), patch.object(d.time,'sleep',sleep), \
             patch.object(d.subprocess,'run',lambda *a,**k:None), contextlib.redirect_stdout(io.StringIO()):
            try:
                try:
                    d.dispatch(manifest,fingerprint)
                except SimulatedInterruption:
                    # Four active jobs survive a dispatcher-process restart.
                    d.dispatch(manifest,fingerprint)
                assert not fail_twice
                # Resume must not resubmit completed lives.
                before=copy.deepcopy(pushes); d.dispatch(manifest,fingerprint); assert pushes==before
            except RuntimeError as exc:
                assert fail_twice and 'exhausted' in str(exc)
        state=json.loads((out/'dispatch.json').read_text())
        assert pushes[target]==2 and maximum<=4 and max_account<=5
        if fail_twice:
            assert any(j['status']=='failed' for j in state['jobs'])
            assert all(j['status']=='complete' for j in state['jobs'] if j['arm']!=jobs[0]['arm'] or j['seed']!=jobs[0]['seed'])
        else:
            assert all(j['status']=='complete' for j in state['jobs'])
        return dict(parts=len(jobs),max_h34j=maximum,max_account=max_account,retry_pushes=pushes[target],terminal_failure=fail_twice,active_restart_tested=interrupted)


def check_account_listing():
    calls=[]
    def cli(*args, **kwargs):
        calls.append(args)
        assert '--sort-by' in args and args[args.index('--sort-by')+1]=='dateRun'
        if '--page-token' not in args:
            return 0,'Next Page Token = next-page\nref,title\nezzzzzx/other,other\n'
        assert args[-2:]==('--page-token','next-page')
        return 0,'ref,title\nezzzzzx/h34j-external,external\n'
    def status(ref):
        return ref+' has status '+('running' if ref.endswith('external') else 'complete')
    with tempfile.TemporaryDirectory(dir=d.OUT,prefix='account-') as temp:
        with patch.object(d,'OUT',Path(temp)),patch.object(d,'cli',cli), \
             patch.object(d,'kernel_status_text',status),contextlib.redirect_stdout(io.StringIO()):
            assert d.account_active()==['ezzzzzx/h34j-external'] and len(calls)==2
            with patch.object(d,'kernel_status_text',lambda ref:'403 forbidden'):
                try: d.account_active()
                except RuntimeError: pass
                else: raise AssertionError('Unreadable account treated as free')
    for body,kind in (('complete','complete'),('error','failed'),('running','active'),('queued','active'),('pending','active')):
        assert d.status_kind('ref has status '+body)==kind
    for body in ('garbled','ref has status unknown'):
        try: d.status_kind(body)
        except RuntimeError: pass
        else: raise AssertionError('Unknown status accepted')
    return dict(pagination=True,unreadable_rejected=True,unknown_rejected=True)


def main():
    manifest,fingerprint=d.prepare()
    records=[check_dispatch(manifest,fingerprint),check_dispatch(manifest,fingerprint,True)]
    account_checks=check_account_listing()
    jobs=d.job_plan(manifest)
    for job in jobs:
        folder=d.BASE/job['tag']; metadata=json.loads((folder/'kernel-metadata.json').read_text())
        assert metadata['is_private'] and not metadata['enable_gpu'] and not metadata['enable_internet']
        assert metadata['kernel_sources']==([job['parent_id']] if job['parent_id'] else [])
        compile((folder/'kernel.py').read_text(),str(folder/'kernel.py'),'exec')
    # Import only the staged source directory in a fresh isolated interpreter.
    subprocess.run([sys.executable,'-I','-B','-c',
        'import sys; sys.path.insert(0,sys.argv[1]); import h34j_long as h; '
        'assert len(h.source_hashes()) == 16',str(d.BASE/'code')],check=True)
    # A fully occupied account cannot submit. Both H34j and unrelated jobs count.
    for refs in ([f'ezzzzzx/h34j-external-{k}' for k in range(4)],
                 [f'ezzzzzx/other-{k}' for k in range(5)]):
        with tempfile.TemporaryDirectory(dir=d.OUT,prefix='offline-cap-') as temp:
            out=Path(temp); d.dump(out/'dispatch.json',dict(code_fingerprint=fingerprint,dataset_ready=True,jobs=jobs))
            class StopPolling(Exception): pass
            def forbidden(*a,**k): raise AssertionError('pushed while full')
            with patch.object(d,'OUT',out),patch.object(d,'STATE',out/'dispatch.json'),patch.object(d,'load_token',lambda:None), \
                 patch.object(d,'account_active',lambda:refs),patch.object(d,'cli',forbidden), \
                 patch.object(d.time,'sleep',side_effect=StopPolling),contextlib.redirect_stdout(io.StringIO()):
                try: d.dispatch(manifest,fingerprint)
                except StopPolling: pass
    d.dump(d.OUT/'offline-checks.json',dict(ok=True,code_fingerprint=fingerprint,dispatch=records,account_checks=account_checks,cap_block_cases=2,metadata_checked=len(jobs)))
    print(json.dumps(dict(ok=True,code_fingerprint=fingerprint,dispatch=records,account_checks=account_checks,cap_block_cases=2,metadata_checked=len(jobs))))


if __name__=='__main__': main()
