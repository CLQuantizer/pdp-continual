"""Offline dispatcher and scoring regression checks. No network or credentials."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj


def main():
    d=module('dispatch'); summary=module('summarize')
    out=ROOT/'results/h34b/offline-check'; out.mkdir(parents=True,exist_ok=True)
    hashes=d.source_hashes()
    actual_smoke=json.loads((ROOT/'results/h34b/smoke.json').read_text())
    if not actual_smoke['ok']:
        with patch.object(d,'load_token',side_effect=AssertionError('Credential load forbidden')),patch.object(d,'cli',side_effect=AssertionError('Network forbidden')):
            try: d.dispatch({},'unused')
            except RuntimeError as exc: assert 'Smoke gate failed' in str(exc)
            else: raise AssertionError('Failed smoke allowed dispatch')
    d.dump(out/'smoke.json',dict(ok=True,n_bytes=250000,source_hashes=hashes))
    jobs=[dict(tag=f'{a}-s{s}',id=f"ezzzzzx/h34b-{a.replace('_','-')}-s{s}",status='pending',attempts=0) for a in d.ARMS for s in (0,1)]
    fingerprint='offline-fixture'
    d.dump(out/'dispatch.json',dict(code_fingerprint=fingerprint,dataset_ready=True,jobs=jobs))
    active=set(); attempts={}; highwater=0
    retry_id=jobs[0]['id']; fetched=[]
    def fake_cli(*args,**kwargs):
        nonlocal highwater
        if args[:2]==('kernels','push'):
            tag=Path(args[-1]).name
            ref=next(j['id'] for j in jobs if j['tag']==tag)
            assert len(active)<3 and len(active)+2<5
            active.add(ref); attempts[ref]=attempts.get(ref,0)+1
            highwater=max(highwater,len(active)); return 0,'pushed'
        if args[:2]==('kernels','status'):
            ref=args[2]; assert ref in active
            active.remove(ref)
            state='error' if ref==retry_id and attempts[ref]==1 else 'complete'
            return 0,ref+' has status '+state
        if args[:2]==('kernels','output'): return 0,'failure logs'
        raise AssertionError(args)
    manifest=dict(hashes=hashes,missing=[],documents=250000)
    with patch.object(d,'OUT',out),patch.object(d,'STATE',out/'dispatch.json'),patch.object(d,'load_token'),patch.object(d,'cli',fake_cli),patch.object(d,'account_active',lambda: list(active)+['ezzzzzx/other-a','ezzzzzx/other-b']),patch.object(d,'fetch',lambda job,*args:fetched.append(job['id'])),patch.object(d.time,'sleep'),patch.object(d.subprocess,'run'):
        d.dispatch(manifest,fingerprint)
        saved=json.loads((out/'dispatch.json').read_text())
        assert all(j['status']=='complete' for j in saved['jobs'])
        assert highwater==3 and attempts[retry_id]==2 and len(fetched)==14
        before=copy.deepcopy(attempts)
        d.dispatch(manifest,fingerprint)
        assert attempts==before
    # Terminal second failure exhausts the single retry and refuses further work.
    jobs2=copy.deepcopy(jobs)
    for j in jobs2: j.update(status='complete',attempts=1)
    jobs2[0].update(status='active',attempts=2)
    d.dump(out/'dispatch.json',dict(code_fingerprint=fingerprint,dataset_ready=True,jobs=jobs2))
    def failure_cli(*args,**kw): return 0,'has status error' if args[1]=='status' else 'logs'
    with patch.object(d,'OUT',out),patch.object(d,'STATE',out/'dispatch.json'),patch.object(d,'load_token'),patch.object(d,'cli',failure_cli):
        try: d.dispatch(manifest,fingerprint)
        except RuntimeError as e: assert 'one retry' in str(e)
        else: raise AssertionError('Retry exhaustion did not stop')
    # Pagination and unknown statuses fail closed, all without touching the API.
    def pages(*args,**kw):
        if args[1]=='list':
            return (0,'ref\nezzzzzx/b\n') if '--page-token' in args else (0,'ref\nezzzzzx/a\nNext Page Token = second')
        return 0,'has status running' if args[-1].endswith('/a') else 'has status complete'
    with patch.object(d,'OUT',out),patch.object(d,'cli',pages): assert d.account_active()==['ezzzzzx/a']
    with patch.object(d,'OUT',out),patch.object(d,'cli',lambda *a,**kw:(0,'ref\nezzzzzx/a\n') if a[1]=='list' else (1,'403 forbidden')):
        try: d.account_active()
        except RuntimeError: pass
        else: raise AssertionError('Unreadable account status accepted')
    # Synthetic complete scoring fixture; full and tail deliberately differ.
    sm=json.loads((ROOT/'results/h34b/smoke.json').read_text())
    prepared=json.loads((ROOT/'kaggle/h34b/code-sha256.json').read_text())
    n=prepared['documents']
    sm.update(ok=True,n_bytes=n,source_hashes=hashes)
    fixture=out/'summary'; fixture.mkdir(exist_ok=True); d.dump(fixture/'smoke.json',sm)
    manifest=json.loads((ROOT/'kaggle/h34b/code-sha256.json').read_text())
    import hashlib
    fp=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    tails={'plain_1':3.3,'plain_2':3.2,'plain_4':3.15,'plain_6':3.10,'plain_4_fixmix':3.22,'plain_4_noskip':3.2,'wide_6':3.25}
    tail_start=40_000
    def full_bits(n, tail):
        return (4.*tail_start+tail*(n-tail_start))/n
    def fake_slices(n, tail):
        rows=[]
        for i in range(0,n,25000):
            a,b=i,min(i+25000,n)
            total=0.
            p0,p1=max(a,0),min(b,tail_start)
            if p1>p0: total+=4.*(p1-p0)
            t0=max(a,tail_start)
            if b>t0: total+=tail*(b-t0)
            rows.append(dict(start=a,end=b,bits_per_byte=total/(b-a)))
        return rows
    for a in d.ARMS:
        for s in (0,1):
            tail=tails[a]; depth=1 if a=='wide_6' else int(a.split('_')[1]); w=.25 if a.endswith('fixmix') else 1.
            layers=[dict(own_bits_per_byte=4.,tail_own_bits_per_byte=3.5,mean_mixer_weight_tail=w,mean_mixer_weight_last_50kb=w,final_mixer_weight=w) for _ in range(depth)]
            r=dict(arm=a,seed=s,n_bytes=n,kernel_id=f"ezzzzzx/h34b-{a.replace('_','-')}-s{s}",code_fingerprint=fp,sha256_e10m=sm['sha256_e10m'],bits_per_byte=full_bits(n,tail),tail_bits_per_byte=tail,tail_edge_bits_per_byte=tail,tail_start=40_000,wall_seconds=1.,layers=layers,max_frozen_weight_change=0,slices=fake_slices(n,tail))
            d.dump(fixture/'kernels'/f'{a}-s{s}'/'results.json',r)
    summary.assemble(fixture)
    result=json.loads((fixture/'results.json').read_text()); assert [x['count'] for x in result['scoring']]==[2,2,2]
    assert (fixture/'RESULTS.md').read_text().startswith('| Tail quantity |')
    # Removing one seed must lower evaluated and pass counts, not infer a mean.
    (fixture/'kernels/plain_6-s1/results.json').unlink()
    summary.assemble(fixture)
    result=json.loads((fixture/'results.json').read_text())
    assert result['scoring'][0]['evaluated']==1 and result['scoring'][0]['mean'] is None
    d.dump(out/'checks.json',dict(ok=True,checks=['failed_smoke_refuses_before_credentials_or_network','concurrency_5_account_3_h34b','one_retry','completed_resume','retry_exhaustion','paginated_account','unknown_status_refusal','tail_scoring_directions','missing_seed_NA'],max_h34b_active=highwater))
    print('Offline dispatcher and summary checks passed; synthetic fixtures in results/h34b/offline-check only.')

if __name__=='__main__': main()
