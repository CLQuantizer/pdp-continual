"""Small network-free checks for dispatch caps, retry, and seed isolation."""
import ast
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import dispatch as d
import summarize as s


def main():
    for path in (d.ROOT/'h35b2_snapshot.py', *d.BASE.glob('*.py')):
        ast.parse(path.read_text(),filename=str(path))
    rows={(seed,f'prog_snap_{lag}'):dict(T_clean=(2 if lag==5 else 3) if seed<9 else (100 if lag==5 else -100))
          for seed in range(6,12) for lag in (5,20)}
    assert s.select_lag(rows)[0]==5
    rows[(8,'prog_snap_20')]['T_clean']=0
    assert s.select_lag(rows)[0]==5
    rows[(8,'prog_snap_20')]['T_clean']=-3
    assert s.select_lag(rows)[0]==20
    rows.pop((7,'prog_snap_5'))
    assert s.select_lag(rows)[0] is None
    for status,expected in [('complete','complete'),('running','active'),('queued','active'),('pending','active'),('error','failed')]:
        assert d.status_kind('kernel has status '+status)==expected
    try:
        d.status_kind('unrecognized')
    except RuntimeError:
        pass
    else:
        raise AssertionError('unknown status accepted')
    checks=[]
    for external in [('a','b','c'),('a','b','c','d'),('ezzzzzx/h35b2-outside','a','b')]:
        with tempfile.TemporaryDirectory(dir=d.OUT) as temp:
            out=Path(temp)
            d.dump(out/'smoke.json', {'ok':True})
            jobs=d.job_specs()
            active=set()
            attempts={j['id']:0 for j in jobs}
            peaks=[0,0]
            fetches=[]
            def fake_cli(*args, **kwargs):
                if args[:2]==('kernels','output'):
                    return 0,'failed-run fixture'
                assert args[:2]==('kernels','push'),args
                tag=Path(args[args.index('-p')+1]).name
                ref=next(j['id'] for j in jobs if j['tag']==tag)
                before=set(external)|active
                assert len(before)<5
                assert sum(x.startswith(d.FAMILY_PREFIX) for x in before)<2
                attempts[ref]+=1
                active.add(ref)
                peaks[0]=max(peaks[0],len(set(external)|active))
                peaks[1]=max(peaks[1],sum(x.startswith(d.FAMILY_PREFIX) for x in set(external)|active))
                return 0,'submitted'
            def report(ref):
                active.discard(ref)
                failed=ref==jobs[0]['id'] and attempts[ref]==1
                return 'kernel has status '+('error' if failed else 'complete')
            with patch.multiple(d,OUT=out,STATE=out/'dispatch.json',load_token=lambda:None,
                                ensure_dataset=lambda *a:None,kernel_report=report,
                                fetch=lambda j,f:fetches.append(j['id']),
                                account_active=lambda:list(set(external)|active),cli=fake_cli), \
                 patch.object(d.time,'sleep',lambda _:None), \
                 patch.object(d.subprocess,'run',lambda *a,**k:None):
                d.dispatch({'jobs':jobs},'offline-fixture')
                first_attempts=attempts.copy()
                d.dispatch({'jobs':jobs},'offline-fixture')
                assert attempts==first_attempts
            state=json.loads((out/'dispatch.json').read_text())
            assert all(j['status']=='complete' for j in state['jobs'])
            assert attempts[jobs[0]['id']]==2 and all(attempts[j['id']]==1 for j in jobs[1:])
            assert len(fetches)==12
            checks.append(dict(external=len(external),external_family=sum(x.startswith(d.FAMILY_PREFIX) for x in external),
                               peak_account=peaks[0],peak_family=peaks[1],one_retry=True,resume_no_new_push=True,complete=6))
    ast.parse(d.kernel_source(d.job_specs()[0],'test',{'h35b2_snapshot.py':'test'}))
    d.dump(d.OUT/'offline-checks.json',dict(ok=True,dispatcher_simulations=checks,
        heldout_isolation=True,selection_missing_guard=True,selection_tie_k5=True,unknown_status_rejected=True))
    print('Offline dispatch caps, retry, source syntax, and seed-isolation checks passed.')

if __name__ == '__main__':
    main()
