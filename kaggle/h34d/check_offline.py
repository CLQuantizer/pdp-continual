"""Check packaging, slot caps, resume, and summary arithmetic without networking."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import h34d_dfa as h


def module(name):
    spec=importlib.util.spec_from_file_location('h34d_'+name,ROOT/'kaggle/h34d'/f'{name}.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class StopPoll(Exception): pass


def main():
    d=module('dispatch'); summary=module('summarize')
    manifest=json.loads((d.BASE/'code-sha256.json').read_text())
    assert manifest['hashes']==h.source_hashes() and manifest['documents']==180000
    for p in [ROOT/'h34d_dfa.py',*d.BASE.rglob('*.py')]: ast.parse(p.read_text())
    folders=[p.parent for p in d.BASE.glob('*/kernel-metadata.json')]
    assert len(folders)==6
    for folder in folders:
        meta=json.loads((folder/'kernel-metadata.json').read_text())
        assert meta['is_private'] is True and meta['enable_gpu'] is False and meta['enable_internet'] is False
        assert meta['dataset_sources']==['ezzzzzx/enwik8','ezzzzzx/pdp-code-h34d']
        assert (folder/meta['code_file']).exists()
    assert d.status_kind('x has status "RUNNING"')=='active'
    assert d.status_kind('x has status "QUEUED"')=='active'
    assert d.status_kind('x has status "COMPLETE"')=='complete'
    assert d.status_kind('x has status "ERROR"')=='failed'
    try: d.status_kind('403 forbidden')
    except RuntimeError: pass
    else: raise AssertionError('unknown status accepted')
    # Exhaust pagination and count queued/pending jobs too.
    listings=iter(['ref\nezzzzzx/a\nNext Page Token = token2','ref\nezzzzzx/b'])
    def listing_cli(*args,**kw): return 0,next(listings)
    with tempfile.TemporaryDirectory(dir=ROOT/'results/h34d') as tmp:
        out=Path(tmp)
        with patch.object(d,'OUT',out),patch.object(d,'cli',listing_cli),patch.object(d,'kernel_status_text',lambda ref:ref+' has status "PENDING"'):
            assert len(d.account_active())==2
        # Drive the actual dispatch loop up to its first polling wait.
        cases=[('account_cap',[f'ezzzzzx/other-{i}' for i in range(5)],0),
               ('experiment_cap',[f'ezzzzzx/h34d-external-{i}' for i in range(3)],0),
               ('free_slots',[],3),('one_account_slot',[f'ezzzzzx/other-{i}' for i in range(4)],1)]
        for name,active,expected in cases:
            case=out/name; case.mkdir()
            h.dump(case/'smoke.json',dict(ok=True,n_bytes=h.N_DATA,source_hashes=manifest['hashes']))
            jobs=[dict(tag=f'{a}-s{s}',id=f'ezzzzzx/h34d-{a.replace("_","-")}-s{s}',status='pending',attempts=0)
                  for a in h.ARMS for s in (0,1)]
            h.dump(case/'dispatch.json',dict(code_fingerprint='offline',dataset_ready=True,jobs=jobs))
            pushed=[]
            def cli(*args,**kw):
                assert args[:2]==('kernels','push'); pushed.append(args[-1]); return 0,'offline push'
            with patch.object(d,'OUT',case),patch.object(d,'STATE',case/'dispatch.json'),patch.object(d,'load_token',lambda:None),patch.object(d,'account_active',lambda:active),patch.object(d,'cli',cli),patch.object(d.time,'sleep',side_effect=StopPoll):
                try: d.dispatch(manifest,'offline')
                except StopPoll: pass
            assert len(pushed)==expected,(name,pushed)
            state=json.loads((case/'dispatch.json').read_text())
            assert sum(j['attempts'] for j in state['jobs'])==expected
            # A resume polls existing active jobs rather than submitting them twice.
            with patch.object(d,'OUT',case),patch.object(d,'STATE',case/'dispatch.json'),patch.object(d,'load_token',lambda:None),patch.object(d,'account_active',lambda:[f'ezzzzzx/other-{i}' for i in range(5)]),patch.object(d,'kernel_status_text',lambda ref:ref+' has status "RUNNING"'),patch.object(d,'cli',side_effect=AssertionError('unexpected resubmission')),patch.object(d.time,'sleep',side_effect=StopPoll):
                try: d.dispatch(manifest,'offline')
                except StopPoll: pass
        # Invalid/private-unreadable account status must prevent any slot assumption.
        with patch.object(d,'OUT',out),patch.object(d,'cli',lambda *a,**kw:(0,'ref\nezzzzzx/a')),patch.object(d,'kernel_status_text',lambda ref:'403 forbidden'):
            try: d.account_active()
            except RuntimeError: pass
            else: raise AssertionError('unreadable account accepted')
        summary.assemble(out/'empty')
        r=json.loads((out/'empty/results.json').read_text())
        assert len(r['missing'])==6 and all(x['evaluated']==0 for x in r['scoring'])
        assert r['scoring'][0]['seeds_meeting']==0 and all(x['seeds_meeting'] is None for x in r['scoring'][1:])
        # Tail reconstruction must price the 40–50 KB edge separately.
        fake=dict(n_bytes=180000,tail_edge_bits_per_byte=2.,slices=[dict(start=i,end=min(i+25000,180000),bits_per_byte=3.) for i in range(0,180000,25000)])
        assert abs(summary.reconstruct(fake,40000)-(2*10000+3*130000)/140000)<1e-12
        # End-to-end synthetic score pairs, kept only in this temporary folder.
        scored=out/'synthetic'; scored.mkdir()
        fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        sm=dict(ok=True,n_bytes=h.N_DATA,source_hashes=h.source_hashes(),sha256_e10m='synthetic')
        h.dump(scored/'smoke.json',sm)
        for arm in h.ARMS:
            for seed in (0,1):
                value=3.2+seed*.1+({'plain_4':0.,'dfa_4':(-.04 if seed==0 else -.02),'dfa_only_4':.1}[arm])
                r=dict(arm=arm,seed=seed,n_bytes=h.N_DATA,tail_start=h.TAIL_START,
                       kernel_id=f"ezzzzzx/h34d-{arm.replace('_','-')}-s{seed}",
                       sha256_e10m='synthetic',code_fingerprint=fingerprint,wall_seconds=1.,
                       bits_per_byte=value,tail_bits_per_byte=value,tail_edge_bits_per_byte=value,
                       slices=[dict(start=i,end=min(i+25000,h.N_DATA),bits_per_byte=value) for i in range(0,h.N_DATA,25000)],
                       layers=[dict(own_bits_per_byte=value,tail_own_bits_per_byte=value,mean_mixer_weight_tail=.25,final_mixer_weight=.25)]*4,
                       dfa_norms=[] if arm=='plain_4' else [dict(start=i,end=min(i+h.CHUNK,h.N_DATA),layer=k,
                           own_norm=1.,raw_dfa_norm=2.,alpha=.5,dfa_norm=1.,only=(arm=='dfa_only_4' and k>=2))
                           for i in range(0,h.N_DATA,h.CHUNK) for k in range(1,5)])
                h.dump(scored/'kernels'/f'{arm}-s{seed}'/'results.json',r)
        summary.assemble(scored)
        result=json.loads((scored/'results.json').read_text())
        assert not result['missing'] and result['scoring'][0]['seeds_meeting']==1
        assert result['scoring'][0]['evaluated']==2 and abs(result['scoring'][0]['mean']+.03)<1e-12
        r['code_fingerprint']='stale'
        try: summary.validate(r,arm,seed,sm,manifest,fingerprint)
        except AssertionError: pass
        else: raise AssertionError('stale scored source accepted')
    print(json.dumps(dict(ok=True,kernels=6,checks=['syntax','metadata','fingerprint','pagination','unknown_status','account_cap','experiment_cap','resume','missing_scores','tail_edge','score_arithmetic','stale_result'])))


if __name__=='__main__': main()
