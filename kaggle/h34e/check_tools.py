"""Offline checks of dispatch slot limits, resume, and summary integrity."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import importlib.util

def load(name):
    spec=importlib.util.spec_from_file_location('h34e_'+name,ROOT/'kaggle/h34e'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

d,s=load('dispatch'),load('summarize')
from h34e_self_distill import ARMS, N_DATA, TAIL_START, source_hashes, dump


class Pause(Exception):
    pass


def check():
    manifest=json.loads((d.BASE/'code-sha256.json').read_text())
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    smoke=dict(ok=True,n_bytes=N_DATA,sha256_e10m='offline-fixture',source_hashes=source_hashes())
    # All network-facing functions are mocked; .env is never accessed.
    with tempfile.TemporaryDirectory(dir=ROOT/'results/h34e') as tmp:
        out=Path(tmp)
        dump(out/'smoke.json',smoke)
        for active,expected in [(['ezzzzzx/h34e-external-a','ezzzzzx/h34e-external-b'],0),
                                (['other/'+str(i) for i in range(5)],0),([],2)]:
            state_path=out/'dispatch.json'
            jobs=[dict(tag=f'{a}-s{seed}',id=f"ezzzzzx/h34e-{a.replace('_','-')}-s{seed}",
                       status='pending',attempts=0) for a in ARMS for seed in (0,1)]
            dump(state_path,dict(code_fingerprint=fingerprint,dataset_ready=True,jobs=jobs))
            calls=[]
            def cli(*args,**kwargs):
                calls.append(args)
                assert args[:2]==('kernels','push')
                return 0,'mock push'
            with patch.object(d,'OUT',out),patch.object(d,'STATE',state_path),patch.object(d,'load_token'),\
                 patch.object(d,'account_active',return_value=active),patch.object(d,'cli',side_effect=cli),\
                 patch.object(d.time,'sleep',side_effect=Pause):
                try: d.dispatch(manifest,fingerprint)
                except Pause: pass
            assert len(calls)==expected,(active,calls)
        # Resume two active jobs, finish them, and reuse both freed slots.
        state=json.loads(state_path.read_text())
        with patch.object(d,'OUT',out),patch.object(d,'STATE',state_path),patch.object(d,'load_token'),\
             patch.object(d,'kernel_status_text',return_value='has status complete'),patch.object(d,'fetch') as fetch,\
             patch.object(d,'account_active',return_value=[]),patch.object(d,'cli',return_value=(0,'mock push')) as cli,\
             patch.object(d.time,'sleep',side_effect=Pause):
            try: d.dispatch(manifest,fingerprint)
            except Pause: pass
            assert fetch.call_count==2 and cli.call_count==2
        resumed=json.loads(state_path.read_text())
        assert sum(j['status']=='complete' for j in resumed['jobs'])==2
        assert sum(j['status']=='active' for j in resumed['jobs'])==2
        # A real H34b result supplies the exact slice accounting for fixtures.
        for seed in (0,1):
            r=json.loads((ROOT/f'results/h34b/kernels/plain_4-s{seed}/results.json').read_text())
            r.update(distill_lambda=0.,layer_lambdas=[0.]*4,sha256_e10m='offline-fixture',
                     code_fingerprint=fingerprint,kernel_id=f'ezzzzzx/h34e-plain-4-s{seed}')
            for layer in r['layers']: layer['mean_mixer_weight_full']=.25
            dump(out/'kernels'/f'plain_4-s{seed}'/'results.json',r)
            s.validate(r,'plain_4',seed,smoke,manifest,fingerprint)
            distilled=copy.deepcopy(r)
            delta=-.04 if seed==0 else -.02
            distilled.update(arm='distill_4',distill_lambda=.5,layer_lambdas=[.5]*4,
                             kernel_id=f'ezzzzzx/h34e-distill-4-s{seed}')
            for key in ('bits_per_byte','tail_bits_per_byte','tail_edge_bits_per_byte'): distilled[key]+=delta
            for sl in distilled['slices']: sl['bits_per_byte']+=delta
            dump(out/'kernels'/f'distill_4-s{seed}'/'results.json',distilled)
            bad=copy.deepcopy(r); bad['bits_per_byte']+=1e-4
            try: s.validate(bad,'plain_4',seed,smoke,manifest,fingerprint)
            except AssertionError: pass
            else: raise AssertionError('Corrupt result accepted')
        s.assemble(out)
        report=json.loads((out/'results.json').read_text())
        first=report['scoring'][0]
        assert first['seeds_meeting']==1 and first['evaluated']==2 and abs(first['mean']+.03)<1e-9
        assert len(report['missing'])==4
        (out/'kernels/distill_4-s1/results.json').unlink()
        s.assemble(out)
        first=json.loads((out/'results.json').read_text())['scoring'][0]
        assert first['seeds_meeting']==1 and first['evaluated']==1 and first['mean'] is None
    print(json.dumps(dict(ok=True,experiment_cap=2,account_cap=5,resume=True,
                          scoring_and_missing_pairs=True,corrupt_results_rejected=True)))


if __name__=='__main__': check()
