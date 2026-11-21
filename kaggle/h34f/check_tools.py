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
    spec=importlib.util.spec_from_file_location('h34f_'+name,ROOT/'kaggle/h34f'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

d,s=load('dispatch'),load('summarize')
from h34f_zero_gate import ARMS, N_DATA, TAIL_START, source_hashes, dump


class Pause(Exception):
    pass


def check():
    manifest=json.loads((d.BASE/'code-sha256.json').read_text())
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    smoke=dict(ok=True,n_bytes=N_DATA,sha256_e10m='offline-fixture',source_hashes=source_hashes())
    # All network-facing functions are mocked; .env is never accessed.
    with tempfile.TemporaryDirectory(dir=ROOT/'results/h34f') as tmp:
        out=Path(tmp)
        dump(out/'smoke.json',smoke)
        for active,expected in [(['ezzzzzx/h34f-external-a','ezzzzzx/h34f-external-b'],0),
                                (['other/'+str(i) for i in range(5)],0),([],2)]:
            state_path=out/'dispatch.json'
            jobs=[dict(tag=f'{a}-s{seed}',id=f"ezzzzzx/h34f-{a.replace('_','-')}-s{seed}",
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
        # Synthetic fixtures exercise slice accounting, all three fixed bars,
        # baseline rejection, and missing-pair counts without kernel execution.
        from h34f_zero_gate import H34B_REFERENCE, initial_gate
        def fixture(arm,seed):
            control='plain_'+arm.split('_')[1]
            ref=H34B_REFERENCE[control][seed]
            early=ref['first_40kb_bits_per_byte']; tail=ref['tail_bits_per_byte']
            if arm=='zgate_4':
                early-=.6 if seed==0 else .2
                tail+=0. if seed==0 else .02
            if arm=='zgate_6':
                early=4.; tail=3.
            slices=[dict(start=i,end=i+20000,bits_per_byte=early if i<40000 else tail)
                    for i in range(0,N_DATA,20000)]
            depth=int(arm.split('_')[1])
            return dict(arm=arm,seed=seed,n_bytes=N_DATA,tail_start=TAIL_START,
                kernel_id=f"ezzzzzx/h34f-{arm.replace('_','-')}-s{seed}",
                sha256_e10m='offline-fixture',code_fingerprint=fingerprint,wall_seconds=1.,
                initial_mixer_weights=initial_gate(arm)[0].tolist(),
                first_40kb_bits_per_byte=early,tail_bits_per_byte=tail,
                bits_per_byte=(early*40000+tail*140000)/N_DATA,slices=slices,
                mixer_slices=[dict(start=x['start'],end=x['end'],mean_weights=[1/depth]*depth) for x in slices],
                layers=[dict(own_bits_per_byte=4.,tail_own_bits_per_byte=4.,mean_mixer_weight_tail=1/depth,
                    mean_mixer_weight_last_50kb=1/depth,final_mixer_weight=1/depth,
                    slices=[dict(start=x['start'],end=x['end'],own_bits_per_byte=4.) for x in slices]) for k in range(depth)])
        for arm in ARMS:
            for seed in (0,1):
                r=fixture(arm,seed)
                s.validate(r,arm,seed,smoke,manifest,fingerprint)
                dump(out/'kernels'/f'{arm}-s{seed}'/'results.json',r)
        for mutate in ('full','baseline','gate','fingerprint'):
            bad=fixture('plain_4',0)
            if mutate=='full': bad['bits_per_byte']+=1e-4
            elif mutate=='gate': bad['initial_mixer_weights']=[1.,0.,0.,0.]
            elif mutate=='fingerprint': bad['code_fingerprint']='wrong'
            else:
                bad['first_40kb_bits_per_byte']+=.01
                bad['bits_per_byte']+=.01*40000/N_DATA
                for sl in bad['slices'][:2]: sl['bits_per_byte']+=.01
            try: s.validate(bad,'plain_4',0,smoke,manifest,fingerprint)
            except AssertionError: pass
            else: raise AssertionError('Corrupt result accepted: '+mutate)
        s.assemble(out)
        report=json.loads((out/'results.json').read_text())
        assert len(report['scoring'])==6 and not report['missing']
        bars={x['quantity']+' '+x['range']:x for x in report['scoring'] if x['threshold'] is not None}
        assert len(bars)==3
        assert bars['zgate_4 - plain_4 Full']['seeds_meeting']==1
        assert bars['zgate_4 - plain_4 Tail']['seeds_meeting']==1
        assert bars['zgate_6 - plain_1 Full']['seeds_meeting']==2
        (out/'kernels/zgate_4-s1/results.json').unlink()
        s.assemble(out)
        first=json.loads((out/'results.json').read_text())['scoring'][0]
        assert first['seeds_meeting']==1 and first['evaluated']==1 and first['mean'] is None
    print(json.dumps(dict(ok=True,experiment_cap=2,account_cap=5,resume=True,
                          scoring_and_missing_pairs=True,corrupt_results_rejected=True)))


if __name__=='__main__': check()
