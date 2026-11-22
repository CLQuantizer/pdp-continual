"""Offline numeric report and real template checks; fixtures stay in /tmp."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT))
import h34h_dense as h


def load(name):
    spec=importlib.util.spec_from_file_location('h34h_test_'+name,BASE/(name+'.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    s=load('summarize'); d=load('dispatch'); checks=[]
    manifest=json.loads((BASE/'code-sha256.json').read_text())
    fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    prior=json.loads((ROOT/'results/h34g/results.json').read_text())
    refs={r['seed']:r for r in prior['runs'] if r['arm']=='dense_6'}
    with tempfile.TemporaryDirectory(prefix='h34h-report-') as folder:
        tmp=Path(folder); s.assemble(tmp)
        r=json.loads((tmp/'results.json').read_text())
        assert len(r['missing'])==8 and not r['runs']
        assert all(x['evaluated']==0 for x in r['scoring'])
        assert len(r['ablation'])==13
        one=[x for x in r['ablation'] if x['depth']==1]
        assert len(one)==3 and all(x['tail']==one[0]['tail'] and x['full']==one[0]['full'] for x in one)
        assert next(x for x in r['ablation'] if x['arm']=='plain_8')['tail']['mean'] is None
        checks.append(dict(check='missing_scores_and_equivalent_depth_one',missing=8,ablation_rows=13))
        h.dump(tmp/'smoke.json',dict(ok=True,n_bytes=h.N_DATA,source_hashes=h.source_hashes(),sha256_e10m=prior['smoke']['sha256_e10m']))
        for arm in h.ARMS:
            for seed in (0,1):
                r=copy.deepcopy(refs[seed]); depth=h.config(arm)['depth']
                delta={'dense_8':(-.021,-.019),'dense_4':(.03,.04),'dense_6_ones':(.1,.2),'dense_6':(0.,0.)}[arm][seed]
                for key in ('bits_per_byte','tail_bits_per_byte','first_40kb_bits_per_byte'): r[key]+=delta
                for sl in r['slices']: sl['bits_per_byte']+=delta
                r['layers']=(r['layers']+[copy.deepcopy(r['layers'][-1]) for _ in range(2)])[:depth]
                for sl in r['mixer_slices']: sl['mean_weights']=(sl['mean_weights']+[0.,0.])[:depth]
                split=arm=='dense_8' and manifest['split_dense_8']
                tag=f'{arm}-s{seed}'; ref=f'ezzzzzx/h34h-{arm.replace("_","-")}-s{seed}'
                r.update(arm=arm,initial_mixer_weights=h.initial_gate(arm)[0].tolist(),kernel_id=ref+('-part2' if split else ''),
                         code_fingerprint=fingerprint,segment_wall_seconds=1.,wall_seconds=2. if split else 1.)
                if split:
                    parent=tmp/'kernels'/tag; parent.mkdir(parents=True)
                    checkpoint=parent/'checkpoint.pt'; checkpoint.write_bytes(b'SYNTHETIC TEST ONLY')
                    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    h.dump(parent/'results.json',dict(arm=arm,seed=seed,n_bytes=h.N_DATA,kernel_id=ref,
                        code_fingerprint=fingerprint,sha256_e10m=r['sha256_e10m'],bytes_seen=90000,offset=89984,pending_bytes=16,
                        segment_wall_seconds=1.,wall_seconds=1.,checkpoint_sha256=digest))
                    r['resumed_checkpoint_sha256']=digest
                dest=tmp/'kernels'/(tag+'-part2' if split else tag); dest.mkdir(parents=True,exist_ok=True)
                h.dump(dest/'results.json',r)
        s.assemble(tmp); r=json.loads((tmp/'results.json').read_text())
        assert not r['missing'] and len(r['runs'])==8
        scored=[x for x in r['scoring'] if x['threshold'] is not None]
        assert [x['threshold'] for x in scored]==[-.02,-.03]
        assert [x['seeds_meeting'] for x in scored]==[1,2]
        assert all(x['threshold'] is None for x in r['scoring'] if x['quantity'].startswith(('dense_6_ones','dense_4')))
        path=tmp/'kernels/dense_6-s0/results.json'; bad=json.loads(path.read_text()); bad['code_fingerprint']='wrong'; h.dump(path,bad)
        try: s.assemble(tmp)
        except AssertionError: pass
        else: raise AssertionError('Mixed fingerprint accepted')
        checks.append(dict(check='synthetic_gap_signs_counts_reference_and_fingerprint',tail_counts=[1,2],bars=[-.02,-.03]))
    with tempfile.TemporaryDirectory(prefix='h34h-template-') as folder:
        tmp=Path(folder); inp=tmp/'input'; work=tmp/'working'; inp.mkdir(); work.mkdir()
        code=inp/'code'; code.mkdir()
        for name in manifest['hashes']: (code/name).symlink_to(ROOT/name)
        (inp/'enwik8').symlink_to(ROOT/'data/enwik8/enwik8')
        template=(BASE/'kernel_template.py').read_text().replace('/kaggle/input',str(inp)).replace('/kaggle/working',str(work)).replace('stop_at=90000','stop_at=144')
        config=dict(arm='dense_8',seed=0,part=1,documents=256,hashes=h.source_hashes(),fingerprint='offline-template',kernel_id='ezzzzzx/h34h-dense-8-s0',parent_id=None)
        script=tmp/'kernel.py'; env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        script.write_text('CONFIG = '+repr(config)+'\n'+template)
        subprocess.run([sys.executable,str(script)],check=True,env=env)
        first=json.loads((work/'results.json').read_text()); assert (first['offset'],first['pending_bytes'])==(128,16)
        parent=inp/'parent'; work.rename(parent); work.mkdir()
        config.update(part=2,kernel_id='ezzzzzx/h34h-dense-8-s0-part2',parent_id='ezzzzzx/h34h-dense-8-s0')
        script.write_text('CONFIG = '+repr(config)+'\n'+template); subprocess.run([sys.executable,str(script)],check=True,env=env)
        resumed=json.loads((work/'results.json').read_text()); assert resumed['resumed_checkpoint_sha256']==first['checkpoint_sha256']
        h.h33.configure(1); raw,_=h.h33.load_checked(str(ROOT/'data/enwik8/enwik8')); whole=h.run('dense_8',0,raw[:256])
        assert whole['bits_per_byte']==resumed['bits_per_byte'] and whole['layers']==resumed['layers'] and whole['mixer_slices']==resumed['mixer_slices']
        checks.append(dict(check='real_kernel_template_split_resume',n_bytes=256,pending_bytes=16,max_bits_abs=0.))
    h.dump(ROOT/'results/h34h/report-checks.json',dict(ok=True,checks=checks)); print(json.dumps(dict(ok=True,checks=checks)))

if __name__=='__main__': main()
