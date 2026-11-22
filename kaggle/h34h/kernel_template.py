"""Private, one-thread CPU run. CONFIG is prepended by offline preparation."""
import os
from pathlib import Path
import sys
import json
import hashlib
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h34h_dense.py'))
assert len(code)==1,code
sys.path.insert(0,str(code[0].parent))
import h34h_dense as h
h.h33.configure(1)
assert h.source_hashes()==CONFIG['hashes'], 'Attached code differs from smoke'
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
raw,digest=h.h33.load_checked(str(raw_paths[0]))
kwargs={}
if CONFIG['part']==1:
    kwargs=dict(checkpoint='/kaggle/working/checkpoint.pt',stop_at=90000)
elif CONFIG['part']==2:
    paths=list(Path('/kaggle/input').rglob('checkpoint.pt'))
    assert len(paths)==1,paths
    checkpoint=paths[0]
    parent=json.loads((checkpoint.parent/'results.json').read_text())
    assert (parent['kernel_id'],parent['code_fingerprint'],parent['sha256_e10m']) == (CONFIG['parent_id'],CONFIG['fingerprint'],digest)
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash==parent['checkpoint_sha256']
    kwargs=dict(resume=checkpoint)
r=h.run(CONFIG['arm'],CONFIG['seed'],raw[:CONFIG['documents']],**kwargs)
if CONFIG['part']!=1: h.check_reference(r)
if CONFIG['part']==2: r['resumed_checkpoint_sha256']=checkpoint_hash
r.update(kernel_id=CONFIG['kernel_id'],sha256_e10m=digest,code_fingerprint=CONFIG['fingerprint'],part=CONFIG['part'])
h.dump('/kaggle/working/results.json',r)
