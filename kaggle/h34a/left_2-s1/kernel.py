"""Private CPU H34a life: left_2, seed 1."""
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h34a_local_depth.py'))
assert len(code)==1, code
sys.path.insert(0,str(code[0].parent))
import h34a_local_depth as h
h.h33.configure(1)
data=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(data)==1, data
raw,digest=h.h33.load_checked(str(data[0]))
r=h.run('left_2',1,raw[:300000])
r.update(sha256_e10m=digest,kernel_id='ezzzzzx/h34a-left-2-s1')
h.dump('/kaggle/working/results.json',r)
