"""H34a3 private CPU, casc_4, seed 0. Partial-sum stack."""
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h34a3_cascade.py'))
assert len(code)==1,code
sys.path.insert(0,str(code[0].parent))
import h34a3_cascade as h
h.h33.configure(1)
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
raw,digest=h.h33.load_checked(str(raw_paths[0]))
r=h.run('casc_4',0,raw[:300_000])
r.update(kernel_id='ezzzzzx/h34a3-casc-4-s0',sha256_e10m=digest,code_fingerprint='0c19e75d63ff41b9ec360bc6299f6c026e90efbbed7a7758541f8e5eeb885591')
h.dump('/kaggle/working/results.json',r)
