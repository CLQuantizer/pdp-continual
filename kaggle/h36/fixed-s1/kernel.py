"""H36 private CPU, fixed, seed 1. Requires H35 corpus."""
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h36_curious_patching.py'))
assert len(code)==1,code
sys.path.insert(0,str(code[0].parent))
assert (code[0].parent/'h35_corpus.py').is_file(),'H35 corpus missing from code dataset'
import h36_curious_patching as h
# H35 locates enwik8 through its own loader; also offer the conventional
# local path for implementations with build(seed)'s default data argument.
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
h.configure()
raw=h.h26.load_enwik8(str(raw_paths[0]))
digest=h.h26.sha256_arr(raw[:h.h26.E10M])
assert digest==h.h26.E10M_SHA256
local=Path('/kaggle/working/data/enwik8/enwik8')
local.parent.mkdir(parents=True,exist_ok=True)
if not local.exists(): local.symlink_to(raw_paths[0])
os.chdir('/kaggle/working')
r=h.run('fixed',1,400)
r.update(kernel_id='ezzzzzx/h36-fixed-s1',sha256_e10m=digest,code_fingerprint='0dc50e774338dd7402f210e55022e344c14c1eab4ebfe6bfed710bc7b3c933b0')
h.dump('/kaggle/working/results.json',r)
