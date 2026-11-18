"""H35b private CPU, seed 6, ['oracle', 'high_loss', 'rho']."""
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h35b_neural_diet.py'))
assert len(code)==1,code
sys.path.insert(0,str(code[0].parent))
import h35b_neural_diet as h
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
if __name__ == '__main__':
    h.kernel_main(6,'/kaggle/working','ezzzzzx/h35b-s6',policies=['oracle', 'high_loss', 'rho'],
                  data=str(raw_paths[0]),fingerprint='4d72c241e4811e637d2f99a4c99f44c69c976a4b5484129e30fd8eb8d601f069')
