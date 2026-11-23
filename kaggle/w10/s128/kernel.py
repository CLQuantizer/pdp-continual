import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CURIO_LR']='0.004'
os.environ['PYTHONHASHSEED']='0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
if os.environ.get('_W10_KERNEL')!='1':
    os.environ['_W10_KERNEL']='1'
    os.execv(sys.executable,[sys.executable]+sys.argv)
if __name__=='__main__':
    hits=sorted(Path('/kaggle/input').rglob('w10_open_sleep.py'))
    if len(hits)!=1:
        raise RuntimeError('expected one W10 code dataset')
    code=str(hits[0].parent)
    sys.path.insert(0,code)
    os.environ['PYTHONPATH']=code
    import w10_open_sleep as w10
    w10.kernel_main(128,'/kaggle/working',('nosleep', 'fixed_random', 'fixed_night', 'pressure_retina'),'ezzzzzx/w10-s128')
