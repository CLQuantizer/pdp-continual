import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CURIO_LR'] = '0.004'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONHASHSEED'] = '0'
if os.environ.get('_W9_KERNEL') != '1':
    os.environ['_W9_KERNEL'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)
if __name__ == '__main__':
    hits = sorted(Path('/kaggle/input').rglob('w9_day_night.py'))
    if len(hits) != 1:
        raise RuntimeError('expected exactly one W9 dataset')
    code = str(hits[0].parent)
    sys.path.insert(0, code)
    os.environ['PYTHONPATH'] = code
    import w9_day_night as w9
    w9.kernel_main(66, '/kaggle/working', ('nosleep', 'fixed_random', 'fixed_night', 'pressure', 'pressure_opp'), 'ezzzzzx/w9-day-night-s66', '')
