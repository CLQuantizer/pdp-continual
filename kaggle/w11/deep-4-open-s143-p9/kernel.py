import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CURIO_LR'] = '0.004'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
if os.environ.get('_W11_KERNEL') != '1':
    os.environ['_W11_KERNEL'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)
if __name__ == '__main__':
    hits = list(Path('/kaggle/input').rglob('w11_deep_cortex.py'))
    assert len(hits) == 1, 'expected one W11 code dataset'
    code = str(hits[0].parent)
    sys.path.insert(0, code)
    os.environ['PYTHONPATH'] = code
    import w11_deep_cortex as w11
    w11.kernel_main(143, 'deep_4_open', 9, 12, 'ezzzzzx/w11-deep-4-open-s143-p9')
