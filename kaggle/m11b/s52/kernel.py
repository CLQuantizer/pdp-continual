import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CURIO_LR'] = '0.004'
os.environ['PYTHONHASHSEED'] = '0'
if os.environ.get('_M11B_KERNEL') != '1':
    os.environ['_M11B_KERNEL'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)

def main():
    hits = sorted(Path('/kaggle/input').rglob('m11b_w5.py'))
    if len(hits) != 1:
        raise RuntimeError(f'expected exactly one code dataset, found {len(hits)}')
    code = str(hits[0].parent)
    sys.path.insert(0, code)
    os.environ['PYTHONPATH'] = code
    import m11b_w5
    m11b_w5.kernel_main(52, '/kaggle/working', workers=4, kernel_id='ezzzzzx/m11b-w5-s52', arms=('age_fixed20', 'age_sleeparm', 'age_sleep_cf', 'age_nosleep'), label='')

if __name__ == '__main__':
    main()
