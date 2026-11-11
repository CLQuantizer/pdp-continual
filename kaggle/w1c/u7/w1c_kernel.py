"""W1c shared cortex, arm U7. Private GPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('w1c_shared.py'))
    if not hits:
        raise SystemExit('w1c_shared.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import w1c_shared
    w1c_shared.kernel_main(
        out='/kaggle/working',
        arm='U7',
        kernel_id='ezzzzzx/w1c-u7',
    )


if __name__ == '__main__':
    main()
