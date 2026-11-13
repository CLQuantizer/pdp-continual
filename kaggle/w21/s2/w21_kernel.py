"""W2-1 headroom, world seed 2. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('w21_headroom.py'))
    if not hits:
        raise SystemExit('w21_headroom.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import w21_headroom
    w21_headroom.kernel_main(
        out='/kaggle/working',
        world_seed=2,
        kernel_id='ezzzzzx/w21-s2',
    )


if __name__ == '__main__':
    main()
