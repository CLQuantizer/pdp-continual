"""H30y seed index 1. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('h30y_roots.py'))
    if not hits:
        raise SystemExit('h30y_roots.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import h30y_roots
    h30y_roots.kernel_main(
        out='/kaggle/working',
        seed_index=1,
        kernel_id='ezzzzzx/h30y-s1',
    )


if __name__ == '__main__':
    main()
