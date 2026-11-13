"""H30w seed index 2. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('h30w_ring.py'))
    if not hits:
        raise SystemExit('h30w_ring.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import h30w_ring
    h30w_ring.kernel_main(
        out='/kaggle/working',
        seed_index=2,
        kernel_id='ezzzzzx/h30w-s2',
    )


if __name__ == '__main__':
    main()
