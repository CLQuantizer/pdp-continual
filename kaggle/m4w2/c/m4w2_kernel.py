"""M4-W2 closed loop, world seeds 7 and 8. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('m4w2.py'))
    if not hits:
        raise SystemExit('m4w2.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import m4w2
    m4w2.kernel_main(
        out='/kaggle/working',
        world_seeds=(7, 8),
        kernel_id='ezzzzzx/m4w2-c',
    )


if __name__ == '__main__':
    main()
