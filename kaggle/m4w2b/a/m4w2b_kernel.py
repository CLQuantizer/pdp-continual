"""M4-W2b closed loop, world seeds 9 and 10. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('m4w2b.py'))
    if not hits:
        raise SystemExit('m4w2b.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import m4w2b
    m4w2b.kernel_main(
        out='/kaggle/working',
        world_seeds=(9, 10),
        kernel_id='ezzzzzx/m4w2b-a',
    )


if __name__ == '__main__':
    main()
