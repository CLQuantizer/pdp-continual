"""W1 headroom, bible and chess. Private GPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('w1_headroom.py'))
    if not hits:
        raise SystemExit('w1_headroom.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import w1_headroom
    w1_headroom.kernel_main(
        out='/kaggle/working',
        channels=('bible', 'chess'),
        kernel_id='ezzzzzx/w1-headroom-a',
    )


if __name__ == '__main__':
    main()
