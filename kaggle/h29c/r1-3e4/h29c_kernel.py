"""H29c arm r1_3e4. Private GPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('h29b_enwik8.py'))
    if not hits:
        raise SystemExit('h29b_enwik8.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import h29b_enwik8
    h29b_enwik8.kernel_main(
        out='/kaggle/working',
        lr=3e-4,
        replay_batch=1,
        arm='r1_3e4',
        n_bytes=10_000_000,
        kernel_id='ezzzzzx/h29c-r1-3e4',
    )


if __name__ == '__main__':
    main()
