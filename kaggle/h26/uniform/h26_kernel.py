"""H26 arm uniform. Private GPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('h26_cued.py'))
    if not hits:
        raise SystemExit('h26_cued.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import h26_cued
    h26_cued.SEED = 0
    print(
        'seed', h26_cued.SEED,
        'replay_seed', h26_cued.REPLAY_SEED + h26_cued.SEED,
        flush=True)
    h26_cued.kernel_main(
        out='/kaggle/working',
        lr=3e-4,
        replay_batch=8,
        arm='uniform',
        n_bytes=3_000_000,
        kernel_id='ezzzzzx/h26-uniform',
    )


if __name__ == '__main__':
    main()
