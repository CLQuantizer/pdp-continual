"""H31 arm noreplay. Private GPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    hits = sorted(Path('/kaggle/input').rglob('h31_sleep.py'))
    if not hits:
        raise SystemExit('h31_sleep.py not found under /kaggle/input')
    code = str(hits[0].parent)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import h31_sleep
    h31_sleep.SEED = 0
    print(
        'seed', h31_sleep.SEED,
        'replay_seed', h31_sleep.REPLAY_SEED + h31_sleep.SEED,
        flush=True)
    h31_sleep.kernel_main(
        out='/kaggle/working',
        lr=3e-4,
        replay_batch=0,
        arm='noreplay',
        n_bytes=3_000_000,
        kernel_id='ezzzzzx/h31-noreplay',
        sleep_block=1,
    )


if __name__ == '__main__':
    main()
