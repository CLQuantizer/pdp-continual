"""H32b cerebellar mixer on E10M. Private GPU kernel, no internet.

The GPU image has no pyppmd. Load the 1.3.1 wheel from the code dataset
before the scored stream, so a bad wheel fails in seconds.
"""
import os
import sys
import zipfile
from pathlib import Path


def ensure_pyppmd():
    wheels = sorted(Path('/kaggle/input').rglob('pyppmd-*.whl'))
    if not wheels:
        raise SystemExit('pyppmd wheel not found under /kaggle/input')
    dest = Path('/tmp/pyppmd_pkgs')
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(dest)
    sys.path.insert(0, str(dest))
    import pyppmd
    blob = pyppmd.compress(b'abc', max_order=16, mem_size=1 << 20, variant='H')
    back = pyppmd.Ppmd7Decoder(16, 1 << 20).decode(blob, 3)
    if back != b'abc':
        raise SystemExit('pyppmd probe roundtrip failed')
    print('pyppmd', wheels[0].name, pyppmd.__file__, 'probe_ok', flush=True)


def main():
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    ensure_pyppmd()
    hits = sorted(Path('/kaggle/input').rglob('h32b_mixer.py'))
    if not hits:
        raise SystemExit('h32b_mixer.py not found under /kaggle/input')
    code = str(hits[0].parent)
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import h32b_mixer
    print(
        'seed', 0,
        'replay_seed', h32b_mixer.h26.REPLAY_SEED,
        flush=True)
    h32b_mixer.kernel_main(out='/kaggle/working', kernel_id='ezzzzzx/h32b-mixer')


if __name__ == '__main__':
    main()
