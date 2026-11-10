"""H29 enwik8 score. Private GPU kernel, no internet. One online pass."""
import os
import sys
from pathlib import Path


def code_dir():
    hits = sorted(Path('/kaggle/input').rglob('h29_enwik8.py'))
    if not hits:
        raise SystemExit('h29_enwik8.py not found under /kaggle/input')
    return str(hits[0].parent)


def main():
    code = code_dir()
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import h29_enwik8
    h29_enwik8.kernel_main(out='/kaggle/working')


if __name__ == '__main__':
    main()
