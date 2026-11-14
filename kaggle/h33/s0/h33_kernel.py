"""H33 seed 0. Private CPU kernel, no internet."""
import os
import sys
from pathlib import Path


def main():
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    hits = sorted(Path('/kaggle/input').rglob('h33_bank_text.py'))
    if not hits:
        raise SystemExit('h33_bank_text.py not found under /kaggle/input')
    code = str(hits[0].parent)
    prev = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code if not prev else code + os.pathsep + prev
    sys.path.insert(0, code)
    print('code_dir', code, flush=True)
    import h33_bank_text
    h33_bank_text.kernel_main(
        out='/kaggle/working',
        seed=0,
        kernel_id='ezzzzzx/h33-bank-text-s0',
    )


if __name__ == '__main__':
    main()
