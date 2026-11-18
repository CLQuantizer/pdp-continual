import os, sys
from pathlib import Path
os.environ['H35C_MODE'] = 'smoke'
code = sorted(Path('/kaggle/input').rglob('h35c_run.py'))
assert len(code) == 1, code
sys.path.insert(0, str(code[0].parent))
import h35c_run
h35c_run.main()
