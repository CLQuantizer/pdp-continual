import os, sys
from pathlib import Path
os.environ['H35C_MODE'] = 'scored'
os.environ['H35C_ARMS'] = 'prog_snap,rho,rand_plus,high_loss,rand,heur,dedup,ppl,cls,pipeline'
os.environ['H35C_SIZE'] = 'small'
os.environ['H35C_SEED'] = '1'
os.environ['H35C_STEPS'] = '3000'
os.environ['H35C_DELTA'] = '1441'
code = sorted(Path('/kaggle/input').rglob('h35c_run.py'))
assert len(code) == 1, code
sys.path.insert(0, str(code[0].parent))
import h35c_run
h35c_run.main()
