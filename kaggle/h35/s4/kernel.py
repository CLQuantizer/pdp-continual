"""Private CPU H35 seed 4 policies ['rand', 'oracle', 'high_loss', 'rho']."""
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

code = sorted(Path("/kaggle/input").rglob("h35_data_diet.py"))
assert len(code) == 1, code
os.environ["PYTHONPATH"] = str(code[0].parent)
sys.path.insert(0, str(code[0].parent))
import h35_data_diet as h

# Kaggle re-executes this file in spawn children. The guard keeps that
# re-exec from starting another pool.
if __name__ == "__main__":
    h.kernel_main(
        4,
        "/kaggle/working",
        'ezzzzzx/h35-s4',
        policies=['rand', 'oracle', 'high_loss', 'rho'],
        budget=800,
        workers=4,
        peek_workers=1,
    )
