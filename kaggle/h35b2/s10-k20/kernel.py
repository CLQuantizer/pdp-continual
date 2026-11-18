"""H35b2 private CPU, seed 10, ['prog_snap_20']."""
import os
import hashlib
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h35b2_snapshot.py'))
assert len(code)==1,code
code_root=code[0].parent
expected_hashes={'h35b2_snapshot.py': '85612d9084f77acc1536e342a57a99df212254764f36fefb5a9ccd764c0b8162', 'h35b_neural_diet.py': '778f717a1ed22bbf99eb6373d0d778f937f8c3c603a7b6cbb52034185034e836', 'h35_corpus.py': '85fec37ac4c47ef9328c4ea6a3c1f2d144f36524bba33a13ddabf2393b377440', 'h26_cued.py': 'dafc1e4574598f3cb91b48b800fcb556f14d75fe70b7fd16e3784bce4f7d2a40'}
for name, expected in expected_hashes.items():
    assert hashlib.sha256((code_root/name).read_bytes()).hexdigest()==expected,name
sys.path.insert(0,str(code_root))
import h35b2_snapshot as h
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
if __name__ == '__main__':
    h.kernel_main(10,'/kaggle/working','ezzzzzx/h35b2-s10-k20',policies=['prog_snap_20'],
                  data=str(raw_paths[0]),fingerprint='591562f99687706c2ea78576444f6b72e4c9173178d55ef82f182083cc700849')
