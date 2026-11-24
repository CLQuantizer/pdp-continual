"""Private one-thread CPU run. Offline preparation prepends CONFIG."""
import os
from pathlib import Path
import sys
import json
import hashlib
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
code = sorted(Path('/kaggle/input').rglob('h34i_generality.py'))
assert len(code) == 1, code
sys.path.insert(0, str(code[0].parent))
import h34i_generality as h
h.h33.configure(1)
assert h.source_hashes() == CONFIG['hashes'], 'Attached code differs from smoke'
paths = list(Path('/kaggle/input').rglob(CONFIG['stream']+'.bin'))
assert len(paths) == 1, paths
raw = h.load_stream(CONFIG['stream'], paths[0].parent, CONFIG['streams'])
digest = CONFIG['streams'][CONFIG['stream']]['sha256']
kwargs = {}
if CONFIG['part'] == 1:
    kwargs = dict(checkpoint='/kaggle/working/checkpoint.pt', stop_at=90000)
elif CONFIG['part'] == 2:
    paths = list(Path('/kaggle/input').rglob('checkpoint.pt'))
    assert len(paths) == 1, paths
    checkpoint = paths[0]
    parent = json.loads((checkpoint.parent/'results.json').read_text())
    assert (parent['kernel_id'], parent['code_fingerprint'], parent['raw_sha256']) == (
        CONFIG['parent_id'], CONFIG['fingerprint'], digest)
    assert (parent['stream'], parent['arm'], parent['seed'], parent['n_bytes']) == (
        CONFIG['stream'], CONFIG['arm'], CONFIG['seed'], CONFIG['documents'])
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash == parent['checkpoint_sha256']
    kwargs = dict(resume=checkpoint)
r = h.run(CONFIG['arm'], CONFIG['seed'], raw[:CONFIG['documents']], **kwargs)
if CONFIG['part'] == 2:
    r['resumed_checkpoint_sha256'] = checkpoint_hash
r.update(stream=CONFIG['stream'], raw_sha256=digest, kernel_id=CONFIG['kernel_id'],
         code_fingerprint=CONFIG['fingerprint'], part=CONFIG['part'])
h.dump('/kaggle/working/results.json', r)
