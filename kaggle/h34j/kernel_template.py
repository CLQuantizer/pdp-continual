"""Private CPU part; CONFIG is prepended during offline staging."""
import os
from pathlib import Path
import sys
import json
import time
import hashlib
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
started = time.perf_counter()
code = list(Path('/kaggle/input').rglob('h34j_long.py'))
assert len(code) == 1
sys.path.insert(0, str(code[0].parent))
import h34j_long as h
h.h33.configure(1)
assert h.source_hashes() == CONFIG['hashes']
paths = [p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size == 100_000_000]
assert len(paths) == 1
raw, digest = h.h33.load_checked(str(paths[0])); raw = raw[:h.N_DATA]
assert digest == CONFIG['sha256_e10m'] and h.sha(raw.tobytes()) == CONFIG['raw_sha256']
kwargs = dict(checkpoint='/kaggle/working/checkpoint.pt', stop_at=CONFIG['end'])
parent_hash = None
if CONFIG['part'] > 1:
    paths = list(Path('/kaggle/input').rglob('checkpoint.pt')); assert len(paths) == 1
    checkpoint = paths[0]; parent = json.loads((checkpoint.parent/'results.json').read_text())
    assert (parent['kernel_id'], parent['code_fingerprint'], parent['raw_sha256'], parent['arm'], parent['seed'], parent['part'], parent['end']) == (
        CONFIG['parent_id'], CONFIG['fingerprint'], CONFIG['raw_sha256'], CONFIG['arm'], CONFIG['seed'], CONFIG['part']-1, CONFIG['start'])
    parent_hash = h.sha(checkpoint.read_bytes()); assert parent_hash == parent['checkpoint_sha256']
    kwargs['resume'] = checkpoint
r = h.run(CONFIG['arm'], CONFIG['seed'], raw, **kwargs)
# H34g's legacy key denotes the prefix before TAIL_START; expose its correct name.
if 'first_40kb_bits_per_byte' in r:
    r['first_200kb_bits_per_byte'] = r.pop('first_40kb_bits_per_byte')
r.update(kernel_id=CONFIG['kernel_id'], code_fingerprint=CONFIG['fingerprint'],
         raw_sha256=CONFIG['raw_sha256'], sha256_e10m=digest, part=CONFIG['part'],
         start=CONFIG['start'], end=CONFIG['end'], resumed_checkpoint_sha256=parent_hash,
         checkpoint_sha256=h.sha(Path('/kaggle/working/checkpoint.pt').read_bytes()),
         kernel_wall_seconds=time.perf_counter()-started)
assert r['kernel_wall_seconds'] <= 5100
h.dump('/kaggle/working/results.json', r)
