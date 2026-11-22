CONFIG = {'arm': 'dense_4', 'seed': 0, 'part': 0, 'documents': 180000, 'hashes': {'h34h_dense.py': 'f089126ca805b2fb2184447f7bb9006d96a80b1489fa712324c33eae40b60e04', 'h34g_depth.py': '74250cda53a21ec1e2852c1bd399c768a8b75a729146a3bd9beb16b96a6e74ee', 'h34f_zero_gate.py': '8e76e01a53e33d0d95d73ab75dfaf1b854eb6603c38cfb9b849bee07cd645008', 'h34b_plain_depth.py': 'bbd77a82b1e8c05ed19cf32f1a503779032be6cd375944608681e2c9347563b7', 'h34a2_logit_sum.py': '31d9314f3dc1c8eddaa34e245075896503ed6fc919c046e17f79d654f25b7dbe', 'h34a_local_depth.py': 'f1470a25aecd93db1aa915a3ae1be3d917d5363dcad2651daadeaa297bcbfb00', 'h26_cued.py': 'dafc1e4574598f3cb91b48b800fcb556f14d75fe70b7fd16e3784bce4f7d2a40', 'h32_mixer.py': 'c512cab2a41aff0512ead37dcda53d7afa5eb48ca40b52579f468ffc64aca7a1', 'h32b_mixer.py': 'fe78c6c3fc628144e72d836b0b113b9af2c44c03d1ff5d34f1ae3ad0b07a1076', 'h33_bank_text.py': 'c1a52b3cfe7d63b42108adcb8b054e57c4358d553aafdef0b59be6a7c67d380c', 'h20_world.py': 'e2101fcfa7e5580e73142648380abd257c8b8dfeeb9431f3545a55c5537f81e2', 'h30w_ring.py': '6ab4927e244faa18cff258e0bf153fc6e56a4b80139b13a12ff4f7bb6187d1ac', 'h30x_trace.py': '852a9cd9826e4c75687996c33beae14001920bb5cc1d49b1b9b095d821f7c5b9', 'h30y_roots.py': 'fa94014039e3d140265bd456d2090ef20d2a88c3d4ecea89b6b834263f87f214'}, 'fingerprint': '068ee92b550d984003b9ae9b2b495de9ad36c17159a28a348d7bbbd6cd8ebcfd', 'kernel_id': 'ezzzzzx/h34h-dense-4-s0', 'parent_id': None}
"""Private, one-thread CPU run. CONFIG is prepended by offline preparation."""
import os
from pathlib import Path
import sys
import json
import hashlib
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h34h_dense.py'))
assert len(code)==1,code
sys.path.insert(0,str(code[0].parent))
import h34h_dense as h
h.h33.configure(1)
assert h.source_hashes()==CONFIG['hashes'], 'Attached code differs from smoke'
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
raw,digest=h.h33.load_checked(str(raw_paths[0]))
kwargs={}
if CONFIG['part']==1:
    kwargs=dict(checkpoint='/kaggle/working/checkpoint.pt',stop_at=90000)
elif CONFIG['part']==2:
    paths=list(Path('/kaggle/input').rglob('checkpoint.pt'))
    assert len(paths)==1,paths
    checkpoint=paths[0]
    parent=json.loads((checkpoint.parent/'results.json').read_text())
    assert (parent['kernel_id'],parent['code_fingerprint'],parent['sha256_e10m']) == (CONFIG['parent_id'],CONFIG['fingerprint'],digest)
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash==parent['checkpoint_sha256']
    kwargs=dict(resume=checkpoint)
r=h.run(CONFIG['arm'],CONFIG['seed'],raw[:CONFIG['documents']],**kwargs)
if CONFIG['part']!=1: h.check_reference(r)
if CONFIG['part']==2: r['resumed_checkpoint_sha256']=checkpoint_hash
r.update(kernel_id=CONFIG['kernel_id'],sha256_e10m=digest,code_fingerprint=CONFIG['fingerprint'],part=CONFIG['part'])
h.dump('/kaggle/working/results.json',r)
