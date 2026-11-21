"""H34e private CPU, distill_1, seed 0. Local supervision plus detached mixture self-distillation."""
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h34e_self_distill.py'))
assert len(code)==1,code
sys.path.insert(0,str(code[0].parent))
import h34e_self_distill as h
h.h33.configure(1)
assert h.source_hashes()=={'h34e_self_distill.py': '97a1c2cc30f2e0c558d5a90e75f490c088e61bf091960bcd74f2de3323704c20', 'h34b_plain_depth.py': 'bbd77a82b1e8c05ed19cf32f1a503779032be6cd375944608681e2c9347563b7', 'h34a2_logit_sum.py': '31d9314f3dc1c8eddaa34e245075896503ed6fc919c046e17f79d654f25b7dbe', 'h34a_local_depth.py': 'f1470a25aecd93db1aa915a3ae1be3d917d5363dcad2651daadeaa297bcbfb00', 'h26_cued.py': 'dafc1e4574598f3cb91b48b800fcb556f14d75fe70b7fd16e3784bce4f7d2a40', 'h32_mixer.py': 'c512cab2a41aff0512ead37dcda53d7afa5eb48ca40b52579f468ffc64aca7a1', 'h32b_mixer.py': 'fe78c6c3fc628144e72d836b0b113b9af2c44c03d1ff5d34f1ae3ad0b07a1076', 'h33_bank_text.py': 'c1a52b3cfe7d63b42108adcb8b054e57c4358d553aafdef0b59be6a7c67d380c', 'h20_world.py': 'e2101fcfa7e5580e73142648380abd257c8b8dfeeb9431f3545a55c5537f81e2', 'h30w_ring.py': '6ab4927e244faa18cff258e0bf153fc6e56a4b80139b13a12ff4f7bb6187d1ac', 'h30x_trace.py': '852a9cd9826e4c75687996c33beae14001920bb5cc1d49b1b9b095d821f7c5b9', 'h30y_roots.py': 'fa94014039e3d140265bd456d2090ef20d2a88c3d4ecea89b6b834263f87f214'}, "Attached code differs from smoke"
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
raw,digest=h.h33.load_checked(str(raw_paths[0]))
r=h.run('distill_1',0,raw[:180000])
if 'distill_1'=='plain_4':
    assert all(abs(r[k]-v)<=1e-6 for k,v in h.H34B_REFERENCE[0].items()), 'H34b baseline mismatch'
r.update(kernel_id='ezzzzzx/h34e-distill-1-s0',sha256_e10m=digest,code_fingerprint='a1d1f0cf7be5646d61f7715942f88a25aaae2255a21abe8c688fb519fead8a65')
h.dump('/kaggle/working/results.json',r)
