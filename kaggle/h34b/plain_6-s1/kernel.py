"""H34b private CPU, plain_6, seed 1. Own-loss stack with learned gate."""
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
code=sorted(Path('/kaggle/input').rglob('h34b_plain_depth.py'))
assert len(code)==1,code
sys.path.insert(0,str(code[0].parent))
import h34b_plain_depth as h
h.h33.configure(1)
assert h.source_hashes()=={'h34b_plain_depth.py': 'd7be431fa05bd064615a5b8ba6e9b54cb2f7e23ada72e2eb841973562794fbb5', 'h34a2_logit_sum.py': '31d9314f3dc1c8eddaa34e245075896503ed6fc919c046e17f79d654f25b7dbe', 'h34a_local_depth.py': 'f1470a25aecd93db1aa915a3ae1be3d917d5363dcad2651daadeaa297bcbfb00', 'h26_cued.py': 'dafc1e4574598f3cb91b48b800fcb556f14d75fe70b7fd16e3784bce4f7d2a40', 'h32_mixer.py': 'c512cab2a41aff0512ead37dcda53d7afa5eb48ca40b52579f468ffc64aca7a1', 'h32b_mixer.py': 'fe78c6c3fc628144e72d836b0b113b9af2c44c03d1ff5d34f1ae3ad0b07a1076', 'h33_bank_text.py': 'c1a52b3cfe7d63b42108adcb8b054e57c4358d553aafdef0b59be6a7c67d380c', 'h20_world.py': 'e2101fcfa7e5580e73142648380abd257c8b8dfeeb9431f3545a55c5537f81e2', 'h30w_ring.py': '6ab4927e244faa18cff258e0bf153fc6e56a4b80139b13a12ff4f7bb6187d1ac', 'h30x_trace.py': '852a9cd9826e4c75687996c33beae14001920bb5cc1d49b1b9b095d821f7c5b9', 'h30y_roots.py': 'fa94014039e3d140265bd456d2090ef20d2a88c3d4ecea89b6b834263f87f214'}, "Attached code differs from smoke"
raw_paths=[p for p in Path('/kaggle/input').rglob('enwik8') if p.is_file() and p.stat().st_size==100_000_000]
assert len(raw_paths)==1,raw_paths
raw,digest=h.h33.load_checked(str(raw_paths[0]))
r=h.run('plain_6',1,raw[:200000])
r.update(kernel_id='ezzzzzx/h34b-plain-6-s1',sha256_e10m=digest,code_fingerprint='46d917ec35c7f4a2b238dfcba7695e6801c85b0e3b0719ac18a05981e88ffda0')
h.dump('/kaggle/working/results.json',r)
