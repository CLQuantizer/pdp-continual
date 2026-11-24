"""H34h: improve the living compressor; dense depth and gate ablations.

Use an isolated globals binding of H34g's unchanged training/checkpoint loop.
No mutation of imported modules; own-loss learning and operation order are exact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import torch
import h34g_depth as prior

CHUNK, EMB, RANK = prior.CHUNK, prior.EMB, prior.RANK
N_DATA, TAIL_START = prior.N_DATA, prior.TAIL_START
h33, dump = prior.h33, prior.dump
ARMS = ('dense_8', 'dense_4', 'dense_6_ones', 'dense_6')


def config(arm):
    if arm not in ARMS: raise ValueError(arm)
    depth = int(arm.split('_')[1])
    return dict(depth=depth, dense=True, n=128, rank=RANK, emb=EMB,
                chunk=CHUNK, dtype='float32', learning='trace', loss='own',
                optimizer='Adam', lr=1e-3, clip=1.,
                mixer_initial=[1.] * depth if arm.endswith('_ones') else [1.] + [0.] * (depth-1))


def build(arm, seed):
    torch.manual_seed(seed)
    return [prior.Layer(128, 0 if k == 0 else 256+RANK, True, True)
            for k in range(config(arm)['depth'])]


def initial_gate(arm):
    return torch.tensor([config(arm)['mixer_initial']], dtype=torch.float32)


def source_hashes():
    return {'h34h_dense.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **prior.source_hashes()}


def run(arm, seed, raw, keep=False, slice_size=20_000, *, capture=None, **kwargs):
    config(arm)
    def construct(a, s):
        layers = build(a, s)
        if capture is not None:
            for k, m in enumerate(layers[:4]):
                score = m.score
                def observed(*args, _score=score, _k=k, **kw):
                    result = _score(*args, **kw)
                    capture.append((_k, result[0].detach().clone()))
                    return result
                m.score = observed
        return layers
    bindings = dict(prior.run.__globals__, build=construct, initial_gate=initial_gate,
                    source_hashes=source_hashes)
    runner = types.FunctionType(prior.run.__code__, bindings, prior.run.__name__, prior.run.__defaults__)
    runner.__kwdefaults__ = prior.run.__kwdefaults__
    return runner(arm, seed, raw, keep, slice_size, **kwargs)


def check_reference(r):
    if r['arm'] == 'dense_6' and r['n_bytes'] == N_DATA:
        refs = {0: (3.320349905, 3.016241787, 4.384728317),
                1: (3.330644108, 3.016160666, 4.431336156)}
        for key, value in zip(('bits_per_byte','tail_bits_per_byte','first_40kb_bits_per_byte'), refs[r['seed']]):
            assert abs(r[key]-value) <= 1e-6, 'H34g mismatch'


def equal(a, b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def smoke(out, data):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    payload=dict(ok=False,n_bytes=N_DATA,threads=torch.get_num_threads(),source_hashes=source_hashes(),log=[])
    def record(check,**values):
        row=dict(check=check,**values); payload['log'].append(row)
        print(json.dumps(row),flush=True); dump(out/'smoke.json',payload)
    try:
        assert torch.get_num_threads()==1
        raw,digest=h33.load_checked(data); payload['sha256_e10m']=digest
        configs=[config(a) for a in ('dense_6','dense_6_ones')]
        diff={k:[a[k] for a in configs] for k in configs[0] if configs[0][k]!=configs[1][k]}
        assert list(diff)==['mixer_initial']
        record('config_diff',configs=dict(zip(('dense_6','dense_6_ones'),configs)),diff=diff)
        timings={a:[] for a in ARMS}
        for seed in (0,1):
            ref,rb=prior.run('dense_6',seed,raw[:8192],keep=True)
            ours,ob=run('dense_6',seed,raw[:8192],keep=True)
            assert np.array_equal(rb,ob)
            record('h34g_equivalence',seed=seed,n_bytes=8192,max_bits_abs=0.,reference_bits=ref['bits_per_byte'],bits_per_byte=ours['bits_per_byte'])
            timings['dense_6'].append(ours['wall_seconds']/8192)
            captures=[]
            for arm in ('dense_4','dense_6'):
                cap=[]; run(arm,seed,raw[:2048],capture=cap); captures.append(cap)
            assert len(captures[0])==len(captures[1])==64
            assert all(ka==kb and torch.equal(a,b) for (ka,a),(kb,b) in zip(*captures))
            record('lower_four_own_predictions',seed=seed,n_bytes=2048,layers=[1,2,3,4],max_logits_abs=0.)
            _,short=run('dense_8',seed,raw[:1024],keep=True)
            _,long=run('dense_8',seed,raw[:2048],keep=True)
            changed=raw[:2048].copy(); changed[1024:]=255-changed[1024:]
            _,altered=run('dense_8',seed,changed,keep=True)
            assert np.array_equal(short,long[:1024]) and np.array_equal(short,altered[:1024])
            record('prequential_prefix',arm='dense_8',seed=seed,n_bytes=1024,max_bits_abs=0.,changed_suffix_max_bits_abs=0.)
        for arm in ARMS:
            r,b=run(arm,0,raw[:8192],keep=True)
            assert np.isfinite(b).all() and all(np.isfinite(s['mean_weights']).all() for s in r['mixer_slices'])
            timings[arm].append(r['wall_seconds']/8192)
            record('finite',arm=arm,n_bytes=8192,bits_per_byte=r['bits_per_byte'],all_finite=True,wall_seconds=r['wall_seconds'])
        # Test both aligned and actual 90-KB-style unaligned boundaries, both seeds.
        for seed in (0,1):
            whole=out/f'whole-s{seed}.pt'; split=out/f'split-s{seed}.pt'
            r,b=run('dense_8',seed,raw[:8192],keep=True,checkpoint=whole)
            timings['dense_8'].append(r['wall_seconds']/8192)
            for boundary in (4096,4112):
                part=run('dense_8',seed,raw[:8192],checkpoint=split,stop_at=boundary)
                rr,bb=run('dense_8',seed,raw[:8192],keep=True,resume=split,checkpoint=split)
                assert np.array_equal(b,bb)
                assert equal({k:v for k,v in r.items() if 'wall' not in k},{k:v for k,v in rr.items() if 'wall' not in k})
                a=torch.load(whole,weights_only=False); c=torch.load(split,weights_only=False)
                assert equal({k:v for k,v in a.items() if k!='wall_seconds'},{k:v for k,v in c.items() if k!='wall_seconds'})
                record('checkpoint_equivalence',arm='dense_8',seed=seed,n_bytes=8192,split_at=boundary,committed_bytes=part['offset'],pending_bytes=part['pending_bytes'],max_bits_abs=0.,state_exact=True)
            whole.unlink(); split.unlink()
        estimates={a:max(ts)*N_DATA for a,ts in timings.items()}
        payload['split_dense_8']=estimates['dense_8']>5100
        segments={a:v/(2 if a=='dense_8' and payload['split_dense_8'] else 1) for a,v in estimates.items()}
        record('wall_estimates',estimated_kernel_seconds=estimates,estimated_segment_seconds=segments,split_dense_8=payload['split_dense_8'],limit=5100,n_bytes=N_DATA)
        assert max(segments.values())<=5100
        payload['ok']=True; dump(out/'smoke.json',payload)
    except BaseException as exc:
        payload['failure']=repr(exc); dump(out/'smoke.json',payload); raise


def main():
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,choices=(0,1),default=0); p.add_argument('--data',default='data/enwik8/enwik8')
    p.add_argument('--out',default='results/h34h'); p.add_argument('--checkpoint'); p.add_argument('--resume'); p.add_argument('--stop-at',type=int)
    args=p.parse_args(); h33.configure(1)
    if args.smoke: return smoke(args.out,args.data)
    if args.arm is None: p.error('--arm required')
    sm=json.loads((Path(args.out)/'smoke.json').read_text())
    assert sm['ok'] and sm['source_hashes']==source_hashes() and sm['n_bytes']==N_DATA
    raw,digest=h33.load_checked(args.data); assert digest==sm['sha256_e10m']
    r=run(args.arm,args.seed,raw[:N_DATA],checkpoint=args.checkpoint,resume=args.resume,stop_at=args.stop_at)
    if args.stop_at is None: check_reference(r)
    r.update(sha256_e10m=digest,kernel_id=f'ezzzzzx/h34h-{args.arm.replace("_","-")}-s{args.seed}')
    dump(Path(args.out)/f'{args.arm}-s{args.seed}.json',r)

if __name__=='__main__': main()
