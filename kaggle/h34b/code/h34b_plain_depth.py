"""H34b: improve the living compressor with independently supervised depth.

Imports H34a2's stateless Layer unchanged. Learned gates start at one per
branch, exactly as H34a2; the fixed ablation starts and stays at 1/D.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import h34a2_logit_sum as prior

CHUNK, EMB, RANK, A, LOG2 = prior.CHUNK, prior.EMB, prior.RANK, prior.A, prior.LOG2
Layer, dump, h33 = prior.Layer, prior.dump, prior.h33
ARMS = ('plain_1','plain_2','plain_4','plain_6','plain_4_fixmix','plain_4_noskip','wide_6')
N_DATA = 180_000
TAIL_START = 40_000
EARLY_BITS_LIMIT = 12


def build(arm, seed, zero=True):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(seed)
    n = 768 if arm == 'wide_6' else 128
    depth = 1 if arm == 'wide_6' else int(arm.split('_')[1])
    return [Layer(n,0 if k==0 else 2*n+RANK,not arm.endswith('noskip'),zero) for k in range(depth)]


def compute_account(arm):
    # H34a2 accounting with actual H34b dimensions and fixed-gate savings.
    rows=[]
    for m in build(arm,0):
        n,u,a,r=m.n,m.dim,A,RANK
        e=EMB if m.has_emb else 0
        p=sum(x.numel() for x in m.parameters())
        terms=dict(inject=2*n*u,state=4*n,trace_lambda=4*n,trace_B=4*n*u,
                   trace_embedding=4*n*a if e else 0,readouts=3*(2*n*a+4*n*r+a*r),
                   quadratic=3*r,grad_roots=16*n,grad_B=8*n*u,
                   grad_embedding_coeff=4*n*a if e else 0,grad_embedding=4*n*a*e,
                   dlam=4*n/CHUNK,softmax=3*a,normalize=2*m.prev+2 if m.prev else 0,adam=6*p/CHUNK)
        rows.append(dict(parameters=p,input_dim=u,terms=terms,macs_per_byte=sum(terms.values()),
                         trace_floats=2*n*(1+u+(a if e else 0)),state_floats=2*n))
    d=len(rows); frozen=arm.endswith('fixmix')
    combine=d*A+3*A if frozen else 2*d*A+3*A+6*d/CHUNK
    return dict(parameters=sum(x['parameters'] for x in rows)+(0 if frozen else d),
                frozen_mixer_constants=d if frozen else 0,macs_per_byte=sum(x['macs_per_byte'] for x in rows)+combine,
                trace_floats=sum(x['trace_floats'] for x in rows),state_floats=sum(x['state_floats'] for x in rows),
                chunk_trace_cache_floats=CHUNK*sum(x['trace_floats'] for x in rows),
                mixer_macs_per_byte=combine,layers=rows)


def run(arm, seed, raw, keep=False, slice_size=25_000):
    t0 = time.perf_counter()
    tokens = torch.tensor(np.array(raw,dtype=np.int64))
    layers = build(arm,seed)
    states = [m.initial() for m in layers]
    opts = [torch.optim.Adam(m.parameters(),lr=1e-3) for m in layers]
    plain = True
    frozen = arm.endswith('fixmix')
    mixer = prior.base.h32b.GeoMixer(arm,1,len(layers),torch.full((1,len(layers)),1/len(layers)) if frozen else torch.ones(1,len(layers)))
    if frozen:
        mixer.W.requires_grad_(False)
    initial_weights = mixer.W.detach().clone()
    max_frozen_weight_change = 0.0
    weight_tail = np.zeros(len(layers))
    weight_tail_n = 0
    bits = np.empty(len(raw),np.float64)
    own = np.empty((len(raw),len(layers)),np.float64)
    nslices = math.ceil(len(raw)/slice_size)
    logsum = np.zeros((nslices,len(layers)))
    logmax = np.zeros_like(logsum)
    weight_sum = np.zeros(len(layers))
    weight_n = 0
    for start in range(0,len(raw),CHUNK):
        end = min(start+CHUNK,len(raw))
        target = tokens[start:end]
        outputs, feats, caches = [], [], []
        for k,m in enumerate(layers):
            u = m.inputs(tokens,start,end,feats[-1] if k else None)
            z,f,states[k],cache = m.score(u,tokens,start,states[k])
            outputs.append(z); feats.append(f); caches.append(cache)
        stacked = torch.stack([z.detach() for z in outputs],1)
        combined = (stacked*mixer.W[0,:,None]).sum(1) if plain else stacked.sum(1)
        losses = F.cross_entropy(combined,target,reduction='none')
        bits[start:end] = losses.detach().double().numpy()/LOG2
        for k,z in enumerate(outputs):
            own[start:end,k] = F.cross_entropy(z.detach(),target,reduction='none').double().numpy()/LOG2
            for idx in range(start//slice_size,(end-1)//slice_size+1):
                a,b = max(start,idx*slice_size)-start,min(end,(idx+1)*slice_size)-start
                absolute = z[a:b].detach().abs()
                logsum[idx,k] += float(absolute.double().sum())
                logmax[idx,k] = max(logmax[idx,k],float(absolute.max()))
        if plain:
            take = max(0,end-max(start,len(raw)-50_000))
            weight_sum += mixer.W.detach().numpy()[0]*take
            weight_n += take
            take_tail = max(0,end-max(start,TAIL_START))
            weight_tail += mixer.W.detach().numpy()[0]*take_tail
            weight_tail_n += take_tail
        err = combined.detach().softmax(-1)
        err[torch.arange(end-start),target] -= 1
        for k,m in enumerate(layers):
            g = outputs[k].detach().softmax(-1)
            g[torch.arange(end-start),target] -= 1
            m.learn(outputs[k],g/(end-start),caches[k])
        if not frozen:
            mixer.opt.zero_grad(set_to_none=True)
            losses.mean().backward()
        for m,opt in zip(layers,opts):
            nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step()
        if not frozen:
            mixer.opt.step()
        if frozen:
            max_frozen_weight_change = max(max_frozen_weight_change,float((mixer.W-initial_weights).abs().max()))
            if not torch.equal(mixer.W,initial_weights):
                raise RuntimeError('Frozen weights changed')
        if not (np.isfinite(bits[start:end]).all() and np.isfinite(own[start:end]).all()
                and torch.isfinite(stacked).all()):
            raise FloatingPointError('nonfinite score/logits')
        if end % 50_048 == 0:
            print(arm,seed,end,float(bits[:end].mean()),time.perf_counter()-t0,flush=True)
        if time.perf_counter()-t0 > 5100:
            raise TimeoutError('85 minute limit')
    slices = [dict(start=i,end=min(i+slice_size,len(raw)),bits_per_byte=float(bits[i:i+slice_size].mean()))
              for i in range(0,len(raw),slice_size)]
    diagnostics = []
    for k,m in enumerate(layers):
        row = dict(own_bits_per_byte=float(own[:,k].mean()),
                   fraction_lambda_lt_half=float((m.lam().abs()<.5).float().mean()),
                   slices=[dict(start=s['start'],end=s['end'],own_bits_per_byte=float(own[s['start']:s['end'],k].mean()),
                                mean_abs_logit=float(logsum[j,k]/((s['end']-s['start'])*A)),
                                max_abs_logit=float(logmax[j,k])) for j,s in enumerate(slices)])
        if plain:
            row['mean_mixer_weight_last_50kb'] = float(weight_sum[k]/weight_n)
            row['mean_mixer_weight_tail'] = float(weight_tail[k]/weight_tail_n) if weight_tail_n else None
            row['final_mixer_weight'] = float(mixer.W.detach()[0,k])
            row['tail_own_bits_per_byte'] = float(own[TAIL_START:,k].mean()) if len(raw)>TAIL_START else None
        diagnostics.append(row)
    edge_end = ((TAIL_START + slice_size - 1) // slice_size) * slice_size
    tail_edge = float(bits[TAIL_START:edge_end].mean()) if len(raw) > edge_end else None
    result = dict(arm=arm,seed=seed,n_bytes=len(raw),bits_per_byte=float(bits.mean()),slices=slices,
                  tail_start=TAIL_START,
                  tail_bits_per_byte=float(bits[TAIL_START:].mean()) if len(raw)>TAIL_START else None,
                  tail_edge_bits_per_byte=tail_edge,
                  layers=diagnostics,max_frozen_weight_change=max_frozen_weight_change,wall_seconds=time.perf_counter()-t0)
    return (result,bits) if keep else result


def source_hashes():
    import hashlib
    root=Path(__file__).resolve().parent
    names=('h34b_plain_depth.py','h34a2_logit_sum.py','h34a_local_depth.py','h26_cued.py',
           'h32_mixer.py','h32b_mixer.py','h33_bank_text.py','h20_world.py',
           'h30w_ring.py','h30x_trace.py','h30y_roots.py')
    return {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}


def smoke(out,data):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    payload=dict(ok=False,n_bytes=N_DATA,threads=1,log=[],source_hashes=source_hashes())
    def record(check,**values):
        row=dict(check=check,**values); payload['log'].append(row)
        print(json.dumps(row),flush=True); dump(out/'smoke.json',payload)
    try:
        raw,digest=h33.load_checked(data); payload['sha256_e10m']=digest
        for seed in (0,1):
            ref,rb=prior.run('plain_4',seed,raw[:8192],keep=True)
            ours,ob=run('plain_4',seed,raw[:8192],keep=True)
            diff=float(np.abs(rb-ob).max())
            record('h34a2_equivalence',seed=seed,max_bits_abs=diff,limit=1e-9,
                   reference_bits=ref['bits_per_byte'],bits=ours['bits_per_byte'])
            assert diff<=1e-9
        fixed=run('plain_4_fixmix',0,raw[:8192])
        weights=[r['final_mixer_weight'] for r in fixed['layers']]
        record('fixed_mixer',weights=weights,max_change=fixed['max_frozen_weight_change'],n_bytes=8192)
        assert weights==[.25]*4 and fixed['max_frozen_weight_change']==0
        _,short=run('plain_6',0,raw[:1024],keep=True)
        _,long=run('plain_6',0,raw[:2048],keep=True)
        diff=float(np.abs(short-long[:1024]).max())
        # Change all future bytes as well as the total length; prefix must agree.
        changed=raw[:2048].copy(); changed[1024:]=255-changed[1024:]
        _,altered=run('plain_6',0,changed,keep=True)
        future_diff=float(np.abs(short-altered[:1024]).max())
        record('prequential_prefix',max_bits_abs=diff,changed_suffix_max_bits_abs=future_diff,
               prefix_bytes=1024,extended_bytes=2048)
        assert diff==future_diff==0
        timings={}
        for arm in ('plain_6','wide_6'):
            r,b=run(arm,0,raw[:16384],keep=True,slice_size=2048)
            early=float(b[:2048].mean())
            timings[arm]=r['wall_seconds']/len(b)
            record('stability_and_timing',arm=arm,n_bytes=len(b),first_2048_bits=early,
                   wall_seconds=r['wall_seconds'],estimate_250kb=timings[arm]*250_000,limit_bits=EARLY_BITS_LIMIT)
            assert early<=EARLY_BITS_LIMIT
        n=N_DATA
        costs={a:compute_account(a)['macs_per_byte'] for a in ARMS}
        estimates={a:timings['plain_6']*n*costs[a]/costs['plain_6'] for a in ARMS}
        estimates['wide_6']=timings['wide_6']*n
        payload['n_bytes']=n
        record('wall_estimates',n_bytes=n,tail_start=TAIL_START,tail_end=n,
               estimated_kernel_seconds=estimates,limit=5100,
               method='Measured 16,384-byte one-thread rates for plain_6 and wide_6; remaining arms MAC-scaled from plain_6; no assumed Kaggle speedup')
        assert max(estimates.values())<=5100
        payload['ok']=True; dump(out/'smoke.json',payload)
    except BaseException as exc:
        payload['failure']=repr(exc); dump(out/'smoke.json',payload); raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true'); p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,choices=(0,1),default=0)
    p.add_argument('--data',default='data/enwik8/enwik8'); p.add_argument('--out',default='results/h34b')
    args=p.parse_args(); h33.configure(1)
    if args.smoke: return smoke(args.out,args.data)
    if args.arm is None: p.error('--arm is required')
    sm=json.loads((Path(args.out)/'smoke.json').read_text()); assert sm['ok']
    assert sm['source_hashes']==source_hashes()
    raw,digest=h33.load_checked(args.data)
    r=run(args.arm,args.seed,raw[:sm['n_bytes']])
    r.update(sha256_e10m=digest,kernel_id=f'ezzzzzx/h34b-{args.arm.replace("_","-")}-s{args.seed}')
    dump(Path(args.out)/f'{args.arm}-s{args.seed}.json',r)

if __name__=='__main__': main()
