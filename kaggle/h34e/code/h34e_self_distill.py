"""H34e: improve the living compressor by broadcasting detached mixture targets.

H34b's local trace learner and chunk update protocol are unchanged. Every
bank/readout uses its own logits; only its output-space error changes.
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
import h34b_plain_depth as prior

CHUNK, EMB, RANK, A, LOG2 = prior.CHUNK, prior.EMB, prior.RANK, prior.A, prior.LOG2
N_DATA, TAIL_START = prior.N_DATA, prior.TAIL_START
Layer, dump, h33 = prior.Layer, prior.dump, prior.h33
ARMS = ('plain_4','distill_4','distill_4_top','distill_1')
# H34b scored reference, retained here so every fetched baseline is checked.
H34B_REFERENCE = {
    0: dict(tail_bits_per_byte=3.091224982,bits_per_byte=3.745828744),
    1: dict(tail_bits_per_byte=3.091462939,bits_per_byte=3.769168530),
}


def layer_lambda(arm,k,depth,value=0.5):
    return 0.0 if arm=='plain_4' or (arm=='distill_4_top' and k==depth-1) else value


def local_error(logits,target,teacher,lam):
    """Unreduced logit gradient; the caller divides by the chunk length."""
    p = logits.detach().softmax(-1)
    own = p.clone()
    own[torch.arange(len(target)),target] -= 1
    if lam == 0:
        return own  # Preserve the exact H34b arithmetic for baseline/zero smoke.
    return (1-lam)*own + lam*(p-teacher.detach())


def run(arm, seed, raw, keep=False, slice_size=25_000, distill_lambda=0.5):
    t0 = time.perf_counter()
    tokens = torch.tensor(np.array(raw,dtype=np.int64))
    layers = prior.build('plain_1' if arm == 'distill_1' else 'plain_4',seed)
    if arm not in ARMS or not 0 <= distill_lambda <= 1:
        raise ValueError((arm,distill_lambda))
    states = [m.initial() for m in layers]
    opts = [torch.optim.Adam(m.parameters(),lr=1e-3) for m in layers]
    plain = True
    frozen = arm.endswith('fixmix')
    mixer = prior.prior.base.h32b.GeoMixer(arm,1,len(layers),torch.full((1,len(layers)),1/len(layers)) if frozen else torch.ones(1,len(layers)))
    if frozen:
        mixer.W.requires_grad_(False)
    initial_weights = mixer.W.detach().clone()
    max_frozen_weight_change = 0.0
    weight_full = np.zeros(len(layers))
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
            weight_full += mixer.W.detach().numpy()[0]*(end-start)
            take = max(0,end-max(start,len(raw)-50_000))
            weight_sum += mixer.W.detach().numpy()[0]*take
            weight_n += take
            take_tail = max(0,end-max(start,TAIL_START))
            weight_tail += mixer.W.detach().numpy()[0]*take_tail
            weight_tail_n += take_tail
        # Snapshot the broadcast teacher before ANY layer learns. The mixture
        # graph is used only for its own mixer CE; no teacher graph reaches a bank.
        teacher = combined.detach().softmax(-1)
        for k,m in enumerate(layers):
            lam = layer_lambda(arm,k,len(layers),distill_lambda)
            g = local_error(outputs[k],target,teacher,lam)
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
            row['mean_mixer_weight_full'] = float(weight_full[k]/len(raw))
            row['mean_mixer_weight_last_50kb'] = float(weight_sum[k]/weight_n)
            row['mean_mixer_weight_tail'] = float(weight_tail[k]/weight_tail_n) if weight_tail_n else None
            row['final_mixer_weight'] = float(mixer.W.detach()[0,k])
            row['tail_own_bits_per_byte'] = float(own[TAIL_START:,k].mean()) if len(raw)>TAIL_START else None
        diagnostics.append(row)
    edge_end = ((TAIL_START + slice_size - 1) // slice_size) * slice_size
    tail_edge = float(bits[TAIL_START:edge_end].mean()) if len(raw) > edge_end else None
    result = dict(arm=arm,seed=seed,distill_lambda=0.0 if arm=='plain_4' else distill_lambda,
                  layer_lambdas=[layer_lambda(arm,k,len(layers),distill_lambda) for k in range(len(layers))],n_bytes=len(raw),bits_per_byte=float(bits.mean()),slices=slices,
                  tail_start=TAIL_START,
                  tail_bits_per_byte=float(bits[TAIL_START:].mean()) if len(raw)>TAIL_START else None,
                  tail_edge_bits_per_byte=tail_edge,
                  layers=diagnostics,max_frozen_weight_change=max_frozen_weight_change,wall_seconds=time.perf_counter()-t0)
    return (result,bits) if keep else result


def source_hashes():
    import hashlib
    root=Path(__file__).resolve().parent
    hashes=prior.source_hashes()
    hashes['h34e_self_distill.py']=hashlib.sha256((root/'h34e_self_distill.py').read_bytes()).hexdigest()
    return hashes


def gradient_smoke(raw):
    """Real 512-byte stack logits, nonzero readouts, all layers and top ablation."""
    tokens=torch.tensor(np.array(raw[:512],dtype=np.int64))
    layers=[m.double() for m in prior.build('plain_4',0,zero=False)]
    outputs=[]; lower=None
    for m in layers:
        z,lower,_,_=m.score(m.inputs(tokens,0,len(tokens),lower),tokens,0,m.initial())
        outputs.append(z)
    weights=torch.tensor([.35,.20,.15,.30],dtype=torch.float64)
    teacher=(torch.stack([z.detach() for z in outputs],1)*weights[None,:,None]).sum(1).softmax(-1)
    assert teacher.dtype==torch.float64 and not teacher.requires_grad
    errors=[]
    for arm in ('plain_4','distill_4','distill_4_top'):
        for k,original in enumerate(outputs):
            z=original.detach().requires_grad_()
            lam=layer_lambda(arm,k,len(layers))
            loss=(1-lam)*F.cross_entropy(z,tokens,reduction='sum')-lam*(teacher*F.log_softmax(z,-1)).sum()
            expected,=torch.autograd.grad(loss,z)
            errors.append(float((local_error(z,tokens,teacher,lam)-expected).abs().max()))
    # The distillation term is exactly zero if the teacher equals this layer.
    self_term=float((outputs[0].detach().softmax(-1)-outputs[0].detach().softmax(-1)).abs().max())
    return dict(n_bytes=512,dtype='float64',max_abs=max(errors),limit=1e-6,
                equal_distribution_distillation_max_abs=self_term,layer_cases=len(errors))


def smoke(out,data):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    payload=dict(ok=False,n_bytes=N_DATA,tail_start=TAIL_START,threads=1,log=[],source_hashes=source_hashes())
    def record(check,**values):
        row=dict(check=check,**values); payload['log'].append(row)
        print(json.dumps(row),flush=True); dump(out/'smoke.json',payload)
    try:
        raw,digest=h33.load_checked(data); payload['sha256_e10m']=digest
        grad=gradient_smoke(raw); record('logit_gradient_autograd',**grad)
        assert grad['max_abs']<=1e-6 and grad['equal_distribution_distillation_max_abs']==0
        timings={}
        for seed in (0,1):
            ref,rb=prior.run('plain_4',seed,raw[:8192],keep=True)
            ours,ob=run('plain_4',seed,raw[:8192],keep=True)
            _,zb=run('distill_4',seed,raw[:8192],keep=True,distill_lambda=0)
            diff=float(np.abs(rb-ob).max()); zero_diff=float(np.abs(ob-zb).max())
            record('lambda_zero_and_h34b_equivalence',seed=seed,n_bytes=8192,
                   h34b_max_bits_abs=diff,lambda_zero_max_bits_abs=zero_diff,limit=0,
                   bits_per_byte=ours['bits_per_byte'])
            assert diff==zero_diff==0
            timings['plain_4']=max(timings.get('plain_4',0),ours['wall_seconds']/8192*N_DATA)
        for arm in ARMS:
            if arm!='plain_4':
                r,b=run(arm,0,raw[:8192],keep=True)
                assert np.isfinite(b).all()
                timings[arm]=r['wall_seconds']/8192*N_DATA
                record('finite',arm=arm,n_bytes=8192,all_finite=True,bits_per_byte=r['bits_per_byte'],wall_seconds=r['wall_seconds'])
            _,short=run(arm,0,raw[:1024],keep=True)
            _,long=run(arm,0,raw[:2048],keep=True)
            changed=raw[:2048].copy(); changed[1024:]=255-changed[1024:]
            _,altered=run(arm,0,changed,keep=True)
            diff=float(np.abs(short-long[:1024]).max()); changed_diff=float(np.abs(short-altered[:1024]).max())
            record('prequential_prefix',arm=arm,prefix_bytes=1024,extended_bytes=2048,
                   max_bits_abs=diff,changed_suffix_max_bits_abs=changed_diff)
            assert diff==changed_diff==0
        record('wall_estimates',n_bytes=N_DATA,estimated_kernel_seconds=timings,limit=5100,
               method='8192-byte one-thread rates; linear extrapolation; no assumed Kaggle speedup')
        assert max(timings.values())<=5100
        payload['ok']=True; dump(out/'smoke.json',payload)
    except BaseException as exc:
        payload['failure']=repr(exc); dump(out/'smoke.json',payload); raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true'); p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,choices=(0,1),default=0)
    p.add_argument('--data',default='data/enwik8/enwik8'); p.add_argument('--out',default='results/h34e')
    args=p.parse_args(); h33.configure(1)
    if args.smoke: return smoke(args.out,args.data)
    if args.arm is None: p.error('--arm is required')
    sm=json.loads((Path(args.out)/'smoke.json').read_text()); assert sm['ok']
    assert sm['source_hashes']==source_hashes()
    raw,digest=h33.load_checked(args.data)
    r=run(args.arm,args.seed,raw[:N_DATA])
    if args.arm=='plain_4':
        assert all(abs(r[k]-v)<=1e-6 for k,v in H34B_REFERENCE[args.seed].items())
    r.update(sha256_e10m=digest,kernel_id=f'ezzzzzx/h34e-{args.arm.replace("_","-")}-s{args.seed}')
    dump(Path(args.out)/f'{args.arm}-s{args.seed}.json',r)

if __name__=='__main__': main()
