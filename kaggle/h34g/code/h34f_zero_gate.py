"""H34f: improve the living compressor by starting deep mixers at layer one.

H34b layer construction, own-loss trace learning, and update order are retained.
Only the initial GeoMixer weights differ for the registered zgate arms.
"""
import argparse
import hashlib
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
ARMS = ('plain_4','zgate_4','zgate_6','plain_6','plain_1')
H34B_REFERENCE = {
    'plain_1': {0: dict(tail_bits_per_byte=3.264204845,bits_per_byte=3.533424381,first_40kb_bits_per_byte=4.475692753),
                1: dict(tail_bits_per_byte=3.285754554,bits_per_byte=3.565700932,first_40kb_bits_per_byte=4.545513255)},
    'plain_4': {0: dict(tail_bits_per_byte=3.091224982,bits_per_byte=3.745828744,first_40kb_bits_per_byte=6.036941910),
                1: dict(tail_bits_per_byte=3.091462939,bits_per_byte=3.769168530,first_40kb_bits_per_byte=6.141138099)},
    'plain_6': {0: dict(tail_bits_per_byte=3.079927197,bits_per_byte=4.107748392,first_40kb_bits_per_byte=7.705122572),
                1: dict(tail_bits_per_byte=3.082687574,bits_per_byte=4.150529688,first_40kb_bits_per_byte=7.887977086)},
}


def build(arm,seed):
    if arm not in ARMS:
        raise ValueError(arm)
    return prior.build(arm.replace('zgate','plain'),seed)


def initial_gate(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    depth=int(arm.split('_')[1])
    w=torch.ones(1,depth)
    if arm.startswith('zgate'):
        w[:,1:]=0
    return w


def run(arm, seed, raw, keep=False, slice_size=20_000):
    t0 = time.perf_counter()
    tokens = torch.tensor(np.array(raw,dtype=np.int64))
    layers = build(arm,seed)
    states = [m.initial() for m in layers]
    opts = [torch.optim.Adam(m.parameters(),lr=1e-3) for m in layers]
    plain = True
    frozen = arm.endswith('fixmix')
    mixer = prior.prior.base.h32b.GeoMixer(arm,1,len(layers),torch.full((1,len(layers)),1/len(layers)) if frozen else initial_gate(arm))
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
    gate_sums = np.zeros_like(logsum)
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
        for idx in range(start//slice_size,(end-1)//slice_size+1):
            count = min(end,(idx+1)*slice_size)-max(start,idx*slice_size)
            gate_sums[idx] += mixer.W.detach().numpy()[0]*count
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
    tail_edge = float(bits[TAIL_START:edge_end].mean()) if edge_end > TAIL_START and len(raw) > edge_end else None
    result = dict(arm=arm,seed=seed,n_bytes=len(raw),bits_per_byte=float(bits.mean()),slices=slices,
                  tail_start=TAIL_START,
                  tail_bits_per_byte=float(bits[TAIL_START:].mean()) if len(raw)>TAIL_START else None,
                  tail_edge_bits_per_byte=tail_edge,
                  first_40kb_bits_per_byte=float(bits[:TAIL_START].mean()),
                  initial_mixer_weights=initial_weights[0].tolist(),
                  mixer_slices=[dict(start=s['start'],end=s['end'],
                      mean_weights=(gate_sums[j]/(s['end']-s['start'])).tolist()) for j,s in enumerate(slices)],
                  layers=diagnostics,max_frozen_weight_change=max_frozen_weight_change,wall_seconds=time.perf_counter()-t0)
    return (result,bits) if keep else result


def source_hashes():
    root=Path(__file__).resolve().parent
    return {'h34f_zero_gate.py':hashlib.sha256((root/'h34f_zero_gate.py').read_bytes()).hexdigest(),
            **prior.source_hashes()}


def smoke(out,data):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    payload=dict(ok=False,n_bytes=N_DATA,threads=torch.get_num_threads(),log=[],source_hashes=source_hashes())
    def record(check,**values):
        row=dict(check=check,**values); payload['log'].append(row)
        print(json.dumps(row),flush=True); dump(out/'smoke.json',payload)
    try:
        assert torch.get_num_threads()==1
        raw,digest=h33.load_checked(data); payload['sha256_e10m']=digest
        timings={a:[] for a in ARMS}
        for seed in (0,1):
            # Compare mixed logits and probabilities, in addition to byte losses.
            predictions=[]
            for arm in ('plain_1','zgate_4'):
                tokens=torch.tensor(np.array(raw[:CHUNK],dtype=np.int64))
                outputs=[]; features=None
                for m in build(arm,seed):
                    u=m.inputs(tokens,0,CHUNK,features)
                    z,features,_,_=m.score(u,tokens,0,m.initial())
                    outputs.append(z.detach())
                mixed=(torch.stack(outputs,1)*initial_gate(arm)[0,:,None]).sum(1)
                predictions.append(mixed)
            assert torch.equal(*predictions) and torch.equal(*[p.softmax(-1) for p in predictions])
            _,single=run('plain_1',seed,raw[:CHUNK],keep=True)
            _,zero=run('zgate_4',seed,raw[:CHUNK],keep=True)
            assert np.array_equal(single,zero)
            record('first_chunk_prediction',seed=seed,n_bytes=CHUNK,max_logits_abs=0.,max_probability_abs=0.,max_bits_abs=0.)
            del outputs, features, m, z, u, _, mixed, predictions
            for arm in ('plain_4','plain_1'):
                ref,rb=prior.run(arm,seed,raw[:8192],keep=True)
                ours,ob=run(arm,seed,raw[:8192],keep=True)
                diff=float(np.abs(rb-ob).max())
                record('h34b_equivalence',arm=arm,seed=seed,n_bytes=8192,max_bits_abs=diff,limit=0,
                       reference_bits=ref['bits_per_byte'],bits_per_byte=ours['bits_per_byte'])
                assert np.array_equal(rb,ob)
                timings[arm].append(ours['wall_seconds']/8192)
        for seed in (0,1):
            _,short=run('zgate_4',seed,raw[:1024],keep=True)
            _,long=run('zgate_4',seed,raw[:2048],keep=True)
            changed=raw[:2048].copy(); changed[1024:]=255-changed[1024:]
            _,altered=run('zgate_4',seed,changed,keep=True)
            diff=float(np.abs(short-long[:1024]).max())
            future=float(np.abs(short-altered[:1024]).max())
            record('prequential_prefix',arm='zgate_4',seed=seed,prefix_bytes=1024,extended_bytes=2048,
                   max_bits_abs=diff,changed_suffix_max_bits_abs=future)
            assert diff==future==0
        for arm in ARMS:
            r,b=run(arm,0,raw[:8192],keep=True)
            finite=bool(np.isfinite(b).all() and all(np.isfinite(s['mean_weights']).all() for s in r['mixer_slices']))
            record('finite',arm=arm,n_bytes=len(b),all_finite=finite,bits_per_byte=r['bits_per_byte'],wall_seconds=r['wall_seconds'])
            assert finite
            timings[arm].append(r['wall_seconds']/len(b))
        estimates={a:max(ts)*N_DATA for a,ts in timings.items()}
        record('wall_estimates',estimated_kernel_seconds=estimates,limit=5100,n_bytes=N_DATA,
               method='8192-byte one-thread rates; slowest measured rate per arm; linear extrapolation; no assumed Kaggle speedup')
        assert max(estimates.values())<=5100
        payload['ok']=True; dump(out/'smoke.json',payload)
    except BaseException as exc:
        payload['failure']=repr(exc); dump(out/'smoke.json',payload); raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true'); p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,choices=(0,1),default=0)
    p.add_argument('--data',default='data/enwik8/enwik8'); p.add_argument('--out',default='results/h34f')
    args=p.parse_args(); h33.configure(1)
    if args.smoke: return smoke(args.out,args.data)
    if args.arm is None: p.error('--arm is required')
    sm=json.loads((Path(args.out)/'smoke.json').read_text())
    assert sm['ok'] and sm['source_hashes']==source_hashes() and sm['n_bytes']==N_DATA
    raw,digest=h33.load_checked(args.data)
    assert digest==sm['sha256_e10m']
    r=run(args.arm,args.seed,raw[:N_DATA])
    if args.arm in H34B_REFERENCE:
        assert all(abs(r[k]-v)<=1e-6 for k,v in H34B_REFERENCE[args.arm][args.seed].items())
    r.update(sha256_e10m=digest,kernel_id=f'ezzzzzx/h34f-{args.arm.replace("_","-")}-s{args.seed}')
    dump(Path(args.out)/f'{args.arm}-s{args.seed}.json',r)

if __name__=='__main__': main()
