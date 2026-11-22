"""H34g: improve the living compressor with growing depth and dense local inputs.

Retains H34f own-loss trace learning and chunk update order. No global gradients.
Checkpoint boundaries buffer an unfinished chunk instead of changing its update.
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
import h34f_zero_gate as prior
import h34b_plain_depth as depth_prior

CHUNK, EMB, RANK, A, LOG2 = prior.CHUNK, prior.EMB, prior.RANK, prior.A, prior.LOG2
N_DATA, TAIL_START = prior.N_DATA, prior.TAIL_START
Layer, dump, h33 = prior.Layer, prior.dump, prior.h33
ARMS = ('zgate_2','zgate_8','dense_6','zgate_6')
H34F_REFERENCE = {
    0: dict(tail_bits_per_byte=3.078244907,bits_per_byte=3.374814145,first_40kb_bits_per_byte=4.412806476),
    1: dict(tail_bits_per_byte=3.077944571,bits_per_byte=3.386931698,first_40kb_bits_per_byte=4.468386643),
}


def build(arm,seed):
    if arm not in ARMS: raise ValueError(arm)
    torch.manual_seed(seed)
    return [Layer(128,0 if k==0 else 256+RANK,True,True) for k in range(int(arm.split('_')[1]))]


def initial_gate(arm):
    if arm not in ARMS: raise ValueError(arm)
    w=torch.zeros(1,int(arm.split('_')[1])); w[0,0]=1
    return w


def layer_input(arm,m,tokens,start,end,feats):
    if not feats or not arm.startswith('dense'):
        return m.inputs(tokens,start,end,feats[-1] if feats else None)
    # Exactly H34b's per-step normalization, then average; never renormalize the mean.
    normalized=[depth_prior.prior.normalize(f) for f in feats]
    lower=normalized[0] if len(normalized)==1 else torch.stack(normalized).mean(0)
    return torch.cat((m.inputs(tokens,start,end),lower),-1)


def run(arm, seed, raw, keep=False, slice_size=20_000, *, checkpoint=None, resume=None, stop_at=None, capture=None):
    t0 = time.perf_counter()
    tokens = torch.tensor(np.array(raw,dtype=np.int64))
    layers = build(arm,seed)
    states = [m.initial() for m in layers]
    opts = [torch.optim.Adam(m.parameters(),lr=1e-3) for m in layers]
    plain = True
    frozen = arm.endswith('fixmix')
    mixer = depth_prior.prior.base.h32b.GeoMixer(arm,1,len(layers),torch.full((1,len(layers)),1/len(layers)) if frozen else initial_gate(arm))
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
    offset, previous_wall = 0, 0.
    if resume is not None:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        assert saved['source_hashes'] == source_hashes()
        assert (saved['arm'],saved['seed'],saved['n_bytes'],saved['slice_size']) == (arm,seed,len(raw),slice_size)
        assert saved['raw_sha256'] == hashlib.sha256(np.asarray(raw,dtype=np.uint8).tobytes()).hexdigest()
        for m,sd,opt,od in zip(layers,saved['layers'],opts,saved['opts']):
            m.load_state_dict(sd); opt.load_state_dict(od)
        states = saved['states']
        with torch.no_grad(): mixer.W.copy_(saved['mixer'])
        mixer.opt.load_state_dict(saved['mixer_opt'])
        offset, previous_wall = saved['offset'], saved['wall_seconds']
        bits, own = saved['bits'], saved['own']
        logsum, logmax, gate_sums = saved['logsum'], saved['logmax'], saved['gate_sums']
        weight_sum, weight_n = saved['weight_sum'], saved['weight_n']
        weight_tail, weight_tail_n = saved['weight_tail'], saved['weight_tail_n']
        torch.set_rng_state(saved['rng'])
    limit = len(raw) if stop_at is None else stop_at
    assert offset <= limit <= len(raw)
    # Never introduce a partial Adam update at an arbitrary kernel boundary.
    committed_limit = limit if limit == len(raw) else limit // CHUNK * CHUNK
    for start in range(offset,committed_limit,CHUNK):
        end = min(start+CHUNK,len(raw))
        target = tokens[start:end]
        outputs, feats, caches = [], [], []
        for k,m in enumerate(layers):
            u = layer_input(arm,m,tokens,start,end,feats)
            z,f,states[k],cache = m.score(u,tokens,start,states[k])
            outputs.append(z); feats.append(f); caches.append(cache)
            if capture is not None and k == 1:
                capture.append((u.detach().clone(),z.detach().clone(),f.detach().clone()))
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
    segment_wall = time.perf_counter()-t0
    total_wall = previous_wall + segment_wall
    if checkpoint is not None:
        path = Path(checkpoint); path.parent.mkdir(parents=True,exist_ok=True)
        saved = dict(source_hashes=source_hashes(),arm=arm,seed=seed,n_bytes=len(raw),slice_size=slice_size,
            raw_sha256=hashlib.sha256(np.asarray(raw,dtype=np.uint8).tobytes()).hexdigest(),
            offset=committed_limit,bytes_seen=limit,pending_bytes=np.asarray(raw[committed_limit:limit]).copy(),
            layers=[m.state_dict() for m in layers],opts=[o.state_dict() for o in opts],states=states,
            mixer=mixer.W.detach(),mixer_opt=mixer.opt.state_dict(),rng=torch.get_rng_state(),
            bits=bits,own=own,logsum=logsum,logmax=logmax,gate_sums=gate_sums,
            weight_sum=weight_sum,weight_n=weight_n,weight_tail=weight_tail,weight_tail_n=weight_tail_n,
            wall_seconds=total_wall)
        temp = path.with_suffix(path.suffix+'.tmp'); torch.save(saved,temp); temp.replace(path)
    if limit < len(raw):
        assert checkpoint is not None
        result=dict(arm=arm,seed=seed,n_bytes=len(raw),bytes_seen=limit,offset=committed_limit,
                    pending_bytes=limit-committed_limit,wall_seconds=total_wall,segment_wall_seconds=segment_wall,
                    checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest())
        return (result,bits[:committed_limit].copy()) if keep else result
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
                  layers=diagnostics,max_frozen_weight_change=max_frozen_weight_change,wall_seconds=total_wall,segment_wall_seconds=segment_wall)
    return (result,bits) if keep else result


def source_hashes():
    return {'h34g_depth.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**prior.source_hashes()}


def check_reference(r):
    if r['arm']=='zgate_6' and r['n_bytes']==N_DATA:
        assert all(abs(r[k]-v)<=1e-6 for k,v in H34F_REFERENCE[r['seed']].items()), 'H34f mismatch'


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
            ref,rb=prior.run('zgate_6',seed,raw[:8192],keep=True)
            ours,ob=run('zgate_6',seed,raw[:8192],keep=True)
            assert np.array_equal(rb,ob)
            record('h34f_equivalence',seed=seed,n_bytes=8192,max_bits_abs=float(np.abs(rb-ob).max()),
                   reference_bits=ref['bits_per_byte'],bits_per_byte=ours['bits_per_byte'])
            timings['zgate_6'].append(ours['wall_seconds']/8192)
            captures=[]
            for arm in ('dense_6','zgate_6'):
                cap=[]; run(arm,seed,raw[:2048],capture=cap); captures.append(cap)
            assert all(torch.equal(a,b) for ca,cb in zip(*captures) for a,b in zip(ca,cb))
            record('layer_2_equivalence',seed=seed,n_bytes=2048,max_input_abs=0.,max_logits_abs=0.,max_features_abs=0.)
            del captures,cap
            for arm in ('dense_6','zgate_8'):
                _,short=run(arm,seed,raw[:1024],keep=True)
                _,long=run(arm,seed,raw[:2048],keep=True)
                changed=raw[:2048].copy(); changed[1024:]=255-changed[1024:]
                _,altered=run(arm,seed,changed,keep=True)
                assert np.array_equal(short,long[:1024]) and np.array_equal(short,altered[:1024])
                record('prequential_prefix',arm=arm,seed=seed,n_bytes=1024,max_bits_abs=0.,changed_suffix_max_bits_abs=0.)
        for arm in ARMS:
            r,b=run(arm,0,raw[:8192],keep=True)
            assert np.isfinite(b).all() and all(np.isfinite(s['mean_weights']).all() for s in r['mixer_slices'])
            record('finite',arm=arm,n_bytes=8192,all_finite=True,bits_per_byte=r['bits_per_byte'],wall_seconds=r['wall_seconds'])
            timings[arm].append(r['wall_seconds']/8192)
        # Exact persisted/resumed comparison, including parameters, traces, Adam and diagnostics.
        def equal(a,b):
            if torch.is_tensor(a): return torch.equal(a,b)
            if isinstance(a,np.ndarray): return np.array_equal(a,b)
            if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        for seed in (0,1):
            whole=out/f'whole-s{seed}.pt'; split=out/f'split-s{seed}.pt'
            r,b=run('zgate_8',seed,raw[:8192],keep=True,checkpoint=whole)
            for boundary in (4096,4112):
                part,_=run('zgate_8',seed,raw[:8192],keep=True,checkpoint=split,stop_at=boundary)
                rr,bb=run('zgate_8',seed,raw[:8192],keep=True,resume=split,checkpoint=split)
                assert np.array_equal(b,bb)
                assert equal({k:v for k,v in r.items() if 'wall' not in k},{k:v for k,v in rr.items() if 'wall' not in k})
                a=torch.load(whole,weights_only=False); c=torch.load(split,weights_only=False)
                assert equal({k:v for k,v in a.items() if k!='wall_seconds'},{k:v for k,v in c.items() if k!='wall_seconds'})
                record('checkpoint_equivalence',arm='zgate_8',seed=seed,n_bytes=8192,split_at=boundary,
                       committed_bytes=part['offset'],pending_bytes=part['pending_bytes'],max_bits_abs=0.,state_exact=True)
            whole.unlink(); split.unlink()
        estimates={a:max(ts)*N_DATA for a,ts in timings.items()}
        payload['split_zgate_8']=estimates['zgate_8']>5100
        segment_estimates={a:seconds/(2 if a=='zgate_8' and payload['split_zgate_8'] else 1) for a,seconds in estimates.items()}
        record('wall_estimates',estimated_kernel_seconds=estimates,estimated_segment_seconds=segment_estimates,
               split_zgate_8=payload['split_zgate_8'],limit=5100,n_bytes=N_DATA)
        assert max(segment_estimates.values())<=5100
        payload['ok']=True; dump(out/'smoke.json',payload)
    except BaseException as exc:
        payload['failure']=repr(exc); dump(out/'smoke.json',payload); raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true'); p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,choices=(0,1),default=0)
    p.add_argument('--data',default='data/enwik8/enwik8'); p.add_argument('--out',default='results/h34g')
    p.add_argument('--checkpoint'); p.add_argument('--resume'); p.add_argument('--stop-at',type=int)
    args=p.parse_args(); h33.configure(1)
    if args.smoke: return smoke(args.out,args.data)
    if args.arm is None: p.error('--arm is required')
    sm=json.loads((Path(args.out)/'smoke.json').read_text())
    assert sm['ok'] and sm['source_hashes']==source_hashes() and sm['n_bytes']==N_DATA
    raw,digest=h33.load_checked(args.data); assert digest==sm['sha256_e10m']
    r=run(args.arm,args.seed,raw[:N_DATA],checkpoint=args.checkpoint,resume=args.resume,stop_at=args.stop_at)
    if args.stop_at is None: check_reference(r)
    r.update(sha256_e10m=digest,kernel_id=f'ezzzzzx/h34g-{args.arm.replace("_","-")}-s{args.seed}')
    dump(Path(args.out)/f'{args.arm}-s{args.seed}.json',r)

if __name__=='__main__': main()
