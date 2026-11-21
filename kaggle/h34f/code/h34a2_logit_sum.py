"""H34a2: detached local banks with stateless inputs and fixed logit sums."""
import argparse
import json
import math
import time
from pathlib import Path
import h34a_local_depth as base
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHUNK, EMB, RANK, A, LOG2 = base.CHUNK, base.EMB, base.RANK, base.A, base.LOG2
ARMS = ('plain_1','plain_2','plain_4','sum_1','sum_2','sum_4',
        'sumown_2','sumown_4','sum_4_noskip','wide_1')
dump = base.dump
h33 = base.h33


def normalize(f):
    f = f.detach()
    return f / (f.square().mean(-1, keepdim=True).sqrt() + 1e-6)


class Layer(base.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.rms2

    def inputs(self, tokens, start, end, lower=None):
        parts = []
        if self.has_emb:
            e = self.embedding(tokens[max(0,start-1):end-1]).detach()
            if start == 0:
                e = torch.cat((e.new_zeros(1, EMB), e))
            parts.append(e)
        if lower is not None:
            parts.append(normalize(lower))
        return torch.cat(parts, -1)


def build(arm, seed, zero=True):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(seed)
    n = 512 if arm == 'wide_1' else 128
    depth = 1 if arm == 'wide_1' else int(arm.split('_')[1])
    return [Layer(n, 0 if k == 0 else 2*n+RANK, arm != 'sum_4_noskip', zero)
            for k in range(depth)]


def compute_account(arm):
    # Reuse H34a's explicit real-MAC accounting; replace only changed terms.
    old = 'wide_1' if arm == 'wide_1' else ('left_4_noskip' if arm.endswith('noskip') else 'plain_'+arm.split('_')[1])
    c = base.compute_account(old)
    for row, m in zip(c['layers'], build(arm,0)):
        row['terms']['rms'] = 0
        row['terms']['normalize'] = 2*m.prev+2 if m.prev else 0  # square/divide per feature, mean scale and epsilon
        row['macs_per_byte'] = sum(row['terms'].values())
    d = len(c['layers'])
    plain = arm.startswith('plain')
    combine = c['mixer_macs_per_byte'] if plain else (d-1)*A+3*A
    c['parameters'] -= 0 if plain else d
    c['mixer_macs_per_byte'] = combine if plain else 0
    c['combiner_macs_per_byte'] = combine
    c['macs_per_byte'] = sum(r['macs_per_byte'] for r in c['layers'])+combine
    return c


def run(arm, seed, raw, keep=False, slice_size=50_000):
    t0 = time.perf_counter()
    tokens = torch.tensor(np.array(raw,dtype=np.int64))
    layers = build(arm,seed)
    states = [m.initial() for m in layers]
    opts = [torch.optim.Adam(m.parameters(),lr=1e-3) for m in layers]
    plain = arm.startswith('plain')
    mixer = base.h32b.GeoMixer(arm,1,len(layers),torch.ones(1,len(layers))) if plain else None
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
        err = combined.detach().softmax(-1)
        err[torch.arange(end-start),target] -= 1
        for k,m in enumerate(layers):
            g = err.clone() if arm.startswith('sum_') or arm == 'wide_1' else outputs[k].detach().softmax(-1)
            if not (arm.startswith('sum_') or arm == 'wide_1'):
                g[torch.arange(end-start),target] -= 1
            m.learn(outputs[k],g/(end-start),caches[k])
        if plain:
            mixer.opt.zero_grad(set_to_none=True)
            losses.mean().backward()
        for m,opt in zip(layers,opts):
            nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step()
        if plain:
            mixer.opt.step()
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
        diagnostics.append(row)
    result = dict(arm=arm,seed=seed,n_bytes=len(raw),bits_per_byte=float(bits.mean()),slices=slices,
                  tail_bits_per_byte=float(bits[50_000:300_000].mean()) if len(raw)>50_000 else None,
                  layers=diagnostics,wall_seconds=time.perf_counter()-t0)
    return (result,bits) if keep else result



def estimate_walls(payload):
    timings = {r['arm']:r['estimated_kernel_seconds'] for r in payload['log'] if r['check']=='timing'}
    costs = {a:compute_account(a)['macs_per_byte'] for a in ARMS}
    estimates = {a:timings['sum_4']*costs[a]/costs['sum_4'] for a in ARMS}
    estimates['wide_1'] = timings['wide_1']
    return estimates


def smoke(out,data):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    payload = dict(ok=False,n_bytes=300_000,threads=1,log=[])
    def record(check,**values):
        row = dict(check=check,**values)
        payload['log'].append(row)
        print(json.dumps(row),flush=True)
        dump(out/'smoke.json',payload)
    try:
        raw,digest = h33.load_checked(data)
        payload['sha256_e10m'] = digest
        tokens = torch.tensor(raw[:512].astype(np.int64))
        layers = build('sum_2',0,zero=False)
        outputs=[]; f=None
        for m in layers:
            z,f,_,_ = m.score(m.inputs(tokens,0,512,f),tokens,0,m.initial())
            outputs.append(z)
        total = sum(outputs)
        grads = torch.autograd.grad(F.cross_entropy(total,tokens,reduction='sum'),outputs)
        g = total.detach().softmax(-1)
        g[torch.arange(512),tokens] -= 1
        errors = [float((v-g).abs().max()) for v in grads]
        record('exact_gradient',max_abs_per_layer=errors,limit=1e-6)
        if max(errors)>1e-6: raise RuntimeError('exact gradient failed')
        low = Layer(zero=False); upper = Layer(prev=320)
        tok = tokens[:128]
        lo,f,_,_ = low.score(low.inputs(tok,0,128),tok,0,low.initial())
        up,_,_,_ = upper.score(upper.inputs(tok,0,128,f),tok,0,upper.initial())
        diff = float((lo.softmax(-1)-(lo+up).softmax(-1)).detach().abs().max())
        record('identity',max_probability_abs=diff,max_added_logit_abs=float(up.detach().abs().max()),limit=1e-9)
        if diff>1e-9 or up.count_nonzero(): raise RuntimeError('identity failed')
        for arm in ('sum_4','plain_4'):
            r = run(arm,0,raw[:16384],slice_size=2048)
            record('early_stability',arm=arm,slices=r['slices'],layers=r['layers'],limit_first_slice=9)
            if r['slices'][0]['bits_per_byte']>9: raise RuntimeError('early stability failed')
        _,short = run('sum_4',0,raw[:1024],keep=True)
        _,long = run('sum_4',0,raw[:2048],keep=True)
        diff = float(np.abs(short-long[:1024]).max())
        record('prequential',max_bits_abs=diff)
        if diff!=0: raise RuntimeError('prequential failed')
        # Same detached lower features, exact trace vs full within-layer BPTT.
        layers = build('sum_2',0,zero=False)
        low,m = layers
        _,f,_,_ = low.score(low.inputs(tokens,0,512),tokens,0,low.initial())
        z,_,_,cache = m.score(m.inputs(tokens,0,512,f),tokens,0,m.initial())
        g = z.detach().softmax(-1); g[torch.arange(512),tokens]-=1
        m.learn(z,g,cache)
        trace = {name:p.grad.clone() for name,p in m.named_parameters()}
        m.zero_grad(set_to_none=True)
        e = m.embedding(tokens[:-1]); e = torch.cat((e.new_zeros(1,EMB),e))
        u = torch.cat((e,normalize(f)),-1)
        inj = torch.complex(u@m.B_re.T,u@m.B_im.T)
        h = m.initial()[0]; hs=[]; lam=m.lam()
        for row in inj:
            h=lam*h+row; hs.append(h)
        F.cross_entropy(m.logits_from(torch.stack(hs)),tokens,reduction='sum').backward()
        rows = [base.h30x.grad_pair(name,trace[name],p.grad) for name,p in m.named_parameters()]
        maximum = max(r['max_rel'] for r in rows if r['name'] in ('nu','theta','B_re','B_im'))
        record('exact_traces',max_relative_error=maximum,gradients=rows,limit=1e-4)
        if maximum>1e-4: raise RuntimeError('trace exactness failed')
        for arm in ARMS:
            r=run(arm,0,raw[:8192])
            record('finite',arm=arm,bits_per_byte=r['bits_per_byte'],wall_seconds=r['wall_seconds'])
        for arm in ('sum_4','wide_1'):
            r=run(arm,0,raw[:32768])
            estimate=r['wall_seconds']*300_000/32768
            record('timing',arm=arm,wall_seconds=r['wall_seconds'],estimated_kernel_seconds=estimate,limit=5100)
            if estimate>5100: raise RuntimeError('85 minute timing failed')
        estimates=estimate_walls(payload)
        record('estimated_walls',estimated_kernel_seconds=estimates,limit=5100,
               method='MAC-scaled from sum_4; wide_1 uses its own measured rate')
        if max(estimates.values())>5100: raise RuntimeError('85 minute all-arm estimate failed')
        payload['ok']=True
        dump(out/'smoke.json',payload)
    except Exception as exc:
        payload['failure']=repr(exc); dump(out/'smoke.json',payload)
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--data',default='data/enwik8/enwik8')
    p.add_argument('--out',default='results/h34a2')
    args=p.parse_args(); h33.configure(1)
    if args.smoke: return smoke(args.out,args.data)
    if args.arm is None: p.error('--arm is required')
    raw,digest=h33.load_checked(args.data)
    r=run(args.arm,args.seed,raw[:300_000])
    r.update(sha256_e10m=digest,kernel_id=f'ezzzzzx/h34a2-{args.arm.replace("_","-")}-s{args.seed}')
    dump(Path(args.out)/'results.json',r)

if __name__=='__main__': main()
