"""H34a3: partial-sum local losses; principle 2, improve the compressor.

All layer mechanics and normalization import H34a2 unchanged. Layer k learns
from CE(sum(z[:k+1]), y), with detached lower features and no cross-layer
gradients. Predictions use the full sum. Optimizer/chunk timing matches H34a2.
"""
import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import h34a2_logit_sum as prior
base = prior.base
CHUNK, A, LOG2 = prior.CHUNK, prior.A, prior.LOG2
dump, h33 = prior.dump, prior.h33
ARMS = ('casc_2', 'casc_4')


def build(arm, seed, zero=True):
    if arm not in (*ARMS, 'casc_1'):
        raise ValueError(arm)
    return prior.build(arm.replace('casc_', 'sum_'), seed, zero)


def partial_sums(stacked):
    parts = [stacked[:, 0]]
    for k in range(1, stacked.shape[1]):
        parts.append(parts[-1] + stacked[:, k])
    return torch.stack(parts, 1)


def cascade_errors(stacked, target):
    partial = partial_sums(stacked.detach())
    errors = partial.softmax(-1)
    errors[torch.arange(len(target)), :, target] -= 1
    return errors.unbind(1)


def compute_account(arm):
    c = prior.compute_account(arm.replace('casc_', 'sum_'))
    # Additional local softmaxes cost 3A each. The error helper also
    # recomputes the partial additions, beyond those used for scoring.
    extra = (int(arm.split('_')[1])-1)*4*A
    c['cascade_extra_macs_per_byte'] = extra
    c['macs_per_byte'] += extra
    return c


def run(arm, seed, raw, keep=False, slice_size=50_000):
    t0 = time.perf_counter()
    tokens = torch.tensor(np.array(raw,dtype=np.int64))
    layers = build(arm,seed)
    states = [m.initial() for m in layers]
    opts = [torch.optim.Adam(m.parameters(),lr=1e-3) for m in layers]
    bits = np.empty(len(raw),np.float64)
    own = np.empty((len(raw),len(layers)),np.float64)
    nslices = math.ceil(len(raw)/slice_size)
    logsum = np.zeros((nslices,len(layers)))
    logmax = np.zeros_like(logsum)
    for start in range(0,len(raw),CHUNK):
        end = min(start+CHUNK,len(raw))
        target = tokens[start:end]
        outputs, feats, caches = [], [], []
        for k,m in enumerate(layers):
            u = m.inputs(tokens,start,end,feats[-1] if k else None)
            z,f,states[k],cache = m.score(u,tokens,start,states[k])
            outputs.append(z); feats.append(f); caches.append(cache)
        stacked = torch.stack([z.detach() for z in outputs],1)
        combined = partial_sums(stacked)[:, -1]
        losses = F.cross_entropy(combined,target,reduction='none')
        bits[start:end] = losses.detach().double().numpy()/LOG2
        for k,z in enumerate(outputs):
            own[start:end,k] = F.cross_entropy(z.detach(),target,reduction='none').double().numpy()/LOG2
            for idx in range(start//slice_size,(end-1)//slice_size+1):
                a,b = max(start,idx*slice_size)-start,min(end,(idx+1)*slice_size)-start
                absolute = z[a:b].detach().abs()
                logsum[idx,k] += float(absolute.double().sum())
                logmax[idx,k] = max(logmax[idx,k],float(absolute.max()))
        errors = cascade_errors(stacked, target)
        for k,m in enumerate(layers):
            m.learn(outputs[k],errors[k]/(end-start),caches[k])
        for m,opt in zip(layers,opts):
            nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt.step()
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
        diagnostics.append(row)
    result = dict(arm=arm,seed=seed,n_bytes=len(raw),bits_per_byte=float(bits.mean()),slices=slices,
                  tail_bits_per_byte=float(bits[50_000:300_000].mean()) if len(raw)>50_000 else None,
                  layers=diagnostics,wall_seconds=time.perf_counter()-t0)
    return (result,bits) if keep else result


def smoke(out, data):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    payload = dict(ok=False, n_bytes=300_000, threads=1, log=[])
    def record(check, **values):
        row = dict(check=check, **values)
        payload['log'].append(row)
        print(__import__('json').dumps(row), flush=True)
        dump(out/'smoke.json', payload)
    try:
        raw, digest = h33.load_checked(data)
        payload['sha256_e10m'] = digest
        _, one = run('casc_1', 0, raw[:8192], keep=True)
        _, old = prior.run('sum_1', 0, raw[:8192], keep=True)
        gap = float(np.abs(one-old).max())
        record('casc1_equals_sum1', n_bytes=8192, max_bits_abs=gap, limit=1e-9)
        assert gap <= 1e-9, 'casc_1 equivalence'
        tokens = torch.tensor(raw[:512].astype(np.int64))
        layers = build('casc_4', 0, zero=False)
        outputs, f = [], None
        for m in layers:
            z, f, _, _ = m.score(m.inputs(tokens,0,512,f), tokens,0,m.initial())
            outputs.append(z)
        errors = cascade_errors(torch.stack(outputs,1), tokens)
        diffs = []
        for k in range(4):
            grad, = torch.autograd.grad(F.cross_entropy(sum(outputs[:k+1]),tokens,reduction='sum'),
                                       outputs[k], retain_graph=True)
            diffs.append(float((grad-errors[k]).abs().max()))
        record('partial_sum_gradients', n_bytes=512, max_abs_per_layer=diffs, limit=1e-6)
        assert max(diffs) <= 1e-6, 'cascade gradients'
        tok = tokens[:128]
        low = prior.Layer(zero=False)
        z, f, _, _ = low.score(low.inputs(tok,0,128),tok,0,low.initial())
        init_diffs = []
        for _ in range(3):
            upper = prior.Layer(prev=320)
            added, f, _, _ = upper.score(upper.inputs(tok,0,128,f),tok,0,upper.initial())
            init_diffs.append(float((z.softmax(-1)-(z+added).softmax(-1)).detach().abs().max()))
            assert added.count_nonzero() == 0
            z = z+added
        record('identity_at_init', max_probability_abs_per_layer=init_diffs, limit=1e-9)
        assert max(init_diffs) <= 1e-9, 'identity'
        _, short = run('casc_4',0,raw[:1024],keep=True)
        _, long = run('casc_4',0,raw[:2048],keep=True)
        gap = float(np.abs(short-long[:1024]).max())
        record('prequential', max_bits_abs=gap)
        assert gap == 0, 'prefix'
        r = run('casc_4',0,raw[:32768],slice_size=2048)
        record('early_stability', slices=r['slices'], layers=r['layers'], limit_first_slice=9)
        assert r['slices'][0]['bits_per_byte'] <= 9, 'early stability'
        estimate = r['wall_seconds']*300_000/32768
        estimates = {a:estimate*compute_account(a)['macs_per_byte']/compute_account('casc_4')['macs_per_byte'] for a in ARMS}
        record('timing', arm='casc_4', n_bytes=32768, wall_seconds=r['wall_seconds'],
               estimated_kernel_seconds=estimates, limit=5100,
               method='Measured casc_4 rate; casc_2 MAC-scaled, no assumed Kaggle speedup')
        assert max(estimates.values()) <= 5100, '85 minute timing'
        payload['ok'] = True
        dump(out/'smoke.json',payload)
    except Exception as exc:
        payload['failure'] = repr(exc)
        dump(out/'smoke.json',payload)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--arm',choices=(*ARMS,'casc_1'))
    p.add_argument('--seed',type=int,choices=(0,1),default=0)
    p.add_argument('--data',default='data/enwik8/enwik8')
    p.add_argument('--out',default='results/h34a3')
    args = p.parse_args(); h33.configure(1)
    if args.smoke:
        return smoke(args.out,args.data)
    if args.arm is None:
        p.error('--arm is required')
    raw,digest = h33.load_checked(args.data)
    r = run(args.arm,args.seed,raw[:300_000])
    r.update(sha256_e10m=digest,kernel_id=f'ezzzzzx/h34a3-{args.arm.replace("_","-")}-s{args.seed}')
    dump(Path(args.out)/'results.json',r)

if __name__ == '__main__':
    main()
