"""H34d: improve the living compressor with own-loss plus direct feedback alignment.

No backward pass between layers. k is ONE-BASED in the prescribed feedback
seed (7000 + 100*seed + k). Norms are Frobenius over a chunk's time/state
entries, using mean-loss gradients. Zero own norm implies alpha=0 (including
zero-readout initialization). Readouts always receive only their own loss.
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
h33, dump = prior.h33, prior.dump
ARMS = ('plain_4', 'dfa_4', 'dfa_only_4')


def feedback(seed, k, n=128, dtype=torch.float32):
    # Independent numpy RNG: never changes the torch initialization stream.
    return torch.tensor(np.random.default_rng(7_000+100*seed+k).normal(
        0, 1/math.sqrt(256), (2*n,256)), dtype=dtype)


class Layer(prior.Layer):
    def learn(self, logits, g, cache, global_error=None, matrix=None,
              only=False, force_alpha=None):
        hr, hi, el, eb, ez = cache
        self.zero_grad(set_to_none=True)
        logits.backward(g)
        with torch.no_grad():
            own = torch.cat((hr.grad, hi.grad), -1)
            own_norm = float(own.norm())
            alpha, raw_norm, dfa_norm = 0., 0., 0.
            state_grad = own
            if global_error is not None:
                dfa = global_error @ matrix.T
                raw_norm = float(dfa.norm())
                alpha = own_norm/raw_norm if raw_norm else 0.
                if force_alpha is not None:
                    alpha = force_alpha
                dfa = dfa*alpha
                dfa_norm = float(dfa.norm())
                state_grad = dfa if only else own+dfa
            c = torch.complex(state_grad[:, :self.n], state_grad[:, self.n:]).conj()
            lam = self.lam()
            self.nu.grad = (c*(el*(-self.nu.exp()*lam))).real.sum(0)
            self.theta.grad = (c*(el*(1j*lam))).real.sum(0)
            self.B_re.grad = (c[:, :, None]*eb).real.sum(0)
            self.B_im.grad = (c[:, :, None]*(1j*eb)).real.sum(0)
            if self.has_emb:
                b = torch.complex(self.B_re[:, :EMB], self.B_im[:, :EMB])
                self.embedding.weight.grad = torch.einsum('tia,ie->ae', c[:, :, None]*ez, b).real
        for p in self.parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError('nonfinite/missing layer gradient')
        return dict(own_norm=own_norm, raw_dfa_norm=raw_norm, alpha=alpha,
                    dfa_norm=dfa_norm, only=only)


def build(arm, seed, zero=True):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(seed)
    return [Layer(128, 0 if k==0 else 256+RANK, True, zero) for k in range(4)]


def run(arm, seed, raw, keep=False, slice_size=25_000, force_alpha=None):
    t0 = time.perf_counter()
    tokens = torch.tensor(np.array(raw,dtype=np.int64))
    layers = build(arm,seed)
    matrices = [feedback(seed,k+1) for k in range(4)]
    norm_log = []
    states = [m.initial() for m in layers]
    opts = [torch.optim.Adam(m.parameters(),lr=1e-3) for m in layers]
    plain = True
    frozen = arm.endswith('fixmix')
    mixer = prior.prior.base.h32b.GeoMixer(arm,1,len(layers),torch.full((1,len(layers)),1/len(layers)) if frozen else torch.ones(1,len(layers)))
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
            row = m.learn(outputs[k],g/(end-start),caches[k],
                          err if arm != 'plain_4' else None,
                          matrices[k], only=(arm=='dfa_only_4' and k>=1),
                          force_alpha=force_alpha)
            if arm != 'plain_4':
                norm_log.append(dict(start=start,end=end,layer=k+1,**row))
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
    result['dfa_norms'] = norm_log
    result['feedback_indexing'] = 'one-based k=1..4; seed=7000+100*seed+k'
    return (result,bits) if keep else result



def source_hashes():
    root=Path(__file__).resolve().parent
    return dict(prior.source_hashes(), **{'h34d_dfa.py':hashlib.sha256((root/'h34d_dfa.py').read_bytes()).hexdigest()})


def trace_check(raw):
    """512 steps, nonzero readouts, detached lower features; DFA surrogate only.

    The autograd reference unrolls the upper layer recurrence, including its
    embedding, with fixed parameters. It does not differentiate across layers.
    Double precision avoids cancellation obscuring trace-rule correctness.
    """
    tokens = torch.tensor(raw[:512].astype(np.int64))
    low,m = [x.double() for x in build('dfa_4',0,zero=False)[:2]]
    lower_z,f,_,_ = low.score(low.inputs(tokens,0,512),tokens,0,low.initial())
    z,_,_,cache = m.score(m.inputs(tokens,0,512,f),tokens,0,m.initial())
    err = (lower_z.detach()+z.detach()).softmax(-1)
    err[torch.arange(512),tokens] -= 1
    matrix = feedback(0,2,dtype=torch.float64)
    # Compute own-state gradient to reproduce the production norm rule.
    g = z.detach().softmax(-1)
    g[torch.arange(512),tokens] -= 1
    g = g/512
    own = torch.cat(torch.autograd.grad(z,cache[:2],g,retain_graph=True),-1)
    direction = err @ matrix.T
    alpha = float(own.norm()/direction.norm())
    direction *= alpha
    row = m.learn(z,g,cache,err,matrix,only=True)
    recurrent = ('nu','theta','B_re','B_im','embedding.weight')
    trace = {name:p.grad.clone() for name,p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    e = m.embedding(tokens[:-1]); e = torch.cat((e.new_zeros(1,EMB),e))
    u = torch.cat((e,prior.prior.normalize(f)),-1)
    inj = torch.complex(u @ m.B_re.T,u @ m.B_im.T)
    h = m.initial()[0]; hs=[]; lam=m.lam()
    for t in range(512):
        h=lam*h+inj[t]; hs.append(h)
    hb = torch.stack(hs)
    surrogate = (torch.cat((hb.real,hb.imag),-1)*direction.detach()).sum()
    surrogate.backward()
    rows = [prior.prior.base.h30x.grad_pair(name,trace[name],p.grad)
            for name,p in m.named_parameters() if name in recurrent]
    # Readouts must have exactly the own-loss gradient, even in DFA-only mode.
    m.zero_grad(set_to_none=True)
    F.cross_entropy(m.logits_from(hb.detach()),tokens).backward()
    readout_diff=max(float((trace[name]-p.grad).abs().max()) for name,p in m.named_parameters()
                     if name not in recurrent)
    return dict(max_relative_error=max(r['max_rel'] for r in rows),gradients=rows,
                readout_own_max_abs=readout_diff,norms=row,n_bytes=512,limit=1e-4)


def smoke(out,data):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    payload=dict(ok=False,n_bytes=N_DATA,tail_start=TAIL_START,threads=1,
                 source_hashes=source_hashes(),log=[])
    def record(check,**values):
        row=dict(check=check,**values); payload['log'].append(row)
        print(json.dumps(row),flush=True); dump(out/'smoke.json',payload)
    try:
        raw,digest=h33.load_checked(data); payload['sha256_e10m']=digest
        check=trace_check(raw); record('dfa_trace_autograd',**check)
        assert check['max_relative_error']<=1e-4
        assert check['readout_own_max_abs']<=1e-10
        estimates={}
        for seed in (0,1):
            ref,rb=prior.run('plain_4',seed,raw[:8192],keep=True)
            plain,pb=run('plain_4',seed,raw[:8192],keep=True)
            dfa,db=run('dfa_4',seed,raw[:8192],keep=True,force_alpha=0.)
            ref_diff=float(np.abs(rb-pb).max()); dfa_diff=float(np.abs(pb-db).max())
            record('alpha_zero_equivalence',seed=seed,n_bytes=8192,
                   h34b_max_bits_abs=ref_diff,dfa_max_bits_abs=dfa_diff,limit=1e-9)
            assert max(ref_diff,dfa_diff)<=1e-9
            estimates['plain_4']=max(estimates.get('plain_4',0.),plain['wall_seconds']*N_DATA/8192)
        for arm in ('dfa_4','dfa_only_4'):
            r,b=run(arm,0,raw[:8192],keep=True)
            rows=r['dfa_norms']
            norm_diff=max(abs(x['dfa_norm']-x['own_norm'])/max(x['own_norm'],1e-30) for x in rows)
            zero_count=sum(x['own_norm']==0 for x in rows)
            record('equal_norm_per_chunk',arm=arm,n_bytes=len(b),chunks=len(b)//CHUNK,
                   layer_chunks=len(rows),max_relative_error=norm_diff,zero_own_count=zero_count,
                   all_finite=all(math.isfinite(x['alpha']) for x in rows),limit=1e-6)
            assert norm_diff<=1e-6 and all(math.isfinite(x['alpha']) for x in rows)
            assert all(x['only']==(arm=='dfa_only_4' and x['layer']>=2) for x in rows)
            estimates[arm]=r['wall_seconds']*N_DATA/len(b)
            # Prefix ends on a chunk boundary: the H34b chunk update protocol.
            prefix=1024
            _,short=run(arm,0,raw[:prefix],keep=True)
            changed=raw[:2048].copy(); changed[prefix:]=255-changed[prefix:]
            _,altered=run(arm,0,changed,keep=True)
            long_diff=float(np.abs(short-b[:prefix]).max())
            changed_diff=float(np.abs(short-altered[:prefix]).max())
            record('prequential_prefix',arm=arm,prefix_bytes=prefix,extended_bytes=[2048,8192],
                   max_bits_abs=long_diff,changed_suffix_max_bits_abs=changed_diff)
            assert long_diff==changed_diff==0
        record('wall_estimates',n_bytes=N_DATA,estimated_kernel_seconds=estimates,limit=5100,
               method='Each arm measured on 8192 bytes, one CPU thread; linear extrapolation; no assumed Kaggle speedup')
        assert max(estimates.values())<=5100
        payload['ok']=True; dump(out/'smoke.json',payload)
    except BaseException as exc:
        payload['failure']=repr(exc); dump(out/'smoke.json',payload); raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true'); p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,choices=(0,1),default=0)
    p.add_argument('--data',default='data/enwik8/enwik8'); p.add_argument('--out',default='results/h34d')
    args=p.parse_args(); h33.configure(1)
    if args.smoke: return smoke(args.out,args.data)
    if args.arm is None: p.error('--arm is required')
    sm=json.loads((Path(args.out)/'smoke.json').read_text())
    assert sm['ok'] and sm['source_hashes']==source_hashes()
    raw,digest=h33.load_checked(args.data)
    r=run(args.arm,args.seed,raw[:N_DATA])
    r.update(sha256_e10m=digest,kernel_id=f'ezzzzzx/h34d-{args.arm.replace("_","-")}-s{args.seed}')
    dump(Path(args.out)/f'{args.arm}-s{args.seed}.json',r)


if __name__=='__main__': main()
