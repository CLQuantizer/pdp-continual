"""Supplemental production-dtype check: own + DFA vs within-layer BPTT."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import h34d_dfa as h


def main():
    h.h33.configure(1)
    raw,_=h.h33.load_checked(ROOT/'data/enwik8/enwik8')
    tokens=torch.tensor(raw[:512].astype(np.int64))
    low,m=h.build('dfa_4',1,zero=False)[:2]
    low_z,f,_,_=low.score(low.inputs(tokens,0,512),tokens,0,low.initial())
    z,_,_,cache=m.score(m.inputs(tokens,0,512,f),tokens,0,m.initial())
    g=z.detach().softmax(-1); g[torch.arange(512),tokens]-=1; g/=512
    err=(low_z.detach()+z.detach()).softmax(-1); err[torch.arange(512),tokens]-=1
    own=torch.cat(torch.autograd.grad(z,cache[:2],g,retain_graph=True),-1)
    direction=err @ h.feedback(1,2).T
    direction*=float(own.norm()/direction.norm())
    norms=m.learn(z,g,cache,err,h.feedback(1,2))
    trace={name:p.grad.clone() for name,p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    e=m.embedding(tokens[:-1]); e=torch.cat((e.new_zeros(1,h.EMB),e))
    u=torch.cat((e,h.prior.prior.normalize(f)),-1)
    inj=torch.complex(u @ m.B_re.T,u @ m.B_im.T)
    state=m.initial()[0]; hs=[]; lam=m.lam()
    for row in inj:
        state=lam*state+row; hs.append(state)
    hb=torch.stack(hs)
    surrogate=(torch.cat((hb.real,hb.imag),-1)*direction.detach()).sum()
    (F.cross_entropy(m.logits_from(hb),tokens)+surrogate).backward()
    rows=[h.prior.prior.base.h30x.grad_pair(name,trace[name],p.grad) for name,p in m.named_parameters()]
    maximum=max(r['max_rel'] for r in rows)
    result=dict(ok=maximum<=1e-4,source_hashes=h.source_hashes(),dtype='float32',n_bytes=512,
                max_relative_error=maximum,limit=1e-4,gradients=rows,norms=norms)
    h.dump(ROOT/'results/h34d/float32.json',result)
    print(json.dumps(result),flush=True)
    assert result['ok']


if __name__=='__main__': main()
