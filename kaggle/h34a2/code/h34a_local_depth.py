"""H34a: local depth, prequential CPU experiment. Numbers only; no verdict."""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import h33_bank_text as h33
import h30w_ring as h30w
import h30x_trace as h30x
import h30y_roots as h30y
import h32b_mixer as h32b

CHUNK, EMB, RANK, A = 128, 32, 64, 256
ARMS = ('plain_1', 'plain_2', 'plain_4', 'left_1', 'left_2', 'left_4', 'left_4_noskip', 'wide_1')
LOG2 = math.log(2)


def dump(path, data):
    h33.dump(path, data)


class Layer(nn.Module):
    def __init__(self, n=128, prev=0, skip=True, zero=True):
        super().__init__()
        self.n, self.prev, self.skip = n, prev, skip
        self.has_emb = not prev or skip
        self.dim = (EMB if self.has_emb else 0) + prev
        # Preserve Bank's RNG draw order, including the overwritten roots draws.
        if self.has_emb:
            self.embedding = nn.Embedding(A, EMB)
        mag = torch.empty(n).uniform_(0.9, 0.999)
        theta = torch.empty(n).uniform_(0, math.pi)
        self.nu = nn.Parameter(torch.log(-torch.log(mag)))
        self.theta = nn.Parameter(theta)
        self.B_re = nn.Parameter(torch.randn(n, self.dim) / math.sqrt(self.dim))
        self.B_im = nn.Parameter(torch.randn(n, self.dim) / math.sqrt(self.dim))
        self.W = nn.Linear(2*n, A, bias=False)
        self.P1 = nn.Linear(2*n, RANK, bias=False)
        self.P2 = nn.Linear(2*n, RANK, bias=False)
        self.V = nn.Linear(RANK, A, bias=False)
        self.register_buffer('rms2', torch.ones(2*n+RANK))
        with torch.no_grad():
            self.nu.fill_(math.log(-math.log(0.98)))
            self.theta.copy_((2*math.pi*torch.arange(n, dtype=torch.float64)/n).float())
            if zero:
                self.W.weight.zero_()
                self.V.weight.zero_()

    def lam(self):
        return torch.exp(-torch.exp(self.nu) + 1j*self.theta)

    def features(self, h):
        z = torch.cat((h.real, h.imag), -1)
        q = self.P1(z)*self.P2(z)
        return z, q

    def logits_from(self, h):
        z, q = self.features(h)
        return self.W(z)+self.V(q)

    def initial(self):
        dtype = torch.complex128 if self.nu.dtype == torch.float64 else torch.complex64
        return (torch.zeros(self.n, dtype=dtype), torch.zeros(self.n, dtype=dtype),
                torch.zeros(self.n, self.dim, dtype=dtype),
                torch.zeros(self.n, A if self.has_emb else 0, dtype=dtype))

    def inputs(self, tokens, start, end, lower=None, lower_rms=None):
        parts = []
        if self.has_emb:
            e = self.embedding(tokens[max(0, start-1):end-1]).detach()
            if start == 0:
                e = torch.cat((e.new_zeros(1, EMB), e))
            parts.append(e)
        if lower is not None:
            parts.append(lower.detach()/lower_rms.detach().sqrt())
        return torch.cat(parts, -1)

    def score(self, u, tokens, start, state):
        h, el, eb, ez = state
        length = len(u)
        with torch.no_grad():
            lam = self.lam()
            inj = torch.complex(u @ self.B_re.T, u @ self.B_im.T)
            hs, els, ebs, ezs = [], [], [], []
            for t in range(length):
                el = lam*el+h
                eb = lam[:, None]*eb+u[t]
                ez = lam[:, None]*ez
                if self.has_emb and start+t > 0:
                    ez[:, int(tokens[start+t-1])] += 1
                h = lam*h+inj[t]
                hs.append(h)
                els.append(el)
                ebs.append(eb)
                if self.has_emb:
                    ezs.append(ez)
            hb = torch.stack(hs)
        hr = hb.real.detach().requires_grad_()
        hi = hb.imag.detach().requires_grad_()
        z, q = self.features(torch.complex(hr, hi))
        logits = self.W(z)+self.V(q)
        features = torch.cat((z, q), -1).detach()
        cache = (hr, hi, torch.stack(els), torch.stack(ebs), torch.stack(ezs) if ezs else None)
        return logits, features, (h, el, eb, ez), cache

    def learn(self, logits, g, cache):
        hr, hi, el, eb, ez = cache
        self.zero_grad(set_to_none=True)
        logits.backward(g)
        with torch.no_grad():
            c = torch.complex(hr.grad, hi.grad).conj()
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

    def update_rms(self, features):
        # Per-byte EMA, committed only after the whole chunk has been scored.
        with torch.no_grad():
            for row in features:
                self.rms2.mul_(0.999).addcmul_(row, row, value=0.001)
            if not torch.isfinite(self.rms2).all():
                raise FloatingPointError('nonfinite RMS')


def build(arm, seed, zero=True, n_override=None):
    torch.manual_seed(seed)
    d = 1 if arm == 'wide_1' else int(arm.split('_')[1])
    n = n_override or (512 if arm == 'wide_1' else 128)
    return [Layer(n, 0 if k == 0 else 2*n+RANK, arm != 'left_4_noskip', zero) for k in range(d)]


def compute_account(arm):
    layers = build(arm, 0)
    rows = []
    for m in layers:
        n, d, a, r = m.n, m.dim, A, RANK
        e = EMB if m.has_emb else 0
        p = a*e + 2*n + 2*n*d + 2*n*a + 4*n*r + a*r
        assert p == sum(x.numel() for x in m.parameters())
        terms = dict(inject=2*n*d, state=4*n, trace_lambda=4*n,
                     trace_B=4*n*d, trace_embedding=4*n*a if e else 0,
                     readouts=3*(2*n*a+4*n*r+a*r), quadratic=3*r,
                     grad_roots=16*n, grad_B=8*n*d,
                     grad_embedding_coeff=4*n*a if e else 0,
                     grad_embedding=4*n*a*e, dlam=4*n/CHUNK,
                     softmax=3*a, rms=3*(2*n+r),
                     normalize=m.prev, adam=6*p/CHUNK)
        rows.append(dict(parameters=p, input_dim=d, terms=terms,
                         macs_per_byte=sum(terms.values()),
                         trace_floats=2*n*(1+d+(a if e else 0)), state_floats=2*n))
    d = len(layers)
    # Detached expert logits: mixer forward and weight gradient, no input gradient.
    mixer_macs = 2*d*A + 3*A + 6*d/CHUNK
    return dict(parameters=sum(x['parameters'] for x in rows)+d,
                macs_per_byte=sum(x['macs_per_byte'] for x in rows)+mixer_macs,
                trace_floats=sum(x['trace_floats'] for x in rows),
                state_floats=sum(x['state_floats'] for x in rows),
                chunk_trace_cache_floats=CHUNK*sum(x['trace_floats'] for x in rows),
                mixer_macs_per_byte=mixer_macs, layers=rows)


def run(arm, seed, raw, frozen_mix=False, zero=True, n_override=None, bypass=False, keep=False, dtype=torch.float32, initial_parameters=None):
    t0 = time.perf_counter()
    tokens = torch.tensor(np.array(raw, dtype=np.int64))
    layers = build(arm, seed, zero, n_override)
    layers = [m.to(dtype=dtype) for m in layers]
    if initial_parameters is not None:
        # Smoke check 1 only: compare learning from identical reference weights.
        with torch.no_grad():
            for name, p in layers[0].named_parameters():
                p.copy_(initial_parameters[name])
    states = [m.initial() for m in layers]
    opts = [torch.optim.Adam(m.parameters(), lr=0.001) for m in layers]
    mixer = h32b.GeoMixer(arm, 1, len(layers), torch.ones(1, len(layers), dtype=dtype))
    bits = np.empty(len(raw), np.float64)
    own = np.empty((len(raw), len(layers)), np.float64)
    weight_sum = np.zeros(len(layers))
    weight_n = 0
    for start in range(0, len(raw), CHUNK):
        end = min(start+CHUNK, len(raw))
        target = tokens[start:end]
        outputs, feats, caches = [], [], []
        for k, m in enumerate(layers):
            u = m.inputs(tokens, start, end, feats[-1] if k else None, layers[k-1].rms2 if k else None)
            logits, f, states[k], cache = m.score(u, tokens, start, states[k])
            outputs.append(logits)
            feats.append(f)
            caches.append(cache)
        # Weighted logits are exactly the geometric mix: log p_k differs by
        # a class-independent constant, removed by log-softmax. Zero logits
        # therefore implement a numerically exact identity expert.
        stacked = torch.stack([x.detach() for x in outputs], 1)
        mixlogits = (stacked*mixer.W[0, :, None]).sum(1)
        if bypass:
            mixlogits = outputs[0].detach()
        losses = F.cross_entropy(mixlogits, target, reduction='none')
        bits[start:end] = losses.detach().double().numpy()/LOG2
        for k, logits in enumerate(outputs):
            own[start:end, k] = F.cross_entropy(logits.detach(), target, reduction='none').double().numpy()/LOG2
        take = max(0, end-max(start, len(raw)-50_000))
        weight_sum += mixer.W.detach().numpy()[0]*take
        weight_n += take
        # All scores exist before any model, mixer, or RMS update.
        mixerr = mixlogits.detach().softmax(-1)
        mixerr[torch.arange(end-start), target] -= 1
        for k, m in enumerate(layers):
            if arm.startswith('left'):
                g = mixerr.clone()
            else:
                g = outputs[k].detach().softmax(-1)
                g[torch.arange(end-start), target] -= 1
            m.learn(outputs[k], g/(end-start), caches[k])
        if not frozen_mix and not bypass:
            mixer.opt.zero_grad(set_to_none=True)
            losses.mean().backward()
        for m, opt, f in zip(layers, opts, feats):
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
            m.update_rms(f)
        if not frozen_mix and not bypass:
            mixer.opt.step()
        if not np.isfinite(bits[start:end]).all():
            raise FloatingPointError('nonfinite bits')
        if end % 50_048 == 0:
            print(arm, seed, end, bits[:end].mean(), time.perf_counter()-t0, flush=True)
        if time.perf_counter()-t0 > 5400:
            raise TimeoutError('90 minute limit')
    result = dict(arm=arm, seed=seed, n_bytes=len(raw), bits_per_byte=float(bits.mean()),
                  slices=[dict(start=i, end=min(i+50_000, len(raw)), bits_per_byte=float(bits[i:i+50_000].mean()))
                          for i in range(0, len(raw), 50_000)],
                  layers=[dict(own_bits_per_byte=float(own[:, k].mean()), mean_mixer_weight_last_50kb=float(weight_sum[k]/weight_n),
                               fraction_lambda_lt_half=float((m.lam().abs()<0.5).float().mean()),
                               feature_rms=m.rms2.sqrt().tolist()) for k, m in enumerate(layers)],
                  wall_seconds=time.perf_counter()-t0)
    if keep:
        return result, bits
    return result


def smoke(out, data):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    payload = dict(ok=False, log=log, dtype='float32', check1_dtype='float64', threads=1)
    def record(name, **values):
        row = dict(check=name, **values)
        log.append(row)
        print(json.dumps(row), flush=True)
        dump(out/'smoke.json', payload)
    raw, digest = h33.load_checked(data)
    payload['sha256_e10m'] = digest
    # 1. The reference H33 runner, same non-zero init and mixer bypass.
    ref_model = h30y.build_arm('roots_learn', A, 0).double()
    initial_parameters = {name: p.detach().clone() for name, p in ref_model.named_parameters()}
    ref_state = h30x.init_state(ref_model)
    opt = torch.optim.Adam(ref_model.parameters(), lr=0.001)
    tokens = torch.tensor(raw[:4096].astype(np.int64))
    ref = []
    for s in range(0, len(tokens), CHUNK):
        loss, ref_state = h30x.trace_chunk(ref_model, tokens, s, s+CHUNK, ref_state, True)
        ref.extend(loss.double().numpy()/LOG2)
        nn.utils.clip_grad_norm_(ref_model.parameters(), 1.)
        opt.step()
    _, ours = run('plain_1', 0, raw[:4096], zero=False, n_override=256, bypass=True, keep=True, dtype=torch.float64,
                  initial_parameters=initial_parameters)
    diff = float(np.max(np.abs(ours-ref)))
    record('h33_equivalence', max_bits_abs=diff, limit=1e-6, copied_reference_initial_parameters=True)
    if diff > 1e-6:
        raise RuntimeError('smoke 1 failed')
    # 2. Full BPTT of layer 2, with same-time layer-1 features detached.
    layers = build('plain_2', 0, zero=False)
    tokens = torch.tensor(raw[:512].astype(np.int64))
    low = layers[0]
    u1 = low.inputs(tokens, 0, 512)
    _, f1, _, _ = low.score(u1, tokens, 0, low.initial())
    m = layers[1]
    u = m.inputs(tokens, 0, 512, f1, low.rms2)
    logits, _, _, cache = m.score(u, tokens, 0, m.initial())
    g = logits.detach().softmax(-1)
    g[torch.arange(512), tokens] -= 1
    m.learn(logits, g, cache)
    trace = {name:p.grad.clone() for name,p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    e = m.embedding(tokens[:-1])
    e = torch.cat((e.new_zeros(1, EMB), e))
    ub = torch.cat((e, f1.detach()/low.rms2.sqrt()), -1)
    inj = torch.complex(ub@m.B_re.T, ub@m.B_im.T)
    h = m.initial()[0]
    hs = []
    lam = m.lam()
    for row in inj:
        h = lam*h+row
        hs.append(h)
    F.cross_entropy(m.logits_from(torch.stack(hs)), tokens, reduction='sum').backward()
    rows = [h30x.grad_pair(name, trace[name], p.grad) for name,p in m.named_parameters()]
    maximum = max(row['max_rel'] for row in rows if row['name'] in ('nu','theta','B_re','B_im'))
    record('exact_traces', max_relative_error=maximum, gradients=rows, limit=1e-4)
    if maximum > 1e-4:
        raise RuntimeError('smoke 2 failed')
    # 3. Non-uniform base distribution plus a fresh upper layer.
    lower = Layer(zero=False)
    upper = Layer(prev=320)
    tok = torch.tensor(raw[:128].astype(np.int64))
    lo, f, _, _ = lower.score(lower.inputs(tok,0,128), tok,0,lower.initial())
    up, _, _, _ = upper.score(upper.inputs(tok,0,128,f,lower.rms2),tok,0,upper.initial())
    identity = float((lo.softmax(-1)-(lo+up).softmax(-1)).abs().max())
    record('identity', max_logit_abs=float(up.abs().max()), max_probability_abs=identity, limit=1e-9)
    if up.count_nonzero() or identity > 1e-9:
        raise RuntimeError('smoke 3 failed')
    _, plain = run('plain_1',0,raw[:8192],frozen_mix=True,keep=True)
    _, left = run('left_1',0,raw[:8192],frozen_mix=True,keep=True)
    eq = float(np.max(np.abs(plain-left)))
    record('plain_left_D1',max_bits_abs=eq)
    if eq != 0:
        raise RuntimeError('smoke 4 failed')
    _, short = run('left_4',0,raw[:1024],keep=True)
    _, long = run('left_4',0,raw[:2048],keep=True)
    eq = float(np.max(np.abs(short-long[:1024])))
    record('prequential',max_bits_abs=eq)
    if eq != 0:
        raise RuntimeError('smoke 5 failed')
    for arm in ARMS:
        r = run(arm,0,raw[:8192])
        record('finite',arm=arm,bits_per_byte=r['bits_per_byte'],wall_seconds=r['wall_seconds'])
    timings = {}
    for arm in ('left_4','wide_1'):
        r = run(arm,0,raw[:32768])
        timings[arm] = {str(n):r['wall_seconds']*n/32768 for n in (500000,300000)}
        record('timing',arm=arm,wall_seconds=r['wall_seconds'],estimated_kernel_seconds=timings[arm])
    n = 300000 if max(t['500000'] for t in timings.values())>5400 else 500000
    payload.update(ok=True,n_bytes=n,timing=timings)
    dump(out/'smoke.json',payload)


def assemble(out):
    out = Path(out)
    smoke_result = json.loads((out/'smoke.json').read_text())
    paths = [out/'kernels'/f'{arm}-s{seed}'/'results.json' for arm in ARMS for seed in (0, 1)]
    runs = [json.loads(path.read_text()) for path in paths]
    assert smoke_result['ok'] and len(runs) == 16
    assert all(r['n_bytes'] == smoke_result['n_bytes'] for r in runs)
    lookup = {(r['arm'], r['seed']): r for r in runs}
    means = {arm: float(np.mean([lookup[arm, s]['bits_per_byte'] for s in (0, 1)])) for arm in ARMS}
    pairs = [('left_4', 'left_1'), ('plain_4', 'plain_1'), ('left_4', 'plain_4'),
             ('left_2', 'plain_2'), ('left_4', 'wide_1'), ('left_4_noskip', 'left_4')]
    gaps = []
    for a, b in pairs:
        values = [lookup[a, s]['bits_per_byte']-lookup[b, s]['bits_per_byte'] for s in (0, 1)]
        gaps.append(dict(gap=f'{a} - {b}', per_seed=values, mean=float(np.mean(values))))
    compute = {arm: compute_account(arm) for arm in ARMS}
    ratio = compute['wide_1']['macs_per_byte']/compute['left_4']['macs_per_byte']
    formulas = dict(
        convention='Real GEMM entry = 1 MAC; complex multiply or multiply-add = 4 real MACs. Adam = 6 P/chunk. Transcendentals, comparisons, reductions, memory copies and scoring-only diagnostics excluded.',
        parameters='A E + 2N + 2N U + 2N A + 4N R + A R, plus D mixer weights; E=32 with embedding, otherwise 0.',
        macs='2NU + 8N + 4NU + 4NA I + 3(2NA+4NR+AR) + 3R + 16N + 8NU + 4NA I + 4NAE + 4N/128 + 3A + 3(2N+R) + Fprev + 6P/128 per layer; mixer 2DA+3A+6D/128. I=1 iff embedding present.',
        traces='2N(1+U+A I) real floats per layer: complex lambda, input-map and embedding traces; hidden state 2N extra; chunk caches 128 times persistent trace floats.')
    result = dict(n_bytes=smoke_result['n_bytes'], runs=runs, means=means, gaps=gaps,
                  compute=compute, compute_formulas=formulas, wide_over_left4_macs=ratio,
                  wide_compute_diff_gt_20pct=bool(abs(ratio-1) > .2), smoke=smoke_result,
                  h33_reference_first_1mb=dict(bank=3.097826, counts_mix6=2.177547, GRU=2.104744),
                  kernel_ids=[r['kernel_id'] for r in runs])
    dump(out/'results.json', result)
    lines = ['# H34a', '', f"Bytes per life: {result['n_bytes']:,}. CPU; float32; chunks 128; seeds 0, 1.", '',
             '| Arm | Seed 0 bits/byte | Seed 1 bits/byte | Mean |', '|---|---:|---:|---:|']
    for arm in ARMS:
        lines.append(f"| {arm} | {lookup[arm,0]['bits_per_byte']:.9f} | {lookup[arm,1]['bits_per_byte']:.9f} | {means[arm]:.9f} |")
    lines += ['', '| Gap | Seed 0 | Seed 1 | Mean |', '|---|---:|---:|---:|']
    for g in gaps:
        lines.append(f"| {g['gap']} | {g['per_seed'][0]:.9f} | {g['per_seed'][1]:.9f} | {g['mean']:.9f} |")
    lines += ['', '| Arm | Seed | Slice start | Slice end | Bits/byte |', '|---|---:|---:|---:|---:|']
    for run_result in runs:
        for s in run_result['slices']:
            lines.append(f"| {run_result['arm']} | {run_result['seed']} | {s['start']} | {s['end']} | {s['bits_per_byte']:.9f} |")
    lines += ['', '| Arm | Seed | Layer | Own bits/byte | Mean weight, last 50 KB | Fraction abs(lambda)<0.5 | RMS min | RMS mean | RMS max |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        for k, layer in enumerate(r['layers'], 1):
            rms = np.asarray(layer['feature_rms'])
            lines.append(f"| {r['arm']} | {r['seed']} | {k} | {layer['own_bits_per_byte']:.9f} | {layer['mean_mixer_weight_last_50kb']:.9f} | {layer['fraction_lambda_lt_half']:.9f} | {rms.min():.9g} | {rms.mean():.9g} | {rms.max():.9g} |")
    lines += ['', 'Full per-feature final running RMS vectors: `results.json`, `runs[].layers[].feature_rms` (z then quadratic features).', '',
              '| Arm | Parameters incl. mixer | MACs/byte, forward + learning | Trace floats | State floats | Chunk trace-cache floats |',
              '|---|---:|---:|---:|---:|---:|']
    for arm, c in compute.items():
        lines.append(f"| {arm} | {c['parameters']} | {c['macs_per_byte']:.3f} | {c['trace_floats']} | {c['state_floats']} | {c['chunk_trace_cache_floats']} |")
    lines += ['', *(f'{key}: {value}' for key, value in formulas.items()), '',
              f"wide_1 / left_4 MACs = {ratio:.9f}; difference exceeds 20% = {result['wide_compute_diff_gt_20pct']}.", '',
              '| H33 reference, first 1 MB (not directly comparable) | Bits/byte |', '|---|---:|',
              '| bank | 3.097826 |', '| counts_mix6 | 2.177547 |', '| GRU | 2.104744 |', '',
              '| Kernel id | Wall seconds |', '|---|---:|']
    lines += [f"| {r['kernel_id']} | {r['wall_seconds']:.3f} |" for r in runs]
    lines += ['', 'Smoke-test output (1 CPU thread; check 1 float64, copied reference initialization; all scored runs float32):',
              '', '```json', json.dumps(smoke_result, indent=2), '```', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--assemble',action='store_true')
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--n-bytes',type=int,default=500000)
    p.add_argument('--data',default='data/enwik8')
    p.add_argument('--out',default='results/h34a')
    args = p.parse_args()
    h33.configure(1)
    if args.assemble:
        assemble(args.out)
        return
    if args.smoke:
        try:
            smoke(args.out,args.data)
        except Exception as exc:
            print('SMOKE FAILED:',repr(exc),flush=True)
            path = Path(args.out)/'smoke.json'
            payload = json.loads(path.read_text()) if path.exists() else dict(ok=False)
            payload['failure']=repr(exc)
            dump(path,payload)
            raise
        return
    raw,digest=h33.load_checked(args.data)
    r=run(args.arm,args.seed,raw[:args.n_bytes])
    r.update(sha256_e10m=digest,kernel_id=f'ezzzzzx/h34a-{args.arm.replace("_","-")}-s{args.seed}')
    dump(Path(args.out)/'results.json',r)


if __name__=='__main__':
    main()
