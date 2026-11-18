"""H36: causal allocation of a byte GRU's steps. Principles 2 and 3.

Cheap counts_mix6 matches H33: smoothing .05, seven H32b weight sets,
order-4 initial weights, Adam .003 in 128-byte chunks; counts are added
AFTER that chunk is scored. Contexts reset at documents. No end-of-stream
partial optimizer steps (prefix invariance). GRU state continues across docs;
its predecessor/patch input resets at docs. Patch input contains only bytes
since the previous step, up to and including x[t-1], never x[t].

Adaptive thresholds use per-byte feedback with EMA(.999) score standard
deviation and gain .01, target .25; ties are not chosen. Deciles use a
nearest-rank window of the previous <=4096 entropies. Buckets with <8 samples use independent .25 startup draws, excluded from
threshold and standard-deviation updates; theta starts at the first finite score. Bucket progress compares successive CHOSEN visits.
Frozen tests preserve all learned/running statistics; recurrent/input state
starts fresh, and the fixed stream phase starts at zero.
"""
import argparse
from bisect import bisect_left, bisect_right, insort
from collections import deque
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import h26_cued as h26
import h32b_mixer as h32b

RULES = ('fixed', 'entropy', 'gain', 'progress')
LN2 = math.log(2)
DOC = 4096


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def configure():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class Window:
    def __init__(self):
        self.queue = deque()
        self.sorted = []

    def add(self, value):
        if len(self.queue) == 4096:
            old = self.queue.popleft()
            self.sorted.pop(bisect_left(self.sorted, old))
        self.queue.append(value)
        insort(self.sorted, value)

    def quantile(self, q):
        if not self.sorted:
            return float('inf')
        return self.sorted[max(0, math.ceil(q * len(self.sorted)) - 1)]

    def decile(self, value):
        # Values on tied boundaries go to the upper decile.
        if not self.sorted:
            return 9
        return min(9, sum(value >= self.quantile(q / 10) for q in range(1, 10)))


class Placement:
    def __init__(self, rule, seed, rho=0.25):
        self.rule, self.rho = rule, rho
        self.startup_rng = np.random.default_rng(3_000 + seed)
        self.entropy = Window()
        self.threshold = None
        self.score_mean = None
        self.score_variance = 0.0
        self.ema = np.zeros(130, dtype=np.float64)
        self.samples = np.zeros(130, dtype=np.int64)
        self.previous = np.full(130, np.nan)

    def select(self, entropy, position, doc_position, learn):
        decile = self.entropy.decile(entropy)
        age = min(12, (doc_position + 1).bit_length() - 1)
        bucket = decile * 13 + age
        if self.rule == 'fixed':
            chosen = position % 4 == 3
        else:
            unscored = self.rule in ('gain', 'progress') and self.samples[bucket] < 8
            if unscored:
                chosen = self.startup_rng.random() < .25
            else:
                score = entropy if self.rule == 'entropy' else float(self.ema[bucket])
                assert math.isfinite(score)
                threshold = score if self.threshold is None else self.threshold
                chosen = score > threshold
                if learn:
                    if self.score_mean is None:
                        self.score_mean = score
                    delta = score-self.score_mean
                    self.score_mean += .001*delta
                    self.score_variance = .999*(self.score_variance+.001*delta*delta)
                    self.threshold = threshold + .01*math.sqrt(self.score_variance)*(int(chosen)-.25)
        if self.rho == 1:
            chosen = True
        if learn:
            self.entropy.add(entropy)
        return bool(chosen), bucket

    def observe(self, bucket, cheap_bits, mixed_bits, gru_bits):
        if self.rule not in ('gain', 'progress'):
            return
        if self.rule == 'gain':
            value = cheap_bits - mixed_bits
        else:
            value = 0.0 if np.isnan(self.previous[bucket]) else self.previous[bucket] - gru_bits
        self.previous[bucket] = gru_bits
        if self.samples[bucket] == 0:
            self.ema[bucket] = value
        else:
            self.ema[bucket] = .99 * self.ema[bucket] + .01 * value
        self.samples[bucket] += 1


class Mixer:
    """GeoMixer with batched learning and detached NumPy per-byte scoring."""
    def __init__(self, name, sets, experts, initial):
        init = torch.zeros(sets, experts)
        init[:, initial] = 1
        self.geo = h32b.GeoMixer(name, sets, experts, init)
        self.weights = self.geo.W.detach().numpy().copy()
        self.pending = []

    def predict(self, logs, selector):
        scores = self.weights[selector] @ logs
        # Match H33's one-hot identity convention.
        w = self.weights[selector]
        if np.count_nonzero(w) == 1 and np.max(w) == 1:
            return scores
        peak = np.max(scores)
        return scores - (peak + np.log(np.exp(scores - peak).sum()))

    def observe(self, logs, selector, target):
        self.pending.append((logs.copy(), selector, target))
        if len(self.pending) != 128:
            return
        x = torch.from_numpy(np.stack([v[0] for v in self.pending]))
        s = torch.tensor([v[1] for v in self.pending])
        y = torch.tensor([v[2] for v in self.pending])
        z = torch.einsum('tj,tjv->tv', self.geo.W[s], x)
        loss = F.cross_entropy(z, y) / LN2
        self.geo.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.geo.opt.step()
        self.weights = self.geo.W.detach().numpy().copy()
        self.pending.clear()


class Cheap:
    def __init__(self):
        self.orders = [h32b.OrderCount(k) for k in range(1, 7)]
        self.mixer = Mixer('counts_mix6', 7, 6, 3)
        self.pending = []
        self.dest = np.empty((6, 256), dtype=np.float32)

    def predict(self, blob, i):
        totals = [o.predict_log(blob, i, self.dest[k]) for k, o in enumerate(self.orders)]
        selector = h32b.weight_set_max(totals, 6)
        logs = self.mixer.predict(self.dest, selector)
        return logs, selector

    def observe(self, blob, i, target, selector):
        self.mixer.observe(self.dest, selector, target)
        self.pending.append((blob, i, target))
        if len(self.pending) == 128:
            for doc, pos, symbol in self.pending:
                for order in self.orders:
                    order.add(doc, pos, symbol)
            self.pending.clear()


class PatchGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 32)
        self.gru = nn.GRU(64, 256)
        self.readout = nn.Linear(256, 256)

    def step(self, previous, patch, state):
        prev = self.embedding.weight[previous] if previous >= 0 else self.embedding.weight.new_zeros(32)
        if patch:
            idx = torch.tensor(patch, dtype=torch.long)
            mean = self.embedding(idx).mean(0)
        else:
            mean = self.embedding.weight.new_zeros(32)
        inp = torch.cat((prev, mean)).reshape(1, 1, 64)
        y, state = self.gru(inp, state)
        logits = self.readout(y.reshape(256))
        return F.log_softmax(logits, dim=-1), state


class Session:
    def __init__(self, rule, seed, rho=.25):
        torch.manual_seed(seed)
        self.cheap = Cheap()
        self.model = PatchGRU()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.mixer = Mixer('cheap_gru', 1, 2, 0)
        self.placement = Placement(rule, seed, rho)
        self.state = None
        self.losses = []
        self.position = 0

    def consume(self, documents, labels, learn=True, keep=False, limit=None):
        clock = time.perf_counter()
        sums, counts, steps = {}, {}, {}
        kept_bits, kept_gru, kept_chosen = [], [], []
        total, chosen_total, gru_sum = 0, 0, 0.0
        context = torch.enable_grad() if learn else torch.no_grad()
        with context:
            for doc_index, (doc, kind) in enumerate(zip(documents, labels)):
                blob = bytes(doc) if isinstance(doc, (bytes, bytearray)) else np.asarray(doc, dtype=np.uint8).tobytes()
                patch, previous = [], -1
                sums.setdefault(str(kind), 0.0)
                counts.setdefault(str(kind), 0)
                steps.setdefault(str(kind), 0)
                for i, target in enumerate(blob):
                    if limit is not None and total >= limit:
                        break
                    cheap_log, selector = self.cheap.predict(blob, i)
                    entropy = float(-np.dot(np.exp(cheap_log.astype(np.float64)), cheap_log) / LN2)
                    chosen, bucket = self.placement.select(entropy, self.position, i, learn)
                    cbits = -float(cheap_log[target]) / LN2
                    bits, gbits = cbits, None
                    if chosen:
                        gru_log, self.state = self.model.step(previous, patch, self.state)
                        gbits = -float(gru_log[target].detach()) / LN2
                        expert_logs = np.stack((cheap_log, gru_log.detach().numpy()))
                        mixed = self.mixer.predict(expert_logs, 0)
                        bits = -float(mixed[target]) / LN2
                        if learn:
                            # Score all three distributions before any of them learns target.
                            self.placement.observe(bucket, cbits, bits, gbits)
                            self.losses.append(-gru_log[target])
                            if len(self.losses) == 32:
                                self.opt.zero_grad(set_to_none=True)
                                torch.stack(self.losses).mean().backward()
                                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                                self.opt.step()
                                self.losses.clear()
                                self.state = self.state.detach()
                            self.mixer.observe(expert_logs, 0, target)
                        chosen_total += 1
                        steps[str(kind)] += 1
                        gru_sum += gbits
                        patch.clear()
                    if learn:
                        self.cheap.observe(blob, i, target, selector)
                    sums[str(kind)] += bits
                    counts[str(kind)] += 1
                    total += 1
                    self.position += 1
                    previous = target
                    patch.append(target)
                    if keep:
                        kept_bits.append(bits)
                        kept_gru.append(gbits)
                        kept_chosen.append(chosen)
                if limit is not None and total >= limit:
                    break
                if learn and (doc_index + 1) % 8 == 0:
                    print(f'{self.placement.rule} docs={doc_index+1} bytes={total} rho={chosen_total/total:.6f} wall_s={time.perf_counter()-clock:.1f}', flush=True)
        clean_n = sum(n for k, n in counts.items() if is_clean(k))
        junk_n = total - clean_n
        clean_bits = sum(v for k, v in sums.items() if is_clean(k))
        junk_bits = sum(v for k, v in sums.items() if not is_clean(k))
        junk_steps = sum(v for k, v in steps.items() if not is_clean(k))
        result = dict(n_bytes=total, chosen=chosen_total, rho=chosen_total / total,
                      bits_per_byte=sum(sums.values()) / total, life_bits_clean=clean_bits,
                      life_bits_junk=junk_bits, life_bpb_clean=clean_bits / clean_n if clean_n else None,
                      life_bpb_junk=junk_bits / junk_n if junk_n else None,
                      bits_by_kind=sums, bytes_by_kind=counts, steps_by_kind=steps,
                      junk_share=junk_steps / chosen_total if chosen_total else 0.0,
                      junk_share_by_kind={k: n / chosen_total if chosen_total else 0.0 for k, n in steps.items() if not is_clean(k)},
                      gru_bits_per_step=gru_sum / chosen_total if chosen_total else None,
                      wall_seconds=time.perf_counter()-clock)
        if not math.isfinite(result['bits_per_byte']):
            raise FloatingPointError('Nonfinite stream bits')
        if keep:
            result.update(byte_bits=kept_bits, gru_bits=kept_gru, selected=kept_chosen)
        return result

    def frozen_test(self, documents):
        # Test never updates counts, mixers, bucket EMAs, sample counts, windows.
        self.state = None
        self.position = 0
        self.losses.clear()
        self.model.eval()
        return self.consume(documents, ['clean'] * len(documents), learn=False)


def is_clean(kind):
    return str(kind).lower() in ('clean', 'c', '0')


def standin(raw):
    clean = np.asarray(raw[:262144], dtype=np.uint8).reshape(64, DOC)
    junk = np.random.default_rng(0).integers(0, 256, (64, DOC), dtype=np.uint8)
    docs, labels = [], []
    for c, j in zip(clean, junk):
        docs.extend((c, j)); labels.extend(('clean', 'J1_random'))
    return docs, labels


def corpus(seed):
    """Require H35 for scored lives; never silently substitute stand-in data."""
    module = importlib.import_module('h35_corpus')
    built = module.build(seed)
    # H35's public contract specifies pool, labels and test. Accept a named
    # mapping too, since its author is implementing the module concurrently.
    if isinstance(built, dict):
        pool = built['pool']
        labels = built['labels']
        test = built.get('test', built.get('test_set', built.get('clean_test')))
    else:
        if len(built) != 3:
            raise ValueError('H35 build must return pool, labels, test; adapt to its documented API before dispatch')
        pool, labels, test = built
    if isinstance(test, (bytes, bytearray)):
        test = np.frombuffer(test, dtype=np.uint8)
    test = np.asarray(test, dtype=np.uint8).reshape(-1, DOC)
    assert len(pool) == len(labels) == 3200 and test.shape == (64, DOC)
    assert all(len(doc) == DOC for doc in pool)
    assert all(is_clean(k) or str(k).startswith('J') for k in labels)
    return pool, labels, test


def run(rule, seed, documents=400):
    configure()
    pool, labels, test = corpus(seed)
    session = Session(rule, seed)
    life = session.consume(pool[:documents], labels[:documents])
    assert abs(life['rho']-.25) <= .02, f"achieved rho {life['rho']}"
    frozen = session.frozen_test(test)
    return dict(rule=rule, seed=seed, documents=documents, corpus='h35_corpus.build',
                clean_test_bpb=frozen['bits_per_byte'], test_rho=frozen['rho'], life=life,
                wall_seconds=life['wall_seconds']+frozen['wall_seconds'])


def smoke(data, out):
    """Sequential gates; persist failure and stop immediately on a failed gate."""
    configure()
    raw = h26.load_enwik8(data)
    digest = h26.sha256_arr(raw[:h26.E10M])
    assert digest == h26.E10M_SHA256
    docs, labels = standin(raw)
    report = dict(ok=False, sha256_e10m=digest, threads=torch.get_num_threads(),
                  standin_bytes=524288, prequential={}, rho1={}, allocation={}, documents=None)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    def save():
        dump(output/'smoke.json', report)
    try:
        for rule in RULES:
            a = Session(rule, 0).consume(docs, labels, keep=True, limit=8192)
            b = Session(rule, 0).consume(docs, labels, keep=True, limit=16384)
            gap = float(np.max(np.abs(np.asarray(a['byte_bits'])-np.asarray(b['byte_bits'][:8192]))))
            report['prequential'][rule] = dict(max_abs_bits=gap, identical=a['byte_bits']==b['byte_bits'][:8192])
            print('PREQUENTIAL', rule, report['prequential'][rule], flush=True)
            save()
            assert report['prequential'][rule]['identical'], f'prefix {rule}'
        clean = [np.asarray(raw[i:i+DOC], dtype=np.uint8) for i in range(0,65536,DOC)]
        dense = Session('fixed',0,rho=1).consume(clean, ['clean']*len(clean), keep=True)
        values = np.asarray(dense['gru_bits'], dtype=np.float64)
        first, last = float(values[:8192].mean()), float(values[-8192:].mean())
        report['rho1'] = dict(rho=dense['rho'], first_8192_gru_bpb=first, last_8192_gru_bpb=last,
                              mixed_bpb=dense['bits_per_byte'],wall_seconds=dense['wall_seconds'])
        print('RHO1', report['rho1'], flush=True)
        save()
        assert dense['rho'] == 1 and last < first, 'dense GRU bits did not fall'
        for rule in RULES:
            row = Session(rule,0).consume(docs, labels, limit=262144)
            wall = row['wall_seconds']
            # Include 64-document frozen clean test at the measured training
            # byte rate; no speculative Kaggle CPU speedup.
            row['estimated_400_document_wall_minutes'] = wall * (400+64)/64/60
            row['estimated_300_document_wall_minutes'] = wall * (300+64)/64/60
            report['allocation'][rule] = row
            print('ALLOCATION',rule,json.dumps(row),flush=True)
            save()
            assert abs(row['rho']-.25) <= .02, f"allocation {rule}: rho={row['rho']:.9f} outside [0.23,0.27]"
            assert math.isfinite(row['junk_share']) and 0 <= row['junk_share'] <= 1
        maximum = max(v['estimated_400_document_wall_minutes'] for v in report['allocation'].values())
        report['documents'] = 400 if maximum <= 85 else 300
        assert max(v[f"estimated_{report['documents']}_document_wall_minutes"] for v in report['allocation'].values()) <= 85, '300-document wall exceeds 85 minutes'
        report['ok'] = True
        save()
        print('SMOKE',json.dumps(report),flush=True)
    except Exception as exc:
        report['failure'] = str(exc)
        save()
        print('SMOKE FAILED:',str(exc),flush=True)
        raise


def budget_smoke(data, out, rules=RULES):
    """Checks 3–5, then a real-corpus 64 KB allocation check for every rule.

    Per-rule partial results persist on failure. Smoke success requires all
    four rules; a selected subset never authorizes dispatch.
    """
    configure()
    raw = h26.load_enwik8(data)
    digest = h26.sha256_arr(raw[:h26.E10M])
    assert digest == h26.E10M_SHA256
    docs, labels = standin(raw)
    output = Path(out)
    path = output/'smoke.json'
    report = dict(ok=False, sha256_e10m=digest, threads=torch.get_num_threads(),
                  controller='feedback .01 * EMA(.999) score std; strict ties; unscored<8 rng3000 .25',
                  allocation={}, h35_checks={}, documents=None)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('controller') == report['controller']:
            report.update(old)
            report['ok'] = False
            report.pop('failure', None)
    def save():
        dump(path, report)
    try:
        for rule in rules:
            row = Session(rule,0).consume(docs, labels, limit=262144)
            wall = row['wall_seconds']
            row['estimated_400_document_wall_minutes'] = wall*(400+64)/64/60
            row['estimated_300_document_wall_minutes'] = wall*(300+64)/64/60
            report['allocation'][rule] = row
            print('ALLOCATION',rule,json.dumps(row),flush=True)
            save()
            assert abs(row['rho']-.25) <= .02, f"allocation {rule}: rho={row['rho']:.9f} outside [0.23,0.27]"
            assert math.isfinite(row['junk_share']) and 0 <= row['junk_share'] <= 1
        pool, kinds, _ = corpus(0)
        for rule in rules:
            row = Session(rule,0).consume(pool,kinds,limit=65536)
            report['h35_checks'][rule] = dict(rule=rule,seed=0,corpus='h35_corpus.build(0)',**row)
            print('H35_CHECK',rule,json.dumps(report['h35_checks'][rule]),flush=True)
            save()
            assert abs(row['rho']-.25) <= .02, f"H35 allocation {rule}: rho={row['rho']}"
            assert math.isfinite(row['junk_share']) and 0 <= row['junk_share'] <= 1
        if set(report['allocation']) != set(RULES) or set(report['h35_checks']) != set(RULES):
            save()
            return
        maximum = max(v['estimated_400_document_wall_minutes'] for v in report['allocation'].values())
        report['documents'] = 400 if maximum <= 85 else 300
        assert max(v[f"estimated_{report['documents']}_document_wall_minutes"] for v in report['allocation'].values()) <= 85, '300-document wall exceeds 85 minutes'
        report['ok'] = True
        save()
    except Exception as exc:
        report['failure'] = str(exc)
        save()
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0,choices=range(6))
    p.add_argument('--rule',choices=RULES)
    p.add_argument('--documents',type=int,choices=(300,400),default=400)
    p.add_argument('--data',default='data/enwik8/enwik8')
    p.add_argument('--out',default='results/h36')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--budget-smoke',action='store_true')
    p.add_argument('--smoke-rules',nargs='+',choices=RULES,default=list(RULES))
    args = p.parse_args()
    if args.budget_smoke:
        budget_smoke(args.data,args.out,args.smoke_rules)
    elif args.smoke:
        smoke(args.data,args.out)
    else:
        if args.rule is None:
            p.error('--rule is required for a scored kernel')
        result = run(args.rule,args.seed,args.documents)
        result['kernel_id'] = f'ezzzzzx/h36-{args.rule}-s{args.seed}'
        dump(Path(args.out)/'results.json',result)
        print(json.dumps(result),flush=True)

if __name__ == '__main__':
    main()
