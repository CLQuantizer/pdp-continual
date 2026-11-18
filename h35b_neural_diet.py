"""H35b: shared byte-GRU data diet; principles 3 and 4.

A document resets hidden state, never weights or Adam moments. Inputs are the
previous byte (zero embedding at document start); targets cannot see themselves.
Training detaches hidden every 64 bytes, scores then steps every 128 bytes.
Trial updates use one Adam step on 256 bytes, with the same 64-byte truncation.
Progress evaluates the candidate tail with its own frozen 256-byte prefix as
context, and reservoir pieces with reset hidden state. Fixed denominator 1280,
even while the four-piece reservoir is filling; one first-256-byte piece per
read document, as H35. All policies start with identical seeded weights.
"""
import argparse
from collections import Counter
import copy
import hashlib
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
import h35_corpus as corpus

POLICIES = ('rand', 'oracle', 'high_loss', 'rho', 'prog_peek')
SEEDS = tuple(range(6, 12))
LN2 = math.log(2)
K = 16


def configure():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def tensor(doc):
    if isinstance(doc, bytes):
        doc = np.frombuffer(doc, dtype=np.uint8)
    return torch.tensor(np.asarray(doc, dtype=np.int64), dtype=torch.long)


class ByteGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 32)
        self.gru = nn.GRU(32, 256)
        self.readout = nn.Linear(256, 256)

    def forward(self, targets, start, end, hidden=None):
        # targets is [time, batch]. All inputs are shifted causal bytes.
        if start == 0:
            first = self.embedding.weight.new_zeros((1, targets.shape[1], 32))
            inputs = torch.cat((first, self.embedding(targets[:end-1])), dim=0)
        else:
            inputs = self.embedding(targets[start-1:end-1])
        output, hidden = self.gru(inputs, hidden)
        return self.readout(output), hidden


class Learner:
    def __init__(self, seed):
        torch.manual_seed(seed)
        self.model = ByteGRU()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def read(self, doc, keep=False, step_bytes=128):
        """One prequential pass; reset hidden at start, carry it between steps."""
        data = tensor(doc).reshape(-1, 1)
        hidden, scored = None, []
        for chunk in range(0, len(data), step_bytes):
            end_chunk = min(chunk + step_bytes, len(data))
            losses = []
            for start in range(chunk, end_chunk, 64):
                end = min(start+64, end_chunk)
                logits, hidden = self.model(data, start, end, hidden)
                losses.append(F.cross_entropy(logits.flatten(0, 1), data[start:end].flatten(), reduction='none'))
                hidden = hidden.detach()
            loss = torch.cat(losses)
            scored.append(loss.detach().double().numpy()/LN2)
            self.opt.zero_grad(set_to_none=True)
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
        bits = np.concatenate(scored)
        return bits if keep else float(bits.sum())

    @torch.no_grad()
    def frozen_totals(self, docs, start_score=0, batch_size=8):
        """Per-document bits; equal-length docs, context before start_score unscored."""
        totals = []
        for offset in range(0, len(docs), batch_size):
            group = docs[offset:offset+batch_size]
            data = torch.stack([tensor(d) for d in group], dim=1)
            hidden = None
            total = torch.zeros(len(group), dtype=torch.float64)
            for start in range(0, len(data), 128):
                end = min(start+128, len(data))
                logits, hidden = self.model(data, start, end, hidden)
                lo = max(start_score, start)-start
                if lo < end-start:
                    losses = F.cross_entropy(logits[lo:].flatten(0, 1), data[start+lo:end].flatten(), reduction='none')
                    total += losses.reshape(-1, len(group)).double().sum(0)/LN2
            totals.extend(total.tolist())
        return np.asarray(totals)

    def test(self, docs):
        return float(self.frozen_totals(docs).sum()/sum(map(len, docs)))

    def checksum(self):
        """Hash tensor bytes and optimizer structure/hyperparameters deterministically."""
        h = hashlib.sha256()
        def visit(obj):
            if torch.is_tensor(obj):
                h.update(str((obj.dtype, tuple(obj.shape))).encode())
                h.update(obj.detach().cpu().numpy().tobytes())
            elif isinstance(obj, dict):
                for key in sorted(obj, key=repr):
                    h.update(repr(key).encode()); visit(obj[key])
            elif isinstance(obj, (list, tuple)):
                h.update(str(len(obj)).encode())
                for value in obj:
                    visit(value)
            else:
                h.update(repr(obj).encode())
        visit(self.model.state_dict())
        visit(self.opt.state_dict())
        # Gradients are also live state; a trial must not modify them.
        visit([p.grad for p in self.model.parameters()])
        return h.hexdigest()


class Reservoir:
    def __init__(self, seed):
        self.rng = np.random.default_rng(2_000 + seed)
        self.pieces, self.n = [], 0

    def add(self, doc):
        piece = np.asarray(doc[:256], dtype=np.uint8).copy()
        self.n += 1
        if len(self.pieces) < 4:
            self.pieces.append(piece)
        else:
            j = int(self.rng.integers(self.n))
            if j < 4:
                self.pieces[j] = piece


def progress_score(live, candidate, pieces, before_reservoir=None, before_tail=None):
    """Discarded trial copies both parameters and optimizer moments/step counters."""
    if before_tail is None:
        before_tail = float(live.frozen_totals([candidate[:512]], start_score=256)[0])
    if before_reservoir is None:
        before_reservoir = float(live.frozen_totals(pieces).sum()) if pieces else 0.
    trial = copy.deepcopy(live)
    trial.read(candidate[:256], step_bytes=256)  # exactly ONE Adam step
    after_tail = float(trial.frozen_totals([candidate[:512]], start_score=256)[0])
    after_reservoir = float(trial.frozen_totals(pieces).sum()) if pieces else 0.
    return (before_tail + before_reservoir - after_tail - after_reservoir)/1280.


def choose(policy, live, pool, unread, rng, reservoir, reference, ref_cache, labels=None):
    ids = np.flatnonzero(unread)
    if policy == 'rand':
        return int(rng.choice(ids))
    if policy == 'oracle':
        return int(rng.choice(np.flatnonzero(unread & (labels == 'clean'))))
    candidates = rng.choice(ids, size=K, replace=False)
    peeks = pool[candidates, :512]
    if policy == 'prog_peek':
        pieces = reservoir.pieces
        res_before = float(live.frozen_totals(pieces).sum()) if pieces else 0.
        tails = live.frozen_totals(peeks, start_score=256)
        scores = [progress_score(live, doc, pieces, res_before, tail) for doc, tail in zip(peeks, tails)]
    else:
        scores = live.frozen_totals(peeks)/512.
        if policy == 'rho':
            fresh = [int(idx) for idx in candidates if int(idx) not in ref_cache]
            if fresh:
                values = reference.frozen_totals(pool[fresh, :512])/512.
                ref_cache.update(zip(fresh, values.tolist()))
            scores -= np.array([ref_cache[int(idx)] for idx in candidates])
        elif policy != 'high_loss':
            raise ValueError(policy)
    assert np.isfinite(scores).all()
    return int(candidates[int(np.argmax(scores))])


def make_reference(seed, raw):
    ref = Learner(seed)
    data = corpus.reference_bytes(raw)
    # Reference text is a contiguous stream, as H35; last short chunk stepped.
    ref.read(data)
    for p in ref.model.parameters():
        p.requires_grad_(False)
    return ref


def run_policy(seed, policy, budget, pool, labels, test, reference=None, checkpoint=25):
    clock = time.perf_counter()
    live = Learner(seed)
    rng = np.random.default_rng(1_000 + seed)
    reservoir, cache = Reservoir(seed), {}
    unread = np.ones(len(pool), dtype=bool)
    selected, checkpoints, life_bits = [], [], 0.
    for count in range(1, budget+1):
        idx = choose(policy, live, pool, unread, rng, reservoir, reference, cache,
                     labels=labels if policy == 'oracle' else None)
        doc = pool[idx]
        life_bits += live.read(doc)
        unread[idx] = False
        selected.append(idx)
        if policy == 'prog_peek':
            reservoir.add(doc)
        if count % checkpoint == 0:
            value = live.test(test)
            checkpoints.append(dict(documents=count, clean_bpb=value))
            print(f'{policy} s{seed} B{count} clean_bpb={value:.6f} wall_s={time.perf_counter()-clock:.1f}', flush=True)
    kinds = Counter(str(labels[idx]) for idx in selected)
    return dict(seed=seed, policy=policy, budget=budget, pool_documents=len(pool),
                checkpoint_every=checkpoint, checkpoints=checkpoints,
                T_clean=float(np.mean([p['clean_bpb'] for p in checkpoints])) if checkpoints else None,
                final=checkpoints[-1]['clean_bpb'] if checkpoints and checkpoints[-1]['documents']==budget else None,
                kinds_read={k:kinds[k] for k in corpus.KINDS},
                shares={k:kinds[k]/budget for k in corpus.KINDS}, selected=selected,
                life_bits=life_bits, wall_seconds=time.perf_counter()-clock,
                reservoir_piece='doc[0:256]', progress_denominator=1280)


def load(seed, data):
    raw, digest = corpus.load_raw(data)
    pool, labels, test = corpus.build(seed, raw=raw)
    return raw, pool[:1600], labels[:1600], test, digest


def kernel_main(seed, out, kernel_id, policies=POLICIES, data='data/enwik8/enwik8', fingerprint=None):
    configure()
    clock = time.perf_counter()
    raw, pool, labels, test, digest = load(seed, data)
    ref = make_reference(seed, raw) if 'rho' in policies else None
    reference_seconds = time.perf_counter()-clock
    runs = []
    for policy in policies:
        runs.append(run_policy(seed, policy, 200, pool, labels, test, ref))
        dump(Path(out)/'results.json', dict(seed=seed, kernel_id=kernel_id, runs=runs,
             policies=list(policies), sha256_e10m=digest, code_fingerprint=fingerprint,
             reference_setup_seconds=reference_seconds, wall_seconds=time.perf_counter()-clock))
    return runs


def plan_kernels(report):
    """Schedule fixed policies only; never change scientific parameters.

    The initial estimate is sequential, with no assumed Kaggle speedup. If it
    exceeds 80 minutes, enumerate two-way splits (prog_peek in the -p kernel)
    and minimize the longer wall. Include reference setup in rho's kernel.
    """
    estimates = report['estimated_policy_minutes']
    reference = report['reference_seconds']/60
    total = sum(estimates.values()) + reference
    report['estimated_unsplit_kernel_minutes'] = total
    report['split'] = total > 80
    if not report['split']:
        groups = [list(POLICIES)]
    else:
        options = []
        others = POLICIES[:-1]
        for mask in range(1 << len(others)):
            progress = [p for i,p in enumerate(others) if mask & (1 << i)] + ['prog_peek']
            primary = [p for p in others if p not in progress]
            if not primary:
                continue
            walls = [sum(estimates[p] for p in group) + (reference if 'rho' in group else 0.)
                     for group in (primary,progress)]
            options.append((max(walls),mask,primary,progress))
        _,_,primary,progress = min(options)
        groups = [primary,progress]
    report['policy_groups'] = groups
    report['estimated_kernel_minutes'] = [sum(estimates[p] for p in group) + (reference if 'rho' in group else 0.)
                                         for group in groups]
    assert max(report['estimated_kernel_minutes']) <= 80, 'split kernel projection exceeds 80 minutes'


def smoke(data, out):
    configure()
    report = dict(ok=False, threads=torch.get_num_threads())
    def save():
        dump(Path(out)/'smoke.json', report)
    try:
        raw, pool, labels, test, digest = load(6, data)
        report['sha256_e10m'] = digest
        report['pool_kinds'] = dict(Counter(map(str, labels)))
        # A completed later update cannot affect any earlier chunk's score.
        a = Learner(6).read(pool[0, :1024], keep=True)
        b = Learner(6).read(pool[0], keep=True)
        gap = float(np.max(np.abs(a-b[:1024])))
        assert np.array_equal(a,b[:1024])
        report['prefix'] = dict(bytes=1024, full_bytes=4096, max_abs_bits=gap)
        # Future bytes within a chunk cannot affect earlier recurrent outputs.
        changed = pool[0, :128].copy(); changed[64:] ^= 255
        a = Learner(6).read(pool[0, :128], keep=True)
        b = Learner(6).read(changed, keep=True)
        assert np.array_equal(a[:64], b[:64])
        report['prefix']['future_mutation_max_abs'] = float(np.max(np.abs(a[:64]-b[:64])))
        save(); print('PREFIX', report['prefix'], flush=True)
        live = Learner(6); live.read(pool[0])  # populate Adam state
        before = live.checksum()
        score = progress_score(live, pool[1], [pool[0,:256]])
        after = live.checksum()
        assert before == after and math.isfinite(score)
        trial = copy.deepcopy(live)
        before_step = [float(s['step']) for s in trial.opt.state.values()]
        trial.read(pool[1,:256], step_bytes=256)
        after_step = [float(s['step']) for s in trial.opt.state.values()]
        assert all(b == a+1 for a,b in zip(before_step,after_step))
        assert all(trial.opt.state[p]['exp_avg'].data_ptr() != live.opt.state[q]['exp_avg'].data_ptr()
                   for p,q in zip(trial.model.parameters(),live.model.parameters()))
        report['trial'] = dict(before=before, after=after, identical=True, score=score,
                               adam_step_increment=1, independent_moments=True)
        save(); print('TRIAL', report['trial'], flush=True)
        clock = time.perf_counter(); ref = make_reference(6,raw)
        report['reference_seconds'] = time.perf_counter()-clock
        report['B8'] = {}
        for policy in POLICIES:
            row = run_policy(6,policy,8,pool,labels,test,ref,checkpoint=8)
            assert row['budget']==8 and math.isfinite(row['T_clean']) and sum(row['kinds_read'].values())==8
            report['B8'][policy] = row
            print('B8', policy, json.dumps(row['kinds_read']), 'seconds', row['wall_seconds'], flush=True)
            save()
        row = run_policy(6,'prog_peek',25,pool,labels,test,ref)
        report['timing_B25'] = row
        # Separate checkpoint cost; production has eight full clean tests.
        clock = time.perf_counter(); Learner(6).test(test)
        report['frozen_test_seconds'] = time.perf_counter()-clock
        report['estimated_policy_minutes'] = {
            policy: (report['B8'][policy]['wall_seconds']-report['frozen_test_seconds'])*25/60
                    +8*report['frozen_test_seconds']/60 for policy in POLICIES}
        report['estimated_policy_minutes']['prog_peek'] = row['wall_seconds']*8/60
        plan_kernels(report)
        report['ok'] = True
        save(); print('TIMING', json.dumps({k:v for k,v in report.items() if k.startswith('estimated') or k in ('split','policy_groups')}), flush=True)
    except Exception as exc:
        report['failure'] = str(exc); save(); raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',type=int,choices=SEEDS,default=6)
    p.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES))
    p.add_argument('--data',default='data/enwik8/enwik8')
    p.add_argument('--out',default='results/h35b')
    p.add_argument('--smoke',action='store_true')
    args = p.parse_args()
    if args.smoke:
        smoke(args.data,args.out)
    else:
        kernel_main(args.seed,args.out,f'ezzzzzx/h35b-s{args.seed}',args.policies,args.data)

if __name__ == '__main__':
    main()
