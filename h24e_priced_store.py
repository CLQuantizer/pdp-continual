"""H24e priced store: principles 1 (history) and 2 (compression).

Protocol: handoffs/2026-11-10_codex_h24e_priced_store.md. No verdicts.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from h21c_replay_dose import GRU, CHUNK, detach, now, mean_std
from h24b_world import F_ALPHA, F_PROCS, BUCKETS, stream_f, value_exposures

CONDITIONS = ('gru', 'full', 'surp', 'surp90')
A = len(F_ALPHA)
CHUNK_BITS = CHUNK * math.log2(A) + 32


class Cortex(GRU):
    def __init__(self):
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(A, 32)
        self.gru = nn.GRU(32, 256)
        self.readout = nn.Linear(256, A)


class Store:
    def __init__(self):
        self.chunks = {}
        self.starts = []
        self.last = {}
        self.hits = np.zeros(4, dtype=np.int64)
        self.trials = np.zeros(4, dtype=np.int64)

    def write(self, start, symbols):
        assert start % CHUNK == 0 and len(symbols) == CHUNK
        payload = np.array(symbols, dtype=np.uint8, copy=True)
        self.chunks[start] = payload
        self.starts.append(start)
        # Index context AND target only when wholly inside this stored chunk.
        for offset in range(8, CHUNK):
            self.last[bytes(payload[offset-8:offset])] = (start, offset)

    def likelihood(self, context, symbol, t):
        if len(context) < 8:
            return 1 / A
        found = self.last.get(bytes(context[-8:]))
        if found is None:
            return 1 / A
        start, offset = found
        assert start + CHUNK <= t and start in self.chunks
        payload = self.chunks[start]
        length = 8
        while (length < 32 and length < offset and length < len(context)
               and context[-length-1] == payload[offset-length-1]):
            length += 1
        bucket = 0 if length < 12 else 1 if length < 16 else 2 if length < 24 else 3
        confidence = (self.hits[bucket] + 1) / (self.trials[bucket] + 2)
        correct = int(payload[offset]) == int(symbol)
        probability = confidence if correct else (1-confidence) / (A-1)
        # The caller has already fixed its prediction; update after likelihood.
        self.trials[bucket] += 1
        self.hits[bucket] += int(correct)
        return probability

    def replay(self, rng):
        starts = [self.starts[int(i)] for i in rng.integers(len(self.starts), size=8)]
        targets = np.stack([self.chunks[s] for s in starts], axis=1).astype(np.int64)
        previous = np.zeros(8, dtype=np.int64)
        missing = np.ones(8, dtype=bool)
        for i, start in enumerate(starts):
            if start-CHUNK in self.chunks:
                previous[i] = self.chunks[start-CHUNK][-1]
                missing[i] = False
        return torch.from_numpy(targets), torch.from_numpy(previous), torch.from_numpy(missing)


def gate(condition, index, score, history):
    if condition == 'gru':
        return False
    if condition == 'full' or index < 8:
        return True
    q = .7 if condition == 'surp' else .9
    return score > float(np.quantile(history, q, method='linear'))


def run(condition, seed, n, out):
    clock = time.monotonic()
    text, labels = stream_f('slow', seed, n)
    procs = np.asarray(labels)
    symbols = np.array([F_ALPHA.index(c) for c in text], dtype=np.uint8)
    x = torch.from_numpy(symbols.astype(np.int64))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Cortex()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    store = Store()
    weights = np.full(2, .5)
    bits = np.empty(n)
    cortex_bits = np.empty(n)
    writes = np.zeros(n//CHUNK, dtype=bool)
    scores, thresholds = [], []
    context = bytearray()
    state = None
    replay_steps = 0

    def step(loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for start in range(0, n, CHUNK):
        end = start + CHUNK
        optimizer.zero_grad(set_to_none=True)
        if start == 0:
            emb = torch.cat((model.embedding.weight.new_zeros(1, 32), model.embedding(x[:end-1])))
        else:
            emb = model.embedding(x[start-1:end-1])
        logits, state = model(emb, state)
        losses = F.cross_entropy(logits, x[start:end], reduction='none')
        gp = logits.detach().double().softmax(-1).numpy()
        cortex_bits[start:end] = losses.detach().double().numpy()/math.log(2)
        if condition == 'gru':
            bits[start:end] = cortex_bits[start:end]
        else:
            for offset, symbol in enumerate(symbols[start:end]):
                t = start + offset
                mp = store.likelihood(context, symbol, t)
                likelihoods = np.array([gp[offset, symbol], mp])
                likelihood = float(weights @ likelihoods)
                bits[t] = -math.log2(likelihood)
                weights *= likelihoods / likelihood
                weights = .999 * weights + .001/2
                context.append(int(symbol))
                if len(context) > 32:
                    del context[0]
        state = detach(state)
        score = float(bits[start:end].mean())
        index = start//CHUNK
        thresholds.append(None if index < 8 or condition in ('gru', 'full') else
                          float(np.quantile(scores, .7 if condition == 'surp' else .9, method='linear')))
        write = gate(condition, index, score, scores)
        scores.append(score)
        # Store admission follows ALL current predictions; no current-chunk recall.
        if write:
            store.write(start, symbols[start:end])
            writes[index] = True
        step(losses.mean())
        if condition != 'gru':
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                targets, previous, missing = store.replay(rng)
                inputs = torch.cat((previous[None, :], targets[:-1]), dim=0)
                emb = model.embedding(inputs)
                # No read of an unstored preceding symbol; zero at such boundaries.
                emb = emb.masked_fill((torch.arange(CHUNK)[:, None] == 0)[:, :, None]
                                      & missing[None, :, None], 0)
                outputs, _ = model.gru(emb, None)
                rlogits = model.readout(outputs)
                step(F.cross_entropy(rlogits.reshape(-1, A), targets.reshape(-1)))
                replay_steps += 1
        if (index+1) % 128 == 0 or end == n:
            print(f'{now()} {condition} seed={seed} n={end}/{n} P={bits[:end].mean():.6f} stored={len(store.starts)} seconds={time.monotonic()-clock:.1f}', flush=True)
    assert np.isfinite(bits).all() and np.isfinite(cortex_bits).all()
    assert len(store.starts) == int(writes.sum())
    assert condition != 'full' or writes.all()
    assert condition != 'gru' or not writes.any()
    if condition != 'gru':
        assert writes[:8].all() and replay_steps == 2*n//CHUNK
    exposures = value_exposures(text, procs, bits)
    for stats in exposures.values():
        stats.pop('recall_bits')
        stats['bits_per_digit'] = stats['bits']/stats['digits'] if stats['digits'] else None
    repeated = [exposures[b] for b in BUCKETS[1:]]
    digits = sum(s['digits'] for s in repeated)
    peak = len(store.starts)*CHUNK_BITS
    result = dict(condition=condition, seed=seed, n=n, P=float(bits.mean()),
        peak_S_bits=peak, peak_S_over_N=peak/n, T=float(bits.mean())+peak/n,
        chunks_written=int(writes.sum()), total_chunks=len(writes), fraction_written=float(writes.mean()),
        per_process_P={p: float(bits[procs == p].mean()) if np.any(procs == p) else None for p in F_PROCS},
        process_counts={p: int(np.sum(procs == p)) for p in F_PROCS},
        F_value_exposures=exposures,
        F_value_at_least_1_prior_bits_per_digit=sum(s['bits'] for s in repeated)/digits if digits else None,
        replay_steps=replay_steps, parameters=sum(p.numel() for p in model.parameters()),
        wall_seconds=time.monotonic()-clock)
    np.savez_compressed(out/f'{condition}_{seed}.npz', symbols=symbols, processes=procs,
                        bits=bits, cortex_bits=cortex_bits, writes=writes,
                        chunk_scores=np.array(scores), thresholds=np.array([np.nan if t is None else t for t in thresholds]))
    return result


def fmt(value):
    return 'NA' if value is None else f'{value:.6f}'


def save(data, out):
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    rows = data['runs']
    lines = ['# H24e measurements', '', f"Started: {data['started']}. Updated: {now()}.", '',
        f"Slow world; N = {data['n']}; seeds = {data['seeds']}; alphabet = 42; CPU threads = 2.", '',
        '| Condition | Seed | P | Peak S bits | Peak S/N | T | Chunks written | Fraction written | Wall seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['P']:.6f} | {r['peak_S_bits']:.3f} | {r['peak_S_over_N']:.6f} | {r['T']:.6f} | {r['chunks_written']} | {r['fraction_written']:.6f} | {r['wall_seconds']:.3f} |")
    lines += ['', '| Condition | P mean ± population std | Peak S/N mean ± std | T mean ± std | Fraction written mean ± std |', '|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        rs = [r for r in rows if r['condition'] == c]
        lines.append('| '+c+' | '+' | '.join(mean_std([r[k] for r in rs]) for k in ('P', 'peak_S_over_N', 'T', 'fraction_written'))+' |')
    lines += ['', '| Condition | Seed | '+' | '.join(F_PROCS)+' |', '|---|---:|'+'---:|'*len(F_PROCS)]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | "+' | '.join(fmt(r['per_process_P'][p]) for p in F_PROCS)+' |')
    lines += ['', '| Condition | Seed | Prior key appearances | Value digits | Bits | Bits/digit |', '|---|---:|---|---:|---:|---:|']
    for r in rows:
        for b, s in r['F_value_exposures'].items():
            lines.append(f"| {r['condition']} | {r['seed']} | {b} | {s['digits']} | {s['bits']:.6f} | {fmt(s['bits_per_digit'])} |")
    lines += ['', '| Condition | Seed | F value bits/digit, ≥1 prior |', '|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | {fmt(r['F_value_at_least_1_prior_bits_per_digit'])} |")
    lines += ['', 'P: prequential whole-stream observation bits/symbol. T = P + peak S/N. S = stored symbols × log2(42) + 32 bits per chunk.',
        'GRU: embedding 32, hidden 256, chunk 64, Adam lr 0.003, clip 1.0; carried and detached live state. All conditions rerun.',
        'Store admission: completed aligned 64-symbol chunks; system prequential chunk P strictly above the linear percentile of all preceding chunk scores; first 8 chunks written. No deletion.',
        'Replay: two Adam steps of eight uniformly sampled stored chunks with replacement per arrival chunk, after live update and write; zero hidden state. Shifted preceding-symbol input comes from the preceding chunk only when stored, otherwise zero embedding.',
        'Match: H21b preceding-8 context, latest stored target, extension up to 32 bounded by stored-chunk start; calibrated four length buckets. Index changes only on completed chunk writes. Confidence updates after each prediction. Fixed share 0.001, initial weights (0.5, 0.5).',
        'F exposure counts use all observed records across visits, independent of store admission. NA denotes no observations.',
        f"Total wall seconds: {data.get('total_wall_seconds', 0):.3f}.", '']
    lines += [f'Runtime note: {note}' for note in data['notes']]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def checks():
    store = Store()
    payload = np.arange(CHUNK, dtype=np.uint8) % A
    assert store.likelihood(bytearray(payload[:8]), payload[8], 64) == 1/A
    store.write(0, payload)
    assert store.likelihood(bytearray(payload[:8]), payload[8], 64) == .5
    assert all(s in store.chunks and o >= 8 for s, o in store.last.values())
    targets, previous, missing = store.replay(np.random.default_rng(0))
    assert targets.shape == (64, 8) and missing.all()
    store.write(64, payload)
    assert all(s in store.chunks for s, _ in store.last.values())
    assert not gate('surp', 8, 1, [1]*8)
    assert gate('surp', 8, 2, [1]*8)
    assert gate('surp90', 0, 0, []) and not gate('gru', 0, 9, [])
    toy = 'abcd=1234;'*8
    stats = value_exposures(toy, np.array(['F']*len(toy)), np.ones(len(toy)))
    assert [stats[b]['digits'] for b in BUCKETS] == [4, 4, 12, 12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    clock = time.monotonic()
    checks()
    out = Path('results/h24e') / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    n, seeds = (8192, [6]) if args.smoke else (131072, [6, 7, 8])
    data = dict(started=now(), n=n, seeds=seeds, principles=['1: history store', '2: compression'],
                notes=[], runs=[], torch_version=torch.__version__, checks_passed=True)
    conditions = list(CONDITIONS)
    if not args.smoke:
        smoke = json.loads((out/'smoke/results.json').read_text())
        estimates = {r['condition']: r['wall_seconds']*16 for r in smoke['runs']}
        projection = len(seeds)*sum(estimates.values())
        data['smoke_projected_wall_seconds'] = projection
        if projection > 95*60:
            conditions.remove('surp90')
            data['notes'].append('surp90 dropped first: smoke projection exceeds 95-minute planning allowance within 100-minute wall budget.')
    save(data, out)
    for condition in conditions:
        if condition == 'surp90' and not args.smoke:
            elapsed = time.monotonic()-clock
            observed = [r['wall_seconds'] for r in data['runs'] if r['condition'] == 'surp']
            if elapsed + max(estimates['surp90'], float(np.mean(observed)))*len(seeds) > 99*60:
                data['notes'].append('surp90 dropped first: measured runtime projects completion beyond 99-minute planning allowance.')
                break
        for seed in seeds:
            data['runs'].append(run(condition, seed, n, out))
            data['total_wall_seconds'] = time.monotonic()-clock
            save(data, out)
    data['finished'] = now()
    data['total_wall_seconds'] = time.monotonic()-clock
    save(data, out)


if __name__ == '__main__':
    main()
