"""H24f measurements: principles 1 (history store) and 2 (compression).
Protocol: handoffs/2026-11-10_codex_h24f_fine_store.md. No verdicts.
"""
import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from h24e_priced_store import Cortex, Store as ChunkStore, A, CHUNK, detach, now, mean_std, fmt
from h24b_world import F_ALPHA, F_PROCS, BUCKETS, stream_f, value_exposures

CONDITIONS = ('gru', 'full', 'full5', 'fine', 'fine3')


class Match5:
    def likelihood(self, context, symbol, t):
        if len(context) < 5:
            return 1/A
        pos = self.last.get(bytes(context[-5:]))
        if pos is None:
            return 1/A
        lo, payload = self.locate(pos)
        offset = pos-lo
        assert pos < t and offset >= 5
        length = 5
        while (length < 32 and length < offset and length < len(context)
               and context[-length-1] == payload[offset-length-1]):
            length += 1
        bucket = 0 if length < 8 else 1 if length < 12 else 2 if length < 16 else 3 if length < 24 else 4
        confidence = (self.hits[bucket]+1)/(self.trials[bucket]+2)
        correct = int(payload[offset]) == int(symbol)
        probability = confidence if correct else (1-confidence)/(A-1)
        self.trials[bucket] += 1
        self.hits[bucket] += int(correct)
        return probability


class Full5(Match5, ChunkStore):
    def __init__(self):
        super().__init__()
        self.hits = np.zeros(5, dtype=np.int64)
        self.trials = np.zeros(5, dtype=np.int64)

    def write(self, start, symbols):
        payload = np.array(symbols, dtype=np.uint8, copy=True)
        self.chunks[start] = payload
        self.starts.append(start)
        for offset in range(5, CHUNK):
            self.last[bytes(payload[offset-5:offset])] = start+offset

    def locate(self, pos):
        lo = pos//CHUNK*CHUNK
        return lo, self.chunks[lo]

    def memory(self):
        return len(self.starts)*(CHUNK*math.log2(A)+32)

    def count(self):
        return len(self.starts)*CHUNK

    def batch(self, rng):
        targets, previous, missing = self.replay(rng)
        inputs = torch.cat((previous[None, :], targets[:-1]), dim=0)
        mask = torch.zeros((CHUNK, 8), dtype=torch.bool)
        mask[0] = missing
        return targets, inputs, mask


class Fine(Match5):
    def __init__(self):
        self.starts, self.windows = [], []
        self.last = {}
        self.hits = np.zeros(5, dtype=np.int64)
        self.trials = np.zeros(5, dtype=np.int64)
        self.stored = 0
        self.peak = 0.0

    def locate(self, pos):
        j = bisect_right(self.starts, pos)-1
        lo = self.starts[j]
        assert j >= 0 and pos < lo+len(self.windows[j])
        return lo, self.windows[j]

    def write(self, lo, payload, join_touching=False):
        # Overlapping windows merge. Touching windows stay separate: a shared
        # endpoint has an empty intersection. join_touching is only for the
        # unscored right tail of one window, written after the next chunk is
        # scored; that tail meets its own head at the chunk boundary.
        hi = lo+len(payload)
        j = bisect_right(self.starts, lo)-1
        if j >= 0:
            end_j = self.starts[j]+len(self.windows[j])
            touches = join_touching and end_j == lo
            if end_j < lo or (end_j == lo and not touches):
                j += 1
        else:
            j += 1
        if join_touching:
            assert j < len(self.starts) and self.starts[j] <= lo
            assert self.starts[j]+len(self.windows[j]) >= lo
        k = j
        merged_lo, merged_hi = lo, hi
        while k < len(self.starts) and self.starts[k] < merged_hi:
            merged_lo = min(merged_lo, self.starts[k])
            merged_hi = max(merged_hi, self.starts[k]+len(self.windows[k]))
            k += 1
        merged = np.empty(merged_hi-merged_lo, dtype=np.uint8)
        removed = 0
        for s, w in zip(self.starts[j:k], self.windows[j:k]):
            merged[s-merged_lo:s-merged_lo+len(w)] = w
            removed += len(w)
        merged[lo-merged_lo:hi-merged_lo] = payload
        self.starts[j:k], self.windows[j:k] = [merged_lo], [merged]
        self.stored += len(merged)-removed
        self.peak = max(self.peak, self.memory())
        # Only contexts crossing the new material, or new targets, need indexing.
        # max preserves the latest target when old windows are joined.
        for pos in range(max(merged_lo+5, lo), min(merged_hi, hi+5)):
            offset = pos-merged_lo
            key = bytes(merged[offset-5:offset])
            self.last[key] = max(pos, self.last.get(key, -1))

    def memory(self):
        return self.stored*math.log2(A)+32*len(self.starts)

    def count(self):
        return self.stored

    def batch(self, rng):
        targets = np.empty((CHUNK, 8), dtype=np.int64)
        inputs = np.zeros_like(targets)
        missing = np.zeros((CHUNK, 8), dtype=bool)
        for col in range(8):
            filled = 0
            while filled < CHUNK:
                w = self.windows[int(rng.integers(len(self.windows)))]
                offset = int(rng.integers(len(w)-CHUNK+1)) if len(w) >= CHUNK else 0
                take = min(CHUNK-filled, len(w)-offset)
                targets[filled:filled+take, col] = w[offset:offset+take]
                if offset:
                    inputs[filled, col] = w[offset-1]
                else:
                    missing[filled, col] = True
                inputs[filled+1:filled+take, col] = w[offset:offset+take-1]
                filled += take
        return tuple(torch.from_numpy(a) for a in (targets, inputs, missing))


def run(condition, seed, n, out, deadline=None):
    clock = time.monotonic()
    text, labels = stream_f('slow', seed, n)
    procs = np.asarray(labels)
    symbols = np.array([F_ALPHA.index(c) for c in text], dtype=np.uint8)
    x = torch.from_numpy(symbols.astype(np.int64))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Cortex()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    store = Full5() if condition == 'full5' else Fine()
    weights = np.full(2, .5)
    bits, cortex_bits = np.empty(n), np.empty(n)
    marked = np.zeros(n, dtype=bool)
    memory_trace, count_trace, window_trace = [], [], []
    context, state, replay_steps, pending = bytearray(), None, 0, 0

    def step(loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for start in range(0, n, CHUNK):
        if deadline is not None and time.monotonic() >= deadline:
            np.savez_compressed(out/f'{condition}_{seed}_partial.npz',
                bits=bits[:start], cortex_bits=cortex_bits[:start], marked=marked[:start],
                memory_trace=np.array(memory_trace), count_trace=np.array(count_trace),
                window_trace=np.array(window_trace))
            raise TimeoutError(f'{condition} seed {seed}: {start}/{n} symbols completed at wall deadline')
        end = start+CHUNK
        optimizer.zero_grad(set_to_none=True)
        emb = (torch.cat((model.embedding.weight.new_zeros(1, 32), model.embedding(x[:end-1])))
               if start == 0 else model.embedding(x[start-1:end-1]))
        logits, state = model(emb, state)
        losses = F.cross_entropy(logits, x[start:end], reduction='none')
        gp = logits.detach().double().softmax(-1).numpy()
        cortex_bits[start:end] = losses.detach().double().numpy()/math.log(2)
        for offset, symbol in enumerate(symbols[start:end]):
            t = start+offset
            likelihoods = np.array([gp[offset, symbol], store.likelihood(context, symbol, t)])
            likelihood = float(weights @ likelihoods)
            bits[t] = -math.log2(likelihood)
            weights *= likelihoods/likelihood
            weights = .999*weights+.001/2
            context.append(int(symbol))
            if len(context) > 32:
                del context[0]
        state = detach(state)
        # Admit only after the entire chunk has been scored. Inclusive [i-16,i+4].
        if condition == 'full5':
            store.write(start, symbols[start:end])
        else:
            if pending > start:
                store.write(start, symbols[start:min(end, pending)], join_touching=True)
            threshold = 4.0 if condition == 'fine' else 3.0
            indices = np.flatnonzero(bits[start:end] > threshold)+start
            marked[indices] = True
            intervals = []
            for i in indices:
                lo, hi = max(0, int(i)-16), min(n, int(i)+5)
                pending = max(pending, hi)
                hi = min(end, hi)
                if intervals and lo < intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(hi, intervals[-1][1]))
                else:
                    intervals.append((lo, hi))
            for lo, hi in intervals:
                store.write(lo, symbols[lo:hi])
        memory_trace.append(store.memory())
        count_trace.append(store.count())
        window_trace.append(len(store.starts))
        step(losses.mean())
        if store.starts:
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                targets, inputs, missing = store.batch(rng)
                emb = model.embedding(inputs).masked_fill(missing[:, :, None], 0)
                outputs, _ = model.gru(emb, None)
                rlogits = model.readout(outputs)
                step(F.cross_entropy(rlogits.reshape(-1, A), targets.reshape(-1)))
                replay_steps += 1
        if (start//CHUNK+1) % 32 == 0 or end == n:
            print(f'{now()} {condition} seed={seed} n={end}/{n} P={bits[:end].mean():.6f} stored={store.count()} seconds={time.monotonic()-clock:.1f}', flush=True)
    assert np.isfinite(bits).all() and replay_steps == 2*n//CHUNK
    mask = np.zeros(n, dtype=bool)
    for lo in store.starts:
        s, w = store.locate(lo)
        mask[s:s+len(w)] = True
    assert int(mask.sum()) == store.count()
    if condition != 'full5':
        expected = np.zeros(n, dtype=bool)
        for i in np.flatnonzero(marked):
            expected[max(0, i-16):min(n, i+5)] = True
        assert np.array_equal(mask, expected)
    exposures = value_exposures(text, procs, bits)
    for stats in exposures.values():
        stats.pop('recall_bits')
        stats['bits_per_digit'] = stats['bits']/stats['digits'] if stats['digits'] else None
    repeated = [exposures[b] for b in BUCKETS[1:]]
    digits = sum(s['digits'] for s in repeated)
    result = dict(condition=condition, seed=seed, n=n, P=float(bits.mean()),
        peak_M_bits=max(max(memory_trace), getattr(store, 'peak', 0)),
        peak_M_over_N=max(max(memory_trace), getattr(store, 'peak', 0))/n,
        stored_symbols=store.count(), stored_windows=len(store.starts), fraction_stored=store.count()/n,
        per_process_P={p: float(bits[procs == p].mean()) if np.any(procs == p) else None for p in F_PROCS},
        process_counts={p: int(np.sum(procs == p)) for p in F_PROCS}, F_value_exposures=exposures,
        F_value_at_least_1_prior_bits_per_digit=sum(s['bits'] for s in repeated)/digits if digits else None,
        replay_steps=replay_steps, parameters=sum(p.numel() for p in model.parameters()), wall_seconds=time.monotonic()-clock)
    np.savez_compressed(out/f'{condition}_{seed}.npz', symbols=symbols, processes=procs,
        bits=bits, cortex_bits=cortex_bits, marked=marked, stored=mask,
        window_starts=np.array(store.starts), window_lengths=np.array([len(store.locate(s)[1]) for s in store.starts]),
        memory_trace=np.array(memory_trace), count_trace=np.array(count_trace), window_trace=np.array(window_trace))
    return result


def baseline_rows(n, seeds):
    source = Path('results/h24e')/('smoke' if n == 8192 else '')/'results.json'
    rows = []
    for r in json.loads(source.read_text())['runs']:
        if r['condition'] not in ('gru', 'full') or r['seed'] not in seeds:
            continue
        assert r['n'] == n
        row = {k: v for k, v in r.items() if k not in ('T', 'peak_S_bits', 'peak_S_over_N', 'fraction_written')}
        row.update(peak_M_bits=r['peak_S_bits'], peak_M_over_N=r['peak_S_over_N'],
            fraction_stored=r['fraction_written'], stored_symbols=r['chunks_written']*CHUNK,
            stored_windows=r['chunks_written'], reused_from=str(source))
        rows.append(row)
    assert len(rows) == 2*len(seeds)
    return rows


def save(data, out):
    order = {c: i for i, c in enumerate(CONDITIONS)}
    data['runs'] = sorted(data['runs'], key=lambda r: (order.get(r['condition'], 99), r['seed']))
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    rows = data['runs']
    lines = ['# H24f measurements', '', f"Started: {data['started']}. Updated: {now()}.", '',
        f"Slow world; N = {data['n']}; seeds = {data['seeds']}; alphabet = 42; CPU threads = 2.", '',
        '| Condition | Seed | P | Peak M/N | Fraction stored | Stored symbols | Windows | Wall seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['P']:.6f} | {r['peak_M_over_N']:.6f} | {r['fraction_stored']:.6f} | {r['stored_symbols']} | {r['stored_windows']} | {r['wall_seconds']:.3f} |")
    lines += ['', '| Condition | Seeds | P mean ± population std | Peak M/N mean ± std | Fraction stored mean ± std |', '|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        rs = [r for r in rows if r['condition'] == c]
        lines.append(f'| {c} | {len(rs)} | '+' | '.join(mean_std([r[k] for r in rs]) for k in ('P', 'peak_M_over_N', 'fraction_stored'))+' |')
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
    lines += ['', '| Condition | F ≥1 prior mean over observed seeds | F ≥1 prior pooled bits/digit |', '|---|---:|---:|']
    for c in CONDITIONS:
        rs = [r for r in rows if r['condition'] == c and r['F_value_at_least_1_prior_bits_per_digit'] is not None]
        stats = [r['F_value_exposures'][b] for r in rs for b in BUCKETS[1:]]
        digits = sum(s['digits'] for s in stats)
        lines.append(f"| {c} | {fmt(float(np.mean([r['F_value_at_least_1_prior_bits_per_digit'] for r in rs])) if rs else None)} | {fmt(sum(s['bits'] for s in stats)/digits if digits else None)} |")
    lines += ['', 'Pareto: P against peak M/N. gru has M = 0. full is H24e, reused.', '',
        '| Condition | Seed | P | Peak M/N |', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['P']:.6f} | {r['peak_M_over_N']:.6f} |")
    lines += ['', '| Condition | Seeds | P mean ± population std | Peak M/N mean ± population std |', '|---|---:|---:|---:|']
    for c in CONDITIONS:
        rs = [r for r in rows if r['condition'] == c]
        lines.append(f'| {c} | {len(rs)} | '+' | '.join(mean_std([r[k] for r in rs]) for k in ('P', 'peak_M_over_N'))+' |')
    lines += ['', 'P: prequential observation bits/symbol. M = stored symbols × log2(42) + 32 bits/window; peak tracked after each merged-window write; denominator is final N.',
        'gru and full measurements reused from H24e; their wall seconds are historical. No reruns. Full5 retains H24e aligned 64-symbol windows and replay sampling.',
        'Match order 5, latest target; backwards extension up to 32 within stored windows. Confidence buckets: 5–7, 8–11, 12–15, 16–23, 24–32. Fixed share 0.001, initial weights (0.5, 0.5).',
        'Fine thresholds: strictly >4.0 and >3.0 final-mixture bits. Inclusive windows [i−16,i+4], clipped to life; overlapping windows merged, touching windows kept separate. A right tail past the chunk is admitted after that later chunk is scored and joined to its own head.',
        'Fine replay: uniformly sample stored windows; uniform 64-symbol segment in long windows, concatenate sampled short windows to 64. Zero embedding at each sampled window boundary; zero initial replay hidden state.',
        'Live GRU and replay schedule: H24e Cortex; two batch-8 Adam replay steps after each scored chunk, live update then replay. Scoring before any current-chunk admission or weight update.',
        'M follows the specified payload-plus-window formula; neural parameters, optimizer, match index and transient live context are outside that formula. F exposure counts use all observations; NA denotes absent observations.',
        f"New-run wall seconds (sum of full5, fine and fine3 run times; gru and full excluded): {data.get('total_wall_seconds', 0):.3f}.", '']
    if 'continuation_wall_seconds' in data:
        lines.append(f"Continuation wall seconds: {data['continuation_wall_seconds']:.3f}.")
    lines += [f'Runtime note: {note}' for note in data['notes']]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def checks():
    payload = np.arange(128, dtype=np.uint8) % A
    s = Full5()
    s.write(0, payload[:64])
    assert s.likelihood(bytearray(payload[:5]), payload[5], 64) == .5
    fine = Fine()
    fine.write(0, payload[:10])
    fine.write(20, payload[20:30])
    fine.write(8, payload[8:25])
    assert fine.starts == [0] and np.array_equal(fine.windows[0], payload[:30])
    assert fine.count() == 30
    for key, pos in fine.last.items():
        lo, w = fine.locate(pos)
        assert bytes(w[pos-lo-5:pos-lo]) == key
    tiny = Fine()
    tiny.write(3, payload[3:10])
    targets, inputs, missing = tiny.batch(np.random.default_rng(0))
    assert targets.shape == (64, 8) and missing.sum() == 80
    assert np.array_equal(targets[:7, 0], payload[3:10])
    assert np.all(inputs.numpy()[missing.numpy()] == 0)
    # No context may jump an unstored gap.
    gap = Fine()
    gap.write(0, payload[:8])
    gap.write(12, payload[12:20])
    assert 12 not in gap.last.values() and 16 not in gap.last.values()
    touch = Fine()
    touch.write(0, payload[:10])
    touch.write(10, payload[10:20])
    assert touch.starts == [0, 10] and touch.count() == 20
    assert touch.last[bytes(payload[10:15])] == 15 and 10 not in touch.last.values()
    joined = Fine()
    joined.write(0, payload[:10])
    joined.write(10, payload[10:14], join_touching=True)
    assert joined.starts == [0] and np.array_equal(joined.windows[0], payload[:14]) and joined.count() == 14


def measured_wall(runs):
    return sum(r['wall_seconds'] for r in runs if r['condition'] in ('full5', 'fine', 'fine3'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--conditions', nargs='*')
    parser.add_argument('--seeds', nargs='*', type=int)
    parser.add_argument('--budget-seconds', type=float)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    clock = time.monotonic()
    checks()
    out = Path('results/h24f')/('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    n, seeds = (8192, [6]) if args.smoke else (131072, [6, 7, 8])
    if args.resume:
        assert not args.smoke
        data = json.loads((out/'results.json').read_text())
        data['notes'] = [note for note in data['notes'] if not note.startswith('fine3 dropped') and not note.startswith('Continuation')]
        data['notes'].append('Continuation: fine recomputed with touching windows kept separate. A fine row without merge=overlap is still the touching-merge result. fine3 runs after fine if this process budget allows.')
        conditions = args.conditions or ['fine', 'fine3']
        seeds = args.seeds or seeds
        deadline = None if args.budget_seconds is None else clock+args.budget_seconds
    else:
        data = dict(started=now(), n=n, seeds=seeds, principles=['1: history store', '2: compression'],
            notes=[], runs=baseline_rows(n, seeds), torch_version=torch.__version__, checks_passed=True)
        conditions = ['full5', 'fine', 'fine3']
        deadline = None
        if not args.smoke:
            smoke = json.loads((out/'smoke/results.json').read_text())
            data['smoke_wall_seconds'] = smoke['total_wall_seconds']
            # Include smoke and a 60-second implementation/startup allowance in the budget.
            deadline = clock+max(0, 6000-smoke['total_wall_seconds']-60)
            data['production_budget_seconds'] = max(0, 6000-smoke['total_wall_seconds']-60)
            estimates = {r['condition']: r['wall_seconds']*16 for r in smoke['runs'] if r['condition'] in conditions}
            data['smoke_projected_wall_seconds'] = 3*sum(estimates.values())
            if data['smoke_projected_wall_seconds'] > 95*60:
                conditions.remove('fine3')
                data['notes'].append('fine3 dropped first: smoke projection exceeds 95-minute planning allowance within 100-minute budget.')
    save(data, out)
    for condition in conditions:
        if condition == 'fine3' and not args.smoke and not args.resume:
            observed = [r['wall_seconds'] for r in data['runs'] if r['condition'] == 'fine']
            if time.monotonic()-clock+3*max(estimates['fine3'], float(np.mean(observed))) > 99*60:
                data['notes'].append('fine3 dropped first: measured runtime projects completion beyond 99-minute planning allowance.')
                break
        for seed in seeds:
            if args.resume and deadline is not None and condition == 'fine3':
                elapsed = time.monotonic()-clock
                observed = [r['wall_seconds'] for r in data['runs'] if r['condition'] == 'fine' and r.get('merge') == 'overlap']
                typical = float(np.mean(observed)) if observed else 800.0
                if elapsed+typical > args.budget_seconds-60:
                    pending = seeds[seeds.index(seed):]
                    data['notes'].append(
                        f"fine3 seeds {pending} not started: remaining budget {args.budget_seconds-elapsed:.0f}s under one observed fine run plus 60s.")
                    close(data, out, clock, args)
                    return
            try:
                # Resume keeps a started run; the budget gate above is what drops fine3.
                result = run(condition, seed, n, out, None if args.resume else deadline)
            except TimeoutError as error:
                data['notes'].append(str(error))
                close(data, out, clock, args)
                return
            if args.resume:
                result['merge'] = 'overlap'
                data['runs'] = [r for r in data['runs'] if not (r['condition'] == condition and r['seed'] == seed)]
                data['runs'].append(result)
                print(f"{now()} RUN_DONE {condition} seed={seed} P={result['P']:.6f} peak_M/N={result['peak_M_over_N']:.6f} fraction={result['fraction_stored']:.6f} seconds={result['wall_seconds']:.1f}", flush=True)
            else:
                data['runs'].append(result)
            close(data, out, clock, args, done=False)
    close(data, out, clock, args, done=True)


def close(data, out, clock, args, done=True):
    if args.resume:
        data['total_wall_seconds'] = measured_wall(data['runs'])
        data['continuation_wall_seconds'] = time.monotonic()-clock
    else:
        data['total_wall_seconds'] = time.monotonic()-clock
    if done:
        data['finished'] = now()
        fine_rows = [r for r in data['runs'] if r['condition'] == 'fine']
        if args.resume and fine_rows and all(r.get('merge') == 'overlap' for r in fine_rows):
            data['notes'] = [note for note in data['notes'] if 'without merge' not in note]
            if not any(note.startswith('fine rows were recomputed') for note in data['notes']):
                data['notes'].append('fine rows were recomputed with touching windows kept separate.')
        save(data, out)
        print(f"{now()} ALL_DONE continuation_seconds={data.get('continuation_wall_seconds', data['total_wall_seconds']):.1f}", flush=True)
    else:
        save(data, out)


if __name__ == '__main__':
    main()
