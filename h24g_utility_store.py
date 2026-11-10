"""H24g measurements: utility store under a memory budget.

Principles: 1 (history store), 2 (compression), 3 (compression progress).
Protocol: handoffs/2026-11-10_grok_h24g_utility_store.md. No verdicts.

Writes follow H24f `fine` (mixture bits > 4, window [i-16, i+4], overlap merges,
touching windows stay separate). Each window carries a utility. When retained
memory exceeds B = 0.15 × full5 peak M, the lowest-utility unprotected windows
are evicted. Peak M is memory still retained after that eviction.
"""
import argparse
from bisect import bisect_right
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from h24b_world import F_ALPHA, F_PROCS, BUCKETS, stream_f, value_exposures
from h24e_priced_store import Cortex, A, CHUNK, detach, now, mean_std, fmt
from h24f_fine_store import Fine

CARRIED = ('gru', 'full', 'full5', 'fine', 'fine3')
NEW = ('util', 'util_m')
CONDITIONS = CARRIED + NEW
PROTECT = 32
DECAY = 0.999
THRESHOLD = 4.0
BUDGET_FRACTION = 0.15
WALL_BUDGET = 6000.0
# Drop replay credit when its measurement forwards cost more than this
# fraction of the wall time that would have been spent without them.
REPLAY_EXTRA_LIMIT = 0.30


def full5_peak_bits(n):
    assert n % CHUNK == 0
    return n * math.log2(A) + 32 * (n // CHUNK)


def budget_bits(n):
    return BUDGET_FRACTION * full5_peak_bits(n)


def surprise_intervals(indices, n, end, pending):
    """Same admission intervals as H24f fine. Touching intervals stay separate."""
    intervals = []
    for i in indices:
        lo, hi = max(0, int(i) - 16), min(n, int(i) + 5)
        pending = max(pending, hi)
        hi = min(end, hi)
        if intervals and lo < intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(hi, intervals[-1][1]))
        else:
            intervals.append((lo, hi))
    return intervals, pending


class UtilityStore(Fine):
    """Fine store plus per-window utility, protection, and whole-window eviction."""

    def __init__(self, budget, proc_ids):
        super().__init__()
        self.budget = float(budget)
        self.proc_ids = proc_ids
        self.utils = []
        self.born = []
        self.chunk = 0
        self.retained_peak = 0.0
        self.match_window = None
        self.index_dirty = False
        self.evicted = 0
        self.evicted_weight = np.zeros(len(F_PROCS), dtype=np.float64)
        self.match_credit_bits = 0.0
        self.replay_credit_bits = 0.0
        self.credit_seconds = 0.0
        self.max_over_budget = 0.0
        self.chunks_over_budget = 0
        self.peak = 0.0

    def _merge_span(self, lo, payload, join_touching):
        hi = lo + len(payload)
        j = bisect_right(self.starts, lo) - 1
        if j >= 0:
            end_j = self.starts[j] + len(self.windows[j])
            touches = join_touching and end_j == lo
            if end_j < lo or (end_j == lo and not touches):
                j += 1
        else:
            j += 1
        k = j
        merged_hi = hi
        while k < len(self.starts) and self.starts[k] < merged_hi:
            merged_hi = max(merged_hi, self.starts[k] + len(self.windows[k]))
            k += 1
        return j, k

    def write(self, lo, payload, join_touching=False, fresh=True):
        # fresh=False is only the deferred right tail of a window already
        # opened. It does not add the 1-bit prior and does not restart protection.
        j, k = self._merge_span(lo, payload, join_touching)
        n0 = len(self.starts)
        prior_u = float(sum(self.utils[j:k])) if k > j else 0.0
        prior_born = self.born[j:k]
        super().write(lo, payload, join_touching=join_touching)
        assert len(self.starts) == n0 - (k - j) + 1
        if fresh:
            utility, born = 1.0 + prior_u, self.chunk
        else:
            utility = prior_u
            born = max(prior_born) if prior_born else self.chunk
        self.utils[j:k] = [utility]
        self.born[j:k] = [born]
        assert len(self.utils) == len(self.born) == len(self.starts)
        self.evict_to_budget()
        self._note_retained()

    def _note_retained(self):
        retained = self.memory()
        if retained > self.retained_peak:
            self.retained_peak = retained
        # Fine.write records the pre-eviction spike. The reported peak is retained memory.
        self.peak = self.retained_peak

    def decay(self):
        utils = self.utils
        for i in range(len(utils)):
            utils[i] *= DECAY

    def credit_match(self, gru_bits, mix_bits):
        j = self.match_window
        if j is None:
            return
        gain = gru_bits - mix_bits
        if gain > 0.0:
            self.utils[j] += gain
            self.match_credit_bits += gain

    def credit_replay(self, gain, sources):
        if gain <= 0.0 or not sources:
            return
        share = gain / len(sources)
        utils = self.utils
        for s in sources:
            utils[s] += share
        self.replay_credit_bits += gain

    def likelihood(self, context, symbol, t):
        self.match_window = None
        if len(context) >= 5:
            pos = self.last.get(bytes(context[-5:]))
            if pos is not None:
                j = bisect_right(self.starts, pos) - 1
                if j >= 0:
                    lo = self.starts[j]
                    if lo <= pos < lo + len(self.windows[j]):
                        self.match_window = j
        return super().likelihood(context, symbol, t)

    def evict_to_budget(self):
        self.stored = sum(len(w) for w in self.windows)
        while self.memory() > self.budget:
            victim = None
            best = None
            chunk = self.chunk
            for i, born in enumerate(self.born):
                if chunk < born + PROTECT:
                    continue
                key = (self.utils[i], self.starts[i], i)
                if best is None or key < best:
                    best = key
                    victim = i
            if victim is None:
                break
            self._drop(victim)
        self.stored = sum(len(w) for w in self.windows)

    def _drop(self, i):
        lo = self.starts[i]
        length = len(self.windows[i])
        counts = np.bincount(self.proc_ids[lo:lo + length], minlength=len(F_PROCS))
        self.evicted += 1
        self.evicted_weight += counts / length
        del self.starts[i]
        del self.windows[i]
        del self.utils[i]
        del self.born[i]
        self.stored -= length
        self.index_dirty = True

    def reindex(self):
        last = {}
        for lo, w in zip(self.starts, self.windows):
            raw = w.tobytes()
            base = lo
            for offset in range(5, len(raw)):
                last[raw[offset - 5:offset]] = base + offset
        self.last = last
        self.index_dirty = False

    def end_chunk(self):
        if self.index_dirty:
            self.reindex()
        self.stored = sum(len(w) for w in self.windows)
        self._assert_consistent()
        over = self.memory() - self.budget
        if over > self.max_over_budget:
            self.max_over_budget = over
        if over > 1e-6:
            self.chunks_over_budget += 1
        self._note_retained()

    def _assert_consistent(self):
        assert len(self.utils) == len(self.born) == len(self.starts) == len(self.windows)
        assert self.stored == sum(len(w) for w in self.windows)
        prev = 0
        for lo, w in zip(self.starts, self.windows):
            assert lo >= prev and len(w) > 0
            prev = lo + len(w)
        if self.memory() > self.budget + 1e-6:
            assert all(self.chunk < born + PROTECT for born in self.born)

    def batch_sources(self, rng):
        targets = np.empty((CHUNK, 8), dtype=np.int64)
        inputs = np.zeros_like(targets)
        missing = np.zeros((CHUNK, 8), dtype=bool)
        sources = []
        nwin = len(self.windows)
        for col in range(8):
            filled = 0
            while filled < CHUNK:
                w_i = int(rng.integers(nwin))
                w = self.windows[w_i]
                assert len(w) > 0
                offset = int(rng.integers(len(w) - CHUNK + 1)) if len(w) >= CHUNK else 0
                take = min(CHUNK - filled, len(w) - offset)
                assert take > 0
                targets[filled:filled + take, col] = w[offset:offset + take]
                if offset:
                    inputs[filled, col] = w[offset - 1]
                else:
                    missing[filled, col] = True
                inputs[filled + 1:filled + take, col] = w[offset:offset + take - 1]
                filled += take
                sources.append(w_i)
        tensors = tuple(torch.from_numpy(a) for a in (targets, inputs, missing))
        return tensors, sources

    def eviction_fractions(self):
        if self.evicted == 0:
            return {p: 0.0 for p in F_PROCS}
        return {p: float(self.evicted_weight[i] / self.evicted) for i, p in enumerate(F_PROCS)}


def gru_sum_bits(model, x, start, state):
    """Sum of GRU cross-entropy bits on [start, start+CHUNK), from `state`.

    Same embedding path as the live scorer. Does not update `state` or weights.
    """
    end = start + CHUNK
    if start == 0:
        emb = torch.cat((model.embedding.weight.new_zeros(1, 32), model.embedding(x[:end - 1])))
    else:
        emb = model.embedding(x[start - 1:end - 1])
    hidden = None if state is None else state.detach().clone()
    with torch.no_grad():
        logits, _ = model(emb, hidden)
        losses = F.cross_entropy(logits, x[start:end], reduction='none')
        return float(losses.detach().double().sum().item()) / math.log(2)


def run(condition, seed, n, out):
    assert condition in NEW
    clock = time.monotonic()
    text, labels = stream_f('slow', seed, n)
    procs = np.asarray(labels)
    proc_ids = np.array([F_PROCS.index(p) for p in procs], dtype=np.int16)
    symbols = np.array([F_ALPHA.index(c) for c in text], dtype=np.uint8)
    x = torch.from_numpy(symbols.astype(np.int64))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Cortex()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    replay_credit = condition == 'util'
    store = UtilityStore(budget_bits(n), proc_ids)
    weights = np.full(2, .5)
    bits, cortex_bits = np.empty(n), np.empty(n)
    marked = np.zeros(n, dtype=bool)
    memory_trace, count_trace, window_trace = [], [], []
    context, state, replay_steps, pending = bytearray(), None, 0, 0
    predicted = None
    measure_max_abs = 0.0
    measure_chunks = 0

    def step(loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for start in range(0, n, CHUNK):
        end = start + CHUNK
        store.chunk = start // CHUNK
        store.decay()
        optimizer.zero_grad(set_to_none=True)
        emb = (torch.cat((model.embedding.weight.new_zeros(1, 32), model.embedding(x[:end - 1])))
               if start == 0 else model.embedding(x[start - 1:end - 1]))
        logits, state = model(emb, state)
        losses = F.cross_entropy(logits, x[start:end], reduction='none')
        gp = logits.detach().double().softmax(-1).numpy()
        cortex_bits[start:end] = losses.detach().double().numpy() / math.log(2)
        if predicted is not None:
            measure_max_abs = max(measure_max_abs, abs(float(cortex_bits[start:end].sum()) - predicted))
            measure_chunks += 1
            predicted = None
        for offset, symbol in enumerate(symbols[start:end]):
            t = start + offset
            likelihoods = np.array([gp[offset, symbol], store.likelihood(context, symbol, t)])
            likelihood = float(weights @ likelihoods)
            bits[t] = -math.log2(likelihood)
            store.credit_match(float(cortex_bits[t]), bits[t])
            weights *= likelihoods / likelihood
            weights = .999 * weights + .001 / 2
            context.append(int(symbol))
            if len(context) > 32:
                del context[0]
        state = detach(state)
        if pending > start:
            store.write(start, symbols[start:min(end, pending)], join_touching=True, fresh=False)
        indices = np.flatnonzero(bits[start:end] > THRESHOLD) + start
        marked[indices] = True
        intervals, pending = surprise_intervals(indices, n, end, pending)
        for lo, hi in intervals:
            if hi > lo:
                store.write(lo, symbols[lo:hi], fresh=True)
        store.end_chunk()
        memory_trace.append(store.memory())
        count_trace.append(store.count())
        window_trace.append(len(store.starts))
        step(losses.mean())
        if store.starts:
            before = None
            if replay_credit and end + CHUNK <= n:
                t0 = time.perf_counter()
                before = gru_sum_bits(model, x, end, state)
                store.credit_seconds += time.perf_counter() - t0
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                (targets, inputs, missing), sources = store.batch_sources(rng)
                emb = model.embedding(inputs).masked_fill(missing[:, :, None], 0)
                outputs, _ = model.gru(emb, None)
                rlogits = model.readout(outputs)
                step(F.cross_entropy(rlogits.reshape(-1, A), targets.reshape(-1)))
                replay_steps += 1
                if before is not None:
                    t0 = time.perf_counter()
                    after = gru_sum_bits(model, x, end, state)
                    store.credit_seconds += time.perf_counter() - t0
                    store.credit_replay(max(0.0, before - after), sources)
                    before = after
            if before is not None:
                predicted = before
        if (start // CHUNK + 1) % 32 == 0 or end == n:
            print(f'{now()} {condition} seed={seed} n={end}/{n} P={bits[:end].mean():.6f} '
                  f'M/N={store.memory() / n:.4f} windows={len(store.starts)} evicted={store.evicted} '
                  f'seconds={time.monotonic() - clock:.1f}', flush=True)
    assert np.isfinite(bits).all()
    mask = np.zeros(n, dtype=bool)
    for lo, w in zip(store.starts, store.windows):
        mask[lo:lo + len(w)] = True
    assert int(mask.sum()) == store.count()
    fractions = store.eviction_fractions()
    if store.evicted:
        assert abs(sum(fractions.values()) - 1.0) < 1e-8
    exposures = value_exposures(text, procs, bits)
    for stats in exposures.values():
        stats.pop('recall_bits')
        stats['bits_per_digit'] = stats['bits'] / stats['digits'] if stats['digits'] else None
    repeated = [exposures[b] for b in BUCKETS[1:]]
    digits = sum(s['digits'] for s in repeated)
    wall = time.monotonic() - clock
    extra = store.credit_seconds / (wall - store.credit_seconds) if wall > store.credit_seconds else None
    utils = np.array(store.utils, dtype=np.float64) if store.utils else np.array([math.nan])
    result = dict(
        condition=condition, seed=seed, n=n, P=float(bits.mean()),
        peak_M_bits=float(store.retained_peak), peak_M_over_N=float(store.retained_peak) / n,
        final_M_bits=float(store.memory()), final_M_over_N=float(store.memory()) / n,
        budget_M_bits=float(store.budget), budget_M_over_N=float(store.budget) / n,
        stored_symbols=int(store.count()), stored_windows=len(store.starts),
        fraction_stored=store.count() / n, evicted_windows=int(store.evicted),
        evicted_fraction_by_process=fractions,
        max_retained_over_budget_bits=float(max(0.0, store.max_over_budget)),
        chunks_over_budget=int(store.chunks_over_budget),
        per_process_P={p: float(bits[procs == p].mean()) if np.any(procs == p) else None for p in F_PROCS},
        process_counts={p: int(np.sum(procs == p)) for p in F_PROCS},
        F_value_exposures=exposures,
        F_value_at_least_1_prior_bits_per_digit=sum(s['bits'] for s in repeated) / digits if digits else None,
        replay_steps=int(replay_steps), replay_credit=bool(replay_credit),
        match_credit_bits=float(store.match_credit_bits),
        replay_credit_bits=float(store.replay_credit_bits),
        credit_seconds=float(store.credit_seconds),
        replay_extra_wall_fraction=None if extra is None else float(extra),
        replay_measure_max_abs_bits=float(measure_max_abs),
        replay_measure_chunks=int(measure_chunks),
        final_utility_min=None if not store.utils else float(utils.min()),
        final_utility_median=None if not store.utils else float(np.median(utils)),
        final_utility_max=None if not store.utils else float(utils.max()),
        parameters=sum(p.numel() for p in model.parameters()),
        wall_seconds=float(wall))
    np.savez_compressed(
        out / f'{condition}_{seed}.npz', symbols=symbols, processes=procs, bits=bits,
        cortex_bits=cortex_bits, marked=marked, stored=mask,
        window_starts=np.array(store.starts), window_lengths=np.array([len(w) for w in store.windows]),
        utilities=np.array(store.utils), born=np.array(store.born),
        memory_trace=np.array(memory_trace), count_trace=np.array(count_trace),
        window_trace=np.array(window_trace),
        evicted_weight=store.evicted_weight, evicted_windows=np.array([store.evicted]))
    return result


def carry_rows(n, seeds):
    path = Path('results/h24f') / ('smoke' if n == 8192 else '') / 'results.json'
    found = []
    for r in json.loads(path.read_text())['runs']:
        if r['condition'] not in CARRIED or r['seed'] not in seeds or r['n'] != n:
            continue
        row = dict(r)
        row['carried_from'] = str(path)
        final_m = row['stored_symbols'] * math.log2(A) + 32 * row['stored_windows']
        row['final_M_bits'] = final_m
        row['final_M_over_N'] = final_m / n
        row['evicted_windows'] = None
        row['evicted_fraction_by_process'] = None
        row['replay_credit'] = None
        found.append(row)
    assert len(found) == len(CARRIED) * len(seeds), (len(found), n, seeds)
    return found


def _rows(data, condition):
    return [r for r in data['runs'] if r['condition'] == condition]


def save(data, out):
    order = {c: i for i, c in enumerate(CONDITIONS)}
    data['runs'] = sorted(data['runs'], key=lambda r: (order.get(r['condition'], 99), r['seed']))
    (out / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    rows = data['runs']
    present = [c for c in CONDITIONS if any(r['condition'] == c for r in rows)]
    lines = ['# H24g measurements', '', f"Started: {data['started']}. Updated: {now()}.", '',
             f"Slow world; N = {data['n']}; seeds = {data['seeds']}; alphabet = 42; CPU threads = 2.",
             f"Budget B/N = {data['budget_M_over_N']:.6f} (0.15 × full5 peak M/N). Bar reference M/N ≤ 0.884 is not scored here.",
             '']
    lines += ['| Condition | Seed | P | Peak M/N | Final M/N | Fraction stored | Stored symbols | Final windows | Evicted windows | Wall seconds |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        ev = 'NA' if r['evicted_windows'] is None else str(r['evicted_windows'])
        lines.append(
            f"| {r['condition']} | {r['seed']} | {r['P']:.6f} | {r['peak_M_over_N']:.6f} | "
            f"{r['final_M_over_N']:.6f} | {r['fraction_stored']:.6f} | {r['stored_symbols']} | "
            f"{r['stored_windows']} | {ev} | {r['wall_seconds']:.3f} |")
    lines += ['', '| Condition | Seeds | P mean ± population std | Peak M/N mean ± std | Final M/N mean ± std | Fraction stored mean ± std |',
              '|---|---:|---:|---:|---:|---:|']
    for c in present:
        rs = _rows(data, c)
        lines.append('| ' + c + ' | ' + str(len(rs)) + ' | ' + ' | '.join(
            mean_std([r[k] for r in rs]) for k in ('P', 'peak_M_over_N', 'final_M_over_N', 'fraction_stored')) + ' |')
    lines += ['', '| Condition | Seed | ' + ' | '.join(F_PROCS) + ' |',
              '|---|---:|' + '---:|' * len(F_PROCS)]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | " + ' | '.join(fmt(r['per_process_P'][p]) for p in F_PROCS) + ' |')
    lines += ['', '| Condition | Seed | Prior key appearances | Value digits | Bits | Bits/digit |',
              '|---|---:|---|---:|---:|---:|']
    for r in rows:
        for b, s in r['F_value_exposures'].items():
            lines.append(f"| {r['condition']} | {r['seed']} | {b} | {s['digits']} | {s['bits']:.6f} | {fmt(s['bits_per_digit'])} |")
    lines += ['', '| Condition | Seed | F value bits/digit, ≥1 prior |', '|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | {fmt(r['F_value_at_least_1_prior_bits_per_digit'])} |")
    lines += ['', '| Condition | F ≥1 prior mean over observed seeds | F ≥1 prior pooled bits/digit |', '|---|---:|---:|']
    for c in present:
        rs = [r for r in _rows(data, c) if r['F_value_at_least_1_prior_bits_per_digit'] is not None]
        stats = [r['F_value_exposures'][b] for r in rs for b in BUCKETS[1:]]
        digits = sum(s['digits'] for s in stats)
        pooled = sum(s['bits'] for s in stats) / digits if digits else None
        mean = float(np.mean([r['F_value_at_least_1_prior_bits_per_digit'] for r in rs])) if rs else None
        lines.append(f'| {c} | {fmt(mean)} | {fmt(pooled)} |')
    new_rows = [r for r in rows if r['condition'] in NEW]
    lines += ['', '| Condition | Seed | Evicted windows | ' + ' | '.join(F_PROCS) + ' |',
              '|---|---:|---:|' + '---:|' * len(F_PROCS)]
    for r in new_rows:
        frac = r['evicted_fraction_by_process']
        lines.append(f"| {r['condition']} | {r['seed']} | {r['evicted_windows']} | "
                     + ' | '.join(f"{frac[p]:.6f}" for p in F_PROCS) + ' |')
    lines += ['', '| Condition | Seed | Match credit bits | Replay credit bits | Measure seconds | Extra wall fraction | Measure max abs bits | Final utility min | median | max |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in new_rows:
        lines.append(
            f"| {r['condition']} | {r['seed']} | {r['match_credit_bits']:.6f} | {r['replay_credit_bits']:.6f} | "
            f"{r['credit_seconds']:.3f} | {fmt(r['replay_extra_wall_fraction'])} | {r['replay_measure_max_abs_bits']:.6f} | "
            f"{fmt(r['final_utility_min'])} | {fmt(r['final_utility_median'])} | {fmt(r['final_utility_max'])} |")
    lines += ['', 'Pareto: P against peak M/N. gru, full, full5, fine and fine3 are carried from H24f (no rerun). '
              'gru has M = 0. full is H24e via H24f.', '',
              '| Condition | Seed | P | Peak M/N |', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['condition']} | {r['seed']} | {r['P']:.6f} | {r['peak_M_over_N']:.6f} |")
    lines += ['', '| Condition | Seeds | P mean ± population std | Peak M/N mean ± population std |', '|---|---:|---:|---:|']
    for c in present:
        rs = _rows(data, c)
        lines.append('| ' + c + ' | ' + str(len(rs)) + ' | ' + ' | '.join(
            mean_std([r[k] for r in rs]) for k in ('P', 'peak_M_over_N')) + ' |')
    lines += ['',
              'P: prequential observation bits/symbol under the fixed-share mixture of the GRU and the order-5 match. '
              'M = stored symbols × log2(42) + 32 bits/window. Peak M is the maximum memory retained after utility eviction; '
              'a write that crosses B is cut back by evicting whole unprotected windows before the peak is updated. '
              'Final M is memory at the last symbol. Denominator is final N.',
              'gru, full, full5, fine and fine3 rows are copied from H24f, including their historical wall seconds. No reruns. '
              'full5 keeps H24e aligned 64-symbol windows. fine keeps H24f surprise windows.',
              'Match order 5, latest stored target; backwards extension up to 32 inside the stored window. '
              'Confidence buckets: 5–7, 8–11, 12–15, 16–23, 24–32. Fixed share 0.001, initial weights (0.5, 0.5).',
              'util and util_m write like fine: mixture bits strictly > 4.0, inclusive window [i−16, i+4] clipped to the life, '
              'overlapping windows merged, touching windows kept separate. The unscored right tail is admitted after the next chunk '
              'is scored and joined to its own head (join_touching). That tail adds no 1-bit prior and does not restart protection.',
              'A newly written window (including one merged with older windows) starts at utility 1 bit plus the utilities of the windows it absorbed, '
              'and is protected for 32 chunks: the birth chunk and the next 31. Chunk indices are 0-based. '
              'At the start of each later chunk, every already stored window is multiplied by 0.999. Credits earned during a chunk are decayed on later chunks only.',
              'Match credit: when the order-5 match hits a stored window, add max(0, GRU-only bits of that symbol − mixture bits of that symbol) to that window. '
              'GRU-only bits are the live cross-entropy. Mixture bits are the prequential fixed-share bits, using mixture weights from before the per-symbol weight update. A miss credits nothing.',
              'Replay credit (util only; util_m is match credit only): after the live Adam step, two batch-8 replay steps follow H24f. '
              'When a next 64-symbol chunk exists, each replay step is charged max(0, GRU bits of that next chunk before the step − GRU bits after the step) / (number of segments in the batch). '
              'GRU bits are the sum of cross-entropy bits from the detached live hidden state, with the same input embedding as live scoring, under torch.no_grad. '
              'The hidden state is not updated by replay or by the measurement. The after-measurement of step 1 is the before-measurement of step 2. '
              'A segment is one contiguous draw from one window while filling the 8×64 batch (short windows are concatenated; a window drawn twice is two segments). '
              'Each segment adds that share to its window. The last chunk has no next chunk, so its replay steps add no replay credit. '
              'Replay still trains the GRU whenever the store is non-empty.',
              'Eviction, after every write: while M > B, drop the unprotected window with the lowest utility, breaking ties by earliest start. '
              'The order-5 index is rebuilt from the surviving windows at the end of the chunk, so an older stored copy of a context becomes the latest target after a newer copy is evicted. '
              'Evicted-window fractions use the true process ids of symbols inside each evicted window, weighted by symbol count, for analysis only. The learner never reads process ids.',
              'Live GRU and replay schedule: H24e Cortex; two batch-8 Adam replay steps after each scored chunk when the store is non-empty, live update then replay. '
              'Scoring and match credit happen before any current-chunk admission or weight update. '
              'Neural parameters, the optimizer, the match index and the transient live context are outside M.',
              'F exposure counts use all observations. NA denotes absent observations. Population standard deviation uses ddof=0.',
              f"New-run wall seconds (sum of util and util_m wall times; carried rows excluded): {data.get('total_wall_seconds', 0):.3f}.",
              '']
    lines += [f'Runtime note: {note}' for note in data.get('notes', [])]
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def checks():
    from h24b_world import F_ALPHA
    assert A == len(F_ALPHA) == 42 and CHUNK == 64
    ratio = full5_peak_bits(131072) / 131072
    assert abs(ratio - (math.log2(A) + 0.5)) < 1e-12
    assert abs(ratio - 5.892317) < 5e-7
    assert budget_bits(131072) / 131072 <= 0.884
    payload = np.arange(128, dtype=np.uint8) % A
    procs = np.zeros(128, dtype=np.int16)
    # Touching stays separate; overlapping merges and sums utility; tail join does not.
    s = UtilityStore(1e12, procs)
    s.chunk = 0
    s.write(0, payload[:10])
    s.write(10, payload[10:20])
    assert s.starts == [0, 10] and s.count() == 20 and s.utils == [1.0, 1.0] and s.born == [0, 0]
    s.utils = [3.0, 4.0]
    s.born = [1, 2]
    s.chunk = 6
    s.write(8, payload[8:25], fresh=True)
    assert s.starts == [0] and np.array_equal(s.windows[0], payload[:25]) and s.count() == 25
    assert abs(s.utils[0] - 8.0) < 1e-9 and s.born == [6]
    tail = UtilityStore(1e12, procs)
    tail.chunk = 3
    tail.write(0, payload[:10])
    tail.utils[0] = 5.0
    tail.born[0] = 3
    tail.chunk = 4
    tail.write(10, payload[10:14], join_touching=True, fresh=False)
    assert tail.starts == [0] and np.array_equal(tail.windows[0], payload[:14])
    assert abs(tail.utils[0] - 5.0) < 1e-9 and tail.born == [3]
    # Interval rule: overlap merges, touching does not.
    merged, pend = surprise_intervals([10, 30], 100, 64, 0)
    assert merged == [(0, 35)] and pend == 35
    separate, _ = surprise_intervals([10, 40], 100, 64, 0)
    assert separate == [(0, 15), (24, 45)]
    # Protection, then lowest utility leaves first. Process mix is symbol-weighted.
    mix = np.zeros(128, dtype=np.int16)
    mix[5:10] = 3  # E30; the low-utility window [0, 10) is half G, half E30
    ev = UtilityStore(1e12, mix)
    ev.chunk = 0
    ev.write(0, payload[:10])
    ev.write(20, payload[20:30])
    ev.utils = [0.1, 10.0]
    ev.chunk = 100
    ev.budget = ev.memory() - 1.0
    ev.evict_to_budget()
    assert ev.starts == [20] and abs(ev.utils[0] - 10.0) < 1e-9
    assert ev.evicted == 1
    frac = ev.eviction_fractions()
    assert abs(frac['G'] - 0.5) < 1e-9 and abs(frac['E30'] - 0.5) < 1e-9
    assert abs(sum(frac.values()) - 1) < 1e-9
    tight = UtilityStore(budget_bits(256), np.zeros(256, dtype=np.int16))
    for i in range(0, 200, 20):
        tight.chunk = 0
        tight.write(i, payload[:20])
    assert tight.memory() > tight.budget  # still inside the protection window
    tight.chunk = 100
    tight.evict_to_budget()
    tight.end_chunk()
    assert tight.memory() <= tight.budget + 1e-6 and tight.evicted > 0
    held = UtilityStore(0.0, procs)
    held.chunk = 0
    held.write(0, payload[:8])
    assert held.starts == [0]
    held.chunk = 31
    held.evict_to_budget()
    assert held.starts == [0]
    held.chunk = 32
    held.evict_to_budget()
    assert held.starts == [] and held.evicted == 1
    # Evicting the latest copy restores the older stored match.
    pattern = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)
    block = np.zeros(80, dtype=np.uint8)
    block[:6] = pattern
    block[30:36] = pattern
    rec = UtilityStore(1e12, np.zeros(80, dtype=np.int16))
    rec.chunk = 0
    rec.write(0, block[:6])
    rec.chunk = 1
    rec.write(30, block[30:36])
    ctx = bytearray(int(v) for v in pattern[:5])
    assert rec.likelihood(ctx, int(pattern[5]), 100) == 0.5 and rec.match_window == 1
    rec.utils = [10.0, 0.0]
    rec.chunk = 100
    rec.budget = 6 * math.log2(A) + 32 + 1.0
    rec.evict_to_budget()
    rec.end_chunk()
    assert rec.starts == [0] and rec.memory() <= rec.budget
    assert abs(rec.likelihood(ctx, int(pattern[5]), 100) - 2 / 3) < 1e-12
    assert rec.match_window == 0
    # Credits.
    rec.match_window = 0
    before_u = rec.utils[0]
    rec.credit_match(5.0, 2.0)
    rec.credit_match(1.0, 4.0)
    rec.match_window = None
    rec.credit_match(9.0, 0.0)
    assert abs(rec.utils[0] - (before_u + 3.0)) < 1e-9
    rec.utils = [1.0, 1.0]
    rec.credit_replay(3.0, [0, 1, 0])
    rec.credit_replay(0.0, [0])
    assert abs(rec.utils[0] - 3.0) < 1e-9 and abs(rec.utils[1] - 2.0) < 1e-9
    assert abs(rec.replay_credit_bits - 3.0) < 1e-9
    rec.decay()
    assert abs(rec.utils[0] - 3.0 * DECAY) < 1e-12
    # Replay sampling matches Fine.batch, including segment boundaries.
    sample = UtilityStore(1e12, procs)
    sample.chunk = 0
    sample.write(3, payload[3:10])
    sample.write(40, payload[40:120])
    got, sources = sample.batch_sources(np.random.default_rng(0))
    ref = sample.batch(np.random.default_rng(0))
    assert all(torch.equal(a, b) for a, b in zip(got, ref))
    assert len(sources) >= 8 and got[0].shape == (64, 8)
    # Reindex cost at the production store size. Informational.
    bench = UtilityStore(1e12, np.zeros(24000, dtype=np.int16))
    rng = np.random.default_rng(1)
    pos = 0
    bench.chunk = 0
    while pos + 40 <= 24000:
        bench.write(pos, rng.integers(0, A, size=40, dtype=np.uint8), fresh=True)
        pos += 40
    t0 = time.perf_counter()
    bench.reindex()
    dt = time.perf_counter() - t0
    print(f'{now()} reindex {bench.count()} symbols {len(bench.starts)} windows in {dt:.3f}s '
          f'(×2048 ≈ {dt * 2048:.0f}s if every chunk evicts)', flush=True)
    assert len(bench.last) > 1000
    carried = carry_rows(131072, [6, 7, 8])
    assert len(carried) == 15
    full5 = next(r for r in carried if r['condition'] == 'full5' and r['seed'] == 6)
    assert abs(full5['peak_M_over_N'] - full5_peak_bits(131072) / 131072) < 1e-9
    assert len(carry_rows(8192, [6])) == 5
    print(f'{now()} checks passed', flush=True)


def configure_threads():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def fresh_data(n, seeds):
    return dict(
        started=now(), n=n, seeds=list(seeds),
        principles=['1: history store', '2: compression', '3: compression progress'],
        budget_M_bits=budget_bits(n), budget_M_over_N=budget_bits(n) / n,
        notes=[], runs=carry_rows(n, seeds), torch_version=torch.__version__, checks_passed=True)


def close(data, out, clock, done):
    data['total_wall_seconds'] = sum(r['wall_seconds'] for r in data['runs'] if r['condition'] in NEW)
    data['elapsed_seconds'] = time.monotonic() - clock
    if done:
        data['finished'] = now()
    save(data, out)
    if done:
        print(f"{now()} ALL_DONE seconds={data['elapsed_seconds']:.1f} new_wall={data['total_wall_seconds']:.1f}", flush=True)


def execute(condition, seed, n, out, data):
    result = run(condition, seed, n, out)
    data['runs'] = [r for r in data['runs'] if not (r['condition'] == condition and r['seed'] == seed)]
    data['runs'].append(result)
    print(f"{now()} RUN_DONE {condition} seed={seed} P={result['P']:.6f} "
          f"peak_M/N={result['peak_M_over_N']:.6f} final_windows={result['stored_windows']} "
          f"evicted={result['evicted_windows']} extra={result['replay_extra_wall_fraction']} "
          f"measure_abs={result['replay_measure_max_abs_bits']:.4f} seconds={result['wall_seconds']:.1f}",
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-replay-credit', action='store_true')
    args = parser.parse_args()
    configure_threads()
    clock = time.monotonic()
    checks()
    if args.checks:
        return
    if args.smoke:
        out = Path('results/h24g/smoke')
        out.mkdir(parents=True, exist_ok=True)
        data = fresh_data(8192, [6])
        save(data, out)
        for condition in NEW:
            if condition == 'util' and args.no_replay_credit:
                continue
            execute(condition, 6, 8192, out, data)
            close(data, out, clock, done=False)
        close(data, out, clock, done=True)
        return
    out = Path('results/h24g')
    out.mkdir(parents=True, exist_ok=True)
    n, seeds = 131072, [6, 7, 8]
    if args.resume and (out / 'results.json').exists():
        data = json.loads((out / 'results.json').read_text())
        assert data['n'] == n
    else:
        data = fresh_data(n, seeds)
        if args.no_replay_credit:
            data['notes'].append('Replay credit (b) disabled by --no-replay-credit before this run.')
    # Smoke decides whether replay-credit forwards are too expensive to keep.
    smoke_path = Path('results/h24g/smoke/results.json')
    assert smoke_path.exists(), 'smoke results missing; run --smoke first'
    smoke = json.loads(smoke_path.read_text())
    data['smoke_wall_seconds'] = smoke.get('elapsed_seconds', smoke.get('total_wall_seconds'))
    smoke_util = [r for r in smoke['runs'] if r['condition'] == 'util' and r['seed'] == 6]
    drop_replay = bool(args.no_replay_credit)
    if smoke_util:
        extra = smoke_util[0]['replay_extra_wall_fraction']
        data['smoke_replay_extra_wall_fraction'] = extra
        data['smoke_replay_measure_max_abs_bits'] = smoke_util[0]['replay_measure_max_abs_bits']
        if extra is not None and extra > REPLAY_EXTRA_LIMIT:
            drop_replay = True
            note = (f"Replay credit (b) dropped: smoke measurement forwards were {extra:.3f} of the "
                    f"no-measurement wall, above {REPLAY_EXTRA_LIMIT:.2f}. Production util is match credit only.")
            if note not in data['notes']:
                data['notes'].append(note)
    elif not drop_replay:
        raise RuntimeError('smoke util run missing; cannot judge replay-credit cost')
    save(data, out)
    done_pairs = {(r['condition'], r['seed']) for r in data['runs'] if r['condition'] in NEW}
    for seed in seeds:
        if ('util', seed) in done_pairs:
            continue
        if drop_replay:
            # Match-credit-only util: skip the replay measurement inside run by using util_m's
            # flag only when the operator asked to drop (b). Keep the condition name `util`.
            result = run_match_only('util', seed, n, out)
            data['runs'] = [r for r in data['runs'] if not (r['condition'] == 'util' and r['seed'] == seed)]
            data['runs'].append(result)
            print(f"{now()} RUN_DONE util seed={seed} P={result['P']:.6f} peak_M/N={result['peak_M_over_N']:.6f} "
                  f"seconds={result['wall_seconds']:.1f}", flush=True)
        else:
            execute('util', seed, n, out, data)
        close(data, out, clock, done=False)
    if drop_replay:
        note = 'util_m not run: with replay credit dropped, util_m is the same procedure as util.'
        if note not in data['notes']:
            data['notes'].append(note)
        close(data, out, clock, done=True)
        return
    util_walls = [r['wall_seconds'] for r in data['runs'] if r['condition'] == 'util']
    typical = float(np.mean(util_walls)) if util_walls else 900.0
    for seed in seeds:
        if ('util_m', seed) in done_pairs:
            continue
        elapsed = time.monotonic() - clock
        remaining_seeds = seeds[seeds.index(seed):]
        # Seeds already skipped above are not in remaining if we continue the loop; recompute.
        remaining_seeds = [s for s in seeds if s >= seed and ('util_m', s) not in done_pairs and not any(
            r['condition'] == 'util_m' and r['seed'] == s for r in data['runs'])]
        need = typical * len(remaining_seeds)
        if elapsed + need > WALL_BUDGET:
            note = (f"util_m seeds {remaining_seeds} not started: elapsed {elapsed:.0f}s plus "
                    f"{len(remaining_seeds)} × observed util {typical:.0f}s exceeds the {WALL_BUDGET:.0f}s budget. "
                    f"util_m dropped first.")
            if note not in data['notes']:
                data['notes'].append(note)
            break
        execute('util_m', seed, n, out, data)
        close(data, out, clock, done=False)
        done_pairs.add(('util_m', seed))
    close(data, out, clock, done=True)


def run_match_only(condition, seed, n, out):
    """Run `condition` with match credit only. Used when replay credit has been dropped."""
    assert condition == 'util'
    # Temporary: run the util_m body, then relabel. Same code path as util_m.
    result = run('util_m', seed, n, out)
    src = out / f'util_m_{seed}.npz'
    dst = out / f'util_{seed}.npz'
    if src.exists():
        src.replace(dst)
    result['condition'] = 'util'
    result['replay_credit'] = False
    return result


if __name__ == '__main__':
    main()
