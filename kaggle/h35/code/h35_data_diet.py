"""H35: choose the next enwik8 document by compression progress.

counts_mix6 is H33's arm: orders 1–6, smoothing 0.05, GeoMixer weight sets
from H32b, Adam lr 0.003, one step per 128-byte chunk. Counts of a chunk are
added after that chunk is scored and the mixer has stepped. A document is one
prequential pass. Order-k context does not cross a document start.

Policies read B documents from a pool whose labels they do not see, except the
oracle. Numbers only. No verdict. Does not commit.
"""
import argparse
import hashlib
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

import h26_cued as h26
import h32_mixer as h32
import h32b_mixer as h32b
import h35_corpus as corpus

ALPHABET = 256
ORDERS = (1, 2, 3, 4, 5, 6)
N_ORDER = 6
N_SETS = 7
CHUNK = 128
ORDER4 = 3
SMOOTH = h32.SMOOTH
DENOM_EXTRA = h32.DENOM_EXTRA
LN2 = h32.LN2
K_CANDIDATES = 32
PEEK = 512
TRIAL = 256
PIECE = 256
RESERVOIR = 16
# Bits saved on the 256-byte eval span plus the reservoir, over the fixed
# denominator 512 (candidate peek) + 4096 (16 reservoir pieces).
PROGRESS_DENOM = PEEK + RESERVOIR * PIECE
B_FULL = 800
CHECKPOINT = 100
POLICIES = ('rand', 'oracle', 'high_loss', 'rho', 'prog_peek')
BITS_ATOL = 1e-6
OVERLAY_ATOL = 1e-9


def configure():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def py(value):
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    if isinstance(value, np.ndarray):
        return py(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(py(obj), indent=2, allow_nan=False) + '\n'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def _score_mixer(mixer, logs, selector, targets):
    report, train = h32.Mixer._nll_bits(
        None,
        mixer.W,
        torch.from_numpy(np.ascontiguousarray(logs, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(selector, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(targets, dtype=np.int64)))
    bits = report.detach().cpu().numpy()
    if not np.isfinite(bits).all():
        raise FloatingPointError(f'nonfinite {mixer.name} bits')
    return bits, train


def _step_mixer(mixer, train):
    mixer.opt.zero_grad(set_to_none=True)
    train.mean().backward()
    mixer.opt.step()


def _selectors(totals):
    """Highest order k with N_k >= 2, else 0. Matches h32b.weight_set_max."""
    sel = np.zeros(int(totals.shape[0]), dtype=np.int64)
    for k in range(1, N_ORDER + 1):
        sel = np.where(totals[:, k - 1] >= 2, k, sel)
    return sel


class CountsMix6:
    """H33 counts_mix6. One store, one geometric mixer, state kept by the caller."""

    def __init__(self):
        self.orders = [h32b.OrderCount(k) for k in ORDERS]
        init = torch.zeros(N_SETS, N_ORDER)
        init[:, ORDER4] = 1
        self.mixer = h32b.GeoMixer('counts_mix6', N_SETS, N_ORDER, init)
        row = self.mixer.W.detach()
        if float(row[0, ORDER4]) != 1.0 or float(row[0].sum()) != 1.0:
            raise RuntimeError('counts_mix6 init is not one-hot on order 4')

    def learn_blob(self, blob, keep=False):
        """Prequential pass. `blob` is bytes. Context is this blob only."""
        n = len(blob)
        if n < 1:
            raise RuntimeError('empty blob')
        kept = np.empty(n, dtype=np.float64) if keep else None
        total = 0.0
        pos = 0
        while pos < n:
            end = min(pos + CHUNK, n)
            bits = self._learn_chunk(blob, pos, end)
            total += float(bits.sum())
            if kept is not None:
                kept[pos:end] = bits
            pos = end
        return kept if keep else total

    def _learn_chunk(self, blob, start, end):
        tlen = end - start
        logs = np.empty((tlen, N_ORDER, ALPHABET), dtype=np.float32)
        dest = np.empty((N_ORDER, ALPHABET), dtype=np.float32)
        selector = np.empty(tlen, dtype=np.int64)
        orders = self.orders
        for t in range(tlen):
            totals = []
            for oi, order in enumerate(orders):
                totals.append(order.predict_log(blob, start + t, dest[oi]))
            logs[t] = dest
            selector[t] = h32b.weight_set_max(totals, N_ORDER)
        symbols = np.empty(tlen, dtype=np.int64)
        for t in range(tlen):
            symbols[t] = blob[start + t]
        bits, train = _score_mixer(self.mixer, logs, selector, symbols)
        _step_mixer(self.mixer, train)
        for t in range(tlen):
            i = start + t
            symbol = int(blob[i])
            for order in orders:
                order.add(blob, i, symbol)
        return bits

    def predict_range(self, blob, start, end, cache=None):
        """Frozen log rows, totals and context keys for blob[start:end).

        Keys are -1 for the empty context. Positions are indexes into `blob`,
        so a range can see bytes before `start`. `cache` stores orders 1–3,
        which repeat inside a frozen selection. Orders 4–6 are almost unique.
        """
        T = end - start
        logs = np.empty((T, N_ORDER, ALPHABET), dtype=np.float32)
        totals = np.empty((T, N_ORDER), dtype=np.int32)
        keys = np.empty((N_ORDER, T), dtype=np.int64)
        orders = self.orders
        for ti, i in enumerate(range(start, end)):
            row = logs[ti]
            for oi, order in enumerate(orders):
                k = oi + 1
                if i < k:
                    key_i = -1
                else:
                    key_i = int.from_bytes(blob[i - k:i], 'big')
                keys[oi, ti] = key_i
                slot = None if cache is None or oi >= 3 else cache[oi].get(key_i)
                if slot is not None:
                    row[oi][:] = slot[0]
                    totals[ti, oi] = slot[1]
                    continue
                totals[ti, oi] = order.predict_log(blob, i, row[oi])
                if cache is not None and oi < 3:
                    cache[oi][key_i] = (row[oi].copy(), int(totals[ti, oi]))
        return logs, totals, keys

    def bits_of(self, logs, totals, symbols):
        sel = _selectors(totals)
        with torch.no_grad():
            bits, _train = _score_mixer(self.mixer, logs, sel, symbols)
        return bits

    def frozen_bits(self, blob, cache=None):
        logs, totals, _keys = self.predict_range(blob, 0, len(blob), cache)
        symbols = np.frombuffer(blob, dtype=np.uint8).astype(np.int64)
        return self.bits_of(logs, totals, symbols)

    def frozen_bpb(self, blob, cache=None):
        bits = self.frozen_bits(blob, cache)
        return float(bits.sum()) / float(bits.shape[0])

    def score_test(self, test):
        total = 0.0
        n = 0
        for doc in test:
            bits = self.frozen_bits(np.asarray(doc, dtype=np.uint8).tobytes())
            total += float(bits.sum())
            n += int(bits.shape[0])
        return total / n

    def checksum(self):
        h = hashlib.sha256()
        for oi, order in enumerate(self.orders):
            keys = sorted(order.N.keys(), key=lambda k: -1 if k is None else int(k))
            if len(keys) != len(order.n):
                raise RuntimeError('count maps diverged')
            h.update(int(oi).to_bytes(4, 'little'))
            h.update(len(keys).to_bytes(8, 'little'))
            for key in keys:
                kb = b'\xff' if key is None else int(key).to_bytes(8, 'little')
                h.update(kb)
                h.update(int(order.N[key]).to_bytes(8, 'little', signed=False))
                for sym, count in sorted(order.n[key].items()):
                    h.update(int(sym).to_bytes(2, 'little'))
                    h.update(int(count).to_bytes(8, 'little', signed=False))
        h.update(self.mixer.W.detach().cpu().numpy().tobytes())
        for state in self.mixer.opt.state.values():
            for val in state.values():
                if torch.is_tensor(val):
                    h.update(val.detach().cpu().numpy().tobytes())
                else:
                    h.update(repr(val).encode())
        return h.hexdigest()


class Overlay:
    """Count deltas on top of a CountsMix6 store. The store itself is not written."""

    def __init__(self):
        self.n = [{} for _ in ORDERS]
        self.N = [{} for _ in ORDERS]

    def add_blob(self, blob):
        for i, sym in enumerate(blob):
            sym = int(sym)
            for oi, k in enumerate(ORDERS):
                key = None if i < k else int.from_bytes(blob[i - k:i], 'big')
                bucket = self.n[oi].get(key)
                if bucket is None:
                    self.n[oi][key] = {sym: 1}
                    self.N[oi][key] = 1
                else:
                    bucket[sym] = bucket.get(sym, 0) + 1
                    self.N[oi][key] += 1


def merged_logp(order, ov_n, ov_N, key, out):
    """Same arithmetic as OrderCount.predict_log, with overlay deltas added."""
    extra_bucket = ov_n.get(key)
    extra = 0 if extra_bucket is None else ov_N[key]
    total = order.N.get(key, 0) + extra
    denom = total + DENOM_EXTRA
    unseen = math.log(SMOOTH / denom)
    out.fill(unseen)
    base = order.n.get(key)
    if extra_bucket is None:
        if base:
            for sym, count in base.items():
                out[sym] = math.log((count + SMOOTH) / denom)
        return total
    if not base:
        for sym, count in extra_bucket.items():
            out[sym] = math.log((count + SMOOTH) / denom)
        return total
    seen = set()
    for sym, count in base.items():
        out[sym] = math.log((count + extra_bucket.get(sym, 0) + SMOOTH) / denom)
        seen.add(sym)
    for sym, count in extra_bucket.items():
        if sym not in seen:
            out[sym] = math.log((count + SMOOTH) / denom)
    return total


def key_index(keys):
    """Map (order, key) to positions. Built once per frozen segment."""
    maps = []
    for oi in range(N_ORDER):
        grouped = {}
        for t, key in enumerate(keys[oi]):
            grouped.setdefault(int(key), []).append(t)
        maps.append({k: np.asarray(v, dtype=np.int32) for k, v in grouped.items()})
    return maps


def apply_overlay(model, logs, totals, index, ov):
    logs = np.array(logs, dtype=np.float32, copy=True)
    totals = np.array(totals, dtype=np.int32, copy=True)
    row = np.empty(ALPHABET, dtype=np.float32)
    for oi, order in enumerate(model.orders):
        pos_map = index[oi]
        for key, extra in ov.N[oi].items():
            kk = -1 if key is None else int(key)
            hit = pos_map.get(kk)
            if hit is None:
                continue
            merged_logp(order, ov.n[oi], ov.N[oi], key, row)
            logs[hit, oi] = row
            totals[hit, oi] += np.int32(extra)
    return logs, totals


def new_cache():
    return [{} for _ in range(3)]


def prepare_blob_range(model, blob, start, end, cache=None):
    logs, totals, keys = model.predict_range(blob, start, end, cache)
    symbols = np.empty(end - start, dtype=np.int64)
    for i, pos in enumerate(range(start, end)):
        symbols[i] = blob[pos]
    return {
        'logs': logs,
        'totals': totals,
        'index': key_index(keys),
        'symbols': symbols,
        'before': model.bits_of(logs, totals, symbols),
    }


def prepare_pieces(model, pieces, cache=None):
    if not pieces:
        return None
    if cache is None:
        cache = new_cache()
    parts = [prepare_blob_range(model, piece, 0, len(piece), cache) for piece in pieces]
    return {
        'logs': np.concatenate([p['logs'] for p in parts], axis=0),
        'totals': np.concatenate([p['totals'] for p in parts], axis=0),
        'index': key_index_from_parts(parts),
        'symbols': np.concatenate([p['symbols'] for p in parts], axis=0),
        'before': np.concatenate([p['before'] for p in parts], axis=0),
    }


def key_index_from_parts(parts):
    """Rebuild a joint index. Piece lengths are PIECE, keys do not cross pieces.

    Each part already has an index of local positions. Shift them.
    """
    maps = [{} for _ in ORDERS]
    offset = 0
    for part in parts:
        n = int(part['symbols'].shape[0])
        for oi in range(N_ORDER):
            for key, hit in part['index'][oi].items():
                shifted = hit + offset
                prev = maps[oi].get(key)
                if prev is None:
                    maps[oi][key] = shifted
                else:
                    maps[oi][key] = np.concatenate((prev, shifted))
        offset += n
    return maps


class Reservoir:
    def __init__(self, rng):
        self.rng = rng
        self.pieces = []
        self.n = 0

    def add(self, doc):
        piece = np.asarray(doc[:PIECE], dtype=np.uint8).tobytes()
        if len(piece) != PIECE:
            raise RuntimeError('reservoir piece')
        self.n += 1
        if len(self.pieces) < RESERVOIR:
            self.pieces.append(piece)
            return
        j = int(self.rng.integers(0, self.n))
        if j < RESERVOIR:
            self.pieces[j] = piece


def progress_score(model, doc_blob, prepared_res, cache=None):
    """Bits saved by a trial update on bytes 0–255, over PROGRESS_DENOM.

    The mixer is not stepped. Counts land in an overlay.
    """
    ov = Overlay()
    ov.add_blob(doc_blob[:TRIAL])
    ev = prepare_blob_range(model, doc_blob, TRIAL, PEEK, cache)
    logs_a, totals_a = apply_overlay(model, ev['logs'], ev['totals'], ev['index'], ov)
    after = model.bits_of(logs_a, totals_a, ev['symbols'])
    saved = float(ev['before'].sum() - after.sum())
    if prepared_res is not None:
        logs_r, totals_r = apply_overlay(
            model, prepared_res['logs'], prepared_res['totals'], prepared_res['index'], ov)
        after_r = model.bits_of(logs_r, totals_r, prepared_res['symbols'])
        saved += float(prepared_res['before'].sum() - after_r.sum())
    return saved / PROGRESS_DENOM


def _unread(alive):
    return np.flatnonzero(alive)


def _candidate_scores(policy, model, pool, cand, ref, ref_cache, prepared_res, cache,
                      ref_rows):
    """`cache` is the live model's frozen rows. `ref_rows` is the reference model's.

    The two stores differ, so the row caches must not be shared.
    """
    scores = np.empty(len(cand), dtype=np.float64)
    fresh_ref = []
    for i, idx in enumerate(cand):
        idx = int(idx)
        if policy == 'prog_peek':
            scores[i] = progress_score(model, pool[idx].tobytes(), prepared_res, cache)
            continue
        blob = pool[idx, :PEEK].tobytes()
        current = model.frozen_bpb(blob, cache)
        if policy == 'high_loss':
            scores[i] = current
        elif policy == 'rho':
            cached = ref_cache.get(idx)
            if cached is None:
                cached = ref.frozen_bpb(blob, ref_rows)
                fresh_ref.append((idx, cached))
            scores[i] = current - cached
        else:
            raise RuntimeError(policy)
    return scores, fresh_ref


def _fork_candidate_scores(policy, model, pool, cand, ref, ref_cache, prepared_res, cache,
                           ref_rows, workers):
    """Score candidates in forked children. The count store is read only."""
    import os
    workers = int(workers)
    if workers <= 1 or len(cand) < 2:
        return _candidate_scores(
            policy, model, pool, cand, ref, ref_cache, prepared_res, cache, ref_rows)
    workers = min(workers, len(cand))
    # Fork after torch has already run. One thread in the parent keeps the
    # child from inheriting a live OpenMP pool.
    torch.set_num_threads(1)
    splits = [arr for arr in np.array_split(np.asarray(cand, dtype=np.int64), workers) if len(arr)]
    readers = []
    pids = []
    for chunk in splits:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                torch.set_num_threads(1)
                scores, fresh = _candidate_scores(
                    policy, model, pool, chunk, ref, ref_cache, prepared_res, cache, ref_rows)
                payload = json.dumps({
                    'scores': scores.tolist(),
                    'fresh': fresh,
                }).encode()
                os.write(write_fd, payload)
            except Exception as exc:
                try:
                    os.write(write_fd, b'ERR' + repr(exc).encode())
                except Exception:
                    pass
            os._exit(0)
        os.close(write_fd)
        readers.append(read_fd)
        pids.append(pid)
    parts = []
    errors = []
    for read_fd, pid, chunk in zip(readers, pids, splits):
        blob = b''
        while True:
            got = os.read(read_fd, 1 << 20)
            if not got:
                break
            blob += got
        os.close(read_fd)
        _pid, status = os.waitpid(pid, 0)
        if os.waitstatus_to_exitcode(status) != 0 or blob.startswith(b'ERR') or not blob:
            errors.append(f'worker {pid} status={status} {blob[:400]!r}')
            continue
        payload = json.loads(blob)
        parts.append((chunk, np.asarray(payload['scores'], dtype=np.float64), payload['fresh']))
    if errors:
        raise RuntimeError('peek workers failed: ' + '; '.join(errors))
    order = {int(idx): i for i, idx in enumerate(cand)}
    scores = np.empty(len(cand), dtype=np.float64)
    fresh_ref = []
    for chunk, chunk_scores, fresh in parts:
        if len(chunk_scores) != len(chunk):
            raise RuntimeError('worker score length')
        for idx, score in zip(chunk, chunk_scores):
            scores[order[int(idx)]] = float(score)
        fresh_ref.extend(fresh)
    return scores, fresh_ref


def select_index(policy, model, pool, labels, alive, rng, ref, ref_cache, reservoir_pieces,
                 peek_workers=1, ref_rows=None):
    unread = _unread(alive)
    if unread.size < 1:
        raise RuntimeError('no unread document')
    if policy == 'rand':
        return int(rng.choice(unread))
    if policy == 'oracle':
        clean = np.flatnonzero(alive & (labels == 'clean'))
        if clean.size < 1:
            raise RuntimeError('oracle ran out of clean documents')
        return int(rng.choice(clean))
    if unread.size < K_CANDIDATES:
        raise RuntimeError(f'unread {unread.size} < {K_CANDIDATES}')
    cand = rng.choice(unread, size=K_CANDIDATES, replace=False)
    cache = new_cache()
    prepared_res = None
    if policy == 'prog_peek':
        prepared_res = prepare_pieces(model, reservoir_pieces, cache)
    scores, fresh_ref = _fork_candidate_scores(
        policy, model, pool, cand, ref, ref_cache, prepared_res, cache, ref_rows, peek_workers)
    for idx, cached in fresh_ref:
        ref_cache[int(idx)] = float(cached)
    best = int(np.argmax(scores))
    return int(cand[best])


def _kind_shares(kinds):
    n = len(kinds)
    out = {}
    for kind in corpus.KINDS:
        out[kind] = kinds.count(kind) / n
    return out


def run_policy(seed, policy, budget, pool, labels, test, reference, log_every=25,
               peek_workers=1):
    if policy not in POLICIES:
        raise RuntimeError(policy)
    configure()
    model = CountsMix6()
    ref = None
    ref_cache = {}
    ref_rows = new_cache() if policy == 'rho' else None
    if policy == 'rho':
        ref = CountsMix6()
        ref.learn_blob(np.asarray(reference, dtype=np.uint8).tobytes())
    rng = np.random.default_rng(1_000 + int(seed))
    reservoir = Reservoir(np.random.default_rng(2_000 + int(seed))) if policy == 'prog_peek' else None
    alive = np.ones(pool.shape[0], dtype=bool)
    kinds = []
    checkpoints = []
    clock = time.perf_counter()
    for n_read in range(1, int(budget) + 1):
        pieces = reservoir.pieces if reservoir is not None else None
        idx = select_index(
            policy, model, pool, labels, alive, rng, ref, ref_cache, pieces,
            peek_workers=peek_workers, ref_rows=ref_rows)
        if not alive[idx]:
            raise RuntimeError('reread')
        alive[idx] = False
        doc = pool[idx]
        model.learn_blob(doc.tobytes())
        kinds.append(str(labels[idx]))
        if reservoir is not None:
            reservoir.add(doc)
        if n_read % CHECKPOINT == 0:
            bpb = model.score_test(test)
            checkpoints.append({'docs': n_read, 'bits_per_byte': bpb})
            print(
                f'H35 seed={seed} policy={policy} docs={n_read} '
                f'test_bpb={bpb:.6f} seconds={time.perf_counter() - clock:.1f}',
                flush=True)
        elif log_every and n_read % log_every == 0:
            print(
                f'H35 seed={seed} policy={policy} docs={n_read} '
                f'seconds={time.perf_counter() - clock:.1f}',
                flush=True)
    wall = time.perf_counter() - clock
    if len(kinds) != int(budget):
        raise RuntimeError('budget')
    shares = _kind_shares(kinds)
    final = checkpoints[-1]['bits_per_byte'] if checkpoints else None
    t_clean = float(np.mean([row['bits_per_byte'] for row in checkpoints])) if checkpoints else None
    return {
        'seed': int(seed),
        'policy': policy,
        'budget': int(budget),
        'T_clean': t_clean,
        'final_bits_per_byte': final,
        'checkpoints': checkpoints,
        'kind_counts': {kind: kinds.count(kind) for kind in corpus.KINDS},
        'kind_share': shares,
        'kinds_read': kinds,
        'wall_seconds': wall,
        'progress_denom': PROGRESS_DENOM,
        'reservoir_piece': 'doc[0:256]',
    }


def h33_counts_mix6_bpb(raw):
    """H33 counts_mix6 on one contiguous stream. Bank is not run; it does not feed counts."""
    import h33_bank_text as h33

    raw = np.asarray(raw, dtype=np.uint8)
    n = int(raw.shape[0])
    blob = raw.tobytes()
    tokens = raw.astype(np.int64, copy=False)
    _bank, counts_mix = h33._new_mixers()
    store = h33.CountStore()
    total = 0.0
    pos = 0
    while pos < n:
        end = min(pos + CHUNK, n)
        tlen = end - pos
        targets = tokens[pos:end]
        logs = np.empty((tlen, N_ORDER, ALPHABET), dtype=np.float32)
        dest = np.empty((N_ORDER, ALPHABET), dtype=np.float32)
        selector = np.empty(tlen, dtype=np.int64)
        for t in range(tlen):
            totals = store.predict(blob, pos + t, dest)
            logs[t] = dest
            selector[t] = h32b.weight_set_max(totals, N_ORDER)
        bits, train = h33._score_mixer(counts_mix, logs, selector, targets)
        h33._step_mixer(counts_mix, train)
        for t in range(tlen):
            store.add(blob, pos + t, int(targets[t]))
        total += float(np.asarray(bits, dtype=np.float64).sum())
        pos = end
    return total / n


def _deepcopy_orders(orders):
    import copy
    return copy.deepcopy(orders)


def _score_with_orders(model, orders, blob, start, end):
    saved = model.orders
    model.orders = orders
    try:
        logs, totals, _keys = model.predict_range(blob, start, end)
        symbols = np.empty(end - start, dtype=np.int64)
        for i, pos in enumerate(range(start, end)):
            symbols[i] = blob[pos]
        return model.bits_of(logs, totals, symbols)
    finally:
        model.orders = saved


def overlay_matches_copy(model, doc_blob, pieces):
    """Return (checksum_same, max_abs_bit_diff, score_diff)."""
    before = model.checksum()
    ov_score = progress_score(model, doc_blob, prepare_pieces(model, pieces))
    after = model.checksum()
    copied = _deepcopy_orders(model.orders)
    trial = doc_blob[:TRIAL]
    for i, sym in enumerate(trial):
        for order in copied:
            order.add(trial, i, int(sym))
    bits_before_ev = _score_with_orders(model, model.orders, doc_blob, TRIAL, PEEK)
    bits_after_ev = _score_with_orders(model, copied, doc_blob, TRIAL, PEEK)
    saved = float(bits_before_ev.sum() - bits_after_ev.sum())
    diffs = [float(np.max(np.abs(bits_before_ev - _score_with_orders(
        model, model.orders, doc_blob, TRIAL, PEEK))))]
    # The line above is zero by identity. Compare after-bits via a second overlay pass.
    ov = Overlay()
    ov.add_blob(trial)
    ev = prepare_blob_range(model, doc_blob, TRIAL, PEEK)
    logs_a, totals_a = apply_overlay(model, ev['logs'], ev['totals'], ev['index'], ov)
    ov_after = model.bits_of(logs_a, totals_a, ev['symbols'])
    diffs.append(float(np.max(np.abs(ov_after - bits_after_ev))))
    if pieces:
        for piece in pieces:
            b0 = _score_with_orders(model, model.orders, piece, 0, len(piece))
            b1 = _score_with_orders(model, copied, piece, 0, len(piece))
            saved += float(b0.sum() - b1.sum())
            part = prepare_blob_range(model, piece, 0, len(piece))
            la, ta = apply_overlay(model, part['logs'], part['totals'], part['index'], ov)
            got = model.bits_of(la, ta, part['symbols'])
            diffs.append(float(np.max(np.abs(got - b1))))
    copy_score = saved / PROGRESS_DENOM
    return before == after, max(diffs), abs(ov_score - copy_score)


def smoke(data_path='data/enwik8/enwik8', save_dir=corpus.DATA_DIR):
    configure()
    raw, digest = corpus.load_raw(data_path)
    lines = [f'sha256_e10m={digest}']
    pool, labels, test = corpus.build(0, raw)
    reference = corpus.reference_bytes(raw)
    counts = {kind: int(np.sum(labels == kind)) for kind in corpus.KINDS}
    lines.append('counts ' + ' '.join(f'{k}={counts[k]}' for k in corpus.KINDS))
    if counts['clean'] != 1600 or any(counts[k] != 320 for k in corpus.KINDS if k != 'clean'):
        raise SystemExit(f'kind counts {counts}')
    if test.shape != (64, 4096) or reference.shape != (100_000,):
        raise SystemExit('test or reference shape')
    for kind in corpus.KINDS:
        idx = int(np.flatnonzero(labels == kind)[0])
        sample = pool[idx, :80].tobytes()
        lines.append(f'sample {kind} {sample!r}')
    clean_set = {pool[i].tobytes() for i in range(pool.shape[0]) if labels[i] == 'clean'}
    dup_idx = np.flatnonzero(labels == 'J3_dup')
    n_ok = sum(1 for i in dup_idx if pool[i].tobytes() in clean_set)
    lines.append(f'J3_dup byte-identical {n_ok}/{len(dup_idx)}')
    if n_ok != len(dup_idx) or len(dup_idx) != 320:
        raise SystemExit('J3 duplicates are not copies of the clean pool')
    # Distinct sources: 16 clean documents, 20 copies each.
    dup_ids = {}
    for i in dup_idx:
        dup_ids.setdefault(pool[i].tobytes(), 0)
        dup_ids[pool[i].tobytes()] += 1
    if len(dup_ids) != 16 or any(v != 20 for v in dup_ids.values()):
        raise SystemExit(f'J3 source counts {sorted(dup_ids.values())}')
    lines.append('J3 sources 16 x 20')

    for c in range(6):
        p, lab, te = corpus.build(c, raw)
        corpus.save(c, p, lab, te, reference, save_dir)
    lines.append(f'saved data/h35/seed0.npz .. seed5.npz sha0={corpus.corpus_sha256(pool, labels, test)}')

    n_match = 20_000
    mine = CountsMix6()
    mine_bits = mine.learn_blob(raw[:n_match].tobytes(), keep=True)
    mine_bpb = float(mine_bits.sum()) / n_match
    ref_bpb = h33_counts_mix6_bpb(raw[:n_match])
    gap = abs(mine_bpb - ref_bpb)
    lines.append(f'learner_bpb={mine_bpb:.10f} h33_bpb={ref_bpb:.10f} abs_diff={gap:.3e}')
    if gap > BITS_ATOL:
        raise SystemExit(f'counts_mix6 mismatch {gap}')

    doc0 = pool[0].tobytes()
    doc1 = pool[1].tobytes()
    a = CountsMix6()
    bits_a = a.learn_blob(doc0, keep=True)
    b = CountsMix6()
    bits_b0 = b.learn_blob(doc0, keep=True)
    b.learn_blob(doc1, keep=False)
    pre_gap = float(np.max(np.abs(bits_a - bits_b0)))
    lines.append(f'prequential_max_abs={pre_gap:.3e}')
    if pre_gap > 0.0:
        raise SystemExit('later document changed earlier bits')

    warm = CountsMix6()
    warm.learn_blob(pool[0].tobytes())
    warm.learn_blob(pool[1].tobytes())
    pieces = [pool[0, :PIECE].tobytes(), pool[1, :PIECE].tobytes()]
    same, bit_gap, score_gap = overlay_matches_copy(warm, pool[2].tobytes(), pieces)
    lines.append(
        f'overlay_checksum_same={int(same)} max_abs_bits={bit_gap:.3e} '
        f'score_abs={score_gap:.3e}')
    if not same or bit_gap > OVERLAY_ATOL or score_gap > OVERLAY_ATOL:
        raise SystemExit(f'overlay check failed same={same} bits={bit_gap} score={score_gap}')

    kind_lines = []
    for policy in POLICIES:
        t0 = time.perf_counter()
        result = run_policy(0, policy, 40, pool, labels, test, reference, log_every=10)
        dt = time.perf_counter() - t0
        shares = ' '.join(f'{k}={result["kind_counts"][k]}' for k in corpus.KINDS)
        line = f'B40 {policy} seconds={dt:.1f} {shares}'
        lines.append(line)
        kind_lines.append(line)
        if sum(result['kind_counts'].values()) != 40:
            raise SystemExit('B40 did not read 40')
        if policy == 'oracle' and result['kind_counts']['clean'] != 40:
            raise SystemExit('oracle read junk')

    t0 = time.perf_counter()
    timed = run_policy(0, 'prog_peek', 100, pool, labels, test, reference, log_every=20)
    dt100 = time.perf_counter() - t0
    # B=800 is 8× the documents. Store growth is near-linear in Python dicts;
    # scale by 8 and leave the measured B=100 checkpoint inside the figure.
    est800 = dt100 * (B_FULL / 100)
    lines.append(
        f'prog_peek_B100_seconds={dt100:.1f} T_clean={timed["T_clean"]:.6f} '
        f'est_B800_seconds={est800:.0f}')
    lines.append(
        'kernel_wall_note one kernel runs up to 4 policies at once; '
        f'prog_peek est {est800/60:.1f} min is the long pole if it stays single-threaded')
    text = '\n'.join(lines)
    print(text, flush=True)
    return {
        'ok': True,
        'sha256_e10m': digest,
        'lines': lines,
        'prog_peek_B100_seconds': dt100,
        'prog_peek_B800_estimate_seconds': est800,
        'B40': kind_lines,
    }


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _fmt(value):
    if value is None:
        return 'na'
    return f'{value:.6f}'


def write_results(payload, path_md, path_json):
    dump(path_json, payload)
    seeds = payload['seeds']
    policies = payload['policies']
    rows = []
    pre_hits = []
    bar1_hits = []
    bar2_hits = []
    for seed in seeds:
        by = payload['by_seed'][str(seed)]
        oracle = by['oracle']['T_clean']
        rand = by['rand']['T_clean']
        prog = by['prog_peek']['T_clean']
        high = by['high_loss']['T_clean']
        d_or = oracle - rand
        d_pr = prog - rand
        d_ph = prog - high
        half = 0.5 * d_or
        pre = int(d_or <= -0.10)
        b1 = int(d_pr <= half)
        b2 = int(d_ph <= -0.05)
        pre_hits.append(pre)
        bar1_hits.append(b1)
        bar2_hits.append(b2)
        rows.append((seed, d_or, d_pr, d_ph, half, pre, b1, b2))
    lines = [
        '# H35 compression-progress data diet',
        '',
        f'Seeds: {seeds}. Budget B = {payload["budget"]}. '
        f'Checkpoints every {CHECKPOINT} documents. '
        f'T_clean is the mean of those checkpoint bits/byte on the clean test set.',
        f'sha256 of the first 10 MB of enwik8: `{payload["sha256_e10m"]}`.',
        '',
        '## Scoring table',
        '',
        'Quantities are on T_clean. pre is 1 when oracle−rand ≤ −0.10. '
        'bar1 is 1 when prog_peek−rand ≤ 0.5·(oracle−rand). '
        'bar2 is 1 when prog_peek−high_loss ≤ −0.05.',
        '',
        '| seed | oracle−rand | prog_peek−rand | prog_peek−high_loss | 0.5·(oracle−rand) | pre | bar1 | bar2 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for seed, d_or, d_pr, d_ph, half, pre, b1, b2 in rows:
        lines.append(
            f'| {seed} | {_fmt(d_or)} | {_fmt(d_pr)} | {_fmt(d_ph)} | {_fmt(half)} | {pre} | {b1} | {b2} |')
    lines.append('')
    lines.append(
        f'pre count {sum(pre_hits)}/{len(seeds)}. '
        f'bar1 count {sum(bar1_hits)}/{len(seeds)}. '
        f'bar2 count {sum(bar2_hits)}/{len(seeds)}.')
    lines.append('')
    lines.append('## T_clean and checkpoints')
    lines.append('')
    lines.append('Mean ± std across seeds, ddof = 0.')
    lines.append('')
    header = '| policy | quantity | ' + ' | '.join(f's{s}' for s in seeds) + ' | mean | std |'
    sep = '|---|---| ' + ' | '.join('---:' for _ in seeds) + ' |---:|---:|'
    lines.append(header)
    lines.append(sep)
    quantities = [('T_clean', lambda run: run['T_clean']),
                  ('final', lambda run: run['final_bits_per_byte'])]
    docs = [row['docs'] for row in payload['by_seed'][str(seeds[0])][policies[0]]['checkpoints']]
    for docs_n in docs:
        quantities.append((f'cp{docs_n}', lambda run, docs_n=docs_n: next(
            row['bits_per_byte'] for row in run['checkpoints'] if row['docs'] == docs_n)))
    for policy in policies:
        for name, fn in quantities:
            vals = [fn(payload['by_seed'][str(s)][policy]) for s in seeds]
            mean, std = _mean_std(vals)
            cells = ' | '.join(_fmt(v) for v in vals)
            lines.append(f'| {policy} | {name} | {cells} | {_fmt(mean)} | {_fmt(std)} |')
    lines.append('')
    lines.append('## Share of read documents')
    lines.append('')
    lines.append(header)
    lines.append(sep)
    for policy in policies:
        for kind in corpus.KINDS:
            vals = [payload['by_seed'][str(s)][policy]['kind_share'][kind] for s in seeds]
            mean, std = _mean_std(vals)
            cells = ' | '.join(_fmt(v) for v in vals)
            lines.append(f'| {policy} | {kind} | {cells} | {_fmt(mean)} | {_fmt(std)} |')
    lines.append('')
    lines.append('## Smoke')
    lines.append('')
    lines.append('```')
    lines.extend(payload.get('smoke_lines', []))
    lines.append('```')
    lines.append('')
    lines.append('## Walls and kernels')
    lines.append('')
    lines.append('| kernel | wall_seconds | slot_seconds | status |')
    lines.append('|---|---:|---:|---|')
    for row in payload['kernels']:
        wall = row.get('wall_seconds', 'na')
        slot = row.get('slot_seconds', 'na')
        wall_s = f'{wall:.1f}' if isinstance(wall, (int, float)) else str(wall)
        slot_s = f'{slot:.1f}' if isinstance(slot, (int, float)) else str(slot)
        lines.append(f'| {row["id"]} | {wall_s} | {slot_s} | {row.get("status", "")} |')
    lines.append('')
    lines.append('Policy walls, seconds:')
    lines.append('')
    lines.append(header)
    lines.append(sep)
    for policy in policies:
        vals = [payload['by_seed'][str(s)][policy]['wall_seconds'] for s in seeds]
        mean, std = _mean_std(vals)
        cells = ' | '.join(f'{v:.1f}' for v in vals)
        lines.append(f'| {policy} | wall_seconds | {cells} | {mean:.1f} | {std:.1f} |')
    lines.append('')
    Path(path_md).parent.mkdir(parents=True, exist_ok=True)
    Path(path_md).write_text('\n'.join(lines) + '\n')


def find_kaggle_enwik8():
    hits = [
        p for p in Path('/kaggle/input').rglob('enwik8')
        if p.is_file() and p.stat().st_size == h26.FULL
    ]
    if len(hits) != 1:
        raise SystemExit(f'expected one enwik8 of {h26.FULL} bytes, found {hits}')
    return str(hits[0])


def _policy_process(seed, policy, budget, data_path, out_path, kernel_id, peek_workers):
    configure()
    raw, digest = corpus.load_raw(data_path)
    pool, labels, test = corpus.build(seed, raw)
    reference = corpus.reference_bytes(raw)
    result = run_policy(
        seed, policy, budget, pool, labels, test, reference,
        log_every=50, peek_workers=peek_workers)
    result['sha256_e10m'] = digest
    result['kernel_id'] = kernel_id
    dump(out_path, result)
    print(f'wrote {out_path} wall={result["wall_seconds"]:.1f}', flush=True)


def kernel_main(seed, out, kernel_id, policies=None, budget=B_FULL, workers=4,
                peek_workers=1):
    """One private CPU kernel. Up to `workers` policies at once.

    `peek_workers` forks that many children to score one policy's candidates.
    Use 1 when several policies already fill the machine. A prog_peek-only
    kernel passes peek_workers=4 and workers=1.
    """
    import multiprocessing as mp

    configure()
    policies = list(policies or POLICIES)
    data_path = find_kaggle_enwik8()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # spawn children re-import this module. The dataset directory is not on
    # their default path unless the parent publishes it.
    code_dir = str(Path(__file__).resolve().parent)
    prior = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = code_dir if not prior else code_dir + os.pathsep + prior
    ctx = mp.get_context('spawn')
    pending = list(policies)
    running = {}
    finished = []
    while pending or running:
        while pending and len(running) < int(workers):
            policy = pending.pop(0)
            path = out / f'{policy}.json'
            proc = ctx.Process(
                target=_policy_process,
                args=(int(seed), policy, int(budget), data_path, str(path), kernel_id,
                      int(peek_workers)))
            proc.start()
            running[policy] = (proc, path, time.perf_counter())
            print(f'start {policy} pid={proc.pid}', flush=True)
        if not running:
            break
        time.sleep(5)
        done = []
        for policy, (proc, path, t0) in running.items():
            if proc.is_alive():
                continue
            proc.join()
            if proc.exitcode != 0:
                raise SystemExit(f'{policy} exit {proc.exitcode}')
            if not path.is_file():
                raise SystemExit(f'{policy} wrote no json')
            done.append(policy)
            finished.append(policy)
            print(f'done {policy} seconds={time.perf_counter() - t0:.1f}', flush=True)
        for policy in done:
            del running[policy]
    payload = {
        'kernel_id': kernel_id,
        'seed': int(seed),
        'budget': int(budget),
        'policies': {},
    }
    for policy in policies:
        payload['policies'][policy] = json.loads((out / f'{policy}.json').read_text())
    dump(out / 'results.json', payload)
    print(f'kernel {kernel_id} complete', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--data', default='data/enwik8/enwik8')
    parser.add_argument('--out', default='results/h35/smoke.json')
    args = parser.parse_args()
    if args.smoke:
        result = smoke(args.data)
        dump(args.out, result)
        return
    raise SystemExit('pass --smoke, or call kernel_main')


if __name__ == '__main__':
    main()
