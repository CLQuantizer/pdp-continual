"""H33: roots bank in place of the GRU, under H32b's count mixer, on 1 MB of enwik8.

One prequential pass per seed. Every byte is scored before anything learns from it.
The bank is h30y roots_learn (h30x trace rule). Mixers do not send gradients into it.
Numbers only. No verdict. Does not commit.
"""
import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import h20_world
import h26_cued as h26
import h30w_ring as h30w
import h30x_trace as h30x
import h30y_roots as h30y
import h32_mixer as h32
import h32b_mixer as h32b

N_BYTES = 1_000_000
SLICE = 100_000
CHUNK = 128
H30Y_CHUNK = 64
ALPHABET = 256
ORDERS = (1, 2, 3, 4, 5, 6)
N_ORDER = 6
BANK_EXPERTS = 7
COUNT_EXPERTS = 6
ORDER4_COLUMN = 3
WEIGHT_WINDOW = 100_000
KERNEL_LIMIT_S = 90 * 60
KERNEL_THREADS = 4
# H32b measured order-dict lookups at 2 MB running 1.75× the 50 KB rate.
COUNT_GROWTH = 1.5
# Two extra threads rarely double a small host. 4/2 with this efficiency is 1.75×.
THREAD_EFFICIENCY = 0.75
BITS_LIMIT = 1e-6
LOG2 = h30w.LOG2
LN2 = h32.LN2
ARMS = (
    'bank',
    'counts_mix6',
    'bank_mix6',
    'count_1',
    'count_2',
    'count_3',
    'count_4',
    'count_5',
    'count_6',
)
H32B_SLICE0 = {
    'gru': 2.104744,
    'mix6': 1.917176,
    'mix4': 1.951636,
    'count_3': 2.961961,
    'ppmd_order_16': 1.969704,
}
BAR_SUBSTITUTION = 0.03
BAR_SLOT = -0.05


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
    text = __import__('json').dumps(py(obj), indent=2, allow_nan=False) + '\n'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def bank_parameter_count(alphabet, n=h30w.UNITS, emb=h30w.EMB, rank=h30w.RANK):
    """Degrees of freedom in h30w.Bank. No biases on W, P1, P2, V."""
    return int(
        alphabet * emb
        + n + n
        + 2 * n * emb
        + alphabet * (2 * n)
        + 2 * rank * (2 * n)
        + alphabet * rank
    )


def gru_parameter_count():
    """h26.ByteGRU. Input size is the embedding width. GRU keeps both biases."""
    hidden = h26.HIDDEN
    emb = h26.EMB
    alphabet = h26.ALPHABET
    return int(
        alphabet * emb
        + 3 * hidden * emb
        + 3 * hidden * hidden
        + 3 * hidden
        + 3 * hidden
        + hidden * alphabet
        + alphabet
    )


def _trace_bytes(n, emb, alphabet):
    """complex64 traces: h, e^λ, E^B, z. Constant in the length of the stream."""
    return int(8 * n * (2 + emb + alphabet))


def compute_account(n_bytes=N_BYTES):
    """Real multiply-adds per byte, forward plus learning, from the shapes.

    A complex multiply is 4 real multiplies and 2 real adds. Each is counted as
    4 real multiply-adds. A complex multiply-add is 4 real multiplies and 4 real
    adds, counted the same way. A real gemm entry is 1. Linear backward is two
    gemms (gradient w.r.t. the input, gradient w.r.t. the weight), so a Linear
    costs 3 × in × out per byte. Adam is 6 real multiplies per parameter per
    step, spread over the chunk.
    """
    n = h30w.UNITS
    emb = h30w.EMB
    alphabet = ALPHABET
    rank = h30w.RANK
    cpx = 4
    params = bank_parameter_count(alphabet)
    if bank_parameter_count(41) != 107360:
        raise AssertionError(f'alphabet-41 parameter formula {bank_parameter_count(41)} != 107360')
    chunk = CHUNK
    terms = [
        ('e_lam', cpx * n, '4 N  —  e^λ ← λ e^λ + h'),
        ('e_B', cpx * n * emb, '4 N E  —  E^B ← λ E^B + x'),
        ('z_decay', cpx * n * alphabet, '4 N A  —  z ← λ z'),
        ('h', cpx * n, '4 N  —  h ← λ h + Bx'),
        ('inject', 2 * n * emb, '2 N E  —  Bx real and imag'),
        ('W', 3 * (2 * n) * alphabet, '3 (2N) A  —  readout W forward + backward'),
        ('P1', 3 * (2 * n) * rank, '3 (2N) R  —  P1 forward + backward'),
        ('P2', 3 * (2 * n) * rank, '3 (2N) R  —  P2 forward + backward'),
        ('quad', 3 * rank, '3 R  —  P1⊙P2 and both backward factors'),
        ('V', 3 * rank * alphabet, '3 R A  —  V forward + backward'),
        ('grad_nu', cpx * 2 * n, '8 N  —  conj(δ) (e^λ ∂λ/∂ν)'),
        ('grad_theta', cpx * 2 * n, '8 N  —  conj(δ) (e^λ ∂λ/∂θ)'),
        ('grad_B_re', cpx * n * emb, '4 N E  —  conj(δ) E^B'),
        ('grad_B_im', cpx * n * emb, '4 N E  —  conj(δ) (i E^B)'),
        ('grad_coeff', cpx * n * alphabet, '4 N A  —  conj(δ) z'),
        ('grad_emb', cpx * n * alphabet * emb, '4 N A E  —  einsum into the embedding'),
        ('dlam', (cpx * n) // chunk, '4 N / 128  —  ∂λ/∂ν once per chunk'),
        ('softmax', 3 * alphabet, '3 A  —  log-softmax backward'),
        ('adam', 6 * params // chunk, '6 P / 128  —  one Adam step per chunk'),
    ]
    bank_macs = 0
    for _name, value, _formula in terms:
        if value != int(value):
            raise AssertionError(f'non-integer bank term {_name}')
        bank_macs += int(value)
    hidden = h26.HIDDEN
    gemb = h26.EMB
    gru_params = gru_parameter_count()
    if gru_params != h26.PARAMS:
        raise AssertionError(f'GRU parameter formula {gru_params} != {h26.PARAMS}')
    if h26.HIDDEN != 1024 or h26.CHUNK != 128 or h26.EMB != 64:
        raise AssertionError('H26 hidden, chunk or embedding left the H32b cortex')
    replay_batch = 8
    live_gru = 9 * hidden * (gemb + hidden)
    live_readout = 3 * hidden * alphabet
    live_softmax = 3 * alphabet
    # r⊙g_n, (1−z)⊙n, z⊙h forward (3H), sigmoid/tanh backward (6H), product rule (6H).
    live_element = 15 * hidden
    live_body = live_gru + live_readout + live_softmax + live_element
    n_chunks = (int(n_bytes) + chunk - 1) // chunk
    live_adam = n_chunks * (6 * gru_params) / float(n_bytes)
    replay_steps = n_chunks * replay_batch * h26.CHUNK
    replay_body = replay_steps * live_body / float(n_bytes)
    replay_adam = n_chunks * (6 * gru_params) / float(n_bytes)
    gru_macs = live_body + live_adam + replay_body + replay_adam
    trace_bytes = _trace_bytes(n, emb, alphabet)
    window = h26.CHUNK
    # One hidden, three gates, and the embedding, float32, for the BPTT window.
    activation_bytes = window * (hidden + 3 * hidden + gemb) * 4
    return {
        'bank_parameters': params,
        'gru_parameters': gru_params,
        'bank_real_macs_per_byte': bank_macs,
        'gru_real_macs_per_byte': gru_macs,
        'gru_live_macs_per_byte': live_body + live_adam,
        'gru_replay_macs_per_byte': replay_body + replay_adam,
        'compute_ratio_bank_over_gru': bank_macs / gru_macs,
        'compute_ratio_bank_over_gru_live': bank_macs / (live_body + live_adam),
        'bank_trace_bytes': trace_bytes,
        'bank_marginal_bytes_per_history_byte': 0,
        'gru_bptt_window_bytes': window,
        'gru_replay_batch': replay_batch,
        'gru_replay_window_bytes': window,
        'gru_activation_bytes': activation_bytes,
        'gru_activation_bytes_per_history_byte': activation_bytes // window,
        'n_bytes_for_adam_average': int(n_bytes),
        'n_chunks': n_chunks,
        'bank_terms': [{'name': name, 'real_macs': int(value), 'formula': formula}
                       for name, value, formula in terms],
        'gru_terms': [
            {'name': 'live_gru_gemm', 'real_macs': live_gru,
             'formula': '9 H (E+H)  —  3 gates, input and recurrent, forward + 2 backward'},
            {'name': 'live_readout', 'real_macs': live_readout,
             'formula': '3 H A  —  Linear forward + backward'},
            {'name': 'live_softmax', 'real_macs': live_softmax, 'formula': '3 A'},
            {'name': 'live_element', 'real_macs': live_element,
             'formula': '15 H  —  three interpolations, gate backward, product rule'},
            {'name': 'live_adam', 'real_macs': live_adam,
             'formula': 'n_chunks * 6 P / n  —  one Adam step on the live chunk'},
            {'name': 'replay_body', 'real_macs': replay_body,
             'formula': 'n_chunks * 8 * 128 / n * live body  —  H32b replay batch 8, state 0'},
            {'name': 'replay_adam', 'real_macs': replay_adam,
             'formula': 'n_chunks * 6 P / n  —  one Adam step on the replay batch'},
        ],
        'formulas': {
            'complex_mul': '4 real multiply-adds',
            'linear': '3 * in * out per byte (forward, input gradient, weight gradient)',
            'adam': '6 real multiplies per parameter per Adam step',
            'bank_trace_bytes': '8 N (2 + E + A)  —  h, e^λ, E^B, z as complex64',
            'gru_window': 'h26.CHUNK = 128. Replay windows are 128 and start at state 0.',
            'gru_activation_bytes': '128 * (H + 3H + E) * 4',
            'replay': 'H32b sets h26.REPLAY_BATCH = 8. Hidden 1024 and chunk 128 are h26.',
        },
    }


class CountStore:
    """Orders 1–6 in one store. h32b.OrderCount, smoothing 0.05 from h32."""

    def __init__(self):
        self.orders = [h32b.OrderCount(k) for k in ORDERS]

    def predict(self, blob, i, dest):
        totals = []
        for ki, order in enumerate(self.orders):
            totals.append(order.predict_log(blob, i, dest[ki]))
        return totals

    def byte_bits(self, blob, i, symbol):
        out = np.empty(N_ORDER, dtype=np.float64)
        for ki, order in enumerate(self.orders):
            p = order.p_byte(blob, i, symbol)
            bit = -math.log2(p)
            if not math.isfinite(bit):
                raise FloatingPointError(f'nonfinite count_{ki + 1} at {i}')
            out[ki] = bit
        return out

    def add(self, blob, i, symbol):
        for order in self.orders:
            order.add(blob, i, symbol)


def _new_mixers():
    bank = h32b.GeoMixer(
        'bank_mix6', BANK_EXPERTS, BANK_EXPERTS, h32b._init_selector(BANK_EXPERTS, BANK_EXPERTS))
    counts_init = torch.zeros(BANK_EXPERTS, COUNT_EXPERTS)
    counts_init[:, ORDER4_COLUMN] = 1
    counts = h32b.GeoMixer('counts_mix6', BANK_EXPERTS, COUNT_EXPERTS, counts_init)
    return bank, counts


def _score_mixer(mixer, logs, selector, targets):
    report, train = h32.Mixer._nll_bits(
        None,
        mixer.W,
        torch.from_numpy(np.ascontiguousarray(logs)),
        torch.from_numpy(selector),
        torch.from_numpy(targets))
    bits = report.detach().cpu().numpy()
    if not np.isfinite(bits).all():
        raise FloatingPointError(f'nonfinite {mixer.name} bits')
    return bits, train


def _step_mixer(mixer, train):
    mixer.opt.zero_grad(set_to_none=True)
    train.mean().backward()
    mixer.opt.step()


def bank_chunk(model, tokens_t, start, end, state):
    """h30x.trace_chunk, plus the natural-log distribution it just scored."""
    holder = {}
    original = model.logits_from

    def wrapped(hidden):
        logits = original(hidden)
        holder['log_p'] = F.log_softmax(logits.detach(), dim=-1).cpu()
        return logits

    model.logits_from = wrapped
    try:
        losses, state = h30x.trace_chunk(model, tokens_t, start, end, state, mean=True)
    finally:
        model.logits_from = original
    if 'log_p' not in holder:
        raise RuntimeError('readout was not called')
    bits = losses.double().numpy() / LOG2
    if not np.isfinite(bits).all():
        raise FloatingPointError(f'nonfinite bank bits at {start}')
    log_p = holder['log_p']
    if log_p.shape != (end - start, int(model.embedding.num_embeddings)):
        raise RuntimeError(f'log_p shape {tuple(log_p.shape)}')
    return bits, log_p, state


def run_bank_only(tokens, alphabet, model_seed, chunk, label):
    """roots_learn with the h30x loop. `chunk` is the only knob (H30y uses 64)."""
    model = h30y.build_arm('roots_learn', alphabet, model_seed)
    model.train()
    if model.nu.dtype != torch.float32:
        raise RuntimeError('bank is not float32')
    params = h30w.count_params(model)
    expect = bank_parameter_count(alphabet)
    if params != expect:
        raise RuntimeError(f'parameters {params} != formula {expect}')
    opt = torch.optim.Adam(model.parameters(), lr=h30w.LR)
    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    n = int(tokens_t.shape[0])
    bits = np.empty(n, dtype=np.float64)
    state = h30x.init_state(model)
    clock = time.perf_counter()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_bits, _log_p, state = bank_chunk(model, tokens_t, start, end, state)
        bits[start:end] = chunk_bits
        nn.utils.clip_grad_norm_(model.parameters(), h30w.CLIP)
        opt.step()
    seconds = time.perf_counter() - clock
    del model
    print(
        f'{label} n={n} chunk={chunk} bits={bits.mean():.6f} seconds={seconds:.3f}',
        flush=True)
    return bits, seconds, params


def run_stream(raw, model_seed, chunk=CHUNK, keep_bits=False, profile=False, log_every=100_000):
    """One pass: bank, bank_mix6, counts_mix6, and count_1..count_6."""
    raw = np.asarray(raw, dtype=np.uint8)
    n = int(raw.shape[0])
    if n < 1:
        raise RuntimeError('empty stream')
    if int(raw.max()) > 255:
        raise RuntimeError('byte outside 0..255')
    blob = raw.tobytes()
    tokens = raw.astype(np.int64, copy=False)
    model = h30y.build_arm('roots_learn', ALPHABET, model_seed)
    model.train()
    params = h30w.count_params(model)
    if params != bank_parameter_count(ALPHABET):
        raise RuntimeError(f'bank parameters {params}')
    with torch.no_grad():
        mag = model.lam().abs()
        init_abs_min = float(mag.min())
        init_abs_max = float(mag.max())
    if abs(init_abs_min - 0.98) > 1e-5 or abs(init_abs_max - 0.98) > 1e-5:
        raise RuntimeError(f'|λ| init {init_abs_min} {init_abs_max}')
    opt = torch.optim.Adam(model.parameters(), lr=h30w.LR)
    tokens_t = torch.as_tensor(tokens, dtype=torch.long)
    state = h30x.init_state(model)
    bank_mix, counts_mix = _new_mixers()
    bank_row0 = torch.zeros(BANK_EXPERTS, dtype=bank_mix.W.dtype)
    bank_row0[0] = 1
    if not torch.equal(bank_mix.W.detach().cpu()[0], bank_row0):
        raise RuntimeError('bank_mix6 init is not one-hot on the bank')
    if float(counts_mix.W.detach()[0, ORDER4_COLUMN]) != 1.0 or float(counts_mix.W.detach()[0].sum()) != 1.0:
        raise RuntimeError('counts_mix6 init is not one-hot on order 4')
    store = CountStore()
    n_slices = (n + SLICE - 1) // SLICE
    sums = {arm: np.zeros(n_slices, dtype=np.float64) for arm in ARMS}
    kept = {arm: [] for arm in ARMS} if keep_bits else None
    weight_start = n - WEIGHT_WINDOW if n >= WEIGHT_WINDOW else None
    weight_sum = {
        'bank_mix6': np.zeros(BANK_EXPERTS, dtype=np.float64),
        'counts_mix6': np.zeros(COUNT_EXPERTS, dtype=np.float64),
    }
    weight_n = 0
    set_counts = np.zeros(BANK_EXPERTS, dtype=np.int64)
    bank_seconds = 0.0
    mixer_seconds = 0.0
    chunk_times = []
    clock = time.perf_counter()
    pos = 0
    while pos < n:
        start = pos
        end = min(start + chunk, n)
        t_chunk = time.perf_counter()
        t0 = time.perf_counter()
        bank_bits, log_p, state = bank_chunk(model, tokens_t, start, end, state)
        bank_seconds += time.perf_counter() - t0
        t1 = time.perf_counter()
        tlen = end - start
        targets = tokens[start:end]
        logs = np.empty((tlen, BANK_EXPERTS, ALPHABET), dtype=np.float32)
        logs[:, 0, :] = log_p.numpy()
        taken = -log_p[torch.arange(tlen), torch.from_numpy(targets)].double().numpy() / LN2
        if float(np.max(np.abs(taken - bank_bits))) > 1e-3:
            raise RuntimeError(
                f'bank log p left the cross-entropy bits at {start} '
                f'by {float(np.max(np.abs(taken - bank_bits))):.3e}')
        dest = np.empty((N_ORDER, ALPHABET), dtype=np.float32)
        selector = np.empty(tlen, dtype=np.int64)
        count_bits = np.empty((tlen, N_ORDER), dtype=np.float64)
        # Mixer counts are the store from before this chunk, as in H32b.
        # Count-arm bits are then taken byte by byte, each add after that byte.
        for t in range(tlen):
            totals = store.predict(blob, start + t, dest)
            logs[t, 1:, :] = dest
            selector[t] = h32b.weight_set_max(totals, N_ORDER)
        set_counts += np.bincount(selector, minlength=BANK_EXPERTS)
        mix_bits, mix_train = _score_mixer(bank_mix, logs, selector, targets)
        count_mix_bits, count_mix_train = _score_mixer(
            counts_mix, logs[:, 1:, :], selector, targets)
        if weight_start is not None and end > weight_start:
            scored_bank_w = bank_mix.W.detach().double().cpu().numpy().copy()
            scored_counts_w = counts_mix.W.detach().double().cpu().numpy().copy()
            for t in range(tlen):
                if start + t < weight_start:
                    continue
                row = int(selector[t])
                weight_sum['bank_mix6'] += scored_bank_w[row]
                weight_sum['counts_mix6'] += scored_counts_w[row]
                weight_n += 1
        _step_mixer(bank_mix, mix_train)
        _step_mixer(counts_mix, count_mix_train)
        nn.utils.clip_grad_norm_(model.parameters(), h30w.CLIP)
        opt.step()
        for t in range(tlen):
            i = start + t
            count_bits[t] = store.byte_bits(blob, i, int(targets[t]))
            store.add(blob, i, int(targets[t]))
        mixer_seconds += time.perf_counter() - t1
        chunk_map = {
            'bank': bank_bits,
            'bank_mix6': mix_bits,
            'counts_mix6': count_mix_bits,
        }
        for k in range(N_ORDER):
            chunk_map[f'count_{k + 1}'] = count_bits[:, k]
        for arm, values in chunk_map.items():
            _deposit(sums[arm], start, values)
            if kept is not None:
                kept[arm].append(np.asarray(values, dtype=np.float64).copy())
        pos = end
        if profile:
            chunk_times.append(time.perf_counter() - t_chunk)
        if end // log_every > start // log_every or end == n:
            done = {arm: float(sums[arm].sum()) / end for arm in ('bank', 'bank_mix6', 'counts_mix6')}
            print(
                f'H33 seed={model_seed} n={end}/{n} bank={done["bank"]:.6f} '
                f'bank_mix6={done["bank_mix6"]:.6f} counts_mix6={done["counts_mix6"]:.6f} '
                f'seconds={time.perf_counter() - clock:.1f}',
                flush=True)
    if int(set_counts.sum()) != n:
        raise RuntimeError(f'weight-set counts {int(set_counts.sum())} != {n}')
    if n >= WEIGHT_WINDOW and weight_n != WEIGHT_WINDOW:
        raise RuntimeError(f'weight window {weight_n} != {WEIGHT_WINDOW}')
    wall = time.perf_counter() - clock
    bits = None
    if kept is not None:
        bits = {arm: np.concatenate(kept[arm]) for arm in ARMS}
        for arm in ARMS:
            if bits[arm].shape != (n,):
                raise RuntimeError(f'{arm} bits length {bits[arm].shape}')
            if abs(float(bits[arm].sum()) - float(sums[arm].sum())) > 1e-6:
                raise RuntimeError(f'{arm} slice sum left the byte sum')
    return _pack_run(
        sums, n, model_seed, params, init_abs_min, init_abs_max, wall,
        bank_seconds, mixer_seconds, weight_sum, weight_n, set_counts, bits,
        chunk_times if profile else None)


def _deposit(sums, start, values):
    values = np.asarray(values, dtype=np.float64)
    pos = 0
    n = int(values.shape[0])
    while pos < n:
        idx = (start + pos) // SLICE
        room = SLICE - ((start + pos) % SLICE)
        take = min(room, n - pos)
        sums[idx] += float(values[pos:pos + take].sum())
        pos += take


def _pack_run(
        sums, n, model_seed, params, init_abs_min, init_abs_max, wall,
        bank_seconds, mixer_seconds, weight_sum, weight_n, set_counts, bits,
        chunk_times):
    arms = {}
    finite = True
    for arm in ARMS:
        total = float(sums[arm].sum())
        slices = []
        for i in range((n + SLICE - 1) // SLICE):
            a = i * SLICE
            b = min(a + SLICE, n)
            sl = float(sums[arm][i])
            slices.append({
                'index': i,
                'start': a,
                'end': b,
                'bytes': b - a,
                'bits': sl,
                'bits_per_byte': sl / (b - a),
            })
            if not math.isfinite(sl):
                finite = False
        if not math.isfinite(total):
            finite = False
        row = {
            'name': arm,
            'bits': total,
            'bits_per_byte': total / n,
            'slices': slices,
            'finite': bool(math.isfinite(total) and all(math.isfinite(s['bits']) for s in slices)),
        }
        arms[arm] = row
    out = {
        'seed': int(model_seed),
        'n_bytes': int(n),
        'chunk': CHUNK,
        'parameters': int(params),
        'init_abs_lam_min': init_abs_min,
        'init_abs_lam_max': init_abs_max,
        'wall_seconds': float(wall),
        'bank_seconds': float(bank_seconds),
        'mixer_seconds': float(mixer_seconds),
        'arms': arms,
        'finite': bool(finite),
        'weight_set_counts': [int(v) for v in set_counts],
        'weights': _weight_payload(weight_sum, weight_n),
    }
    if bits is not None:
        out['bits'] = bits
    if chunk_times is not None:
        out['chunk_times'] = chunk_times
    return out


BANK_MIX_NAMES = ('bank', 'count_1', 'count_2', 'count_3', 'count_4', 'count_5', 'count_6')
COUNT_MIX_NAMES = ('count_1', 'count_2', 'count_3', 'count_4', 'count_5', 'count_6')


def _weight_payload(weight_sum, weight_n):
    if weight_n == 0:
        return None
    return {
        'n_bytes': int(weight_n),
        'bank_mix6': {
            'experts': list(BANK_MIX_NAMES),
            'mean': [float(v) for v in (weight_sum['bank_mix6'] / weight_n)],
        },
        'counts_mix6': {
            'experts': list(COUNT_MIX_NAMES),
            'mean': [float(v) for v in (weight_sum['counts_mix6'] / weight_n)],
        },
    }


def load_checked(path):
    raw = h26.load_enwik8(path)
    digest = h26.sha256_arr(raw[:h26.E10M])
    if digest != h26.E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest} != {h26.E10M_SHA256}')
    return raw, digest


def count_reference(raw):
    """Sequential count_1..count_4 from h32.CountBank. Returns four bit sums."""
    raw = np.asarray(raw, dtype=np.uint8)
    blob = raw.tobytes()
    bank = h32.CountBank()
    totals = np.zeros(4, dtype=np.float64)
    for i, symbol in enumerate(blob):
        symbol = int(symbol)
        for k, order in enumerate(bank.orders):
            totals[k] += -math.log2(order.p_byte(blob, i, symbol))
        bank.add(blob, i, symbol)
    return totals


def configure(n_threads):
    torch.set_num_threads(int(n_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def smoke_bank_match():
    if h30w.CHUNK != H30Y_CHUNK:
        raise RuntimeError(f'h30w.CHUNK is {h30w.CHUNK}, H30y scored with 64')
    text, _labels = h20_world.stream('slow', 6, 4096)
    tokens = h30w.encode_text(text)
    if len(tokens) != 4096:
        raise RuntimeError(f'H20 smoke length {len(tokens)}')
    alphabet = len(h20_world.ALPHA)
    if alphabet != 41:
        raise RuntimeError(f'H20 alphabet {alphabet}')
    ref = h30y.run_arm(
        'roots_learn', alphabet, tokens, 0, 'h30y_roots_learn_h20_s6', progress_every=10_000)
    mine, seconds, params = run_bank_only(
        tokens, alphabet, 0, H30Y_CHUNK, 'h33_bank_h20_s6')
    delta = _max_abs(ref['bits'], mine)
    return {
        'ok': bool(np.isfinite(delta) and delta <= BITS_LIMIT),
        'world': 'h20_slow',
        'world_seed': 6,
        'model_seed': 0,
        'alphabet': alphabet,
        'n': 4096,
        'chunk': H30Y_CHUNK,
        'parameters': int(params),
        'h30y_bits_per_symbol': float(ref['bits_per_symbol']),
        'h33_bits_per_symbol': float(mine.mean()),
        'max_abs_delta': delta,
        'seconds': float(seconds),
    }


def count_store_totals(raw, n_orders=4):
    """Same CountStore the stream uses, one byte at a time."""
    raw = np.asarray(raw, dtype=np.uint8)
    blob = raw.tobytes()
    store = CountStore()
    totals = np.zeros(n_orders, dtype=np.float64)
    for i, symbol in enumerate(blob):
        symbol = int(symbol)
        bits = store.byte_bits(blob, i, symbol)
        totals += bits[:n_orders]
        store.add(blob, i, symbol)
    return totals


def smoke_counts(raw):
    n = 20_000
    ref = count_reference(raw[:n])
    mine = count_store_totals(raw[:n])
    deltas = {}
    ok = True
    for k in range(4):
        arm = f'count_{k + 1}'
        delta = abs(float(mine[k]) - float(ref[k]))
        deltas[arm] = {
            'h33_bits': float(mine[k]),
            'h32_count_bank_bits': float(ref[k]),
            'abs_delta': delta,
        }
        if delta > BITS_LIMIT:
            ok = False
    return {'ok': ok, 'n': n, 'orders': deltas}


def smoke_prequential(raw):
    short = run_stream(raw[:1024], 0, keep_bits=True, log_every=1024)
    long = run_stream(raw[:2048], 0, keep_bits=True, log_every=2048)
    rows = {}
    ok = True
    for arm in ARMS:
        a = short['bits'][arm]
        b = long['bits'][arm][:1024]
        identical = bool(np.array_equal(a, b))
        rows[arm] = {'identical': identical, 'max_abs_delta': _max_abs(a, b)}
        ok = ok and identical
    return {
        'ok': ok,
        'n_short': 1024,
        'n_long': 2048,
        'seed': 0,
        'arms': rows,
        'short_seconds': short['wall_seconds'],
        'long_seconds': long['wall_seconds'],
    }


def _kernel_estimate(timed):
    n = int(timed['n_bytes'])
    threads = int(torch.get_num_threads())
    if threads < 1:
        raise RuntimeError('no torch threads')
    speedup = 1.0 + THREAD_EFFICIENCY * (KERNEL_THREADS / threads - 1.0)
    if speedup < 1.0:
        speedup = 1.0
    grown = float(timed['bank_seconds']) + COUNT_GROWTH * float(timed['mixer_seconds'])
    vps_linear = grown * (N_BYTES / n)
    times = np.asarray(timed['chunk_times'], dtype=np.float64)
    late = float(times[-(len(times) // 4):].mean())
    mean = float(times.mean())
    rate_ratio = late / mean if mean > 0 else 1.0
    kernel_est = vps_linear / speedup * max(1.0, rate_ratio)
    return {
        'timed_bytes': n,
        'wall_seconds': float(timed['wall_seconds']),
        'bank_seconds': float(timed['bank_seconds']),
        'mixer_seconds': float(timed['mixer_seconds']),
        'vps_threads': threads,
        'kernel_threads': KERNEL_THREADS,
        'thread_efficiency': THREAD_EFFICIENCY,
        'thread_speedup': speedup,
        'count_growth': COUNT_GROWTH,
        'late_over_mean_chunk_time': rate_ratio,
        'vps_extrapolation_seconds': vps_linear,
        'kernel_estimate_seconds': kernel_est,
        'kernel_limit_seconds': KERNEL_LIMIT_S,
        'fits': bool(kernel_est <= KERNEL_LIMIT_S),
    }


def smoke_time(raw):
    n = 32_768
    got = run_stream(raw[:n], 0, keep_bits=True, profile=True, log_every=8192)
    finite_8192 = all(bool(np.isfinite(got['bits'][arm][:8192]).all()) for arm in ARMS)
    finite_all = bool(got['finite']) and all(bool(np.isfinite(got['bits'][arm]).all()) for arm in ARMS)
    ref = count_reference(raw[:20_000])
    prefix = {}
    prefix_ok = True
    for k in range(4):
        arm = f'count_{k + 1}'
        total = float(got['bits'][arm][:20_000].sum())
        delta = abs(total - float(ref[k]))
        prefix[arm] = delta
        if delta > BITS_LIMIT:
            prefix_ok = False
    estimate = _kernel_estimate(got)
    return {
        'ok': bool(finite_8192 and finite_all and prefix_ok and estimate['fits']),
        'finite_8192': finite_8192,
        'finite_32768': finite_all,
        'count_prefix_20000_abs_delta': prefix,
        'count_prefix_ok': prefix_ok,
        'bits_per_byte_32768': {arm: got['arms'][arm]['bits_per_byte'] for arm in ARMS},
        'estimate': estimate,
    }


def smoke(data_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure(int(os.environ.get('OMP_NUM_THREADS', '2')))
    account = compute_account(N_BYTES)
    print(
        f'H33 smoke torch={torch.__version__} threads={torch.get_num_threads()} '
        f'bank_params={account["bank_parameters"]} gru_params={account["gru_parameters"]} '
        f'bank_macs={account["bank_real_macs_per_byte"]} '
        f'gru_macs={account["gru_real_macs_per_byte"]:.3f}',
        flush=True)
    payload = {
        'experiment': 'h33',
        'torch': torch.__version__,
        'threads': int(torch.get_num_threads()),
        'compute': account,
        'kernels_pushed': False,
    }
    checks = (
        ('bank_match', smoke_bank_match),
        ('counts', None),
        ('prequential', None),
        ('time', None),
    )
    raw = None
    digest = None
    reason = None
    for name, fn in checks:
        if name == 'bank_match':
            row = fn()
        else:
            if raw is None:
                raw, digest = load_checked(data_path)
                payload['sha256_e10m'] = digest
                payload['data_path'] = str(data_path)
                print(f'H33 enwik8 sha256_e10m={digest}', flush=True)
            if name == 'counts':
                row = smoke_counts(raw)
            elif name == 'prequential':
                row = smoke_prequential(raw)
            else:
                row = smoke_time(raw)
        payload[name] = row
        dump(out_dir / 'smoke.json', payload)
        print(f'H33 smoke {name} ok={row["ok"]}', flush=True)
        if not row['ok']:
            reason = name
            break
    payload['passed'] = reason is None
    payload['stop'] = reason
    dump(out_dir / 'smoke.json', payload)
    if reason is not None:
        write_stop(out_dir, payload)
        print(f'SMOKE_FAIL {reason}. Kernels not pushed.', flush=True)
        raise SystemExit(2)
    print('smoke_ok', flush=True)
    return payload


def _fmt(value, digits=6):
    if value is None:
        return 'na'
    return f'{value:.{digits}f}'


def _slice_table(rows, bits_key='bits_per_byte'):
    lines = [
        '| Slice | Start | End | Bits | Bits/byte |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | {row['start']} | {row['end']} | "
            f"{row['bits']:.3f} | {_fmt(row['bits_per_byte'])} |")
    return lines


def _weight_lines(weights):
    lines = [
        '| Mixer | Expert | Mean weight |',
        '|---|---|---:|',
    ]
    for mixer in ('bank_mix6', 'counts_mix6'):
        block = weights[mixer]
        for name, value in zip(block['experts'], block['mean']):
            lines.append(f'| {mixer} | {name} | {_fmt(value)} |')
    lines.append(f"| bytes |  | {weights['n_bytes']} |")
    return lines


def write_stop(out_dir, smoke_payload):
    out_dir = Path(out_dir)
    estimate = (smoke_payload.get('time') or {}).get('estimate') or {}
    lines = [
        '# H33 bank under the count mixer',
        '',
        'Scored kernels were not run.',
        f"stop: {smoke_payload.get('stop')}.",
        f"kernel_estimate_seconds: {_fmt(estimate.get('kernel_estimate_seconds'), 3)}.",
        f"kernel_limit_seconds: {KERNEL_LIMIT_S}.",
        '',
    ]
    text = '\n'.join(lines)
    (out_dir / 'RESULTS.md').write_text(text)
    dump(out_dir / 'results.json', {
        'experiment': 'h33',
        'kernels_pushed': False,
        'stop': smoke_payload.get('stop'),
        'smoke': smoke_payload,
    })


def _mean(a, b):
    return 0.5 * (float(a) + float(b))


def _arm_bpb(run, arm):
    return float(run['arms'][arm]['bits_per_byte'])


def _gaps(run):
    bank_mix = _arm_bpb(run, 'bank_mix6')
    counts_mix = _arm_bpb(run, 'counts_mix6')
    bank = _arm_bpb(run, 'bank')
    return {
        'bank_mix6_minus_h32b_mix6': bank_mix - H32B_SLICE0['mix6'],
        'bank_mix6_minus_counts_mix6': bank_mix - counts_mix,
        'bank_minus_h32b_gru': bank - H32B_SLICE0['gru'],
    }


def assemble(seed_payloads, smoke_payload, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_seed = {}
    for payload in seed_payloads:
        seed = int(payload['seed'])
        if seed in by_seed:
            raise RuntimeError(f'duplicate seed {seed}')
        by_seed[seed] = payload
    if set(by_seed) != {0, 1}:
        raise RuntimeError(f'seeds {sorted(by_seed)}')
    runs = {seed: by_seed[seed]['run'] for seed in (0, 1)}
    for seed, run in runs.items():
        if int(run['n_bytes']) != N_BYTES:
            raise RuntimeError(f'seed {seed} n_bytes {run["n_bytes"]}')
        if not run['finite']:
            raise RuntimeError(f'seed {seed} is nonfinite')
        if run['weights'] is None or int(run['weights']['n_bytes']) != WEIGHT_WINDOW:
            raise RuntimeError(f'seed {seed} weight window')
    count_delta = {}
    for arm in ('counts_mix6',) + tuple(f'count_{k}' for k in range(1, 7)):
        count_delta[arm] = abs(runs[0]['arms'][arm]['bits'] - runs[1]['arms'][arm]['bits'])
    gaps = {seed: _gaps(runs[seed]) for seed in (0, 1)}
    gap_mean = {key: _mean(gaps[0][key], gaps[1][key]) for key in gaps[0]}
    compute = by_seed[0]['compute']
    result = {
        'experiment': 'h33',
        'n_bytes': N_BYTES,
        'sha256_e10m': by_seed[0]['sha256_e10m'],
        'chunk': CHUNK,
        'kernels_pushed': True,
        'kernels': {str(seed): by_seed[seed]['kernel_id'] for seed in (0, 1)},
        'wall_seconds': {str(seed): by_seed[seed]['session_wall_seconds'] for seed in (0, 1)},
        'h32b_slice0': H32B_SLICE0,
        'bars': {'substitution': BAR_SUBSTITUTION, 'slot': BAR_SLOT},
        'bits_per_byte': {
            arm: {
                'seed_0': _arm_bpb(runs[0], arm),
                'seed_1': _arm_bpb(runs[1], arm),
                'mean': _mean(_arm_bpb(runs[0], arm), _arm_bpb(runs[1], arm)),
            }
            for arm in ARMS
        },
        'bits': {
            arm: {
                'seed_0': runs[0]['arms'][arm]['bits'],
                'seed_1': runs[1]['arms'][arm]['bits'],
            }
            for arm in ARMS
        },
        'slices': {
            str(seed): {arm: runs[seed]['arms'][arm]['slices'] for arm in ARMS}
            for seed in (0, 1)
        },
        'gaps': {'seed_0': gaps[0], 'seed_1': gaps[1], 'mean': gap_mean},
        'count_seed_abs_bits': count_delta,
        'weights': {str(seed): runs[seed]['weights'] for seed in (0, 1)},
        'weight_set_counts': {
            str(seed): runs[seed]['weight_set_counts'] for seed in (0, 1)
        },
        'compute': compute,
        'smoke': smoke_payload,
        'init_abs_lam': {
            str(seed): {
                'min': runs[seed]['init_abs_lam_min'],
                'max': runs[seed]['init_abs_lam_max'],
            }
            for seed in (0, 1)
        },
    }
    dump(out_dir / 'results.json', result)
    (out_dir / 'RESULTS.md').write_text(render(result))
    return result


def render(result):
    bpb = result['bits_per_byte']
    lines = [
        '# H33 bank under the count mixer on the first 1 MB of enwik8',
        '',
        f"Bytes: {result['n_bytes']}. sha256 of the first 10 MB: `{result['sha256_e10m']}`.",
        'Each byte is scored before any arm learns from it. Chunk 128.',
        'Bank: roots_learn, N = 256, θ_k = 2πk/256, |λ| = 0.98, order-2 readout,',
        'h30x trace rule, Adam lr 0.001, clip 1.0, float32, embedding 32.',
        'bank_mix6: H32b mix6 with the bank in the cortex slot. Init 1 on the bank.',
        'counts_mix6: orders 1–6 only. Init 1 on order 4. Weight sets are H32b\'s,',
        'highest order k with N_k ≥ 2, else 0.',
        'Smoothing 0.05. Mixer Adam lr 0.003. One count store. Counts of a chunk are',
        'added after both mixers have scored that chunk and stepped. Mixer gradients',
        'do not enter the bank. Count arms do not use the model seed.',
        '',
        '## Bits/byte',
        '',
        '| Arm | Seed 0 | Seed 1 | Mean |',
        '|---|---:|---:|---:|',
    ]
    for arm in ARMS:
        row = bpb[arm]
        if arm in ('bank', 'bank_mix6'):
            lines.append(
                f"| {arm} | {_fmt(row['seed_0'])} | {_fmt(row['seed_1'])} | {_fmt(row['mean'])} |")
        else:
            lines.append(f"| {arm} | {_fmt(row['seed_0'])} |  |  |")
    lines += [
        '',
        'Count-arm absolute bit difference between seed 0 and seed 1:',
        '',
        '| Arm | Abs bits |',
        '|---|---:|',
    ]
    for arm, delta in result['count_seed_abs_bits'].items():
        lines.append(f'| {arm} | {delta:.8e} |')
    lines += ['', '## Per 100 KB slice', '']
    for arm in ('bank', 'bank_mix6'):
        for seed in ('0', '1'):
            lines.append(f'### {arm} seed {seed}')
            lines.append('')
            lines.extend(_slice_table(result['slices'][seed][arm]))
            lines.append('')
    for arm in ('counts_mix6',) + tuple(f'count_{k}' for k in range(1, 7)):
        lines.append(f'### {arm}')
        lines.append('')
        lines.extend(_slice_table(result['slices']['0'][arm]))
        lines.append('')
    ref = result['h32b_slice0']
    lines += [
        '## H32b slice 0',
        '',
        '| Arm | Bits/byte |',
        '|---|---:|',
        f"| gru | {_fmt(ref['gru'])} |",
        f"| mix6 | {_fmt(ref['mix6'])} |",
        f"| mix4 | {_fmt(ref['mix4'])} |",
        f"| count_3 | {_fmt(ref['count_3'])} |",
        f"| ppmd_order_16 | {_fmt(ref['ppmd_order_16'])} |",
        '',
        '## Gaps',
        '',
        '| Gap | Seed 0 | Seed 1 | Mean |',
        '|---|---:|---:|---:|',
    ]
    for key in (
        'bank_mix6_minus_h32b_mix6',
        'bank_mix6_minus_counts_mix6',
        'bank_minus_h32b_gru',
    ):
        lines.append(
            f"| {key} | {_fmt(result['gaps']['seed_0'][key])} | "
            f"{_fmt(result['gaps']['seed_1'][key])} | {_fmt(result['gaps']['mean'][key])} |")
    lines += [
        '',
        f"substitution threshold: {BAR_SUBSTITUTION}. slot threshold: {BAR_SLOT}.",
        '',
        '## Mean selected weights, last 100 KB',
        '',
    ]
    for seed in ('0', '1'):
        lines.append(f'### seed {seed}')
        lines.append('')
        lines.extend(_weight_lines(result['weights'][seed]))
        lines.append('')
    comp = result['compute']
    lines += [
        '## Compute',
        '',
        'Real multiply-adds. A complex multiply counts as 4. A Linear counts as',
        '3 × in × out per byte (forward, input gradient, weight gradient). Adam counts',
        'as 6 real multiplies per parameter per step. GRU replay is the H32b uniform',
        'cortex: h26 hidden 1024, chunk 128, replay batch 8, each window from state 0.',
        '',
        '| Cortex | Parameters | Real MACs/byte | History state | State bytes | Per history byte |',
        '|---|---:|---:|---|---:|---:|',
        f"| bank | {comp['bank_parameters']} | {comp['bank_real_macs_per_byte']} | "
        f"traces h, e^λ, E^B, z | {comp['bank_trace_bytes']} | "
        f"{comp['bank_marginal_bytes_per_history_byte']} |",
        f"| gru | {comp['gru_parameters']} | {_fmt(comp['gru_real_macs_per_byte'], 3)} | "
        f"BPTT window {comp['gru_bptt_window_bytes']} | {comp['gru_activation_bytes']} | "
        f"{comp['gru_activation_bytes_per_history_byte']} |",
        '',
        f"compute_ratio bank/gru = {_fmt(comp['compute_ratio_bank_over_gru'], 6)}.",
        f"compute_ratio bank/gru_live = {_fmt(comp['compute_ratio_bank_over_gru_live'], 6)}.",
        f"gru live MACs/byte = {_fmt(comp['gru_live_macs_per_byte'], 3)}.",
        f"gru replay MACs/byte = {_fmt(comp['gru_replay_macs_per_byte'], 3)}.",
        '',
        'Bank terms, real MACs/byte:',
        '',
        '| Term | Real MACs | Formula |',
        '|---|---:|---|',
    ]
    for term in comp['bank_terms']:
        lines.append(f"| {term['name']} | {term['real_macs']} | {term['formula']} |")
    lines += [
        '',
        'GRU terms, real MACs/byte, averaged over the 1 MB stream:',
        '',
        '| Term | Real MACs | Formula |',
        '|---|---:|---|',
    ]
    for term in comp['gru_terms']:
        lines.append(f"| {term['name']} | {_fmt(term['real_macs'], 3)} | {term['formula']} |")
    lines += [
        '',
        f"Bank trace bytes = {comp['formulas']['bank_trace_bytes']}.",
        f"GRU activations = {comp['formulas']['gru_activation_bytes']}.",
        f"{comp['formulas']['gru_window']} {comp['formulas']['replay']}",
        '',
        '## Smoke',
        '',
    ]
    smoke = result.get('smoke') or {}
    lines.extend(_smoke_lines(smoke))
    lines += [
        '',
        '## Kernels',
        '',
        '| Seed | Kernel | Wall seconds |',
        '|---:|---|---:|',
    ]
    for seed in ('0', '1'):
        lines.append(
            f"| {seed} | {result['kernels'][seed]} | {_fmt(result['wall_seconds'][seed], 3)} |")
    lines.append('')
    return '\n'.join(lines)


def _smoke_lines(smoke):
    if not smoke:
        return ['Smoke payload was not attached.']
    lines = [f"threads: {smoke.get('threads')}. torch: {smoke.get('torch')}.", '']
    match = smoke.get('bank_match') or {}
    if match:
        lines.append(
            f"bank match H20 slow seed 6, model seed 0, alphabet 41, n=4096, chunk 64: "
            f"h30y {_fmt(match.get('h30y_bits_per_symbol'))} h33 {_fmt(match.get('h33_bits_per_symbol'))} "
            f"max_abs_delta {match.get('max_abs_delta'):.3e} seconds {_fmt(match.get('seconds'), 3)} "
            f"ok {match.get('ok')}."
        )
    counts = smoke.get('counts') or {}
    if counts:
        lines.append(f"count_1..count_4 vs h32.CountBank on {counts.get('n')} bytes:")
        for arm, row in (counts.get('orders') or {}).items():
            lines.append(
                f"- {arm}: h33 {row['h33_bits']:.6f} h32 {row['h32_count_bank_bits']:.6f} "
                f"abs_delta {row['abs_delta']:.3e}")
    preq = smoke.get('prequential') or {}
    if preq:
        lines.append(
            f"prequential first 1024 of a 1024-byte run vs a 2048-byte run, seed 0, "
            f"ok {preq.get('ok')}.")
        for arm, row in (preq.get('arms') or {}).items():
            lines.append(
                f"- {arm}: identical {row['identical']} max_abs_delta {row['max_abs_delta']:.3e}")
    timed = smoke.get('time') or {}
    if timed:
        est = timed.get('estimate') or {}
        lines.append(
            f"finite first 8192: {timed.get('finite_8192')}. "
            f"finite 32768: {timed.get('finite_32768')}.")
        lines.append(
            f"32768 wall seconds {_fmt(est.get('wall_seconds'), 3)}, "
            f"bank {_fmt(est.get('bank_seconds'), 3)}, "
            f"mixer {_fmt(est.get('mixer_seconds'), 3)}.")
        lines.append(
            f"kernel estimate seconds {_fmt(est.get('kernel_estimate_seconds'), 3)} "
            f"(vps threads {est.get('vps_threads')}, speedup {_fmt(est.get('thread_speedup'), 3)}, "
            f"count growth {est.get('count_growth')}, "
            f"late/mean chunk {_fmt(est.get('late_over_mean_chunk_time'), 3)}, "
            f"limit {est.get('kernel_limit_seconds')}, fits {est.get('fits')}).")
    return lines


def kernel_main(out, seed, kernel_id, data_path=None, n_bytes=N_BYTES):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(seed)
    if seed not in (0, 1):
        raise SystemExit(f'seed {seed} is outside 0, 1')
    configure(int(os.environ.get('OMP_NUM_THREADS', str(KERNEL_THREADS))))
    if data_path is None:
        data_path = h26.find_kaggle_enwik8()
    raw, digest = load_checked(data_path)
    print(
        f'H33 kernel {kernel_id} seed={seed} threads={torch.get_num_threads()} '
        f'n={n_bytes} sha256_e10m={digest}',
        flush=True)
    clock = time.monotonic()
    got = run_stream(raw[:n_bytes], seed, log_every=100_000)
    session = time.monotonic() - clock
    if 'bits' in got:
        del got['bits']
    if not got['finite']:
        raise SystemExit('nonfinite scored run')
    payload = {
        'experiment': 'h33',
        'kernel_id': kernel_id,
        'seed': seed,
        'n_bytes': int(n_bytes),
        'sha256_e10m': digest,
        'torch': torch.__version__,
        'numpy': np.__version__,
        'threads': int(torch.get_num_threads()),
        'session_wall_seconds': float(session),
        'compute': compute_account(n_bytes),
        'run': got,
    }
    dump(out / 'results.json', payload)
    print(
        'KERNEL_DONE ' + __import__('json').dumps({
            'kernel_id': kernel_id,
            'seed': seed,
            'bank': got['arms']['bank']['bits_per_byte'],
            'bank_mix6': got['arms']['bank_mix6']['bits_per_byte'],
            'counts_mix6': got['arms']['counts_mix6']['bits_per_byte'],
            'session_wall_seconds': session,
            'finite': got['finite'],
        }),
        flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--kernel', action='store_true')
    parser.add_argument('--assemble', nargs=2, metavar=('SEED0_JSON', 'SEED1_JSON'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--kernel-id', default='')
    parser.add_argument('--data', default='data/enwik8/enwik8')
    parser.add_argument('--out', default='results/h33')
    parser.add_argument('--smoke-json', default='results/h33/smoke.json')
    parser.add_argument('--n', type=int, default=N_BYTES)
    args = parser.parse_args()
    if args.smoke:
        smoke(args.data, args.out)
        return
    if args.kernel:
        kernel_id = args.kernel_id or f'ezzzzzx/h33-bank-text-s{args.seed}'
        kernel_main(args.out, args.seed, kernel_id, data_path=args.data, n_bytes=args.n)
        return
    if args.assemble:
        import json
        payloads = [json.loads(Path(path).read_text()) for path in args.assemble]
        smoke_payload = json.loads(Path(args.smoke_json).read_text())
        assemble(payloads, smoke_payload, args.out)
        return
    raise SystemExit('pass --smoke, --kernel, or --assemble')


if __name__ == '__main__':
    main()

