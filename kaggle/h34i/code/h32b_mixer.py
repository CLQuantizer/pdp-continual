"""H32b: H32's mixer on E10M, plus a cortex-only temperature and orders 5–6.

The cortex is one H32 pass (H26 arm uniform). Its loss is the only gradient
that reaches it. One count store of orders 1–6 feeds every arm. Each mixer
reads counts from before the chunk, takes one Adam step, and the chunk is
added after every mixer has scored it.

Numbers only. No verdict. Does not commit.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h26_cued as h26
import h32_mixer as h32

ALPHABET = h32.ALPHABET
SMOOTH = h32.SMOOTH
DENOM_EXTRA = h32.DENOM_EXTRA
MIX_LR = h32.MIX_LR
LN2 = h32.LN2
E10M = h26.E10M
E3M = h26.E3M
SLICE = h26.SLICE
CHUNK = h26.CHUNK
PPMD_ORDER = 16
PPMD_MEM = 1 << 30
ORDER56_LIMIT = 24 * 10**9
# H29d table prints 1.831624. Seed 0 r8_3e4 is the H29c run.
H29D_R8_SEED0_BITS = 18316242.855636712
H29D_R8_SEED0_BPB = 1.8316242855636713
H29D_R8_SEED0_TABLE_BPB = 1.831624
H29D_PPMD16_BPB = 1.837254
# H29c r8_3e4 E10M wall on a T4. Same cortex as this run.
CORTEX_GPU_E10M_SECONDS = 3783.846
# H32 E3M stream wall on a T4, orders 1–4 and one mixer included.
H32_E3M_STREAM_SECONDS = 1317.900
GPU_HOUR_BUDGET = 2.2
# Dict lookups at 2 MB are 1.75× the 50 KB rate and only 4% slower than at 1 MB.
# About 40% of mixer time is that dict work. 1.5× on the whole mixer wall covers
# a further rise without also doubling key packing, Adam, and the matmul.
COUNT_GROWTH = 1.5
PPMD_ALLOWANCE_SECONDS = 120.0

# Columns of the shared log tensor: 0 = cortex, k = order k.
MIX4_COLS = (0, 1, 2, 3, 4)
TEMP4_COLS = (0,)
MIX6_COLS = (0, 1, 2, 3, 4, 5, 6)


def dump(path, obj):
    h32.dump(path, obj)


def _fmt(value, digits=6):
    if value is None:
        return 'na'
    return f'{value:.{digits}f}'


class OrderMemoryStop(RuntimeError):
    """Order-5 and order-6 stores crossed the memory stop."""

    def __init__(self, n, order56_bytes, rss_bytes):
        self.n = int(n)
        self.order56_bytes = int(order56_bytes)
        self.rss_bytes = int(rss_bytes)
        super().__init__(
            f'order-5 and order-6 dicts are {order56_bytes} bytes at n={n} '
            f'(rss {rss_bytes} bytes)')


class OrderCount(h32.OrderCount):
    """h32.OrderCount with the same keys packed by int.from_bytes."""

    def key(self, blob, i):
        k = self.k
        if i < k:
            return None
        return int.from_bytes(blob[i - k:i], 'big')


def weight_set_max(totals, max_k):
    """Highest order k <= max_k with N_k >= 2, else 0."""
    for k in range(max_k, 0, -1):
        if totals[k - 1] >= 2:
            return k
    return 0


def peak_rss_bytes():
    status = Path('/proc/self/status')
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith('VmHWM:'):
                return int(line.split()[1]) * 1024
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def current_rss_bytes():
    status = Path('/proc/self/status')
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    return peak_rss_bytes()


def measure_order_bytes(order):
    """Approximate deep size of one order's dicts, from a 32-bucket sample."""
    nmap = order.n
    total = sys.getsizeof(nmap) + sys.getsizeof(order.N)
    n_items = len(nmap)
    if n_items == 0:
        return total
    step = max(1, n_items // 32)
    acc = 0
    seen = 0
    for i, (key, bucket) in enumerate(nmap.items()):
        if i % step != 0:
            continue
        acc += sys.getsizeof(key) + sys.getsizeof(bucket)
        for symbol, count in bucket.items():
            acc += sys.getsizeof(symbol) + sys.getsizeof(count)
        seen += 1
        if seen >= 32:
            break
    mean = acc / seen
    n_extra = 0
    for key, count in order.N.items():
        n_extra = sys.getsizeof(key) + sys.getsizeof(count)
        break
    return int(total + (mean + n_extra) * n_items)


class GeoMixer:
    """One geometric mixer. Adam lr 3e-3, one step per chunk, after the score."""

    def __init__(self, name, n_sets, n_experts, init):
        self.name = name
        self.n_sets = int(n_sets)
        self.n_experts = int(n_experts)
        self.W = torch.nn.Parameter(init.clone())
        self.opt = torch.optim.Adam([self.W], lr=MIX_LR)
        self.bits = np.zeros(0, dtype=np.float64)
        if tuple(init.shape) != (self.n_sets, self.n_experts):
            raise AssertionError(f'{name} init shape {tuple(init.shape)}')

    def ensure(self, n_slices):
        if self.bits.shape[0] >= n_slices:
            return
        extra = n_slices - int(self.bits.shape[0])
        self.bits = np.concatenate((self.bits, np.zeros(extra)))


def _init_selector(n_sets, n_experts):
    init = torch.zeros(n_sets, n_experts)
    init[:, 0] = 1
    return init


def _init_temperature(n_sets):
    return torch.ones(n_sets, 1)


class Session:
    """Shared counts and the three mixers. consume() is the LiveCapture hook."""

    def __init__(self, blob, update=True, log_every=200_000):
        self.blob = blob
        self.update = bool(update)
        self.log_every = int(log_every)
        self.counts = [OrderCount(k) for k in range(1, 7)]
        self.mix4 = GeoMixer('mix4', 5, 5, _init_selector(5, 5))
        self.temp4 = GeoMixer('temp4', 5, 1, _init_temperature(5))
        self.mix6 = GeoMixer('mix6', 7, 7, _init_selector(7, 7))
        self.mixers = (self.mix4, self.temp4, self.mix6)
        self.pos = 0
        self.count_bits = [np.zeros(0, dtype=np.float64) for _ in range(6)]
        self.s4_counts = np.zeros(5, dtype=np.int64)
        self.s6_counts = np.zeros(7, dtype=np.int64)
        self._slice_n = 0
        self._mark = -1
        self.mixer_wall = 0.0
        self.peak_rss = peak_rss_bytes()
        self.order56_bytes = 0
        self._mem_mark = -1

    def _grow(self, end):
        need = (end + SLICE - 1) // SLICE
        if need <= self._slice_n:
            return
        extra = need - self._slice_n
        for mixer in self.mixers:
            mixer.ensure(need)
        self.count_bits = [
            np.concatenate((row, np.zeros(extra))) for row in self.count_bits]
        self._slice_n = need

    def _score(self, mixer, logs, cols, s, targets):
        if cols == MIX6_COLS:
            x = logs
        else:
            x = np.ascontiguousarray(logs[:, cols, :])
        report, train = h32.Mixer._nll_bits(
            None,
            mixer.W,
            torch.from_numpy(x),
            torch.from_numpy(s),
            torch.from_numpy(targets))
        bits = report.detach().cpu().numpy()
        return bits, train

    def _check_memory(self, end):
        mark = end // 1_000_000
        if mark <= self._mem_mark and end != len(self.blob):
            return
        self._mem_mark = mark
        order56 = measure_order_bytes(self.counts[4]) + measure_order_bytes(self.counts[5])
        rss = current_rss_bytes()
        self.order56_bytes = max(self.order56_bytes, order56)
        self.peak_rss = max(self.peak_rss, peak_rss_bytes())
        if order56 > ORDER56_LIMIT and rss > 16 * 10**9:
            raise OrderMemoryStop(end, order56, rss)
        if rss > 28 * 10**9:
            raise OrderMemoryStop(end, order56, rss)

    def consume(self, log_p, targets):
        """Score every mixer on this chunk, step each one, then add counts."""
        t0 = time.perf_counter()
        targets = np.asarray(targets, dtype=np.int64)
        tlen = int(targets.shape[0])
        start = self.pos
        end = start + tlen
        if bytes(int(b) for b in targets) != self.blob[start:end]:
            raise AssertionError(f'targets diverged from the stream at {start}')
        self._grow(end)
        logs = np.empty((tlen, 7, ALPHABET), dtype=np.float32)
        logs[:, 0, :] = log_p.numpy()
        s4 = np.empty(tlen, dtype=np.int64)
        s6 = np.empty(tlen, dtype=np.int64)
        dest = np.empty((6, ALPHABET), dtype=np.float32)
        blob = self.blob
        orders = self.counts
        for t in range(tlen):
            totals = []
            for ki, order in enumerate(orders):
                totals.append(order.predict_log(blob, start + t, dest[ki]))
            logs[t, 1:, :] = dest
            s4[t] = h32.weight_set(totals[:4])
            s6[t] = weight_set_max(totals, 6)
        self.s4_counts += np.bincount(s4, minlength=5)
        self.s6_counts += np.bincount(s6, minlength=7)
        scored = [
            self._score(self.mix4, logs, MIX4_COLS, s4, targets),
            self._score(self.temp4, logs, TEMP4_COLS, s4, targets),
            self._score(self.mix6, logs, MIX6_COLS, s6, targets),
        ]
        for mixer, (bits, train) in zip(self.mixers, scored):
            if not np.isfinite(bits).all():
                raise FloatingPointError(f'nonfinite {mixer.name} bits at {start}')
            h26.deposit(mixer.bits, start, bits)
            if self.update:
                loss = train.mean()
                mixer.opt.zero_grad(set_to_none=True)
                loss.backward()
                mixer.opt.step()
        for t in range(tlen):
            i = start + t
            b = int(targets[t])
            for ki, order in enumerate(orders):
                p = order.p_byte(blob, i, b)
                bit = -math.log2(p)
                if not math.isfinite(bit):
                    raise FloatingPointError(f'nonfinite count_{ki + 1} at {i}')
                self.count_bits[ki][i // SLICE] += bit
            for order in orders:
                order.add(blob, i, b)
        self.pos = end
        self._check_memory(end)
        self.mixer_wall += time.perf_counter() - t0
        mark = end // self.log_every
        if mark > self._mark or end == len(blob):
            self._mark = mark
            self._log(end)

    def _log(self, end):
        mix4 = float(self.mix4.bits.sum()) / end
        temp4 = float(self.temp4.bits.sum()) / end
        mix6 = float(self.mix6.bits.sum()) / end
        rss = current_rss_bytes()
        self.peak_rss = max(self.peak_rss, peak_rss_bytes())
        print(
            f'H32b n={end}/{len(self.blob)} mix4_bpb={mix4:.6f} '
            f'temp4_bpb={temp4:.6f} mix6_bpb={mix6:.6f} '
            f'rss_bytes={rss} order56_bytes={self.order56_bytes} '
            f's4={self.s4_counts.tolist()} s6={self.s6_counts.tolist()}',
            flush=True)

    def _arm(self, name, sums, n):
        bits = float(np.asarray(sums, dtype=np.float64).sum())
        slices = []
        flat = np.asarray(sums, dtype=np.float64)
        for i in range((n + SLICE - 1) // SLICE):
            a = i * SLICE
            b = min(a + SLICE, n)
            sl = float(flat[i])
            slices.append({
                'index': i,
                'start': a,
                'end': b,
                'bytes': b - a,
                'bits': sl,
                'bits_per_byte': sl / (b - a),
            })
        return {
            'name': name,
            'bits': bits,
            'bits_per_byte': bits / n,
            'slices': slices,
            'finite': bool(np.isfinite(bits) and all(np.isfinite(sl['bits']) for sl in slices)),
        }

    def result(self, n):
        if self.pos != n:
            raise AssertionError(f'mixers saw {self.pos} bytes, expected {n}')
        if int(self.s4_counts.sum()) != n or int(self.s6_counts.sum()) != n:
            raise AssertionError('weight-set counts do not cover the stream')
        arms = {}
        for mixer in self.mixers:
            arms[mixer.name] = self._arm(mixer.name, mixer.bits, n)
        for k, sums in enumerate(self.count_bits, start=1):
            arms[f'count_{k}'] = self._arm(f'count_{k}', sums, n)
        self.peak_rss = max(self.peak_rss, peak_rss_bytes())
        order56 = measure_order_bytes(self.counts[4]) + measure_order_bytes(self.counts[5])
        self.order56_bytes = max(self.order56_bytes, order56)

        def weights(mixer):
            return mixer.W.detach().double().cpu().numpy().tolist()

        return {
            'arms': arms,
            'W': {
                'mix4': weights(self.mix4),
                'temp4': weights(self.temp4),
                'mix6': weights(self.mix6),
            },
            's4_counts': [int(v) for v in self.s4_counts],
            's4_share': [float(v) / n for v in self.s4_counts],
            's6_counts': [int(v) for v in self.s6_counts],
            's6_share': [float(v) / n for v in self.s6_counts],
            'mixer_wall_seconds': self.mixer_wall,
            'peak_rss_bytes': int(self.peak_rss),
            'order56_bytes': int(self.order56_bytes),
        }


def run_stream(raw, n_bytes, device, update=True, log_every=200_000):
    n_bytes = int(n_bytes)
    raw = np.asarray(raw[:n_bytes], dtype=np.uint8)
    blob = raw.tobytes()
    run = h32._prepare_cortex(device)
    session = Session(blob, update=update, log_every=log_every)
    capture = h32.LiveCapture(run.model, session)
    run.model.nll = capture.nll
    t0 = time.monotonic()
    run.advance(raw, n_bytes)
    capture.flush()
    wall = time.monotonic() - t0
    cortex = run.result()
    mixed = session.result(n_bytes)
    gap = abs(capture.captured_bits - float(cortex['bits']))
    if gap > 1e-6:
        raise AssertionError(
            f'captured log p bits {capture.captured_bits:.12f} != '
            f'cortex bits {cortex["bits"]:.12f} (gap {gap:.3e})')
    expect_steps = (n_bytes + CHUNK - 1) // CHUNK if n_bytes >= CHUNK else 0
    if int(cortex['replay_steps']) != expect_steps:
        raise AssertionError(
            f'replay_steps {cortex["replay_steps"]} != {expect_steps}')
    cortex_arm = {
        'name': 'gru',
        'bits': float(cortex['bits']),
        'bits_per_byte': float(cortex['bits_per_byte']),
        'slices': cortex['slices'],
        'finite': bool(np.isfinite(cortex['bits'])),
        'replay_steps': int(cortex['replay_steps']),
        'parameters': int(cortex['parameters']),
        'seed': int(cortex['seed']),
        'replay_seed': int(cortex['replay_seed']),
        'lr': float(cortex['lr']),
        'captured_bits': float(capture.captured_bits),
        'captured_gap': float(gap),
    }
    return {
        'cortex': cortex_arm,
        'mixer': mixed,
        'wall_seconds': wall,
        'mixer_wall_seconds': mixed['mixer_wall_seconds'],
        'n_bytes': n_bytes,
        'device': str(device),
        'update': bool(update),
        'peak_rss_bytes': mixed['peak_rss_bytes'],
        'order56_bytes': mixed['order56_bytes'],
    }


def _assert_keys_match():
    blob = bytes(range(256)) + bytes(reversed(range(256))) + b'abcde' * 20
    for k in range(1, 7):
        slow = h32.OrderCount(k)
        fast = OrderCount(k)
        for i in range(len(blob)):
            if slow.key(blob, i) != fast.key(blob, i):
                raise SystemExit(f'key mismatch order {k} at {i}')
            slow.add(blob, i, blob[i])
            fast.add(blob, i, blob[i])
        dest_s = np.empty(ALPHABET, dtype=np.float32)
        dest_f = np.empty(ALPHABET, dtype=np.float32)
        i = len(blob) - 1
        slow.predict_log(blob, i, dest_s)
        fast.predict_log(blob, i, dest_f)
        if not np.array_equal(dest_s, dest_f):
            raise SystemExit(f'predict_log mismatch order {k}')
        if slow.p_byte(blob, i, blob[i]) != fast.p_byte(blob, i, blob[i]):
            raise SystemExit(f'p_byte mismatch order {k}')


def _assert_weight_set():
    samples = [
        [0, 0, 0, 0],
        [2, 0, 0, 0],
        [5, 2, 1, 0],
        [9, 4, 3, 2],
        [1, 1, 1, 1],
    ]
    for totals in samples:
        if weight_set_max(totals, 4) != h32.weight_set(totals):
            raise SystemExit(f'weight set mismatch on {totals}')
    if weight_set_max([3, 2, 0, 0, 2, 1], 6) != 5:
        raise SystemExit('order-6 weight set failed')
    if weight_set_max([0, 0, 0, 0, 0, 0], 6) != 0:
        raise SystemExit('empty weight set failed')


def _assert_frozen_identity():
    """Initial rows select the cortex, so the reported bits are the cortex bits."""
    torch.manual_seed(0)
    tlen = 32
    log_p = torch.log_softmax(torch.randn(tlen, ALPHABET), dim=-1)
    y = torch.randint(0, ALPHABET, (tlen,))
    ref = -log_p[torch.arange(tlen), y].double() / LN2
    x6 = torch.randn(tlen, 7, ALPHABET)
    x6[:, 0, :] = log_p
    s6 = torch.randint(0, 7, (tlen,))
    report6, _train6 = h32.Mixer._nll_bits(None, _init_selector(7, 7), x6, s6, y)
    x4 = x6[:, :5, :].contiguous()
    s4 = torch.clamp(s6, max=4)
    report4, _train4 = h32.Mixer._nll_bits(None, _init_selector(5, 5), x4, s4, y)
    xt = log_p[:, None, :].contiguous()
    report_t, _train_t = h32.Mixer._nll_bits(None, _init_temperature(5), xt, s4, y)
    for name, report in (('mix4', report4), ('temp4', report_t), ('mix6', report6)):
        gap = float((report.double() - ref).abs().max())
        if gap > 1e-6:
            raise SystemExit(f'frozen identity {name} gap {gap:.3e}')


def hand_p5_last():
    """Order-5 probability of the last byte of 'abcdeAabcdeAabcdeB'.

    The last byte is B. Its context is the preceding abcde. Two earlier
    contexts abcde were each followed by A. N_5 = 2, n_5(B) = 0.
    """
    s = b'abcdeAabcdeAabcdeB'
    n_b = 0
    total = 0
    last = s[-1]
    ctx = s[-6:-1]
    if ctx != b'abcde':
        raise SystemExit(f'order-5 hand context is {ctx!r}')
    for i in range(len(s) - 1):
        if i < 5:
            continue
        if s[i - 5:i] == ctx:
            total += 1
            if s[i] == last:
                n_b += 1
    p = (n_b + SMOOTH) / (total + DENOM_EXTRA)
    return s, p, n_b, total, ctx


def count_check_order5():
    s, hand, n_b, total, ctx = hand_p5_last()
    order = OrderCount(5)
    for i in range(len(s) - 1):
        order.add(s, i, s[i])
    code = order.p_byte(s, len(s) - 1, s[-1])
    ref = h32.OrderCount(5)
    for i in range(len(s) - 1):
        ref.add(s, i, s[i])
    ref_p = ref.p_byte(s, len(s) - 1, s[-1])
    print(
        f'H32b count check string={s.decode()} order=5 context={ctx.decode()} '
        f'N_5={total} n_5(last)={n_b}',
        flush=True)
    print(f'H32b hand p_5(last)={hand:.16f}', flush=True)
    print(f'H32b code p_5(last)={code:.16f}', flush=True)
    if total != 2 or n_b != 0:
        raise SystemExit(f'hand count was N={total} n_b={n_b}, expected 2 and 0')
    if abs(code - hand) > 1e-12 or abs(ref_p - hand) > 1e-12:
        raise SystemExit(
            f'order-5 count check failed: code {code:.16f} h32 {ref_p:.16f} hand {hand:.16f}')
    return {
        'string': s.decode(),
        'context': ctx.decode(),
        'order': 5,
        'N_5': int(total),
        'n_5_last': int(n_b),
        'hand_p_5': hand,
        'code_p_5': code,
        'h32_p_5': ref_p,
        'abs_diff': abs(code - hand),
    }


def count_checks():
    _assert_keys_match()
    _assert_weight_set()
    _assert_frozen_identity()
    order2 = h32.count_check()
    order5 = count_check_order5()
    return {'order_2': order2, 'order_5': order5}


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def estimate_gpu_hours(n_timed, mixer_wall):
    """Kernel GPU hours for E10M from the CPU mixer timing.

    Cortex time is the measured H29c r8 T4 wall. Count and mixer work stays
    on CPU and is added on top. The 65536-byte mixer time is scaled to E10M
    and multiplied by COUNT_GROWTH. Order-1..6 predict on this host was 84 µs/byte
    near 50 KB, 141 µs/byte at 1 MB and 147 µs/byte at 2 MB. H32's measured E3M
    overhead, scaled to six orders and 10 MB with the same growth, is the other
    estimate. The larger one is kept. PPMd has a fixed allowance.
    """
    n_timed = int(n_timed)
    cortex = CORTEX_GPU_E10M_SECONDS
    h32_count = H32_E3M_STREAM_SECONDS - cortex * (E3M / E10M)
    h32_scaled = h32_count * (E10M / E3M) * (6 / 4) * COUNT_GROWTH
    measured_scaled = float(mixer_wall) * (E10M / n_timed) * COUNT_GROWTH
    mix = max(h32_scaled, measured_scaled)
    seconds = cortex + mix + PPMD_ALLOWANCE_SECONDS
    hours = seconds / 3600.0
    return {
        'n_timed': n_timed,
        'mixer_wall_seconds': float(mixer_wall),
        'cortex_gpu_seconds_e10m': cortex,
        'h32_e3m_count_seconds': h32_count,
        'h32_scaled_count_seconds': h32_scaled,
        'measured_scaled_count_seconds': measured_scaled,
        'count_growth': COUNT_GROWTH,
        'ppmd_allowance_seconds': PPMD_ALLOWANCE_SECONDS,
        'estimated_seconds': seconds,
        'estimated_gpu_hours': hours,
        'budget_gpu_hours': GPU_HOUR_BUDGET,
        'within_budget': bool(hours <= GPU_HOUR_BUDGET),
    }


def _finite_map(got):
    flags = {'gru': bool(got['cortex']['finite'])}
    for name, row in got['mixer']['arms'].items():
        flags[name] = bool(row['finite'])
    return flags


def smoke(data_path, out_path):
    """VPS CPU checks. Exit nonzero on failure. Does not push a kernel."""
    checked = count_checks()
    h26.configure()
    raw = h26.load_enwik8(data_path)
    digest = h26.sha256_arr(raw[:E10M])
    if digest != h26.E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest}')
    cpu = torch.device('cpu')
    n = 8192
    print('H32b smoke reference H32 8192', flush=True)
    ref = h32.run_stream(raw, n, cpu, update=True, record_frozen=False)
    print('H32b smoke learned 8192', flush=True)
    got = run_stream(raw, n, cpu, update=True, log_every=8192)
    print('H32b smoke frozen 8192', flush=True)
    held = run_stream(raw, n, cpu, update=False, log_every=8192)
    gru = got['cortex']['bits']
    ref_gru = ref['cortex']['bits']
    mix4 = got['mixer']['arms']['mix4']['bits']
    ref_mix = ref['mixer']['arms']['mix']['bits']
    diff_gru = abs(gru - ref_gru)
    diff_mix4 = abs(mix4 - ref_mix)
    temp4 = held['mixer']['arms']['temp4']['bits']
    mix6 = held['mixer']['arms']['mix6']['bits']
    held_gru = held['cortex']['bits']
    diff_temp4 = abs(temp4 - held_gru)
    diff_mix6 = abs(mix6 - held_gru)
    finite = _finite_map(got)
    finite_held = _finite_map(held)
    ok_gru = diff_gru <= 1e-6
    ok_mix4 = diff_mix4 <= 1e-6
    ok_temp4 = diff_temp4 <= 1e-6
    ok_mix6 = diff_mix6 <= 1e-6
    ok_finite = all(finite.values()) and all(finite_held.values())
    print(
        f'H32b smoke 8192 gru_abs={diff_gru:.3e} mix4_abs={diff_mix4:.3e} '
        f'temp4_abs={diff_temp4:.3e} mix6_abs={diff_mix6:.3e} finite={ok_finite}',
        flush=True)
    checks_8192 = bool(ok_gru and ok_mix4 and ok_temp4 and ok_mix6 and ok_finite)
    if not checks_8192:
        payload = {
            'passed': False,
            'failed': '8192',
            'n_bytes': n,
            'abs_diff_gru': diff_gru,
            'abs_diff_mix4': diff_mix4,
            'abs_diff_temp4': diff_temp4,
            'abs_diff_mix6': diff_mix6,
            'finite': finite,
            'finite_frozen': finite_held,
            'count_checks': checked,
        }
        dump(out_path, payload)
        raise SystemExit('H32b smoke failed on 8192')
    n_time = 65536
    print('H32b smoke time 65536', flush=True)
    timed = run_stream(raw, n_time, cpu, update=True, log_every=8192)
    finite_timed = _finite_map(timed)
    if not all(finite_timed.values()):
        dump(out_path, {'passed': False, 'failed': '65536_finite', 'finite': finite_timed})
        raise SystemExit('H32b smoke 65536 was nonfinite')
    estimate = estimate_gpu_hours(n_time, timed['mixer_wall_seconds'])
    print(
        f'H32b smoke 65536 wall_s={timed["wall_seconds"]:.3f} '
        f'mixer_s={timed["mixer_wall_seconds"]:.3f} '
        f'estimate_h={estimate["estimated_gpu_hours"]:.4f} '
        f'within_budget={estimate["within_budget"]}',
        flush=True)
    payload = {
        'n_bytes': n,
        'device': 'cpu',
        'count_checks': checked,
        'h32_gru_bits': ref_gru,
        'h32_mix_bits': ref_mix,
        'gru_bits': gru,
        'mix4_bits': mix4,
        'abs_diff_gru': diff_gru,
        'abs_diff_mix4': diff_mix4,
        'within_1e-6_gru': ok_gru,
        'within_1e-6_mix4': ok_mix4,
        'frozen_gru_bits': held_gru,
        'temp4_frozen_bits': temp4,
        'mix6_frozen_bits': mix6,
        'abs_diff_temp4': diff_temp4,
        'abs_diff_mix6': diff_mix6,
        'within_1e-6_temp4': ok_temp4,
        'within_1e-6_mix6': ok_mix6,
        'finite': finite,
        'finite_frozen': finite_held,
        'wall_seconds_8192': got['wall_seconds'],
        'frozen_wall_seconds_8192': held['wall_seconds'],
        'reference_wall_seconds_8192': ref['wall_seconds'],
        'timing': {
            'n_bytes': n_time,
            'wall_seconds': timed['wall_seconds'],
            'mixer_wall_seconds': timed['mixer_wall_seconds'],
            'gru_bits': timed['cortex']['bits'],
            'mix4_bits': timed['mixer']['arms']['mix4']['bits'],
            'temp4_bits': timed['mixer']['arms']['temp4']['bits'],
            'mix6_bits': timed['mixer']['arms']['mix6']['bits'],
            'finite': finite_timed,
            'peak_rss_bytes': timed['peak_rss_bytes'],
            'order56_bytes': timed['order56_bytes'],
        },
        'estimate': estimate,
        'passed': bool(checks_8192 and estimate['within_budget']),
        'sha256_e10m': digest,
    }
    dump(out_path, payload)
    if not estimate['within_budget']:
        raise SystemExit(
            f'H32b GPU estimate {estimate["estimated_gpu_hours"]:.4f} h '
            f'exceeds {GPU_HOUR_BUDGET} h')
    return payload


def ppmd_order16(raw):
    """H29 PPMd on E10M: pyppmd variant H, order 16, mem 1073741824.

    One-shot compress is the headline length. Incremental 1 MB encodes must
    match it; flush bytes sit on the last slice.
    """
    import pyppmd
    prefix = raw[:E10M].tobytes()
    if len(prefix) != E10M:
        raise AssertionError('E10M length')
    t0 = time.monotonic()
    one = pyppmd.compress(prefix, max_order=PPMD_ORDER, mem_size=PPMD_MEM, variant='H')
    back = pyppmd.Ppmd7Decoder(PPMD_ORDER, PPMD_MEM).decode(one, len(prefix))
    if back != prefix:
        raise RuntimeError('ppmd H order 16 roundtrip failed')
    comp = pyppmd.Ppmd7Encoder(PPMD_ORDER, PPMD_MEM)
    parts = []
    for start in range(0, len(prefix), SLICE):
        parts.append(comp.encode(prefix[start:start + SLICE]))
    parts.append(comp.flush())
    blob = b''.join(parts)
    if blob != one:
        raise RuntimeError(
            f'ppmd incremental length {len(blob)} != one-shot {len(one)}')
    wall = time.monotonic() - t0
    body = parts[:-1]
    flush = parts[-1]
    slices = []
    for i, part in enumerate(body):
        size = len(part) + (len(flush) if i == len(body) - 1 else 0)
        nbytes = len(prefix[i * SLICE:(i + 1) * SLICE])
        slices.append({
            'index': i,
            'start': i * SLICE,
            'end': i * SLICE + nbytes,
            'bytes': nbytes,
            'compressed_bytes': size,
            'bits': 8.0 * size,
            'bits_per_byte': 8.0 * size / nbytes,
        })
    compressed = len(one)
    bits = 8.0 * compressed
    return {
        'name': 'ppmd_order_16',
        'method': 'ppmd H order 16 mem 1073741824',
        'variant': 'H',
        'order': PPMD_ORDER,
        'mem_size': PPMD_MEM,
        'compressed_bytes': compressed,
        'bits': bits,
        'bits_per_byte': bits / E10M,
        'h29d_bits_per_byte': H29D_PPMD16_BPB,
        'slices': slices,
        'roundtrip': True,
        'incremental_matches_oneshot': True,
        'wall_seconds': wall,
        'finite': True,
        'bits_per_byte_definition': '8 * compressed_bytes / 10000000',
    }


def _gaps(gru, arms, ppmd):
    mix4 = arms['mix4']['bits_per_byte']
    temp4 = arms['temp4']['bits_per_byte']
    mix6 = arms['mix6']['bits_per_byte']
    gru_bpb = gru['bits_per_byte']
    ppmd_bpb = ppmd['bits_per_byte']
    return {
        'mix4_minus_ppmd': mix4 - ppmd_bpb,
        'mix4_minus_temp4': mix4 - temp4,
        'temp4_minus_gru': temp4 - gru_bpb,
        'mix6_minus_mix4': mix6 - mix4,
        'mix4_minus_gru': mix4 - gru_bpb,
    }


def kernel_main(out='/kaggle/working', kernel_id='ezzzzzx/h32b-mixer'):
    t0 = time.monotonic()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the scored run')
    device = torch.device('cuda')
    path = h26.find_kaggle_enwik8()
    print(f'H32b data {path} bytes={path.stat().st_size}', flush=True)
    raw = h26.load_enwik8(path)
    digest = h26.sha256_arr(raw[:E10M])
    if digest != h26.E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest} != {h26.E10M_SHA256}')
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f'H32b device cuda name={gpu_name} torch={torch.__version__} '
        f'n={E10M} seed=0 replay_seed={h26.REPLAY_SEED} lr=0.0003 mix_lr={MIX_LR}',
        flush=True)
    try:
        got = run_stream(raw, E10M, device, update=True, log_every=200_000)
    except OrderMemoryStop as exc:
        payload = {
            'experiment': 'h32b',
            'kernel_id': kernel_id,
            'stopped_for_memory': True,
            'n_bytes_at_stop': exc.n,
            'order56_bytes': exc.order56_bytes,
            'rss_bytes': exc.rss_bytes,
            'sha256_e10m': digest,
            'gpu_name': gpu_name,
        }
        dump(Path(out) / 'results.json', payload)
        print('KERNEL_MEMORY_STOP ' + __import__('json').dumps(payload), flush=True)
        raise SystemExit(str(exc)) from exc
    print('H32b ppmd order 16', flush=True)
    ppmd = ppmd_order16(raw)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    session = time.monotonic() - t0
    gru = got['cortex']
    arms = got['mixer']['arms']
    gaps = _gaps(gru, arms, ppmd)
    finite = bool(
        gru['finite']
        and all(row['finite'] for row in arms.values())
        and ppmd['finite'])
    payload = {
        'experiment': 'h32b',
        'kernel_id': kernel_id,
        'stopped_for_memory': False,
        'n_bytes': E10M,
        'sha256_e10m': digest,
        'gpu_name': gpu_name,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'seed': 0,
        'replay_seed': h26.REPLAY_SEED,
        'cortex_lr': 3e-4,
        'mix_lr': MIX_LR,
        'hidden': h26.HIDDEN,
        'replay_batch': 8,
        'arm': 'uniform',
        'session_wall_seconds': session,
        'stream_wall_seconds': got['wall_seconds'],
        'mixer_wall_seconds': got['mixer_wall_seconds'],
        'peak_rss_bytes': got['peak_rss_bytes'],
        'order56_bytes': got['order56_bytes'],
        'gru': gru,
        'mixer': got['mixer'],
        'ppmd': ppmd,
        'gaps': gaps,
        'h29d_r8_seed0_bits': H29D_R8_SEED0_BITS,
        'h29d_r8_seed0_bits_per_byte': H29D_R8_SEED0_BPB,
        'h29d_r8_seed0_table_bits_per_byte': H29D_R8_SEED0_TABLE_BPB,
        'gru_minus_h29d_r8_seed0': gru['bits_per_byte'] - H29D_R8_SEED0_BPB,
        'gru_minus_h29d_r8_seed0_bits': gru['bits'] - H29D_R8_SEED0_BITS,
        'finite': finite,
    }
    dump(Path(out) / 'results.json', payload)
    print('KERNEL_DONE ' + __import__('json').dumps({
        'kernel_id': kernel_id,
        'n': E10M,
        'gru_bits_per_byte': gru['bits_per_byte'],
        'mix4_bits_per_byte': arms['mix4']['bits_per_byte'],
        'temp4_bits_per_byte': arms['temp4']['bits_per_byte'],
        'mix6_bits_per_byte': arms['mix6']['bits_per_byte'],
        'ppmd_bits_per_byte': ppmd['bits_per_byte'],
        'gaps': gaps,
        'gru_minus_h29d_r8_seed0': payload['gru_minus_h29d_r8_seed0'],
        'replay_steps': gru['replay_steps'],
        'session_wall_seconds': session,
        'peak_rss_bytes': got['peak_rss_bytes'],
        'order56_bytes': got['order56_bytes'],
        'gpu_name': gpu_name,
        'finite': finite,
        'sha256_e10m': digest,
    }), flush=True)
    if not finite:
        raise SystemExit('nonfinite scored run')
    return payload


def _slice_table(rows):
    lines = [
        '| Slice | Start | End | Bits | Bits/byte |',
        '|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | {row['start']} | {row['end']} | "
            f"{row['bits']:.3f} | {_fmt(row['bits_per_byte'])} |")
    return lines


def _weight_table(w, shares, counts):
    n_set = len(w)
    n_exp = len(w[0])
    if n_exp == 1:
        header = '| s | w_gru | Byte share | Bytes |'
        rule = '|---:|---:|---:|---:|'
    else:
        cols = ' | '.join(['w_gru'] + [f'w_{k}' for k in range(1, n_exp)])
        header = f'| s | {cols} | Byte share | Bytes |'
        rule = '|---:|' + '---:|' * (n_exp + 2)
    lines = [header, rule]
    for s in range(n_set):
        cells = ' | '.join(_fmt(w[s][j], 6) for j in range(n_exp))
        share = shares[s] if s < len(shares) else 0.0
        count = counts[s] if s < len(counts) else 0
        lines.append(f'| {s} | {cells} | {_fmt(share, 6)} | {count} |')
    return lines


def write_report(scored, smoke_payload, meta, out_dir):
    out_dir = Path(out_dir)
    if scored.get('stopped_for_memory'):
        return _write_memory_stop(scored, smoke_payload, meta, out_dir)
    gru = scored['gru']
    arms = scored['mixer']['arms']
    ppmd = scored['ppmd']
    order = [('gru', gru)]
    for k in range(1, 7):
        order.append((f'count_{k}', arms[f'count_{k}']))
    order.extend([
        ('mix4', arms['mix4']),
        ('temp4', arms['temp4']),
        ('mix6', arms['mix6']),
        ('ppmd_order_16', ppmd),
    ])
    gaps = scored['gaps']
    elapsed = meta.get('log_elapsed_seconds')
    elapsed_source = 'log_clock' if elapsed is not None else 'session_wall'
    if elapsed is None:
        elapsed = float(scored['session_wall_seconds'])
    gpu_hours = float(elapsed) / 3600.0
    lines = [
        '# H32b cerebellar mixer on enwik8 E10M',
        '',
        f"E10M bytes: {E10M}. sha256: `{scored['sha256_e10m']}`.",
        'GRU bits/byte = sum of live −log2 p / 10000000. The mixers and the count',
        'models are scored the same way. PPMd bits/byte = 8 × compressed bytes / 10000000.',
        'Replay is not part of the code length. Mixer gradients do not enter the cortex.',
        '',
        'Cortex: H26 uniform, lr 0.0003, seed 0, replay seed 30000, hidden 1024,',
        'chunk 128, replay batch 8, sleep block 1. One cortex pass feeds every arm.',
        'Mixers: Adam lr 0.003, one step per chunk. mix4 and mix6 rows start at',
        '[1, 0, …, 0]. temp4 starts at 1 on every row. Count smoothing 0.05.',
        'Orders 1–6 share one count store. Counts of a chunk are added after every',
        'mixer has scored that chunk and taken its Adam step.',
        '',
        '## E10M',
        '',
        '| Arm | Bits/byte | Bits |',
        '|---|---:|---:|',
    ]
    for name, row in order:
        lines.append(f"| {name} | {_fmt(row['bits_per_byte'])} | {row['bits']:.3f} |")
    lines += ['', '## 1 MB slices', '']
    for name, row in order:
        lines += [f'### {name}', '']
        lines += _slice_table(row['slices'])
        lines.append('')
    lines += [
        '## Gaps',
        '',
        'Pre-registered differences. Not scored in this file.',
        '',
        f"mix4 − PPMd order 16 = {_fmt(gaps['mix4_minus_ppmd'])} "
        f"({_fmt(arms['mix4']['bits_per_byte'])} − {_fmt(ppmd['bits_per_byte'])}).",
        f"mix4 − temp4 = {_fmt(gaps['mix4_minus_temp4'])} "
        f"({_fmt(arms['mix4']['bits_per_byte'])} − {_fmt(arms['temp4']['bits_per_byte'])}).",
        f"temp4 − gru = {_fmt(gaps['temp4_minus_gru'])} "
        f"({_fmt(arms['temp4']['bits_per_byte'])} − {_fmt(gru['bits_per_byte'])}).",
        f"mix6 − mix4 = {_fmt(gaps['mix6_minus_mix4'])} "
        f"({_fmt(arms['mix6']['bits_per_byte'])} − {_fmt(arms['mix4']['bits_per_byte'])}).",
        f"mix4 − gru = {_fmt(gaps['mix4_minus_gru'])} "
        f"({_fmt(arms['mix4']['bits_per_byte'])} − {_fmt(gru['bits_per_byte'])}).",
        '',
        'Bar 1, not scored here: mix4 − PPMd(order 16, E10M) ≤ −0.05 bits/byte.',
        'Bar 2, not scored here: mix4 − temp4 ≤ −0.05 bits/byte.',
        'Report only: mix6 − mix4 and temp4 − gru.',
        'Seed-to-seed std of the cortex on E10M is 0.004 (H29d).',
        '',
        f"PPMd order 16 recomputed: {_fmt(ppmd['bits_per_byte'])} "
        f"({ppmd['compressed_bytes']} compressed bytes).",
        f"PPMd order 16 H29d: {_fmt(H29D_PPMD16_BPB)}.",
        '',
        '## Mixer weights',
        '',
        'mix4 and temp4 share s: the highest order k ≤ 4 with N_k(ctx) ≥ 2, else 0.',
        'Columns of mix4 are [gru, count_1, count_2, count_3, count_4].',
        'temp4 has one column, gru.',
        '',
        '### mix4',
        '',
    ]
    w = scored['mixer']['W']
    lines += _weight_table(w['mix4'], scored['mixer']['s4_share'], scored['mixer']['s4_counts'])
    lines += ['', '### temp4', '']
    lines += _weight_table(w['temp4'], scored['mixer']['s4_share'], scored['mixer']['s4_counts'])
    lines += [
        '',
        '### mix6',
        '',
        's is the highest order k ≤ 6 with N_k(ctx) ≥ 2, else 0.',
        'Columns are [gru, count_1, count_2, count_3, count_4, count_5, count_6].',
        '',
    ]
    lines += _weight_table(w['mix6'], scored['mixer']['s6_share'], scored['mixer']['s6_counts'])
    lines += [
        '',
        '### Weight-set shares',
        '',
        's4 is the set used by mix4 and temp4. s6 is the set used by mix6.',
        '',
        '| s | s4 bytes | s4 share | s6 bytes | s6 share |',
        '|---:|---:|---:|---:|---:|',
    ]
    for s in range(7):
        b4 = scored['mixer']['s4_counts'][s] if s < 5 else 0
        p4 = scored['mixer']['s4_share'][s] if s < 5 else 0.0
        b6 = scored['mixer']['s6_counts'][s]
        p6 = scored['mixer']['s6_share'][s]
        lines.append(f'| {s} | {b4} | {_fmt(p4, 6)} | {b6} | {_fmt(p6, 6)} |')
    lines += [
        '',
        '## Cortex check',
        '',
        'Check, not a gate. H29d seed 0 r8_3e4 is the H29c run.',
        f"H29d r8 seed 0 bits: {H29D_R8_SEED0_BITS:.12f}.",
        f"H29d r8 seed 0 bits/byte: {H29D_R8_SEED0_BPB:.12f}.",
        f"H29d table bits/byte: {H29D_R8_SEED0_TABLE_BPB:.6f}.",
        f"This gru bits: {gru['bits']:.12f}.",
        f"This gru bits/byte: {gru['bits_per_byte']:.12f}.",
        f"Difference bits: {scored['gru_minus_h29d_r8_seed0_bits']:.6e}.",
        f"Difference bits/byte: {scored['gru_minus_h29d_r8_seed0']:.6e}.",
        '',
        '## Compute',
        '',
        '| Item | Value |',
        '|---|---|',
        f"| Kernel | {scored['kernel_id']} |",
        f"| Kernel version | {meta.get('version')} |",
        f"| Status | {meta.get('status')} |",
        f"| GPU | {scored.get('gpu_name')} |",
        f"| Torch | {scored.get('torch_version')} |",
        f"| Elapsed seconds | {elapsed:.3f} |",
        f"| Elapsed source | {elapsed_source} |",
        f"| Script wall seconds | {scored['session_wall_seconds']:.3f} |",
        f"| Stream wall seconds | {scored['stream_wall_seconds']:.3f} |",
        f"| Mixer wall seconds | {scored['mixer_wall_seconds']:.3f} |",
        f"| PPMd wall seconds | {ppmd['wall_seconds']:.3f} |",
        f"| GPU hours | {gpu_hours:.4f} |",
        f"| Peak RSS bytes | {scored['peak_rss_bytes']} |",
        f"| Order 5+6 bytes | {scored['order56_bytes']} |",
        f"| Replay steps | {gru['replay_steps']} |",
        '',
        '## Smoke',
        '',
        'CPU. Not a score.',
        '',
    ]
    lines += _smoke_lines(smoke_payload)
    lines += [
        '',
        '## Protocol',
        '',
        '- One online pass. The cortex is H26 arm uniform: score the chunk, one Adam',
        '  step on its own loss, then one replay step of 8 windows. Seed 0, replay seed 30000.',
        '- Cortex log p is log_softmax of the scoring logits, detached. Natural log.',
        '- Count models: p(b) = (n + 0.05) / (N + 0.05 · 256). Orders 1–6. Exact dict counts.',
        '- A byte with fewer than k previous bytes uses the empty context.',
        '- count_k is per-byte prequential: the count is added after that byte is scored.',
        '- The mixer scores a 128-byte chunk with counts from before the chunk.',
        '  p_mix = softmax(Σ_j w_{s,j} log p_j). Then one Adam step on that mixer',
        '  (lr 0.003, mean −log2 p_mix). The chunk is added after mix4, temp4 and mix6',
        '  have each scored and stepped.',
        '- mix4 inputs are [log p_gru, log p_1 … log p_4]. s is the highest k ≤ 4 with',
        '  N_k ≥ 2, else 0. W is 5 × 5, every row initialised to [1, 0, 0, 0, 0].',
        '- temp4 inputs are [log p_gru] only, with the same s as mix4. W is 5 × 1, every entry 1.',
        '- mix6 inputs are [log p_gru, log p_1 … log p_6]. s is the highest k ≤ 6 with',
        '  N_k ≥ 2, else 0. W is 7 × 7, every row initialised to [1, 0, 0, 0, 0, 0, 0].',
        '- A row still equal to its initial selector is scored as the cortex. The Adam',
        '  loss is always log_softmax.',
        '- PPMd is pyppmd variant H (Ppmd7), order 16, mem_size 1073741824, raw stream',
        '  plus flush, the H29 call. Slice lengths are incremental encodes on that encoder;',
        '  flush bytes are on the last slice. The incremental blob matches the one-shot blob.',
        '- E10M is bytes [0, 10000000). 1 MB slices are 1000000 bytes.',
        '- Code is h32b_mixer.py. It imports h32_mixer.py and h26_cued.py. Private dataset',
        '  ezzzzzx/pdp-code-h32b. Data dataset ezzzzzx/enwik8.',
        '',
    ]
    text = '\n'.join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'RESULTS.md').write_text(text)
    results = {
        'experiment': 'h32b',
        'n_bytes': E10M,
        'sha256_e10m': scored['sha256_e10m'],
        'kernel_id': scored['kernel_id'],
        'kernel_version': meta.get('version'),
        'kernel_status': meta.get('status'),
        'gpu_name': scored.get('gpu_name'),
        'torch_version': scored.get('torch_version'),
        'elapsed_seconds': elapsed,
        'elapsed_source': elapsed_source,
        'session_wall_seconds': scored['session_wall_seconds'],
        'stream_wall_seconds': scored['stream_wall_seconds'],
        'mixer_wall_seconds': scored['mixer_wall_seconds'],
        'gpu_hours': gpu_hours,
        'peak_rss_bytes': scored['peak_rss_bytes'],
        'order56_bytes': scored['order56_bytes'],
        'stopped_for_memory': False,
        'arms': {name: row for name, row in order},
        'gaps': gaps,
        'W': w,
        's4_counts': scored['mixer']['s4_counts'],
        's4_share': scored['mixer']['s4_share'],
        's6_counts': scored['mixer']['s6_counts'],
        's6_share': scored['mixer']['s6_share'],
        'h29d_r8_seed0_bits': H29D_R8_SEED0_BITS,
        'h29d_r8_seed0_bits_per_byte': H29D_R8_SEED0_BPB,
        'h29d_r8_seed0_table_bits_per_byte': H29D_R8_SEED0_TABLE_BPB,
        'gru_minus_h29d_r8_seed0': scored['gru_minus_h29d_r8_seed0'],
        'gru_minus_h29d_r8_seed0_bits': scored['gru_minus_h29d_r8_seed0_bits'],
        'h29d_ppmd16_bits_per_byte': H29D_PPMD16_BPB,
        'ppmd_bits_per_byte': ppmd['bits_per_byte'],
        'ppmd_compressed_bytes': ppmd['compressed_bytes'],
        'smoke': smoke_payload,
        'finite': scored['finite'],
    }
    dump(out_dir / 'results.json', results)
    return results


def _smoke_lines(smoke_payload):
    c2 = smoke_payload['count_checks']['order_2']
    c5 = smoke_payload['count_checks']['order_5']
    timing = smoke_payload['timing']
    est = smoke_payload['estimate']
    lines = [
        f"Bytes: {smoke_payload['n_bytes']}.",
        f"H32 gru bits: {smoke_payload['h32_gru_bits']:.12f}.",
        f"gru bits: {smoke_payload['gru_bits']:.12f}.",
        f"Absolute difference: {smoke_payload['abs_diff_gru']:.3e}.",
        f"Within 1e-6: {smoke_payload['within_1e-6_gru']}.",
        '',
        f"H32 mix bits: {smoke_payload['h32_mix_bits']:.12f}.",
        f"mix4 bits: {smoke_payload['mix4_bits']:.12f}.",
        f"Absolute difference: {smoke_payload['abs_diff_mix4']:.3e}.",
        f"Within 1e-6: {smoke_payload['within_1e-6_mix4']}.",
        '',
        f"Frozen gru bits: {smoke_payload['frozen_gru_bits']:.12f}.",
        f"Frozen temp4 bits: {smoke_payload['temp4_frozen_bits']:.12f}.",
        f"Absolute difference: {smoke_payload['abs_diff_temp4']:.3e}.",
        f"Within 1e-6: {smoke_payload['within_1e-6_temp4']}.",
        f"Frozen mix6 bits: {smoke_payload['mix6_frozen_bits']:.12f}.",
        f"Absolute difference: {smoke_payload['abs_diff_mix6']:.3e}.",
        f"Within 1e-6: {smoke_payload['within_1e-6_mix6']}.",
        '',
        'Count check, last byte of abracadabraabracadab, order 2, context da.',
        f"N_2 = {c2['N_2']}, n_2(b) = {c2['n_2_b']}.",
        f"Hand p_2 = {c2['hand_p_2']:.16f}.",
        f"Code p_2 = {c2['code_p_2']:.16f}.",
        '',
        f"Count check, last byte of {c5['string']}, order 5, context {c5['context']}.",
        f"N_5 = {c5['N_5']}, n_5(last) = {c5['n_5_last']}.",
        f"Hand p_5 = {c5['hand_p_5']:.16f}.",
        f"Code p_5 = {c5['code_p_5']:.16f}.",
        '',
        'Finite, learned 8192:',
        '',
    ]
    for name, flag in smoke_payload['finite'].items():
        lines.append(f'- {name}: {flag}')
    lines += ['', 'Finite, frozen 8192:', '']
    for name, flag in smoke_payload['finite_frozen'].items():
        lines.append(f'- {name}: {flag}')
    lines += [
        '',
        f"Timing bytes: {timing['n_bytes']}.",
        f"Wall seconds: {timing['wall_seconds']:.3f}.",
        f"Mixer wall seconds: {timing['mixer_wall_seconds']:.3f}.",
        f"Estimated kernel seconds: {est['estimated_seconds']:.3f}.",
        f"Estimated GPU hours: {est['estimated_gpu_hours']:.4f}.",
        f"Budget GPU hours: {est['budget_gpu_hours']:.1f}.",
        f"Within budget: {est['within_budget']}.",
        f"Smoke passed: {smoke_payload['passed']}.",
    ]
    return lines


def _write_memory_stop(scored, smoke_payload, meta, out_dir):
    lines = [
        '# H32b cerebellar mixer on enwik8 E10M',
        '',
        'Stopped before a score. The order-5 and order-6 count dicts crossed the memory stop.',
        '',
        f"Bytes scored before the stop: {scored.get('n_bytes_at_stop')}.",
        f"Order 5+6 bytes: {scored.get('order56_bytes')}.",
        f"RSS bytes: {scored.get('rss_bytes')}.",
        f"Kernel: {scored.get('kernel_id')}.",
        f"Kernel version: {meta.get('version')}.",
        f"Status: {meta.get('status')}.",
        '',
        '## Smoke',
        '',
        'CPU. Not a score.',
        '',
    ]
    if smoke_payload:
        lines += _smoke_lines(smoke_payload)
    text = '\n'.join(lines) + '\n'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'RESULTS.md').write_text(text)
    results = {
        'experiment': 'h32b',
        'stopped_for_memory': True,
        'n_bytes_at_stop': scored.get('n_bytes_at_stop'),
        'order56_bytes': scored.get('order56_bytes'),
        'rss_bytes': scored.get('rss_bytes'),
        'kernel_id': scored.get('kernel_id'),
        'kernel_version': meta.get('version'),
        'kernel_status': meta.get('status'),
        'smoke': smoke_payload,
    }
    dump(out_dir / 'results.json', results)
    return results


def assemble(out_dir, kernel_json, smoke_json, log_path, version, status):
    scored = __import__('json').loads(Path(kernel_json).read_text())
    smoke_payload = __import__('json').loads(Path(smoke_json).read_text())
    elapsed = None
    if log_path and Path(log_path).is_file():
        elapsed = h26.log_elapsed_seconds(log_path)
    meta = {
        'version': version,
        'status': status,
        'log_elapsed_seconds': elapsed,
    }
    return write_report(scored, smoke_payload, meta, out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--count-check', action='store_true')
    ap.add_argument('--data', default='data/enwik8/enwik8')
    ap.add_argument('--out', default='results/h32b')
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--kernel-json')
    ap.add_argument('--smoke-json')
    ap.add_argument('--log')
    ap.add_argument('--kernel-version', type=int)
    ap.add_argument('--kernel-status')
    args = ap.parse_args()
    if args.count_check:
        count_checks()
        return
    if args.smoke:
        smoke(args.data, Path(args.out) / 'smoke.json')
        return
    if args.assemble:
        assemble(
            args.out, args.kernel_json, args.smoke_json, args.log,
            args.kernel_version, args.kernel_status)
        return
    raise SystemExit('pass --smoke, --count-check, or --assemble')


if __name__ == '__main__':
    main()
