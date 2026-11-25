"""H32: count models of orders 1–4 beside the H26 uniform cortex, mixed online.

The cortex is h26_cued.StreamRun arm uniform. Its loss is the only gradient
that reaches it. Log in the mixer is natural log, so softmax(log p) = p.
Numbers only. No verdict. Does not commit.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h26_cued as h26

ALPHABET = 256
ORDERS = (1, 2, 3, 4)
SMOOTH = 0.05
DENOM_EXTRA = SMOOTH * ALPHABET
MIX_LR = 3e-3
N_EXPERTS = 5
LN2 = math.log(2)
E3M = h26.E3M
SLICE = h26.SLICE
CHUNK = h26.CHUNK
H26_UNIFORM_E3M_BITS = 5874459.001424253
H26_UNIFORM_E3M_BPB = 1.958153000474751
# CPU smoke figure from results/h26/RESULTS.md. Informational on this host.
H26_UNIFORM_8192_BITS = 42063.298164869797
PPMD_ORDER = 16
PPMD_MEM = 1 << 30


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def _fmt(value, digits=6):
    if value is None:
        return 'na'
    return f'{value:.{digits}f}'


class OrderCount:
    """Exact counts for one order. Empty context when fewer than k bytes precede."""

    def __init__(self, k):
        self.k = int(k)
        self.n = {}
        self.N = {}

    def key(self, blob, i):
        if i < self.k:
            return None
        acc = 0
        base = i - self.k
        for j in range(self.k):
            acc = (acc << 8) | blob[base + j]
        return acc

    def predict_log(self, blob, i, out):
        """Natural log probabilities into out[256]. Returns N_k(ctx)."""
        key = self.key(blob, i)
        total = self.N.get(key, 0)
        denom = total + DENOM_EXTRA
        unseen = math.log(SMOOTH / denom)
        out.fill(unseen)
        bucket = self.n.get(key)
        if bucket:
            for b, c in bucket.items():
                out[b] = math.log((c + SMOOTH) / denom)
        return total

    def p_byte(self, blob, i, b):
        key = self.key(blob, i)
        total = self.N.get(key, 0)
        c = 0
        bucket = self.n.get(key)
        if bucket is not None:
            c = bucket.get(b, 0)
        return (c + SMOOTH) / (total + DENOM_EXTRA)

    def add(self, blob, i, b):
        key = self.key(blob, i)
        bucket = self.n.get(key)
        if bucket is None:
            self.n[key] = {b: 1}
            self.N[key] = 1
        else:
            bucket[b] = bucket.get(b, 0) + 1
            self.N[key] += 1


class CountBank:
    def __init__(self):
        self.orders = [OrderCount(k) for k in ORDERS]

    def context_totals(self, blob, i, dest):
        """dest is (4, 256) float32 natural logs. Returns N for orders 1..4."""
        totals = []
        for ki, order in enumerate(self.orders):
            totals.append(order.predict_log(blob, i, dest[ki]))
        return totals

    def add(self, blob, i, b):
        for order in self.orders:
            order.add(blob, i, b)


def weight_set(totals):
    """Highest order k with N_k >= 2, else 0. totals are orders 1..4."""
    for k in range(4, 0, -1):
        if totals[k - 1] >= 2:
            return k
    return 0


def hand_p2_last():
    """Order-2 probability of the last byte of 'abracadabraabracadab'.

    Bytes 0..18 have been scored. The only earlier context 'da' is bytes 6–7,
    followed by byte 8 ('b'). N_2 = 1, n_2('b') = 1.
    """
    s = b'abracadabraabracadab'
    n_b = 0
    total = 0
    for i in range(len(s) - 1):
        if i < 2:
            continue
        if s[i - 2:i] == b'da':
            total += 1
            if s[i] == ord('b'):
                n_b += 1
    p = (n_b + SMOOTH) / (total + DENOM_EXTRA)
    return p, n_b, total


class Mixer:
    """Geometric mixer. One Adam step on W per cortex chunk, after the score."""

    def __init__(self, blob, update=True, record_frozen=False):
        self.blob = blob
        self.update = bool(update)
        self.record_frozen = bool(record_frozen)
        init = torch.zeros(N_EXPERTS, N_EXPERTS)
        init[:, 0] = 1
        self.W = torch.nn.Parameter(init)
        self.opt = torch.optim.Adam([self.W], lr=MIX_LR)
        self.W0 = init.detach().clone()
        self.counts = CountBank()
        self.pos = 0
        self.mix_bits = np.zeros(0, dtype=np.float64)
        self.frozen_bits = np.zeros(0, dtype=np.float64)
        self.count_bits = [np.zeros(0, dtype=np.float64) for _ in ORDERS]
        self.s_counts = np.zeros(N_EXPERTS, dtype=np.int64)
        self._slice_n = 0
        self._mark = -1
        # Scored after the cortex pass so the extra matmul cannot sit between chunks.
        self._frozen_chunks = []

    def _grow(self, end):
        need = (end + SLICE - 1) // SLICE
        if need <= self._slice_n:
            return
        extra = need - self._slice_n
        self.mix_bits = np.concatenate((self.mix_bits, np.zeros(extra)))
        self.frozen_bits = np.concatenate((self.frozen_bits, np.zeros(extra)))
        self.count_bits = [
            np.concatenate((row, np.zeros(extra))) for row in self.count_bits]
        self._slice_n = need

    def _nll_bits(self, W, x, s, y):
        """Return (reported bits, training bits).

        Reported bits use softmax(sum w_j log p_j). A row that is still exactly
        [1, 0, 0, 0, 0] has scores equal to log p_gru, and softmax of a
        normalised log-probability is the identity. Float32 log_softmax leaves
        a partition residual, so those rows keep scores. The Adam loss always
        uses log_softmax, which is the gradient of that same map.
        """
        w = W[s]
        scores = torch.einsum('tj,tjv->tv', w, x)
        log_mix = torch.log_softmax(scores, dim=-1)
        mask = (w[:, 0] == 1) & torch.all(w[:, 1:] == 0, dim=-1)
        if bool(mask.any().item()):
            chosen = torch.where(mask[:, None], scores, log_mix)
        else:
            chosen = log_mix
        idx = torch.arange(y.shape[0])
        # Divide in float64, matching the cortex (losses.double() / log(2)).
        report = -chosen[idx, y].double() / LN2
        train = -log_mix[idx, y] / LN2
        return report, train

    def consume(self, log_p, targets):
        """Score one cortex chunk, step W, then update counts.

        log_p is the cortex's natural-log distribution, detached, shape (T, 256).
        Count-only arms are per-byte prequential. The mixer sees counts from
        before this chunk, then the chunk is added after the Adam step.
        """
        targets = np.asarray(targets, dtype=np.int64)
        tlen = int(targets.shape[0])
        start = self.pos
        end = start + tlen
        if bytes(int(b) for b in targets) != self.blob[start:end]:
            raise AssertionError(f'targets diverged from the stream at {start}')
        self._grow(end)
        x = np.empty((tlen, N_EXPERTS, ALPHABET), dtype=np.float32)
        x[:, 0, :] = log_p.numpy()
        s = np.empty(tlen, dtype=np.int64)
        dest = np.empty((4, ALPHABET), dtype=np.float32)
        for t in range(tlen):
            totals = self.counts.context_totals(self.blob, start + t, dest)
            x[t, 1:, :] = dest
            s[t] = weight_set(totals)
        self.s_counts += np.bincount(s, minlength=N_EXPERTS)
        xt = torch.from_numpy(x)
        st = torch.from_numpy(s)
        yt = torch.from_numpy(targets)
        report, train = self._nll_bits(self.W, xt, st, yt)
        bits = report.detach().cpu().numpy()
        if not np.isfinite(bits).all():
            raise FloatingPointError(f'nonfinite mix bits at {start}')
        h26.deposit(self.mix_bits, start, bits)
        if self.update:
            loss = train.mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        if self.record_frozen:
            self._frozen_chunks.append((start, x.copy(), s.copy(), targets.copy()))
        for t in range(tlen):
            i = start + t
            b = int(targets[t])
            for ki, order in enumerate(self.counts.orders):
                p = order.p_byte(self.blob, i, b)
                bit = -math.log2(p)
                if not math.isfinite(bit):
                    raise FloatingPointError(f'nonfinite count_{ki + 1} at {i}')
                self.count_bits[ki][i // SLICE] += bit
            self.counts.add(self.blob, i, b)
        self.pos = end
        # 200000 is not a multiple of 128, so log when the chunk crosses a mark.
        mark = end // 200_000
        if mark > self._mark or end == len(self.blob):
            self._mark = mark
            self._log(end)

    def finalize_frozen(self):
        if not self.record_frozen:
            return
        with torch.no_grad():
            for start, x, s, targets in self._frozen_chunks:
                report, _train = self._nll_bits(
                    self.W0,
                    torch.from_numpy(x),
                    torch.from_numpy(s),
                    torch.from_numpy(targets))
                bits = report.cpu().numpy()
                if not np.isfinite(bits).all():
                    raise FloatingPointError(f'nonfinite frozen mix bits at {start}')
                h26.deposit(self.frozen_bits, start, bits)
        self._frozen_chunks.clear()

    def _log(self, end):
        mix = float(self.mix_bits.sum()) / end
        c1 = float(self.count_bits[0].sum()) / end
        print(
            f'H32 n={end}/{len(self.blob)} mix_bpb={mix:.6f} count_1_bpb={c1:.6f} '
            f's={self.s_counts.tolist()}',
            flush=True)

    def _arm(self, name, sums, n):
        bits = float(sums.sum())
        slices = []
        for i in range((n + SLICE - 1) // SLICE):
            a = i * SLICE
            b = min(a + SLICE, n)
            sl = float(sums[i])
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
            raise AssertionError(f'mixer saw {self.pos} bytes, expected {n}')
        if int(self.s_counts.sum()) != n:
            raise AssertionError('weight-set counts do not cover the stream')
        arms = {
            'mix': self._arm('mix', self.mix_bits, n),
        }
        for k, sums in enumerate(self.count_bits, start=1):
            arms[f'count_{k}'] = self._arm(f'count_{k}', sums, n)
        frozen = None
        if self.record_frozen:
            frozen = self._arm('mix_frozen', self.frozen_bits, n)
        best_k = min(ORDERS, key=lambda k: arms[f'count_{k}']['bits'])
        return {
            'arms': arms,
            'frozen': frozen,
            'W': self.W.detach().double().cpu().numpy().tolist(),
            's_counts': [int(v) for v in self.s_counts],
            's_share': [float(v) / n for v in self.s_counts],
            'best_count_order': int(best_k),
            'best_count_bits_per_byte': arms[f'count_{best_k}']['bits_per_byte'],
        }


class LiveCapture:
    """Replacement nll. Same ops as ByteGRU.nll, plus a detached log-softmax."""

    def __init__(self, model, mixer):
        self.model = model
        self.mixer = mixer
        self.pending = None
        self.captured_bits = 0.0

    def flush(self):
        if self.pending is None:
            return
        log_p, targets = self.pending
        self.pending = None
        idx = torch.arange(targets.shape[0])
        taken = -log_p[idx, torch.from_numpy(targets)].double() / LN2
        self.captured_bits += float(taken.sum())
        self.mixer.consume(log_p, targets)

    def nll(self, prev, targets, state):
        live = prev.shape[1] == 1
        if live:
            # Previous chunk's mixer work, after that chunk's cortex step.
            self.flush()
        emb = self.model.embedding(prev.clamp_min(0))
        emb = emb.masked_fill((prev < 0).unsqueeze(-1), 0)
        y, new_state = self.model.gru(emb, state)
        logits = self.model.readout(y)
        losses = F.cross_entropy(
            logits.reshape(-1, ALPHABET), targets.reshape(-1), reduction='none')
        if live:
            flat = logits.detach().reshape(-1, ALPHABET)
            log_p = F.log_softmax(flat, dim=-1).to('cpu')
            tgt = targets.detach().reshape(-1).to('cpu').numpy().astype(np.int64, copy=False)
            self.pending = (log_p, tgt)
        return losses, new_state


def _prepare_cortex(device):
    h26.SEED = 0
    h26.REPLAY_BATCH = 8
    h26.configure()
    run = h26.StreamRun(
        3e-4, True, device, 'uniform', sleep_block=1, arm='uniform', retrieval_log=0)
    if h26.REPLAY_BATCH != 8:
        raise AssertionError(f'replay batch {h26.REPLAY_BATCH}')
    if run.arm != 'uniform' or run.n_cued_per_step != 0 or run.n_uniform_per_step != 8:
        raise AssertionError('cortex is not the uniform arm')
    return run


def run_stream(raw, n_bytes, device, update=True, record_frozen=False):
    n_bytes = int(n_bytes)
    raw = np.asarray(raw[:n_bytes], dtype=np.uint8)
    blob = raw.tobytes()
    run = _prepare_cortex(device)
    mixer = Mixer(blob, update=update, record_frozen=record_frozen)
    capture = LiveCapture(run.model, mixer)
    run.model.nll = capture.nll
    t0 = time.monotonic()
    run.advance(raw, n_bytes)
    capture.flush()
    mixer.finalize_frozen()
    wall = time.monotonic() - t0
    cortex = run.result()
    mixed = mixer.result(n_bytes)
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
        'n_bytes': n_bytes,
        'device': str(device),
    }


def count_check():
    hand, n_b, total = hand_p2_last()
    s = b'abracadabraabracadab'
    bank = CountBank()
    for i in range(len(s) - 1):
        bank.add(s, i, s[i])
    code = bank.orders[1].p_byte(s, len(s) - 1, s[-1])
    print(
        f'H32 count check string=abracadabraabracadab '
        f'N_2(da)={total} n_2(da,b)={n_b}',
        flush=True)
    print(f'H32 hand p_2(last)={hand:.16f}', flush=True)
    print(f'H32 code p_2(last)={code:.16f}', flush=True)
    if total != 1 or n_b != 1:
        raise SystemExit(f'hand count was N={total} n_b={n_b}, expected 1 and 1')
    if abs(code - hand) > 1e-12:
        raise SystemExit(f'count check failed: code {code:.16f} hand {hand:.16f}')
    return {
        'string': 'abracadabraabracadab',
        'context': 'da',
        'N_2': int(total),
        'n_2_b': int(n_b),
        'hand_p_2': hand,
        'code_p_2': code,
        'abs_diff': abs(code - hand),
    }


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def smoke(data_path, out_path):
    """VPS CPU checks. Exit nonzero on failure."""
    checked = count_check()
    h26.configure()
    raw = h26.load_enwik8(data_path)
    digest = h26.sha256_arr(raw[:E3M])
    if digest != h26.E3M_SHA256:
        raise SystemExit(f'E3M sha256 {digest}')
    n = 8192
    cpu = torch.device('cpu')
    print('H32 smoke reference H26 uniform 8192', flush=True)
    ref = h26.run_arm(raw, n, 3e-4, 8, 'uniform', cpu, retrieval_log=0)
    print('H32 smoke learned mixer 8192', flush=True)
    got = run_stream(raw, n, cpu, update=True, record_frozen=False)
    print('H32 smoke frozen mixer 8192', flush=True)
    # A second pass with W held at its initial value. Same cortex schedule.
    held = run_stream(raw, n, cpu, update=False, record_frozen=False)
    gru = got['cortex']['bits']
    ref_bits = float(ref['bits'])
    frozen = held['mixer']['arms']['mix']['bits']
    frozen_gru = held['cortex']['bits']
    diff_h26 = abs(gru - ref_bits)
    diff_frozen = abs(frozen - frozen_gru)
    diff_frozen_bpb = diff_frozen / n
    arms = {'gru': got['cortex']}
    arms.update(got['mixer']['arms'])
    finite = {name: bool(row['finite']) for name, row in arms.items()}
    finite['mix_frozen'] = bool(held['mixer']['arms']['mix']['finite'])
    finite['gru_frozen'] = bool(held['cortex']['finite'])
    published_gap = abs(ref_bits - H26_UNIFORM_8192_BITS)
    # Both checks are on the total bits. Frozen W is the initial selector,
    # so softmax(log p_gru) is p_gru.
    ok_h26 = diff_h26 <= 1e-6
    ok_frozen = diff_frozen <= 1e-6
    ok_finite = all(finite.values())
    payload = {
        'n_bytes': n,
        'device': 'cpu',
        'count_check': checked,
        'h26_uniform_bits': ref_bits,
        'h26_uniform_replay_steps': int(ref['replay_steps']),
        'gru_bits': gru,
        'gru_replay_steps': int(got['cortex']['replay_steps']),
        'gru_minus_h26': gru - ref_bits,
        'abs_diff_h26': diff_h26,
        'within_1e-6_h26': ok_h26,
        'published_8192_bits': H26_UNIFORM_8192_BITS,
        'abs_diff_published': published_gap,
        'mix_frozen_bits': frozen,
        'gru_frozen_bits': frozen_gru,
        'mix_frozen_minus_gru': frozen - frozen_gru,
        'abs_diff_frozen': diff_frozen,
        'abs_diff_frozen_bits_per_byte': diff_frozen_bpb,
        'within_1e-6_frozen': ok_frozen,
        'mix_bits': arms['mix']['bits'],
        'finite': finite,
        'all_finite': ok_finite,
        'captured_gap': got['cortex']['captured_gap'],
        'wall_seconds': got['wall_seconds'],
        'passed': bool(ok_h26 and ok_frozen and ok_finite),
    }
    dump(out_path, payload)
    print(
        f'H32 smoke gru={gru:.12f} h26={ref_bits:.12f} abs={diff_h26:.3e} '
        f'frozen={frozen:.12f} frozen_abs={diff_frozen:.3e} '
        f'frozen_bpb_abs={diff_frozen_bpb:.3e} finite={ok_finite} '
        f'passed={payload["passed"]}',
        flush=True)
    if not payload['passed']:
        raise SystemExit('H32 smoke failed')
    return payload


def ppmd_order16(raw):
    """H29 PPMd: pyppmd variant H, order 16, mem 1073741824, raw stream plus flush.

    One-shot compress is the headline length. Incremental 1 MB encodes must
    match it; flush bytes sit on the last slice.
    """
    import pyppmd
    prefix = raw[:E3M].tobytes()
    if len(prefix) != E3M:
        raise AssertionError('E3M length')
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
    return {
        'name': 'ppmd_order_16',
        'method': 'ppmd H order 16 mem 1073741824',
        'variant': 'H',
        'order': PPMD_ORDER,
        'mem_size': PPMD_MEM,
        'compressed_bytes': compressed,
        'bits': 8.0 * compressed,
        'bits_per_byte': 8.0 * compressed / E3M,
        'slices': slices,
        'roundtrip': True,
        'incremental_matches_oneshot': True,
        'wall_seconds': wall,
        'finite': True,
        'bits_per_byte_definition': '8 * compressed_bytes / 3000000',
    }


def kernel_main(out='/kaggle/working', kernel_id='ezzzzzx/h32-mixer'):
    t0 = time.monotonic()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the scored run')
    device = torch.device('cuda')
    path = h26.find_kaggle_enwik8()
    print(f'H32 data {path} bytes={path.stat().st_size}', flush=True)
    raw = h26.load_enwik8(path)
    digest = h26.sha256_arr(raw[:E3M])
    if digest != h26.E3M_SHA256:
        raise SystemExit(f'E3M sha256 {digest} != {h26.E3M_SHA256}')
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f'H32 device cuda name={gpu_name} torch={torch.__version__} '
        f'n={E3M} seed=0 replay_seed={h26.REPLAY_SEED} lr=0.0003 mix_lr={MIX_LR}',
        flush=True)
    got = run_stream(raw, E3M, device, update=True, record_frozen=False)
    print('H32 ppmd order 16', flush=True)
    ppmd = ppmd_order16(raw)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    session = time.monotonic() - t0
    gru = got['cortex']
    mix = got['mixer']['arms']['mix']
    payload = {
        'experiment': 'h32',
        'kernel_id': kernel_id,
        'n_bytes': E3M,
        'sha256_e3m': digest,
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
        'gru': gru,
        'mixer': got['mixer'],
        'ppmd': ppmd,
        'gaps': {
            'mix_minus_gru': mix['bits_per_byte'] - gru['bits_per_byte'],
            'mix_minus_ppmd': mix['bits_per_byte'] - ppmd['bits_per_byte'],
        },
        'h26_uniform_e3m_bits': H26_UNIFORM_E3M_BITS,
        'h26_uniform_e3m_bpb': H26_UNIFORM_E3M_BPB,
        'gru_minus_h26_uniform_bits': gru['bits'] - H26_UNIFORM_E3M_BITS,
        'finite': bool(
            gru['finite']
            and all(row['finite'] for row in got['mixer']['arms'].values())
            and ppmd['finite']),
    }
    dump(Path(out) / 'results.json', payload)
    print('KERNEL_DONE ' + json.dumps({
        'kernel_id': kernel_id,
        'n': E3M,
        'gru_bits_per_byte': gru['bits_per_byte'],
        'mix_bits_per_byte': mix['bits_per_byte'],
        'ppmd_bits_per_byte': ppmd['bits_per_byte'],
        'mix_minus_gru': payload['gaps']['mix_minus_gru'],
        'mix_minus_ppmd': payload['gaps']['mix_minus_ppmd'],
        'gru_minus_h26_uniform_bits': payload['gru_minus_h26_uniform_bits'],
        'best_count_order': got['mixer']['best_count_order'],
        'replay_steps': gru['replay_steps'],
        'session_wall_seconds': session,
        'gpu_name': gpu_name,
        'finite': payload['finite'],
        'sha256_e3m': digest,
    }), flush=True)
    if not payload['finite']:
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


def write_report(scored, smoke_payload, meta, out_dir):
    out_dir = Path(out_dir)
    gru = scored['gru']
    arms = scored['mixer']['arms']
    ppmd = scored['ppmd']
    order = [
        ('gru', gru),
        ('count_1', arms['count_1']),
        ('count_2', arms['count_2']),
        ('count_3', arms['count_3']),
        ('count_4', arms['count_4']),
        ('mix', arms['mix']),
        ('ppmd_order_16', ppmd),
    ]
    gaps = scored['gaps']
    w = scored['mixer']['W']
    shares = scored['mixer']['s_share']
    counts = scored['mixer']['s_counts']
    elapsed = meta.get('log_elapsed_seconds')
    elapsed_source = 'log_clock' if elapsed is not None else 'session_wall'
    if elapsed is None:
        elapsed = float(scored['session_wall_seconds'])
    gpu_hours = float(elapsed) / 3600.0
    lines = [
        '# H32 cerebellar mixer on enwik8 E3M',
        '',
        f"E3M bytes: {E3M}. sha256: `{scored['sha256_e3m']}`.",
        'GRU bits/byte = sum of live −log2 p / 3000000. The mixer and the count',
        'models are scored the same way. PPMd bits/byte = 8 × compressed bytes / 3000000.',
        'Replay is not part of the code length. The mixer gradient does not enter the cortex.',
        '',
        'Cortex: H26 uniform, lr 0.0003, seed 0, replay seed 30000, hidden 1024,',
        'chunk 128, replay batch 8, sleep block 1. Mixer: Adam lr 0.003, one step',
        'per chunk, W initialised to [1, 0, 0, 0, 0] on every row. Count smoothing 0.05.',
        '',
        '## E3M',
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
        f"mix − gru = {_fmt(gaps['mix_minus_gru'])} "
        f"({_fmt(arms['mix']['bits_per_byte'])} − {_fmt(gru['bits_per_byte'])}).",
        f"mix − PPMd order 16 = {_fmt(gaps['mix_minus_ppmd'])} "
        f"({_fmt(arms['mix']['bits_per_byte'])} − {_fmt(ppmd['bits_per_byte'])}).",
        '',
        'Bar 1, not scored here: mix − gru ≤ −0.05 bits/byte on E3M.',
        'Bar 2, not scored here: mix − PPMd(order 16, E3M) ≤ −0.05 bits/byte.',
        '',
        f"Best single count order: {scored['mixer']['best_count_order']} "
        f"at {_fmt(scored['mixer']['best_count_bits_per_byte'])} bits/byte.",
        '',
        '## Mixer weights',
        '',
        'Rows are weight sets s = 0..4. Columns are experts '
        '[gru, count_1, count_2, count_3, count_4].',
        '',
        '| s | w_gru | w_1 | w_2 | w_3 | w_4 | Byte share | Bytes |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for s in range(N_EXPERTS):
        row = w[s]
        lines.append(
            f"| {s} | {_fmt(row[0], 6)} | {_fmt(row[1], 6)} | {_fmt(row[2], 6)} | "
            f"{_fmt(row[3], 6)} | {_fmt(row[4], 6)} | {_fmt(shares[s], 6)} | {counts[s]} |")
    lines += [
        '',
        '## Cortex check',
        '',
        f"H26 uniform E3M bits: {H26_UNIFORM_E3M_BITS:.12f}.",
        f"This gru bits: {gru['bits']:.12f}.",
        f"Difference: {scored['gru_minus_h26_uniform_bits']:.6e}.",
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
        f"| PPMd wall seconds | {ppmd['wall_seconds']:.3f} |",
        f"| GPU hours | {gpu_hours:.4f} |",
        f"| Replay steps | {gru['replay_steps']} |",
        '',
        '## Smoke',
        '',
        'CPU. Not a score.',
        '',
        f"Bytes: {smoke_payload['n_bytes']}.",
        f"H26 uniform bits: {smoke_payload['h26_uniform_bits']:.12f}.",
        f"gru bits: {smoke_payload['gru_bits']:.12f}.",
        f"Absolute difference: {smoke_payload['abs_diff_h26']:.3e}.",
        f"Within 1e-6: {smoke_payload['within_1e-6_h26']}.",
        '',
        f"Frozen mix bits: {smoke_payload['mix_frozen_bits']:.12f}.",
        f"Absolute difference versus gru: {smoke_payload['abs_diff_frozen']:.3e}.",
        f"Bits/byte difference: {smoke_payload['abs_diff_frozen_bits_per_byte']:.3e}.",
        f"Within 1e-6: {smoke_payload['within_1e-6_frozen']}.",
        '',
        'Count check, last byte of abracadabraabracadab, order 2, context da.',
        f"N_2 = {smoke_payload['count_check']['N_2']}, "
        f"n_2(b) = {smoke_payload['count_check']['n_2_b']}.",
        f"Hand p_2 = {smoke_payload['count_check']['hand_p_2']:.16f}.",
        f"Code p_2 = {smoke_payload['count_check']['code_p_2']:.16f}.",
        '',
        'Finite:',
        '',
    ]
    for name, flag in smoke_payload['finite'].items():
        lines.append(f'- {name}: {flag}')
    lines += [
        '',
        '## Protocol',
        '',
        '- One online pass. The cortex is H26 arm uniform: score the chunk, one Adam',
        '  step on its own loss, then one replay step of 8 windows. Seed 0, replay seed 30000.',
        '- Cortex log p is log_softmax of the scoring logits, detached. Natural log.',
        '- Count models: p(b) = (n + 0.05) / (N + 0.05 · 256). Orders 1–4. Exact dict counts.',
        '- A byte with fewer than k previous bytes uses the empty context.',
        '- count_k is per-byte prequential: the count is added after that byte is scored.',
        '- The mixer scores a 128-byte chunk with counts from before the chunk, with',
        '  p_mix = softmax(Σ_j w_{s,j} log p_j). s is the highest order whose context',
        '  has N ≥ 2, else 0. Then one Adam step on W (lr 0.003, mean −log2 p_mix).',
        '  Then the chunk is added to the counts.',
        '- W is 5 × 5, every row initialised to [1, 0, 0, 0, 0].',
        '- PPMd is pyppmd variant H (Ppmd7), order 16, mem_size 1073741824, raw stream',
        '  plus flush, the H29 call. Slice lengths are incremental encodes on that encoder;',
        '  flush bytes are on the last slice. The incremental blob matches the one-shot blob.',
        '- E3M is bytes [0, 3000000). 1 MB slices are 1000000 bytes.',
        '- Code is h32_mixer.py. Private dataset ezzzzzx/pdp-code-h32. Data dataset ezzzzzx/enwik8.',
        '',
    ]
    text = '\n'.join(lines)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'RESULTS.md').write_text(text)
    results = {
        'experiment': 'h32',
        'n_bytes': E3M,
        'sha256_e3m': scored['sha256_e3m'],
        'kernel_id': scored['kernel_id'],
        'kernel_version': meta.get('version'),
        'kernel_status': meta.get('status'),
        'gpu_name': scored.get('gpu_name'),
        'torch_version': scored.get('torch_version'),
        'elapsed_seconds': elapsed,
        'elapsed_source': elapsed_source,
        'session_wall_seconds': scored['session_wall_seconds'],
        'stream_wall_seconds': scored['stream_wall_seconds'],
        'gpu_hours': gpu_hours,
        'arms': {name: row for name, row in order},
        'gaps': gaps,
        'best_count_order': scored['mixer']['best_count_order'],
        'best_count_bits_per_byte': scored['mixer']['best_count_bits_per_byte'],
        'W': w,
        's_counts': counts,
        's_share': shares,
        'gru_minus_h26_uniform_bits': scored['gru_minus_h26_uniform_bits'],
        'h26_uniform_e3m_bits': H26_UNIFORM_E3M_BITS,
        'smoke': smoke_payload,
        'finite': scored['finite'],
    }
    dump(out_dir / 'results.json', results)
    return results


def assemble(out_dir, kernel_json, smoke_json, log_path, version, status):
    scored = json.loads(Path(kernel_json).read_text())
    smoke_payload = json.loads(Path(smoke_json).read_text())
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
    ap.add_argument('--out', default='results/h32')
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--kernel-json')
    ap.add_argument('--smoke-json')
    ap.add_argument('--log')
    ap.add_argument('--kernel-version', type=int)
    ap.add_argument('--kernel-status')
    args = ap.parse_args()
    if args.count_check:
        count_check()
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
