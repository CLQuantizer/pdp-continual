"""H29d: H29b with a --seed flag. Numbers only. No verdict.

Principle 2 (improve the compressor), with store-everything replay (principle 1).
Same model, chunking, score-before-step order, clipping, replay-window sampling,
and live -log2 p code length as H29b. --seed sets SEED. The torch init seed is
SEED and the replay sampler seed is REPLAY_SEED + SEED. Nothing else differs.

Does not commit.
"""
import argparse
import hashlib
import json
import lzma
import math
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ALPHABET = 256
EMB = 64
HIDDEN = 1024
CHUNK = 128
REPLAY_BATCH = 32
CLIP = 1.0
LR_TRIALS = (1e-3, 3e-3)
SEED = 0
REPLAY_SEED = 30_000
PROBE = 1_000_000
NOREPLAY_N = 2_000_000
E10M = 10_000_000
FULL = 100_000_000
SLICE = 1_000_000
SMOKE_N = 100_000
BUDGET_S = 3 * 3600
PARAMS = 3_627_264
LOG2 = math.log(2)
OFFSETS = np.arange(CHUNK, dtype=np.int64)[:, None]
KERNEL_ID = 'ezzzzzx/h29-enwik8'
E10M_SHA256 = '5985c81c39d927ae0e169625790ca4d9e7d1531270c8b09ad73176a375bb3d97'
XZ_BPB = 2.176205
PPMD16_BPB = 1.837254
ARM_SPEC = (
    ('nr_1e3', 1e-3, 0),
    ('r8_1e3', 1e-3, 8),
    ('nr_3e4', 3e-4, 0),
)
PUBLISHED_FULL = (
    ('xz', 1.976),
    ('ppmd J1', 1.711),
    ('cmix v21', 1.170),
)


def configure():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha256_arr(raw):
    return hashlib.sha256(raw.tobytes()).hexdigest()


def load_enwik8(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size != FULL:
        raise SystemExit(f'{path} has {raw.size} bytes, expected {FULL}')
    return raw


class ByteGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(ALPHABET, EMB)
        self.gru = nn.GRU(EMB, HIDDEN)
        self.readout = nn.Linear(HIDDEN, ALPHABET)

    def nll(self, prev, targets, state):
        # -1 is an absent predecessor, not a 257th symbol.
        emb = self.embedding(prev.clamp_min(0))
        emb = emb.masked_fill((prev < 0).unsqueeze(-1), 0)
        y, state = self.gru(emb, state)
        logits = self.readout(y)
        losses = F.cross_entropy(
            logits.reshape(-1, ALPHABET), targets.reshape(-1), reduction='none')
        return losses, state


def deposit(slice_sums, start, values):
    pos = 0
    n = len(values)
    while pos < n:
        idx = (start + pos) // SLICE
        room = SLICE - ((start + pos) % SLICE)
        take = min(room, n - pos)
        slice_sums[idx] += float(values[pos:pos + take].sum())
        pos += take


def sample_replay(raw, n_seen, rng):
    """Uniform over starts that have a full 128-byte window. With replacement."""
    high = n_seen - CHUNK + 1
    starts = rng.integers(0, high, size=REPLAY_BATCH, dtype=np.int64)
    idx = starts[None, :] + OFFSETS
    targets = raw[idx].astype(np.int64)
    prev = np.empty(idx.shape, dtype=np.int64)
    at_zero = idx == 0
    if np.any(at_zero):
        prev[at_zero] = -1
        keep = ~at_zero
        prev[keep] = raw[idx[keep] - 1]
    else:
        prev[:] = raw[idx - 1]
    return prev, targets


class StreamRun:
    def __init__(self, lr, replay, device, phase):
        self.lr = float(lr)
        self.replay = bool(replay)
        self.device = device
        self.phase = phase
        self.t0 = time.monotonic()
        torch.manual_seed(SEED)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(SEED)
        self.model = ByteGRU().to(device)
        self.model.train()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.rng = np.random.default_rng(REPLAY_SEED + SEED)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if self.n_params != PARAMS:
            raise AssertionError(f'parameter count {self.n_params} != {PARAMS}')
        self.slice_sums = np.zeros(0, dtype=np.float64)
        self.total_bits = 0.0
        self.n_done = 0
        self.replay_steps = 0
        self.state = None
        self.first_chunk_bits = None
        self.second_chunk_bits = None
        self._chunk_index = 0
        self._log_mark = -1
        self._reported_slices = 0

    def _ensure(self, n_target):
        need = (n_target + SLICE - 1) // SLICE
        if len(self.slice_sums) < need:
            extra = np.zeros(need - len(self.slice_sums), dtype=np.float64)
            self.slice_sums = np.concatenate((self.slice_sums, extra))

    def _step(self, losses):
        self.opt.zero_grad(set_to_none=True)
        losses.mean().backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        self.opt.step()

    def _log(self, end, n_target):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        running = self.total_bits / end
        done = end // SLICE
        slice_bpb = ''
        if done > 0 and end % SLICE == 0:
            slice_bpb = f' slice_bpb={self.slice_sums[done - 1] / SLICE:.6f}'
        elif end == n_target and end % SLICE != 0:
            idx = (end - 1) // SLICE
            nbytes = end - idx * SLICE
            slice_bpb = f' slice_bpb={self.slice_sums[idx] / nbytes:.6f}'
        print(
            f'H29 phase={self.phase} n={end}/{n_target} bpb={running:.6f}'
            f'{slice_bpb} replay_steps={self.replay_steps} '
            f'wall_s={time.monotonic() - self.t0:.1f}',
            flush=True)

    def advance(self, raw, n_target):
        if n_target > len(raw):
            raise AssertionError(f'n_target {n_target} exceeds file {len(raw)}')
        if n_target < self.n_done:
            raise AssertionError('advance would rewind the stream')
        self._ensure(n_target)
        while self.n_done < n_target:
            start = self.n_done
            end = min(start + CHUNK, n_target)
            tlen = end - start
            if start == 0:
                prev_np = np.empty(tlen, dtype=np.int64)
                prev_np[0] = -1
                if tlen > 1:
                    prev_np[1:] = raw[:tlen - 1]
            else:
                prev_np = raw[start - 1:end - 1].astype(np.int64, copy=False)
            tgt_np = raw[start:end].astype(np.int64, copy=False)
            prev_t = torch.tensor(prev_np, dtype=torch.long, device=self.device).unsqueeze(1)
            tgt_t = torch.tensor(tgt_np, dtype=torch.long, device=self.device).unsqueeze(1)
            # Score this chunk under the weights and state from before the update.
            losses, state = self.model.nll(prev_t, tgt_t, self.state)
            bit_vals = (losses.detach().double() / LOG2).cpu().numpy()
            if not np.isfinite(bit_vals).all():
                raise FloatingPointError(f'{self.phase}: nonfinite bits at {start}')
            if self._chunk_index == 0:
                self.first_chunk_bits = bit_vals.copy()
            elif self._chunk_index == 1:
                self.second_chunk_bits = bit_vals.copy()
            deposit(self.slice_sums, start, bit_vals)
            self.total_bits += float(bit_vals.sum())
            self.state = state.detach()
            self._step(losses)
            del losses
            if self.replay and end >= CHUNK:
                r_prev, r_tgt = sample_replay(raw, end, self.rng)
                r_prev_t = torch.tensor(r_prev, dtype=torch.long, device=self.device)
                r_tgt_t = torch.tensor(r_tgt, dtype=torch.long, device=self.device)
                r_losses, _ = self.model.nll(r_prev_t, r_tgt_t, None)
                self._step(r_losses)
                del r_losses
                self.replay_steps += 1
            self._chunk_index += 1
            self.n_done = end
            mark = end // 200_000
            if mark > self._log_mark or end == n_target:
                self._log_mark = mark
                while self._reported_slices < end // SLICE:
                    i = self._reported_slices
                    print(
                        f'H29 phase={self.phase} slice={i} '
                        f'slice_bpb={self.slice_sums[i] / SLICE:.6f} '
                        f'n={end}/{n_target} bpb={self.total_bits / end:.6f}',
                        flush=True)
                    self._reported_slices += 1
                self._log(end, n_target)
        gap = abs(self.total_bits - float(self.slice_sums.sum()))
        if gap > 1e-4:
            raise AssertionError(f'slice sums drifted by {gap}')
        return self.result()

    def result(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        n = int(self.n_done)
        slices = []
        for i in range((n + SLICE - 1) // SLICE):
            start = i * SLICE
            end = min(start + SLICE, n)
            bits = float(self.slice_sums[i])
            slices.append({
                'index': i,
                'start': start,
                'end': end,
                'bytes': end - start,
                'bits': bits,
                'bits_per_byte': bits / (end - start),
            })
        total = float(self.total_bits)
        replay_tokens = int(self.replay_steps * REPLAY_BATCH * CHUNK)
        return {
            'phase': self.phase,
            'lr': self.lr,
            'replay': self.replay,
            'seed': SEED,
            'replay_seed': REPLAY_SEED + SEED,
            'n': n,
            'bits': total,
            'bits_per_byte': total / n,
            'slices': slices,
            'replay_steps': int(self.replay_steps),
            'replay_tokens': replay_tokens,
            'traffic': replay_tokens / n,
            'wall_seconds': time.monotonic() - self.t0,
            'parameters': int(self.n_params),
            'device': str(self.device),
            'first_chunk_bits_per_byte': (
                None if self.first_chunk_bits is None else float(self.first_chunk_bits.mean())),
        }


def _reference_first_chunk(raw):
    torch.manual_seed(SEED)
    model = ByteGRU()
    prev = torch.empty(CHUNK, 1, dtype=torch.long)
    prev[0, 0] = -1
    prev[1:, 0] = torch.from_numpy(raw[:CHUNK - 1].astype(np.int64))
    tgt = torch.from_numpy(raw[:CHUNK].astype(np.int64))[:, None]
    with torch.no_grad():
        losses, _ = model.nll(prev, tgt, None)
    return (losses.double() / LOG2).numpy()


def sanity_checks():
    """Prequential order, carried state, and replay-between-chunks. CPU, 512 bytes."""
    sums = np.zeros(3, dtype=np.float64)
    deposit(sums, 999_900, np.ones(200, dtype=np.float64))
    if not (sums[0] == 100 and sums[1] == 100 and sums[2] == 0):
        raise AssertionError(f'deposit split failed: {sums}')
    probe = ByteGRU()
    n_params = sum(p.numel() for p in probe.parameters())
    if n_params != PARAMS:
        raise AssertionError(f'parameter count {n_params} != {PARAMS}')
    del probe
    raw = (np.arange(512) % 256).astype(np.uint8)
    ref = _reference_first_chunk(raw)
    cpu = torch.device('cpu')
    replay = StreamRun(1e-3, True, cpu, 'sanity-replay')
    replay.advance(raw, 256)
    noreplay = StreamRun(1e-3, False, cpu, 'sanity-noreplay')
    noreplay.advance(raw, 256)
    frozen_r = StreamRun(0.0, True, cpu, 'sanity-frozen-replay')
    frozen_r.advance(raw, 256)
    frozen_n = StreamRun(0.0, False, cpu, 'sanity-frozen-noreplay')
    frozen_n.advance(raw, 256)
    if replay.replay_steps != 2 or noreplay.replay_steps != 0:
        raise AssertionError(
            f'replay steps replay={replay.replay_steps} noreplay={noreplay.replay_steps}')
    if not np.allclose(replay.first_chunk_bits, ref, atol=1e-5):
        raise AssertionError('first chunk was not scored under the initial weights')
    if not np.allclose(replay.first_chunk_bits, noreplay.first_chunk_bits, atol=1e-5):
        raise AssertionError('replay changed the chunk that is scored before it runs')
    if abs(float(replay.second_chunk_bits.mean() - noreplay.second_chunk_bits.mean())) < 1e-6:
        raise AssertionError('replay did not change the next live chunk')
    if not np.allclose(frozen_r.first_chunk_bits, frozen_n.first_chunk_bits, atol=1e-5):
        raise AssertionError('lr 0 first chunks diverged')
    if not np.allclose(frozen_r.second_chunk_bits, frozen_n.second_chunk_bits, atol=1e-5):
        raise AssertionError('replay corrupted the carried live state')
    print(
        f'H29 sanity ok params={n_params} '
        f'first_bpb={float(ref.mean()):.4f} '
        f'replay_chunk1={float(replay.second_chunk_bits.mean()):.4f} '
        f'noreplay_chunk1={float(noreplay.second_chunk_bits.mean()):.4f}',
        flush=True)


def run_cpu(raw, n, lr, replay, phase):
    run = StreamRun(lr, replay, torch.device('cpu'), phase)
    run.advance(raw, n)
    out = run.result()
    out['finite'] = True
    return out


def _cmd(args, data):
    proc = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode('utf-8', 'replace')[:400]
        raise RuntimeError(f'{args} failed ({proc.returncode}): {err}')
    return proc.stdout


def _timed_baseline(name, fn):
    t0 = time.monotonic()
    size = fn()
    return {
        'method': name,
        'compressed_bytes': int(size),
        'bits_per_byte': 8.0 * int(size) / E10M,
        'wall_seconds': time.monotonic() - t0,
        'roundtrip': True,
    }


def baselines(raw, out_dir):
    prefix = raw[:E10M].tobytes()
    if len(prefix) != E10M:
        raise AssertionError('E10M prefix length')
    rows = []

    def xz():
        blob = _cmd(['xz', '-9e', '-T1', '-c'], prefix)
        if _cmd(['xz', '-dc'], blob) != prefix:
            raise RuntimeError('xz roundtrip failed')
        return len(blob)

    def zstd():
        blob = _cmd(['zstd', '--ultra', '-22', '--long=27', '-c'], prefix)
        if _cmd(['zstd', '-dc'], blob) != prefix:
            raise RuntimeError('zstd roundtrip failed')
        return len(blob)

    def bz2_9():
        import bz2
        blob = bz2.compress(prefix, compresslevel=9)
        if bz2.decompress(blob) != prefix:
            raise RuntimeError('bz2 roundtrip failed')
        return len(blob)

    def lzma_9e():
        blob = lzma.compress(prefix, preset=9 | lzma.PRESET_EXTREME)
        if lzma.decompress(blob) != prefix:
            raise RuntimeError('lzma roundtrip failed')
        return len(blob)

    def ppmd(order):
        import pyppmd
        mem = 1 << 30
        blob = pyppmd.compress(prefix, max_order=order, mem_size=mem, variant='H')
        back = pyppmd.Ppmd7Decoder(order, mem).decode(blob, len(prefix))
        if back != prefix:
            raise RuntimeError(f'ppmd H order {order} roundtrip failed')
        return len(blob)

    specs = [
        ('xz -9e -T1', xz),
        ('zstd --ultra -22 --long=27', zstd),
        ('bz2 level 9', bz2_9),
        ('lzma preset 9|EXTREME', lzma_9e),
        ('ppmd H order 6 mem 1073741824', lambda: ppmd(6)),
        ('ppmd H order 16 mem 1073741824', lambda: ppmd(16)),
    ]
    for name, fn in specs:
        print(f'H29 baseline start {name}', flush=True)
        row = _timed_baseline(name, fn)
        print(
            f'H29 baseline done {name} bytes={row["compressed_bytes"]} '
            f'bpb={row["bits_per_byte"]:.6f} wall_s={row["wall_seconds"]:.1f}',
            flush=True)
        rows.append(row)
    payload = {
        'n': E10M,
        'sha256_file': sha256_arr(raw),
        'sha256_e10m': sha256_arr(raw[:E10M]),
        'bits_per_byte_definition': '8 * compressed_bytes / 10000000',
        'methods': rows,
    }
    dump(Path(out_dir) / 'baselines.json', payload)
    return payload


def _pick_lr(probes):
    """Lower bits/byte wins. A tie keeps 1e-3, which is probes[0]."""
    best = min(probes, key=lambda row: (row['bits_per_byte'], row['lr']))
    return best


def score_gpu(raw, out_dir):
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the scored run')
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f'H29 device cuda name={gpu_name} torch={torch.__version__}', flush=True)
    t0 = time.monotonic()
    probes = []
    for lr in LR_TRIALS:
        phase = f'probe-lr{lr:g}'
        print(f'H29 phase={phase} start n_target={PROBE} replay=1', flush=True)
        run = StreamRun(lr, True, device, phase)
        run.advance(raw, PROBE)
        row = run.result()
        projected = row['wall_seconds'] * (E10M / PROBE)
        print(
            f'H29 phase={phase} done n={row["n"]} bpb={row["bits_per_byte"]:.6f} '
            f'wall_s={row["wall_seconds"]:.1f} projected_e10m_s={projected:.1f}',
            flush=True)
        probes.append(row)
        dump(Path(out_dir) / 'h29_gru_partial.json', {'probes': probes})
    chosen = _pick_lr(probes)
    lr = chosen['lr']
    print(f'H29 chosen_lr={lr:g} probe_bpb={chosen["bits_per_byte"]:.6f}', flush=True)
    print(f'H29 phase=score start n_target={E10M} lr={lr:g} replay=1', flush=True)
    score = StreamRun(lr, True, device, 'score')
    score.advance(raw, E10M)
    e10 = score.result()
    print(
        f'H29 phase=score done n={e10["n"]} bpb={e10["bits_per_byte"]:.6f} '
        f'replay_steps={e10["replay_steps"]} wall_s={e10["wall_seconds"]:.1f}',
        flush=True)
    dump(Path(out_dir) / 'h29_gru_partial.json', {'probes': probes, 'chosen_lr': lr, 'e10m': e10})
    print(f'H29 phase=noreplay start n_target={NOREPLAY_N} lr={lr:g} replay=0', flush=True)
    noreplay = StreamRun(lr, False, device, 'noreplay')
    noreplay.advance(raw, NOREPLAY_N)
    noreplay_row = noreplay.result()
    print(
        f'H29 phase=noreplay done n={noreplay_row["n"]} '
        f'bpb={noreplay_row["bits_per_byte"]:.6f} wall_s={noreplay_row["wall_seconds"]:.1f}',
        flush=True)
    used = time.monotonic() - t0
    # 15% slack so a slow tail does not run the session past the 3 hour budget.
    projected_rest = 9.0 * e10['wall_seconds'] * 1.15
    stretch = None
    stretch_reason = ''
    if e10['wall_seconds'] >= 3600:
        stretch_reason = (
            f'E10M wall {e10["wall_seconds"]:.1f}s is not under 3600s')
    elif used + projected_rest >= BUDGET_S:
        stretch_reason = (
            f'E10M wall {e10["wall_seconds"]:.1f}s < 3600s but projected remaining '
            f'{projected_rest:.1f}s (9x E10M plus 15%) plus used {used:.1f}s '
            f'does not fit in {BUDGET_S}s')

    def pack(session, stretch_row, reason):
        return {
            'kernel_id': KERNEL_ID,
            'gpu_name': gpu_name,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'sha256_file': sha256_arr(raw),
            'sha256_e10m': sha256_arr(raw[:E10M]),
            'file_bytes': int(raw.size),
            'prefix_repr': raw[:40].tobytes().decode('ascii', 'replace'),
            'seed': SEED,
            'replay_seed': REPLAY_SEED + SEED,
            'parameters': PARAMS,
            'architecture': {
                'embedding': EMB,
                'hidden': HIDDEN,
                'layers': 1,
                'alphabet': ALPHABET,
                'chunk': CHUNK,
                'replay_batch': REPLAY_BATCH,
                'grad_clip': CLIP,
                'optimizer': 'Adam',
            },
            'probes': probes,
            'chosen_lr': lr,
            'e10m': e10,
            'noreplay_2mb': noreplay_row,
            'replay_2mb': _prefix_result(e10, NOREPLAY_N),
            'stretch': stretch_row,
            'stretch_reason': reason,
            'session_wall_seconds': session,
            'gpu_hours_script': session / 3600.0,
            'budget_seconds': BUDGET_S,
        }

    # The scored E10M result is on disk before a stretch can be killed by the budget.
    session = time.monotonic() - t0
    payload = pack(
        session, None,
        stretch_reason or 'pre-stretch snapshot; the full file is not in this file yet')
    dump(Path(out_dir) / 'h29_gru.json', payload)
    if stretch_reason:
        print(f'H29 phase=stretch skip reason={stretch_reason}', flush=True)
    else:
        print(
            f'H29 phase=stretch start n_from={E10M} n_to={FULL} '
            f'projected_rest_s={projected_rest:.1f} used_s={used:.1f}',
            flush=True)
        score.advance(raw, FULL)
        stretch = score.result()
        for old, new in zip(e10['slices'], stretch['slices'][:len(e10['slices'])]):
            if abs(old['bits'] - new['bits']) > 1e-4:
                raise AssertionError('stretch rewrote E10M slice bits')
        session = time.monotonic() - t0
        payload = pack(session, stretch, '')
        dump(Path(out_dir) / 'h29_gru.json', payload)
        print(
            f'H29 phase=stretch done n={stretch["n"]} bpb={stretch["bits_per_byte"]:.6f} '
            f'wall_s={stretch["wall_seconds"]:.1f}',
            flush=True)
    if payload['session_wall_seconds'] > BUDGET_S + 60 and payload['stretch'] is not None:
        print(f'H29 budget overrun session_s={payload["session_wall_seconds"]:.1f}', flush=True)
    print('KERNEL_DONE ' + json.dumps({
        'kernel_id': KERNEL_ID,
        'chosen_lr': lr,
        'e10m_bpb': e10['bits_per_byte'],
        'e10m_wall_s': e10['wall_seconds'],
        'noreplay_2mb_bpb': noreplay_row['bits_per_byte'],
        'session_wall_seconds': payload['session_wall_seconds'],
        'stretch': payload['stretch'] is not None,
    }), flush=True)
    return payload


def _prefix_result(row, n):
    """Bits of a scored prefix, taken from 1 MB slices. n must be a multiple of SLICE."""
    if n % SLICE != 0:
        raise AssertionError('prefix slice requires a multiple of 1000000')
    need = n // SLICE
    bits = sum(s['bits'] for s in row['slices'][:need])
    return {
        'n': n,
        'bits': bits,
        'bits_per_byte': bits / n,
        'source': row['phase'],
        'lr': row['lr'],
        'replay': row['replay'],
        'replay_steps': (n // CHUNK) if row['replay'] else 0,
    }


def find_kaggle_enwik8():
    hits = [
        p for p in Path('/kaggle/input').rglob('*')
        if p.is_file() and p.name == 'enwik8' and p.stat().st_size == FULL
    ]
    if len(hits) != 1:
        raise SystemExit(f'expected one enwik8 file of {FULL} bytes, found {hits}')
    return hits[0]


def run_arm(raw, n_bytes, lr, replay_batch, arm, device):
    """One prequential pass. Batch 0 is no replay. StreamRun is the H29 step."""
    global REPLAY_BATCH
    replay_batch = int(replay_batch)
    n_bytes = int(n_bytes)
    if replay_batch < 0:
        raise SystemExit('replay-batch must be >= 0')
    if n_bytes <= 0 or n_bytes > len(raw):
        raise SystemExit(f'n-bytes {n_bytes} is outside 1..{len(raw)}')
    REPLAY_BATCH = replay_batch
    run = StreamRun(float(lr), replay_batch > 0, device, arm)
    run.advance(raw, n_bytes)
    out = run.result()
    finite = bool(np.isfinite(out['bits'])) and all(
        np.isfinite(sl['bits']) for sl in out['slices'])
    out['finite'] = finite
    out['replay_batch'] = replay_batch
    if not finite:
        raise FloatingPointError(f'{arm}: nonfinite bits')
    if n_bytes % CHUNK == 0 and n_bytes >= CHUNK:
        expect_steps = (n_bytes // CHUNK) if replay_batch > 0 else 0
        if out['replay_steps'] != expect_steps:
            raise AssertionError(
                f'{arm}: replay_steps {out["replay_steps"]} != {expect_steps}')
        expect_traffic = float(replay_batch) if replay_batch > 0 else 0.0
        if abs(out['traffic'] - expect_traffic) > 1e-9:
            raise AssertionError(
                f'{arm}: traffic {out["traffic"]} != {expect_traffic}')
    return out


def _pack(raw, digest, arm, lr, replay_batch, n_bytes, row, session, gpu_name, kernel_id):
    return {
        'arm': arm,
        'kernel_id': kernel_id,
        'lr': float(lr),
        'replay_batch': int(replay_batch),
        'n_bytes': int(n_bytes),
        'gpu_name': gpu_name,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'sha256_file': sha256_arr(raw),
        'sha256_e10m': digest,
        'file_bytes': int(raw.size),
        'seed': SEED,
        'replay_seed': REPLAY_SEED + SEED,
        'parameters': int(row['parameters']),
        'session_wall_seconds': session,
        'e10m': row,
    }


def kernel_main(out, lr, replay_batch, arm, n_bytes, kernel_id):
    configure()
    t0 = time.monotonic()
    global REPLAY_BATCH
    REPLAY_BATCH = 32
    sanity_checks()
    path = find_kaggle_enwik8()
    print(f'H29b data {path} bytes={path.stat().st_size}', flush=True)
    raw = load_enwik8(path)
    digest = sha256_arr(raw[:E10M])
    file_digest = sha256_arr(raw)
    print(f'H29b sha256_file={file_digest} sha256_e10m={digest}', flush=True)
    if digest != E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest} != {E10M_SHA256}')
    if int(n_bytes) != E10M:
        raise SystemExit(f'scored n-bytes is {n_bytes}, expected {E10M}')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the scored run')
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f'H29b arm={arm} device cuda name={gpu_name} torch={torch.__version__} '
        f'lr={float(lr):g} replay_batch={int(replay_batch)} n={int(n_bytes)}',
        flush=True)
    row = run_arm(raw, n_bytes, lr, replay_batch, arm, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    session = time.monotonic() - t0
    payload = _pack(
        raw, digest, arm, lr, replay_batch, n_bytes, row, session, gpu_name, kernel_id)
    dump(Path(out) / f'{arm}.json', payload)
    print('KERNEL_DONE ' + json.dumps({
        'kernel_id': kernel_id,
        'arm': arm,
        'lr': float(lr),
        'replay_batch': int(replay_batch),
        'n': row['n'],
        'bits': row['bits'],
        'bits_per_byte': row['bits_per_byte'],
        'replay_steps': row['replay_steps'],
        'traffic': row['traffic'],
        'wall_seconds': row['wall_seconds'],
        'session_wall_seconds': session,
        'gpu_name': gpu_name,
        'sha256_e10m': digest,
        'finite': row['finite'],
    }), flush=True)
    return payload


def _fmt(value, digits=6):
    return f'{value:.{digits}f}'


def _load_arm(out_dir, arm):
    path = Path(out_dir) / 'raw' / arm / f'{arm}.json'
    if not path.is_file():
        raise SystemExit(f'missing {path}')
    payload = json.loads(path.read_text())
    if payload.get('arm') != arm:
        raise SystemExit(f'{path} arm is {payload.get("arm")}')
    if payload.get('sha256_e10m') != E10M_SHA256:
        raise SystemExit(f'{arm} E10M sha256 {payload.get("sha256_e10m")}')
    row = payload['e10m']
    if row['n'] != E10M or not row['finite']:
        raise SystemExit(f'{arm} is not a finite E10M run')
    if len(row['slices']) != 10:
        raise SystemExit(f'{arm} has {len(row["slices"])} slices')
    for i, sl in enumerate(row['slices']):
        if sl['index'] != i or sl['bytes'] != SLICE or sl['start'] != i * SLICE:
            raise SystemExit(f'{arm} slice {i} is not a 1 MB block')
    return payload


def write_report(out_dir):
    out_dir = Path(out_dir)
    kernels = json.loads((out_dir / 'kernels.json').read_text())
    smoke = None
    smoke_path = out_dir / 'smoke.json'
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text())
    arms = []
    for order, (name, lr, batch) in enumerate(ARM_SPEC):
        payload = _load_arm(out_dir, name)
        if abs(payload['lr'] - lr) > 1e-12 or int(payload['replay_batch']) != batch:
            raise SystemExit(
                f'{name} ran lr={payload["lr"]} batch={payload["replay_batch"]}')
        row = payload['e10m']
        meta = kernels[name]
        elapsed = meta.get('log_elapsed_seconds')
        if elapsed is None:
            elapsed = payload['session_wall_seconds']
            elapsed_source = 'script_wall'
        else:
            elapsed_source = 'log_clock'
        arms.append({
            'order': order,
            'arm': name,
            'lr': payload['lr'],
            'replay_batch': int(payload['replay_batch']),
            'bits': row['bits'],
            'bits_per_byte': row['bits_per_byte'],
            'slices': row['slices'],
            'replay_steps': row['replay_steps'],
            'traffic': row['traffic'],
            'wall_seconds': row['wall_seconds'],
            'gpu_name': payload['gpu_name'],
            'kernel_id': meta.get('id', payload['kernel_id']),
            'kernel_version': meta.get('version'),
            'elapsed_seconds': elapsed,
            'elapsed_source': elapsed_source,
            'session_wall_seconds': payload['session_wall_seconds'],
            'log_elapsed_seconds': meta.get('log_elapsed_seconds'),
            'status': meta.get('status'),
            'url': meta.get('url'),
            'minus_xz': row['bits_per_byte'] - XZ_BPB,
            'minus_ppmd16': row['bits_per_byte'] - PPMD16_BPB,
            'torch_version': payload['torch_version'],
            'sha256_e10m': payload['sha256_e10m'],
            'parameters': payload['parameters'],
            'seed': payload['seed'],
            'replay_seed': payload['replay_seed'],
        })
    by_name = {row['arm']: row for row in arms}
    gpu_hours = [row['elapsed_seconds'] / 3600.0 for row in arms]
    lowest = min(arms, key=lambda row: (row['bits_per_byte'], row['order']))
    first = lowest['slices'][0]['bits_per_byte']
    last = lowest['slices'][-1]['bits_per_byte']
    lines = [
        '# H29b enwik8 dose',
        '',
        f'E10M bytes: {E10M}. sha256: `{E10M_SHA256}`.',
        'Baselines are the H29 measurements. They were not rerun.',
        f'xz -9e bits/byte: {XZ_BPB:.6f}. PPMd H order 16 bits/byte: {PPMD16_BPB:.6f}.',
        'GRU bits/byte = sum of live −log2 p / 10000000. Replay is not part of the code length.',
        '',
        '## E10M',
        '',
        '| Arm | lr | Replay batch | Bits/byte | Replay steps | Traffic | Wall seconds | GPU | Minus xz -9e | Minus PPMd order 16 |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|',
    ]
    for row in arms:
        lines.append(
            f"| {row['arm']} | {row['lr']:g} | {row['replay_batch']} | "
            f"{_fmt(row['bits_per_byte'])} | {row['replay_steps']} | {_fmt(row['traffic'])} | "
            f"{row['wall_seconds']:.3f} | {row['gpu_name']} | {_fmt(row['minus_xz'])} | "
            f"{_fmt(row['minus_ppmd16'])} |")
    lines += [
        '',
        'Minus xz -9e = arm bits/byte − 2.176205. Minus PPMd order 16 = arm bits/byte − 1.837254.',
        '',
    ]
    for row in arms:
        lines += [
            f"## {row['arm']} 1 MB slices",
            '',
            '| Slice | Start | End | Bits | Bits/byte |',
            '|---:|---:|---:|---:|---:|',
        ]
        for sl in row['slices']:
            lines.append(
                f"| {sl['index']} | {sl['start']} | {sl['end']} | {_fmt(sl['bits'], 3)} | "
                f"{_fmt(sl['bits_per_byte'])} |")
        lines.append('')
    h29_noreplay = Path(__file__).resolve().parent / 'results' / 'h29' / 'h29_gru.json'
    cross = None
    if h29_noreplay.is_file():
        h29_row = json.loads(h29_noreplay.read_text())['noreplay_2mb']
        cross = []
        for sl, old in zip(by_name['nr_1e3']['slices'][:2], h29_row['slices']):
            cross.append({
                'index': sl['index'],
                'h29b_bits': sl['bits'],
                'h29_bits': old['bits'],
                'difference': sl['bits'] - old['bits'],
            })
    lines += [
        '## Reported differences',
        '',
        'Pre-registered. Not scored in this file.',
        '',
        f"r8_1e3 − nr_1e3 = {_fmt(by_name['r8_1e3']['bits_per_byte'] - by_name['nr_1e3']['bits_per_byte'])} "
        f"({_fmt(by_name['r8_1e3']['bits_per_byte'])} − {_fmt(by_name['nr_1e3']['bits_per_byte'])}).",
        f"nr_3e4 − nr_1e3 = {_fmt(by_name['nr_3e4']['bits_per_byte'] - by_name['nr_1e3']['bits_per_byte'])} "
        f"({_fmt(by_name['nr_3e4']['bits_per_byte'])} − {_fmt(by_name['nr_1e3']['bits_per_byte'])}).",
        '',
        f"Lowest E10M bits/byte: {lowest['arm']} {_fmt(lowest['bits_per_byte'])}.",
        f"That arm, slice 0: {_fmt(first)}. Slice 9: {_fmt(last)}. "
        f"Slice 9 − slice 0: {_fmt(last - first)}.",
        '',
        '## Kernels',
        '',
        'Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that',
        'clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.',
        'Script wall is time.monotonic from kernel_main entry through the result write. Queue time',
        'is not included. Total GPU hours is the sum across the three kernels.',
        '',
        '| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours |',
        '|---|---:|---|---:|---|---:|---:|',
    ]
    for row in arms:
        lines.append(
            f"| {row['kernel_id']} | {row['kernel_version']} | {row['status']} | "
            f"{row['elapsed_seconds']:.3f} | {row['elapsed_source']} | "
            f"{row['session_wall_seconds']:.3f} | {row['elapsed_seconds'] / 3600.0:.4f} |")
    lines += [
        '',
        f"Total GPU hours: {_fmt(sum(gpu_hours), 4)}.",
        '',
        'Kaggle accepted two GPU sessions at a time. ezzzzzx/h29b-nr-3e4 version 1 was pushed after',
        'ezzzzzx/h29b-nr-1e3 reached COMPLETE. No kernel was rerun.',
        '',
        '## Protocol',
        '',
        '- One online pass per arm. Prequential: −log2 p of each byte is recorded before that chunk is trained.',
        '- GRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam, clip 1.0. Seed 0. Replay RNG 30000.',
        '- Live state is carried and detached every 128 bytes. Replay state is zero.',
        '- Replay starts are uniform on 0 .. L−128 inclusive, with replacement. Batch 0 skips that step.',
        '- E10M is bytes [0, 10000000). 1 MB slices are 1000000 bytes.',
        '- Traffic = replay steps × replay batch × 128 / bytes.',
        '- The same E10M bytes are the calibration. No held-out split.',
        '',
    ]
    if cross is not None:
        lines += [
            '## nr_1e3 against the H29 no-replay slices',
            '',
            'H29 no-replay, lr 0.001, first 2 MB. Slice bits, not rerun.',
            '',
            '| Slice | H29b bits | H29 bits | Difference |',
            '|---:|---:|---:|---:|',
        ]
        for row in cross:
            lines.append(
                f"| {row['index']} | {_fmt(row['h29b_bits'], 6)} | {_fmt(row['h29_bits'], 6)} | "
                f"{row['difference']:.3e} |")
        lines.append('')
    if smoke is not None:
        lines += [
            '## Smoke, first 4096 bytes, CPU',
            '',
            'Not a score.',
            '',
            f"h29 model path bits: {smoke['h29_bits']:.12f}.",
            f"h29b --lr 1e-3 --replay-batch 32 bits: {smoke['h29b_bits']:.12f}.",
            f"Absolute difference: {smoke['abs_diff']:.3e}.",
            '',
            '| Arm | Bits | Finite | Replay steps | Traffic |',
            '|---|---:|---|---:|---:|',
        ]
        for name, _lr, _batch in ARM_SPEC:
            item = smoke['arms'][name]
            lines.append(
                f"| {name} | {item['bits']:.6f} | {item['finite']} | "
                f"{item['replay_steps']} | {item['traffic']:.6f} |")
        lines.append('')
    text = '\n'.join(lines)
    (out_dir / 'RESULTS.md').write_text(text)
    merged = {
        'e10m_bytes': E10M,
        'sha256_e10m': E10M_SHA256,
        'baselines': {
            'xz_9e_bits_per_byte': XZ_BPB,
            'ppmd_h_order_16_bits_per_byte': PPMD16_BPB,
            'source': 'H29 measured values, not rerun',
        },
        'arms': {
            row['arm']: {
                'lr': row['lr'],
                'replay_batch': row['replay_batch'],
                'bits': row['bits'],
                'bits_per_byte': row['bits_per_byte'],
                'slices': [
                    {
                        'index': sl['index'],
                        'start': sl['start'],
                        'end': sl['end'],
                        'bits': sl['bits'],
                        'bits_per_byte': sl['bits_per_byte'],
                    }
                    for sl in row['slices']
                ],
                'replay_steps': row['replay_steps'],
                'traffic': row['traffic'],
                'wall_seconds': row['wall_seconds'],
                'gpu_name': row['gpu_name'],
                'minus_xz_9e': row['minus_xz'],
                'minus_ppmd_h_order_16': row['minus_ppmd16'],
                'kernel_id': row['kernel_id'],
                'kernel_version': row['kernel_version'],
                'elapsed_seconds': row['elapsed_seconds'],
                'elapsed_source': row['elapsed_source'],
                'session_wall_seconds': row['session_wall_seconds'],
                'log_elapsed_seconds': row['log_elapsed_seconds'],
                'kernel_status': row['status'],
                'torch_version': row['torch_version'],
                'parameters': row['parameters'],
                'seed': row['seed'],
                'replay_seed': row['replay_seed'],
            }
            for row in arms
        },
        'differences': {
            'r8_1e3_minus_nr_1e3': by_name['r8_1e3']['bits_per_byte'] - by_name['nr_1e3']['bits_per_byte'],
            'nr_3e4_minus_nr_1e3': by_name['nr_3e4']['bits_per_byte'] - by_name['nr_1e3']['bits_per_byte'],
        },
        'lowest_e10m': {
            'arm': lowest['arm'],
            'bits_per_byte': lowest['bits_per_byte'],
            'slice0_bits_per_byte': first,
            'slice9_bits_per_byte': last,
            'slice9_minus_slice0': last - first,
        },
        'gpu_hours_total': sum(gpu_hours),
        'nr_1e3_minus_h29_noreplay_slice_bits': cross,
        'smoke': smoke,
    }
    dump(out_dir / 'results.json', merged)
    print(f'H29b wrote {out_dir / "RESULTS.md"} and results.json', flush=True)


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--lr', type=float)
    ap.add_argument('--replay-batch', type=int)
    ap.add_argument('--arm')
    ap.add_argument('--n-bytes', type=int)
    ap.add_argument('--out')
    ap.add_argument('--data', default='data/enwik8/enwik8')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    SEED = int(args.seed)
    if args.assemble:
        write_report(args.out or 'results/h29b')
        return
    missing = [
        flag for flag, value in (
            ('--lr', args.lr),
            ('--replay-batch', args.replay_batch),
            ('--arm', args.arm),
            ('--n-bytes', args.n_bytes),
            ('--out', args.out),
        ) if value is None
    ]
    if missing:
        raise SystemExit('missing ' + ' '.join(missing))
    configure()
    raw = load_enwik8(args.data)
    digest = sha256_arr(raw[:E10M])
    if digest != E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest} != {E10M_SHA256}')
    print(
        f'H29b file bytes={raw.size} sha256_e10m={digest} arm={args.arm} '
        f'lr={float(args.lr):g} replay_batch={int(args.replay_batch)} n={int(args.n_bytes)} '
        f'seed={SEED} replay_seed={REPLAY_SEED + SEED}',
        flush=True)
    t0 = time.monotonic()
    row = run_arm(raw, args.n_bytes, args.lr, args.replay_batch, args.arm, torch.device('cpu'))
    session = time.monotonic() - t0
    payload = _pack(
        raw, digest, args.arm, args.lr, args.replay_batch, args.n_bytes, row, session,
        'cpu', None)
    dump(Path(args.out) / f'{args.arm}.json', payload)
    print(
        f'H29b arm={args.arm} bits={row["bits"]:.12f} bpb={row["bits_per_byte"]:.12f} '
        f'finite={row["finite"]} replay_steps={row["replay_steps"]} '
        f'traffic={row["traffic"]:.6f} wall_s={row["wall_seconds"]:.3f}',
        flush=True)


if __name__ == '__main__':
    main()
