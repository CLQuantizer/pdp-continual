"""H29: one prequential pass of the cortex over enwik8. Numbers only. No verdict.

Principle 2 (improve the compressor): passive calibration, no curiosity.
Creates only this file, kaggle/h29/* and results/h29/*. Does not commit.

Protocol, fixed by handoffs/2026-11-10_grok_h29_enwik8.md:
- Byte alphabet 256. GRU embedding 64, hidden 1024, 1 layer.
- Adam, grad clip 1.0 (global norm). Learning rate 1e-3 or 3e-3, chosen on
  the first 1_000_000 bytes (with replay); the lower bits/byte wins, ties
  keep 1e-3. The scored run is a fresh model.
- Chunks of 128 bytes. Truncated BPTT 128. Hidden state is carried across
  live chunks and detached at each boundary. The first live input is a zero
  embedding (no start symbol); later chunks take the true previous byte.
- Each byte's -log2 p is recorded before that chunk's Adam step.
- Then one replay step: 32 windows of 128 bytes, starts uniform on
  {0, ..., L-128}, with replacement, zero hidden state, true previous byte
  (zero embedding at position 0). This is H27b hist_e1 scaled (chunk 128,
  batch 32, one step per chunk).
- Code length is the sum of those live -log2 p values only.
- No-replay uses the chosen learning rate on the first 2_000_000 bytes.
- Score is the first 10_000_000 bytes (E10M). Full enwik8 is a stretch only
  when the E10M pass took under 1 hour and the remaining pass fits in the
  3 GPU-hour budget.
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


def kernel_main(out='/kaggle/working'):
    configure()
    sanity_checks()
    path = find_kaggle_enwik8()
    print(f'H29 data {path} bytes={path.stat().st_size}', flush=True)
    raw = load_enwik8(path)
    score_gpu(raw, out)


def _fmt(value, digits=6):
    return f'{value:.{digits}f}'


def write_report(out_dir):
    out_dir = Path(out_dir)
    smoke = json.loads((out_dir / 'smoke.json').read_text())
    base = json.loads((out_dir / 'baselines.json').read_text())
    gru = json.loads((out_dir / 'h29_gru.json').read_text())
    kernel = {}
    kernel_path = out_dir / 'kernel.json'
    if kernel_path.exists():
        kernel = json.loads(kernel_path.read_text())
    if gru['sha256_file'] != base['sha256_file']:
        raise SystemExit('enwik8 sha256 differs between the VPS file and the kernel')
    if gru['sha256_e10m'] != base['sha256_e10m']:
        raise SystemExit('E10M sha256 differs between the VPS file and the kernel')
    e10 = gru['e10m']
    methods = {row['method']: row for row in base['methods']}
    xz = methods['xz -9e -T1']
    ppmd_rows = [row for row in base['methods'] if row['method'].startswith('ppmd ')]
    better_ppmd = min(ppmd_rows, key=lambda row: row['bits_per_byte'])
    last = e10['slices'][-1]
    if last['start'] != 9_000_000 or last['bytes'] != 1_000_000:
        raise SystemExit(f'last E10M slice is not the last 1 MB: {last}')
    replay_2 = gru['replay_2mb']
    noreplay_2 = gru['noreplay_2mb']
    lines = [
        '# H29 enwik8',
        '',
        f"File bytes: {gru['file_bytes']}. sha256: `{gru['sha256_file']}`.",
        f"E10M bytes: {E10M}. sha256: `{gru['sha256_e10m']}`.",
        f"Prefix starts: `{gru['prefix_repr']}`.",
        '',
        '## E10M bits/byte',
        '',
        '| Method | Compressed bytes | Bits/byte | Wall seconds |',
        '|---|---:|---:|---:|',
    ]
    for row in base['methods']:
        lines.append(
            f"| {row['method']} | {row['compressed_bytes']} | {_fmt(row['bits_per_byte'])} | "
            f"{row['wall_seconds']:.3f} |")
    lines.append(
        f"| GRU replay lr {gru['chosen_lr']:g} | prequential | {_fmt(e10['bits_per_byte'])} | "
        f"{e10['wall_seconds']:.3f} |")
    lines += [
        '',
        'Baseline bits/byte = 8 × compressed bytes / 10000000, container included, round-trip checked.',
        'GRU bits/byte = sum of live −log2 p / 10000000. Replay is not part of the code length.',
        '',
        '## GRU learning curve, E10M, 1 MB slices',
        '',
        '| Slice | Start | End | Bits | Bits/byte |',
        '|---:|---:|---:|---:|---:|',
    ]
    for sl in e10['slices']:
        lines.append(
            f"| {sl['index']} | {sl['start']} | {sl['end']} | {_fmt(sl['bits'], 3)} | "
            f"{_fmt(sl['bits_per_byte'])} |")
    lines += [
        '',
        f"Last 1 MB (bytes 9000000:10000000): {_fmt(last['bits_per_byte'])} bits/byte.",
        '',
        '## Learning-rate probe, first 1 MB, replay on',
        '',
        '| lr | Bits/byte | Wall seconds | Replay steps |',
        '|---:|---:|---:|---:|',
    ]
    for row in gru['probes']:
        lines.append(
            f"| {row['lr']:g} | {_fmt(row['bits_per_byte'])} | {row['wall_seconds']:.3f} | "
            f"{row['replay_steps']} |")
    lines.append(f"Chosen lr: {gru['chosen_lr']:g}.")
    lines += [
        '',
        '## Replay versus no-replay, first 2 MB, chosen lr',
        '',
        '| Variant | Bits/byte | Bits | Wall seconds | Replay steps |',
        '|---|---:|---:|---:|---:|',
        f"| replay (prefix of the E10M run) | {_fmt(replay_2['bits_per_byte'])} | "
        f"{_fmt(replay_2['bits'], 3)} | (inside the E10M pass) | {replay_2['replay_steps']} |",
        f"| no-replay | {_fmt(noreplay_2['bits_per_byte'])} | {_fmt(noreplay_2['bits'], 3)} | "
        f"{noreplay_2['wall_seconds']:.3f} | {noreplay_2['replay_steps']} |",
        '',
        f"Difference (replay − no-replay) bits/byte: "
        f"{_fmt(replay_2['bits_per_byte'] - noreplay_2['bits_per_byte'])}.",
        '',
        '## Published full-enwik8 references (Mahoney LTCB)',
        '',
        'Not measured here. Full-file numbers; an E10M number is on less data.',
        '',
        '| Compressor | Bits/byte |',
        '|---|---:|',
    ]
    for name, value in PUBLISHED_FULL:
        lines.append(f'| {name} | {value:.3f} |')
    lines += [
        '',
        f"GRU last 1 MB of E10M: {_fmt(last['bits_per_byte'])}.",
        '',
        '## Bar inputs',
        '',
        'Pre-registered. Not scored in this file.',
        '',
        f"1. GRU E10M − xz -9e E10M = {_fmt(e10['bits_per_byte'] - xz['bits_per_byte'])} "
        f"({_fmt(e10['bits_per_byte'])} − {_fmt(xz['bits_per_byte'])}).",
        f"2. GRU E10M − better PPMd ({better_ppmd['method']}) = "
        f"{_fmt(e10['bits_per_byte'] - better_ppmd['bits_per_byte'])} "
        f"({_fmt(e10['bits_per_byte'])} − {_fmt(better_ppmd['bits_per_byte'])}).",
        f"3. Last 1 MB {_fmt(last['bits_per_byte'])}. "
        f"Replay 2 MB {_fmt(replay_2['bits_per_byte'])}. "
        f"No-replay 2 MB {_fmt(noreplay_2['bits_per_byte'])}.",
        '',
        '## Compute',
        '',
        '| Item | Value |',
        '|---|---|',
        f"| Kernel | {kernel.get('id', gru['kernel_id'])} |",
        f"| Kernel version | {kernel.get('version', '')} |",
        f"| GPU | {gru['gpu_name']} |",
        f"| Session wall seconds | {gru['session_wall_seconds']:.3f} |",
        f"| GPU hours (session wall / 3600) | {gru['gpu_hours_script']:.4f} |",
        f"| E10M wall seconds | {e10['wall_seconds']:.3f} |",
        f"| Probe walls seconds | {', '.join(f'{row['wall_seconds']:.3f}' for row in gru['probes'])} |",
        f"| No-replay wall seconds | {noreplay_2['wall_seconds']:.3f} |",
        f"| Stretch | {gru['stretch_reason'] or 'ran'} |",
        f"| Parameters | {gru['parameters']} |",
        f"| Torch | {gru['torch_version']} |",
        '',
        '## Smoke, first 100000 bytes, CPU',
        '',
        '| Bits/byte | Wall seconds | Replay steps | Finite | Device |',
        '|---:|---:|---:|---|---|',
        f"| {_fmt(smoke['bits_per_byte'])} | {smoke['wall_seconds']:.3f} | {smoke['replay_steps']} | "
        f"{smoke['finite']} | {smoke['device']} |",
        '',
        '## Protocol',
        '',
        '- One online pass. Prequential: −log2 p of each byte is recorded before that chunk is trained.',
        '- GRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam, clip 1.0. Seed 0. Replay RNG 30000.',
        '- Live state is carried and detached every 128 bytes. Replay state is zero. Replay batch 32.',
        '- Replay starts are uniform on 0 .. L−128 inclusive, with replacement.',
        f"- Chosen learning rate {gru['chosen_lr']:g} from the two 1 MB probes. Scored model is fresh.",
        '- E10M is bytes [0, 10000000). 1 MB slices are 1000000 bytes.',
        '- No-replay is a fresh model at the chosen learning rate on bytes [0, 2000000).',
        '- PPMd is pyppmd variant H (Ppmd7), orders 6 and 16, mem_size 1073741824, raw stream plus flush.',
        '- Smoke is not a score.',
        '',
    ]
    if gru['stretch'] is not None:
        full = gru['stretch']
        lines += [
            '## Stretch, full enwik8',
            '',
            f"Bits/byte: {_fmt(full['bits_per_byte'])}. Bits: {_fmt(full['bits'], 3)}. "
            f"Wall seconds (including the E10M prefix): {full['wall_seconds']:.3f}.",
            '',
            '| Slice | Bits/byte |',
            '|---:|---:|',
        ]
        for sl in full['slices']:
            lines.append(f"| {sl['index']} | {_fmt(sl['bits_per_byte'])} |")
        lines.append('')
    text = '\n'.join(lines)
    (out_dir / 'RESULTS.md').write_text(text)
    merged = {
        'sha256_file': gru['sha256_file'],
        'sha256_e10m': gru['sha256_e10m'],
        'file_bytes': gru['file_bytes'],
        'baselines': base['methods'],
        'probes': gru['probes'],
        'chosen_lr': gru['chosen_lr'],
        'e10m': e10,
        'replay_2mb': replay_2,
        'noreplay_2mb': noreplay_2,
        'last_1mb_bits_per_byte': last['bits_per_byte'],
        'published_full_enwik8_ltcb': [
            {'name': name, 'bits_per_byte': value} for name, value in PUBLISHED_FULL
        ],
        'bar_inputs': {
            'gru_minus_xz': e10['bits_per_byte'] - xz['bits_per_byte'],
            'gru_minus_better_ppmd': e10['bits_per_byte'] - better_ppmd['bits_per_byte'],
            'better_ppmd': better_ppmd['method'],
        },
        'stretch': gru['stretch'],
        'stretch_reason': gru['stretch_reason'],
        'kernel': {
            'id': kernel.get('id', gru['kernel_id']),
            'version': kernel.get('version'),
            'gpu_name': gru['gpu_name'],
            'session_wall_seconds': gru['session_wall_seconds'],
            'gpu_hours_script': gru['gpu_hours_script'],
        },
        'smoke': smoke,
        'architecture': gru['architecture'],
        'seed': SEED,
        'replay_seed': REPLAY_SEED + SEED,
    }
    dump(out_dir / 'results.json', merged)
    print(f'H29 wrote {out_dir / "RESULTS.md"} and results.json', flush=True)


def main():
    configure()
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--baselines', action='store_true')
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--check', type=int, default=0)
    ap.add_argument('--data', default='data/enwik8/enwik8')
    ap.add_argument('--out', default='results/h29')
    args = ap.parse_args()
    out = Path(args.out)
    if args.assemble:
        write_report(out)
        return
    if args.smoke or args.baselines or args.check:
        sanity_checks()
    raw = None
    if args.smoke or args.baselines or args.check:
        raw = load_enwik8(args.data)
        print(
            f'H29 file bytes={raw.size} sha256={sha256_arr(raw)} '
            f'e10m_sha256={sha256_arr(raw[:E10M])}',
            flush=True)
    if args.check:
        n = int(args.check)
        t0 = time.monotonic()
        row = run_cpu(raw, n, 1e-3, True, 'check')
        print(
            f'H29 check n={n} bpb={row["bits_per_byte"]:.6f} '
            f'wall_s={time.monotonic() - t0:.1f} per_chunk_s='
            f'{(time.monotonic() - t0) / math.ceil(n / CHUNK):.3f}',
            flush=True)
    if args.smoke:
        row = run_cpu(raw, SMOKE_N, 1e-3, True, 'smoke')
        row['sha256_file'] = sha256_arr(raw)
        row['n_requested'] = SMOKE_N
        dump(out / 'smoke.json', row)
        print(
            f'H29 smoke n={row["n"]} bpb={row["bits_per_byte"]:.6f} '
            f'wall_s={row["wall_seconds"]:.1f} replay_steps={row["replay_steps"]}',
            flush=True)
    if args.baselines:
        baselines(raw, out)


if __name__ == '__main__':
    main()
