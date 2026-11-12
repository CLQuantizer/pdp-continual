"""H31: thalamic sleep gate. Numbers only. No verdict.

Same model, data path, sha256 check, score-before-step order and replay sampler
as H29d. --seed sets SEED. The torch init seed is SEED and the replay sampler
seed is REPLAY_SEED + SEED. --sleep-block B (integer >= 1, default 1) batches
the replay dose: each live chunk is scored and updated, then after every B live
chunks the cortex takes B replay steps back to back. B = 1 is H29d's interleaved
replay. A final sleep covers a remainder shorter than B so the replay-step count
matches the interleaved run. Replay is not scored. The live hidden state is not
touched by replay.

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
E3M = 3_000_000
E3M_SHA256 = '72953fccfeb78e55251024bd6ffbe269d1d8d730ce2e4fb028117a3613c45aac'
FULL = 100_000_000
SLICE = 1_000_000
STRETCH = 128 * 1024
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
    def __init__(self, lr, replay, device, phase, sleep_block=1):
        self.lr = float(lr)
        self.replay = bool(replay)
        self.device = device
        self.phase = phase
        self.sleep_block = int(sleep_block)
        if self.sleep_block < 1:
            raise ValueError('sleep_block must be >= 1')
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
        self._replay_owed = 0
        self.state = None
        self.first_chunk_bits = None
        self.second_chunk_bits = None
        self._chunk_index = 0
        self._log_mark = -1
        self._reported_slices = 0
        self.chunk_bits = []
        self.chunk_bytes = []
        self.sleeps = []
        self.step_kinds = []

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

    def _replay_block(self, raw, n_seen, steps, chunk_hi):
        """B replay steps on bytes seen so far. Each step uses a zero hidden state.

        chunk_hi is the exclusive chunk index of the awake period this sleep follows.
        """
        steps = int(steps)
        if steps < 1:
            raise AssertionError('replay block needs at least one step')
        if n_seen < CHUNK:
            raise AssertionError(f'replay n_seen {n_seen} is below one window')
        self.sleeps.append({
            'chunk_lo': int(chunk_hi - steps),
            'chunk_hi': int(chunk_hi),
            'n_seen': int(n_seen),
            'replay_steps': steps,
        })
        for i in range(steps):
            r_prev, r_tgt = sample_replay(raw, n_seen, self.rng)
            r_prev_t = torch.tensor(r_prev, dtype=torch.long, device=self.device)
            r_tgt_t = torch.tensor(r_tgt, dtype=torch.long, device=self.device)
            r_losses, _ = self.model.nll(r_prev_t, r_tgt_t, None)
            self._step(r_losses)
            del r_losses
            self.replay_steps += 1
            self.step_kinds.append('R')
            if steps >= 1024 and ((i + 1) % 1024 == 0 or i + 1 == steps):
                print(
                    f'H31 phase={self.phase} sleep {i + 1}/{steps} '
                    f'n_seen={n_seen} replay_steps={self.replay_steps} '
                    f'wall_s={time.monotonic() - self.t0:.1f}',
                    flush=True)

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
            f'H31 phase={self.phase} n={end}/{n_target} bpb={running:.6f}'
            f'{slice_bpb} replay_steps={self.replay_steps} '
            f'wall_s={time.monotonic() - self.t0:.1f}',
            flush=True)

    def _report_slices(self, end, n_target):
        while self._reported_slices < end // SLICE:
            i = self._reported_slices
            print(
                f'H31 phase={self.phase} slice={i} '
                f'slice_bpb={self.slice_sums[i] / SLICE:.6f} '
                f'n={end}/{n_target} bpb={self.total_bits / end:.6f}',
                flush=True)
            self._reported_slices += 1

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
            chunk_bits = float(bit_vals.sum())
            self.total_bits += chunk_bits
            self.chunk_bits.append(chunk_bits)
            self.chunk_bytes.append(int(tlen))
            self.step_kinds.append('L')
            self.state = state.detach()
            self._step(losses)
            del losses
            if self.replay and end >= CHUNK:
                # B = 1 is H29d: one replay step immediately after the live step.
                if self.sleep_block == 1:
                    r_prev, r_tgt = sample_replay(raw, end, self.rng)
                    r_prev_t = torch.tensor(r_prev, dtype=torch.long, device=self.device)
                    r_tgt_t = torch.tensor(r_tgt, dtype=torch.long, device=self.device)
                    r_losses, _ = self.model.nll(r_prev_t, r_tgt_t, None)
                    self._step(r_losses)
                    del r_losses
                    self.replay_steps += 1
                    self.step_kinds.append('R')
                    self.sleeps.append({
                        'chunk_lo': int(self._chunk_index),
                        'chunk_hi': int(self._chunk_index + 1),
                        'n_seen': int(end),
                        'replay_steps': 1,
                    })
                else:
                    self._replay_owed += 1
                    if self._replay_owed >= self.sleep_block:
                        self._replay_block(raw, end, self._replay_owed, self._chunk_index + 1)
                        self._replay_owed = 0
            self._chunk_index += 1
            self.n_done = end
            mark = end // 200_000
            # A remainder sleep still owed runs after this loop. Log it there.
            pending_final = (
                end == n_target and self.replay and self.sleep_block != 1
                and self._replay_owed > 0)
            if (mark > self._log_mark or end == n_target) and not pending_final:
                self._log_mark = mark
                self._report_slices(end, n_target)
                self._log(end, n_target)
        if self._replay_owed:
            if not self.replay or self.sleep_block == 1:
                raise AssertionError('replay remainder with no batched sleep')
            self._replay_block(raw, self.n_done, self._replay_owed, self._chunk_index)
            self._replay_owed = 0
            self._report_slices(self.n_done, n_target)
            self._log(self.n_done, n_target)
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
        out = {
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
            'sleep_block': int(self.sleep_block),
            'wall_seconds': time.monotonic() - self.t0,
            'parameters': int(self.n_params),
            'device': str(self.device),
            'chunk_bits': [float(x) for x in self.chunk_bits],
            'chunk_bytes': [int(x) for x in self.chunk_bytes],
            'sleeps': list(self.sleeps),
            'first_chunk_bits_per_byte': (
                None if self.first_chunk_bits is None else float(self.first_chunk_bits.mean())),
        }
        if n <= 2048:
            out['step_kinds'] = ''.join(self.step_kinds)
        return out


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


def run_arm(raw, n_bytes, lr, replay_batch, arm, device, sleep_block=1):
    """One prequential pass. Batch 0 is no replay. Sleep block 1 is the H29d step."""
    global REPLAY_BATCH
    replay_batch = int(replay_batch)
    n_bytes = int(n_bytes)
    sleep_block = int(sleep_block)
    if replay_batch < 0:
        raise SystemExit('replay-batch must be >= 0')
    if sleep_block < 1:
        raise SystemExit('sleep-block must be >= 1')
    if n_bytes <= 0 or n_bytes > len(raw):
        raise SystemExit(f'n-bytes {n_bytes} is outside 1..{len(raw)}')
    REPLAY_BATCH = replay_batch
    run = StreamRun(float(lr), replay_batch > 0, device, arm, sleep_block=sleep_block)
    run.advance(raw, n_bytes)
    out = run.result()
    finite = bool(np.isfinite(out['bits'])) and all(
        np.isfinite(sl['bits']) for sl in out['slices'])
    out['finite'] = finite
    out['replay_batch'] = replay_batch
    if not finite:
        raise FloatingPointError(f'{arm}: nonfinite bits')
    if replay_batch > 0 and n_bytes >= CHUNK:
        expect_steps = (n_bytes + CHUNK - 1) // CHUNK
    else:
        expect_steps = 0
    if out['replay_steps'] != expect_steps:
        raise AssertionError(
            f'{arm}: replay_steps {out["replay_steps"]} != {expect_steps}')
    if n_bytes % CHUNK == 0 and n_bytes >= CHUNK:
        expect_traffic = float(replay_batch) if replay_batch > 0 else 0.0
        if abs(out['traffic'] - expect_traffic) > 1e-9:
            raise AssertionError(
                f'{arm}: traffic {out["traffic"]} != {expect_traffic}')
    return out


def _pack(raw, digest, e3m_digest, arm, lr, replay_batch, sleep_block, n_bytes, row, session, gpu_name, kernel_id):
    return {
        'arm': arm,
        'kernel_id': kernel_id,
        'lr': float(lr),
        'replay_batch': int(replay_batch),
        'sleep_block': int(sleep_block),
        'n_bytes': int(n_bytes),
        'gpu_name': gpu_name,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'sha256_file': sha256_arr(raw),
        'sha256_e10m': digest,
        'sha256_e3m': e3m_digest,
        'file_bytes': int(raw.size),
        'seed': SEED,
        'replay_seed': REPLAY_SEED + SEED,
        'parameters': int(row['parameters']),
        'session_wall_seconds': session,
        'e3m': row,
    }


def kernel_main(out, lr, replay_batch, arm, n_bytes, kernel_id, sleep_block=1):
    configure()
    t0 = time.monotonic()
    global REPLAY_BATCH
    REPLAY_BATCH = 32
    sanity_checks()
    path = find_kaggle_enwik8()
    print(f'H31 data {path} bytes={path.stat().st_size}', flush=True)
    raw = load_enwik8(path)
    digest = sha256_arr(raw[:E10M])
    e3m_digest = sha256_arr(raw[:E3M])
    file_digest = sha256_arr(raw)
    print(
        f'H31 sha256_file={file_digest} sha256_e10m={digest} sha256_e3m={e3m_digest}',
        flush=True)
    if digest != E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest} != {E10M_SHA256}')
    if e3m_digest != E3M_SHA256:
        raise SystemExit(f'E3M sha256 {e3m_digest} != {E3M_SHA256}')
    if int(n_bytes) != E3M:
        raise SystemExit(f'scored n-bytes is {n_bytes}, expected {E3M}')
    if int(sleep_block) < 1:
        raise SystemExit('sleep-block must be >= 1')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the scored run')
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f'H31 arm={arm} device cuda name={gpu_name} torch={torch.__version__} '
        f'lr={float(lr):g} replay_batch={int(replay_batch)} '
        f'sleep_block={int(sleep_block)} n={int(n_bytes)} seed={SEED}',
        flush=True)
    row = run_arm(raw, n_bytes, lr, replay_batch, arm, device, sleep_block=sleep_block)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    session = time.monotonic() - t0
    payload = _pack(
        raw, digest, e3m_digest, arm, lr, replay_batch, sleep_block, n_bytes, row,
        session, gpu_name, kernel_id)
    dump(Path(out) / f'{arm}.json', payload)
    print('KERNEL_DONE ' + json.dumps({
        'kernel_id': kernel_id,
        'arm': arm,
        'lr': float(lr),
        'replay_batch': int(replay_batch),
        'sleep_block': int(sleep_block),
        'n': row['n'],
        'bits': row['bits'],
        'bits_per_byte': row['bits_per_byte'],
        'replay_steps': row['replay_steps'],
        'traffic': row['traffic'],
        'wall_seconds': row['wall_seconds'],
        'session_wall_seconds': session,
        'gpu_name': gpu_name,
        'sha256_e3m': e3m_digest,
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


def describe_schedule(n_chunks, sleep_block):
    """Live (scored) and replay steps for n_chunks full 128-byte chunks.

    Each sleep's replay steps share one n_seen: the bytes read before that sleep.
    A remainder shorter than B is one final sleep, so the replay count equals n_chunks.
    """
    sleep_block = int(sleep_block)
    n_chunks = int(n_chunks)
    if sleep_block < 1:
        raise ValueError('sleep_block must be >= 1')
    lines = [f'B={sleep_block} chunks={n_chunks}']
    owed = 0
    step = 0
    for i in range(n_chunks):
        start = i * CHUNK
        end = start + CHUNK
        lines.append(f'{step} live chunk={i} scored bytes=[{start},{end})')
        step += 1
        owed += 1
        if owed >= sleep_block:
            for j in range(owed):
                lines.append(f'{step} replay sleep_step={j + 1}/{owed} n_seen={end}')
                step += 1
            owed = 0
    if owed:
        end = n_chunks * CHUNK
        for j in range(owed):
            lines.append(
                f'{step} replay sleep_step={j + 1}/{owed} n_seen={end} remainder=1')
            step += 1
    return lines


def schedule_kinds(n_chunks, sleep_block):
    owed = 0
    kinds = []
    for _ in range(int(n_chunks)):
        kinds.append('L')
        owed += 1
        if owed >= int(sleep_block):
            kinds.extend(['R'] * owed)
            owed = 0
    if owed:
        kinds.extend(['R'] * owed)
    return ''.join(kinds)


CHUNKS_PER_STRETCH = STRETCH // CHUNK


def morning_after(chunk_bits, chunk_bytes, sleeps):
    """Mean bits/byte on the 128 KB awake stretch after a sleep versus before it.

    A sleep is included when the awake period that just ended and the next awake
    period each contain a full 1024 chunks of 128 bytes. Before is the last 1024
    chunks before the sleep. After is the first 1024 chunks of the next period.
    """
    if len(chunk_bits) != len(chunk_bytes):
        raise SystemExit('chunk_bits and chunk_bytes differ in length')
    rows = []
    for i, sleep in enumerate(sleeps[:-1]):
        nxt = sleeps[i + 1]
        before_lo = int(sleep['chunk_hi']) - CHUNKS_PER_STRETCH
        before_hi = int(sleep['chunk_hi'])
        after_lo = int(nxt['chunk_lo'])
        after_hi = int(nxt['chunk_lo']) + CHUNKS_PER_STRETCH
        if before_lo < int(sleep['chunk_lo']):
            continue
        if after_hi > int(nxt['chunk_hi']):
            continue
        if before_lo < 0 or after_hi > len(chunk_bits):
            continue
        if after_lo != int(sleep['chunk_hi']):
            continue
        if any(int(b) != CHUNK for b in chunk_bytes[before_lo:before_hi]):
            continue
        if any(int(b) != CHUNK for b in chunk_bytes[after_lo:after_hi]):
            continue
        before_bits = float(sum(chunk_bits[before_lo:before_hi]))
        after_bits = float(sum(chunk_bits[after_lo:after_hi]))
        before_bpb = before_bits / STRETCH
        after_bpb = after_bits / STRETCH
        rows.append({
            'sleep_index': i,
            'n_seen': int(sleep['n_seen']),
            'replay_steps': int(sleep['replay_steps']),
            'before_chunk_lo': before_lo,
            'before_chunk_hi': before_hi,
            'after_chunk_lo': after_lo,
            'after_chunk_hi': after_hi,
            'before_bits': before_bits,
            'after_bits': after_bits,
            'before_bits_per_byte': before_bpb,
            'after_bits_per_byte': after_bpb,
            'after_minus_before': after_bpb - before_bpb,
        })
    n = len(rows)
    if n == 0:
        return {
            'stretch_bytes': STRETCH,
            'n_sleeps': 0,
            'mean_bits_per_byte_before': None,
            'mean_bits_per_byte_after': None,
            'mean_after_minus_before': None,
            'sleeps': [],
        }
    return {
        'stretch_bytes': STRETCH,
        'n_sleeps': n,
        'mean_bits_per_byte_before': sum(r['before_bits_per_byte'] for r in rows) / n,
        'mean_bits_per_byte_after': sum(r['after_bits_per_byte'] for r in rows) / n,
        'mean_after_minus_before': sum(r['after_minus_before'] for r in rows) / n,
        'sleeps': rows,
    }


def log_elapsed_seconds(path):
    path = Path(path)
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())
    times = [float(row['time']) for row in rows if isinstance(row, dict) and 'time' in row]
    if not times:
        return None
    return max(times)


def _load_h31_arm(out_dir, arm):
    path = Path(out_dir) / 'raw' / arm / f'{arm}.json'
    if not path.is_file():
        raise SystemExit(f'missing {path}')
    payload = json.loads(path.read_text())
    if payload.get('arm') != arm:
        raise SystemExit(f'{path} arm is {payload.get("arm")}')
    if payload.get('sha256_e3m') != E3M_SHA256:
        raise SystemExit(f'{arm} E3M sha256 {payload.get("sha256_e3m")}')
    if payload.get('sha256_e10m') != E10M_SHA256:
        raise SystemExit(f'{arm} E10M sha256 {payload.get("sha256_e10m")}')
    row = payload['e3m']
    if row['n'] != E3M or not row['finite']:
        raise SystemExit(f'{arm} is not a finite E3M run')
    if len(row['slices']) != 3:
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
    order = ('inter', 'sleep_64', 'sleep_1k', 'sleep_16k', 'noreplay')
    expected_batch = {'inter': 8, 'sleep_64': 8, 'sleep_1k': 8, 'sleep_16k': 8, 'noreplay': 0}
    expected_block = {'inter': 1, 'sleep_64': 64, 'sleep_1k': 1024, 'sleep_16k': 16384, 'noreplay': 1}
    arms = []
    for name in order:
        payload = _load_h31_arm(out_dir, name)
        if int(payload['replay_batch']) != expected_batch[name]:
            raise SystemExit(f'{name} replay_batch {payload["replay_batch"]}')
        if int(payload['sleep_block']) != expected_block[name]:
            raise SystemExit(f'{name} sleep_block {payload["sleep_block"]}')
        if abs(float(payload['lr']) - 3e-4) > 1e-15:
            raise SystemExit(f'{name} lr {payload["lr"]}')
        if int(payload['seed']) != 0:
            raise SystemExit(f'{name} seed {payload["seed"]}')
        row = payload['e3m']
        meta = kernels[name]
        elapsed = meta.get('log_elapsed_seconds')
        if elapsed is None:
            elapsed = payload['session_wall_seconds']
            elapsed_source = 'script_wall'
        else:
            elapsed_source = meta.get('elapsed_source') or 'log_clock'
        morning = None
        if name in ('sleep_1k', 'sleep_16k'):
            morning = morning_after(row['chunk_bits'], row['chunk_bytes'], row['sleeps'])
        arms.append({
            'arm': name,
            'lr': payload['lr'],
            'replay_batch': int(payload['replay_batch']),
            'sleep_block': None if name == 'noreplay' else int(payload['sleep_block']),
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
            'torch_version': payload['torch_version'],
            'parameters': payload['parameters'],
            'seed': payload['seed'],
            'replay_seed': payload['replay_seed'],
            'morning_after': morning,
        })
    by_name = {row['arm']: row for row in arms}
    replay_counts = {row['arm']: row['replay_steps'] for row in arms if row['arm'] != 'noreplay'}
    if len(set(replay_counts.values())) != 1:
        raise SystemExit(f'replay steps differ: {replay_counts}')
    if arms[-1]['replay_steps'] != 0:
        raise SystemExit('noreplay replay_steps is not 0')
    base = by_name['inter']['bits_per_byte']
    gaps = {
        'sleep_64_minus_inter': by_name['sleep_64']['bits_per_byte'] - base,
        'sleep_1k_minus_inter': by_name['sleep_1k']['bits_per_byte'] - base,
        'sleep_16k_minus_inter': by_name['sleep_16k']['bits_per_byte'] - base,
        'noreplay_minus_inter': by_name['noreplay']['bits_per_byte'] - base,
    }
    gpu_hours = [row['elapsed_seconds'] / 3600.0 for row in arms]
    lines = [
        '# H31 sleep gate on enwik8 E3M',
        '',
        f'E3M bytes: {E3M}. sha256: `{E3M_SHA256}`.',
        'E10M sha256, checked and not scored: '
        f'`{E10M_SHA256}`.',
        'GRU bits/byte = sum of live −log2 p / 3000000. Replay is not part of the code length.',
        'No bytes are scored during sleep.',
        '',
        'Fixed settings: lr 0.0003, seed 0, replay seed 30000, embedding 64, hidden 1024,',
        'chunk 128, clip 1.0, Adam. Replay batch 8, except noreplay batch 0.',
        '',
        '## E3M',
        '',
        '| Arm | Sleep block | Replay batch | Bits/byte | Replay steps | Wall seconds | GPU |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in arms:
        block = 'none' if row['sleep_block'] is None else str(row['sleep_block'])
        lines.append(
            f"| {row['arm']} | {block} | {row['replay_batch']} | "
            f"{_fmt(row['bits_per_byte'])} | {row['replay_steps']} | "
            f"{row['wall_seconds']:.3f} | {row['gpu_name']} |")
    lines += ['', '## 1 MB slices', '']
    for row in arms:
        lines += [
            f"### {row['arm']}",
            '',
            '| Slice | Start | End | Bits | Bits/byte |',
            '|---:|---:|---:|---:|---:|',
        ]
        for sl in row['slices']:
            lines.append(
                f"| {sl['index']} | {sl['start']} | {sl['end']} | {_fmt(sl['bits'], 3)} | "
                f"{_fmt(sl['bits_per_byte'])} |")
        lines.append('')
    lines += [
        '## Gaps',
        '',
        'Pre-registered differences. Not scored in this file.',
        '',
        f"sleep_64 − inter = {_fmt(gaps['sleep_64_minus_inter'])} "
        f"({_fmt(by_name['sleep_64']['bits_per_byte'])} − {_fmt(base)}).",
        f"sleep_1k − inter = {_fmt(gaps['sleep_1k_minus_inter'])} "
        f"({_fmt(by_name['sleep_1k']['bits_per_byte'])} − {_fmt(base)}).",
        f"sleep_16k − inter = {_fmt(gaps['sleep_16k_minus_inter'])} "
        f"({_fmt(by_name['sleep_16k']['bits_per_byte'])} − {_fmt(base)}).",
        f"noreplay − inter = {_fmt(gaps['noreplay_minus_inter'])} "
        f"({_fmt(by_name['noreplay']['bits_per_byte'])} − {_fmt(base)}).",
        '',
        '## Morning after',
        '',
        '128 KB = 131072 bytes = 1024 chunks. For each sleep that has a full 128 KB awake',
        'stretch immediately before it and a full 128 KB awake stretch immediately after it,',
        'bits/byte of those stretches. The mean is over those sleeps. Awake bits only.',
        '',
    ]
    for name in ('sleep_1k', 'sleep_16k'):
        item = by_name[name]['morning_after']
        lines += [
            f'### {name}',
            '',
            f"Sleeps in the mean: {item['n_sleeps']}.",
            f"Mean bits/byte before: {_fmt(item['mean_bits_per_byte_before'])}.",
            f"Mean bits/byte after: {_fmt(item['mean_bits_per_byte_after'])}.",
            f"Mean after − before: {_fmt(item['mean_after_minus_before'])}.",
            '',
            '| Sleep | Bytes seen | Bits/byte before | Bits/byte after | After − before |',
            '|---:|---:|---:|---:|---:|',
        ]
        for sl in item['sleeps']:
            lines.append(
                f"| {sl['sleep_index']} | {sl['n_seen']} | "
                f"{_fmt(sl['before_bits_per_byte'])} | {_fmt(sl['after_bits_per_byte'])} | "
                f"{_fmt(sl['after_minus_before'])} |")
        lines.append('')
    lines += [
        '## Kernels',
        '',
        'Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that',
        'clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.',
        'Script wall is time.monotonic from kernel_main entry through the result write. Queue time',
        'is not included. Total GPU hours is the sum across the five kernels.',
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
    ]
    note = kernels.get('note')
    if note:
        lines += [note, '']
    lines += [
        '## Protocol',
        '',
        '- One online pass per arm. Prequential: −log2 p of each byte is recorded before that chunk is trained.',
        '- Awake: score the 128-byte chunk, then one Adam step on it. No replay step while awake, except sleep block 1.',
        '- Sleep: after every B live chunks, B replay steps back to back. Each step is batch 8, window 128, hidden state 0.',
        '- Replay starts are uniform on bytes seen so far, the same sampler as H29d. One rng draw per replay step.',
        '- Sleep block 1 replays once after every live chunk. That is the H29d path.',
        '- A final sleep covers a remainder shorter than B. Replay steps match across the replay arms.',
        '- The live hidden state is detached after the live score and is not replaced by replay.',
        '- E3M is bytes [0, 3000000). 1 MB slices are 1000000 bytes.',
        '- Traffic = replay steps × replay batch × 128 / bytes.',
        '- Code is h31_sleep.py. Private dataset ezzzzzx/pdp-code-h31. Data dataset ezzzzzx/enwik8.',
        '',
    ]
    if smoke is not None:
        lines += [
            '## Smoke',
            '',
            'CPU. Not a score.',
            '',
            f"Bytes: {smoke['n_bytes']}.",
            f"h29d bits: {smoke['h29d_bits']:.12f}.",
            f"h31 --sleep-block 1 bits: {smoke['h31_inter_bits']:.12f}.",
            f"Absolute difference: {smoke['abs_diff']:.3e}.",
            f"Within 1e-6: {smoke['within_1e-6']}.",
            '',
            f"sleep block 16 replay steps: {smoke['sleep_16']['replay_steps']}.",
            f"inter replay steps: {smoke['sleep_16']['inter_replay_steps']}.",
            f"Counts equal: {smoke['sleep_16']['steps_match']}.",
            f"sleep block 16 bits finite: {smoke['sleep_16']['finite']}.",
            f"sleep block 16 bits: {smoke['sleep_16']['bits']:.12f}.",
            '',
            'Schedule, B = 4, first 12 chunks. live = scored. replay = not scored.',
            '',
        ]
        for line in smoke['schedule_lines']:
            lines.append(line)
        lines += [
            '',
            f"Schedule kinds: {smoke['schedule_kinds']}.",
            f"Run step kinds: {smoke['run_step_kinds']}.",
            f"Schedule matches the run: {smoke['schedule_match']}.",
            '',
        ]
        if 'remainder_200' in smoke:
            rem = smoke['remainder_200']
            lines += [
                f"Remainder 200 bytes, sleep block 4 replay steps: {rem['replay_steps']}.",
                f"Sleep block 1 replay steps: {rem['inter_replay_steps']}.",
                f"Counts equal: {rem['steps_match']}. Both finite: {rem['finite']}.",
                '',
            ]
    text = '\n'.join(lines)
    if not text.endswith('\n'):
        text += '\n'
    (out_dir / 'RESULTS.md').write_text(text)
    merged = {
        'e3m_bytes': E3M,
        'sha256_e3m': E3M_SHA256,
        'sha256_e10m': E10M_SHA256,
        'arms': {
            row['arm']: {
                'lr': row['lr'],
                'replay_batch': row['replay_batch'],
                'sleep_block': row['sleep_block'],
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
        'gaps': gaps,
        'morning_after': {
            name: by_name[name]['morning_after'] for name in ('sleep_1k', 'sleep_16k')
        },
        'gpu_hours_total': sum(gpu_hours),
        'replay_steps_matched': replay_counts['inter'],
        'smoke': smoke,
    }
    dump(out_dir / 'results.json', merged)
    print(f'H31 wrote {out_dir / "RESULTS.md"} and results.json', flush=True)


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
    ap.add_argument('--sleep-block', type=int, default=1)
    args = ap.parse_args()
    SEED = int(args.seed)
    if int(args.sleep_block) < 1:
        raise SystemExit('--sleep-block must be an integer >= 1')
    if args.assemble:
        write_report(args.out or 'results/h31')
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
    e3m_digest = sha256_arr(raw[:E3M])
    if digest != E10M_SHA256:
        raise SystemExit(f'E10M sha256 {digest} != {E10M_SHA256}')
    if e3m_digest != E3M_SHA256:
        raise SystemExit(f'E3M sha256 {e3m_digest} != {E3M_SHA256}')
    print(
        f'H31 file bytes={raw.size} sha256_e3m={e3m_digest} sha256_e10m={digest} '
        f'arm={args.arm} lr={float(args.lr):g} replay_batch={int(args.replay_batch)} '
        f'sleep_block={int(args.sleep_block)} n={int(args.n_bytes)} '
        f'seed={SEED} replay_seed={REPLAY_SEED + SEED}',
        flush=True)
    t0 = time.monotonic()
    row = run_arm(
        raw, args.n_bytes, args.lr, args.replay_batch, args.arm, torch.device('cpu'),
        sleep_block=args.sleep_block)
    session = time.monotonic() - t0
    payload = _pack(
        raw, digest, e3m_digest, args.arm, args.lr, args.replay_batch, args.sleep_block,
        args.n_bytes, row, session, 'cpu', None)
    dump(Path(args.out) / f'{args.arm}.json', payload)
    kinds = row.get('step_kinds', '')
    print(
        f'H31 arm={args.arm} bits={row["bits"]:.12f} bpb={row["bits_per_byte"]:.12f} '
        f'finite={row["finite"]} replay_steps={row["replay_steps"]} '
        f'sleep_block={row["sleep_block"]} traffic={row["traffic"]:.6f} '
        f'wall_s={row["wall_seconds"]:.3f} step_kinds={kinds}',
        flush=True)


if __name__ == '__main__':
    main()
