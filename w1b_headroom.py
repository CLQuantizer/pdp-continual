"""W1b: dna, noise and dark added to the W1-0 channels. Numbers only. No verdict.

The cortex, the chunking, the score-before-step order, the clip, the replay
window, and the seeds are the W1-0 copy of the H29c setting: lr 3e-4, replay
batch 8, seed 0. Bible, chess, squad and btc are not rerun. Does not commit.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import subprocess
import sys
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
REPLAY_BATCH = 8
CLIP = 1.0
SEED = 0
REPLAY_SEED = 30_000
SLICE = 1_000_000
PARAMS = 3_627_264
LOG2 = math.log(2)
OFFSETS = np.arange(CHUNK, dtype=np.int64)[:, None]
LR = 3e-4
TRAIN_N = 2_000_000
HELD_N = 100_000
BUDGET = 2_000_000
CHECKPOINTS = (0, 62_500, 125_000, 250_000, 500_000, 1_000_000, 2_000_000)
W1_CHANNELS = ('bible', 'chess', 'squad', 'btc')
NEW_CHANNELS = ('dna', 'noise', 'dark')
FULL_CHANNELS = W1_CHANNELS + NEW_CHANNELS
LEARNABLE_CHANNELS = W1_CHANNELS + ('dna',)
SYNTHETIC_CHANNELS = ('noise', 'dark')
# Kept so a copied call site still names the W1-0 set. W1b does not rerun it.
CHANNELS = W1_CHANNELS
EXCERPT_N = 300
STREAM_N = 2_100_000
NOISE_SEED = 777
W1_SMOKE_BITS = 19362.707616529544
DNA_BASES = frozenset('ACGTN')
UNIT = 62_500
PPMD_ORDER = 16
PPMD_MEM = 1 << 30
# A single Ppmd7 encode of the chess stream does not roundtrip (pyppmd 1.3.1).
# Pieces of this size do, and they match one-shot bytes when one-shot roundtrips.
PPMD_PIECE = 65_536

SOURCE_NAMES = {
    'bible': 'pg10.txt',
    'chess': 'games.csv',
    'squad': 'dev-v1.1.json',
    'btc': 'BTC_5min.csv',
    'dna': 'human.txt',
}
LOCAL_SOURCES = {
    'bible': Path('data/pg10.txt'),
    'chess': Path('data/kaggle_samples/lichess_games/games.csv'),
    'squad': Path('data/kaggle_samples/squad/dev-v1.1.json'),
    'btc': Path('data/kaggle_samples/btc_lob5/BTC_5min.csv'),
    'dna': Path('data/w1b/human.txt'),
}


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


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def as_u8(blob):
    return np.array(np.frombuffer(blob, dtype=np.uint8), dtype=np.uint8, copy=True)


def excerpt_text(blob):
    raw = blob[:EXCERPT_N]
    end = len(raw)
    while end > 0:
        try:
            text = raw[:end].decode('utf-8')
        except UnicodeDecodeError:
            end -= 1
            continue
        return text, end
    return '', 0


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
            f'W1 phase={self.phase} n={end}/{n_target} bpb={running:.6f}'
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
                        f'W1 phase={self.phase} slice={i} '
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
        }


def score_block(model, raw, device):
    """Held-out live -log2 p. Zero state, chunks of 128, state carried. No gradient."""
    n = len(raw)
    if n <= 0:
        raise AssertionError('empty held-out block')
    total = 0.0
    state = None
    pos = 0
    with torch.no_grad():
        while pos < n:
            start = pos
            end = min(start + CHUNK, n)
            tlen = end - start
            if start == 0:
                prev_np = np.empty(tlen, dtype=np.int64)
                prev_np[0] = -1
                if tlen > 1:
                    prev_np[1:] = raw[:tlen - 1]
            else:
                prev_np = raw[start - 1:end - 1].astype(np.int64, copy=False)
            tgt_np = raw[start:end].astype(np.int64, copy=False)
            prev_t = torch.tensor(prev_np, dtype=torch.long, device=device).unsqueeze(1)
            tgt_t = torch.tensor(tgt_np, dtype=torch.long, device=device).unsqueeze(1)
            losses, state = model.nll(prev_t, tgt_t, state)
            bit_vals = (losses.double() / LOG2).cpu().numpy()
            if not np.isfinite(bit_vals).all():
                raise FloatingPointError('nonfinite held-out bits')
            total += float(bit_vals.sum())
            state = state.detach()
            del losses
            pos = end
    return total


def run_learner(train, held, device, phase, checkpoints):
    global REPLAY_BATCH
    REPLAY_BATCH = 8
    run = StreamRun(LR, True, device, phase)
    held_rows = []
    for n in checkpoints:
        if n > run.n_done:
            run.advance(train, n)
        if run.n_done != n:
            raise AssertionError(f'{phase}: n_done {run.n_done} != checkpoint {n}')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.monotonic()
        bits = score_block(run.model, held, device)
        row = {
            'n': int(n),
            'bits': bits,
            'bits_per_byte': bits / len(held),
            'held_bytes': int(len(held)),
            'finite': bool(np.isfinite(bits)),
            'eval_seconds': time.monotonic() - t0,
        }
        if not row['finite']:
            raise FloatingPointError(f'{phase}: nonfinite L at n={n}')
        print(
            f'W1 phase={phase} heldout n={n} L={row["bits_per_byte"]:.6f} '
            f'bits={bits:.6f} eval_s={row["eval_seconds"]:.1f}',
            flush=True)
        held_rows.append(row)
    trained = run.result()
    return trained, held_rows


def build_bible(raw):
    start_at = raw.find(b'*** START')
    if start_at < 0:
        raise SystemExit('bible: no *** START line')
    line_end = raw.find(b'\n', start_at)
    if line_end < 0:
        raise SystemExit('bible: START line has no newline')
    body_start = line_end + 1
    end_at = raw.find(b'*** END', body_start)
    if end_at < 0:
        raise SystemExit('bible: no *** END line')
    end_line = raw.rfind(b'\n', body_start, end_at) + 1
    return raw[body_start:end_line]


def build_chess(raw):
    text = raw.decode('utf-8')
    reader = csv.DictReader(io.StringIO(text, newline=''))
    if reader.fieldnames is None or 'moves' not in reader.fieldnames:
        raise SystemExit('chess: moves column missing')
    moves = []
    for row in reader:
        value = row.get('moves')
        if value is None:
            raise SystemExit('chess: null moves field')
        moves.append(value)
    if not moves:
        raise SystemExit('chess: no games')
    return ('\n'.join(moves) + '\n').encode('utf-8')


def build_squad(raw):
    data = json.loads(raw.decode('utf-8'))
    parts = []
    for article in data['data']:
        for para in article['paragraphs']:
            parts.append(para['context'])
            for qa in para['qas']:
                parts.append(qa['question'])
                answers = qa.get('answers') or []
                if not answers or 'text' not in answers[0]:
                    raise SystemExit('squad: question has no first answer')
                parts.append(answers[0]['text'])
    if not parts:
        raise SystemExit('squad: empty plain text')
    return ('\n'.join(parts) + '\n').encode('utf-8')


def build_btc(raw):
    if not raw:
        raise SystemExit('btc: empty file')
    return raw


def build_dna(raw):
    """Sequence column only. Header and class labels are dropped."""
    text = raw.decode('utf-8')
    lines = text.splitlines()
    if len(lines) < 2:
        raise SystemExit('dna: no sequence rows')
    header_field = lines[0].split('\t', 1)[0].strip().lower()
    if header_field != 'sequence':
        raise SystemExit(f'dna: header field is {header_field!r}')
    seqs = []
    for lineno, line in enumerate(lines[1:], start=2):
        if line.strip() == '':
            continue
        field = line.split('\t', 1)[0].strip()
        if field == '':
            raise SystemExit(f'dna: empty sequence on line {lineno}')
        seq = field.upper()
        bad = [ch for ch in sorted(set(seq)) if ch not in DNA_BASES]
        if bad:
            raise SystemExit(f'dna: bases outside ACGTN on line {lineno}: {bad}')
        seqs.append(seq)
    if not seqs:
        raise SystemExit('dna: no sequences')
    return ('\n'.join(seqs) + '\n').encode('ascii')


def build_noise(_raw):
    rng = np.random.default_rng(NOISE_SEED)
    values = rng.integers(0, 256, size=STREAM_N, dtype=np.uint8)
    if values.shape != (STREAM_N,) or values.dtype != np.uint8:
        raise SystemExit(f'noise: unexpected array {values.shape} {values.dtype}')
    return values.tobytes()


def build_dark(_raw):
    return bytes(STREAM_N)


BUILDERS = {
    'bible': build_bible,
    'chess': build_chess,
    'squad': build_squad,
    'btc': build_btc,
    'dna': build_dna,
    'noise': build_noise,
    'dark': build_dark,
}


def split_stream(stream):
    n = len(stream)
    if n >= TRAIN_N + HELD_N:
        train = stream[:TRAIN_N]
        held = stream[TRAIN_N:TRAIN_N + HELD_N]
        mode = 'prefix_2000000_100000'
    else:
        train_n = (n * 95) // 100
        if train_n <= 0 or train_n >= n:
            raise SystemExit(f'stream length {n} cannot be split 95/5')
        train = stream[:train_n]
        held = stream[train_n:]
        mode = 'ratio_95_5'
        print(
            f'W1 short stream bytes={n} train={train_n} held={n - train_n} mode={mode}',
            flush=True)
    return train, held, mode


def checkpoints_for(train_n):
    pts = tuple(n for n in CHECKPOINTS if n <= train_n)
    if not pts or pts[0] != 0:
        raise SystemExit(f'no checkpoint at 0 for train_n={train_n}')
    return pts


def load_local_channel(channel):
    path = LOCAL_SOURCES[channel]
    if not path.is_file():
        raise SystemExit(f'missing {path}')
    raw = path.read_bytes()
    stream = BUILDERS[channel](raw)
    train, held, mode = split_stream(stream)
    text, nbytes = excerpt_text(stream)
    info = {
        'channel': channel,
        'source_path': str(path),
        'source_bytes': len(raw),
        'source_sha256': sha256_bytes(raw),
        'stream_bytes': len(stream),
        'stream_sha256': sha256_bytes(stream),
        'split': mode,
        'train_bytes': len(train),
        'train_sha256': sha256_bytes(train),
        'held_bytes': len(held),
        'held_sha256': sha256_bytes(held),
        'excerpt': text,
        'excerpt_bytes': nbytes,
        'checkpoints': list(checkpoints_for(len(train))),
    }
    return info, train, held


def find_kaggle_source(channel):
    name = SOURCE_NAMES[channel]
    root = Path('/kaggle/input')
    hits = []
    for p in root.rglob(name):
        if not p.is_file():
            continue
        if any(part.startswith('pdp-code-') for part in p.parts):
            continue
        hits.append(p)
    if len(hits) != 1:
        raise SystemExit(f'{channel}: expected one {name} under {root}, found {hits}')
    return hits[0]


def _cmd(args, data):
    proc = subprocess.run(args, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode('utf-8', 'replace')[:400]
        raise RuntimeError(f'{args} failed ({proc.returncode}): {err}')
    return proc.stdout


def compress_baselines(train):
    rows = {}

    def xz():
        blob = _cmd(['xz', '-9e', '-T1', '-c'], train)
        if _cmd(['xz', '-dc'], blob) != train:
            raise RuntimeError('xz roundtrip failed')
        return len(blob)

    def ppmd():
        import pyppmd
        comp = pyppmd.Ppmd7Encoder(PPMD_ORDER, PPMD_MEM)
        parts = []
        for start in range(0, len(train), PPMD_PIECE):
            parts.append(comp.encode(train[start:start + PPMD_PIECE]))
        parts.append(comp.flush())
        blob = b''.join(parts)
        back = pyppmd.Ppmd7Decoder(PPMD_ORDER, PPMD_MEM).decode(blob, len(train))
        if back != train:
            raise RuntimeError('ppmd H order 16 roundtrip failed')
        return len(blob)

    for name, fn in (('xz_9e', xz), ('ppmd_h_order_16', ppmd)):
        print(f'W1 baseline start {name} bytes={len(train)}', flush=True)
        t0 = time.monotonic()
        size = fn()
        rows[name] = {
            'compressed_bytes': int(size),
            'bits_per_byte': 8.0 * int(size) / len(train),
            'train_bytes': len(train),
            'wall_seconds': time.monotonic() - t0,
            'roundtrip': True,
            'encode_piece_bytes': PPMD_PIECE if name.startswith('ppmd') else None,
        }
        print(
            f'W1 baseline done {name} bytes={size} bpb={rows[name]["bits_per_byte"]:.6f} '
            f'wall_s={rows[name]["wall_seconds"]:.1f}',
            flush=True)
    return rows


def run_channel_bytes(channel, raw, device, kernel_id, gpu_name):
    stream = BUILDERS[channel](raw)
    train_b, held_b, mode = split_stream(stream)
    text, nbytes = excerpt_text(stream)
    train = as_u8(train_b)
    held = as_u8(held_b)
    points = checkpoints_for(len(train))
    if TRAIN_N in points and 500_000 not in points:
        raise SystemExit(f'{channel}: uniform checkpoint 500000 missing')
    print(
        f'W1 channel={channel} stream_bytes={len(stream)} sha256={sha256_bytes(stream)} '
        f'split={mode} train={len(train)} held={len(held)} device={device}',
        flush=True)
    t0 = time.monotonic()
    trained, held_rows = run_learner(train, held, device, channel, points)
    wall = time.monotonic() - t0
    finite = bool(trained['bits'] == trained['bits'] and all(row['finite'] for row in held_rows))
    if not finite or not np.isfinite(trained['bits']):
        raise FloatingPointError(f'{channel}: nonfinite training bits')
    return {
        'channel': channel,
        'kernel_id': kernel_id,
        'gpu_name': gpu_name,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'lr': LR,
        'replay_batch': 8,
        'seed': SEED,
        'replay_seed': REPLAY_SEED + SEED,
        'parameters': PARAMS,
        'source_bytes': len(raw),
        'source_sha256': sha256_bytes(raw),
        'stream_bytes': len(stream),
        'stream_sha256': sha256_bytes(stream),
        'split': mode,
        'train_bytes': len(train_b),
        'train_sha256': sha256_bytes(train_b),
        'held_bytes': len(held_b),
        'held_sha256': sha256_bytes(held_b),
        'excerpt': text,
        'excerpt_bytes': nbytes,
        'checkpoints': list(points),
        'prequential': trained,
        'heldout': held_rows,
        'channel_wall_seconds': wall,
        'finite': finite,
    }


def kernel_main(out, channels, kernel_id):
    configure()
    t0 = time.monotonic()
    out = Path(out)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the scored run')
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f'W1b kernel={kernel_id} gpu={gpu_name} torch={torch.__version__} '
        f'numpy={np.__version__} channels={list(channels)}',
        flush=True)
    summaries = []
    for channel in channels:
        if channel in SYNTHETIC_CHANNELS:
            raw = BUILDERS[channel](b'')
            print(f'W1b source {channel} synthetic bytes={len(raw)}', flush=True)
        else:
            path = find_kaggle_source(channel)
            print(f'W1b source {channel} {path} bytes={path.stat().st_size}', flush=True)
            raw = path.read_bytes()
        payload = run_channel_bytes(channel, raw, device, kernel_id, gpu_name)
        if payload['stream_bytes'] < TRAIN_N + HELD_N:
            raise SystemExit(
                f'{channel}: stream {payload["stream_bytes"]} is shorter than {TRAIN_N + HELD_N}')
        if payload['train_bytes'] != TRAIN_N or payload['held_bytes'] != HELD_N:
            raise SystemExit(
                f'{channel}: split train={payload["train_bytes"]} held={payload["held_bytes"]}')
        stream = BUILDERS[channel](raw)
        train_b = stream[:TRAIN_N]
        if sha256_bytes(train_b) != payload['train_sha256']:
            raise SystemExit(f'{channel}: train bytes diverged from the scored stream')
        if channel in SYNTHETIC_CHANNELS and sha256_bytes(stream) != payload['stream_sha256']:
            raise SystemExit(f'{channel}: synthetic stream diverged from the source bytes')
        (out / f'{channel}.train.bin').write_bytes(train_b)
        dump(out / f'{channel}.json', payload)
        summaries.append({
            'channel': channel,
            'stream_sha256': payload['stream_sha256'],
            'train_sha256': payload['train_sha256'],
            'bits': payload['prequential']['bits'],
            'bits_per_byte': payload['prequential']['bits_per_byte'],
            'replay_steps': payload['prequential']['replay_steps'],
            'finite': payload['finite'],
            'channel_wall_seconds': payload['channel_wall_seconds'],
        })
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    session = time.monotonic() - t0
    meta = {
        'kernel_id': kernel_id,
        'gpu_name': gpu_name,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'session_wall_seconds': session,
        'channels': summaries,
    }
    dump(out / 'kernel.json', meta)
    print('KERNEL_DONE ' + json.dumps(meta), flush=True)
    return meta


def _fmt(value, digits=6):
    return f'{value:.{digits}f}'


def _log_elapsed(log_path):
    text = Path(log_path).read_text(errors='replace')
    times = []
    key = '"time":'
    start = 0
    while True:
        at = text.find(key, start)
        if at < 0:
            break
        at += len(key)
        end = at
        while end < len(text) and text[end] in ' 0123456789.':
            end += 1
        try:
            times.append(float(text[at:end]))
        except ValueError:
            pass
        start = end
    if not times:
        return None
    return max(times)


def _status_name(status):
    text = str(status).strip().strip('"')
    if '.' in text:
        text = text.rsplit('.', 1)[-1]
    return text


def _bpb_table(held_rows):
    table = {}
    for row in held_rows:
        n = int(row['n'])
        value = float(row['bits_per_byte'])
        if not math.isfinite(value):
            raise SystemExit(f'nonfinite bits/byte at n={n}')
        table[n] = value
    return table


def _dp_oracle(names, bpb, budget):
    """Independent check of the exhaustive minimum. Units of 62500 bytes."""
    if budget % UNIT != 0:
        raise SystemExit(f'budget {budget} is not a multiple of {UNIT}')
    for n in CHECKPOINTS:
        if n % UNIT != 0:
            raise SystemExit(f'checkpoint {n} is not a multiple of {UNIT}')
    limit = budget // UNIT
    dp = {0: 0.0}
    choice = {0: ()}
    for name in names:
        nxt = {}
        nxt_choice = {}
        for used, cost in dp.items():
            alloc = choice[used]
            for n in CHECKPOINTS:
                nu = used + (n // UNIT)
                if nu > limit:
                    continue
                ncost = cost + bpb[name][n]
                prev = nxt.get(nu)
                if prev is None or ncost < prev:
                    nxt[nu] = ncost
                    nxt_choice[nu] = alloc + (n,)
        if not nxt:
            raise SystemExit('dp has no feasible allocation')
        dp = nxt
        choice = nxt_choice
    used = min(dp, key=dp.get)
    return dp[used] / len(names), choice[used]


def compute_set_headroom(names, bpb):
    """Uniform minus the exhaustive oracle. Budget is 500000 bytes times k."""
    names = tuple(names)
    k = len(names)
    if k < 1:
        raise SystemExit('empty channel set')
    budget = 500_000 * k
    for name in names:
        missing = [n for n in CHECKPOINTS if n not in bpb[name]]
        if missing:
            raise SystemExit(f'{name} missing checkpoints {missing}')
    uniform = sum(bpb[name][500_000] for name in names) / k
    best = None
    ties = []
    n_feasible = 0
    for alloc in itertools.product(CHECKPOINTS, repeat=k):
        if sum(alloc) > budget:
            continue
        n_feasible += 1
        mean = sum(bpb[names[i]][alloc[i]] for i in range(k)) / k
        if best is None or mean < best:
            best = mean
            ties = [alloc]
        elif abs(mean - best) <= 1e-12:
            ties.append(alloc)
    if best is None:
        raise SystemExit('no feasible allocation')
    dp_mean, dp_alloc = _dp_oracle(names, bpb, budget)
    if abs(dp_mean - best) > 1e-9:
        raise SystemExit(f'exhaustive oracle {best:.12f} != dp {dp_mean:.12f}')
    allocations = []
    for alloc in ties:
        row = {name: int(alloc[i]) for i, name in enumerate(names)}
        row['sum_bytes'] = int(sum(alloc))
        row['mean_bits_per_byte'] = sum(bpb[names[i]][alloc[i]] for i in range(k)) / k
        allocations.append(row)
    return {
        'channels': list(names),
        'k': k,
        'budget_bytes': budget,
        'grid': list(CHECKPOINTS),
        'n_grid': len(CHECKPOINTS),
        'n_tuples': len(CHECKPOINTS) ** k,
        'n_feasible': n_feasible,
        'uniform_allocation': {name: 500_000 for name in names},
        'uniform_bits_per_byte': uniform,
        'oracle_bits_per_byte': best,
        'headroom_bits_per_byte': uniform - best,
        'oracle_allocations': allocations,
        'dp_check_bits_per_byte': dp_mean,
        'dp_check_allocation': {name: int(n) for name, n in zip(names, dp_alloc)},
    }


def check_w1_search(path):
    data = json.loads(Path(path).read_text())
    names = W1_CHANNELS
    bpb = {name: _bpb_table(data['channels'][name]['heldout']) for name in names}
    head = compute_set_headroom(names, bpb)
    ref = data['headroom']
    ok_feasible = head['n_feasible'] == ref['n_feasible']
    ok_uniform = abs(head['uniform_bits_per_byte'] - ref['uniform_bits_per_byte']) <= 1e-12
    ok_oracle = abs(head['oracle_bits_per_byte'] - ref['oracle_bits_per_byte']) <= 1e-12
    ok_head = abs(head['headroom_bits_per_byte'] - ref['headroom_bits_per_byte']) <= 1e-12
    ref_allocs = ref['oracle_allocations']
    got = head['oracle_allocations']
    ok_alloc = len(got) == len(ref_allocs)
    if ok_alloc:
        for a, b in zip(got, ref_allocs):
            for name in names:
                if int(a[name]) != int(b[name]):
                    ok_alloc = False
            if abs(a['sum_bytes'] - int(b['sum_bytes'])) != 0:
                ok_alloc = False
            if abs(a['mean_bits_per_byte'] - float(b['mean_bits_per_byte'])) > 1e-12:
                ok_alloc = False
    if not (ok_feasible and ok_uniform and ok_oracle and ok_head and ok_alloc):
        raise SystemExit(
            'W1 headroom search mismatch '
            f'feasible={ok_feasible} uniform={ok_uniform} oracle={ok_oracle} '
            f'headroom={ok_head} allocation={ok_alloc} '
            f'got={head["headroom_bits_per_byte"]:.12f} ref={ref["headroom_bits_per_byte"]:.12f}')
    print(
        f'W1b check-w1 headroom={head["headroom_bits_per_byte"]:.6f} '
        f'feasible={head["n_feasible"]} allocations={len(got)}',
        flush=True)
    return {
        'ok': True,
        'headroom_bits_per_byte': head['headroom_bits_per_byte'],
        'n_feasible': head['n_feasible'],
        'n_allocations': len(got),
    }


def _finite_run(channel, trained, held_rows):
    if not math.isfinite(trained['bits']):
        raise SystemExit(f'{channel}: training bits are not finite')
    for sl in trained['slices']:
        if not math.isfinite(sl['bits']):
            raise SystemExit(f'{channel}: slice bits are not finite')
    for row in held_rows:
        if not row['finite'] or not math.isfinite(row['bits']):
            raise SystemExit(f'{channel}: held-out bits are not finite at n={row["n"]}')


def _smoke_run(channel, train, held):
    trained, held_rows = run_learner(train, held, torch.device('cpu'), f'smoke_{channel}', (0, 4096))
    _finite_run(channel, trained, held_rows)
    return {
        'channel': channel,
        'train_bytes': int(len(train)),
        'held_bytes': int(len(held)),
        'bits': trained['bits'],
        'finite': True,
        'replay_steps': trained['replay_steps'],
        'heldout': held_rows,
        'wall_seconds': trained['wall_seconds'],
    }


def smoke_synthetic(channel):
    stream = BUILDERS[channel](b'')
    if len(stream) != STREAM_N:
        raise SystemExit(f'{channel}: stream length {len(stream)}')
    train = as_u8(stream[:4096])
    held = as_u8(stream[4096:4096 + 1024])
    print(f'W1b smoke {channel} train=4096 held=1024', flush=True)
    return _smoke_run(channel, train, held)


def smoke_bible(bible_path):
    bible_path = Path(bible_path)
    raw = bible_path.read_bytes()
    stream = build_bible(raw)
    if len(stream) < 4096 + 1024:
        raise SystemExit(f'bible stream is {len(stream)} bytes')
    train = as_u8(stream[:4096])
    held = as_u8(stream[4096:4096 + 1024])
    print('W1b smoke bible train=4096 held=1024', flush=True)
    row = _smoke_run('bible', train, held)
    diff = abs(row['bits'] - W1_SMOKE_BITS)
    row['w1_smoke_bits'] = W1_SMOKE_BITS
    row['abs_diff'] = diff
    row['within_1e-6'] = diff <= 1e-6
    row['source_sha256'] = sha256_bytes(raw)
    print(
        f'W1b smoke bible bits={row["bits"]:.12f} w1={W1_SMOKE_BITS:.12f} '
        f'abs_diff={diff:.3e}',
        flush=True)
    return row


def smoke(bible_path, out_dir, w1_path):
    configure()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    search = check_w1_search(w1_path)
    noise = smoke_synthetic('noise')
    dark = smoke_synthetic('dark')
    dark_by_n = {int(row['n']): row for row in dark['heldout']}
    dark_n0 = dark_by_n[0]['bits']
    dark_n4096 = dark_by_n[4096]['bits']
    dark_fell = dark_n4096 < dark_n0
    print(
        f'W1b smoke dark heldout n0={dark_n0:.6f} n4096={dark_n4096:.6f} fell={dark_fell}',
        flush=True)
    bible = smoke_bible(bible_path)
    payload = {
        'w1_search': search,
        'noise': noise,
        'dark': dark,
        'bible': bible,
        'noise_finite': True,
        'dark_finite': True,
        'dark_heldout_bits_n0': dark_n0,
        'dark_heldout_bits_n4096': dark_n4096,
        'dark_fell': dark_fell,
        'bible_within_1e-6': bible['within_1e-6'],
    }
    dump(out_dir / 'smoke.json', payload)
    if not dark_fell:
        raise SystemExit(
            f'dark held-out bits did not fall: n0={dark_n0:.6f} n4096={dark_n4096:.6f}')
    if not bible['within_1e-6']:
        raise SystemExit(f'bible smoke bits differ from W1 by {bible["abs_diff"]:.3e}')
    print('W1b smoke ok', flush=True)
    return payload


def load_built(channel, dna_path):
    if channel in SYNTHETIC_CHANNELS:
        raw = BUILDERS[channel](b'')
        source_path = 'synthetic'
    elif channel == 'dna':
        dna_path = Path(dna_path)
        if not dna_path.is_file():
            raise SystemExit(f'missing {dna_path}')
        raw = dna_path.read_bytes()
        source_path = 'data/w1b/human.txt'
    else:
        raise SystemExit(f'not a W1b channel: {channel}')
    stream = BUILDERS[channel](raw)
    if channel == 'dna':
        _assert_dna(raw, stream)
    elif channel == 'dark':
        if stream != bytes(STREAM_N):
            raise SystemExit('dark stream is not 2100000 zero bytes')
    elif channel == 'noise':
        if len(stream) != STREAM_N:
            raise SystemExit(f'noise stream length {len(stream)}')
    if len(stream) < TRAIN_N + HELD_N:
        raise SystemExit(f'{channel}: stream {len(stream)} is shorter than {TRAIN_N + HELD_N}')
    train, held, mode = split_stream(stream)
    if mode != 'prefix_2000000_100000' or len(train) != TRAIN_N or len(held) != HELD_N:
        raise SystemExit(f'{channel}: split {mode} train={len(train)} held={len(held)}')
    text, nbytes = excerpt_text(stream)
    info = {
        'channel': channel,
        'source_path': source_path,
        'source_bytes': len(raw),
        'source_sha256': sha256_bytes(raw),
        'stream_bytes': len(stream),
        'stream_sha256': sha256_bytes(stream),
        'split': mode,
        'train_bytes': len(train),
        'train_sha256': sha256_bytes(train),
        'held_bytes': len(held),
        'held_sha256': sha256_bytes(held),
        'excerpt': text if channel == 'dna' else None,
        'excerpt_bytes': nbytes if channel == 'dna' else 0,
        'numpy_version': np.__version__,
    }
    if channel == 'noise':
        info['prefix8'] = list(stream[:8])
    return info, train


def _assert_dna(raw, stream):
    bad = sorted(set(stream) - set(b'ACGTN\n'))
    if bad:
        raise SystemExit(f'dna stream has bytes {bad[:12]}')
    if any(bytes([d]) in stream for d in range(ord('0'), ord('9') + 1)):
        raise SystemExit('dna stream contains a digit')
    lines = raw.decode('utf-8').splitlines()
    data_rows = [ln for ln in lines[1:] if ln.strip()]
    first = data_rows[0].split('\t', 1)[0].strip().upper()
    head, sep, _rest = stream.partition(b'\n')
    if sep != b'\n' or head.decode('ascii') != first:
        raise SystemExit('dna first sequence does not match the file')
    if stream.count(b'\n') != len(data_rows) or not stream.endswith(b'\n'):
        raise SystemExit(
            f'dna lines {stream.count(b"\\n")} != rows {len(data_rows)}')
    if b'\t' in stream:
        raise SystemExit('dna stream contains a tab')


def write_streams(out_dir, dna_path):
    out_dir = Path(out_dir)
    rows = []
    trains = {}
    for channel in NEW_CHANNELS:
        info, train = load_built(channel, dna_path)
        print(
            f'W1b stream {channel} bytes={info["stream_bytes"]} '
            f'sha256={info["stream_sha256"]} train={info["train_sha256"]}',
            flush=True)
        rows.append(info)
        trains[channel] = train
    dump(out_dir / 'local_streams.json', {'channels': rows, 'numpy_version': np.__version__})
    return rows, trains


def write_baselines(out_dir, dna_path, trains=None):
    out_dir = Path(out_dir)
    if trains is None:
        _rows, trains = write_streams(out_dir, dna_path)
    baselines = {}
    for channel in NEW_CHANNELS:
        print(f'W1b baselines {channel}', flush=True)
        baselines[channel] = compress_baselines(trains[channel])
        baselines[channel]['train_sha256'] = sha256_bytes(trains[channel])
        dump(out_dir / 'baselines.json', {'channels': baselines, 'numpy_version': np.__version__})
    return baselines


def _held_lookup(payload):
    return {int(row['n']): row for row in payload['heldout']}


def _alloc_table(head):
    names = head['channels']
    lines = [
        '| ' + ' | '.join(names) + ' | Sum | Mean bits/byte |',
        '|' + '|'.join(['---:'] * (len(names) + 2)) + '|',
    ]
    for alloc in head['oracle_allocations']:
        cells = [str(alloc[name]) for name in names]
        cells.append(str(alloc['sum_bytes']))
        cells.append(_fmt(alloc['mean_bits_per_byte'], 6))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return lines


def _head_lines(title, head):
    lines = [
        f'### {title}',
        '',
        f'Channels: {", ".join(head["channels"])}.',
        f'Budget {head["budget_bytes"]} bytes. Grid {list(head["grid"])}. '
        f'Feasible allocations {head["n_feasible"]} of {head["n_tuples"]}.',
        '',
        f'Uniform mean L_c(500000): {_fmt(head["uniform_bits_per_byte"], 6)} bits/byte.',
        f'Oracle mean: {_fmt(head["oracle_bits_per_byte"], 6)} bits/byte.',
        f'Headroom (uniform − oracle): {_fmt(head["headroom_bits_per_byte"], 6)} bits/byte.',
        '',
        f'Oracle allocations: {len(head["oracle_allocations"])}.',
        '',
    ]
    lines.extend(_alloc_table(head))
    lines.append('')
    return lines


def render_results(doc):
    names = list(FULL_CHANNELS)
    lines = [
        '# W1b headroom',
        '',
        'These are separate learners per channel. The real machine shares one cortex, so this is an estimate.',
        '',
        'Bible, chess, squad and btc are the W1-0 runs. They were not rerun.',
        '',
        '## New streams',
        '',
        '| Channel | Source bytes | Source sha256 | Stream bytes | Stream sha256 | Split | Train bytes | Train sha256 | Held-out bytes | Held-out sha256 |',
        '|---|---:|---|---:|---|---|---:|---|---:|---|',
    ]
    for channel in NEW_CHANNELS:
        row = doc['channels'][channel]
        lines.append(
            f"| {channel} | {row['source_bytes']} | `{row['source_sha256']}` | "
            f"{row['stream_bytes']} | `{row['stream_sha256']}` | {row['split']} | "
            f"{row['train_bytes']} | `{row['train_sha256']}` | {row['held_bytes']} | "
            f"`{row['held_sha256']}` |")
    lines += [
        '',
        'Noise and dark are synthetic. Their source bytes are the stream.',
        '',
        '## Excerpt',
        '',
        '### dna',
        '',
        f'{doc["channels"]["dna"]["excerpt_bytes"]} bytes.',
        '',
        '```',
        doc['channels']['dna']['excerpt'].replace('```', '` ` `'),
        '```',
        '',
        '## Held-out L_c(n), bits/byte',
        '',
        '| n | ' + ' | '.join(names) + ' |',
        '|---:|' + '|'.join(['---:'] * len(names)) + '|',
    ]
    for n in CHECKPOINTS:
        cells = [_fmt(doc['channels'][channel]['heldout_bpb'][str(n)]) for channel in names]
        lines.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    lines += [
        '',
        'Held-out bits:',
        '',
        '| n | ' + ' | '.join(names) + ' |',
        '|---:|' + '|'.join(['---:'] * len(names)) + '|',
    ]
    for n in CHECKPOINTS:
        cells = [_fmt(doc['channels'][channel]['heldout_bits'][str(n)], 3) for channel in names]
        lines.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Prequential 1 MB slices', '']
    for channel in NEW_CHANNELS:
        row = doc['channels'][channel]['prequential']
        lines += [
            f'### {channel}',
            '',
            f'Bits {_fmt(row["bits"], 6)}. Bits/byte {_fmt(row["bits_per_byte"])}. '
            f'Replay steps {row["replay_steps"]}. Traffic {_fmt(row["traffic"])}. '
            f'Wall seconds {row["wall_seconds"]:.3f}.',
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
        '## xz -9e and PPMd order 16',
        '',
        'Training bytes of the new channels. xz is `xz -9e -T1`. PPMd is variant H, order 16, mem 1073741824.',
        'PPMd is encoded in pieces of 65536 bytes on one encoder, then flushed.',
        '',
        '| Channel | xz bytes | xz bits/byte | PPMd bytes | PPMd bits/byte |',
        '|---|---:|---:|---:|---:|',
    ]
    for channel in NEW_CHANNELS:
        base = doc['baselines'][channel]
        lines.append(
            f"| {channel} | {base['xz_9e']['compressed_bytes']} | "
            f"{_fmt(base['xz_9e']['bits_per_byte'])} | "
            f"{base['ppmd_h_order_16']['compressed_bytes']} | "
            f"{_fmt(base['ppmd_h_order_16']['bits_per_byte'])} |")
    lines += ['', '## Headroom', '']
    lines.extend(_head_lines('full', doc['headroom']['full']))
    lines.extend(_head_lines('learnable', doc['headroom']['learnable']))
    if doc['learnable_headroom_below_0_05']:
        lines.append(
            'Most of the gain comes from avoiding noise and dark rather than from choosing between learnable streams.')
        lines.append('')
    if doc.get('noise_generator_note'):
        lines.append(doc['noise_generator_note'])
        lines.append('')
    row = doc['kernels'][0]
    lines += [
        '## Kernel',
        '',
        'Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that',
        'clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.',
        'Script wall is time.monotonic from kernel_main entry through the result write. Queue time',
        'is not included.',
        '',
        '| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours | GPU |',
        '|---|---:|---|---:|---|---:|---:|---|',
        f"| {row['kernel_id']} | {row['version']} | {row['status']} | "
        f"{row['elapsed_seconds']:.3f} | {row['elapsed_source']} | "
        f"{row['session_wall_seconds']:.3f} | {row['gpu_hours']:.4f} | {row['gpu_name']} |",
        '',
        f'Total GPU hours: {_fmt(doc["gpu_hours_total"], 4)}.',
        '',
        '## Smoke',
        '',
        'CPU. Noise and dark: 4096 training bytes, 1024 held-out bytes, checkpoints 0 and 4096.',
        f'Dark held-out bits at n=0: {_fmt(doc["smoke"]["dark_heldout_bits_n0"], 6)}.',
        f'Dark held-out bits at n=4096: {_fmt(doc["smoke"]["dark_heldout_bits_n4096"], 6)}.',
        f'Dark held-out bits fell: {doc["smoke"]["dark_fell"]}.',
        f'Noise training finite: {doc["smoke"]["noise_finite"]}. Dark training finite: {doc["smoke"]["dark_finite"]}.',
        f'Bible training bits: {doc["smoke"]["bible"]["bits"]:.12f}.',
        f'W1-0 smoke bits: {doc["smoke"]["bible"]["w1_smoke_bits"]:.12f}.',
        f'Absolute difference: {doc["smoke"]["bible"]["abs_diff"]:.3e}.',
        f'Within 1e-6: {doc["smoke"]["bible"]["within_1e-6"]}.',
        '',
        '## Protocol',
        '',
        '- One fresh learner per channel. lr 0.0003. Replay batch 8. Seed 0. Replay RNG 30000.',
        '- ByteGRU embedding 64, hidden 1024, 1 layer, alphabet 256. Adam, clip 1.0. 3627264 parameters.',
        '- Prequential: each chunk is scored before that chunk is trained. Live state is carried and detached.',
        '- Replay state is zero. Replay starts are uniform on 0 .. n_seen−128 inclusive, with replacement.',
        '- The learner advances to each checkpoint with that loop. A checkpoint that is not a multiple of 128 ends on a short chunk, then training continues.',
        '- Held-out evaluation uses no gradient, a zero initial state, chunks of 128, and carried state. L_c(n) is live −log2 p bits divided by held-out bytes.',
        '- The training stream is the first 2000000 bytes. The held-out block is the next 100000 bytes.',
        '- Bible, chess, squad and btc L_c(n) are the W1-0 curves. Those channels were not rerun.',
        '- DNA bytes are the sequence column of human.txt, one sequence per line, in file order. The header and the class labels are dropped. Bases are A, C, G, T and N.',
        '- Noise bytes are numpy.random.default_rng(777).integers(0, 256, size=2100000, dtype=uint8).',
        '- Dark bytes are 2100000 zero bytes.',
        '- xz -9e -T1 and PPMd variant H, order 16, mem 1073741824, on the training bytes of dna, noise and dark. PPMd is encoded in 65536-byte pieces on one encoder, then flushed.',
        '- For a set of k channels the budget is 500000 times k bytes. Each n_c is on the checkpoint grid and the sum of n_c is at most the budget.',
        '- Uniform is the mean of L_c(500000). Oracle is the minimum mean of L_c(n_c) over that grid, by exhaustive search of the 7^k tuples. Headroom is uniform minus oracle.',
        '- Full set: bible, chess, squad, btc, dna, noise, dark. Learnable set: bible, chess, squad, btc, dna.',
        '',
    ]
    return '\n'.join(lines)


def _require_run(channel, payload):
    if payload.get('channel') != channel:
        raise SystemExit(f'{channel} payload channel is {payload.get("channel")}')
    if abs(float(payload['lr']) - LR) > 1e-15 or int(payload['replay_batch']) != 8:
        raise SystemExit(f'{channel} lr/batch {payload["lr"]} {payload["replay_batch"]}')
    if int(payload['seed']) != 0 or not payload['finite']:
        raise SystemExit(f'{channel} is not a finite seed-0 run')
    if int(payload['train_bytes']) != TRAIN_N or int(payload['held_bytes']) != HELD_N:
        raise SystemExit(f'{channel} split train={payload["train_bytes"]} held={payload["held_bytes"]}')
    if int(payload['parameters']) != PARAMS:
        raise SystemExit(f'{channel} parameters {payload["parameters"]}')
    held = _held_lookup(payload)
    missing = [n for n in CHECKPOINTS if n not in held]
    if missing:
        raise SystemExit(f'{channel} missing checkpoints {missing}')
    for n in CHECKPOINTS:
        if int(held[n]['held_bytes']) != HELD_N or not held[n]['finite']:
            raise SystemExit(f'{channel} held-out n={n} is not a finite {HELD_N}-byte score')
    return held


def assemble(out_dir, dna_path, w1_path, kernel_version, kernel_status, log_path):
    out_dir = Path(out_dir)
    raw_dir = out_dir / 'raw'
    smoke_payload = json.loads((out_dir / 'smoke.json').read_text())
    if not smoke_payload['bible']['within_1e-6'] or not smoke_payload['dark_fell']:
        raise SystemExit('smoke checks did not pass')
    w1 = json.loads(Path(w1_path).read_text())
    kernel_meta = json.loads((raw_dir / 'kernel.json').read_text())
    if kernel_meta.get('kernel_id') != 'ezzzzzx/w1b-headroom':
        raise SystemExit(f'kernel id {kernel_meta.get("kernel_id")}')
    payloads = {}
    trains = {}
    for channel in NEW_CHANNELS:
        payload = json.loads((raw_dir / f'{channel}.json').read_text())
        _require_run(channel, payload)
        blob = (raw_dir / f'{channel}.train.bin').read_bytes()
        if len(blob) != TRAIN_N or sha256_bytes(blob) != payload['train_sha256']:
            raise SystemExit(f'{channel} train bin does not match the scored stream')
        payloads[channel] = payload
        trains[channel] = blob
    local_rows, _local_trains = write_streams(out_dir, dna_path)
    local_by = {row['channel']: row for row in local_rows}
    dna_local = local_by['dna']
    dna = payloads['dna']
    if dna_local['source_sha256'] != dna['source_sha256']:
        raise SystemExit(
            f'dna source sha local {dna_local["source_sha256"]} != kernel {dna["source_sha256"]}')
    if dna_local['stream_sha256'] != dna['stream_sha256']:
        raise SystemExit(
            f'dna stream sha local {dna_local["stream_sha256"]} != kernel {dna["stream_sha256"]}')
    if local_by['dark']['stream_sha256'] != payloads['dark']['stream_sha256']:
        raise SystemExit('dark stream sha local != kernel')
    noise_note = None
    if local_by['noise']['train_sha256'] != payloads['noise']['train_sha256']:
        noise_note = (
            f'Local numpy {np.__version__} noise train sha256 `{local_by["noise"]["train_sha256"]}`.'
            f' Kernel numpy {payloads["noise"].get("numpy_version")} noise train sha256'
            f' `{payloads["noise"]["train_sha256"]}`. Baselines use the kernel training bytes.'
        )
        print('W1b ' + noise_note, flush=True)
    base_path = out_dir / 'baselines.json'
    reuse = False
    if base_path.is_file():
        existing = json.loads(base_path.read_text())['channels']
        reuse = all(
            existing.get(channel, {}).get('train_sha256') == payloads[channel]['train_sha256']
            and existing.get(channel, {}).get('xz_9e', {}).get('roundtrip') is True
            and existing.get(channel, {}).get('ppmd_h_order_16', {}).get('roundtrip') is True
            for channel in NEW_CHANNELS)
    if reuse:
        baselines = existing
        print('W1b baselines reused', flush=True)
    else:
        baselines = write_baselines(out_dir, dna_path, trains)
    bpb = {}
    bits = {}
    for channel in W1_CHANNELS:
        held_rows = w1['channels'][channel]['heldout']
        table = _bpb_table(held_rows)
        bpb[channel] = table
        bits[channel] = {int(row['n']): float(row['bits']) for row in held_rows}
        if int(w1['channels'][channel]['held_bytes']) != HELD_N:
            raise SystemExit(f'{channel} W1 held-out length changed')
    for channel in NEW_CHANNELS:
        held = _held_lookup(payloads[channel])
        bpb[channel] = {n: held[n]['bits_per_byte'] for n in CHECKPOINTS}
        bits[channel] = {n: held[n]['bits'] for n in CHECKPOINTS}
    print('W1b headroom full search', flush=True)
    full = compute_set_headroom(FULL_CHANNELS, bpb)
    print(
        f'W1b full uniform={full["uniform_bits_per_byte"]:.6f} '
        f'oracle={full["oracle_bits_per_byte"]:.6f} '
        f'headroom={full["headroom_bits_per_byte"]:.6f}',
        flush=True)
    print('W1b headroom learnable search', flush=True)
    learnable = compute_set_headroom(LEARNABLE_CHANNELS, bpb)
    print(
        f'W1b learnable uniform={learnable["uniform_bits_per_byte"]:.6f} '
        f'oracle={learnable["oracle_bits_per_byte"]:.6f} '
        f'headroom={learnable["headroom_bits_per_byte"]:.6f}',
        flush=True)
    log_s = None
    if log_path and Path(log_path).is_file():
        log_s = _log_elapsed(log_path)
    script_s = float(kernel_meta['session_wall_seconds'])
    if log_s is None:
        elapsed = script_s
        source = 'script_wall'
    else:
        elapsed = log_s
        source = 'log_clock'
    status = _status_name(kernel_status)
    kernel_row = {
        'kernel_id': 'ezzzzzx/w1b-headroom',
        'version': int(kernel_version),
        'status': status,
        'elapsed_seconds': elapsed,
        'elapsed_source': source,
        'session_wall_seconds': script_s,
        'log_elapsed_seconds': log_s,
        'gpu_name': kernel_meta.get('gpu_name'),
        'gpu_hours': elapsed / 3600.0,
        'channels': [row['channel'] for row in kernel_meta['channels']],
    }
    channels_out = {}
    for channel in W1_CHANNELS:
        src = w1['channels'][channel]
        channels_out[channel] = {
            'rerun': False,
            'source_bytes': src['source_bytes'],
            'source_sha256': src['source_sha256'],
            'stream_bytes': src['stream_bytes'],
            'stream_sha256': src['stream_sha256'],
            'split': src['split'],
            'train_bytes': src['train_bytes'],
            'train_sha256': src['train_sha256'],
            'held_bytes': src['held_bytes'],
            'held_sha256': src['held_sha256'],
            'heldout_bpb': {str(n): bpb[channel][n] for n in CHECKPOINTS},
            'heldout_bits': {str(n): bits[channel][n] for n in CHECKPOINTS},
        }
    for channel in NEW_CHANNELS:
        src = payloads[channel]
        channels_out[channel] = {
            'rerun': True,
            'source_bytes': src['source_bytes'],
            'source_sha256': src['source_sha256'],
            'stream_bytes': src['stream_bytes'],
            'stream_sha256': src['stream_sha256'],
            'split': src['split'],
            'train_bytes': src['train_bytes'],
            'train_sha256': src['train_sha256'],
            'held_bytes': src['held_bytes'],
            'held_sha256': src['held_sha256'],
            'excerpt': src['excerpt'] if channel == 'dna' else None,
            'excerpt_bytes': src['excerpt_bytes'] if channel == 'dna' else 0,
            'kernel_id': src['kernel_id'],
            'gpu_name': src['gpu_name'],
            'torch_version': src['torch_version'],
            'numpy_version': src['numpy_version'],
            'lr': src['lr'],
            'replay_batch': src['replay_batch'],
            'seed': src['seed'],
            'parameters': src['parameters'],
            'finite': src['finite'],
            'prequential': src['prequential'],
            'heldout': src['heldout'],
            'heldout_bpb': {str(n): bpb[channel][n] for n in CHECKPOINTS},
            'heldout_bits': {str(n): bits[channel][n] for n in CHECKPOINTS},
            'channel_wall_seconds': src['channel_wall_seconds'],
            'local_stream_sha256': local_by[channel]['stream_sha256'],
            'local_train_sha256': local_by[channel]['train_sha256'],
        }
    below = learnable['headroom_bits_per_byte'] < 0.05
    doc = {
        'separate_learners_per_channel': True,
        'w1_curves': 'results/w1_headroom/results.json',
        'channels': channels_out,
        'headroom': {'full': full, 'learnable': learnable},
        'learnable_headroom_below_0_05': below,
        'noise_generator_note': noise_note,
        'baselines': baselines,
        'kernels': [kernel_row],
        'gpu_hours_total': kernel_row['gpu_hours'],
        'smoke': smoke_payload,
    }
    text = render_results(doc)
    (out_dir / 'RESULTS.md').write_text(text)
    dump(out_dir / 'results.json', doc)
    dump(out_dir / 'kernels.json', {'w1b': kernel_row})
    print(f'W1b wrote {out_dir / "RESULTS.md"} and results.json', flush=True)
    return doc


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sm = sub.add_parser('smoke')
    sm.add_argument('--bible', default='data/pg10.txt')
    sm.add_argument('--out', default='results/w1b_headroom')
    sm.add_argument('--w1', default='results/w1_headroom/results.json')
    st = sub.add_parser('streams')
    st.add_argument('--dna', default='data/w1b/human.txt')
    st.add_argument('--out', default='results/w1b_headroom')
    ba = sub.add_parser('baselines')
    ba.add_argument('--dna', default='data/w1b/human.txt')
    ba.add_argument('--out', default='results/w1b_headroom')
    ck = sub.add_parser('check-w1')
    ck.add_argument('--w1', default='results/w1_headroom/results.json')
    asm = sub.add_parser('assemble')
    asm.add_argument('--dna', default='data/w1b/human.txt')
    asm.add_argument('--out', default='results/w1b_headroom')
    asm.add_argument('--w1', default='results/w1_headroom/results.json')
    asm.add_argument('--kernel-version', type=int, required=True)
    asm.add_argument('--kernel-status', required=True)
    asm.add_argument('--log', required=True)
    args = parser.parse_args()
    if args.cmd == 'smoke':
        smoke(args.bible, args.out, args.w1)
    elif args.cmd == 'streams':
        write_streams(args.out, args.dna)
    elif args.cmd == 'baselines':
        write_baselines(args.out, args.dna)
    elif args.cmd == 'check-w1':
        check_w1_search(args.w1)
    elif args.cmd == 'assemble':
        assemble(args.out, args.dna, args.w1, args.kernel_version, args.kernel_status, args.log)
    else:
        raise SystemExit(f'unknown cmd {args.cmd}')


if __name__ == '__main__':
    main()
