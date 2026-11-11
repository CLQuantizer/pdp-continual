"""W1-0: real-data channels and oracle allocation headroom. Numbers only. No verdict.

The cortex, the chunking, the score-before-step order, the clip, the replay
window, and the seeds are the H29c setting of h29b_enwik8.py: lr 3e-4, replay
batch 8, seed 0. Held-out evaluation is extra. Does not commit.
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
CHANNELS = ('bible', 'chess', 'squad', 'btc')
EXCERPT_N = 300
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
}
LOCAL_SOURCES = {
    'bible': Path('data/pg10.txt'),
    'chess': Path('data/kaggle_samples/lichess_games/games.csv'),
    'squad': Path('data/kaggle_samples/squad/dev-v1.1.json'),
    'btc': Path('data/kaggle_samples/btc_lob5/BTC_5min.csv'),
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


BUILDERS = {
    'bible': build_bible,
    'chess': build_chess,
    'squad': build_squad,
    'btc': build_btc,
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
    hits = [
        p for p in root.rglob(name)
        if p.is_file() and 'pdp-code-w1' not in p.parts
    ]
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
        f'W1 kernel={kernel_id} gpu={gpu_name} torch={torch.__version__} '
        f'channels={list(channels)}',
        flush=True)
    summaries = []
    for channel in channels:
        path = find_kaggle_source(channel)
        print(f'W1 source {channel} {path} bytes={path.stat().st_size}', flush=True)
        payload = run_channel_bytes(channel, path.read_bytes(), device, kernel_id, gpu_name)
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
        'session_wall_seconds': session,
        'channels': summaries,
    }
    dump(out / 'kernel.json', meta)
    print('KERNEL_DONE ' + json.dumps(meta), flush=True)
    return meta


def _reference_bits(train):
    folder = Path('/tmp/w1_smoke_ref')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'train.bin'
    train.tofile(path)
    script = folder / 'ref.py'
    repo = str(Path(__file__).resolve().parent)
    script.write_text(
        'import sys\n'
        f'sys.path.insert(0, {repo!r})\n'
        'import numpy as np\n'
        'import torch\n'
        'import h29b_enwik8\n'
        'h29b_enwik8.configure()\n'
        'raw = np.fromfile(sys.argv[1], dtype=np.uint8)\n'
        'n = int(sys.argv[2])\n'
        'row = h29b_enwik8.run_arm(raw, n, 3e-4, 8, "equiv", torch.device("cpu"))\n'
        'print("REF_BITS", f\'{row["bits"]:.12f}\', int(row["finite"]), int(row["replay_steps"]))\n'
    )
    proc = subprocess.run(
        [sys.executable, str(script), str(path), str(len(train))],
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(
            'h29 reference failed\n' + proc.stdout[-2000:] + '\n' + proc.stderr[-2000:])
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith('REF_BITS ')), None)
    if line is None:
        raise SystemExit('h29 reference produced no REF_BITS line\n' + proc.stdout[-2000:])
    parts = line.split()
    return {
        'bits': float(parts[1]),
        'finite': bool(int(parts[2])),
        'replay_steps': int(parts[3]),
    }


def smoke(bible_path, out_dir):
    configure()
    bible_path = Path(bible_path)
    raw = bible_path.read_bytes()
    stream = build_bible(raw)
    if len(stream) < 4096 + 1024:
        raise SystemExit(f'bible stream is {len(stream)} bytes')
    train = as_u8(stream[:4096])
    held = as_u8(stream[4096:4096 + 1024])
    device = torch.device('cpu')
    print('W1 smoke official checkpoints 0 and 4096', flush=True)
    trained, held_rows = run_learner(train, held, device, 'smoke', (0, 4096))
    ref = _reference_bits(train)
    diff = abs(trained['bits'] - ref['bits'])
    print(
        f'W1 smoke bits={trained["bits"]:.12f} ref={ref["bits"]:.12f} '
        f'abs_diff={diff:.3e} finite={trained["bits"] == trained["bits"]}',
        flush=True)
    held_finite = all(row['finite'] and np.isfinite(row['bits']) for row in held_rows)
    train_finite = bool(np.isfinite(trained['bits'])) and all(
        np.isfinite(sl['bits']) for sl in trained['slices'])
    if not train_finite or not held_finite or not ref['finite']:
        raise SystemExit('smoke bits are not finite')
    if diff > 1e-6:
        raise SystemExit(f'smoke training bits differ from h29 by {diff:.3e}')
    if trained['replay_steps'] != 32 or ref['replay_steps'] != 32:
        raise SystemExit(
            f'smoke replay steps w1={trained["replay_steps"]} h29={ref["replay_steps"]}')
    print('W1 smoke isolation eval at 2048', flush=True)
    trained_mid, _ = run_learner(train, held, device, 'smoke_mid', (0, 2048, 4096))
    mid_diff = abs(trained_mid['bits'] - ref['bits'])
    print(f'W1 smoke mid abs_diff={mid_diff:.3e}', flush=True)
    if mid_diff > 1e-6:
        raise SystemExit(f'mid held-out eval changed training bits by {mid_diff:.3e}')
    payload = {
        'bible_source_sha256': sha256_bytes(raw),
        'bible_source_bytes': len(raw),
        'stream_bytes': len(stream),
        'stream_sha256': sha256_bytes(stream),
        'train_bytes': 4096,
        'held_bytes': 1024,
        'lr': LR,
        'replay_batch': 8,
        'seed': SEED,
        'w1_bits': trained['bits'],
        'h29_bits': ref['bits'],
        'abs_diff': diff,
        'within_1e-6': diff <= 1e-6,
        'train_finite': train_finite,
        'held_finite': held_finite,
        'replay_steps': trained['replay_steps'],
        'h29_replay_steps': ref['replay_steps'],
        'mid_eval_abs_diff': mid_diff,
        'heldout': held_rows,
        'wall_seconds': trained['wall_seconds'],
    }
    dump(Path(out_dir) / 'smoke.json', payload)
    print('W1 smoke ok', flush=True)
    return payload


def write_local_streams(out_dir):
    rows = []
    trains = {}
    for channel in CHANNELS:
        info, train, _held = load_local_channel(channel)
        print(
            f'W1 stream {channel} bytes={info["stream_bytes"]} '
            f'sha256={info["stream_sha256"]} split={info["split"]} '
            f'train={info["train_bytes"]} held={info["held_bytes"]}',
            flush=True)
        rows.append(info)
        trains[channel] = train
    dump(Path(out_dir) / 'local_streams.json', {'channels': rows})
    return rows, trains


def write_baselines(out_dir):
    rows, trains = write_local_streams(out_dir)
    by_name = {row['channel']: row for row in rows}
    baselines = {}
    for channel in CHANNELS:
        print(f'W1 baselines {channel}', flush=True)
        baselines[channel] = compress_baselines(trains[channel])
        baselines[channel]['train_sha256'] = by_name[channel]['train_sha256']
        dump(Path(out_dir) / 'baselines.json', {'channels': baselines})
    return baselines


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


def _held_table(payload):
    return {int(row['n']): row for row in payload['heldout']}


def compute_headroom(channels):
    tables = []
    for channel in CHANNELS:
        table = _held_table(channels[channel])
        missing = [n for n in CHECKPOINTS if n not in table]
        if missing:
            raise SystemExit(f'{channel} missing checkpoints {missing}')
        tables.append(table)
    uniform_alloc = (500_000, 500_000, 500_000, 500_000)
    uniform = sum(tables[i][500_000]['bits_per_byte'] for i in range(4)) / 4.0
    best = None
    ties = []
    n_feasible = 0
    for alloc in itertools.product(CHECKPOINTS, repeat=4):
        if sum(alloc) > BUDGET:
            continue
        n_feasible += 1
        mean = sum(tables[i][alloc[i]]['bits_per_byte'] for i in range(4)) / 4.0
        if best is None or mean < best:
            best = mean
            ties = [alloc]
        elif abs(mean - best) <= 1e-12:
            ties.append(alloc)
    if best is None:
        raise SystemExit('no feasible allocation')
    allocations = []
    for alloc in ties:
        allocations.append({
            'bible': alloc[0],
            'chess': alloc[1],
            'squad': alloc[2],
            'btc': alloc[3],
            'sum_bytes': int(sum(alloc)),
            'mean_bits_per_byte': sum(tables[i][alloc[i]]['bits_per_byte'] for i in range(4)) / 4.0,
        })
    return {
        'budget_bytes': BUDGET,
        'grid': list(CHECKPOINTS),
        'n_grid': len(CHECKPOINTS),
        'n_tuples': len(CHECKPOINTS) ** 4,
        'n_feasible': n_feasible,
        'uniform_allocation': {
            'bible': uniform_alloc[0],
            'chess': uniform_alloc[1],
            'squad': uniform_alloc[2],
            'btc': uniform_alloc[3],
        },
        'uniform_bits_per_byte': uniform,
        'oracle_bits_per_byte': best,
        'headroom_bits_per_byte': uniform - best,
        'oracle_allocations': allocations,
    }


def assemble(out_dir):
    out_dir = Path(out_dir)
    local = json.loads((out_dir / 'local_streams.json').read_text())
    local_by = {row['channel']: row for row in local['channels']}
    baselines = json.loads((out_dir / 'baselines.json').read_text())['channels']
    kernels_meta = json.loads((out_dir / 'kernels.json').read_text())
    smoke_payload = json.loads((out_dir / 'smoke.json').read_text())
    loaded = {}
    for channel in CHANNELS:
        hits = list((out_dir / 'raw').rglob(f'{channel}.json'))
        if len(hits) != 1:
            raise SystemExit(f'expected one {channel}.json, found {hits}')
        payload = json.loads(hits[0].read_text())
        if payload.get('channel') != channel:
            raise SystemExit(f'{hits[0]} channel is {payload.get("channel")}')
        if payload['stream_sha256'] != local_by[channel]['stream_sha256']:
            raise SystemExit(
                f'{channel} stream sha256 kernel {payload["stream_sha256"]} '
                f'!= local {local_by[channel]["stream_sha256"]}')
        if payload['train_sha256'] != baselines[channel]['train_sha256']:
            raise SystemExit(f'{channel} baseline is not on the kernel training bytes')
        if abs(payload['lr'] - LR) > 1e-15 or int(payload['replay_batch']) != 8:
            raise SystemExit(f'{channel} lr/batch {payload["lr"]} {payload["replay_batch"]}')
        if int(payload['seed']) != 0 or not payload['finite']:
            raise SystemExit(f'{channel} is not a finite seed-0 run')
        if payload['train_bytes'] != TRAIN_N or payload['held_bytes'] != HELD_N:
            raise SystemExit(
                f'{channel} split train={payload["train_bytes"]} held={payload["held_bytes"]}')
        loaded[channel] = payload
    head = compute_headroom(loaded)
    kernel_rows = []
    for name, meta in kernels_meta.items():
        log_s = meta.get('log_elapsed_seconds')
        script_s = meta['session_wall_seconds']
        if log_s is None:
            elapsed = script_s
            source = 'script_wall'
        else:
            elapsed = log_s
            source = 'log_clock'
        kernel_rows.append({
            'name': name,
            'kernel_id': meta['id'],
            'version': meta.get('version'),
            'status': meta.get('status'),
            'elapsed_seconds': elapsed,
            'elapsed_source': source,
            'session_wall_seconds': script_s,
            'log_elapsed_seconds': log_s,
            'gpu_name': meta.get('gpu_name'),
            'gpu_hours': elapsed / 3600.0,
            'channels': meta.get('channels'),
        })
    gpu_hours = sum(row['gpu_hours'] for row in kernel_rows)
    lines = [
        '# W1 headroom',
        '',
        'These are separate learners per channel. The real machine shares one cortex, so this is an estimate.',
        '',
        '## Streams',
        '',
        '| Channel | Source bytes | Source sha256 | Stream bytes | Stream sha256 | Split | Train bytes | Train sha256 | Held-out bytes | Held-out sha256 |',
        '|---|---:|---|---:|---|---|---:|---|---:|---|',
    ]
    for channel in CHANNELS:
        row = loaded[channel]
        lines.append(
            f"| {channel} | {row['source_bytes']} | `{row['source_sha256']}` | "
            f"{row['stream_bytes']} | `{row['stream_sha256']}` | {row['split']} | "
            f"{row['train_bytes']} | `{row['train_sha256']}` | {row['held_bytes']} | "
            f"`{row['held_sha256']}` |")
    lines += ['', 'No stream was shorter than 2100000 bytes.', '', '## Excerpts', '']
    for channel in CHANNELS:
        row = loaded[channel]
        lines.append(f'### {channel}')
        lines.append('')
        lines.append(f'{row["excerpt_bytes"]} bytes.')
        lines.append('')
        lines.append('```')
        lines.append(row['excerpt'].replace('```', '` ` `'))
        lines.append('```')
        lines.append('')
    lines += [
        'Chess excerpt is the moves column only.',
        '',
        '## Held-out L_c(n), bits/byte',
        '',
        '| n | bible | chess | squad | btc |',
        '|---:|---:|---:|---:|---:|',
    ]
    for n in CHECKPOINTS:
        cells = [_fmt(_held_table(loaded[channel])[n]['bits_per_byte']) for channel in CHANNELS]
        lines.append(f"| {n} | {' | '.join(cells)} |")
    lines += ['', 'Held-out bits:', '', '| n | bible | chess | squad | btc |', '|---:|---:|---:|---:|---:|']
    for n in CHECKPOINTS:
        cells = [_fmt(_held_table(loaded[channel])[n]['bits'], 3) for channel in CHANNELS]
        lines.append(f"| {n} | {' | '.join(cells)} |")
    lines += ['', '## Prequential 1 MB slices', '']
    for channel in CHANNELS:
        row = loaded[channel]['prequential']
        lines += [
            f'### {channel}',
            '',
            f'Bits { _fmt(row["bits"], 6) }. Bits/byte {_fmt(row["bits_per_byte"])}. '
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
        'Training stream only. xz is `xz -9e -T1`. PPMd is variant H, order 16, mem 1073741824.',
        'PPMd is encoded in pieces of 65536 bytes on one encoder, then flushed.',
        'A single pyppmd.compress call does not roundtrip on the chess stream. Piece encoding does.',
        'On bible, squad and btc the piece encoding matches that one-shot byte string.',
        '',
        '| Channel | xz bytes | xz bits/byte | PPMd bytes | PPMd bits/byte |',
        '|---|---:|---:|---:|---:|',
    ]
    for channel in CHANNELS:
        base = baselines[channel]
        lines.append(
            f"| {channel} | {base['xz_9e']['compressed_bytes']} | "
            f"{_fmt(base['xz_9e']['bits_per_byte'])} | "
            f"{base['ppmd_h_order_16']['compressed_bytes']} | "
            f"{_fmt(base['ppmd_h_order_16']['bits_per_byte'])} |")
    lines += [
        '',
        '## Headroom',
        '',
        f'Budget {BUDGET} bytes. Grid {list(CHECKPOINTS)}. '
        f'Feasible allocations {head["n_feasible"]} of {head["n_tuples"]}.',
        '',
        f'Uniform mean L_c(500000): {_fmt(head["uniform_bits_per_byte"], 6)} bits/byte.',
        f'Oracle mean: {_fmt(head["oracle_bits_per_byte"], 6)} bits/byte.',
        f'Headroom (uniform − oracle): {_fmt(head["headroom_bits_per_byte"], 6)} bits/byte.',
        '',
        f'Oracle allocations: {len(head["oracle_allocations"])}.',
        '',
        '| bible | chess | squad | btc | Sum | Mean bits/byte |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for alloc in head['oracle_allocations']:
        lines.append(
            f"| {alloc['bible']} | {alloc['chess']} | {alloc['squad']} | {alloc['btc']} | "
            f"{alloc['sum_bytes']} | {_fmt(alloc['mean_bits_per_byte'], 6)} |")
    lines += [
        '',
        '## Kernels',
        '',
        'Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that',
        'clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.',
        'Script wall is time.monotonic from kernel_main entry through the result write. Queue time',
        'is not included. Total GPU hours is the sum across the two kernels.',
        '',
        '| Kernel | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours | GPU |',
        '|---|---:|---|---:|---|---:|---:|---|',
    ]
    for row in kernel_rows:
        lines.append(
            f"| {row['kernel_id']} | {row['version']} | {row['status']} | "
            f"{row['elapsed_seconds']:.3f} | {row['elapsed_source']} | "
            f"{row['session_wall_seconds']:.3f} | {row['gpu_hours']:.4f} | {row['gpu_name']} |")
    lines += [
        '',
        f'Total GPU hours: {_fmt(gpu_hours, 4)}.',
        '',
        '## Smoke',
        '',
        'CPU. Bible training bytes 4096. Held-out bytes 1024. Checkpoints 0 and 4096.',
        f'W1 bits: {smoke_payload["w1_bits"]:.12f}.',
        f'H29c-equivalent bits: {smoke_payload["h29_bits"]:.12f}.',
        f'Absolute difference: {smoke_payload["abs_diff"]:.3e}.',
        f'Within 1e-6: {smoke_payload["within_1e-6"]}.',
        f'Training finite: {smoke_payload["train_finite"]}. Held-out finite: {smoke_payload["held_finite"]}.',
        f'Replay steps: {smoke_payload["replay_steps"]}.',
        f'Mid-eval absolute difference: {smoke_payload["mid_eval_abs_diff"]:.3e}.',
        '',
        '| n | Held-out bits | Bits/byte |',
        '|---:|---:|---:|',
    ]
    for row in smoke_payload['heldout']:
        lines.append(f"| {row['n']} | {_fmt(row['bits'], 6)} | {_fmt(row['bits_per_byte'])} |")
    lines += [
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
        '- Bible bytes are the text between the *** START line and the *** END line.',
        '- Chess bytes are the moves column, one game per line. Other columns are dropped.',
        '- SQuAD bytes are, per paragraph, the context, then each question and its first answer text, newline-separated.',
        '- BTC bytes are the raw csv, header included.',
        '- xz -9e -T1 and PPMd variant H, order 16, mem 1073741824, on the training bytes. PPMd is encoded in 65536-byte pieces on one encoder.',
        '- Uniform is the mean of L_c(500000). Oracle is the minimum mean of L_c(n_c) over the 7^4 grid with the sum of n_c at most 2000000. Headroom is uniform minus oracle.',
        '',
    ]
    text = '\n'.join(lines)
    (out_dir / 'RESULTS.md').write_text(text)
    merged = {
        'separate_learners_per_channel': True,
        'channels': {},
        'headroom': head,
        'kernels': kernel_rows,
        'gpu_hours_total': gpu_hours,
        'smoke': smoke_payload,
        'baselines': baselines,
    }
    for channel in CHANNELS:
        row = loaded[channel]
        merged['channels'][channel] = {
            'source_bytes': row['source_bytes'],
            'source_sha256': row['source_sha256'],
            'stream_bytes': row['stream_bytes'],
            'stream_sha256': row['stream_sha256'],
            'split': row['split'],
            'train_bytes': row['train_bytes'],
            'train_sha256': row['train_sha256'],
            'held_bytes': row['held_bytes'],
            'held_sha256': row['held_sha256'],
            'excerpt': row['excerpt'],
            'excerpt_bytes': row['excerpt_bytes'],
            'lr': row['lr'],
            'replay_batch': row['replay_batch'],
            'seed': row['seed'],
            'replay_seed': row['replay_seed'],
            'parameters': row['parameters'],
            'gpu_name': row['gpu_name'],
            'kernel_id': row['kernel_id'],
            'torch_version': row['torch_version'],
            'prequential': row['prequential'],
            'heldout': row['heldout'],
            'channel_wall_seconds': row['channel_wall_seconds'],
            'finite': row['finite'],
            'baselines': baselines[channel],
        }
    dump(out_dir / 'results.json', merged)
    print(f'W1 wrote {out_dir / "RESULTS.md"} and results.json', flush=True)
    print(
        f'W1 uniform={head["uniform_bits_per_byte"]:.6f} '
        f'oracle={head["oracle_bits_per_byte"]:.6f} '
        f'headroom={head["headroom_bits_per_byte"]:.6f} '
        f'gpu_hours={gpu_hours:.4f}',
        flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sm = sub.add_parser('smoke')
    sm.add_argument('--bible', default='data/pg10.txt')
    sm.add_argument('--out', default='results/w1_headroom')
    st = sub.add_parser('streams')
    st.add_argument('--out', default='results/w1_headroom')
    ba = sub.add_parser('baselines')
    ba.add_argument('--out', default='results/w1_headroom')
    asm = sub.add_parser('assemble')
    asm.add_argument('--out', default='results/w1_headroom')
    args = parser.parse_args()
    if args.cmd == 'smoke':
        smoke(args.bible, args.out)
    elif args.cmd == 'streams':
        write_local_streams(args.out)
    elif args.cmd == 'baselines':
        write_baselines(args.out)
    elif args.cmd == 'assemble':
        assemble(args.out)
    else:
        raise SystemExit(f'unknown cmd {args.cmd}')


if __name__ == '__main__':
    main()
