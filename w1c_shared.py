"""W1c: one shared H29c cortex on W1b's 7 channels. Numbers only. No verdict.

The cortex, the chunking, the score-before-step order, the clip, and the
seeds are the W1b copy of the H29c setting: lr 3e-4, replay batch 8, seed 0.
Weights are shared. Each channel keeps its own carried hidden state.
Does not commit.
"""
import argparse
import csv
import hashlib
import io
import json
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

ALPHABET = 256
EMB = 64
HIDDEN = 1024
CHUNK = 128
REPLAY_BATCH = 8
CLIP = 1.0
SEED = 0
REPLAY_SEED = 30_000
PARAMS = 3_627_264
LOG2 = math.log(2)
LR = 3e-4
TRAIN_N = 2_000_000
HELD_N = 100_000
BUDGET = 3_500_000
CHECKPOINTS = (875_000, 1_750_000, 3_500_000)
STREAM_N = 2_100_000
NOISE_SEED = 777
DNA_BASES = frozenset('ACGTN')
LOG_EVERY = 100_000
SMOKE_BUDGET = 7 * 512
SMOKE_HELD = 1_024

CHANNELS = ('bible', 'chess', 'squad', 'btc', 'dna', 'noise', 'dark')
LEARNABLE = ('bible', 'chess', 'squad', 'btc', 'dna')
SYNTHETIC_CHANNELS = ('noise', 'dark')

# Fixed schedules. Each row sums to BUDGET.
ARMS = {
    'U7': {
        'bible': 500_000,
        'chess': 500_000,
        'squad': 500_000,
        'btc': 500_000,
        'dna': 500_000,
        'noise': 500_000,
        'dark': 500_000,
    },
    'N0': {
        'bible': 583_333,
        'chess': 583_333,
        'squad': 583_334,
        'btc': 583_333,
        'dna': 583_333,
        'noise': 0,
        'dark': 583_334,
    },
    'OR': {
        'bible': 1_000_000,
        'chess': 1_000_000,
        'squad': 1_000_000,
        'btc': 250_000,
        'dna': 125_000,
        'noise': 0,
        'dark': 125_000,
    },
}

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

# Train and held-out hashes from W1-0 (bible, chess, squad, btc) and W1b
# (dna, noise, dark). A mismatch means this is not the same stream.
EXPECTED = {
    'bible': {
        'stream_sha256': 'ef0b6942e0ac1174072d3e4420cd5e05aca97fe86af2088b986c5a6f493f77a6',
        'train_sha256': 'fa9eb395ac956821a48432d7337ee35e78d7611dbdc75240445e945700737642',
        'held_sha256': 'de2429965badbd30dc4eac76f4df2fc843e5ff96e3bbfafd05d1fee608a0dde5',
    },
    'chess': {
        'stream_sha256': '7791c438b0f21b8f8ec6b3b1f421eaa29d526647ed21eaddc2da9795c29d5b38',
        'train_sha256': '4986e78f72b9cc7fe1fa85229c56c7359df8afaa7aa426df2be0614cf723ef0b',
        'held_sha256': 'c0b520a6b4b43e22db26c1555e288bac280e5fc947ae987d1ecc42176ee23111',
    },
    'squad': {
        'stream_sha256': '1348449a968759680bb85a2803250a8b904d1c2a572d6ddb9ed757a2eafd1cb0',
        'train_sha256': '42003a1ff6ab605a2f0afe8080e233718cc0b08be61b3b155ec5f647e1c00a0a',
        'held_sha256': 'fb97f239566374dbea842207371fa2477938b9cbedf6d2f0283b7878586d1c32',
    },
    'btc': {
        'stream_sha256': '095ee45ed97240a12e64cb2e038e04eef537da520851b1be4f05e1eee91395b9',
        'train_sha256': '3ba760b1a58801976c68e834954a16f737e4cfcac1132b78fc667a80efbb713a',
        'held_sha256': '4ba57af0b7dc6b1bb237bd9ddf6d0249496b19e10a9cc803c29e2281e8263a51',
    },
    'dna': {
        'stream_sha256': 'e42a27d03e6b6248fa4c34bcb0cd2c68fd96e28600971edd0d9fb585fbdbbba7',
        'train_sha256': '449c7a12825c53e0a5343c4eb05467245681ac8feb94687f8c19f18ad71bf229',
        'held_sha256': '381022715867364fda2ced39782996096abb085a26bcb4f38b1e06228ec8b0c3',
    },
    'noise': {
        'stream_sha256': 'ee85587d25b9ad4044d603b2a87a46fc07c3341bfd61bcba287623e6a6be4e2e',
        'train_sha256': '1e1bf139cca67a5858d2662b1f0639fefbf988c4f18bb0ff1ffdf22033c8c3f9',
        'held_sha256': 'c40fe11aec22ef630945c1ce6a17f8f7c8814bdc06c5509258dd803e95102b08',
    },
    'dark': {
        'stream_sha256': 'c2f2ff9f44898670c8a5020b4987f78845bcfce6f6678594c8559a64c65346ed',
        'train_sha256': '13aea96040f2133033d103008d5d96cfe98b3361f7202d77bea97b2424a7a6cd',
        'held_sha256': '9192c25b734fcbadbe32dadc28089c60db0e39f90cc20ce2e5733f57261acc0c',
    },
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
    if n < TRAIN_N + HELD_N:
        raise SystemExit(f'stream length {n} is shorter than {TRAIN_N + HELD_N}')
    train = stream[:TRAIN_N]
    held = stream[TRAIN_N:TRAIN_N + HELD_N]
    return train, held


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


def load_raw(channel):
    if channel in SYNTHETIC_CHANNELS:
        return b'', 'synthetic'
    kaggle_root = Path('/kaggle/input')
    if kaggle_root.is_dir():
        path = find_kaggle_source(channel)
    else:
        path = LOCAL_SOURCES[channel]
        if not path.is_file():
            raise SystemExit(f'missing {path}')
    print(f'W1c source {channel} {path} bytes={path.stat().st_size}', flush=True)
    return path.read_bytes(), str(path)


def load_streams():
    trains = {}
    helds = {}
    meta = {}
    for channel in CHANNELS:
        raw, origin = load_raw(channel)
        stream = BUILDERS[channel](raw)
        train_b, held_b = split_stream(stream)
        stream_sha = sha256_bytes(stream)
        train_sha = sha256_bytes(train_b)
        held_sha = sha256_bytes(held_b)
        exp = EXPECTED[channel]
        if stream_sha != exp['stream_sha256'] or train_sha != exp['train_sha256'] or held_sha != exp['held_sha256']:
            raise SystemExit(
                f'{channel}: hash mismatch stream={stream_sha} train={train_sha} held={held_sha}')
        trains[channel] = as_u8(train_b)
        helds[channel] = as_u8(held_b)
        meta[channel] = {
            'origin': origin,
            'stream_bytes': len(stream),
            'stream_sha256': stream_sha,
            'train_bytes': len(train_b),
            'train_sha256': train_sha,
            'held_bytes': len(held_b),
            'held_sha256': held_sha,
            'split': 'prefix_2000000_100000',
        }
        print(
            f'W1c stream {channel} bytes={len(stream)} train={train_sha} held={held_sha}',
            flush=True)
    return trains, helds, meta


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


def pick_channel(read, quotas):
    """Smallest bytes/quota. Ties go to the lowest channel index. Quota 0 is skipped."""
    best = None
    best_i = None
    for i, channel in enumerate(CHANNELS):
        quota = quotas[channel]
        if quota <= 0 or read[channel] >= quota:
            continue
        if best is None:
            best, best_i = channel, i
            continue
        left = read[channel] * quotas[best]
        right = read[best] * quota
        if left < right or (left == right and i < best_i):
            best, best_i = channel, i
    return best


def plan_reads(quotas, budget, checkpoints):
    """Read order. Each step is (channel, position, nbytes, total_before).

    A step is min(128, remaining quota, bytes until the next checkpoint).
    Quotas and the 875000-spaced checkpoints are not all multiples of 128,
    so the step that lands on one is short. The sums still match exactly.
    """
    quotas = {c: int(quotas[c]) for c in CHANNELS}
    budget = int(budget)
    if sum(quotas.values()) != budget:
        raise AssertionError(f'quotas sum {sum(quotas.values())} != budget {budget}')
    for n in checkpoints:
        if not 0 < int(n) <= budget:
            raise AssertionError(f'checkpoint {n} outside 1..{budget}')
    ck = tuple(sorted({int(n) for n in checkpoints}))
    read = {c: 0 for c in CHANNELS}
    total = 0
    steps = []
    hit = []
    next_i = 0
    while total < budget:
        channel = pick_channel(read, quotas)
        if channel is None:
            raise AssertionError(f'no channel left at total {total}')
        room = quotas[channel] - read[channel]
        until = budget - total
        if next_i < len(ck):
            until = min(until, ck[next_i] - total)
        n = min(CHUNK, room, until)
        if n <= 0:
            raise AssertionError(
                f'empty read at {total} channel {channel} room {room} until {until}')
        steps.append((channel, read[channel], n, total))
        read[channel] += n
        total += n
        if next_i < len(ck) and total == ck[next_i]:
            hit.append(total)
            next_i += 1
        elif next_i < len(ck) and total > ck[next_i]:
            raise AssertionError(f'passed checkpoint {ck[next_i]} at {total}')
    if total != budget:
        raise AssertionError(f'total {total} != budget {budget}')
    for channel in CHANNELS:
        if read[channel] != quotas[channel]:
            raise AssertionError(
                f'{channel} read {read[channel]} != quota {quotas[channel]}')
    if hit != list(ck):
        raise AssertionError(f'checkpoints hit {hit} expected {list(ck)}')
    return steps


def scale_quotas(quotas, budget):
    """Largest-remainder scale. A zero quota stays zero."""
    raw = [int(quotas[c]) for c in CHANNELS]
    total = sum(raw)
    if total <= 0:
        raise AssertionError('quota total is 0')
    budget = int(budget)
    floors = [budget * q // total for q in raw]
    out = floors[:]
    rem = budget - sum(floors)
    order = sorted(
        range(len(raw)),
        key=lambda i: ((budget * raw[i]) % total, -i),
        reverse=True)
    for k in range(rem):
        out[order[k]] += 1
    scaled = {c: out[i] for i, c in enumerate(CHANNELS)}
    for channel, quota in zip(CHANNELS, raw):
        if quota == 0 and scaled[channel] != 0:
            raise AssertionError(f'scaled a zero quota on {channel}')
    if sum(scaled.values()) != budget:
        raise AssertionError('scaled quotas do not sum to the budget')
    return scaled


def prev_and_target(raw, start, end):
    n = end - start
    tgt = raw[start:end].astype(np.int64, copy=False)
    if start == 0:
        prev = np.empty(n, dtype=np.int64)
        prev[0] = -1
        if n > 1:
            prev[1:] = raw[:n - 1]
    else:
        prev = raw[start - 1:end - 1].astype(np.int64, copy=False)
    return prev, tgt


def draw_channel(read, eligible, rng):
    total_w = 0
    for channel in eligible:
        total_w += int(read[channel])
    if total_w <= 0:
        raise AssertionError('replay weights sum to 0')
    draw = int(rng.integers(0, total_w))
    acc = 0
    for channel in eligible:
        acc += int(read[channel])
        if draw < acc:
            return channel
    raise AssertionError('channel draw fell through')


def sample_replay(trains, read, rng):
    """Eight windows. Channel ~ bytes read. Start uniform inside the read prefix."""
    for channel in CHANNELS:
        if 0 < int(read[channel]) < CHUNK:
            raise AssertionError(
                f'{channel} has {read[channel]} bytes, short of a 128-byte window')
    eligible = [c for c in CHANNELS if int(read[c]) >= CHUNK]
    if not eligible:
        raise AssertionError('replay before any channel has 128 bytes')
    prev = np.empty((CHUNK, REPLAY_BATCH), dtype=np.int64)
    targets = np.empty((CHUNK, REPLAY_BATCH), dtype=np.int64)
    for b in range(REPLAY_BATCH):
        channel = draw_channel(read, eligible, rng)
        seen = int(read[channel])
        high = seen - CHUNK + 1
        start = int(rng.integers(0, high))
        if start < 0 or start + CHUNK > seen:
            raise AssertionError(
                f'replay window outside read bytes channel={channel} '
                f'start={start} read={seen}')
        raw = trains[channel]
        p, t = prev_and_target(raw, start, start + CHUNK)
        prev[:, b] = p
        targets[:, b] = t
    return prev, targets


class SharedRun:
    def __init__(self, device, phase):
        self.device = device
        self.phase = phase
        self.t0 = time.monotonic()
        torch.manual_seed(SEED)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(SEED)
        self.model = ByteGRU().to(device)
        self.model.train()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.rng = np.random.default_rng(REPLAY_SEED + SEED)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if self.n_params != PARAMS:
            raise AssertionError(f'parameter count {self.n_params} != {PARAMS}')
        self.states = {c: None for c in CHANNELS}
        self.bits = {c: 0.0 for c in CHANNELS}
        self.read = {c: 0 for c in CHANNELS}
        self.replay_steps = 0
        self.chunk_count = 0

    def _step(self, losses):
        self.opt.zero_grad(set_to_none=True)
        losses.mean().backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        self.opt.step()

    def live(self, channel, raw, start, n):
        end = start + n
        if end > len(raw):
            raise AssertionError(f'{channel}: read {end} past train {len(raw)}')
        prev_np, tgt_np = prev_and_target(raw, start, end)
        prev_t = torch.tensor(prev_np, dtype=torch.long, device=self.device).unsqueeze(1)
        tgt_t = torch.tensor(tgt_np, dtype=torch.long, device=self.device).unsqueeze(1)
        # Score this chunk under the weights and state from before the update.
        losses, state = self.model.nll(prev_t, tgt_t, self.states[channel])
        bit_vals = (losses.detach().double() / LOG2).cpu().numpy()
        if not np.isfinite(bit_vals).all():
            raise FloatingPointError(f'{self.phase}: nonfinite bits on {channel} at {start}')
        self.bits[channel] += float(bit_vals.sum())
        self.states[channel] = state.detach()
        self._step(losses)
        del losses

    def replay(self, trains):
        prev, tgt = sample_replay(trains, self.read, self.rng)
        prev_t = torch.tensor(prev, dtype=torch.long, device=self.device)
        tgt_t = torch.tensor(tgt, dtype=torch.long, device=self.device)
        losses, _ = self.model.nll(prev_t, tgt_t, None)
        self._step(losses)
        del losses
        self.replay_steps += 1


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def eval_heldout(model, helds, device, n_done):
    _sync(device)
    t0 = time.monotonic()
    channels = {}
    for channel in CHANNELS:
        bits = score_block(model, helds[channel], device)
        nbytes = int(len(helds[channel]))
        row = {
            'bits': float(bits),
            'bits_per_byte': float(bits / nbytes),
            'held_bytes': nbytes,
            'finite': bool(math.isfinite(bits)),
        }
        if not row['finite']:
            raise FloatingPointError(f'nonfinite held-out bits on {channel} at n={n_done}')
        channels[channel] = row
        print(
            f'W1c heldout n={n_done} channel={channel} '
            f'L={row["bits_per_byte"]:.6f} bits={bits:.3f}',
            flush=True)
    m7 = sum(channels[c]['bits_per_byte'] for c in CHANNELS) / len(CHANNELS)
    learnable5 = sum(channels[c]['bits_per_byte'] for c in LEARNABLE) / len(LEARNABLE)
    if not math.isfinite(m7) or not math.isfinite(learnable5):
        raise FloatingPointError(f'nonfinite mean at n={n_done}')
    print(
        f'W1c heldout n={n_done} M7={m7:.6f} learnable5={learnable5:.6f} '
        f'eval_s={time.monotonic() - t0:.1f}',
        flush=True)
    return {
        'n': int(n_done),
        'channels': channels,
        'M7': float(m7),
        'learnable5': float(learnable5),
        'eval_seconds': time.monotonic() - t0,
    }


def run_arm(phase, trains, helds, quotas, budget, checkpoints, device):
    quotas = {c: int(quotas[c]) for c in CHANNELS}
    budget = int(budget)
    for channel in CHANNELS:
        if quotas[channel] > len(trains[channel]):
            raise AssertionError(f'{channel}: quota exceeds train')
        if len(helds[channel]) <= 0:
            raise AssertionError(f'{channel}: empty held-out')
    steps = plan_reads(quotas, budget, checkpoints)
    ck = {int(n) for n in checkpoints}
    run = SharedRun(device, phase)
    held_rows = []
    total = 0
    log_mark = 0
    print(
        f'W1c phase={phase} start budget={budget} chunks={len(steps)} '
        f'device={device} params={run.n_params}',
        flush=True)
    for channel, pos, n, total_before in steps:
        if total != total_before or pos != run.read[channel]:
            raise AssertionError(f'{phase}: plan and reader diverged at {total}')
        run.live(channel, trains[channel], pos, n)
        run.read[channel] += n
        total += n
        run.chunk_count += 1
        # Replay sees the bytes of the chunk just read.
        run.replay(trains)
        if total in ck:
            held_rows.append(eval_heldout(run.model, helds, device, total))
        mark = total // LOG_EVERY
        if mark > log_mark or total == budget:
            log_mark = mark
            _sync(device)
            live_bits = sum(run.bits.values())
            print(
                f'W1c phase={phase} n={total}/{budget} '
                f'live_bpb={live_bits / total:.6f} replay_steps={run.replay_steps} '
                f'wall_s={time.monotonic() - run.t0:.1f}',
                flush=True)
    if run.replay_steps != run.chunk_count:
        raise AssertionError(
            f'{phase}: replay steps {run.replay_steps} != chunks {run.chunk_count}')
    if [row['n'] for row in held_rows] != sorted(ck):
        raise AssertionError(f'{phase}: held-out rows missed a checkpoint')
    for channel in CHANNELS:
        if run.read[channel] != quotas[channel]:
            raise AssertionError(
                f'{phase}: {channel} read {run.read[channel]} != quota {quotas[channel]}')
    total_bits = float(sum(run.bits.values()))
    gap = abs(total_bits - sum(run.bits.values()))
    if gap > 1e-6:
        raise AssertionError(f'{phase}: bit sum drifted by {gap}')
    if total != budget or not math.isfinite(total_bits):
        raise AssertionError(f'{phase}: total {total} bits {total_bits}')
    _sync(device)
    preq = {}
    for channel in CHANNELS:
        nbytes = int(run.read[channel])
        bits = float(run.bits[channel])
        preq[channel] = {
            'bytes': nbytes,
            'bits': bits,
            'bits_per_byte': None if nbytes == 0 else float(bits / nbytes),
        }
    traffic = run.replay_steps * REPLAY_BATCH * CHUNK / budget
    return {
        'phase': phase,
        'lr': LR,
        'replay_batch': REPLAY_BATCH,
        'seed': SEED,
        'replay_seed': REPLAY_SEED + SEED,
        'parameters': int(run.n_params),
        'device': str(device),
        'budget': budget,
        'quotas': quotas,
        'bytes_read': {c: int(run.read[c]) for c in CHANNELS},
        'prequential': {
            'channels': preq,
            'total_bits': total_bits,
            'total_bytes': int(total),
            'bits_per_byte': float(total_bits / total),
            'replay_steps': int(run.replay_steps),
            'replay_windows': int(run.replay_steps * REPLAY_BATCH),
            'chunk_count': int(run.chunk_count),
            'traffic': float(traffic),
            'replay_outside_read': 0,
        },
        'heldout': held_rows,
        'finite': True,
        'run_wall_seconds': time.monotonic() - run.t0,
    }


def _held_cut(helds, n):
    out = {}
    for channel in CHANNELS:
        if len(helds[channel]) < n:
            raise AssertionError(f'{channel}: held-out shorter than {n}')
        out[channel] = helds[channel][:n]
    return out


def smoke(out_dir):
    configure()
    t0 = time.monotonic()
    trains, helds, meta = load_streams()
    plans = {}
    for arm, quotas in ARMS.items():
        steps = plan_reads(quotas, BUDGET, CHECKPOINTS)
        read = {c: 0 for c in CHANNELS}
        for channel, _pos, n, _total in steps:
            read[channel] += n
        if read != {c: int(quotas[c]) for c in CHANNELS}:
            raise AssertionError(f'{arm}: full plan bytes != quotas')
        plans[arm] = {
            'chunks': len(steps),
            'bytes_read': read,
            'checkpoints': list(CHECKPOINTS),
        }
        print(
            f'W1c plan arm={arm} chunks={len(steps)} '
            + ' '.join(f'{c}={read[c]}' for c in CHANNELS),
            flush=True)
    u7 = scale_quotas(ARMS['U7'], SMOKE_BUDGET)
    if u7 != {c: 512 for c in CHANNELS}:
        raise AssertionError(f'U7 smoke quotas {u7}')
    n0 = scale_quotas(ARMS['N0'], SMOKE_BUDGET)
    if n0['noise'] != 0:
        raise AssertionError(f'N0 smoke gave noise {n0["noise"]} bytes')
    positive = [n0[c] for c in CHANNELS if c != 'noise']
    if sum(n0.values()) != SMOKE_BUDGET or max(positive) - min(positive) > 1:
        raise AssertionError(f'N0 smoke quotas {n0}')
    held = _held_cut(helds, SMOKE_HELD)
    device = torch.device('cpu')
    u7_run = run_arm('U7-smoke', trains, held, u7, SMOKE_BUDGET, (SMOKE_BUDGET,), device)
    if u7_run['bytes_read'] != {c: 512 for c in CHANNELS}:
        raise AssertionError(f'U7 smoke read {u7_run["bytes_read"]}')
    if not u7_run['finite'] or not math.isfinite(u7_run['prequential']['total_bits']):
        raise AssertionError('U7 smoke bits are not finite')
    if abs(u7_run['prequential']['traffic'] - 8.0) > 1e-9:
        raise AssertionError(f'U7 smoke traffic {u7_run["prequential"]["traffic"]}')
    if u7_run['prequential']['replay_outside_read'] != 0:
        raise AssertionError('U7 smoke replay left the read bytes')
    n0_run = run_arm('N0-smoke', trains, held, n0, SMOKE_BUDGET, (SMOKE_BUDGET,), device)
    if n0_run['bytes_read']['noise'] != 0:
        raise AssertionError(f'N0 smoke noise read {n0_run["bytes_read"]["noise"]}')
    if sum(n0_run['bytes_read'].values()) != SMOKE_BUDGET:
        raise AssertionError('N0 smoke did not read the budget')
    if not n0_run['finite'] or not math.isfinite(n0_run['prequential']['total_bits']):
        raise AssertionError('N0 smoke bits are not finite')
    if n0_run['prequential']['replay_outside_read'] != 0:
        raise AssertionError('N0 smoke replay left the read bytes')
    payload = {
        'budget': SMOKE_BUDGET,
        'held_bytes': SMOKE_HELD,
        'device': 'cpu',
        'plans': plans,
        'u7_quotas': u7,
        'n0_quotas': n0,
        'u7': {
            'bytes_read': u7_run['bytes_read'],
            'finite': u7_run['finite'],
            'total_bits': u7_run['prequential']['total_bits'],
            'replay_steps': u7_run['prequential']['replay_steps'],
            'chunk_count': u7_run['prequential']['chunk_count'],
            'traffic': u7_run['prequential']['traffic'],
            'replay_outside_read': 0,
            'M7': u7_run['heldout'][-1]['M7'],
            'learnable5': u7_run['heldout'][-1]['learnable5'],
        },
        'n0': {
            'bytes_read': n0_run['bytes_read'],
            'noise_bytes': n0_run['bytes_read']['noise'],
            'finite': n0_run['finite'],
            'total_bits': n0_run['prequential']['total_bits'],
            'replay_steps': n0_run['prequential']['replay_steps'],
            'chunk_count': n0_run['prequential']['chunk_count'],
            'traffic': n0_run['prequential']['traffic'],
            'replay_outside_read': 0,
            'M7': n0_run['heldout'][-1]['M7'],
            'learnable5': n0_run['heldout'][-1]['learnable5'],
        },
        'streams': meta,
        'wall_seconds': time.monotonic() - t0,
        'checks': {
            'u7_each_512': True,
            'u7_finite': True,
            'u7_replay_in_range': True,
            'u7_traffic_8': True,
            'n0_noise_0': True,
            'n0_finite': True,
            'n0_replay_in_range': True,
        },
    }
    out = Path(out_dir)
    dump(out / 'smoke.json', payload)
    print(
        f'W1c SMOKE_OK u7_bits={payload["u7"]["total_bits"]:.3f} '
        f'n0_noise={payload["n0"]["noise_bytes"]} '
        f'n0_bits={payload["n0"]["total_bits"]:.3f} '
        f'wall_s={payload["wall_seconds"]:.1f}',
        flush=True)
    return payload


def kernel_main(out, arm, kernel_id):
    configure()
    t0 = time.monotonic()
    out = Path(out)
    if arm not in ARMS:
        raise SystemExit(f'unknown arm {arm}')
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the scored run')
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f'W1c kernel={kernel_id} arm={arm} gpu={gpu_name} torch={torch.__version__} '
        f'numpy={np.__version__}',
        flush=True)
    trains, helds, meta = load_streams()
    result = run_arm(arm, trains, helds, ARMS[arm], BUDGET, CHECKPOINTS, device)
    result['arm'] = arm
    result['kernel_id'] = kernel_id
    result['gpu_name'] = gpu_name
    result['torch_version'] = torch.__version__
    result['numpy_version'] = np.__version__
    result['streams'] = meta
    result['session_wall_seconds'] = time.monotonic() - t0
    dump(out / 'result.json', result)
    summary = {
        'kernel_id': kernel_id,
        'arm': arm,
        'gpu_name': gpu_name,
        'session_wall_seconds': result['session_wall_seconds'],
        'bytes_read': result['bytes_read'],
        'M7': {row['n']: row['M7'] for row in result['heldout']},
        'learnable5': {row['n']: row['learnable5'] for row in result['heldout']},
        'finite': result['finite'],
        'replay_steps': result['prequential']['replay_steps'],
        'traffic': result['prequential']['traffic'],
    }
    dump(out / 'kernel.json', summary)
    print('KERNEL_DONE ' + json.dumps(summary), flush=True)
    return result


def _fmt(value, digits=6):
    if value is None:
        return ''
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


def _kernel_row(arm, payload, log_path, status, version):
    session = float(payload['session_wall_seconds'])
    log_elapsed = None if log_path is None else _log_elapsed(log_path)
    if log_elapsed is None:
        elapsed = session
        source = 'script_wall'
    else:
        elapsed = float(log_elapsed)
        source = 'log_clock'
    return {
        'arm': arm,
        'kernel_id': payload['kernel_id'],
        'version': int(version),
        'status': _status_name(status),
        'elapsed_seconds': elapsed,
        'elapsed_source': source,
        'session_wall_seconds': session,
        'log_elapsed_seconds': log_elapsed,
        'gpu_name': payload.get('gpu_name'),
        'gpu_hours': elapsed / 3600.0,
    }


def _check_arm(payload, arm):
    if payload.get('arm') != arm:
        raise SystemExit(f'{arm}: result arm is {payload.get("arm")}')
    if int(payload['budget']) != BUDGET:
        raise SystemExit(f'{arm}: budget {payload["budget"]}')
    read = payload['bytes_read']
    quotas = payload['quotas']
    for channel in CHANNELS:
        if int(read[channel]) != int(quotas[channel]) or int(quotas[channel]) != int(ARMS[arm][channel]):
            raise SystemExit(f'{arm}: {channel} bytes {read[channel]} quota {quotas[channel]}')
        exp = EXPECTED[channel]['train_sha256']
        got = payload['streams'][channel]['train_sha256']
        if got != exp:
            raise SystemExit(f'{arm}: {channel} train hash {got}')
    if not payload['finite']:
        raise SystemExit(f'{arm}: not finite')
    rows = {int(row['n']): row for row in payload['heldout']}
    if set(rows) != set(CHECKPOINTS):
        raise SystemExit(f'{arm}: checkpoints {sorted(rows)}')
    return rows


def headroom_of(arms):
    m7 = {}
    learnable = {}
    for arm, payload in arms.items():
        final = [row for row in payload['heldout'] if int(row['n']) == BUDGET][0]
        m7[arm] = float(final['M7'])
        learnable[arm] = float(final['learnable5'])
    best_value = min(m7['N0'], m7['OR'])
    best = [arm for arm in ('N0', 'OR') if m7[arm] == best_value]
    return {
        'at_bytes': BUDGET,
        'M7': m7,
        'learnable5': learnable,
        'headroom_shared': float(m7['U7'] - best_value),
        'minimum_arms': best,
        'minimum_M7': float(best_value),
        'noisy_tv_cost': float(m7['U7'] - m7['N0']),
    }


def render_results(doc):
    lines = [
        '# W1c shared cortex',
        '',
        'One cortex, seven channels. Weights are shared. Each channel has its own hidden state.',
        '',
        '## Streams',
        '',
        '| Channel | Stream bytes | Stream sha256 | Train bytes | Train sha256 | Held-out bytes | Held-out sha256 |',
        '|---|---:|---|---:|---|---:|---|',
    ]
    streams = doc['streams']
    for channel in CHANNELS:
        row = streams[channel]
        lines.append(
            f"| {channel} | {row['stream_bytes']} | `{row['stream_sha256']}` | "
            f"{row['train_bytes']} | `{row['train_sha256']}` | {row['held_bytes']} | "
            f"`{row['held_sha256']}` |")
    lines += ['', '## Quotas', '', '| Arm | ' + ' | '.join(CHANNELS) + ' | Sum |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('U7', 'N0', 'OR'):
        quotas = doc['arms'][arm]['quotas']
        cells = ' | '.join(str(int(quotas[c])) for c in CHANNELS)
        lines.append(f'| {arm} | {cells} | {sum(int(quotas[c]) for c in CHANNELS)} |')
    for arm in ('U7', 'N0', 'OR'):
        payload = doc['arms'][arm]
        preq = payload['prequential']
        lines += [
            '',
            f'## {arm}',
            '',
            '### Bytes read',
            '',
            '| Channel | Quota | Bytes read |',
            '|---|---:|---:|',
        ]
        for channel in CHANNELS:
            lines.append(
                f"| {channel} | {int(payload['quotas'][channel])} | "
                f"{int(payload['bytes_read'][channel])} |")
        lines += [
            '',
            '### Prequential',
            '',
            f"Total bits {preq['total_bits']:.6f}. Bits/byte {_fmt(preq['bits_per_byte'])}. "
            f"Replay steps {preq['replay_steps']}. Traffic {_fmt(preq['traffic'])}. "
            f"Chunks {preq['chunk_count']}. Run seconds {payload['run_wall_seconds']:.3f}.",
            '',
            '| Channel | Bytes | Bits | Bits/byte |',
            '|---|---:|---:|---:|',
        ]
        for channel in CHANNELS:
            row = preq['channels'][channel]
            lines.append(
                f"| {channel} | {row['bytes']} | {row['bits']:.6f} | {_fmt(row['bits_per_byte'])} |")
        lines += [
            '',
            '### Held-out L_c, bits/byte',
            '',
            '| n | ' + ' | '.join(CHANNELS) + ' | M7 | learnable5 |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for row in payload['heldout']:
            cells = ' | '.join(_fmt(row['channels'][c]['bits_per_byte']) for c in CHANNELS)
            lines.append(
                f"| {int(row['n'])} | {cells} | {_fmt(row['M7'])} | {_fmt(row['learnable5'])} |")
        lines += [
            '',
            'Held-out bits:',
            '',
            '| n | ' + ' | '.join(CHANNELS) + ' |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for row in payload['heldout']:
            cells = ' | '.join(f"{row['channels'][c]['bits']:.3f}" for c in CHANNELS)
            lines.append(f"| {int(row['n'])} | {cells} |")
    head = doc['headroom']
    lines += [
        '',
        '## Headroom',
        '',
        f"At {head['at_bytes']} bytes.",
        '',
        '| Arm | M7 | learnable5 |',
        '|---|---:|---:|',
    ]
    for arm in ('U7', 'N0', 'OR'):
        lines.append(f"| {arm} | {_fmt(head['M7'][arm])} | {_fmt(head['learnable5'][arm])} |")
    mins = ', '.join(head['minimum_arms'])
    lines += [
        '',
        f"Minimum of M7(N0) and M7(OR): {_fmt(head['minimum_M7'])} ({mins}).",
        f"headroom_shared = M7(U7) − min(M7(N0), M7(OR)) = {_fmt(head['headroom_shared'])}.",
        f"M7(U7) − M7(N0) = {_fmt(head['noisy_tv_cost'])}.",
        '',
        '## Kernels',
        '',
        'Elapsed seconds is the Kaggle session log clock (timestamp of the last log line) when that',
        'clock is present, and the script wall otherwise. GPU hours = elapsed seconds / 3600.',
        'Script wall is time.monotonic from kernel_main entry through the result write. Queue time',
        'is not included.',
        '',
        '| Kernel | Arm | Version | Status | Elapsed seconds | Elapsed source | Script wall seconds | GPU hours | GPU |',
        '|---|---|---:|---|---:|---|---:|---:|---|',
    ]
    for row in doc['kernels']:
        lines.append(
            f"| {row['kernel_id']} | {row['arm']} | {row['version']} | {row['status']} | "
            f"{row['elapsed_seconds']:.3f} | {row['elapsed_source']} | "
            f"{row['session_wall_seconds']:.3f} | {row['gpu_hours']:.4f} | {row['gpu_name']} |")
    lines += [
        '',
        f"Total GPU hours: {doc['gpu_hours_total']:.4f}.",
        '',
        '## Smoke',
        '',
        'CPU. U7 budget 3584, quota 512 per channel, held-out cut to 1024. '
        'N0 is the N0 quota row scaled to the same budget by largest remainder.',
        '',
        f"U7 bytes read: " + ', '.join(
            f"{c}={doc['smoke']['u7']['bytes_read'][c]}" for c in CHANNELS) + '.',
        f"U7 finite: {doc['smoke']['u7']['finite']}. "
        f"U7 bits: {doc['smoke']['u7']['total_bits']:.6f}. "
        f"U7 traffic: {doc['smoke']['u7']['traffic']:.6f}. "
        f"U7 replay outside read: {doc['smoke']['u7']['replay_outside_read']}.",
        f"N0 noise bytes: {doc['smoke']['n0']['noise_bytes']}. "
        f"N0 finite: {doc['smoke']['n0']['finite']}. "
        f"N0 bits: {doc['smoke']['n0']['total_bits']:.6f}. "
        f"N0 replay outside read: {doc['smoke']['n0']['replay_outside_read']}.",
        f"Smoke wall seconds: {doc['smoke']['wall_seconds']:.3f}.",
        '',
        '## Protocol',
        '',
        '- One H29c cortex per arm. ByteGRU embedding 64, hidden 1024, 1 layer, alphabet 256. '
        f"Adam lr {LR:g}, clip {CLIP:g}. {PARAMS} parameters. Seed {SEED}. Replay RNG {REPLAY_SEED + SEED}.",
        '- Chunks of 128. A chunk is min(128, remaining quota, bytes until the next checkpoint). '
        'Score the chunk (live −log2 p) before its Adam step. Detach the channel hidden state '
        'at the chunk boundary. Weights are shared. Hidden state is per channel, zero at the '
        "channel's first byte.",
        '- Before each chunk, read the channel with the smallest (bytes read) / quota among '
        'channels with quota > 0 and bytes still remaining. Ties go to the lowest index in '
        'bible, chess, squad, btc, dna, noise, dark.',
        '- After each live chunk, one Adam step on a batch of 8 windows of 128 bytes, zero '
        'initial state. The channel is drawn with probability proportional to bytes read so '
        'far. The start is uniform on 0 .. bytes_read−128 inclusive. A window that would '
        'leave the read prefix raises.',
        '- Held-out score at 875000, 1750000 and 3500000 total bytes. No gradient, zero '
        'initial state, chunks of 128, state carried. L_c is bits / held-out bytes. '
        'M7 is the mean of the 7 L_c values. learnable5 is the mean of bible, chess, squad, '
        'btc and dna.',
        '- headroom_shared = M7(U7) − min(M7(N0), M7(OR)) at 3500000 bytes. '
        'M7(U7) − M7(N0) is the noisy-TV comparison.',
        '- Training bytes are the first 2000000 of each W1b stream. Held-out bytes are the next '
        '100000. Noise is numpy.random.default_rng(777).integers(0, 256, size=2100000, dtype=uint8). '
        'Dark is 2100000 zero bytes. DNA is the sequence column of human.txt.',
        '- Quotas: U7 is 500000 on every channel. N0 is 0 on noise and the remaining 3500000 '
        'split across the other six (583333, 583333, 583334, 583333, 583333, 583334). '
        'OR is 1000000, 1000000, 1000000, 250000, 125000, 0, 125000.',
        '',
    ]
    return '\n'.join(lines)


def assemble(out_dir, arm_paths, log_paths, statuses, versions, smoke_path):
    arms = {}
    kernels = []
    for arm in ('U7', 'N0', 'OR'):
        payload = json.loads(Path(arm_paths[arm]).read_text())
        _check_arm(payload, arm)
        arms[arm] = payload
        kernels.append(_kernel_row(
            arm, payload, log_paths.get(arm), statuses[arm], versions[arm]))
    streams = arms['U7']['streams']
    for arm in ('N0', 'OR'):
        for channel in CHANNELS:
            if arms[arm]['streams'][channel]['train_sha256'] != streams[channel]['train_sha256']:
                raise SystemExit(f'{arm}: {channel} train hash differs from U7')
    smoke_payload = json.loads(Path(smoke_path).read_text())
    if not smoke_payload['checks']['u7_each_512'] or not smoke_payload['checks']['n0_noise_0']:
        raise SystemExit('smoke checks are not set')
    doc = {
        'arms': arms,
        'streams': streams,
        'headroom': headroom_of(arms),
        'kernels': kernels,
        'gpu_hours_total': float(sum(row['gpu_hours'] for row in kernels)),
        'smoke': smoke_payload,
    }
    text = render_results(doc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'RESULTS.md').write_text(text)
    # The arm payloads are large enough; store them once inside doc.
    dump(out_dir / 'results.json', doc)
    print(f'W1c wrote {out_dir / "RESULTS.md"} and results.json', flush=True)
    print(
        f"W1c headroom_shared={doc['headroom']['headroom_shared']:.6f} "
        f"min={','.join(doc['headroom']['minimum_arms'])} "
        f"noisy_tv={doc['headroom']['noisy_tv_cost']:.6f} "
        f"gpu_hours={doc['gpu_hours_total']:.4f}",
        flush=True)
    return doc


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    sm = sub.add_parser('smoke')
    sm.add_argument('--out', default='results/w1c_shared')
    asm = sub.add_parser('assemble')
    asm.add_argument('--out', default='results/w1c_shared')
    asm.add_argument('--smoke', default='results/w1c_shared/smoke.json')
    for arm, flag in (('U7', 'u7'), ('N0', 'n0'), ('OR', 'or')):
        asm.add_argument(f'--{flag}', required=True)
        asm.add_argument(f'--{flag}-log', default=None)
        asm.add_argument(f'--{flag}-status', required=True)
        asm.add_argument(f'--{flag}-version', type=int, required=True)
    args = parser.parse_args()
    if args.cmd == 'smoke':
        smoke(args.out)
    elif args.cmd == 'assemble':
        assemble(
            args.out,
            {'U7': args.u7, 'N0': args.n0, 'OR': args.__dict__['or']},
            {'U7': args.u7_log, 'N0': args.n0_log, 'OR': args.or_log},
            {'U7': args.u7_status, 'N0': args.n0_status, 'OR': args.or_status},
            {'U7': args.u7_version, 'N0': args.n0_version, 'OR': args.or_version},
            args.smoke)
    else:
        raise SystemExit(f'unknown cmd {args.cmd}')


if __name__ == '__main__':
    main()
