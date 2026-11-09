"""H21a: fixed, prequential substrate comparison. Principle 2: improve the compressor.

No parameter selection or retuning. Run --smoke before the full run.
Fast weights: e_t = Emb(x_{t-1}), z_t = [e_t; y_{t-1}];
k_t = softmax(K z_t), q_t = softmax(Q z_t), v_t = tanh(V z_t),
beta_t = sigmoid(b z_t), W_t = W_{t-1} + beta_t
(v_t - W_{t-1} k_t) k_t^T; y_t = W_t q_t;
p(x_t | x_<t) = softmax(R y_t). W_0 = 0, y_0 = 0.
The first input embedding is zero (no artificial observed start symbol).
Both fast-weight state and previous output are detached at chunk boundaries.
"""
import argparse
from datetime import datetime, timezone, timedelta
import json
import lzma
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from h20_world import stream, ALPHA, PROCS, WORLDS, kt_floors

CHUNK = 64
N = 131072
TZ = timezone(timedelta(hours=8))


def now():
    return datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S UTC+8')


class GRU(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embedding = nn.Embedding(len(ALPHA), 32)
        self.gru = nn.GRU(32, hidden)
        self.readout = nn.Linear(hidden, len(ALPHA))

    def forward(self, x, state):
        y, state = self.gru(x[:, None, :], state)
        return self.readout(y[:, 0]), state


class LinearFastWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(ALPHA), 32)
        self.projection = nn.Linear(32 + 64, 3 * 64 + 1)
        self.readout = nn.Linear(64, len(ALPHA))

    def forward(self, x, state):
        if state is None:
            w, y = x.new_zeros(64, 64), x.new_zeros(64)
        else:
            w, y = state
        outputs = []
        for e in x:
            k, q, v, beta = torch.split(self.projection(torch.cat((e, y))), [64, 64, 64, 1])
            k, q = k.softmax(-1), q.softmax(-1)
            w = w + beta.sigmoid() * torch.outer(v.tanh() - w @ k, k)
            y = w @ q
            outputs.append(y)
        return self.readout(torch.stack(outputs)), (w, y)


def detach(state):
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()


def summarize(bits, procs):
    return dict(bits_per_symbol=float(bits.mean()),
                by_process={p: float(bits[procs == p].mean()) if np.any(procs == p) else None for p in PROCS},
                process_counts={p: int(np.sum(procs == p)) for p in PROCS},
                eighths=[float(b.mean()) for b in np.array_split(bits, 8)])


def train(name, text, procs, seed, mix, progress_every=256):
    torch.manual_seed(seed)
    model = LinearFastWeights() if name == 'lin_fw' else GRU(512 if name == 'gru_big' else 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.tensor([ALPHA.index(c) for c in text], dtype=torch.long)
    bits = np.empty(len(x), dtype=np.float64)
    state = None
    started = now()
    clock = time.monotonic()
    diagnostic = None
    # First complete chunk at or beyond 20%; compare the same prefix with KT.
    diagnostic_end = math.ceil(0.2 * len(x) / CHUNK) * CHUNK
    for start in range(0, len(x), CHUNK):
        end = min(start + CHUNK, len(x))
        optimizer.zero_grad(set_to_none=True)
        if start == 0:
            emb = torch.cat((model.embedding.weight.new_zeros(1, 32), model.embedding(x[:end-1])), dim=0)
        else:
            emb = model.embedding(x[start-1:end-1])
        logits, state = model(emb, state)
        losses = F.cross_entropy(logits, x[start:end], reduction='none')
        # Capture code lengths under unchanged weights BEFORE this chunk's Adam step.
        bits[start:end] = losses.detach().double().numpy() / math.log(2)
        if not np.all(np.isfinite(bits[start:end])):
            raise FloatingPointError(f'{name}: nonfinite bits at {start}')
        state = detach(state)
        losses.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if end == diagnostic_end:
            diagnostic = dict(n=end, bits_per_symbol=float(bits[:end].mean()),
                              kt_mixture=float(mix[:end].mean()),
                              above_kt=bool(bits[:end].mean() > mix[:end].mean()))
            print(f'{now()} {name} seed={seed} 20% diagnostic {diagnostic}', flush=True)
        if (start // CHUNK + 1) % progress_every == 0 or end == len(x):
            print(f'{now()} {name} seed={seed} n={end}/{len(x)} bits={bits[:end].mean():.6f} seconds={time.monotonic()-clock:.2f}', flush=True)
    elapsed = time.monotonic() - clock
    return dict(learner=name, seed=seed, n=len(x), started=started, finished=now(),
                wall_seconds=elapsed, parameters=sum(p.numel() for p in model.parameters()),
                diagnostic_20_percent=diagnostic, **summarize(bits, procs))


def mean_std(values):
    values = [v for v in values if v is not None]
    return 'NA' if not values else f'{np.mean(values):.6f} ± {np.std(values, ddof=0):.6f}'


def report(data, out):
    rows = data['runs']
    lines = ['# H21a measurements', '', f"Started: {data['started']}. Updated: {now()}.", '',
             f"N = {data['n']}; seeds = {data['seeds']}; CPU threads = 2; population std (ddof=0).", '',
             '| Learner / floor | slow bits/symbol | fast bits/symbol |', '|---|---:|---:|']
    names = ['gru', 'gru_big', 'lin_fw', 'kt_mixture', 'lzma_9e']
    for name in names:
        cells = [mean_std([r['bits_per_symbol'] for r in rows if r['world'] == w and r['learner'] == name]) for w in WORLDS]
        lines.append(f"| {name} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Per-process bits/symbol, slow world; arithmetic mean over seeds with observations (counts below).', '',
              '| Learner | ' + ' | '.join(PROCS) + ' |', '|---|' + '---:|' * len(PROCS)]
    for name in names[:-1]:
        rs = [r for r in rows if r['world'] == 'slow' and r['learner'] == name]
        vals = [[r['by_process'][p] for r in rs if r['by_process'][p] is not None] for p in PROCS]
        lines.append(f'| {name} | ' + ' | '.join(f'{np.mean(v):.6f}' if v else 'NA' for v in vals) + ' |')
    lines += ['', 'Slow-world learning curve, bits/symbol (mean over seeds).', '',
              '| Learner | ' + ' | '.join(str(i) for i in range(1, 9)) + ' |', '|---|' + '---:|' * 8]
    for name in names[:-2]:
        rs = [r for r in rows if r['world'] == 'slow' and r['learner'] == name]
        lines.append(f'| {name} | ' + ' | '.join(f'{np.mean([r["eighths"][i] for r in rs]):.6f}' if rs else 'NA' for i in range(8)) + ' |')
    lines += ['', '| World | Seed | Learner / floor | Wall seconds |', '|---|---:|---|---:|']
    lines += [f"| {r['world']} | {r['seed']} | {r['learner']} | {r['wall_seconds']:.3f} |" for r in rows]
    lines += ['', '| World | Seed | ' + ' | '.join(PROCS) + ' |', '|---|---:|' + '---:|' * len(PROCS)]
    for r in rows:
        if r['learner'] == 'kt_mixture':
            lines.append(f"| {r['world']} | {r['seed']} | " + ' | '.join(str(r['process_counts'][p]) for p in PROCS) + ' |')
    lines += ['', 'Fast-weight 20% diagnostics (same-prefix KT comparison; no tuning or early termination).', '',
              '| World | Seed | N scored | lin_fw bits | KT bits | Above KT |', '|---|---:|---:|---:|---:|---|']
    for r in rows:
        if r['learner'] == 'lin_fw':
            d = r['diagnostic_20_percent']
            lines.append(f"| {r['world']} | {r['seed']} | {d['n']} | {d['bits_per_symbol']:.6f} | {d['kt_mixture']:.6f} | {d['above_kt']} |")
    lines += ['', 'Protocol: embedding 32; GRU hidden 256 / 512; readout 41; chunks 64; Adam lr 0.003; clip 1.0.',
              'Initial hidden state zero; first input embedding zero; input is shifted by one symbol; state carried and detached after every chunk. No process IDs are passed to learners.',
              'lin_fw: z_t=[Emb(x_(t-1)); y_(t-1)], k_t=softmax(Kz_t), q_t=softmax(Qz_t), v_t=tanh(Vz_t), beta_t=sigmoid(bz_t), W_t=W_(t-1)+beta_t(v_t-W_(t-1)k_t)k_t^T, y_t=W_t q_t, logits_t=R y_t. Dimension 64; W_0=0, y_0=0; projections/readout affine. Both W and y detached between chunks.',
              'First 20% diagnostic uses the first complete 64-symbol chunk at or beyond 20%. All scores are recorded before the chunk optimizer step.',
              'lzma uses preset 9 | PRESET_EXTREME including container overhead; KT uses h20_world.kt_floors on identical streams.',
              'Scope: compressor-only measurement; no auxiliary exemplar store; no two-part T or recurrent-state charge added to the specified prequential score.', '']
    lines += [f'Runtime note: {note}' for note in data['notes']]
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def save(data, out):
    (out / 'results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    report(data, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default='results/h21a')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n, seeds = (8192, [6]) if args.smoke else (N, [6, 7, 8])
    data = dict(started=now(), n=n, seeds=seeds, torch_version=torch.__version__,
                principle='2: improve the compressor', notes=[], runs=[])
    clock = time.monotonic()
    names = ['gru', 'gru_big', 'lin_fw']
    if not args.smoke:
        smoke_path = out / 'smoke' / 'results.json'
        smoke = json.loads(smoke_path.read_text())
        estimates = {name: 16 * sum(r['wall_seconds'] for r in smoke['runs'] if r['learner'] == name) for name in names}
        data['smoke_projected_seconds_per_stream'] = estimates
        # Reserve 10% for floors/overhead; mandatory order of budget reductions.
        if estimates['gru'] > 25 * 60 or 6 * sum(estimates.values()) > 0.9 * 90 * 60:
            names.remove('gru_big')
            data['notes'].append(f'gru_big omitted: smoke projection for six streams with all three learners = {6 * sum(estimates.values()) / 60:.2f} minutes; planning allowance = 81 minutes within the 90-minute total budget. Both worlds retained.')
    worlds = ['slow'] if args.smoke else list(WORLDS)
    if not args.smoke and 6 * sum(estimates[k] for k in names) > 0.9 * 90 * 60:
        worlds = ['slow']
        data['notes'].append('Fast world omitted from full run based on smoke runtime projection after dropping gru_big.')
    for world in worlds:
        for seed in seeds:
            text, labels = stream(world, seed, n)
            procs = np.array(labels)
            start = time.monotonic()
            _, mix, _ = kt_floors(text)
            data['runs'].append(dict(world=world, seed=seed, learner='kt_mixture', n=n,
                                     wall_seconds=time.monotonic()-start, **summarize(mix, procs)))
            start = time.monotonic()
            compressed = lzma.compress(text.encode(), preset=9 | lzma.PRESET_EXTREME)
            data['runs'].append(dict(world=world, seed=seed, learner='lzma_9e', n=n,
                                     compressed_bytes=len(compressed), bits_per_symbol=8*len(compressed)/n,
                                     wall_seconds=time.monotonic()-start))
            save(data, out)
            for name in names.copy():
                r = train(name, text, procs, seed, mix)
                data['runs'].append(dict(world=world, **r))
                save(data, out)
                if name == 'gru' and r['wall_seconds'] > 25*60 and 'gru_big' in names:
                    names.remove('gru_big')
                    data['notes'].append('gru_big omitted after GRU exceeded 25 minutes per stream.')
            print(f'{now()} completed {world} seed={seed}; elapsed={time.monotonic()-clock:.1f}s', flush=True)
    data['finished'] = now()
    data['total_wall_seconds'] = time.monotonic()-clock
    save(data, out)


if __name__ == '__main__':
    main()
