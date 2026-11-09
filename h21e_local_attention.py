"""H21e local attention; principles 1 (history) and 2 (compressor).

Protocol: handoffs/2026-11-10_codex_h21e_local_attention.md. No verdicts.
Run --smoke before the full run. All outputs stay in results/h21e.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from h21a_substrate import CHUNK, N, ALPHA, PROCS, summarize, mean_std, now, stream
from h21b_cortex_store import mixture

NAME = 'att_rb8x2'
BASELINES = ['gru_rb8x2', 'gru_rb8x2+match', 'kt_mixture', 'lzma_9e']
NAMES = [NAME, NAME+'+match'] + BASELINES


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.qkv = nn.Linear(64, 192)
        self.proj = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.mlp = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))

    def forward(self, h, allowed):
        b, t, _ = h.shape
        q, k, v = self.qkv(self.norm1(h)).reshape(b, t, 3, 4, 16).permute(2, 0, 3, 1, 4).unbind(0)
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        h = h + self.proj(z.transpose(1, 2).reshape(b, t, 64))
        return h + self.mlp(self.norm2(h))


class LocalAttention(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(len(ALPHA), 64)
        self.position = nn.Embedding(CHUNK, 64)
        self.blocks = nn.ModuleList([Block() for _ in range(layers)])
        self.norm = nn.LayerNorm(64)
        self.readout = nn.Linear(64, len(ALPHA))
        i = torch.arange(2*CHUNK)
        # Shifted inputs: keys i-63 .. i represent the previous 64 symbols.
        self.register_buffer('local', (i[:, None] >= i[None, :]) &
                             (i[:, None]-i[None, :] < CHUNK))

    def forward(self, x, starts):
        indices = starts[:, None] + torch.arange(-CHUNK, CHUNK)[None, :]
        previous = indices-1
        h = self.embedding(x[previous.clamp_min(0)])
        h = h.masked_fill((previous < 0)[:, :, None], 0)
        # Absolute slots 0..63, reused for context and target chunks.
        h = h + self.position(torch.arange(2*CHUNK) % CHUNK)[None, :, :]
        valid = indices >= 0
        allowed = self.local[None, None, :, :] & valid[:, None, None, :]
        # Dummy negative queries cannot affect valid queries; avoid empty rows.
        allowed = allowed | ((~valid)[:, None, :, None] &
                             torch.eye(2*CHUNK, dtype=torch.bool)[None, None, :, :])
        for block in self.blocks:
            h = block(h, allowed)
        return self.readout(self.norm(h[:, CHUNK:]))


def check_causality():
    torch.manual_seed(123)
    model = LocalAttention()
    x = torch.randint(len(ALPHA), (256,))
    with torch.no_grad():
        before = model(x, torch.tensor([0, 64]))
        altered = x.clone()
        altered[96:] = (altered[96:]+1) % len(ALPHA)
        after = model(altered, torch.tensor([0, 64]))
    assert torch.equal(before[0], after[0])
    assert torch.equal(before[1, :33], after[1, :33])
    assert torch.isfinite(before).all()


def cortex(text, seed, layers, progress_every=128):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LocalAttention(layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.tensor([ALPHA.index(c) for c in text], dtype=torch.long)
    assert len(x) % CHUNK == 0
    probs = np.empty((len(x), len(ALPHA)), dtype=np.float64)
    bits = np.empty(len(x), dtype=np.float64)
    parameters = sum(p.numel() for p in model.parameters())
    print(f'{now()} {NAME} seed={seed} layers={layers} parameters={parameters}', flush=True)
    clock = time.monotonic()

    def step(loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for start in range(0, len(x), CHUNK):
        end = start+CHUNK
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, torch.tensor([start]))[0]
        losses = F.cross_entropy(logits, x[start:end], reduction='none')
        bits[start:end] = losses.detach().double().numpy()/math.log(2)
        probs[start:end] = logits.detach().double().softmax(-1).numpy()
        step(losses.mean())
        for _ in range(2):
            starts = torch.from_numpy(rng.integers(0, end-CHUNK+1, size=8))
            targets = starts[:, None] + torch.arange(CHUNK)[None, :]
            assert int(targets.max()) < end
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, starts)
            step(F.cross_entropy(logits.reshape(-1, len(ALPHA)), x[targets].reshape(-1)))
        if (start//CHUNK+1) % progress_every == 0 or end == len(x):
            print(f'{now()} {NAME} seed={seed} n={end}/{len(x)} bits={bits[:end].mean():.6f} seconds={time.monotonic()-clock:.2f}', flush=True)
    assert np.isfinite(bits).all()
    assert np.allclose(probs.sum(axis=1), 1)
    assert np.allclose(bits, -np.log2(probs[np.arange(len(x)), x.numpy()]), atol=2e-6)
    return probs, bits, dict(wall_seconds=time.monotonic()-clock, parameters=parameters,
                            layers=layers, live_steps=len(x)//CHUNK,
                            replay_steps=2*len(x)//CHUNK, replay_batch_size=8,
                            replay_chunks=16*len(x)//CHUNK)


def save(data, out):
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    rows = data['runs']
    lines = ['# H21e measurements', '', f"Started: {data['started']}. Updated: {now()}.", '',
             f"Slow world; N = {data['n']}; seeds = {data['seeds']}; CPU threads = 2; population std (ddof=0).", '',
             '| Learner / floor | Mean ± std bits/symbol | '+' | '.join(f'Seed {s}' for s in data['seeds'])+' |',
             '|---|---:|'+'---:|'*len(data['seeds'])]
    for name in NAMES:
        found = {r['seed']: r for r in rows if r['learner'] == name}
        lines.append('| '+name+' | '+mean_std([r['bits_per_symbol'] for r in found.values()])+' | '+
                     ' | '.join(f"{found[s]['bits_per_symbol']:.6f}" if s in found else 'NA' for s in data['seeds'])+' |')
    lines += ['', 'Per-process bits/symbol; arithmetic mean over seeds with observations.', '',
              '| Learner | '+' | '.join(PROCS)+' |', '|---|'+'---:|'*len(PROCS)]
    for name in NAMES:
        vals = [[r.get('by_process', {}).get(p) for r in rows if r['learner'] == name] for p in PROCS]
        lines.append('| '+name+' | '+' | '.join(f'{np.mean([v for v in vs if v is not None]):.6f}' if any(v is not None for v in vs) else 'NA' for vs in vals)+' |')
    lines += ['', '| Seed | Learner | '+' | '.join(PROCS)+' |', '|---:|---|'+'---:|'*len(PROCS)]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['learner']} | "+' | '.join('NA' if r.get('by_process', {}).get(p) is None else f"{r['by_process'][p]:.6f}" for p in PROCS)+' |')
    lines += ['', '| Seed | Learner | Wall seconds | Parameters |', '|---:|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['learner']} | {r['wall_seconds']:.3f} | {r.get('parameters', 'NA')} |")
    lines += ['', f"Attention layers: {data['layers']}; d=64; heads=4; MLP=256; pre-LayerNorm; final LayerNorm; dropout=0; readout=41.",
              'Learned absolute position slots 0–63 repeat across the context and target halves of the 128-position input.',
              'Inputs shifted by one symbol; zero symbol embedding before global position 1. Causal keys i−63 through i (64 preceding observed symbols); negative context positions masked. Loss only on the final 64 targets.',
              'Each chunk scored before its live Adam step; lr=0.001; clip=1.0; then two replay Adam steps, batch 8. Uniform starts 0 through scored_prefix_length−64 inclusive with replacement, each with preceding 64-symbol context.',
              'GRU, KT and lzma scores/timings copied from H21c; no reruns. Match probabilities/timings reused from H21b. Fixed-share mixture initial weights (0.5,0.5), alpha=0.001.',
              'Mixture wall seconds include cortex, reused match timing and mixture update. Parameters for mixture rows count the cortex only.',
              'Prequential observation bits; retained history unpriced as in H21b. lzma has no per-process attribution.',
              'Full probabilities and code lengths saved in learner-specific NPZ files.',
              f"Causality check: {data['causality_check']}. Probability normalization, finite bits, and bit/probability agreement checked for each new stream."]
    if 'smoke_projection_seconds' in data:
        lines += ['', f"Two-layer smoke wall seconds: {data['smoke_seconds']:.3f}. Projected three-seed cortex seconds: {data['smoke_projection_seconds']:.3f}."]
    lines += ['', *data['notes']]
    for key in ['total_wall_seconds', 'smoke_plus_full_wall_seconds']:
        if key in data:
            lines.append(f"{key}: {data[key]:.3f}.")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    clock = time.monotonic()
    check_causality()
    out = Path('results/h21e') / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    suffix = 'smoke' if args.smoke else ''
    source = Path('results/h21c')/suffix
    match_source = Path('results/h21b')/suffix
    old = json.loads((source/'results.json').read_text())
    match_old = json.loads((match_source/'results.json').read_text())
    n, seeds = (8192, [6]) if args.smoke else (N, [6, 7, 8])
    assert old['n'] == n
    data = dict(started=now(), n=n, seeds=seeds, layers=2, torch_version=torch.__version__,
                principles=['1: store everything', '2: improve the compressor'],
                causality_check='passed', runs=[], notes=[], baseline_source=str(source/'results.json'))
    for r in old['runs']:
        if r['world'] == 'slow' and r['seed'] in seeds and r['learner'] in BASELINES:
            data['runs'].append(dict(copy.deepcopy(r), reused_from=str(source/'results.json')))
    assert len(data['runs']) == len(seeds)*len(BASELINES)
    smoke_elapsed = 0
    if not args.smoke:
        smoke = json.loads((out/'smoke/results.json').read_text())
        seconds = next(r['wall_seconds'] for r in smoke['runs'] if r['learner'] == NAME)
        smoke_elapsed = smoke['total_wall_seconds']
        projection = seconds*16*3
        data.update(smoke_seconds=seconds, smoke_projection_seconds=projection)
        if projection+smoke_elapsed > 0.9*90*60:
            data['layers'] = 1
            data['notes'].append('One-layer variant used: two-layer projection plus smoke exceeds the 81-minute planning allowance within the 90-minute budget.')
    save(data, out)
    for seed in seeds:
        text, labels = stream('slow', seed, n)
        procs = np.array(labels)
        x = np.array([ALPHA.index(c) for c in text], dtype=np.uint8)
        with np.load(match_source/f'slow_{seed}.npz') as previous:
            assert np.array_equal(x, previous['symbols'])
            assert np.array_equal(procs, previous['processes'])
            mp = previous['match_probs'].copy()
        probs, bits, timing = cortex(text, seed, data['layers'])
        data['runs'].append(dict(world='slow', seed=seed, learner=NAME, n=n, **summarize(bits, procs), **timing))
        mbits, elapsed = mixture(probs, mp, x)
        match_time = next(r['wall_seconds'] for r in match_old['runs'] if r['world'] == 'slow' and r['seed'] == seed and r['learner'] == 'match')
        data['runs'].append(dict(world='slow', seed=seed, learner=NAME+'+match', n=n,
                                **summarize(mbits, procs), parameters=timing['parameters'],
                                wall_seconds=timing['wall_seconds']+match_time+elapsed,
                                mixture_only_wall_seconds=elapsed))
        np.savez_compressed(out/f'slow_{seed}_{NAME}.npz', symbols=x, processes=procs,
                            probs=probs, bits=bits, mixture_bits=mbits)
        save(data, out)
    data.update(finished=now(), total_wall_seconds=time.monotonic()-clock)
    data['smoke_plus_full_wall_seconds'] = smoke_elapsed+data['total_wall_seconds']
    save(data, out)


if __name__ == '__main__':
    main()
