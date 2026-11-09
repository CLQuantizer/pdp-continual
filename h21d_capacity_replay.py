"""H21d capacity/replay measurements; principles 1 (history) and 2 (compressor).

Protocol: handoffs/2026-11-10_codex_h21d_capacity_replay.md. No verdicts.
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
from h21a_substrate import GRU, CHUNK, N, ALPHA, PROCS, detach, summarize, mean_std, now, stream
from h21b_cortex_store import mixture

# hidden size, replay Adam steps per live chunk, replay batch size
VARIANTS = {'gru_rb8x4': (256, 4, 8), 'gru_rb16x2': (256, 2, 16),
            'gru512_rb8x2': (512, 2, 8)}
BASELINES = ['gru_rb8x2', 'gru_rb8x2+match', 'kt_mixture', 'lzma_9e']
NAMES = BASELINES[:2] + [v for name in VARIANTS for v in (name, name+'+match')] + BASELINES[2:]
BUDGET_SECONDS = 90*60


def cortex(text, seed, name, progress_every=128):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hidden, steps, batch = VARIANTS[name]
    model = GRU(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.tensor([ALPHA.index(c) for c in text], dtype=torch.long)
    probs = np.empty((len(x), len(ALPHA)), dtype=np.float64)
    bits = np.empty(len(x), dtype=np.float64)
    state = None
    replay_steps = 0
    started = time.monotonic()

    def forward(start, end, hidden):
        if start == 0:
            emb = torch.cat((model.embedding.weight.new_zeros(1, 32), model.embedding(x[:end-1])))
        else:
            emb = model.embedding(x[start-1:end-1])
        return model(emb, hidden)

    def step(loss):
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    for start in range(0, len(x), CHUNK):
        end = min(start+CHUNK, len(x))
        optimizer.zero_grad(set_to_none=True)
        logits, state = forward(start, end, state)
        losses = F.cross_entropy(logits, x[start:end], reduction='none')
        bits[start:end] = losses.detach().double().numpy()/math.log(2)
        probs[start:end] = logits.detach().double().softmax(-1).numpy()
        state = detach(state)
        step(losses.mean())
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            if batch == 1:
                rstart = int(rng.integers(0, end-CHUNK+1))
                rlogits, _ = forward(rstart, rstart+CHUNK, None)
                loss = F.cross_entropy(rlogits, x[rstart:rstart+CHUNK])
            else:
                # Independent uniform draws with replacement, just as H21b.
                starts = torch.from_numpy(rng.integers(0, end-CHUNK+1, size=batch))
                indices = torch.arange(CHUNK)[:, None] + starts[None, :]
                assert int(indices.max()) < end
                emb = model.embedding(x[(indices-1).clamp_min(0)])
                # Every sequence starts with zero hidden state; only global t=0
                # also has a zero input embedding. Other starts use x[t-1].
                emb = emb.masked_fill((indices == 0)[:, :, None], 0)
                outputs, _ = model.gru(emb, None)
                rlogits = model.readout(outputs)
                loss = F.cross_entropy(rlogits.reshape(-1, len(ALPHA)), x[indices].reshape(-1))
            step(loss)
            replay_steps += 1
        if (start//CHUNK+1) % progress_every == 0 or end == len(x):
            print(f'{now()} {name} seed={seed} n={end}/{len(x)} bits={bits[:end].mean():.6f} seconds={time.monotonic()-started:.2f}', flush=True)
    assert np.isfinite(bits).all()
    assert np.allclose(probs.sum(axis=1), 1)
    assert np.allclose(bits, -np.log2(probs[np.arange(len(x)), x.numpy()]), atol=2e-6)
    return probs, bits, dict(wall_seconds=time.monotonic()-started,
        parameters=sum(p.numel() for p in model.parameters()), hidden=hidden, replay_steps=replay_steps,
        replay_batch_size=batch, replay_chunks=replay_steps*batch)


def save(data, out):
    (out/'results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    rows = data['runs']
    lines = ['# H21d measurements', '', f"Started: {data['started']}. Updated: {now()}.", '',
        f"Slow world; N = {data['n']}; seeds = {data['seeds']}; CPU threads = 2; population std (ddof=0).", '',
        '| Learner / floor | Mean ± std bits/symbol | ' + ' | '.join(f'Seed {s}' for s in data['seeds']) + ' |',
        '|---|---:|' + '---:|'*len(data['seeds'])]
    for name in NAMES:
        rs = [r for r in rows if r['learner'] == name]
        found = {r['seed']: r for r in rs}
        lines.append('| '+name+' | '+mean_std([r['bits_per_symbol'] for r in rs])+' | '+' | '.join(f"{found[s]['bits_per_symbol']:.6f}" if s in found else 'NA' for s in data['seeds'])+' |')
    lines += ['', 'Per-process bits/symbol; arithmetic mean over seeds with observations.', '',
        '| Learner | '+' | '.join(PROCS)+' |', '|---|'+'---:|'*len(PROCS)]
    for name in NAMES:
        vals = [[r.get('by_process', {}).get(p) for r in rows if r['learner'] == name] for p in PROCS]
        lines.append('| '+name+' | '+' | '.join(f'{np.mean([v for v in vs if v is not None]):.6f}' if any(v is not None for v in vs) else 'NA' for vs in vals)+' |')
    lines += ['', '| Seed | Learner | '+' | '.join(PROCS)+' |', '|---:|---|'+'---:|'*len(PROCS)]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['learner']} | "+' | '.join('NA' if r.get('by_process', {}).get(p) is None else f"{r['by_process'][p]:.6f}" for p in PROCS)+' |')
    lines += ['', '| Seed | Learner | Wall seconds |', '|---:|---|---:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['learner']} | {r['wall_seconds']:.3f} |")
    lines += ['', 'Reference scores and timings copied from H21c; no reference reruns. Match probabilities and timings reused from H21b.',
        'Embedding 32, chunk 64, Adam lr 0.003, clip 1.0; H21b shifted inputs and detached carried live state.',
        'gru_rb8x4: hidden 256, four replay steps, batch 8. gru_rb16x2: hidden 256, two replay steps, batch 16. gru512_rb8x2: hidden 512, two replay steps, batch 8.',
        'Replay follows each scored chunk and its live Adam step. Uniform starts 0 through scored_prefix_length−64 inclusive, with replacement; zero replay hidden state; preceding-symbol input (zero embedding at global position 0). Live state is preserved during replay. Batched loss is the mean over all batch × 64 targets.',
        'Mixtures reuse H21b initial weights (0.5, 0.5) and fixed share α = 0.001; wall seconds include cortex, reused match timing, and mixture update.',
        'Scores are prequential observation bits; retained history is unpriced, as specified in H21b. lzma has no per-process attribution.',
        'Full predictive distributions and code lengths are saved in learner-specific NPZ files.', '']
    if 'smoke_projection' in data:
        lines += ['Smoke projection (8192 symbols, seed 6; ×16 symbols ×3 seeds).', '',
                  '| Learner | Smoke wall seconds | Projected full wall seconds |', '|---|---:|---:|']
        for name, seconds in data['smoke_projection'].items():
            lines.append(f'| {name} | {seconds/48:.3f} | {seconds:.3f} |')
        lines += ['', f"Projected retained cortex wall seconds: {data['retained_projected_seconds']:.3f}.", '']
    lines += ['Runtime note: '+note for note in data['notes']]
    if 'total_wall_seconds' in data:
        lines += ['', f"Total full-run wall seconds: {data['total_wall_seconds']:.3f}.",
                  f"Smoke plus full-run wall seconds: {data['total_wall_seconds']+data.get('smoke_wall_seconds', 0):.3f}."]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    out = Path('results/h21d') / ('smoke' if args.smoke else '')
    out.mkdir(parents=True, exist_ok=True)
    source = Path('results/h21c') / ('smoke' if args.smoke else '')
    match_source = Path('results/h21b') / ('smoke' if args.smoke else '')
    old = json.loads((source/'results.json').read_text())
    match_data = json.loads((match_source/'results.json').read_text())
    n, seeds = (8192, [6]) if args.smoke else (N, [6, 7, 8])
    assert old['n'] == match_data['n'] == n
    clock = time.monotonic()
    data = dict(started=now(), n=n, seeds=seeds, principles=['1: store everything', '2: improve the compressor'],
        torch_version=torch.__version__, notes=[], runs=[], baseline_source=str(source/'results.json'),
        match_source=str(match_source), budget_seconds=BUDGET_SECONDS)
    for r in old['runs']:
        if r['world'] == 'slow' and r['seed'] in seeds and r['learner'] in BASELINES:
            row = copy.deepcopy(r)
            row['reused_from'] = str(source/'results.json')
            data['runs'].append(row)
    assert len(data['runs']) == len(seeds)*len(BASELINES)
    variants = list(VARIANTS)
    if not args.smoke:
        smoke = json.loads((out/'smoke/results.json').read_text())
        data['smoke_wall_seconds'] = smoke['total_wall_seconds']
        projection = {r['learner']: r['wall_seconds']*16*len(seeds)
                      for r in smoke['runs'] if r['learner'] in VARIANTS}
        assert set(projection) == set(VARIANTS)
        data['smoke_projection'] = projection
        # Reserve 10% for mixtures, serialization, and runtime variability.
        allowance = 0.9*(BUDGET_SECONDS-data['smoke_wall_seconds'])
        for drop in ['gru_rb8x4', 'gru_rb16x2']:
            if sum(projection[v] for v in variants) > allowance:
                variants.remove(drop)
                data['notes'].append(f'{drop} and {drop}+match omitted: smoke projection exceeds {allowance:.3f}-second planning allowance within 90-minute budget; prescribed drop order gru_rb8x4, then gru_rb16x2.')
        data['retained_projected_seconds'] = sum(projection[v] for v in variants)
    data['retained_variants'] = variants
    save(data, out)
    # Preserve the wider cortex first, then the remaining replay variants.
    order = [v for v in ['gru512_rb8x2', 'gru_rb16x2', 'gru_rb8x4'] if v in variants]
    for name in order:
        for seed in seeds:
            text, labels = stream('slow', seed, n)
            procs = np.array(labels)
            x = np.array([ALPHA.index(c) for c in text], dtype=np.uint8)
            with np.load(match_source/f'slow_{seed}.npz') as previous:
                assert np.array_equal(x, previous['symbols'])
                assert np.array_equal(procs, previous['processes'])
                mp = previous['match_probs'].copy()
            gp, bits, timing = cortex(text, seed, name)
            data['runs'].append(dict(world='slow', seed=seed, learner=name, n=n, **summarize(bits, procs), **timing))
            mbits, elapsed = mixture(gp, mp, x)
            match_time = next(r['wall_seconds'] for r in match_data['runs'] if r['world'] == 'slow' and r['seed'] == seed and r['learner'] == 'match')
            data['runs'].append(dict(world='slow', seed=seed, learner=name+'+match', n=n, **summarize(mbits, procs),
                wall_seconds=timing['wall_seconds']+match_time+elapsed, mixture_only_wall_seconds=elapsed))
            np.savez_compressed(out/f'slow_{seed}_{name}.npz', symbols=x, processes=procs, probs=gp, bits=bits, mixture_bits=mbits)
            save(data, out)
    data['finished'] = now()
    data['total_wall_seconds'] = time.monotonic()-clock
    save(data, out)


if __name__ == '__main__':
    main()

